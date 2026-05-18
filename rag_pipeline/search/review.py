from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

from ..config.search_config import (
    DEFAULT_LLM_CONTEXT_DEDUP_THRESHOLD,
    DEFAULT_LLM_CONTEXT_MAX_TOKENS,
    DEFAULT_LLM_CONTEXT_MAX_TOKENS_PER_EVIDENCE,
)
from .memory import call_openai_compatible_json, llm_config_is_ready
from .models import AnswerSynthesis, EvidenceItem, QueryPlan
from .synthesis import build_llm_context_payload, llm_response_is_grounded, validate_answer_citations


def review_answer_with_fallback(
    *,
    query: str,
    plan: QueryPlan,
    answer: AnswerSynthesis,
    evidence_items: Sequence[EvidenceItem],
    llm_config: Dict[str, Any] | None,
    core_top_k: int,
    support_top_k: int,
    max_context_tokens: int = DEFAULT_LLM_CONTEXT_MAX_TOKENS,
    max_tokens_per_evidence: int = DEFAULT_LLM_CONTEXT_MAX_TOKENS_PER_EVIDENCE,
    context_dedup_threshold: float = DEFAULT_LLM_CONTEXT_DEDUP_THRESHOLD,
) -> Tuple[AnswerSynthesis, Dict[str, Any]]:
    if answer.status != "answered" or not str(answer.answer or "").strip():
        answer.review_status = "skipped"
        return answer, {
            "type": "answer_review",
            "source": "skipped",
            "reason": "answer_not_reviewable",
        }

    heuristic_issues = []
    has_inline_reference = "[" in answer.answer or "证据：" in answer.answer
    if answer.grounding_mode == "extractive" and not has_inline_reference:
        heuristic_issues.append("answer_has_no_inline_citations")
    if not evidence_items:
        heuristic_issues.append("no_evidence_items")

    review_meta: Dict[str, Any] = {
        "type": "answer_review",
        "source": "heuristic",
        "approved": not heuristic_issues,
        "issues": heuristic_issues,
    }

    if not llm_config_is_ready(llm_config):
        answer.review_status = "approved" if not heuristic_issues else "warning"
        answer.review_issues = heuristic_issues
        if heuristic_issues:
            answer.gaps = list(dict.fromkeys(answer.gaps + heuristic_issues))
        return answer, review_meta

    evidence_payload, evidence_index, context_stats = build_llm_context_payload(
        query=query,
        plan=plan,
        evidence_items=evidence_items,
        core_top_k=core_top_k,
        support_top_k=support_top_k,
        max_context_tokens=max_context_tokens,
        max_tokens_per_evidence=max_tokens_per_evidence,
        dedup_threshold=context_dedup_threshold,
    )
    system_prompt = (
        "你负责审查行研 RAG 智能体生成的证据约束回答。"
        "只返回 JSON，包含这些键：approved、issues、should_refuse、revised_answer、confidence_adjustment、note。"
        "只能使用给定证据。请检查回答是否先给结论、是否受证据约束、是否符合行业研究表达。"
        "如果市场规模、增速、份额、盈利能力、政策、最新进展、估值或风险判断缺少证据，请降低评价。"
        "如果回答过度推断、遗漏关键约束，或把宣传性语言当成事实，请用 [E#] 引用修订，或建议拒答。"
        "修订后的每个事实性句子都必须保留行内证据引用。"
    )
    user_payload = {
        "query": query,
        "task_type": plan.task_type,
        "answer": answer.to_dict(),
        "evidence": evidence_payload,
    }
    try:
        response = call_openai_compatible_json(
            config=llm_config or {},
            system_prompt=system_prompt,
            user_payload=user_payload,
        )
        payload = response.get("payload", {})
        issues = [str(item).strip() for item in payload.get("issues", []) if str(item).strip()]
        revised_answer = str(payload.get("revised_answer") or "").strip()
        approved = bool(payload.get("approved", False))
        should_refuse = bool(payload.get("should_refuse", False))

        if revised_answer:
            revised_citations = validate_answer_citations(revised_answer, evidence_index)
            if llm_response_is_grounded(revised_answer, revised_citations):
                answer.answer = revised_answer
                answer.citations = revised_citations
                answer.grounding_mode = f"{answer.grounding_mode}+reviewed"
            else:
                issues.append("reviewer_revision_failed_citation_validation")

        if should_refuse:
            answer.status = "insufficient_evidence"
            answer.refusal_reason = "answer_review_rejected_grounding"
            answer.grounding_mode = f"{answer.grounding_mode}+reviewed_refusal"

        if "confidence_adjustment" in payload:
            try:
                adjustment = float(payload.get("confidence_adjustment") or 0.0)
                answer.confidence = round(max(0.0, min(1.0, float(answer.confidence) + adjustment)), 4)
            except (TypeError, ValueError):
                pass

        answer.review_status = "approved" if approved and not issues else "warning"
        answer.review_issues = list(dict.fromkeys(answer.review_issues + issues))
        if issues:
            answer.gaps = list(dict.fromkeys(answer.gaps + issues))

        review_meta = {
            "type": "answer_review",
            "source": "llm",
            "approved": approved,
            "issues": issues,
            "request": user_payload,
            "response": payload,
            "usage": response.get("usage", {}),
            "context_stats": context_stats,
        }
        return answer, review_meta
    except Exception as exc:
        heuristic_issues.append(str(exc))
        answer.review_status = "warning"
        answer.review_issues = list(dict.fromkeys(answer.review_issues + heuristic_issues))
        answer.gaps = list(dict.fromkeys(answer.gaps + heuristic_issues))
        return answer, {
            "type": "answer_review",
            "source": "fallback_heuristic",
            "approved": False,
            "issues": heuristic_issues,
            "error": str(exc),
        }
