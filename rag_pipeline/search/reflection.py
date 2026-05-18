from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

from .memory import call_openai_compatible_json, llm_config_is_ready


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def unique_preserve(items: Sequence[str], max_items: int | None = None) -> List[str]:
    values: List[str] = []
    seen = set()
    for item in items:
        cleaned = str(item or "").strip()
        if not cleaned:
            continue
        key = normalize_text(cleaned)
        if not key or key in seen:
            continue
        seen.add(key)
        values.append(cleaned)
        if max_items and len(values) >= max_items:
            break
    return values


def evidence_chunk_uids(evidence_items: Sequence[Dict[str, Any]]) -> List[str]:
    return [str(item.get("chunk_uid") or "").strip() for item in evidence_items if str(item.get("chunk_uid") or "").strip()]


def evidence_overlap_ratio(left_uids: Sequence[str], right_uids: Sequence[str]) -> float:
    left = {str(item).strip() for item in left_uids if str(item).strip()}
    right = {str(item).strip() for item in right_uids if str(item).strip()}
    if not left or not right:
        return 0.0
    return len(left & right) / max(min(len(left), len(right)), 1)


def build_missing_aspects(
    *,
    plan: Dict[str, Any],
    answer: Dict[str, Any],
    coverage: Dict[str, Any],
    topic_consistency: Dict[str, Any],
) -> List[str]:
    missing: List[str] = []
    for item in coverage.get("missing", []) or []:
        missing.append(str(item))
    refusal_reason = str(answer.get("refusal_reason") or "").strip()
    if refusal_reason:
        missing.append(refusal_reason.replace("_", " "))
    if not topic_consistency.get("passed", True):
        missing.append("topic consistency")
    task_type = str(plan.get("task_type") or "").strip()
    if task_type == "comparison":
        missing.append("comparison dimensions")
    if task_type == "market":
        missing.append("market scale, competitive landscape, profitability, or risk evidence")
    if task_type == "trend":
        missing.append("trend drivers, commercialization pace, risk, or time range")
    return unique_preserve(missing, max_items=6)


def heuristic_reflection_query(
    *,
    original_query: str,
    plan: Dict[str, Any],
    evidence_items: Sequence[Dict[str, Any]],
    missing_aspects: Sequence[str],
    previous_queries: Sequence[str],
) -> str:
    fragments: List[str] = []
    for item in plan.get("entity_terms", []) or []:
        cleaned = str(item or "").strip()
        if cleaned:
            fragments.append(cleaned)
    for item in missing_aspects:
        cleaned = str(item or "").strip()
        if cleaned:
            fragments.append(cleaned)
    if evidence_items:
        best = evidence_items[0]
        for field in ["doc_title", "section_title", "group"]:
            cleaned = str(best.get(field) or "").strip()
            if cleaned:
                fragments.append(cleaned)
    query = " ".join(unique_preserve([original_query] + fragments, max_items=8)).strip()
    if not query:
        query = str(original_query or "").strip()
    normalized_previous = {normalize_text(item) for item in previous_queries if normalize_text(item)}
    if normalize_text(query) in normalized_previous:
        for aspect in missing_aspects:
            candidate = f"{original_query} {aspect}".strip()
            if normalize_text(candidate) not in normalized_previous:
                return candidate
    return query


def reflect_on_evidence(
    *,
    original_query: str,
    plan: Dict[str, Any],
    evidence_items: Sequence[Dict[str, Any]],
    answer: Dict[str, Any],
    coverage: Dict[str, Any],
    topic_consistency: Dict[str, Any],
    previous_queries: Sequence[str],
    hop_index: int,
    max_hops: int,
    llm_config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    def _strong_proof_required(payload: Dict[str, Any]) -> bool:
        containers = []
        for key in ["hypotheses", "search_tasks", "evidence_goals"]:
            value = payload.get(key)
            if isinstance(value, list):
                containers.extend(item for item in value if isinstance(item, dict))
        return any(
            str(item.get("proof_standard") or "").strip().lower() == "strong"
            or bool(item.get("counter_evidence_required", False))
            for item in containers
        )

    def _has_counter_signal(items: Sequence[Dict[str, Any]]) -> bool:
        for item in items:
            if bool(item.get("counter_evidence") or item.get("is_counter_evidence")):
                return True
            marker_text = " ".join(
                str(item.get(key) or "")
                for key in [
                    "proof_role",
                    "evidence_type",
                    "intent",
                    "label",
                    "tags",
                    "section_title",
                    "content",
                    "text",
                    "quote",
                ]
            ).lower()
            if "counter" in marker_text or "反证" in marker_text or "相反证据" in marker_text:
                return True
        return False

    missing_aspects = build_missing_aspects(
        plan=plan,
        answer=answer,
        coverage=coverage,
        topic_consistency=topic_consistency,
    )
    proof_gap = _strong_proof_required(plan) and not _has_counter_signal(evidence_items)
    if proof_gap:
        missing_aspects = unique_preserve([*missing_aspects, "strong_proof_requires_counter_evidence"], max_items=8)
    sufficient = bool(
        answer.get("status") == "answered"
        and coverage.get("passed", True)
        and topic_consistency.get("passed", True)
        and not proof_gap
    )
    if hop_index >= max_hops:
        return {
            "sufficient": sufficient,
            "missing_aspects": missing_aspects,
            "rewritten_query": "",
            "reason": "max_hops_reached",
            "source": "guardrail",
            "llm_call": {},
        }

    if llm_config_is_ready(llm_config):
        system_prompt = (
            "你是行研 RAG 智能体内部的反思节点。"
            "只返回 JSON，包含这些键：sufficient、missing_aspects、rewritten_query、reason。"
            "如果当前证据已经充分，请让 rewritten_query 为空。"
            "如果证据不足，请生成一条简洁的补充检索查询，重点补齐市场、增速、竞争格局、价值链、盈利能力、政策、风险或时间口径证据。"
        )
        user_payload = {
            "original_query": original_query,
            "plan": plan,
            "answer": answer,
            "coverage": coverage,
            "topic_consistency": topic_consistency,
            "previous_queries": list(previous_queries),
            "evidence": list(evidence_items)[:8],
        }
        try:
            response = call_openai_compatible_json(
                config=llm_config or {},
                system_prompt=system_prompt,
                user_payload=user_payload,
            )
            payload = response.get("payload", {})
            rewritten_query = str(payload.get("rewritten_query") or "").strip()
            llm_missing = unique_preserve(
                [str(item).strip() for item in payload.get("missing_aspects", []) if str(item).strip()] + missing_aspects,
                max_items=8,
            )
            llm_sufficient = bool(payload.get("sufficient", False)) and not proof_gap
            if proof_gap and not rewritten_query:
                rewritten_query = heuristic_reflection_query(
                    original_query=original_query,
                    plan=plan,
                    evidence_items=evidence_items,
                    missing_aspects=llm_missing,
                    previous_queries=previous_queries,
                )
            return {
                "sufficient": llm_sufficient,
                "missing_aspects": llm_missing,
                "rewritten_query": rewritten_query,
                "reason": "strong_proof_counter_gap" if proof_gap else str(payload.get("reason") or "").strip(),
                "source": "llm",
                "llm_call": {
                    "type": "reflection",
                    "request": user_payload,
                    "response": payload,
                    "usage": response.get("usage", {}),
                },
            }
        except Exception as exc:
            llm_call = {
                "type": "reflection",
                "error": str(exc),
            }
        else:
            llm_call = {}
    else:
        llm_call = {}

    rewritten_query = ""
    reason = "sufficient_evidence"
    if not sufficient:
        rewritten_query = heuristic_reflection_query(
            original_query=original_query,
            plan=plan,
            evidence_items=evidence_items,
            missing_aspects=missing_aspects,
            previous_queries=previous_queries,
        )
        reason = "coverage_gap" if coverage.get("missing") else "insufficient_answer_support"
        if not topic_consistency.get("passed", True):
            reason = "topic_drift"
        if proof_gap:
            reason = "strong_proof_counter_gap"
    return {
        "sufficient": sufficient,
        "missing_aspects": missing_aspects,
        "rewritten_query": rewritten_query,
        "reason": reason,
        "source": "heuristic",
        "llm_call": llm_call,
    }
