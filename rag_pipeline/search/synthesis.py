from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..config.search_config import (
    DEFAULT_LLM_CONTEXT_DEDUP_THRESHOLD,
    DEFAULT_LLM_CONTEXT_MAX_TOKENS,
    DEFAULT_LLM_CONTEXT_MAX_TOKENS_PER_EVIDENCE,
)
from .context_builder import build_context_pack
from .memory import call_openai_compatible_json, llm_config_is_ready, normalize_llm_config
from .models import AnswerSynthesis, EvidenceItem, QueryPlan


_CITATION_RE = re.compile(r"\[(E\d+)\]")
_SECTION_HEADING_RE = re.compile(
    r"^(?:#{1,6}\s*)?(?:\*\*)?"
    r"(?:核心判断|投资结论|结论|摘要|分析框架|证据要点|关键依据|市场空间与增长|竞争格局/产业链|"
    r"关键变量|趋势方向|驱动因素|落地节奏|原因拆解|传导机制|证据强弱|对比维度|适用场景|"
    r"现状事实|关键变化|后续观察|风险与不确定性|证据缺口|风险与证据缺口|后续可追问|本次使用的本地证据)"
    r"(?:\*\*)?\s*[:：]?$"
)


def paragraph_requires_citation(paragraph: str) -> bool:
    cleaned = str(paragraph or "").strip()
    cleaned = re.sub(r"^[>\s*\-+]*(?:\d+[.、]\s*)?", "", cleaned).strip()
    return bool(cleaned) and not _SECTION_HEADING_RE.match(cleaned)


def partition_evidence_items(
    evidence_items: Sequence[EvidenceItem],
    *,
    core_top_k: int,
    support_top_k: int,
) -> Tuple[List[EvidenceItem], List[EvidenceItem]]:
    ranked = sorted(evidence_items, key=lambda item: float(item.evidence_score), reverse=True)
    core = ranked[: max(1, int(core_top_k))]
    support_pool = ranked[len(core) : max(len(core), int(support_top_k))]
    return core, support_pool


def build_llm_evidence_payload(
    evidence_items: Sequence[EvidenceItem],
    *,
    core_top_k: int,
    support_top_k: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, EvidenceItem]]:
    placeholder_plan = QueryPlan(
        original_query="",
        intent="",
        task_type="fact",
        normalized_query="",
    )
    payload, index_map, _ = build_context_pack(
        query="",
        plan=placeholder_plan,
        evidence_items=evidence_items,
        core_top_k=core_top_k,
        support_top_k=support_top_k,
        max_context_tokens=DEFAULT_LLM_CONTEXT_MAX_TOKENS,
        max_tokens_per_evidence=DEFAULT_LLM_CONTEXT_MAX_TOKENS_PER_EVIDENCE,
        dedup_threshold=DEFAULT_LLM_CONTEXT_DEDUP_THRESHOLD,
    )
    return payload, index_map


def build_llm_context_payload(
    *,
    query: str,
    plan: QueryPlan,
    evidence_items: Sequence[EvidenceItem],
    core_top_k: int,
    support_top_k: int,
    max_context_tokens: int,
    max_tokens_per_evidence: int,
    dedup_threshold: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, EvidenceItem], Dict[str, Any]]:
    return build_context_pack(
        query=query,
        plan=plan,
        evidence_items=evidence_items,
        core_top_k=core_top_k,
        support_top_k=support_top_k,
        max_context_tokens=max_context_tokens,
        max_tokens_per_evidence=max_tokens_per_evidence,
        dedup_threshold=dedup_threshold,
    )


def build_legacy_llm_evidence_payload(
    evidence_items: Sequence[EvidenceItem],
    *,
    core_top_k: int,
    support_top_k: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, EvidenceItem]]:
    core_items, support_items = partition_evidence_items(
        evidence_items,
        core_top_k=core_top_k,
        support_top_k=support_top_k,
    )
    payload: List[Dict[str, Any]] = []
    index_map: Dict[str, EvidenceItem] = {}
    merged_items = list(core_items) + list(support_items)
    for index, item in enumerate(merged_items, start=1):
        evidence_id = f"E{index}"
        index_map[evidence_id] = item
        payload.append(
            {
                "id": evidence_id,
                "tier": "core" if index <= len(core_items) else "support",
                "doc_title": item.doc_title,
                "section_title": item.section_title,
                "chunk_uid": item.chunk_uid,
                "group": item.group,
                "quote": item.quote,
                "evidence_score": round(float(item.evidence_score), 4),
                "final_score": round(float(item.final_score), 4),
                "citation": item.citation,
            }
        )
    return payload, index_map


def validate_answer_citations(answer_text: str, evidence_index: Dict[str, EvidenceItem]) -> List[str]:
    citations = _CITATION_RE.findall(str(answer_text or ""))
    unique = []
    seen = set()
    for citation in citations:
        if citation in evidence_index and citation not in seen:
            seen.add(citation)
            unique.append(citation)
    return unique


def normalize_answer_citation_marks(answer_text: str) -> str:
    return re.sub(r"[【\[]\s*(E\d+)\s*[】\]]", r"[\1]", str(answer_text or ""))


def add_missing_inline_citations(answer_text: str, citations: Sequence[str]) -> str:
    valid_citations = [citation for citation in citations if re.fullmatch(r"E\d+", str(citation or ""))]
    if not valid_citations:
        return str(answer_text or "").strip()
    default_citation_text = "".join(f"[{citation}]" for citation in valid_citations[:2])
    lines = []
    for line in str(answer_text or "").splitlines():
        stripped = line.strip()
        if paragraph_requires_citation(stripped) and not _CITATION_RE.search(stripped):
            lines.append(f"{line.rstrip()}{default_citation_text}")
        else:
            lines.append(line)
    return "\n".join(lines).strip()


def llm_response_is_grounded(answer_text: str, citations: Sequence[str]) -> bool:
    text = str(answer_text or "").strip()
    if not text:
        return False
    paragraphs = [part.strip() for part in re.split(r"\n+", text) if part.strip()]
    if not paragraphs:
        return False
    if not citations:
        return False
    citation_required = [paragraph for paragraph in paragraphs if paragraph_requires_citation(paragraph)]
    if not citation_required:
        return False
    supported_paragraphs = 0
    for paragraph in citation_required:
        if _CITATION_RE.search(paragraph):
            supported_paragraphs += 1
    return supported_paragraphs >= max(1, len(citation_required) - 1)


def build_industry_answer_blueprint(plan: QueryPlan) -> Dict[str, Any]:
    task_blueprints = {
        "market": {
            "preferred_sections": ["核心判断", "市场空间与增长", "竞争格局/产业链", "关键变量", "风险与证据缺口"],
            "must_consider": ["市场规模", "增速/渗透率", "竞争格局", "供需变化", "价格/盈利能力", "数据口径"],
        },
        "trend": {
            "preferred_sections": ["核心判断", "趋势方向", "驱动因素", "落地节奏", "风险与证据缺口"],
            "must_consider": ["技术/政策/需求驱动", "应用落地", "产业链传导", "时间口径", "反向证据"],
        },
        "root_cause": {
            "preferred_sections": ["核心判断", "原因拆解", "传导机制", "证据强弱", "风险与证据缺口"],
            "must_consider": ["直接原因", "结构性因素", "数据或案例支撑", "可验证指标"],
        },
        "comparison": {
            "preferred_sections": ["核心判断", "对比维度", "适用场景", "风险与证据缺口"],
            "must_consider": ["商业模式", "成本/效率", "竞争壁垒", "适用边界"],
        },
        "status": {
            "preferred_sections": ["核心判断", "现状事实", "关键变化", "后续观察", "风险与证据缺口"],
            "must_consider": ["最新时间口径", "进展", "问题/约束", "后续指标"],
        },
    }
    default_blueprint = {
        "preferred_sections": ["核心判断", "关键依据", "风险与证据缺口"],
        "must_consider": ["主体", "事实", "时间口径", "证据边界"],
    }
    blueprint = dict(task_blueprints.get(plan.task_type, default_blueprint))
    blueprint["style"] = [
        "结论先行，但避免超出证据做投资建议或预测",
        "优先提炼行业逻辑、核心变量、传导链条和证据缺口",
        "同一段里不要堆砌过多来源，保留最能支撑判断的引用",
    ]
    return blueprint


def synthesize_with_llm(
    *,
    query: str,
    plan: QueryPlan,
    evidence_items: Sequence[EvidenceItem],
    llm_config: Dict[str, Any],
    core_top_k: int,
    support_top_k: int,
    max_context_tokens: int,
    max_tokens_per_evidence: int,
    context_dedup_threshold: float,
) -> Tuple[AnswerSynthesis, Dict[str, Any]]:
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
        "你是一个面向产业研究、投研尽调和管理层简报场景的行研 RAG 智能体。只返回 JSON，不要输出 JSON 以外的文字。"
        "默认使用中文，除非用户明确要求其他语言。你的回答要像资深行研分析师，而不是检索日志或资料摘抄。"
        "核心原则：1) 只能使用给定证据，不使用外部知识；"
        "2) 每个实质性事实、判断、因果链或趋势判断都必须带证据 ID，例如 [E1]；"
        "3) 先给核心判断，再拆行业逻辑、关键变量、驱动因素、竞争/产业链/商业模式、风险与证据缺口；"
        "4) 区分“证据能支持的结论”和“仍缺少的口径”，不要把材料里的愿景、宣传语或个案直接当行业结论；"
        "5) 遇到证据不足、时间口径不清、数据互相冲突时，必须保守表述，并在 gaps/refusal_reason 中说明；"
        "6) 不要提到检索、chunk、prompt、JSON、内部得分等系统细节；"
        "7) 不要编造市场规模、增速、份额、财务指标或最新进展，证据没有就明确缺口。"
        "回答建议结构：核心判断；关键依据/分析框架；风险与证据缺口。"
        "只返回 JSON，字段必须包括：answer、citations、confidence、gaps、followups、refusal_reason、evidence_notes。"
        "示例：{'answer':'核心判断：当前证据只支持谨慎看多该方向的产业化潜力，但还不足以量化市场空间。[E1]\\n"
        "关键依据：材料显示需求侧和技术侧同时出现推动因素，因此更适合把机会拆成驱动、落地和风险三层观察。[E1][E2]',"
        "'citations':['E1','E2'],'confidence':0.74,'gaps':['缺少市场规模和最新时间口径'],"
        "'followups':['补充市场规模、竞争格局、盈利能力证据'],'refusal_reason':''}."
    )
    user_payload = {
        "query": query,
        "query_plan": {
            "task_type": plan.task_type,
            "intent": plan.intent,
            "theme_terms": plan.theme_terms,
            "entity_terms": plan.entity_terms,
            "constraint_terms": plan.constraint_terms,
            "time_terms": plan.time_terms,
            "evidence_focus": plan.evidence_focus,
            "needs_multi_hop": plan.needs_multi_hop,
        },
        "context_policy": {
            "evidence_order": "按相关性从高到低使用证据",
            "citation_rule": "每个事实性句子都用 [E#] 标注来源",
            "uncertainty_rule": "证据不完整时必须拒答或保守限定",
            "style_rule": "以行研 RAG 智能体口吻撰写，先给结论再展开分析",
        },
        "answer_blueprint": build_industry_answer_blueprint(plan),
        "evidence": evidence_payload,
    }
    response = call_openai_compatible_json(
        config=llm_config,
        system_prompt=system_prompt,
        user_payload=user_payload,
    )
    payload = response.get("payload", {})
    answer_text = normalize_answer_citation_marks(str(payload.get("answer") or "").strip())
    cited_ids = validate_answer_citations(answer_text, evidence_index)
    cited_ids.extend(
        citation
        for citation in payload.get("citations", []) or []
        if isinstance(citation, str) and citation in evidence_index and citation not in cited_ids
    )
    if not llm_response_is_grounded(answer_text, cited_ids) and cited_ids:
        answer_text = add_missing_inline_citations(answer_text, cited_ids)
        cited_ids = validate_answer_citations(answer_text, evidence_index)
    if not llm_response_is_grounded(answer_text, cited_ids):
        raise RuntimeError("LLM answer failed citation validation.")

    confidence = max(0.0, min(1.0, float(payload.get("confidence") or 0.0)))
    answer = AnswerSynthesis(
        status="answered" if not payload.get("refusal_reason") else "insufficient_evidence",
        confidence=round(confidence, 4),
        answer=answer_text,
        refusal_reason=str(payload.get("refusal_reason") or "").strip(),
        citations=cited_ids,
        gaps=[str(item).strip() for item in payload.get("gaps", []) if str(item).strip()],
        followups=[str(item).strip() for item in payload.get("followups", []) if str(item).strip()],
        grounding_mode="llm_grounded",
        llm_model=normalize_llm_config(llm_config).get("model", ""),
    )
    llm_call = {
        "type": "grounded_synthesis",
        "request": user_payload,
        "response": payload,
        "usage": response.get("usage", {}),
        "context_stats": context_stats,
    }
    return answer, llm_call


def synthesize_answer_with_fallback(
    *,
    query: str,
    plan: QueryPlan,
    evidence_items: Sequence[EvidenceItem],
    min_evidence: int,
    min_evidence_score: float,
    max_claims: int,
    llm_config: Optional[Dict[str, Any]],
    core_top_k: int,
    support_top_k: int,
    fallback_synthesizer: Callable[..., AnswerSynthesis],
    max_context_tokens: int = DEFAULT_LLM_CONTEXT_MAX_TOKENS,
    max_tokens_per_evidence: int = DEFAULT_LLM_CONTEXT_MAX_TOKENS_PER_EVIDENCE,
    context_dedup_threshold: float = DEFAULT_LLM_CONTEXT_DEDUP_THRESHOLD,
) -> Tuple[AnswerSynthesis, Dict[str, Any]]:
    _, _, context_stats = build_llm_context_payload(
        query=query,
        plan=plan,
        evidence_items=evidence_items,
        core_top_k=core_top_k,
        support_top_k=support_top_k,
        max_context_tokens=max_context_tokens,
        max_tokens_per_evidence=max_tokens_per_evidence,
        dedup_threshold=context_dedup_threshold,
    )
    fallback = fallback_synthesizer(
        query=query,
        plan=plan,
        evidence_items=evidence_items,
        min_evidence=min_evidence,
        min_evidence_score=min_evidence_score,
        max_claims=max_claims,
    )
    llm_meta: Dict[str, Any] = {
        "type": "grounded_synthesis",
        "source": "fallback_extractive",
        "response": fallback.to_dict(),
        "context_stats": context_stats,
    }
    if fallback.status != "answered":
        return fallback, llm_meta
    if not llm_config_is_ready(llm_config):
        return fallback, llm_meta

    try:
        llm_answer, llm_call = synthesize_with_llm(
            query=query,
            plan=plan,
            evidence_items=evidence_items,
            llm_config=dict(llm_config or {}),
            core_top_k=core_top_k,
            support_top_k=support_top_k,
            max_context_tokens=max_context_tokens,
            max_tokens_per_evidence=max_tokens_per_evidence,
            context_dedup_threshold=context_dedup_threshold,
        )
        llm_answer.claims = fallback.claims
        llm_answer.conflicts = fallback.conflicts
        if not llm_answer.citations:
            llm_answer.citations = fallback.citations
        return llm_answer, llm_call
    except Exception as exc:
        llm_meta = {
            "type": "grounded_synthesis",
            "source": "fallback_extractive",
            "error": str(exc),
            "response": fallback.to_dict(),
            "context_stats": context_stats,
        }
        return fallback, llm_meta
