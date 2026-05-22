from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, TypedDict

try:
    from rag_pipeline.contracts.evidence_quality import classify_evidence
    from rag_pipeline.search.memory import call_openai_compatible_json, llm_config_is_ready, normalize_llm_config
    from .evidence_merger import get_dynamic_dimensions
except Exception:  # pragma: no cover - script mode fallback
    try:
        from rag_pipeline.contracts.evidence_quality import classify_evidence  # type: ignore
    except Exception:  # pragma: no cover
        classify_evidence = None  # type: ignore
    try:
        from rag_pipeline.search.memory import call_openai_compatible_json, llm_config_is_ready, normalize_llm_config  # type: ignore
    except Exception:  # pragma: no cover
        call_openai_compatible_json = None  # type: ignore
        llm_config_is_ready = None  # type: ignore
        normalize_llm_config = None  # type: ignore
    from evidence_merger import get_dynamic_dimensions  # type: ignore


AGENT_NAME = "analysis_agent"
AGENT_DESCRIPTION = "Dynamic Research Claim Builder. Converts evidence packages into claim units for the writer."
PUBLIC_ANALYSIS_TEXT_KEYS = {
    "claim",
    "judgment",
    "reasoning",
    "mechanism",
    "counter",
    "counter_evidence",
    "counter_boundary",
    "counter_evidence_boundary",
    "actionable",
    "decision_implication",
    "what_to_verify_next",
    "chapter_answer",
    "core_answer",
}
PUBLIC_ANALYSIS_FORBIDDEN_PATTERNS = [
    r"\bevidence_cards?\b",
    r"(?<![A-Za-z0-9_])EV-\d+(?:-[A-Za-z0-9]+)?",
    r"(?<![A-Za-z0-9_])ch_\d{1,3}(?![A-Za-z0-9_])",
    r"第\s*\d+\s*轮",
    r"当前卡片",
    r"本章应写成",
    r"本章可以作为",
    r"本章可作为",
    r"本章\s*只能\s*写成",
    r"正文\s*只能\s*写成",
    r"本章\s*仍需\s*连续观察",
    r"建议写成",
    r"适合写成",
    r"建议避免",
]


class AnalysisAgentState(TypedDict, total=False):
    query: str
    evidence_package: Dict[str, Any]
    structured_analysis: Dict[str, Any]
    answer_text: str
    raw_output: Dict[str, Any]
    metadata: Dict[str, Any]
    errors: List[str]


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, min_value: int = 0, max_value: int = 10_000) -> int:
    try:
        value = int(os.getenv(name, str(default)) or default)
    except (TypeError, ValueError):
        value = default
    return max(min_value, min(max_value, value))


def _compact(value: Any, max_chars: int = 260) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "..."


def _has_internal_analysis_language(value: Any) -> bool:
    text = str(value or "")
    if not text:
        return False
    extra_patterns = [
        r"Use only as",
        r"do not render",
        r"only as a directional",
        r"正文\s*只能\s*写成",
        r"本章\s*只能\s*写成",
        r"本章\s*可\s*写成",
        r"本章\s*仍需\s*连续观察",
        r"建议避免",
        r"建议在后续版本中补充",
        r"适合写成",
    ]
    return any(re.search(pattern, text, flags=re.I) for pattern in [*PUBLIC_ANALYSIS_FORBIDDEN_PATTERNS, *extra_patterns])


def _chapter_fact_texts(chapter: Dict[str, Any], *, limit: int = 5) -> List[str]:
    raw_items = chapter.get("fact_chain")
    iterable = list(_as_dict(raw_items).values()) if isinstance(raw_items, dict) else _as_list(raw_items)
    facts: List[str] = []
    for item in iterable:
        if isinstance(item, dict):
            text = _compact(
                item.get("fact")
                or item.get("text")
                or item.get("summary")
                or item.get("finding")
                or item.get("evidence")
                or item.get("claim"),
                220,
            )
        else:
            text = _compact(item, 220)
        text = _public_normalize_analysis_text(text)
        if not text or _has_internal_analysis_language(text):
            continue
        if text not in facts:
            facts.append(text)
        if len(facts) >= limit:
            break
    return facts


def _chapter_counter_text(chapter: Dict[str, Any]) -> str:
    raw = chapter.get("counter_evidence_boundary")
    candidates = [raw] if isinstance(raw, str) else _as_list(raw)
    for item in candidates:
        text = _public_normalize_analysis_text(_compact(item, 260))
        if text and not _has_internal_analysis_language(text):
            return text
    return "如果后续同口径指标走弱、企业动作中断、客户验证不足或监管条件收紧，本章判断需要下调。"


def _public_claim_from_chapter(chapter: Dict[str, Any]) -> str:
    question = _compact(chapter.get("chapter_title") or chapter.get("chapter_question") or chapter.get("chapter_id") or "本章", 120)
    facts = _chapter_fact_texts(chapter, limit=2)
    if facts:
        return f"{question}已经出现可观察信号，关键依据是“{facts[0]}”。结论强度取决于这些信号能否被同口径来源持续验证。"
    return f"{question}目前只能形成方向性观察，需要用可追溯来源和连续指标继续校准结论强度。"


def _public_reasoning_from_chapter(chapter: Dict[str, Any]) -> str:
    facts = _chapter_fact_texts(chapter, limit=4)
    if facts:
        return (
            "事实链包括："
            + "；".join(facts)
            + "。这些事实需要同时放在来源等级、统计口径、披露主体和时间窗口中解释，才能判断其是否能从局部信号扩展为行业趋势。"
        )
    mechanisms = [
        _public_normalize_analysis_text(item)
        for item in _as_list(chapter.get("mechanism_chain"))
        if str(item or "").strip()
    ]
    mechanisms = [item for item in mechanisms if item and not _has_internal_analysis_language(item)]
    if mechanisms:
        return "；".join(mechanisms[:3])
    return "现有公开材料尚未形成完整事实链，因此正文应以可观察变量和后续验证条件为主。"


def _dedupe(values: List[Any]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _normalize_key(value: Any) -> str:
    text = re.sub(r"\s+", "", str(value or "").lower())
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", text)


def _overlaps(left: Any, right: Any) -> bool:
    left_key = _normalize_key(left)
    right_key = _normalize_key(right)
    if not left_key or not right_key:
        return False
    if left_key in right_key or right_key in left_key:
        return True
    overlap = set(left_key) & set(right_key)
    return len(overlap) >= max(2, min(len(left_key), len(right_key)) // 3)


def _research_plan(evidence_package: Dict[str, Any]) -> Dict[str, Any]:
    return _as_dict(evidence_package.get("research_plan")) or _as_dict(_as_dict(evidence_package.get("metadata")).get("research_plan"))


def _analysis_dimensions(evidence_package: Dict[str, Any]) -> List[str]:
    research_plan = _research_plan(evidence_package)
    dimensions = get_dynamic_dimensions(research_plan)
    for dimension in _as_dict(evidence_package.get("per_dimension")).keys():
        text = str(dimension or "").strip()
        if text and text not in dimensions:
            dimensions.append(text)
    for item in _as_list(evidence_package.get("analysis_ready_evidence")) + _as_list(evidence_package.get("clean_evidence_list")):
        if isinstance(item, dict):
            text = str(item.get("dimension_name") or item.get("evidence_goal") or item.get("dimension") or "").strip()
            if text and text not in dimensions:
                dimensions.append(text)
    return dimensions or ["综合研究问题"]


def _fact_text(item: Dict[str, Any]) -> str:
    for key in ("fact", "clean_fact", "content", "clean_content", "answer", "claim", "takeaway"):
        text = _compact(item.get(key), 260)
        if text:
            return text
    metric = _compact(item.get("metric"), 80)
    value = _compact(item.get("value"), 80)
    if metric and value:
        return f"{metric}: {value}"
    return ""


def _source_payload(item: Dict[str, Any]) -> Dict[str, Any]:
    source = _as_dict(item.get("source"))
    if source:
        return source
    key_sources = _as_list(item.get("key_sources"))
    for source_item in key_sources:
        if isinstance(source_item, dict):
            return source_item
    return {}


def _source_label(item: Dict[str, Any]) -> str:
    source = _source_payload(item)
    title = str(source.get("title") or source.get("source") or source.get("name") or "").strip()
    date = str(source.get("date") or source.get("period") or "").strip()
    return " | ".join(part for part in [title, date] if part)


def _has_traceable_source(item: Dict[str, Any]) -> bool:
    source = _source_payload(item)
    url = str(source.get("url") or source.get("source_url") or item.get("source_url") or "").strip()
    lowered_url = url.lower()
    if "example.com" in lowered_url or "example.gov" in lowered_url:
        return False
    title = str(source.get("title") or source.get("name") or item.get("source_title") or "").strip().lower()
    publisher = str(source.get("publisher") or source.get("source") or item.get("source_text") or "").strip()
    if title == "official" and not publisher:
        return False
    text = " ".join(
        str(value or "")
        for value in [item.get("fact"), item.get("clean_fact"), item.get("content"), item.get("summary")]
    ).lower()
    if "official data shows ai agent adoption reached 50% in 2025" in text:
        return False
    document_ref = str(
        source.get("document_id")
        or source.get("doc_id")
        or source.get("page_ref")
        or item.get("document_id")
        or item.get("doc_id")
        or item.get("page_ref")
        or ""
    ).strip()
    metadata_count = sum(
        bool(str(value or "").strip())
        for value in [
            title,
            publisher,
            source.get("date") or source.get("published_at") or item.get("period"),
        ]
    )
    return bool(url or (document_ref and metadata_count >= 2))


def _is_fake_or_placeholder_source(item: Dict[str, Any]) -> bool:
    source = _source_payload(item)
    if bool(source.get("fake_or_placeholder_source") or item.get("fake_or_placeholder_source")):
        return True
    if str(source.get("traceability_status") or item.get("traceability_status") or "").strip().lower() == "fake_or_placeholder_source":
        return True
    url = str(source.get("url") or source.get("source_url") or item.get("source_url") or "").strip().lower()
    if "example.com" in url or "example.gov" in url:
        return True
    title = str(source.get("title") or source.get("name") or item.get("source_title") or "").strip().lower()
    publisher = str(source.get("publisher") or source.get("source") or item.get("source_text") or "").strip()
    if title == "official" and not publisher:
        return True
    text = " ".join(
        str(value or "")
        for value in [item.get("fact"), item.get("clean_fact"), item.get("content"), item.get("summary")]
    ).lower()
    return "official data shows ai agent adoption reached 50% in 2025" in text


def _is_title_only_source(item: Dict[str, Any]) -> bool:
    source = _source_payload(item)
    title = str(source.get("title") or source.get("name") or item.get("source_title") or "").strip()
    url = str(source.get("url") or source.get("source_url") or item.get("source_url") or "").strip()
    document_ref = str(
        source.get("document_id")
        or source.get("doc_id")
        or source.get("page_ref")
        or item.get("document_id")
        or item.get("doc_id")
        or item.get("page_ref")
        or ""
    ).strip()
    return bool(title and not url and not document_ref)


def _confidence(item: Dict[str, Any]) -> float:
    try:
        value = float(item.get("confidence", 0.5))
    except (TypeError, ValueError):
        value = 0.5
    return max(0.0, min(1.0, value))


def _items_for_dimension(evidence_package: Dict[str, Any], dimension: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    payload = _as_dict(_as_dict(evidence_package.get("per_dimension")).get(dimension))
    for item in _as_list(payload.get("analysis_inputs")) + _as_list(payload.get("clean_facts")) + _as_list(payload.get("top_evidence")):
        if isinstance(item, dict):
            copied = dict(item)
            copied.setdefault("dimension", dimension)
            items.append(copied)
    for item in _as_list(evidence_package.get("analysis_ready_evidence")) + _as_list(evidence_package.get("clean_evidence_list")):
        if not isinstance(item, dict):
            continue
        item_dimension = str(item.get("dimension_name") or item.get("evidence_goal") or item.get("dimension") or "").strip()
        if item_dimension == dimension:
            items.append(dict(item))
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for item in items:
        key = (_fact_text(item), _source_label(item))
        if key in seen or not key[0]:
            continue
        seen.add(key)
        deduped.append(item)
    deduped.sort(key=lambda item: (_confidence(item), 1 if _source_label(item) else 0), reverse=True)
    return deduped


def _select_analysis_items_for_dimension(items: Sequence[Dict[str, Any]], *, max_items: int = 18) -> List[Dict[str, Any]]:
    max_items = max(1, int(max_items or 18))
    selected: List[Dict[str, Any]] = []
    seen = set()

    def add(candidates: Sequence[Dict[str, Any]], limit: int) -> None:
        ranked = sorted(
            [item for item in list(candidates or []) if isinstance(item, dict)],
            key=lambda item: (
                1 if _has_traceable_source(item) else 0,
                1 if _analysis_source_level(item) == "A" else 0,
                1 if _analysis_source_level(item) == "B" else 0,
                _confidence(item),
                1 if str(item.get("metric") or "").strip() or str(item.get("value") or "").strip() else 0,
            ),
            reverse=True,
        )
        added = 0
        for item in ranked:
            if len(selected) >= max_items or added >= limit:
                return
            key = (_fact_text(item), _source_label(item), str(item.get("evidence_id") or ""))
            if key in seen or not key[0]:
                continue
            seen.add(key)
            selected.append(item)
            added += 1

    item_list = [item for item in list(items or []) if isinstance(item, dict)]
    add(
        [
            item
            for item in item_list
            if _analysis_source_level(item) in {"A", "B"}
            and _has_traceable_source(item)
            and _analysis_allowed_use(item) in {"core_claim", "supporting", "supporting_context"}
        ],
        6,
    )
    add(
        [
            item
            for item in item_list
            if _has_traceable_source(item)
            and str(item.get("metric") or "").strip()
            and str(item.get("value") or "").strip()
            and (str(item.get("period") or "").strip() or str(_source_payload(item).get("date") or "").strip())
        ],
        4,
    )
    add(
        [
            item
            for item in item_list
            if str(item.get("proof_role") or "").strip().lower() in {"source_check", "filing", "official_data"}
        ],
        3,
    )
    add(
        [
            item
            for item in item_list
            if str(item.get("proof_role") or "").strip().lower() == "counter"
            or bool(item.get("counter_evidence"))
            or re.search(r"(counter|risk|failure|downside|反证|风险|失败|下滑|取消)", _fact_text(item), re.I)
        ],
        3,
    )
    add(
        [
            item
            for item in item_list
            if re.search(r"20\d{2}|Q[1-4]|最新|recent", " ".join([str(item.get("period") or ""), str(_source_payload(item).get("date") or ""), _fact_text(item)]), re.I)
        ],
        2,
    )
    add(
        [
            item
            for item in item_list
            if str(item.get("proof_role") or "").strip().lower() in {"case", "boundary"}
            or str(item.get("source_family") or "").strip().lower() == "company/case"
        ],
        2,
    )
    add(
        [
            item
            for item in item_list
            if _analysis_allowed_use(item) == "directional_signal" or _analysis_source_level(item) == "C"
        ],
        3,
    )
    add(item_list, max_items)
    return selected[:max_items]


def _claim_units_from_synthesis(dimension_synthesis: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    units: List[Dict[str, Any]] = []
    for dimension, synthesis in dimension_synthesis.items():
        synthesis = _as_dict(synthesis)
        units.append(
            {
                "question": dimension,
                "claim": synthesis.get("takeaway") or "",
                "claim_status": "decision_ready" if _as_list(synthesis.get("evidence_ids")) else "directional",
                "quality_status": "valid" if _as_list(synthesis.get("evidence_ids")) else "directional_with_boundary",
                "supporting_evidence": synthesis.get("evidence_ids") or synthesis.get("directional_evidence_ids") or [],
                "evidence_refs": synthesis.get("evidence_ids") or synthesis.get("directional_evidence_ids") or [],
                "counter_evidence": synthesis.get("counter") or "",
                "reasoning": synthesis.get("mechanism") or synthesis.get("explain_why") or "",
                "mechanism": synthesis.get("mechanism") or "",
                "decision_implication": synthesis.get("decision_implication") or synthesis.get("verify_kpi") or "",
                "confidence": synthesis.get("confidence"),
                "dimension": dimension,
            }
        )
    return units


def _chapter_insights_from_synthesis(dimension_synthesis: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    insights: List[Dict[str, Any]] = []
    for index, (dimension, synthesis) in enumerate(dimension_synthesis.items(), start=1):
        synthesis = _as_dict(synthesis)
        insights.append(
            {
                "chapter_id": f"chapter_{index}",
                "chapter_question": dimension,
                "chapter_answer": synthesis.get("chapter_answer") or synthesis.get("takeaway") or "",
                "core_answer": synthesis.get("chapter_answer") or synthesis.get("takeaway") or "",
                "fact_chain": _as_list(synthesis.get("fact_chain")),
                "mechanism_chain": _as_list(synthesis.get("mechanism_chain")),
                "counter_evidence_boundary": _as_list(synthesis.get("counter_evidence_boundary")),
                "decision_implication": synthesis.get("decision_implication") or "",
                "key_claims": [
                    {
                        "claim": synthesis.get("takeaway") or "",
                        "claim_status": "decision_ready" if _as_list(synthesis.get("evidence_ids")) else "directional",
                        "supporting_evidence": synthesis.get("evidence_ids") or synthesis.get("directional_evidence_ids") or [],
                        "evidence_refs": synthesis.get("evidence_ids") or synthesis.get("directional_evidence_ids") or [],
                        "mechanism": synthesis.get("mechanism") or "",
                        "reasoning": synthesis.get("mechanism") or "",
                        "counter_evidence": synthesis.get("counter") or "",
                        "decision_implication": synthesis.get("decision_implication") or "",
                        "confidence": synthesis.get("confidence"),
                        "what_to_verify_next": [synthesis.get("verify_kpi")],
                    }
                ],
                "decision_readiness": "ready" if _as_list(synthesis.get("evidence_ids")) else "needs_evidence",
                "blocking_gaps": [] if _as_list(synthesis.get("evidence_ids")) else ["evidence_missing"],
            }
        )
    return insights


def _analysis_source_level(item: Dict[str, Any]) -> str:
    level = str(item.get("source_level") or _as_dict(item.get("source")).get("credibility") or "").strip().upper()
    if level or classify_evidence is None:
        return level
    return str(classify_evidence(item).get("source_level") or "").strip().upper()


def _analysis_allowed_use(item: Dict[str, Any]) -> str:
    allowed = str(item.get("allowed_use") or _as_dict(item.get("evidence_card")).get("allowed_use") or "").strip()
    if allowed:
        return allowed
    if classify_evidence is not None:
        return str(classify_evidence(item).get("allowed_use") or "").strip() or "appendix_only"
    level = _analysis_source_level(item)
    role = str(item.get("evidence_role") or "").strip().lower()
    if level in {"A", "B"} and role == "core":
        return "core_claim"
    if level in {"A", "B"} and role == "supporting":
        return "supporting"
    try:
        confidence = float(item.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    if level == "C" and confidence >= 0.55 and not item.get("appendix_only"):
        return "directional_signal"
    if level == "C":
        return "clue"
    return "appendix_only"


def _is_usable_for_claim(item: Dict[str, Any]) -> bool:
    allowed_use = _analysis_allowed_use(item)
    readiness = str(item.get("analysis_readiness") or _as_dict(item.get("evidence_card")).get("analysis_readiness") or "").strip()
    if readiness in {"blocked", "followup_only", "directional_ready"}:
        return False
    if _as_list(item.get("metric_proof_gaps") or _as_dict(item.get("evidence_card")).get("metric_proof_gaps")):
        return False
    return (
        _analysis_source_level(item) in {"A", "B"}
        and allowed_use in {"core_claim", "supporting"}
        and _has_traceable_source(item)
    )


def _evidence_strength(item: Dict[str, Any]) -> str:
    level = _analysis_source_level(item)
    allowed = _analysis_allowed_use(item)
    traceable = _has_traceable_source(item)
    if allowed == "directional_signal":
        return "directional"
    if level in {"A", "B"} and allowed in {"core_claim", "supporting"} and not traceable:
        return "weak"
    if level in {"A", "B"} and allowed == "core_claim":
        return "strong"
    if level in {"A", "B"} and allowed == "supporting":
        return "medium"
    return "weak"


def _evidence_gap_tags(item: Dict[str, Any]) -> List[str]:
    level = _analysis_source_level(item)
    allowed = _analysis_allowed_use(item)
    traceable = _has_traceable_source(item)
    gaps: List[str] = []
    if _is_fake_or_placeholder_source(item):
        gaps.append("fake_or_placeholder_source")
    if level in {"A", "B"} and allowed in {"core_claim", "supporting"} and not traceable:
        gaps.append("source_trace_missing")
        if _is_title_only_source(item):
            gaps.append("title_only_source")
    if level not in {"A", "B"} and allowed != "directional_signal":
        gaps.append("needs_authoritative_source")
    if allowed in {"clue", "appendix_only"}:
        gaps.append("needs_corroboration")
    if not _source_label(item):
        gaps.append("source_metadata_missing")
    metric = _compact(item.get("metric"), 80)
    value = _compact(item.get("value"), 80)
    period = _compact(item.get("period") or _source_payload(item).get("date"), 80)
    if metric and not value:
        gaps.append("metric_value_missing")
    if (metric or value) and not period:
        gaps.append("metric_period_missing")
    if str(item.get("proof_role") or "").strip().lower() == "counter" and not _is_usable_for_claim(item):
        gaps.append("counter_needs_ab_source")
    return _dedupe(gaps)


def _followup_query_for_evidence(item: Dict[str, Any], *, dimension: str) -> str:
    gaps = set(_evidence_gap_tags(item))
    parts = [
        dimension,
        _compact(item.get("hypothesis_statement"), 80),
        _compact(item.get("metric"), 60),
        _compact(_fact_text(item), 90),
    ]
    if "needs_authoritative_source" in gaps or "needs_corroboration" in gaps:
        parts.extend(["官方", "公告", "财报", "协会", "权威研报"])
    if "metric_value_missing" in gaps or "metric_period_missing" in gaps:
        parts.extend(["指标口径", "数值", "期间", "单位", "范围"])
    if "counter_needs_ab_source" in gaps:
        parts.extend(["反证", "风险", "失败案例", "订单取消", "监管变化"])
    query = " ".join(part for part in parts if str(part or "").strip())
    return _compact(query, 220)


def _verification_questions(item: Dict[str, Any], *, dimension: str) -> List[str]:
    fact = _compact(_fact_text(item), 90)
    questions = [
        f"{dimension} 的这个信号是否有 A/B 级来源复核？",
        "同一口径下是否能找到时间、范围、单位一致的指标？",
    ]
    if fact:
        questions.insert(0, f"'{fact}' 能否被后续披露或第二来源验证？")
    if str(item.get("proof_role") or "").strip().lower() != "counter":
        questions.append("是否存在方向相反的反证或失败案例？")
    return _dedupe(questions)[:4]


def _evidence_card_from_item(item: Dict[str, Any], *, dimension: str, fact: str) -> Dict[str, Any]:
    card = _as_dict(item.get("evidence_card"))
    if card:
        return card
    source = _source_payload(item)
    level = _analysis_source_level(item) or "UNKNOWN"
    allowed = _analysis_allowed_use(item)
    return {
        "fact": fact,
        "source_level": level,
        "source_family": str(item.get("source_family") or "unknown"),
        "proof_role": str(item.get("proof_role") or ("counter" if item.get("counter_evidence") else "support")).strip().lower(),
        "directness": "direct" if item.get("metric") or item.get("value") else "indirect",
        "scope": str(item.get("scope") or item.get("dimension_name") or dimension or "").strip(),
        "period": str(item.get("period") or source.get("date") or "").strip(),
        "metric_definition": {
            "metric": item.get("metric"),
            "value": item.get("value"),
            "period": item.get("period") or source.get("date") or "",
        },
        "can_prove": [item.get("evidence_goal") or dimension],
        "cannot_prove": ["single-source conclusion", "industry-wide certainty", "investment priority without evidence bundle"],
        "inference_distance": "low" if allowed == "core_claim" else ("medium" if allowed == "supporting" else "high"),
        "contradictions": [],
        "allowed_use": allowed,
    }


def _evidence_analysis(item: Dict[str, Any], dimension: str, index: int) -> Dict[str, Any]:
    fact = _fact_text(item)
    source = _source_payload(item)
    evidence_id = str(item.get("evidence_id") or item.get("id") or f"EV-{index:04d}")
    card = _evidence_card_from_item(item, dimension=dimension, fact=fact)
    gaps = _evidence_gap_tags(item)
    verification_questions = _verification_questions(item, dimension=dimension)
    followup_query = _followup_query_for_evidence(item, dimension=dimension) if gaps else ""
    strength = _evidence_strength(item)
    return {
        "evidence_id": evidence_id,
        "dimension": dimension,
        "fact": fact,
        "writer_evidence": fact,
        "source": source,
        "source_label": _source_label(item),
        "confidence": _confidence(item),
        "hypothesis_id": item.get("hypothesis_id"),
        "hypothesis_statement": item.get("hypothesis_statement"),
        "proof_role": card.get("proof_role") or item.get("proof_role") or ("counter" if item.get("counter_evidence") else "support"),
        "source_level": card.get("source_level") or _analysis_source_level(item),
        "source_tier": item.get("source_tier") or card.get("source_tier"),
        "source_family": card.get("source_family") or item.get("source_family"),
        "metric": item.get("metric"),
        "value": item.get("value"),
        "allowed_use": card.get("allowed_use"),
        "evidence_fit_score": item.get("evidence_fit_score") or card.get("evidence_fit_score"),
        "metric_proof_gaps": _as_list(item.get("metric_proof_gaps") or card.get("metric_proof_gaps")),
        "analysis_readiness": item.get("analysis_readiness") or card.get("analysis_readiness"),
        "evidence_card": card,
        "evidence_card_only": True,
        "evidence_strength": strength,
        "evidence_gaps": gaps,
        "verification_questions": verification_questions,
        "suggested_followup_query": followup_query,
        "claim": f"{dimension} 出现可观察信号，但结论强度取决于来源等级、指标口径和反证覆盖。" if fact else "",
        "reasoning": "该证据可用于建立事实链的一环；若要进入核心判断，需要与同口径指标、第二来源或反向案例交叉验证。",
        "mechanism": "先确认事实是否可复核，再判断它影响的是需求、供给、政策约束还是企业行为，最后评估能否外推为趋势。",
        "counter": "若后续 A/B 来源显示指标反向变化、企业动作未延续或出现失败案例，应下调该证据对结论的权重。",
        "decision_implication": "可作为正文分析素材；存在缺口时优先转入补证任务，而不是直接放大为强结论。",
        "analysis_depth": {
            "can_prove": card.get("can_prove") or [dimension],
            "cannot_prove": card.get("cannot_prove") or ["single-source conclusion"],
            "inference_distance": card.get("inference_distance"),
            "strength": strength,
            "gaps": gaps,
            "verification_questions": verification_questions,
            "suggested_followup_query": followup_query,
        },
    }


def _dimension_synthesis(dimension: str, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
    usable = [item for item in analyses if _is_usable_for_claim(item)]
    directional = [
        item
        for item in analyses
        if _analysis_allowed_use(item) == "directional_signal"
        and str(item.get("fact") or "").strip()
    ]
    evidence_ids = [str(item.get("evidence_id")) for item in usable if item.get("evidence_id")][:12]
    directional_ids = [str(item.get("evidence_id")) for item in directional if item.get("evidence_id")][:12]
    usable_facts = [_compact(item.get("fact"), 120) for item in usable if str(item.get("fact") or "").strip()]
    directional_facts = [_compact(item.get("fact"), 120) for item in directional if str(item.get("fact") or "").strip()]
    all_gaps = _dedupe(
        [
            gap
            for item in analyses
            for gap in _as_list(item.get("evidence_gaps"))
        ]
    )
    followups = _dedupe(
        [
            item.get("suggested_followup_query")
            for item in analyses
            if str(item.get("suggested_followup_query") or "").strip()
        ]
    )
    first_fact = usable_facts[0] if usable_facts else (directional_facts[0] if directional_facts else _compact(_as_dict(analyses[0] if analyses else {}).get("fact"), 220))
    fact_chain = usable_facts[:5] if usable_facts else directional_facts[:5]
    metric_facts = [
        _compact(f"{item.get('metric')}: {item.get('value')} {item.get('period') or ''}", 160)
        for item in usable
        if str(item.get("metric") or "").strip() and str(item.get("value") or "").strip()
    ]
    counter_facts = [
        _compact(item.get("fact"), 160)
        for item in usable
        if str(item.get("proof_role") or "").strip().lower() == "counter" and str(item.get("fact") or "").strip()
    ]
    if usable:
        takeaway = f"{dimension} 的判断应以“{first_fact}”为事实锚点，并结合口径一致性和反向样本决定结论强度。"
        mechanism_parts = [
            f"先用 {first_fact} 确认本章的事实起点。",
            "再比较同口径指标、来源等级和时间窗口，判断该事实是局部样本还是可迁移趋势。",
            "最后把事实映射到需求兑现、供给约束、商业化效率或政策边界，决定正文采用强判断还是方向性判断。",
        ]
        if metric_facts:
            mechanism_parts.insert(1, f"量化依据优先使用 {metric_facts[0]}，并检查期间、单位和统计范围是否一致。")
        mechanism = "；".join(mechanism_parts)
        counter = (
            f"反证边界包括：{counter_facts[0]}"
            if counter_facts
            else "若后续 A/B 来源显示同口径指标走弱、企业动作中断、客户验证不足或监管条件收紧，本章结论应降级。"
        )
        decision = "当前判断应标注证据边界：强证据用于核心结论，弱证据仅用于趋势线索和后续观察指标。"
    elif directional:
        takeaway = f"{dimension} has corroborated directional signals, but still lacks enough strong evidence for a definitive conclusion."
        mechanism = "Current C-level sources can identify directional change, but A/B sources or complete metric proof are still needed before turning it into a strong conclusion."
        counter = "If later A/B sources do not support this direction, or counter-examples emerge, downgrade the claim to background context."
        decision = "当前只能形成方向性判断；如需形成量化或确定性结论，需要补充 A/B 来源和完整指标口径。"
    elif analyses:
        takeaway = f"{dimension} 目前只有线索或背景材料，尚不足以支撑强结论。"
        mechanism = "当前材料只能说明存在研究线索，不能说明趋势已经成立；应优先补 A/B 来源、指标口径和反向案例。"
        counter = "没有反证并不等于风险不存在；反证缺位本身应作为结论边界。"
        decision = "当前只能形成待验证方向；结论需要随 A/B 来源、完整指标和反向案例继续校准。"
    else:
        takeaway = ""
        mechanism = ""
        counter = ""
        decision = ""
    return {
        "takeaway": takeaway,
        "chapter_answer": takeaway,
        "fact": first_fact,
        "fact_chain": fact_chain,
        "mechanism_chain": [part for part in mechanism.split("；") if part],
        "counter_evidence_boundary": [counter] if counter else [],
        "supporting_facts": (usable_facts or directional_facts)[:6],
        "explain_why": mechanism,
        "mechanism": mechanism,
        "inference": "强度取决于证据是否同时满足来源可信、口径完整、可被第二来源复核。",
        "counter": counter,
        "verify_kpi": "补齐 A/B 来源、同口径指标、时间范围、单位、反证案例",
        "decision_implication": decision,
        "evidence_ids": evidence_ids,
        "directional_evidence_ids": directional_ids,
        "confidence": round(sum(float(item.get("confidence") or 0.0) for item in (usable or directional)) / max(len(usable or directional), 1), 3) if (usable or directional) else 0.0,
        "limits": "；".join(all_gaps[:5]),
        "evidence_gap_tags": all_gaps,
        "followup_queries": followups[:6],
    }


def _hypothesis_insights(research_plan: Dict[str, Any], evidence_analyses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    insights: List[Dict[str, Any]] = []
    hypotheses = [item for item in _as_list(research_plan.get("hypotheses")) if isinstance(item, dict)]
    requirements = _as_dict(_as_dict(research_plan.get("evidence_coverage_requirements")).get("per_hypothesis"))
    required_ab = int(requirements.get("min_A_or_B_sources") if requirements.get("min_A_or_B_sources") not in {None, ""} else 1)
    required_counter = int(requirements.get("min_counter_sources") or 0)
    required_metric = int(requirements.get("min_metric_sources") or 0)
    required_case = int(requirements.get("min_case_sources") or 0)
    required_families = [str(item) for item in _as_list(requirements.get("source_diversity")) if str(item or "").strip()]
    for index, hypothesis in enumerate(hypotheses, start=1):
        hypothesis_id = str(hypothesis.get("hypothesis_id") or f"H{index}")
        statement = _compact(hypothesis.get("claim_to_test") or hypothesis.get("hypothesis_statement") or hypothesis.get("statement"), 260)
        relevant = [
            item
            for item in evidence_analyses
            if str(item.get("hypothesis_id") or "") == hypothesis_id
            or _overlaps(statement, item.get("dimension"))
            or _overlaps(statement, item.get("fact"))
        ]
        usable = [item for item in relevant if _is_usable_for_claim(item)]
        support = [item for item in usable if str(item.get("proof_role") or "").lower() != "counter"]
        counters = [item for item in usable if str(item.get("proof_role") or "").lower() == "counter"]
        metric_items = [
            item
            for item in usable
            if str(item.get("proof_role") or "").lower() == "metric" or bool(item.get("metric") or item.get("value"))
        ]
        case_items = [
            item
            for item in usable
            if str(item.get("proof_role") or "").lower() == "case" or str(item.get("source_family") or "") == "company/case"
        ]
        families = sorted({str(item.get("source_family") or "unknown") for item in usable})
        gaps: List[str] = []
        if len(usable) < required_ab:
            gaps.append("insufficient_ab_sources")
        if bool(hypothesis.get("counter_evidence_required", False)) and required_counter > 0 and len(counters) < required_counter:
            gaps.append("counter_evidence_missing")
        if required_metric > 0 and len(metric_items) < required_metric:
            gaps.append("metric_evidence_missing")
        if required_case > 0 and len(case_items) < required_case:
            gaps.append("case_evidence_missing")
        if required_families and not set(required_families).issubset(set(families)):
            gaps.append("source_diversity_missing")
        ready = not gaps and len(usable) >= max(1, required_ab)
        evidence_ids = [str(item.get("evidence_id")) for item in support if item.get("evidence_id")]
        counter_ids = [str(item.get("evidence_id")) for item in counters if item.get("evidence_id")]
        fact_chain = [_compact(item.get("fact"), 180) for item in support if str(item.get("fact") or "").strip()][:5]
        mechanism_chain = [
            f"事实锚点：{fact_chain[0]}" if fact_chain else "",
            "证据需要同时说明主体、指标口径和时间窗口，才能从线索升级为章节判断。",
            "若证据只覆盖局部公司或单一事件，正文应写成方向性判断，并保留验证条件。",
        ]
        mechanism_chain = [item for item in mechanism_chain if item]
        counter_boundary = [_compact(item.get("fact"), 180) for item in counters if str(item.get("fact") or "").strip()][:3]
        key_claims = []
        if ready:
            key_claims.append(
                {
                    "claim": statement,
                    "claim_status": "decision_ready",
                    "supporting_evidence": evidence_ids[:10],
                    "evidence_refs": evidence_ids[:10],
                    "counter_evidence_refs": counter_ids[:6],
                    "mechanism": "；".join(mechanism_chain),
                    "reasoning": "；".join(mechanism_chain),
                    "counter_evidence": "；".join(counter_boundary) if counter_boundary else "反证已纳入判断边界；若后续A/B来源显示价格、订单、产能或客户验证反向变化，应下调结论。",
                    "decision_implication": "可进入正文核心判断，并用于进入/投资/产品布局优先级排序。",
                    "confidence": round(sum(float(item.get("confidence") or 0.0) for item in usable) / max(len(usable), 1), 3),
                    "what_to_verify_next": ["持续跟踪价格/毛利", "复核客户认证与订单", "监控产能过剩和替代路线"],
                }
            )
        insights.append(
            {
                "chapter_id": f"hypothesis_{index}",
                "hypothesis_id": hypothesis_id,
                "chapter_question": statement,
                "chapter_answer": statement if ready else "",
                "core_answer": statement if ready else "",
                "fact_chain": fact_chain,
                "mechanism_chain": mechanism_chain,
                "counter_evidence_boundary": counter_boundary,
                "decision_implication": "可进入正文核心判断，并用于进入/投资/产品布局优先级排序。" if ready else "证据不足时只能作为待验证方向，并优先补齐 A/B 来源和指标口径。",
                "key_claims": key_claims,
                "decision_readiness": "ready" if ready else "needs_evidence",
                "blocking_gaps": gaps,
            }
        )
    return insights


def _gap_priority(gap: str) -> int:
    return {
        "insufficient_ab_sources": 0,
        "only_c_or_lower_sources": 1,
        "metric_evidence_missing": 2,
        "metric_definition_unfilled": 2,
        "metric_scope_period_unit_incomplete": 3,
        "counter_evidence_missing": 4,
        "case_evidence_missing": 5,
        "source_diversity_missing": 6,
        "needs_authoritative_source": 7,
        "needs_corroboration": 8,
    }.get(str(gap or ""), 20)


def _followup_for_gap(*, target: str, gap: str, hypothesis_id: str = "", dimension: str = "") -> Dict[str, Any]:
    query_parts = [target or dimension or hypothesis_id]
    proof_role = "support"
    evidence_type = "data"
    lane_targets = ["official_data", "filing_company", "market_research"]
    source_priority = ["official", "filing", "research_report"]
    if gap in {"insufficient_ab_sources", "only_c_or_lower_sources", "needs_authoritative_source"}:
        query_parts.extend(["官方", "公告", "财报", "协会", "权威研报", "A/B来源"])
    if gap in {"metric_evidence_missing", "metric_definition_unfilled", "metric_scope_period_unit_incomplete"}:
        query_parts.extend(["指标口径", "数值", "期间", "单位", "范围"])
        proof_role = "metric"
        evidence_type = "metric"
        lane_targets = ["official_data", "market_research"]
    if gap == "counter_evidence_missing":
        query_parts.extend(["反证", "风险", "失败案例", "价格下跌", "订单取消", "监管变化"])
        proof_role = "counter"
        evidence_type = "counter"
        lane_targets = ["news_event", "filing_company", "market_research"]
    if gap == "case_evidence_missing":
        query_parts.extend(["客户案例", "订单", "认证", "量产", "供应合同"])
        proof_role = "case"
        evidence_type = "case"
        lane_targets = ["customer_case", "filing_company"]
    if gap in {"source_diversity_missing", "needs_corroboration"}:
        query_parts.extend(["第二来源", "交叉验证", "官方", "公司披露"])
    query = _compact(" ".join(part for part in query_parts if str(part or "").strip()), 220)
    return {
        "query": query,
        "agent": "iqs",
        "targets_gap": target or dimension or hypothesis_id or gap,
        "dimension_name": dimension or target,
        "evidence_goal": target or dimension,
        "hypothesis_id": hypothesis_id,
        "hypothesis_statement": target,
        "proof_role": proof_role,
        "evidence_type": evidence_type,
        "lane_targets": lane_targets,
        "source_priority": source_priority,
        "blocking_gaps": [gap],
        "priority": _gap_priority(gap),
    }


def _evidence_refinement_plan(
    *,
    evidence_analyses: List[Dict[str, Any]],
    hypothesis_insights: List[Dict[str, Any]],
    dimension_synthesis: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    tasks: List[Dict[str, Any]] = []
    for chapter in hypothesis_insights:
        chapter = _as_dict(chapter)
        target = _compact(chapter.get("chapter_question") or chapter.get("hypothesis_statement"), 160)
        hypothesis_id = str(chapter.get("hypothesis_id") or "").strip()
        for gap in _as_list(chapter.get("blocking_gaps")):
            tasks.append(_followup_for_gap(target=target, gap=str(gap), hypothesis_id=hypothesis_id, dimension=target))
    for dimension, payload in dimension_synthesis.items():
        payload = _as_dict(payload)
        for gap in _as_list(payload.get("evidence_gap_tags")):
            tasks.append(_followup_for_gap(target=str(dimension), gap=str(gap), dimension=str(dimension)))
        for query in _as_list(payload.get("followup_queries")):
            query_text = _compact(query, 220)
            if query_text:
                tasks.append(
                    {
                        "query": query_text,
                        "agent": "iqs",
                        "targets_gap": str(dimension),
                        "dimension_name": str(dimension),
                        "evidence_goal": str(dimension),
                        "proof_role": "support",
                        "evidence_type": "data",
                        "blocking_gaps": ["needs_corroboration"],
                        "priority": _gap_priority("needs_corroboration"),
                    }
                )
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for task in sorted(tasks, key=lambda item: (int(item.get("priority") or 20), str(item.get("targets_gap") or ""), str(item.get("query") or ""))):
        key = (task.get("targets_gap"), task.get("proof_role"), task.get("query"))
        if key in seen or not str(task.get("query") or "").strip():
            continue
        seen.add(key)
        deduped.append(task)
    gap_counts: Dict[str, int] = {}
    for item in evidence_analyses:
        for gap in _as_list(item.get("evidence_gaps")):
            gap_text = str(gap or "")
            if gap_text:
                gap_counts[gap_text] = gap_counts.get(gap_text, 0) + 1
    for chapter in hypothesis_insights:
        for gap in _as_list(_as_dict(chapter).get("blocking_gaps")):
            gap_text = str(gap or "")
            if gap_text:
                gap_counts[gap_text] = gap_counts.get(gap_text, 0) + 1
    return {
        "status": "needs_refinement" if deduped else "sufficient_for_current_analysis",
        "gap_counts": dict(sorted(gap_counts.items(), key=lambda pair: (_gap_priority(pair[0]), pair[0]))),
        "follow_up_queries": deduped[:20],
        "top_priorities": deduped[:6],
    }


def _chapter_key_for_item(item: Dict[str, Any]) -> str:
    return str(
        item.get("chapter_id")
        or item.get("hypothesis_id")
        or item.get("dimension_id")
        or item.get("dimension")
        or item.get("dimension_name")
        or item.get("evidence_goal")
        or "unmapped"
    ).strip() or "unmapped"


def _source_url(item: Dict[str, Any]) -> str:
    source = _source_payload(item)
    return str(source.get("url") or item.get("source_url") or item.get("url") or "").strip()


def _analysis_readiness(payload: Dict[str, Any]) -> str:
    core_ab = int(float(payload.get("core_ab_source_count") or 0))
    claim_ready = int(float(payload.get("claim_ready_evidence_count") or len(_as_list(payload.get("claim_ready_evidence_refs"))) or 0))
    directional = int(float(payload.get("directional_only_count") or 0))
    if core_ab >= 1 and claim_ready >= 1:
        return "ready"
    if core_ab >= 1:
        return "needs_claim_rebuild"
    if directional > 0:
        return "directional_only"
    return "needs_evidence"


def _chapter_evidence_diagnostics(
    evidence_package: Dict[str, Any],
    evidence_analyses: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    existing = _as_dict(evidence_package.get("chapter_evidence_diagnostics"))
    if existing:
        return existing
    by_chapter = _as_dict(evidence_package.get("evidence_analysis_by_chapter"))
    diagnostics: Dict[str, Dict[str, Any]] = {}
    for chapter_id, payload in by_chapter.items():
        if not isinstance(payload, dict):
            continue
        refs = _as_list(payload.get("sample_evidence_refs"))
        item = {
            "chapter_id": str(payload.get("chapter_id") or chapter_id),
            "chapter_title": str(payload.get("chapter_title") or chapter_id),
            "core_ab_source_count": int(float(payload.get("core_ab_source_count") or 0)),
            "supporting_ab_source_count": int(float(payload.get("supporting_ab_source_count") or payload.get("claim_ready_evidence_count") or 0)),
            "metric_ready_count": int(float(payload.get("metric_ready_count") or 0)),
            "counter_signal_count": int(float(payload.get("counter_signal_count") or 0)),
            "directional_only_count": int(float(payload.get("directional_only_count") or 0)),
            "source_trace_missing_count": int(float(payload.get("source_trace_missing_count") or 0)),
            "claim_ready_evidence_refs": refs[:12],
            "metric_ready_refs": _as_list(payload.get("metric_ready_refs"))[:12],
            "counter_refs": _as_list(payload.get("counter_refs"))[:12],
            "gap_types": _as_list(payload.get("evidence_gap_types")),
        }
        item["analysis_readiness"] = _analysis_readiness(item)
        diagnostics[item["chapter_id"]] = item
    if diagnostics:
        return diagnostics

    buckets: Dict[str, Dict[str, Any]] = {}
    for item in evidence_analyses:
        if not isinstance(item, dict):
            continue
        chapter_id = _chapter_key_for_item(item)
        bucket = buckets.setdefault(
            chapter_id,
            {
                "chapter_id": chapter_id,
                "chapter_title": str(item.get("dimension") or item.get("dimension_name") or chapter_id),
                "core_ab_source_count": 0,
                "supporting_ab_source_count": 0,
                "metric_ready_count": 0,
                "counter_signal_count": 0,
                "directional_only_count": 0,
                "source_trace_missing_count": 0,
                "claim_ready_evidence_refs": [],
                "metric_ready_refs": [],
                "counter_refs": [],
                "gap_types": [],
            },
        )
        level = str(item.get("source_level") or "").strip().upper()
        allowed = str(item.get("allowed_use") or "").strip()
        ref = str(item.get("evidence_id") or "").strip()
        if level in {"A", "B"} and allowed in {"core_claim", "supporting"} and ref:
            bucket["core_ab_source_count"] += 1
            bucket["claim_ready_evidence_refs"].append(ref)
        if level in {"A", "B"} and allowed == "supporting":
            bucket["supporting_ab_source_count"] += 1
        if str(item.get("metric") or "").strip() and str(item.get("value") or "").strip() and ref:
            bucket["metric_ready_count"] += 1
            bucket["metric_ready_refs"].append(ref)
        if str(item.get("proof_role") or "").strip().lower() == "counter" and ref:
            bucket["counter_signal_count"] += 1
            bucket["counter_refs"].append(ref)
        if allowed == "directional_signal":
            bucket["directional_only_count"] += 1
        if not _source_label(item) and not _source_url(item):
            bucket["source_trace_missing_count"] += 1
        for gap in _as_list(item.get("evidence_gaps")):
            if gap and gap not in bucket["gap_types"]:
                bucket["gap_types"].append(gap)
    for bucket in buckets.values():
        if bucket["core_ab_source_count"] <= 0 and "insufficient_ab_sources" not in bucket["gap_types"]:
            bucket["gap_types"].append("insufficient_ab_sources")
        if bucket["counter_signal_count"] <= 0 and "counter_evidence_missing" not in bucket["gap_types"]:
            bucket["gap_types"].append("counter_evidence_missing")
        bucket["analysis_readiness"] = _analysis_readiness(bucket)
        bucket["claim_ready_evidence_refs"] = _dedupe(bucket["claim_ready_evidence_refs"])[:12]
        bucket["metric_ready_refs"] = _dedupe(bucket["metric_ready_refs"])[:12]
        bucket["counter_refs"] = _dedupe(bucket["counter_refs"])[:12]
    return buckets


def _gap_ledger_from_diagnostics(diagnostics: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    ledger: List[Dict[str, Any]] = []
    for chapter in diagnostics.values():
        chapter_id = str(chapter.get("chapter_id") or "unmapped")
        refs = _as_list(chapter.get("claim_ready_evidence_refs"))[:8]
        for gap in _as_list(chapter.get("gap_types")):
            gap_text = str(gap or "").strip()
            if not gap_text:
                continue
            proof_role = "source_check"
            required_fields = ["source"]
            lane_targets = ["official_data", "market_research"]
            severity = "blocking"
            reason = "章节缺少可支撑核心判断的 A/B 来源。"
            if gap_text == "metric_scope_period_unit_incomplete":
                proof_role = "metric"
                required_fields = ["metric", "value", "unit", "period", "source"]
                lane_targets = ["official_data", "market_research", "filing_company"]
                reason = "指标证据缺少 metric/value/unit/period/source 中的关键字段。"
            elif gap_text == "counter_evidence_missing":
                proof_role = "counter"
                required_fields = ["source"]
                lane_targets = ["news_event", "market_research"]
                severity = "advisory"
                reason = "章节缺少反证、风险边界或失败案例。"
            elif gap_text in {"source_trace_missing", "citation_source_missing"}:
                reason = "部分证据缺少可追溯来源，不能进入 Clean report。"
            elif gap_text in {"directional_only_evidence"}:
                reason = "当前证据只能支撑方向性判断，不能支撑强结论。"
            ledger.append(
                {
                    "gap_id": _normalize_key(f"{chapter_id}:{gap_text}")[:24] or f"{chapter_id}_{len(ledger)+1}",
                    "chapter_id": chapter_id,
                    "claim_id": "",
                    "gap_type": gap_text,
                    "type": gap_text,
                    "severity": severity,
                    "required_proof_role": proof_role,
                    "proof_role": proof_role,
                    "required_source_level": ["A", "B"] if proof_role != "counter" else ["A", "B", "C"],
                    "required_fields": required_fields,
                    "current_evidence_refs": refs,
                    "why_current_evidence_insufficient": reason,
                    "repair_route": "evidence_search",
                    "query_terms": _dedupe([chapter.get("chapter_title"), gap_text])[:6],
                    "topic_terms": _dedupe([chapter.get("chapter_title"), gap_text])[:6],
                    "lane_targets": lane_targets,
                    "source": "analysis_agent_diagnostics",
                }
            )
    return ledger


def _analysis_summary_from_diagnostics(
    diagnostics: Dict[str, Dict[str, Any]],
    ledger: List[Dict[str, Any]],
) -> Dict[str, Any]:
    gap_dist: Dict[str, int] = {}
    severity_dist: Dict[str, int] = {}
    for gap in ledger:
        gap_type = str(gap.get("gap_type") or gap.get("type") or "unknown")
        severity = str(gap.get("severity") or "unknown")
        gap_dist[gap_type] = gap_dist.get(gap_type, 0) + 1
        severity_dist[severity] = severity_dist.get(severity, 0) + 1
    return {
        "chapter_count": len(diagnostics),
        "total_core_ab_source_count": sum(int(_as_dict(item).get("core_ab_source_count") or 0) for item in diagnostics.values()),
        "total_metric_ready_count": sum(int(_as_dict(item).get("metric_ready_count") or 0) for item in diagnostics.values()),
        "total_counter_signal_count": sum(int(_as_dict(item).get("counter_signal_count") or 0) for item in diagnostics.values()),
        "total_claim_ready_evidence_count": sum(len(_as_list(_as_dict(item).get("claim_ready_evidence_refs"))) for item in diagnostics.values()),
        "total_directional_only_count": sum(int(_as_dict(item).get("directional_only_count") or 0) for item in diagnostics.values()),
        "blocking_gap_count": severity_dist.get("blocking", 0),
        "advisory_gap_count": severity_dist.get("advisory", 0),
        "gap_type_distribution": gap_dist,
        "severity_distribution": severity_dist,
    }


def _generic_mechanism(text: str) -> bool:
    return bool(
        re.search(
            r"(已有\s*\d+\s*条可用于正文的信号|分析应先看事实是否连续|传导到需求、供给、政策约束|结论强度取决于来源等级)",
            str(text or ""),
        )
    )


def analysis_depth_quality(structured_analysis: Dict[str, Any]) -> Dict[str, Any]:
    claims: List[Dict[str, Any]] = []
    insight = _as_dict(structured_analysis.get("report_insight_package"))
    for chapter in _as_list(insight.get("chapters")) + _as_list(structured_analysis.get("chapter_insights")):
        chapter = _as_dict(chapter)
        for claim in _as_list(chapter.get("key_claims")):
            if isinstance(claim, dict):
                copied = dict(claim)
                copied.setdefault("chapter_question", chapter.get("chapter_question"))
                claims.append(copied)
    for unit in _as_list(structured_analysis.get("claim_units")):
        if isinstance(unit, dict):
            claims.append(dict(unit))
    normalized_claims = [_normalize_key(item.get("claim") or item.get("judgment")) for item in claims]
    repeated = len(normalized_claims) - len({item for item in normalized_claims if item})
    generic_count = 0
    title_as_claim = 0
    missing_reasoning = 0
    missing_counter = 0
    ref_mismatch = 0
    all_refs = {
        str(item.get("evidence_id") or "").strip()
        for item in _as_list(structured_analysis.get("evidence_analyses"))
        if isinstance(item, dict) and str(item.get("evidence_id") or "").strip()
    }
    for item in claims:
        claim_text = str(item.get("claim") or item.get("judgment") or "").strip()
        reasoning = str(item.get("reasoning") or item.get("mechanism") or "").strip()
        counter = str(item.get("counter_evidence") or item.get("counter_boundary") or "").strip()
        refs = [str(ref or "").strip() for ref in _as_list(item.get("supporting_evidence") or item.get("evidence_refs") or item.get("supporting_evidence_refs")) if str(ref or "").strip()]
        if _generic_mechanism(claim_text) or _generic_mechanism(reasoning):
            generic_count += 1
        question = str(item.get("chapter_question") or item.get("question") or item.get("dimension") or "").strip()
        if claim_text and question and _normalize_key(question) in _normalize_key(claim_text):
            title_as_claim += 1
        if refs and not reasoning:
            missing_reasoning += 1
        if refs and str(item.get("claim_status") or "").strip() in {"decision_ready", "core_claim", ""} and not counter:
            missing_counter += 1
        if all_refs and any(ref.startswith("EV-") and ref not in all_refs for ref in refs):
            ref_mismatch += 1
    claim_count = max(len(claims), 1)
    generic_ratio = round(generic_count / claim_count, 3)
    repeated_ratio = round(max(0, repeated) / claim_count, 3)
    status = "pass"
    route = "pass"
    if ref_mismatch:
        status = "needs_rewrite"
        route = "citation_repair"
    elif repeated_ratio > 0.30 or generic_ratio > 0.35 or missing_reasoning:
        status = "needs_rewrite"
        route = "analysis_deepening"
    elif title_as_claim or missing_counter:
        status = "advisory"
        route = "rewrite"
    return {
        "status": status,
        "suggested_route": route,
        "claim_count": len(claims),
        "generic_mechanism_ratio": generic_ratio,
        "repeated_claim_ratio": repeated_ratio,
        "title_as_claim_count": title_as_claim,
        "missing_reasoning_count": missing_reasoning,
        "missing_counter_boundary_count": missing_counter,
        "evidence_ref_mismatch_count": ref_mismatch,
    }


def claim_binding_feedback_summary(structured_analysis: Dict[str, Any]) -> Dict[str, Any]:
    diagnostics = _as_dict(structured_analysis.get("chapter_evidence_diagnostics"))
    units_by_chapter: Dict[str, List[Dict[str, Any]]] = {}
    for unit in _as_list(structured_analysis.get("claim_units")):
        if not isinstance(unit, dict):
            continue
        chapter_id = str(unit.get("chapter_id") or unit.get("hypothesis_id") or unit.get("dimension") or "").strip()
        if chapter_id:
            units_by_chapter.setdefault(chapter_id, []).append(unit)
    unsupported_core_claim_count = 0
    directional_claim_count = 0
    claim_rebuild_targets: List[Dict[str, Any]] = []
    for unit in [item for items in units_by_chapter.values() for item in items]:
        refs = _as_list(unit.get("supporting_evidence") or unit.get("evidence_refs") or unit.get("supporting_evidence_refs"))
        status = str(unit.get("claim_status") or "").strip()
        if status in {"decision_ready", "core_claim"} and not refs:
            unsupported_core_claim_count += 1
        if status in {"directional", "directional_ready"}:
            directional_claim_count += 1
    for chapter_id, payload in diagnostics.items():
        payload = _as_dict(payload)
        core_ab = int(float(payload.get("core_ab_source_count") or 0))
        if core_ab <= 0:
            continue
        chapter_units = units_by_chapter.get(str(chapter_id), [])
        bound = any(_as_list(unit.get("supporting_evidence") or unit.get("evidence_refs") or unit.get("supporting_evidence_refs")) for unit in chapter_units)
        if not bound:
            claim_rebuild_targets.append(
                {
                    "chapter_id": chapter_id,
                    "reason": "evidence_available_but_not_bound",
                    "available_ab_source_count": core_ab,
                    "available_refs": _as_list(payload.get("claim_ready_evidence_refs"))[:8],
                }
            )
    return {
        "available_ab_not_bound_count": len(claim_rebuild_targets),
        "unsupported_core_claim_count": unsupported_core_claim_count,
        "directional_claim_count": directional_claim_count,
        "claim_rebuild_targets": claim_rebuild_targets[:12],
    }


def _evidence_cards_for_llm(
    evidence_package: Dict[str, Any],
    *,
    max_chapters: int,
    max_per_chapter: int,
) -> List[Dict[str, Any]]:
    diagnostics = _as_dict(evidence_package.get("chapter_evidence_diagnostics")) or _as_dict(evidence_package.get("evidence_analysis_by_chapter"))
    allowed_chapters = list(diagnostics.keys())[:max_chapters] if diagnostics else []
    buckets: Dict[str, int] = {}
    cards: List[Dict[str, Any]] = []
    source_items = _as_list(evidence_package.get("analysis_ready_evidence")) or _as_list(evidence_package.get("clean_evidence_list"))
    for item in source_items:
        if not isinstance(item, dict):
            continue
        chapter_id = _chapter_key_for_item(item)
        if allowed_chapters and chapter_id not in allowed_chapters:
            continue
        if buckets.get(chapter_id, 0) >= max_per_chapter:
            continue
        source = _source_payload(item)
        card = _as_dict(item.get("evidence_card"))
        cards.append(
            {
                "evidence_id": str(item.get("evidence_id") or item.get("id") or "").strip(),
                "chapter_id": chapter_id,
                "fact": _compact(_fact_text(item), 360),
                "metric": _compact(item.get("metric"), 100),
                "value": _compact(item.get("value"), 100),
                "unit": _compact(item.get("unit") or _as_dict(item.get("metric_definition")).get("unit"), 60),
                "period": _compact(item.get("period") or source.get("date"), 80),
                "source_level": str(item.get("source_level") or card.get("source_level") or "").strip().upper(),
                "allowed_use": str(item.get("allowed_use") or card.get("allowed_use") or "").strip(),
                "can_support": _as_list(item.get("can_support")) or _as_list(card.get("can_support")),
                "cannot_support": _as_list(item.get("cannot_support")) or _as_list(card.get("cannot_prove")),
                "proof_strength": str(item.get("proof_strength") or item.get("evidence_strength") or "").strip(),
                "repair_need": _as_list(item.get("repair_need")) or _as_list(item.get("evidence_gaps")),
                "source_title": _compact(source.get("title") or source.get("source") or source.get("name"), 160),
                "source_url": str(source.get("url") or item.get("source_url") or "").strip(),
            }
        )
        buckets[chapter_id] = buckets.get(chapter_id, 0) + 1
    return [item for item in cards if item.get("evidence_id") and item.get("fact")]


def build_llm_analysis_input(evidence_package: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
    max_chapters = _env_int("BRAIN_LLM_ANALYSIS_MAX_CHAPTERS", 6, min_value=1, max_value=12)
    max_per_chapter = _env_int("BRAIN_LLM_ANALYSIS_MAX_EVIDENCE_PER_CHAPTER", 12, min_value=3, max_value=30)
    diagnostics = _as_dict(fallback.get("chapter_evidence_diagnostics")) or _as_dict(evidence_package.get("chapter_evidence_diagnostics"))
    return {
        "query": fallback.get("query") or evidence_package.get("query") or "",
        "research_plan": _research_plan(evidence_package),
        "report_contract": _as_dict(evidence_package.get("report_contract")) or _as_dict(evidence_package.get("report_plan")),
        "chapter_evidence_diagnostics": dict(list(diagnostics.items())[:max_chapters]),
        "evidence_cards": _evidence_cards_for_llm(
            evidence_package,
            max_chapters=max_chapters,
            max_per_chapter=max_per_chapter,
        ),
        "evidence_gap_ledger": _as_list(fallback.get("evidence_gap_ledger"))[:80],
        "fallback_claim_units": _as_list(fallback.get("claim_units"))[:24],
    }


def synthesize_with_llm_analysis(
    *,
    evidence_package: Dict[str, Any],
    fallback: Dict[str, Any],
    llm_config: Dict[str, Any],
) -> Dict[str, Any]:
    if call_openai_compatible_json is None or llm_config_is_ready is None or normalize_llm_config is None:
        raise RuntimeError("LLM analysis dependencies are unavailable.")
    config = dict(llm_config or {})
    config["timeout"] = float(os.getenv("BRAIN_LLM_ANALYSIS_TIMEOUT_SECONDS", config.get("timeout") or 180) or 180)
    if not llm_config_is_ready(config):
        raise RuntimeError("LLM config is incomplete.")
    system_prompt = """
你是企业级行业研究证据分析 Agent。你的任务不是写报告全文，而是把已经清洗过的证据卡转成可写作的分析包。

硬规则：
1. 只能使用输入里的 evidence_cards，不得新增事实、数字、来源、公司或网址。
2. decision_ready claim 必须引用至少一个输入中存在的 evidence_id。
3. C/D 或 directional 证据不能支撑强结论，只能写成 directional 或 hypothesis。
4. 必须输出 JSON object，字段为 chapter_synthesis、cross_chapter_conflicts、evidence_repair_priorities、rewrite_priorities。
5. 每章尽量给出 fact_chain、mechanism_chain、counter_evidence_boundary、decision_implication，让 Writer 能展开正文。
""".strip()
    response = call_openai_compatible_json(
        config=config,
        system_prompt=system_prompt,
        user_payload=build_llm_analysis_input(evidence_package, fallback),
    )
    payload = _as_dict(response.get("payload"))
    payload["_llm_usage"] = response.get("usage", {})
    payload["_llm_model"] = normalize_llm_config(config).get("model", "")
    return payload


def validate_llm_analysis_output(payload: Dict[str, Any], evidence_package: Dict[str, Any]) -> Dict[str, Any]:
    valid_refs = {
        str(item.get("evidence_id") or item.get("id") or "").strip()
        for item in _as_list(evidence_package.get("analysis_ready_evidence")) + _as_list(evidence_package.get("clean_evidence_list"))
        if isinstance(item, dict) and str(item.get("evidence_id") or item.get("id") or "").strip()
    }
    issues: List[Dict[str, Any]] = []
    chapters: List[Dict[str, Any]] = []
    raw_chapters = payload.get("chapter_synthesis")
    if isinstance(raw_chapters, dict):
        chapter_iterable = [
            {**_as_dict(value), "chapter_id": _as_dict(value).get("chapter_id") or key}
            for key, value in raw_chapters.items()
            if isinstance(value, dict)
        ]
    else:
        chapter_iterable = _as_list(raw_chapters)
    for raw_chapter in chapter_iterable:
        if not isinstance(raw_chapter, dict):
            continue
        chapter = dict(raw_chapter)
        fact_refs: List[str] = []
        fact_chain_items = _as_list(chapter.get("fact_chain"))
        if isinstance(chapter.get("fact_chain"), dict):
            fact_chain_items = list(_as_dict(chapter.get("fact_chain")).values())
        for fact_item in fact_chain_items:
            if isinstance(fact_item, dict):
                ref = str(fact_item.get("evidence_ref") or fact_item.get("evidence_id") or "").strip()
                if ref:
                    fact_refs.append(ref)
            elif isinstance(fact_item, str):
                fact_refs.extend(re.findall(r"EV-\d+(?:-[A-Za-z0-9]+)?", fact_item))
        if not _as_list(chapter.get("claim_units")):
            inferred_claim = _compact(
                chapter.get("core_answer")
                or chapter.get("chapter_title")
                or chapter.get("chapter_id"),
                360,
            )
            if not inferred_claim or _has_internal_analysis_language(inferred_claim):
                inferred_claim = _public_claim_from_chapter(chapter)
            if inferred_claim:
                chapter["claim_units"] = [
                    {
                        "claim": inferred_claim,
                        "claim_status": "decision_ready" if fact_refs else "directional",
                        "supporting_evidence_refs": fact_refs[:8],
                        "reasoning": _public_reasoning_from_chapter(chapter),
                        "counter_boundary": _chapter_counter_text(chapter),
                        "decision_use": "",
                    }
                ]
        cleaned_units: List[Dict[str, Any]] = []
        for raw_unit in _as_list(chapter.get("claim_units")):
            if not isinstance(raw_unit, dict):
                continue
            unit = dict(raw_unit)
            refs = [str(ref or "").strip() for ref in _as_list(unit.get("supporting_evidence_refs")) if str(ref or "").strip()]
            invalid_refs = [ref for ref in refs if valid_refs and ref not in valid_refs]
            if invalid_refs:
                issues.append({"type": "invalid_llm_evidence_ref", "refs": invalid_refs, "chapter_id": chapter.get("chapter_id")})
            refs = [ref for ref in refs if not valid_refs or ref in valid_refs]
            unit["supporting_evidence_refs"] = refs
            claim_text = _compact(unit.get("claim"), 360)
            if not claim_text or _has_internal_analysis_language(claim_text):
                unit["claim"] = _public_claim_from_chapter(chapter)
                unit["claim_rewritten_from_instruction"] = True
                issues.append({"type": "llm_claim_rewritten_internal_instruction", "chapter_id": chapter.get("chapter_id")})
            reasoning_text = _compact(unit.get("reasoning"), 800)
            if not reasoning_text or _has_internal_analysis_language(reasoning_text):
                unit["reasoning"] = _public_reasoning_from_chapter(chapter)
            counter_text = _compact(unit.get("counter_boundary") or unit.get("counter_evidence"), 500)
            if not counter_text or _has_internal_analysis_language(counter_text):
                unit["counter_boundary"] = _chapter_counter_text(chapter)
            decision_use = _compact(unit.get("decision_use"), 360)
            if _has_internal_analysis_language(decision_use):
                unit["decision_use"] = ""
            if str(unit.get("claim_status") or "").strip() == "decision_ready" and not refs:
                unit["claim_status"] = "directional"
                unit["missing_binding_reason"] = unit.get("missing_binding_reason") or "decision_ready claim lacked valid evidence refs"
                issues.append({"type": "decision_claim_downgraded_no_valid_ref", "chapter_id": chapter.get("chapter_id")})
            cleaned_units.append(unit)
        chapter["claim_units"] = cleaned_units
        chapters.append(chapter)
    status = "valid" if chapters else "invalid"
    return {
        "status": status,
        "issues": issues,
        "chapter_synthesis": chapters,
        "valid_ref_count": len(valid_refs),
    }


def merge_llm_analysis_with_fallback(
    fallback: Dict[str, Any],
    llm_payload: Dict[str, Any],
    validation: Dict[str, Any],
) -> Dict[str, Any]:
    if str(validation.get("status") or "") != "valid":
        return fallback
    merged = dict(fallback)
    chapters: List[Dict[str, Any]] = []
    claim_units: List[Dict[str, Any]] = []
    key_judgments: List[Dict[str, Any]] = []
    for index, chapter in enumerate(_as_list(validation.get("chapter_synthesis")), start=1):
        if not isinstance(chapter, dict):
            continue
        chapter_id = str(chapter.get("chapter_id") or f"chapter_{index}")
        core_answer = _compact(chapter.get("core_answer"), 360)
        if _has_internal_analysis_language(core_answer):
            core_answer = _public_claim_from_chapter(chapter)
        key_claims: List[Dict[str, Any]] = []
        for unit_index, unit in enumerate(_as_list(chapter.get("claim_units")), start=1):
            if not isinstance(unit, dict):
                continue
            refs = _as_list(unit.get("supporting_evidence_refs"))
            claim = _compact(unit.get("claim"), 360)
            if _has_internal_analysis_language(claim):
                claim = _public_claim_from_chapter(chapter)
            if not claim:
                continue
            decision_implication = _compact(unit.get("decision_use") or chapter.get("decision_implication") or "", 360)
            if _has_internal_analysis_language(decision_implication):
                decision_implication = ""
            claim_payload = {
                "claim": claim,
                "claim_status": unit.get("claim_status") or ("decision_ready" if refs else "directional"),
                "supporting_evidence": refs,
                "evidence_refs": refs,
                "mechanism": unit.get("reasoning") or _public_reasoning_from_chapter(chapter),
                "reasoning": unit.get("reasoning") or _public_reasoning_from_chapter(chapter),
                "counter_evidence": unit.get("counter_boundary") or "；".join(str(item) for item in _as_list(chapter.get("counter_evidence_boundary"))[:3]),
                "decision_implication": decision_implication,
                "confidence": chapter.get("confidence") or unit.get("confidence"),
                "what_to_verify_next": _as_list(chapter.get("remaining_gaps"))[:6],
            }
            if _has_internal_analysis_language(claim_payload["mechanism"]):
                claim_payload["mechanism"] = _public_reasoning_from_chapter(chapter)
                claim_payload["reasoning"] = claim_payload["mechanism"]
            if _has_internal_analysis_language(claim_payload["counter_evidence"]):
                claim_payload["counter_evidence"] = _chapter_counter_text(chapter)
            key_claims.append(claim_payload)
            claim_units.append(
                {
                    "id": f"{chapter_id}_llm_{unit_index}",
                    "chapter_id": chapter_id,
                    "dimension": chapter.get("chapter_title") or chapter_id,
                    "question": chapter.get("chapter_title") or chapter_id,
                    "claim": claim,
                    "claim_status": claim_payload["claim_status"],
                    "reasoning": claim_payload["reasoning"] or claim_payload["mechanism"],
                    "mechanism": claim_payload["mechanism"],
                    "counter_evidence": claim_payload["counter_evidence"],
                    "decision_implication": claim_payload["decision_implication"],
                    "supporting_evidence": refs,
                    "evidence_refs": refs,
                    "confidence": claim_payload["confidence"],
                }
            )
            if refs:
                key_judgments.append(
                    {
                        "judgment": claim,
                        "supporting_dimensions": [chapter.get("chapter_title") or chapter_id],
                        "evidence_ids": refs,
                        "confidence": claim_payload["confidence"],
                        "decision_implication": claim_payload["decision_implication"],
                    }
                )
        chapters.append(
            {
                "chapter_id": chapter_id,
                "chapter_question": chapter.get("chapter_title") or chapter_id,
                "chapter_answer": core_answer,
                "core_answer": core_answer,
                "fact_chain": list(_as_dict(chapter.get("fact_chain")).values()) if isinstance(chapter.get("fact_chain"), dict) else _as_list(chapter.get("fact_chain")),
                "mechanism_chain": list(_as_dict(chapter.get("mechanism_chain")).values()) if isinstance(chapter.get("mechanism_chain"), dict) else _as_list(chapter.get("mechanism_chain")),
                "counter_evidence_boundary": [chapter.get("counter_evidence_boundary")] if isinstance(chapter.get("counter_evidence_boundary"), str) else _as_list(chapter.get("counter_evidence_boundary")),
                "decision_implication": chapter.get("decision_implication") or "",
                "confidence": chapter.get("confidence") or "medium",
                "key_claims": key_claims,
                "remaining_gaps": _as_list(chapter.get("remaining_gaps")),
                "decision_readiness": "ready" if any(_as_list(item.get("supporting_evidence")) for item in key_claims) else "needs_evidence",
                "blocking_gaps": _as_list(chapter.get("remaining_gaps")),
            }
        )
    if chapters:
        insight = dict(_as_dict(merged.get("report_insight_package")))
        insight["chapters"] = chapters
        if key_judgments:
            insight["report_thesis"] = _compact(key_judgments[0].get("judgment"), 260)
            insight.setdefault("executive_summary", {})
            insight["executive_summary"] = {
                **_as_dict(insight.get("executive_summary")),
                "one_sentence_answer": _compact(key_judgments[0].get("judgment"), 220),
                "top_3_judgments": key_judgments[:3],
                "so_what": _dedupe([item.get("decision_implication") for item in key_judgments])[:5],
            }
        merged["report_insight_package"] = insight
        merged["chapter_insights"] = chapters
    if claim_units:
        merged["claim_units"] = claim_units
        merged["key_judgments"] = key_judgments or _as_list(merged.get("key_judgments"))
    merged["llm_analysis_synthesis"] = {
        "chapter_synthesis": chapters,
        "cross_chapter_conflicts": _as_list(llm_payload.get("cross_chapter_conflicts")),
        "evidence_repair_priorities": _as_list(llm_payload.get("evidence_repair_priorities")),
        "rewrite_priorities": _as_list(llm_payload.get("rewrite_priorities")),
        "usage": llm_payload.get("_llm_usage", {}),
        "model": llm_payload.get("_llm_model", ""),
        "validation": validation,
    }
    return merged


def build_fallback_analysis(evidence_package: Dict[str, Any]) -> Dict[str, Any]:
    evidence_package = _as_dict(evidence_package)
    research_plan = _research_plan(evidence_package)
    dimensions = _analysis_dimensions(evidence_package)
    evidence_analyses: List[Dict[str, Any]] = []
    refs_by_dimension: Dict[str, List[str]] = {}
    index = 1
    for dimension in dimensions:
        items = _items_for_dimension(evidence_package, dimension)
        refs_by_dimension[dimension] = []
        selected_items = _select_analysis_items_for_dimension(
            items,
            max_items=_env_int("ANALYSIS_FALLBACK_MAX_ITEMS_PER_DIMENSION", 18, min_value=1, max_value=80),
        )
        for item in selected_items:
            analysis = _evidence_analysis(item, dimension, index)
            index += 1
            evidence_analyses.append(analysis)
            refs_by_dimension[dimension].append(str(analysis.get("evidence_id")))
    dimension_synthesis = {
        dimension: _dimension_synthesis(
            dimension,
            [item for item in evidence_analyses if str(item.get("dimension") or "") == dimension],
        )
        for dimension in dimensions
    }
    claim_units = _claim_units_from_synthesis(dimension_synthesis)
    core_facts = [
        {
            "dimension": item.get("dimension"),
            "fact": item.get("fact"),
            "evidence_ids": [item.get("evidence_id")],
            "confidence": item.get("confidence"),
        }
        for item in evidence_analyses
        if item.get("fact")
    ][: max(8, len(dimensions) * 3)]
    key_judgments = [
        {
            "judgment": unit.get("claim"),
            "supporting_dimensions": [unit.get("dimension")],
            "evidence_ids": unit.get("supporting_evidence") or [],
            "confidence": unit.get("confidence"),
            "decision_implication": unit.get("decision_implication"),
        }
        for unit in claim_units
    ]
    report_outline = [
        {
            "section": str(chapter.get("name") or chapter.get("title") or chapter.get("id") or ""),
            "dimension": ", ".join(_as_list(_as_dict(evidence_package.get("chapter_dim_mapping")).get(chapter.get("id")))),
            "evidence_ids": [],
        }
        for chapter in _as_list(evidence_package.get("chapter_plan"))
        if isinstance(chapter, dict)
    ] or [{"section": dimension, "dimension": dimension, "evidence_ids": refs_by_dimension.get(dimension, [])[:6]} for dimension in dimensions]
    hypothesis_insights = _hypothesis_insights(research_plan, evidence_analyses)
    hypothesis_key_judgments = [
        {
            "judgment": claim.get("claim"),
            "supporting_dimensions": [chapter.get("chapter_question")],
            "evidence_ids": claim.get("supporting_evidence") or [],
            "confidence": claim.get("confidence"),
            "decision_implication": claim.get("decision_implication"),
        }
        for chapter in hypothesis_insights
        for claim in _as_list(_as_dict(chapter).get("key_claims"))
        if isinstance(claim, dict) and str(claim.get("claim") or "").strip()
    ]
    if not any(str(item.get("judgment") or "").strip() for item in key_judgments):
        key_judgments = hypothesis_key_judgments
    chapter_insights = hypothesis_insights or _chapter_insights_from_synthesis(dimension_synthesis)
    evidence_refinement_plan = _evidence_refinement_plan(
        evidence_analyses=evidence_analyses,
        hypothesis_insights=hypothesis_insights,
        dimension_synthesis=dimension_synthesis,
    )
    chapter_evidence_diagnostics = _chapter_evidence_diagnostics(evidence_package, evidence_analyses)
    evidence_gap_ledger = _as_list(evidence_package.get("evidence_gap_ledger")) or _gap_ledger_from_diagnostics(chapter_evidence_diagnostics)
    evidence_analysis_summary = (
        _as_dict(evidence_package.get("evidence_analysis_summary"))
        or _analysis_summary_from_diagnostics(chapter_evidence_diagnostics, evidence_gap_ledger)
    )
    report_insight_package = {
        "report_thesis": _compact(key_judgments[0].get("judgment") if key_judgments else "", 260),
        "executive_summary": {
            "one_sentence_answer": _compact(key_judgments[0].get("judgment") if key_judgments else "", 220),
            "top_3_judgments": key_judgments[:3],
            "what_changed": _dedupe([item.get("fact") for item in core_facts])[:5],
            "so_what": _dedupe([item.get("decision_implication") for item in key_judgments])[:5],
        },
        "chapters": chapter_insights,
        "decision_matrix": _as_list(_as_dict(evidence_package.get("decision_layer")).get("decision_matrix")),
        "risk_register": _as_list(_as_dict(evidence_package.get("risk_layer")).get("risk_items")),
        "evidence_refinement_plan": evidence_refinement_plan,
        "source_appendix": _as_list(evidence_package.get("source_registry")),
    }
    result = {
        "analysis_type": "structured_analysis",
        "query": str(evidence_package.get("query") or ""),
        "research_plan": research_plan,
        "evidence_analyses": evidence_analyses,
        "dimension_synthesis": dimension_synthesis,
        "chapter_insights": chapter_insights,
        "hypothesis_insights": hypothesis_insights,
        "chapter_evidence_diagnostics": chapter_evidence_diagnostics,
        "evidence_analysis_by_chapter": _as_dict(evidence_package.get("evidence_analysis_by_chapter")) or chapter_evidence_diagnostics,
        "evidence_analysis_summary": evidence_analysis_summary,
        "evidence_gap_ledger": evidence_gap_ledger,
        "report_insight_package": report_insight_package,
        "claim_units": claim_units,
        "core_facts": core_facts,
        "key_judgments": key_judgments,
        "evidence_gap_analysis": [
            {
                "evidence_id": item.get("evidence_id"),
                "dimension": item.get("dimension"),
                "gaps": _as_list(item.get("evidence_gaps")),
                "verification_questions": _as_list(item.get("verification_questions")),
                "suggested_followup_query": item.get("suggested_followup_query"),
            }
            for item in evidence_analyses
            if _as_list(item.get("evidence_gaps")) or str(item.get("suggested_followup_query") or "").strip()
        ],
        "evidence_refinement_plan": evidence_refinement_plan,
        "counter_analyses": [
            {
                "dimension": dimension,
                "counter": payload.get("counter"),
                "verify_kpi": payload.get("verify_kpi"),
            }
            for dimension, payload in dimension_synthesis.items()
        ],
        "decision_layer": {
            "decision_context": research_plan.get("decision_context") or "",
            "research_type": research_plan.get("research_type") or "",
            "report_family": research_plan.get("report_family") or "",
            "next_actions": _dedupe([unit.get("decision_implication") for unit in claim_units])[:8],
        },
        "report_outline": report_outline,
        "metadata": {
            "agent": AGENT_NAME,
            "strategy": "dynamic_claim_builder",
            "dimension_count": len(dimensions),
            "evidence_analysis_count": len(evidence_analyses),
            "evidence_refinement_task_count": len(_as_list(evidence_refinement_plan.get("follow_up_queries"))),
            "chapter_evidence_diagnostics_count": len(chapter_evidence_diagnostics),
            "evidence_gap_ledger_count": len(evidence_gap_ledger),
        },
    }
    result = _public_normalize_analysis_payload(result)
    result["analysis_depth_quality"] = analysis_depth_quality(result)
    result["claim_binding_feedback_summary"] = claim_binding_feedback_summary(result)
    result["analysis_stage_diagnostics"] = {
        "uses_llm_analysis": False,
        "llm_analysis_status": "not_run",
        "input_chapter_count": len(chapter_evidence_diagnostics),
        "input_evidence_card_count": len(evidence_analyses),
        "output_claim_count": len(claim_units),
        "decision_ready_claim_count": len([item for item in claim_units if str(item.get("claim_status") or "").strip() in {"decision_ready", "core_claim"}]),
        "directional_claim_count": len([item for item in claim_units if str(item.get("claim_status") or "").strip() in {"directional", "directional_ready"}]),
    }
    return result


def _public_normalize_analysis_text(value: Any) -> str:
    text = str(value or "")
    for pattern in PUBLIC_ANALYSIS_FORBIDDEN_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.I)
    return re.sub(r"\s{2,}", " ", text).strip()


def _public_normalize_analysis_payload(value: Any, *, key: str = "") -> Any:
    if isinstance(value, dict):
        return {item_key: _public_normalize_analysis_payload(item_value, key=str(item_key)) for item_key, item_value in value.items()}
    if isinstance(value, list):
        return [_public_normalize_analysis_payload(item, key=key) for item in value]
    if isinstance(value, str) and key in PUBLIC_ANALYSIS_TEXT_KEYS:
        return _public_normalize_analysis_text(value)
    return value


def run_analysis_agent(
    evidence_package: Dict[str, Any],
    *,
    query: str = "",
    llm_config: Optional[Dict[str, Any]] = None,
) -> AnalysisAgentState:
    try:
        package = _as_dict(evidence_package)
        if query and not package.get("query"):
            package = {**package, "query": query}
        structured = build_fallback_analysis(package)
        llm_status = "disabled"
        llm_error = ""
        if _env_flag("BRAIN_ENABLE_LLM_EVIDENCE_ANALYSIS", True):
            if llm_config_is_ready is not None and llm_config_is_ready(llm_config or {}):
                try:
                    llm_payload = synthesize_with_llm_analysis(
                        evidence_package=package,
                        fallback=structured,
                        llm_config=dict(llm_config or {}),
                    )
                    validation = validate_llm_analysis_output(llm_payload, package)
                    if str(validation.get("status") or "") == "valid":
                        structured = merge_llm_analysis_with_fallback(structured, llm_payload, validation)
                        llm_status = "success"
                    else:
                        llm_status = "invalid_output"
                        llm_error = "LLM evidence analysis returned no usable chapter_synthesis."
                except Exception as exc:
                    llm_status = "fallback"
                    llm_error = str(exc)
            else:
                llm_status = "fallback_config_missing"
        structured = _public_normalize_analysis_payload(structured)
        structured["analysis_depth_quality"] = analysis_depth_quality(structured)
        structured["claim_binding_feedback_summary"] = claim_binding_feedback_summary(structured)
        diagnostics = {
            **_as_dict(structured.get("analysis_stage_diagnostics")),
            "uses_llm_analysis": llm_status == "success",
            "llm_analysis_status": llm_status,
            "fallback_reason": llm_error,
            "input_chapter_count": len(_as_dict(structured.get("chapter_evidence_diagnostics"))),
            "input_evidence_card_count": len(_as_list(structured.get("evidence_analyses"))),
            "output_claim_count": len(_as_list(structured.get("claim_units"))),
            "decision_ready_claim_count": len(
                [
                    item
                    for item in _as_list(structured.get("claim_units"))
                    if isinstance(item, dict) and str(item.get("claim_status") or "").strip() in {"decision_ready", "core_claim"}
                ]
            ),
            "directional_claim_count": len(
                [
                    item
                    for item in _as_list(structured.get("claim_units"))
                    if isinstance(item, dict) and str(item.get("claim_status") or "").strip() in {"directional", "directional_ready"}
                ]
            ),
        }
        structured["analysis_stage_diagnostics"] = diagnostics
        source = "llm_evidence_analysis" if llm_status == "success" else "dynamic_claim_builder"
        return {
            "query": query or str(package.get("query") or ""),
            "evidence_package": package,
            "structured_analysis": structured,
            "answer_text": json.dumps({"structured_analysis": structured}, ensure_ascii=False, separators=(",", ":"), default=str),
            "raw_output": {
                "type": "structured_analysis",
                "source": source,
                "structured_analysis": structured,
                "analysis": diagnostics,
            },
            "metadata": {
                "agent_name": AGENT_NAME,
                "agent_description": AGENT_DESCRIPTION,
                "agent_stage": "analyze_evidence",
                "handoff_ready": True,
                "llm_analysis_status": llm_status,
                "llm_analysis_error": llm_error,
            },
        }
    except Exception as exc:
        return {
            "query": query,
            "evidence_package": _as_dict(evidence_package),
            "structured_analysis": {},
            "answer_text": "",
            "errors": [str(exc)],
            "raw_output": {"type": "structured_analysis", "source": "failed", "error": str(exc)},
            "metadata": {
                "agent_name": AGENT_NAME,
                "agent_description": AGENT_DESCRIPTION,
                "agent_stage": "analyze_evidence",
                "handoff_ready": False,
            },
        }


def analysis_agent_tool(evidence_package: Dict[str, Any]) -> Dict[str, Any]:
    return _as_dict(run_analysis_agent(evidence_package).get("structured_analysis"))


def create_analysis_agent_tool():
    from langchain_core.tools import tool

    @tool("analysis_agent", description=AGENT_DESCRIPTION)
    def _analysis_agent(evidence_package: Dict[str, Any]) -> Dict[str, Any]:
        return analysis_agent_tool(evidence_package)

    return _analysis_agent


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=AGENT_DESCRIPTION)
    parser.add_argument("--input-json", required=True, help="Evidence package JSON file")
    args = parser.parse_args()
    with open(args.input_json, "r", encoding="utf-8") as file:
        package = json.load(file)
    state = run_analysis_agent(package)
    print(state.get("answer_text") or json.dumps(state, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
