from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypedDict

try:
    from rag_pipeline.contracts.claim_roles import classify_claim_unit_roles
    from rag_pipeline.contracts.evidence_support_validation import (
        incomplete_metric_cards_for_numeric_claim,
        validate_claim_supported_by_facts,
    )
    from rag_pipeline.contracts.research_reflection import build_research_reflection_memo
    from rag_pipeline.contracts.evidence_quality import classify_evidence
    from rag_pipeline.search.memory import call_openai_compatible_json, llm_config_is_ready, normalize_llm_config
    from .evidence_merger import get_dynamic_dimensions
    from .summary_quality import sanitize_summary_judgments
except Exception:  # pragma: no cover - script mode fallback
    try:
        from rag_pipeline.contracts.claim_roles import classify_claim_unit_roles  # type: ignore
    except Exception:  # pragma: no cover
        classify_claim_unit_roles = None  # type: ignore
    try:
        from rag_pipeline.contracts.evidence_quality import classify_evidence  # type: ignore
    except Exception:  # pragma: no cover
        classify_evidence = None  # type: ignore
    try:
        from rag_pipeline.contracts.evidence_support_validation import (  # type: ignore
            incomplete_metric_cards_for_numeric_claim,
            validate_claim_supported_by_facts,
        )
    except Exception:  # pragma: no cover
        incomplete_metric_cards_for_numeric_claim = None  # type: ignore
        validate_claim_supported_by_facts = None  # type: ignore
    try:
        from rag_pipeline.contracts.research_reflection import build_research_reflection_memo  # type: ignore
    except Exception:  # pragma: no cover
        build_research_reflection_memo = None  # type: ignore
    try:
        from rag_pipeline.search.memory import call_openai_compatible_json, llm_config_is_ready, normalize_llm_config  # type: ignore
    except Exception:  # pragma: no cover
        call_openai_compatible_json = None  # type: ignore
        llm_config_is_ready = None  # type: ignore
        normalize_llm_config = None  # type: ignore
    from evidence_merger import get_dynamic_dimensions  # type: ignore
    from summary_quality import sanitize_summary_judgments  # type: ignore


AGENT_NAME = "analysis_agent"
AGENT_DESCRIPTION = "Dynamic Research Claim Builder. Converts evidence packages into claim units for the writer."
CHAPTER_EVIDENCE_COLLECTIONS = (
    "core_evidence",
    "supporting_evidence",
    "metric_evidence",
    "case_evidence",
    "counter_evidence",
    "directional_evidence",
    "sample_evidence",
    "table_evidence",
    "clue_evidence",
)
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


def _parse_structured_string(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "[{":
        return value
    if text.endswith("...") or text.count("{") != text.count("}") or text.count("[") != text.count("]"):
        return value
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except Exception:
            continue
        if isinstance(parsed, (dict, list)):
            return parsed
    return value


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


def _int_or_zero(value: Any) -> int:
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


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


GENERIC_LLM_CLAIM_PATTERNS = [
    r"目前只能形成方向性观察",
    r"需要用可追溯来源和连续指标",
    r"尚不足以支撑强结论",
    r"目前只有线索或背景材料",
    r"证据不足",
    r"建议补证",
    r"正文应以",
    r"后续验证",
    r"可追溯来源继续校准",
    r"\?{6,}",
    r"still lacks enough strong evidence",
    r"lacks enough strong evidence for a definitive conclusion",
    r"证据不足",
    r"建议补证",
    r"正文应以",
    r"方向性观察",
    r"后续验证",
    r"继续校准",
]


def _is_generic_llm_claim(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    return any(re.search(pattern, text, flags=re.I) for pattern in GENERIC_LLM_CLAIM_PATTERNS)


def _safe_public_claim_from_chapter(chapter: Dict[str, Any]) -> str:
    question = _compact(chapter.get("chapter_title") or chapter.get("chapter_question") or chapter.get("chapter_id") or "本章", 120)
    facts = _chapter_fact_texts(chapter, limit=2)
    if facts:
        return _claim_from_public_fact(question, facts[0])
    return ""


def _safe_public_reasoning_from_chapter(chapter: Dict[str, Any]) -> str:
    facts = _chapter_fact_texts(chapter, limit=4)
    if facts:
        return _reasoning_from_public_facts(facts)
    mechanisms = [
        _public_normalize_analysis_text(item)
        for item in _as_list(chapter.get("mechanism_chain"))
        if str(item or "").strip()
    ]
    mechanisms = [item for item in mechanisms if item and not _has_internal_analysis_language(item)]
    if mechanisms:
        return "；".join(mechanisms[:3])
    return ""


def _safe_chapter_counter_text(chapter: Dict[str, Any]) -> str:
    raw = chapter.get("counter_evidence_boundary")
    candidates = [raw] if isinstance(raw, str) else _as_list(raw)
    for item in candidates:
        text = _public_normalize_analysis_text(_compact(item, 260))
        if text and not _has_internal_analysis_language(text):
            return text
    return ""


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
    return ""


def _public_claim_from_chapter(chapter: Dict[str, Any]) -> str:
    question = _compact(chapter.get("chapter_title") or chapter.get("chapter_question") or chapter.get("chapter_id") or "本章", 120)
    facts = _chapter_fact_texts(chapter, limit=2)
    if facts:
        return _claim_from_public_fact(question, facts[0])
    return ""


def _public_reasoning_from_chapter(chapter: Dict[str, Any]) -> str:
    facts = _chapter_fact_texts(chapter, limit=4)
    if facts:
        return _reasoning_from_public_facts(facts)
    mechanisms = [
        _public_normalize_analysis_text(item)
        for item in _as_list(chapter.get("mechanism_chain"))
        if str(item or "").strip()
    ]
    mechanisms = [item for item in mechanisms if item and not _has_internal_analysis_language(item)]
    if mechanisms:
        return "；".join(mechanisms[:3])
    return ""


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
    for chapter in _as_list(evidence_package.get("chapter_evidence_packages")):
        if not isinstance(chapter, dict):
            continue
        for value in (chapter.get("chapter_id"), chapter.get("chapter_title"), chapter.get("chapter_question")):
            text = str(value or "").strip()
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


VERIFIED_SOURCE_STATUSES = {"readpage_verified", "document_verified"}
DOCUMENT_SOURCE_RE = re.compile(
    r"(\.pdf(?:$|\?)|annual[-_ ]?report|financial[-_ ]?report|filing|prospectus|"
    r"announcement|disclosure|standard|whitepaper|policy|regulation|official|gov\.|\.gov|exchange)",
    re.I,
)


def _source_verification_status(item: Dict[str, Any]) -> str:
    source = _source_payload(item)
    explicit = str(
        item.get("source_verification_status")
        or source.get("source_verification_status")
        or item.get("verification_status")
        or source.get("verification_status")
        or ""
    ).strip().lower()
    if explicit in {"search_result_only", "readpage_verified", "document_verified", "inaccessible"}:
        return explicit
    if not _has_traceable_source(item):
        return "inaccessible"
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
    source_text = " ".join(str(value or "") for value in [url, source.get("source_type"), source.get("title"), item.get("source_type"), item.get("source_family")])
    if document_ref or DOCUMENT_SOURCE_RE.search(source_text):
        return "document_verified"
    if bool(
        source.get("readpage_verified")
        or source.get("auto_readpage")
        or source.get("readpage_priority")
        or item.get("readpage_verified")
        or item.get("auto_readpage")
        or item.get("readpage_priority")
    ):
        return "readpage_verified"
    for key in ("mainText", "main_text", "markdown", "content", "text", "quote", "page_content"):
        if str(source.get(key) or item.get(key) or "").strip():
            return "readpage_verified"
    return "search_result_only"


def _has_verified_source(item: Dict[str, Any]) -> bool:
    return _source_verification_status(item) in VERIFIED_SOURCE_STATUSES


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


NOISY_PUBLIC_FACT_RE = re.compile(
    r"(skip\s+to\s+content|picture\s+intentionally\s+omitted|cookie|login|"
    r"search\s+results?|related\s+articles?|download\s+pdf|javascript|"
    r"登录|导航|搜索|点击|下载|目录|网页快照|炒股就看|金麒麟|股吧|百度百科)",
    re.I,
)


def _public_fact_quality(item: Dict[str, Any]) -> Dict[str, Any]:
    return _as_dict(item.get("public_fact_quality"))


def _public_fact_card(item: Dict[str, Any]) -> Dict[str, Any]:
    quality = _public_fact_quality(item)
    return (
        _as_dict(item.get("public_fact_card"))
        or _as_dict(quality.get("public_fact_card"))
        or _as_dict(_as_dict(item.get("evidence_card")).get("public_fact_card"))
    )


def _distilled_public_fact(item: Dict[str, Any]) -> str:
    card = _public_fact_card(item)
    quality = _public_fact_quality(item)
    for payload in (card, quality, item):
        for key in ("distilled_fact", "fact", "clean_fact", "summary", "object", "finding"):
            text = _compact(_as_dict(payload).get(key), 260)
            if text:
                return text
    subject = _compact(card.get("subject"), 80)
    action = _compact(card.get("action"), 80)
    obj = _compact(card.get("object"), 180)
    if subject and (action or obj):
        return _compact(" ".join(part for part in (subject, action, obj) if part), 260)
    return ""


def _is_noisy_public_fact(text: Any) -> bool:
    value = str(text or "").strip()
    if not value or len(value) < 8:
        return True
    if NOISY_PUBLIC_FACT_RE.search(value):
        return True
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?%?", value):
        return True
    return False


def _is_public_quality_card(item: Dict[str, Any]) -> bool:
    if not isinstance(item, dict):
        return False
    if not str(item.get("evidence_id") or item.get("id") or "").strip():
        return False
    if _is_fake_or_placeholder_source(item) or _is_title_only_source(item):
        return False
    if bool(item.get("source_title_url_mismatch_suspected") or item.get("source_mismatch")):
        return False
    quality = _public_fact_quality(item)
    if quality and quality.get("eligible_for_report") is False:
        return False
    if quality and str(quality.get("rejection_reason") or "").strip():
        return False
    if not _has_traceable_source(item):
        return False
    if _is_noisy_public_fact(_distilled_public_fact(item)):
        return False
    level = _analysis_source_level(item)
    allowed = _analysis_allowed_use(item)
    return bool(level in {"A", "B", "C"} or allowed == "directional_signal")


def _source_identity_key(item: Dict[str, Any]) -> str:
    source = _source_payload(item)
    url = str(source.get("url") or source.get("source_url") or item.get("source_url") or "").strip().lower()
    if url:
        return "url:" + re.sub(r"#.*$", "", url).rstrip("/")
    document_ref = str(
        source.get("document_id")
        or source.get("doc_id")
        or source.get("page_ref")
        or item.get("document_id")
        or item.get("doc_id")
        or item.get("page_ref")
        or ""
    ).strip().lower()
    if document_ref:
        return "doc:" + document_ref
    title = str(source.get("title") or source.get("name") or item.get("source_title") or "").strip().lower()
    publisher = str(source.get("publisher") or source.get("source") or item.get("source_text") or "").strip().lower()
    date = str(source.get("date") or source.get("published_at") or item.get("period") or "").strip().lower()
    combined = "|".join(part for part in (publisher, title, date) if part)
    return "meta:" + _normalize_key(combined) if combined else ""


def _ensure_sentence(text: Any) -> str:
    value = _compact(text, 260).strip()
    if not value:
        return ""
    if value[-1] in ".!?。！？":
        return value
    return value + "。"


def _claim_from_public_fact(dimension: Any, fact: Any, strength: str = "") -> str:
    return _ensure_sentence(fact)


def _reasoning_from_public_facts(facts: List[str]) -> str:
    cleaned = [_ensure_sentence(item) for item in facts if not _is_noisy_public_fact(item)]
    cleaned = [item for item in cleaned if item]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    head = cleaned[0]
    rest = "；".join(cleaned[1:3])
    return f"{head} 同时，{rest} 进一步说明同一变化方向。"


GAP_TO_VERIFY_ACTION = {
    "source_trace_missing": "补齐来源的原始链接或文档编号",
    "title_only_source": "补充来源正文，不能只用标题做引用",
    "fake_or_placeholder_source": "替换占位来源为可访问的官方或权威披露",
    "source_not_verified": "对来源做 readpage 或文档级别的复核",
    "needs_authoritative_source": "补 A/B 级官方或权威披露",
    "needs_corroboration": "补第二条独立来源做交叉验证",
    "source_metadata_missing": "补齐来源标题、发布主体与日期",
    "metric_value_missing": "补齐指标的具体数值",
    "metric_period_missing": "补齐指标的时间窗口与统计口径",
    "counter_needs_ab_source": "为反证补 A/B 级独立来源",
    "insufficient_ab_sources": "补充 A/B 级核心来源以建立主结论",
    "counter_evidence_missing": "补反向案例或失败样本以校准结论边界",
}


def _build_mechanism_chain(
    fact_chain: List[str],
    metric_facts: List[str],
    claim_strength: str,
    distinct_source_count: int,
) -> List[str]:
    parts: List[str] = []
    if fact_chain:
        parts.append(_ensure_sentence(fact_chain[0]))
    if metric_facts:
        parts.append(f"可比口径需要围绕 {metric_facts[0]} 继续校准。")
    if len(fact_chain) >= 2:
        parts.append(f"{_ensure_sentence(fact_chain[1])} 这为前述判断提供了另一个侧面的验证。")
    if claim_strength == "strong" and distinct_source_count >= 2:
        parts.append("多条独立且可核验来源指向同一方向，结论可以作为本章的主要判断。")
    elif claim_strength == "moderate":
        parts.append("已有可信事实支撑，但样本覆盖和统计口径仍决定结论能否继续上调。")
    elif claim_strength == "directional":
        parts.append("当前材料只能支撑审慎判断，不能直接放大为定量或全行业结论。")
    elif claim_strength == "weak":
        parts.append("现有材料覆盖面较窄，本章仅保留阶段性观察。")
    return [item for item in parts if item]


def _build_verify_kpi(
    all_gaps: List[str],
    followups: List[str],
    metric_facts: List[str],
) -> str:
    actions: List[str] = []
    for gap in all_gaps[:5]:
        action = GAP_TO_VERIFY_ACTION.get(gap)
        if action and action not in actions:
            actions.append(action)
    if not actions and followups:
        actions.append(_compact(followups[0], 120))
    if not actions:
        if metric_facts:
            metric_name = metric_facts[0].split(":")[0].strip()
            return f"持续追踪指标 {metric_name} 的同口径数据，并补充第二来源做交叉复核。"
        return "持续追踪本章关键指标的同口径数据，并补充第二来源做交叉复核。"
    return "；".join(actions)


def _build_decision_implication(claim_strength: str, first_fact: str) -> str:
    if claim_strength == "strong":
        return "可纳入本章核心结论，并作为下游决策（进入/投资/资源排序）的事实依据。"
    if claim_strength == "moderate":
        return "可作为本章主结论的支撑，但需在反向样本和口径一致性条件下保留弹性。"
    if claim_strength == "directional":
        return "只能作为趋势性提示，下游决策不应直接放大该信号，需等待 A/B 级来源补齐。"
    return "现有材料只能用于本章的背景说明，正式结论需要在更多来源到位后再形成。"


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
        dimension_keys = {
            _normalize_key(dimension),
        }
        item_keys = {
            _normalize_key(item.get("chapter_id")),
            _normalize_key(item.get("hypothesis_id")),
            _normalize_key(item.get("dimension_id")),
            _normalize_key(item.get("dimension_name")),
            _normalize_key(item.get("evidence_goal")),
            _normalize_key(item.get("dimension")),
        }
        item_keys = {key for key in item_keys if key}
        if dimension_keys & item_keys or any(_overlaps(dimension, key) for key in item_keys):
            items.append(dict(item))
    for chapter in _as_list(evidence_package.get("chapter_evidence_packages")):
        if not isinstance(chapter, dict):
            continue
        chapter_keys = {
            _normalize_key(chapter.get("chapter_id")),
            _normalize_key(chapter.get("hypothesis_id")),
            _normalize_key(chapter.get("dimension_id")),
            _normalize_key(chapter.get("chapter_title")),
            _normalize_key(chapter.get("chapter_question")),
        }
        chapter_keys = {key for key in chapter_keys if key}
        dimension_key = _normalize_key(dimension)
        if dimension_key not in chapter_keys and not any(_overlaps(dimension, key) for key in chapter_keys):
            continue
        for collection in CHAPTER_EVIDENCE_COLLECTIONS:
            for item in _as_list(chapter.get(collection)):
                if not isinstance(item, dict):
                    continue
                copied = dict(item)
                copied.setdefault("dimension", dimension)
                copied.setdefault("chapter_id", chapter.get("chapter_id"))
                copied.setdefault("dimension_name", chapter.get("chapter_title") or dimension)
                items.append(copied)
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
            and _has_verified_source(item)
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
        claim = synthesis.get("takeaway") or ""
        if not str(claim or "").strip():
            continue
        units.append(
            {
                "question": dimension,
                "claim": claim,
                "claim_status": "decision_ready" if _as_list(synthesis.get("evidence_ids")) else "directional",
                "claim_strength": synthesis.get("claim_strength") or ("moderate" if _as_list(synthesis.get("evidence_ids")) else "directional"),
                "quality_status": "valid" if _as_list(synthesis.get("evidence_ids")) else "directional_with_boundary",
                "supporting_evidence": synthesis.get("evidence_ids") or synthesis.get("directional_evidence_ids") or [],
                "evidence_refs": synthesis.get("evidence_ids") or synthesis.get("directional_evidence_ids") or [],
                "counter_evidence": synthesis.get("counter") or "",
                "reasoning": synthesis.get("mechanism") or synthesis.get("explain_why") or "",
                "mechanism": synthesis.get("mechanism") or "",
                "decision_implication": synthesis.get("decision_implication") or "",
                "confidence": synthesis.get("confidence"),
                "dimension": dimension,
            }
        )
    return units


def _chapter_insights_from_synthesis(
    dimension_synthesis: Dict[str, Dict[str, Any]],
    chapter_id_lookup: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Build chapter_insights from a per-dimension synthesis map.

    `chapter_id_lookup` maps a dimension string to the canonical chapter_id
    used by `chapter_evidence_diagnostics`. When absent we fall back to the
    raw dimension string itself — this is critical because
    `_chapter_key_for_item` (used to key `chapter_evidence_diagnostics`)
    also falls back to the raw dimension. If both sides normalize, English
    dimensions with spaces/underscores stop matching. Using a synthetic
    `chapter_{index}` is the absolute last resort and only fires when the
    dimension key itself is empty.
    """

    insights: List[Dict[str, Any]] = []
    chapter_id_lookup = chapter_id_lookup or {}
    for index, (dimension, synthesis) in enumerate(dimension_synthesis.items(), start=1):
        synthesis = _as_dict(synthesis)
        chapter_id = str(
            chapter_id_lookup.get(dimension)
            or str(dimension or "").strip()
            or f"chapter_{index}"
        )
        claim = synthesis.get("takeaway") or ""
        key_claims = []
        if str(claim or "").strip():
            key_claims.append(
                {
                    "claim": claim,
                    "claim_status": "decision_ready" if _as_list(synthesis.get("evidence_ids")) else "directional",
                    "claim_strength": synthesis.get("claim_strength") or ("moderate" if _as_list(synthesis.get("evidence_ids")) else "directional"),
                    "supporting_evidence": synthesis.get("evidence_ids") or synthesis.get("directional_evidence_ids") or [],
                    "evidence_refs": synthesis.get("evidence_ids") or synthesis.get("directional_evidence_ids") or [],
                    "mechanism": synthesis.get("mechanism") or "",
                    "reasoning": synthesis.get("mechanism") or "",
                    "counter_evidence": synthesis.get("counter") or "",
                    "decision_implication": synthesis.get("decision_implication") or "",
                    "confidence": synthesis.get("confidence"),
                    "what_to_verify_next": [synthesis.get("verify_kpi")] if synthesis.get("verify_kpi") else [],
                }
            )
        insights.append(
            {
                "chapter_id": chapter_id,
                "chapter_question": dimension,
                "chapter_answer": synthesis.get("chapter_answer") or synthesis.get("takeaway") or "",
                "core_answer": synthesis.get("chapter_answer") or synthesis.get("takeaway") or "",
                "fact_chain": _as_list(synthesis.get("fact_chain")),
                "mechanism_chain": _as_list(synthesis.get("mechanism_chain")),
                "counter_evidence_boundary": _as_list(synthesis.get("counter_evidence_boundary")),
                "decision_implication": synthesis.get("decision_implication") or "",
                "key_claims": key_claims,
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
        and _has_verified_source(item)
    )


def _evidence_strength(item: Dict[str, Any]) -> str:
    level = _analysis_source_level(item)
    allowed = _analysis_allowed_use(item)
    traceable = _has_traceable_source(item)
    verified = _has_verified_source(item)
    if allowed == "directional_signal":
        return "directional"
    if level in {"A", "B"} and allowed == "core_claim":
        return "strong"
    if level in {"A", "B"} and allowed in {"core_claim", "supporting"} and not traceable:
        return "weak"
    if level in {"A", "B"} and allowed in {"core_claim", "supporting"} and not verified:
        return "moderate"
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
    if level in {"A", "B"} and allowed in {"core_claim", "supporting"} and traceable and not _has_verified_source(item):
        gaps.append("source_not_verified")
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
        "chapter_id": item.get("chapter_id"),
        "dimension_id": item.get("dimension_id"),
        "evidence_goal": item.get("evidence_goal"),
        "dimension": dimension,
        "fact": fact,
        "writer_evidence": fact,
        "source": source,
        "source_label": _source_label(item),
        "source_verification_status": _source_verification_status(item),
        "source_verified": _has_verified_source(item),
        "confidence": _confidence(item),
        "hypothesis_id": item.get("hypothesis_id"),
        "hypothesis_statement": item.get("hypothesis_statement"),
        "proof_role": card.get("proof_role") or item.get("proof_role") or ("counter" if item.get("counter_evidence") else "support"),
        "source_level": card.get("source_level") or _analysis_source_level(item),
        "source_tier": item.get("source_tier") or card.get("source_tier"),
        "source_family": card.get("source_family") or item.get("source_family"),
        "metric": item.get("metric"),
        "value": item.get("value"),
        "allowed_use": card.get("allowed_use") or item.get("allowed_use"),
        "evidence_fit_score": item.get("evidence_fit_score") or card.get("evidence_fit_score"),
        "metric_proof_gaps": _as_list(item.get("metric_proof_gaps") or card.get("metric_proof_gaps")),
        "analysis_readiness": item.get("analysis_readiness") or card.get("analysis_readiness"),
        "evidence_card": card,
        "evidence_card_only": True,
        "evidence_strength": strength,
        "claim_strength": strength,
        "evidence_gaps": gaps,
        "verification_questions": verification_questions,
        "suggested_followup_query": followup_query,
        # NOTE: per-evidence claim/reasoning/mechanism/counter/decision_implication are
        # intentionally left empty so that downstream consumers must derive them from
        # actual analysis output (LLM synthesis or _dimension_synthesis) rather than
        # restate the same hardcoded template for every evidence card.
        "claim": "",
        "reasoning": "",
        "mechanism": "",
        "counter": "",
        "decision_implication": "",
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
    verified_source_keys = {
        _source_identity_key(item)
        for item in usable
        if _has_verified_source(item)
    }
    verified_source_keys = {key for key in verified_source_keys if key}
    if len(verified_source_keys) >= 2:
        claim_strength = "strong"
    elif usable:
        claim_strength = "moderate"
    elif directional:
        claim_strength = "directional"
    else:
        claim_strength = "weak"
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
    distinct_verified_source_count = len({_source_identity_key(item) for item in usable if _has_verified_source(item) and _source_identity_key(item)})
    if not first_fact:
        return {
            "takeaway": "",
            "chapter_answer": "",
            "fact": "",
            "fact_chain": [],
            "mechanism_chain": [],
            "counter_evidence_boundary": [],
            "supporting_facts": [],
            "explain_why": "",
            "mechanism": "",
            "inference": "",
            "counter": "",
            "verify_kpi": "",
            "decision_implication": "",
            "evidence_ids": [],
            "directional_evidence_ids": [],
            "claim_strength": "weak",
            "distinct_verified_ab_source_count": 0,
            "confidence": 0.0,
            "limits": "；".join(all_gaps[:5]),
            "evidence_gap_tags": all_gaps,
            "followup_queries": followups[:6],
        }
    takeaway = _claim_from_public_fact(dimension, first_fact, claim_strength)
    mechanism_chain = _build_mechanism_chain(
        fact_chain=fact_chain,
        metric_facts=metric_facts,
        claim_strength=claim_strength,
        distinct_source_count=distinct_verified_source_count,
    )
    mechanism = "\n".join(mechanism_chain) if mechanism_chain else ""
    counter = _ensure_sentence(counter_facts[0]) if counter_facts else "反向样本或失败案例仍需补充，用于校准该判断的适用边界。"
    verify_kpi = _build_verify_kpi(all_gaps, followups, metric_facts)
    decision_implication = _build_decision_implication(claim_strength, first_fact)
    source_items = usable or directional
    return {
        "takeaway": takeaway,
        "chapter_answer": takeaway,
        "fact": first_fact,
        "fact_chain": fact_chain,
        "mechanism_chain": mechanism_chain,
        "counter_evidence_boundary": [counter] if counter else [],
        "supporting_facts": (usable_facts or directional_facts)[:6],
        "explain_why": mechanism,
        "mechanism": mechanism,
        "inference": mechanism,
        "counter": counter,
        "verify_kpi": verify_kpi,
        "decision_implication": decision_implication,
        "evidence_ids": evidence_ids,
        "directional_evidence_ids": directional_ids,
        "claim_strength": claim_strength,
        "distinct_verified_ab_source_count": distinct_verified_source_count,
        "confidence": round(sum(_confidence(item) for item in source_items) / max(len(source_items), 1), 3) if source_items else 0.0,
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
        metric_facts = [
            _compact(f"{item.get('metric')}: {item.get('value')} {item.get('period') or ''}", 160)
            for item in metric_items
            if str(item.get("metric") or "").strip() and str(item.get("value") or "").strip()
        ]
        distinct_verified_source_count = len(
            {
                _source_identity_key(item)
                for item in usable
                if _has_verified_source(item) and _source_identity_key(item)
            }
        )
        claim_strength = "strong" if ready and distinct_verified_source_count >= 2 else ("moderate" if support else "directional")
        mechanism_chain = _build_mechanism_chain(
            fact_chain=fact_chain,
            metric_facts=metric_facts,
            claim_strength=claim_strength,
            distinct_source_count=distinct_verified_source_count,
        )
        counter_boundary = [_compact(item.get("fact"), 180) for item in counters if str(item.get("fact") or "").strip()][:3]
        evidence_chapter_id = next(
            (
                str(item.get("chapter_id") or "").strip()
                for item in support + usable + relevant
                if str(item.get("chapter_id") or "").strip()
            ),
            "",
        )
        if evidence_chapter_id:
            chapter_id = evidence_chapter_id
            chapter_id_source = "evidence_chapter_id"
        elif hypothesis_id:
            chapter_id = hypothesis_id
            chapter_id_source = "hypothesis_id"
        else:
            chapter_id = _normalize_key(statement) or f"chapter_{index}"
            chapter_id_source = "normalized_statement" if _normalize_key(statement) else "fallback_index"
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
                    "counter_evidence": "；".join(counter_boundary),
                    "decision_implication": _build_decision_implication(claim_strength, fact_chain[0] if fact_chain else statement),
                    "confidence": round(sum(float(item.get("confidence") or 0.0) for item in usable) / max(len(usable), 1), 3),
                    "what_to_verify_next": _build_verify_kpi(gaps, [], metric_facts),
                }
            )
        insights.append(
            {
                "chapter_id": chapter_id,
                "chapter_id_source": chapter_id_source,
                "hypothesis_id": hypothesis_id,
                "chapter_question": statement,
                "chapter_answer": statement if ready else "",
                "core_answer": statement if ready else "",
                "fact_chain": fact_chain,
                "mechanism_chain": mechanism_chain,
                "counter_evidence_boundary": counter_boundary,
                "decision_implication": _build_decision_implication(claim_strength, fact_chain[0] if fact_chain else statement) if fact_chain else "",
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


def _chapter_id_alias_set(*candidates: Any) -> List[str]:
    """Return a de-duplicated list of chapter-id aliases derived from candidates.

    Each candidate may be a chapter_id, hypothesis_id, dimension name, title,
    or question. We emit two forms for each: the raw stripped string and a
    normalized key (lowercased, connector-stripped). Downstream lookups can
    then resolve a chapter regardless of which form the caller has.
    """

    aliases: List[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        raw = str(candidate or "").strip()
        if not raw or raw in seen:
            continue
        seen.add(raw)
        aliases.append(raw)
        norm = _normalize_key(raw)
        if norm and norm not in seen:
            seen.add(norm)
            aliases.append(norm)
    return aliases


def resolve_chapter_id(
    diagnostics: Dict[str, Dict[str, Any]],
    query_id: Any,
) -> str:
    """Map any chapter id/title/dimension form to a key present in `diagnostics`.

    Cross-agent lookups frequently fail when one side stores `"ch_01"` and the
    other stores `"ch 01"` or the original dimension name. This helper checks
    each chapter's `chapter_id_aliases` list (set by `_chapter_evidence_diagnostics`)
    and returns the diagnostics key whose alias matches, or `""` if nothing matches.
    """

    raw = str(query_id or "").strip()
    if not raw:
        return ""
    if raw in diagnostics:
        return raw
    norm = _normalize_key(raw)
    if norm and norm in diagnostics:
        return norm
    for key, payload in diagnostics.items():
        aliases = _as_list(_as_dict(payload).get("chapter_id_aliases"))
        if raw in aliases or (norm and norm in aliases):
            return key
    return ""


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
        item["chapter_id_aliases"] = _chapter_id_alias_set(
            item["chapter_id"],
            chapter_id,
            item["chapter_title"],
            payload.get("chapter_question"),
            payload.get("hypothesis_id"),
            payload.get("dimension"),
            payload.get("dimension_name"),
        )
        item["analysis_readiness"] = _analysis_readiness(item)
        diagnostics[item["chapter_id"]] = item
    if diagnostics:
        return diagnostics

    buckets: Dict[str, Dict[str, Any]] = {}
    aliases_by_chapter: Dict[str, List[str]] = {}
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
        existing_aliases = aliases_by_chapter.setdefault(chapter_id, [])
        for new_alias in _chapter_id_alias_set(
            chapter_id,
            item.get("chapter_id"),
            item.get("hypothesis_id"),
            item.get("dimension_id"),
            item.get("dimension"),
            item.get("dimension_name"),
            item.get("evidence_goal"),
        ):
            if new_alias and new_alias not in existing_aliases:
                existing_aliases.append(new_alias)
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
    for chapter_id, bucket in buckets.items():
        if bucket["core_ab_source_count"] <= 0 and "insufficient_ab_sources" not in bucket["gap_types"]:
            bucket["gap_types"].append("insufficient_ab_sources")
        if bucket["counter_signal_count"] <= 0 and "counter_evidence_missing" not in bucket["gap_types"]:
            bucket["gap_types"].append("counter_evidence_missing")
        bucket["analysis_readiness"] = _analysis_readiness(bucket)
        bucket["claim_ready_evidence_refs"] = _dedupe(bucket["claim_ready_evidence_refs"])[:12]
        bucket["metric_ready_refs"] = _dedupe(bucket["metric_ready_refs"])[:12]
        bucket["counter_refs"] = _dedupe(bucket["counter_refs"])[:12]
        bucket["chapter_id_aliases"] = _chapter_id_alias_set(
            *aliases_by_chapter.get(chapter_id, []),
            bucket["chapter_id"],
            bucket["chapter_title"],
        )
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
    """Compute claim-quality metrics over a deduplicated set of claims.

    Earlier versions of this function counted the same claim once per container
    (report_insight_package.chapters[*].key_claims, chapter_insights[*].key_claims,
    and claim_units), which artificially inflated `repeated_claim_ratio` because
    merge_llm_analysis_with_fallback stores the same logical claim in all three
    structures. The fix is twofold:

    1. Deduplicate by `(chapter_id, normalized_claim_text)` across all sources
       before measuring anything — the same logical claim is now counted once.
    2. Measure `repeated_claim_ratio` semantically: the share of distinct claim
       texts that appear in more than one chapter (real recycling), instead of
       counting storage-layer duplication.
    """

    insight = _as_dict(structured_analysis.get("report_insight_package"))
    chapter_sources = _as_list(insight.get("chapters")) + _as_list(structured_analysis.get("chapter_insights"))
    claim_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def _record(chapter_id: str, claim: Dict[str, Any], chapter_question: str = "") -> None:
        normalized = _normalize_key(claim.get("claim") or claim.get("judgment"))
        if not normalized:
            return
        key = (str(chapter_id or "").strip(), normalized)
        if key in claim_by_key:
            return
        copied = dict(claim)
        if chapter_question:
            copied.setdefault("chapter_question", chapter_question)
        copied.setdefault("chapter_id", chapter_id)
        claim_by_key[key] = copied

    for chapter in chapter_sources:
        chapter = _as_dict(chapter)
        chapter_id = str(chapter.get("chapter_id") or "").strip()
        question = str(chapter.get("chapter_question") or "").strip()
        for claim in _as_list(chapter.get("key_claims")):
            if isinstance(claim, dict):
                _record(chapter_id, claim, question)

    for unit in _as_list(structured_analysis.get("claim_units")):
        if isinstance(unit, dict):
            chapter_id = str(unit.get("chapter_id") or unit.get("dimension") or "").strip()
            _record(chapter_id, unit, str(unit.get("dimension") or unit.get("question") or ""))

    claims = list(claim_by_key.values())
    claim_count = max(len(claims), 1)

    # Repeated ratio: share of distinct claim texts that occur in more than one
    # chapter. This isolates real cross-chapter recycling from storage duplication.
    claim_text_to_chapters: Dict[str, set] = {}
    for (chapter_id, normalized), _ in claim_by_key.items():
        claim_text_to_chapters.setdefault(normalized, set()).add(chapter_id)
    cross_chapter_repeats = sum(1 for chapters in claim_text_to_chapters.values() if len(chapters) > 1)
    repeated_ratio = round(cross_chapter_repeats / max(len(claim_text_to_chapters), 1), 3)

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
        refs = [
            str(ref or "").strip()
            for ref in _as_list(item.get("supporting_evidence") or item.get("evidence_refs") or item.get("supporting_evidence_refs"))
            if str(ref or "").strip()
        ]
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
    generic_ratio = round(generic_count / claim_count, 3)
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
        "cross_chapter_claim_repeats": cross_chapter_repeats,
        "distinct_claim_text_count": len(claim_text_to_chapters),
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
        # Resolve the unit's chapter id through the diagnostics alias table so
        # that minor surface differences (ch_01 vs ch-01 vs the raw dimension
        # name) still bind units to chapters. Falls back to the raw value when
        # no alias matches — keeps the legacy behaviour for unmapped units.
        raw_chapter_id = str(unit.get("chapter_id") or unit.get("hypothesis_id") or unit.get("dimension") or "").strip()
        if not raw_chapter_id:
            continue
        chapter_id = resolve_chapter_id(diagnostics, raw_chapter_id) or raw_chapter_id
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


def _chapter_filter_for_llm(evidence_package: Dict[str, Any], *, max_chapters: int) -> Dict[str, Dict[str, Any]]:
    packages = [item for item in _as_list(evidence_package.get("chapter_evidence_packages")) if isinstance(item, dict)]
    if packages:
        chapter_filter: Dict[str, Dict[str, Any]] = {}
        for package in packages[:max_chapters]:
            chapter_id = str(package.get("chapter_id") or "").strip()
            if not chapter_id:
                continue
            aliases = _dedupe(
                [
                    chapter_id,
                    package.get("chapter_title"),
                    package.get("chapter_question"),
                    package.get("title"),
                    *_as_list(package.get("chapter_id_aliases")),
                ]
            )[:12]
            chapter_filter[chapter_id] = {
                "chapter_id": chapter_id,
                "chapter_title": package.get("chapter_title") or package.get("title") or chapter_id,
                "chapter_question": package.get("chapter_question") or package.get("chapter_title") or package.get("title") or chapter_id,
                "chapter_id_aliases": aliases,
            }
        if chapter_filter:
            return chapter_filter
    diagnostics = _as_dict(evidence_package.get("chapter_evidence_diagnostics")) or _as_dict(evidence_package.get("evidence_analysis_by_chapter"))
    allowed_chapters = list(diagnostics.keys())[:max_chapters] if diagnostics else []
    return {key: _as_dict(diagnostics.get(key)) for key in allowed_chapters} if allowed_chapters else diagnostics


def _evidence_cards_for_llm(
    evidence_package: Dict[str, Any],
    *,
    max_chapters: int,
    max_per_chapter: int,
) -> List[Dict[str, Any]]:
    ledger_cards = _ledger_evidence_cards_for_llm(
        evidence_package,
        max_chapters=max_chapters,
        max_per_chapter=max_per_chapter,
    )
    if ledger_cards:
        return ledger_cards
    chapter_filter = _chapter_filter_for_llm(evidence_package, max_chapters=max_chapters)
    buckets: Dict[str, int] = {}
    cards: List[Dict[str, Any]] = []
    seen_refs: set[str] = set()
    source_items = _as_list(evidence_package.get("analysis_ready_evidence")) + _as_list(evidence_package.get("clean_evidence_list"))
    ranked_items = sorted(
        [item for item in source_items if isinstance(item, dict) and _is_public_quality_card(item)],
        key=lambda item: (
            1 if _has_verified_source(item) else 0,
            1 if _analysis_source_level(item) == "A" else 0,
            1 if _analysis_source_level(item) == "B" else 0,
            1 if _analysis_allowed_use(item) == "directional_signal" else 0,
            _confidence(item),
        ),
        reverse=True,
    )
    for item in ranked_items:
        if not isinstance(item, dict):
            continue
        evidence_id = str(item.get("evidence_id") or item.get("id") or "").strip()
        if not evidence_id or evidence_id in seen_refs:
            continue
        raw_chapter_id = _chapter_key_for_item(item)
        chapter_id = raw_chapter_id
        if chapter_filter:
            resolved_chapter_id = resolve_chapter_id(chapter_filter, raw_chapter_id)
            if not resolved_chapter_id:
                continue
            chapter_id = resolved_chapter_id
        if buckets.get(chapter_id, 0) >= max_per_chapter:
            continue
        source = _source_payload(item)
        card = _as_dict(item.get("evidence_card"))
        fact_card = _public_fact_card(item)
        fact = _distilled_public_fact(item)
        search_task = _as_dict(item.get("search_task"))
        requirement_id = str(
            item.get("requirement_id")
            or item.get("evidence_requirement_id")
            or search_task.get("requirement_id")
            or search_task.get("evidence_requirement_id")
            or search_task.get("slot_id")
            or ""
        ).strip()
        hypothesis_id = str(item.get("hypothesis_id") or search_task.get("hypothesis_id") or "").strip()
        source_id = str(
            item.get("source_id")
            or item.get("source_ref")
            or item.get("citation_ref")
            or source.get("source_ref")
            or source.get("document_id")
            or source.get("url")
            or ""
        ).strip()
        search_task_id = str(item.get("search_task_id") or search_task.get("task_id") or search_task.get("id") or "").strip()
        lineage = {
            key: value
            for key, value in {
                "chapter_id": chapter_id,
                "hypothesis_id": hypothesis_id,
                "requirement_id": requirement_id,
                "fact_id": evidence_id,
                "source_id": source_id,
                "search_task_id": search_task_id,
            }.items()
            if value
        }
        cards.append(
            {
                "evidence_id": evidence_id,
                "chapter_id": chapter_id,
                "hypothesis_id": hypothesis_id,
                "requirement_id": requirement_id,
                "analysis_role": str(item.get("analysis_role") or card.get("analysis_role") or "").strip(),
                "analysis_eligible": bool(item.get("analysis_eligible") if "analysis_eligible" in item else card.get("analysis_eligible")),
                "allowed_use": str(item.get("allowed_use") or card.get("allowed_use") or "").strip(),
                "source_id": source_id,
                "search_task_id": search_task_id,
                "lineage": lineage,
                "public_fact_card": fact_card,
                "distilled_fact": _compact(fact, 360),
                "fact": _compact(fact, 360),
                "metric": _compact(item.get("metric"), 100),
                "value": _compact(item.get("value"), 100),
                "unit": _compact(item.get("unit") or _as_dict(item.get("metric_definition")).get("unit"), 60),
                "period": _compact(item.get("period") or source.get("date"), 80),
                "source_level": str(item.get("source_level") or card.get("source_level") or "").strip().upper(),
                "allowed_use": str(item.get("allowed_use") or card.get("allowed_use") or "").strip(),
                "proof_role": str(item.get("proof_role") or card.get("proof_role") or "").strip().lower(),
                "source_verification_status": _source_verification_status(item),
                "can_support": _as_list(item.get("can_support")) or _as_list(card.get("can_support")),
                "cannot_support": _as_list(item.get("cannot_support")) or _as_list(card.get("cannot_prove")),
                "proof_strength": str(item.get("proof_strength") or item.get("evidence_strength") or "").strip(),
                "repair_need": _as_list(item.get("repair_need")) or _as_list(item.get("evidence_gaps")),
                "source_title": _compact(source.get("title") or source.get("source") or source.get("name"), 160),
                "source_url": str(source.get("url") or item.get("source_url") or "").strip(),
            }
        )
        seen_refs.add(evidence_id)
        buckets[chapter_id] = buckets.get(chapter_id, 0) + 1
    return [item for item in cards if item.get("evidence_id") and item.get("fact")]


def _artifact_ledger_run_id_for_analysis(evidence_package: Dict[str, Any]) -> str:
    return str(
        evidence_package.get("artifact_ledger_run_id")
        or evidence_package.get("stage_snapshot_run_id")
        or evidence_package.get("run_id")
        or os.getenv("REPORT_STAGE_SNAPSHOT_RUN_ID")
        or ""
    ).strip()


def _ledger_evidence_cards_for_llm(
    evidence_package: Dict[str, Any],
    *,
    max_chapters: int,
    max_per_chapter: int,
) -> List[Dict[str, Any]]:
    if not _env_flag("ARTIFACT_LEDGER_ANALYSIS_CONTEXT_ENABLED", True):
        return []
    run_id = _artifact_ledger_run_id_for_analysis(evidence_package)
    if not run_id:
        return []
    try:
        from rag_pipeline.context.context_view_builder import build_analysis_context_view
    except Exception:
        return []

    requirements_by_chapter = _requirements_by_chapter_for_llm(evidence_package)
    requirement_chapter_lookup = {
        str(requirement.get("requirement_id") or "").strip(): chapter_id
        for chapter_id, requirements in requirements_by_chapter.items()
        for requirement in requirements
        if str(requirement.get("requirement_id") or "").strip()
    }
    requirement_ids = list(requirement_chapter_lookup.keys())
    views: List[Dict[str, Any]] = []
    try:
        if requirement_ids:
            for requirement_id in requirement_ids[: max_chapters * max_per_chapter]:
                view = build_analysis_context_view(run_id, requirement_id=requirement_id)
                if view.get("status") == "ready":
                    views.append(view)
        else:
            view = build_analysis_context_view(run_id)
            if view.get("status") == "ready":
                views.append(view)
    except Exception:
        return []
    if not views:
        return []

    chapter_filter = _chapter_filter_for_llm(evidence_package, max_chapters=max_chapters)
    source_lookup: Dict[str, Dict[str, Any]] = {}
    cards: List[Dict[str, Any]] = []
    buckets: Dict[str, int] = {}
    seen_refs: set[str] = set()
    for view in views:
        for source in _as_list(view.get("source_registry_slice")):
            if isinstance(source, dict):
                source_id = str(source.get("run_source_id") or source.get("source_id") or "").strip()
                if source_id:
                    source_lookup[source_id] = source
        for fact in _as_list(view.get("usable_fact_cards")):
            if not isinstance(fact, dict):
                continue
            evidence_id = str(fact.get("fact_id") or fact.get("evidence_id") or "").strip()
            if not evidence_id or evidence_id in seen_refs:
                continue
            requirement_id = str(fact.get("requirement_id") or "").strip()
            chapter_id = requirement_chapter_lookup.get(requirement_id) or str(fact.get("chapter_id") or "").strip()
            if chapter_filter:
                resolved_chapter_id = resolve_chapter_id(chapter_filter, chapter_id)
                if not resolved_chapter_id:
                    continue
                chapter_id = resolved_chapter_id
            if not chapter_id:
                chapter_id = "artifact_ledger"
            if buckets.get(chapter_id, 0) >= max_per_chapter:
                continue
            source_id = str(fact.get("source_id") or "").strip()
            source = source_lookup.get(source_id, {})
            fact_text = _compact(fact.get("fact"), 360)
            if not fact_text:
                continue
            lineage = {
                key: value
                for key, value in {
                    "chapter_id": chapter_id,
                    "requirement_id": requirement_id,
                    "fact_id": evidence_id,
                    "source_id": source_id,
                    "artifact_ledger_run_id": run_id,
                }.items()
                if value
            }
            cards.append(
                {
                    "evidence_id": evidence_id,
                    "chapter_id": chapter_id,
                    "hypothesis_id": "",
                    "requirement_id": requirement_id,
                    "analysis_role": str(fact.get("analysis_role") or "").strip(),
                    "analysis_eligible": True,
                    "allowed_use": str(fact.get("allowed_use") or "").strip(),
                    "source_id": source_id,
                    "search_task_id": "",
                    "lineage": lineage,
                    "public_fact_card": {
                        "fact": fact_text,
                        "fact_type": str(fact.get("analysis_role") or fact.get("allowed_use") or "").strip(),
                        "source_level": str(fact.get("source_level") or source.get("source_level") or "").strip(),
                    },
                    "distilled_fact": fact_text,
                    "fact": fact_text,
                    "metric": _compact(fact.get("metric"), 100),
                    "value": _compact(fact.get("value"), 100),
                    "unit": _compact(fact.get("unit"), 60),
                    "period": _compact(fact.get("period") or source.get("published_at"), 80),
                    "source_level": str(fact.get("source_level") or source.get("source_level") or "").strip().upper(),
                    "proof_role": str(fact.get("analysis_role") or "").strip().lower(),
                    "source_verification_status": str(source.get("verification_status") or "").strip(),
                    "can_support": [],
                    "cannot_support": [],
                    "proof_strength": "",
                    "repair_need": [],
                    "source_title": _compact(source.get("title"), 160),
                    "source_url": str(source.get("canonical_url") or source.get("url") or "").strip(),
                }
            )
            seen_refs.add(evidence_id)
            buckets[chapter_id] = buckets.get(chapter_id, 0) + 1
            if len(buckets) >= max_chapters and all(count >= max_per_chapter for count in buckets.values()):
                break
    return cards


def build_llm_analysis_input(evidence_package: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
    max_chapters = _env_int("BRAIN_LLM_ANALYSIS_MAX_CHAPTERS", 6, min_value=1, max_value=12)
    max_per_chapter = _env_int("BRAIN_LLM_ANALYSIS_MAX_EVIDENCE_PER_CHAPTER", 12, min_value=3, max_value=30)
    diagnostics = _chapter_filter_for_llm(evidence_package, max_chapters=max_chapters) or _as_dict(fallback.get("chapter_evidence_diagnostics"))
    return {
        "query": fallback.get("query") or evidence_package.get("query") or "",
        "research_plan": _research_plan(evidence_package),
        "report_contract": _as_dict(evidence_package.get("report_contract")) or _as_dict(evidence_package.get("report_plan")),
        "chapter_evidence_diagnostics": dict(list(diagnostics.items())[:max_chapters]),
        "fact_cards": _evidence_cards_for_llm(
            evidence_package,
            max_chapters=max_chapters,
            max_per_chapter=max_per_chapter,
        ),
    }


def _text_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item or "").strip() for item in value if str(item or "").strip()]
    if str(value or "").strip():
        return [str(value).strip()]
    return []


def _chapter_payload_metadata(
    chapter_id: str,
    diagnostics: Dict[str, Any],
    fallback: Dict[str, Any],
) -> Dict[str, str]:
    diagnostic = _as_dict(diagnostics.get(chapter_id))
    title = (
        diagnostic.get("chapter_title")
        or diagnostic.get("title")
        or diagnostic.get("chapter_question")
        or ""
    )
    question = (
        diagnostic.get("chapter_question")
        or diagnostic.get("question")
        or diagnostic.get("chapter_title")
        or title
        or chapter_id
    )
    if not title:
        for chapter in _as_list(_as_dict(fallback.get("report_insight_package")).get("chapters")) + _as_list(fallback.get("chapter_insights")):
            if not isinstance(chapter, dict):
                continue
            if str(chapter.get("chapter_id") or "") == chapter_id:
                title = str(chapter.get("chapter_title") or chapter.get("chapter_question") or "").strip()
                question = str(chapter.get("chapter_question") or title or question).strip()
                break
    return {
        "chapter_id": chapter_id,
        "chapter_title": _compact(title or question or chapter_id, 120),
        "chapter_question": _compact(question or title or chapter_id, 180),
    }


def _compact_llm_fact_card(card: Dict[str, Any], *, max_fact_chars: int) -> Dict[str, Any]:
    public_card = _as_dict(card.get("public_fact_card"))
    block_affinity = _text_list(public_card.get("block_affinity") or card.get("block_affinity"))
    fact_type = str(public_card.get("fact_type") or card.get("fact_type") or card.get("proof_role") or "").strip()
    return {
        "evidence_id": str(card.get("evidence_id") or "").strip(),
        "hypothesis_id": str(card.get("hypothesis_id") or "").strip(),
        "requirement_id": str(card.get("requirement_id") or "").strip(),
        "analysis_role": str(card.get("analysis_role") or "").strip(),
        "analysis_eligible": bool(card.get("analysis_eligible")),
        "allowed_use": str(card.get("allowed_use") or "").strip(),
        "lineage": _as_dict(card.get("lineage")),
        "distilled_fact": _compact(card.get("distilled_fact") or card.get("fact"), max_fact_chars),
        "fact_type": fact_type,
        "source_level": str(card.get("source_level") or "").strip().upper(),
        "source_verification_status": str(card.get("source_verification_status") or "").strip(),
        "proof_role": str(card.get("proof_role") or "").strip(),
        "block_affinity": block_affinity,
        "metric": _compact(card.get("metric"), 80),
        "value": _compact(card.get("value"), 80),
        "unit": _compact(card.get("unit"), 40),
        "period": _compact(card.get("period"), 80),
        "source_title": _compact(card.get("source_title"), 120),
        "source_url": str(card.get("source_url") or "").strip(),
    }


def _requirements_by_chapter_for_llm(evidence_package: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    contract = _as_dict(evidence_package.get("report_contract")) or _as_dict(evidence_package.get("report_plan"))
    requirements = _as_list(_as_dict(contract.get("evidence_requirements")).get("requirements"))
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for item in requirements:
        requirement = _as_dict(item)
        chapter_id = str(requirement.get("chapter_id") or "").strip()
        requirement_id = str(requirement.get("requirement_id") or "").strip()
        if not chapter_id or not requirement_id:
            continue
        grouped.setdefault(chapter_id, []).append(
            {
                "requirement_id": requirement_id,
                "hypothesis_id": str(requirement.get("hypothesis_id") or "").strip(),
                "proof_role": str(requirement.get("proof_role") or "").strip(),
                "required_fields": _as_list(requirement.get("required_fields")),
                "min_source_level": str(requirement.get("min_source_level") or "").strip(),
                "claim_strength_ceiling": str(requirement.get("claim_strength_ceiling") or "").strip(),
            }
        )
    return grouped


def build_llm_analysis_input_v2(evidence_package: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
    max_chapters = _env_int("BRAIN_LLM_ANALYSIS_MAX_CHAPTERS", 8, min_value=1, max_value=30)
    max_per_chapter = _env_int("BRAIN_LLM_ANALYSIS_MAX_FACTS_PER_CHAPTER", 8, min_value=1, max_value=30)
    max_fact_chars = _env_int("BRAIN_LLM_ANALYSIS_MAX_FACT_CHARS", 260, min_value=40, max_value=800)
    diagnostics = _chapter_filter_for_llm(evidence_package, max_chapters=max_chapters) or _as_dict(fallback.get("chapter_evidence_diagnostics"))
    cards = _evidence_cards_for_llm(
        evidence_package,
        max_chapters=max_chapters,
        max_per_chapter=max_per_chapter,
    )
    requirements_by_chapter = _requirements_by_chapter_for_llm(evidence_package)
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    chapter_order: List[str] = []
    for card in cards:
        chapter_id = str(card.get("chapter_id") or "").strip()
        if not chapter_id:
            continue
        if chapter_id not in grouped:
            grouped[chapter_id] = []
            chapter_order.append(chapter_id)
        if len(grouped[chapter_id]) < max_per_chapter:
            compact_card = _compact_llm_fact_card(card, max_fact_chars=max_fact_chars)
            if compact_card.get("evidence_id") and compact_card.get("distilled_fact"):
                grouped[chapter_id].append(compact_card)
    chapters: List[Dict[str, Any]] = []
    for chapter_id in chapter_order[:max_chapters]:
        fact_cards = grouped.get(chapter_id) or []
        if not fact_cards:
            continue
        metadata = _chapter_payload_metadata(chapter_id, diagnostics, fallback)
        chapters.append(
            {
                **metadata,
                "evidence_requirements": requirements_by_chapter.get(chapter_id, []),
                "allowed_evidence_ids": [item["evidence_id"] for item in fact_cards],
                "fact_cards": fact_cards,
            }
        )
    return {
        "query": fallback.get("query") or evidence_package.get("query") or "",
        "chapters": chapters,
    }


def _llm_analysis_run_id(evidence_package: Dict[str, Any]) -> str:
    raw = str(
        evidence_package.get("run_id")
        or evidence_package.get("report_id")
        or os.getenv("REPORT_RUN_ID")
        or "default"
    )
    return re.sub(r"[^A-Za-z0-9_.\-\u4e00-\u9fff]+", "_", raw).strip("_") or "default"


def _llm_analysis_cache_path(evidence_package: Dict[str, Any], chapter_id: str, chapter_input: Dict[str, Any]) -> Path:
    root = Path(os.getenv("BRAIN_LLM_ANALYSIS_CACHE_PATH") or "output/cache/llm_analysis")
    run_id = _llm_analysis_run_id(evidence_package)
    digest = hashlib.sha256(
        json.dumps(chapter_input, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:24]
    safe_chapter = re.sub(r"[^A-Za-z0-9_.\-\u4e00-\u9fff]+", "_", str(chapter_id or "chapter")).strip("_") or "chapter"
    return root / run_id / safe_chapter / f"{digest}.json"


def _load_llm_analysis_cache(path: Path) -> Optional[Dict[str, Any]]:
    if not _env_flag("BRAIN_LLM_ANALYSIS_CACHE_ENABLED", True):
        return None
    try:
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            cached = json.load(handle)
        return _as_dict(cached)
    except Exception:
        return None


def _store_llm_analysis_cache(path: Path, payload: Dict[str, Any]) -> None:
    if not _env_flag("BRAIN_LLM_ANALYSIS_CACHE_ENABLED", True):
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
        tmp_path.replace(path)
    except Exception:
        return


def _llm_chapter_system_prompt() -> str:
    return """
You are an evidence-to-claim analyst for a Chinese industry research report.
You analyze one chapter at a time.
Use only the provided fact_cards. Do not invent facts, numbers, companies, sources, URLs, or citations.
Return strict JSON with: chapter_id, claim_units, analysis_limits.

Each claim_unit must include:
- claim: one complete Chinese judgment sentence;
- requirement_ids: requirement_id values from the input fact_cards when available;
- fact_ids: exact input evidence_id values supporting the claim;
- source_ids: source_id values from cited fact_cards when available;
- hypothesis_id: hypothesis_id from the chapter or input fact_cards when available;
- used_evidence_ids: exact evidence_id values from the input;
- evidence_basis: concise evidence sentences derived only from input facts;
- reasoning_chain: mechanism explanation sentences;
- limitation_boundary: specific boundary conditions;
- claim_strength: strong, moderate, directional, or limited_evidence;
- claim_strength_ceiling: maximum claim strength allowed by the cited fact cards;
- analysis_role: claimable, directional, contextual, counter, metric, case, or technology;
- source_support_map: object mapping claim/mechanism/boundary to used evidence ids;
- paragraph_seed: one concise paragraph seed for downstream composition;
- block_affinity: metric_reconciliation, case_comparison, technology_maturity, risk_trigger, or integrated_signal.

A/B verified evidence may support strong or moderate claims.
B/C traceable or qualitative evidence may still support a directional or limited_evidence claim.
claim_strength must never exceed claim_strength_ceiling.
If requirement_ids cannot be derived from the cited fact_cards, still make a directional or limited_evidence claim with requirement_ids left empty — do NOT reject it; the binding is carried by used_evidence_ids.
Missing hard metrics (market size, growth rate, adoption rate) is NOT a reason to abstain: make a directional/limited_evidence claim grounded in the qualitative signal and state the boundary in limitation_boundary instead.
Only return no claim_units when the chapter genuinely has NO relevant evidence at all — not merely because verifiable numbers are absent.
Aim for role diversity within the chapter (this materially improves report completeness), but NEVER fabricate to fill a role:
- when the evidence contains a risk, failure, limitation or contradicting signal, surface it as a counter claim (analysis_role=counter, block_affinity=risk_trigger);
- when the evidence supports an actionable judgment, set a concrete decision_use on that claim (what a decision-maker should do or watch).
Forbidden public claim language: 证据不足, 建议补证, 正文应以, 方向性观察, 后续验证, 继续校准.
""".strip()


def synthesize_chapter_with_llm_analysis(
    *,
    evidence_package: Dict[str, Any],
    chapter_payload: Dict[str, Any],
    llm_config: Dict[str, Any],
) -> Dict[str, Any]:
    if call_openai_compatible_json is None or normalize_llm_config is None:
        raise RuntimeError("LLM analysis dependencies are unavailable.")
    config = dict(llm_config or {})
    config["timeout"] = float(os.getenv("BRAIN_LLM_ANALYSIS_TIMEOUT_SECONDS", config.get("timeout") or 120) or 120)
    chapter_id = str(chapter_payload.get("chapter_id") or "chapter")
    user_payload = {
        "query": _compact(evidence_package.get("query"), 180),
        "chapter": chapter_payload,
    }
    normalized_config = normalize_llm_config(config) if normalize_llm_config is not None else {}
    cache_input = {
        **user_payload,
        "prompt_version": "llm_analysis_v2_2026_06_roles",
        "model": normalized_config.get("model", ""),
    }
    cache_path = _llm_analysis_cache_path(evidence_package, chapter_id, cache_input)
    cached = _load_llm_analysis_cache(cache_path)
    if cached:
        result = _as_dict(cached.get("result"))
        result["_llm_cache_hit"] = True
        result["_llm_cache_path"] = str(cache_path)
        return result
    started = time.time()
    response = call_openai_compatible_json(
        config=config,
        system_prompt=_llm_chapter_system_prompt(),
        user_payload=user_payload,
    )
    raw_payload = _as_dict(response.get("payload"))
    if "chapter_synthesis" not in raw_payload:
        raw_payload = {
            "chapter_synthesis": [
                {
                    "chapter_id": raw_payload.get("chapter_id") or chapter_id,
                    "chapter_title": chapter_payload.get("chapter_title"),
                    "claim_units": _as_list(raw_payload.get("claim_units")),
                    "analysis_limits": _as_list(raw_payload.get("analysis_limits")),
                }
            ],
            "analysis_limits": _as_list(raw_payload.get("analysis_limits")),
        }
    raw_payload["_llm_usage"] = response.get("usage", {})
    raw_payload["_llm_model"] = normalized_config.get("model", "")
    raw_payload["_llm_cache_hit"] = False
    raw_payload["_llm_elapsed_seconds"] = round(time.time() - started, 3)
    raw_payload["_llm_cache_path"] = str(cache_path)
    _store_llm_analysis_cache(
        cache_path,
        {
            "compact_input": cache_input,
            "raw_output": response,
            "result": raw_payload,
            "model": raw_payload.get("_llm_model"),
            "usage": raw_payload.get("_llm_usage"),
            "created_at": time.time(),
        },
    )
    return raw_payload


def _is_transient_llm_error(exc: Exception) -> bool:
    """Heuristic: should this per-chapter LLM failure be retried?

    Network blips / timeouts / rate limits / 5xx are transient — one DNS hiccup
    should not zero out the whole analysis stage (exactly what knocked out a live
    baseline run: 9/9 chapters lost to one ``getaddrinfo failed``). Parse and
    validation errors are *not* retried.
    """
    text = f"{type(exc).__name__}: {exc}".lower()
    # Parse / format / schema failures are NOT transient even when wrapped in
    # LLMCallError ("LLM response is not valid JSON"): retrying rarely helps and
    # just wastes calls. Check these first so they override the generic markers.
    non_transient_markers = (
        "not valid json", "invalid json", "jsondecode", "json decode",
        "unterminated", "expecting value", "expecting property",
        "could not parse", "failed to parse", "schema", "validation",
    )
    if any(marker in text for marker in non_transient_markers):
        return False
    transient_markers = (
        "timeout", "timed out", "getaddrinfo", "temporarily", "temporary",
        "connection", "reset by peer", "econnreset", "rate limit", "ratelimit",
        "too many requests", "429", "500", "502", "503", "504",
        "llmcallerror", "urlopen", "ssl", "unavailable", "overloaded",
    )
    return any(marker in text for marker in transient_markers)


def synthesize_with_llm_analysis_v2(
    *,
    evidence_package: Dict[str, Any],
    fallback: Dict[str, Any],
    llm_config: Dict[str, Any],
) -> Dict[str, Any]:
    if llm_config_is_ready is None:
        raise RuntimeError("LLM analysis dependencies are unavailable.")
    config = dict(llm_config or {})
    config["timeout"] = float(os.getenv("BRAIN_LLM_ANALYSIS_TIMEOUT_SECONDS", config.get("timeout") or 120) or 120)
    if not llm_config_is_ready(config):
        raise RuntimeError("LLM config is incomplete.")
    llm_input = build_llm_analysis_input_v2(evidence_package, fallback)
    chapters = _as_list(llm_input.get("chapters"))
    concurrency = _env_int("BRAIN_LLM_ANALYSIS_CONCURRENCY", 3, min_value=1, max_value=8)
    raw_chapters: List[Dict[str, Any]] = []
    chapter_results: List[Dict[str, Any]] = []
    usage: Dict[str, Any] = {}
    cache_hits = 0
    failed = 0
    retry_total = 0
    max_retries = _env_int("BRAIN_LLM_ANALYSIS_MAX_RETRIES", 2, min_value=0, max_value=5)
    retry_base_seconds = float(os.getenv("BRAIN_LLM_ANALYSIS_RETRY_BASE_SECONDS", "0.5") or 0.5)

    def worker(chapter_payload: Dict[str, Any]) -> Dict[str, Any]:
        last_error = ""
        for attempt in range(max_retries + 1):
            try:
                return {
                    "chapter_id": chapter_payload.get("chapter_id"),
                    "payload": synthesize_chapter_with_llm_analysis(
                        evidence_package=evidence_package,
                        chapter_payload=chapter_payload,
                        llm_config=config,
                    ),
                    "error": "",
                    "attempts": attempt + 1,
                }
            except Exception as exc:
                last_error = str(exc)
                if attempt < max_retries and _is_transient_llm_error(exc):
                    time.sleep(retry_base_seconds * (2 ** attempt))
                    continue
                return {
                    "chapter_id": chapter_payload.get("chapter_id"),
                    "payload": {},
                    "error": last_error,
                    "attempts": attempt + 1,
                }
        return {"chapter_id": chapter_payload.get("chapter_id"), "payload": {}, "error": last_error, "attempts": max_retries + 1}

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_map = {executor.submit(worker, chapter): chapter for chapter in chapters}
        for future in as_completed(future_map):
            result = future.result()
            retry_total += max(0, int(result.get("attempts") or 1) - 1)
            payload = _as_dict(result.get("payload"))
            error = str(result.get("error") or "")
            if error:
                failed += 1
                chapter_results.append({"chapter_id": result.get("chapter_id"), "status": "error", "error": error})
                continue
            if payload.get("_llm_cache_hit"):
                cache_hits += 1
            usage_items = _as_dict(payload.get("_llm_usage"))
            for key, value in usage_items.items():
                if isinstance(value, (int, float)):
                    usage[key] = usage.get(key, 0) + value
            for chapter in _as_list(payload.get("chapter_synthesis")):
                if isinstance(chapter, dict):
                    raw_chapters.append(chapter)
            chapter_results.append(
                {
                    "chapter_id": result.get("chapter_id"),
                    "status": "cached" if payload.get("_llm_cache_hit") else "called",
                    "claim_count": sum(len(_as_list(ch.get("claim_units"))) for ch in _as_list(payload.get("chapter_synthesis")) if isinstance(ch, dict)),
                    "cache_path": payload.get("_llm_cache_path"),
                }
            )
    return {
        "chapter_synthesis": raw_chapters,
        "cross_chapter_conflicts": [],
        "evidence_repair_priorities": [],
        "rewrite_priorities": [],
        "_llm_usage": usage,
        "_llm_model": (normalize_llm_config(config).get("model", "") if normalize_llm_config is not None else ""),
        "_llm_input_version": "v2",
        "_llm_analysis_mode": "per_chapter",
        "_llm_chapter_results": chapter_results,
        "_llm_cache_hit_count": cache_hits,
        "_llm_failed_chapter_count": failed,
        "_llm_submitted_chapter_count": len(chapters),
        "_llm_retry_count": retry_total,
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
You are an evidence-to-analysis agent for a Chinese industry research report.
Use only input fact_cards. Do not invent facts, numbers, sources, companies, URLs, or citations.
Return one JSON object with chapter_synthesis, cross_chapter_conflicts, evidence_repair_priorities, and rewrite_priorities.

For each chapter_synthesis item:
- include chapter_id and chapter_title when available;
- include at most 2-3 claim_units;
- every claim_unit must include claim, used_evidence_ids, evidence_basis, reasoning_chain, limitation_boundary, and claim_strength;
- used_evidence_ids must be exact evidence_id values from input fact_cards;
- A/B readpage_verified or document_verified cards may support moderate/strong claims;
- B/C or directional cards may only support directional/limited claims.

Never output internal diagnostics as public claims. Forbidden claim language includes: 证据不足, 建议补证, 正文应以, 方向性观察, 后续验证, 可追溯来源继续校准.
If a chapter lacks usable evidence, put the limitation in analysis_limits and do not create a claim_unit for that chapter.
Output public analysis in Chinese unless the evidence itself is English-only.
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


def _refs_from_llm_chapter(chapter: Dict[str, Any], valid_refs: set[str]) -> List[str]:
    refs: List[str] = []
    for key in ("used_evidence_ids", "evidence_refs", "supporting_evidence_refs", "supporting_evidence"):
        refs.extend(str(ref or "").strip() for ref in _as_list(chapter.get(key)) if str(ref or "").strip())
    cleaned = _dedupe([ref for ref in refs if valid_refs and ref in valid_refs])
    return cleaned[:12]


NON_DROPPING_CLAIM_ISSUES = {
    "claim_support_needs_repair",
    "decision_claim_downgraded_no_valid_ref",
    "llm_numeric_claim_incomplete_metric_fact",
}

REPAIRABLE_CLAIM_ISSUES = {
    "claim_support_needs_repair",
}


def _claim_drop_issue_counts(issue_counts: Dict[str, int]) -> Dict[str, int]:
    drop_counts: Dict[str, int] = {}
    for issue_type, count in issue_counts.items():
        if issue_type in NON_DROPPING_CLAIM_ISSUES:
            continue
        is_drop = (
            issue_type.startswith("llm_claim")
            or issue_type in {
                "invalid_llm_evidence_ref",
                "claim_support_validator_unavailable",
                "llm_claim_unit_missing_requirement_ids",
                "llm_claim_strength_exceeds_ceiling",
            }
        )
        if is_drop and count > 0:
            drop_counts[issue_type] = count
    return drop_counts


def _claim_repair_issue_counts(issue_counts: Dict[str, int]) -> Dict[str, int]:
    return {
        issue_type: count
        for issue_type, count in issue_counts.items()
        if issue_type in REPAIRABLE_CLAIM_ISSUES and count > 0
    }


def _correctness_filter_summary(
    *,
    raw_claim_count: int,
    usable_claim_count: int,
    issue_counts: Dict[str, int],
) -> Dict[str, Any]:
    drop_issue_counts = _claim_drop_issue_counts(issue_counts)
    deferred_issue_counts = _claim_repair_issue_counts(issue_counts)
    dropped_by_filter_count = sum(drop_issue_counts.values())
    deferred_by_filter_count = sum(deferred_issue_counts.values())
    min_usable_claims = _env_int(
        "BRAIN_LLM_ANALYSIS_THIN_REPORT_MIN_USABLE_CLAIMS",
        3,
        min_value=1,
        max_value=50,
    )
    required_usable = min(min_usable_claims, max(raw_claim_count, 1))
    thin_report_risk = bool(
        raw_claim_count > 0
        and usable_claim_count < required_usable
        and (dropped_by_filter_count > 0 or deferred_by_filter_count > 0)
    )
    recommended_mode = "normal"
    if thin_report_risk:
        if deferred_by_filter_count > 0:
            recommended_mode = "repair_then_rebuild"
        else:
            recommended_mode = "insufficient_analysis_stub" if usable_claim_count <= 0 else "limited_evidence_draft"
    return {
        "raw_claim_count": raw_claim_count,
        "usable_claim_count": usable_claim_count,
        "dropped_by_filter_count": dropped_by_filter_count,
        "deferred_by_filter_count": deferred_by_filter_count,
        "drop_issue_counts": drop_issue_counts,
        "deferred_issue_counts": deferred_issue_counts,
        "min_usable_claims": min_usable_claims,
        "thin_report_risk": thin_report_risk,
        "recommended_mode": recommended_mode,
    }


def _support_gap_type(support_payload: Dict[str, Any]) -> str:
    unsupported_numbers = _as_list(support_payload.get("unsupported_numbers"))
    unsupported_entities = _as_list(support_payload.get("unsupported_entities"))
    unsupported_terms = _as_list(support_payload.get("unsupported_terms"))
    if "no_cited_fact_cards" in {str(item) for item in unsupported_terms}:
        return "claim_missing_cited_fact_cards"
    if unsupported_numbers or unsupported_entities:
        return "claim_support_entity_or_metric_mismatch"
    return "claim_support_anchor_mismatch"


def _support_required_fields(support_payload: Dict[str, Any]) -> List[str]:
    fields: List[str] = ["source"]
    if _as_list(support_payload.get("unsupported_numbers")):
        fields.extend(["metric", "value", "unit", "period"])
    if _as_list(support_payload.get("unsupported_entities")):
        fields.append("entity_match")
    unsupported_terms = _as_list(support_payload.get("unsupported_terms"))
    if unsupported_terms and len(fields) <= 1:
        fields.extend(["supporting_fact", "source_text"])
    return _dedupe(fields)


def _claim_support_repair_priority(
    *,
    chapter: Dict[str, Any],
    unit: Dict[str, Any],
    claim_text: str,
    refs: Sequence[str],
    cited_cards: Sequence[Dict[str, Any]],
    support_payload: Dict[str, Any],
) -> Dict[str, Any]:
    requirement_ids = _dedupe(
        [
            *[str(req or "").strip() for req in _as_list(unit.get("requirement_ids")) if str(req or "").strip()],
            *[
                str(card.get("requirement_id") or "").strip()
                for card in cited_cards
                if str(card.get("requirement_id") or "").strip()
            ],
        ]
    )
    source_ids = _dedupe(
        [
            str(card.get("source_id") or _as_dict(card.get("lineage")).get("source_id") or "").strip()
            for card in cited_cards
            if str(card.get("source_id") or _as_dict(card.get("lineage")).get("source_id") or "").strip()
        ]
    )
    gap_type = _support_gap_type(support_payload)
    claim_id = str(unit.get("claim_id") or "").strip()
    return {
        "schema_version": "claim_support_repair_priority_v1",
        "gap_type": gap_type,
        "gap_id": f"{chapter.get('chapter_id') or 'chapter'}_{claim_id or 'claim'}_{gap_type}",
        "chapter_id": chapter.get("chapter_id"),
        "claim_id": claim_id,
        "requirement_ids": requirement_ids,
        "evidence_refs": list(refs),
        "source_ids": source_ids,
        "claim": _compact(claim_text, 360),
        "unsupported_terms": _as_list(support_payload.get("unsupported_terms")),
        "unsupported_numbers": _as_list(support_payload.get("unsupported_numbers")),
        "unsupported_entities": _as_list(support_payload.get("unsupported_entities")),
        "required_fields": _support_required_fields(support_payload),
        "proof_role": unit.get("proof_role") or "support",
        "success_criteria": "Only rebuild this claim when cited fact cards directly support the claim text and all required fields are present.",
        "reject_if": ["snippet_only", "no_source_url", "off_topic_source", "unsupported_entity_or_number"],
        "allowed_for_writing": False,
        "writing_permission": "not_allowed_until_repaired",
        "recommended_action": "repair_evidence_binding_then_rebuild_claim",
    }


def _semantic_judge_status(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"yes", "true", "supported", "support", "pass", "passed"}:
        return "supported"
    if text in {"partial", "partially_supported", "partially supported"}:
        return "partial"
    if text in {"no", "false", "unsupported", "not_supported", "fail", "failed"}:
        return "unsupported"
    return text or "unknown"


def _semantic_judge_accepts(result: Dict[str, Any]) -> bool:
    if not result:
        return False
    if result.get("supports_claim") is False:
        return False
    status = _semantic_judge_status(result.get("status") or result.get("verdict") or result.get("support_status"))
    return status == "supported"


def _semantic_judge_enabled(llm_config: Optional[Dict[str, Any]]) -> bool:
    return bool(llm_config) and _env_flag("BRAIN_ENABLE_LLM_SEMANTIC_JUDGE", True)


def _semantic_judge_fail_closed() -> bool:
    return _env_flag("BRAIN_LLM_SEMANTIC_JUDGE_FAIL_CLOSED", True)


SEMANTIC_JUDGE_PROMPT_VERSION = "semantic_claim_support_judge_v1_2026_06_strict"


def _semantic_judge_fact_payload(cards: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for card in cards[:12]:
        item = _as_dict(card)
        source = _as_dict(item.get("source"))
        payload.append(
            {
                "evidence_id": str(item.get("evidence_id") or item.get("id") or "").strip(),
                "fact": _compact(
                    item.get("distilled_fact")
                    or item.get("fact")
                    or item.get("clean_fact")
                    or item.get("content")
                    or item.get("summary"),
                    420,
                ),
                "metric": _compact(item.get("metric") or item.get("indicator"), 120),
                "value": _compact(item.get("value") or item.get("display_value") or item.get("numeric_value"), 80),
                "unit": _compact(item.get("unit") or item.get("numeric_unit"), 40),
                "period": _compact(item.get("period") or item.get("time_or_scope") or item.get("date"), 80),
                "source_title": _compact(item.get("source_title") or source.get("title"), 180),
                "source_level": str(item.get("source_level") or source.get("source_level") or "").strip(),
                "allowed_use": str(item.get("allowed_use") or "").strip(),
                "source_verification_status": str(
                    item.get("source_verification_status") or item.get("verification_status") or source.get("verification_status") or ""
                ).strip(),
            }
        )
    return payload


def _semantic_judge_system_prompt() -> str:
    return """
You are a strict evidence support judge for a publishable research report.
Decide whether the cited fact cards semantically support the claim.

Rules:
- Use only the provided fact cards; never use outside knowledge.
- A source merely mentioning the same topic is not support.
- Numbers, dates, companies, scope, causality, competitive claims, and risk claims must be directly grounded in the cited facts.
- Return unsupported when support is only partial, adjacent, speculative, or too generic.
- Output one JSON object only: {"status":"supported|unsupported","reason":"...","confidence":0.0-1.0,"unsupported_terms":[]}.
""".strip()


def _semantic_judge_cache_path(*, claim_text: str, cited_cards: Sequence[Dict[str, Any]], llm_config: Dict[str, Any]) -> Path:
    root = Path(os.getenv("BRAIN_LLM_SEMANTIC_JUDGE_CACHE_PATH") or "output/cache/semantic_judge")
    normalized = normalize_llm_config(llm_config) if normalize_llm_config is not None else {}
    cache_input = {
        "prompt_version": SEMANTIC_JUDGE_PROMPT_VERSION,
        "model": normalized.get("model") or llm_config.get("model") or "",
        "claim": claim_text,
        "cited_fact_cards": _semantic_judge_fact_payload(cited_cards),
    }
    digest = hashlib.sha256(
        json.dumps(cache_input, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:32]
    return root / f"{digest}.json"


def _load_semantic_judge_cache(path: Path) -> Optional[Dict[str, Any]]:
    if not _env_flag("BRAIN_LLM_SEMANTIC_JUDGE_CACHE_ENABLED", True):
        return None
    try:
        if not path.exists():
            return None
        return _as_dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return None


def _store_semantic_judge_cache(path: Path, payload: Dict[str, Any]) -> None:
    if not _env_flag("BRAIN_LLM_SEMANTIC_JUDGE_CACHE_ENABLED", True):
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        tmp_path.replace(path)
    except Exception:
        return


def _llm_semantic_claim_support_judge(
    *,
    claim_text: str,
    cited_cards: Sequence[Dict[str, Any]],
    chapter_id: Any = "",
    claim_id: Any = "",
    llm_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not _semantic_judge_enabled(llm_config):
        return {"status": "skipped_disabled_or_missing_config"}
    if call_openai_compatible_json is None or llm_config_is_ready is None:
        return {"status": "error", "reason": "semantic_judge_dependencies_unavailable"}
    config = dict(llm_config or {})
    config["timeout"] = float(os.getenv("BRAIN_LLM_SEMANTIC_JUDGE_TIMEOUT_SECONDS", config.get("timeout") or 90) or 90)
    config["temperature"] = 0
    if not llm_config_is_ready(config):
        return {"status": "error", "reason": "semantic_judge_config_incomplete"}
    cache_path = _semantic_judge_cache_path(claim_text=claim_text, cited_cards=cited_cards, llm_config=config)
    cached = _load_semantic_judge_cache(cache_path)
    if cached:
        result = _as_dict(cached.get("result") or cached)
        result["cache_hit"] = True
        result["cache_path"] = str(cache_path)
        return result
    try:
        response = call_openai_compatible_json(
            config=config,
            system_prompt=_semantic_judge_system_prompt(),
            user_payload={
                "schema_version": "semantic_claim_support_judge_v1",
                "chapter_id": str(chapter_id or ""),
                "claim_id": str(claim_id or ""),
                "claim": claim_text,
                "cited_fact_cards": _semantic_judge_fact_payload(cited_cards),
                "instruction": "Return supported only if the cited facts directly support the full claim.",
            },
        )
    except Exception as exc:
        return {"status": "error", "reason": str(exc)}
    result = _as_dict(response.get("payload"))
    status = _semantic_judge_status(result.get("status") or result.get("verdict") or result.get("support_status"))
    output = {
        "status": status,
        "reason": _compact(result.get("reason") or result.get("rationale") or "", 360),
        "confidence": result.get("confidence"),
        "unsupported_terms": _as_list(result.get("unsupported_terms")),
        "usage": response.get("usage") or {},
        "model": (normalize_llm_config(config).get("model", "") if normalize_llm_config is not None else ""),
        "cache_hit": False,
        "cache_path": str(cache_path),
    }
    _store_semantic_judge_cache(
        cache_path,
        {
            "schema_version": "semantic_judge_cache_v1",
            "prompt_version": SEMANTIC_JUDGE_PROMPT_VERSION,
            "result": output,
            "created_at": time.time(),
        },
    )
    return output


def validate_llm_analysis_output(
    payload: Dict[str, Any],
    evidence_package: Dict[str, Any],
    *,
    llm_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    input_cards = _evidence_cards_for_llm(
        evidence_package,
        max_chapters=_env_int("BRAIN_LLM_ANALYSIS_MAX_CHAPTERS", 12, min_value=1, max_value=100),
        max_per_chapter=_env_int("BRAIN_LLM_ANALYSIS_MAX_EVIDENCE_PER_CHAPTER", 30, min_value=1, max_value=200),
    )
    card_by_ref = {
        str(item.get("evidence_id") or "").strip(): item
        for item in input_cards
        if str(item.get("evidence_id") or "").strip()
    }
    requirement_contract_required = bool(
        any(str(item.get("requirement_id") or "").strip() for item in input_cards)
        or _as_list(
            _as_dict(
                _as_dict(_as_dict(evidence_package).get("report_contract"))
                .get("evidence_requirements")
            ).get("requirements")
        )
    )
    valid_refs = set(card_by_ref.keys())
    issues: List[Dict[str, Any]] = []
    chapters: List[Dict[str, Any]] = []
    valid_examples: List[Dict[str, Any]] = []
    rejected_examples: List[Dict[str, Any]] = []
    deferred_examples: List[Dict[str, Any]] = []
    claim_repair_priorities: List[Dict[str, Any]] = []
    semantic_judge_counts: Dict[str, int] = {}
    semantic_judge_usage: Dict[str, Any] = {}
    if not valid_refs:
        issue_counts = {"no_valid_input_evidence_refs": 1}
        return {
            "status": "invalid",
            "reason": "no_valid_input_evidence_refs",
            "issues": [{"type": "no_valid_input_evidence_refs"}],
            "chapter_synthesis": [],
            "valid_ref_count": 0,
            "usable_claim_count": 0,
            "dropped_claim_count": 0,
            "usable_chapter_count": 0,
            "llm_raw_chapter_count": 0,
            "llm_raw_claim_count": 0,
            "llm_validation_issue_counts": issue_counts,
            "llm_validation_issue_examples": [{"type": "no_valid_input_evidence_refs"}],
            "llm_valid_claim_examples": [],
            "llm_rejected_claim_examples": [],
            "llm_deferred_claim_examples": [],
            "claim_repair_priorities": [],
            "deferred_claim_count": 0,
            "llm_semantic_judge_counts": {},
            "llm_semantic_judge_usage": {},
            "correctness_filter_summary": _correctness_filter_summary(
                raw_claim_count=0,
                usable_claim_count=0,
                issue_counts=issue_counts,
            ),
        }
    raw_chapters = payload.get("chapter_synthesis")
    if isinstance(raw_chapters, dict):
        chapter_iterable = [
            {**_as_dict(value), "chapter_id": _as_dict(value).get("chapter_id") or key}
            for key, value in raw_chapters.items()
            if isinstance(value, dict)
        ]
    else:
        chapter_iterable = [
            _parse_structured_string(item)
            for item in _as_list(raw_chapters)
        ]
    raw_chapter_count = len([item for item in chapter_iterable if isinstance(item, dict)])
    raw_claim_count = sum(
        len(_as_list(_as_dict(item).get("claim_units")))
        for item in chapter_iterable
        if isinstance(item, dict)
    )
    for raw_chapter in chapter_iterable:
        if not isinstance(raw_chapter, dict):
            continue
        chapter = dict(raw_chapter)
        fact_refs = _refs_from_llm_chapter(chapter, valid_refs)
        if not _as_list(chapter.get("claim_units")):
            issues.append({"type": "llm_chapter_missing_claim_units", "chapter_id": chapter.get("chapter_id")})
        cleaned_units: List[Dict[str, Any]] = []
        for unit_index, raw_unit in enumerate(_as_list(chapter.get("claim_units")), start=1):
            if not isinstance(raw_unit, dict):
                continue
            unit = dict(raw_unit)
            refs = [
                str(ref or "").strip()
                for ref in _as_list(unit.get("used_evidence_ids"))
                if str(ref or "").strip()
            ]
            if not refs:
                issue = {"type": "llm_claim_missing_used_evidence_ids", "chapter_id": chapter.get("chapter_id")}
                issues.append(issue)
                if len(rejected_examples) < 5:
                    rejected_examples.append({**issue, "claim": _compact(unit.get("claim"), 160)})
                continue
            invalid_refs = [ref for ref in refs if valid_refs and ref not in valid_refs]
            if invalid_refs:
                issue = {"type": "invalid_llm_evidence_ref", "refs": invalid_refs, "chapter_id": chapter.get("chapter_id")}
                issues.append(issue)
                if len(rejected_examples) < 5:
                    rejected_examples.append({**issue, "claim": _compact(unit.get("claim"), 160)})
                continue
            refs = [ref for ref in refs if not valid_refs or ref in valid_refs]
            unit["supporting_evidence_refs"] = refs
            unit["evidence_refs"] = refs
            claim_text = _compact(unit.get("claim"), 360)
            if not claim_text or _has_internal_analysis_language(claim_text) or _is_generic_llm_claim(claim_text):
                issue = {"type": "llm_claim_unit_dropped_internal_or_generic", "chapter_id": chapter.get("chapter_id")}
                issues.append(issue)
                if len(rejected_examples) < 5:
                    rejected_examples.append({**issue, "claim": claim_text})
                continue
            evidence_basis = _as_list(unit.get("evidence_basis"))
            reasoning_chain = _as_list(unit.get("reasoning_chain"))
            limitation_boundary = _as_list(unit.get("limitation_boundary"))
            if not evidence_basis and str(unit.get("evidence_basis") or "").strip():
                evidence_basis = [unit.get("evidence_basis")]
            if not reasoning_chain and str(unit.get("reasoning_chain") or "").strip():
                reasoning_chain = [unit.get("reasoning_chain")]
            if not reasoning_chain and str(unit.get("reasoning") or "").strip():
                reasoning_chain = [unit.get("reasoning")]
            if not limitation_boundary and str(unit.get("limitation_boundary") or "").strip():
                limitation_boundary = [unit.get("limitation_boundary")]
            if not limitation_boundary and str(unit.get("counter_boundary") or unit.get("counter_evidence") or "").strip():
                limitation_boundary = [unit.get("counter_boundary") or unit.get("counter_evidence")]
            evidence_basis = [
                _public_normalize_analysis_text(_compact(item, 360))
                for item in evidence_basis
                if str(item or "").strip()
            ]
            reasoning_chain = [
                _public_normalize_analysis_text(_compact(item, 500))
                for item in reasoning_chain
                if str(item or "").strip()
            ]
            limitation_boundary = [
                _public_normalize_analysis_text(_compact(item, 360))
                for item in limitation_boundary
                if str(item or "").strip()
            ]
            if not evidence_basis or not reasoning_chain:
                issue = {"type": "llm_claim_unit_dropped_missing_basis_or_reasoning", "chapter_id": chapter.get("chapter_id")}
                issues.append(issue)
                if len(rejected_examples) < 5:
                    rejected_examples.append({**issue, "claim": claim_text})
                continue
            if any(_has_internal_analysis_language(item) or _is_generic_llm_claim(item) for item in [claim_text, *evidence_basis, *reasoning_chain, *limitation_boundary]):
                issue = {"type": "llm_claim_unit_dropped_internal_text", "chapter_id": chapter.get("chapter_id")}
                issues.append(issue)
                if len(rejected_examples) < 5:
                    rejected_examples.append({**issue, "claim": claim_text})
                continue
            unit["claim"] = claim_text
            unit["evidence_basis"] = evidence_basis
            unit["reasoning_chain"] = reasoning_chain
            unit["limitation_boundary"] = limitation_boundary
            unit["reasoning"] = "\n".join(reasoning_chain)
            unit["counter_boundary"] = "\n".join(limitation_boundary)
            unit["claim_id"] = str(unit.get("claim_id") or unit.get("id") or f"{chapter.get('chapter_id') or 'chapter'}_claim_{unit_index}")
            unit["supporting_fact_refs"] = refs
            unit["used_fact_refs"] = refs
            source_support_map = _as_dict(unit.get("source_support_map"))
            if not source_support_map:
                source_support_map = {"claim": refs, "mechanism": refs, "boundary": refs}
            else:
                source_support_map = {
                    "claim": _dedupe(_as_list(source_support_map.get("claim")) or refs),
                    "mechanism": _dedupe(_as_list(source_support_map.get("mechanism")) or refs),
                    "boundary": _dedupe(_as_list(source_support_map.get("boundary")) or refs),
                }
            unit["source_support_map"] = source_support_map
            strength_text = str(unit.get("claim_strength") or "").strip().lower()
            if not str(unit.get("analysis_role") or "").strip():
                unit["analysis_role"] = (
                    "claimable"
                    if strength_text in {"strong", "moderate"}
                    else ("contextual" if strength_text in {"contextual"} else "directional")
                )
            block_affinity = unit.get("block_affinity")
            if isinstance(block_affinity, str) and block_affinity.strip():
                unit["block_affinity"] = [block_affinity.strip()]
            unit["paragraph_seed"] = _compact(
                unit.get("paragraph_seed")
                or " ".join([claim_text, reasoning_chain[0] if reasoning_chain else "", limitation_boundary[0] if limitation_boundary else ""]),
                520,
            )
            decision_use = _compact(unit.get("decision_use"), 360)
            if _has_internal_analysis_language(decision_use):
                unit["decision_use"] = ""
            if str(unit.get("claim_status") or "").strip() == "decision_ready" and not refs:
                unit["claim_status"] = "directional"
                unit["missing_binding_reason"] = unit.get("missing_binding_reason") or "decision_ready claim lacked valid evidence refs"
                issues.append({"type": "decision_claim_downgraded_no_valid_ref", "chapter_id": chapter.get("chapter_id")})
            cited_cards = [_as_dict(card_by_ref.get(ref)) for ref in refs if _as_dict(card_by_ref.get(ref))]
            if validate_claim_supported_by_facts is None:
                issue = {
                    "type": "claim_support_validator_unavailable",
                    "chapter_id": chapter.get("chapter_id"),
                    "claim_id": unit.get("claim_id"),
                    "evidence_refs": refs,
                    "status": "validator_unavailable",
                }
                issues.append(issue)
                if len(rejected_examples) < 5:
                    rejected_examples.append({**issue, "claim": claim_text})
                continue
            else:
                support_result = validate_claim_supported_by_facts(claim_text, cited_cards)
                if not getattr(support_result, "supported", False):
                    support_payload = support_result.to_dict() if hasattr(support_result, "to_dict") else {}
                    repair_priority = _claim_support_repair_priority(
                        chapter=chapter,
                        unit=unit,
                        claim_text=claim_text,
                        refs=refs,
                        cited_cards=cited_cards,
                        support_payload=support_payload,
                    )
                    issue = {
                        "type": "claim_support_needs_repair",
                        "chapter_id": chapter.get("chapter_id"),
                        "claim_id": unit.get("claim_id"),
                        "evidence_refs": refs,
                        "gap_type": repair_priority.get("gap_type"),
                        "writing_permission": "not_allowed_until_repaired",
                        **support_payload,
                    }
                    issues.append(issue)
                    claim_repair_priorities.append(repair_priority)
                    if len(deferred_examples) < 8:
                        deferred_examples.append(
                            {
                                **issue,
                                "claim": claim_text,
                                "repair_priority": repair_priority,
                                "claim_status": "needs_repair",
                                "evidence_use_level": "diagnostic_only",
                            }
                        )
                    continue
                unit["claim_support_status"] = support_result.status
            semantic_judge = _llm_semantic_claim_support_judge(
                claim_text=claim_text,
                cited_cards=cited_cards,
                chapter_id=chapter.get("chapter_id"),
                claim_id=unit.get("claim_id"),
                llm_config=llm_config,
            )
            semantic_status = _semantic_judge_status(semantic_judge.get("status"))
            semantic_judge_counts[semantic_status] = semantic_judge_counts.get(semantic_status, 0) + 1
            if semantic_judge.get("cache_hit"):
                semantic_judge_counts["cache_hit"] = semantic_judge_counts.get("cache_hit", 0) + 1
            if not semantic_status.startswith("skipped"):
                semantic_judge_counts["attempted"] = semantic_judge_counts.get("attempted", 0) + 1
            if not semantic_judge.get("cache_hit"):
                for key, value in _as_dict(semantic_judge.get("usage")).items():
                    if isinstance(value, (int, float)):
                        semantic_judge_usage[key] = semantic_judge_usage.get(key, 0) + value
            unit["semantic_judge_status"] = semantic_status
            if semantic_status.startswith("skipped"):
                unit["semantic_judge_skipped_reason"] = semantic_status
            elif semantic_status == "error":
                issue = {
                    "type": "llm_claim_semantic_judge_error",
                    "chapter_id": chapter.get("chapter_id"),
                    "claim_id": unit.get("claim_id"),
                    "evidence_refs": refs,
                    "semantic_judge": semantic_judge,
                }
                issues.append(issue)
                if _semantic_judge_fail_closed():
                    if len(rejected_examples) < 5:
                        rejected_examples.append({**issue, "claim": claim_text})
                    continue
            elif not _semantic_judge_accepts(semantic_judge):
                issue = {
                    "type": "llm_claim_semantic_judge_unsupported",
                    "chapter_id": chapter.get("chapter_id"),
                    "claim_id": unit.get("claim_id"),
                    "evidence_refs": refs,
                    "semantic_judge": semantic_judge,
                }
                issues.append(issue)
                if len(rejected_examples) < 5:
                    rejected_examples.append({**issue, "claim": claim_text})
                continue
            else:
                unit["semantic_judge"] = {
                    key: value
                    for key, value in semantic_judge.items()
                    if key not in {"usage"}
                }
            if incomplete_metric_cards_for_numeric_claim is not None:
                metric_gaps = incomplete_metric_cards_for_numeric_claim(claim_text, cited_cards)
                if metric_gaps:
                    missing_fields = _dedupe(
                        [
                            str(field or "").strip()
                            for gap in metric_gaps
                            for field in _as_list(_as_dict(gap).get("missing_fields"))
                            if str(field or "").strip()
                        ]
                    )
                    issue = {
                        "type": "llm_numeric_claim_incomplete_metric_fact",
                        "chapter_id": chapter.get("chapter_id"),
                        "claim_id": unit.get("claim_id"),
                        "evidence_refs": refs,
                        "metric_gaps": metric_gaps,
                        "downgraded_to": "directional",
                    }
                    issues.append(issue)
                    unit["claim_status"] = "directional"
                    unit["claim_strength"] = "directional"
                    unit["claim_strength_ceiling"] = "directional"
                    unit["analysis_role"] = "directional"
                    unit["evidence_use_level"] = "directional_signal"
                    unit["writing_permission"] = "cautious_with_boundary"
                    unit["metric_completeness_status"] = "incomplete"
                    unit["metric_missing_fields"] = missing_fields
                    boundary_note = (
                        "metric fields incomplete: "
                        + (", ".join(missing_fields) if missing_fields else "unknown")
                        + "; use only as a directional signal until repaired"
                    )
                    limitation_boundary = _dedupe([*limitation_boundary, boundary_note])
                    unit["limitation_boundary"] = limitation_boundary
                    unit["counter_boundary"] = "\n".join(limitation_boundary)
            inferred_requirement_ids = _dedupe(
                [
                    *[
                        str(req or "").strip()
                        for req in _as_list(unit.get("requirement_ids"))
                        if str(req or "").strip()
                    ],
                    *[
                        str(card.get("requirement_id") or "").strip()
                        for card in cited_cards
                        if str(card.get("requirement_id") or "").strip()
                    ],
                ]
            )
            unit["requirement_ids"] = inferred_requirement_ids
            if not inferred_requirement_ids and requirement_contract_required:
                issue = {"type": "llm_claim_unit_missing_requirement_ids", "chapter_id": chapter.get("chapter_id")}
                issues.append(issue)
                if len(rejected_examples) < 5:
                    rejected_examples.append({**issue, "claim": claim_text, "evidence_refs": refs})
                continue
            hypothesis_id = str(unit.get("hypothesis_id") or chapter.get("hypothesis_id") or "").strip()
            if not hypothesis_id:
                hypothesis_id = next(
                    (
                        str(card.get("hypothesis_id") or "").strip()
                        for card in cited_cards
                        if str(card.get("hypothesis_id") or "").strip()
                    ),
                    "",
                )
            if hypothesis_id:
                unit["hypothesis_id"] = hypothesis_id
            if not str(unit.get("claim_strength_ceiling") or "").strip():
                source_levels = {str(card.get("source_level") or "").strip().upper() for card in cited_cards}
                allowed_uses = {str(card.get("allowed_use") or "").strip().lower() for card in cited_cards}
                if source_levels & {"A", "B"} and not {"directional_signal", "clue", "appendix_only"} & allowed_uses:
                    unit["claim_strength_ceiling"] = "moderate"
                else:
                    unit["claim_strength_ceiling"] = "directional"
            claim_strength = str(unit.get("claim_strength") or unit.get("claim_status") or "directional").strip().lower()
            if not claim_strength:
                claim_strength = "directional"
            unit["claim_strength"] = claim_strength
            ceiling = str(unit.get("claim_strength_ceiling") or "").strip().lower()
            if ceiling and _claim_strength_score(claim_strength) > _claim_strength_score(ceiling):
                issue = {
                    "type": "llm_claim_strength_exceeds_ceiling",
                    "chapter_id": chapter.get("chapter_id"),
                    "claim_strength": claim_strength,
                    "claim_strength_ceiling": ceiling,
                }
                issues.append(issue)
                if len(rejected_examples) < 5:
                    rejected_examples.append({**issue, "claim": claim_text, "evidence_refs": refs})
                continue
            source_ids = _dedupe(
                [
                    str(card.get("source_id") or _as_dict(card.get("lineage")).get("source_id") or "").strip()
                    for card in cited_cards
                    if str(card.get("source_id") or _as_dict(card.get("lineage")).get("source_id") or "").strip()
                ]
            )
            search_task_ids = _dedupe(
                [
                    str(card.get("search_task_id") or _as_dict(card.get("lineage")).get("search_task_id") or "").strip()
                    for card in cited_cards
                    if str(card.get("search_task_id") or _as_dict(card.get("lineage")).get("search_task_id") or "").strip()
                ]
            )
            unit["fact_ids"] = refs
            unit["source_ids"] = source_ids
            unit["lineage"] = {
                key: value
                for key, value in {
                    "chapter_id": chapter.get("chapter_id"),
                    "hypothesis_id": unit.get("hypothesis_id"),
                    "requirement_ids": inferred_requirement_ids,
                    "fact_ids": refs,
                    "source_ids": source_ids,
                    "search_task_ids": search_task_ids,
                }.items()
                if value not in (None, "", [])
            }
            if classify_claim_unit_roles is not None:
                role_result = classify_claim_unit_roles(
                    unit,
                    {ref: card_by_ref[ref] for ref in refs if ref in card_by_ref},
                )
                unit.update(role_result)
            cleaned_units.append(unit)
            if len(valid_examples) < 5:
                valid_examples.append(
                    {
                        "chapter_id": chapter.get("chapter_id"),
                        "claim": claim_text,
                        "evidence_refs": refs,
                        "claim_strength": unit.get("claim_strength") or unit.get("claim_status"),
                    }
                )
        chapter["claim_units"] = cleaned_units
        if cleaned_units:
            chapters.append(chapter)
    status = "valid" if any(_as_list(chapter.get("claim_units")) for chapter in chapters) else "invalid_output_no_usable_claims"
    issue_counts: Dict[str, int] = {}
    for issue in issues:
        issue_type = str(issue.get("type") or "unknown")
        issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
    usable_claim_count = sum(len(_as_list(chapter.get("claim_units"))) for chapter in chapters)
    return {
        "status": status,
        "issues": issues,
        "chapter_synthesis": chapters,
        "valid_ref_count": len(valid_refs),
        "usable_claim_count": usable_claim_count,
        "dropped_claim_count": len([item for item in issues if str(item.get("type") or "").startswith("llm_claim")]),
        "usable_chapter_count": len(chapters),
        "llm_raw_chapter_count": raw_chapter_count,
        "llm_raw_claim_count": raw_claim_count,
        "llm_validation_issue_counts": issue_counts,
        "llm_validation_issue_examples": issues[:8],
        "llm_valid_claim_examples": valid_examples,
        "llm_rejected_claim_examples": rejected_examples,
        "llm_deferred_claim_examples": deferred_examples,
        "claim_repair_priorities": claim_repair_priorities,
        "deferred_claim_count": len(claim_repair_priorities),
        "llm_semantic_judge_counts": semantic_judge_counts,
        "llm_semantic_judge_usage": semantic_judge_usage,
        "correctness_filter_summary": _correctness_filter_summary(
            raw_claim_count=raw_claim_count,
            usable_claim_count=usable_claim_count,
            issue_counts=issue_counts,
        ),
    }


def _claim_strength_score(value: Any) -> int:
    strength = str(value or "").strip().lower()
    return {
        "strong": 4,
        "decision_ready": 4,
        "moderate": 3,
        "medium": 3,
        "limited_evidence": 2,
        "directional": 1,
        "weak": 0,
    }.get(strength, 0)


def _claim_confidence_score(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _claim_ref_count(value: Dict[str, Any]) -> int:
    return len(_as_list(value.get("evidence_ids") or value.get("evidence_refs") or value.get("supporting_evidence")))


def _rank_key_judgments(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    unique: List[Dict[str, Any]] = []
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        text = _normalize_key(item.get("judgment") or item.get("claim"))
        if not text or text in seen:
            continue
        seen.add(text)
        unique.append(item)
    return sorted(
        unique,
        key=lambda item: (
            _claim_strength_score(item.get("claim_strength") or item.get("claim_status")),
            _claim_ref_count(item),
            _claim_confidence_score(item.get("confidence")),
        ),
        reverse=True,
    )


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
    seen_chapter_ids: set = set()
    for index, chapter in enumerate(_as_list(validation.get("chapter_synthesis")), start=1):
        if not isinstance(chapter, dict):
            continue
        # Prefer the LLM-provided chapter_id; otherwise derive a stable id from
        # the chapter title (normalized). Using `f"chapter_{index}"` is a last
        # resort and only fires when neither id nor title is available — this
        # keeps chapter_ids aligned with `chapter_evidence_diagnostics` keys.
        raw_chapter_id = str(chapter.get("chapter_id") or "").strip()
        if not raw_chapter_id:
            raw_chapter_id = _normalize_key(chapter.get("chapter_title") or chapter.get("chapter_question"))
        if not raw_chapter_id:
            raw_chapter_id = f"chapter_{index}"
        chapter_id = raw_chapter_id
        suffix = 2
        while chapter_id in seen_chapter_ids:
            chapter_id = f"{raw_chapter_id}_{suffix}"
            suffix += 1
        seen_chapter_ids.add(chapter_id)
        core_answer = _compact(chapter.get("core_answer"), 360)
        if _has_internal_analysis_language(core_answer):
            core_answer = _safe_public_claim_from_chapter(chapter)
        key_claims: List[Dict[str, Any]] = []
        for unit_index, unit in enumerate(_as_list(chapter.get("claim_units")), start=1):
            if not isinstance(unit, dict):
                continue
            refs = _as_list(unit.get("supporting_evidence_refs"))
            claim = _compact(unit.get("claim"), 360)
            if _has_internal_analysis_language(claim) or _is_generic_llm_claim(claim):
                claim = _safe_public_claim_from_chapter(chapter)
            reasoning = unit.get("reasoning") or _safe_public_reasoning_from_chapter(chapter)
            if _has_internal_analysis_language(reasoning):
                reasoning = _safe_public_reasoning_from_chapter(chapter)
            if not claim or not refs or not str(reasoning or "").strip():
                continue
            evidence_basis = [
                _compact(item, 260)
                for item in _as_list(unit.get("evidence_basis"))
                if str(item or "").strip()
            ]
            block_affinity = str(unit.get("block_affinity") or "").strip()
            decision_implication = _compact(unit.get("decision_use") or chapter.get("decision_implication") or "", 360)
            if _has_internal_analysis_language(decision_implication):
                decision_implication = ""
            claim_payload = {
                "claim": claim,
                "claim_status": unit.get("claim_status") or ("decision_ready" if refs else "directional"),
                "claim_strength": unit.get("claim_strength") or unit.get("claim_status") or ("moderate" if refs else "directional"),
                "claim_strength_ceiling": unit.get("claim_strength_ceiling"),
                "evidence_use_level": unit.get("evidence_use_level"),
                "writing_permission": unit.get("writing_permission"),
                "metric_completeness_status": unit.get("metric_completeness_status"),
                "metric_missing_fields": _as_list(unit.get("metric_missing_fields")),
                "requirement_ids": _as_list(unit.get("requirement_ids")),
                "fact_ids": _as_list(unit.get("fact_ids")) or refs,
                "source_ids": _as_list(unit.get("source_ids")),
                "supporting_evidence": refs,
                "evidence_refs": refs,
                "evidence_basis": evidence_basis,
                "supporting_facts": evidence_basis,
                "block_type": block_affinity,
                "claim_roles": _as_list(unit.get("claim_roles")),
                "primary_claim_role": unit.get("primary_claim_role"),
                "claim_role_contract_version": unit.get("claim_role_contract_version"),
                "role_reasons": _as_list(unit.get("role_reasons")),
                "mechanism": reasoning,
                "reasoning": reasoning,
                "counter_evidence": unit.get("counter_boundary") or "；".join(str(item) for item in _as_list(chapter.get("counter_evidence_boundary"))[:3]),
                "decision_implication": decision_implication,
                "confidence": chapter.get("confidence") or unit.get("confidence"),
                "what_to_verify_next": _as_list(chapter.get("remaining_gaps"))[:6],
            }
            if _has_internal_analysis_language(claim_payload["mechanism"]):
                claim_payload["mechanism"] = _safe_public_reasoning_from_chapter(chapter)
                claim_payload["reasoning"] = claim_payload["mechanism"]
            if _has_internal_analysis_language(claim_payload["counter_evidence"]):
                claim_payload["counter_evidence"] = _safe_chapter_counter_text(chapter)
            key_claims.append(claim_payload)
            claim_units.append(
                {
                    "id": f"{chapter_id}_llm_{unit_index}",
                    "chapter_id": chapter_id,
                    "dimension": chapter.get("chapter_title") or chapter_id,
                    "question": chapter.get("chapter_title") or chapter_id,
                    "claim": claim,
                    "claim_status": claim_payload["claim_status"],
                    "claim_strength": claim_payload["claim_strength"],
                    "claim_strength_ceiling": claim_payload["claim_strength_ceiling"],
                    "evidence_use_level": claim_payload["evidence_use_level"],
                    "writing_permission": claim_payload["writing_permission"],
                    "metric_completeness_status": claim_payload["metric_completeness_status"],
                    "metric_missing_fields": claim_payload["metric_missing_fields"],
                    "requirement_ids": claim_payload["requirement_ids"],
                    "fact_ids": claim_payload["fact_ids"],
                    "source_ids": claim_payload["source_ids"],
                    "reasoning": claim_payload["reasoning"] or claim_payload["mechanism"],
                    "mechanism": claim_payload["mechanism"],
                    "counter_evidence": claim_payload["counter_evidence"],
                    "decision_implication": claim_payload["decision_implication"],
                    "evidence_basis": evidence_basis,
                    "supporting_facts": evidence_basis,
                    "block_type": block_affinity,
                    "output_type": block_affinity,
                    "layout_section_role": block_affinity,
                    "claim_roles": claim_payload["claim_roles"],
                    "primary_claim_role": claim_payload["primary_claim_role"],
                    "claim_role_contract_version": claim_payload["claim_role_contract_version"],
                    "role_reasons": claim_payload["role_reasons"],
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
                        "claim_strength": claim_payload["claim_strength"],
                        "evidence_use_level": claim_payload["evidence_use_level"],
                        "writing_permission": claim_payload["writing_permission"],
                        "metric_completeness_status": claim_payload["metric_completeness_status"],
                        "metric_missing_fields": claim_payload["metric_missing_fields"],
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
        existing_chapters = _as_list(insight.get("chapters")) or _as_list(merged.get("chapter_insights"))
        chapter_map: Dict[str, Dict[str, Any]] = {}
        for existing in existing_chapters:
            if not isinstance(existing, dict):
                continue
            key = str(existing.get("chapter_id") or existing.get("chapter_question") or len(chapter_map) + 1)
            chapter_map[key] = existing
        for chapter in chapters:
            key = str(chapter.get("chapter_id") or chapter.get("chapter_question") or len(chapter_map) + 1)
            chapter_map[key] = {**_as_dict(chapter_map.get(key)), **chapter}
        insight["chapters"] = list(chapter_map.values())
        ranked_judgments = _rank_key_judgments(key_judgments + _as_list(merged.get("key_judgments")))
        summary_judgments, summary_quality = sanitize_summary_judgments(ranked_judgments, max_items=3)
        insight["executive_summary_quality"] = {
            **_as_dict(insight.get("executive_summary_quality")),
            **summary_quality,
            "executive_summary_fallback_used": False,
        }
        insight.setdefault("executive_summary", {})
        if summary_judgments:
            insight["report_thesis"] = _compact(summary_judgments[0].get("judgment"), 260)
            insight["executive_summary"] = {
                **_as_dict(insight.get("executive_summary")),
                "one_sentence_answer": _compact(summary_judgments[0].get("judgment"), 220),
                "top_3_judgments": summary_judgments,
                "so_what": _dedupe([item.get("decision_implication") for item in summary_judgments])[:5],
            }
        else:
            insight["report_thesis"] = ""
            insight["executive_summary"] = {
                **_as_dict(insight.get("executive_summary")),
                "one_sentence_answer": "",
                "top_3_judgments": [],
            }
        merged["report_insight_package"] = insight
        merged["chapter_insights"] = list(chapter_map.values())
    if claim_units:
        existing_units = [item for item in _as_list(merged.get("claim_units")) if isinstance(item, dict)]
        merged_units: List[Dict[str, Any]] = []
        seen_units = set()
        for unit in claim_units + existing_units:
            key = (
                str(unit.get("chapter_id") or unit.get("dimension") or ""),
                _normalize_key(unit.get("claim")),
                tuple(_as_list(unit.get("evidence_refs") or unit.get("supporting_evidence"))),
            )
            if not key[1] or key in seen_units:
                continue
            seen_units.add(key)
            merged_units.append(unit)
        merged["claim_units"] = merged_units
        merged["key_judgments"] = _rank_key_judgments(key_judgments + _as_list(merged.get("key_judgments")))
    evidence_repair_priorities: List[Dict[str, Any]] = []
    seen_repair_ids: set[str] = set()
    for item in [
        *_as_list(llm_payload.get("evidence_repair_priorities")),
        *_as_list(validation.get("claim_repair_priorities")),
    ]:
        if not isinstance(item, dict):
            continue
        key = str(item.get("gap_id") or item.get("claim_id") or item.get("claim") or len(evidence_repair_priorities)).strip()
        if key in seen_repair_ids:
            continue
        seen_repair_ids.add(key)
        evidence_repair_priorities.append(item)
    merged["llm_analysis_synthesis"] = {
        "chapter_synthesis": chapters,
        "cross_chapter_conflicts": _as_list(llm_payload.get("cross_chapter_conflicts")),
        "evidence_repair_priorities": evidence_repair_priorities,
        "rewrite_priorities": _as_list(llm_payload.get("rewrite_priorities")),
        "usage": llm_payload.get("_llm_usage", {}),
        "model": llm_payload.get("_llm_model", ""),
        "validation": validation,
    }
    if evidence_repair_priorities:
        merged["evidence_repair_priorities"] = evidence_repair_priorities
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
    ranked_key_judgments = _rank_key_judgments(key_judgments)
    summary_judgments, summary_quality = sanitize_summary_judgments(ranked_key_judgments, max_items=3)
    chapter_id_lookup = {
        dimension: next(
            (
                str(item.get("chapter_id") or "").strip()
                for item in evidence_analyses
                if str(item.get("dimension") or "").strip() == dimension
                and str(item.get("chapter_id") or "").strip()
            ),
            "",
        )
        for dimension in dimensions
    }
    chapter_insights = hypothesis_insights or _chapter_insights_from_synthesis(
        dimension_synthesis, chapter_id_lookup
    )
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
    research_reflection_memo = (
        build_research_reflection_memo(evidence_package)
        if build_research_reflection_memo is not None
        else {}
    )
    report_insight_package = {
        "report_thesis": _compact(summary_judgments[0].get("judgment") if summary_judgments else "", 260),
        "executive_summary": {
            "one_sentence_answer": _compact(summary_judgments[0].get("judgment") if summary_judgments else "", 220),
            "top_3_judgments": summary_judgments,
            "what_changed": _dedupe([item.get("fact") for item in core_facts])[:5],
            "so_what": _dedupe([item.get("decision_implication") for item in summary_judgments])[:5],
        },
        "executive_summary_quality": {
            **summary_quality,
            "executive_summary_fallback_used": False,
        },
        "chapters": chapter_insights,
        "decision_matrix": _as_list(_as_dict(evidence_package.get("decision_layer")).get("decision_matrix")),
        "risk_register": _as_list(_as_dict(evidence_package.get("risk_layer")).get("risk_items")),
        "evidence_refinement_plan": evidence_refinement_plan,
        "research_reflection_memo": research_reflection_memo,
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
        "research_reflection_memo": research_reflection_memo,
        "report_insight_package": report_insight_package,
        "claim_units": claim_units,
        "core_facts": core_facts,
        "key_judgments": ranked_key_judgments,
        "executive_summary_quality": report_insight_package["executive_summary_quality"],
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


def _parse_structured_tree(value: Any) -> Any:
    parsed = _parse_structured_string(value)
    if isinstance(parsed, dict):
        return {str(key): _parse_structured_tree(item) for key, item in parsed.items()}
    if isinstance(parsed, list):
        return [_parse_structured_tree(item) for item in parsed]
    return parsed


def _dict_list(value: Any) -> List[Dict[str, Any]]:
    return [item for item in _as_list(value) if isinstance(item, dict)]


def _structured_analysis_contract(structured: Dict[str, Any]) -> Dict[str, Any]:
    insight = _as_dict(structured.get("report_insight_package"))
    chapter_count = len(_dict_list(structured.get("chapter_insights"))) or len(_dict_list(insight.get("chapters")))
    claim_count = len(_dict_list(structured.get("claim_units")))
    evidence_count = len(_dict_list(structured.get("evidence_analyses")))
    llm = _as_dict(structured.get("llm_analysis_synthesis"))
    llm_chapters = _as_list(llm.get("chapter_synthesis"))
    unparsed_llm_chapter_count = len([item for item in llm_chapters if isinstance(item, str) and item.strip()])
    valid = bool((claim_count or chapter_count) and evidence_count)
    issues: List[str] = []
    if not claim_count:
        issues.append("claim_units_missing")
    if not chapter_count:
        issues.append("chapter_insights_missing")
    if not evidence_count:
        issues.append("evidence_analyses_missing")
    if unparsed_llm_chapter_count:
        issues.append("unparsed_or_truncated_llm_chapter_synthesis")
    return {
        "structured_analysis_valid": valid,
        "claim_unit_count": claim_count,
        "chapter_insight_count": chapter_count,
        "evidence_analysis_count": evidence_count,
        "unparsed_llm_chapter_synthesis_count": unparsed_llm_chapter_count,
        "issues": issues,
    }


STRUCTURAL_REBUILD_REASONS = frozenset(
    {
        "invalid_structured_analysis_contract",
        "unparsed_or_truncated_llm_chapter_synthesis",
        "chapter_binding_failed",
        "unbound_ab_evidence",
    }
)


def _structured_analysis_rebuild_reasons(
    structured: Dict[str, Any], contract: Dict[str, Any]
) -> List[str]:
    """Return all rebuild signals (structural + quality), in stable order.

    `ensure_valid_structured_analysis` uses
    :data:`STRUCTURAL_REBUILD_REASONS` to decide whether to actually rebuild.
    Quality-only signals (e.g. high repeated_claim_ratio, title_as_claim)
    are still surfaced in diagnostics but no longer throw away a valid
    LLM analysis. This avoids the previous failure mode where a good LLM
    output was rebuilt from a deterministic fallback solely because the
    quality heuristics misfired on triple-counted claims.
    """

    reasons: List[str] = []
    if not contract.get("structured_analysis_valid"):
        reasons.append("invalid_structured_analysis_contract")
    if int(contract.get("unparsed_llm_chapter_synthesis_count") or 0) > 0:
        reasons.append("unparsed_or_truncated_llm_chapter_synthesis")
    quality = _as_dict(structured.get("analysis_depth_quality")) or analysis_depth_quality(structured)
    repeated_ratio = 0.0
    try:
        repeated_ratio = float(quality.get("repeated_claim_ratio") or 0.0)
    except (TypeError, ValueError):
        repeated_ratio = 0.0
    if str(quality.get("status") or "").strip().lower() == "needs_rewrite":
        reasons.append("needs_rewrite_quality")
    if repeated_ratio > 0.30:
        reasons.append("repeated_claim_ratio_high")
    try:
        title_as_claim_count = int(float(quality.get("title_as_claim_count") or 0))
    except (TypeError, ValueError):
        title_as_claim_count = 0
    if title_as_claim_count > 0:
        reasons.append("title_as_claim")
    feedback = _as_dict(structured.get("claim_binding_feedback_summary")) or claim_binding_feedback_summary(structured)
    try:
        unbound_count = int(float(feedback.get("available_ab_not_bound_count") or 0))
    except (TypeError, ValueError):
        unbound_count = 0
    if unbound_count > 0:
        reasons.append("unbound_ab_evidence")
    evidence_items = _dict_list(structured.get("evidence_analyses"))
    if evidence_items:
        missing_chapter = len([item for item in evidence_items if not str(item.get("chapter_id") or "").strip()])
        if missing_chapter / max(len(evidence_items), 1) > 0.30:
            reasons.append("chapter_binding_failed")
    return _dedupe(reasons)


def _structural_reasons(reasons: Sequence[str]) -> List[str]:
    return [reason for reason in reasons if reason in STRUCTURAL_REBUILD_REASONS]


def _quality_only_reasons(reasons: Sequence[str]) -> List[str]:
    return [reason for reason in reasons if reason not in STRUCTURAL_REBUILD_REASONS]


# Quality-only signals that downstream agents (e.g. claim_builder) should still
# treat as a "strict mode" trigger even though they don't justify rebuilding
# the entire structured_analysis. Listed here so the policy lives in one place
# and is published in analysis_contract_status for downstream consumers.
STRICT_CLAIM_BUILD_REASONS = frozenset(
    {
        "invalid_structured_analysis_contract",
        "unparsed_or_truncated_llm_chapter_synthesis",
        "chapter_binding_failed",
        "unbound_ab_evidence",
        "needs_rewrite_quality",
        "repeated_claim_ratio_high",
        "title_as_claim",
    }
)


def _should_force_strict_claim_building(reasons: Sequence[str]) -> bool:
    return any(reason in STRICT_CLAIM_BUILD_REASONS for reason in reasons)


def ensure_valid_structured_analysis(
    structured_analysis: Dict[str, Any],
    evidence_package: Dict[str, Any],
    *,
    rebuild_reason: str = "invalid_structured_analysis_contract",
) -> Dict[str, Any]:
    """Return a writer-consumable analysis payload.

    The report pipeline cannot consume stringified/truncated dicts. When the
    LLM result or a compacted payload loses the real contract, rebuild from the
    evidence package instead of letting Claim/Chapter fall back to templates.
    """

    parsed = _as_dict(_parse_structured_tree(structured_analysis))
    nested = _as_dict(parsed.get("structured_analysis"))
    if nested:
        merged = dict(nested)
        for key, value in parsed.items():
            if key not in {"structured_analysis"} and key not in merged:
                merged[key] = value
        parsed = merged
    contract = _structured_analysis_contract(parsed)
    rebuild_reasons = _structured_analysis_rebuild_reasons(parsed, contract)
    structural_reasons = _structural_reasons(rebuild_reasons)
    quality_only_reasons = _quality_only_reasons(rebuild_reasons)
    llm_validation = _as_dict(_as_dict(parsed.get("llm_analysis_synthesis")).get("validation"))
    valid_llm_claims_present = (
        str(llm_validation.get("status") or "") == "valid"
        and _int_or_zero(llm_validation.get("usable_claim_count")) > 0
    )
    if valid_llm_claims_present:
        suppressed_structural = [
            reason
            for reason in structural_reasons
            if reason in {"chapter_binding_failed", "unbound_ab_evidence"}
        ]
        if suppressed_structural:
            structural_reasons = [
                reason
                for reason in structural_reasons
                if reason not in {"chapter_binding_failed", "unbound_ab_evidence"}
            ]
            quality_only_reasons = _dedupe([*quality_only_reasons, *suppressed_structural])
    already_rebuilt = bool(parsed.get("analysis_rebuilt_from_evidence"))
    # Only structural reasons (missing/garbled contract, unbound evidence,
    # unparsed LLM output, mass chapter binding failure) justify throwing away
    # the current analysis and rebuilding from scratch. Quality-only signals
    # (repeated ratio, generic mechanism, title-as-claim) are now annotated
    # in diagnostics so downstream consumers can react, but the LLM result
    # is preserved.
    should_force_strict = _should_force_strict_claim_building(rebuild_reasons)
    if not structural_reasons or already_rebuilt:
        parsed["analysis_contract_status"] = {
            **contract,
            "quality_rebuild_reasons": rebuild_reasons,
            "structural_rebuild_reasons": structural_reasons,
            "quality_only_warnings": quality_only_reasons,
            "should_force_strict_claim_building": should_force_strict,
            "analysis_rebuilt_from_evidence": bool(parsed.get("analysis_rebuilt_from_evidence")),
        }
        return parsed
    rebuilt = build_fallback_analysis(evidence_package)
    rebuilt = _public_normalize_analysis_payload(rebuilt)
    rebuilt_contract = _structured_analysis_contract(rebuilt)
    rebuilt["analysis_rebuilt_from_evidence"] = True
    rebuilt["analysis_contract_status"] = {
        **rebuilt_contract,
        "previous_contract": contract,
        "analysis_rebuilt_from_evidence": True,
        "rebuild_reason": rebuild_reason,
        "quality_rebuild_reasons": rebuild_reasons,
        "should_force_strict_claim_building": should_force_strict,
        "structural_rebuild_reasons": structural_reasons,
        "quality_only_warnings": quality_only_reasons,
    }
    diagnostics = _as_dict(rebuilt.get("analysis_stage_diagnostics"))
    diagnostics["analysis_rebuilt_from_evidence"] = True
    diagnostics["analysis_rebuild_reason"] = rebuild_reason
    diagnostics["analysis_rebuild_reasons"] = rebuild_reasons
    diagnostics["structural_rebuild_reasons"] = structural_reasons
    diagnostics["quality_only_warnings"] = quality_only_reasons
    diagnostics["previous_contract_issues"] = contract.get("issues")
    rebuilt["analysis_stage_diagnostics"] = diagnostics
    return rebuilt


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
        llm_validation: Dict[str, Any] = {}
        llm_enabled = _env_flag("BRAIN_ENABLE_LLM_EVIDENCE_ANALYSIS", True)
        llm_ready = bool(llm_config_is_ready is not None and llm_config_is_ready(llm_config or {}))
        llm_attempted = False
        quality_path_requested = (
            str(os.getenv("REPORT_QUALITY_MODE") or "").strip().lower() == "high"
            or str(os.getenv("REPORT_REPLAY_EXECUTION_MODE") or "").strip() == "quality_llm_replay"
        )
        if llm_enabled:
            if llm_ready:
                llm_attempted = True
                try:
                    use_v2_analysis = (
                        str(os.getenv("BRAIN_LLM_ANALYSIS_INPUT_VERSION") or "v2").strip().lower() == "v2"
                        or str(os.getenv("BRAIN_LLM_ANALYSIS_MODE") or "per_chapter").strip().lower() == "per_chapter"
                    )
                    if use_v2_analysis:
                        llm_payload = synthesize_with_llm_analysis_v2(
                            evidence_package=package,
                            fallback=structured,
                            llm_config=dict(llm_config or {}),
                        )
                    else:
                        llm_payload = synthesize_with_llm_analysis(
                            evidence_package=package,
                            fallback=structured,
                            llm_config=dict(llm_config or {}),
                        )
                    validation = validate_llm_analysis_output(llm_payload, package, llm_config=dict(llm_config or {}))
                    llm_validation = validation
                    if str(validation.get("status") or "") == "valid":
                        structured = merge_llm_analysis_with_fallback(structured, llm_payload, validation)
                        submitted_chapters = _env_int("BRAIN_LLM_ANALYSIS_MAX_CHAPTERS", 8, min_value=1, max_value=100)
                        if use_v2_analysis:
                            submitted_chapters = _int_or_zero(llm_payload.get("_llm_submitted_chapter_count")) or submitted_chapters
                        valid_chapters = _int_or_zero(validation.get("usable_chapter_count"))
                        failed_chapters = _int_or_zero(llm_payload.get("_llm_failed_chapter_count"))
                        llm_status = "partial_success" if use_v2_analysis and (failed_chapters or valid_chapters < submitted_chapters) else "success"
                    else:
                        llm_status = "invalid_output"
                        llm_error = "LLM evidence analysis returned no usable chapter_synthesis."
                except Exception as exc:
                    llm_status = "fallback"
                    llm_error = str(exc)
            else:
                llm_status = "fallback_config_missing"
        structured = _public_normalize_analysis_payload(structured)
        structured = ensure_valid_structured_analysis(
            structured,
            package,
            rebuild_reason="llm_or_compacted_analysis_invalid",
        )
        research_reflection_memo = (
            build_research_reflection_memo(package, structured_analysis=structured)
            if build_research_reflection_memo is not None
            else _as_dict(structured.get("research_reflection_memo"))
        )
        structured["research_reflection_memo"] = research_reflection_memo
        insight = _as_dict(structured.get("report_insight_package"))
        if insight:
            structured["report_insight_package"] = {
                **insight,
                "research_reflection_memo": research_reflection_memo,
            }
        structured["analysis_depth_quality"] = analysis_depth_quality(structured)
        structured["claim_binding_feedback_summary"] = claim_binding_feedback_summary(structured)
        rebuilt_after_llm = bool(
            structured.get("analysis_rebuilt_from_evidence")
            or _as_dict(structured.get("analysis_contract_status")).get("analysis_rebuilt_from_evidence")
            or _as_dict(structured.get("analysis_stage_diagnostics")).get("analysis_rebuilt_from_evidence")
        )
        final_llm_used = llm_status in {"success", "partial_success"} and not rebuilt_after_llm and _int_or_zero(llm_validation.get("usable_claim_count")) > 0
        final_llm_status = "success_then_rebuilt" if llm_status in {"success", "partial_success"} and rebuilt_after_llm else llm_status
        final_analysis_source = (
            "llm_evidence_analysis"
            if final_llm_used and llm_status == "success"
            else "llm_partial_merged"
            if final_llm_used and llm_status == "partial_success"
            else (
            "deterministic_rebuild" if rebuilt_after_llm else "dynamic_claim_builder"
            )
        )
        quality_path_degraded = bool(quality_path_requested and not final_llm_used)
        if not quality_path_degraded:
            quality_path_degradation_reason = ""
        elif not llm_enabled:
            quality_path_degradation_reason = "llm_analysis_disabled"
        elif not llm_ready:
            quality_path_degradation_reason = "fallback_config_missing"
        elif final_llm_status == "success_then_rebuilt":
            quality_path_degradation_reason = "success_then_rebuilt"
        else:
            quality_path_degradation_reason = final_llm_status or llm_error or "llm_not_used"
        diagnostics = {
            **_as_dict(structured.get("analysis_stage_diagnostics")),
            "uses_llm_analysis": final_llm_used,
            "llm_analysis_attempted": llm_attempted,
            "llm_analysis_status": final_llm_status,
            "final_analysis_source": final_analysis_source,
            "deterministic_synthesis_used": not final_llm_used,
            "quality_path_requested": quality_path_requested,
            "quality_path_degraded": quality_path_degraded,
            "quality_path_degradation_reason": quality_path_degradation_reason,
            "llm_input_valid_ref_count": llm_validation.get("valid_ref_count"),
            "llm_usable_claim_count": llm_validation.get("usable_claim_count", 0),
            "llm_dropped_claim_count": llm_validation.get("dropped_claim_count", 0),
            "llm_deferred_claim_count": llm_validation.get("deferred_claim_count", 0),
            "llm_usable_chapter_count": llm_validation.get("usable_chapter_count", 0),
            "llm_valid_chapter_count": llm_validation.get("usable_chapter_count", 0),
            "llm_failed_chapter_count": (
                max(
                    _int_or_zero(llm_payload.get("_llm_failed_chapter_count")),
                    _int_or_zero(llm_payload.get("_llm_submitted_chapter_count")) - _int_or_zero(llm_validation.get("usable_chapter_count")),
                )
                if "llm_payload" in locals()
                else 0
            ),
            "llm_analysis_cache_hit_count": llm_payload.get("_llm_cache_hit_count", 0) if "llm_payload" in locals() else 0,
            "llm_raw_chapter_count": llm_validation.get("llm_raw_chapter_count", 0),
            "llm_raw_claim_count": llm_validation.get("llm_raw_claim_count", 0),
            "llm_validation_issue_counts": llm_validation.get("llm_validation_issue_counts", {}),
            "llm_validation_issue_examples": llm_validation.get("llm_validation_issue_examples", []),
            "llm_valid_claim_examples": llm_validation.get("llm_valid_claim_examples", []),
            "llm_rejected_claim_examples": llm_validation.get("llm_rejected_claim_examples", []),
            "llm_deferred_claim_examples": llm_validation.get("llm_deferred_claim_examples", []),
            "claim_repair_priorities": llm_validation.get("claim_repair_priorities", []),
            "llm_semantic_judge_counts": llm_validation.get("llm_semantic_judge_counts", {}),
            "llm_semantic_judge_usage": llm_validation.get("llm_semantic_judge_usage", {}),
            "llm_validation_status": llm_validation.get("status") or ("not_run" if llm_status in {"disabled", "fallback_config_missing"} else llm_status),
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
            "research_reflection_status": research_reflection_memo.get("status"),
            "research_reflection_write_mode": research_reflection_memo.get("write_mode"),
            "research_reflection_seed_count": len(_as_list(research_reflection_memo.get("next_search_task_seeds"))),
        }
        structured["analysis_stage_diagnostics"] = diagnostics
        source = final_analysis_source
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
                "llm_analysis_status": final_llm_status,
                "llm_analysis_error": llm_error,
                "final_analysis_source": final_analysis_source,
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
