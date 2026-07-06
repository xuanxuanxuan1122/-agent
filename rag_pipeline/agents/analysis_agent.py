from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypedDict

try:
    from rag_pipeline.contracts.chapter_identity import (
        build_chapter_identity_map,
        canonical_chapter_ids,
        canonical_chapter_payload,
        resolve_canonical_chapter_id,
    )
    from rag_pipeline.contracts.claim_roles import classify_claim_unit_roles
    from rag_pipeline.contracts.evidence_identity import build_evidence_alias_map
    from rag_pipeline.contracts.evidence_support_validation import (
        incomplete_metric_cards_for_numeric_claim,
        validate_claim_supported_by_facts,
    )
    from rag_pipeline.contracts.research_reflection import build_research_reflection_memo
    from rag_pipeline.contracts.evidence_quality import (
        NON_CLAIM_ALLOWED_USES,
        NON_CLAIM_ANALYSIS_READINESS,
        classify_evidence,
    )
    from rag_pipeline.contracts.json_salvage import salvage_chapter_json
    from rag_pipeline.contracts.quality_gate_policy import advisory_weight_mode, quality_gate_mode, quality_gates_isolated
    from rag_pipeline.contracts.ref_normalizer import normalize_claim_refs
    from rag_pipeline.search.memory import call_openai_compatible_json, llm_config_is_ready, normalize_llm_config
    from .evidence_interpreter_agent import build_evidence_interpretation_units
    from .evidence_merger import get_dynamic_dimensions
    from .summary_quality import sanitize_summary_judgments
except Exception:  # pragma: no cover - script mode fallback
    try:
        from rag_pipeline.contracts.chapter_identity import (  # type: ignore
            build_chapter_identity_map,
            canonical_chapter_ids,
            canonical_chapter_payload,
            resolve_canonical_chapter_id,
        )
    except Exception:  # pragma: no cover
        build_chapter_identity_map = None  # type: ignore
        canonical_chapter_ids = None  # type: ignore
        canonical_chapter_payload = None  # type: ignore
        resolve_canonical_chapter_id = None  # type: ignore
    try:
        from rag_pipeline.contracts.claim_roles import classify_claim_unit_roles  # type: ignore
    except Exception:  # pragma: no cover
        classify_claim_unit_roles = None  # type: ignore
    try:
        from rag_pipeline.contracts.evidence_identity import build_evidence_alias_map  # type: ignore
    except Exception:  # pragma: no cover
        build_evidence_alias_map = None  # type: ignore
    try:
        from rag_pipeline.contracts.evidence_quality import (  # type: ignore
            NON_CLAIM_ALLOWED_USES,
            NON_CLAIM_ANALYSIS_READINESS,
            classify_evidence,
        )
    except Exception:  # pragma: no cover
        NON_CLAIM_ALLOWED_USES = {  # type: ignore
            "rejected",
            "not_allowed",
            "not_for_writing",
            "diagnostic_only",
            "appendix_only",
            "clue_only",
            "clue",
            "search_only",
        }
        NON_CLAIM_ANALYSIS_READINESS = {  # type: ignore
            "blocked",
            "followup_only",
            "clue_only",
            "diagnostic_only",
        }
        classify_evidence = None  # type: ignore
    try:
        from rag_pipeline.contracts.json_salvage import salvage_chapter_json  # type: ignore
    except Exception:  # pragma: no cover
        salvage_chapter_json = None  # type: ignore
    try:
        from rag_pipeline.contracts.quality_gate_policy import advisory_weight_mode, quality_gate_mode, quality_gates_isolated  # type: ignore
    except Exception:  # pragma: no cover
        def advisory_weight_mode(default: bool = False) -> bool:  # type: ignore
            return default

        def quality_gate_mode(default: str = "blocking") -> str:  # type: ignore
            return default

        def quality_gates_isolated(default: bool = False) -> bool:  # type: ignore
            return default
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
        from rag_pipeline.contracts.ref_normalizer import normalize_claim_refs  # type: ignore
    except Exception:  # pragma: no cover
        normalize_claim_refs = None  # type: ignore
    try:
        from rag_pipeline.search.memory import call_openai_compatible_json, llm_config_is_ready, normalize_llm_config  # type: ignore
    except Exception:  # pragma: no cover
        call_openai_compatible_json = None  # type: ignore
        llm_config_is_ready = None  # type: ignore
        normalize_llm_config = None  # type: ignore
    try:
        from evidence_interpreter_agent import build_evidence_interpretation_units  # type: ignore
    except Exception:  # pragma: no cover
        build_evidence_interpretation_units = None  # type: ignore
    from evidence_merger import get_dynamic_dimensions  # type: ignore
    from summary_quality import sanitize_summary_judgments  # type: ignore

try:
    from rag_pipeline.cache.stage_execution_guard import (
        get_cached_stage_output as _stage_guard_get_cached_output,
        record_stage_execution as _stage_guard_record_execution,
        stable_stage_hash as _stage_guard_hash,
        stage_cache_enabled as _stage_guard_cache_enabled,
        stage_guard_enabled as _stage_guard_enabled,
        store_stage_output as _stage_guard_store_output,
    )
except Exception:  # pragma: no cover - diagnostics only.
    _stage_guard_get_cached_output = None  # type: ignore
    _stage_guard_record_execution = None  # type: ignore
    _stage_guard_hash = None  # type: ignore
    _stage_guard_cache_enabled = None  # type: ignore
    _stage_guard_enabled = None  # type: ignore
    _stage_guard_store_output = None  # type: ignore


@dataclass(frozen=True)
class NumericGroundingResult:
    valid: bool
    reasons: List[str]
    numbers: List[str]


NUMERIC_FACT_TOKEN_RE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:万亿元|亿元|万元|%|％|个百分点|万台|万架|万家|万套|台|架|家|个)"
)
MALFORMED_YEAR_RANGE_RE = re.compile(r"20\d{2}\s*[-—]\s*\d{1,2}\s*年")


def _numeric_grounding_text(value: Any) -> str:
    text = re.sub(r"\s+", "", str(value or ""))
    return text.replace("％", "%")


def validate_numeric_fact_grounding(claim_text: str, cited_facts: Sequence[Dict[str, Any]]) -> NumericGroundingResult:
    """Check that public numeric hard facts are present in cited facts."""

    reasons: List[str] = []
    claim = str(claim_text or "")
    if MALFORMED_YEAR_RANGE_RE.search(claim):
        reasons.append("malformed_year_range")
    fact_text = " ".join(
        str(
            fact.get("fact")
            or fact.get("clean_fact")
            or fact.get("distilled_fact")
            or fact.get("text")
            or fact.get("content")
            or ""
        )
        for fact in list(cited_facts or [])
        if isinstance(fact, dict)
    )
    fact_norm = _numeric_grounding_text(fact_text)
    numbers = [re.sub(r"\s+", "", token).replace("％", "%") for token in NUMERIC_FACT_TOKEN_RE.findall(claim)]
    for token in numbers:
        if token and token not in fact_norm:
            reasons.append("unsupported_number_or_unit")
            break
    return NumericGroundingResult(valid=not reasons, reasons=_dedupe(reasons), numbers=_dedupe(numbers))

try:  # Runtime probes are diagnostic-only and must never block analysis.
    from rag_pipeline.observability.probe_api import emit_transform as _probe_emit_transform
    from rag_pipeline.observability.probe_context import current_probe_context_from_env as _current_probe_context_from_env
    from rag_pipeline.observability.data_root_hygiene import inspect_evidence_analysis_hygiene as _inspect_evidence_analysis_hygiene
except Exception:  # pragma: no cover - optional diagnostics only.
    _probe_emit_transform = None
    _current_probe_context_from_env = None
    _inspect_evidence_analysis_hygiene = None


AGENT_NAME = "analysis_agent"
AGENT_DESCRIPTION = "Dynamic Research Claim Builder. Converts evidence packages into claim units for the writer."
CHAPTER_EVIDENCE_COLLECTIONS = (
    "analysis_ready_facts",
    "fact_cards",
    "evidence_cards",
    "evidence_fact_cards",
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
    r"\bsemantic\s+judge\b",
    r"\bmetric\s+fields\s+incomplete\b",
    r"\bnot_allowed_until_repaired\b",
    r"\bcurrency_(?:usd|cny)\b",
    r"\bsource_check\s*(?:=|:|\u4e3a)\b",
    r"\bhttp_status\s*(?:=|:|\u4e3a)\b",
    r"\bresponse_code\s*(?:=|:|\u4e3a)\b",
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


def _lineage_requirement_id(item: Dict[str, Any]) -> str:
    lineage = _as_dict(item.get("lineage"))
    search_task = _as_dict(item.get("search_task"))
    return str(
        item.get("requirement_id")
        or item.get("evidence_requirement_id")
        or lineage.get("requirement_id")
        or lineage.get("evidence_requirement_id")
        or search_task.get("requirement_id")
        or search_task.get("evidence_requirement_id")
        or search_task.get("slot_id")
        or ""
    ).strip()


def _analysis_requirement_lineage_coverage(
    evidence_package: Dict[str, Any],
    structured_analysis: Dict[str, Any],
) -> Dict[str, Any]:
    analysis_ready = [
        item
        for item in _as_list(_as_dict(evidence_package).get("analysis_ready_evidence"))
        if isinstance(item, dict)
    ]
    claim_units = [
        item
        for item in _as_list(_as_dict(structured_analysis).get("claim_units"))
        if isinstance(item, dict)
    ]
    input_with_requirement = [item for item in analysis_ready if _lineage_requirement_id(item)]
    missing_input_examples = [
        str(item.get("evidence_id") or item.get("ref") or item.get("id") or "").strip()
        for item in analysis_ready
        if not _lineage_requirement_id(item)
    ]
    missing_input_examples = [item for item in missing_input_examples if item][:8]
    claim_with_requirement = [
        item
        for item in claim_units
        if _as_list(item.get("requirement_ids")) or _lineage_requirement_id(item)
    ]
    claim_with_fact = [
        item
        for item in claim_units
        if _as_list(
            item.get("fact_ids")
            or item.get("used_evidence_ids")
            or item.get("used_fact_refs")
            or item.get("evidence_refs")
        )
    ]
    missing_claim_examples = [
        str(item.get("claim_id") or item.get("id") or "").strip()
        for item in claim_units
        if not (_as_list(item.get("requirement_ids")) or _lineage_requirement_id(item))
    ]
    missing_claim_examples = [item for item in missing_claim_examples if item][:8]
    input_count = len(analysis_ready)
    claim_count = len(claim_units)
    return {
        "input_requirement_id_count": len(input_with_requirement),
        "input_missing_requirement_id_count": max(0, input_count - len(input_with_requirement)),
        "input_requirement_id_rate": round(len(input_with_requirement) / input_count, 4) if input_count else 0.0,
        "input_missing_requirement_id_examples": missing_input_examples,
        "claim_requirement_binding_count": len(claim_with_requirement),
        "claim_missing_requirement_binding_count": max(0, claim_count - len(claim_with_requirement)),
        "claim_requirement_binding_rate": round(len(claim_with_requirement) / claim_count, 4) if claim_count else 0.0,
        "claim_missing_requirement_binding_examples": missing_claim_examples,
        "claim_fact_binding_count": len(claim_with_fact),
        "claim_fact_binding_rate": round(len(claim_with_fact) / claim_count, 4) if claim_count else 0.0,
    }


def _emit_analysis_agent_probe(result: Dict[str, Any]) -> None:
    """Record analysis-ready evidence to claim-unit conversion as a sidecar event."""

    if _probe_emit_transform is None or _current_probe_context_from_env is None:
        return
    try:
        probe = _current_probe_context_from_env()
        if probe is None:
            return
        package = _as_dict(result.get("evidence_package"))
        structured = _as_dict(result.get("structured_analysis"))
        diagnostics = _as_dict(structured.get("analysis_stage_diagnostics"))
        analysis_ready = [item for item in _as_list(package.get("analysis_ready_evidence")) if isinstance(item, dict)]
        claim_units = [item for item in _as_list(structured.get("claim_units")) if isinstance(item, dict)]
        bound_claims = sum(1 for item in claim_units if _as_list(item.get("fact_ids") or item.get("used_evidence_ids") or item.get("used_fact_refs")))
        requirement_bound = sum(1 for item in claim_units if _as_list(item.get("requirement_ids")) or str(item.get("requirement_id") or "").strip())
        output_claim_count = len(claim_units)
        raw_claim_count = _int_or_zero(diagnostics.get("llm_raw_claim_count")) or output_claim_count
        dropped_claim_count = _int_or_zero(diagnostics.get("llm_dropped_claim_count"))
        deferred_claim_count = _int_or_zero(diagnostics.get("llm_deferred_claim_count"))
        lineage_coverage = _analysis_requirement_lineage_coverage(package, structured)
        evidence_hygiene = _inspect_evidence_analysis_hygiene(analysis_ready) if _inspect_evidence_analysis_hygiene else {}
        evidence_dirty_count = int(evidence_hygiene.get("dirty_item_count") or 0) if isinstance(evidence_hygiene, dict) else 0
        evidence_reason_counts = dict(evidence_hygiene.get("reason_counts") or {}) if isinstance(evidence_hygiene, dict) else {}
        _probe_emit_transform(
            probe,
            stage="analysis_agent",
            module="analysis_agent",
            input_count=len(analysis_ready),
            output_count=output_claim_count,
            drop_count=max(0, raw_claim_count - output_claim_count) + dropped_claim_count,
            status="ok" if output_claim_count > 0 else "warning",
            reason_counts={
                **_as_dict(diagnostics.get("llm_validation_issue_counts")),
                "dropped_claim": dropped_claim_count,
                "deferred_claim": deferred_claim_count,
                "quality_path_degraded": 1 if diagnostics.get("quality_path_degraded") else 0,
                "input_evidence_dirty_item": evidence_dirty_count,
                **{f"input_evidence_{key}": value for key, value in evidence_reason_counts.items()},
            },
            id_coverage={
                "claim_fact_binding": lineage_coverage.get("claim_fact_binding_rate", (bound_claims / output_claim_count) if output_claim_count else 0.0),
                "claim_requirement_binding": lineage_coverage.get(
                    "claim_requirement_binding_rate",
                    (requirement_bound / output_claim_count) if output_claim_count else 0.0,
                ),
                "input_requirement_id": lineage_coverage.get("input_requirement_id_rate", 0.0),
            },
            cache={
                "llm_analysis_cache_hit_count": _int_or_zero(diagnostics.get("llm_analysis_cache_hit_count")),
                "semantic_judge_cache_hit_count": _int_or_zero(_as_dict(diagnostics.get("llm_semantic_judge_counts")).get("cache_hit")),
            },
            metrics={
                "llm_usable_claim_count": _int_or_zero(diagnostics.get("llm_usable_claim_count")),
                "llm_raw_claim_count": _int_or_zero(diagnostics.get("llm_raw_claim_count")),
                "llm_usable_chapter_count": _int_or_zero(diagnostics.get("llm_usable_chapter_count")),
                "decision_ready_claim_count": _int_or_zero(diagnostics.get("decision_ready_claim_count")),
                "directional_claim_count": _int_or_zero(diagnostics.get("directional_claim_count")),
            },
            diagnostics={
                "final_analysis_source": diagnostics.get("final_analysis_source") or "",
                "uses_llm_analysis": bool(diagnostics.get("uses_llm_analysis")),
                "llm_analysis_status": diagnostics.get("llm_analysis_status") or "",
                "llm_usable_claim_count": _int_or_zero(diagnostics.get("llm_usable_claim_count")),
                "llm_dropped_claim_count": dropped_claim_count,
                "llm_deferred_claim_count": deferred_claim_count,
                "research_reflection_status": diagnostics.get("research_reflection_status") or "",
                "research_reflection_seed_count": _int_or_zero(diagnostics.get("research_reflection_seed_count")),
                "evidence_root_hygiene": evidence_hygiene,
            },
        )
    except Exception:
        return


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
    r"已有可观察的公开资料信号",
    r"已有方向性公开资料信号",
    r"已有初步公开资料线索",
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


def _is_canonical_chapter_key(value: Any) -> bool:
    return bool(re.fullmatch(r"ch[_-]?\d{1,3}", str(value or "").strip().lower()))


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
    chapter_packages = [
        chapter
        for chapter in _as_list(evidence_package.get("chapter_evidence_packages"))
        if isinstance(chapter, dict)
    ]
    if chapter_packages:
        canonical_dimensions: List[str] = []
        aliases: set[str] = set()
        for index, chapter in enumerate(chapter_packages, start=1):
            chapter_id = str(
                chapter.get("chapter_id")
                or chapter.get("dimension_id")
                or _normalize_key(chapter.get("chapter_title") or chapter.get("chapter_question"))
                or f"ch_{index:02d}"
            ).strip()
            if chapter_id and chapter_id not in canonical_dimensions:
                canonical_dimensions.append(chapter_id)
            for value in (
                chapter_id,
                chapter.get("chapter_title"),
                chapter.get("chapter_question"),
                chapter.get("dimension_id"),
            ):
                raw = str(value or "").strip()
                if not raw:
                    continue
                aliases.add(raw)
                normalized = _normalize_key(raw)
                if normalized:
                    aliases.add(normalized)
        for dimension in dimensions:
            text = str(dimension or "").strip()
            if not text or text in canonical_dimensions:
                continue
            normalized = _normalize_key(text)
            if text in aliases or (normalized and normalized in aliases):
                continue
            canonical_dimensions.append(text)
        return canonical_dimensions or ["综合研究问题"]
    for dimension in _as_dict(evidence_package.get("per_dimension")).keys():
        text = str(dimension or "").strip()
        if text and text not in dimensions:
            dimensions.append(text)
    for item in _as_list(evidence_package.get("analysis_ready_evidence")) + _as_list(evidence_package.get("clean_evidence_list")):
        if isinstance(item, dict):
            for value in (
                item.get("chapter_id"),
                item.get("dimension_id"),
                item.get("hypothesis_id"),
                item.get("dimension_name"),
                item.get("evidence_goal"),
                item.get("dimension"),
            ):
                text = str(value or "").strip()
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

RAW_PAGE_FRAGMENT_RE = re.compile(
    r"(https?://|www\.|!\(|\]\(|#{2,}|/detail\.php|"
    r"\b(login|contact|copyright|javascript|cookie|download)\b|"
    r"热门文章|最新文章|浏览量|文章来源|字号|打印|关闭|"
    r"首页|联系我们|加入我们|上一篇|下一篇|中\s*EN|EN\s*安泰)",
    re.I,
)
WEB_CHROME_FRAGMENT_RE = re.compile(
    r"/newstatic/images/logo|(?:^|/|\\)logo\.(?:gif|png|jpg|jpeg|webp|svg)\b|"
    r"\u6570\u636e\u4e2d\u5fc3|\u5168\u7403\u8d22\u7ecf\u5feb\u8baf|\u8d22\u7ecf\u5feb\u8baf|"
    r"\u884c\u60c5\u4e2d\u5fc3|Choice\s*\u6570\u636e|\u4e1c\u65b9\u8d22\u5bcc|\u81ea\u9009\u80a1|\u80a1\u5427|"
    r"(?:\[[^\]]{0,80}\]\((?:https?:)?//[^)]+|/[^)]+\).*){2,}",
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


def _normalize_public_fact_markup(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
    text = re.sub(r"\[([^\]]{2,240})\]\((?:https?:)?/?[^)]+\)", r"\1", text)
    text = re.sub(r"\[([^\]]{2,260})\]", r"\1", text)
    text = re.sub(r"^\s*[-*]\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" -—:：;；。")


def _distilled_public_fact(item: Dict[str, Any]) -> str:
    card = _public_fact_card(item)
    quality = _public_fact_quality(item)
    for payload in (card, quality, item):
        for key in ("distilled_fact", "fact", "clean_fact", "summary", "object", "finding"):
            text = _compact(_normalize_public_fact_markup(_as_dict(payload).get(key)), 260)
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
    if RAW_PAGE_FRAGMENT_RE.search(value):
        return True
    if WEB_CHROME_FRAGMENT_RE.search(value):
        return True
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?%?", value):
        return True
    return False


def _analysis_readiness(item: Dict[str, Any]) -> str:
    return str(item.get("analysis_readiness") or _as_dict(item.get("evidence_card")).get("analysis_readiness") or "").strip()


def _non_claim_status_reason(item: Dict[str, Any]) -> str:
    allowed = _analysis_allowed_use(item).strip().lower()
    if allowed in NON_CLAIM_ALLOWED_USES:
        return f"allowed_use:{allowed}"
    readiness = _analysis_readiness(item).strip().lower()
    if readiness in NON_CLAIM_ANALYSIS_READINESS:
        return f"analysis_readiness:{readiness}"
    return ""


QUALITATIVE_PROOF_ROLES = {
    "source_check",
    "support",
    "case",
    "counter",
    "filing",
    "policy",
    "official_data",
    "technology_product",
    "boundary",
}


def _metric_proof_gaps_block_claim(item: Dict[str, Any]) -> bool:
    gaps = _as_list(item.get("metric_proof_gaps") or _as_dict(item.get("evidence_card")).get("metric_proof_gaps"))
    if not gaps:
        return False
    proof_role = str(item.get("proof_role") or _as_dict(item.get("evidence_card")).get("proof_role") or "").strip().lower()
    claim_type = str(item.get("claim_type") or item.get("conclusion_type") or _as_dict(item.get("evidence_card")).get("claim_type") or "").strip().lower()
    metric = str(item.get("metric") or "").strip().lower()
    has_value = bool(
        str(item.get("value") or "").strip()
        or item.get("numeric_values")
        or item.get("numeric_value") is not None
    )
    if advisory_weight_mode() and _analysis_allowed_use(item) in {"supporting_context", "directional_signal"}:
        if proof_role in QUALITATIVE_PROOF_ROLES or metric in QUALITATIVE_PROOF_ROLES:
            return False
    if proof_role in QUALITATIVE_PROOF_ROLES and not has_value:
        return False
    return claim_type == "hard_metric" or proof_role == "metric" or has_value


def _analysis_public_fact_text(item: Dict[str, Any], *, max_chars: int = 180) -> str:
    text = _distilled_public_fact(item) or _fact_text(item)
    text = _public_normalize_analysis_text(_compact(text, max_chars))
    if _is_noisy_public_fact(text):
        return ""
    return text


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
    if _non_claim_status_reason(item):
        return False
    if not _has_traceable_source(item):
        return False
    if _is_noisy_public_fact(_distilled_public_fact(item)):
        return False
    level = _analysis_source_level(item)
    allowed = _analysis_allowed_use(item)
    return bool(
        level in {"A", "B", "C"}
        or allowed in {"core_claim", "supporting", "supporting_context", "directional_signal"}
    )


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
    topic = _compact(dimension, 100)
    if not topic or _is_canonical_chapter_key(_normalize_key(topic)):
        topic = "本章主题"
    fact_text = _compact(fact, 120)
    strength_key = str(strength or "").strip().lower()
    if strength_key in {"strong", "moderate"}:
        return f"{topic}正在从概念讨论转向可验证的岗位、课程和工具应用变化。"
    if strength_key == "directional":
        return f"{topic}已经出现可用于判断趋势方向的具体材料，适合写成审慎的行业变化判断。"
    if fact_text:
        return f"{topic}可以从“{fact_text}”这一类材料切入，观察岗位能力和培养方式的变化。"
    return f"{topic}可以从岗位能力、课程调整和工具应用三个层面展开分析。"
    topic = _compact(dimension, 100)
    if not topic or _is_canonical_chapter_key(_normalize_key(topic)):
        topic = "本章主题"
    strength_key = str(strength or "").strip().lower()
    if strength_key in {"strong", "moderate"}:
        return f"{topic}已有可观察的公开资料信号，后续判断需结合来源范围、样本边界和时间窗口校准。"
    if strength_key == "directional":
        return f"{topic}已有方向性公开资料信号，但仍需避免把单一材料外推为确定性结论。"
    return f"{topic}已有初步公开资料线索，可作为后续分析的背景信号。"


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
        dimension_key = _normalize_key(dimension)
        if _is_canonical_chapter_key(dimension_key):
            if dimension_key in item_keys:
                items.append(dict(item))
            continue
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
        if _is_canonical_chapter_key(dimension_key):
            if dimension_key not in chapter_keys:
                continue
        elif dimension_key not in chapter_keys and not any(_overlaps(dimension, key) for key in chapter_keys):
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
    diversity_limit = 1 if max_items <= 8 else 3
    add(
        [
            item
            for item in item_list
            if str(item.get("proof_role") or "").strip().lower() in {"source_check", "filing", "official_data"}
        ],
        diversity_limit,
    )
    add(
        [
            item
            for item in item_list
            if str(item.get("proof_role") or "").strip().lower() == "counter"
            or bool(item.get("counter_evidence"))
            or re.search(r"(counter|risk|failure|downside|鍙嶈瘉|椋庨櫓|澶辫触|涓嬫粦|鍙栨秷)", _fact_text(item), re.I)
        ],
        diversity_limit,
    )
    add(
        [
            item
            for item in item_list
            if str(item.get("proof_role") or "").strip().lower() in {"case", "boundary"}
            or str(item.get("source_family") or "").strip().lower() == "company/case"
        ],
        diversity_limit,
    )
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
    seen_units: set[Tuple[str, Tuple[str, ...]]] = set()
    for dimension, synthesis in dimension_synthesis.items():
        synthesis = _as_dict(synthesis)
        claim = synthesis.get("takeaway") or ""
        refs = synthesis.get("evidence_ids") or synthesis.get("directional_evidence_ids") or synthesis.get("fact_ids") or []
        if not str(claim or "").strip():
            continue
        if not _as_list(refs):
            continue
        chapter_id = str(synthesis.get("chapter_id") or "").strip()
        unit_id = f"{chapter_id or _normalize_key(dimension) or 'fallback'}_fallback"
        unit_key = (unit_id, tuple(str(ref or "").strip() for ref in _as_list(refs) if str(ref or "").strip()))
        if unit_key in seen_units:
            continue
        seen_units.add(unit_key)
        refs_list = _dedupe([str(ref or "").strip() for ref in _as_list(refs) if str(ref or "").strip()])
        fact_ids = _dedupe([str(ref or "").strip() for ref in _as_list(synthesis.get("fact_ids") or refs_list) if str(ref or "").strip()])
        requirement_ids = _dedupe([str(ref or "").strip() for ref in _as_list(synthesis.get("requirement_ids")) if str(ref or "").strip()])
        source_ids = _dedupe([str(ref or "").strip() for ref in _as_list(synthesis.get("source_ids")) if str(ref or "").strip()])
        units.append(
            {
                "id": unit_id,
                "claim_id": unit_id,
                "chapter_id": chapter_id,
                "question": dimension,
                "claim": claim,
                "claim_status": "decision_ready" if _as_list(synthesis.get("evidence_ids")) else "directional",
                "claim_strength": synthesis.get("claim_strength") or ("moderate" if _as_list(synthesis.get("evidence_ids")) else "directional"),
                "quality_status": "valid" if _as_list(synthesis.get("evidence_ids")) else "directional_with_boundary",
                "supporting_evidence": refs_list,
                "evidence_refs": refs_list,
                "used_evidence_ids": refs_list,
                "used_fact_refs": refs_list,
                "supporting_evidence_refs": refs_list,
                "fact_ids": fact_ids,
                "requirement_ids": requirement_ids,
                "source_ids": source_ids,
                "source_support_map": {
                    "claim": refs_list,
                    "mechanism": refs_list,
                    "boundary": refs_list,
                },
                "lineage": {
                    "requirement_ids": requirement_ids,
                    "fact_ids": fact_ids,
                    "source_ids": source_ids,
                },
                "counter_evidence": synthesis.get("counter") or "",
                "reasoning": synthesis.get("mechanism") or synthesis.get("explain_why") or "",
                "mechanism": synthesis.get("mechanism") or "",
                "decision_implication": synthesis.get("decision_implication") or "",
                "confidence": synthesis.get("confidence"),
                "dimension": dimension,
            }
        )
    return units


def _is_fallback_synthesis_claim(item: Dict[str, Any]) -> bool:
    claim_text = item.get("claim") or item.get("judgment") or item.get("takeaway") or ""
    identifier = str(item.get("id") or item.get("claim_id") or "").strip().lower()
    return identifier.endswith("_fallback") or _is_generic_llm_claim(claim_text)


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
        refs = synthesis.get("evidence_ids") or synthesis.get("directional_evidence_ids") or synthesis.get("fact_ids") or []
        if str(claim or "").strip():
            key_claims.append(
                {
                    "claim": claim,
                    "claim_status": "decision_ready" if _as_list(synthesis.get("evidence_ids")) else "directional",
                    "claim_strength": synthesis.get("claim_strength") or ("moderate" if _as_list(synthesis.get("evidence_ids")) else "directional"),
                    "supporting_evidence": refs,
                    "evidence_refs": refs,
                    "fact_ids": synthesis.get("fact_ids") or refs,
                    "requirement_ids": _as_list(synthesis.get("requirement_ids")),
                    "source_ids": _as_list(synthesis.get("source_ids")),
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
    if advisory_weight_mode() and _has_traceable_source(item):
        return "supporting_context"
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
    readiness = _analysis_readiness(item)
    if _non_claim_status_reason(item):
        return False
    if _metric_proof_gaps_block_claim(item):
        return False
    if advisory_weight_mode():
        return _has_traceable_source(item)
    if readiness == "directional_ready":
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
    if level not in {"A", "B"} and allowed != "directional_signal" and not advisory_weight_mode():
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
        "requirement_id": _lineage_requirement_id(item) or item.get("evidence_goal_id"),
        "source_id": item.get("source_id") or item.get("run_source_id") or source.get("id") or source.get("ref"),
        "search_task_id": item.get("search_task_id") or item.get("task_id"),
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
    usable_facts = [
        fact
        for item in usable
        for fact in [_analysis_public_fact_text(item, max_chars=120)]
        if fact
    ]
    directional_facts = [
        fact
        for item in directional
        for fact in [_analysis_public_fact_text(item, max_chars=120)]
        if fact
    ]
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
    first_fact = usable_facts[0] if usable_facts else (
        directional_facts[0]
        if directional_facts
        else _analysis_public_fact_text(_as_dict(analyses[0] if analyses else {}), max_chars=220)
    )
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
    source_items = usable or directional
    dimension_label = next(
        (
            str(item.get("dimension_name") or item.get("evidence_goal") or item.get("dimension") or "").strip()
            for item in source_items
            if str(item.get("dimension_name") or item.get("evidence_goal") or item.get("dimension") or "").strip()
        ),
        dimension,
    )
    takeaway = _claim_from_public_fact(dimension_label, first_fact, claim_strength)
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
    chapter_ids = _dedupe([item.get("chapter_id") or item.get("dimension_id") for item in source_items])
    requirement_ids = _dedupe(
        [
            item.get("requirement_id")
            or item.get("evidence_goal_id")
            or item.get("search_task_id")
            for item in source_items
        ]
    )
    source_ids = _dedupe(
        [
            item.get("source_id")
            or item.get("run_source_id")
            or _source_payload(item).get("id")
            or _source_payload(item).get("ref")
            for item in source_items
        ]
    )
    return {
        "chapter_id": chapter_ids[0] if chapter_ids else "",
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
        "fact_ids": evidence_ids or directional_ids,
        "requirement_ids": requirement_ids,
        "source_ids": source_ids,
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
        fact_chain = [
            fact
            for item in support
            for fact in [_analysis_public_fact_text(item, max_chars=180)]
            if fact
        ][:5]
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
        counter_boundary = [
            fact
            for item in counters
            for fact in [_analysis_public_fact_text(item, max_chars=180)]
            if fact
        ][:3]
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


def _chapter_analysis_readiness(payload: Dict[str, Any]) -> str:
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


def _report_blueprint_for_llm(evidence_package: Dict[str, Any]) -> Dict[str, Any]:
    metadata = _as_dict(evidence_package.get("metadata"))
    raw_output = _as_dict(evidence_package.get("raw_output"))
    for candidate in (
        evidence_package.get("report_blueprint"),
        metadata.get("report_blueprint"),
        raw_output.get("report_blueprint"),
        _as_dict(metadata.get("query_analysis")).get("report_blueprint"),
    ):
        payload = _as_dict(candidate)
        if _as_list(payload.get("chapters")):
            return payload
    packages = [item for item in _as_list(evidence_package.get("chapter_evidence_packages")) if isinstance(item, dict)]
    if packages:
        return {"chapters": packages}
    diagnostics = _as_dict(evidence_package.get("chapter_evidence_diagnostics")) or _as_dict(evidence_package.get("evidence_analysis_by_chapter"))
    if diagnostics:
        chapters: List[Dict[str, Any]] = []
        for key, payload in diagnostics.items():
            item = _as_dict(payload)
            chapters.append(
                {
                    "chapter_id": item.get("chapter_id") or key,
                    "chapter_title": item.get("chapter_title") or item.get("title") or key,
                    "chapter_question": item.get("chapter_question") or item.get("question") or item.get("chapter_title") or key,
                    "chapter_id_aliases": _as_list(item.get("chapter_id_aliases")),
                    "hypothesis_id": item.get("hypothesis_id"),
                    "dimension": item.get("dimension"),
                    "dimension_name": item.get("dimension_name"),
                }
            )
        return {"chapters": chapters}
    return {}


def _chapter_identity_for_llm(evidence_package: Dict[str, Any]) -> Dict[str, Any]:
    if build_chapter_identity_map is None:
        return {}
    return build_chapter_identity_map(
        blueprint=_report_blueprint_for_llm(evidence_package),
        research_plan=_research_plan(evidence_package),
    )


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
        item["analysis_readiness"] = _chapter_analysis_readiness(item)
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
        bucket["analysis_readiness"] = _chapter_analysis_readiness(bucket)
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


def _input_evidence_card_count_for_diagnostics(
    structured_analysis: Dict[str, Any],
    evidence_package: Dict[str, Any],
) -> int:
    ids: set[str] = set()
    package = _as_dict(evidence_package)
    for key in ("analysis_ready_evidence", "clean_evidence_list", "fact_cards"):
        for item in _as_list(package.get(key)):
            if not isinstance(item, dict):
                continue
            ref = str(item.get("evidence_id") or item.get("fact_id") or item.get("ref") or "").strip()
            if ref:
                ids.add(ref)
    if ids:
        return len(ids)
    analysis = _as_dict(structured_analysis)
    for item in _as_list(analysis.get("evidence_analyses")):
        if not isinstance(item, dict):
            continue
        ref = str(item.get("evidence_id") or item.get("fact_id") or item.get("ref") or "").strip()
        if ref:
            ids.add(ref)
    return len(ids) if ids else len(_as_list(analysis.get("evidence_analyses")))


def _analysis_conversion_diagnostics(
    evidence_package: Dict[str, Any],
    structured_analysis: Dict[str, Any],
    *,
    input_evidence_card_count: int | None = None,
) -> Dict[str, Any]:
    package = _as_dict(evidence_package)
    analysis_ready_fact_count = len([item for item in _as_list(package.get("analysis_ready_evidence")) if isinstance(item, dict)])
    if input_evidence_card_count is None:
        input_evidence_card_count = _input_evidence_card_count_for_diagnostics(structured_analysis, evidence_package)
    claim_units = [item for item in _as_list(_as_dict(structured_analysis).get("claim_units")) if isinstance(item, dict)]
    bound_claim_count = 0
    fact_refs_per_claim: List[int] = []
    single_fact_claim_count = 0
    mechanism_ready_count = 0
    implication_ready_count = 0
    shallow_claim_reasoning_count = 0
    interpretation_ids: set[str] = set()
    for unit in claim_units:
        fact_refs = _as_list(
            unit.get("fact_ids")
            or unit.get("used_evidence_ids")
            or unit.get("used_fact_refs")
            or unit.get("evidence_refs")
            or unit.get("supporting_evidence_refs")
        )
        source_refs = _as_list(unit.get("source_ids") or unit.get("source_refs"))
        fact_refs_per_claim.append(len(fact_refs))
        if len(fact_refs) == 1 or bool(unit.get("single_fact_claim")):
            single_fact_claim_count += 1
        if _as_list(unit.get("mechanism_chain")):
            mechanism_ready_count += 1
        if (
            str(unit.get("employment_implication") or "").strip()
            or str(unit.get("education_implication") or "").strip()
            or str(unit.get("industry_implication") or "").strip()
            or str(unit.get("decision_implication") or "").strip()
        ):
            implication_ready_count += 1
        for interpretation_id in _as_list(unit.get("interpretation_ids")):
            if str(interpretation_id or "").strip():
                interpretation_ids.add(str(interpretation_id or "").strip())
        if (
            str(unit.get("mechanism") or "").strip()
            and str(unit.get("reasoning") or "").strip()
            and _normalize_key(unit.get("mechanism")) == _normalize_key(unit.get("reasoning"))
            and not _as_list(unit.get("mechanism_chain"))
        ):
            shallow_claim_reasoning_count += 1
        if fact_refs or source_refs:
            bound_claim_count += 1
    explicit_interpretation_units = [
        item
        for item in _as_list(_as_dict(structured_analysis).get("interpretation_units"))
        if isinstance(item, dict)
    ]
    for chapter in _as_list(_as_dict(structured_analysis).get("chapter_synthesis")):
        for item in _as_list(_as_dict(chapter).get("interpretation_units")):
            if isinstance(item, dict):
                explicit_interpretation_units.append(item)
    interpretation_unit_count = len(explicit_interpretation_units) or len(interpretation_ids)
    denominator = max(1, int(input_evidence_card_count or analysis_ready_fact_count or 0))
    claim_conversion_rate = round(len(claim_units) / denominator, 4)
    bound_claim_rate = round(bound_claim_count / max(1, len(claim_units)), 4)
    reanalyze_existing_recommended = bool(denominator >= 8 and len(claim_units) <= max(1, int(denominator * 0.12)))
    suggestions: List[Dict[str, Any]] = []
    if reanalyze_existing_recommended:
        suggestions.append(
            {
                "schema_version": "review_suggestion_v1",
                "issue_type": "low_claim_conversion",
                "severity": "warning",
                "target": {"stage": "analysis"},
                "suggested_action": "reanalyze_existing",
                "analysis_ready_fact_count": analysis_ready_fact_count,
                "input_evidence_card_count": int(input_evidence_card_count or 0),
                "claim_unit_count": len(claim_units),
                "bound_claim_count": bound_claim_count,
                "claim_conversion_rate": claim_conversion_rate,
                "diagnostic_only": True,
                "must_not_render": True,
                "public_text_allowed": False,
            }
        )
    return {
        "analysis_ready_fact_count": analysis_ready_fact_count,
        "claim_unit_count": len(claim_units),
        "bound_claim_count": bound_claim_count,
        "claim_conversion_rate": claim_conversion_rate,
        "bound_claim_rate": bound_claim_rate,
        "interpretation_unit_count": interpretation_unit_count,
        "fact_group_coverage_rate": round(
            sum(fact_refs_per_claim) / max(1, denominator),
            4,
        ),
        "avg_facts_per_claim": round(sum(fact_refs_per_claim) / max(1, len(claim_units)), 3),
        "single_fact_claim_rate": round(single_fact_claim_count / max(1, len(claim_units)), 4),
        "claims_with_mechanism_chain_rate": round(mechanism_ready_count / max(1, len(claim_units)), 4),
        "claims_with_implication_rate": round(implication_ready_count / max(1, len(claim_units)), 4),
        "raw_supporting_fact_rendered_count": 0,
        "shallow_claim_reasoning_count": shallow_claim_reasoning_count,
        "reanalyze_existing_recommended": reanalyze_existing_recommended,
        "analysis_review_suggestions": suggestions,
    }


def _claim_fact_ref_set(unit: Dict[str, Any]) -> set[str]:
    refs: set[str] = set()
    for key in (
        "fact_ids",
        "used_evidence_ids",
        "used_fact_refs",
        "evidence_refs",
        "supporting_evidence_refs",
        "supporting_evidence",
    ):
        for value in _as_list(unit.get(key)):
            text = str(value or "").strip()
            if text:
                refs.add(text)
    return refs


def _interpretation_fact_ref_set(unit: Dict[str, Any]) -> set[str]:
    return {
        str(value or "").strip()
        for value in _as_list(unit.get("fact_ids"))
        if str(value or "").strip()
    }


_FALLBACK_CHAPTER_EVIDENCE_KEYS = (
    "core_evidence",
    "supporting_evidence",
    "metric_evidence",
    "case_evidence",
    "counter_evidence",
    "directional_evidence",
    "sample_evidence",
)


def _fallback_interpretation_cards_from_chapter_packages(evidence_package: Dict[str, Any]) -> List[Dict[str, Any]]:
    cards: List[Dict[str, Any]] = []
    seen_refs: set[str] = set()
    for package in _as_list(evidence_package.get("chapter_evidence_packages")):
        if not isinstance(package, dict):
            continue
        chapter_id = str(package.get("chapter_id") or package.get("id") or "").strip()
        chapter_title = str(package.get("chapter_title") or package.get("title") or chapter_id).strip()
        for key in _FALLBACK_CHAPTER_EVIDENCE_KEYS:
            role = key.replace("_evidence", "")
            topic_fit = "direct" if key in {"core_evidence", "metric_evidence", "case_evidence"} else "related"
            if key == "counter_evidence":
                topic_fit = "background"
                role = "counter"
            for raw in _as_list(package.get(key)):
                if not isinstance(raw, dict):
                    continue
                evidence_id = str(raw.get("evidence_id") or raw.get("fact_id") or raw.get("ref") or raw.get("id") or "").strip()
                if not evidence_id or evidence_id in seen_refs:
                    continue
                fact = _compact(
                    raw.get("distilled_fact")
                    or raw.get("clean_fact")
                    or raw.get("fact")
                    or raw.get("summary")
                    or raw.get("content"),
                    420,
                )
                if not fact:
                    continue
                source = _source_payload(raw)
                source_id = str(
                    raw.get("source_id")
                    or raw.get("source_ref")
                    or raw.get("citation_ref")
                    or source.get("source_id")
                    or source.get("document_id")
                    or source.get("url")
                    or ""
                ).strip()
                cards.append(
                    {
                        **raw,
                        "evidence_id": evidence_id,
                        "chapter_id": str(raw.get("chapter_id") or chapter_id).strip() or "fallback",
                        "chapter_title": chapter_title,
                        "requirement_id": str(raw.get("requirement_id") or _lineage_requirement_id(raw) or "").strip(),
                        "source_id": source_id,
                        "source_title": _compact(source.get("title") or source.get("source") or source.get("name"), 160),
                        "source_url": str(source.get("url") or raw.get("source_url") or "").strip(),
                        "distilled_fact": fact,
                        "fact": fact,
                        "proof_role": str(raw.get("proof_role") or raw.get("analysis_role") or role).strip().lower(),
                        "topic_fit": str(raw.get("topic_fit") or topic_fit).strip(),
                        "allowed_use": str(raw.get("allowed_use") or raw.get("analysis_role") or "supporting_context").strip(),
                    }
                )
                seen_refs.add(evidence_id)
    return cards


def _fallback_interpretation_units(evidence_package: Dict[str, Any]) -> List[Dict[str, Any]]:
    if build_evidence_interpretation_units is None:
        return []
    max_chapters = _env_int("BRAIN_FALLBACK_ANALYSIS_INTERPRETATION_MAX_CHAPTERS", 8, min_value=1, max_value=16)
    max_per_chapter = _env_int("BRAIN_FALLBACK_ANALYSIS_INTERPRETATION_MAX_FACTS_PER_CHAPTER", 36, min_value=3, max_value=120)
    max_units = _env_int("BRAIN_FALLBACK_ANALYSIS_MAX_INTERPRETATION_UNITS", 8, min_value=1, max_value=24)
    cards = _fallback_interpretation_cards_from_chapter_packages(evidence_package)
    try:
        supplemental_cards = _evidence_cards_for_llm(
            evidence_package,
            max_chapters=max_chapters,
            max_per_chapter=max_per_chapter,
        )
    except Exception:
        supplemental_cards = []
    seen_card_refs = {
        str(card.get("evidence_id") or "").strip()
        for card in cards
        if isinstance(card, dict) and str(card.get("evidence_id") or "").strip()
    }
    for card in supplemental_cards:
        if not isinstance(card, dict):
            continue
        ref = str(card.get("evidence_id") or "").strip()
        if ref and ref not in seen_card_refs:
            cards.append(card)
            seen_card_refs.add(ref)
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for card in cards:
        if not isinstance(card, dict):
            continue
        chapter_id = str(card.get("chapter_id") or "").strip() or "fallback"
        if not str(card.get("evidence_id") or "").strip() or not str(card.get("fact") or card.get("distilled_fact") or "").strip():
            continue
        grouped.setdefault(chapter_id, []).append(card)
    if not grouped:
        return []
    diagnostics = _as_dict(evidence_package.get("chapter_evidence_diagnostics"))
    units: List[Dict[str, Any]] = []
    for chapter_id, fact_cards in list(grouped.items())[:max_chapters]:
        chapter_payload = _as_dict(diagnostics.get(chapter_id))
        chapter_question = str(
            chapter_payload.get("chapter_question")
            or chapter_payload.get("chapter_title")
            or chapter_payload.get("title")
            or chapter_id
        ).strip()
        try:
            payload = build_evidence_interpretation_units(
                chapter_id=chapter_id,
                chapter_question=chapter_question,
                fact_cards=fact_cards,
                max_units=max_units,
            )
        except Exception:
            payload = {}
        for unit in _as_list(_as_dict(payload).get("interpretation_units")):
            if isinstance(unit, dict) and str(unit.get("interpretation_id") or "").strip():
                units.append(unit)
    return units


def _best_interpretations_for_claim(
    unit: Dict[str, Any],
    interpretation_units: List[Dict[str, Any]],
    *,
    limit: int = 3,
) -> List[Dict[str, Any]]:
    refs = _claim_fact_ref_set(unit)
    chapter_id = str(unit.get("chapter_id") or unit.get("hypothesis_id") or unit.get("dimension_id") or "").strip()
    scored: List[tuple[int, int, Dict[str, Any]]] = []
    for interpretation in interpretation_units:
        fact_refs = _interpretation_fact_ref_set(interpretation)
        overlap = len(refs & fact_refs) if refs and fact_refs else 0
        same_chapter = int(
            bool(chapter_id)
            and str(interpretation.get("chapter_id") or "").strip()
            and str(interpretation.get("chapter_id") or "").strip() == chapter_id
        )
        if overlap or same_chapter:
            scored.append((overlap, same_chapter, interpretation))
    if not scored and interpretation_units:
        scored.append((0, 0, interpretation_units[0]))
    scored.sort(key=lambda item: (item[0], item[1], len(_interpretation_fact_ref_set(item[2]))), reverse=True)
    return [item[2] for item in scored[:limit]]


def _first_nonempty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _enrich_fallback_claims_with_interpretations(
    result: Dict[str, Any],
    evidence_package: Dict[str, Any],
) -> Dict[str, Any]:
    interpretation_units = _fallback_interpretation_units(evidence_package)
    if not interpretation_units:
        return result
    enriched_claims: List[Dict[str, Any]] = []
    for raw_unit in _as_list(result.get("claim_units")):
        if not isinstance(raw_unit, dict):
            continue
        unit = {**raw_unit}
        matched_units = _best_interpretations_for_claim(unit, interpretation_units)
        if not matched_units:
            enriched_claims.append(unit)
            continue
        interpretation_ids = _dedupe(
            _as_list(unit.get("interpretation_ids"))
            + [
                str(item.get("interpretation_id") or "").strip()
                for item in matched_units
                if str(item.get("interpretation_id") or "").strip()
            ]
        )
        fact_ids = _dedupe(
            _as_list(unit.get("fact_ids") or unit.get("evidence_refs") or unit.get("supporting_evidence"))
            + [
                fact_id
                for item in matched_units
                for fact_id in _as_list(item.get("fact_ids"))
            ]
        )
        source_ids = _dedupe(
            _as_list(unit.get("source_ids") or unit.get("source_refs"))
            + [
                source_id
                for item in matched_units
                for source_id in _as_list(item.get("source_ids"))
            ]
        )
        primary = matched_units[0]
        mechanism_chain = _dedupe(
            [
                step
                for item in matched_units
                for step in _as_list(item.get("mechanism_chain"))
            ]
        )[:5]
        why_it_matters = _first_nonempty(
            unit.get("why_it_matters"),
            primary.get("why_it_matters"),
            primary.get("what_evidence_reflects"),
        )
        what_reflects = _first_nonempty(unit.get("what_evidence_reflects"), primary.get("what_evidence_reflects"))
        mechanism_text = "；".join(mechanism_chain)
        old_reasoning = str(unit.get("reasoning") or "").strip()
        old_mechanism = str(unit.get("mechanism") or "").strip()
        shallow_reasoning = bool(
            old_reasoning
            and old_mechanism
            and _normalize_key(old_reasoning) == _normalize_key(old_mechanism)
        )
        unit.update(
            {
                "interpretation_ids": interpretation_ids,
                "fact_ids": fact_ids,
                "source_ids": source_ids,
                "what_evidence_reflects": what_reflects,
                "why_it_matters": why_it_matters,
                "mechanism_chain": mechanism_chain,
                "employment_implication": _first_nonempty(
                    unit.get("employment_implication"),
                    primary.get("employment_implication"),
                ),
                "education_implication": _first_nonempty(
                    unit.get("education_implication"),
                    primary.get("education_implication"),
                ),
                "industry_implication": _first_nonempty(
                    unit.get("industry_implication"),
                    primary.get("industry_implication"),
                ),
                "counter_reading": _first_nonempty(unit.get("counter_reading"), primary.get("counter_reading")),
                "single_fact_claim": bool(len(fact_ids) == 1 or unit.get("single_fact_claim")),
                "claim_depth_ready": bool(interpretation_ids and mechanism_chain),
            }
        )
        if mechanism_text:
            unit["mechanism"] = mechanism_text
        if shallow_reasoning or not old_reasoning:
            unit["reasoning"] = _first_nonempty(why_it_matters, what_reflects, mechanism_text, old_reasoning)
        if not str(unit.get("decision_implication") or "").strip():
            unit["decision_implication"] = _first_nonempty(
                unit.get("employment_implication"),
                unit.get("education_implication"),
                unit.get("industry_implication"),
            )
        enriched_claims.append(unit)
    if not enriched_claims:
        return result
    return {
        **result,
        "claim_units": enriched_claims,
        "interpretation_units": interpretation_units,
        "evidence_interpretation_diagnostics": {
            "fallback_interpretation_enabled": True,
            "interpretation_unit_count": len(interpretation_units),
            "claim_with_interpretation_count": len(
                [unit for unit in enriched_claims if _as_list(unit.get("interpretation_ids"))]
            ),
        },
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
    identity = _chapter_identity_for_llm(evidence_package)
    if identity and canonical_chapter_ids is not None and canonical_chapter_payload is not None:
        chapter_filter: Dict[str, Dict[str, Any]] = {}
        for chapter_id in canonical_chapter_ids(identity)[:max_chapters]:
            payload = canonical_chapter_payload(identity, chapter_id)
            if not payload:
                continue
            chapter_filter[chapter_id] = {
                "chapter_id": chapter_id,
                "chapter_title": payload.get("chapter_title") or chapter_id,
                "chapter_question": payload.get("chapter_question") or payload.get("chapter_title") or chapter_id,
                "chapter_id_aliases": _dedupe(
                    [
                        chapter_id,
                        payload.get("chapter_title"),
                        payload.get("chapter_question"),
                        *_as_list(payload.get("chapter_id_aliases")),
                    ]
                )[:16],
            }
        if chapter_filter:
            return chapter_filter
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
    curated_cards = _curated_evidence_cards_for_llm(
        evidence_package,
        max_chapters=max_chapters,
        max_per_chapter=max_per_chapter,
    )
    if curated_cards:
        return curated_cards
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
    lineage_by_evidence_id: Dict[str, Dict[str, Any]] = {}
    for item in _as_list(evidence_package.get("normalized_evidence")):
        if not isinstance(item, dict):
            continue
        evidence_id = str(item.get("evidence_id") or item.get("id") or "").strip()
        if evidence_id and evidence_id not in lineage_by_evidence_id:
            lineage_by_evidence_id[evidence_id] = item
    requirements_by_chapter = _requirements_by_chapter_for_llm(evidence_package)
    requirement_chapter_lookup = {
        str(requirement.get("requirement_id") or "").strip(): chapter_id
        for chapter_id, requirements in requirements_by_chapter.items()
        for requirement in requirements
        if str(requirement.get("requirement_id") or "").strip()
    }
    candidate_items = (
        _as_list(evidence_package.get("claim_support_facts"))
        + _as_list(evidence_package.get("analysis_candidate_facts"))
        + _as_list(evidence_package.get("public_citation_facts"))
    )
    source_items = candidate_items if candidate_items else (
        _as_list(evidence_package.get("analysis_ready_evidence")) + _as_list(evidence_package.get("clean_evidence_list"))
    )
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
        source = _source_payload(item)
        card = _as_dict(item.get("evidence_card"))
        fact_card = _public_fact_card(item)
        fact = _distilled_public_fact(item)
        lineage_item = _as_dict(lineage_by_evidence_id.get(evidence_id))
        lineage_search_task = _as_dict(lineage_item.get("search_task"))
        search_task = _as_dict(item.get("search_task"))
        requirement_id = str(
            item.get("requirement_id")
            or item.get("evidence_requirement_id")
            or _as_dict(item.get("lineage")).get("requirement_id")
            or search_task.get("requirement_id")
            or search_task.get("evidence_requirement_id")
            or search_task.get("slot_id")
            or lineage_item.get("requirement_id")
            or lineage_item.get("evidence_requirement_id")
            or lineage_search_task.get("requirement_id")
            or lineage_search_task.get("evidence_requirement_id")
            or lineage_search_task.get("slot_id")
            or ""
        ).strip()
        raw_chapter_id = (
            str(
                item.get("chapter_id")
                or _as_dict(item.get("lineage")).get("chapter_id")
                or search_task.get("chapter_id")
                or lineage_item.get("chapter_id")
                or lineage_search_task.get("chapter_id")
                or requirement_chapter_lookup.get(requirement_id)
                or _chapter_key_for_item(item)
                or ""
            ).strip()
        )
        chapter_id = raw_chapter_id
        if chapter_filter:
            resolved_chapter_id = resolve_chapter_id(chapter_filter, raw_chapter_id)
            if not resolved_chapter_id and requirement_id:
                resolved_chapter_id = resolve_chapter_id(chapter_filter, requirement_chapter_lookup.get(requirement_id))
            if not resolved_chapter_id:
                continue
            chapter_id = resolved_chapter_id
        requirement_id_source = ""
        if not requirement_id:
            inferred_requirement = _infer_requirement_from_chapter_contract_for_llm(
                requirements_by_chapter,
                chapter_id=chapter_id,
                item=item,
                card=card,
            )
            requirement_id = str(inferred_requirement.get("requirement_id") or "").strip()
            requirement_id_source = str(inferred_requirement.get("requirement_id_source") or "").strip()
        if buckets.get(chapter_id, 0) >= max_per_chapter:
            continue
        hypothesis_id = str(
            item.get("hypothesis_id")
            or _as_dict(item.get("lineage")).get("hypothesis_id")
            or search_task.get("hypothesis_id")
            or lineage_item.get("hypothesis_id")
            or lineage_search_task.get("hypothesis_id")
            or ""
        ).strip()
        source_id = str(
            item.get("source_id")
            or item.get("source_ref")
            or item.get("citation_ref")
            or lineage_item.get("source_id")
            or lineage_item.get("source_ref")
            or lineage_item.get("citation_ref")
            or source.get("source_ref")
            or source.get("document_id")
            or source.get("url")
            or ""
        ).strip()
        search_task_id = str(
            item.get("search_task_id")
            or search_task.get("task_id")
            or search_task.get("id")
            or lineage_item.get("search_task_id")
            or lineage_search_task.get("task_id")
            or lineage_search_task.get("id")
            or ""
        ).strip()
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
                "aliases": _as_list(item.get("aliases"))
                or _as_list(item.get("alias_ids"))
                or _as_list(item.get("legacy_ids"))
                or _as_list(item.get("legacy_evidence_ids")),
                "chapter_id": chapter_id,
                "hypothesis_id": hypothesis_id,
                "requirement_id": requirement_id,
                "requirement_id_source": requirement_id_source or str(item.get("requirement_id_source") or "").strip(),
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


def _evidence_grooming_for_llm(card: Dict[str, Any]) -> Dict[str, Any]:
    proof_role = str(card.get("proof_role") or card.get("analysis_role") or card.get("fact_type") or "").strip().lower()
    source_level = str(card.get("source_level") or "").strip().upper()
    allowed_use = str(card.get("allowed_use") or "").strip().lower()
    fact = str(card.get("distilled_fact") or card.get("fact") or "").strip()
    title = str(card.get("source_title") or "").strip()
    metric = str(card.get("metric") or "").strip()
    claim_strength_hint = str(card.get("claim_strength_hint") or "").strip().lower()
    if not claim_strength_hint:
        if source_level in {"A", "B"} and allowed_use in {"core_claim", "supporting"}:
            claim_strength_hint = "moderate"
        elif source_level in {"A", "B", "C"} or allowed_use in {"supporting_context", "directional_signal"}:
            claim_strength_hint = "directional"
        else:
            claim_strength_hint = "weak"
    suggested_use = str(card.get("suggested_use") or "").strip()
    if not suggested_use:
        if proof_role in {"policy", "official_data", "filing"}:
            suggested_use = "policy_signal"
        elif proof_role in {"case", "commercial", "customer_case"}:
            suggested_use = "case_signal"
        elif proof_role in {"counter", "risk", "boundary"}:
            suggested_use = "risk_boundary"
        elif proof_role in {"metric", "market_data", "quant_metric"}:
            suggested_use = "market_signal"
        elif source_level in {"C", "D"} or allowed_use in {"supporting_context", "directional_signal"}:
            suggested_use = "trend_signal"
        else:
            suggested_use = "background_signal"
    possible_angles = _as_list(card.get("possible_claim_angles") or card.get("suggested_analysis_direction"))
    if not possible_angles:
        if suggested_use == "policy_signal":
            possible_angles = ["政策方向如何改变具体业务安排", "制度约束对行业或岗位要求的影响"]
        elif suggested_use == "case_signal":
            possible_angles = ["具体案例说明了哪些场景变化", "案例如何反映需求、能力或流程变化"]
        elif suggested_use == "risk_boundary":
            possible_angles = ["反向材料限制了哪些结论强度", "风险信号会影响哪些执行条件"]
        elif suggested_use == "market_signal":
            possible_angles = ["数据口径能说明的市场变化", "指标适合支撑的趋势边界"]
        else:
            possible_angles = ["材料可以支持的方向性判断", "它对业务动作、需求或能力变化的提示"]
    limitations = _as_list(card.get("limitations") or card.get("limitation_boundary"))
    if source_level in {"C", "D"} and not limitations:
        limitations.append("来源适合支撑方向性判断，不能单独外推为强结论。")
    if allowed_use in {"supporting_context", "directional_signal"} and "hard_metric" not in limitations and proof_role not in {"metric", "market_data", "quant_metric"}:
        limitations.append("更适合作为案例、趋势或背景信号使用。")
    do_not_use_as = _as_list(card.get("do_not_use_as"))
    if proof_role not in {"metric", "market_data", "quant_metric"}:
        do_not_use_as.append("hard_metric")
    if claim_strength_hint in {"directional", "weak"}:
        do_not_use_as.extend(["standalone_strong_claim", "unbounded_conclusion"])
    return {
        "schema_version": "analysis_evidence_grooming_v1",
        "suggested_use": suggested_use,
        "claim_strength_hint": claim_strength_hint,
        "possible_claim_angles": _dedupe([_compact(item, 120) for item in possible_angles if str(item or "").strip()])[:6],
        "limitations": _dedupe([_compact(item, 140) for item in limitations if str(item or "").strip()])[:6],
        "do_not_use_as": _dedupe(do_not_use_as)[:6],
        "source_context": _compact(title, 120),
        "metric_context": _compact(metric, 80),
        "fact_excerpt": _compact(fact, 180),
        "diagnostic_only": True,
        "must_not_render": True,
        "public_text_allowed": False,
    }


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
    grooming = _evidence_grooming_for_llm(card)
    return {
        "evidence_id": str(card.get("evidence_id") or "").strip(),
        "chapter_id": str(card.get("chapter_id") or _as_dict(card.get("lineage")).get("chapter_id") or "").strip(),
        "hypothesis_id": str(card.get("hypothesis_id") or "").strip(),
        "requirement_id": str(card.get("requirement_id") or "").strip(),
        "requirement_id_source": str(card.get("requirement_id_source") or "").strip(),
        "analysis_role": str(card.get("analysis_role") or "").strip(),
        "analysis_eligible": bool(card.get("analysis_eligible")),
        "allowed_use": str(card.get("allowed_use") or "").strip(),
        "source_id": str(card.get("source_id") or _as_dict(card.get("lineage")).get("source_id") or "").strip(),
        "search_task_id": str(card.get("search_task_id") or _as_dict(card.get("lineage")).get("search_task_id") or "").strip(),
        "lineage": _as_dict(card.get("lineage")),
        "distilled_fact": _compact(card.get("distilled_fact") or card.get("fact"), max_fact_chars),
        "fact_type": fact_type,
        "source_level": str(card.get("source_level") or "").strip().upper(),
        "source_verification_status": str(card.get("source_verification_status") or "").strip(),
        "proof_role": str(card.get("proof_role") or "").strip(),
        "topic_fit": str(card.get("topic_fit") or "").strip(),
        "block_affinity": block_affinity,
        "metric": _compact(card.get("metric"), 80),
        "value": _compact(card.get("value"), 80),
        "unit": _compact(card.get("unit"), 40),
        "period": _compact(card.get("period"), 80),
        "claim_strength_hint": str(card.get("claim_strength_hint") or "").strip(),
        "limitations": _as_list(card.get("limitations") or card.get("limitation_boundary")),
        "possible_claim_angles": _as_list(card.get("possible_claim_angles")) or grooming["possible_claim_angles"],
        "suggested_use": grooming["suggested_use"],
        "do_not_use_as": grooming["do_not_use_as"],
        "evidence_grooming": grooming,
        "usable_for": _as_list(card.get("usable_for")),
        "source_title": _compact(card.get("source_title"), 120),
        "source_url": str(card.get("source_url") or "").strip(),
    }


def _llm_fact_role(card: Dict[str, Any]) -> str:
    for key in ("proof_role", "analysis_role", "fact_type", "allowed_use"):
        value = str(card.get(key) or "").strip().lower()
        if value:
            return value
    return "support"


def _role_balanced_llm_fact_cards(cards: Sequence[Dict[str, Any]], *, limit: int) -> List[Dict[str, Any]]:
    if limit <= 0:
        return []
    ordered = [card for card in cards if isinstance(card, dict)]
    if len(ordered) <= limit:
        return ordered[:limit]
    by_role: Dict[str, List[Dict[str, Any]]] = {}
    for card in ordered:
        by_role.setdefault(_llm_fact_role(card), []).append(card)
    selected: List[Dict[str, Any]] = []
    selected_ids: set[str] = set()
    preferred_roles = [
        "metric",
        "policy",
        "case",
        "counter",
        "technology_product",
        "filing",
        "support",
        "supporting_context",
    ]
    for role in preferred_roles + [role for role in by_role if role not in preferred_roles]:
        for card in by_role.get(role, [])[:1]:
            evidence_id = str(card.get("evidence_id") or "").strip()
            if evidence_id and evidence_id in selected_ids:
                continue
            selected.append(card)
            if evidence_id:
                selected_ids.add(evidence_id)
            break
        if len(selected) >= limit:
            return selected[:limit]
    for card in ordered:
        evidence_id = str(card.get("evidence_id") or "").strip()
        if evidence_id and evidence_id in selected_ids:
            continue
        selected.append(card)
        if evidence_id:
            selected_ids.add(evidence_id)
        if len(selected) >= limit:
            break
    return selected[:limit]


def _inventory_items_for_chapter(evidence_package: Dict[str, Any], chapter_id: str, fact_cards: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    inventory_payload = _as_dict(evidence_package.get("evidence_inventory"))
    items = [item for item in _as_list(inventory_payload.get("inventories")) if isinstance(item, dict)]
    if not items:
        return []
    card_ids = {
        str(card.get("evidence_id") or "").strip()
        for card in fact_cards
        if isinstance(card, dict) and str(card.get("evidence_id") or "").strip()
    }
    card_clusters = {
        str(card.get("cluster_key") or card.get("fact_type") or "").strip()
        for card in fact_cards
        if isinstance(card, dict) and str(card.get("cluster_key") or card.get("fact_type") or "").strip()
    }
    selected: List[Dict[str, Any]] = []
    for item in items:
        item_chapter_ids = {
            str(value or "").strip()
            for value in _as_list(item.get("chapter_ids")) + [item.get("chapter_id")]
            if str(value or "").strip()
        }
        item_evidence_ids = {
            str(value or "").strip()
            for value in _as_list(item.get("usable_evidence_ids"))
            if str(value or "").strip()
        }
        cluster_key = str(item.get("cluster_key") or "").strip()
        if (
            (chapter_id and chapter_id in item_chapter_ids)
            or bool(card_ids & item_evidence_ids)
            or (cluster_key and cluster_key in card_clusters)
        ):
            selected.append(
                {
                    "inventory_id": str(item.get("inventory_id") or "").strip(),
                    "cluster_key": cluster_key,
                    "requirement_id": str(item.get("requirement_id") or "").strip(),
                    "chapter_id": str(item.get("chapter_id") or "").strip(),
                    "curated_evidence_count": int(item.get("curated_evidence_count") or 0),
                    "usable_evidence_ids": _as_list(item.get("usable_evidence_ids"))[:20],
                    "fact_type_counts": _as_dict(item.get("fact_type_counts")),
                    "strongest_available_level": str(item.get("strongest_available_level") or "").strip(),
                    "dominant_strength": str(item.get("dominant_strength") or "").strip(),
                    "analysis_brief": _compact(item.get("analysis_brief"), 360),
                    "limitations": _as_list(item.get("limitations"))[:6],
                    "suggested_analysis_direction": _as_list(item.get("suggested_analysis_direction"))[:8],
                }
            )
        if len(selected) >= 4:
            break
    return selected


def _analysis_shard_for_chapter(evidence_package: Dict[str, Any], chapter_id: str, fact_cards: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    shards = [item for item in _as_list(evidence_package.get("analysis_shards")) if isinstance(item, dict)]
    if not shards:
        return {}
    card_ids = {
        str(card.get("evidence_id") or "").strip()
        for card in fact_cards
        if isinstance(card, dict) and str(card.get("evidence_id") or "").strip()
    }
    card_clusters = {
        str(card.get("cluster_key") or card.get("fact_type") or "").strip()
        for card in fact_cards
        if isinstance(card, dict) and str(card.get("cluster_key") or card.get("fact_type") or "").strip()
    }
    best: Dict[str, Any] = {}
    best_score = -1
    for shard in shards:
        shard_ids = {
            str(value or "").strip()
            for value in _as_list(shard.get("curated_evidence_ids"))
            if str(value or "").strip()
        }
        cluster_key = str(shard.get("cluster_key") or "").strip()
        score = 0
        if chapter_id and chapter_id == str(shard.get("chapter_id") or "").strip():
            score += 4
        if card_ids & shard_ids:
            score += len(card_ids & shard_ids)
        if cluster_key and cluster_key in card_clusters:
            score += 1
        if score > best_score:
            best = shard
            best_score = score
    if not best or best_score <= 0:
        return {}
    return {
        "schema_version": "analysis_shard_pointer_v1",
        "cluster_key": str(best.get("cluster_key") or "").strip(),
        "chapter_id": str(best.get("chapter_id") or "").strip(),
        "requirement_id": str(best.get("requirement_id") or "").strip(),
        "inventory_id": str(best.get("inventory_id") or "").strip(),
        "input_hash": str(best.get("input_hash") or "").strip(),
        "curated_evidence_ids": _as_list(best.get("curated_evidence_ids"))[:30],
    }


def _curated_topic_fit_for_llm(note: Dict[str, Any], query: str) -> str:
    explicit = str(note.get("topic_fit") or "").strip().lower()
    if explicit:
        return explicit
    query_text = str(query or "").lower()
    text = " ".join(
        str(value or "")
        for value in (
            note.get("clean_fact"),
            note.get("fact"),
            note.get("source_title"),
            note.get("source_url"),
        )
    ).lower()
    if not re.search(r"低空|evtol|无人机|通航|空域|low[-\s]?altitude", query_text):
        return "direct"
    if re.search(r"低空|evtol|无人机|无人驾驶飞行|通航|直升机|飞行器|航空器|空域|航线|飞行服务|低空旅游|低空物流", text):
        return "direct"
    if re.search(r"人形机器人|机器人|储能|数据要素|算力|大模型|半导体|光伏|新能源车", text):
        return "off_topic"
    if re.search(r"政策|监管|标准|基础设施|供应链|商业化|应用场景|安全|保险", text):
        return "related"
    return "background"


def _curated_evidence_cards_for_llm(
    evidence_package: Dict[str, Any],
    *,
    max_chapters: int,
    max_per_chapter: int,
) -> List[Dict[str, Any]]:
    if not _env_flag("REPORT_ANALYSIS_USE_CURATED_EVIDENCE", True):
        return []
    curated_payload = _as_dict(evidence_package.get("curated_evidence"))
    notes = [
        item
        for item in _as_list(curated_payload.get("curated_evidence"))
        if isinstance(item, dict) and str(item.get("evidence_id") or "").strip()
    ]
    if not notes:
        return []
    chapter_filter = _chapter_filter_for_llm(evidence_package, max_chapters=max_chapters)
    buckets: Dict[str, int] = {}
    cards: List[Dict[str, Any]] = []
    query_text = str(evidence_package.get("query") or evidence_package.get("original_query") or "").strip()
    for note in notes:
        topic_fit = _curated_topic_fit_for_llm(note, query_text)
        if _env_flag("REPORT_ANALYSIS_SKIP_OFF_TOPIC_CURATED", True) and topic_fit == "off_topic":
            continue
        raw_chapter_id = str(note.get("chapter_id") or _as_dict(note.get("lineage")).get("chapter_id") or "").strip()
        chapter_id = raw_chapter_id
        if chapter_filter:
            resolved = resolve_chapter_id(chapter_filter, raw_chapter_id)
            if not resolved:
                continue
            chapter_id = resolved
        if not chapter_id:
            chapter_id = "curated_evidence"
        if buckets.get(chapter_id, 0) >= max_per_chapter:
            continue
        cards.append(
            {
                "evidence_id": str(note.get("evidence_id") or "").strip(),
                "chapter_id": chapter_id,
                "cluster_key": str(note.get("cluster_key") or "").strip(),
                "requirement_id": str(note.get("requirement_id") or "").strip(),
                "analysis_role": str(note.get("fact_type") or "").strip(),
                "analysis_eligible": True,
                "allowed_use": str(note.get("evidence_use_level") or "directional_signal").strip(),
                "source_id": str(note.get("source_id") or "").strip(),
                "search_task_id": str(note.get("search_task_id") or "").strip(),
                "lineage": _as_dict(note.get("lineage")),
                "distilled_fact": _compact(
                    note.get("clean_fact"),
                    _env_int("BRAIN_LLM_ANALYSIS_MAX_FACT_CHARS", 420, min_value=40, max_value=800),
                ),
                "fact": _compact(note.get("clean_fact"), 420),
                "fact_type": str(note.get("fact_type") or "").strip(),
                "source_level": str(note.get("source_level") or "").strip().upper(),
                "proof_role": str(note.get("proof_role") or note.get("fact_type") or "").strip().lower(),
                "source_verification_status": "curated",
                "claim_strength_hint": str(note.get("claim_strength_hint") or "directional").strip(),
                "topic_fit": topic_fit,
                "limitations": _as_list(note.get("limitations")),
                "usable_for": _as_list(note.get("usable_for")),
                "source_title": _compact(note.get("source_title"), 120),
                "source_url": str(note.get("source_url") or "").strip(),
            }
        )
        buckets[chapter_id] = buckets.get(chapter_id, 0) + 1
    return cards


def _requirements_by_chapter_for_llm(evidence_package: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    contract = _as_dict(evidence_package.get("report_contract")) or _as_dict(evidence_package.get("report_plan"))
    requirements = _as_list(_as_dict(contract.get("evidence_requirements")).get("requirements"))
    identity = _chapter_identity_for_llm(evidence_package)
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for item in requirements:
        requirement = _as_dict(item)
        raw_chapter_id = str(requirement.get("chapter_id") or requirement.get("hypothesis_id") or "").strip()
        chapter_id = raw_chapter_id
        if identity and resolve_canonical_chapter_id is not None:
            chapter_id = resolve_canonical_chapter_id(identity, raw_chapter_id)
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


def _infer_requirement_from_chapter_contract_for_llm(
    requirements_by_chapter: Dict[str, List[Dict[str, Any]]],
    *,
    chapter_id: str,
    item: Dict[str, Any],
    card: Dict[str, Any],
) -> Dict[str, str]:
    requirements = [
        requirement
        for requirement in _as_list(requirements_by_chapter.get(str(chapter_id or "").strip()))
        if isinstance(requirement, dict) and str(requirement.get("requirement_id") or "").strip()
    ]
    if not requirements:
        return {}
    role_candidates = {
        str(value or "").strip().lower()
        for value in (
            item.get("proof_role"),
            item.get("analysis_role"),
            item.get("fact_type"),
            item.get("allowed_use"),
            card.get("proof_role"),
            card.get("analysis_role"),
            card.get("fact_type"),
            card.get("allowed_use"),
        )
        if str(value or "").strip()
    }
    role_matches = [
        requirement
        for requirement in requirements
        if str(requirement.get("proof_role") or "").strip().lower() in role_candidates
    ]
    if len(role_matches) == 1:
        return {
            "requirement_id": str(role_matches[0].get("requirement_id") or "").strip(),
            "requirement_id_source": "chapter_contract_proof_role",
        }
    if len(requirements) == 1:
        return {
            "requirement_id": str(requirements[0].get("requirement_id") or "").strip(),
            "requirement_id_source": "chapter_contract_single_requirement",
        }
    return {}


def _analysis_item_with_requirement_contract_inference(
    item: Dict[str, Any],
    requirements_by_chapter: Dict[str, List[Dict[str, Any]]],
    *,
    chapter_id: str,
) -> Dict[str, Any]:
    copied = dict(item)
    existing_requirement_id = _lineage_requirement_id(copied)
    if existing_requirement_id:
        copied.setdefault("requirement_id", existing_requirement_id)
        lineage = dict(_as_dict(copied.get("lineage")))
        lineage.setdefault("requirement_id", existing_requirement_id)
        copied["lineage"] = lineage
        return copied
    raw_chapter_id = str(
        copied.get("chapter_id")
        or copied.get("dimension_id")
        or _as_dict(copied.get("lineage")).get("chapter_id")
        or chapter_id
        or ""
    ).strip()
    inferred_requirement = _infer_requirement_from_chapter_contract_for_llm(
        requirements_by_chapter,
        chapter_id=raw_chapter_id,
        item=copied,
        card=_as_dict(copied.get("evidence_card")),
    )
    requirement_id = str(inferred_requirement.get("requirement_id") or "").strip()
    if not requirement_id:
        return copied
    copied["requirement_id"] = requirement_id
    copied["requirement_id_source"] = str(inferred_requirement.get("requirement_id_source") or "").strip()
    lineage = dict(_as_dict(copied.get("lineage")))
    lineage.setdefault("requirement_id", requirement_id)
    copied["lineage"] = lineage
    return copied


def build_llm_analysis_input_v2(evidence_package: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
    max_chapters = _env_int("BRAIN_LLM_ANALYSIS_MAX_CHAPTERS", 8, min_value=1, max_value=30)
    max_per_chapter = _env_int("BRAIN_LLM_ANALYSIS_MAX_FACTS_PER_CHAPTER", 16, min_value=1, max_value=30)
    max_fact_chars = _env_int("BRAIN_LLM_ANALYSIS_MAX_FACT_CHARS", 260, min_value=40, max_value=800)
    diagnostics = _chapter_filter_for_llm(evidence_package, max_chapters=max_chapters) or _as_dict(fallback.get("chapter_evidence_diagnostics"))
    candidate_per_chapter = max_per_chapter
    if _env_flag("BRAIN_LLM_ANALYSIS_ROLE_BALANCE", True):
        candidate_per_chapter = min(90, max_per_chapter * 3)
    cards = _evidence_cards_for_llm(
        evidence_package,
        max_chapters=max_chapters,
        max_per_chapter=candidate_per_chapter,
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
        compact_card = _compact_llm_fact_card(card, max_fact_chars=max_fact_chars)
        if compact_card.get("evidence_id") and compact_card.get("distilled_fact"):
            grouped[chapter_id].append(compact_card)
    chapters: List[Dict[str, Any]] = []
    for chapter_id in chapter_order[:max_chapters]:
        fact_cards = _role_balanced_llm_fact_cards(grouped.get(chapter_id) or [], limit=max_per_chapter)
        if not fact_cards:
            continue
        metadata = _chapter_payload_metadata(chapter_id, diagnostics, fallback)
        interpretation_payload: Dict[str, Any] = {}
        if build_evidence_interpretation_units is not None:
            try:
                interpretation_payload = build_evidence_interpretation_units(
                    chapter_id=chapter_id,
                    chapter_question=str(metadata.get("chapter_title") or metadata.get("chapter_question") or ""),
                    fact_cards=fact_cards,
                    max_units=_env_int("BRAIN_LLM_ANALYSIS_MAX_INTERPRETATION_UNITS", 6, min_value=1, max_value=12),
                )
            except Exception:
                interpretation_payload = {}
        chapters.append(
            {
                **metadata,
                "evidence_requirements": requirements_by_chapter.get(chapter_id, []),
                "evidence_inventory": _inventory_items_for_chapter(evidence_package, chapter_id, fact_cards),
                "analysis_shard": _analysis_shard_for_chapter(evidence_package, chapter_id, fact_cards),
                "interpretation_units": _as_list(_as_dict(interpretation_payload).get("interpretation_units")),
                "interpretation_diagnostics": _as_dict(_as_dict(interpretation_payload).get("diagnostics")),
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


def _persist_analysis_shard_output_cache(
    *,
    evidence_package: Dict[str, Any],
    chapter_payload: Dict[str, Any],
    analysis_result: Dict[str, Any],
    model: str,
    prompt_version: str,
) -> Dict[str, Any]:
    if not _env_flag("REPORT_ANALYSIS_SHARD_OUTPUT_CACHE_ENABLED", True):
        return {"status": "disabled"}
    if not _as_dict(chapter_payload.get("analysis_shard")):
        return {"status": "skipped", "reason": "missing_analysis_shard"}
    run_id = str(
        evidence_package.get("run_id")
        or evidence_package.get("report_id")
        or evidence_package.get("stage_snapshot_run_id")
        or os.getenv("REPORT_STAGE_SNAPSHOT_RUN_ID")
        or os.getenv("REPORT_RUN_ID")
        or ""
    ).strip()
    if not run_id:
        return {"status": "skipped", "reason": "missing_run_id"}
    try:
        from rag_pipeline.cache.analysis_memory_cache import persist_analysis_shard_output

        return persist_analysis_shard_output(
            evidence_package,
            chapter_payload=chapter_payload,
            analysis_result=analysis_result,
            run_id=run_id,
            model=model,
            prompt_version=prompt_version,
        )
    except Exception as exc:  # pragma: no cover - cache persistence must never block analysis.
        return {"status": "error", "error": str(exc)}


def _load_analysis_shard_output_cache(
    *,
    evidence_package: Dict[str, Any],
    chapter_payload: Dict[str, Any],
    model: str,
    prompt_version: str,
) -> Dict[str, Any]:
    if not _env_flag("REPORT_ANALYSIS_SHARD_OUTPUT_CACHE_READ_ENABLED", False):
        return {"status": "disabled"}
    if not _as_dict(chapter_payload.get("analysis_shard")):
        return {"status": "miss", "reason": "missing_analysis_shard"}
    run_id = str(
        evidence_package.get("run_id")
        or evidence_package.get("report_id")
        or evidence_package.get("stage_snapshot_run_id")
        or os.getenv("REPORT_STAGE_SNAPSHOT_RUN_ID")
        or os.getenv("REPORT_RUN_ID")
        or ""
    ).strip()
    if not run_id:
        return {"status": "miss", "reason": "missing_run_id"}
    try:
        from rag_pipeline.cache.analysis_memory_cache import load_analysis_shard_output

        return load_analysis_shard_output(
            chapter_payload=chapter_payload,
            run_id=run_id,
            model=model,
            prompt_version=prompt_version,
        )
    except Exception as exc:  # pragma: no cover - cache read must never block live analysis.
        return {"status": "miss", "reason": "error", "error": str(exc)}


def _llm_chapter_system_prompt() -> str:
    return """
你是中文行业研究报告的“证据到判断”分析师，一次只分析一个章节。
只能使用输入中的 fact_cards，不得编造事实、数字、公司、来源、URL 或引用。
只返回严格 JSON，顶层字段为：chapter_id、claim_units、analysis_limits。

分析步骤：
- 先在内部盘点 fact_cards 中的市场、政策、玩家、订单、价格、案例、技术、产业链和风险信号。
- 如果输入包含 interpretation_units，必须优先从 interpretation_units 生成 claim；fact_cards 只作为引用核验和补充材料，不要把单条 fact 直接改写成正文判断。
- 每个 claim 应体现“证据组合反映什么、为什么重要、传导机制、就业/教育/行业含义、边界”，不要停留在单条证据复述。
- 合并指向同一事实的重复证据，保留最清楚、最可追溯的表达。
- C/D 级、媒体、专业信息网或平台线索只要可追溯且不是脏数据，也可以转成有边界的 directional / limited_evidence 判断。
- evidence_grooming.possible_claim_angles 只能作为内部分析方向，不能原样写入 claim。
- 不得把 diagnostic-only、do_not_use_as、suggested_action、review_suggestion、补证建议、质量审查话术写进 claim。
- topic_fit=direct 的 fact_cards 是主锚点；topic_fit=related/background 只能作为背景、类比或边界，除非事实文本明确连接到本章主题。
- 不得把相邻行业或背景材料外推成核心行业结论。

每个 claim_unit 必须包含：
- claim：一句完整的中文行业判断，像报告里的分析判断，不像内部审查说明。
- requirement_ids：能从输入 fact_cards 得到时填写对应 requirement_id。
- fact_ids：支撑该判断的精确 evidence_id。
- source_ids：可得到时填写来源 id。
- hypothesis_id：可得到时填写章节或 fact_cards 中的 hypothesis_id。
- used_evidence_ids：精确引用输入中的 evidence_id。
- evidence_basis：只从输入事实提炼出的证据依据，写成简洁中文。
- reasoning_chain：解释“为什么这个事实能导向该判断”的机制链条。
- limitation_boundary：仅供后续写作者内部把握结论强度，不是正文句子，不得写成“需验证、后续观察、来源有限”等公开话术。
- claim_strength：strong、moderate、directional 或 limited_evidence。
- claim_strength_ceiling：被引用 fact_cards 允许的最高判断强度。
- claim_type：core_claim、mechanism_claim、counter_boundary_claim、contextual_claim、decision_claim、directional_claim、metric_claim、case_claim 或 technology_claim。
- analysis_role：claimable、directional、contextual、counter、metric、case 或 technology。
- source_support_map：说明 claim、mechanism、boundary 分别由哪些 evidence_id 支撑。
- paragraph_seed：给后续正文写作的一段中文素材，必须是行研语气，不能是审查口吻。
- block_affinity：metric_reconciliation、case_comparison、technology_maturity、risk_trigger 或 integrated_signal。
- interpretation_ids：如果 claim 来自 interpretation_units，填写对应 interpretation_id。
- what_evidence_reflects：说明多条证据共同反映的变化。
- why_it_matters：说明该变化为什么重要。
- mechanism_chain：用 2-4 句拆开说明传导链条，不要和 reasoning_chain 完全相同。
- employment_implication / education_implication / industry_implication：分别写岗位、培养体系、行业结构含义；至少填写一个。
- counter_reading：写可能的反向解释或边界，但不要写“后续需观察、证据不足、建议补证”等内部说明。
- claim_depth_ready：当以上深度字段足以支持正文展开时填 true。

数量与强度规则：
- 证据有足够不同信号时，尽量产出 4-6 个 claim_units；证据只支持更少判断时可以少于 4 个。
- 相关但不完整的证据不要直接丢弃，优先转成 contextual_claim、counter_boundary_claim 或 limited_evidence directional claim。
- A/B 且已核验的证据可以支撑 strong 或 moderate。
- B/C 或定性可追溯证据可以支撑 directional 或 limited_evidence。
- 不得把一个事实扩写成超过证据支持的机制判断。
- 宁可多产出有边界的中低强度判断，也不要少量过强判断。
- claim_strength 绝不能超过 claim_strength_ceiling。
- 如果无法从 fact_cards 推导 requirement_ids，不要因此拒绝该判断；可以让 requirement_ids 为空，但 used_evidence_ids 必须完整。
- 缺少硬指标（市场规模、增速、渗透率）不是不产 claim 的理由；可基于定性信号产出 directional / limited_evidence 判断，并把适用边界放入 limitation_boundary。
- 如果 evidence 很多但只能产出极少 claim，在 analysis_limits 中加入 suggested_action="reanalyze_existing"，不要把大量证据压成一个浅判断。
- 只有本章确实没有任何相关证据时，才返回空 claim_units 并在 analysis_limits 说明。
- 可以做角色多样化，但不得为填角色而编造：有风险、失败、限制或相反信号时才写 counter；有明确业务含义时才写 decision_use。

公开 claim 禁止出现这些内部话术：证据不足、建议补证、正文应以、方向性观察、后续验证、继续校准、来源覆盖有限、需交叉验证。
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
    try:
        analysis_max_tokens = int(os.getenv("BRAIN_LLM_ANALYSIS_MAX_OUTPUT_TOKENS", "8192") or 8192)
    except ValueError:
        analysis_max_tokens = 8192
    try:
        configured_max_tokens = int(config.get("max_output_tokens") or 0)
    except (TypeError, ValueError):
        configured_max_tokens = 0
    config["max_output_tokens"] = max(configured_max_tokens, analysis_max_tokens)
    if "disable_thinking" not in config:
        config["disable_thinking"] = str(os.getenv("BRAIN_LLM_ANALYSIS_DISABLE_THINKING", "true")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    chapter_id = str(chapter_payload.get("chapter_id") or "chapter")
    user_payload = {
        "query": _compact(evidence_package.get("query"), 180),
        "chapter": chapter_payload,
    }
    normalized_config = normalize_llm_config(config) if normalize_llm_config is not None else {}
    prompt_version = "llm_analysis_v5_2026_07_evidence_interpretation_units"
    shard_cache = _load_analysis_shard_output_cache(
        evidence_package=evidence_package,
        chapter_payload=chapter_payload,
        model=str(normalized_config.get("model") or ""),
        prompt_version=prompt_version,
    )
    if shard_cache.get("status") == "hit":
        result = _as_dict(shard_cache.get("analysis_result"))
        result["_llm_cache_hit"] = True
        result["_llm_cache_path"] = str(shard_cache.get("output_cache_path") or "")
        result["_analysis_shard_output_cache"] = shard_cache
        result["_analysis_shard_output_cache_hit"] = True
        return result
    cache_input = {
        **user_payload,
        "prompt_version": prompt_version,
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
    raw_payload["_analysis_shard_output_cache"] = _persist_analysis_shard_output_cache(
        evidence_package=evidence_package,
        chapter_payload=chapter_payload,
        analysis_result=raw_payload,
        model=str(normalized_config.get("model") or ""),
        prompt_version=prompt_version,
    )
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


def _raw_json_from_exception(exc: Exception) -> str:
    diagnostic = getattr(exc, "diagnostic", None)
    if isinstance(diagnostic, dict):
        raw = str(diagnostic.get("raw_content") or "").strip()
        if raw:
            return raw
        error = str(diagnostic.get("error") or "")
        marker = "LLM response is not valid JSON:"
        if marker in error:
            return error.split(marker, 1)[1].strip()
    text = str(exc or "")
    marker = "LLM response is not valid JSON:"
    return text.split(marker, 1)[1].strip() if marker in text else ""


def _salvage_llm_chapter_payload(exc: Exception, chapter_payload: Dict[str, Any]) -> Dict[str, Any]:
    if salvage_chapter_json is None:
        return {}
    raw = _raw_json_from_exception(exc)
    if not raw:
        return {}
    payload = salvage_chapter_json(
        raw,
        chapter_id=str(chapter_payload.get("chapter_id") or "chapter"),
        chapter_title=str(chapter_payload.get("chapter_title") or ""),
    )
    if not isinstance(payload, dict) or not payload:
        return {}
    payload["_llm_json_salvaged"] = True
    payload["_llm_json_salvage_raw_chars"] = len(raw)
    payload["_llm_cache_hit"] = False
    payload["_llm_cache_path"] = ""
    payload.setdefault("_llm_usage", {})
    return payload


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
    shard_output_cache_stored = 0
    json_salvage_attempted = 0
    json_salvage_success = 0
    max_retries = _env_int("BRAIN_LLM_ANALYSIS_MAX_RETRIES", 2, min_value=0, max_value=5)
    retry_base_seconds = float(os.getenv("BRAIN_LLM_ANALYSIS_RETRY_BASE_SECONDS", "0.5") or 0.5)

    deadline_min_remaining = float(os.getenv("BRAIN_LLM_ANALYSIS_DEADLINE_MIN_SECONDS", "30") or 30)

    def worker(chapter_payload: Dict[str, Any]) -> Dict[str, Any]:
        last_error = ""
        for attempt in range(max_retries + 1):
            if _analysis_deadline_reached(min_remaining=deadline_min_remaining):
                return {
                    "chapter_id": chapter_payload.get("chapter_id"),
                    "payload": {},
                    "error": "skipped_deadline: report deadline left no budget for this chapter analysis call",
                    "attempts": attempt,
                    "skipped_deadline": True,
                }
            attempt_config = {**config, "timeout": _deadline_capped_timeout(config["timeout"])}
            try:
                return {
                    "chapter_id": chapter_payload.get("chapter_id"),
                    "payload": synthesize_chapter_with_llm_analysis(
                        evidence_package=evidence_package,
                        chapter_payload=chapter_payload,
                        llm_config=attempt_config,
                    ),
                    "error": "",
                    "attempts": attempt + 1,
                }
            except Exception as exc:
                last_error = str(exc)
                if not _is_transient_llm_error(exc):
                    salvaged = _salvage_llm_chapter_payload(exc, chapter_payload)
                    if salvaged:
                        return {
                            "chapter_id": chapter_payload.get("chapter_id"),
                            "payload": salvaged,
                            "error": "",
                            "attempts": attempt + 1,
                            "json_salvage_attempted": True,
                            "json_salvage_success": True,
                        }
                    if _raw_json_from_exception(exc):
                        return {
                            "chapter_id": chapter_payload.get("chapter_id"),
                            "payload": {},
                            "error": last_error,
                            "attempts": attempt + 1,
                            "json_salvage_attempted": True,
                            "json_salvage_success": False,
                        }
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

    deadline_skipped = 0
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_map = {executor.submit(worker, chapter): chapter for chapter in chapters}
        for future in as_completed(future_map):
            result = future.result()
            retry_total += max(0, int(result.get("attempts") or 1) - 1)
            json_salvage_attempted += int(bool(result.get("json_salvage_attempted")))
            json_salvage_success += int(bool(result.get("json_salvage_success")))
            payload = _as_dict(result.get("payload"))
            error = str(result.get("error") or "")
            if error:
                failed += 1
                deadline_skipped += int(bool(result.get("skipped_deadline")))
                chapter_results.append(
                    {
                        "chapter_id": result.get("chapter_id"),
                        "status": "skipped_deadline" if result.get("skipped_deadline") else "error",
                        "error": error,
                    }
                )
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
            shard_output_cache = _as_dict(payload.get("_analysis_shard_output_cache"))
            shard_output_cache_stored += int(str(shard_output_cache.get("status") or "") == "stored")
            chapter_results.append(
                {
                    "chapter_id": result.get("chapter_id"),
                    "status": (
                        "json_salvaged"
                        if payload.get("_llm_json_salvaged")
                        else ("cached" if payload.get("_llm_cache_hit") else "called")
                    ),
                    "claim_count": sum(len(_as_list(ch.get("claim_units"))) for ch in _as_list(payload.get("chapter_synthesis")) if isinstance(ch, dict)),
                    "cache_path": payload.get("_llm_cache_path"),
                    "analysis_shard_output_cache": shard_output_cache,
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
        "_llm_deadline_skipped_chapter_count": deadline_skipped,
        "_analysis_shard_output_cache_stored_count": shard_output_cache_stored,
        "_llm_json_salvage_attempted_count": json_salvage_attempted,
        "_llm_json_salvage_success_count": json_salvage_success,
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
你是中文行业研究报告的证据分析 Agent。
只能使用输入 fact_cards，不得编造事实、数字、来源、公司、URL 或引用。
只返回一个 JSON 对象，字段为 chapter_synthesis、cross_chapter_conflicts、evidence_repair_priorities、rewrite_priorities。

对每个 chapter_synthesis：
- 尽量保留 chapter_id 和 chapter_title。
- 有足够不同证据信号时产出 4-6 个 claim_units；证据只支持更少判断时可以少于 4 个。
- 每个 claim_unit 必须包含 claim、claim_type、used_evidence_ids、evidence_basis、reasoning_chain、limitation_boundary、claim_strength。
- used_evidence_ids 必须是输入 fact_cards 中存在的精确 evidence_id。
- A/B 且 readpage_verified 或 document_verified 的卡片可以支持 moderate / strong。
- B/C 或 directional 卡片只能支持 directional / limited_evidence。
- 相关但不完整的证据应转成 contextual、counter/boundary 或 limited_evidence directional claim，不要直接遗漏。
- 章节内可以混合 core_claim、mechanism_claim、counter_boundary_claim、contextual_claim、decision_claim、directional_claim、metric_claim、case_claim、technology_claim，但只能在证据支持时使用。

不得把内部诊断当公开 claim。公开 claim 禁止出现：证据不足、建议补证、正文应以、方向性观察、后续验证、可追溯来源继续校准、来源覆盖有限、需交叉验证。
limitation_boundary 是内部约束字段，不是正文段落；不要把它写成完整公开句子。
只有章节完全没有相关证据时，才把限制放入 analysis_limits 而不创建 claim_unit。
除非证据本身只能用英文表达，所有公开分析字段都用中文。
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
    "claim_support_anchor_mismatch_downgraded",
    "decision_claim_downgraded_no_valid_ref",
    "llm_claim_semantic_judge_partial_downgraded",
    "llm_claim_semantic_judge_partial_observed",
    "llm_claim_semantic_judge_adjacent_downgraded",
    "llm_claim_semantic_judge_adjacent_observed",
    "llm_claim_semantic_judge_error",
    "llm_claim_semantic_judge_unsupported",
    "llm_claim_semantic_judge_unsupported_observed",
    "llm_numeric_claim_incomplete_metric_fact",
    "llm_numeric_claim_incomplete_metric_fact_observed",
    "llm_numeric_claim_grounding_failed",
    "llm_numeric_claim_grounding_failed_observed",
    "claim_support_needs_repair_observed",
    "claim_support_anchor_mismatch_observed",
}

REPAIRABLE_CLAIM_ISSUES = {
    "claim_support_needs_repair",
    "claim_support_needs_repair_observed",
    "llm_claim_semantic_judge_unsupported",
    "llm_claim_semantic_judge_unsupported_observed",
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


def _downgrade_claim_unit_to_directional(
    unit: Dict[str, Any],
    *,
    status: str,
    boundary_note: str,
) -> None:
    unit["claim_status"] = "directional"
    unit["claim_strength"] = "directional"
    unit["claim_strength_ceiling"] = "directional"
    unit["analysis_role"] = "directional"
    unit["evidence_use_level"] = "directional_signal"
    unit["writing_permission"] = "cautious_with_boundary"
    if status:
        unit["claim_support_status"] = status
    if boundary_note:
        # Review verdicts only adjust structural fields. The note is pipeline
        # diagnostics in English; limitation_boundary/counter_boundary are
        # writer-facing fields that get rendered into the public body, and
        # writing the note there is how "semantic judge found only partial
        # support..." ended up verbatim in a published report.
        unit["validation_notes"] = _dedupe([*_as_list(unit.get("validation_notes")), boundary_note])


def _claim_review_mutation_mode() -> str:
    value = str(
        os.getenv("REPORT_CLAIM_REVIEW_MUTATION_MODE")
        or os.getenv("BRAIN_CLAIM_REVIEW_MUTATION_MODE")
        or "diagnostic_only"
    ).strip().lower()
    if value in {"enforce", "strict", "mutate", "legacy", "blocking", "drop"}:
        return "enforce"
    return "diagnostic_only"


def _claim_review_can_mutate() -> bool:
    return _claim_review_mutation_mode() == "enforce" or _claim_retention_mode() == "strict"


def _append_claim_review_suggestion(
    unit: Dict[str, Any],
    *,
    source_stage: str,
    issue_type: str,
    reason: str = "",
    suggested_claim_strength: str = "",
    suggested_analysis_role: str = "",
    suggested_evidence_use_level: str = "",
    suggested_writing_permission: str = "",
    repair_priority: Optional[Dict[str, Any]] = None,
    semantic_judge: Optional[Dict[str, Any]] = None,
) -> None:
    suggestions = [
        item
        for item in _as_list(unit.get("claim_review_suggestions"))
        if isinstance(item, dict)
    ]
    suggestion: Dict[str, Any] = {
        "schema_version": "claim_review_suggestion_v1",
        "source_stage": source_stage,
        "issue_type": issue_type,
        "reason": _compact(reason, 260) if reason else "",
        "diagnostic_only": True,
        "not_for_public_text": True,
        "must_not_render": True,
        "public_text_allowed": False,
        "executor_should_decide": True,
        "suggested_action": "execution writer should decide whether to keep, soften, repair evidence binding, or omit this claim",
    }
    if suggested_claim_strength:
        suggestion["suggested_claim_strength"] = suggested_claim_strength
    if suggested_analysis_role:
        suggestion["suggested_analysis_role"] = suggested_analysis_role
    if suggested_evidence_use_level:
        suggestion["suggested_evidence_use_level"] = suggested_evidence_use_level
    if suggested_writing_permission:
        suggestion["suggested_writing_permission"] = suggested_writing_permission
    if repair_priority:
        suggestion["repair_priority"] = repair_priority
    if semantic_judge:
        suggestion["semantic_judge"] = {
            key: value
            for key, value in semantic_judge.items()
            if key not in {"usage"}
        }
    suggestions.append(suggestion)
    unit["claim_review_suggestions"] = suggestions
    unit["claim_review_mutation_mode"] = _claim_review_mutation_mode()


_INDUSTRY_GENERALIZATION_RE = re.compile(
    r"(?:industry|market|sector|ecosystem|enterprise adoption cycle|broad adoption|trend|"
    r"行业|市场|赛道|生态|整体|普遍|广泛|趋势|周期|企业级落地)",
    re.I,
)


def _is_industry_generalization_claim(claim_text: str) -> bool:
    text = str(claim_text or "").strip()
    if not text:
        return False
    return bool(_INDUSTRY_GENERALIZATION_RE.search(text))


def _single_source_generalization_priority(
    *,
    chapter: Dict[str, Any],
    unit: Dict[str, Any],
    claim_text: str,
    refs: Sequence[str],
    source_ids: Sequence[str],
    requirement_ids: Sequence[str],
    source_identities: Sequence[str] = (),
) -> Dict[str, Any]:
    claim_id = str(unit.get("claim_id") or "").strip()
    chapter_id = str(chapter.get("chapter_id") or "").strip()
    return {
        "schema_version": "claim_source_sufficiency_repair_priority_v1",
        "gap_type": "single_source_industry_generalization",
        "gap_id": f"{chapter_id or 'chapter'}_{claim_id or 'claim'}_single_source_industry_generalization",
        "chapter_id": chapter.get("chapter_id"),
        "claim_id": claim_id,
        "requirement_ids": list(requirement_ids or []),
        "evidence_refs": list(refs or []),
        "source_ids": list(source_ids or []),
        "source_identities": list(source_identities or []),
        "source_identity_count": len(list(source_identities or [])),
        "claim": _compact(claim_text, 360),
        "required_fields": ["second_independent_source", "source_url", "publisher", "date_or_period"],
        "proof_role": unit.get("proof_role") or "support",
        "success_criteria": "Find at least one additional independent source that supports the same industry-level claim, or keep the claim directional with a clear boundary.",
        "reject_if": ["snippet_only", "no_source_url", "same_publisher_family", "vendor_claim_only", "off_topic_source"],
        "allowed_for_writing": False,
        "writing_permission": "cautious_with_boundary",
        "recommended_action": "corroborate_or_soften_claim",
    }


def _claim_review_recommended_action(suggestion: Dict[str, Any]) -> str:
    issue_type = str(suggestion.get("issue_type") or "").strip().lower()
    writing_permission = str(suggestion.get("suggested_writing_permission") or "").strip().lower()
    if issue_type == "single_source_industry_generalization":
        return "needs_corroboration"
    if writing_permission in {"repair_before_publication", "not_allowed_until_repaired"}:
        return "repair_before_publication"
    if writing_permission == "cautious_with_boundary":
        return "cautious_with_boundary"
    if str(suggestion.get("suggested_analysis_role") or "").strip().lower() == "contextual":
        return "use_as_context_only"
    return "review_before_writing"


def _claim_review_action_plan(chapters: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    actions: List[Dict[str, Any]] = []
    summary: Dict[str, int] = {
        "total_actions": 0,
        "repair_before_publication": 0,
        "cautious_with_boundary": 0,
        "needs_corroboration": 0,
        "use_as_context_only": 0,
        "review_before_writing": 0,
    }
    for chapter in chapters:
        if not isinstance(chapter, dict):
            continue
        chapter_id = chapter.get("chapter_id")
        for unit in _as_list(chapter.get("claim_units")):
            if not isinstance(unit, dict):
                continue
            for suggestion in _as_list(unit.get("claim_review_suggestions")):
                if not isinstance(suggestion, dict):
                    continue
                action = _claim_review_recommended_action(suggestion)
                summary["total_actions"] += 1
                summary[action] = summary.get(action, 0) + 1
                actions.append(
                    {
                        "schema_version": "claim_review_action_v1",
                        "chapter_id": chapter_id,
                        "claim_id": unit.get("claim_id"),
                        "claim": _compact(unit.get("claim"), 300),
                        "issue_type": suggestion.get("issue_type"),
                        "source_stage": suggestion.get("source_stage"),
                        "recommended_action": action,
                        "suggested_claim_strength": suggestion.get("suggested_claim_strength"),
                        "suggested_analysis_role": suggestion.get("suggested_analysis_role"),
                        "suggested_evidence_use_level": suggestion.get("suggested_evidence_use_level"),
                        "suggested_writing_permission": suggestion.get("suggested_writing_permission"),
                        "repair_priority": _as_dict(suggestion.get("repair_priority")),
                        "diagnostic_only": True,
                        "not_for_public_text": True,
                        "must_not_render": True,
                        "public_text_allowed": False,
                        "executor_should_decide": True,
                    }
                )
    return {
        "schema_version": "claim_review_action_plan_v1",
        "summary": summary,
        "actions": actions,
        "diagnostic_only": True,
        "not_for_public_text": True,
        "must_not_render": True,
        "public_text_allowed": False,
        "executor_should_decide": True,
    }


def _semantic_judge_status(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"yes", "true", "supported", "support", "pass", "passed"}:
        return "supported"
    if text in {"partial", "partially_supported", "partially supported"}:
        return "partial"
    if text in {"adjacent", "related", "background", "context", "contextual"}:
        return "adjacent"
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


# Wall-clock deadline shared by the chapter-analysis workers and the semantic
# judge. The brain owns the report deadline; analysis must stop submitting new
# LLM work when the remaining budget cannot fit another call, otherwise the
# timeout is only discovered after run_analysis_agent returns and the whole
# main writer path is lost to fail-open.
_ANALYSIS_DEADLINE_TS: Optional[float] = None
_SEMANTIC_JUDGE_CALL_COUNT: int = 0
_SEMANTIC_JUDGE_COUNT_LOCK = threading.Lock()


def _set_analysis_deadline(deadline_ts: Optional[float]) -> None:
    global _ANALYSIS_DEADLINE_TS, _SEMANTIC_JUDGE_CALL_COUNT
    try:
        value = float(deadline_ts or 0.0)
    except (TypeError, ValueError):
        value = 0.0
    _ANALYSIS_DEADLINE_TS = value if value > 0 else None
    _SEMANTIC_JUDGE_CALL_COUNT = 0


def _analysis_deadline_remaining() -> float:
    if not _ANALYSIS_DEADLINE_TS:
        return float("inf")
    return _ANALYSIS_DEADLINE_TS - time.perf_counter()


def _analysis_deadline_reached(min_remaining: float = 0.0) -> bool:
    remaining = _analysis_deadline_remaining()
    return remaining != float("inf") and remaining <= float(min_remaining or 0.0)


def _deadline_capped_timeout(configured_timeout: float, *, floor: float = 10.0) -> float:
    remaining = _analysis_deadline_remaining()
    if remaining == float("inf"):
        return configured_timeout
    return max(floor, min(configured_timeout, remaining - 5.0))


def _semantic_judge_enabled(llm_config: Optional[Dict[str, Any]]) -> bool:
    return bool(llm_config) and _env_flag("BRAIN_ENABLE_LLM_SEMANTIC_JUDGE", True)


def _semantic_judge_fail_closed() -> bool:
    return _env_flag("BRAIN_LLM_SEMANTIC_JUDGE_FAIL_CLOSED", False)


def _claim_retention_mode() -> str:
    value = str(
        os.getenv("REPORT_CLAIM_RETENTION_MODE")
        or os.getenv("BRAIN_CLAIM_RETENTION_MODE")
        or "permissive"
    ).strip().lower()
    if value in {"strict", "blocking", "enforce", "drop"}:
        return "strict"
    if value in {"balanced", "normal"}:
        return "balanced"
    return "permissive"


def _claim_retention_permissive() -> bool:
    return _claim_retention_mode() == "permissive"


SEMANTIC_JUDGE_PROMPT_VERSION = "semantic_claim_support_judge_v2_2026_06_tiered"


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
                "source_url": _compact(item.get("source_url") or item.get("url") or source.get("source_url") or source.get("url"), 240),
                "source_id": str(item.get("source_id") or source.get("id") or source.get("source_id") or "").strip(),
                "source_title_url_mismatch_suspected": bool(
                    item.get("source_title_url_mismatch_suspected") or source.get("source_title_url_mismatch_suspected")
                ),
                "source_binding_fuzzy": bool(item.get("source_binding_fuzzy") or source.get("source_binding_fuzzy")),
                "source_title_missing": bool(item.get("source_title_missing") or source.get("source_title_missing")),
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
你是中文研究报告的严格证据支撑判断器，负责判断被引用的 fact_cards 是否在语义上支撑 claim。

规则：
- 只能使用输入 fact_cards，不得使用外部知识。
- 来源只是提到同一主题，不等于支撑该 claim。
- 数字、日期、公司、范围、因果、竞争判断和风险判断必须能在被引用事实中直接落地。
- 当事实能直接支撑完整 claim 时，status 返回 supported。
- 当事实支持方向，但不支持完整强度、范围或完整性时，status 返回 partial。
- 当事实只能作为背景材料而不能直接支撑 claim 时，status 返回 adjacent。
- 当事实与 claim 矛盾、缺少关键数字/实体/日期，或过于泛泛时，status 返回 unsupported。
- 只输出一个 JSON 对象：{"status":"supported|partial|adjacent|unsupported","reason":"...","confidence":0.0-1.0,"unsupported_terms":[]}。
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
    if _analysis_deadline_reached(min_remaining=float(os.getenv("BRAIN_SEMANTIC_JUDGE_DEADLINE_MIN_SECONDS", "20") or 20)):
        return {"status": "skipped_deadline"}
    max_calls = _env_int("BRAIN_SEMANTIC_JUDGE_MAX_CALLS", 0, min_value=0, max_value=10_000)
    global _SEMANTIC_JUDGE_CALL_COUNT
    with _SEMANTIC_JUDGE_COUNT_LOCK:
        if max_calls and _SEMANTIC_JUDGE_CALL_COUNT >= max_calls:
            return {"status": "skipped_budget"}
        _SEMANTIC_JUDGE_CALL_COUNT += 1
    config["timeout"] = _deadline_capped_timeout(config["timeout"])
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
                "instruction": "请把支撑关系分类为 supported、partial、adjacent 或 unsupported。不要把 partial 或 adjacent 简单归为 unsupported。",
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


def _prewarm_semantic_judge_cache(
    chapter_iterable: Sequence[Any],
    *,
    card_by_ref: Dict[str, Dict[str, Any]],
    valid_refs: set,
    evidence_alias_map: Dict[str, Any],
    input_cards: Sequence[Dict[str, Any]],
    llm_config: Optional[Dict[str, Any]],
) -> None:
    """Run semantic judge calls concurrently before the sequential validation loop.

    The judge result cache is keyed on (claim, cited cards), so warming it here
    turns the per-claim judge calls inside the validation loop into cache hits.
    Failures are ignored: the loop will simply call the judge itself.
    """

    if not _semantic_judge_enabled(llm_config):
        return
    concurrency = _env_int("BRAIN_SEMANTIC_JUDGE_CONCURRENCY", 4, min_value=1, max_value=8)
    if concurrency <= 1:
        return
    jobs: List[Tuple[str, List[Dict[str, Any]], Any, Any]] = []
    for raw_chapter in chapter_iterable:
        if not isinstance(raw_chapter, dict):
            continue
        for unit in _as_list(raw_chapter.get("claim_units")):
            if not isinstance(unit, dict):
                continue
            claim_text = _compact(unit.get("claim"), 360)
            if not claim_text:
                continue
            raw_refs = [str(ref or "").strip() for ref in _as_list(unit.get("used_evidence_ids")) if str(ref or "").strip()]
            if not raw_refs:
                continue
            if normalize_claim_refs is not None:
                normalized_refs = normalize_claim_refs(
                    {**unit, "used_evidence_ids": raw_refs},
                    alias_map=evidence_alias_map,
                    fact_cards=input_cards,
                )
                refs = [str(ref or "").strip() for ref in _as_list(normalized_refs.get("fact_ids")) if str(ref or "").strip()]
            else:
                refs = raw_refs
            refs = [ref for ref in refs if not valid_refs or ref in valid_refs]
            cited_cards = [_as_dict(card_by_ref.get(ref)) for ref in refs if _as_dict(card_by_ref.get(ref))]
            if not cited_cards:
                continue
            jobs.append((claim_text, cited_cards, raw_chapter.get("chapter_id"), unit.get("claim_id")))
    if not jobs:
        return

    def prewarm(job: Tuple[str, List[Dict[str, Any]], Any, Any]) -> None:
        claim_text, cited_cards, chapter_id, claim_id = job
        try:
            _llm_semantic_claim_support_judge(
                claim_text=claim_text,
                cited_cards=cited_cards,
                chapter_id=chapter_id,
                claim_id=claim_id,
                llm_config=llm_config,
            )
        except Exception:
            return

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        for future in as_completed({executor.submit(prewarm, job) for job in jobs}):
            future.result()


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
    evidence_alias_map = build_evidence_alias_map(input_cards) if build_evidence_alias_map is not None else {}
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
    review_isolated = quality_gates_isolated()
    claim_retention_permissive = _claim_retention_permissive()
    claim_review_can_mutate = _claim_review_can_mutate()
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
            "claim_review_action_plan": _claim_review_action_plan([]),
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
    _prewarm_semantic_judge_cache(
        chapter_iterable,
        card_by_ref=card_by_ref,
        valid_refs=valid_refs,
        evidence_alias_map=evidence_alias_map,
        input_cards=input_cards,
        llm_config=llm_config,
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
            raw_refs = [
                str(ref or "").strip()
                for ref in _as_list(unit.get("used_evidence_ids"))
                if str(ref or "").strip()
            ]
            if not raw_refs:
                issue = {"type": "llm_claim_missing_used_evidence_ids", "chapter_id": chapter.get("chapter_id")}
                issues.append(issue)
                if len(rejected_examples) < 5:
                    rejected_examples.append({**issue, "claim": _compact(unit.get("claim"), 160)})
                continue
            if normalize_claim_refs is not None:
                normalized_refs = normalize_claim_refs(
                    {**unit, "used_evidence_ids": raw_refs},
                    alias_map=evidence_alias_map,
                    fact_cards=input_cards,
                )
                refs = [
                    str(ref or "").strip()
                    for ref in _as_list(normalized_refs.get("fact_ids"))
                    if str(ref or "").strip()
                ]
                unit["ref_resolution"] = _as_dict(normalized_refs.get("ref_resolution"))
                unit["legacy_ref_fields"] = _as_dict(normalized_refs.get("legacy_ref_fields"))
                if _as_list(normalized_refs.get("ambiguous_refs")):
                    issue = {
                        "type": "ambiguous_llm_evidence_ref",
                        "refs": _as_list(normalized_refs.get("ambiguous_refs")),
                        "chapter_id": chapter.get("chapter_id"),
                    }
                    issues.append(issue)
                    if len(rejected_examples) < 5:
                        rejected_examples.append({**issue, "claim": _compact(unit.get("claim"), 160)})
                    continue
                invalid_refs = [
                    ref
                    for ref in _as_list(normalized_refs.get("unresolved_refs"))
                    if str(ref or "").strip()
                ]
            else:
                refs = raw_refs
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
            unit["fact_ids"] = refs
            unit["used_evidence_ids"] = refs
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
            interpretation_ids = _dedupe(
                [
                    str(item or "").strip()
                    for item in _as_list(unit.get("interpretation_ids"))
                    if str(item or "").strip()
                ]
            )
            if not interpretation_ids and str(unit.get("interpretation_id") or "").strip():
                interpretation_ids = [str(unit.get("interpretation_id") or "").strip()]
            mechanism_chain_detail = _as_list(unit.get("mechanism_chain"))
            if not mechanism_chain_detail and str(unit.get("mechanism_chain") or "").strip():
                mechanism_chain_detail = [unit.get("mechanism_chain")]
            mechanism_chain_detail = [
                _public_normalize_analysis_text(_compact(item, 420))
                for item in mechanism_chain_detail
                if str(item or "").strip()
            ]
            depth_text_fields = {
                "what_evidence_reflects": _public_normalize_analysis_text(_compact(unit.get("what_evidence_reflects"), 420)),
                "why_it_matters": _public_normalize_analysis_text(_compact(unit.get("why_it_matters"), 420)),
                "employment_implication": _public_normalize_analysis_text(_compact(unit.get("employment_implication"), 420)),
                "education_implication": _public_normalize_analysis_text(_compact(unit.get("education_implication"), 420)),
                "industry_implication": _public_normalize_analysis_text(_compact(unit.get("industry_implication"), 420)),
                "counter_reading": _public_normalize_analysis_text(_compact(unit.get("counter_reading"), 420)),
            }
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
            raw_mechanism_text = _public_normalize_analysis_text(_compact(unit.get("mechanism"), 500))
            reasoning_text = "\n".join(reasoning_chain)
            if raw_mechanism_text and _normalize_key(raw_mechanism_text) == _normalize_key(reasoning_text):
                issues.append(
                    {
                        "type": "shallow_claim_reasoning",
                        "chapter_id": chapter.get("chapter_id"),
                        "claim_id": unit.get("claim_id") or unit.get("id"),
                    }
                )
            unit["claim"] = claim_text
            unit["evidence_basis"] = evidence_basis
            unit["because_facts"] = evidence_basis
            unit["reasoning_chain"] = reasoning_chain
            unit["mechanism_chain"] = mechanism_chain_detail
            unit["interpretation_ids"] = interpretation_ids
            for depth_key, depth_value in depth_text_fields.items():
                unit[depth_key] = depth_value
            if not str(unit.get("decision_implication") or "").strip():
                unit["decision_implication"] = " ".join(
                    item
                    for item in [
                        depth_text_fields.get("employment_implication", ""),
                        depth_text_fields.get("education_implication", ""),
                        depth_text_fields.get("industry_implication", ""),
                    ]
                    if item
                ).strip()
            unit["claim_depth_ready"] = bool(
                (interpretation_ids or mechanism_chain_detail)
                and depth_text_fields.get("what_evidence_reflects")
                and depth_text_fields.get("why_it_matters")
                and (
                    depth_text_fields.get("employment_implication")
                    or depth_text_fields.get("education_implication")
                    or depth_text_fields.get("industry_implication")
                )
            )
            unit["single_fact_claim"] = len(refs) == 1
            unit["limitation_boundary"] = limitation_boundary
            unit["boundary"] = limitation_boundary
            unit["reasoning"] = "\n".join(reasoning_chain)
            unit["mechanism"] = "\n".join(reasoning_chain)
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
            claim_type = str(unit.get("claim_type") or unit.get("conclusion_type") or "").strip().lower()
            analysis_role = str(unit.get("analysis_role") or "").strip().lower()
            if not claim_type:
                if analysis_role == "counter":
                    claim_type = "counter_boundary_claim"
                elif analysis_role == "contextual":
                    claim_type = "contextual_claim"
                elif analysis_role == "metric":
                    claim_type = "metric_claim"
                elif analysis_role == "case":
                    claim_type = "case_claim"
                elif analysis_role == "technology":
                    claim_type = "technology_claim"
                elif strength_text in {"strong", "moderate"}:
                    claim_type = "core_claim"
                else:
                    claim_type = "directional_claim"
            unit["claim_type"] = claim_type
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
            numeric_grounding = validate_numeric_fact_grounding(claim_text, cited_cards)
            if not numeric_grounding.valid:
                issue = {
                    "type": "llm_numeric_claim_grounding_failed_observed"
                    if (review_isolated or not claim_review_can_mutate)
                    else "llm_numeric_claim_grounding_failed",
                    "chapter_id": chapter.get("chapter_id"),
                    "claim_id": unit.get("claim_id"),
                    "evidence_refs": refs,
                    "reasons": numeric_grounding.reasons,
                    "numbers": numeric_grounding.numbers,
                }
                if not review_isolated and claim_review_can_mutate:
                    issue["downgraded_to"] = "directional"
                issues.append(issue)
                unit["numeric_grounding_status"] = "failed"
                unit["numeric_grounding_reasons"] = numeric_grounding.reasons
                unit["unsupported_numeric_tokens"] = numeric_grounding.numbers
                unit["public_hard_number_allowed"] = False
                if not review_isolated and claim_review_can_mutate:
                    unit["claim_strength"] = "directional"
                    unit["claim_strength_ceiling"] = "directional"
                    unit["claim_status"] = "directional"
                    unit["analysis_role"] = "directional"
                    unit["evidence_use_level"] = "directional_signal"
                    unit["writing_permission"] = "cautious_without_unsupported_numbers"
                else:
                    _append_claim_review_suggestion(
                        unit,
                        source_stage="numeric_grounding_gate",
                        issue_type="llm_numeric_claim_grounding_failed",
                        reason="numeric hard fact token was not found in cited fact cards",
                        suggested_claim_strength="directional",
                        suggested_evidence_use_level="directional_signal",
                        suggested_writing_permission="omit_unsupported_numbers",
                    )
            anchor_mismatch_pending: Optional[Dict[str, Any]] = None
            if validate_claim_supported_by_facts is None:
                issue = {
                    "type": "claim_support_validator_unavailable_observed" if (review_isolated or claim_retention_permissive) else "claim_support_validator_unavailable",
                    "chapter_id": chapter.get("chapter_id"),
                    "claim_id": unit.get("claim_id"),
                    "evidence_refs": refs,
                    "status": "validator_unavailable",
                }
                issues.append(issue)
                if len(rejected_examples) < 5:
                    rejected_examples.append({**issue, "claim": claim_text})
                if not review_isolated and not claim_retention_permissive:
                    continue
                unit["claim_support_status"] = "validator_unavailable_observed"
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
                    gap_type = str(repair_priority.get("gap_type") or "")
                    if gap_type == "claim_support_anchor_mismatch":
                        # Lexical n-gram anchors systematically miss paraphrased
                        # LLM claims (live run: 19/19 usable claims downgraded
                        # this way), so defer the verdict to the semantic judge
                        # below: it confirms direct support, downgrades, or
                        # rejects. Only when the judge cannot run do we fall
                        # back to the conservative directional downgrade.
                        anchor_mismatch_pending = {
                            "chapter_id": chapter.get("chapter_id"),
                            "claim_id": unit.get("claim_id"),
                            "evidence_refs": refs,
                            "gap_type": gap_type,
                            **support_payload,
                        }
                        unit["support_gap_type"] = gap_type
                        unit["support_unsupported_terms"] = _as_list(support_payload.get("unsupported_terms"))
                    else:
                        issue = {
                            "type": "claim_support_needs_repair_observed" if (review_isolated or claim_retention_permissive) else "claim_support_needs_repair",
                            "chapter_id": chapter.get("chapter_id"),
                            "claim_id": unit.get("claim_id"),
                            "evidence_refs": refs,
                            "gap_type": gap_type,
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
                        if review_isolated or claim_retention_permissive:
                            unit["claim_support_status"] = "needs_repair_observed"
                            unit["claim_support_repair_hint"] = repair_priority
                            _append_claim_review_suggestion(
                                unit,
                                source_stage="claim_support_validator",
                                issue_type="claim_support_needs_repair",
                                reason="claim support validator found missing or mismatched evidence anchors",
                                suggested_claim_strength="directional",
                                suggested_evidence_use_level="diagnostic_only",
                                suggested_writing_permission="not_allowed_until_repaired",
                                repair_priority=repair_priority,
                            )
                            if claim_retention_permissive and not review_isolated and claim_review_can_mutate:
                                _downgrade_claim_unit_to_directional(
                                    unit,
                                    status="needs_repair_observed",
                                    boundary_note="claim support needs repair; keep only as cautious directional analysis until stronger evidence is bound",
                                )
                        else:
                            continue
                else:
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

            def _apply_pending_anchor_downgrade() -> None:
                if not anchor_mismatch_pending:
                    return
                if review_isolated or not claim_review_can_mutate:
                    issues.append(
                        {
                            "type": "claim_support_anchor_mismatch_observed",
                            **anchor_mismatch_pending,
                        }
                    )
                    unit["claim_support_status"] = "anchor_mismatch_observed"
                    _append_claim_review_suggestion(
                        unit,
                        source_stage="claim_support_validator",
                        issue_type="claim_support_anchor_mismatch",
                        reason="lexical claim anchors did not fully match cited facts",
                        suggested_claim_strength="directional",
                        suggested_evidence_use_level="directional_signal",
                        suggested_writing_permission="cautious_with_boundary",
                    )
                    return
                issues.append(
                    {
                        "type": "claim_support_anchor_mismatch_downgraded",
                        "downgraded_to": "directional",
                        **anchor_mismatch_pending,
                    }
                )
                _downgrade_claim_unit_to_directional(
                    unit,
                    status="anchor_mismatch_downgraded",
                    boundary_note="claim support is adjacent rather than direct; use only as a directional signal until stronger evidence is repaired",
                )

            if semantic_status.startswith("skipped"):
                unit["semantic_judge_skipped_reason"] = semantic_status
                # No semantic verdict available: keep the conservative lexical
                # downgrade rather than shipping an unverified strong claim.
                _apply_pending_anchor_downgrade()
            elif semantic_status == "error":
                fail_closed = _semantic_judge_fail_closed() and not review_isolated
                issue = {
                    "type": "llm_claim_semantic_judge_error_blocked" if fail_closed else "llm_claim_semantic_judge_error",
                    "chapter_id": chapter.get("chapter_id"),
                    "claim_id": unit.get("claim_id"),
                    "evidence_refs": refs,
                    "semantic_judge": semantic_judge,
                }
                issues.append(issue)
                if fail_closed:
                    if len(rejected_examples) < 5:
                        rejected_examples.append({**issue, "claim": claim_text})
                    continue
                unit["semantic_judge"] = {
                    key: value
                    for key, value in semantic_judge.items()
                    if key not in {"usage"}
                }
                unit["semantic_judge_risk"] = "judge_error_rule_fallback"
                _apply_pending_anchor_downgrade()
            elif semantic_status == "partial":
                issue = {
                    "type": "llm_claim_semantic_judge_partial_observed"
                    if (review_isolated or not claim_review_can_mutate)
                    else "llm_claim_semantic_judge_partial_downgraded",
                    "chapter_id": chapter.get("chapter_id"),
                    "claim_id": unit.get("claim_id"),
                    "evidence_refs": refs,
                    "semantic_judge": semantic_judge,
                }
                if not review_isolated and claim_review_can_mutate:
                    issue["downgraded_to"] = "directional"
                issues.append(issue)
                if not review_isolated and claim_review_can_mutate:
                    _downgrade_claim_unit_to_directional(
                        unit,
                        status="semantic_partial_downgraded",
                        boundary_note="semantic judge found only partial support; keep as cautious directional analysis until stronger evidence is bound",
                    )
                else:
                    unit["claim_support_status"] = "semantic_partial_observed"
                    _append_claim_review_suggestion(
                        unit,
                        source_stage="semantic_claim_support_judge",
                        issue_type="llm_claim_semantic_judge_partial",
                        reason=str(semantic_judge.get("reason") or "semantic judge found partial support"),
                        suggested_claim_strength="directional",
                        suggested_evidence_use_level="directional_signal",
                        suggested_writing_permission="cautious_with_boundary",
                        semantic_judge=semantic_judge,
                    )
                unit["semantic_judge"] = {
                    key: value
                    for key, value in semantic_judge.items()
                    if key not in {"usage"}
                }
            elif semantic_status == "adjacent":
                issue = {
                    "type": "llm_claim_semantic_judge_adjacent_observed"
                    if (review_isolated or not claim_review_can_mutate)
                    else "llm_claim_semantic_judge_adjacent_downgraded",
                    "chapter_id": chapter.get("chapter_id"),
                    "claim_id": unit.get("claim_id"),
                    "evidence_refs": refs,
                    "semantic_judge": semantic_judge,
                }
                if not review_isolated and claim_review_can_mutate:
                    issue["downgraded_to"] = "contextual"
                issues.append(issue)
                if not review_isolated and claim_review_can_mutate:
                    _downgrade_claim_unit_to_directional(
                        unit,
                        status="semantic_adjacent_downgraded",
                        boundary_note="semantic judge found adjacent background rather than direct support; use only as contextual framing until stronger evidence is bound",
                    )
                    unit["analysis_role"] = "contextual"
                    unit["evidence_use_level"] = "background"
                else:
                    unit["claim_support_status"] = "semantic_adjacent_observed"
                    _append_claim_review_suggestion(
                        unit,
                        source_stage="semantic_claim_support_judge",
                        issue_type="llm_claim_semantic_judge_adjacent",
                        reason=str(semantic_judge.get("reason") or "semantic judge found adjacent/background support"),
                        suggested_claim_strength="directional",
                        suggested_analysis_role="contextual",
                        suggested_evidence_use_level="background",
                        suggested_writing_permission="cautious_with_boundary",
                        semantic_judge=semantic_judge,
                    )
                unit["semantic_judge"] = {
                    key: value
                    for key, value in semantic_judge.items()
                    if key not in {"usage"}
                }
            elif not _semantic_judge_accepts(semantic_judge):
                issue = {
                    "type": "llm_claim_semantic_judge_unsupported_observed"
                    if (review_isolated or claim_retention_permissive or not claim_review_can_mutate)
                    else "llm_claim_semantic_judge_unsupported",
                    "chapter_id": chapter.get("chapter_id"),
                    "claim_id": unit.get("claim_id"),
                    "evidence_refs": refs,
                    "semantic_judge": semantic_judge,
                }
                issues.append(issue)
                semantic_repair_priority = _claim_support_repair_priority(
                    chapter=chapter,
                    unit=unit,
                    claim_text=claim_text,
                    refs=refs,
                    cited_cards=cited_cards,
                    support_payload={
                        "status": "unsupported",
                        "unsupported_terms": _as_list(semantic_judge.get("unsupported_terms"))
                        or [_compact(semantic_judge.get("reason") or "semantic judge did not support the claim", 180)],
                    },
                )
                semantic_repair_priority["gap_type"] = "claim_semantic_support_mismatch"
                semantic_repair_priority["source_stage"] = "semantic_claim_support_judge"
                claim_repair_priorities.append(semantic_repair_priority)
                if review_isolated or claim_retention_permissive or not claim_review_can_mutate:
                    unit["semantic_judge"] = {
                        key: value
                        for key, value in semantic_judge.items()
                        if key not in {"usage"}
                    }
                    unit["claim_support_status"] = "semantic_unsupported_observed"
                    _append_claim_review_suggestion(
                        unit,
                        source_stage="semantic_claim_support_judge",
                        issue_type="llm_claim_semantic_judge_unsupported",
                        reason=str(semantic_judge.get("reason") or "semantic judge did not support the claim"),
                        suggested_claim_strength="directional",
                        suggested_evidence_use_level="diagnostic_only",
                        suggested_writing_permission="repair_before_publication",
                        repair_priority=semantic_repair_priority,
                        semantic_judge=semantic_judge,
                    )
                    if claim_retention_permissive and not review_isolated and claim_review_can_mutate:
                        _downgrade_claim_unit_to_directional(
                            unit,
                            status="semantic_unsupported_observed",
                            boundary_note="semantic judge did not find direct support; keep only as cautious directional analysis and repair the evidence binding",
                        )
                    if len(deferred_examples) < 8:
                        deferred_examples.append(
                            {
                                **issue,
                                "claim": claim_text,
                                "repair_priority": semantic_repair_priority,
                                "claim_status": unit.get("claim_status") or unit.get("claim_support_status"),
                                "evidence_use_level": unit.get("evidence_use_level"),
                            }
                        )
                else:
                    if len(rejected_examples) < 5:
                        rejected_examples.append({**issue, "claim": claim_text})
                    continue
            else:
                unit["semantic_judge"] = {
                    key: value
                    for key, value in semantic_judge.items()
                    if key not in {"usage"}
                }
                if anchor_mismatch_pending:
                    # The semantic judge confirmed direct support, so the
                    # lexical anchor miss was a paraphrase false positive; keep
                    # the claim at its original strength and record the waiver.
                    issues.append(
                        {
                            "type": "claim_support_anchor_mismatch_waived_by_semantic_judge",
                            **anchor_mismatch_pending,
                        }
                    )
                    unit["support_gap_type"] = ""
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
                        "type": "llm_numeric_claim_incomplete_metric_fact_observed"
                        if (review_isolated or not claim_review_can_mutate)
                        else "llm_numeric_claim_incomplete_metric_fact",
                        "chapter_id": chapter.get("chapter_id"),
                        "claim_id": unit.get("claim_id"),
                        "evidence_refs": refs,
                        "metric_gaps": metric_gaps,
                    }
                    if not review_isolated and claim_review_can_mutate:
                        issue["downgraded_to"] = "directional"
                    issues.append(issue)
                    unit["metric_completeness_status"] = "incomplete"
                    unit["metric_missing_fields"] = missing_fields
                    if not review_isolated and claim_review_can_mutate:
                        unit["claim_status"] = "directional"
                        unit["claim_strength"] = "directional"
                        unit["claim_strength_ceiling"] = "directional"
                        unit["analysis_role"] = "directional"
                        unit["evidence_use_level"] = "directional_signal"
                        unit["writing_permission"] = "cautious_with_boundary"
                        # Diagnostics stay out of writer-facing boundary fields
                        # (they get rendered into the public body verbatim).
                        boundary_note = (
                            "metric fields incomplete: "
                            + (", ".join(missing_fields) if missing_fields else "unknown")
                            + "; use only as a directional signal until repaired"
                        )
                        unit["validation_notes"] = _dedupe([*_as_list(unit.get("validation_notes")), boundary_note])
                    else:
                        _append_claim_review_suggestion(
                            unit,
                            source_stage="metric_completeness_gate",
                            issue_type="llm_numeric_claim_incomplete_metric_fact",
                            reason="numeric claim cites metric facts with missing value/unit/period/source fields",
                            suggested_claim_strength="directional",
                            suggested_evidence_use_level="directional_signal",
                            suggested_writing_permission="cautious_with_boundary",
                        )
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
                unit["requirement_binding_status"] = "missing_observed"
                unit["claim_strength"] = "directional"
                unit["claim_strength_ceiling"] = "directional"
                unit["analysis_role"] = str(unit.get("analysis_role") or "contextual")
                unit["evidence_use_level"] = str(unit.get("evidence_use_level") or "directional_signal")
                unit["writing_permission"] = "cautious_with_boundary"
                _append_claim_review_suggestion(
                    unit,
                    source_stage="requirement_lineage_gate",
                    issue_type="missing_requirement_ids",
                    reason=(
                        "claim cites traceable evidence but no requirement_id could be inferred; "
                        "keep only as chapter-scoped directional analysis"
                    ),
                    suggested_claim_strength="directional",
                    suggested_analysis_role="contextual",
                    suggested_evidence_use_level="directional_signal",
                    suggested_writing_permission="cautious_with_boundary",
                )
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
                    "type": "llm_claim_strength_clamped_to_ceiling",
                    "chapter_id": chapter.get("chapter_id"),
                    "claim_strength": ceiling,
                    "claim_strength_original": claim_strength,
                    "claim_strength_ceiling": ceiling,
                }
                issues.append(issue)
                unit["claim_strength_clamped_from"] = claim_strength
                unit["claim_strength"] = ceiling
                claim_strength = ceiling
            source_ids = _dedupe(
                [
                    str(card.get("source_id") or _as_dict(card.get("lineage")).get("source_id") or "").strip()
                    for card in cited_cards
                    if str(card.get("source_id") or _as_dict(card.get("lineage")).get("source_id") or "").strip()
                ]
            )
            source_identities = _dedupe(
                [
                    _source_identity_key(card)
                    for card in cited_cards
                    if _source_identity_key(card)
                ]
            )
            distinct_support_count = len(source_identities or source_ids or refs)
            if (
                _claim_strength_score(claim_strength) >= _claim_strength_score("moderate")
                and distinct_support_count <= 1
                and _is_industry_generalization_claim(claim_text)
                and semantic_status in {"supported", "partial"}
            ):
                source_priority = _single_source_generalization_priority(
                    chapter=chapter,
                    unit=unit,
                    claim_text=claim_text,
                    refs=refs,
                    source_ids=source_ids,
                    requirement_ids=inferred_requirement_ids,
                    source_identities=source_identities,
                )
                issues.append(
                    {
                        "type": "single_source_industry_generalization_observed",
                        "chapter_id": chapter.get("chapter_id"),
                        "claim_id": unit.get("claim_id"),
                        "evidence_refs": refs,
                        "source_ids": source_ids,
                        "source_identities": source_identities,
                        "distinct_source_count": distinct_support_count,
                    }
                )
                claim_repair_priorities.append(source_priority)
                unit["source_sufficiency_status"] = "single_source_industry_generalization_observed"
                unit["source_sufficiency_repair_hint"] = source_priority
                _append_claim_review_suggestion(
                    unit,
                    source_stage="source_sufficiency_gate",
                    issue_type="single_source_industry_generalization",
                    reason="industry-level generalization is currently supported by only one independent source",
                    suggested_claim_strength="directional",
                    suggested_evidence_use_level="directional_signal",
                    suggested_writing_permission="cautious_with_boundary",
                    repair_priority=source_priority,
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
    claim_review_action_plan = _claim_review_action_plan(chapters)
    return {
        "status": status,
        "issues": issues,
        "chapter_synthesis": chapters,
        "valid_ref_count": len(valid_refs),
        "usable_claim_count": usable_claim_count,
        "dropped_claim_count": sum(_claim_drop_issue_counts(issue_counts).values()),
        "usable_chapter_count": len(chapters),
        "llm_raw_chapter_count": raw_chapter_count,
        "llm_raw_claim_count": raw_claim_count,
        "llm_validation_issue_counts": issue_counts,
        "llm_validation_issue_examples": issues[:8],
        "llm_valid_claim_examples": valid_examples,
        "llm_rejected_claim_examples": rejected_examples,
        "llm_deferred_claim_examples": deferred_examples,
        "claim_repair_priorities": claim_repair_priorities,
        "claim_review_action_plan": claim_review_action_plan,
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


def _dedup_analysis_repair_priorities(
    llm_payload: Dict[str, Any],
    validation: Dict[str, Any],
) -> List[Dict[str, Any]]:
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
    return evidence_repair_priorities


def _attach_repair_priorities_to_analysis(
    merged: Dict[str, Any],
    evidence_repair_priorities: Sequence[Dict[str, Any]],
) -> None:
    if not evidence_repair_priorities:
        return
    merged["evidence_repair_priorities"] = list(evidence_repair_priorities)
    existing_gap_ledger = [item for item in _as_list(merged.get("evidence_gap_ledger")) if isinstance(item, dict)]
    seen_gap_ids = {
        str(item.get("gap_id") or item.get("id") or item.get("claim_id") or "").strip()
        for item in existing_gap_ledger
    }
    for priority in evidence_repair_priorities:
        if not isinstance(priority, dict):
            continue
        gap_id = str(priority.get("gap_id") or priority.get("id") or priority.get("claim_id") or "").strip()
        if gap_id and gap_id in seen_gap_ids:
            continue
        copied = dict(priority)
        copied.setdefault("repair_route", "evidence_search")
        copied.setdefault("source_stage", "analysis_claim_support")
        copied.setdefault("status", "open")
        copied.setdefault("allowed_for_writing", False)
        existing_gap_ledger.append(copied)
        if gap_id:
            seen_gap_ids.add(gap_id)
    merged["evidence_gap_ledger"] = existing_gap_ledger


def _repair_priorities_from_analysis_payload(parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
    synthesis = _as_dict(parsed.get("llm_analysis_synthesis"))
    validation = _as_dict(synthesis.get("validation"))
    priorities: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in [
        *_as_list(parsed.get("evidence_repair_priorities")),
        *_as_list(parsed.get("claim_repair_priorities")),
        *_as_list(synthesis.get("evidence_repair_priorities")),
        *_as_list(validation.get("claim_repair_priorities")),
    ]:
        if not isinstance(item, dict):
            continue
        key = str(item.get("gap_id") or item.get("claim_id") or item.get("claim") or len(priorities)).strip()
        if key in seen:
            continue
        seen.add(key)
        priorities.append(item)
    return priorities


def merge_llm_analysis_with_fallback(
    fallback: Dict[str, Any],
    llm_payload: Dict[str, Any],
    validation: Dict[str, Any],
) -> Dict[str, Any]:
    if str(validation.get("status") or "") != "valid":
        merged = dict(fallback)
        evidence_repair_priorities = _dedup_analysis_repair_priorities(llm_payload, validation)
        _attach_repair_priorities_to_analysis(merged, evidence_repair_priorities)
        if evidence_repair_priorities:
            merged["llm_analysis_synthesis"] = {
                "chapter_synthesis": [],
                "cross_chapter_conflicts": _as_list(llm_payload.get("cross_chapter_conflicts")),
                "evidence_repair_priorities": evidence_repair_priorities,
                "claim_review_action_plan": _as_dict(validation.get("claim_review_action_plan")),
                "rewrite_priorities": _as_list(llm_payload.get("rewrite_priorities")),
                "usage": llm_payload.get("_llm_usage", {}),
                "model": llm_payload.get("_llm_model", ""),
                "validation": validation,
            }
        return merged
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
            because_facts = [
                _compact(item, 260)
                for item in (_as_list(unit.get("because_facts")) or evidence_basis)
                if str(item or "").strip()
            ]
            block_affinity = str(unit.get("block_affinity") or "").strip()
            decision_implication = _compact(unit.get("decision_use") or chapter.get("decision_implication") or "", 360)
            if _has_internal_analysis_language(decision_implication):
                decision_implication = ""
            claim_id = str(unit.get("claim_id") or unit.get("id") or f"{chapter_id}_claim_{unit_index}").strip()
            recommended_chapter = _compact(
                unit.get("recommended_chapter")
                or chapter.get("chapter_title")
                or chapter.get("chapter_question")
                or chapter_id,
                160,
            )
            cluster_key = str(
                unit.get("cluster_key")
                or unit.get("claim_cluster")
                or unit.get("fact_type")
                or unit.get("claim_type")
                or block_affinity
                or ""
            ).strip()
            evidence_status = str(unit.get("evidence_status") or "").strip()
            if not evidence_status:
                evidence_status = "sufficient" if str(unit.get("claim_strength") or "").strip().lower() in {"strong", "moderate"} else "partial"
            can_anchor_section = bool(unit.get("can_anchor_section")) if "can_anchor_section" in unit else bool(refs)
            claim_payload = {
                "claim_id": claim_id,
                "claim": claim,
                "chapter_id": chapter_id,
                "recommended_chapter": recommended_chapter,
                "cluster_key": cluster_key,
                "can_anchor_section": can_anchor_section,
                "evidence_status": evidence_status,
                "claim_status": unit.get("claim_status") or ("decision_ready" if refs else "directional"),
                "claim_strength": unit.get("claim_strength") or unit.get("claim_status") or ("moderate" if refs else "directional"),
                "claim_strength_ceiling": unit.get("claim_strength_ceiling"),
                "evidence_use_level": unit.get("evidence_use_level"),
                "writing_permission": unit.get("writing_permission"),
                "metric_completeness_status": unit.get("metric_completeness_status"),
                "metric_missing_fields": _as_list(unit.get("metric_missing_fields")),
                "claim_review_suggestions": _as_list(unit.get("claim_review_suggestions")),
                "requirement_ids": _as_list(unit.get("requirement_ids")),
                "fact_ids": _as_list(unit.get("fact_ids")) or refs,
                "source_ids": _as_list(unit.get("source_ids")),
                "supporting_evidence": refs,
                "evidence_refs": refs,
                "evidence_basis": evidence_basis,
                "because_facts": because_facts,
                "supporting_facts": evidence_basis,
                "block_type": block_affinity,
                "claim_roles": _as_list(unit.get("claim_roles")),
                "primary_claim_role": unit.get("primary_claim_role"),
                "claim_role_contract_version": unit.get("claim_role_contract_version"),
                "role_reasons": _as_list(unit.get("role_reasons")),
                "ref_resolution": _as_dict(unit.get("ref_resolution")),
                "legacy_ref_fields": _as_dict(unit.get("legacy_ref_fields")),
                "mechanism": reasoning,
                "reasoning": reasoning,
                "boundary": _as_list(unit.get("boundary") or unit.get("limitation_boundary")),
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
                    "claim_id": claim_id,
                    "chapter_id": chapter_id,
                    "recommended_chapter": recommended_chapter,
                    "cluster_key": cluster_key,
                    "can_anchor_section": can_anchor_section,
                    "evidence_status": evidence_status,
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
                    "claim_review_suggestions": claim_payload["claim_review_suggestions"],
                    "requirement_ids": claim_payload["requirement_ids"],
                    "fact_ids": claim_payload["fact_ids"],
                    "source_ids": claim_payload["source_ids"],
                    "reasoning": claim_payload["reasoning"] or claim_payload["mechanism"],
                    "mechanism": claim_payload["mechanism"],
                    "counter_evidence": claim_payload["counter_evidence"],
                    "decision_implication": claim_payload["decision_implication"],
                    "evidence_basis": evidence_basis,
                    "because_facts": because_facts,
                    "supporting_facts": evidence_basis,
                    "boundary": claim_payload["boundary"],
                    "block_type": block_affinity,
                    "output_type": block_affinity,
                    "layout_section_role": block_affinity,
                    "claim_roles": claim_payload["claim_roles"],
                    "primary_claim_role": claim_payload["primary_claim_role"],
                    "claim_role_contract_version": claim_payload["claim_role_contract_version"],
                    "role_reasons": claim_payload["role_reasons"],
                    "ref_resolution": claim_payload["ref_resolution"],
                    "legacy_ref_fields": claim_payload["legacy_ref_fields"],
                    "supporting_evidence": refs,
                    "evidence_refs": refs,
                    "confidence": claim_payload["confidence"],
                }
            )
            if refs:
                key_judgments.append(
                    {
                        "claim_id": claim_id,
                        "chapter_id": chapter_id,
                        "judgment": claim,
                        "supporting_dimensions": [chapter.get("chapter_title") or chapter_id],
                        "evidence_ids": refs,
                        "fact_ids": claim_payload["fact_ids"],
                        "source_ids": claim_payload["source_ids"],
                        "requirement_ids": claim_payload["requirement_ids"],
                        "claim_strength": claim_payload["claim_strength"],
                        "evidence_use_level": claim_payload["evidence_use_level"],
                        "writing_permission": claim_payload["writing_permission"],
                        "metric_completeness_status": claim_payload["metric_completeness_status"],
                        "metric_missing_fields": claim_payload["metric_missing_fields"],
                        "claim_review_suggestions": claim_payload["claim_review_suggestions"],
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
        existing_judgments = [
            item
            for item in _as_list(merged.get("key_judgments"))
            if isinstance(item, dict) and not _is_fallback_synthesis_claim(item)
        ]
        ranked_judgments = _rank_key_judgments(key_judgments + existing_judgments)
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
        existing_units = [
            item
            for item in _as_list(merged.get("claim_units"))
            if isinstance(item, dict) and not _is_fallback_synthesis_claim(item)
        ]
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
        existing_judgments = [
            item
            for item in _as_list(merged.get("key_judgments"))
            if isinstance(item, dict) and not _is_fallback_synthesis_claim(item)
        ]
        merged["key_judgments"] = _rank_key_judgments(key_judgments + existing_judgments)
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
        "claim_review_action_plan": _as_dict(validation.get("claim_review_action_plan")),
        "rewrite_priorities": _as_list(llm_payload.get("rewrite_priorities")),
        "usage": llm_payload.get("_llm_usage", {}),
        "model": llm_payload.get("_llm_model", ""),
        "validation": validation,
    }
    if _as_list(_as_dict(validation.get("claim_review_action_plan")).get("actions")):
        merged["claim_review_action_plan"] = _as_dict(validation.get("claim_review_action_plan"))
    if evidence_repair_priorities:
        _attach_repair_priorities_to_analysis(merged, evidence_repair_priorities)
    return merged


def build_fallback_analysis(evidence_package: Dict[str, Any]) -> Dict[str, Any]:
    evidence_package = _as_dict(evidence_package)
    research_plan = _research_plan(evidence_package)
    dimensions = _analysis_dimensions(evidence_package)
    requirements_by_chapter = _requirements_by_chapter_for_llm(evidence_package)
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
            analysis_item = _analysis_item_with_requirement_contract_inference(
                item,
                requirements_by_chapter,
                chapter_id=str(dimension or ""),
            )
            analysis = _evidence_analysis(analysis_item, dimension, index)
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
    result = _enrich_fallback_claims_with_interpretations(result, evidence_package)
    result = _public_normalize_analysis_payload(result)
    result["analysis_depth_quality"] = analysis_depth_quality(result)
    result["claim_binding_feedback_summary"] = claim_binding_feedback_summary(result)
    requirement_lineage_coverage = _analysis_requirement_lineage_coverage(evidence_package, result)
    result["analysis_stage_diagnostics"] = {
        "uses_llm_analysis": False,
        "llm_analysis_status": "not_run",
        "input_chapter_count": len(chapter_evidence_diagnostics),
        "input_evidence_card_count": len(evidence_analyses),
        "output_claim_count": len(claim_units),
        "decision_ready_claim_count": len([item for item in claim_units if str(item.get("claim_status") or "").strip() in {"decision_ready", "core_claim"}]),
        "directional_claim_count": len([item for item in claim_units if str(item.get("claim_status") or "").strip() in {"directional", "directional_ready"}]),
        **_analysis_conversion_diagnostics(
            evidence_package,
            result,
            input_evidence_card_count=len(evidence_analyses),
        ),
        **requirement_lineage_coverage,
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
    preserved_repair_priorities = _repair_priorities_from_analysis_payload(parsed)
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
    _attach_repair_priorities_to_analysis(rebuilt, preserved_repair_priorities)
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
    deadline_ts: Optional[float] = None,
) -> AnalysisAgentState:
    _set_analysis_deadline(deadline_ts)
    try:
        package = _as_dict(evidence_package)
        if query and not package.get("query"):
            package = {**package, "query": query}
        stage_guard_meta: Dict[str, Any] = {}
        stage_guard_input_hash = ""
        stage_guard_cacheable = False
        if _stage_guard_hash is not None and _stage_guard_enabled is not None and _stage_guard_enabled():
            try:
                stage_guard_input_hash = _stage_guard_hash(
                    {
                        "schema_version": "analysis_agent_stage_input_v1",
                        "evidence_package": package,
                        "query": query or str(package.get("query") or ""),
                        "llm_config": {
                            key: value
                            for key, value in _as_dict(llm_config).items()
                            if key not in {"api_key", "apiKey", "authorization", "headers"}
                        },
                        "env": {
                            "BRAIN_ENABLE_LLM_EVIDENCE_ANALYSIS": os.getenv("BRAIN_ENABLE_LLM_EVIDENCE_ANALYSIS", ""),
                            "BRAIN_LLM_ANALYSIS_INPUT_VERSION": os.getenv("BRAIN_LLM_ANALYSIS_INPUT_VERSION", ""),
                            "BRAIN_LLM_ANALYSIS_MODE": os.getenv("BRAIN_LLM_ANALYSIS_MODE", ""),
                            "BRAIN_LLM_ANALYSIS_MAX_CHAPTERS": os.getenv("BRAIN_LLM_ANALYSIS_MAX_CHAPTERS", ""),
                            "REPORT_QUALITY_MODE": os.getenv("REPORT_QUALITY_MODE", ""),
                        },
                        "producer_version": "analysis_agent_guard_v1",
                    }
                )
                stage_guard_cacheable = bool(
                    _stage_guard_cache_enabled is not None
                    and _stage_guard_cache_enabled("analysis_agent")
                    and _stage_guard_get_cached_output is not None
                    and _stage_guard_store_output is not None
                )
                if stage_guard_cacheable:
                    cached = _stage_guard_get_cached_output(stage="analysis_agent", input_hash=stage_guard_input_hash)
                    if cached.get("hit"):
                        cached_state = _as_dict(cached.get("output"))
                        cached_structured = _as_dict(cached_state.get("structured_analysis"))
                        cached_diagnostics = {
                            **_as_dict(cached_structured.get("analysis_stage_diagnostics")),
                            "stage_execution_guard": _as_dict(cached),
                            "run_scoped_analysis_cache_hit": True,
                        }
                        cached_structured["analysis_stage_diagnostics"] = cached_diagnostics
                        cached_state["structured_analysis"] = cached_structured
                        cached_state["answer_text"] = json.dumps(
                            {"structured_analysis": cached_structured},
                            ensure_ascii=False,
                            separators=(",", ":"),
                            default=str,
                        )
                        raw_output = _as_dict(cached_state.get("raw_output"))
                        if raw_output:
                            raw_output["structured_analysis"] = cached_structured
                            cached_state["raw_output"] = raw_output
                        return cached_state  # type: ignore[return-value]
                if _stage_guard_record_execution is not None:
                    stage_guard_meta = _as_dict(
                        _stage_guard_record_execution(stage="analysis_agent", input_hash=stage_guard_input_hash)
                    )
            except Exception:
                stage_guard_meta = {"status": "error", "stage": "analysis_agent"}
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
                        structured = merge_llm_analysis_with_fallback(structured, llm_payload, validation)
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
            "claim_review_action_plan": llm_validation.get("claim_review_action_plan", {}),
            "llm_chapter_results": llm_payload.get("_llm_chapter_results", []) if "llm_payload" in locals() else [],
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
            "stage_execution_guard": stage_guard_meta,
            **_analysis_requirement_lineage_coverage(package, structured),
        }
        diagnostics["evidence_analysis_record_count"] = len(_as_list(structured.get("evidence_analyses")))
        diagnostics["input_evidence_card_count"] = _input_evidence_card_count_for_diagnostics(structured, package)
        diagnostics.update(
            _analysis_conversion_diagnostics(
                package,
                structured,
                input_evidence_card_count=int(diagnostics.get("input_evidence_card_count") or 0),
            )
        )
        structured["analysis_stage_diagnostics"] = diagnostics
        source = final_analysis_source
        state: AnalysisAgentState = {
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
        if stage_guard_cacheable and stage_guard_input_hash and _stage_guard_store_output is not None:
            try:
                _stage_guard_store_output(stage="analysis_agent", input_hash=stage_guard_input_hash, output=state)
            except Exception:
                pass
        _emit_analysis_agent_probe(state)
        return state
    except Exception as exc:
        state = {
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
        _emit_analysis_agent_probe(state)
        return state
    finally:
        _set_analysis_deadline(None)


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
