from __future__ import annotations

import json
import copy
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, TypedDict

try:
    from rag_pipeline.contracts.evidence_quality import apply_evidence_quality_contract
    from rag_pipeline.contracts.research_reflection import build_research_reflection_memo
    from rag_pipeline.contracts.source_registry import pick_refs as _contract_pick_refs
    from .analytics import run_analytics_agents
    from .chapter_evidence_builder import build_chapter_evidence_packages_from_evidence_package
    from .chapter_narrative_agent import run_chapter_narrative
    from .chapter_argument_agent import run_chapter_argument_agent
    from .claim_builder_agent import run_claim_builder_agent
    from .decision_synthesis_agent import run_decision_synthesis_agent
    from .evidence_binder import build_materials_payload_from_packages, run_evidence_binder
    from .evidence_synthesizer import run_evidence_synthesizer
    from .final_writer_agent import run_final_writer_agent
    from .micro_layout_agent import run_micro_layout_agent
    from .package_contracts import validate_pipeline_packages
    from .pre_layout_agent import run_pre_layout_agent
    from .public_report_sanitizer import has_internal_gap_language, sanitize_public_markdown
    from .qa_agent import run_qa_agent
    from .qa_agent import validate_enterprise_report as _validate_enterprise_report
    from .qa_agent import validate_report_narrative_quality
    from .rewrite_agent import run_rewrite_agent
    from .risk_agent import run_risk_agent
    from .table_agent import run_table_agent
    from .table_validator import validate_table_package
except Exception:  # pragma: no cover - direct script mode fallback
    try:
        from rag_pipeline.contracts.evidence_quality import apply_evidence_quality_contract  # type: ignore
        from rag_pipeline.contracts.research_reflection import build_research_reflection_memo  # type: ignore
        from rag_pipeline.contracts.source_registry import pick_refs as _contract_pick_refs  # type: ignore
    except Exception:  # pragma: no cover
        apply_evidence_quality_contract = None  # type: ignore
        build_research_reflection_memo = None  # type: ignore
        _contract_pick_refs = None  # type: ignore
    from analytics import run_analytics_agents  # type: ignore
    from chapter_evidence_builder import build_chapter_evidence_packages_from_evidence_package  # type: ignore
    from chapter_narrative_agent import run_chapter_narrative  # type: ignore
    from chapter_argument_agent import run_chapter_argument_agent  # type: ignore
    from claim_builder_agent import run_claim_builder_agent  # type: ignore
    from decision_synthesis_agent import run_decision_synthesis_agent  # type: ignore
    from evidence_binder import build_materials_payload_from_packages, run_evidence_binder  # type: ignore
    from evidence_synthesizer import run_evidence_synthesizer  # type: ignore
    from final_writer_agent import run_final_writer_agent  # type: ignore
    from micro_layout_agent import run_micro_layout_agent  # type: ignore
    from package_contracts import validate_pipeline_packages  # type: ignore
    from pre_layout_agent import run_pre_layout_agent  # type: ignore
    from public_report_sanitizer import has_internal_gap_language, sanitize_public_markdown  # type: ignore
    from qa_agent import run_qa_agent  # type: ignore
    from qa_agent import validate_enterprise_report as _validate_enterprise_report  # type: ignore
    from qa_agent import validate_report_narrative_quality  # type: ignore
    from rewrite_agent import run_rewrite_agent  # type: ignore
    from risk_agent import run_risk_agent  # type: ignore
    from table_agent import run_table_agent  # type: ignore
    from table_validator import validate_table_package  # type: ignore


AGENT_NAME = "writer_agent"
AGENT_DESCRIPTION = "Compatibility orchestrator for the split report-production pipeline. Final writing is delegated to FinalWriterAgent."


class WriterAgentState(TypedDict, total=False):
    query: str
    writer_report: Dict[str, Any]
    answer_text: str
    raw_output: Dict[str, Any]
    errors: List[str]
    metadata: Dict[str, Any]


@dataclass
class ReportLayout:
    report_type: str
    title_template: str
    subtitle_template: str
    sections: List[str] = field(default_factory=list)


# Legacy export only. The new planner must not default to these dimensions.
INDUSTRY_DIMENSIONS = [
    "市场规模与增速",
    "竞争格局",
    "政策与监管环境",
    "技术路线与产业链",
    "资本动态",
]


PUBLIC_REBUILD_PATTERNS = [
    r"正文\s*只能\s*写成",
    r"本章\s*只能\s*写成",
    r"本章\s*可\s*写成",
    r"本章\s*应\s*写成",
    r"建议避免",
    r"建议在后续版本中补充",
    r"后续版本中补充",
    r"第\s*\d+\s*轮",
    r"\bclaim_status\b",
    r"\bevidence_cards?\b",
]

TABLE_INTERNAL_TRACE_PATTERNS = [
    r"第\s*\d+\s*轮\s*[｜|:：]",
    r"(?:竞争对比|政策监管|技术产业链|市场规模|成本|金额)\s*=\s*(?:[；;]|$)",
    r"\bopenai_task_\d+\b",
]

ENTERPRISE_INDUSTRY_LAYOUT = ReportLayout(
    report_type="industry_deep",
    title_template="{query}研究报告",
    subtitle_template="问题驱动、证据绑定、判断先行",
    sections=["摘要与关键判断", "关键数据", "动态章节", "策略含义", "反向信号", "研究口径"],
)

GENERIC_DYNAMIC_LAYOUT = ReportLayout(
    report_type="topic_report",
    title_template="{query}研究报告",
    subtitle_template="问题驱动、证据绑定、判断先行",
    sections=["摘要与关键判断", "动态章节", "策略含义", "反向信号", "研究口径"],
)


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _renumber_public_chapter_headings(markdown: str) -> str:
    chapter_index = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal chapter_index
        chapter_index += 1
        title = re.sub(r"^\d+\.\s*", "", match.group(1).strip())
        return f"## {chapter_index}. {title}"

    return re.sub(r"^##\s+\d+\.\s+(.+?)\s*$", replace, markdown or "", flags=re.M)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _clean_standard() -> str:
    value = str(os.getenv("REPORT_CLEAN_STANDARD", "balanced") or "").strip().lower()
    if value in {"strict", "balanced", "relaxed"}:
        return value
    return "balanced"


def _count_value(value: Any) -> int:
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


def _evidence_health_summary_from_package(evidence_package: Dict[str, Any]) -> Dict[str, Any]:
    package = _as_dict(evidence_package)
    summary = _as_dict(package.get("summary"))
    metadata = _as_dict(package.get("metadata"))
    return (
        _as_dict(package.get("evidence_health_summary"))
        or _as_dict(summary.get("evidence_health_summary"))
        or _as_dict(metadata.get("evidence_health_summary"))
    )


def _package_warning_types(package_quality_report: Dict[str, Any]) -> List[str]:
    warnings = [
        _as_dict(item).get("type")
        for item in _as_list(package_quality_report.get("warnings"))
        if isinstance(item, dict)
    ]
    return [str(item) for item in warnings if str(item or "").strip()]


def _truthy_quality_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "required", "failed"}
    return bool(value)


def _qa_pending_repair_reasons(qa_result: Dict[str, Any]) -> List[str]:
    qa = _as_dict(qa_result)
    reasons: List[str] = []
    qa_passed = bool(qa.get("passed"))
    report_family = str(qa.get("report_family") or "").strip().lower()
    deep_report = report_family == "industry_deep_report" or "deep" in report_family
    repair_required = _truthy_quality_flag(qa.get("repair_required"))
    has_followup_split = any(
        key in qa
        for key in (
            "blocking_followups",
            "advisory_followups",
            "blocking_evidence_repair_followups",
            "blocking_content_repair_followups",
        )
    )
    if repair_required:
        reasons.append("repair_required")
    if deep_report and not bool(qa.get("publishable")):
        reasons.append("deep_report_not_publishable")
    if has_followup_split:
        if _as_list(qa.get("blocking_followups")):
            reasons.append("blocking_followups")
        if _as_list(qa.get("blocking_evidence_repair_followups")):
            reasons.append("blocking_evidence_repair_followups")
        if _as_list(qa.get("blocking_content_repair_followups")):
            reasons.append("blocking_content_repair_followups")
        if deep_report:
            high_advisory = [
                item
                for item in _as_list(qa.get("advisory_followups"))
                if str(_as_dict(item).get("priority") or "").strip().lower() == "high"
            ]
            if high_advisory:
                reasons.append("high_priority_advisory_followups")
    else:
        if _as_list(qa.get("repair_followups")):
            reasons.append("repair_followups")
        if _as_list(qa.get("evidence_repair_followups")):
            reasons.append("evidence_repair_followups")
        if _as_list(qa.get("content_repair_followups")):
            reasons.append("content_repair_followups")
        if _as_list(_as_dict(qa.get("deep_evaluation")).get("required_followups")) and not (qa_passed and not repair_required):
            reasons.append("required_followups")
    result: List[str] = []
    seen = set()
    for reason in reasons:
        if reason in seen:
            continue
        seen.add(reason)
        result.append(reason)
    return result


def _qa_has_pending_repair(qa_result: Dict[str, Any]) -> bool:
    return bool(_qa_pending_repair_reasons(qa_result))


def _writer_ready_for_final(
    *,
    markdown: str,
    qa_result: Dict[str, Any],
    package_passed: bool,
    package_warning_blocked: bool,
) -> bool:
    return (
        bool(_as_dict(qa_result).get("passed"))
        and not _qa_has_pending_repair(qa_result)
        and bool(package_passed)
        and not bool(package_warning_blocked)
        and not has_internal_gap_language(str(markdown or ""))
    )


def _compact(value: Any, max_chars: int = 260) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "..."


def _dedupe(values: Iterable[Any], *, limit: int = 20) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        text = _compact(value, 180)
        if not text:
            continue
        key = re.sub(r"\s+", "", text.lower())
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def _env_int(name: str, default: int, *, min_value: int = 0, max_value: int = 1_000_000) -> int:
    try:
        value = int(os.getenv(name, str(default)) or default)
    except (TypeError, ValueError):
        value = default
    return max(min_value, min(max_value, value))


def _body_table_budget() -> int:
    if os.getenv("REPORT_MAX_BODY_TABLES") is not None:
        requested = _env_int("REPORT_MAX_BODY_TABLES", 6, min_value=0, max_value=20)
    else:
        requested = 6
    hard_limit = _env_int("REPORT_HARD_MAX_BODY_TABLES", 12, min_value=0, max_value=50)
    return max(0, min(requested, hard_limit))


def _per_chapter_table_budget() -> int:
    return _env_int("REPORT_MAX_BODY_TABLES_PER_CHAPTER", 1, min_value=0, max_value=6)


def _refs_from_evidence_package(package: Dict[str, Any], *, limit: int = 8) -> List[str]:
    refs: List[Any] = []
    for collection in ("core_evidence", "supporting_evidence", "table_evidence", "evidence_items"):
        for item in _as_list(package.get(collection)):
            if not isinstance(item, dict):
                continue
            refs.extend(_pick_refs(item, limit=3))
            for key in ("ref", "evidence_id", "source_ref"):
                if item.get(key):
                    refs.append(item.get(key))
    return _dedupe(refs, limit=limit)


def _first_public_unit_by_chapter(argument_units: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_chapter: Dict[str, Dict[str, Any]] = {}
    for unit in list(argument_units or []):
        if not isinstance(unit, dict) or unit.get("omit_from_report"):
            continue
        if unit.get("public_render") is False:
            continue
        chapter_id = str(unit.get("chapter_id") or "").strip()
        if not chapter_id:
            continue
        if chapter_id not in by_chapter:
            by_chapter[chapter_id] = unit
    return by_chapter


def _unit_lookup(argument_units: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for unit in list(argument_units or []):
        if not isinstance(unit, dict) or unit.get("omit_from_report"):
            continue
        if unit.get("public_render") is False:
            continue
        for key in (unit.get("section_id"), unit.get("section_title"), unit.get("question")):
            text = str(key or "").strip()
            if text and text not in lookup:
                lookup[text] = unit
    return lookup


def _scan_public_rebuild_hits(value: Any, *, path: str = "", limit: int = 24) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []

    def walk(node: Any, node_path: str) -> None:
        if len(hits) >= limit:
            return
        if isinstance(node, dict):
            for key, child in node.items():
                walk(child, f"{node_path}.{key}" if node_path else str(key))
                if len(hits) >= limit:
                    return
            return
        if isinstance(node, list):
            for index, child in enumerate(node[:80]):
                walk(child, f"{node_path}[{index}]")
                if len(hits) >= limit:
                    return
            return
        if not isinstance(node, str):
            return
        text = str(node or "")
        for pattern in PUBLIC_REBUILD_PATTERNS:
            if re.search(pattern, text, flags=re.I):
                hits.append(
                    {
                        "path": node_path,
                        "pattern": pattern,
                        "snippet": _compact(text, 180),
                    }
                )
                return

    walk(value, path)
    return hits


def needs_public_rebuild(
    *,
    argument_units: Optional[Sequence[Dict[str, Any]]] = None,
    chapter_packages: Optional[Sequence[Dict[str, Any]]] = None,
    structured_analysis: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    hits: List[Dict[str, Any]] = []
    hits.extend(_scan_public_rebuild_hits(argument_units or [], path="argument_units", limit=16))
    hits.extend(_scan_public_rebuild_hits(chapter_packages or [], path="chapter_packages", limit=16))
    hits.extend(_scan_public_rebuild_hits(_as_dict(structured_analysis), path="structured_analysis", limit=8))
    quality = _as_dict(_as_dict(structured_analysis).get("analysis_depth_quality"))
    status = str(quality.get("status") or "").strip().lower()
    repeated_ratio = _safe_float(quality.get("repeated_claim_ratio"), 0.0)
    title_as_claim_count = int(_safe_float(quality.get("title_as_claim_count"), 0.0))
    ref_mismatch_count = int(_safe_float(quality.get("evidence_ref_mismatch_count"), 0.0))
    if status == "needs_rewrite":
        hits.append(
            {
                "path": "structured_analysis.analysis_depth_quality.status",
                "pattern": "needs_rewrite",
                "snippet": "analysis_depth_quality requested rewrite",
            }
        )
    if repeated_ratio > 0.30:
        hits.append(
            {
                "path": "structured_analysis.analysis_depth_quality.repeated_claim_ratio",
                "pattern": "repeated_claim_ratio",
                "snippet": f"repeated_claim_ratio={repeated_ratio}",
            }
        )
    if title_as_claim_count > 0:
        hits.append(
            {
                "path": "structured_analysis.analysis_depth_quality.title_as_claim_count",
                "pattern": "title_as_claim_count",
                "snippet": f"title_as_claim_count={title_as_claim_count}",
            }
        )
    if ref_mismatch_count > 0:
        hits.append(
            {
                "path": "structured_analysis.analysis_depth_quality.evidence_ref_mismatch_count",
                "pattern": "evidence_ref_mismatch_count",
                "snippet": f"evidence_ref_mismatch_count={ref_mismatch_count}",
            }
        )
    return {
        "required": bool(hits),
        "hit_count": len(hits),
        "hits": hits[:24],
        "argument_unit_count_before": len([item for item in list(argument_units or []) if isinstance(item, dict)]),
        "chapter_package_count_before": len([item for item in list(chapter_packages or []) if isinstance(item, dict)]),
    }


def rebuild_public_argument_pipeline(
    *,
    chapter_evidence_packages: Sequence[Dict[str, Any]],
    micro_layouts: Sequence[Dict[str, Any]],
    structured_analysis: Dict[str, Any],
    report_blueprint: Dict[str, Any],
    table_packages: Sequence[Dict[str, Any]],
    llm_client: Any = None,
) -> Dict[str, Any]:
    rebuilt_units = run_claim_builder_agent(
        chapter_evidence_packages=chapter_evidence_packages,
        micro_layouts=micro_layouts,
        structured_analysis=structured_analysis,
        llm_client=llm_client,
    )
    rebuilt_chapters = run_chapter_argument_agent(
        report_blueprint=report_blueprint,
        micro_layouts=micro_layouts,
        argument_units=rebuilt_units,
        table_packages=table_packages,
        chapter_evidence_packages=chapter_evidence_packages,
        llm_client=llm_client,
    )
    return {
        "argument_units": _as_list(rebuilt_units),
        "chapter_packages": _as_list(rebuilt_chapters),
        "argument_unit_count_after": len([item for item in _as_list(rebuilt_units) if isinstance(item, dict)]),
        "chapter_package_count_after": len([item for item in _as_list(rebuilt_chapters) if isinstance(item, dict)]),
    }


def _chapter_evidence_packages_are_compacted(packages: Optional[Sequence[Dict[str, Any]]]) -> bool:
    checked = 0
    compacted = 0
    for package in list(packages or []):
        if not isinstance(package, dict):
            continue
        checked += 1
        has_counts = bool(package.get("evidence_counts")) or any(
            package.get(f"{key}_count") not in (None, "", 0)
            for key in (
                "core_evidence",
                "supporting_evidence",
                "metric_evidence",
                "case_evidence",
                "counter_evidence",
                "directional_evidence",
                "sample_evidence",
            )
        )
        has_substantive_lists = any(
            _as_list(package.get(key))
            for key in (
                "core_evidence",
                "supporting_evidence",
                "metric_evidence",
                "case_evidence",
                "counter_evidence",
                "directional_evidence",
                "table_evidence",
                "evidence_items",
                "analysis_ready_evidence",
            )
        )
        if has_counts and not has_substantive_lists:
            compacted += 1
    return bool(checked and compacted >= max(1, checked // 2))


def _chapter_evidence_hydrated_count(packages: Optional[Sequence[Dict[str, Any]]]) -> int:
    total = 0
    for package in list(packages or []):
        if not isinstance(package, dict):
            continue
        for key in (
            "core_evidence",
            "supporting_evidence",
            "metric_evidence",
            "case_evidence",
            "counter_evidence",
            "directional_evidence",
        ):
            total += _safe_len(package.get(key))
    return total


def _chapter_evidence_signal_count(packages: Optional[Sequence[Dict[str, Any]]]) -> int:
    total = 0
    for package in list(packages or []):
        if not isinstance(package, dict):
            continue
        counts = _as_dict(package.get("evidence_counts"))
        for key in (
            "core_evidence",
            "supporting_evidence",
            "metric_evidence",
            "case_evidence",
            "counter_evidence",
            "directional_evidence",
            "sample_evidence",
        ):
            total += _safe_len(package.get(key))
            try:
                total += int(float(package.get(f"{key}_count") or counts.get(key) or 0))
            except (TypeError, ValueError):
                pass
    return total


def _chapter_evidence_layered_count(packages: Optional[Sequence[Dict[str, Any]]]) -> int:
    total = 0
    for package in list(packages or []):
        if not isinstance(package, dict):
            continue
        for key in ("metric_evidence", "case_evidence", "counter_evidence", "directional_evidence"):
            total += _safe_len(package.get(key))
            try:
                total += int(float(package.get(f"{key}_count") or 0))
            except (TypeError, ValueError):
                pass
    return total


def _chapter_evidence_selection_score(packages: Optional[Sequence[Dict[str, Any]]]) -> int:
    layered = _chapter_evidence_layered_count(packages)
    signal = _chapter_evidence_signal_count(packages)
    hydrated = _chapter_evidence_hydrated_count(packages)
    return layered * 3 + signal * 2 + hydrated


def _fill_section_from_unit(section: Dict[str, Any], unit: Dict[str, Any], fallback_refs: Sequence[str]) -> Dict[str, Any]:
    result = dict(section)
    if not str(result.get("section_title") or "").strip():
        result["section_title"] = unit.get("section_title") or unit.get("question") or "证据边界"
    if not str(result.get("claim") or "").strip():
        result["claim"] = unit.get("claim") or unit.get("section_title") or result.get("section_title")
    if not str(result.get("reasoning") or "").strip() and unit.get("reasoning"):
        result["reasoning"] = unit.get("reasoning")
    if not str(result.get("counter_evidence") or "").strip():
        result["counter_evidence"] = unit.get("counter_evidence") or "仍需跟踪反证信号和口径变化。"
    if not str(result.get("actionable") or "").strip():
        result["actionable"] = unit.get("actionable") or unit.get("decision_implication") or "优先补充可核验来源并持续跟踪。"
    if not _as_list(result.get("evidence_refs")):
        refs = _pick_refs(unit, limit=8) or list(fallback_refs or [])
        if refs:
            result["evidence_refs"] = refs
    return result


def _sanitize_public_table_cell(value: Any, *, max_chars: int = 180) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return ""
    text = re.sub(r"第\s*\d+\s*轮\s*[｜|:：]\s*", "", text)
    text = re.sub(r"^[^：:]{0,120}(?:报告|研究|query|查询|检索)[^：:]{0,120}[：:]\s*", "", text, flags=re.I)
    text = re.sub(r"(?:竞争对比|政策监管|技术产业链|市场规模|成本|金额)\s*=\s*(?=；|;|$)", "", text)
    text = re.sub(r"(?:；\s*){2,}", "；", text).strip(" ；;，,")
    if any(re.search(pattern, text, flags=re.I) for pattern in TABLE_INTERNAL_TRACE_PATTERNS):
        return ""
    if len(text) > max_chars:
        text = text[: max(0, max_chars - 1)].rstrip() + "..."
    return text


def _sanitize_table_for_public(table: Dict[str, Any], diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(table)
    rows = []
    dropped_rows = 0
    for row in _as_list(result.get("rows")):
        row_dict = dict(row) if isinstance(row, dict) else {"cells": _as_list(row)}
        cells = [_sanitize_public_table_cell(cell, max_chars=160) for cell in _as_list(row_dict.get("cells"))]
        if not cells or any(not str(cell or "").strip() for cell in cells[:1]):
            dropped_rows += 1
            continue
        if not any(str(cell or "").strip() for cell in cells):
            dropped_rows += 1
            continue
        row_dict["cells"] = cells
        row_claim = _sanitize_public_table_cell(row_dict.get("row_claim"), max_chars=240)
        if row_claim:
            row_dict["row_claim"] = row_claim
        rows.append(row_dict)
    if dropped_rows:
        diagnostics["sanitized_table_row_count"] = int(diagnostics.get("sanitized_table_row_count") or 0) + dropped_rows
        result.setdefault("reject_reasons", [])
        if isinstance(result["reject_reasons"], list):
            result["reject_reasons"].append("internal_or_empty_table_rows_removed")
    result["rows"] = rows
    if not rows:
        result["should_render"] = False
        result.setdefault("reject_reasons", [])
        if isinstance(result["reject_reasons"], list):
            result["reject_reasons"].append("no_public_table_rows_after_sanitization")
    return result


def _demote_invalid_table(table: Dict[str, Any], diagnostics: Dict[str, Any], *, nested: bool = False) -> Dict[str, Any]:
    was_renderable = bool(table.get("should_render")) and not table.get("appendix_only")
    result = _sanitize_table_for_public(dict(table), diagnostics)
    if not result.get("should_render") or result.get("appendix_only"):
        if was_renderable and not result.get("appendix_only") and not result.get("demoted_from_clean_report"):
            result["demoted_from_clean_report"] = True
            diagnostics["demoted_table_count"] = int(diagnostics.get("demoted_table_count") or 0) + 1
            if len(diagnostics.setdefault("demoted_tables", [])) < 8:
                diagnostics["demoted_tables"].append(
                    {
                        "table_id": result.get("table_id"),
                        "chapter_id": result.get("chapter_id"),
                        "nested": nested,
                        "error_types": _as_list(result.get("reject_reasons"))[:6],
                    }
                )
        return result
    validation = validate_table_package(result)
    if validation.get("passed"):
        return result
    result["should_render"] = False
    result["demoted_from_clean_report"] = True
    result["table_validation_for_clean"] = validation
    reasons = result.setdefault("reject_reasons", [])
    if isinstance(reasons, list):
        reasons.append("contract_normalizer_table_validation_failed")
    diagnostics["demoted_table_count"] = int(diagnostics.get("demoted_table_count") or 0) + 1
    if len(diagnostics.setdefault("demoted_tables", [])) < 8:
        diagnostics["demoted_tables"].append(
            {
                "table_id": result.get("table_id"),
                "chapter_id": result.get("chapter_id"),
                "nested": nested,
                "error_types": [str(_as_dict(item).get("type") or item) for item in _as_list(validation.get("errors"))][:6],
            }
        )
    return result


def _normalize_public_packages_for_contract(
    *,
    chapter_evidence_packages: Sequence[Dict[str, Any]],
    micro_layouts: Sequence[Dict[str, Any]],
    table_packages: Sequence[Dict[str, Any]],
    argument_units: Sequence[Dict[str, Any]],
    chapter_packages: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    diagnostics: Dict[str, Any] = {
        "demoted_table_count": 0,
        "omitted_section_count": 0,
        "omitted_chapter_count": 0,
        "filled_section_count": 0,
        "demoted_tables": [],
        "omitted_sections": [],
    }
    evidence_by_chapter = {
        str(package.get("chapter_id") or "").strip(): package
        for package in list(chapter_evidence_packages or [])
        if isinstance(package, dict) and str(package.get("chapter_id") or "").strip()
    }
    units = [dict(unit) for unit in list(argument_units or []) if isinstance(unit, dict)]
    for unit in units:
        if unit.get("omit_from_report") or unit.get("public_render") is False:
            continue
        chapter_id = str(unit.get("chapter_id") or "").strip()
        fallback_refs = _refs_from_evidence_package(_as_dict(evidence_by_chapter.get(chapter_id)))
        claim = _compact(unit.get("claim") or unit.get("section_title") or unit.get("question"), 220)
        if claim and not str(unit.get("reasoning") or "").strip():
            unit["reasoning"] = f"{claim} 这一判断需要结合已绑定证据、时间口径和反向样本持续验证。"
        if not str(unit.get("counter_evidence") or "").strip():
            unit["counter_evidence"] = "反向边界在于后续官方披露、订单或经营指标不能延续当前信号。"
        if not str(unit.get("actionable") or unit.get("decision_implication") or "").strip():
            unit["actionable"] = "优先验证关键来源、指标口径和反向样本，再决定是否提高判断权重。"
        if not _as_list(unit.get("evidence_refs")) and fallback_refs:
            unit["evidence_refs"] = list(fallback_refs)
    first_unit = _first_public_unit_by_chapter(units)
    unit_lookup = _unit_lookup(units)

    normalized_tables = [
        _demote_invalid_table(dict(table), diagnostics)
        for table in list(table_packages or [])
        if isinstance(table, dict)
    ]

    normalized_layouts: List[Dict[str, Any]] = []
    for layout in list(micro_layouts or []):
        if not isinstance(layout, dict):
            continue
        copied = copy.deepcopy(layout)
        chapter_id = str(copied.get("chapter_id") or "").strip()
        fallback_refs = _refs_from_evidence_package(_as_dict(evidence_by_chapter.get(chapter_id)))
        followups = _as_list(copied.get("follow_up_queries"))
        normalized_sections: List[Dict[str, Any]] = []
        for section in _as_list(copied.get("sections")):
            if not isinstance(section, dict):
                continue
            section_id = str(section.get("section_id") or "").strip()
            section_title = str(section.get("section_title") or "").strip()
            unit = unit_lookup.get(section_id) or unit_lookup.get(section_title) or first_unit.get(chapter_id) or {}
            next_section = dict(section)
            if unit:
                next_section = _fill_section_from_unit(next_section, unit, fallback_refs)
            elif fallback_refs and not _as_list(next_section.get("required_evidence_refs")):
                next_section["required_evidence_refs"] = list(fallback_refs)
            if not str(next_section.get("section_title") or "").strip():
                next_section["section_title"] = "证据边界"
            if not _as_list(next_section.get("required_evidence_refs")) and not followups:
                diagnostics["omitted_section_count"] += 1
                if len(diagnostics["omitted_sections"]) < 12:
                    diagnostics["omitted_sections"].append({"chapter_id": chapter_id, "section_id": section_id, "source": "micro_layout"})
                continue
            normalized_sections.append(next_section)
        copied["sections"] = normalized_sections
        normalized_layouts.append(copied)

    normalized_chapters: List[Dict[str, Any]] = []
    for chapter in list(chapter_packages or []):
        if not isinstance(chapter, dict):
            continue
        copied = copy.deepcopy(chapter)
        chapter_id = str(copied.get("chapter_id") or "").strip()
        fallback_refs = _refs_from_evidence_package(_as_dict(evidence_by_chapter.get(chapter_id)))
        fallback_unit = first_unit.get(chapter_id) or {}
        if not str(copied.get("lead") or "").strip() and fallback_unit:
            copied["lead"] = _compact(fallback_unit.get("claim") or fallback_unit.get("reasoning"), 260)
        copied["table_packages"] = [
            _demote_invalid_table(dict(table), diagnostics, nested=True)
            for table in _as_list(copied.get("table_packages"))
            if isinstance(table, dict)
        ]
        normalized_sections = []
        for section in _as_list(copied.get("sections")):
            if not isinstance(section, dict):
                continue
            section_id = str(section.get("section_id") or "").strip()
            section_title = str(section.get("section_title") or "").strip()
            unit = unit_lookup.get(section_id) or unit_lookup.get(section_title) or fallback_unit
            next_section = _fill_section_from_unit(dict(section), unit, fallback_refs) if unit else dict(section)
            if not _as_list(next_section.get("evidence_refs")) and fallback_refs:
                next_section["evidence_refs"] = list(fallback_refs)
            if (
                not str(next_section.get("claim") or "").strip()
                or not _as_list(next_section.get("evidence_refs"))
            ):
                diagnostics["omitted_section_count"] += 1
                if len(diagnostics["omitted_sections"]) < 12:
                    diagnostics["omitted_sections"].append({"chapter_id": chapter_id, "section_id": section_id, "source": "chapter_package"})
                continue
            if next_section != section:
                diagnostics["filled_section_count"] += 1
            normalized_sections.append(next_section)
        public_tables = [
            table for table in _as_list(copied.get("table_packages")) if isinstance(table, dict) and table.get("should_render")
        ]
        if not normalized_sections and not public_tables:
            if fallback_refs and str(copied.get("lead") or "").strip():
                normalized_sections.append(
                    {
                        "section_id": f"{chapter_id or 'chapter'}_evidence_boundary",
                        "section_title": "证据边界",
                        "claim": _compact(copied.get("lead"), 220),
                        "reasoning": _compact(copied.get("lead"), 260),
                        "counter_evidence": "仍需跟踪反证信号和口径变化。",
                        "actionable": "优先补充可核验来源并持续跟踪。",
                        "evidence_refs": fallback_refs,
                    }
                )
                diagnostics["filled_section_count"] += 1
            else:
                copied["omit_from_report"] = True
                copied["omit_from_clean_report"] = True
                copied["omit_reason"] = "contract_normalizer_no_supported_sections"
                diagnostics["omitted_chapter_count"] += 1
        copied["sections"] = normalized_sections
        normalized_chapters.append(copied)

    return {
        "micro_layouts": normalized_layouts,
        "table_packages": normalized_tables,
        "argument_units": units,
        "chapter_packages": normalized_chapters,
        "summary": diagnostics,
    }


def _env_float(name: str, default: float, *, min_value: float = 0.0, max_value: float = 10.0) -> float:
    try:
        value = float(os.getenv(name, str(default)) or default)
    except (TypeError, ValueError):
        value = default
    return max(min_value, min(max_value, value))


def _payload_mode(name: str, default: str = "summary") -> str:
    value = str(os.getenv(name, default) or default).strip().lower()
    return value if value in {"summary", "full"} else default


def _include_full_pipeline_artifacts() -> bool:
    if _env_flag("REPORT_INCLUDE_FULL_PIPELINE_ARTIFACTS", False):
        return True
    return _payload_mode("REPORT_PIPELINE_PAYLOAD_MODE", "summary") == "full"


def _include_full_debug_payload() -> bool:
    if _env_flag("REPORT_INCLUDE_FULL_DEBUG", False):
        return True
    if _include_full_pipeline_artifacts():
        return True
    return _payload_mode("REPORT_DEBUG_PAYLOAD_MODE", "summary") == "full"


def _safe_len(value: Any) -> int:
    if isinstance(value, (list, tuple, set, dict)):
        return len(value)
    return 0


def _pick_refs(item: Dict[str, Any], *, limit: int = 6) -> List[str]:
    if _contract_pick_refs is not None:
        return _contract_pick_refs(item, limit=limit)
    refs: List[Any] = []
    for key in ("source_refs", "evidence_refs", "supporting_evidence", "refs"):
        refs.extend(_as_list(item.get(key)))
    for key in ("source_ref", "ref", "evidence_id"):
        if item.get(key):
            refs.append(item.get(key))
    return _dedupe(refs, limit=limit)


def _compact_mapping(item: Dict[str, Any], keys: Sequence[str], *, text_chars: int = 220) -> Dict[str, Any]:
    compacted: Dict[str, Any] = {}
    for key in keys:
        value = item.get(key)
        if value in (None, "", [], {}):
            continue
        if isinstance(value, (str, int, float, bool)):
            compacted[key] = _compact(value, text_chars) if isinstance(value, str) else value
        elif isinstance(value, list):
            compacted[key] = [_compact(entry, 120) for entry in value[:6] if str(entry or "").strip()]
        elif isinstance(value, dict):
            compacted[key] = {
                str(sub_key): _compact(sub_value, 120)
                for sub_key, sub_value in list(value.items())[:8]
                if sub_value not in (None, "", [], {})
            }
    return compacted


def _compact_evidence_item(item: Dict[str, Any]) -> Dict[str, Any]:
    compacted = _compact_mapping(
        item,
        [
            "evidence_id",
            "ref",
            "source_ref",
            "source_level",
            "source_type",
            "metric",
            "indicator",
            "value",
            "display_value",
            "unit",
            "period",
            "region",
            "confidence",
            "metric_validation_status",
        ],
        text_chars=120,
    )
    fact = item.get("fact") or item.get("clean_fact") or item.get("summary") or item.get("content")
    if fact:
        compacted["fact"] = _compact(fact, 160)
    refs = _pick_refs(item, limit=4)
    if refs:
        compacted["refs"] = refs
    return compacted


def _compact_issue_list(items: Sequence[Any], *, limit: int = 4) -> List[Any]:
    compacted: List[Any] = []
    for item in list(items or []):
        if isinstance(item, dict):
            compacted.append(
                _compact_mapping(
                    item,
                    ["type", "required", "actual", "severity", "suggestion", "follow_up_query"],
                    text_chars=110,
                )
            )
        else:
            compacted.append(_compact(item, 140))
        if len(compacted) >= limit:
            break
    return compacted


def _compact_evidence_package(package: Dict[str, Any]) -> Dict[str, Any]:
    evidence_layers = (
        "core_evidence",
        "supporting_evidence",
        "metric_evidence",
        "case_evidence",
        "counter_evidence",
        "directional_evidence",
        "sample_evidence",
        "table_evidence",
        "clue_evidence",
        "appendix_evidence",
        "evidence_items",
    )
    counts = {
        key: _safe_len(package.get(key))
        for key in evidence_layers
        if _safe_len(package.get(key))
    }
    compact_layers: Dict[str, List[Dict[str, Any]]] = {}
    for collection in evidence_layers:
        layer_items = []
        for item in _as_list(package.get(collection)):
            if isinstance(item, dict):
                layer_items.append(_compact_evidence_item(item))
            if len(layer_items) >= 3:
                break
        if layer_items:
            compact_layers[collection] = layer_items
    samples: List[Dict[str, Any]] = []
    for collection in (
        "core_evidence",
        "supporting_evidence",
        "metric_evidence",
        "case_evidence",
        "counter_evidence",
        "directional_evidence",
        "sample_evidence",
        "table_evidence",
        "clue_evidence",
    ):
        for item in _as_list(package.get(collection)):
            if isinstance(item, dict):
                samples.append(_compact_evidence_item(item))
            if len(samples) >= 8:
                break
        if len(samples) >= 8:
            break
    quality = _as_dict(package.get("evidence_quality_summary"))
    return {
        key: value
        for key, value in {
            "chapter_id": package.get("chapter_id"),
            "chapter_title": _compact(package.get("chapter_title"), 160),
            "chapter_question": _compact(package.get("chapter_question"), 180),
            "evidence_counts": counts,
            "core_evidence_count": counts.get("core_evidence", 0),
            "supporting_evidence_count": counts.get("supporting_evidence", 0),
            "metric_evidence_count": counts.get("metric_evidence", 0),
            "case_evidence_count": counts.get("case_evidence", 0),
            "counter_evidence_count": counts.get("counter_evidence", 0),
            "directional_evidence_count": counts.get("directional_evidence", 0),
            **compact_layers,
            "sample_evidence": samples[:2],
            "missing_evidence": _compact_issue_list(_as_list(package.get("missing_evidence")), limit=3),
            "evidence_quality_summary": _compact_mapping(quality, list(quality.keys()), text_chars=120),
            "metadata": _compact_mapping(_as_dict(package.get("metadata")), list(_as_dict(package.get("metadata")).keys()), text_chars=120),
            "unresolved_evidence_refs": _as_list(package.get("unresolved_evidence_refs"))[:8],
            "unresolved_evidence_ref_count": package.get("unresolved_evidence_ref_count"),
        }.items()
        if value not in (None, "", [], {})
    }


def _compact_table_package(table: Dict[str, Any]) -> Dict[str, Any]:
    rows = _as_list(table.get("rows"))
    columns = _as_list(table.get("columns"))
    compacted = _compact_mapping(
        table,
        [
            "table_id",
            "chapter_id",
            "title",
            "table_title",
            "table_type",
            "role",
            "should_render",
            "appendix_only",
            "source_level",
        ],
        text_chars=180,
    )
    compacted["row_count"] = len(rows)
    compacted["column_count"] = len(columns)
    refs = _pick_refs(table, limit=8)
    if refs:
        compacted["evidence_refs"] = refs
    return compacted


def _compact_argument_unit(unit: Dict[str, Any]) -> Dict[str, Any]:
    compacted = _compact_mapping(
        unit,
        [
            "chapter_id",
            "section_id",
            "section_title",
            "block_type",
            "confidence",
            "claim_status",
            "quality_status",
            "public_render",
            "omit_from_report",
        ],
        text_chars=160,
    )
    if unit.get("claim"):
        compacted["claim"] = _compact(unit.get("claim"), 140)
    for key in ("reasoning", "counter_evidence", "actionable"):
        if unit.get(key):
            compacted[f"{key}_chars"] = len(str(unit.get(key) or ""))
    refs = _pick_refs(unit, limit=8)
    if refs:
        compacted["evidence_refs"] = refs
    return compacted


def _compact_chapter_package(chapter: Dict[str, Any]) -> Dict[str, Any]:
    sections = [_compact_argument_unit(section) for section in _as_list(chapter.get("sections")) if isinstance(section, dict)]
    summary = _as_dict(chapter.get("chapter_summary"))
    return {
        key: value
        for key, value in {
            "chapter_id": chapter.get("chapter_id"),
            "chapter_title": _compact(chapter.get("chapter_title"), 160),
            "chapter_question": _compact(chapter.get("chapter_question"), 180),
            "lead": _compact(chapter.get("lead"), 220),
            "section_count": len(_as_list(chapter.get("sections"))),
            "table_count": len(_as_list(chapter.get("table_packages"))),
            "sections": sections[:3],
            "chapter_summary": _compact_mapping(summary, list(summary.keys()), text_chars=180),
            "evidence_gap_count": len(_as_list(chapter.get("evidence_gaps"))),
            "missing_proof_count": len(_as_list(chapter.get("missing_proof_standards"))),
            "omit_from_report": chapter.get("omit_from_report"),
        }.items()
        if value not in (None, "", [], {})
    }


def _compact_analytics_output(output: Dict[str, Any]) -> Dict[str, Any]:
    compacted = _compact_mapping(
        output,
        ["agent", "agent_name", "analysis_type", "chapter_id", "title", "summary", "status"],
        text_chars=220,
    )
    for key in ("tables", "metrics", "matrix", "findings", "risk_register", "timeline"):
        if key in output:
            compacted[f"{key}_count"] = _safe_len(output.get(key))
    return compacted


def _compact_micro_layout(layout: Dict[str, Any]) -> Dict[str, Any]:
    sections = [_as_dict(section) for section in _as_list(layout.get("sections"))]
    return {
        key: value
        for key, value in {
            "chapter_id": layout.get("chapter_id"),
            "chapter_title": _compact(layout.get("chapter_title") or layout.get("title"), 160),
            "section_count": len(sections),
            "sections": [
                _compact_mapping(
                    section,
                    ["section_id", "section_title", "section_role", "block_type", "output_type"],
                    text_chars=120,
                )
                for section in sections[:2]
            ],
            "table_count": len(_as_list(layout.get("tables") or layout.get("table_specs"))),
        }.items()
        if value not in (None, "", [], {})
    }


def _compact_sequence(items: Sequence[Any], compact_one: Any, *, limit: int = 40) -> List[Any]:
    compacted: List[Any] = []
    for item in list(items or []):
        if isinstance(item, dict):
            compacted.append(compact_one(item))
        else:
            compacted.append(_compact(item, 180))
        if len(compacted) >= limit:
            break
    return compacted


def _pipeline_artifact_summary(
    *,
    chapter_evidence_packages: Sequence[Dict[str, Any]],
    evidence_graph: Dict[str, Any],
    analytics_outputs: Sequence[Dict[str, Any]],
    micro_layouts: Sequence[Dict[str, Any]],
    table_packages: Sequence[Dict[str, Any]],
    argument_units: Sequence[Dict[str, Any]],
    chapter_packages: Sequence[Dict[str, Any]],
    source_registry: Sequence[Dict[str, Any]],
    qa_result: Dict[str, Any],
    package_quality_report: Dict[str, Any],
) -> Dict[str, Any]:
    rendered_tables = [table for table in table_packages if isinstance(table, dict) and table.get("should_render") and not table.get("appendix_only")]
    rendered_high_value_tables = [table for table in rendered_tables if str(table.get("table_value_tier") or "") == "high"]
    public_units = [unit for unit in argument_units if isinstance(unit, dict) and unit.get("public_render") and not unit.get("omit_from_report")]
    public_chapters = [chapter for chapter in chapter_packages if isinstance(chapter, dict) and not chapter.get("omit_from_report")]
    return {
        "chapter_count": len(public_chapters),
        "evidence_package_count": len([item for item in chapter_evidence_packages if isinstance(item, dict)]),
        "analytics_output_count": len([item for item in analytics_outputs if isinstance(item, dict)]),
        "micro_layout_count": len([item for item in micro_layouts if isinstance(item, dict)]),
        "table_package_count": len([item for item in table_packages if isinstance(item, dict)]),
        "rendered_table_count": len(rendered_tables),
        "rendered_high_value_table_count": len(rendered_high_value_tables),
        "argument_unit_count": len([item for item in argument_units if isinstance(item, dict)]),
        "public_argument_unit_count": len(public_units),
        "source_count": len([item for item in source_registry if isinstance(item, dict)]),
        "graph_node_count": _safe_len(evidence_graph.get("nodes")),
        "graph_edge_count": _safe_len(evidence_graph.get("edges")),
        "qa_passed": bool(qa_result.get("passed")),
        "quality_score": qa_result.get("quality_score"),
        "package_passed": bool(package_quality_report.get("passed")),
        "package_warning_count": len(_as_list(package_quality_report.get("warnings"))),
    }


def _table_planning_summary(micro_layouts: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    chapters_with_table: List[Dict[str, Any]] = []
    chapters_without_table: List[Dict[str, Any]] = []
    request_count = 0
    why_no_table_samples: List[Dict[str, Any]] = []
    for layout in micro_layouts:
        if not isinstance(layout, dict):
            continue
        chapter_id = str(layout.get("chapter_id") or "").strip()
        plan = _as_dict(layout.get("table_planning"))
        requests = [request for request in _as_list(layout.get("table_requests")) if isinstance(request, dict) and request.get("need_table") is not False]
        request_count += len(requests)
        if requests or plan.get("need_table"):
            chapters_with_table.append(
                {
                    "chapter_id": chapter_id,
                    "table_type": plan.get("table_type") or (requests[0].get("table_type") if requests else ""),
                    "placement_slot": plan.get("placement_slot") or (requests[0].get("placement_slot") if requests else ""),
                    "reason": plan.get("why_table_needed") or (requests[0].get("why_table_needed") if requests else ""),
                }
            )
        else:
            reason = plan.get("why_no_table") or "No table request was planned for this chapter."
            chapters_without_table.append({"chapter_id": chapter_id, "reason": reason})
            if len(why_no_table_samples) < 6:
                why_no_table_samples.append({"chapter_id": chapter_id, "reason": reason})
    return {
        "chapter_count": len([layout for layout in micro_layouts if isinstance(layout, dict)]),
        "chapters_with_table": chapters_with_table,
        "chapters_without_table": chapters_without_table,
        "llm_table_request_count": request_count,
        "why_no_table_samples": why_no_table_samples,
    }


def _table_quality_summary(table_packages: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    candidate_count = 0
    validated_count = 0
    rendered_count = 0
    repaired_count = 0
    demoted_count = 0
    drop_count = 0
    calculation_error_count = 0
    missing_claim_binding_count = 0
    type_distribution: Dict[str, int] = {}
    render_tier_distribution: Dict[str, int] = {}
    reject_reason_distribution: Dict[str, int] = {}
    demoted_samples: List[Dict[str, Any]] = []
    for table in table_packages:
        if not isinstance(table, dict):
            continue
        candidate_count += 1
        table_type = str(table.get("table_type") or "unknown")
        type_distribution[table_type] = type_distribution.get(table_type, 0) + 1
        validation = _as_dict(table.get("validation") or table.get("table_validation_for_clean"))
        errors = _as_list(validation.get("errors")) + _as_list(table.get("validation_errors"))
        warnings = _as_list(validation.get("warnings"))
        reject_reasons = [str(reason or "").strip() for reason in _as_list(table.get("reject_reasons")) if str(reason or "").strip()]
        for reason in reject_reasons:
            reject_reason_distribution[reason] = reject_reason_distribution.get(reason, 0) + 1
        if table.get("should_render") and not table.get("appendix_only"):
            render_tier = "body_high_value" if str(table.get("table_value_tier") or "") == "high" else "body"
        elif table.get("appendix_only"):
            render_tier = "appendix"
        else:
            render_tier = "drop"
            drop_count += 1
        render_tier_distribution[render_tier] = render_tier_distribution.get(render_tier, 0) + 1
        if validation:
            validated_count += 1
        if table.get("should_render") and not table.get("appendix_only"):
            rendered_count += 1
        else:
            demoted_count += 1
            if len(demoted_samples) < 8:
                demoted_samples.append(
                    {
                        "table_id": table.get("table_id"),
                        "chapter_id": table.get("chapter_id"),
                        "table_type": table_type,
                        "reject_reasons": _as_list(table.get("reject_reasons")),
                    }
                )
        if table.get("repaired") or table.get("repair_applied"):
            repaired_count += 1
        if any(str(_as_dict(error).get("type") or "").startswith("metric_") for error in errors):
            calculation_error_count += 1
        if any(_as_dict(item).get("type") == "missing_table_anchor" for item in [*errors, *warnings]):
            missing_claim_binding_count += 1
    return {
        "candidate_count": candidate_count,
        "validated_count": validated_count,
        "rendered_count": rendered_count,
        "repaired_count": repaired_count,
        "demoted_count": demoted_count,
        "drop_count": drop_count,
        "calculation_error_count": calculation_error_count,
        "missing_claim_binding_count": missing_claim_binding_count,
        "table_type_distribution": type_distribution,
        "render_tier_distribution": render_tier_distribution,
        "reject_reason_distribution": reject_reason_distribution,
        "demoted_samples": demoted_samples,
    }


def _table_placement_summary(table_packages: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    placement_by_slot: Dict[str, int] = {}
    low_confidence: List[Dict[str, Any]] = []
    placed = 0
    rendered = 0
    fallback_count = 0
    for table in table_packages:
        if not isinstance(table, dict) or not table.get("should_render") or table.get("appendix_only"):
            continue
        rendered += 1
        slot = str(table.get("placement_slot") or "chapter_end")
        placement_by_slot[slot] = placement_by_slot.get(slot, 0) + 1
        if slot == "chapter_end":
            fallback_count += 1
        if table.get("anchor_section_id") or table.get("anchor_block_type"):
            placed += 1
        elif len(low_confidence) < 8:
            low_confidence.append(
                {
                    "table_id": table.get("table_id"),
                    "chapter_id": table.get("chapter_id"),
                    "placement_slot": slot,
                    "reason": "missing_anchor_section_or_block",
                }
            )
    return {
        "placed_table_count": placed,
        "unplaced_table_count": max(0, rendered - placed),
        "chapter_end_fallback_count": fallback_count,
        "placement_by_slot": placement_by_slot,
        "low_confidence_placements": low_confidence,
    }


def _table_gap_summary(table_packages: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    requirement_count = 0
    follow_up_count = 0
    missing_fields: Dict[str, int] = {}
    samples: List[Dict[str, Any]] = []
    rendered_high_value_table_count = 0
    for table in table_packages:
        if not isinstance(table, dict):
            continue
        if table.get("should_render") and not table.get("appendix_only") and str(table.get("table_value_tier") or "") == "high":
            rendered_high_value_table_count += 1
        requirements = [_as_dict(item) for item in _as_list(table.get("table_evidence_requirements")) if isinstance(item, dict)]
        followups = [_as_dict(item) for item in _as_list(table.get("table_follow_up_queries")) if isinstance(item, dict)]
        requirement_count += len(requirements)
        follow_up_count += len(followups)
        for requirement in requirements:
            for field in _as_list(requirement.get("missing_fields")):
                key = str(field or "").strip()
                if key:
                    missing_fields[key] = missing_fields.get(key, 0) + 1
            if len(samples) < 8:
                samples.append(
                    {
                        "table_id": requirement.get("table_id") or table.get("table_id"),
                        "chapter_id": requirement.get("chapter_id") or table.get("chapter_id"),
                        "table_type": requirement.get("table_type") or table.get("table_type"),
                        "missing_fields": _as_list(requirement.get("missing_fields")),
                        "query": requirement.get("query"),
                    }
                )
    return {
        "table_evidence_requirement_count": requirement_count,
        "table_follow_up_count": follow_up_count,
        "rendered_high_value_table_count": rendered_high_value_table_count,
        "missing_field_distribution": missing_fields,
        "samples": samples,
    }


def _stage_quality_card(
    *,
    chapter_evidence_packages: Sequence[Dict[str, Any]],
    table_quality_summary: Dict[str, Any],
    table_gap_summary: Dict[str, Any],
    analysis_stage_diagnostics: Dict[str, Any],
    final_citation_audit: Dict[str, Any],
) -> Dict[str, Any]:
    evidence_totals: Dict[str, int] = {
        "chapter_count": 0,
        "candidate_fact_count": 0,
        "eligible_fact_count": 0,
        "filtered_fact_count": 0,
        "resolved_diagnostic_ref_count": 0,
        "unresolved_ref_count": 0,
        "relevance_rejected_count": 0,
        "matched_after_relevance_count": 0,
        "hydrated_evidence_count": 0,
        "empty_chapter_count": 0,
    }
    layer_counts: Dict[str, int] = {}
    weakest_chapters: List[Dict[str, Any]] = []
    for package in chapter_evidence_packages:
        if not isinstance(package, dict):
            continue
        evidence_totals["chapter_count"] += 1
        funnel = _as_dict(package.get("evidence_binding_funnel") or _as_dict(package.get("metadata")).get("evidence_binding_funnel"))
        if not funnel:
            funnel = {
                "hydrated_evidence_count": _count_value(package.get("hydrated_evidence_count")),
                "unresolved_ref_count": _count_value(package.get("unresolved_evidence_ref_count")),
                "matched_after_relevance_count": _count_value(package.get("matched_evidence_count")),
                "eligible_fact_count": _count_value(package.get("writable_fact_count")),
                "layer_counts": _as_dict(package.get("evidence_binding_counts")),
            }
        for key in (
            "candidate_fact_count",
            "eligible_fact_count",
            "filtered_fact_count",
            "resolved_diagnostic_ref_count",
            "unresolved_ref_count",
            "relevance_rejected_count",
            "matched_after_relevance_count",
            "hydrated_evidence_count",
        ):
            evidence_totals[key] += _count_value(funnel.get(key))
        if _count_value(funnel.get("hydrated_evidence_count")) <= 0:
            evidence_totals["empty_chapter_count"] += 1
            if len(weakest_chapters) < 8:
                weakest_chapters.append(
                    {
                        "chapter_id": package.get("chapter_id"),
                        "chapter_title": package.get("chapter_title"),
                        "reason": "no_hydrated_evidence",
                        "unresolved_ref_count": _count_value(funnel.get("unresolved_ref_count")),
                        "relevance_rejected_count": _count_value(funnel.get("relevance_rejected_count")),
                    }
                )
        for layer, count in _as_dict(funnel.get("layer_counts")).items():
            key = str(layer or "").strip()
            if key:
                layer_counts[key] = layer_counts.get(key, 0) + _count_value(count)

    table_summary = _as_dict(table_quality_summary)
    table_gap = _as_dict(table_gap_summary)
    analysis = _as_dict(analysis_stage_diagnostics)
    citation = _as_dict(final_citation_audit)
    citationless_removed_count = (
        _count_value(citation.get("citationless_factual_removed_count"))
        or _count_value(citation.get("citationless_factual_sentence_removed_count"))
        + _count_value(citation.get("citationless_factual_bullet_removed_count"))
        + _count_value(citation.get("citationless_short_factual_line_removed_count"))
    )
    top_blockers: List[str] = []
    if evidence_totals["empty_chapter_count"]:
        top_blockers.append("chapter_evidence_empty")
    if evidence_totals["unresolved_ref_count"]:
        top_blockers.append("unresolved_evidence_refs")
    if evidence_totals["relevance_rejected_count"]:
        top_blockers.append("chapter_relevance_rejections")
    if _count_value(table_summary.get("drop_count")):
        top_blockers.append("table_drop")
    if _as_dict(table_gap.get("missing_field_distribution")):
        top_blockers.append("table_metric_fields_missing")
    if _count_value(analysis.get("llm_failed_chapter_count") or analysis.get("failed_chapter_count")):
        top_blockers.append("analysis_partial_failure")
    if bool(citation.get("citation_rebind_required")):
        top_blockers.append("citation_rebind_required")
    if str(citation.get("final_citation_reconciliation_status") or "").strip().lower() not in {"", "ok"}:
        top_blockers.append("final_citation_not_ok")

    status = "green"
    if top_blockers:
        status = "yellow"
    if bool(citation.get("citation_rebind_required")) or "final_citation_not_ok" in top_blockers:
        status = "red"
    return {
        "schema_version": "stage_quality_card_v1",
        "status": status,
        "top_blockers": _dedupe(top_blockers),
        "evidence_binding": {
            **evidence_totals,
            "layer_counts": layer_counts,
            "weakest_chapters": weakest_chapters,
        },
        "table": {
            "candidate_count": _count_value(table_summary.get("candidate_count")),
            "rendered_count": _count_value(table_summary.get("rendered_count")),
            "rendered_high_value_table_count": _count_value(table_gap.get("rendered_high_value_table_count")),
            "drop_count": _count_value(table_summary.get("drop_count")),
            "demoted_count": _count_value(table_summary.get("demoted_count")),
            "render_tier_distribution": _as_dict(table_summary.get("render_tier_distribution")),
            "reject_reason_distribution": _as_dict(table_summary.get("reject_reason_distribution")),
            "missing_field_distribution": _as_dict(table_gap.get("missing_field_distribution")),
            "table_follow_up_count": _count_value(table_gap.get("table_follow_up_count")),
        },
        "analysis": {
            "final_analysis_source": analysis.get("final_analysis_source"),
            "usable_claim_count": _count_value(
                analysis.get("llm_usable_claim_count")
                or analysis.get("usable_claim_count")
                or analysis.get("output_claim_count")
            ),
            "failed_chapter_count": _count_value(analysis.get("llm_failed_chapter_count") or analysis.get("failed_chapter_count")),
            "semantic_judge_counts": _as_dict(analysis.get("llm_semantic_judge_counts")),
            "validation_issue_counts": _as_dict(analysis.get("llm_validation_issue_counts")),
        },
        "citation": {
            "final_citation_reconciliation_status": citation.get("final_citation_reconciliation_status"),
            "citation_rebind_required": bool(citation.get("citation_rebind_required")),
            "citationless_factual_removed_count": citationless_removed_count,
            "factual_body_without_citations_count": _count_value(citation.get("factual_body_without_citations_count")),
            "final_unresolved_citation_removed_count": _count_value(citation.get("final_unresolved_citation_removed_count")),
        },
    }


def _table_follow_up_queries(table_packages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    followups: List[Dict[str, Any]] = []
    seen = set()
    for table in table_packages:
        if not isinstance(table, dict):
            continue
        for item in _as_list(table.get("table_follow_up_queries")):
            if not isinstance(item, dict):
                continue
            query = _compact(item.get("query"), 240)
            key = (query, str(item.get("table_id") or ""), str(item.get("chapter_id") or ""))
            if not query or key in seen:
                continue
            seen.add(key)
            followups.append({**item, "query": query})
    return followups


def _normalize_followup_query_item(item: Any) -> Dict[str, Any]:
    if isinstance(item, dict):
        query = _compact(item.get("query") or item.get("suggested_query"), 240)
        return {**item, "query": query} if query else dict(item)
    query = _compact(item, 240)
    return {"query": query} if query else {}


def _merge_table_followups_into_refinement_plan(plan: Dict[str, Any], table_followups: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    merged = copy.deepcopy(_as_dict(plan))
    existing = [
        normalized
        for item in _as_list(merged.get("follow_up_queries"))
        for normalized in [_normalize_followup_query_item(item)]
        if normalized
    ]
    seen = {
        (
            _compact(item.get("query"), 240),
            str(item.get("table_id") or ""),
            str(item.get("chapter_id") or ""),
        )
        for item in existing
    }
    for item in table_followups:
        if not isinstance(item, dict):
            continue
        query = _compact(item.get("query"), 240)
        key = (query, str(item.get("table_id") or ""), str(item.get("chapter_id") or ""))
        if not query or key in seen:
            continue
        seen.add(key)
        existing.append(dict(item))
    merged["follow_up_queries"] = existing
    merged["table_follow_up_count"] = len(table_followups)
    return merged


def _table_appendix_rows(table_packages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    appendix_tables: List[Dict[str, Any]] = []
    for table in table_packages:
        if not isinstance(table, dict):
            continue
        rows = [
            _as_list(_as_dict(row).get("cells"))
            for row in _as_list(table.get("appendix_rows"))
            if isinstance(row, dict)
        ]
        if not rows:
            continue
        appendix_tables.append(
            {
                "table_id": table.get("table_id"),
                "chapter_id": table.get("chapter_id"),
                "title": table.get("title"),
                "headers": _as_list(table.get("headers")),
                "rows": rows,
                "evidence_refs": _as_list(table.get("evidence_refs")),
            }
        )
    return appendix_tables


def _compact_graph(graph: Dict[str, Any]) -> Dict[str, Any]:
    metadata = _as_dict(graph.get("metadata"))
    return {
        "node_count": _safe_len(graph.get("nodes")),
        "edge_count": _safe_len(graph.get("edges")),
        "conflict_count": _safe_len(graph.get("conflicts")),
        "metadata": _compact_mapping(metadata, list(metadata.keys()), text_chars=120),
    }


def _compact_debug_or_quality(value: Dict[str, Any]) -> Dict[str, Any]:
    return _compact_mapping(value, list(value.keys()), text_chars=220)


def _pipeline_artifacts_payload(
    *,
    chapter_evidence_packages: Sequence[Dict[str, Any]],
    source_quality_map: Dict[str, Any],
    metric_normalization_table: Sequence[Dict[str, Any]],
    coverage_matrix: Sequence[Dict[str, Any]],
    missing_proof_standards: Sequence[Any],
    evidence_graph: Dict[str, Any],
    analytics_outputs: Sequence[Dict[str, Any]],
    micro_layouts: Sequence[Dict[str, Any]],
    table_packages: Sequence[Dict[str, Any]],
    argument_units: Sequence[Dict[str, Any]],
    chapter_packages: Sequence[Dict[str, Any]],
    decision_package: Dict[str, Any],
    risk_package: Dict[str, Any],
    appendix_package: Dict[str, Any],
    source_registry: Sequence[Dict[str, Any]],
    qa_result: Dict[str, Any],
    package_quality_report: Dict[str, Any],
) -> Dict[str, Any]:
    full = _include_full_pipeline_artifacts()
    summary = _pipeline_artifact_summary(
        chapter_evidence_packages=chapter_evidence_packages,
        evidence_graph=evidence_graph,
        analytics_outputs=analytics_outputs,
        micro_layouts=micro_layouts,
        table_packages=table_packages,
        argument_units=argument_units,
        chapter_packages=chapter_packages,
        source_registry=source_registry,
        qa_result=qa_result,
        package_quality_report=package_quality_report,
    )
    if full:
        return {
            "pipeline_payload_mode": "full",
            "pipeline_artifact_summary": summary,
            "chapter_evidence_packages": list(chapter_evidence_packages or []),
            "source_quality_map": source_quality_map,
            "metric_normalization_table": list(metric_normalization_table or []),
            "coverage_matrix": list(coverage_matrix or []),
            "missing_proof_standards": list(missing_proof_standards or []),
            "evidence_graph": evidence_graph,
            "analytics_outputs": list(analytics_outputs or []),
            "micro_layouts": list(micro_layouts or []),
            "table_packages": list(table_packages or []),
            "argument_units": list(argument_units or []),
            "chapter_packages": list(chapter_packages or []),
            "decision_package": decision_package,
            "risk_package": risk_package,
            "appendix_package": appendix_package,
        }
    return {
        "pipeline_payload_mode": "summary",
        "pipeline_artifact_summary": summary,
        "chapter_evidence_packages": _compact_sequence(chapter_evidence_packages, _compact_evidence_package, limit=12),
        "source_quality_map": {
            "source_count": len(source_quality_map),
            "levels": _compact_mapping(source_quality_map, list(source_quality_map.keys())[:20], text_chars=80),
        },
        "metric_normalization_table": _compact_sequence(metric_normalization_table, lambda item: _compact_mapping(item, list(item.keys()), text_chars=120), limit=25),
        "coverage_matrix": _compact_sequence(coverage_matrix, lambda item: _compact_mapping(item, list(item.keys()), text_chars=120), limit=25),
        "missing_proof_standards": [_compact(item, 180) for item in list(missing_proof_standards or [])[:20]],
        "evidence_graph": _compact_graph(evidence_graph),
        "analytics_outputs": _compact_sequence(analytics_outputs, _compact_analytics_output, limit=12),
        "micro_layouts": _compact_sequence(micro_layouts, _compact_micro_layout, limit=12),
        "table_packages": _compact_sequence(table_packages, _compact_table_package, limit=20),
        "argument_units": _compact_sequence(argument_units, _compact_argument_unit, limit=8),
        "chapter_packages": _compact_sequence(chapter_packages, _compact_chapter_package, limit=12),
        "decision_package": _compact_debug_or_quality(decision_package),
        "risk_package": _compact_debug_or_quality(risk_package),
        "appendix_package": {
            "metric_normalization_count": _safe_len(appendix_package.get("metric_normalization_table")),
            "coverage_matrix_count": _safe_len(appendix_package.get("coverage_matrix")),
            "missing_proof_count": _safe_len(appendix_package.get("missing_proof_standards")),
            "analytics_output_count": _safe_len(appendix_package.get("analytics_outputs")),
        },
    }


def _body_char_count(markdown: str) -> int:
    body = re.split(r"\n##\s*(?:\u9644\u5f55|附錄|研究口径|研究口徑|附录)", str(markdown or ""), maxsplit=1)[0]
    return len(body)


def _section_text_chars(section: Dict[str, Any]) -> int:
    return sum(
        len(str(section.get(key) or ""))
        for key in ["section_title", "claim", "reasoning", "mechanism", "counter_evidence", "actionable", "decision_implication"]
    )


BAD_EXPANSION_FACT_PATTERNS = [
    r"联网分析\s*Agent\s*失败",
    r"Retrieval\.TestUserQueryExceeded",
    r"Retrieval\.",
    r"TestUserQueryExceeded",
    r"query\s+exceed(?:ed)?\s+the\s+limit",
    r"目前更像局部信号",
    r"证据链如何支撑",
    r"把单点资料转化为可以和其他章节互相校准",
]


HARD_DELIVERY_FORBIDDEN_PATTERNS = [
    r"联网分析\s*Agent\s*失败",
    r"Retrieval\.",
    r"TestUserQueryExceeded",
    r"query\s+exceed(?:ed)?\s+the\s+limit",
    r"(?<![A-Za-z0-9_])EV-\d+(?:-[A-Za-z0-9]+)?",
    r"\bevidence_cards?\b",
    r"(?<![A-Za-z0-9_])ch_\d{1,3}(?![A-Za-z0-9_])",
    r"当前卡片",
    r"本章应写成",
    r"本章可以作为",
    r"本章可作为",
    r"建议写成",
    r"适合写成",
    r"暂无可核验数据",
    r"本维度暂无可核验数据",
    r"建议后续补充调研",
    r"\binsufficient_ab_sources\b",
    r"\bmetric_evidence_missing\b",
    r"\bcase_evidence_missing\b",
    r"\bcounter_evidence_missing\b",
]

SOFT_DELIVERY_GAP_PATTERNS = {
    r"\binsufficient_ab_sources\b",
    r"\bmetric_evidence_missing\b",
    r"\bcase_evidence_missing\b",
    r"\bcounter_evidence_missing\b",
}

GENERIC_REPORT_FAMILIES = {
    "dynamic_research",
    "dynamic_research_report",
    "topic_report",
    "general_report",
    "research_report",
}


def _deep_report_family(value: Any) -> bool:
    family = str(value or "").strip().lower()
    return family == "industry_deep_report" or "deep" in family


def _report_family_from(payload: Dict[str, Any]) -> str:
    payload = _as_dict(payload)
    return str(payload.get("report_family") or payload.get("report_type") or payload.get("type") or "").strip()


def _report_family_delivery_blockers(
    *,
    research_plan: Dict[str, Any],
    report_blueprint: Dict[str, Any],
    report_plan: Dict[str, Any],
) -> List[Dict[str, Any]]:
    values = {
        "research_plan": _report_family_from(research_plan),
        "report_blueprint": _report_family_from(report_blueprint),
        "report_plan": _report_family_from(report_plan),
    }
    normalized = {
        key: value.strip().lower()
        for key, value in values.items()
        if value and value.strip().lower() not in GENERIC_REPORT_FAMILIES
    }
    distinct = sorted(set(normalized.values()))
    if len(distinct) <= 1:
        return []
    return [{"type": "report_family_mismatch", "families": values}]


def _layout_delivery_blockers(report_blueprint: Dict[str, Any]) -> List[Dict[str, Any]]:
    layout_validation = _as_dict(_as_dict(report_blueprint).get("layout_validation"))
    try:
        blocking_count = int(float(layout_validation.get("blocking_count") or 0))
    except (TypeError, ValueError):
        blocking_count = 0
    blocking_issues = _as_list(layout_validation.get("blocking_issues")) or _as_list(layout_validation.get("errors"))
    if blocking_count <= 0 and not blocking_issues:
        return []
    return [
        {
            "type": "layout_validation_failed",
            "blocking_count": blocking_count or len(blocking_issues),
            "issues": blocking_issues[:8],
        }
    ]

GAP_LABELS = {
    "insufficient_ab_sources": "高等级来源数量未达标",
    "insufficient_ab_core_sources": "核心高等级来源数量未达标",
    "metric_evidence_missing": "关键指标口径缺失",
    "case_evidence_missing": "案例或主体样本缺失",
    "counter_evidence_missing": "反向证据缺失",
    "source_diversity_missing": "来源类型不够多元",
    "only_c_or_lower_sources": "只有 C 级或更低来源",
    "public_chapter_without_ab_sources": "正文章节缺少高等级来源",
    "no_ab_sources_for_core_hypotheses": "核心假设缺少高等级来源",
    "missing_proof_standards": "核心证明标准未达成",
    "mechanism_explanation_weak": "影响路径解释未达标",
    "counter_evidence_weak": "反向条件覆盖未达标",
    "report_body_below_target_chars": "正文篇幅未达到目标",
    "search_tasks_dropped": "检索任务被截断",
    "iqs_lane_no_success": "检索通道没有成功结果",
    "low_ab_core_coverage": "核心证据高等级覆盖未达标",
}


def _is_bad_expansion_fact(value: Any) -> bool:
    text = str(value or "")
    if not text:
        return True
    return any(re.search(pattern, text, flags=re.I) for pattern in BAD_EXPANSION_FACT_PATTERNS)


def _delivery_gate_mode() -> str:
    raw = os.getenv("REPORT_DELIVERY_GATE_MODE") or "balanced"
    mode = str(raw).strip().lower()
    if mode in {"speed", "fast", "loose", "draft", "balanced", "quick_market_scan"}:
        return "balanced"
    if mode in {"strict", "deep_strict", "due_diligence", "investment_due_diligence"}:
        return "strict"
    return mode or "balanced"


def _strict_delivery_gate() -> bool:
    return _delivery_gate_mode() == "strict"


def _coverage_has_usable_signal(coverage_rows: Sequence[Dict[str, Any]]) -> bool:
    for item in list(coverage_rows or []):
        if not isinstance(item, dict):
            continue
        if bool(item.get("decision_ready")):
            return True
        if str(item.get("claim_status") or "").strip().lower() in {"directional", "directional_ready"}:
            return True
        if int(item.get("actual_ab_sources") or 0) > 0:
            return True
        if int(item.get("directional_c_sources") or 0) > 0:
            return True
        if _as_list(item.get("evidence_refs")):
            return True
    return False


def _estimated_public_body_chars(chapter_packages: Sequence[Dict[str, Any]]) -> int:
    total = 0
    for chapter in list(chapter_packages or []):
        if not isinstance(chapter, dict) or chapter.get("omit_from_report"):
            continue
        total += len(str(chapter.get("chapter_title") or "")) + len(str(chapter.get("lead") or ""))
        for section in _as_list(chapter.get("sections")):
            if isinstance(section, dict) and not section.get("omit_from_report"):
                total += _section_text_chars(section)
        for fact in _as_list(chapter.get("chapter_fact_digest")):
            total += len(str(fact or ""))
    return total


def _chapter_evidence_refs(chapter: Dict[str, Any], *, limit: int = 10) -> List[str]:
    refs: List[Any] = []
    for section in _as_list(chapter.get("sections")):
        if isinstance(section, dict):
            refs.extend(_as_list(section.get("evidence_refs")))
    return _dedupe(refs, limit=limit)


def _chapter_facts(chapter: Dict[str, Any], *, limit: int = 12) -> List[str]:
    facts = [fact for fact in _as_list(chapter.get("chapter_fact_digest")) if not _is_bad_expansion_fact(fact)]
    for section in _as_list(chapter.get("sections")):
        if not isinstance(section, dict):
            continue
        for fact in _as_list(section.get("supporting_facts")):
            if not _is_bad_expansion_fact(fact):
                facts.append(fact)
    return _dedupe(facts, limit=limit)


def _expansion_section(chapter: Dict[str, Any], *, kind: str, index: int) -> Dict[str, Any]:
    title = _compact(chapter.get("chapter_title") or "本章", 120)
    facts = _chapter_facts(chapter, limit=12)
    refs = _chapter_evidence_refs(chapter, limit=10)
    fact_text = "；".join(facts[:8]) or "现有材料已经提供了若干可观察信号，但仍需要放在同一条判断链中解释"
    if kind == "evidence_chain":
        section_title = "关键事实与判断依据"
        claim = f"{title}的判断应同时落在时间窗口、指标口径和主体行为上，避免把单点新闻外推为行业趋势。"
        reasoning = (
            f"本章可用事实主要包括：{fact_text}。这些事实的价值不在于数量，而在于它们能否互相校准：如果政策、企业行为、市场指标和案例方向一致，结论强度会上升；"
            f"如果只有单一来源或单一口径支持，则更适合把结论写成阶段性判断。围绕“{title}”，正文需要解释事实之间的关系，而不是把证据简单堆叠。"
            f"第一层要看事实是否覆盖了核心主体，第二层要看指标是否有时间和范围，第三层要看是否存在能推翻判断的反向样本。"
        )
        counter = "如果新增信号只来自同类来源，或关键指标缺少时间、范围和主体口径，本章判断应保持边界，不能扩大到全行业结论。"
        actionable = "更有价值的观察顺序是：同口径指标是否连续、企业披露是否兑现、官方或协会数据是否印证，以及是否出现足以削弱主线的反向案例。"
    elif kind == "mechanism_boundary":
        section_title = "机制传导、约束条件与边界"
        claim = f"{title}的核心不是静态描述，而是判断变量如何传导、在哪些条件下会失效。"
        reasoning = (
            f"从机制上看，本章事实可以拆成输入变量、传导环节和结果变量三层：输入变量决定约束来源，传导环节决定变化速度，结果变量决定机会能否兑现。"
            f"目前材料中的关键线索是：{fact_text}。如果这些线索能沿着同一方向传导，章节结论就更稳；如果某一层出现断点，例如执行滞后、客户验证不足、成本压力上升或外部规则变化，结论就需要收缩。"
            f"因此，本章正文应同时写清楚“为什么成立”和“什么情况下不成立”，这样才能避免把短期信号误写成长期趋势。"
        )
        counter = "边界主要来自三类变化：一是政策或外部环境改变，二是供需和价格指标反向，三是企业或客户行为没有跟随。任一变量反向，都会降低判断强度。"
        actionable = "跟踪时应把关键变量拆成政策/规则、供给/产能、需求/客户、价格/利润和反证样本五组，并按月或按季度更新判断。"
    elif kind == "opportunity_risk":
        section_title = "机会兑现路径与风险映射"
        claim = f"{title}的机会只在证据能够落到具体环节、具体主体和具体约束时才有决策价值。"
        reasoning = (
            f"机会兑现路径通常要经过三个步骤：先确认需求或约束真实存在，再确认哪些主体能够承接，最后判断利润、现金流或战略价值能否留下来。"
            f"本章目前能够利用的事实包括：{fact_text}。这些事实可以帮助区分“确定性较高的环节”和“仍需观察的线索”。"
            f"如果一个环节同时具备多来源验证、客户或政策牵引、可跟踪指标和较少反向样本，它更适合进入正文主线；如果只具备话题热度或个别案例，则应放在风险和观察指标里。"
        )
        counter = "最大风险是把局部案例外推为全局机会，或者忽略产能、价格、客户认证、政策执行和竞争加剧带来的反向压力。"
        actionable = "正文结论应给出优先级：先看高确定性环节，再看需要补证的中性线索，最后列出必须放弃或降级判断的触发条件。"
    elif kind == "verification":
        section_title = "后续验证指标与结论更新方式"
        claim = f"{title}需要被持续验证，而不是在一次报告中固定为不变结论。"
        reasoning = (
            f"本章后续验证应围绕可量化指标、可核验案例和反向样本展开。现有事实基础是：{fact_text}。"
            f"这些事实已经能支撑一个初步判断，但要让报告达到行研深度，还需要说明哪些指标会提高置信度、哪些指标会降低置信度。"
            f"有效的更新方式不是简单追加新闻，而是把新证据放回原有变量链，观察它改变的是需求、供给、竞争、政策、技术还是财务质量。"
        )
        counter = "如果后续指标无法复现、反向案例增多，或新增来源与原有判断冲突，本章应进入重新评估状态。"
        actionable = "建议建立验证清单，持续跟踪核心指标、A/B级来源、代表性企业行为和反向触发器，并在证据结构变化时更新章节结论。"
    elif kind == "data_lens":
        section_title = "指标口径与可比性"
        claim = f"{title}需要把指标放回口径中解释，尤其要区分总量、结构、增速、价格和利润质量。"
        reasoning = (
            f"行研报告最容易失真的地方，是把不同口径的指标直接并列。围绕“{title}”，现有事实包括：{fact_text}。"
            f"这些事实进入正文时，应先说明它们对应的是市场总量、企业样本、区域样本、时间窗口还是单一事件。"
            f"如果一个指标只覆盖局部企业，它更适合作为案例；如果指标来自官方、协会或可复核的企业披露，并且能和其他来源交叉验证，它才能进入核心判断。"
            f"因此，本章扩展不是为了增加篇幅，而是把数据放到可比框架里：同一指标看趋势，不同指标看传导，不同来源看一致性，反向样本看失效条件。"
        )
        counter = "如果指标口径混杂、统计范围不同或时间窗口不一致，章节应明确保留判断边界，避免把短期波动写成结构性趋势。"
        actionable = "后续补证优先选择带有范围、期间、单位和来源等级的指标，并把企业披露、行业统计和政策文件放在同一比较框架中。"
    else:
        section_title = "情景分层与决策含义"
        claim = f"{title}应当形成情景化判断，而不是只给单一结论。"
        reasoning = (
            f"在基准情景下，如果现有事实继续沿着同一方向演进，{title}可以作为正文主线的一部分；在乐观情景下，政策、需求、客户或技术变量共同改善，机会会从局部环节扩散到更多主体；"
            f"在谨慎情景下，如果反向样本增多或关键指标转弱，结论就应降级。当前可用于情景分层的事实包括：{fact_text}。"
            f"这种写法能让报告更接近真实决策过程：它不只回答“现在怎么看”，还回答“什么条件下改变看法”。"
        )
        counter = "如果未来出现政策执行弱化、客户导入放缓、价格利润恶化、技术替代不达预期或外部冲击升级，章节判断要重新排序。"
        actionable = "正文应给出基准、乐观和谨慎三类观察口径，并把每类口径对应的验证指标写清楚，方便后续滚动更新。"
    return {
        "section_id": f"{chapter.get('chapter_id') or 'chapter'}_expand_{index}",
        "section_title": section_title,
        "claim": claim,
        "reasoning": reasoning,
        "mechanism": reasoning,
        "counter_evidence": counter,
        "actionable": actionable,
        "decision_implication": actionable,
        "what_to_verify_next": [],
        "supporting_facts": facts[:8],
        "confidence": _as_dict(chapter.get("chapter_summary")).get("confidence") or "medium",
        "evidence_refs": refs,
        "render_blocks": [],
        "public_render": True,
        "expansion_generated": True,
    }


def _chapter_expandable(chapter: Dict[str, Any]) -> bool:
    if chapter.get("omit_from_report"):
        return False
    refs = _chapter_evidence_refs(chapter, limit=20)
    facts = _chapter_facts(chapter, limit=12)
    if len(refs) < _env_int("REPORT_MIN_EXPANSION_EVIDENCE_REFS", 2, min_value=0, max_value=20):
        return False
    if len(facts) < _env_int("REPORT_MIN_EXPANSION_FACTS", 2, min_value=0, max_value=20):
        return False
    quality = _as_dict(chapter.get("evidence_quality_summary"))
    if quality:
        core_count = int(quality.get("core_evidence_count") or 0)
        core_ab_count = int(quality.get("core_ab_source_count") or 0)
        supporting_count = int(quality.get("supporting_evidence_count") or 0)
        table_count = int(quality.get("table_evidence_count") or 0)
        sample_count = len(_as_list(chapter.get("sample_evidence")))
        level_distribution = _as_dict(quality.get("source_level_distribution"))
        ab_count = core_ab_count
        for level in ("A", "B"):
            try:
                ab_count += int(level_distribution.get(level) or 0)
            except (TypeError, ValueError):
                continue
        usable_count = core_count + supporting_count + table_count + sample_count
        min_core = _env_int("REPORT_MIN_EXPANSION_CORE_EVIDENCE", 1, min_value=0, max_value=20)
        min_ab = _env_int("REPORT_MIN_EXPANSION_AB_CORE", 1, min_value=0, max_value=20)
        if core_count < min_core and ab_count < min_ab and usable_count < max(2, min_core):
            return False
    missing_types = {
        str(item.get("type") or "")
        for item in _as_list(chapter.get("evidence_gaps"))
        if isinstance(item, dict)
    }
    if "insufficient_core_evidence" in missing_types and not quality.get("core_ab_source_count"):
        return False
    return True


def _gap_label(value: Any) -> str:
    text = str(value or "").strip()
    return GAP_LABELS.get(text, text.replace("_", " ") if text else "未命名缺口")


def _coverage_gap_rows(coverage_matrix: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in list(coverage_matrix or []):
        if not isinstance(item, dict):
            continue
        gaps = _as_list(item.get("blocking_gaps"))
        if gaps or not bool(item.get("decision_ready")):
            rows.append(item)
    return rows


def _hard_delivery_blockers(
    *,
    markdown: str,
    qa_result: Dict[str, Any],
    package_quality_report: Dict[str, Any],
    coverage_matrix: Sequence[Dict[str, Any]],
    delivery_gate: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    blockers: List[Dict[str, Any]] = []
    text = str(markdown or "")
    strict_gate = _strict_delivery_gate()
    gate = _as_dict(delivery_gate)
    if gate.get("diagnostic_only") or str(gate.get("tier") or "").strip() == "diagnostic_only":
        blockers.append(
            {
                "type": "diagnostic_only",
                "detail": gate,
                "count": len(_as_list(gate.get("blocking_reasons"))),
            }
        )
    for pattern in HARD_DELIVERY_FORBIDDEN_PATTERNS:
        if re.search(pattern, text, flags=re.I):
            if not strict_gate and pattern in SOFT_DELIVERY_GAP_PATTERNS:
                continue
            blockers.append({"type": "forbidden_public_text", "pattern": pattern})

    coverage_rows = [item for item in list(coverage_matrix or []) if isinstance(item, dict)]
    gap_rows = _coverage_gap_rows(coverage_rows)
    if coverage_rows and not any(bool(item.get("decision_ready")) for item in coverage_rows):
        if strict_gate:
            blockers.append({"type": "no_decision_ready_hypotheses", "count": len(coverage_rows)})
        elif not _coverage_has_usable_signal(coverage_rows):
            blockers.append({"type": "no_publishable_evidence", "count": len(coverage_rows)})

    warning_types = set(_package_warning_types(package_quality_report))
    package_errors = [
        _as_dict(item)
        for item in _as_list(package_quality_report.get("blocking_errors") or package_quality_report.get("errors"))
    ]
    empty_pipeline_errors = [
        item
        for item in package_errors
        if str(item.get("package") or "") in {"chapter_evidence_packages", "argument_units", "chapter_packages"}
        and str(item.get("type") or item.get("issue_type") or "") in {"empty", "sections_empty"}
    ]
    if empty_pipeline_errors:
        blockers.append({"type": "pipeline_empty_package", "count": len(empty_pipeline_errors), "examples": empty_pipeline_errors[:5]})
    qa_errors = [_as_dict(item) for item in _as_list(qa_result.get("errors"))]
    if any(str(item.get("type") or "") in {"chapter_packages_missing", "report_markdown_empty"} for item in qa_errors):
        blockers.append({"type": "pipeline_empty_package", "count": len(qa_errors), "examples": qa_errors[:5]})
    for item in _as_list(_as_dict(qa_result.get("render_gate")).get("blockers")):
        payload = _as_dict(item)
        if payload:
            blockers.append({"type": payload.get("type") or "render_gate_blocker", "detail": payload})

    deduped: List[Dict[str, Any]] = []
    seen = set()
    for item in blockers:
        item_type = str(item.get("type") or "")
        if item_type == "forbidden_public_text":
            key = (item_type, "", "")
        else:
            key = (item_type, str(item.get("pattern") or ""), str(item.get("count") or ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _delivery_gate_from_evidence_package(evidence_package: Dict[str, Any]) -> Dict[str, Any]:
    def as_int(value: Any) -> int:
        try:
            return int(float(value or 0))
        except (TypeError, ValueError):
            return 0

    summary = _as_dict(_as_dict(evidence_package).get("summary"))
    health = _evidence_health_summary_from_package(evidence_package)
    summary_readpage_coverage = _as_dict(summary.get("readpage_coverage"))
    summary_has_readpage = "succeeded" in summary_readpage_coverage
    health_has_readpage = "readpage_succeeded" in health
    summary_readpage_succeeded = as_int(summary_readpage_coverage.get("succeeded"))
    health_readpage_succeeded = as_int(health.get("readpage_succeeded"))
    derived_inconsistencies = list(_as_list(health.get("inconsistencies")))
    if summary_has_readpage and health_has_readpage and summary_readpage_succeeded != health_readpage_succeeded:
        derived_inconsistencies.append(
            {
                "type": "readpage_succeeded_mismatch",
                "summary_readpage_succeeded": summary_readpage_succeeded,
                "health_readpage_succeeded": health_readpage_succeeded,
            }
        )
    health_inconsistent = bool(health.get("inconsistent")) or bool(derived_inconsistencies)
    if health_inconsistent:
        return {
            "tier": "diagnostic_only",
            "publishable": False,
            "draft_allowed": False,
            "diagnostic_only": True,
            "blocking_reasons": [
                {
                    "type": "evidence_health_summary_inconsistent",
                    "details": derived_inconsistencies,
                }
            ],
            "evidence_health_summary": {**health, "inconsistencies": derived_inconsistencies},
        }
    gate = _as_dict(summary.get("delivery_gate"))
    if gate:
        gate = dict(gate)
        preflight = (
            _as_dict(_as_dict(evidence_package).get("evidence_preflight_summary"))
            or _as_dict(summary.get("evidence_preflight_summary"))
            or _as_dict(_as_dict(_as_dict(evidence_package).get("metadata")).get("evidence_preflight_summary"))
        )
        if preflight:
            gate["evidence_preflight_summary"] = preflight
            if not bool(preflight.get("ready_for_clean_writer")):
                gate["publishable"] = False
                gate["tier"] = "diagnostic_only" if bool(preflight.get("diagnostic_only")) else "limited_review_draft"
                gate["draft_allowed"] = bool(preflight.get("review_draft_allowed")) and gate["tier"] != "diagnostic_only"
                gate["diagnostic_only"] = gate["tier"] == "diagnostic_only"
                reason_key = "blocking_reasons" if gate["diagnostic_only"] else "review_reasons"
                reasons = _as_list(gate.get(reason_key))
                reasons.append({"type": "evidence_preflight_not_ready", "evidence_preflight_summary": preflight})
                gate[reason_key] = reasons
        evidence_signal = _as_dict(gate.get("evidence_signal"))
        if health_has_readpage:
            evidence_signal["readpage_succeeded"] = max(as_int(evidence_signal.get("readpage_succeeded")), health_readpage_succeeded)
        if evidence_signal:
            gate["evidence_signal"] = evidence_signal
        if _count_value(health.get("analysis_ready_ab_count")) > 0 and _count_value(health.get("traceable_ab_source_count")) <= 0:
            gate["tier"] = "limited_review_draft" if _count_value(health.get("analysis_ready_count")) > 0 else "diagnostic_only"
            gate["publishable"] = False
            gate["draft_allowed"] = gate["tier"] != "diagnostic_only"
            gate["diagnostic_only"] = gate["tier"] == "diagnostic_only"
            reasons = _as_list(gate.get("review_reasons" if gate["tier"] != "diagnostic_only" else "blocking_reasons"))
            reasons.append({"type": "traceable_ab_sources_missing", "evidence_health_summary": health})
            if gate["tier"] == "diagnostic_only":
                gate["blocking_reasons"] = reasons
            else:
                gate["review_reasons"] = reasons
            gate["evidence_health_summary"] = health
        if _count_value(health.get("analysis_ready_ab_count")) > 0 and _count_value(health.get("distinct_verified_ab_source_count")) <= 0:
            gate["publishable"] = False
            gate["draft_allowed"] = True
            gate["diagnostic_only"] = False
            gate["tier"] = "limited_review_draft"
            reasons = _as_list(gate.get("review_reasons"))
            reasons.append({"type": "verified_ab_sources_missing", "evidence_health_summary": health})
            gate["review_reasons"] = reasons
            gate["evidence_health_summary"] = health
        return gate
    publishable_gate = _as_dict(summary.get("publishable_evidence_gate"))
    if bool(publishable_gate.get("passed")):
        return {"tier": "publishable_clean", "publishable": True, "draft_allowed": True, "diagnostic_only": False}
    analysis_ready = as_int(summary.get("analysis_ready_count"))
    source_dist = _as_dict(summary.get("source_level_distribution"))
    ab_count = as_int(source_dist.get("A")) + as_int(source_dist.get("B"))
    readpage = max(summary_readpage_succeeded, health_readpage_succeeded)
    diagnostic = analysis_ready <= 0 and ab_count <= 0 and readpage <= 0
    return {
        "tier": "diagnostic_only" if diagnostic else "limited_review_draft",
        "publishable": False,
        "draft_allowed": not diagnostic,
        "diagnostic_only": diagnostic,
        "review_reasons": _as_list(publishable_gate.get("blocking_reasons")),
        "blocking_reasons": [{"type": "no_usable_evidence_signal"}] if diagnostic else [],
    }


def _evidence_limitations_from_delivery_gate(delivery_gate: Dict[str, Any], qa_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    limitations: List[Dict[str, Any]] = []
    for reason in _as_list(_as_dict(delivery_gate).get("review_reasons")) + _as_list(_as_dict(delivery_gate).get("blocking_reasons")):
        payload = _as_dict(reason)
        if payload:
            limitations.append(payload)
    for followup in _as_list(_as_dict(qa_result).get("blocking_evidence_repair_followups")) + _as_list(_as_dict(qa_result).get("evidence_repair_followups")):
        payload = _as_dict(followup)
        if payload:
            limitations.append({"type": payload.get("type") or "evidence_repair_followup", "detail": payload})
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for item in limitations:
        key = (str(item.get("type") or ""), str(item.get("chapter_id") or ""), str(item.get("hypothesis_id") or ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped[:20]


FATAL_DELIVERY_BLOCKER_TYPES = {
    "pipeline_empty_package",
}


def _fatal_delivery_blockers(blockers: Sequence[Dict[str, Any]], *, markdown: str) -> List[Dict[str, Any]]:
    fatal = [
        _as_dict(item)
        for item in list(blockers or [])
        if str(_as_dict(item).get("type") or "").strip() in FATAL_DELIVERY_BLOCKER_TYPES
    ]
    if not str(markdown or "").strip():
        fatal.append({"type": "report_markdown_empty"})
    return fatal


def _quality_findings_from_review(
    *,
    delivery_blockers: Sequence[Dict[str, Any]],
    qa_result: Dict[str, Any],
    package_quality_report: Dict[str, Any],
    delivery_gate: Dict[str, Any],
    table_gap_summary: Dict[str, Any],
) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    for item in list(delivery_blockers or []):
        payload = _as_dict(item)
        if payload:
            findings.append({"source": "delivery_gate", **payload})
    for item in _as_list(_as_dict(qa_result).get("errors")):
        payload = _as_dict(item)
        findings.append({"source": "qa", **(payload or {"type": str(item)})})
    for item in _as_list(_as_dict(qa_result).get("warnings")):
        payload = _as_dict(item)
        findings.append({"source": "qa_warning", **(payload or {"type": str(item)})})
    for item in _as_list(_as_dict(qa_result).get("quality_findings")):
        payload = _as_dict(item)
        findings.append({"source": payload.get("source") or "qa_quality", **(payload or {"type": str(item)})})
    for item in _as_list(_as_dict(package_quality_report).get("blocking_errors")) + _as_list(_as_dict(package_quality_report).get("errors")):
        payload = _as_dict(item)
        findings.append({"source": "package_quality", **(payload or {"type": str(item)})})
    for item in _as_list(_as_dict(delivery_gate).get("review_reasons")) + _as_list(_as_dict(delivery_gate).get("blocking_reasons")):
        payload = _as_dict(item)
        if payload:
            findings.append({"source": "evidence_gate", **payload})
    if _count_value(_as_dict(table_gap_summary).get("table_follow_up_count")) > 0:
        findings.append({"source": "table_quality", "type": "table_followups_pending", "detail": table_gap_summary})

    deduped: List[Dict[str, Any]] = []
    seen = set()
    for item in findings:
        item_type = str(item.get("type") or item.get("issue_type") or item.get("source") or "quality_finding")
        key = (
            item_type,
            str(item.get("chapter_id") or item.get("hypothesis_id") or item.get("path") or ""),
            str(item.get("pattern") or item.get("package") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        item["type"] = item_type
        deduped.append(item)
    return deduped[:80]


def _quality_score_from_findings(
    *,
    qa_result: Dict[str, Any],
    quality_findings: Sequence[Dict[str, Any]],
    evidence_health_summary: Dict[str, Any],
    delivery_gate: Dict[str, Any],
) -> int:
    raw_qa = _as_dict(qa_result).get("quality_score")
    try:
        score = int(float(raw_qa))
    except (TypeError, ValueError):
        score = 72
    if bool(_as_dict(qa_result).get("passed")):
        score = max(score, 78)
    if _as_dict(delivery_gate).get("publishable"):
        score = max(score, 90)

    clean_blockers = [
        _as_dict(item) for item in list(quality_findings or [])
        if str(_as_dict(item).get("finding_category") or _as_dict(item).get("qa_category") or "").strip()
        in {"clean_blocker", "render_blocker"}
    ]
    soft_findings = max(0, len(list(quality_findings or [])) - len(clean_blockers))
    score -= min(20, len(clean_blockers) * 4)
    score -= min(12, soft_findings)
    if _count_value(_as_dict(evidence_health_summary).get("distinct_verified_ab_source_count")) <= 0:
        score -= 12
    if _count_value(_as_dict(evidence_health_summary).get("readpage_succeeded")) <= 0:
        score -= 8
    if bool(_as_dict(evidence_health_summary).get("inconsistent")):
        score -= 20
    return max(0, min(100, score))


def _quality_grade(score: int) -> str:
    if score >= 90:
        return "可发布级"
    if score >= 75:
        return "高质量但需人工复核"
    if score >= 60:
        return "证据有限但可参考"
    return "强风险报告，仅供内部研判"


def _claim_strength_from_gate(delivery_gate: Dict[str, Any], evidence_health_summary: Dict[str, Any]) -> str:
    if bool(_as_dict(delivery_gate).get("publishable")):
        return "strong"
    verified_ab = _count_value(_as_dict(evidence_health_summary).get("distinct_verified_ab_source_count"))
    traceable_ab = _count_value(_as_dict(evidence_health_summary).get("traceable_ab_source_count"))
    analysis_ab = _count_value(_as_dict(evidence_health_summary).get("analysis_ready_ab_count"))
    analysis_ready = _count_value(_as_dict(evidence_health_summary).get("analysis_ready_count"))
    if verified_ab >= 2:
        return "moderate"
    if analysis_ab > 0 or analysis_ready > 0:
        return "directional"
    return "weak"


def _public_claim_strength_label(claim_strength: str) -> str:
    return {
        "strong": "强支撑判断",
        "moderate": "中等支撑判断",
        "directional": "方向性判断",
        "weak": "弱支撑判断",
    }.get(str(claim_strength or "").strip().lower(), "方向性判断")


def _downgrade_overconfident_language(markdown: str, *, claim_strength: str) -> str:
    if claim_strength in {"strong", "moderate"}:
        return markdown
    replacements = [
        ("可以确定", "现有证据初步显示"),
        ("已经证明", "现有证据显示"),
        ("确定趋势", "方向性信号"),
        ("必然", "可能"),
        ("一定会", "可能会"),
        ("判断为", "初步判断为"),
        ("核心结论是", "待验证判断是"),
    ]
    text = str(markdown or "")
    for old, new in replacements:
        text = text.replace(old, new)
    return text


PUBLIC_QUALITY_FINDING_LABELS = {
    "chapter_core_ab_below_minimum": "部分核心章节的高质量来源不足",
    "report_counter_sources_below_minimum": "报告级反向样本不足",
    "blocking_evidence_gaps": "部分论证链仍有缺口",
    "readpage_evidence_missing": "部分来源未完成正文级核验",
    "missing_sources_appendix": "来源附录不完整",
    "table_validation_error": "表格数据口径需要复核",
    "public_chapter_without_ab_sources": "部分章节缺少高质量来源支撑",
    "report_body_below_target_chars": "正文篇幅低于深度报告目标",
    "search_tasks_dropped": "部分检索任务未完成",
    "iqs_lane_no_success": "部分主检索通道没有有效结果",
    "iqs_lane_timeout_without_signal": "部分主检索通道超时",
    "page_results_zero": "部分检索通道未得到可用页面",
    "qa_not_passed": "自动质量审查未通过",
    "deep_report_blocking_gap": "深度审查发现关键缺陷",
    "package_contract": "报告结构合同存在问题",
    "table_quality": "表格质量需要复核",
    "internal_gap_language_leaked": "正文存在需清理的内部痕迹",
    "publishable_evidence_gate_failed": "可发布证据门槛未通过",
    "traceable_ab_sources_missing": "可追溯高质量来源不足",
    "evidence_health_summary_inconsistent": "证据健康摘要存在不一致",
    "forbidden_public_text": "正文存在需要清理的内部痕迹",
    "low_ab_core_coverage": "核心章节高质量来源覆盖不足",
    "core_proof_gaps": "核心证明材料仍不完整",
    "layout_validation_failed": "版式结构仍需复核",
    "missing_proof_standards": "部分证明标准未达成",
}


def _public_quality_label(value: Any) -> str:
    raw = str(value or "quality_finding").strip()
    if not raw:
        return "质量问题"
    return PUBLIC_QUALITY_FINDING_LABELS.get(raw, "质量审查发现")


def _format_quality_finding(item: Dict[str, Any]) -> str:
    payload = _as_dict(item)
    label = _public_quality_label(payload.get("type") or payload.get("issue_type") or payload.get("source"))
    parts: List[str] = []
    chapter = payload.get("chapter_title") or payload.get("chapter_id")
    if chapter:
        parts.append(f"章节：{_compact(chapter, 60)}")
    if payload.get("actual") is not None and payload.get("required") is not None:
        parts.append(f"当前 {payload.get('actual')} / 要求 {payload.get('required')}")
    chapters = _as_list(payload.get("chapters"))
    if chapters:
        parts.append(f"涉及 {len(chapters)} 个章节")
    count = payload.get("count")
    if count is not None:
        parts.append(f"数量 {count}")
    if not parts and payload.get("quality_score") is not None:
        parts.append(f"质量分 {payload.get('quality_score')}")
    suffix = "；".join(str(part) for part in parts if str(part).strip())
    return f"{label}" + (f"（{suffix}）" if suffix else "")


def _render_quality_scorecard(
    *,
    query: str,
    quality_score: int,
    clean_report_eligible: bool,
    claim_strength: str,
    evidence_health_summary: Dict[str, Any],
    quality_findings: Sequence[Dict[str, Any]],
    delivery_gate: Dict[str, Any],
    evidence_package: Dict[str, Any],
) -> str:
    if clean_report_eligible:
        clean_text = "是"
    else:
        clean_text = "否，本报告按证据强度降级生成"
    health = _as_dict(evidence_health_summary)
    source_dist = _as_dict(health.get("source_level_distribution"))
    review_reasons = _as_list(_as_dict(delivery_gate).get("review_reasons")) + _as_list(_as_dict(delivery_gate).get("blocking_reasons"))
    lines = [
        "## 报告质量评分与证据限制",
        "",
        f"- 质量总分：{quality_score}/100（{_quality_grade(quality_score)}）",
        f"- Clean 资格：{clean_text}",
        f"- 结论强度：{_public_claim_strength_label(claim_strength)}",
        (
            "- 证据体量："
            f"可进入分析材料 {_count_value(health.get('analysis_ready_count'))} 条，"
            f"清洗后事实 {_count_value(health.get('clean_fact_count'))} 条，"
            f"可追溯 A/B 来源 {_count_value(health.get('traceable_ab_source_count'))} 个，"
            f"正文级核验成功 {_count_value(health.get('readpage_succeeded'))} 次"
        ),
        (
            "- 来源等级："
            f"A={_count_value(source_dist.get('A'))}，"
            f"B={_count_value(source_dist.get('B'))}，"
            f"C={_count_value(source_dist.get('C'))}，"
            f"D={_count_value(source_dist.get('D'))}"
        ),
    ]
    if review_reasons or quality_findings:
        lines.extend(["", "### 主要未达标项"])
        for item in list(review_reasons)[:6]:
            payload = _as_dict(item)
            if payload:
                lines.append(f"- {_format_quality_finding(payload)}")
        for item in list(quality_findings)[:8]:
            payload = _as_dict(item)
            lines.append(f"- {_format_quality_finding(payload)}")
    lines.extend(
        [
            "",
            "### 使用说明",
            "本报告已经按现有可用证据生成正式分析；未达到 Clean 标准的部分已在上方列出，并在正文中按证据强度降低结论语气。B/C 级来源只用于趋势、背景和案例线索，不等同于强审计证据。",
            "",
        ]
    )
    return "\n".join(lines).strip()


def _prepend_quality_scorecard(markdown: str, scorecard: str) -> str:
    text = str(markdown or "").strip()
    if not text or "## 报告质量评分与证据限制" in text:
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("# "):
        return "\n".join([lines[0], "", scorecard, "", *lines[1:]]).strip()
    return f"{scorecard}\n\n{text}".strip()


def _render_not_ready_report(
    *,
    query: str,
    report_blueprint: Dict[str, Any],
    blockers: Sequence[Dict[str, Any]],
    qa_result: Dict[str, Any],
    coverage_matrix: Sequence[Dict[str, Any]],
    missing_proof_standards: Sequence[Dict[str, Any]],
    search_task_schedule: Dict[str, Any],
    lane_coverage: Dict[str, Any],
) -> str:
    research_object = _compact(report_blueprint.get("research_object") or query or "当前研究问题", 140)
    lines = [
        f"# 研究未完成：{research_object}",
        "",
        "本次材料没有达到正式报告交付门槛，因此不输出正文判断。当前只保留问题拆解、证据缺口和补证任务，避免把未验证材料写成深度报告。",
        "",
        "## 阻断原因",
    ]
    if blockers:
        for blocker in list(blockers)[:10]:
            blocker_type = str(blocker.get("type") or "quality_gate").strip()
            label = _gap_label(blocker_type)
            if blocker_type == "forbidden_public_text":
                label = "草稿包含内部错误或质量标记，已阻断正式报告"
            elif blocker_type == "no_decision_ready_hypotheses":
                label = "核心假设尚无可下判断项"
            elif blocker_type == "pipeline_empty_package":
                label = "证据绑定、论证单元或章节包为空，不能继续生成正式报告"
            elif blocker_type == "qa_not_passed":
                label = f"QA 未通过，质量分 {blocker.get('quality_score')}"
            lines.append(f"- {label}")
    else:
        lines.append("- 当前材料未达到正式交付门槛。")

    lines.extend(["", "## 证据缺口"])
    gap_rows = _coverage_gap_rows(list(coverage_matrix or [])) or [
        item for item in list(missing_proof_standards or []) if isinstance(item, dict)
    ]
    if gap_rows:
        for item in gap_rows[:8]:
            item = _as_dict(item)
            title = _compact(item.get("hypothesis_statement") or item.get("chapter_title") or item.get("hypothesis_id") or "待验证假设", 120)
            gaps = [_gap_label(gap) for gap in _as_list(item.get("blocking_gaps"))]
            if not gaps:
                gaps = ["尚未达到可下判断门槛"]
            lines.append(f"- {title}：{'、'.join(_dedupe(gaps, limit=5))}")
    else:
        lines.append("- 尚未形成可复核的覆盖矩阵，需要重新检索政策原文、官方/协会数据、公司公告和反向样本。")

    lines.extend(["", "## 需补证任务"])
    followups: List[str] = []
    for gap in _as_list(_as_dict(qa_result.get("deep_evaluation")).get("required_followups")):
        gap = _as_dict(gap)
        gap_type = _gap_label(gap.get("type"))
        if gap_type:
            followups.append(gap_type)
    dropped = _as_list(search_task_schedule.get("dropped_tasks"))
    if dropped:
        followups.append("重新拆分过长检索任务，避免检索查询超限")
    failed_lanes = [
        lane
        for lane, coverage in lane_coverage.items()
        if int(_as_dict(coverage).get("scheduled") or 0) and int(_as_dict(coverage).get("succeeded") or 0) == 0
    ]
    if failed_lanes:
        followups.append("重跑失败检索通道：" + "、".join(str(item) for item in failed_lanes[:4]))
    if not followups:
        followups.extend(
            [
                "补充政策原文、出口管制清单、CHIPS Act 相关执行材料",
                "补充半导体协会、海关、财报和交易所披露中的同口径指标",
                "补充设备、EDA、先进制程、封测、成熟制程和客户认证案例",
                "补充能够推翻主线判断的反向证据",
            ]
        )
    for item in _dedupe(followups, limit=8):
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## 当前可安全使用的输出",
            "可以保留本次的问题拆解、检索任务和证据缺口；正式正文、关键数据、结论卡片和建议部分需要在补证后重新生成。",
        ]
    )
    return "\n".join(lines)


def _expand_chapter_packages_for_body_target(chapter_packages: Sequence[Dict[str, Any]], *, target_chars: int) -> List[Dict[str, Any]]:
    packages = [dict(chapter) for chapter in list(chapter_packages or []) if isinstance(chapter, dict)]
    if not target_chars or not _env_flag("REPORT_ENABLE_BODY_EXPANSION", True):
        return packages
    mode = str(os.getenv("REPORT_QUALITY_MODE") or os.getenv("QUALITY_MODE") or "").strip().lower()
    if mode in {"high", "strict", "deep_strict", "due_diligence", "investment_due_diligence"} and not _env_flag(
        "REPORT_ENABLE_DETERMINISTIC_BODY_EXPANSION",
        False,
    ):
        return packages
    current = _estimated_public_body_chars(packages)
    target = int(target_chars * _env_float("REPORT_BODY_EXPANSION_TARGET_RATIO", 0.95, min_value=0.5, max_value=2.0))
    if current >= target:
        return packages
    expandable = [chapter for chapter in packages if _chapter_expandable(chapter)]
    if not expandable:
        return packages
    kinds = ["evidence_chain", "mechanism_boundary", "opportunity_risk", "verification", "data_lens", "decision_scenarios"]
    max_per_chapter = _env_int("REPORT_MAX_EXPANSION_SECTIONS_PER_CHAPTER", 4, min_value=0, max_value=8)
    added_chars = 0
    for round_index, kind in enumerate(kinds[:max_per_chapter], start=1):
        for chapter in expandable:
            sections = [dict(section) for section in _as_list(chapter.get("sections")) if isinstance(section, dict)]
            if any(section.get("expansion_generated") and section.get("section_title") == kind for section in sections):
                continue
            section = _expansion_section(chapter, kind=kind, index=round_index)
            if not _as_list(section.get("evidence_refs")):
                continue
            sections.append(section)
            chapter["sections"] = sections
            added_chars += _section_text_chars(section)
            if current + added_chars >= target:
                return packages
    return packages


def select_report_layout(report_plan: Dict[str, Any]) -> ReportLayout:
    plan = _as_dict(report_plan)
    report_type = str(plan.get("report_type") or plan.get("report_family") or "").strip()
    if report_type == "legacy_industry_deep":
        return ENTERPRISE_INDUSTRY_LAYOUT
    return GENERIC_DYNAMIC_LAYOUT


def _research_plan_from_inputs(
    report_plan: Dict[str, Any],
    evidence_package: Dict[str, Any],
    structured_analysis: Dict[str, Any],
) -> Dict[str, Any]:
    return (
        _as_dict(structured_analysis.get("research_plan"))
        or _as_dict(evidence_package.get("research_plan"))
        or _as_dict(_as_dict(evidence_package.get("metadata")).get("research_plan"))
        or _as_dict(report_plan.get("research_plan"))
        or _as_dict(report_plan)
    )


def _infer_dimensions_from_inputs(evidence_package: Dict[str, Any], structured_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    names: List[str] = []
    for key in _as_dict(evidence_package.get("per_dimension")).keys():
        text = str(key or "").strip()
        if text:
            names.append(text)
    for collection in (
        _as_list(evidence_package.get("analysis_ready_evidence")),
        _as_list(evidence_package.get("clean_evidence_list")),
        _as_list(structured_analysis.get("evidence_analyses")),
        _as_list(structured_analysis.get("claim_units")),
    ):
        for item in collection:
            if not isinstance(item, dict):
                continue
            text = (
                str(item.get("dimension_name") or "").strip()
                or str(item.get("dimension") or "").strip()
                or str(item.get("question") or "").strip()
            )
            if text:
                names.append(text)
    return [
        {
            "dimension_id": f"dim_{index}",
            "dimension_name": name,
            "purpose": f"围绕“{name}”形成可验证判断",
        }
        for index, name in enumerate(_dedupe(names, limit=12), start=1)
    ]


def prepare_dimension_materials(items: Sequence[Dict[str, Any]], *, dimension: str = "") -> List[Dict[str, Any]]:
    prepared: List[Dict[str, Any]] = []
    mode = str(os.getenv("REPORT_QUALITY_MODE") or os.getenv("QUALITY_MODE") or "balanced").strip().lower()
    strict_quality = mode in {"strict", "deep_strict", "due_diligence", "investment_due_diligence"} or str(
        os.getenv("STRICT_EVIDENCE_MODE") or ""
    ).strip().lower() in {"1", "true", "yes", "on", "strict"}
    for item in list(items or []):
        if not isinstance(item, dict):
            continue
        source_level = str(item.get("source_level") or _as_dict(item.get("source")).get("credibility") or "").strip().upper()
        role = str(item.get("evidence_role") or item.get("role") or "").strip().lower()
        semantic_status = str(item.get("semantic_status") or "").strip().lower()
        if role in {"rejected", "spam", "irrelevant", "blacklisted"}:
            continue
        if semantic_status in {"rejected", "spam", "irrelevant", "blacklisted"}:
            continue
        copied = dict(item)
        copied.setdefault("dimension", dimension)
        if apply_evidence_quality_contract is not None:
            copied = apply_evidence_quality_contract(
                copied,
                strict_quality=strict_quality,
                directional_c_min_confidence=_env_float(
                    "REPORT_DIRECTIONAL_C_MIN_CONFIDENCE",
                    0.55,
                    min_value=0.0,
                    max_value=1.0,
                ),
            )
            prepared.append(copied)
            continue
        try:
            confidence = float(copied.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0

        if source_level == "D":
            copied["appendix_only"] = True
            copied["enterprise_usable"] = False
            copied["followup_seed"] = True
            copied["usage_tier"] = "clue_low_quality"
        elif (
            source_level == "C"
            and confidence >= _env_float("REPORT_DIRECTIONAL_C_MIN_CONFIDENCE", 0.55, min_value=0.0, max_value=1.0)
            and role not in {"appendix", "appendix_only"}
        ):
            copied["appendix_only"] = False
            copied["enterprise_usable"] = True
            copied["can_support_claim_if_corrobated"] = True
            copied["allowed_use"] = copied.get("allowed_use") or "directional_signal"
            copied["usage_tier"] = "directional_signal"
        elif source_level == "C" or role in {"clue", "appendix", "appendix_only"}:
            copied["appendix_only"] = True
            copied["enterprise_usable"] = False
            copied["can_support_claim_if_corrobated"] = False
            copied["usage_tier"] = "appendix_or_corroboration"
        elif role == "supporting":
            copied["appendix_only"] = False
            copied["enterprise_usable"] = True
            copied["usage_tier"] = "supporting"
        elif role == "core":
            copied["appendix_only"] = False
            copied["enterprise_usable"] = True
            copied["usage_tier"] = "core"
        else:
            copied["appendix_only"] = True
            copied["enterprise_usable"] = False
            copied["usage_tier"] = "weak_clue"
        prepared.append(copied)
    return prepared


def collect_materials(
    *,
    child_outputs: Optional[Dict[str, Dict[str, Any]]] = None,
    evidence_pool: Optional[Sequence[Dict[str, Any]]] = None,
    evidence_package: Optional[Dict[str, Any]] = None,
    structured_analysis: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    blueprint = {
        "chapters": [
            {
                "chapter_id": "core_question",
                "chapter_title": "核心研究问题",
                "chapter_question": "核心研究问题",
            }
        ]
    }
    binder = run_evidence_binder(
        report_blueprint=blueprint,
        evidence_package=evidence_package,
        structured_analysis=structured_analysis,
        child_outputs=child_outputs,
        evidence_pool=evidence_pool,
    )
    return build_materials_payload_from_packages(binder)


def _layout_plan_from_packages(
    report_blueprint: Dict[str, Any],
    micro_layouts: Sequence[Dict[str, Any]],
    chapter_evidence_packages: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    micro_by_id = {
        str(layout.get("chapter_id") or ""): layout
        for layout in micro_layouts
        if isinstance(layout, dict)
    }
    evidence_by_id = {
        str(package.get("chapter_id") or ""): package
        for package in chapter_evidence_packages
        if isinstance(package, dict)
    }
    chapters: List[Dict[str, Any]] = []
    layout_gaps: List[Dict[str, Any]] = []
    global_followups: List[Dict[str, Any]] = []
    for chapter in _as_list(report_blueprint.get("chapters")):
        if not isinstance(chapter, dict):
            continue
        chapter_id = str(chapter.get("chapter_id") or "")
        micro = _as_dict(micro_by_id.get(chapter_id))
        evidence_package = _as_dict(evidence_by_id.get(chapter_id))
        blocks = [dict(block) for block in _as_list(micro.get("blocks")) if isinstance(block, dict)]
        if not blocks:
            for section in _as_list(micro.get("sections")):
                section = _as_dict(section)
                blocks.append(
                    {
                        "block_id": section.get("section_id"),
                        "block_type": section.get("output_type") or "argument",
                        "title": section.get("section_title"),
                        "purpose": section.get("section_role"),
                        "required_evidence": _as_list(section.get("required_evidence_refs")),
                        "min_words": section.get("min_words") or 160,
                    }
                )
        chapters.append(
            {
                "chapter_id": chapter_id,
                "chapter_title": chapter.get("chapter_title"),
                "dimension": chapter.get("dimension") or chapter.get("chapter_title"),
                "chapter_role": chapter.get("chapter_role") or chapter.get("chapter_question"),
                "chapter_question": chapter.get("chapter_question"),
                "layout_type": micro.get("layout_type") or "argument_first",
                "blocks": blocks,
                "table_requests": _as_list(micro.get("table_requests")),
                "evidence_needs": chapter.get("evidence_goals") or [],
                "follow_up_queries": _as_list(micro.get("follow_up_queries")),
            }
        )
        global_followups.extend(_as_list(micro.get("follow_up_queries")))
        missing = _as_list(evidence_package.get("missing_evidence"))
        if missing:
            for item in missing:
                item = _as_dict(item)
                query = str(item.get("follow_up_query") or "").strip()
                if query:
                    global_followups.append(
                        {
                            "query": query,
                            "agent": "iqs_lane_1",
                            "targets_gap": chapter.get("chapter_title") or chapter_id,
                            "reason": item.get("type") or "missing_evidence",
                            "source_priority": ["official", "financial_report", "association", "research"],
                        }
                    )
            layout_gaps.append(
                {
                    "dimension": chapter.get("chapter_title"),
                    "missing": [_compact(_as_dict(item).get("suggestion") or _as_dict(item).get("type"), 160) for item in missing],
                    "chapter_id": chapter_id,
                }
            )
    return {
        "report_type": report_blueprint.get("report_family") or "topic_report",
        "research_object": report_blueprint.get("research_object"),
        "narrative": report_blueprint.get("narrative"),
        "layout_strategy": _as_dict(report_blueprint.get("layout_strategy")),
        "report_shell": _as_dict(report_blueprint.get("report_shell")),
        "chapters": chapters,
        "global_follow_up_queries": global_followups,
        "layout_gaps": layout_gaps,
        "quality_rules": _as_dict(report_blueprint.get("quality_rules")),
    }


def _all_conflicts(chapter_evidence_packages: Sequence[Dict[str, Any]], evidence_graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    conflicts: List[Dict[str, Any]] = []
    for package in chapter_evidence_packages:
        if isinstance(package, dict):
            conflicts.extend([item for item in _as_list(package.get("conflicts")) if isinstance(item, dict)])
    conflicts.extend([item for item in _as_list(_as_dict(evidence_graph).get("conflicts")) if isinstance(item, dict)])
    return conflicts


def validate_dynamic_report(markdown: str, layout_plan: Optional[Dict[str, Any]] = None, materials: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    del layout_plan, materials
    qa = run_qa_agent(report_markdown=markdown, chapter_packages=[{"sections": [{"claim": "x", "reasoning": "x", "counter_evidence": "x", "actionable": "x"}]}])
    return {
        "ok": bool(markdown.strip()),
        "passed": bool(qa.get("passed")),
        "errors": _as_list(qa.get("errors")),
        "warnings": _as_list(qa.get("warnings")),
    }


def validate_enterprise_report(
    *,
    markdown: str,
    layout: Any = None,
    chapter_packages: Optional[Sequence[Dict[str, Any]]] = None,
    materials: Optional[Dict[str, Any]] = None,
    materials_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return _validate_enterprise_report(
        markdown=markdown,
        layout=layout,
        chapter_packages=chapter_packages,
        materials=materials,
        materials_payload=materials_payload,
    )


def render_dynamic_report(
    *,
    query: str = "",
    report_blueprint: Optional[Dict[str, Any]] = None,
    chapter_packages: Optional[Sequence[Dict[str, Any]]] = None,
    analytics_outputs: Optional[Sequence[Dict[str, Any]]] = None,
    table_packages: Optional[Sequence[Dict[str, Any]]] = None,
    decision_package: Optional[Dict[str, Any]] = None,
    risk_package: Optional[Dict[str, Any]] = None,
    appendix_package: Optional[Dict[str, Any]] = None,
    source_registry: Optional[Sequence[Dict[str, Any]]] = None,
    evidence_package: Optional[Dict[str, Any]] = None,
    chapter_evidence_packages: Optional[Sequence[Dict[str, Any]]] = None,
    **_: Any,
) -> Dict[str, Any]:
    extra = _as_dict(_)
    return run_final_writer_agent(
        query=query,
        report_blueprint=report_blueprint,
        chapter_packages=chapter_packages,
        table_packages=table_packages,
        decision_package=decision_package,
        risk_package=risk_package,
        appendix_package=appendix_package,
        source_registry=source_registry,
        evidence_package=evidence_package,
        chapter_evidence_packages=chapter_evidence_packages,
        claim_units=_as_list(extra.get("argument_units")),
        analysis_claim_units=_as_list(_as_dict(extra.get("structured_analysis")).get("claim_units")),
    )


def build_pipeline_debug_snapshot(
    *,
    evidence_package: Optional[Dict[str, Any]] = None,
    filter_funnel: Optional[Dict[str, Any]] = None,
    research_plan: Optional[Dict[str, Any]] = None,
    report_blueprint: Optional[Dict[str, Any]] = None,
    search_task_schedule: Optional[Dict[str, Any]] = None,
    lane_coverage: Optional[Dict[str, Any]] = None,
    chapter_evidence_packages: Optional[Sequence[Dict[str, Any]]] = None,
    evidence_graph: Optional[Dict[str, Any]] = None,
    analytics_outputs: Optional[Sequence[Dict[str, Any]]] = None,
    micro_layouts: Optional[Sequence[Dict[str, Any]]] = None,
    table_packages: Optional[Sequence[Dict[str, Any]]] = None,
    argument_units: Optional[Sequence[Dict[str, Any]]] = None,
    chapter_packages: Optional[Sequence[Dict[str, Any]]] = None,
    decision_package: Optional[Dict[str, Any]] = None,
    risk_package: Optional[Dict[str, Any]] = None,
    package_quality_report: Optional[Dict[str, Any]] = None,
    qa_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    evidence_package = _as_dict(evidence_package)
    evidence_graph = _as_dict(evidence_graph)
    decision_package = _as_dict(decision_package)
    risk_package = _as_dict(risk_package)
    package_quality_report = _as_dict(package_quality_report)
    qa_result = _as_dict(qa_result)
    dropped_sections = [
        {
            "chapter_id": chapter.get("chapter_id"),
            "chapter_title": chapter.get("chapter_title"),
            "internal_reason": chapter.get("internal_reason"),
            "dropped_sections": _as_list(chapter.get("dropped_sections")),
        }
        for chapter in list(chapter_packages or [])
        if isinstance(chapter, dict)
        and (chapter.get("omit_from_report") or _as_list(chapter.get("dropped_sections")))
    ]
    if not _include_full_debug_payload():
        compact_dropped_sections = [
            {
                "chapter_id": item.get("chapter_id"),
                "chapter_title": _compact(item.get("chapter_title"), 140),
                "internal_reason": _compact(item.get("internal_reason"), 120),
                "dropped_count": len(_as_list(item.get("dropped_sections"))),
            }
            for item in dropped_sections[:12]
            if isinstance(item, dict)
        ]
        return {
            "debug_payload_mode": "summary",
            "filter_funnel": _compact_debug_or_quality(_as_dict(filter_funnel) or _as_dict(evidence_package.get("filter_funnel"))),
            "dropped_sections": compact_dropped_sections,
            "research_plan": _compact_debug_or_quality(_as_dict(research_plan)),
            "report_blueprint": {
                **_compact_debug_or_quality(_as_dict(report_blueprint)),
                "chapter_count": len(_as_list(_as_dict(report_blueprint).get("chapters"))),
            },
            "search_task_schedule": _compact_debug_or_quality(_as_dict(search_task_schedule)),
            "lane_coverage": _compact_debug_or_quality(_as_dict(lane_coverage)),
            "pipeline_artifact_summary": _pipeline_artifact_summary(
                chapter_evidence_packages=[item for item in list(chapter_evidence_packages or []) if isinstance(item, dict)],
                evidence_graph=evidence_graph,
                analytics_outputs=[item for item in list(analytics_outputs or []) if isinstance(item, dict)],
                micro_layouts=[item for item in list(micro_layouts or []) if isinstance(item, dict)],
                table_packages=[item for item in list(table_packages or []) if isinstance(item, dict)],
                argument_units=[item for item in list(argument_units or []) if isinstance(item, dict)],
                chapter_packages=[item for item in list(chapter_packages or []) if isinstance(item, dict)],
                source_registry=[],
                qa_result=qa_result,
                package_quality_report=package_quality_report,
            ),
            "evidence_graph": _compact_graph(evidence_graph),
            "decision_package": _compact_debug_or_quality(decision_package),
            "risk_package": _compact_debug_or_quality(risk_package),
            "package_quality_report": _compact_debug_or_quality(package_quality_report),
            "qa_result": _compact_debug_or_quality(qa_result),
        }
    return {
        "debug_payload_mode": "full",
        "filter_funnel": _as_dict(filter_funnel) or _as_dict(evidence_package.get("filter_funnel")),
        "dropped_sections": dropped_sections,
        "research_plan": _as_dict(research_plan),
        "report_blueprint": _as_dict(report_blueprint),
        "search_task_schedule": _as_dict(search_task_schedule),
        "lane_coverage": _as_dict(lane_coverage),
        "chapter_evidence_packages": list(chapter_evidence_packages or []),
        "evidence_graph": evidence_graph,
        "analytics_outputs": list(analytics_outputs or []),
        "micro_layouts": list(micro_layouts or []),
        "table_packages": list(table_packages or []),
        "argument_units": list(argument_units or []),
        "chapter_packages": list(chapter_packages or []),
        "decision_package": decision_package,
        "risk_package": risk_package,
        "package_quality_report": package_quality_report,
        "qa_result": qa_result,
    }


def build_writer_report(
    *,
    query: str = "",
    child_outputs: Optional[Dict[str, Dict[str, Any]]] = None,
    evidence_pool: Optional[Sequence[Dict[str, Any]]] = None,
    evidence_package: Optional[Dict[str, Any]] = None,
    structured_analysis: Optional[Dict[str, Any]] = None,
    report_plan: Optional[Dict[str, Any]] = None,
    layout_plan: Optional[Dict[str, Any]] = None,
    report_blueprint: Optional[Dict[str, Any]] = None,
    chapter_evidence_packages: Optional[Sequence[Dict[str, Any]]] = None,
    micro_layouts: Optional[Sequence[Dict[str, Any]]] = None,
    analytics_outputs: Optional[Sequence[Dict[str, Any]]] = None,
    table_packages: Optional[Sequence[Dict[str, Any]]] = None,
    argument_units: Optional[Sequence[Dict[str, Any]]] = None,
    chapter_packages: Optional[Sequence[Dict[str, Any]]] = None,
    decision_package: Optional[Dict[str, Any]] = None,
    risk_package: Optional[Dict[str, Any]] = None,
    appendix_package: Optional[Dict[str, Any]] = None,
    source_registry: Optional[Sequence[Dict[str, Any]]] = None,
    search_task_schedule: Optional[Dict[str, Any]] = None,
    lane_coverage: Optional[Dict[str, Any]] = None,
    llm_client: Any = None,
) -> Dict[str, Any]:
    evidence_package = _as_dict(evidence_package)
    structured_analysis = _as_dict(structured_analysis)
    report_insight_package = _as_dict(structured_analysis.get("report_insight_package"))
    search_task_schedule = _as_dict(search_task_schedule)
    lane_coverage = _as_dict(lane_coverage)
    report_plan = _as_dict(report_plan) or _as_dict(structured_analysis.get("report_plan")) or _as_dict(evidence_package.get("report_plan"))
    research_plan = _research_plan_from_inputs(report_plan, evidence_package, structured_analysis)
    if report_plan and not research_plan.get("report_family"):
        research_plan = {**research_plan, "report_family": report_plan.get("report_family") or report_plan.get("report_type")}
    if not _as_list(research_plan.get("dimensions")):
        inferred_dimensions = _infer_dimensions_from_inputs(evidence_package, structured_analysis)
        if inferred_dimensions:
            research_plan = {**research_plan, "dimensions": inferred_dimensions}

    report_blueprint = _as_dict(report_blueprint) or run_pre_layout_agent(
        query=query,
        research_plan=research_plan,
        report_plan=report_plan,
        llm_client=llm_client,
    )
    config_warnings: List[Dict[str, Any]] = []
    configured_proof_mode = str(os.getenv("REPORT_PROOF_MODE") or "").strip().lower()
    if (
        configured_proof_mode == "quick_market_scan"
        and _deep_report_family(report_blueprint.get("report_family") or research_plan.get("report_family"))
        and not _env_flag("REPORT_ALLOW_QUICK_PROOF_FOR_DEEP", False)
    ):
        config_warnings.append(
            {
                "type": "quick_proof_upgraded_for_deep_report",
                "configured_mode": configured_proof_mode,
                "effective_mode": "deep_industry_report",
            }
        )

    if _chapter_evidence_packages_are_compacted(chapter_evidence_packages):
        chapter_evidence_packages = None

    if chapter_evidence_packages is None or source_registry is None:
        binder = run_evidence_binder(
            research_plan=research_plan,
            report_blueprint=report_blueprint,
            evidence_package=evidence_package,
            structured_analysis=structured_analysis,
            child_outputs=child_outputs,
            evidence_pool=evidence_pool,
        )
        chapter_evidence_packages = _as_list(binder.get("chapter_evidence_packages"))
        source_registry = _as_list(binder.get("source_registry"))
        footnotes = _as_list(binder.get("footnotes"))
        binder_metadata = _as_dict(binder.get("metadata"))
        source_quality_map = _as_dict(binder.get("source_quality_map"))
        metric_normalization_table = _as_list(binder.get("metric_normalization_table"))
        coverage_matrix = _as_list(binder.get("coverage_matrix"))
        missing_proof_standards = _as_list(binder.get("missing_proof_standards"))
        evidence_refinement_plan = _as_dict(binder.get("evidence_refinement_plan"))
        research_proof_profile = _as_dict(binder.get("research_proof_profile"))
        mandatory_proof_checks = _as_list(binder.get("mandatory_proof_checks"))
    else:
        footnotes = []
        binder_metadata = {}
        source_quality_map = {}
        metric_normalization_table = []
        coverage_matrix = []
        missing_proof_standards = []
        evidence_refinement_plan = _as_dict(_as_dict(structured_analysis).get("evidence_refinement_plan"))
        research_proof_profile = _as_dict(_as_dict(structured_analysis).get("research_proof_profile"))
        mandatory_proof_checks = _as_list(_as_dict(structured_analysis).get("mandatory_proof_checks"))

    rebuilt_chapter_evidence_packages = build_chapter_evidence_packages_from_evidence_package(
        report_blueprint=report_blueprint,
        evidence_package=evidence_package,
        existing_chapter_evidence_packages=chapter_evidence_packages,
        source_registry=source_registry,
    )
    rebuilt_signal_count = _chapter_evidence_signal_count(rebuilt_chapter_evidence_packages)
    existing_signal_count = _chapter_evidence_signal_count(chapter_evidence_packages)
    rebuilt_hydrated_count = _chapter_evidence_hydrated_count(rebuilt_chapter_evidence_packages)
    existing_hydrated_count = _chapter_evidence_hydrated_count(chapter_evidence_packages)
    rebuilt_layered_count = _chapter_evidence_layered_count(rebuilt_chapter_evidence_packages)
    existing_layered_count = _chapter_evidence_layered_count(chapter_evidence_packages)
    rebuilt_selection_score = _chapter_evidence_selection_score(rebuilt_chapter_evidence_packages)
    existing_selection_score = _chapter_evidence_selection_score(chapter_evidence_packages)
    prefer_rebuilt_for_layers = rebuilt_layered_count > 0 and existing_layered_count <= 0
    prefer_rebuilt_for_score = rebuilt_selection_score > existing_selection_score
    if rebuilt_chapter_evidence_packages and rebuilt_hydrated_count > 0 and (
        prefer_rebuilt_for_layers
        or prefer_rebuilt_for_score
        or (rebuilt_hydrated_count >= existing_hydrated_count and rebuilt_signal_count >= existing_signal_count)
        or existing_hydrated_count <= 0
    ):
        chapter_evidence_packages = rebuilt_chapter_evidence_packages
        evidence_package["chapter_evidence_packages"] = list(chapter_evidence_packages)
        evidence_package["chapter_evidence_rebuild_diagnostics"] = {
            "status": "used_rebuilt_packages",
            "reason": "layered_or_scored_package_preferred",
            "rebuilt_signal_count": rebuilt_signal_count,
            "existing_signal_count": existing_signal_count,
            "rebuilt_hydrated_count": rebuilt_hydrated_count,
            "existing_hydrated_count": existing_hydrated_count,
            "rebuilt_layered_count": rebuilt_layered_count,
            "existing_layered_count": existing_layered_count,
            "rebuilt_selection_score": rebuilt_selection_score,
            "existing_selection_score": existing_selection_score,
        }
    elif rebuilt_chapter_evidence_packages:
        evidence_package.setdefault("chapter_evidence_rebuild_diagnostics", {})
        evidence_package["chapter_evidence_rebuild_diagnostics"] = {
            "status": "kept_existing_packages",
            "reason": "rebuilt_package_had_lower_selection_score",
            "rebuilt_signal_count": rebuilt_signal_count,
            "existing_signal_count": existing_signal_count,
            "rebuilt_hydrated_count": rebuilt_hydrated_count,
            "existing_hydrated_count": existing_hydrated_count,
            "rebuilt_layered_count": rebuilt_layered_count,
            "existing_layered_count": existing_layered_count,
            "rebuilt_selection_score": rebuilt_selection_score,
            "existing_selection_score": existing_selection_score,
        }

    evidence_graph = run_evidence_synthesizer(
        chapter_evidence_packages=chapter_evidence_packages,
        llm_client=llm_client,
    )
    analytics_outputs = list(analytics_outputs or run_analytics_agents(
        report_blueprint=report_blueprint,
        chapter_evidence_packages=chapter_evidence_packages,
        evidence_graph=evidence_graph,
        metric_normalization_table=metric_normalization_table,
        coverage_matrix=coverage_matrix,
    ))

    micro_layouts = list(micro_layouts or run_micro_layout_agent(
        report_blueprint=report_blueprint,
        chapter_evidence_packages=chapter_evidence_packages,
        structured_analysis=structured_analysis,
        llm_client=llm_client,
    ))
    table_packages = copy.deepcopy(list(table_packages or run_table_agent(
        chapter_evidence_packages=chapter_evidence_packages,
        micro_layouts=micro_layouts,
        analytics_outputs=analytics_outputs,
        llm_client=llm_client,
    )))
    body_table_budget = _body_table_budget()
    per_chapter_table_budget = _per_chapter_table_budget()
    rendered_tables = 0
    rendered_tables_by_chapter: Dict[str, int] = {}
    for table in table_packages:
        if not isinstance(table, dict) or not table.get("should_render") or table.get("appendix_only"):
            continue
        chapter_id = str(table.get("chapter_id") or "").strip()
        if rendered_tables >= body_table_budget:
            table["should_render"] = False
            table.setdefault("reject_reasons", []).append("global_body_table_budget_exceeded")
            continue
        if rendered_tables_by_chapter.get(chapter_id, 0) >= per_chapter_table_budget:
            table["should_render"] = False
            table.setdefault("reject_reasons", []).append("chapter_body_table_budget_exceeded")
            continue
        rendered_tables += 1
        rendered_tables_by_chapter[chapter_id] = rendered_tables_by_chapter.get(chapter_id, 0) + 1
    public_rebuild_summary = needs_public_rebuild(
        argument_units=argument_units,
        chapter_packages=chapter_packages,
        structured_analysis=structured_analysis,
    )
    if public_rebuild_summary.get("required") or not argument_units or not chapter_packages:
        rebuild_result = rebuild_public_argument_pipeline(
            chapter_evidence_packages=chapter_evidence_packages,
            micro_layouts=micro_layouts,
            structured_analysis=structured_analysis,
            report_blueprint=report_blueprint,
            table_packages=table_packages,
            llm_client=llm_client,
        )
        argument_units = _as_list(rebuild_result.get("argument_units"))
        chapter_packages = _as_list(rebuild_result.get("chapter_packages"))
        public_rebuild_summary = {
            **public_rebuild_summary,
            "triggered": bool(public_rebuild_summary.get("required")),
            "generated_missing_inputs": bool(not public_rebuild_summary.get("argument_unit_count_before") or not public_rebuild_summary.get("chapter_package_count_before")),
            "argument_unit_count_after": rebuild_result.get("argument_unit_count_after"),
            "chapter_package_count_after": rebuild_result.get("chapter_package_count_after"),
        }
    else:
        argument_units = list(argument_units or [])
        chapter_packages = list(chapter_packages or [])
        public_rebuild_summary = {**public_rebuild_summary, "triggered": False}
    package_normalization = _normalize_public_packages_for_contract(
        chapter_evidence_packages=chapter_evidence_packages,
        micro_layouts=micro_layouts,
        table_packages=table_packages,
        argument_units=argument_units,
        chapter_packages=chapter_packages,
    )
    micro_layouts = _as_list(package_normalization.get("micro_layouts"))
    table_packages = _as_list(package_normalization.get("table_packages"))
    argument_units = _as_list(package_normalization.get("argument_units"))
    chapter_packages = _as_list(package_normalization.get("chapter_packages"))
    package_normalization_summary = _as_dict(package_normalization.get("summary"))
    package_normalization_summary["public_rebuild"] = public_rebuild_summary
    table_followups = _table_follow_up_queries(table_packages)
    evidence_refinement_plan = _merge_table_followups_into_refinement_plan(evidence_refinement_plan, table_followups)
    public_chapter_packages = [
        chapter
        for chapter in chapter_packages
        if isinstance(chapter, dict) and not chapter.get("omit_from_report")
    ]
    public_argument_units = [
        unit
        for unit in argument_units
        if isinstance(unit, dict) and unit.get("public_render") is True and not unit.get("omit_from_report")
    ]
    tables_enabled = str(os.getenv("REPORT_ENABLE_TABLES", "true")).strip().lower() in {"1", "true", "yes", "on"}
    public_table_packages = [
        table
        for table in table_packages
        if tables_enabled and isinstance(table, dict) and table.get("should_render") and not table.get("appendix_only")
    ]
    target_body_chars = _env_int("REPORT_TARGET_BODY_CHARS", 0, min_value=0, max_value=100000)
    public_chapter_packages = _expand_chapter_packages_for_body_target(public_chapter_packages, target_chars=target_body_chars)
    chapter_narrative_diagnostics: Dict[str, Any] = {
        "enabled": False,
        "status": "skipped",
        "skipped_reason": "not_run",
    }
    try:
        public_chapter_packages, chapter_narrative_diagnostics = run_chapter_narrative(
            chapter_packages=public_chapter_packages,
            report_blueprint=report_blueprint,
            llm_config=None,
            quality_context=_as_dict(_as_dict(structured_analysis).get("analysis_stage_diagnostics")),
        )
    except Exception as exc:
        chapter_narrative_diagnostics = {
            "enabled": True,
            "status": "yellow",
            "skipped_reason": "",
            "attempted_count": 0,
            "success_count": 0,
            "fallback_count": len(public_chapter_packages),
            "rejected_count": 0,
            "rejected_reasons": {},
            "failure_reasons": {f"runtime_error:{type(exc).__name__}": 1},
        }
    expanded_public_by_id = {str(chapter.get("chapter_id") or ""): chapter for chapter in public_chapter_packages}
    chapter_packages = [
        expanded_public_by_id.get(str(chapter.get("chapter_id") or ""), chapter)
        if isinstance(chapter, dict) and not chapter.get("omit_from_report")
        else chapter
        for chapter in chapter_packages
    ]
    decision_package = _as_dict(decision_package) or run_decision_synthesis_agent(
        report_blueprint=report_blueprint,
        chapter_packages=public_chapter_packages,
        argument_units=public_argument_units,
        table_packages=public_table_packages,
        llm_client=llm_client,
    )
    risk_package = _as_dict(risk_package) or run_risk_agent(
        chapter_packages=public_chapter_packages,
        argument_units=public_argument_units,
        evidence_conflicts=_all_conflicts(chapter_evidence_packages, evidence_graph),
        decision_package=decision_package,
        llm_client=llm_client,
    )
    package_quality_report = validate_pipeline_packages(
        report_blueprint=report_blueprint,
        chapter_evidence_packages=chapter_evidence_packages,
        micro_layouts=micro_layouts,
        table_packages=table_packages,
        argument_units=argument_units,
        chapter_packages=chapter_packages,
    )
    appendix_payload = {
        **_as_dict(appendix_package),
        "metric_normalization_table": metric_normalization_table,
        "coverage_matrix": coverage_matrix,
        "missing_proof_standards": missing_proof_standards,
        "analytics_outputs": analytics_outputs,
        "table_appendix_rows": _table_appendix_rows(table_packages),
    }

    writer_output = run_final_writer_agent(
        query=query,
        report_blueprint=report_blueprint,
        chapter_packages=public_chapter_packages,
        table_packages=public_table_packages,
        decision_package=decision_package,
        risk_package=risk_package,
        appendix_package=appendix_payload,
        source_registry=source_registry,
        evidence_package=evidence_package,
        chapter_evidence_packages=chapter_evidence_packages,
        claim_units=argument_units,
    )
    writer_output["report_markdown"] = str(writer_output.get("report_markdown") or "")
    rendered_source_registry = _as_list(writer_output.get("source_registry")) or list(source_registry or [])
    rendered_footnotes = [
        f"{source.get('ref')} "
        + " | ".join(
            part
            for part in [
                str(source.get("title") or "").strip(),
                str(source.get("date") or "").strip(),
                str(source.get("url") or "").strip(),
            ]
            if part
        )
        for source in rendered_source_registry
        if isinstance(source, dict) and str(source.get("ref") or "").strip()
    ] or list(footnotes or [])
    writer_output["estimated_chars"] = len(str(writer_output.get("report_markdown") or ""))
    writer_output["estimated_body_chars"] = len(
        re.split(r"\n##\s*(?:\u9644\u5f55|附錄|研究口径|研究口徑)", str(writer_output.get("report_markdown") or ""), maxsplit=1)[0]
    )
    writer_output["target_body_chars"] = target_body_chars
    retrieval_strategy_summary = (
        _as_dict(_as_dict(evidence_package.get("metadata")).get("retrieval_strategy_summary"))
        or _as_dict(_as_dict(evidence_package.get("summary")).get("retrieval_strategy_summary"))
    )
    evidence_health_summary = _evidence_health_summary_from_package(evidence_package)
    research_reflection_memo = (
        _as_dict(structured_analysis.get("research_reflection_memo"))
        or _as_dict(_as_dict(structured_analysis.get("report_insight_package")).get("research_reflection_memo"))
        or _as_dict(evidence_package.get("research_reflection_memo"))
    )
    if not research_reflection_memo and build_research_reflection_memo is not None:
        research_reflection_memo = build_research_reflection_memo(
            evidence_package,
            structured_analysis=structured_analysis,
        )
    qa_result = run_qa_agent(
        report_markdown=str(writer_output.get("report_markdown") or ""),
        report_blueprint=report_blueprint,
        chapter_packages=public_chapter_packages,
        table_packages=public_table_packages,
        decision_package=decision_package,
        risk_package=risk_package,
        package_quality_report=package_quality_report,
        search_task_schedule=search_task_schedule,
        lane_coverage=lane_coverage,
        retrieval_strategy_summary=retrieval_strategy_summary,
        metric_normalization_table=metric_normalization_table,
        coverage_matrix=coverage_matrix,
        missing_proof_standards=missing_proof_standards,
        analytics_outputs=analytics_outputs,
        evidence_health_summary=evidence_health_summary,
    )
    if qa_result.get("rewrite_required"):
        rewrite_instructions = [
            *[str(item) for item in _as_list(qa_result.get("rewrite_instructions")) if str(item).strip()],
            *[
                str(_as_dict(item).get("message") or _as_dict(item).get("type") or item)
                for item in [*_as_list(qa_result.get("errors")), *_as_list(qa_result.get("issues"))]
                if str(item).strip()
            ],
        ]
        rewritten = run_rewrite_agent(
            report_markdown=str(writer_output.get("report_markdown") or ""),
            qa_result=qa_result,
            rewrite_instructions=rewrite_instructions,
        )
        if rewritten.get("changed"):
            candidate_markdown = sanitize_public_markdown(rewritten.get("report_markdown") or "")
            candidate_output = {
                **writer_output,
                "report_markdown": candidate_markdown,
                "estimated_chars": len(candidate_markdown),
                "estimated_body_chars": len(
                    re.split(r"\n##\s*(?:\u9644\u5f55|附錄|研究口径|研究口徑)", candidate_markdown, maxsplit=1)[0]
                ),
            }
            candidate_qa = run_qa_agent(
                report_markdown=str(candidate_output.get("report_markdown") or ""),
                report_blueprint=report_blueprint,
                chapter_packages=public_chapter_packages,
                table_packages=public_table_packages,
                decision_package=decision_package,
                risk_package=risk_package,
                package_quality_report=package_quality_report,
                search_task_schedule=search_task_schedule,
                lane_coverage=lane_coverage,
                retrieval_strategy_summary=retrieval_strategy_summary,
                metric_normalization_table=metric_normalization_table,
                coverage_matrix=coverage_matrix,
                missing_proof_standards=missing_proof_standards,
                analytics_outputs=analytics_outputs,
                evidence_health_summary=evidence_health_summary,
            )
            current_error_count = len(_as_list(qa_result.get("errors")))
            candidate_error_count = len(_as_list(candidate_qa.get("errors")))
            current_score = int(qa_result.get("quality_score") or 0)
            candidate_score = int(candidate_qa.get("quality_score") or 0)
            improves_errors = candidate_error_count < current_error_count
            improves_score = candidate_score > current_score
            fixes_pass = bool(candidate_qa.get("passed")) and not bool(qa_result.get("passed"))
            if fixes_pass or improves_errors or (candidate_error_count <= current_error_count and improves_score):
                writer_output = candidate_output
                rendered_source_registry = _as_list(writer_output.get("source_registry")) or rendered_source_registry
                rendered_footnotes = [
                    f"{source.get('ref')} "
                    + " | ".join(
                        part
                        for part in [
                            str(source.get("title") or "").strip(),
                            str(source.get("date") or "").strip(),
                            str(source.get("url") or "").strip(),
                        ]
                        if part
                    )
                    for source in rendered_source_registry
                    if isinstance(source, dict) and str(source.get("ref") or "").strip()
                ] or rendered_footnotes
                qa_result = candidate_qa

    layout = _as_dict(layout_plan) or _layout_plan_from_packages(report_blueprint, micro_layouts, chapter_evidence_packages)
    package_passed = bool(package_quality_report.get("passed"))
    public_markdown = str(writer_output.get("report_markdown") or "")
    delivery_gate = _delivery_gate_from_evidence_package(evidence_package)
    delivery_tier = str(delivery_gate.get("tier") or "limited_review_draft").strip() or "limited_review_draft"
    package_warning_types = _package_warning_types(package_quality_report)
    package_warning_blocked = (
        _env_flag("QA_BLOCK_LOW_AB_CORE_COVERAGE", False)
        and "low_ab_core_coverage" in package_warning_types
    )
    table_gap_summary = _table_gap_summary(table_packages)
    delivery_blockers = _hard_delivery_blockers(
        markdown=public_markdown,
        qa_result=qa_result,
        package_quality_report=package_quality_report,
        coverage_matrix=coverage_matrix,
        delivery_gate=delivery_gate,
    )
    delivery_blockers.extend(_layout_delivery_blockers(report_blueprint))
    delivery_blockers.extend(
        _report_family_delivery_blockers(
            research_plan=research_plan,
            report_blueprint=report_blueprint,
            report_plan=report_plan,
        )
    )
    qa_pending_repair_reasons = _qa_pending_repair_reasons(qa_result)
    qa_pending_repair = bool(qa_pending_repair_reasons)
    quality_findings = _quality_findings_from_review(
        delivery_blockers=delivery_blockers,
        qa_result=qa_result,
        package_quality_report=package_quality_report,
        delivery_gate=delivery_gate,
        table_gap_summary=table_gap_summary,
    )
    fatal_delivery_blockers = _fatal_delivery_blockers(delivery_blockers, markdown=public_markdown)
    clean_report_eligible = False
    clean_content_eligible = False
    clean_output_enabled = _env_flag("REPORT_WRITE_CLEAN_REPORT", False)
    clean_candidate_gate = _as_dict(_as_dict(qa_result).get("clean_gate"))
    clean_candidate_eligible = bool(clean_candidate_gate.get("clean_candidate_eligible"))
    claim_strength = _claim_strength_from_gate(delivery_gate, evidence_health_summary)
    quality_score = _quality_score_from_findings(
        qa_result=qa_result,
        quality_findings=quality_findings,
        evidence_health_summary=evidence_health_summary,
        delivery_gate=delivery_gate,
    )
    if fatal_delivery_blockers:
        blocked_markdown = _render_not_ready_report(
            query=query,
            report_blueprint=report_blueprint,
            blockers=fatal_delivery_blockers,
            qa_result=qa_result,
            coverage_matrix=coverage_matrix,
            missing_proof_standards=missing_proof_standards,
            search_task_schedule=search_task_schedule,
            lane_coverage=lane_coverage,
        )
        writer_output = {
            **writer_output,
            "report_markdown": "",
            "diagnostic_markdown": blocked_markdown,
            "blocked_report_markdown": blocked_markdown,
            "estimated_chars": 0,
            "estimated_body_chars": 0,
            "blocked_draft_chars": len(public_markdown),
        }
        public_markdown = ""
        report_status = "diagnostic_only"
        delivery_tier = "diagnostic_only"
        message = "公开资料条件不足，正式报告未生成。"
    else:
        report_ready = _writer_ready_for_final(
            markdown=public_markdown,
            qa_result=qa_result,
            package_passed=package_passed,
            package_warning_blocked=package_warning_blocked,
        ) and delivery_tier == "publishable_clean"
        clean_standard = _clean_standard()
        strict_clean_content_eligible = bool(report_ready and not delivery_blockers)
        render_gate = _as_dict(_as_dict(qa_result).get("render_gate"))
        balanced_clean_content_eligible = bool(
            clean_candidate_eligible
            and public_markdown.strip()
            and not bool(render_gate.get("blocked"))
            and not bool(_as_dict(delivery_gate).get("diagnostic_only"))
            and str(delivery_tier or "").strip() != "diagnostic_only"
            and not has_internal_gap_language(public_markdown)
        )
        if clean_standard == "strict":
            clean_content_eligible = strict_clean_content_eligible
        elif clean_standard == "relaxed":
            clean_content_eligible = bool(
                public_markdown.strip()
                and not bool(render_gate.get("blocked"))
                and not bool(_as_dict(delivery_gate).get("diagnostic_only"))
                and not has_internal_gap_language(public_markdown)
            )
        else:
            clean_content_eligible = bool(strict_clean_content_eligible or balanced_clean_content_eligible)
        clean_report_eligible = clean_content_eligible
        clean_candidate_eligible = bool(clean_candidate_eligible or clean_content_eligible)
        report_status = "final_clean" if clean_content_eligible else "formal_scored"
        if report_status == "final_clean":
            message = ""
        elif qa_pending_repair:
            message = "报告已按证据强度降级生成；QA 建议继续补证/重写后再发布 Clean 版本。"
        else:
            message = "报告已按证据强度降级生成，建议结合评分和缺陷清单人工复核。"
        if report_status == "formal_scored":
            delivery_tier = "scored_formal_report"
            public_markdown = _downgrade_overconfident_language(public_markdown, claim_strength=claim_strength)
        scorecard = _render_quality_scorecard(
            query=query,
            quality_score=quality_score,
            clean_report_eligible=clean_report_eligible,
            claim_strength=claim_strength,
            evidence_health_summary=evidence_health_summary,
            quality_findings=quality_findings,
            delivery_gate=delivery_gate,
            evidence_package=evidence_package,
        )
        writer_output["score_markdown"] = scorecard
        writer_output["report_markdown"] = public_markdown
        writer_output["estimated_chars"] = len(public_markdown)
        writer_output["estimated_body_chars"] = len(
            re.split(r"\n##\s*(?:\u9644\u5f55|附录|附錄|研究口径|研究口徑)", public_markdown, maxsplit=1)[0]
        )
    debug_snapshot = build_pipeline_debug_snapshot(
        evidence_package=evidence_package,
        research_plan=research_plan,
        report_blueprint=report_blueprint,
        search_task_schedule=search_task_schedule,
        lane_coverage=lane_coverage,
        chapter_evidence_packages=chapter_evidence_packages,
        evidence_graph=evidence_graph,
        analytics_outputs=analytics_outputs,
        micro_layouts=micro_layouts,
        table_packages=table_packages,
        argument_units=argument_units,
        chapter_packages=chapter_packages,
        decision_package=decision_package,
        risk_package=risk_package,
        package_quality_report=package_quality_report,
        qa_result=qa_result,
    )
    debug_snapshot["table_gap_summary"] = table_gap_summary
    debug_snapshot["table_follow_up_count"] = table_gap_summary.get("table_follow_up_count", 0)
    debug_snapshot["rendered_high_value_table_count"] = table_gap_summary.get("rendered_high_value_table_count", 0)
    artifact_payload = _pipeline_artifacts_payload(
        chapter_evidence_packages=[item for item in list(chapter_evidence_packages or []) if isinstance(item, dict)],
        source_quality_map=source_quality_map,
        metric_normalization_table=[item for item in list(metric_normalization_table or []) if isinstance(item, dict)],
        coverage_matrix=[item for item in list(coverage_matrix or []) if isinstance(item, dict)],
        missing_proof_standards=list(missing_proof_standards or []),
        evidence_graph=_as_dict(evidence_graph),
        analytics_outputs=[item for item in list(analytics_outputs or []) if isinstance(item, dict)],
        micro_layouts=[item for item in list(micro_layouts or []) if isinstance(item, dict)],
        table_packages=[item for item in list(table_packages or []) if isinstance(item, dict)],
        argument_units=[item for item in list(argument_units or []) if isinstance(item, dict)],
        chapter_packages=[item for item in list(chapter_packages or []) if isinstance(item, dict)],
        decision_package=_as_dict(decision_package),
        risk_package=_as_dict(risk_package),
        appendix_package=_as_dict(appendix_payload),
        source_registry=[item for item in list(rendered_source_registry or []) if isinstance(item, dict)],
        qa_result=_as_dict(qa_result),
        package_quality_report=_as_dict(package_quality_report),
    )
    table_planning_summary = _table_planning_summary(micro_layouts)
    table_quality_summary = _table_quality_summary(table_packages)
    table_placement_summary = _table_placement_summary(table_packages)
    evidence_analysis_summary = _as_dict(evidence_package.get("evidence_analysis_summary"))
    evidence_gap_ledger = _as_list(evidence_package.get("evidence_gap_ledger"))
    evidence_analysis_by_chapter = _as_dict(evidence_package.get("evidence_analysis_by_chapter"))
    chapter_evidence_diagnostics = (
        _as_dict(evidence_package.get("chapter_evidence_diagnostics"))
        or _as_dict(structured_analysis.get("chapter_evidence_diagnostics"))
        or evidence_analysis_by_chapter
    )
    analysis_depth_quality = _as_dict(structured_analysis.get("analysis_depth_quality"))
    analysis_stage_diagnostics = _as_dict(structured_analysis.get("analysis_stage_diagnostics"))
    llm_analysis_synthesis = _as_dict(structured_analysis.get("llm_analysis_synthesis"))
    claim_binding_feedback_summary = _as_dict(structured_analysis.get("claim_binding_feedback_summary"))
    stage_quality_card = _stage_quality_card(
        chapter_evidence_packages=[item for item in list(chapter_evidence_packages or []) if isinstance(item, dict)],
        table_quality_summary=table_quality_summary,
        table_gap_summary=table_gap_summary,
        analysis_stage_diagnostics=analysis_stage_diagnostics,
        final_citation_audit=_as_dict(writer_output.get("final_citation_audit")),
    )
    debug_snapshot["stage_quality_card"] = stage_quality_card
    render_artifacts = {
        "payload_mode": "full",
        "structured_analysis": structured_analysis,
        "chapter_evidence_packages": [
            item for item in list(chapter_evidence_packages or []) if isinstance(item, dict)
        ],
        "chapter_packages": [
            item for item in list(chapter_packages or []) if isinstance(item, dict)
        ],
        "table_packages": [
            item for item in list(table_packages or []) if isinstance(item, dict)
        ],
        "source_registry": [
            item for item in list(rendered_source_registry or []) if isinstance(item, dict)
        ],
        "micro_layouts": [
            item for item in list(micro_layouts or []) if isinstance(item, dict)
        ],
        "argument_units": [
            item for item in list(argument_units or []) if isinstance(item, dict)
        ],
        "evidence_package": evidence_package,
        "research_reflection_memo": research_reflection_memo,
        "citation_manifest": _as_dict(writer_output.get("citation_manifest")),
        "final_citation_audit": _as_dict(writer_output.get("final_citation_audit")),
        "source_claim_support": _as_dict(writer_output.get("source_claim_support")),
        "analysis_transfer": _as_dict(writer_output.get("analysis_transfer")),
        "ref_lineage_diagnostics": _as_dict(writer_output.get("ref_lineage_diagnostics")),
        "chapter_narrative": chapter_narrative_diagnostics,
        "stage_quality_card": stage_quality_card,
    }
    return {
        **writer_output,
        "chapter_narrative": chapter_narrative_diagnostics,
        "report_type": report_blueprint.get("report_family") or select_report_layout(report_plan).report_type,
        "report_status": report_status,
        "delivery_tier": delivery_tier,
        "delivery_gate": delivery_gate,
        "draft_mode": "scored_formal" if report_status == "formal_scored" else ("diagnostic_only" if report_status == "diagnostic_only" else ""),
        "evidence_limitations": _evidence_limitations_from_delivery_gate(delivery_gate, qa_result),
        "message": message,
        "config_warnings": config_warnings,
        "delivery_blockers": delivery_blockers,
        "fatal_delivery_blockers": fatal_delivery_blockers,
        "quality_findings": quality_findings,
        "quality_score": quality_score,
        "quality_grade": _quality_grade(quality_score),
        "clean_report_eligible": clean_report_eligible,
        "clean_content_eligible": clean_content_eligible,
        "clean_candidate_eligible": clean_candidate_eligible,
        "clean_output_enabled": clean_output_enabled,
        "clean_standard": _clean_standard(),
        "claim_strength": claim_strength,
        "delivery_gate_mode": _delivery_gate_mode(),
        "skip_reformatter": report_status == "diagnostic_only",
        "package_warning_blocked": package_warning_blocked,
        "qa_pending_repair": qa_pending_repair,
        "qa_pending_repair_reasons": qa_pending_repair_reasons,
        "report_blueprint": report_blueprint,
        "search_tasks": _as_list(research_plan.get("search_tasks")),
        "search_task_schedule": search_task_schedule,
        "lane_coverage": lane_coverage,
        "layout_plan": layout,
        "report_insight_package": report_insight_package,
        "evidence_refinement_plan": evidence_refinement_plan,
        "research_proof_profile": research_proof_profile,
        "mandatory_proof_checks": mandatory_proof_checks,
        "table_planning_summary": table_planning_summary,
        "table_quality_summary": table_quality_summary,
        "table_placement_summary": table_placement_summary,
        "table_gap_summary": table_gap_summary,
        "stage_quality_card": stage_quality_card,
        "evidence_health_summary": evidence_health_summary,
        "research_reflection_memo": research_reflection_memo,
        "evidence_analysis_summary": evidence_analysis_summary,
        "evidence_gap_ledger": evidence_gap_ledger,
        "evidence_analysis_by_chapter": evidence_analysis_by_chapter,
        "chapter_evidence_diagnostics": chapter_evidence_diagnostics,
        "analysis_depth_quality": analysis_depth_quality,
        "analysis_stage_diagnostics": analysis_stage_diagnostics,
        "llm_analysis_synthesis": llm_analysis_synthesis,
        "claim_binding_feedback_summary": claim_binding_feedback_summary,
        **artifact_payload,
        "render_artifacts": render_artifacts,
        "source_registry": list(rendered_source_registry or []),
        "footnotes": rendered_footnotes,
        "validation": {
            **qa_result,
            "ok": bool(writer_output.get("report_markdown")) and report_status != "diagnostic_only",
        },
        "qa_result": qa_result,
        "required_followups": _as_list(qa_result.get("repair_followups"))
        or _as_list(_as_dict(qa_result.get("deep_evaluation")).get("required_followups"))
        or list(table_followups),
        "package_quality_report": package_quality_report,
        "package_normalization_summary": package_normalization_summary,
        "debug_snapshot": debug_snapshot,
        "metadata": {
            "writer": AGENT_NAME,
            "strategy": "split_final_writer_pipeline",
            "uses_llm": bool(llm_client),
            "evidence_pool_count": len([item for item in list(evidence_pool or []) if isinstance(item, dict)]),
            **binder_metadata,
            "search_task_schedule": search_task_schedule,
            "lane_coverage": lane_coverage,
            "analytics_output_count": len(analytics_outputs),
            "table_planning_summary": table_planning_summary,
            "table_quality_summary": table_quality_summary,
            "table_placement_summary": table_placement_summary,
            "table_gap_summary": table_gap_summary,
            "public_rebuild_triggered": bool(public_rebuild_summary.get("triggered")),
            "public_rebuild_summary": public_rebuild_summary,
            "table_follow_up_count": table_gap_summary.get("table_follow_up_count", 0),
            "rendered_high_value_table_count": table_gap_summary.get("rendered_high_value_table_count", 0),
            "evidence_analysis_summary": evidence_analysis_summary,
            "evidence_gap_count": len(evidence_gap_ledger),
            "chapter_evidence_diagnostics_count": len(chapter_evidence_diagnostics),
            "analysis_depth_quality": analysis_depth_quality,
            "analysis_stage_diagnostics": analysis_stage_diagnostics,
            "claim_binding_feedback_summary": claim_binding_feedback_summary,
            "chapter_narrative": chapter_narrative_diagnostics,
            "stage_quality_card": stage_quality_card,
        },
    }


def run_writer_agent(
    *,
    query: str = "",
    child_outputs: Optional[Dict[str, Dict[str, Any]]] = None,
    evidence_pool: Optional[Sequence[Dict[str, Any]]] = None,
    evidence_package: Optional[Dict[str, Any]] = None,
    structured_analysis: Optional[Dict[str, Any]] = None,
    report_plan: Optional[Dict[str, Any]] = None,
    layout_plan: Optional[Dict[str, Any]] = None,
    report_blueprint: Optional[Dict[str, Any]] = None,
    chapter_packages: Optional[Sequence[Dict[str, Any]]] = None,
    analytics_outputs: Optional[Sequence[Dict[str, Any]]] = None,
    table_packages: Optional[Sequence[Dict[str, Any]]] = None,
    decision_package: Optional[Dict[str, Any]] = None,
    risk_package: Optional[Dict[str, Any]] = None,
    appendix_package: Optional[Dict[str, Any]] = None,
    source_registry: Optional[Sequence[Dict[str, Any]]] = None,
    search_task_schedule: Optional[Dict[str, Any]] = None,
    lane_coverage: Optional[Dict[str, Any]] = None,
    llm_client: Any = None,
) -> WriterAgentState:
    try:
        writer_report = build_writer_report(
            query=query,
            child_outputs=child_outputs,
            evidence_pool=evidence_pool,
            evidence_package=evidence_package,
            structured_analysis=structured_analysis,
            report_plan=report_plan,
            layout_plan=layout_plan,
            report_blueprint=report_blueprint,
            chapter_packages=chapter_packages,
            analytics_outputs=analytics_outputs,
            table_packages=table_packages,
            decision_package=decision_package,
            risk_package=risk_package,
            appendix_package=appendix_package,
            source_registry=source_registry,
            search_task_schedule=search_task_schedule,
            lane_coverage=lane_coverage,
            llm_client=llm_client,
        )
        return {
            "query": query,
            "writer_report": writer_report,
            "answer_text": str(writer_report.get("report_markdown") or ""),
            "raw_output": {
                "query": query,
                "writer_report": writer_report,
                "writer": {
                    "type": "writer_report",
                    "source": "split_final_writer_pipeline",
                    "uses_llm": bool(llm_client),
                    "source_count": writer_report.get("source_count", 0),
                    "estimated_chars": writer_report.get("estimated_chars", 0),
                    "qa_passed": _as_dict(writer_report.get("qa_result")).get("passed"),
                    "package_contract_passed": _as_dict(writer_report.get("package_quality_report")).get("passed"),
                },
                "debug_snapshot": writer_report.get("debug_snapshot"),
            },
            "metadata": {
                "agent_name": AGENT_NAME,
                "agent_description": AGENT_DESCRIPTION,
                "agent_stage": "write_report_compat_orchestrator",
                "handoff_ready": bool(writer_report.get("report_markdown")),
            },
        }
    except Exception as exc:
        return {
            "query": query,
            "errors": [str(exc)],
            "answer_text": "",
            "raw_output": {"query": query, "writer": {"type": "writer_report", "source": "failed", "error": str(exc)}},
            "metadata": {
                "agent_name": AGENT_NAME,
                "agent_description": AGENT_DESCRIPTION,
                "agent_stage": "write_report_compat_orchestrator",
                "handoff_ready": False,
            },
        }


def writer_agent_tool(payload: Dict[str, Any]) -> Dict[str, Any]:
    state = run_writer_agent(
        query=str(payload.get("query") or ""),
        child_outputs=_as_dict(payload.get("child_outputs")),
        evidence_pool=_as_list(payload.get("evidence_pool")),
        evidence_package=_as_dict(payload.get("evidence_package")),
        structured_analysis=_as_dict(payload.get("structured_analysis")),
        report_plan=_as_dict(payload.get("report_plan")),
        layout_plan=_as_dict(payload.get("layout_plan")),
        report_blueprint=_as_dict(payload.get("report_blueprint")),
        chapter_packages=_as_list(payload.get("chapter_packages")),
        analytics_outputs=_as_list(payload.get("analytics_outputs")),
        table_packages=_as_list(payload.get("table_packages")),
        decision_package=_as_dict(payload.get("decision_package")),
        risk_package=_as_dict(payload.get("risk_package")),
        appendix_package=_as_dict(payload.get("appendix_package")),
        source_registry=_as_list(payload.get("source_registry")),
        search_task_schedule=_as_dict(payload.get("search_task_schedule")),
        lane_coverage=_as_dict(payload.get("lane_coverage")),
    )
    return _as_dict(state.get("writer_report"))


def create_writer_agent_tool():
    from langchain_core.tools import tool

    @tool("writer_agent", description=AGENT_DESCRIPTION)
    def _writer_agent(payload: Dict[str, Any]) -> Dict[str, Any]:
        return writer_agent_tool(payload)

    return _writer_agent


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=AGENT_DESCRIPTION)
    parser.add_argument("query", nargs="*", help="research topic")
    parser.add_argument("--input-json", help="JSON file containing writer inputs")
    args = parser.parse_args()
    payload: Dict[str, Any] = {}
    if args.input_json:
        with open(args.input_json, "r", encoding="utf-8") as file:
            payload = json.load(file)
    query = " ".join(args.query).strip() or str(payload.get("query") or "")
    state = run_writer_agent(
        query=query,
        child_outputs=_as_dict(payload.get("child_outputs")),
        evidence_pool=_as_list(payload.get("evidence_pool")),
        evidence_package=_as_dict(payload.get("evidence_package")),
        structured_analysis=_as_dict(payload.get("structured_analysis")),
        report_plan=_as_dict(payload.get("report_plan")),
        layout_plan=_as_dict(payload.get("layout_plan")),
    )
    print(state.get("answer_text") or json.dumps(state, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
