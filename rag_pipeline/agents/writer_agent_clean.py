from __future__ import annotations

import json
import copy
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, TypedDict

try:
    from .analytics import run_analytics_agents
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
except Exception:  # pragma: no cover - direct script mode fallback
    from analytics import run_analytics_agents  # type: ignore
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


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _package_warning_types(package_quality_report: Dict[str, Any]) -> List[str]:
    warnings = [
        _as_dict(item).get("type")
        for item in _as_list(package_quality_report.get("warnings"))
        if isinstance(item, dict)
    ]
    return [str(item) for item in warnings if str(item or "").strip()]


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
        requested = _env_int("REPORT_MAX_BODY_TABLES", 6, min_value=0, max_value=50)
    else:
        requested = 8
    hard_limit = _env_int("REPORT_HARD_MAX_BODY_TABLES", 12, min_value=0, max_value=50)
    return max(0, min(requested, hard_limit))


def _per_chapter_table_budget() -> int:
    return _env_int("REPORT_MAX_BODY_TABLES_PER_CHAPTER", 3, min_value=0, max_value=20)


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
    counts = {
        key: _safe_len(package.get(key))
        for key in (
            "core_evidence",
            "supporting_evidence",
            "table_evidence",
            "clue_evidence",
            "appendix_evidence",
            "evidence_items",
        )
        if _safe_len(package.get(key))
    }
    samples: List[Dict[str, Any]] = []
    for collection in ("core_evidence", "table_evidence", "supporting_evidence", "clue_evidence"):
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
            "sample_evidence": samples[:2],
            "missing_evidence": _compact_issue_list(_as_list(package.get("missing_evidence")), limit=3),
            "evidence_quality_summary": _compact_mapping(quality, list(quality.keys()), text_chars=120),
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
    public_units = [unit for unit in argument_units if isinstance(unit, dict) and unit.get("public_render") and not unit.get("omit_from_report")]
    public_chapters = [chapter for chapter in chapter_packages if isinstance(chapter, dict) and not chapter.get("omit_from_report")]
    return {
        "chapter_count": len(public_chapters),
        "evidence_package_count": len([item for item in chapter_evidence_packages if isinstance(item, dict)]),
        "analytics_output_count": len([item for item in analytics_outputs if isinstance(item, dict)]),
        "micro_layout_count": len([item for item in micro_layouts if isinstance(item, dict)]),
        "table_package_count": len([item for item in table_packages if isinstance(item, dict)]),
        "rendered_table_count": len(rendered_tables),
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
        if str(item.get("claim_status") or "").strip().lower() == "directional":
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
) -> List[Dict[str, Any]]:
    blockers: List[Dict[str, Any]] = []
    text = str(markdown or "")
    strict_gate = _strict_delivery_gate()
    for pattern in HARD_DELIVERY_FORBIDDEN_PATTERNS:
        if re.search(pattern, text, flags=re.I):
            if not strict_gate and pattern in SOFT_DELIVERY_GAP_PATTERNS:
                continue
            blockers.append({"type": "forbidden_public_text", "pattern": pattern})

    deep_eval = _as_dict(qa_result.get("deep_evaluation"))
    block_deep_gaps = (
        strict_gate
        or _env_flag("REPORT_BLOCK_ON_DEEP_GAPS", False)
    )
    if block_deep_gaps:
        for gap in _as_list(deep_eval.get("blocking_gaps")):
            gap = _as_dict(gap)
            gap_type = str(gap.get("type") or "").strip()
            blockers.append({"type": gap_type or "deep_report_blocking_gap", "detail": gap})

    coverage_rows = [item for item in list(coverage_matrix or []) if isinstance(item, dict)]
    gap_rows = _coverage_gap_rows(coverage_rows)
    if coverage_rows and not any(bool(item.get("decision_ready")) for item in coverage_rows):
        if strict_gate:
            blockers.append({"type": "no_decision_ready_hypotheses", "count": len(coverage_rows)})
        elif not _coverage_has_usable_signal(coverage_rows):
            blockers.append({"type": "no_publishable_evidence", "count": len(coverage_rows)})
    if _env_flag("REPORT_BLOCK_ON_PROOF_GAPS", strict_gate):
        hard_gap_types = {"insufficient_ab_sources", "metric_evidence_missing", "case_evidence_missing", "counter_evidence_missing"}
        hard_gap_rows = [
            item
            for item in gap_rows
            if hard_gap_types.intersection({str(gap or "") for gap in _as_list(item.get("blocking_gaps"))})
        ]
        if hard_gap_rows:
            blockers.append({"type": "core_proof_gaps", "count": len(hard_gap_rows), "examples": hard_gap_rows[:5]})

    warning_types = set(_package_warning_types(package_quality_report))
    if _env_flag("REPORT_BLOCK_LOW_AB_CORE_COVERAGE", strict_gate) and "low_ab_core_coverage" in warning_types:
        blockers.append({"type": "low_ab_core_coverage"})
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
    if not bool(qa_result.get("passed")) and _env_flag("REPORT_BLOCK_ON_QA_FAILURE", strict_gate):
        blockers.append({"type": "qa_not_passed", "quality_score": qa_result.get("quality_score")})

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
    **_: Any,
) -> Dict[str, Any]:
    return run_final_writer_agent(
        query=query,
        report_blueprint=report_blueprint,
        chapter_packages=chapter_packages,
        table_packages=table_packages,
        decision_package=decision_package,
        risk_package=risk_package,
        appendix_package=appendix_package,
        source_registry=source_registry,
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
    argument_units = list(argument_units or run_claim_builder_agent(
        chapter_evidence_packages=chapter_evidence_packages,
        micro_layouts=micro_layouts,
        structured_analysis=structured_analysis,
        llm_client=llm_client,
    ))
    chapter_packages = list(chapter_packages or run_chapter_argument_agent(
        report_blueprint=report_blueprint,
        micro_layouts=micro_layouts,
        argument_units=argument_units,
        table_packages=table_packages,
        chapter_evidence_packages=chapter_evidence_packages,
        llm_client=llm_client,
    ))
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
    target_body_chars = _env_int("REPORT_TARGET_BODY_CHARS", 20000, min_value=0, max_value=100000)
    public_chapter_packages = _expand_chapter_packages_for_body_target(public_chapter_packages, target_chars=target_body_chars)
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
    )
    writer_output["report_markdown"] = sanitize_public_markdown(writer_output.get("report_markdown") or "")
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
        metric_normalization_table=metric_normalization_table,
        coverage_matrix=coverage_matrix,
        missing_proof_standards=missing_proof_standards,
        analytics_outputs=analytics_outputs,
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
                metric_normalization_table=metric_normalization_table,
                coverage_matrix=coverage_matrix,
                missing_proof_standards=missing_proof_standards,
                analytics_outputs=analytics_outputs,
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
    package_warning_types = _package_warning_types(package_quality_report)
    package_warning_blocked = (
        _env_flag("QA_BLOCK_LOW_AB_CORE_COVERAGE", False)
        and "low_ab_core_coverage" in package_warning_types
    )
    delivery_blockers = _hard_delivery_blockers(
        markdown=public_markdown,
        qa_result=qa_result,
        package_quality_report=package_quality_report,
        coverage_matrix=coverage_matrix,
    )
    if delivery_blockers:
        blocked_markdown = _render_not_ready_report(
            query=query,
            report_blueprint=report_blueprint,
            blockers=delivery_blockers,
            qa_result=qa_result,
            coverage_matrix=coverage_matrix,
            missing_proof_standards=missing_proof_standards,
            search_task_schedule=search_task_schedule,
            lane_coverage=lane_coverage,
        )
        writer_output = {
            **writer_output,
            "report_markdown": "",
            "blocked_report_markdown": blocked_markdown,
            "estimated_chars": 0,
            "estimated_body_chars": 0,
            "blocked_draft_chars": len(public_markdown),
        }
        public_markdown = ""
        report_status = "not_ready"
        message = "公开资料条件不足，正式报告未生成。"
    else:
        report_status = "final" if qa_result.get("passed") and package_passed and not package_warning_blocked and not has_internal_gap_language(public_markdown) else "review_required"
        message = "" if report_status == "final" else "正式报告仍需人工复核后发布。"
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
    return {
        **writer_output,
        "report_type": report_blueprint.get("report_family") or select_report_layout(report_plan).report_type,
        "report_status": report_status,
        "message": message,
        "delivery_blockers": delivery_blockers,
        "delivery_gate_mode": _delivery_gate_mode(),
        "skip_reformatter": report_status == "not_ready",
        "package_warning_blocked": package_warning_blocked,
        "report_blueprint": report_blueprint,
        "search_tasks": _as_list(research_plan.get("search_tasks")),
        "search_task_schedule": search_task_schedule,
        "lane_coverage": lane_coverage,
        "layout_plan": layout,
        "report_insight_package": report_insight_package,
        "evidence_refinement_plan": evidence_refinement_plan,
        "research_proof_profile": research_proof_profile,
        "mandatory_proof_checks": mandatory_proof_checks,
        **artifact_payload,
        "source_registry": list(rendered_source_registry or []),
        "footnotes": rendered_footnotes,
        "validation": {
            **qa_result,
            "ok": bool(writer_output.get("report_markdown")) and report_status != "not_ready",
        },
        "qa_result": qa_result,
        "required_followups": _as_list(_as_dict(qa_result.get("deep_evaluation")).get("required_followups")),
        "package_quality_report": package_quality_report,
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
