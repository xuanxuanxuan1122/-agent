from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Sequence

try:
    from .public_report_sanitizer import INTERNAL_GAP_PATTERNS, SAFE_PUBLIC_TERMS
    from .research_proof_registry import research_maturity
except Exception:  # pragma: no cover - direct script mode fallback
    from public_report_sanitizer import INTERNAL_GAP_PATTERNS, SAFE_PUBLIC_TERMS  # type: ignore
    from research_proof_registry import research_maturity  # type: ignore


AGENT_NAME = "qa_agent"
AGENT_DESCRIPTION = "QA Agent. Independent quality gate for final report packages."
PUBLIC_EV_ID_PATTERN = r"(?<![A-Za-z0-9_])EV-\d+(?:-[A-Za-z0-9]+)?"
BODY_SOURCE_TABLE_HEADER_RE = re.compile(
    r"\|\s*[^|\n]*(?:引用|来源|资料来源|来源等级|判断用途|evidence_refs|evidence|source|ref)[^|\n]*\s*\|",
    re.I,
)


INTERNAL_LABELS = [
    "章节判断",
    "关键事实速览",
    "证据深读",
    "原文事实",
    "行业形势含义",
    "投资/产品判断",
    "与上下章节的联动",
    "战略含义与行动建议",
    "进入综合决策章的变量",
    "本章核心判断",
    "本章结论",
    "本章小结",
    "图表解读",
    "资料来源：",
    "报告使用方式",
    "核心判断",
    "关键判断",
    "证据依据",
    "传导链条",
    "判断边界",
    "决策含义",
    "行动含义",
    "判断含义",
    "本章综合分析",
    "机制拆解与变量联动",
    "反证、边界与结论失效条件",
    "决策含义与后续观察优先级",
    "主要结论：",
    "资料支撑：",
]
INTERNAL_LABELS.extend(
    [
        "章节判断",
        "关键事实速览",
        "证据深读",
        "原文事实",
        "可引用事实",
        "机制与边界",
        "进入综合决策章的变量",
        "本章核心判断",
        "本章结论",
        "全球口径",
        "中国口径",
        "增速口径",
        "核心判断",
        "机制拆解",
        "反证边界",
        "决策含义",
    ]
)

EVIDENCE_LISTING_PATTERNS = [
    r"证据提供了[^。]{0,80}证据",
    r"后续判断的重点是确认该口径",
    r"同类数据存在口径差异",
    r"进入综合决策章的变量",
    r"已有可核验证据",
    r"已有可验证证据",
    r"该证据可作为判断输入",
    r"需结合来源等级、时间范围和口径边界复核",
    r"尚未发现足以推翻该判断的反向证据",
    r"关联证据[:：]",
    PUBLIC_EV_ID_PATTERN,
    r"\bevidence_cards?\b",
    r"当前卡片",
    r"本章应写成",
    r"本章可以作为",
    r"本章可作为",
    r"正文\s*只能\s*写成",
    r"本章\s*只能\s*写成",
    r"本章\s*可\s*写成",
    r"本章\s*应\s*写成",
    r"本章\s*仍需\s*连续观察",
    r"建议避免",
    r"建议在后续版本中补充",
    r"建议写成",
    r"适合写成",
    r"本章可用来源约\d+条",
    r"A/B层级来源约\d+条",
    r"来源层级分布为",
    r"本章写作时应",
    r"当前最直接的支持点是",
    r"当前可用于判断的事实组合包括",
    r"不作为每章正文",
    r"正文只保留",
]
PUBLIC_FORBIDDEN_PATTERNS = [
    r"证据不足",
    r"暂无可核验数据",
    r"低置信方向判断",
    r"不能作为确定性结论",
    r"建议后续补充调研",
    r"A/B\s*级来源不足",
    r"权威来源交叉验证",
    r"待验证事项",
    r"needs_corroboration",
    r"当前表格证据不足",
    r"核心判断[:：]",
    r"关键判断[:：]",
    r"证据依据[:：]",
    r"传导链条[:：]",
    r"判断边界[:：]",
    r"决策含义[:：]",
    r"行动含义[:：]",
    r"判断含义[:：]",
    r"本章综合分析",
    r"机制拆解与变量联动",
    r"反证、边界与结论失效条件",
    r"决策含义与后续观察优先级",
    r"主要结论[:：]",
    r"资料支撑[:：]",
    r"关联证据[:：]",
    PUBLIC_EV_ID_PATTERN,
    r"\bevidence_cards?\b",
    r"当前卡片",
    r"本章应写成",
    r"本章可以作为",
    r"本章可作为",
    r"建议写成",
    r"适合写成",
    r"本章可用来源约\d+条",
    r"A/B层级来源约\d+条",
    r"来源层级分布为",
    r"本章写作时应",
    r"当前最直接的支持点是",
    r"当前可用于判断的事实组合包括",
    r"不作为每章正文",
    r"正文只保留",
    r"evidence_refs",
    r"claim_status",
    r"render_blocks",
    *INTERNAL_GAP_PATTERNS,
]
PUBLIC_FORBIDDEN_PATTERNS.extend(
    [
        r"章节判断",
        r"关键事实速览",
        r"证据深读",
        r"本章结论",
        r"全球口径",
        r"中国口径",
        r"增速口径",
        r"可引用事实",
        r"机制与边界",
        r"进入综合决策章的变量",
        r"核心判断[:：]",
        r"机制拆解",
        r"反证边界",
        r"决策含义[:：]",
    ]
)
PUBLIC_FORBIDDEN_PATTERNS.extend(
    [
        r"材料中最有解释力的事实组合是",
        r"当前事实组合是",
        r"对应的章节结论是",
        r"影响路径可以概括为",
        r"后续变化主要集中在",
        r"放在章节顺序中看",
        r"证据覆盖矩阵",
        r"coverage_matrix",
        r"actual_ab_sources",
        r"required_ab_sources",
        r"blocking_gaps",
        r"insufficient_ab_sources",
        r"case_evidence_missing",
        r"counter_evidence_missing",
        r"metric_scope_period_unit_incomplete",
    ]
)


PUBLIC_FORBIDDEN_PATTERNS.extend(
    [
        r"\bch_\d{1,3}\b",
        r"\bpolicy_summary\b",
        r"\bpolicy_impact\b",
        r"第\s*\d+\s*轮",
        r"\bcoverage_matrix\b",
        r"\bsource_registry\b",
    ]
)

SAFE_PUBLIC_TERMS = [
    "全球口径",
    "中国口径",
    "增速口径",
    "机制与边界",
    "机制拆解",
    "决策含义",
    "可引用事实",
]


def _mask_safe_public_terms(text: str) -> str:
    value = str(text or "")
    for term in SAFE_PUBLIC_TERMS:
        value = value.replace(term, "")
    return value


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, min_value: int = 0, max_value: int = 100) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return default
    return max(min_value, min(max_value, value))


def _count_value(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _strict_quality_mode() -> bool:
    mode = str(os.getenv("REPORT_QUALITY_MODE") or os.getenv("QUALITY_MODE") or "balanced").strip().lower()
    if mode in {"speed", "fast", "loose", "draft", "balanced", "quick_market_scan"}:
        return False
    if mode in {"strict", "deep_strict", "due_diligence", "investment_due_diligence"}:
        return True
    raw = os.getenv("STRICT_EVIDENCE_MODE")
    if raw is None:
        return False
    return str(raw).strip().lower() in {"1", "true", "yes", "on", "strict"}


def _is_deep_report_family(report_blueprint: Dict[str, Any]) -> bool:
    family = str(_as_dict(report_blueprint).get("report_family") or _as_dict(report_blueprint).get("report_type") or "").strip().lower()
    return family == "industry_deep_report" or "deep" in family


def _table_count(markdown: str) -> int:
    return len(re.findall(r"(?m)^\|\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|\s*$", str(markdown or "")))


def _is_table_separator(line: str) -> bool:
    return bool(re.match(r"^\|\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|\s*$", str(line or "").strip()))


def _body_table_has_source_header(markdown: str) -> bool:
    lines = str(markdown or "").splitlines()
    for index, line in enumerate(lines[:-1]):
        if "|" not in line or not BODY_SOURCE_TABLE_HEADER_RE.search(line):
            continue
        if _is_table_separator(lines[index + 1]):
            return True
    return False


def _table_fatigue_warnings(markdown: str, table_packages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    warnings: List[Dict[str, Any]] = []
    body_table_count = _table_count(markdown)
    max_body_tables = _env_int("REPORT_MAX_BODY_TABLES", 6, min_value=0, max_value=20)
    max_rows = _env_int("REPORT_MAX_BODY_TABLE_ROWS", 6, min_value=1, max_value=20)
    per_chapter_limit = _env_int("REPORT_MAX_BODY_TABLES_PER_CHAPTER", 1, min_value=0, max_value=6)
    if body_table_count > max_body_tables:
        warnings.append({"type": "table_fatigue_body_table_count", "actual": body_table_count, "limit": max_body_tables})
    by_chapter: Dict[str, int] = {}
    for table in table_packages:
        if not isinstance(table, dict) or not table.get("should_render") or table.get("appendix_only"):
            continue
        chapter_id = str(table.get("chapter_id") or "").strip() or "unknown"
        by_chapter[chapter_id] = by_chapter.get(chapter_id, 0) + 1
        rows = [row for row in _as_list(table.get("rows")) if isinstance(row, dict)]
        if len(rows) > max_rows:
            warnings.append({"type": "table_fatigue_table_too_long", "table_id": table.get("table_id"), "rows": len(rows), "limit": max_rows})
    for chapter_id, count in sorted(by_chapter.items()):
        if count > per_chapter_limit:
            warnings.append({"type": "table_fatigue_chapter_table_count", "chapter_id": chapter_id, "actual": count, "limit": per_chapter_limit})
    return warnings


def _analytics_quality(analytics_outputs: Optional[Sequence[Dict[str, Any]]]) -> Dict[str, Any]:
    outputs = [item for item in list(analytics_outputs or []) if isinstance(item, dict)]
    tables: List[Dict[str, Any]] = []
    calculations: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    for output in outputs:
        for table in _as_list(output.get("tables")):
            if isinstance(table, dict):
                tables.append(table)
        for calculation in _as_list(output.get("calculations")):
            if isinstance(calculation, dict):
                calculations.append(calculation)

    for table in tables:
        rows = [_as_dict(row) for row in _as_list(table.get("rows")) if isinstance(row, dict)]
        if rows and not any(_as_list(row.get("evidence_refs")) for row in rows):
            errors.append(
                {
                    "type": "analytics_table_missing_evidence_refs",
                    "table_id": table.get("table_id"),
                    "agent": table.get("agent"),
                }
            )

    for calculation in calculations:
        if str(calculation.get("type") or "").strip().lower() != "cagr":
            continue
        missing = [
            key
            for key in ("start_value", "end_value", "periods", "result")
            if calculation.get(key) in (None, "")
        ]
        if missing:
            errors.append(
                {
                    "type": "analytics_cagr_incomplete",
                    "calculation_id": calculation.get("calculation_id"),
                    "missing": missing,
                }
            )
            continue
        try:
            start_value = float(calculation.get("start_value"))
            end_value = float(calculation.get("end_value"))
            periods = float(calculation.get("periods"))
        except (TypeError, ValueError):
            errors.append(
                {
                    "type": "analytics_cagr_non_numeric",
                    "calculation_id": calculation.get("calculation_id"),
                }
            )
            continue
        if start_value <= 0 or end_value <= 0 or periods <= 0:
            errors.append(
                {
                    "type": "analytics_cagr_invalid_range",
                    "calculation_id": calculation.get("calculation_id"),
                    "start_value": start_value,
                    "end_value": end_value,
                    "periods": periods,
                }
            )
        if len(_as_list(calculation.get("evidence_refs"))) < 2:
            warnings.append(
                {
                    "type": "analytics_cagr_weak_traceability",
                    "calculation_id": calculation.get("calculation_id"),
                }
            )

    return {
        "output_count": len(outputs),
        "table_count": len(tables),
        "calculation_count": len(calculations),
        "errors": errors,
        "warnings": warnings,
    }


def validate_report_narrative_quality(markdown: str) -> Dict[str, Any]:
    issues: List[Dict[str, Any]] = []
    text = str(markdown or "")
    for pattern in EVIDENCE_LISTING_PATTERNS:
        if re.search(pattern, text):
            issues.append({"type": "evidence_listing_style", "pattern": pattern})
            break
    return {"passed": not issues, "issues": issues}


def validate_no_internal_gap_language(markdown: str) -> List[Dict[str, Any]]:
    errors: List[Dict[str, Any]] = []
    scan_text = _mask_safe_public_terms(str(markdown or ""))
    for pattern in PUBLIC_FORBIDDEN_PATTERNS:
        hits = re.findall(pattern, scan_text, re.I)
        if hits:
            errors.append(
                {
                    "type": "internal_gap_language_leaked",
                    "severity": "blocking",
                    "pattern": pattern,
                    "count": len(hits),
                }
            )
    return errors


def evaluate_deep_report(
    *,
    report_markdown: str,
    report_blueprint: Dict[str, Any],
    chapter_packages: Sequence[Dict[str, Any]],
    table_packages: Sequence[Dict[str, Any]],
    decision_package: Dict[str, Any],
    risk_package: Dict[str, Any],
    search_task_schedule: Dict[str, Any],
    lane_coverage: Dict[str, Any],
    metric_normalization_table: Optional[Sequence[Dict[str, Any]]] = None,
    analytics_outputs: Optional[Sequence[Dict[str, Any]]] = None,
    coverage_matrix: Optional[Sequence[Dict[str, Any]]] = None,
    missing_proof_standards: Optional[Sequence[Dict[str, Any]]] = None,
    evidence_health_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    text = str(report_markdown or "")
    body_text = re.split(r"\n##\s*(?:\u9644\u5f55|附錄|研究口径|研究口徑)", text, maxsplit=1)[0]
    target_body_chars = _env_int("REPORT_TARGET_BODY_CHARS", 20000, min_value=0, max_value=100000)
    strict_mode = _strict_quality_mode()
    deep_report = _is_deep_report_family(report_blueprint)
    target_body_blocking = strict_mode or deep_report or _env_flag("REPORT_TARGET_BODY_CHARS_BLOCKING", True)
    public_chapters = [chapter for chapter in chapter_packages if isinstance(chapter, dict) and not chapter.get("omit_from_report")]
    sections = [
        section
        for chapter in public_chapters
        for section in _as_list(chapter.get("sections"))
        if isinstance(section, dict)
    ]
    evidence_ref_count = sum(len(_as_list(section.get("evidence_refs"))) for section in sections)
    causal_sections = [
        section
        for section in sections
        if re.search(r"因为|由于|导致|传导|机制|所以|therefore|because|drive|lead", str(section.get("reasoning") or ""), re.I)
    ]
    counter_sections = [section for section in sections if str(section.get("counter_evidence") or "").strip()]
    causal_seen = {id(section) for section in causal_sections}
    for section in sections:
        mechanism_text = " ".join(
            str(section.get(key) or "")
            for key in ("reasoning", "mechanism", "claim")
        )
        if id(section) not in causal_seen and re.search(
            r"因为|由于|导致|传导|机制|所以|驱动|影响|转化|推导|therefore|because|drive|lead",
            mechanism_text,
            re.I,
        ):
            causal_sections.append(section)
            causal_seen.add(id(section))
    actionable_sections = [section for section in sections if str(section.get("actionable") or section.get("decision_implication") or "").strip()]
    rendered_tables = [
        table
        for table in table_packages
        if isinstance(table, dict)
        and table.get("should_render")
        and not table.get("appendix_only")
        and not _as_list(table.get("validation_errors"))
    ]
    analytics_quality = _analytics_quality(analytics_outputs)
    analytics_table_count = int(analytics_quality.get("table_count") or 0)
    analytics_calculation_count = int(analytics_quality.get("calculation_count") or 0)
    metric_rows = [item for item in list(metric_normalization_table or []) if isinstance(item, dict)]
    complete_metric_rows = [item for item in metric_rows if not _as_list(item.get("missing_fields"))]
    coverage_rows = [item for item in list(coverage_matrix or []) if isinstance(item, dict)]
    decision_ready_rows = [item for item in coverage_rows if bool(item.get("decision_ready"))]
    proof_gap_rows = [
        item
        for item in (list(missing_proof_standards or []) or coverage_rows)
        if isinstance(item, dict) and _as_list(item.get("blocking_gaps"))
    ]
    mandatory_proof_checks = [
        check
        for row in coverage_rows
        if isinstance(row, dict)
        for check in _as_list(row.get("mandatory_proof_checks"))
        if isinstance(check, dict)
    ]
    missing_mandatory_proofs = [
        check
        for check in mandatory_proof_checks
        if check.get("status") == "missing" and check.get("required")
    ]
    maturity = research_maturity(
        report_text=text,
        coverage_rows=coverage_rows,
        table_count=len(rendered_tables),
        metric_rows=len(metric_rows),
        complete_metric_rows=len(complete_metric_rows),
    )
    risk_items = _as_list(risk_package.get("risk_items"))
    decision_items = _as_list(decision_package.get("decision_items")) or _as_list(decision_package.get("actions")) or _as_list(decision_package.get("recommendations"))
    dropped_count = int(search_task_schedule.get("dropped_count") or len(_as_list(search_task_schedule.get("dropped_tasks"))))
    dropped_blocking_count = int(
        search_task_schedule.get("dropped_blocking_count")
        or _as_dict(search_task_schedule.get("dropped_summary")).get("blocking_dropped_count")
        or 0
    )
    failed_lanes = [
        lane
        for lane, coverage in lane_coverage.items()
        if int(_as_dict(coverage).get("scheduled") or 0) and int(_as_dict(coverage).get("succeeded") or 0) == 0
    ]
    scheduled_lanes = [
        lane
        for lane, coverage in lane_coverage.items()
        if int(_as_dict(coverage).get("scheduled") or 0)
    ]
    page_zero_lanes = [
        lane
        for lane, coverage in lane_coverage.items()
        if int(_as_dict(coverage).get("scheduled") or 0)
        and int(_as_dict(coverage).get("page_results") or 0) == 0
    ]
    timeout_lanes = [
        lane
        for lane, coverage in lane_coverage.items()
        if int(_as_dict(coverage).get("scheduled") or 0)
        and int(_as_dict(coverage).get("timed_out_task_count") or _as_dict(coverage).get("timed_out") or 0) > 0
        and int(_as_dict(coverage).get("usable_source_count") or _as_dict(coverage).get("key_sources") or 0) == 0
    ]
    min_depth_score = _env_int("QA_DEEP_MIN_DEPTH_SCORE", 75 if deep_report else 70, min_value=0, max_value=100)
    block_dropped_tasks = strict_mode or deep_report or _env_flag("QA_DEEP_BLOCK_DROPPED_TASKS", True)
    block_lane_failures = strict_mode or deep_report or _env_flag("QA_DEEP_BLOCK_LANE_FAILURES", True)
    block_page_results_zero = strict_mode or _env_flag("QA_DEEP_BLOCK_PAGE_RESULTS_ZERO", True)
    block_any_page_zero_lane = strict_mode or _env_flag("QA_DEEP_BLOCK_ANY_PAGE_ZERO_LANE", True)
    block_proof_gaps = strict_mode or deep_report or _env_flag("QA_DEEP_BLOCK_PROOF_GAPS", True)
    block_structure_gaps = strict_mode or _env_flag("QA_DEEP_BLOCK_STRUCTURE_GAPS", False)
    min_core_ab_per_chapter = _env_int("QA_DEEP_MIN_CORE_AB_PER_CHAPTER", 2 if deep_report else 3, min_value=1, max_value=20)
    coverage_ready_bonus = int(8 * (len(decision_ready_rows) / max(len(coverage_rows), 1))) if coverage_rows else 0

    scores = {
        "question_definition": 10 if report_blueprint.get("report_family") and public_chapters else 6,
        "evidence_strength": min(20, evidence_ref_count * 3 + coverage_ready_bonus),
        "data_completeness": 15 if rendered_tables or complete_metric_rows else (8 if metric_rows or re.search(r"\d|%|亿元|万|台|CAGR|同比|增速|市场规模", text, re.I) else 4),
        "mechanism": min(15, len(causal_sections) * 5),
        "counter_evidence": min(15, len(counter_sections) * 5 + (3 if risk_items else 0)),
        "decision_value": min(15, len(actionable_sections) * 4 + (5 if decision_items else 0)),
        "writing_quality": 10 if text.strip() and not validate_no_internal_gap_language(text) else 4,
    }
    blocking_gaps: List[Dict[str, Any]] = []
    if dropped_count > 0 and block_dropped_tasks and (strict_mode or not deep_report or dropped_blocking_count > 0):
        blocking_gaps.append({"type": "search_tasks_dropped", "dropped_count": dropped_count, "dropped_blocking_count": dropped_blocking_count})
    if failed_lanes and block_lane_failures:
        blocking_gaps.append({"type": "iqs_lane_no_success", "lanes": failed_lanes})
    if timeout_lanes and block_lane_failures:
        blocking_gaps.append({"type": "iqs_lane_timeout_without_signal", "lanes": timeout_lanes})
    if page_zero_lanes and block_any_page_zero_lane:
        blocking_gaps.append({"type": "page_results_zero", "lanes": page_zero_lanes})
    elif page_zero_lanes and len(page_zero_lanes) >= max(1, len(scheduled_lanes)) and block_page_results_zero:
        blocking_gaps.append({"type": "page_results_zero", "lanes": page_zero_lanes})
    if strict_mode and public_chapters and evidence_ref_count < len(public_chapters):
        blocking_gaps.append({"type": "insufficient_core_evidence_refs", "required": len(public_chapters), "actual": evidence_ref_count})
    structure_followups: List[Dict[str, Any]] = []
    if sections and len(counter_sections) < max(1, len(sections) // 2):
        gap = {"type": "counter_evidence_weak", "required": max(1, len(sections) // 2), "actual": len(counter_sections)}
        structure_followups.append({**gap, "priority": "medium"})
        if block_structure_gaps:
            blocking_gaps.append(gap)
    if sections and len(causal_sections) < max(1, len(sections) // 2):
        gap = {"type": "mechanism_explanation_weak", "required": max(1, len(sections) // 2), "actual": len(causal_sections)}
        structure_followups.append({**gap, "priority": "medium"})
        if block_structure_gaps:
            blocking_gaps.append(gap)
    if proof_gap_rows and block_proof_gaps:
        blocking_gaps.append(
            {
                "type": "missing_proof_standards",
                "count": len(proof_gap_rows),
                "examples": proof_gap_rows[:5],
            }
        )
    length_followup: Dict[str, Any] = {}
    if target_body_chars and len(body_text) < target_body_chars:
        length_followup = {
            "type": "report_body_below_target_chars",
            "required": target_body_chars,
            "actual": len(body_text),
            "priority": "high",
        }
        if target_body_blocking:
            blocking_gaps.append(length_followup)
    total_ab_sources = sum(int(_as_dict(item).get("actual_ab_sources") or 0) for item in coverage_rows)
    if strict_mode and coverage_rows and total_ab_sources <= 0:
        blocking_gaps.append({"type": "no_ab_sources_for_core_hypotheses"})
    counter_missing = [
        item
        for item in coverage_rows
        if "counter_evidence_missing" in _as_list(_as_dict(item).get("blocking_gaps"))
    ]
    if strict_mode and counter_missing:
        structure_followups.append(
            {
                "type": "chapter_counter_evidence_advisory",
                "count": len(counter_missing),
                "examples": counter_missing[:5],
                "priority": "medium",
            }
        )
    weak_source_chapters = []
    for chapter in public_chapters:
        quality = _as_dict(chapter.get("evidence_quality_summary"))
        if not quality:
            continue
        level_dist = _as_dict(quality.get("source_level_distribution"))
        core_ab = int(quality.get("core_ab_source_count") or 0)
        only_c_or_lower = bool(level_dist) and not (int(level_dist.get("A") or 0) + int(level_dist.get("B") or 0))
        if core_ab < min_core_ab_per_chapter or only_c_or_lower:
            weak_source_chapters.append(
                {
                    "chapter_id": chapter.get("chapter_id"),
                    "chapter_title": chapter.get("chapter_title"),
                    "source_level_distribution": level_dist,
                    "core_ab_source_count": core_ab,
                    "required_core_ab_source_count": min_core_ab_per_chapter,
                }
            )
    if weak_source_chapters:
        blocking_gaps.append({"type": "public_chapter_without_ab_sources", "chapters": weak_source_chapters[:5]})

    weak_chapters = [
        {
            "chapter_id": chapter.get("chapter_id"),
            "chapter_title": chapter.get("chapter_title"),
            "reason": "no_public_sections_or_tables",
        }
        for chapter in public_chapters
        if not _as_list(chapter.get("sections")) and not _as_list(chapter.get("table_packages"))
    ]
    required_followups = list(blocking_gaps)
    required_followups.extend(structure_followups)
    if length_followup and length_followup not in required_followups:
        required_followups.append(length_followup)
    if dropped_count > 0 and not block_dropped_tasks:
        required_followups.append(
            {
                "type": "search_tasks_dropped",
                "dropped_count": dropped_count,
                "priority": "medium",
                "suggested_query": "补齐被截断 search_tasks 对应的核心证据目标",
            }
        )
    if failed_lanes and not block_lane_failures:
        required_followups.append(
            {
                "type": "iqs_lane_no_success",
                "lanes": failed_lanes,
                "priority": "medium",
                "suggested_query": "对无成功结果的证据类型 lane 进行定向补证",
            }
        )
    for item in proof_gap_rows[:8]:
        required_followups.append(
            {
                "type": "missing_proof_standard",
                "hypothesis_id": item.get("hypothesis_id"),
                "hypothesis_statement": item.get("hypothesis_statement"),
                "blocking_gaps": _as_list(item.get("blocking_gaps")),
                "priority": "high",
                "suggested_query": f"{item.get('hypothesis_statement') or item.get('hypothesis_id') or ''} A/B来源 反证 指标口径 官方 公告 研报",
            }
        )
    for check in missing_mandatory_proofs[:10]:
        required_followups.append(
            {
                "type": "mandatory_proof_missing",
                "proof_id": check.get("proof_id"),
                "mandatory_proof_id": check.get("proof_id"),
                "label": check.get("label"),
                "targets_gap": check.get("label") or check.get("proof_id"),
                "dimension_name": check.get("label") or check.get("proof_id"),
                "evidence_goal": check.get("label") or check.get("proof_id"),
                "severity": check.get("severity"),
                "priority": "high" if str(check.get("severity") or "").lower() == "high" else "medium",
                "suggested_query": check.get("query") or check.get("label"),
                "lane_targets": _as_list(check.get("lane_targets")),
                "source_priority": _as_list(check.get("source_priority")),
                "proof_role": check.get("proof_role"),
                "evidence_type": check.get("evidence_type"),
                "blocking_gaps": ["mandatory_proof_missing"],
            }
        )
    depth_score = max(0, min(100, sum(scores.values())))
    return {
        "depth_score": depth_score,
        "publishable": depth_score >= min_depth_score and not blocking_gaps,
        "minimum_depth_score": min_depth_score,
        "scores": scores,
        "coverage_summary": {
            "coverage_rows": len(coverage_rows),
            "decision_ready_rows": len(decision_ready_rows),
            "proof_gap_rows": len(proof_gap_rows),
            "metric_rows": len(metric_rows),
            "complete_metric_rows": len(complete_metric_rows),
            "rendered_validated_tables": len(rendered_tables),
            "analytics_tables": analytics_table_count,
            "analytics_calculations": analytics_calculation_count,
            "mandatory_proof_total": len(mandatory_proof_checks),
            "mandatory_proof_missing": len(missing_mandatory_proofs),
            "research_maturity_level": maturity.get("level"),
            "research_maturity_score": maturity.get("score"),
        },
        "research_maturity": maturity,
        "analytics_quality": analytics_quality,
        "blocking_gaps": blocking_gaps,
        "weak_chapters": weak_chapters,
        "required_followups": required_followups,
        "rewrite_instructions": [
            "把薄弱章节改写为更清晰的机制链、边界条件和决策动作；缺口优先进入下一轮补证任务。"
        ] if (blocking_gaps or structure_followups) else [],
    }


FATAL_QA_ERROR_TYPES = {
    "report_markdown_empty",
    "internal_label_or_template_phrase",
    "internal_gap_language_leaked",
    "body_table_contains_source_header",
    "bad_table_metric",
    "missing_sources_appendix",
}


HIGH_QA_ERROR_TYPES = {
    "chapter_packages_missing",
    "chapter_sections_missing",
    "argument_unit_missing_evidence_refs",
    "analytics_quality",
    "table_quality",
    "package_contract",
    "deep_report_blocking_gap",
}

RENDER_FATAL_QA_TYPES = {
    "report_markdown_empty",
    "chapter_packages_missing",
    "pipeline_empty_package",
    "unremovable_fake_source_pollution",
    "fake_or_placeholder_source_unremovable",
}

REPORT_BODY_SPLIT_PATTERN = r"\n##\s*(?:附录|研究口径与来源|研究口径|资料来源|数据来源|来源附录|参考来源|参考资料|来源|Appendix|Sources|References)"
SOURCE_APPENDIX_HEADING_PATTERN = r"(?mi)^##+\s*(?:研究口径与来源|资料来源|数据来源|来源附录|参考来源|参考资料|来源|Sources|References)(?:\s|$|[：:])"


EVIDENCE_REPAIR_FOLLOWUP_TYPES = {
    "mandatory_proof_missing",
    "missing_proof_standard",
    "missing_proof_standards",
    "core_claim_without_ab_source",
    "insufficient_ab_sources",
    "insufficient_ab_core_sources",
    "no_ab_sources_for_core_hypotheses",
    "core_hypothesis_counter_missing",
    "public_chapter_without_ab_sources",
    "metric_evidence_missing",
    "metric_definition_unfilled",
    "metric_scope_period_unit_incomplete",
    "counter_evidence_missing",
    "case_evidence_missing",
    "source_diversity_missing",
    "only_c_or_lower_sources",
    "iqs_lane_no_success",
    "iqs_lane_partial_failure",
    "search_tasks_dropped",
    "page_results_zero",
    "iqs_lane_timeout_without_signal",
}


def _is_evidence_repair_followup(item: Any) -> bool:
    payload = _as_dict(item)
    markers = {
        re.sub(r"\s+", "_", str(value or "").strip().lower())
        for value in [
            payload.get("type"),
            payload.get("reason"),
            payload.get("gap_type"),
            payload.get("proof_role"),
            payload.get("evidence_type"),
        ]
        if str(value or "").strip()
    }
    markers.update(
        re.sub(r"\s+", "_", str(value or "").strip().lower())
        for value in _as_list(payload.get("blocking_gaps"))
        if str(value or "").strip()
    )
    if markers.intersection(EVIDENCE_REPAIR_FOLLOWUP_TYPES):
        return True
    if _as_list(payload.get("source_priority")) or _as_list(payload.get("lane_targets")):
        return True
    return bool(payload.get("suggested_query") and (payload.get("proof_id") or payload.get("hypothesis_id")))


def _package_evidence_followups(package_quality_report: Dict[str, Any]) -> List[Dict[str, Any]]:
    followups: List[Dict[str, Any]] = []
    for issue in _as_list(package_quality_report.get("errors")) + _as_list(package_quality_report.get("warnings")):
        payload = _as_dict(issue)
        issue_type = str(payload.get("type") or _as_dict(payload.get("detail")).get("type") or "").strip()
        if issue_type not in EVIDENCE_REPAIR_FOLLOWUP_TYPES:
            continue
        followups.append(
            {
                "type": issue_type,
                "source": "package_contract",
                "priority": "high" if str(payload.get("severity") or "error") == "error" else "medium",
                "path": payload.get("path"),
                "suggested_query": "补齐核心判断所需 A/B 来源、指标口径和反向样本",
                "blocking_gaps": [issue_type],
            }
        )
    return followups[:10]


def _qa_scoring_mode() -> str:
    explicit = os.environ.get("QA_SCORING_MODE")
    if explicit:
        return explicit.strip().lower()
    if _strict_quality_mode():
        return "strict"
    return "score"


def _qa_issue_type(issue: Dict[str, Any]) -> str:
    return str(_as_dict(issue).get("type") or "").strip()


def _qa_finding_category(issue: Dict[str, Any]) -> str:
    payload = _as_dict(issue)
    explicit = str(payload.get("qa_category") or payload.get("finding_category") or "").strip()
    if explicit:
        return explicit
    issue_type = _qa_issue_type(payload)
    if issue_type in RENDER_FATAL_QA_TYPES:
        return "render_blocker"
    if issue_type in FATAL_QA_ERROR_TYPES or issue_type in HIGH_QA_ERROR_TYPES:
        return "clean_blocker"
    return "readability_finding"


def _section_claim_strength(section: Dict[str, Any]) -> str:
    payload = _as_dict(section)
    source_quality = _as_dict(payload.get("source_quality"))
    candidates = [
        payload.get("claim_strength"),
        payload.get("strength"),
        payload.get("evidence_strength"),
        source_quality.get("claim_strength"),
        source_quality.get("grade"),
        payload.get("claim_status"),
        payload.get("quality_status"),
        payload.get("confidence"),
    ]
    text = " ".join(str(item or "").strip().lower() for item in candidates if str(item or "").strip())
    if payload.get("observation_only") or payload.get("layout_generated") and not payload.get("evidence_backed"):
        return "observation"
    if any(token in text for token in ("strong", "decision_ready", "core_claim", "high")):
        return "strong"
    if any(token in text for token in ("moderate", "medium", "supporting")):
        return "moderate"
    if any(token in text for token in ("directional", "limited", "context", "weak", "low", "observation")):
        return "directional"
    if not payload.get("evidence_backed"):
        return "directional"
    return "moderate"


def _section_is_limited_or_observation(section: Dict[str, Any]) -> bool:
    strength = _section_claim_strength(section)
    if strength in {"directional", "observation", "weak", "limited"}:
        return True
    payload = _as_dict(section)
    status = str(payload.get("claim_status") or payload.get("quality_status") or "").strip().lower()
    return status in {"directional", "directional_ready", "limited_evidence", "context_only", "observation_only"}


def _chapter_has_hydrated_signal(chapter: Dict[str, Any]) -> bool:
    payload = _as_dict(chapter)
    if _as_list(payload.get("chapter_fact_digest")):
        return True
    if _as_list(payload.get("table_packages")):
        return True
    quality = _as_dict(payload.get("evidence_quality_summary"))
    for key in (
        "core_evidence_count",
        "supporting_evidence_count",
        "metric_evidence_count",
        "case_evidence_count",
        "counter_evidence_count",
        "directional_evidence_count",
        "sample_evidence_count",
        "hydrated_layer_item_count",
    ):
        try:
            if int(float(payload.get(key) or quality.get(key) or 0)) > 0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _qa_issue_severity(issue: Dict[str, Any], *, strict_mode: bool) -> str:
    issue_type = _qa_issue_type(issue)
    category = _qa_finding_category(issue)
    if category == "readability_finding":
        return "medium"
    if strict_mode and issue_type in HIGH_QA_ERROR_TYPES:
        return "fatal"
    if issue_type in FATAL_QA_ERROR_TYPES:
        return "fatal"
    if issue_type in HIGH_QA_ERROR_TYPES:
        return "high"
    return "medium"


def _score_qa_result(
    *,
    errors: List[Dict[str, Any]],
    warnings: List[Dict[str, Any]],
    deep_evaluation: Dict[str, Any],
    strict_mode: bool,
) -> Dict[str, Any]:
    fatal_errors = [item for item in errors if _qa_issue_severity(item, strict_mode=strict_mode) == "fatal"]
    high_errors = [item for item in errors if _qa_issue_severity(item, strict_mode=strict_mode) == "high"]
    medium_errors = [item for item in errors if _qa_issue_severity(item, strict_mode=strict_mode) == "medium"]
    depth_score = int(deep_evaluation.get("depth_score") or 0)
    min_depth = int(deep_evaluation.get("minimum_depth_score") or 0)
    depth_deficit = max(0, min_depth - depth_score)
    penalties = {
        "fatal_errors": min(90, len(fatal_errors) * (45 if strict_mode else 40)),
        "high_errors": min(45, len(high_errors) * (14 if strict_mode else 8)),
        "medium_errors": min(30, len(medium_errors) * (8 if strict_mode else 4)),
        "warnings": min(
            _env_int("QA_WARNING_PENALTY_CAP", 20 if strict_mode else 12, min_value=0, max_value=100),
            len(warnings) * _env_int("QA_WARNING_PENALTY_EACH", 4 if strict_mode else 1, min_value=0, max_value=20),
        ),
        "depth_deficit": min(20, int(depth_deficit * (0.6 if strict_mode else 0.25))),
    }
    score = max(0, 100 - sum(penalties.values()))
    return {
        "score": score,
        "penalties": penalties,
        "fatal_errors": fatal_errors,
        "high_errors": high_errors,
        "medium_errors": medium_errors,
    }


def _qa_render_gate(errors: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    blockers = [
        _as_dict(item)
        for item in list(errors or [])
        if _qa_issue_type(_as_dict(item)) in RENDER_FATAL_QA_TYPES
    ]
    return {
        "can_render_formal_report": not blockers,
        "blockers": blockers,
        "blocked": bool(blockers),
    }


def _qa_clean_gate(
    *,
    passed: bool,
    errors: Sequence[Dict[str, Any]],
    warnings: Sequence[Dict[str, Any]],
    deep_evaluation: Dict[str, Any],
    score: int,
    min_pass_score: int,
) -> Dict[str, Any]:
    blockers = [_as_dict(item) for item in list(errors or []) if _as_dict(item)]
    for gap in _as_list(_as_dict(deep_evaluation).get("blocking_gaps")):
        payload = _as_dict(gap)
        blockers.append({"type": payload.get("type") or "deep_report_blocking_gap", "source": "deep_evaluator", "detail": payload})
    return {
        "eligible": bool(passed),
        "blockers": blockers,
        "warnings": [_as_dict(item) for item in list(warnings or []) if _as_dict(item)],
        "quality_score": score,
        "minimum_pass_score": min_pass_score,
        "publishable": bool(_as_dict(deep_evaluation).get("publishable")) and bool(passed),
    }


def _qa_quality_findings(
    *,
    errors: Sequence[Dict[str, Any]],
    warnings: Sequence[Dict[str, Any]],
    deep_evaluation: Dict[str, Any],
    score_payload: Dict[str, Any],
) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    for item in list(errors or []):
        payload = _as_dict(item)
        if payload:
            findings.append({"source": "qa_error", **payload})
    for item in list(warnings or []):
        payload = _as_dict(item)
        if payload:
            findings.append({"source": "qa_warning", **payload})
    for gap in _as_list(_as_dict(deep_evaluation).get("blocking_gaps")):
        payload = _as_dict(gap)
        findings.append({"source": "deep_evaluator", "type": payload.get("type") or "deep_report_blocking_gap", "detail": payload})
    penalties = _as_dict(score_payload.get("penalties"))
    for name, value in penalties.items():
        try:
            numeric = int(float(value or 0))
        except (TypeError, ValueError):
            numeric = 0
        if numeric > 0:
            findings.append({"source": "qa_score", "type": f"score_penalty_{name}", "penalty": numeric})
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for item in findings:
        item_type = str(item.get("type") or item.get("source") or "quality_finding")
        item["finding_category"] = _qa_finding_category(item)
        key = (
            item_type,
            str(item.get("chapter_id") or item.get("section_id") or item.get("table_id") or ""),
            str(item.get("source") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        item["type"] = item_type
        deduped.append(item)
    return deduped[:120]


def run_qa_agent(
    *,
    report_markdown: str = "",
    report_blueprint: Optional[Dict[str, Any]] = None,
    chapter_packages: Optional[Sequence[Dict[str, Any]]] = None,
    table_packages: Optional[Sequence[Dict[str, Any]]] = None,
    decision_package: Optional[Dict[str, Any]] = None,
    risk_package: Optional[Dict[str, Any]] = None,
    package_quality_report: Optional[Dict[str, Any]] = None,
    search_task_schedule: Optional[Dict[str, Any]] = None,
    lane_coverage: Optional[Dict[str, Any]] = None,
    retrieval_strategy_summary: Optional[Dict[str, Any]] = None,
    metric_normalization_table: Optional[Sequence[Dict[str, Any]]] = None,
    analytics_outputs: Optional[Sequence[Dict[str, Any]]] = None,
    coverage_matrix: Optional[Sequence[Dict[str, Any]]] = None,
    missing_proof_standards: Optional[Sequence[Dict[str, Any]]] = None,
    evidence_health_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    report_blueprint = _as_dict(report_blueprint)
    decision_package = _as_dict(decision_package)
    risk_package = _as_dict(risk_package)
    package_quality_report = _as_dict(package_quality_report)
    search_task_schedule = _as_dict(search_task_schedule)
    lane_coverage = _as_dict(lane_coverage)
    retrieval_strategy_summary = _as_dict(retrieval_strategy_summary)
    evidence_health_summary = _as_dict(evidence_health_summary)
    text = str(report_markdown or "")
    body_text = re.split(REPORT_BODY_SPLIT_PATTERN, text, maxsplit=1, flags=re.I)[0]
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    analytics_quality = _analytics_quality(analytics_outputs)
    strict_mode = _strict_quality_mode()
    deep_report = _is_deep_report_family(report_blueprint)

    if not text.strip():
        errors.append({"type": "report_markdown_empty"})
    if len(text.strip()) < 200:
        warnings.append({"type": "report_markdown_short"})
    safe_public_terms = set(SAFE_PUBLIC_TERMS)
    for label in INTERNAL_LABELS:
        if label in safe_public_terms:
            continue
        if label and label in text:
            errors.append({"type": "internal_label_or_template_phrase", "phrase": label})
    if _body_table_has_source_header(body_text):
        errors.append({"type": "body_table_contains_source_header"})
    has_citations = bool(re.search(r"\[\d{1,3}\]", text))
    has_sources_appendix = bool(re.search(SOURCE_APPENDIX_HEADING_PATTERN, text))
    appendix_blocking = strict_mode or _env_flag("QA_DEEP_EVALUATOR_BLOCKING", True)
    if has_citations and not has_sources_appendix and appendix_blocking and _env_flag("QA_REQUIRE_SOURCES_APPENDIX", True):
        errors.append({"type": "missing_sources_appendix"})

    narrative = validate_report_narrative_quality(text)
    for issue in _as_list(narrative.get("issues")):
        errors.append(issue)
    errors.extend(validate_no_internal_gap_language(text))

    chapter_packages = [item for item in list(chapter_packages or []) if isinstance(item, dict)]
    if not chapter_packages:
        errors.append({"type": "chapter_packages_missing"})
    for chapter in chapter_packages:
        if chapter.get("omit_from_report"):
            continue
        sections = [_as_dict(item) for item in _as_list(chapter.get("sections"))]
        public_tables = [
            table
            for table in _as_list(chapter.get("table_packages"))
            if isinstance(table, dict) and table.get("should_render")
        ]
        if not sections and not public_tables:
            issue = {
                "type": "chapter_sections_missing_with_evidence" if _chapter_has_hydrated_signal(chapter) else "chapter_sections_missing_no_evidence",
                "chapter_id": chapter.get("chapter_id"),
                "qa_category": "readability_finding",
            }
            warnings.append(issue)
            continue
        for section in sections:
            limited_section = _section_is_limited_or_observation(section)
            if limited_section:
                hard_fields = ("claim",)
                soft_fields = ("reasoning", "counter_evidence", "actionable")
            else:
                hard_fields = ("claim", "reasoning", "counter_evidence", "actionable") if strict_mode else ("claim",)
                soft_fields = () if strict_mode else ("reasoning", "counter_evidence", "actionable")
            hard_missing = [key for key in hard_fields if not str(section.get(key) or "").strip()]
            soft_missing = [key for key in soft_fields if not str(section.get(key) or "").strip()]
            if hard_missing:
                errors.append(
                    {
                        "type": "argument_unit_incomplete",
                        "chapter_id": chapter.get("chapter_id"),
                        "section_id": section.get("section_id"),
                        "missing": hard_missing,
                        "claim_strength": _section_claim_strength(section),
                        "qa_category": "readability_finding" if limited_section else "clean_blocker",
                    }
                )
            if soft_missing:
                warnings.append(
                    {
                        "type": "argument_unit_soft_missing_fields",
                        "chapter_id": chapter.get("chapter_id"),
                        "section_id": section.get("section_id"),
                        "missing": soft_missing,
                        "claim_strength": _section_claim_strength(section),
                        "qa_category": "readability_finding",
                    }
                )
            if not _as_list(section.get("evidence_refs")):
                issue = {
                    "type": "argument_unit_missing_evidence_refs",
                    "chapter_id": chapter.get("chapter_id"),
                    "section_id": section.get("section_id"),
                    "claim_strength": _section_claim_strength(section),
                    "qa_category": "readability_finding" if limited_section else "clean_blocker",
                }
                if limited_section:
                    warnings.append(issue)
                else:
                    errors.append(issue)

    for table in list(table_packages or []):
        if not isinstance(table, dict):
            continue
        table_errors = _as_list(table.get("validation_errors")) or _as_list(_as_dict(table.get("validation")).get("errors"))
        if table.get("should_render"):
            for error in table_errors:
                errors.append({"type": "table_quality", "detail": error, "table_id": table.get("table_id")})
                detail_type = str(_as_dict(error).get("type") or "").strip()
                if detail_type in {"metric_row_missing_fields", "packed_numeric_cell", "malformed_numeric", "missing_table_evidence_refs"}:
                    errors.append({"type": "bad_table_metric", "detail": error, "table_id": table.get("table_id")})
        elif table_errors:
            warnings.append({"type": "table_rejected", "table_id": table.get("table_id"), "validation_errors": table_errors})

    table_fatigue_warnings = _table_fatigue_warnings(body_text, list(table_packages or []))
    warnings.extend(table_fatigue_warnings)

    for error in _as_list(analytics_quality.get("errors")):
        errors.append({"type": "analytics_quality", "detail": error})
    for warning in _as_list(analytics_quality.get("warnings")):
        warnings.append({"type": "analytics_quality", "detail": warning})

    for error in _as_list(package_quality_report.get("errors")):
        errors.append({"type": "package_contract", "detail": error})
    for warning in _as_list(package_quality_report.get("warnings")):
        warning_type = str(_as_dict(warning).get("type") or _as_dict(warning).get("issue_type") or "").strip()
        if deep_report and warning_type == "table_validation_error":
            errors.append({"type": "table_quality", "detail": warning})
        else:
            warnings.append({"type": "package_contract", "detail": warning})

    if evidence_health_summary:
        health_inconsistencies = _as_list(evidence_health_summary.get("inconsistencies"))
        if evidence_health_summary.get("inconsistent") or health_inconsistencies:
            errors.append(
                {
                    "type": "evidence_health_summary_inconsistent",
                    "detail": health_inconsistencies,
                    "evidence_health_summary": evidence_health_summary,
                }
            )
        if deep_report and _count_value(evidence_health_summary.get("analysis_ready_ab_count")) > 0 and _count_value(evidence_health_summary.get("traceable_ab_source_count")) <= 0:
            errors.append(
                {
                    "type": "traceable_ab_sources_missing",
                    "evidence_health_summary": evidence_health_summary,
                }
            )
        if deep_report and _count_value(evidence_health_summary.get("analysis_ready_ab_count")) > 0 and _count_value(evidence_health_summary.get("distinct_verified_ab_source_count")) <= 0:
            errors.append(
                {
                    "type": "verified_ab_sources_missing",
                    "evidence_health_summary": evidence_health_summary,
                }
            )
        if deep_report and evidence_health_summary.get("publishable_evidence_gate_passed") is False:
            errors.append(
                {
                    "type": "publishable_evidence_gate_failed",
                    "evidence_health_summary": evidence_health_summary,
                }
            )

    dropped_count = int(search_task_schedule.get("dropped_count") or len(_as_list(search_task_schedule.get("dropped_tasks"))))
    if dropped_count > 0:
        warnings.append(
            {
                "type": "search_tasks_dropped",
                "dropped_count": dropped_count,
                "message": "Some planner search_tasks were truncated; the report may miss part of the evidence goals.",
            }
        )
    for lane, coverage in lane_coverage.items():
        coverage = _as_dict(coverage)
        if int(coverage.get("scheduled") or 0) and int(coverage.get("succeeded") or 0) == 0:
            warnings.append({"type": "iqs_lane_no_success", "lane": lane, "coverage": coverage})
        if int(coverage.get("failed") or 0) > 0:
            warnings.append({"type": "iqs_lane_partial_failure", "lane": lane, "coverage": coverage})
        if int(coverage.get("scheduled") or 0) and int(coverage.get("page_results") or 0) == 0:
            warnings.append({"type": "page_results_zero", "lane": lane, "coverage": coverage})
        if (
            int(coverage.get("scheduled") or 0)
            and int(coverage.get("timed_out_task_count") or coverage.get("timed_out") or 0) > 0
            and int(coverage.get("usable_source_count") or coverage.get("key_sources") or 0) == 0
        ):
            warnings.append({"type": "iqs_lane_timeout_without_signal", "lane": lane, "coverage": coverage})

    openai_repair_summary = _as_dict(retrieval_strategy_summary.get("openai_web_repair_summary"))
    if openai_repair_summary.get("disabled_after_consecutive_failures"):
        warnings.append(
            {
                "type": "openai_web_repair_disabled",
                "reason": openai_repair_summary.get("last_skip_reason") or openai_repair_summary.get("last_failure_reason"),
                "summary": openai_repair_summary,
            }
        )
    if _count_value(openai_repair_summary.get("failed_count")) > 0 and _count_value(openai_repair_summary.get("success_count")) <= 0:
        warnings.append({"type": "openai_web_repair_failed", "summary": openai_repair_summary})

    if report_blueprint.get("report_family") and decision_package.get("report_family"):
        if str(report_blueprint.get("report_family")) != str(decision_package.get("report_family")):
            warnings.append({"type": "report_family_mismatch"})
    if not _as_list(risk_package.get("risk_items")):
        warnings.append({"type": "risk_items_missing"})

    deep_evaluation = evaluate_deep_report(
        report_markdown=text,
        report_blueprint=report_blueprint,
        chapter_packages=chapter_packages,
        table_packages=list(table_packages or []),
        decision_package=decision_package,
        risk_package=risk_package,
        search_task_schedule=search_task_schedule,
        lane_coverage=lane_coverage,
        metric_normalization_table=metric_normalization_table,
        analytics_outputs=analytics_outputs,
        coverage_matrix=coverage_matrix,
        missing_proof_standards=missing_proof_standards,
    )
    deep_blocking = strict_mode or deep_report or _env_flag("QA_DEEP_EVALUATOR_BLOCKING", True)
    if deep_blocking:
        for gap in _as_list(deep_evaluation.get("blocking_gaps")):
            errors.append({"type": "deep_report_blocking_gap", "detail": gap})

    scoring_mode = _qa_scoring_mode()
    min_pass_score = _env_int("QA_MIN_PASS_SCORE", 75 if (strict_mode or deep_report) else 60, min_value=0, max_value=100)
    score_payload = _score_qa_result(
        errors=errors,
        warnings=warnings,
        deep_evaluation=deep_evaluation,
        strict_mode=strict_mode,
    )
    score = int(score_payload["score"])
    fatal_errors = _as_list(score_payload.get("fatal_errors"))
    if scoring_mode in {"legacy", "hard"}:
        passed = not errors and score >= min_pass_score and (bool(deep_evaluation.get("publishable")) if deep_blocking else True)
    else:
        passed = not fatal_errors and score >= min_pass_score and (bool(deep_evaluation.get("publishable")) if deep_blocking else True)
    soft_errors = _as_list(score_payload.get("high_errors")) + _as_list(score_payload.get("medium_errors"))
    deep_followups = [
        *_as_list(_as_dict(deep_evaluation).get("required_followups")),
        *_package_evidence_followups(package_quality_report),
    ]
    evidence_repair_followups = [item for item in deep_followups if _is_evidence_repair_followup(item)]
    content_repair_followups = [item for item in deep_followups if not _is_evidence_repair_followup(item)]
    error_blocking = strict_mode or deep_report or scoring_mode in {"legacy", "hard"}
    warning_blocking = _env_flag("QA_REPAIR_WARNINGS", False)
    blocking_followups: List[Dict[str, Any]] = []
    for item in fatal_errors:
        blocking_followups.append({"type": _qa_issue_type(item) or "fatal_error", "source": "fatal_error", "detail": item})
    if error_blocking:
        for item in soft_errors:
            if _qa_finding_category(item) == "readability_finding":
                continue
            blocking_followups.append({"type": _qa_issue_type(item) or "qa_error", "source": "qa_error", "detail": item})
    if warning_blocking:
        for item in warnings:
            blocking_followups.append({"type": _qa_issue_type(item) or "qa_warning", "source": "qa_warning", "detail": item})
    if deep_blocking:
        blocking_followups.extend(_as_dict(item) for item in _as_list(deep_evaluation.get("blocking_gaps")))

    blocking_evidence_repair_followups = [item for item in blocking_followups if _is_evidence_repair_followup(item)]
    blocking_content_repair_followups = [item for item in blocking_followups if not _is_evidence_repair_followup(item)]
    advisory_followups = [item for item in deep_followups if item not in blocking_followups]
    advisory_evidence_repair_followups = [item for item in advisory_followups if _is_evidence_repair_followup(item)]
    advisory_content_repair_followups = [item for item in advisory_followups if not _is_evidence_repair_followup(item)]
    repair_required = bool(blocking_followups)
    rewrite_repair_required = bool(fatal_errors) or bool(blocking_content_repair_followups)
    rewrite_only_fatal = _env_flag("QA_REWRITE_ONLY_FATAL", False)
    rewrite_required = bool(fatal_errors) or (not rewrite_only_fatal and rewrite_repair_required)
    render_gate = _qa_render_gate(errors)
    clean_gate = _qa_clean_gate(
        passed=passed,
        errors=errors,
        warnings=warnings,
        deep_evaluation=deep_evaluation,
        score=score,
        min_pass_score=min_pass_score,
    )
    qa_quality_findings = _qa_quality_findings(
        errors=errors,
        warnings=warnings,
        deep_evaluation=deep_evaluation,
        score_payload=score_payload,
    )
    readability_findings = [
        item for item in qa_quality_findings if _qa_finding_category(item) == "readability_finding"
    ]
    render_blocking_followups = _as_list(render_gate.get("blockers"))
    return {
        "agent": AGENT_NAME,
        "passed": passed,
        "scoring_mode": scoring_mode,
        "quality_score": score,
        "minimum_pass_score": min_pass_score,
        "score_breakdown": score_payload.get("penalties"),
        "fatal_errors": fatal_errors,
        "soft_errors": soft_errors,
        "render_gate": render_gate,
        "clean_gate": clean_gate,
        "quality_findings": qa_quality_findings,
        "readability_findings": readability_findings,
        "repair_required": repair_required,
        "render_repair_required": bool(render_blocking_followups),
        "repair_followups": deep_followups,
        "blocking_followups": blocking_followups,
        "render_blocking_followups": render_blocking_followups,
        "advisory_followups": advisory_followups,
        "evidence_repair_followups": evidence_repair_followups,
        "content_repair_followups": content_repair_followups,
        "blocking_evidence_repair_followups": blocking_evidence_repair_followups,
        "blocking_content_repair_followups": blocking_content_repair_followups,
        "advisory_evidence_repair_followups": advisory_evidence_repair_followups,
        "advisory_content_repair_followups": advisory_content_repair_followups,
        "depth_score": deep_evaluation.get("depth_score"),
        "publishable": deep_evaluation.get("publishable"),
        "research_maturity": _as_dict(deep_evaluation).get("research_maturity"),
        "deep_evaluator_blocking": deep_blocking,
        "deep_evaluation": deep_evaluation,
        "rewrite_instructions": _as_list(_as_dict(deep_evaluation).get("rewrite_instructions")),
        "analytics_quality": analytics_quality,
        "issues": errors + warnings,
        "errors": errors,
        "warnings": warnings,
        "rewrite_required": rewrite_required,
        "clean_format": {
            "table_count": _table_count(body_text),
            "has_body_source_table_header": _body_table_has_source_header(body_text),
            "table_fatigue_warnings": table_fatigue_warnings,
        },
        "package_quality": package_quality_report,
        "search_task_schedule": search_task_schedule,
        "lane_coverage": lane_coverage,
        "retrieval_strategy_summary": retrieval_strategy_summary,
        "evidence_health_summary": evidence_health_summary,
        "report_family": report_blueprint.get("report_family"),
    }


def validate_enterprise_report(
    *,
    markdown: str,
    layout: Any = None,
    chapter_packages: Optional[Sequence[Dict[str, Any]]] = None,
    materials: Optional[Dict[str, Any]] = None,
    materials_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    del layout, materials, materials_payload
    result = run_qa_agent(report_markdown=markdown, chapter_packages=chapter_packages or [])
    return {
        **result,
        "passed": bool(result.get("passed")),
        "errors": _as_list(result.get("errors")),
        "warnings": _as_list(result.get("warnings")),
    }
