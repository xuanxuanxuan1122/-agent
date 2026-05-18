from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .table_validator import validate_table_package

try:
    from .analytics.analytics_contracts import iter_analytics_tables
except Exception:  # pragma: no cover - direct script mode fallback.
    from analytics.analytics_contracts import iter_analytics_tables  # type: ignore


AGENT_NAME = "table_agent"
AGENT_DESCRIPTION = "Table Agent. Designs and validates evidence-backed table packages."
GENERIC_METRIC_NAMES = {"", "数据指标", "数据点", "关键数据", "比例/增速", "占比", "比例", "增速"}
POLLUTED_ENTITY_CELL_RE = re.compile(
    r"(?:^LOOK\s*[~\-_=#]+$|[~]{4,}|增长质量需要按细分场景验证|"
    r"^\d{4}年.{0,50}(?:报告|分析|研判|前瞻|展望|白皮书|研究)|"
    r"(?:报告|分析|研判|前瞻|展望|白皮书|研究报告)$)"
)


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _compact(value: Any, max_chars: int = 160) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "..."


def _is_polluted_entity_cell(value: Any) -> bool:
    text = _compact(value, 160)
    if not text:
        return False
    if len(text) > 36 and ("行业" in text or "报告" in text or "研究" in text):
        return True
    return bool(POLLUTED_ENTITY_CELL_RE.search(text))


def _dedupe(values: Iterable[Any], *, limit: int = 8) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        text = _compact(value, 120)
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


def _env_int(name: str, default: int, *, min_value: int = 0, max_value: int = 100) -> int:
    try:
        value = int(os.getenv(name, str(default)) or default)
    except (TypeError, ValueError):
        value = default
    return max(min_value, min(max_value, value))


def _body_table_budget(*, chapter_count: int = 0, candidate_count: int = 0) -> int:
    if os.getenv("REPORT_MAX_BODY_TABLES") is not None:
        requested = _env_int("REPORT_MAX_BODY_TABLES", 6, min_value=0, max_value=50)
    else:
        requested = min(8, max(4, chapter_count + 2, min(candidate_count, 6)))
    hard_limit = _env_int("REPORT_HARD_MAX_BODY_TABLES", 12, min_value=0, max_value=50)
    return max(0, min(requested, hard_limit))


def _per_chapter_table_budget() -> int:
    return _env_int("REPORT_MAX_BODY_TABLES_PER_CHAPTER", 3, min_value=0, max_value=20)


TOPIC_CHAPTER_KEYWORDS = [
    (
        {"market_metric_table", "regional_share_table", "cagr_calculation", "market_analytics"},
        ("\u5e02\u573a", "\u89c4\u6a21", "\u589e\u901f", "\u4ef7\u683c", "tam", "cagr"),
    ),
    (
        {"competitor_matrix", "competitor_analysis"},
        ("\u7ade\u4e89", "\u73a9\u5bb6", "\u5bf9\u624b", "\u58c1\u5792", "\u4efd\u989d"),
    ),
    (
        {"risk_register", "regulatory_impact"},
        ("\u653f\u7b56", "\u76d1\u7ba1", "\u6cd5\u89c4", "\u98ce\u9669", "\u51fa\u53e3\u7ba1\u5236"),
    ),
    (
        {"technology_roadmap", "technology_roadmap_matrix"},
        ("\u6280\u672f", "\u8def\u7ebf", "\u6210\u719f", "\u66ff\u4ee3", "\u5de5\u827a"),
    ),
    (
        {"investment_priority_table", "investor_insight"},
        ("\u6295\u8d44", "\u8d44\u672c", "\u878d\u8d44", "ipo", "\u4f30\u503c"),
    ),
]


LOCALIZED_ANALYTICS_TABLE_COPY = {
    "market_metric_table": {
        "title": "市场指标与口径表",
        "headers": ["指标", "范围", "期间", "数值", "单位", "来源等级"],
        "takeaway": "市场数据在进入正文前已拆分为指标、范围、期间、单位和来源等级，避免把不同口径的数据直接混用。",
        "decision_implication": "优先使用口径完整且来源等级较高的指标作为量化主线，缺口数据只作为边界或后续验证项。",
    },
    "cagr_calculation": {
        "title": "CAGR 推算校验表",
        "headers": ["指标", "范围", "期间", "CAGR", "基期到末期"],
        "takeaway": "增长率来自可追溯的基期和末期数据，不直接复制未经校验的增速表述。",
        "decision_implication": "只有起止时间、单位和范围可比的增长率，才适合支撑市场扩张判断。",
    },
    "regional_share_table": {
        "title": "市场份额与区域/主体拆分表",
        "headers": ["指标", "区域/主体", "期间", "份额/数值", "来源等级"],
        "takeaway": "份额类指标已与规模类指标分开呈现，避免把比例和金额放在同一口径下比较。",
        "decision_implication": "份额数据只有在分母清楚时，才适合作为区域或竞争结论的依据。",
    },
    "competitor_matrix": {
        "title": "竞争格局对照表",
        "headers": ["企业/来源", "维度", "期间", "信号", "来源等级"],
        "takeaway": "竞争信息按维度拆开呈现，避免把份额、技术、客户和出口暴露混成一个笼统分数。",
        "decision_implication": "用该表识别领先环节、风险暴露和仍需补证的竞争维度。",
    },
    "risk_register": {
        "title": "政策影响与风险登记表",
        "headers": ["政策/文件", "影响环节", "时间窗口", "风险等级", "影响"],
        "takeaway": "政策影响先按环节、时间和风险等级拆分，再进入正文判断。",
        "decision_implication": "高风险条目应进入监测清单，并作为情景边界和反向触发条件。",
    },
    "investment_priority_table": {
        "title": "投资优先级矩阵",
        "headers": ["企业/来源", "信号", "期间", "数值", "评分", "分层"],
        "takeaway": "投资排序与正文判断分离，并保留到财务、融资或市场信号的引用链路。",
        "decision_implication": "优先级可用于初筛；观察和风险复核项需要结合边界条件再下结论。",
    },
    "technology_roadmap": {
        "title": "技术路线成熟度矩阵",
        "headers": ["技术/来源", "类别", "成熟度", "期间", "影响"],
        "takeaway": "技术证据先按类别和成熟度映射，再用于支撑突破、替代或落地节奏判断。",
        "decision_implication": "成熟度较高的技术可支持近期采用，成熟度较低的技术应作为期权或监测项。",
    },
}


CELL_VALUE_TRANSLATIONS = {
    "Market share": "市场份额",
    "Technology": "技术能力",
    "Business operations": "经营环节",
    "Low": "低",
    "Medium": "中",
    "High": "高",
    "Watch": "观察",
    "Priority": "优先",
    "Risk review": "风险复核",
}


def _table_priority(package: Dict[str, Any]) -> int:
    table_type = str(package.get("table_type") or "").strip().lower()
    analytics_type = str(package.get("analytics_type") or "").strip().lower()
    if table_type == "cagr_calculation":
        return 10
    if table_type == "market_metric_table":
        return 20
    if table_type == "competitor_matrix":
        return 30
    if analytics_type == "regulatory_impact" or table_type == "risk_register":
        return 40
    if table_type == "investment_priority_table":
        return 50
    if table_type == "technology_roadmap":
        return 60
    if table_type == "regional_share_table":
        return 70
    return 100


def _headers_for_request(request: Dict[str, Any], chapter_question: str) -> List[str]:
    table_type = str(request.get("table_type") or "").strip()
    if table_type == "customer_painpoint_matrix":
        return ["场景", "需求痛点", "付费主体", "影响方向"]
    if table_type == "segment_matrix":
        return ["细分场景", "需求/约束", "成熟信号", "影响方向"]
    if table_type == "policy_timeline":
        return ["政策事项", "执行机制", "影响主体", "观察指标"]
    if table_type == "player_matrix":
        return ["参与者", "能力/动作", "已验证信号", "竞争影响"]
    if table_type == "technology_maturity":
        return ["场景/环节", "技术信号", "成熟度", "验证指标"]
    if table_type == "metric_reconciliation":
        return ["指标", "差异来源", "建议采用", "使用边界"]
    if table_type == "risk_register":
        return ["风险事项", "触发原因", "影响范围", "缓释动作"]
    if table_type == "unit_economics":
        return ["经济项", "关键假设", "验证方法", "策略影响"]
    if table_type == "case_comparison":
        return ["案例", "适用场景", "有效信号", "可复制边界"]
    if "谁" in chapter_question or "客户" in chapter_question or "用户" in chapter_question:
        return ["对象/场景", "需求信号", "约束条件", "后续影响"]
    return ["对象/场景", "关键事实", "适用边界", "后续动作"]


def _subject(item: Dict[str, Any]) -> str:
    source = _as_dict(item.get("source"))
    fact = str(item.get("fact") or "")
    for key in ("subject", "company", "entity", "scene", "metric"):
        value = _compact(item.get(key), 80)
        if value and not _is_polluted_entity_cell(value):
            return value
    match = re.search(r"([\u4e00-\u9fffA-Za-z0-9]{2,24}(?:公司|集团|科技|智能|场景|客户|用户|政策|平台))", fact)
    if match:
        return match.group(1)
    title = _compact(source.get("title"), 80)
    if title and len(title) <= 28 and not _is_polluted_entity_cell(title):
        return title
    return ""


def _row_for_item(item: Dict[str, Any], headers: Sequence[str]) -> Dict[str, Any]:
    fact = _compact(item.get("fact"), 180)
    metric = _compact(item.get("metric"), 80)
    value = _compact(item.get("value"), 80)
    period = _compact(item.get("period"), 60)
    subject = _subject(item)
    signal = " / ".join(part for part in [metric, value] if part) or fact
    boundary = " / ".join(part for part in [period, _compact(item.get("source_level"), 20)] if part) or "需用高等级来源复核"
    implication = "该信号只有与反例和高等级来源同向时，影响才会扩大" if not item.get("appendix_only") else "仅适合作为附录线索"
    row_claim = _row_claim(item, subject=subject, signal=signal, boundary=boundary, implication=implication)
    cells_by_header = {
        "场景": subject,
        "细分场景": subject,
        "对象/场景": subject,
        "参与者": subject,
        "主体": subject,
        "政策事项": subject,
        "事项": subject,
        "指标": metric or subject,
        "需求痛点": fact,
        "需求/约束": fact,
        "成熟信号": signal,
        "进入判断": implication,
        "影响方向": implication,
        "竞争判断": implication,
        "竞争影响": implication,
        "需求信号": fact,
        "关键事实": fact,
        "能力/动作": fact,
        "技术信号": fact,
        "执行机制": fact,
        "差异来源": signal,
        "付费主体": _compact(item.get("payer") or item.get("customer") or "需继续验证", 80),
        "可验证指标": signal,
        "已验证信号": signal,
        "成熟度判断": implication,
        "成熟度": implication,
        "建议采用": signal,
        "约束条件": boundary,
        "判断边界": boundary,
        "适用边界": boundary,
        "行动含义": implication,
        "后续动作": implication,
        "影响主体": subject,
        "观察指标": signal,
        "验证指标": signal,
        "使用边界": boundary,
        "判断含义": implication,
        "后续影响": implication,
        "风险事项": subject,
        "触发原因": fact,
        "影响范围": implication,
        "缓释动作": "优先验证触发条件，并设置监测阈值",
        "经济项": subject,
        "关键假设": fact,
        "验证方法": boundary,
        "决策含义": implication,
        "策略影响": implication,
        "案例": subject,
        "适用场景": subject,
        "有效信号": signal,
        "可复制边界": boundary,
    }
    cells = [_compact(cells_by_header.get(str(header), fact), 160) for header in headers]
    return {
        "cells": cells,
        "row_claim": row_claim,
        "evidence_refs": _dedupe(
            [
                item.get("ref"),
                item.get("evidence_id"),
                item.get("source_ref"),
                item.get("citation_ref"),
                *_as_list(item.get("source_refs")),
            ],
            limit=3,
        ),
    }


def _row_has_valid_leading_cell(headers: Sequence[str], row: Dict[str, Any]) -> bool:
    cells = _as_list(row.get("cells"))
    if not cells:
        return False
    first_cell = str(cells[0] or "").strip()
    if _is_polluted_entity_cell(first_cell):
        return False
    first_header = str(next(iter(headers), "") or "").strip()
    subject_like_headers = {
        "场景",
        "细分场景",
        "对象/场景",
        "参与者",
        "主体",
        "政策事项",
        "事项",
        "风险事项",
        "经济项",
        "案例",
        "适用场景",
    }
    if first_header in subject_like_headers and not first_cell:
        return False
    return True


def _row_claim(item: Dict[str, Any], *, subject: str, signal: str, boundary: str, implication: str) -> str:
    table_type = str(_as_dict(item.get("search_task")).get("table_type") or "").strip()
    fact = _compact(item.get("fact"), 160)
    level = str(item.get("source_level") or "").upper()
    allowed_use = str(item.get("allowed_use") or _as_dict(item.get("evidence_card")).get("allowed_use") or "").strip()
    source_note = f"{level}级来源" if level in {"A", "B", "C", "D"} else "当前来源"
    signal_text = _compact(signal if signal and signal != fact else fact, 90)
    boundary_text = _compact(boundary, 90)
    if level == "C" and allowed_use == "directional_signal" and not item.get("appendix_only"):
        return f"{subject}只能作为方向性信号使用；表内依据是{signal_text}，后续需要连续指标或A/B来源校准。"
    if level in {"C", "D"} or item.get("appendix_only"):
        return f"{subject}的证据等级偏弱，表内只保留为背景线索；可用边界是{boundary_text}。"
    if table_type == "risk_register":
        trigger = _compact(item.get("trigger") or fact or signal_text, 90)
        return f"{subject}的风险触发点是{trigger}；如果该信号兑现，应先降级判断再安排缓释动作。"
    if signal and signal != fact:
        return f"{subject}的表内信号是{signal_text}；结论强度取决于{boundary_text}和反向证据是否同步。"
    if implication:
        return f"{subject}基于{source_note}进入表格；对本章的影响是{_compact(implication, 100)}。"
    return f"{subject}基于{source_note}提供可核验线索；是否能代表可复制场景仍取决于{boundary_text}。"


def _takeaway(rows: Sequence[Dict[str, Any]], chapter_question: str) -> str:
    if not rows:
        return "当前不生成正文表格。"
    first = _compact(rows[0].get("row_claim"), 140)
    if chapter_question:
        return f"围绕“{chapter_question}”，{first}是否能代表可复制场景，是影响结论强度的关键。"
    return f"{first}是否能代表可复制场景，是影响结论强度的关键。"


def _select_table_evidence(items: Sequence[Dict[str, Any]], *, limit: int = 10) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    seen = set()
    for item in items:
        if not isinstance(item, dict) or item.get("appendix_only") or item.get("excluded"):
            continue
        level = str(item.get("source_level") or "").strip().upper()
        if level not in {"A", "B"} and str(os.getenv("REPORT_ALLOW_C_LEVEL_BODY_TABLES", "false")).strip().lower() not in {"1", "true", "yes", "on"}:
            continue
        metric = _compact(item.get("metric"), 80)
        value = _compact(item.get("value"), 80)
        fact = str(item.get("fact") or "")
        subject = item.get("company") or item.get("enterprise") or item.get("entity") or item.get("subject")
        if subject and _is_polluted_entity_cell(subject):
            continue
        if metric in GENERIC_METRIC_NAMES and value and re.search(r"\d", value):
            continue
        if (metric or value or re.search(r"\d", fact)) and not (item.get("source_ref") or item.get("source_refs")):
            continue
        key = re.sub(r"\s+", "", str(item.get("fact") or item.get("metric") or item.get("value") or "").lower())
        if not key or key in seen:
            continue
        seen.add(key)
        selected.append(item)
        if len(selected) >= limit:
            break
    return selected


def _table_reject_reasons(rows: Sequence[Dict[str, Any]], selected: Sequence[Dict[str, Any]]) -> List[str]:
    reasons: List[str] = []
    if len(selected) < 2:
        reasons.append("body_evidence_count_lt_2")
    if len(rows) < 2:
        reasons.append("body_rows_lt_2")
    weak_refs = 0
    for row in rows:
        if not _as_list(_as_dict(row).get("evidence_refs")):
            weak_refs += 1
    if rows and weak_refs / max(len(rows), 1) > 0.5:
        reasons.append("majority_rows_missing_evidence_refs")
    return reasons


def _chapter_records(
    chapter_evidence_packages: Optional[Sequence[Dict[str, Any]]],
    micro_layouts: Optional[Sequence[Dict[str, Any]]],
) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    seen = set()
    for source in (chapter_evidence_packages, micro_layouts):
        for item in list(source or []):
            if not isinstance(item, dict):
                continue
            chapter_id = str(item.get("chapter_id") or "").strip()
            if not chapter_id or chapter_id in seen:
                continue
            seen.add(chapter_id)
            records.append(
                {
                    "chapter_id": chapter_id,
                    "chapter_title": str(item.get("chapter_title") or item.get("title") or "").strip(),
                    "chapter_question": str(item.get("chapter_question") or "").strip(),
                }
            )
    return records


def _evidence_ref_chapter_map(chapter_evidence_packages: Optional[Sequence[Dict[str, Any]]]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    evidence_keys = (
        "core_evidence",
        "supporting_evidence",
        "sample_evidence",
        "table_evidence",
        "appendix_evidence",
        "clue_evidence",
        "analysis_ready_evidence",
    )
    for package in list(chapter_evidence_packages or []):
        if not isinstance(package, dict):
            continue
        chapter_id = str(package.get("chapter_id") or "").strip()
        if not chapter_id:
            continue
        for key in evidence_keys:
            for item in _as_list(package.get(key)):
                if not isinstance(item, dict):
                    continue
                for ref_key in ("ref", "evidence_id", "source_ref"):
                    ref = str(item.get(ref_key) or "").strip()
                    if ref:
                        result.setdefault(ref, chapter_id)
                for ref in _as_list(item.get("source_refs")):
                    ref_text = str(ref or "").strip()
                    if ref_text:
                        result.setdefault(ref_text, chapter_id)
    return result


def _keyword_chapter_id(package: Dict[str, Any], chapters: Sequence[Dict[str, str]]) -> str:
    profile = " ".join(
        str(package.get(key) or "").strip().lower()
        for key in ("table_type", "analytics_type", "analytics_agent", "title")
    )
    topic_keywords: Sequence[str] = ()
    for signals, keywords in TOPIC_CHAPTER_KEYWORDS:
        if any(signal in profile for signal in signals):
            topic_keywords = keywords
            break
    if not topic_keywords:
        return ""
    for chapter in chapters:
        haystack = " ".join(
            [
                str(chapter.get("chapter_title") or "").lower(),
                str(chapter.get("chapter_question") or "").lower(),
            ]
        )
        if any(keyword.lower() in haystack for keyword in topic_keywords):
            return str(chapter.get("chapter_id") or "").strip()
    return ""


def _ref_majority_chapter_id(package: Dict[str, Any], ref_chapters: Dict[str, str]) -> str:
    counts: Dict[str, int] = {}
    for ref in _as_list(package.get("evidence_refs")):
        chapter_id = ref_chapters.get(str(ref or "").strip())
        if not chapter_id:
            continue
        counts[chapter_id] = counts.get(chapter_id, 0) + 1
    if not counts:
        return ""
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _normalize_analytics_chapter_ids(
    packages: Sequence[Dict[str, Any]],
    *,
    chapters: Sequence[Dict[str, str]],
    ref_chapters: Dict[str, str],
) -> None:
    valid_ids = {str(chapter.get("chapter_id") or "").strip() for chapter in chapters}
    first_chapter = next((str(chapter.get("chapter_id") or "").strip() for chapter in chapters if str(chapter.get("chapter_id") or "").strip()), "")
    for package in packages:
        if not isinstance(package, dict):
            continue
        inferred = _keyword_chapter_id(package, chapters) or _ref_majority_chapter_id(package, ref_chapters)
        current = str(package.get("chapter_id") or "").strip()
        if inferred:
            package["chapter_id"] = inferred
        elif current not in valid_ids and first_chapter:
            package["chapter_id"] = first_chapter


def _apply_render_budget(
    packages: Sequence[Dict[str, Any]],
    *,
    max_body_tables: int,
    per_chapter_limit: int,
    chapter_order: Sequence[str],
) -> None:
    eligible = [
        package
        for package in packages
        if isinstance(package, dict) and package.get("should_render") and not package.get("appendix_only")
    ]
    if not eligible:
        return

    for package in eligible:
        package["should_render"] = False

    order_index = {str(chapter_id): index for index, chapter_id in enumerate(chapter_order)}
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for package in eligible:
        chapter_id = str(package.get("chapter_id") or "").strip()
        grouped.setdefault(chapter_id, []).append(package)
    for chapter_id, group in grouped.items():
        group.sort(key=lambda package: (_table_priority(package), str(package.get("table_id") or "")))

    ordered_chapters = sorted(
        grouped,
        key=lambda chapter_id: (order_index.get(chapter_id, len(order_index)), chapter_id),
    )
    selected: set[int] = set()
    rendered_by_chapter: Dict[str, int] = {}
    rendered_total = 0
    while rendered_total < max_body_tables:
        progressed = False
        for chapter_id in ordered_chapters:
            if rendered_total >= max_body_tables:
                break
            if rendered_by_chapter.get(chapter_id, 0) >= per_chapter_limit:
                continue
            group = grouped.get(chapter_id) or []
            while group and id(group[0]) in selected:
                group.pop(0)
            if not group:
                continue
            package = group.pop(0)
            selected.add(id(package))
            rendered_by_chapter[chapter_id] = rendered_by_chapter.get(chapter_id, 0) + 1
            rendered_total += 1
            progressed = True
        if not progressed:
            break

    for package in eligible:
        if id(package) in selected:
            package["should_render"] = True
            continue
        chapter_id = str(package.get("chapter_id") or "").strip()
        reasons = package.setdefault("reject_reasons", [])
        if max_body_tables <= 0 or rendered_total >= max_body_tables:
            reasons.append("global_body_table_budget_exceeded")
        elif rendered_by_chapter.get(chapter_id, 0) >= per_chapter_limit:
            reasons.append("chapter_body_table_budget_exceeded")
        else:
            reasons.append("body_table_budget_not_selected")


def _localized_analytics_copy(table_type: str) -> Dict[str, Any]:
    return _as_dict(LOCALIZED_ANALYTICS_TABLE_COPY.get(str(table_type or "").strip()))


def _localize_analytics_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    copied = dict(row)
    cells = [
        CELL_VALUE_TRANSLATIONS.get(str(cell or "").strip(), cell)
        for cell in _as_list(row.get("cells"))
    ]
    if cells and _is_polluted_entity_cell(cells[0]):
        return None
    copied["cells"] = cells
    return copied


def _analytics_table_packages(
    analytics_outputs: Optional[Sequence[Dict[str, Any]]],
    *,
    body_row_limit: int,
    appendix_row_limit: int,
) -> List[Dict[str, Any]]:
    packages: List[Dict[str, Any]] = []
    for index, table in enumerate(iter_analytics_tables(analytics_outputs), start=1):
        table = _as_dict(table)
        rows = [row for row in _as_list(table.get("rows")) if isinstance(row, dict)]
        if not rows:
            continue
        localized = _localized_analytics_copy(str(table.get("table_type") or "analytics_table"))
        body_rows = [
            localized_row
            for row in rows[:body_row_limit]
            for localized_row in [_localize_analytics_row(row)]
            if localized_row is not None
        ]
        appendix_rows = [
            localized_row
            for row in (_as_list(table.get("appendix_rows")) or rows[body_row_limit : body_row_limit + appendix_row_limit])
            if isinstance(row, dict)
            for localized_row in [_localize_analytics_row(row)]
            if localized_row is not None
        ]
        if not body_rows and not appendix_rows:
            continue
        evidence_refs = _dedupe(
            [
                ref
                for row in rows
                for ref in _as_list(_as_dict(row).get("evidence_refs"))
            ],
            limit=30,
        )
        package = {
            "agent": AGENT_NAME,
            "table_id": str(table.get("table_id") or f"analytics_t{index}"),
            "chapter_id": str(table.get("chapter_id") or "analytics"),
            "table_type": str(table.get("table_type") or "analytics_table"),
            "title": str(localized.get("title") or table.get("title") or "Analytics table"),
            "purpose": str(table.get("purpose") or localized.get("takeaway") or ""),
            "headers": _as_list(localized.get("headers")) or _as_list(table.get("headers")),
            "rows": body_rows,
            "appendix_rows": appendix_rows,
            "takeaway": str(localized.get("takeaway") or table.get("takeaway") or ""),
            "decision_implication": str(localized.get("decision_implication") or table.get("decision_implication") or ""),
            "limitations": _as_list(table.get("limitations")),
            "should_render": True,
            "appendix_only": False,
            "high_quality_evidence_count": len(evidence_refs),
            "evidence_refs": evidence_refs,
            "analytics_type": table.get("analytics_type"),
            "analytics_agent": table.get("analytics_agent") or table.get("analytics_source"),
            "reject_reasons": [],
            "validation_errors": [],
        }
        validation = validate_table_package(package)
        package["validation"] = validation
        package["validation_errors"] = validation.get("errors", [])
        package["should_render"] = bool(validation.get("passed"))
        packages.append(package)
    packages.sort(key=_table_priority)
    return packages


def run_table_agent(
    *,
    chapter_evidence_packages: Optional[Sequence[Dict[str, Any]]] = None,
    micro_layouts: Optional[Sequence[Dict[str, Any]]] = None,
    analytics_outputs: Optional[Sequence[Dict[str, Any]]] = None,
    llm_client: Any = None,
) -> List[Dict[str, Any]]:
    del llm_client
    enable_tables = str(os.getenv("REPORT_ENABLE_TABLES", "true")).strip().lower() in {"1", "true", "yes", "on"}
    if not enable_tables:
        return []
    packages_by_id = {
        str(package.get("chapter_id") or ""): package
        for package in list(chapter_evidence_packages or [])
        if isinstance(package, dict)
    }
    table_packages: List[Dict[str, Any]] = []
    body_row_limit = int(os.getenv("REPORT_MAX_BODY_TABLE_ROWS", "8"))
    appendix_row_limit = int(os.getenv("REPORT_MAX_APPENDIX_TABLE_ROWS", "30"))
    analytics_packages = _analytics_table_packages(
        analytics_outputs,
        body_row_limit=body_row_limit,
        appendix_row_limit=appendix_row_limit,
    )
    chapters = _chapter_records(chapter_evidence_packages, micro_layouts)
    _normalize_analytics_chapter_ids(
        analytics_packages,
        chapters=chapters,
        ref_chapters=_evidence_ref_chapter_map(chapter_evidence_packages),
    )
    max_body_tables = _body_table_budget(
        chapter_count=len([item for item in list(micro_layouts or []) if isinstance(item, dict)]),
        candidate_count=len(analytics_packages),
    )
    table_packages.extend(analytics_packages)
    for layout in list(micro_layouts or []):
        if not isinstance(layout, dict):
            continue
        chapter_id = str(layout.get("chapter_id") or "")
        evidence_package = _as_dict(packages_by_id.get(chapter_id))
        chapter_question = str(evidence_package.get("chapter_question") or "").strip()
        evidence_items = [
            item
            for key in ("table_evidence", "sample_evidence", "core_evidence", "supporting_evidence")
            for item in _as_list(evidence_package.get(key))
            if isinstance(item, dict)
        ]
        for request_index, request in enumerate(_as_list(layout.get("table_requests")), start=1):
            if not isinstance(request, dict):
                continue
            headers = _headers_for_request(request, chapter_question)
            selected = _select_table_evidence(evidence_items, limit=10)
            rows = []
            for item in selected:
                row = _row_for_item(item, headers)
                if not _row_has_valid_leading_cell(headers, row):
                    continue
                rows.append(row)
            reject_reasons = _table_reject_reasons(rows, selected)
            body_rows = rows[:body_row_limit]
            appendix_rows = rows[body_row_limit : body_row_limit + appendix_row_limit]
            retained_refs = _dedupe(
                [
                    ref
                    for row in rows
                    for ref in _as_list(_as_dict(row).get("evidence_refs"))
                ],
                limit=30,
            )
            package = {
                "agent": AGENT_NAME,
                "table_id": str(request.get("table_id") or f"{chapter_id}_t{request_index}"),
                "chapter_id": chapter_id,
                "table_type": str(request.get("table_type") or "evidence_matrix"),
                "title": str(request.get("title") or f"{evidence_package.get('chapter_title') or '本章'}核心变量对照"),
                "purpose": str(request.get("purpose") or ""),
                "headers": headers,
                "rows": body_rows,
                "appendix_rows": appendix_rows,
                "takeaway": _takeaway(rows, chapter_question),
                "decision_implication": "若表内信号继续被高等级来源验证，可纳入章节分析；若出现反向条件，应先校准边界。",
                "limitations": [
                    "表格优先使用已绑定到本章的正文证据；C级方向性信号只用于趋势和边界比较。",
                    "如果同一指标存在来源差异，正文只保留后续影响，并在后续观察中校准口径。",
                ],
                "should_render": True,
                "appendix_only": False,
                "high_quality_evidence_count": len(retained_refs),
                "evidence_refs": retained_refs,
                "reject_reasons": reject_reasons,
                "validation_errors": [],
            }
            validation = validate_table_package(package)
            package["validation"] = validation
            package["validation_errors"] = validation.get("errors", [])
            package["should_render"] = bool(validation.get("passed")) and not reject_reasons
            table_packages.append(package)
    chapter_order = [chapter.get("chapter_id", "") for chapter in chapters]
    _apply_render_budget(
        table_packages,
        max_body_tables=max_body_tables,
        per_chapter_limit=_per_chapter_table_budget(),
        chapter_order=chapter_order,
    )
    return table_packages
