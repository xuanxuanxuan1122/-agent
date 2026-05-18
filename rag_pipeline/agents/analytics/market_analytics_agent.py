from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .analytics_contracts import as_dict, as_list, compact, dedupe, make_output, make_table


AGENT_NAME = "market_analytics_agent"
AGENT_DESCRIPTION = "Builds market metric, CAGR, share, and forecast analytics from bound evidence."

MARKET_TERMS = (
    "market",
    "tam",
    "sam",
    "cagr",
    "growth",
    "share",
    "shipment",
    "revenue",
    "sales",
    "forecast",
    "\u5e02\u573a",
    "\u89c4\u6a21",
    "\u589e\u901f",
    "\u589e\u957f",
    "\u4efd\u989d",
    "\u5360\u6bd4",
    "\u51fa\u8d27",
    "\u9500\u91cf",
    "\u8425\u6536",
    "\u6536\u5165",
    "\u9884\u6d4b",
)

SHARE_TERMS = ("share", "\u4efd\u989d", "\u5360\u6bd4", "\u6e17\u900f\u7387", "\u5e02\u5360")
GROWTH_TERMS = ("cagr", "growth", "\u589e\u901f", "\u589e\u957f", "\u540c\u6bd4", "\u590d\u5408")
SIZE_TERMS = ("market size", "tam", "\u5e02\u573a\u89c4\u6a21", "\u89c4\u6a21", "\u7a7a\u95f4")
SHIPMENT_TERMS = ("shipment", "sales volume", "\u51fa\u8d27", "\u9500\u91cf", "\u4ea7\u91cf")
REVENUE_TERMS = ("revenue", "sales", "\u8425\u6536", "\u6536\u5165")
REGION_HINTS = (
    "global",
    "china",
    "us",
    "usa",
    "europe",
    "japan",
    "\u5168\u7403",
    "\u4e2d\u56fd",
    "\u7f8e\u56fd",
    "\u6b27\u6d32",
    "\u65e5\u672c",
    "\u6d77\u5916",
)


def _text_blob(*values: Any) -> str:
    return " ".join(str(value or "") for value in values)


def _contains_any(text: str, terms: Sequence[str]) -> bool:
    lower = text.lower()
    return any(term.lower() in lower for term in terms if term)


def _metric_kind(metric: str, fact: str) -> str:
    text = _text_blob(metric, fact)
    if _contains_any(text, SHARE_TERMS):
        return "market_share"
    if _contains_any(text, GROWTH_TERMS):
        return "growth"
    if _contains_any(text, SIZE_TERMS):
        return "market_size"
    if _contains_any(text, SHIPMENT_TERMS):
        return "shipment"
    if _contains_any(text, REVENUE_TERMS):
        return "revenue"
    if _contains_any(text, MARKET_TERMS):
        return "market_metric"
    return ""


def _parse_number(value: Any) -> Optional[float]:
    text = str(value or "").replace(",", "")
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _extract_year(*values: Any) -> Optional[int]:
    text = _text_blob(*values)
    years = [int(item) for item in re.findall(r"\b(20\d{2}|19\d{2})\b", text)]
    if not years:
        return None
    return max(years)


def _normalize_unit(unit: Any, value: Any, fact: Any) -> str:
    text = _text_blob(unit, value, fact).lower()
    if "%" in text or "percent" in text or "\u767e\u5206\u6bd4" in text:
        return "%"
    if "\u4ebf\u7f8e\u5143" in text or "usd" in text or "dollar" in text:
        return "USD"
    if "\u4ebf\u5143" in text or "\u4eba\u6c11\u5e01" in text or "rmb" in text or "cny" in text:
        return "RMB"
    if "\u4e07\u53f0" in text or "\u53f0" in text or "unit" in text:
        return "units"
    if "\u4e07\u7247" in text or "\u5343\u7247" in text or "wafer" in text:
        return "wafers"
    cleaned = compact(unit, 40)
    return cleaned


_SCOPE_QUESTION_MARKERS = ("?", "？", "如何", "怎样", "怎么", "能否", "能不能", "是否", "有没有", "哪些")


def _is_scope_polluted(value: str) -> bool:
    """A scope cell should be a short geo/segment label like "中国" or "全球".

    When the value is over 40 chars or contains question-style markers, it is
    almost certainly leaked chapter_question text (e.g. "规模、增速和价格信号是否支持
    机会判断？") and must be dropped to avoid polluting the metric table.
    """
    if not value:
        return False
    if len(value) > 40:
        return True
    return any(marker in value for marker in _SCOPE_QUESTION_MARKERS)


def _scope_from_text(scope: Any, subject: Any, fact: Any) -> str:
    explicit = compact(scope, 80)
    if explicit and not _is_scope_polluted(explicit):
        return explicit
    text = _text_blob(subject, fact)
    for hint in REGION_HINTS:
        if hint and hint.lower() in text.lower():
            return hint
    candidate = compact(subject, 80)
    if candidate and _is_scope_polluted(candidate):
        return ""
    return candidate


def _source_ref(item: Dict[str, Any]) -> str:
    for value in (
        item.get("evidence_ref"),
        item.get("source_ref"),
        item.get("citation_ref"),
        item.get("ref"),
        item.get("evidence_id"),
        *as_list(item.get("source_refs")),
    ):
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _row_key(row: Dict[str, Any]) -> str:
    return re.sub(
        r"\s+",
        "",
        "|".join(
            str(row.get(key) or "").lower()
            for key in ("chapter_id", "kind", "metric_name", "scope", "period", "value_display", "evidence_ref")
        ),
    )


def _metric_from_payload(item: Dict[str, Any], *, chapter_id: str = "", chapter_title: str = "") -> Optional[Dict[str, Any]]:
    raw = as_dict(item.get("raw"))
    metric_name = compact(item.get("metric_name") or item.get("metric") or item.get("indicator"), 100)
    value_display = compact(item.get("value") or item.get("display_value"), 120)
    fact = compact(item.get("fact") or item.get("clean_fact") or item.get("content") or item.get("text"), 320)
    if not metric_name and not value_display and not fact:
        return None
    kind = _metric_kind(metric_name, fact)
    if not kind:
        return None
    period = compact(item.get("period") or raw.get("period") or raw.get("date"), 80)
    year = _extract_year(period, fact)
    unit = _normalize_unit(item.get("unit") or raw.get("unit"), value_display, fact)
    value_number = _parse_number(value_display or fact)
    scope = _scope_from_text(item.get("scope") or raw.get("scope") or raw.get("region"), item.get("subject"), fact)
    evidence_ref = _source_ref(item)
    return {
        "chapter_id": compact(item.get("chapter_id") or chapter_id, 80),
        "chapter_title": compact(item.get("chapter_title") or chapter_title, 160),
        "kind": kind,
        "metric_name": metric_name or kind,
        "subject": compact(item.get("subject") or item.get("company") or raw.get("subject"), 100),
        "scope": scope,
        "period": period or (str(year) if year else ""),
        "year": year,
        "unit": unit,
        "cagr_eligible": bool(unit and scope),
        "value_display": value_display or fact,
        "value_number": value_number,
        "source_level": compact(item.get("source_level"), 20),
        "confidence": item.get("confidence"),
        "evidence_ref": evidence_ref,
        "evidence_refs": [evidence_ref] if evidence_ref else [],
        "fact": fact,
    }


def _iter_candidate_items(
    *,
    chapter_evidence_packages: Sequence[Dict[str, Any]],
    metric_normalization_table: Sequence[Dict[str, Any]],
) -> Iterable[Dict[str, Any]]:
    for item in list(metric_normalization_table or []):
        if isinstance(item, dict):
            yield item
    for package in list(chapter_evidence_packages or []):
        if not isinstance(package, dict):
            continue
        chapter_id = str(package.get("chapter_id") or "")
        chapter_title = str(package.get("chapter_title") or "")
        for collection in ("core_evidence", "supporting_evidence", "table_evidence", "evidence_items"):
            for item in as_list(package.get(collection)):
                if isinstance(item, dict):
                    yield {**item, "chapter_id": item.get("chapter_id") or chapter_id, "chapter_title": item.get("chapter_title") or chapter_title}


def _collect_market_metrics(
    *,
    chapter_evidence_packages: Sequence[Dict[str, Any]],
    metric_normalization_table: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    metrics: List[Dict[str, Any]] = []
    seen = set()
    for item in _iter_candidate_items(
        chapter_evidence_packages=chapter_evidence_packages,
        metric_normalization_table=metric_normalization_table,
    ):
        row = _metric_from_payload(item)
        if not row:
            continue
        key = _row_key(row)
        if key in seen:
            continue
        seen.add(key)
        metrics.append(row)
    metrics.sort(
        key=lambda row: (
            str(row.get("chapter_id") or ""),
            int(row.get("year") or 0),
            str(row.get("kind") or ""),
        )
    )
    return metrics


def _cagr_group_key(row: Dict[str, Any]) -> Tuple[str, str, str, str]:
    return (
        str(row.get("kind") or ""),
        str(row.get("metric_name") or "").lower(),
        str(row.get("scope") or "").lower(),
        str(row.get("unit") or "").lower(),
    )


def _looks_like_year_value(value_number: float, unit: str, fact: str) -> bool:
    """Detect that a "value" was actually a year that leaked into value_number.

    Triggers when:
    - value is in [1900, 2099]
    - and unit is generic ("raw" / empty / "year" / "年")
    - and the fact text does not carry any magnitude word that would justify it.
    """
    if not 1900 <= value_number <= 2099:
        return False
    norm_unit = (unit or "").strip().lower()
    if norm_unit not in {"", "raw", "year", "年", "yr"}:
        return False
    magnitude_words = ("亿", "万", "千", "百", "%", "billion", "million", "thousand", "trillion")
    return not any(w in (fact or "") for w in magnitude_words)


def _ratio_too_extreme(start: float, end: float) -> bool:
    """When start vs end differ by >5x we are almost certainly mixing units
    (e.g. 亿元 vs 万亿元 collapsed to the same unit family) and the resulting
    CAGR will be a nonsense -99% / +2000% number that contaminates the table."""
    if start <= 0 or end <= 0:
        return True
    ratio = max(start, end) / min(start, end)
    return ratio > 5.0


def _derive_cagr(metrics: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = {}
    for row in metrics:
        if row.get("kind") in {"growth", "market_share"}:
            continue
        if not row.get("cagr_eligible") or not row.get("scope") or not row.get("unit"):
            continue
        if not row.get("year") or not isinstance(row.get("value_number"), (int, float)):
            continue
        value_number = float(row.get("value_number") or 0)
        if value_number <= 0:
            continue
        # Drop rows where the "value" is actually a year mis-extracted by the
        # numeric regex (e.g. "2025" -> value_number=2025.0 with unit=raw).
        if _looks_like_year_value(value_number, str(row.get("unit") or ""), str(row.get("fact") or "")):
            continue
        buckets.setdefault(_cagr_group_key(row), []).append(row)

    calculations: List[Dict[str, Any]] = []
    for key, rows in buckets.items():
        unique_years = {}
        for row in rows:
            unique_years[int(row.get("year"))] = row
        if len(unique_years) < 2:
            continue
        first_year = min(unique_years)
        last_year = max(unique_years)
        first = unique_years[first_year]
        last = unique_years[last_year]
        n = last_year - first_year
        if n <= 0:
            continue
        start = float(first.get("value_number") or 0)
        end = float(last.get("value_number") or 0)
        if start <= 0 or end <= 0:
            continue
        # Skip CAGR rows where the two paired values are >5x apart. This
        # almost always means the two evidence rows are in different magnitudes
        # (亿元 vs 万亿元) but got bucketed together because the unit family
        # collapses them. Showing -99% or +2000% growth here destroys the
        # report's credibility.
        if _ratio_too_extreme(start, end):
            continue
        cagr = math.pow(end / start, 1 / n) - 1
        if n == 1 and abs(cagr) > 2.0:
            continue
        if n >= 5 and abs(cagr) > 0.60:
            continue
        calculations.append(
            {
                "calculation_id": f"CAGR-{len(calculations) + 1:03d}",
                "type": "cagr",
                "metric_name": last.get("metric_name") or key[1],
                "scope": last.get("scope") or first.get("scope"),
                "unit": last.get("unit") or first.get("unit"),
                "start_year": first_year,
                "end_year": last_year,
                "periods": n,
                "start_value": start,
                "end_value": end,
                "result": cagr,
                "result_display": f"{cagr * 100:.1f}%",
                "evidence_refs": dedupe(as_list(first.get("evidence_refs")) + as_list(last.get("evidence_refs")), limit=6),
                "formula": "(end/start)^(1/n)-1",
            }
        )
    return calculations


def _share_rows(metrics: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for item in metrics:
        if item.get("kind") == "market_share" or str(item.get("unit") or "") == "%":
            rows.append(item)
    return rows[:12]


def _format_value_compact(value: Any) -> str:
    """Render a numeric value without spurious trailing `.0` so the CAGR base→end
    column reads `5784 → 12534` rather than `5784.0 -> 12534.0`."""
    if value is None:
        return ""
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        # Trim insignificant trailing zeros (e.g. 1.20 -> 1.2)
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def _metric_table_rows(metrics: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for item in metrics[:12]:
        scope = item.get("scope")
        if scope and _is_scope_polluted(str(scope)):
            scope = ""
        cells = [
            item.get("metric_name"),
            scope,
            item.get("period"),
            item.get("value_display"),
            item.get("unit"),
            item.get("source_level"),
        ]
        row_claim = "该指标须同时披露范围、期间、单位与来源等级,才进入正文判断。"
        rows.append({"cells": [compact(cell, 120) for cell in cells], "row_claim": row_claim, "evidence_refs": as_list(item.get("evidence_refs"))})
    return rows


def _cagr_table_rows(calculations: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for item in calculations[:8]:
        scope = item.get("scope")
        if scope and _is_scope_polluted(str(scope)):
            scope = ""
        start_disp = _format_value_compact(item.get("start_value"))
        end_disp = _format_value_compact(item.get("end_value"))
        unit_disp = item.get("unit") or ""
        if not unit_disp or str(unit_disp).strip().lower() == "raw":
            continue
        base_to_latest = f"{start_disp} → {end_disp}".strip()
        if unit_disp:
            base_to_latest = f"{base_to_latest} {unit_disp}".strip()
        cells = [
            item.get("metric_name"),
            scope,
            f"{item.get('start_year')}-{item.get('end_year')}",
            item.get("result_display"),
            base_to_latest,
        ]
        row_claim = "CAGR 来自基期与末期同口径数值的实际推算,不直接复用未经校验的增速表述。"
        rows.append({"cells": [compact(cell, 120) for cell in cells], "row_claim": row_claim, "evidence_refs": as_list(item.get("evidence_refs"))})
    return rows


def _share_table_rows(metrics: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for item in _share_rows(metrics):
        scope = item.get("scope") or item.get("subject")
        if scope and _is_scope_polluted(str(scope)):
            scope = ""
        cells = [
            item.get("metric_name"),
            scope,
            item.get("period"),
            item.get("value_display"),
            item.get("source_level"),
        ]
        row_claim = "份额类指标只在同范围、同时间窗口内可比,跨范围比较需说明边界。"
        rows.append({"cells": [compact(cell, 120) for cell in cells], "row_claim": row_claim, "evidence_refs": as_list(item.get("evidence_refs"))})
    return rows


def _preferred_chapter_id(metrics: Sequence[Dict[str, Any]], report_blueprint: Dict[str, Any]) -> str:
    for item in metrics:
        if item.get("chapter_id"):
            return str(item.get("chapter_id"))
    for chapter in as_list(report_blueprint.get("chapters")):
        if isinstance(chapter, dict):
            return str(chapter.get("chapter_id") or "market")
    return "market"


def _build_tables(metrics: Sequence[Dict[str, Any]], calculations: Sequence[Dict[str, Any]], report_blueprint: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not metrics:
        return []
    chapter_id = _preferred_chapter_id(metrics, report_blueprint)
    tables: List[Dict[str, Any]] = []
    metric_rows = _metric_table_rows(metrics)
    if len(metric_rows) >= 2:
        tables.append(
            make_table(
                table_id=f"{chapter_id}_market_metrics",
                chapter_id=chapter_id,
                table_type="market_metric_table",
                title="市场指标与口径表",
                headers=["指标", "范围", "期间", "数值", "单位", "来源等级"],
                rows=metric_rows,
                takeaway="市场数据已按指标、范围、期间、单位与来源等级拆开,避免直接合并不同口径。",
                purpose="在表格与正文渲染前对市场指标做口径归一。",
                decision_implication="优先使用口径完整、A/B 等级来源的指标作为量化主线,缺口数据仅作边界或附录。",
                limitations=["表中各行来自已有证据,不会凭空补齐缺失的范围、单位或期间。"],
                analytics_source=AGENT_NAME,
            )
        )
    cagr_rows = _cagr_table_rows(calculations)
    if len(cagr_rows) >= 1:
        tables.append(
            make_table(
                table_id=f"{chapter_id}_cagr_calculation",
                chapter_id=chapter_id,
                table_type="cagr_calculation",
                title="CAGR 推算校验表",
                headers=["指标", "范围", "期间", "CAGR", "基期 → 末期"],
                rows=cagr_rows,
                takeaway="CAGR 行由历史与最新值实际推算得到,而非直接引用,且保留成对的来源引用。",
                purpose="使增长率假设可被审计追溯。",
                decision_implication="只有起止值可追溯、口径与单位一致的增长率,才能支撑市场扩张判断。",
                limitations=["当数值、年份或单位不可比时,该行 CAGR 直接跳过。"],
                analytics_source=AGENT_NAME,
            )
        )
    share_rows = _share_table_rows(metrics)
    if len(share_rows) >= 2:
        tables.append(
            make_table(
                table_id=f"{chapter_id}_market_share",
                chapter_id=chapter_id,
                table_type="regional_share_table",
                title="市场份额与区域/主体拆分表",
                headers=["指标", "区域/主体", "期间", "份额/数值", "来源等级"],
                rows=share_rows,
                takeaway="份额类指标与规模类指标分开呈现,避免比例与金额混在同一口径下比较。",
                purpose="支持区域或主体的份额比较。",
                decision_implication="只有分母清晰时,份额行才适合支撑竞争或区域判断。",
                limitations=["表格保留原始份额披露,不会在缺少明确分母时强行合计 100%。"],
                analytics_source=AGENT_NAME,
            )
        )
    return tables


def run_market_analytics_agent(
    *,
    report_blueprint: Optional[Dict[str, Any]] = None,
    chapter_evidence_packages: Optional[Sequence[Dict[str, Any]]] = None,
    evidence_graph: Optional[Dict[str, Any]] = None,
    metric_normalization_table: Optional[Sequence[Dict[str, Any]]] = None,
    coverage_matrix: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    del evidence_graph, coverage_matrix
    report_blueprint = as_dict(report_blueprint)
    metrics = _collect_market_metrics(
        chapter_evidence_packages=list(chapter_evidence_packages or []),
        metric_normalization_table=list(metric_normalization_table or []),
    )
    calculations = _derive_cagr(metrics)
    tables = _build_tables(metrics, calculations, report_blueprint)
    warnings: List[str] = []
    if not metrics:
        warnings.append("No market metrics with usable numeric or share signals were found.")
    if metrics and not calculations:
        warnings.append("No comparable start/end market values were available for CAGR calculation.")
    summary = (
        f"Extracted {len(metrics)} market metric rows, "
        f"derived {len(calculations)} CAGR checks, and prepared {len(tables)} analytics tables."
    )
    return make_output(
        agent=AGENT_NAME,
        analytics_type="market_analytics",
        summary=summary,
        metrics=metrics,
        calculations=calculations,
        tables=tables,
        warnings=warnings,
        trace={
            "metric_count": len(metrics),
            "calculation_count": len(calculations),
            "table_count": len(tables),
        },
    )
