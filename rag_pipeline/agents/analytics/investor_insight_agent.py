from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .analytics_contracts import as_list, compact, make_output, make_table
from .evidence_utils import (
    contains_any,
    evidence_subject,
    evidence_text,
    extract_year,
    iter_candidate_items,
    parse_number,
    row_dedupe_key,
    source_level,
    source_ref,
)


AGENT_NAME = "investor_insight_agent"

INVESTOR_TERMS = (
    "revenue",
    "profit",
    "gross margin",
    "cash flow",
    "ebitda",
    "ipo",
    "funding",
    "financing",
    "m&a",
    "merger",
    "acquisition",
    "valuation",
    "pe",
    "ps",
    "cagr",
    "market share",
    "\u6536\u5165",
    "\u8425\u6536",
    "\u5229\u6da6",
    "\u6bdb\u5229\u7387",
    "\u73b0\u91d1\u6d41",
    "\u878d\u8d44",
    "ipo",
    "\u5e76\u8d2d",
    "\u4f30\u503c",
    "\u5e02\u5360",
)
RISK_TERMS = (
    "risk",
    "decline",
    "loss",
    "debt",
    "sanction",
    "customer concentration",
    "\u98ce\u9669",
    "\u4e8f\u635f",
    "\u4e0b\u6ed1",
    "\u503a\u52a1",
    "\u5ba2\u6237\u96c6\u4e2d",
    "\u5236\u88c1",
)
FINANCIAL_STRENGTH_TERMS = ("profit", "gross margin", "cash flow", "\u5229\u6da6", "\u6bdb\u5229\u7387", "\u73b0\u91d1\u6d41")
MARKET_POTENTIAL_TERMS = ("cagr", "growth", "market share", "funding", "\u589e\u901f", "\u589e\u957f", "\u5e02\u5360", "\u878d\u8d44")


_SOURCE_LEVEL_BONUS = {"a": 18, "b": 10, "c": -8, "d": -22}

_LOW_QUALITY_URL_HINTS = (
    "zhihu.com",
    "baike.baidu.com",
    "zhidao.baidu.com",
    "tieba.baidu.com",
    "wenwen.sogou.com",
    "iask.sina",
    "bbs.",
    "/forum/",
    "/post/",
)


def _investment_score(text: str, value: Optional[float], source_level_code: str = "", source_ref_text: str = "") -> int:
    """Heuristic investor score.

    Previously the score was driven entirely by keyword hits and would land
    at 92 for almost every row that contained any investor term (because the
    base 50 + financial 18 + market 16 + value 8 sums to 92), which is why
    the rendered "Investment priority matrix" looked like every row was
    high-priority "优先". Now we also factor in:
    - the evidence source level (A/B uplift, C/D heavy penalty)
    - a hard penalty when the source URL is from forum / Q&A / wiki sites
    """
    score = 50
    if contains_any(text, FINANCIAL_STRENGTH_TERMS):
        score += 18
    if contains_any(text, MARKET_POTENTIAL_TERMS):
        score += 16
    if value is not None and value > 0:
        score += 8
    if contains_any(text, RISK_TERMS):
        score -= 22
    level_key = (source_level_code or "").strip().lower()
    score += _SOURCE_LEVEL_BONUS.get(level_key, 0)
    if source_ref_text:
        haystack = source_ref_text.lower()
        if any(hint in haystack for hint in _LOW_QUALITY_URL_HINTS):
            score -= 25
    return max(0, min(100, score))


def _tier(score: int) -> str:
    if score >= 80:
        return "优先"
    if score >= 60:
        return "观察"
    if score >= 40:
        return "存疑"
    return "剔除"


def _collect_rows(
    *,
    chapter_evidence_packages: Sequence[Dict[str, Any]],
    metric_normalization_table: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen = set()
    for item in iter_candidate_items(
        chapter_evidence_packages=chapter_evidence_packages,
        metric_normalization_table=metric_normalization_table,
    ):
        text = evidence_text(item)
        metric_text = " ".join(
            str(item.get(key) or "")
            for key in ("metric_name", "metric", "indicator", "value", "subject")
        )
        full_text = f"{metric_text} {text}"
        if not contains_any(full_text, INVESTOR_TERMS):
            continue
        ref = source_ref(item)
        value = parse_number(item.get("value") or text)
        level_code = source_level(item)
        score = _investment_score(full_text, value, source_level_code=level_code, source_ref_text=str(ref or ""))
        subject = evidence_subject(item)
        if not subject:
            continue
        row = {
            "chapter_id": compact(item.get("chapter_id"), 80),
            "company": subject,
            "metric": compact(item.get("metric_name") or item.get("metric") or item.get("indicator") or "investment signal", 100),
            "period": compact(item.get("period") or str(extract_year(text) or ""), 80),
            "value_display": compact(item.get("value") or text, 120),
            "score": score,
            "tier": _tier(score),
            "source_level": level_code,
            "evidence_ref": ref,
            "evidence_refs": [ref] if ref else [],
        }
        key = row_dedupe_key(row, ("company", "metric", "period", "value_display", "evidence_ref"))
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    rows.sort(key=lambda row: (-int(row.get("score") or 0), str(row.get("company") or "")))
    # Cap the number of rows that share the same evidence source so a single
    # article cannot produce three rows in the priority matrix (which is what
    # used to happen with "海光信息 2025 年报发布" appearing 3x).
    capped: List[Dict[str, Any]] = []
    per_source_count: Dict[str, int] = {}
    for row in rows:
        ref_key = str(row.get("evidence_ref") or row.get("company") or "").strip().lower()
        per_source_count[ref_key] = per_source_count.get(ref_key, 0) + 1
        if per_source_count[ref_key] > 1:
            continue
        capped.append(row)
    return capped[:16]


def _table_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result = []
    for row in rows[:10]:
        cells = [
            row.get("company"),
            row.get("metric"),
            row.get("period"),
            row.get("value_display"),
            row.get("score"),
            row.get("tier"),
        ]
        result.append(
            {
                "cells": [compact(cell, 120) for cell in cells],
                "row_claim": "投资优先级综合考虑市场潜力、财务信号、风险暴露与来源等级,所有评分均回链到具体证据。",
                "evidence_refs": as_list(row.get("evidence_refs")),
            }
        )
    return result


def run_investor_insight_agent(
    *,
    report_blueprint: Optional[Dict[str, Any]] = None,
    chapter_evidence_packages: Optional[Sequence[Dict[str, Any]]] = None,
    metric_normalization_table: Optional[Sequence[Dict[str, Any]]] = None,
    market_analytics: Optional[Sequence[Dict[str, Any]]] = None,
    **_: Any,
) -> Dict[str, Any]:
    del report_blueprint, market_analytics
    rows = _collect_rows(
        chapter_evidence_packages=list(chapter_evidence_packages or []),
        metric_normalization_table=list(metric_normalization_table or []),
    )
    tables = []
    if len(rows) >= 2:
        tables.append(
            make_table(
                table_id="investor_priority_matrix",
                chapter_id=rows[0].get("chapter_id") or "investment",
                table_type="investment_priority_table",
                title="投资优先级矩阵",
                headers=["企业/来源", "信号", "期间", "数值", "评分", "分层"],
                rows=_table_rows(rows),
                takeaway="投资排序综合财务、市场、风险与来源等级信号,与正文判断分离,可追溯到具体证据。",
                purpose="为面向投资人的章节提供企业优先级筛选。",
                decision_implication="优先级行可直接用于初筛;观察、存疑、剔除行需附明确边界条件。",
                limitations=["评分为启发式综合分,适合作为排序参考,不作为估值结论。"],
                analytics_source=AGENT_NAME,
            )
        )
    return make_output(
        agent=AGENT_NAME,
        analytics_type="investor_insight",
        summary=f"Prepared {len(rows)} investor signal rows and {len(tables)} tables.",
        metrics=list(rows),
        tables=tables,
        warnings=[] if rows else ["No investor-grade financial or transaction signals were found."],
        trace={"signal_count": len(rows), "table_count": len(tables)},
    )
