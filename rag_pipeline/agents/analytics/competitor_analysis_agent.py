from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .analytics_contracts import as_list, compact, make_output, make_table
from .evidence_utils import (
    contains_any,
    evidence_subject,
    evidence_text,
    extract_year,
    first_term,
    iter_candidate_items,
    row_dedupe_key,
    source_level,
    source_ref,
)


AGENT_NAME = "competitor_analysis_agent"

COMPETITION_TERMS = (
    "competitor",
    "competition",
    "market share",
    "share",
    "customer",
    "export",
    "capacity",
    "technology",
    "supplier",
    "leader",
    "\u7ade\u4e89",
    "\u7ade\u54c1",
    "\u4efd\u989d",
    "\u5e02\u5360",
    "\u5ba2\u6237",
    "\u51fa\u53e3",
    "\u4ea7\u80fd",
    "\u6280\u672f",
    "\u4f9b\u5e94\u5546",
)
DIMENSION_TERMS = {
    "Market share": ("market share", "share", "\u4efd\u989d", "\u5e02\u5360"),
    "Technology": ("technology", "node", "patent", "r&d", "\u6280\u672f", "\u5de5\u827a", "\u4e13\u5229", "\u7814\u53d1"),
    "Customer exposure": ("customer", "client", "\u5ba2\u6237", "\u5927\u5ba2\u6237"),
    "Export exposure": ("export", "overseas", "tariff", "\u51fa\u53e3", "\u6d77\u5916", "\u5173\u7a0e"),
    "Capacity": ("capacity", "production", "\u4ea7\u80fd", "\u4ea7\u7ebf", "\u91cf\u4ea7"),
}


def _dimension(text: str) -> str:
    for label, terms in DIMENSION_TERMS.items():
        if contains_any(text, terms):
            return label
    return "Competitive signal"


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
        if not contains_any(full_text, COMPETITION_TERMS):
            continue
        ref = source_ref(item)
        dimension = _dimension(full_text)
        subject = evidence_subject(item)
        if not subject:
            continue
        row = {
            "chapter_id": compact(item.get("chapter_id"), 80),
            "company": subject,
            "dimension": dimension,
            "signal": compact(item.get("value") or text, 140),
            "period": compact(item.get("period") or str(extract_year(text) or ""), 80),
            "source_level": source_level(item),
            "keyword": first_term(full_text, COMPETITION_TERMS),
            "evidence_ref": ref,
            "evidence_refs": [ref] if ref else [],
        }
        key = row_dedupe_key(row, ("company", "dimension", "signal", "period", "evidence_ref"))
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    rows.sort(key=lambda row: (str(row.get("company") or ""), str(row.get("dimension") or "")))
    return rows[:18]


def _table_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result = []
    for row in rows[:10]:
        cells = [
            row.get("company"),
            row.get("dimension"),
            row.get("period"),
            row.get("signal"),
            row.get("source_level"),
        ]
        result.append(
            {
                "cells": [compact(cell, 120) for cell in cells],
                "row_claim": "Competitive comparison is valid only within the stated dimension, period, and evidence source level.",
                "evidence_refs": as_list(row.get("evidence_refs")),
            }
        )
    return result


def run_competitor_analysis_agent(
    *,
    report_blueprint: Optional[Dict[str, Any]] = None,
    chapter_evidence_packages: Optional[Sequence[Dict[str, Any]]] = None,
    metric_normalization_table: Optional[Sequence[Dict[str, Any]]] = None,
    **_: Any,
) -> Dict[str, Any]:
    del report_blueprint
    rows = _collect_rows(
        chapter_evidence_packages=list(chapter_evidence_packages or []),
        metric_normalization_table=list(metric_normalization_table or []),
    )
    tables = []
    if len(rows) >= 2:
        tables.append(
            make_table(
                table_id="competitor_comparison_matrix",
                chapter_id=rows[0].get("chapter_id") or "competition",
                table_type="competitor_matrix",
                title="Competitor comparison matrix",
                headers=["Company", "Dimension", "Period", "Signal", "Source level"],
                rows=_table_rows(rows),
                takeaway="Competitive claims are grouped by dimension so market share, technology, customer, and export exposure are not mixed as one score.",
                purpose="Support competitive landscape sections.",
                decision_implication="Use this matrix to identify where a company leads, where exposure is high, and which dimensions need more evidence.",
                limitations=["Signals are extracted from available evidence and do not force full peer coverage."],
                analytics_source=AGENT_NAME,
            )
        )
    return make_output(
        agent=AGENT_NAME,
        analytics_type="competitor_analysis",
        summary=f"Prepared {len(rows)} competitor comparison rows and {len(tables)} tables.",
        matrices=list(rows),
        tables=tables,
        warnings=[] if rows else ["No competitor, share, customer, export, or technology comparison signals were found."],
        trace={"row_count": len(rows), "table_count": len(tables)},
    )
