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


AGENT_NAME = "regulatory_impact_agent"

POLICY_TERMS = (
    "bis",
    "export control",
    "sanction",
    "restriction",
    "regulation",
    "law",
    "act",
    "license",
    "tariff",
    "compliance",
    "\u51fa\u53e3\u7ba1\u5236",
    "\u5236\u88c1",
    "\u6cd5\u89c4",
    "\u653f\u7b56",
    "\u6cd5\u6848",
    "\u8bb8\u53ef",
    "\u5173\u7a0e",
    "\u5408\u89c4",
)
HIGH_RISK_TERMS = (
    "ban",
    "prohibit",
    "sanction",
    "entity list",
    "license required",
    "\u7981\u6b62",
    "\u5236\u88c1",
    "\u5b9e\u4f53\u6e05\u5355",
    "\u8bb8\u53ef",
)
MEDIUM_RISK_TERMS = ("restrict", "review", "tariff", "control", "\u9650\u5236", "\u5ba1\u67e5", "\u5173\u7a0e", "\u7ba1\u5236")
SEGMENT_TERMS = {
    "Advanced chips": ("chip", "semiconductor", "gpu", "ai chip", "\u82af\u7247", "\u534a\u5bfc\u4f53", "\u7b97\u529b"),
    "Equipment": ("equipment", "lithography", "tool", "\u8bbe\u5907", "\u5149\u523b", "\u673a\u53f0"),
    "Materials": ("material", "chemical", "gas", "\u6750\u6599", "\u5316\u5b66\u54c1", "\u7279\u6c14"),
    "Trade": ("export", "import", "tariff", "\u51fa\u53e3", "\u8fdb\u53e3", "\u5173\u7a0e"),
}


def _risk_level(text: str) -> str:
    if contains_any(text, HIGH_RISK_TERMS):
        return "High"
    if contains_any(text, MEDIUM_RISK_TERMS):
        return "Medium"
    return "Low"


def _segment(text: str) -> str:
    for label, terms in SEGMENT_TERMS.items():
        if contains_any(text, terms):
            return label
    return "Business operations"


def _collect_rows(*, chapter_evidence_packages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen = set()
    for item in iter_candidate_items(chapter_evidence_packages=chapter_evidence_packages):
        text = evidence_text(item)
        metric_text = " ".join(str(item.get(key) or "") for key in ("title", "subject", "metric", "value"))
        full_text = f"{metric_text} {text}"
        if not contains_any(full_text, POLICY_TERMS):
            continue
        ref = source_ref(item)
        row = {
            "chapter_id": compact(item.get("chapter_id"), 80),
            "policy": evidence_subject(item, fallback=first_term(full_text, POLICY_TERMS) or "Policy item"),
            "segment": _segment(full_text),
            "time_window": compact(item.get("period") or str(extract_year(full_text) or ""), 80),
            "risk_level": _risk_level(full_text),
            "impact": compact(text, 160),
            "source_level": source_level(item),
            "evidence_ref": ref,
            "evidence_refs": [ref] if ref else [],
        }
        key = row_dedupe_key(row, ("policy", "segment", "time_window", "impact", "evidence_ref"))
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    rows.sort(key=lambda row: ({"High": 0, "Medium": 1, "Low": 2}.get(str(row.get("risk_level")), 3), str(row.get("policy") or "")))
    return rows[:14]


def _table_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result = []
    for row in rows[:10]:
        cells = [
            row.get("policy"),
            row.get("segment"),
            row.get("time_window"),
            row.get("risk_level"),
            row.get("impact"),
        ]
        result.append(
            {
                "cells": [compact(cell, 120) for cell in cells],
                "row_claim": "Policy impact should be interpreted through affected segment, time window, and risk level.",
                "evidence_refs": as_list(row.get("evidence_refs")),
            }
        )
    return result


def run_regulatory_impact_agent(
    *,
    report_blueprint: Optional[Dict[str, Any]] = None,
    chapter_evidence_packages: Optional[Sequence[Dict[str, Any]]] = None,
    **_: Any,
) -> Dict[str, Any]:
    del report_blueprint
    rows = _collect_rows(chapter_evidence_packages=list(chapter_evidence_packages or []))
    tables = []
    if len(rows) >= 2:
        tables.append(
            make_table(
                table_id="regulatory_impact_register",
                chapter_id=rows[0].get("chapter_id") or "regulatory",
                table_type="risk_register",
                title="Regulatory impact register",
                headers=["Policy", "Affected segment", "Time window", "Risk", "Impact"],
                rows=_table_rows(rows),
                takeaway="Policy exposure is organized by affected segment and risk level before it enters the report narrative.",
                purpose="Support regulatory and policy-risk sections.",
                decision_implication="High-risk rows should drive monitoring priorities and scenario boundaries.",
                limitations=["Risk level is a rules-based first pass and should be reviewed against the original policy text."],
                analytics_source=AGENT_NAME,
            )
        )
    return make_output(
        agent=AGENT_NAME,
        analytics_type="regulatory_impact",
        summary=f"Prepared {len(rows)} regulatory impact rows and {len(tables)} tables.",
        matrices=list(rows),
        tables=tables,
        warnings=[] if rows else ["No policy, regulation, export-control, or compliance signals were found."],
        trace={"row_count": len(rows), "table_count": len(tables)},
    )
