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


AGENT_NAME = "technology_roadmap_agent"

TECH_TERMS = (
    "advanced packaging",
    "chiplet",
    "risc-v",
    "riscv",
    "lithography",
    "material",
    "process node",
    "yield",
    "pilot",
    "mass production",
    "roadmap",
    "\u5148\u8fdb\u5c01\u88c5",
    "chiplet",
    "\u5149\u523b",
    "\u6750\u6599",
    "\u5236\u7a0b",
    "\u826f\u7387",
    "\u4e2d\u8bd5",
    "\u91cf\u4ea7",
    "\u6280\u672f\u8def\u7ebf",
)
LOW_TERMS = ("concept", "lab", "prototype", "pilot", "\u5b9e\u9a8c\u5ba4", "\u6837\u673a", "\u4e2d\u8bd5")
HIGH_TERMS = ("mass production", "commercial", "qualified", "volume", "\u91cf\u4ea7", "\u5546\u7528", "\u5bfc\u5165")
IMPACT_TERMS = ("cost", "performance", "yield", "power", "supply", "\u6210\u672c", "\u6027\u80fd", "\u826f\u7387", "\u529f\u8017", "\u4f9b\u5e94")
CATEGORY_TERMS = {
    "Packaging": ("packaging", "chiplet", "\u5c01\u88c5"),
    "Architecture": ("risc-v", "riscv", "architecture", "\u67b6\u6784"),
    "Equipment": ("lithography", "etch", "deposition", "\u5149\u523b", "\u523b\u8680", "\u8584\u819c"),
    "Materials": ("material", "substrate", "photoresist", "\u6750\u6599", "\u57fa\u677f", "\u5149\u523b\u80f6"),
    "Process": ("process node", "yield", "\u5236\u7a0b", "\u826f\u7387"),
}


def _maturity(text: str) -> str:
    if contains_any(text, HIGH_TERMS):
        return "High"
    if contains_any(text, LOW_TERMS):
        return "Low"
    return "Medium"


def _category(text: str) -> str:
    for label, terms in CATEGORY_TERMS.items():
        if contains_any(text, terms):
            return label
    return "Technology"


def _collect_rows(*, chapter_evidence_packages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen = set()
    for item in iter_candidate_items(chapter_evidence_packages=chapter_evidence_packages):
        text = evidence_text(item)
        metric_text = " ".join(str(item.get(key) or "") for key in ("title", "subject", "metric", "value"))
        full_text = f"{metric_text} {text}"
        if not contains_any(full_text, TECH_TERMS):
            continue
        ref = source_ref(item)
        row = {
            "chapter_id": compact(item.get("chapter_id"), 80),
            "technology": evidence_subject(item, fallback=first_term(full_text, TECH_TERMS) or "Technology node"),
            "category": _category(full_text),
            "maturity": _maturity(full_text),
            "period": compact(item.get("period") or str(extract_year(full_text) or ""), 80),
            "impact": compact(text if contains_any(full_text, IMPACT_TERMS) else item.get("value") or text, 160),
            "source_level": source_level(item),
            "evidence_ref": ref,
            "evidence_refs": [ref] if ref else [],
        }
        key = row_dedupe_key(row, ("technology", "category", "maturity", "period", "impact", "evidence_ref"))
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    rows.sort(key=lambda row: ({"High": 0, "Medium": 1, "Low": 2}.get(str(row.get("maturity")), 3), str(row.get("category") or "")))
    return rows[:14]


def _table_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result = []
    for row in rows[:10]:
        cells = [
            row.get("technology"),
            row.get("category"),
            row.get("maturity"),
            row.get("period"),
            row.get("impact"),
        ]
        result.append(
            {
                "cells": [compact(cell, 120) for cell in cells],
                "row_claim": "Technology maturity is usable only with a visible category, timing signal, and cited evidence.",
                "evidence_refs": as_list(row.get("evidence_refs")),
            }
        )
    return result


def run_technology_roadmap_agent(
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
                table_id="technology_roadmap_matrix",
                chapter_id=rows[0].get("chapter_id") or "technology",
                table_type="technology_roadmap",
                title="Technology roadmap matrix",
                headers=["Technology", "Category", "Maturity", "Period", "Impact"],
                rows=_table_rows(rows),
                takeaway="Technology evidence is mapped by category and maturity before it is used to support breakthrough or substitution claims.",
                purpose="Support technology-roadmap sections.",
                decision_implication="High-maturity rows can support near-term adoption; low-maturity rows should be framed as option value or monitoring targets.",
                limitations=["Maturity is a rules-based estimate from disclosed signals and should not replace expert technical diligence."],
                analytics_source=AGENT_NAME,
            )
        )
    return make_output(
        agent=AGENT_NAME,
        analytics_type="technology_roadmap",
        summary=f"Prepared {len(rows)} technology roadmap rows and {len(tables)} tables.",
        matrices=list(rows),
        tables=tables,
        warnings=[] if rows else ["No technology roadmap, maturity, or process-node signals were found."],
        trace={"row_count": len(rows), "table_count": len(tables)},
    )
