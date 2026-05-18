from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Sequence


def as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def compact(value: Any, max_chars: int = 180) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "..."


def dedupe(values: Iterable[Any], *, limit: int = 20, max_chars: int = 180) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        text = compact(value, max_chars)
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


def evidence_refs_from_rows(rows: Sequence[Dict[str, Any]]) -> List[str]:
    refs: List[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        refs.extend(str(item or "").strip() for item in as_list(row.get("evidence_refs")) if str(item or "").strip())
    return dedupe(refs, limit=50)


def make_table(
    *,
    table_id: str,
    chapter_id: str,
    table_type: str,
    title: str,
    headers: Sequence[Any],
    rows: Sequence[Dict[str, Any]],
    takeaway: str = "",
    purpose: str = "",
    decision_implication: str = "",
    limitations: Sequence[Any] | None = None,
    analytics_source: str = "",
) -> Dict[str, Any]:
    public_rows = [row for row in rows if isinstance(row, dict) and as_list(row.get("cells"))]
    return {
        "table_id": compact(table_id, 120),
        "chapter_id": compact(chapter_id, 80),
        "table_type": compact(table_type or "analytics_table", 80),
        "title": compact(title, 160),
        "purpose": compact(purpose, 240),
        "headers": [compact(header, 80) for header in headers],
        "rows": public_rows,
        "appendix_rows": [],
        "takeaway": compact(takeaway, 360),
        "decision_implication": compact(decision_implication, 360),
        "limitations": [compact(item, 220) for item in list(limitations or []) if compact(item, 220)],
        "analytics_source": compact(analytics_source, 80),
        "evidence_refs": evidence_refs_from_rows(public_rows),
    }


def make_output(
    *,
    agent: str,
    analytics_type: str,
    summary: str = "",
    metrics: Sequence[Dict[str, Any]] | None = None,
    calculations: Sequence[Dict[str, Any]] | None = None,
    matrices: Sequence[Dict[str, Any]] | None = None,
    tables: Sequence[Dict[str, Any]] | None = None,
    warnings: Sequence[Any] | None = None,
    trace: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    metrics = [item for item in list(metrics or []) if isinstance(item, dict)]
    calculations = [item for item in list(calculations or []) if isinstance(item, dict)]
    matrices = [item for item in list(matrices or []) if isinstance(item, dict)]
    tables = [item for item in list(tables or []) if isinstance(item, dict)]
    refs = []
    for item in [*metrics, *calculations, *matrices]:
        refs.extend(as_list(item.get("evidence_refs")))
        ref = item.get("evidence_ref")
        if ref:
            refs.append(ref)
    for table in tables:
        refs.extend(as_list(table.get("evidence_refs")))
    return {
        "agent": compact(agent, 80),
        "analytics_type": compact(analytics_type, 80),
        "summary": compact(summary, 800),
        "metrics": metrics,
        "calculations": calculations,
        "matrices": matrices,
        "tables": tables,
        "warnings": [compact(item, 260) for item in list(warnings or []) if compact(item, 260)],
        "evidence_refs": dedupe(refs, limit=80),
        "trace": as_dict(trace),
    }


def iter_analytics_tables(analytics_outputs: Sequence[Dict[str, Any]] | None) -> List[Dict[str, Any]]:
    tables: List[Dict[str, Any]] = []
    for output in list(analytics_outputs or []):
        if not isinstance(output, dict):
            continue
        for table in as_list(output.get("tables")):
            if isinstance(table, dict):
                tables.append({**table, "analytics_type": output.get("analytics_type"), "analytics_agent": output.get("agent")})
    return tables
