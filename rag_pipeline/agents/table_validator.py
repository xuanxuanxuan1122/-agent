from __future__ import annotations

import os
import re
from typing import Any, Dict, List


BANNED_TABLE_HEADERS = {
    "\u5f15\u7528",
    "\u6765\u6e90",
    "\u8d44\u6599\u6765\u6e90",
    "\u5224\u65ad\u7528\u9014",
    "\u62a5\u544a\u4f7f\u7528\u65b9\u5f0f",
    "\u53e3\u5f84",
    "source",
    "sources",
    "ref",
    "refs",
    "evidence",
}
SOURCE_HEADER_PATTERN = re.compile(r"(?:来源|引用|证据|evidence|source|ref)", re.IGNORECASE)


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _numeric_payload_is_packed(cell: str) -> bool:
    numeric_scan = re.sub(r"\b20\d{2}-\d{1,2}-\d{1,2}\b", "", cell)
    numeric_tokens = re.findall(r"\d+(?:\.\d+)?%?", numeric_scan)
    year_markers = re.findall(r"\b(?:19|20)\d{2}\b", numeric_scan)
    has_packed_separator = bool(re.search(r"[,，;；/、]|(?:19|20)\d{2}\s*[:：年]", numeric_scan))
    return bool(has_packed_separator and (len(numeric_tokens) >= 4 or len(year_markers) >= 2))


def _row_field(row: Dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, list):
            value = next((item for item in value if str(item or "").strip()), "")
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _metric_contract_missing(row: Dict[str, Any], table_type: str) -> List[str]:
    missing = []
    metric = _row_field(row, "metric", "metric_name", "indicator")
    value = _row_field(row, "value", "value_display", "result_display", "share")
    source = _row_field(row, "source", "source_ref", "evidence_ref", "ref", "evidence_refs")
    period = _row_field(row, "period", "date", "time_window")
    unit = _row_field(row, "unit")
    if not metric:
        missing.append("metric")
    if not value:
        missing.append("value")
    if not unit:
        missing.append("unit")
    if not period:
        missing.append("period")
    if not source:
        missing.append("source")
    return missing


def validate_table_package(package: Dict[str, Any]) -> Dict[str, Any]:
    headers = [str(item or "").strip() for item in _as_list(package.get("headers"))]
    rows = [row for row in _as_list(package.get("rows")) if isinstance(row, dict)]
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    table_type = str(package.get("table_type") or "").strip()
    row_ref_count = sum(1 for row in rows if _as_list(row.get("evidence_refs")))
    table_refs = _as_list(package.get("evidence_refs"))

    if len(headers) < 2:
        errors.append({"type": "too_few_headers", "message": "Body tables need at least two visible columns."})
    blank_headers = [index for index, header in enumerate(headers, start=1) if not header]
    if blank_headers:
        errors.append({"type": "blank_header", "columns": blank_headers})
    duplicate_headers = sorted({header for header in headers if header and headers.count(header) > 1})
    if duplicate_headers:
        warnings.append({"type": "duplicate_headers", "headers": duplicate_headers})
    minimum_rows = 1 if table_type == "cagr_calculation" else 2
    if len(rows) < minimum_rows:
        errors.append({"type": "too_few_rows", "message": "Body tables need enough rows before rendering.", "minimum_rows": minimum_rows})
    try:
        max_body_rows = int(os.getenv("REPORT_MAX_BODY_TABLE_ROWS", "6") or "6")
    except (TypeError, ValueError):
        max_body_rows = 6
    if len(rows) > max_body_rows:
        warnings.append({"type": "too_many_rows", "message": "Body table rows above the public limit should be truncated or moved to appendix rows."})

    banned = [
        header
        for header in headers
        if header.lower() in BANNED_TABLE_HEADERS
        or header in BANNED_TABLE_HEADERS
        or "\u53e3\u5f84" in header
        or SOURCE_HEADER_PATTERN.search(header)
    ]
    if banned:
        errors.append({"type": "banned_headers", "headers": banned})
    if not table_refs and row_ref_count == 0:
        errors.append({"type": "missing_table_evidence_refs", "message": "Body tables need traceable evidence_refs."})
    if not str(package.get("anchor_section_id") or package.get("claim_id") or package.get("argument_unit_id") or "").strip():
        issue = {"type": "missing_table_anchor", "message": "Body tables should bind to a section, claim, or argument unit."}
        if str(os.getenv("REPORT_REQUIRE_TABLE_ANCHOR", "false")).strip().lower() in {"1", "true", "yes", "on"}:
            errors.append(issue)
        else:
            warnings.append(issue)

    for index, row in enumerate(rows, start=1):
        cells = [str(cell or "") for cell in _as_list(row.get("cells"))]
        if headers and len(cells) != len(headers):
            errors.append(
                {
                    "type": "column_count_mismatch",
                    "row": index,
                    "header_count": len(headers),
                    "cell_count": len(cells),
                }
            )
        if not any(cell.strip() for cell in cells):
            errors.append({"type": "empty_row", "row": index})
        if not str(row.get("row_claim") or "").strip():
            errors.append({"type": "missing_row_claim", "row": index})
        if not _as_list(row.get("evidence_refs")):
            errors.append({"type": "missing_evidence_ref", "row": index})
        for cell in cells:
            if _numeric_payload_is_packed(cell):
                errors.append({"type": "packed_numeric_cell", "row": index, "cell": cell[:120]})
        if table_type in {"market_metric_table", "cagr_calculation", "regional_share_table", "metric_reconciliation"}:
            missing_metric_fields = _metric_contract_missing(row, table_type)
            if missing_metric_fields:
                errors.append({"type": "metric_row_missing_fields", "row": index, "fields": missing_metric_fields})
            if not _row_field(row, "unit") and "unit" not in missing_metric_fields:
                warnings.append({"type": "metric_row_missing_unit", "row": index})

    return {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "row_count": len(rows),
        "header_count": len(headers),
    }
