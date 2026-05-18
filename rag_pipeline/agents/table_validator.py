from __future__ import annotations

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


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _numeric_payload_is_packed(cell: str) -> bool:
    numeric_scan = re.sub(r"\b20\d{2}-\d{1,2}-\d{1,2}\b", "", cell)
    numeric_tokens = re.findall(r"\d+(?:\.\d+)?%?", numeric_scan)
    year_markers = re.findall(r"\b(?:19|20)\d{2}\b", numeric_scan)
    has_packed_separator = bool(re.search(r"[,，;；/、]|(?:19|20)\d{2}\s*[:：年]", numeric_scan))
    return bool(has_packed_separator and (len(numeric_tokens) >= 4 or len(year_markers) >= 2))


def validate_table_package(package: Dict[str, Any]) -> Dict[str, Any]:
    headers = [str(item or "").strip() for item in _as_list(package.get("headers"))]
    rows = [row for row in _as_list(package.get("rows")) if isinstance(row, dict)]
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    if len(headers) < 2:
        errors.append({"type": "too_few_headers", "message": "Body tables need at least two visible columns."})
    blank_headers = [index for index, header in enumerate(headers, start=1) if not header]
    if blank_headers:
        errors.append({"type": "blank_header", "columns": blank_headers})
    duplicate_headers = sorted({header for header in headers if header and headers.count(header) > 1})
    if duplicate_headers:
        warnings.append({"type": "duplicate_headers", "headers": duplicate_headers})
    if len(rows) < 2:
        errors.append({"type": "too_few_rows", "message": "Body tables need at least two rows before rendering."})
    if len(rows) > 8:
        warnings.append({"type": "too_many_rows", "message": "Body tables above eight rows should be truncated or moved to appendix rows."})

    banned = [
        header
        for header in headers
        if header.lower() in BANNED_TABLE_HEADERS
        or header in BANNED_TABLE_HEADERS
        or "\u53e3\u5f84" in header
    ]
    if banned:
        errors.append({"type": "banned_headers", "headers": banned})

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

    return {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "row_count": len(rows),
        "header_count": len(headers),
    }
