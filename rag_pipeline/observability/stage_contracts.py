from __future__ import annotations

from typing import Any, Dict, Iterable, List

VALID_STAGE_STATUSES = {"ok", "warning", "error", "degraded", "skipped"}
REQUIRED_STAGE_PACKET_FIELDS = (
    "schema_version",
    "run_id",
    "stage",
    "status",
    "input_count",
    "output_count",
    "drop_count",
    "id_coverage",
    "diagnostic_only",
    "must_not_render",
    "public_text_allowed",
)


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return default


def _count_value(payload: Dict[str, Any], field: str) -> Any:
    if field in payload:
        return payload.get(field)
    nested_key = {
        "input_count": "input",
        "output_count": "output",
        "drop_count": "drop",
    }.get(field)
    if nested_key:
        return _as_dict(payload.get(nested_key)).get("count")
    return None


def _has_required_field(payload: Dict[str, Any], field: str) -> bool:
    if field in {"input_count", "output_count", "drop_count"}:
        return _count_value(payload, field) is not None
    if field == "id_coverage":
        return field in payload or "id_coverage" in _as_dict(payload.get("output")) or "id_coverage" in _as_dict(payload.get("input"))
    return field in payload


def _id_coverage_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    return (
        _as_dict(payload.get("id_coverage"))
        or _as_dict(_as_dict(payload.get("output")).get("id_coverage"))
        or _as_dict(_as_dict(payload.get("input")).get("id_coverage"))
    )


def validate_stage_packet(packet: Dict[str, Any]) -> Dict[str, Any]:
    """Validate one diagnostic stage packet without judging business quality."""

    payload = _as_dict(packet)
    errors: List[str] = []
    warnings: List[str] = []
    for field in REQUIRED_STAGE_PACKET_FIELDS:
        if not _has_required_field(payload, field):
            errors.append(f"missing_required_field:{field}")

    status = str(payload.get("status") or "").strip().lower()
    if status and status not in VALID_STAGE_STATUSES:
        errors.append(f"invalid_status:{status}")

    for field in ("input_count", "output_count", "drop_count"):
        if _safe_int(_count_value(payload, field), default=-1) < 0:
            errors.append(f"negative_count:{field}")

    if payload.get("diagnostic_only") is not True:
        errors.append("diagnostic_only_must_be_true")
    if payload.get("must_not_render") is not True:
        errors.append("must_not_render_must_be_true")
    if payload.get("public_text_allowed") is not False:
        errors.append("public_text_allowed_must_be_false")

    id_coverage = _id_coverage_payload(payload)
    for key, value in id_coverage.items():
        try:
            ratio = float(value)
        except (TypeError, ValueError):
            errors.append(f"invalid_id_coverage:{key}")
            continue
        if ratio < 0 or ratio > 1:
            errors.append(f"id_coverage_out_of_range:{key}")

    event_type = str(payload.get("event_type") or "").strip().lower()
    if (
        event_type in {"", "transform_result"}
        and _safe_int(_count_value(payload, "input_count")) > 0
        and _safe_int(_count_value(payload, "output_count")) == 0
    ):
        warnings.append("zero_output_from_nonzero_input")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "stage": str(payload.get("stage") or "unknown"),
    }


def validate_stage_packets(packets: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate stage-probe packet shape and diagnostic-only boundaries."""

    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    packet_count = 0
    stages: List[str] = []
    for index, packet in enumerate(packets):
        packet_count += 1
        result = validate_stage_packet(packet)
        stage = str(result.get("stage") or f"#{index}")
        stages.append(stage)
        if result.get("errors"):
            errors.append({"index": index, "stage": stage, "errors": result.get("errors")})
        if result.get("warnings"):
            warnings.append({"index": index, "stage": stage, "warnings": result.get("warnings")})

    return {
        "ok": not errors,
        "packet_count": packet_count,
        "stages": stages,
        "errors": errors,
        "warnings": warnings,
    }
