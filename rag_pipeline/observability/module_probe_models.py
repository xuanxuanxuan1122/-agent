from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

SCHEMA_VERSION = "module_probe_event_v1"
VALID_EVENT_TYPES = {"input_received", "transform_result", "decision_observed"}
SENSITIVE_KEY_FRAGMENTS = (
    "api_key",
    "apikey",
    "authorization",
    "password",
    "secret",
    "token",
    "prompt",
    "raw_html",
    "raw_text",
    "raw_content",
    "maintext",
    "page_content",
)


def as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if value is None:
        return []
    return [value]


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_rate(numerator: Any, denominator: Any) -> float:
    den = safe_float(denominator)
    if den <= 0:
        return 0.0
    return round(safe_float(numerator) / den, 4)


def sanitize(value: Any, *, key: str = "", depth: int = 0) -> Any:
    if _is_sensitive_key(key):
        return "[redacted]"
    if depth > 5:
        return "[truncated]"
    if isinstance(value, dict):
        return {str(k): sanitize(v, key=str(k), depth=depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(item, depth=depth + 1) for item in value[:60]]
    if isinstance(value, tuple):
        return [sanitize(item, depth=depth + 1) for item in list(value)[:60]]
    if isinstance(value, str):
        text = value.strip()
        if len(text) > 500:
            return text[:500] + "...[truncated]"
        return text
    return value


def _is_sensitive_key(key: Any) -> bool:
    text = str(key or "").strip().lower()
    if text in {"total_tokens", "input_tokens", "output_tokens", "prompt_tokens", "completion_tokens"}:
        return False
    return any(fragment in text for fragment in SENSITIVE_KEY_FRAGMENTS)


def make_module_probe_event(
    *,
    run_id: str,
    seq: int,
    stage: str,
    module: Optional[str] = None,
    event_type: str,
    status: str = "ok",
    input_count: int = 0,
    output_count: int = 0,
    drop_count: int = 0,
    reason_counts: Optional[Dict[str, Any]] = None,
    id_coverage: Optional[Dict[str, Any]] = None,
    lineage_edges: Optional[Iterable[Any]] = None,
    cache: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    normalized_event = str(event_type or "transform_result").strip().lower()
    if normalized_event not in VALID_EVENT_TYPES:
        normalized_event = "transform_result"
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": str(run_id or "run").strip() or "run",
        "seq": safe_int(seq),
        "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
        "stage": str(stage or "unknown").strip() or "unknown",
        "module": str(module or stage or "unknown").strip() or "unknown",
        "event_type": normalized_event,
        "status": str(status or "ok").strip().lower() or "ok",
        "input": {
            "count": max(0, safe_int(input_count)),
            "id_coverage": sanitize(id_coverage or {}),
        },
        "output": {
            "count": max(0, safe_int(output_count)),
            "id_coverage": sanitize(id_coverage or {}),
        },
        "drop": {
            "count": max(0, safe_int(drop_count)),
            "reason_counts": sanitize(reason_counts or {}),
        },
        "cache": {
            "hit_count": safe_int(as_dict(cache).get("cache_hit_count") or as_dict(cache).get("hit_count")),
            "miss_count": safe_int(as_dict(cache).get("cache_miss_count") or as_dict(cache).get("miss_count")),
            "stale_count": safe_int(as_dict(cache).get("stale_count")),
            "quarantined_count": safe_int(as_dict(cache).get("quarantined_count")),
        },
        "lineage_edges": sanitize(list(lineage_edges or [])),
        "metrics": sanitize(metrics or {}),
        "diagnostics": sanitize(diagnostics or {}),
        "diagnostic_only": True,
        "must_not_render": True,
        "public_text_allowed": False,
    }
