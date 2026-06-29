from __future__ import annotations

import copy
import hashlib
import json
import os
import threading
from collections import OrderedDict
from typing import Any, Dict, Tuple


SCHEMA_VERSION = "stage_execution_guard_v1"

_LOCK = threading.RLock()
_EXECUTION_STATS: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
_OUTPUT_CACHE: "OrderedDict[Tuple[str, str, str], Any]" = OrderedDict()


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, min_value: int = 1, max_value: int = 10000) -> int:
    try:
        value = int(os.getenv(name, str(default)) or default)
    except (TypeError, ValueError):
        value = default
    return max(min_value, min(max_value, value))


def _json_safe_default(value: Any) -> Any:
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return dict(value.__dict__)
        except Exception:
            pass
    return str(value)


def _raw_run_id() -> str:
    return str(
        os.getenv("RUNTIME_PROBE_RUN_ID")
        or os.getenv("REPORT_STAGE_SNAPSHOT_RUN_ID")
        or os.getenv("REPORT_RUN_ID")
        or os.getenv("STAGE_EXECUTION_GUARD_RUN_ID")
        or ""
    ).strip()


def current_run_id() -> str:
    return _raw_run_id() or "default"


def has_explicit_run_context() -> bool:
    return bool(_raw_run_id())


def stable_stage_hash(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=_json_safe_default)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stage_guard_enabled() -> bool:
    return _env_flag("REPORT_STAGE_EXECUTION_GUARD_ENABLED", True)


def stage_cache_enabled(stage: str) -> bool:
    if not stage_guard_enabled():
        return False
    if not has_explicit_run_context() and not _env_flag("REPORT_STAGE_GUARD_ALLOW_DEFAULT_RUN_CACHE", False):
        return False
    if not _env_flag("REPORT_STAGE_EXECUTION_GUARD_CACHE_ENABLED", True):
        return False
    stage_key = "".join(ch if ch.isalnum() else "_" for ch in str(stage or "").upper()).strip("_")
    return _env_flag(f"REPORT_STAGE_GUARD_CACHE_{stage_key}", True)


def reset_stage_execution_guard() -> None:
    with _LOCK:
        _EXECUTION_STATS.clear()
        _OUTPUT_CACHE.clear()


def record_stage_execution(
    *,
    stage: str,
    input_hash: str,
    run_id: str | None = None,
    cache_hit: bool = False,
) -> Dict[str, Any]:
    safe_run_id = str(run_id or current_run_id()).strip() or "default"
    safe_stage = str(stage or "unknown").strip() or "unknown"
    safe_hash = str(input_hash or "").strip() or "missing"
    key = (safe_run_id, safe_stage, safe_hash)
    with _LOCK:
        stats = _EXECUTION_STATS.setdefault(
            key,
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": safe_run_id,
                "stage": safe_stage,
                "input_hash": safe_hash,
                "invocation_count": 0,
                "cache_hit_count": 0,
            },
        )
        stats["invocation_count"] = int(stats.get("invocation_count") or 0) + 1
        if cache_hit:
            stats["cache_hit_count"] = int(stats.get("cache_hit_count") or 0) + 1
        stats["duplicate_stage_execution"] = int(stats.get("invocation_count") or 0) > 1
        stats["cache_hit"] = bool(cache_hit)
        return dict(stats)


def get_cached_stage_output(*, stage: str, input_hash: str, run_id: str | None = None) -> Dict[str, Any]:
    safe_run_id = str(run_id or current_run_id()).strip() or "default"
    safe_stage = str(stage or "unknown").strip() or "unknown"
    safe_hash = str(input_hash or "").strip() or "missing"
    key = (safe_run_id, safe_stage, safe_hash)
    with _LOCK:
        if key not in _OUTPUT_CACHE:
            return {"hit": False, "run_id": safe_run_id, "stage": safe_stage, "input_hash": safe_hash}
        value = copy.deepcopy(_OUTPUT_CACHE[key])
        _OUTPUT_CACHE.move_to_end(key)
    meta = record_stage_execution(stage=safe_stage, input_hash=safe_hash, run_id=safe_run_id, cache_hit=True)
    return {"hit": True, "output": value, **meta}


def store_stage_output(*, stage: str, input_hash: str, output: Any, run_id: str | None = None) -> Dict[str, Any]:
    safe_run_id = str(run_id or current_run_id()).strip() or "default"
    safe_stage = str(stage or "unknown").strip() or "unknown"
    safe_hash = str(input_hash or "").strip() or "missing"
    key = (safe_run_id, safe_stage, safe_hash)
    with _LOCK:
        _OUTPUT_CACHE[key] = copy.deepcopy(output)
        _OUTPUT_CACHE.move_to_end(key)
        max_entries = _env_int("REPORT_STAGE_GUARD_MAX_ENTRIES", 128, min_value=1, max_value=10000)
        while len(_OUTPUT_CACHE) > max_entries:
            _OUTPUT_CACHE.popitem(last=False)
    return {"stored": True, "run_id": safe_run_id, "stage": safe_stage, "input_hash": safe_hash}


def get_stage_execution_summary(run_id: str | None = None) -> Dict[str, Any]:
    safe_run_id = str(run_id or current_run_id()).strip() or "default"
    with _LOCK:
        rows = [dict(value) for key, value in _EXECUTION_STATS.items() if key[0] == safe_run_id]
        cache_size = sum(1 for key in _OUTPUT_CACHE if key[0] == safe_run_id)
    duplicate_rows = [row for row in rows if row.get("duplicate_stage_execution")]
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": safe_run_id,
        "execution_count": len(rows),
        "duplicate_stage_count": len(duplicate_rows),
        "cache_entry_count": cache_size,
        "cache_hit_count": sum(int(row.get("cache_hit_count") or 0) for row in rows),
        "stages": rows,
    }
