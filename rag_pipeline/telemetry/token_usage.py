from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


_LOCK = threading.Lock()
_RUN_STARTED_AT = time.time()
_EVENTS: List[Dict[str, Any]] = []
_SUMMARY: Dict[str, Any] = {
    "call_count": 0,
    "known_usage_calls": 0,
    "unknown_usage_calls": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0,
    "by_model": {},
    "by_task": {},
    "by_profile": {},
}


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on", "y"}


def token_monitor_enabled() -> bool:
    return _env_flag("RAG_TOKEN_MONITOR_ENABLED", False)


def _coerce_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None
    return max(0, number)


def normalize_usage(usage: Any) -> Dict[str, Any]:
    payload = usage if isinstance(usage, dict) else {}
    input_tokens = _coerce_int(
        payload.get("input_tokens")
        or payload.get("prompt_tokens")
        or payload.get("prompt_token_count")
        or payload.get("input_token_count")
    )
    output_tokens = _coerce_int(
        payload.get("output_tokens")
        or payload.get("completion_tokens")
        or payload.get("completion_token_count")
        or payload.get("output_token_count")
    )
    total_tokens = _coerce_int(payload.get("total_tokens") or payload.get("total_token_count"))
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    known = any(value is not None for value in (input_tokens, output_tokens, total_tokens))
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "known": known,
        "raw_usage_keys": sorted(str(key) for key in payload.keys()),
    }


def _bucket(summary: Dict[str, Any], key: str, normalized: Dict[str, Any]) -> None:
    bucket = summary.setdefault(key, {})
    name = str(normalized.get(key.removeprefix("by_")) or "unknown").strip() or "unknown"
    item = bucket.setdefault(
        name,
        {
            "call_count": 0,
            "known_usage_calls": 0,
            "unknown_usage_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        },
    )
    item["call_count"] += 1
    if normalized["usage_known"]:
        item["known_usage_calls"] += 1
    else:
        item["unknown_usage_calls"] += 1
    for field in ("input_tokens", "output_tokens", "total_tokens"):
        if normalized.get(field) is not None:
            item[field] += int(normalized[field])


def _jsonl_path() -> Path:
    configured = str(os.getenv("RAG_TOKEN_MONITOR_JSONL_PATH") or "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.cwd() / "traces" / f"token_usage_{os.getpid()}.jsonl"


def _write_jsonl(event: Dict[str, Any]) -> None:
    if not _env_flag("RAG_TOKEN_MONITOR_JSONL", False):
        return
    try:
        path = _jsonl_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
    except Exception:
        return


def _print_event(event: Dict[str, Any]) -> None:
    if not (_env_flag("RAG_TOKEN_MONITOR_PRINT", True) and _env_flag("RAG_TOKEN_MONITOR_PRINT_EACH_CALL", True)):
        return
    task = event.get("task") or "unknown"
    model = event.get("model") or "unknown"
    if not event.get("usage_known"):
        message = f"[TOKEN] task={task} model={model} usage=unknown | call_recorded=true"
    else:
        message = (
            f"[TOKEN] task={task} model={model} "
            f"+in={event.get('input_tokens') or 0} +out={event.get('output_tokens') or 0} "
            f"+total={event.get('total_tokens') or 0} | "
            f"model_total={event.get('model_total_tokens') or 0} | "
            f"run_total={event.get('run_total_tokens') or 0}"
        )
    print(message, file=sys.stderr, flush=True)


def record_llm_usage(
    *,
    usage: Any,
    provider: str = "",
    model: str = "",
    task: str = "",
    profile: str = "",
    api: str = "",
    elapsed_ms: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    if not token_monitor_enabled():
        return None
    normalized_usage = normalize_usage(usage)
    event = {
        "sequence": 0,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "provider": str(provider or "").strip() or "unknown",
        "model": str(model or "").strip() or "unknown",
        "task": str(task or "").strip() or "unknown",
        "profile": str(profile or "").strip() or "unknown",
        "api": str(api or "").strip() or "unknown",
        "usage_known": bool(normalized_usage["known"]),
        "input_tokens": normalized_usage["input_tokens"],
        "output_tokens": normalized_usage["output_tokens"],
        "total_tokens": normalized_usage["total_tokens"],
        "elapsed_ms": elapsed_ms,
        "raw_usage_keys": normalized_usage["raw_usage_keys"],
    }
    with _LOCK:
        _SUMMARY["call_count"] += 1
        if event["usage_known"]:
            _SUMMARY["known_usage_calls"] += 1
        else:
            _SUMMARY["unknown_usage_calls"] += 1
        for field in ("input_tokens", "output_tokens", "total_tokens"):
            if event[field] is not None:
                _SUMMARY[field] += int(event[field])
        bucket_payload = {
            **event,
            "by_model": event["model"],
            "by_task": event["task"],
            "by_profile": event["profile"],
        }
        _bucket(_SUMMARY, "by_model", bucket_payload)
        _bucket(_SUMMARY, "by_task", bucket_payload)
        _bucket(_SUMMARY, "by_profile", bucket_payload)
        event["sequence"] = int(_SUMMARY["call_count"])
        event["run_total_tokens"] = int(_SUMMARY["total_tokens"])
        event["model_total_tokens"] = int(
            _SUMMARY["by_model"].get(event["model"], {}).get("total_tokens", 0)
        )
        _EVENTS.append(dict(event))
        max_events = _coerce_int(os.getenv("RAG_TOKEN_MONITOR_MAX_EVENTS")) or 10000
        if len(_EVENTS) > max_events:
            del _EVENTS[: len(_EVENTS) - max_events]
    _print_event(event)
    _write_jsonl(event)
    return event


def token_usage_summary() -> Dict[str, Any]:
    with _LOCK:
        summary = json.loads(json.dumps(_SUMMARY, ensure_ascii=False))
    summary["enabled"] = token_monitor_enabled()
    summary["run_started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(_RUN_STARTED_AT))
    summary["elapsed_seconds"] = round(time.time() - _RUN_STARTED_AT, 3)
    return summary


def token_usage_events() -> List[Dict[str, Any]]:
    with _LOCK:
        return [dict(event) for event in _EVENTS]


def token_usage_payload() -> Dict[str, Any]:
    return {
        "token_usage_summary": token_usage_summary(),
        "token_usage_events": token_usage_events(),
    }

