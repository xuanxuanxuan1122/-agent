from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence

from rag_pipeline.contracts.source_strategy import source_strategy_for_role


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return []


def _dedupe(values: Iterable[Any], *, limit: int = 16) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def _role(task: Dict[str, Any]) -> str:
    return str(task.get("proof_role") or task.get("evidence_type") or "support").strip().lower() or "support"


def _gap_type(task: Dict[str, Any]) -> str:
    return str(task.get("gap_type") or task.get("type") or "").strip().lower()


def _field_focus(task: Dict[str, Any], *, missing_fields: Sequence[Any] | None = None) -> str:
    missing = [str(item or "").strip().lower() for item in _as_list(list(missing_fields or [])) if str(item or "").strip()]
    gap_type = _gap_type(task)
    role = _role(task)
    for field in ("period", "unit", "source", "value", "metric", "scope"):
        if field in missing or field in gap_type:
            return field
    if role == "counter" or "counter" in gap_type:
        return "counter_signal"
    if "source" in gap_type or "citation" in gap_type:
        return "source"
    if role == "metric":
        return "period"
    return "source"


def _repair_route(task: Dict[str, Any], role: str) -> str:
    gap_type = _gap_type(task)
    if role == "counter" or "counter" in gap_type:
        return "counter_evidence_search"
    if role == "metric" or "metric" in gap_type:
        return "metric_source_search"
    if "source" in gap_type or "citation" in gap_type:
        return "source_trace_search"
    if role in {"case", "customer_case"}:
        return "case_source_search"
    return "evidence_search"


def _query_without_avoid(base_parts: Sequence[Any], avoid_queries: Sequence[Any]) -> str:
    avoid_keys = {re.sub(r"\s+", " ", str(item or "").strip()).lower() for item in avoid_queries if str(item or "").strip()}
    parts = []
    for part in base_parts:
        text = re.sub(r"\s+", " ", str(part or "").strip())
        if not text:
            continue
        if text.lower() in avoid_keys:
            continue
        parts.append(text)
    return " ".join(_dedupe(parts, limit=18))[:240].rstrip()


def dispatch_repair_seed(task: Dict[str, Any], *, failed_queries: Sequence[Any] | None = None) -> Dict[str, Any]:
    payload = dict(_as_dict(task))
    role = _role(payload)
    strategy = source_strategy_for_role(role, overrides=_as_dict(payload.get("source_strategy")))
    focus = str(payload.get("required_field_focus") or _field_focus(payload, missing_fields=payload.get("missing_fields") or payload.get("missing"))).strip()
    avoid_queries = _dedupe(failed_queries or payload.get("avoid_queries") or [], limit=12)
    enhancers = _dedupe([*_as_list(payload.get("query_enhancers")), *strategy.get("query_enhancers", [])], limit=12)
    query = _query_without_avoid(
        [
            payload.get("query") or payload.get("suggested_query") or payload.get("evidence_goal"),
            focus,
            *_as_list(payload.get("required_fields")),
            *enhancers[:4],
        ],
        avoid_queries,
    )
    payload.update(
        {
            "repair_dispatch_version": "repair_dispatcher_v1",
            "repair_route": payload.get("repair_route") or _repair_route(payload, role),
            "required_field_focus": focus,
            "source_strategy": strategy,
            "source_priority": _dedupe([*_as_list(payload.get("source_priority")), *strategy.get("source_priority", [])], limit=12),
            "query_enhancers": enhancers,
            "avoid_queries": avoid_queries,
            "query": query or str(payload.get("query") or payload.get("suggested_query") or "").strip(),
            "allowed_for_writing": False,
        }
    )
    return payload


def rejected_span_repair_summary(rejected_spans: Sequence[Dict[str, Any]], *, search_task: Dict[str, Any]) -> Dict[str, Any]:
    spans = [dict(item) for item in rejected_spans if isinstance(item, dict)]
    reason_counts = Counter(str(item.get("reason") or "unknown").strip() or "unknown" for item in spans)
    missing_counter: Counter[str] = Counter()
    for item in spans:
        for field in _as_list(item.get("missing_fields")):
            text = str(field or "").strip()
            if text:
                missing_counter[text] += 1
    if not spans:
        return {"status": "ok", "reject_reason_counts": {}, "missing_field_counts": {}, "repair_task_seed": {}}
    top_missing = missing_counter.most_common(1)[0][0] if missing_counter else ""
    seed_input = {
        **_as_dict(search_task),
        "missing_fields": [top_missing] if top_missing else [],
        "gap_type": _as_dict(search_task).get("gap_type") or next(iter(reason_counts.keys()), "readpage_rejected_spans"),
    }
    return {
        "summary_version": "rejected_span_repair_summary_v1",
        "status": "needs_repair",
        "reject_reason_counts": dict(reason_counts),
        "missing_field_counts": dict(missing_counter),
        "repair_task_seed": dispatch_repair_seed(seed_input),
    }
