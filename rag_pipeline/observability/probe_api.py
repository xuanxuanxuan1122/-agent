from __future__ import annotations

import threading
from typing import Any, Dict, Optional

from rag_pipeline.observability.module_probe_models import as_dict, as_list, safe_int
from rag_pipeline.observability.probe_bus import RuntimeProbeBus
from rag_pipeline.observability.probe_context import ProbeContext

_BUS_LOCK = threading.Lock()
_BUS_BY_PATH: Dict[str, RuntimeProbeBus] = {}


def _bus(probe: Optional[ProbeContext]) -> Optional[RuntimeProbeBus]:
    if probe is None:
        return None
    key = str(probe.live_path)
    with _BUS_LOCK:
        bus = _BUS_BY_PATH.get(key)
        if bus is None or bus.context != probe:
            bus = RuntimeProbeBus(probe)
            _BUS_BY_PATH[key] = bus
        return bus


def emit_input(
    probe: Optional[ProbeContext],
    *,
    stage: str,
    module: str,
    input_count: int,
    id_coverage: Optional[Dict[str, Any]] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    bus = _bus(probe)
    if bus is None:
        return {"enabled": False, "emitted": False, "reason": "missing_probe_context"}
    return bus.emit(
        stage=stage,
        module=module,
        event_type="input_received",
        input_count=input_count,
        id_coverage=id_coverage or {},
        diagnostics=diagnostics or {},
    )


def emit_transform(
    probe: Optional[ProbeContext],
    *,
    stage: str,
    module: str,
    input_count: int,
    output_count: int,
    drop_count: int = 0,
    status: str = "ok",
    reason_counts: Optional[Dict[str, Any]] = None,
    id_coverage: Optional[Dict[str, Any]] = None,
    cache: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    bus = _bus(probe)
    if bus is None:
        return {"enabled": False, "emitted": False, "reason": "missing_probe_context"}
    return bus.emit(
        stage=stage,
        module=module,
        event_type="transform_result",
        status=status,
        input_count=input_count,
        output_count=output_count,
        drop_count=drop_count,
        reason_counts=reason_counts or {},
        id_coverage=id_coverage or {},
        cache=cache or {},
        metrics=metrics or {},
        diagnostics=diagnostics or {},
    )


def emit_decision(
    probe: Optional[ProbeContext],
    *,
    stage: str,
    module: str,
    input_count: int = 0,
    output_count: int = 0,
    drop_count: int = 0,
    status: str = "warning",
    reason_counts: Optional[Dict[str, Any]] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    bus = _bus(probe)
    if bus is None:
        return {"enabled": False, "emitted": False, "reason": "missing_probe_context"}
    return bus.emit(
        stage=stage,
        module=module,
        event_type="decision_observed",
        status=status,
        input_count=input_count,
        output_count=output_count,
        drop_count=drop_count,
        reason_counts=reason_counts or {},
        diagnostics={
            "observation_only": True,
            **as_dict(diagnostics),
        },
    )


def _payload_count(payload: Any) -> int:
    if isinstance(payload, dict):
        for key in (
            "analysis_ready_evidence",
            "clean_evidence_list",
            "claim_units",
            "sections",
            "chapter_packages",
            "table_packages",
            "errors",
            "warnings",
        ):
            value = payload.get(key)
            if isinstance(value, list):
                return len(value)
        return 1 if payload else 0
    if isinstance(payload, list):
        return len(payload)
    return 1 if payload is not None else 0


def emit_stage_snapshot(
    probe: Optional[ProbeContext],
    *,
    stage_name: str,
    payload: Any,
    snapshot_result: Optional[Dict[str, Any]] = None,
    summary: Optional[Dict[str, Any]] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    result = as_dict(snapshot_result)
    stored = bool(result.get("stored", True))
    payload_count = _payload_count(payload)
    output_count = payload_count if stored else 0
    return emit_transform(
        probe,
        stage=f"stage_snapshot:{stage_name}",
        module="stage_snapshot_cache",
        input_count=payload_count,
        output_count=output_count,
        drop_count=max(0, payload_count - output_count),
        status="ok" if stored else "warning",
        reason_counts={"snapshot_not_stored": 0 if stored else 1},
        metrics={"full_payload_bytes": safe_int(result.get("full_payload_bytes"))},
        diagnostics={
            "stage_snapshot": True,
            "stage_name": stage_name,
            "payload_type": type(payload).__name__,
            "summary": as_dict(summary),
            "diagnostics": as_dict(diagnostics),
            "snapshot_result": {
                "stored": stored,
                "replayable": bool(result.get("replayable")),
                "full_payload_bytes": safe_int(result.get("full_payload_bytes")),
                "full_payload_too_large": bool(result.get("full_payload_too_large")),
                "error": result.get("error") or "",
            },
            "list_count": len(as_list(payload)) if isinstance(payload, list) else 0,
        },
    )
