from __future__ import annotations

import json
import logging
import threading
from typing import Any, Dict

from rag_pipeline.observability.module_probe_models import make_module_probe_event, safe_int
from rag_pipeline.observability.probe_context import ProbeContext

logger = logging.getLogger(__name__)


class RuntimeProbeBus:
    """Fail-open append-only writer for runtime module probe events."""

    def __init__(self, context: ProbeContext) -> None:
        self.context = context
        self._lock = threading.Lock()
        self._seq = 0

    def emit(
        self,
        *,
        stage: str,
        module: str,
        event_type: str,
        status: str = "ok",
        input_count: int = 0,
        output_count: int = 0,
        drop_count: int = 0,
        reason_counts: Dict[str, Any] | None = None,
        id_coverage: Dict[str, Any] | None = None,
        cache: Dict[str, Any] | None = None,
        metrics: Dict[str, Any] | None = None,
        diagnostics: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        if not self.context.enabled:
            return {"enabled": False, "emitted": False, "path": str(self.context.live_path)}
        try:
            with self._lock:
                self._seq += 1
                event = make_module_probe_event(
                    run_id=self.context.run_id,
                    seq=self._seq,
                    stage=stage,
                    module=module,
                    event_type=event_type,
                    status=status,
                    input_count=safe_int(input_count),
                    output_count=safe_int(output_count),
                    drop_count=safe_int(drop_count),
                    reason_counts=reason_counts or {},
                    id_coverage=id_coverage or {},
                    cache=cache or {},
                    metrics=metrics or {},
                    diagnostics=diagnostics or {},
                )
                self.context.output_dir.mkdir(parents=True, exist_ok=True)
                with self.context.live_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
            return {"enabled": True, "emitted": True, "path": str(self.context.live_path), "event": event}
        except Exception as exc:  # pragma: no cover - observability cannot block report generation.
            logger.warning("Runtime probe emit failed", extra={"run_id": self.context.run_id, "error": str(exc)})
            return {"enabled": True, "emitted": False, "path": str(self.context.live_path), "error": str(exc)}
