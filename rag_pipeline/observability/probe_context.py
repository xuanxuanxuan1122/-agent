from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_output_dir() -> Path:
    raw = str(os.getenv("RUNTIME_PROBE_OUTPUT_DIR") or "output/cache/runtime_probes").strip()
    path = Path(raw)
    if path.is_absolute():
        return path
    return _project_root() / path


@dataclass(frozen=True)
class ProbeContext:
    run_id: str
    output_dir: Path
    base_name: str
    enabled: bool = True

    @property
    def live_path(self) -> Path:
        return self.output_dir / f"{self.base_name}.module_probe.live.jsonl"


def create_probe_context(
    *,
    run_id: str,
    output_dir: Optional[Path | str] = None,
    base_name: Optional[str] = None,
    enabled: Optional[bool] = None,
) -> ProbeContext:
    safe_run_id = str(run_id or "run").strip() or "run"
    root = Path(output_dir) if output_dir is not None else _default_output_dir()
    return ProbeContext(
        run_id=safe_run_id,
        output_dir=root,
        base_name=str(base_name or safe_run_id).strip() or safe_run_id,
        enabled=_env_flag("RUNTIME_PROBE_ENABLED", True) if enabled is None else bool(enabled),
    )


def activate_probe_context_env(context: ProbeContext) -> None:
    """Expose a report probe context to nested agents in the current process."""

    if context.enabled:
        os.environ["RUNTIME_PROBE_RUN_ID"] = context.run_id
        os.environ["RUNTIME_PROBE_BASE_NAME"] = context.base_name
        os.environ["RUNTIME_PROBE_OUTPUT_DIR"] = str(context.output_dir)


def current_probe_context_from_env() -> Optional[ProbeContext]:
    """Rebuild the current runtime probe context for lower-level agents.

    The report orchestrator owns the probe context, but many data transforms
    happen several layers below it. Reconstructing a context from env keeps
    those probes diagnostic-only without threading observability objects
    through every agent signature.
    """

    run_id = str(
        os.getenv("RUNTIME_PROBE_RUN_ID")
        or os.getenv("REPORT_STAGE_SNAPSHOT_RUN_ID")
        or os.getenv("REPORT_RUN_ID")
        or ""
    ).strip()
    if not run_id:
        return None
    base_name = str(os.getenv("RUNTIME_PROBE_BASE_NAME") or run_id).strip() or run_id
    output_dir = str(os.getenv("RUNTIME_PROBE_OUTPUT_DIR") or "").strip()
    return create_probe_context(
        run_id=run_id,
        output_dir=output_dir or None,
        base_name=base_name,
    )
