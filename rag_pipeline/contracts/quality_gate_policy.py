from __future__ import annotations

import os
from typing import Any


ISOLATED_QUALITY_GATE_MODES = {
    "isolated",
    "observe",
    "observe_only",
    "audit_only",
    "diagnostic",
    "diagnostic_only",
    "bypass",
    "off",
    "disabled",
}

BLOCKING_QUALITY_GATE_MODES = {
    "blocking",
    "enforce",
    "enforced",
    "inline",
    "normal",
    "strict",
}


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on", "y"}


def quality_gate_mode(default: str = "blocking") -> str:
    raw = (
        os.getenv("REPORT_QUALITY_GATE_MODE")
        or os.getenv("REPORT_REVIEW_GATE_MODE")
        or os.getenv("REPORT_GATE_MODE")
        or default
    )
    return str(raw or default).strip().lower()


def quality_gates_isolated(default: bool = False) -> bool:
    mode = quality_gate_mode("isolated" if default else "blocking")
    if mode in ISOLATED_QUALITY_GATE_MODES:
        return True
    if mode in BLOCKING_QUALITY_GATE_MODES:
        return False
    return _env_flag("REPORT_ISOLATE_QUALITY_GATES", default)


def quality_gate_diagnostic(payload: Any) -> Any:
    """Marker helper for callers that preserve diagnostics without enforcement."""

    return payload
