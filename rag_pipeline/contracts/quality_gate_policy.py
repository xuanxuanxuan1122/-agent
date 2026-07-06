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

PUBLIC_SIGNAL_EVIDENCE_MODES = {
    "public_signal",
    "signal",
    "signals",
    "public_signals",
    "open_signal",
    "open_source",
    "recall",
    "permissive",
    "complete",
    "draft",
}

ADVISORY_WEIGHT_EVIDENCE_MODES = {
    "advisory",
    "advisory_weight",
    "advisory_weights",
    "weight_only",
    "model_weight",
    "model_weighted",
    "llm_weight",
    "llm_weighted",
    "no_grade_gate",
    "no_source_gate",
    "no_quality_gate",
    "open_evidence",
}

STRICT_RESEARCH_EVIDENCE_MODES = {
    "strict",
    "strict_research",
    "publish_strict",
    "audit",
    "audit_strict",
    "blocking",
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


def evidence_mode(default: str = "strict_research") -> str:
    raw = (
        os.getenv("REPORT_EVIDENCE_MODE")
        or os.getenv("REPORT_SOURCE_MODE")
        or os.getenv("REPORT_RESEARCH_MODE")
        or default
    )
    mode = str(raw or default).strip().lower()
    if mode in ADVISORY_WEIGHT_EVIDENCE_MODES:
        return "advisory_weight"
    if mode in PUBLIC_SIGNAL_EVIDENCE_MODES:
        return "public_signal"
    if mode in STRICT_RESEARCH_EVIDENCE_MODES:
        return "strict_research"
    return mode or default


def public_signal_mode(default: bool = False) -> bool:
    mode = evidence_mode("public_signal" if default else "strict_research")
    if mode == "public_signal":
        return True
    if mode == "advisory_weight":
        return True
    if mode == "strict_research":
        return False
    return _env_flag("REPORT_PUBLIC_SIGNAL_MODE", default)


def advisory_weight_mode(default: bool = False) -> bool:
    mode = evidence_mode("advisory_weight" if default else "strict_research")
    if mode == "advisory_weight":
        return True
    if mode == "public_signal":
        return True
    if mode == "strict_research":
        return False
    return _env_flag("REPORT_ADVISORY_WEIGHT_MODE", default)


def quality_gates_isolated(default: bool = False) -> bool:
    mode = quality_gate_mode("isolated" if default else "blocking")
    if mode in ISOLATED_QUALITY_GATE_MODES:
        return True
    if mode in BLOCKING_QUALITY_GATE_MODES:
        return False
    return _env_flag("REPORT_ISOLATE_QUALITY_GATES", default)


def main_chain_only_mode(default: bool = False) -> bool:
    """Run the report pipeline as a straight analysis -> writer chain.

    In this mode review, QA, sanitizer, final audit, source gates, reformatter,
    and table/render quality logic may still emit diagnostics, but they must not
    edit public markdown or block delivery.
    """

    if _env_flag("REPORT_MAIN_CHAIN_ONLY", default):
        return True
    mode = quality_gate_mode("main_chain" if default else "blocking")
    return mode in {
        "main_chain",
        "main_chain_only",
        "straight_chain",
        "writer_only",
        "no_quality_gates",
        "no_review",
        "no_gates",
    }


def quality_gate_diagnostic(payload: Any) -> Any:
    """Marker helper for callers that preserve diagnostics without enforcement."""

    return payload
