from __future__ import annotations

import copy
from typing import Any, Dict, List

from rag_pipeline.observability.blueprint_evidence_alignment import build_blueprint_evidence_alignment


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _phase_version(phase: str, *, writer_started: bool) -> str:
    if writer_started or phase in {"pre_write_lock", "writer_started", "post_writer_audit"}:
        return "blueprint_locked"
    if phase in {"post_repair", "after_repair"}:
        return "blueprint_v2"
    if phase in {"post_evidence_merge", "after_evidence_merge"}:
        return "blueprint_v1"
    return "blueprint_v1"


def _suggestion_action(warning: str, *, writer_started: bool) -> str:
    if writer_started:
        return "next_run_blueprint_suggestion"
    if warning == "chapter_starved":
        return "repair_before_writing"
    if warning == "chapter_overloaded":
        return "consider_split_or_reorder"
    if warning == "chapter_misaligned":
        return "consider_rename_or_repair"
    return "review_chapter_alignment"


def _alignment_suggestions(alignment: Dict[str, Any], *, writer_started: bool) -> List[Dict[str, Any]]:
    suggestions: List[Dict[str, Any]] = []
    for chapter_id, payload in _as_dict(alignment.get("chapters")).items():
        item = _as_dict(payload)
        warnings = _as_list(item.get("warnings"))
        for warning in warnings:
            suggestions.append(
                {
                    "chapter_id": str(item.get("chapter_id") or chapter_id).strip(),
                    "warning": str(warning or "").strip(),
                    "action": _suggestion_action(str(warning or "").strip(), writer_started=writer_started),
                    "reason": str(warning or "").strip(),
                    "claimable_fact_count": int(item.get("claimable_fact_count") or 0),
                    "diagnostic_only": True,
                    "must_not_render": True,
                    "public_text_allowed": False,
                }
            )
    return suggestions


def _with_version_metadata(blueprint: Dict[str, Any], version: str, phase: str) -> Dict[str, Any]:
    copied = copy.deepcopy(_as_dict(blueprint))
    metadata = dict(_as_dict(copied.get("metadata")))
    metadata["blueprint_version"] = version
    metadata["blueprint_refresh_phase"] = phase
    copied["metadata"] = metadata
    return copied


def build_staged_blueprint_refresh(
    *,
    report_blueprint: Dict[str, Any],
    alignment: Dict[str, Any],
    phase: str,
    writer_started: bool = False,
) -> Dict[str, Any]:
    """Return a versioned blueprint refresh plan without mutating public text."""

    version = _phase_version(str(phase or ""), writer_started=writer_started)
    original = copy.deepcopy(_as_dict(report_blueprint))
    blueprint = original if writer_started else _with_version_metadata(original, version, str(phase or ""))
    suggestions = _alignment_suggestions(_as_dict(alignment), writer_started=writer_started)
    lineage = [
        {
            "old_chapter_id": item.get("chapter_id"),
            "new_chapter_id": item.get("chapter_id"),
            "relation": "same_chapter",
            "reason": item.get("reason"),
            "affected_requirement_ids": [],
            "affected_evidence_ids": [],
        }
        for item in suggestions
    ]
    return {
        "schema_version": "staged_blueprint_refresh_v1",
        "diagnostic_only": True,
        "must_not_render": True,
        "public_text_allowed": False,
        "blueprint_version": version,
        "phase": str(phase or ""),
        "writer_locked": bool(writer_started),
        "blueprint": blueprint,
        "suggestions": suggestions,
        "lineage": lineage,
        "repair_task_seeds": _as_list(_as_dict(alignment).get("repair_task_seeds")) if not writer_started else [],
    }


def _repair_seed_gap(seed: Dict[str, Any]) -> Dict[str, Any]:
    return {
        **dict(seed),
        "gap_id": str(seed.get("gap_id") or "").strip(),
        "gap_type": str(seed.get("gap_type") or "chapter_alignment_gap").strip(),
        "status": str(seed.get("repair_status") or "still_insufficient").strip(),
        "severity": str(seed.get("severity") or "warning").strip(),
        "source": str(seed.get("source") or "blueprint_evidence_alignment").strip(),
        "source_stage": str(seed.get("source_stage") or "blueprint_evidence_alignment").strip(),
        "allowed_for_writing": False,
        "diagnostic_only": True,
        "must_not_render": True,
        "public_text_allowed": False,
    }


def _append_unique_gap(items: List[Any], gap: Dict[str, Any]) -> None:
    gap_id = str(gap.get("gap_id") or "").strip()
    existing = {
        str(_as_dict(item).get("gap_id") or "").strip()
        for item in items
        if isinstance(item, dict)
    }
    if gap_id and gap_id in existing:
        return
    items.append(gap)


def attach_staged_blueprint_refresh(
    *,
    report_blueprint: Dict[str, Any],
    evidence_package: Dict[str, Any],
    phase: str,
    writer_started: bool = False,
) -> Dict[str, Any]:
    """Attach blueprint/evidence alignment diagnostics and repair seeds to an evidence package."""

    package = evidence_package if isinstance(evidence_package, dict) else {}
    alignment = build_blueprint_evidence_alignment(
        report_blueprint=_as_dict(report_blueprint),
        evidence_package=package,
    )
    refresh = build_staged_blueprint_refresh(
        report_blueprint=_as_dict(report_blueprint),
        alignment=alignment,
        phase=phase,
        writer_started=writer_started,
    )
    metadata = package.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
        package["metadata"] = metadata
    metadata["blueprint_evidence_alignment"] = alignment
    metadata["blueprint_refresh"] = refresh
    package["blueprint_evidence_alignment"] = alignment
    package["blueprint_refresh"] = refresh
    if not writer_started:
        gaps = package.setdefault("evidence_gap_ledger", [])
        repairs = package.setdefault("evidence_repair_priorities", [])
        if not isinstance(gaps, list):
            gaps = []
            package["evidence_gap_ledger"] = gaps
        if not isinstance(repairs, list):
            repairs = []
            package["evidence_repair_priorities"] = repairs
        for seed in _as_list(alignment.get("repair_task_seeds")):
            gap = _repair_seed_gap(_as_dict(seed))
            _append_unique_gap(gaps, gap)
            _append_unique_gap(repairs, gap)
    return refresh
