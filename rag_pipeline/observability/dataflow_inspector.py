from __future__ import annotations

from typing import Any, Dict, Iterable, List


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _packet_count(packet: Dict[str, Any], field: str) -> int:
    if field in packet:
        return _safe_int(packet.get(field))
    nested_key = {
        "input_count": "input",
        "output_count": "output",
        "drop_count": "drop",
    }.get(field)
    if nested_key:
        return _safe_int(_as_dict(packet.get(nested_key)).get("count"))
    return 0


def _packet_id_coverage(packet: Dict[str, Any]) -> Dict[str, Any]:
    return (
        _as_dict(packet.get("id_coverage"))
        or _as_dict(_as_dict(packet.get("output")).get("id_coverage"))
        or _as_dict(_as_dict(packet.get("input")).get("id_coverage"))
    )


def _is_funnel_packet(packet: Dict[str, Any]) -> bool:
    event_type = str(packet.get("event_type") or "").strip().lower()
    return event_type in {"", "transform_result"}


def _lineage_gaps_for_packet(packet: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    gaps: Dict[str, Dict[str, Any]] = {}
    id_coverage = _packet_id_coverage(packet)
    output_count = max(0, _packet_count(packet, "output_count"))
    for field, ratio_value in id_coverage.items():
        ratio = max(0.0, min(1.0, _safe_float(ratio_value)))
        missing_count = int(round(output_count * (1 - ratio))) if output_count else 0
        diagnostics = _as_dict(packet.get("diagnostics"))
        coverage_details = _as_dict(diagnostics.get("id_coverage_details"))
        detail = _as_dict(coverage_details.get(field))
        if "missing_count" in detail:
            missing_count = _safe_int(detail.get("missing_count"))
        if missing_count > 0:
            gaps[str(field)] = {
                "coverage": ratio,
                "missing_count": missing_count,
                "output_count": output_count,
            }
    return gaps


def _bottlenecks(packets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for packet in packets:
        input_count = _packet_count(packet, "input_count")
        output_count = _packet_count(packet, "output_count")
        drop_count = _packet_count(packet, "drop_count")
        if input_count <= 0:
            continue
        drop_ratio = drop_count / input_count if input_count else 0
        if drop_count > 0 or output_count == 0:
            candidates.append(
                {
                    "stage": str(packet.get("stage") or "unknown"),
                    "input_count": input_count,
                    "output_count": output_count,
                    "drop_count": drop_count,
                    "drop_ratio": round(drop_ratio, 4),
                    "status": str(packet.get("status") or "unknown"),
                }
            )
    return sorted(candidates, key=lambda item: (item["drop_ratio"], item["drop_count"]), reverse=True)[:8]


def _recommendations(report: Dict[str, Any]) -> List[str]:
    recommendations: List[str] = []
    funnel = _as_dict(report.get("funnel"))
    if _safe_int(_as_dict(funnel.get("evidence_merge")).get("drop_count")) > 0:
        recommendations.append("Inspect evidence normalization and cache contamination before tightening downstream gates.")
    if _safe_int(_as_dict(funnel.get("claim_builder")).get("drop_count")) > 0:
        recommendations.append("Inspect claim lineage fields before changing claim quality thresholds.")
    if _safe_int(_as_dict(funnel.get("section_builder")).get("drop_count")) > 0:
        recommendations.append("Inspect claim-to-section binding and used_fact_refs transfer.")
    lineage_gaps = _as_dict(report.get("lineage_gaps"))
    if lineage_gaps:
        recommendations.append("Fix missing lineage IDs at the first stage that reports them, not at final audit.")
    if not recommendations:
        recommendations.append("No major data-transfer bottleneck detected by stage probe.")
    return recommendations[:8]


def build_dataflow_report(packets: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a compact, diagnostic-only report from stage probe packets."""

    packet_list = [packet for packet in packets if isinstance(packet, dict) and _is_funnel_packet(packet)]
    run_id = str(packet_list[0].get("run_id") or "") if packet_list else ""
    funnel: Dict[str, Dict[str, Any]] = {}
    lineage_gaps: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for packet in packet_list:
        stage = str(packet.get("stage") or "unknown")
        funnel[stage] = {
            "input_count": _packet_count(packet, "input_count"),
            "output_count": _packet_count(packet, "output_count"),
            "drop_count": _packet_count(packet, "drop_count"),
            "status": str(packet.get("status") or "unknown"),
        }
        stage_gaps = _lineage_gaps_for_packet(packet)
        if stage_gaps:
            lineage_gaps[stage] = stage_gaps

    report: Dict[str, Any] = {
        "schema_version": "dataflow_report_v1",
        "run_id": run_id,
        "stage_count": len(packet_list),
        "funnel": funnel,
        "lineage_gaps": lineage_gaps,
        "bottlenecks": _bottlenecks(packet_list),
        "diagnostic_only": True,
        "must_not_render": True,
        "public_text_allowed": False,
    }
    report["recommendations"] = _recommendations(report)
    return report


def render_dataflow_summary(report: Dict[str, Any]) -> str:
    """Render a human-readable diagnostic summary that must not enter the report body."""

    payload = _as_dict(report)
    lines = [
        f"# Dataflow Summary: {payload.get('run_id') or 'unknown'}",
        "",
        "> Diagnostic only. Do not render this content in the public report.",
        "",
        "## Stage Funnel",
        "| Stage | Input | Output | Drop | Status |",
        "|---|---:|---:|---:|---|",
    ]
    for stage, item in _as_dict(payload.get("funnel")).items():
        row = _as_dict(item)
        lines.append(
            f"| {stage} | {_safe_int(row.get('input_count'))} | {_safe_int(row.get('output_count'))} | {_safe_int(row.get('drop_count'))} | {row.get('status') or 'unknown'} |"
        )

    lines.extend(["", "## Lineage Gaps"])
    gaps = _as_dict(payload.get("lineage_gaps"))
    if not gaps:
        lines.append("- None recorded.")
    else:
        for stage, fields in gaps.items():
            for field, detail in _as_dict(fields).items():
                item = _as_dict(detail)
                lines.append(
                    f"- {stage}.{field}: missing={_safe_int(item.get('missing_count'))}, coverage={item.get('coverage')}"
                )

    lines.extend(["", "## Bottlenecks"])
    bottlenecks = payload.get("bottlenecks") if isinstance(payload.get("bottlenecks"), list) else []
    if not bottlenecks:
        lines.append("- None recorded.")
    else:
        for item in bottlenecks:
            row = _as_dict(item)
            lines.append(
                f"- {row.get('stage')}: drop={_safe_int(row.get('drop_count'))}/{_safe_int(row.get('input_count'))} ({row.get('drop_ratio')})"
            )

    lines.extend(["", "## Recommendations"])
    recommendations = payload.get("recommendations") if isinstance(payload.get("recommendations"), list) else []
    if not recommendations:
        lines.append("- None.")
    else:
        lines.extend(f"- {item}" for item in recommendations)
    return "\n".join(lines).rstrip() + "\n"
