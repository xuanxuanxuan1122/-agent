from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from rag_pipeline.observability.health_metrics import build_health_metrics
from rag_pipeline.observability.lineage_tracker import build_lineage_graph
from rag_pipeline.observability.module_probe_models import as_dict, make_module_probe_event, safe_int
from rag_pipeline.observability.stage_probe import build_stage_probe_packets

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def module_probe_enabled() -> bool:
    return _env_flag("MODULE_PROBE_ENABLED", True)


def _module_name(stage: str) -> str:
    return {
        "search_plan": "research_planner",
        "query_rewrite": "web_analysis_agent.query_rewrite",
        "web_result_filter": "web_result_filter",
        "readpage": "iqs_readpage",
        "fact_extractor": "readpage_fact_extractor_agent",
        "evidence_merge": "evidence_merger",
        "llm_analysis": "analysis_agent",
        "claim_builder": "claim_builder_agent",
        "section_builder": "chapter_argument_agent",
        "body_rewrite": "section_body_rewrite_agent",
        "chapter_narrative": "chapter_narrative_agent",
        "public_narrative_gate": "public_report_sanitizer",
        "table_builder": "table_agent",
        "qa_score": "writer_qa_agent",
        "final_citation_audit": "final_citation_reconciler",
        "source_gate": "source_claim_support_gate",
        "reformatter": "reformatter_agent",
        "artifact_ledger": "artifact_ledger",
        "stage_snapshot": "stage_snapshot_cache",
        "score_gap_ledger": "score_gap_ledger",
        "citation_manifest": "citation_manifest_builder",
        "final_audit": "final_audit_agent",
        "writer": "writer_agent_clean",
        "cache_summary": "cache_layers",
    }.get(stage, stage or "unknown")


def _decision_needed(packet: Dict[str, Any]) -> bool:
    status = str(packet.get("status") or "").strip().lower()
    positive_reason = any(safe_int(value) > 0 for value in as_dict(packet.get("reason_counts")).values())
    return status in {"warning", "error", "degraded"} or safe_int(packet.get("drop_count")) > 0 or positive_reason


def build_module_probe_events(
    *,
    run_id: str,
    writer_package: Dict[str, Any],
    writer_report: Optional[Dict[str, Any]] = None,
    stage_packets: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Expand stage packets into fine-grained diagnostic module events."""

    packets = stage_packets or build_stage_probe_packets(run_id=run_id, writer_package=writer_package, writer_report=writer_report)
    events: List[Dict[str, Any]] = []
    seq = 0
    for packet in packets:
        stage = str(packet.get("stage") or "unknown")
        module = _module_name(stage)
        status = str(packet.get("status") or "ok")
        input_count = safe_int(packet.get("input_count"))
        output_count = safe_int(packet.get("output_count"))
        drop_count = safe_int(packet.get("drop_count"))
        reason_counts = as_dict(packet.get("reason_counts"))
        id_coverage = as_dict(packet.get("id_coverage"))
        cache = as_dict(packet.get("cache"))
        diagnostics = as_dict(packet.get("diagnostics"))
        seq += 1
        events.append(
            make_module_probe_event(
                run_id=run_id,
                seq=seq,
                stage=stage,
                module=module,
                event_type="input_received",
                status=status,
                input_count=input_count,
                output_count=0,
                drop_count=0,
                id_coverage=id_coverage,
                cache=cache,
                diagnostics={"lineage": as_dict(packet.get("lineage"))},
            )
        )
        seq += 1
        events.append(
            make_module_probe_event(
                run_id=run_id,
                seq=seq,
                stage=stage,
                module=module,
                event_type="transform_result",
                status=status,
                input_count=input_count,
                output_count=output_count,
                drop_count=drop_count,
                reason_counts=reason_counts,
                id_coverage=id_coverage,
                cache=cache,
                diagnostics=diagnostics,
            )
        )
        if _decision_needed(packet):
            seq += 1
            events.append(
                make_module_probe_event(
                    run_id=run_id,
                    seq=seq,
                    stage=stage,
                    module=module,
                    event_type="decision_observed",
                    status=status,
                    input_count=input_count,
                    output_count=output_count,
                    drop_count=drop_count,
                    reason_counts=reason_counts,
                    id_coverage=id_coverage,
                    cache=cache,
                    diagnostics={
                        "suggested_action": "inspect_stage_before_changing_business_logic",
                        "observation_only": True,
                        "stage_diagnostics": diagnostics,
                    },
                )
            )
    return events


def write_module_probe_from_package(
    *,
    run_id: str,
    output_dir: Path | str,
    writer_package: Dict[str, Any],
    writer_report: Optional[Dict[str, Any]] = None,
    base_name: Optional[str] = None,
    stage_packets: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Write module_probe, lineage_graph, and health_metrics sidecars."""

    if not module_probe_enabled():
        return {
            "enabled": False,
            "module_probe_path": "",
            "lineage_graph_path": "",
            "health_metrics_path": "",
            "event_count": 0,
        }
    safe_run_id = str(run_id or "run").strip() or "run"
    safe_base_name = str(base_name or safe_run_id).strip() or safe_run_id
    root = Path(output_dir)
    module_path = root / f"{safe_base_name}.module_probe.jsonl"
    lineage_path = root / f"{safe_base_name}.lineage_graph.json"
    health_path = root / f"{safe_base_name}.health_metrics.json"
    try:
        root.mkdir(parents=True, exist_ok=True)
        packets = stage_packets or build_stage_probe_packets(run_id=safe_run_id, writer_package=writer_package, writer_report=writer_report)
        events = build_module_probe_events(
            run_id=safe_run_id,
            writer_package=writer_package,
            writer_report=writer_report,
            stage_packets=packets,
        )
        lineage_graph = build_lineage_graph(run_id=safe_run_id, writer_package=writer_package, writer_report=writer_report)
        health_metrics = build_health_metrics(
            run_id=safe_run_id,
            writer_package=writer_package,
            writer_report=writer_report,
            stage_packets=packets,
        )
        with module_path.open("w", encoding="utf-8") as handle:
            for event in events:
                handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        lineage_path.write_text(json.dumps(lineage_graph, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        health_path.write_text(json.dumps(health_metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {
            "enabled": True,
            "run_id": safe_run_id,
            "module_probe_path": str(module_path),
            "lineage_graph_path": str(lineage_path),
            "health_metrics_path": str(health_path),
            "event_count": len(events),
            "lineage_node_count": len(lineage_graph.get("nodes") or []),
            "lineage_edge_count": len(lineage_graph.get("edges") or []),
        }
    except Exception as exc:  # pragma: no cover - observability must not block report generation.
        logger.warning("Module probe write failed", extra={"run_id": safe_run_id, "error": str(exc)})
        return {
            "enabled": False,
            "run_id": safe_run_id,
            "module_probe_path": str(module_path),
            "lineage_graph_path": str(lineage_path),
            "health_metrics_path": str(health_path),
            "event_count": 0,
            "error": str(exc),
        }
