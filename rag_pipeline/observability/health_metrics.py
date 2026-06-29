from __future__ import annotations

from typing import Any, Dict, List, Optional

from rag_pipeline.observability.module_probe_models import as_dict, safe_int, safe_rate
from rag_pipeline.observability.stage_probe import build_stage_probe_packets

SCHEMA_VERSION = "health_metrics_v1"


def _by_stage(packets: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(packet.get("stage") or ""): packet for packet in packets if isinstance(packet, dict)}


def _rate(packet: Dict[str, Any]) -> float:
    return safe_rate(packet.get("output_count"), packet.get("input_count"))


def _count(packet: Dict[str, Any], field: str) -> int:
    return safe_int(packet.get(field))


def _artifact_fact_card_count(writer_package: Dict[str, Any]) -> int:
    artifact_ledger = as_dict(writer_package.get("artifact_ledger"))
    candidates = [
        artifact_ledger.get("fact_card_count"),
        as_dict(artifact_ledger.get("diagnostics")).get("fact_card_count"),
        as_dict(writer_package.get("artifact_diagnostics")).get("fact_card_count"),
        as_dict(writer_package.get("raw_output")).get("fact_card_count"),
    ]
    for candidate in candidates:
        count = safe_int(candidate)
        if count > 0:
            return count
    return 0


def _top_reason_pairs(reason_counts: Dict[str, Any]) -> List[List[Any]]:
    pairs: List[List[Any]] = []
    for key, value in reason_counts.items():
        count = safe_int(value)
        if key and count > 0:
            pairs.append([str(key), count])
    return sorted(pairs, key=lambda item: item[1], reverse=True)[:10]


def build_health_metrics(
    *,
    run_id: str,
    writer_package: Dict[str, Any],
    writer_report: Optional[Dict[str, Any]] = None,
    stage_packets: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Compute core conversion rates for report pipeline diagnosis."""

    packets = stage_packets or build_stage_probe_packets(run_id=run_id, writer_package=writer_package, writer_report=writer_report)
    by_stage = _by_stage(packets)
    search_filter = by_stage.get("web_result_filter", {})
    readpage = by_stage.get("readpage", {})
    fact_extractor = by_stage.get("fact_extractor", {})
    evidence_merge = by_stage.get("evidence_merge", {})
    llm_analysis = by_stage.get("llm_analysis", {})
    claim_builder = by_stage.get("claim_builder", {})
    section_builder = by_stage.get("section_builder", {})
    citation = by_stage.get("final_citation_audit") or by_stage.get("citation_manifest", {})
    repair = by_stage.get("score_gap_ledger") or by_stage.get("evidence_repair", {})
    cache_summary = by_stage.get("cache_summary", {})
    writer = by_stage.get("writer", {})
    writer_diag = as_dict(writer.get("diagnostics"))
    evidence_merge_reasons = as_dict(evidence_merge.get("reason_counts"))
    if not evidence_merge_reasons:
        evidence_merge_reasons = as_dict(as_dict(as_dict(evidence_merge.get("diagnostics")).get("evidence_root_hygiene")).get("reason_counts"))
    fact_card_count = _count(fact_extractor, "output_count") or _artifact_fact_card_count(writer_package)
    llm_diag = as_dict(llm_analysis.get("diagnostics"))
    llm_raw_claim_count = safe_int(llm_diag.get("llm_raw_claim_count")) or _count(claim_builder, "input_count") or _count(llm_analysis, "output_count")
    llm_usable_claim_count = safe_int(llm_diag.get("llm_usable_claim_count")) or _count(llm_analysis, "output_count")
    fact_to_claim_rate = safe_rate(llm_usable_claim_count or _count(llm_analysis, "output_count"), _count(evidence_merge, "output_count") or _count(llm_analysis, "input_count"))
    llm_claim_acceptance_rate = safe_rate(llm_usable_claim_count, llm_raw_claim_count)
    claim_binding_rate = _rate(claim_builder)

    rates = {
        "search_result_accept_rate": _rate(search_filter),
        "readpage_success_rate": _rate(readpage),
        "fact_extraction_rate": _rate(fact_extractor),
        "analysis_ready_rate": _rate(evidence_merge),
        "claim_conversion_rate": fact_to_claim_rate,
        "fact_to_claim_rate": fact_to_claim_rate,
        "llm_claim_acceptance_rate": llm_claim_acceptance_rate,
        "claim_binding_rate": claim_binding_rate,
        "bound_claim_rate": claim_binding_rate,
        "section_binding_rate": _rate(section_builder),
        "citation_binding_rate": _rate(citation),
        "repair_success_rate": _rate(repair),
        "cache_hit_rate": _rate(cache_summary),
    }
    counts = {
        "raw_search_result_count": _count(search_filter, "input_count"),
        "accepted_search_result_count": _count(search_filter, "output_count"),
        "readpage_attempted_count": _count(readpage, "input_count"),
        "readpage_success_count": _count(readpage, "output_count"),
        "fact_card_count": fact_card_count,
        "analysis_ready_fact_count": _count(evidence_merge, "output_count"),
        "llm_raw_claim_count": llm_raw_claim_count,
        "llm_usable_claim_count": llm_usable_claim_count,
        "raw_claim_count": _count(claim_builder, "input_count"),
        "bound_claim_count": _count(claim_builder, "output_count"),
        "section_count": _count(section_builder, "input_count"),
        "evidence_backed_section_count": _count(section_builder, "output_count"),
        "final_body_char_count": safe_int(writer_diag.get("body_char_count") or writer_diag.get("final_body_char_count")),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": str(run_id or "run").strip() or "run",
        "rates": rates,
        "counts": counts,
        "dirty_evidence_top_reasons": _top_reason_pairs(evidence_merge_reasons),
        "stage_count": len(packets),
        "diagnostic_only": True,
        "must_not_render": True,
        "public_text_allowed": False,
    }
