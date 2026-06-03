from __future__ import annotations

from typing import Any, Dict, List


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _int(value: Any) -> int:
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / max(1, denominator), 3)


def _metric(value: Any, status: str, reason: str = "") -> Dict[str, Any]:
    return {"value": value, "status": status, "reason": reason}


def _worst(statuses: list[str]) -> str:
    if "red" in statuses:
        return "red"
    if "yellow" in statuses:
        return "yellow"
    return "green"


def build_report_health_card(payload: Dict[str, Any]) -> Dict[str, Any]:
    layout = _as_dict(payload.get("layout"))
    chapter_evidence = _as_dict(payload.get("chapter_evidence"))
    analysis = _as_dict(payload.get("analysis"))
    summary = _as_dict(payload.get("summary"))
    valid_fact_cards = _int(
        chapter_evidence.get("total_valid_fact_card_count")
        or chapter_evidence.get("valid_fact_card_count")
        or payload.get("valid_fact_card_count")
    )
    rendered = _int(layout.get("rendered_block_count") or layout.get("layout_block_rendered_count"))
    dropped = _int(layout.get("dropped_block_count") or layout.get("layout_block_dropped"))
    must_render = _int(layout.get("must_render_block_count"))
    rendered_must = _int(layout.get("rendered_must_block_count"))
    candidate_blocks = _int(layout.get("candidate_block_count"))
    evidence_backed = _int(layout.get("evidence_backed_block_count") or layout.get("layout_block_evidence_backed_count"))
    snippet_count = _int(layout.get("snippet_like_text_count") or payload.get("snippet_like_text_count"))
    snippet_dropped_count = _int(layout.get("snippet_like_text_dropped_count"))
    repeated_fact_count = _int(layout.get("repeated_fact_demoted_count") or layout.get("repeated_evidence_id_within_chapter_count"))
    ocr_artifacts = _int(layout.get("ocr_artifact_normalized_count"))
    omitted_chapters = _int(layout.get("chapter_omitted_no_evidence_count"))
    core_omitted_chapters = (
        _int(layout.get("core_chapter_omitted_no_evidence_count"))
        if "core_chapter_omitted_no_evidence_count" in layout
        else omitted_chapters
    )
    optional_omitted_chapters = _int(layout.get("optional_chapter_omitted_count"))
    composer_variable_explanations = _int(layout.get("composer_variable_explanation_count"))
    body_rewrite = _as_dict(payload.get("body_rewrite") or layout.get("body_rewrite"))
    chapter_narrative = _as_dict(payload.get("chapter_narrative") or layout.get("chapter_narrative"))
    body_status = str(payload.get("body_composition_status") or layout.get("body_composition_status") or "unknown")
    citation_manifest = _as_dict(payload.get("citation_manifest"))
    final_citation = _as_dict(payload.get("final_citation_audit"))
    ref_lineage_diagnostics = _as_dict(payload.get("ref_lineage_diagnostics"))
    manifest_status = str(citation_manifest.get("citation_manifest_status") or payload.get("source_appendix_status") or "unknown")
    final_status = str(final_citation.get("final_citation_reconciliation_status") or "")
    source_status = final_status or manifest_status
    quality_posture = _as_dict(payload.get("quality_posture"))
    quality_mode = str(payload.get("quality_mode") or quality_posture.get("mode") or payload.get("execution_mode") or "").strip().lower()
    high_quality_mode = quality_mode in {"true", "1", "yes", "high", "strict", "due_diligence", "live_quality_full", "quality_llm_replay"}
    body_chars = _int(payload.get("body_char_count") or payload.get("body_chars"))
    target_body_chars = _int(payload.get("target_body_chars")) or 20000
    min_body_chars = int(target_body_chars * 0.9) if target_body_chars else 0
    h3_count = _int(payload.get("h3_count"))
    llm_usable_claim_count = _int(analysis.get("llm_usable_claim_count") or payload.get("llm_usable_claim_count"))
    llm_usable_chapter_count = _int(analysis.get("llm_usable_chapter_count") or payload.get("llm_usable_chapter_count"))
    summary_count = _int(summary.get("executive_summary_valid_judgment_count") or payload.get("summary_valid_judgment_count"))
    rendered_ratio = _ratio(rendered_must, must_render) if must_render else _ratio(rendered, rendered + dropped)
    evidence_ratio = _ratio(evidence_backed, rendered)
    metrics: Dict[str, Dict[str, Any]] = {}
    metrics["body_composition_status"] = _metric(
        body_status,
        "red" if body_status == "fact_passthrough" else ("yellow" if body_status in {"unknown", "dropped"} else "green"),
    )
    rewrite_enabled = bool(body_rewrite.get("enabled"))
    rewrite_success = _int(body_rewrite.get("success_count")) + _int(body_rewrite.get("cache_hit_count"))
    rewrite_failures = _int(body_rewrite.get("fallback_count")) + _int(body_rewrite.get("rejected_count"))
    rewrite_polluted = _int(body_rewrite.get("polluted_count"))
    rewrite_submitted = _int(body_rewrite.get("submitted_count") or body_rewrite.get("called_count"))
    if rewrite_polluted:
        rewrite_status = "red"
    elif rewrite_enabled and rewrite_submitted == 0 and evidence_backed == 0:
        rewrite_status = "yellow"
    elif rewrite_enabled and rewrite_failures:
        rewrite_status = "yellow"
    else:
        rewrite_status = "green"
    metrics["body_rewrite_status"] = _metric(
        "not_requested"
        if not rewrite_enabled
        else "no_rewritable_sections"
        if rewrite_submitted == 0 and evidence_backed == 0
        else ("rewritten" if rewrite_success else "fallback_available"),
        rewrite_status,
    )
    narrative_enabled = bool(chapter_narrative.get("enabled"))
    narrative_success = _int(chapter_narrative.get("success_count")) + _int(chapter_narrative.get("cache_hit_count"))
    narrative_failures = _int(chapter_narrative.get("fallback_count")) + _int(chapter_narrative.get("rejected_count"))
    narrative_attempted = _int(chapter_narrative.get("attempted_count"))
    narrative_status_value = (
        "not_requested"
        if not narrative_enabled
        else str(chapter_narrative.get("skipped_reason") or "skipped")
        if narrative_attempted == 0 and not narrative_success
        else "rewritten"
        if narrative_success
        else "fallback_available"
    )
    narrative_status = (
        "green"
        if not narrative_enabled or (narrative_success and not narrative_failures)
        else "yellow"
        if narrative_enabled and (narrative_success or narrative_failures or chapter_narrative.get("skipped_reason"))
        else "red"
    )
    metrics["chapter_narrative_status"] = _metric(narrative_status_value, narrative_status)
    metrics["valid_fact_card_count"] = _metric(
        valid_fact_cards,
        "red" if valid_fact_cards == 0 and evidence_backed == 0 else ("yellow" if valid_fact_cards == 0 else "green"),
    )
    metrics["evidence_backed_section_ratio"] = _metric(
        evidence_ratio,
        "red" if evidence_ratio < 0.3 else ("yellow" if evidence_ratio < 0.55 else "green"),
    )
    metrics["planned_vs_rendered_section_ratio"] = _metric(
        rendered_ratio,
        "red" if rendered_ratio < 0.4 else ("yellow" if rendered_ratio < 0.65 else "green"),
    )
    metrics["citation_manifest_status"] = _metric(
        manifest_status,
        "green"
        if manifest_status == "ok"
        else ("yellow" if manifest_status in {"warning", "unknown", "not_requested"} else "red"),
    )
    final_citation_status = final_status or source_status
    final_missing_refs = _as_list(final_citation.get("final_missing_appendix_refs")) if isinstance(final_citation.get("final_missing_appendix_refs"), list) else []
    final_removed_refs = _int(final_citation.get("final_unresolved_citation_removed_count"))
    factual_body_without_citations = _int(final_citation.get("factual_body_without_citations_count"))
    metrics["final_citation_status"] = _metric(
        final_citation_status,
        "red"
        if final_missing_refs or factual_body_without_citations
        else ("yellow" if final_removed_refs else ("green" if final_citation_status == "ok" else "red")),
    )
    quality_degraded_reasons: List[str] = []
    if high_quality_mode:
        if llm_usable_claim_count < 10:
            quality_degraded_reasons.append("llm_usable_claim_count_below_minimum")
        if llm_usable_chapter_count < 4:
            quality_degraded_reasons.append("llm_usable_chapter_count_below_minimum")
        if evidence_backed < 6:
            quality_degraded_reasons.append("evidence_backed_section_count_below_minimum")
        if min_body_chars and body_chars < min_body_chars:
            quality_degraded_reasons.append("body_chars_below_minimum")
        if h3_count and h3_count < 12:
            quality_degraded_reasons.append("h3_count_below_minimum")
        if factual_body_without_citations:
            quality_degraded_reasons.append("factual_body_without_citations")
        if final_citation_status != "ok":
            quality_degraded_reasons.append("final_citation_status_not_ok")
    metrics["high_quality_density_status"] = _metric(
        "degraded" if quality_degraded_reasons else ("not_requested" if not high_quality_mode else "ok"),
        "yellow" if quality_degraded_reasons else "green",
    )
    metrics["missing_source_ref_count"] = _metric(
        _int(citation_manifest.get("missing_source_ref_count") or payload.get("missing_source_ref_count")),
        "red" if _int(citation_manifest.get("missing_source_ref_count") or payload.get("missing_source_ref_count")) else "green",
    )
    metrics["orphan_citation_count"] = _metric(
        _int(citation_manifest.get("orphan_citation_count") or payload.get("orphan_citation_count")),
        "red" if _int(citation_manifest.get("orphan_citation_count") or payload.get("orphan_citation_count")) else "green",
    )
    metrics["excluded_source_count"] = _metric(
        _int(citation_manifest.get("excluded_source_count") or payload.get("excluded_source_count")),
        "red" if _int(citation_manifest.get("excluded_source_count") or payload.get("excluded_source_count")) else "green",
    )
    metrics["snippet_like_text_count"] = _metric(snippet_count, "red" if snippet_count > 0 else ("yellow" if snippet_dropped_count > 0 else "green"))
    metrics["repeated_fact_count"] = _metric(repeated_fact_count, "yellow" if repeated_fact_count > 0 else "green")
    metrics["ocr_artifact_normalized_count"] = _metric(ocr_artifacts, "yellow" if ocr_artifacts else "green")
    metrics["claim_to_evidence_binding_status"] = _metric(
        str(payload.get("claim_to_evidence_binding_status") or ("ok" if evidence_backed > 0 else "weak")),
        "red" if rendered and evidence_backed == 0 else ("yellow" if evidence_backed < rendered else "green"),
    )
    metrics["chapter_omitted_no_evidence_count"] = _metric(
        core_omitted_chapters,
        "yellow" if core_omitted_chapters else "green",
    )
    metrics["composer_variable_explanation_count"] = _metric(
        composer_variable_explanations,
        "green" if composer_variable_explanations >= evidence_backed or evidence_backed == 0 else "yellow",
    )
    metrics["summary_valid_judgment_count"] = _metric(summary_count, "yellow" if summary_count == 0 else "green")
    metrics["source_appendix_status"] = _metric(
        source_status,
        "green" if source_status == "ok" else ("yellow" if source_status in {"unknown", "not_requested"} else "red"),
    )
    metrics["final_analysis_source"] = _metric(str(analysis.get("final_analysis_source") or payload.get("final_analysis_source") or "unknown"), "green")
    overall_statuses = [
        item["status"]
        for name, item in metrics.items()
        if name != "body_rewrite_status" or item.get("status") == "red"
    ]
    overall = _worst(overall_statuses)
    if quality_degraded_reasons and overall == "green":
        overall = "yellow"
    return {
        "overall_status": overall,
        "metrics": metrics,
        "body_composition_status": body_status,
        "body_rewrite_status": metrics["body_rewrite_status"]["status"],
        "body_rewrite": body_rewrite,
        "chapter_narrative_status": metrics["chapter_narrative_status"]["status"],
        "chapter_narrative": chapter_narrative,
        "quality_mode": quality_mode,
        "quality_path_degraded": bool(quality_degraded_reasons) or bool(payload.get("quality_path_degraded")),
        "quality_degraded_reasons": quality_degraded_reasons,
        "body_char_count": body_chars,
        "target_body_chars": target_body_chars,
        "body_char_gap": max(0, target_body_chars - body_chars) if target_body_chars else 0,
        "h3_count": h3_count,
        "valid_fact_card_count": valid_fact_cards,
        "evidence_backed_section_ratio": evidence_ratio,
        "planned_vs_rendered_section_ratio": rendered_ratio,
        "must_render_block_count": must_render,
        "candidate_block_count": candidate_blocks,
        "rendered_must_block_count": rendered_must,
        "llm_claim_to_block_match_count": _int(layout.get("llm_claim_to_block_match_count")),
        "llm_claim_unmatched_count": _int(layout.get("llm_claim_unmatched_count")),
        "must_block_matched_by_llm_claim_count": _int(layout.get("must_block_matched_by_llm_claim_count")),
        "must_block_dropped_no_matching_claim_count": _int(layout.get("must_block_dropped_no_matching_claim_count")),
        "snippet_like_text_count": snippet_count,
        "snippet_like_text_dropped_count": snippet_dropped_count,
        "repeated_fact_count": repeated_fact_count,
        "ocr_artifact_normalized_count": ocr_artifacts,
        "chapter_omitted_no_evidence_count": core_omitted_chapters,
        "core_chapter_omitted_no_evidence_count": core_omitted_chapters,
        "optional_chapter_omitted_count": optional_omitted_chapters,
        "block_drop_reason_examples": layout.get("block_drop_reason_examples") or [],
        "composer_variable_explanation_count": composer_variable_explanations,
        "ref_lineage_diagnostics": ref_lineage_diagnostics,
        "summary_valid_judgment_count": summary_count,
        "source_appendix_status": source_status,
        "citation_manifest_status": manifest_status,
        "final_citation_status_after_render": metrics["final_citation_status"]["value"],
        "final_body_citation_refs": final_citation.get("final_body_citation_refs") or [],
        "final_appendix_refs": final_citation.get("final_appendix_refs") or [],
        "final_missing_appendix_refs": final_citation.get("final_missing_appendix_refs") or [],
        "factual_body_without_citations_count": factual_body_without_citations,
        "citationless_fact_examples": final_citation.get("citationless_fact_examples") or [],
        "final_unresolved_citation_removed_count": final_removed_refs,
        "missing_source_ref_count": metrics["missing_source_ref_count"]["value"],
        "orphan_citation_count": metrics["orphan_citation_count"]["value"],
        "excluded_source_count": metrics["excluded_source_count"]["value"],
        "final_analysis_source": metrics["final_analysis_source"]["value"],
    }
