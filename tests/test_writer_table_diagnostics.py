from rag_pipeline.agents.writer_agent_clean import build_writer_report, _stage_quality_card, _table_quality_summary
from tests.helpers import sample_evidence_package, sample_structured_analysis


def test_table_quality_summary_exposes_render_tiers_and_reject_reasons():
    summary = _table_quality_summary(
        [
            {"table_id": "body", "should_render": True, "table_value_tier": "high"},
            {"table_id": "appendix", "should_render": False, "appendix_only": True, "reject_reasons": ["appendix_only"]},
            {"table_id": "drop", "should_render": False, "reject_reasons": ["body_rows_lt_2", "no_public_table_rows_after_sanitization"]},
        ]
    )

    assert summary["render_tier_distribution"]["body_high_value"] == 1
    assert summary["render_tier_distribution"]["appendix"] == 1
    assert summary["render_tier_distribution"]["drop"] == 1
    assert summary["reject_reason_distribution"]["body_rows_lt_2"] == 1
    assert summary["drop_count"] == 1


def test_stage_quality_card_aggregates_binding_table_and_citation_signals():
    table_summary = _table_quality_summary(
        [
            {"table_id": "body", "should_render": True, "table_value_tier": "high"},
            {"table_id": "drop", "should_render": False, "reject_reasons": ["body_rows_lt_2"]},
        ]
    )

    card = _stage_quality_card(
        chapter_evidence_packages=[
            {
                "chapter_id": "ch_01",
                "evidence_binding_funnel": {
                    "candidate_fact_count": 5,
                    "eligible_fact_count": 4,
                    "filtered_fact_count": 1,
                    "resolved_diagnostic_ref_count": 2,
                    "unresolved_ref_count": 1,
                    "relevance_rejected_count": 1,
                    "matched_after_relevance_count": 3,
                    "hydrated_evidence_count": 3,
                    "layer_counts": {"case_evidence": 2, "metric_evidence": 1},
                },
            },
            {
                "chapter_id": "ch_02",
                "evidence_binding_funnel": {
                    "candidate_fact_count": 2,
                    "eligible_fact_count": 2,
                    "unresolved_ref_count": 0,
                    "relevance_rejected_count": 0,
                    "matched_after_relevance_count": 0,
                    "hydrated_evidence_count": 0,
                    "layer_counts": {},
                },
            },
        ],
        table_quality_summary=table_summary,
        table_gap_summary={"missing_field_distribution": {"period": 1}, "table_follow_up_count": 1},
        analysis_stage_diagnostics={"llm_usable_claim_count": 7, "llm_failed_chapter_count": 1, "final_analysis_source": "llm_partial_merged"},
        final_citation_audit={
            "final_citation_reconciliation_status": "ok",
            "citation_rebind_required": True,
            "citationless_factual_removed_count": 3,
        },
    )

    assert card["schema_version"] == "stage_quality_card_v1"
    assert card["evidence_binding"]["candidate_fact_count"] == 7
    assert card["evidence_binding"]["unresolved_ref_count"] == 1
    assert card["evidence_binding"]["empty_chapter_count"] == 1
    assert card["evidence_binding"]["layer_counts"]["case_evidence"] == 2
    assert card["table"]["drop_count"] == 1
    assert card["table"]["reject_reason_distribution"]["body_rows_lt_2"] == 1
    assert card["analysis"]["usable_claim_count"] == 7
    assert card["citation"]["citation_rebind_required"] is True
    assert "citation_rebind_required" in card["top_blockers"]


def test_writer_report_exposes_stage_quality_card_in_all_debug_surfaces():
    report = build_writer_report(
        query="AI Agent enterprise adoption",
        evidence_package=sample_evidence_package(),
        structured_analysis=sample_structured_analysis(),
    )

    assert report["stage_quality_card"]["schema_version"] == "stage_quality_card_v1"
    assert report["debug_snapshot"]["stage_quality_card"]["schema_version"] == "stage_quality_card_v1"
    assert report["render_artifacts"]["stage_quality_card"]["schema_version"] == "stage_quality_card_v1"
    assert report["metadata"]["stage_quality_card"]["schema_version"] == "stage_quality_card_v1"
