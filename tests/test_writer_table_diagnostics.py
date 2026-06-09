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
        handoff_contract_summary={
            "ok": False,
            "failed_contracts": ["analysis_to_writer"],
            "results": {
                "analysis_to_writer": {
                    "ok": False,
                    "errors": ["claim_missing_fact_or_evidence_refs"],
                    "summary": {"claim_count": 2, "missing_fact_or_evidence_refs_count": 1},
                }
            },
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
    assert card["handoff"]["ok"] is False
    assert card["handoff"]["failed_contracts"] == ["analysis_to_writer"]
    assert card["handoff"]["results"]["analysis_to_writer"]["summary"]["missing_fact_or_evidence_refs_count"] == 1
    assert "handoff_contract_failed" in card["top_blockers"]
    assert "citation_rebind_required" in card["top_blockers"]


def test_writer_report_exposes_stage_quality_card_in_all_debug_surfaces():
    report = build_writer_report(
        query="AI Agent enterprise adoption",
        evidence_package=sample_evidence_package(),
        structured_analysis=sample_structured_analysis(),
    )

    assert report["stage_quality_card"]["schema_version"] == "stage_quality_card_v1"
    assert report["stage_quality_card"]["handoff"]["schema_version"] == "handoff_contract_summary_v1"
    assert report["handoff_contract_summary"]["schema_version"] == "handoff_contract_summary_v1"
    assert report["debug_snapshot"]["stage_quality_card"]["schema_version"] == "stage_quality_card_v1"
    assert report["render_artifacts"]["stage_quality_card"]["schema_version"] == "stage_quality_card_v1"
    assert report["metadata"]["stage_quality_card"]["schema_version"] == "stage_quality_card_v1"
    assert report["metadata"]["handoff_contract_summary"]["schema_version"] == "handoff_contract_summary_v1"


def test_writer_report_normalizes_argument_unit_refs_to_canonical_fields():
    report = build_writer_report(
        query="AI Agent adoption",
        evidence_package={
            "analysis_ready_evidence": [
                {
                    "evidence_id": "EV-04-22",
                    "aliases": ["EV-04-L22"],
                    "source_id": "SRC-1",
                    "requirement_id": "REQ-1",
                    "data_point": "Enterprise deployments are moving into workflow automation.",
                    "source": {"title": "Official report", "url": "https://example.com/report"},
                }
            ],
            "source_registry": [{"source_id": "SRC-1", "url": "https://example.com/report"}],
        },
        structured_analysis={
            "claim_units": [
                {
                    "claim_id": "CL-1",
                    "chapter_id": "ch_01",
                    "claim": "Enterprise deployments are moving into workflow automation.",
                    "evidence_refs": ["EV-04-L22"],
                    "requirement_ids": ["REQ-1"],
                }
            ]
        },
        argument_units=[
            {
                "claim_id": "CL-1",
                "chapter_id": "ch_01",
                "claim": "Enterprise deployments are moving into workflow automation.",
                "evidence_refs": ["EV-04-L22"],
                "public_render": True,
            }
        ],
        chapter_packages=[
            {
                "chapter_id": "ch_01",
                "chapter_title": "Adoption",
                "sections": [
                    {
                        "section_id": "s1",
                        "claim_id": "CL-1",
                        "claim": "Enterprise deployments are moving into workflow automation.",
                        "evidence_refs": ["EV-04-L22"],
                    }
                ],
            }
        ],
        source_registry=[{"source_id": "SRC-1", "ref": "[1]", "url": "https://example.com/report"}],
    )

    unit = report["render_artifacts"]["argument_units"][0]
    assert unit["fact_ids"] == ["EV-04-22"]
    assert unit["evidence_refs"] == ["EV-04-22"]
    assert unit["source_ids"] == ["SRC-1"]
    assert unit["requirement_ids"] == ["REQ-1"]
    assert unit["ref_resolution"]["alias_resolved_ref_count"] == 1
    section = report["render_artifacts"]["chapter_packages"][0]["sections"][0]
    assert section["fact_ids"] == ["EV-04-22"]
    assert section["evidence_refs"] == ["EV-04-22"]
