from __future__ import annotations

from rag_pipeline.contracts.metric_asset import build_metric_assets, metric_is_complete
from rag_pipeline.quality.conversion_summary import build_quality_conversion_summary
from rag_pipeline.quality.regression import build_run_quality_snapshot


def test_metric_assets_classify_complete_and_missing_fields():
    assets = build_metric_assets(
        [
            {
                "evidence_id": "EV-rate",
                "proof_role": "metric",
                "metric": "enterprise adoption rate",
                "value": "40%",
                "unit": "",
                "period": "2025",
                "source_id": "SRC-1",
                "source_level": "B",
            },
            {
                "evidence_id": "EV-missing",
                "proof_role": "metric",
                "metric": "market size",
                "value": "100",
                "unit": "亿元",
                "period": "",
                "source_id": "SRC-2",
            },
        ]
    )

    assert len(assets) == 2
    assert metric_is_complete(assets[0]) is True
    assert assets[0]["unit"] == "%"
    assert assets[0]["complete"] is True
    assert assets[0]["table_ready"] is True
    assert assets[1]["complete"] is False
    assert assets[1]["missing_fields"] == ["period"]


def test_quality_conversion_summary_builds_canonical_funnels_and_recommendations():
    evidence_package = {
        "normalized_evidence": [
            {"evidence_id": "EV-1", "source_id": "SRC-1", "requirement_id": "REQ-1"},
            {"evidence_id": "EV-1", "source_id": "SRC-1", "requirement_id": "REQ-1"},
            {"fact": "missing id"},
        ],
        "clean_evidence_list": [
            {"evidence_id": "EV-1", "source_id": "SRC-1", "requirement_id": "REQ-1"},
            {"evidence_id": "EV-2", "source_id": "SRC-2", "requirement_id": "REQ-2"},
        ],
        "analysis_ready_evidence": [
            {
                "evidence_id": "EV-1",
                "aliases": ["EV-legacy"],
                "source_id": "SRC-1",
                "requirement_id": "REQ-1",
                "chapter_id": "ch_01",
                "proof_role": "metric",
                "metric": "enterprise adoption rate",
                "value": "40%",
                "period": "2025",
                "source_level": "B",
            },
            {
                "evidence_id": "EV-2",
                "source_id": "SRC-2",
                "requirement_id": "REQ-2",
                "chapter_id": "ch_02",
                "proof_role": "metric",
                "metric": "market size",
                "value": "100",
                "unit": "亿元",
                "period": "",
            },
        ],
        "core_evidence": [{"evidence_id": "EV-1"}],
        "supporting_evidence": [{"evidence_id": "EV-2"}],
        "clue_evidence": [{"evidence_id": "EV-3"}],
        "evidence_gap_ledger": [
            {"gap_id": "GAP-1", "requirement_id": "REQ-2", "status": "evidence_found"},
            {"gap_id": "GAP-2", "requirement_id": "REQ-1", "status": "claim_bound"},
            {"gap_id": "GAP-3", "requirement_id": "REQ-3", "status": "still_insufficient"},
        ],
    }
    structured_analysis = {
        "claim_units": [
            {
                "claim_id": "CL-1",
                "claim": "Bound and renderable claim.",
                "used_evidence_ids": ["EV-legacy"],
                "claim_strength": "moderate",
            },
            {
                "claim_id": "CL-2",
                "claim": "Diagnostic claim.",
                "fact_ids": ["EV-2"],
                "source_ids": ["SRC-2"],
                "requirement_ids": ["REQ-2"],
                "allowed_use": "repair_needed",
            },
            {"claim_id": "CL-3", "claim": "Unbound claim."},
        ]
    }
    writer_package = {
        "chapter_packages": [
            {
                "chapter_id": "ch_01",
                "sections": [{"section_id": "SEC-1", "claim_id": "CL-1", "used_fact_refs": ["EV-legacy"]}],
            }
        ]
    }
    writer_report = {
        "final_citation_audit": {
            "citationless_factual_removed_count": 2,
            "final_unresolved_citation_removed_count": 1,
            "citation_rebind_required": True,
        }
    }

    summary = build_quality_conversion_summary(
        evidence_package=evidence_package,
        structured_analysis=structured_analysis,
        writer_report=writer_report,
        writer_package=writer_package,
    )

    assert summary["schema_version"] == "quality_conversion_summary_v1"
    assert summary["evidence_funnel"]["normalized_count"] == 2
    assert summary["evidence_funnel"]["normalized_missing_canonical_id_count"] == 1
    assert summary["evidence_funnel"]["analysis_ready_count"] == 2
    assert summary["claim_funnel"]["total_claim_count"] == 3
    assert summary["claim_funnel"]["bound_claim_count"] == 2
    assert summary["claim_funnel"]["renderable_claim_count"] == 1
    assert summary["claim_funnel"]["section_bound_claim_count"] == 1
    assert summary["claim_funnel"]["diagnostic_only_claim_count"] == 1
    assert summary["metric_funnel"]["metric_candidate_count"] == 2
    assert summary["metric_funnel"]["complete_metric_count"] == 1
    assert summary["metric_funnel"]["metric_missing_field_counts"]["period"] == 1
    assert summary["citation_funnel"]["final_deleted_fact_count"] == 3
    assert summary["citation_funnel"]["citation_rebind_required"] is True
    assert summary["repair_funnel"]["strict_closed_gap_count"] == 1
    assert summary["repair_funnel"]["signal_only_gap_count"] == 1
    assert summary["by_requirement"]["REQ-1"]["bound_claim_count"] == 1
    assert summary["by_chapter"]["ch_01"]["analysis_ready_count"] == 1
    actions = {item["action"] for item in summary["recommendations"]}
    assert "metric_repair_search" in actions
    assert "citation_rebind_or_section_rewrite" in actions


def test_quality_snapshot_exposes_conversion_summary():
    package = {
        "metadata": {"run_id": "run-conv", "topic_id": "topic"},
        "evidence_package": {
            "clean_evidence_list": [{"evidence_id": "EV-1", "source_id": "SRC-1", "requirement_id": "REQ-1"}],
            "analysis_ready_evidence": [{"evidence_id": "EV-1", "source_id": "SRC-1", "requirement_id": "REQ-1"}],
        },
        "structured_analysis": {
            "claim_units": [
                {"claim_id": "CL-1", "fact_ids": ["EV-1"], "source_ids": ["SRC-1"], "requirement_ids": ["REQ-1"]}
            ]
        },
        "writer_report": {"report_status": "final_clean", "quality_score": 80},
    }

    snapshot = build_run_quality_snapshot(package)

    assert snapshot["quality_conversion_summary"]["schema_version"] == "quality_conversion_summary_v1"
    assert snapshot["quality_conversion_summary"]["claim_funnel"]["bound_claim_count"] == 1
