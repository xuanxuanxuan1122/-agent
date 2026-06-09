from rag_pipeline.contracts.handoff_contracts import (
    validate_citation_reconciliation,
    validate_evidence_package_for_analysis,
    validate_repair_priorities_for_dispatch,
    validate_structured_analysis_for_writer,
    validate_writer_report_for_final,
)


def test_evidence_package_contract_reports_missing_ids_and_sources():
    result = validate_evidence_package_for_analysis(
        {
            "analysis_ready_evidence": [
                {
                    "evidence_id": "EV-1",
                    "data_point": "Official metric is available.",
                    "source_id": "SRC-1",
                },
                {
                    "data_point": "Found evidence without a stable id.",
                    "source_id": "SRC-missing",
                    "allowed_use": "public",
                },
            ],
            "source_registry": [{"source_id": "SRC-1", "url": "https://example.com/report"}],
        }
    )

    assert result.ok is False
    assert "evidence_missing_evidence_id" in result.errors
    assert "public_evidence_source_unresolved" in result.errors
    assert result.summary["analysis_candidate_count"] == 2
    assert result.summary["missing_evidence_id_count"] == 1


def test_evidence_package_contract_treats_missing_allowed_use_as_warning_not_public_error():
    result = validate_evidence_package_for_analysis(
        {
            "analysis_ready_evidence": [
                {
                    "evidence_id": "EV-unknown-use",
                    "fact": "Potential evidence has not been tiered yet.",
                    "source_id": "SRC-missing",
                }
            ],
            "source_registry": [],
        }
    )

    assert result.ok is True
    assert "public_evidence_source_unresolved" not in result.errors
    assert "evidence_allowed_use_missing" in result.warnings
    assert result.summary["public_candidate_count"] == 0
    assert result.summary["missing_allowed_use_count"] == 1


def test_structured_analysis_contract_requires_claim_lineage_to_writer():
    result = validate_structured_analysis_for_writer(
        {
            "claim_units": [
                {
                    "claim_id": "CL-1",
                    "claim": "Claim with lineage.",
                    "fact_ids": ["EV-1"],
                    "requirement_ids": ["REQ-1"],
                },
                {"claim": "Claim missing lineage."},
            ]
        },
        evidence_package={"analysis_ready_evidence": [{"evidence_id": "EV-1"}]},
    )

    assert result.ok is False
    assert "claim_missing_claim_id" in result.errors
    assert "claim_missing_fact_or_evidence_refs" in result.errors
    assert result.summary["claim_count"] == 2
    assert result.summary["claims_with_resolved_refs_count"] == 1


def test_structured_analysis_contract_resolves_alias_refs_before_writer():
    result = validate_structured_analysis_for_writer(
        {
            "claim_units": [
                {
                    "claim_id": "CL-1",
                    "claim": "Claim with alias lineage.",
                    "evidence_refs": ["EV-04-L22"],
                    "requirement_ids": ["REQ-1"],
                },
            ]
        },
        evidence_package={
            "analysis_ready_evidence": [
                {
                    "evidence_id": "EV-04-22",
                    "aliases": ["EV-04-L22"],
                    "source_id": "SRC-1",
                }
            ]
        },
    )

    assert result.ok is True
    assert result.summary["claims_with_resolved_refs_count"] == 1
    assert result.summary["evidence_ref_resolution"]["alias_resolved_ref_count"] == 1
    assert result.summary["evidence_ref_resolution"]["unresolved_ref_count"] == 0


def test_repair_priorities_contract_requires_safe_dispatch_fields():
    result = validate_repair_priorities_for_dispatch(
        {
            "evidence_gap_ledger": [
                {
                    "gap_id": "GAP-1",
                    "gap_type": "metric_scope_period_unit_incomplete",
                    "repair_route": "evidence_search",
                    "allowed_for_writing": False,
                },
                {"gap_type": "counter_evidence_missing", "allowed_for_writing": True},
            ]
        }
    )

    assert result.ok is False
    assert "repair_priority_missing_gap_id" in result.errors
    assert "repair_priority_missing_repair_route" in result.errors
    assert "repair_priority_allowed_for_writing_must_be_false" in result.errors
    assert result.summary["repair_priority_count"] == 2
    assert result.summary["dispatch_ready_count"] == 1


def test_writer_report_contract_blocks_factual_body_without_citation():
    result = validate_writer_report_for_final(
        {
            "report_markdown": "## Market\n\nThe market reached 100 billion yuan in 2025.\n\nSupported line [1].",
            "source_registry": [{"ref": "[1]", "url": "https://example.com/report"}],
        }
    )

    assert result.ok is False
    assert "writer_factual_line_without_citation" in result.errors
    assert result.summary["citationless_factual_line_count"] == 1


def test_writer_report_contract_detects_chinese_factual_body_without_citation():
    result = validate_writer_report_for_final(
        {
            "report_markdown": "## 市场\n\n2025年该市场规模达到100亿元，融资和采购活动继续增加。\n\n有引用的判断 [1]。",
            "source_registry": [{"ref": "[1]", "url": "https://example.com/report"}],
        }
    )

    assert result.ok is False
    assert "writer_factual_line_without_citation" in result.errors
    assert result.summary["citationless_factual_line_count"] == 1


def test_citation_reconciliation_contract_resolves_markdown_refs():
    result = validate_citation_reconciliation(
        markdown="Claim [1]. Missing [2].",
        citation_manifest={"items": [{"ref": "[1]", "source_id": "SRC-1"}]},
        source_registry=[{"source_id": "SRC-1", "url": "https://example.com/report"}],
    )

    assert result.ok is False
    assert "citation_ref_missing_from_manifest" in result.errors
    assert result.summary["markdown_ref_count"] == 2
    assert result.summary["resolved_ref_count"] == 1
