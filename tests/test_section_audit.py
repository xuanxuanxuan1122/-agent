from __future__ import annotations

from rag_pipeline.contracts.section_audit import audit_section_claim_roles


def test_section_audit_turns_missing_boundary_and_metric_fields_into_score_gaps():
    audit = audit_section_claim_roles(
        {
            "section_id": "SEC-1",
            "chapter_id": "ch_01",
            "claim_ids": ["CL-1"],
            "used_fact_refs": ["EV-1"],
            "evidence_backed": True,
        },
        {
            "CL-1": {
                "claim_id": "CL-1",
                "requirement_ids": ["H1_metric"],
                "fact_ids": ["EV-1"],
                "claim_strength": "moderate",
                "claim_roles": ["core_claim", "metric_claim"],
            }
        },
        {
            "EV-1": {
                "fact_id": "EV-1",
                "requirement_id": "H1_metric",
                "proof_role": "metric",
                "metric": "adoption",
                "value": "42",
                "unit": "",
                "period": "",
                "source_id": "SRC-1",
            }
        },
    )

    assert audit["status"] == "needs_repair"
    assert {item["type"] for item in audit["findings"]} == {
        "section_missing_counter_boundary",
        "section_metric_missing_fields",
    }
    gaps = {item["gap_type"]: item for item in audit["score_gaps"]}
    assert gaps["counter_boundary_missing"]["requirement_id"] == "H1_metric"
    assert gaps["counter_boundary_missing"]["section_id"] == "SEC-1"
    assert gaps["counter_boundary_missing"]["retry_plan"]["proof_role"] == "counter"
    assert gaps["metric_scope_period_unit_incomplete"]["missing"] == ["unit", "period"]
    assert gaps["metric_scope_period_unit_incomplete"]["retry_plan"]["required_fields"] == [
        "metric",
        "value",
        "unit",
        "period",
        "source",
    ]
    assert gaps["metric_scope_period_unit_incomplete"]["retry_plan"]["allowed_for_writing"] is False


def test_section_audit_passes_when_core_claim_has_boundary_and_complete_metric():
    audit = audit_section_claim_roles(
        {
            "section_id": "SEC-2",
            "chapter_id": "ch_01",
            "claim_ids": ["CL-2"],
            "used_fact_refs": ["EV-2"],
            "evidence_backed": True,
        },
        {
            "CL-2": {
                "claim_id": "CL-2",
                "requirement_ids": ["H1_metric"],
                "fact_ids": ["EV-2"],
                "claim_strength": "moderate",
                "claim_roles": ["core_claim", "metric_claim", "boundary_claim"],
                "limitation_boundary": "Limited to reported survey scope.",
            }
        },
        {
            "EV-2": {
                "fact_id": "EV-2",
                "requirement_id": "H1_metric",
                "proof_role": "metric",
                "metric": "adoption",
                "value": "42",
                "unit": "%",
                "period": "2025",
                "source_id": "SRC-1",
            }
        },
    )

    assert audit["status"] == "pass"
    assert audit["findings"] == []
    assert audit["score_gaps"] == []
