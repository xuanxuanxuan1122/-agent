from __future__ import annotations

from rag_pipeline.contracts.claim_roles import classify_claim_unit_roles


def test_metric_claim_role_prefers_metric_primary_and_keeps_core_strength():
    result = classify_claim_unit_roles(
        {
            "claim": "Enterprise AI agent adoption is rising.",
            "claim_strength": "moderate",
            "block_affinity": ["metric_reconciliation"],
            "fact_ids": ["EV-metric"],
        },
        {
            "EV-metric": {
                "proof_role": "metric",
                "metric": "adoption",
                "value": "42",
                "unit": "%",
                "period": "2025",
            }
        },
    )

    assert result["primary_claim_role"] == "metric_claim"
    assert "metric_claim" in result["claim_roles"]
    assert "core_claim" in result["claim_roles"]
    assert result["claim_role_contract_version"] == "claim_roles_v1"


def test_counter_evidence_does_not_become_core_claim():
    result = classify_claim_unit_roles(
        {
            "claim": "Security concerns limit adoption.",
            "claim_strength": "moderate",
            "analysis_role": "counter",
            "fact_ids": ["EV-risk"],
        },
        {"EV-risk": {"proof_role": "counter", "allowed_use": "counter"}},
    )

    assert result["primary_claim_role"] == "counter_claim"
    assert "counter_claim" in result["claim_roles"]
    assert "core_claim" not in result["claim_roles"]


def test_mechanism_and_boundary_roles_are_detected_from_claim_payload():
    result = classify_claim_unit_roles(
        {
            "claim": "Workflow integration explains adoption.",
            "claim_strength": "directional",
            "reasoning_chain": ["Workflow orchestration reduces manual handoffs."],
            "limitation_boundary": ["The claim is limited to disclosed enterprise samples."],
        },
        {},
    )

    assert "mechanism_claim" in result["claim_roles"]
    assert "boundary_claim" in result["claim_roles"]
    assert result["primary_claim_role"] == "mechanism_claim"
