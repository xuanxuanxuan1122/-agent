from __future__ import annotations

from rag_pipeline.contracts.artifact_validation import (
    validate_claim_unit_lineage,
    validate_context_view,
    validate_fact_card_for_context,
    validate_section_lineage,
)


def test_fact_card_validation_requires_resolvable_source():
    result = validate_fact_card_for_context(
        {"fact_id": "EV-1", "source_id": "SRC-404", "status": "validated", "fact": "A fact."},
        {},
    )

    assert result.ok is False
    assert "source_missing" in result.errors


def test_claim_unit_strength_cannot_exceed_ceiling():
    result = validate_claim_unit_lineage(
        {
            "claim_id": "CL-1",
            "claim_strength": "strong",
            "claim_strength_ceiling": "directional",
            "fact_ids": ["EV-1"],
        },
        [{"fact_id": "EV-1", "status": "validated"}],
    )

    assert result.ok is False
    assert "claim_strength_exceeds_ceiling" in result.errors


def test_claim_unit_requires_requirement_ids_for_lineage():
    result = validate_claim_unit_lineage(
        {
            "claim_id": "CL-1",
            "claim_strength": "directional",
            "claim_strength_ceiling": "directional",
            "fact_ids": ["EV-1"],
        },
        [{"fact_id": "EV-1", "status": "validated"}],
    )

    assert result.ok is False
    assert "claim_unit_missing_requirement_ids" in result.errors


def test_section_validation_rejects_missing_or_stale_fact_refs():
    result = validate_section_lineage(
        {"section_id": "SEC-1", "used_fact_refs": ["EV-1", "EV-stale"], "claim_ids": ["CL-1"]},
        [{"claim_id": "CL-1", "fact_ids": ["EV-1"]}],
        [{"fact_id": "EV-1", "status": "validated"}, {"fact_id": "EV-stale", "status": "stale"}],
        [{"run_source_id": "SRC-1"}],
    )

    assert result.ok is False
    assert "section_references_unusable_fact" in result.errors


def test_writer_context_validation_rejects_forbidden_fields_and_unbound_claim_text():
    result = validate_context_view(
        {
            "task": "write_section",
            "status": "ready",
            "claim_units": [{"claim": "Company A reached 12 deployments in 2025."}],
            "usable_fact_cards": [{"fact": "Company A disclosed enterprise deployments."}],
            "raw_page": "forbidden",
            "section_draft": "forbidden",
        },
        task_type="write_section",
    )

    assert result.ok is False
    assert "writer_context_contains_forbidden_field" in result.errors
    assert "writer_context_unbound_numeric_or_date" in result.errors


def test_writer_context_validation_rejects_unbound_company_names():
    result = validate_context_view(
        {
            "task": "write_section",
            "status": "ready",
            "claim_units": [{"claim": "Acme Corp expanded agent deployments."}],
            "usable_fact_cards": [{"fact": "Agent deployments were disclosed by an unnamed customer."}],
        },
        task_type="write_section",
    )

    assert result.ok is False
    assert "writer_context_unbound_company_name" in result.errors
