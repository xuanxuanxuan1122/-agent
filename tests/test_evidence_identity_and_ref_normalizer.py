from rag_pipeline.contracts.evidence_identity import (
    build_evidence_alias_map,
    canonicalize_evidence_id,
    resolve_evidence_refs,
)
from rag_pipeline.contracts.ref_normalizer import normalize_claim_refs


def test_evidence_alias_map_resolves_explicit_aliases_without_fuzzy_merge():
    fact_cards = [
        {
            "evidence_id": "EV-04-22",
            "aliases": ["EV-04-L22", "ev_04_l22"],
            "source_id": "SRC-1",
        },
        {
            "evidence_id": "EV-04-23",
            "aliases": ["EV-04-L23"],
            "source_id": "SRC-2",
        },
    ]

    alias_map = build_evidence_alias_map(fact_cards)
    resolved = resolve_evidence_refs(["EV-04-L22", "EV-04-22", "EV-04-L24"], alias_map)

    assert canonicalize_evidence_id("ev_04_l22", alias_map) == "EV-04-22"
    assert resolved["resolved_fact_ids"] == ["EV-04-22"]
    assert resolved["alias_resolved_refs"] == [{"raw_ref": "EV-04-L22", "canonical_ref": "EV-04-22"}]
    assert resolved["unresolved_refs"] == ["EV-04-L24"]
    assert resolved["ambiguous_refs"] == []
    assert canonicalize_evidence_id("EV-04-L24", alias_map) == ""


def test_evidence_alias_map_marks_conflicting_alias_as_ambiguous():
    alias_map = build_evidence_alias_map(
        [
            {"evidence_id": "EV-A", "aliases": ["EV-shared"]},
            {"evidence_id": "EV-B", "aliases": ["EV-shared"]},
        ]
    )

    resolved = resolve_evidence_refs(["EV-shared"], alias_map)

    assert resolved["resolved_fact_ids"] == []
    assert resolved["unresolved_refs"] == []
    assert resolved["ambiguous_refs"] == ["EV-shared"]


def test_normalize_claim_refs_outputs_canonical_fields_and_keeps_legacy_diagnostics():
    fact_cards = [
        {
            "evidence_id": "EV-04-22",
            "aliases": ["EV-04-L22"],
            "source_id": "SRC-1",
            "requirement_id": "REQ-1",
        }
    ]
    alias_map = build_evidence_alias_map(fact_cards)

    normalized = normalize_claim_refs(
        {
            "claim_id": "CL-1",
            "claim": "A supported claim.",
            "evidence_refs": ["EV-04-L22"],
            "supporting_evidence_refs": ["EV-ignored-legacy"],
            "source_ref": "SRC-1",
            "goal_id": "REQ-legacy",
        },
        alias_map=alias_map,
        fact_cards=fact_cards,
        source_alias_map={"SRC-1": "SRC-1"},
    )

    assert normalized["fact_ids"] == ["EV-04-22"]
    assert normalized["source_ids"] == ["SRC-1"]
    assert normalized["requirement_ids"] == ["REQ-1"]
    assert normalized["legacy_ref_fields"]["evidence_refs"] == ["EV-04-L22"]
    assert normalized["ref_resolution"]["alias_resolved_refs"] == [
        {"raw_ref": "EV-04-L22", "canonical_ref": "EV-04-22"}
    ]
    assert normalized["unresolved_refs"] == ["EV-ignored-legacy"]


def test_normalize_claim_refs_preserves_existing_fact_ids_without_fact_cards():
    normalized = normalize_claim_refs(
        {
            "claim_id": "CL-pass",
            "fact_ids": ["EV-existing"],
            "source_ids": ["SRC-existing"],
            "requirement_ids": ["REQ-existing"],
        }
    )

    assert normalized["fact_ids"] == ["EV-existing"]
    assert normalized["source_ids"] == ["SRC-existing"]
    assert normalized["requirement_ids"] == ["REQ-existing"]
    assert normalized["unresolved_refs"] == []
