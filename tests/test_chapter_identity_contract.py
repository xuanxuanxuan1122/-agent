from rag_pipeline.contracts.chapter_identity import (
    build_chapter_identity_map,
    canonical_chapter_ids,
    resolve_canonical_chapter_id,
)


def test_resolves_title_and_hypothesis_alias_to_canonical_chapter():
    blueprint = {
        "chapters": [
            {
                "chapter_id": "ch_01",
                "chapter_title": "Market validation",
                "chapter_question": "Is demand real?",
                "hypothesis_id": "H1",
            }
        ]
    }

    identity = build_chapter_identity_map(blueprint=blueprint, research_plan={})

    assert resolve_canonical_chapter_id(identity, "ch_01") == "ch_01"
    assert resolve_canonical_chapter_id(identity, "H1") == "ch_01"
    assert resolve_canonical_chapter_id(identity, "Market validation") == "ch_01"
    assert resolve_canonical_chapter_id(identity, "Is demand real?") == "ch_01"
    assert canonical_chapter_ids(identity) == ["ch_01"]


def test_resolves_hypothesis_alias_from_research_plan_by_order():
    blueprint = {
        "chapters": [
            {"chapter_id": "ch_01", "chapter_title": "Demand"},
            {"chapter_id": "ch_02", "chapter_title": "Supply"},
        ]
    }
    research_plan = {
        "hypotheses": [
            {"hypothesis_id": "H1", "statement": "Demand is emerging"},
            {"hypothesis_id": "H2", "statement": "Supply is fragmented"},
        ]
    }

    identity = build_chapter_identity_map(blueprint=blueprint, research_plan=research_plan)

    assert resolve_canonical_chapter_id(identity, "H1") == "ch_01"
    assert resolve_canonical_chapter_id(identity, "H2") == "ch_02"


def test_unknown_raw_chapter_id_is_rejected_not_passed_through():
    identity = build_chapter_identity_map(
        blueprint={"chapters": [{"chapter_id": "ch_01", "chapter_title": "Market"}]},
        research_plan={},
    )

    assert resolve_canonical_chapter_id(identity, "H9") == ""
    assert resolve_canonical_chapter_id(identity, "Not a chapter") == ""
