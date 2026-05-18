from rag_pipeline.agents.writer_agent_clean import prepare_dimension_materials


def test_confident_c_level_evidence_is_directional_signal():
    items = [
        {
            "metric": "市场规模",
            "value": "34亿元",
            "source_level": "C",
            "evidence_role": "clue",
            "semantic_status": "ok",
            "confidence": 0.6,
        }
    ]

    prepared = prepare_dimension_materials(items, dimension="市场规模与增速")

    assert prepared[0]["appendix_only"] is False
    assert prepared[0]["enterprise_usable"] is True
    assert prepared[0]["allowed_use"] == "directional_signal"
    assert prepared[0]["usage_tier"] == "directional_signal"


def test_d_level_kept_as_followup_clue_and_rejected_removed():
    items = [
        {"source_level": "D", "evidence_role": "clue", "semantic_status": "ok"},
        {"source_level": "A", "evidence_role": "rejected", "semantic_status": "ok"},
        {"source_level": "A", "evidence_role": "core", "semantic_status": "rejected"},
    ]

    prepared = prepare_dimension_materials(items, dimension="市场规模与增速")

    assert len(prepared) == 1
    assert prepared[0]["source_level"] == "D"
    assert prepared[0]["appendix_only"] is True
    assert prepared[0]["enterprise_usable"] is False
    assert prepared[0]["followup_seed"] is True
