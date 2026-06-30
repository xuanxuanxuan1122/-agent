from rag_pipeline.agents.layout_claim_matcher import BLOCK_TYPES, claim_supported_block_types


def _renderable_claim(role: str) -> dict:
    return {
        "claim": "政策与产业信号推动具身智能进入快速发展期",
        "proof_role": role,
        "evidence_refs": ["EV-1"],
        "supporting_facts": [{"fact_id": "EV-1"}],
        "claim_strength": "moderate",
    }


def test_profile_analytical_block_types_are_recognized():
    for block_type in ("mechanism_chain", "policy_timeline", "stakeholder_map", "value_chain_map"):
        assert block_type in BLOCK_TYPES


def test_claim_roles_map_to_profile_analytical_blocks_not_only_integrated_signal():
    # Previously every role collapsed to integrated_signal because these block
    # types were unknown to the matcher; now the profile-expected blocks appear.
    assert "policy_timeline" in claim_supported_block_types(_renderable_claim("policy_original"))
    assert "stakeholder_map" in claim_supported_block_types(_renderable_claim("official_data"))
    assert "mechanism_chain" in claim_supported_block_types(_renderable_claim("mechanism"))
    # Existing mappings still hold.
    assert "risk_trigger" in claim_supported_block_types(_renderable_claim("counter"))
    assert "metric_reconciliation" in claim_supported_block_types(_renderable_claim("metric"))
