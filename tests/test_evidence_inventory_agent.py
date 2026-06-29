from rag_pipeline.agents.evidence_inventory_agent import build_evidence_inventory


def test_inventory_groups_curated_evidence_by_cluster_and_fact_type():
    curated_payload = {
        "curated_evidence": [
            {
                "evidence_id": "CE-market-1",
                "requirement_id": "REQ-market",
                "chapter_id": "ch_market",
                "cluster_key": "market",
                "fact_type": "market_signal",
                "clean_fact": "A media report says small-batch humanoid robot delivery is starting in 2026.",
                "source_level": "C",
                "claim_strength_hint": "directional",
                "usable_for": ["market_size", "commercialization"],
                "limitations": ["secondary source"],
            },
            {
                "evidence_id": "CE-policy-1",
                "requirement_id": "REQ-policy",
                "chapter_id": "ch_policy",
                "fact_type": "policy_signal",
                "clean_fact": "A local policy document lists embodied intelligence as a supported industry direction.",
                "source_level": "A",
                "claim_strength_hint": "moderate",
                "usable_for": ["policy"],
            },
        ]
    }

    result = build_evidence_inventory(curated_payload, query="humanoid robot industry")

    assert result["schema_version"] == "evidence_inventory_v1"
    assert result["inventory_count"] == 2
    market = result["inventories_by_cluster"]["market"]
    assert market["curated_evidence_count"] == 1
    assert market["usable_evidence_ids"] == ["CE-market-1"]
    assert market["fact_type_counts"]["market_signal"] == 1
    assert market["dominant_strength"] == "directional"
    assert "small-batch humanoid robot delivery" in market["analysis_brief"]
    policy = result["inventories_by_cluster"]["policy"]
    assert policy["strongest_available_level"] == "A"
    assert policy["dominant_strength"] == "moderate"


def test_inventory_marks_empty_payload_insufficient():
    result = build_evidence_inventory({"curated_evidence": []}, query="empty")

    assert result["status"] == "insufficient"
    assert result["inventory_count"] == 0
