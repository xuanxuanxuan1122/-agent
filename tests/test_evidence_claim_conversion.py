from rag_pipeline.observability.evidence_claim_conversion import build_evidence_claim_conversion_monitor


def test_conversion_monitor_tracks_curated_evidence_to_claim_rate():
    writer_package = {
        "evidence_package": {
            "analysis_ready_evidence": [
                {"evidence_id": "EV-1", "chapter_id": "ch_01", "requirement_id": "REQ-1", "source_id": "SRC-1"},
                {"evidence_id": "EV-2", "chapter_id": "ch_01", "requirement_id": "REQ-2", "source_id": "SRC-2"},
            ],
            "curated_evidence": {
                "curated_evidence": [
                    {"evidence_id": "EV-1", "merged_evidence_ids": ["EV-1"]},
                    {"evidence_id": "EV-2", "merged_evidence_ids": ["EV-2"]},
                ]
            },
        },
        "structured_analysis": {
            "claim_units": [
                {
                    "claim_id": "CL-1",
                    "chapter_id": "ch_01",
                    "fact_ids": ["EV-1"],
                    "source_ids": ["SRC-1"],
                }
            ]
        },
        "sections": [{"chapter_id": "ch_01", "claim_ids": ["CL-1"]}],
    }

    result = build_evidence_claim_conversion_monitor(writer_package=writer_package)

    totals = result["totals"]
    assert totals["curated_evidence_count"] == 2
    assert totals["curated_used_in_claim_count"] == 1
    assert totals["curated_to_claim_rate"] == 0.5


def test_conversion_monitor_tracks_inventory_to_claim_rate():
    writer_package = {
        "evidence_package": {
            "analysis_ready_evidence": [
                {"evidence_id": "EV-1", "chapter_id": "ch_01", "requirement_id": "REQ-1", "source_id": "SRC-1"},
                {"evidence_id": "EV-2", "chapter_id": "ch_01", "requirement_id": "REQ-2", "source_id": "SRC-2"},
            ],
            "curated_evidence": {
                "curated_evidence": [
                    {"evidence_id": "EV-1"},
                    {"evidence_id": "EV-2"},
                ]
            },
            "evidence_inventory": {
                "inventories": [
                    {
                        "inventory_id": "INV-market",
                        "cluster_key": "market",
                        "usable_evidence_ids": ["EV-1", "EV-2"],
                    }
                ]
            },
            "analysis_shards": [
                {
                    "cluster_key": "market",
                    "curated_evidence_ids": ["EV-1", "EV-2"],
                    "input_hash": "hash-market",
                }
            ],
        },
        "structured_analysis": {
            "claim_units": [
                {
                    "claim_id": "CL-1",
                    "chapter_id": "ch_01",
                    "fact_ids": ["EV-2"],
                    "source_ids": ["SRC-2"],
                }
            ]
        },
    }

    result = build_evidence_claim_conversion_monitor(writer_package=writer_package)

    totals = result["totals"]
    assert totals["inventory_cluster_count"] == 1
    assert totals["inventory_evidence_id_count"] == 2
    assert totals["inventory_used_in_claim_count"] == 1
    assert totals["inventory_to_claim_rate"] == 0.5
    assert totals["analysis_shard_count"] == 1


def test_conversion_monitor_tracks_analysis_shard_cache_statuses():
    writer_package = {
        "evidence_package": {
            "analysis_shards": [
                {"cluster_key": "market", "curated_evidence_ids": ["EV-1"], "input_hash": "hash-market"},
                {"cluster_key": "policy", "curated_evidence_ids": ["EV-2"], "input_hash": "hash-policy"},
                {"cluster_key": "risk", "curated_evidence_ids": ["EV-3"], "input_hash": "hash-risk"},
            ]
        },
        "structured_analysis": {
            "analysis_stage_diagnostics": {
                "llm_chapter_results": [
                    {
                        "chapter_id": "ch_market",
                        "analysis_shard_output_cache": {
                            "status": "hit",
                            "input_hash": "hash-market",
                            "output_cache_path": "cache/market.json",
                        },
                    },
                    {
                        "chapter_id": "ch_policy",
                        "analysis_shard_output_cache": {
                            "status": "miss",
                            "reason": "contract_mismatch",
                            "input_hash": "hash-policy",
                        },
                    },
                    {
                        "chapter_id": "ch_risk",
                        "analysis_shard_output_cache": {
                            "status": "stored",
                            "input_hash": "hash-risk",
                        },
                    },
                ]
            }
        },
    }

    result = build_evidence_claim_conversion_monitor(writer_package=writer_package)

    totals = result["totals"]
    assert totals["analysis_shard_cache_hit_count"] == 1
    assert totals["analysis_shard_cache_miss_count"] == 1
    assert totals["analysis_shard_output_cache_stored_count"] == 1
    assert totals["analysis_shard_cache_saved_llm_call_count"] == 1
    assert result["analysis_shard_cache"]["miss_reason_counts"] == {"contract_mismatch": 1}
    assert result["analysis_shard_cache"]["items"][0]["status"] == "hit"
