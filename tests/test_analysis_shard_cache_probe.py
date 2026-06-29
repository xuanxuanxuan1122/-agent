from rag_pipeline.agents import analysis_agent
from rag_pipeline.observability.stage_probe import build_stage_probe_packets


def _evidence_package():
    return {
        "run_id": "run-shard-probe",
        "query": "humanoid robot industry",
        "curated_evidence": {
            "curated_evidence": [
                {
                    "evidence_id": "CE-1",
                    "chapter_id": "ch_01",
                    "cluster_key": "market",
                    "clean_fact": "A vertical media report says pilot delivery is beginning.",
                    "fact_type": "market_signal",
                    "source_id": "SRC-1",
                    "source_url": "https://www.cs.com.cn/news/humanoid",
                }
            ]
        },
        "evidence_inventory": {
            "inventories": [
                {
                    "inventory_id": "INV-market",
                    "cluster_key": "market",
                    "chapter_id": "ch_01",
                    "usable_evidence_ids": ["CE-1"],
                    "analysis_brief": "Market signal can support a bounded directional claim.",
                }
            ]
        },
        "analysis_shards": [
            {
                "cluster_key": "market",
                "chapter_id": "ch_01",
                "input_hash": "hash-market-probe",
                "curated_evidence_ids": ["CE-1"],
            }
        ],
        "chapter_evidence_diagnostics": {
            "ch_01": {"chapter_id": "ch_01", "chapter_title": "Market", "chapter_question": "Market?"}
        },
    }


def test_second_analysis_run_hits_shard_cache_and_stage_probe_surfaces_it(monkeypatch, tmp_path):
    call_count = {"count": 0}

    def fake_llm(**kwargs):
        call_count["count"] += 1
        return {
            "payload": {
                "chapter_id": "ch_01",
                "claim_units": [
                    {
                        "claim": "Pilot delivery is beginning.",
                        "used_evidence_ids": ["CE-1"],
                        "fact_ids": ["CE-1"],
                        "source_ids": ["SRC-1"],
                        "claim_strength": "directional",
                    }
                ],
            },
            "usage": {},
        }

    monkeypatch.setattr(analysis_agent, "call_openai_compatible_json", fake_llm)
    monkeypatch.setattr(analysis_agent, "llm_config_is_ready", lambda cfg: True)
    monkeypatch.setattr(analysis_agent, "normalize_llm_config", lambda cfg: {"model": "deepseek-test"})
    monkeypatch.setenv("BRAIN_LLM_ANALYSIS_CACHE_ENABLED", "false")
    monkeypatch.setenv("BRAIN_LLM_ANALYSIS_CONCURRENCY", "1")
    monkeypatch.setenv("BRAIN_LLM_ANALYSIS_MAX_RETRIES", "0")
    monkeypatch.setenv("REPORT_ANALYSIS_SHARD_OUTPUT_CACHE_READ_ENABLED", "true")
    monkeypatch.setenv("ARTIFACT_LEDGER_ENABLED", "false")
    monkeypatch.setenv("ANALYSIS_MEMORY_CACHE_PATH", str(tmp_path / "analysis_memory"))
    evidence_package = _evidence_package()

    analysis_agent.synthesize_with_llm_analysis_v2(
        evidence_package=evidence_package,
        fallback={"query": evidence_package["query"]},
        llm_config={"provider": "fake", "model": "deepseek-test"},
    )
    second = analysis_agent.synthesize_with_llm_analysis_v2(
        evidence_package=evidence_package,
        fallback={"query": evidence_package["query"]},
        llm_config={"provider": "fake", "model": "deepseek-test"},
    )

    writer_package = {
        "evidence_package": evidence_package,
        "structured_analysis": {
            "analysis_stage_diagnostics": {
                "llm_chapter_results": second["_llm_chapter_results"],
            },
            "claim_units": [
                {"claim_id": "CL-1", "chapter_id": "ch_01", "fact_ids": ["CE-1"], "source_ids": ["SRC-1"]}
            ],
        },
    }
    packet = {item["stage"]: item for item in build_stage_probe_packets(run_id="run-shard-probe", writer_package=writer_package)}[
        "evidence_claim_conversion"
    ]

    assert call_count["count"] == 1
    assert second["_llm_chapter_results"][0]["analysis_shard_output_cache"]["status"] == "hit"
    assert packet["cache"]["analysis_shard_cache_hit_count"] == 1
    assert packet["cache"]["analysis_shard_cache_saved_llm_call_count"] == 1
