import json
from pathlib import Path

from rag_pipeline.cache.analysis_memory_cache import (
    build_analysis_shards,
    load_analysis_shard_output,
    persist_analysis_memory_cache,
    persist_analysis_shard_output,
)
from rag_pipeline.cache.artifact_store import ArtifactStore


def _evidence_package():
    return {
        "run_id": "run-cache-1",
        "evidence_inventory": {
            "schema_version": "evidence_inventory_v1",
            "status": "ready",
            "inventories": [
                {
                    "inventory_id": "INV-market",
                    "cluster_key": "market",
                    "chapter_id": "ch_market",
                    "usable_evidence_ids": ["CE-1", "CE-2"],
                    "analysis_brief": "Market evidence supports a bounded commercialization argument.",
                }
            ],
        },
        "curated_evidence": {
            "curated_evidence": [
                {
                    "evidence_id": "CE-1",
                    "cluster_key": "market",
                    "chapter_id": "ch_market",
                    "clean_fact": "A vertical media report says pilot delivery is beginning.",
                    "source_url": "https://www.cs.com.cn/news/humanoid",
                },
                {
                    "evidence_id": "CE-2",
                    "cluster_key": "market",
                    "chapter_id": "ch_market",
                    "clean_fact": "A company website describes a small batch deployment case.",
                    "source_url": "https://www.example-company.cn/case",
                },
            ]
        },
    }


def test_build_analysis_shards_from_inventory_and_curated_evidence():
    shards = build_analysis_shards(_evidence_package())

    assert len(shards) == 1
    shard = shards[0]
    assert shard["schema_version"] == "analysis_shard_input_v1"
    assert shard["cluster_key"] == "market"
    assert shard["chapter_id"] == "ch_market"
    assert shard["curated_evidence_ids"] == ["CE-1", "CE-2"]
    assert shard["input_hash"]
    assert "bounded commercialization" in shard["analysis_brief"]


def test_persist_analysis_memory_cache_writes_files_and_ledger_artifacts(tmp_path):
    store = ArtifactStore(path=tmp_path / "ledger.sqlite", object_root=tmp_path / "objects", inline_max_bytes=32)
    store.upsert_run(run_id="run-cache-1", query="q", status="running")

    result = persist_analysis_memory_cache(
        _evidence_package(),
        run_id="run-cache-1",
        cache_root=tmp_path / "analysis_memory",
        artifact_store=store,
    )

    assert result["status"] == "stored"
    assert Path(result["inventory_cache_path"]).exists()
    assert len(result["analysis_shard_cache_paths"]) == 1
    shard_path = Path(result["analysis_shard_cache_paths"][0])
    assert json.loads(shard_path.read_text(encoding="utf-8"))["cluster_key"] == "market"

    inventory_artifact = store.get_artifact(result["inventory_artifact_id"])
    assert inventory_artifact["stage"] == "evidence_inventory"
    assert inventory_artifact["artifact_type"] == "evidence_inventory"
    shard_artifact = store.get_artifact(result["analysis_shard_artifact_ids"][0])
    assert shard_artifact["stage"] == "analysis_shard_input"
    assert shard_artifact["artifact_type"] == "analysis_shard_input"


def test_persist_analysis_shard_output_writes_file_and_ledger_artifact(tmp_path):
    store = ArtifactStore(path=tmp_path / "ledger.sqlite", object_root=tmp_path / "objects", inline_max_bytes=32)
    store.upsert_run(run_id="run-cache-1", query="q", status="running")
    evidence_package = _evidence_package()
    shard = build_analysis_shards(evidence_package)[0]

    result = persist_analysis_shard_output(
        evidence_package,
        chapter_payload={"chapter_id": "ch_market", "analysis_shard": shard},
        analysis_result={
            "chapter_synthesis": [
                {
                    "chapter_id": "ch_market",
                    "claim_units": [{"claim": "Pilot delivery is beginning.", "used_evidence_ids": ["CE-1"]}],
                }
            ]
        },
        run_id="run-cache-1",
        cache_root=tmp_path / "analysis_memory",
        artifact_store=store,
        model="deepseek-test",
        prompt_version="prompt-v1",
    )

    assert result["status"] == "stored"
    output_path = Path(result["output_cache_path"])
    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "analysis_shard_output_v1"
    assert payload["input_hash"] == shard["input_hash"]
    assert payload["model"] == "deepseek-test"
    artifact = store.get_artifact(result["artifact_id"])
    assert artifact["stage"] == "analysis_shard_output"
    assert artifact["artifact_type"] == "analysis_shard_output"


def test_load_analysis_shard_output_requires_exact_contract_match(tmp_path):
    evidence_package = _evidence_package()
    shard = build_analysis_shards(evidence_package)[0]
    analysis_result = {
        "chapter_synthesis": [
            {
                "chapter_id": "ch_market",
                "claim_units": [{"claim": "Pilot delivery is beginning.", "used_evidence_ids": ["CE-1"]}],
            }
        ]
    }
    persist_analysis_shard_output(
        evidence_package,
        chapter_payload={"chapter_id": "ch_market", "analysis_shard": shard},
        analysis_result=analysis_result,
        run_id="run-cache-1",
        cache_root=tmp_path / "analysis_memory",
        artifact_store=None,
        model="deepseek-test",
        prompt_version="prompt-v1",
    )

    hit = load_analysis_shard_output(
        chapter_payload={"chapter_id": "ch_market", "analysis_shard": shard},
        run_id="run-cache-1",
        cache_root=tmp_path / "analysis_memory",
        model="deepseek-test",
        prompt_version="prompt-v1",
    )
    model_miss = load_analysis_shard_output(
        chapter_payload={"chapter_id": "ch_market", "analysis_shard": shard},
        run_id="run-cache-1",
        cache_root=tmp_path / "analysis_memory",
        model="other-model",
        prompt_version="prompt-v1",
    )
    prompt_miss = load_analysis_shard_output(
        chapter_payload={"chapter_id": "ch_market", "analysis_shard": shard},
        run_id="run-cache-1",
        cache_root=tmp_path / "analysis_memory",
        model="deepseek-test",
        prompt_version="prompt-v2",
    )

    assert hit["status"] == "hit"
    assert hit["analysis_result"] == analysis_result
    assert model_miss["status"] == "miss"
    assert model_miss["reason"] == "contract_mismatch"
    assert prompt_miss["status"] == "miss"
    assert prompt_miss["reason"] == "contract_mismatch"
