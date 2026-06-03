from __future__ import annotations

import json
from pathlib import Path

from rag_pipeline.cache.stage_snapshot_cache import (
    list_stage_snapshots,
    load_stage_snapshot,
    snapshot_is_replayable,
    write_stage_snapshot,
)


def test_stage_snapshot_writes_manifest_and_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("STAGE_SNAPSHOT_CACHE_ENABLED", "true")
    monkeypatch.setenv("STAGE_SNAPSHOT_CACHE_PATH", str(tmp_path))
    payload = {
        "report_blueprint": {"chapters": [{"chapter_id": "ch_01", "chapter_title": "Demand"}]},
        "analysis_ready_evidence": [
            {"ref": "E1", "source_ref": "S1", "fact": "AI Agent adoption signal", "chapter_id": "ch_01"}
        ],
        "source_registry": [{"ref": "S1", "url": "https://example.org/source"}],
    }

    result = write_stage_snapshot("evidence_package", "run-1", payload, summary={"input": "unit"})

    assert result["stored"] is True
    assert result["replayable"] is True
    manifest_path = Path(result["full_payload_path"]).with_name("manifest.json")
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["stage_name"] == "evidence_package"

    loaded = load_stage_snapshot("run-1", "evidence_package")
    assert loaded["status"] == "loaded"
    assert loaded["payload"] == payload
    assert [item["stage_name"] for item in list_stage_snapshots("run-1")] == ["evidence_package"]


def test_stage_snapshot_compresses_large_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("STAGE_SNAPSHOT_CACHE_ENABLED", "true")
    monkeypatch.setenv("STAGE_SNAPSHOT_CACHE_PATH", str(tmp_path))
    monkeypatch.setenv("STAGE_SNAPSHOT_COMPRESS_LARGE_PAYLOAD", "true")
    payload = {"text": "x" * (1024 * 1024 + 32)}

    result = write_stage_snapshot("writer_report", "run-large", payload)

    assert result["full_payload_compressed"] is True
    loaded = load_stage_snapshot("run-large", "writer_report")
    assert loaded["payload"] == payload


def test_replayable_contract_rejects_non_replay_stages():
    assert snapshot_is_replayable("research_plan", {"chapters": []}) is False
    assert snapshot_is_replayable("chapter_packages", [{"chapter_id": "ch_01"}]) is True
    assert snapshot_is_replayable("chapter_packages", {"count": 1}) is False


def test_evidence_package_replayable_requires_blueprint_evidence_and_sources():
    assert snapshot_is_replayable("evidence_package", {"analysis_ready_evidence": [{"ref": "E1"}]}) is False
    assert (
        snapshot_is_replayable(
            "evidence_package",
            {
                "report_blueprint": {"chapters": [{"chapter_id": "ch_01"}]},
                "analysis_ready_evidence": [{"ref": "E1", "source_ref": "S1", "chapter_id": "ch_01"}],
                "source_registry": [{"ref": "S1", "url": "https://example.org/source"}],
            },
        )
        is True
    )
