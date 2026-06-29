from __future__ import annotations

import json
import os
import time
from pathlib import Path

from rag_pipeline.cache.stage_snapshot_cache import (
    list_stage_snapshots,
    load_stage_snapshot,
    prune_stage_snapshots,
    snapshot_is_replayable,
    write_stage_snapshot,
)


def test_stage_snapshot_writes_manifest_and_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("STAGE_SNAPSHOT_CACHE_ENABLED", "true")
    monkeypatch.setenv("STAGE_SNAPSHOT_CACHE_PATH", str(tmp_path))
    monkeypatch.setenv("EVIDENCE_CACHE_READ_ENABLED", "false")
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


def test_prune_stage_snapshots_keeps_current_and_newest(tmp_path, monkeypatch):
    monkeypatch.setenv("STAGE_SNAPSHOT_CACHE_PATH", str(tmp_path))
    monkeypatch.setenv("STAGE_SNAPSHOT_RETENTION_MAX_RUNS", "2")
    monkeypatch.setenv("STAGE_SNAPSHOT_RETENTION_MAX_AGE_DAYS", "0")
    now = time.time()
    # newest -> oldest: current, new, mid, old
    for offset, name in enumerate(["run_current", "run_new", "run_mid", "run_old"]):
        stage_dir = tmp_path / name / "evidence_package"
        stage_dir.mkdir(parents=True, exist_ok=True)
        (stage_dir / "manifest.json").write_text("{}", encoding="utf-8")
        stamp = now - offset * 100
        os.utime(tmp_path / name, (stamp, stamp))

    result = prune_stage_snapshots(keep_run_id="run_current")

    remaining = sorted(child.name for child in tmp_path.iterdir() if child.is_dir())
    assert result["status"] == "pruned"
    # current is always kept and does NOT count toward the cap -> current + newest 2
    assert "run_current" in remaining
    assert "run_new" in remaining and "run_mid" in remaining
    assert "run_old" not in remaining
    assert result["deleted_count"] == 1


def test_replayable_only_skips_full_payload_for_non_replayable_stage(tmp_path, monkeypatch):
    monkeypatch.setenv("STAGE_SNAPSHOT_CACHE_ENABLED", "true")
    monkeypatch.setenv("STAGE_SNAPSHOT_CACHE_PATH", str(tmp_path))
    monkeypatch.setenv("STAGE_SNAPSHOT_FULL_PAYLOAD_REPLAYABLE_ONLY", "true")

    # "research_plan" is not a replayable stage -> full payload is skipped, but
    # the manifest (diagnostics) is still written.
    result = write_stage_snapshot("research_plan", "run-ro", {"big": "x" * 5000})
    assert result["stored"] is True
    assert result["replayable"] is False
    assert result["full_payload_path"] == ""
    assert result["full_payload_skipped"] == "non_replayable_payload_skipped"
    assert (tmp_path / "run-ro" / "research_plan" / "manifest.json").exists()
    assert not (tmp_path / "run-ro" / "research_plan" / "payload.json").exists()
    assert load_stage_snapshot("run-ro", "research_plan")["status"] == "loaded"

    # A replayable stage still persists its full payload under the same lever.
    replayable_payload = {
        "report_blueprint": {"chapters": [{"chapter_id": "ch_01"}]},
        "analysis_ready_evidence": [{"ref": "E1", "source_ref": "S1", "chapter_id": "ch_01"}],
        "source_registry": [{"ref": "S1", "url": "https://example.org/source"}],
    }
    replay_result = write_stage_snapshot("evidence_package", "run-ro", replayable_payload)
    assert replay_result["replayable"] is True
    assert replay_result["full_payload_path"] != ""
    assert replay_result["full_payload_skipped"] == ""


def test_prune_stage_snapshots_disabled_when_no_limits(tmp_path, monkeypatch):
    monkeypatch.setenv("STAGE_SNAPSHOT_CACHE_PATH", str(tmp_path))
    monkeypatch.setenv("STAGE_SNAPSHOT_RETENTION_MAX_RUNS", "0")
    monkeypatch.setenv("STAGE_SNAPSHOT_RETENTION_MAX_AGE_DAYS", "0")
    (tmp_path / "run_a" / "evidence_package").mkdir(parents=True, exist_ok=True)
    result = prune_stage_snapshots(keep_run_id="run_current")
    assert result["status"] == "disabled"
    assert (tmp_path / "run_a").exists()


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
