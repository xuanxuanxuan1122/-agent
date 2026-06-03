from __future__ import annotations

import os
from pathlib import Path

from rag_pipeline.cache.stage_snapshot_cache import stage_snapshot_cache_root
from rag_pipeline.cache.topic_bundle_cache import _candidate_bundle_dirs, _json_write, build_topic_key, load_topic_bundle, topic_bundle_cache_root


def test_relative_cache_paths_resolve_to_project_root(monkeypatch):
    monkeypatch.delenv("TOPIC_BUNDLE_CACHE_PATH", raising=False)
    monkeypatch.delenv("STAGE_SNAPSHOT_CACHE_PATH", raising=False)

    assert topic_bundle_cache_root().as_posix().endswith("current_rag_pipeline/output/cache/topic_bundles")
    assert stage_snapshot_cache_root().as_posix().endswith("current_rag_pipeline/output/cache/stage_snapshots")


def test_topic_bundle_query_alias_scan_finds_same_query_with_different_hash(tmp_path, monkeypatch):
    monkeypatch.setenv("TOPIC_BUNDLE_CACHE_PATH", str(tmp_path))
    query = "AI Agent ecosystem report"
    bundle_dir = tmp_path / "ai_agent_ecosystem_report__oldhash"
    _json_write(
        bundle_dir / "manifest.json",
        {
            "schema_version": 1,
            "topic_key": bundle_dir.name,
            "query": query,
            "query_normalized": "ai agent ecosystem report",
            "updated_at": "2026-01-01T00:00:00Z",
        },
    )
    _json_write(bundle_dir / "evidence_package.json", {"analysis_ready_evidence": [{"ref": "E1"}]})
    _json_write(bundle_dir / "source_registry.json", [{"ref": "S1", "url": "https://example.org/a", "title": "A"}])
    _json_write(bundle_dir / "chapter_evidence_packages.json", [{"chapter_id": "ch_01", "core_evidence": [{"ref": "E1"}]}])

    new_key = build_topic_key(query, {"report_family": "industry_deep_report"}, {})
    assert new_key != bundle_dir.name
    candidates = _candidate_bundle_dirs(query, new_key)
    assert bundle_dir in candidates
    loaded = load_topic_bundle(query=query, research_plan={"report_family": "industry_deep_report"}, report_blueprint={})
    assert loaded["found"] is True
    assert Path(loaded["path"]) == bundle_dir
