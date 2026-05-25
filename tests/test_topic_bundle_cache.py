from __future__ import annotations

from rag_pipeline.cache.topic_bundle_cache import (
    build_topic_key,
    bundle_to_writer_inputs,
    is_compacted_evidence_package,
    load_topic_bundle,
    preflight_topic_bundle,
    store_topic_bundle,
)


def _base_evidence_package():
    return {
        "analysis_ready_evidence": [
            {
                "evidence_id": "ev1",
                "fact": "Enterprise AI agents are moving from pilots to workflow automation.",
                "source_level": "B",
                "source_url": "https://www.nist.gov/ai/agents",
                "source_title": "AI agents guidance",
            }
        ],
        "source_registry": [
            {
                "ref": "S1",
                "title": "AI agents guidance",
                "url": "https://www.nist.gov/ai/agents",
                "publisher": "NIST",
                "source_level": "B",
            }
        ],
        "evidence_health_summary": {"analysis_ready_count": 1, "traceable_ab_source_count": 1},
    }


def _chapter_packages():
    return [
        {
            "chapter_id": "ch_01",
            "chapter_title": "Demand validation",
            "core_evidence": [
                {
                    "evidence_id": "ev1",
                    "fact": "Enterprise AI agents are moving from pilots to workflow automation.",
                    "source_level": "B",
                    "source_url": "https://www.nist.gov/ai/agents",
                }
            ],
        }
    ]


def test_topic_key_is_stable_for_same_scope():
    query = "AI Agent生态发展报告：从工具到智能体的范式跃迁"

    assert build_topic_key(query) == build_topic_key(query)


def test_store_load_preflight_usable_bundle(tmp_path, monkeypatch):
    monkeypatch.setenv("TOPIC_BUNDLE_CACHE_PATH", str(tmp_path))
    monkeypatch.setenv("TOPIC_BUNDLE_CACHE_ENABLED", "true")
    monkeypatch.setenv("TOPIC_BUNDLE_CACHE_REQUIRE_HYDRATED_EVIDENCE", "true")

    stored = store_topic_bundle(
        query="AI Agent report",
        report_blueprint={"report_family": "industry_deep_report", "chapters": [{"chapter_id": "ch_01", "chapter_title": "Demand validation"}]},
        evidence_package=_base_evidence_package(),
        structured_analysis={"claim_units": [{"claim": "Demand is emerging from workflow automation."}]},
        chapter_evidence_packages=_chapter_packages(),
        micro_layouts=[],
        table_packages=[],
        stage="test",
    )
    assert stored["stored"] is True

    loaded = load_topic_bundle("AI Agent report")
    preflight = preflight_topic_bundle(loaded, query="AI Agent report")

    assert loaded["found"] is True
    assert preflight["status"] == "usable"
    assert preflight["seedable"] is True
    assert preflight["usable_for_skip_search"] is True
    assert preflight["can_skip_search"] is False
    inputs = bundle_to_writer_inputs(loaded, preflight=preflight)
    assert inputs["evidence_package"]["analysis_ready_evidence"]
    assert inputs["seed_evidence_count"] == 1


def test_skip_search_requires_explicit_config(tmp_path, monkeypatch):
    monkeypatch.setenv("TOPIC_BUNDLE_CACHE_PATH", str(tmp_path))
    monkeypatch.setenv("TOPIC_BUNDLE_CACHE_ALLOW_SKIP_SEARCH", "true")

    store_topic_bundle(
        query="AI Agent report",
        evidence_package=_base_evidence_package(),
        structured_analysis={"claim_units": [{"claim": "Demand is emerging from workflow automation."}]},
        chapter_evidence_packages=_chapter_packages(),
    )

    preflight = preflight_topic_bundle(load_topic_bundle("AI Agent report"), query="AI Agent report")

    assert preflight["status"] == "usable"
    assert preflight["can_skip_search"] is True


def test_fake_or_title_only_bundle_is_polluted(tmp_path, monkeypatch):
    monkeypatch.setenv("TOPIC_BUNDLE_CACHE_PATH", str(tmp_path))
    package = _base_evidence_package()
    package["source_registry"] = [{"ref": "S1", "title": "Official", "source_level": "A"}]

    store_topic_bundle(
        query="polluted topic",
        evidence_package=package,
        structured_analysis={},
        chapter_evidence_packages=_chapter_packages(),
    )

    preflight = preflight_topic_bundle(load_topic_bundle("polluted topic"), query="polluted topic")

    assert preflight["status"] == "polluted"
    assert "title_only_source" in preflight["reasons"] or "fake_or_placeholder_source" in preflight["reasons"]


def test_count_only_chapter_package_is_partial_not_usable(tmp_path, monkeypatch):
    monkeypatch.setenv("TOPIC_BUNDLE_CACHE_PATH", str(tmp_path))
    monkeypatch.setenv("TOPIC_BUNDLE_CACHE_REQUIRE_HYDRATED_EVIDENCE", "true")

    store_topic_bundle(
        query="count only topic",
        evidence_package=_base_evidence_package(),
        structured_analysis={},
        chapter_evidence_packages=[{"chapter_id": "ch_01", "evidence_counts": {"core_evidence": 2}}],
    )

    preflight = preflight_topic_bundle(load_topic_bundle("count only topic"), query="count only topic")

    assert preflight["status"] == "partial"
    assert preflight["can_skip_search"] is False
    assert preflight["reason"] == "chapter_evidence_not_hydrated"


def test_analysis_rebuild_required_skips_old_analysis_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("TOPIC_BUNDLE_CACHE_PATH", str(tmp_path))
    monkeypatch.setenv("TOPIC_BUNDLE_CACHE_REUSE_ANALYSIS", "false")

    store_topic_bundle(
        query="rewrite topic",
        evidence_package=_base_evidence_package(),
        structured_analysis={"analysis_depth_quality": {"status": "needs_rewrite"}, "claim_units": [{"claim": "bad"}]},
        chapter_evidence_packages=_chapter_packages(),
    )

    loaded = load_topic_bundle("rewrite topic")
    preflight = preflight_topic_bundle(loaded, query="rewrite topic")
    inputs = bundle_to_writer_inputs(loaded, preflight=preflight)

    assert preflight["status"] == "partial"
    assert preflight["analysis_rebuild_required"] is True
    assert inputs["structured_analysis"] == {}


def test_compacted_summary_bundle_is_not_reusable(tmp_path, monkeypatch):
    monkeypatch.setenv("TOPIC_BUNDLE_CACHE_PATH", str(tmp_path))
    compact_package = {
        "payload_mode": "summary",
        "analysis_ready_evidence": {"count": 10, "sample": [{"evidence_id": "ev1"}]},
        "normalized_evidence": {"count": 10, "sample": [{"evidence_id": "ev1"}]},
        "source_registry": _base_evidence_package()["source_registry"],
    }

    stored = store_topic_bundle(
        query="summary topic",
        evidence_package=compact_package,
        structured_analysis={},
        chapter_evidence_packages=_chapter_packages(),
    )
    preflight = preflight_topic_bundle(load_topic_bundle("summary topic"), query="summary topic")

    assert is_compacted_evidence_package(compact_package) is True
    assert stored["stored"] is False
    assert stored["reason"] == "summary_only_not_reusable"
    assert preflight["status"] == "summary_only"
    assert preflight["seedable"] is False


def test_compacted_fallback_does_not_overwrite_full_bundle(tmp_path, monkeypatch):
    monkeypatch.setenv("TOPIC_BUNDLE_CACHE_PATH", str(tmp_path))
    store_topic_bundle(
        query="preserve topic",
        evidence_package=_base_evidence_package(),
        structured_analysis={"claim_units": [{"claim": "Demand is emerging from workflow automation."}]},
        chapter_evidence_packages=_chapter_packages(),
        stored_from="brain_full_payload",
    )
    compact_package = {
        "payload_mode": "summary",
        "analysis_ready_evidence": {"count": 10, "sample": [{"evidence_id": "evx"}]},
        "source_registry": _base_evidence_package()["source_registry"],
    }
    stored = store_topic_bundle(
        query="preserve topic",
        evidence_package=compact_package,
        structured_analysis={},
        chapter_evidence_packages=_chapter_packages(),
        stored_from="full_report_compacted_fallback",
    )
    loaded = load_topic_bundle("preserve topic")
    inputs = bundle_to_writer_inputs(loaded, preflight=preflight_topic_bundle(loaded, query="preserve topic"))

    assert stored["stored"] is False
    assert stored["existing_full_bundle_preserved"] is True
    assert inputs["seed_evidence_count"] == 1


def test_topic_seed_is_wrapped_as_merger_pool_item():
    from rag_pipeline.agents.brain_agent import _merge_topic_seed_with_live_evidence

    state = {
        "query": "AI Agent report",
        "topic_bundle_seed": {
            "topic_key": "topic-key",
            "path": "cache/path",
            "seed_evidence": [
                {
                    "evidence_id": "ev1",
                    "fact": "Enterprise AI agents are moving from pilots to workflow automation.",
                    "source_url": "https://www.nist.gov/ai/agents",
                    "source_title": "AI agents guidance",
                    "source_level": "B",
                }
            ],
        },
    }

    merged = _merge_topic_seed_with_live_evidence(state, [])

    assert len(merged) == 1
    assert merged[0]["agent"] == "topic_bundle_cache"
    assert merged[0]["status"] == "success"
    assert merged[0]["raw_data_points"][0]["evidence_origin"] == "topic_bundle_cache"
    assert merged[0]["raw_data_points"][0]["evidence"]
