from __future__ import annotations

import json

from rag_pipeline.cache.trusted_source_cache import lookup_trusted_sources, store_trusted_sources_from_package


def test_trusted_source_cache_rejects_generic_cross_topic_match(tmp_path, monkeypatch):
    monkeypatch.setenv("TRUSTED_SOURCE_CACHE_ENABLED", "true")
    monkeypatch.setenv("TRUSTED_SOURCE_CACHE_PATH", str(tmp_path / "trusted_sources.jsonl"))
    store_trusted_sources_from_package(
        query="ai agent official data",
        evidence_package={
            "analysis_ready_evidence": [
                {
                    "fact": "Official data reports enterprise AI agent adoption reached 42% in 2025.",
                    "metric": "enterprise AI agent adoption",
                    "value": "42",
                    "unit": "%",
                    "period": "2025",
                    "source_level": "A",
                    "proof_role": "source_check",
                    "confidence": 0.82,
                    "source": {
                        "title": "National enterprise AI adoption bulletin",
                        "url": "https://data.beijing.gov.cn/ai-agent-data",
                        "source_type": "official",
                    },
                }
            ]
        },
    )

    generic_only_hits = lookup_trusted_sources(
        {"query": "foldable hinge official data", "proof_role": "source_check"},
        min_source_level=["A", "B"],
        required_fields=["source"],
    )
    exact_topic_hits = lookup_trusted_sources(
        {"query": "ai agent official data", "proof_role": "source_check"},
        min_source_level=["A", "B"],
        required_fields=["source"],
    )

    assert generic_only_hits == []
    assert exact_topic_hits


def test_trusted_source_cache_rejects_placeholder_sources(tmp_path, monkeypatch):
    monkeypatch.setenv("TRUSTED_SOURCE_CACHE_ENABLED", "true")
    monkeypatch.setenv("TRUSTED_SOURCE_CACHE_PATH", str(tmp_path / "trusted_sources.jsonl"))

    summary = store_trusted_sources_from_package(
        query="ai agent official data",
        evidence_package={
            "analysis_ready_evidence": [
                {
                    "fact": "Official data shows AI agent adoption reached 50% in 2025.",
                    "metric": "AI agent adoption",
                    "value": "50",
                    "unit": "%",
                    "period": "2025",
                    "source_level": "A",
                    "proof_role": "metric",
                    "source": {
                        "title": "Official AI Agent Statistics",
                        "url": "https://example.gov/ai-agent-statistics",
                        "source_type": "official",
                    },
                }
            ]
        },
    )
    hits = lookup_trusted_sources(
        {"query": "ai agent official data", "proof_role": "metric"},
        min_source_level=["A", "B"],
        required_fields=["source"],
    )

    assert summary["stored_count"] == 0
    assert hits == []


def test_trusted_source_cache_rejects_official_statistics_placeholder_title(tmp_path, monkeypatch):
    monkeypatch.setenv("TRUSTED_SOURCE_CACHE_ENABLED", "true")
    monkeypatch.setenv("TRUSTED_SOURCE_CACHE_PATH", str(tmp_path / "trusted_sources.jsonl"))

    summary = store_trusted_sources_from_package(
        query="ai agent official data",
        evidence_package={
            "analysis_ready_evidence": [
                {
                    "fact": "Enterprise AI agent adoption continued to rise in 2025.",
                    "source_level": "A",
                    "proof_role": "source_check",
                    "source": {
                        "title": "Official AI Agent Statistics",
                        "url": "https://www.stats.gov.cn/ai-agent-statistics",
                        "source_type": "official",
                    },
                }
            ]
        },
    )

    assert summary["stored_count"] == 0


def test_trusted_source_cache_lookup_filters_legacy_placeholder_entries(tmp_path, monkeypatch):
    path = tmp_path / "trusted_sources.jsonl"
    monkeypatch.setenv("TRUSTED_SOURCE_CACHE_ENABLED", "true")
    monkeypatch.setenv("TRUSTED_SOURCE_CACHE_PATH", str(path))
    entry = {
        "id": "trusted:dirty",
        "topic_key": "ai agent official data",
        "topic_terms": ["ai", "agent", "official", "data"],
        "source_url": "https://example.gov/ai-agent-statistics",
        "source_domain": "example.gov",
        "title": "Official AI Agent Statistics",
        "publisher": "",
        "source_level": "A",
        "source_type": "official",
        "proof_role": "metric",
        "allowed_use": "core_claim",
        "traceability_status": "traceable",
        "fact_description": "Official data shows AI agent adoption reached 50% in 2025.",
        "metric_name": "AI agent adoption",
        "value": "50",
        "unit": "%",
        "period": "2025",
        "raw": {
            "source": {
                "title": "Official AI Agent Statistics",
                "url": "https://example.gov/ai-agent-statistics",
                "source_type": "official",
            }
        },
    }
    path.write_text(json.dumps(entry, ensure_ascii=False) + "\n", encoding="utf-8")

    hits = lookup_trusted_sources(
        {"query": "ai agent official data", "proof_role": "metric"},
        min_source_level=["A", "B"],
        required_fields=["source"],
    )

    assert hits == []
