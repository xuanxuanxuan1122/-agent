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


def test_trusted_source_cache_requires_topic_anchor_match_for_repair_seed(tmp_path, monkeypatch):
    monkeypatch.setenv("TRUSTED_SOURCE_CACHE_ENABLED", "true")
    monkeypatch.setenv("TRUSTED_SOURCE_CACHE_PATH", str(tmp_path / "trusted_sources.jsonl"))
    store_trusted_sources_from_package(
        query="\u4e2d\u56fd\u4eba\u5f62\u673a\u5668\u4eba\u4ea7\u4e1a \u5b98\u65b9",
        evidence_package={
            "analysis_ready_evidence": [
                {
                    "fact": "\u4eba\u5f62\u673a\u5668\u4eba\u4ea7\u4e1a\u94fe\u4f01\u4e1a\u53c2\u4e0e\u4ea7\u4e1a\u5927\u4f1a\u548c\u5c55\u793a\u6d3b\u52a8\u3002",
                    "source_level": "A",
                    "proof_role": "source_check",
                    "confidence": 0.82,
                    "source": {
                        "title": "\u4e2d\u56fd\u4eba\u5f62\u673a\u5668\u4eba\u4ea7\u4e1a\u52a8\u6001",
                        "url": "https://www.ncsti.gov.cn/robot",
                        "source_type": "official",
                    },
                }
            ]
        },
    )
    store_trusted_sources_from_package(
        query="\u4e2d\u56fd\u4f4e\u7a7a\u7ecf\u6d4e\u4ea7\u4e1a\u94fe \u5b98\u65b9",
        evidence_package={
            "analysis_ready_evidence": [
                {
                    "fact": "\u4f4e\u7a7a\u7ecf\u6d4e\u4ea7\u4e1a\u94fe\u5728\u8fdb\u535a\u4f1a\u51fa\u73b0\u8ba2\u5355\u548c\u573a\u666f\u843d\u5730\u4fe1\u53f7\u3002",
                    "source_level": "A",
                    "proof_role": "source_check",
                    "confidence": 0.82,
                    "source": {
                        "title": "\u4f4e\u7a7a\u7ecf\u6d4e\u5728\u8fdb\u535a\u4f1a\u8fce\u6765\u9ad8\u5149\u65f6\u523b",
                        "url": "https://www.henan.gov.cn/low-altitude",
                        "source_type": "official",
                    },
                }
            ]
        },
    )

    hits = lookup_trusted_sources(
        {
            "query": "\u4e2d\u56fd\u4f4e\u7a7a\u7ecf\u6d4e\u4ea7\u4e1a\u94fe \u5b98\u65b9 \u6570\u636e",
            "proof_role": "source_check",
            "topic_anchor_terms": [
                "\u4e2d\u56fd\u4f4e\u7a7a\u7ecf\u6d4e\u4ea7\u4e1a\u94fe",
                "\u4f4e\u7a7a\u7ecf\u6d4e",
            ],
            "research_object": "\u4e2d\u56fd\u4f4e\u7a7a\u7ecf\u6d4e\u4ea7\u4e1a\u94fe",
        },
        min_source_level=["A", "B"],
        required_fields=["source"],
    )

    assert hits
    assert all("\u4eba\u5f62\u673a\u5668\u4eba" not in str(hit.get("fact_description") or "") for hit in hits)
    assert any("\u4f4e\u7a7a\u7ecf\u6d4e" in str(hit.get("fact_description") or "") for hit in hits)


def test_trusted_source_cache_derives_topic_anchor_from_repair_query(tmp_path, monkeypatch):
    monkeypatch.setenv("TRUSTED_SOURCE_CACHE_ENABLED", "true")
    monkeypatch.setenv("TRUSTED_SOURCE_CACHE_PATH", str(tmp_path / "trusted_sources.jsonl"))
    store_trusted_sources_from_package(
        query="\u4e2d\u56fd\u4eba\u5f62\u673a\u5668\u4eba\u4ea7\u4e1a \u5b98\u65b9",
        evidence_package={
            "analysis_ready_evidence": [
                {
                    "fact": "\u4eba\u5f62\u673a\u5668\u4eba\u4ea7\u4e1a\u94fe\u4f01\u4e1a\u53c2\u4e0e\u4ea7\u4e1a\u5927\u4f1a\u548c\u5c55\u793a\u6d3b\u52a8\u3002",
                    "source_level": "A",
                    "proof_role": "source_check",
                    "source": {
                        "title": "\u4e2d\u56fd\u4eba\u5f62\u673a\u5668\u4eba\u4ea7\u4e1a\u52a8\u6001",
                        "url": "https://www.ncsti.gov.cn/robot",
                        "source_type": "official",
                    },
                }
            ]
        },
    )

    hits = lookup_trusted_sources(
        {
            "query": "\u4e2d\u56fd\u4f4e\u7a7a\u7ecf\u6d4e\u4ea7\u4e1a\u94fe\u662f\u5426\u5b58\u5728\u771f\u5b9e\u9700\u6c42 \u5b98\u65b9 \u7edf\u8ba1",
            "proof_role": "source_check",
        },
        min_source_level=["A", "B"],
        required_fields=["source"],
    )

    assert hits == []


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
