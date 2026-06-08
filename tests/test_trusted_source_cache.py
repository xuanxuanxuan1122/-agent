from __future__ import annotations

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
                        "title": "Official AI Agent Statistics",
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
