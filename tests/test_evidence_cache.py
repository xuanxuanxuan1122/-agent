from __future__ import annotations

from rag_pipeline.cache.evidence_cache import lookup_evidence, lookup_search, store_evidence_from_package, store_search
from rag_pipeline.agents import brain_agent as brain_agent_module


def test_persistent_search_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("EVIDENCE_CACHE_PATH", str(tmp_path / "evidence_cache.sqlite"))
    monkeypatch.setenv("EVIDENCE_CACHE_SQLITE_JOURNAL_MODE", "MEMORY")
    monkeypatch.setenv("IQS_SEARCH_CACHE_ENABLED", "true")
    query = "foldable hinge official source"
    options = {"engineType": "Deep", "timeRange": "NoLimit", "contents": "mainText"}
    task = {"proof_role": "source_check"}
    payload = {
        "query": query,
        "results": [{"title": "Official filing", "url": "https://www.sec.gov/filing", "snippet": "hinge"}],
        "search_trace": [{"primary_engine": "Deep"}],
        "errors": [],
    }

    store_search(query, options, task, payload)
    cached = lookup_search(query, options, task)

    assert cached["cache"]["hit"] is True
    assert cached["cache"]["layer"] == "search_cache"
    assert cached["results"][0]["url"] == "https://www.sec.gov/filing"


def test_persistent_search_cache_preserves_hydrated_page_and_fact_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("EVIDENCE_CACHE_PATH", str(tmp_path / "evidence_cache.sqlite"))
    monkeypatch.setenv("EVIDENCE_CACHE_SQLITE_JOURNAL_MODE", "MEMORY")
    monkeypatch.setenv("IQS_SEARCH_CACHE_ENABLED", "true")
    query = "AI Agent official adoption metric"
    options = {"engineType": "Deep", "timeRange": "NoLimit", "contents": "mainText"}
    task = {"proof_role": "metric", "chapter_id": "ch_01"}
    payload = {
        "query": query,
        "results": [{"title": "Official report", "url": "https://example.gov/ai-agent"}],
        "page_results": [{"title": "Official report", "url": "https://example.gov/ai-agent", "content": "body"}],
        "extracted_fact_cards": [{"fact_id": "FC-1", "fact": "Adoption metric", "source_url": "https://example.gov/ai-agent"}],
        "fact_extractor": {"attempted": 1, "success_count": 1},
        "search_trace": [],
        "errors": [],
        "cache": {"hydrated": True},
    }

    store_search(query, options, task, payload)
    cached = lookup_search(query, options, task)

    assert cached["cache"]["hit"] is True
    assert cached["cache"]["layer"] == "search_cache"
    assert cached["page_results"][0]["url"] == "https://example.gov/ai-agent"
    assert cached["extracted_fact_cards"][0]["fact_id"] == "FC-1"
    assert cached["fact_extractor"]["success_count"] == 1


def test_negative_search_cache_is_query_specific(tmp_path, monkeypatch):
    monkeypatch.setenv("EVIDENCE_CACHE_PATH", str(tmp_path / "evidence_cache.sqlite"))
    monkeypatch.setenv("EVIDENCE_CACHE_SQLITE_JOURNAL_MODE", "MEMORY")
    monkeypatch.setenv("IQS_SEARCH_CACHE_ENABLED", "true")
    options = {"engineType": "LiteAdvanced", "timeRange": "NoLimit", "contents": "mainText"}
    task = {"proof_role": "source_check"}

    store_search("empty query", options, task, {"query": "empty query", "results": [], "search_trace": [], "errors": []})

    assert lookup_search("empty query", options, task)["cache"]["negative"] is True
    assert lookup_search("different query", options, task) is None


def test_search_cache_key_includes_request_size(tmp_path, monkeypatch):
    monkeypatch.setenv("EVIDENCE_CACHE_PATH", str(tmp_path / "evidence_cache.sqlite"))
    monkeypatch.setenv("EVIDENCE_CACHE_SQLITE_JOURNAL_MODE", "MEMORY")
    monkeypatch.setenv("IQS_SEARCH_CACHE_ENABLED", "true")
    query = "foldable iphone hinge yield"
    task = {"proof_role": "metric", "lane_targets": ["official_data"]}
    small_options = {
        "engineType": "LiteAdvanced",
        "timeRange": "OneWeek",
        "contents": "mainText",
        "phase": "followup",
        "numResults": 1,
        "maxSearchTasks": 4,
        "search_task": task,
    }
    large_options = dict(small_options)
    large_options["numResults"] = 10

    store_search(
        query,
        small_options,
        task,
        {"query": query, "results": [{"title": "A", "url": "https://www.sec.gov/a"}], "search_trace": [], "errors": []},
    )

    assert lookup_search(query, small_options, task)
    assert lookup_search(query, large_options, task) is None


def test_search_cache_respects_iqs_cache_flag(tmp_path, monkeypatch):
    monkeypatch.setenv("EVIDENCE_CACHE_PATH", str(tmp_path / "evidence_cache.sqlite"))
    monkeypatch.setenv("EVIDENCE_CACHE_SQLITE_JOURNAL_MODE", "MEMORY")
    monkeypatch.setenv("IQS_SEARCH_CACHE_ENABLED", "true")
    query = "cache disable probe"
    options = {"engineType": "LiteAdvanced", "timeRange": "NoLimit", "contents": "mainText"}
    store_search(query, options, {}, {"query": query, "results": [{"title": "x", "url": "https://www.sec.gov"}], "errors": []})

    monkeypatch.setenv("IQS_SEARCH_CACHE_ENABLED", "false")

    assert lookup_search(query, options, {}) is None


def test_negative_cache_skips_hard_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("EVIDENCE_CACHE_PATH", str(tmp_path / "evidence_cache.sqlite"))
    monkeypatch.setenv("EVIDENCE_CACHE_SQLITE_JOURNAL_MODE", "MEMORY")
    monkeypatch.setenv("IQS_SEARCH_CACHE_ENABLED", "true")
    options = {"engineType": "LiteAdvanced", "timeRange": "NoLimit", "contents": "mainText"}

    store_search("soft empty", options, {}, {"query": "soft empty", "results": [], "search_trace": [], "errors": ["primary result count insufficient"]})
    store_search("timeout empty", options, {}, {"query": "timeout empty", "results": [], "search_trace": [], "errors": ["request timeout"]})

    assert lookup_search("soft empty", options, {})["cache"]["negative"] is True
    assert lookup_search("timeout empty", options, {}) is None


def test_evidence_cache_satisfies_ab_repair_task(tmp_path, monkeypatch):
    monkeypatch.setenv("EVIDENCE_CACHE_PATH", str(tmp_path / "evidence_cache.sqlite"))
    monkeypatch.setenv("EVIDENCE_CACHE_SQLITE_JOURNAL_MODE", "MEMORY")
    query = "foldable hinge official source"
    package = {
        "analysis_ready_evidence": [
            {
                "fact": "Apple foldable hinge validation requires high cycle reliability and supplier qualification.",
                "metric": "cycle reliability",
                "value": "200000 times",
                "period": "2025",
                "numeric_unit": "count",
                "source_level": "A",
                "proof_role": "source_check",
                "confidence": 0.82,
                "source": {
                    "title": "Company filing",
                    "url": "https://www.sec.gov/filing",
                    "source_type": "financial_report",
                    "date": "2025",
                },
            }
        ]
    }

    summary = store_evidence_from_package(query=query, evidence_package=package, report_id="unit", run_id="run")
    hits = lookup_evidence(
        {"query": query, "proof_role": "source_check", "blocking_gaps": ["insufficient_ab_sources"]},
        min_source_level=["A", "B"],
        required_fields=["source"],
    )

    assert summary["stored_count"] == 1
    assert hits
    assert hits[0]["source_level"] == "A"


def test_evidence_cache_rejects_unsourced_ab_fact(tmp_path, monkeypatch):
    monkeypatch.setenv("EVIDENCE_CACHE_PATH", str(tmp_path / "evidence_cache.sqlite"))
    monkeypatch.setenv("EVIDENCE_CACHE_SQLITE_JOURNAL_MODE", "MEMORY")
    package = {
        "analysis_ready_evidence": [
            {
                "fact": "Apple foldable display yield remains a manufacturing bottleneck.",
                "source_level": "A",
                "proof_role": "source_check",
                "confidence": 0.9,
            }
        ]
    }

    summary = store_evidence_from_package(query="apple foldable yield", evidence_package=package)
    hits = lookup_evidence(
        {"query": "apple foldable yield", "proof_role": "source_check", "blocking_gaps": ["insufficient_ab_sources"]},
        min_source_level=["A", "B"],
        required_fields=["source"],
    )

    assert summary["stored_count"] == 0
    assert summary["skipped_count"] == 1
    assert hits == []


def test_metric_cache_hit_requires_complete_metric_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("EVIDENCE_CACHE_PATH", str(tmp_path / "evidence_cache.sqlite"))
    monkeypatch.setenv("EVIDENCE_CACHE_SQLITE_JOURNAL_MODE", "MEMORY")
    query = "foldable phone shipment metric"
    store_evidence_from_package(
        query=query,
        evidence_package={
            "analysis_ready_evidence": [
                {
                    "fact": "Foldable phone shipments increased.",
                    "metric": "shipments",
                    "value": "12",
                    "source_level": "B",
                    "proof_role": "metric",
                    "source": {"title": "Industry research", "url": "https://www.idc.com/report", "source_type": "research"},
                }
            ]
        },
    )

    hits = lookup_evidence(
        {"query": query, "proof_role": "metric", "blocking_gaps": ["metric_scope_period_unit_incomplete"]},
        min_source_level=["A", "B"],
        required_fields=["metric", "period", "unit", "source"],
    )

    assert hits == []


def test_evidence_cache_respects_lane_source_type(tmp_path, monkeypatch):
    monkeypatch.setenv("EVIDENCE_CACHE_PATH", str(tmp_path / "evidence_cache.sqlite"))
    monkeypatch.setenv("EVIDENCE_CACHE_SQLITE_JOURNAL_MODE", "MEMORY")
    query = "foldable hinge supplier qualification"
    store_evidence_from_package(
        query=query,
        evidence_package={
            "analysis_ready_evidence": [
                {
                    "fact": "Supplier qualification is discussed in an industry research note.",
                    "source_level": "B",
                    "proof_role": "source_check",
                    "source": {"title": "Industry research", "url": "https://www.idc.com/research", "source_type": "research"},
                }
            ]
        },
    )

    official_hits = lookup_evidence(
        {"query": query, "proof_role": "source_check", "lane_targets": ["official_data"]},
        min_source_level=["A", "B", "C"],
        required_fields=["source"],
    )
    research_hits = lookup_evidence(
        {"query": query, "proof_role": "source_check", "lane_targets": ["market_research"]},
        min_source_level=["A", "B", "C"],
        required_fields=["source"],
    )

    assert official_hits == []
    assert research_hits


def test_evidence_cache_rejects_generic_cross_topic_match(tmp_path, monkeypatch):
    monkeypatch.setenv("EVIDENCE_CACHE_PATH", str(tmp_path / "evidence_cache.sqlite"))
    monkeypatch.setenv("EVIDENCE_CACHE_SQLITE_JOURNAL_MODE", "MEMORY")
    store_evidence_from_package(
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
                        "url": "https://example.gov/ai-agent-data",
                        "source_type": "official",
                    },
                }
            ]
        },
    )

    generic_only_hits = lookup_evidence(
        {"query": "foldable hinge official data", "proof_role": "source_check", "lane_targets": ["official_data"]},
        min_source_level=["A", "B"],
        required_fields=["source"],
    )
    exact_topic_hits = lookup_evidence(
        {"query": "ai agent official data", "proof_role": "source_check", "lane_targets": ["official_data"]},
        min_source_level=["A", "B"],
        required_fields=["source"],
    )

    assert generic_only_hits == []
    assert exact_topic_hits


def test_brain_core_cache_hit_becomes_seed_and_keeps_live_task(tmp_path, monkeypatch):
    monkeypatch.setenv("EVIDENCE_CACHE_PATH", str(tmp_path / "evidence_cache.sqlite"))
    monkeypatch.setenv("EVIDENCE_CACHE_SQLITE_JOURNAL_MODE", "MEMORY")
    query = "foldable hinge official source"
    store_evidence_from_package(
        query=query,
        evidence_package={
            "analysis_ready_evidence": [
                {
                    "fact": "Foldable hinge supplier qualification was disclosed in a company filing.",
                    "source_level": "A",
                    "proof_role": "source_check",
                    "confidence": 0.8,
                    "source": {"title": "Company filing", "url": "https://www.sec.gov/filing", "source_type": "financial_report"},
                }
            ]
        },
    )
    search_task = brain_agent_module.normalize_search_task(
        {
            "query": query,
            "agent": "iqs",
            "proof_role": "source_check",
            "blocking_gaps": ["insufficient_ab_sources"],
        },
        fallback_index=1,
    )

    cache_results, remaining, cache_only_skipped = brain_agent_module._apply_evidence_cache_to_followup_tasks(
        [{"query": query, "agent": "iqs", "targets_gap": "H1", "search_task": search_task}],
        state={"metadata": {}},
        round_number=1,
    )

    assert cache_results
    assert remaining
    assert not cache_only_skipped
    assert cache_results[0]["status"] == "success"
    assert cache_results[0]["cache_seed"] is True
    assert cache_results[0]["live_refresh_required"] is True
    assert cache_results[0]["cache"]["layer"] == "evidence_cache"
    assert remaining[0]["search_task"]["cache_seed_available"] is True
    assert remaining[0]["search_task"]["live_refresh_required"] is True


def test_brain_non_core_cache_hit_skips_network(tmp_path, monkeypatch):
    monkeypatch.setenv("EVIDENCE_CACHE_PATH", str(tmp_path / "evidence_cache.sqlite"))
    monkeypatch.setenv("EVIDENCE_CACHE_SQLITE_JOURNAL_MODE", "MEMORY")
    query = "foldable hinge supporting context"
    store_evidence_from_package(
        query=query,
        evidence_package={
            "analysis_ready_evidence": [
                {
                    "fact": "Foldable hinge supporting context was summarized by an industry source.",
                    "source_level": "B",
                    "proof_role": "support",
                    "confidence": 0.7,
                    "source": {"title": "Industry research", "url": "https://www.idc.com/support", "source_type": "research"},
                }
            ]
        },
    )
    search_task = brain_agent_module.normalize_search_task(
        {
            "query": query,
            "agent": "iqs",
            "proof_role": "support",
            "lane_targets": ["market_research"],
        },
        fallback_index=1,
    )

    cache_results, remaining, cache_only_skipped = brain_agent_module._apply_evidence_cache_to_followup_tasks(
        [{"query": query, "agent": "iqs", "targets_gap": "support", "search_task": search_task}],
        state={"metadata": {}},
        round_number=1,
    )

    assert cache_results
    assert not remaining
    assert cache_only_skipped
    assert cache_results[0]["cache_seed"] is True
    assert cache_results[0]["live_refresh_required"] is False


def test_brain_cache_hit_preserves_repair_lineage_and_gap_summary(tmp_path, monkeypatch):
    monkeypatch.setenv("EVIDENCE_CACHE_PATH", str(tmp_path / "evidence_cache.sqlite"))
    monkeypatch.setenv("EVIDENCE_CACHE_SQLITE_JOURNAL_MODE", "MEMORY")
    query = "enterprise ai agent adoption metric official source"
    store_evidence_from_package(
        query=query,
        evidence_package={
            "analysis_ready_evidence": [
                {
                    "fact": "Official data reports enterprise AI agent adoption reached 42% in 2025.",
                    "metric": "enterprise AI agent adoption",
                    "value": "42",
                    "unit": "%",
                    "period": "2025",
                    "source_level": "A",
                    "proof_role": "metric",
                    "confidence": 0.82,
                    "source": {"title": "Official data", "url": "https://www.sec.gov/ai-agent-data", "source_type": "official_data"},
                }
            ]
        },
    )
    search_task = brain_agent_module.normalize_search_task(
        {
            "query": query,
            "agent": "iqs",
            "gap_id": "GAP-metric",
            "requirement_id": "H1_metric",
            "proof_role": "metric",
            "required_fields": ["metric", "value", "unit", "period", "source"],
            "lane_targets": ["official_data"],
        },
        fallback_index=1,
    )
    state = {"metadata": {}}

    cache_results, remaining, cache_only_skipped = brain_agent_module._apply_evidence_cache_to_followup_tasks(
        [{"query": query, "agent": "iqs", "targets_gap": "metric", "search_task": search_task}],
        state=state,
        round_number=2,
    )

    assert cache_results
    result = cache_results[0]
    assert result["gap_id"] == "GAP-metric"
    assert result["requirement_id"] == "H1_metric"
    assert result["proof_role"] == "metric"
    assert result["required_fields"] == ["metric", "value", "unit", "period", "source"]
    assert result["cache"]["gap_id"] == "GAP-metric"
    assert result["cache"]["requirement_id"] == "H1_metric"
    assert result["raw_data_points"][0]["gap_id"] == "GAP-metric"
    assert result["raw_data_points"][0]["requirement_id"] == "H1_metric"
    assert not remaining
    assert cache_only_skipped
    gap_summary = state["metadata"]["evidence_cache_summary"]["by_gap"]["GAP-metric"]
    assert gap_summary["requirement_id"] == "H1_metric"
    assert gap_summary["cache_hit_count"] == 1
    assert gap_summary["cache_only_skip_count"] == 1
    assert gap_summary["live_refresh_required_count"] == 0


def test_brain_core_ab_cache_hit_requires_source(tmp_path, monkeypatch):
    monkeypatch.setenv("EVIDENCE_CACHE_PATH", str(tmp_path / "evidence_cache.sqlite"))
    monkeypatch.setenv("EVIDENCE_CACHE_SQLITE_JOURNAL_MODE", "MEMORY")
    query = "foldable hinge official source"
    search_task = brain_agent_module.normalize_search_task(
        {
            "query": query,
            "agent": "iqs",
            "proof_role": "source_check",
            "blocking_gaps": ["insufficient_ab_sources"],
        },
        fallback_index=1,
    )

    assert "source" in brain_agent_module._required_cache_fields_for_task(search_task)
