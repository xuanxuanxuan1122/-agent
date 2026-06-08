import json

from rag_pipeline.agents import web_analysis_agent as web


def _ready_query_rewrite(monkeypatch):
    monkeypatch.setenv("IQS_ENABLE_LLM_QUERY_REWRITE", "true")
    monkeypatch.setenv("QUERY_REWRITE_MAX_CALLS_PER_REPORT", "2")
    monkeypatch.setenv("QUERY_REWRITE_MAX_INPUT_CHARS", "6000")
    monkeypatch.setenv("QUERY_REWRITE_CACHE_ENABLED", "true")
    monkeypatch.setattr(web, "llm_config_is_ready", lambda config: True)
    monkeypatch.setattr(web, "build_llm_config", lambda task_name="query_rewrite": {"model": "mock-query-rewrite"})
    if hasattr(web, "reset_query_rewrite_budget"):
        web.reset_query_rewrite_budget("test-run")


def test_llm_query_rewrite_disabled_does_not_call_model(monkeypatch):
    monkeypatch.setenv("IQS_ENABLE_LLM_QUERY_REWRITE", "false")
    called = {"count": 0}

    def fake_call(**kwargs):
        called["count"] += 1
        return {"payload": {"queries": []}}

    monkeypatch.setattr(web, "call_openai_compatible_json", fake_call)

    plan = web.build_llm_query_plan("AI Agent market", {"run_id": "disabled-run"})

    assert plan == []
    assert called["count"] == 0


def test_llm_query_rewrite_uses_compact_payload_not_full_research_plan(monkeypatch):
    _ready_query_rewrite(monkeypatch)
    captured = {}
    huge_research_plan = {
        "search_tasks": [{"task_id": f"task-{i}", "evidence_goal": "x" * 500} for i in range(200)],
        "large_internal_notes": "y" * 80_000,
    }
    search_task = {
        "task_id": "task-1",
        "requirement_id": "H1_metric",
        "gap_id": "GAP-metric",
        "proof_role": "metric",
        "required_fields": ["metric", "value", "unit", "period", "source"],
        "required_source_level": ["A", "B"],
        "lane_targets": ["official_data", "market_research"],
        "success_criteria": "Only count repaired when metric/value/unit/period/source are all present.",
        "reject_if": ["snippet_only", "no_date", "no_source_url"],
        "freshness_required": True,
        "evidence_goal": "find adoption metrics",
        "must_have_terms": ["AI Agent", "adoption"],
        "forbidden_terms": ["招聘"],
        "source_priority": ["official", "report"],
    }

    def fake_call(**kwargs):
        captured["payload"] = kwargs["user_payload"]
        return {
            "payload": {
                "queries": [
                    {
                        "text": "AI Agent adoption official report",
                        "intent": "data",
                        "requirement_id": "H1_metric",
                        "gap_id": "GAP-metric",
                        "proof_role": "metric",
                        "required_fields": ["metric", "value", "unit", "period", "source"],
                        "source_priority": ["official_data"],
                        "must_have_terms": ["AI Agent"],
                    }
                ]
            }
        }

    monkeypatch.setattr(web, "call_openai_compatible_json", fake_call)

    plan = web.build_llm_query_plan(
        "AI Agent market",
        {"run_id": "test-run", "research_plan": huge_research_plan, "search_task": search_task},
        research_plan=huge_research_plan,
        search_task=search_task,
    )

    payload = captured["payload"]
    serialized = json.dumps(payload, ensure_ascii=False)
    assert plan
    assert "research_plan" not in payload
    assert "large_internal_notes" not in serialized
    assert len(serialized) <= 6000
    assert payload["search_task"]["task_id"] == "task-1"
    assert payload["search_task"]["requirement_id"] == "H1_metric"
    assert payload["search_task"]["gap_id"] == "GAP-metric"
    assert payload["search_task"]["proof_role"] == "metric"
    assert payload["search_task"]["required_fields"] == ["metric", "value", "unit", "period", "source"]
    assert payload["search_task"]["required_source_level"] == ["A", "B"]
    assert payload["search_task"]["success_criteria"].startswith("Only count")
    assert "snippet_only" in payload["search_task"]["reject_if"]
    assert payload["search_task"]["freshness_required"] is True
    assert plan[0]["requirement_id"] == "H1_metric"
    assert plan[0]["gap_id"] == "GAP-metric"
    assert plan[0]["required_fields"] == ["metric", "value", "unit", "period", "source"]


def test_task_query_plan_metric_queries_use_contract_fields():
    plan = web.build_task_query_plan(
        "AI Agent market size",
        {
            "search_task": {
                "task_id": "task-metric",
                "requirement_id": "H1_metric",
                "gap_id": "GAP-metric",
                "query": "AI Agent adoption",
                "proof_role": "metric",
                "required_fields": ["metric", "value", "unit", "period", "source"],
                "required_source_level": ["A", "B"],
                "lane_targets": ["official_data", "market_research"],
                "success_criteria": "Only count repaired when metric/value/unit/period/source are all present.",
                "reject_if": ["snippet_only", "no_date", "no_source_url"],
                "source_strategy": {
                    "source_priority": ["official_data", "market_research", "annual_report"],
                    "query_enhancers": ["report", "survey", "pdf", "annual report"],
                },
                "required_field_focus": "period",
            }
        },
    )

    assert plan
    assert all(item["requirement_id"] == "H1_metric" for item in plan)
    assert all(item["gap_id"] == "GAP-metric" for item in plan)
    assert all(item["required_fields"] == ["metric", "value", "unit", "period", "source"] for item in plan)
    assert all(item["required_field_focus"] == "period" for item in plan)
    assert all(item["source_strategy"]["source_priority"][0] == "official_data" for item in plan)
    assert any(any(term in item["text"] for term in ("官方", "统计", "report", "annual report", "survey")) for item in plan)


def test_llm_query_rewrite_call_cap_and_cache(monkeypatch):
    _ready_query_rewrite(monkeypatch)
    calls = {"count": 0}

    def fake_call(**kwargs):
        calls["count"] += 1
        return {"payload": {"queries": [{"text": f"AI Agent query {calls['count']}", "intent": "data"}]}}

    monkeypatch.setattr(web, "call_openai_compatible_json", fake_call)
    base_options = {"run_id": "test-run", "search_task": {"task_id": "same-task", "proof_role": "case"}}

    first = web.build_llm_query_plan("AI Agent", base_options)
    second = web.build_llm_query_plan("AI Agent", base_options)
    third = web.build_llm_query_plan("AI Agent second", {"run_id": "test-run", "search_task": {"task_id": "task-2"}})
    fourth = web.build_llm_query_plan("AI Agent third", {"run_id": "test-run", "search_task": {"task_id": "task-3"}})

    assert first == second
    assert third
    assert fourth == []
    assert calls["count"] == 2
    assert web.query_rewrite_diagnostics("test-run")["query_rewrite_budget_exhausted"] is True
