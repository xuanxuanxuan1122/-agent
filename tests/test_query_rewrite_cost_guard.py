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


def test_llm_query_rewrite_preserves_topic_anchor(monkeypatch):
    _ready_query_rewrite(monkeypatch)
    captured = {}
    search_task = {
        "task_id": "task-low-altitude",
        "requirement_id": "H1_metric",
        "gap_id": "GAP-low-altitude",
        "query": "中国低空经济 市场规模 official",
        "proof_role": "metric",
        "required_fields": ["metric", "value", "unit", "period", "source"],
        "topic_anchor_terms": ["中国低空经济"],
        "topic_anchor_status": "ok",
        "must_have_terms": ["中国低空经济", "市场规模"],
    }

    def fake_call(**kwargs):
        captured["payload"] = kwargs["user_payload"]
        return {"payload": {"queries": [{"text": "市场规模 official statistics 2026", "intent": "data"}]}}

    monkeypatch.setattr(web, "call_openai_compatible_json", fake_call)

    plan = web.build_llm_query_plan(
        "中国低空经济商业化",
        {"run_id": "test-run", "search_task": search_task},
        search_task=search_task,
    )

    assert captured["payload"]["search_task"]["topic_anchor_terms"] == ["中国低空经济"]
    assert plan
    assert "低空经济" in plan[0]["text"]
    assert plan[0]["topic_anchor_terms"] == ["中国低空经济"]


def test_task_acceptance_filter_rejects_missing_topic_anchor():
    search_task = {
        "task_id": "task-low-altitude",
        "query": "中国低空经济 市场规模 official",
        "must_have_terms": ["市场规模"],
        "topic_anchor_terms": ["中国低空经济"],
        "source_priority": ["official"],
    }
    off_topic = {
        "title": "中国供应链市场规模统计报告",
        "snippet": "官方统计显示供应链数字化市场规模持续增长。",
        "url": "https://example.org/report",
    }
    on_topic = {
        "title": "中国低空经济市场规模统计报告",
        "snippet": "报告披露低空经济应用场景和市场规模数据。",
        "url": "https://example.org/report",
    }

    rejected = web.task_acceptance_filter(off_topic, {"search_task": search_task})
    accepted = web.task_acceptance_filter(on_topic, {"search_task": search_task})

    assert rejected["accepted"] is False
    assert rejected["reason"] == "topic_anchor_missing"
    assert accepted["accepted"] is True


def test_task_acceptance_filter_keeps_indirect_numeric_context_from_traceable_source():
    search_task = {
        "task_id": "task-agent-demand",
        "query": "AI Agent 企业落地 市场空间",
        "must_have_terms": ["市场空间"],
        "topic_anchor_terms": ["AI Agent", "智能体"],
        "source_priority": ["gov.cn"],
    }
    item = {
        "title": "官方统计年鉴披露软件和信息服务业收入",
        "snippet": "2025年软件和信息服务业收入同比增长，为企业数字化投入提供背景口径。",
        "summary": "该指标不能直接证明 AI Agent 需求，但可以作为数字化预算和行业景气度的背景材料。",
        "url": "https://www.gov.cn/example/statistics",
        "credibility_score": 0.8,
    }

    result = web.task_acceptance_filter(item, {"search_task": search_task})

    assert result["accepted"] is True
    assert result["role_hint"] == "metric_context_fact"
    assert result["reason"] == "topic_anchor_missing_context_fact"
    assert result["context_only"] is True


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
