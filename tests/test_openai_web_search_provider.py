from __future__ import annotations

from rag_pipeline.agents import brain_agent as brain_agent_module
from rag_pipeline.agents.evidence_merger import merge_evidence_package
from rag_pipeline.agents.openai_web_search_provider import (
    CHILD_AGENT_NAME,
    PROVIDER_NAME,
    build_openai_web_request_payload,
    normalize_openai_web_response,
)


def _mock_response():
    return {
        "model": "gpt-5.5",
        "output": [
            {
                "type": "web_search_call",
                "action": {
                    "sources": [
                        {
                            "title": "Company annual report",
                            "url": "https://www.sec.gov/Archives/example-10k",
                            "snippet": "Revenue reached 12 billion dollars in 2025.",
                        }
                    ]
                },
            },
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": (
                            '{"summary":"Revenue evidence found.",'
                            '"evidence":[{"claim":"The company reported revenue of 12 billion dollars in 2025.",'
                            '"metric":"营收","value":"12 billion dollars","period":"2025",'
                            '"source_title":"Company annual report",'
                            '"source_url":"https://www.sec.gov/Archives/example-10k","confidence":0.82}]}'
                        ),
                        "annotations": [
                            {
                                "type": "url_citation",
                                "title": "Company annual report",
                                "url": "https://www.sec.gov/Archives/example-10k",
                            }
                        ],
                    }
                ],
            },
        ],
    }


def test_openai_web_response_normalizes_to_existing_evidence_contract():
    child = normalize_openai_web_response(
        _mock_response(),
        query="company revenue",
        search_task={"proof_role": "source_check", "source_priority": ["filing"]},
        targets_gap="insufficient_ab_sources",
        round_number=2,
    )

    assert child["status"] == "success"
    assert child["key_sources"][0]["provider"] == PROVIDER_NAME
    assert child["key_sources"][0]["source_level"] == "A"
    assert child["key_sources"][0]["source_type"] == "financial_report"
    assert child["raw_data_points"][0]["provider"] == PROVIDER_NAME
    assert child["raw_data_points"][0]["source_url"] == "https://www.sec.gov/Archives/example-10k"
    assert child["raw_data_points"][0]["metric"] == "营收"


def test_openai_web_request_uses_required_tool_choice_and_allowed_domains(monkeypatch):
    monkeypatch.setenv("OPENAI_WEB_SEARCH_ALLOWED_DOMAINS", "sec.gov, stats.gov.cn")
    monkeypatch.setenv("OPENAI_WEB_SEARCH_REASONING_EFFORT", "high")

    payload = build_openai_web_request_payload(
        "revenue filing",
        search_task={"proof_role": "source_check", "source_priority": ["filing"]},
    )

    assert payload["tool_choice"] == "required"
    assert payload["include"] == ["web_search_call.action.sources"]
    assert payload["tools"][0]["type"] == "web_search"
    assert payload["tools"][0]["filters"]["allowed_domains"] == ["sec.gov", "stats.gov.cn"]
    assert payload["reasoning"] == {"effort": "high"}


def test_openai_web_request_does_not_filter_generic_official_tasks(monkeypatch):
    monkeypatch.delenv("OPENAI_WEB_SEARCH_ALLOWED_DOMAINS", raising=False)
    monkeypatch.setenv("OPENAI_WEB_SEARCH_INFER_ALLOWED_DOMAINS", "true")

    payload = build_openai_web_request_payload(
        "AI Agent official source_check market evidence",
        search_task={"proof_role": "source_check", "source_priority": ["official", "filing"]},
    )

    assert "filters" not in payload["tools"][0]


def test_openai_web_request_infers_specific_target_domains(monkeypatch):
    monkeypatch.delenv("OPENAI_WEB_SEARCH_ALLOWED_DOMAINS", raising=False)
    monkeypatch.setenv("OPENAI_WEB_SEARCH_INFER_ALLOWED_DOMAINS", "true")

    payload = build_openai_web_request_payload(
        "NIST AI Risk Management Framework official publication page",
        search_task={"proof_role": "source_check", "source_priority": ["official"]},
    )

    assert payload["tools"][0]["filters"]["allowed_domains"] == ["nist.gov"]


def test_openai_gap_repair_tasks_are_added_only_when_enabled(monkeypatch):
    task = {
        "query": "company filing revenue",
        "agent": "iqs",
        "targets_gap": "insufficient_ab_sources",
        "search_task": {
            "query": "company filing revenue",
            "proof_role": "source_check",
            "blocking_gaps": ["insufficient_ab_sources"],
            "source_priority": ["official", "filing"],
        },
    }

    monkeypatch.setenv("OPENAI_WEB_SEARCH_ENABLED", "false")
    assert brain_agent_module._build_openai_web_gap_repair_tasks([task], state={"metadata": {}}, round_number=1) == []

    monkeypatch.setenv("OPENAI_WEB_SEARCH_ENABLED", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    state = {"metadata": {}}
    repaired = brain_agent_module._build_openai_web_gap_repair_tasks([task], state=state, round_number=1)

    assert len(repaired) == 1
    assert repaired[0]["agent"] == PROVIDER_NAME
    assert repaired[0]["search_task"]["provider"] == PROVIDER_NAME
    assert repaired[0]["search_task"]["retrieval_mode"] == "openai_repair"
    assert repaired[0]["search_task"]["repair_source"] == "openai_web_gap_repair"
    assert repaired[0]["search_task"]["primary_provider"] == PROVIDER_NAME
    assert state["metadata"]["openai_web_search_task_count"] == 1


def test_openai_gap_repair_uses_expanded_budget_and_diversifies_chapters(monkeypatch):
    monkeypatch.setenv("OPENAI_WEB_SEARCH_ENABLED", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("OPENAI_WEB_SEARCH_MAX_GAP_REPAIR_TASKS_PER_ROUND", raising=False)
    monkeypatch.delenv("OPENAI_WEB_SEARCH_MAX_TASKS_PER_REPORT", raising=False)
    tasks = []
    for index in range(5):
        chapter_id = f"ch_{index + 1:02d}"
        tasks.append(
            {
                "query": f"{chapter_id} official evidence",
                "agent": "iqs",
                "targets_gap": "insufficient_ab_sources",
                "search_task": {
                    "query": f"{chapter_id} official evidence",
                    "chapter_id": chapter_id,
                    "proof_role": "source_check",
                    "blocking_gaps": ["insufficient_ab_sources"],
                    "source_priority": ["official"],
                },
            }
        )
    state = {"metadata": {}}

    repaired = brain_agent_module._build_openai_web_gap_repair_tasks(tasks, state=state, round_number=1)

    assert len(repaired) == 4
    assert len({item["search_task"]["chapter_id"] for item in repaired}) == 4
    summary = state["metadata"]["openai_web_search_summary"]
    assert summary["max_per_round"] == 4
    assert summary["max_per_report"] == 20
    assert summary["hard_max_per_report"] == 24


def test_openai_gap_repair_stops_after_consecutive_failures(monkeypatch):
    monkeypatch.setenv("OPENAI_WEB_SEARCH_ENABLED", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    state = {"metadata": {"openai_web_search_summary": {"consecutive_failed_tasks": 2}}}
    task = {
        "query": "company filing revenue",
        "agent": "iqs",
        "targets_gap": "insufficient_ab_sources",
        "search_task": {
            "query": "company filing revenue",
            "proof_role": "source_check",
            "blocking_gaps": ["insufficient_ab_sources"],
            "source_priority": ["official", "filing"],
        },
    }

    repaired = brain_agent_module._build_openai_web_gap_repair_tasks([task], state=state, round_number=2)

    assert repaired == []
    assert state["metadata"]["openai_web_search_summary"]["disabled_after_consecutive_failures"] is True


def test_openai_gap_repair_counts_source_only_result_as_failed():
    state = {"metadata": {}}
    brain_agent_module._record_openai_web_gap_repair_result(
        state,
        {
            "status": "partial",
            "key_sources": [{"title": "Source", "url": "https://example.com"}],
            "raw_data_points": [],
            "limitations": {"failure_reason": "no_raw_evidence"},
        },
    )

    summary = state["metadata"]["openai_web_search_summary"]
    assert summary["failed_count"] == 1
    assert summary.get("success_count", 0) == 0


def test_direct_openai_followup_is_rerouted_and_not_executed(monkeypatch):
    monkeypatch.setenv("OPENAI_WEB_SEARCH_ENABLED", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("EVIDENCE_CACHE_ENABLED", "false")

    def fail_openai(**kwargs):
        raise AssertionError("direct openai_web follow-up should not call the provider")

    def fake_web_analysis(*args, **kwargs):
        return {"answer_text": "", "raw_output": {"search_results": [], "page_results": []}, "errors": []}

    monkeypatch.setattr(brain_agent_module, "run_openai_web_search_child", fail_openai)
    monkeypatch.setattr(brain_agent_module, "run_web_analysis_agent", fake_web_analysis)
    state = {"metadata": {}}

    results = brain_agent_module.run_followup_queries(
        follow_up_queries=[
            {
                "query": "direct openai task",
                "agent": PROVIDER_NAME,
                "targets_gap": "generic follow-up",
            }
        ],
        round_number=1,
        state=state,
    )

    assert results
    assert all(item["agent"] != PROVIDER_NAME for item in results)
    summary = state["metadata"]["openai_web_search_summary"]
    assert summary["invalid_direct_invocation_count"] == 1
    assert summary["last_invalid_direct_invocation_action"] == "rerouted_to_iqs"
    assert "openai_web_search_task_count" not in state["metadata"]


def test_invalid_openai_contract_is_blocked_before_provider_call(monkeypatch):
    monkeypatch.setenv("OPENAI_WEB_SEARCH_ENABLED", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def fail_openai(**kwargs):
        raise AssertionError("invalid openai_web contract should not call the provider")

    monkeypatch.setattr(brain_agent_module, "run_openai_web_search_child", fail_openai)
    state = {"metadata": {}}

    result = brain_agent_module._run_single_followup(
        agent=PROVIDER_NAME,
        query="direct openai task",
        targets_gap="insufficient_ab_sources",
        round_number=1,
        state=state,
        search_task={"query": "direct openai task", "agent": PROVIDER_NAME},
    )

    assert result["status"] == "failed"
    assert result["limitations"]["failure_reason"] == "openai_web_invalid_direct_invocation"
    summary = state["metadata"]["openai_web_search_summary"]
    assert summary["invalid_direct_invocation_count"] == 1
    assert summary["failed_count"] == 1


def test_openai_web_evidence_enters_analysis_ready_package(monkeypatch, tmp_path):
    monkeypatch.setenv("EVIDENCE_CACHE_ENABLED", "false")
    monkeypatch.setenv("EVIDENCE_CACHE_PATH", str(tmp_path / "evidence_cache.sqlite"))
    child = normalize_openai_web_response(
        _mock_response(),
        query="company revenue",
        search_task={"proof_role": "source_check", "source_priority": ["filing"]},
        targets_gap="insufficient_ab_sources",
        round_number=1,
    )
    pool_item = {
        "round": 1,
        "agent": PROVIDER_NAME,
        "child_agent": CHILD_AGENT_NAME,
        "query": "company revenue",
        "targets_gap": "insufficient_ab_sources",
        "status": child["status"],
        "confidence": child["confidence"],
        "answer": child["answer"],
        "key_sources": child["key_sources"],
        "raw_data_points": child["raw_data_points"],
        "limitations": child["limitations"],
    }

    package = merge_evidence_package(
        original_query="company revenue",
        evidence_pool=[pool_item],
        research_plan={"query": "company revenue", "dimensions": [{"dimension_name": "财务数据"}]},
    )

    assert package["analysis_ready_evidence"]
    assert "core_ab_by_chapter" in package["summary"]
    assert "source_family_distribution" in package["summary"]
    assert "publishable_evidence_gate" in package["summary"]
    item = package["analysis_ready_evidence"][0]
    assert item["source_level"] == "A"
    assert item["source"]["url"] == "https://www.sec.gov/Archives/example-10k"


def test_openai_summary_only_response_does_not_create_raw_evidence():
    response = {
        "model": "gpt-5.5",
        "output": [
            {
                "type": "web_search_call",
                "action": {
                    "sources": [
                        {
                            "title": "Market source",
                            "url": "https://example.com/market",
                            "snippet": "Market reached 10 billion dollars.",
                        }
                    ]
                },
            },
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": '{"summary":"The market reached 10 billion dollars in 2025."}',
                    }
                ],
            },
        ],
    }

    child = normalize_openai_web_response(
        response,
        query="market size",
        search_task={"retrieval_mode": "openai_repair", "repair_source": "openai_web_gap_repair"},
        targets_gap="insufficient_ab_sources",
        round_number=1,
    )

    assert child["status"] == "failed"
    assert child["key_sources"]
    assert child["raw_data_points"] == []
    assert child["used"] is False


def test_openai_web_does_not_upgrade_self_media_or_aggregators_to_ab_sources():
    response = {
        "model": "gpt-5.5",
        "output": [
            {
                "type": "web_search_call",
                "action": {
                    "sources": [
                        {
                            "title": "Industry report repost",
                            "url": "https://caifuhao.eastmoney.com/news/202601011200000000",
                            "snippet": "A reposted market-size claim.",
                            "source_type": "research",
                        },
                        {
                            "title": "Tencent news aggregation",
                            "url": "https://view.inews.qq.com/a/20260101A00000",
                            "snippet": "A news aggregation item.",
                            "source_type": "research",
                        },
                    ]
                },
            }
        ],
    }

    child = normalize_openai_web_response(
        response,
        query="market size",
        search_task={"proof_role": "source_check", "source_priority": ["research"]},
        targets_gap="insufficient_ab_sources",
        round_number=1,
    )

    levels = {source["url"]: source["source_level"] for source in child["key_sources"]}
    assert levels["https://caifuhao.eastmoney.com/news/202601011200000000"] == "D"
    assert levels["https://view.inews.qq.com/a/20260101A00000"] == "C"


def test_iqs_retrieval_routing_modes_by_lane(monkeypatch):
    monkeypatch.setenv("BRAIN_ENABLE_RETRIEVAL_MODE_ROUTING", "true")

    official = brain_agent_module._task_for_lane(
        {"query": "AI market statistics annual report", "proof_role": "metric"},
        "official_data",
        "iqs_lane_1",
    )
    news = brain_agent_module._task_for_lane(
        {"query": "latest AI Agent price news", "proof_role": "support"},
        "news_event",
        "iqs_lane_4",
    )
    policy_news = brain_agent_module._task_for_lane(
        {"query": "AI regulation official announcement original policy", "proof_role": "source_check"},
        "news_event",
        "iqs_lane_4",
    )
    customer_case = brain_agent_module._task_for_lane(
        {"query": "customer implementation tender case", "proof_role": "case"},
        "customer_case",
        "iqs_lane_6",
    )

    assert official["retrieval_mode"] == "deep"
    assert official["primary_provider"] == "iqs_deep"
    assert official["prefer_deep"] is True
    assert news["retrieval_mode"] == "normal"
    assert news["primary_provider"] == "iqs_normal"
    assert news["prefer_deep"] is False
    assert policy_news["retrieval_mode"] == "hybrid"
    assert policy_news["primary_provider"] == "iqs_deep"
    assert customer_case["retrieval_mode"] == "hybrid"


def test_openai_title_only_sources_are_rejected():
    response = {
        "model": "gpt-5.5",
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": (
                            '{"summary":"Title-only source should not count.",'
                            '"sources":[{"title":"Only a title"}],'
                            '"evidence":[{"claim":"The market reached 10 billion dollars in 2025.",'
                            '"source_title":"Only a title","confidence":0.9}]}'
                        ),
                    }
                ],
            }
        ],
    }

    child = normalize_openai_web_response(
        response,
        query="market size",
        search_task={"retrieval_mode": "openai_repair", "repair_source": "openai_web_gap_repair"},
        targets_gap="insufficient_ab_sources",
        round_number=1,
    )

    assert child["status"] == "failed"
    assert child["key_sources"] == []
    assert child["raw_data_points"] == []


def test_retrieval_strategy_summary_records_lane_and_openai_counts():
    deep_task = brain_agent_module._task_for_lane(
        {"query": "official statistics", "proof_role": "metric"},
        "official_data",
        "iqs_lane_1",
    )
    normal_task = brain_agent_module._task_for_lane(
        {"query": "latest event news", "proof_role": "support"},
        "news_event",
        "iqs_lane_4",
    )
    state = {
        "search_task_schedule": {"scheduled_tasks": [deep_task, normal_task]},
        "metadata": {
            "openai_web_search_summary": {
                "gap_repair_task_count": 1,
                "failed_count": 1,
                "consecutive_failed_tasks": 1,
            }
        },
    }

    package = brain_agent_module._annotate_evidence_package_runtime(
        {"summary": {"source_level_distribution": {"A": 1, "B": 1}}},
        lane_coverage={
            "iqs_lane_1": {"scheduled": 1, "succeeded": 1, "ab_source_count": 2},
            "iqs_lane_4": {"scheduled": 1, "succeeded": 0, "timed_out_task_count": 1, "ab_source_count": 0},
        },
        state=state,
    )
    summary = package["metadata"]["retrieval_strategy_summary"]

    assert summary["scheduled_by_mode"]["deep"] == 1
    assert summary["scheduled_by_mode"]["normal"] == 1
    assert summary["scheduled_by_mode"]["openai_repair"] == 1
    assert summary["totals"]["timed_out_task_count"] == 1
    assert summary["totals"]["ab_source_count"] >= 2
    assert summary["openai_web_repair_summary"]["failed_count"] == 1
