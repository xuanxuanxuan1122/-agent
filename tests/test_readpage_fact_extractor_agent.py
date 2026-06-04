from __future__ import annotations

from rag_pipeline.agents import brain_agent, web_analysis_agent
from rag_pipeline.agents.readpage_fact_extractor_agent import (
    _source_ref,
    extract_fact_cards_from_pages,
    reset_budget,
    validate_extracted_fact_payload,
)
from rag_pipeline.flows.report.full_report import _readpage_fact_extractor_diagnostics
from rag_pipeline.flows.report.full_report import render_score_markdown


def test_readpage_extractor_rejects_navigation_and_keeps_body_fact(monkeypatch):
    def fake_llm(*, config, system_prompt, user_payload):
        return {
            "payload": {
                "fact_cards": [
                    {
                        "subject": "navigation",
                        "action_or_signal": "Skip to content Login Contact",
                        "variable": "navigation",
                        "distilled_fact": "Skip to content Login Contact",
                        "fact_type": "case",
                        "source_url": "https://www.salesforce.com/news/agentforce",
                        "source_ref": "S1",
                        "source_level": "B",
                        "source_verification_status": "readpage_verified",
                        "proof_role": "case",
                        "block_affinity": ["case_comparison"],
                        "claim_strength_hint": "directional",
                    },
                    {
                        "subject": "Salesforce Agentforce",
                        "action_or_signal": "disclosed customer-service workflow deployment",
                        "variable": "customer_case",
                        "time_or_scope": "2025",
                        "distilled_fact": "Salesforce disclosed Agentforce customer-service workflow deployments in 2025, showing that enterprise agents are being tested in support operations.",
                        "fact_type": "case",
                        "source_url": "https://www.salesforce.com/news/agentforce",
                        "source_ref": "S1",
                        "source_level": "B",
                        "source_verification_status": "readpage_verified",
                        "proof_role": "case",
                        "block_affinity": ["case_comparison", "customer_painpoint_matrix"],
                        "claim_strength_hint": "directional",
                    },
                ],
                "rejected_spans": [{"reason": "navigation_text", "text": "Skip to content"}],
            },
            "usage": {},
        }

    monkeypatch.setattr("rag_pipeline.agents.readpage_fact_extractor_agent.llm_config_is_ready", lambda config: True)
    monkeypatch.setattr("rag_pipeline.agents.readpage_fact_extractor_agent.call_openai_compatible_json", fake_llm)
    monkeypatch.setenv("READPAGE_FACT_EXTRACTOR_CACHE_ENABLED", "false")

    result = extract_fact_cards_from_pages(
        query="AI Agent industry report",
        page_results=[
            {
                "source_id": "S1",
                "title": "Agentforce customer story",
                "url": "https://www.salesforce.com/news/agentforce",
                "content": "Skip to content Login Contact. Salesforce disclosed Agentforce customer-service workflow deployments in 2025.",
            }
        ],
        search_task={"proof_role": "case", "chapter_id": "ch_02"},
    )

    assert result["attempted"] == 1
    assert result["success_count"] == 1
    assert result["rejected_span_count"] >= 1
    assert len(result["fact_cards"]) == 1
    card = result["fact_cards"][0]
    assert card["subject"] == "Salesforce Agentforce"
    assert card["proof_role"] == "case"
    assert "case_comparison" in card["block_affinity"]
    assert card["chapter_id"] == "ch_02"


def test_iqs_research_node_passes_search_options_to_fact_extractor(monkeypatch):
    captured = {}

    monkeypatch.setattr(web_analysis_agent, "iqs_api_key_is_configured", lambda: True)
    monkeypatch.setattr(
        web_analysis_agent,
        "run_iqs_optimized_search",
        lambda query, options: {
            "results": [{"title": "Official AI Agent report", "url": "https://example.gov/ai-agent"}],
            "errors": [],
            "query_plan": [],
            "search_tasks": [],
            "search_trace": [],
            "quality_processing": {},
        },
    )
    monkeypatch.setattr(
        web_analysis_agent,
        "select_auto_readpage_urls",
        lambda search_results, explicit_urls, search_options: ["https://example.gov/ai-agent"],
    )
    monkeypatch.setattr(
        web_analysis_agent,
        "call_iqs_readpage",
        lambda url, timeout_ms: {
            "url": url,
            "title": "Official AI Agent report",
            "content": "Official report states enterprise AI agents are being deployed in workflow automation.",
        },
    )

    def fake_extract(*, query, page_results, search_task):
        captured["search_task"] = search_task
        return {"fact_cards": [], "attempted": len(page_results), "success_count": 0}

    monkeypatch.setattr(web_analysis_agent, "extract_fact_cards_from_pages", fake_extract)

    state = {
        "query": "AI Agent official data",
        "urls": [],
        "search_options": {"search_task": {"proof_role": "metric", "chapter_id": "ch_01"}},
    }

    result = web_analysis_agent.iqs_research_node(state)

    assert "errors" not in result
    assert captured["search_task"] == {"proof_role": "metric", "chapter_id": "ch_01"}


def test_process_web_results_uses_query_when_search_task_lacks_query():
    results, meta = web_analysis_agent.process_web_results(
        "OpenAI 2025",
        [
            {
                "title": "OpenAI announces 2025 enterprise AI agent updates",
                "url": "https://openai.com/index/agent-updates",
                "snippet": "OpenAI shared 2025 updates for enterprise AI agents and workflow automation.",
            }
        ],
        options={"search_task": {"proof_role": "official_data", "chapter_id": "ch_smoke"}},
    )

    assert len(results) == 1
    assert meta["task_filtered_count"] == 0
    assert results[0]["task_filter"]["reason"] == "task_relevance_pass"


def test_readpage_extractor_rejects_incomplete_metric_and_internal_claim():
    payload = {
        "fact_cards": [
            {
                "subject": "AI Agent adoption",
                "action_or_signal": "reached",
                "variable": "adoption",
                "value": "50",
                "unit": "%",
                "distilled_fact": "AI Agent adoption: 50%",
                "fact_type": "metric",
                "source_url": "https://www.salesforce.com/news/adoption",
                "source_ref": "S1",
                "source_level": "B",
                "source_verification_status": "readpage_verified",
                "proof_role": "metric",
                "block_affinity": ["metric_reconciliation"],
            },
            {
                "subject": "internal",
                "action_or_signal": "needs repair",
                "variable": "diagnostic",
                "time_or_scope": "2025",
                "distilled_fact": "Insufficient evidence; this chapter needs more evidence before drafting.",
                "fact_type": "case",
                "source_url": "https://www.salesforce.com/news/diagnostic",
                "source_ref": "S2",
                "source_level": "B",
                "source_verification_status": "readpage_verified",
                "proof_role": "case",
                "block_affinity": ["case_comparison"],
            },
        ]
    }

    result = validate_extracted_fact_payload(payload, source_url="https://www.salesforce.com/news/adoption", source_ref="S1", proof_role="metric")

    assert result["fact_cards"] == []
    reasons = {item["reason"] for item in result["rejected_spans"]}
    assert "metric_missing_scope_or_period" in reasons
    assert "internal_or_claim_like_text" in reasons


def test_brain_normalize_prefers_extracted_fact_cards_over_regex_fallback():
    web_state = {
        "answer_text": "銆愪簨瀹炪€慉I Agent adoption: 50% [0]",
        "raw_output": {
            "search_options": {"search_task": {"task_id": "t1", "proof_role": "case", "chapter_id": "ch_02"}},
            "search_results": [{"source_id": 0, "title": "Search result", "url": "https://example.org/search", "snippet": "AI Agent adoption: 50%"}],
            "page_results": [{"source_id": "S1", "title": "Agentforce", "url": "https://example.org/agentforce", "content": "body"}],
            "extracted_fact_cards": [
                {
                    "evidence_id": "RFC-S1-1",
                    "ref": "RFC-S1-1",
                    "source_ref": "S1",
                    "source_url": "https://example.org/agentforce",
                    "source_title": "Agentforce",
                    "source_level": "B",
                    "source_verification_status": "readpage_verified",
                    "fact": "Salesforce disclosed Agentforce customer-service workflow deployments in 2025.",
                    "clean_fact": "Salesforce disclosed Agentforce customer-service workflow deployments in 2025.",
                    "distilled_fact": "Salesforce disclosed Agentforce customer-service workflow deployments in 2025.",
                    "proof_role": "case",
                    "fact_type": "case",
                    "block_affinity": ["case_comparison"],
                    "public_fact_card": {"subject": "Salesforce Agentforce", "distilled_fact": "Salesforce disclosed Agentforce customer-service workflow deployments in 2025."},
                }
            ],
            "synthesis": {"source": "llm", "structured_payload": {"answer": {"evidence": "AI Agent adoption: 50% [0]"}}, "confidence": 0.7},
            "fact_extractor": {"attempted": 1, "success_count": 1, "fact_card_count": 1},
        },
        "metadata": {},
    }

    child = brain_agent.normalize_web_child_output(web_state, route="web", errors=[])

    assert len(child["raw_data_points"]) == 1
    assert child["raw_data_points"][0]["evidence_origin"] == "readpage_fact_extractor"
    assert child["raw_data_points"][0]["fact_type"] == "case"
    assert child["limitations"]["fact_extractor"]["regex_fallback_used"] is False


def test_score_report_exposes_readpage_fact_extractor_diagnostics():
    markdown = render_score_markdown(
        query="AI Agent report",
        writer_report={"quality_score": 62, "report_status": "formal_scored"},
        writer_package={
            "fact_extractor": {
                "attempted": 2,
                "success_count": 1,
                "fact_card_count": 3,
                "rejected_span_count": 4,
                "invalid_metric_count": 1,
                "cache_hit_count": 1,
                "llm_error_count": 0,
                "regex_fallback_used": False,
                "fallback_used": False,
                "status": "success",
                "model": "deepseek-v4-pro",
            }
        },
        final_audit_result={},
        reformatter_result={},
    )

    assert "## Readpage Fact Extractor" in markdown
    assert "readpage_fact_extractor_attempted: 2" in markdown
    assert "fact_card_count: 3" in markdown
    assert "invalid_metric_count: 1" in markdown


def test_source_ref_preserves_zero_source_id():
    assert _source_ref({"source_id": 0, "source_ref": "OLD"}, fallback="fallback") == "0"


def test_cached_fact_card_runtime_context_is_overwritten():
    payload = {
        "fact_cards": [
            {
                "subject": "Salesforce Agentforce",
                "action_or_signal": "disclosed customer-service workflow deployment",
                "variable": "customer_case",
                "time_or_scope": "2025 enterprise support workflow",
                "distilled_fact": "Salesforce disclosed Agentforce customer-service workflow deployments in 2025.",
                "fact_type": "case",
                "source_url": "https://old.example.com/page",
                "source_ref": "OLD-SRC",
                "source_level": "B",
                "source_verification_status": "readpage_verified",
                "proof_role": "old_role",
                "chapter_id": "old_chapter",
                "dimension_id": "old_dimension",
                "task_id": "old_task",
                "evidence_goal": "old_goal",
                "block_affinity": ["case_comparison"],
            }
        ]
    }

    result = validate_extracted_fact_payload(
        payload,
        source_url="https://www.salesforce.com/news/agentforce",
        source_ref="0",
        source_level="B",
        verification_status="readpage_verified",
        proof_role="case",
        chapter_id="ch_02",
        search_task={"task_id": "t-current", "dimension_id": "dim-current", "evidence_goal": "case_goal"},
    )

    assert len(result["fact_cards"]) == 1
    card = result["fact_cards"][0]
    assert card["source_url"] == "https://www.salesforce.com/news/agentforce"
    assert card["source_ref"] == "0"
    assert card["proof_role"] == "case"
    assert card["chapter_id"] == "ch_02"
    assert card["dimension_id"] == "dim-current"
    assert card["task_id"] == "t-current"
    assert card["evidence_goal"] == "case_goal"
    assert card["cached_context"]["source_ref"] == "OLD-SRC"
    assert card["cached_context"]["chapter_id"] == "old_chapter"


def test_score_extractor_diagnostics_ignores_chapter_fact_card_counts():
    diagnostics = _readpage_fact_extractor_diagnostics(
        {
            "fact_extractor": {"attempted": 1, "success_count": 1, "fact_card_count": 2},
            "chapter_evidence_packages": [
                {"chapter_id": "ch_01", "fact_card_count": 99},
                {"chapter_id": "ch_02", "evidence_counts": {"fact_card_count": 42}},
            ],
        }
    )

    assert diagnostics["attempted"] == 1
    assert diagnostics["fact_card_count"] == 2


def test_brain_regex_fallback_flag_requires_actual_raw_points():
    web_state = {
        "answer_text": "",
        "raw_output": {
            "search_options": {"search_task": {"task_id": "t1", "proof_role": "case", "chapter_id": "ch_02"}},
            "search_results": [{"source_id": 0, "title": "Search result", "url": "https://example.org/search", "snippet": "No structured fact"}],
            "synthesis": {"source": "llm", "structured_payload": {"answer": {"evidence": ""}}, "confidence": 0.7},
            "fact_extractor": {"attempted": 1, "success_count": 0, "fact_card_count": 0},
        },
        "metadata": {},
    }

    child = brain_agent.normalize_web_child_output(web_state, route="web", errors=[])

    assert child["raw_data_points"] == []
    assert child["limitations"]["fact_extractor"]["regex_fallback_used"] is False
    assert child["limitations"]["fact_extractor"]["regex_fallback_point_count"] == 0
    assert child["limitations"]["fact_extractor"]["extractor_empty_without_regex_points"] is True


def test_brain_regex_fallback_records_actual_point_count():
    web_state = {
        "answer_text": "AI Agent customer deployments expanded in 2025 [0]",
        "raw_output": {
            "search_options": {"search_task": {"task_id": "t1", "proof_role": "case", "chapter_id": "ch_02"}},
            "search_results": [{"source_id": 0, "title": "Agent deployment", "url": "https://example.org/agent", "snippet": "AI Agent customer deployments expanded"}],
            "synthesis": {
                "source": "llm",
                "structured_payload": {"answer": {"evidence": "AI Agent customer deployments expanded in 2025 [0]"}},
                "confidence": 0.7,
            },
            "fact_extractor": {"attempted": 1, "success_count": 0, "fact_card_count": 0},
        },
        "metadata": {},
    }

    child = brain_agent.normalize_web_child_output(web_state, route="web", errors=[])

    assert len(child["raw_data_points"]) == 1
    assert child["limitations"]["fact_extractor"]["regex_fallback_used"] is True
    assert child["limitations"]["fact_extractor"]["regex_fallback_point_count"] == 1
    assert child["limitations"]["fact_extractor"]["extractor_empty_without_regex_points"] is False


def test_report_level_budget_limits_llm_extractor_calls(monkeypatch):
    calls = []

    def fake_llm(*, config, system_prompt, user_payload):
        calls.append(user_payload["source"]["url"])
        return {
            "payload": {
                "fact_cards": [
                    {
                        "subject": "Salesforce Agentforce",
                        "action_or_signal": "reported enterprise agent deployment",
                        "variable": "customer_case",
                        "time_or_scope": "2025",
                        "distilled_fact": "Salesforce reported enterprise agent deployment in 2025.",
                        "fact_type": "case",
                        "source_url": user_payload["source"]["url"],
                        "source_ref": "S",
                        "source_level": "B",
                        "source_verification_status": "readpage_verified",
                        "proof_role": "case",
                        "block_affinity": ["case_comparison"],
                    }
                ]
            },
            "usage": {},
        }

    monkeypatch.setattr("rag_pipeline.agents.readpage_fact_extractor_agent.llm_config_is_ready", lambda config: True)
    monkeypatch.setattr("rag_pipeline.agents.readpage_fact_extractor_agent.call_openai_compatible_json", fake_llm)
    monkeypatch.setenv("READPAGE_FACT_EXTRACTOR_CACHE_ENABLED", "false")
    monkeypatch.setenv("READPAGE_FACT_EXTRACTOR_MAX_CALLS_PER_REPORT", "2")
    monkeypatch.setenv("REPORT_STAGE_SNAPSHOT_RUN_ID", "budget-test-readpage-extractor")

    pages = [
        {
            "source_id": f"S{index}",
            "title": f"Agentforce page {index}",
            "url": f"https://www.salesforce.com/news/agentforce-{index}",
            "content": "Salesforce reported enterprise agent deployment in 2025 for support workflows.",
        }
        for index in range(5)
    ]

    result = extract_fact_cards_from_pages(query="AI Agent", page_results=pages, search_task={"proof_role": "case", "chapter_id": "ch_02"})

    assert len(calls) == 2
    assert result["attempted"] == 4
    assert result["budget_limit"] == 2
    assert result["budget_used"] == 2
    assert result["budget_exhausted"] is True


def test_budget_used_is_per_call_delta_and_resets(monkeypatch):
    calls = []

    def fake_llm(*, config, system_prompt, user_payload):
        calls.append(user_payload["source"]["url"])
        return {
            "payload": {
                "fact_cards": [
                    {
                        "subject": "Salesforce Agentforce",
                        "action_or_signal": "reported enterprise agent deployment",
                        "variable": "customer_case",
                        "time_or_scope": "2025",
                        "distilled_fact": "Salesforce reported enterprise agent deployment in 2025.",
                        "fact_type": "case",
                        "source_url": user_payload["source"]["url"],
                        "source_ref": "S",
                        "source_level": "B",
                        "source_verification_status": "readpage_verified",
                        "proof_role": "case",
                        "block_affinity": ["case_comparison"],
                    }
                ]
            },
            "usage": {},
        }

    monkeypatch.setattr("rag_pipeline.agents.readpage_fact_extractor_agent.llm_config_is_ready", lambda config: True)
    monkeypatch.setattr("rag_pipeline.agents.readpage_fact_extractor_agent.call_openai_compatible_json", fake_llm)
    monkeypatch.setenv("READPAGE_FACT_EXTRACTOR_CACHE_ENABLED", "false")
    monkeypatch.setenv("READPAGE_FACT_EXTRACTOR_MAX_CALLS_PER_REPORT", "4")
    monkeypatch.setenv("REPORT_STAGE_SNAPSHOT_RUN_ID", "budget-delta-test")
    reset_budget("budget-delta-test")

    def pages(offset):
        return [
            {
                "source_id": f"S{offset}-{index}",
                "title": f"Agentforce page {offset}-{index}",
                "url": f"https://www.salesforce.com/news/agentforce-{offset}-{index}",
                "content": "Salesforce reported enterprise agent deployment in 2025 for support workflows.",
            }
            for index in range(2)
        ]

    first = extract_fact_cards_from_pages(query="AI Agent", page_results=pages(1), search_task={"proof_role": "case", "chapter_id": "ch_02"})
    second = extract_fact_cards_from_pages(query="AI Agent", page_results=pages(2), search_task={"proof_role": "case", "chapter_id": "ch_02"})

    assert first["budget_used"] == 2
    assert second["budget_used"] == 2
    assert len(calls) == 4

    reset_budget("budget-delta-test")
    third = extract_fact_cards_from_pages(query="AI Agent", page_results=pages(3)[:1], search_task={"proof_role": "case", "chapter_id": "ch_02"})

    assert third["budget_used"] == 1
    assert third["budget_exhausted"] is False


def test_brain_fact_extractor_diagnostics_survive_to_score_paths():
    web_state = {
        "answer_text": "AI Agent customer deployments expanded in 2025 [0]",
        "raw_output": {
            "search_options": {"search_task": {"task_id": "t1", "proof_role": "case", "chapter_id": "ch_02"}},
            "search_results": [{"source_id": 0, "title": "Agent deployment", "url": "https://example.org/agent", "snippet": "AI Agent customer deployments expanded"}],
            "synthesis": {
                "source": "llm",
                "structured_payload": {"answer": {"evidence": "AI Agent customer deployments expanded in 2025 [0]"}},
                "confidence": 0.7,
            },
            "fact_extractor": {
                "attempted": 2,
                "success_count": 1,
                "fact_card_count": 3,
                "regex_fallback_point_count": 1,
                "budget_used": 2,
                "budget_limit": 40,
                "status": "success",
                "model": "deepseek-v4-pro",
            },
        },
        "metadata": {"readpage_fact_extractor": {"attempted": 2, "success_count": 1, "fact_card_count": 3, "budget_used": 2}},
    }
    child = brain_agent.normalize_web_child_output(web_state, route="web", errors=[])
    diagnostics = brain_agent._aggregate_readpage_fact_extractor_diagnostics({"web_analysis_agent": child})
    raw_diagnostics = brain_agent._aggregate_readpage_fact_extractor_diagnostics({}, extra_payloads=[web_state])
    evidence_package = {"metadata": {}}
    writer_report = {"render_artifacts": {"metadata": {}}}

    brain_agent._attach_readpage_fact_extractor_diagnostics(evidence_package, writer_report, diagnostics)
    score_diag = _readpage_fact_extractor_diagnostics({"evidence_package": evidence_package, "writer_report": writer_report})

    assert diagnostics["attempted"] == 2
    assert diagnostics["fact_card_count"] == 3
    assert diagnostics["budget_used"] == 2
    assert raw_diagnostics["attempted"] == 2
    assert raw_diagnostics["fact_card_count"] == 3
    assert score_diag["fact_card_count"] == 3
    assert score_diag["budget_used"] == 2
