from __future__ import annotations

import json

from rag_pipeline.agents import brain_agent, web_analysis_agent
from rag_pipeline.agents.readpage_fact_extractor_agent import (
    _split_page_into_chunks,
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


def test_readpage_fact_card_low_source_level_is_directional_not_rejected():
    from rag_pipeline.agents import readpage_fact_extractor_agent as agent

    card = {
        "distilled_fact": "垂直媒体报道称，2025年人形机器人商业化订单开始增多。",
        "fact_type": "case",
        "source_url": "https://www.cs.com.cn/news/humanoid",
        "source_ref": "S1",
        "source_level": "C",
        "source_verification_status": "readpage_verified",
        "proof_role": "case",
    }
    task = {"required_source_level": ["A", "B"], "proof_role": "case"}

    normalized, rejected = agent._validated_card(
        card,
        source_url="https://www.cs.com.cn/news/humanoid",
        source_ref="S1",
        source_level="C",
        verification_status="readpage_verified",
        proof_role="case",
        search_task=task,
    )

    assert rejected == []
    assert normalized is not None
    assert normalized["allowed_use"] == "directional_signal"
    assert normalized["claim_strength_hint"] == "directional"
    assert normalized["source_level_gap"] == {"required": ["A", "B"], "actual": "C"}


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


def test_iqs_research_node_reuses_hydrated_search_cache_without_readpage_or_extraction(monkeypatch):
    monkeypatch.setattr(web_analysis_agent, "iqs_api_key_is_configured", lambda: True)
    monkeypatch.setattr(
        web_analysis_agent,
        "run_iqs_optimized_search",
        lambda query, options: {
            "results": [{"title": "Official AI Agent report", "url": "https://example.gov/ai-agent"}],
            "page_results": [
                {
                    "url": "https://example.gov/ai-agent",
                    "title": "Official AI Agent report",
                    "content": "Official report states enterprise AI agents are being deployed.",
                    "source_id": 1,
                }
            ],
            "extracted_fact_cards": [
                {
                    "fact_id": "FC-1",
                    "fact": "Enterprise AI agents are being deployed.",
                    "source_url": "https://example.gov/ai-agent",
                    "source_verification_status": "readpage_verified",
                }
            ],
            "fact_extractor": {"attempted": 1, "success_count": 1, "status": "success"},
            "errors": [],
            "query_plan": [],
            "search_tasks": [],
            "search_trace": [],
            "quality_processing": {},
            "cache": {"hit": True, "layer": "search_cache", "hydrated": True},
        },
    )
    monkeypatch.setattr(
        web_analysis_agent,
        "select_auto_readpage_urls",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("readpage URL selection should be skipped")),
    )
    monkeypatch.setattr(
        web_analysis_agent,
        "call_iqs_readpage",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("readpage should be skipped")),
    )
    monkeypatch.setattr(
        web_analysis_agent,
        "extract_fact_cards_from_pages",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("fact extraction should be skipped")),
    )

    result = web_analysis_agent.iqs_research_node(
        {
            "query": "AI Agent official data",
            "urls": [],
            "search_options": {"search_task": {"proof_role": "metric", "chapter_id": "ch_01"}},
        }
    )

    assert result["page_results"][0]["url"] == "https://example.gov/ai-agent"
    assert result["extracted_fact_cards"][0]["fact_id"] == "FC-1"
    assert result["metadata"]["readpage_fact_extractor"]["cache_reused"] is True
    assert result["metadata"]["auto_readpage"]["skipped_by_hydrated_cache"] is True


def test_iqs_research_node_writes_hydrated_search_cache_after_page_fact_extraction(monkeypatch):
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
            "content": "Official report states enterprise AI agents are being deployed.",
        },
    )
    monkeypatch.setattr(
        web_analysis_agent,
        "extract_fact_cards_from_pages",
        lambda *, query, page_results, search_task: {
            "fact_cards": [
                {
                    "fact_id": "FC-1",
                    "fact": "Enterprise AI agents are being deployed.",
                    "source_url": "https://example.gov/ai-agent",
                    "source_verification_status": "readpage_verified",
                }
            ],
            "attempted": len(page_results),
            "success_count": len(page_results),
            "status": "success",
        },
    )

    def fake_store_search(query, search_options, search_task, payload):
        captured["query"] = query
        captured["search_options"] = search_options
        captured["search_task"] = search_task
        captured["payload"] = payload

    monkeypatch.setattr(web_analysis_agent, "store_persistent_search_cache", fake_store_search)

    result = web_analysis_agent.iqs_research_node(
        {
            "query": "AI Agent official data",
            "urls": [],
            "search_options": {"search_task": {"proof_role": "metric", "chapter_id": "ch_01"}},
        }
    )

    assert result["extracted_fact_cards"][0]["fact_id"] == "FC-1"
    assert captured["search_task"] == {"proof_role": "metric", "chapter_id": "ch_01"}
    assert captured["payload"]["page_results"][0]["url"] == "https://example.gov/ai-agent"
    assert captured["payload"]["extracted_fact_cards"][0]["fact_id"] == "FC-1"
    assert captured["payload"]["fact_extractor"]["success_count"] == 1
    assert captured["payload"]["cache"]["hydrated"] is True


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


def test_readpage_extractor_user_payload_carries_prompt_contract_v2(monkeypatch):
    captured = {}

    def fake_llm(*, config, system_prompt, user_payload):
        captured["system_prompt"] = system_prompt
        captured["user_payload"] = user_payload
        return {
            "payload": {
                "fact_cards": [
                    {
                        "subject": "AI Agent adoption",
                        "action_or_signal": "reported enterprise adoption rate",
                        "variable": "enterprise adoption rate",
                        "value": "50",
                        "unit": "%",
                        "time_or_scope": "2025 enterprise survey",
                        "distilled_fact": "The verified page reports a 50% enterprise AI Agent adoption rate in its 2025 survey.",
                        "fact_type": "metric",
                        "source_url": "https://www.salesforce.com/news/agent-adoption",
                        "source_ref": "S1",
                        "source_level": "B",
                        "source_verification_status": "readpage_verified",
                        "proof_role": "metric",
                        "block_affinity": ["metric_reconciliation"],
                    }
                ]
            },
            "usage": {},
        }

    monkeypatch.setattr("rag_pipeline.agents.readpage_fact_extractor_agent.llm_config_is_ready", lambda config: True)
    monkeypatch.setattr("rag_pipeline.agents.readpage_fact_extractor_agent.call_openai_compatible_json", fake_llm)
    monkeypatch.setenv("READPAGE_FACT_EXTRACTOR_CACHE_ENABLED", "false")

    result = extract_fact_cards_from_pages(
        query="AI Agent adoption official metric",
        page_results=[
            {
                "source_id": "S1",
                "title": "Agent adoption survey",
                "url": "https://www.salesforce.com/news/agent-adoption",
                "content": "The verified page reports a 50% enterprise AI Agent adoption rate in its 2025 survey.",
            }
        ],
        search_task={
            "task_id": "ST-H1",
            "requirement_id": "H1_metric",
            "gap_id": "GAP-metric",
            "proof_role": "metric",
            "required_fields": ["metric", "value", "unit", "period", "source"],
            "required_source_level": ["A", "B"],
            "success_criteria": "只有 metric/value/unit/period/source 全部存在才算修复成功。",
            "reject_if": ["snippet_only", "no_date", "no_source_url"],
        },
    )

    payload = captured["user_payload"]
    assert payload["schema_version"] == "readpage_fact_card_v2"
    assert "只能使用输入的 page_text" in captured["system_prompt"]
    assert payload["search_task"]["requirement_id"] == "H1_metric"
    assert payload["search_task"]["gap_id"] == "GAP-metric"
    assert payload["search_task"]["required_fields"] == ["metric", "value", "unit", "period", "source"]
    assert payload["search_task"]["required_source_level"] == ["A", "B"]
    assert "no_source_url" in payload["search_task"]["reject_if"]
    card = result["fact_cards"][0]
    assert result["schema_version"] == "readpage_fact_card_v2"
    assert card["requirement_id"] == "H1_metric"
    assert card["gap_id"] == "GAP-metric"
    assert card["search_task_id"] == "ST-H1"
    assert card["allowed_use"] == "supporting"


def test_metric_fact_card_missing_required_contract_fields_is_rejected():
    payload = {
        "fact_cards": [
            {
                "subject": "AI Agent adoption",
                "action_or_signal": "reported adoption rate",
                "variable": "adoption",
                "value": "50",
                "distilled_fact": "The page reports AI Agent adoption at 50.",
                "fact_type": "metric",
                "proof_role": "metric",
                "source_level": "B",
                "source_verification_status": "readpage_verified",
                "block_affinity": ["metric_reconciliation"],
            }
        ]
    }

    result = validate_extracted_fact_payload(
        payload,
        proof_role="metric",
        search_task={
            "task_id": "ST-H1",
            "requirement_id": "H1_metric",
            "gap_id": "GAP-metric",
            "proof_role": "metric",
            "required_fields": ["metric", "value", "unit", "period", "source"],
        },
    )

    assert result["fact_cards"] == []
    rejected = result["rejected_spans"][0]
    assert rejected["reason"] == "metric_missing_scope_or_period"
    assert {"unit", "period", "source"} <= set(rejected["missing_fields"])


def test_metric_percent_value_can_supply_unit_field():
    payload = {
        "fact_cards": [
            {
                "subject": "AI Agent adoption",
                "action_or_signal": "reported adoption rate",
                "variable": "adoption",
                "metric": "adoption rate",
                "value": "50%",
                "unit": "",
                "period": "2025",
                "source_url": "https://www.salesforce.com/news/report",
                "distilled_fact": "The page reports AI Agent adoption at 50% in 2025.",
                "fact_type": "metric",
                "proof_role": "metric",
                "source_level": "B",
                "source_verification_status": "readpage_verified",
                "block_affinity": ["metric_reconciliation"],
            }
        ]
    }

    result = validate_extracted_fact_payload(
        payload,
        proof_role="metric",
        search_task={
            "task_id": "ST-H1",
            "requirement_id": "H1_metric",
            "gap_id": "GAP-metric",
            "proof_role": "metric",
            "required_fields": ["metric", "value", "unit", "period", "source"],
        },
    )

    assert len(result["fact_cards"]) == 1
    assert result["rejected_spans"] == []


def test_metric_fact_card_repairs_table_fields_from_distilled_fact():
    payload = {
        "fact_cards": [
            {
                "subject": "AI Agent market",
                "action_or_signal": "market size reached 120 billion yuan",
                "variable": "",
                "metric": "",
                "value": "",
                "unit": "",
                "period": "",
                "source_url": "https://www.salesforce.com/news/report",
                "distilled_fact": "The report says AI Agent market size reached 120 billion yuan in 2025.",
                "fact_type": "metric",
                "proof_role": "metric",
                "source_level": "B",
                "source_verification_status": "readpage_verified",
                "block_affinity": ["metric_reconciliation"],
            }
        ]
    }

    result = validate_extracted_fact_payload(
        payload,
        proof_role="metric",
        search_task={
            "task_id": "ST-H1",
            "requirement_id": "H1_metric",
            "gap_id": "GAP-metric",
            "proof_role": "metric",
            "required_fields": ["metric", "value", "unit", "period", "source"],
        },
    )

    assert len(result["fact_cards"]) == 1
    card = result["fact_cards"][0]
    assert card["metric"]
    assert card["value"] == "120"
    assert card["unit"] == "billion yuan"
    assert card["period"] == "2025"
    assert result["rejected_spans"] == []


def test_fallback_metric_sentence_yields_table_ready_fact_card(monkeypatch):
    monkeypatch.setattr("rag_pipeline.agents.readpage_fact_extractor_agent.llm_config_is_ready", lambda config: False)
    monkeypatch.setenv("READPAGE_FACT_EXTRACTOR_CACHE_ENABLED", "false")
    pages = [
        {
            "source_id": "S1",
            "title": "AI Agent market report",
            "url": "https://www.salesforce.com/news/report",
            "content": "The report says AI Agent market size reached 120 billion yuan in 2025.",
            "source_level": "B",
        }
    ]

    result = extract_fact_cards_from_pages(
        query="AI Agent market size metric",
        page_results=pages,
        search_task={
            "task_id": "ST-H1",
            "requirement_id": "H1_metric",
            "gap_id": "GAP-metric",
            "proof_role": "metric",
            "required_fields": ["metric", "value", "unit", "period", "source"],
        },
    )

    assert result["fallback_used"] is True
    assert len(result["fact_cards"]) == 1
    card = result["fact_cards"][0]
    assert card["fact_type"] == "metric"
    assert card["value"] == "120"
    assert card["unit"] == "billion yuan"
    assert card["period"] == "2025"


def test_fallback_skips_markdown_chrome_before_metric_sentence(monkeypatch):
    monkeypatch.setattr("rag_pipeline.agents.readpage_fact_extractor_agent.llm_config_is_ready", lambda config: False)
    monkeypatch.setenv("READPAGE_FACT_EXTRACTOR_CACHE_ENABLED", "false")
    pages = [
        {
            "source_id": "S1",
            "title": "Embodied AI market report",
            "url": "https://www.askci.com/news/chanye/example.shtml",
            "content": "\n".join(
                [
                    "Embodied AI market report ![](//img.example/share.png) [share](https://www.askci.com/share)",
                    "- [2025-2030 related report![](//img.example/hot.png)](https://www.askci.com/reports/example)",
                    "The research institute says China embodied AI market size reached 9150 billion yuan in 2025.",
                    "Analysts forecast China embodied AI market size will reach 10904 billion yuan in 2026.",
                ]
            ),
            "source_level": "B",
        }
    ]

    result = extract_fact_cards_from_pages(
        query="embodied AI market size metric",
        page_results=pages,
        search_task={
            "task_id": "ST-H1",
            "requirement_id": "H1_metric",
            "gap_id": "GAP-metric",
            "proof_role": "metric",
            "required_fields": ["metric", "value", "unit", "period", "source"],
        },
    )

    assert result["fallback_used"] is True
    assert len(result["fact_cards"]) >= 1
    facts = " ".join(str(item.get("distilled_fact") or "") for item in result["fact_cards"])
    assert "9150 billion yuan" in facts
    assert "share.png" not in facts


def test_fallback_rejects_browser_warning_and_link_only_chrome(monkeypatch):
    monkeypatch.setattr("rag_pipeline.agents.readpage_fact_extractor_agent.llm_config_is_ready", lambda config: False)
    monkeypatch.setenv("READPAGE_FACT_EXTRACTOR_CACHE_ENABLED", "false")

    result = extract_fact_cards_from_pages(
        query="embodied AI funding",
        page_results=[
            {
                "source_id": "S1",
                "title": "Browser warning",
                "url": "https://www.wap.cfi.cn/p20260319003907.html",
                "content": "\n".join(
                    [
                        "# Your connection is not private",
                        "Attackers might be trying to steal your information from www.wap.cfi.cn.",
                        "- [Finance](//finance.eastmoney.com/)",
                        "- [Focus](//finance.eastmoney.com/yaowen.html)",
                    ]
                ),
                "source_level": "C",
            }
        ],
        search_task={"task_id": "ST-risk", "requirement_id": "REQ-risk", "proof_role": "funding"},
    )

    assert result["fallback_used"] is True
    assert result["fact_cards"] == []


def test_readpage_rejected_spans_emit_repair_summary(monkeypatch):
    def fake_llm(*, config, system_prompt, user_payload):
        return {
            "payload": {
                "fact_cards": [
                    {
                        "subject": "AI Agent adoption",
                        "action_or_signal": "reported adoption",
                        "variable": "adoption",
                        "value": "50",
                        "distilled_fact": "The page reports AI Agent adoption at 50.",
                        "fact_type": "metric",
                        "source_level": "B",
                        "source_verification_status": "readpage_verified",
                        "proof_role": "metric",
                        "block_affinity": ["metric_reconciliation"],
                    }
                ]
            },
            "usage": {},
        }

    monkeypatch.setattr("rag_pipeline.agents.readpage_fact_extractor_agent.llm_config_is_ready", lambda config: True)
    monkeypatch.setattr("rag_pipeline.agents.readpage_fact_extractor_agent.call_openai_compatible_json", fake_llm)
    monkeypatch.setenv("READPAGE_FACT_EXTRACTOR_CACHE_ENABLED", "false")

    result = extract_fact_cards_from_pages(
        query="AI Agent adoption",
        page_results=[
            {
                "source_id": "S1",
                "title": "Agent adoption",
                "url": "https://www.salesforce.com/news/agent-adoption",
                "content": "The page reports AI Agent adoption at 50.",
            }
        ],
        search_task={
            "task_id": "ST-H1",
            "requirement_id": "H1_metric",
            "gap_id": "GAP-metric",
            "proof_role": "metric",
            "query": "AI Agent adoption",
            "required_fields": ["metric", "value", "unit", "period", "source"],
        },
    )

    summary = result["rejected_span_repair_summary"]
    assert result["fact_cards"] == []
    assert summary["status"] == "needs_repair"
    assert summary["reject_reason_counts"]["metric_missing_scope_or_period"] == 1
    assert summary["repair_task_seed"]["gap_id"] == "GAP-metric"
    assert summary["repair_task_seed"]["required_field_focus"] in {"unit", "period", "source"}


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
    monkeypatch.setenv("READPAGE_FACT_EXTRACTOR_MAX_PAGES_PER_TASK", "4")
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


def test_fallback_fact_cards_are_cached_after_llm_error(monkeypatch, tmp_path):
    calls = []

    def failing_llm(*, config, system_prompt, user_payload):
        calls.append(user_payload["source"]["url"])
        raise RuntimeError("model output truncated")

    monkeypatch.setattr("rag_pipeline.agents.readpage_fact_extractor_agent.llm_config_is_ready", lambda config: True)
    monkeypatch.setattr("rag_pipeline.agents.readpage_fact_extractor_agent.call_openai_compatible_json", failing_llm)
    monkeypatch.setenv("READPAGE_FACT_EXTRACTOR_CACHE_ENABLED", "true")
    monkeypatch.setenv("READPAGE_FACT_EXTRACTOR_CACHE_PATH", str(tmp_path / "fact_cache"))
    monkeypatch.setenv("READPAGE_FACT_EXTRACTOR_MAX_CALLS_PER_REPORT", "10")
    monkeypatch.setenv("REPORT_STAGE_SNAPSHOT_RUN_ID", "fallback-cache-test")
    reset_budget("fallback-cache-test")

    page = {
        "source_id": "SRC-market",
        "title": "Humanoid robot market",
        "url": "https://www.salesforce.com/news/humanoid-market",
        "content": "The report says humanoid robot market size reached 120 billion yuan in 2025.",
        "source_verification_status": "readpage_verified",
        "source_level": "B",
    }
    task = {
        "proof_role": "metric",
        "chapter_id": "CH_market",
        "requirement_id": "REQ_market",
        "required_fields": ["metric", "value", "unit", "period", "source"],
    }

    first = extract_fact_cards_from_pages(query="humanoid robot market size metric", page_results=[page], search_task=task)
    second = extract_fact_cards_from_pages(query="humanoid robot market size metric", page_results=[page], search_task=task)

    assert first["fallback_used"] is True
    assert first["llm_error_count"] == 1
    assert first["fact_card_count"] >= 1
    assert first["cache_hit_count"] == 0
    assert second["cache_hit_count"] == 1
    assert second["llm_error_count"] == 0
    assert second["fact_card_count"] == first["fact_card_count"]
    assert calls == ["https://www.salesforce.com/news/humanoid-market"]


def test_long_page_uses_chunk_extraction_before_whole_page_llm(monkeypatch, tmp_path):
    calls = []

    def fake_llm(*, config, system_prompt, user_payload):
        calls.append(user_payload.get("chunk", {}).get("chunk_index"))
        assert user_payload.get("chunk"), "long pages should be chunked before calling the LLM"
        return {
            "payload": {
                "fact_cards": [
                    {
                        "subject": "humanoid robot market",
                        "action_or_signal": "Humanoid robot market size reached 120 billion yuan in 2025.",
                        "variable": "market size",
                        "distilled_fact": "Humanoid robot market size reached 120 billion yuan in 2025.",
                        "fact_type": "metric",
                        "metric": "market size",
                        "value": "120",
                        "unit": "billion yuan",
                        "period": "2025",
                        "source_url": user_payload["source"]["url"],
                        "source_ref": user_payload["source"]["source_ref"],
                        "source_level": "B",
                        "source_verification_status": "readpage_verified",
                        "proof_role": "metric",
                        "block_affinity": ["metric_reconciliation"],
                        "claim_strength_hint": "directional",
                    }
                ]
            },
            "usage": {},
        }

    monkeypatch.setattr("rag_pipeline.agents.readpage_fact_extractor_agent.llm_config_is_ready", lambda config: True)
    monkeypatch.setattr("rag_pipeline.agents.readpage_fact_extractor_agent.call_openai_compatible_json", fake_llm)
    monkeypatch.setenv("READPAGE_FACT_EXTRACTOR_CACHE_ENABLED", "false")
    monkeypatch.setenv("READPAGE_FACT_EXTRACTOR_CHUNK_ENABLED", "true")
    monkeypatch.setenv("READPAGE_FACT_EXTRACTOR_CHUNK_TRIGGER_CHARS", "300")
    monkeypatch.setenv("READPAGE_FACT_EXTRACTOR_MAX_CHUNKS_PER_PAGE", "2")
    monkeypatch.setenv("READPAGE_FACT_EXTRACTOR_MAX_CALLS_PER_REPORT", "10")
    monkeypatch.setenv("REPORT_STAGE_SNAPSHOT_RUN_ID", "chunk-long-page-test")
    reset_budget("chunk-long-page-test")

    page = {
        "source_id": "SRC-market",
        "title": "Humanoid robot market",
        "url": "https://www.salesforce.com/news/humanoid-market",
        "content": "\n\n".join(
            [
                "# Navigation",
                "Login Contact Subscribe " * 20,
                "## Market size",
                "Humanoid robot market size reached 120 billion yuan in 2025.",
                "## Extra context",
                "Manufacturers are expanding deployment pilots. " * 20,
            ]
        ),
        "source_level": "B",
    }

    result = extract_fact_cards_from_pages(
        query="humanoid robot market size metric",
        page_results=[page],
        search_task={
            "proof_role": "metric",
            "chapter_id": "CH_market",
            "requirement_id": "REQ_market",
            "required_fields": ["metric", "value", "unit", "period", "source"],
        },
    )

    assert result["chunk_mode_used"] is True
    assert result["chunk_trigger_reason"] == "page_too_long"
    assert result["chunk_attempted"] >= 1
    assert result["fact_card_count"] == 1
    card = result["fact_cards"][0]
    assert card["chunk_index"] >= 1
    assert "120 billion yuan" in card["chunk_evidence_span"]
    assert calls and all(item is not None for item in calls)


def test_chunk_extraction_results_are_cached(monkeypatch, tmp_path):
    calls = []

    def fake_llm(*, config, system_prompt, user_payload):
        calls.append(user_payload.get("chunk", {}).get("chunk_index"))
        return {
            "payload": {
                "fact_cards": [
                    {
                        "subject": "humanoid robot market",
                        "action_or_signal": "Humanoid robot market size reached 120 billion yuan in 2025.",
                        "variable": "market size",
                        "distilled_fact": "Humanoid robot market size reached 120 billion yuan in 2025.",
                        "fact_type": "metric",
                        "metric": "market size",
                        "value": "120",
                        "unit": "billion yuan",
                        "period": "2025",
                        "source_url": user_payload["source"]["url"],
                        "source_ref": user_payload["source"]["source_ref"],
                        "source_level": "B",
                        "source_verification_status": "readpage_verified",
                        "proof_role": "metric",
                        "block_affinity": ["metric_reconciliation"],
                    }
                ]
            },
            "usage": {},
        }

    monkeypatch.setattr("rag_pipeline.agents.readpage_fact_extractor_agent.llm_config_is_ready", lambda config: True)
    monkeypatch.setattr("rag_pipeline.agents.readpage_fact_extractor_agent.call_openai_compatible_json", fake_llm)
    monkeypatch.setenv("READPAGE_FACT_EXTRACTOR_CACHE_ENABLED", "true")
    monkeypatch.setenv("READPAGE_FACT_EXTRACTOR_CACHE_PATH", str(tmp_path / "fact_cache"))
    monkeypatch.setenv("READPAGE_FACT_EXTRACTOR_CHUNK_ENABLED", "true")
    monkeypatch.setenv("READPAGE_FACT_EXTRACTOR_CHUNK_TRIGGER_CHARS", "200")
    monkeypatch.setenv("READPAGE_FACT_EXTRACTOR_MAX_CHUNKS_PER_PAGE", "1")
    monkeypatch.setenv("READPAGE_FACT_EXTRACTOR_MAX_CALLS_PER_REPORT", "10")
    monkeypatch.setenv("REPORT_STAGE_SNAPSHOT_RUN_ID", "chunk-cache-test")
    reset_budget("chunk-cache-test")

    page = {
        "source_id": "SRC-market",
        "title": "Humanoid robot market",
        "url": "https://www.salesforce.com/news/humanoid-market",
        "content": "## Market size\nHumanoid robot market size reached 120 billion yuan in 2025.\n" + ("context " * 80),
        "source_level": "B",
    }
    task = {
        "proof_role": "metric",
        "chapter_id": "CH_market",
        "requirement_id": "REQ_market",
        "required_fields": ["metric", "value", "unit", "period", "source"],
    }

    first = extract_fact_cards_from_pages(query="humanoid robot market size metric", page_results=[page], search_task=task)
    second = extract_fact_cards_from_pages(query="humanoid robot market size metric", page_results=[page], search_task=task)

    assert first["chunk_fact_card_count"] == 1
    assert first["chunk_cache_hit_count"] == 0
    assert second["chunk_cache_hit_count"] == 1
    assert second["fact_card_count"] == first["fact_card_count"]
    assert calls == [1]


def test_chunk_extraction_caps_fact_cards_and_sends_output_limits(monkeypatch):
    def noisy_llm(*, config, system_prompt, user_payload):
        assert user_payload["chunk"]["max_fact_cards"] == 3
        assert user_payload["chunk"]["max_fact_chars"] <= 220
        cards = []
        for index in range(6):
            cards.append(
                {
                    "subject": f"case {index}",
                    "action_or_signal": f"Customer deployment signal {index} in 2026.",
                    "variable": "deployment",
                    "distilled_fact": f"Customer deployment signal {index} in 2026.",
                    "fact_type": "case",
                    "source_url": user_payload["source"]["url"],
                    "source_ref": user_payload["source"]["source_ref"],
                    "source_level": "B",
                    "source_verification_status": "readpage_verified",
                    "proof_role": "case",
                    "block_affinity": ["case_comparison"],
                }
            )
        return {"payload": {"fact_cards": cards}, "usage": {}}

    monkeypatch.setattr("rag_pipeline.agents.readpage_fact_extractor_agent.llm_config_is_ready", lambda config: True)
    monkeypatch.setattr("rag_pipeline.agents.readpage_fact_extractor_agent.call_openai_compatible_json", noisy_llm)
    monkeypatch.setenv("READPAGE_FACT_EXTRACTOR_CACHE_ENABLED", "false")
    monkeypatch.setenv("READPAGE_FACT_EXTRACTOR_CHUNK_ENABLED", "true")
    monkeypatch.setenv("READPAGE_FACT_EXTRACTOR_CHUNK_TRIGGER_CHARS", "300")
    monkeypatch.setenv("READPAGE_FACT_EXTRACTOR_CHUNK_MAX_FACT_CARDS", "3")
    monkeypatch.setenv("READPAGE_FACT_EXTRACTOR_MAX_CHUNKS_PER_PAGE", "1")
    monkeypatch.setenv("READPAGE_FACT_EXTRACTOR_MAX_CALLS_PER_REPORT", "10")
    monkeypatch.setenv("REPORT_STAGE_SNAPSHOT_RUN_ID", "chunk-output-limit-test")
    reset_budget("chunk-output-limit-test")

    result = extract_fact_cards_from_pages(
        query="customer deployment cases",
        page_results=[
            {
                "source_id": "SRC-case",
                "title": "Deployment cases",
                "url": "https://finance.people.com.cn/case",
                "content": "## Cases\n" + ("Customer deployment signal in 2026. " * 40),
                "source_level": "B",
            }
        ],
        search_task={"proof_role": "case", "chapter_id": "CH_case", "requirement_id": "REQ_case"},
    )

    assert result["chunk_mode_used"] is True
    assert result["chunk_fact_card_count"] == 3
    assert result["fact_card_count"] == 3


def test_medium_page_uses_chunk_extraction_by_default(monkeypatch):
    calls = []

    def fake_llm(*, config, system_prompt, user_payload):
        calls.append(user_payload.get("chunk", {}).get("chunk_index"))
        return {
            "payload": {
                "fact_cards": [
                    {
                        "subject": "embodied intelligence deployment",
                        "action_or_signal": "State Grid planned to invest 6.8 billion yuan to procure 8,500 robots in 2026.",
                        "variable": "procurement",
                        "distilled_fact": "State Grid planned to invest 6.8 billion yuan to procure 8,500 robots in 2026.",
                        "fact_type": "case",
                        "source_url": user_payload["source"]["url"],
                        "source_ref": user_payload["source"]["source_ref"],
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
    monkeypatch.delenv("READPAGE_FACT_EXTRACTOR_CHUNK_TRIGGER_CHARS", raising=False)
    monkeypatch.setenv("READPAGE_FACT_EXTRACTOR_CHUNK_ENABLED", "true")
    monkeypatch.setenv("READPAGE_FACT_EXTRACTOR_MAX_CHUNKS_PER_PAGE", "1")
    monkeypatch.setenv("READPAGE_FACT_EXTRACTOR_MAX_CALLS_PER_REPORT", "10")
    monkeypatch.setenv("REPORT_STAGE_SNAPSHOT_RUN_ID", "chunk-default-threshold-test")
    reset_budget("chunk-default-threshold-test")

    page = {
        "source_id": "SRC-medium",
        "title": "Embodied intelligence goes to factory",
        "url": "https://finance.people.com.cn/embodied",
        "content": "## Case signal\nState Grid planned to invest 6.8 billion yuan to procure 8,500 robots in 2026.\n"
        + ("Industrial deployment context. " * 85),
        "source_level": "B",
    }

    result = extract_fact_cards_from_pages(
        query="embodied intelligence case procurement",
        page_results=[page],
        search_task={"proof_role": "case", "chapter_id": "CH_case", "requirement_id": "REQ_case"},
    )

    assert len(page["content"]) >= 1800
    assert result["chunk_mode_used"] is True
    assert result["fact_card_count"] == 1
    assert calls == [1]


def test_zero_valid_llm_result_falls_back_to_rule_extraction(monkeypatch):
    calls = []

    def empty_llm(*, config, system_prompt, user_payload):
        calls.append(user_payload["source"]["url"])
        return {"payload": {"fact_cards": []}, "usage": {}}

    monkeypatch.setattr("rag_pipeline.agents.readpage_fact_extractor_agent.llm_config_is_ready", lambda config: True)
    monkeypatch.setattr("rag_pipeline.agents.readpage_fact_extractor_agent.call_openai_compatible_json", empty_llm)
    monkeypatch.setenv("READPAGE_FACT_EXTRACTOR_CACHE_ENABLED", "false")
    monkeypatch.setenv("READPAGE_FACT_EXTRACTOR_CHUNK_ENABLED", "false")
    monkeypatch.setenv("READPAGE_FACT_EXTRACTOR_ZERO_VALID_FALLBACK_ENABLED", "true")
    monkeypatch.setenv("READPAGE_FACT_EXTRACTOR_MAX_CALLS_PER_REPORT", "10")
    monkeypatch.setenv("REPORT_STAGE_SNAPSHOT_RUN_ID", "zero-valid-fallback-test")
    reset_budget("zero-valid-fallback-test")

    result = extract_fact_cards_from_pages(
        query="local policy embodied intelligence target",
        page_results=[
                {
                    "source_id": "SRC-policy",
                    "title": "Local AI action plan",
                    "url": "https://www.lg.gov.cn/policy",
                    "content": "Longgang District released the AI Longgang Three-Year Action Plan for 2025-2027, targeting AI application in more than 80% of key industries by 2027.",
                "source_level": "A",
                "source_verification_status": "readpage_verified",
            }
        ],
        search_task={"proof_role": "policy", "chapter_id": "CH_policy", "requirement_id": "REQ_policy"},
    )

    assert calls == ["https://www.lg.gov.cn/policy"]
    assert result["zero_valid_fallback_used"] is True, json.dumps(result, ensure_ascii=False)
    assert result["fallback_used"] is True
    assert result["fact_card_count"] >= 1
    assert any("2027" in str(card.get("distilled_fact") or "") for card in result["fact_cards"])


def test_rule_fallback_skips_pdf_table_header_and_keeps_fact_sentence(monkeypatch):
    monkeypatch.setattr("rag_pipeline.agents.readpage_fact_extractor_agent.llm_config_is_ready", lambda config: False)
    monkeypatch.setenv("READPAGE_FACT_EXTRACTOR_CACHE_ENABLED", "false")
    monkeypatch.setenv("READPAGE_FACT_EXTRACTOR_ZERO_VALID_FALLBACK_ENABLED", "true")
    monkeypatch.setenv("REPORT_STAGE_SNAPSHOT_RUN_ID", "pdf-header-fallback-test")
    reset_budget("pdf-header-fallback-test")

    page = {
        "source_id": "SRC-pdf",
        "title": "\u4f4e\u7a7a\u7ecf\u6d4e\u4ea7\u4e1a\u94fe\u7814\u7a76\u4e13\u9898",
        "url": "https://pdf.dfcfw.com/pdf/low-altitude.pdf",
        "content": "\n".join(
            [
                "|||证券研究报告 ||22026年02月08日 026 年02 月06 日| |---|---|---|---| |低空经济产业链研究专题一|从产品到生态、从试点到常态，低空经济的发展潜力与机遇||优于大市| |核心观点|行业研究·行业专题||| |国防军工 优于大市·维持|||",
                "\u4f4e\u7a7a\u7ecf\u6d4e\u662f\u4f9d\u6258\u4f4e\u7a7a\u822a\u7a7a\u6d3b\u52a8\u5e26\u52a8\u76f8\u5173\u4ea7\u4e1a\u521b\u65b0\u548c\u573a\u666f\u5e94\u7528\u5f62\u6210\u7684\u7efc\u5408\u6027\u7ecf\u6d4e\u5f62\u6001\u3002",
                "\u622a\u81f32024\u5e74\uff0c\u6211\u56fd\u5728\u518c\u901a\u7528\u822a\u7a7a\u5668\u603b\u91cf\u3001\u901a\u7528\u822a\u7a7a\u4f01\u4e1a\u6570\u91cf\u5747\u521b\u65b0\u9ad8\u3002",
            ]
        ),
        "source_level": "B",
        "source_verification_status": "document_verified",
    }

    result = extract_fact_cards_from_pages(
        query="\u4e2d\u56fd\u4f4e\u7a7a\u7ecf\u6d4e\u4ea7\u4e1a\u94fe",
        page_results=[page],
        search_task={"proof_role": "support", "chapter_id": "CH_market", "requirement_id": "REQ_market"},
    )

    facts = "\n".join(str(card.get("distilled_fact") or "") for card in result["fact_cards"])
    assert result["fallback_used"] is True
    assert result["fact_card_count"] >= 1
    assert "证券研究报告" not in facts
    assert "|||" not in facts
    assert "\u4f4e\u7a7a\u7ecf\u6d4e" in facts


def test_chunk_ranking_prefers_order_contract_and_customer_case_segments(monkeypatch):
    monkeypatch.setenv("READPAGE_FACT_EXTRACTOR_CHUNK_MAX_CHARS", "360")
    monkeypatch.setenv("READPAGE_FACT_EXTRACTOR_MAX_CHUNKS_PER_PAGE", "1")
    weak_sections = []
    for index in range(12):
        weak_sections.extend(
            [
                f"## Weak context {index}",
                f"2025年，人形机器人公司在展会、演示和试点场景中继续推进技术验证，第{index}批样机展示了运动控制能力。",
                "Manufacturing clients are watching product stability and pilot progress. " * 4,
            ]
        )

    page = {
        "title": "Humanoid robot commercialization",
        "content": "\n\n".join(
            [
                "# Industry background",
                "Humanoid robot companies are showing demos at conferences. " * 8,
                "姚卯青称，当前人形机器人的投资回报率ROI不及成熟的机械臂系统，真正的价值在于补足非标准化环节。",
                *weak_sections,
                "## Order signal",
                "2025年下半年，多家中国人形机器人企业披露千台级订单，包括优必选、宇树科技、智元机器人、松延动力、星尘智能、智平方、众擎机器人、加速进化8家企业。",
                "智元机器人与龙旗科技签下数亿元合作协议，计划部署近千台机器人；星尘智能披露与仙工智能的千台级合作。",
            ]
        ),
    }

    chunks = _split_page_into_chunks(page, proof_role="case")
    joined = "\n".join(str(item.get("chunk_text") or "") for item in chunks)

    assert len(chunks) == 1
    assert "千台级订单" in joined
    assert "数亿元合作协议" in joined


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
