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
            "success_criteria": "Only count repaired when metric/value/unit/period/source are all present.",
            "reject_if": ["snippet_only", "no_date", "no_source_url"],
        },
    )

    payload = captured["user_payload"]
    assert payload["schema_version"] == "readpage_fact_card_v2"
    assert "Use only the supplied page text" in captured["system_prompt"]
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
