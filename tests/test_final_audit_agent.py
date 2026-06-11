from __future__ import annotations

from rag_pipeline.flows.report import final_audit_agent


def _configure_deepseek_final_audit(monkeypatch):
    monkeypatch.setenv("RAG_MODEL_FINAL_AUDIT_PROFILE", "deepseek-v4-pro")
    monkeypatch.setenv("RAG_LLM_PROFILE_DEEPSEEK_V4_PRO_PROVIDER", "openai_compatible")
    monkeypatch.setenv("RAG_LLM_PROFILE_DEEPSEEK_V4_PRO_URL", "https://api.deepseek.example/chat/completions")
    monkeypatch.setenv("RAG_LLM_PROFILE_DEEPSEEK_V4_PRO_API_KEY", "deepseek-test")
    monkeypatch.setenv("RAG_LLM_PROFILE_DEEPSEEK_V4_PRO_MODEL", "deepseek-v4-pro")
    monkeypatch.setenv("RAG_LLM_PROFILE_DEEPSEEK_V4_PRO_TIMEOUT", "240")
    monkeypatch.setenv("RAG_LLM_PROFILE_DEEPSEEK_V4_PRO_MAX_OUTPUT_TOKENS", "32000")


def test_final_audit_disabled_skips_without_calling_model(monkeypatch):
    monkeypatch.setenv("REPORT_ENABLE_FINAL_AUDIT", "false")

    result = final_audit_agent.run_final_audit(report_markdown="# Report")

    assert result["status"] == "skipped"
    assert result["skipped_reason"] == "disabled"


def test_final_audit_fatal_blocks_when_blocking_enabled(monkeypatch):
    _configure_deepseek_final_audit(monkeypatch)
    monkeypatch.setenv("REPORT_ENABLE_FINAL_AUDIT", "true")
    monkeypatch.setenv("REPORT_FINAL_AUDIT_BLOCKING", "true")
    captured = {}

    def fake_call_openai_compatible_json(*, config, system_prompt, user_payload):
        captured["config"] = config
        captured["user_payload"] = user_payload
        return {
            "payload": {
                "status": "fatal",
                "overall_score": 35,
                "critical_findings": [{"severity": "fatal", "message": "Unsupported investment conclusion"}],
                "publish_recommendation": "hold",
                "summary": "Do not deliver without repair.",
            },
            "usage": {"total_tokens": 10},
            "llm_call": {
                "task": "final_audit",
                "profile": "deepseek-v4-pro",
                "model": "deepseek-v4-pro",
                "api": "openai_compatible_chat_json",
                "status": "success",
            },
        }

    monkeypatch.setattr(final_audit_agent, "call_openai_compatible_json", fake_call_openai_compatible_json)

    result = final_audit_agent.run_final_audit(
        report_markdown="# Report\n\nConclusion [1]\n\n## 来源附录\n- [1] Source | https://www.stats.gov.cn/source",
        validation={"passed": True, "quality_score": 90},
        clean_evidence={"sources": [{"id": "1", "title": "Source", "url": "https://www.stats.gov.cn/source"}]},
        writer_package_payload={"quality_gate_state": {"status": "publishable"}},
        query="industry report",
    )

    assert captured["config"]["model"] == "deepseek-v4-pro"
    assert captured["user_payload"]["reformatter_validation"]["passed"] is True
    assert result["status"] == "fatal"
    assert result["blocked"] is True
    assert result["audit"]["publish_recommendation"] == "hold"
    assert result["llm_call"]["model"] == "deepseek-v4-pro"


def test_isolated_quality_gate_observes_fatal_audit_without_blocking(monkeypatch):
    _configure_deepseek_final_audit(monkeypatch)
    monkeypatch.setenv("REPORT_ENABLE_FINAL_AUDIT", "true")
    monkeypatch.setenv("REPORT_FINAL_AUDIT_BLOCKING", "true")
    monkeypatch.setenv("REPORT_QUALITY_GATE_MODE", "isolated")

    def fake_call_openai_compatible_json(*, config, system_prompt, user_payload):
        return {
            "payload": {
                "status": "fatal",
                "overall_score": 35,
                "critical_findings": [{"severity": "fatal", "message": "Unsupported citation"}],
                "publish_recommendation": "hold",
                "summary": "Observe only in isolated mode.",
            },
            "usage": {"total_tokens": 10},
            "llm_call": {"task": "final_audit", "model": "deepseek-v4-pro", "status": "success"},
        }

    monkeypatch.setattr(final_audit_agent, "call_openai_compatible_json", fake_call_openai_compatible_json)

    result = final_audit_agent.run_final_audit(
        report_markdown="# Report\n\nConclusion [1]\n\n## 来源附录\n- [1] Source | https://example.org/source",
        validation={"passed": True, "quality_score": 90},
        clean_evidence={"sources": [{"id": "1", "title": "Source", "url": "https://example.org/source"}]},
    )

    assert result["status"] == "fatal"
    assert result["blocked"] is False
    assert result["blocking"] is False
    assert result["quality_gate_mode"] == "isolated"
    assert result["audit"]["publish_recommendation"] == "hold"


def test_isolated_quality_gate_preserves_deterministic_fatal_when_llm_config_missing(monkeypatch):
    monkeypatch.setenv("REPORT_ENABLE_FINAL_AUDIT", "true")
    monkeypatch.setenv("REPORT_FINAL_AUDIT_BLOCKING", "true")
    monkeypatch.setenv("REPORT_QUALITY_GATE_MODE", "isolated")
    monkeypatch.delenv("RAG_MODEL_FINAL_AUDIT_PROFILE", raising=False)
    monkeypatch.setattr(final_audit_agent, "llm_config_is_ready", lambda _config: False)

    result = final_audit_agent.run_final_audit(
        report_markdown="# Report\n\nUnsupported conclusion [1].",
        clean_evidence={"sources": []},
    )

    assert result["status"] == "fatal"
    assert result["blocked"] is False
    assert result["blocking"] is False
    assert result["audit"]["publish_recommendation"] == "hold"
    assert result["skipped_reason"] == "config_missing"


def test_final_audit_payload_includes_compact_repair_gap_context(monkeypatch):
    _configure_deepseek_final_audit(monkeypatch)
    monkeypatch.setenv("REPORT_ENABLE_FINAL_AUDIT", "true")
    monkeypatch.setenv("REPORT_FINAL_AUDIT_BLOCKING", "false")
    captured = {}

    def fake_call_openai_compatible_json(*, config, system_prompt, user_payload):
        captured["system_prompt"] = system_prompt
        captured["user_payload"] = user_payload
        return {
            "payload": {
                "status": "warning",
                "overall_score": 72,
                "critical_findings": [
                    {
                        "type": "evidence_gap",
                        "severity": "high",
                        "requirement_id": "H1_metric",
                        "gap_id": "GAP-metric",
                        "section_id": "SEC-1",
                        "message": "Metric period is missing.",
                    }
                ],
                "publish_recommendation": "publish_with_caveats",
                "summary": "Needs one metric repair.",
            },
            "usage": {"total_tokens": 10},
            "llm_call": {"task": "final_audit", "model": "deepseek-v4-pro", "status": "success"},
        }

    monkeypatch.setattr(final_audit_agent, "call_openai_compatible_json", fake_call_openai_compatible_json)

    result = final_audit_agent.run_final_audit(
        report_markdown="# Report\n\nConclusion [1]\n\n## 来源附录\n- [1] Source | https://www.stats.gov.cn/source",
        validation={"passed": True},
        clean_evidence={"sources": [{"id": "1", "title": "Source", "url": "https://www.stats.gov.cn/source"}]},
        writer_package_payload={
            "score_gaps": [
                {
                    "gap_id": "GAP-metric",
                    "requirement_id": "H1_metric",
                    "section_id": "SEC-1",
                    "gap_type": "metric_scope_period_unit_incomplete",
                    "status": "still_insufficient",
                    "missing": ["period"],
                    "retry_plan": {"raw_page": "forbidden raw page", "query_terms": ["AI Agent adoption"]},
                }
            ],
            "requirement_gap_summary": {"H1_metric": {"open_gap_count": 1}},
            "final_citation_audit": {
                "final_citation_reconciliation_status": "ok",
                "final_body_citation_refs": ["1"],
                "final_appendix_refs": ["1"],
            },
        },
        query="industry report",
    )

    payload = captured["user_payload"]
    assert "requirement_id" in captured["system_prompt"]
    assert payload["score_gaps"][0]["gap_id"] == "GAP-metric"
    assert payload["score_gaps"][0]["requirement_id"] == "H1_metric"
    assert payload["score_gaps"][0]["retry_plan"]["query_terms"] == ["AI Agent adoption"]
    assert "raw_page" not in str(payload["score_gaps"])
    assert payload["requirement_gap_summary"]["H1_metric"]["open_gap_count"] == 1
    assert payload["final_citation_audit"]["final_citation_reconciliation_status"] == "ok"
    assert result["audit"]["critical_findings"][0]["gap_id"] == "GAP-metric"


def test_final_audit_sets_large_default_output_budget(monkeypatch):
    monkeypatch.setenv("RAG_MODEL_FINAL_AUDIT_PROFILE", "deepseek-v4-pro")
    monkeypatch.setenv("RAG_LLM_PROFILE_DEEPSEEK_V4_PRO_PROVIDER", "openai_compatible")
    monkeypatch.setenv("RAG_LLM_PROFILE_DEEPSEEK_V4_PRO_URL", "https://api.deepseek.example/chat/completions")
    monkeypatch.setenv("RAG_LLM_PROFILE_DEEPSEEK_V4_PRO_API_KEY", "deepseek-test")
    monkeypatch.setenv("RAG_LLM_PROFILE_DEEPSEEK_V4_PRO_MODEL", "deepseek-v4-pro")
    monkeypatch.setenv("REPORT_ENABLE_FINAL_AUDIT", "true")
    monkeypatch.setenv("REPORT_FINAL_AUDIT_BLOCKING", "false")
    monkeypatch.delenv("RAG_LLM_PROFILE_DEEPSEEK_V4_PRO_MAX_OUTPUT_TOKENS", raising=False)
    captured = {}

    def fake_call_openai_compatible_json(*, config, system_prompt, user_payload):
        captured["config"] = config
        return {
            "payload": {
                "status": "pass",
                "overall_score": 88,
                "critical_findings": [],
                "publish_recommendation": "publish_with_caveats",
                "summary": "ok",
            },
            "usage": {"total_tokens": 10},
            "llm_call": {"task": "final_audit", "model": "deepseek-v4-pro", "status": "success"},
        }

    monkeypatch.setattr(final_audit_agent, "call_openai_compatible_json", fake_call_openai_compatible_json)

    result = final_audit_agent.run_final_audit(
        report_markdown="# Report\n\nConclusion [1]\n\n## 来源附录\n- [1] Source | https://www.stats.gov.cn/source",
        clean_evidence={"sources": [{"id": "1", "title": "Source", "url": "https://www.stats.gov.cn/source"}]},
    )

    assert result["success"] is True
    assert int(captured["config"]["max_output_tokens"]) >= 8192


def test_final_audit_drops_false_future_date_fatal_when_date_is_not_future(monkeypatch):
    _configure_deepseek_final_audit(monkeypatch)
    monkeypatch.setenv("REPORT_ENABLE_FINAL_AUDIT", "true")
    monkeypatch.setenv("REPORT_FINAL_AUDIT_BLOCKING", "true")
    monkeypatch.setenv("REPORT_FINAL_AUDIT_CURRENT_DATE", "2026-06-02")
    captured = {}

    def fake_call_openai_compatible_json(*, config, system_prompt, user_payload):
        captured["user_payload"] = user_payload
        return {
            "payload": {
                "status": "fatal",
                "overall_score": 0,
                "critical_findings": [
                    {
                        "type": "data_conflict",
                        "severity": "fatal",
                        "message": "Source [1] is dated 2026-01-27, over a year in the future.",
                        "evidence_hint": "Source [1] date",
                        "suggested_fix": "Replace the future-dated source.",
                    }
                ],
                "citation_issues": [
                    {
                        "severity": "fatal",
                        "issue": "The link points to a future-dated IR Q&A page (2026-01-27).",
                        "source_id": "[1]",
                    }
                ],
                "publish_recommendation": "hold",
                "summary": "Future dated citation.",
            },
            "usage": {"total_tokens": 10},
            "llm_call": {"task": "final_audit", "model": "deepseek-v4-pro", "status": "success"},
        }

    monkeypatch.setattr(final_audit_agent, "call_openai_compatible_json", fake_call_openai_compatible_json)

    result = final_audit_agent.run_final_audit(
        report_markdown="# Report\n\nConclusion [1]\n\n## 来源附录\n- [1] Source | 2026-01-27 | https://example.org/source",
        clean_evidence={"sources": [{"id": "1", "title": "Source", "url": "https://example.org/source", "date": "2026-01-27"}]},
        writer_package_payload={"quality_gate_state": {"status": "publishable"}},
        query="industry report",
    )

    assert captured["user_payload"]["current_date"] == "2026-06-02"
    assert result["blocked"] is False
    assert result["status"] != "fatal"
    assert result["audit"]["date_sanity_removed_findings_count"] == 2
    assert result["audit"]["critical_findings"] == []
    assert result["audit"]["citation_issues"] == []


def test_deterministic_audit_blocks_placeholder_sources():
    result = final_audit_agent.run_deterministic_audit(
        report_markdown=(
            "# Report\n\n"
            "Official data shows AI agent adoption reached 50% in 2025. [1]\n\n"
            "## Sources\n- [1] Official | https://example.gov/ai-agent-statistics"
        ),
        clean_evidence={
            "sources": [
                {
                    "id": "1",
                    "title": "Official",
                    "url": "https://example.gov/ai-agent-statistics",
                    "source_level": "A",
                }
            ]
        },
    )

    assert result["fatal"] is True
    finding_types = {item["type"] for item in result["findings"]}
    assert "fake_or_placeholder_evidence" in finding_types
    assert "fake_or_placeholder_source" in finding_types


def test_deterministic_audit_does_not_duplicate_citationless_body_as_final_gap():
    result = final_audit_agent.run_deterministic_audit(
        report_markdown=(
            "# Report\n\n"
            "Official data shows enterprise adoption signals are rising. [1]\n\n"
            "30 AI Agent enterprise landing cases.\n\n"
            "## Sources\n- [1] Official | https://www.stats.gov.cn/source"
        ),
        clean_evidence={
            "sources": [
                {
                    "id": "1",
                    "title": "Official",
                    "url": "https://www.stats.gov.cn/source",
                    "source_level": "A",
                }
            ]
        },
        writer_package_payload={
            "final_citation_audit": {
                "final_citation_reconciliation_status": "blocked",
                "final_missing_appendix_refs": [],
                "final_body_citation_refs": ["1"],
                "final_appendix_refs": ["1"],
                "factual_body_without_citations_count": 1,
                "citationless_fact_examples": ["30 AI Agent enterprise landing cases."],
            }
        },
    )

    finding_types = [item["type"] for item in result["findings"]]
    assert "citationless_factual_body" in finding_types
    assert "final_citation_gap" not in finding_types
    assert result["fatal"] is True


def test_deterministic_audit_title_only_only_fatal_when_cited():
    unused = final_audit_agent.run_deterministic_audit(
        report_markdown="# Report\n\nConclusion [1].\n\n## Sources\n- [1] Source | https://www.stats.gov.cn/source",
        writer_package_payload={
            "source_registry": [
                {"ref": "[1]", "title": "Source", "url": "https://www.stats.gov.cn/source"},
                {"ref": "[2]", "title": "Unused title only"},
            ]
        },
    )
    unused_types = {item["type"]: item.get("severity") for item in unused["findings"]}
    assert unused["fatal"] is False
    assert unused_types.get("title_only_source_candidate") == "medium"

    cited = final_audit_agent.run_deterministic_audit(
        report_markdown="# Report\n\nConclusion [2].\n\n## Sources\n- [2] Unused title only",
        writer_package_payload={"source_registry": [{"ref": "[2]", "title": "Unused title only"}]},
    )
    assert cited["fatal"] is True
    assert any(item["type"] == "title_only_source" for item in cited["findings"])

