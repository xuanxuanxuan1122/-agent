from __future__ import annotations

from rag_pipeline.flows.report import final_audit_agent


def _configure_gpt55_final_audit(monkeypatch):
    monkeypatch.setenv("RAG_MODEL_FINAL_AUDIT_PROFILE", "gpt-5.5")
    monkeypatch.setenv("RAG_LLM_PROFILE_GPT_5_5_PROVIDER", "openai_compatible")
    monkeypatch.setenv("RAG_LLM_PROFILE_GPT_5_5_URL", "https://api.openai.com/v1/responses")
    monkeypatch.setenv("RAG_LLM_PROFILE_GPT_5_5_API_KEY", "sk-test")
    monkeypatch.setenv("RAG_LLM_PROFILE_GPT_5_5_MODEL", "gpt-5.5")
    monkeypatch.setenv("RAG_LLM_PROFILE_GPT_5_5_TIMEOUT", "240")
    monkeypatch.setenv("RAG_LLM_PROFILE_GPT_5_5_REASONING_EFFORT", "high")
    monkeypatch.setenv("RAG_LLM_PROFILE_GPT_5_5_MAX_OUTPUT_TOKENS", "32000")


def test_final_audit_disabled_skips_without_calling_model(monkeypatch):
    monkeypatch.setenv("REPORT_ENABLE_FINAL_AUDIT", "false")

    result = final_audit_agent.run_final_audit(report_markdown="# Report")

    assert result["status"] == "skipped"
    assert result["skipped_reason"] == "disabled"


def test_final_audit_fatal_blocks_when_blocking_enabled(monkeypatch):
    _configure_gpt55_final_audit(monkeypatch)
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
                "profile": "gpt-5.5",
                "model": "gpt-5.5",
                "api": "openai_responses_json",
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

    assert captured["config"]["model"] == "gpt-5.5"
    assert captured["config"]["reasoning_effort"] == "high"
    assert captured["user_payload"]["reformatter_validation"]["passed"] is True
    assert result["status"] == "fatal"
    assert result["blocked"] is True
    assert result["audit"]["publish_recommendation"] == "hold"
    assert result["llm_call"]["model"] == "gpt-5.5"


def test_final_audit_drops_false_future_date_fatal_when_date_is_not_future(monkeypatch):
    _configure_gpt55_final_audit(monkeypatch)
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
            "llm_call": {"task": "final_audit", "model": "gpt-5.5", "status": "success"},
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

