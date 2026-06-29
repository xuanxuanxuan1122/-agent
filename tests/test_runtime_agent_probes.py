import json

from rag_pipeline.agents.analysis_agent import run_analysis_agent
from rag_pipeline.agents.claim_builder_agent import run_claim_builder_agent
from rag_pipeline.agents.evidence_merger import merge_evidence_package
from rag_pipeline.agents.final_writer_agent import run_final_writer_agent
from rag_pipeline.agents.qa_agent import run_qa_agent
from rag_pipeline.agents.readpage_fact_extractor_agent import extract_fact_cards_from_pages
from rag_pipeline.flows.report import final_audit_agent
from rag_pipeline.flows.report import full_report
from rag_pipeline.observability.probe_context import current_probe_context_from_env


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_current_probe_context_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("RUNTIME_PROBE_RUN_ID", "run-env")
    monkeypatch.setenv("RUNTIME_PROBE_BASE_NAME", "run-env-base")
    monkeypatch.setenv("RUNTIME_PROBE_OUTPUT_DIR", str(tmp_path))

    probe = current_probe_context_from_env()

    assert probe is not None
    assert probe.run_id == "run-env"
    assert probe.base_name == "run-env-base"
    assert probe.live_path == tmp_path / "run-env-base.module_probe.live.jsonl"


def test_readpage_fact_extractor_emits_runtime_probe(monkeypatch, tmp_path):
    monkeypatch.setenv("RUNTIME_PROBE_RUN_ID", "run-fact")
    monkeypatch.setenv("RUNTIME_PROBE_BASE_NAME", "run-fact")
    monkeypatch.setenv("RUNTIME_PROBE_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("READPAGE_FACT_EXTRACTOR_CACHE_ENABLED", "false")
    monkeypatch.setenv("READPAGE_FACT_EXTRACTOR_MAX_CALLS_PER_REPORT", "0")

    result = extract_fact_cards_from_pages(
        query="AI Agent adoption",
        page_results=[
            {
                "url": "https://example.org/report",
                "source_id": "SRC-1",
                "source_level": "B",
                "mainText": "2025年，AI Agent 在企业客服和软件开发场景加速落地，多个厂商披露客户试点案例。",
            }
        ],
        search_task={"task_id": "T-1", "requirement_id": "REQ-1", "proof_role": "case", "chapter_id": "ch_01"},
    )

    assert result["attempted"] == 1
    live_path = tmp_path / "run-fact.module_probe.live.jsonl"
    events = _read_jsonl(live_path)
    event = next(item for item in events if item["stage"] == "fact_extractor")
    assert event["event_type"] == "transform_result"
    assert event["input"]["count"] == 1
    assert event["output"]["count"] == result["fact_card_count"]
    assert event["diagnostics"]["requirement_id"] == "REQ-1"
    assert event["diagnostics"]["search_task_id"] == "T-1"
    assert event["diagnostic_only"] is True
    assert event["public_text_allowed"] is False


def test_readpage_fact_extractor_probe_preserves_raw_input_count_when_page_text_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("RUNTIME_PROBE_RUN_ID", "run-fact-empty")
    monkeypatch.setenv("RUNTIME_PROBE_BASE_NAME", "run-fact-empty")
    monkeypatch.setenv("RUNTIME_PROBE_OUTPUT_DIR", str(tmp_path))

    result = extract_fact_cards_from_pages(
        query="AI Agent adoption",
        page_results=[{"url": "https://www.example.org/empty", "source_id": "SRC-empty"}],
        search_task={"task_id": "T-empty", "requirement_id": "REQ-empty", "proof_role": "case"},
    )

    assert result["status"] == "no_readpage_text"
    events = _read_jsonl(tmp_path / "run-fact-empty.module_probe.live.jsonl")
    event = next(item for item in events if item["stage"] == "fact_extractor")
    assert event["input"]["count"] == 1
    assert event["output"]["count"] == 0
    assert event["drop"]["reason_counts"]["no_readpage_text"] == 1


def test_readpage_fact_extractor_probe_reports_source_root_hygiene(monkeypatch, tmp_path):
    monkeypatch.setenv("RUNTIME_PROBE_RUN_ID", "run-fact-dirty")
    monkeypatch.setenv("RUNTIME_PROBE_BASE_NAME", "run-fact-dirty")
    monkeypatch.setenv("RUNTIME_PROBE_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("READPAGE_FACT_EXTRACTOR_CACHE_ENABLED", "false")
    monkeypatch.setenv("READPAGE_FACT_EXTRACTOR_MAX_CALLS_PER_REPORT", "0")

    extract_fact_cards_from_pages(
        query="AI Agent adoption",
        page_results=[
            {
                "url": "https://example.org/dirty",
                "source_id": "SRC-dirty",
                "source_level": "C",
                "mainText": "成本在2000为70%，这一指标用于判断市场成熟度。转化愿景为现实 引言",
            },
            {
                "url": "https://www.gov.cn/policy",
                "source_id": "SRC-policy",
                "source_level": "A",
                "mainText": "政策提出促进人工智能产业发展。",
                "source_type": "policy",
            },
        ],
        search_task={"task_id": "T-dirty", "requirement_id": "REQ-dirty", "proof_role": "metric"},
    )

    events = _read_jsonl(tmp_path / "run-fact-dirty.module_probe.live.jsonl")
    event = next(item for item in events if item["stage"] == "fact_extractor")
    hygiene = event["diagnostics"]["source_root_hygiene"]
    assert hygiene["dirty_item_count"] == 1
    assert hygiene["reason_counts"]["malformed_metric_span"] == 1
    assert hygiene["clean_item_count"] == 1
    assert event["drop"]["reason_counts"]["source_dirty_item"] == 1


def test_evidence_merger_emits_runtime_probe(monkeypatch, tmp_path):
    monkeypatch.setenv("RUNTIME_PROBE_RUN_ID", "run-merger")
    monkeypatch.setenv("RUNTIME_PROBE_BASE_NAME", "run-merger")
    monkeypatch.setenv("RUNTIME_PROBE_OUTPUT_DIR", str(tmp_path))

    package = merge_evidence_package(
        original_query="AI Agent industry report",
        evidence_pool=[
            {
                "status": "success",
                "confidence": 0.9,
                "key_sources": [
                    {
                        "title": "Official AI Agent Statistics",
                        "url": "https://www.stats.gov.cn/ai-agent-statistics",
                        "source_type": "official",
                    }
                ],
                "raw_data_points": [
                    {
                        "chapter_id": "ch_1",
                        "dimension": "industry overview",
                        "metric": "adoption",
                        "value": "50%",
                        "period": "2025",
                        "source_title": "Official AI Agent Statistics",
                        "source_url": "https://www.stats.gov.cn/ai-agent-statistics",
                        "source_type": "official",
                        "evidence": "Official statistics show AI agent adoption continued to rise in 2025.",
                    }
                ],
                "page_results": [
                    {
                        "url": "https://www.stats.gov.cn/ai-agent-statistics",
                        "content": "Official report body",
                        "auto_readpage": True,
                    }
                ],
                "metadata": {"auto_readpage": {"attempted": 1, "succeeded": 1}},
            }
        ],
        research_plan={"chapter_structure": [{"chapter_id": "ch_1", "chapter_title": "Industry overview"}]},
    )

    live_path = tmp_path / "run-merger.module_probe.live.jsonl"
    events = _read_jsonl(live_path)
    event = next(item for item in events if item["stage"] == "evidence_merger")
    assert event["event_type"] == "transform_result"
    assert event["input"]["count"] == 1
    assert event["output"]["count"] == package["summary"]["analysis_ready_count"]
    assert event["diagnostics"]["clean_fact_count"] == package["summary"]["clean_fact_count"]
    assert event["diagnostics"]["analysis_ready_count"] == package["summary"]["analysis_ready_count"]
    assert event["diagnostic_only"] is True


def test_evidence_merger_probe_reports_analysis_root_hygiene(monkeypatch, tmp_path):
    monkeypatch.setenv("RUNTIME_PROBE_RUN_ID", "run-merger-dirty")
    monkeypatch.setenv("RUNTIME_PROBE_BASE_NAME", "run-merger-dirty")
    monkeypatch.setenv("RUNTIME_PROBE_OUTPUT_DIR", str(tmp_path))

    package = {
        "summary": {"clean_fact_count": 2, "analysis_ready_count": 2},
        "analysis_ready_evidence": [
            {
                "evidence_id": "EV-dirty",
                "fact": "成本在2000为70%，这一指标用于判断市场成熟度。",
                "metric": "数据指标",
                "value": "240825",
                "source": {"title": "Report", "url": "https://example.org/r"},
            },
            {
                "evidence_id": "EV-clean",
                "fact": "政策提出促进人工智能产业发展。",
                "requirement_id": "REQ-clean",
                "source_id": "SRC-clean",
                "search_task_id": "T-clean",
                "source": {"title": "Policy", "url": "https://www.gov.cn/policy"},
            },
        ],
        "metadata": {"evidence_cache_store": {"cache_hit_count": 2, "polluted_count": 1}},
    }

    from rag_pipeline.agents import evidence_merger

    evidence_merger._emit_evidence_merger_probe(package, input_count=2)

    events = _read_jsonl(tmp_path / "run-merger-dirty.module_probe.live.jsonl")
    event = next(item for item in events if item["stage"] == "evidence_merger")
    hygiene = event["diagnostics"]["evidence_root_hygiene"]
    assert hygiene["dirty_item_count"] == 1
    assert hygiene["reason_counts"]["malformed_metric_span"] == 1
    assert hygiene["missing_lineage_counts"]["requirement_id"] == 1
    assert event["diagnostics"]["cache_root_hygiene"]["polluted_count"] == 1
    assert event["drop"]["reason_counts"]["evidence_dirty_item"] == 1


def test_analysis_agent_emits_runtime_probe(monkeypatch, tmp_path):
    monkeypatch.setenv("RUNTIME_PROBE_RUN_ID", "run-analysis")
    monkeypatch.setenv("RUNTIME_PROBE_BASE_NAME", "run-analysis")
    monkeypatch.setenv("RUNTIME_PROBE_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("BRAIN_ENABLE_LLM_EVIDENCE_ANALYSIS", "false")

    result = run_analysis_agent(
        {
            "query": "AI Agent enterprise adoption",
            "evidence_health_summary": {
                "analysis_ready_count": 1,
                "traceable_ab_source_count": 1,
                "distinct_verified_ab_source_count": 1,
            },
            "analysis_ready_evidence": [
                {
                    "evidence_id": "EV-1",
                    "chapter_id": "ch_01",
                    "requirement_id": "H1_case",
                    "fact": "Enterprise AI agent deployments are moving into workflow automation.",
                    "source_level": "B",
                    "source": {"title": "Verified source", "url": "https://www.example.org/source"},
                }
            ],
        }
    )

    structured = result["structured_analysis"]
    diagnostics = structured["analysis_stage_diagnostics"]
    live_path = tmp_path / "run-analysis.module_probe.live.jsonl"
    events = _read_jsonl(live_path)
    event = next(item for item in events if item["stage"] == "analysis_agent")
    assert event["event_type"] == "transform_result"
    assert event["input"]["count"] == 1
    assert event["output"]["count"] == len(structured["claim_units"])
    assert event["diagnostics"]["final_analysis_source"] == diagnostics["final_analysis_source"]
    assert event["diagnostics"]["llm_usable_claim_count"] == diagnostics["llm_usable_claim_count"]
    assert event["diagnostic_only"] is True


def test_claim_builder_emits_runtime_probe(monkeypatch, tmp_path):
    monkeypatch.setenv("RUNTIME_PROBE_RUN_ID", "run-claim-builder")
    monkeypatch.setenv("RUNTIME_PROBE_BASE_NAME", "run-claim-builder")
    monkeypatch.setenv("RUNTIME_PROBE_OUTPUT_DIR", str(tmp_path))

    units = run_claim_builder_agent(
        chapter_evidence_packages=[
            {
                "chapter_id": "ch_01",
                "chapter_title": "Demand validation",
                "core_evidence": [
                    {
                        "evidence_id": "EV-1",
                        "fact": "Enterprise AI agent deployments are moving into workflow automation.",
                        "source": {"title": "Verified source", "url": "https://www.example.org/source"},
                        "source_level": "B",
                    }
                ],
            }
        ],
        structured_analysis={
            "claim_units": [
                {
                    "claim_id": "C-1",
                    "chapter_id": "ch_01",
                    "claim": "AI Agent deployments are moving into workflow automation.",
                    "used_evidence_ids": ["EV-1"],
                    "fact_ids": ["EV-1"],
                    "requirement_ids": ["H1_case"],
                    "claim_strength": "directional",
                    "reasoning_chain": ["The evidence describes workflow deployments."],
                    "limitation_boundary": ["Limited to disclosed deployment samples."],
                }
            ]
        },
    )

    live_path = tmp_path / "run-claim-builder.module_probe.live.jsonl"
    events = _read_jsonl(live_path)
    event = next(item for item in events if item["stage"] == "claim_builder")
    public_units = [item for item in units if item.get("public_render")]
    omitted_units = [item for item in units if item.get("omit_from_report")]
    assert event["event_type"] == "transform_result"
    assert event["input"]["count"] == 1
    assert event["output"]["count"] == len(public_units)
    assert event["drop"]["count"] == len(omitted_units)
    assert event["diagnostics"]["argument_unit_count"] == len(units)
    assert event["diagnostic_only"] is True


def test_final_writer_emits_runtime_probe(monkeypatch, tmp_path):
    monkeypatch.setenv("RUNTIME_PROBE_RUN_ID", "run-final-writer")
    monkeypatch.setenv("RUNTIME_PROBE_BASE_NAME", "run-final-writer")
    monkeypatch.setenv("RUNTIME_PROBE_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("REPORT_FINAL_WRITER_SOURCE_APPENDIX", "true")

    output = run_final_writer_agent(
        query="AI Agent",
        report_blueprint={
            "report_shell": {"front_blocks": [], "back_blocks": ["appendix"]},
            "chapters": [{"chapter_id": "ch_01", "chapter_title": "Demand validation"}],
        },
        chapter_packages=[
            {
                "chapter_id": "ch_01",
                "chapter_title": "Demand validation",
                "sections": [
                    {
                        "section_id": "s1",
                        "section_title": "Customer deployment",
                        "claim": "Enterprise AI agents are moving from demos into customer-service workflows.",
                        "reasoning": "Workflow deployment requires permissions, integration, and process ownership.",
                        "mechanism": "A production workflow is a stronger signal than a demo.",
                        "used_fact_refs": ["EV-1"],
                        "evidence_refs": ["EV-1"],
                        "evidence_backed": True,
                    }
                ],
            }
        ],
        source_registry=[
            {
                "ref": "[7]",
                "evidence_id": "EV-1",
                "title": "Salesforce Agentforce deployment note",
                "url": "https://www.salesforce.com/news/agentforce",
                "source_level": "B",
            }
        ],
    )

    events = _read_jsonl(tmp_path / "run-final-writer.module_probe.live.jsonl")
    event = next(item for item in events if item["stage"] == "final_writer")
    assert event["event_type"] == "transform_result"
    assert event["input"]["count"] == 1
    assert event["output"]["count"] >= 1
    assert event["diagnostics"]["markdown_chars"] == len(output["report_markdown"])
    assert event["diagnostics"]["citation_manifest_status"] == output["citation_manifest"]["citation_manifest_status"]
    assert event["diagnostic_only"] is True


def test_qa_agent_emits_runtime_probe(monkeypatch, tmp_path):
    monkeypatch.setenv("RUNTIME_PROBE_RUN_ID", "run-qa")
    monkeypatch.setenv("RUNTIME_PROBE_BASE_NAME", "run-qa")
    monkeypatch.setenv("RUNTIME_PROBE_OUTPUT_DIR", str(tmp_path))

    result = run_qa_agent(report_markdown="# Short report", chapter_packages=[])

    events = _read_jsonl(tmp_path / "run-qa.module_probe.live.jsonl")
    event = next(item for item in events if item["stage"] == "qa_agent")
    assert event["event_type"] == "transform_result"
    assert event["input"]["count"] == 1
    assert event["output"]["count"] == 1
    assert event["drop"]["count"] == len(result["errors"]) + len(result["fatal_errors"])
    assert event["diagnostics"]["quality_score"] == result["quality_score"]
    assert event["diagnostic_only"] is True


def test_final_audit_emits_runtime_probe_on_disabled_return(monkeypatch, tmp_path):
    monkeypatch.setenv("RUNTIME_PROBE_RUN_ID", "run-final-audit")
    monkeypatch.setenv("RUNTIME_PROBE_BASE_NAME", "run-final-audit")
    monkeypatch.setenv("RUNTIME_PROBE_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("REPORT_ENABLE_FINAL_AUDIT", "false")

    result = final_audit_agent.run_final_audit(report_markdown="# Report")

    events = _read_jsonl(tmp_path / "run-final-audit.module_probe.live.jsonl")
    event = next(item for item in events if item["stage"] == "final_audit")
    assert event["event_type"] == "transform_result"
    assert event["input"]["count"] == 1
    assert event["output"]["count"] == 0
    assert event["diagnostics"]["status"] == result["status"]
    assert event["diagnostic_only"] is True


def test_final_audit_probe_keeps_fatal_observed_separate_from_blocked(monkeypatch, tmp_path):
    monkeypatch.setenv("RUNTIME_PROBE_RUN_ID", "run-final-audit-fatal")
    monkeypatch.setenv("RUNTIME_PROBE_BASE_NAME", "run-final-audit-fatal")
    monkeypatch.setenv("RUNTIME_PROBE_OUTPUT_DIR", str(tmp_path))

    result = {
        "enabled": True,
        "success": True,
        "status": "fatal",
        "blocked": False,
        "blocking": False,
        "audit": {
            "overall_score": 30,
            "critical_findings": [{"type": "citation_issue", "severity": "fatal"}],
        },
    }
    final_audit_agent._emit_final_audit_probe(result, report_markdown="# Report")

    events = _read_jsonl(tmp_path / "run-final-audit-fatal.module_probe.live.jsonl")
    event = next(item for item in events if item["stage"] == "final_audit")
    assert event["status"] == "error"
    assert event["drop"]["count"] == 1
    assert event["diagnostics"]["blocked"] is False
    assert event["diagnostics"]["fatal_observed"] is True


def test_full_report_loads_final_audit_runner_with_absolute_fallback(monkeypatch):
    original_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 1 and name == "final_audit_agent":
            raise ImportError("attempted relative import with no known parent package")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)

    assert full_report._load_final_audit_runner() is final_audit_agent.run_final_audit
