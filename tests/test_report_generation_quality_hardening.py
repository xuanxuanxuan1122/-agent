from __future__ import annotations

import inspect

from rag_pipeline.agents.brain_agent import _lane_coverage_from_state
from rag_pipeline.agents.evidence_binder import _report_proof_mode
from rag_pipeline.agents.evidence_merger import _source_traceability_payload
from rag_pipeline.agents.final_writer_agent import run_final_writer_agent
from rag_pipeline.agents.public_report_sanitizer import public_narrative_leak_audit, sanitize_public_markdown
from rag_pipeline.agents.qa_agent import run_qa_agent, validate_no_internal_gap_language
from rag_pipeline.agents.report_profile_registry import select_report_profile
from rag_pipeline.flows.report import full_report
from rag_pipeline.flows.report.final_audit_agent import run_deterministic_audit


def test_lane_coverage_preserves_planned_missing_lane_state():
    coverage = _lane_coverage_from_state(
        {
            "query_analysis": {
                "agent_tasks": {
                    "iqs_lane_3": [{"query": "task 1"}, {"query": "task 2"}],
                }
            }
        }
    )

    lane = coverage["iqs_lane_3"]
    assert lane["planned_task_count"] == 2
    assert lane["scheduled"] == 2
    assert lane["execution_status"] == "missing_state"


def test_deep_report_lane_timeout_blocks_qa(monkeypatch):
    monkeypatch.setenv("REPORT_TARGET_BODY_CHARS", "0")
    monkeypatch.setenv("QA_DEEP_EVALUATOR_BLOCKING", "true")

    result = run_qa_agent(
        report_markdown="# AI Agent生态发展报告\n\n这是一段已经进入正文的判断。",
        report_blueprint={
            "report_family": "industry_deep_report",
            "chapters": [{"chapter_id": "ch1", "chapter_title": "需求是否成立"}],
        },
        chapter_packages=[
            {
                "chapter_id": "ch1",
                "chapter_title": "需求是否成立",
                "sections": [
                    {
                        "section_title": "核心判断",
                        "claim": "AI Agent生态仍需要证据验证。",
                        "reasoning": "因为企业采用、预算和产品成熟度会共同影响落地节奏。",
                        "counter_evidence": "若客户预算收缩，采用节奏可能放缓。",
                    }
                ],
                "evidence_quality_summary": {"core_ab_source_count": 0, "source_level_distribution": {"C": 1}},
            }
        ],
        search_task_schedule={"scheduled_tasks": [{"scheduled_lane": "iqs_lane_1"}]},
        lane_coverage={
            "iqs_lane_1": {
                "scheduled": 3,
                "succeeded": 0,
                "timed_out_task_count": 3,
                "usable_source_count": 0,
                "page_results": 0,
                "execution_status": "timed_out",
            }
        },
    )

    gap_types = {
        item["detail"]["type"]
        for item in result["errors"]
        if item.get("type") == "deep_report_blocking_gap" and isinstance(item.get("detail"), dict)
    }
    assert result["passed"] is False
    assert result["publishable"] is False
    assert "iqs_lane_no_success" in gap_types
    assert "iqs_lane_timeout_without_signal" in gap_types


def test_internal_report_markers_are_fatal_or_flagged():
    errors = validate_no_internal_gap_language("正文和EV-01-L14相邻，同时泄漏 evidence_cards 与本章应写成。")
    assert errors

    audit = run_deterministic_audit(
        report_markdown="# 报告\n\n正文和EV-01-L14相邻，同时泄漏 evidence_cards 与本章应写成。\n\n## 数据来源\n- [1] 来源 | https://example.com",
        clean_evidence={"sources": [{"id": "1", "title": "来源", "url": "https://example.com"}]},
    )
    finding_types = {item["type"] for item in audit["findings"]}
    assert audit["fatal"] is True
    assert "internal_evidence_id" in finding_types
    assert "internal_evidence_cards" in finding_types
    assert "internal_draft_instruction" in finding_types


def test_empty_markdown_table_is_removed_as_a_block():
    markdown = "正文。\n\n**空表**\n\n| 指标 | 数值 |\n| --- | --- |\n\n后文。"

    cleaned = sanitize_public_markdown(markdown)

    assert "| 指标 | 数值 |" not in cleaned
    assert "**空表**" not in cleaned
    assert "后文" in cleaned


def test_deterministic_audit_blocks_missing_appendix_and_title_only_source():
    missing_appendix = run_deterministic_audit(
        report_markdown="# 报告\n\n正文引用来源[1]。",
        clean_evidence={"sources": [{"id": "1", "title": "来源", "url": "https://example.com"}]},
    )
    assert any(item["type"] == "missing_sources_appendix" for item in missing_appendix["findings"])

    title_only = run_deterministic_audit(
        report_markdown="# 报告\n\n正文引用来源[1]。\n\n## 数据来源\n- [1] 只有标题",
        writer_package_payload={"source_registry": [{"ref": "[1]", "title": "只有标题"}]},
    )
    assert any(item["type"] == "title_only_source" for item in title_only["findings"])


def test_final_writer_forces_source_appendix_when_body_has_citations(monkeypatch):
    monkeypatch.setenv("REPORT_FINAL_WRITER_SOURCE_APPENDIX", "false")

    result = run_final_writer_agent(
        query="AI Agent生态发展报告：从工具到智能体的范式跃迁",
        report_blueprint={
            "report_family": "industry_deep_report",
            "research_object": "AI Agent生态",
            "report_shell": {"front_blocks": [], "back_blocks": []},
        },
        chapter_packages=[
            {
                "chapter_title": "需求是否成立",
                "sections": [
                    {
                        "section_title": "核心判断",
                        "claim": "企业采用节奏仍需观察[1]。",
                        "reasoning": "预算、产品成熟度和集成成本共同影响落地速度。",
                    }
                ],
            }
        ],
        source_registry=[{"ref": "[1]", "title": "来源一", "url": "https://www.stats.gov.cn/1"}],
    )

    markdown = result["report_markdown"]
    assert "https://www.stats.gov.cn/1" in markdown
    assert markdown.count("[1]") >= 2


def test_public_sanitizer_removes_soft_internal_narrative_language():
    markdown = (
        "# 报告\n\n"
        "研究主线：先界定研究对象，再判断需求、供给、机会与风险。\n\n"
        "## 1. 需求是否成立\n"
        "### 本节技术观察\n"
        "该证据来自公司官方问答，披露了具体的Agent产品功能和覆盖场景。\n"
        "该证据仅反映单一公司的产品部署情况，未提供用户规模。[1]\n\n"
        "## 来源附录\n"
        "- [1] 来源 | https://example.com/source\n"
    )

    cleaned = sanitize_public_markdown(markdown)
    body = cleaned.split("## 来源附录", 1)[0]

    for phrase in ["研究主线", "本节技术观察", "该证据来自", "该证据仅反映"]:
        assert phrase not in body
    assert "## 来源附录" in cleaned
    assert public_narrative_leak_audit(cleaned)["blocker_count"] == 0


def test_final_writer_excludes_diagnostic_global_blocks_from_public_report():
    result = run_final_writer_agent(
        query="AI Agent企业级落地与商业化验证",
        report_blueprint={
            "report_family": "industry_deep_report",
            "report_shell": {
                "front_blocks": ["policy_summary"],
                "back_blocks": ["execution_risks", "monitoring_indicators", "appendix"],
            },
        },
        chapter_packages=[
            {
                "chapter_id": "ch_01",
                "chapter_title": "需求是否成立",
                "sections": [
                    {
                        "section_id": "ch_01_s1",
                        "section_title": "客户部署是否出现",
                        "claim": "企业已经披露AI Agent在客户服务流程中的部署样本。[1]",
                        "reasoning": "该样本说明企业级部署开始从试点进入具体流程。",
                        "citation_refs": ["[1]"],
                        "evidence_refs": ["E1"],
                        "evidence_backed": True,
                    }
                ],
            }
        ],
        decision_package={
            "core_judgments": [{"judgment": "企业级AI Agent部署开始进入客户服务流程。"}],
            "watchlist": [{"metric": "后续观察本章相关的指标口径、企业披露和客户案例。"}],
        },
        risk_package={
            "risk_items": [
                {"risk_type": "执行边界风险", "severity": "high", "description": "样本数量不足。"}
            ]
        },
        source_registry=[{"ref": "[1]", "title": "来源一", "url": "https://www.stats.gov.cn/source"}],
    )

    markdown = result["report_markdown"]

    for phrase in ["政策摘要", "政策影响：", "执行风险", "监测指标", "应对：", "执行边界风险", "后续观察本章"]:
        assert phrase not in markdown
    assert "企业已经披露AI Agent在客户服务流程中的部署样本" in markdown
    assert "来源一" in markdown
    assert result["public_narrative_leak_audit"]["blocker_count"] == 0


def test_deep_report_quick_proof_mode_auto_upgrades(monkeypatch):
    monkeypatch.setenv("REPORT_PROOF_MODE", "quick_market_scan")
    monkeypatch.delenv("REPORT_ALLOW_QUICK_PROOF_FOR_DEEP", raising=False)

    assert _report_proof_mode({"report_family": "industry_deep_report"}, {}) == "deep_industry_report"

    monkeypatch.setenv("REPORT_ALLOW_QUICK_PROOF_FOR_DEEP", "true")
    assert _report_proof_mode({"report_family": "industry_deep_report"}, {}) == "quick_market_scan"


def test_title_only_source_is_not_traceable():
    title_only = _source_traceability_payload({"source": {"title": "只有标题"}})
    assert title_only["has_source_ref"] is False

    url_source = _source_traceability_payload({"source": {"title": "有URL", "url": "https://www.stats.gov.cn/source"}})
    assert url_source["has_source_ref"] is True


def test_ai_agent_ecosystem_query_selects_industry_deep_report():
    profile = select_report_profile("AI Agent生态发展报告：从工具到智能体的范式跃迁")
    assert profile["name"] == "industry_deep_report"


def test_reformatter_fallback_uses_distinct_writer_path():
    source = inspect.getsource(full_report.main)

    assert "write_markdown(clean_output_path, fallback_report)" not in source
    assert "write_markdown(clean_output_path, report_markdown)" not in source
    assert "_fallback_writer.md" in source
    assert '"fallback_output_path"' in source
    assert '"fallback_draft_path"' in source
    assert "clean_report_written = bool(" in source
