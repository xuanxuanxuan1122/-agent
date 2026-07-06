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

    cleaned = sanitize_public_markdown(markdown, mode="enforce")

    assert "| 指标 | 数值 |" not in cleaned
    assert "**空表**" not in cleaned
    assert "后文" in cleaned


def test_public_sanitizer_is_diagnostic_only_by_default(monkeypatch):
    monkeypatch.delenv("REPORT_PUBLIC_SANITIZER_MUTATION_MODE", raising=False)
    markdown = "Body.\n\n**Empty table**\n\n| Metric | Value |\n| --- | --- |\n\nTail."

    cleaned = sanitize_public_markdown(markdown)

    assert cleaned == markdown


def test_finalize_public_report_enforces_public_sanitizer_at_publish_boundary():
    markdown = (
        "# 报告\n\n"
        "## 1. 行情判断\n"
        "证据不足，不能作为确定性结论，但价格和库存已经出现方向性变化。"
    )

    cleaned = full_report.finalize_public_report(markdown)

    assert "## 1. 行情判断" in cleaned
    assert "价格和库存已经出现方向性变化" in cleaned
    assert "证据不足" not in cleaned


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

    cleaned = sanitize_public_markdown(markdown, mode="enforce")
    body = cleaned.split("## 来源附录", 1)[0]

    for phrase in ["研究主线", "本节技术观察", "该证据来自", "该证据仅反映"]:
        assert phrase not in body
    assert "## 来源附录" in cleaned
    assert public_narrative_leak_audit(cleaned)["blocker_count"] == 0


def test_public_sanitizer_drops_review_boundary_sentence_but_keeps_analysis_and_citation():
    markdown = (
        "# \u62a5\u544a\n\n"
        "## 1. \u6559\u80b2\u57f9\u517b\u8c03\u6574\n"
        "\u56fd\u5bb6\u804c\u4e1a\u6559\u80b2\u6539\u9769\u5bfc\u5411\u4e0e\u884c\u4e1a\u8c03\u7814\u5171\u540c\u63a8\u52a8\u4f1a\u8ba1\u4e13\u4e1a\u57f9\u517b\u65b9\u6848\u5411\u201c\u5927\u6570\u636e+\u5408\u89c4\u98ce\u63a7\u201d\u65b9\u5411\u8fed\u4ee3\u3002"
        "\u8fd9\u8868\u660e\u6559\u80b2\u7aef\u5df2\u8bc6\u522b\u5230\u4f20\u7edf\u4f1a\u8ba1\u6280\u80fd\u4e0eAI\u65f6\u4ee3\u9700\u6c42\u7684\u65ad\u5c42\u3002"
        "\u4ec5\u53cd\u6620\u804c\u4e1a\u6559\u80b2\u4f53\u7cfb\u5185\u7684\u89c4\u5212\u52a8\u5411\uff0c"
        "\u672a\u8986\u76d6\u672c\u79d1\u53ca\u4ee5\u4e0a\u5b66\u672f\u578b\u4f1a\u8ba1\u6559\u80b2\u7684\u540c\u6b65\u8c03\u6574\u8282\u594f\u3002[1]\n\n"
        "## \u6765\u6e90\u9644\u5f55\n"
        "- [1] \u6765\u6e90 | https://example.com/source\n"
    )

    cleaned = sanitize_public_markdown(markdown, mode="enforce")
    body = cleaned.split("## \u6765\u6e90\u9644\u5f55", 1)[0]

    assert "\u5927\u6570\u636e+\u5408\u89c4\u98ce\u63a7" in body
    assert "\u9700\u6c42\u7684\u65ad\u5c42" in body
    assert "\u4ec5\u53cd\u6620" not in body
    assert "\u672a\u8986\u76d6" not in body
    assert "[1]" in body
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


def test_public_sanitizer_removes_template_scaffold_and_truncated_headings():
    template = (
        "\u5bf9\u201c\u5e02\u573a\u9700\u6c42\u201d\u8fd9\u4e00\u5224\u65ad\u800c\u8a00"
        "\uff0c\u5173\u952e\u4e0d\u53ea\u662f\u4e8b\u5b9e\u662f\u5426\u51fa\u73b0"
        "\uff0c\u800c\u662f\u5b83\u5982\u4f55\u6539\u53d8\u9700\u6c42\u5151\u73b0\u3002"
        "\u5982\u679c\u628a\u5b83\u653e\u5728\u62a5\u544a\u4e3b\u7ebf\u4e2d"
        "\uff0c\u8f83\u7a33\u59a5\u7684\u5199\u6cd5\u662f\u5148\u786e\u8ba4\u4e8b\u5b9e\u80fd\u591f\u652f\u6491\u7684\u6700\u4f4e\u7ed3\u8bba\u3002"
        "\u56e0\u6b64\uff0c\u8fd9\u4e00\u6bb5\u66f4\u9002\u5408\u4f5c\u4e3a\u6709\u8fb9\u754c\u7684\u5206\u6790\u4fe1\u53f7\u6765\u4f7f\u7528"
        "\uff0c\u907f\u514d\u5355\u4e00\u6750\u6599\u627f\u62c5\u8fc7\u5f3a\u7ed3\u8bba\u3002"
    )
    markdown = (
        "# Report\n\n"
        "## \u6838\u5fc3\u89c2\u70b9\u4e0e\u4e3b\u8981\u7ed3\u8bba\n"
        "- \u673a\u4f1a\u5224\u65ad\uff1a\u5e02\u573a\u89c4\u6a21\u9884\u6d4b\u663e\u793a\u9700\u6c42\u6b63\u5728\u6269\u5f20\u3002\n"
        "- \u673a\u4f1a\u5224\u65ad\uff1a\u5df2\u6709\u4f01\u4e1a\u8ba2\u5355\u548c\u573a\u666f\u843d\u5730\u4fe1\u53f7\u3002[1]\n\n"
        "## 1. \u9700\u6c42\u9a8c\u8bc1\n"
        "### 2025\u5e74\u4f4e\u7a7a\u7ecf\u6d4e\u89c4\u6a21\u9884\u8ba1\u8fbe8519\u4ebf\u5143\u5e76\u4e8e2026\u5e74. . . \uff0842\uff09\n"
        "\u5e02\u573a\u9884\u6d4b\u63d0\u4f9b\u4e86\u9700\u6c42\u6269\u5f20\u7684\u65b9\u5411\u6027\u4fe1\u53f7\u3002[1]"
        + template
        + "\n\n## \u6765\u6e90\u9644\u5f55\n- [1] Source | https://example.com/source\n"
    )

    cleaned = sanitize_public_markdown(markdown, mode="enforce")
    body = cleaned.split("## \u6765\u6e90\u9644\u5f55", 1)[0]

    assert ". . ." not in body
    assert "\uff0842\uff09" not in body
    assert "\u5173\u952e\u4e0d\u53ea\u662f\u4e8b\u5b9e\u662f\u5426\u51fa\u73b0" not in body
    assert "\u8f83\u7a33\u59a5\u7684\u5199\u6cd5" not in body
    assert "\u66f4\u9002\u5408\u4f5c\u4e3a\u6709\u8fb9\u754c\u7684\u5206\u6790\u4fe1\u53f7" not in body
    assert "\u5e02\u573a\u9884\u6d4b\u63d0\u4f9b\u4e86\u9700\u6c42\u6269\u5f20" in body
    assert "\u5e02\u573a\u89c4\u6a21\u9884\u6d4b\u663e\u793a\u9700\u6c42\u6b63\u5728\u6269\u5f20" not in body
    assert "\u5df2\u6709\u4f01\u4e1a\u8ba2\u5355\u548c\u573a\u666f\u843d\u5730\u4fe1\u53f7\u3002[1]" in body
