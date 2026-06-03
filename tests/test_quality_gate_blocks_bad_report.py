from rag_pipeline.agents.writer_agent_clean import (
    ENTERPRISE_INDUSTRY_LAYOUT,
    _hard_delivery_blockers,
    build_writer_report,
    validate_enterprise_report,
    validate_report_narrative_quality,
)
from rag_pipeline.agents.markdown_renderer import render_appendix, render_decision_package
from rag_pipeline.agents.qa_agent import run_qa_agent, validate_no_internal_gap_language
from rag_pipeline.flows.report.full_report import render_score_markdown, reformatter_structure_loss_reason


def test_quality_gate_blocks_final_status():
    bad_markdown = """
## 第二章 市场规模与增速
### 2.1 章节判断
### 2.2 证据深读
**原文事实**
市场规模约114亿元[3]
"""

    validation = validate_enterprise_report(
        markdown=bad_markdown,
        layout=ENTERPRISE_INDUSTRY_LAYOUT,
        chapter_packages=[],
        materials={},
        materials_payload={},
    )

    assert validation["passed"] is False
    assert any(item["type"] == "internal_label_or_template_phrase" for item in validation["errors"])


def test_empty_material_returns_diagnostic_only():
    result = build_writer_report(
        query="智能农业机器人",
        evidence_package={"analysis_ready_evidence": []},
        structured_analysis={"evidence_analyses": []},
    )

    assert result["report_status"] == "diagnostic_only"
    assert result["validation"]["passed"] is False
    assert result["report_markdown"] == ""


def test_narrative_gate_blocks_evidence_listing_style():
    bad_markdown = """
## 第二章 市场规模与增速

市场证据提供了规模、增速、出货等证据，需结合口径边界使用。

中国增速口径为17.2% / 246亿元 / 118亿元，更适合判断渗透节奏。后续判断的重点是确认该口径能否与客户付费相互印证。

同类数据存在口径差异，后续测算应先统一范围、年份和收入边界。进入综合决策章的变量：TAM、SAM、SOM、出货量。
"""

    validation = validate_report_narrative_quality(bad_markdown)

    assert validation["passed"] is False
    assert any(item["type"] == "evidence_listing_style" for item in validation["issues"])


def test_balanced_delivery_gate_keeps_directional_draft(monkeypatch):
    monkeypatch.setenv("REPORT_DELIVERY_GATE_MODE", "balanced")
    qa_result = {
        "passed": False,
        "quality_score": 56,
        "deep_evaluator_blocking": False,
        "deep_evaluation": {
            "blocking_gaps": [{"type": "missing_proof_standards"}],
        },
    }
    package_quality_report = {"warnings": [{"type": "low_ab_core_coverage"}]}
    coverage_matrix = [
        {
            "decision_ready": False,
            "claim_status": "directional",
            "actual_ab_sources": 0,
            "directional_c_sources": 2,
            "blocking_gaps": ["insufficient_ab_sources", "metric_evidence_missing"],
            "evidence_refs": ["EV-1"],
        }
    ]

    blockers = _hard_delivery_blockers(
        markdown="draft with insufficient_ab_sources marker",
        qa_result=qa_result,
        package_quality_report=package_quality_report,
        coverage_matrix=coverage_matrix,
    )

    assert blockers == []


def test_strict_delivery_gate_blocks_proof_and_qa_failures(monkeypatch):
    monkeypatch.setenv("REPORT_DELIVERY_GATE_MODE", "strict")
    qa_result = {
        "passed": False,
        "quality_score": 56,
        "deep_evaluator_blocking": True,
        "deep_evaluation": {
            "blocking_gaps": [{"type": "missing_proof_standards"}],
        },
    }
    package_quality_report = {"warnings": [{"type": "low_ab_core_coverage"}]}
    coverage_matrix = [
        {
            "decision_ready": False,
            "claim_status": "directional",
            "actual_ab_sources": 0,
            "directional_c_sources": 2,
            "blocking_gaps": ["insufficient_ab_sources", "metric_evidence_missing"],
            "evidence_refs": ["EV-1"],
        }
    ]

    blockers = _hard_delivery_blockers(
        markdown="draft with insufficient_ab_sources marker",
        qa_result=qa_result,
        package_quality_report=package_quality_report,
        coverage_matrix=coverage_matrix,
    )
    blocker_types = {item["type"] for item in blockers}

    assert "forbidden_public_text" in blocker_types
    assert "core_proof_gaps" not in blocker_types
    assert "low_ab_core_coverage" not in blocker_types
    assert "qa_not_passed" not in blocker_types


def test_qa_failed_stays_clean_blocker_but_allows_formal_render(monkeypatch):
    monkeypatch.setenv("QA_SCORING_MODE", "score")
    monkeypatch.setenv("QA_DEEP_EVALUATOR_BLOCKING", "true")
    report = (
        "# AI Agent report\n\n"
        "## Market signal\n\n"
        "Enterprise AI Agent adoption is still uneven, but available evidence [1] supports a directional discussion."
    )
    qa = run_qa_agent(
        report_markdown=report,
        report_blueprint={"report_family": "industry_deep_report"},
        chapter_packages=[
            {
                "chapter_id": "ch_01",
                "sections": [
                    {
                        "section_id": "s1",
                        "claim": "Enterprise adoption is directional rather than fully proven.",
                        "reasoning": "The evidence is limited and should lower claim strength.",
                        "counter_evidence": "Public verified counter samples remain insufficient.",
                        "actionable": "Track verified deployments and renewals.",
                        "evidence_refs": ["[1]"],
                    }
                ],
            }
        ],
    )

    assert qa["passed"] is False
    assert qa["clean_gate"]["eligible"] is False
    assert qa["render_gate"]["can_render_formal_report"] is True
    assert not qa["render_blocking_followups"]
    assert any(item.get("type") == "missing_sources_appendix" for item in qa["quality_findings"])


def test_qa_retrieval_gap_is_not_clean_blocker_when_evidence_exists(monkeypatch):
    monkeypatch.setenv("QA_SCORING_MODE", "score")
    monkeypatch.setenv("QA_DEEP_EVALUATOR_BLOCKING", "true")
    report = (
        "# AI Agent report\n\n"
        "## Market signal\n\n"
        "Enterprise adoption is evidenced by verified deployment signals [1].\n\n"
        "## 来源附录\n- [1] Source | https://www.stats.gov.cn/source"
    )
    qa = run_qa_agent(
        report_markdown=report,
        report_blueprint={"report_family": "industry_deep_report"},
        chapter_packages=[
            {
                "chapter_id": "ch_01",
                "chapter_fact_digest": ["verified fact"],
                "sections": [
                    {
                        "section_id": "s1",
                        "claim": "Enterprise adoption has early evidence.",
                        "reasoning": "Verified deployment signals support a cautious view.",
                        "counter_evidence": "Scale remains uneven.",
                        "actionable": "Track renewals.",
                        "evidence_refs": ["[1]"],
                    }
                ],
            }
        ],
        lane_coverage={"news_event": {"scheduled": 1, "succeeded": 0, "page_results": 0}},
        evidence_health_summary={"analysis_ready_count": 3, "distinct_verified_ab_source_count": 1},
    )

    assert qa["render_gate"]["can_render_formal_report"] is True
    assert not any(
        item.get("type") == "deep_report_blocking_gap"
        and item.get("detail", {}).get("type") in {"iqs_lane_no_success", "page_results_zero"}
        for item in qa["clean_gate"]["clean_blockers"]
    )
    assert any(
        item.get("type") == "deep_report_blocking_gap"
        and item.get("detail", {}).get("type") in {"iqs_lane_no_success", "page_results_zero"}
        and item.get("finding_category") == "readability_finding"
        for item in qa["quality_findings"]
    )


def test_qa_clean_candidate_gate_for_short_directional_report(monkeypatch):
    monkeypatch.setenv("QA_SCORING_MODE", "score")
    monkeypatch.setenv("QA_CLEAN_CANDIDATE_MIN_SCORE", "0")
    report = (
        "# AI Agent report\n\n"
        "## Directional signal\n\n"
        "Available B/C evidence supports only a directional view [1].\n\n"
        "## 来源附录\n- [1] Source | https://example.org/source"
    )
    qa = run_qa_agent(
        report_markdown=report,
        report_blueprint={"report_family": "industry_deep_report"},
        chapter_packages=[
            {
                "chapter_id": "ch_01",
                "sections": [
                    {
                        "section_id": "s1",
                        "claim": "Available B/C evidence supports only a directional view.",
                        "claim_strength": "directional",
                        "evidence_refs": ["[1]"],
                    }
                ],
            }
        ],
        evidence_health_summary={"analysis_ready_count": 1, "distinct_verified_ab_source_count": 0},
    )

    assert qa["passed"] is False
    assert qa["clean_gate"]["clean_content_eligible"] is False
    assert qa["clean_gate"]["clean_candidate_eligible"] is True


def test_score_report_separates_clean_content_from_output_switch():
    markdown = render_score_markdown(
        query="AI Agent",
        writer_report={
            "quality_score": 80,
            "quality_grade": "高质量但需人工复核",
            "clean_content_eligible": True,
            "clean_output_enabled": False,
            "clean_report_written": False,
            "clean_report_eligible": False,
            "clean_standard": "balanced",
            "report_status": "final_clean",
            "delivery_tier": "publishable_clean",
        },
        writer_package={},
        final_audit_result={"blocked": False, "status": "pass"},
        reformatter_result={"enabled": False, "status": "skipped"},
    )

    assert "Clean 资格：否" in markdown
    assert "Clean 内容资格：是" in markdown
    assert "Clean 标准：balanced" in markdown
    assert "Clean 输出开关：关闭" in markdown
    assert "Clean 文件已写出：否" in markdown


def test_score_report_exposes_analysis_llm_contract_diagnostics():
    markdown = render_score_markdown(
        query="AI Agent",
        writer_report={"quality_score": 60, "report_status": "formal_scored"},
        writer_package={
            "writer_report": {
                "render_artifacts": {
                    "payload_mode": "full",
                    "argument_units": [{"claim": "x"}],
                    "chapter_packages": [{"chapter_id": "ch_01"}],
                    "structured_analysis": {
                        "analysis_stage_diagnostics": {
                            "uses_llm_analysis": False,
                            "llm_analysis_status": "success_then_rebuilt",
                            "final_analysis_source": "deterministic_rebuild",
                            "deterministic_synthesis_used": True,
                            "llm_validation_status": "valid",
                            "llm_input_valid_ref_count": 4,
                            "llm_usable_claim_count": 2,
                            "llm_dropped_claim_count": 1,
                            "llm_usable_chapter_count": 1,
                        },
                        "analysis_contract_status": {"structured_analysis_valid": True},
                        "claim_units": [{"claim": "x"}],
                        "evidence_analyses": [{"evidence_id": "EV-1"}],
                    },
                }
            }
        },
        final_audit_result={"blocked": False, "status": "pass"},
        reformatter_result={"enabled": False, "status": "skipped"},
    )

    assert "final_analysis_source: deterministic_rebuild" in markdown
    assert "deterministic_synthesis_used: True" in markdown
    assert "llm_usable_claim_count: 2" in markdown
    assert "llm_dropped_claim_count: 1" in markdown


def test_public_renderer_hides_internal_reference_analysis_and_coverage_matrix(monkeypatch):
    monkeypatch.delenv("REPORT_RENDER_FINAL_REFERENCE_ANALYSIS", raising=False)
    monkeypatch.delenv("REPORT_RENDER_COVERAGE_MATRIX", raising=False)

    decision_markdown = render_decision_package(
        {
            "decision_thesis": "供应链正在从效率优先转向安全优先。",
            "chapter_syntheses": [
                {
                    "chapter_title": "供应链重构",
                    "chapter_summary": {
                        "key_takeaway": "区域化和友岸化正在抬高冗余产能权重。",
                        "mechanisms": ["出口管制改变设备可得性。"],
                        "what_to_verify_next": ["先进制程设备许可变化。"],
                    },
                }
            ],
        }
    )
    appendix_markdown = render_appendix(
        [],
        {
            "coverage_matrix": [
                {
                    "hypothesis_statement": "供应链区域化",
                    "actual_ab_sources": 0,
                    "required_ab_sources": 2,
                    "blocking_gaps": ["insufficient_ab_sources"],
                }
            ]
        },
    )

    combined = decision_markdown + "\n" + appendix_markdown
    assert "对应的章节结论是" not in combined
    assert "影响路径可以概括为" not in combined
    assert "证据覆盖矩阵" not in combined
    assert "insufficient_ab_sources" not in combined


def test_qa_flags_internal_process_language():
    errors = validate_no_internal_gap_language(
        "材料中最有解释力的事实组合是：X。后续变化主要集中在：Y。insufficient_ab_sources"
    )
    assert errors


def test_reformatter_condensed_publishable_report_does_not_fallback(monkeypatch):
    monkeypatch.setenv("REPORT_REFORMATTER_ALLOW_STRUCTURAL_CONDENSE", "true")
    monkeypatch.setenv("REPORT_REFORMATTER_MIN_DENSE_RATIO_PERCENT", "45")
    monkeypatch.setenv("REPORT_REFORMATTER_MIN_ACCEPT_CHARS", "12000")
    writer = "# 模板稿\n" + "\n".join(f"### 模板小节{i}\n" + ("证据清单。" * 220) for i in range(50))
    clean = "# 成稿\n" + "\n".join(f"## 成熟章节{i}\n" + ("自然分析段落。" * 620) for i in range(8))

    assert reformatter_structure_loss_reason(clean, writer) == ""
