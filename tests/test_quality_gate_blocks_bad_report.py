from rag_pipeline.agents.writer_agent_clean import (
    ENTERPRISE_INDUSTRY_LAYOUT,
    _hard_delivery_blockers,
    build_writer_report,
    validate_enterprise_report,
    validate_report_narrative_quality,
)
from rag_pipeline.agents.markdown_renderer import render_appendix, render_decision_package
from rag_pipeline.agents.qa_agent import validate_no_internal_gap_language
from rag_pipeline.flows.report.full_report import reformatter_structure_loss_reason


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


def test_failed_gate_returns_review_required():
    result = build_writer_report(
        query="智能农业机器人",
        evidence_package={"analysis_ready_evidence": []},
        structured_analysis={"evidence_analyses": []},
    )

    assert result["report_status"] == "review_required"
    assert result["validation"]["passed"] is False
    assert "质量门禁未通过" in result["message"]


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
    assert "core_proof_gaps" in blocker_types
    assert "low_ab_core_coverage" in blocker_types
    assert "qa_not_passed" in blocker_types


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
