import re

from rag_pipeline.agents.final_writer_agent import run_final_writer_agent
from rag_pipeline.agents.public_report_sanitizer import sanitize_public_markdown
from rag_pipeline.agents.report_health import build_report_health_card
from rag_pipeline.flows.report.full_report import render_score_markdown


def test_report_health_card_marks_fact_passthrough_as_red():
    health = build_report_health_card(
        {
            "layout": {
                "snippet_like_text_dropped_count": 1,
                "evidence_backed_block_count": 1,
                "rendered_block_count": 6,
                "dropped_block_count": 8,
                "chapter_omitted_no_evidence_count": 1,
            },
            "chapter_evidence": {"total_valid_fact_card_count": 3},
            "analysis": {"final_analysis_source": "deterministic_rebuild"},
            "summary": {"executive_summary_valid_judgment_count": 0},
            "source_appendix_status": "ok",
            "body_composition_status": "fact_passthrough",
        }
    )

    assert health["overall_status"] == "red"
    assert health["metrics"]["body_composition_status"]["status"] == "red"
    assert health["metrics"]["evidence_backed_section_ratio"]["status"] == "red"


def test_report_health_card_exposes_body_rewrite_status():
    health = build_report_health_card(
        {
            "layout": {
                "layout_block_rendered_count": 2,
                "layout_block_evidence_backed_count": 2,
                "must_render_block_count": 2,
                "rendered_must_block_count": 2,
                "composer_variable_explanation_count": 2,
            },
            "chapter_evidence": {"total_valid_fact_card_count": 2},
            "analysis": {"final_analysis_source": "deterministic_rebuild"},
            "summary": {"executive_summary_valid_judgment_count": 1},
            "citation_manifest": {"citation_manifest_status": "ok"},
            "body_composition_status": "composed",
            "body_rewrite": {
                "enabled": True,
                "called_count": 2,
                "success_count": 1,
                "fallback_count": 1,
            },
        }
    )

    assert health["body_rewrite_status"] == "yellow"
    assert health["metrics"]["body_rewrite_status"]["status"] == "yellow"
    assert health["overall_status"] == "green"


def test_report_health_card_exposes_chapter_narrative_status():
    health = build_report_health_card(
        {
            "layout": {
                "layout_block_rendered_count": 6,
                "layout_block_evidence_backed_count": 6,
                "must_render_block_count": 6,
                "rendered_must_block_count": 6,
                "composer_variable_explanation_count": 6,
                "chapter_narrative": {
                    "enabled": True,
                    "attempted_count": 3,
                    "success_count": 2,
                    "fallback_count": 1,
                    "rejected_reasons": {"missing_required_citations": 1},
                },
            },
            "chapter_evidence": {"total_valid_fact_card_count": 6},
            "analysis": {"final_analysis_source": "llm_partial_merged"},
            "summary": {"executive_summary_valid_judgment_count": 1},
            "citation_manifest": {"citation_manifest_status": "ok"},
            "body_composition_status": "composed",
        }
    )

    assert health["chapter_narrative_status"] == "yellow"
    assert health["metrics"]["chapter_narrative_status"]["status"] == "yellow"
    assert health["chapter_narrative"]["fallback_count"] == 1


def test_report_health_card_marks_factual_body_without_citations_as_red():
    health = build_report_health_card(
        {
            "layout": {
                "layout_block_rendered_count": 2,
                "layout_block_evidence_backed_count": 2,
                "must_render_block_count": 2,
                "rendered_must_block_count": 2,
            },
            "chapter_evidence": {"total_valid_fact_card_count": 5},
            "analysis": {"final_analysis_source": "llm_partial_merged"},
            "summary": {"executive_summary_valid_judgment_count": 1},
            "citation_manifest": {"citation_manifest_status": "ok"},
            "final_citation_audit": {
                "final_citation_reconciliation_status": "blocked",
                "final_body_citation_refs": [],
                "final_appendix_refs": [],
                "final_missing_appendix_refs": [],
                "factual_body_without_citations_count": 2,
                "citationless_fact_examples": ["2025年 AI Agent 市场规模增长。"],
            },
            "body_composition_status": "composed",
        }
    )

    assert health["final_citation_status_after_render"] == "blocked"
    assert health["factual_body_without_citations_count"] == 2
    assert health["metrics"]["final_citation_status"]["status"] == "red"
    assert health["overall_status"] == "red"


def test_high_quality_density_gate_marks_thin_report_degraded():
    health = build_report_health_card(
        {
            "quality_mode": "high",
            "body_char_count": 2046,
            "h3_count": 2,
            "layout": {
                "layout_block_rendered_count": 3,
                "layout_block_evidence_backed_count": 3,
                "must_render_block_count": 4,
                "rendered_must_block_count": 3,
            },
            "chapter_evidence": {"total_valid_fact_card_count": 23},
            "analysis": {
                "final_analysis_source": "llm_partial_merged",
                "llm_usable_claim_count": 11,
                "llm_usable_chapter_count": 4,
            },
            "summary": {"executive_summary_valid_judgment_count": 1},
            "citation_manifest": {"citation_manifest_status": "ok"},
            "final_citation_audit": {"final_citation_reconciliation_status": "ok"},
            "body_composition_status": "composed",
        }
    )

    assert health["quality_path_degraded"] is True
    assert "body_chars_below_minimum" in health["quality_degraded_reasons"]
    assert "h3_count_below_minimum" in health["quality_degraded_reasons"]
    assert health["overall_status"] == "yellow"


def test_report_health_card_marks_enabled_rewrite_with_no_rewritable_sections():
    health = build_report_health_card(
        {
            "layout": {
                "layout_block_rendered_count": 0,
                "layout_block_evidence_backed_count": 0,
                "must_render_block_count": 2,
                "rendered_must_block_count": 0,
            },
            "chapter_evidence": {"total_valid_fact_card_count": 0},
            "analysis": {"final_analysis_source": "deterministic_rebuild"},
            "summary": {"executive_summary_valid_judgment_count": 0},
            "citation_manifest": {"citation_manifest_status": "ok"},
            "body_composition_status": "composed",
            "body_rewrite": {
                "enabled": True,
                "submitted_count": 0,
                "called_count": 0,
                "success_count": 0,
                "fallback_count": 0,
            },
        }
    )

    assert health["metrics"]["body_rewrite_status"]["value"] == "no_rewritable_sections"
    assert health["metrics"]["body_rewrite_status"]["status"] == "yellow"


def test_report_health_card_allows_composed_directional_report_as_yellow_or_green():
    health = build_report_health_card(
        {
            "layout": {
                "snippet_like_text_dropped_count": 0,
                "evidence_backed_block_count": 4,
                "rendered_block_count": 6,
                "dropped_block_count": 2,
                "chapter_omitted_no_evidence_count": 0,
            },
            "chapter_evidence": {"total_valid_fact_card_count": 6},
            "analysis": {"final_analysis_source": "deterministic_rebuild"},
            "summary": {"executive_summary_valid_judgment_count": 2},
            "source_appendix_status": "ok",
            "body_composition_status": "composed",
        }
    )

    assert health["overall_status"] in {"green", "yellow"}
    assert health["metrics"]["valid_fact_card_count"]["value"] == 6
    assert health["metrics"]["planned_vs_rendered_section_ratio"]["value"] > 0


def test_report_health_card_exposes_actionable_examples_and_composer_density():
    health = build_report_health_card(
        {
            "layout": {
                "must_render_block_count": 4,
                "rendered_must_block_count": 3,
                "layout_block_rendered_count": 3,
                "layout_block_evidence_backed_count": 3,
                "core_chapter_omitted_no_evidence_count": 1,
                "optional_chapter_omitted_count": 2,
                "block_drop_reason_examples": [
                    {"chapter_id": "ch_02", "block_type": "metric_reconciliation", "reason": "missing_metric_subject_or_scope"}
                ],
                "composer_variable_explanation_count": 3,
            },
            "chapter_evidence": {"total_valid_fact_card_count": 8},
            "source_appendix_status": "ok",
            "body_composition_status": "composed",
            "summary_valid_judgment_count": 1,
            "ref_lineage_diagnostics": {"filtered_refs": [{"ref": "EV-06-26", "reason": "unresolved_ref"}]},
        }
    )

    assert health["core_chapter_omitted_no_evidence_count"] == 1
    assert health["optional_chapter_omitted_count"] == 2
    assert health["composer_variable_explanation_count"] == 3
    assert health["block_drop_reason_examples"][0]["reason"] == "missing_metric_subject_or_scope"
    assert health["ref_lineage_diagnostics"]["filtered_refs"][0]["ref"] == "EV-06-26"


def test_final_writer_source_appendix_uses_claim_refs_not_only_body_citations(monkeypatch):
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
                        "claim": "Enterprise AI agents are appearing in customer-service workflow deployment.",
                        "reasoning": "The deployment signal matters because it connects user demand with operational workflows.",
                        "mechanism": "Workflow deployment is stronger than a demo because it requires integration and permission control.",
                        "evidence_refs": ["EV-1"],
                        "used_fact_refs": ["EV-1"],
                        "supporting_facts": ["Salesforce disclosed Agentforce customer-service workflow deployments."],
                        "evidence_backed": True,
                    }
                ],
                "table_packages": [],
            }
        ],
        source_registry=[
            {
                "ref": "[7]",
                "evidence_id": "EV-1",
                "title": "Salesforce Agentforce customer-service workflow deployments",
                "url": "https://www.salesforce.com/agentforce",
                "source_level": "B",
            }
        ],
    )

    markdown = output["report_markdown"]
    assert "Salesforce Agentforce customer-service workflow deployments" in markdown
    assert "##" in markdown and "[1]" in markdown


def test_score_markdown_includes_report_health_card():
    markdown = render_score_markdown(
        query="AI Agent",
        writer_report={"report_status": "formal_scored", "quality_score": 72},
        writer_package={
            "quality_posture": {
                "mode": "high",
                "disabled": {"query_rewrite": True, "self_refine": True},
                "query_rewrite_max_calls": "4",
                "query_rewrite_max_input_chars": "6000",
            },
            "raw_output": {
                "metadata": {
                    "repair_task_selection_summary": {
                        "task_count": 2,
                        "by_proof_role": {"metric": 1, "counter": 1},
                        "deep_budget_exhausted_count": 0,
                    },
                    "openai_web_search_summary": {
                        "gap_repair_task_count": 1,
                        "last_planned_by_proof_role": {"metric": 1},
                        "last_skip_reason": "budget_exhausted",
                    },
                }
            },
            "writer_report": {
                "render_artifacts": {
                    "chapter_packages": [
                        {
                            "chapter_id": "ch_01",
                            "sections": [
                                {
                                    "section_id": "s1",
                                    "block_type": "case_comparison",
                                    "evidence_backed": True,
                                    "composition_status": "composed",
                                    "body_composition_status": "composed",
                                    "body_rewrite_status": "rewritten",
                                    "body_rewrite": {
                                        "status": "rewritten",
                                        "llm_called": True,
                                        "cache_hit": False,
                                        "failure_reason": "",
                                    },
                                    "evidence_refs": ["[1]"],
                                    "supporting_facts": ["Salesforce disclosed Agentforce deployments."],
                                }
                            ],
                        }
                    ],
                    "chapter_narrative": {
                        "enabled": True,
                        "attempted_count": 1,
                        "success_count": 1,
                        "fallback_count": 0,
                    },
                    "micro_layouts": [
                        {
                            "chapter_id": "ch_01",
                            "sections": [{"section_id": "s1", "block_type": "case_comparison"}],
                        }
                    ],
                }
            },
            "structured_analysis": {"analysis_stage_diagnostics": {"final_analysis_source": "deterministic_rebuild"}},
        },
        final_audit_result={},
        reformatter_result={},
    )

    assert "Report HealthCard" in markdown
    assert "body_composition_status" in markdown
    assert "body_rewrite_status" in markdown
    assert "chapter_narrative_status" in markdown
    assert "quality_posture_mode: high" in markdown
    assert "query_rewrite_disabled: True" in markdown
    assert "Directed Evidence Repair" in markdown
    assert "repair_task_count_by_reason: {'metric': 1}" in markdown
    assert "selected_repair_task_count_by_reason: {'metric': 1, 'counter': 1}" in markdown
    assert "repair_budget_exhausted: True" in markdown
    assert "Chapter Narrative Diagnostics" in markdown
    assert "Body Rewrite Diagnostics" in markdown
    assert "body_rewrite_success_count" in markdown
    assert "evidence_backed_section_ratio" in markdown


def test_score_markdown_body_char_count_excludes_source_appendix():
    body = (
        "# AI Agent\u7814\u7a76\u62a5\u544a\n\n"
        "## 1. \u9700\u6c42\u9a8c\u8bc1\n"
        "### \u5ba2\u6237\u90e8\u7f72\u4fe1\u53f7\n"
        + "\u4f01\u4e1a\u7ea7 AI Agent \u5df2\u8fdb\u5165\u5ba2\u6237\u6d41\u7a0b\u3002[1]" * 4
        + "\n\n"
    )
    appendix = (
        "## \u6765\u6e90\u9644\u5f55\n"
        + "- [1] \u6765\u6e90A | https://example.org/a\n" * 80
    )
    markdown = render_score_markdown(
        query="AI Agent",
        writer_report={
            "report_status": "formal_scored",
            "quality_score": 80,
            "report_markdown": body + appendix,
            "target_body_chars": 20000,
        },
        writer_package={},
        final_audit_result={},
        reformatter_result={},
    )

    expected_body_chars = len(re.sub(r"\s+", "", body))
    assert f"- body_char_count: {expected_body_chars}" in markdown


def test_score_markdown_counts_global_cached_body_rewrites_as_successes():
    markdown = render_score_markdown(
        query="AI Agent",
        writer_report={"report_status": "formal_scored", "quality_score": 80},
        writer_package={
            "writer_report": {
                "render_artifacts": {
                    "chapter_packages": [
                        {
                            "chapter_id": "ch_01",
                            "body_rewrite_global": {
                                "enabled": True,
                                "submitted_count": 0,
                                "called_count": 0,
                                "success_count": 4,
                                "cache_hit_count": 4,
                                "rejected_count": 0,
                                "fallback_count": 0,
                                "skipped_count": 3,
                                "failure_reasons": {"not_composed": 3},
                            },
                            "sections": [
                                {
                                    "section_id": "s1",
                                    "block_type": "case_comparison",
                                    "evidence_backed": True,
                                    "body_composition_status": "composed",
                                    "body_rewrite_status": "cached",
                                    "body_rewrite": {
                                        "status": "cached",
                                        "cache_hit": True,
                                        "llm_called": False,
                                    },
                                    "evidence_refs": ["[1]"],
                                    "supporting_facts": ["Agent deployment fact."],
                                }
                            ],
                        }
                    ],
                    "micro_layouts": [{"chapter_id": "ch_01", "blocks": [{"block_type": "case_comparison"}]}],
                }
            }
        },
        final_audit_result={},
        reformatter_result={},
    )

    assert "body_rewrite_success_count: 4" in markdown
    assert "body_rewrite_cache_hit_count: 4" in markdown


def test_public_sanitizer_normalizes_ocr_artifacts_and_empty_punctuation():
    markdown = "## 章节\n\n中⼼心发布报告（）。管理理团队提到⼤大模型应⽤用落地.。"

    cleaned = sanitize_public_markdown(markdown)

    assert "中⼼心" not in cleaned
    assert "管理理" not in cleaned
    assert "⼤大" not in cleaned
    assert "应⽤用" not in cleaned
    assert "（）" not in cleaned
    assert ".。" not in cleaned
