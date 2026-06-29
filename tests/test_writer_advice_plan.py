from __future__ import annotations

from rag_pipeline.agents.writer_agent_clean import (
    _apply_writer_advice_to_argument_units,
    build_writer_advice_plan,
    build_writer_report,
)


def test_writer_advice_plan_turns_repair_context_into_non_renderable_actions():
    structured_analysis = {
        "post_qa_repair_context": {
            "rewrite_required": True,
            "rewrite_reasons": [
                {
                    "schema_version": "repair_execution_suggestion_v1",
                    "issue_type": "body_short",
                    "repair_action": "rewrite_with_caveat",
                    "repair_route": "rewrite_with_caveat",
                    "target": {"chapter_id": "CH_market", "claim_id": "CL1"},
                    "diagnostic_only": True,
                    "must_not_render": True,
                    "public_text_allowed": False,
                },
                {
                    "schema_version": "repair_execution_suggestion_v1",
                    "issue_type": "chapter_mismatch",
                    "repair_action": "recompose_outline",
                    "repair_route": "recompose_outline",
                    "target": {"chapter_id": "plan_empty"},
                    "diagnostic_only": True,
                    "must_not_render": True,
                    "public_text_allowed": False,
                },
            ],
        },
        "claim_units": [
            {
                "claim_id": "CL1",
                "chapter_id": "CH_market",
                "claim": "Enterprise AI agent adoption is an early but visible signal.",
                "fact_ids": ["F1"],
                "source_ids": ["S1"],
                "claim_strength": "directional",
            }
        ],
    }

    plan = build_writer_advice_plan(
        structured_analysis=structured_analysis,
        report_plan={},
        writer_report=None,
    )

    assert plan["schema_version"] == "writer_advice_plan_v1"
    assert plan["diagnostic_only"] is True
    assert plan["must_not_render"] is True
    assert plan["public_text_allowed"] is False
    assert plan["summary"]["action_count"] >= 2
    assert plan["summary"]["expand_claim_writing_count"] == 1
    assert plan["summary"]["recompose_outline_count"] == 1
    assert plan["claim_actions"][0]["claim_id"] == "CL1"
    assert plan["claim_actions"][0]["action"] == "expand_claim_writing"
    assert plan["claim_actions"][0]["tone"] == "directional"
    assert plan["claim_actions"][0]["must_preserve_fact_refs"] is True
    assert plan["chapter_actions"][0]["chapter_id"] == "CH_market"
    assert plan["chapter_actions"][0]["action"] == "expand_with_caveat"
    assert all(item["must_not_render"] is True for item in plan["claim_actions"])
    assert all(item["public_text_allowed"] is False for item in plan["chapter_actions"])


def test_writer_advice_marks_argument_units_without_mutating_original():
    units = [
        {
            "claim_id": "CL1",
            "chapter_id": "CH_market",
            "claim": "Adoption is an early signal.",
            "claim_strength": "moderate",
            "used_fact_refs": ["EV-1"],
        }
    ]
    advice_plan = {
        "claim_actions": [
            {
                "claim_id": "CL1",
                "chapter_id": "CH_market",
                "action": "expand_claim_writing",
                "tone": "directional",
                "must_preserve_fact_refs": True,
                "do_not_add_new_facts": True,
                "must_not_render": True,
                "public_text_allowed": False,
            }
        ],
        "chapter_actions": [],
    }

    updated = _apply_writer_advice_to_argument_units(units, advice_plan)

    assert units[0].get("writer_advice_actions") is None
    assert updated[0]["writer_advice_actions"][0]["action"] == "expand_claim_writing"
    assert updated[0]["writer_advice_tone"] == "directional"
    assert updated[0]["writer_advice_do_not_add_new_facts"] is True
    assert updated[0]["claim_strength"] == "directional"
    assert updated[0]["used_fact_refs"] == ["EV-1"]


def test_writer_advice_applies_chapter_actions_to_matching_units():
    units = [
        {
            "claim_id": "CL1",
            "chapter_id": "CH_market",
            "claim": "Adoption is an early signal.",
            "claim_strength": "directional",
            "used_fact_refs": ["EV-1"],
        },
        {
            "claim_id": "CL2",
            "chapter_id": "CH_policy",
            "claim": "Policy support is visible.",
            "claim_strength": "directional",
            "used_fact_refs": ["EV-2"],
        },
    ]
    advice_plan = {
        "claim_actions": [],
        "chapter_actions": [
            {
                "chapter_id": "CH_market",
                "action": "reanalyze_existing",
                "issue_type": "section_ref_binding_invalid_only",
                "tone": "structural",
                "must_preserve_fact_refs": True,
                "do_not_add_new_facts": True,
                "must_not_render": True,
                "public_text_allowed": False,
            }
        ],
    }

    updated = _apply_writer_advice_to_argument_units(units, advice_plan)

    assert updated[0]["writer_advice_actions"][0]["action"] == "reanalyze_existing"
    assert updated[0]["writer_advice_actions"][0]["issue_type"] == "section_ref_binding_invalid_only"
    assert updated[0]["writer_advice_do_not_add_new_facts"] is True
    assert "writer_advice_actions" not in updated[1]


def test_writer_report_exposes_advice_plan_without_rendering_internal_terms(monkeypatch):
    monkeypatch.setenv("REPORT_BLUEPRINT_SOURCE", "claim_first")
    monkeypatch.setenv("REPORT_TABLES_ENABLED", "false")
    monkeypatch.setenv("REPORT_ENABLE_LLM_BODY_REWRITE", "false")

    report = build_writer_report(
        query="AI Agent adoption",
        report_blueprint={
            "chapters": [
                {"chapter_id": "CH_market", "chapter_title": "Market signals"},
            ]
        },
        evidence_package={
            "analysis_ready_evidence": [
                {
                    "evidence_id": "F1",
                    "source_id": "S1",
                    "data_point": "Enterprise workflow adoption signal",
                    "source": {"url": "https://example.com/report"},
                }
            ],
            "source_registry": [{"source_id": "S1", "ref": "[1]", "url": "https://example.com/report"}],
        },
        structured_analysis={
            "post_qa_repair_context": {
                "rewrite_required": True,
                "rewrite_reasons": [
                    {
                        "schema_version": "repair_execution_suggestion_v1",
                        "issue_type": "body_short",
                        "repair_action": "rewrite_with_caveat",
                        "repair_route": "rewrite_with_caveat",
                        "target": {"chapter_id": "CH_market", "claim_id": "CL1"},
                        "diagnostic_only": True,
                        "must_not_render": True,
                        "public_text_allowed": False,
                    }
                ],
            },
            "claim_units": [
                {
                    "claim_id": "CL1",
                    "claim": "Enterprise workflow adoption is a directional market signal.",
                    "chapter_id": "CH_market",
                    "cluster_key": "market",
                    "fact_ids": ["F1"],
                    "source_ids": ["S1"],
                    "used_fact_refs": ["F1"],
                    "claim_strength": "moderate",
                    "can_anchor_section": True,
                    "claim_review_suggestions": [
                        {
                            "schema_version": "review_suggestion_v1",
                            "issue_type": "section_ref_binding_invalid_only",
                            "suggested_action": "reanalyze_existing",
                            "target": {"chapter_id": "CH_market", "claim_id": "CL1", "section_id": "CH_market_s1"},
                            "diagnostic_only": True,
                            "must_not_render": True,
                            "public_text_allowed": False,
                        }
                    ],
                }
            ],
        },
        source_registry=[{"source_id": "S1", "ref": "[1]", "url": "https://example.com/report"}],
    )

    assert report["writer_advice_plan"]["summary"]["expand_claim_writing_count"] == 1
    assert report["writer_advice_plan"]["summary"]["reanalyze_existing_count"] >= 1
    assert report["metadata"]["writer_advice_summary"]["action_count"] >= 2
    assert any(
        item.get("issue_type") == "section_ref_binding_invalid_only"
        and item.get("repair_action") == "reanalyze_existing"
        for item in report.get("required_followups", [])
    )
    rendered_text = str(report.get("report_markdown") or "")
    for token in ("rewrite_with_caveat", "repair_execution_suggestion", "diagnostic_only", "writer_advice_plan"):
        assert token not in rendered_text


def test_writer_report_rebuilds_advice_after_fallback_claim_builder(monkeypatch):
    monkeypatch.setenv("REPORT_BLUEPRINT_SOURCE", "claim_first")
    monkeypatch.setenv("REPORT_TABLES_ENABLED", "false")
    monkeypatch.setenv("REPORT_ENABLE_LLM_BODY_REWRITE", "false")

    report = build_writer_report(
        query="Accounting employment",
        report_blueprint={
            "chapters": [
                {"chapter_id": "ch_01", "chapter_title": "Employment signals"},
            ]
        },
        evidence_package={
            "source_registry": [{"source_id": "S1", "ref": "[1]", "url": "https://example.com/accounting"}],
        },
        structured_analysis={},
        chapter_evidence_packages=[
            {
                "chapter_id": "ch_01",
                "chapter_title": "Employment signals",
                "core_evidence": [],
                "supporting_evidence": [],
            }
        ],
        micro_layouts=[
            {
                "chapter_id": "ch_01",
                "sections": [
                    {
                        "section_id": "ch_01_s1",
                        "section_title": "Employment demand",
                        "block_type": "integrated_signal",
                        "output_type": "integrated_signal",
                        "section_role": "integrated_signal",
                        "required_evidence_refs": ["EV-stale"],
                    }
                ],
            }
        ],
        source_registry=[{"source_id": "S1", "ref": "[1]", "url": "https://example.com/accounting"}],
    )

    summary = report["writer_advice_plan"]["summary"]
    assert summary["reanalyze_existing_count"] >= 1
    assert any(
        item.get("issue_type") == "section_ref_binding_invalid_only"
        and item.get("repair_action") == "reanalyze_existing"
        for item in report.get("required_followups", [])
    )
    units = report["render_artifacts"]["argument_units"]
    assert any(unit.get("section_ref_binding_status") == "invalid_only" for unit in units)
