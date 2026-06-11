import json
import os

import rag_pipeline.agents.analysis_agent as analysis_agent
import rag_pipeline.agents.analysis_agent as analysis_agent
from rag_pipeline.flows.report import full_report
from rag_pipeline.search.memory import LLMCallError

from rag_pipeline.agents.analysis_agent import (
    _chapter_evidence_diagnostics,
    _chapter_insights_from_synthesis,
    _evidence_analysis,
    _evidence_cards_for_llm,
    _build_mechanism_chain,
    _hypothesis_insights,
    _reasoning_from_public_facts,
    analysis_depth_quality,
    build_llm_analysis_input,
    build_llm_analysis_input_v2,
    ensure_valid_structured_analysis,
    merge_llm_analysis_with_fallback,
    resolve_chapter_id,
    run_analysis_agent,
    validate_llm_analysis_output,
)
from rag_pipeline.agents.claim_builder_agent import (
    _matches as claim_builder_matches,
    _norm_chapter_id,
    run_claim_builder_agent,
)
from rag_pipeline.agents.chapter_argument_agent import run_chapter_argument_agent
from rag_pipeline.agents.final_writer_agent import (
    _normalize_citation_ref,
    _source_allowed_for_report,
    _strip_orphan_citations,
)
from rag_pipeline.agents.report_contracts import text_has_factual_claim
from rag_pipeline.agents.readpage_fact_extractor_agent import _validated_card
from rag_pipeline.agents.chapter_evidence_builder import _public_fact_quality
from rag_pipeline.agents.markdown_renderer import render_table_package


def _evidence(ref="EV-1", *, chapter_id="ch_01", level="B", allowed_use="core_claim", proof_role="support"):
    return {
        "evidence_id": ref,
        "ref": ref,
        "chapter_id": chapter_id,
        "fact": "Enterprise agent deployments are moving from pilots into workflow automation.",
        "source_level": level,
        "allowed_use": allowed_use,
        "proof_role": proof_role,
        "source_verification_status": "readpage_verified",
        "source": {
            "title": "Verified source",
            "url": f"https://example.org/{ref}",
            "source_verification_status": "readpage_verified",
        },
    }


def test_factual_detector_does_not_block_opportunity_framing_line():
    framing = "\u673a\u4f1a\u5224\u65ad\uff1aOpenAI \u4e0e Microsoft \u7684\u6280\u672f\u3001\u4f9b\u5e94\u548c\u76d1\u7ba1\u7ea6\u675f\u4f1a\u5982\u4f55\u6539\u53d8\u673a\u4f1a\u6392\u5e8f"
    factual = "\u6e17\u900f\u7387\u4e3a10%\uff0c\u671f\u95f4\u4e3a2011\u5e74"

    assert text_has_factual_claim(framing) is False
    assert text_has_factual_claim(factual) is True


def test_factual_detector_does_not_treat_enterprise_or_institution_as_fact_alone():
    assert text_has_factual_claim("\u4f01\u4e1a\u843d\u5730\u7684\u6838\u5fc3\u95ee\u9898\u5728\u4e8e\u6d41\u7a0b\u6539\u9020\u3002") is False
    assert text_has_factual_claim("\u673a\u6784\u5ba2\u6237\u66f4\u5173\u6ce8\u79c1\u6709\u5316\u90e8\u7f72\u3002") is False
    assert text_has_factual_claim("\u4f01\u4e1a2025\u5e74\u8425\u6536\u589e\u957f\u3002") is True


def test_public_fact_card_v2_rejects_isolated_metric_and_navigation_text():
    isolated_metric = {
        "evidence_id": "EV-METRIC",
        "fact": "AI Agent adoption: 50%",
        "metric": "adoption",
        "value": "50%",
        "unit": "%",
        "source_level": "B",
        "source_url": "https://www.salesforce.com/news/agent-adoption",
        "source_verification_status": "readpage_verified",
    }
    navigation_text = {
        "evidence_id": "EV-NAV",
        "fact": "Skip to content Product Solutions Resources Login Contact us",
        "source_level": "B",
        "source_url": "https://www.salesforce.com/news/navigation",
        "source_verification_status": "readpage_verified",
    }

    assert _public_fact_quality(isolated_metric)["eligible_for_report"] is False
    assert "no_subject_or_scope" in _public_fact_quality(isolated_metric)["rejection_reason"]
    assert _public_fact_quality(navigation_text)["eligible_for_report"] is False
    assert "navigation_text" in _public_fact_quality(navigation_text)["rejection_reason"]


def test_build_fallback_analysis_does_not_select_metric_fragment_as_report_thesis(monkeypatch):
    def fake_claim_units(_dimension_synthesis):
        return [
            {
                "question": "\u6307\u6807\u7247\u6bb5",
                "claim": "\u6e17\u900f\u7387\uff1b2023\u5e74",
                "claim_strength": "strong",
                "supporting_evidence": ["EV-BAD"],
                "evidence_refs": ["EV-BAD"],
                "confidence": 0.99,
            },
            {
                "question": "\u9700\u6c42\u9a8c\u8bc1",
                "claim": (
                    "\u4f01\u4e1a\u7ea7 AI Agent \u7684\u9700\u6c42\u6b63\u5728\u4ece\u5de5\u5177\u8bd5\u7528"
                    "\u8f6c\u5411\u4e1a\u52a1\u90e8\u7f72\uff0c\u4f46\u4ed8\u8d39\u6df1\u5ea6\u4ecd\u53d6\u51b3\u4e8e ROI "
                    "\u4e0e\u6743\u9650\u6cbb\u7406\u3002"
                ),
                "claim_strength": "moderate",
                "supporting_evidence": ["EV-GOOD"],
                "evidence_refs": ["EV-GOOD"],
                "confidence": 0.8,
            },
        ]

    monkeypatch.setattr(analysis_agent, "_claim_units_from_synthesis", fake_claim_units)

    result = analysis_agent.build_fallback_analysis(
        {
            "query": "AI Agent",
            "per_dimension": {"\u7efc\u5408\u5224\u65ad": []},
        }
    )

    insight = result["report_insight_package"]
    assert insight["report_thesis"] != "\u6e17\u900f\u7387\uff1b2023\u5e74"
    assert "\u4f01\u4e1a\u7ea7 AI Agent" in insight["report_thesis"]
    assert all(
        item.get("judgment") != "\u6e17\u900f\u7387\uff1b2023\u5e74"
        for item in insight["executive_summary"]["top_3_judgments"]
    )


def test_render_table_package_filters_diagnostic_metric_table():
    table = {
        "title": "市场指标与口径表",
        "should_render": True,
        "headers": ["指标", "范围", "期间", "数值", "单位", "后续影响"],
        "rows": [
            {"cells": ["渗透率", "", "2023年", "达到100%", "%", "该指标须同时披露范围、期间、单位与来源等级,才进入正文判断。"]},
            {"cells": ["份额", "", "2024年", "约1%", "%", "该指标须同时披露范围、期间、单位与来源等级,才进入正文判断。"]},
            {"cells": ["市场份额", "", "2026年", "35.8%", "%", "该指标须同时披露范围、期间、单位与来源等级,才进入正文判断。"]},
        ],
        "takeaway": "表中各行来自已有证据。",
        "limitations": ["使用边界：表中各行来自已有证据,不会凭空补齐缺失的范围、单位或期间。"],
    }

    assert render_table_package(table) == ""


def test_chapter_argument_requires_mechanism_for_evidence_backed_section():
    packages = run_chapter_argument_agent(
        report_blueprint={"chapters": [{"chapter_id": "ch_01", "title": "Demand validation"}]},
        micro_layouts=[
            {
                "chapter_id": "ch_01",
                "sections": [{"section_id": "s1", "block_type": "case_comparison", "section_title": "Customer signal"}],
            }
        ],
        argument_units=[
            {
                "chapter_id": "ch_01",
                "section_id": "s1",
                "block_type": "case_comparison",
                "claim": "Customer deployments exist.",
                "supporting_facts": ["Salesforce disclosed Agentforce usage in customer service workflows."],
                "used_fact_refs": ["S1"],
                "evidence_refs": ["S1"],
                "fact_card_to_block_match": True,
                "claim_strength": "moderate",
            }
        ],
    )

    assert packages[0]["sections"] == []
    assert any(item.get("reason") in {"not_public", "layout_section_without_public_evidence"} for item in packages[0]["dropped_sections"])


def test_structured_analysis_needs_rewrite_triggers_evidence_rebuild():
    evidence_package = {
        "query": "AI Agent industry report",
        "analysis_ready_evidence": [_evidence()],
        "chapter_evidence_packages": [
            {
                "chapter_id": "ch_01",
                "chapter_title": "Demand validation",
                "core_evidence": [_evidence()],
            }
        ],
    }
    structured = {
        "claim_units": [{"chapter_id": "ch_01", "claim": "generic", "evidence_refs": []}],
        "chapter_insights": [{"chapter_id": "ch_01", "chapter_question": "Demand validation"}],
        "evidence_analyses": [{"evidence_id": "EV-1", "fact": "x"}],
        "analysis_depth_quality": {"status": "needs_rewrite", "repeated_claim_ratio": 0.95},
        "claim_binding_feedback_summary": {"available_ab_not_bound_count": 1},
    }

    result = ensure_valid_structured_analysis(structured, evidence_package)

    assert result["analysis_rebuilt_from_evidence"] is True
    assert "needs_rewrite_quality" in result["analysis_contract_status"]["quality_rebuild_reasons"]
    assert result["analysis_stage_diagnostics"]["analysis_rebuilt_from_evidence"] is True


def test_rebuild_preserves_llm_claim_repair_priorities():
    evidence_package = {
        "query": "AI Agent industry report",
        "analysis_ready_evidence": [_evidence("EV-1")],
        "chapter_evidence_packages": [
            {
                "chapter_id": "ch_01",
                "chapter_title": "Demand validation",
                "core_evidence": [_evidence("EV-1")],
            }
        ],
    }
    repair_priority = {
        "schema_version": "claim_support_repair_priority_v1",
        "gap_id": "ch_01_bad_claim_claim_support_entity_or_metric_mismatch",
        "gap_type": "claim_support_entity_or_metric_mismatch",
        "chapter_id": "ch_01",
        "claim_id": "bad_claim",
        "claim": "Unsupported market-size claim",
        "required_fields": ["source", "metric", "value", "unit", "period"],
        "allowed_for_writing": False,
    }
    structured = {
        "claim_units": [{"chapter_id": "ch_01", "claim": "generic", "evidence_refs": []}],
        "chapter_insights": [{"chapter_id": "ch_01", "chapter_question": "Demand validation"}],
        "evidence_analyses": [{"evidence_id": "EV-1", "fact": "x"}],
        "analysis_depth_quality": {"status": "needs_rewrite", "repeated_claim_ratio": 0.95},
        "claim_binding_feedback_summary": {"available_ab_not_bound_count": 1},
        "evidence_repair_priorities": [repair_priority],
        "evidence_gap_ledger": [],
        "llm_analysis_synthesis": {
            "chapter_synthesis": [],
            "evidence_repair_priorities": [repair_priority],
            "validation": {
                "status": "invalid_output_no_usable_claims",
                "claim_repair_priorities": [repair_priority],
            },
        },
    }

    result = ensure_valid_structured_analysis(structured, evidence_package)

    assert result["analysis_rebuilt_from_evidence"] is True
    assert result["evidence_repair_priorities"][0]["gap_id"] == repair_priority["gap_id"]
    preserved_gap = [
        item
        for item in result["evidence_gap_ledger"]
        if item.get("gap_id") == repair_priority["gap_id"]
    ][0]
    assert preserved_gap["gap_type"] == "claim_support_entity_or_metric_mismatch"
    assert preserved_gap["source_stage"] == "analysis_claim_support"
    assert preserved_gap["allowed_for_writing"] is False


def test_valid_llm_claims_are_not_rebuilt_due_to_legacy_binding_warnings():
    structured = {
        "claim_units": [
            {
                "chapter_id": "ch_01",
                "claim": "AI Agent demand is moving from pilots into workflow deployment.",
                "evidence_refs": ["EV-1"],
                "supporting_evidence": ["EV-1"],
                "reasoning": "Workflow deployments indicate operational adoption.",
                "counter_evidence": "The conclusion is limited to disclosed samples.",
            }
        ],
        "chapter_insights": [
            {
                "chapter_id": "ch_01",
                "chapter_question": "Demand validation",
                "key_claims": [
                    {
                        "claim": "AI Agent demand is moving from pilots into workflow deployment.",
                        "evidence_refs": ["EV-1"],
                        "supporting_evidence": ["EV-1"],
                    }
                ],
            }
        ],
        "evidence_analyses": [{"evidence_id": "EV-1", "fact": "x"}],
        "llm_analysis_synthesis": {
            "validation": {
                "status": "valid",
                "usable_claim_count": 1,
            }
        },
        "claim_binding_feedback_summary": {"available_ab_not_bound_count": 1},
        "analysis_depth_quality": {"status": "ok", "repeated_claim_ratio": 0.0},
    }

    result = ensure_valid_structured_analysis(structured, {"analysis_ready_evidence": [_evidence("EV-1")]})

    assert result.get("analysis_rebuilt_from_evidence") is not True
    assert result["claim_units"][0]["claim"].startswith("AI Agent demand")
    assert "chapter_binding_failed" in result["analysis_contract_status"]["quality_only_warnings"]
    assert "unbound_ab_evidence" in result["analysis_contract_status"]["quality_only_warnings"]


def test_llm_claim_without_refs_is_rejected_instead_of_inferred():
    evidence_package = {"analysis_ready_evidence": [_evidence("EV-22")]}
    payload = {
        "chapter_synthesis": [
            {
                "chapter_id": "ch_01",
                "chapter_title": "Demand validation",
                "fact_chain": ["Enterprise workflow evidence is present (EV-22)."],
                "evidence_refs": ["EV-22"],
                "claim_units": [
                    {
                        "claim": "目前只能形成方向性观察，需要用可追溯来源和连续指标继续校准结论强度。",
                        "claim_status": "directional",
                        "supporting_evidence_refs": [],
                        "reasoning": "",
                    }
                ],
            }
        ]
    }

    validation = validate_llm_analysis_output(payload, evidence_package)
    assert validation["status"] == "invalid_output_no_usable_claims"
    assert validation["chapter_synthesis"] == []
    assert any(item["type"] == "llm_claim_missing_used_evidence_ids" for item in validation["issues"])


def test_llm_claim_with_valid_used_evidence_ids_is_accepted():
    evidence_package = {"analysis_ready_evidence": [_evidence("EV-22")]}
    payload = {
        "chapter_synthesis": [
            {
                "chapter_id": "ch_01",
                "chapter_title": "Demand validation",
                "claim_units": [
                    {
                        "claim": "Enterprise agents are moving into workflow automation.",
                        "claim_status": "decision_ready",
                        "claim_strength": "moderate",
                        "used_evidence_ids": ["EV-22"],
                        "evidence_basis": ["Enterprise workflow evidence is present."],
                        "reasoning_chain": ["Workflow deployments indicate operational adoption."],
                        "limitation_boundary": ["The claim still depends on comparable customer samples."],
                    }
                ],
            }
        ]
    }

    validation = validate_llm_analysis_output(payload, evidence_package)
    unit = validation["chapter_synthesis"][0]["claim_units"][0]

    assert validation["status"] == "valid"
    assert unit["supporting_evidence_refs"] == ["EV-22"]
    assert unit["reasoning"] == "Workflow deployments indicate operational adoption."


def test_llm_claim_with_unsupported_entity_and_metric_is_deferred_for_repair():
    evidence_package = {
        "analysis_ready_evidence": [
            {
                **_evidence("EV-GTC", chapter_id="ch_02", level="A"),
                "fact": "NVIDIA announced GTC sessions for enterprise AI infrastructure.",
                "distilled_fact": "NVIDIA announced GTC sessions for enterprise AI infrastructure.",
                "source_title": "NVIDIA GTC",
            }
        ]
    }
    payload = {
        "chapter_synthesis": [
            {
                "chapter_id": "ch_02",
                "claim_units": [
                    {
                        "claim_id": "bad-price-claim",
                        "claim": "MiniMax M2.5 API price is about 1/15 of Claude Opus 4.6.",
                        "claim_strength": "directional",
                        "used_evidence_ids": ["EV-GTC"],
                        "evidence_basis": ["NVIDIA announced GTC sessions for enterprise AI infrastructure."],
                        "reasoning": "The cited evidence discusses NVIDIA infrastructure events.",
                    }
                ],
            }
        ]
    }

    validation = validate_llm_analysis_output(payload, evidence_package)

    assert validation["usable_claim_count"] == 0
    assert validation["deferred_claim_count"] == 1
    assert validation["dropped_claim_count"] == 0
    assert validation["status"] == "invalid_output_no_usable_claims"
    assert validation["llm_validation_issue_counts"]["claim_support_needs_repair"] == 1
    assert validation["llm_deferred_claim_examples"][0]["unsupported_terms"]
    repair = validation["claim_repair_priorities"][0]
    assert repair["claim_id"] == "bad-price-claim"
    assert repair["gap_type"] == "claim_support_entity_or_metric_mismatch"
    assert repair["writing_permission"] == "not_allowed_until_repaired"
    assert repair["evidence_refs"] == ["EV-GTC"]
    assert validation["correctness_filter_summary"]["thin_report_risk"] is True
    assert validation["correctness_filter_summary"]["recommended_mode"] == "repair_then_rebuild"
    assert validation["correctness_filter_summary"]["deferred_issue_counts"] == {"claim_support_needs_repair": 1}


def test_llm_claim_with_anchor_mismatch_is_downgraded_directional_candidate():
    claim_text = "\u4f01\u4e1a\u7ea7AI Agent\u7ade\u4e89\u683c\u5c40\u6b63\u5728\u5411\u573a\u666f\u843d\u5730\u548c\u6e20\u9053\u751f\u6001\u5206\u5316\u3002"
    evidence_text = "\u7edf\u8ba1\u90e8\u95e8\u901a\u8fc7\u5b98\u7f51\u548c\u7edf\u8ba1\u5e74\u9274\u53d1\u5e03AI Agent\u76f8\u5173\u7edf\u8ba1\u6570\u636e\u3002"
    evidence_package = {
        "analysis_ready_evidence": [
            {
                **_evidence("EV-channel", chapter_id="ch_02", level="B"),
                "fact": evidence_text,
                "distilled_fact": evidence_text,
                "source_title": "\u7edf\u8ba1\u53d1\u5e03\u6e20\u9053\u8bf4\u660e",
            }
        ]
    }
    payload = {
        "chapter_synthesis": [
            {
                "chapter_id": "ch_02",
                "claim_units": [
                    {
                        "claim_id": "anchor-mismatch",
                        "claim": claim_text,
                        "claim_strength": "moderate",
                        "used_evidence_ids": ["EV-channel"],
                        "evidence_basis": [evidence_text],
                        "reasoning": "\u5f53\u524d\u8bc1\u636e\u53ea\u80fd\u4f5c\u4e3a\u65b9\u5411\u6027\u80cc\u666f\u3002",
                    }
                ],
            }
        ]
    }

    validation = validate_llm_analysis_output(payload, evidence_package)
    unit = validation["chapter_synthesis"][0]["claim_units"][0]

    assert validation["status"] == "valid"
    assert validation["usable_claim_count"] == 1
    assert validation["deferred_claim_count"] == 0
    assert validation["llm_validation_issue_counts"]["claim_support_anchor_mismatch_downgraded"] == 1
    assert unit["claim_strength"] == "directional"
    assert unit["claim_status"] == "directional"
    assert unit["claim_support_status"] == "anchor_mismatch_downgraded"
    assert unit["writing_permission"] == "cautious_with_boundary"
    assert unit["evidence_use_level"] == "directional_signal"


def test_llm_semantic_judge_rejects_semantically_unsupported_claim(monkeypatch):
    calls = []

    def fake_semantic_judge(*, claim_text, cited_cards, chapter_id, claim_id, llm_config):
        calls.append(
            {
                "claim_text": claim_text,
                "chapter_id": chapter_id,
                "claim_id": claim_id,
                "llm_config": llm_config,
                "cited_count": len(cited_cards),
            }
        )
        return {
            "status": "unsupported",
            "reason": "The cited source only says AI Agent is discussed, not that deployments are competitive differentiators.",
            "confidence": 0.94,
        }

    monkeypatch.setattr(analysis_agent, "_llm_semantic_claim_support_judge", fake_semantic_judge)
    evidence_package = {
        "analysis_ready_evidence": [
            {
                **_evidence("EV-agent", chapter_id="ch_02", level="B"),
                "fact": "The report discusses AI Agent enterprise software trends.",
                "distilled_fact": "The report discusses AI Agent enterprise software trends.",
                "source_title": "AI Agent trends",
            }
        ]
    }
    payload = {
        "chapter_synthesis": [
            {
                "chapter_id": "ch_02",
                "claim_units": [
                    {
                        "claim_id": "semantic-mismatch",
                        "claim": "AI Agent enterprise software trends are becoming a competitive differentiator.",
                        "claim_strength": "directional",
                        "used_evidence_ids": ["EV-agent"],
                        "evidence_basis": ["The report discusses AI Agent enterprise software trends."],
                        "reasoning": "The cited evidence discusses the trend.",
                    }
                ],
            }
        ]
    }

    validation = validate_llm_analysis_output(payload, evidence_package, llm_config={"model": "judge"})

    assert calls and calls[0]["claim_id"] == "semantic-mismatch"
    assert validation["usable_claim_count"] == 0
    assert validation["dropped_claim_count"] == 0
    assert validation["deferred_claim_count"] == 1
    assert validation["llm_validation_issue_counts"]["llm_claim_semantic_judge_unsupported"] == 1
    assert validation["llm_semantic_judge_counts"]["attempted"] == 1
    assert validation["llm_semantic_judge_counts"]["unsupported"] == 1
    assert validation["llm_rejected_claim_examples"][0]["semantic_judge"]["status"] == "unsupported"
    assert validation["correctness_filter_summary"]["drop_issue_counts"] == {}
    assert validation["correctness_filter_summary"]["deferred_issue_counts"] == {
        "llm_claim_semantic_judge_unsupported": 1
    }


def test_llm_semantic_judge_partial_downgrades_claim(monkeypatch):
    monkeypatch.setattr(
        analysis_agent,
        "_llm_semantic_claim_support_judge",
        lambda **_kwargs: {"status": "partial", "reason": "Evidence supports direction, not the full strength.", "confidence": 0.72},
    )
    evidence_package = {
        "analysis_ready_evidence": [
            {
                **_evidence("EV-agent", chapter_id="ch_02", level="B"),
                "fact": "Enterprise AI Agent adoption is moving from pilots into workflow automation.",
                "distilled_fact": "Enterprise AI Agent adoption is moving from pilots into workflow automation.",
                "source_title": "AI Agent adoption",
            }
        ]
    }
    payload = {
        "chapter_synthesis": [
            {
                "chapter_id": "ch_02",
                "claim_units": [
                    {
                        "claim_id": "semantic-partial",
                        "claim": "Enterprise AI Agent adoption is moving from pilots into workflow automation.",
                        "claim_strength": "moderate",
                        "used_evidence_ids": ["EV-agent"],
                        "evidence_basis": ["Enterprise AI Agent adoption is moving from pilots into workflow automation."],
                        "reasoning": "The cited evidence supports the adoption transition direction.",
                    }
                ],
            }
        ]
    }

    validation = validate_llm_analysis_output(payload, evidence_package, llm_config={"model": "judge"})
    unit = validation["chapter_synthesis"][0]["claim_units"][0]

    assert validation["usable_claim_count"] == 1
    assert validation["llm_validation_issue_counts"]["llm_claim_semantic_judge_partial_downgraded"] == 1
    assert validation["llm_semantic_judge_counts"]["partial"] == 1
    assert unit["semantic_judge_status"] == "partial"


def test_isolated_quality_gate_observes_semantic_judge_without_mutating_claim(monkeypatch):
    monkeypatch.setenv("REPORT_QUALITY_GATE_MODE", "isolated")
    monkeypatch.setattr(
        analysis_agent,
        "_llm_semantic_claim_support_judge",
        lambda **_kwargs: {
            "status": "partial",
            "reason": "The evidence only partially supports the scope.",
            "confidence": 0.8,
        },
    )
    evidence_package = {
        "analysis_ready_evidence": [
            _evidence(
                "EV-SAFE",
                chapter_id="ch_01",
                allowed_use="supporting",
                proof_role="support",
            )
        ]
    }
    payload = {
        "chapter_synthesis": [
            {
                "chapter_id": "ch_01",
                "claim_units": [
                    {
                        "claim_id": "claim-1",
                        "claim": "Enterprise agent deployments are moving from pilots into workflow automation.",
                        "used_evidence_ids": ["EV-SAFE"],
                        "evidence_basis": [
                            "Enterprise agent deployments are moving from pilots into workflow automation."
                        ],
                        "reasoning_chain": [
                            "The cited deployment signal directly supports a workflow automation discussion."
                        ],
                        "limitation_boundary": ["Keep scope bounded to workflow automation."],
                        "claim_strength": "moderate",
                        "claim_strength_ceiling": "moderate",
                    }
                ],
            }
        ]
    }

    validation = validate_llm_analysis_output(payload, evidence_package, llm_config={"model": "judge"})

    assert validation["usable_claim_count"] == 1
    issue_counts = validation["llm_validation_issue_counts"]
    assert issue_counts["llm_claim_semantic_judge_partial_observed"] == 1
    assert "llm_claim_semantic_judge_partial_downgraded" not in issue_counts
    unit = validation["chapter_synthesis"][0]["claim_units"][0]
    assert unit["claim_strength"] == "moderate"
    assert unit["semantic_judge_status"] == "partial"
    assert "semantic judge" not in str(unit.get("counter_boundary") or "").lower()
    assert "semantic judge" not in str(unit.get("limitation_boundary") or "").lower()


def test_llm_semantic_judge_adjacent_downgrades_claim(monkeypatch):
    monkeypatch.setattr(
        analysis_agent,
        "_llm_semantic_claim_support_judge",
        lambda **_kwargs: {"status": "adjacent", "reason": "Evidence is relevant background but not direct support.", "confidence": 0.68},
    )
    evidence_package = {
        "analysis_ready_evidence": [
            {
                **_evidence("EV-agent", chapter_id="ch_02", level="B"),
                "fact": "Enterprise AI Agent adoption signals demand formation.",
                "distilled_fact": "Enterprise AI Agent adoption signals demand formation.",
                "source_title": "AI Agent adoption",
            }
        ]
    }
    payload = {
        "chapter_synthesis": [
            {
                "chapter_id": "ch_02",
                "claim_units": [
                    {
                        "claim_id": "semantic-adjacent",
                        "claim": "Enterprise AI Agent adoption signals demand formation.",
                        "claim_strength": "moderate",
                        "used_evidence_ids": ["EV-agent"],
                        "evidence_basis": ["Enterprise AI Agent adoption signals demand formation."],
                        "reasoning": "The cited evidence is relevant to the demand signal.",
                    }
                ],
            }
        ]
    }

    validation = validate_llm_analysis_output(payload, evidence_package, llm_config={"model": "judge"})
    unit = validation["chapter_synthesis"][0]["claim_units"][0]

    assert validation["usable_claim_count"] == 1
    assert validation["dropped_claim_count"] == 0
    assert validation["llm_validation_issue_counts"]["llm_claim_semantic_judge_adjacent_downgraded"] == 1
    assert validation["llm_semantic_judge_counts"]["adjacent"] == 1
    assert unit["semantic_judge_status"] == "adjacent"
    assert unit["claim_strength"] == "directional"
    assert unit["analysis_role"] == "contextual"
    assert unit["evidence_use_level"] == "background"
    assert unit["writing_permission"] == "cautious_with_boundary"


def test_llm_semantic_judge_error_does_not_drop_formal_claim_by_default(monkeypatch):
    monkeypatch.delenv("BRAIN_LLM_SEMANTIC_JUDGE_FAIL_CLOSED", raising=False)
    monkeypatch.setattr(
        analysis_agent,
        "_llm_semantic_claim_support_judge",
        lambda **_kwargs: {"status": "error", "reason": "judge timeout"},
    )
    evidence_package = {
        "analysis_ready_evidence": [
            {
                **_evidence("EV-agent", chapter_id="ch_02", level="B"),
                "fact": "Enterprise AI Agent adoption is moving from pilots into workflow automation.",
                "distilled_fact": "Enterprise AI Agent adoption is moving from pilots into workflow automation.",
                "source_title": "AI Agent adoption",
            }
        ]
    }
    payload = {
        "chapter_synthesis": [
            {
                "chapter_id": "ch_02",
                "claim_units": [
                    {
                        "claim_id": "semantic-error",
                        "claim": "Enterprise AI Agent adoption is moving from pilots into workflow automation.",
                        "claim_strength": "directional",
                        "used_evidence_ids": ["EV-agent"],
                        "evidence_basis": ["Enterprise AI Agent adoption is moving from pilots into workflow automation."],
                        "reasoning": "The cited evidence directly supports the transition.",
                    }
                ],
            }
        ]
    }

    validation = validate_llm_analysis_output(payload, evidence_package, llm_config={"model": "judge"})
    unit = validation["chapter_synthesis"][0]["claim_units"][0]

    assert validation["usable_claim_count"] == 1
    assert validation["dropped_claim_count"] == 0
    assert validation["llm_validation_issue_counts"]["llm_claim_semantic_judge_error"] == 1
    assert unit["semantic_judge_status"] == "error"


def test_llm_semantic_judge_pass_allows_claim(monkeypatch):
    monkeypatch.setattr(
        analysis_agent,
        "_llm_semantic_claim_support_judge",
        lambda **_kwargs: {"status": "supported", "reason": "Supported by cited evidence.", "confidence": 0.9},
    )
    evidence_package = {
        "analysis_ready_evidence": [
            {
                **_evidence("EV-agent", chapter_id="ch_02", level="B"),
                "fact": "The report says enterprise AI Agent adoption is moving from pilots into workflow automation.",
                "distilled_fact": "The report says enterprise AI Agent adoption is moving from pilots into workflow automation.",
                "source_title": "AI Agent adoption",
            }
        ]
    }
    payload = {
        "chapter_synthesis": [
            {
                "chapter_id": "ch_02",
                "claim_units": [
                    {
                        "claim_id": "semantic-supported",
                        "claim": "Enterprise AI Agent adoption is moving from pilots into workflow automation.",
                        "claim_strength": "directional",
                        "used_evidence_ids": ["EV-agent"],
                        "evidence_basis": ["The report says enterprise AI Agent adoption is moving from pilots into workflow automation."],
                        "reasoning": "The cited evidence directly supports the adoption transition.",
                    }
                ],
            }
        ]
    }

    validation = validate_llm_analysis_output(payload, evidence_package, llm_config={"model": "judge"})

    assert validation["usable_claim_count"] == 1
    assert validation["llm_semantic_judge_counts"]["supported"] == 1
    assert validation["chapter_synthesis"][0]["claim_units"][0]["semantic_judge_status"] == "supported"


def test_llm_semantic_judge_uses_cache_and_temperature_zero(monkeypatch, tmp_path):
    calls = []

    def fake_llm(*, config, system_prompt, user_payload):
        calls.append({"config": dict(config), "user_payload": user_payload})
        return {
            "payload": {"status": "supported", "reason": "Directly supported.", "confidence": 0.91},
            "usage": {"total_tokens": 123},
        }

    monkeypatch.setattr(analysis_agent, "call_openai_compatible_json", fake_llm)
    monkeypatch.setattr(analysis_agent, "llm_config_is_ready", lambda cfg: True)
    monkeypatch.setattr(analysis_agent, "normalize_llm_config", lambda cfg: {"model": cfg.get("model", "judge")})
    monkeypatch.setenv("BRAIN_LLM_SEMANTIC_JUDGE_CACHE_PATH", str(tmp_path))
    monkeypatch.setenv("BRAIN_LLM_SEMANTIC_JUDGE_CACHE_ENABLED", "true")
    evidence_package = {
        "run_id": "run-semantic-cache",
        "analysis_ready_evidence": [
            {
                **_evidence("EV-agent", chapter_id="ch_02", level="B"),
                "fact": "The source says AI Agent adoption is moving into enterprise workflow automation.",
                "distilled_fact": "The source says AI Agent adoption is moving into enterprise workflow automation.",
                "source_title": "AI Agent adoption",
            }
        ],
    }
    payload = {
        "chapter_synthesis": [
            {
                "chapter_id": "ch_02",
                "claim_units": [
                    {
                        "claim_id": "semantic-cache",
                        "claim": "AI Agent adoption is moving into enterprise workflow automation.",
                        "claim_strength": "directional",
                        "used_evidence_ids": ["EV-agent"],
                        "evidence_basis": ["The source says AI Agent adoption is moving into enterprise workflow automation."],
                        "reasoning": "The cited evidence directly describes workflow adoption.",
                    }
                ],
            }
        ]
    }

    first = validate_llm_analysis_output(payload, evidence_package, llm_config={"model": "judge", "temperature": 0.7})
    second = validate_llm_analysis_output(payload, evidence_package, llm_config={"model": "judge", "temperature": 0.7})

    assert first["usable_claim_count"] == 1
    assert second["usable_claim_count"] == 1
    assert len(calls) == 1
    assert calls[0]["config"]["temperature"] == 0
    assert calls[0]["user_payload"]["schema_version"] == "semantic_claim_support_judge_v1"
    assert second["llm_semantic_judge_counts"]["cache_hit"] == 1
    assert second["llm_semantic_judge_usage"] == {}


def test_llm_analysis_prompts_request_typed_claims_without_two_to_three_cap(monkeypatch):
    chapter_prompt = analysis_agent._llm_chapter_system_prompt()
    assert "4-6 claim_units" in chapter_prompt
    assert "claim_type" in chapter_prompt
    assert "contextual" in chapter_prompt
    assert "boundary" in chapter_prompt
    assert "2-3 claim_units" not in chapter_prompt

    captured = {}

    def fake_llm(*, config, system_prompt, user_payload):
        captured["system_prompt"] = system_prompt
        return {"payload": {"chapter_synthesis": [], "evidence_repair_priorities": []}, "usage": {}}

    monkeypatch.setattr(analysis_agent, "call_openai_compatible_json", fake_llm)
    monkeypatch.setattr(analysis_agent, "llm_config_is_ready", lambda cfg: True)
    monkeypatch.setattr(analysis_agent, "normalize_llm_config", lambda cfg: {"model": "fake"})

    analysis_agent.synthesize_with_llm_analysis(
        evidence_package={"analysis_ready_evidence": [_evidence("EV-1")]},
        fallback={},
        llm_config={"provider": "fake", "model": "fake"},
    )

    global_prompt = captured["system_prompt"]
    assert "4-6 claim_units" in global_prompt
    assert "claim_type" in global_prompt
    assert "contextual" in global_prompt
    assert "2-3 claim_units" not in global_prompt


def test_llm_analysis_salvages_truncated_chapter_json(monkeypatch, tmp_path):
    raw = (
        '{"chapter_synthesis":[{"chapter_id":"ch_01","claim_units":[{'
        '"claim_id":"CL-SALVAGE",'
        '"claim":"AI Agent deployments are moving into enterprise workflow automation.",'
        '"claim_type":"directional_claim",'
        '"used_evidence_ids":["EV-1"],'
        '"claim_strength":"directional",'
        '"one_sentence_reason":"The cited source describes enterprise workflow automation."'
        '}]}'
    )

    def fake_llm(*, config, system_prompt, user_payload):
        raise LLMCallError(
            "LLM response is not valid JSON: truncated",
            diagnostic={"raw_content": raw, "error": "LLM response is not valid JSON: truncated"},
        )

    monkeypatch.setattr(analysis_agent, "call_openai_compatible_json", fake_llm)
    monkeypatch.setattr(analysis_agent, "llm_config_is_ready", lambda cfg: True)
    monkeypatch.setattr(analysis_agent, "normalize_llm_config", lambda cfg: {"model": cfg.get("model", "fake")})
    monkeypatch.setenv("BRAIN_LLM_ANALYSIS_CACHE_PATH", str(tmp_path))

    result = analysis_agent.synthesize_with_llm_analysis_v2(
        evidence_package={
            "query": "AI Agent enterprise deployment",
            "analysis_ready_evidence": [
                {
                    **_evidence("EV-1", chapter_id="ch_01", level="B"),
                    "distilled_fact": "AI Agent deployments are moving into enterprise workflow automation.",
                    "fact": "AI Agent deployments are moving into enterprise workflow automation.",
                }
            ],
        },
        fallback={"query": "AI Agent enterprise deployment"},
        llm_config={"provider": "fake", "model": "fake"},
    )

    assert result["_llm_failed_chapter_count"] == 0
    assert result["_llm_json_salvage_attempted_count"] == 1
    assert result["_llm_json_salvage_success_count"] == 1
    assert result["chapter_synthesis"][0]["claim_units"][0]["claim_id"] == "CL-SALVAGE"
    assert result["_llm_chapter_results"][0]["status"] == "json_salvaged"


def test_llm_claim_support_validator_unavailable_is_not_silent(monkeypatch):
    monkeypatch.setattr(analysis_agent, "validate_claim_supported_by_facts", None)
    evidence_package = {
        "analysis_ready_evidence": [
            {
                **_evidence("EV-1", chapter_id="ch_02", level="A"),
                "fact": "MiniMax disclosed an enterprise AI Agent workflow deployment.",
                "distilled_fact": "MiniMax disclosed an enterprise AI Agent workflow deployment.",
            }
        ]
    }
    payload = {
        "chapter_synthesis": [
            {
                "chapter_id": "ch_02",
                "claim_units": [
                    {
                        "claim_id": "claim-without-validator",
                        "claim": "MiniMax disclosed an enterprise AI Agent workflow deployment.",
                        "claim_strength": "directional",
                        "used_evidence_ids": ["EV-1"],
                        "evidence_basis": ["MiniMax disclosed an enterprise AI Agent workflow deployment."],
                        "reasoning": "The cited evidence is directly about the deployment.",
                    }
                ],
            }
        ]
    }

    validation = validate_llm_analysis_output(payload, evidence_package)

    assert validation["usable_claim_count"] == 0
    assert validation["status"] == "invalid_output_no_usable_claims"
    assert validation["llm_validation_issue_counts"]["claim_support_validator_unavailable"] == 1
    assert validation["llm_rejected_claim_examples"][0]["type"] == "claim_support_validator_unavailable"


def test_llm_chinese_qualitative_claim_with_offtopic_fact_is_rejected():
    evidence_package = {
        "analysis_ready_evidence": [
            {
                **_evidence("EV-channel", chapter_id="ch_02", level="B"),
                "fact": "统计部门通过官网、统计年鉴和官方社交媒体发布AI Agent相关统计数据。",
                "distilled_fact": "统计部门通过官网、统计年鉴和官方社交媒体发布AI Agent相关统计数据。",
                "source_title": "统计发布渠道说明",
            }
        ]
    }
    payload = {
        "chapter_synthesis": [
            {
                "chapter_id": "ch_02",
                "claim_units": [
                    {
                        "claim_id": "bad-cn-context-claim",
                        "claim": "企业级AI Agent竞争格局正在向场景落地和渠道生态分化。",
                        "claim_strength": "directional",
                        "used_evidence_ids": ["EV-channel"],
                        "evidence_basis": ["统计部门通过官网、统计年鉴和官方社交媒体发布AI Agent相关统计数据。"],
                        "reasoning": "The cited evidence only describes a publication channel.",
                    }
                ],
            }
        ]
    }

    validation = validate_llm_analysis_output(payload, evidence_package)

    unit = validation["chapter_synthesis"][0]["claim_units"][0]

    assert validation["usable_claim_count"] == 1
    assert validation["deferred_claim_count"] == 0
    assert validation["status"] == "valid"
    assert validation["llm_validation_issue_counts"]["claim_support_anchor_mismatch_downgraded"] == 1
    assert unit["claim_support_status"] == "anchor_mismatch_downgraded"
    assert unit["claim_strength"] == "directional"
    assert unit["writing_permission"] == "cautious_with_boundary"


def test_llm_numeric_claim_with_incomplete_metric_fact_is_downgraded():
    evidence = _evidence("EV-METRIC", chapter_id="ch_02", level="B")
    evidence.update(
        {
            "fact": "MiniMax M2.5 API price is about 1/15 of Claude Opus 4.6.",
            "distilled_fact": "MiniMax M2.5 API price is about 1/15 of Claude Opus 4.6.",
            "fact_type": "metric",
            "metric": "API price ratio",
            "value": "1/15",
            "unit": "",
            "period": "",
            "source_url": "",
            "source_ref": "",
        }
    )
    evidence_package = {"analysis_ready_evidence": [evidence]}
    payload = {
        "chapter_synthesis": [
            {
                "chapter_id": "ch_02",
                "claim_units": [
                    {
                        "claim_id": "incomplete-metric-claim",
                        "claim": "MiniMax M2.5 API price is about 1/15 of Claude Opus 4.6.",
                        "claim_strength": "directional",
                        "used_evidence_ids": ["EV-METRIC"],
                        "evidence_basis": ["MiniMax M2.5 API price is about 1/15 of Claude Opus 4.6."],
                        "reasoning": "The cited metric compares the API price ratio.",
                    }
                ],
            }
        ]
    }

    validation = validate_llm_analysis_output(payload, evidence_package)
    unit = validation["chapter_synthesis"][0]["claim_units"][0]

    assert validation["status"] == "valid"
    assert validation["usable_claim_count"] == 1
    assert validation["llm_validation_issue_counts"]["llm_numeric_claim_incomplete_metric_fact"] == 1
    assert unit["claim_strength"] == "directional"
    assert unit["claim_status"] == "directional"
    assert unit["evidence_use_level"] == "directional_signal"
    assert unit["writing_permission"] == "cautious_with_boundary"
    assert unit["metric_completeness_status"] == "incomplete"
    assert unit["metric_missing_fields"] == ["unit", "period"]
    assert any("metric fields incomplete" in item for item in unit["limitation_boundary"])

    merged = merge_llm_analysis_with_fallback({}, payload, validation)
    merged_unit = merged["claim_units"][0]
    key_claim = merged["chapter_insights"][0]["key_claims"][0]
    judgment = merged["key_judgments"][0]

    for item in (merged_unit, key_claim, judgment):
        assert item["evidence_use_level"] == "directional_signal"
        assert item["writing_permission"] == "cautious_with_boundary"
        assert item["metric_completeness_status"] == "incomplete"
        assert item["metric_missing_fields"] == ["unit", "period"]


def test_llm_claim_repair_priorities_are_promoted_to_evidence_gap_ledger():
    evidence_package = {
        "analysis_ready_evidence": [
            {
                **_evidence("EV-GTC", chapter_id="ch_02", level="A"),
                "fact": "NVIDIA announced GTC sessions for enterprise AI infrastructure.",
                "distilled_fact": "NVIDIA announced GTC sessions for enterprise AI infrastructure.",
                "source_title": "NVIDIA GTC",
            }
        ]
    }
    payload = {
        "chapter_synthesis": [
            {
                "chapter_id": "ch_02",
                "claim_units": [
                    {
                        "claim_id": "bad-price-claim",
                        "claim": "MiniMax M2.5 API price is about 1/15 of Claude Opus 4.6.",
                        "claim_strength": "directional",
                        "used_evidence_ids": ["EV-GTC"],
                        "evidence_basis": ["NVIDIA announced GTC sessions for enterprise AI infrastructure."],
                        "reasoning": "The cited evidence discusses NVIDIA infrastructure events.",
                    }
                ],
            }
        ]
    }

    validation = validate_llm_analysis_output(payload, evidence_package)
    merged = merge_llm_analysis_with_fallback({"evidence_gap_ledger": []}, payload, validation)

    assert validation["status"] == "invalid_output_no_usable_claims"
    assert merged["evidence_repair_priorities"][0]["gap_type"] == "claim_support_entity_or_metric_mismatch"
    gap = merged["evidence_gap_ledger"][0]
    assert gap["gap_type"] == "claim_support_entity_or_metric_mismatch"
    assert gap["repair_route"] == "evidence_search"
    assert gap["source_stage"] == "analysis_claim_support"
    assert gap["status"] == "open"
    assert gap["allowed_for_writing"] is False


def test_llm_validator_normalizes_analysis_first_claim_contract():
    evidence_package = {"analysis_ready_evidence": [_evidence("EV-ANALYSIS")]}
    payload = {
        "chapter_synthesis": [
            {
                "chapter_id": "ch_01",
                "chapter_title": "Demand validation",
                "claim_units": [
                    {
                        "claim": "Enterprise agents are moving into workflow automation.",
                        "claim_strength": "directional",
                        "analysis_role": "directional",
                        "used_evidence_ids": ["EV-ANALYSIS"],
                        "evidence_basis": ["Enterprise workflow evidence is present."],
                        "reasoning_chain": ["Workflow deployments indicate operational adoption."],
                        "limitation_boundary": ["The claim is limited to disclosed customer samples."],
                        "block_affinity": "case_comparison",
                    }
                ],
            }
        ]
    }

    validation = validate_llm_analysis_output(payload, evidence_package)
    unit = validation["chapter_synthesis"][0]["claim_units"][0]

    assert unit["claim_id"] == "ch_01_claim_1"
    assert unit["supporting_fact_refs"] == ["EV-ANALYSIS"]
    assert unit["source_support_map"] == {
        "claim": ["EV-ANALYSIS"],
        "mechanism": ["EV-ANALYSIS"],
        "boundary": ["EV-ANALYSIS"],
    }
    assert unit["paragraph_seed"]
    assert unit["analysis_role"] == "directional"


def test_llm_validator_resolves_evidence_alias_to_canonical_fact_id(monkeypatch):
    monkeypatch.setattr(
        analysis_agent,
        "_llm_semantic_claim_support_judge",
        lambda **_kwargs: {"status": "supported", "reason": "Supported by cited evidence.", "confidence": 0.9},
    )
    evidence = _evidence("EV-agent")
    evidence["aliases"] = ["EV-agent-L1"]
    evidence["source_id"] = "SRC-agent"
    evidence["requirement_id"] = "REQ-agent"
    evidence_package = {"analysis_ready_evidence": [evidence]}
    payload = {
        "chapter_synthesis": [
            {
                "chapter_id": "ch_01",
                "chapter_title": "Demand validation",
                "claim_units": [
                    {
                        "claim_id": "CL-alias",
                        "claim": "Enterprise agent deployments are moving from pilots into workflow automation.",
                        "claim_strength": "directional",
                        "analysis_role": "directional",
                        "used_evidence_ids": ["EV-agent-L1"],
                        "evidence_basis": ["Enterprise workflow evidence is present."],
                        "reasoning_chain": ["Workflow deployments indicate operational adoption."],
                        "limitation_boundary": ["The claim is limited to disclosed customer samples."],
                    }
                ],
            }
        ]
    }

    validation = validate_llm_analysis_output(payload, evidence_package, llm_config={"model": "judge"})
    unit = validation["chapter_synthesis"][0]["claim_units"][0]

    assert validation["status"] == "valid"
    assert unit["fact_ids"] == ["EV-agent"]
    assert unit["evidence_refs"] == ["EV-agent"]
    assert unit["source_ids"] == ["SRC-agent"]
    assert unit["requirement_ids"] == ["REQ-agent"]
    assert unit["ref_resolution"]["alias_resolved_ref_count"] == 1
    merged = merge_llm_analysis_with_fallback({"claim_units": []}, payload, validation)
    merged_unit = merged["claim_units"][0]
    assert merged_unit["fact_ids"] == ["EV-agent"]
    assert merged_unit["evidence_refs"] == ["EV-agent"]
    assert merged_unit["ref_resolution"]["alias_resolved_ref_count"] == 1


def test_llm_input_v2_carries_requirement_contract_fields():
    evidence = _evidence("EV-REQ")
    evidence.update(
        {
            "hypothesis_id": "H1",
            "requirement_id": "H1_case",
            "analysis_role": "case",
            "analysis_eligible": True,
            "allowed_use": "directional_signal",
            "search_task_id": "task_case_1",
            "source_id": "SRC-1",
        }
    )
    evidence_package = {
        "query": "AI Agent report",
        "report_contract": {
            "evidence_requirements": {
                "requirements": [
                    {
                        "requirement_id": "H1_case",
                        "chapter_id": "ch_01",
                        "proof_role": "case",
                        "required_fields": ["company", "use_case", "source_ref"],
                        "claim_strength_ceiling": "directional",
                    }
                ]
            }
        },
        "chapter_evidence_packages": [
            {
                "chapter_id": "ch_01",
                "chapter_title": "Demand validation",
            }
        ],
        "analysis_ready_evidence": [evidence],
    }

    payload = build_llm_analysis_input_v2(evidence_package, {})
    card = payload["chapters"][0]["fact_cards"][0]

    assert card["requirement_id"] == "H1_case"
    assert card["hypothesis_id"] == "H1"
    assert card["analysis_role"] == "case"
    assert card["allowed_use"] == "directional_signal"
    assert card["lineage"]["search_task_id"] == "task_case_1"
    assert payload["chapters"][0]["evidence_requirements"][0]["requirement_id"] == "H1_case"


def test_llm_validator_preserves_requirement_ids_and_lineage():
    evidence = _evidence("EV-REQ")
    evidence.update(
        {
            "requirement_id": "H1_case",
            "hypothesis_id": "H1",
            "source_id": "SRC-1",
            "proof_role": "case",
            "analysis_role": "case",
        }
    )
    evidence_package = {"analysis_ready_evidence": [evidence]}
    payload = {
        "chapter_synthesis": [
            {
                "chapter_id": "ch_01",
                "chapter_title": "Demand validation",
                "claim_units": [
                    {
                        "claim": "Enterprise agents are moving into workflow automation.",
                        "claim_strength": "directional",
                        "claim_strength_ceiling": "directional",
                        "hypothesis_id": "H1",
                        "requirement_ids": ["H1_case"],
                        "used_evidence_ids": ["EV-REQ"],
                        "evidence_basis": ["Workflow deployment evidence is present."],
                        "reasoning_chain": ["Workflow deployment indicates operational adoption."],
                        "limitation_boundary": ["The claim is limited to disclosed samples."],
                        "source_support_map": {"claim": ["EV-REQ"], "mechanism": ["EV-REQ"], "boundary": ["EV-REQ"]},
                    }
                ],
            }
        ]
    }

    validation = validate_llm_analysis_output(payload, evidence_package)
    unit = validation["chapter_synthesis"][0]["claim_units"][0]

    assert unit["requirement_ids"] == ["H1_case"]
    assert unit["hypothesis_id"] == "H1"
    assert unit["claim_strength_ceiling"] == "directional"
    assert unit["lineage"]["requirement_ids"] == ["H1_case"]
    assert unit["lineage"]["fact_ids"] == ["EV-REQ"]
    assert unit["primary_claim_role"] == "case_claim"
    assert "case_claim" in unit["claim_roles"]
    assert "boundary_claim" in unit["claim_roles"]


def test_llm_validator_rejects_claim_without_requirement_lineage():
    evidence = _evidence("EV-NO-REQ")
    evidence_package = {
        "analysis_ready_evidence": [evidence],
        "report_contract": {
            "evidence_requirements": {
                "requirements": [
                    {
                        "requirement_id": "H1_case",
                        "chapter_id": "ch_01",
                        "proof_role": "case",
                    }
                ]
            }
        },
    }
    payload = {
        "chapter_synthesis": [
            {
                "chapter_id": "ch_01",
                "claim_units": [
                    {
                        "claim": "Enterprise agents are moving into workflow automation.",
                        "claim_strength": "directional",
                        "claim_strength_ceiling": "directional",
                        "used_evidence_ids": ["EV-NO-REQ"],
                        "evidence_basis": ["Workflow deployment evidence is present."],
                        "reasoning_chain": ["Workflow deployment indicates operational adoption."],
                        "limitation_boundary": ["The claim is limited to disclosed samples."],
                    }
                ],
            }
        ]
    }

    validation = validate_llm_analysis_output(payload, evidence_package)

    assert validation["status"] == "invalid_output_no_usable_claims"
    assert "llm_claim_unit_missing_requirement_ids" in validation["llm_validation_issue_counts"]


def test_llm_validator_rejects_claim_strength_above_ceiling():
    evidence = _evidence("EV-CEILING")
    evidence.update({"requirement_id": "H1_case", "source_level": "B", "allowed_use": "directional_signal"})
    evidence_package = {"analysis_ready_evidence": [evidence]}
    payload = {
        "chapter_synthesis": [
            {
                "chapter_id": "ch_01",
                "claim_units": [
                    {
                        "claim": "Enterprise agents have definitively become mainstream infrastructure.",
                        "claim_strength": "strong",
                        "claim_strength_ceiling": "directional",
                        "requirement_ids": ["H1_case"],
                        "used_evidence_ids": ["EV-CEILING"],
                        "evidence_basis": ["Workflow deployment evidence is present."],
                        "reasoning_chain": ["Workflow deployment indicates operational adoption."],
                        "limitation_boundary": ["The claim is limited to disclosed samples."],
                    }
                ],
            }
        ]
    }

    validation = validate_llm_analysis_output(payload, evidence_package)

    assert validation["status"] == "invalid_output_no_usable_claims"
    assert "llm_claim_strength_exceeds_ceiling" in validation["llm_validation_issue_counts"]


def test_llm_input_filters_dirty_cards_and_internal_diagnostics():
    dirty = _evidence("EV-DIRTY")
    dirty["fact"] = "Skip to content login menu picture intentionally omitted"
    dirty["source"] = {"title": "Official", "url": "https://example.gov/fake"}
    clean = _evidence("EV-CLEAN")
    evidence_package = {
        "chapter_evidence_diagnostics": {"ch_01": {}},
        "analysis_ready_evidence": [dirty, clean],
    }
    fallback = {
        "evidence_gap_ledger": [{"type": "internal_gap"}],
        "claim_units": [{"claim": "old fallback claim"}],
    }

    payload = build_llm_analysis_input(evidence_package, fallback)

    assert "evidence_gap_ledger" not in payload
    assert "fallback_claim_units" not in payload
    assert "evidence_cards" not in payload
    assert [item["evidence_id"] for item in payload["fact_cards"]] == ["EV-CLEAN"]


def test_llm_input_uses_chapter_aliases_for_fact_cards():
    evidence_package = {
        "chapter_evidence_diagnostics": {
            "ch_01": {
                "chapter_id_aliases": ["dim-demand", "hyp-demand", "AI Agent demand validation"],
            }
        },
        "analysis_ready_evidence": [
            {
                "evidence_id": "EV-ALIAS",
                "dimension_id": "dim-demand",
                "hypothesis_id": "hyp-demand",
                "fact": "Salesforce disclosed Agentforce customer-service workflow deployments in 2025.",
                "source_level": "B",
                "allowed_use": "supporting",
                "proof_role": "case",
                "source_verification_status": "readpage_verified",
                "source": {
                    "title": "Salesforce Agentforce customer story",
                    "url": "https://www.salesforce.com/news/agentforce",
                    "source_verification_status": "readpage_verified",
                },
                "public_fact_card": {
                    "subject": "Salesforce Agentforce",
                    "distilled_fact": "Salesforce disclosed Agentforce customer-service workflow deployments in 2025.",
                    "block_affinity": ["case_comparison"],
                },
            }
        ],
    }

    cards = _evidence_cards_for_llm(evidence_package, max_chapters=1, max_per_chapter=4)

    assert [card["evidence_id"] for card in cards] == ["EV-ALIAS"]
    assert cards[0]["chapter_id"] == "ch_01"


def test_llm_input_prefers_chapter_package_canonical_ids():
    evidence_package = {
        "chapter_evidence_packages": [
            {
                "chapter_id": "ch_01",
                "chapter_title": "Demand validation",
                "chapter_question": "AI Agent demand validation",
            }
        ],
        "analysis_ready_evidence": [
            {
                "evidence_id": "EV-CANON",
                "chapter_id": "AI Agent demand validation",
                "fact": "Salesforce disclosed Agentforce customer-service workflow deployments in 2025.",
                "source_level": "A",
                "allowed_use": "core_claim",
                "proof_role": "case",
                "source_verification_status": "readpage_verified",
                "source": {
                    "title": "Salesforce Agentforce customer story",
                    "url": "https://www.salesforce.com/news/agentforce",
                    "source_verification_status": "readpage_verified",
                },
            }
        ],
    }

    payload = build_llm_analysis_input_v2(evidence_package, {})

    assert payload["chapters"][0]["chapter_id"] == "ch_01"
    assert payload["chapters"][0]["allowed_evidence_ids"] == ["EV-CANON"]


def test_build_llm_analysis_input_v2_is_compact_and_excludes_diagnostics(monkeypatch):
    monkeypatch.setenv("BRAIN_LLM_ANALYSIS_MAX_FACTS_PER_CHAPTER", "1")
    monkeypatch.setenv("BRAIN_LLM_ANALYSIS_MAX_FACT_CHARS", "42")
    clean = _evidence("EV-CLEAN")
    clean["fact"] = "Salesforce disclosed Agentforce deployments in customer-service workflow automation during 2025."
    clean["public_fact_card"] = {
        "distilled_fact": clean["fact"],
        "fact_type": "case",
        "block_affinity": ["case_comparison"],
    }
    dirty = _evidence("EV-DIRTY")
    dirty["fact"] = "Skip to content Login Menu Products Resources"
    evidence_package = {
        "query": "AI Agent industry report",
        "research_plan": {"huge": "x" * 5000},
        "evidence_gap_ledger": [{"internal": True}],
        "chapter_evidence_diagnostics": {
            "ch_01": {
                "chapter_title": "Demand validation",
                "chapter_question": "Does AI Agent demand move into workflow deployment?",
            }
        },
        "analysis_ready_evidence": [dirty, clean],
    }

    payload = build_llm_analysis_input_v2(evidence_package, {"claim_units": [{"claim": "fallback"}]})
    payload_text = json.dumps(payload, ensure_ascii=False)

    assert "research_plan" not in payload
    assert "evidence_gap_ledger" not in payload_text
    assert "fallback" not in payload_text
    assert list(payload.keys()) == ["query", "chapters"]
    chapter = payload["chapters"][0]
    assert chapter["chapter_id"] == "ch_01"
    assert chapter["allowed_evidence_ids"] == ["EV-CLEAN"]
    assert len(chapter["fact_cards"]) == 1
    card = chapter["fact_cards"][0]
    assert set(
        [
            "evidence_id",
            "distilled_fact",
            "fact_type",
            "source_level",
            "source_verification_status",
            "proof_role",
            "block_affinity",
            "metric",
            "value",
            "unit",
            "period",
            "source_title",
            "source_url",
        ]
    ).issubset(card)
    assert len(card["distilled_fact"]) <= 45


def test_high_quality_posture_feeds_more_fact_cards_to_llm_analysis(monkeypatch):
    posture_keys = (
        *full_report.HIGH_EVIDENCE_DEPTH_DEFAULTS.keys(),
        *full_report.HIGH_WRITING_QUALITY_DEFAULTS.keys(),
    )
    previous_env = {key: os.environ.get(key) for key in posture_keys}
    for key in posture_keys:
        monkeypatch.delenv(key, raising=False)
    full_report.apply_report_quality_posture("high")
    evidence_package = {
        "query": "AI Agent industry report",
        "chapter_evidence_diagnostics": {
            "ch_01": {
                "chapter_title": "Demand validation",
                "chapter_question": "Does AI Agent demand move into workflow deployment?",
            }
        },
        "analysis_ready_evidence": [
            {
                **_evidence(f"EV-{index:02d}", proof_role="case"),
                "fact": f"Enterprise AI agent deployment case {index} entered repeatable workflow operations.",
                "public_fact_card": {
                    "distilled_fact": f"Enterprise AI agent deployment case {index} entered repeatable workflow operations.",
                    "fact_type": "case",
                    "block_affinity": ["case_comparison"],
                },
            }
            for index in range(1, 21)
        ],
    }

    payload = build_llm_analysis_input_v2(evidence_package, {})

    assert len(payload["chapters"][0]["fact_cards"]) >= 16
    for key, value in previous_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def test_llm_validator_exposes_issue_counts_and_examples():
    evidence_package = {"analysis_ready_evidence": [_evidence("EV-OK")]}
    payload = {
        "chapter_synthesis": [
            {
                "chapter_id": "ch_01",
                "claim_units": [
                    {
                        "claim": "Enterprise agents are moving into workflow automation.",
                        "used_evidence_ids": ["EV-OK"],
                        "evidence_basis": ["The source describes workflow automation deployments."],
                        "reasoning_chain": ["Deployments inside workflows indicate operational adoption."],
                        "limitation_boundary": ["The conclusion is limited to the disclosed deployment samples."],
                        "claim_strength": "moderate",
                    },
                    {
                        "claim": "This claim has no refs.",
                        "used_evidence_ids": [],
                        "evidence_basis": ["x"],
                        "reasoning_chain": ["y"],
                    },
                    {
                        "claim": "AI Agent adoption is accelerating.",
                        "used_evidence_ids": ["EV-MISSING"],
                        "evidence_basis": ["x"],
                        "reasoning_chain": ["y"],
                    },
                ],
            }
        ]
    }

    validation = validate_llm_analysis_output(payload, evidence_package)

    assert validation["status"] == "valid"
    assert validation["llm_raw_chapter_count"] == 1
    assert validation["llm_raw_claim_count"] == 3
    assert validation["llm_validation_issue_counts"]["llm_claim_missing_used_evidence_ids"] == 1
    assert validation["llm_validation_issue_counts"]["invalid_llm_evidence_ref"] == 1
    assert validation["llm_valid_claim_examples"][0]["evidence_refs"] == ["EV-OK"]
    assert len(validation["llm_rejected_claim_examples"]) >= 2


def test_llm_validator_accepts_string_basis_reasoning_and_boundary():
    evidence_package = {"analysis_ready_evidence": [_evidence("EV-OK")]}
    payload = {
        "chapter_synthesis": [
            {
                "chapter_id": "ch_01",
                "claim_units": [
                    {
                        "claim": "AI Agent standardization activity indicates a maturing deployment environment.",
                        "used_evidence_ids": ["EV-OK"],
                        "evidence_basis": "The source describes enterprise workflow automation deployments.",
                        "reasoning_chain": "Workflow deployments indicate operational adoption rather than tool trials.",
                        "limitation_boundary": "The claim is limited to disclosed deployment samples.",
                        "claim_strength": "moderate",
                    }
                ],
            }
        ]
    }

    validation = validate_llm_analysis_output(payload, evidence_package)

    assert validation["status"] == "valid"
    unit = validation["chapter_synthesis"][0]["claim_units"][0]
    assert unit["evidence_basis"] == ["The source describes enterprise workflow automation deployments."]
    assert unit["reasoning"] == "Workflow deployments indicate operational adoption rather than tool trials."
    assert unit["counter_boundary"] == "The claim is limited to disclosed deployment samples."


def test_per_chapter_llm_analysis_partial_success_is_preserved(monkeypatch, tmp_path):
    evidence_package = {
        "query": "AI Agent industry report",
        "chapter_evidence_diagnostics": {
            "ch_01": {"chapter_title": "Demand validation"},
            "ch_02": {"chapter_title": "Technology maturity"},
        },
        "analysis_ready_evidence": [
            _evidence("EV-1", chapter_id="ch_01", level="A"),
            _evidence("EV-2", chapter_id="ch_02", level="B"),
        ],
        "chapter_evidence_packages": [
            {"chapter_id": "ch_01", "chapter_title": "Demand validation", "core_evidence": [_evidence("EV-1", chapter_id="ch_01", level="A")]},
            {"chapter_id": "ch_02", "chapter_title": "Technology maturity", "core_evidence": [_evidence("EV-2", chapter_id="ch_02", level="B")]},
        ],
    }
    calls = []

    def fake_llm(**kwargs):
        if kwargs["user_payload"].get("schema_version") == "semantic_claim_support_judge_v1":
            return {"payload": {"status": "supported", "reason": "Directly supported.", "confidence": 0.9}, "usage": {}}
        chapter = kwargs["user_payload"]["chapter"]
        calls.append(chapter["chapter_id"])
        if chapter["chapter_id"] == "ch_01":
            return {
                "payload": {
                    "chapter_id": "ch_01",
                    "claim_units": [
                        {
                            "claim": "AI Agent demand is moving from pilots into workflow deployment.",
                            "used_evidence_ids": ["EV-1"],
                            "evidence_basis": ["The verified source describes workflow automation deployments."],
                            "reasoning_chain": ["Workflow deployment is a stronger demand signal than tool trial."],
                            "limitation_boundary": ["The conclusion is limited to disclosed enterprise deployment samples."],
                            "claim_strength": "moderate",
                            "block_affinity": "case_comparison",
                        }
                    ],
                    "analysis_limits": [],
                },
                "usage": {"input_tokens": 100, "output_tokens": 80},
            }
        return {
            "payload": {
                "chapter_id": "ch_02",
                "claim_units": [
                    {
                        "claim": "证据不足，建议补证后再判断。",
                        "used_evidence_ids": [],
                        "evidence_basis": [],
                        "reasoning_chain": [],
                    }
                ],
            },
            "usage": {},
        }

    monkeypatch.setattr(analysis_agent, "call_openai_compatible_json", fake_llm)
    monkeypatch.setattr(analysis_agent, "llm_config_is_ready", lambda cfg: True)
    monkeypatch.setattr(analysis_agent, "normalize_llm_config", lambda cfg: {"model": "fake"})
    monkeypatch.setenv("BRAIN_ENABLE_LLM_EVIDENCE_ANALYSIS", "true")
    monkeypatch.setenv("REPORT_QUALITY_MODE", "high")
    monkeypatch.setenv("BRAIN_LLM_ANALYSIS_MODE", "per_chapter")
    monkeypatch.setenv("BRAIN_LLM_ANALYSIS_INPUT_VERSION", "v2")
    monkeypatch.setenv("BRAIN_LLM_ANALYSIS_CACHE_ENABLED", "false")
    monkeypatch.setenv("BRAIN_LLM_ANALYSIS_CACHE_PATH", str(tmp_path))

    result = run_analysis_agent(evidence_package, llm_config={"provider": "fake", "model": "fake"})
    diagnostics = result["structured_analysis"]["analysis_stage_diagnostics"]

    assert set(calls) == {"ch_01", "ch_02"}
    assert diagnostics["llm_analysis_attempted"] is True
    assert diagnostics["uses_llm_analysis"] is True
    assert diagnostics["llm_analysis_status"] == "partial_success"
    assert diagnostics["final_analysis_source"] == "llm_partial_merged"
    assert diagnostics["llm_usable_claim_count"] >= 1
    assert diagnostics["llm_valid_chapter_count"] == 1
    assert diagnostics["llm_failed_chapter_count"] == 1
    assert diagnostics["llm_semantic_judge_counts"]["supported"] == 1
    assert {item["chapter_id"] for item in diagnostics["llm_chapter_results"]} == {"ch_01", "ch_02"}


def test_per_chapter_llm_analysis_uses_cache_hit(monkeypatch, tmp_path):
    evidence_package = {
        "query": "AI Agent industry report",
        "chapter_evidence_diagnostics": {"ch_01": {"chapter_title": "Demand validation"}},
        "analysis_ready_evidence": [_evidence("EV-1", chapter_id="ch_01", level="A")],
        "chapter_evidence_packages": [
            {"chapter_id": "ch_01", "chapter_title": "Demand validation", "core_evidence": [_evidence("EV-1", chapter_id="ch_01", level="A")]}
        ],
    }
    call_count = {"count": 0}

    def fake_llm(**kwargs):
        if kwargs["user_payload"].get("schema_version") == "semantic_claim_support_judge_v1":
            return {"payload": {"status": "supported", "reason": "Directly supported.", "confidence": 0.9}, "usage": {}}
        call_count["count"] += 1
        return {
            "payload": {
                "chapter_id": "ch_01",
                "claim_units": [
                    {
                        "claim": "AI Agent demand is moving from pilots into workflow deployment.",
                        "used_evidence_ids": ["EV-1"],
                        "evidence_basis": ["The source describes workflow deployments."],
                        "reasoning_chain": ["Workflow deployment indicates operational adoption."],
                        "limitation_boundary": ["The conclusion is limited to disclosed deployment samples."],
                        "claim_strength": "moderate",
                    }
                ],
            },
            "usage": {},
        }

    monkeypatch.setattr(analysis_agent, "call_openai_compatible_json", fake_llm)
    monkeypatch.setattr(analysis_agent, "llm_config_is_ready", lambda cfg: True)
    monkeypatch.setattr(analysis_agent, "normalize_llm_config", lambda cfg: {"model": "fake"})
    monkeypatch.setenv("BRAIN_ENABLE_LLM_EVIDENCE_ANALYSIS", "true")
    monkeypatch.setenv("REPORT_QUALITY_MODE", "high")
    monkeypatch.setenv("BRAIN_LLM_ANALYSIS_MODE", "per_chapter")
    monkeypatch.setenv("BRAIN_LLM_ANALYSIS_INPUT_VERSION", "v2")
    monkeypatch.setenv("BRAIN_LLM_ANALYSIS_CACHE_ENABLED", "true")
    monkeypatch.setenv("BRAIN_LLM_ANALYSIS_CACHE_PATH", str(tmp_path))

    first = run_analysis_agent(evidence_package, llm_config={"provider": "fake", "model": "fake"})
    second = run_analysis_agent(evidence_package, llm_config={"provider": "fake", "model": "fake"})

    assert call_count["count"] == 1
    assert first["structured_analysis"]["analysis_stage_diagnostics"]["llm_analysis_cache_hit_count"] == 0
    assert second["structured_analysis"]["analysis_stage_diagnostics"]["llm_analysis_cache_hit_count"] == 1


def test_analysis_depth_quality_deduplicates_claim_across_storage_containers():
    claim = "企业智能体需求开始从试用转向流程部署。"
    unit = {
        "chapter_id": "ch_01",
        "claim": claim,
        "supporting_evidence": ["EV-1"],
        "reasoning": "客户采购和部署案例同时出现。",
        "counter_evidence": "仍需续约数据验证。",
    }
    structured = {
        "report_insight_package": {
            "chapters": [
                {
                    "chapter_id": "ch_01",
                    "chapter_question": "需求验证",
                    "key_claims": [dict(unit)],
                }
            ]
        },
        "chapter_insights": [
            {
                "chapter_id": "ch_01",
                "chapter_question": "需求验证",
                "key_claims": [dict(unit)],
            }
        ],
        "claim_units": [dict(unit)],
        "evidence_analyses": [{"evidence_id": "EV-1"}],
    }

    quality = analysis_depth_quality(structured)

    assert quality["claim_count"] == 1
    assert quality["repeated_claim_ratio"] == 0.0


def test_fallback_reasoning_helpers_do_not_emit_public_template_labels():
    facts = [
        "客户案例显示智能体已进入采购流程。",
        "企业披露开始将工具调用纳入流程自动化。",
    ]
    text = "\n".join(
        [
            _reasoning_from_public_facts(facts),
            "\n".join(_build_mechanism_chain(facts, [], "directional", 1)),
        ]
    )

    for phrase in ["事实链", "事实锚点", "交叉信号", "仅有 C 级", "共同构成本章判断的事实基础"]:
        assert phrase not in text


def test_hypothesis_insights_preserve_hypothesis_id_and_avoid_template_mechanism():
    evidence = _evidence("EV-H1", chapter_id="ch_01", level="A", allowed_use="core_claim")
    analysis = _evidence_analysis(evidence, "企业智能体是否进入流程部署", 1)
    analysis["hypothesis_id"] = "H1"
    research_plan = {
        "hypotheses": [
            {
                "hypothesis_id": "H1",
                "claim_to_test": "企业智能体是否进入流程部署",
            }
        ],
        "evidence_coverage_requirements": {
            "per_hypothesis": {"min_A_or_B_sources": 1}
        },
    }

    insights = _hypothesis_insights(research_plan, [analysis])

    assert insights[0]["chapter_id"] in {"H1", "ch_01"}
    assert insights[0].get("chapter_id_source") in {"hypothesis_id", "evidence_chapter_id"}
    public_text = "\n".join(str(item) for item in insights[0].get("mechanism_chain", []))
    for phrase in ["事实锚点", "事实链", "交叉信号", "更适合留在观察层"]:
        assert phrase not in public_text


def test_llm_validation_rejects_refs_when_no_valid_input_cards():
    payload = {
        "chapter_synthesis": [
            {
                "chapter_id": "ch_01",
                "claim_units": [
                    {
                        "claim": "AI Agent adoption is accelerating.",
                        "used_evidence_ids": ["EV-FAKE"],
                        "evidence_basis": ["x"],
                        "reasoning_chain": ["y"],
                        "limitation_boundary": ["z"],
                    }
                ],
            }
        ]
    }

    validation = validate_llm_analysis_output(payload, {})

    assert validation["status"] == "invalid"
    assert validation["reason"] == "no_valid_input_evidence_refs"


def test_llm_success_with_quality_warning_preserves_final_source(monkeypatch):
    evidence_package = {
        "query": "AI Agent industry report",
        "analysis_ready_evidence": [_evidence("EV-1")],
        "chapter_evidence_packages": [
            {
                "chapter_id": "ch_01",
                "chapter_title": "Demand validation",
                "core_evidence": [_evidence("EV-1")],
            }
        ],
    }

    def fake_llm(**kwargs):
        if kwargs["user_payload"].get("schema_version") == "semantic_claim_support_judge_v1":
            return {"payload": {"status": "supported", "reason": "Directly supported.", "confidence": 0.9}, "usage": {}}
        return {
            "payload": {
                "chapter_synthesis": [
                    {
                        "chapter_id": "ch_01",
                        "chapter_title": "Demand validation",
                        "claim_units": [
                            {
                                "claim": "Demand validation",
                                "claim_status": "decision_ready",
                                "claim_strength": "moderate",
                                "used_evidence_ids": ["EV-1"],
                                "evidence_basis": ["Enterprise workflow evidence is present."],
                                "reasoning_chain": ["Workflow deployments indicate operational adoption."],
                                "limitation_boundary": ["The claim depends on comparable customer samples."],
                            }
                        ],
                    }
                ]
            },
            "usage": {},
        }

    monkeypatch.setattr(analysis_agent, "call_openai_compatible_json", fake_llm)
    monkeypatch.setattr(analysis_agent, "llm_config_is_ready", lambda cfg: True)
    monkeypatch.setattr(analysis_agent, "normalize_llm_config", lambda cfg: {"model": "fake"})
    monkeypatch.setenv("BRAIN_ENABLE_LLM_EVIDENCE_ANALYSIS", "true")

    result = run_analysis_agent(evidence_package, llm_config={"provider": "fake", "model": "fake"})
    diagnostics = result["structured_analysis"]["analysis_stage_diagnostics"]

    assert diagnostics.get("analysis_rebuilt_from_evidence") is not True
    assert diagnostics["uses_llm_analysis"] is True
    assert diagnostics["llm_analysis_attempted"] is True
    assert diagnostics["quality_path_degraded"] is False
    assert diagnostics["llm_analysis_status"] == "success"
    assert diagnostics["final_analysis_source"] == "llm_evidence_analysis"
    assert result["structured_analysis"]["analysis_contract_status"]["quality_rebuild_reasons"]
    assert result["structured_analysis"]["analysis_contract_status"]["structural_rebuild_reasons"] == []


def test_quality_mode_marks_invalid_llm_analysis_as_degraded(monkeypatch):
    evidence_package = {
        "query": "AI Agent",
        "analysis_ready_evidence": [_evidence("EV-1")],
        "chapter_evidence_diagnostics": {"ch_01": {"chapter_id": "ch_01"}},
    }

    def fake_llm(**kwargs):
        return {
            "payload": {
                "chapter_synthesis": [
                    {
                        "chapter_id": "ch_01",
                        "claim_units": [
                            {
                                "claim": "证据不足，需要补证后再判断。",
                                "used_evidence_ids": [],
                                "evidence_basis": [],
                                "reasoning_chain": [],
                                "limitation_boundary": [],
                            }
                        ],
                    }
                ]
            },
            "usage": {},
        }

    monkeypatch.setattr(analysis_agent, "call_openai_compatible_json", fake_llm)
    monkeypatch.setattr(analysis_agent, "llm_config_is_ready", lambda cfg: True)
    monkeypatch.setattr(analysis_agent, "normalize_llm_config", lambda cfg: {"model": "fake"})
    monkeypatch.setenv("BRAIN_ENABLE_LLM_EVIDENCE_ANALYSIS", "true")
    monkeypatch.setenv("REPORT_QUALITY_MODE", "high")

    result = run_analysis_agent(evidence_package, llm_config={"provider": "fake", "model": "fake"})
    diagnostics = result["structured_analysis"]["analysis_stage_diagnostics"]

    assert diagnostics["llm_analysis_attempted"] is True
    assert diagnostics["uses_llm_analysis"] is False
    assert diagnostics["quality_path_degraded"] is True
    assert diagnostics["quality_path_degradation_reason"] == "invalid_output"


def test_invalid_llm_analysis_promotes_claim_repair_priorities_to_structured_output(monkeypatch, tmp_path):
    evidence_package = {
        "query": "AI Agent",
        "analysis_ready_evidence": [
            {
                **_evidence("EV-1", chapter_id="ch_01", level="A"),
                "fact": "The source discusses enterprise AI workflow pilots.",
                "distilled_fact": "The source discusses enterprise AI workflow pilots.",
                "source_title": "Enterprise AI pilots",
            }
        ],
        "chapter_evidence_diagnostics": {"ch_01": {"chapter_id": "ch_01"}},
        "chapter_evidence_packages": [
            {
                "chapter_id": "ch_01",
                "chapter_title": "Demand validation",
                "core_evidence": [_evidence("EV-1", chapter_id="ch_01", level="A")],
            }
        ],
    }

    def fake_llm(**kwargs):
        if kwargs["user_payload"].get("schema_version") == "semantic_claim_support_judge_v1":
            return {"payload": {"status": "unsupported", "reason": "not direct", "confidence": 0.9}, "usage": {}}
        return {
            "payload": {
                "chapter_id": "ch_01",
                "claim_units": [
                    {
                        "claim_id": "bad_metric_claim",
                        "claim": "AI Agent market size reached 3158 billion dollars in 2024.",
                        "used_evidence_ids": ["EV-1"],
                        "evidence_basis": ["The source discusses enterprise AI workflow pilots."],
                        "reasoning_chain": ["A market-size claim needs direct metric evidence."],
                        "limitation_boundary": ["Needs direct market-size evidence."],
                        "claim_strength": "moderate",
                    }
                ],
            },
            "usage": {},
        }

    monkeypatch.setattr(analysis_agent, "call_openai_compatible_json", fake_llm)
    monkeypatch.setattr(analysis_agent, "llm_config_is_ready", lambda cfg: True)
    monkeypatch.setattr(analysis_agent, "normalize_llm_config", lambda cfg: {"model": "fake"})
    monkeypatch.setenv("BRAIN_ENABLE_LLM_EVIDENCE_ANALYSIS", "true")
    monkeypatch.setenv("REPORT_QUALITY_MODE", "high")
    monkeypatch.setenv("BRAIN_LLM_ANALYSIS_MODE", "per_chapter")
    monkeypatch.setenv("BRAIN_LLM_ANALYSIS_INPUT_VERSION", "v2")
    monkeypatch.setenv("BRAIN_LLM_ANALYSIS_CACHE_ENABLED", "false")
    monkeypatch.setenv("BRAIN_LLM_ANALYSIS_CACHE_PATH", str(tmp_path))

    result = run_analysis_agent(evidence_package, llm_config={"provider": "fake", "model": "fake"})
    structured = result["structured_analysis"]
    diagnostics = structured["analysis_stage_diagnostics"]

    assert diagnostics["llm_analysis_status"] == "invalid_output"
    assert diagnostics["claim_repair_priorities"]
    assert structured["evidence_repair_priorities"][0]["claim_id"] == "bad_metric_claim"
    gap = structured["evidence_gap_ledger"][0]
    assert gap["claim_id"] == "bad_metric_claim"
    assert gap["source_stage"] == "analysis_claim_support"


def test_claim_builder_strict_layout_does_not_fill_metric_block_from_core_evidence():
    package = {
        "chapter_id": "ch_01",
        "chapter_title": "Metric validation",
        "core_evidence": [_evidence("EV-CORE")],
    }
    layout = {
        "chapter_id": "ch_01",
        "sections": [
            {
                "section_id": "ch_01_metric",
                "section_title": "Metric reconciliation",
                "block_type": "metric_reconciliation",
            }
        ],
    }
    structured = {
        "analysis_depth_quality": {"status": "needs_rewrite", "repeated_claim_ratio": 0.9},
        "claim_units": [],
    }

    units = run_claim_builder_agent(
        chapter_evidence_packages=[package],
        micro_layouts=[layout],
        structured_analysis=structured,
    )

    assert units == []


def test_claim_builder_strict_layout_uses_matching_metric_evidence():
    metric = _evidence("EV-METRIC", proof_role="metric")
    metric["metric"] = "adoption"
    metric["value"] = "42%"
    package = {
        "chapter_id": "ch_01",
        "chapter_title": "Metric validation",
        "metric_evidence": [metric],
    }
    layout = {
        "chapter_id": "ch_01",
        "sections": [
            {
                "section_id": "ch_01_metric",
                "section_title": "Metric reconciliation",
                "block_type": "metric_reconciliation",
            }
        ],
    }
    structured = {
        "analysis_depth_quality": {"status": "needs_rewrite", "repeated_claim_ratio": 0.9},
        "claim_units": [],
    }

    units = run_claim_builder_agent(
        chapter_evidence_packages=[package],
        micro_layouts=[layout],
        structured_analysis=structured,
    )

    assert len(units) == 1
    assert units[0]["block_type"] == "metric_reconciliation"
    assert units[0]["evidence_backed"] is True
    assert units[0]["mechanism"]
    assert units[0]["mechanism"] != metric["fact"]
    public_text = "\n".join(
        str(units[0].get(key) or "")
        for key in ("claim", "reasoning", "counter_evidence", "actionable")
    )
    assert "这组事实更适合作为方向性判断" not in public_text
    assert "不能直接外推" not in public_text
    assert "事实锚点显示" not in public_text


def test_claim_builder_fact_card_directional_section_has_no_template_tail():
    card = {
        "subject": "Salesforce",
        "action": "disclosed",
        "object": "Salesforce disclosed Agentforce as an AI Agent layer used in customer service workflows.",
        "fact": "Salesforce disclosed Agentforce as an AI Agent layer used in customer service workflows.",
        "source_ref": "S1",
        "source_level": "C",
        "fact_type": "case",
        "analysis_variable": "客户落地与应用场景",
        "variable": "客户落地与应用场景",
        "block_affinity": ["customer_painpoint_matrix"],
        "claim_strength_hint": "directional",
        "directional_only": True,
    }
    evidence = {
        "ref": "E1",
        "source_ref": "S1",
        "source_level": "C",
        "allowed_use": "directional_signal",
        "public_fact_quality": {"eligible_for_report": True, "public_fact_card": card},
        "public_fact_card": card,
        "fact": card["fact"],
    }
    package = {
        "chapter_id": "ch_01",
        "chapter_title": "Demand validation",
        "directional_evidence": [evidence],
        "chapter_analysis": {"fact_cards": [card], "claim_strength": "directional"},
    }
    layout = {
        "chapter_id": "ch_01",
        "sections": [
            {
                "section_id": "ch_01_case",
                "section_title": "Customer signal",
                "block_type": "customer_painpoint_matrix",
            }
        ],
    }

    units = run_claim_builder_agent(
        chapter_evidence_packages=[package],
        micro_layouts=[layout],
        structured_analysis={"analysis_depth_quality": {"status": "needs_rewrite"}, "claim_units": []},
    )

    assert len(units) == 1
    text = "\n".join(str(units[0].get(key) or "") for key in ("claim", "reasoning", "counter_evidence"))
    assert units[0]["claim_strength"] == "directional"
    assert units[0]["evidence_backed"] is True
    assert "该信号可作为本章的审慎结论" not in text
    assert "边界在于样本是否代表主流需求" not in text
    assert "事实依据包括" not in text
    assert "可复核事实显示" not in text
    assert "可核验事实显示" not in text
    assert "若相反样本或口径差异扩大" not in text
    assert "分析重点是这些事实之间是否指向同一变量" not in text


def test_claim_builder_technology_block_does_not_consume_metric_card():
    card = {
        "subject": "Market source",
        "action": "shows",
        "object": "AI Agent market size reached 10 billion dollars in the sample dataset.",
        "fact": "AI Agent market size reached 10 billion dollars in the sample dataset.",
        "source_ref": "S1",
        "source_level": "B",
        "fact_type": "metric",
        "analysis_variable": "指标口径与可比性",
        "variable": "指标口径与可比性",
        "block_affinity": ["metric_reconciliation"],
        "claim_strength_hint": "moderate",
    }
    metric = {
        "ref": "E1",
        "source_ref": "S1",
        "source_level": "B",
        "metric": "market_size",
        "value": "10",
        "public_fact_quality": {"eligible_for_report": True, "public_fact_card": card},
        "public_fact_card": card,
        "fact": card["fact"],
    }
    package = {
        "chapter_id": "ch_01",
        "chapter_title": "Technology maturity",
        "metric_evidence": [metric],
        "chapter_analysis": {"fact_cards": [card], "claim_strength": "moderate"},
    }
    layout = {
        "chapter_id": "ch_01",
        "sections": [
            {
                "section_id": "ch_01_tech",
                "section_title": "Technology maturity",
                "block_type": "technology_maturity",
            }
        ],
    }

    units = run_claim_builder_agent(
        chapter_evidence_packages=[package],
        micro_layouts=[layout],
        structured_analysis={"analysis_depth_quality": {"status": "needs_rewrite"}, "claim_units": []},
    )

    assert units == []


# ---------------------------------------------------------------------------
# Regression tests for the optimization pass — keep contract drift visible.
# ---------------------------------------------------------------------------


def test_chapter_insights_uses_evidence_chapter_id_for_english_dimensions():
    """`_chapter_insights_from_synthesis` must not normalize away spaces.

    Previously the fallback produced `"demandvalidation"` while the
    diagnostics keyed off the raw `"demand validation"`, so
    `claim_binding_feedback_summary` could not find the chapter. The fix
    keeps the raw dimension string when no explicit chapter_id is known.
    """

    dimension_synthesis = {
        "demand validation": {
            "takeaway": "需求信号已经出现，但需要持续验证。",
            "evidence_ids": ["EV-1"],
            "claim_strength": "moderate",
        }
    }
    insights = _chapter_insights_from_synthesis(dimension_synthesis, {})
    assert len(insights) == 1
    # Either the canonical chapter_id from a lookup wins, or we fall back to
    # the raw dimension string — never a synthetic "chapter_1" or a normalized
    # form that loses spaces.
    assert insights[0]["chapter_id"] in {"demand validation", "ch_demand"}


def test_chapter_insights_chapter_id_lookup_wins_over_dimension():
    insights = _chapter_insights_from_synthesis(
        {"demand validation": {"takeaway": "x", "evidence_ids": ["EV-1"]}},
        {"demand validation": "ch_42"},
    )
    assert insights[0]["chapter_id"] == "ch_42"


def test_norm_chapter_id_tolerates_connectors():
    assert _norm_chapter_id("ch_01") == _norm_chapter_id("ch-01")
    assert _norm_chapter_id("ch 01") == _norm_chapter_id("CH/01")
    assert _norm_chapter_id("") == ""


def test_matches_resolves_chapter_id_with_punctuation_difference():
    """`_matches` should bind a unit whose chapter_id only differs by punctuation."""

    unit = {"chapter_id": "ch-01"}
    package = {"chapter_id": "ch_01", "chapter_title": "irrelevant"}
    assert claim_builder_matches(unit, package) is True


def test_resolve_chapter_id_via_aliases():
    diagnostics = {
        "ch_01": {
            "chapter_id": "ch_01",
            "chapter_title": "Demand validation",
            "chapter_id_aliases": ["ch_01", "ch01", "Demand validation", "demandvalidation"],
        }
    }
    assert resolve_chapter_id(diagnostics, "Demand validation") == "ch_01"
    assert resolve_chapter_id(diagnostics, "ch-01") == "ch_01"
    assert resolve_chapter_id(diagnostics, "unknown") == ""


def test_chapter_evidence_diagnostics_emits_alias_set():
    evidence_analyses = [
        {
            "evidence_id": "EV-1",
            "chapter_id": "ch_01",
            "dimension": "Demand validation",
            "dimension_name": "需求验证",
            "source_level": "B",
            "allowed_use": "core_claim",
            "source": {"url": "https://example.org/page", "title": "ok"},
        }
    ]
    diagnostics = _chapter_evidence_diagnostics({}, evidence_analyses)
    chapter = next(iter(diagnostics.values()))
    aliases = chapter["chapter_id_aliases"]
    # Raw + normalized forms for every distinct id surface.
    assert "ch_01" in aliases
    assert "Demand validation" in aliases
    assert "需求验证" in aliases


def test_should_force_strict_claim_building_field_published():
    """When quality is poor, contract status should explicitly request strict mode."""

    evidence_package = {
        "analysis_ready_evidence": [
            {
                "evidence_id": "EV-1",
                "chapter_id": "ch_01",
                "fact": "x",
                "source_level": "B",
                "allowed_use": "core_claim",
                "source": {"url": "https://example.org/EV-1", "title": "ok"},
            }
        ],
    }
    structured = {
        "claim_units": [{"chapter_id": "ch_01", "claim": "generic", "evidence_refs": ["EV-1"]}],
        "chapter_insights": [{"chapter_id": "ch_01", "chapter_question": "Demand"}],
        "evidence_analyses": [{"evidence_id": "EV-1"}],
        "analysis_depth_quality": {"status": "needs_rewrite", "repeated_claim_ratio": 0.8},
    }
    result = ensure_valid_structured_analysis(structured, evidence_package)
    contract = result.get("analysis_contract_status") or {}
    assert contract.get("should_force_strict_claim_building") is True


def test_strip_orphan_citations_handles_bare_number_refs():
    """Sources stored as `"42"` (no brackets) must still be recognised."""

    markdown = "...[1] some text [42] more text [99]..."
    registry = [
        {"ref": "1", "title": "One"},
        {"ref": "[42]", "title": "Forty-two"},
    ]
    output = _strip_orphan_citations(markdown, registry)
    assert "[1]" in output
    assert "[42]" in output
    # `[99]` is an orphan — must be stripped.
    assert "[99]" not in output


def test_normalize_citation_ref_handles_large_numbers():
    assert _normalize_citation_ref("1234") == "[1234]"
    assert _normalize_citation_ref("[ 99 ]") == "[99]"
    assert _normalize_citation_ref("") == ""


def test_source_allowed_for_report_only_runs_topic_filter_when_topic_matches():
    """Off-topic queries should not apply the AI-Agent-specific blocklist."""

    petroleum_source = {
        "url": "https://www.sinopec.com/annual-report-2025",
        "title": "中国石油 2025 年报",
        "source_level": "B",
        "source_type": "official_data",
    }
    # A query unrelated to AI Agent → the topic filter must stay quiet.
    assert _source_allowed_for_report(petroleum_source, query="储能行业 2026") is True
    # A query that IS about AI Agent → the petroleum source should be excluded.
    assert _source_allowed_for_report(petroleum_source, query="ai agent 行业研究") is False


def test_readpage_evidence_id_includes_task_and_role_slots():
    """The same source url + index, seen by two tasks, must produce distinct ids."""

    base_card = {
        "subject": "ACME",
        "action_or_signal": "ships",
        "distilled_fact": "ACME shipped 1m units in 2025.",
        "fact_type": "metric",
        "variable": "shipments",
        "scope": "global",
        "value": "1m",
        "unit": "units",
    }
    card_a, _ = _validated_card(
        dict(base_card),
        source_url="https://reuters.com/news/12345",
        source_ref="S-1",
        source_level="B",
        verification_status="readpage_verified",
        proof_role="metric",
        chapter_id="ch_01",
        search_task={"task_id": "T1", "proof_role": "metric"},
        index=0,
    )
    card_b, _ = _validated_card(
        dict(base_card),
        source_url="https://reuters.com/news/12345",
        source_ref="S-1",
        source_level="B",
        verification_status="readpage_verified",
        proof_role="case",
        chapter_id="ch_02",
        search_task={"task_id": "T2", "proof_role": "case"},
        index=0,
    )
    assert card_a is not None and card_b is not None
    assert card_a["evidence_id"] != card_b["evidence_id"]


def test_validate_llm_analysis_output_early_return_carries_counter_fields():
    """The no-valid-refs early return must still surface count fields."""

    payload = {
        "chapter_synthesis": [
            {
                "chapter_id": "ch_01",
                "claim_units": [
                    {
                        "claim": "x",
                        "used_evidence_ids": ["EV-FAKE"],
                        "evidence_basis": ["a"],
                        "reasoning_chain": ["b"],
                        "limitation_boundary": ["c"],
                    }
                ],
            }
        ]
    }
    validation = validate_llm_analysis_output(payload, {})
    # Even on the early-exit path we publish the counts so downstream
    # diagnostics don't show `None`.
    for key in ("usable_claim_count", "dropped_claim_count", "usable_chapter_count"):
        assert key in validation
        assert validation[key] == 0


def test_load_stage_snapshot_returns_corrupt_on_bad_manifest(tmp_path, monkeypatch):
    """A truncated manifest.json must not crash the loader."""

    from rag_pipeline.cache import stage_snapshot_cache

    monkeypatch.setenv("STAGE_SNAPSHOT_CACHE_PATH", str(tmp_path))
    run_dir = tmp_path / "run-1" / "evidence_package"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text("{not json", encoding="utf-8")
    result = stage_snapshot_cache.load_stage_snapshot("run-1", "evidence_package")
    assert result["status"] == "corrupt"
    assert "error" in result


def _anchor_mismatch_fixture():
    """Claim paraphrases the evidence so lexical n-gram anchors miss while the
    semantics genuinely support it (the live-run 19/19 false-downgrade shape)."""

    evidence_package = {
        "analysis_ready_evidence": [
            {
                **_evidence("EV-prop", chapter_id="ch_03", level="B"),
                "fact": "广州市典型案例集显示，AI物业经理智能体已落地300多个项目，管理超2000万平米，助客户降低管理成本60-70%。",
                "distilled_fact": "广州市典型案例集显示，AI物业经理智能体已落地300多个项目，管理超2000万平米，助客户降低管理成本60-70%。",
                "source_title": "广州案例集",
            }
        ]
    }
    payload = {
        "chapter_synthesis": [
            {
                "chapter_id": "ch_03",
                "claim_units": [
                    {
                        "claim_id": "anchor-paraphrase",
                        "claim": "AI Agent在物业管理环节已实现规模化商业落地，并通过典型案例证明了显著降本增效成果。",
                        "claim_strength": "moderate",
                        "used_evidence_ids": ["EV-prop"],
                        "evidence_basis": ["广州市典型案例集显示AI物业经理智能体已落地300多个项目。"],
                        "reasoning": "案例集中的落地规模与降本数据支持该判断。",
                    }
                ],
            }
        ]
    }
    return evidence_package, payload


def test_anchor_mismatch_waived_when_semantic_judge_confirms_support(monkeypatch):
    monkeypatch.setattr(
        analysis_agent,
        "_llm_semantic_claim_support_judge",
        lambda **_kwargs: {"status": "supported", "reason": "facts directly support the claim", "confidence": 0.9},
    )
    evidence_package, payload = _anchor_mismatch_fixture()

    validation = validate_llm_analysis_output(payload, evidence_package, llm_config={"model": "judge"})

    assert validation["usable_claim_count"] == 1
    unit = validation["chapter_synthesis"][0]["claim_units"][0]
    # The judge's verdict outranks the lexical anchor miss: no downgrade.
    assert unit["claim_strength"] == "moderate"
    issue_counts = validation["llm_validation_issue_counts"]
    assert "claim_support_anchor_mismatch_downgraded" not in issue_counts
    assert issue_counts["claim_support_anchor_mismatch_waived_by_semantic_judge"] == 1


def test_anchor_mismatch_downgrades_when_semantic_judge_unavailable(monkeypatch):
    monkeypatch.setattr(
        analysis_agent,
        "_llm_semantic_claim_support_judge",
        lambda **_kwargs: {"status": "skipped_disabled_or_missing_config"},
    )
    evidence_package, payload = _anchor_mismatch_fixture()

    validation = validate_llm_analysis_output(payload, evidence_package, llm_config={"model": "judge"})

    assert validation["usable_claim_count"] == 1
    unit = validation["chapter_synthesis"][0]["claim_units"][0]
    # No semantic verdict: keep the conservative directional downgrade.
    assert unit["claim_strength"] == "directional"
    assert validation["llm_validation_issue_counts"]["claim_support_anchor_mismatch_downgraded"] == 1


def test_anchor_mismatch_defers_to_judge_partial_downgrade(monkeypatch):
    monkeypatch.setattr(
        analysis_agent,
        "_llm_semantic_claim_support_judge",
        lambda **_kwargs: {"status": "partial", "reason": "direction supported, strength not", "confidence": 0.7},
    )
    evidence_package, payload = _anchor_mismatch_fixture()

    validation = validate_llm_analysis_output(payload, evidence_package, llm_config={"model": "judge"})

    assert validation["usable_claim_count"] == 1
    unit = validation["chapter_synthesis"][0]["claim_units"][0]
    assert unit["claim_strength"] == "directional"
    issue_counts = validation["llm_validation_issue_counts"]
    # The judge's own downgrade is the verdict of record; the anchor miss is
    # not double-counted as a second downgrade issue.
    assert issue_counts["llm_claim_semantic_judge_partial_downgraded"] == 1
    assert "claim_support_anchor_mismatch_downgraded" not in issue_counts
