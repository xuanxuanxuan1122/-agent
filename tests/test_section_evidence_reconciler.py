from __future__ import annotations

from rag_pipeline.agents.claim_builder_agent import run_claim_builder_agent
from rag_pipeline.agents.section_evidence_reconciler import reconcile_section_evidence_refs
from rag_pipeline.agents.writer_agent_clean import build_writer_advice_plan
from rag_pipeline.contracts.repair_dispatcher import dispatch_repair_seed


def _clean_package():
    return {
        "chapter_id": "ch_01",
        "chapter_title": "\u4f4e\u7a7a\u7ecf\u6d4e\u653f\u7b56\u4e0e\u65e0\u4eba\u673a\u914d\u9001",
        "core_evidence": [
            {
                "ref": "EV-clean",
                "evidence_id": "EV-clean",
                "source_ref": "[1]",
                "source_level": "A",
                "public_fact_quality": {
                    "eligible_for_report": True,
                    "eligible_for_citation": True,
                    "public_fact_card": {
                        "fact": "\u4f5b\u5c71\u5e02\u6210\u7acb\u4f4e\u7a7a\u7ecf\u6d4e\u53d1\u5c55\u6709\u9650\u516c\u53f8\uff0c\u4ee5\u5e02\u573a\u5316\u8fd0\u4f5c\u63a8\u52a8\u4f4e\u7a7a\u7ecf\u6d4e\u4ea7\u4e1a\u53d1\u5c55\u3002",
                        "source_ref": "[1]",
                        "fact_type": "case",
                    },
                },
            }
        ],
    }


def _layout_with_stale_ref():
    return [
        {
            "chapter_id": "ch_01",
            "sections": [
                {
                    "section_id": "ch_01_s1",
                    "section_title": "\u5e02\u573a\u9700\u6c42\u662f\u5426\u5df2\u7ecf\u51fa\u73b0",
                    "block_type": "integrated_signal",
                    "output_type": "integrated_signal",
                    "section_role": "integrated_signal",
                    "required_evidence_refs": ["EV-stale"],
                }
            ],
        }
    ]


def test_reconciler_rebinds_stale_required_ref_to_clean_chapter_evidence():
    reconciled = reconcile_section_evidence_refs(
        micro_layouts=_layout_with_stale_ref(),
        chapter_evidence_packages=[_clean_package()],
        structured_analysis={},
    )

    section = reconciled[0]["sections"][0]
    assert section["planned_required_evidence_refs"] == ["EV-stale"]
    assert section["dropped_required_evidence_refs"] == ["EV-stale"]
    assert section["bound_evidence_refs"] == ["EV-clean"]
    assert section["ref_binding_status"] == "rebound"
    assert section["ref_binding_action"] == "write"
    assert section["rebound_evidence_refs"] == [
        {
            "from_ref": "EV-stale",
            "to_ref": "EV-clean",
            "reason": "same_chapter_clean_fact_fallback",
        }
    ]
    metrics = reconciled[0]["ref_binding_metrics"]
    assert metrics["planned_required_ref_count"] == 1
    assert metrics["valid_required_ref_count"] == 0
    assert metrics["dropped_required_ref_count"] == 1
    assert metrics["rebound_ref_count"] == 1
    assert metrics["unbound_section_count"] == 0


def test_claim_builder_uses_reconciled_bound_refs_instead_of_stale_required_refs():
    units = run_claim_builder_agent(
        chapter_evidence_packages=[_clean_package()],
        micro_layouts=_layout_with_stale_ref(),
        structured_analysis={},
    )

    public_units = [unit for unit in units if unit.get("public_render") and not unit.get("omit_from_report")]
    assert public_units
    unit = public_units[0]
    assert "\u4f5b\u5c71\u5e02" in unit["claim"]
    assert unit["used_fact_refs"] == ["[1]"]
    assert unit["section_ref_binding_status"] == "rebound"
    assert unit["planned_required_evidence_refs"] == ["EV-stale"]
    assert unit["bound_evidence_refs"] == ["EV-clean"]


def test_claim_builder_does_not_keep_invalid_only_stale_refs():
    package = {
        "chapter_id": "ch_01",
        "chapter_title": "\u4f4e\u7a7a\u7ecf\u6d4e\u653f\u7b56\u4e0e\u65e0\u4eba\u673a\u914d\u9001",
        "core_evidence": [],
        "supporting_evidence": [],
    }

    units = run_claim_builder_agent(
        chapter_evidence_packages=[package],
        micro_layouts=_layout_with_stale_ref(),
        structured_analysis={},
    )

    assert units
    for unit in units:
        refs = list(unit.get("used_fact_refs") or []) + list(unit.get("evidence_refs") or [])
        assert "EV-stale" not in refs
        assert unit.get("section_ref_binding_status") == "invalid_only"
        assert unit.get("ref_binding_action") is None or unit.get("section_ref_binding_action") == "reanalyze_existing"


def test_invalid_only_section_ref_binding_becomes_reanalysis_advice():
    package = {
        "chapter_id": "ch_01",
        "chapter_title": "\u4f4e\u7a7a\u7ecf\u6d4e\u653f\u7b56\u4e0e\u65e0\u4eba\u673a\u914d\u9001",
        "core_evidence": [],
        "supporting_evidence": [],
    }

    units = run_claim_builder_agent(
        chapter_evidence_packages=[package],
        micro_layouts=_layout_with_stale_ref(),
        structured_analysis={},
    )

    invalid_units = [unit for unit in units if unit.get("section_ref_binding_status") == "invalid_only"]
    assert invalid_units
    suggestion = invalid_units[0]["claim_review_suggestions"][0]
    assert suggestion["issue_type"] == "section_ref_binding_invalid_only"
    assert suggestion["suggested_action"] == "reanalyze_existing"
    assert suggestion["diagnostic_only"] is True
    assert suggestion["must_not_render"] is True
    assert suggestion["public_text_allowed"] is False
    assert suggestion["target"]["section_id"] == "ch_01_s1"
    assert suggestion["detail"]["dropped_required_evidence_refs"] == ["EV-stale"]

    seed = dispatch_repair_seed(suggestion)
    assert seed["repair_route"] == "reanalyze_existing"
    assert seed["allowed_for_writing"] is False

    advice_plan = build_writer_advice_plan(
        structured_analysis={"claim_units": invalid_units},
        report_plan={},
    )
    assert advice_plan["summary"]["reanalyze_existing_count"] >= 1
    assert advice_plan["chapter_actions"][0]["action"] == "reanalyze_existing"
    assert advice_plan["chapter_actions"][0]["must_not_render"] is True
