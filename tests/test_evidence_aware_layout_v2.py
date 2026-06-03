from rag_pipeline.agents.block_schema import select_blocks_for_chapter
from rag_pipeline.agents.claim_builder_agent import run_claim_builder_agent
from rag_pipeline.agents.chapter_argument_agent import run_chapter_argument_agent
from rag_pipeline.agents.report_health import build_report_health_card


def test_layout_demotes_missing_metric_block_to_candidate_not_must_render():
    blocks = select_blocks_for_chapter(
        {"chapter_id": "ch_01", "module_keys": ["market_size"]},
        evidence_package={
            "chapter_id": "ch_01",
            "case_evidence": [
                {
                    "evidence_id": "EV-CASE",
                    "public_fact_quality": {"eligible_for_report": True},
                    "public_fact_card": {
                        "distilled_fact": "Salesforce disclosed customer workflow deployment.",
                        "block_affinity": ["case_comparison"],
                        "fact_type": "case",
                    },
                }
            ],
        },
    )

    must_blocks = [block["block_type"] for block in blocks if block.get("render_plan") == "must_render"]
    candidate_blocks = [block["block_type"] for block in blocks if block.get("render_plan") == "candidate"]
    assert "metric_reconciliation" not in must_blocks
    assert "metric_reconciliation" in candidate_blocks


def test_layout_uses_integrated_signal_when_fact_cards_have_no_specific_affinity():
    blocks = select_blocks_for_chapter(
        {"chapter_id": "ch_01", "module_keys": ["market_size"]},
        evidence_package={
            "chapter_id": "ch_01",
            "directional_evidence": [
                {
                    "evidence_id": "EV-GEN",
                    "source_ref": "[1]",
                    "source_level": "C",
                    "public_fact_quality": {"eligible_for_report": True},
                    "public_fact_card": {
                        "distilled_fact": "Several vendors described workflow-agent adoption as an early customer signal.",
                        "fact_type": "support",
                        "block_affinity": [],
                    },
                }
            ],
        },
    )

    must_blocks = [block["block_type"] for block in blocks if block.get("render_plan") == "must_render"]
    assert must_blocks == ["integrated_signal"]


def test_table_only_items_do_not_create_public_must_render_blocks():
    blocks = select_blocks_for_chapter(
        {"chapter_id": "ch_05", "module_keys": ["technology"]},
        evidence_package={
            "chapter_id": "ch_05",
            "table_evidence": [
                {
                    "evidence_id": "EV-TABLE",
                    "source_ref": "[9]",
                    "public_fact_quality": {"eligible_for_report": True},
                    "public_fact_card": {
                        "subject": "AI Agent vendor table",
                        "variable": "technology signal",
                        "distilled_fact": "The table lists technology keywords but has no validated body fact.",
                        "fact_type": "technology",
                        "block_affinity": ["technology_maturity"],
                    },
                }
            ],
        },
    )

    must_blocks = [block["block_type"] for block in blocks if block.get("render_plan") == "must_render"]
    candidate_blocks = [block["block_type"] for block in blocks if block.get("render_plan") == "candidate"]
    assert must_blocks == []
    assert "technology_maturity" in candidate_blocks


def test_case_affinity_does_not_make_metric_block_must_render():
    blocks = select_blocks_for_chapter(
        {"chapter_id": "ch_06", "module_keys": ["market_size"]},
        evidence_package={
            "chapter_id": "ch_06",
            "case_evidence": [
                {
                    "evidence_id": "EV-CASE",
                    "source_ref": "[8]",
                    "public_fact_quality": {"eligible_for_report": True},
                    "public_fact_card": {
                        "subject": "AI Agent commercial rollout",
                        "action_or_signal": "landed in a customer scenario",
                        "variable": "customer deployment",
                        "distilled_fact": "AI Agent commercial rollout appeared in a customer deployment case.",
                        "fact_type": "case",
                        "block_affinity": [
                            "metric_reconciliation",
                            "unit_economics",
                            "case_comparison",
                        ],
                    },
                }
            ],
        },
    )

    must_blocks = [block["block_type"] for block in blocks if block.get("render_plan") == "must_render"]
    assert "metric_reconciliation" not in must_blocks
    assert must_blocks in (["case_comparison"], ["unit_economics"], ["integrated_signal"])


def test_layout_feasibility_reports_missing_required_fact_fields():
    from rag_pipeline.agents.block_schema import can_render_block_from_evidence

    result = can_render_block_from_evidence(
        "metric_reconciliation",
        {
            "chapter_id": "ch_01",
            "metric_evidence": [
                {
                    "evidence_id": "EV-M",
                    "public_fact_quality": {"eligible_for_report": True},
                    "public_fact_card": {
                        "distilled_fact": "Market adoption reached 50%.",
                        "fact_type": "metric",
                        "block_affinity": ["metric_reconciliation"],
                    },
                }
            ],
        },
    )

    assert result["can_render"] is False
    assert result["reason"] == "missing_metric_subject_or_scope"
    assert "subject" in result["missing_fields"]
    assert "time_or_scope" in result["missing_fields"]


def test_optional_empty_chapter_does_not_make_healthcard_yellow():
    health = build_report_health_card(
        {
            "layout": {
                "must_render_block_count": 4,
                "rendered_must_block_count": 4,
                "layout_block_rendered_count": 4,
                "layout_block_evidence_backed_count": 4,
                "optional_chapter_omitted_count": 3,
                "core_chapter_omitted_no_evidence_count": 0,
            },
            "chapter_evidence": {"total_valid_fact_card_count": 4},
            "source_appendix_status": "ok",
            "body_composition_status": "composed",
            "summary_valid_judgment_count": 1,
        }
    )

    assert health["chapter_omitted_no_evidence_count"] == 0
    assert health["optional_chapter_omitted_count"] == 3
    assert health["metrics"]["chapter_omitted_no_evidence_count"]["status"] == "green"


def test_healthcard_ratio_uses_must_render_blocks_when_present():
    health = build_report_health_card(
        {
            "layout": {
                "must_render_block_count": 3,
                "rendered_must_block_count": 2,
                "candidate_block_count": 9,
                "layout_block_rendered_count": 2,
                "layout_block_dropped": 9,
                "layout_block_evidence_backed_count": 2,
            },
            "chapter_evidence": {"total_valid_fact_card_count": 5},
            "source_appendix_status": "ok",
            "body_composition_status": "composed",
            "summary_valid_judgment_count": 1,
        }
    )

    assert health["planned_vs_rendered_section_ratio"] == 0.667
    assert health["candidate_block_count"] == 9
    assert health["metrics"]["planned_vs_rendered_section_ratio"]["status"] == "green"


def test_must_render_block_has_claim_and_chapter_generation_path():
    chapter = {"chapter_id": "ch_01", "chapter_title": "Customer deployment", "module_keys": ["customer"]}
    evidence_package = {
        "chapter_id": "ch_01",
        "case_evidence": [
            {
                "evidence_id": "EV-CASE",
                "source_ref": "[1]",
                "source_level": "B",
                "public_fact_quality": {"eligible_for_report": True},
                "public_fact_card": {
                    "subject": "Salesforce Agentforce",
                    "variable": "customer workflow deployment",
                    "distilled_fact": "Salesforce disclosed customer-service workflow deployments.",
                    "fact_type": "case",
                    "block_affinity": ["case_comparison"],
                },
            }
        ],
    }
    blocks = select_blocks_for_chapter(chapter, evidence_package=evidence_package)
    must_blocks = [block for block in blocks if block.get("render_plan") == "must_render"]
    layout_sections = [
        {**block, "section_id": block.get("section_id") or block.get("block_id") or f"section_{index}"}
        for index, block in enumerate(must_blocks, start=1)
    ]

    units = run_claim_builder_agent(
        micro_layouts=[{"chapter_id": "ch_01", "sections": layout_sections}],
        chapter_evidence_packages=[evidence_package],
    )
    packages = run_chapter_argument_agent(
        report_blueprint={"chapters": [chapter]},
        micro_layouts=[{"chapter_id": "ch_01", "sections": layout_sections}],
        argument_units=units,
        chapter_evidence_packages=[evidence_package],
    )

    assert must_blocks
    assert packages[0]["sections"]
    assert packages[0]["sections"][0]["used_fact_refs"] == ["EV-CASE"]
