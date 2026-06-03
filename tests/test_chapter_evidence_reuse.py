from rag_pipeline.agents.chapter_argument_agent import run_chapter_argument_agent


def test_same_evidence_id_is_not_rendered_twice_in_one_chapter():
    fact = "Salesforce Agentforce \u5ba2\u670d\u6d41\u7a0b\u90e8\u7f72\u5df2\u7ecf\u51fa\u73b0\u53ef\u8ffd\u8e2a\u5ba2\u6237\u52a8\u4f5c\u3002"
    packages = run_chapter_argument_agent(
        report_blueprint={"chapters": [{"chapter_id": "ch_01", "chapter_title": "Deployment"}]},
        micro_layouts=[
            {
                "chapter_id": "ch_01",
                "sections": [
                    {
                        "section_id": "ch_01_case",
                        "block_type": "case_comparison",
                        "dynamic_section_title": "Customer deployment",
                        "required_evidence_refs": ["E1"],
                    },
                    {
                        "section_id": "ch_01_competition",
                        "block_type": "competitive_positioning",
                        "dynamic_section_title": "Player movement",
                        "required_evidence_refs": ["E1"],
                    },
                ],
            }
        ],
        argument_units=[
            {
                "chapter_id": "ch_01",
                "section_id": "ch_01_case",
                "block_type": "case_comparison",
                "claim": "\u5ba2\u6237\u90e8\u7f72\u5df2\u7ecf\u51fa\u73b0\u3002",
                "reasoning": fact,
                "mechanism": fact,
                "supporting_facts": [fact],
                "used_fact_refs": ["E1"],
                "evidence_refs": ["E1"],
                "fact_card_to_block_match": True,
                "public_render": True,
            },
            {
                "chapter_id": "ch_01",
                "section_id": "ch_01_competition",
                "block_type": "competitive_positioning",
                "claim": "\u73a9\u5bb6\u52a8\u4f5c\u5df2\u7ecf\u51fa\u73b0\u3002",
                "reasoning": fact,
                "mechanism": fact,
                "supporting_facts": [fact],
                "used_fact_refs": ["E1"],
                "evidence_refs": ["E1"],
                "fact_card_to_block_match": True,
                "public_render": True,
            },
        ],
        chapter_evidence_packages=[
            {
                "chapter_id": "ch_01",
                "case_evidence": [{"evidence_id": "E1", "ref": "E1", "distilled_fact": fact}],
                "supporting_evidence": [{"evidence_id": "E1", "ref": "E1", "distilled_fact": fact}],
            }
        ],
    )

    sections = packages[0]["sections"]
    rendered_ref_sets = [
        tuple(section.get("used_fact_refs") or section.get("evidence_refs") or [])
        for section in sections
    ]
    assert rendered_ref_sets.count(("E1",)) == 1
    assert any(
        item.get("reason") == "repeated_evidence_id_within_chapter"
        for item in packages[0].get("dropped_sections", [])
    )


def test_llm_analysis_claims_can_reuse_same_ref_for_distinct_angles():
    packages = run_chapter_argument_agent(
        report_blueprint={"chapters": [{"chapter_id": "ch_01", "chapter_title": "Deployment"}]},
        micro_layouts=[
            {
                "chapter_id": "ch_01",
                "sections": [
                    {
                        "section_id": "ch_01_case",
                        "block_type": "case_comparison",
                        "dynamic_section_title": "Customer deployment",
                        "required_evidence_refs": ["E1"],
                    },
                    {
                        "section_id": "ch_01_integration",
                        "block_type": "integrated_signal",
                        "dynamic_section_title": "Workflow mechanism",
                        "required_evidence_refs": ["E1"],
                    },
                ],
            }
        ],
        argument_units=[
            {
                "claim_id": "CL-1",
                "chapter_id": "ch_01",
                "section_id": "ch_01_case",
                "block_type": "case_comparison",
                "claim": "Customer deployment has moved beyond generic product demos.",
                "reasoning": "The deployment signal is useful for judging whether demand exists.",
                "mechanism": "The deployment signal is useful for judging whether demand exists.",
                "supporting_facts": ["The cited fact card shows customer workflow deployment."],
                "used_fact_refs": ["E1"],
                "evidence_refs": ["E1"],
                "evidence_basis": ["The cited fact card shows customer workflow deployment."],
                "source_support_map": {"claim": ["E1"], "mechanism": ["E1"]},
                "claim_strength": "directional",
                "fact_card_to_block_match": True,
                "public_render": True,
            },
            {
                "claim_id": "CL-2",
                "chapter_id": "ch_01",
                "section_id": "ch_01_integration",
                "block_type": "integrated_signal",
                "claim": "The same deployment evidence also points to integration and permission-control work.",
                "reasoning": "The mechanism angle is different from the demand-validation angle.",
                "mechanism": "The mechanism angle is different from the demand-validation angle.",
                "supporting_facts": ["The cited fact card is used here to explain integration requirements."],
                "used_fact_refs": ["E1"],
                "evidence_refs": ["E1"],
                "evidence_basis": ["The cited fact card is used here to explain integration requirements."],
                "source_support_map": {"claim": ["E1"], "mechanism": ["E1"]},
                "claim_strength": "directional",
                "fact_card_to_block_match": True,
                "public_render": True,
            },
        ],
    )

    sections = packages[0]["sections"]
    assert [section.get("claim_id") for section in sections] == ["CL-1", "CL-2"]
    assert not any(
        item.get("reason") == "repeated_evidence_id_within_chapter"
        for item in packages[0].get("dropped_sections", [])
    )


def test_same_fact_text_is_not_rendered_twice_with_different_refs():
    fact = "AI Agent\u5546\u4e1a\u5316\u843d\u5730\uff0c3.3\u4e07\u4ebf\u8d5b\u9053\u52a0\u901f\u7206\u53d1\u3002"
    packages = run_chapter_argument_agent(
        report_blueprint={"chapters": [{"chapter_id": "ch_01", "chapter_title": "Commercialization"}]},
        micro_layouts=[
            {
                "chapter_id": "ch_01",
                "sections": [
                    {
                        "section_id": "ch_01_case",
                        "block_type": "case_comparison",
                        "dynamic_section_title": "Case signal",
                        "required_evidence_refs": ["E1"],
                    },
                    {
                        "section_id": "ch_01_competition",
                        "block_type": "competitive_positioning",
                        "dynamic_section_title": "Player signal",
                        "required_evidence_refs": ["E2"],
                    },
                ],
            }
        ],
        argument_units=[
            {
                "chapter_id": "ch_01",
                "section_id": "ch_01_case",
                "block_type": "case_comparison",
                "claim": fact,
                "reasoning": fact,
                "mechanism": fact,
                "supporting_facts": [fact],
                "used_fact_refs": ["E1"],
                "evidence_refs": ["E1"],
                "fact_card_to_block_match": True,
                "public_render": True,
            },
            {
                "chapter_id": "ch_01",
                "section_id": "ch_01_competition",
                "block_type": "competitive_positioning",
                "claim": fact,
                "reasoning": fact,
                "mechanism": fact,
                "supporting_facts": [fact],
                "used_fact_refs": ["E2"],
                "evidence_refs": ["E2"],
                "fact_card_to_block_match": True,
                "public_render": True,
            },
        ],
    )

    public_text = "\n".join(
        str(value or "")
        for section in packages[0]["sections"]
        for value in (section.get("claim"), section.get("reasoning"), section.get("mechanism"))
    )
    assert public_text.count(fact) == 1
    assert any(
        item.get("reason") == "repeated_fact_within_chapter"
        for item in packages[0].get("dropped_sections", [])
    )
