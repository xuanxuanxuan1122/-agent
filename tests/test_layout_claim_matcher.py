from rag_pipeline.agents.block_schema import select_blocks_for_chapter
from rag_pipeline.agents.claim_builder_agent import run_claim_builder_agent
from rag_pipeline.agents.chapter_argument_agent import run_chapter_argument_agent
from rag_pipeline.agents.micro_layout_agent import run_micro_layout_agent
from rag_pipeline.agents.layout_claim_matcher import (
    claim_supported_block_types,
    fallback_block_for_claim,
    match_claims_to_blocks,
)


def _claim(**overrides):
    payload = {
        "chapter_id": "ch_01",
        "claim": "Enterprise AI Agent demand is moving into workflow deployment.",
        "used_fact_refs": ["EV-1"],
        "evidence_refs": ["EV-1"],
        "evidence_basis": ["Salesforce disclosed customer-service workflow deployments."],
        "reasoning": "Workflow deployment requires repeatable operations and permission controls.",
        "claim_strength": "moderate",
        "public_render": True,
    }
    payload.update(overrides)
    return payload


def test_claim_supported_block_types_uses_block_affinity():
    claim = _claim(block_affinity=["case_comparison"], fact_type="case")

    assert claim_supported_block_types(claim)[:1] == ["case_comparison"]


def test_claim_without_affinity_falls_back_to_integrated_signal():
    claim = _claim(block_affinity=[], fact_type="")

    assert fallback_block_for_claim(claim) == "integrated_signal"
    assert "integrated_signal" in claim_supported_block_types(claim)


def test_match_claims_to_blocks_is_one_to_one_and_uses_fallback():
    claims = [
        _claim(evidence_refs=["EV-1"], block_affinity=["case_comparison"]),
        _claim(evidence_refs=["EV-2"], used_fact_refs=["EV-2"], block_affinity=[]),
    ]
    blocks = [
        {"block_id": "b1", "block_type": "case_comparison"},
        {"block_id": "b2", "block_type": "integrated_signal"},
    ]

    result = match_claims_to_blocks("ch_01", claims, blocks)

    assert result["matches"]["b1"]["evidence_refs"] == ["EV-1"]
    assert result["matches"]["b2"]["evidence_refs"] == ["EV-2"]
    assert result["matched_count"] == 2
    assert result["unmatched_count"] == 0


def test_select_blocks_can_promote_llm_claim_supported_blocks():
    blocks = select_blocks_for_chapter(
        {"chapter_id": "ch_01", "module_keys": ["technology"]},
        evidence_package={"chapter_id": "ch_01"},
        claim_units_by_chapter={"ch_01": [_claim(block_affinity=["case_comparison"], fact_type="case")]},
        limit=3,
    )

    must_blocks = [block for block in blocks if block.get("render_plan") == "must_render"]
    assert [block["block_type"] for block in must_blocks] == ["case_comparison"]
    assert must_blocks[0]["selection_reason"] == "llm_claim_supported"


def test_micro_layout_uses_structured_analysis_claims_for_must_blocks():
    layouts = run_micro_layout_agent(
        report_blueprint={"chapters": [{"chapter_id": "ch_01", "module_keys": ["technology"]}]},
        chapter_evidence_packages=[{"chapter_id": "ch_01"}],
        structured_analysis={"claim_units": [_claim(block_affinity=["case_comparison"], fact_type="case")]},
    )

    must_blocks = layouts[0]["must_render_blocks"]
    assert [block["block_type"] for block in must_blocks] == ["case_comparison"]
    assert layouts[0]["claim_layout_match_diagnostics"]["llm_claim_to_block_match_count"] == 1


def test_llm_claims_flow_through_layout_to_evidence_backed_sections(monkeypatch):
    monkeypatch.setenv("REPORT_ENABLE_LLM_BODY_REWRITE", "false")
    chapters = [
        {"chapter_id": "ch_01", "module_keys": ["technology"], "chapter_title": "Demand"},
        {"chapter_id": "ch_02", "module_keys": ["technology"], "chapter_title": "Competition"},
    ]
    claims = [
        _claim(
            chapter_id="ch_01",
            block_affinity=["case_comparison"],
            fact_type="case",
            evidence_refs=["EV-1"],
            used_fact_refs=["EV-1"],
            evidence_basis=["Salesforce disclosed workflow deployments for enterprise users."],
            supporting_facts=["Salesforce disclosed workflow deployments for enterprise users."],
        ),
        _claim(
            chapter_id="ch_02",
            block_affinity=[],
            fact_type="",
            evidence_refs=["EV-2"],
            used_fact_refs=["EV-2"],
            evidence_basis=["Several vendors described early customer demand for workflow agents."],
            supporting_facts=["Several vendors described early customer demand for workflow agents."],
        ),
    ]
    structured_analysis = {"claim_units": claims}
    chapter_evidence_packages = [{"chapter_id": "ch_01"}, {"chapter_id": "ch_02"}]

    layouts = run_micro_layout_agent(
        report_blueprint={"chapters": chapters},
        chapter_evidence_packages=chapter_evidence_packages,
        structured_analysis=structured_analysis,
    )
    units = run_claim_builder_agent(
        chapter_evidence_packages=chapter_evidence_packages,
        micro_layouts=layouts,
        structured_analysis=structured_analysis,
    )
    packages = run_chapter_argument_agent(
        report_blueprint={"chapters": chapters},
        micro_layouts=layouts,
        argument_units=units,
        chapter_evidence_packages=chapter_evidence_packages,
    )

    rendered = [section for package in packages for section in package.get("sections", [])]
    assert len(rendered) >= 2
    assert all(section.get("evidence_backed") for section in rendered)
    assert {section.get("body_composition_status") for section in rendered} == {"composed"}


def test_chapter_argument_preserves_analysis_claim_identity_and_support_map(monkeypatch):
    monkeypatch.setenv("REPORT_ENABLE_LLM_BODY_REWRITE", "false")
    claim = _claim(
        claim_id="claim_ch01_1",
        hypothesis_id="H1",
        requirement_ids=["H1_case"],
        claim_strength_ceiling="directional",
        lineage={"requirement_ids": ["H1_case"], "fact_ids": ["EV-1"], "source_ids": ["SRC-1"]},
        source_support_map={"claim": ["EV-1"], "mechanism": ["EV-1"], "boundary": ["EV-1"]},
        analysis_role="directional",
        paragraph_seed="Enterprise workflow deployment is a directional demand signal.",
        block_affinity=["case_comparison"],
        fact_type="case",
        supporting_facts=["Salesforce disclosed workflow deployments for enterprise users."],
    )
    chapters = [{"chapter_id": "ch_01", "module_keys": ["technology"], "chapter_title": "Demand"}]
    layouts = run_micro_layout_agent(
        report_blueprint={"chapters": chapters},
        chapter_evidence_packages=[{"chapter_id": "ch_01"}],
        structured_analysis={"claim_units": [claim]},
    )
    units = run_claim_builder_agent(
        chapter_evidence_packages=[{"chapter_id": "ch_01"}],
        micro_layouts=layouts,
        structured_analysis={"claim_units": [claim]},
    )
    packages = run_chapter_argument_agent(
        report_blueprint={"chapters": chapters},
        micro_layouts=layouts,
        argument_units=units,
        chapter_evidence_packages=[{"chapter_id": "ch_01"}],
    )

    section = packages[0]["sections"][0]
    assert section["claim_id"] == "claim_ch01_1"
    assert section["analysis_role"] == "directional"
    assert section["source_support_map"] == {"claim": ["EV-1"], "mechanism": ["EV-1"], "boundary": ["EV-1"]}
    assert section["hypothesis_id"] == "H1"
    assert section["requirement_ids"] == ["H1_case"]
    assert section["claim_strength_ceiling"] == "directional"
    assert section["lineage"]["requirement_ids"] == ["H1_case"]


def test_claim_builder_prefers_llm_claim_over_chapter_analysis_fallback():
    claim = _claim(
        claim_id="claim_ch01_llm",
        chapter_id="ch_01",
        block_affinity=["case_comparison"],
        fact_type="case",
        evidence_refs=["EV-1"],
        used_fact_refs=["EV-1"],
        evidence_basis=["The LLM analysis ties the Salesforce pilot to workflow demand."],
        supporting_facts=["The LLM analysis ties the Salesforce pilot to workflow demand."],
        reasoning="The pilot matters because workflow deployment is closer to repeatable production use than a generic demo.",
        mechanism="Workflow deployment requires integration into repeatable operations.",
        claim_strength="directional",
        analysis_role="directional",
        source_support_map={"claim": ["EV-1"], "mechanism": ["EV-1"], "boundary": ["EV-1"]},
    )
    layouts = [
        {
            "chapter_id": "ch_01",
            "sections": [
                {
                    "section_id": "ch_01_case",
                    "section_title": "Workflow demand signal",
                    "block_type": "case_comparison",
                    "output_type": "case_comparison",
                }
            ],
        }
    ]
    chapter_evidence_packages = [
        {
            "chapter_id": "ch_01",
            "chapter_title": "Demand",
            "chapter_question": "Is there workflow demand?",
            "case_evidence": [
                {
                    "evidence_id": "EV-1",
                    "source_ref": "[1]",
                    "public_fact_card": {
                        "fact": "Salesforce disclosed workflow deployment pilots.",
                        "fact_type": "case",
                        "block_affinity": ["case_comparison"],
                        "source_ref": "[1]",
                    },
                }
            ],
        }
    ]

    units = run_claim_builder_agent(
        chapter_evidence_packages=chapter_evidence_packages,
        micro_layouts=layouts,
        structured_analysis={"claim_units": [claim]},
    )

    assert units
    assert units[0]["claim_id"] == "claim_ch01_llm"
    assert units[0]["source_support_map"] == {"claim": ["EV-1"], "mechanism": ["EV-1"], "boundary": ["EV-1"]}


def test_claim_builder_emits_multiple_public_llm_claims_per_chapter(monkeypatch):
    monkeypatch.setenv("REPORT_EXTRA_LLM_CLAIMS_PER_CHAPTER", "4")
    claims = [
        _claim(
            claim_id=f"claim_ch01_{index}",
            chapter_id="ch_01",
            claim=f"Distinct analysis angle {index} should be rendered.",
            block_affinity=["case_comparison"] if index == 1 else ["integrated_signal"],
            fact_type="case",
            evidence_refs=[f"EV-{index}"],
            used_fact_refs=[f"EV-{index}"],
            evidence_basis=[f"Fact card {index} supports a distinct analysis angle."],
            supporting_facts=[f"Fact card {index} supports a distinct analysis angle."],
            reasoning=f"Reasoning chain {index} explains a different mechanism.",
            mechanism=f"Reasoning chain {index} explains a different mechanism.",
            claim_strength="directional",
            analysis_role="directional",
            source_support_map={"claim": [f"EV-{index}"], "mechanism": [f"EV-{index}"], "boundary": [f"EV-{index}"]},
        )
        for index in range(1, 4)
    ]

    chapter_evidence_packages = [
            {
                "chapter_id": "ch_01",
                "chapter_title": "Demand",
                "case_evidence": [
                    {
                        "evidence_id": f"EV-{index}",
                        "source_ref": f"[{index}]",
                        "public_fact_card": {
                            "fact": f"Fact card {index} supports a distinct analysis angle.",
                            "fact_type": "case",
                            "block_affinity": ["case_comparison"],
                            "source_ref": f"[{index}]",
                        },
                    }
                    for index in range(1, 4)
                ],
            }
        ]
    micro_layouts = [
            {
                "chapter_id": "ch_01",
                "sections": [
                    {
                        "section_id": "ch_01_case",
                        "section_title": "Workflow demand signal",
                        "block_type": "case_comparison",
                        "output_type": "case_comparison",
                    }
                ],
            }
        ]

    units = run_claim_builder_agent(
        chapter_evidence_packages=chapter_evidence_packages,
        micro_layouts=micro_layouts,
        structured_analysis={"claim_units": claims},
    )

    public_ids = [unit.get("claim_id") for unit in units if unit.get("public_render")]
    assert public_ids == ["claim_ch01_1", "claim_ch01_2", "claim_ch01_3"]

    packages = run_chapter_argument_agent(
        report_blueprint={"chapters": [{"chapter_id": "ch_01", "chapter_title": "Demand"}]},
        micro_layouts=micro_layouts,
        argument_units=units,
        chapter_evidence_packages=chapter_evidence_packages,
    )

    rendered_ids = [section.get("claim_id") for section in packages[0]["sections"]]
    assert rendered_ids == ["claim_ch01_1", "claim_ch01_2", "claim_ch01_3"]


def test_claim_builder_prefers_top_level_llm_claim_over_report_insight_claim():
    llm_claim = _claim(
        claim_id="claim_top_level_llm",
        chapter_id="ch_01",
        block_affinity=["case_comparison"],
        fact_type="case",
        evidence_refs=["EV-1"],
        used_fact_refs=["EV-1"],
        evidence_basis=["The top-level LLM ClaimUnit ties the pilot to workflow demand."],
        supporting_facts=["The top-level LLM ClaimUnit ties the pilot to workflow demand."],
        reasoning="The workflow pilot matters because it shows repeatable deployment pressure.",
        mechanism="Repeatable workflow deployment is closer to production demand than a generic demo.",
        claim_strength="directional",
        analysis_role="directional",
        source_support_map={"claim": ["EV-1"], "mechanism": ["EV-1"], "boundary": ["EV-1"]},
    )
    units = run_claim_builder_agent(
        chapter_evidence_packages=[
            {
                "chapter_id": "ch_01",
                "chapter_title": "Demand",
                "chapter_question": "Is there workflow demand?",
                "case_evidence": [{"evidence_id": "EV-1", "source_ref": "[1]"}],
            }
        ],
        micro_layouts=[
            {
                "chapter_id": "ch_01",
                "sections": [
                    {
                        "section_id": "ch_01_case",
                        "section_title": "Workflow demand signal",
                        "block_type": "case_comparison",
                    }
                ],
            }
        ],
        structured_analysis={
            "report_insight_package": {
                "chapters": [
                    {
                        "chapter_id": "ch_01",
                        "chapter_question": "Is there workflow demand?",
                        "key_claims": [
                            {
                                "claim": "The legacy report insight is less specific.",
                                "supporting_evidence": ["EV-1"],
                                "supporting_fact": "The legacy report insight references the same source.",
                            }
                        ],
                    }
                ]
            },
            "claim_units": [llm_claim],
        },
    )

    assert units
    assert units[0]["claim_id"] == "claim_top_level_llm"
    assert units[0]["analysis_role"] == "directional"


def test_claim_builder_matches_llm_claim_to_package_by_evidence_ref_when_chapter_alias_differs():
    claim = _claim(
        claim_id="claim_ch01_by_ref",
        chapter_id="ch_01",
        block_affinity=["case_comparison"],
        fact_type="case",
        evidence_refs=["EV-REF-1"],
        used_fact_refs=["EV-REF-1"],
        supporting_fact_refs=["EV-REF-1"],
        evidence_basis=["The LLM analysis uses the same evidence ref as the chapter package."],
        supporting_facts=["The LLM analysis uses the same evidence ref as the chapter package."],
        source_support_map={"claim": ["EV-REF-1"], "mechanism": ["EV-REF-1"], "boundary": ["EV-REF-1"]},
    )
    units = run_claim_builder_agent(
        chapter_evidence_packages=[
            {
                "chapter_id": "demand_validation",
                "chapter_title": "Demand validation",
                "case_evidence": [
                    {
                        "evidence_id": "EV-REF-1",
                        "source_ref": "[1]",
                        "public_fact_card": {"fact": "A customer deployment validates demand.", "source_ref": "[1]"},
                    }
                ],
            }
        ],
        micro_layouts=[
            {
                "chapter_id": "demand_validation",
                "sections": [
                    {
                        "section_id": "demand_case",
                        "section_title": "Demand signal",
                        "block_type": "case_comparison",
                    }
                ],
            }
        ],
        structured_analysis={"claim_units": [claim]},
    )

    assert units
    assert units[0]["claim_id"] == "claim_ch01_by_ref"
    assert units[0]["chapter_id"] == "demand_validation"


def test_claim_builder_renders_leftover_llm_claim_as_integrated_signal():
    claims = [
        _claim(
            claim_id="claim_ch01_case",
            chapter_id="ch_01",
            block_affinity=["case_comparison"],
            fact_type="case",
            evidence_refs=["EV-1"],
            used_fact_refs=["EV-1"],
            evidence_basis=["Case evidence supports the first section."],
            supporting_facts=["Case evidence supports the first section."],
        ),
        _claim(
            claim_id="claim_ch01_leftover",
            chapter_id="ch_01",
            block_affinity=[],
            fact_type="",
            evidence_refs=["EV-2"],
            used_fact_refs=["EV-2"],
            supporting_fact_refs=["EV-2"],
            evidence_basis=["A second valid LLM claim should still become a public integrated signal."],
            supporting_facts=["A second valid LLM claim should still become a public integrated signal."],
            claim_strength="directional",
            analysis_role="directional",
            source_support_map={"claim": ["EV-2"], "mechanism": ["EV-2"], "boundary": ["EV-2"]},
        ),
    ]

    units = run_claim_builder_agent(
        chapter_evidence_packages=[
            {
                "chapter_id": "ch_01",
                "chapter_title": "Demand",
                "case_evidence": [{"evidence_id": "EV-1", "source_ref": "[1]"}],
                "supporting_evidence": [{"evidence_id": "EV-2", "source_ref": "[2]"}],
            }
        ],
        micro_layouts=[
            {
                "chapter_id": "ch_01",
                "sections": [
                    {
                        "section_id": "ch_01_case",
                        "section_title": "Case signal",
                        "block_type": "case_comparison",
                    }
                ],
            }
        ],
        structured_analysis={"claim_units": claims},
    )

    claim_ids = [unit.get("claim_id") for unit in units]
    assert "claim_ch01_case" in claim_ids
    assert "claim_ch01_leftover" in claim_ids
    leftover = next(unit for unit in units if unit.get("claim_id") == "claim_ch01_leftover")
    assert leftover["block_type"] == "integrated_signal"
    assert leftover["section_id"].startswith("ch_01_llm")


def test_claim_builder_leftover_integrated_signal_prefers_llm_claim_over_legacy_insight(monkeypatch):
    monkeypatch.setenv("REPORT_EXTRA_LLM_CLAIMS_PER_CHAPTER", "1")
    llm_claim = _claim(
        claim_id="claim_leftover_llm_priority",
        chapter_id="ch_01",
        block_affinity=[],
        fact_type="",
        evidence_refs=["EV-2"],
        used_fact_refs=["EV-2"],
        evidence_basis=["The top-level LLM claim should win the one extra public section slot."],
        supporting_facts=["The top-level LLM claim should win the one extra public section slot."],
        claim_strength="directional",
        analysis_role="directional",
        source_support_map={"claim": ["EV-2"]},
    )
    units = run_claim_builder_agent(
        chapter_evidence_packages=[
            {
                "chapter_id": "ch_01",
                "chapter_title": "Demand",
                "case_evidence": [{"evidence_id": "EV-1", "source_ref": "[1]"}],
                "supporting_evidence": [{"evidence_id": "EV-2", "source_ref": "[2]"}],
            }
        ],
        micro_layouts=[
            {
                "chapter_id": "ch_01",
                "sections": [
                    {
                        "section_id": "ch_01_case",
                        "section_title": "Case signal",
                        "block_type": "case_comparison",
                    }
                ],
            }
        ],
        structured_analysis={
            "report_insight_package": {
                "chapters": [
                    {
                        "chapter_id": "ch_01",
                        "chapter_question": "Is there demand?",
                        "key_claims": [
                            {
                                "claim": "The legacy extra insight should not consume the only extra slot.",
                                "supporting_evidence": ["EV-2"],
                                "supporting_fact": "Legacy fact.",
                            }
                        ],
                    }
                ]
            },
            "claim_units": [
                _claim(
                    claim_id="claim_layout_case",
                    chapter_id="ch_01",
                    block_affinity=["case_comparison"],
                    fact_type="case",
                    evidence_refs=["EV-1"],
                    used_fact_refs=["EV-1"],
                    evidence_basis=["The layout section consumes this claim."],
                    supporting_facts=["The layout section consumes this claim."],
                ),
                llm_claim,
            ],
        },
    )

    assert any(unit.get("claim_id") == "claim_layout_case" for unit in units)
    assert any(unit.get("claim_id") == "claim_leftover_llm_priority" for unit in units)


def test_claim_builder_preserves_valid_llm_claim_under_strict_quality_warning():
    units = run_claim_builder_agent(
        chapter_evidence_packages=[{"chapter_id": "ch_01"}],
        micro_layouts=[],
        structured_analysis={
            "analysis_contract_status": {"should_force_strict_claim_building": True},
            "claim_units": [
                _claim(
                    claim_id="claim_1",
                    chapter_id="ch_01",
                    claim="Enterprise AI Agent demand has a directional workflow-deployment signal.",
                    evidence_refs=["EV-1"],
                    used_fact_refs=["EV-1"],
                    evidence_basis=["A traceable source describes enterprise workflow deployment."],
                    supporting_facts=["A traceable source describes enterprise workflow deployment."],
                    reasoning="The signal is directional because it reflects one traceable deployment context.",
                    mechanism="Workflow deployment matters because it requires integration into repeatable operations.",
                    claim_strength="directional",
                    analysis_role="directional",
                    source_support_map={"claim": ["EV-1"], "mechanism": ["EV-1"], "boundary": ["EV-1"]},
                    public_render=True,
                )
            ],
        },
    )

    assert units
    assert units[0]["public_render"] is True
    assert units[0].get("omit_from_report") is not True
    assert units[0]["claim_strength"] == "directional"
    assert units[0].get("internal_reason") not in {"public_blocking_language_or_missing_refs", "no_core_or_supporting_evidence"}
