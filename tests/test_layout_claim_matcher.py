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


def test_claim_builder_uses_required_refs_for_facts_and_citations(monkeypatch):
    monkeypatch.setenv("REPORT_TEMPLATE_FALLBACKS", "0")
    package = {
        "chapter_id": "ch_01",
        "chapter_title": "\u4f4e\u7a7a\u7ecf\u6d4e\u653f\u7b56\u4e0e\u65e0\u4eba\u673a\u914d\u9001",
        "core_evidence": [
            {
                "ref": "EV-policy",
                "evidence_id": "EV-policy",
                "source_ref": "[1]",
                "source_level": "A",
                "public_fact_quality": {
                    "eligible_for_report": True,
                    "public_fact_card": {
                        "fact": "\u4f4e\u7a7a\u4fdd\u9669\u4f53\u7cfb\u653f\u7b56\u5c06\u65e0\u4eba\u9a7e\u9a76\u822a\u7a7a\u5668\u8d23\u4efb\u4fdd\u9669\u4f5c\u4e3a\u5236\u5ea6\u5efa\u8bbe\u65b9\u5411\u3002",
                        "source_ref": "[1]",
                        "fact_type": "source_check",
                    },
                },
            },
            {
                "ref": "EV-case",
                "evidence_id": "EV-case",
                "source_ref": "[8]",
                "source_level": "C",
                "public_fact_quality": {
                    "eligible_for_report": True,
                    "public_fact_card": {
                        "fact": "\u4f5b\u5c71\u5e02\u6210\u7acb\u4f4e\u7a7a\u7ecf\u6d4e\u53d1\u5c55\u6709\u9650\u516c\u53f8\uff0c\u4ee5\u5e02\u573a\u5316\u8fd0\u4f5c\u63a8\u52a8\u4f4e\u7a7a\u7ecf\u6d4e\u4ea7\u4e1a\u53d1\u5c55\u3002",
                        "source_ref": "[8]",
                        "fact_type": "case",
                    },
                },
            },
        ],
    }
    layouts = [
        {
            "chapter_id": "ch_01",
            "sections": [
                {
                    "section_id": "ch_01_b1",
                    "section_title": "\u5e02\u573a\u9700\u6c42\u662f\u5426\u5df2\u7ecf\u51fa\u73b0",
                    "block_type": "integrated_signal",
                    "output_type": "integrated_signal",
                    "section_role": "integrated_signal",
                    "required_evidence_refs": ["EV-case"],
                }
            ],
        }
    ]

    units = run_claim_builder_agent(
        chapter_evidence_packages=[package],
        micro_layouts=layouts,
        structured_analysis={},
    )

    assert units
    unit = units[0]
    assert unit["public_render"] is True
    assert unit["omit_from_report"] is False
    assert "\u4f5b\u5c71\u5e02" in unit["claim"]
    assert "\u4f4e\u7a7a\u4fdd\u9669" not in unit["claim"]
    assert unit["evidence_refs"] == ["[8]"]
    assert "\u66f4\u9002\u5408\u8bf4\u660e" not in unit["claim"]
    assert "\u5c40\u90e8\u6d41\u7a0b" not in unit["claim"]


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


def test_claim_builder_does_not_repeat_chapter_analysis_fallback_after_matched_claim():
    claim = _claim(
        claim_id="claim_ch01_only",
        chapter_id="ch_01",
        claim="The low-altitude economy market-size signal is already sufficient for one section.",
        block_affinity=["case_comparison"],
        fact_type="case",
        evidence_refs=["EV-1"],
        used_fact_refs=["EV-1"],
        evidence_basis=["A traceable source reports a low-altitude economy market-size signal."],
        supporting_facts=["A traceable source reports a low-altitude economy market-size signal."],
        reasoning="The signal supports one bounded market-size section, not repeated layout filler.",
        mechanism="Market-size evidence can anchor a single commercialisation argument.",
        claim_strength="directional",
        analysis_role="directional",
        source_support_map={"claim": ["EV-1"], "mechanism": ["EV-1"], "boundary": ["EV-1"]},
    )
    units = run_claim_builder_agent(
        chapter_evidence_packages=[
            {
                "chapter_id": "ch_01",
                "chapter_title": "Market signal",
                "chapter_question": "How strong is the market-size signal?",
                "chapter_analysis": {"claim_strength": "directional"},
                "case_evidence": [
                    {
                        "evidence_id": "EV-1",
                        "source_ref": "[1]",
                        "public_fact_card": {
                            "fact": "A traceable source reports a low-altitude economy market-size signal.",
                            "fact_type": "case",
                            "block_affinity": ["case_comparison"],
                            "source_ref": "[1]",
                        },
                    }
                ],
            }
        ],
        micro_layouts=[
            {
                "chapter_id": "ch_01",
                "sections": [
                    {
                        "section_id": "ch_01_s1",
                        "section_title": "Market signal",
                        "block_type": "case_comparison",
                        "output_type": "case_comparison",
                    },
                    {
                        "section_id": "ch_01_s2",
                        "section_title": "Commercialisation signal",
                        "block_type": "case_comparison",
                        "output_type": "case_comparison",
                    },
                    {
                        "section_id": "ch_01_s3",
                        "section_title": "Player layout signal",
                        "block_type": "case_comparison",
                        "output_type": "case_comparison",
                    },
                ],
            }
        ],
        structured_analysis={"claim_units": [claim]},
    )

    public_units = [unit for unit in units if unit.get("public_render")]
    assert [unit.get("claim_id") for unit in public_units] == ["claim_ch01_only"]
    assert len({_normalize for _normalize in (unit.get("claim") for unit in public_units)}) == 1


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


def test_claim_builder_does_not_move_claim_to_other_existing_chapter_by_shared_ref():
    claims = [
        _claim(
            claim_id="claim_ch01",
            chapter_id="ch_01",
            claim="Chapter one should keep its own claim.",
            evidence_refs=["EV-SHARED"],
            used_fact_refs=["EV-SHARED"],
            evidence_basis=["Shared evidence can appear in more than one package."],
            supporting_facts=["Shared evidence can appear in more than one package."],
        ),
        _claim(
            claim_id="claim_ch04",
            chapter_id="ch_04",
            claim="Chapter four should not be rendered as a chapter one extra.",
            evidence_refs=["EV-SHARED"],
            used_fact_refs=["EV-SHARED"],
            evidence_basis=["Shared evidence can appear in more than one package."],
            supporting_facts=["Shared evidence can appear in more than one package."],
        ),
    ]
    common_fact = {
        "evidence_id": "EV-SHARED",
        "source_ref": "[1]",
        "public_fact_card": {
            "fact": "Shared evidence can appear in more than one package.",
            "fact_type": "case",
            "block_affinity": ["case_comparison"],
            "source_ref": "[1]",
        },
    }

    units = run_claim_builder_agent(
        chapter_evidence_packages=[
            {"chapter_id": "ch_01", "chapter_title": "Chapter one", "case_evidence": [common_fact]},
            {"chapter_id": "ch_04", "chapter_title": "Chapter four", "case_evidence": [common_fact]},
        ],
        micro_layouts=[
            {
                "chapter_id": "ch_01",
                "sections": [{"section_id": "ch_01_s1", "block_type": "case_comparison"}],
            },
            {
                "chapter_id": "ch_04",
                "sections": [{"section_id": "ch_04_s1", "block_type": "case_comparison"}],
            },
        ],
        structured_analysis={"claim_units": claims},
    )

    public_units = [unit for unit in units if unit.get("public_render")]
    by_claim = {unit.get("claim_id"): unit for unit in public_units}
    assert by_claim["claim_ch01"]["chapter_id"] == "ch_01"
    assert by_claim["claim_ch04"]["chapter_id"] == "ch_04"
    assert [unit.get("claim_id") for unit in public_units].count("claim_ch04") == 1


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
