from rag_pipeline.agents.report_contracts import ClaimUnit, EvidenceFactCard
from rag_pipeline.agents.public_narrative_bridge import build_public_bridge_pack
from rag_pipeline.agents.section_composer import compose_section_paragraph


def test_claim_builder_preserves_claim_depth_pack_from_structured_analysis():
    from rag_pipeline.agents.claim_builder_agent import run_claim_builder_agent

    package = {
        "chapter_id": "ch_depth",
        "chapter_title": "Depth chapter",
        "supporting_evidence": [
            {
                "evidence_id": "EV-depth-1",
                "distilled_fact": "A named customer uses the workflow in a real support process.",
                "source_level": "C",
            }
        ],
    }
    depth_pack = {
        "schema_version": "claim_depth_pack_v1",
        "claim_id": "CL-depth",
        "judgement": "The workflow signal is commercially relevant.",
        "evidence_chain": "The source ties the signal to a named workflow rather than a generic announcement.",
        "mechanism": "That matters because workflow integration shows a repeatable path from trial to daily use.",
        "segmentation": "The signal is strongest for support teams and weaker for unstructured internal pilots.",
        "implication": "The report should treat the case as a commercialization signal.",
        "boundary": "The evidence is still directional because it comes from one public example.",
        "used_fact_refs": ["EV-depth-1"],
        "used_source_ids": ["SRC-depth-1"],
    }

    units = run_claim_builder_agent(
        chapter_evidence_packages=[package],
        structured_analysis={
            "claim_units": [
                {
                    "claim_id": "CL-depth",
                    "chapter_id": "ch_depth",
                    "section_id": "ch_depth_s1",
                    "claim": "The workflow signal is commercially relevant.",
                    "used_fact_refs": ["EV-depth-1"],
                    "evidence_refs": ["EV-depth-1"],
                    "evidence_basis": ["A named customer uses the workflow in a real support process."],
                    "claim_depth_pack": depth_pack,
                }
            ]
        },
    )

    assert units
    assert units[0].get("claim_depth_pack") == depth_pack


def test_section_composer_uses_claim_depth_pack_for_rich_claim_body(monkeypatch):
    monkeypatch.setenv("REPORT_BLUEPRINT_SOURCE", "claim_first")
    monkeypatch.setenv("REPORT_COMPOSER_EXPAND_TO_TARGET", "false")
    depth_pack = {
        "schema_version": "claim_depth_pack_v1",
        "claim_id": "CL-depth",
        "judgement": "The workflow signal is commercially relevant.",
        "evidence_chain": "The cited fact links the signal to a named customer workflow.",
        "mechanism": "Workflow integration matters because it connects product capability with repeatable daily use.",
        "segmentation": "The signal is stronger in support operations than in loose experimentation.",
        "implication": "This should be written as a commercialization signal, not as a market-wide conclusion.",
        "boundary": "The conclusion remains directional because the evidence is a single public example.",
        "used_fact_refs": ["EV-depth-1"],
        "used_source_ids": ["SRC-depth-1"],
    }
    result = compose_section_paragraph(
        fact_cards=[
            EvidenceFactCard.from_legacy_dict(
                {
                    "evidence_id": "EV-depth-1",
                    "chapter_id": "ch_depth",
                    "source_id": "SRC-depth-1",
                    "source_level": "C",
                    "public_fact_card": {
                        "subject": "Customer workflow",
                        "distilled_fact": "A named customer uses the workflow in a real support process.",
                        "fact_type": "case",
                    },
                }
            )
        ],
        claim_unit=ClaimUnit.from_legacy_dict(
            {
                "claim_id": "CL-depth",
                "claim": "The workflow signal is commercially relevant.",
                "used_fact_refs": ["EV-depth-1"],
                "claim_depth_pack": depth_pack,
            }
        ),
        block_type="case_comparison",
    )

    paragraph = result["paragraph"]
    assert result["composer_expansion_status"] == "claim_depth_pack"
    assert result["used_fact_refs"] == ["EV-depth-1"]
    assert "named customer workflow" in paragraph
    assert "repeatable daily use" in paragraph
    assert "single public example" in paragraph


def test_section_composer_expands_depth_pack_to_target_when_enabled(monkeypatch):
    monkeypatch.setenv("REPORT_BLUEPRINT_SOURCE", "claim_first")
    monkeypatch.setenv("REPORT_COMPOSER_EXPAND_TO_TARGET", "true")
    monkeypatch.setenv("REPORT_COMPOSER_TARGET_SECTION_CHARS", "520")
    depth_pack = {
        "schema_version": "claim_depth_pack_v1",
        "claim_id": "CL-depth-target",
        "judgement": "The demand signal is visible but still directional.",
        "evidence_chain": "The cited facts show live pilots and procurement activity.",
        "mechanism": "Pilots matter when they enter repeatable workflows.",
        "used_fact_refs": ["EV-depth-target-1"],
    }

    result = compose_section_paragraph(
        fact_cards=[
            EvidenceFactCard.from_legacy_dict(
                {
                    "evidence_id": "EV-depth-target-1",
                    "chapter_id": "ch_depth",
                    "source_id": "SRC-depth-target-1",
                    "source_level": "C",
                    "public_fact_card": {
                        "subject": "Low-altitude pilots",
                        "distilled_fact": "Several cities disclose low-altitude pilot routes and procurement activity.",
                        "fact_type": "case",
                        "variable": "pilot route deployment",
                    },
                }
            )
        ],
        claim_unit=ClaimUnit.from_legacy_dict(
            {
                "claim_id": "CL-depth-target",
                "claim": "The demand signal is visible but still directional.",
                "used_fact_refs": ["EV-depth-target-1"],
                "claim_depth_pack": depth_pack,
                "claim_strength": "directional",
            }
        ),
        block_type="case_comparison",
        chapter_question="Is low-altitude demand commercially repeatable?",
    )

    paragraph = result["paragraph"]
    assert result["composer_expansion_status"] == "claim_depth_pack_expanded"
    assert result["composer_paragraph_chars"] >= 520
    assert "live pilots and procurement activity" in paragraph
    assert "repeatable workflows" in paragraph
    assert "场景深度" in paragraph or "单点样本" in paragraph


def test_section_composer_long_expansion_uses_public_narrative_not_template_scaffold(monkeypatch):
    monkeypatch.setenv("REPORT_BLUEPRINT_SOURCE", "claim_first")
    monkeypatch.setenv("REPORT_COMPOSER_EXPAND_TO_TARGET", "true")
    monkeypatch.setenv("REPORT_COMPOSER_TARGET_SECTION_CHARS", "820")
    depth_pack = {
        "schema_version": "claim_depth_pack_v1",
        "claim_id": "CL-natural-depth",
        "judgement": "会计岗位能力正在从基础核算转向数据分析与业务理解。",
        "evidence_chain": "公开材料显示，财务会计类专业目录纳入财务大数据分析、智能成本核算和预算管理等能力模块。",
        "mechanism": "这些能力模块会改变学校课程、实训工具和企业筛选标准之间的连接方式。",
        "used_fact_refs": ["EV-natural-depth-1"],
    }

    result = compose_section_paragraph(
        fact_cards=[
            EvidenceFactCard.from_legacy_dict(
                {
                    "evidence_id": "EV-natural-depth-1",
                    "chapter_id": "ch_depth",
                    "source_id": "SRC-natural-depth-1",
                    "source_level": "B",
                    "public_fact_card": {
                        "subject": "财务会计类专业目录",
                        "distilled_fact": "专业目录提到财务大数据分析、智能成本核算和预算管理等能力模块。",
                        "fact_type": "policy",
                        "variable": "能力模块变化",
                    },
                }
            )
        ],
        claim_unit=ClaimUnit.from_legacy_dict(
            {
                "claim_id": "CL-natural-depth",
                "claim": "会计岗位能力正在从基础核算转向数据分析与业务理解。",
                "used_fact_refs": ["EV-natural-depth-1"],
                "claim_depth_pack": depth_pack,
                "claim_strength": "directional",
            }
        ),
        block_type="integrated_signal",
        chapter_question="AI如何改变会计学就业能力要求？",
    )

    paragraph = result["paragraph"]
    assert result["composer_expansion_status"] == "claim_depth_pack_expanded"
    assert result["composer_paragraph_chars"] >= 820
    assert "财务大数据分析" in paragraph
    assert "课程" in paragraph or "实训" in paragraph
    for forbidden in ("读者", "报告主线", "单一材料", "整体定论", "更适合作为有边界的分析材料"):
        assert forbidden not in paragraph


def test_section_composer_writer_advice_expands_claim_depth_pack_when_global_expand_disabled(monkeypatch):
    monkeypatch.setenv("REPORT_BLUEPRINT_SOURCE", "claim_first")
    monkeypatch.setenv("REPORT_COMPOSER_EXPAND_TO_TARGET", "false")
    monkeypatch.setenv("REPORT_COMPOSER_TARGET_SECTION_CHARS", "520")
    depth_pack = {
        "schema_version": "claim_depth_pack_v1",
        "claim_id": "CL-depth-advice",
        "judgement": "The workflow signal is visible but still directional.",
        "evidence_chain": "The cited fact ties the signal to a recurring customer-support workflow.",
        "mechanism": "Workflow depth matters because repeated use is more informative than a loose pilot.",
        "used_fact_refs": ["EV-depth-advice-1"],
    }

    result = compose_section_paragraph(
        fact_cards=[
            EvidenceFactCard.from_legacy_dict(
                {
                    "evidence_id": "EV-depth-advice-1",
                    "chapter_id": "ch_depth",
                    "source_id": "SRC-depth-advice-1",
                    "source_level": "C",
                    "public_fact_card": {
                        "subject": "Enterprise workflow",
                        "distilled_fact": "A public case describes AI agent use in a recurring customer-support workflow.",
                        "fact_type": "case",
                        "variable": "workflow deployment",
                    },
                }
            )
        ],
        claim_unit=ClaimUnit.from_legacy_dict(
            {
                "claim_id": "CL-depth-advice",
                "claim": "The workflow signal is visible but still directional.",
                "used_fact_refs": ["EV-depth-advice-1"],
                "claim_depth_pack": depth_pack,
                "claim_strength": "directional",
                "writer_advice_actions": [
                    {
                        "action": "expand_claim_writing",
                        "issue_type": "body_short",
                        "must_not_render": True,
                        "public_text_allowed": False,
                    }
                ],
            }
        ),
        block_type="case_comparison",
        chapter_question="Can public cases support commercialization analysis?",
    )

    paragraph = result["paragraph"]
    assert result["composer_expansion_status"] == "claim_depth_pack_writer_advice_expanded"
    assert result["composer_paragraph_chars"] >= 520
    assert "recurring customer-support workflow" in paragraph
    for token in (
        "writer_advice",
        "expand_claim_writing",
        "body_short",
        "must_not_render",
        "diagnostic_only",
        "方向性观察",
        "后续变化交叉验证",
        "后续是否重复出现",
        "后续判断应继续",
        "现有材料能够覆盖",
    ):
        assert token not in paragraph


def test_section_composer_uses_writer_advice_to_expand_without_internal_tokens(monkeypatch):
    monkeypatch.setenv("REPORT_BLUEPRINT_SOURCE", "claim_first")
    monkeypatch.setenv("REPORT_COMPOSER_EXPAND_TO_TARGET", "false")
    monkeypatch.setenv("REPORT_COMPOSER_TARGET_SECTION_CHARS", "520")

    result = compose_section_paragraph(
        fact_cards=[
            EvidenceFactCard.from_legacy_dict(
                {
                    "evidence_id": "EV-advice-1",
                    "chapter_id": "ch_advice",
                    "source_id": "SRC-advice-1",
                    "source_level": "C",
                    "public_fact_card": {
                        "subject": "Enterprise workflow",
                        "distilled_fact": "A public case describes AI agent use in a recurring customer-support workflow.",
                        "fact_type": "case",
                        "variable": "workflow deployment",
                    },
                }
            )
        ],
        claim_unit=ClaimUnit.from_legacy_dict(
            {
                "claim_id": "CL-advice",
                "claim": "The public case is a directional commercialization signal.",
                "used_fact_refs": ["EV-advice-1"],
                "claim_strength": "directional",
                "writer_advice_actions": [
                    {
                        "action": "expand_claim_writing",
                        "issue_type": "body_short",
                        "tone": "directional",
                        "writing_goal": "Expand with mechanism and boundary; do not render this instruction.",
                        "must_not_render": True,
                        "public_text_allowed": False,
                    }
                ],
                "writer_advice_tone": "directional",
                "writer_advice_do_not_add_new_facts": True,
            }
        ),
        block_type="case_comparison",
        chapter_question="Can public cases support commercialization analysis?",
    )

    paragraph = result["paragraph"]
    assert result["composer_expansion_status"] == "writer_advice_expanded"
    assert result["composer_paragraph_chars"] >= 520
    assert "recurring customer-support workflow" in paragraph
    assert "workflow deployment" in paragraph or "Enterprise workflow" in paragraph
    for token in (
        "writer_advice",
        "expand_claim_writing",
        "body_short",
        "must_not_render",
        "diagnostic_only",
        "方向性观察",
        "后续变化交叉验证",
        "后续是否重复出现",
        "后续判断应继续",
        "现有材料能够覆盖",
    ):
        assert token not in paragraph


def test_section_composer_uses_narrative_supporting_claims_without_plan_leak(monkeypatch):
    monkeypatch.setenv("REPORT_BLUEPRINT_SOURCE", "claim_first")
    monkeypatch.setenv("REPORT_COMPOSER_EXPAND_TO_TARGET", "false")

    result = compose_section_paragraph(
        fact_cards=[
            EvidenceFactCard.from_legacy_dict(
                {
                    "evidence_id": "EV-narrative-1",
                    "chapter_id": "ch_narrative",
                    "source_id": "SRC-narrative-1",
                    "source_level": "C",
                    "public_fact_card": {
                        "subject": "Enterprise workflow",
                        "distilled_fact": "A public case describes AI agent use in customer service.",
                        "fact_type": "case",
                        "variable": "customer service workflow",
                    },
                }
            ),
            EvidenceFactCard.from_legacy_dict(
                {
                    "evidence_id": "EV-narrative-2",
                    "chapter_id": "ch_narrative",
                    "source_id": "SRC-narrative-2",
                    "source_level": "C",
                    "public_fact_card": {
                        "subject": "Enterprise workflow",
                        "distilled_fact": "Another public case describes AI agent use in operations coordination.",
                        "fact_type": "case",
                        "variable": "operations coordination workflow",
                    },
                }
            ),
        ],
        claim_unit=ClaimUnit.from_legacy_dict(
            {
                "claim_id": "CL-narrative-main",
                "claim": "AI agent adoption is moving into repeatable enterprise workflows.",
                "used_fact_refs": ["EV-narrative-1", "EV-narrative-2"],
                "evidence_refs": ["EV-narrative-1", "EV-narrative-2"],
                "narrative_role": "evidence_progression",
                "narrative_supporting_claims": [
                    "Operations coordination is a second scenario signal.",
                ],
                "narrative_writing_goal": "internal planning only",
                "paragraph_plan_id": "ch_narrative_p01",
            }
        ),
        block_type="case_comparison",
    )

    paragraph = result["paragraph"]
    assert "Operations coordination is a second scenario signal" in paragraph
    assert "single sample" in paragraph or "场景信号" in paragraph
    for token in ("internal planning only", "paragraph_plan_id", "narrative_writing_goal", "writer_advice"):
        assert token not in paragraph


def test_rebuild_pipeline_adds_claim_depth_pack_when_missing(monkeypatch):
    from rag_pipeline.agents.writer_agent_clean import rebuild_public_argument_pipeline

    monkeypatch.setenv("REPORT_BLUEPRINT_SOURCE", "claim_first")
    package = {
        "chapter_id": "ch_depth",
        "chapter_title": "Depth chapter",
        "supporting_evidence": [
            {
                "evidence_id": "EV-depth-1",
                "source_ref": "EV-depth-1",
                "source_level": "C",
                "public_fact_card": {
                    "distilled_fact": "A named customer uses the workflow in a real support process.",
                    "fact_type": "case",
                    "block_affinity": ["case_comparison"],
                },
                "public_fact_quality": {"eligible_for_report": True},
            }
        ],
    }

    result = rebuild_public_argument_pipeline(
        chapter_evidence_packages=[package],
        micro_layouts=[],
        structured_analysis={
            "claim_units": [
                {
                    "claim_id": "CL-depth",
                    "chapter_id": "ch_depth",
                    "section_id": "ch_depth_s1",
                    "claim": "The workflow signal is commercially relevant.",
                    "used_fact_refs": ["EV-depth-1"],
                    "evidence_refs": ["EV-depth-1"],
                    "evidence_basis": ["A named customer uses the workflow in a real support process."],
                    "claim_strength": "directional",
                }
            ]
        },
        report_blueprint={"chapters": [{"chapter_id": "ch_depth", "chapter_title": "Depth chapter"}]},
        table_packages=[],
    )

    argument_unit = result["argument_units"][0]
    section = result["chapter_packages"][0]["sections"][0]
    assert argument_unit.get("claim_depth_pack", {}).get("schema_version") == "claim_depth_pack_v1"
    assert argument_unit["claim_depth_pack"]["used_fact_refs"] == ["EV-depth-1"]
    assert section.get("composer_expansion_status") == "claim_depth_pack"


def test_claim_builder_adds_depth_pack_on_direct_entrypoint(monkeypatch):
    from rag_pipeline.agents.claim_builder_agent import run_claim_builder_agent

    monkeypatch.setenv("REPORT_BLUEPRINT_SOURCE", "claim_first")
    package = {
        "chapter_id": "ch_direct",
        "chapter_title": "Direct entrypoint chapter",
        "supporting_evidence": [
            {
                "evidence_id": "EV-direct-1",
                "source_ref": "EV-direct-1",
                "source_level": "C",
                "public_fact_card": {
                    "distilled_fact": "Several cities disclose low-altitude pilot routes and procurement activity.",
                    "fact_type": "case",
                    "block_affinity": ["case_comparison"],
                },
                "public_fact_quality": {"eligible_for_report": True},
            }
        ],
    }

    units = run_claim_builder_agent(
        chapter_evidence_packages=[package],
        micro_layouts=[],
        structured_analysis={
            "claim_units": [
                {
                    "claim_id": "CL-direct",
                    "chapter_id": "ch_direct",
                    "claim": "The demand signal is visible but still directional.",
                    "used_fact_refs": ["EV-direct-1"],
                    "evidence_refs": ["EV-direct-1"],
                    "evidence_basis": ["Several cities disclose low-altitude pilot routes and procurement activity."],
                    "claim_strength": "directional",
                }
            ]
        },
    )

    assert units
    assert units[0].get("claim_depth_pack", {}).get("schema_version") == "claim_depth_pack_v1"
    assert units[0]["claim_depth_pack"]["used_fact_refs"] == ["EV-direct-1"]


def test_claim_depth_pack_fallback_matches_chinese_report_language():
    from rag_pipeline.agents.claim_deepener_agent import build_claim_depth_pack

    unit = {
        "claim_id": "CL-zh",
        "claim": "会计教育正在从传统核算训练转向数字化财务能力培养。",
        "used_fact_refs": ["EV-zh-1"],
    }
    evidence_by_ref = {
        "EV-zh-1": {
            "evidence_id": "EV-zh-1",
            "source_id": "SRC-zh-1",
            "distilled_fact": "多所高校公开材料提到会计专业增加数据分析和智能财务相关课程。",
        }
    }

    pack = build_claim_depth_pack(unit, evidence_by_ref=evidence_by_ref)

    assert "公开材料提到" in pack["evidence_chain"]
    assert "The available evidence" not in pack["evidence_chain"]
    assert "结论" in pack["boundary"]


def test_claim_depth_pack_fallback_uses_public_prose_not_writer_instructions():
    from rag_pipeline.agents.claim_deepener_agent import build_claim_depth_pack

    unit = {
        "claim_id": "CL-public-prose",
        "claim": "低空经济试点已经出现可观察的需求信号。",
        "used_fact_refs": ["EV-public-prose-1"],
    }
    evidence_by_ref = {
        "EV-public-prose-1": {
            "evidence_id": "EV-public-prose-1",
            "source_id": "SRC-public-prose-1",
            "distilled_fact": "多地公开材料披露低空航线试点和采购动作。",
        }
    }

    pack = build_claim_depth_pack(unit, evidence_by_ref=evidence_by_ref)
    public_text = " ".join(str(pack.get(key) or "") for key in ("segmentation", "implication", "evidence_chain"))

    assert "写作时应" not in public_text
    assert "报告可以把" not in public_text
    assert "这些材料中直接披露的动作" not in public_text
    assert "这一点能够作为分析展开的基础" not in public_text
    assert "主体行动是否持续" in public_text or "更可复核的分析结论" in public_text
    assert "订单、运营频次" not in public_text
    assert "低空航线试点和采购动作" in public_text
    for forbidden in ("公开事实包括", "这组事实", "这些事实", "这一判断", "围绕“", "后续应"):
        assert forbidden not in public_text


def test_claim_depth_pack_fallback_omits_internal_segmentation_template():
    from rag_pipeline.agents.claim_deepener_agent import build_claim_depth_pack

    unit = {
        "claim_id": "CL-public-template",
        "claim": "低空经济试点已经出现可观察的需求信号。",
        "used_fact_refs": ["EV-public-template-1"],
    }
    evidence_by_ref = {
        "EV-public-template-1": {
            "evidence_id": "EV-public-template-1",
            "source_id": "SRC-public-template-1",
            "distilled_fact": "多地公开材料披露低空航线试点和采购动作。",
        }
    }

    pack = build_claim_depth_pack(unit, evidence_by_ref=evidence_by_ref)
    public_text = " ".join(str(pack.get(key) or "") for key in ("segmentation", "implication", "evidence_chain"))

    assert "公开信息已经给出具体政策" not in public_text
    assert "背景性描述只用于补充理解" not in public_text
    assert "不单独承担结论" not in public_text
    assert "低空航线试点和采购动作" in public_text
    assert "低空经济试点已经出现可观察的需求信号" not in public_text


def test_claim_depth_pack_fallback_uses_block_specific_public_bridge():
    from rag_pipeline.agents.claim_deepener_agent import build_claim_depth_pack

    evidence_by_ref = {
        "EV-case": {
            "evidence_id": "EV-case",
            "source_id": "SRC-case",
            "distilled_fact": "公开案例显示企业在客服流程中测试智能体。",
        },
        "EV-risk": {
            "evidence_id": "EV-risk",
            "source_id": "SRC-risk",
            "distilled_fact": "媒体披露部分地区加强无人机飞行安全监管。",
        },
    }

    case_pack = build_claim_depth_pack(
        {
            "claim_id": "CL-case",
            "claim": "客户试点说明需求已经从概念讨论进入流程验证。",
            "used_fact_refs": ["EV-case"],
            "block_type": "case_comparison",
        },
        evidence_by_ref=evidence_by_ref,
    )
    risk_pack = build_claim_depth_pack(
        {
            "claim_id": "CL-risk",
            "claim": "安全监管会影响低空商业化推进节奏。",
            "used_fact_refs": ["EV-risk"],
            "block_type": "risk_trigger",
        },
        evidence_by_ref=evidence_by_ref,
    )

    assert "明确主体" in case_pack["segmentation"] or "具体场景" in case_pack["segmentation"]
    assert "风险" in risk_pack["segmentation"] or "约束" in risk_pack["segmentation"]
    assert case_pack["segmentation"] != risk_pack["segmentation"]


def _not_allowed_claim_review_suggestion() -> dict:
    return {
        "schema_version": "claim_review_suggestion_v1",
        "issue_type": "llm_claim_semantic_judge_unsupported",
        "diagnostic_only": True,
        "must_not_render": True,
        "public_text_allowed": False,
        "suggested_writing_permission": "repair_before_publication",
        "repair_priority": {
            "allowed_for_writing": False,
            "writing_permission": "not_allowed_until_repaired",
            "recommended_action": "repair_evidence_binding_then_rebuild_claim",
        },
    }


def test_claim_deepener_does_not_thicken_not_allowed_claims():
    from rag_pipeline.agents.claim_deepener_agent import build_claim_depth_pack

    unit = {
        "claim_id": "CL-bad",
        "claim": "AI模型Tokens价格下降会降低低空飞行器商业运营成本。",
        "used_fact_refs": ["EV-ai-1"],
        "claim_review_suggestions": [_not_allowed_claim_review_suggestion()],
    }
    evidence_by_ref = {
        "EV-ai-1": {
            "evidence_id": "EV-ai-1",
            "source_id": "SRC-ai-1",
            "distilled_fact": "AI模型Tokens价格出现下降。",
        }
    }

    assert build_claim_depth_pack(unit, evidence_by_ref=evidence_by_ref) == {}


def test_claim_builder_omits_not_allowed_until_repaired_claims():
    from rag_pipeline.agents.claim_builder_agent import run_claim_builder_agent

    units = run_claim_builder_agent(
        chapter_evidence_packages=[
            {
                "chapter_id": "ch_bad",
                "supporting_evidence": [
                    {
                        "evidence_id": "EV-ai-1",
                        "distilled_fact": "AI模型Tokens价格出现下降。",
                        "source_level": "C",
                    }
                ],
            }
        ],
        structured_analysis={
            "claim_units": [
                {
                    "claim_id": "CL-bad",
                    "chapter_id": "ch_bad",
                    "claim": "AI模型Tokens价格下降会降低低空飞行器商业运营成本。",
                    "evidence_refs": ["EV-ai-1"],
                    "claim_review_suggestions": [_not_allowed_claim_review_suggestion()],
                }
            ]
        },
    )

    assert units
    assert units[0]["public_render"] is False
    assert units[0]["omit_from_report"] is True
    assert units[0]["omit_reason"] == "claim_review_not_allowed_until_repaired"


def test_claim_builder_does_not_inherit_unrelated_package_facts_for_structured_claims():
    from rag_pipeline.agents.claim_builder_agent import run_claim_builder_agent

    units = run_claim_builder_agent(
        chapter_evidence_packages=[
            {
                "chapter_id": "ch_low_altitude",
                "chapter_title": "低空经济需求验证",
                "core_evidence": [
                    {
                        "evidence_id": "EV-related",
                        "source_ref": "EV-related",
                        "distilled_fact": "2023年低空飞行器制造和运营服务贡献近55%，基础设施和飞行保障潜力尚未充分显现。",
                        "public_fact_card": {
                            "distilled_fact": "2023年低空飞行器制造和运营服务贡献近55%，基础设施和飞行保障潜力尚未充分显现。",
                            "fact_type": "metric",
                        },
                    }
                ],
                "supporting_evidence": [
                    {
                        "evidence_id": "EV-unrelated",
                        "source_ref": "EV-unrelated",
                        "distilled_fact": "出货/部署为100家（2023年12）",
                        "metric": "出货/部署",
                        "value": "100家",
                        "content_shape_issues": ["generic_metric_name"],
                    }
                ],
            }
        ],
        structured_analysis={
            "claim_units": [
                {
                    "claim_id": "CL-infra",
                    "chapter_id": "ch_low_altitude",
                    "claim": "低空经济中基础设施与飞行保障环节发展滞后。",
                    "evidence_refs": ["EV-related"],
                    "evidence_basis": ["2023年低空飞行器制造和运营服务贡献近55%，基础设施和飞行保障潜力尚未充分显现。"],
                    "claim_strength": "directional",
                }
            ]
        },
    )

    assert units
    supporting_text = "\n".join(units[0].get("supporting_facts") or [])
    assert "基础设施和飞行保障潜力" in supporting_text
    assert "出货/部署" not in supporting_text
    assert "100家" not in supporting_text


def test_chapter_argument_does_not_fallback_to_unrelated_package_facts_for_explicit_refs():
    from rag_pipeline.agents.chapter_argument_agent import run_chapter_argument_agent

    packages = run_chapter_argument_agent(
        report_blueprint={
            "chapters": [
                {"chapter_id": "ch_low_altitude", "chapter_title": "低空经济需求验证"}
            ]
        },
        chapter_evidence_packages=[
            {
                "chapter_id": "ch_low_altitude",
                "chapter_title": "低空经济需求验证",
                "chapter_question": "低空经济是否有真实需求？",
                "core_evidence": [
                    {
                        "evidence_id": "EV-other",
                        "source_ref": "EV-other",
                        "distilled_fact": "近年来全球低空经济市场规模呈现出持续增长的态势。",
                        "public_fact_card": {
                            "distilled_fact": "近年来全球低空经济市场规模呈现出持续增长的态势。",
                            "fact_type": "counter",
                            "block_affinity": ["risk_trigger"],
                        },
                    }
                ],
            }
        ],
        argument_units=[
            {
                "claim_id": "CL-infra",
                "chapter_id": "ch_low_altitude",
                "section_id": "sec-infra",
                "section_title": "基础设施滞后",
                "claim": "低空经济中基础设施与飞行保障环节发展滞后。",
                "evidence_refs": ["EV-missing"],
                "used_fact_refs": ["EV-missing"],
                "evidence_basis": ["2023年低空飞行器制造和运营服务贡献近55%，基础设施和飞行保障潜力尚未充分显现。"],
                "supporting_facts": ["2023年低空飞行器制造和运营服务贡献近55%，基础设施和飞行保障潜力尚未充分显现。"],
                "claim_strength": "directional",
                "block_type": "risk_trigger",
                "public_render": True,
            }
        ],
        micro_layouts=[],
    )

    rendered = "\n".join(
        "\n".join(
            [
                str(section.get("composed_paragraph") or ""),
                str(section.get("reasoning") or ""),
                str(section.get("claim") or ""),
                "\n".join(str(item) for item in section.get("supporting_facts") or []),
            ]
        )
        for chapter in packages
        for section in chapter.get("sections", [])
    )
    assert "基础设施和飞行保障潜力" in rendered
    assert "全球低空经济市场规模" not in rendered
    assert "出货/部署" not in rendered
    assert "100家" not in rendered


def test_chapter_argument_filters_press_conference_question_snippets_from_fact_anchors():
    from rag_pipeline.agents.chapter_argument_agent import run_chapter_argument_agent

    packages = run_chapter_argument_agent(
        report_blueprint={
            "chapters": [
                {"chapter_id": "ch_demand", "chapter_title": "需求变化"}
            ]
        },
        chapter_evidence_packages=[
            {
                "chapter_id": "ch_demand",
                "chapter_title": "需求变化",
                "core_evidence": [
                    {
                        "evidence_id": "[3]",
                        "source_ref": "[3]",
                        "citation_ref": "[3]",
                        "public_fact_card": {
                            "distilled_fact": "（2026年3月16日） 第一财经记者： 2026年是“十五五”的开局之年，请问您如何评价经济运行表现？谢谢。",
                            "fact_type": "counter",
                            "block_affinity": ["risk_trigger"],
                        },
                    }
                ],
            }
        ],
        argument_units=[
            {
                "claim_id": "CL-demand",
                "chapter_id": "ch_demand",
                "section_id": "sec-demand",
                "section_title": "短期需求压力",
                "claim": "国内需求疲软与制造业景气度收缩可能抑制低空经济短期商业化需求释放。",
                "evidence_refs": ["[3]"],
                "used_fact_refs": ["[3]"],
                "evidence_basis": ["2025年10月制造业PMI为49.0%，处于收缩区间，制造业景气水平有所回落。"],
                "supporting_facts": ["2025年10月制造业PMI为49.0%，处于收缩区间，制造业景气水平有所回落。"],
                "claim_strength": "directional",
                "block_type": "risk_trigger",
                "public_render": True,
            }
        ],
        micro_layouts=[],
    )

    rendered = "\n".join(
        "\n".join(
            [
                str(section.get("composed_paragraph") or ""),
                str(section.get("reasoning") or ""),
                "\n".join(str(item) for item in section.get("supporting_facts") or []),
            ]
        )
        for chapter in packages
        for section in chapter.get("sections", [])
    )

    assert "制造业PMI为49.0%" in rendered
    assert "第一财经记者" not in rendered
    assert "请问您如何评价" not in rendered


def test_public_bridge_pack_does_not_emit_report_process_language():
    pack = build_public_bridge_pack(
        claim="\u4f1a\u8ba1\u5c97\u4f4d\u80fd\u529b\u6b63\u5728\u4ece\u57fa\u7840\u6838\u7b97\u8f6c\u5411\u6570\u636e\u5206\u6790\u548c\u4e1a\u52a1\u7406\u89e3\u3002",
        evidence_texts=[
            "\u4e13\u4e1a\u76ee\u5f55\u5c06\u8d22\u52a1\u5927\u6570\u636e\u5206\u6790\u3001\u667a\u80fd\u6210\u672c\u6838\u7b97\u548c\u9884\u7b97\u7ba1\u7406\u7eb3\u5165\u8bad\u7ec3\u5185\u5bb9\u3002"
        ],
        block_type="integrated_signal",
        claim_strength="directional",
    )
    text = " ".join(str(value or "") for value in pack.values())

    for forbidden in (
        "\u8bdd\u9898\u70ed\u5ea6",
        "\u62a5\u544a\u4e3b\u7ebf",
        "\u80cc\u666f\u7ebf\u7d22",
        "\u8bb2\u6e05\u695a",
        "\u62a5\u544a\u4e2d\u7684\u5206\u6790\u4ef7\u503c",
    ):
        assert forbidden not in text
    assert "\u4e3b\u4f53\u884c\u52a8" in text or "\u5c97\u4f4d\u4efb\u52a1" in text


def test_section_composer_long_expansion_does_not_emit_writing_process_language(monkeypatch):
    monkeypatch.setenv("REPORT_BLUEPRINT_SOURCE", "claim_first")
    monkeypatch.setenv("REPORT_COMPOSER_TARGET_SECTION_CHARS", "850")
    depth_pack = {
        "schema_version": "claim_depth_pack_v1",
        "claim_id": "CL-process-free-depth",
        "judgement": "\u4f1a\u8ba1\u5c97\u4f4d\u80fd\u529b\u6b63\u5728\u4ece\u57fa\u7840\u6838\u7b97\u8f6c\u5411\u6570\u636e\u5206\u6790\u548c\u4e1a\u52a1\u7406\u89e3\u3002",
        "evidence_chain": "\u516c\u5f00\u6750\u6599\u663e\u793a\uff0c\u8bfe\u7a0b\u8bad\u7ec3\u7eb3\u5165\u667a\u80fd\u6838\u7b97\u3001\u8d22\u52a1\u5927\u6570\u636e\u5206\u6790\u548c\u9884\u7b97\u7ba1\u7406\u3002",
        "mechanism": "\u8fd9\u4e9b\u80fd\u529b\u6a21\u5757\u4f1a\u6539\u53d8\u5b66\u6821\u8bfe\u7a0b\u3001\u5b9e\u8bad\u5de5\u5177\u548c\u4f01\u4e1a\u7b5b\u9009\u6807\u51c6\u4e4b\u95f4\u7684\u8fde\u63a5\u65b9\u5f0f\u3002",
        "used_fact_refs": ["EV-process-free-1"],
    }

    result = compose_section_paragraph(
        fact_cards=[
            EvidenceFactCard.from_legacy_dict(
                {
                    "evidence_id": "EV-process-free-1",
                    "chapter_id": "ch_depth",
                    "source_id": "SRC-process-free-1",
                    "source_level": "B",
                    "public_fact_card": {
                        "subject": "\u4f1a\u8ba1\u4e13\u4e1a\u8bfe\u7a0b\u8bad\u7ec3",
                        "distilled_fact": "\u516c\u5f00\u6750\u6599\u5c06\u8d22\u52a1\u5927\u6570\u636e\u5206\u6790\u3001\u667a\u80fd\u6210\u672c\u6838\u7b97\u548c\u9884\u7b97\u7ba1\u7406\u5217\u4e3a\u80fd\u529b\u6a21\u5757\u3002",
                        "fact_type": "policy",
                        "variable": "\u80fd\u529b\u6a21\u5757\u53d8\u5316",
                    },
                }
            )
        ],
        claim_unit=ClaimUnit.from_legacy_dict(
            {
                "claim_id": "CL-process-free-depth",
                "claim": "\u4f1a\u8ba1\u5c97\u4f4d\u80fd\u529b\u6b63\u5728\u4ece\u57fa\u7840\u6838\u7b97\u8f6c\u5411\u6570\u636e\u5206\u6790\u548c\u4e1a\u52a1\u7406\u89e3\u3002",
                "used_fact_refs": ["EV-process-free-1"],
                "claim_depth_pack": depth_pack,
                "claim_strength": "directional",
            }
        ),
        block_type="integrated_signal",
        chapter_question="\u4f1a\u8ba1\u5c31\u4e1a\u7684\u80fd\u529b\u8981\u6c42\u5982\u4f55\u53d8\u5316\uff1f",
    )
    paragraph = result["paragraph"]

    for forbidden in (
        "\u8fd9\u79cd\u5199\u6cd5",
        "\u6750\u6599\u7f57\u5217",
        "\u62a5\u544a\u4e3b\u7ebf",
        "\u8bdd\u9898\u70ed\u5ea6",
        "\u80cc\u666f\u7ebf\u7d22",
    ):
        assert forbidden not in paragraph
    assert "\u80fd\u529b\u6a21\u5757" in paragraph
