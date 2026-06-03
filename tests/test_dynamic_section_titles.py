from rag_pipeline.agents.block_schema import select_blocks_for_chapter
from rag_pipeline.agents.chapter_argument_agent import _public_section_title, run_chapter_argument_agent
from rag_pipeline.agents.claim_builder_agent import run_claim_builder_agent
from rag_pipeline.agents.markdown_renderer import render_section
from rag_pipeline.agents.micro_layout_agent import (
    _candidate_titles_for_section,
    _title_from_subject,
    _title_from_variable,
    generate_dynamic_section_title,
    run_micro_layout_agent,
)


def _case_item():
    card = {
        "analysis_variable": "流程部署",
        "fact_type": "case",
        "block_affinity": ["customer_painpoint_matrix", "case_comparison"],
        "fact": "企业客户将 AI Agent 接入客服流程。",
    }
    return {
        "evidence_id": "E1",
        "ref": "E1",
        "source_level": "C",
        "proof_role": "case",
        "fact": "企业客户将 AI Agent 接入客服流程。",
        "public_fact_card": card,
        "public_fact_quality": {"eligible_for_report": True, "public_fact_card": card},
    }


def _metric_item():
    card = {
        "analysis_variable": "市场规模",
        "fact_type": "metric",
        "block_affinity": ["metric_reconciliation"],
        "fact": "市场研究机构给出 AI Agent 市场规模预测。",
    }
    return {
        "evidence_id": "E2",
        "ref": "E2",
        "source_level": "B",
        "proof_role": "metric",
        "metric": "市场规模",
        "fact": "市场研究机构给出 AI Agent 市场规模预测。",
        "public_fact_card": card,
        "public_fact_quality": {"eligible_for_report": True, "public_fact_card": card},
    }


def _fact_card_item(ref: str, *, subject: str, variable: str, block_type: str, fact_type: str = "case", proof_role: str = "case"):
    fact = f"{subject}在{variable}上出现可追踪动作。"
    card = {
        "subject": subject,
        "action_or_signal": "出现可追踪动作",
        "variable": variable,
        "analysis_variable": variable,
        "fact_type": fact_type,
        "block_affinity": [block_type],
        "fact": fact,
        "distilled_fact": fact,
        "source_ref": ref,
        "claim_strength_hint": "directional",
    }
    return {
        "evidence_id": ref,
        "ref": ref,
        "source_ref": ref,
        "source_level": "C",
        "proof_role": proof_role,
        "fact": fact,
        "distilled_fact": fact,
        "public_fact_card": card,
        "public_fact_quality": {"eligible_for_report": True, "public_fact_card": card},
    }


def test_generate_dynamic_title_never_returns_generic_fact_label():
    package = {
        "chapter_title": "真实需求是否进入流程部署",
        "chapter_question": "AI Agent 是否从试用进入流程部署？",
    }
    payload = generate_dynamic_section_title(package, "evidence_matrix", [_case_item()])

    assert payload["title_source"] == "dynamic"
    assert payload["dynamic_section_title"]
    assert payload["dynamic_section_title"] != "事实依据"
    assert "证据" not in payload["dynamic_section_title"]
    assert "口径" not in payload["dynamic_section_title"]
    assert "流程" in payload["dynamic_section_title"] or "部署" in payload["dynamic_section_title"]


def test_unit_economics_without_commercial_evidence_is_not_selected():
    blocks = select_blocks_for_chapter(
        {"chapter_id": "ch_01", "module_keys": ["business_model"]},
        evidence_package={"chapter_id": "ch_01", "case_evidence": [_case_item()]},
    )

    block_types = {block["block_type"] for block in blocks}
    assert "unit_economics" not in block_types
    assert block_types.intersection({"case_comparison", "customer_painpoint_matrix", "signal_validation"})


def test_micro_layout_titles_are_unique_across_similar_case_blocks():
    packages = [
        {
            "chapter_id": "ch_a",
            "chapter_title": "客服场景是否开始进入流程部署",
            "chapter_question": "客服场景是否开始进入流程部署？",
            "case_evidence": [
                _fact_card_item("E1", subject="Salesforce Agentforce", variable="客服流程部署", block_type="case_comparison")
            ],
        },
        {
            "chapter_id": "ch_b",
            "chapter_title": "IT 运维是否开始进入流程部署",
            "chapter_question": "IT 运维是否开始进入流程部署？",
            "case_evidence": [
                _fact_card_item("E2", subject="ServiceNow AI Agent", variable="IT 运维部署", block_type="case_comparison")
            ],
        },
    ]
    layouts = run_micro_layout_agent(
        report_blueprint={
            "chapters": [
                {"chapter_id": "ch_a", "chapter_title": "客服场景是否开始进入流程部署"},
                {"chapter_id": "ch_b", "chapter_title": "IT 运维是否开始进入流程部署"},
            ]
        },
        chapter_evidence_packages=packages,
    )

    titles = [
        section.get("section_title")
        for layout in layouts
        for section in layout.get("sections", [])
        if section.get("section_title")
    ]
    assert titles
    assert len(titles) == len(set(titles))
    assert "代表性案例对比" not in titles
    assert "反向信号与失效条件" not in titles


def test_dynamic_title_filters_internal_role_variables():
    item = _fact_card_item(
        "E3",
        subject="NIST AI RMF",
        variable="official_me",
        block_type="risk_trigger",
        fact_type="counter",
        proof_role="counter",
    )
    payload = generate_dynamic_section_title(
        {"chapter_title": "治理边界会怎样影响企业级 Agent 落地"},
        "risk_trigger",
        [item],
    )

    title = payload.get("dynamic_section_title") or ""
    assert title
    assert "official_me" not in title
    assert "source_check" not in title
    assert "counter" not in title
    assert "_" not in title


def test_dynamic_title_does_not_use_publisher_as_subject():
    item = _fact_card_item(
        "E5",
        subject="",
        variable="\u5ba2\u6237\u843d\u5730",
        block_type="case_comparison",
        fact_type="case",
        proof_role="case",
    )
    item["publisher"] = "\u7231\u96c6\u5fae-ijiwei"
    item["source_title"] = "\u7231\u96c6\u5fae-ijiwei"
    item["public_fact_card"]["subject"] = ""
    item["public_fact_card"]["publisher"] = "\u7231\u96c6\u5fae-ijiwei"

    payload = generate_dynamic_section_title(
        {"chapter_title": "\u5ba2\u6237\u6848\u4f8b\u662f\u5426\u80fd\u8bc1\u660e\u843d\u5730"},
        "case_comparison",
        [item],
    )

    title = payload.get("dynamic_section_title") or ""
    assert title
    assert "ijiwei" not in title.lower()
    assert "\u7231\u96c6\u5fae" not in title


def test_unknown_metric_variable_uses_block_lens_not_mechanical_where_title():
    item = _fact_card_item(
        "E6",
        subject="",
        variable="\u653f\u7b56\u76ee\u6807",
        block_type="case_comparison",
        fact_type="case",
        proof_role="case",
    )
    item["metric"] = "\u653f\u7b56\u76ee\u6807"
    item["public_fact_card"]["analysis_variable"] = "\u653f\u7b56\u76ee\u6807"

    payload = generate_dynamic_section_title(
        {"chapter_title": "\u6280\u672f\u7ea6\u675f\u5982\u4f55\u5f71\u54cd\u843d\u5730"},
        "case_comparison",
        [item],
    )

    title = payload.get("dynamic_section_title") or ""
    assert title
    assert "\u653f\u7b56\u76ee\u6807\u5728\u54ea\u91cc\u53d1\u751f" not in title
    assert "\u5728\u54ea\u91cc\u53d1\u751f" not in title


def test_title_template_guard_avoids_tail_keyword_collision():
    assert "\u6536\u5165\u80fd\u5426\u8f6c\u6210\u6536\u5165" not in _title_from_subject(
        "\u6536\u5165",
        "\u6536\u5165",
        "unit_economics",
    )
    assert "\u6536\u5165\u80fd\u5426\u8f6c\u6210\u6536\u5165" not in _title_from_variable(
        "\u6536\u5165",
        "unit_economics",
    )
    assert "\u5e02\u573a\u89c4\u6a21\u4f1a\u600e\u6837\u63a8\u7ffb\u7ed3\u8bba" not in _title_from_variable(
        "\u5e02\u573a\u89c4\u6a21",
        "risk_trigger",
    )
    assert "\u5e02\u573a\u89c4\u6a21\u4f1a\u600e\u6837\u6539\u53d8\u5224\u65ad" not in _title_from_variable(
        "\u5e02\u573a\u89c4\u6a21",
        "risk_trigger",
    )
    assert "\u51fa\u8d27/\u90e8\u7f72\u80fd\u5426\u8f6c\u6210\u6536\u5165" not in _title_from_variable(
        "\u51fa\u8d27/\u90e8\u7f72",
        "unit_economics",
    )


def test_long_question_like_subject_does_not_become_section_title():
    title = _title_from_subject(
        "\u9700\u6c42\u53d8\u5316\u6765\u81ea\u54ea\u91cc\uff0c\u80fd\u5426\u6301\u7eed\u5151\u73b0\uff1f",
        "\u9700\u6c42",
        "case_comparison",
    )

    assert "\u9700\u6c42\u53d8\u5316\u6765\u81ea\u54ea\u91cc" not in title
    assert len(title) <= 24


def test_question_like_chapter_hint_does_not_become_progress_title():
    package = {
        "chapter_id": "ch_05",
        "chapter_title": "\u9700\u6c42\u53d8\u5316\u6765\u81ea\u54ea\u91cc\uff0c\u80fd\u5426\u6301\u7eed\u5151\u73b0\uff1f",
        "chapter_question": "\u9700\u6c42\u53d8\u5316\u6765\u81ea\u54ea\u91cc\uff0c\u80fd\u5426\u6301\u7eed\u5151\u73b0\uff1f",
        "supporting_evidence": [
            _fact_card_item(
                "E7",
                subject="",
                variable="\u9700\u6c42\u53d8\u5316\u6765\u81ea\u54ea\u91cc\uff0c\u80fd\u5426\u6301\u7eed\u5151\u73b0\uff1f",
                block_type="unit_economics",
                fact_type="case",
                proof_role="case",
            )
        ],
    }

    titles = _candidate_titles_for_section(package, {"block_type": "unit_economics", "section_id": "s1"})

    assert titles
    assert not any("\u7684\u8fdb\u5c55" in title for title in titles)
    assert not any("\uff1f" in title for title in titles)


def test_directional_claim_builder_does_not_emit_weak_template_sentences():
    item = _fact_card_item(
        "E4",
        subject="Salesforce Agentforce",
        variable="客服流程部署",
        block_type="case_comparison",
        fact_type="case",
        proof_role="case",
    )
    units = run_claim_builder_agent(
        chapter_evidence_packages=[
            {
                "chapter_id": "ch_01",
                "chapter_title": "真实需求是否进入流程部署",
                "case_evidence": [item],
                "directional_evidence": [item],
                "chapter_analysis": {"fact_cards": [item["public_fact_card"]], "claim_strength": "directional"},
            }
        ],
        micro_layouts=[
            {
                "chapter_id": "ch_01",
                "sections": [
                    {
                        "section_id": "ch_01_case",
                        "block_type": "case_comparison",
                        "dynamic_section_title": "客服流程部署到哪一步",
                        "required_evidence_refs": ["E4"],
                    }
                ],
            }
        ],
        structured_analysis={},
    )

    text = "\n".join(str(unit.get(key) or "") for unit in units for key in ("claim", "reasoning", "counter_evidence"))
    for phrase in ["只能形成初步信号", "暂不宜外推", "低强度判断", "更多独立来源复核", "更多客户样本或反向案例"]:
        assert phrase not in text


def test_micro_layout_passes_dynamic_title_to_chapter_argument():
    package = {
        "chapter_id": "ch_01",
        "chapter_title": "真实需求是否进入流程部署",
        "chapter_question": "AI Agent 是否从试用进入流程部署？",
        "case_evidence": [_case_item()],
    }
    layouts = run_micro_layout_agent(
        report_blueprint={"chapters": [{"chapter_id": "ch_01", "chapter_title": "真实需求是否进入流程部署"}]},
        chapter_evidence_packages=[package],
    )

    titles = [section["section_title"] for section in layouts[0]["sections"]]
    assert titles
    assert "事实依据" not in titles
    assert "商业化证据" not in titles
    assert all(section.get("dynamic_section_title") for section in layouts[0]["sections"])


def test_public_section_title_preserves_dynamic_title():
    layout_section = {
        "block_type": "metric_reconciliation",
        "section_title": "事实依据",
        "dynamic_section_title": "市场空间到底有多大",
    }

    assert (
        _public_section_title({}, {"chapter_title": "市场规模是否可比"}, index=1, layout_section=layout_section)
        == "市场空间到底有多大"
    )


def test_metric_reconciliation_can_be_selected_with_metric_evidence():
    blocks = select_blocks_for_chapter(
        {"chapter_id": "ch_01", "module_keys": ["market_size"]},
        evidence_package={"chapter_id": "ch_01", "metric_evidence": [_metric_item()]},
    )

    assert any(block["block_type"] == "metric_reconciliation" for block in blocks)


def test_chapter_argument_exposes_natural_section_plan_not_generic_title():
    dynamic_title = "客户流程是否开始部署"
    fact = "Salesforce disclosed Agentforce customer-service workflow deployments in 2025."
    evidence = {
        "ref": "E1",
        "evidence_id": "E1",
        "source_ref": "S1",
        "source_level": "C",
        "proof_role": "case",
        "fact": fact,
        "public_fact_card": {
            "subject": "Salesforce Agentforce",
            "distilled_fact": fact,
            "block_affinity": ["case_comparison"],
            "analysis_variable": "客服流程部署",
        },
        "public_fact_quality": {"eligible_for_report": True},
    }
    packages = run_chapter_argument_agent(
        report_blueprint={"chapters": [{"chapter_id": "ch_01", "chapter_title": "真实需求是否进入流程部署"}]},
        micro_layouts=[
            {
                "chapter_id": "ch_01",
                "sections": [
                    {
                        "section_id": "ch_01_case",
                        "section_title": "事实依据",
                        "dynamic_section_title": dynamic_title,
                        "block_type": "case_comparison",
                        "required_evidence_refs": ["E1"],
                    }
                ],
            }
        ],
        argument_units=[
            {
                "chapter_id": "ch_01",
                "section_id": "ch_01_case",
                "block_type": "case_comparison",
                "claim": "企业级智能体已经开始进入客服流程部署。",
                "evidence_basis": [fact],
                "supporting_facts": [fact],
                "mechanism": "客服流程具备高频、可度量、可回退的特征，因此更容易成为企业智能体试点入口。",
                "used_fact_refs": ["E1"],
                "evidence_refs": ["E1"],
                "fact_card_to_block_match": True,
                "claim_strength": "directional",
            }
        ],
        chapter_evidence_packages=[{"chapter_id": "ch_01", "case_evidence": [evidence], "directional_evidence": [evidence]}],
    )

    section = packages[0]["sections"][0]
    assert section["section_title"] == dynamic_title
    assert section["section_plan"]["public_title"] == dynamic_title
    assert section["section_plan"]["used_fact_refs"] == ["E1"]
    assert "证据" not in section["section_plan"]["public_title"]
    assert "口径" not in section["section_plan"]["public_title"]


def test_renderer_uses_section_plan_title_and_not_generic_section_title():
    rendered = "\n".join(
        render_section(
            {
                "section_title": "事实依据",
                "section_plan": {
                    "public_title": "客户流程是否开始部署",
                    "used_fact_refs": ["E1"],
                    "paragraph_plan": "判断句 → 关键事实 → 机制解释",
                },
                "render_blocks": [
                    {
                        "type": "paragraph",
                        "text": "Salesforce disclosed Agentforce customer-service workflow deployments in 2025.",
                    }
                ],
                "evidence_refs": ["E1"],
                "evidence_backed": True,
            }
        )
    )

    assert "### 客户流程是否开始部署" in rendered
    assert "### 事实依据" not in rendered


def test_public_section_title_does_not_fallback_to_section_observation_label():
    title = _public_section_title(
        {},
        {"chapter_title": "技术成熟度是否支持生产部署"},
        index=1,
        layout_section={"block_type": "technology_maturity", "section_title": "本节技术观察"},
    )

    assert title != "本节技术观察"
    assert "本节" not in title
    assert "观察" not in title
