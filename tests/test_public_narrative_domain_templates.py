from rag_pipeline.agents.chapter_recomposer_agent import recompose_chapters_from_claims
from rag_pipeline.agents.markdown_renderer import _public_section_expansion_sentences
from rag_pipeline.agents.public_narrative_bridge import build_public_bridge_pack
from rag_pipeline.agents.report_contracts import ClaimUnit, EvidenceFactCard
from rag_pipeline.agents.section_composer import compose_section_paragraph


FORBIDDEN_HARD_TEMPLATE_TERMS = (
    "订单",
    "客户付费",
    "商业化节奏",
    "示范项目",
    "规模化部署",
    "出货",
    "部署",
    "需求信号",
    "使用反馈",
    "发展节奏",
    "落地情况",
)


def _assert_no_hard_template_terms(text: str) -> None:
    for term in FORBIDDEN_HARD_TEMPLATE_TERMS:
        assert term not in text


def test_public_bridge_does_not_force_commercialization_template_for_employment_topic():
    pack = build_public_bridge_pack(
        claim="会计学专业就业需要结合岗位结构、课程体系和AI工具影响分析。",
        evidence_texts=[
            "教育部专业目录提到财务会计类培养面向会计核算、财务大数据分析和智能成本核算。",
            "会计行业人才发展规划强调高素质专业化会计人才队伍建设。",
        ],
        block_type="integrated_signal",
        claim_strength="directional",
    )

    text = " ".join(str(value or "") for value in pack.values())

    _assert_no_hard_template_terms(text)
    assert "岗位" in text or "课程" in text or "AI" in text
    assert "教育部专业目录" in text
    assert "资源投入" not in text
    assert "流程调整" not in text
    assert "外部约束" not in text


def test_section_composer_expands_employment_topic_without_hard_industry_template(monkeypatch):
    monkeypatch.setenv("REPORT_BLUEPRINT_SOURCE", "claim_first")
    monkeypatch.setenv("REPORT_COMPOSER_TARGET_SECTION_CHARS", "650")

    card = EvidenceFactCard(
        evidence_id="EV-EDU",
        chapter_id="ch_01",
        subject="会计学专业",
        variable="就业能力",
        action_or_signal="培养面向",
        distilled_fact="教育部专业目录提到财务会计类培养面向会计核算、财务大数据分析和智能成本核算。",
        fact_type="directional",
        block_affinity=["integrated_signal"],
        source_ref="[1]",
        source_level="A",
    )
    claim = ClaimUnit(
        claim_id="CL-EDU",
        chapter_id="ch_01",
        claim="会计学专业就业需要结合岗位结构、课程体系和AI工具影响分析。",
        evidence_refs=["EV-EDU"],
        evidence_basis=[
            "教育部专业目录提到财务会计类培养面向会计核算、财务大数据分析和智能成本核算。",
        ],
        reasoning_chain="这类证据更适合分析课程能力与岗位能力之间的匹配关系。",
        limitation_boundary="公开材料没有直接给出薪酬、招聘数量或AI替代比例。",
        claim_strength="directional",
    )

    result = compose_section_paragraph(
        fact_cards=[card],
        claim_unit=claim,
        block_type="integrated_signal",
        chapter_question="会计学专业就业趋势和AI影响应该怎么看？",
    )

    paragraph = result["paragraph"]
    assert result["composition_status"] in {"composed", "composed_directional"}
    _assert_no_hard_template_terms(paragraph)
    assert "岗位" in paragraph or "课程" in paragraph or "AI" in paragraph
    assert "流程调整" in paragraph or "条件" in paragraph or "时间窗口" in paragraph


def test_renderer_template_expansion_is_domain_neutral_for_employment_topic():
    sentences = _public_section_expansion_sentences(
        {
            "section_title": "AI 对会计岗位结构的影响",
            "block_type": "integrated_signal",
            "evidence_backed": True,
            "citation_refs": ["[1]"],
        }
    )

    text = " ".join(sentences)

    _assert_no_hard_template_terms(text)
    assert "岗位" in text or "会计" in text or "具体业务" in text
    assert "影响路径" not in text
    assert "章节判断" not in text


def test_renderer_template_expansion_fallback_avoids_methodology_labels():
    sentences = _public_section_expansion_sentences(
        {
            "section_title": "会计课程与岗位要求变化",
            "block_type": "unknown_block",
            "evidence_backed": True,
            "citation_refs": ["[1]"],
        }
    )

    text = " ".join(sentences)

    assert "影响路径" not in text
    assert "章节判断" not in text
    assert "主体、场景、时间窗口" not in text


def test_recomposer_uses_chinese_limitations_title_not_english_fallback():
    result = recompose_chapters_from_claims(
        plan_blueprint={"chapters": [{"chapter_id": "ch_01", "chapter_title": "会计就业趋势"}]},
        structured_analysis={
            "claim_units": [
                {
                    "claim_id": "CL-LIMIT",
                    "claim": "公开材料暂未充分覆盖会计岗位结构和AI替代比例。",
                    "chapter_id": "ch_01",
                    "cluster_key": "limitations",
                    "claim_strength": "weak",
                    "fact_ids": ["EV-1"],
                    "source_ids": ["S1"],
                    "can_anchor_section": True,
                }
            ]
        },
    )

    titles = " ".join(chapter.get("chapter_title", "") for chapter in result.get("final_chapters", []))

    assert "Limitations" not in titles
    assert "Issues to Verify" not in titles
    assert "限制" in titles or "待验证" in titles or "边界" in titles
