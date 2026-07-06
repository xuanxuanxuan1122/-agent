from rag_pipeline.agents.chapter_recomposer_agent import recompose_chapters_from_claims
from rag_pipeline.agents.markdown_renderer import _public_section_expansion_sentences
from rag_pipeline.agents.public_narrative_bridge import build_public_bridge_pack
from rag_pipeline.agents.report_contracts import ClaimUnit, EvidenceFactCard
from rag_pipeline.agents.section_composer import _join_public_sentences, compose_section_paragraph


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
    assert "任务边界" in paragraph or "能力组合" in paragraph or "评价标准" in paragraph


def test_section_composer_does_not_render_narrative_role_bridge(monkeypatch):
    monkeypatch.setenv("REPORT_BLUEPRINT_SOURCE", "claim_first")
    monkeypatch.setenv("REPORT_COMPOSER_TARGET_SECTION_CHARS", "650")

    card = EvidenceFactCard(
        evidence_id="EV-JOB",
        chapter_id="ch_01",
        subject="会计就业市场",
        variable="能力需求",
        distilled_fact="春招市场显示AIGC设计师、AI Agent开发工程师等岗位走俏，部分企业也披露基础岗位编制调整压力。",
        fact_type="directional",
        block_affinity=["integrated_signal"],
        source_ref="[1]",
        source_level="C",
    )
    claim = ClaimUnit(
        claim_id="CL-JOB",
        chapter_id="ch_01",
        claim="会计岗位的核心能力要求正由传统账务处理向大模型辅助的智能财务分析迁移",
        evidence_refs=["EV-JOB"],
        evidence_basis=[
            "春招市场显示AIGC设计师、AI Agent开发工程师等岗位走俏。",
        ],
        reasoning_chain="岗位变化会把课程、工具和组织流程放到同一条能力链上重新评估。",
        claim_strength="directional",
        raw={
            "narrative_role": "mechanism",
            "narrative_supporting_claims": [
                "市场需求端的新兴AI岗位热度与传统企业因AI应用预期缩减基础人力的信号相互印证。",
            ],
        },
    )

    result = compose_section_paragraph(
        fact_cards=[card],
        claim_unit=claim,
        block_type="integrated_signal",
        chapter_question="会计就业能力如何变化？",
    )

    paragraph = result["paragraph"]
    assert "这些材料放在一起看" not in paragraph
    assert "事实之间的因果关系和执行条件" not in paragraph
    assert "春招市场显示AIGC" in paragraph
    assert "岗位走俏" in paragraph


def test_section_composer_adds_sentence_boundary_between_joined_parts():
    paragraph = _join_public_sentences(
        [
            "会计岗位的核心能力要求正由传统账务处理向智能财务分析迁移",
            "春招市场显示AIGC设计师、AI Agent开发工程师等岗位走俏。",
        ]
    )

    assert "迁移 春招" not in paragraph
    assert "迁移。春招" in paragraph


def test_section_composer_uses_claim_topic_when_fact_subject_is_generic(monkeypatch):
    monkeypatch.setenv("REPORT_BLUEPRINT_SOURCE", "claim_first")
    monkeypatch.setenv("REPORT_COMPOSER_EXPAND_TO_TARGET", "true")
    monkeypatch.setenv("REPORT_COMPOSER_TARGET_SECTION_CHARS", "650")

    card = EvidenceFactCard(
        evidence_id="EV-GENERIC-SUBJECT",
        chapter_id="ch_01",
        subject="相关主体",
        variable="需求变化",
        distilled_fact="春招市场显示AIGC设计师、AI Agent开发工程师等岗位走俏，部分企业披露基础岗位编制调整压力。",
        fact_type="directional",
        block_affinity=["integrated_signal"],
        source_ref="[1]",
        source_level="C",
    )
    claim = ClaimUnit(
        claim_id="CL-GENERIC-SUBJECT",
        chapter_id="ch_01",
        claim="会计就业市场呈现替代与创造并存的结构性分化。",
        evidence_refs=["EV-GENERIC-SUBJECT"],
        evidence_basis=["春招市场显示AIGC设计师、AI Agent开发工程师等岗位走俏。"],
        claim_strength="directional",
    )

    result = compose_section_paragraph(
        fact_cards=[card],
        claim_unit=claim,
        block_type="integrated_signal",
        chapter_question="会计就业结构如何变化？",
    )

    paragraph = result["paragraph"]
    assert "相关场景" not in paragraph
    assert "这一场景" not in paragraph
    assert "会计就业市场" in paragraph


def test_section_composer_does_not_turn_full_claim_into_subject_template(monkeypatch):
    monkeypatch.setenv("REPORT_BLUEPRINT_SOURCE", "claim_first")
    monkeypatch.setenv("REPORT_COMPOSER_TARGET_SECTION_CHARS", "650")

    card = EvidenceFactCard(
        evidence_id="EV-LONG-SUBJECT",
        chapter_id="ch_01",
        subject="会计就业市场呈现替代与创造并存的结构性分化",
        variable="需求变化",
        distilled_fact="春招材料显示AIGC设计师、AI Agent开发工程师等岗位走俏，部分企业也披露基础岗位编制调整压力。",
        fact_type="directional",
        block_affinity=["integrated_signal"],
        source_ref="[1]",
        source_level="C",
    )
    claim = ClaimUnit(
        claim_id="CL-LONG-SUBJECT",
        chapter_id="ch_01",
        claim="会计就业市场呈现替代与创造并存的结构性分化。",
        evidence_refs=["EV-LONG-SUBJECT"],
        evidence_basis=[
            "春招材料显示AIGC设计师、AI Agent开发工程师等岗位走俏，部分企业也披露基础岗位编制调整压力。",
        ],
        claim_strength="directional",
    )

    result = compose_section_paragraph(
        fact_cards=[card],
        claim_unit=claim,
        block_type="integrated_signal",
        chapter_question="AI如何改变会计就业结构？",
    )

    paragraph = result["paragraph"]
    assert "结合现有公开信息" not in paragraph
    assert "的业务安排、资源投入或服务组合" not in paragraph
    assert "影响会计就业市场呈现替代与创造并存的结构性分化的业务安排" not in paragraph
    assert "任务边界" in paragraph or "能力组合" in paragraph or "评价标准" in paragraph


def test_section_composer_removes_policy_boilerplate_and_metric_labels(monkeypatch):
    monkeypatch.setenv("REPORT_BLUEPRINT_SOURCE", "claim_first")
    monkeypatch.setenv("REPORT_COMPOSER_TARGET_SECTION_CHARS", "620")

    noisy_policy = EvidenceFactCard(
        evidence_id="EV-NOISY-POLICY",
        chapter_id="ch_01",
        subject="广东省人工智能应用政策",
        variable="政策影响",
        distilled_fact="出货/部署: 11 《广东省加快推进人工智能全域全时全行业高水平应用行动方案》已经省人民政府同意，现印发给你们，请认真贯彻执行。附件1 会计改革与发展“十四五”规划纲要（征求意见稿）“十四五”时期是关键时期。",
        fact_type="policy",
        block_affinity=["integrated_signal"],
        source_ref="[1]",
        source_level="B",
    )
    clean_policy = EvidenceFactCard(
        evidence_id="EV-CLEAN-POLICY",
        chapter_id="ch_01",
        subject="会计人才培养政策",
        variable="政策影响",
        distilled_fact="政策文件与行业协会活动共同推动会计行业数字化转型，并影响高校专业设置与企业招聘能力标准。",
        fact_type="policy",
        block_affinity=["integrated_signal"],
        source_ref="[2]",
        source_level="B",
    )
    claim = ClaimUnit(
        claim_id="CL-POLICY",
        chapter_id="ch_01",
        claim="政策与行业组织正在推动会计人才能力标准调整。",
        evidence_refs=["EV-NOISY-POLICY", "EV-CLEAN-POLICY"],
        evidence_basis=[
            "政策文件与行业协会活动共同推动会计行业数字化转型，并影响高校专业设置与企业招聘能力标准。",
        ],
        claim_strength="directional",
    )

    result = compose_section_paragraph(
        fact_cards=[noisy_policy, clean_policy],
        claim_unit=claim,
        block_type="integrated_signal",
        chapter_question="会计教育培养如何调整？",
    )

    paragraph = result["paragraph"]
    assert "出货/部署" not in paragraph
    assert "相关进展:" not in paragraph
    assert "现印发给你们" not in paragraph
    assert "请认真贯彻执行" not in paragraph
    assert "附件1" not in paragraph
    assert "征求意见稿" not in paragraph
    assert "会计行业数字化转型" in paragraph


def test_section_composer_does_not_use_long_case_claim_as_subject(monkeypatch):
    monkeypatch.setenv("REPORT_BLUEPRINT_SOURCE", "claim_first")
    monkeypatch.setenv("REPORT_COMPOSER_TARGET_SECTION_CHARS", "620")

    card = EvidenceFactCard(
        evidence_id="EV-CASE-LONG",
        chapter_id="ch_02",
        subject="会计教育与继续教育渠道正加速整合低代码平台与AI数据工具",
        variable="课程结构变化",
        distilled_fact="高校与培训机构推出AI财务创新课程，并把低代码平台、数据分析工具纳入实践训练。",
        fact_type="case",
        block_affinity=["case_comparison"],
        source_ref="[1]",
        source_level="C",
    )
    claim = ClaimUnit(
        claim_id="CL-CASE-LONG",
        chapter_id="ch_02",
        claim="会计教育与继续教育渠道正加速整合低代码平台与AI数据工具。",
        evidence_refs=["EV-CASE-LONG"],
        evidence_basis=["高校与培训机构推出AI财务创新课程，并把低代码平台、数据分析工具纳入实践训练。"],
        claim_strength="directional",
    )

    result = compose_section_paragraph(
        fact_cards=[card],
        claim_unit=claim,
        block_type="case_comparison",
        chapter_question="会计教育培养如何调整？",
    )

    paragraph = result["paragraph"]
    assert "会计教育与继续教育渠道正加速整合低代码平台与AI的具体动作" not in paragraph
    assert "会计教育与继续教育渠道正加速整合低代码平台与AI这一场景" not in paragraph
    assert "行动样本" in paragraph or "实践训练" in paragraph


def test_section_composer_technology_lens_does_not_force_education_workflow_template(monkeypatch):
    monkeypatch.setenv("REPORT_BLUEPRINT_SOURCE", "claim_first")
    monkeypatch.setenv("REPORT_COMPOSER_TARGET_SECTION_CHARS", "700")

    card = EvidenceFactCard(
        evidence_id="EV-TECH-CAPEX",
        chapter_id="ch_tech",
        subject="AI基础设施资本开支",
        variable="技术成熟度",
        distilled_fact="大型科技企业持续增加AI基础设施资本开支，云端算力、模型训练和推理服务投入同步扩大。",
        fact_type="technology",
        block_affinity=["technology_maturity"],
        source_ref="[1]",
        source_level="B",
    )
    claim = ClaimUnit(
        claim_id="CL-TECH-CAPEX",
        chapter_id="ch_tech",
        claim="AI基础设施资本开支正在改变会计岗位的工具环境和自动化边界。",
        evidence_refs=["EV-TECH-CAPEX"],
        evidence_basis=["大型科技企业持续增加AI基础设施资本开支，云端算力、模型训练和推理服务投入同步扩大。"],
        claim_strength="directional",
    )

    result = compose_section_paragraph(
        fact_cards=[card],
        claim_unit=claim,
        block_type="technology_maturity",
        chapter_question="AI基础设施投入如何影响会计岗位？",
    )

    paragraph = result["paragraph"]
    assert "课程设置" not in paragraph
    assert "岗位分工或服务交付的先后顺序" not in paragraph
    assert "学校、企业和培训机构" not in paragraph
    assert "算力" in paragraph or "系统" in paragraph or "集成" in paragraph or "自动化" in paragraph


def test_section_composer_risk_lens_does_not_force_education_workflow_template(monkeypatch):
    monkeypatch.setenv("REPORT_BLUEPRINT_SOURCE", "claim_first")
    monkeypatch.setenv("REPORT_COMPOSER_TARGET_SECTION_CHARS", "700")

    card = EvidenceFactCard(
        evidence_id="EV-RISK-AUDIT",
        chapter_id="ch_risk",
        subject="AI审计应用边界",
        variable="责任边界",
        distilled_fact="行业观点认为AI可以提升审计效率，但关键判断、责任归属和异常事项处理仍需要人工审慎复核。",
        fact_type="risk",
        block_affinity=["risk_trigger"],
        source_ref="[1]",
        source_level="C",
    )
    claim = ClaimUnit(
        claim_id="CL-RISK-AUDIT",
        chapter_id="ch_risk",
        claim="AI不会直接消除会计与审计岗位，责任边界会成为能力升级的关键约束。",
        evidence_refs=["EV-RISK-AUDIT"],
        evidence_basis=["行业观点认为AI可以提升审计效率，但关键判断、责任归属和异常事项处理仍需要人工审慎复核。"],
        claim_strength="directional",
    )

    result = compose_section_paragraph(
        fact_cards=[card],
        claim_unit=claim,
        block_type="risk_trigger",
        chapter_question="AI替代会计岗位的风险边界在哪里？",
    )

    paragraph = result["paragraph"]
    assert "课程设置" not in paragraph
    assert "学校、企业和培训机构" not in paragraph
    assert "责任" in paragraph or "复核" in paragraph or "风险" in paragraph


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
