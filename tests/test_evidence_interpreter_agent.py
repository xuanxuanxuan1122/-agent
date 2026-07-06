from rag_pipeline.agents.evidence_interpreter_agent import build_evidence_interpretation_units


def _fact(ref, *, text, requirement_id="REQ-1", topic_fit="direct", proof_role="case"):
    return {
        "evidence_id": ref,
        "requirement_id": requirement_id,
        "chapter_id": "ch_01",
        "source_id": f"SRC-{ref}",
        "distilled_fact": text,
        "topic_fit": topic_fit,
        "proof_role": proof_role,
        "source_title": f"Source {ref}",
        "source_url": f"https://example.com/{ref}",
    }


def test_interpreter_dedupes_repeated_evidence_id_into_one_primary_group():
    result = build_evidence_interpretation_units(
        chapter_id="ch_01",
        chapter_question="AI如何改变会计岗位能力要求？",
        fact_cards=[
            _fact("EV-1", text="代账场景中AI工具使单人服务户数提升10倍以上。", proof_role="metric"),
            _fact("EV-1", text="代账场景中AI工具使单人服务户数提升10倍以上。", proof_role="metric"),
            _fact("EV-2", text="审计数字化调研显示83%的受访者已在审计中应用数字化技术。", proof_role="metric"),
            _fact("EV-3", text="高校会计专业开始增加AI审计与数据分析课程。", proof_role="case"),
        ],
    )

    units = result["interpretation_units"]
    primary_refs = [ref for unit in units for ref in unit["fact_ids"]]
    assert primary_refs.count("EV-1") == 1
    assert result["diagnostics"]["unique_fact_count"] == 3
    assert result["diagnostics"]["duplicate_fact_count"] == 1


def test_interpreter_builds_required_public_depth_fields_from_related_facts():
    result = build_evidence_interpretation_units(
        chapter_id="ch_01",
        chapter_question="AI如何改变会计就业和培养体系？",
        fact_cards=[
            _fact("EV-1", text="代账场景中AI工具使单人服务户数提升10倍以上。", proof_role="metric"),
            _fact("EV-2", text="审计数字化调研显示83%的受访者已在审计中应用数字化技术。", proof_role="metric"),
            _fact("EV-3", text="高校会计专业开始增加AI审计与数据分析课程。", proof_role="case"),
            _fact("EV-4", text="财政部规划强调会计信息化和高素质专业化人才建设。", proof_role="policy"),
        ],
    )

    unit = result["interpretation_units"][0]
    assert unit["schema_version"] == "evidence_interpretation_unit_v1"
    assert unit["chapter_id"] == "ch_01"
    assert unit["fact_ids"]
    assert unit["core_judgment"]
    assert unit["what_evidence_reflects"]
    assert unit["why_it_matters"]
    assert unit["mechanism_chain"]
    assert unit["employment_implication"]
    assert unit["education_implication"]
    assert unit["industry_implication"]
    assert unit["public_text_allowed"] is True
    assert unit["diagnostic_only"] is False


def test_interpreter_consolidates_small_same_cluster_groups_across_requirements():
    result = build_evidence_interpretation_units(
        chapter_id="ch_metric",
        chapter_question="会计学专业在AI时代的岗位能力变化",
        fact_cards=[
            _fact(
                f"EV-{index}",
                text=f"会计AI应用相关公开数据样本 {index} 显示企业正在增加数字化投入。",
                requirement_id=f"H{index}",
                proof_role="metric",
            )
            for index in range(1, 7)
        ],
        max_units=6,
    )

    units = result["interpretation_units"]
    assert len(units) == 1
    assert len(units[0]["fact_ids"]) == 6
    assert units[0]["single_fact_interpretation"] is False


def test_interpreter_does_not_use_background_fact_as_standalone_core_judgment():
    result = build_evidence_interpretation_units(
        chapter_id="ch_01",
        chapter_question="AI如何改变会计就业？",
        fact_cards=[
            _fact("EV-BG", text="宏观服务业规划强调创新发展。", topic_fit="background", proof_role="policy"),
        ],
    )

    unit = result["interpretation_units"][0]
    assert unit["single_fact_interpretation"] is True
    assert unit["claim_strength"] == "weak"
    assert unit["core_judgment"] == ""
    assert "宏观服务业规划" in unit["counter_reading"]


def test_interpreter_blocks_dirty_login_error_and_internal_diagnostic_text():
    result = build_evidence_interpretation_units(
        chapter_id="ch_01",
        chapter_question="AI如何改变会计就业？",
        fact_cards=[
            _fact("EV-LOGIN", text="Login Sign in Skip to content Contact us", proof_role="case"),
            _fact("EV-ERR", text="HTTP 404 error page not found", proof_role="case"),
            _fact("EV-DIAG", text="diagnostic_only source_check QA fatal score_gap", proof_role="case"),
            _fact("EV-GOOD", text="企业财务团队开始用AI工具缩短月结和分析流程。", proof_role="case"),
        ],
    )

    unit_refs = [ref for unit in result["interpretation_units"] for ref in unit["fact_ids"]]
    assert unit_refs == ["EV-GOOD"]
    assert result["diagnostics"]["blocked_dirty_fact_count"] == 3


def test_interpreter_uses_query_domain_instead_of_accounting_template_for_industry_topic():
    result = build_evidence_interpretation_units(
        chapter_id="ch_ops",
        chapter_question="县域新能源商用车充换电网络的资产利用率与运营商盈利模型如何形成？",
        fact_cards=[
            _fact("EV-1", text="财政部等部门开展县域充换电设施补短板试点，要求补齐农村地区公共充换电基础设施短板。", proof_role="policy"),
            _fact("EV-2", text="地方规划将重卡物流、园区运输和城乡配送列为新能源商用车应用场景。", proof_role="case"),
            _fact("EV-3", text="运营商需要通过站点选址、车辆周转和电池调度提升换电站资产利用率。", proof_role="metric"),
        ],
    )

    rendered = "\n".join(
        str(value)
        for unit in result["interpretation_units"]
        for value in [
            unit.get("core_judgment"),
            unit.get("what_evidence_reflects"),
            unit.get("why_it_matters"),
            unit.get("employment_implication"),
            unit.get("education_implication"),
            unit.get("industry_implication"),
        ]
    )
    assert "会计" not in rendered
    assert "课程" not in rendered
    assert "县域商用车充换电网络" in rendered
