from rag_pipeline.agents.public_narrative_bridge import build_public_bridge_pack


def _joined(pack: dict) -> str:
    return " ".join(str(value or "") for value in pack.values())


def test_public_bridge_pack_is_claim_specific_and_public_safe():
    pack = build_public_bridge_pack(
        claim="低空航线试点已经出现可观察的需求信号。",
        evidence_texts=[
            "多地公开材料披露低空航线试点和采购动作。",
            "地方平台发布低空飞行服务场景建设进展。",
        ],
        block_type="case_comparison",
        claim_strength="directional",
        boundary="试点仍不能直接证明大规模常态化运营。",
    )

    text = _joined(pack)

    assert pack["schema_version"] == "public_narrative_bridge_v1"
    assert len(pack["mechanism_bridge"]) >= 120
    assert len(pack["implication_bridge"]) >= 110
    assert "低空航线试点已经出现可观察的需求信号" in pack["claim_head"]
    assert "低空航线试点已经出现可观察的需求信号" not in pack["mechanism_bridge"]
    assert "公开材料" in pack["evidence_context"]
    assert "低空航线试点和采购动作" in pack["evidence_context"]
    assert "采购" in text or "服务场景" in text
    assert "试点仍不能直接证明大规模常态化运营" in pack["boundary_bridge"]
    for forbidden in (
        "diagnostic_only",
        "score_gap",
        "repair_task_seed",
        "search_more",
        "review_suggestion",
        "source_check",
        "semantic_judge",
        "补证建议",
        "审查建议",
        "内部诊断",
        "Public evidence points include",
        "正文应",
        "报告应",
        "公开事实包括",
        "事实说明",
        "这组事实",
        "这些事实",
        "这一判断",
        "围绕“",
        "后续应",
        "行业含义在于",
        "作为早期产业信号理解",
        "信息罗列",
        "外推边界",
        "这些材料把",
        "判断成立的条件",
        "事实指向",
        "已披露相关事实",
        "报告能否",
        "报告需要",
        "读者",
    ):
        assert forbidden not in text


def test_public_bridge_pack_varies_by_claim_and_block_type():
    case_pack = build_public_bridge_pack(
        claim="客户试点说明需求已经从概念讨论进入流程验证。",
        evidence_texts=["公开案例显示企业在客服流程中测试智能体。"],
        block_type="case_comparison",
        claim_strength="directional",
    )
    risk_pack = build_public_bridge_pack(
        claim="安全事故会削弱低空商业化推进节奏。",
        evidence_texts=["媒体披露部分地区加强无人机飞行安全监管。"],
        block_type="risk_trigger",
        claim_strength="directional",
    )

    assert case_pack["mechanism_bridge"] != risk_pack["mechanism_bridge"]
    assert "客户试点说明需求已经从概念讨论进入流程验证" in case_pack["claim_head"]
    assert "客户试点说明需求已经从概念讨论进入流程验证" not in case_pack["mechanism_bridge"]
    assert "安全事故会削弱低空商业化推进节奏" in risk_pack["claim_head"]
    assert "安全事故会削弱低空商业化推进节奏" not in risk_pack["mechanism_bridge"]
    assert "风险" in risk_pack["mechanism_bridge"] or "约束" in risk_pack["mechanism_bridge"]
    assert set(case_pack["template_keys"]) != set(risk_pack["template_keys"])
