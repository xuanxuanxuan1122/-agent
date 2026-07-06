from __future__ import annotations

from rag_pipeline.agents.chapter_recomposer_agent import recompose_chapters_from_claims


def test_claim_first_recomposer_drops_empty_plan_chapters_and_clusters_claims():
    plan_blueprint = {
        "chapters": [
            {"chapter_id": "plan_market", "chapter_title": "Original market chapter"},
            {"chapter_id": "plan_policy", "chapter_title": "Original policy chapter"},
            {"chapter_id": "plan_empty", "chapter_title": "Original empty chapter"},
            {"chapter_id": "plan_risk", "chapter_title": "Original risk chapter"},
            {"chapter_id": "plan_ops", "chapter_title": "Original operations chapter"},
            {"chapter_id": "plan_finance", "chapter_title": "Original finance chapter"},
        ]
    }
    structured_analysis = {
        "claim_units": [
            {
                "claim_id": "CL-market",
                "claim": "Market demand is shifting toward repeatable workflow scenarios.",
                "chapter_id": "plan_market",
                "cluster_key": "market",
                "fact_ids": ["F-market"],
                "source_ids": ["S-market"],
                "claim_strength": "moderate",
                "can_anchor_section": True,
            },
            {
                "claim_id": "CL-risk",
                "claim": "Implementation risk is concentrated in integration and compliance.",
                "chapter_id": "plan_risk",
                "cluster_key": "risk",
                "fact_ids": ["F-risk"],
                "source_ids": ["S-risk"],
                "claim_strength": "directional",
                "can_anchor_section": True,
            },
            {
                "claim_id": "CL-ecosystem",
                "claim": "Developer ecosystem activity is a separate adoption signal.",
                "recommended_chapter": "Developer ecosystem",
                "cluster_key": "ecosystem",
                "fact_ids": ["F-eco"],
                "source_ids": ["S-eco"],
                "claim_strength": "weak",
                "can_anchor_section": True,
            },
        ]
    }

    result = recompose_chapters_from_claims(
        plan_blueprint=plan_blueprint,
        structured_analysis=structured_analysis,
        evidence_package={},
        query="AI Agent industry report",
    )

    assert result["status"] == "claim_first"
    assert result["plan_blueprint"]["blueprint_role"] == "research_plan"
    assert result["plan_blueprint"]["final_outline_locked"] is False
    assert len(result["claim_clusters"]) == 3
    assert len(result["final_chapters"]) == 2
    assert "plan_empty" in result["dropped_plan_chapter_ids"]
    assert all(chapter["claim_ids"] for chapter in result["final_chapters"])
    assert all(chapter["fact_ids"] for chapter in result["final_chapters"])
    assert any(chapter["recomposition_action"] == "merged" for chapter in result["final_chapters"])
    limitation_chapters = [chapter for chapter in result["final_chapters"] if chapter["writing_mode"] == "limitations"]
    assert len(limitation_chapters) == 1
    assert set(limitation_chapters[0]["claim_ids"]) == {"CL-risk", "CL-ecosystem"}
    assert result["metrics"]["claim_to_final_chapter_transfer_rate"] == 1.0


def test_recomposer_falls_back_to_plan_when_no_bound_claims():
    result = recompose_chapters_from_claims(
        plan_blueprint={"chapters": [{"chapter_id": "plan_1", "chapter_title": "Plan only"}]},
        structured_analysis={"claim_units": [{"claim_id": "CL-empty", "claim": "No refs"}]},
        evidence_package={},
    )

    assert result["status"] == "fallback_to_plan"
    assert result["final_chapters"] == []
    assert result["metrics"]["reanalyze_existing_recommended"] is True


def test_recomposer_orders_final_chapters_by_plan_order_not_claim_arrival_order():
    result = recompose_chapters_from_claims(
        plan_blueprint={
            "chapters": [
                {"chapter_id": "ch_01", "chapter_title": "Market first"},
                {"chapter_id": "ch_02", "chapter_title": "Competition second"},
            ]
        },
        structured_analysis={
            "claim_units": [
                {
                    "claim_id": "CL-competition",
                    "claim": "Competition has already formed around leading players.",
                    "chapter_id": "ch_02",
                    "cluster_key": "competition",
                    "fact_ids": ["F2"],
                    "source_ids": ["S2"],
                    "claim_strength": "moderate",
                },
                {
                    "claim_id": "CL-market",
                    "claim": "Market demand is still the first question to verify.",
                    "chapter_id": "ch_01",
                    "cluster_key": "market",
                    "fact_ids": ["F1"],
                    "source_ids": ["S1"],
                    "claim_strength": "moderate",
                },
            ]
        },
        evidence_package={},
    )

    assert [chapter["chapter_id"] for chapter in result["final_chapters"]] == ["ch_01", "ch_02"]


def test_recomposer_does_not_turn_claim_roles_into_industry_report_chapters():
    result = recompose_chapters_from_claims(
        plan_blueprint={
            "chapters": [
                {"chapter_id": "ch_01", "chapter_title": "政策环境与行业背景"},
                {"chapter_id": "ch_04", "chapter_title": "资产利用率与成本结构分析"},
            ]
        },
        structured_analysis={
            "claim_units": [
                {
                    "claim_id": "CL-policy",
                    "claim": "县域充换电网络正在被纳入地方产业规划。",
                    "chapter_id": "ch_01",
                    "recommended_chapter": "政策环境与行业背景",
                    "cluster_key": "core_claim",
                    "fact_ids": ["F-policy"],
                    "source_ids": ["S-policy"],
                    "claim_strength": "moderate",
                },
                {
                    "claim_id": "CL-utilization",
                    "claim": "运营商盈利模型取决于换电站资产利用率和车辆周转密度。",
                    "chapter_id": "ch_04",
                    "recommended_chapter": "资产利用率与成本结构分析",
                    "cluster_key": "mechanism_claim",
                    "fact_ids": ["F-util"],
                    "source_ids": ["S-util"],
                    "claim_strength": "moderate",
                },
            ]
        },
        evidence_package={},
        query="县域新能源商用车充换电网络：重卡物流场景、资产利用率与运营商盈利模型研究",
    )

    titles = [chapter["chapter_title"] for chapter in result["final_chapters"]]
    assert "政策环境与行业背景" in titles
    assert "资产利用率与成本结构分析" in titles
    assert "核心判断与方向选择" not in titles
    assert "传导机制与能力重构" not in titles


def test_recomposer_moves_single_thin_cluster_to_limitations_when_core_exists():
    result = recompose_chapters_from_claims(
        plan_blueprint={"chapters": [{"chapter_id": "plan_market", "chapter_title": "Market"}]},
        structured_analysis={
            "claim_units": [
                {
                    "claim_id": "CL-core",
                    "claim": "Market adoption has a repeatable demand signal.",
                    "chapter_id": "plan_market",
                    "cluster_key": "market",
                    "fact_ids": ["F-core"],
                    "source_ids": ["S-core"],
                    "claim_strength": "moderate",
                },
                {
                    "claim_id": "CL-thin",
                    "claim": "One case suggests ecosystem experimentation remains uneven.",
                    "cluster_key": "ecosystem",
                    "fact_ids": ["F-thin"],
                    "source_ids": ["S-thin"],
                    "claim_strength": "directional",
                },
            ]
        },
        evidence_package={},
    )

    assert [(chapter["chapter_id"], chapter["writing_mode"]) for chapter in result["final_chapters"]] == [
        ("plan_market", "core_chapter"),
        ("CH_limitations", "limitations"),
    ]
    assert "plan_market" in result["final_chapters"][0]["chapter_id_aliases"]
    assert result["final_chapters"][1]["claim_ids"] == ["CL-thin"]


def test_recomposer_normalizes_missing_claim_ids_into_section_plan_refs():
    result = recompose_chapters_from_claims(
        plan_blueprint={"chapters": [{"chapter_id": "ch_01", "chapter_title": "真实需求验证"}]},
        structured_analysis={
            "claim_units": [
                {
                    "id": "ch_01_llm_1",
                    "claim": "春节场景付费租赁显示短期需求已经出现。",
                    "chapter_id": "ch_01",
                    "fact_ids": ["EV-1"],
                    "source_ids": ["S1"],
                    "claim_strength": "directional",
                },
                {
                    "claim": "政策文件为人形机器人商业化提供制度环境。",
                    "chapter_id": "ch_01",
                    "used_fact_refs": ["EV-2"],
                    "source_ids": ["S2"],
                    "claim_strength": "moderate",
                },
            ]
        },
        evidence_package={},
    )

    assert result["metrics"]["claim_to_final_chapter_transfer_rate"] == 1.0
    assert result["final_chapters"][0]["chapter_id"] == "ch_01"
    assert "ch_01" in result["final_chapters"][0]["chapter_id_aliases"]
    assert [unit["claim_id"] for unit in result["normalized_claim_units"]] == [
        "ch_01_llm_1",
        "ch_01_claim_2",
    ]
    section_plan = result["final_chapters"][0]["section_plan"]
    assert [section["claim_id"] for section in section_plan] == ["ch_01_llm_1", "ch_01_claim_2"]
    assert section_plan[0]["required_evidence_refs"] == ["EV-1"]
    assert section_plan[1]["required_evidence_refs"] == ["EV-2"]
    assert section_plan[0]["matched_llm_claim"]["claim_id"] == "ch_01_llm_1"


def test_recomposer_replaces_repeated_generic_section_titles_with_claim_context():
    generic_title = "\u5e02\u573a\u7a7a\u95f4\u5230\u5e95\u6709\u591a\u5927"
    result = recompose_chapters_from_claims(
        plan_blueprint={"chapters": [{"chapter_id": "ch_market", "chapter_title": "Market"}]},
        structured_analysis={
            "claim_units": [
                {
                    "claim_id": "CL-rental",
                    "claim": "Humanoid robot rental demand is appearing first in seasonal service scenarios.",
                    "section_title": generic_title,
                    "chapter_id": "ch_market",
                    "cluster_key": "market",
                    "fact_ids": ["EV-rental"],
                    "source_ids": ["S1"],
                    "claim_strength": "directional",
                },
                {
                    "claim_id": "CL-factory",
                    "claim": "Factory pilot deployments are a separate commercialization signal from rental demand.",
                    "section_title": generic_title,
                    "chapter_id": "ch_market",
                    "cluster_key": "market",
                    "fact_ids": ["EV-factory"],
                    "source_ids": ["S2"],
                    "claim_strength": "directional",
                },
            ]
        },
        evidence_package={},
    )

    section_titles = [section["section_title"] for section in result["final_chapters"][0]["section_plan"]]

    assert generic_title not in section_titles
    assert len(set(section_titles)) == 2
    assert section_titles[0].startswith("Humanoid robot rental demand")
    assert section_titles[1].startswith("Factory pilot deployments")


def test_recomposer_merges_clusters_that_resolve_to_same_public_chapter_title():
    result = recompose_chapters_from_claims(
        plan_blueprint={
            "chapters": [
                {
                    "chapter_id": "ch_market",
                    "chapter_title": "Market demand and commercialization signals",
                }
            ]
        },
        structured_analysis={
            "claim_units": [
                {
                    "claim_id": "CL-metric",
                    "claim": "Market forecasts show a measurable demand signal.",
                    "chapter_id": "ch_market",
                    "cluster_key": "metric_claim",
                    "fact_ids": ["EV-1"],
                    "source_ids": ["S1"],
                    "claim_strength": "moderate",
                },
                {
                    "claim_id": "CL-case",
                    "claim": "Local pilots show commercialization moving into concrete scenarios.",
                    "chapter_id": "ch_market",
                    "cluster_key": "case_claim",
                    "fact_ids": ["EV-2"],
                    "source_ids": ["S2"],
                    "claim_strength": "directional",
                },
            ]
        },
        evidence_package={},
    )

    assert len(result["claim_clusters"]) == 2
    assert len(result["final_chapters"]) == 1
    chapter = result["final_chapters"][0]
    assert chapter["chapter_title"] == "Market demand and commercialization signals"
    assert chapter["claim_ids"] == ["CL-metric", "CL-case"]
    assert set(chapter["fact_ids"]) == {"EV-1", "EV-2"}
    assert chapter["recomposition_action"] == "merged"


def test_recomposer_keeps_distinct_cluster_keys_when_generic_titles_match_without_single_plan_anchor():
    repeated_title = "会计学专业在AI时代的就业变化"
    structured_analysis = {
        "claim_units": [
            {
                "claim_id": "CL-context",
                "claim": "AI adoption changes accounting employment through workflow restructuring.",
                "recommended_chapter": repeated_title,
                "chapter_id": "ch_01",
                "cluster_key": "contextual_claim",
                "fact_ids": ["EV-1"],
                "source_ids": ["S1"],
                "claim_strength": "directional",
            },
            {
                "claim_id": "CL-context-2",
                "claim": "Accounting education is responding to the same employment shift.",
                "recommended_chapter": repeated_title,
                "chapter_id": "ch_02",
                "cluster_key": "contextual_claim",
                "fact_ids": ["EV-1b"],
                "source_ids": ["S1b"],
                "claim_strength": "directional",
            },
            {
                "claim_id": "CL-risk",
                "claim": "Compliance constraints limit how fast accounting tasks can be automated.",
                "recommended_chapter": repeated_title,
                "chapter_id": "ch_02",
                "cluster_key": "counter_boundary_claim",
                "fact_ids": ["EV-2"],
                "source_ids": ["S2"],
                "claim_strength": "directional",
            },
            {
                "claim_id": "CL-risk-2",
                "claim": "Audit responsibility keeps human review in the loop.",
                "recommended_chapter": repeated_title,
                "chapter_id": "ch_03",
                "cluster_key": "counter_boundary_claim",
                "fact_ids": ["EV-2b"],
                "source_ids": ["S2b"],
                "claim_strength": "directional",
            },
            {
                "claim_id": "CL-mechanism",
                "claim": "Curriculum updates need to connect AI tools with audit judgment.",
                "recommended_chapter": repeated_title,
                "chapter_id": "ch_03",
                "cluster_key": "mechanism_claim",
                "fact_ids": ["EV-3"],
                "source_ids": ["S3"],
                "claim_strength": "directional",
            },
            {
                "claim_id": "CL-mechanism-2",
                "claim": "Training providers are packaging AI tools with accounting practice cases.",
                "recommended_chapter": repeated_title,
                "chapter_id": "ch_01",
                "cluster_key": "mechanism_claim",
                "fact_ids": ["EV-3b"],
                "source_ids": ["S3b"],
                "claim_strength": "directional",
            },
        ]
    }

    result = recompose_chapters_from_claims(
        plan_blueprint={
            "chapters": [
                {"chapter_id": "ch_01", "chapter_title": "就业变化"},
                {"chapter_id": "ch_02", "chapter_title": "合规边界"},
                {"chapter_id": "ch_03", "chapter_title": "培养调整"},
            ]
        },
        structured_analysis=structured_analysis,
        evidence_package={},
        query="会计学专业在AI时代的就业变化",
    )

    assert len(result["claim_clusters"]) == 3
    assert len(result["final_chapters"]) == 3
    assert all(len(chapter["claim_ids"]) == 2 for chapter in result["final_chapters"])


def test_recomposer_uses_role_titles_for_repeated_generic_recommended_chapters():
    repeated_title = "会计学专业在AI时代的就业变化"
    result = recompose_chapters_from_claims(
        plan_blueprint={"chapters": []},
        structured_analysis={
            "claim_units": [
                {
                    "claim_id": "CL-context",
                    "claim": "Policy and market background shape accounting employment changes.",
                    "recommended_chapter": repeated_title,
                    "cluster_key": "contextual_claim",
                    "fact_ids": ["EV-1"],
                    "source_ids": ["S1"],
                    "claim_strength": "directional",
                },
                {
                    "claim_id": "CL-mechanism",
                    "claim": "AI tools shift accounting work from data entry to judgment and control.",
                    "recommended_chapter": repeated_title,
                    "cluster_key": "mechanism_claim",
                    "fact_ids": ["EV-2"],
                    "source_ids": ["S2"],
                    "claim_strength": "directional",
                },
                {
                    "claim_id": "CL-boundary",
                    "claim": "Audit accountability limits full automation in accounting roles.",
                    "recommended_chapter": repeated_title,
                    "cluster_key": "counter_boundary_claim",
                    "fact_ids": ["EV-3"],
                    "source_ids": ["S3"],
                    "claim_strength": "directional",
                },
            ]
        },
        evidence_package={},
        query=repeated_title,
    )

    titles = [chapter["chapter_title"] for chapter in result["final_chapters"]]

    assert len(titles) == 3
    assert len(set(titles)) == 3
    assert repeated_title not in titles
    assert "就业与培养环境变化" in titles
    assert "AI工具推动的能力重构" in titles
    assert "就业风险与培养边界" in titles


def test_recomposer_keeps_semantic_adjacent_claim_as_boundary_section_not_core_view():
    result = recompose_chapters_from_claims(
        plan_blueprint={"chapters": [{"chapter_id": "ch_risk", "chapter_title": "Risk boundaries"}]},
        structured_analysis={
            "claim_units": [
                {
                    "claim_id": "CL-adjacent",
                    "claim": "Macro oil price pressure is only adjacent context for low-altitude demand.",
                    "chapter_id": "ch_risk",
                    "cluster_key": "risk",
                    "fact_ids": ["EV-oil"],
                    "source_ids": ["S-oil"],
                    "claim_strength": "moderate",
                    "semantic_judge_status": "adjacent",
                    "claim_review_suggestions": [
                        {
                            "issue_type": "llm_claim_semantic_judge_adjacent",
                            "suggested_claim_strength": "directional",
                        }
                    ],
                }
            ]
        },
        evidence_package={},
    )

    chapter = result["final_chapters"][0]
    section = chapter["section_plan"][0]

    assert chapter["writing_mode"] in {"directional_observation", "limitations"}
    assert section["must_not_use_as_core_view"] is True
    assert section["section_role"] == "boundary_context"
    assert section["matched_llm_claim"]["claim_strength"] == "directional"


def test_recomposer_compacts_long_claim_titles_at_natural_boundary():
    result = recompose_chapters_from_claims(
        plan_blueprint={"chapters": [{"chapter_id": "ch_employment", "chapter_title": "Plan"}]},
        structured_analysis={
            "claim_units": [
                {
                    "claim_id": "CL-accounting",
                    "claim": (
                        "\u4f1a\u8ba1\u5b66\u4e13\u4e1a\u5c31\u4e1a\u7684\u73b0\u5b9e\u4ef7\u503c"
                        "\u5e94\u4ece\u4eba\u624d\u9700\u6c42\u3001\u5c97\u4f4d\u7ed3\u6784"
                        "\u548c\u7ec4\u7ec7\u8d22\u52a1\u6cbb\u7406\u9700\u6c42\u9a8c\u8bc1"
                    ),
                    "recommended_chapter": (
                        "\u4f1a\u8ba1\u5b66\u4e13\u4e1a\u5c31\u4e1a\u7684\u73b0\u5b9e\u4ef7\u503c"
                        "\u5e94\u4ece\u4eba\u624d\u9700\u6c42\u3001\u5c97\u4f4d\u7ed3\u6784"
                        "\u548c\u7ec4\u7ec7\u8d22\u52a1\u6cbb\u7406\u9700\u6c42\u9a8c\u8bc1"
                    ),
                    "chapter_id": "ch_employment",
                    "cluster_key": "core_claim",
                    "fact_ids": ["EV-1"],
                    "source_ids": ["S1"],
                    "claim_strength": "moderate",
                    "can_anchor_section": True,
                }
            ]
        },
        evidence_package={},
    )

    chapter = result["final_chapters"][0]
    section = chapter["section_plan"][0]

    assert chapter["chapter_title"] == "\u4f1a\u8ba1\u5b66\u4e13\u4e1a\u5c31\u4e1a\u7684\u73b0\u5b9e\u4ef7\u503c"
    assert section["section_title"] == "\u4f1a\u8ba1\u5b66\u4e13\u4e1a\u5c31\u4e1a\u7684\u73b0\u5b9e\u4ef7\u503c"
    assert not chapter["chapter_title"].endswith("\u8d22")
