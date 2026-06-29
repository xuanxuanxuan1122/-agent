from __future__ import annotations

from rag_pipeline.agents.chapter_narrative_planner import (
    apply_narrative_plan_to_final_chapters,
    apply_narrative_plan_to_claim_units,
    build_chapter_narrative_plan,
)


def test_narrative_plan_merges_related_claims_into_public_paragraph_plan():
    final_chapters = [
        {
            "chapter_id": "ch_market",
            "chapter_title": "商业化节奏与需求验证",
            "claim_ids": ["CL-setup", "CL-case-1", "CL-case-2", "CL-risk"],
            "writing_mode": "core_chapter",
        }
    ]
    claim_units = [
        {
            "claim_id": "CL-setup",
            "chapter_id": "ch_market",
            "cluster_key": "market",
            "claim": "公开信息显示，需求验证正在从概念讨论转向可落地场景。",
            "fact_ids": ["EV-setup"],
            "source_ids": ["SRC-setup"],
            "claim_strength": "moderate",
        },
        {
            "claim_id": "CL-case-1",
            "chapter_id": "ch_market",
            "cluster_key": "case",
            "claim": "试点项目已经进入客户服务流程。",
            "fact_ids": ["EV-case-1"],
            "source_ids": ["SRC-case"],
            "claim_strength": "directional",
        },
        {
            "claim_id": "CL-case-2",
            "chapter_id": "ch_market",
            "cluster_key": "case",
            "claim": "同类项目还出现在销售支持和运营协同场景。",
            "fact_ids": ["EV-case-2"],
            "source_ids": ["SRC-case"],
            "claim_strength": "directional",
        },
        {
            "claim_id": "CL-risk",
            "chapter_id": "ch_market",
            "cluster_key": "risk",
            "claim": "交付节奏仍受预算和组织流程约束。",
            "fact_ids": ["EV-risk"],
            "source_ids": ["SRC-risk"],
            "claim_strength": "weak",
        },
    ]

    plan = build_chapter_narrative_plan(
        final_chapters=final_chapters,
        claim_units=claim_units,
    )

    assert plan["schema_version"] == "chapter_narrative_plan_v1"
    assert plan["metrics"]["chapter_count"] == 1
    assert plan["metrics"]["merged_claim_count"] >= 1
    assert plan["metrics"]["avg_claims_per_paragraph"] > 1

    chapter_plan = plan["chapter_plans"][0]
    roles = [item["role"] for item in chapter_plan["paragraph_plans"]]
    assert roles[0] == "setup"
    assert "evidence_progression" in roles
    assert roles[-1] == "constraint"

    evidence_plan = next(item for item in chapter_plan["paragraph_plans"] if item["role"] == "evidence_progression")
    assert evidence_plan["main_claim_id"] == "CL-case-1"
    assert evidence_plan["supporting_claim_ids"] == ["CL-case-2"]
    assert set(evidence_plan["fact_ids"]) == {"EV-case-1", "EV-case-2"}
    assert evidence_plan["must_not_render"] is True
    assert evidence_plan["public_text_allowed"] is False


def test_narrative_plan_application_keeps_main_claim_and_merges_supporting_refs():
    plan = {
        "paragraph_plan_by_claim_id": {
            "CL-main": {
                "paragraph_id": "p1",
                "role": "evidence_progression",
                "main_claim_id": "CL-main",
                "supporting_claim_ids": ["CL-support"],
                "claim_ids": ["CL-main", "CL-support"],
                "fact_ids": ["EV-main", "EV-support"],
                "source_ids": ["SRC-main", "SRC-support"],
                "transition_in": "这一信号需要和场景扩展放在一起看。",
                "transition_out": "因此判断重点转向商业闭环。",
                "writing_goal": "internal planning only",
                "must_not_render": True,
                "public_text_allowed": False,
            },
            "CL-support": {
                "paragraph_id": "p1",
                "role": "evidence_progression",
                "main_claim_id": "CL-main",
                "supporting_claim_ids": ["CL-support"],
                "claim_ids": ["CL-main", "CL-support"],
                "fact_ids": ["EV-main", "EV-support"],
                "source_ids": ["SRC-main", "SRC-support"],
                "must_not_render": True,
                "public_text_allowed": False,
            },
        }
    }
    units = [
        {
            "claim_id": "CL-main",
            "claim": "主判断。",
            "fact_ids": ["EV-main"],
            "source_ids": ["SRC-main"],
        },
        {
            "claim_id": "CL-support",
            "claim": "补充判断。",
            "fact_ids": ["EV-support"],
            "source_ids": ["SRC-support"],
        },
    ]

    enriched = apply_narrative_plan_to_claim_units(units, plan)

    main = next(item for item in enriched if item["claim_id"] == "CL-main")
    support = next(item for item in enriched if item["claim_id"] == "CL-support")

    assert main["paragraph_plan_id"] == "p1"
    assert main["narrative_role"] == "evidence_progression"
    assert main["paragraph_supporting_claim_ids"] == ["CL-support"]
    assert main["narrative_supporting_claims"] == ["补充判断。"]
    assert set(main["fact_ids"]) == {"EV-main", "EV-support"}
    assert set(main["source_ids"]) == {"SRC-main", "SRC-support"}
    assert main["narrative_do_not_render"] is True

    assert support["omit_from_report"] is True
    assert support["narrative_merged_into_claim_id"] == "CL-main"
    assert support["public_render"] is False


def test_narrative_plan_rewrites_final_chapter_section_plan_to_paragraphs():
    final_chapters = [
        {
            "chapter_id": "ch_market",
            "chapter_title": "商业化节奏",
            "claim_ids": ["CL-main", "CL-support"],
            "section_plan": [
                {
                    "section_id": "old_1",
                    "claim_id": "CL-main",
                    "section_title": "客户服务流程",
                    "matched_llm_claim": {"claim_id": "CL-main", "claim": "主判断。"},
                },
                {
                    "section_id": "old_2",
                    "claim_id": "CL-support",
                    "section_title": "运营协同场景",
                    "matched_llm_claim": {"claim_id": "CL-support", "claim": "补充判断。"},
                },
            ],
        }
    ]
    narrative_plan = {
        "chapter_plans": [
            {
                "chapter_id": "ch_market",
                "paragraph_plans": [
                    {
                        "paragraph_id": "ch_market_p01",
                        "role": "evidence_progression",
                        "main_claim_id": "CL-main",
                        "supporting_claim_ids": ["CL-support"],
                        "claim_ids": ["CL-main", "CL-support"],
                        "fact_ids": ["EV-main", "EV-support"],
                        "source_ids": ["SRC-main", "SRC-support"],
                        "must_not_render": True,
                        "public_text_allowed": False,
                    }
                ],
            }
        ]
    }

    rewritten = apply_narrative_plan_to_final_chapters(final_chapters, narrative_plan)

    assert len(rewritten[0]["section_plan"]) == 1
    section = rewritten[0]["section_plan"][0]
    assert section["section_id"] == "ch_market_p01"
    assert section["claim_id"] == "CL-main"
    assert section["claim_ids"] == ["CL-main", "CL-support"]
    assert section["required_evidence_refs"] == ["EV-main", "EV-support"]
    assert section["narrative_role"] == "evidence_progression"
    assert section["supporting_claim_ids"] == ["CL-support"]
    assert rewritten[0]["narrative_paragraph_count"] == 1
