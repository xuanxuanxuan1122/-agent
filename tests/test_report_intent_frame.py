from rag_pipeline.agents.brain_agent import build_search_tasks_for_goal
from rag_pipeline.agents.chapter_recomposer_agent import recompose_chapters_from_claims
from rag_pipeline.agents.dynamic_search_schema import normalize_research_plan
from rag_pipeline.agents import web_analysis_agent
from rag_pipeline.agents.research_planner import RESEARCH_PLANNER_SYSTEM


ACCOUNTING_QUERY = "会计学专业在AI时代的就业变化、岗位能力要求与教育培养调整研究"

COMMERCIAL_FORBIDDEN = ("市场规模", "竞争格局", "主要玩家", "头部玩家", "机会排序", "估值", "投资", "资本流向")


def test_research_planner_prompt_is_industry_report_production_first():
    assert "行业报告生产 AI" in RESEARCH_PLANNER_SYSTEM
    assert "企业研究报告" not in RESEARCH_PLANNER_SYSTEM
    assert "investment_memo" not in RESEARCH_PLANNER_SYSTEM
    assert "company_due_diligence" not in RESEARCH_PLANNER_SYSTEM
    assert "briefing_note" not in RESEARCH_PLANNER_SYSTEM


def test_normalize_research_plan_defaults_to_dynamic_research_report():
    plan = normalize_research_plan({"query": "具身智能产业发展趋势"}, query="具身智能产业发展趋势")

    assert plan["report_family"] == "dynamic_research_report"


def test_query_rewrite_prompt_no_longer_uses_due_diligence_branch():
    source = web_analysis_agent.build_llm_query_plan.__code__.co_consts
    joined = "\n".join(str(item) for item in source if isinstance(item, str))

    assert "company_due_diligence" not in joined


def test_profession_education_query_sets_report_intent_and_forbidden_terms():
    plan = normalize_research_plan(
        {
            "query": ACCOUNTING_QUERY,
            "research_object": ACCOUNTING_QUERY,
            "chapters": [
                {
                    "chapter_id": "ch_bad",
                    "chapter_title": "竞争格局由哪些玩家、能力、成本和渠道变量决定",
                    "core_question": "会计学专业是否存在可验证市场空间和主要玩家机会？",
                }
            ],
        },
        query=ACCOUNTING_QUERY,
    )

    assert plan["research_domain"] == "academic_or_professional_field"
    assert plan["report_intent"] == "profession_education_employment"
    assert {"市场规模", "主要玩家", "估值", "资本流向"} <= set(plan["global_forbidden_terms"])


def test_profession_education_normalize_rewrites_commercial_plan_chapters():
    plan = normalize_research_plan(
        {
            "query": ACCOUNTING_QUERY,
            "research_object": ACCOUNTING_QUERY,
            "chapters": [
                {
                    "chapter_id": "ch_01",
                    "chapter_title": "会计学专业是否存在真实需求和可验证市场空间，而不是概念热度",
                    "core_question": "会计学专业是否存在真实需求和可验证市场空间，而不是概念热度",
                },
                {
                    "chapter_id": "ch_02",
                    "chapter_title": "会计学专业的竞争格局由哪些玩家、能力、成本和渠道变量决定",
                    "core_question": "会计学专业的竞争格局由哪些玩家、能力、成本和渠道变量决定",
                },
                {
                    "chapter_id": "ch_03",
                    "chapter_title": "技术、供应、监管或替代约束会如何改变机会排序",
                    "core_question": "技术、供应、监管或替代约束会如何改变机会排序",
                },
            ],
        },
        query=ACCOUNTING_QUERY,
    )

    titles = [chapter["chapter_title"] for chapter in plan["chapters"]]
    assert len(titles) >= 4
    assert not any(term in title for title in titles for term in COMMERCIAL_FORBIDDEN)
    assert any("就业" in title and "岗位" in title for title in titles)
    assert any("能力" in title for title in titles)
    assert any("培养" in title or "课程" in title for title in titles)
    assert any("风险" in title or "边界" in title for title in titles)
    assert all(chapter.get("chapter_role") == "profession_education_frame" for chapter in plan["chapters"])


def test_profession_education_rewrite_is_query_driven_not_fixed_five_chapter_template():
    query = "会计学专业课程改革与人才培养研究"
    plan = normalize_research_plan(
        {
            "query": query,
            "research_object": query,
            "chapters": [
                {
                    "chapter_id": "ch_bad",
                    "chapter_title": "竞争格局由哪些玩家、能力、成本和渠道变量决定",
                    "core_question": "竞争格局由哪些玩家、能力、成本和渠道变量决定",
                }
            ],
        },
        query=query,
    )

    titles = [chapter["chapter_title"] for chapter in plan["chapters"]]
    assert 2 <= len(titles) < 5
    assert any("课程" in title or "培养" in title for title in titles)
    assert not any("企业用人标准" in title for title in titles)
    assert not any("就业风险" in title for title in titles)
    assert not any(term in title for title in titles for term in COMMERCIAL_FORBIDDEN)


def test_accounting_search_tasks_use_employment_frame_not_market_frame():
    plan = {
        "query": ACCOUNTING_QUERY,
        "research_object": ACCOUNTING_QUERY,
        "report_family": "industry_deep_report",
        "global_required_terms": ["会计学专业", "AI", "就业变化"],
    }
    chapter = {
        "chapter_id": "ch_bad",
        "chapter_title": "竞争格局由哪些玩家、能力、成本和渠道变量决定",
        "core_question": "会计学专业是否存在可验证市场空间和主要玩家机会？",
        "required_evidence_mix": ["company_filing", "market_research"],
    }
    goal = {
        "goal_id": "H_bad_metric",
        "requirement_id": "H_bad_metric",
        "question": "会计学专业市场规模、主要玩家、资本流向和投资机会",
        "proof_role": "metric",
        "must_have_terms": ["市场规模", "主要玩家", "资本流向"],
        "required_fields": ["metric", "value", "unit", "period", "source"],
    }

    tasks = build_search_tasks_for_goal(chapter=chapter, goal=goal, research_plan=plan)

    assert tasks
    queries = [task["query"] for task in tasks]
    assert all(any(term in query for term in ("会计学", "会计", "就业", "岗位", "课程", "人才培养")) for query in queries)
    assert not any(term in query for query in queries for term in COMMERCIAL_FORBIDDEN)
    assert all("profession_education_employment" == task.get("report_intent") for task in tasks)
    assert all({"市场规模", "主要玩家", "资本流向"} <= set(task.get("forbidden_terms") or []) for task in tasks)


def test_accounting_claim_first_recomposition_replaces_commercial_cluster_titles():
    result = recompose_chapters_from_claims(
        query=ACCOUNTING_QUERY,
        plan_blueprint={
            "query": ACCOUNTING_QUERY,
            "research_object": ACCOUNTING_QUERY,
            "chapters": [
                {
                    "chapter_id": "plan_bad",
                    "chapter_title": "主要玩家与竞争格局",
                    "core_question": "竞争格局由哪些玩家决定？",
                }
            ],
        },
        structured_analysis={
            "claim_units": [
                {
                    "claim_id": "CL-accounting-ability",
                    "chapter_id": "plan_bad",
                    "cluster_key": "competition",
                    "recommended_chapter": "主要玩家与竞争格局",
                    "section_title": "玩家动作有哪些差异",
                    "claim": "会计岗位能力正在从基础核算转向数据解释、流程治理和智能财税工具协同。",
                    "fact_ids": ["EV-1", "EV-2"],
                    "source_ids": ["SRC-1"],
                    "requirement_ids": ["REQ-employment"],
                    "claim_strength": "directional",
                    "can_anchor_section": True,
                }
            ]
        },
        evidence_package={},
    )

    titles = [chapter["chapter_title"] for chapter in result["final_chapters"]]
    assert titles
    assert not any(term in title for title in titles for term in ("主要玩家", "竞争格局", "机会排序", "投资", "资本"))
    assert any(any(term in title for term in ("岗位", "能力", "培养", "就业")) for title in titles)
