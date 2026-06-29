from rag_pipeline.agents.brain_agent import build_search_tasks_for_goal, expand_search_tasks_from_chapters
from rag_pipeline.agents.dynamic_search_schema import normalize_research_plan


def _low_altitude_plan():
    return {
        "query": "中国低空经济商业化落地机会与风险：政策、产业链、应用场景与投资判断",
        "research_object": "中国低空经济",
        "global_required_terms": ["中国低空经济"],
        "report_family": "industry_deep_report",
    }


def _low_altitude_chapter():
    return {
        "chapter_id": "ch_01",
        "chapter_title": "中国低空经济是否存在真实需求和可验证市场空间，而不是概念热度",
        "core_question": "低空经济商业化是否有真实需求、政策牵引和可验证市场空间",
        "chapter_question": "低空经济商业化是否有真实需求、政策牵引和可验证市场空间",
        "required_evidence_mix": ["official_data", "market_research", "company_filing"],
    }


def _low_altitude_goal():
    return {
        "goal_id": "H1_metric",
        "requirement_id": "H1_metric",
        "question": "中国低空经济市场规模、应用场景和政策落地的指标口径",
        "proof_role": "metric",
        "must_have_terms": ["中国低空经济", "市场规模", "应用场景"],
        "source_priority": ["official_data", "market_research", "company_filing"],
        "required_source_levels": ["A", "B"],
        "required_fields": ["metric", "value", "unit", "period", "source"],
    }


def test_low_altitude_search_tasks_keep_topic_anchor():
    tasks = build_search_tasks_for_goal(
        chapter=_low_altitude_chapter(),
        goal=_low_altitude_goal(),
        research_plan=_low_altitude_plan(),
    )

    assert tasks
    for task in tasks:
        assert any(anchor in task["query"] for anchor in ["低空经济", "eVTOL", "无人机", "通航"])
        assert "供应链 中国市场" not in task["query"]
        assert task.get("topic_anchor_status") in {"ok", "repaired"}
        assert "中国低空经济" in task.get("topic_anchor_terms", [])
        assert "存在真实需求" not in task.get("topic_anchor_terms", [])


def test_supply_chain_anchor_is_allowed_when_it_is_research_object():
    plan = {
        "query": "中国供应链数字化商业化机会与风险",
        "research_object": "中国供应链数字化",
        "global_required_terms": ["中国供应链数字化"],
        "report_family": "industry_deep_report",
    }
    chapter = {
        "chapter_id": "ch_01",
        "chapter_title": "中国供应链数字化的市场空间与客户需求",
        "core_question": "中国供应链数字化是否存在可验证需求",
        "required_evidence_mix": ["official_data", "market_research"],
    }
    goal = {
        "goal_id": "H1_metric",
        "requirement_id": "H1_metric",
        "question": "中国供应链数字化市场规模和客户需求指标",
        "proof_role": "metric",
        "must_have_terms": ["中国供应链数字化", "市场规模"],
        "required_fields": ["metric", "value", "unit", "period", "source"],
    }

    tasks = build_search_tasks_for_goal(chapter=chapter, goal=goal, research_plan=plan)

    assert tasks
    assert all("供应链" in task["query"] for task in tasks)
    assert all(task.get("topic_anchor_status") in {"ok", "repaired"} for task in tasks)
    assert all("中国供应链数字化" in task.get("topic_anchor_terms", []) for task in tasks)


def test_expanded_search_tasks_keep_low_altitude_anchor():
    research_plan = _low_altitude_plan()
    chapter = _low_altitude_chapter()
    expanded = expand_search_tasks_from_chapters(research_plan, {"chapters": [chapter]})

    queries = [task["query"] for task in expanded.get("search_tasks", [])]

    assert queries
    assert all(any(anchor in query for anchor in ["低空经济", "eVTOL", "无人机", "通航"]) for query in queries)
    assert not any("供应链 中国市场" in query for query in queries)


def test_normalized_research_plan_segments_topic_anchor_from_dimensions():
    query = (
        "\u4e2d\u56fd\u4f4e\u7a7a\u7ecf\u6d4e\u5546\u4e1a\u5316\u843d\u5730\u673a\u4f1a\u4e0e\u98ce\u9669"
        "\uff1a\u653f\u7b56\u3001\u4ea7\u4e1a\u94fe\u3001\u5e94\u7528\u573a\u666f\u4e0e\u6295\u8d44\u5224\u65ad"
    )
    plan = normalize_research_plan(
        {
            "query": query,
            "research_object": "\u4e2d\u56fd\u4f4e\u7a7a\u7ecf\u6d4e",
            "global_required_terms": ["\u4e2d\u56fd\u4f4e\u7a7a\u7ecf\u6d4e"],
            "search_tasks": [
                {
                    "task_id": "task-metric",
                    "query": "\u5e02\u573a\u89c4\u6a21 \u5b98\u65b9\u7edf\u8ba1",
                    "proof_role": "metric",
                    "must_have_terms": ["\u5e02\u573a\u89c4\u6a21"],
                }
            ],
        },
        query=query,
    )

    assert "\u4e2d\u56fd\u4f4e\u7a7a\u7ecf\u6d4e" in plan["topic_anchor_terms"]
    assert "\u4ea7\u4e1a\u94fe" not in plan["topic_anchor_terms"]
    assert "\u4f9b\u5e94\u94fe" not in plan["topic_anchor_terms"]
    assert {"\u653f\u7b56", "\u4ea7\u4e1a\u94fe", "\u5e94\u7528\u573a\u666f", "\u6295\u8d44\u5224\u65ad"} <= set(plan["evidence_dimensions"])
    assert "\u5546\u4e1a\u5316" in plan["generic_dimensions"]
    assert plan["search_tasks"][0]["topic_anchor_terms"] == plan["topic_anchor_terms"]
    assert "\u4f4e\u7a7a\u7ecf\u6d4e" in plan["search_tasks"][0]["query"]


def test_normalize_research_plan_prefers_original_unicode_query_anchor_over_wrong_planner_object():
    query = "\u4e2d\u56fd\u4f4e\u7a7a\u7ecf\u6d4e\u4ea7\u4e1a\u94fe\u5546\u4e1a\u5316\u673a\u4f1a\u4e0e\u98ce\u9669\u5206\u6790\uff082026\uff09"
    plan = normalize_research_plan(
        {
            "query": "\u4eba\u5de5\u667a\u80fd\u82af\u7247\u884c\u4e1a\u5e02\u573a\u89c4\u6a21\u4e0e\u7ade\u4e89\u683c\u5c402026",
            "research_object": "wind energy windenergy windpowercapac",
            "global_required_terms": ["wind energy"],
            "chapters": [
                {
                    "chapter_id": "ch_01",
                    "chapter_title": "\u662f\u5426\u5b58\u5728\u771f\u5b9e\u9700\u6c42",
                    "core_question": "\u662f\u5426\u5b58\u5728\u771f\u5b9e\u9700\u6c42",
                }
            ],
            "search_tasks": [
                {
                    "task_id": "task_metric",
                    "chapter_id": "ch_01",
                    "query": "\u5e02\u573a\u89c4\u6a21 \u9700\u6c42\u589e\u901f",
                    "proof_role": "metric",
                }
            ],
        },
        query=query,
    )

    assert "\u4f4e\u7a7a\u7ecf\u6d4e" in plan["research_object"]
    assert any("\u4f4e\u7a7a\u7ecf\u6d4e" in term for term in plan["topic_anchor_terms"])
    assert not any("wind energy" in term for term in plan["topic_anchor_terms"])
    assert "\u4f4e\u7a7a\u7ecf\u6d4e" in plan["search_tasks"][0]["query"]


def test_normalize_research_plan_preserves_research_domain_for_monitoring():
    plan = normalize_research_plan(
        {
            "query": "\u4f1a\u8ba1\u5b66",
            "research_object": "\u4f1a\u8ba1\u5b66",
            "research_domain": "academic_or_professional_field",
        },
        query="\u4f1a\u8ba1\u5b66",
    )

    assert plan["research_domain"] == "academic_or_professional_field"


def test_real_unicode_low_altitude_tasks_keep_query_anchor_and_quality():
    plan = {
        "query": "\u4e2d\u56fd\u4f4e\u7a7a\u7ecf\u6d4e\u4ea7\u4e1a\u94fe\u5546\u4e1a\u5316\u673a\u4f1a\u4e0e\u98ce\u9669\u5206\u6790\uff082026\uff09",
        "research_object": "\u4e2d\u56fd\u4f4e\u7a7a\u7ecf\u6d4e\u4ea7\u4e1a\u94fe",
        "global_required_terms": ["\u4e2d\u56fd\u4f4e\u7a7a\u7ecf\u6d4e", "\u4f4e\u7a7a\u7ecf\u6d4e"],
        "report_family": "industry_deep_report",
    }
    chapter = {
        "chapter_id": "ch_01",
        "chapter_title": "\u4f4e\u7a7a\u7ecf\u6d4e\u771f\u5b9e\u9700\u6c42\u4e0e\u5e02\u573a\u7a7a\u95f4",
        "core_question": "\u4f4e\u7a7a\u7ecf\u6d4e\u662f\u5426\u5df2\u6709\u53ef\u9a8c\u8bc1\u9700\u6c42\u548c\u5e02\u573a\u7a7a\u95f4\uff1f",
    }
    goal = {
        "goal_id": "H1_metric",
        "question": "\u5e02\u573a\u89c4\u6a21\u3001\u5e94\u7528\u573a\u666f\u548c\u653f\u7b56\u843d\u5730\u8bc1\u636e",
        "proof_role": "metric",
        "required_fields": ["metric", "value", "unit", "period", "source"],
        "lane_targets": ["official_data", "market_research"],
    }

    tasks = build_search_tasks_for_goal(chapter=chapter, goal=goal, research_plan=plan)

    assert tasks
    assert all("\u4f4e\u7a7a\u7ecf\u6d4e" in task["query"] for task in tasks)
    assert all(task["query_quality"]["has_topic_anchor"] for task in tasks)
    assert not any("wind energy" in task["query"].lower() for task in tasks)


def test_expanded_search_tasks_backfill_hypothesis_id_by_chapter_order():
    research_plan = {
        "query": "Low altitude economy report",
        "research_object": "Low altitude economy",
        "hypotheses": [
            {"hypothesis_id": "H1", "statement": "Demand is real"},
            {"hypothesis_id": "H2", "statement": "Supply is fragmented"},
        ],
    }
    report_blueprint = {
        "chapters": [
            {"chapter_id": "ch_01", "chapter_title": "Demand validation"},
            {"chapter_id": "ch_02", "chapter_title": "Supply pattern"},
        ]
    }

    expanded = expand_search_tasks_from_chapters(research_plan, report_blueprint)

    by_chapter = {}
    for task in expanded["search_tasks"]:
        by_chapter.setdefault(task["chapter_id"], set()).add(task.get("hypothesis_id"))
    assert by_chapter["ch_01"] == {"H1"}
    assert by_chapter["ch_02"] == {"H2"}
