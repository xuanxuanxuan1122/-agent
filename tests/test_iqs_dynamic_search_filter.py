from rag_pipeline.agents.brain_agent import build_search_tasks_for_goal
from rag_pipeline.agents.web_analysis_agent import task_acceptance_filter


def test_dynamic_chapter_search_terms_are_short_anchors():
    chapter = {
        "chapter_id": "ch_01",
        "chapter_title": "新能源汽车新型材料中，电池功能材料比轻量化/结构材料更具短期放量确定性？",
        "core_question": "新能源汽车新型材料中，电池功能材料比轻量化/结构材料更具短期放量确定性",
        "required_evidence_mix": ["official_data", "market_research", "counter_evidence"],
    }
    goal = {
        "goal_id": "ch_01_metric",
        "question": "新能源汽车新型材料中，电池功能材料比轻量化/结构材料更具短期放量确定性：用指标、时间、范围和单位回答本章核心问题",
        "proof_role": "metric",
    }
    research_plan = {"query": "现在新能源汽车的新型材料在市场的行情怎么样？"}

    task = build_search_tasks_for_goal(chapter=chapter, goal=goal, research_plan=research_plan)[0]

    assert "新能源汽车" in task["must_have_terms"]
    assert "新型材料" in task["must_have_terms"]
    assert all(len(term) <= 24 for term in task["must_have_terms"])
    assert not any("更具短期放量确定性" in term for term in task["must_have_terms"])
    assert "更具短期放量确定性" not in task["query"]


def test_iqs_filter_expands_legacy_sentence_must_terms():
    legacy_must = "新能源汽车新型材料中，电池功能材料比轻量化/结构材料更具短期放量确定性"
    task = {
        "query": "现在新能源汽车的新型材料在市场的行情怎么样？",
        "must_have_terms": [legacy_must],
        "source_priority": ["official", "research"],
    }
    item = {
        "title": "新能源汽车电池新型材料市场规模与订单变化",
        "snippet": "动力电池功能材料、轻量化结构材料的价格、产能和客户订单出现分化。",
        "summary": "2026年相关材料企业继续披露订单和产能扩张。",
        "url": "https://example.com/report",
    }

    result = task_acceptance_filter(item, {"search_task": task})

    assert result["accepted"] is True
    assert "新能源汽车" in result["matched_terms"]
    assert "新型材料" in result["matched_terms"]
