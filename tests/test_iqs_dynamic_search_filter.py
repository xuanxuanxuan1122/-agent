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
def test_search_task_carries_requirement_contract_fields():
    chapter = {
        "chapter_id": "ch_01",
        "chapter_title": "AI Agent workflow demand",
        "core_question": "Is workflow deployment demand real?",
        "required_evidence_mix": ["customer_case"],
    }
    goal = {
        "goal_id": "H1_case",
        "requirement_id": "H1_case",
        "hypothesis_id": "H1",
        "question": "Find enterprise AI Agent customer deployment cases.",
        "proof_role": "case",
        "required_fields": ["company", "use_case", "deployment_scope", "source_ref"],
        "claim_strength_ceiling": "directional",
    }
    research_plan = {"query": "AI Agent workflow adoption"}

    task = build_search_tasks_for_goal(chapter=chapter, goal=goal, research_plan=research_plan)[0]

    assert task["requirement_id"] == "H1_case"
    assert task["hypothesis_id"] == "H1"
    assert task["required_fields"] == ["company", "use_case", "deployment_scope", "source_ref"]
    assert task["claim_strength_ceiling"] == "directional"


def test_search_task_query_uses_required_fields_and_source_contract_terms():
    chapter = {
        "chapter_id": "ch_01",
        "chapter_title": "AI Agent enterprise deployment demand",
        "core_question": "Is enterprise AI Agent deployment demand real?",
        "required_evidence_mix": ["official_data", "market_research"],
    }
    goal = {
        "goal_id": "H1_metric",
        "requirement_id": "H1_metric",
        "question": "Find 2025 enterprise AI Agent adoption metrics with value, unit, period, and source.",
        "proof_role": "metric",
        "required_fields": ["metric", "value", "unit", "period", "source_ref"],
        "lane_targets": ["official_data", "market_research"],
    }
    research_plan = {
        "query": "AI Agent enterprise deployment market report",
        "research_object": "AI Agent enterprise deployment",
        "global_required_terms": ["enterprise AI Agent"],
        "report_family": "industry_deep_report",
    }

    task = build_search_tasks_for_goal(chapter=chapter, goal=goal, research_plan=research_plan)[0]

    query = task["query"].lower()
    contract = task["query_contract"]
    assert "ai agent" in query
    assert any(term in query for term in ["metric", "value", "unit", "period", "source"])
    assert any(term in query for term in ["official", "report", "research"])
    assert contract["requirement_id"] == "H1_metric"
    assert contract["required_fields"] == ["metric", "value", "unit", "period", "source_ref"]


def test_generated_evidence_goals_default_requirement_id_to_goal_id():
    from rag_pipeline.agents.brain_agent import build_evidence_goals_for_chapter

    chapter = {
        "chapter_id": "ch_01",
        "chapter_title": "AI Agent workflow demand",
        "core_question": "Is workflow deployment demand real?",
        "required_evidence_roles": ["metric", "case"],
    }

    goals = build_evidence_goals_for_chapter(chapter, {"query": "AI Agent workflow adoption"})

    assert goals
    assert all(goal.get("requirement_id") == goal.get("goal_id") for goal in goals)
