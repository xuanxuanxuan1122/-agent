from rag_pipeline.agents.brain_agent import build_search_tasks_for_goal
from rag_pipeline.agents.evidence_merger import (
    _build_evidence as build_merged_evidence_for_test,
    task_acceptance_filter as merger_task_acceptance_filter,
)
from rag_pipeline.agents.web_analysis_agent import task_acceptance_filter


SOLID_STATE_BATTERY_CN = "\u4e2d\u56fd\u56fa\u6001\u7535\u6c60"
CHINA_CN = "\u4e2d\u56fd"
BEIJING_AI_MARKET = "2025\u5e74\u5317\u4eac\u4eba\u5de5\u667a\u80fd\u6838\u5fc3\u4ea7\u4e1a\u89c4\u6a21\u8fbe4500\u4ebf\u5143"
MARKET_SIZE_CN = "\u5e02\u573a\u89c4\u6a21"


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


def test_chinese_search_task_uses_localized_terms_and_quality_summary():
    chapter = {
        "chapter_id": "ch_01",
        "chapter_title": "低空经济产业链全景与商业化成熟度",
        "core_question": "低空经济产业链哪些环节具备商业化条件？",
        "required_evidence_mix": ["official_data", "market_research"],
    }
    goal = {
        "goal_id": "H1_technology",
        "requirement_id": "H1_technology",
        "question": "补齐低空经济产业链技术、产品、标准和能力约束证据",
        "proof_role": "technology_product",
        "required_fields": ["capability", "constraint", "source_ref"],
        "lane_targets": ["technology_product", "official_data"],
    }
    research_plan = {
        "query": "中国低空经济产业链商业化机会与风险分析（2026）",
        "research_object": "中国低空经济产业链",
        "global_required_terms": ["中国低空经济产业链", "低空经济"],
        "report_family": "industry_deep_report",
    }

    task = build_search_tasks_for_goal(chapter=chapter, goal=goal, research_plan=research_plan)[0]

    query = task["query"].lower()
    assert "中国低空经济产业链" in task["query"]
    assert "product" not in query
    assert "source" not in query
    assert task["query_quality"]["has_topic_anchor"] is True
    assert task["query_quality"]["has_placeholder_terms"] is False
    assert task["query_quality"]["score"] >= 0.75


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


def test_merger_task_filter_rejects_result_missing_explicit_topic_anchor():
    task = {
        "task_id": "task-solid-state",
        "query": f"{SOLID_STATE_BATTERY_CN} policy market size",
        "topic_anchor_terms": [SOLID_STATE_BATTERY_CN, CHINA_CN],
        "must_have_terms": [MARKET_SIZE_CN],
        "source_priority": ["official_data"],
        "proof_role": "metric",
    }
    item = {
        "content": BEIJING_AI_MARKET,
        "metric": MARKET_SIZE_CN,
        "value": "4500\u4ebf\u5143",
        "source_level": "A",
        "source": {
            "title": "\u5317\u4eac\u4eba\u5de5\u667a\u80fd\u4ea7\u4e1a\u53d1\u5c55",
            "url": "https://www.ncsti.gov.cn/example",
            "source_type": "official",
        },
    }

    result = merger_task_acceptance_filter(item, task)

    assert result["accepted"] is False
    assert result["reason"] == "topic_anchor_missing"
    assert result["role_hint"] == "rejected"


def test_merger_task_filter_accepts_domestic_solid_state_policy_anchor_variants():
    task = {
        "task_id": "task-solid-state-policy",
        "query": f"{SOLID_STATE_BATTERY_CN} policy",
        "topic_anchor_terms": [SOLID_STATE_BATTERY_CN, CHINA_CN],
        "must_have_terms": ["\u653f\u7b56"],
        "source_priority": ["official_data"],
        "proof_role": "source_check",
    }
    item = {
        "content": "\u6211\u56fd\u56fa\u6001\u7535\u6c60\u4ea7\u4e1a\u653f\u7b56\u6301\u7eed\u63a8\u52a8\u6280\u672f\u9a8c\u8bc1\u548c\u793a\u8303\u5e94\u7528\u3002",
        "metric": "\u653f\u7b56",
        "value": "",
        "source_level": "A",
        "source": {
            "title": "\u56fa\u6001\u7535\u6c60\u4ea7\u4e1a\u653f\u7b56",
            "url": "https://www.gov.cn/example",
            "source_type": "official",
        },
    }

    result = merger_task_acceptance_filter(item, task)

    assert result["accepted"] is True
    assert result["reason"] in {"chapter_or_report_relevance_pass", "task_relevance_pass"}


def test_build_evidence_uses_repair_topic_terms_and_keeps_market_size_with_growth_dimension():
    item = {
        "task_id": "dynamic_iqs_001",
        "requirement_id": "ch_01_metric",
        "proof_role": "metric",
        "evidence_type": "data",
        "metric": MARKET_SIZE_CN,
        "search_task": {
            "task_id": "dynamic_iqs_001",
            "requirement_id": "ch_01_metric",
            "proof_role": "metric",
            "evidence_type": "data",
            "must_have_terms": [MARKET_SIZE_CN],
            "source_priority": ["market_research"],
            "topic_terms": [
                "\u4e2d\u56fd\u4f4e\u7a7a\u7ecf\u6d4e\u4ea7\u4e1a\u94fe\u662f\u5426\u5b58\u5728\u771f\u5b9e\u9700\u6c42\u548c\u53ef\u9a8c\u8bc1\u5e02\u573a\u7a7a\u95f4"
            ],
        },
    }
    source = {
        "title": "\u4f4e\u7a7a\u7ecf\u6d4e\u76842025\uff1a\u5546\u4e1a\u5316\u8fdb\u7a0b\u63d0\u901f\u9614\u6b65\u8fc8\u5411\u4e07\u4ebf\u65b0\u84dd\u6d77",
        "url": "https://stcn.com/article/detail/3559095.html",
        "source_type": "research",
    }
    content = (
        "2025\u5e74\u4e2d\u56fd\u4f4e\u7a7a\u7ecf\u6d4e\u5e02\u573a\u89c4\u6a21\u8fbe5615\u4ebf\u5143\uff0c"
        "2029\u5e74\u5c06\u589e\u81f313904\u4ebf\u5143\uff0c2035\u5e74\u6709\u671b\u7a81\u78343.5\u4e07\u4ebf\u5143\u3002"
    )

    evidence = build_merged_evidence_for_test(
        raw_id="001",
        item=item,
        content=content,
        dimension="\u5e02\u573a\u89c4\u6a21 \u9700\u6c42\u589e\u901f",
        metric=MARKET_SIZE_CN,
        value="00",
        source=source,
        current_date=__import__("datetime").datetime(2026, 6, 24),
        research_plan={},
    )

    assert evidence["topic_anchor_terms"] == item["search_task"]["topic_terms"]
    assert evidence["task_accepted"] is True
    assert evidence["task_acceptance_reason"] in {"task_relevance_pass", "chapter_or_report_relevance_pass"}
    assert evidence["metric_kind"] == "market_size"
    assert evidence["semantic_status"] != "rejected"


def test_build_evidence_inherits_plan_topic_anchor_and_rejects_off_topic_metric():
    item = {
        "task_id": "task-1",
        "requirement_id": "H1_metric",
        "proof_role": "metric",
        "source_level": "A",
    }
    source = {
        "title": "\u5317\u4eac\u4eba\u5de5\u667a\u80fd\u4ea7\u4e1a\u53d1\u5c55",
        "url": "https://www.ncsti.gov.cn/example",
        "source_type": "official",
    }
    research_plan = {
        "query": SOLID_STATE_BATTERY_CN,
        "research_object": SOLID_STATE_BATTERY_CN,
        "topic_anchor_terms": [SOLID_STATE_BATTERY_CN, CHINA_CN],
        "search_tasks": [
            {
                "task_id": "task-1",
                "requirement_id": "H1_metric",
                "proof_role": "metric",
                "must_have_terms": [MARKET_SIZE_CN],
                "source_priority": ["official_data"],
            }
        ],
    }

    evidence = build_merged_evidence_for_test(
        raw_id="001",
        item=item,
        content=BEIJING_AI_MARKET,
        dimension="\u5e02\u573a",
        metric=MARKET_SIZE_CN,
        value="4500\u4ebf\u5143",
        source=source,
        current_date=__import__("datetime").datetime(2026, 6, 16),
        research_plan=research_plan,
    )

    assert evidence["topic_anchor_terms"] == [SOLID_STATE_BATTERY_CN, CHINA_CN]
    assert evidence["task_accepted"] is False
    assert evidence["task_acceptance_reason"] == "topic_anchor_missing"
    assert evidence["semantic_status"] == "rejected"
