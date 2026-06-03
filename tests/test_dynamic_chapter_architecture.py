from rag_pipeline.agents.brain_agent import (
    coverage_units_from_state,
    evaluate_coverage_fallback,
    expand_search_tasks_from_chapters,
)
from rag_pipeline.agents.evidence_binder import bind_evidence_to_chapters
from rag_pipeline.agents.pre_layout_agent import run_pre_layout_agent
from rag_pipeline.agents.research_planner import run_research_planner_agent
from rag_pipeline.agents.markdown_renderer import render_chapter_package, render_final_reference_analysis


FORBIDDEN_EXACT_TITLES = {
    "市场规模与增速",
    "竞争格局",
    "政策与监管环境",
    "技术路线与产业链",
    "资本动态",
}

FORBIDDEN_MICRO_HEADINGS = [
    "章节判断",
    "关键事实速览",
    "证据深读",
    "本章结论",
    "进入综合决策章的变量",
    "原文事实",
]


def test_research_planner_builds_closed_loop_from_dynamic_hypotheses():
    query = "building material market outlook"
    plan = run_research_planner_agent(query=query)
    blueprint = run_pre_layout_agent(query=query, research_plan=plan)
    expanded = expand_search_tasks_from_chapters(plan, blueprint)

    assert plan["quality_rules"]["chapters_come_from_hypotheses"] is True
    assert plan["quality_rules"]["disable_fixed_fallback_templates"] is True
    assert not plan.get("legacy_planner_chapters")
    assert blueprint["quality_rules"]["chapter_source"] == "problem_framing_hypotheses"

    chapter_ids = {chapter["chapter_id"] for chapter in blueprint["chapters"]}
    task_chapter_ids = {task.get("chapter_id") for task in expanded["search_tasks"]}
    assert chapter_ids
    assert task_chapter_ids
    assert task_chapter_ids <= chapter_ids


def test_pre_layout_rewrites_legacy_titles_and_keeps_chapter_contract():
    blueprint = run_pre_layout_agent(
        query="建筑材料的行情怎么样",
        research_plan={
            "research_object": "建筑材料",
            "chapters": [
                {
                    "chapter_id": "ch_01",
                    "chapter_title": "市场规模与增速",
                    "core_question": "当前建材行情到底是整体下行还是结构分化",
                    "required_evidence_mix": ["official_data", "market_price", "counter_evidence"],
                }
            ],
        },
    )

    chapters = blueprint["chapters"]
    assert not any(ch["chapter_title"] in FORBIDDEN_EXACT_TITLES for ch in chapters)
    for chapter in chapters:
        assert chapter["core_question"]
        assert chapter["required_evidence_mix"]
        assert chapter["min_total_sources"] >= 4
        assert chapter["min_ab_sources"] >= 1


def test_search_tasks_are_expanded_from_chapters():
    blueprint = run_pre_layout_agent(
        query="建筑材料的行情怎么样",
        research_plan={
            "query": "建筑材料的行情怎么样",
            "research_object": "建筑材料",
            "chapters": [
                {
                    "chapter_id": "ch_01",
                    "chapter_title": "当前建材行情是整体下行还是结构分化？",
                    "core_question": "判断建材行业当前行情的主矛盾是什么",
                    "required_evidence_mix": ["official_data", "market_research", "counter_evidence"],
                    "min_total_sources": 6,
                    "min_ab_sources": 2,
                }
            ],
        },
    )
    plan = expand_search_tasks_from_chapters({"query": "建筑材料的行情怎么样"}, blueprint)
    chapter_ids = {chapter["chapter_id"] for chapter in blueprint["chapters"]}

    assert plan["search_tasks"]
    for task in plan["search_tasks"]:
        assert task["chapter_id"] in chapter_ids
        assert task["evidence_goal"]
        assert task["proof_role"] in {"metric", "support", "counter", "case", "source_check", "technology_product"}
        assert task["lane_targets"]


def test_coverage_is_evaluated_by_chapter():
    state = {
        "query": "建筑材料的行情怎么样",
        "report_blueprint": {
            "chapters": [
                {
                    "chapter_id": "ch_01",
                    "chapter_title": "当前建材行情是整体下行还是结构分化？",
                    "core_question": "判断建材行业当前行情的主矛盾是什么",
                    "required_evidence_mix": ["official_data", "market_research", "counter_evidence"],
                    "min_total_sources": 2,
                    "min_ab_sources": 1,
                    "min_counter_sources": 1,
                }
            ]
        },
    }
    units = coverage_units_from_state(state)
    evaluation = evaluate_coverage_fallback(
        original_query=state["query"],
        evidence_pool=[],
        coverage_units=units,
        loop_number=1,
        max_loops=5,
        prev_coverage_score=0,
        min_gain=0.1,
        previous_queries=[state["query"]],
        max_followup_queries=10,
    )

    assert set(evaluation["chapter_scores"].keys()) == {units[0]["unit_title"]}
    assert evaluation["follow_up_queries"]
    assert evaluation["follow_up_queries"][0]["chapter_id"] == "ch_01"


def test_renderer_uses_render_blocks_without_fixed_micro_headings():
    markdown = render_chapter_package(
        {
            "chapter_title": "水泥和玻纤的行情已经不是同一个周期",
            "sections": [
                {
                    "section_id": "ch_01_sec_01",
                    "section_title": "水泥和玻纤的行情已经不是同一个周期",
                    "render_blocks": [
                        {"type": "paragraph", "label": "", "text": "当前建筑材料行业不能用单一景气度概括。"},
                        {"type": "evidence_list", "label": "关键证据", "evidence_refs": ["E001", "E004"]},
                    ],
                    "evidence_refs": ["E001", "E004"],
                }
            ],
        },
        1,
    )

    for phrase in FORBIDDEN_MICRO_HEADINGS:
        assert phrase not in markdown
    assert "当前建筑材料行业不能用单一景气度概括。" in markdown
    assert "E001" not in markdown


def test_final_reference_analysis_uses_dynamic_report_logic_not_fixed_industry_chain(monkeypatch):
    monkeypatch.setenv("REPORT_RENDER_FINAL_REFERENCE_ANALYSIS", "true")
    lines = render_final_reference_analysis(
        {
            "chapter_syntheses": [
                {
                    "chapter_title": "用户预算收缩先改变采购节奏",
                    "chapter_question": "用户预算变化如何影响采购节奏",
                    "chapter_summary": {
                        "key_takeaway": "预算审批变慢会先改变采购节奏，再影响交付安排。",
                        "mechanisms": ["预算审批周期拉长后，采购批次会被拆小"],
                        "counter_evidence": ["若核心客户仍保持年度框架采购，结论需要降级"],
                        "next_actions": ["优先核验核心客户的年度采购框架"],
                        "what_to_verify_next": ["采购批次和付款周期"],
                    },
                }
            ]
        }
    )
    text = "\n".join(lines)

    fixed_chain_phrases = [
        "竞争格局决定",
        "政策和技术决定",
        "资本与交易信号",
        "规模和增速决定",
        "需求、供给、竞争、政策、技术和资本",
    ]
    for phrase in fixed_chain_phrases:
        assert phrase not in text
    assert "用户预算收缩先改变采购节奏" in text


def test_chapter_evidence_package_keeps_deep_inventory_and_thresholds():
    chapter = {
        "chapter_id": "ch_01",
        "chapter_title": "当前建材行情是整体下行还是结构分化？",
        "chapter_question": "判断建材行业当前行情的主矛盾是什么",
        "min_total_sources": 6,
        "min_ab_sources": 2,
        "min_counter_sources": 1,
        "required_evidence_mix": ["official_data", "market_research", "counter_evidence"],
    }
    items = []
    for index in range(6):
        items.append(
            {
                "evidence_id": f"E{index}",
                "ref": f"E{index}",
                "chapter_id": "ch_01",
                "chapter_title": chapter["chapter_title"],
                "dimension": chapter["chapter_title"],
                "evidence_goal": chapter["chapter_question"],
                "fact": f"建材行情证据 {index}",
                "source_level": "A" if index < 2 else "B",
                "evidence_role": "core" if index < 3 else "supporting",
                "allowed_use": "core_claim" if index < 3 else "supporting",
                "confidence": 0.9,
                "proof_role": "counter" if index == 5 else "metric",
            }
        )

    package = bind_evidence_to_chapters(items, [chapter], [])[0]
    assert package["source_count"] == 6
    assert package["ab_source_count"] >= package["min_ab_sources"]
    assert package["counter_source_count"] >= package["min_counter_sources"]
