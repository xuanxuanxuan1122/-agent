from rag_pipeline.agents.brain_agent import (
    coverage_units_from_state,
    evaluate_coverage_fallback,
    expand_search_tasks_from_chapters,
)
from rag_pipeline.agents.dynamic_search_schema import normalize_research_plan
from rag_pipeline.agents.evidence_binder import bind_evidence_to_chapters
from rag_pipeline.agents.pre_layout_agent import run_pre_layout_agent
from rag_pipeline.agents.problem_framing_agent import apply_problem_framing, run_problem_framing_agent
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


def test_problem_framing_uses_academic_template_for_broad_discipline_topics():
    framing = run_problem_framing_agent(query="\u4f1a\u8ba1\u5b66")
    text = " ".join(
        [str(framing.get("core_question") or "")]
        + [str(item.get("statement") or "") for item in framing.get("hypotheses") or []]
        + [
            str(term)
            for item in framing.get("hypotheses") or []
            for terms in (item.get("evidence_bundle") or {}).values()
            for term in terms
        ]
    )

    assert framing["research_object"] == "\u4f1a\u8ba1\u5b66"
    assert any(term in text for term in ["\u5b66\u79d1", "\u4e13\u4e1a", "\u8bfe\u7a0b"])
    assert any(term in text for term in ["\u5c97\u4f4d", "\u804c\u4e1a", "\u4eba\u624d"])
    for forbidden in [
        "\u5e02\u573a\u7a7a\u95f4",
        "\u6982\u5ff5\u70ed\u5ea6",
        "\u5546\u4e1a\u5316\u8bc1\u636e",
        "\u5ba2\u6237",
        "\u8ba2\u5355",
        "\u73a9\u5bb6",
    ]:
        assert forbidden not in text


def test_academic_topic_search_tasks_do_not_use_commercial_query_terms():
    query = "\u4f1a\u8ba1\u5b66"
    framing = run_problem_framing_agent(query=query)
    plan = apply_problem_framing({"query": query, "research_object": query, "chapters": []}, framing)
    blueprint = run_pre_layout_agent(query=query, research_plan=plan)
    expanded = expand_search_tasks_from_chapters(plan, blueprint)
    text = " ".join(
        str(value or "")
        for task in expanded.get("search_tasks") or []
        for value in [task.get("query"), task.get("evidence_goal")]
    )

    assert expanded.get("search_tasks")
    assert any(term in text for term in ["\u5c31\u4e1a", "\u5c97\u4f4d", "\u8bfe\u7a0b", "\u8bc1\u4e66"])
    for forbidden in [
        "\u5ba2\u6237",
        "\u8ba2\u5355",
        "\u91c7\u8d2d",
        "\u8d22\u62a5",
        "\u62db\u80a1\u4e66",
        "\u6295\u8d44\u8005\u5173\u7cfb",
        "\u4ea7\u54c1\u6807\u51c6",
    ]:
        assert forbidden not in text


def test_problem_framing_preserves_specific_planner_chapters():
    query = "\u4e2d\u56fd\u5de5\u4e1a\u8f6f\u4ef6MES/APS/PLM\u56fd\u4ea7\u66ff\u4ee3\uff1a\u79bb\u6563\u5236\u9020\u843d\u5730\u8def\u5f84\u3001\u91c7\u8d2d\u51b3\u7b56\u94fe\u4e0e\u5b9e\u65bd\u5931\u8d25\u98ce\u9669\u7814\u7a76"
    plan = {
        "query": query,
        "research_object": "\u4e2d\u56fd\u5de5\u4e1a\u8f6f\u4ef6MES/APS/PLM\u56fd\u4ea7\u66ff\u4ee3",
        "chapters": [
            {
                "chapter_id": "ch_01",
                "chapter_title": "MES/APS/PLM\u5728\u79bb\u6563\u5236\u9020\u7684\u5e94\u7528\u73b0\u72b6\u4e0e\u66ff\u4ee3\u9700\u6c42",
                "core_question": "\u79bb\u6563\u5236\u9020\u4f01\u4e1a\u7684\u771f\u5b9e\u66ff\u4ee3\u9700\u6c42\u662f\u4ec0\u4e48\uff1f",
            },
            {
                "chapter_id": "ch_02",
                "chapter_title": "\u79bb\u6563\u5236\u9020\u843d\u5730\u8def\u5f84\u4e0e\u5b9e\u65bd\u9636\u6bb5\u5206\u6790",
                "core_question": "\u4e0d\u540c\u89c4\u6a21\u5236\u9020\u4f01\u4e1a\u5982\u4f55\u5206\u9636\u6bb5\u843d\u5730\uff1f",
            },
            {
                "chapter_id": "ch_03",
                "chapter_title": "\u91c7\u8d2d\u51b3\u7b56\u94fe\uff1a\u5173\u952e\u89d2\u8272\u3001\u6d41\u7a0b\u4e0e\u5f71\u54cd\u56e0\u7d20",
                "core_question": "\u91c7\u8d2d\u51b3\u7b56\u7531\u54ea\u4e9b\u89d2\u8272\u4e3b\u5bfc\uff1f",
            },
            {
                "chapter_id": "ch_04",
                "chapter_title": "\u5b9e\u65bd\u5931\u8d25\u98ce\u9669\u8bc6\u522b\u4e0e\u5f52\u56e0\u5206\u6790",
                "core_question": "\u5931\u8d25\u9879\u76ee\u7684\u6839\u672c\u539f\u56e0\u662f\u4ec0\u4e48\uff1f",
            },
        ],
        "evidence_goals": [
            {
                "goal_id": "goal_purchase_chain",
                "chapter_id": "ch_03",
                "question": "\u91c7\u8d2d\u51b3\u7b56\u94fe\u4e2dIT\u3001\u751f\u4ea7\u3001\u8d22\u52a1\u4e0e\u7ba1\u7406\u5c42\u7684\u89d2\u8272\u5206\u5de5",
                "proof_role": "case",
            }
        ],
        "search_tasks": [
            {
                "task_id": "hypothesis_H3_case",
                "chapter_id": "ch_01",
                "dimension_id": "ch_01",
                "query": "\u5de5\u4e1a\u8f6f\u4ef6 \u91c7\u8d2d\u51b3\u7b56\u94fe \u5ba2\u6237\u6848\u4f8b",
                "proof_role": "case",
            }
        ],
    }
    framing = run_problem_framing_agent(query=query)

    merged = apply_problem_framing(plan, framing)

    titles = [chapter["chapter_title"] for chapter in merged["chapters"]]
    assert "\u91c7\u8d2d\u51b3\u7b56\u94fe\uff1a\u5173\u952e\u89d2\u8272\u3001\u6d41\u7a0b\u4e0e\u5f71\u54cd\u56e0\u7d20" in titles
    assert "\u5b9e\u65bd\u5931\u8d25\u98ce\u9669\u8bc6\u522b\u4e0e\u5f52\u56e0\u5206\u6790" in titles
    assert merged["evidence_goals"][0]["chapter_id"] == "ch_03"
    assert merged["search_tasks"][0]["chapter_id"] == "ch_03"
    assert merged["search_tasks"][0]["dimension_id"] == "ch_03"
    assert not any(
        str(item.get("dimension_id") or "").startswith("hypothesis")
        for item in merged.get("hypotheses") or []
    )
    assert merged["quality_rules"]["chapters_come_from_hypotheses"] is False
    assert not any(
        item.get("reason") == "replaced_by_problem_framing_hypotheses"
        for item in merged.get("dropped_template_sections") or []
    )


def test_preserved_planner_search_tasks_are_compacted_when_lane_terms_leak():
    query = "\u53bf\u57df\u65b0\u80fd\u6e90\u5546\u7528\u8f66\u5145\u6362\u7535\u7f51\u7edc\uff1a\u91cd\u5361\u7269\u6d41\u573a\u666f\u3001\u8d44\u4ea7\u5229\u7528\u7387\u4e0e\u8fd0\u8425\u5546\u76c8\u5229\u6a21\u578b\u7814\u7a76"
    statement = "\u5728\u91cd\u5361\u9ad8\u9891\u7269\u6d41\u7ebf\u8def\u4e0a\uff0c\u6362\u7535\u6a21\u5f0f\u8d44\u4ea7\u5229\u7528\u7387\u9ad8\u4e8e\u5145\u7535\u6869\uff0c\u53ef\u5b9e\u73b0\u66f4\u4f18\u76c8\u5229"
    plan = {
        "query": query,
        "chapters": [
            {
                "chapter_id": "ch_01",
                "chapter_title": "\u53bf\u57df\u65b0\u80fd\u6e90\u91cd\u5361\u7269\u6d41\u573a\u666f\u4e0e\u5145\u6362\u7535\u9700\u6c42\u5206\u6790",
                "core_question": "\u91cd\u5361\u7269\u6d41\u573a\u666f\u5bf9\u5145\u6362\u7535\u7f51\u7edc\u6709\u4ec0\u4e48\u9700\u6c42\uff1f",
            },
            {
                "chapter_id": "ch_02",
                "chapter_title": "\u8fd0\u8425\u5546\u76c8\u5229\u6a21\u578b\u6784\u5efa\u4e0e\u654f\u611f\u6027\u5206\u6790",
                "core_question": "\u5145\u6362\u7535\u7f51\u7edc\u7684\u76c8\u5229\u95e8\u69db\u662f\u4ec0\u4e48\uff1f",
            },
            {
                "chapter_id": "ch_03",
                "chapter_title": "\u5178\u578b\u6848\u4f8b\u4e0e\u6295\u8d44\u56de\u62a5\u6d4b\u7b97",
                "core_question": "\u54ea\u4e9b\u53bf\u57df\u9879\u76ee\u80fd\u8bc1\u660e\u6295\u8d44\u56de\u62a5\u6a21\u578b\uff1f",
            }
        ],
        "search_tasks": [
            {
                "task_id": "task-1",
                "chapter_id": "ch_01",
                "dimension_id": "ch_01",
                "hypothesis_id": "H1",
                "proof_role": "metric",
                "query": f"{query} {statement} official data market research evidence",
            }
        ],
    }
    framing = {
        "hypotheses": [
            {
                "hypothesis_id": "H1",
                "statement": statement,
                "must_prove": ["\u8d44\u4ea7\u5229\u7528\u7387", "\u5355\u6b21\u8865\u80fd\u6210\u672c", "\u8fd0\u8425\u5546\u76c8\u5229"],
            }
        ]
    }

    merged = apply_problem_framing(plan, framing)
    compact_task = merged["search_tasks"][0]
    compact_query = compact_task["query"]

    assert compact_task["task_id"] == "task-1"
    assert compact_task["query_compacted_from_hypothesis"] is True
    assert "official data" not in compact_query
    assert "market research" not in compact_query
    assert statement not in compact_query
    assert "\u8d44\u4ea7\u5229\u7528\u7387" in compact_query


def test_search_task_compaction_removes_repeated_topic_anchor_and_chinese_lane_terms():
    query = "\u53bf\u57df\u65b0\u80fd\u6e90\u5546\u7528\u8f66\u5145\u6362\u7535\u7f51\u7edc\uff1a\u91cd\u5361\u7269\u6d41\u573a\u666f\u3001\u8d44\u4ea7\u5229\u7528\u7387\u4e0e\u8fd0\u8425\u5546\u76c8\u5229\u6a21\u578b\u7814\u7a76"
    statement = "\u6362\u7535\u6a21\u5f0f\u5728\u91cd\u5361\u7269\u6d41\u573a\u666f\u4e2d\u6bd4\u5145\u7535\u6a21\u5f0f\u5177\u6709\u66f4\u9ad8\u7684\u8d44\u4ea7\u5229\u7528\u7387\u548c\u66f4\u5feb\u7684\u6295\u8d44\u56de\u62a5"
    plan = {
        "query": query,
        "chapters": [
            {"chapter_id": "ch_01", "chapter_title": "\u91cd\u5361\u7269\u6d41\u573a\u666f\u4e0e\u5145\u6362\u7535\u9700\u6c42", "core_question": "\u9700\u6c42\u5982\u4f55\uff1f"},
            {"chapter_id": "ch_02", "chapter_title": "\u8d44\u4ea7\u5229\u7528\u7387\u5f71\u54cd\u56e0\u7d20", "core_question": "\u5229\u7528\u7387\u7531\u4ec0\u4e48\u51b3\u5b9a\uff1f"},
            {"chapter_id": "ch_03", "chapter_title": "\u8fd0\u8425\u5546\u76c8\u5229\u6a21\u578b", "core_question": "\u5982\u4f55\u76c8\u5229\uff1f"},
        ],
        "search_tasks": [
            {
                "task_id": "counter-1",
                "chapter_id": "ch_01",
                "hypothesis_id": "H1",
                "proof_role": "counter",
                "must_have_terms": ["\u91cd\u5361\u7269\u6d41\u573a\u666f", "\u8d44\u4ea7\u5229\u7528\u7387", "\u8fd0\u8425\u5546\u76c8\u5229\u6a21\u578b"],
                "query": f"{query} \u53bf\u57df\u65b0\u80fd\u6e90\u5546\u7528\u8f66\u5145\u6362\u7535\u7f51\u7edc \u91cd\u5361\u7269\u6d41\u573a\u666f \u8d44\u4ea7\u5229\u7528\u7387 \u8fd0\u8425\u5546\u76c8\u5229\u6a21\u578b \u53cd\u8bc1 \u98ce\u9669 \u5931\u8d25\u6848\u4f8b \u8d1f\u9762 \u66ff\u4ee3\u65b9\u6848 \u5ba2\u6237\u4e0d\u4e70\u8d26",
            }
        ],
    }
    framing = {"hypotheses": [{"hypothesis_id": "H1", "statement": statement, "must_disprove": ["\u5229\u7528\u7387\u4e0d\u8db3", "\u76c8\u5229\u56f0\u96be"]}]}

    merged = apply_problem_framing(plan, framing)
    compact_query = merged["search_tasks"][0]["query"]

    assert compact_query.count("\u53bf\u57df\u65b0\u80fd\u6e90\u5546\u7528\u8f66\u5145\u6362\u7535\u7f51\u7edc") <= 1
    assert "\u5ba2\u6237\u4e0d\u4e70\u8d26" not in compact_query
    assert "\u53cd\u8bc1 \u98ce\u9669 \u5931\u8d25\u6848\u4f8b" not in compact_query


def test_normalized_search_tasks_add_role_specific_query_hints():
    query = "\u53bf\u57df\u65b0\u80fd\u6e90\u5546\u7528\u8f66\u5145\u6362\u7535\u7f51\u7edc\uff1a\u91cd\u5361\u7269\u6d41\u573a\u666f\u3001\u8d44\u4ea7\u5229\u7528\u7387\u4e0e\u8fd0\u8425\u5546\u76c8\u5229\u6a21\u578b\u7814\u7a76"
    base_query = "\u53bf\u57df\u65b0\u80fd\u6e90\u5546\u7528\u8f66\u5145\u6362\u7535\u7f51\u7edc \u91cd\u5361\u7269\u6d41\u573a\u666f \u8d44\u4ea7\u5229\u7528\u7387 \u8fd0\u8425\u5546"
    plan = normalize_research_plan(
        {
            "query": query,
            "research_object": "\u53bf\u57df\u65b0\u80fd\u6e90\u5546\u7528\u8f66\u5145\u6362\u7535\u7f51\u7edc",
            "chapters": [
                {
                    "chapter_id": "ch_01",
                    "chapter_title": "\u91cd\u5361\u7269\u6d41\u573a\u666f\u4e0e\u5145\u6362\u7535\u9700\u6c42",
                    "core_question": "\u9700\u6c42\u5982\u4f55\uff1f",
                }
            ],
            "search_tasks": [
                {"task_id": "m", "chapter_id": "ch_01", "proof_role": "metric", "query": base_query},
                {"task_id": "c", "chapter_id": "ch_01", "proof_role": "case", "query": base_query},
                {"task_id": "r", "chapter_id": "ch_01", "proof_role": "counter", "query": base_query},
            ],
        },
        query=query,
    )
    by_role = {task["proof_role"]: task["query"] for task in plan["search_tasks"]}

    assert "\u6307\u6807" in by_role["metric"] or "\u6570\u636e" in by_role["metric"]
    assert "\u6848\u4f8b" in by_role["case"] or "\u9879\u76ee" in by_role["case"]
    assert "\u5931\u8d25" in by_role["counter"] or "\u98ce\u9669" in by_role["counter"]
    assert len({by_role["metric"], by_role["case"], by_role["counter"]}) == 3


def test_problem_framing_search_task_queries_are_field_oriented():
    query = "\u53bf\u57df\u65b0\u80fd\u6e90\u5546\u7528\u8f66\u5145\u6362\u7535\u7f51\u7edc\uff1a\u91cd\u5361\u7269\u6d41\u573a\u666f\u3001\u8d44\u4ea7\u5229\u7528\u7387\u4e0e\u8fd0\u8425\u5546\u76c8\u5229\u6a21\u578b\u7814\u7a76"
    framing = {
        "hypotheses": [
            {
                "hypothesis_id": "H1",
                "statement": "\u53bf\u57df\u91cd\u5361\u5145\u6362\u7535\u7f51\u7edc\u8d44\u4ea7\u5229\u7528\u7387\u4e0d\u8db3\u662f\u8fd0\u8425\u5546\u76c8\u5229\u56f0\u96be\u7684\u4e3b\u8981\u539f\u56e0",
                "must_prove": ["\u8d44\u4ea7\u5229\u7528\u7387", "\u5355\u7ad9\u6536\u5165", "\u7535\u4ef7\u6210\u672c"],
                "must_disprove": ["\u9ad8\u5229\u7528\u7387\u76c8\u5229\u9879\u76ee", "\u5229\u7528\u7387\u4e0d\u8db3\u4f46\u76c8\u5229\u6848\u4f8b"],
                "evidence_bundle": {
                    "metric": ["\u8d44\u4ea7\u5229\u7528\u7387", "\u5355\u7ad9\u6536\u5165", "\u5145\u7535\u91cf"],
                    "case": ["\u53bf\u57df\u91cd\u5361\u6362\u7535\u7ad9\u6848\u4f8b", "\u8fd0\u8425\u5546\u9879\u76ee"],
                    "counter": ["\u9ad8\u5229\u7528\u7387\u76c8\u5229\u9879\u76ee", "\u5931\u8d25\u9879\u76ee"],
                },
                "counter_evidence_required": True,
            }
        ]
    }

    plan = apply_problem_framing({"query": query, "research_object": query, "chapters": []}, framing)
    queries = [str(task.get("query") or "") for task in plan.get("search_tasks") or []]

    assert queries
    assert all(len(item) <= 96 for item in queries)
    assert all("official data" not in item and "market research" not in item for item in queries)
    assert all(
        "\u8d44\u4ea7\u5229\u7528\u7387\u4e0d\u8db3\u662f\u8fd0\u8425\u5546\u76c8\u5229\u56f0\u96be\u7684\u4e3b\u8981\u539f\u56e0" not in item
        for item in queries
    )
    assert any("\u8d44\u4ea7\u5229\u7528\u7387" in item and "\u5355\u7ad9\u6536\u5165" in item for item in queries)
    assert any("\u5931\u8d25\u9879\u76ee" in item or "\u76c8\u5229\u9879\u76ee" in item for item in queries)


def test_problem_framing_uses_query_aspects_when_planner_chapters_missing():
    query = "\u53bf\u57df\u65b0\u80fd\u6e90\u5546\u7528\u8f66\u5145\u6362\u7535\u7f51\u7edc\uff1a\u91cd\u5361\u7269\u6d41\u573a\u666f\u3001\u8d44\u4ea7\u5229\u7528\u7387\u4e0e\u8fd0\u8425\u5546\u76c8\u5229\u6a21\u578b\u7814\u7a76"
    framing = {
        "hypotheses": [
            {
                "hypothesis_id": "H1",
                "statement": "\u9700\u6c42\u662f\u5426\u771f\u5b9e\u5b58\u5728",
                "must_prove": ["\u9700\u6c42", "\u573a\u666f"],
            },
            {
                "hypothesis_id": "H2",
                "statement": "\u8d44\u4ea7\u5229\u7528\u7387\u662f\u76c8\u5229\u6a21\u578b\u7684\u5173\u952e\u53d8\u91cf",
                "must_prove": ["\u8d44\u4ea7\u5229\u7528\u7387", "\u76c8\u5229\u6a21\u578b"],
            },
        ]
    }

    plan = apply_problem_framing({"query": query, "research_object": query, "chapters": []}, framing)
    titles = [str(chapter.get("chapter_title") or "") for chapter in plan.get("chapters") or []]

    assert plan["quality_rules"]["chapter_source"] == "query_aspect_chapters"
    assert plan["quality_rules"]["chapters_come_from_hypotheses"] is False
    assert any("\u91cd\u5361\u7269\u6d41\u573a\u666f" in title for title in titles)
    assert any("\u8d44\u4ea7\u5229\u7528\u7387" in title for title in titles)
    assert any("\u8fd0\u8425\u5546\u76c8\u5229\u6a21\u578b" in title for title in titles)


def test_pre_layout_does_not_rewrite_specific_planner_chapters():
    query = "\u4e2d\u56fd\u5de5\u4e1a\u8f6f\u4ef6MES/APS/PLM\u56fd\u4ea7\u66ff\u4ee3\uff1a\u79bb\u6563\u5236\u9020\u843d\u5730\u8def\u5f84\u3001\u91c7\u8d2d\u51b3\u7b56\u94fe\u4e0e\u5b9e\u65bd\u5931\u8d25\u98ce\u9669\u7814\u7a76"
    blueprint = run_pre_layout_agent(
        query=query,
        research_plan={
            "query": query,
            "report_family": "industry_deep_report",
            "research_object": "\u5de5\u4e1a\u8f6f\u4ef6MES/APS/PLM\u56fd\u4ea7\u66ff\u4ee3",
            "quality_rules": {
                "chapters_come_from_hypotheses": False,
                "chapter_source": "llm_planner_with_problem_framing",
            },
            "chapters": [
                {
                    "chapter_id": "ch_01",
                    "chapter_title": "\u79bb\u6563\u5236\u9020\u9700\u6c42\u7279\u70b9\u4e0e\u9009\u578b\u6807\u51c6",
                    "core_question": "\u79bb\u6563\u5236\u9020\u5bf9MES/APS/PLM\u7684\u9700\u6c42\u548c\u9009\u578b\u6807\u51c6\u662f\u4ec0\u4e48\uff1f",
                },
                {
                    "chapter_id": "ch_02",
                    "chapter_title": "\u91c7\u8d2d\u51b3\u7b56\u94fe\u4e0e\u6d41\u7a0b",
                    "core_question": "\u91c7\u8d2d\u51b3\u7b56\u7531\u54ea\u4e9b\u89d2\u8272\u548c\u6d41\u7a0b\u51b3\u5b9a\uff1f",
                },
                {
                    "chapter_id": "ch_03",
                    "chapter_title": "\u5b9e\u65bd\u5931\u8d25\u98ce\u9669\u8bc6\u522b\u4e0e\u7ba1\u63a7",
                    "core_question": "\u5b9e\u65bd\u5931\u8d25\u4e3b\u8981\u6765\u81ea\u54ea\u4e9b\u7ec4\u7ec7\u548c\u6280\u672f\u56e0\u7d20\uff1f",
                },
            ],
        },
    )

    titles = [chapter["chapter_title"] for chapter in blueprint["chapters"]]
    assert "\u91c7\u8d2d\u51b3\u7b56\u94fe\u4e0e\u6d41\u7a0b" in titles
    assert "\u5b9e\u65bd\u5931\u8d25\u98ce\u9669\u8bc6\u522b\u4e0e\u7ba1\u63a7" in titles
    assert not any("\u653f\u7b56\u3001\u76d1\u7ba1\u6216\u5916\u90e8\u89c4\u5219" in title for title in titles)


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
