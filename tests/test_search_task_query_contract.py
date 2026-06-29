from rag_pipeline.agents.brain_agent import build_search_tasks_for_goal


def _plan():
    return {
        "query": "\u4e2d\u56fd\u4f4e\u7a7a\u7ecf\u6d4e\u5546\u4e1a\u5316\u843d\u5730\u673a\u4f1a\u4e0e\u98ce\u9669",
        "research_object": "\u4e2d\u56fd\u4f4e\u7a7a\u7ecf\u6d4e",
        "global_required_terms": ["\u4e2d\u56fd\u4f4e\u7a7a\u7ecf\u6d4e"],
        "report_family": "industry_deep_report",
    }


def _chapter(chapter_id, title, question):
    return {
        "chapter_id": chapter_id,
        "chapter_title": title,
        "core_question": question,
        "chapter_question": question,
        "required_evidence_mix": ["market_research", "customer_case"],
    }


def _goal(goal_id, question, proof_role="support", terms=None):
    return {
        "goal_id": goal_id,
        "requirement_id": goal_id,
        "question": question,
        "proof_role": proof_role,
        "must_have_terms": terms or [],
        "required_fields": ["fact", "source_ref"],
    }


def test_chapter_aware_search_query_keeps_chapter_focus_terms():
    chapter = _chapter(
        "ch_02",
        "\u7ade\u4e89\u683c\u5c40\u7531\u54ea\u4e9b\u73a9\u5bb6\u3001\u6210\u672c\u548c\u6e20\u9053\u53d8\u91cf\u51b3\u5b9a",
        "\u4f4e\u7a7a\u7ecf\u6d4e\u7684\u73a9\u5bb6\u3001\u6210\u672c\u7ed3\u6784\u548c\u6e20\u9053\u80fd\u529b\u5982\u4f55\u51b3\u5b9a\u7ade\u4e89\u683c\u5c40",
    )
    goal = _goal(
        "H2_support",
        "\u627e\u5230\u4f4e\u7a7a\u7ecf\u6d4e\u73a9\u5bb6\u3001\u6210\u672c\u3001\u6e20\u9053\u7684\u8bc1\u636e",
        terms=[
            "\u73a9\u5bb6",
            "\u6210\u672c",
            "\u6e20\u9053",
            "\u7ade\u4e89\u683c\u5c40",
        ],
    )

    tasks = build_search_tasks_for_goal(chapter=chapter, goal=goal, research_plan=_plan())

    assert tasks
    for task in tasks:
        query = task["query"]
        assert "\u4f4e\u7a7a\u7ecf\u6d4e" in query
        assert any(term in query for term in ["\u73a9\u5bb6", "\u6210\u672c", "\u6e20\u9053", "\u7ade\u4e89\u683c\u5c40"])
        assert task.get("chapter_focus_status") in {"ok", "repaired"}
        assert task.get("chapter_focus_terms")


def test_search_queries_differ_for_different_chapter_focuses():
    demand_chapter = _chapter(
        "ch_01",
        "\u771f\u5b9e\u9700\u6c42\u548c\u53ef\u9a8c\u8bc1\u5e02\u573a\u7a7a\u95f4",
        "\u4f4e\u7a7a\u7ecf\u6d4e\u662f\u5426\u6709\u771f\u5b9e\u9700\u6c42\u548c\u5e02\u573a\u89c4\u6a21",
    )
    risk_chapter = _chapter(
        "ch_04",
        "\u6280\u672f\u3001\u4f9b\u5e94\u3001\u76d1\u7ba1\u6216\u66ff\u4ee3\u7ea6\u675f",
        "\u76d1\u7ba1\u3001\u6280\u672f\u6210\u719f\u5ea6\u548c\u66ff\u4ee3\u65b9\u6848\u5982\u4f55\u6539\u53d8\u673a\u4f1a\u6392\u5e8f",
    )
    demand_goal = _goal("H1_support", "\u627e\u5230\u9700\u6c42\u548c\u5e02\u573a\u7a7a\u95f4\u8bc1\u636e", terms=["\u9700\u6c42", "\u5e02\u573a\u89c4\u6a21"])
    risk_goal = _goal("H4_counter", "\u627e\u5230\u76d1\u7ba1\u3001\u6280\u672f\u548c\u66ff\u4ee3\u7ea6\u675f\u8bc1\u636e", proof_role="counter", terms=["\u76d1\u7ba1", "\u6280\u672f", "\u66ff\u4ee3\u7ea6\u675f"])

    demand_queries = {task["query"] for task in build_search_tasks_for_goal(chapter=demand_chapter, goal=demand_goal, research_plan=_plan())}
    risk_queries = {task["query"] for task in build_search_tasks_for_goal(chapter=risk_chapter, goal=risk_goal, research_plan=_plan())}

    assert demand_queries
    assert risk_queries
    assert demand_queries.isdisjoint(risk_queries)
    assert all(any(term in query for term in ["\u9700\u6c42", "\u5e02\u573a\u89c4\u6a21"]) for query in demand_queries)
    assert all(any(term in query for term in ["\u76d1\u7ba1", "\u6280\u672f", "\u66ff\u4ee3\u7ea6\u675f"]) for query in risk_queries)
