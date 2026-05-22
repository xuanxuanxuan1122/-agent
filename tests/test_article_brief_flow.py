import pytest

from rag_pipeline.agents.article_brief import build_article_brief
from rag_pipeline.agents.markdown_renderer import render_cover
from rag_pipeline.agents.pre_layout_agent import run_pre_layout_agent
from rag_pipeline.agents.research_planner import run_research_planner_agent
from rag_pipeline.flows.report.full_report import resolve_runtime_article_brief


def test_article_brief_parses_labeled_title_and_direction():
    raw = "主标题：《大模型破晓，视觉纪元开启》\n副标题：——2026年国产AI视频工具竞争格局与底层基础设施演进报告"

    brief = build_article_brief(raw_query=raw)

    assert brief["display_title"] == "大模型破晓，视觉纪元开启"
    assert brief["display_subtitle"] == "2026年国产AI视频工具竞争格局与底层基础设施演进报告"
    assert brief["direction"] == "2026年国产AI视频工具竞争格局与底层基础设施演进报告"
    assert brief["planning_query"] == "2026年国产AI视频工具竞争格局与底层基础设施演进报告"
    assert brief["planning_query_source"] == "direction"
    assert brief["direction_missing"] is False
    assert brief["parsed_from"] == "labeled_query"


def test_runtime_article_brief_prompts_with_defaults():
    answers = iter(["", "2026年国产AI视频工具竞争格局与基础设施演进报告"])

    brief = resolve_runtime_article_brief(
        raw_query="大模型破晓，视觉纪元开启",
        no_interactive=False,
        input_fn=lambda _prompt: next(answers),
    )

    assert brief["display_title"] == "大模型破晓，视觉纪元开启"
    assert brief["direction"] == "2026年国产AI视频工具竞争格局与基础设施演进报告"
    assert brief["interactive_confirmed"] is True


def test_article_brief_title_only_uses_title_as_planning_query():
    brief = build_article_brief(title="中国AI行业焦虑与机遇报告")

    assert brief["main_title"] == "中国AI行业焦虑与机遇报告"
    assert brief["direction"] == ""
    assert brief["display_title"] == "中国AI行业焦虑与机遇报告"
    assert brief["display_subtitle"] == ""
    assert brief["planning_query"] == brief["display_title"]
    assert brief["planning_query_source"] == "main_title"
    assert brief["direction_missing"] is True


def test_runtime_article_brief_non_interactive_accepts_title_only():
    brief = resolve_runtime_article_brief(
        raw_query="大模型破晓，视觉纪元开启",
        no_interactive=True,
    )

    assert brief["display_title"] == "大模型破晓，视觉纪元开启"
    assert brief["direction"] == ""
    assert brief["display_subtitle"] == ""
    assert brief["planning_query"] == "大模型破晓，视觉纪元开启"
    assert brief["planning_query_source"] == "main_title"
    assert brief["direction_missing"] is True
    assert brief["interactive_confirmed"] is False


def test_runtime_article_brief_interactive_accepts_empty_direction():
    answers = iter(["", ""])

    brief = resolve_runtime_article_brief(
        raw_query="大模型破晓，视觉纪元开启",
        no_interactive=False,
        input_fn=lambda _prompt: next(answers),
    )

    assert brief["display_title"] == "大模型破晓，视觉纪元开启"
    assert brief["direction"] == ""
    assert brief["display_subtitle"] == ""
    assert brief["planning_query"] == "大模型破晓，视觉纪元开启"
    assert brief["direction_missing"] is True
    assert brief["interactive_confirmed"] is True


def test_runtime_article_brief_non_interactive_requires_title():
    with pytest.raises(RuntimeError, match="title"):
        resolve_runtime_article_brief(
            direction="2026年国产AI视频工具竞争格局与基础设施演进报告",
            no_interactive=True,
        )


def test_research_plan_and_blueprint_preserve_article_brief(monkeypatch):
    monkeypatch.setenv("BRAIN_ENABLE_LLM_RESEARCH_PLANNER", "0")
    brief = build_article_brief(
        title="大模型破晓，视觉纪元开启",
        direction="2026年国产AI视频工具竞争格局与底层基础设施演进报告",
    )

    plan = run_research_planner_agent(query=brief["planning_query"], article_brief=brief)
    blueprint = run_pre_layout_agent(query=brief["planning_query"], research_plan=plan)

    assert plan["query"] == "2026年国产AI视频工具竞争格局与底层基础设施演进报告"
    assert plan["planning_query"] == plan["query"]
    assert plan["research_object"] == "国产AI视频工具"
    assert plan["report_title"] == "大模型破晓，视觉纪元开启"
    assert plan["report_subtitle"] == "2026年国产AI视频工具竞争格局与底层基础设施演进报告"
    assert all("国产AI视频工具" in str(task.get("query") or "") for task in plan["search_tasks"])
    assert not any("人工智能行业" in str(chapter.get("chapter_title") or "") for chapter in plan["chapters"])
    assert blueprint["report_title"] == "大模型破晓，视觉纪元开启"
    assert blueprint["report_subtitle"] == "2026年国产AI视频工具竞争格局与底层基础设施演进报告"
    assert blueprint["article_brief"]["planning_query"] == plan["query"]


def test_render_cover_uses_display_title_and_subtitle():
    markdown = render_cover(
        "2026年国产AI视频工具竞争格局与底层基础设施演进报告",
        {
            "report_title": "大模型破晓，视觉纪元开启",
            "report_subtitle": "2026年国产AI视频工具竞争格局与底层基础设施演进报告",
            "research_object": "国产AI视频工具",
        },
    )

    assert markdown.splitlines()[0] == "# 大模型破晓，视觉纪元开启"
    assert "——2026年国产AI视频工具竞争格局与底层基础设施演进报告" in markdown


def test_title_only_research_plan_and_blueprint_infer_directions(monkeypatch):
    monkeypatch.setenv("BRAIN_ENABLE_LLM_RESEARCH_PLANNER", "0")
    brief = build_article_brief(title="中国AI行业焦虑与机遇报告")

    plan = run_research_planner_agent(query=brief["planning_query"], article_brief=brief)
    blueprint = run_pre_layout_agent(query=brief["planning_query"], research_plan=plan)

    assert plan["query"] == "中国AI行业焦虑与机遇报告"
    assert plan["planning_query"] == plan["query"]
    assert plan["report_title"] == "中国AI行业焦虑与机遇报告"
    assert plan["report_subtitle"] == ""
    assert plan["article_brief"]["direction_missing"] is True
    assert plan["quality_rules"]["infer_directions_from_title_when_direction_missing"] is True
    assert plan["chapters"]
    assert plan["search_tasks"]
    assert blueprint["report_title"] == "中国AI行业焦虑与机遇报告"
    assert blueprint["report_subtitle"] == ""
    assert blueprint["article_brief"]["planning_query"] == plan["query"]


def test_render_cover_omits_subtitle_when_title_only():
    markdown = render_cover(
        "中国AI行业焦虑与机遇报告",
        {
            "report_title": "中国AI行业焦虑与机遇报告",
            "report_subtitle": "",
            "research_object": "中国人工智能行业",
        },
    )

    lines = markdown.splitlines()
    assert lines[0] == "# 中国AI行业焦虑与机遇报告"
    assert not any(line.startswith("——") for line in lines)
