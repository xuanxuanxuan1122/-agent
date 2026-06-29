import pytest

from rag_pipeline.agents.article_brief import build_article_brief
from rag_pipeline.agents.markdown_renderer import render_cover
from rag_pipeline.agents.pre_layout_agent import run_pre_layout_agent
from rag_pipeline.agents.research_planner import run_research_planner_agent
from rag_pipeline.flows.report.full_report import resolve_runtime_article_brief
from rag_pipeline.search.memory import LLMCallError


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


def test_research_planner_marks_truncated_llm_fallback(monkeypatch):
    monkeypatch.setenv("BRAIN_ENABLE_LLM_RESEARCH_PLANNER", "1")
    monkeypatch.setattr(
        "rag_pipeline.agents.research_planner.llm_config_is_ready",
        lambda _config: True,
    )
    monkeypatch.setattr(
        "rag_pipeline.agents.research_planner.run_problem_framing_agent",
        lambda **_kwargs: {},
    )

    def fail_with_truncated_json(**_kwargs):
        raise LLMCallError(
            "LLM response is not valid JSON",
            diagnostic={
                "task": "planning",
                "model": "deepseek-v4-pro",
                "status": "failed",
                "error": "LLM response is not valid JSON",
                "finish_reason": "length",
                "max_output_tokens": 4096,
            },
        )

    monkeypatch.setattr(
        "rag_pipeline.agents.research_planner.call_openai_compatible_json",
        fail_with_truncated_json,
    )

    plan = run_research_planner_agent(query="中国低空经济产业链商业化机会与风险分析（2026）")

    assert plan["planning_status"] == "fallback_seed"
    assert plan["planner_llm_degraded"] is True
    assert plan["planner_degraded_reason"] == "json_truncated"
    assert plan["planner_finish_reason"] == "length"
    assert plan["allow_search"] is True
    assert plan["allow_full_report"] is False


def test_research_planner_recovers_from_truncation_with_compact_retry(monkeypatch):
    monkeypatch.setenv("BRAIN_ENABLE_LLM_RESEARCH_PLANNER", "1")
    monkeypatch.setenv("REPORT_PLANNING_COMPACT_RETRY", "1")
    monkeypatch.setattr(
        "rag_pipeline.agents.research_planner.llm_config_is_ready",
        lambda _config: True,
    )
    monkeypatch.setattr(
        "rag_pipeline.agents.research_planner.run_problem_framing_agent",
        lambda **_kwargs: {},
    )
    calls = {"count": 0}

    def first_fails_second_is_compact(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise LLMCallError(
                "LLM response is not valid JSON",
                diagnostic={
                    "task": "planning",
                    "model": "deepseek-v4-pro",
                    "status": "failed",
                    "error": "LLM response is not valid JSON",
                    "finish_reason": "length",
                },
            )
        assert kwargs["user_payload"]["compact_retry"] is True
        return {
            "payload": {
                "query": "中国低空经济产业链商业化机会与风险分析（2026）",
                "research_type": "industry_scan",
                "report_family": "industry_deep_report",
                "research_object": "中国低空经济产业链",
                "chapters": [
                    {
                        "chapter_id": "ch_01",
                        "chapter_title": "低空经济商业化成熟度如何判断？",
                        "core_question": "哪些环节已经具备规模化商业化条件？",
                    }
                ],
                "evidence_goals": [
                    {
                        "goal_id": "H1_metric",
                        "chapter_id": "ch_01",
                        "proof_role": "metric",
                        "required_fields": ["metric", "value", "unit", "period", "source_ref"],
                    }
                ],
                "global_required_terms": ["中国低空经济产业链", "低空经济"],
            },
            "llm_call": {"task": "planning", "model": "deepseek-v4-pro", "status": "ok"},
        }

    monkeypatch.setattr(
        "rag_pipeline.agents.research_planner.call_openai_compatible_json",
        first_fails_second_is_compact,
    )

    plan = run_research_planner_agent(query="中国低空经济产业链商业化机会与风险分析（2026）")

    assert calls["count"] == 2
    assert plan["planning_status"] == "repaired"
    assert plan["planner_llm_degraded"] is False
    assert plan["planner_degraded_reason"] == "compact_retry_after_json_truncated"
    assert plan["chapters"][0]["chapter_title"] == "低空经济商业化成熟度如何判断？"
    assert plan["allow_full_report"] is True


def test_research_planner_uses_compact_first_by_default(monkeypatch):
    monkeypatch.setenv("BRAIN_ENABLE_LLM_RESEARCH_PLANNER", "1")
    monkeypatch.delenv("REPORT_PLANNING_COMPACT_FIRST", raising=False)
    monkeypatch.setattr(
        "rag_pipeline.agents.research_planner.llm_config_is_ready",
        lambda _config: True,
    )
    monkeypatch.setattr(
        "rag_pipeline.agents.research_planner.run_problem_framing_agent",
        lambda **_kwargs: {},
    )
    calls = []

    def compact_first_success(**kwargs):
        calls.append(kwargs)
        assert kwargs["user_payload"]["compact_first"] is True
        assert "Do not include search_tasks" in kwargs["system_prompt"]
        return {
            "payload": {
                "query": "\u4e2d\u56fd\u4f4e\u7a7a\u7ecf\u6d4e\u4ea7\u4e1a\u94fe\u5546\u4e1a\u5316\u673a\u4f1a\u4e0e\u98ce\u9669\u5206\u6790\uff082026\uff09",
                "research_type": "industry_scan",
                "report_family": "industry_deep_report",
                "research_object": "\u4e2d\u56fd\u4f4e\u7a7a\u7ecf\u6d4e\u4ea7\u4e1a\u94fe",
                "chapters": [
                    {
                        "chapter_id": "ch_01",
                        "chapter_title": "\u4f4e\u7a7a\u7ecf\u6d4e\u5546\u4e1a\u5316\u6210\u719f\u5ea6\u5982\u4f55\u5224\u65ad\uff1f",
                        "core_question": "\u54ea\u4e9b\u73af\u8282\u5df2\u7ecf\u5177\u5907\u89c4\u6a21\u5316\u5546\u4e1a\u5316\u6761\u4ef6\uff1f",
                    }
                ],
                "evidence_goals": [
                    {
                        "goal_id": "H1_metric",
                        "chapter_id": "ch_01",
                        "proof_role": "metric",
                        "required_fields": ["metric", "value", "unit", "period", "source_ref"],
                    }
                ],
            },
            "llm_call": {"task": "planning", "model": "deepseek-v4-pro", "status": "ok"},
        }

    monkeypatch.setattr(
        "rag_pipeline.agents.research_planner.call_openai_compatible_json",
        compact_first_success,
    )

    plan = run_research_planner_agent(query="\u4e2d\u56fd\u4f4e\u7a7a\u7ecf\u6d4e\u4ea7\u4e1a\u94fe\u5546\u4e1a\u5316\u673a\u4f1a\u4e0e\u98ce\u9669\u5206\u6790\uff082026\uff09")

    assert len(calls) == 1
    assert plan["planning_status"] == "ok"
    assert plan["planner_mode"] == "compact_first"
    assert plan["planner_llm_degraded"] is False
    assert plan["allow_full_report"] is True


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
