from __future__ import annotations

import inspect

from rag_pipeline.agents import (
    analysis_agent,
    chapter_narrative_agent,
    readpage_fact_extractor_agent,
    research_planner,
    section_body_rewrite_agent,
)
from rag_pipeline.flows.report import final_audit_agent, reformatter_agent


ENGLISH_DIRECTIVE_PATTERNS = (
    "You are ",
    "Return only",
    "Return strict",
    "Do not ",
    "Use only ",
    "Never ",
    "Only return",
    "Output JSON",
    "strict JSON",
)


def _assert_chinese_prompt(name: str, prompt: str) -> None:
    chinese_chars = sum(1 for ch in prompt if "\u4e00" <= ch <= "\u9fff")
    assert chinese_chars >= 80, f"{name} should be written primarily in Chinese"
    for pattern in ENGLISH_DIRECTIVE_PATTERNS:
        assert pattern not in prompt, f"{name} still contains English directive: {pattern!r}"


def test_core_llm_prompts_are_chinese_first():
    prompts = {
        "analysis_chapter": analysis_agent._llm_chapter_system_prompt(),
        "analysis_semantic_judge": analysis_agent._semantic_judge_system_prompt(),
        "research_planner": research_planner.RESEARCH_PLANNER_SYSTEM,
        "research_planner_compact": research_planner.RESEARCH_PLANNER_COMPACT_SYSTEM,
        "readpage_fact_extractor": readpage_fact_extractor_agent._system_prompt(),
        "chapter_narrative": chapter_narrative_agent._system_prompt(),
        "section_body_rewrite": section_body_rewrite_agent._system_prompt(),
        "final_audit": final_audit_agent.FINAL_AUDIT_SYSTEM_PROMPT,
        "reformatter": reformatter_agent.REFORMATTER_SYSTEM_PROMPT,
        "reformatter_polish": reformatter_agent.REFORMATTER_POLISH_SYSTEM_PROMPT,
    }
    for name, prompt in prompts.items():
        _assert_chinese_prompt(name, prompt)


def test_inline_llm_prompt_sources_are_chinese_first():
    source_blocks = {
        "rewrite_agent": inspect.getsource(__import__("rag_pipeline.agents.rewrite_agent", fromlist=[""])),
        "web_analysis_agent": inspect.getsource(__import__("rag_pipeline.agents.web_analysis_agent", fromlist=[""])),
        "search_reflection": inspect.getsource(__import__("rag_pipeline.search.reflection", fromlist=[""])),
        "search_synthesis": inspect.getsource(__import__("rag_pipeline.search.synthesis", fromlist=[""])),
        "search_review": inspect.getsource(__import__("rag_pipeline.search.review", fromlist=[""])),
        "search_memory": inspect.getsource(__import__("rag_pipeline.search.memory", fromlist=[""])),
        "search_engine": inspect.getsource(__import__("rag_pipeline.search.engine", fromlist=[""])),
    }
    for name, source in source_blocks.items():
        relevant_lines = []
        for line in source.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "parser.add_argument" in stripped or "help=" in stripped:
                continue
            if "system_prompt" in stripped or "user_payload" in stripped or stripped.startswith('"') or stripped.startswith("'"):
                relevant_lines.append(stripped)
        prompt_like_source = "\n".join(relevant_lines)
        for pattern in ENGLISH_DIRECTIVE_PATTERNS:
            assert pattern not in prompt_like_source, f"{name} still contains English prompt directive: {pattern!r}"
