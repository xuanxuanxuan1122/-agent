from __future__ import annotations

import os
import re
from argparse import Namespace

from rag_pipeline.flows.report.full_report import apply_llm_profile_to_environment, select_llm_profile
from rag_pipeline.config import search_config
from rag_pipeline.agents.report_profile_registry import select_report_profile


def _profile(monkeypatch, name: str, *, model: str, url: str, disable_thinking: str) -> None:
    key = re.sub(r"[^A-Z0-9]+", "_", name.upper()).strip("_")
    monkeypatch.setenv(f"RAG_LLM_PROFILE_{key}_PROVIDER", "openai_compatible")
    monkeypatch.setenv(f"RAG_LLM_PROFILE_{key}_URL", url)
    monkeypatch.setenv(f"RAG_LLM_PROFILE_{key}_API_KEY", f"{name}-key")
    monkeypatch.setenv(f"RAG_LLM_PROFILE_{key}_MODEL", model)
    monkeypatch.setenv(f"RAG_LLM_PROFILE_{key}_TIMEOUT", "240")
    monkeypatch.setenv(f"RAG_LLM_PROFILE_{key}_DISABLE_THINKING", disable_thinking)


def test_apply_llm_profile_updates_only_execution_model(monkeypatch):
    _profile(
        monkeypatch,
        "deepseek-v4-pro",
        model="deepseek-v4-pro",
        url="https://api.deepseek.example/chat/completions",
        disable_thinking="false",
    )
    monkeypatch.setenv("RAG_LLM_DISABLE_THINKING", "1")
    for prefix in ("RAG_LLM_PLANNER", "RAG_LLM_ANSWER_REVIEW", "RAG_LLM_REFLECTION"):
        monkeypatch.setenv(f"{prefix}_MODEL", f"{prefix}-sentinel")
        monkeypatch.setenv(f"{prefix}_URL", f"https://{prefix.lower()}.example/v1")

    apply_llm_profile_to_environment("deepseek-v4-pro")

    assert "deepseek-v4-pro" == os.environ["RAG_LLM_SYNTHESIS_MODEL"]
    assert "false" == os.environ["RAG_LLM_SYNTHESIS_DISABLE_THINKING"]
    assert "1" == os.environ["RAG_LLM_DISABLE_THINKING"]
    for prefix in ("RAG_LLM_PLANNER", "RAG_LLM_ANSWER_REVIEW", "RAG_LLM_REFLECTION"):
        assert f"{prefix}-sentinel" == os.environ[f"{prefix}_MODEL"]


def test_select_llm_profile_keeps_active_profile_unchanged(monkeypatch):
    _profile(
        monkeypatch,
        "qwen",
        model="qwen3.6-plus",
        url="https://dashscope.example/v1",
        disable_thinking="true",
    )
    _profile(
        monkeypatch,
        "deepseek-v4-pro",
        model="deepseek-v4-pro",
        url="https://api.deepseek.example/chat/completions",
        disable_thinking="false",
    )
    monkeypatch.setenv("RAG_LLM_ACTIVE_PROFILE", "qwen")
    monkeypatch.setenv("RAG_LLM_EXECUTION_PROFILE", "qwen")

    selected = select_llm_profile(
        Namespace(llm_profile="deepseek-v4-pro", select_llm=False, no_interactive_input=True)
    )

    assert selected == "deepseek-v4-pro"
    assert os.environ["RAG_LLM_EXECUTION_PROFILE"] == "deepseek-v4-pro"
    assert os.environ["RAG_LLM_ACTIVE_PROFILE"] == "qwen"
    assert os.environ["RAG_LLM_SYNTHESIS_MODEL"] == "deepseek-v4-pro"


def test_build_llm_config_for_task_routes_to_function_profile(monkeypatch):
    _profile(
        monkeypatch,
        "qwen",
        model="qwen3.6-plus",
        url="https://dashscope.example/v1",
        disable_thinking="true",
    )
    _profile(
        monkeypatch,
        "deepseek-v4-pro",
        model="deepseek-v4-pro",
        url="https://api.deepseek.example/chat/completions",
        disable_thinking="false",
    )
    _profile(
        monkeypatch,
        "gpt-5.5",
        model="gpt-5.5",
        url="https://api.openai.example/v1",
        disable_thinking="false",
    )
    monkeypatch.setenv("RAG_LLM_PROFILE_GPT_5_5_REASONING_EFFORT", "high")
    monkeypatch.setenv("RAG_LLM_PROFILE_GPT_5_5_MAX_OUTPUT_TOKENS", "32000")
    monkeypatch.setenv("RAG_MODEL_PLANNING_PROFILE", "gpt-5.5")
    monkeypatch.setenv("RAG_MODEL_QA_PROFILE", "gpt-5.5")
    monkeypatch.setenv("RAG_MODEL_FINAL_AUDIT_PROFILE", "gpt-5.5")

    planning = search_config.build_llm_config_for_task("planning")
    qa = search_config.build_llm_config_for_task("qa")
    final_audit = search_config.build_llm_config_for_task("final_audit")

    assert planning["model"] == "gpt-5.5"
    assert qa["model"] == "gpt-5.5"
    assert final_audit["model"] == "gpt-5.5"
    assert planning["fallback_config"]["model"] == "deepseek-v4-pro"
    assert qa["fallback_config"]["model"] == "deepseek-v4-pro"
    assert final_audit["fallback_config"]["model"] == "deepseek-v4-pro"
    assert final_audit["reasoning_effort"] == "high"
    assert final_audit["max_output_tokens"] == 32000


def test_quality_tasks_force_gpt55_profile_when_available(monkeypatch):
    _profile(
        monkeypatch,
        "qwen",
        model="qwen3.6-plus",
        url="https://dashscope.example/v1",
        disable_thinking="true",
    )
    _profile(
        monkeypatch,
        "deepseek-v4-pro",
        model="deepseek-v4-pro",
        url="https://api.deepseek.example/chat/completions",
        disable_thinking="false",
    )
    _profile(
        monkeypatch,
        "gpt-5.5",
        model="gpt-5.5",
        url="https://api.openai.example/v1",
        disable_thinking="false",
    )
    monkeypatch.setenv("RAG_MODEL_REFORMATTER_PROFILE", "qwen")
    monkeypatch.setenv("RAG_MODEL_REVIEW_STAGE2_PROFILE", "qwen")

    reformatter = search_config.build_llm_config_for_task("reformatter")
    review = search_config.build_llm_config_for_task("review_stage2")

    assert reformatter["model"] == "gpt-5.5"
    assert review["model"] == "gpt-5.5"
    assert reformatter["fallback_config"]["model"] == "deepseek-v4-pro"
    assert review["fallback_config"]["model"] == "deepseek-v4-pro"
    assert reformatter["forced_quality_profile"] is True


def test_industry_ecosystem_report_routes_to_industry_deep_report():
    profile = select_report_profile("AI Agent生态发展报告：从工具到智能体的范式跃迁")

    assert profile["name"] == "industry_deep_report"


def test_build_llm_config_from_profile_handles_model_pool_names(monkeypatch):
    profiles = {
        "qwen": "qwen3.6-plus",
        "deepseek-v4-pro": "deepseek-v4-pro",
        "deepseek-v4-flash": "deepseek-v4-flash",
        "gpt-5.5": "gpt-5.5",
        "gemini-3.5-flash": "gemini-3.5-flash",
    }
    for profile, model in profiles.items():
        _profile(
            monkeypatch,
            profile,
            model=model,
            url=f"https://{profile}.example/v1",
            disable_thinking="false",
        )

    for profile, model in profiles.items():
        config = search_config.build_llm_config_from_profile(profile)
        assert config["provider"] == "openai_compatible"
        assert config["model"] == model
        assert config["url"] == f"https://{profile}.example/v1"


def test_build_llm_config_for_task_falls_back_to_legacy_synthesis(monkeypatch):
    for name in list(os.environ):
        if name.startswith("RAG_MODEL_"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RAG_FORCE_GPT55_QUALITY_TASKS", "false")
    monkeypatch.setenv("RAG_MODEL_QA_PROFILE", "deepseek-v4-pro")
    monkeypatch.setenv("RAG_LLM_PROFILE_DEEPSEEK_V4_PRO_URL", "")
    monkeypatch.setenv("RAG_LLM_PROFILE_DEEPSEEK_V4_PRO_API_KEY", "")
    monkeypatch.setenv("RAG_LLM_PROFILE_DEEPSEEK_V4_PRO_MODEL", "")
    monkeypatch.setattr(search_config, "DEFAULT_LLM_SYNTHESIS_PROVIDER", "openai_compatible")
    monkeypatch.setattr(search_config, "DEFAULT_LLM_SYNTHESIS_URL", "https://dashscope.example/v1")
    monkeypatch.setattr(search_config, "DEFAULT_LLM_SYNTHESIS_API_KEY", "fallback-key")
    monkeypatch.setattr(search_config, "DEFAULT_LLM_SYNTHESIS_MODEL", "qwen3.6-plus")
    monkeypatch.setattr(search_config, "DEFAULT_LLM_SYNTHESIS_TIMEOUT", 180.0)
    monkeypatch.setattr(search_config, "DEFAULT_LLM_SYNTHESIS_DISABLE_THINKING", True)

    config = search_config.build_llm_config_for_task("qa")

    assert config["model"] == "qwen3.6-plus"
    assert config["url"] == "https://dashscope.example/v1"
    assert config["disable_thinking"] is True
