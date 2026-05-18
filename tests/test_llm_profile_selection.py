from __future__ import annotations

import os
from argparse import Namespace

from rag_pipeline.flows.report.full_report import apply_llm_profile_to_environment, select_llm_profile


def _profile(monkeypatch, name: str, *, model: str, url: str, disable_thinking: str) -> None:
    key = name.upper().replace("-", "_")
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
