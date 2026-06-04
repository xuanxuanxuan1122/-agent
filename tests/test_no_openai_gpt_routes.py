from __future__ import annotations

from pathlib import Path

from rag_pipeline.agents.dynamic_search_schema import normalize_search_task
from rag_pipeline.config import search_config


def _profile(monkeypatch, name: str, *, model: str, url: str = "https://example.invalid/v1") -> None:
    key = search_config.normalize_llm_profile(name)
    monkeypatch.setenv(f"RAG_LLM_PROFILE_{key}_PROVIDER", "openai_compatible")
    monkeypatch.setenv(f"RAG_LLM_PROFILE_{key}_URL", url)
    monkeypatch.setenv(f"RAG_LLM_PROFILE_{key}_API_KEY", f"{name}-key")
    monkeypatch.setenv(f"RAG_LLM_PROFILE_{key}_MODEL", model)


def test_quality_task_profile_is_not_forced_to_removed_model(monkeypatch):
    removed_model = "gp" + "t-" + "5.5"
    _profile(monkeypatch, "deepseek-v4-pro", model="deepseek-v4-pro")
    _profile(monkeypatch, removed_model, model=removed_model, url="https://api." + "openai.com/v1/responses")
    monkeypatch.setenv("RAG_FORCE_" + "GP" + "T" + "55_QUALITY_TASKS", "true")
    monkeypatch.setenv("RAG_MODEL_QA_PROFILE", "deepseek-v4-pro")

    config = search_config.build_llm_config_for_task("qa")

    assert config["model"] == "deepseek-v4-pro"
    assert "fallback_config" not in config
    assert not config.get("forced_quality_profile")


def test_removed_model_profile_is_not_selected_from_stale_env(monkeypatch):
    removed_model = "gp" + "t-" + "5.5"
    _profile(monkeypatch, removed_model, model=removed_model, url="https://api." + "openai.com/v1/responses")
    monkeypatch.setenv("RAG_MODEL_QA_PROFILE", removed_model)
    monkeypatch.setattr(search_config, "DEFAULT_LLM_SYNTHESIS_PROVIDER", "openai_compatible")
    monkeypatch.setattr(search_config, "DEFAULT_LLM_SYNTHESIS_URL", "https://dashscope.example/v1")
    monkeypatch.setattr(search_config, "DEFAULT_LLM_SYNTHESIS_API_KEY", "fallback-key")
    monkeypatch.setattr(search_config, "DEFAULT_LLM_SYNTHESIS_MODEL", "qwen3.6-plus")
    monkeypatch.setattr(search_config, "DEFAULT_LLM_SYNTHESIS_TIMEOUT", 180.0)
    monkeypatch.setattr(search_config, "DEFAULT_LLM_SYNTHESIS_DISABLE_THINKING", True)

    config = search_config.build_llm_config_for_task("qa")

    assert config["model"] == "qwen3.6-plus"
    assert config["url"] == "https://dashscope.example/v1"


def test_dynamic_search_task_rejects_removed_repair_mode():
    task = normalize_search_task(
        {
            "task_id": "t1",
            "query": "AI Agent market evidence",
            "agent": "openai_" + "web",
            "provider": "openai_" + "web",
            "retrieval_mode": "openai_" + "repair",
        }
    )

    assert task["agent"] == "iqs"
    assert task["retrieval_mode"] == ""


def test_env_example_has_no_removed_model_or_web_search_defaults():
    env_text = Path(".env.example").read_text(encoding="utf-8")

    forbidden = [
        "gp" + "t-" + "5.5",
        "GP" + "T" + "55",
        "RAG_FORCE_" + "GP" + "T" + "55",
        "api." + "openai.com",
        "OPENAI_" + "API_KEY",
        "OPENAI_" + "WEB_SEARCH",
    ]
    for token in forbidden:
        assert token not in env_text


def test_removed_web_search_provider_file_is_removed():
    assert not Path("rag_pipeline/agents/" + "openai_" + "web_search_provider.py").exists()


def test_removed_web_search_repair_labels_are_not_present_in_runtime_code():
    runtime_files = [
        Path("rag_pipeline/agents/brain_agent.py"),
        Path("rag_pipeline/agents/evidence_merger.py"),
        Path("rag_pipeline/flows/report/full_report.py"),
    ]
    combined = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in runtime_files)

    assert "repairable_by_" + "openai" not in combined
    assert "IQS/" + "OpenAI retrieval" not in combined
