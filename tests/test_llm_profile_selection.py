from __future__ import annotations

import os
import re
from argparse import Namespace
from pathlib import Path

from rag_pipeline.flows.report.full_report import (
    apply_llm_profile_to_environment,
    apply_selected_profile_to_task_model_routing,
    select_llm_profile,
)
from rag_pipeline.config import search_config
from rag_pipeline.agents import research_planner
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
    monkeypatch.setenv("RAG_MODEL_PLANNING_PROFILE", "deepseek-v4-pro")
    monkeypatch.setenv("RAG_MODEL_QA_PROFILE", "deepseek-v4-pro")
    monkeypatch.setenv("RAG_MODEL_FINAL_AUDIT_PROFILE", "deepseek-v4-pro")

    planning = search_config.build_llm_config_for_task("planning")
    qa = search_config.build_llm_config_for_task("qa")
    final_audit = search_config.build_llm_config_for_task("final_audit")

    assert planning["model"] == "deepseek-v4-pro"
    assert qa["model"] == "deepseek-v4-pro"
    assert final_audit["model"] == "deepseek-v4-pro"
    assert "fallback_config" not in planning
    assert "fallback_config" not in qa
    assert "fallback_config" not in final_audit


def test_cli_llm_profile_can_override_quality_task_model_routing(monkeypatch):
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
    monkeypatch.setenv("RAG_MODEL_PLANNING_PROFILE", "deepseek-v4-pro")
    monkeypatch.setenv("RAG_MODEL_BODY_REWRITE_PROFILE", "deepseek-v4-pro")
    monkeypatch.setenv("RAG_MODEL_FINAL_AUDIT_PROFILE", "deepseek-v4-pro")
    monkeypatch.setenv("READPAGE_FACT_EXTRACTOR_MODEL_PROFILE", "deepseek-v4-pro")

    selected = select_llm_profile(Namespace(llm_profile="qwen", select_llm=False, no_interactive_input=True))
    overrides = apply_selected_profile_to_task_model_routing(selected, force=True)

    assert overrides["RAG_MODEL_PLANNING_PROFILE"] == "deepseek-v4-pro"
    assert overrides["RAG_MODEL_BODY_REWRITE_PROFILE"] == "deepseek-v4-pro"
    assert overrides["RAG_MODEL_FINAL_AUDIT_PROFILE"] == "deepseek-v4-pro"
    assert overrides["READPAGE_FACT_EXTRACTOR_MODEL_PROFILE"] == "deepseek-v4-pro"
    assert os.environ["RAG_MODEL_PLANNING_PROFILE"] == "qwen"
    assert os.environ["RAG_MODEL_BODY_REWRITE_PROFILE"] == "qwen"
    assert os.environ["RAG_MODEL_FINAL_AUDIT_PROFILE"] == "qwen"
    assert os.environ["READPAGE_FACT_EXTRACTOR_MODEL_PROFILE"] == "qwen"


def test_research_planner_llm_config_enforces_large_output_floor(monkeypatch):
    monkeypatch.setenv("REPORT_PLANNING_MAX_OUTPUT_TOKENS", "8192")
    monkeypatch.setattr(
        research_planner,
        "build_llm_config_for_task",
        lambda task_name: {"model": "planner", "max_output_tokens": 4096},
    )

    config = research_planner._llm_config()

    assert config["max_output_tokens"] == 8192


def test_quality_tasks_respect_explicit_function_profile(monkeypatch):
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
    monkeypatch.setenv("RAG_MODEL_REFORMATTER_PROFILE", "qwen")
    monkeypatch.setenv("RAG_MODEL_REVIEW_STAGE2_PROFILE", "qwen")

    reformatter = search_config.build_llm_config_for_task("reformatter")
    review = search_config.build_llm_config_for_task("review_stage2")

    assert reformatter["model"] == "qwen3.6-plus"
    assert review["model"] == "qwen3.6-plus"
    assert "fallback_config" not in reformatter
    assert "fallback_config" not in review
    assert not reformatter.get("forced_quality_profile")


def test_industry_ecosystem_report_routes_to_industry_deep_report():
    profile = select_report_profile("AI Agent生态发展报告：从工具到智能体的范式跃迁")

    assert profile["name"] == "industry_deep_report"


def test_industrial_software_replacement_report_is_not_policy_impact():
    query = "中国工业软件MES/APS/PLM国产替代：离散制造落地路径、采购决策链与实施失败风险研究"
    profile = select_report_profile(query, {"report_family": "policy_impact_report"})

    assert profile["name"] == "industry_deep_report"


def test_business_model_industry_report_with_policy_words_stays_industry_deep():
    query = "县域新能源商用车充换电网络：重卡物流场景、资产利用率与运营商盈利模型研究"
    profile = select_report_profile(query, {"report_family": "policy_impact_report"})

    assert profile["name"] == "industry_deep_report"


def test_cxo_order_recovery_report_with_act_disruption_stays_industry_deep():
    query = "医药CXO出海订单恢复：美国生物安全法案扰动、产能利用率与客户结构变化研究"
    profile = select_report_profile(query, {"report_family": "policy_impact_report"})

    assert profile["name"] == "industry_deep_report"


def test_legacy_decision_profiles_are_isolated_from_default_report_mainline(monkeypatch):
    monkeypatch.delenv("REPORT_ENABLE_LEGACY_REPORT_PROFILES", raising=False)

    assert select_report_profile("AI industry briefing", {"report_family": "briefing_note"})["name"] == "industry_deep_report"
    assert select_report_profile("Company due diligence report", {"report_family": "company_due_diligence_report"})["name"] == "industry_deep_report"
    assert select_report_profile("Robotics investment memo", {"report_family": "investment_memo"})["name"] == "industry_deep_report"


def test_legacy_decision_profiles_can_be_enabled_for_old_product_lines(monkeypatch):
    monkeypatch.setenv("REPORT_ENABLE_LEGACY_REPORT_PROFILES", "true")

    assert select_report_profile("AI industry briefing", {"report_family": "briefing_note"})["name"] == "briefing_note"
    assert select_report_profile("Company due diligence report", {"report_family": "company_due_diligence_report"})["name"] == "company_due_diligence_report"
    assert select_report_profile("Robotics investment memo", {"report_family": "investment_memo"})["name"] == "investment_memo"


def test_build_llm_config_from_profile_handles_model_pool_names(monkeypatch):
    profiles = {
        "qwen": "qwen3.6-plus",
        "deepseek-v4-pro": "deepseek-v4-pro",
        "deepseek-v4-flash": "deepseek-v4-flash",
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


def test_env_example_does_not_advertise_removed_model_profiles():
    env_example = Path(__file__).resolve().parents[1] / ".env.example"
    content = env_example.read_text(encoding="utf-8")
    removed_profile_name = "ge" + "mini"
    removed_profile_env_prefix = "RAG_LLM_PROFILE_" + "GE" + "MINI"

    assert removed_profile_name not in content.lower()
    assert removed_profile_env_prefix not in content


def test_build_llm_config_for_task_falls_back_to_legacy_synthesis(monkeypatch):
    for name in list(os.environ):
        if name.startswith("RAG_MODEL_"):
            monkeypatch.delenv(name, raising=False)
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
