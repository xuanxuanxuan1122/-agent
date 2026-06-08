import os

import pytest

from rag_pipeline.flows.report import full_report


HIGH_COST_POSTURE_KEYS = [
    "IQS_ENABLE_LLM_QUERY_REWRITE",
    "IQS_ENABLE_SELF_REFINE",
    "FULL_REPORT_IQS_ENABLE_SELF_REFINE",
    "BRAIN_AGENT_TEXT_SELF_REFINE",
    "REPORT_CONTINUOUS_EVIDENCE_LOOP",
]

EVIDENCE_BUDGET_KEYS = [
    "FULL_REPORT_IQS_MAX_QUERIES",
    "FULL_REPORT_IQS_MAX_SEARCH_TASKS",
    "FULL_REPORT_IQS_RESULTS_PER_QUERY",
    "FULL_REPORT_IQS_RERANK_TOP_K",
    "FULL_REPORT_IQS_RERANK_MAX_DOCS",
    "FULL_REPORT_IQS_RERANK_PREFILTER_MAX_DOCS",
    "BRAIN_INITIAL_LANE_ADAPTIVE_SEARCH_BUDGET",
    "BRAIN_FOLLOWUP_ADAPTIVE_SEARCH_BUDGET",
    "IQS_AUTO_READPAGE_TOP_N",
    "IQS_AUTO_READPAGE_REQUIRED_TOP_N",
    "IQS_AUTO_READPAGE_MIN_SCORE",
    "IQS_AUTO_READPAGE_REQUIRED_MIN_SCORE",
    "IQS_READPAGE_PARALLEL_WORKERS",
    "READPAGE_FACT_EXTRACTOR_MAX_CALLS_PER_REPORT",
    "READPAGE_FACT_EXTRACTOR_MAX_PAGES_PER_TASK",
    "READPAGE_FACT_EXTRACTOR_MAX_CHARS_PER_PAGE",
    "BRAIN_LLM_ANALYSIS_MAX_CHAPTERS",
    "BRAIN_LLM_ANALYSIS_MAX_FACTS_PER_CHAPTER",
    "BRAIN_LLM_ANALYSIS_MAX_FACT_CHARS",
    "BRAIN_LLM_ANALYSIS_CONCURRENCY",
    "REPORT_FACTS_PER_CHAPTER_ARGUMENTS",
    "REPORT_CHAPTER_FACT_DIGEST_LIMIT",
]

WRITING_BUDGET_KEYS = [
    "BRAIN_ENABLE_LLM_EVIDENCE_ANALYSIS",
    "REPORT_ENABLE_LLM_BODY_REWRITE",
    "REPORT_BODY_REWRITE_MAX_SECTIONS",
    "REPORT_BODY_REWRITE_MAX_ELAPSED_SECONDS",
    "REPORT_BODY_REWRITE_CONCURRENCY",
    "REPORT_BODY_REWRITE_MAX_EXPANSION_RATIO",
    "REPORT_BODY_REWRITE_TARGET_SECTION_CHARS",
    "REPORT_ENABLE_LLM_CHAPTER_NARRATIVE",
    "REPORT_CHAPTER_NARRATIVE_MAX_CHAPTERS",
    "REPORT_TARGET_BODY_CHARS",
    "REPORT_TARGET_BODY_CHARS_BLOCKING",
    "REPORT_COMPOSER_TARGET_SECTION_CHARS",
    "REPORT_RENDER_MIN_SECTION_CHARS",
]

DEEPSEEK_QUALITY_MODEL_KEYS = [
    "RAG_MODEL_PLANNING_PROFILE",
    "RAG_MODEL_EVIDENCE_MERGE_PROFILE",
    "RAG_MODEL_COVERAGE_EVAL_PROFILE",
    "RAG_MODEL_RISK_PROFILE",
    "RAG_MODEL_DECISION_PROFILE",
    "RAG_MODEL_REFORMATTER_PROFILE",
    "RAG_MODEL_BODY_REWRITE_PROFILE",
    "RAG_MODEL_QA_PROFILE",
    "RAG_MODEL_REFLECTION_PROFILE",
    "RAG_MODEL_FINAL_AUDIT_PROFILE",
    "READPAGE_FACT_EXTRACTOR_MODEL_PROFILE",
]

QWEN_WEB_SEARCH_MODEL_KEYS = [
    "RAG_MODEL_QUERY_REWRITE_PROFILE",
    "RAG_MODEL_WEB_SUMMARY_PROFILE",
]

MODEL_ROUTING_KEYS = [*DEEPSEEK_QUALITY_MODEL_KEYS, *QWEN_WEB_SEARCH_MODEL_KEYS]

POSTURE_KEYS = [*HIGH_COST_POSTURE_KEYS, *EVIDENCE_BUDGET_KEYS, *WRITING_BUDGET_KEYS, *MODEL_ROUTING_KEYS]


@pytest.fixture(autouse=True)
def restore_posture_env():
    previous = {key: os.environ.get(key) for key in POSTURE_KEYS}
    yield
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _clear_posture_env(monkeypatch):
    for key in POSTURE_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_high_quality_posture_keeps_high_cost_search_expansion_disabled(monkeypatch):
    _clear_posture_env(monkeypatch)

    posture = full_report.apply_report_quality_posture("high")

    assert posture["mode"] == "high"
    for key in HIGH_COST_POSTURE_KEYS:
        assert os.environ[key] == "false"
    assert posture["disabled"]["query_rewrite"] is True
    assert posture["disabled"]["self_refine"] is True


def test_high_quality_posture_increases_evidence_search_and_analysis_capacity(monkeypatch):
    _clear_posture_env(monkeypatch)

    posture = full_report.apply_report_quality_posture("high")
    options = full_report.full_report_iqs_options()

    assert posture["mode"] == "high"
    assert options["enable_self_refine"] is False
    assert options["max_queries"] >= 4
    assert options["max_search_tasks"] >= 32
    assert options["results_per_query"] >= 80
    assert os.environ["BRAIN_INITIAL_LANE_ADAPTIVE_SEARCH_BUDGET"] == "false"
    assert os.environ["BRAIN_FOLLOWUP_ADAPTIVE_SEARCH_BUDGET"] == "false"
    assert int(os.environ["READPAGE_FACT_EXTRACTOR_MAX_CALLS_PER_REPORT"]) >= 80
    assert int(os.environ["READPAGE_FACT_EXTRACTOR_MAX_PAGES_PER_TASK"]) >= 6
    assert int(os.environ["BRAIN_LLM_ANALYSIS_MAX_CHAPTERS"]) >= 12
    assert int(os.environ["BRAIN_LLM_ANALYSIS_MAX_FACTS_PER_CHAPTER"]) >= 16
    assert int(os.environ["REPORT_FACTS_PER_CHAPTER_ARGUMENTS"]) >= 24
    assert posture["evidence_depth"]["llm_analysis_max_facts_per_chapter"] == os.environ["BRAIN_LLM_ANALYSIS_MAX_FACTS_PER_CHAPTER"]


def test_high_quality_posture_enables_quality_writing_without_hard_longform_target(monkeypatch):
    _clear_posture_env(monkeypatch)

    posture = full_report.apply_report_quality_posture("high")

    assert posture["mode"] == "high"
    assert os.environ["BRAIN_ENABLE_LLM_EVIDENCE_ANALYSIS"] == "true"
    assert os.environ["REPORT_ENABLE_LLM_BODY_REWRITE"] == "true"
    assert int(os.environ["REPORT_BODY_REWRITE_MAX_SECTIONS"]) >= 24
    assert int(os.environ["REPORT_BODY_REWRITE_MAX_ELAPSED_SECONDS"]) >= 900
    assert os.environ["REPORT_ENABLE_LLM_CHAPTER_NARRATIVE"] == "true"
    assert int(os.environ["REPORT_CHAPTER_NARRATIVE_MAX_CHAPTERS"]) >= 12
    assert int(os.environ["REPORT_TARGET_BODY_CHARS"]) == 0
    assert os.environ["REPORT_TARGET_BODY_CHARS_BLOCKING"] == "false"
    assert int(os.environ["REPORT_COMPOSER_TARGET_SECTION_CHARS"]) >= 450
    assert int(os.environ["REPORT_RENDER_MIN_SECTION_CHARS"]) == 0


def test_high_quality_posture_routes_quality_models_to_deepseek_and_web_search_to_qwen(monkeypatch):
    _clear_posture_env(monkeypatch)

    posture = full_report.apply_report_quality_posture("high")

    for key in DEEPSEEK_QUALITY_MODEL_KEYS:
        assert os.environ[key] == "deepseek-v4-pro"
        assert posture["model_routing"][key] == "deepseek-v4-pro"
    for key in QWEN_WEB_SEARCH_MODEL_KEYS:
        assert os.environ[key] == "qwen"
        assert posture["model_routing"][key] == "qwen"


def test_quality_posture_replaces_stale_removed_model_profiles(monkeypatch):
    removed_model = "gp" + "t-" + "5.5"
    _clear_posture_env(monkeypatch)
    monkeypatch.setenv("RAG_MODEL_QA_PROFILE", removed_model)
    monkeypatch.setenv("RAG_MODEL_WEB_SUMMARY_PROFILE", removed_model)

    posture = full_report.apply_report_quality_posture("high")

    assert os.environ["RAG_MODEL_QA_PROFILE"] == "deepseek-v4-pro"
    assert os.environ["RAG_MODEL_WEB_SUMMARY_PROFILE"] == "qwen"
    assert posture["replaced_removed_model_profiles"]["RAG_MODEL_QA_PROFILE"] == removed_model
    assert posture["replaced_removed_model_profiles"]["RAG_MODEL_WEB_SUMMARY_PROFILE"] == removed_model


def test_quality_posture_locks_non_default_model_routes_to_deepseek_and_qwen(monkeypatch):
    _clear_posture_env(monkeypatch)
    monkeypatch.setenv("RAG_MODEL_QA_PROFILE", "qwen")
    monkeypatch.setenv("RAG_MODEL_QUERY_REWRITE_PROFILE", "deepseek-v4-pro")

    posture = full_report.apply_report_quality_posture("high")

    assert os.environ["RAG_MODEL_QA_PROFILE"] == "deepseek-v4-pro"
    assert os.environ["RAG_MODEL_QUERY_REWRITE_PROFILE"] == "qwen"
    assert posture["overridden_model_profiles"]["RAG_MODEL_QA_PROFILE"] == "qwen"
    assert posture["overridden_model_profiles"]["RAG_MODEL_QUERY_REWRITE_PROFILE"] == "deepseek-v4-pro"


def test_quality_posture_can_preserve_custom_model_routes_when_escape_hatch_is_set(monkeypatch):
    _clear_posture_env(monkeypatch)
    monkeypatch.setenv("REPORT_ALLOW_CUSTOM_MODEL_ROUTING", "true")
    monkeypatch.setenv("RAG_MODEL_QA_PROFILE", "qwen")
    monkeypatch.setenv("RAG_MODEL_QUERY_REWRITE_PROFILE", "deepseek-v4-pro")

    posture = full_report.apply_report_quality_posture("high")

    assert os.environ["RAG_MODEL_QA_PROFILE"] == "qwen"
    assert os.environ["RAG_MODEL_QUERY_REWRITE_PROFILE"] == "deepseek-v4-pro"
    assert posture["preserved_explicit"]["RAG_MODEL_QA_PROFILE"] == "qwen"
    assert posture["preserved_explicit"]["RAG_MODEL_QUERY_REWRITE_PROFILE"] == "deepseek-v4-pro"


def test_quality_posture_preserves_explicit_user_env(monkeypatch):
    _clear_posture_env(monkeypatch)
    monkeypatch.setenv("IQS_ENABLE_LLM_QUERY_REWRITE", "true")
    monkeypatch.setenv("FULL_REPORT_IQS_MAX_SEARCH_TASKS", "12")

    posture = full_report.apply_report_quality_posture("high")

    assert os.environ["IQS_ENABLE_LLM_QUERY_REWRITE"] == "true"
    assert posture["preserved_explicit"]["IQS_ENABLE_LLM_QUERY_REWRITE"] == "true"
    assert os.environ["FULL_REPORT_IQS_MAX_SEARCH_TASKS"] == "12"
    assert posture["preserved_explicit"]["FULL_REPORT_IQS_MAX_SEARCH_TASKS"] == "12"
    assert os.environ["IQS_ENABLE_SELF_REFINE"] == "false"


def test_longform_target_upgrades_balanced_mode_to_high_writing_path():
    mode = full_report.resolve_report_quality_mode("balanced", "20000")

    assert mode == "high"


def test_balanced_mode_stays_balanced_without_longform_target():
    mode = full_report.resolve_report_quality_mode("balanced", "8000")

    assert mode == "balanced"


def test_strict_research_posture_is_the_only_mode_that_allows_search_expansion(monkeypatch):
    _clear_posture_env(monkeypatch)

    posture = full_report.apply_report_quality_posture("strict_research")

    assert posture["mode"] == "strict_research"
    assert os.environ["IQS_ENABLE_LLM_QUERY_REWRITE"] == "true"
    assert os.environ["FULL_REPORT_IQS_ENABLE_SELF_REFINE"] == "true"
    assert os.environ["REPORT_CONTINUOUS_EVIDENCE_LOOP"] == "false"
    assert posture["disabled"]["query_rewrite"] is False
