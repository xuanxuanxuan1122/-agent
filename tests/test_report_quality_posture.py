import os

from rag_pipeline.flows.report import full_report


POSTURE_KEYS = [
    "IQS_ENABLE_LLM_QUERY_REWRITE",
    "IQS_ENABLE_SELF_REFINE",
    "FULL_REPORT_IQS_ENABLE_SELF_REFINE",
    "BRAIN_AGENT_TEXT_SELF_REFINE",
    "REPORT_CONTINUOUS_EVIDENCE_LOOP",
]


def _clear_posture_env(monkeypatch):
    for key in POSTURE_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_high_quality_posture_keeps_high_cost_search_expansion_disabled(monkeypatch):
    _clear_posture_env(monkeypatch)

    posture = full_report.apply_report_quality_posture("high")

    assert posture["mode"] == "high"
    for key in POSTURE_KEYS:
        assert os.environ[key] == "false"
    assert posture["disabled"]["query_rewrite"] is True
    assert posture["disabled"]["self_refine"] is True


def test_quality_posture_preserves_explicit_user_env(monkeypatch):
    _clear_posture_env(monkeypatch)
    monkeypatch.setenv("IQS_ENABLE_LLM_QUERY_REWRITE", "true")

    posture = full_report.apply_report_quality_posture("high")

    assert os.environ["IQS_ENABLE_LLM_QUERY_REWRITE"] == "true"
    assert posture["preserved_explicit"]["IQS_ENABLE_LLM_QUERY_REWRITE"] == "true"
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
