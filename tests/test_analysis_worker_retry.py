"""P3: per-chapter LLM analysis must retry transient errors and preserve partial
success, so one network blip cannot zero out the whole analysis stage."""
from __future__ import annotations

import rag_pipeline.agents.analysis_agent as aa


def test_is_transient_llm_error_classification():
    assert aa._is_transient_llm_error(RuntimeError("LLMCallError: <urlopen error [Errno 11001] getaddrinfo failed>"))
    assert aa._is_transient_llm_error(TimeoutError("request timed out"))
    assert aa._is_transient_llm_error(RuntimeError("HTTP 503 Service Unavailable"))
    # parse/validation problems are not transient -> must not be retried,
    # even when wrapped in LLMCallError (the real planner/merge failure shape).
    assert not aa._is_transient_llm_error(ValueError("invalid JSON: missing claim_units"))
    assert not aa._is_transient_llm_error(KeyError("chapter_id"))
    assert not aa._is_transient_llm_error(RuntimeError("LLMCallError: LLM response is not valid JSON: {"))
    assert not aa._is_transient_llm_error(RuntimeError("LLMCallError: Unterminated string starting at line 24"))


def _stub_llm_env(monkeypatch, *, retries: str, concurrency: str = "2"):
    monkeypatch.setenv("BRAIN_LLM_ANALYSIS_MAX_RETRIES", retries)
    monkeypatch.setenv("BRAIN_LLM_ANALYSIS_RETRY_BASE_SECONDS", "0")  # no real sleep in tests
    monkeypatch.setenv("BRAIN_LLM_ANALYSIS_CONCURRENCY", concurrency)
    monkeypatch.setattr(aa, "llm_config_is_ready", lambda c: True)
    monkeypatch.setattr(aa, "normalize_llm_config", lambda c: {"model": "test-model"})


def test_worker_retries_transient_then_succeeds(monkeypatch):
    _stub_llm_env(monkeypatch, retries="2", concurrency="1")
    monkeypatch.setattr(aa, "build_llm_analysis_input_v2", lambda ep, fb: {"chapters": [{"chapter_id": "ch_01"}]})
    calls = {"n": 0}

    def flaky(*, evidence_package, chapter_payload, llm_config):
        calls["n"] += 1
        if calls["n"] < 3:  # fail twice (transient), succeed on the 3rd
            raise RuntimeError("LLMCallError: getaddrinfo failed")
        return {"chapter_synthesis": [{"chapter_id": "ch_01", "claim_units": [{"claim": "x"}]}], "_llm_cache_hit": False, "_llm_usage": {}}

    monkeypatch.setattr(aa, "synthesize_chapter_with_llm_analysis", flaky)
    out = aa.synthesize_with_llm_analysis_v2(evidence_package={}, fallback={}, llm_config={"model": "m", "api_key": "k", "url": "u"})

    assert out["_llm_failed_chapter_count"] == 0
    assert out["_llm_retry_count"] == 2
    assert len(out["chapter_synthesis"]) == 1


def test_worker_partial_success_one_chapter_persistently_fails(monkeypatch):
    _stub_llm_env(monkeypatch, retries="1")
    monkeypatch.setattr(
        aa, "build_llm_analysis_input_v2", lambda ep, fb: {"chapters": [{"chapter_id": "ch_01"}, {"chapter_id": "ch_02"}]}
    )

    def selective(*, evidence_package, chapter_payload, llm_config):
        if chapter_payload.get("chapter_id") == "ch_01":
            raise RuntimeError("getaddrinfo failed")  # always transient-fails
        return {"chapter_synthesis": [{"chapter_id": "ch_02", "claim_units": [{"claim": "y"}]}], "_llm_cache_hit": False, "_llm_usage": {}}

    monkeypatch.setattr(aa, "synthesize_chapter_with_llm_analysis", selective)
    out = aa.synthesize_with_llm_analysis_v2(evidence_package={}, fallback={}, llm_config={"model": "m", "api_key": "k", "url": "u"})

    # ch_01 exhausts retries and fails; ch_02 survives -> partial success preserved
    assert out["_llm_failed_chapter_count"] == 1
    assert len(out["chapter_synthesis"]) == 1
    assert out["chapter_synthesis"][0]["chapter_id"] == "ch_02"


def test_worker_does_not_retry_non_transient_errors(monkeypatch):
    _stub_llm_env(monkeypatch, retries="3", concurrency="1")
    monkeypatch.setattr(aa, "build_llm_analysis_input_v2", lambda ep, fb: {"chapters": [{"chapter_id": "ch_01"}]})
    calls = {"n": 0}

    def bad_parse(*, evidence_package, chapter_payload, llm_config):
        calls["n"] += 1
        raise ValueError("invalid JSON: missing claim_units")

    monkeypatch.setattr(aa, "synthesize_chapter_with_llm_analysis", bad_parse)
    out = aa.synthesize_with_llm_analysis_v2(evidence_package={}, fallback={}, llm_config={"model": "m", "api_key": "k", "url": "u"})

    assert calls["n"] == 1  # no retries on a non-transient error
    assert out["_llm_failed_chapter_count"] == 1
    assert out["_llm_retry_count"] == 0
