from __future__ import annotations

from rag_pipeline.cache.stage_execution_guard import (
    get_cached_stage_output,
    get_stage_execution_summary,
    record_stage_execution,
    reset_stage_execution_guard,
    stable_stage_hash,
    store_stage_output,
)
from rag_pipeline.agents.analysis_agent import run_analysis_agent
from rag_pipeline.agents import brain_agent as brain_agent_module
from rag_pipeline.agents.claim_builder_agent import run_claim_builder_agent


def test_stage_execution_guard_records_duplicate_invocations(monkeypatch):
    monkeypatch.setenv("RUNTIME_PROBE_RUN_ID", "run-dup")
    reset_stage_execution_guard()
    input_hash = stable_stage_hash({"claim_ids": ["CL-1"], "chapter": "CH"})

    first = record_stage_execution(stage="claim_builder", input_hash=input_hash)
    second = record_stage_execution(stage="claim_builder", input_hash=input_hash)
    summary = get_stage_execution_summary("run-dup")

    assert first["invocation_count"] == 1
    assert first["duplicate_stage_execution"] is False
    assert second["invocation_count"] == 2
    assert second["duplicate_stage_execution"] is True
    assert summary["duplicate_stage_count"] == 1


def test_stage_execution_guard_output_cache_returns_deepcopy(monkeypatch):
    monkeypatch.setenv("RUNTIME_PROBE_RUN_ID", "run-cache")
    reset_stage_execution_guard()
    input_hash = stable_stage_hash({"stage": "claim_builder", "items": [1, 2]})
    output = [{"claim_id": "CL-1", "nested": {"refs": ["EV-1"]}}]

    assert get_cached_stage_output(stage="claim_builder", input_hash=input_hash)["hit"] is False
    store_stage_output(stage="claim_builder", input_hash=input_hash, output=output)

    first = get_cached_stage_output(stage="claim_builder", input_hash=input_hash)
    first["output"][0]["nested"]["refs"].append("EV-mutated")
    second = get_cached_stage_output(stage="claim_builder", input_hash=input_hash)
    summary = get_stage_execution_summary("run-cache")

    assert first["hit"] is True
    assert second["hit"] is True
    assert second["output"][0]["nested"]["refs"] == ["EV-1"]
    assert summary["cache_hit_count"] == 2


def test_claim_builder_uses_stage_guard_cache_for_same_run(monkeypatch):
    monkeypatch.setenv("RUNTIME_PROBE_RUN_ID", "run-claim-cache")
    monkeypatch.setenv("REPORT_STAGE_EXECUTION_GUARD_ENABLED", "true")
    monkeypatch.setenv("REPORT_STAGE_EXECUTION_GUARD_CACHE_ENABLED", "true")
    reset_stage_execution_guard()

    package = {
        "chapter_id": "CH_market",
        "chapter_title": "Market signals",
        "claim_ids": ["CL-1"],
        "core_evidence": [
            {
                "evidence_id": "EV-1",
                "ref": "EV-1",
                "source_id": "S1",
                "source_ref": "[1]",
                "source_level": "B",
                "fact": "Market demand signal is visible in public data.",
            }
        ],
    }
    structured = {
        "claim_units": [
            {
                "claim_id": "CL-1",
                "chapter_id": "CH_market",
                "claim": "Market demand has a visible public signal.",
                "fact_ids": ["EV-1"],
                "source_ids": ["S1"],
                "supporting_facts": ["Market demand signal is visible in public data."],
                "claim_strength": "directional",
            }
        ]
    }

    first = run_claim_builder_agent(chapter_evidence_packages=[package], structured_analysis=structured)
    first[0]["claim_id"] = "MUTATED"
    second = run_claim_builder_agent(chapter_evidence_packages=[package], structured_analysis=structured)
    summary = get_stage_execution_summary("run-claim-cache")

    assert second[0]["claim_id"] == "CL-1"
    assert summary["cache_hit_count"] == 1
    assert summary["duplicate_stage_count"] == 1


def test_analysis_agent_uses_run_scoped_stage_cache_for_same_input(monkeypatch):
    monkeypatch.setenv("RUNTIME_PROBE_RUN_ID", "run-analysis-dup")
    monkeypatch.setenv("BRAIN_ENABLE_LLM_EVIDENCE_ANALYSIS", "false")
    reset_stage_execution_guard()
    evidence_package = {
        "query": "market",
        "analysis_ready_evidence": [
            {
                "evidence_id": "EV-1",
                "chapter_id": "ch_01",
                "fact": "Market demand signal is visible in public data.",
                "source": {"url": "https://example.com/report"},
            }
        ],
    }

    first = run_analysis_agent(evidence_package)
    second = run_analysis_agent(evidence_package)
    first_guard = first["structured_analysis"]["analysis_stage_diagnostics"]["stage_execution_guard"]
    second_diag = second["structured_analysis"]["analysis_stage_diagnostics"]
    second_guard = second_diag["stage_execution_guard"]

    assert first_guard["invocation_count"] == 1
    assert first_guard["duplicate_stage_execution"] is False
    assert second_guard["invocation_count"] == 2
    assert second_guard["duplicate_stage_execution"] is True
    assert second_guard["cache_hit"] is True
    assert second_diag["run_scoped_analysis_cache_hit"] is True


def test_brain_analysis_material_cache_ignores_transient_metadata(monkeypatch):
    monkeypatch.setenv("RUNTIME_PROBE_RUN_ID", "run-brain-material-cache")
    monkeypatch.setenv("BRAIN_ENABLE_LLM_EVIDENCE_ANALYSIS", "false")
    reset_stage_execution_guard()

    calls = {"count": 0}

    def fake_run_analysis_agent(evidence_package, *, query="", llm_config=None, deadline_ts=None):
        calls["count"] += 1
        return {
            "query": query,
            "evidence_package": evidence_package,
            "structured_analysis": {
                "claim_units": [{"claim_id": "CL-1", "claim": "cached"}],
                "analysis_stage_diagnostics": {"call_count": calls["count"]},
            },
            "answer_text": "",
            "raw_output": {"analysis": {"call_count": calls["count"]}},
        }

    monkeypatch.setattr(brain_agent_module, "run_analysis_agent", fake_run_analysis_agent)
    base_package = {
        "query": "market",
        "analysis_ready_evidence": [{"evidence_id": "EV-1", "fact": "same evidence"}],
        "source_registry": [{"source_id": "S1", "url": "https://example.com"}],
        "metadata": {"runtime_trace": {"round": 1}},
    }
    changed_transient_package = {
        **base_package,
        "metadata": {"runtime_trace": {"round": 2}, "debug_snapshot": {"ts": "later"}},
    }

    first = brain_agent_module._run_brain_analysis_agent(base_package, query="market", llm_config={"model": "fake"})
    second = brain_agent_module._run_brain_analysis_agent(changed_transient_package, query="market", llm_config={"model": "fake"})
    second_diag = second["structured_analysis"]["analysis_stage_diagnostics"]
    summary = get_stage_execution_summary("run-brain-material-cache")

    assert calls["count"] == 1
    assert first["structured_analysis"]["analysis_stage_diagnostics"]["brain_analysis_cache_hit"] is False
    assert second_diag["brain_analysis_cache_hit"] is True
    assert summary["cache_hit_count"] == 1
