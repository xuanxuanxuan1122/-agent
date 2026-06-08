from __future__ import annotations

import json

from rag_pipeline.observability.run_trace import RunTraceContext, write_run_trace_from_package


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_run_trace_event_schema_and_sensitive_diagnostics_are_sanitized(tmp_path, monkeypatch):
    monkeypatch.setenv("RUN_TRACE_SAMPLE_LIMIT", "2")
    trace = RunTraceContext(run_id="run-1", output_dir=tmp_path)

    event = trace.emit(
        stage="llm_analysis",
        event="completed",
        status="degraded",
        input_count=4,
        output_count=1,
        drop_count=3,
        reason_counts={"missing_used_evidence_ids": 3},
        sample_ids=["EV-1", "EV-2", "EV-3"],
        message="LLM claims validated",
        diagnostics={
            "api_key": "secret",
            "authorization": "Bearer secret",
            "prompt": "full prompt should not be stored",
            "raw_html": "<html>large page</html>",
            "safe_count": 3,
        },
    )

    events = _read_jsonl(trace.trace_path)
    assert len(events) == 1
    stored = events[0]
    assert event["seq"] == 1
    assert stored["run_id"] == "run-1"
    assert stored["stage"] == "llm_analysis"
    assert stored["status"] == "degraded"
    assert stored["sample_ids"] == ["EV-1", "EV-2"]
    assert stored["diagnostics"]["api_key"] == "[redacted]"
    assert stored["diagnostics"]["authorization"] == "[redacted]"
    assert stored["diagnostics"]["prompt"] == "[redacted]"
    assert stored["diagnostics"]["raw_html"] == "[redacted]"
    assert stored["diagnostics"]["safe_count"] == 3


def test_run_trace_summary_shows_stage_funnel_and_actionable_checks(tmp_path):
    trace = RunTraceContext(run_id="run-2", output_dir=tmp_path)
    trace.emit(
        stage="web_result_filter",
        event="completed",
        status="warning",
        input_count=20,
        output_count=0,
        drop_count=20,
        reason_counts={"low_task_relevance_reject": 20},
    )
    trace.emit(
        stage="llm_analysis",
        event="completed",
        status="ok",
        input_count=4,
        output_count=3,
        drop_count=1,
        reason_counts={"missing_basis_or_reasoning": 1},
    )

    summary_path = trace.write_summary(final_status="warning")
    summary = summary_path.read_text(encoding="utf-8")

    assert "## Stage Funnel" in summary
    assert "| web_result_filter | 20 | 0 | 20 | warning |" in summary
    assert "low_task_relevance_reject" in summary
    assert "web_result_filter accepted 0 of 20" in summary


def test_write_run_trace_from_package_extracts_core_pipeline_events(tmp_path, monkeypatch):
    monkeypatch.setenv("RUN_TRACE_ENABLED", "true")
    writer_package = {
        "quality_mode": True,
        "report_execution_mode": "quality_llm_replay",
        "quality_posture": {
            "mode": "high",
            "disabled": {"query_rewrite": True, "self_refine": True},
            "query_rewrite_max_calls": "4",
            "query_rewrite_max_input_chars": "6000",
        },
        "fact_extractor": {
            "attempted": 2,
            "success_count": 1,
            "fact_card_count": 5,
            "rejected_span_count": 1,
            "invalid_metric_count": 1,
            "cache_hit_count": 1,
        },
        "raw_output": {
            "metadata": {
                "query_plan": [{"query": "AI Agent official data"}],
                "search_tasks": [{"query": "AI Agent official data", "lane_type": "official_data"}],
                "repair_task_selection_summary": {
                    "task_count": 3,
                    "by_proof_role": {"metric": 1, "counter": 1, "case": 1},
                    "deep_budget_exhausted_count": 0,
                },
                "query_rewrite_diagnostics": {
                    "query_rewrite_call_count": 0,
                    "query_rewrite_input_chars_total": 0,
                    "query_rewrite_cache_hit_count": 0,
                    "query_rewrite_budget_exhausted": False,
                },
                "auto_readpage": {"attempted": 2, "succeeded": 1, "failed": 1, "errors": ["timeout"]},
            },
            "quality_processing": {
                "raw_count": 20,
                "normalized_count": 5,
                "task_filtered_count": 15,
                "task_filter_reasons": {"low_task_relevance_reject": 15},
            },
        },
        "evidence_package": {
            "raw_data_points": [{"id": "raw-1"}, {"id": "raw-2"}],
            "analysis_ready_evidence": [{"evidence_id": "EV-1"}],
            "clean_evidence_list": [{"evidence_id": "EV-1"}],
            "source_registry": [{"source_ref": "S1", "source_level": "A"}],
        },
        "structured_analysis": {
            "analysis_stage_diagnostics": {
                "llm_analysis_attempted": True,
                "llm_input_chapter_count": 4,
                "llm_usable_claim_count": 3,
                "llm_dropped_claim_count": 1,
                "llm_validation_issue_counts": {"missing_refs": 1},
                "llm_semantic_judge_counts": {"attempted": 3, "supported": 2, "unsupported": 1},
                "llm_semantic_judge_usage": {"total_tokens": 1200},
                "final_analysis_source": "llm_partial_merged",
            }
        },
        "chapter_packages": [
            {
                "chapter_id": "ch_01",
                "sections": [{"section_id": "s1", "evidence_backed": True}],
                "body_rewrite_global": {
                    "enabled": True,
                    "submitted_count": 1,
                    "success_count": 1,
                    "cache_hit_count": 0,
                    "fallback_count": 0,
                },
            }
        ],
        "render_artifacts": {
            "chapter_narrative": {
                "enabled": True,
                "attempted_count": 1,
                "success_count": 1,
                "fallback_count": 0,
                "rejected_reasons": {},
            }
        },
        "citation_manifest": {
            "citation_manifest_status": "ok",
            "section_citation_refs": {"s1": ["[1]"]},
            "missing_evidence_refs": [],
            "orphan_citation_count": 0,
            "excluded_cited_sources": [],
        },
        "public_narrative_leak_audit": {
            "public_narrative_leak_input_count": 3,
            "public_narrative_leak_removed_count": 2,
            "public_narrative_leak_remaining_count": 0,
            "skipped_global_block_count": 1,
            "public_narrative_leak_reason_counts": {"evidence_processing_language": 1, "diagnostic_global_block": 1},
            "public_narrative_leak_examples": [{"reason": "evidence_processing_language", "text": "该证据来自"}],
        },
        "final_audit_result": {"status": "passed", "blocked": False},
        "report_delivery_status": {"formal_report_written": True, "score_report_written": True},
        "writer_report": {"quality_score": 78, "clean_content_eligible": False},
        "score_gaps": [
            {"gap_id": "GAP-1", "status": "evidence_found", "gap_type": "metric_scope_period_unit_incomplete"},
            {"gap_id": "GAP-2", "status": "still_insufficient", "gap_type": "counter_evidence_missing"},
        ],
    }

    result = write_run_trace_from_package(
        run_id="run-3",
        output_dir=tmp_path,
        writer_package=writer_package,
        final_status="completed",
    )

    events = _read_jsonl(tmp_path / "run-3.trace.jsonl")
    stages = {event["stage"] for event in events}
    assert result["enabled"] is True
    assert result["trace_path"].endswith("run-3.trace.jsonl")
    assert {
        "search_plan",
        "query_rewrite",
        "iqs_search",
        "web_result_filter",
        "readpage",
        "fact_extractor",
        "evidence_repair",
        "evidence_merge",
        "llm_analysis",
        "body_rewrite",
        "chapter_narrative",
        "public_narrative_gate",
        "citation_manifest",
        "writer",
    }.issubset(stages)
    query_rewrite_event = next(event for event in events if event["stage"] == "query_rewrite")
    assert query_rewrite_event["status"] == "skipped"
    assert query_rewrite_event["diagnostics"]["self_refine_disabled_reason"] == "quality_posture"
    repair_event = next(event for event in events if event["stage"] == "evidence_repair")
    assert repair_event["input_count"] == 3
    assert repair_event["output_count"] == 3
    assert repair_event["reason_counts"]["metric"] == 1
    assert repair_event["diagnostics"]["selected_repair_task_count_by_reason"] == {
        "metric": 1,
        "counter": 1,
        "case": 1,
    }
    assert repair_event["diagnostics"]["repair_effectiveness"]["attempted_gap_count"] == 3
    assert repair_event["diagnostics"]["repair_effectiveness"]["closed_gap_count"] == 1
    assert repair_event["diagnostics"]["repair_effectiveness"]["closure_rate"] == 1 / 3
    assert repair_event["diagnostics"]["self_refine_disabled_reason"] == "quality_posture"
    llm_event = next(event for event in events if event["stage"] == "llm_analysis")
    assert llm_event["diagnostics"]["llm_semantic_judge_counts"]["attempted"] == 3
    assert llm_event["diagnostics"]["llm_semantic_judge_usage"]["total_tokens"] == 1200
    public_gate_event = next(event for event in events if event["stage"] == "public_narrative_gate")
    assert public_gate_event["status"] == "ok"
    assert public_gate_event["drop_count"] == 3
    assert public_gate_event["reason_counts"]["evidence_processing_language"] == 1
    assert (tmp_path / "run-3.trace_summary.md").exists()
