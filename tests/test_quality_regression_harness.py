from __future__ import annotations

import json
from pathlib import Path

from rag_pipeline.contracts.evidence_admission import decide_evidence_admission
from rag_pipeline.quality.regression import (
    build_run_quality_snapshot,
    load_quality_snapshots_from_paths,
    summarize_quality_regression_suite,
    summarize_repair_effectiveness,
    summarize_topic_regression,
    validate_golden_topic_suite,
)
from rag_pipeline.quality.executor import build_quality_execution_plan, run_quality_regression_execution


def _writer_package(
    *,
    run_id: str,
    score: int,
    blocked: bool,
    usable_claims: int,
    thin_report_risk: bool = False,
) -> dict:
    return {
        "metadata": {
            "run_id": run_id,
            "topic_id": "ai-agent-enterprise",
            "domain": "software_ai",
            "query": "AI Agent enterprise adoption opportunities and risks",
        },
        "structured_analysis": {
            "analysis_stage_diagnostics": {
                "llm_usable_claim_count": usable_claims,
                "llm_dropped_claim_count": 35 - usable_claims,
                "correctness_filter_summary": {
                    "thin_report_risk": thin_report_risk,
                    "recommended_mode": "limited_evidence_draft" if thin_report_risk else "normal",
                },
            }
        },
        "writer_report": {
            "quality_score": score,
            "report_status": "final_clean" if not blocked else "formal_scored",
            "estimated_chars": 12000 if not thin_report_risk else 3600,
        },
        "final_audit_result": {
            "status": "fatal" if blocked else "passed",
            "blocked": blocked,
            "critical_findings": [
                {
                    "type": "citation_semantic_mismatch",
                    "severity": "fatal",
                    "message": "Cited source does not support the claim.",
                }
            ]
            if blocked
            else [],
        },
        "report_delivery_status": {"formal_report_written": not blocked},
    }


def test_topic_regression_summary_measures_pass_rate_variance_and_fatals():
    runs = [
        build_run_quality_snapshot(
            {
                **_writer_package(run_id="run-1", score=80, blocked=False, usable_claims=32),
                "run_metrics": {"total_tokens": 100000, "duration_seconds": 1200, "cost_usd": 4.5},
            }
        ),
        build_run_quality_snapshot(
            {
                **_writer_package(run_id="run-2", score=72, blocked=True, usable_claims=35),
                "run_metrics": {"total_tokens": 220000, "duration_seconds": 2800, "cost_usd": 9.2},
            }
        ),
        build_run_quality_snapshot(
            {
                **_writer_package(run_id="run-3", score=66, blocked=False, usable_claims=9, thin_report_risk=True),
                "run_metrics": {"total_tokens": 130000, "duration_seconds": 1500, "cost_usd": 5.1},
            }
        ),
    ]

    summary = summarize_topic_regression(runs, min_publish_score=70, max_tokens_per_run=180000, max_duration_seconds=2400)

    assert summary["topic_id"] == "ai-agent-enterprise"
    assert summary["run_count"] == 3
    assert summary["pass_count"] == 1
    assert summary["pass_rate"] == 1 / 3
    assert summary["score_mean"] == 72.67
    assert summary["score_stddev"] > 5
    assert summary["score_range"] == 14
    assert summary["fatal_type_counts"]["citation_semantic_mismatch"] == 1
    assert summary["thin_report_risk_count"] == 1
    assert summary["token_mean"] == 150000
    assert summary["token_max"] == 220000
    assert summary["duration_seconds_max"] == 2800
    assert summary["cost_usd_total"] == 18.8
    assert summary["stability_status"] == "unstable"
    assert "raise_pass_rate" in summary["recommended_actions"]
    assert "reduce_cost_or_latency" in summary["recommended_actions"]


def test_repair_effectiveness_tracks_closed_gaps_not_just_task_count():
    writer_package = {
        "metadata": {
            "repair_task_selection_summary": {
                "task_count": 4,
                "by_proof_role": {"metric": 2, "counter": 1, "case": 1},
            }
        },
        "score_gaps": [
            {"gap_id": "GAP-1", "status": "evidence_found", "gap_type": "metric_scope_period_unit_incomplete"},
            {"gap_id": "GAP-2", "status": "cache_satisfied", "gap_type": "case_evidence_missing"},
            {"gap_id": "GAP-3", "status": "still_insufficient", "gap_type": "counter_evidence_missing"},
            {"gap_id": "GAP-4", "status": "live_search_required", "gap_type": "metric_scope_period_unit_incomplete"},
        ],
    }
    writer_report = {
        "post_qa_repair_trace": [
            {
                "repair_result_summary": {
                    "new_usable_evidence_count": 2,
                    "new_ab_source_count": 1,
                    "signal_count": 2,
                    "empty_success_count": 1,
                }
            }
        ]
    }

    summary = summarize_repair_effectiveness(writer_package=writer_package, writer_report=writer_report)

    assert summary["attempted_gap_count"] == 4
    assert summary["closed_gap_count"] == 2
    assert summary["open_gap_count"] == 2
    assert summary["closure_rate"] == 0.5
    assert summary["new_usable_evidence_count"] == 2
    assert summary["new_ab_source_count"] == 1
    assert summary["by_gap_status"]["still_insufficient"] == 1
    assert summary["effectiveness_status"] == "partial"


def test_evidence_admission_decision_unifies_publishable_directional_and_repair_actions():
    publishable = decide_evidence_admission(
        {
            "evidence_id": "EV-A",
            "status": "validated",
            "source_level": "A",
            "proof_role": "case",
            "source_verification_status": "readpage_verified",
            "allowed_use": "writing",
        }
    )
    incomplete_metric = decide_evidence_admission(
        {
            "evidence_id": "EV-M",
            "status": "validated",
            "source_level": "B",
            "proof_role": "metric",
            "metric": "adoption rate",
            "value": "40%",
            "unit": "%",
            "period": "",
            "source_url": "https://example.org/report",
        }
    )
    rejected = decide_evidence_admission(
        {
            "evidence_id": "EV-R",
            "status": "rejected",
            "source_level": "D",
            "allowed_use": "clue",
        }
    )

    assert publishable["verdict"] == "publishable"
    assert publishable["allowed_use"] == "writing"
    assert incomplete_metric["verdict"] == "directional"
    assert "metric_fields_incomplete" in incomplete_metric["reasons"]
    assert incomplete_metric["repair_action"] == "repair_metric_fields"
    assert rejected["verdict"] == "reject"
    assert "status_rejected" in rejected["reasons"]


def test_percent_metric_value_does_not_require_separate_unit_field():
    decision = decide_evidence_admission(
        {
            "evidence_id": "EV-rate",
            "status": "validated",
            "source_level": "B",
            "proof_role": "metric",
            "metric": "adoption rate",
            "value": "40%",
            "unit": "",
            "period": "2025",
            "source_url": "https://example.org/report",
        }
    )

    assert decision["verdict"] == "publishable"
    assert "unit" not in decision["metric_missing_fields"]


def test_golden_topic_suite_requires_cross_domain_repeated_runs():
    ready = validate_golden_topic_suite(
        [
            {"topic_id": "ai-agent", "domain": "software_ai", "query": "AI agent adoption", "repeat_count": 3},
            {"topic_id": "solar", "domain": "renewable_energy", "query": "PV inverter demand", "repeat_count": 3},
            {"topic_id": "farm-machinery", "domain": "traditional_manufacturing", "query": "farm machinery export", "repeat_count": 2},
            {"topic_id": "long-tail", "domain": "niche_industry", "query": "specialized sensor supply", "repeat_count": 2},
        ],
        min_domains=3,
        min_repeat_count=2,
    )
    overfit_risk = validate_golden_topic_suite(
        [
            {"topic_id": "ai-agent-1", "domain": "software_ai", "query": "AI agent adoption", "repeat_count": 1},
            {"topic_id": "ai-agent-2", "domain": "software_ai", "query": "AI agent pricing", "repeat_count": 1},
        ],
        min_domains=3,
        min_repeat_count=2,
    )

    assert ready["status"] == "ready"
    assert ready["domain_count"] == 4
    assert ready["planned_run_count"] == 10
    assert overfit_risk["status"] == "not_ready"
    assert "insufficient_domain_coverage" in overfit_risk["issues"]
    assert "repeat_count_too_low" in overfit_risk["issues"]


def test_quality_regression_runner_loads_writer_packages_and_groups_by_topic(tmp_path):
    first = tmp_path / "run-1.writer_package.json"
    second = tmp_path / "run-2.writer_package.json"
    first.write_text(
        json.dumps(_writer_package(run_id="run-1", score=80, blocked=False, usable_claims=20), ensure_ascii=False),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps(_writer_package(run_id="run-2", score=65, blocked=True, usable_claims=18), ensure_ascii=False),
        encoding="utf-8",
    )

    snapshots = load_quality_snapshots_from_paths([first, second])
    suite = summarize_quality_regression_suite(snapshots, min_publish_score=70)

    assert len(snapshots) == 2
    assert suite["topic_count"] == 1
    assert suite["run_count"] == 2
    assert suite["topics"][0]["topic_id"] == "ai-agent-enterprise"
    assert suite["topics"][0]["pass_count"] == 1
    assert suite["overall_status"] == "unstable"


def test_quality_regression_executor_runs_topics_repeats_and_feeds_summary(tmp_path):
    topics = [
        {"topic_id": "ai-agent", "domain": "software_ai", "query": "AI agent adoption", "repeat_count": 2},
        {"topic_id": "solar", "domain": "renewable_energy", "query": "PV inverter demand", "repeat_count": 1},
    ]
    calls = []

    def fake_runner(command, *, cwd, env, timeout):
        calls.append({"command": command, "cwd": cwd, "env": env, "timeout": timeout})
        output_dir = tmp_path
        for index, value in enumerate(command):
            if value == "--output-dir":
                output_dir = Path(command[index + 1])
                break
        output_dir.mkdir(parents=True, exist_ok=True)
        topic_id = output_dir.parent.name
        repeat_index = int(output_dir.name.rsplit("_", 1)[-1])
        package = _writer_package(
            run_id=f"{topic_id}-{repeat_index}",
            score=82,
            blocked=False,
            usable_claims=24,
        )
        package["metadata"].update(
            {
                "run_id": f"{topic_id}-{repeat_index}",
                "topic_id": topic_id,
                "domain": "software_ai" if topic_id == "ai-agent" else "renewable_energy",
                "query": "AI agent adoption" if topic_id == "ai-agent" else "PV inverter demand",
            }
        )
        package["run_metrics"] = {"total_tokens": 120000, "duration_seconds": 900}
        (output_dir / "writer_package.json").write_text(json.dumps(package, ensure_ascii=False), encoding="utf-8")
        return {"returncode": 0, "stdout": "ok", "stderr": "", "duration_seconds": 1.25}

    plan = build_quality_execution_plan(topics, output_root=tmp_path, extra_args=["--supervisor-max-loops", "1"])
    result = run_quality_regression_execution(
        topics,
        output_root=tmp_path,
        runner=fake_runner,
        python_executable="python",
        timeout_seconds=3600,
        extra_args=["--supervisor-max-loops", "1"],
        min_publish_score=70,
    )

    assert len(plan) == 3
    assert len(calls) == 3
    assert all(call["command"][0] == "python" for call in calls)
    assert all("-m" in call["command"] and "rag_pipeline.flows.report.full_report" in call["command"] for call in calls)
    assert all("--no-interactive-input" in call["command"] for call in calls)
    assert result["execution_summary"]["planned_run_count"] == 3
    assert result["execution_summary"]["completed_run_count"] == 3
    assert result["suite_summary"]["run_count"] == 3
    assert result["suite_summary"]["topic_count"] == 2
    assert result["suite_summary"]["overall_status"] == "stable"
