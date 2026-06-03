from __future__ import annotations

from pathlib import Path

import pytest

from rag_pipeline.cache.stage_snapshot_cache import write_stage_snapshot
from rag_pipeline.flows.report import full_report

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
import sys

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from replay_stage import replay_stage  # noqa: E402


def _source_registry():
    return [
        {
            "ref": "S1",
            "title": "Example AI Agent deployment source",
            "url": "https://www.salesforce.com/news/stories/agentforce/",
            "source_level": "B",
            "traceability_status": "traceable",
        }
    ]


def _evidence_package():
    return {
        "query": "AI Agent deployment",
        "report_blueprint": {
            "chapters": [
                {
                    "chapter_id": "ch_01",
                    "chapter_title": "AI Agent deployment",
                    "chapter_question": "Is AI Agent deployment observable?",
                }
            ],
            "report_shell": {"front_blocks": [], "back_blocks": ["appendix"]},
        },
        "analysis_ready_evidence": [
            {
                "ref": "E1",
                "evidence_id": "E1",
                "source_ref": "S1",
                "source_level": "B",
                "url": "https://www.salesforce.com/news/stories/agentforce/",
                "fact": "A vendor disclosed AI Agent workflow deployment in customer operations.",
                "distilled_fact": "A vendor disclosed AI Agent workflow deployment in customer operations.",
                "chapter_id": "ch_01",
                "proof_role": "case",
                "fact_type": "case",
                "public_fact_card": {
                    "subject": "vendor",
                    "action_or_signal": "disclosed workflow deployment",
                    "variable": "deployment depth",
                    "distilled_fact": "A vendor disclosed AI Agent workflow deployment in customer operations.",
                    "source_ref": "S1",
                    "source_level": "B",
                    "fact_type": "case",
                    "block_affinity": ["case_comparison"],
                },
            }
        ],
        "source_registry": _source_registry(),
    }


def test_replay_from_evidence_snapshot_rebuilds_missing_chapter_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("STAGE_SNAPSHOT_CACHE_ENABLED", "true")
    monkeypatch.setenv("STAGE_SNAPSHOT_CACHE_PATH", str(tmp_path / "snapshots"))
    monkeypatch.setenv("REPORT_ENABLE_FINAL_AUDIT", "false")
    run_id = "20260101_000001_partial_live"
    write_stage_snapshot("evidence_package", run_id, _evidence_package())

    result = replay_stage(run_id=run_id, from_stage="evidence_package", output_dir=tmp_path / "out")

    assert Path(result["report_path"]).exists()
    assert Path(result["score_path"]).exists()
    assert result["stage_snapshot_replay"]["from_stage"] == "evidence_package"


def test_fail_open_rebuild_from_state_writes_formal_artifacts_when_evidence_exists(monkeypatch):
    def fake_rebuild_from_package(*, query, state, timeout_context):
        return {
            "answer_text": "# Report\n\nEvidence-backed report.",
            "writer_report": {
                "report_markdown": "# Report\n\nEvidence-backed report.",
                "report_status": "formal_scored",
                "quality_score": 70,
            },
            "raw_output": {},
            "evidence_package": _evidence_package(),
            "metadata": {"source": "fake"},
        }

    monkeypatch.setattr(full_report, "_run_fail_open_rebuild_from_package", fake_rebuild_from_package)
    result = full_report.run_fail_open_rebuild_from_state(
        query="AI Agent deployment",
        state={"evidence_package": _evidence_package()},
        timeout_context={"max_seconds": 60, "timeout_stage": "brain"},
    )

    assert result["answer_text"]
    assert result["writer_report"]["live_timeout"]["fail_open_path_used"] is True
    assert result["writer_report"]["live_timeout"]["partial_artifact_used"] == "evidence_package"


def test_fail_open_rebuild_without_evidence_returns_diagnostic_only():
    result = full_report.run_fail_open_rebuild_from_state(
        query="AI Agent deployment",
        state={"raw_output": {"search_results": [{"title": "not enough"}]}},
        timeout_context={"max_seconds": 60, "timeout_stage": "brain"},
    )

    assert result["answer_text"] == ""
    assert result["writer_report"]["report_status"] == "diagnostic_only"
    assert result["writer_report"]["live_timeout"]["fail_open_path_used"] is False
    assert "diagnostic_markdown" in result["writer_report"]


def test_timeout_score_markdown_records_fail_open_status():
    markdown = full_report.render_score_markdown(
        query="AI Agent",
        writer_report={
            "report_status": "formal_scored",
            "quality_score": 70,
            "live_timeout": {
                "live_deadline_seconds": 60,
                "timeout_triggered": True,
                "timeout_stage": "brain",
                "fail_open_path_used": True,
                "partial_artifact_used": "evidence_package",
            },
        },
        writer_package={},
        final_audit_result={},
        reformatter_result={},
    )

    assert "Live Timeout / Fail-Open" in markdown
    assert "timeout_triggered: True" in markdown
    assert "partial_artifact_used: evidence_package" in markdown
