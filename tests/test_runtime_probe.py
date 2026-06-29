import json

from rag_pipeline.flows.report.full_report import emit_runtime_stage_snapshot_probe_safe
from rag_pipeline.observability.probe_api import emit_transform
from rag_pipeline.observability.probe_context import create_probe_context
from rag_pipeline.observability.probe_runtime import summarize_runtime_events


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_runtime_probe_bus_writes_live_module_probe_events(tmp_path):
    probe = create_probe_context(run_id="run-live", output_dir=tmp_path, base_name="run-live")

    result = emit_transform(
        probe,
        stage="fact_extractor",
        module="readpage_fact_extractor_agent",
        input_count=4,
        output_count=9,
        drop_count=2,
        reason_counts={"snippet_only": 2},
        id_coverage={"requirement_id": 0.75, "source_id": 1.0},
        cache={"cache_hit_count": 1},
    )

    assert result["emitted"] is True
    assert probe.live_path.exists()
    events = _read_jsonl(probe.live_path)
    assert len(events) == 1
    event = events[0]
    assert event["schema_version"] == "module_probe_event_v1"
    assert event["event_type"] == "transform_result"
    assert event["input"]["count"] == 4
    assert event["output"]["count"] == 9
    assert event["drop"]["reason_counts"]["snippet_only"] == 2
    assert event["cache"]["hit_count"] == 1
    assert event["diagnostic_only"] is True
    assert event["must_not_render"] is True
    assert event["public_text_allowed"] is False


def test_stage_snapshot_runtime_probe_hook_is_fail_open_and_diagnostic_only(tmp_path):
    probe = create_probe_context(run_id="run-snapshot", output_dir=tmp_path, base_name="run-snapshot")

    result = emit_runtime_stage_snapshot_probe_safe(
        probe,
        stage_name="evidence_package",
        payload={"analysis_ready_evidence": [{"evidence_id": "EV-1"}]},
        snapshot_result={"stored": True, "replayable": True, "full_payload_bytes": 120},
        summary={"input": "test"},
        diagnostics={"note": "debug only"},
    )

    assert result["emitted"] is True
    events = _read_jsonl(probe.live_path)
    assert len(events) == 1
    event = events[0]
    assert event["stage"] == "stage_snapshot:evidence_package"
    assert event["module"] == "stage_snapshot_cache"
    assert event["output"]["count"] == 1
    assert event["diagnostics"]["stage_snapshot"] is True
    assert event["diagnostics"]["snapshot_result"]["replayable"] is True
    assert event["diagnostic_only"] is True


def test_runtime_probe_summary_uses_transform_events_only(tmp_path):
    probe = create_probe_context(run_id="run-summary", output_dir=tmp_path, base_name="run-summary")
    emit_transform(probe, stage="readpage", module="iqs_readpage", input_count=10, output_count=6, drop_count=4)
    emit_transform(probe, stage="claim_builder", module="claim_builder_agent", input_count=8, output_count=5, drop_count=3)

    summary = summarize_runtime_events(probe.live_path)

    assert summary["event_count"] == 2
    assert summary["stage_counts"]["readpage"]["input_count"] == 10
    assert summary["stage_counts"]["readpage"]["output_count"] == 6
    assert summary["rates"]["readpage_success_rate"] == 0.6
    assert summary["rates"]["bound_claim_rate"] == 0.625


def test_runtime_probe_sequence_increments_across_emit_calls(tmp_path):
    probe = create_probe_context(run_id="run-seq", output_dir=tmp_path, base_name="run-seq")

    emit_transform(probe, stage="readpage", module="iqs_readpage", input_count=2, output_count=2)
    emit_transform(probe, stage="fact_extractor", module="readpage_fact_extractor_agent", input_count=2, output_count=1)
    emit_transform(probe, stage="analysis_agent", module="analysis_agent", input_count=1, output_count=1)

    events = _read_jsonl(probe.live_path)

    assert [event["seq"] for event in events] == [1, 2, 3]


def test_runtime_probe_summary_uses_current_stage_names(tmp_path):
    probe = create_probe_context(run_id="run-stage-alias", output_dir=tmp_path, base_name="run-stage-alias")

    emit_transform(probe, stage="evidence_merger", module="evidence_merger", input_count=10, output_count=7)
    emit_transform(probe, stage="analysis_agent", module="analysis_agent", input_count=7, output_count=4)

    summary = summarize_runtime_events(probe.live_path)

    assert summary["rates"]["analysis_ready_rate"] == 0.7
    assert summary["rates"]["claim_conversion_rate"] == 0.5714
    assert summary["rates"]["fact_to_claim_rate"] == 0.5714


def test_runtime_probe_summary_uses_readpage_health_when_stage_event_is_missing(tmp_path):
    probe = create_probe_context(run_id="run-readpage-health", output_dir=tmp_path, base_name="run-readpage-health")

    emit_transform(
        probe,
        stage="evidence_merger",
        module="evidence_merger",
        input_count=10,
        output_count=7,
        diagnostics={
            "evidence_health_summary": {
                "readpage_attempted": 87,
                "readpage_succeeded": 86,
            }
        },
    )

    summary = summarize_runtime_events(probe.live_path)

    assert summary["rates"]["readpage_success_rate"] == 0.9885
    assert summary["metrics"]["readpage_attempted"] == 87
    assert summary["metrics"]["readpage_succeeded"] == 86


def test_runtime_probe_summary_splits_claim_conversion_rates(tmp_path):
    probe = create_probe_context(run_id="run-claim-rates", output_dir=tmp_path, base_name="run-claim-rates")

    emit_transform(probe, stage="evidence_merger", module="evidence_merger", input_count=120, output_count=100)
    emit_transform(
        probe,
        stage="analysis_agent",
        module="analysis_agent",
        input_count=100,
        output_count=18,
        drop_count=6,
        diagnostics={"llm_raw_claim_count": 24, "llm_usable_claim_count": 18},
    )
    emit_transform(probe, stage="claim_builder", module="claim_builder_agent", input_count=24, output_count=18)

    summary = summarize_runtime_events(probe.live_path)

    assert summary["rates"]["fact_to_claim_rate"] == 0.18
    assert summary["rates"]["llm_claim_acceptance_rate"] == 0.75
    assert summary["rates"]["claim_binding_rate"] == 0.75
    assert summary["metrics"]["llm_raw_claim_count"] == 24
    assert summary["metrics"]["llm_usable_claim_count"] == 18


def test_runtime_probe_summary_surfaces_final_body_char_count(tmp_path):
    probe = create_probe_context(run_id="run-body-metrics", output_dir=tmp_path, base_name="run-body-metrics")

    emit_transform(
        probe,
        stage="writer",
        module="final_writer",
        input_count=1,
        output_count=1,
        diagnostics={
            "final_body_char_count": 5290,
            "target_body_chars": 20000,
        },
    )

    summary = summarize_runtime_events(probe.live_path)

    assert summary["metrics"]["final_body_char_count"] == 5290
    assert summary["metrics"]["target_body_chars"] == 20000


def test_runtime_probe_summary_surfaces_root_hygiene_and_earliest_issue(tmp_path):
    probe = create_probe_context(run_id="run-root-hygiene", output_dir=tmp_path, base_name="run-root-hygiene")

    emit_transform(
        probe,
        stage="fact_extractor",
        module="readpage_fact_extractor_agent",
        input_count=2,
        output_count=1,
        diagnostics={
            "source_root_hygiene": {
                "dirty_item_count": 1,
                "clean_item_count": 1,
                "reason_counts": {"malformed_metric_span": 1},
            }
        },
    )
    emit_transform(
        probe,
        stage="evidence_merger",
        module="evidence_merger",
        input_count=1,
        output_count=1,
        diagnostics={
            "evidence_root_hygiene": {
                "dirty_item_count": 1,
                "missing_lineage_counts": {"requirement_id": 1},
                "reason_counts": {"artifact_like_value": 1},
            },
            "cache_root_hygiene": {
                "polluted_count": 2,
                "quarantined_count": 1,
                "dirty_hit_count": 1,
                "reason_counts": {"cache_polluted_count": 2, "cache_hit_polluted": 1},
            },
            "source_identity_hygiene": {
                "dirty_item_count": 3,
                "reason_counts": {"same_title_many_hosts": 3},
            },
        },
    )

    summary = summarize_runtime_events(probe.live_path)

    assert summary["root_hygiene"]["status"] == "warning"
    assert summary["root_hygiene"]["earliest_issue_stage"] == "fact_extractor"
    assert summary["root_hygiene"]["issue_count_by_stage"]["fact_extractor"] == 1
    assert summary["root_hygiene"]["issue_count_by_stage"]["evidence_merger"] == 9
    assert summary["root_hygiene"]["reason_counts"]["source:malformed_metric_span"] == 1
    assert summary["root_hygiene"]["reason_counts"]["evidence:artifact_like_value"] == 1
    assert summary["root_hygiene"]["reason_counts"]["cache:cache_polluted_count"] == 2
    assert summary["root_hygiene"]["reason_counts"]["source_identity:same_title_many_hosts"] == 3
    assert summary["root_hygiene"]["by_stage"]["evidence_merger"]["cache"]["polluted_count"] == 2
    assert summary["root_hygiene"]["by_stage"]["evidence_merger"]["source_identity"]["dirty_item_count"] == 3
    assert summary["root_hygiene"]["by_stage"]["evidence_merger"]["evidence"]["missing_lineage_counts"]["requirement_id"] == 1
    assert summary["diagnostic_only"] is True
