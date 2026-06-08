import inspect

from rag_pipeline.flows.report import full_report


def test_final_artifact_ledger_updates_are_written_to_state_json():
    source = inspect.getsource(full_report.main)
    finish_index = source.index("artifact_ledger_finish = finish_artifact_ledger_run_safe")
    state_write_index = source.index("write_state_json(state_path, state_dict)", finish_index)
    package_write_index = source.index("write_writer_package()", finish_index)

    assert finish_index < state_write_index < package_write_index


def test_repair_traces_are_exported_for_artifact_ledger_status_updates():
    payload = full_report.repair_trace_payload_from_state(
        state_dict={
            "evidence_preflight_trace": [
                {"gap_ledger": [{"gap_id": "GAP-pre", "status": "evidence_found"}]}
            ]
        },
        raw_output={
            "layout_refinement_trace": [
                {"gap_ledger": [{"gap_id": "GAP-layout", "status": "searched_no_signal"}]}
            ]
        },
        writer_report={
            "post_qa_repair_trace": [
                {"gap_ledger": [{"gap_id": "GAP-postqa", "status": "evidence_found"}]}
            ]
        },
    )

    assert [item["gap_ledger"][0]["gap_id"] for item in payload["evidence_preflight_trace"]] == ["GAP-pre"]
    assert [item["gap_ledger"][0]["gap_id"] for item in payload["layout_refinement_trace"]] == ["GAP-layout"]
    assert [item["gap_ledger"][0]["gap_id"] for item in payload["post_qa_repair_trace"]] == ["GAP-postqa"]
