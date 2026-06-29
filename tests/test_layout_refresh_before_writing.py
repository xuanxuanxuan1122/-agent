from rag_pipeline.agents.blueprint_refresh import attach_staged_blueprint_refresh, build_staged_blueprint_refresh


def _blueprint():
    return {
        "chapters": [
            {"chapter_id": "ch_01", "chapter_title": "\u5e02\u573a\u9700\u6c42"},
            {"chapter_id": "ch_02", "chapter_title": "\u7ade\u4e89\u683c\u5c40"},
        ]
    }


def test_layout_refresh_suggests_repair_for_starved_chapter_before_writing():
    alignment = {
        "schema_version": "blueprint_evidence_alignment_v1",
        "chapters": {
            "ch_01": {"chapter_id": "ch_01", "status": "ok", "warnings": [], "claimable_fact_count": 4},
            "ch_02": {"chapter_id": "ch_02", "status": "starved", "warnings": ["chapter_starved"], "claimable_fact_count": 0},
        },
        "repair_task_seeds": [{"gap_id": "blueprint:ch_02:chapter_starved", "chapter_id": "ch_02"}],
    }

    refresh = build_staged_blueprint_refresh(
        report_blueprint=_blueprint(),
        alignment=alignment,
        phase="post_evidence_merge",
        writer_started=False,
    )

    assert refresh["schema_version"] == "staged_blueprint_refresh_v1"
    assert refresh["blueprint_version"] == "blueprint_v1"
    assert refresh["blueprint"]["metadata"]["blueprint_version"] == "blueprint_v1"
    assert refresh["suggestions"][0]["action"] == "repair_before_writing"
    assert refresh["suggestions"][0]["chapter_id"] == "ch_02"
    assert refresh["repair_task_seeds"][0]["chapter_id"] == "ch_02"


def test_layout_refresh_does_not_modify_locked_blueprint_after_writer_starts():
    alignment = {
        "chapters": {
            "ch_02": {"chapter_id": "ch_02", "status": "starved", "warnings": ["chapter_starved"]},
        }
    }
    locked = _blueprint()

    refresh = build_staged_blueprint_refresh(
        report_blueprint=locked,
        alignment=alignment,
        phase="post_writer_audit",
        writer_started=True,
    )

    assert refresh["blueprint_version"] == "blueprint_locked"
    assert refresh["blueprint"] == locked
    assert refresh["suggestions"][0]["action"] == "next_run_blueprint_suggestion"
    assert refresh["writer_locked"] is True


def test_attach_staged_blueprint_refresh_writes_diagnostics_and_repair_seeds():
    evidence_package = {"analysis_ready_facts": []}

    refresh = attach_staged_blueprint_refresh(
        report_blueprint={
            "chapters": [
                {
                    "chapter_id": "ch_04",
                    "chapter_title": "\u76d1\u7ba1\u548c\u6280\u672f\u7ea6\u675f",
                    "chapter_question": "\u76d1\u7ba1\u548c\u6280\u672f\u7ea6\u675f\u662f\u5426\u6539\u53d8\u673a\u4f1a\u6392\u5e8f",
                }
            ]
        },
        evidence_package=evidence_package,
        phase="post_evidence_merge",
        writer_started=False,
    )

    metadata = evidence_package["metadata"]
    assert metadata["blueprint_refresh"]["schema_version"] == "staged_blueprint_refresh_v1"
    assert metadata["blueprint_evidence_alignment"]["chapter_starved_count"] == 1
    assert evidence_package["blueprint_refresh"] == refresh
    assert evidence_package["blueprint_evidence_alignment"]["repair_task_seed_count"] == 1
    assert evidence_package["evidence_gap_ledger"][0]["source_stage"] == "blueprint_evidence_alignment"
    assert evidence_package["evidence_gap_ledger"][0]["allowed_for_writing"] is False
    assert evidence_package["evidence_repair_priorities"][0]["gap_type"] == "chapter_starved"
