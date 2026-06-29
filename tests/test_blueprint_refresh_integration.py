from rag_pipeline.agents import brain_agent
from rag_pipeline.flows.report import full_report


def test_pre_writer_snapshot_attaches_blueprint_refresh_and_repair_seed():
    state = {
        "report_blueprint": {
            "chapters": [
                {
                    "chapter_id": "ch_04",
                    "chapter_title": "\u76d1\u7ba1\u548c\u6280\u672f\u7ea6\u675f",
                    "chapter_question": "\u76d1\u7ba1\u548c\u6280\u672f\u7ea6\u675f\u5982\u4f55\u5f71\u54cd\u4f4e\u7a7a\u7ecf\u6d4e",
                }
            ]
        }
    }
    evidence_package = {"analysis_ready_facts": []}

    brain_agent._emit_pre_writer_snapshots(state, evidence_package, {"claim_units": []})

    assert evidence_package["metadata"]["blueprint_refresh"]["blueprint_version"] == "blueprint_v1"
    assert evidence_package["metadata"]["blueprint_evidence_alignment"]["chapter_starved_count"] == 1
    assert evidence_package["evidence_gap_ledger"][0]["gap_type"] == "chapter_starved"
    assert evidence_package["evidence_gap_ledger"][0]["allowed_for_writing"] is False


def test_pre_writer_snapshot_attaches_blueprint_refresh_when_chapter_packages_exist():
    state = {
        "report_blueprint": {
            "chapters": [
                {
                    "chapter_id": "ch_01",
                    "chapter_title": "\u5e02\u573a\u9700\u6c42",
                    "chapter_question": "\u5e02\u573a\u9700\u6c42\u662f\u5426\u53ef\u9a8c\u8bc1",
                }
            ]
        }
    }
    evidence_package = {
        "analysis_ready_facts": [],
        "chapter_evidence_packages": [{"chapter_id": "ch_01", "chapter_title": "\u5e02\u573a\u9700\u6c42"}],
    }

    brain_agent._emit_pre_writer_snapshots(state, evidence_package, {"claim_units": []})

    assert evidence_package["metadata"]["blueprint_evidence_alignment"]["chapter_starved_count"] == 1
    assert evidence_package["blueprint_refresh"]["blueprint_version"] == "blueprint_v1"


def test_full_report_blueprint_refresh_helper_attaches_to_evidence_package():
    evidence_package = {"analysis_ready_facts": []}

    full_report._attach_blueprint_refresh_to_evidence_package(
        report_blueprint={
            "chapters": [
                {
                    "chapter_id": "ch_01",
                    "chapter_title": "\u5e02\u573a\u9700\u6c42",
                    "chapter_question": "\u5e02\u573a\u9700\u6c42\u662f\u5426\u53ef\u9a8c\u8bc1",
                }
            ]
        },
        evidence_package=evidence_package,
        phase="post_evidence_merge",
    )

    assert evidence_package["metadata"]["blueprint_refresh"]["blueprint_version"] == "blueprint_v1"
    assert evidence_package["evidence_gap_ledger"][0]["source_stage"] == "blueprint_evidence_alignment"
