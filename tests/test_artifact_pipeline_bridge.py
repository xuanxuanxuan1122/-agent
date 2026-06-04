from __future__ import annotations

from rag_pipeline.cache.artifact_pipeline_bridge import ingest_writer_package_artifacts
from rag_pipeline.cache.artifact_store import ArtifactStore
from rag_pipeline.context.context_view_builder import build_writer_context_view


def test_bridge_ingests_requirement_to_section_and_score_gap_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_LEDGER_PATH", str(tmp_path / "artifact_ledger.sqlite"))
    monkeypatch.setenv("ARTIFACT_OBJECT_ROOT", str(tmp_path / "objects"))
    store = ArtifactStore()
    store.upsert_run(run_id="run-smoke", query="AI Agent", status="running")

    writer_package = {
        "evidence_package": {
            "report_contract": {
                "evidence_requirements": {
                    "requirements": [
                        {
                            "requirement_id": "H1_case",
                            "chapter_id": "ch_01",
                            "proof_role": "case",
                            "required_fields": ["company", "use_case"],
                            "claim_strength_ceiling": "directional",
                        }
                    ]
                }
            },
            "search_task_schedule": {
                "tasks": [{"task_id": "ST-1", "requirement_id": "H1_case", "query": "official customer case"}]
            },
            "analysis_ready_evidence": [
                {
                    "evidence_id": "EV-1",
                    "requirement_id": "H1_case",
                    "search_task_id": "ST-1",
                    "source_id": "SRC-1",
                    "fact": "Company A disclosed an AI agent workflow deployment.",
                    "source_level": "A",
                    "source": {"id": "SRC-1", "url": "https://example.com/case", "title": "Official case"},
                }
            ],
        },
        "source_registry": [{"id": "SRC-1", "url": "https://example.com/case", "title": "Official case"}],
        "argument_units": [
            {
                "claim_id": "CL-1",
                "claim": "Company A has disclosed a workflow deployment.",
                "requirement_ids": ["H1_case"],
                "used_fact_refs": ["EV-1"],
                "source_ids": ["SRC-1"],
                "claim_strength": "directional",
                "claim_strength_ceiling": "directional",
            }
        ],
        "chapter_packages": [
            {
                "chapter_id": "ch_01",
                "sections": [
                    {
                        "section_id": "SEC-1",
                        "claim_id": "CL-1",
                        "requirement_ids": ["H1_case"],
                        "used_fact_refs": ["EV-1"],
                        "evidence_backed": True,
                    }
                ],
            }
        ],
    }
    writer_report = {
        "qa_result": {
            "quality_findings": [
                {
                    "requirement_id": "H1_case",
                    "section_id": "SEC-1",
                    "gap_type": "case_boundary_missing",
                    "missing": ["limitation boundary"],
                    "severity": "medium",
                }
            ]
        }
    }

    summary = ingest_writer_package_artifacts(
        store,
        run_id="run-smoke",
        writer_package=writer_package,
        writer_report=writer_report,
    )

    assert summary["requirement_count"] == 1
    assert summary["fact_card_count"] == 1
    assert summary["claim_unit_count"] == 1
    assert summary["section_count"] == 1
    assert summary["score_gap_count"] == 1
    edge_targets = {
        (edge["to_type"], edge["to_id"])
        for edge in store.traverse_lineage("run-smoke", "requirement", "H1_case", max_depth=4)
    }
    assert ("section", "SEC-1") in edge_targets
    assert any(target_type == "score_gap" for target_type, _target_id in edge_targets)

    view = build_writer_context_view("run-smoke", "SEC-1")
    assert view["status"] == "ready"
    assert view["used_fact_refs"] == ["EV-1"]
