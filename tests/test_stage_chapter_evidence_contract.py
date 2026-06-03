from __future__ import annotations

from rag_pipeline.agents.chapter_evidence_builder import build_chapter_evidence_packages_from_evidence_package


def test_traceable_bc_case_evidence_hydrates_directional_or_case_layer():
    source_registry = [
        {
            "ref": "S1",
            "url": "https://research.example/agent-case",
            "title": "Agent case study",
            "source_level": "C",
            "traceability_status": "traceable",
        }
    ]
    evidence_package = {
        "analysis_ready_evidence": [
            {
                "ref": "E1",
                "source_ref": "S1",
                "source_level": "C",
                "url": "https://research.example/agent-case",
                "fact": "Enterprise customer case for AI Agent workflow automation.",
                "chapter_id": "ch_01",
                "proof_role": "case",
                "allowed_use": "directional_signal",
            }
        ],
        "source_registry": source_registry,
    }
    report_blueprint = {
        "chapters": [
            {
                "chapter_id": "ch_01",
                "chapter_title": "Enterprise AI Agent Demand",
                "chapter_question": "Which customer cases indicate AI Agent demand?",
            }
        ]
    }

    packages = build_chapter_evidence_packages_from_evidence_package(
        report_blueprint=report_blueprint,
        evidence_package=evidence_package,
        source_registry=source_registry,
    )

    assert len(packages) == 1
    package = packages[0]
    assert package["hydrated_evidence"] is True
    assert package["case_evidence_count"] >= 1
    assert package["core_evidence_count"] == 0
    assert package["case_evidence"][0]["source_traceable"] is True


def test_unresolved_refs_are_diagnostic_not_silent_drop():
    evidence_package = {
        "analysis_ready_evidence": [],
        "evidence_analysis_by_chapter": {
            "ch_01": {
                "sample_evidence_refs": ["missing-ref"],
            }
        },
    }
    report_blueprint = {"chapters": [{"chapter_id": "ch_01", "chapter_title": "Demand"}]}

    packages = build_chapter_evidence_packages_from_evidence_package(
        report_blueprint=report_blueprint,
        evidence_package=evidence_package,
        source_registry=[],
    )

    assert packages[0]["hydrated_evidence"] is False
    assert packages[0]["unresolved_evidence_refs"] == ["missing-ref"]
