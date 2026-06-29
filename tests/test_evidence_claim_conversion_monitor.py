from rag_pipeline.observability.evidence_claim_conversion import build_evidence_claim_conversion_monitor


def test_evidence_claim_conversion_monitor_explains_chapter_level_losses():
    package = {
        "evidence_package": {
            "analysis_ready_evidence": [
                {"evidence_id": "EV-1", "chapter_id": "ch_01", "requirement_id": "REQ-1", "source_id": "SRC-1"},
                {"evidence_id": "EV-2", "chapter_id": "ch_02", "source_id": "SRC-2"},
                {"evidence_id": "EV-3", "requirement_id": "REQ-3", "source_id": "SRC-3"},
            ]
        },
        "structured_analysis": {
            "claim_units": [
                {
                    "claim_id": "CL-1",
                    "chapter_id": "ch_01",
                    "fact_ids": ["EV-1"],
                    "source_ids": ["SRC-1"],
                    "requirement_ids": ["REQ-1"],
                },
                {
                    "claim_id": "CL-2",
                    "chapter_id": "ch_02",
                    "fact_ids": [],
                    "source_ids": [],
                    "unresolved_refs": ["EV-04-L22"],
                },
            ]
        },
        "chapter_packages": [
            {
                "chapter_id": "ch_01",
                "sections": [{"section_id": "SEC-1", "claim_id": "CL-1", "used_fact_refs": ["EV-1"]}],
            },
            {
                "chapter_id": "ch_02",
                "sections": [{"section_id": "SEC-2", "claim_id": "CL-2", "used_fact_refs": []}],
            },
        ],
    }

    monitor = build_evidence_claim_conversion_monitor(writer_package=package)

    assert monitor["schema_version"] == "evidence_claim_conversion_monitor_v1"
    assert monitor["totals"]["analysis_ready_facts"] == 3
    assert monitor["totals"]["bound_claims"] == 1
    assert monitor["totals"]["backed_sections"] == 1
    assert monitor["loss_reason_counts"]["no_chapter_match"] == 1
    assert monitor["loss_reason_counts"]["no_requirement_id"] == 1
    assert monitor["loss_reason_counts"]["claim_unbound"] == 1
    assert monitor["loss_reason_counts"]["unresolved_L_ref"] == 1
    assert monitor["totals"]["unresolved_L_ref_count"] == 1
    assert monitor["loss_reason_counts"]["section_unbacked"] == 1
    assert monitor["chapters"]["ch_01"]["backed_sections"] == 1
    assert monitor["chapters"]["ch_02"]["loss_reasons"]["claim_unbound"] == 1


def test_evidence_claim_conversion_monitor_merges_final_chapter_aliases():
    package = {
        "evidence_package": {
            "analysis_ready_evidence": [
                {"evidence_id": "EV-1", "chapter_id": "ch_02", "requirement_id": "REQ-1", "source_id": "SRC-1"}
            ]
        },
        "structured_analysis": {
            "claim_units": [
                {
                    "claim_id": "CL-1",
                    "chapter_id": "ch_02",
                    "fact_ids": ["EV-1"],
                    "source_ids": ["SRC-1"],
                    "requirement_ids": ["REQ-1"],
                }
            ]
        },
        "final_chapters": [
            {
                "chapter_id": "CH_ch_02",
                "chapter_title": "商业化节奏",
                "chapter_id_aliases": ["ch_02"],
                "source_plan_chapter_ids": ["ch_02"],
            }
        ],
        "chapter_packages": [
            {
                "chapter_id": "CH_ch_02",
                "sections": [{"section_id": "SEC-1", "claim_id": "CL-1", "used_fact_refs": ["EV-1"]}],
            }
        ],
    }

    monitor = build_evidence_claim_conversion_monitor(writer_package=package)

    assert "ch_02" not in monitor["chapters"]
    bucket = monitor["chapters"]["CH_ch_02"]
    assert bucket["analysis_ready_facts"] == 1
    assert bucket["bound_claims"] == 1
    assert bucket["backed_sections"] == 1
    assert monitor["totals"]["analysis_input_cards"] == 1
