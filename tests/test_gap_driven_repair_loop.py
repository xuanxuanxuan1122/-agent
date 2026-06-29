from rag_pipeline.observability.blueprint_evidence_alignment import build_blueprint_evidence_alignment


def test_starved_chapter_alignment_generates_repair_task_seed():
    blueprint = {
        "chapters": [
            {
                "chapter_id": "ch_04",
                "chapter_title": "\u76d1\u7ba1\u548c\u6280\u672f\u7ea6\u675f",
                "chapter_question": "\u76d1\u7ba1\u3001\u6280\u672f\u6210\u719f\u5ea6\u548c\u66ff\u4ee3\u65b9\u6848\u5982\u4f55\u6539\u53d8\u673a\u4f1a\u6392\u5e8f",
            }
        ]
    }

    alignment = build_blueprint_evidence_alignment(
        report_blueprint=blueprint,
        evidence_package={"analysis_ready_facts": []},
    )

    seeds = alignment["repair_task_seeds"]
    assert len(seeds) == 1
    seed = seeds[0]
    assert seed["schema_version"] == "repair_task_seed_v2"
    assert seed["chapter_id"] == "ch_04"
    assert seed["gap_type"] == "chapter_starved"
    assert seed["allowed_for_writing"] is False
    assert seed["diagnostic_only"] is True
    assert "\u76d1\u7ba1" in seed["query"] or "\u6280\u672f" in seed["query"]
    assert seed["missing_fields"] == ["fact", "source_ref"]
