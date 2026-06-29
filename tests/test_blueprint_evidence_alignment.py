from rag_pipeline.observability.blueprint_evidence_alignment import build_blueprint_evidence_alignment


def test_blueprint_evidence_alignment_marks_starved_chapter():
    blueprint = {
        "chapters": [
            {"chapter_id": "ch_01", "chapter_title": "\u5e02\u573a\u9700\u6c42"},
            {"chapter_id": "ch_02", "chapter_title": "\u7ade\u4e89\u683c\u5c40"},
        ]
    }
    evidence_package = {
        "clean_facts": [
            {
                "fact_id": "F1",
                "chapter_id": "ch_01",
                "proof_role": "metric",
                "requirement_id": "H1_metric",
            }
        ]
    }

    alignment = build_blueprint_evidence_alignment(
        report_blueprint=blueprint,
        evidence_package=evidence_package,
    )

    assert alignment["schema_version"] == "blueprint_evidence_alignment_v1"
    assert alignment["chapter_count"] == 2
    assert alignment["chapter_starved_count"] == 1
    ch_02 = alignment["chapters"]["ch_02"]
    assert ch_02["status"] == "starved"
    assert "chapter_starved" in ch_02["warnings"]


def test_blueprint_evidence_alignment_marks_overloaded_and_misaligned_chapter():
    blueprint = {
        "chapters": [
            {"chapter_id": "ch_01", "chapter_title": "\u5e02\u573a\u9700\u6c42"},
            {"chapter_id": "ch_02", "chapter_title": "\u7ade\u4e89\u683c\u5c40"},
        ]
    }
    facts = [
        {
            "fact_id": f"F{index}",
            "chapter_id": "ch_01",
            "proof_role": "support",
            "title": "\u653f\u7b56\u8865\u8d34\u548c\u5ba2\u6237\u8ba2\u5355",
            "content": "\u653f\u7b56\u8865\u8d34\u548c\u5ba2\u6237\u8ba2\u5355\u63d0\u4f9b\u884c\u4e1a\u4fe1\u53f7",
        }
        for index in range(8)
    ]
    evidence_package = {"analysis_ready_facts": facts}

    alignment = build_blueprint_evidence_alignment(
        report_blueprint=blueprint,
        evidence_package=evidence_package,
        overload_ratio=0.7,
    )

    ch_01 = alignment["chapters"]["ch_01"]
    assert "chapter_overloaded" in ch_01["warnings"]
    assert "chapter_misaligned" in ch_01["warnings"]
    assert alignment["chapter_overloaded_count"] == 1
    assert alignment["chapter_misaligned_count"] == 1
