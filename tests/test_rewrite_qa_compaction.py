import json

from rag_pipeline.agents.rewrite_agent import _compact_qa_for_llm


def test_compact_qa_for_llm_removes_large_packages_and_keeps_actionable_findings():
    qa_result = {
        "errors": [{"type": "citation", "severity": "fatal", "message": "缺少来源附录"}],
        "warnings": [{"type": "thin_body", "message": "正文过薄"}],
        "rewrite_instructions": ["补齐引用"],
        "chapter_packages": [{"chapter_id": "ch_01", "raw": "x" * 20000}],
        "evidence_health_summary": {"raw_blob": "y" * 20000},
        "deep_evaluation": {
            "rewrite_instructions": ["压缩处理稿口吻"],
            "issues": [{"type": "style", "severity": "warning", "message": "处理稿口吻残留"}],
            "chapter_packages": [{"chapter_id": "ch_02", "raw": "z" * 20000}],
        },
    }

    compact = _compact_qa_for_llm(qa_result)
    dumped = json.dumps(compact, ensure_ascii=False)

    assert "chapter_packages" not in compact
    assert "evidence_health_summary" not in compact
    assert "chapter_packages" not in compact.get("deep_evaluation", {})
    assert "缺少来源附录" in dumped
    assert "补齐引用" in dumped
    assert len(dumped) < 30000
