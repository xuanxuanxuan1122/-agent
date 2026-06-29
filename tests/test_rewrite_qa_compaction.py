import json

from rag_pipeline.agents.rewrite_agent import _compact_qa_for_llm, _downgrade_semantic_claims
from rag_pipeline.agents.writer_agent_clean import _reconcile_rewritten_candidate_markdown


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


def test_semantic_downgrade_does_not_insert_fixed_boundary_template():
    text = "# 报告\n\n阶段性判断显示机会正在形成。"

    rewritten = _downgrade_semantic_claims(text)

    assert "现有公开材料更适合支持方向性观察" not in rewritten
    assert rewritten.startswith("# 报告")


def test_rewritten_candidate_is_reconciled_before_adoption(monkeypatch):
    calls = []

    def fake_reconcile(markdown, *, citation_manifest, source_registry, appendix_package):
        calls.append(
            {
                "markdown": markdown,
                "citation_manifest": citation_manifest,
                "source_registry": source_registry,
                "appendix_package": appendix_package,
            }
        )
        return (
            "Body [1]\n\n## Sources\n- [1] Source A",
            [{"ref": "[1]", "title": "Source A", "url": "https://example.org/a"}],
            {"final_citation_reconciliation_status": "ok"},
        )

    monkeypatch.setattr(
        "rag_pipeline.agents.writer_agent_clean.reconcile_final_markdown_appendix",
        fake_reconcile,
    )

    markdown, sources, audit = _reconcile_rewritten_candidate_markdown(
        "Body [9]",
        writer_output={
            "citation_manifest": {"appendix_sources": [{"ref": "[9]", "title": "Source A"}]},
            "final_citation_audit": {"previous": True},
        },
        source_registry=[{"ref": "[9]", "title": "Source A", "url": "https://example.org/a"}],
        appendix_payload={"appendix": True},
    )

    assert calls
    assert calls[0]["markdown"] == "Body [9]"
    assert markdown.startswith("Body [1]")
    assert sources[0]["ref"] == "[1]"
    assert audit["final_citation_reconciliation_status"] == "ok"
