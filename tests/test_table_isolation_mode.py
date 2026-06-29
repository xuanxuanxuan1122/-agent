from __future__ import annotations

from rag_pipeline.agents.qa_agent import run_qa_agent


def test_table_quality_is_review_suggestion_when_tables_are_isolated(monkeypatch):
    monkeypatch.setenv("REPORT_TABLES_ENABLED", "false")
    monkeypatch.setenv("REPORT_TABLES_AFFECT_QA", "false")

    result = run_qa_agent(
        report_markdown=(
            "# Report\n\n"
            "## Market\n\n"
            "A supported section with evidence [1].\n\n"
            "## Sources\n\n"
            "[1] Example source | https://example.com/report\n"
        ),
        report_blueprint={"report_family": "industry_deep_report"},
        chapter_packages=[
            {
                "chapter_id": "CH-market",
                "sections": [
                    {
                        "section_id": "SEC-1",
                        "claim": "A supported section with evidence.",
                        "reasoning": "The cited source supports a cautious view.",
                        "counter_evidence": "Public evidence remains limited.",
                        "actionable": "Treat this as a directional signal.",
                        "evidence_refs": ["EV-1"],
                    }
                ],
            }
        ],
        table_packages=[
            {
                "table_id": "T-bad",
                "chapter_id": "CH-market",
                "should_render": True,
                "validation_errors": [{"type": "metric_row_missing_fields"}],
            }
        ],
    )

    assert "table_quality" not in {item.get("type") for item in result["errors"]}
    assert "bad_table_metric" not in {item.get("type") for item in result["errors"]}
    suggestions = result.get("review_suggestions", [])
    assert any(item.get("issue_type") == "table_quality" for item in suggestions)
    assert all(item.get("diagnostic_only") is True for item in suggestions)
    assert all(item.get("must_not_render") is True for item in suggestions)
