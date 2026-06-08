from __future__ import annotations

from rag_pipeline.contracts.report_contract import build_report_contract_from_package
from rag_pipeline.contracts.requirement_quality import validate_requirement_quality


def test_metric_requirement_quality_rejects_generic_field_only():
    result = validate_requirement_quality(
        {
            "requirement_id": "H1_market",
            "chapter_id": "ch_01",
            "proof_role": "metric",
            "required_fields": ["market_size"],
            "min_source_level": "C",
        }
    )

    assert result["status"] == "needs_repair"
    issue_types = {item["type"] for item in result["issues"]}
    assert "metric_missing_required_fields" in issue_types
    assert "metric_requires_ab_source_level" in issue_types
    assert result["suggested_required_fields"] == ["metric", "value", "unit", "period", "scope", "source_ref"]


def test_report_contract_attaches_requirement_quality_and_source_strategy():
    contract = build_report_contract_from_package(
        {
            "query": "AI Agent workflow adoption",
            "evidence_package": {
                "metadata": {
                    "research_plan": {
                        "query": "AI Agent workflow adoption",
                        "chapters": [
                            {
                                "chapter_id": "ch_01",
                                "chapter_title": "Workflow demand",
                                "required_evidence_roles": ["metric", "counter"],
                                "minimum_source_level": "B",
                            }
                        ],
                    }
                }
            },
        }
    )

    requirements = contract["evidence_requirements"]["requirements"]
    metric = next(item for item in requirements if item["proof_role"] == "metric")
    counter = next(item for item in requirements if item["proof_role"] == "counter")

    assert metric["requirement_quality_check"]["status"] == "pass"
    assert metric["source_strategy"]["source_priority"][:2] == ["official_data", "market_research"]
    assert "snippet_only" in metric["reject_if"]
    assert "metric/value/unit/period" in metric["success_criteria"]
    assert counter["source_strategy"]["query_enhancers"][:3] == ["failure", "cost", "ROI unclear"]
    assert counter["claim_strength_ceiling"] == "directional"
