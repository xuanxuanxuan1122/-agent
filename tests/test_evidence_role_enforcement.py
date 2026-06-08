from rag_pipeline.agents.writer_agent_clean import prepare_dimension_materials
from rag_pipeline.contracts.evidence_quality import apply_evidence_quality_contract


def test_confident_c_level_evidence_is_directional_signal():
    items = [
        {
            "metric": "市场规模",
            "value": "34亿元",
            "source_level": "C",
            "evidence_role": "clue",
            "semantic_status": "ok",
            "confidence": 0.6,
        }
    ]

    prepared = prepare_dimension_materials(items, dimension="市场规模与增速")

    assert prepared[0]["appendix_only"] is False
    assert prepared[0]["enterprise_usable"] is True
    assert prepared[0]["allowed_use"] == "directional_signal"
    assert prepared[0]["usage_tier"] == "directional_signal"


def test_d_level_kept_as_followup_clue_and_rejected_removed():
    items = [
        {"source_level": "D", "evidence_role": "clue", "semantic_status": "ok"},
        {"source_level": "A", "evidence_role": "rejected", "semantic_status": "ok"},
        {"source_level": "A", "evidence_role": "core", "semantic_status": "rejected"},
    ]

    prepared = prepare_dimension_materials(items, dimension="市场规模与增速")

    assert len(prepared) == 1
    assert prepared[0]["source_level"] == "D"
    assert prepared[0]["appendix_only"] is True
    assert prepared[0]["enterprise_usable"] is False
    assert prepared[0]["followup_seed"] is True


def test_evidence_admission_marks_c_level_traceable_as_directional_analysis():
    item = apply_evidence_quality_contract(
        {
            "source_level": "C",
            "evidence_role": "supporting",
            "semantic_status": "ok",
            "confidence": 0.68,
            "source_url": "https://news.example.cn/agent-case",
            "content": "A customer pilot described workflow deployment for an AI Agent.",
        }
    )

    assert item["analysis_eligible"] is True
    assert item["analysis_role"] == "directional"
    assert item["evidence_admission_reason"] == "directional_signal"
    assert item["evidence_admission_decision"]["verdict"] == "directional"
    assert item["admission_verdict"] == "directional"
    assert item["evidence_card"]["admission_verdict"] == "directional"


def test_evidence_admission_keeps_incomplete_metric_as_contextual_not_metric_claim():
    item = apply_evidence_quality_contract(
        {
            "source_level": "B",
            "evidence_role": "supporting",
            "semantic_status": "ok",
            "confidence": 0.8,
            "proof_role": "metric",
            "metric": "market size",
            "value": "100",
            "source_url": "https://research.example.cn/agent-market",
            "content": "The report mentions a market-size figure but does not disclose a unit or comparable scope.",
        }
    )

    assert item["analysis_eligible"] is True
    assert item["analysis_role"] == "contextual"
    assert item["evidence_admission_reason"] == "metric_scope_period_unit_incomplete"
    assert item["evidence_admission_decision"]["verdict"] == "directional"
    assert item["evidence_admission_decision"]["repair_action"] == "repair_metric_fields"
    assert "metric_fields_incomplete" in item["admission_reasons"]


def test_evidence_admission_blocks_rejected_or_untraceable_low_quality_evidence():
    item = apply_evidence_quality_contract(
        {
            "source_level": "D",
            "evidence_role": "clue",
            "semantic_status": "ok",
            "content": "A forum rumor mentions AI Agent growth.",
        }
    )

    assert item["analysis_eligible"] is False
    assert item["analysis_role"] == "rejected"
    assert item["evidence_admission_reason"] in {"appendix_only", "untraceable_or_low_quality"}
    assert item["evidence_admission_decision"]["verdict"] in {"appendix_only", "reject"}
