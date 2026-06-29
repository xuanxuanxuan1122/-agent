from __future__ import annotations

from rag_pipeline.agents.analysis_agent import validate_numeric_fact_grounding


def test_malformed_year_range_is_rejected():
    claim = "中国低空经济的市场规模在2022-20年估算为5000亿元。"
    facts = [
        {
            "fact_id": "EV-1",
            "fact": "2023年市场规模达到5059.5亿元，2024年增至6702.5亿元。",
        }
    ]

    result = validate_numeric_fact_grounding(claim, facts)

    assert not result.valid
    assert "malformed_year_range" in result.reasons


def test_unsupported_large_unit_is_rejected():
    claim = "2025年低空经济规模超过60万亿元。"
    facts = [{"fact_id": "EV-1", "fact": "2024年低空经济市场规模约6702.5亿元。"}]

    result = validate_numeric_fact_grounding(claim, facts)

    assert not result.valid
    assert "unsupported_number_or_unit" in result.reasons


def test_supported_number_and_unit_passes():
    claim = "2024年低空经济市场规模约6702.5亿元。"
    facts = [{"fact_id": "EV-1", "fact": "2024年低空经济市场规模约6702.5亿元。"}]

    result = validate_numeric_fact_grounding(claim, facts)

    assert result.valid
    assert result.reasons == []
    assert "6702.5亿元" in result.numbers

