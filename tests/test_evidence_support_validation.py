from __future__ import annotations

from rag_pipeline.contracts.evidence_support_validation import (
    incomplete_metric_cards_for_numeric_claim,
    validate_claim_supported_by_facts,
)


def test_numeric_support_requires_exact_number_token_not_substring():
    result = validate_claim_supported_by_facts(
        "The adoption rate reached 40% in the sample.",
        [
            {
                "fact": "A different source reports 140% growth for a different metric.",
                "source_title": "Growth report",
            }
        ],
    )

    assert result.supported is False
    assert result.unsupported_numbers == ["40%"]


def test_numeric_support_matches_exact_normalized_number_token():
    result = validate_claim_supported_by_facts(
        "The adoption rate reached 40% in the sample.",
        [
            {
                "fact": "The same source states the adoption rate reached 40%.",
                "source_title": "Adoption report",
            }
        ],
    )

    assert result.supported is True


def test_numeric_support_matches_percent_spacing_decimal_and_chinese_percent():
    for fact in (
        "The same source states the adoption rate reached 40 %.",
        "The same source states the adoption rate reached 0.4.",
        "The same source states adoption reached \u767e\u5206\u4e4b\u56db\u5341.",
    ):
        result = validate_claim_supported_by_facts(
            "The adoption rate reached 40% in the sample.",
            [{"fact": fact, "source_title": "Adoption report"}],
        )

        assert result.supported is True, fact


def test_percent_metric_value_does_not_count_as_missing_unit():
    gaps = incomplete_metric_cards_for_numeric_claim(
        "The adoption rate reached 40% in 2025.",
        [
            {
                "evidence_id": "EV-rate",
                "fact_type": "metric",
                "metric": "adoption rate",
                "value": "40%",
                "unit": "",
                "period": "2025",
                "source_url": "https://example.org/report",
            }
        ],
    )

    assert gaps == []


def test_chinese_qualitative_claim_requires_material_semantic_anchor_overlap():
    result = validate_claim_supported_by_facts(
        "企业级AI Agent竞争格局正在向场景落地和渠道生态分化。",
        [
            {
                "fact": "统计部门通过官网、统计年鉴和官方社交媒体发布AI Agent相关统计数据。",
                "source_title": "统计发布渠道说明",
            }
        ],
    )

    assert result.supported is False
    assert "竞争格局" in result.unsupported_terms
    assert "场景落地" in result.unsupported_terms


def test_chinese_qualitative_claim_with_shared_material_anchors_is_supported():
    result = validate_claim_supported_by_facts(
        "企业级AI Agent竞争格局正在向场景落地和渠道生态分化。",
        [
            {
                "fact": "报告指出，企业级AI Agent的竞争格局开始围绕场景落地、渠道生态和交付能力展开。",
                "source_title": "企业级AI Agent竞争格局报告",
            }
        ],
    )

    assert result.supported is True
