from rag_pipeline.agents.claim_builder_agent import run_claim_builder_agent
from rag_pipeline.agents.public_report_sanitizer import sanitize_public_markdown


def test_missing_proof_standards_downgrade_to_public_directional_claim():
    evidence = {
        "evidence_id": "EV-001",
        "ref": "EV-001",
        "fact": "水泥价格指数环比下行，但库存去化开始改善。",
        "source_level": "C",
        "evidence_role": "clue",
        "allowed_use": "directional_signal",
        "appendix_only": False,
        "confidence": 0.62,
    }
    package = {
        "chapter_id": "ch_01",
        "chapter_title": "行情是否已经见底",
        "chapter_question": "价格、库存和需求是否共同指向底部修复？",
        "supporting_evidence": [evidence],
        "evidence_items": [evidence],
        "missing_proof_standards": [
            {"hypothesis_id": "H1", "blocking_gaps": ["counter_evidence_missing"]}
        ],
    }

    units = run_claim_builder_agent(chapter_evidence_packages=[package], micro_layouts=[])

    assert units[0]["public_render"] is True
    assert units[0]["omit_from_report"] is False
    assert units[0]["claim_status"] == "directional"
    assert "证据不足" not in units[0]["claim"]
    assert "低置信" not in units[0]["claim"]


def test_sanitizer_rewrites_internal_gap_language_instead_of_dropping_block():
    markdown = "## 1. 行情判断\n证据不足，不能作为确定性结论，但价格和库存已经出现方向性变化。"

    cleaned = sanitize_public_markdown(markdown, mode="enforce")

    assert "## 1. 行情判断" in cleaned
    assert "价格和库存已经出现方向性变化" in cleaned
    assert "证据不足" not in cleaned
    assert "不能作为确定性结论" not in cleaned
