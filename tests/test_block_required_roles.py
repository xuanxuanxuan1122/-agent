from rag_pipeline.agents.block_schema import _claim_first_blocks_from_section_plan


def test_claim_first_blocks_always_declare_required_evidence_roles():
    chapter = {
        "chapter_id": "ch_01",
        "section_plan": [
            # No required_evidence_roles on the section -> must fall back to the
            # block-type's canonical roles instead of an empty list.
            {"section_id": "ch_01_s1", "section_title": "市场规模与口径", "block_type": "metric_reconciliation"},
            # Unknown block_type collapses to integrated_signal, which still has roles.
            {"section_id": "ch_01_s2", "section_title": "可验证信号", "block_type": "totally_unknown_type"},
        ],
    }

    blocks = _claim_first_blocks_from_section_plan(chapter, chapter_claims=[], limit=4)

    assert blocks
    for block in blocks:
        assert block["required_evidence_roles"], (
            f"block {block.get('block_id')} ({block.get('block_type')}) has empty required_evidence_roles"
        )


def test_claim_first_blocks_keep_explicit_section_roles():
    chapter = {
        "chapter_id": "ch_02",
        "section_plan": [
            {
                "section_id": "ch_02_s1",
                "section_title": "竞争格局",
                "block_type": "competitive_positioning",
                "required_evidence_roles": ["case", "metric"],
            }
        ],
    }

    blocks = _claim_first_blocks_from_section_plan(chapter, chapter_claims=[], limit=4)

    assert blocks[0]["required_evidence_roles"] == ["case", "metric"]
