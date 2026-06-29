from __future__ import annotations

from rag_pipeline.agents.claim_builder_agent import bind_section_fact_refs_from_claims
from rag_pipeline.agents.chapter_argument_agent import run_chapter_argument_agent
from rag_pipeline.agents.markdown_renderer import render_section


def test_section_inherits_used_fact_refs_from_claim():
    sections = [
        {
            "section_id": "s1",
            "claim_id": "c1",
            "title": "商业化场景",
            "used_fact_refs": [],
        }
    ]
    claims = [
        {
            "claim_id": "c1",
            "fact_ids": ["EV-01", "EV-02"],
            "source_ids": ["SRC-01"],
        }
    ]

    result = bind_section_fact_refs_from_claims(sections, claims)

    assert result[0]["used_fact_refs"] == ["EV-01", "EV-02"]
    assert result[0]["evidence_refs"] == ["EV-01", "EV-02"]
    assert result[0]["evidence_backed"] is True


def test_section_without_refs_is_marked_transition_only():
    sections = [
        {
            "section_id": "s2",
            "title": "过渡段",
            "claim_id": None,
            "used_fact_refs": [],
        }
    ]

    result = bind_section_fact_refs_from_claims(sections, [])

    assert result[0]["used_fact_refs"] == []
    assert result[0]["evidence_backed"] is False
    assert result[0]["section_role"] == "transition_or_synthesis"
    assert result[0]["public_fact_claim_allowed"] is False


def test_chapter_argument_drops_unreferenced_public_fact_section():
    packages = run_chapter_argument_agent(
        report_blueprint={"chapters": [{"chapter_id": "ch_01", "chapter_title": "低空经济"}]},
        micro_layouts=[
            {
                "chapter_id": "ch_01",
                "sections": [
                    {
                        "section_id": "s_unref",
                        "block_type": "integrated_signal",
                        "section_role": "integrated_signal",
                    }
                ],
            }
        ],
        argument_units=[
            {
                "chapter_id": "ch_01",
                "section_id": "s_unref",
                "claim_id": "CL-unref",
                "claim": "2025年低空经济规模超过60万亿元。",
                "reasoning": "该判断缺少可追溯事实引用。",
                "evidence_refs": [],
                "used_fact_refs": [],
                "supporting_facts": ["缺少引用的事实句。"],
                "evidence_basis": ["缺少引用的事实句。"],
                "claim_strength": "directional",
                "public_render": True,
            }
        ],
        chapter_evidence_packages=[{"chapter_id": "ch_01"}],
    )

    section = packages[0]["sections"][0]

    assert section["public_render"] is False
    assert section["omit_from_report"] is True
    assert section["composition_status"] == "dropped"
    assert section["used_fact_refs"] == []


def test_renderer_skips_unreferenced_fact_section_even_if_marked_backed():
    section = {
        "section_id": "ch_02_llm_extra_1",
        "section_role": "integrated_signal",
        "block_type": "integrated_signal",
        "claim": "A market share reached 70% in 2025.",
        "render_blocks": [{"type": "paragraph", "text": "A market share reached 70% in 2025."}],
        "used_fact_refs": [],
        "evidence_refs": [],
        "evidence_backed": True,
        "public_render": True,
    }

    assert render_section(section) == []


def test_renderer_allows_uncited_transition_section():
    section = {
        "section_id": "transition",
        "section_role": "transition_or_synthesis",
        "block_type": "transition",
        "render_blocks": [{"type": "paragraph", "text": "This chapter connects the previous findings to the next question."}],
        "used_fact_refs": [],
        "evidence_refs": [],
    }

    assert render_section(section)
