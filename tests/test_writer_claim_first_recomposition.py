from __future__ import annotations

from rag_pipeline.agents.block_schema import select_blocks_for_chapter
from rag_pipeline.agents.chapter_argument_agent import run_chapter_argument_agent
from rag_pipeline.agents.claim_builder_agent import run_claim_builder_agent
from rag_pipeline.agents.micro_layout_agent import run_micro_layout_agent
from rag_pipeline.agents.writer_agent_clean import (
    _sync_chapter_section_refs_from_argument_units,
    build_writer_report,
    needs_public_rebuild,
)


def test_writer_uses_claim_first_final_chapters_instead_of_empty_plan_chapters(monkeypatch):
    monkeypatch.setenv("REPORT_BLUEPRINT_SOURCE", "claim_first")
    monkeypatch.setenv("REPORT_TABLES_ENABLED", "false")

    report = build_writer_report(
        query="AI Agent adoption",
        report_blueprint={
            "chapters": [
                {"chapter_id": "plan_market", "chapter_title": "Plan market"},
                {"chapter_id": "plan_empty", "chapter_title": "Plan empty"},
            ]
        },
        evidence_package={
            "analysis_ready_evidence": [
                {
                    "evidence_id": "F1",
                    "source_id": "S1",
                    "data_point": "Workflow adoption signal",
                    "source": {"url": "https://example.com/report"},
                }
            ],
            "source_registry": [{"source_id": "S1", "ref": "[1]", "url": "https://example.com/report"}],
        },
        structured_analysis={
            "claim_units": [
                {
                    "claim_id": "CL1",
                    "claim": "Workflow adoption is a directional market signal.",
                    "chapter_id": "plan_market",
                    "cluster_key": "market",
                    "fact_ids": ["F1"],
                    "source_ids": ["S1"],
                    "claim_strength": "directional",
                    "can_anchor_section": True,
                }
            ]
        },
        source_registry=[{"source_id": "S1", "ref": "[1]", "url": "https://example.com/report"}],
    )

    assert report["chapter_recomposition"]["status"] == "claim_first"
    assert len(report["plan_blueprint"]["chapters"]) == 2
    assert len(report["final_chapters"]) == 1
    assert len(report["report_blueprint"]["chapters"]) == 1
    assert report["report_blueprint"]["chapters"][0]["claim_ids"] == ["CL1"]
    assert report["chapter_narrative_plan"]["schema_version"] == "chapter_narrative_plan_v1"
    assert report["render_artifacts"]["chapter_narrative_plan"]["schema_version"] == "chapter_narrative_plan_v1"
    assert report["table_isolation_summary"]["tables_isolated"] is True


def test_claim_first_section_plan_drives_micro_layout_blocks():
    blocks = select_blocks_for_chapter(
        {
            "chapter_id": "CH_market",
            "chapter_title": "市场空间与商业化节奏",
            "chapter_role": "claim_driven_final_chapter",
            "section_plan": [
                {
                    "section_id": "CH_market_sec_01",
                    "claim_id": "CL1",
                    "section_title": "春节租赁验证短期付费需求",
                    "required_evidence_refs": ["EV-1"],
                    "matched_llm_claim": {
                        "claim_id": "CL1",
                        "claim": "春节场景付费租赁显示短期需求已经出现。",
                        "fact_ids": ["EV-1"],
                    },
                },
                {
                    "section_id": "CH_market_sec_02",
                    "claim_id": "CL2",
                    "section_title": "政策文件提供制度环境",
                    "required_evidence_refs": ["EV-2"],
                    "matched_llm_claim": {
                        "claim_id": "CL2",
                        "claim": "政策文件为人形机器人商业化提供制度环境。",
                        "fact_ids": ["EV-2"],
                    },
                },
            ],
            "layout_policy": {"block_selection_source": "claim_units"},
        },
        evidence_package={"chapter_id": "CH_market"},
        claim_units_by_chapter={
            "CH_market": [
                {"claim_id": "CL1", "claim": "春节场景付费租赁显示短期需求已经出现。", "fact_ids": ["EV-1"]},
                {"claim_id": "CL2", "claim": "政策文件为人形机器人商业化提供制度环境。", "fact_ids": ["EV-2"]},
            ]
        },
    )

    assert [block["block_id"] for block in blocks[:2]] == ["CH_market_sec_01", "CH_market_sec_02"]
    assert [block["claim_id"] for block in blocks[:2]] == ["CL1", "CL2"]
    assert all(block["selection_reason"] == "claim_first_section_plan" for block in blocks[:2])
    assert blocks[0]["required_evidence_refs"] == ["EV-1"]


def test_claim_first_micro_layout_keeps_distinct_claims_sharing_evidence_ref():
    layouts = run_micro_layout_agent(
        report_blueprint={
            "chapters": [
                {
                    "chapter_id": "CH_market",
                    "chapter_title": "市场空间与商业化节奏",
                    "chapter_role": "claim_driven_final_chapter",
                    "layout_policy": {"block_selection_source": "claim_units"},
                    "section_plan": [
                        {
                            "section_id": "CH_market_sec_01",
                            "claim_id": "CL1",
                            "section_title": "需求已经出现",
                            "required_evidence_refs": ["EV-1"],
                            "matched_llm_claim": {
                                "claim_id": "CL1",
                                "claim": "春节租赁显示短期需求已经出现。",
                                "fact_ids": ["EV-1"],
                            },
                        },
                        {
                            "section_id": "CH_market_sec_02",
                            "claim_id": "CL2",
                            "section_title": "需求仍有季节性边界",
                            "required_evidence_refs": ["EV-1"],
                            "matched_llm_claim": {
                                "claim_id": "CL2",
                                "claim": "同一春节租赁证据也说明需求仍集中在节庆场景。",
                                "fact_ids": ["EV-1"],
                            },
                        },
                    ],
                }
            ]
        },
        chapter_evidence_packages=[
            {
                "chapter_id": "CH_market",
                "chapter_title": "市场空间与商业化节奏",
                "core_evidence": [
                    {
                        "ref": "EV-1",
                        "evidence_id": "EV-1",
                        "proof_role": "support",
                        "source_level": "B",
                        "public_fact_card": {"fact_type": "support", "distilled_fact": "春节租赁活跃。"},
                    }
                ],
            }
        ],
        structured_analysis={
            "claim_units": [
                {"claim_id": "CL1", "chapter_id": "CH_market", "claim": "春节租赁显示短期需求已经出现。", "fact_ids": ["EV-1"]},
                {"claim_id": "CL2", "chapter_id": "CH_market", "claim": "同一春节租赁证据也说明需求仍集中在节庆场景。", "fact_ids": ["EV-1"]},
            ]
        },
    )

    assert [section["section_id"] for section in layouts[0]["sections"]] == [
        "CH_market_sec_01",
        "CH_market_sec_02",
    ]


def test_claim_first_claim_builder_keeps_all_matched_claims(monkeypatch):
    monkeypatch.setenv("REPORT_BLUEPRINT_SOURCE", "claim_first")

    units = run_claim_builder_agent(
        chapter_evidence_packages=[
            {
                "chapter_id": "CH_market",
                "chapter_title": "市场空间与商业化节奏",
                "core_evidence": [
                    {
                        "evidence_id": f"EV-{index}",
                        "ref": f"EV-{index}",
                        "source_id": "S1",
                        "source_ref": "[1]",
                        "proof_role": "support",
                        "source_level": "B",
                        "fact": f"事实 {index}",
                    }
                    for index in range(1, 6)
                ],
            }
        ],
        structured_analysis={
            "claim_units": [
                {
                    "claim_id": f"CL{index}",
                    "chapter_id": "CH_market",
                    "claim": f"判断 {index} 可以用于正文展开。",
                    "fact_ids": [f"EV-{index}"],
                    "source_ids": ["S1"],
                    "supporting_facts": [f"事实 {index}"],
                    "claim_strength": "directional",
                }
                for index in range(1, 6)
            ]
        },
    )

    public_ids = [unit["claim_id"] for unit in units if unit.get("public_render") and not unit.get("omit_from_report")]
    assert public_ids == ["CL1", "CL2", "CL3", "CL4", "CL5"]


def test_claim_first_claim_builder_keeps_all_matched_claims_with_sparse_layout(monkeypatch):
    monkeypatch.setenv("REPORT_BLUEPRINT_SOURCE", "claim_first")
    monkeypatch.setenv("REPORT_EXTRA_LLM_CLAIMS_PER_CHAPTER", "1")

    units = run_claim_builder_agent(
        chapter_evidence_packages=[
            {
                "chapter_id": "CH_market",
                "chapter_title": "Market signals",
                "chapter_role": "claim_driven_final_chapter",
                "layout_policy": {"block_selection_source": "claim_units"},
                "core_evidence": [
                    {
                        "evidence_id": f"EV-{index}",
                        "ref": f"EV-{index}",
                        "source_id": "S1",
                        "source_ref": "[1]",
                        "proof_role": "support",
                        "source_level": "B",
                        "fact": f"Fact {index}",
                    }
                    for index in range(1, 6)
                ],
            }
        ],
        micro_layouts=[
            {
                "chapter_id": "CH_market",
                "sections": [
                    {"section_id": "CH_market_s1", "block_type": "thesis", "section_title": "Claim 1"},
                    {"section_id": "CH_market_s2", "block_type": "mechanism", "section_title": "Claim 2"},
                ],
            }
        ],
        structured_analysis={
            "claim_units": [
                {
                    "claim_id": f"CL{index}",
                    "chapter_id": "CH_market",
                    "section_id": f"CL{index}_section",
                    "claim": f"Claim {index} should remain available for the public report.",
                    "fact_ids": [f"EV-{index}"],
                    "source_ids": ["S1"],
                    "supporting_facts": [f"Fact {index}"],
                    "claim_strength": "directional",
                }
                for index in range(1, 6)
            ]
        },
    )

    public_ids = [
        unit.get("claim_id")
        for unit in units
        if unit.get("claim_id") and unit.get("public_render") and not unit.get("omit_from_report")
    ]
    assert public_ids == ["CL1", "CL2", "CL3", "CL4", "CL5"]


def test_claim_first_handoff_uses_claim_ids_when_final_chapter_id_is_synthetic(monkeypatch):
    monkeypatch.setenv("REPORT_BLUEPRINT_SOURCE", "claim_first")
    monkeypatch.setenv("REPORT_EXTRA_LLM_CLAIMS_PER_CHAPTER", "4")

    final_chapter = {
        "chapter_id": "CH_contextual_claim",
        "chapter_title": "Contextual commercialization signals",
        "chapter_role": "claim_driven_final_chapter",
        "claim_ids": ["CL-context", "CL-market"],
        "chapter_id_aliases": ["CH_contextual_claim", "ch_01", "ch_02"],
        "source_plan_chapter_ids": ["ch_01", "ch_02"],
        "section_plan": [
            {
                "section_id": "CH_contextual_claim_sec_01",
                "claim_id": "CL-context",
                "section_title": "Consumer interest is an early signal",
                "block_type": "integrated_signal",
                "required_evidence_refs": ["EV-1"],
            },
            {
                "section_id": "CH_contextual_claim_sec_02",
                "claim_id": "CL-market",
                "section_title": "Market sizing supports commercialization",
                "block_type": "integrated_signal",
                "required_evidence_refs": ["EV-2"],
            },
        ],
    }
    evidence_package = {
        "chapter_id": "CH_contextual_claim",
        "chapter_title": "Contextual commercialization signals",
        "chapter_role": "claim_driven_final_chapter",
        "claim_ids": ["CL-context", "CL-market"],
        "chapter_id_aliases": ["CH_contextual_claim", "ch_01", "ch_02"],
        "source_plan_chapter_ids": ["ch_01", "ch_02"],
        "section_plan": final_chapter["section_plan"],
        "core_evidence": [
            {
                "evidence_id": "EV-1",
                "ref": "EV-1",
                "source_id": "S1",
                "source_ref": "[1]",
                "proof_role": "signal",
                "source_level": "C",
                "fact": "Surveyed consumers show early interest in low-altitude services.",
            },
            {
                "evidence_id": "EV-2",
                "ref": "EV-2",
                "source_id": "S2",
                "source_ref": "[2]",
                "proof_role": "metric",
                "source_level": "B",
                "fact": "A market report estimates the industry has entered a multi-hundred-billion yuan scale.",
            },
        ],
    }
    structured_analysis = {
        "claim_units": [
            {
                "claim_id": "CL-context",
                "chapter_id": "ch_01",
                "claim": "Consumer interest can be treated as an early directional commercialization signal.",
                "fact_ids": ["EV-1"],
                "source_ids": ["S1"],
                "supporting_facts": ["Surveyed consumers show early interest in low-altitude services."],
                "claim_strength": "directional",
                "claim_roles": ["context_claim"],
            },
            {
                "claim_id": "CL-market",
                "chapter_id": "ch_02",
                "claim": "Market sizing evidence supports a cautious commercialization discussion.",
                "fact_ids": ["EV-2"],
                "source_ids": ["S2"],
                "supporting_facts": ["A market report estimates the industry has entered a multi-hundred-billion yuan scale."],
                "claim_strength": "moderate",
                "claim_roles": ["metric_claim"],
            },
        ]
    }

    units = run_claim_builder_agent(
        chapter_evidence_packages=[evidence_package],
        micro_layouts=[
            {
                "chapter_id": "CH_contextual_claim",
                "sections": final_chapter["section_plan"],
            }
        ],
        structured_analysis=structured_analysis,
    )
    public_units = [unit for unit in units if unit.get("public_render") and not unit.get("omit_from_report")]

    assert [unit["claim_id"] for unit in public_units] == ["CL-context", "CL-market"]
    assert {unit["chapter_id"] for unit in public_units} == {"CH_contextual_claim"}
    assert public_units[0]["source_chapter_id"] == "ch_01"
    assert public_units[0]["force_public_render_context"] is True

    packages = run_chapter_argument_agent(
        report_blueprint={"chapters": [final_chapter]},
        micro_layouts=[
            {
                "chapter_id": "CH_contextual_claim",
                "sections": final_chapter["section_plan"],
            }
        ],
        argument_units=units,
        chapter_evidence_packages=[evidence_package],
    )
    sections = packages[0]["sections"]

    assert [section["claim_id"] for section in sections] == ["CL-context", "CL-market"]
    assert all(section["evidence_backed"] for section in sections)
    assert packages[0]["chapter_id"] == "CH_contextual_claim"
    assert not packages[0]["omit_from_report"]


def test_claim_builder_dedupes_repeated_llm_extra_section_titles_across_report(monkeypatch):
    monkeypatch.setenv("REPORT_BLUEPRINT_SOURCE", "claim_first")
    monkeypatch.setenv("REPORT_EXTRA_LLM_CLAIMS_PER_CHAPTER", "4")
    repeated_title = "市场空间在哪里"

    packages = [
        {
            "chapter_id": chapter_id,
            "chapter_title": f"Chapter {chapter_id}",
            "core_evidence": [
                {
                    "evidence_id": f"{chapter_id}-EV-{index}",
                    "ref": f"{chapter_id}-EV-{index}",
                    "source_id": "S1",
                    "source_ref": "[1]",
                    "proof_role": "support",
                    "source_level": "B",
                    "fact": f"{chapter_id} fact {index}",
                }
                for index in range(1, 4)
            ],
        }
        for chapter_id in ("CH_market", "CH_competition")
    ]
    claims = []
    for chapter_id, prefix in (("CH_market", "Market demand"), ("CH_competition", "Competition pattern")):
        for index in range(1, 4):
            claims.append(
                {
                    "claim_id": f"{chapter_id}-CL-{index}",
                    "chapter_id": chapter_id,
                    "section_title": repeated_title,
                    "claim": f"{prefix} signal {index} should become a distinct public section.",
                    "fact_ids": [f"{chapter_id}-EV-{index}"],
                    "source_ids": ["S1"],
                    "supporting_facts": [f"{chapter_id} fact {index}"],
                    "claim_strength": "directional",
                }
            )

    units = run_claim_builder_agent(
        chapter_evidence_packages=packages,
        structured_analysis={"claim_units": claims},
    )

    titles = [
        unit.get("section_title")
        for unit in units
        if unit.get("claim_id") and unit.get("public_render") and not unit.get("omit_from_report")
    ]
    assert repeated_title not in titles
    assert len(titles) == len(set(titles))


def test_claim_first_section_builder_keeps_distinct_claims_reusing_same_evidence():
    packages = run_chapter_argument_agent(
        report_blueprint={
            "chapters": [
                {
                    "chapter_id": "CH_market",
                    "chapter_title": "Market signals",
                    "chapter_role": "claim_driven_final_chapter",
                    "claim_ids": ["CL1", "CL2"],
                }
            ]
        },
        micro_layouts=[
            {
                "chapter_id": "CH_market",
                "sections": [
                    {"section_id": "CH_market_s1", "block_type": "integrated_signal"},
                    {"section_id": "CH_market_s2", "block_type": "integrated_signal"},
                ],
            }
        ],
        argument_units=[
            {
                "claim_id": "CL1",
                "chapter_id": "CH_market",
                "section_id": "CH_market_s1",
                "claim": "The first angle uses the shared evidence.",
                "reasoning": "The same evidence supports this first interpretation.",
                "evidence_refs": ["EV-1"],
                "used_fact_refs": ["EV-1"],
                "supporting_facts": ["Shared evidence fact."],
                "public_render": True,
            },
            {
                "claim_id": "CL2",
                "chapter_id": "CH_market",
                "section_id": "CH_market_s2",
                "claim": "The second angle also uses the shared evidence.",
                "reasoning": "The same evidence supports a second public interpretation.",
                "evidence_refs": ["EV-1"],
                "used_fact_refs": ["EV-1"],
                "supporting_facts": ["Shared evidence fact."],
                "public_render": True,
            },
        ],
    )

    sections = packages[0]["sections"]
    assert [section.get("claim_id") for section in sections] == ["CL1", "CL2"]
    assert packages[0]["dropped_sections"] == []


def test_writer_section_sync_preserves_claim_lineage_fields():
    synced = _sync_chapter_section_refs_from_argument_units(
        [
            {
                "chapter_id": "CH_market",
                "sections": [
                    {
                        "section_id": "CH_market_s1",
                        "claim": "The public section already has body text.",
                    }
                ],
            }
        ],
        [
            {
                "claim_id": "CL1",
                "chapter_id": "CH_market",
                "section_id": "CH_market_s1",
                "requirement_ids": ["REQ-1"],
                "fact_ids": ["EV-1"],
                "source_ids": ["SRC-1"],
                "claim_strength": "directional",
                "evidence_refs": ["EV-1"],
                "used_fact_refs": ["EV-1"],
                "source_support_map": {"claim": ["EV-1"]},
            }
        ],
    )

    section = synced[0]["sections"][0]
    assert section["claim_id"] == "CL1"
    assert section["requirement_ids"] == ["REQ-1"]
    assert section["fact_ids"] == ["EV-1"]
    assert section["source_ids"] == ["SRC-1"]
    assert section["claim_strength"] == "directional"
    assert section["evidence_refs"] == ["EV-1"]
    assert section["used_fact_refs"] == ["EV-1"]


def test_claim_first_public_rebuild_triggers_when_old_handoff_lacks_claim_ids(monkeypatch):
    monkeypatch.setenv("REPORT_PUBLIC_REBUILD_MIN_CLAIM_COVERAGE", "0.85")

    summary = needs_public_rebuild(
        structured_analysis={
            "claim_units": [
                {"claim_id": "CL1", "claim": "Claim one.", "fact_ids": ["EV-1"]},
                {"claim_id": "CL2", "claim": "Claim two.", "fact_ids": ["EV-2"]},
            ]
        },
        argument_units=[
            {
                "section_id": "old_s1",
                "claim": "Old fallback section.",
                "evidence_refs": ["EV-1"],
                "public_render": True,
            }
        ],
        chapter_packages=[
            {
                "chapter_id": "CH_old",
                "sections": [
                    {
                        "section_id": "old_s1",
                        "claim": "Old fallback section.",
                        "evidence_refs": ["EV-1"],
                    }
                ],
            }
        ],
    )

    assert summary["required"] is True
    hit = next(item for item in summary["hits"] if item["pattern"] == "claim_first_handoff_coverage_below_minimum")
    assert hit["structured_claim_count"] == 2
    assert hit["argument_claim_count"] == 0
    assert hit["section_claim_count"] == 0
