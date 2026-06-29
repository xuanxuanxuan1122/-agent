from __future__ import annotations

from rag_pipeline.agents.chapter_evidence_builder import build_chapter_evidence_packages_from_evidence_package


def test_traceable_bc_case_evidence_hydrates_directional_or_case_layer():
    source_registry = [
        {
            "ref": "S1",
            "url": "https://research.example/agent-case",
            "title": "Agent case study",
            "source_level": "C",
            "traceability_status": "traceable",
        }
    ]
    evidence_package = {
        "analysis_ready_evidence": [
            {
                "ref": "E1",
                "source_ref": "S1",
                "source_level": "C",
                "url": "https://research.example/agent-case",
                "fact": "Enterprise customer case for AI Agent workflow automation.",
                "chapter_id": "ch_01",
                "proof_role": "case",
                "allowed_use": "directional_signal",
            }
        ],
        "source_registry": source_registry,
    }
    report_blueprint = {
        "chapters": [
            {
                "chapter_id": "ch_01",
                "chapter_title": "Enterprise AI Agent Demand",
                "chapter_question": "Which customer cases indicate AI Agent demand?",
            }
        ]
    }

    packages = build_chapter_evidence_packages_from_evidence_package(
        report_blueprint=report_blueprint,
        evidence_package=evidence_package,
        source_registry=source_registry,
    )

    assert len(packages) == 1
    package = packages[0]
    assert package["hydrated_evidence"] is True
    assert package["case_evidence_count"] >= 1
    assert package["core_evidence_count"] == 0
    assert package["case_evidence"][0]["source_traceable"] is True


def test_unresolved_refs_are_diagnostic_not_silent_drop():
    evidence_package = {
        "analysis_ready_evidence": [],
        "evidence_analysis_by_chapter": {
            "ch_01": {
                "sample_evidence_refs": ["missing-ref"],
            }
        },
    }
    report_blueprint = {"chapters": [{"chapter_id": "ch_01", "chapter_title": "Demand"}]}

    packages = build_chapter_evidence_packages_from_evidence_package(
        report_blueprint=report_blueprint,
        evidence_package=evidence_package,
        source_registry=[],
    )

    assert packages[0]["hydrated_evidence"] is False
    assert packages[0]["unresolved_evidence_refs"] == ["missing-ref"]


def test_recomposed_chapter_uses_source_plan_chapter_id_for_evidence_binding(monkeypatch):
    monkeypatch.setenv("REPORT_EVIDENCE_MODE", "advisory_weight")
    source_registry = [
        {
            "ref": "S1",
            "url": "https://robot.example/supply-risk",
            "title": "Humanoid robot supply constraint report",
            "source_level": "C",
            "traceability_status": "traceable",
        }
    ]
    evidence_package = {
        "analysis_ready_evidence": [
            {
                "ref": "E1",
                "source_ref": "S1",
                "source_level": "C",
                "url": "https://robot.example/supply-risk",
                "fact": "Humanoid robot supply chain constraints include actuator cost, battery endurance, and safety certification delays.",
                "chapter_id": "ch_04",
                "proof_role": "case",
                "allowed_use": "directional_signal",
                "public_fact_quality": {
                    "eligible_for_report": True,
                    "eligible_for_citation": True,
                    "fact_type": "case",
                    "public_fact_card": {
                        "fact": "Humanoid robot supply chain constraints include actuator cost, battery endurance, and safety certification delays.",
                        "source_ref": "E1",
                        "claim_strength_hint": "directional",
                    },
                },
            }
        ],
        "evidence_analysis_by_chapter": {
            "ch_04": {"sample_evidence_refs": ["E1"]},
        },
        "source_registry": source_registry,
    }
    report_blueprint = {
        "chapters": [
            {
                "chapter_id": "CH_ch_04",
                "chapter_title": "Technology, supply, regulatory, and substitute constraints",
                "chapter_question": "How do constraints change opportunity ranking?",
                "source_plan_chapter_ids": ["ch_04"],
            }
        ]
    }

    packages = build_chapter_evidence_packages_from_evidence_package(
        report_blueprint=report_blueprint,
        evidence_package=evidence_package,
        source_registry=source_registry,
    )

    package = packages[0]
    assert package["chapter_id"] == "CH_ch_04"
    assert package["hydrated_evidence"] is True
    assert package["evidence_binding_funnel"]["resolved_diagnostic_ref_count"] == 1
    assert package["binding_reasons"]["evidence_analysis_ref"] == 1
    assert package["fact_card_count"] >= 1


def test_chapter_evidence_rejects_offtopic_source_check_even_with_chapter_id():
    source_registry = [
        {
            "ref": "S1",
            "url": "https://www.stats.gov.cn/agent-channel",
            "title": "AI Agent statistics publication channel",
            "source_level": "B",
            "traceability_status": "traceable",
        }
    ]
    evidence_package = {
        "analysis_ready_evidence": [
            {
                "ref": "E1",
                "source_ref": "S1",
                "source_level": "B",
                "url": "https://www.stats.gov.cn/agent-channel",
                "fact": "The statistics office publishes AI Agent statistics through its website, yearbook, and official social media accounts.",
                "chapter_id": "ch_01",
                "proof_role": "source_check",
                "allowed_use": "supporting",
            }
        ],
        "source_registry": source_registry,
    }
    report_blueprint = {
        "chapters": [
            {
                "chapter_id": "ch_01",
                "chapter_title": "Enterprise AI Agent competitive landscape and vendor differentiation",
                "chapter_question": "Which vendors, products, and deployment patterns shape the competitive landscape?",
            }
        ]
    }

    packages = build_chapter_evidence_packages_from_evidence_package(
        report_blueprint=report_blueprint,
        evidence_package=evidence_package,
        source_registry=source_registry,
    )

    package = packages[0]
    assert package["hydrated_evidence"] is False
    assert package["metadata"]["chapter_relevance_rejected_count"] == 1
    assert package["metadata"]["chapter_relevance_rejected_refs"] == ["E1"]


def test_chapter_evidence_rejects_high_grade_offtopic_report_anchor(monkeypatch):
    monkeypatch.setenv("REPORT_EVIDENCE_MODE", "advisory_weight")
    source_registry = [
        {
            "ref": "S1",
            "url": "https://battery.example/solid-state",
            "title": "Solid-state battery commercialization report",
            "source_level": "C",
            "traceability_status": "traceable",
        },
        {
            "ref": "S2",
            "url": "https://sjj.qinghai.gov.cn/compute-bluepaper.pdf",
            "title": "Qinghai compute power bluepaper",
            "source_level": "A",
            "traceability_status": "traceable",
        },
    ]
    evidence_package = {
        "query": "中国固态电池商业化落地机会与风险",
        "analysis_ready_evidence": [
            {
                "ref": "E1",
                "source_ref": "S1",
                "source_level": "C",
                "url": "https://battery.example/solid-state",
                "fact": "固态电池企业正在推进车规验证和小批量装车，商业化节奏取决于成本、良率和供应链配套。",
                "chapter_id": "ch_01",
                "proof_role": "case",
                "allowed_use": "supporting_context",
            },
            {
                "ref": "E2",
                "source_ref": "S2",
                "source_level": "A",
                "url": "https://sjj.qinghai.gov.cn/compute-bluepaper.pdf",
                "fact": "青海综合算力指数蓝皮书显示，全球算力需求高速增长，人工智能训练带动数据中心投资。",
                "chapter_id": "ch_01",
                "proof_role": "source_check",
                "allowed_use": "supporting",
            },
        ],
        "source_registry": source_registry,
    }
    report_blueprint = {
        "query": "中国固态电池商业化落地机会与风险",
        "topic_anchor_terms": ["固态电池", "全固态电池", "半固态电池"],
        "chapters": [
            {
                "chapter_id": "ch_01",
                "chapter_title": "固态电池商业化进展与落地节奏",
                "chapter_question": "哪些验证、装车和供应链信号能说明固态电池商业化正在推进？",
            }
        ],
    }

    packages = build_chapter_evidence_packages_from_evidence_package(
        report_blueprint=report_blueprint,
        evidence_package=evidence_package,
        source_registry=source_registry,
    )

    package = packages[0]
    refs = {
        item.get("ref")
        for key in (
            "core_evidence",
            "supporting_evidence",
            "case_evidence",
            "directional_evidence",
            "sample_evidence",
        )
        for item in package.get(key, [])
    }
    assert "E1" in refs
    assert "E2" not in refs
    assert package["metadata"]["chapter_relevance_rejected_refs"] == ["E2"]
    assert package["metadata"]["chapter_relevance_rejection_reasons"] == {"report_topic_anchor_missing": 1}


def test_chapter_evidence_report_topic_ignores_generic_short_terms(monkeypatch):
    monkeypatch.setenv("REPORT_EVIDENCE_MODE", "advisory_weight")
    source_registry = [
        {
            "ref": "S1",
            "url": "https://example.gov/robotics",
            "title": "Humanoid robotics report",
            "source_level": "A",
            "traceability_status": "traceable",
        }
    ]
    evidence_package = {
        "query": "\u4f4e\u7a7a\u7ecf\u6d4e\u653f\u7b56\u5bf9\u65e0\u4eba\u673a\u914d\u9001\u7684\u5f71\u54cd",
        "analysis_ready_evidence": [
            {
                "ref": "E1",
                "source_ref": "S1",
                "source_level": "A",
                "url": "https://example.gov/robotics",
                "fact": "\u4e2d\u56fd\u4eba\u5f62\u673a\u5668\u4eba\u4ea7\u4e1a\u4f4d\u4e8e\u67d0\u7ecf\u6d4e\u6280\u672f\u5f00\u53d1\u533a\uff0c\u5e76\u53d7\u5230\u672a\u6765\u4ea7\u4e1a\u653f\u7b56\u652f\u6301\u3002",
                "chapter_id": "ch_01",
                "proof_role": "source_check",
                "allowed_use": "supporting",
            }
        ],
        "source_registry": source_registry,
    }
    report_blueprint = {
        "query": "\u4f4e\u7a7a\u7ecf\u6d4e\u653f\u7b56\u5bf9\u65e0\u4eba\u673a\u914d\u9001\u7684\u5f71\u54cd",
        "topic_anchor_terms": ["\u4f4e\u7a7a\u7ecf\u6d4e", "\u65e0\u4eba\u673a\u914d\u9001", "\u65e0\u4eba\u673a"],
        "chapters": [
            {
                "chapter_id": "ch_01",
                "chapter_title": "\u4f4e\u7a7a\u7ecf\u6d4e\u653f\u7b56\u4e0e\u65e0\u4eba\u673a\u914d\u9001\u5546\u4e1a\u5316",
                "chapter_question": "\u653f\u7b56\u5982\u4f55\u5f71\u54cd\u65e0\u4eba\u673a\u914d\u9001\u5546\u4e1a\u5316\uff1f",
            }
        ],
    }

    packages = build_chapter_evidence_packages_from_evidence_package(
        report_blueprint=report_blueprint,
        evidence_package=evidence_package,
        source_registry=source_registry,
    )

    package = packages[0]
    assert package["hydrated_evidence"] is False
    assert package["metadata"]["chapter_relevance_rejected_refs"] == ["E1"]
    assert package["metadata"]["chapter_relevance_rejection_reasons"] == {"report_topic_anchor_missing": 1}


def test_rebuilt_chapter_evidence_does_not_keep_stale_existing_layers(monkeypatch):
    monkeypatch.setenv("REPORT_EVIDENCE_MODE", "advisory_weight")
    source_registry = [
        {
            "ref": "S1",
            "url": "https://example.gov/low-altitude",
            "title": "Low altitude policy",
            "source_level": "A",
            "traceability_status": "traceable",
        }
    ]
    dirty_nav = "[](/newstatic/images/logo.gif)](//www.eastmoney.com/) [\u6570\u636e\u4e2d\u5fc3](/center/) [Choice\u6570\u636e](//choice.eastmoney.com)"
    evidence_package = {
        "query": "\u4f4e\u7a7a\u7ecf\u6d4e\u65e0\u4eba\u673a\u914d\u9001",
        "analysis_ready_evidence": [
            {
                "ref": "E1",
                "source_ref": "S1",
                "source_level": "A",
                "url": "https://example.gov/low-altitude",
                "fact": "\u4f4e\u7a7a\u7ecf\u6d4e\u653f\u7b56\u660e\u786e\u652f\u6301\u65e0\u4eba\u673a\u7269\u6d41\u914d\u9001\u8bd5\u70b9\u548c\u8fd0\u884c\u573a\u666f\u62d3\u5c55\u3002",
                "chapter_id": "ch_01",
                "proof_role": "source_check",
                "allowed_use": "supporting",
            }
        ],
        "source_registry": source_registry,
    }
    report_blueprint = {
        "query": "\u4f4e\u7a7a\u7ecf\u6d4e\u65e0\u4eba\u673a\u914d\u9001",
        "topic_anchor_terms": ["\u4f4e\u7a7a\u7ecf\u6d4e", "\u65e0\u4eba\u673a\u914d\u9001"],
        "chapters": [
            {
                "chapter_id": "ch_01",
                "chapter_title": "\u4f4e\u7a7a\u7ecf\u6d4e\u653f\u7b56\u4e0e\u65e0\u4eba\u673a\u914d\u9001",
            }
        ],
    }
    existing = [
        {
            "chapter_id": "ch_01",
            "chapter_title": "\u4f4e\u7a7a\u7ecf\u6d4e\u653f\u7b56\u4e0e\u65e0\u4eba\u673a\u914d\u9001",
            "table_evidence": [{"ref": "DIRTY", "source_ref": "S-DIRTY", "fact": dirty_nav}],
            "clue_evidence": [{"ref": "DIRTY", "source_ref": "S-DIRTY", "fact": dirty_nav}],
            "evidence_counts": {"table_evidence": 1, "clue_evidence": 1},
        }
    ]

    packages = build_chapter_evidence_packages_from_evidence_package(
        report_blueprint=report_blueprint,
        evidence_package=evidence_package,
        existing_chapter_evidence_packages=existing,
        source_registry=source_registry,
    )

    package = packages[0]
    assert package["hydrated_evidence"] is True
    assert "table_evidence" not in package
    assert "clue_evidence" not in package
    assert package["evidence_counts"]["evidence_items"] >= 1


def test_chapter_evidence_distills_mixed_source_fact_tail(monkeypatch):
    monkeypatch.setenv("REPORT_EVIDENCE_MODE", "advisory_weight")
    source_registry = [
        {
            "ref": "S1",
            "url": "https://example.gov/low-altitude-company",
            "title": "\u4f5b\u5c71\u5e02\u4f4e\u7a7a\u7ecf\u6d4e\u4f01\u4e1a\u4fe1\u606f",
            "source_level": "C",
            "traceability_status": "traceable",
        }
    ]
    evidence_package = {
        "query": "\u4f4e\u7a7a\u7ecf\u6d4e\u65e0\u4eba\u673a\u914d\u9001",
        "analysis_ready_evidence": [
            {
                "ref": "E1",
                "source_ref": "S1",
                "source_level": "C",
                "url": "https://example.gov/low-altitude-company",
                "fact": (
                    "\u4f5b\u5c71\u5e02\u6210\u7acb\u4f4e\u7a7a\u7ecf\u6d4e\u53d1\u5c55\u6709\u9650\u516c\u53f8\uff0c"
                    "\u7ecf\u8425\u8303\u56f4\u5305\u62ec\u822a\u7a7a\u8fd0\u8f93\u8bbe\u5907\u9500\u552e\u548c\u667a\u80fd\u65e0\u4eba\u98de\u884c\u5668\u5236\u9020\uff1b"
                    "[PDF] \u4e0a\u5e02\u516c\u53f82024 \u5e74\u5e74\u5ea6\u8d22\u52a1\u62a5\u544a\u4f1a\u8ba1\u76d1\u7ba1\u62a5\u544a\u62ab\u9732\u4e86\u4e0e\u4e3b\u9898\u65e0\u5173\u7684\u5e74\u62a5\u4fe1\u606f\u3002"
                    "\u6309\u671f\u62ab\u9732\u5e74\u5ea6\u8d22 \u52a1\u62a5\u544a\u7684\u4e0a\u5e02\u516c\u53f8\u4e2d\uff0c\u4e3b\u677f\u3001\u521b\u4e1a\u677f\u3001\u79d1\u521b\u677f\u548c\u5317\u4ea4\u6240\u6837\u672c\u4e0e\u4e3b\u9898\u65e0\u5173\u3002"
                ),
                "chapter_id": "ch_01",
                "proof_role": "case",
                "allowed_use": "supporting_context",
            }
        ],
        "source_registry": source_registry,
    }
    report_blueprint = {
        "query": "\u4f4e\u7a7a\u7ecf\u6d4e\u65e0\u4eba\u673a\u914d\u9001",
        "topic_anchor_terms": ["\u4f4e\u7a7a\u7ecf\u6d4e", "\u65e0\u4eba\u673a"],
        "chapters": [
            {
                "chapter_id": "ch_01",
                "chapter_title": "\u4f4e\u7a7a\u7ecf\u6d4e\u653f\u7b56\u4e0e\u65e0\u4eba\u673a\u914d\u9001",
            }
        ],
    }

    packages = build_chapter_evidence_packages_from_evidence_package(
        report_blueprint=report_blueprint,
        evidence_package=evidence_package,
        source_registry=source_registry,
    )

    package = packages[0]
    assert package["hydrated_evidence"] is True
    cards = [
        item
        for key in ("core_evidence", "case_evidence", "directional_evidence", "supporting_evidence", "sample_evidence")
        for item in package.get(key, [])
    ]
    distilled = " ".join(
        str(
            (item.get("public_fact_quality") or {}).get("distilled_fact")
            or (item.get("public_fact_card") or {}).get("distilled_fact")
            or item.get("distilled_fact")
            or ""
        )
        for item in cards
    )
    assert "\u4f5b\u5c71\u5e02" in distilled
    assert "\u4f4e\u7a7a\u7ecf\u6d4e" in distilled
    assert "\u4e0a\u5e02\u516c\u53f8" not in distilled
    assert "\u8d22\u52a1\u62a5\u544a" not in distilled
    assert "\u6309\u671f\u62ab\u9732" not in distilled


def test_chapter_evidence_advisory_mode_keeps_traceable_d_source(monkeypatch):
    monkeypatch.setenv("REPORT_EVIDENCE_MODE", "advisory_weight")
    source_registry = [
        {
            "ref": "S1",
            "url": "https://blog.example/solid-state-battery-field-notes",
            "title": "Solid-state battery field notes",
            "source_level": "D",
            "traceability_status": "traceable",
        }
    ]
    evidence_package = {
        "query": "中国固态电池商业化落地机会与风险",
        "analysis_ready_evidence": [
            {
                "ref": "E1",
                "source_ref": "S1",
                "source_level": "D",
                "url": "https://blog.example/solid-state-battery-field-notes",
                "fact": "固态电池试产线走访记录提到，样品验证、良率爬坡和材料成本仍是近期商业化落地的主要约束。",
                "chapter_id": "ch_01",
                "proof_role": "case",
                "allowed_use": "supporting_context",
            }
        ],
        "source_registry": source_registry,
    }
    report_blueprint = {
        "query": "中国固态电池商业化落地机会与风险",
        "topic_anchor_terms": ["固态电池"],
        "chapters": [
            {
                "chapter_id": "ch_01",
                "chapter_title": "固态电池商业化进展与落地节奏",
                "chapter_question": "哪些验证、装车和供应链信号能说明固态电池商业化正在推进？",
            }
        ],
    }

    packages = build_chapter_evidence_packages_from_evidence_package(
        report_blueprint=report_blueprint,
        evidence_package=evidence_package,
        source_registry=source_registry,
    )

    package = packages[0]
    assert package["hydrated_evidence"] is True
    assert any(item.get("ref") == "E1" for item in package["case_evidence"] + package["directional_evidence"])


def test_chapter_evidence_exports_binding_funnel_for_loss_diagnostics():
    source_registry = [
        {
            "ref": "S1",
            "url": "https://research.example/agent-demand",
            "title": "Agent demand report",
            "source_level": "B",
            "traceability_status": "traceable",
        },
        {
            "ref": "S2",
            "url": "https://research.example/offtopic",
            "title": "Agent publication channel",
            "source_level": "B",
            "traceability_status": "traceable",
        },
    ]
    evidence_package = {
        "analysis_ready_evidence": [
            {
                "ref": "E1",
                "source_ref": "S1",
                "source_level": "B",
                "url": "https://research.example/agent-demand",
                "fact": "Enterprise AI Agent deployment cases indicate workflow automation demand.",
                "chapter_id": "ch_01",
                "proof_role": "case",
                "allowed_use": "directional_signal",
            },
            {
                "ref": "E2",
                "source_ref": "S2",
                "source_level": "B",
                "url": "https://research.example/offtopic",
                "fact": "The statistics office publishes AI Agent statistics through its website and yearbook.",
                "chapter_id": "ch_01",
                "proof_role": "source_check",
                "allowed_use": "supporting",
            },
        ],
        "evidence_analysis_by_chapter": {
            "ch_01": {"sample_evidence_refs": ["E1", "missing-ref"]},
        },
        "source_registry": source_registry,
    }
    report_blueprint = {
        "chapters": [
            {
                "chapter_id": "ch_01",
                "chapter_title": "Enterprise AI Agent deployment demand",
                "chapter_question": "Which customer cases indicate AI Agent demand?",
            }
        ]
    }

    packages = build_chapter_evidence_packages_from_evidence_package(
        report_blueprint=report_blueprint,
        evidence_package=evidence_package,
        source_registry=source_registry,
    )

    package = packages[0]
    funnel = package["evidence_binding_funnel"]
    assert funnel["candidate_fact_count"] == 2
    assert funnel["eligible_fact_count"] == 2
    assert funnel["resolved_diagnostic_ref_count"] == 1
    assert funnel["unresolved_ref_count"] == 1
    assert funnel["matched_before_relevance_count"] >= 1
    assert funnel["matched_after_relevance_count"] >= 1
    assert funnel["hydrated_evidence_count"] == package["hydrated_evidence_count"]
    assert funnel["layer_counts"]["case_evidence"] >= 1
    assert package["metadata"]["evidence_binding_funnel"] == funnel
