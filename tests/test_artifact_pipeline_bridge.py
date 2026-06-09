from __future__ import annotations

from rag_pipeline.cache.artifact_pipeline_bridge import ingest_writer_package_artifacts
from rag_pipeline.cache.artifact_store import ArtifactStore
from rag_pipeline.context.context_view_builder import build_writer_context_view


def test_bridge_ingests_requirement_to_section_and_score_gap_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_LEDGER_PATH", str(tmp_path / "artifact_ledger.sqlite"))
    monkeypatch.setenv("ARTIFACT_OBJECT_ROOT", str(tmp_path / "objects"))
    store = ArtifactStore()
    store.upsert_run(run_id="run-smoke", query="AI Agent", status="running")

    writer_package = {
        "evidence_package": {
            "report_contract": {
                "evidence_requirements": {
                    "requirements": [
                        {
                            "requirement_id": "H1_case",
                            "chapter_id": "ch_01",
                            "proof_role": "case",
                            "required_fields": ["company", "use_case"],
                            "claim_strength_ceiling": "directional",
                        }
                    ]
                }
            },
            "search_task_schedule": {
                "tasks": [{"task_id": "ST-1", "requirement_id": "H1_case", "query": "official customer case"}]
            },
            "analysis_ready_evidence": [
                {
                    "evidence_id": "EV-1",
                    "requirement_id": "H1_case",
                    "search_task_id": "ST-1",
                    "source_id": "SRC-1",
                    "fact": "Company A disclosed an AI agent workflow deployment.",
                    "source_level": "A",
                    "source": {"id": "SRC-1", "url": "https://example.com/case", "title": "Official case"},
                }
            ],
        },
        "source_registry": [{"id": "SRC-1", "url": "https://example.com/case", "title": "Official case"}],
        "argument_units": [
            {
                "claim_id": "CL-1",
                "claim": "Company A has disclosed a workflow deployment.",
                "requirement_ids": ["H1_case"],
                "used_fact_refs": ["EV-1"],
                "source_ids": ["SRC-1"],
                "claim_strength": "directional",
                "claim_strength_ceiling": "directional",
            }
        ],
        "chapter_packages": [
            {
                "chapter_id": "ch_01",
                "sections": [
                    {
                        "section_id": "SEC-1",
                        "claim_id": "CL-1",
                        "requirement_ids": ["H1_case"],
                        "used_fact_refs": ["EV-1"],
                        "evidence_backed": True,
                    }
                ],
            }
        ],
    }
    writer_report = {
        "qa_result": {
            "quality_findings": [
                {
                    "requirement_id": "H1_case",
                    "section_id": "SEC-1",
                    "gap_type": "case_boundary_missing",
                    "missing": ["limitation boundary"],
                    "severity": "medium",
                }
            ]
        }
    }

    summary = ingest_writer_package_artifacts(
        store,
        run_id="run-smoke",
        writer_package=writer_package,
        writer_report=writer_report,
    )

    assert summary["requirement_count"] == 1
    assert summary["fact_card_count"] == 1
    assert summary["claim_unit_count"] == 1
    assert summary["section_count"] == 1
    assert summary["score_gap_count"] == 1
    edge_targets = {
        (edge["to_type"], edge["to_id"])
        for edge in store.traverse_lineage("run-smoke", "requirement", "H1_case", max_depth=4)
    }
    assert ("section", "SEC-1") in edge_targets
    assert any(target_type == "score_gap" for target_type, _target_id in edge_targets)

    view = build_writer_context_view("run-smoke", "SEC-1")
    assert view["status"] == "ready"
    assert view["used_fact_refs"] == ["EV-1"]


def test_bridge_backfills_claim_section_requirements_via_fact_layer(tmp_path, monkeypatch):
    # The claim/section carry NO requirement_ids of their own, but cite a fact
    # that resolves to a requirement. The fact-layer fallback must project the
    # requirement (and source) onto the denormalized claim_unit/section columns,
    # and lineage_edge_total must report the persisted graph (not the per-call delta).
    monkeypatch.setenv("ARTIFACT_LEDGER_PATH", str(tmp_path / "artifact_ledger.sqlite"))
    monkeypatch.setenv("ARTIFACT_OBJECT_ROOT", str(tmp_path / "objects"))
    store = ArtifactStore()
    store.upsert_run(run_id="run-bf", query="AI Agent", status="running")

    writer_package = {
        "evidence_package": {
            "report_contract": {
                "evidence_requirements": {
                    "requirements": [
                        {"requirement_id": "H1_metric", "chapter_id": "ch_01", "proof_role": "metric"}
                    ]
                }
            },
            "analysis_ready_evidence": [
                {
                    "evidence_id": "EV-1",
                    "requirement_id": "H1_metric",
                    "source_id": "SRC-1",
                    "fact": "An official source reports an adoption metric.",
                    "source_level": "A",
                    "source": {"id": "SRC-1", "url": "https://example.com/metric", "title": "Official"},
                }
            ],
        },
        "source_registry": [{"id": "SRC-1", "url": "https://example.com/metric", "title": "Official"}],
        "argument_units": [
            {"claim_id": "CL-1", "claim": "Adoption is rising.", "used_fact_refs": ["EV-1"]}
        ],
        "chapter_packages": [
            {"chapter_id": "ch_01", "sections": [{"section_id": "SEC-1", "claim_id": "CL-1", "used_fact_refs": ["EV-1"]}]}
        ],
    }

    summary = ingest_writer_package_artifacts(store, run_id="run-bf", writer_package=writer_package, writer_report={})

    claim = store.list_claim_units("run-bf", claim_ids=["CL-1"])[0]
    assert "H1_metric" in claim["requirement_ids"]
    assert "SRC-1" in claim["source_ids"]
    with store._connect() as conn:
        section_reqs = conn.execute(
            "SELECT requirement_ids_json FROM sections WHERE run_id=? AND section_id=?",
            ("run-bf", "SEC-1"),
        ).fetchone()[0]
    assert "H1_metric" in section_reqs
    edge_targets = {
        (edge["to_type"], edge["to_id"])
        for edge in store.traverse_lineage("run-bf", "requirement", "H1_metric", max_depth=4)
    }
    assert ("claim_unit", "CL-1") in edge_targets
    assert ("section", "SEC-1") in edge_targets

    # lineage_edge_total reflects the real table; re-ingest delta is 0 but total holds.
    assert summary["lineage_edge_total"] >= 1
    second = ingest_writer_package_artifacts(store, run_id="run-bf", writer_package=writer_package, writer_report={})
    assert second["lineage_edge_count"] == 0
    assert second["lineage_edge_total"] == summary["lineage_edge_total"]


def test_bridge_canonicalizes_legacy_fact_refs_before_persisting_claims(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_LEDGER_PATH", str(tmp_path / "artifact_ledger.sqlite"))
    monkeypatch.setenv("ARTIFACT_OBJECT_ROOT", str(tmp_path / "objects"))
    store = ArtifactStore()
    store.upsert_run(run_id="run-alias", query="AI Agent", status="running")

    writer_package = {
        "evidence_package": {
            "report_contract": {
                "evidence_requirements": {
                    "requirements": [
                        {"requirement_id": "H1_metric", "chapter_id": "ch_01", "proof_role": "metric"}
                    ]
                }
            },
            "analysis_ready_evidence": [
                {
                    "evidence_id": "EV-canonical",
                    "aliases": ["EV-legacy"],
                    "requirement_id": "H1_metric",
                    "source_id": "SRC-1",
                    "fact": "A canonical fact carries a legacy evidence alias.",
                    "source": {"id": "SRC-1", "url": "https://example.com/metric"},
                }
            ],
        },
        "source_registry": [{"id": "SRC-1", "url": "https://example.com/metric"}],
        "argument_units": [
            {
                "claim_id": "CL-alias",
                "claim": "The claim cites the legacy evidence id.",
                "used_fact_refs": ["EV-legacy"],
            }
        ],
        "chapter_packages": [
            {
                "chapter_id": "ch_01",
                "sections": [
                    {"section_id": "SEC-alias", "claim_id": "CL-alias", "used_fact_refs": ["EV-legacy"]}
                ],
            }
        ],
    }

    ingest_writer_package_artifacts(store, run_id="run-alias", writer_package=writer_package, writer_report={})

    claim = store.list_claim_units("run-alias", claim_ids=["CL-alias"])[0]
    assert claim["fact_ids"] == ["EV-canonical"]
    assert claim["requirement_ids"] == ["H1_metric"]
    assert claim["source_ids"] == ["SRC-1"]
    with store._connect() as conn:
        row = conn.execute(
            "SELECT used_fact_refs_json, requirement_ids_json FROM sections WHERE run_id=? AND section_id=?",
            ("run-alias", "SEC-alias"),
        ).fetchone()
    assert "EV-canonical" in row[0]
    assert "EV-legacy" not in row[0]
    assert "H1_metric" in row[1]


def test_bridge_persists_section_audit_score_gaps(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_LEDGER_PATH", str(tmp_path / "artifact_ledger.sqlite"))
    monkeypatch.setenv("ARTIFACT_OBJECT_ROOT", str(tmp_path / "objects"))
    store = ArtifactStore()
    store.upsert_run(run_id="run-section-audit", query="AI Agent", status="running")

    writer_package = {
        "evidence_package": {
            "report_contract": {
                "evidence_requirements": {
                    "requirements": [
                        {
                            "requirement_id": "H1_metric",
                            "chapter_id": "ch_01",
                            "proof_role": "metric",
                            "required_fields": ["metric", "value", "unit", "period", "source"],
                        }
                    ]
                }
            },
            "analysis_ready_evidence": [
                {
                    "evidence_id": "EV-1",
                    "requirement_id": "H1_metric",
                    "source_id": "SRC-1",
                    "fact": "A survey reports an enterprise AI agent adoption value.",
                    "proof_role": "metric",
                    "metric": "adoption",
                    "value": "42",
                    "unit": "",
                    "period": "",
                    "source": {"id": "SRC-1", "url": "https://example.com/metric", "title": "Metric"},
                }
            ],
        },
        "argument_units": [
            {
                "claim_id": "CL-1",
                "claim": "Enterprise AI agent adoption is rising.",
                "requirement_ids": ["H1_metric"],
                "used_fact_refs": ["EV-1"],
                "claim_strength": "moderate",
                "claim_strength_ceiling": "moderate",
                "claim_roles": ["core_claim", "metric_claim"],
            }
        ],
        "chapter_packages": [
            {
                "chapter_id": "ch_01",
                "sections": [
                    {
                        "section_id": "SEC-1",
                        "claim_id": "CL-1",
                        "requirement_ids": ["H1_metric"],
                        "used_fact_refs": ["EV-1"],
                        "evidence_backed": True,
                    }
                ],
            }
        ],
    }

    summary = ingest_writer_package_artifacts(
        store,
        run_id="run-section-audit",
        writer_package=writer_package,
        writer_report={},
    )

    gaps = store.list_score_gaps("run-section-audit", requirement_id="H1_metric")
    gap_types = {gap["gap_type"] for gap in gaps}
    assert summary["score_gap_count"] == 2
    assert "counter_boundary_missing" in gap_types
    assert "metric_scope_period_unit_incomplete" in gap_types
    metric_gap = next(gap for gap in gaps if gap["gap_type"] == "metric_scope_period_unit_incomplete")
    assert metric_gap["section_id"] == "SEC-1"
    assert metric_gap["missing"] == ["unit", "period"]
    assert metric_gap["retry_plan"]["source_stage"] == "section_audit"
    assert metric_gap["retry_plan"]["allowed_for_writing"] is False


def test_bridge_recovers_section_claim_ids_from_fact_overlap(tmp_path, monkeypatch):
    # A section that carries used_fact_refs but no claim_ids (the composer often
    # fails to propagate them) must be bound back to the claim_unit it consumes
    # via fact overlap, so the section is provably claim-backed, not just fact-backed.
    monkeypatch.setenv("ARTIFACT_LEDGER_PATH", str(tmp_path / "artifact_ledger.sqlite"))
    monkeypatch.setenv("ARTIFACT_OBJECT_ROOT", str(tmp_path / "objects"))
    store = ArtifactStore()
    store.upsert_run(run_id="run-sec", query="AI Agent", status="running")

    writer_package = {
        "evidence_package": {
            "analysis_ready_evidence": [
                {
                    "evidence_id": "EV-1",
                    "source_id": "SRC-1",
                    "fact": "Company A disclosed an AI agent deployment.",
                    "source_level": "A",
                    "source": {"id": "SRC-1", "url": "https://example.com/case", "title": "case"},
                }
            ]
        },
        "argument_units": [
            {"claim_id": "CL-1", "claim": "Company A deployed.", "used_fact_refs": ["EV-1"], "source_ids": ["SRC-1"]}
        ],
        "chapter_packages": [
            {
                "chapter_id": "ch_01",
                "sections": [
                    # NO claim_id / claim_ids — only a fact ref that overlaps CL-1.
                    {"section_id": "SEC-1", "used_fact_refs": ["EV-1"], "evidence_backed": True}
                ],
            }
        ],
    }

    ingest_writer_package_artifacts(store, run_id="run-sec", writer_package=writer_package)
    with store._connect() as conn:
        claim_ids_json = conn.execute(
            "SELECT claim_ids_json FROM sections WHERE run_id=? AND section_id=?", ("run-sec", "SEC-1")
        ).fetchone()[0]
    assert "CL-1" in claim_ids_json


def test_bridge_repeated_ingest_does_not_duplicate_lineage_edges(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_LEDGER_PATH", str(tmp_path / "artifact_ledger.sqlite"))
    monkeypatch.setenv("ARTIFACT_OBJECT_ROOT", str(tmp_path / "objects"))
    store = ArtifactStore()
    store.upsert_run(run_id="run-repeat", query="AI Agent", status="running")

    writer_package = {
        "evidence_package": {
            "report_contract": {
                "evidence_requirements": {
                    "requirements": [{"requirement_id": "H1_case", "chapter_id": "ch_01", "proof_role": "case"}]
                }
            },
            "evidence_gap_ledger": [
                {
                    "gap_id": "GAP-1",
                    "requirement_id": "H1_case",
                    "chapter_id": "ch_01",
                    "gap_type": "case_evidence_missing",
                    "severity": "blocking",
                }
            ],
        }
    }

    first_summary = ingest_writer_package_artifacts(store, run_id="run-repeat", writer_package=writer_package)
    second_summary = ingest_writer_package_artifacts(store, run_id="run-repeat", writer_package=writer_package)

    with store._connect() as conn:
        edge_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM lineage_edges
            WHERE run_id = ?
              AND from_type = ?
              AND from_id = ?
              AND to_type = ?
              AND to_id = ?
              AND relation = ?
            """,
            ("run-repeat", "requirement", "H1_case", "score_gap", "GAP-1", "gap"),
        ).fetchone()[0]

    assert first_summary["lineage_edge_count"] == 1
    assert second_summary["lineage_edge_count"] == 0
    assert edge_count == 1


def test_bridge_infers_live_research_plan_requirements_for_facts_and_gaps(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_LEDGER_PATH", str(tmp_path / "artifact_ledger.sqlite"))
    monkeypatch.setenv("ARTIFACT_OBJECT_ROOT", str(tmp_path / "objects"))
    store = ArtifactStore()
    store.upsert_run(run_id="run-live-plan", query="AI Agent", status="running")

    writer_package = {
        "evidence_package": {
            "research_plan": {
                "evidence_goals": [
                    {
                        "requirement_id": "H1_metric",
                        "goal_id": "H1_metric",
                        "chapter_id": "ch_01",
                        "hypothesis_id": "H1",
                        "proof_role": "metric",
                        "required_fields": ["metric", "value", "unit", "period", "scope", "source_ref"],
                        "required_source_levels": ["A", "B"],
                        "claim_strength_ceiling": "moderate",
                    },
                    {
                        "requirement_id": "H1_counter",
                        "goal_id": "H1_counter",
                        "chapter_id": "ch_01",
                        "chapter_title": "Demand validation",
                        "hypothesis_id": "H1",
                        "proof_role": "counter",
                        "required_fields": ["source"],
                        "required_source_levels": ["A", "B"],
                        "claim_strength_ceiling": "directional",
                    },
                ],
                "search_tasks": [
                    {
                        "task_id": "ST-H1-metric",
                        "requirement_id": "H1_metric",
                        "chapter_id": "ch_01",
                        "proof_role": "metric",
                        "query": "AI agent adoption metric official source",
                    }
                ],
            },
            "analysis_ready_evidence": [
                {
                    "evidence_id": "EV-metric",
                    "task_id": "ST-H1-metric",
                    "chapter_id": "ch_01",
                    "dimension_id": "ch_01",
                    "fact": "An official source reports an adoption metric.",
                    "metric": "adoption",
                    "source_level": "A",
                    "evidence_card": {"proof_role": "metric"},
                    "source": {"id": "SRC-1", "url": "https://example.com/metric", "title": "Official metric"},
                }
            ],
            "evidence_gap_ledger": [
                {
                    "gap_id": "GAP-counter",
                    "chapter_id": "Demand validation",
                    "gap_type": "counter_evidence_missing",
                    "severity": "blocking",
                    "required_fields": ["source"],
                }
            ],
        }
    }

    summary = ingest_writer_package_artifacts(
        store,
        run_id="run-live-plan",
        writer_package=writer_package,
        writer_report={},
    )

    assert summary["requirement_count"] == 2
    assert summary["search_task_count"] == 1
    metric_fact = store.list_fact_cards("run-live-plan", requirement_id="H1_metric")[0]
    assert metric_fact["fact_id"] == "EV-metric"
    counter_gap = store.list_score_gaps("run-live-plan", gap_id="GAP-counter")[0]
    assert counter_gap["requirement_id"] == "H1_counter"
    edge_targets = {
        (edge["to_type"], edge["to_id"])
        for edge in store.traverse_lineage("run-live-plan", "requirement", "H1_counter", max_depth=2)
    }
    assert ("score_gap", "GAP-counter") in edge_targets


def test_bridge_persists_evidence_gap_ledger_as_repair_score_gaps(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_LEDGER_PATH", str(tmp_path / "artifact_ledger.sqlite"))
    monkeypatch.setenv("ARTIFACT_OBJECT_ROOT", str(tmp_path / "objects"))
    store = ArtifactStore()
    store.upsert_run(run_id="run-repair", query="AI Agent", status="running")

    writer_package = {
        "evidence_package": {
            "report_contract": {
                "evidence_requirements": {
                    "requirements": [
                        {
                            "requirement_id": "H1_metric",
                            "chapter_id": "ch_01",
                            "proof_role": "metric",
                            "required_fields": ["metric", "value", "unit", "period", "source"],
                            "claim_strength_ceiling": "directional",
                        }
                    ]
                }
            },
            "evidence_gap_ledger": [
                {
                    "gap_id": "evidence_gap_ch_01_metric",
                    "requirement_id": "H1_metric",
                    "chapter_id": "ch_01",
                    "gap_type": "metric_scope_period_unit_incomplete",
                    "severity": "blocking",
                    "required_proof_role": "metric",
                    "required_fields": ["metric", "value", "unit", "period", "source"],
                    "query_terms": ["enterprise AI agent adoption"],
                    "lane_targets": ["official_data", "market_research"],
                    "current_evidence_refs": ["EV-old"],
                    "why_current_evidence_insufficient": "The current metric has no period or unit.",
                    "repair_route": "evidence_search",
                }
            ],
        }
    }

    summary = ingest_writer_package_artifacts(
        store,
        run_id="run-repair",
        writer_package=writer_package,
        writer_report={},
    )

    assert summary["score_gap_count"] == 1
    gap = store.list_score_gaps("run-repair", requirement_id="H1_metric")[0]
    assert gap["gap_id"] == "evidence_gap_ch_01_metric"
    assert gap["gap_type"] == "metric_scope_period_unit_incomplete"
    assert gap["severity"] == "blocking"
    assert gap["missing"] == ["metric", "value", "unit", "period", "source"]
    assert gap["retry_plan"]["source_stage"] == "evidence_gap_ledger"
    assert gap["retry_plan"]["proof_role"] == "metric"
    assert gap["retry_plan"]["required_fields"] == ["metric", "value", "unit", "period", "source"]
    assert gap["retry_plan"]["lane_targets"] == ["official_data", "market_research"]
    assert gap["retry_plan"]["query_terms"] == ["enterprise AI agent adoption"]
    assert gap["retry_plan"]["current_evidence_refs"] == ["EV-old"]
    assert gap["retry_plan"]["current_insufficiency"] == "The current metric has no period or unit."
    assert gap["retry_plan"]["allowed_for_writing"] is False
    edge_targets = {
        (edge["to_type"], edge["to_id"])
        for edge in store.traverse_lineage("run-repair", "requirement", "H1_metric", max_depth=2)
    }
    assert ("score_gap", "evidence_gap_ch_01_metric") in edge_targets


def test_bridge_persists_claim_repair_priorities_as_score_gaps(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_LEDGER_PATH", str(tmp_path / "artifact_ledger.sqlite"))
    monkeypatch.setenv("ARTIFACT_OBJECT_ROOT", str(tmp_path / "objects"))
    store = ArtifactStore()
    store.upsert_run(run_id="run-claim-repair", query="AI Agent", status="running")

    writer_package = {
        "structured_analysis": {
            "llm_analysis_synthesis": {
                "evidence_repair_priorities": [
                    {
                        "schema_version": "claim_support_repair_priority_v1",
                        "gap_id": "ch_02_bad_claim_claim_support_entity_or_metric_mismatch",
                        "gap_type": "claim_support_entity_or_metric_mismatch",
                        "chapter_id": "ch_02",
                        "claim_id": "bad_claim",
                        "requirement_ids": ["H2_competition"],
                        "evidence_refs": ["EV-GTC"],
                        "required_fields": ["source", "entity_match"],
                        "proof_role": "support",
                        "success_criteria": "Only rebuild when the cited facts directly support the claim.",
                        "reject_if": ["off_topic_source"],
                        "writing_permission": "not_allowed_until_repaired",
                    }
                ]
            }
        }
    }

    summary = ingest_writer_package_artifacts(
        store,
        run_id="run-claim-repair",
        writer_package=writer_package,
        writer_report={},
    )

    assert summary["score_gap_count"] == 1
    gap = store.list_score_gaps("run-claim-repair", requirement_id="H2_competition")[0]
    assert gap["gap_type"] == "claim_support_entity_or_metric_mismatch"
    assert gap["missing"] == ["source", "entity_match"]
    assert gap["retry_plan"]["source_stage"] == "claim_repair_priority"
    assert gap["retry_plan"]["success_criteria"] == "Only rebuild when the cited facts directly support the claim."
    assert gap["retry_plan"]["reject_if"] == ["off_topic_source"]


def test_bridge_persists_research_reflection_memo_as_artifact_and_repair_gaps(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_LEDGER_PATH", str(tmp_path / "artifact_ledger.sqlite"))
    monkeypatch.setenv("ARTIFACT_OBJECT_ROOT", str(tmp_path / "objects"))
    store = ArtifactStore()
    store.upsert_run(run_id="run-reflection", query="AI Agent", status="running")

    memo = {
        "schema_version": "research_reflection_memo_v1",
        "status": "limited",
        "write_mode": "limited_review_draft",
        "allowed_for_writing": False,
        "coverage_by_requirement": [
            {
                "requirement_id": "H1_metric",
                "chapter_id": "ch_01",
                "status": "insufficient",
                "missing_fields": ["metric", "value", "unit", "period", "source"],
            }
        ],
        "next_search_task_seeds": [
            {
                "schema_version": "repair_task_seed_v2",
                "query": "enterprise AI agent adoption official statistics 2026 metric value unit period",
                "gap_id": "GAP-reflection-metric",
                "requirement_id": "H1_metric",
                "chapter_id": "ch_01",
                "section_id": "SEC-1",
                "gap_type": "metric_scope_period_unit_incomplete",
                "repair_status": "still_insufficient",
                "proof_role": "metric",
                "required_fields": ["metric", "value", "unit", "period", "source"],
                "required_source_level": ["A", "B"],
                "lane_targets": ["official_data", "market_research"],
                "success_criteria": "Only accept a traceable metric with value, unit, period, and source.",
                "reject_if": ["snippet_only", "no_date", "no_source_url"],
                "preferred_source_patterns": ["official_data", "industry_report_pdf"],
                "freshness_required": True,
                "max_cache_age_hours": 24,
                "allowed_for_writing": False,
            }
        ],
    }
    writer_package = {
        "structured_analysis": {
            "research_reflection_memo": memo,
        }
    }

    summary = ingest_writer_package_artifacts(
        store,
        run_id="run-reflection",
        writer_package=writer_package,
        writer_report={},
    )

    assert summary["research_reflection_artifact_count"] == 1
    assert summary["research_reflection_seed_count"] == 1
    gap = store.list_score_gaps("run-reflection", gap_id="GAP-reflection-metric")[0]
    assert gap["requirement_id"] == "H1_metric"
    assert gap["section_id"] == "SEC-1"
    assert gap["status"] == "still_insufficient"
    assert gap["retry_plan"]["source_stage"] == "research_reflection_memo"
    assert gap["retry_plan"]["required_fields"] == ["metric", "value", "unit", "period", "source"]
    assert gap["retry_plan"]["reject_if"] == ["snippet_only", "no_date", "no_source_url"]
    assert gap["retry_plan"]["success_criteria"].startswith("Only accept")
    assert gap["retry_plan"]["freshness_required"] is True
    assert gap["retry_plan"]["max_cache_age_hours"] == 24
    assert gap["retry_plan"]["allowed_for_writing"] is False

    edge_targets = {
        (edge["to_type"], edge["to_id"])
        for edge in store.traverse_lineage("run-reflection", "requirement", "H1_metric", max_depth=4)
    }
    assert any(target_type == "artifact" for target_type, _target_id in edge_targets)
    assert ("score_gap", "GAP-reflection-metric") in edge_targets
    with store._connect() as conn:
        artifact_count = conn.execute(
            "SELECT COUNT(*) FROM artifacts WHERE run_id=? AND artifact_type=?",
            ("run-reflection", "research_reflection_memo"),
        ).fetchone()[0]
    assert artifact_count == 1


def test_bridge_ingests_final_audit_findings_as_score_gaps(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_LEDGER_PATH", str(tmp_path / "artifact_ledger.sqlite"))
    monkeypatch.setenv("ARTIFACT_OBJECT_ROOT", str(tmp_path / "objects"))
    store = ArtifactStore()
    store.upsert_run(run_id="run-audit", query="AI Agent", status="running")

    summary = ingest_writer_package_artifacts(
        store,
        run_id="run-audit",
        writer_package={},
        writer_report={},
        final_audit_result={
            "audit": {
                "critical_findings": [
                    {
                        "gap_id": "AUDIT-1",
                        "requirement_id": "H1_case",
                        "section_id": "SEC-1",
                        "type": "unbacked_numeric_claim",
                        "severity": "fatal",
                        "message": "A numeric claim is not backed by an allowed fact card.",
                        "suggested_fix": "Repair the evidence binding before delivery.",
                    }
                ]
            }
        },
    )

    assert summary["score_gap_count"] == 1
    gap = store.list_score_gaps("run-audit", gap_id="AUDIT-1")[0]
    assert gap["requirement_id"] == "H1_case"
    assert gap["section_id"] == "SEC-1"
    assert gap["gap_type"] == "unbacked_numeric_claim"
    assert gap["severity"] == "fatal"
    assert gap["missing"] == ["A numeric claim is not backed by an allowed fact card."]


def test_bridge_marks_score_gap_cache_satisfied_from_cache_summary(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_LEDGER_PATH", str(tmp_path / "artifact_ledger.sqlite"))
    monkeypatch.setenv("ARTIFACT_OBJECT_ROOT", str(tmp_path / "objects"))
    store = ArtifactStore()
    store.upsert_run(run_id="run-cache-close", query="AI Agent", status="running")

    writer_package = {
        "evidence_package": {
            "evidence_gap_ledger": [
                {
                    "gap_id": "GAP-cache",
                    "requirement_id": "H1_metric",
                    "chapter_id": "ch_01",
                    "gap_type": "metric_scope_period_unit_incomplete",
                    "severity": "blocking",
                    "required_fields": ["metric", "value", "unit", "period", "source"],
                }
            ]
        }
    }
    writer_report = {
        "evidence_cache_summary": {
            "by_gap": {
                "GAP-cache": {
                    "requirement_id": "H1_metric",
                    "cache_hit_count": 1,
                    "cache_only_skip_count": 1,
                    "live_refresh_required_count": 0,
                }
            }
        }
    }

    summary = ingest_writer_package_artifacts(
        store,
        run_id="run-cache-close",
        writer_package=writer_package,
        writer_report=writer_report,
    )

    gap = store.list_score_gaps("run-cache-close", gap_id="GAP-cache")[0]
    assert summary["score_gap_status_update_count"] == 1
    assert gap["status"] == "cache_satisfied"
    assert gap["retry_plan"]["repair_status_source"] == "evidence_cache_summary"
    assert gap["retry_plan"]["cache_hit_count"] == 1
    assert gap["retry_plan"]["cache_only_skip_count"] == 1


def test_bridge_marks_score_gap_from_repair_gap_ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_LEDGER_PATH", str(tmp_path / "artifact_ledger.sqlite"))
    monkeypatch.setenv("ARTIFACT_OBJECT_ROOT", str(tmp_path / "objects"))
    store = ArtifactStore()
    store.upsert_run(run_id="run-live-close", query="AI Agent", status="running")

    writer_package = {
        "evidence_package": {
            "evidence_gap_ledger": [
                {
                    "gap_id": "GAP-found",
                    "requirement_id": "H1_case",
                    "gap_type": "case_evidence_missing",
                    "severity": "blocking",
                    "required_fields": ["source"],
                },
                {
                    "gap_id": "GAP-empty",
                    "requirement_id": "H2_counter",
                    "gap_type": "counter_evidence_missing",
                    "severity": "blocking",
                    "required_fields": ["source"],
                },
            ]
        }
    }
    writer_report = {
        "post_qa_repair_trace": [
            {
                "gap_ledger": [
                    {"gap_id": "GAP-found", "status": "evidence_found", "signal_count": 1, "result_count": 2},
                    {"gap_id": "GAP-empty", "status": "searched_no_signal", "signal_count": 0, "result_count": 1},
                ]
            }
        ]
    }

    summary = ingest_writer_package_artifacts(
        store,
        run_id="run-live-close",
        writer_package=writer_package,
        writer_report=writer_report,
    )

    found = store.list_score_gaps("run-live-close", gap_id="GAP-found")[0]
    empty = store.list_score_gaps("run-live-close", gap_id="GAP-empty")[0]
    assert summary["score_gap_status_update_count"] == 2
    assert found["status"] == "evidence_found"
    assert found["retry_plan"]["repair_status_source"] == "repair_gap_ledger"
    assert found["retry_plan"]["signal_count"] == 1
    assert empty["status"] == "still_insufficient"
    assert empty["retry_plan"]["repair_status_source"] == "repair_gap_ledger"
