from __future__ import annotations

from rag_pipeline.agents import brain_agent as brain_agent_module


def test_analysis_repair_priorities_sync_back_to_evidence_package():
    evidence_package = {
        "evidence_gap_ledger": [
            {"gap_id": "existing-gap", "gap_type": "counter_evidence_missing"}
        ]
    }
    structured_analysis = {
        "evidence_repair_priorities": [
            {
                "schema_version": "claim_support_repair_priority_v1",
                "gap_id": "semantic-gap",
                "claim_id": "claim-1",
                "gap_type": "claim_semantic_support_mismatch",
                "required_fields": ["source"],
                "source_stage": "semantic_claim_support_judge",
            }
        ],
        "evidence_gap_ledger": [
            {
                "gap_id": "binding-gap",
                "claim_id": "claim-2",
                "gap_type": "claim_support_entity_or_metric_mismatch",
                "required_fields": ["metric", "value", "source"],
            }
        ],
    }

    summary = brain_agent_module._sync_analysis_repair_priorities_to_evidence_package(
        evidence_package,
        structured_analysis,
    )

    gap_ids = {item["gap_id"] for item in evidence_package["evidence_gap_ledger"]}
    assert {"existing-gap", "semantic-gap", "binding-gap"} <= gap_ids
    assert summary["added_gap_count"] == 2
    semantic_gap = next(item for item in evidence_package["evidence_gap_ledger"] if item["gap_id"] == "semantic-gap")
    assert semantic_gap["repair_route"] == "evidence_search"
    assert semantic_gap["allowed_for_writing"] is False
    assert evidence_package["evidence_repair_priorities"][0]["gap_id"] == "semantic-gap"


def test_repair_context_seeds_become_sanitized_followup_tasks():
    view = {
        "status": "ready",
        "repair_task_seeds": [
            {
                "schema_version": "repair_task_seed_v2",
                "query": "enterprise AI agent adoption official metric source",
                "agent": "iqs",
                "gap_id": "GAP-live",
                "requirement_id": "H1_metric",
                "proof_role": "metric",
                "required_fields": ["metric", "value", "unit", "period", "source"],
                "required_source_level": ["A", "B"],
                "lane_targets": ["official_data"],
                "success_criteria": "Only count as repaired when metric/value/unit/period/source are all present.",
                "reject_if": ["snippet_only", "no_date", "no_source_url", "marketing_copy_only"],
                "preferred_source_patterns": ["official_data", "market_research"],
                "repair_priority_score": 170,
                "repair_priority_reason": "source_stage:section_audit, severity:blocking",
                "freshness_required": True,
                "max_cache_age_hours": 24,
                "cache_seed_available": True,
                "live_refresh_required": True,
                "origin_payload": {"fact": "Forbidden cached fact text."},
                "raw_page": "Forbidden raw page.",
            }
        ],
    }

    tasks, skipped = brain_agent_module._repair_tasks_from_context_view(
        view,
        origin_node="artifact_ledger",
        loop_name="ledger_repair",
        max_tasks=4,
        seen_keys=set(),
    )

    assert skipped == 0
    assert len(tasks) == 1
    assert tasks[0]["gap_id"] == "GAP-live"
    assert tasks[0]["requirement_id"] == "H1_metric"
    assert tasks[0]["schema_version"] == "repair_task_seed_v2"
    assert tasks[0]["proof_role"] == "metric"
    assert tasks[0]["required_fields"] == ["metric", "value", "unit", "period", "source"]
    assert tasks[0]["required_source_level"] == ["A", "B"]
    assert tasks[0]["success_criteria"].startswith("Only count")
    assert "snippet_only" in tasks[0]["reject_if"]
    assert tasks[0]["preferred_source_patterns"] == ["official_data", "market_research"]
    assert tasks[0]["repair_priority_score"] == 170
    assert "section_audit" in tasks[0]["repair_priority_reason"]
    assert tasks[0]["freshness_required"] is True
    assert tasks[0]["max_cache_age_hours"] == 24
    assert tasks[0]["origin_node"] == "artifact_ledger"
    assert tasks[0]["loop_name"] == "ledger_repair"
    assert tasks[0]["cache_seed_available"] is True
    assert tasks[0]["live_refresh_required"] is True
    assert "origin_payload" not in tasks[0]
    assert "raw_page" not in tasks[0]
    assert "Forbidden cached fact text" not in str(tasks[0])


def test_repair_context_schedule_tasks_are_preferred_and_keep_cache_scope():
    view = {
        "status": "ready",
        "repair_task_seeds": [
            {
                "query": "old seed should not be scheduled",
                "gap_id": "GAP-old",
                "requirement_id": "H1_metric",
                "proof_role": "metric",
            }
        ],
        "search_task_schedule": {
            "schema_version": "repair_search_task_schedule_v1",
            "tasks": [
                {
                    "schema_version": "repair_search_task_v1",
                    "query": "enterprise AI agent adoption official statistics period unit",
                    "agent": "iqs",
                    "gap_id": "GAP-live",
                    "requirement_id": "H1_metric",
                    "proof_role": "metric",
                    "required_fields": ["metric", "value", "unit", "period", "source"],
                    "cache_lookup_key": "repair:abc123",
                    "cache_scope": {
                        "requirement_id": "H1_metric",
                        "gap_id": "GAP-live",
                        "proof_role": "metric",
                        "required_fields": ["metric", "value", "unit", "period", "source"],
                    },
                    "freshness_required": True,
                    "max_cache_age_hours": 24,
                    "live_refresh_required": True,
                    "allowed_for_writing": False,
                    "raw_page": "Forbidden raw page.",
                }
            ],
        },
    }

    tasks, skipped = brain_agent_module._repair_tasks_from_context_view(
        view,
        origin_node="artifact_ledger",
        loop_name="ledger_repair",
        max_tasks=4,
        seen_keys=set(),
    )

    assert skipped == 0
    assert len(tasks) == 1
    assert tasks[0]["gap_id"] == "GAP-live"
    assert tasks[0]["query"].startswith("enterprise AI agent")
    assert tasks[0]["cache_lookup_key"] == "repair:abc123"
    assert tasks[0]["cache_scope"]["gap_id"] == "GAP-live"
    assert tasks[0]["freshness_required"] is True
    assert tasks[0]["live_refresh_required"] is True
    assert "GAP-old" not in str(tasks)
    assert "raw_page" not in tasks[0]


def test_repair_context_seed_dedup_uses_shared_seen_keys():
    view = {
        "status": "ready",
        "repair_task_seeds": [
            {
                "query": "enterprise AI agent adoption official metric source",
                "agent": "iqs",
                "gap_id": "GAP-live",
                "proof_role": "metric",
            }
        ],
    }
    seen = set()

    first, first_skipped = brain_agent_module._repair_tasks_from_context_view(
        view,
        origin_node="artifact_ledger",
        loop_name="ledger_repair",
        max_tasks=4,
        seen_keys=seen,
    )
    second, second_skipped = brain_agent_module._repair_tasks_from_context_view(
        view,
        origin_node="artifact_ledger",
        loop_name="ledger_repair",
        max_tasks=4,
        seen_keys=seen,
    )

    assert len(first) == 1
    assert first_skipped == 0
    assert second == []
    assert second_skipped == 1


def test_evidence_preflight_includes_ledger_repair_seed(monkeypatch):
    captured = {}

    def fake_ledger_items(*, state, max_tasks, seen_keys):
        return (
            [
                {
                    "query": "enterprise AI agent adoption official metric source",
                    "agent": "iqs",
                    "gap_id": "GAP-live",
                    "requirement_id": "H1_metric",
                    "proof_role": "metric",
                }
            ],
            {"status": "ready", "repair_task_seeds": [{"gap_id": "GAP-live"}]},
            0,
        )

    def fake_binder(**_kwargs):
        return {"evidence_refinement_plan": {"status": "no_tasks", "top_priorities": [], "follow_up_queries": []}}

    def fake_followups(*, follow_up_queries, round_number, state):
        captured["follow_up_queries"] = follow_up_queries
        captured["round_number"] = round_number
        return [
            {
                "status": "success",
                "answer": "",
                "key_sources": [],
                "raw_data_points": [],
                "search_task": follow_up_queries[0],
            }
        ]

    monkeypatch.setattr(brain_agent_module, "_ledger_repair_items_from_state", fake_ledger_items)
    monkeypatch.setattr("rag_pipeline.agents.evidence_binder.run_evidence_binder", fake_binder)
    monkeypatch.setattr(brain_agent_module, "run_followup_queries", fake_followups)

    result = brain_agent_module._run_evidence_preflight_round(
        state={"metadata": {}, "stage_snapshot_run_id": "run-a"},
        children={},
        evidence_pool=[],
        evidence_package={},
        structured_analysis={},
        report_plan={},
        query="AI agent adoption",
        max_followups=4,
        started=0.0,
    )

    trace = result["evidence_preflight_trace"][0]
    assert trace["ledger_repair_seed_count"] == 1
    assert trace["ledger_repair_skipped_count"] == 0
    assert captured["follow_up_queries"][0]["gap_id"] == "GAP-live"
    assert captured["follow_up_queries"][0]["origin_node"] == "artifact_ledger"
    assert result["updated"] is False


def test_post_qa_repair_includes_ledger_repair_seed(monkeypatch):
    captured = {}

    def fake_ledger_items(*, state, max_tasks, seen_keys):
        return (
            [
                {
                    "query": "enterprise AI agent adoption official metric source",
                    "agent": "iqs",
                    "gap_id": "GAP-postqa",
                    "requirement_id": "H2_metric",
                    "proof_role": "metric",
                }
            ],
            {"status": "ready", "repair_task_seeds": [{"gap_id": "GAP-postqa"}]},
            0,
        )

    def fake_followups(*, follow_up_queries, round_number, state):
        captured["follow_up_queries"] = follow_up_queries
        captured["round_number"] = round_number
        return [
            {
                "status": "success",
                "answer": "",
                "key_sources": [],
                "raw_data_points": [],
                "search_task": follow_up_queries[0],
            }
        ]

    monkeypatch.setattr(brain_agent_module, "_post_qa_repair_needed", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        brain_agent_module,
        "_post_qa_repair_plan",
        lambda *_args, **_kwargs: {
            "status": "planned",
            "evidence_followups": [],
            "rewrite_required": False,
        },
    )
    monkeypatch.setattr(brain_agent_module, "_ledger_repair_items_from_state", fake_ledger_items)
    monkeypatch.setattr(brain_agent_module, "run_followup_queries", fake_followups)

    result = brain_agent_module._run_post_qa_repair_round(
        state={"metadata": {}, "stage_snapshot_run_id": "run-a"},
        children={},
        best={
            "writer_report": {"report_status": "needs_review"},
            "evidence_pool": [],
            "evidence_package": {},
            "structured_analysis": {},
            "analysis_state": {},
            "layout_refinement_trace": [],
        },
        report_plan={},
        query="AI agent adoption",
        search_task_schedule={},
        lane_coverage={},
        max_followups=4,
        started=0.0,
    )

    trace = result["post_qa_repair_trace"][0]
    assert trace["ledger_repair_seed_count"] == 1
    assert trace["ledger_repair_skipped_count"] == 0
    assert captured["follow_up_queries"][0]["gap_id"] == "GAP-postqa"
    assert captured["follow_up_queries"][0]["origin_node"] == "artifact_ledger"
    assert trace["status"] == "no_new_evidence_signal"


def test_layout_refinement_includes_ledger_repair_seed(monkeypatch):
    captured = {}

    def fake_ledger_items(*, state, max_tasks, seen_keys):
        return (
            [
                {
                    "query": "enterprise AI agent adoption official metric source",
                    "agent": "iqs",
                    "gap_id": "GAP-layout",
                    "requirement_id": "H3_metric",
                    "proof_role": "metric",
                }
            ],
            {"status": "ready", "repair_task_seeds": [{"gap_id": "GAP-layout"}]},
            0,
        )

    def fake_writer_agent(**_kwargs):
        return {
            "writer_report": {
                "report_status": "needs_review",
                "report_markdown": "draft",
                "qa_result": {"passed": False},
                "layout_plan": {"layout_gaps": []},
            }
        }

    def fake_followups(*, follow_up_queries, round_number, state):
        captured["follow_up_queries"] = follow_up_queries
        captured["round_number"] = round_number
        return [
            {
                "status": "success",
                "answer": "",
                "key_sources": [],
                "raw_data_points": [],
                "search_task": follow_up_queries[0],
            }
        ]

    def fake_post_qa(*, best, **_kwargs):
        return {**best, "post_qa_repair_trace": [{"status": "not_tested"}]}

    monkeypatch.setattr(
        brain_agent_module,
        "_run_evidence_preflight_round",
        lambda **_kwargs: {"updated": False, "evidence_preflight_trace": [{"status": "not_tested"}]},
    )
    monkeypatch.setattr(brain_agent_module, "run_writer_agent", fake_writer_agent)
    monkeypatch.setattr(brain_agent_module, "_attach_reformatter_preflight_feedback", lambda **kwargs: kwargs["writer_report"])
    monkeypatch.setattr(brain_agent_module, "_layout_followup_queries_from_writer_report", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(brain_agent_module, "_ledger_repair_items_from_state", fake_ledger_items)
    monkeypatch.setattr(brain_agent_module, "run_followup_queries", fake_followups)
    monkeypatch.setattr(brain_agent_module, "_run_post_qa_repair_round", fake_post_qa)

    result = brain_agent_module.run_writer_with_layout_refinement(
        state={
            "metadata": {},
            "stage_snapshot_run_id": "run-a",
            "query": "AI agent adoption",
            "enable_followup_loop": True,
            "layout_max_refinement_rounds": 1,
        },
        children={},
        evidence_pool=[],
        evidence_package={},
        structured_analysis={},
        report_plan={},
        analysis_state={},
    )

    initial_trace = result["layout_refinement_trace"][0]
    round_trace = result["layout_refinement_trace"][1]
    assert initial_trace["ledger_repair_seed_count"] == 1
    assert initial_trace["ledger_repair_skipped_count"] == 0
    assert captured["follow_up_queries"][0]["gap_id"] == "GAP-layout"
    assert captured["follow_up_queries"][0]["origin_node"] == "artifact_ledger"
    assert round_trace["stop_reason"] == "no_new_evidence_signal"


def test_layout_refinement_syncs_writer_section_audit_before_ledger_repair(tmp_path, monkeypatch):
    captured = {}
    run_id = "run-layout-section-audit"

    monkeypatch.setenv("ARTIFACT_LEDGER_PATH", str(tmp_path / "artifact_ledger.sqlite"))
    monkeypatch.setenv("ARTIFACT_OBJECT_ROOT", str(tmp_path / "objects"))
    monkeypatch.setenv("ARTIFACT_LEDGER_ENABLED", "true")
    monkeypatch.setenv("BRAIN_ENABLE_LEDGER_REPAIR_CONTEXT", "true")

    def fake_writer_agent(**_kwargs):
        return {
            "writer_report": {
                "report_status": "needs_review",
                "report_markdown": "draft",
                "qa_result": {"passed": False},
                "layout_plan": {"layout_gaps": []},
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
        }

    def fake_followups(*, follow_up_queries, round_number, state):
        captured["follow_up_queries"] = follow_up_queries
        return [
            {
                "status": "success",
                "answer": "",
                "key_sources": [],
                "raw_data_points": [],
                "search_task": follow_up_queries[0],
            }
        ]

    def fake_post_qa(*, best, **_kwargs):
        return {**best, "post_qa_repair_trace": [{"status": "not_tested"}]}

    monkeypatch.setattr(
        brain_agent_module,
        "_run_evidence_preflight_round",
        lambda **_kwargs: {"updated": False, "evidence_preflight_trace": [{"status": "not_tested"}]},
    )
    monkeypatch.setattr(brain_agent_module, "run_writer_agent", fake_writer_agent)
    monkeypatch.setattr(brain_agent_module, "_attach_reformatter_preflight_feedback", lambda **kwargs: kwargs["writer_report"])
    monkeypatch.setattr(brain_agent_module, "_layout_followup_queries_from_writer_report", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(brain_agent_module, "run_followup_queries", fake_followups)
    monkeypatch.setattr(brain_agent_module, "_run_post_qa_repair_round", fake_post_qa)

    result = brain_agent_module.run_writer_with_layout_refinement(
        state={
            "metadata": {},
            "stage_snapshot_run_id": run_id,
            "query": "AI agent adoption",
            "enable_followup_loop": True,
            "layout_max_refinement_rounds": 1,
            "research_plan": {
                "evidence_goals": [
                    {
                        "requirement_id": "H1_metric",
                        "chapter_id": "ch_01",
                        "proof_role": "metric",
                        "required_fields": ["metric", "value", "unit", "period", "source"],
                    }
                ]
            },
        },
        children={},
        evidence_pool=[],
        evidence_package={
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
        structured_analysis={},
        report_plan={},
        analysis_state={},
    )

    initial_trace = result["layout_refinement_trace"][0]
    assert initial_trace["writer_artifact_ledger_sync"]["score_gap_count"] == 2
    assert initial_trace["ledger_repair_seed_count"] == 2
    assert [item["gap_type"] for item in captured["follow_up_queries"][:2]] == [
        "metric_scope_period_unit_incomplete",
        "counter_boundary_missing",
    ]
    assert captured["follow_up_queries"][0]["origin_node"] == "artifact_ledger"


def test_evidence_preflight_syncs_gap_ledger_before_building_ledger_seed(tmp_path, monkeypatch):
    captured = {}
    run_id = "run-preflight-ledger"

    monkeypatch.setenv("ARTIFACT_LEDGER_PATH", str(tmp_path / "artifact_ledger.sqlite"))
    monkeypatch.setenv("ARTIFACT_OBJECT_ROOT", str(tmp_path / "objects"))
    monkeypatch.setenv("ARTIFACT_LEDGER_ENABLED", "true")
    monkeypatch.setenv("BRAIN_ENABLE_LEDGER_REPAIR_CONTEXT", "true")

    def fake_binder(**_kwargs):
        return {"evidence_refinement_plan": {"status": "no_tasks", "top_priorities": [], "follow_up_queries": []}}

    def fake_followups(*, follow_up_queries, round_number, state):
        captured["follow_up_queries"] = follow_up_queries
        captured["round_number"] = round_number
        return [
            {
                "status": "success",
                "answer": "",
                "key_sources": [],
                "raw_data_points": [],
                "search_task": follow_up_queries[0],
            }
        ]

    monkeypatch.setattr("rag_pipeline.agents.evidence_binder.run_evidence_binder", fake_binder)
    monkeypatch.setattr(brain_agent_module, "run_followup_queries", fake_followups)

    result = brain_agent_module._run_evidence_preflight_round(
        state={
            "metadata": {},
            "stage_snapshot_run_id": run_id,
            "research_plan": {
                "evidence_goals": [
                    {
                        "goal_id": "H1_metric",
                        "chapter_id": "ch_01",
                        "chapter_title": "Demand validation",
                        "proof_role": "metric",
                        "required_fields": ["metric", "value", "unit", "period", "source"],
                        "required_source_levels": ["A", "B"],
                    }
                ],
                "search_tasks": [
                    {
                        "task_id": "ST-H1",
                        "requirement_id": "H1_metric",
                        "chapter_id": "ch_01",
                        "proof_role": "metric",
                    }
                ],
            },
        },
        children={},
        evidence_pool=[],
        evidence_package={
            "evidence_gap_ledger": [
                {
                    "gap_id": "GAP-metric",
                    "chapter_id": "ch_01",
                    "proof_role": "metric",
                    "required_fields": ["metric", "value", "unit", "period", "source"],
                    "query_terms": ["enterprise AI agent adoption"],
                    "status": "live_search_required",
                    "repair_route": "evidence_search",
                    "why_current_evidence_insufficient": "Need authoritative adoption metric with period and source.",
                    "source": "evidence_gap_ledger",
                }
            ]
        },
        structured_analysis={},
        report_plan={},
        query="AI agent adoption",
        max_followups=2,
        started=0.0,
    )

    trace = result["evidence_preflight_trace"][0]
    assert trace["ledger_repair_view_status"] == "ready"
    assert trace["ledger_repair_seed_count"] == 1
    assert trace["ledger_gap_sync"]["score_gap_count"] == 1
    assert captured["follow_up_queries"][0]["gap_id"] == "GAP-metric"
    assert captured["follow_up_queries"][0]["requirement_id"] == "H1_metric"
    assert captured["follow_up_queries"][0]["origin_node"] == "artifact_ledger"
