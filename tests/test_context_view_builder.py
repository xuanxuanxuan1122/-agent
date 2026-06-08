from __future__ import annotations

from pathlib import Path

from rag_pipeline.cache.artifact_store import ArtifactStore
from rag_pipeline.context.context_view_builder import (
    build_analysis_context_view,
    build_repair_context_view,
    build_writer_context_view,
)


def _configure(tmp_path: Path, monkeypatch) -> ArtifactStore:
    monkeypatch.setenv("ARTIFACT_LEDGER_PATH", str(tmp_path / "artifact_ledger.sqlite"))
    monkeypatch.setenv("ARTIFACT_OBJECT_ROOT", str(tmp_path / "objects"))
    return ArtifactStore()


def test_analysis_context_view_only_returns_usable_fact_cards(tmp_path, monkeypatch):
    store = _configure(tmp_path, monkeypatch)
    store.upsert_run(run_id="run-a", query="q", status="running")
    store.upsert_evidence_requirement(
        run_id="run-a",
        requirement_id="H1_case",
        chapter_id="ch_01",
        proof_role="case",
        required_fields=["company", "use_case"],
        status="open",
    )
    store.upsert_source(
        run_id="run-a",
        run_source_id="SRC-1",
        source={"canonical_url": "https://example.com/a", "title": "Official A", "content_hash": "a", "source_level": "A"},
    )
    store.upsert_fact_card(
        run_id="run-a",
        fact_id="EV-good",
        requirement_id="H1_case",
        source_id="SRC-1",
        fact="A verified customer deployment was disclosed.",
        source_level="A",
        allowed_use="supporting",
        analysis_eligible=True,
        analysis_role="case",
        status="validated",
    )
    store.upsert_fact_card(
        run_id="run-a",
        fact_id="EV-rejected",
        requirement_id="H1_case",
        source_id="SRC-1",
        fact="Rejected clue.",
        status="rejected",
    )

    view = build_analysis_context_view("run-a", requirement_id="H1_case")

    assert view["status"] == "ready"
    assert [item["fact_id"] for item in view["usable_fact_cards"]] == ["EV-good"]
    assert view["source_registry_slice"][0]["run_source_id"] == "SRC-1"
    assert "raw_page" not in view
    assert "search_snippets" not in view


def test_writer_context_view_excludes_diagnostics_and_stale_facts(tmp_path, monkeypatch):
    store = _configure(tmp_path, monkeypatch)
    store.upsert_run(run_id="run-a", query="q", status="running")
    store.upsert_source(
        run_id="run-a",
        run_source_id="SRC-1",
        source={"canonical_url": "https://example.com/a", "title": "Official A", "content_hash": "a", "source_level": "A"},
    )
    store.upsert_fact_card(
        run_id="run-a",
        fact_id="EV-1",
        requirement_id="H1_case",
        source_id="SRC-1",
        fact="A verified customer deployment was disclosed.",
        source_level="A",
        allowed_use="supporting",
        status="validated",
    )
    store.upsert_fact_card(
        run_id="run-a",
        fact_id="EV-stale",
        requirement_id="H1_case",
        source_id="SRC-1",
        fact="Old evidence.",
        status="stale",
    )
    store.upsert_claim_unit(
        run_id="run-a",
        claim_id="CL-1",
        payload={
            "claim": "Enterprise agents are entering workflow deployment.",
            "claim_strength_ceiling": "moderate",
            "limitation_boundary": "Limited to disclosed enterprise samples.",
        },
        requirement_ids=["H1_case"],
        fact_ids=["EV-1", "EV-stale"],
        source_ids=["SRC-1"],
        status="validated",
    )
    store.upsert_section(
        run_id="run-a",
        section_id="SEC-1",
        payload={"claim": "Enterprise agents are entering workflow deployment.", "raw_page": "forbidden"},
        requirement_ids=["H1_case"],
        claim_ids=["CL-1"],
        used_fact_refs=["EV-1", "EV-stale"],
        evidence_backed=True,
        status="validated",
    )
    store.upsert_score_gap(
        run_id="run-a",
        gap_id="GAP-1",
        requirement_id="H1_case",
        chapter_id="ch_01",
        section_id="SEC-1",
        gap_type="metric_period_missing",
        missing=["period"],
        status="open",
    )

    view = build_writer_context_view("run-a", "SEC-1")

    assert view["status"] == "ready"
    assert [item["fact_id"] for item in view["usable_fact_cards"]] == ["EV-1"]
    assert "EV-stale" not in view["used_fact_refs"]
    assert "score_gaps" not in view
    assert "retry_plan" not in view
    assert "raw_page" not in view


def test_writer_context_view_does_not_fallback_to_all_claims_when_section_has_no_claim_ids(tmp_path, monkeypatch):
    store = _configure(tmp_path, monkeypatch)
    store.upsert_run(run_id="run-a", query="q", status="running")
    store.upsert_source(
        run_id="run-a",
        run_source_id="SRC-1",
        source={"canonical_url": "https://example.com/a", "title": "Official A", "content_hash": "a", "source_level": "A"},
    )
    store.upsert_fact_card(
        run_id="run-a",
        fact_id="EV-1",
        requirement_id="H1_case",
        source_id="SRC-1",
        fact="A verified customer deployment was disclosed.",
        source_level="A",
        allowed_use="supporting",
        status="validated",
    )
    store.upsert_claim_unit(
        run_id="run-a",
        claim_id="CL-other",
        payload={"claim": "A claim owned by another section."},
        requirement_ids=["H1_case"],
        fact_ids=["EV-1"],
        source_ids=["SRC-1"],
        status="validated",
    )
    store.upsert_section(
        run_id="run-a",
        section_id="SEC-empty",
        payload={"title": "No claim bound yet"},
        requirement_ids=["H1_case"],
        claim_ids=[],
        used_fact_refs=["EV-1"],
        evidence_backed=True,
        status="validated",
    )

    view = build_writer_context_view("run-a", "SEC-empty")

    assert view["status"] == "insufficient"
    assert view["instruction"] == "do_not_infer"
    assert view["claim_units"] == []
    assert view["usable_fact_cards"] == []
    assert view["used_fact_refs"] == []


def test_insufficient_views_return_do_not_infer(tmp_path, monkeypatch):
    store = _configure(tmp_path, monkeypatch)
    store.upsert_run(run_id="run-a", query="q", status="running")
    store.upsert_evidence_requirement(
        run_id="run-a",
        requirement_id="H1_case",
        chapter_id="ch_01",
        proof_role="case",
        missing=["customer case"],
        status="insufficient",
    )
    store.upsert_score_gap(
        run_id="run-a",
        gap_id="GAP-1",
        requirement_id="H1_case",
        chapter_id="ch_01",
        gap_type="case_missing",
        missing=["customer case"],
        retry_plan={"next_search_task": "AI Agent enterprise customer case official source"},
        status="open",
    )

    analysis_view = build_analysis_context_view("run-a", requirement_id="H1_case")
    repair_view = build_repair_context_view("run-a", requirement_id="H1_case")

    assert analysis_view["status"] == "insufficient"
    assert analysis_view["instruction"] == "do_not_infer"
    assert repair_view["score_gaps"][0]["gap_id"] == "GAP-1"
    assert "usable_fact_cards" not in repair_view


def test_empty_repair_context_view_returns_do_not_infer(tmp_path, monkeypatch):
    store = _configure(tmp_path, monkeypatch)
    store.upsert_run(run_id="run-a", query="q", status="running")
    store.upsert_evidence_requirement(
        run_id="run-a",
        requirement_id="H1_case",
        chapter_id="ch_01",
        proof_role="case",
        status="open",
    )

    view = build_repair_context_view("run-a", requirement_id="H1_case")

    assert view["status"] == "insufficient"
    assert view["instruction"] == "do_not_infer"
    assert view["score_gaps"] == []


def test_repair_context_view_keeps_unresolved_repair_statuses(tmp_path, monkeypatch):
    store = _configure(tmp_path, monkeypatch)
    store.upsert_run(run_id="run-a", query="q", status="running")
    store.upsert_score_gap(
        run_id="run-a",
        gap_id="GAP-still",
        requirement_id="H1_case",
        chapter_id="ch_01",
        gap_type="case_missing",
        missing=["source"],
        status="still_insufficient",
    )
    store.upsert_score_gap(
        run_id="run-a",
        gap_id="GAP-live",
        requirement_id="H1_case",
        chapter_id="ch_01",
        gap_type="case_missing",
        missing=["source"],
        status="live_search_required",
    )
    store.upsert_score_gap(
        run_id="run-a",
        gap_id="GAP-closed",
        requirement_id="H1_case",
        chapter_id="ch_01",
        gap_type="case_missing",
        missing=["source"],
        status="cache_satisfied",
    )

    view = build_repair_context_view("run-a", requirement_id="H1_case")

    assert view["status"] == "ready"
    assert [item["gap_id"] for item in view["score_gaps"]] == ["GAP-live", "GAP-still"]


def test_repair_context_view_builds_precise_search_task_seeds(tmp_path, monkeypatch):
    store = _configure(tmp_path, monkeypatch)
    store.upsert_run(run_id="run-a", query="q", status="running")
    store.upsert_evidence_requirement(
        run_id="run-a",
        requirement_id="H1_metric",
        chapter_id="ch_01",
        proof_role="metric",
        required_fields=["metric", "value", "unit", "period", "source"],
        min_source_level=["A", "B"],
        status="insufficient",
    )
    store.upsert_score_gap(
        run_id="run-a",
        gap_id="GAP-live",
        requirement_id="H1_metric",
        chapter_id="ch_01",
        section_id="SEC-1",
        gap_type="metric_scope_period_unit_incomplete",
        severity="blocking",
        missing=["metric", "value", "unit", "period", "source"],
        retry_plan={
            "source_stage": "evidence_cache_summary",
            "proof_role": "metric",
            "required_fields": ["metric", "value", "unit", "period", "source"],
            "required_source_level": ["A", "B"],
            "lane_targets": ["official_data"],
            "query_terms": ["enterprise AI agent adoption", "official metric"],
            "current_insufficiency": "Cache seed exists but live verification is still required.",
            "live_refresh_required_count": 1,
            "cache_hit_count": 1,
            "origin_payload": {"fact": "Forbidden cached fact text."},
        },
        status="live_search_required",
    )
    store.upsert_score_gap(
        run_id="run-a",
        gap_id="GAP-still",
        requirement_id="H1_metric",
        chapter_id="ch_01",
        gap_type="case_evidence_missing",
        severity="blocking",
        missing=["source"],
        retry_plan={
            "source_stage": "repair_gap_ledger",
            "proof_role": "case",
            "lane_targets": ["market_research"],
            "current_insufficiency": "Prior search returned no usable signal.",
            "result_count": 2,
            "signal_count": 0,
        },
        status="still_insufficient",
    )

    view = build_repair_context_view("run-a", requirement_id="H1_metric")
    by_gap = {item["gap_id"]: item for item in view["score_gaps"]}
    view_text = str(view)

    assert [item["gap_id"] for item in view["repair_task_seeds"]] == ["GAP-live", "GAP-still"]
    live_seed = by_gap["GAP-live"]["repair_task_seed"]
    assert view["repair_task_seeds"][0] == live_seed
    assert live_seed["gap_id"] == "GAP-live"
    assert live_seed["requirement_id"] == "H1_metric"
    assert live_seed["schema_version"] == "repair_task_seed_v2"
    assert live_seed["agent"] == "iqs"
    assert live_seed["proof_role"] == "metric"
    assert live_seed["required_fields"] == ["metric", "value", "unit", "period", "source"]
    assert live_seed["required_source_level"] == ["A", "B"]
    assert live_seed["lane_targets"] == ["official_data"]
    assert live_seed["allowed_for_writing"] is False
    assert "metric/value/unit/period/source" in live_seed["success_criteria"]
    assert {"snippet_only", "no_date", "no_source_url", "marketing_copy_only"} <= set(live_seed["reject_if"])
    assert live_seed["freshness_required"] is True
    assert "official_data" in live_seed["preferred_source_patterns"]
    assert live_seed["live_refresh_required"] is True
    assert live_seed["cache_seed_available"] is True
    assert "enterprise AI agent adoption" in live_seed["query"]
    assert "metric" in live_seed["query"]

    still_seed = by_gap["GAP-still"]["repair_task_seed"]
    assert still_seed["gap_id"] == "GAP-still"
    assert still_seed["repair_status"] == "still_insufficient"
    assert still_seed["avoid_repeating_failed_query"] is True
    assert "no_source_url" in still_seed["reject_if"]
    assert still_seed["previous_result_count"] == 2
    assert still_seed["previous_signal_count"] == 0
    assert "Prior search returned no usable signal" in still_seed["query"]

    assert "Forbidden cached fact text" not in view_text


def test_repair_context_view_exposes_cache_keyed_search_task_schedule(tmp_path, monkeypatch):
    store = _configure(tmp_path, monkeypatch)
    store.upsert_run(run_id="run-a", query="q", status="running")
    store.upsert_evidence_requirement(
        run_id="run-a",
        requirement_id="H1_metric",
        chapter_id="ch_01",
        proof_role="metric",
        required_fields=["metric", "value", "unit", "period", "source"],
        min_source_level=["A", "B"],
        freshness_required=True,
        max_cache_age_hours=24,
        status="insufficient",
    )
    store.upsert_score_gap(
        run_id="run-a",
        gap_id="GAP-live",
        requirement_id="H1_metric",
        chapter_id="ch_01",
        section_id="SEC-1",
        gap_type="metric_scope_period_unit_incomplete",
        severity="blocking",
        missing=["metric", "value", "unit", "period", "source"],
        retry_plan={
            "source_stage": "research_reflection_memo",
            "schema_version": "repair_task_seed_v2",
            "proof_role": "metric",
            "required_fields": ["metric", "value", "unit", "period", "source"],
            "required_source_level": ["A", "B"],
            "lane_targets": ["official_data"],
            "query_seed": "enterprise AI agent adoption official statistics",
            "success_criteria": "Only accept traceable metric evidence.",
            "reject_if": ["snippet_only", "no_date"],
            "freshness_required": True,
            "max_cache_age_hours": 24,
            "cache_hit_count": 1,
            "live_refresh_required_count": 1,
        },
        status="live_search_required",
    )
    store.upsert_score_gap(
        run_id="run-a",
        gap_id="GAP-closed",
        requirement_id="H1_metric",
        chapter_id="ch_01",
        gap_type="metric_scope_period_unit_incomplete",
        missing=["source"],
        status="cache_satisfied",
    )

    first = build_repair_context_view("run-a", requirement_id="H1_metric")
    second = build_repair_context_view("run-a", requirement_id="H1_metric")

    schedule = first["search_task_schedule"]
    assert schedule["schema_version"] == "repair_search_task_schedule_v1"
    assert schedule["cache_boundary"] == "ledger_score_gaps_only_no_fact_text"
    assert schedule["task_count"] == 1
    task = schedule["tasks"][0]
    assert task["task_id"].startswith("RT-GAP-live-")
    assert task["gap_id"] == "GAP-live"
    assert task["requirement_id"] == "H1_metric"
    assert task["query"] == first["repair_task_seeds"][0]["query"]
    assert task["cache_lookup_key"] == second["search_task_schedule"]["tasks"][0]["cache_lookup_key"]
    assert task["cache_scope"] == {
        "requirement_id": "H1_metric",
        "gap_id": "GAP-live",
        "proof_role": "metric",
        "required_fields": ["metric", "value", "unit", "period", "source"],
    }
    assert task["freshness_required"] is True
    assert task["max_cache_age_hours"] == 24
    assert task["live_refresh_required"] is True
    assert task["allowed_for_writing"] is False
    assert "GAP-closed" not in str(schedule)
    assert "traceable metric evidence" in task["success_criteria"]
    assert "snippet_only" in task["reject_if"]


def test_repair_context_view_prioritizes_section_audit_gaps(tmp_path, monkeypatch):
    store = _configure(tmp_path, monkeypatch)
    store.upsert_run(run_id="run-a", query="q", status="running")
    store.upsert_evidence_requirement(
        run_id="run-a",
        requirement_id="H1_metric",
        chapter_id="ch_01",
        proof_role="metric",
        required_fields=["metric", "value", "unit", "period", "source"],
        status="insufficient",
    )
    store.upsert_score_gap(
        run_id="run-a",
        gap_id="GAP-old-generic",
        requirement_id="H1_metric",
        chapter_id="ch_01",
        section_id="SEC-9",
        gap_type="case_evidence_missing",
        severity="medium",
        missing=["source"],
        retry_plan={
            "source_stage": "evidence_gap_ledger",
            "proof_role": "case",
            "current_insufficiency": "Generic case evidence is thin.",
        },
        status="open",
    )
    store.upsert_score_gap(
        run_id="run-a",
        gap_id="SA-SEC-1-counter-boundary-H1",
        requirement_id="H1_metric",
        chapter_id="ch_01",
        section_id="SEC-1",
        gap_type="counter_boundary_missing",
        severity="medium",
        missing=["counter_boundary"],
        retry_plan={
            "source_stage": "section_audit",
            "section_audit_version": "section_audit_v1",
            "finding_type": "section_missing_counter_boundary",
            "proof_role": "counter",
            "required_fields": ["source"],
            "allowed_for_writing": False,
        },
        status="open",
    )
    store.upsert_score_gap(
        run_id="run-a",
        gap_id="SA-SEC-1-metric-fields-H1",
        requirement_id="H1_metric",
        chapter_id="ch_01",
        section_id="SEC-1",
        gap_type="metric_scope_period_unit_incomplete",
        severity="blocking",
        missing=["unit", "period"],
        retry_plan={
            "source_stage": "section_audit",
            "section_audit_version": "section_audit_v1",
            "finding_type": "section_metric_missing_fields",
            "proof_role": "metric",
            "required_fields": ["metric", "value", "unit", "period", "source"],
            "allowed_for_writing": False,
        },
        status="open",
    )

    view = build_repair_context_view("run-a", requirement_id="H1_metric")

    assert [item["gap_id"] for item in view["repair_task_seeds"]][:2] == [
        "SA-SEC-1-metric-fields-H1",
        "SA-SEC-1-counter-boundary-H1",
    ]
    metric_seed = view["repair_task_seeds"][0]
    counter_seed = view["repair_task_seeds"][1]
    assert metric_seed["repair_priority_score"] > counter_seed["repair_priority_score"]
    assert "section_audit" in metric_seed["repair_priority_reason"]
    assert metric_seed["required_field_focus"] == "period"
    assert metric_seed["repair_route"] == "metric_source_search"
    assert "metric_scope_period_unit_incomplete" in metric_seed["query"]
    assert counter_seed["proof_role"] == "counter"
    assert counter_seed["repair_route"] == "counter_evidence_search"
    assert "counter_evidence" in counter_seed["source_priority"]
    assert "support_only_counter_missing" in counter_seed["reject_if"]
    assert "counter_boundary_missing" in counter_seed["query"]
    assert view["repair_task_seeds"][-1]["gap_id"] == "GAP-old-generic"


def test_repair_context_view_returns_direction_only_without_quoteable_payload(tmp_path, monkeypatch):
    store = _configure(tmp_path, monkeypatch)
    store.upsert_run(run_id="run-a", query="q", status="running")
    store.upsert_evidence_requirement(
        run_id="run-a",
        requirement_id="H1_metric",
        chapter_id="ch_01",
        proof_role="metric",
        required_fields=["metric", "value", "unit", "period", "source"],
        status="insufficient",
    )
    store.upsert_score_gap(
        run_id="run-a",
        gap_id="GAP-metric",
        requirement_id="H1_metric",
        chapter_id="ch_01",
        section_id="SEC-1",
        gap_type="metric_scope_period_unit_incomplete",
        severity="blocking",
        missing=["metric", "value", "unit", "period", "source"],
        retry_plan={
            "source_stage": "evidence_gap_ledger",
            "proof_role": "metric",
            "required_fields": ["metric", "value", "unit", "period", "source"],
            "lane_targets": ["official_data"],
            "query_terms": ["enterprise AI agent adoption"],
            "current_evidence_refs": ["EV-old"],
            "current_insufficiency": "The current metric has no period or unit.",
            "origin_payload": {
                "fact": "Forbidden cached fact text.",
                "section_draft": "Forbidden old section body.",
                "raw_page": "Forbidden raw page.",
            },
            "allowed_for_writing": False,
        },
        status="open",
    )

    view = build_repair_context_view("run-a", requirement_id="H1_metric")
    view_text = str(view)

    assert view["status"] == "ready"
    assert view["instruction"] == "find_missing_evidence_do_not_write_body_text"
    assert view["score_gaps"][0]["proof_role"] == "metric"
    assert view["score_gaps"][0]["required_fields"] == ["metric", "value", "unit", "period", "source"]
    assert view["score_gaps"][0]["lane_targets"] == ["official_data"]
    assert view["score_gaps"][0]["query_terms"] == ["enterprise AI agent adoption"]
    assert view["score_gaps"][0]["current_evidence_refs"] == ["EV-old"]
    assert view["score_gaps"][0]["allowed_for_writing"] is False
    assert "Forbidden cached fact text" not in view_text
    assert "Forbidden old section body" not in view_text
    assert "Forbidden raw page" not in view_text


def test_repair_context_view_turns_claim_support_gap_into_search_task_seed(tmp_path, monkeypatch):
    store = _configure(tmp_path, monkeypatch)
    store.upsert_run(run_id="run-claim-gap", query="q", status="running")
    store.upsert_score_gap(
        run_id="run-claim-gap",
        gap_id="GAP-claim-support",
        requirement_id="H2_competition",
        chapter_id="ch_02",
        section_id="",
        gap_type="claim_support_entity_or_metric_mismatch",
        severity="blocking",
        missing=["source", "entity_match"],
        retry_plan={
            "source_stage": "claim_repair_priority",
            "proof_role": "support",
            "required_fields": ["source", "entity_match"],
            "current_evidence_refs": ["EV-GTC"],
            "success_criteria": "Only rebuild when cited facts directly support the claim.",
            "reject_if": ["off_topic_source"],
            "allowed_for_writing": False,
        },
        status="open",
    )

    view = build_repair_context_view("run-claim-gap", requirement_id="H2_competition")
    seed = view["repair_task_seeds"][0]
    task = view["search_task_schedule"]["tasks"][0]

    assert view["status"] == "ready"
    assert seed["gap_type"] == "claim_support_entity_or_metric_mismatch"
    assert seed["source_stage"] == "claim_repair_priority"
    assert seed["required_fields"] == ["source", "entity_match"]
    assert seed["success_criteria"] == "Only rebuild when cited facts directly support the claim."
    assert seed["reject_if"] == ["off_topic_source"]
    assert seed["allowed_for_writing"] is False
    assert task["gap_id"] == "GAP-claim-support"
    assert task["cache_scope"]["requirement_id"] == "H2_competition"
    assert task["cache_scope"]["required_fields"] == ["source", "entity_match"]
