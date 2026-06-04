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
