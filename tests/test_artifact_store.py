from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from rag_pipeline.cache.artifact_store import ArtifactStore


def _store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(
        path=tmp_path / "artifact_ledger.sqlite",
        object_root=tmp_path / "objects",
        inline_max_bytes=96,
    )


def test_artifact_store_initializes_schema_wal_and_run_scoped_requirements(tmp_path, monkeypatch):
    monkeypatch.setenv("EVIDENCE_CACHE_READ_ENABLED", "false")
    store = _store(tmp_path)

    store.upsert_run(run_id="run-a", query="AI Agent adoption", report_type="industry", status="running")
    store.upsert_run(run_id="run-b", query="AI Agent adoption", report_type="industry", status="running")
    store.upsert_evidence_requirement(
        run_id="run-a",
        requirement_id="H1_case",
        chapter_id="ch_01",
        proof_role="case",
        required_fields=["company", "use_case", "source_ref"],
        claim_strength_ceiling="directional",
        status="open",
    )
    store.upsert_evidence_requirement(
        run_id="run-b",
        requirement_id="H1_case",
        chapter_id="ch_02",
        proof_role="metric",
        required_fields=["metric", "value", "period", "source_ref"],
        claim_strength_ceiling="moderate",
        status="open",
    )

    assert store.sqlite_journal_mode().lower() == "wal"
    assert store.get_evidence_requirement("run-a", "H1_case")["chapter_id"] == "ch_01"
    assert store.get_evidence_requirement("run-b", "H1_case")["chapter_id"] == "ch_02"


def test_run_source_ids_are_local_and_map_to_canonical_sources(tmp_path):
    store = _store(tmp_path)
    store.upsert_run(run_id="run-a", query="q", status="running")
    store.upsert_run(run_id="run-b", query="q", status="running")

    source_a = store.upsert_source(
        run_id="run-a",
        run_source_id="SRC-001",
        source={
            "canonical_url": "https://example.com/a",
            "title": "A",
            "publisher": "Example",
            "content_hash": "hash-a",
            "source_level": "A",
            "status": "validated",
        },
    )
    source_b = store.upsert_source(
        run_id="run-b",
        run_source_id="SRC-001",
        source={
            "canonical_url": "https://example.com/b",
            "title": "B",
            "publisher": "Example",
            "content_hash": "hash-b",
            "source_level": "B",
            "status": "validated",
        },
    )

    assert source_a["canonical_source_id"] != source_b["canonical_source_id"]
    assert store.resolve_run_source("run-a", "SRC-001")["canonical_url"] == "https://example.com/a"
    assert store.resolve_run_source("run-b", "SRC-001")["canonical_url"] == "https://example.com/b"


def test_source_upsert_drops_nested_dict_publisher_metadata(tmp_path):
    store = _store(tmp_path)
    store.upsert_run(run_id="run-a", query="q", status="running")
    nested_source = {
        "title": "IDC source that should not become publisher",
        "url": "https://mfe-prod.idc.com/getdoc.jsp?containerId=prCHC53669525",
        "source_type": "research",
        "web_final_score": 0.68,
    }

    store.upsert_source(
        run_id="run-a",
        run_source_id="SRC-001",
        source={
            "canonical_url": "https://www.stats.gov.cn/example",
            "title": "Government statistics page",
            "publisher": nested_source,
            "source": nested_source,
            "source_level": "A",
            "status": "validated",
        },
    )

    resolved = store.resolve_run_source("run-a", "SRC-001")

    assert resolved["publisher"] == ""
    assert "web_final_score" not in resolved["publisher"]


def test_artifact_payloads_inline_small_json_and_spill_large_payloads_to_files(tmp_path):
    store = _store(tmp_path)
    store.upsert_run(run_id="run-a", query="q", status="running")

    small = store.record_artifact(
        run_id="run-a",
        stage="research_plan",
        artifact_type="research_plan",
        payload={"chapters": [{"chapter_id": "ch_01"}]},
        status="validated",
    )
    large = store.record_artifact(
        run_id="run-a",
        stage="writer_report",
        artifact_type="writer_report",
        payload={"text": "x" * 500},
        status="validated",
    )

    small_row = store.get_artifact(small["artifact_id"])
    large_row = store.get_artifact(large["artifact_id"])

    assert json.loads(small_row["payload_json"])["chapters"][0]["chapter_id"] == "ch_01"
    assert small_row["storage_uri"] == ""
    assert large_row["payload_json"] == ""
    assert large_row["storage_uri"]
    assert Path(large_row["storage_uri"]).exists()
    assert json.loads(Path(large_row["storage_uri"]).read_text(encoding="utf-8"))["text"].startswith("xxx")


def test_lineage_edges_allow_requirement_to_score_gap_traversal(tmp_path):
    store = _store(tmp_path)
    store.upsert_run(run_id="run-a", query="q", status="running")
    store.upsert_evidence_requirement(
        run_id="run-a",
        requirement_id="H1_case",
        chapter_id="ch_01",
        proof_role="case",
        status="open",
    )
    store.upsert_fact_card(
        run_id="run-a",
        fact_id="EV-1",
        requirement_id="H1_case",
        source_id="SRC-1",
        fact="Salesforce disclosed enterprise agent workflow deployment.",
        status="validated",
    )
    store.upsert_claim_unit(
        run_id="run-a",
        claim_id="CL-1",
        payload={"claim": "Enterprise agents are entering workflow deployment."},
        requirement_ids=["H1_case"],
        fact_ids=["EV-1"],
        source_ids=["SRC-1"],
        status="validated",
    )
    store.upsert_section(
        run_id="run-a",
        section_id="SEC-1",
        payload={"claim": "Enterprise agents are entering workflow deployment."},
        requirement_ids=["H1_case"],
        claim_ids=["CL-1"],
        used_fact_refs=["EV-1"],
        evidence_backed=True,
        status="validated",
    )
    store.upsert_score_gap(
        run_id="run-a",
        gap_id="GAP-1",
        requirement_id="H1_case",
        chapter_id="ch_01",
        section_id="SEC-1",
        gap_type="weak_evidence_binding",
        severity="warning",
        missing=["A/B corroboration"],
        status="open",
    )
    store.add_lineage_edge("run-a", "requirement", "H1_case", "fact_card", "EV-1", "requires")
    store.add_lineage_edge("run-a", "fact_card", "EV-1", "claim_unit", "CL-1", "supports")
    store.add_lineage_edge("run-a", "claim_unit", "CL-1", "section", "SEC-1", "renders")
    store.add_lineage_edge("run-a", "section", "SEC-1", "score_gap", "GAP-1", "diagnosed_by")

    traversal = store.traverse_lineage("run-a", "requirement", "H1_case", max_depth=5)

    assert ("fact_card", "EV-1") in {(item["to_type"], item["to_id"]) for item in traversal}
    assert ("section", "SEC-1") in {(item["to_type"], item["to_id"]) for item in traversal}
    assert ("score_gap", "GAP-1") in {(item["to_type"], item["to_id"]) for item in traversal}


def test_lineage_edge_insert_is_idempotent(tmp_path):
    store = _store(tmp_path)
    store.upsert_run(run_id="run-a", query="q", status="running")

    store.add_lineage_edge("run-a", "requirement", "H1_case", "score_gap", "GAP-1", "gap")
    store.add_lineage_edge("run-a", "requirement", "H1_case", "score_gap", "GAP-1", "gap")

    with store._connect() as conn:
        count = conn.execute(
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
            ("run-a", "requirement", "H1_case", "score_gap", "GAP-1", "gap"),
        ).fetchone()[0]

    assert count == 1


def test_lineage_edge_insert_respects_disabled_store(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_LEDGER_ENABLED", "false")
    store = _store(tmp_path)

    added = store.add_lineage_edge("run-a", "requirement", "H1_case", "score_gap", "GAP-1", "gap")

    assert added is False


def test_connect_context_manager_closes_sqlite_connection(tmp_path):
    store = _store(tmp_path)
    store.upsert_run(run_id="run-a", query="q", status="running")

    with store._connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1

    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")


def test_prune_runs_keeps_current_and_newest_and_purges_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_LEDGER_RETENTION_MAX_RUNS", "2")
    monkeypatch.setenv("ARTIFACT_LEDGER_RETENTION_MAX_AGE_DAYS", "0")
    monkeypatch.setenv("ARTIFACT_LEDGER_VACUUM_ON_PRUNE", "0")
    store = _store(tmp_path)

    # created_at oldest -> newest so ordering is deterministic
    for index, run_id in enumerate(["r_old", "r_mid", "r_new", "r_current"]):
        store.upsert_run(run_id=run_id, query="q", status="running")
        store.upsert_fact_card(run_id=run_id, fact_id="EV-1", source_id="S1", fact="fact")
        stamp = f"2026-06-01T0{index}:00:00Z"
        with store._connect() as conn:
            conn.execute("UPDATE runs SET created_at=? WHERE run_id=?", (stamp, run_id))
            conn.commit()

    result = store.prune_runs(keep_run_id="r_current")

    remaining = {row["run_id"] for row in store.list_run_ids()}
    assert result["status"] == "pruned"
    assert result["deleted_count"] == 1
    # current always kept (not counted against the cap) + newest 2 others
    assert {"r_current", "r_new", "r_mid"} == remaining
    assert "r_old" not in remaining
    # rows for the purged run are actually gone; current run rows remain
    assert store.list_fact_cards("r_old") == []
    assert store.list_fact_cards("r_current")


def test_record_artifact_pointer_mode_skips_payload_serialization(tmp_path):
    store = _store(tmp_path)
    store.upsert_run(run_id="run-ptr", query="q", status="running")

    # Pointer artifact: caller supplies storage_uri + size + hash. The recorded
    # size/hash must be exactly what was passed (NOT recomputed from payload) —
    # proof that the (here deliberately mismatched-size) payload was not
    # re-serialized to measure it.
    result = store.record_artifact(
        run_id="run-ptr",
        stage="writer_report",
        artifact_type="writer_report",
        payload={"body": "x" * 100},
        storage_uri=str(tmp_path / "snap" / "writer_report" / "payload.json"),
        storage_bytes=987654,
        content_hash="deadbeefcafe",
    )

    assert result.payload_inline is False
    assert result.bytes == 987654
    assert result.output_hash == "deadbeefcafe"
    row = store.get_artifact(result.artifact_id)
    assert row["storage_bytes"] == 987654
    assert row["payload_json"] in ("", None)
    assert row["storage_uri"].endswith("payload.json")


def test_prune_runs_disabled_when_no_limits(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_LEDGER_RETENTION_MAX_RUNS", "0")
    monkeypatch.setenv("ARTIFACT_LEDGER_RETENTION_MAX_AGE_DAYS", "0")
    store = _store(tmp_path)
    store.upsert_run(run_id="r1", query="q", status="running")

    result = store.prune_runs(keep_run_id="r_current")

    assert result["status"] == "disabled"
    assert {row["run_id"] for row in store.list_run_ids()} == {"r1"}
