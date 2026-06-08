from __future__ import annotations

import json
from pathlib import Path

from rag_pipeline.cache.artifact_store import ArtifactStore


def _store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(
        path=tmp_path / "artifact_ledger.sqlite",
        object_root=tmp_path / "objects",
        inline_max_bytes=96,
    )


def test_artifact_store_initializes_schema_wal_and_run_scoped_requirements(tmp_path):
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
