"""P5: claim/section integrity floor. Unsupported claims must not be 'validated',
and missing requirement_ids (a known upstream id-granularity gap) only downgrades
to 'directional' rather than rejecting an otherwise evidence-backed claim."""
from __future__ import annotations

from rag_pipeline.cache.artifact_store import (
    ArtifactStore,
    _enforced_claim_status,
    _enforced_section_status,
)


def test_enforced_claim_status_floor():
    full = _enforced_claim_status("validated", facts=["EV-1"], sources=["SRC-1"], claim_text="x", requirement_ids=["R1"])
    assert full == "validated"
    # missing requirement only downgrades (id-granularity gap leaves it empty)
    assert _enforced_claim_status("validated", facts=["EV-1"], sources=["SRC-1"], claim_text="x", requirement_ids=[]) == "directional"
    # missing real backing rejects the 'validated' status
    assert _enforced_claim_status("validated", facts=[], sources=["SRC-1"], claim_text="x", requirement_ids=["R1"]) == "unsupported"
    assert _enforced_claim_status("validated", facts=["EV-1"], sources=[], claim_text="x", requirement_ids=["R1"]) == "unsupported"
    assert _enforced_claim_status("validated", facts=["EV-1"], sources=["SRC-1"], claim_text="   ", requirement_ids=["R1"]) == "unsupported"
    # a non-validated incoming status passes through untouched
    assert _enforced_claim_status("pending", facts=[], sources=[], claim_text="", requirement_ids=[]) == "pending"


def test_enforced_section_status_floor():
    assert _enforced_section_status("validated", claim_ids=["CL-1"], used_fact_refs=[]) == "validated"
    assert _enforced_section_status("validated", claim_ids=[], used_fact_refs=["EV-1"]) == "validated"
    assert _enforced_section_status("validated", claim_ids=[], used_fact_refs=[]) == "unsupported"


def test_upsert_enforces_integrity(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_LEDGER_PATH", str(tmp_path / "l.sqlite"))
    monkeypatch.setenv("ARTIFACT_OBJECT_ROOT", str(tmp_path / "o"))
    store = ArtifactStore()
    store.upsert_run(run_id="r", query="q", status="running")

    store.upsert_claim_unit(run_id="r", claim_id="CL-bad", payload={"claim": "text but no evidence"}, status="validated")
    store.upsert_claim_unit(
        run_id="r",
        claim_id="CL-directional",
        payload={"claim": "evidence-backed but no requirement"},
        fact_ids=["EV-1"],
        source_ids=["SRC-1"],
        status="validated",
    )
    store.upsert_claim_unit(
        run_id="r",
        claim_id="CL-good",
        payload={"claim": "x"},
        requirement_ids=["R1"],
        fact_ids=["EV-1"],
        source_ids=["SRC-1"],
        status="validated",
    )
    claims = {c["claim_id"]: c["status"] for c in store.list_claim_units("r")}
    assert claims["CL-bad"] == "unsupported"
    assert claims["CL-directional"] == "directional"
    assert claims["CL-good"] == "validated"

    store.upsert_section(run_id="r", section_id="SEC-empty", payload={}, status="validated")
    store.upsert_section(run_id="r", section_id="SEC-ok", payload={}, claim_ids=["CL-good"], status="validated")
    assert store.get_section("r", "SEC-empty")["status"] == "unsupported"
    assert store.get_section("r", "SEC-ok")["status"] == "validated"
