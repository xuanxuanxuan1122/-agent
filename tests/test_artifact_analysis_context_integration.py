from __future__ import annotations

from rag_pipeline.agents.analysis_agent import build_llm_analysis_input_v2
from rag_pipeline.cache.artifact_store import ArtifactStore


def test_analysis_input_can_use_artifact_ledger_context_view(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_LEDGER_PATH", str(tmp_path / "artifact_ledger.sqlite"))
    monkeypatch.setenv("ARTIFACT_OBJECT_ROOT", str(tmp_path / "objects"))
    monkeypatch.setenv("REPORT_STAGE_SNAPSHOT_RUN_ID", "run-analysis")
    monkeypatch.setenv("ARTIFACT_LEDGER_ANALYSIS_CONTEXT_ENABLED", "true")
    store = ArtifactStore()
    store.upsert_run(run_id="run-analysis", query="AI Agent", status="running")
    store.upsert_evidence_requirement(
        run_id="run-analysis",
        requirement_id="H1_case",
        chapter_id="ch_01",
        proof_role="case",
        required_fields=["company", "use_case"],
        claim_strength_ceiling="directional",
        status="open",
    )
    store.upsert_source(
        run_id="run-analysis",
        run_source_id="SRC-1",
        source={"canonical_url": "https://example.com/case", "title": "Official case", "source_level": "A"},
    )
    store.upsert_fact_card(
        run_id="run-analysis",
        fact_id="EV-ledger",
        requirement_id="H1_case",
        source_id="SRC-1",
        fact="Company A disclosed an AI agent workflow deployment.",
        source_level="A",
        allowed_use="supporting",
        analysis_role="case",
        status="validated",
    )

    payload = build_llm_analysis_input_v2(
        {
            "query": "AI Agent",
            "report_contract": {
                "evidence_requirements": {
                    "requirements": [
                        {
                            "requirement_id": "H1_case",
                            "chapter_id": "ch_01",
                            "proof_role": "case",
                        }
                    ]
                }
            },
            "chapter_evidence_packages": [{"chapter_id": "ch_01", "chapter_title": "Demand validation"}],
        },
        {},
    )

    card = payload["chapters"][0]["fact_cards"][0]
    assert card["evidence_id"] == "EV-ledger"
    assert card["requirement_id"] == "H1_case"
    assert card["lineage"]["artifact_ledger_run_id"] == "run-analysis"
