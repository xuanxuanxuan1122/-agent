from __future__ import annotations

from rag_pipeline.observability.health_metrics import build_health_metrics
from rag_pipeline.observability.stage_probe import section_binding_diagnostics


def test_health_metrics_uses_artifact_ledger_fact_count_when_extractor_missing():
    package = {
        "artifact_ledger": {"fact_card_count": 428},
        "evidence_package": {
            "analysis_ready_evidence": [{"fact_id": "EV-1"}],
        },
    }

    metrics = build_health_metrics(run_id="run-dataflow", writer_package=package)

    assert metrics["counts"]["fact_card_count"] == 428


def test_section_binding_reports_missing_examples():
    sections = [
        {"section_id": "s1", "used_fact_refs": ["EV-1"]},
        {"section_id": "s2", "used_fact_refs": []},
    ]

    diag = section_binding_diagnostics(sections)

    assert diag["section_count"] == 2
    assert diag["evidence_backed_section_count"] == 1
    assert diag["section_binding_rate"] == 0.5
    assert diag["missing_used_fact_refs"] == 1
    assert diag["missing_examples"] == ["s2"]


def test_health_metrics_surfaces_dirty_evidence_top_reasons():
    package = {
        "evidence_package": {
            "raw_data_points": [{"raw_id": "R1"}],
            "analysis_ready_evidence": [{"fact_id": "EV-1"}],
        },
    }
    stage_packets = [
        {
            "stage": "evidence_merge",
            "input_count": 1,
            "output_count": 1,
            "reason_counts": {"generic_metric_name": 3, "browser_or_login_fragment": 2},
        }
    ]

    metrics = build_health_metrics(
        run_id="run-dataflow",
        writer_package=package,
        stage_packets=stage_packets,
    )

    assert metrics["dirty_evidence_top_reasons"] == [
        ["generic_metric_name", 3],
        ["browser_or_login_fragment", 2],
    ]


def test_health_metrics_splits_claim_conversion_rates():
    stage_packets = [
        {"stage": "evidence_merge", "input_count": 120, "output_count": 100},
        {
            "stage": "llm_analysis",
            "input_count": 100,
            "output_count": 18,
            "diagnostics": {"llm_raw_claim_count": 24, "llm_usable_claim_count": 18},
        },
        {"stage": "claim_builder", "input_count": 24, "output_count": 18},
    ]

    metrics = build_health_metrics(
        run_id="run-dataflow",
        writer_package={},
        stage_packets=stage_packets,
    )

    assert metrics["rates"]["claim_conversion_rate"] == 0.18
    assert metrics["rates"]["fact_to_claim_rate"] == 0.18
    assert metrics["rates"]["llm_claim_acceptance_rate"] == 0.75
    assert metrics["rates"]["claim_binding_rate"] == 0.75
    assert metrics["counts"]["llm_raw_claim_count"] == 24
    assert metrics["counts"]["llm_usable_claim_count"] == 18
