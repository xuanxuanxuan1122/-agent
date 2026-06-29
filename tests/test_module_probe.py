import json

from rag_pipeline.observability.health_metrics import build_health_metrics
from rag_pipeline.observability.lineage_tracker import build_lineage_graph
from rag_pipeline.observability.module_probe_writer import build_module_probe_events, write_module_probe_from_package


def _writer_package():
    return {
        "raw_output": {
            "metadata": {
                "query_plan": [{"query": "AI Agent adoption"}],
                "search_tasks": [
                    {
                        "task_id": "T-1",
                        "requirement_id": "REQ-1",
                        "proof_role": "case",
                        "query": "AI Agent adoption case source",
                    }
                ],
                "auto_readpage": {"attempted": 2, "succeeded": 1, "failed": 1},
                "repair_task_selection_summary": {
                    "task_count": 2,
                    "post_policy_task_count": 1,
                    "by_proof_role": {"case": 2},
                },
            },
            "fact_extractor": {
                "attempted": 2,
                "fact_card_count": 3,
                "rejected_span_count": 1,
                "cache_hit_count": 1,
            },
            "quality_processing": {
                "raw_count": 4,
                "normalized_count": 3,
                "task_filtered_count": 1,
                "task_filter_reasons": {"low_task_relevance": 1},
            },
        },
        "evidence_package": {
            "raw_data_points": [
                {"raw_id": "R1", "requirement_id": "REQ-1", "search_task_id": "T-1"},
                {"raw_id": "R2", "requirement_id": "REQ-1", "search_task_id": "T-1"},
            ],
            "clean_evidence_list": [
                {
                    "evidence_id": "EV-1",
                    "fact_id": "EV-1",
                    "requirement_id": "REQ-1",
                    "search_task_id": "T-1",
                    "source_id": "SRC-1",
                    "source_level": "B",
                }
            ],
            "analysis_ready_evidence": [
                {
                    "evidence_id": "EV-1",
                    "fact_id": "EV-1",
                    "requirement_id": "REQ-1",
                    "search_task_id": "T-1",
                    "source_id": "SRC-1",
                    "source_level": "B",
                }
            ],
            "metadata": {
                "evidence_cache_store": {
                    "cache_hit_count": 2,
                    "cache_miss_count": 1,
                    "polluted_count": 1,
                    "quarantined_count": 1,
                    "hit_items": [{"cache_key": "bad", "status": "polluted"}],
                }
            },
        },
        "structured_analysis": {
            "claim_units": [
                {
                    "claim_id": "CL-1",
                    "requirement_ids": ["REQ-1"],
                    "fact_ids": ["EV-1"],
                    "source_ids": ["SRC-1"],
                    "evidence_refs": ["EV-1"],
                },
                {"claim_id": "CL-2", "requirement_ids": ["REQ-1"], "fact_ids": [], "source_ids": []},
            ],
            "analysis_stage_diagnostics": {
                "llm_usable_claim_count": 2,
                "llm_raw_claim_count": 3,
                "llm_validation_issue_counts": {"missing_source_ids": 1},
            },
        },
        "chapter_packages": [
            {
                "chapter_id": "ch_01",
                "sections": [
                    {
                        "section_id": "SEC-1",
                        "claim_id": "CL-1",
                        "requirement_ids": ["REQ-1"],
                        "used_fact_refs": ["EV-1"],
                        "evidence_refs": ["EV-1"],
                    },
                    {"section_id": "SEC-2", "claim_id": "CL-2", "used_fact_refs": []},
                ],
            }
        ],
        "final_citation_audit": {
            "final_citation_reconciliation_status": "ok",
            "final_body_citation_refs": ["[1]", "[2]"],
            "final_appendix_refs": ["[1]"],
            "final_missing_appendix_refs": ["[2]"],
        },
        "score_gaps": [
            {"gap_id": "G-1", "status": "still_insufficient", "requirement_id": "REQ-1"},
            {"gap_id": "G-2", "status": "evidence_found", "requirement_id": "REQ-1"},
        ],
        "writer_report": {"quality_score": 68},
    }


def test_module_probe_events_expand_stage_packets_without_public_text():
    events = build_module_probe_events(run_id="run-module", writer_package=_writer_package())
    event_types = {event["event_type"] for event in events}
    transform = next(event for event in events if event["stage"] == "fact_extractor" and event["event_type"] == "transform_result")

    assert {"input_received", "transform_result", "decision_observed"}.issubset(event_types)
    assert transform["input"]["count"] == 2
    assert transform["output"]["count"] == 3
    assert transform["drop"]["count"] == 1
    assert transform["cache"]["hit_count"] == 1
    assert all(event["diagnostic_only"] is True for event in events)
    assert all(event["must_not_render"] is True for event in events)
    assert all(event["public_text_allowed"] is False for event in events)


def test_cache_summary_probe_surfaces_root_hygiene():
    events = build_module_probe_events(run_id="run-module", writer_package=_writer_package())
    event = next(item for item in events if item["stage"] == "cache_summary" and item["event_type"] == "transform_result")

    hygiene = event["diagnostics"]["cache_root_hygiene"]
    assert hygiene["polluted_count"] == 1
    assert hygiene["quarantined_count"] == 1
    assert hygiene["dirty_hit_count"] == 1
    assert event["drop"]["reason_counts"]["cache_polluted_count"] == 1


def test_lineage_graph_tracks_requirement_to_section_and_gap():
    graph = build_lineage_graph(run_id="run-module", writer_package=_writer_package())
    edge_keys = {(edge["from"], edge["to"], edge["relation"]) for edge in graph["edges"]}

    assert ("REQ-1", "T-1", "plans_search") in edge_keys
    assert ("T-1", "EV-1", "retrieves_fact") in edge_keys
    assert ("EV-1", "CL-1", "supports_claim") in edge_keys
    assert ("CL-1", "SEC-1", "renders_section") in edge_keys
    assert ("REQ-1", "G-1", "has_score_gap") in edge_keys
    assert graph["coverage"]["fact_to_claim_edge_count"] == 1
    assert graph["coverage"]["claim_to_section_edge_count"] == 2


def test_health_metrics_report_core_conversion_rates():
    metrics = build_health_metrics(run_id="run-module", writer_package=_writer_package())

    assert metrics["rates"]["search_result_accept_rate"] == 0.75
    assert metrics["rates"]["readpage_success_rate"] == 0.5
    assert metrics["rates"]["fact_extraction_rate"] == 1.5
    assert metrics["rates"]["bound_claim_rate"] == 0.5
    assert metrics["rates"]["section_binding_rate"] == 0.5
    assert metrics["rates"]["citation_binding_rate"] == 0.5
    assert metrics["rates"]["repair_success_rate"] == 0.5
    assert metrics["diagnostic_only"] is True
    assert metrics["must_not_render"] is True


def test_health_metrics_uses_readpage_coverage_when_auto_readpage_missing():
    package = _writer_package()
    package["raw_output"]["metadata"].pop("auto_readpage")
    package["evidence_package"]["metadata"]["readpage_coverage"] = {
        "attempted": 96,
        "succeeded": 86,
        "failed": 10,
    }

    metrics = build_health_metrics(run_id="run-module", writer_package=package)

    assert metrics["counts"]["readpage_attempted_count"] == 96
    assert metrics["counts"]["readpage_success_count"] == 86
    assert metrics["rates"]["readpage_success_rate"] == 0.8958


def test_write_module_probe_from_package_writes_sidecars(tmp_path):
    result = write_module_probe_from_package(run_id="run-module", output_dir=tmp_path, writer_package=_writer_package())

    assert result["enabled"] is True
    module_probe = tmp_path / "run-module.module_probe.jsonl"
    lineage = tmp_path / "run-module.lineage_graph.json"
    health = tmp_path / "run-module.health_metrics.json"
    assert module_probe.exists()
    assert lineage.exists()
    assert health.exists()
    first_event = json.loads(module_probe.read_text(encoding="utf-8").splitlines()[0])
    assert first_event["schema_version"] == "module_probe_event_v1"
    assert json.loads(lineage.read_text(encoding="utf-8"))["schema_version"] == "lineage_graph_v1"
    assert json.loads(health.read_text(encoding="utf-8"))["schema_version"] == "health_metrics_v1"
