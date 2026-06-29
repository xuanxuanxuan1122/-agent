import json

from rag_pipeline.observability.dataflow_inspector import build_dataflow_report
from rag_pipeline.observability.stage_contracts import validate_stage_packets
from rag_pipeline.observability.stage_probe import build_stage_probe_packets, write_stage_probe_from_package


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
                "query_rewrite_diagnostics": {
                    "query_rewrite_call_count": 2,
                    "query_rewrite_cache_hit_count": 1,
                    "query_rewrite_budget_exhausted": False,
                },
                "repair_task_selection_summary": {
                    "task_count": 3,
                    "post_policy_task_count": 2,
                    "by_proof_role": {"metric": 1, "counter": 2},
                    "deep_budget_exhausted_count": 1,
                },
            },
            "fact_extractor": {
                "attempted": 3,
                "fact_card_count": 5,
                "rejected_span_count": 2,
                "invalid_metric_count": 1,
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
                {"raw_id": "R3"},
                {"raw_id": "R4"},
            ],
            "clean_evidence_list": [
                {
                    "evidence_id": "EV-1",
                    "requirement_id": "REQ-1",
                    "search_task_id": "T-1",
                    "source_id": "SRC-1",
                    "source": {"url": "https://example.org/source-1"},
                    "source_level": "B",
                },
                {
                    "evidence_id": "EV-2",
                    "requirement_id": "REQ-1",
                    "search_task_id": "T-1",
                    "source_id": "SRC-1",
                    "source": {"url": "https://example.org/source-1"},
                    "source_level": "B",
                },
            ],
            "analysis_ready_evidence": [
                {
                    "evidence_id": "EV-1",
                    "fact_id": "EV-1",
                    "requirement_id": "REQ-1",
                    "search_task_id": "T-1",
                    "source_id": "SRC-1",
                    "source": {"url": "https://example.org/source-1"},
                    "source_level": "B",
                }
            ],
            "source_registry": [
                {
                    "source_id": "SRC-1",
                    "title": "Specific AI Agent Market Report",
                    "url": "https://example.org/source-1",
                },
                {
                    "source_id": "SRC-2",
                    "title": "Specific AI Agent Market Report",
                    "url": "https://mirror.example.net/source-2",
                },
                {
                    "source_id": "SRC-3",
                    "title": "Specific AI Agent Market Report",
                    "url": "https://another.example.com/source-3",
                },
            ],
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
                {"claim_id": "CL-2", "fact_ids": [], "source_ids": []},
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
                "body_rewrite_global": {
                    "submitted_count": 2,
                    "success_count": 1,
                    "cache_hit_count": 1,
                    "fallback_count": 1,
                },
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
        "render_artifacts": {
            "chapter_narrative": {
                "attempted_count": 1,
                "success_count": 1,
                "fallback_count": 0,
                "cache_hit_count": 0,
            }
        },
        "table_packages": [
            {"table_id": "TB-1", "should_render": True, "rows": [{"cells": ["metric", "value"]}]},
            {"table_id": "TB-2", "should_render": False, "reject_reasons": ["no_complete_metric"]},
        ],
        "public_narrative_leak_audit": {
            "public_narrative_leak_input_count": 3,
            "public_narrative_leak_removed_count": 1,
            "public_narrative_leak_remaining_count": 0,
            "skipped_global_block_count": 1,
            "public_narrative_leak_reason_counts": {"internal_diagnostic": 1},
        },
        "qa_result": {
            "passed": False,
            "quality_score": 68,
            "errors": [{"code": "thin_body"}],
            "warnings": [{"code": "single_source"}],
        },
        "final_citation_audit": {
            "final_citation_reconciliation_status": "ok",
            "final_body_citation_refs": ["[1]", "[2]"],
            "final_appendix_refs": ["[1]"],
            "final_missing_appendix_refs": ["[2]"],
            "final_unresolved_citation_removed_count": 1,
            "factual_body_without_citations_count": 0,
        },
        "source_claim_support": {
            "source_gate_mode": "balanced",
            "checked_source_count": 3,
            "supported_source_count": 2,
            "empty_chapter_omitted_after_source_gate_count": 1,
        },
        "reformatter_result": {
            "status": "passed",
            "attempted": 1,
            "passed": True,
            "repair_attempt_count": 1,
        },
        "artifact_ledger": {
            "enabled": True,
            "status": "completed",
            "artifact_count": 8,
        },
        "stage_snapshot_index": [
            {"stage_name": "evidence_package", "replayable": True},
            {"stage_name": "writer_report", "replayable": False},
        ],
        "score_gaps": [
            {"gap_id": "G-1", "status": "still_insufficient", "requirement_id": "REQ-1"},
            {"gap_id": "G-2", "status": "evidence_found", "requirement_id": "REQ-2"},
        ],
        "writer_report": {
            "quality_score": 68,
            "report_markdown": "# Report\n\n## 正文\n\n### 机会\n\n这是第一段内容。[1]\n\n### 风险\n\n这是第二段内容。[2]\n\n## Sources\n\n[1] Example",
            "target_body_chars": 20000,
            "final_audit_result": {"status": "warning"},
        },
    }


def test_stage_probe_packets_capture_data_transfer_and_are_diagnostic_only():
    packets = build_stage_probe_packets(run_id="run-probe", writer_package=_writer_package())
    by_stage = {packet["stage"]: packet for packet in packets}

    assert {"search_plan", "evidence_merge", "llm_analysis", "claim_builder", "section_builder", "writer"}.issubset(by_stage)
    assert by_stage["evidence_merge"]["input_count"] == 4
    assert by_stage["evidence_merge"]["output_count"] == 1
    assert by_stage["evidence_merge"]["diagnostics"]["source_identity_hygiene"]["dirty_item_count"] == 3
    assert by_stage["claim_builder"]["input_count"] == 2
    assert by_stage["claim_builder"]["output_count"] == 1
    assert by_stage["section_builder"]["input_count"] == 2
    assert by_stage["section_builder"]["output_count"] == 1
    assert by_stage["claim_builder"]["id_coverage"]["source_ids"] == 0.5
    assert by_stage["section_builder"]["id_coverage"]["used_fact_refs"] == 0.5
    assert by_stage["writer"]["diagnostics"]["final_body_char_count"] > 0
    assert by_stage["writer"]["diagnostics"]["target_body_chars"] == 20000
    assert all(packet["diagnostic_only"] is True for packet in packets)
    assert all(packet["must_not_render"] is True for packet in packets)

    validation = validate_stage_packets(packets)
    assert validation["ok"] is True
    assert validation["packet_count"] == len(packets)


def test_stage_probe_tracks_analysis_ready_lineage_and_role_coverage():
    package = {
        "evidence_package": {
            "raw_data_points": [{"raw_id": "R1"}, {"raw_id": "R2"}],
            "analysis_ready_evidence": [
                {
                    "evidence_id": "EV-1",
                    "lineage": {
                        "chapter_id": "ch_01",
                        "requirement_id": "REQ-1",
                        "search_task_id": "T-1",
                        "source_id": "SRC-1",
                        "proof_role": "case",
                        "analysis_role": "case",
                    },
                },
                {"evidence_id": "EV-2"},
            ],
        }
    }

    packets = build_stage_probe_packets(run_id="run-probe", writer_package=package)
    evidence_merge = {packet["stage"]: packet for packet in packets}["evidence_merge"]

    assert evidence_merge["id_coverage"]["chapter_id"] == 0.5
    assert evidence_merge["id_coverage"]["proof_role"] == 0.5
    assert evidence_merge["id_coverage"]["analysis_role"] == 0.5
    assert evidence_merge["diagnostics"]["id_coverage_details"]["chapter_id"]["missing_count"] == 1


def test_stage_probe_emits_blueprint_evidence_alignment_packet():
    package = {
        "report_blueprint": {
            "chapters": [
                {"chapter_id": "ch_01", "chapter_title": "\u5e02\u573a\u9700\u6c42"},
                {"chapter_id": "ch_02", "chapter_title": "\u7ade\u4e89\u683c\u5c40"},
            ]
        },
        "evidence_package": {
            "analysis_ready_facts": [
                {
                    "fact_id": "F1",
                    "chapter_id": "ch_01",
                    "proof_role": "metric",
                    "title": "\u5e02\u573a\u9700\u6c42\u589e\u957f",
                }
            ]
        },
    }

    packets = build_stage_probe_packets(run_id="run-probe", writer_package=package)
    by_stage = {packet["stage"]: packet for packet in packets}

    assert "blueprint_evidence_alignment" in by_stage
    alignment = by_stage["blueprint_evidence_alignment"]
    assert alignment["status"] == "warning"
    assert alignment["diagnostics"]["chapter_starved_count"] == 1
    assert alignment["diagnostics"]["chapters"]["ch_02"]["status"] == "starved"


def test_stage_probe_blueprint_alignment_reads_top_level_chapter_packages():
    package = {
        "report_blueprint": {
            "chapters": [
                {"chapter_id": "ch_01", "chapter_title": "\u5e02\u573a\u9700\u6c42"},
            ]
        },
        "evidence_package": {},
        "chapter_evidence_packages": [
            {
                "chapter_id": "ch_01",
                "chapter_title": "\u5e02\u573a\u9700\u6c42",
                "analysis_ready_facts": [
                    {
                        "fact_id": "F1",
                        "chapter_id": "ch_01",
                        "proof_role": "metric",
                        "fact": "\u5e02\u573a\u9700\u6c42\u6709\u660e\u786e\u589e\u957f\u4fe1\u53f7\u3002",
                    }
                ],
            }
        ],
    }

    packets = build_stage_probe_packets(run_id="run-probe", writer_package=package)
    by_stage = {packet["stage"]: packet for packet in packets}

    alignment = by_stage["blueprint_evidence_alignment"]
    assert alignment["status"] == "ok"
    assert alignment["diagnostics"]["chapter_starved_count"] == 0
    assert alignment["diagnostics"]["chapters"]["ch_01"]["analysis_ready_count"] == 1


def test_stage_probe_emits_evidence_claim_conversion_packet():
    package = {
        "evidence_package": {
            "analysis_ready_evidence": [
                {"evidence_id": "EV-1", "chapter_id": "ch_01", "requirement_id": "REQ-1", "source_id": "SRC-1"},
            ]
        },
        "structured_analysis": {
            "claim_units": [
                {"claim_id": "CL-1", "chapter_id": "ch_01", "fact_ids": ["EV-1"], "source_ids": ["SRC-1"]},
                {"claim_id": "CL-2", "chapter_id": "ch_01", "fact_ids": [], "source_ids": []},
            ]
        },
        "chapter_packages": [
            {
                "chapter_id": "ch_01",
                "sections": [
                    {"section_id": "SEC-1", "claim_id": "CL-1", "used_fact_refs": ["EV-1"]},
                    {"section_id": "SEC-2", "claim_id": "CL-2", "used_fact_refs": []},
                ],
            }
        ],
    }

    packets = build_stage_probe_packets(run_id="run-probe", writer_package=package)
    by_stage = {packet["stage"]: packet for packet in packets}

    assert "evidence_claim_conversion" in by_stage
    packet = by_stage["evidence_claim_conversion"]
    assert packet["status"] == "warning"
    assert packet["diagnostics"]["totals"]["bound_claims"] == 1
    assert packet["diagnostics"]["loss_reason_counts"]["claim_unbound"] == 1


def test_stage_probe_surfaces_analysis_shard_cache_summary():
    package = {
        "evidence_package": {
            "analysis_shards": [
                {"cluster_key": "market", "curated_evidence_ids": ["EV-1"], "input_hash": "hash-market"},
            ],
        },
        "structured_analysis": {
            "analysis_stage_diagnostics": {
                "llm_chapter_results": [
                    {
                        "chapter_id": "ch_market",
                        "analysis_shard_output_cache": {
                            "status": "hit",
                            "input_hash": "hash-market",
                            "output_cache_path": "cache/market.json",
                        },
                    }
                ]
            },
            "claim_units": [
                {"claim_id": "CL-1", "chapter_id": "ch_market", "fact_ids": ["EV-1"], "source_ids": ["SRC-1"]},
            ],
        },
    }

    packets = build_stage_probe_packets(run_id="run-probe", writer_package=package)
    packet = {item["stage"]: item for item in packets}["evidence_claim_conversion"]

    assert packet["cache"]["analysis_shard_cache_hit_count"] == 1
    assert packet["cache"]["analysis_shard_cache_saved_llm_call_count"] == 1


def test_dataflow_report_flags_first_major_drop_and_lineage_gaps():
    packets = build_stage_probe_packets(run_id="run-probe", writer_package=_writer_package())

    report = build_dataflow_report(packets)

    assert report["run_id"] == "run-probe"
    assert report["funnel"]["evidence_merge"]["input_count"] == 4
    assert report["funnel"]["claim_builder"]["drop_count"] == 1
    assert report["lineage_gaps"]["claim_builder"]["source_ids"]["missing_count"] == 1
    assert any(item["stage"] == "evidence_merge" for item in report["bottlenecks"])


def test_dataflow_report_and_validator_accept_nested_module_probe_packets():
    packets = [
        {
            "schema_version": "module_probe_event_v1",
            "run_id": "run-module",
            "stage": "evidence_merge",
            "status": "warning",
            "event_type": "transform_result",
            "input": {"count": 10},
            "output": {"count": 7, "id_coverage": {"requirement_id": 0.75}},
            "drop": {"count": 3},
            "diagnostics": {
                "id_coverage_details": {
                    "requirement_id": {"missing_count": 2, "total_count": 7},
                }
            },
            "diagnostic_only": True,
            "must_not_render": True,
            "public_text_allowed": False,
        }
    ]

    validation = validate_stage_packets(packets)
    report = build_dataflow_report(packets)

    assert validation["ok"] is True
    assert report["funnel"]["evidence_merge"]["input_count"] == 10
    assert report["funnel"]["evidence_merge"]["output_count"] == 7
    assert report["funnel"]["evidence_merge"]["drop_count"] == 3
    assert report["lineage_gaps"]["evidence_merge"]["requirement_id"]["missing_count"] == 2


def test_dataflow_report_uses_transform_events_not_input_probe_events():
    packets = [
        {
            "schema_version": "module_probe_event_v1",
            "run_id": "run-module",
            "stage": "evidence_merge",
            "status": "ok",
            "event_type": "input_received",
            "input": {"count": 10},
            "output": {"count": 0},
            "drop": {"count": 0},
            "diagnostic_only": True,
            "must_not_render": True,
            "public_text_allowed": False,
        },
        {
            "schema_version": "module_probe_event_v1",
            "run_id": "run-module",
            "stage": "evidence_merge",
            "status": "warning",
            "event_type": "transform_result",
            "input": {"count": 10},
            "output": {"count": 7, "id_coverage": {"requirement_id": 0.8}},
            "drop": {"count": 3},
            "diagnostic_only": True,
            "must_not_render": True,
            "public_text_allowed": False,
        },
    ]

    validation = validate_stage_packets(packets)
    report = build_dataflow_report(packets)

    assert validation["warnings"] == []
    assert report["stage_count"] == 1
    assert report["funnel"]["evidence_merge"]["input_count"] == 10
    assert report["funnel"]["evidence_merge"]["output_count"] == 7
    assert len(report["bottlenecks"]) == 1


def test_write_stage_probe_from_package_writes_jsonl_and_summary(tmp_path):
    result = write_stage_probe_from_package(
        run_id="run-probe",
        output_dir=tmp_path,
        writer_package=_writer_package(),
    )

    assert result["enabled"] is True
    jsonl = tmp_path / "run-probe.stage_probe.jsonl"
    summary = tmp_path / "run-probe.dataflow_summary.md"
    module_probe = tmp_path / "run-probe.module_probe.jsonl"
    lineage_graph = tmp_path / "run-probe.lineage_graph.json"
    health_metrics = tmp_path / "run-probe.health_metrics.json"
    assert jsonl.exists()
    assert summary.exists()
    assert module_probe.exists()
    assert lineage_graph.exists()
    assert health_metrics.exists()
    assert result["module_probe_path"].endswith("run-probe.module_probe.jsonl")
    assert result["lineage_graph_path"].endswith("run-probe.lineage_graph.json")
    assert result["health_metrics_path"].endswith("run-probe.health_metrics.json")
    lines = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()]
    assert lines[0]["run_id"] == "run-probe"
    assert "## Stage Funnel" in summary.read_text(encoding="utf-8")


def test_stage_probe_covers_function_module_subflows_and_cache_hits():
    packets = build_stage_probe_packets(run_id="run-probe", writer_package=_writer_package())
    by_stage = {packet["stage"]: packet for packet in packets}

    assert {
        "query_rewrite",
        "evidence_repair",
        "fact_extractor",
        "body_rewrite",
        "chapter_narrative",
        "public_narrative_gate",
        "reformatter",
        "cache_summary",
    }.issubset(by_stage)
    assert by_stage["query_rewrite"]["input_count"] == 2
    assert by_stage["query_rewrite"]["cache"]["cache_hit_count"] == 1
    assert by_stage["evidence_repair"]["input_count"] == 3
    assert by_stage["evidence_repair"]["output_count"] == 2
    assert by_stage["fact_extractor"]["input_count"] == 3
    assert by_stage["fact_extractor"]["output_count"] == 5
    assert by_stage["fact_extractor"]["drop_count"] == 2
    assert by_stage["body_rewrite"]["input_count"] == 2
    assert by_stage["body_rewrite"]["output_count"] == 2
    assert by_stage["public_narrative_gate"]["drop_count"] == 2
    assert by_stage["cache_summary"]["diagnostics"]["total_cache_hit_count"] >= 2


def test_stage_probe_covers_sidecar_quality_and_cache_flow():
    packets = build_stage_probe_packets(run_id="run-probe", writer_package=_writer_package())
    by_stage = {packet["stage"]: packet for packet in packets}

    assert {
        "table_builder",
        "qa_score",
        "final_citation_audit",
        "source_gate",
        "artifact_ledger",
        "stage_snapshot",
        "score_gap_ledger",
    }.issubset(by_stage)
    assert by_stage["table_builder"]["input_count"] == 2
    assert by_stage["table_builder"]["output_count"] == 1
    assert by_stage["qa_score"]["status"] == "warning"
    assert by_stage["qa_score"]["diagnostics"]["quality_score"] == 68
    assert by_stage["final_citation_audit"]["drop_count"] == 2
    assert by_stage["source_gate"]["input_count"] == 3
    assert by_stage["source_gate"]["output_count"] == 2
    assert by_stage["stage_snapshot"]["output_count"] == 1
    assert by_stage["score_gap_ledger"]["input_count"] == 2
    assert by_stage["score_gap_ledger"]["reason_counts"]["still_insufficient"] == 1
