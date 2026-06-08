from rag_pipeline.contracts.research_reflection import build_research_reflection_memo
from rag_pipeline.agents.analysis_agent import run_analysis_agent


def test_research_reflection_memo_turns_open_gaps_into_precise_search_seeds():
    memo = build_research_reflection_memo(
        {
            "evidence_health_summary": {
                "analysis_ready_count": 3,
                "traceable_ab_source_count": 1,
                "distinct_verified_ab_source_count": 1,
                "publishable_evidence_gate_passed": False,
            },
            "analysis_ready_evidence": [
                {
                    "evidence_id": "EV-case",
                    "requirement_id": "H1_case",
                    "chapter_id": "ch_01",
                    "source_id": "SRC-1",
                    "source_level": "B",
                    "proof_role": "case",
                    "allowed_use": "directional_signal",
                    "fact": "This text should not become a quoteable writer context.",
                }
            ],
            "coverage_matrix": [
                {
                    "requirement_id": "H1_metric",
                    "blocking_gaps": ["metric_scope_period_unit_incomplete"],
                    "current_evidence_refs": ["EV-old"],
                }
            ],
            "evidence_gap_ledger": [
                {
                    "gap_id": "GAP-metric",
                    "requirement_id": "H1_metric",
                    "chapter_id": "ch_01",
                    "gap_type": "metric_scope_period_unit_incomplete",
                    "severity": "high",
                    "status": "still_insufficient",
                    "missing": ["unit", "period"],
                    "retry_plan": {
                        "proof_role": "metric",
                        "required_fields": ["metric", "value", "unit", "period", "source"],
                        "lane_targets": ["official_data"],
                        "query_terms": ["enterprise AI agent adoption"],
                        "failed_queries": ["old generic query"],
                    },
                }
            ],
        }
    )

    assert memo["schema_version"] == "research_reflection_memo_v1"
    assert memo["status"] == "limited"
    assert memo["write_mode"] == "limited_review_draft"
    assert memo["enough_to_write"] is True
    assert memo["enough_for_publishable"] is False
    assert memo["allowed_for_writing"] is False
    assert memo["known_finding_refs"][0]["evidence_id"] == "EV-case"
    assert "fact" not in memo["known_finding_refs"][0]
    assert memo["coverage_by_requirement"][0]["requirement_id"] == "H1_metric"
    assert memo["coverage_by_requirement"][0]["missing_fields"] == ["unit", "period"]

    seed = memo["next_search_task_seeds"][0]
    assert seed["schema_version"] == "repair_task_seed_v2"
    assert seed["gap_id"] == "GAP-metric"
    assert seed["requirement_id"] == "H1_metric"
    assert seed["proof_role"] == "metric"
    assert seed["required_fields"] == ["unit", "period"]
    assert "official_data" in seed["preferred_source_patterns"]
    assert "no_date" in seed["reject_if"]
    assert seed["allowed_for_writing"] is False
    assert seed["avoid_repeating_failed_query"] is True
    assert "old generic query" not in seed["query"]


def test_research_reflection_memo_marks_publishable_when_gate_and_ab_sources_pass():
    memo = build_research_reflection_memo(
        {
            "evidence_health_summary": {
                "analysis_ready_count": 8,
                "traceable_ab_source_count": 3,
                "distinct_verified_ab_source_count": 2,
                "publishable_evidence_gate_passed": True,
            },
            "analysis_ready_evidence": [
                {"evidence_id": "EV-1", "requirement_id": "H1_metric", "source_level": "A"},
                {"evidence_id": "EV-2", "requirement_id": "H2_case", "source_level": "B"},
            ],
        }
    )

    assert memo["status"] == "sufficient"
    assert memo["write_mode"] == "publishable_draft"
    assert memo["enough_to_write"] is True
    assert memo["enough_for_publishable"] is True
    assert memo["next_search_task_seeds"] == []


def test_run_analysis_agent_publishes_research_reflection_memo(monkeypatch):
    monkeypatch.setenv("BRAIN_ENABLE_LLM_EVIDENCE_ANALYSIS", "false")

    result = run_analysis_agent(
        {
            "query": "AI Agent enterprise adoption",
            "evidence_health_summary": {
                "analysis_ready_count": 1,
                "traceable_ab_source_count": 1,
                "distinct_verified_ab_source_count": 1,
            },
            "analysis_ready_evidence": [
                {
                    "evidence_id": "EV-1",
                    "chapter_id": "ch_01",
                    "requirement_id": "H1_case",
                    "fact": "Enterprise AI agent deployments are moving into workflow automation.",
                    "source_level": "B",
                    "source": {"title": "Verified source", "url": "https://example.org/source"},
                }
            ],
            "evidence_gap_ledger": [
                {
                    "gap_id": "GAP-counter",
                    "requirement_id": "H2_counter",
                    "gap_type": "counter_boundary_missing",
                    "severity": "medium",
                    "retry_plan": {"proof_role": "counter", "query_terms": ["AI agent ROI unclear"]},
                }
            ],
        }
    )

    structured = result["structured_analysis"]
    memo = structured["research_reflection_memo"]

    assert memo["schema_version"] == "research_reflection_memo_v1"
    assert memo["status"] == "limited"
    assert memo["next_search_task_seeds"][0]["proof_role"] == "counter"
    assert structured["report_insight_package"]["research_reflection_memo"] == memo
    assert structured["analysis_stage_diagnostics"]["research_reflection_status"] == "limited"
    assert structured["analysis_stage_diagnostics"]["research_reflection_seed_count"] == 1
