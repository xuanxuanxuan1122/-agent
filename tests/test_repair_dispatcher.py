from __future__ import annotations

from rag_pipeline.contracts.repair_dispatcher import dispatch_repair_seed, rejected_span_repair_summary


def test_dispatcher_routes_metric_missing_period_to_report_pdf_search():
    seed = dispatch_repair_seed(
        {
            "gap_id": "GAP-metric",
            "requirement_id": "H1_metric",
            "gap_type": "metric_scope_period_unit_incomplete",
            "proof_role": "metric",
            "required_fields": ["metric", "value", "unit", "period", "source"],
            "query": "AI Agent adoption",
        },
        failed_queries=["AI Agent market size"],
    )

    assert seed["required_field_focus"] == "period"
    assert seed["repair_route"] == "metric_source_search"
    assert seed["source_strategy"]["source_priority"][:2] == ["official_data", "market_research"]
    assert {"report", "survey", "pdf", "annual report"} <= set(seed["query_enhancers"])
    assert seed["avoid_queries"] == ["AI Agent market size"]
    assert "AI Agent market size" not in seed["query"]
    assert {"研报", "调研", "统计", "测算口径"} & set(seed["query_enhancers"])
    assert any(term in seed["query"] for term in ["研报", "调研", "统计", "测算口径"])


def test_dispatcher_routes_counter_gap_to_negative_evidence():
    seed = dispatch_repair_seed(
        {
            "gap_id": "GAP-counter",
            "requirement_id": "H1_counter",
            "gap_type": "counter_evidence_missing",
            "proof_role": "counter",
            "query": "AI Agent deployment",
        }
    )

    assert seed["repair_route"] == "counter_evidence_search"
    assert seed["required_field_focus"] == "counter_signal"
    assert seed["source_strategy"]["source_priority"][0] == "counter_evidence"
    assert {"failure", "cost", "ROI unclear", "compliance"} <= set(seed["query_enhancers"])
    assert {"失败案例", "成本过高", "ROI不明", "合规"} & set(seed["query_enhancers"])
    assert any(term in seed["query"] for term in ["失败案例", "成本过高", "ROI不明", "合规"])


def test_dispatcher_routes_case_gap_with_chinese_case_enhancers():
    seed = dispatch_repair_seed(
        {
            "gap_id": "GAP-case",
            "requirement_id": "H1_case",
            "gap_type": "customer_case_missing",
            "proof_role": "case",
            "query": "AI Agent enterprise workflow",
        }
    )

    assert seed["repair_route"] == "case_source_search"
    assert {"客户案例", "落地案例", "采购", "中标"} & set(seed["query_enhancers"])
    assert any(term in seed["query"] for term in ["客户案例", "落地案例", "采购", "中标"])


def test_rejected_span_summary_turns_missing_fields_into_repair_seed():
    summary = rejected_span_repair_summary(
        [
            {"reason": "metric_missing_scope_or_period", "missing_fields": ["period", "unit"]},
            {"reason": "metric_missing_scope_or_period", "missing_fields": ["period"]},
            {"reason": "navigation_or_low_quality_text"},
        ],
        search_task={
            "task_id": "ST-H1",
            "requirement_id": "H1_metric",
            "gap_id": "GAP-metric",
            "proof_role": "metric",
            "query": "AI Agent adoption",
            "required_fields": ["metric", "value", "unit", "period", "source"],
        },
    )

    assert summary["status"] == "needs_repair"
    assert summary["reject_reason_counts"]["metric_missing_scope_or_period"] == 2
    assert summary["missing_field_counts"]["period"] == 2
    assert summary["repair_task_seed"]["gap_id"] == "GAP-metric"
    assert summary["repair_task_seed"]["required_field_focus"] == "period"
    assert "raw_page" not in str(summary)
