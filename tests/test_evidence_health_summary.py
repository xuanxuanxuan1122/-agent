from rag_pipeline.agents.analysis_agent import _select_analysis_items_for_dimension
from rag_pipeline.agents.evidence_merger import _evidence_health_inconsistencies, merge_evidence_package
from rag_pipeline.agents.qa_agent import run_qa_agent
from rag_pipeline.agents.writer_agent_clean import _delivery_gate_from_evidence_package


def _package_for_source(source_payload):
    return merge_evidence_package(
        original_query="AI Agent industry report",
        evidence_pool=[
            {
                "status": "success",
                "confidence": 0.9,
                "key_sources": [source_payload],
                "raw_data_points": [
                    {
                        "chapter_id": "ch_1",
                        "dimension": "industry overview",
                        "metric": "adoption",
                        "value": "50%",
                        "period": "2025",
                        "source_title": source_payload.get("title"),
                        "source_url": source_payload.get("url", ""),
                        "source_type": source_payload.get("source_type", "official"),
                        "evidence": "Official statistics show AI agent adoption continued to rise in 2025.",
                    }
                ],
                "page_results": [{"url": source_payload.get("url", ""), "content": "Official report body", "auto_readpage": True}],
                "metadata": {"auto_readpage": {"attempted": 1, "succeeded": 1}},
            }
        ],
        children={},
        research_plan={"chapter_structure": [{"chapter_id": "ch_1", "chapter_title": "Industry overview"}]},
    )


def test_evidence_health_summary_counts_traceable_sources():
    package = _package_for_source(
        {
            "title": "Official AI Agent Statistics",
            "url": "https://www.stats.gov.cn/ai-agent-statistics",
            "source_type": "official",
        }
    )

    health = package["evidence_health_summary"]

    assert health["raw_data_point_count"] >= 1
    assert health["normalized_evidence_count"] >= 1
    assert health["analysis_ready_count"] >= 1
    assert health["analysis_ready_ab_count"] >= 1
    assert health["source_registry_count"] >= 1
    assert health["traceable_source_count"] >= 1
    assert health["traceable_ab_source_count"] >= 1
    assert health["inconsistent"] is False


def test_title_only_source_does_not_count_as_traceable_ab():
    package = _package_for_source(
        {
            "title": "Title Only Official Source",
            "source_type": "official",
        }
    )

    health = package["evidence_health_summary"]

    assert health["source_registry_count"] >= 1
    assert health["title_only_source_count"] >= 1
    assert health["traceable_ab_source_count"] == 0
    assert all(item.get("source_level") == "D" for item in package["source_registry"])


def test_placeholder_source_does_not_count_as_ab_or_traceable():
    package = merge_evidence_package(
        original_query="AI Agent industry report",
        evidence_pool=[
            {
                "status": "success",
                "confidence": 0.9,
                "key_sources": [{"title": "Official", "url": "https://example.gov/ai-agent-statistics", "source_type": "official"}],
                "raw_data_points": [
                    {
                        "chapter_id": "ch_1",
                        "dimension": "industry overview",
                        "metric": "adoption",
                        "value": "50%",
                        "period": "2025",
                        "source_title": "Official",
                        "source_url": "https://example.gov/ai-agent-statistics",
                        "source_type": "official",
                        "evidence": "Official data shows AI agent adoption reached 50% in 2025.",
                    }
                ],
                "page_results": [{"url": "https://example.gov/ai-agent-statistics", "content": "Placeholder", "auto_readpage": True}],
                "metadata": {"auto_readpage": {"attempted": 1, "succeeded": 1}},
            }
        ],
        children={},
        research_plan={"chapter_structure": [{"chapter_id": "ch_1", "chapter_title": "Industry overview"}]},
    )

    health = package["evidence_health_summary"]
    assert health["fake_or_placeholder_source_count"] >= 1
    assert health["traceable_ab_source_count"] == 0
    assert all(item.get("source_level") != "A" for item in package["source_registry"])


def test_evidence_health_inconsistency_blocks_writer_and_qa():
    health = {
        "analysis_ready_count": 2,
        "analysis_ready_ab_count": 1,
        "source_registry_count": 0,
        "raw_data_point_count": 2,
        "normalized_evidence_count": 2,
    }
    health["inconsistencies"] = _evidence_health_inconsistencies(health)
    health["inconsistent"] = bool(health["inconsistencies"])

    gate = _delivery_gate_from_evidence_package({"evidence_health_summary": health, "summary": {}})
    qa = run_qa_agent(
        report_markdown="# Report\n\nBody " * 80,
        report_blueprint={"report_family": "industry_deep_report"},
        chapter_packages=[{"chapter_id": "ch_1", "sections": [{"claim": "x", "reasoning": "y", "counter_evidence": "z", "actionable": "a", "evidence_refs": ["EV-1"]}]}],
        risk_package={"risk_items": [{"risk": "x"}]},
        evidence_health_summary=health,
    )

    assert gate["diagnostic_only"] is True
    assert any(item["type"] == "evidence_health_summary_inconsistent" for item in qa["errors"])


def test_fallback_analysis_stratified_selection_keeps_counter_and_metric():
    directional = [
        {
            "evidence_id": f"EV-D{i}",
            "fact": f"Directional signal {i}",
            "source_level": "C",
            "allowed_use": "directional_signal",
            "confidence": 0.99,
            "source": {"title": f"Media {i}", "url": f"https://media.example.org/{i}"},
        }
        for i in range(30)
    ]
    metric_ready = {
        "evidence_id": "EV-METRIC",
        "fact": "Official metric shows adoption reached 50% in 2025.",
        "metric": "adoption",
        "value": "50%",
        "period": "2025",
        "source_level": "A",
        "allowed_use": "core_claim",
        "confidence": 0.7,
        "source": {"title": "Official metric", "url": "https://www.stats.gov.cn/metric", "date": "2025"},
    }
    counter = {
        "evidence_id": "EV-COUNTER",
        "fact": "A failed deployment case creates counter evidence.",
        "proof_role": "counter",
        "source_level": "B",
        "allowed_use": "supporting",
        "confidence": 0.6,
        "source": {"title": "Research counter", "url": "https://research.example.org/counter"},
    }

    selected = _select_analysis_items_for_dimension([*directional, metric_ready, counter], max_items=8)
    selected_ids = {item["evidence_id"] for item in selected}

    assert "EV-METRIC" in selected_ids
    assert "EV-COUNTER" in selected_ids


def test_fallback_analysis_stratified_selection_reserves_role_diversity():
    metrics = [
        {
            "evidence_id": f"EV-METRIC-{i}",
            "fact": f"Official metric {i} shows adoption reached {40 + i}% in 2025.",
            "proof_role": "metric",
            "metric": "adoption",
            "value": f"{40 + i}%",
            "period": "2025",
            "source_level": "A",
            "allowed_use": "core_claim",
            "confidence": 0.9 - i * 0.01,
            "source": {"title": f"Official metric {i}", "url": f"https://www.stats.gov.cn/metric/{i}"},
        }
        for i in range(8)
    ]
    role_items = [
        {
            "evidence_id": "EV-POLICY",
            "fact": "Official policy sets enterprise AI Agent governance boundaries.",
            "proof_role": "source_check",
            "source_level": "A",
            "allowed_use": "supporting",
            "confidence": 0.72,
            "source": {"title": "Policy", "url": "https://gov.example/policy"},
        },
        {
            "evidence_id": "EV-COUNTER",
            "fact": "Research records deployment failures and ROI uncertainty.",
            "proof_role": "counter",
            "source_level": "B",
            "allowed_use": "supporting",
            "confidence": 0.68,
            "source": {"title": "Research counter", "url": "https://research.example/counter"},
        },
        {
            "evidence_id": "EV-CASE",
            "fact": "A customer case shows workflow deployment in production.",
            "proof_role": "case",
            "source_level": "B",
            "allowed_use": "supporting",
            "confidence": 0.66,
            "source": {"title": "Customer case", "url": "https://company.example/case"},
        },
    ]

    selected = _select_analysis_items_for_dimension([*metrics, *role_items], max_items=6)
    selected_ids = {item["evidence_id"] for item in selected}

    assert {"EV-POLICY", "EV-COUNTER", "EV-CASE"}.issubset(selected_ids)
