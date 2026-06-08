import inspect
from pathlib import Path

from rag_pipeline.agents.analytics.competitor_analysis_agent import _collect_rows
from rag_pipeline.agents.analytics.evidence_utils import evidence_subject, is_valid_entity_subject, source_ref
from rag_pipeline.agents.analytics.investor_insight_agent import _collect_rows as _collect_investor_rows
from rag_pipeline.agents.analytics.market_analytics_agent import _derive_cagr, run_market_analytics_agent
from rag_pipeline.agents.analysis_agent import run_analysis_agent
from rag_pipeline.agents.brain_agent import (
    _dedupe_followup_tasks,
    _followup_signal_diagnostics,
    _followup_query_key,
    _followup_result_has_signal,
    _gap_ledger_from_followups,
    _infer_lane_types_for_task,
    _layout_followup_queries_from_writer_report,
    _local_rag_enabled,
    _post_qa_repair_followup_payload,
    _post_qa_repair_needed,
    _post_qa_repair_plan,
    _post_qa_repair_query_from_item,
    _repair_quality_gain,
    _repair_result_summary,
    _repair_seen_keys_for_state,
    _repair_tasks_from_items,
    _route_agents,
    _select_high_value_repair_tasks,
    _run_post_qa_repair_round,
    _select_quality_first_initial_lanes,
    _lane_early_stop_decision,
    _lane_health_summary_from_coverage,
    _apply_lane_circuit_breaker_to_tasks,
    _search_options_for_task,
    _select_lane_tasks_for_budget,
    _substantive_followup_results,
    _topic_seed_terms,
    _writer_quality_key,
    build_initial_evidence_pool,
    build_loop_health_summary,
    build_search_tasks_for_goal,
    route_query,
    select_child_agents,
)
from rag_pipeline.agents import brain_agent as brain_agent_module
from rag_pipeline.agents import evidence_merger as evidence_merger_module
from rag_pipeline.agents import web_analysis_agent as web_analysis_agent_module
from rag_pipeline.agents.claim_builder_agent import run_claim_builder_agent
from rag_pipeline.agents.evidence_binder import run_evidence_binder
from rag_pipeline.agents.evidence_merger import merge_evidence_package
from rag_pipeline.agents.final_writer_agent import run_final_writer_agent
from rag_pipeline.agents.qa_agent import run_qa_agent
from rag_pipeline.agents.package_contracts import (
    package_issue_summary,
    validate_argument_units,
    validate_chapter_packages,
    validate_micro_layouts,
    validate_report_blueprint,
)
from rag_pipeline.agents.markdown_renderer import render_appendix, render_table_package
from rag_pipeline.agents.problem_framing_agent import run_problem_framing_agent
from rag_pipeline.agents.research_proof_registry import mandatory_proof_checks, select_research_proof_profile
from rag_pipeline.agents.table_agent import _row_for_item, _row_has_valid_leading_cell, _subject, run_table_agent
from rag_pipeline.agents.public_report_sanitizer import sanitize_public_markdown
from rag_pipeline.agents.writer_agent_clean import _normalize_public_packages_for_contract, _qa_has_pending_repair, _writer_ready_for_final
from rag_pipeline.contracts.evidence_quality import apply_evidence_quality_contract
from rag_pipeline.contracts.quality_gate import build_quality_gate_state
from rag_pipeline.contracts.query_builder import build_query_package
from rag_pipeline.contracts.report_contract import build_report_contract_from_package
from rag_pipeline.contracts.source_registry import pick_refs, renumber_sources_by_first_citation
from rag_pipeline.flows.report import full_report
from rag_pipeline.flows.report.evidence_extractor import _infer_credibility, _is_meaningful_fact, extract_clean_evidence_from_package
from rag_pipeline.flows.report.llm_review_agent import build_structured_review, split_markdown_for_review
from rag_pipeline.flows.report.review_pipeline import run_review_pipeline_sync
from rag_pipeline.flows.report.reformatter_agent import (
    _auto_expand_analysis_for_length,
    _citation_density_issues,
    _reformatter_needs_repair,
    _source_diversity_floor,
    _target_body_chars,
    build_reformatter_payload,
    build_reformatter_repair_plan,
    clean_reformatted_report,
    validate_reformatted_report,
)


LONG_AI_CHAPTER_TITLE = (
    "\u4e2d\u56fd\u4eba\u5de5\u667a\u80fd\u884c\u4e1a\u4ecd\u6709"
    "\u4ea7\u4e1a\u89c4\u6a21\u548c\u5e94\u7528\u6269\u5f20\u673a\u4f1a"
    "\uff0c\u4f46\u589e\u957f\u8d28\u91cf\u9700\u8981\u6309\u7ec6\u5206"
    "\u573a\u666f\u9a8c\u8bc1"
)


def test_entity_subject_rejects_dirty_titles_and_long_fallback():
    dirty = {"source": {"title": "LOOK ~~~~~~~~~~~~~~~~~~~~~"}}

    assert evidence_subject(dirty, fallback=LONG_AI_CHAPTER_TITLE) == ""
    assert is_valid_entity_subject("LOOK ~~~~~~~~~~~~~~~~~~~~~") is False
    assert evidence_subject({"entity": "\u4e2d\u56fd\u4fe1\u901a\u9662"}) == "\u4e2d\u56fd\u4fe1\u901a\u9662"


def test_multisector_us_china_policy_query_does_not_use_ev_material_template():
    query = "中美新定位下的产业分化：从关税、出口管制与市场准入看半导体、新能源、消费品与互联网的承压与受益"

    framing = run_problem_framing_agent(query=query)
    text = " ".join(
        [str(framing.get("core_question") or "")]
        + [str(item.get("statement") or "") for item in framing.get("hypotheses") or []]
    )

    assert "新能源汽车新型材料" not in text
    assert "半导体" in text
    assert "出口管制" in text or "关税" in text


def test_evidence_extractor_keeps_report_table_rows_and_dynamic_headings():
    pkg = {
        "query": "测试报告",
        "writer_report": {
            "source_registry": [
                {"ref": "[1]", "title": "来源一", "url": "https://example.com/1"},
                {"ref": "[2]", "title": "来源二", "url": "https://example.com/2"},
                {"ref": "[3]", "title": "来源三", "url": "https://example.com/3"},
            ],
            "report_markdown": """
# 测试报告

## 第一章 半导体出口管制影响
| 指标 | 事实 | 来源 |
| --- | --- | --- |
| 出口许可 | 先进芯片出口许可收紧，企业需要重新安排供应链[1] | [1] |
| 替代路径 | 成熟制程和封测环节出现订单转移，但仍受设备可得性约束[2] | [2] |

## 第二章 新能源关税与市场准入
新能源出口面对更高关税和本地化准入要求，海外产能与非美市场成为对冲变量[3]。

## 数据来源
[1] 来源一
""",
        },
    }

    clean = extract_clean_evidence_from_package(pkg)
    dimensions = clean["dimensions"]
    all_facts = [item for items in dimensions.values() for item in items]

    assert len(all_facts) >= 3
    assert any("第一章 半导体出口管制影响" in key for key, items in dimensions.items() if items)
    assert any("出口许可" in item["text"] for item in all_facts)
    assert any("新能源出口" in item["text"] for item in all_facts)
    assert clean["evidence_ledger"]
    assert clean["metadata"]["evidence_ledger_summary"]["ledger_entry_count"] >= len(all_facts)
    assert clean["metadata"]["evidence_ledger_summary"]["used_in_body_count"] >= 3


def _handoff_sources(count=300):
    return [
        {
            "id": index,
            "title": f"Research source {index}",
            "url": f"https://example.com/source-{index}",
            "date": "2026-01-01",
            "source_type": "research",
            "credibility": "B",
        }
        for index in range(1, count + 1)
    ]


def _handoff_fact(index, *, level="B"):
    return {
        "evidence_id": f"EV-{index:03d}",
        "source_id": str(index),
        "source_level": level,
        "source_type": "research",
        "dimension": "competition",
        "fact": (
            f"Vendor evidence item {index} reports a measurable 2026 market signal, "
            f"including adoption count {100 + index} and enterprise deployment context."
        ),
        "metric": "adoption count",
        "value": str(100 + index),
        "unit": "deployments",
        "period": "2026",
        "evidence_role": "supporting",
        "allowed_use": "supporting",
        "appendix_only": False,
        "source": {
            "id": index,
            "title": f"Research source {index}",
            "url": f"https://example.com/source-{index}",
            "date": "2026-01-01",
            "source_type": "research",
            "credibility": level,
        },
    }


def test_reformatter_evidence_package_priority_recovers_full_fact_set():
    sources = _handoff_sources(300)
    facts = [_handoff_fact(index) for index in range(1, 301)]
    pkg = {
        "query": "handoff test",
        "evidence_package": {
            "payload_mode": "summary",
            "summary": {"clean_fact_count": 300, "analysis_ready_count": 90},
            "normalized_evidence": {"count": 0, "sample": []},
        },
        "reformatter_evidence_package": {
            "payload_mode": "handoff",
            "sources": sources,
            "source_registry": sources,
            "clean_evidence_list": facts,
            "analysis_ready_evidence": facts[:90],
            "per_dimension": {"competition": {"clean_facts": facts}},
        },
        "writer_report": {"source_registry": sources, "report_markdown": "Body cites [1] and [2]."},
    }

    clean = extract_clean_evidence_from_package(pkg)
    recovered = [item for values in clean["dimensions"].values() for item in values]

    assert len(recovered) >= 250
    assert len(clean["sources"]) >= 250
    assert any(item["source_id"] == "1" for item in recovered)


def test_evidence_handoff_diagnostics_flags_compaction_and_source_collapse():
    sources = _handoff_sources(80)
    facts = [_handoff_fact(index) for index in range(1, 41)]
    package = {
        "evidence_package": {
            "summary": {"clean_fact_count": 300, "analysis_ready_count": 90},
            "metadata": {"raw_evidence_count": 504, "normalized_count": 504, "kept_count": 391},
        },
        "reformatter_evidence_package": {
            "payload_mode": "handoff",
            "sources": sources,
            "clean_evidence_list": facts,
        },
        "writer_report": {"source_registry": sources},
    }

    diagnostics = full_report.build_evidence_handoff_diagnostics(
        package,
        clean_evidence={"dimensions": {"competition": facts}},
        validation={"source_pool_count": 16, "unique_cited_source_count": 11},
    )

    assert diagnostics["counts"]["handoff_facts"] == 40
    assert "evidence_handoff_compacted" in diagnostics["flags"]
    assert "source_pool_collapsed" in diagnostics["flags"]


def test_web_normalization_preserves_rerank_scores_on_sources_and_raw_points():
    web_state = {
        "answer_text": "Market adoption reached 42% in 2026 [id:1].",
        "raw_output": {
            "dimension": "market",
            "search_results": [
                {
                    "source_id": 1,
                    "title": "Relevant report",
                    "url": "https://example.com/report",
                    "date": "2026-01-01",
                    "snippet": "Market adoption reached 42% in 2026.",
                    "web_final_score": 0.82,
                    "web_rerank_score": 0.91,
                    "web_rerank_rank": 1,
                    "task_term_score": 0.7,
                }
            ],
            "quality_processing": {"rerank": {"input_count": 12, "returned_count": 8}},
            "synthesis": {
                "confidence": 0.7,
                "structured_payload": {
                    "answer": {"evidence": "Market adoption reached 42% in 2026 [id:1]."},
                    "key_sources": [
                        {
                            "id": 1,
                            "title": "Relevant report",
                            "url": "https://example.com/report",
                            "snippet": "Market adoption reached 42% in 2026.",
                        }
                    ],
                    "confidence": 0.7,
                },
            },
        },
    }

    child = brain_agent_module.normalize_web_child_output(web_state, route="web", errors=[])

    assert child["key_sources"][0]["web_final_score"] == 0.82
    assert child["key_sources"][0]["retrieval_relevance_score"] == 0.82
    assert child["raw_data_points"][0]["web_rerank_score"] == 0.91
    assert child["raw_data_points"][0]["retrieval_relevance_score"] == 0.82
    assert child["rerank_diagnostics"]["returned_count"] == 8


def test_evidence_merger_uses_rerank_score_as_light_relevance_signal():
    package = merge_evidence_package(
        original_query="test market report",
        evidence_pool=[
            {
                "status": "success",
                "confidence": 0.7,
                "raw_data_points": [
                    {
                        "dimension": "market",
                        "metric": "adoption",
                        "value": "42%",
                        "period": "2026",
                        "evidence": "Market adoption reached 42% in 2026.",
                        "source_title": "Low relevance source",
                        "source_url": "https://example.com/low",
                        "source_type": "media",
                        "web_final_score": 0.2,
                    },
                    {
                        "dimension": "market",
                        "metric": "adoption",
                        "value": "43%",
                        "period": "2026",
                        "evidence": "Market adoption reached 43% in 2026.",
                        "source_title": "High relevance source",
                        "source_url": "https://example.com/high",
                        "source_type": "media",
                        "web_final_score": 0.9,
                    },
                ],
            }
        ],
    )

    facts = package["clean_evidence_list"]

    assert facts[0]["source"]["title"] == "High relevance source"
    assert facts[0]["retrieval_relevance_score"] == 0.9
    assert package["chapter_evidence"]
    assert package["summary"]["rerank_diagnostics"]["evidence_with_rerank_score_count"] >= 2


def test_evidence_package_reports_rerank_score_retention():
    package = merge_evidence_package(
        original_query="test report",
        evidence_pool=[
            {
                "status": "success",
                "confidence": 0.7,
                "rerank_diagnostics": {"input_count": 6, "returned_count": 4},
                "raw_data_points": [
                    {
                        "dimension": "market",
                        "metric": "growth",
                        "value": "25%",
                        "period": "2026",
                        "evidence": "Growth reached 25% in 2026.",
                        "source_title": "Reranked source",
                        "source_url": "https://example.com/reranked",
                        "source_type": "research",
                        "web_rerank_score": 0.88,
                    }
                ],
            }
        ],
    )

    diagnostics = package["rerank_diagnostics"]

    assert diagnostics["rerank_input_count"] == 6
    assert diagnostics["rerank_returned_count"] == 4
    assert diagnostics["evidence_with_rerank_score_count"] >= 1
    assert diagnostics["top_rerank_sources"][0]["title"] == "Reranked source"


def test_rerank_score_does_not_upgrade_source_level():
    package = merge_evidence_package(
        original_query="test report",
        evidence_pool=[
            {
                "status": "success",
                "confidence": 0.7,
                "raw_data_points": [
                    {
                        "dimension": "market",
                        "metric": "case signal",
                        "value": "30%",
                        "period": "2026",
                        "evidence": "A media case signal reported 30% adoption in 2026.",
                        "source_title": "Media case",
                        "source_url": "https://example.com/media",
                        "source_type": "media",
                        "web_final_score": 0.99,
                    }
                ],
            }
        ],
    )

    fact = package["clean_evidence_list"][0]

    assert fact["retrieval_relevance_score"] == 0.99
    assert fact["source_level"] not in {"A", "B"}


def test_report_contract_from_writer_package_keeps_chapter_requirements():
    package = {
        "query": "中美科技产业再连接",
        "evidence_package": {
            "metadata": {
                "research_plan": {
                    "query": "中美科技产业再连接",
                    "research_object": "中美科技产业互动",
                    "decision_context": "investment_or_market_entry",
                    "report_family": "industry_report",
                    "chapters": [
                        {
                            "chapter_id": "ch_01",
                            "chapter_title": "公司中国利益与政策边界",
                            "core_question": "中国利益是否被硬数据支撑",
                            "required_evidence_mix": ["official_data", "company_filing", "counter_evidence"],
                            "min_total_sources": 8,
                            "min_ab_sources": 2,
                            "min_counter_sources": 1,
                        }
                    ],
                }
            }
        },
    }

    contract = build_report_contract_from_package(package)

    assert contract["status"] == "active"
    assert contract["research_object"] == "中美科技产业互动"
    assert contract["chapters"][0]["chapter_id"] == "ch_01"
    assert contract["chapters"][0]["min_ab_sources"] == 2
    assert contract["quality_thresholds"]["minimum_unique_sources_when_available"] >= 8


def test_evidence_quality_contract_classifies_source_levels_and_allowed_use():
    core = apply_evidence_quality_contract(
        {
            "source_level": "A",
            "evidence_role": "core",
            "semantic_status": "ok",
            "confidence": 0.8,
            "metric": "shipments",
            "value": "120",
            "content": "Vendor shipments reached 120 units in 2025.",
        }
    )
    directional = apply_evidence_quality_contract(
        {
            "source_level": "C",
            "evidence_role": "clue",
            "semantic_status": "ok",
            "confidence": 0.7,
            "content": "Channel checks point to early demand.",
        }
    )
    weak = apply_evidence_quality_contract({"source_level": "D", "evidence_role": "clue", "content": "rumor"})

    assert core["allowed_use"] == "core_claim"
    assert core["appendix_only"] is False
    assert core["evidence_card"]["confidence_score"] >= 0.8
    assert directional["allowed_use"] == "directional_signal"
    assert directional["source_tier"].startswith("C")
    assert directional["analysis_readiness"] == "directional_ready"
    assert directional["appendix_only"] is False
    assert weak["allowed_use"] == "appendix_only"
    assert weak["appendix_only"] is True


def test_ab_hard_metric_keeps_proof_gaps_without_rejection():
    item = apply_evidence_quality_contract(
        {
            "source": {
                "source_type": "official",
                "title": "Official statistics",
                "url": "https://www.stats.gov.cn/example",
            },
            "evidence_role": "core",
            "semantic_status": "ok",
            "confidence": 0.86,
            "claim_type": "hard_metric",
            "metric": "market size",
            "value": "120",
            "period": "2025",
            "content": "Official statistics show the market size reached 120 in 2025.",
        }
    )

    assert item["source_level"] == "A"
    assert item["source_tier"] == "A1"
    assert item["allowed_use"] == "core_claim"
    assert set(item["metric_proof_gaps"]) >= {"scope", "unit"}
    assert item["analysis_readiness"] == "context_only"
    assert item["evidence_fit_score"] < 1.0


def test_balanced_evidence_contract_treats_consulting_as_b_source(monkeypatch):
    monkeypatch.setenv("REPORT_QUALITY_MODE", "balanced")
    item = apply_evidence_quality_contract(
        {
            "source": {"source_type": "consulting", "title": "Major consultancy report"},
            "evidence_role": "supporting",
            "semantic_status": "ok",
            "confidence": 0.72,
            "content": "The market is moving from model demos to workflow integration.",
        }
    )

    assert item["source_level"] == "B"
    assert item["source_subtier"] == "B+"
    assert item["allowed_use"] == "supporting"
    assert item["claim_type"] == "industry_analysis"


def test_binder_balanced_allows_multi_c_for_industry_directional_ready(monkeypatch):
    monkeypatch.setenv("REPORT_QUALITY_MODE", "balanced")
    result = run_evidence_binder(
        research_plan={
            "report_mode": "deep_industry_report",
            "hypotheses": [
                {
                    "hypothesis_id": "H1",
                    "statement": "AI video tools are shifting competition toward workflow integration.",
                    "claim_type": "industry_analysis",
                }
            ],
            "evidence_coverage_requirements": {"per_hypothesis": {"min_A_or_B_sources": 2}},
        },
        report_blueprint={"chapters": [{"chapter_id": "c1", "chapter_title": "Competition"}]},
        evidence_pool=[
            {
                "fact": "A major media report says AI video tools are adding workflow features.",
                "source_level": "C",
                "source": {"title": "Media one", "url": "https://media-one.example/a"},
                "hypothesis_id": "H1",
                "evidence_role": "supporting",
                "proof_role": "support",
                "confidence": 0.72,
            },
            {
                "fact": "Another public source reports creators are adopting workflow templates.",
                "source_level": "C",
                "source": {"title": "Media two", "url": "https://media-two.example/b"},
                "hypothesis_id": "H1",
                "evidence_role": "supporting",
                "proof_role": "support",
                "confidence": 0.74,
            },
        ],
    )

    row = result["coverage_matrix"][0]
    assert row["claim_status"] == "directional_ready"
    assert row["decision_ready"] is True
    assert row["readiness_level"] == "directional_ready"
    assert "insufficient_ab_sources" not in row["blocking_gaps"]
    assert row["directional_c_distinct_sources"] == 2


def test_binder_does_not_count_same_domain_c_sources_as_independent(monkeypatch):
    monkeypatch.setenv("REPORT_QUALITY_MODE", "balanced")
    result = run_evidence_binder(
        research_plan={
            "report_mode": "deep_industry_report",
            "hypotheses": [
                {
                    "hypothesis_id": "H1",
                    "statement": "AI video tools are shifting competition toward workflow integration.",
                    "claim_type": "industry_analysis",
                }
            ],
            "evidence_coverage_requirements": {"per_hypothesis": {"min_A_or_B_sources": 2}},
        },
        report_blueprint={"chapters": [{"chapter_id": "c1", "chapter_title": "Competition"}]},
        evidence_pool=[
            {
                "fact": "A public report says AI video tools are adding workflow features.",
                "source_level": "C",
                "source": {"title": "Media one", "url": "https://media.example/a"},
                "hypothesis_id": "H1",
                "evidence_role": "supporting",
                "proof_role": "support",
                "confidence": 0.72,
            },
            {
                "fact": "Another page on the same site reports workflow template adoption.",
                "source_level": "C",
                "source": {"title": "Media two", "url": "https://media.example/b"},
                "hypothesis_id": "H1",
                "evidence_role": "supporting",
                "proof_role": "support",
                "confidence": 0.74,
            },
        ],
    )

    row = result["coverage_matrix"][0]
    assert row["directional_c_sources"] == 2
    assert row["directional_c_distinct_sources"] == 1
    assert row["readiness_level"] == "context_only"
    assert "insufficient_ab_sources" in row["blocking_gaps"]


def test_evidence_gap_ledger_does_not_block_balanced_multi_c_industry(monkeypatch):
    monkeypatch.setenv("REPORT_QUALITY_MODE", "balanced")
    monkeypatch.setenv("EVIDENCE_CACHE_ENABLED", "false")

    package = merge_evidence_package(
        original_query="AI video workflow competition",
        evidence_pool=[
            {
                "status": "success",
                "agent": "iqs",
                "query": "AI video workflow competition",
                "raw_data_points": [
                    {
                        "evidence": "Creator teams are adopting workflow templates in AI video tools.",
                        "metric": "workflow adoption signal",
                        "value": "observed case",
                        "period": "2026",
                        "source_title": "Authoritative media one",
                        "source_url": "https://media-one.example/a",
                        "source_type": "media",
                        "confidence": 0.73,
                        "dimension_name": "workflow integration",
                        "evidence_goal": "workflow integration",
                        "must_have_terms": ["workflow"],
                        "source_priority": ["media"],
                        "hypothesis_id": "H1",
                        "hypothesis_statement": "AI video tools are shifting competition toward workflow integration.",
                        "claim_type": "industry_analysis",
                        "proof_role": "support",
                    },
                    {
                        "evidence": "A second source reports AI video vendors bundling workflow features.",
                        "metric": "workflow adoption signal",
                        "value": "observed case",
                        "period": "2026",
                        "source_title": "Authoritative media two",
                        "source_url": "https://media-two.example/b",
                        "source_type": "media",
                        "confidence": 0.74,
                        "dimension_name": "workflow integration",
                        "evidence_goal": "workflow integration",
                        "must_have_terms": ["workflow"],
                        "source_priority": ["media"],
                        "hypothesis_id": "H1",
                        "hypothesis_statement": "AI video tools are shifting competition toward workflow integration.",
                        "claim_type": "industry_analysis",
                        "proof_role": "support",
                    },
                ],
            }
        ],
        research_plan={"query": "AI video workflow competition"},
    )

    assert package["evidence_analysis_summary"]["balanced_directional_ready_chapter_count"] >= 1
    blocking_types = {
        gap.get("gap_type")
        for gap in package["evidence_gap_ledger"]
        if gap.get("severity") == "blocking"
    }
    assert "insufficient_ab_sources" not in blocking_types
    assert "directional_only_evidence" not in blocking_types
    assert "metric_scope_period_unit_incomplete" not in blocking_types
    assert package["evidence_analysis_summary"]["chapter_advisory_gap_type_distribution"][
        "metric_scope_period_unit_incomplete"
    ] == 1


def test_binder_still_blocks_hard_metric_when_only_c_sources(monkeypatch):
    monkeypatch.setenv("REPORT_QUALITY_MODE", "balanced")
    result = run_evidence_binder(
        research_plan={
            "report_mode": "deep_industry_report",
            "hypotheses": [
                {
                    "hypothesis_id": "H1",
                    "statement": "AI video market size reached 10 billion yuan.",
                    "claim_type": "hard_metric",
                }
            ],
            "evidence_coverage_requirements": {"per_hypothesis": {"min_A_or_B_sources": 1}},
        },
        report_blueprint={"chapters": [{"chapter_id": "c1", "chapter_title": "Market size"}]},
        evidence_pool=[
            {
                "fact": "A media article mentions AI video market size reached 10 billion yuan.",
                "source_level": "C",
                "source": {"title": "Media one", "url": "https://media.example/metric"},
                "hypothesis_id": "H1",
                "evidence_role": "supporting",
                "proof_role": "metric",
                "metric": "market size",
                "value": "10 billion yuan",
                "confidence": 0.78,
            }
        ],
    )

    row = result["coverage_matrix"][0]
    assert row["claim_type"] == "hard_metric"
    assert "insufficient_ab_sources" in row["blocking_gaps"]
    assert "metric_scope_period_unit_incomplete" in row["blocking_gaps"]
    assert set(row["metric_proof_gaps"]) >= {"period", "scope", "unit"}
    assert row["readiness_level"] == "context_only"
    assert row["decision_ready"] is False


def test_report_contract_adds_question_driven_block_and_title_policy():
    package = {
        "query": "foldable phone hinge bottleneck",
        "evidence_package": {
            "metadata": {
                "research_plan": {
                    "query": "foldable phone hinge bottleneck",
                    "chapters": [
                        {
                            "chapter_id": "ch_01",
                            "chapter_title": "\u5e02\u573a\u89c4\u6a21\u4e0e\u589e\u901f\uff1f",
                            "core_question": "Which bottleneck changes adoption?",
                            "required_evidence_mix": ["metric", "counter"],
                            "min_ab_sources": 2,
                        }
                    ],
                }
            }
        },
    }

    contract = build_report_contract_from_package(package)
    chapter = contract["chapters"][0]
    issue_types = {item["type"] for item in contract["contract_issues"]}

    assert chapter["chapter_title"] == "\u5e02\u573a\u89c4\u6a21\u4e0e\u589e\u901f"
    assert "metric" in chapter["required_evidence_roles"]
    assert "risk_trigger" in chapter["expected_blocks"]
    assert chapter["minimum_source_level"] == "A"
    assert "legacy_template_chapter_title" in issue_types
    assert "source_check" in chapter["required_evidence_roles"]
    assert "counter" in chapter["required_evidence_roles"]


def test_package_issue_summary_dedupes_blocking_errors():
    issue = {
        "package": "argument_units",
        "type": "core_claim_without_ab_source",
        "path": "[2]",
        "message": "decision-ready claims must bind at least one A/B source.",
    }

    summary = package_issue_summary([issue, dict(issue)])

    assert summary["count"] == 1
    assert summary["by_type"]["core_claim_without_ab_source"] == 1


def test_planner_contract_quality_hints_do_not_hard_block_clean_report():
    blueprint = {
        "chapters": [
            {
                "chapter_id": "ch_01",
                "chapter_title": "\u5e02\u573a\u89c4\u6a21\u4e0e\u589e\u901f",
                "chapter_question": "What evidence supports demand?",
                "core_question": "What evidence supports demand?",
                "min_total_sources": 0,
                "min_ab_sources": 0,
                "layout_policy": {},
            }
        ]
    }

    result = validate_report_blueprint(blueprint)
    warning_types = {item["type"] for item in result["warnings"]}

    assert result["passed"] is True
    assert "missing_report_family" in warning_types
    assert "legacy_fixed_chapter_title" in warning_types
    assert "missing_required_evidence_mix" in warning_types
    assert "weak_min_total_sources" in warning_types
    assert "weak_min_ab_sources" in warning_types


def test_argument_unit_counter_and_actionable_are_advisory_not_blocking():
    result = validate_argument_units(
        [
            {
                "question": "What changed?",
                "claim": "Demand evidence is directional.",
                "reasoning": "because official and market sources point in the same direction",
                "evidence_refs": ["1"],
                "claim_status": "directional",
                "source_quality": {"claim_ab_count": 0, "allowed_use_distribution": {}},
            }
        ]
    )
    warning_types = {item["type"] for item in result["warnings"]}

    assert result["passed"] is True
    assert "missing_counter_evidence" in warning_types
    assert "missing_actionable" in warning_types


def test_micro_layout_shape_hints_are_advisory_when_sections_exist():
    result = validate_micro_layouts(
        [
            {
                "chapter_id": "ch_01",
                "blocks": [{"block_type": "unknown"}],
                "sections": [{"section_title": "Evidence", "required_evidence_refs": ["1"]}],
            }
        ]
    )
    warning_types = {item["type"] for item in result["warnings"]}

    assert result["passed"] is True
    assert "missing_layout_type" in warning_types
    assert "unknown_block_type" in warning_types


def test_chapter_package_with_lead_but_no_sections_is_advisory():
    result = validate_chapter_packages([{"chapter_id": "ch_01", "lead": "Supported lead paragraph."}])

    assert result["passed"] is True
    assert {item["type"] for item in result["warnings"]} == {"sections_empty"}


def test_public_package_normalizer_fills_sections_and_demotes_bad_tables():
    normalized = _normalize_public_packages_for_contract(
        chapter_evidence_packages=[
            {
                "chapter_id": "ch_01",
                "core_evidence": [{"ref": "1", "source_ref": "1", "source_level": "A"}],
            }
        ],
        micro_layouts=[],
        table_packages=[
            {
                "table_id": "t1",
                "chapter_id": "ch_01",
                "purpose": "compare",
                "takeaway": "takeaway",
                "headers": ["metric"],
                "rows": [],
                "should_render": True,
            }
        ],
        argument_units=[
            {
                "chapter_id": "ch_01",
                "section_id": "s1",
                "section_title": "supported section",
                "claim": "supported claim",
                "reasoning": "because the source is direct",
                "counter_evidence": "watch boundary",
                "actionable": "verify source",
                "evidence_refs": ["1"],
                "public_render": True,
            }
        ],
        chapter_packages=[
            {
                "chapter_id": "ch_01",
                "lead": "lead",
                "sections": [{"section_id": "s1"}],
                "table_packages": [],
            }
        ],
    )

    assert normalized["table_packages"][0]["should_render"] is False
    assert normalized["summary"]["demoted_table_count"] == 1
    section = normalized["chapter_packages"][0]["sections"][0]
    assert section["claim"] == "supported claim"
    assert section["evidence_refs"] == ["1"]
    chapter_validation = validate_chapter_packages(normalized["chapter_packages"])
    assert chapter_validation["passed"] is True


def test_public_package_normalizer_fills_argument_unit_advisory_fields():
    normalized = _normalize_public_packages_for_contract(
        chapter_evidence_packages=[
            {
                "chapter_id": "ch_01",
                "core_evidence": [{"ref": "1", "source_ref": "1", "source_level": "A"}],
            }
        ],
        micro_layouts=[],
        table_packages=[],
        argument_units=[
            {
                "chapter_id": "ch_01",
                "question": "What changed?",
                "claim": "Demand evidence is directional.",
                "public_render": True,
                "claim_status": "directional",
                "source_quality": {"claim_ab_count": 0, "allowed_use_distribution": {}},
            }
        ],
        chapter_packages=[],
    )
    unit = normalized["argument_units"][0]
    result = validate_argument_units(normalized["argument_units"])

    assert unit["reasoning"]
    assert unit["counter_evidence"]
    assert unit["actionable"]
    assert unit["evidence_refs"] == ["1"]
    assert result["passed"] is True


def test_reformatter_adaptive_citation_gate_caps_configured_threshold(monkeypatch):
    monkeypatch.setenv("REPORT_REFORMATTER_ADAPTIVE_GATES", "true")
    monkeypatch.setenv("REPORT_REFORMATTER_MIN_BODY_CITATIONS", "40")
    monkeypatch.setenv("REPORT_REFORMATTER_MIN_UNIQUE_BODY_SOURCES", "30")
    facts = [
        {"text": f"fact {index}", "source": str((index % 12) + 1)}
        for index in range(80)
    ]
    clean_evidence = {"dimensions": {"test": facts}}
    markdown = "\n".join(f"paragraph [{(index % 6) + 1}]" for index in range(9))

    issues = _citation_density_issues(markdown, clean_evidence)
    total_issue = next(item for item in issues if item["reason"] == "total citation count is too low")
    unique_issue = next(item for item in issues if item["reason"] == "unique cited source count is too low")

    assert total_issue["required"] == 10
    assert total_issue["configured_required"] == 40
    assert unique_issue["required"] == 8
    assert unique_issue["configured_required"] == 30
    assert _source_diversity_floor(12) == 0


def test_qa_deep_required_followups_are_advisory_when_qa_passed():
    qa = {
        "passed": True,
        "repair_required": False,
        "deep_evaluation": {"required_followups": [{"type": "missing_proof_standard"}]},
    }

    assert _qa_has_pending_repair(qa) is False
    assert full_report.writer_report_pending_repair_reasons({"qa_result": qa, "required_followups": qa["deep_evaluation"]["required_followups"]}) == []


def test_local_rag_is_disabled_by_default_for_main_flow(monkeypatch):
    monkeypatch.delenv("BRAIN_ENABLE_LOCAL_RAG", raising=False)
    monkeypatch.delenv("REPORT_ENABLE_LOCAL_RAG", raising=False)

    route, reason = route_query("robotics industry opportunity", "all")

    assert _local_rag_enabled() is False
    assert route == "web"
    assert "Local RAG is disabled" in reason
    assert "rag" not in _route_agents("all")
    selected_nodes = select_child_agents({"route": "all"})
    assert "industry_rag_agent" not in selected_nodes
    assert any(str(node).startswith("iqs_lane_") for node in selected_nodes)


def test_local_rag_can_be_reenabled_explicitly(monkeypatch):
    monkeypatch.setenv("BRAIN_ENABLE_LOCAL_RAG", "1")

    route, reason = route_query("robotics industry opportunity", "all")

    assert _local_rag_enabled() is True
    assert route == "all"
    assert "route=all" in reason
    assert "rag" in _route_agents("all")
    assert "industry_rag_agent" in select_child_agents({"route": "all"})


def test_full_report_default_route_is_iqs_only(monkeypatch):
    monkeypatch.delenv("FULL_REPORT_ROUTE", raising=False)

    parser = full_report.build_arg_parser()
    args = parser.parse_args(["--query", "test topic"])

    assert args.route == "web"


def test_disabled_local_rag_followups_are_routed_to_iqs(monkeypatch):
    monkeypatch.delenv("BRAIN_ENABLE_LOCAL_RAG", raising=False)
    monkeypatch.delenv("REPORT_ENABLE_LOCAL_RAG", raising=False)
    monkeypatch.setenv("EVIDENCE_CACHE_EVIDENCE_READ_ENABLED", "false")
    monkeypatch.setenv("EVIDENCE_CACHE_SEARCH_READ_ENABLED", "false")
    seen_agents = []

    def fake_run_single_followup(**kwargs):
        seen_agents.append(kwargs["agent"])
        return {
            "round": kwargs["round_number"],
            "agent": kwargs["agent"],
            "child_agent": brain_agent_module.IQS_ROLE_CONFIGS.get(kwargs["agent"], {}).get("child", "web_analysis_agent"),
            "query": kwargs["query"],
            "targets_gap": kwargs["targets_gap"],
            "status": "failed",
            "confidence": 0.0,
            "answer": "",
            "key_sources": [],
        }

    monkeypatch.setattr(brain_agent_module, "_run_single_followup", fake_run_single_followup)

    results = brain_agent_module.run_followup_queries(
        follow_up_queries=[{"query": "foldable hinge official data", "agent": "both", "targets_gap": "missing official data"}],
        round_number=2,
        state={"session_id": ""},
    )

    assert results
    assert seen_agents
    assert "rag" not in seen_agents
    assert all(agent in brain_agent_module.IQS_ROLE_CONFIGS for agent in seen_agents)


def test_disabled_local_rag_child_is_not_added_to_evidence_pool(monkeypatch):
    monkeypatch.delenv("BRAIN_ENABLE_LOCAL_RAG", raising=False)
    monkeypatch.delenv("REPORT_ENABLE_LOCAL_RAG", raising=False)
    children = {
        "industry_rag_agent": {
            "status": "skipped",
            "confidence": 0.0,
            "note": "Local RAG is disabled for the main flow.",
        },
        "iqs_lane_1_agent": {
            "status": "success",
            "confidence": 0.7,
            "answer": "Official data shows a measurable signal.",
            "key_sources": [{"title": "Official", "url": "https://example.com"}],
            "raw_data_points": [{"metric": "shipments", "value": "10", "source_url": "https://example.com"}],
        },
    }

    pool = build_initial_evidence_pool(original_query="test", children=children)

    agents = [item["agent"] for item in pool]
    assert "rag" not in agents
    assert "iqs_lane_1" in agents


def test_followup_signal_requires_substantive_evidence_payload():
    assert _followup_result_has_signal(
        [{"status": "success", "confidence": 0.95, "answer": "", "key_sources": [], "raw_data_points": []}]
    ) is False
    assert _followup_result_has_signal(
        [{"status": "success", "confidence": 0.1, "key_sources": [{"title": "Official filing", "url": "https://example.com"}]}]
    ) is True
    assert _followup_result_has_signal(
        [{"status": "partial", "raw_data_points": [{"metric": "shipments", "value": "12%", "source_url": "https://example.com"}]}]
    ) is True
    assert _followup_result_has_signal(
        [{"status": "success", "answer": "没有找到足够公开资料，无法确认该判断。", "confidence": 0.9}]
    ) is False

    diagnostics = _followup_signal_diagnostics(
        [
            {"status": "success", "confidence": 0.9},
            {"status": "failed", "limitations": {"failure_reason": "timeout"}},
        ]
    )
    assert diagnostics["signal_count"] == 0
    assert diagnostics["empty_success_count"] == 1
    assert diagnostics["failed_count"] == 1


def test_technology_topic_forces_technology_product_lane():
    lanes = _infer_lane_types_for_task(
        {
            "query": "iPhone foldable hinge UTG yield technical bottleneck patent validation",
            "proof_role": "source_check",
            "source_priority": ["patent", "technical_standard"],
        }
    )

    assert "technology_product" in lanes


def test_ai_agent_search_tasks_keep_agentic_topic_anchors():
    chapter = {
        "chapter_id": "ch_01",
        "chapter_title": "AI Agent生态发展是否存在真实需求和可验证市场空间",
        "core_question": "AI Agent生态发展是否存在真实需求和可验证市场空间",
        "required_evidence_mix": ["official_data", "market_research", "customer_case", "technology_product"],
    }
    goal = {
        "goal_id": "ch_01_metric",
        "question": "AI Agent市场规模、客户采用和企业落地证据",
        "proof_role": "metric",
    }
    tasks = build_search_tasks_for_goal(
        chapter=chapter,
        goal=goal,
        research_plan={"query": "AI Agent生态发展报告：从工具到智能体的范式跃迁", "report_family": "industry_deep_report"},
    )

    assert any("AI Agent" in task["query"] or "Agentic AI" in task["query"] or "智能体" in task["query"] for task in tasks)
    assert any(task.get("prefer_deep") for task in tasks)


def test_auto_readpage_prioritizes_authoritative_urls(monkeypatch):
    monkeypatch.setenv("IQS_AUTO_READPAGE_ENABLED", "true")
    monkeypatch.setenv("IQS_AUTO_READPAGE_TOP_N", "2")
    urls = web_analysis_agent_module.select_auto_readpage_urls(
        [
            {"title": "self media repost", "url": "https://caifuhao.eastmoney.com/news/1"},
            {"title": "Company 10-K annual report", "url": "https://www.sec.gov/Archives/example-10k"},
            {"title": "Official product docs customer case", "url": "https://www.microsoft.com/en-us/ai/customer-stories"},
            {"title": "Sohu repost", "url": "https://m.sohu.com/a/123"},
        ],
        search_options={"proof_role": "source_check", "source_priority": ["filing", "official", "customer"]},
    )

    assert "https://www.sec.gov/Archives/example-10k" in urls
    assert "https://www.microsoft.com/en-us/ai/customer-stories" in urls
    assert all("eastmoney" not in url and "sohu" not in url for url in urls)


def test_source_registry_utilities_keep_refs_consistent():
    item = {"source_refs": ["[3]"], "evidence_refs": ["[1]", "[3]"], "source_ref": "[2]"}
    markdown, sources = renumber_sources_by_first_citation(
        "A fact[3] then another fact[1].",
        [{"ref": "[1]", "title": "one"}, {"ref": "[2]", "title": "two"}, {"ref": "[3]", "title": "three"}],
    )

    assert pick_refs(item, limit=4) == ["[3]", "[1]", "[2]"]
    assert markdown == "A fact[1] then another fact[2]."
    assert [source["title"] for source in sources[:3]] == ["three", "one", "two"]


def test_package_claim_gate_blocks_decision_ready_claim_without_ab_source():
    result = validate_argument_units(
        [
            {
                "question": "Where is the opportunity?",
                "claim": "The opportunity is already decision-ready.",
                "reasoning": "because the signal maps to demand",
                "counter_evidence": "opposite evidence could narrow it",
                "actionable": "track conversion and orders",
                "evidence_refs": ["[1]"],
                "claim_status": "decision_ready",
                "source_quality": {
                    "claim_ab_count": 0,
                    "ab_count": 0,
                    "allowed_use_distribution": {"directional_signal": 1},
                },
            }
        ]
    )

    assert result["passed"] is False
    assert any(item["type"] == "core_claim_without_ab_source" for item in result["errors"])


def test_package_claim_gate_allows_balanced_multi_c_directional_support(monkeypatch):
    monkeypatch.setenv("REPORT_QUALITY_MODE", "balanced")
    result = validate_argument_units(
        [
            {
                "question": "Where is the opportunity?",
                "claim": "Workflow integration is becoming a competitive axis.",
                "reasoning": "because multiple public signals show vendors moving from demos to workflow adoption",
                "counter_evidence": "weak enterprise conversion would narrow the conclusion",
                "actionable": "track enterprise feature releases and customer adoption",
                "evidence_refs": ["[1]", "[2]"],
                "claim_status": "decision_ready",
                "claim_type": "industry_analysis",
                "source_quality": {
                    "claim_ab_count": 0,
                    "ab_count": 0,
                    "directional_c_distinct_sources": 2,
                    "allowed_use_distribution": {"directional_signal": 2},
                },
            }
        ]
    )

    assert result["passed"] is True
    assert not any(item["type"] == "core_claim_without_ab_source" for item in result["errors"])


def test_claim_builder_downgrades_decision_claim_without_ab_source():
    units = run_claim_builder_agent(
        chapter_evidence_packages=[
            {
                "chapter_id": "ch_01",
                "chapter_title": "Opportunity",
                "chapter_question": "Where is the opportunity?",
                "supporting_evidence": [
                    {
                        "source_ref": "[1]",
                        "source_level": "C",
                        "allowed_use": "directional_signal",
                        "evidence_role": "clue",
                        "fact": "A directional signal exists.",
                    }
                ],
            }
        ],
        structured_analysis={
            "claim_units": [
                {
                    "chapter_id": "ch_01",
                    "question": "Where is the opportunity?",
                    "claim": "The opportunity is already decision-ready.",
                    "reasoning": "because the signal maps to demand",
                    "counter_evidence": "opposite evidence could narrow it",
                    "actionable": "track conversion and orders",
                    "evidence_refs": ["[1]"],
                    "claim_status": "decision_ready",
                }
            ]
        },
    )

    assert units[0]["claim_status"] == "directional"
    assert units[0]["claim_downgraded_reason"] == "decision_ready_without_ab_source"
    result = validate_argument_units(units)
    assert not any(item["type"] == "core_claim_without_ab_source" for item in result["errors"])


def test_claim_builder_keeps_decision_claim_with_ab_source():
    units = run_claim_builder_agent(
        chapter_evidence_packages=[
            {
                "chapter_id": "ch_01",
                "chapter_title": "Opportunity",
                "chapter_question": "Where is the opportunity?",
                "core_evidence": [
                    {
                        "source_ref": "[1]",
                        "source_level": "A",
                        "allowed_use": "core_claim",
                        "evidence_role": "core",
                        "fact": "An official metric supports the claim.",
                    }
                ],
            }
        ],
        structured_analysis={
            "claim_units": [
                {
                    "chapter_id": "ch_01",
                    "question": "Where is the opportunity?",
                    "claim": "The opportunity is decision-ready.",
                    "reasoning": "because official demand metrics support it",
                    "counter_evidence": "opposite evidence could narrow it",
                    "actionable": "track conversion and orders",
                    "evidence_refs": ["[1]"],
                    "claim_status": "decision_ready",
                }
            ]
        },
    )

    assert units[0]["claim_status"] == "decision_ready"
    assert units[0]["source_quality"]["claim_ab_count"] >= 1


def test_cache_only_core_blocker_requires_claim_binding_error():
    report = {
        "package_quality_report": {"blocking_errors": [{"type": "core_claim_without_ab_source"}]},
        "evidence_cache_summary": {"cache_live_refresh_miss_count": 41},
    }
    assert full_report.cache_only_core_claim_block_count(report) == 1

    report["package_quality_report"] = {"blocking_errors": []}
    assert full_report.cache_only_core_claim_block_count(report) == 0


def test_query_builder_standardizes_lane_query_package():
    package = build_query_package(
        {
            "query": "foldable iPhone hinge yield",
            "must_have_terms": ["Apple", "hinge"],
            "forbidden_terms": ["rumor"],
            "proof_role": "metric",
            "source_priority": ["official", "filing"],
        },
        lane_type="official_data",
        lane_focus="official filings",
    )

    assert package["query_contract_version"]
    assert package["proof_role"] == "metric"
    assert package["lane_type"] == "official_data"
    assert package["must_have_terms"] == ["Apple", "hinge"]


def test_quality_gate_state_routes_evidence_repairs_before_degrade():
    state = build_quality_gate_state(
        writer_status="final",
        writer_not_ready=False,
        writer_pending_repair_reasons=["evidence_repair_followups"],
        reformatter_result={"status": "skipped"},
        report_contract={"contract_version": "0.1.0", "quality_thresholds": {"minimum_body_chars": 12000}},
    )

    assert state["status"] == "needs_evidence"
    assert state["state"] == "qa_failed_needs_evidence"
    assert state["loop_target"] == "evidence_refinement"
    assert state["evidence_required"] is True
    assert state["degrade_allowed"] is False


def test_quality_gate_routes_report_contract_title_issue_to_rewrite():
    state = build_quality_gate_state(
        writer_status="final",
        writer_not_ready=False,
        writer_pending_repair_reasons=[],
        reformatter_result={"status": "skipped"},
        report_contract={
            "contract_version": "0.1.0",
            "contract_issues": [{"type": "legacy_template_chapter_title", "severity": "rewrite"}],
        },
    )

    assert state["status"] == "needs_rewrite"
    assert state["next_action"] == "rewrite"
    assert state["contract_repair_reasons"] == ["legacy_template_chapter_title"]


def test_quality_gate_routes_exhausted_post_qa_evidence_to_manual_review():
    state = build_quality_gate_state(
        writer_status="review_required",
        writer_not_ready=False,
        writer_pending_repair_reasons=["evidence_repair_followups", "required_followups"],
        reformatter_result={"status": "skipped"},
        report_contract={"contract_version": "0.1.0"},
        writer_report={"post_qa_repair": {"status": "no_new_evidence_signal", "stop_reason": "no_new_evidence_signal"}},
    )

    assert state["status"] == "needs_degrade_or_manual_review"
    assert state["next_action"] == "manual_review"
    assert state["loop_target"] == "manual_review"
    assert state["degrade_allowed"] is True
    assert state["publishable"] is False
    assert "post_qa_no_new_evidence_signal" in state["blocking_reasons"]


def test_repair_quality_gain_detects_claim_binding_improvement():
    before_report = {
        "report_status": "review_required",
        "package_quality_report": {
            "blocking_errors": [
                {"type": "core_claim_without_ab_source", "path": "argument_units[0]"}
            ]
        },
    }
    after_report = {
        "report_status": "final",
        "package_quality_report": {"blocking_errors": []},
    }
    package = {"summary": {"source_level_distribution": {"A": 1, "B": 0}, "evidence_count": 1}}

    gain = _repair_quality_gain(before_report, after_report, package, package)

    assert gain["has_quality_gain"] is True
    assert gain["claim_binding_delta"] == 1
    assert gain["next_route"] == "continue"


def test_lane_budget_protects_metric_source_check_and_filing_per_chapter():
    lane_tasks = []
    for chapter_id in ["ch_01", "ch_02", "ch_03", "ch_04"]:
        for role in ["support", "metric", "source_check", "filing", "case"]:
            lane_tasks.append(
                {
                    "task_id": f"{chapter_id}_{role}",
                    "chapter_id": chapter_id,
                    "proof_role": role,
                    "scheduled_lane_type": "official_data",
                    "query": f"{chapter_id} {role}",
                }
            )

    selected, dropped = _select_lane_tasks_for_budget(lane_tasks, 12)
    by_chapter = {}
    for task in selected:
        by_chapter.setdefault(task["chapter_id"], set()).add(task["proof_role"])

    assert len(selected) == 12
    assert dropped
    assert all({"metric", "source_check", "filing"}.issubset(roles) for roles in by_chapter.values())


def test_lane_budget_protects_case_counter_and_technology_when_budget_allows():
    lane_tasks = []
    protected_roles = ["support", "metric", "source_check", "filing", "case", "counter", "technology_product"]
    for chapter_id in ["ch_01", "ch_02", "ch_03"]:
        for role in protected_roles:
            lane_tasks.append(
                {
                    "task_id": f"{chapter_id}_{role}",
                    "chapter_id": chapter_id,
                    "proof_role": role,
                    "scheduled_lane_type": "market_research",
                    "query": f"{chapter_id} {role}",
                }
            )

    selected, dropped = _select_lane_tasks_for_budget(lane_tasks, 18)
    by_chapter = {}
    for task in selected:
        by_chapter.setdefault(task["chapter_id"], set()).add(task["proof_role"])

    assert len(selected) == 18
    assert dropped
    assert all(
        {"metric", "source_check", "filing", "case", "counter", "technology_product"}.issubset(roles)
        for roles in by_chapter.values()
    )


def test_quality_first_initial_lanes_force_industry_customer_and_technology(monkeypatch):
    monkeypatch.setenv("BRAIN_IQS_INITIAL_LANE_QUALITY_FIRST", "true")
    monkeypatch.setenv("BRAIN_IQS_INITIAL_LANE_TOP_N", "4")
    monkeypatch.setenv("BRAIN_IQS_INITIAL_DISABLED_LANES", "customer_case")
    selection = _select_quality_first_initial_lanes(
        query="AI Agent生态发展报告：从工具到智能体的范式跃迁",
        agents=[f"iqs_lane_{index}" for index in range(1, 7)],
        dynamic_tasks=[],
        research_plan={"report_family": "industry_deep_report"},
    )

    assert selection["enabled"] is True
    assert "technology_product" in selection["selected_lane_types"]
    assert "customer_case" in selection["selected_lane_types"]
    assert len(selection["selected_roles"]) == 6


def test_lane_early_stop_requires_quality_threshold(monkeypatch):
    monkeypatch.setenv("BRAIN_IQS_LANE_EARLY_STOP_ENABLED", "true")
    monkeypatch.setenv("BRAIN_IQS_LANE_EARLY_STOP_MIN_SECONDS", "0")
    payloads = [
        {
            "key_sources": [{"title": "official filing", "url": "https://example.gov/a", "source_type": "official"}],
            "page_results": [{"title": "official filing", "url": "https://example.gov/a", "source_type": "official"}],
            "task_result": {"task": {"chapter_id": "ch_01", "proof_role": "metric"}},
        },
        {
            "raw_data_points": [{"metric": "shipments", "value": "10", "source": "official"}],
            "page_results": [{"title": "annual report", "url": "https://example.gov/b", "source_type": "official"}],
            "task_result": {"task": {"chapter_id": "ch_02", "proof_role": "source_check"}},
        },
    ]

    decision = _lane_early_stop_decision(payloads, started_at=0)

    assert decision["early_stopped"] is True
    assert decision["early_stop_reason"] in {"ab_source_and_metric_found", "enough_ab_sources"}


def test_high_value_repair_selector_keeps_core_tasks_and_skips_length(monkeypatch):
    monkeypatch.setenv("BRAIN_REPAIR_MAX_TASKS_PER_ROUND", "3")
    monkeypatch.setenv("BRAIN_POST_QA_REPAIR_MAX_EVIDENCE_TASKS", "3")
    monkeypatch.setenv("BRAIN_REPAIR_MAX_TASKS_PER_CHAPTER", "5")
    tasks = []
    for index, reason in enumerate(
        [
            "support",
            "core_claim_without_ab_source",
            "metric_scope_period_unit_incomplete",
            "counter_evidence_missing",
            "citation_source_missing",
        ],
        start=1,
    ):
        tasks.append(
            {
                "query": f"query {index}",
                "agent": "iqs_lane_1",
                "search_task": {
                    "query": f"query {index}",
                    "gap_id": f"g{index}",
                    "chapter_id": "ch_01",
                    "blocking_gaps": [reason],
                    "proof_role": "source_check",
                    "loop_name": "post_qa_repair",
                },
            }
        )
    tasks.append(
        {
            "query": "length",
            "agent": "iqs_lane_1",
            "search_task": {"query": "length", "type": "report_body_below_target_chars", "chapter_id": "ch_02"},
        }
    )

    selected = _select_high_value_repair_tasks(tasks, state={"metadata": {}}, round_number=1)
    selected_markers = {item["search_task"]["blocking_gaps"][0] for item in selected}

    assert len(selected) == 3
    assert "core_claim_without_ab_source" in selected_markers
    assert "metric_scope_period_unit_incomplete" in selected_markers
    assert all(item["search_task"].get("type") != "report_body_below_target_chars" for item in selected)


def test_lane_circuit_breaker_reroutes_core_and_skips_low_priority():
    state = {
        "iqs_lane_3_state": {
            "raw_output": {
                "lane_coverage": {
                    "scheduled": 4,
                    "timed_out_task_count": 3,
                    "usable_source_count": 0,
                }
            }
        },
        "metadata": {},
    }
    tasks = [
        {
            "query": "core",
            "agent": "iqs_lane_3",
            "search_task": {"query": "core", "proof_role": "metric", "blocking_gaps": ["insufficient_ab_sources"]},
        },
        {
            "query": "support",
            "agent": "iqs_lane_3",
            "search_task": {"query": "support", "proof_role": "support"},
        },
    ]

    updated = _apply_lane_circuit_breaker_to_tasks(tasks, state)

    assert len(updated) == 1
    assert updated[0]["agent"] == "iqs_lane_1"
    assert updated[0]["search_task"]["lane_circuit_breaker"]["reason"] == "timeout_exhausted"


def test_post_qa_repair_missing_proof_query_is_short_and_lane_targeted():
    payload = _post_qa_repair_followup_payload(
        {
            "type": "missing_proof_standard",
            "hypothesis_id": "H2",
            "hypothesis_statement": "iPhone折叠屏受阻，谁能突破技术瓶颈？又有哪些产业机会？的行情是否得到价格、产能、订单和盈利质量支撑",
            "blocking_gaps": ["insufficient_ab_sources", "metric_scope_period_unit_incomplete"],
            "suggested_query": "iPhone折叠屏受阻，谁能突破技术瓶颈？又有哪些产业机会？的行情是否得到价格、产能、订单和盈利质量支撑 A/B来源 反证 指标口径 官方 公告 研报",
        }
    )

    assert len(payload["query"]) <= 120
    assert "行情是否得到价格、产能、订单和盈利质量支撑" not in payload["query"]
    assert payload["lane_targets"][:3] == ["official_data", "filing_company", "market_research"]
    assert "technology_product" in payload["lane_targets"]
    assert payload["source_priority"] == ["官方", "公告", "财报", "协会", "统计", "权威研报"]


def test_post_qa_repair_queries_keep_hypotheses_distinct():
    base = {
        "type": "missing_proof_standard",
        "blocking_gaps": ["insufficient_ab_sources", "metric_scope_period_unit_incomplete"],
        "suggested_query": "foldable iPhone bottleneck opportunity",
    }
    queries = {
        hypothesis_id: _post_qa_repair_query_from_item(
            {
                **base,
                "hypothesis_id": hypothesis_id,
                "hypothesis_statement": statement,
            }
        )
        for hypothesis_id, statement in {
            "H1": "foldable iPhone demand shipments penetration replacement",
            "H2": "foldable iPhone supplier price capacity orders margin",
            "H3": "foldable iPhone mass production validation customer certification",
            "H4": "foldable iPhone risk counter evidence delay failure",
        }.items()
    }

    assert len(set(queries.values())) == 4
    assert "shipments" in queries["H1"] or "出货量" in queries["H1"] or "需求" in queries["H1"]
    assert "price" in queries["H2"] or "价格" in queries["H2"]
    assert "validation" in queries["H3"] or "量产验证" in queries["H3"]
    assert "risk" in queries["H4"] or "风险事件" in queries["H4"] or "反证" in queries["H4"]


def test_post_qa_repair_body_length_is_rewrite_only():
    plan = _post_qa_repair_plan(
        {
            "report_status": "review_required",
            "qa_pending_repair": True,
            "required_followups": [
                {"type": "report_body_below_target_chars", "required": 20000, "actual": 8500, "priority": "high"}
            ],
            "qa_result": {"deep_evaluation": {"required_followups": []}},
        },
        max_queries=5,
    )

    assert plan["evidence_followups"] == []
    assert plan["rewrite_required"] is True
    assert plan["skipped_non_evidence"]


def test_report_contract_derives_short_titles_and_default_fields():
    package = {
        "query": "iPhone折叠屏受阻，谁能突破技术瓶颈？又有哪些产业机会？",
        "report_blueprint": {
            "report_family": "supply_chain_report",
            "chapters": [
                {
                    "chapter_id": "ch_01",
                    "chapter_title": "iPhone折叠屏受阻，谁能突破技术瓶颈？又有哪些产业机会？是否存在真实需求，而不是概念热度",
                    "chapter_question": "iPhone折叠屏受阻，谁能突破技术瓶颈？又有哪些产业机会？是否存在真实需求，而不是概念热度",
                }
            ],
        },
    }

    contract = build_report_contract_from_package(package)
    chapter = contract["chapters"][0]
    issue_types = {item["type"] for item in contract["contract_issues"]}

    assert chapter["chapter_title"] == "真实需求验证"
    assert chapter["minimum_source_level"] in {"A", "B"}
    assert {"metric", "source_check", "counter"}.issubset(set(chapter["required_evidence_roles"]))
    assert {"thesis", "evidence_matrix", "mechanism_chain", "risk_trigger"}.issubset(set(chapter["expected_blocks"]))
    assert "chapter_required_evidence_roles_defaulted" in issue_types


def test_full_report_diagnostics_summarize_evidence_and_review_gaps():
    writer_report = {
        "report_status": "review_required",
        "post_qa_repair": {"status": "no_new_evidence_signal", "has_signal": False},
        "qa_result": {
            "passed": True,
            "quality_score": 88,
            "repair_required": True,
            "repair_followups": [
                {
                    "type": "missing_proof_standard",
                    "hypothesis_id": "H2",
                    "blocking_gaps": ["insufficient_ab_sources"],
                    "priority": "high",
                }
            ],
        },
    }
    chapter_packages = [
        {
            "chapter_id": "ch_02",
            "chapter_title": "行情支撑与盈利质量",
            "evidence_quality_summary": {"core_ab_source_count": 0, "core_evidence_count": 0},
            "missing_evidence": [{"type": "insufficient_core_evidence"}],
        }
    ]
    search_schedule = {
        "dropped_count": 1,
        "dropped_tasks": [
            {
                "task_id": "t1",
                "chapter_id": "ch_02",
                "proof_role": "filing",
                "scheduled_lane_type": "filing_company",
                "drop_reason": "max_tasks_per_lane",
                "query": "foldable filing",
            }
        ],
    }

    gap_summary = full_report.summarize_evidence_gaps(
        writer_report=writer_report,
        chapter_evidence_packages=chapter_packages,
        search_task_schedule=search_schedule,
        post_qa_repair_trace=[{"status": "no_new_evidence_signal", "has_signal": False}],
    )
    diagnostic = full_report.build_review_diagnostic(
        writer_report=writer_report,
        report_blueprint={"chapters": [{"chapter_title": "This title is intentionally far too long to be a clean report section title?"}]},
        chapter_evidence_packages=chapter_packages,
        package_quality_report={"warnings": [{"type": "table_validation_error"}]},
        evidence_gap_summary=gap_summary,
    )

    assert gap_summary["status"] == "has_gaps"
    assert gap_summary["search_tasks_dropped"]["by_proof_role"]["filing"] == 1
    assert "low_directness" in gap_summary["chapter_gaps"][0]["gap_reasons"]
    assert diagnostic["status"] == "needs_review"
    assert diagnostic["checks"]["post_qa_repair_failed"] is True
    blocker_summary = full_report.build_qa_blocker_summary(
        writer_report=writer_report,
        evidence_gap_summary=gap_summary,
        review_diagnostic=diagnostic,
        reformatter_result={"status": "skipped"},
        writer_pending_repair_reasons=["evidence_repair_followups"],
    )
    assert blocker_summary["status"] == "blocked"
    assert "evidence_gap" in blocker_summary["advisory_types"]
    assert "post_qa_no_new_evidence_signal" in blocker_summary["blocker_types"]
    assert "qa_repair_required" in blocker_summary["blocker_types"]
    assert (
        full_report.clean_report_blocked_reason(
            writer_publishable=False,
            writer_not_ready=False,
            reformatter_skip_reason="",
            qa_blocker_summary=blocker_summary,
        )
        == "post_qa_no_new_evidence_signal"
    )


def test_full_report_priority_output_writes_review_draft_even_when_not_publishable():
    output_dir = Path("output/_unit_priority_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "sample_review_draft.md"
    result = full_report.write_priority_report_output(
        output_path=output_path,
        writer_report={"report_status": "review_required", "report_markdown": "# Draft\n\nBody [1]"},
        report_markdown="",
        writer_status="review_required",
        reason="qa_repair_required",
    )

    assert result["output_written"] is True
    assert result["clean_report"] is False
    assert output_path.exists()
    assert "Body" in output_path.read_text(encoding="utf-8")


def test_llm_review_schema_flags_invalid_citation_and_uncited_inference():
    report = (
        "# 测试报告\n\n"
        "## 核心判断\n"
        "Therefore this market opportunity is already clear enough for priority investment.\n\n"
        "有来源的事实可以保留，但这里引用了不存在的来源[9]。\n"
    )

    review = build_structured_review(
        original_report=report,
        revised_report=report,
        evidence={"sources": [{"id": "1", "title": "来源一"}]},
    )

    assert review["status"] == "needs_evidence"
    assert review["evidence_required"] is True
    assert any(item.get("type") == "invalid_citation" for item in review["citation_issues"])
    assert any(item.get("type") == "uncited_inference" for item in review["logic_issues"])
    assert review["evidence_followups"]


def test_review_pipeline_returns_structured_review_when_llm_skipped():
    result = run_review_pipeline_sync(
        "# 测试报告\n\n## 主体分析\n正文事实需要保留引用[1]。\n",
        skip_llm_review=True,
    )

    assert result["stage2_skipped"] is True
    assert result["structured_review"]["schema_version"]
    assert result["structured_review"]["revised_report"] == result["final_report"]


def test_llm_review_splitter_keeps_large_report_bounded():
    markdown = "# 测试报告\n\n" + "\n\n".join(f"## 章节{i}\n" + ("正文内容。" * 250) for i in range(1, 6))

    chunks = split_markdown_for_review(markdown, max_chars=1800)

    assert len(chunks) >= 5
    assert all(item["input_chars"] <= 1800 for item in chunks)


def test_reformatter_blocks_large_source_pool_with_too_few_unique_citations(monkeypatch):
    monkeypatch.setenv("REPORT_REFORMATTER_VALIDATION_MODE", "score")
    monkeypatch.setenv("REPORT_REFORMATTER_MIN_PASS_SCORE", "60")
    sources = [{"id": str(i), "title": f"来源{i}", "url": f"https://example.com/{i}"} for i in range(1, 65)]
    markdown = (
        "# 测试报告\n\n"
        "## 主体分析\n"
        + "这段报告反复依赖少数来源，无法代表完整证据池[1][2][3][4][5][6]。" * 40
        + "\n\n## 数据来源\n"
        + "\n".join(f"[{i}] 来源{i}" for i in range(1, 65))
    )

    validation = validate_reformatted_report(markdown, sources, {"sources": sources, "dimensions": {}})

    assert validation["passed"] is False
    assert any(item.get("type") == "source_diversity_too_low" for item in validation["fatal_blockers"])


def test_reformatter_source_diversity_uses_usable_evidence_sources(monkeypatch):
    monkeypatch.setenv("REPORT_REFORMATTER_VALIDATION_MODE", "score")
    monkeypatch.setenv("REPORT_REFORMATTER_MIN_PASS_SCORE", "60")
    sources = [{"id": str(i), "title": f"来源{i}", "url": f"https://example.com/{i}"} for i in range(1, 149)]
    usable_facts = [
        {"text": f"可核验事实{i}包含足够正文信息", "source": str(i)}
        for i in range(1, 22)
    ]
    markdown = (
        "# 测试报告\n\n"
        "## 主体分析\n"
        + " ".join(f"正文事实链条需要保留来源[{i}]。" for i in range(1, 17))
        + ("正文补充分析。" * 160)
    )

    validation = validate_reformatted_report(
        markdown,
        sources,
        {"sources": sources, "dimensions": {"主体分析": usable_facts}},
    )

    assert validation["source_pool_count"] == 21
    assert validation["source_registry_count"] == 148
    assert not any(item.get("type") == "source_diversity_too_low" for item in validation["fatal_blockers"])


def test_reformatter_repair_plan_routes_sparse_evidence_to_followup(monkeypatch):
    monkeypatch.setenv("REPORT_REFORMATTER_EVIDENCE_LOOP_MIN_FACTS", "18")
    validation = {
        "passed": False,
        "source_diversity_required": 8,
        "source_pool_count": 3,
        "unique_cited_source_count": 2,
        "body_length_issues": [{"actual": 4000, "required": 12000}],
        "fatal_blockers": [{"type": "source_diversity_too_low", "actual": 2, "required": 8}],
    }
    clean_evidence = {
        "topic": "中美产业分化",
        "sources": [{"id": "1", "title": "来源1"}],
        "dimensions": {
            "半导体出口管制": [{"text": "事实1", "source": "1"}],
            "新能源关税": [],
        },
    }

    plan = build_reformatter_repair_plan(validation, clean_evidence, topic="中美产业分化")

    assert plan["status"] == "needs_evidence_refinement"
    assert plan["follow_up_queries"]
    assert "半导体出口管制" in plan["follow_up_queries"][0]["query"] or "新能源关税" in plan["follow_up_queries"][0]["query"]


def test_reformatter_repair_plan_does_not_refine_passed_validation():
    plan = build_reformatter_repair_plan(
        {
            "passed": True,
            "source_diversity_required": 8,
            "unique_cited_source_count": 8,
        },
        {
            "sources": [{"id": "1", "title": "来源1"}],
            "dimensions": {"主体分析": [{"text": "事实1", "source": "1"}]},
        },
    )

    assert plan["status"] == "passed"
    assert plan["follow_up_queries"] == []


def test_reformatter_repair_plan_does_not_treat_soft_score_pass_as_done(monkeypatch):
    monkeypatch.setenv("REPORT_REFORMATTER_REPAIR_SOFT_ISSUES", "true")
    validation = {
        "passed": True,
        "hard_pass": False,
        "soft_issue_count": 2,
        "source_diversity_required": 8,
        "source_pool_count": 18,
        "unique_cited_source_count": 6,
        "citation_density_issues": [{"actual": 6, "required": 18}],
    }
    clean_evidence = {
        "topic": "中美产业分化",
        "sources": [{"id": str(i), "title": f"来源{i}"} for i in range(1, 19)],
        "dimensions": {
            "半导体出口管制": [
                {"text": f"事实{i}", "source": str(i)}
                for i in range(1, 19)
            ]
        },
    }

    plan = build_reformatter_repair_plan(validation, clean_evidence, topic="中美产业分化")

    assert plan["status"] == "needs_text_repair"
    assert "citation_density_can_be_fixed_from_existing_evidence" in plan["text_repair_reasons"]


def test_reformatter_repair_plan_still_queries_when_dimensions_have_some_facts(monkeypatch):
    monkeypatch.setenv("REPORT_REFORMATTER_MIN_FACTS_PER_DIMENSION", "2")
    clean_evidence = {
        "topic": "中美产业分化",
        "sources": [{"id": str(i), "title": f"来源{i}"} for i in range(1, 4)],
        "dimensions": {
            "半导体出口管制": [
                {"text": "事实1", "source": "1"},
                {"text": "事实2", "source": "2"},
            ],
            "新能源关税": [
                {"text": "事实3", "source": "3"},
                {"text": "事实4", "source": "1"},
            ],
        },
    }

    plan = build_reformatter_repair_plan(
        {
            "passed": False,
            "source_diversity_required": 8,
            "source_pool_count": 3,
            "unique_cited_source_count": 2,
        },
        clean_evidence,
        topic="中美产业分化",
    )

    assert plan["status"] == "needs_evidence_refinement"
    assert plan["follow_up_queries"]


def test_writer_quality_prefers_resolved_reformatter_preflight():
    base = {
        "report_status": "final",
        "validation": {"passed": True, "quality_score": 80, "errors": [], "warnings": []},
        "layout_plan": {"layout_gaps": []},
        "estimated_chars": 10000,
    }
    needs_evidence = {
        **base,
        "reformatter_preflight": {"repair_plan": {"status": "needs_evidence_refinement"}},
    }
    needs_text = {
        **base,
        "reformatter_preflight": {"repair_plan": {"status": "needs_text_repair"}},
    }

    assert _writer_quality_key(needs_text) > _writer_quality_key(needs_evidence)


def test_reformatter_keeps_cited_source_appendix_by_default(monkeypatch):
    monkeypatch.delenv("REPORT_REFORMATTER_SOURCE_APPENDIX", raising=False)
    monkeypatch.delenv("REPORT_REFORMATTER_REQUIRE_SOURCE_APPENDIX", raising=False)
    markdown = (
        "# 测试报告\n\n"
        "## 正文分析\n"
        "正文事实需要保留正文内引用[1]。\n\n"
        "## 数据来源列表\n"
        "[1] Acme source\n"
    )

    cleaned = clean_reformatted_report(markdown, [{"id": "1", "title": "Acme source"}])
    validation_markdown = clean_reformatted_report(
        "# 测试报告\n\n## 正文分析\n" + ("正文事实需要保留正文内引用[1]。" * 120),
        [{"id": "1", "title": "Acme source"}],
    )
    validation = validate_reformatted_report(
        validation_markdown,
        [{"id": "1", "title": "Acme source"}],
        {"sources": [{"id": "1", "title": "Acme source"}], "dimensions": {}},
    )

    assert "## 参考资料" in cleaned
    assert "Acme source" in cleaned
    assert "数据来源列表" not in cleaned
    assert "[1]" in cleaned
    assert not any(item.get("type") == "missing_sources_appendix" for item in validation["fatal_blockers"])


def test_reformatter_blocks_missing_source_appendix_when_required(monkeypatch):
    monkeypatch.setenv("REPORT_REFORMATTER_SOURCE_APPENDIX", "none")
    monkeypatch.setenv("REPORT_REFORMATTER_REQUIRE_SOURCE_APPENDIX", "true")
    monkeypatch.setenv("REPORT_REFORMATTER_VALIDATION_MODE", "score")
    monkeypatch.setenv("REPORT_REFORMATTER_MIN_BODY_CHARS", "0")
    markdown = (
        "# 测试报告\n\n"
        "## 正文分析\n"
        + ("正文事实需要保留正文内引用[1]。" * 80)
    )

    validation = validate_reformatted_report(
        markdown,
        [{"id": "1", "title": "Acme source"}],
        {
            "sources": [{"id": "1", "title": "Acme source"}],
            "dimensions": {"正文分析": [{"text": "事实", "source": "1"}]},
        },
    )

    assert validation["passed"] is False
    assert any(item.get("type") == "missing_sources_appendix" for item in validation["fatal_blockers"])


def test_reformatter_payload_preserves_article_title_and_subject():
    package = {
        "query": "2026年国产AI视频工具竞争格局与底层基础设施演进报告",
        "article_brief": {
            "display_title": "大模型破晓，视觉纪元开启",
            "display_subtitle": "2026年国产AI视频工具竞争格局与底层基础设施演进报告",
            "planning_query": "2026年国产AI视频工具竞争格局与底层基础设施演进报告",
        },
        "report_blueprint": {
            "report_title": "大模型破晓，视觉纪元开启",
            "report_subtitle": "2026年国产AI视频工具竞争格局与底层基础设施演进报告",
        },
        "source_registry": [{"id": "1", "title": "来源一", "url": "https://example.com"}],
        "evidence_package": {
            "normalized_evidence": [
                {"text": "国产AI视频工具出现客户使用案例和产品竞争信号。", "source_id": "1", "dimension": "竞争格局"}
            ]
        },
    }

    clean = extract_clean_evidence_from_package(package)
    payload = build_reformatter_payload(clean)

    assert clean["report_title"] == "大模型破晓，视觉纪元开启"
    assert clean["report_subtitle"] == "2026年国产AI视频工具竞争格局与底层基础设施演进报告"
    assert clean["research_object"] == "国产AI视频工具"
    assert payload["display_title"] == "大模型破晓，视觉纪元开启"
    assert payload["display_subtitle"] == "2026年国产AI视频工具竞争格局与底层基础设施演进报告"


def test_reformatter_flags_topic_drift_from_specific_subject(monkeypatch):
    monkeypatch.setenv("REPORT_REFORMATTER_SOURCE_APPENDIX", "cited")
    monkeypatch.setenv("REPORT_REFORMATTER_VALIDATION_MODE", "score")
    monkeypatch.setenv("REPORT_REFORMATTER_MIN_BODY_CHARS", "0")
    sources = [{"id": "1", "title": "来源一", "url": "https://example.com"}]
    markdown = clean_reformatted_report(
        "# 大模型破晓，视觉纪元开启\n\n"
        "## 通用判断\n"
        + ("人工智能行业仍有产业规模和应用扩张机会，需要关注人工智能行业的算力、数据与商业化。人工智能行业竞争正在变化[1]。" * 20),
        sources,
    )

    validation = validate_reformatted_report(
        markdown,
        sources,
        {"sources": sources, "research_object": "国产AI视频工具", "dimensions": {}},
    )

    assert validation["passed"] is False
    assert any(item.get("type") == "topic_drift" for item in validation["repair_blockers"])


def test_reformatter_flags_high_frequency_cited_source_without_url(monkeypatch):
    monkeypatch.setenv("REPORT_REFORMATTER_SOURCE_APPENDIX", "cited")
    monkeypatch.setenv("REPORT_REFORMATTER_VALIDATION_MODE", "score")
    monkeypatch.setenv("REPORT_REFORMATTER_MIN_BODY_CHARS", "0")
    monkeypatch.setenv("REPORT_REFORMATTER_MISSING_URL_CITATION_MIN", "3")
    sources = [{"id": "1", "title": "缺URL来源"}]
    markdown = clean_reformatted_report(
        "# 测试报告\n\n## 正文分析\n" + ("这是一条需要复核来源URL的事实[1]。" * 5),
        sources,
    )

    validation = validate_reformatted_report(markdown, sources, {"sources": sources, "dimensions": {}})

    assert "URL缺失，需复核" in markdown
    assert validation["passed"] is False
    assert any(item.get("type") == "cited_source_url_missing" for item in validation["repair_blockers"])


def test_final_writer_does_not_render_appendix_by_default(monkeypatch):
    monkeypatch.delenv("REPORT_FINAL_WRITER_SOURCE_APPENDIX", raising=False)

    result = run_final_writer_agent(
        query="Appendix regression",
        report_blueprint={"report_shell": {"front_blocks": [], "back_blocks": ["appendix"]}},
        source_registry=[{"ref": "[1]", "title": "Acme source", "url": "https://example.com"}],
        appendix_package={"notes": ["Acme source"]},
    )

    assert "Acme source" not in result["report_markdown"]


def test_competitor_rows_skip_unknown_subjects():
    rows = _collect_rows(
        chapter_evidence_packages=[
            {
                "chapter_id": "c1",
                "chapter_title": LONG_AI_CHAPTER_TITLE,
                "core_evidence": [
                    {
                        "fact": "\u7ade\u4e89\u683c\u5c40\u663e\u793a\u5934\u90e8\u4f01\u4e1a\u4efd\u989d\u53d8\u5316",
                        "source": {"title": "LOOK ~~~~~~~~~~~~~~~~~~~~~"},
                        "source_ref": "[1]",
                        "source_level": "A",
                    }
                ],
            }
        ],
        metric_normalization_table=[],
    )

    assert rows == []


def test_investor_rows_skip_unknown_subjects():
    rows = _collect_investor_rows(
        chapter_evidence_packages=[
            {
                "chapter_id": "c1",
                "chapter_title": LONG_AI_CHAPTER_TITLE,
                "core_evidence": [
                    {
                        "fact": "\u878d\u8d44\u548c\u6295\u8d44\u4fe1\u53f7\u9700\u8981\u7ee7\u7eed\u89c2\u5bdf",
                        "source": {"title": "LOOK ~~~~~~~~~~~~~~~~~~~~~"},
                        "source_ref": "[1]",
                        "source_level": "A",
                    }
                ],
            }
        ],
        metric_normalization_table=[],
    )

    assert rows == []


def test_body_table_subject_does_not_fallback_to_dirty_source_title():
    item = {
        "fact": "\u7ade\u4e89\u683c\u5c40\u663e\u793a\u5934\u90e8\u4f01\u4e1a\u4efd\u989d\u53d8\u5316",
        "source": {"title": "LOOK ~~~~~~~~~~~~~~~~~~~~~"},
    }

    assert _subject(item) == ""


def test_table_rows_use_source_ref_and_drop_blank_subject_rows():
    item = {
        "fact": "\u7ade\u4e89\u683c\u5c40\u663e\u793a\u5934\u90e8\u4f01\u4e1a\u4efd\u989d\u53d8\u5316",
        "source": {"title": "LOOK ~~~~~~~~~~~~~~~~~~~~~"},
        "source_ref": "[7]",
    }
    row = _row_for_item(item, ["\u5bf9\u8c61/\u573a\u666f", "\u5173\u952e\u4e8b\u5b9e"])

    assert row["evidence_refs"] == ["[7]"]
    assert source_ref(item) == "[7]"
    assert _row_has_valid_leading_cell(["\u5bf9\u8c61/\u573a\u666f", "\u5173\u952e\u4e8b\u5b9e"], row) is False


def test_body_table_quality_count_uses_retained_rows_only():
    packages = run_table_agent(
        chapter_evidence_packages=[
            {
                "chapter_id": "c1",
                "chapter_title": "\u7ade\u4e89\u683c\u5c40",
                "chapter_question": "\u7ade\u4e89\u683c\u5c40\u5982\u4f55\u53d8\u5316",
                "table_evidence": [
                    {
                        "fact": "\u963f\u91cc\u4e91\u5728\u6a21\u578b\u670d\u52a1\u7ade\u4e89\u683c\u5c40\u4e2d\u6301\u7eed\u6295\u5165",
                        "subject": "\u963f\u91cc\u4e91",
                        "source_level": "A",
                        "source_ref": "[1]",
                    },
                    {
                        "fact": "\u534e\u4e3a\u4e91\u5728\u7b97\u529b\u548c\u653f\u4f01\u5ba2\u6237\u4fa7\u5f62\u6210\u7ade\u4e89\u4fe1\u53f7",
                        "subject": "\u534e\u4e3a\u4e91",
                        "source_level": "B",
                        "source_ref": "[2]",
                    },
                    {
                        "fact": "\u7ade\u4e89\u683c\u5c40\u663e\u793a\u5934\u90e8\u4f01\u4e1a\u4efd\u989d\u53d8\u5316",
                        "source": {"title": "LOOK ~~~~~~~~~~~~~~~~~"},
                        "source_level": "A",
                        "source_ref": "[3]",
                    },
                ],
            }
        ],
        micro_layouts=[
            {
                "chapter_id": "c1",
                "table_requests": [
                    {
                        "table_id": "t1",
                        "table_type": "player_matrix",
                        "title": "\u7ade\u4e89\u683c\u5c40",
                    }
                ],
            }
        ],
        analytics_outputs=[],
    )
    package = packages[0]

    assert len(package["rows"]) == 2
    assert package["high_quality_evidence_count"] == 2
    assert package["evidence_refs"] == ["[1]", "[2]"]


def test_cagr_sanity_filters_extreme_pairs_but_keeps_normal_pairs():
    bad_cagr = _derive_cagr(
        [
            {
                "kind": "market_size",
                "metric_name": "\u5e02\u573a\u89c4\u6a21",
                "scope": "\u5168\u7403",
                "unit": "USD",
                "year": 2025,
                "value_number": 250,
                "cagr_eligible": True,
                "evidence_refs": ["[1]"],
            },
            {
                "kind": "market_size",
                "metric_name": "\u5e02\u573a\u89c4\u6a21",
                "scope": "\u5168\u7403",
                "unit": "USD",
                "year": 2026,
                "value_number": 1587,
                "cagr_eligible": True,
                "evidence_refs": ["[2]"],
            },
        ]
    )
    good_cagr = _derive_cagr(
        [
            {
                "kind": "market_size",
                "metric_name": "\u5e02\u573a\u89c4\u6a21",
                "scope": "\u4e2d\u56fd",
                "unit": "RMB",
                "year": 2024,
                "value_number": 5784,
                "cagr_eligible": True,
                "evidence_refs": ["[1]"],
            },
            {
                "kind": "market_size",
                "metric_name": "\u5e02\u573a\u89c4\u6a21",
                "scope": "\u4e2d\u56fd",
                "unit": "RMB",
                "year": 2026,
                "value_number": 7000,
                "cagr_eligible": True,
                "evidence_refs": ["[2]"],
            },
        ]
    )

    assert bad_cagr == []
    assert len(good_cagr) == 1
    assert 0.09 < good_cagr[0]["result"] < 0.11


def test_market_analytics_keeps_refs_and_excludes_extreme_cagr_rows():
    out = run_market_analytics_agent(
        chapter_evidence_packages=[],
        metric_normalization_table=[
            {
                "metric_name": "\u5e02\u573a\u89c4\u6a21",
                "value": "250 \u4ebf\u7f8e\u5143",
                "scope": "\u5168\u7403",
                "period": "2025",
                "source_ref": "[1]",
                "source_level": "A",
            },
            {
                "metric_name": "\u5e02\u573a\u89c4\u6a21",
                "value": "1587 \u4ebf\u7f8e\u5143",
                "scope": "\u5168\u7403",
                "period": "2026",
                "source_ref": "[2]",
                "source_level": "A",
            },
            {
                "metric_name": "\u5e02\u573a\u89c4\u6a21",
                "value": "5784 \u4ebf\u5143",
                "scope": "\u4e2d\u56fd",
                "period": "2024",
                "source_ref": "[3]",
                "source_level": "A",
            },
            {
                "metric_name": "\u5e02\u573a\u89c4\u6a21",
                "value": "7000 \u4ebf\u5143",
                "scope": "\u4e2d\u56fd",
                "period": "2026",
                "source_ref": "[4]",
                "source_level": "A",
            },
        ],
    )
    cagr_tables = [table for table in out["tables"] if table["table_type"] == "cagr_calculation"]

    assert cagr_tables
    assert not any("1587" in str(row) and "250" in str(row) for row in cagr_tables[0]["rows"])
    assert any("10.0%" in str(row) for row in cagr_tables[0]["rows"])
    assert all(row["evidence_refs"] for table in out["tables"] for row in table["rows"])


def test_market_analytics_table_packages_validate_after_field_normalization(monkeypatch):
    monkeypatch.setenv("REPORT_MAX_BODY_TABLES_PER_CHAPTER", "2")
    monkeypatch.setenv("REPORT_MAX_BODY_TABLES", "6")
    analytics = run_market_analytics_agent(
        chapter_evidence_packages=[],
        report_blueprint={"chapters": [{"chapter_id": "ch_01"}]},
        metric_normalization_table=[
            {
                "metric_name": "\u5e02\u573a\u89c4\u6a21",
                "value": "100 \u4ebf\u5143",
                "scope": "\u4e2d\u56fd",
                "period": "2022",
                "unit": "\u4ebf\u5143",
                "source_ref": "[1]",
                "source_level": "A",
            },
            {
                "metric_name": "\u5e02\u573a\u89c4\u6a21",
                "value": "121 \u4ebf\u5143",
                "scope": "\u4e2d\u56fd",
                "period": "2024",
                "unit": "\u4ebf\u5143",
                "source_ref": "[2]",
                "source_level": "A",
            },
        ],
    )
    packages = run_table_agent(
        chapter_evidence_packages=[{"chapter_id": "ch_01", "chapter_title": "\u5e02\u573a"}],
        micro_layouts=[],
        analytics_outputs=[analytics],
    )

    by_type = {package["table_type"]: package for package in packages}
    assert by_type["market_metric_table"]["validation_errors"] == []
    assert by_type["cagr_calculation"]["validation_errors"] == []
    assert by_type["market_metric_table"]["should_render"] is True
    assert by_type["cagr_calculation"]["should_render"] is True
    assert by_type["market_metric_table"]["rows"][0]["metric"]
    assert by_type["market_metric_table"]["rows"][0]["source"]


def test_table_agent_emits_requirements_and_followups_when_metric_fields_missing():
    packages = run_table_agent(
        chapter_evidence_packages=[
            {
                "chapter_id": "ch_01",
                "chapter_title": "\u5e02\u573a",
                "chapter_question": "\u5e02\u573a\u89c4\u6a21\u5982\u4f55",
                "core_evidence": [],
            }
        ],
        micro_layouts=[
            {
                "chapter_id": "ch_01",
                "table_requests": [
                    {
                        "table_id": "t_missing",
                        "need_table": True,
                        "table_type": "market_metric_table",
                        "query": "\u4e2d\u56fd\u673a\u5668\u4eba\u5e02\u573a\u89c4\u6a21 CAGR",
                    }
                ],
            }
        ],
        analytics_outputs=[],
    )
    package = packages[0]
    requirement = package["table_evidence_requirements"][0]

    assert package["should_render"] is False
    assert {"metric", "value", "unit", "period", "source"}.issubset(set(requirement["missing_fields"]))
    assert package["table_follow_up_queries"][0]["target"] == "table_evidence_requirements"


def test_render_table_package_hides_internal_columns_and_uses_chinese_labels():
    markdown = render_table_package(
        {
            "should_render": True,
            "title": "\u5173\u952e\u6307\u6807",
            "headers": ["\u6307\u6807", "\u6570\u503c", "\u6765\u6e90\u7b49\u7ea7", "evidence_refs"],
            "rows": [
                {"cells": ["\u89c4\u6a21", "100", "A", "[1]"]},
                {"cells": ["\u589e\u901f", "10%", "A", "[2]"]},
            ],
            "takeaway": "\u6307\u6807\u53ef\u6bd4",
            "decision_implication": "\u53ef\u4ee5\u652f\u6491\u673a\u4f1a\u5224\u65ad",
            "limitations": ["\u9700\u6301\u7eed\u8ddf\u8e2a"],
            "evidence_refs": ["[1]", "[2]"],
        }
    )

    assert "\u6765\u6e90" not in markdown
    assert "evidence_refs" not in markdown
    assert "Decision implication" not in markdown
    assert "Boundary" not in markdown
    assert "\u5224\u65ad\u542b\u4e49\uff1a" in markdown
    assert "[1][2]" in markdown


def test_per_chapter_table_budget_keeps_highest_value_table_only():
    packages = run_table_agent(
        chapter_evidence_packages=[
            {
                "chapter_id": "c1",
                "chapter_title": "\u7ade\u4e89\u683c\u5c40",
                "chapter_question": "\u7ade\u4e89\u683c\u5c40\u5982\u4f55",
                "table_evidence": [
                    {"fact": "\u4f01\u4e1aA\u6269\u4ea7", "subject": "\u4f01\u4e1aA", "source_level": "A", "source_ref": "[1]"},
                    {"fact": "\u4f01\u4e1aB\u5ba2\u6237\u589e\u52a0", "subject": "\u4f01\u4e1aB", "source_level": "A", "source_ref": "[2]"},
                ],
            }
        ],
        micro_layouts=[
            {
                "chapter_id": "c1",
                "table_requests": [
                    {"table_id": "low", "table_type": "player_matrix", "title": "\u73a9\u5bb6"},
                    {"table_id": "high", "table_type": "competitor_matrix", "title": "\u7ade\u4e89"},
                ],
            }
        ],
        analytics_outputs=[],
    )

    rendered = [package for package in packages if package.get("should_render")]
    assert len(rendered) == 1
    assert rendered[0]["table_id"] == "high"


def test_appendix_rows_render_as_appendix_detail():
    appendix = render_appendix(
        [],
        {
            "table_appendix_rows": [
                {
                    "title": "\u5173\u952e\u6307\u6807",
                    "headers": ["\u6307\u6807", "\u6570\u503c", "\u6765\u6e90"],
                    "rows": [["\u89c4\u6a21", "100", "[1]"], ["\u589e\u901f", "10%", "[2]"]],
                }
            ]
        },
    )

    assert "\u5173\u952e\u6307\u6807\uff08\u9644\u5f55\u660e\u7ec6\uff09" in appendix
    assert "|\u6765\u6e90|" not in appendix.replace(" ", "")
    assert "100" in appendix


def test_qa_flags_body_source_headers_and_table_fatigue():
    qa = run_qa_agent(
        report_markdown=(
            "# \u62a5\u544a\n\n"
            "## \u6b63\u6587\n"
            "| \u6307\u6807 | \u6765\u6e90\u7b49\u7ea7 |\n"
            "| --- | --- |\n"
            "| \u89c4\u6a21 | A |\n"
        ),
        report_blueprint={"report_family": "industry_report"},
        chapter_packages=[
            {
                "chapter_id": "ch_01",
                "sections": [
                    {
                        "claim": "\u5e02\u573a\u89c4\u6a21\u53ef\u89c2",
                        "reasoning": "\u56e0\u4e3a\u6307\u6807\u6709\u589e\u957f",
                        "counter_evidence": "\u9700\u8ddf\u8e2a\u53cd\u5411\u4fe1\u53f7",
                        "actionable": "\u7ee7\u7eed\u76d1\u6d4b",
                        "evidence_refs": ["[1]"],
                    }
                ],
            }
        ],
        table_packages=[],
        decision_package={"decision_items": ["track"]},
        risk_package={"risk_items": ["risk"]},
    )

    assert any(error["type"] == "body_table_contains_source_header" for error in qa["errors"])
    assert qa["clean_format"]["has_body_source_table_header"] is True

    row_value_ok = run_qa_agent(
        report_markdown=(
            "# \u62a5\u544a\n\n"
            "## \u6b63\u6587\n"
            "| \u6307\u6807 | \u5224\u65ad\u542b\u4e49 |\n"
            "| --- | --- |\n"
            "| \u89c4\u6a21 | \u6765\u6e90\u4e8e\u5b98\u65b9\u7edf\u8ba1\u7684\u53ef\u6bd4\u6570\u636e |\n"
        ),
        report_blueprint={"report_family": "industry_report"},
        chapter_packages=[
            {
                "chapter_id": "ch_01",
                "sections": [
                    {
                        "claim": "\u5e02\u573a\u89c4\u6a21\u53ef\u89c2",
                        "reasoning": "\u56e0\u4e3a\u6307\u6807\u6709\u589e\u957f",
                        "counter_evidence": "\u9700\u8ddf\u8e2a\u53cd\u5411\u4fe1\u53f7",
                        "actionable": "\u7ee7\u7eed\u76d1\u6d4b",
                        "evidence_refs": ["[1]"],
                    }
                ],
            }
        ],
        table_packages=[],
        decision_package={"decision_items": ["track"]},
        risk_package={"risk_items": ["risk"]},
    )
    assert row_value_ok["clean_format"]["has_body_source_table_header"] is False


def test_credibility_title_semantics_do_not_upgrade_media_repost_to_ab():
    level = _infer_credibility(
        "https://m.thepaper.cn/newsDetail_forward_1",
        "\u4e2d\u56fd\u4fe1\u901a\u9662\u53d1\u5e032024\u5e74\u4eba\u5de5\u667a\u80fd\u4ea7\u4e1a\u767d\u76ae\u4e66",
    )
    low_level = _infer_credibility(
        "https://wenku.baidu.com/view/1",
        "\u4e2d\u56fd\u4fe1\u901a\u9662\u53d1\u5e032024\u5e74\u4eba\u5de5\u667a\u80fd\u4ea7\u4e1a\u767d\u76ae\u4e66",
    )

    assert level == "C"
    assert low_level == "D"


def test_schema_like_bullets_are_blocked_by_extractor_and_reformatter():
    schema_line = "\u5e02\u573a\u89c4\u6a21\uff1b\u4e2d\u56fd\uff1b2024-2026"
    markdown = (
        "# \u6d4b\u8bd5\u62a5\u544a\n\n"
        "## \u5173\u952e\u6570\u636e\n"
        f"- {schema_line}\n\n"
        "## \u6b63\u6587\n"
        "\u8fd9\u91cc\u6709\u8db3\u591f\u591a\u7684\u6b63\u6587\u5185\u5bb9\u7528\u4e8e\u89e6\u53d1\u6821\u9a8c\u3002"
        + ("\u6b63\u6587\u5185\u5bb9\u3002" * 120)
        + "\n\n## \u7814\u7a76\u53e3\u5f84\u4e0e\u6765\u6e90\n"
        "[1] \u4e2d\u56fd\u4fe1\u901a\u9662\u62a5\u544a\n"
    )

    validation = validate_reformatted_report(
        markdown,
        [{"ref": "[1]", "title": "\u4e2d\u56fd\u4fe1\u901a\u9662\u62a5\u544a"}],
        {"sources": [{"ref": "[1]", "title": "\u4e2d\u56fd\u4fe1\u901a\u9662\u62a5\u544a"}]},
    )

    assert _is_meaningful_fact(schema_line) is False
    assert validation["has_sources_appendix"] is True
    assert validation["schema_like_bullets"]


def test_reformatter_body_target_adapts_to_sparse_evidence(monkeypatch):
    monkeypatch.setenv("REPORT_REFORMATTER_MIN_BODY_CHARS", "18000")
    monkeypatch.setenv("REPORT_REFORMATTER_LENGTH_TARGET_MODE", "adaptive")
    monkeypatch.setenv("REPORT_REFORMATTER_FULL_LENGTH_MIN_FACTS", "30")
    monkeypatch.setenv("REPORT_REFORMATTER_MIN_BODY_CHARS_FLOOR", "6000")
    monkeypatch.setenv("REPORT_REFORMATTER_CHARS_PER_FACT_TARGET", "900")
    monkeypatch.setenv("REPORT_REFORMATTER_SPARSE_EVIDENCE_OVERHEAD_CHARS", "3000")
    sparse = {
        "dimensions": {
            "\u4e8b\u4ef6\u4e0e\u4f01\u4e1a\u52a8\u4f5c": [
                {"text": f"\u53ef\u6838\u9a8c\u4e8b\u5b9e{i}", "source": str(i)}
                for i in range(1, 6)
            ]
        }
    }
    rich = {
        "dimensions": {
            "\u4e8b\u4ef6\u4e0e\u4f01\u4e1a\u52a8\u4f5c": [
                {"text": f"\u53ef\u6838\u9a8c\u4e8b\u5b9e{i}", "source": str(i)}
                for i in range(1, 31)
            ]
        }
    }

    assert _target_body_chars(sparse) == 7500
    assert _target_body_chars(rich) == 18000

    monkeypatch.setenv("REPORT_REFORMATTER_LENGTH_TARGET_MODE", "fixed")
    assert _target_body_chars(sparse) == 18000


def test_reformatter_auto_expands_length_only_failure(monkeypatch):
    monkeypatch.setenv("REPORT_REFORMATTER_MIN_BODY_CHARS", "2500")
    monkeypatch.setenv("REPORT_REFORMATTER_LENGTH_TARGET_MODE", "fixed")
    monkeypatch.setenv("REPORT_REFORMATTER_AUTO_EXPAND_ANALYSIS", "true")
    monkeypatch.setenv("REPORT_REFORMATTER_AUTO_EXPAND_MIN_FACTS", "8")
    monkeypatch.setenv("REPORT_REFORMATTER_AUTO_EXPAND_MAX_CHARS", "5000")
    sources = [{"id": str(i), "title": f"\u6765\u6e90{i}", "url": "https://example.com"} for i in range(1, 17)]
    dimensions = {}
    for dim_idx, dimension in enumerate(
        [
            "\u4f01\u4e1a\u8bbf\u95ee\u4e0e\u4f9b\u5e94\u94fe\u52a8\u4f5c",
            "\u653f\u7b56\u4e0e\u5408\u89c4\u8fb9\u754c",
            "\u5ba2\u6237\u9700\u6c42\u4e0e\u8ba2\u5355\u4fe1\u53f7",
            "\u8d44\u672c\u5f00\u652f\u4e0e\u6280\u672f\u8def\u5f84",
        ],
        start=0,
    ):
        dimensions[dimension] = [
            {
                "text": (
                    f"{dimension}\u4e2d\u7684\u53ef\u6838\u9a8c\u4e8b\u5b9e{j}"
                    "\u8868\u660e\u4f01\u4e1a\u52a8\u4f5c\u548c\u4ea7\u4e1a\u9884\u671f\u6b63\u5728\u53d8\u5316"
                ),
                "source": str(dim_idx * 4 + j),
                "source_quality": "high",
            }
            for j in range(1, 5)
        ]
    clean_evidence = {"dimensions": dimensions, "sources": sources}
    markdown = clean_reformatted_report(
        "# \u6d4b\u8bd5\u62a5\u544a\n\n"
        "## \u6838\u5fc3\u5224\u65ad\n"
        "\u73b0\u6709\u6750\u6599\u53ef\u4ee5\u652f\u6491\u65b9\u5411\u6027\u5224\u65ad[1]\u3002\n",
        sources,
    )
    validation = validate_reformatted_report(markdown, sources, clean_evidence)

    expanded = clean_reformatted_report(
        _auto_expand_analysis_for_length(markdown, clean_evidence, validation, sources),
        sources,
    )
    expanded_validation = validate_reformatted_report(expanded, sources, clean_evidence)

    assert "\u8bc1\u636e\u94fe\u7684\u8fde\u7eed\u9a8c\u8bc1\u4e0e\u7ed3\u8bba\u8fb9\u754c" in expanded
    assert "\u54ea\u4e9b\u53d8\u91cf\u771f\u6b63\u6539\u53d8\u5224\u65ad" not in expanded
    assert expanded_validation["body_chars_without_sources"] > validation["body_chars_without_sources"]
    assert not expanded_validation["repeated_boilerplate_issues"]
    assert not expanded_validation["paragraph_length_issues"]


def test_reformatter_score_mode_allows_minor_soft_validation_issues(monkeypatch):
    monkeypatch.setenv("REPORT_REFORMATTER_VALIDATION_MODE", "score")
    monkeypatch.setenv("REPORT_REFORMATTER_MIN_PASS_SCORE", "60")
    monkeypatch.setenv("REPORT_REFORMATTER_LENGTH_TARGET_MODE", "fixed")
    monkeypatch.setenv("REPORT_REFORMATTER_MIN_BODY_CHARS", "1200")
    sources = [{"id": "1", "title": "\u6743\u5a01\u6765\u6e90", "url": "https://example.com"}]
    supporting_body = "\n\n".join(
        f"\u7b2c{i}\u7ec4\u5206\u6790\u53ea\u7528\u6765\u8865\u8db3\u6b63\u6587\u5c55\u5f00\uff0c\u4e0d\u989d\u5916\u5f15\u5165\u65b0\u4e8b\u5b9e\u5224\u65ad[1]\u3002"
        for i in range(1, 48)
    )
    markdown = (
        "# \u6d4b\u8bd5\u62a5\u544a\n\n"
        "## \u4e3b\u4f53\u5206\u6790\n"
        "2026\u5e74\u51fa\u73b0\u7684\u4f01\u4e1a\u52a8\u4f5c\u8bf4\u660e\u9700\u8981\u8fdb\u4e00\u6b65\u5206\u6790[1]\u3002\n\n"
        f"{supporting_body}\n\n"
        "## \u6682\u65e0\u5c55\u5f00\n"
    )
    validation = validate_reformatted_report(
        markdown,
        sources,
        {
            "dimensions": {
                "\u4e3b\u4f53\u5206\u6790": [
                    {"text": "\u4f01\u4e1a\u52a8\u4f5c\u9700\u8981\u7ee7\u7eed\u9a8c\u8bc1", "source": "1"}
                ]
            },
            "sources": sources,
        },
    )

    assert validation["passed"] is True
    assert validation["hard_pass"] is False
    assert validation["empty_section_count"] >= 1
    assert validation["quality_score"] >= validation["minimum_pass_score"]
    assert _reformatter_needs_repair(validation) is True


def test_reformatter_can_disable_soft_repair_after_score_pass(monkeypatch):
    monkeypatch.setenv("REPORT_REFORMATTER_REPAIR_SOFT_ISSUES", "false")
    validation = {
        "passed": True,
        "hard_pass": False,
        "soft_issue_count": 2,
        "quality_score": 80,
    }

    assert _reformatter_needs_repair(validation) is False


def test_public_sanitizer_does_not_reintroduce_forbidden_template_openers():
    markdown = (
        "# \u62a5\u544a\n\n"
        "\u5df2\u62ab\u9732\u7684\u5173\u952e\u4e8b\u5b9e\u5305\u62ec\uff1a\u6837\u672c\u4e2d\u7684\u6750\u6599\u663e\u793a\u9700\u6c42\u6b63\u5728\u53d8\u5316\u3002"
    )
    cleaned = sanitize_public_markdown(markdown)

    assert "\u516c\u5f00\u4fe1\u606f\u663e\u793a" not in cleaned


def test_failed_reformatter_does_not_publish_writer_fallback_as_clean_report():
    source = inspect.getsource(full_report.main)

    assert "write_markdown(clean_output_path, fallback_report)" not in source
    assert '"fallback_draft_path"' in source
    assert '"fallback_draft_written"' in source
    assert "reformatter_blocked_clean" in source
    assert "final_stdout_allowed" in source
    assert "REPORT_REVIEW_REQUIRED_EXIT_NONZERO" in source
    assert "REPORT_REFORMATTER_FAILURE_EXIT_NONZERO" in source
    assert "attach_structured_review_to_writer_report" in source
    assert "review_evidence_required" in source
    assert "review_rewrite_required" in source


def test_writer_final_gate_allows_balanced_advisory_followups():
    qa = {
        "passed": True,
        "repair_required": False,
        "blocking_followups": [],
        "advisory_followups": [
            {"type": "missing_proof_standard", "suggested_query": "official data 2026"}
        ],
        "deep_evaluation": {
            "required_followups": [
                {"type": "missing_proof_standard", "suggested_query": "official data 2026"}
            ]
        },
        "evidence_repair_followups": [{"type": "missing_proof_standard"}],
        "advisory_evidence_repair_followups": [{"type": "missing_proof_standard"}],
    }

    assert _qa_has_pending_repair(qa) is False
    assert _writer_ready_for_final(
        markdown="# Report\n\nBody with enough public text.",
        qa_result=qa,
        package_passed=True,
        package_warning_blocked=False,
    ) is True


def test_writer_final_gate_blocks_hard_qa_followups():
    qa = {
        "passed": True,
        "repair_required": True,
        "blocking_followups": [{"type": "missing_proof_standard"}],
        "blocking_evidence_repair_followups": [{"type": "missing_proof_standard"}],
        "advisory_followups": [],
        "deep_evaluation": {"required_followups": [{"type": "missing_proof_standard"}]},
    }

    assert _qa_has_pending_repair(qa) is True
    assert _writer_ready_for_final(
        markdown="# Report\n\nBody with enough public text.",
        qa_result=qa,
        package_passed=True,
        package_warning_blocked=False,
    ) is False


def test_full_report_publish_gate_allows_advisory_qa_followups():
    writer_report = {
        "report_status": "final",
        "report_markdown": "# Report\n\nBody",
        "qa_result": {
            "passed": True,
            "repair_required": False,
            "blocking_followups": [],
            "advisory_followups": [{"type": "report_body_below_target_chars"}],
            "deep_evaluation": {"required_followups": [{"type": "report_body_below_target_chars"}]},
        },
    }

    reasons = full_report.writer_report_pending_repair_reasons(writer_report)

    assert reasons == []


def test_full_report_blocker_summary_keeps_advisories_nonblocking():
    writer_report = {
        "report_status": "final",
        "qa_result": {
            "passed": True,
            "repair_required": False,
            "blocking_followups": [],
            "advisory_followups": [{"type": "report_body_below_target_chars"}],
        },
    }

    blocker_summary = full_report.build_qa_blocker_summary(
        writer_report=writer_report,
        evidence_gap_summary={"status": "has_gaps", "chapter_gaps": [{"chapter_id": "ch_01"}]},
        review_diagnostic={"checks": {"table_validation_warnings": 1}},
        reformatter_result={"status": "skipped"},
        writer_pending_repair_reasons=[],
    )

    assert blocker_summary["status"] == "advisory"
    assert blocker_summary["blocker_types"] == []
    assert {"evidence_gap", "table_validation_warnings", "qa_advisory_followups"}.issubset(
        set(blocker_summary["advisory_types"])
    )
    assert (
        full_report.clean_report_blocked_reason(
            writer_publishable=True,
            writer_not_ready=False,
            reformatter_skip_reason="",
            qa_blocker_summary=blocker_summary,
        )
        == ""
    )


def test_full_report_publish_gate_blocks_hard_qa_repair():
    writer_report = {
        "report_status": "final",
        "report_markdown": "# Report\n\nBody",
        "qa_result": {
            "passed": True,
            "repair_required": True,
            "blocking_followups": [{"type": "report_body_below_target_chars"}],
            "blocking_content_repair_followups": [{"type": "report_body_below_target_chars"}],
            "deep_evaluation": {"required_followups": [{"type": "report_body_below_target_chars"}]},
        },
    }

    reasons = full_report.writer_report_pending_repair_reasons(writer_report)

    assert "repair_required" in reasons
    assert "blocking_followups" in reasons


def test_review_pending_repair_reasons_route_evidence_to_quality_gate():
    review_result = {
        "structured_review": {
            "status": "needs_evidence",
            "evidence_required": True,
            "evidence_followups": [{"type": "invalid_citation", "source_id": "99"}],
            "citation_issues": [{"type": "invalid_citation", "source_id": "99"}],
        }
    }

    reasons = full_report.review_result_pending_repair_reasons(review_result)
    gate = full_report.quality_gate_state(
        writer_status="final",
        writer_not_ready=False,
        writer_pending_repair_reasons=reasons,
        reformatter_result={"status": "skipped", "skipped_reason": "review_evidence_required"},
    )

    assert reasons == ["review_evidence_required"]
    assert gate["next_action"] == "evidence_refinement"
    assert gate["evidence_required"] is True
    assert gate["publishable"] is False


def test_review_pending_repair_reasons_route_rewrite_to_quality_gate():
    review_result = {
        "structured_review": {
            "status": "needs_rewrite",
            "rewrite_required": True,
            "logic_issues": [{"type": "duplicate_paragraph", "line": 12}],
        }
    }

    reasons = full_report.review_result_pending_repair_reasons(review_result)
    gate = full_report.quality_gate_state(
        writer_status="final",
        writer_not_ready=False,
        writer_pending_repair_reasons=reasons,
        reformatter_result={"status": "skipped", "skipped_reason": "review_rewrite_required"},
    )

    assert reasons == ["review_rewrite_required"]
    assert gate["next_action"] == "rewrite"
    assert gate["rewrite_required"] is True
    assert gate["publishable"] is False


def test_empty_review_result_does_not_add_publish_gate_reasons():
    writer_report = {
        "report_status": "final",
        "report_markdown": "# Report\n\nBody",
        "qa_result": {"passed": True},
    }

    assert full_report.review_result_pending_repair_reasons({}) == []
    assert full_report.merge_writer_review_pending_repair_reasons(writer_report, {}) == []


def test_structured_review_fields_attach_to_writer_report():
    writer_report = {"report_status": "final"}
    review_result = {
        "structured_review": {
            "status": "needs_evidence",
            "evidence_followups": [{"type": "uncited_inference", "line": 8}],
            "logic_issues": [{"type": "uncited_inference", "line": 8}],
            "citation_issues": [{"type": "invalid_citation", "source_id": "7"}],
        }
    }

    updated = full_report.attach_structured_review_to_writer_report(writer_report, review_result)

    assert updated["review_status"] == "needs_evidence"
    assert updated["review_evidence_followups"][0]["type"] == "uncited_inference"
    assert updated["review_logic_issues"][0]["line"] == 8
    assert updated["review_citation_issues"][0]["source_id"] == "7"


def test_iqs_topic_seed_preserves_named_executives():
    query = (
        "\u6709\u9650\u518d\u8fde\u63a5\uff1a\u9a6c\u65af\u514b\u3001"
        "\u5e93\u514b\u3001\u9ec4\u4ec1\u52cb\u968f\u8bbf\u80cc\u540e"
        "\u7684\u4e2d\u7f8e\u79d1\u6280\u65b0\u5747\u8861"
    )
    chapter = {
        "chapter_id": "ch_01",
        "chapter_title": query + "\u662f\u5426\u5b58\u5728\u771f\u5b9e\u9700\u6c42",
        "core_question": query + "\u662f\u5426\u5b58\u5728\u771f\u5b9e\u9700\u6c42",
        "required_evidence_mix": ["official_data", "market_research", "company_filing"],
    }
    goal = {
        "goal_id": "H1_metric",
        "proof_role": "metric",
        "question": query + "\uff1a\u8865\u9f50\u6307\u6807\u53e3\u5f84",
        "must_have_terms": ["\u9700\u6c42\u589e\u901f", "\u91c7\u8d2d\u4e3b\u4f53"],
        "expected_metrics": ["\u8ba2\u5355", "\u91c7\u8d2d\u4e3b\u4f53"],
        "source_priority": ["\u5b98\u65b9", "\u516c\u544a"],
    }
    plan = {
        "query": query,
        "research_object": "\u4e2d\u7f8e\u79d1\u6280\u4ea7\u4e1a\u4e92\u52a8\u6a21\u5f0f",
        "global_forbidden_terms": [],
        "global_required_terms": ["2024", "2025", "2026"],
    }

    task = build_search_tasks_for_goal(chapter=chapter, goal=goal, research_plan=plan)[0]

    assert _topic_seed_terms(query, chapter, goal)[:3] == [
        "\u9a6c\u65af\u514b",
        "\u5e93\u514b",
        "\u9ec4\u4ec1\u52cb",
    ]
    assert "\u9a6c\u65af\u514b" in task["query"]
    assert "\u5e93\u514b" in task["query"]
    assert "\u9ec4\u4ec1\u52cb" in task["query"]


def test_initial_iqs_lane_budget_caps_heavy_full_report_options(monkeypatch):
    monkeypatch.setenv("REPORT_QUALITY_MODE", "balanced")
    for name in [
        "BRAIN_INITIAL_LANE_MAX_QUERIES",
        "BRAIN_INITIAL_LANE_MAX_SEARCH_TASKS",
        "BRAIN_INITIAL_LANE_RESULTS_PER_QUERY",
        "BRAIN_INITIAL_LANE_RERANK_TOP_K",
        "BRAIN_INITIAL_LANE_RERANK_MAX_DOCS",
        "BRAIN_INITIAL_LANE_RERANK_PREFILTER_MAX_DOCS",
    ]:
        monkeypatch.delenv(name, raising=False)
    state = {
        "web_search_options": {
            "search_profile": "initial",
            "max_queries": 6,
            "max_search_tasks": 24,
            "results_per_query": 80,
            "rerank_top_k": 40,
            "rerank_max_docs": 100,
            "rerank_prefilter_max_docs": 100,
            "enable_self_refine": True,
        },
        "query_analysis": {"research_plan": {}},
    }
    task = {
        "query": "\u9a6c\u65af\u514b \u5e93\u514b \u9ec4\u4ec1\u52cb \u5b98\u65b9 \u516c\u544a",
        "proof_role": "metric",
    }

    options = _search_options_for_task(state, task, "initial")

    assert options["max_queries"] == 4
    assert options["max_search_tasks"] == 10
    assert options["results_per_query"] == 50
    assert options["rerank_top_k"] == 18
    assert options["enable_self_refine"] is False


def test_analysis_agent_outputs_evidence_gaps_and_refinement_plan():
    package = {
        "query": "\u4e2d\u7f8e\u79d1\u6280\u4ea7\u4e1a\u4e92\u52a8",
        "research_plan": {
            "hypotheses": [
                {
                    "hypothesis_id": "H1",
                    "statement": "\u9ad8\u7ba1\u4e92\u8bbf\u6b63\u5728\u6539\u53d8\u4f9b\u5e94\u94fe\u9884\u671f",
                    "counter_evidence_required": True,
                }
            ],
            "evidence_coverage_requirements": {
                "per_hypothesis": {
                    "min_A_or_B_sources": 2,
                    "min_counter_sources": 1,
                    "min_metric_sources": 1,
                }
            },
        },
        "clean_evidence_list": [
            {
                "evidence_id": "EV-1",
                "dimension": "\u4f9b\u5e94\u94fe\u9884\u671f",
                "fact": "\u6743\u5a01\u6765\u6e90\u63d0\u5230\u4f01\u4e1a\u6b63\u5728\u8bc4\u4f30\u4f9b\u5e94\u94fe\u8c03\u6574",
                "source_level": "A",
                "source": {"title": "\u6743\u5a01\u62a5\u544a"},
                "evidence_role": "core",
                "allowed_use": "core_claim",
                "hypothesis_id": "H1",
                "metric": "\u4f9b\u5e94\u94fe\u8c03\u6574",
                "value": "\u5df2\u62ab\u9732",
            }
        ],
    }

    structured = run_analysis_agent(package)["structured_analysis"]
    evidence = structured["evidence_analyses"][0]
    synthesis = structured["dimension_synthesis"]["\u4f9b\u5e94\u94fe\u9884\u671f"]
    plan = structured["evidence_refinement_plan"]

    assert evidence["analysis_depth"]["strength"] == "strong"
    assert "metric_period_missing" in evidence["evidence_gaps"]
    assert synthesis["mechanism"]
    assert synthesis["counter"]
    assert plan["status"] == "needs_refinement"
    assert plan["follow_up_queries"]


def test_evidence_binder_exports_refinement_plan_for_followup_loop():
    research_plan = {
        "hypotheses": [
            {
                "hypothesis_id": "H1",
                "statement": "\u8bbf\u95ee\u6d3b\u52a8\u4f1a\u5e26\u6765\u4f9b\u5e94\u94fe\u8c03\u6574",
                "counter_evidence_required": True,
            }
        ],
        "evidence_coverage_requirements": {
            "per_hypothesis": {
                "min_A_or_B_sources": 2,
                "min_counter_sources": 1,
                "min_metric_sources": 1,
            }
        },
    }
    result = run_evidence_binder(
        research_plan=research_plan,
        report_blueprint={
            "chapters": [
                {
                    "chapter_id": "c1",
                    "chapter_title": "\u4f9b\u5e94\u94fe\u8c03\u6574",
                    "chapter_question": "\u8bbf\u95ee\u6d3b\u52a8\u662f\u5426\u6539\u53d8\u4f9b\u5e94\u94fe",
                }
            ]
        },
        evidence_pool=[
            {
                "fact": "\u6743\u5a01\u6765\u6e90\u63d0\u5230\u4f01\u4e1a\u6b63\u5728\u8bc4\u4f30\u4f9b\u5e94\u94fe\u8c03\u6574",
                "source": {"title": "\u6743\u5a01\u62a5\u544a", "url": "https://www.gov.cn/test", "credibility": "A"},
                "dimension": "\u4f9b\u5e94\u94fe\u8c03\u6574",
                "hypothesis_id": "H1",
                "hypothesis_statement": "\u8bbf\u95ee\u6d3b\u52a8\u4f1a\u5e26\u6765\u4f9b\u5e94\u94fe\u8c03\u6574",
                "evidence_role": "core",
                "proof_role": "support",
                "metric": "\u8c03\u6574\u52a8\u4f5c",
                "value": "\u5df2\u62ab\u9732",
                "confidence": 0.9,
            }
        ],
    )
    plan = result["evidence_refinement_plan"]

    assert plan["status"] == "needs_refinement"
    assert plan["follow_up_queries"]
    assert any(
        "insufficient_ab_sources" in item.get("blocking_gaps", [])
        or "counter_evidence_missing" in item.get("blocking_gaps", [])
        for item in plan["follow_up_queries"]
    )


def _minimal_publishable_qa_payload():
    markdown = (
        "# Test Report\n\n"
        "## Chapter One\n"
        "This chapter explains the market mechanism because demand signals drive supply responses. [1]\n"
    )
    section = {
        "section_id": "s1",
        "claim": "Demand signals can support a directional judgment.",
        "reasoning": "Because demand signals drive supplier behavior, the evidence can support a cautious directional conclusion.",
        "counter_evidence": "If demand weakens or policy execution changes, the conclusion should be downgraded.",
        "actionable": "Track official disclosures and company filings.",
        "evidence_refs": ["[1]"],
    }
    return {
        "report_markdown": markdown,
        "report_blueprint": {"report_family": "industry_report"},
        "chapter_packages": [
            {
                "chapter_id": "c1",
                "chapter_title": "Chapter One",
                "sections": [section],
            }
        ],
        "table_packages": [],
        "decision_package": {"report_family": "industry_report", "decision_items": ["Track official disclosures"]},
        "risk_package": {"risk_items": ["Policy execution may change"]},
        "package_quality_report": {
            "passed": True,
            "errors": [],
            "blocking_errors": [],
            "warnings": [{"type": "missing_chapter_role"} for _ in range(20)],
        },
        "search_task_schedule": {"dropped_count": 3},
        "lane_coverage": {"iqs_lane_1": {"scheduled": 3, "succeeded": 1, "failed": 2}},
        "metric_normalization_table": [],
        "analytics_outputs": [],
        "coverage_matrix": [
            {
                "decision_ready": False,
                "actual_ab_sources": 0,
                "directional_c_sources": 1,
                "evidence_refs": ["[1]"],
                "blocking_gaps": ["insufficient_ab_sources", "metric_scope_period_unit_incomplete"],
            }
        ],
        "missing_proof_standards": [
            {
                "hypothesis_id": "H1",
                "hypothesis_statement": "Directional claim",
                "blocking_gaps": ["insufficient_ab_sources", "metric_scope_period_unit_incomplete"],
            }
        ],
    }


def test_balanced_qa_does_not_block_body_length_or_soft_evidence_gaps(monkeypatch):
    monkeypatch.setenv("REPORT_QUALITY_MODE", "balanced")
    monkeypatch.setenv("QA_DEEP_EVALUATOR_BLOCKING", "false")
    monkeypatch.setenv("REPORT_TARGET_BODY_CHARS_BLOCKING", "false")
    monkeypatch.setenv("QA_MIN_PASS_SCORE", "65")
    monkeypatch.setenv("QA_WARNING_PENALTY_EACH", "2")
    monkeypatch.setenv("QA_WARNING_PENALTY_CAP", "12")
    payload = _minimal_publishable_qa_payload()

    qa = run_qa_agent(**payload)

    assert qa["passed"] is True
    assert qa["deep_evaluator_blocking"] is False
    assert qa["quality_score"] >= 65
    assert qa["repair_required"] is False
    assert qa["blocking_followups"] == []
    assert any(item.get("type") == "report_body_below_target_chars" for item in qa["deep_evaluation"]["required_followups"])
    assert any(item.get("type") == "missing_proof_standard" for item in qa["deep_evaluation"]["required_followups"])
    assert any(item.get("type") == "report_body_below_target_chars" for item in qa["advisory_followups"])
    assert any(item.get("type") == "missing_proof_standard" for item in qa["advisory_followups"])


def test_balanced_qa_treats_section_reasoning_fields_as_soft(monkeypatch):
    monkeypatch.setenv("REPORT_QUALITY_MODE", "balanced")
    monkeypatch.setenv("QA_DEEP_EVALUATOR_BLOCKING", "false")
    monkeypatch.setenv("REPORT_TARGET_BODY_CHARS_BLOCKING", "false")
    payload = _minimal_publishable_qa_payload()
    payload["chapter_packages"][0]["sections"][0].pop("reasoning")
    payload["chapter_packages"][0]["sections"][0].pop("counter_evidence")
    payload["chapter_packages"][0]["sections"][0].pop("actionable")

    qa = run_qa_agent(**payload)

    assert qa["passed"] is True
    assert not any(item.get("type") == "argument_unit_incomplete" for item in qa["errors"])
    assert any(item.get("type") == "argument_unit_soft_missing_fields" for item in qa["warnings"])


def test_balanced_qa_score_mode_does_not_hard_block_nonfatal_errors(monkeypatch):
    monkeypatch.setenv("REPORT_QUALITY_MODE", "balanced")
    monkeypatch.setenv("QA_SCORING_MODE", "score")
    monkeypatch.setenv("QA_DEEP_EVALUATOR_BLOCKING", "false")
    monkeypatch.setenv("REPORT_TARGET_BODY_CHARS_BLOCKING", "false")
    monkeypatch.setenv("QA_MIN_PASS_SCORE", "60")
    payload = _minimal_publishable_qa_payload()
    payload["chapter_packages"][0]["sections"][0]["evidence_refs"] = []

    qa = run_qa_agent(**payload)

    assert any(item.get("type") == "argument_unit_missing_evidence_refs" for item in qa["errors"])
    assert qa["fatal_errors"] == []
    assert qa["soft_errors"]
    assert qa["passed"] is True
    assert qa["repair_required"] is False
    assert qa["rewrite_required"] is False
    assert qa["blocking_followups"] == []


def test_evidence_binder_creates_mandatory_proof_followups_for_tech_geopolitics():
    research_plan = {
        "report_mode": "deep_industry_report",
        "report_family": "industry_report",
        "hypotheses": [
            {
                "hypothesis_id": "H1",
                "statement": "\u9a6c\u65af\u514b\u3001\u5e93\u514b\u3001\u9ec4\u4ec1\u52cb\u968f\u8bbf\u4f53\u73b0\u4e2d\u7f8e\u79d1\u6280\u6709\u9650\u518d\u8fde\u63a5\uff0c\u4f46\u82f1\u4f1f\u8fbe\u51fa\u53e3\u7ba1\u5236\u3001\u82f9\u679c\u4e2d\u56fd\u6536\u5165\u548c\u7279\u65af\u62c9\u4e0a\u6d77\u5de5\u5382\u9700\u8981\u786c\u6570\u636e\u9a8c\u8bc1",
                "metric_definitions": [
                    {"metric_name": "NVIDIA China data center revenue"},
                    {"metric_name": "Apple Greater China net sales"},
                    {"metric_name": "Tesla Shanghai delivery/export"},
                ],
            }
        ],
        "evidence_goals": [
            {
                "hypothesis_id": "H1",
                "goal_id": "g1",
                "question": "\u9a8c\u8bc1 NVIDIA Apple Tesla \u4e2d\u56fd\u76f8\u5173\u786c\u6570\u636e",
                "min_sources": 1,
            }
        ],
    }
    report_blueprint = {
        "research_object": "\u6709\u9650\u518d\u8fde\u63a5\uff1a\u9a6c\u65af\u514b\u3001\u5e93\u514b\u3001\u9ec4\u4ec1\u52cb\u968f\u8bbf\u80cc\u540e\u7684\u4e2d\u7f8e\u79d1\u6280\u65b0\u5747\u8861",
        "narrative": "\u4e2d\u7f8e\u79d1\u6280\u6709\u9650\u518d\u8fde\u63a5",
        "chapters": [
            {
                "chapter_id": "c1",
                "chapter_title": "\u516c\u53f8\u4e2d\u56fd\u5229\u76ca\u4e0e\u653f\u7b56\u8fb9\u754c",
                "chapter_question": "\u8fd9\u4e9b\u7f8e\u56fd\u79d1\u6280\u5de8\u5934\u7684\u4e2d\u56fd\u5229\u76ca\u662f\u5426\u88ab\u786c\u6570\u636e\u652f\u6491",
            }
        ],
    }
    result = run_evidence_binder(
        research_plan=research_plan,
        report_blueprint=report_blueprint,
        evidence_pool=[
            {
                "hypothesis_id": "H1",
                "fact": "\u6709\u5a92\u4f53\u79f0\u9a6c\u65af\u514b\u3001\u5e93\u514b\u548c\u9ec4\u4ec1\u52cb\u7684\u968f\u8bbf\u53cd\u6620\u7f8e\u56fd\u79d1\u6280\u4f01\u4e1a\u5bf9\u4e2d\u56fd\u5e02\u573a\u4ecd\u6709\u5174\u8da3\uff0c\u4f46\u672a\u62ab\u9732 NVIDIA\u3001Apple \u6216 Tesla \u7684\u4e2d\u56fd\u8ba2\u5355\u3001\u6536\u5165\u6216\u4ea7\u91cf\u6570\u636e\u3002",
                "source": {
                    "title": "\u9a6c\u65af\u514b\u5e93\u514b\u9ec4\u4ec1\u52cb\u968f\u8bbf\u89c2\u5bdf",
                    "url": "https://www.thepaper.cn/newsDetail_forward_123",
                },
                "source_level": "C",
                "confidence": 0.8,
            }
        ],
    )

    assert result["research_proof_profile"]["profile_id"] == "tech_geopolitics"
    assert any("mandatory_proof_missing" in row.get("blocking_gaps", []) for row in result["coverage_matrix"])
    queries = " ".join(item.get("query", "") for item in result["evidence_refinement_plan"]["follow_up_queries"])
    assert "NVIDIA" in queries
    assert "Apple" in queries
    assert "Tesla" in queries


def test_mandatory_proof_requires_strong_source_to_clear_gap():
    profile = select_research_proof_profile(query="NVIDIA H20 China export license")

    weak_checks = mandatory_proof_checks(
        profile,
        [
            {
                "ref": "EV-C",
                "source_level": "C",
                "source_family": "news/secondary",
                "fact": "NVIDIA H20 export license China data center revenue charge was mentioned by a news article.",
                "source": {"title": "media article", "url": "https://example.com/news"},
            }
        ],
    )
    nvidia_weak = next(item for item in weak_checks if item["proof_id"] == "nvidia_export_control_status")

    strong_checks = mandatory_proof_checks(
        profile,
        [
            {
                "ref": "EV-A",
                "source_level": "A",
                "source_family": "official/filing",
                "fact": "NVIDIA H20 export license China data center revenue charge was disclosed in a filing.",
                "source": {"title": "NVIDIA 8-K filing", "url": "https://www.sec.gov/Archives/example"},
            }
        ],
    )
    nvidia_strong = next(item for item in strong_checks if item["proof_id"] == "nvidia_export_control_status")

    assert nvidia_weak["status"] == "weak_found"
    assert nvidia_weak["matched_refs"] == []
    assert nvidia_weak["weak_matched_refs"] == ["EV-C"]
    assert nvidia_strong["status"] == "found"
    assert nvidia_strong["matched_refs"] == ["EV-A"]


def test_binder_uses_report_profile_for_coverage_rows_even_when_plan_is_generic():
    result = run_evidence_binder(
        research_plan={
            "report_mode": "deep_industry_report",
            "hypotheses": [
                {
                    "hypothesis_id": "H1",
                    "statement": "\u6838\u5fc3\u5224\u65ad\u9700\u8981\u786c\u8bc1\u636e\u9a8c\u8bc1",
                }
            ],
        },
        report_blueprint={
            "research_object": "\u6709\u9650\u518d\u8fde\u63a5\uff1a\u9a6c\u65af\u514b\u3001\u5e93\u514b\u3001\u9ec4\u4ec1\u52cb\u968f\u8bbf\u80cc\u540e\u7684\u4e2d\u7f8e\u79d1\u6280\u65b0\u5747\u8861",
            "chapters": [{"chapter_id": "c1", "chapter_title": "\u4e2d\u7f8e\u79d1\u6280\u65b0\u5747\u8861"}],
        },
        evidence_pool=[],
    )

    assert result["research_proof_profile"]["profile_id"] == "tech_geopolitics"
    assert result["coverage_matrix"][0]["proof_profile_id"] == "tech_geopolitics"


def test_qa_reports_research_maturity_and_mandatory_proof_followups(monkeypatch):
    monkeypatch.setenv("REPORT_QUALITY_MODE", "balanced")
    monkeypatch.setenv("QA_SCORING_MODE", "score")
    monkeypatch.setenv("QA_DEEP_EVALUATOR_BLOCKING", "false")
    monkeypatch.setenv("REPORT_TARGET_BODY_CHARS_BLOCKING", "false")
    payload = _minimal_publishable_qa_payload()
    payload["coverage_matrix"][0]["mandatory_proof_checks"] = [
        {
            "proof_id": "nvidia_export_control_status",
            "label": "NVIDIA export-control and China license status",
            "status": "missing",
            "severity": "high",
            "required": True,
            "query": "NVIDIA H20 H200 Blackwell China export license BIS 8-K 10-K data center revenue",
            "lane_targets": ["filing_company", "official_data"],
            "source_priority": ["filing", "official"],
        }
    ]

    qa = run_qa_agent(**payload)

    assert qa["passed"] is True
    assert qa["research_maturity"]["level"] == "framework_draft"
    assert qa["deep_evaluation"]["coverage_summary"]["mandatory_proof_missing"] == 1
    assert any(item.get("type") == "mandatory_proof_missing" for item in qa["repair_followups"])
    assert qa["repair_required"] is False
    assert any(item.get("type") == "mandatory_proof_missing" for item in qa["advisory_evidence_repair_followups"])


def test_evidence_only_qa_followups_do_not_force_rewrite(monkeypatch):
    monkeypatch.setenv("REPORT_QUALITY_MODE", "balanced")
    monkeypatch.setenv("QA_SCORING_MODE", "score")
    monkeypatch.setenv("QA_DEEP_EVALUATOR_BLOCKING", "false")
    monkeypatch.setenv("REPORT_TARGET_BODY_CHARS", "0")
    monkeypatch.setenv("QA_REPAIR_WARNINGS", "false")
    payload = _minimal_publishable_qa_payload()
    payload["search_task_schedule"] = {}
    payload["lane_coverage"] = {}
    payload["coverage_matrix"][0]["blocking_gaps"] = ["mandatory_proof_missing"]
    payload["coverage_matrix"][0]["mandatory_proof_checks"] = [
        {
            "proof_id": "apple_china_sales_supply_chain",
            "label": "Apple Greater China sales and China supply-chain dependence",
            "status": "missing",
            "severity": "high",
            "required": True,
            "query": "Apple 2025 10-K Greater China net sales supply chain China suppliers",
            "lane_targets": ["filing_company", "official_data"],
            "source_priority": ["filing", "official"],
        }
    ]
    payload["missing_proof_standards"] = []

    qa = run_qa_agent(**payload)

    assert qa["passed"] is True
    assert qa["repair_required"] is False
    assert qa["rewrite_required"] is False
    assert qa["content_repair_followups"] == []
    assert qa["blocking_followups"] == []
    assert any(item.get("type") == "mandatory_proof_missing" for item in qa["evidence_repair_followups"])
    assert any(item.get("type") == "mandatory_proof_missing" for item in qa["advisory_evidence_repair_followups"])


def test_writer_report_mandatory_proof_followups_feed_brain_layout_loop():
    writer_report = {
        "qa_result": {
            "deep_evaluation": {
                "required_followups": [
                    {
                        "type": "mandatory_proof_missing",
                        "proof_id": "tesla_china_factory_sales",
                        "label": "Tesla Shanghai production, export, sales and margin signal",
                        "suggested_query": "Tesla 2025 10-K Shanghai factory China deliveries export gross margin",
                        "lane_targets": ["filing_company", "official_data"],
                        "source_priority": ["filing", "official"],
                    }
                ]
            }
        }
    }

    followups = _layout_followup_queries_from_writer_report(writer_report, max_queries=3)

    assert followups
    assert followups[0]["query"].startswith("Tesla 2025 10-K")
    assert followups[0]["lane_targets"] == ["filing_company", "official_data"]
    assert followups[0]["mandatory_proof_id"] == "tesla_china_factory_sales"


def test_mandatory_proof_followups_use_proof_id_as_target_key():
    writer_report = {
        "qa_result": {
            "deep_evaluation": {
                "required_followups": [
                    {
                        "type": "mandatory_proof_missing",
                        "proof_id": proof_id,
                        "mandatory_proof_id": proof_id,
                        "label": proof_id,
                        "suggested_query": f"{proof_id} query",
                        "lane_targets": ["filing_company"],
                        "source_priority": ["filing"],
                        "blocking_gaps": ["mandatory_proof_missing"],
                    }
                    for proof_id in [
                        "nvidia_export_control_status",
                        "apple_china_sales_supply_chain",
                        "tesla_china_factory_sales",
                        "policy_counter_trigger",
                    ]
                ]
            }
        }
    }

    followups = _layout_followup_queries_from_writer_report(writer_report, max_queries=10)

    assert [item.get("mandatory_proof_id") for item in followups] == [
        "apple_china_sales_supply_chain",
        "nvidia_export_control_status",
        "policy_counter_trigger",
        "tesla_china_factory_sales",
    ]


def test_followup_task_dedupe_prefers_specific_iqs_lane_over_generic_iqs():
    tasks = [
        {
            "query": "Apple foldable iPhone suppliers official data",
            "agent": "iqs",
            "targets_gap": "supplier proof",
            "search_task": {"gap_id": "apple_supplier_proof", "query": "Apple foldable iPhone suppliers official data"},
        },
        {
            "query": "Apple foldable iPhone suppliers official data",
            "agent": "iqs_lane_1",
            "targets_gap": "supplier proof",
            "search_task": {"gap_id": "apple_supplier_proof", "query": "Apple foldable iPhone suppliers official data"},
        },
        {
            "query": "Apple foldable iPhone suppliers official data",
            "agent": "rag",
            "targets_gap": "supplier proof",
            "search_task": {"gap_id": "apple_supplier_proof", "query": "Apple foldable iPhone suppliers official data"},
        },
    ]

    deduped = _dedupe_followup_tasks(tasks)

    assert [item["agent"] for item in deduped] == ["iqs_lane_1", "rag"]


def test_followup_query_key_keeps_specific_iqs_lanes_distinct():
    lane_1 = {
        "query": "Apple foldable iPhone supplier proof",
        "agent": "iqs_lane_1",
        "gap_id": "apple_supplier_proof",
        "targets_gap": "supplier proof",
    }
    lane_2 = {**lane_1, "agent": "iqs_lane_2"}

    assert _followup_query_key(lane_1) != _followup_query_key(lane_2)


def test_followup_signal_requires_substantive_evidence():
    weak_result = {
        "status": "success",
        "confidence": 0.0,
        "answer": "没有找到可确认的信息。",
        "key_sources": [],
        "raw_data_points": [],
    }
    strong_result = {
        "status": "success",
        "confidence": 0.42,
        "answer": "Company filing includes supplier exposure and project timing.",
        "key_sources": [{"title": "10-K", "url": "https://example.com"}],
        "raw_data_points": [],
    }

    assert _followup_result_has_signal([weak_result]) is False
    assert _followup_result_has_signal([strong_result]) is True


def test_gap_ledger_tracks_followup_signal_by_gap():
    followups = [
        {"gap_id": "gap_a", "query": "official source A", "targets_gap": "A"},
        {"gap_id": "gap_b", "query": "official source B", "targets_gap": "B"},
    ]
    results = [
        {
            "status": "success",
            "confidence": 0.5,
            "answer": "Useful sourced answer",
            "key_sources": [{"title": "Official", "url": "https://example.com/a"}],
            "search_task": {"gap_id": "gap_a"},
        },
        {
            "status": "success",
            "confidence": 0.0,
            "answer": "没有找到可确认的信息。",
            "key_sources": [],
            "raw_data_points": [],
            "search_task": {"gap_id": "gap_b"},
        },
    ]

    ledger = {item["gap_id"]: item for item in _gap_ledger_from_followups(followups, results)}

    assert ledger["gap_a"]["status"] == "evidence_found"
    assert ledger["gap_b"]["status"] == "searched_no_signal"


def test_repair_result_summary_counts_only_substantive_followups():
    empty_success = {
        "status": "success",
        "confidence": 0.95,
        "answer": "",
        "key_sources": [],
        "raw_data_points": [],
    }
    sourced_result = {
        "status": "success",
        "answer": "Company filing discloses foldable hinge validation progress.",
        "key_sources": [{"title": "Company filing", "url": "https://example.com/filing"}],
        "raw_data_points": [],
    }

    usable = _substantive_followup_results([empty_success, sourced_result])
    summary = _repair_result_summary([empty_success, sourced_result], usable_results=usable)

    assert usable == [sourced_result]
    assert summary["signal_count"] == 1
    assert summary["empty_success_count"] == 1
    assert summary["new_usable_evidence_count"] == 1
    assert summary["new_ab_source_count"] == 1


def test_repair_tasks_skip_rewrite_only_followups():
    tasks, skipped = _repair_tasks_from_items(
        [
            {"type": "report_body_below_target_chars", "query": "expand report body"},
            {
                "type": "insufficient_ab_sources",
                "hypothesis_id": "H1",
                "blocking_gaps": ["insufficient_ab_sources"],
                "query": "foldable hinge official filing source",
            },
        ],
        origin_node="unit",
        loop_name="evidence_preflight",
        max_tasks=4,
    )

    assert skipped == 1
    assert len(tasks) == 1
    assert tasks[0]["origin_node"] == "unit"
    assert tasks[0]["loop_name"] == "evidence_preflight"
    assert tasks[0]["gap_id"]


def test_repair_seen_keys_can_be_shared_across_repair_loops():
    state = {}
    seen = _repair_seen_keys_for_state(state)

    first_tasks, first_skipped = _repair_tasks_from_items(
        [
            {
                "type": "insufficient_ab_sources",
                "hypothesis_id": "H1",
                "blocking_gaps": ["insufficient_ab_sources"],
                "query": "enterprise agent official deployment source",
            }
        ],
        origin_node="coverage_evaluation",
        loop_name="supervisor_coverage",
        max_tasks=2,
        seen_keys=seen,
    )
    second_tasks, second_skipped = _repair_tasks_from_items(
        [
            {
                "type": "insufficient_ab_sources",
                "hypothesis_id": "H1",
                "blocking_gaps": ["insufficient_ab_sources"],
                "query": "enterprise agent official deployment source",
            }
        ],
        origin_node="writer_qa",
        loop_name="post_qa_repair",
        max_tasks=2,
        seen_keys=_repair_seen_keys_for_state(state),
    )

    assert len(first_tasks) == 1
    assert first_skipped == 0
    assert second_tasks == []
    assert second_skipped == 1
    assert seen is _repair_seen_keys_for_state(state)


def test_research_reflection_memo_seeds_flow_into_layout_repair_tasks(monkeypatch):
    monkeypatch.setenv("BRAIN_ENABLE_POST_QA_REPAIR", "true")
    writer_report = {
        "report_status": "draft",
        "research_reflection_memo": {
            "schema_version": "research_reflection_memo_v1",
            "next_search_task_seeds": [
                {
                    "schema_version": "repair_task_seed_v2",
                    "query": "enterprise AI agent adoption official metric unit period source",
                    "gap_id": "GAP-metric",
                    "requirement_id": "H1_metric",
                    "chapter_id": "ch_01",
                    "section_id": "sec_01",
                    "gap_type": "metric_scope_period_unit_incomplete",
                    "repair_status": "still_insufficient",
                    "proof_role": "metric",
                    "required_fields": ["metric", "value", "unit", "period", "source"],
                    "required_source_level": ["A", "B"],
                    "lane_targets": ["official_data"],
                    "success_criteria": "metric/value/unit/period/source must all be traceable",
                    "reject_if": ["snippet_only", "no_date", "no_source_url"],
                    "preferred_source_patterns": ["official_data", "market_research"],
                    "freshness_required": True,
                    "max_cache_age_hours": 24,
                    "source": "research_reflection_memo",
                    "allowed_for_writing": False,
                    "avoid_repeating_failed_query": True,
                }
            ],
        },
    }

    assert _post_qa_repair_needed(writer_report) is True

    followups = _layout_followup_queries_from_writer_report(writer_report, max_queries=4)
    assert len(followups) == 1
    followup = followups[0]
    assert followup["gap_id"] == "GAP-metric"
    assert followup["requirement_id"] == "H1_metric"
    assert followup["required_fields"] == ["metric", "value", "unit", "period", "source"]
    assert followup["reject_if"] == ["snippet_only", "no_date", "no_source_url"]
    assert followup["success_criteria"] == "metric/value/unit/period/source must all be traceable"
    assert followup["source"] == "research_reflection_memo"

    tasks, skipped = _repair_tasks_from_items(
        followups,
        origin_node="writer_qa",
        loop_name="post_qa_repair",
        max_tasks=4,
    )

    assert skipped == 0
    assert len(tasks) == 1
    task = tasks[0]
    assert task["gap_id"] == "GAP-metric"
    assert task["requirement_id"] == "H1_metric"
    assert task["proof_role"] == "metric"
    assert task["required_fields"] == ["metric", "value", "unit", "period", "source"]
    assert task["preferred_source_patterns"] == ["official_data", "market_research"]
    assert task["allowed_for_writing"] is False


def test_evidence_merger_filters_empty_followup_results_from_analysis_pool():
    empty_success = {
        "status": "success",
        "confidence": 0.95,
        "answer": "",
        "key_sources": [],
        "raw_data_points": [],
    }
    data_result = {
        "status": "success",
        "answer": "",
        "key_sources": [{"title": "Official data", "url": "https://example.com/data"}],
        "raw_data_points": [
            {
                "metric": "yield rate",
                "value": "82%",
                "period": "2025",
                "source_url": "https://example.com/data",
                "evidence": "Official data reports 2025 yield rate at 82%.",
            }
        ],
    }

    package = merge_evidence_package(
        original_query="foldable display bottleneck",
        evidence_pool=[empty_success, data_result],
        children={},
        research_plan={},
    )

    assert package["metadata"]["source_pool_filtered_count"] == 1
    assert package["summary"]["evidence_count"] >= 1
    assert package["summary"]["ready_for_analysis"] in {True, False}
    assert package["evidence_filter_funnel_by_chapter"]


def test_loop_health_summary_exposes_stop_reason_and_signal_counts():
    summary = build_loop_health_summary(
        supervisor_trace=[
            {
                "stop_reason": "coverage_exhausted_with_gaps",
                "attempted_task_count": 2,
                "repair_result_summary": {"signal_count": 0, "empty_success_count": 1, "failed_count": 1},
            }
        ],
        evidence_preflight_trace=[
            {
                "status": "no_signal",
                "attempted_task_count": 1,
                "repair_result_summary": {"signal_count": 0, "new_usable_evidence_count": 0},
            }
        ],
        layout_refinement_trace=[],
        post_qa_repair_trace=[],
    )

    assert summary["supervisor_coverage"]["stop_reason"] == "coverage_exhausted_with_gaps"
    assert summary["supervisor_coverage"]["signal_count"] == 0
    assert summary["evidence_preflight"]["status"] == "no_signal"


def test_deep_repair_policy_marks_only_evidence_repair_tasks(monkeypatch):
    monkeypatch.setenv("IQS_DEEP_REPAIR_ENABLED", "true")
    state = {"metadata": {}}
    evidence_task = brain_agent_module.normalize_search_task(
        {
            "query": "foldable hinge official filing source",
            "agent": "iqs",
            "gap_id": "gap_ab",
            "targets_gap": "H1",
            "blocking_gaps": ["insufficient_ab_sources"],
            "proof_role": "source_check",
            "loop_name": "evidence_preflight",
        },
        fallback_index=1,
    )
    rewrite_task = brain_agent_module.normalize_search_task(
        {
            "query": "expand report body",
            "agent": "iqs",
            "gap_id": "length",
            "targets_gap": "length",
            "type": "report_body_below_target_chars",
        },
        fallback_index=2,
    )

    marked = brain_agent_module._apply_deep_repair_policy_to_tasks(
        [
            {"query": evidence_task["query"], "agent": "iqs", "targets_gap": "H1", "search_task": evidence_task},
            {"query": rewrite_task["query"], "agent": "iqs", "targets_gap": "length", "search_task": rewrite_task},
        ],
        state=state,
        round_number=1,
    )

    assert marked[0]["search_task"]["prefer_deep"] is True
    assert marked[0]["search_task"]["engineTypes"][0] == "Deep"
    assert marked[1]["search_task"].get("prefer_deep") is not True
    summary = brain_agent_module._repair_task_summary_after_policy(
        [{"query": evidence_task["query"], "agent": "iqs", "targets_gap": "H1", "search_task": evidence_task}],
        [{"status": "failed", "search_task": marked[0]["search_task"]}],
    )
    assert summary["pre_deep_task_count"] == 0
    assert summary["post_deep_task_count"] == 1


def test_deep_repair_policy_enforces_round_budget(monkeypatch):
    monkeypatch.setenv("IQS_DEEP_REPAIR_MAX_TASKS_PER_ROUND", "1")
    state = {"metadata": {}}
    tasks = []
    for index in range(3):
        search_task = brain_agent_module.normalize_search_task(
            {
                "query": f"official source {index}",
                "agent": "iqs",
                "gap_id": f"gap_{index}",
                "blocking_gaps": ["mandatory_proof_missing"],
                "proof_role": "source_check",
                "loop_name": "post_qa_repair",
            },
            fallback_index=index + 1,
        )
        tasks.append({"query": search_task["query"], "agent": "iqs", "targets_gap": "gap", "search_task": search_task})

    marked = brain_agent_module._apply_deep_repair_policy_to_tasks(tasks, state=state, round_number=1)

    assert sum(1 for item in marked if item["search_task"].get("prefer_deep")) == 1
    assert any(item["search_task"].get("deep_skip_reason") == "round_budget_exhausted" for item in marked)


def test_web_iqs_deep_options_and_fallback_chain(monkeypatch):
    monkeypatch.setenv("IQS_DEEP_REPAIR_TIMEOUT_MS", "60000")
    monkeypatch.setenv("IQS_DEEP_REPAIR_NUM_RESULTS", "10")
    calls = []

    def fake_search(query, options):
        calls.append(options["engineType"])
        if options["engineType"] == "Deep":
            return []
        return [{"title": "Official", "url": "https://example.com", "snippet": "source"}]

    options = web_analysis_agent_module._options_for_query_item(
        {"text": "foldable hinge source", "engineType": "Deep"},
        {"prefer_deep": True, "deep_reason": "unit"},
    )

    assert options["engineType"] == "Deep"
    assert options["timeout"] >= 60000
    assert 1 <= options["numResults"] <= 50

    monkeypatch.setattr(web_analysis_agent_module, "call_iqs_search", fake_search)
    results, trace = web_analysis_agent_module.call_iqs_search_with_fallback(
        {"text": "foldable hinge source", "engineType": "Deep"},
        {"prefer_deep": True, "deep_reason": "unit"},
    )

    assert calls[:2] == ["Deep", "LiteAdvanced"]
    assert trace["fallback_used"] is True
    assert trace["primary_engine"] == "Deep"
    assert results


def test_post_qa_repair_plan_prioritizes_missing_proofs_over_dropped_tasks():
    writer_report = {
        "report_status": "review_required",
        "qa_pending_repair": True,
        "required_followups": [
            {"type": "search_tasks_dropped", "suggested_query": "补齐被截断 search tasks"},
            {
                "type": "missing_proof_standard",
                "hypothesis_id": "H1",
                "hypothesis_statement": "需求是否真实",
                "blocking_gaps": ["metric_scope_period_unit_incomplete"],
            },
            {
                "type": "missing_proof_standard",
                "hypothesis_id": "H2",
                "hypothesis_statement": "订单是否真实",
                "blocking_gaps": ["insufficient_ab_sources"],
            },
        ],
    }

    plan = _post_qa_repair_plan(writer_report, max_queries=2)

    assert [item["hypothesis_id"] for item in plan["evidence_followups"]] == ["H1", "H2"]
    assert plan["skipped_duplicate_followups"] >= 1
    assert "官方" in plan["evidence_followups"][0]["source_priority"]


def test_post_qa_repair_plan_routes_body_length_to_rewrite_only():
    writer_report = {
        "report_status": "review_required",
        "qa_pending_repair": True,
        "qa_result": {
            "content_repair_followups": [{"type": "report_body_below_target_chars", "required": 20000, "actual": 8000}],
            "deep_evaluation": {
                "required_followups": [
                    {"type": "report_body_below_target_chars", "required": 20000, "actual": 8000}
                ]
            },
        },
    }

    plan = _post_qa_repair_plan(writer_report, max_queries=4)

    assert plan["evidence_followups"] == []
    assert plan["rewrite_required"] is True
    assert any(item["type"] == "report_body_below_target_chars" for item in plan["rewrite_reasons"])


def test_post_qa_repair_round_with_signal_reruns_writer(monkeypatch):
    calls = {"followup": 0, "writer": 0}

    def fake_followups(**kwargs):
        calls["followup"] += 1
        return [
            {
                "status": "success",
                "agent": "iqs",
                "query": "official source",
                "answer": "Official filing confirms orders and metric period.",
                "confidence": 0.8,
                "key_sources": [{"title": "Official filing", "url": "https://example.com"}],
            }
        ]

    def fake_merge(**kwargs):
        return {"metadata": {}, "chapter_evidence_packages": []}

    def fake_analysis(package, query=""):
        return {"structured_analysis": {"analysis": "ok"}, "raw_output": {"analysis": {"source": "test"}}}

    def fake_writer(**kwargs):
        calls["writer"] += 1
        assert kwargs["structured_analysis"]["post_qa_repair_context"]["evidence_followups"]
        return {
            "writer_report": {
                "report_status": "final",
                "report_markdown": "# fixed\n\n正文",
                "qa_result": {"passed": True},
            }
        }

    monkeypatch.setattr(brain_agent_module, "run_followup_queries", fake_followups)
    monkeypatch.setattr(brain_agent_module, "merge_evidence_package", fake_merge)
    monkeypatch.setattr(brain_agent_module, "run_analysis_agent", fake_analysis)
    monkeypatch.setattr(brain_agent_module, "run_writer_agent", fake_writer)
    monkeypatch.setattr(brain_agent_module, "_attach_reformatter_preflight_feedback", lambda **kwargs: kwargs["writer_report"])

    result = _run_post_qa_repair_round(
        state={"query": "test"},
        children={},
        best={
            "writer_report": {
                "report_status": "review_required",
                "qa_pending_repair": True,
                "required_followups": [
                    {"type": "missing_proof_standard", "hypothesis_id": "H1", "hypothesis_statement": "需求验证"}
                ],
            },
            "evidence_pool": [],
            "evidence_package": {},
            "structured_analysis": {},
            "analysis_state": {},
            "layout_refinement_trace": [],
        },
        report_plan={},
        query="test",
        search_task_schedule={},
        lane_coverage={},
        max_followups=4,
        started=0.0,
    )

    assert calls == {"followup": 1, "writer": 1}
    assert result["writer_report"]["report_status"] == "final"
    assert result["post_qa_repair_trace"][0]["status"] == "completed"
    assert result["writer_report"]["post_qa_repair"]["is_best"] is True


def test_post_qa_repair_round_no_signal_does_not_rerun_writer(monkeypatch):
    def fake_followups(**kwargs):
        return [{"status": "failed", "agent": "iqs", "query": "official source", "answer": "", "confidence": 0.0}]

    def fail_writer(**kwargs):
        raise AssertionError("writer should not run when post-QA evidence repair has no signal")

    monkeypatch.setattr(brain_agent_module, "run_followup_queries", fake_followups)
    monkeypatch.setattr(brain_agent_module, "run_writer_agent", fail_writer)

    result = _run_post_qa_repair_round(
        state={"query": "test"},
        children={},
        best={
            "writer_report": {
                "report_status": "review_required",
                "qa_pending_repair": True,
                "required_followups": [
                    {"type": "missing_proof_standard", "hypothesis_id": "H1", "hypothesis_statement": "需求验证"}
                ],
            },
            "evidence_pool": [],
            "evidence_package": {},
            "structured_analysis": {},
            "analysis_state": {},
            "layout_refinement_trace": [],
        },
        report_plan={},
        query="test",
        search_task_schedule={},
        lane_coverage={},
        max_followups=4,
        started=0.0,
    )

    assert result["writer_report"]["report_status"] == "review_required"
    assert result["post_qa_repair_trace"][0]["status"] == "no_new_evidence_signal"
    assert result["writer_report"]["post_qa_repair"]["has_signal"] is False


def test_post_qa_repair_round_rewrite_only_skips_followup(monkeypatch):
    def fail_followups(**kwargs):
        raise AssertionError("rewrite-only post-QA repair should not search")

    def fake_writer(**kwargs):
        assert kwargs["structured_analysis"]["post_qa_repair_context"]["rewrite_required"] is True
        return {
            "writer_report": {
                "report_status": "final",
                "report_markdown": "# rewritten\n\n正文扩写",
                "qa_result": {"passed": True},
            }
        }

    monkeypatch.setattr(brain_agent_module, "run_followup_queries", fail_followups)
    monkeypatch.setattr(brain_agent_module, "run_writer_agent", fake_writer)
    monkeypatch.setattr(brain_agent_module, "_attach_reformatter_preflight_feedback", lambda **kwargs: kwargs["writer_report"])

    result = _run_post_qa_repair_round(
        state={"query": "test"},
        children={},
        best={
            "writer_report": {
                "report_status": "review_required",
                "qa_pending_repair": True,
                "qa_result": {
                    "content_repair_followups": [
                        {"type": "report_body_below_target_chars", "required": 20000, "actual": 8000}
                    ]
                },
            },
            "evidence_pool": [],
            "evidence_package": {},
            "structured_analysis": {},
            "analysis_state": {},
            "layout_refinement_trace": [],
        },
        report_plan={},
        query="test",
        search_task_schedule={},
        lane_coverage={},
        max_followups=4,
        started=0.0,
    )

    assert result["writer_report"]["report_status"] == "final"
    assert result["post_qa_repair_trace"][0]["plan"]["rewrite_required"] is True
    assert result["post_qa_repair_trace"][0]["has_signal"] is None


def test_post_qa_repair_trace_is_exported_from_brain_state():
    source = inspect.getsource(brain_agent_module.merge_outputs_node)

    assert "post_qa_repair_trace" in source
    assert "post_qa_repair_rounds" in source


def test_strict_qa_still_blocks_body_length(monkeypatch):
    monkeypatch.setenv("REPORT_QUALITY_MODE", "strict")
    monkeypatch.setenv("REPORT_TARGET_BODY_CHARS_BLOCKING", "false")
    payload = _minimal_publishable_qa_payload()

    qa = run_qa_agent(**payload)

    assert qa["passed"] is False
    assert qa["repair_required"] is True
    assert qa["deep_evaluator_blocking"] is True
    assert any(item.get("type") == "report_body_below_target_chars" for item in qa["blocking_followups"])
    assert any(
        item.get("type") == "deep_report_blocking_gap"
        and item.get("detail", {}).get("type") == "report_body_below_target_chars"
        for item in qa["errors"]
    )


def test_delivery_gate_allows_limited_review_draft_for_partial_evidence(monkeypatch):
    monkeypatch.setenv("REPORT_DELIVERY_POLICY", "three_tier")
    monkeypatch.setenv("REPORT_REQUIRE_READPAGE_EVIDENCE", "true")
    monkeypatch.setenv("REPORT_MIN_CORE_AB_SOURCES_PER_CHAPTER", "3")
    package = merge_evidence_package(
        original_query="AI Agent industry report",
        evidence_pool=[
            {
                "status": "success",
                "answer": "IDC report says AI agent adoption is rising in 2025.",
                "key_sources": [
                    {
                        "title": "IDC AI report",
                        "url": "https://example.com/idc",
                        "source_type": "research",
                        "source_level": "B",
                    }
                ],
                "raw_data_points": [
                    {
                        "chapter_id": "ch_1",
                        "chapter_title": "Industry overview",
                        "metric": "adoption",
                        "value": "rising",
                        "period": "2025",
                        "source_url": "https://example.com/idc",
                        "source_type": "research",
                        "evidence": "IDC report says AI agent adoption is rising in 2025.",
                    }
                ],
                "page_results": [{"url": "https://example.com/idc", "content": "report body", "auto_readpage": True}],
                "metadata": {"auto_readpage": {"attempted": 1, "succeeded": 1}},
                "search_task": {"chapter_id": "ch_1", "proof_role": "metric"},
            }
        ],
        children={},
        research_plan={"chapter_structure": [{"chapter_id": "ch_1", "chapter_title": "Industry overview"}]},
    )

    gate = package["summary"]["delivery_gate"]

    assert gate["tier"] == "limited_review_draft"
    assert gate["draft_allowed"] is True
    assert gate["publishable"] is False


def test_delivery_gate_marks_empty_package_diagnostic_only(monkeypatch):
    monkeypatch.setenv("REPORT_DELIVERY_POLICY", "three_tier")
    package = merge_evidence_package(
        original_query="AI Agent industry report",
        evidence_pool=[],
        children={},
        research_plan={"chapter_structure": [{"chapter_id": "ch_1", "chapter_title": "Industry overview"}]},
    )

    gate = package["summary"]["delivery_gate"]

    assert gate["tier"] == "diagnostic_only"
    assert gate["diagnostic_only"] is True
    assert gate["draft_allowed"] is False


def test_delivery_gate_publishable_clean_when_publishable_gate_passes():
    gate = evidence_merger_module._delivery_gate(
        publishable_gate={"passed": True, "blocking_reasons": []},
        evidence_count=5,
        clean_fact_count=5,
        analysis_ready_count=5,
        analysis_ready_ab_count=3,
        source_distribution={"A": 1, "B": 2},
        readpage_coverage={"attempted": 2, "succeeded": 2},
        core_ab_by_chapter={"ch_1": 3},
        evidence_gap_ledger=[],
    )

    assert gate["tier"] == "publishable_clean"
    assert gate["publishable"] is True


def test_gap_fuse_stops_after_consecutive_no_gain(monkeypatch):
    monkeypatch.setenv("BRAIN_GAP_MAX_ATTEMPTS", "3")
    monkeypatch.setenv("BRAIN_GAP_MAX_NO_GAIN_ROUNDS", "2")
    state = {"metadata": {}}
    tasks = [
        {
            "query": "official source",
            "agent": "iqs",
            "targets_gap": "gap A",
            "search_task": {"query": "official source", "gap_id": "gap_a", "proof_role": "source_check"},
        }
    ]
    for round_number in (1, 2):
        selected = brain_agent_module._filter_exhausted_gap_tasks(tasks, state=state)
        brain_agent_module._record_gap_attempts(selected, state=state, round_number=round_number)
        brain_agent_module._record_gap_attempt_results(
            [
                {
                    "status": "success",
                    "answer": "no source",
                    "key_sources": [],
                    "raw_data_points": [],
                    "search_task": selected[0]["search_task"],
                }
            ],
            state=state,
        )

    selected = brain_agent_module._filter_exhausted_gap_tasks(tasks, state=state)
    summary = state["metadata"]["gap_attempt_summary"]

    assert selected == []
    assert summary["evidence_exhausted"] is True
    assert summary["gaps"]["gap_a"]["exhausted"] is True
