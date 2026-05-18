import inspect

from rag_pipeline.agents.analytics.competitor_analysis_agent import _collect_rows
from rag_pipeline.agents.analytics.evidence_utils import evidence_subject, is_valid_entity_subject, source_ref
from rag_pipeline.agents.analytics.investor_insight_agent import _collect_rows as _collect_investor_rows
from rag_pipeline.agents.analytics.market_analytics_agent import _derive_cagr, run_market_analytics_agent
from rag_pipeline.agents.analysis_agent import run_analysis_agent
from rag_pipeline.agents.brain_agent import (
    _layout_followup_queries_from_writer_report,
    _search_options_for_task,
    _topic_seed_terms,
    _writer_quality_key,
    build_search_tasks_for_goal,
)
from rag_pipeline.agents.evidence_binder import run_evidence_binder
from rag_pipeline.agents.final_writer_agent import run_final_writer_agent
from rag_pipeline.agents.qa_agent import run_qa_agent
from rag_pipeline.agents.problem_framing_agent import run_problem_framing_agent
from rag_pipeline.agents.research_proof_registry import mandatory_proof_checks, select_research_proof_profile
from rag_pipeline.agents.table_agent import _row_for_item, _row_has_valid_leading_cell, _subject, run_table_agent
from rag_pipeline.agents.public_report_sanitizer import sanitize_public_markdown
from rag_pipeline.flows.report import full_report
from rag_pipeline.flows.report.evidence_extractor import _infer_credibility, _is_meaningful_fact, extract_clean_evidence_from_package
from rag_pipeline.flows.report.reformatter_agent import (
    _auto_expand_analysis_for_length,
    _reformatter_needs_repair,
    _target_body_chars,
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


def test_reformatter_drops_source_appendix_by_default(monkeypatch):
    monkeypatch.delenv("REPORT_REFORMATTER_SOURCE_APPENDIX", raising=False)
    markdown = (
        "# 测试报告\n\n"
        "## 正文分析\n"
        "正文事实需要保留正文内引用[1]。\n\n"
        "## 数据来源列表\n"
        "[1] Acme source\n"
    )

    cleaned = clean_reformatted_report(markdown, [{"id": "1", "title": "Acme source"}])
    validation = validate_reformatted_report(
        "# 测试报告\n\n## 正文分析\n" + ("正文事实需要保留正文内引用[1]。" * 120),
        [{"id": "1", "title": "Acme source"}],
        {"sources": [{"id": "1", "title": "Acme source"}], "dimensions": {}},
    )

    assert "数据来源" not in cleaned
    assert "Acme source" not in cleaned
    assert "[1]" in cleaned
    assert not any(item.get("type") == "missing_sources_appendix" for item in validation["fatal_blockers"])


def test_reformatter_blocks_unexpected_source_appendix_when_disabled(monkeypatch):
    monkeypatch.delenv("REPORT_REFORMATTER_SOURCE_APPENDIX", raising=False)
    monkeypatch.setenv("REPORT_REFORMATTER_VALIDATION_MODE", "score")
    monkeypatch.setenv("REPORT_REFORMATTER_MIN_BODY_CHARS", "0")
    markdown = (
        "# 测试报告\n\n"
        "## 正文分析\n"
        + ("正文事实需要保留正文内引用[1]。" * 80)
        + "\n\n## 数据来源\n"
        "[1] Acme source\n"
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
    assert any(item.get("type") == "unexpected_sources_appendix" for item in validation["repair_blockers"])


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


def test_credibility_title_semantics_can_upgrade_media_repost():
    level = _infer_credibility(
        "https://m.thepaper.cn/newsDetail_forward_1",
        "\u4e2d\u56fd\u4fe1\u901a\u9662\u53d1\u5e032024\u5e74\u4eba\u5de5\u667a\u80fd\u4ea7\u4e1a\u767d\u76ae\u4e66",
    )
    low_level = _infer_credibility(
        "https://wenku.baidu.com/view/1",
        "\u4e2d\u56fd\u4fe1\u901a\u9662\u53d1\u5e032024\u5e74\u4eba\u5de5\u667a\u80fd\u4ea7\u4e1a\u767d\u76ae\u4e66",
    )

    assert level == "A"
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


def test_failed_reformatter_writes_writer_fallback_as_clean_report():
    source = inspect.getsource(full_report.main)

    assert "write_markdown(clean_output_path, fallback_report)" in source
    assert '"fallback_writer"' in source
    assert "fallback_output_path" in source
    assert "reformatter_blocked_clean" in source
    assert "final_stdout_allowed" in source
    assert "REPORT_REVIEW_REQUIRED_EXIT_NONZERO" in source
    assert "REPORT_REFORMATTER_FAILURE_EXIT_NONZERO" in source


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
    assert any(item.get("type") == "report_body_below_target_chars" for item in qa["deep_evaluation"]["required_followups"])
    assert any(item.get("type") == "missing_proof_standard" for item in qa["deep_evaluation"]["required_followups"])


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
    assert qa["repair_required"] is True
    assert qa["rewrite_required"] is True


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
    assert qa["repair_required"] is True
    assert qa["rewrite_required"] is False
    assert qa["content_repair_followups"] == []
    assert any(item.get("type") == "mandatory_proof_missing" for item in qa["evidence_repair_followups"])


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


def test_strict_qa_still_blocks_body_length(monkeypatch):
    monkeypatch.setenv("REPORT_QUALITY_MODE", "strict")
    monkeypatch.setenv("REPORT_TARGET_BODY_CHARS_BLOCKING", "false")
    payload = _minimal_publishable_qa_payload()

    qa = run_qa_agent(**payload)

    assert qa["passed"] is False
    assert qa["deep_evaluator_blocking"] is True
    assert any(
        item.get("type") == "deep_report_blocking_gap"
        and item.get("detail", {}).get("type") == "report_body_below_target_chars"
        for item in qa["errors"]
    )
