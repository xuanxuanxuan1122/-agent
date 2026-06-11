from rag_pipeline.agents.final_writer_agent import _render_key_data_block, run_final_writer_agent
from rag_pipeline.agents.markdown_renderer import (
    render_appendix,
    render_chapter_package,
    render_executive_summary,
    render_section,
    render_table_package,
)
from rag_pipeline.agents.public_report_sanitizer import sanitize_public_markdown
from rag_pipeline.flows.report.full_report import finalize_formal_report


def test_metric_fact_is_rendered_as_sentence_not_bare_label():
    lines = render_section(
        {
            "section_title": "\u5e02\u573a\u89c4\u6a21\u80fd\u5426\u9a8c\u8bc1",
            "block_type": "metric_reconciliation",
            "render_blocks": [
                {
                    "type": "paragraph",
                    "text": "\u5e02\u573a\u89c4\u6a21: \u8fbe8.2\u4ebf\u5143\u3002",
                    "fact_type": "metric",
                    "metric": "\u5e02\u573a\u89c4\u6a21",
                    "value": "\u8fbe8.2\u4ebf\u5143",
                    "period": "2025\u5e74",
                    "scope": "\u4e2d\u56fd\u4eba\u5f62\u673a\u5668\u4eba",
                }
            ],
        }
    )

    text = "\n".join(lines)
    assert "\u5e02\u573a\u89c4\u6a21: \u8fbe8.2\u4ebf\u5143" not in text
    assert "\u5e02\u573a\u89c4\u6a21" in text
    assert "\u8fbe8.2\u4ebf\u5143" in text


def test_key_data_block_requires_public_citation_and_scans_past_first_row():
    block = _render_key_data_block(
        "\u5173\u952e\u6570\u636e",
        {},
        [
            {
                "should_render": True,
                "headers": ["\u6307\u6807", "\u6570\u503c"],
                "rows": [
                    {"metric": "\u65e0\u5f15\u7528\u6307\u6807", "value": "30\u4e2a"},
                    {"metric": "\u6709\u5f15\u7528\u6307\u6807", "value": "42%", "citation_ref": "[2]"},
                    {"metric": "\u5907\u9009\u6307\u6807", "value": "12%", "citation_refs": ["[3]"]},
                ],
            }
        ],
    )

    assert "\u65e0\u5f15\u7528\u6307\u6807" not in block
    assert "\u6709\u5f15\u7528\u6307\u6807" in block
    assert "[2]" in block


def test_short_cited_section_expands_for_longform_mode(monkeypatch):
    monkeypatch.setenv("REPORT_ENABLE_RENDERER_TEMPLATE_EXPANSION", "true")
    monkeypatch.setenv("REPORT_RENDER_MIN_SECTION_CHARS", "520")
    lines = render_section(
        {
            "section_title": "\u4ed8\u8d39\u8f6c\u5316\u5728\u54ea\u91cc\u53d1\u751f",
            "block_type": "case_comparison",
            "body_composition_status": "composed",
            "evidence_backed": True,
            "citation_refs": ["[1]"],
            "used_fact_refs": ["EV-1"],
            "render_blocks": [
                {
                    "type": "paragraph",
                    "text": "\u4f01\u4e1a\u7ea7 AI Agent \u5df2\u8fdb\u5165\u5ba2\u670d\u5de5\u4f5c\u6d41\u3002",
                }
            ],
        }
    )

    body = "\n".join(line for line in lines if not line.startswith("###"))
    assert len(body.replace(" ", "")) >= 420
    assert "[1]" in body
    assert "\u4ed8\u8d39\u8f6c\u5316" in body or "\u90e8\u7f72" in body


def test_chapter_heading_rewrites_internal_evidence_terms():
    markdown = render_chapter_package(
        {
            "chapter_title": "\u54ea\u4e9b\u73af\u8282\u5df2\u6709\u5546\u4e1a\u5316\u8bc1\u636e\uff0c\u54ea\u4e9b\u4ecd\u5904\u4e8e\u6982\u5ff5\u6216\u8bd5\u70b9",
            "sections": [
                {
                    "section_title": "\u5546\u4e1a\u5316\u4fe1\u53f7\u662f\u5426\u6e05\u6670",
                    "render_blocks": [
                        {
                            "type": "paragraph",
                            "text": "\u4f01\u4e1a\u7ea7 AI Agent \u5df2\u8fdb\u5165\u5ba2\u6237\u6d41\u7a0b\u3002[1]",
                        }
                    ],
                    "citation_refs": ["[1]"],
                    "evidence_backed": True,
                }
            ],
            "table_packages": [],
        },
        1,
    )

    assert "\u5546\u4e1a\u5316\u8bc1\u636e" not in markdown
    assert "\u5546\u4e1a\u5316\u4fe1\u53f7" in markdown


def test_executive_summary_filters_metric_fragments_and_keeps_public_judgment():
    markdown = render_executive_summary(
        {
            "core_judgments": [
                {"judgment": "\u6e17\u900f\u7387\uff1b2023\u5e74"},
                {"judgment": "adoption: 50%"},
                {
                    "judgment": (
                        "\u4f01\u4e1a\u7ea7 AI Agent \u7684\u9700\u6c42\u6b63\u5728\u4ece\u5de5\u5177\u8bd5\u7528"
                        "\u8f6c\u5411\u4e1a\u52a1\u90e8\u7f72\uff0c\u4f46\u4ed8\u8d39\u6df1\u5ea6\u4ecd\u53d6\u51b3\u4e8e ROI "
                        "\u4e0e\u6743\u9650\u6cbb\u7406\u3002"
                    )
                },
            ]
        },
        [],
    )

    assert "\u6e17\u900f\u7387\uff1b2023\u5e74" not in markdown
    assert "adoption: 50%" not in markdown
    assert "\u4f01\u4e1a\u7ea7 AI Agent" in markdown
    assert "\u6838\u5fc3\u89c2\u70b9\u4e0e\u4e3b\u8981\u7ed3\u8bba" in markdown


def test_executive_summary_omits_block_when_all_judgments_are_fragments():
    markdown = render_executive_summary(
        {
            "core_judgments": [
                {"judgment": "\u6e17\u900f\u7387\uff1b2023\u5e74"},
                {"judgment": "\u5e02\u573a\u89c4\u6a21: 8.2\u4ebf\u5143"},
                {"judgment": "adoption: 50%"},
            ]
        },
        [],
    )

    assert "\u6838\u5fc3\u89c2\u70b9\u4e0e\u4e3b\u8981\u7ed3\u8bba" not in markdown
    assert "\u6e17\u900f\u7387" not in markdown
    assert "adoption" not in markdown


def test_finalize_formal_report_removes_stale_invalid_executive_summary_block():
    markdown = finalize_formal_report(
        "# AI Agent\u7814\u7a76\u62a5\u544a\n\n"
        "## \u6838\u5fc3\u89c2\u70b9\u4e0e\u4e3b\u8981\u7ed3\u8bba\n"
        "- \u6e17\u900f\u7387\uff1b2023\u5e74\n"
        "\n"
        "## 1. \u9700\u6c42\u9a8c\u8bc1\n"
        "### \u5546\u4e1a\u5316\u4fe1\u53f7\u662f\u5426\u6e05\u6670\n"
        "\u5df2\u51fa\u73b0\u53ef\u8ffd\u8e2a\u9700\u6c42\u4fe1\u53f7\u3002"
    )

    assert "\u6838\u5fc3\u89c2\u70b9\u4e0e\u4e3b\u8981\u7ed3\u8bba" not in markdown
    assert "\u6e17\u900f\u7387\uff1b2023\u5e74" not in markdown
    assert "\u9700\u6c42\u9a8c\u8bc1" in markdown


def test_metric_claim_without_render_blocks_is_rendered_as_sentence():
    markdown = render_chapter_package(
        {
            "chapter_title": "\u5e02\u573a\u89c4\u6a21",
            "sections": [
                {
                    "section_title": "\u5546\u4e1a\u5316\u4fe1\u53f7\u662f\u5426\u6e05\u6670",
                    "block_type": "metric_reconciliation",
                    "claim": "\u5e02\u573a\u89c4\u6a21: \u8fbe8.2\u4ebf\u5143\u3002",
                    "evidence_refs": ["E1"],
                    "evidence_backed": True,
                }
            ],
            "table_packages": [],
        },
        1,
    )

    assert "\u5e02\u573a\u89c4\u6a21: \u8fbe8.2\u4ebf\u5143" not in markdown
    assert "\u5e02\u573a\u89c4\u6a21" in markdown
    assert "\u8fbe8.2\u4ebf\u5143" in markdown


def test_manifest_citation_replaces_stale_trailing_render_block_citation():
    markdown = render_chapter_package(
        {
            "chapter_title": "Demand validation",
            "sections": [
                {
                    "section_title": "Deployment signal",
                    "render_blocks": [
                        {
                            "type": "paragraph",
                            "text": "The deployment signal is visible in customer workflow evidence [8].",
                        }
                    ],
                    "citation_refs": ["[1]"],
                    "evidence_refs": ["[1]"],
                    "evidence_backed": True,
                }
            ],
            "table_packages": [],
        },
        1,
    )

    assert "[1]" in markdown
    assert "[8]" not in markdown


def test_hypothesis_id_section_title_is_not_rendered_as_public_h3():
    markdown = render_chapter_package(
        {
            "chapter_title": "Risk boundary",
            "sections": [
                {
                    "section_title": "H4",
                    "claim": "AI Agent risk boundaries depend on permissions, security, and integration cost.",
                    "render_blocks": [
                        {
                            "type": "paragraph",
                            "text": "AI Agent risk boundaries depend on permissions, security, and integration cost [1].",
                        }
                    ],
                    "citation_refs": ["[1]"],
                    "evidence_backed": True,
                }
            ],
            "table_packages": [],
        },
        1,
    )

    assert "### H4" not in markdown
    assert "risk boundaries" in markdown


def test_metric_render_block_is_rewritten_even_when_section_block_is_commercial():
    lines = render_section(
        {
            "section_title": "\u5546\u4e1a\u5316\u4fe1\u53f7\u662f\u5426\u6e05\u6670",
            "block_type": "unit_economics",
            "render_blocks": [{"type": "paragraph", "text": "\u51fa\u8d27/\u90e8\u7f72: \u8d85140\u5bb6"}],
        }
    )

    text = "\n".join(lines)
    assert "\u51fa\u8d27/\u90e8\u7f72: \u8d85140\u5bb6" not in text
    assert "\u51fa\u8d27/\u90e8\u7f72" in text
    assert "\u8d85140\u5bb6" in text


def test_final_writer_rewrites_residual_bare_metric_lines():
    output = run_final_writer_agent(
        query="\u4eba\u5f62\u673a\u5668\u4eba",
        report_blueprint={"chapters": [{"chapter_id": "ch_01", "chapter_title": "\u5e02\u573a\u9700\u6c42"}]},
        chapter_packages=[
            {
                "chapter_id": "ch_01",
                "chapter_title": "\u5e02\u573a\u9700\u6c42",
                "sections": [
                    {
                        "section_id": "s_metric",
                        "section_title": "\u5546\u4e1a\u5316\u4fe1\u53f7\u662f\u5426\u6e05\u6670",
                        "claim": "\u5e02\u573a\u89c4\u6a21: \u8fbe8.2\u4ebf\u5143\u3002",
                        "evidence_refs": ["EV-METRIC"],
                        "used_fact_refs": ["EV-METRIC"],
                        "supporting_facts": [
                            {
                                "source_ref": "EV-METRIC",
                                "value": "\u8fbe8.2",
                                "unit": "\u4ebf\u5143",
                                "period": "2025\u5e74",
                            }
                        ],
                        "evidence_backed": True,
                    }
                ],
            }
        ],
        source_registry=[
            {
                "ref": "EV-METRIC",
                "evidence_id": "EV-METRIC",
                "title": "\u4eba\u5f62\u673a\u5668\u4eba\u5e02\u573a\u89c4\u6a21\u62a5\u544a",
                "url": "https://www.salesforce.com/news/humanoid-market-size",
                "source_level": "B",
            }
        ],
    )

    markdown = output["report_markdown"]
    assert "\u5e02\u573a\u89c4\u6a21: \u8fbe8.2\u4ebf\u5143" not in markdown
    assert "\u5e02\u573a\u89c4\u6a21\u8fbe8.2\u4ebf\u5143" in markdown


def test_news_snippet_inside_claim_is_removed_before_rendering():
    markdown = render_chapter_package(
        {
            "chapter_title": "\u98ce\u9669\u8fb9\u754c",
            "sections": [
                {
                    "section_title": "\u53cd\u5411\u4fe1\u53f7\u5982\u4f55\u5f71\u54cd\u5224\u65ad",
                    "claim": (
                        "\u98ce\u9669\u4e8b\u5b9e\u7528\u4e8e\u6821\u51c6\u4e50\u89c2\u5224\u65ad\uff1b"
                        "\u4e00\u76c6\u51b7\u6c34\u7ec8\u4e8e\u6cfc\u5230\u4e86\u706b\u70ed\u7684\u5177\u8eab\u667a\u80fd\u8d5b\u9053"
                    ),
                    "evidence_refs": ["E1"],
                    "evidence_backed": True,
                }
            ],
            "table_packages": [],
        },
        1,
    )

    assert "\u4e00\u76c6\u51b7\u6c34" not in markdown


def test_headline_like_claims_are_removed_before_rendering():
    markdown = render_chapter_package(
        {
            "chapter_title": "\u5546\u4e1a\u5316\u8fdb\u5c55",
            "sections": [
                {
                    "section_title": "\u5546\u4e1a\u5316\u4fe1\u53f7\u662f\u5426\u6e05\u6670",
                    "claim": "AI Agent\u5546\u4e1a\u5316\u843d\u5730\uff0c3.3\u4e07\u4ebf\u8d5b\u9053\u52a0\u901f\u7206\u53d1\u3002",
                    "evidence_refs": ["E1"],
                    "evidence_backed": True,
                },
                {
                    "section_title": "\u5ba2\u6237\u843d\u5730",
                    "claim": "\u6784\u5efa\u4f01\u4e1a\u7ea7 AI Agent \u8d22\u62a5\u5206\u6790\u6d41\u6c34\u7ebf",
                    "evidence_refs": ["E2"],
                    "evidence_backed": True,
                },
            ],
            "table_packages": [],
        },
        1,
    )

    assert "3.3\u4e07\u4ebf\u8d5b\u9053\u52a0\u901f\u7206\u53d1" not in markdown
    assert "\u6784\u5efa\u4f01\u4e1a\u7ea7 AI Agent \u8d22\u62a5\u5206\u6790\u6d41\u6c34\u7ebf" not in markdown


def test_empty_chapter_package_is_not_rendered_as_h2_shell():
    markdown = render_chapter_package(
        {
            "chapter_title": "\u6ca1\u6709\u8bc1\u636e\u7684\u7ae0\u8282",
            "sections": [],
            "table_packages": [],
            "chapter_omitted_no_evidence": True,
        },
        1,
    )

    assert markdown == ""


def test_chapter_heading_is_compacted_before_rendering():
    markdown = render_chapter_package(
        {
            "chapter_title": (
                "\u4eba\u5f62\u673a\u5668\u4eba\u4ece\u201c\u8868\u6f14\u578b\u79d1\u6280\u201d\u8fdb\u5165"
                "\u201c\u771f\u5b9e\u573a\u666f\u9a8c\u8bc1 + \u8d44\u672c\u5b9a\u4ef7\u201d\u9636\u6bb5\u4e86\u5417"
                "\u662f\u5426\u5b58\u5728\u771f\u5b9e\u9700\u6c42\u548c\u5e02\u573a\u7a7a\u95f4"
            ),
            "sections": [
                {
                    "section_title": "\u9700\u6c42\u9a8c\u8bc1",
                    "claim": "\u5df2\u51fa\u73b0\u53ef\u8ffd\u8e2a\u9700\u6c42\u4fe1\u53f7\u3002",
                    "evidence_refs": ["E1"],
                    "evidence_backed": True,
                }
            ],
            "table_packages": [],
        },
        1,
    )

    first_line = markdown.splitlines()[0]
    assert len(first_line) <= 36
    assert "\u4eba\u5f62\u673a\u5668\u4eba\u4ece" not in first_line


def test_repeated_section_titles_are_rewritten_with_claim_context():
    markdown = render_chapter_package(
        {
            "chapter_title": "Workflow adoption",
            "sections": [
                {
                    "section_title": "Market signal",
                    "claim": "Enterprise workflows show repeatable AI agent deployment.",
                    "render_blocks": [{"type": "paragraph", "text": "Enterprise workflows show repeatable AI agent deployment."}],
                    "evidence_refs": ["E1"],
                    "evidence_backed": True,
                },
                {
                    "section_title": "Market signal",
                    "claim": "Government procurement shows public-sector demand.",
                    "render_blocks": [{"type": "paragraph", "text": "Government procurement shows public-sector demand."}],
                    "evidence_refs": ["E2"],
                    "evidence_backed": True,
                },
            ],
            "table_packages": [],
        },
        1,
    )

    assert markdown.count("### Market signal") == 1
    assert "### Government procurement" in markdown


def test_chapter_lead_drops_source_title_snippet():
    snippet = (
        "AI 时代，唯一确定的是数据｜爱分析访谈 - 电子工程专辑"
        "（2026-05-21T00:00:00+08:00）：数据是穿越周期的壁垒，"
        "以下为本次访谈实录。"
    )
    markdown = render_chapter_package(
        {
            "chapter_title": "Workflow adoption",
            "lead": snippet,
            "sections": [
                {
                    "section_title": "Workflow demand",
                    "render_blocks": [
                        {
                            "type": "paragraph",
                            "text": "Enterprise workflows show repeatable AI agent deployment.",
                        }
                    ],
                    "evidence_refs": ["E1"],
                    "evidence_backed": True,
                }
            ],
            "table_packages": [],
        },
        1,
    )

    assert "电子工程专辑" not in markdown
    assert "以下为本次访谈实录" not in markdown
    assert "Enterprise workflows show repeatable AI agent deployment." in markdown


def test_chapter_lead_gets_first_section_citation_when_rendered():
    markdown = render_chapter_package(
        {
            "chapter_title": "Workflow adoption",
            "lead": "本章判断企业智能体是否已经进入流程部署，并观察样本能否支撑付费转化。",
            "sections": [
                {
                    "section_title": "Workflow demand",
                    "render_blocks": [
                        {
                            "type": "paragraph",
                            "text": "Enterprise workflows show repeatable AI agent deployment.",
                        }
                    ],
                    "citation_refs": ["[3]"],
                    "evidence_refs": ["E1"],
                    "evidence_backed": True,
                }
            ],
            "table_packages": [],
        },
        1,
    )

    assert "本章判断企业智能体是否已经进入流程部署，并观察样本能否支撑付费转化。[3]" in markdown


def test_executive_summary_omits_factual_bullets_without_public_citation():
    markdown = render_executive_summary(
        {
            "core_judgments": [
                {
                    "label": "机会判断",
                    "judgment": "AI Agent在A股上市公司中已形成广泛关注，近3800家公司在年报中提及AI相关内容。",
                },
                {
                    "label": "机会判断",
                    "judgment": "AI Agent已经进入公共部门部署。",
                    "citation_refs": ["[2]"],
                },
            ]
        },
        [],
    )

    assert "近3800家公司" not in markdown
    assert "AI Agent已经进入公共部门部署。[2]" in markdown


def test_executive_summary_omits_market_and_ipo_claims_without_citation():
    markdown = render_executive_summary(
        {
            "core_judgments": [
                {
                    "label": "机会判断",
                    "judgment": "资本市场对AI Agent相关企业给予高估值，反映投资者对市场空间的乐观预期。",
                },
                {
                    "label": "机会判断",
                    "judgment": "AI Agent生态中已出现冲刺IPO的独角兽企业，表明资本市场对工业AI智能体领域的关注。",
                },
                {
                    "label": "机会判断",
                    "judgment": "AI Agent生态中已出现可追踪部署样本，说明企业智能体正在从试点走向流程部署。",
                    "citation_refs": ["[4]"],
                },
            ]
        },
        [],
    )

    assert "高估值" not in markdown
    assert "独角兽" not in markdown
    assert "AI Agent生态中已出现可追踪部署样本，说明企业智能体正在从试点走向流程部署。[4]" in markdown


def test_render_appendix_keeps_public_sources_and_omits_diagnostic_tables():
    rendered = render_appendix(
        [{"ref": "[1]", "title": "来源A", "url": "https://example.org/a"}],
        {
            "metric_normalization_table": [
                {
                    "metric_name": "CAGR",
                    "subject": "AI Agent",
                    "scope": "全球",
                    "period": "2028年",
                    "unit": "%",
                    "value": "41%",
                    "source_level": "B",
                }
            ],
            "table_appendix_rows": [
                {
                    "title": "市场指标与口径表",
                    "headers": ["指标", "范围", "期间", "数值", "单位", "后续影响"],
                    "rows": [
                        [
                            "CAGR",
                            "全球",
                            "2028年",
                            "41",
                            "%",
                            "该指标须同时披露范围、期间、单位与来源等级,才进入正文判断。",
                        ],
                        [
                            "市场规模",
                            "全球",
                            "2028年",
                            "100",
                            "亿美元",
                            "该指标须同时披露范围、期间、单位与来源等级,才进入正文判断。",
                        ],
                    ],
                    "should_render": True,
                    "validation_status": "passed",
                }
            ],
        },
    )

    assert rendered.count("## 来源附录") == 1
    assert "- [1]" in rendered
    assert "指标口径表" not in rendered
    assert "附录明细" not in rendered
    assert "后续影响" not in rendered
    assert "该指标须" not in rendered


def test_render_appendix_keeps_public_table_rows_without_source_registry():
    rendered = render_appendix(
        [],
        {
            "table_appendix_rows": [
                {
                    "title": "Key metric appendix",
                    "headers": ["metric", "scope", "value"],
                    "rows": [
                        ["adoption", "enterprise", "42%"],
                        ["deployment", "pilot", "18%"],
                    ],
                    "should_render": True,
                    "validation_status": "passed",
                }
            ],
        },
    )

    assert rendered
    assert "Key metric appendix" in rendered
    assert "| metric | scope | value |" in rendered


def test_sanitize_public_markdown_removes_analysis_scaffold_language():
    markdown = (
        "# AI Agent研究报告\n\n"
        "## 1. 需求验证\n"
        "### ch_01\n"
        "### 关键事实与判断依据\n"
        "本章判断应以“市场规模: 100亿美元”为事实锚点。\n"
        "先用 增速: 44.2% 确认本章的事实起点。\n"
        "后续重点跟踪同口径指标、反向样本和执行进展。\n"
        "可复核材料指向：某行业报告显示市场规模增长。[1]\n"
        "这些事实来自不同类型来源且方向一致时，可以支撑较强结论。\n"
        "来源集中、口径不一致或缺少反向样本时，结论会保留边界。\n\n"
        "这张表显示，成本的表内信号是成本 / 62.5%。[1]\n"
        "后续影响：若表内信号继续被高等级来源验证，可纳入章节分析。[1]\n"
        "使用边界：表格优先使用已绑定到本章的正文证据。[1]\n\n"
        "本章需要按连续指标和反向样本拆解，避免把单点信号直接外推为行业结论。\n"
        "这一判断目前更适合作为背景条件，结论强度取决于后续连续指标和相反样本的变化。\n"
        "### 空标题\n"
        "### 被清空标题\n"
        "后续重点跟踪同口径指标。\n"
        "## 来源附录\n"
        "- [1] 来源A | https://example.org/a\n"
    )

    cleaned = sanitize_public_markdown(markdown, mode="enforce")

    for phrase in [
        "事实锚点",
        "事实起点",
        "后续重点跟踪",
        "可复核材料指向",
        "这些事实来自不同类型来源",
        "来源集中、口径不一致",
        "这张表显示",
        "后续影响",
        "使用边界",
        "需要按连续指标",
        "避免把单点信号直接外推",
        "更适合作为背景条件",
        "结论强度取决",
        "### ch_01",
        "关键事实与判断依据",
        "### 空标题",
        "### 被清空标题",
    ]:
        assert phrase not in cleaned
    assert "## 来源附录" in cleaned


def test_sanitize_public_markdown_removes_evidence_repair_signals_from_public_body():
    markdown = (
        "# AI Agent研究报告\n\n"
        "## 1. 商业化验证\n"
        "商业化证据主要集中在金融、采购、政企等少数行业，其他行业缺乏明确案例；多数证据为2025-2026年报告，时效性有限；来源多为B级或C级，可靠性中等。\n\n"
        "可公开事实显示，采购系统中的智能体部署已经进入供应商、合同和订单管理流程。[1]\n\n"
        "## 来源附录\n"
        "- [1] 来源A | https://example.org/a\n"
    )

    cleaned = sanitize_public_markdown(markdown, mode="enforce")

    for phrase in [
        "商业化证据主要集中",
        "其他行业缺乏明确案例",
        "多数证据为2025-2026年报告",
        "时效性有限",
        "来源多为B级或C级",
        "可靠性中等",
    ]:
        assert phrase not in cleaned
    assert "采购系统中的智能体部署" in cleaned
    assert "## 来源附录" in cleaned


def test_render_table_package_drops_public_diagnostic_columns():
    markdown = render_table_package(
        {
            "should_render": True,
            "title": "竞争格局对照表",
            "headers": ["Metric", "Value", "Competitive signal", "Risk boundary"],
            "rows": [
                {
                    "cells": ["客户部署", "2个案例", "Competitive signal from vendor", "Risk boundary needs review"],
                    "evidence_refs": ["[1]"],
                },
                {
                    "cells": ["产品发布", "3项能力", "Competitive signal from product", "Risk boundary needs review"],
                    "evidence_refs": ["[2]"],
                },
            ],
            "evidence_refs": ["[1]", "[2]"],
        }
    )

    assert markdown
    assert "Competitive signal" not in markdown
    assert "Risk boundary" not in markdown
    assert "客户部署" in markdown
    assert "产品发布" in markdown


def test_render_section_does_not_fill_short_body_with_public_templates(monkeypatch):
    from rag_pipeline.agents.markdown_renderer import render_section

    monkeypatch.setenv("REPORT_RENDER_MIN_SECTION_CHARS", "900")
    lines = render_section(
        {
            "section_id": "s1",
            "section_title": "Workflow deployment signal",
            "block_type": "case_comparison",
            "evidence_backed": True,
            "citation_refs": ["[1]"],
            "render_blocks": [
                {
                    "type": "paragraph",
                    "text": "Enterprise workflows show repeatable AI agent deployment.",
                }
            ],
        }
    )

    body = "\n".join(lines)
    assert "Enterprise workflows show repeatable AI agent deployment" in body
    assert len(body) < 300


def test_render_table_package_keeps_investment_diagnostic_table_score_only():
    markdown = render_table_package(
        {
            "should_render": True,
            "title": "投资优先级矩阵",
            "table_type": "investment_priority_table",
            "headers": ["对象", "评分", "存疑", "raw URL"],
            "rows": [
                {
                    "cells": ["Vendor A", "85", "来源仍需核验", "https://example.invalid/raw"],
                    "evidence_refs": ["[1]"],
                },
                {
                    "cells": ["Vendor B", "72", "口径不一致", "https://example.invalid/raw2"],
                    "evidence_refs": ["[2]"],
                },
            ],
            "evidence_refs": ["[1]", "[2]"],
        }
    )

    assert markdown == ""
