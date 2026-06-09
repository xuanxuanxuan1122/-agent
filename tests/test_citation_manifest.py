from rag_pipeline.agents.citation_manifest import (
    attach_manifest_citations,
    build_citation_manifest,
    evidence_source_entries_from_package,
    manifest_appendix_sources,
    merge_source_registries,
)
from rag_pipeline.agents.final_writer_agent import (
    _traceable_source_registry,
    finalize_markdown_citations,
    run_final_writer_agent,
)
from rag_pipeline.flows.report.final_audit_agent import run_deterministic_audit


def test_final_citation_reconciliation_renumbers_body_and_appendix_by_final_body_order():
    body = "Metric evidence appears later [7], then case evidence [9], then risk evidence [8]."
    manifest = {
        "appendix_sources": [
            {"ref": "[7]", "title": "Metric source", "url": "https://example.org/metric"},
            {"ref": "[8]", "title": "Risk source", "url": "https://example.org/risk"},
            {"ref": "[9]", "title": "Case source", "url": "https://example.org/case"},
        ],
        "evidence_to_citation": {"EV-M": "[7]", "EV-R": "[8]", "EV-C": "[9]"},
    }

    rewritten, sources, diagnostics = finalize_markdown_citations(body, manifest, manifest["appendix_sources"])

    assert rewritten == "Metric evidence appears later [1], then case evidence [2], then risk evidence [3]."
    assert [source["ref"] for source in sources] == ["[1]", "[2]", "[3]"]
    assert [source["title"] for source in sources] == ["Metric source", "Case source", "Risk source"]
    assert diagnostics["final_citation_reconciliation_status"] == "ok"
    assert diagnostics["final_missing_appendix_refs"] == []


def test_final_citation_reconciliation_removes_unresolved_final_body_refs():
    body = "The report cites a valid source [7], a missing source [8], and another valid source [9]."
    sources = [
        {"ref": "[7]", "title": "Valid source A", "url": "https://example.org/a"},
        {"ref": "[9]", "title": "Valid source B", "url": "https://example.org/b"},
    ]

    rewritten, appendix_sources, diagnostics = finalize_markdown_citations(body, {"appendix_sources": sources}, sources)

    assert "[8]" not in rewritten
    assert "[1]" in rewritten and "[2]" in rewritten
    assert [source["ref"] for source in appendix_sources] == ["[1]", "[2]"]
    assert diagnostics["final_unresolved_citation_removed_count"] == 1
    assert diagnostics["final_unresolved_citation_refs"] == ["[8]"]


def test_final_citation_reconciliation_collapses_adjacent_duplicate_refs():
    body = "同一事实被重复引用时不应留下重复脚注 [7][8]，直接重复也应收口 [9][9]。"
    sources = [
        {"ref": "[7]", "title": "Shared source A", "url": "https://example.org/shared"},
        {"ref": "[8]", "title": "Shared source B", "url": "https://example.org/shared"},
        {"ref": "[9]", "title": "Other source", "url": "https://example.org/other"},
    ]

    rewritten, appendix_sources, diagnostics = finalize_markdown_citations(body, {"appendix_sources": sources}, sources)

    assert "[1][1]" not in rewritten
    assert "[2][2]" not in rewritten
    assert "重复脚注 [1]" in rewritten
    assert "直接重复也应收口 [2]" in rewritten
    assert [source["ref"] for source in appendix_sources] == ["[1]", "[2]"]
    assert diagnostics["final_duplicate_citation_removed_count"] == 2


def test_final_citation_reconciliation_keeps_newline_after_collapsed_refs():
    body = "上一章结尾 [7][8]\n\n## 下一章\n正文 [9]。"
    sources = [
        {"ref": "[7]", "title": "Shared source A", "url": "https://example.org/shared"},
        {"ref": "[8]", "title": "Shared source B", "url": "https://example.org/shared"},
        {"ref": "[9]", "title": "Other source", "url": "https://example.org/other"},
    ]

    rewritten, _appendix_sources, diagnostics = finalize_markdown_citations(body, {"appendix_sources": sources}, sources)

    assert "上一章结尾 [1]\n\n## 下一章" in rewritten
    assert "[1][1]" not in rewritten
    assert diagnostics["final_duplicate_citation_removed_count"] == 1


def test_final_citation_reconciliation_prefers_manifest_public_ref_over_source_alias():
    body = "The market sizing claim uses the manifest public citation [3]."
    manifest = {
        "appendix_sources": [
            {
                "ref": "[3]",
                "title": "Official market statistics",
                "url": "https://www.stats.gov.cn/tjsj/zxfb/202605/t20260501.html",
                "evidence_refs": ["EV-M"],
            }
        ],
        "evidence_to_citation": {"EV-M": "[3]"},
    }
    stale_registry = [
        {
            "ref": "[2]",
            "title": "Unrelated risk source",
            "url": "https://example.org/risk",
            "evidence_refs": ["EV-R", "[3]"],
            "source_refs": ["EV-R", "[3]"],
        }
    ]

    rewritten, sources, diagnostics = finalize_markdown_citations(body, manifest, stale_registry)

    assert rewritten == "The market sizing claim uses the manifest public citation [1]."
    assert sources[0]["title"] == "Official market statistics"
    assert sources[0]["url"] == "https://www.stats.gov.cn/tjsj/zxfb/202605/t20260501.html"
    assert diagnostics["final_citation_reconciliation_status"] == "ok"


def test_final_citation_reconciliation_manifest_public_refs_override_manifest_aliases():
    body = "The market sizing claim uses the manifest public citation [3]."
    manifest = {
        "appendix_sources": [
            {
                "ref": "[2]",
                "title": "Unrelated risk source",
                "url": "https://example.org/risk",
                "evidence_refs": ["EV-R", "[3]"],
                "source_refs": ["EV-R", "[3]"],
            },
            {
                "ref": "[3]",
                "title": "Official market statistics",
                "url": "https://www.stats.gov.cn/tjsj/zxfb/202605/t20260501.html",
                "evidence_refs": ["EV-M"],
            },
        ],
        "evidence_to_citation": {"EV-M": "[3]", "EV-R": "[2]"},
    }

    rewritten, sources, diagnostics = finalize_markdown_citations(body, manifest, manifest["appendix_sources"])

    assert rewritten == "The market sizing claim uses the manifest public citation [1]."
    assert sources[0]["title"] == "Official market statistics"
    assert sources[0]["url"] == "https://www.stats.gov.cn/tjsj/zxfb/202605/t20260501.html"
    assert diagnostics["final_citation_reconciliation_status"] == "ok"


def test_final_citation_reconciliation_drops_long_factual_body_without_citations():
    body = (
        "2025年 AI 生成视音频内容超过 20 亿条，企业级 Agent 市场规模继续扩张，"
        "并且该判断同时涉及市场规模、企业采购、产业政策和收入预测等多个可核验事实，"
        "因此不能作为无引用短句被静默删除。这个段落还继续描述供应商竞争、客户预算、"
        "产品部署周期和监管约束对行业增长的影响，长度足以代表正文段落而不是孤立数据行。"
    )

    rewritten, appendix_sources, diagnostics = finalize_markdown_citations(body, {"appendix_sources": []}, [])

    assert rewritten == ""
    assert appendix_sources == []
    assert diagnostics["final_citation_reconciliation_status"] == "ok"
    assert diagnostics["citationless_factual_sentence_removed_count"] == 1
    assert diagnostics["factual_body_without_citations_count"] == 0
    assert diagnostics["citationless_fact_examples"] == []


def test_final_citation_reconciliation_drops_trailing_uncited_factual_sentence_in_cited_paragraph():
    body = (
        "Enterprise AI Agent adoption is moving from pilots into workflow automation [1]. "
        "OpenAI revenue reached 20 billion in 2025."
    )
    manifest = {
        "appendix_sources": [
            {"ref": "[1]", "title": "Enterprise AI adoption", "url": "https://example.org/adoption"}
        ]
    }

    rewritten, appendix_sources, diagnostics = finalize_markdown_citations(body, manifest, manifest["appendix_sources"])

    assert "workflow automation [1]" in rewritten
    assert "OpenAI revenue reached 20 billion" not in rewritten
    assert appendix_sources[0]["ref"] == "[1]"
    assert diagnostics["final_citation_reconciliation_status"] == "ok"
    assert diagnostics["citationless_factual_sentence_removed_count"] == 1
    assert diagnostics["factual_body_without_citations_count"] == 0


def test_final_citation_reconciliation_drops_citationless_factual_bullets():
    body = "\n".join(
        [
            "- \u673a\u4f1a\u5224\u65ad\uff1aOpenAI \u4e0e Microsoft \u7684\u6280\u672f\u548c\u76d1\u7ba1\u7ea6\u675f\u4f1a\u5982\u4f55\u6539\u53d8\u673a\u4f1a\u6392\u5e8f",
            "- \u6e17\u900f\u7387\u4e3a10%\uff0c\u671f\u95f4\u4e3a2011\u5e74",
            "- Directional synthesis without a concrete factual assertion.",
        ]
    )

    rewritten, appendix_sources, diagnostics = finalize_markdown_citations(body, {"appendix_sources": []}, [])

    assert "\u673a\u4f1a\u5224\u65ad" in rewritten
    assert "\u6e17\u900f\u7387\u4e3a10%" not in rewritten
    assert "Directional synthesis" in rewritten
    assert appendix_sources == []
    assert diagnostics["final_citation_reconciliation_status"] == "ok"
    assert diagnostics["citationless_factual_bullet_removed_count"] == 1
    assert diagnostics["factual_body_without_citations_count"] == 0


def test_final_citation_reconciliation_drops_short_citationless_factual_lines():
    body = "\n".join(
        [
            "\u6e17\u900f\u7387\u4e3a10%\uff0c\u671f\u95f4\u4e3a2011\u5e74",
            "This longer paragraph is analytical framing without a concrete numeric claim.",
        ]
    )

    rewritten, appendix_sources, diagnostics = finalize_markdown_citations(body, {"appendix_sources": []}, [])

    assert "\u6e17\u900f\u7387\u4e3a10%" not in rewritten
    assert "analytical framing" in rewritten
    assert appendix_sources == []
    assert diagnostics["final_citation_reconciliation_status"] == "ok"
    assert diagnostics["citationless_short_factual_line_removed_count"] == 1
    assert diagnostics["factual_body_without_citations_count"] == 0


def test_final_citation_reconciliation_marks_rebind_required_when_many_facts_removed(monkeypatch):
    monkeypatch.setenv("REPORT_FINAL_CITATION_REBIND_REMOVAL_THRESHOLD", "3")
    body = "\n".join(
        [
            "OpenAI revenue reached 20 billion in 2025.",
            "Microsoft revenue reached 30 billion in 2025.",
            "Google revenue reached 40 billion in 2025.",
            "Salesforce revenue reached 50 billion in 2025.",
        ]
    )

    rewritten, appendix_sources, diagnostics = finalize_markdown_citations(body, {"appendix_sources": []}, [])

    assert rewritten == ""
    assert appendix_sources == []
    assert diagnostics["citationless_factual_removed_count"] == 4
    assert diagnostics["citation_rebind_required"] is True
    assert diagnostics["clean_report_eligible"] is False
    assert diagnostics["citation_rebind_reason"] == "citationless_factual_removal_exceeded_threshold"


def test_final_citation_reconciliation_rebinds_when_removal_count_reaches_threshold(monkeypatch):
    monkeypatch.setenv("REPORT_FINAL_CITATION_REBIND_REMOVAL_THRESHOLD", "5")
    body = "\n".join(
        [
            "OpenAI revenue reached 20 billion in 2025.",
            "Microsoft revenue reached 30 billion in 2025.",
            "Google revenue reached 40 billion in 2025.",
            "Salesforce revenue reached 50 billion in 2025.",
            "Anthropic revenue reached 60 billion in 2025.",
        ]
    )

    rewritten, _, diagnostics = finalize_markdown_citations(body, {"appendix_sources": []}, [])

    assert rewritten == ""
    assert diagnostics["citationless_factual_removed_count"] == 5
    assert diagnostics["citation_rebind_required"] is True
    assert diagnostics["citation_binding_quality_low"] is True
    assert diagnostics["repair_required"] == "citation_rebind"
    assert diagnostics["clean_report_eligible"] is False


def test_manifest_maps_section_evidence_ref_to_public_citation():
    chapters = [
        {
            "chapter_id": "ch_01",
            "sections": [
                {
                    "section_id": "s1",
                    "used_fact_refs": ["EV-1"],
                    "evidence_refs": ["EV-1"],
                }
            ],
        }
    ]
    sources = [
        {
            "ref": "[7]",
            "evidence_id": "EV-1",
            "title": "Salesforce Agentforce deployment note",
            "url": "https://www.salesforce.com/news/agentforce",
        }
    ]

    manifest = build_citation_manifest(chapters=chapters, claim_units=[], source_registry=sources)
    attached = attach_manifest_citations(chapters, manifest)

    assert manifest["citation_manifest_status"] == "ok"
    assert manifest["evidence_to_citation"]["EV-1"] == "[1]"
    assert attached[0]["sections"][0]["citation_refs"] == ["[1]"]
    assert manifest_appendix_sources(manifest)[0]["ref"] == "[1]"


def test_evidence_source_entries_scans_dynamic_chapter_evidence_buckets():
    entries = evidence_source_entries_from_package(
        evidence_package={
            "chapter_evidence": {
                "ch3": [
                    {
                        "evidence_id": "EV-10-06",
                        "source_level": "A",
                        "source_verification_status": "document_verified",
                        "value": "222.00万元",
                        "unit": "currency_cny",
                        "metric": "采购预算",
                        "source": {
                            "title": "曲靖经开区 AI 技能培训采购公告",
                            "url": "https://www.ccgp.gov.cn/cggg/local/202605/t20260515_123456.htm",
                        },
                    }
                ]
            }
        }
    )

    assert len(entries) == 1
    assert entries[0]["evidence_id"] == "EV-10-06"
    assert "EV-10-06" in entries[0]["evidence_refs"]
    assert entries[0]["title"] == "曲靖经开区 AI 技能培训采购公告"
    assert entries[0]["url"] == "https://www.ccgp.gov.cn/cggg/local/202605/t20260515_123456.htm"


def test_traceable_source_registry_does_not_mark_generic_titles_as_cross_domain_mismatch():
    kept, excluded = _traceable_source_registry(
        [
            {
                "ref": "EV-1",
                "title": "Official AI Agent Statistics",
                "url": "https://rsj.qj.gov.cn/view/gsgg/144714.html",
                "source_level": "A",
                "source_type": "official",
            },
            {
                "ref": "EV-2",
                "title": "Official AI Agent Statistics",
                "url": "https://www.stats.gov.cn/tjsj/zxfb/202605/t20260501.html",
                "source_level": "A",
                "source_type": "official",
            },
        ],
        query="AI Agent生态发展报告",
    )

    assert len(kept) == 2
    assert excluded == []
    assert all(not source.get("source_title_url_mismatch_suspected") for source in kept)
    assert {source["ref"] for source in kept} == {"EV-1", "EV-2"}


def test_manifest_falls_back_to_chapter_section_when_claim_section_id_is_stale():
    chapters = [
        {
            "chapter_id": "ch_01",
            "sections": [
                {
                    "section_id": "rendered_s1",
                    "section_title": "Customer deployment",
                }
            ],
        }
    ]
    claims = [
        {
            "chapter_id": "ch_01",
            "section_id": "stale_layout_s9",
            "evidence_refs": ["EV-1"],
        }
    ]
    sources = [
        {
            "ref": "[7]",
            "evidence_id": "EV-1",
            "title": "Salesforce Agentforce deployment note",
            "url": "https://www.salesforce.com/news/agentforce",
        }
    ]

    manifest = build_citation_manifest(chapters=chapters, claim_units=claims, source_registry=sources)
    attached = attach_manifest_citations(chapters, manifest)

    assert manifest["citation_manifest_status"] == "ok"
    assert attached[0]["sections"][0]["citation_refs"] == ["[1]"]


def test_final_writer_renders_manifest_appendix_from_claim_refs(monkeypatch):
    monkeypatch.setenv("REPORT_FINAL_WRITER_SOURCE_APPENDIX", "true")

    output = run_final_writer_agent(
        query="AI Agent",
        report_blueprint={
            "report_shell": {"front_blocks": [], "back_blocks": ["appendix"]},
            "chapters": [{"chapter_id": "ch_01", "chapter_title": "Demand validation"}],
        },
        chapter_packages=[
            {
                "chapter_id": "ch_01",
                "chapter_title": "Demand validation",
                "sections": [
                    {
                        "section_id": "s1",
                        "section_title": "Customer deployment",
                        "claim": "Enterprise AI agents are moving from demos into customer-service workflows.",
                        "reasoning": "Workflow deployment matters because it requires permissions, integration, and process ownership.",
                        "mechanism": "A production workflow is a stronger signal than a demo.",
                        "used_fact_refs": ["EV-1"],
                        "evidence_refs": ["EV-1"],
                        "evidence_backed": True,
                    }
                ],
            }
        ],
        source_registry=[
            {
                "ref": "[7]",
                "evidence_id": "EV-1",
                "title": "Salesforce Agentforce deployment note",
                "url": "https://www.salesforce.com/news/agentforce",
                "source_level": "B",
            }
        ],
    )

    markdown = output["report_markdown"]
    assert output["citation_manifest"]["citation_manifest_status"] == "ok"
    assert "##" in markdown and "[1]" in markdown
    assert "Salesforce Agentforce deployment note" in markdown
    assert run_deterministic_audit(
        report_markdown=markdown,
        writer_package_payload={"writer_report": output, "citation_manifest": output["citation_manifest"]},
    )["fatal"] is False


def test_manifest_blocks_cited_title_only_source():
    chapters = [{"chapter_id": "ch_01", "sections": [{"section_id": "s1", "evidence_refs": ["EV-2"]}]}]
    sources = [{"ref": "[2]", "evidence_id": "EV-2", "title": "Untethered title only source"}]

    manifest = build_citation_manifest(chapters=chapters, claim_units=[], source_registry=sources)

    assert manifest["citation_manifest_status"] == "blocked"
    assert manifest["excluded_cited_sources"]
    assert manifest["appendix_sources"] == []


def test_manifest_maps_ev_ref_through_chapter_evidence_source_alias():
    chapters = [
        {
            "chapter_id": "ch_01",
            "sections": [
                {
                    "section_id": "s1",
                    "used_fact_refs": ["EV-01-13"],
                    "evidence_refs": ["EV-01-13"],
                }
            ],
        }
    ]
    source_registry = [
        {
            "ref": "[12]",
            "title": "Government deployment announcement",
            "url": "https://www.gov.cn/agent-deployment",
            "source_level": "A",
        }
    ]
    chapter_evidence_packages = [
        {
            "chapter_id": "ch_01",
            "core_evidence": [
                {
                    "evidence_id": "EV-01-13",
                    "source_ref": "EV-01-13",
                    "source_url": "https://www.gov.cn/agent-deployment",
                    "source_title": "Government deployment announcement",
                    "source_level": "A",
                }
            ],
        }
    ]

    bridged_sources = merge_source_registries(
        source_registry,
        evidence_source_entries_from_package(chapter_evidence_packages=chapter_evidence_packages),
    )
    manifest = build_citation_manifest(chapters=chapters, claim_units=[], source_registry=bridged_sources)

    assert manifest["citation_manifest_status"] == "ok"
    assert manifest["missing_evidence_refs"] == []
    assert manifest["evidence_to_citation"]["EV-01-13"] == "[1]"
    assert manifest["appendix_sources"][0]["url"] == "https://www.gov.cn/agent-deployment"


def test_merge_source_registries_replaces_generic_title_with_specific_evidence_title():
    generic_registry = [
        {
            "ref": "[14]",
            "title": "Official AI Agent Statistics",
            "source_title": "Official AI Agent Statistics",
            "url": "https://www.fmprc.gov.cn/eng/zy/gb/202507/t20250729_11679232.html",
            "source_url": "https://www.fmprc.gov.cn/eng/zy/gb/202507/t20250729_11679232.html",
        }
    ]
    evidence_entries = [
        {
            "evidence_id": "EV-GOV",
            "source_ref": "[14]",
            "citation_ref": "[14]",
            "source_title": "Global AI Governance Action Plan",
            "source_url": "https://www.fmprc.gov.cn/eng/zy/gb/202507/t20250729_11679232.html",
        }
    ]

    merged = merge_source_registries(generic_registry, evidence_entries)

    assert len(merged) == 1
    assert merged[0]["title"] == "Global AI Governance Action Plan"
    assert merged[0]["source_title"] == "Global AI Governance Action Plan"
    assert merged[0]["url"] == generic_registry[0]["url"]


def test_merge_source_registries_keeps_specific_title_over_weaker_generic_title():
    specific_registry = [
        {
            "ref": "[8]",
            "title": "Salesforce Agentforce customer workflow deployments",
            "source_title": "Salesforce Agentforce customer workflow deployments",
            "url": "https://www.salesforce.com/news/agentforce",
            "source_url": "https://www.salesforce.com/news/agentforce",
        }
    ]
    weaker_entries = [
        {
            "evidence_id": "EV-SF",
            "source_ref": "[8]",
            "citation_ref": "[8]",
            "source_title": "Official AI Agent Statistics",
            "source_url": "https://www.salesforce.com/news/agentforce",
        }
    ]

    merged = merge_source_registries(specific_registry, weaker_entries)

    assert len(merged) == 1
    assert merged[0]["title"] == "Salesforce Agentforce customer workflow deployments"
    assert merged[0]["source_title"] == "Salesforce Agentforce customer workflow deployments"


def test_evidence_source_entries_drop_dict_string_publisher_metadata():
    dict_string = "{'title': 'Unrelated source', 'url': 'https://www.sc.gov.cn/noise'}"
    entries = evidence_source_entries_from_package(
        chapter_evidence_packages=[
            {
                "chapter_id": "ch_01",
                "core_evidence": [
                    {
                        "evidence_id": "EV-1",
                        "source_ref": "[1]",
                        "source_url": "https://irm.cninfo.com.cn/ircs/question/questionDetail?questionId=1",
                        "publisher": dict_string,
                    }
                ],
            }
        ]
    )

    assert len(entries) == 1
    assert dict_string not in str(entries[0])
    assert not entries[0].get("publisher")


def test_merge_source_registries_drops_dict_string_source_metadata():
    dict_string = "{'title': 'Unrelated source', 'url': 'https://www.sc.gov.cn/noise'}"
    merged = merge_source_registries(
        [
            {
                "ref": "[1]",
                "title": "Investor relations Q&A",
                "url": "https://irm.cninfo.com.cn/ircs/question/questionDetail?questionId=1",
                "publisher": dict_string,
                "source": dict_string,
            }
        ]
    )

    assert len(merged) == 1
    assert not merged[0].get("publisher")
    assert not merged[0].get("source")
    assert "Investor relations Q&A" in merged[0]["title"]


def test_merge_source_registries_does_not_merge_different_traceable_urls_by_alias_ref():
    merged = merge_source_registries(
        [
            {
                "ref": "[1]",
                "title": "Investor relations Q&A",
                "url": "https://irm.cninfo.com.cn/ircs/question/questionDetail?questionId=1",
                "source_refs": ["[1]", "[3]"],
            },
            {
                "ref": "[3]",
                "title": "Siping science bureau article",
                "url": "http://kjj.siping.gov.cn/kjxx/kpxcl/202605/t20260512_766150.html",
            },
        ]
    )

    assert len(merged) == 2
    assert {source["url"] for source in merged} == {
        "https://irm.cninfo.com.cn/ircs/question/questionDetail?questionId=1",
        "http://kjj.siping.gov.cn/kjxx/kpxcl/202605/t20260512_766150.html",
    }
    assert all(source.get("url") == source.get("source_url", source.get("url")) for source in merged)


def test_manifest_dedupes_public_refs_by_logical_source_not_object_identity():
    chapters = [
        {
            "chapter_id": "ch_01",
            "sections": [
                {"section_id": "s1", "used_fact_refs": ["EV-1"]},
                {"section_id": "s2", "used_fact_refs": ["EV-2"]},
            ],
        }
    ]
    sources = [
        {
            "ref": "EV-1",
            "evidence_id": "EV-1",
            "title": "Shared deployment source",
            "url": "https://example.org/shared-source",
        },
        {
            "ref": "EV-2",
            "evidence_id": "EV-2",
            "title": "Shared deployment source copy",
            "url": "https://example.org/shared-source",
        },
    ]

    manifest = build_citation_manifest(chapters=chapters, claim_units=[], source_registry=sources)

    assert manifest["citation_manifest_status"] == "ok"
    assert manifest["evidence_to_citation"]["EV-1"] == "[1]"
    assert manifest["evidence_to_citation"]["EV-2"] == "[1]"
    assert len(manifest["appendix_sources"]) == 1


def test_manifest_filters_dead_link_sources_from_public_appendix():
    chapters = [{"chapter_id": "ch_01", "sections": [{"section_id": "s1", "used_fact_refs": ["EV-404"]}]}]
    sources = [
        {
            "ref": "EV-404",
            "evidence_id": "EV-404",
            "title": "页面未找到",
            "url": "https://example.org/404",
            "summary": "404 not found",
        }
    ]

    manifest = build_citation_manifest(chapters=chapters, claim_units=[], source_registry=sources)

    assert manifest["citation_manifest_status"] == "blocked"
    assert manifest["appendix_sources"] == []
    assert manifest["excluded_cited_sources"][0]["reason"] == "dead_link"


def test_manifest_attaches_claim_refs_to_matching_section_when_section_refs_are_missing():
    chapters = [{"chapter_id": "ch_01", "sections": [{"section_id": "s1", "claim": "Workflow deployment is visible."}]}]
    claims = [{"chapter_id": "ch_01", "section_id": "s1", "evidence_refs": ["EV-1"]}]
    sources = [
        {
            "ref": "EV-1",
            "evidence_id": "EV-1",
            "title": "Workflow deployment source",
            "url": "https://example.org/workflow",
        }
    ]

    manifest = build_citation_manifest(chapters=chapters, claim_units=claims, source_registry=sources)
    attached = attach_manifest_citations(chapters, manifest)

    assert manifest["citation_manifest_status"] == "ok"
    assert attached[0]["sections"][0]["citation_refs"] == ["[1]"]


def test_final_writer_filters_unresolved_refs_before_rendered_manifest(monkeypatch):
    monkeypatch.setenv("REPORT_FINAL_WRITER_SOURCE_APPENDIX", "true")

    output = run_final_writer_agent(
        query="AI Agent",
        report_blueprint={
            "report_shell": {"front_blocks": [], "back_blocks": ["appendix"]},
            "chapters": [{"chapter_id": "ch_01", "chapter_title": "Demand validation"}],
        },
        chapter_packages=[
            {
                "chapter_id": "ch_01",
                "chapter_title": "Demand validation",
                "sections": [
                    {
                        "section_id": "s1",
                        "section_title": "Workflow deployment",
                        "claim": "Workflow deployment is visible.",
                        "reasoning": "A production workflow requires integration and permissions.",
                        "mechanism": "Integration turns a demo into an operating signal.",
                        "used_fact_refs": ["EV-OK", "EV-06-26"],
                        "evidence_refs": ["EV-OK", "EV-06-26"],
                        "citation_refs": ["[202]"],
                        "supporting_facts": ["Salesforce disclosed workflow deployments."],
                        "evidence_backed": True,
                    }
                ],
            }
        ],
        source_registry=[
                {
                    "ref": "[7]",
                    "evidence_id": "EV-OK",
                    "title": "Workflow deployment source",
                    "url": "https://www.salesforce.com/news/workflow",
                    "source_level": "B",
                }
        ],
    )

    manifest = output["citation_manifest"]
    assert manifest["citation_manifest_status"] == "ok"
    assert manifest["missing_source_ref_count"] == 0
    assert manifest["filtered_unresolved_ref_count"] == 0
    assert output["ref_lineage_diagnostics"]["filtered_unresolved_ref_count"] >= 1
    assert "EV-06-26" not in output["report_markdown"]
    assert "[202]" not in output["report_markdown"]


def test_final_writer_drops_factual_section_when_all_refs_unresolved(monkeypatch):
    monkeypatch.setenv("REPORT_FINAL_WRITER_SOURCE_APPENDIX", "true")

    output = run_final_writer_agent(
        query="AI Agent",
        report_blueprint={
            "report_shell": {"front_blocks": [], "back_blocks": ["appendix"]},
            "chapters": [{"chapter_id": "ch_01", "chapter_title": "Deployment"}],
        },
        chapter_packages=[
            {
                "chapter_id": "ch_01",
                "chapter_title": "Deployment",
                "sections": [
                    {
                        "section_id": "s_unresolved",
                        "section_title": "OpenAI deployment",
                        "claim": "OpenAI 在 2026 年发布企业级 Agent 收入数据。",
                        "reasoning": "OpenAI 在 2026 年发布企业级 Agent 收入数据。",
                        "mechanism": "该收入数据会影响商业化判断。",
                        "used_fact_refs": ["EV-MISSING"],
                        "evidence_refs": ["EV-MISSING"],
                        "supporting_facts": ["OpenAI 在 2026 年发布企业级 Agent 收入数据。"],
                        "render_blocks": [
                            {"type": "paragraph", "text": "OpenAI 在 2026 年发布企业级 Agent 收入数据。"}
                        ],
                        "evidence_backed": True,
                    }
                ],
            }
        ],
        source_registry=[],
    )

    assert "OpenAI 在 2026 年发布企业级 Agent 收入数据" not in output["report_markdown"]
    support = output["source_claim_support"]
    assert support["factual_section_without_resolved_ref_count"] == 1
    assert support["section_dropped_due_to_unresolved_refs_count"] == 1
    assert support["citationless_fact_examples"]


def test_final_writer_drops_metric_claim_when_metric_fact_is_not_structured(monkeypatch):
    monkeypatch.setenv("REPORT_FINAL_WRITER_SOURCE_APPENDIX", "true")

    output = run_final_writer_agent(
        query="AI Agent",
        report_blueprint={
            "report_shell": {"front_blocks": [], "back_blocks": ["appendix"]},
            "chapters": [{"chapter_id": "ch_01", "chapter_title": "Market sizing"}],
        },
        chapter_packages=[
            {
                "chapter_id": "ch_01",
                "chapter_title": "Market sizing",
                "sections": [
                    {
                        "section_id": "s_metric",
                        "section_title": "Market size",
                        "block_type": "metric_reconciliation",
                        "claim": "The AI Agent market is broad: the global market is expected to reach 2168 billion dollars by 2035, with a CAGR of 40.15%.",
                        "reasoning": "The forecast implies a strong market-size conclusion.",
                        "used_fact_refs": ["EV-WEAK"],
                        "evidence_refs": ["EV-WEAK"],
                        "supporting_facts": ["A generic industry article mentioned AI Agent opportunities without structured metric fields."],
                        "evidence_basis": ["A generic industry article mentioned AI Agent opportunities without structured metric fields."],
                        "render_blocks": [
                            {
                                "type": "paragraph",
                                "text": "The AI Agent market is broad: the global market is expected to reach 2168 billion dollars by 2035, with a CAGR of 40.15%.",
                            }
                        ],
                        "evidence_backed": True,
                    }
                ],
            }
        ],
        source_registry=[
            {
                "ref": "EV-WEAK",
                "evidence_id": "EV-WEAK",
                "title": "Generic AI Agent opportunities article",
                "url": "https://example.org/generic-agent-opportunities",
                "source_level": "C",
                "source_verification_status": "search_result_only",
            }
        ],
    )

    assert "2168 billion dollars" not in output["report_markdown"]
    support = output["source_claim_support"]
    assert support["metric_claim_without_metric_fact_count"] == 1
    assert support["section_dropped_due_to_source_claim_mismatch_count"] == 1


def test_final_writer_keeps_metric_claim_when_structured_metric_lives_in_evidence_package(monkeypatch):
    monkeypatch.setenv("REPORT_FINAL_WRITER_SOURCE_APPENDIX", "true")

    output = run_final_writer_agent(
        query="AI Agent",
        report_blueprint={
            "report_shell": {"front_blocks": [], "back_blocks": ["appendix"]},
            "chapters": [{"chapter_id": "ch_01", "chapter_title": "Market sizing"}],
        },
        chapter_packages=[
            {
                "chapter_id": "ch_01",
                "chapter_title": "Market sizing",
                "sections": [
                    {
                        "section_id": "s_metric",
                        "section_title": "Market size",
                        "block_type": "metric_reconciliation",
                        "claim": "The AI Agent market size reached 8.2 billion yuan in 2024.",
                        "reasoning": "The market size metric provides a comparable baseline for demand-space analysis.",
                        "used_fact_refs": ["EV-METRIC"],
                        "evidence_refs": ["EV-METRIC"],
                        "supporting_facts": ["The AI Agent market size reached 8.2 billion yuan in 2024."],
                        "render_blocks": [
                            {
                                "type": "paragraph",
                                "text": "The AI Agent market size reached 8.2 billion yuan in 2024.",
                            }
                        ],
                        "evidence_backed": True,
                    }
                ],
            }
        ],
        source_registry=[
            {
                "ref": "[9]",
                "title": "AI Agent market size disclosure",
                "url": "https://www.salesforce.com/news/ai-agent-market-size",
                "source_level": "B",
                "evidence_refs": ["EV-METRIC"],
            }
        ],
        evidence_package={
            "analysis_ready_evidence": [
                {
                    "evidence_id": "EV-METRIC",
                    "source_ref": "[9]",
                    "source_url": "https://www.salesforce.com/news/ai-agent-market-size",
                    "source_title": "AI Agent market size disclosure",
                    "source_level": "B",
                    "metric": "market size",
                    "value": "8.2",
                    "unit": "billion yuan",
                    "period": "2024",
                    "fact_type": "metric",
                }
            ],
            "source_registry": [
                {
                    "ref": "[9]",
                    "title": "AI Agent market size disclosure",
                    "url": "https://www.salesforce.com/news/ai-agent-market-size",
                    "source_level": "B",
                    "evidence_refs": ["EV-METRIC"],
                }
            ],
        },
    )

    assert "8.2 billion yuan" in output["report_markdown"]
    assert "[1]" in output["report_markdown"]
    assert output["source_claim_support"]["metric_claim_without_metric_fact_count"] == 0
    assert output["final_citation_audit"]["final_citation_reconciliation_status"] == "ok"


def test_final_writer_removes_unresolved_final_body_citation_before_appendix(monkeypatch):
    monkeypatch.setenv("REPORT_FINAL_WRITER_SOURCE_APPENDIX", "true")

    output = run_final_writer_agent(
        query="AI Agent",
        report_blueprint={
            "report_shell": {"front_blocks": [], "back_blocks": ["appendix"]},
            "chapters": [{"chapter_id": "ch_01", "chapter_title": "Demand validation"}],
        },
        chapter_packages=[
            {
                "chapter_id": "ch_01",
                "chapter_title": "Demand validation",
                "sections": [
                    {
                        "section_id": "s1",
                        "section_title": "Deployment",
                        "claim": "Workflow deployment is visible [8].",
                        "reasoning": "A production workflow requires integration and permissions [8].",
                        "used_fact_refs": ["EV-OK"],
                        "evidence_refs": ["EV-OK"],
                        "citation_refs": ["[8]"],
                        "render_blocks": [{"type": "paragraph", "text": "Workflow deployment is visible [8]."}],
                        "supporting_facts": ["Salesforce disclosed workflow deployments."],
                        "evidence_backed": True,
                    }
                ],
            }
        ],
        source_registry=[
            {
                "ref": "EV-OK",
                "evidence_id": "EV-OK",
                "title": "Workflow deployment source",
                "url": "https://www.salesforce.com/news/workflow",
                "source_level": "B",
            }
        ],
    )

    markdown = output["report_markdown"]
    body, appendix = markdown.split("##", 1) if "##" in markdown else (markdown, "")
    assert "[8]" not in markdown
    assert "[1]" in markdown
    assert output["final_citation_audit"]["final_unresolved_citation_removed_count"] == 0
    assert set(ref for ref in output["final_citation_audit"]["final_body_citation_refs"]) == set(
        ref for ref in output["final_citation_audit"]["final_appendix_refs"]
    )


def test_final_writer_drops_factual_section_when_manifest_filters_its_only_source(monkeypatch):
    monkeypatch.setenv("REPORT_FINAL_WRITER_SOURCE_APPENDIX", "true")

    output = run_final_writer_agent(
        query="AI Agent",
        report_blueprint={
            "report_shell": {"front_blocks": [], "back_blocks": ["appendix"]},
            "chapters": [{"chapter_id": "ch_01", "chapter_title": "Demand validation"}],
        },
        chapter_packages=[
            {
                "chapter_id": "ch_01",
                "chapter_title": "Demand validation",
                "sections": [
                    {
                        "section_id": "s1",
                        "section_title": "Deployment",
                        "claim": "Salesforce disclosed Agentforce workflow deployments.",
                        "reasoning": "The deployment evidence supports the customer workflow signal.",
                        "used_fact_refs": ["EV-BAD"],
                        "evidence_refs": ["EV-BAD"],
                        "render_blocks": [
                            {
                                "type": "paragraph",
                                "text": "Salesforce disclosed Agentforce workflow deployments [268].",
                            }
                        ],
                        "supporting_facts": ["Salesforce disclosed Agentforce workflow deployments."],
                        "evidence_backed": True,
                    }
                ],
            }
        ],
        source_registry=[
            {
                "ref": "EV-BAD",
                "evidence_id": "EV-BAD",
                "title": "Salesforce official statistics show workflow deployments",
                "url": "https://www.salesforce.com/news/workflow",
                "source_level": "B",
            }
        ],
    )

    assert "workflow deployments" not in output["report_markdown"]
    support = output["source_claim_support"]
    assert support["section_dropped_due_to_source_claim_mismatch_count"] == 1
    assert support["source_claim_mismatch_examples"][0]["reason"] == "manifest_citation_missing"
    assert output["final_citation_audit"]["factual_body_without_citations_count"] == 0


def test_final_writer_drops_evidence_backed_section_without_manifest_citation(monkeypatch):
    monkeypatch.setenv("REPORT_FINAL_WRITER_SOURCE_APPENDIX", "true")

    output = run_final_writer_agent(
        query="AI Agent",
        report_blueprint={
            "report_shell": {"front_blocks": [], "back_blocks": ["appendix"]},
            "chapters": [{"chapter_id": "ch_01", "chapter_title": "Technology maturity"}],
        },
        chapter_packages=[
            {
                "chapter_id": "ch_01",
                "chapter_title": "Technology maturity",
                "sections": [
                    {
                        "section_id": "s_tech",
                        "section_title": "Technology constraint",
                        "block_type": "technology_maturity",
                        "claim": "Technology maturity constrains production deployment.",
                        "reasoning": "Tool calls, permission control, and integration cost determine whether the workflow can enter production.",
                        "render_blocks": [
                            {
                                "type": "paragraph",
                                "text": "Technology maturity constrains production deployment.",
                            }
                        ],
                        "supporting_facts": ["A technical maturity observation without a resolvable citation."],
                        "evidence_backed": True,
                    }
                ],
            }
        ],
        source_registry=[],
    )

    assert "Technology maturity constrains" not in output["report_markdown"]
    assert output["source_claim_support"]["source_claim_mismatch_examples"][0]["reason"] == "manifest_citation_missing"
    assert output["final_citation_audit"]["factual_body_without_citations_count"] == 0


def test_final_writer_recomputes_citation_audit_after_public_gate(monkeypatch):
    from rag_pipeline.agents import final_writer_agent

    monkeypatch.setenv("REPORT_FINAL_WRITER_SOURCE_APPENDIX", "true")

    def fake_public_gate(markdown: str):
        cleaned = str(markdown or "").replace("2026 market size reached 10 billion.", "")
        return cleaned, {
            "public_narrative_leak_input_count": 1,
            "public_narrative_leak_remaining_count": 0,
            "public_narrative_leak_removed_count": 1,
            "public_narrative_leak_reason_counts": {"test": 1},
            "public_narrative_leak_examples": ["2026 market size reached 10 billion."],
            "public_narrative_leak_remaining_examples": [],
        }

    monkeypatch.setattr(final_writer_agent, "apply_public_narrative_gate", fake_public_gate)

    output = run_final_writer_agent(
        query="AI Agent",
        report_blueprint={
            "report_shell": {"front_blocks": [], "back_blocks": []},
            "chapters": [{"chapter_id": "ch_01", "chapter_title": "Market"}],
        },
        chapter_packages=[
            {
                "chapter_id": "ch_01",
                "chapter_title": "Market",
                "sections": [
                    {
                        "section_id": "s1",
                        "section_title": "Market signal",
                        "claim": "2026 market size reached 10 billion.",
                        "reasoning": "2026 market size reached 10 billion.",
                        "render_blocks": [{"type": "paragraph", "text": "2026 market size reached 10 billion."}],
                        "evidence_backed": False,
                    }
                ],
            }
        ],
        source_registry=[],
    )

    assert "2026 market size reached 10 billion" not in output["report_markdown"]
    assert output["final_citation_audit"]["final_citation_reconciliation_status"] == "ok"
    assert output["final_citation_audit"]["factual_body_without_citations_count"] == 0


def test_final_writer_preserves_analysis_claim_from_single_company_source(monkeypatch):
    monkeypatch.setenv("REPORT_FINAL_WRITER_SOURCE_APPENDIX", "true")

    output = run_final_writer_agent(
        query="AI Agent",
        report_blueprint={
            "report_shell": {"front_blocks": [], "back_blocks": ["appendix"]},
            "chapters": [{"chapter_id": "ch_01", "chapter_title": "Demand validation"}],
        },
        chapter_packages=[
            {
                "chapter_id": "ch_01",
                "chapter_title": "Demand validation",
                "sections": [
                    {
                        "section_id": "s_broad",
                        "section_title": "Demand signal",
                        "block_type": "case_comparison",
                        "claim": "AI Agent has formed verifiable demand across the education field and covers teaching, learning, and management scenarios.",
                        "reasoning": "A single company Q&A discloses a product feature list for one education deployment.",
                        "used_fact_refs": ["EV-QA"],
                        "evidence_refs": ["EV-QA"],
                        "render_blocks": [
                            {
                                "type": "paragraph",
                                "text": "AI Agent has formed verifiable demand across the education field and covers teaching, learning, and management scenarios.",
                            }
                        ],
                        "supporting_facts": [
                            {
                                "evidence_id": "EV-QA",
                                "distilled_fact": "A single company Q&A discloses AI Agent product functions for one education deployment.",
                                "source_title": "Investor relations Q&A",
                            }
                        ],
                        "evidence_backed": True,
                    }
                ],
            }
        ],
        source_registry=[
            {
                "ref": "EV-QA",
                "evidence_id": "EV-QA",
                "title": "Investor relations Q&A",
                "url": "https://irm.cninfo.com.cn/ircs/question/questionDetail?questionId=1",
                "source_level": "B",
                "source_verification_status": "readpage_verified",
            }
        ],
    )

    assert "formed verifiable demand across the education field" in output["report_markdown"]
    assert output["source_claim_support"]["weak_source_strong_claim_demoted_count"] == 0
    assert output["source_claim_support"]["demoted_section_count"] == 0
    assert output["source_claim_support"]["section_dropped_due_to_source_claim_mismatch_count"] == 0


def test_final_writer_reports_analysis_claim_to_section_transfer(monkeypatch):
    monkeypatch.setenv("REPORT_FINAL_WRITER_SOURCE_APPENDIX", "true")

    output = run_final_writer_agent(
        query="AI Agent",
        report_blueprint={
            "report_shell": {"front_blocks": [], "back_blocks": ["appendix"]},
            "chapters": [{"chapter_id": "ch_01", "chapter_title": "Demand validation"}],
        },
        claim_units=[
            {
                "claim_id": "claim-rendered",
                "chapter_id": "ch_01",
                "claim": "Salesforce disclosed an enterprise AI Agent pilot.",
                "used_fact_refs": ["EV-1"],
                "evidence_refs": ["EV-1"],
                "source_support_map": {"claim": ["EV-1"], "mechanism": ["EV-1"], "boundary": ["EV-1"]},
                "claim_strength": "directional",
                "analysis_role": "directional",
            },
            {
                "claim_id": "claim-not-rendered",
                "chapter_id": "ch_01",
                "claim": "A second analyzed claim exists but no public section consumes it.",
                "used_fact_refs": ["EV-2"],
                "evidence_refs": ["EV-2"],
                "source_support_map": {"claim": ["EV-2"]},
                "claim_strength": "directional",
                "analysis_role": "directional",
            },
        ],
        chapter_packages=[
            {
                "chapter_id": "ch_01",
                "chapter_title": "Demand validation",
                "sections": [
                    {
                        "section_id": "s_rendered",
                        "section_title": "Enterprise pilot signal",
                        "claim_id": "claim-rendered",
                        "analysis_role": "directional",
                        "claim_strength": "directional",
                        "source_support_map": {"claim": ["EV-1"], "mechanism": ["EV-1"], "boundary": ["EV-1"]},
                        "claim": "Salesforce disclosed an enterprise AI Agent pilot.",
                        "reasoning": "The disclosure indicates that enterprise pilots are available as directional demand evidence.",
                        "used_fact_refs": ["EV-1"],
                        "evidence_refs": ["EV-1"],
                        "render_blocks": [
                            {
                                "type": "paragraph",
                                "text": "Salesforce disclosed an enterprise AI Agent pilot as directional demand evidence.",
                            }
                        ],
                        "supporting_facts": [
                            {
                                "evidence_id": "EV-1",
                                "distilled_fact": "Salesforce disclosed an enterprise AI Agent pilot.",
                                "source_title": "Salesforce enterprise AI Agent pilot note",
                            }
                        ],
                        "evidence_backed": True,
                    }
                ],
            }
        ],
        source_registry=[
            {
                "ref": "EV-1",
                "evidence_id": "EV-1",
                "title": "Salesforce enterprise AI Agent pilot note",
                "url": "https://www.salesforce.com/news/agent-pilot",
                "source_level": "C",
                "source_verification_status": "search_result_only",
            },
            {
                "ref": "EV-2",
                "evidence_id": "EV-2",
                "title": "Salesforce second pilot note",
                "url": "https://www.salesforce.com/news/agent-pilot-2",
                "source_level": "C",
                "source_verification_status": "search_result_only",
            },
        ],
    )

    transfer = output["analysis_transfer"]
    assert transfer["analysis_claim_count"] == 2
    assert transfer["rendered_analysis_section_count"] == 1
    assert transfer["claim_lost_after_analysis_count"] == 1
    assert transfer["claim_to_section_transfer_rate"] == 0.5
    assert transfer["analysis_claim_ids_rendered"] == ["claim-rendered"]
    assert transfer["claim_lost_after_analysis_reasons"] == {"not_rendered_in_public_sections": 1}


def test_final_writer_counts_analysis_transfer_by_refs_when_section_loses_claim_id(monkeypatch):
    monkeypatch.setenv("REPORT_FINAL_WRITER_SOURCE_APPENDIX", "true")

    output = run_final_writer_agent(
        query="AI Agent",
        report_blueprint={
            "report_shell": {"front_blocks": [], "back_blocks": ["appendix"]},
            "chapters": [{"chapter_id": "ch_01", "chapter_title": "Demand validation"}],
        },
        analysis_claim_units=[
            {
                "claim_id": "claim-ref-rendered",
                "chapter_id": "ch_01",
                "claim": "Traceable deployment evidence supports a directional demand claim.",
                "used_fact_refs": ["EV-REF-ONLY"],
                "evidence_refs": ["EV-REF-ONLY"],
                "source_support_map": {"claim": ["EV-REF-ONLY"]},
                "claim_strength": "directional",
                "analysis_role": "directional",
            }
        ],
        chapter_packages=[
            {
                "chapter_id": "ch_01",
                "chapter_title": "Demand validation",
                "sections": [
                    {
                        "section_id": "s_ref_rendered",
                        "section_title": "Directional demand signal",
                        "analysis_role": "directional",
                        "claim_strength": "directional",
                        "source_support_map": {"claim": ["EV-REF-ONLY"]},
                        "claim": "Traceable deployment evidence supports a directional demand claim.",
                        "reasoning": "The evidence indicates a deployment signal that can support directional analysis.",
                        "used_fact_refs": ["EV-REF-ONLY"],
                        "evidence_refs": ["EV-REF-ONLY"],
                        "render_blocks": [
                            {
                                "type": "paragraph",
                                "text": "Traceable deployment evidence supports a directional demand claim.",
                            }
                        ],
                        "supporting_facts": [
                            {
                                "evidence_id": "EV-REF-ONLY",
                                "distilled_fact": "Traceable deployment evidence supports a directional demand claim.",
                                "source_title": "Traceable deployment note",
                            }
                        ],
                        "evidence_backed": True,
                    }
                ],
            }
        ],
        source_registry=[
            {
                "ref": "EV-REF-ONLY",
                "evidence_id": "EV-REF-ONLY",
                "title": "Salesforce AI Agent deployment note",
                "url": "https://www.salesforce.com/news/ai-agent-deployment",
                "source_level": "C",
                "source_verification_status": "search_result_only",
            }
        ],
    )

    transfer = output["analysis_transfer"]
    assert transfer["rendered_analysis_claim_count"] == 1
    assert transfer["claim_to_section_transfer_rate"] == 1.0
    assert transfer["analysis_claim_ids_rendered"] == ["claim-ref-rendered"]


def test_final_writer_balanced_gate_preserves_weak_source_claim_after_analysis(monkeypatch):
    monkeypatch.setenv("REPORT_FINAL_WRITER_SOURCE_APPENDIX", "true")
    monkeypatch.delenv("REPORT_SOURCE_CLAIM_GATE_MODE", raising=False)

    output = run_final_writer_agent(
        query="AI Agent",
        report_blueprint={
            "report_shell": {"front_blocks": [], "back_blocks": ["appendix"]},
            "chapters": [{"chapter_id": "ch_01", "chapter_title": "Demand validation"}],
        },
        chapter_packages=[
            {
                "chapter_id": "ch_01",
                "chapter_title": "Demand validation",
                "sections": [
                    {
                        "section_id": "s_weak",
                        "section_title": "Demand signal",
                        "block_type": "case_comparison",
                        "claim": "The AI Agent market has a strong and broad enterprise adoption signal.",
                        "reasoning": "A search-result-only source describes one enterprise pilot.",
                        "used_fact_refs": ["EV-WEAK"],
                        "evidence_refs": ["EV-WEAK"],
                        "render_blocks": [
                            {
                                "type": "paragraph",
                                "text": "The AI Agent market has a strong and broad enterprise adoption signal.",
                            }
                        ],
                        "supporting_facts": [
                            {
                                "evidence_id": "EV-WEAK",
                                "distilled_fact": "A search-result-only source describes one enterprise AI Agent pilot.",
                                "source_title": "Enterprise AI Agent pilot note",
                            }
                        ],
                        "evidence_backed": True,
                    }
                ],
            }
        ],
        source_registry=[
            {
                "ref": "EV-WEAK",
                "evidence_id": "EV-WEAK",
                "title": "Enterprise AI Agent pilot note",
                "url": "https://www.salesforce.com/news/agent-pilot",
                "source_level": "C",
                "source_verification_status": "search_result_only",
            }
        ],
    )

    markdown = output["report_markdown"]
    assert "strong and broad enterprise adoption" in markdown
    assert "directional signal" not in markdown
    assert "Enterprise AI Agent pilot note" in markdown
    support = output["source_claim_support"]
    assert support["source_gate_mode"] == "balanced"
    assert support["demoted_section_count"] == 0
    assert support["section_dropped_due_to_source_claim_mismatch_count"] == 0
    assert support["hard_dropped_section_count"] == 0


def test_final_writer_preserves_analyzed_directional_claim_from_traceable_c_source(monkeypatch):
    monkeypatch.setenv("REPORT_FINAL_WRITER_SOURCE_APPENDIX", "true")
    monkeypatch.setenv("REPORT_SOURCE_CLAIM_GATE_MODE", "balanced")

    output = run_final_writer_agent(
        query="AI Agent",
        report_blueprint={
            "report_shell": {"front_blocks": [], "back_blocks": ["appendix"]},
            "chapters": [{"chapter_id": "ch_01", "chapter_title": "Demand validation"}],
        },
        chapter_packages=[
            {
                "chapter_id": "ch_01",
                "chapter_title": "Demand validation",
                "sections": [
                    {
                        "section_id": "s_directional",
                        "section_title": "Enterprise pilot signal",
                        "block_type": "case_comparison",
                        "claim": "The AI Agent market has a strong and broad enterprise adoption signal.",
                        "reasoning": "The analysis layer classified this as directional because it comes from one traceable C-level source.",
                        "claim_strength": "directional",
                        "analysis_role": "directional",
                        "used_fact_refs": ["EV-WEAK"],
                        "evidence_refs": ["EV-WEAK"],
                        "source_support_map": {
                            "claim": ["EV-WEAK"],
                            "mechanism": ["EV-WEAK"],
                            "boundary": ["EV-WEAK"],
                        },
                        "render_blocks": [
                            {
                                "type": "paragraph",
                                "text": "The AI Agent market has a strong and broad enterprise adoption signal.",
                            }
                        ],
                        "supporting_facts": [
                            {
                                "evidence_id": "EV-WEAK",
                                "distilled_fact": "A search-result-only source describes one enterprise AI Agent pilot.",
                                "source_title": "Enterprise AI Agent pilot note",
                            }
                        ],
                        "evidence_backed": True,
                    }
                ],
            }
        ],
        source_registry=[
            {
                "ref": "EV-WEAK",
                "evidence_id": "EV-WEAK",
                "title": "Enterprise AI Agent pilot note",
                "url": "https://www.salesforce.com/news/agent-pilot",
                "source_level": "C",
                "source_verification_status": "search_result_only",
            }
        ],
    )

    markdown = output["report_markdown"]
    assert "strong and broad enterprise adoption signal" in markdown
    assert "directional signal" not in markdown
    assert output["source_claim_support"]["demoted_section_count"] == 0
    assert output["source_claim_support"]["section_dropped_due_to_source_claim_mismatch_count"] == 0


def test_final_writer_strict_gate_preserves_weak_source_claim_after_analysis(monkeypatch):
    monkeypatch.setenv("REPORT_FINAL_WRITER_SOURCE_APPENDIX", "true")
    monkeypatch.setenv("REPORT_SOURCE_CLAIM_GATE_MODE", "strict")

    output = run_final_writer_agent(
        query="AI Agent",
        report_blueprint={
            "report_shell": {"front_blocks": [], "back_blocks": ["appendix"]},
            "chapters": [{"chapter_id": "ch_01", "chapter_title": "Demand validation"}],
        },
        chapter_packages=[
            {
                "chapter_id": "ch_01",
                "chapter_title": "Demand validation",
                "sections": [
                    {
                        "section_id": "s_weak",
                        "section_title": "Demand signal",
                        "block_type": "case_comparison",
                        "claim": "The AI Agent market has a strong enterprise adoption signal.",
                        "reasoning": "A search-result-only source describes one enterprise pilot.",
                        "used_fact_refs": ["EV-WEAK"],
                        "evidence_refs": ["EV-WEAK"],
                        "render_blocks": [{"type": "paragraph", "text": "The AI Agent market has a strong enterprise adoption signal."}],
                        "supporting_facts": ["A search-result-only source describes one enterprise AI Agent pilot."],
                        "evidence_backed": True,
                    }
                ],
            }
        ],
        source_registry=[
            {
                "ref": "EV-WEAK",
                "evidence_id": "EV-WEAK",
                "title": "Enterprise AI Agent pilot note",
                "url": "https://www.salesforce.com/news/agent-pilot",
                "source_level": "C",
                "source_verification_status": "search_result_only",
            }
        ],
    )

    assert "strong enterprise adoption" in output["report_markdown"]
    support = output["source_claim_support"]
    assert support["source_gate_mode"] == "strict"
    assert support["section_dropped_due_to_source_claim_mismatch_count"] == 0
    assert support["hard_dropped_section_count"] == 0


def test_final_writer_recovers_citation_from_supporting_fact_source_url(monkeypatch):
    monkeypatch.setenv("REPORT_FINAL_WRITER_SOURCE_APPENDIX", "true")

    output = run_final_writer_agent(
        query="AI Agent",
        report_blueprint={
            "report_shell": {"front_blocks": [], "back_blocks": ["appendix"]},
            "chapters": [{"chapter_id": "ch_01", "chapter_title": "Deployment"}],
        },
        chapter_packages=[
            {
                "chapter_id": "ch_01",
                "chapter_title": "Deployment",
                "sections": [
                    {
                        "section_id": "s_source_url_fact",
                        "section_title": "Workflow deployment",
                        "block_type": "case_comparison",
                        "claim": "Salesforce disclosed Agentforce workflow deployments in 2025.",
                        "reasoning": "The disclosed deployment supports an enterprise workflow adoption signal.",
                        "render_blocks": [
                            {
                                "type": "paragraph",
                                "text": "Salesforce disclosed Agentforce workflow deployments in 2025.",
                            }
                        ],
                        "supporting_facts": [
                            {
                                "distilled_fact": "Salesforce disclosed Agentforce workflow deployments in 2025.",
                                "source_title": "Salesforce Agentforce workflow deployments",
                                "source_url": "https://www.salesforce.com/news/agentforce-workflows",
                            }
                        ],
                        "evidence_backed": True,
                    }
                ],
            }
        ],
        source_registry=[
            {
                "ref": "SRC-SF",
                "title": "Salesforce Agentforce workflow deployments",
                "url": "https://www.salesforce.com/news/agentforce-workflows",
                "source_level": "B",
                "source_verification_status": "readpage_verified",
            }
        ],
    )

    markdown = output["report_markdown"]
    assert "Workflow deployment" in markdown
    assert "[1]" in markdown
    assert "Salesforce Agentforce workflow deployments" in markdown
    assert output["source_claim_support"]["section_dropped_due_to_source_claim_mismatch_count"] == 0
    assert output["ref_lineage_diagnostics"]["section_ref_recovered_count"] == 1


def test_final_writer_omits_chapter_when_all_sections_dropped_after_citation_gate(monkeypatch):
    monkeypatch.setenv("REPORT_FINAL_WRITER_SOURCE_APPENDIX", "true")

    output = run_final_writer_agent(
        query="AI Agent",
        report_blueprint={
            "report_shell": {"front_blocks": [], "back_blocks": ["appendix"]},
            "chapters": [{"chapter_id": "ch_01", "chapter_title": "Technology maturity"}],
        },
        chapter_packages=[
            {
                "chapter_id": "ch_01",
                "chapter_title": "Technology maturity",
                "lead": "Technology maturity should be judged together with verifiable facts.",
                "sections": [
                    {
                        "section_id": "s_tech",
                        "section_title": "Technology constraint",
                        "block_type": "technology_maturity",
                        "claim": "Technology maturity constrains production deployment.",
                        "reasoning": "Tool calls and permission control determine whether the workflow can enter production.",
                        "render_blocks": [{"type": "paragraph", "text": "Technology maturity constrains production deployment."}],
                        "supporting_facts": ["A technical maturity observation without a resolvable citation."],
                        "evidence_backed": True,
                    }
                ],
            }
        ],
        source_registry=[],
    )

    markdown = output["report_markdown"]
    assert "## 1. Technology maturity" not in markdown
    assert "Technology maturity should be judged" not in markdown
    assert output["source_claim_support"]["empty_chapter_omitted_after_source_gate_count"] == 1


def test_final_writer_explains_unresolved_ref_when_source_was_excluded(monkeypatch):
    monkeypatch.setenv("REPORT_FINAL_WRITER_SOURCE_APPENDIX", "true")

    output = run_final_writer_agent(
        query="AI Agent",
        report_blueprint={
            "report_shell": {"front_blocks": [], "back_blocks": ["appendix"]},
            "chapters": [{"chapter_id": "ch_01", "chapter_title": "Demand validation"}],
        },
        chapter_packages=[
            {
                "chapter_id": "ch_01",
                "chapter_title": "Demand validation",
                "sections": [
                    {
                        "section_id": "s_fake_source",
                        "section_title": "Deployment",
                        "claim": "Enterprise AI Agent demand is moving into workflow pilots.",
                        "reasoning": "The cited source is intentionally a placeholder and must not support public prose.",
                        "used_fact_refs": ["S-PLACEHOLDER"],
                        "evidence_refs": ["S-PLACEHOLDER"],
                        "render_blocks": [
                            {
                                "type": "paragraph",
                                "text": "Enterprise AI Agent demand is moving into workflow pilots.",
                            }
                        ],
                        "supporting_facts": ["Enterprise AI Agent demand is moving into workflow pilots."],
                        "evidence_backed": True,
                    }
                ],
            }
        ],
        source_registry=[
            {
                "ref": "S-PLACEHOLDER",
                "evidence_id": "S-PLACEHOLDER",
                "title": "Placeholder source",
                "url": "https://example.com/ai-agent-placeholder",
                "source_level": "B",
            }
        ],
    )

    reasons = output["citation_manifest"].get("filtered_unresolved_ref_reasons") or []
    assert reasons
    assert reasons[0]["ref"] == "S-PLACEHOLDER"
    assert reasons[0]["reason"] == "fake_or_placeholder_source"


def test_final_writer_resolves_refs_from_nested_chapter_evidence_packages(monkeypatch):
    monkeypatch.setenv("REPORT_FINAL_WRITER_SOURCE_APPENDIX", "true")

    output = run_final_writer_agent(
        query="AI Agent",
        report_blueprint={
            "report_shell": {"front_blocks": [], "back_blocks": ["appendix"]},
            "chapters": [{"chapter_id": "ch_01", "chapter_title": "Demand validation"}],
        },
        chapter_packages=[
            {
                "chapter_id": "ch_01",
                "chapter_title": "Demand validation",
                "sections": [
                    {
                        "section_id": "s1",
                        "section_title": "Deployment",
                        "claim": "Salesforce disclosed Agentforce workflow deployments.",
                        "reasoning": "The deployment evidence supports the customer workflow signal.",
                        "used_fact_refs": ["EV-NESTED"],
                        "evidence_refs": ["EV-NESTED"],
                        "render_blocks": [
                            {
                                "type": "paragraph",
                                "text": "Salesforce disclosed Agentforce workflow deployments.",
                            }
                        ],
                        "supporting_facts": ["Salesforce disclosed Agentforce workflow deployments."],
                        "evidence_backed": True,
                    }
                ],
            }
        ],
        evidence_package={
            "chapter_evidence_packages": [
                {
                    "chapter_id": "ch_01",
                    "evidence_items": [
                        {
                            "evidence_id": "EV-NESTED",
                            "ref": "EV-NESTED",
                            "source_ref": "[75]",
                            "source_title": "Salesforce Agentforce customer workflow deployments",
                            "source_url": "https://www.salesforce.com/news/agentforce-workflows",
                            "source_level": "B",
                        }
                    ],
                }
            ]
        },
    )

    assert "workflow deployments" in output["report_markdown"]
    assert "[1]" in output["report_markdown"]
    assert "EV-NESTED" not in output["report_markdown"]
    assert output["ref_lineage_diagnostics"]["filtered_unresolved_ref_count"] == 0
    assert output["final_citation_audit"]["final_citation_reconciliation_status"] == "ok"


def test_final_writer_keeps_non_metric_case_section_with_year(monkeypatch):
    monkeypatch.setenv("REPORT_FINAL_WRITER_SOURCE_APPENDIX", "true")

    output = run_final_writer_agent(
        query="AI Agent",
        report_blueprint={
            "report_shell": {"front_blocks": [], "back_blocks": ["appendix"]},
            "chapters": [{"chapter_id": "ch_01", "chapter_title": "Demand validation"}],
        },
        chapter_packages=[
            {
                "chapter_id": "ch_01",
                "chapter_title": "Demand validation",
                "sections": [
                    {
                        "section_id": "s_case",
                        "section_title": "Workflow deployment",
                        "block_type": "case_comparison",
                        "claim": "Salesforce described Agentforce workflow deployments in 2026 customer operations.",
                        "reasoning": "The case points to deployment depth rather than a market-size metric.",
                        "used_fact_refs": ["EV-CASE-YEAR"],
                        "evidence_refs": ["EV-CASE-YEAR"],
                        "render_blocks": [
                            {
                                "type": "paragraph",
                                "text": "Salesforce described Agentforce workflow deployments in 2026 customer operations.",
                            }
                        ],
                        "supporting_facts": [
                            "Salesforce described Agentforce workflow deployments in 2026 customer operations."
                        ],
                        "evidence_backed": True,
                    }
                ],
            }
        ],
        source_registry=[
            {
                "ref": "EV-CASE-YEAR",
                "evidence_id": "EV-CASE-YEAR",
                "title": "Salesforce Agentforce workflow deployments",
                "url": "https://www.salesforce.com/news/agentforce-workflows",
                "source_level": "B",
            }
        ],
    )

    assert "workflow deployments" in output["report_markdown"]
    assert output["source_claim_support"]["metric_claim_without_metric_fact_count"] == 0
    assert output["final_citation_audit"]["final_citation_reconciliation_status"] == "ok"


def test_final_writer_keeps_case_section_with_metric_boundary_but_no_metric_claim(monkeypatch):
    monkeypatch.setenv("REPORT_FINAL_WRITER_SOURCE_APPENDIX", "true")

    output = run_final_writer_agent(
        query="AI Agent",
        report_blueprint={
            "report_shell": {"front_blocks": [], "back_blocks": ["appendix"]},
            "chapters": [{"chapter_id": "ch_01", "chapter_title": "Demand validation"}],
        },
        chapter_packages=[
            {
                "chapter_id": "ch_01",
                "chapter_title": "Demand validation",
                "sections": [
                    {
                        "section_id": "s_case",
                        "section_title": "Workflow deployment",
                        "block_type": "case_comparison",
                        "claim": "Salesforce described Agentforce workflow deployments for customer service.",
                        "reasoning": "The case supports deployment depth in a specific workflow.",
                        "counter_evidence": "The evidence does not by itself prove total market size.",
                        "used_fact_refs": ["EV-CASE-BOUNDARY"],
                        "evidence_refs": ["EV-CASE-BOUNDARY"],
                        "render_blocks": [
                            {
                                "type": "paragraph",
                                "text": "Salesforce described Agentforce workflow deployments for customer service.",
                            }
                        ],
                        "supporting_facts": [
                            "Salesforce described Agentforce workflow deployments for customer service."
                        ],
                        "evidence_backed": True,
                    }
                ],
            }
        ],
        source_registry=[
            {
                "ref": "EV-CASE-BOUNDARY",
                "evidence_id": "EV-CASE-BOUNDARY",
                "title": "Salesforce Agentforce workflow deployments",
                "url": "https://www.salesforce.com/news/agentforce-workflows",
                "source_level": "B",
            }
        ],
    )

    assert "workflow deployments" in output["report_markdown"]
    assert output["source_claim_support"]["metric_claim_without_metric_fact_count"] == 0
    assert output["final_citation_audit"]["final_citation_reconciliation_status"] == "ok"


def test_final_writer_preserves_metric_claim_after_analysis_topic_checks_are_upstream(monkeypatch):
    monkeypatch.setenv("REPORT_FINAL_WRITER_SOURCE_APPENDIX", "true")

    output = run_final_writer_agent(
        query="AI Agent生态发展报告",
        report_blueprint={
            "report_shell": {"front_blocks": [], "back_blocks": ["appendix"]},
            "chapters": [{"chapter_id": "ch_01", "chapter_title": "Market validation"}],
        },
        chapter_packages=[
            {
                "chapter_id": "ch_01",
                "chapter_title": "Market validation",
                "sections": [
                    {
                        "section_id": "s_metric",
                        "section_title": "Market metric",
                        "block_type": "metric_reconciliation",
                        "claim": "Market metrics show a broad opportunity.",
                        "reasoning": "The market size trend supports a broad opportunity.",
                        "used_fact_refs": ["EV-MISMATCH"],
                        "evidence_refs": ["EV-MISMATCH"],
                        "render_blocks": [
                            {
                                "type": "paragraph",
                                "text": "The market size trend supports a broad opportunity.",
                            }
                        ],
                        "supporting_facts": [
                            {
                                "distilled_fact": "A local technology bureau article describes generic AI market statistics.",
                                "value": "7470",
                                "unit": "亿",
                                "period": "2026",
                                "source_ref": "EV-MISMATCH",
                                "source_title": "四平市科技局科普宣传文章",
                            }
                        ],
                        "evidence_backed": True,
                    }
                ],
            }
        ],
        source_registry=[
            {
                "ref": "EV-MISMATCH",
                "evidence_id": "EV-MISMATCH",
                "title": "四平市科技局科普宣传文章",
                "url": "http://kjj.siping.gov.cn/kjxx/kpxcl/202605/t20260512_766150.html",
                "source_level": "B",
                "metric_facts": [
                    {
                        "value": "7470",
                        "unit": "亿",
                        "period": "2026",
                        "source_ref": "EV-MISMATCH",
                    }
                ],
            }
        ],
    )

    assert "broad opportunity" in output["report_markdown"]
    assert output["source_claim_support"]["metric_claim_without_metric_fact_count"] == 0
    assert output["source_claim_support"]["section_dropped_due_to_source_claim_mismatch_count"] == 0


def test_final_writer_drops_metric_claim_when_source_is_placeholder_even_after_analysis(monkeypatch):
    monkeypatch.setenv("REPORT_FINAL_WRITER_SOURCE_APPENDIX", "true")

    output = run_final_writer_agent(
        query="AI Agent生态发展报告",
        report_blueprint={
            "report_shell": {"front_blocks": [], "back_blocks": ["appendix"]},
            "chapters": [{"chapter_id": "ch_01", "chapter_title": "Market validation"}],
        },
        chapter_packages=[
            {
                "chapter_id": "ch_01",
                "chapter_title": "Market validation",
                "sections": [
                    {
                        "section_id": "s_metric",
                        "section_title": "Market metric",
                        "block_type": "metric_reconciliation",
                        "claim": "The AI Agent market size shows a comparable signal.",
                        "reasoning": "The market size metric calibrates scale and comparability.",
                        "used_fact_refs": ["EV-DATE-METRIC"],
                        "evidence_refs": ["EV-DATE-METRIC"],
                        "render_blocks": [
                            {"type": "paragraph", "text": "The market size metric calibrates scale and comparability."}
                        ],
                        "supporting_facts": [
                            {
                                "distilled_fact": "A dated government page was crawled.",
                                "value": "-05",
                                "unit": "unknown",
                                "period": "2026-05-12T15:26:00+08:00",
                                "source_ref": "EV-DATE-METRIC",
                            }
                        ],
                        "evidence_backed": True,
                    }
                ],
            }
        ],
        source_registry=[
            {
                "ref": "EV-DATE-METRIC",
                "evidence_id": "EV-DATE-METRIC",
                "title": "AI Agent official page",
                "url": "https://example.org/ai-agent-page",
                "source_level": "B",
                "metric_facts": [
                    {
                        "value": "-05",
                        "unit": "unknown",
                        "period": "2026-05-12T15:26:00+08:00",
                        "source_ref": "EV-DATE-METRIC",
                    }
                ],
            }
        ],
    )

    assert "calibrates scale" not in output["report_markdown"]
    assert output["source_claim_support"]["section_dropped_due_to_source_claim_mismatch_count"] == 1
    assert output["source_claim_support"]["section_dropped_due_to_unresolved_refs_count"] == 1


def test_final_writer_preserves_metric_section_after_analysis_ref_binding_is_upstream(monkeypatch):
    monkeypatch.setenv("REPORT_FINAL_WRITER_SOURCE_APPENDIX", "true")

    output = run_final_writer_agent(
        query="AI Agent生态发展报告",
        report_blueprint={
            "report_shell": {"front_blocks": [], "back_blocks": ["appendix"]},
            "chapters": [{"chapter_id": "ch_01", "chapter_title": "Commercialization"}],
        },
        chapter_packages=[
            {
                "chapter_id": "ch_01",
                "chapter_title": "Commercialization",
                "sections": [
                    {
                        "section_id": "s_metric",
                        "section_title": "关键指标如何变化",
                        "block_type": "metric_reconciliation",
                        "claim": "AI Agent commercialization metrics can calibrate market size.",
                        "reasoning": "7470亿规模增长趋势与企业应用指南，这一事实用于校准市场指标的规模和可比性。",
                        "used_fact_refs": ["EV-WRONG-REF"],
                        "evidence_refs": ["EV-WRONG-REF"],
                        "render_blocks": [
                            {
                                "type": "paragraph",
                                "text": "7470亿规模增长趋势与企业应用指南，这一事实用于校准市场指标的规模和可比性。",
                            }
                        ],
                        "supporting_facts": ["7470亿规模增长趋势与企业应用指南"],
                        "evidence_backed": True,
                    }
                ],
            }
        ],
        source_registry=[
            {
                "ref": "EV-WRONG-REF",
                "evidence_id": "EV-WRONG-REF",
                "title": "Official AI Agent procurement budget",
                "url": "http://kjj.siping.gov.cn/kjxx/kpxcl/202605/t20260512_766150.html",
                "source_level": "A",
                "metric_facts": [
                    {
                        "value": "222.00万元",
                        "unit": "currency_cny",
                        "period": "2026",
                        "source_ref": "EV-WRONG-REF",
                        "distilled_fact": "政府AI技能培训采购预算：222.00万元（2026年曲靖经开区项目）",
                    }
                ],
            }
        ],
    )

    assert "7470亿" in output["report_markdown"]
    assert output["source_claim_support"]["section_dropped_due_to_source_claim_mismatch_count"] == 0
