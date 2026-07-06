from rag_pipeline.agents.evidence_merger import _extract_numeric_values, build_evidence_package, normalize_evidence_items


def test_source_publisher_does_not_fall_back_to_nested_source_dict():
    evidence_pool = [
        {
            "agent": "search",
            "key_sources": [
                {
                    "title": "IDC AI Agent market report",
                    "url": "https://example.com/idc",
                    "source_type": "research",
                    "source": {
                        "title": "Nested source object should not become publisher",
                        "url": "https://example.com/nested",
                    },
                }
            ],
            "raw_data_points": [
                {
                    "evidence": "Official policy says AI agents are an important application form in 2026.",
                    "metric": "policy signal",
                    "value": "2026",
                    "source_title": "Central policy on intelligent agents",
                    "source_url": "https://gov.example/policy",
                    "source_type": "official",
                    "proof_role": "source_check",
                    "evidence_type": "official_data",
                }
            ],
        }
    ]

    normalized, _ = normalize_evidence_items(evidence_pool)

    source = normalized[0]["source"]
    assert source["title"] == "Central policy on intelligent agents"
    assert source["url"] == "https://gov.example/policy"
    assert source["publisher"] == ""
    assert "Nested source object" not in source["publisher"]


def test_developer_ecosystem_signal_is_kept_as_qualitative_evidence():
    evidence_pool = [
        {
            "agent": "search",
            "key_sources": [
                {
                    "title": "Developer ecosystem update",
                    "url": "https://example.com/developer-ecosystem",
                    "source_type": "media",
                }
            ],
            "raw_data_points": [
                {
                    "evidence": "开发者生态活跃，插件、SDK、API 文档和开源集成持续更新。",
                    "metric": "",
                    "value": "",
                    "source_title": "Developer ecosystem update",
                    "source_url": "https://example.com/developer-ecosystem",
                    "source_type": "media",
                }
            ],
        }
    ]

    normalized, _ = normalize_evidence_items(evidence_pool)

    assert normalized
    assert "开发者生态活跃" in normalized[0]["content"]


def test_public_fact_payload_preserves_proof_role_and_evidence_type():
    package = build_evidence_package(
        evidence_items=[
            {
                "evidence_id": "EV-POLICY",
                "dimension": "policy",
                "fact": "Central policy defines intelligent agents in 2026 and encourages regulated adoption.",
                "clean_fact": "Central policy defines intelligent agents in 2026 and encourages regulated adoption.",
                "metric": "policy signal",
                "value": "2026",
                "period": "2026-05-08",
                "source": {
                    "title": "Central policy on intelligent agents",
                    "url": "https://gov.example/policy",
                    "date": "2026-05-08",
                    "source_type": "official",
                    "publisher": "Central office",
                },
                "source_level": "A",
                "source_verification_status": "document_verified",
                "source_verified": True,
                "confidence": 0.8,
                "evidence_role": "supporting",
                "allowed_use": "supporting",
                "semantic_status": "weak_relevance",
                "proof_role": "source_check",
                "evidence_type": "official_data",
                "claim_type": "policy_signal",
            }
        ],
        top_k=4,
    )

    item = package["analysis_ready_evidence"][0]
    assert item["proof_role"] == "source_check"
    assert item["evidence_type"] == "official_data"
    assert item["claim_type"] == "policy_signal"


def test_ab_non_metric_evidence_can_be_promoted_beyond_context_when_traceable():
    package = build_evidence_package(
        evidence_items=[
            {
                "evidence_id": "EV-FILING",
                "dimension": "company filing",
                "fact": "The company filing describes enterprise agent deployment in 2025 as part of commercial products.",
                "clean_fact": "The company filing describes enterprise agent deployment in 2025 as part of commercial products.",
                "metric": "filing disclosure",
                "value": "2025",
                "period": "2025",
                "source": {
                    "title": "Company annual report",
                    "url": "https://ir.example/annual-report",
                    "date": "2025",
                    "source_type": "financial_report",
                    "publisher": "Example Inc.",
                },
                "source_level": "A",
                "source_verification_status": "document_verified",
                "source_verified": True,
                "confidence": 0.82,
                "evidence_role": "supporting",
                "allowed_use": "supporting_context",
                "semantic_status": "context_support",
                "task_acceptance_reason": "low_task_relevance_keep_as_clue",
                "proof_role": "filing",
                "evidence_type": "filing",
                "claim_type": "source_check",
            }
        ],
        top_k=4,
    )

    item = package["analysis_ready_evidence"][0]
    assert item["proof_role"] == "filing"
    assert item["claim_scope"] == "supporting"
    assert "supporting_claim" in item["can_support"]
    assert "complete_hard_metric_claim" not in item["cannot_support"]
    assert "metric_scope_period_unit_incomplete" not in item["repair_need"]


def test_policy_evidence_from_metric_task_is_reclassified_as_source_check():
    evidence_pool = [
        {
            "agent": "search",
            "search_task": {
                "task_id": "T-metric",
                "proof_role": "metric",
                "evidence_type": "official_data",
            },
            "key_sources": [
                {
                    "title": "Central policy on intelligent agents",
                    "url": "https://gov.example/policy",
                    "source_type": "official",
                    "publisher": "Central office",
                }
            ],
            "raw_data_points": [
                {
                    "evidence": "The 2026 policy guideline defines intelligent agents and sets regulatory adoption boundaries.",
                    "metric": "policy signal",
                    "value": "2026",
                    "source_title": "Central policy on intelligent agents",
                    "source_url": "https://gov.example/policy",
                    "source_type": "official",
                    "source_publisher": "Central office",
                }
            ],
        }
    ]

    normalized, _ = normalize_evidence_items(evidence_pool)

    assert normalized[0]["original_proof_role"] == "metric"
    assert normalized[0]["proof_role"] == "source_check"


def test_numeric_extraction_ignores_url_ids_and_calendar_days():
    detail_text = "Report page https://www.fxbaogao.com/detail/5084889 summarizes AI Agent adoption."
    date_text = "The 2025 World AI Conference published its governance action plan on July 26."

    assert "5084889" not in _extract_numeric_values(detail_text)
    assert "26" not in _extract_numeric_values(date_text)


def test_explicit_value_artifacts_do_not_become_numeric_values():
    evidence_pool = [
        {
            "agent": "search",
            "key_sources": [
                {
                    "title": "FX report detail page",
                    "url": "https://www.fxbaogao.com/detail/5084889",
                    "source_type": "media",
                }
            ],
            "raw_data_points": [
                {
                    "evidence": "Report page https://www.fxbaogao.com/detail/5084889 summarizes AI Agent adoption.",
                    "metric": "data indicator",
                    "value": "5084889",
                    "source_title": "FX report detail page",
                    "source_url": "https://www.fxbaogao.com/detail/5084889",
                    "source_type": "media",
                }
            ],
        }
    ]

    normalized, _ = normalize_evidence_items(evidence_pool)

    assert normalized[0]["value"] == ""
    assert normalized[0]["numeric_value"] is None
    assert normalized[0]["semantic_status"] == "rejected"


def test_isolated_quality_gate_keeps_appendix_clues_analysis_ready(monkeypatch):
    monkeypatch.setenv("REPORT_QUALITY_GATE_MODE", "isolated")
    package = build_evidence_package(
        evidence_items=[
            {
                "evidence_id": "EV-CLUE",
                "dimension": "policy",
                "fact": "A 2026 policy guideline defines intelligent agents and sets governance boundaries.",
                "clean_fact": "A 2026 policy guideline defines intelligent agents and sets governance boundaries.",
                "metric": "policy signal",
                "value": "2026",
                "period": "2026",
                "source": {
                    "title": "Policy guideline",
                    "url": "https://gov.example/policy",
                    "source_type": "official",
                    "publisher": "Government office",
                },
                "source_level": "C",
                "source_verification_status": "readpage_verified",
                "source_verified": True,
                "confidence": 0.35,
                "evidence_role": "clue",
                "allowed_use": "appendix_only",
                "appendix_only": True,
                "semantic_status": "weak_relevance",
                "proof_role": "source_check",
                "evidence_type": "official_data",
            }
        ],
        top_k=4,
    )

    by_id = {item["evidence_id"]: item for item in package["analysis_ready_evidence"]}
    assert "EV-CLUE" in by_id
    assert by_id["EV-CLUE"]["allowed_use"] == "supporting_context"
    assert by_id["EV-CLUE"]["appendix_only"] is False
    assert by_id["EV-CLUE"]["quality_gate_observations"]
    assert by_id["EV-CLUE"]["analysis_input"]["quality_gate_observations"]


def test_topic_anchor_missing_clue_is_not_promoted_to_analysis_ready(monkeypatch):
    monkeypatch.setenv("REPORT_QUALITY_GATE_MODE", "isolated")
    package = build_evidence_package(
        evidence_items=[
            {
                "evidence_id": "EV-OFF-TOPIC",
                "dimension": "market signal",
                "fact": "Humanoid robot exports reached 11.32 billion yuan in the first quarter.",
                "clean_fact": "Humanoid robot exports reached 11.32 billion yuan in the first quarter.",
                "metric": "market size",
                "value": "11.32 billion yuan",
                "period": "2026",
                "source": {
                    "title": "Humanoid robot industry update",
                    "url": "https://gov.example/robot",
                    "source_type": "official",
                    "publisher": "Government office",
                },
                "source_level": "A",
                "source_verification_status": "readpage_verified",
                "source_verified": True,
                "confidence": 0.82,
                "evidence_role": "clue",
                "allowed_use": "appendix_only",
                "appendix_only": True,
                "semantic_status": "weak_relevance",
                "semantic_reason": "topic_anchor_missing",
                "task_acceptance_reason": "topic_anchor_missing",
                "proof_role": "source_check",
                "evidence_type": "official_data",
            }
        ],
        top_k=4,
    )

    by_id = {item["evidence_id"]: item for item in package["analysis_ready_evidence"]}
    clean_by_id = {item["evidence_id"]: item for item in package["clean_evidence_list"]}
    assert "EV-OFF-TOPIC" not in by_id
    assert clean_by_id["EV-OFF-TOPIC"]["evidence_role"] == "clue"


def test_recall_first_mode_promotes_traceable_media_clue_analysis_ready(monkeypatch):
    monkeypatch.delenv("REPORT_QUALITY_GATE_MODE", raising=False)
    monkeypatch.delenv("REPORT_EVIDENCE_RECALL_MODE", raising=False)
    package = build_evidence_package(
        evidence_items=[
            {
                "evidence_id": "EV-MEDIA-CLUE",
                "dimension": "market signal",
                "fact": "A 2026 business media article reports that enterprises are testing AI Agent workflow assistants.",
                "clean_fact": "A 2026 business media article reports that enterprises are testing AI Agent workflow assistants.",
                "metric": "deployment signal",
                "value": "2026",
                "period": "2026",
                "source": {
                    "title": "Business media AI Agent workflow report",
                    "url": "https://www.yicai.com/news/ai-agent-workflow",
                    "source_type": "media",
                    "publisher": "Yicai",
                },
                "source_level": "C",
                "source_verification_status": "search_result_only",
                "source_verified": False,
                "confidence": 0.32,
                "evidence_role": "clue",
                "allowed_use": "appendix_only",
                "appendix_only": True,
                "semantic_status": "weak_relevance",
                "proof_role": "case",
                "evidence_type": "media_signal",
            }
        ],
        top_k=4,
    )

    by_id = {item["evidence_id"]: item for item in package["analysis_ready_evidence"]}
    assert "EV-MEDIA-CLUE" in by_id
    item = by_id["EV-MEDIA-CLUE"]
    assert item["allowed_use"] == "supporting_context"
    assert item["appendix_only"] is False
    assert item["evidence_role"] == "supporting"
    assert item["quality_gate_observations"]


def test_pdf_header_shape_issue_is_diagnostic_not_analysis_ready(monkeypatch):
    monkeypatch.setenv("REPORT_EVIDENCE_MODE", "advisory")
    package = build_evidence_package(
        evidence_items=[
            {
                "evidence_id": "EV-PDF-SHELL",
                "dimension": "market signal",
                "fact": "PDF: industry report doi: 10.1234/abcd | ---- | page header table fragment.",
                "clean_fact": "PDF: industry report doi: 10.1234/abcd | ---- | page header table fragment.",
                "metric": "report fragment",
                "value": "10.1234",
                "period": "2026",
                "source": {
                    "title": "PDF page shell",
                    "url": "https://reports.example/pdf-shell",
                    "source_type": "media",
                    "publisher": "Reports Example",
                },
                "source_level": "C",
                "source_verification_status": "readpage_verified",
                "source_verified": True,
                "confidence": 0.82,
                "evidence_role": "supporting",
                "allowed_use": "supporting_context",
                "semantic_status": "ok",
                "proof_role": "case",
                "evidence_type": "media_signal",
            },
            {
                "evidence_id": "EV-CLEAN-SIGNAL",
                "dimension": "market signal",
                "fact": "A 2026 business media article reports enterprises testing workflow assistants.",
                "clean_fact": "A 2026 business media article reports enterprises testing workflow assistants.",
                "metric": "deployment signal",
                "value": "2026",
                "period": "2026",
                "source": {
                    "title": "Business media workflow report",
                    "url": "https://news.example/workflow",
                    "source_type": "media",
                    "publisher": "News Example",
                },
                "source_level": "C",
                "source_verification_status": "readpage_verified",
                "source_verified": True,
                "confidence": 0.62,
                "evidence_role": "supporting",
                "allowed_use": "supporting_context",
                "semantic_status": "ok",
                "proof_role": "case",
                "evidence_type": "media_signal",
            },
        ],
        top_k=4,
    )

    analysis_ready_ids = {item["evidence_id"] for item in package["analysis_ready_evidence"]}
    clean_by_id = {item["evidence_id"]: item for item in package["clean_evidence_list"]}

    assert "EV-CLEAN-SIGNAL" in analysis_ready_ids
    assert "EV-PDF-SHELL" not in analysis_ready_ids
    assert clean_by_id["EV-PDF-SHELL"]["allowed_use"] == "diagnostic_only"
    assert clean_by_id["EV-PDF-SHELL"]["analysis_readiness"] == "clue_only"
    assert clean_by_id["EV-PDF-SHELL"]["analysis_ready_exclusion_reason"] == "pdf_table_or_header_fragment"


def test_generic_metric_short_field_artifact_is_diagnostic_not_analysis_ready(monkeypatch):
    monkeypatch.setenv("REPORT_EVIDENCE_MODE", "advisory")
    package = build_evidence_package(
        evidence_items=[
            {
                "evidence_id": "EV-GENERIC-BAD",
                "dimension": "market signal",
                "fact": "data indicator 2026",
                "clean_fact": "data indicator 2026",
                "metric": "data indicator",
                "value": "2026",
                "period": "2026",
                "source": {
                    "title": "Thin search result",
                    "url": "https://news.example/thin",
                    "source_type": "media",
                },
                "source_level": "C",
                "source_verification_status": "search_result_only",
                "confidence": 0.7,
                "evidence_role": "supporting",
                "allowed_use": "supporting_context",
                "semantic_status": "ok",
            }
        ],
        top_k=4,
    )

    analysis_ready_ids = {item["evidence_id"] for item in package["analysis_ready_evidence"]}
    clean_by_id = {item["evidence_id"]: item for item in package["clean_evidence_list"]}

    assert "EV-GENERIC-BAD" not in analysis_ready_ids
    assert clean_by_id["EV-GENERIC-BAD"]["allowed_use"] == "diagnostic_only"
    assert clean_by_id["EV-GENERIC-BAD"]["analysis_ready_exclusion_reason"] == "generic_metric_field_artifact"


def test_page_shell_toc_fragment_is_diagnostic_not_analysis_ready(monkeypatch):
    monkeypatch.setenv("REPORT_EVIDENCE_MODE", "advisory")
    shell_text = (
        "\u76ee \u5f55 \u8ba9\u673a\u5668\u4eba\u66f4\u667a\u6167 / 01 "
        "\u4eba\u5f62\u673a\u5668\u4eba\u5341\u5927\u5e94\u7528\u573a\u666f / 02 "
        "\u5177\u8eab\u667a\u80fd\u673a\u5668\u4eba\u5341\u5927\u8d8b\u52bf / 03 "
        "\u4ea7\u4e1a\u672a\u6765\u65b9\u5411 / 04"
    )
    package = build_evidence_package(
        evidence_items=[
            {
                "evidence_id": "EV-TOC-SHELL",
                "dimension": "market signal",
                "fact": shell_text,
                "clean_fact": shell_text,
                "metric": "key fact",
                "value": "01",
                "period": "2025",
                "source": {
                    "title": "Robot conference PDF",
                    "url": "https://gov.example/robot.pdf",
                    "source_type": "official",
                },
                "source_level": "A",
                "source_verification_status": "readpage_verified",
                "confidence": 0.8,
                "evidence_role": "supporting",
                "allowed_use": "supporting_context",
                "semantic_status": "ok",
            }
        ],
        top_k=4,
    )

    analysis_ready_ids = {item["evidence_id"] for item in package["analysis_ready_evidence"]}
    clean_by_id = {item["evidence_id"]: item for item in package["clean_evidence_list"]}

    assert "EV-TOC-SHELL" not in analysis_ready_ids
    assert clean_by_id["EV-TOC-SHELL"]["allowed_use"] == "diagnostic_only"
    assert clean_by_id["EV-TOC-SHELL"]["analysis_ready_exclusion_reason"] == "page_shell_or_toc_fragment"


def test_missing_requirement_id_gets_chapter_role_fallback_for_analysis_ready(monkeypatch):
    monkeypatch.setenv("REPORT_EVIDENCE_MODE", "advisory")
    package = build_evidence_package(
        evidence_items=[
            {
                "evidence_id": "EV-DYNAMIC-CASE",
                "chapter_id": "ch_02",
                "search_task_id": "dynamic_iqs_003",
                "proof_role": "case",
                "dimension": "channel signal",
                "fact": "A 2026 media report says robot makers are testing embodied intelligence products in factory inspection scenarios.",
                "clean_fact": "A 2026 media report says robot makers are testing embodied intelligence products in factory inspection scenarios.",
                "metric": "case signal",
                "value": "",
                "period": "2026",
                "source": {
                    "title": "Robot deployment report",
                    "url": "https://news.example/robot-deployment",
                    "source_type": "media",
                },
                "source_level": "C",
                "source_verification_status": "readpage_verified",
                "confidence": 0.72,
                "evidence_role": "supporting",
                "allowed_use": "supporting_context",
                "semantic_status": "ok",
            }
        ],
        top_k=4,
    )

    item = package["analysis_ready_evidence"][0]

    assert item["requirement_id"] == "ch_02_case"
    assert item["lineage"]["requirement_id"] == "ch_02_case"
    assert item["requirement_id_inferred"] is True
    assert item["requirement_id_source"] == "chapter_proof_role_fallback"


def test_generic_metric_with_meaningful_qualitative_fact_stays_analysis_ready(monkeypatch):
    monkeypatch.setenv("REPORT_EVIDENCE_MODE", "advisory")
    package = build_evidence_package(
        evidence_items=[
            {
                "evidence_id": "EV-GENERIC-GOOD",
                "dimension": "market signal",
                "fact": "A 2026 business media report says enterprises are testing AI Agent workflow assistants in customer-service scenarios.",
                "clean_fact": "A 2026 business media report says enterprises are testing AI Agent workflow assistants in customer-service scenarios.",
                "metric": "data indicator",
                "value": "2026",
                "period": "2026",
                "source": {
                    "title": "Business media workflow report",
                    "url": "https://news.example/workflow-assistants",
                    "source_type": "media",
                },
                "source_level": "C",
                "source_verification_status": "readpage_verified",
                "confidence": 0.7,
                "evidence_role": "supporting",
                "allowed_use": "supporting_context",
                "semantic_status": "ok",
            }
        ],
        top_k=4,
    )

    by_id = {item["evidence_id"]: item for item in package["analysis_ready_evidence"]}

    assert "EV-GENERIC-GOOD" in by_id
    assert "generic_metric_name" in by_id["EV-GENERIC-GOOD"]["content_shape_issues"]
    assert by_id["EV-GENERIC-GOOD"]["allowed_use"] == "supporting_context"


def test_market_size_metric_is_not_treated_as_generic_field_artifact(monkeypatch):
    monkeypatch.setenv("REPORT_EVIDENCE_MODE", "advisory")
    package = build_evidence_package(
        evidence_items=[
            {
                "evidence_id": "EV-MARKET-SIZE",
                "dimension": "market signal",
                "fact": "2025年中国低空经济市场规模预计达5615亿元",
                "clean_fact": "2025年中国低空经济市场规模预计达5615亿元",
                "metric": "市场规模",
                "value": "5615",
                "unit": "亿元",
                "period": "2025",
                "source": {
                    "title": "Low altitude economy report",
                    "url": "https://report.example/low-altitude",
                    "source_type": "research_report",
                },
                "source_level": "C",
                "source_verification_status": "readpage_verified",
                "confidence": 0.7,
                "evidence_role": "supporting",
                "allowed_use": "supporting_context",
                "semantic_status": "ok",
                "requirement_id": "H1_metric",
            }
        ],
        top_k=4,
    )

    by_id = {item["evidence_id"]: item for item in package["analysis_ready_evidence"]}

    assert "EV-MARKET-SIZE" in by_id
    assert "generic_metric_name" not in by_id["EV-MARKET-SIZE"].get("content_shape_issues", [])
    assert by_id["EV-MARKET-SIZE"].get("analysis_ready_exclusion_reason", "") == ""
    assert by_id["EV-MARKET-SIZE"]["unit"] == "亿元"


def test_public_metric_payload_recovers_value_from_number_with_unit(monkeypatch):
    monkeypatch.setenv("REPORT_EVIDENCE_MODE", "advisory")
    package = build_evidence_package(
        evidence_items=[
            {
                "evidence_id": "EV-MARKET-SIZE-LONG",
                "dimension": "market signal",
                "fact": "2035年中国低空经济市场规模有望突破3.5万亿元",
                "clean_fact": "2035年中国低空经济市场规模有望突破3.5万亿元",
                "metric": "市场规模",
                "value": "",
                "unit": "万亿元",
                "period": "2035",
                "source": {
                    "title": "Low altitude economy report",
                    "url": "https://report.example/low-altitude",
                    "source_type": "research_report",
                },
                "source_level": "C",
                "source_verification_status": "readpage_verified",
                "confidence": 0.7,
                "evidence_role": "supporting",
                "allowed_use": "supporting_context",
                "semantic_status": "ok",
                "requirement_id": "H1_metric",
            }
        ],
        top_k=4,
    )

    by_id = {item["evidence_id"]: item for item in package["analysis_ready_evidence"]}

    assert by_id["EV-MARKET-SIZE-LONG"]["value"] == "3.5"
    assert by_id["EV-MARKET-SIZE-LONG"]["unit"] == "万亿元"


def test_ab_policy_evidence_uses_chapter_relevance_when_metric_query_is_too_narrow():
    evidence_pool = [
        {
            "agent": "search",
            "search_task": {
                "task_id": "T-market-metric",
                "query": "AI Agent enterprise market size adoption rate",
                "proof_role": "metric",
                "evidence_type": "official_data",
                "must_have_terms": ["market size", "adoption rate"],
                "source_priority": ["official_data"],
                "global_required_terms": ["AI Agent"],
                "research_object": "AI Agent enterprise adoption",
            },
            "key_sources": [
                {
                    "title": "Central policy on intelligent agents",
                    "url": "https://gov.example/policy",
                    "source_type": "official",
                    "publisher": "Central office",
                }
            ],
            "raw_data_points": [
                {
                    "evidence": "The 2026 policy guideline defines AI Agent as an important application form and sets regulated adoption boundaries.",
                    "metric": "policy signal",
                    "value": "2026",
                    "source_title": "Central policy on intelligent agents",
                    "source_url": "https://gov.example/policy",
                    "source_type": "official",
                    "source_publisher": "Central office",
                }
            ],
        }
    ]

    normalized, _ = normalize_evidence_items(evidence_pool)

    item = normalized[0]
    assert item["proof_role"] == "source_check"
    assert item["task_accepted"] is True
    assert item["task_acceptance_reason"] == "chapter_or_report_relevance_pass"
    assert item["evidence_role"] in {"core", "supporting"}
    assert item["appendix_only"] is False


def test_advisory_metric_semantic_gap_is_observed_not_rejected(monkeypatch):
    monkeypatch.setenv("REPORT_EVIDENCE_MODE", "advisory")
    evidence_pool = [
        {
            "agent": "search",
            "key_sources": [
                {
                    "title": "Industry media adoption report",
                    "url": "https://news.example/adoption",
                    "source_type": "media",
                }
            ],
            "raw_data_points": [
                {
                    "evidence": (
                        "The 2026 industry media report says enterprise AI Agent adoption is accelerating, "
                        "but does not publish a precise percentage."
                    ),
                    "metric": "CAGR",
                    "value": "2026",
                    "source_title": "Industry media adoption report",
                    "source_url": "https://news.example/adoption",
                    "source_type": "media",
                    "proof_role": "metric",
                }
            ],
        }
    ]

    normalized, _ = normalize_evidence_items(evidence_pool)

    assert normalized
    item = normalized[0]
    assert item["semantic_status"] != "rejected"
    assert item["evidence_role"] != "rejected"
    assert item["metric_semantic_observation"]["status"] == "exclude"


def test_advisory_topic_anchor_missing_is_rejected_as_off_topic(monkeypatch):
    monkeypatch.setenv("REPORT_EVIDENCE_MODE", "advisory")
    evidence_pool = [
        {
            "agent": "search",
            "search_task": {
                "task_id": "T-low-altitude-market",
                "query": "low altitude economy market size",
                "proof_role": "case",
                "evidence_type": "media_signal",
                "topic_anchor_terms": ["low altitude economy"],
                "must_have_terms": ["market size"],
                "source_priority": ["media"],
            },
            "key_sources": [
                {
                    "title": "Robot deployment report",
                    "url": "https://news.example/robot-deployment",
                    "source_type": "media",
                }
            ],
            "raw_data_points": [
                {
                    "evidence": "A 2026 media report says robot makers are testing factory inspection products.",
                    "metric": "case signal",
                    "value": "2026",
                    "source_title": "Robot deployment report",
                    "source_url": "https://news.example/robot-deployment",
                    "source_type": "media",
                }
            ],
        }
    ]

    normalized, _ = normalize_evidence_items(evidence_pool)

    assert normalized
    item = normalized[0]
    assert item["task_accepted"] is False
    assert item["task_acceptance_reason"] == "topic_anchor_missing"
    assert item["semantic_status"] == "rejected"
    assert item["evidence_role"] == "rejected"


def test_advisory_weight_keeps_chapter_local_employment_signal_when_global_anchor_missing(monkeypatch):
    monkeypatch.setenv("REPORT_EVIDENCE_MODE", "advisory_weight")
    evidence_pool = [
        {
            "agent": "iqs_lane_1_agent",
            "search_task": {
                "task_id": "T-accounting-employment",
                "query": "会计学专业就业的职业路径正在被财务共享、数智化工具和AI改造 官方 统计 协会 白皮书 数据",
                "proof_role": "source_check",
                "evidence_type": "data",
                "chapter_id": "ch_03",
                "dimension_id": "ch_03",
                "dimension_name": "会计学专业就业的职业路径正在被财务共享、数智化工具和AI改造",
                "evidence_goal": "会计学专业就业的职业路径正在被财务共享、数智化工具和AI改造",
                "research_object": "会计学专业就业",
                "topic_anchor_terms": ["会计学专业就业", "会计学专业就业趋势与AI影响", "会计学专业"],
                "global_required_terms": ["会计学专业就业", "会计学专业", "就业趋势", "AI影响", "财务数字化", "业财融合", "RPA", "大模型"],
                "source_priority": ["official_data"],
            },
            "key_sources": [
                {
                    "title": "学生发展",
                    "url": "http://kjxy.example.edu.cn/15675/list1.htm",
                    "source_type": "media",
                }
            ],
            "raw_data_points": [
                {
                    "evidence": (
                        "为深化校企合作，拓宽学生实习就业渠道，华润电力随州新能源公司财务部、"
                        "人力资源经理和核算会计经理到访会计学院开展调研座谈。"
                    ),
                    "metric": "source_check",
                    "value": "",
                    "source_title": "学生发展",
                    "source_url": "http://kjxy.example.edu.cn/15675/list1.htm",
                    "source_type": "media",
                    "proof_role": "source_check",
                }
            ],
        }
    ]

    normalized, _ = normalize_evidence_items(evidence_pool)
    package = build_evidence_package(evidence_items=normalized, top_k=8)

    assert normalized
    item = normalized[0]
    assert item["task_accepted"] is True
    assert item["task_acceptance_reason"] == "local_task_relevance_pass"
    assert item["semantic_status"] != "rejected"
    assert item["evidence_role"] != "rejected"
    assert package["analysis_ready_evidence"]
    assert package["analysis_ready_evidence"][0]["task_acceptance_reason"] == "local_task_relevance_pass"


def test_source_only_metadata_line_stays_out_of_analysis_ready(monkeypatch):
    monkeypatch.setenv("REPORT_EVIDENCE_MODE", "advisory_weight")
    evidence_pool = [
        {
            "agent": "iqs_lane_1_agent",
            "search_task": {
                "task_id": "T-accounting-metadata",
                "query": "会计学专业就业方向及前景",
                "proof_role": "counter",
                "chapter_id": "ch_02",
                "dimension_id": "ch_02",
                "dimension_name": "会计学专业就业的知识体系需要区分课程基础、专业能力、实务训练和证书要求",
                "evidence_goal": "会计学专业就业的知识体系需要区分课程基础、专业能力、实务训练和证书要求",
                "research_object": "会计学专业就业",
                "topic_anchor_terms": ["会计学专业就业", "会计学专业就业趋势与AI影响", "会计学专业"],
                "global_required_terms": ["会计学专业就业", "会计学专业", "就业趋势", "AI影响", "财务数字化"],
            },
            "key_sources": [
                {
                    "title": "会计学专业代码？会计学专业就业方向及前景？",
                    "url": "https://kaoyan.example.cn/zaizhi/article/p2426",
                    "source_type": "media",
                }
            ],
            "raw_data_points": [
                {
                    "evidence": "发布时间： 2025-06-23 阅读量： 6456 来源： 在职考研",
                    "metric": "counter",
                    "value": "",
                    "source_title": "会计学专业代码？会计学专业就业方向及前景？",
                    "source_url": "https://kaoyan.example.cn/zaizhi/article/p2426",
                    "source_type": "media",
                    "proof_role": "counter",
                }
            ],
        }
    ]

    normalized, _ = normalize_evidence_items(evidence_pool)
    package = build_evidence_package(evidence_items=normalized, top_k=8)

    assert normalized
    item = normalized[0]
    assert item["task_accepted"] is False
    assert item["task_acceptance_reason"] == "topic_anchor_missing"
    assert item["semantic_status"] == "rejected"
    assert package["analysis_ready_evidence"] == []


def test_answer_nav_fragments_are_skipped_when_structured_raw_points_exist(monkeypatch):
    monkeypatch.delenv("EVIDENCE_MERGER_SKIP_ANSWER_LINES_WHEN_RAW_POINTS", raising=False)
    evidence_pool = [
        {
            "agent": "iqs_lane_1_agent",
            "answer": (
                "1. 2026年中国锂电池行业报告目录与章节列表 [1]\n"
                "2. 相关阅读：短剧服务、营销工具、会议议程 [1]"
            ),
            "key_sources": [
                {
                    "title": "Low-altitude economy policy source",
                    "url": "https://gov.example/low-altitude-policy",
                    "source_type": "official",
                }
            ],
            "raw_data_points": [
                {
                    "evidence": "The 2026 policy says low-altitude economy applications are expanding into logistics and emergency response.",
                    "metric": "policy signal",
                    "value": "2026",
                    "source_title": "Low-altitude economy policy source",
                    "source_url": "https://gov.example/low-altitude-policy",
                    "source_type": "official",
                }
            ],
        }
    ]

    normalized, metadata = normalize_evidence_items(evidence_pool)

    evidence_ids = [str(item.get("evidence_id") or "") for item in normalized]
    contents = "\n".join(str(item.get("content") or "") for item in normalized)
    assert normalized
    assert all("-L" not in evidence_id for evidence_id in evidence_ids)
    assert "锂电池行业报告目录" not in contents
    assert "low-altitude economy applications" in contents
    assert metadata["skipped_answer_line_count"] == 2
    assert metadata["skip_answer_lines_with_raw_points"] is False


def test_useful_answer_lines_are_kept_when_raw_points_exist_by_default(monkeypatch):
    monkeypatch.delenv("EVIDENCE_MERGER_SKIP_ANSWER_LINES_WHEN_RAW_POINTS", raising=False)
    evidence_pool = [
        {
            "agent": "iqs_lane_1_agent",
            "answer": (
                "1. A 2026 official policy says low-altitude logistics and emergency response "
                "applications are expanding from pilots into routine city operations.[1]"
            ),
            "key_sources": [
                {
                    "title": "Low-altitude economy policy source",
                    "url": "https://gov.example/low-altitude-policy",
                    "source_type": "official",
                }
            ],
            "raw_data_points": [
                {
                    "evidence": "The 2026 policy says low-altitude economy applications are expanding into logistics.",
                    "metric": "policy signal",
                    "value": "2026",
                    "source_title": "Low-altitude economy policy source",
                    "source_url": "https://gov.example/low-altitude-policy",
                    "source_type": "official",
                }
            ],
        }
    ]

    normalized, metadata = normalize_evidence_items(evidence_pool)

    evidence_ids = {str(item.get("evidence_id") or "") for item in normalized}
    contents = "\n".join(str(item.get("content") or "") for item in normalized)
    assert any("-L" in evidence_id for evidence_id in evidence_ids)
    assert "routine city operations" in contents
    assert metadata["skipped_answer_line_count"] == 0
    assert metadata["skip_answer_lines_with_raw_points"] is False


def test_evidence_package_exposes_analysis_claim_and_public_fact_pools():
    package = build_evidence_package(
        evidence_items=[
            {
                "evidence_id": "EV-POLICY",
                "dimension": "policy",
                "fact": "Central policy defines intelligent agents in 2026 and encourages regulated adoption.",
                "clean_fact": "Central policy defines intelligent agents in 2026 and encourages regulated adoption.",
                "metric": "policy signal",
                "value": "2026",
                "period": "2026",
                "source": {"title": "Central policy", "url": "https://gov.example/policy", "source_type": "official"},
                "source_level": "A",
                "confidence": 0.8,
                "evidence_role": "supporting",
                "allowed_use": "supporting",
                "semantic_status": "weak_relevance",
                "proof_role": "source_check",
            },
            {
                "evidence_id": "EV-CONTEXT",
                "dimension": "market",
                "fact": "Vertical media reports that several local pilots have begun testing agent workflows in 2026.",
                "clean_fact": "Vertical media reports that several local pilots have begun testing agent workflows in 2026.",
                "metric": "case signal",
                "value": "2026",
                "period": "2026",
                "source": {"title": "Industry media", "url": "https://media.example/agent", "source_type": "media"},
                "source_level": "C",
                "confidence": 0.58,
                "evidence_role": "supporting",
                "allowed_use": "supporting_context",
                "semantic_status": "context_support",
                "proof_role": "case",
            },
            {
                "evidence_id": "EV-DIAG",
                "dimension": "market",
                "fact": "login sign in cookie policy request a demo",
                "clean_fact": "login sign in cookie policy request a demo",
                "metric": "source_check",
                "value": "200",
                "source": {"title": "Login page", "url": "https://example.com/login", "source_type": "web"},
                "source_level": "D",
                "confidence": 0.2,
                "evidence_role": "supporting",
                "allowed_use": "supporting",
                "semantic_status": "weak_relevance",
            },
        ],
        top_k=4,
    )

    analysis_ids = {item["evidence_id"] for item in package["analysis_candidate_facts"]}
    claim_ids = {item["evidence_id"] for item in package["claim_support_facts"]}
    citation_ids = {item["evidence_id"] for item in package["public_citation_facts"]}
    assert {"EV-POLICY", "EV-CONTEXT"} <= analysis_ids
    assert {"EV-POLICY", "EV-CONTEXT"} <= claim_ids
    assert "EV-POLICY" in citation_ids
    assert "EV-DIAG" not in analysis_ids
    assert "EV-DIAG" not in claim_ids
    assert "EV-DIAG" not in citation_ids
    assert package["summary"]["analysis_candidate_fact_count"] == len(package["analysis_candidate_facts"])
    assert package["summary"]["claim_support_fact_count"] == len(package["claim_support_facts"])
    assert package["summary"]["public_citation_fact_count"] == len(package["public_citation_facts"])


def test_missing_china_scope_anchor_does_not_reject_core_topic_fact(monkeypatch):
    monkeypatch.setenv("REPORT_EVIDENCE_MODE", "advisory")
    evidence_pool = [
        {
            "agent": "iqs_lane_1_agent",
            "search_task": {
                "task_id": "T-low-altitude-market",
                "query": "中国低空经济产业链 市场规模 官方数据 2026",
                "proof_role": "metric",
                "evidence_type": "official_data",
                "topic_anchor_terms": ["中国低空经济产业链", "低空经济产业链", "低空经济"],
                "global_required_terms": ["中国低空经济产业链", "低空经济"],
                "must_have_terms": ["低空经济", "市场规模"],
                "source_priority": ["official_data"],
                "research_object": "中国低空经济产业链",
            },
            "key_sources": [
                {
                    "title": "低空经济产业发展行动方案",
                    "url": "https://gov.example/low-altitude-plan",
                    "source_type": "official",
                }
            ],
            "raw_data_points": [
                {
                    "evidence": "到2026年，低空经济产业规模达到600亿元，建设通用机场和垂直起降点。",
                    "metric": "产业规模",
                    "value": "600",
                    "unit": "亿元",
                    "source_title": "低空经济产业发展行动方案",
                    "source_url": "https://gov.example/low-altitude-plan",
                    "source_type": "official",
                    "proof_role": "metric",
                }
            ],
        }
    ]

    normalized, _ = normalize_evidence_items(evidence_pool)

    assert normalized
    item = normalized[0]
    assert item["task_acceptance_reason"] != "topic_anchor_missing"
    assert item["semantic_status"] != "rejected"
    assert item["evidence_role"] != "rejected"


def test_source_only_weak_topic_match_is_not_analysis_ready(monkeypatch):
    monkeypatch.setenv("REPORT_EVIDENCE_MODE", "advisory")
    evidence_pool = [
        {
            "agent": "iqs_lane_1_agent",
            "search_task": {
                "task_id": "T-low-altitude-support",
                "query": "中国低空经济产业链 市场规模 2026",
                "proof_role": "metric",
                "evidence_type": "market_research",
                "topic_anchor_terms": ["中国低空经济产业链", "低空经济产业链", "低空经济"],
                "global_required_terms": ["中国低空经济产业链", "低空经济"],
                "must_have_terms": ["低空经济", "市场规模"],
                "source_priority": ["market_research"],
                "research_object": "中国低空经济产业链",
            },
            "key_sources": [
                {
                    "title": "2026年中国低空经济产业链研究报告",
                    "url": "https://research.example/low-altitude",
                    "source_type": "research",
                }
            ],
            "raw_data_points": [
                {
                    "evidence": "报告数据显示：北京微短剧产值规模约336.2亿元，占全国总量三分之二。",
                    "metric": "support",
                    "value": "336.2",
                    "unit": "亿元",
                    "source_title": "2026年中国低空经济产业链研究报告",
                    "source_url": "https://research.example/low-altitude",
                    "source_type": "research",
                    "proof_role": "metric",
                }
            ],
        }
    ]

    normalized, _ = normalize_evidence_items(evidence_pool)
    package = build_evidence_package(evidence_items=normalized, top_k=8)

    assert normalized
    item = normalized[0]
    assert item["source_only_topic_match"] is True
    assert item["content_topic_hit"] is False
    assert item["task_acceptance_reason"] == "low_task_relevance_keep_as_clue"
    assert package["analysis_ready_evidence"] == []
    assert package["clean_evidence_list"][0]["analysis_ready_exclusion_reason"] == "source_only_topic_match"


def test_scraped_nav_headline_listing_is_a_blocking_shape_issue():
    from rag_pipeline.contracts.evidence_quality import (
        BLOCKING_CONTENT_SHAPE_ISSUES,
        evidence_content_shape_issues,
    )

    # Real scraped garbage: slash-separated headlines with orphan ordinals.
    garbage = {
        "fact": "全球大咖纵论机器人 / 04 产业未来方向 / 07 新火花 世界机器人大会 / 08 让具身体更智能",
    }
    issues = evidence_content_shape_issues(garbage)
    assert "nav_or_listing_fragment" in issues
    assert "nav_or_listing_fragment" in BLOCKING_CONTENT_SHAPE_ISSUES


def test_dates_and_normal_prose_are_not_flagged_as_nav_listing():
    from rag_pipeline.contracts.evidence_quality import evidence_content_shape_issues

    # Date ranges (no space before slash) and ordinary prose must not trip the detector.
    assert "nav_or_listing_fragment" not in evidence_content_shape_issues(
        {"fact": "2024/05 至 2025/06 中国市场规模由5000亿元增至6700亿元。"}
    )
    assert "nav_or_listing_fragment" not in evidence_content_shape_issues(
        {"fact": "市场规模在2025年达到约14亿美元，增速显著。"}
    )


def test_fallback_requirement_id_covers_non_canonical_chapter_ids():
    from rag_pipeline.agents.evidence_merger import _fallback_requirement_id

    # Canonical ch_NN chapters keep the existing readable form.
    assert _fallback_requirement_id("ch_01", "metric") == "ch_01_metric"

    # A fact whose chapter_id is the question text (carried from a web/followup
    # lane) must still receive a stable requirement id instead of "" so the claim
    # is not flagged missing_requirement_ids and demoted to directional.
    question = "中国具身智能机器人是否存在真实需求和可验证市场空间"
    rid = _fallback_requirement_id(question, "metric")
    assert rid
    assert rid.startswith("ch_") and rid.endswith("_metric")
    # Deterministic: same chapter+role -> same requirement id (consistent grouping).
    assert rid == _fallback_requirement_id(question, "metric")
    # Different chapter text -> different requirement id.
    assert rid != _fallback_requirement_id(question + "（不同章节）", "metric")

    # Empty chapter id remains unbindable.
    assert _fallback_requirement_id("", "metric") == ""
