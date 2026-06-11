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
