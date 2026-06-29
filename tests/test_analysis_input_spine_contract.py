from rag_pipeline.agents.analysis_agent import build_llm_analysis_input_v2
from rag_pipeline.agents.evidence_merger import build_clean_facts


def test_clean_fact_preserves_search_task_lineage_for_analysis():
    clean_facts, duplicate_count = build_clean_facts(
        [
                {
                    "evidence_id": "EV-1",
                    "fact": "Policy documents describe low-altitude logistics pilots in 2025.",
                    "clean_fact": "Policy documents describe low-altitude logistics pilots in 2025.",
                    "metric": "policy_signal",
                "source": {
                    "title": "Official policy notice",
                    "url": "https://gov.cn/policy/low-altitude-logistics",
                    "source_type": "government",
                    "main_text": "Policy documents describe low-altitude logistics pilots in 2025.",
                },
                "source_id": "SRC-1",
                "source_level": "A",
                "allowed_use": "supporting_context",
                "search_task": {
                    "task_id": "task_ch01_h1_policy",
                    "chapter_id": "ch_01",
                    "requirement_id": "H1_policy",
                    "hypothesis_id": "H1",
                    "proof_role": "policy",
                    "analysis_role": "policy",
                },
            }
        ]
    )

    assert duplicate_count == 0
    fact = clean_facts[0]
    assert fact["chapter_id"] == "ch_01"
    assert fact["requirement_id"] == "H1_policy"
    assert fact["search_task_id"] == "task_ch01_h1_policy"
    assert fact["hypothesis_id"] == "H1"
    assert fact["proof_role"] == "policy"
    assert fact["analysis_role"] == "policy"
    assert fact["lineage"]["chapter_id"] == "ch_01"


def test_llm_analysis_input_maps_hypothesis_aliases_and_drops_unknown_chapters(monkeypatch):
    monkeypatch.setenv("ARTIFACT_LEDGER_ANALYSIS_CONTEXT_ENABLED", "false")
    evidence_package = {
        "query": "Low-altitude economy report",
        "research_plan": {
            "hypotheses": [
                {"hypothesis_id": "H1", "statement": "Demand is real"},
                {"hypothesis_id": "H2", "statement": "Supply is fragmented"},
            ]
        },
        "report_blueprint": {
            "chapters": [
                {"chapter_id": "ch_01", "chapter_title": "Demand validation"},
                {"chapter_id": "ch_02", "chapter_title": "Supply pattern"},
            ]
        },
        "chapter_evidence_packages": [
            {"chapter_id": "ch_01", "chapter_title": "Demand validation"},
            {"chapter_id": "ch_02", "chapter_title": "Supply pattern"},
        ],
        "analysis_ready_evidence": [
            {
                "evidence_id": "EV-H1",
                "chapter_id": "H1",
                "fact": "Policy documents describe low-altitude logistics pilots in 2025.",
                "source": {
                    "title": "Official policy notice",
                    "url": "https://gov.cn/policy/low-altitude-logistics",
                    "main_text": "Policy documents describe low-altitude logistics pilots in 2025.",
                },
                "source_id": "SRC-1",
                "source_level": "A",
                "allowed_use": "supporting_context",
                "proof_role": "policy",
                "analysis_role": "policy",
                "requirement_id": "H1_policy",
                "search_task_id": "task_ch01_h1_policy",
            },
            {
                "evidence_id": "EV-UNKNOWN",
                "chapter_id": "H9",
                "fact": "This unrelated fact must not create an H9 analysis chapter.",
                "source": {
                    "title": "Media article",
                    "url": "https://news.cn/article/unrelated",
                    "main_text": "This unrelated fact must not create an H9 analysis chapter.",
                },
                "source_id": "SRC-9",
                "source_level": "C",
                "allowed_use": "supporting_context",
                "proof_role": "case",
                "analysis_role": "case",
            },
        ],
    }

    payload = build_llm_analysis_input_v2(evidence_package, {})

    assert [chapter["chapter_id"] for chapter in payload["chapters"]] == ["ch_01"]
    chapter = payload["chapters"][0]
    assert chapter["allowed_evidence_ids"] == ["EV-H1"]
    assert chapter["fact_cards"][0]["chapter_id"] == "ch_01"
    assert chapter["fact_cards"][0]["requirement_id"] == "H1_policy"
    assert chapter["fact_cards"][0]["proof_role"] == "policy"
    assert chapter["fact_cards"][0]["analysis_role"] == "policy"


def test_llm_analysis_input_role_balances_fact_cards(monkeypatch):
    monkeypatch.setenv("ARTIFACT_LEDGER_ANALYSIS_CONTEXT_ENABLED", "false")
    monkeypatch.setenv("BRAIN_LLM_ANALYSIS_MAX_FACTS_PER_CHAPTER", "3")
    metrics = [
        {
            "evidence_id": f"EV-M{index}",
            "chapter_id": "ch_01",
            "fact": f"Market metric signal {index} is disclosed for 2025.",
            "source": {
                "title": f"Official metric source {index}",
                "url": f"https://gov.cn/metric/{index}",
                "main_text": f"Market metric signal {index} is disclosed for 2025.",
            },
            "source_id": f"SRC-M{index}",
            "source_level": "A",
            "allowed_use": "supporting_context",
            "proof_role": "metric",
            "analysis_role": "metric",
        }
        for index in range(1, 6)
    ]
    counter = {
        "evidence_id": "EV-C1",
        "chapter_id": "ch_01",
        "fact": "A media report describes delivery delays and cost pressure in pilot projects.",
        "source": {
            "title": "Industry media risk report",
            "url": "https://news.cn/risk/low-altitude-delays",
            "main_text": "A media report describes delivery delays and cost pressure in pilot projects.",
        },
        "source_id": "SRC-C1",
        "source_level": "C",
        "allowed_use": "supporting_context",
        "proof_role": "counter",
        "analysis_role": "counter",
    }
    evidence_package = {
        "report_blueprint": {"chapters": [{"chapter_id": "ch_01", "chapter_title": "Demand validation"}]},
        "analysis_ready_evidence": [*metrics, counter],
    }

    payload = build_llm_analysis_input_v2(evidence_package, {})

    cards = payload["chapters"][0]["fact_cards"]
    assert len(cards) == 3
    assert "counter" in {card["proof_role"] for card in cards}
