from rag_pipeline.agents import brain_agent
from rag_pipeline.agents.evidence_curator_agent import curate_evidence_batch
from rag_pipeline.flows.report import full_report


def _package():
    return {
        "run_id": "run-curator-flow",
        "query": "humanoid robot industry analysis",
        "analysis_ready_evidence": [
            {
                "evidence_id": "EV-1",
                "fact": "A vertical media report says humanoid robot makers are moving from demos into small-batch delivery in 2026.",
                "source_level": "C",
                "source": {
                    "url": "https://www.cs.com.cn/news/humanoid-robot-delivery",
                    "title": "Humanoid robot delivery signal",
                },
            },
            {
                "evidence_id": "EV-login",
                "fact": "Login Cookie Privacy Policy Please enable JavaScript",
                "source": {"url": "https://www.example-research.cn/login"},
            },
        ],
    }


def test_full_report_attaches_curated_evidence(monkeypatch, tmp_path):
    monkeypatch.setenv("REPORT_EVIDENCE_CURATOR_ENABLED", "true")
    monkeypatch.setenv("ARTIFACT_LEDGER_ENABLED", "false")
    monkeypatch.setenv("ANALYSIS_MEMORY_CACHE_PATH", str(tmp_path / "analysis_memory"))

    result = full_report._attach_curated_evidence(_package())

    assert result["curated_evidence"]["curated_evidence_count"] == 1
    assert result["evidence_inventory"]["inventory_count"] == 1
    assert result["analysis_shards"][0]["cluster_key"]
    assert result["analysis_memory_cache"]["status"] == "stored"
    assert result["curated_evidence"]["dirty_blocked_count"] == 1
    assert result["metadata"]["curated_evidence_count"] == 1
    assert result["metadata"]["evidence_inventory_count"] == 1


def test_brain_agent_attaches_curated_evidence(monkeypatch, tmp_path):
    monkeypatch.setenv("REPORT_EVIDENCE_CURATOR_ENABLED", "true")
    monkeypatch.setenv("ARTIFACT_LEDGER_ENABLED", "false")
    monkeypatch.setenv("ANALYSIS_MEMORY_CACHE_PATH", str(tmp_path / "analysis_memory"))

    result = brain_agent._attach_curated_evidence_to_package(_package())

    assert result["curated_evidence"]["curated_evidence_count"] == 1
    assert result["evidence_inventory"]["inventory_count"] == 1
    assert result["analysis_shards"][0]["cluster_key"]
    assert result["analysis_memory_cache"]["status"] == "stored"
    assert result["curated_evidence"]["dirty_blocked_count"] == 1
    assert result["metadata"]["curated_input_count"] == 2


def test_curator_classifies_chinese_industry_signals_for_analysis():
    result = curate_evidence_batch(
        [
            {
                "evidence_id": "EV-market",
                "fact": "2026年低空经济市场规模预计继续增长，eVTOL订单金额超过300亿元。",
                "source_level": "B",
                "source": {"url": "https://www.china-air.org.cn/market"},
            },
            {
                "evidence_id": "EV-policy",
                "fact": "地方政府出台低空经济高质量发展实施方案，明确基础设施和监管试点。",
                "source_level": "A",
                "source": {"url": "https://www.gov.cn/policy"},
            },
            {
                "evidence_id": "EV-case",
                "fact": "长三角直升机航线在春运期间出现公交化运营案例，应用场景开始验证。",
                "source_level": "C",
                "source": {"url": "https://www.caacnews.com.cn/case"},
            },
        ],
        query="中国低空经济产业链商业化机会与风险分析",
    )

    by_id = {item["evidence_id"]: item for item in result["curated_evidence"]}
    assert by_id["EV-market"]["fact_type"] == "market_signal"
    assert by_id["EV-market"]["claim_strength_hint"] == "moderate"
    assert "commercialization" in by_id["EV-market"]["usable_for"]
    assert by_id["EV-policy"]["fact_type"] == "policy_signal"
    assert by_id["EV-policy"]["claim_strength_hint"] == "moderate"
    assert by_id["EV-case"]["fact_type"] == "case_signal"
    assert by_id["EV-case"]["claim_strength_hint"] == "directional"


def test_curator_keeps_but_marks_off_topic_adjacent_industry_for_low_altitude_query():
    result = curate_evidence_batch(
        [
            {
                "evidence_id": "EV-low-altitude",
                "fact": "低空经济政策明确支持无人机物流、低空旅游和基础设施建设。",
                "source_level": "A",
                "source": {"url": "https://www.gov.cn/low-altitude-policy"},
            },
            {
                "evidence_id": "EV-humanoid",
                "fact": "中国人形机器人潜在市场空间有望达到22.8万亿元，供应链配套能力完善。",
                "source_level": "B",
                "source": {"url": "https://www.sz.gov.cn/humanoid-robot-market"},
            },
        ],
        query="中国低空经济产业链商业化机会与风险分析",
    )

    by_id = {item["evidence_id"]: item for item in result["curated_evidence"]}
    assert by_id["EV-low-altitude"]["topic_fit"] == "direct"
    assert by_id["EV-low-altitude"]["can_support_claim"] is True
    assert by_id["EV-humanoid"]["topic_fit"] == "off_topic"
    assert by_id["EV-humanoid"]["can_support_claim"] is False
    assert by_id["EV-humanoid"]["evidence_use_level"] == "background_only"
