from rag_pipeline.agents.evidence_curator_agent import curate_evidence_batch


def test_curator_keeps_directional_media_signal_with_boundary():
    evidence = [
        {
            "evidence_id": "EV-1",
            "fact": "中证网报道称，Omdia数据显示2025年中国厂商在人形机器人出货量方面领先。",
            "source_level": "C",
            "allowed_use": "directional_signal",
            "source": {"url": "https://www.cs.com.cn/news/humanoid", "title": "中证网报道"},
        }
    ]

    result = curate_evidence_batch(evidence, query="中国人形机器人产业商业化机会与风险分析")

    assert result["status"] == "ready"
    assert result["curated_evidence_count"] == 1
    note = result["curated_evidence"][0]
    assert note["evidence_id"] == "EV-1"
    assert note["can_support_claim"] is True
    assert note["claim_strength_hint"] == "directional"
    assert "directional" in note["evidence_use_level"]
    assert note["limitations"]


def test_curator_blocks_dirty_login_evidence():
    evidence = [
        {
            "evidence_id": "EV-login",
            "fact": "Login Cookie Privacy Policy Please enable JavaScript",
            "source": {"url": "https://www.acme-research.cn/login"},
        }
    ]

    result = curate_evidence_batch(evidence, query="中国人形机器人")

    assert result["curated_evidence_count"] == 0
    assert result["dirty_blocked_count"] == 1
    assert result["dirty_blocked"][0]["dirty_gate"]["status"] == "blocked"


def test_curator_dedupes_same_fact_by_text_and_source():
    evidence = [
        {
            "evidence_id": "EV-a",
            "fact": "2025年全球人形机器人出货约1.3万台，中国占90%份额。",
            "source": {"url": "https://www.omdia-research.cn/a"},
        },
        {
            "evidence_id": "EV-b",
            "fact": "2025年全球人形机器人出货约1.3万台，中国占90%份额。",
            "source": {"url": "https://www.omdia-research.cn/a"},
        },
    ]

    result = curate_evidence_batch(evidence, query="中国人形机器人")

    assert result["curated_evidence_count"] == 1
    assert result["deduped_count"] == 1
    assert result["curated_evidence"][0]["merged_evidence_ids"] == ["EV-a", "EV-b"]
