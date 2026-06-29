from rag_pipeline.contracts.evidence_dirty_gate import evaluate_dirty_gate


def test_dirty_gate_blocks_login_and_navigation_pages():
    item = {
        "evidence_id": "EV-login",
        "fact": "Login Sign in Cookie Privacy Policy Please enable JavaScript",
        "source": {"url": "https://www.acme-research.cn/login", "title": "Login"},
    }

    result = evaluate_dirty_gate(item)

    assert result["status"] == "blocked"
    assert "page_shell_or_login" in result["reasons"]
    assert result["public_text_allowed"] is False


def test_dirty_gate_allows_traceable_low_tier_industry_signal():
    item = {
        "evidence_id": "EV-signal",
        "fact": "前瞻产业研究院称，2025年全球人形机器人出货约1.7万台，产业进入小批量验证阶段。",
        "source_level": "C",
        "source": {"url": "https://www.qianzhan.com/report/detail/abc", "title": "产业研究报告"},
    }

    result = evaluate_dirty_gate(item)

    assert result["status"] == "allowed"
    assert result["reasons"] == []
    assert result["public_text_allowed"] is True


def test_dirty_gate_does_not_block_incomplete_metric_by_itself():
    item = {
        "evidence_id": "EV-metric",
        "fact": "机构预测中国人形机器人市场规模将在2026年突破200亿元。",
        "metric": "市场规模",
        "value": "200亿元",
        "unit": "",
        "period": "",
        "source": {"url": "https://www.research-cn.com/market", "title": "市场预测"},
    }

    result = evaluate_dirty_gate(item)

    assert result["status"] == "allowed"
    assert "metric_incomplete" in result["warnings"]
    assert "metric_incomplete" not in result["reasons"]
