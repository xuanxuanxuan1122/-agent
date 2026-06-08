"""Evidence quality gate: scraped structural markup (HTML headings, table cells,
financial-report table markers) and error-page boilerplate are titles/structure,
not verifiable facts. They must be rejected at extraction so the analyst LLM is
not handed a chapter of junk and forced to abstain (the observed cause of most
chapters producing zero claims)."""
from __future__ import annotations

from rag_pipeline.agents.readpage_fact_extractor_agent import _looks_bad_text


def test_structural_markup_is_rejected():
    junk = [
        "<h5>Knowledge Atlas Technology Joint Stock Company Limited</h5>",
        "<th>[Table_StockNameRptType] 比亚迪 2556.HK 港股公司财务</th>",
        "<td>2024-09-09</td>",
        "[Table_StockName] 营收同比 40%",
        "<title>AI Agent 行业研究报告</title>",
    ]
    for text in junk:
        assert _looks_bad_text(text) is True, f"markup junk not rejected: {text!r}"


def test_error_page_boilerplate_is_rejected():
    for text in [
        "This page isn't working",
        "If the problem continues, contact the site owner.",
        "Please enable JavaScript to continue.",
        "Book a demo to see AI agents in action",
    ]:
        assert _looks_bad_text(text) is True


def test_institutional_publishing_boilerplate_is_rejected():
    # Off-topic "how an agency publishes data" boilerplate binds to chapters and
    # gets glued into paragraphs with mismatched citations (the FinalAudit
    # "引用来源与内容不匹配" fatal). Reject the boilerplate...
    assert _looks_bad_text(
        "国家统计局通过官方网站、数据发布库、《中国统计年鉴》等统计出版物、新闻发布会和两微一端等渠道发布统计数据，"
        "以满足不同用户群体获取统计数据的多样化需求"
    ) is True
    # ...but never a real statistics-bureau DATA fact.
    assert _looks_bad_text("国家统计局数据显示2025年中国AI产业规模达9000亿元，同比增长40%。") is False


def test_real_facts_survive_the_gate():
    good = [
        "Oracle在Fusion Cloud Applications中嵌入专业AI Agent，重塑财务、供应链、HR等部门的工作方式。",
        "客服团队每天重复回答80%的同类问题，AI Agent可提供系统性解决方案。",
        "AI物业经理智能体已覆盖超过300个项目，管理面积超2000万平方米，降低管理成本60-70%。",
    ]
    for text in good:
        assert _looks_bad_text(text) is False, f"real fact wrongly rejected: {text!r}"
