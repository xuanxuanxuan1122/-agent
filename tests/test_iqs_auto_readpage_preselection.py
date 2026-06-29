from rag_pipeline.agents.web_analysis_agent import select_auto_readpage_urls


def test_auto_readpage_soft_rerank_prefers_traceable_sources_before_content_platforms(monkeypatch):
    monkeypatch.setenv("IQS_AUTO_READPAGE_ENABLED", "true")
    monkeypatch.setenv("IQS_AUTO_READPAGE_TOP_N", "3")
    monkeypatch.delenv("IQS_AUTO_READPAGE_MIN_SCORE", raising=False)

    urls = select_auto_readpage_urls(
        [
            {
                "title": "人形机器人融资681亿元超去年全年，量产兑现的下一个瓶颈在哪？",
                "summary": "今日头条转载，包含融资和量产瓶颈讨论。",
                "url": "https://m.toutiao.com/article/123",
            },
            {
                "title": "2026年中国具身智能市场规模及相关上市企业分布情况预测分析",
                "summary": "中商产业研究院显示，2025年市场规模约9150亿元。",
                "url": "https://www.askci.com/news/chanye/20260610/example.shtml",
            },
            {
                "title": "产业观察：具身智能加速进厂融资井喷",
                "summary": "人民网经济科技频道报道产业应用、订单、融资和商业化风险。",
                "url": "https://finance.people.com.cn/n1/2026/0610/c1004-example.html",
            },
            {
                "title": "杭州市经济和信息化局关于具身智能机器人产业建议的答复",
                "summary": "到2027年底实现整机企业总产值超200亿元，产业链总产值超300亿元。",
                "url": "https://jxj.hangzhou.gov.cn/art/2026/art_example.html",
            },
            {
                "title": "搜狐转载：具身智能市场爆发",
                "summary": "移动端转载页。",
                "url": "https://m.sohu.com/a/123456",
            },
        ],
        search_options={
            "proof_role": "metric",
            "lane_targets": ["official_data", "market_research", "risk_counter"],
            "source_priority": ["official_data", "market_research", "mainstream_media"],
        },
    )

    assert urls == [
        "https://jxj.hangzhou.gov.cn/art/2026/art_example.html",
        "https://finance.people.com.cn/n1/2026/0610/c1004-example.html",
        "https://www.askci.com/news/chanye/20260610/example.shtml",
    ]


def test_auto_readpage_keeps_content_platform_as_fallback_clues_when_no_better_source(monkeypatch):
    monkeypatch.setenv("IQS_AUTO_READPAGE_ENABLED", "true")
    monkeypatch.setenv("IQS_AUTO_READPAGE_TOP_N", "2")
    monkeypatch.delenv("IQS_AUTO_READPAGE_MIN_SCORE", raising=False)

    urls = select_auto_readpage_urls(
        [
            {
                "title": "具身智能机器人落地案例：工厂巡检与订单",
                "summary": "内容平台线索，含案例、订单、场景关键词。",
                "url": "https://m.toutiao.com/article/456",
            },
            {
                "title": "具身智能商业化难点与成本",
                "summary": "博客线索，讨论ROI、成本、可靠性。",
                "url": "https://blog.csdn.net/example/article/details/1",
            },
        ],
        search_options={"proof_role": "case", "lane_targets": ["customer_case", "risk_counter"]},
    )

    assert urls == [
        "https://m.toutiao.com/article/456",
        "https://blog.csdn.net/example/article/details/1",
    ]


def test_auto_readpage_diversifies_source_buckets_when_good_sources_compete(monkeypatch):
    monkeypatch.setenv("IQS_AUTO_READPAGE_ENABLED", "true")
    monkeypatch.setenv("IQS_AUTO_READPAGE_TOP_N", "4")
    monkeypatch.setenv("IQS_AUTO_READPAGE_MAX_PER_SOURCE_BUCKET", "2")
    monkeypatch.delenv("IQS_AUTO_READPAGE_MIN_SCORE", raising=False)

    urls = select_auto_readpage_urls(
        [
            {
                "title": "无锡机器人产业政策与具身智能基地",
                "summary": "政府页面，政策、场景、产业链。",
                "url": "https://www.wuxi.gov.cn/doc/1.shtml",
            },
            {
                "title": "北京机器人产业政策与人形机器人示范",
                "summary": "政府页面，政策、规划、产业链。",
                "url": "https://www.beijing.gov.cn/doc/2.html",
            },
            {
                "title": "杭州具身智能机器人产业建议答复",
                "summary": "政府页面，2027年整机企业总产值超200亿元。",
                "url": "https://jxj.hangzhou.gov.cn/art/3.html",
            },
            {
                "title": "产业观察：具身智能加速进厂融资井喷",
                "summary": "人民网经济科技频道，订单、融资、商业化风险。",
                "url": "https://finance.people.com.cn/n1/2026/0610/example.html",
            },
            {
                "title": "2026年中国具身智能市场规模预测",
                "summary": "中商产业研究院，市场规模、增长率、上市公司分布。",
                "url": "https://www.askci.com/news/chanye/example.shtml",
            },
        ],
        search_options={
            "proof_role": "metric",
            "lane_targets": ["official_data", "market_research", "risk_counter"],
            "source_priority": ["official_data", "market_research", "mainstream_media"],
        },
    )

    assert "https://www.wuxi.gov.cn/doc/1.shtml" in urls[:2]
    assert "https://finance.people.com.cn/n1/2026/0610/example.html" in urls
    assert "https://www.askci.com/news/chanye/example.shtml" in urls
    assert len([url for url in urls if ".gov.cn" in url]) <= 2


def test_auto_readpage_preserves_industry_research_when_media_sources_score_higher(monkeypatch):
    monkeypatch.setenv("IQS_AUTO_READPAGE_ENABLED", "true")
    monkeypatch.setenv("IQS_AUTO_READPAGE_TOP_N", "4")
    monkeypatch.setenv("IQS_AUTO_READPAGE_MAX_PER_SOURCE_BUCKET", "2")
    monkeypatch.delenv("IQS_AUTO_READPAGE_MIN_SCORE", raising=False)

    urls = select_auto_readpage_urls(
        [
            {
                "title": "无锡机器人产业政策与具身智能基地",
                "summary": "政府页面，政策、场景、产业链。",
                "url": "https://www.wuxi.gov.cn/doc/1.shtml",
            },
            {
                "title": "北京机器人产业政策与人形机器人示范",
                "summary": "政府页面，政策、规划、产业链。",
                "url": "https://www.beijing.gov.cn/doc/2.html",
            },
            {
                "title": "新华社：具身智能产业订单、融资、场景全面提速",
                "summary": "主流媒体，订单、融资、商业化、风险、成本、市场规模。",
                "url": "https://www.news.cn/politics/example.html",
            },
            {
                "title": "人民网：具身智能进厂融资井喷，商业化风险仍待观察",
                "summary": "主流媒体，订单、融资、商业化、风险、成本、市场规模。",
                "url": "https://finance.people.com.cn/n1/example.html",
            },
            {
                "title": "中商产业研究院：2026年中国具身智能市场规模预测",
                "summary": "行业研究，2025年市场规模约9150亿元，2026年将达到10904亿元。",
                "url": "https://www.askci.com/news/chanye/example.shtml",
            },
            {
                "title": "今日头条：具身智能机器人订单融资风险成本全景梳理",
                "summary": "内容平台，订单、融资、商业化、风险、成本、市场规模、案例。",
                "url": "https://m.toutiao.com/article/999",
            },
        ],
        search_options={
            "proof_role": "metric",
            "lane_targets": ["official_data", "market_research", "risk_counter"],
            "source_priority": ["official_data", "market_research", "mainstream_media"],
        },
    )

    assert "https://www.askci.com/news/chanye/example.shtml" in urls
    assert len([url for url in urls if "people.com.cn" in url or "news.cn" in url]) <= 1
    assert "https://m.toutiao.com/article/999" not in urls
