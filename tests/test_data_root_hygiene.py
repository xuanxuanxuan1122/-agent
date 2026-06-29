from rag_pipeline.observability.data_root_hygiene import (
    inspect_cache_hygiene,
    inspect_evidence_analysis_hygiene,
    inspect_source_identity_hygiene,
    inspect_source_hygiene,
)


def test_source_hygiene_flags_dirty_metric_fragments_but_not_policy_or_media():
    dirty = inspect_source_hygiene(
        [
            {
                "title": "AI Agent 市场研究",
                "url": "https://example.org/report",
                "mainText": "该页面写成成本在2000为70%，这一指标用于判断市场成熟度。转化愿景为现实 引言",
            },
            {
                "title": "国务院关于人工智能治理的政策文件",
                "url": "https://www.gov.cn/zhengce/example",
                "content": "政策明确提出加强人工智能治理、促进产业发展。",
                "source_type": "policy",
            },
            {
                "title": "权威媒体报道 AI Agent 落地案例",
                "url": "https://finance.eastmoney.com/a/example.html",
                "snippet": "报道披露多家企业正在推进 AI Agent 试点。",
                "source_type": "media",
            },
        ]
    )

    assert dirty["dirty_item_count"] == 1
    assert dirty["reason_counts"]["malformed_metric_span"] == 1
    assert dirty["reason_counts"]["source_heading_fragment"] == 1
    assert dirty["reason_counts"].get("policy_source", 0) == 0
    assert dirty["reason_counts"].get("media_source", 0) == 0
    assert dirty["clean_item_count"] == 2
    assert dirty["diagnostic_only"] is True
    assert dirty["public_text_allowed"] is False


def test_evidence_analysis_hygiene_counts_dirty_values_and_missing_lineage():
    summary = inspect_evidence_analysis_hygiene(
        [
            {
                "evidence_id": "EV-1",
                "fact": "公开材料显示 AI Agent 正进入企业工作流。",
                "metric": "数据指标",
                "value": "240825",
                "source": {"title": "报告", "url": "https://example.org/r"},
            },
            {
                "evidence_id": "EV-2",
                "fact": "政策提出促进人工智能产业发展。",
                "requirement_id": "REQ-2",
                "source_id": "SRC-2",
                "search_task_id": "T-2",
                "source": {"title": "政策文件", "url": "https://www.gov.cn/policy"},
            },
        ]
    )

    assert summary["dirty_item_count"] == 1
    assert summary["missing_lineage_counts"]["requirement_id"] == 1
    assert summary["missing_lineage_counts"]["source_id"] == 1
    assert summary["missing_lineage_counts"]["search_task_id"] == 1
    assert summary["reason_counts"]["generic_metric_name"] == 1
    assert summary["reason_counts"]["artifact_like_value"] == 1
    assert summary["reason_counts"].get("policy_source", 0) == 0


def test_evidence_analysis_hygiene_does_not_flag_official_homepage_chrome_words():
    summary = inspect_evidence_analysis_hygiene(
        [
            {
                "evidence_id": "EV-OFFICIAL-DATA",
                "fact": (
                    "\u8bbe\u4e3a\u9996\u9875 | \u90ae\u7bb1\u767b\u5f55 | "
                    "\u7f51\u7ad9\u65e0\u969c\u788d\u3002"
                    "\u5e02\u573a\u76d1\u7ba1\u5927\u6570\u636e\u4e2d\u5fc3\u663e\u793a\uff0c"
                    "\u622a\u81f32024\u5e7412\u6708\u5e95\uff0c\u4f01\u4e1a\u6570\u91cf"
                    "\u8f832020\u5e74\u5e95\u589e\u957f206.73%\u3002"
                ),
                "metric": "\u589e\u901f",
                "value": "206.73%",
                "unit": "%",
                "period": "2024-12",
                "requirement_id": "H1_metric",
                "source_id": "SRC-1",
                "search_task_id": "T-1",
                "source": {"title": "Official data", "url": "https://www.gov.cn/robot"},
            }
        ]
    )

    assert summary["dirty_item_count"] == 0
    assert "web_chrome_or_login" not in summary["reason_counts"]


def test_cache_hygiene_surfaces_polluted_hits_and_quarantine_counts():
    summary = inspect_cache_hygiene(
        {
            "cache_hit_count": 5,
            "cache_miss_count": 2,
            "quarantined_count": 1,
            "polluted_count": 2,
            "hit_items": [
                {"cache_key": "good", "status": "ok"},
                {"cache_key": "bad", "status": "polluted", "source_url": "https://example.org/bad"},
            ],
        }
    )

    assert summary["cache_hit_count"] == 5
    assert summary["polluted_count"] == 2
    assert summary["quarantined_count"] == 1
    assert summary["dirty_hit_count"] == 1
    assert summary["reason_counts"]["cache_polluted_count"] == 2
    assert summary["reason_counts"]["cache_quarantined_count"] == 1
    assert summary["reason_counts"]["cache_hit_polluted"] == 1


def test_cache_hygiene_tolerates_non_numeric_counters():
    summary = inspect_cache_hygiene(
        {
            "cache_hit_count": "not-a-number",
            "polluted_count": "bad",
            "quarantined_count": None,
            "hit_items": [{"cache_key": "bad", "status": "poisoned"}],
        }
    )

    assert summary["cache_hit_count"] == 0
    assert summary["polluted_count"] == 0
    assert summary["dirty_hit_count"] == 1
    assert summary["reason_counts"]["cache_hit_polluted"] == 1


def test_source_identity_hygiene_flags_specific_title_reused_across_many_hosts():
    summary = inspect_source_identity_hygiene(
        [
            {
                "title": "CAICT AI Agent Technical Development Report",
                "url": "https://www.caict.ac.cn/report.pdf",
            },
            {
                "title": "CAICT AI Agent Technical Development Report",
                "url": "https://data.beijing.gov.cn/dataset.html",
            },
            {
                "title": "CAICT AI Agent Technical Development Report",
                "url": "https://blog.example.net/copied-title",
            },
        ]
    )

    assert summary["dirty_item_count"] == 3
    assert summary["reason_counts"]["same_title_many_hosts"] == 3
    assert summary["reason_counts"]["source_title_url_mismatch"] == 3
    assert summary["reason_counts"]["fallback_title_reused"] == 3
    assert summary["suspicious_title_groups"][0]["host_count"] == 3


def test_source_identity_hygiene_ignores_generic_titles_across_hosts():
    summary = inspect_source_identity_hygiene(
        [
            {
                "title": "Official AI Agent Statistics",
                "url": "https://rsj.qj.gov.cn/view/gsgg/144714.html",
            },
            {
                "title": "Official AI Agent Statistics",
                "url": "https://www.stats.gov.cn/tjsj/zxfb/202605/t20260501.html",
            },
            {
                "title": "Official AI Agent Statistics",
                "url": "https://data.beijing.gov.cn/report.html",
            },
        ]
    )

    assert summary["dirty_item_count"] == 0
    assert summary["reason_counts"] == {}


def test_evidence_analysis_hygiene_counts_non_claim_status_leaks():
    summary = inspect_evidence_analysis_hygiene(
        [
            {
                "evidence_id": "EV-DIAG",
                "fact": "A traceable diagnostic note should not be in analysis ready.",
                "allowed_use": "diagnostic_only",
                "analysis_readiness": "clue_only",
                "content_shape_issues": ["pdf_table_or_header_fragment"],
                "requirement_id": "REQ-1",
                "source_id": "SRC-1",
                "search_task_id": "T-1",
                "source": {"title": "Diagnostic", "url": "https://example.org/diag"},
            },
            {
                "evidence_id": "EV-CLEAN",
                "fact": "A media article reports enterprises testing workflow assistants in 2026.",
                "requirement_id": "REQ-2",
                "source_id": "SRC-2",
                "search_task_id": "T-2",
                "source": {"title": "Media report", "url": "https://example.org/report"},
            },
        ]
    )

    assert summary["dirty_item_count"] == 1
    assert summary["reason_counts"]["diagnostic_only_in_analysis_ready"] == 1
    assert summary["reason_counts"]["clue_only_in_analysis_ready"] == 1
    assert summary["reason_counts"]["shape_issue_in_analysis_ready"] == 1
