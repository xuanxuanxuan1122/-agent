from __future__ import annotations

from rag_pipeline.agents.final_writer_agent import _source_allowed_for_report
from rag_pipeline.agents.readpage_fact_extractor_agent import validate_extracted_fact_payload
from rag_pipeline.cache.evidence_cache import EvidenceCache
from rag_pipeline.contracts.evidence_quality import apply_evidence_quality_contract, classify_evidence
from rag_pipeline.contracts.quality_gate_policy import advisory_weight_mode, evidence_mode, main_chain_only_mode, public_signal_mode
from rag_pipeline.flows.report.final_audit_agent import _normalize_audit_payload


def test_public_signal_mode_env_aliases(monkeypatch):
    monkeypatch.setenv("REPORT_EVIDENCE_MODE", "public_signal")

    assert evidence_mode() == "public_signal"
    assert public_signal_mode() is True
    assert advisory_weight_mode() is True


def test_main_chain_only_mode_env_alias(monkeypatch):
    monkeypatch.setenv("REPORT_MAIN_CHAIN_ONLY", "true")

    assert main_chain_only_mode() is True


def test_public_signal_mode_keeps_self_media_as_directional_signal(monkeypatch):
    monkeypatch.setenv("REPORT_EVIDENCE_MODE", "public_signal")

    result = apply_evidence_quality_contract(
        {
            "evidence_id": "EV-SOCIAL-1",
            "source_level": "D",
            "source_type": "self_media",
            "source": {
                "url": "https://www.zhihu.com/question/solid-state-battery",
                "title": "固态电池商业化讨论",
                "source_type": "self_media",
            },
            "content": "多位产业链从业者讨论固态电池样车验证和量产节奏。",
            "clean_fact": "多位产业链从业者讨论固态电池样车验证和量产节奏。",
            "proof_role": "case",
            "semantic_status": "ok",
            "confidence": 0.25,
        }
    )

    assert result["allowed_use"] == "supporting_context"
    assert result["appendix_only"] is False
    assert result["analysis_eligible"] is True
    assert result["analysis_role"] == "case"
    assert result["claim_strength_ceiling"] == "strong"
    assert result["evidence_weight_policy"] == "advisory_only"


def test_advisory_weight_mode_does_not_appendix_low_confidence_c_source(monkeypatch):
    monkeypatch.setenv("REPORT_EVIDENCE_MODE", "advisory_weight")

    result = apply_evidence_quality_contract(
        {
            "evidence_id": "EV-C-LOW",
            "source_level": "C",
            "source_type": "vertical_media",
            "source": {
                "url": "https://industry.example.cn/solid-state-battery-signal",
                "title": "固态电池商业化信号",
                "source_type": "vertical_media",
            },
            "content": "垂直行业媒体报道多家企业披露固态电池样车验证和产线计划。",
            "proof_role": "case",
            "semantic_status": "ok",
            "confidence": 0.1,
        }
    )

    assert result["analysis_eligible"] is True
    assert result["appendix_only"] is False
    assert result["allowed_use"] == "supporting_context"
    assert result["evidence_weight_policy"] == "advisory_only"
    assert result["evidence_weight_hint"]["source_level"] == "C"


def test_advisory_weight_mode_does_not_hard_ceiling_d_source(monkeypatch):
    monkeypatch.setenv("REPORT_EVIDENCE_MODE", "advisory_weight")

    result = apply_evidence_quality_contract(
        {
            "evidence_id": "EV-D-SOCIAL",
            "source_level": "D",
            "source_type": "social",
            "source": {
                "url": "https://www.douyin.com/video/example",
                "title": "固态电池试装讨论",
                "source_type": "social",
            },
            "content": "短视频平台公开讨论某电动车固态电池试装和续航表现，适合作为市场现象线索。",
            "proof_role": "case",
            "semantic_status": "ok",
            "confidence": 0.05,
        }
    )

    assert result["analysis_eligible"] is True
    assert result["appendix_only"] is False
    assert result["allowed_use"] == "supporting_context"
    assert result["claim_strength_ceiling"] == "strong"
    assert result["evidence_weight_hint"]["model_instruction"] == "weigh_this_evidence_in_context_do_not_filter_by_source_grade"


def test_advisory_weight_classification_keeps_traceable_d_source(monkeypatch):
    monkeypatch.setenv("REPORT_EVIDENCE_MODE", "advisory_weight")

    result = classify_evidence(
        {
            "source_level": "D",
            "source_type": "social",
            "source_url": "https://www.douyin.com/video/example",
            "source_title": "低空经济现场讨论",
            "fact": "多地低空经济企业开始披露订单试点和场景合作。",
            "allowed_use": "supporting_context",
        }
    )

    assert result.get("status") != "rejected"
    assert result.get("allowed_use") == "supporting_context"
    assert result.get("usage_tier") == "advisory_weight"
    assert result.get("evidence_weight_policy") == "advisory_only"


def test_advisory_weight_rejects_raw_pdf_table_fragment(monkeypatch):
    monkeypatch.setenv("REPORT_EVIDENCE_MODE", "advisory_weight")

    result = classify_evidence(
        {
            "source_url": "https://example.org/report.pdf",
            "source_title": "中国地质 PDF",
            "fact": "PDF：| 第 53 卷第 1 期 | 中 国 地 质 | Vol. 1 | doi: 10",
            "source_level": "A",
        }
    )

    assert result.get("status") == "rejected"
    assert result.get("allowed_use") == "not_for_writing"
    assert "pdf_table_or_header_fragment" in result.get("reasons", [])


def test_advisory_weight_rejects_browser_login_fragment(monkeypatch):
    monkeypatch.setenv("REPORT_EVIDENCE_MODE", "advisory_weight")

    result = classify_evidence(
        {
            "source_url": "https://example.org/login",
            "source_title": "登录后继续访问",
            "fact": "请登录后继续访问 Chrome JavaScript cookie 返回首页 注册 登录",
            "source_level": "B",
        }
    )

    assert result.get("status") == "rejected"
    assert result.get("allowed_use") == "not_for_writing"
    assert "browser_or_login_fragment" in result.get("reasons", [])


def test_readpage_public_signal_accepts_low_authority_traceable_fact(monkeypatch):
    monkeypatch.setenv("REPORT_EVIDENCE_MODE", "public_signal")

    result = validate_extracted_fact_payload(
        {
            "fact_cards": [
                {
                    "subject": "固态电池用户讨论",
                    "distilled_fact": "公开讨论集中在续航、成本和量产时间表。",
                    "fact_type": "case",
                    "source_url": "https://www.zhihu.com/question/solid-state-battery",
                    "source_ref": "S1",
                    "source_level": "D",
                    "source_verification_status": "readpage_verified",
                    "proof_role": "case",
                }
            ]
        },
        source_url="https://www.zhihu.com/question/solid-state-battery",
        source_ref="S1",
        source_level="D",
        verification_status="readpage_verified",
        proof_role="case",
    )

    assert result["fact_cards"]
    assert result["fact_cards"][0]["allowed_use"] == "directional_signal"
    assert not result["rejected_spans"]


def test_source_gate_public_signal_keeps_traceable_self_media(monkeypatch):
    monkeypatch.setenv("REPORT_EVIDENCE_MODE", "public_signal")

    assert (
        _source_allowed_for_report(
            {
                "ref": "[1]",
                "title": "固态电池商业化讨论",
                "url": "https://www.zhihu.com/question/solid-state-battery",
                "source_type": "self_media",
                "source_level": "D",
            },
            query="中国固态电池商业化机会",
        )
        is True
    )


def test_evidence_cache_public_signal_stores_d_level_directional_signal(monkeypatch, tmp_path):
    monkeypatch.setenv("REPORT_EVIDENCE_MODE", "public_signal")
    monkeypatch.setenv("EVIDENCE_CACHE_READ_ENABLED", "true")
    cache = EvidenceCache(tmp_path / "evidence.sqlite")

    summary = cache.store_evidence_from_package(
        query="中国固态电池商业化机会",
        evidence_package={
            "analysis_ready_evidence": [
                {
                    "evidence_id": "EV-D-1",
                    "source_level": "D",
                    "source_type": "self_media",
                    "allowed_use": "directional_signal",
                    "fact": "公开讨论集中在固态电池成本、续航和量产时间表。",
                    "source": {
                        "url": "https://www.zhihu.com/question/solid-state-battery",
                        "title": "固态电池商业化讨论",
                        "source_type": "self_media",
                    },
                    "proof_role": "case",
                }
            ]
        },
    )

    assert summary["stored_count"] == 1
    hits = cache.lookup_evidence(
        {
            "query": "中国固态电池 商业化 讨论",
            "proof_role": "case",
            "topic_anchor_terms": ["固态电池"],
        }
    )
    assert hits
    assert hits[0]["source_level"] == "D"


def test_final_audit_public_signal_softens_weak_source_fatal(monkeypatch):
    monkeypatch.setenv("REPORT_EVIDENCE_MODE", "public_signal")

    result = _normalize_audit_payload(
        {
            "status": "fatal",
            "overall_score": 62,
            "critical_findings": [
                {
                    "type": "weak_source",
                    "severity": "fatal",
                    "message": "单一来源且来源偏弱，建议补充官方数据。",
                }
            ],
            "publish_recommendation": "hold",
        }
    )

    assert result["status"] == "warning"
    assert result["publish_recommendation"] == "publish_with_caveats"
    assert result["critical_findings"][0]["severity"] == "high"
    assert result["critical_findings"][0]["public_signal_downgraded"] is True


def test_final_audit_public_signal_keeps_hard_citation_fatal(monkeypatch):
    monkeypatch.setenv("REPORT_EVIDENCE_MODE", "public_signal")

    result = _normalize_audit_payload(
        {
            "status": "fatal",
            "overall_score": 40,
            "critical_findings": [
                {
                    "type": "citation_issue",
                    "severity": "fatal",
                    "message": "引用来源与正文事实完全不匹配。",
                }
            ],
            "publish_recommendation": "hold",
        }
    )

    assert result["status"] == "fatal"
    assert result["publish_recommendation"] == "hold"
    assert result["critical_findings"][0]["severity"] == "fatal"
