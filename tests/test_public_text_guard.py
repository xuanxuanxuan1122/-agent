from __future__ import annotations

from rag_pipeline.agents.readpage_fact_extractor_agent import validate_extracted_fact_payload
from rag_pipeline.agents.claim_builder_agent import run_claim_builder_agent
from rag_pipeline.contracts.public_text_guard import public_text_quality


def test_public_text_guard_rejects_web_chrome_artifacts():
    dirty = "!(/_next/static/media/logo.abb103bc.png) 产品 ! 产品 ADP 知识引擎 资源 登录"

    result = public_text_quality(dirty)

    assert result["ok"] is False
    assert result["severity"] == "reject"
    assert "markdown_image" in result["reasons"]
    assert "next_static_asset" in result["reasons"]


def test_public_text_guard_rejects_spaced_markdown_image_and_portal_chrome():
    dirty = "西浦君谋AI 学习超市 图书馆 IT 融合门户 --- ! [Xi'an Jiaotong-Liverpool University logo]"

    result = public_text_quality(dirty)

    assert result["ok"] is False
    assert result["severity"] == "reject"
    assert "markdown_image" in result["reasons"]
    assert "navigation_chrome" in result["reasons"]


def test_public_text_guard_rejects_eastmoney_navigation_chrome():
    dirty = (
        "[](/newstatic/images/logo.gif)](//www.eastmoney.com/) "
        "[\u6570\u636e\u4e2d\u5fc3](/center/) "
        "[\u5168\u7403\u8d22\u7ecf\u5feb\u8baf](//kuaixun.eastmoney.com/) "
        "[\u884c\u60c5\u4e2d\u5fc3](//quote.eastmoney.com/center/) "
        "[Choice\u6570\u636e](//choice.eastmoney.com)"
    )

    result = public_text_quality(dirty)

    assert result["ok"] is False
    assert result["severity"] == "reject"
    assert "static_asset" in result["reasons"]
    assert "navigation_chrome" in result["reasons"]


def test_readpage_fact_validation_rejects_dirty_public_text():
    payload = {
        "fact_cards": [
            {
                "distilled_fact": "!(/_next/static/media/logo.abb103bc.png) 产品 ! 产品 ADP 知识引擎 资源 登录",
                "fact_type": "case",
            }
        ]
    }

    result = validate_extracted_fact_payload(
        payload,
        source_url="https://cloud.tencent.com/adp",
        source_ref="SRC-ADP",
        source_level="B",
        verification_status="readpage_verified",
        proof_role="case",
    )

    assert result["fact_cards"] == []
    assert result["rejected_spans"][0]["reason"] == "dirty_public_text"


def test_readpage_fact_validation_rejects_eastmoney_navigation_chrome():
    dirty = (
        "[](/newstatic/images/logo.gif)](//www.eastmoney.com/) "
        "[\u6570\u636e\u4e2d\u5fc3](/center/) "
        "[\u5168\u7403\u8d22\u7ecf\u5feb\u8baf](//kuaixun.eastmoney.com/) "
        "[\u884c\u60c5\u4e2d\u5fc3](//quote.eastmoney.com/center/) "
        "[Choice\u6570\u636e](//choice.eastmoney.com)"
    )
    payload = {"fact_cards": [{"distilled_fact": dirty, "fact_type": "source_check"}]}

    result = validate_extracted_fact_payload(
        payload,
        source_url="https://data.eastmoney.com/stockdata/002416.html",
        source_ref="SRC-EASTMONEY",
        source_level="C",
        verification_status="readpage_verified",
        proof_role="source_check",
    )

    assert result["fact_cards"] == []
    assert result["rejected_spans"][0]["reason"] == "dirty_public_text"


def test_claim_builder_does_not_publish_navigation_chrome_fact():
    dirty = (
        "[](/newstatic/images/logo.gif)](//www.eastmoney.com/) "
        "[\u6570\u636e\u4e2d\u5fc3](/center/) "
        "[\u5168\u7403\u8d22\u7ecf\u5feb\u8baf](//kuaixun.eastmoney.com/) "
        "[\u884c\u60c5\u4e2d\u5fc3](//quote.eastmoney.com/center/) "
        "[Choice\u6570\u636e](//choice.eastmoney.com)"
    )
    units = run_claim_builder_agent(
        chapter_evidence_packages=[
            {
                "chapter_id": "ch_01",
                "chapter_title": "\u4f4e\u7a7a\u7ecf\u6d4e\u653f\u7b56\u5f71\u54cd",
                "chapter_question": "\u653f\u7b56\u5982\u4f55\u5f71\u54cd\u65e0\u4eba\u673a\u914d\u9001",
                "supporting_evidence": [
                    {
                        "evidence_id": "EV-DIRTY",
                        "source_ref": "[1]",
                        "fact": dirty,
                        "source_level": "C",
                    }
                ],
                "evidence_items": [
                    {
                        "evidence_id": "EV-DIRTY",
                        "source_ref": "[1]",
                        "fact": dirty,
                        "source_level": "C",
                    }
                ],
            }
        ],
        micro_layouts=[],
        structured_analysis={},
    )

    assert units
    assert all(unit.get("public_render") is False for unit in units)
    assert all(unit.get("omit_from_report") is True for unit in units)


def test_claim_builder_fallback_does_not_inject_ai_agent_for_non_ai_topic():
    units = run_claim_builder_agent(
        chapter_evidence_packages=[
            {
                "chapter_id": "ch_01",
                "chapter_title": "\u4f4e\u7a7a\u7ecf\u6d4e\u653f\u7b56\u4e0e\u65e0\u4eba\u673a\u914d\u9001",
                "chapter_question": "\u653f\u7b56\u5982\u4f55\u5f71\u54cd\u65e0\u4eba\u673a\u914d\u9001\u5546\u4e1a\u5316\uff1f",
                "supporting_evidence": [
                    {
                        "evidence_id": "EV-LOW",
                        "source_ref": "[1]",
                        "fact": "\u4f5b\u5c71\u6210\u7acb\u4f4e\u7a7a\u7ecf\u6d4e\u53d1\u5c55\u516c\u53f8\uff0c\u63a8\u52a8\u65e0\u4eba\u673a\u914d\u9001\u548c\u4f4e\u7a7a\u5e94\u7528\u573a\u666f\u843d\u5730\u3002",
                        "source_level": "C",
                    }
                ],
            }
        ],
        micro_layouts=[],
        structured_analysis={},
    )

    rendered_text = " ".join(
        str(unit.get(key) or "")
        for unit in units
        for key in ("claim", "reasoning", "mechanism", "decision_implication")
    )
    assert "AI Agent" not in rendered_text
