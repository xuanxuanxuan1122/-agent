from __future__ import annotations

from rag_pipeline.agents.readpage_fact_extractor_agent import validate_extracted_fact_payload
from rag_pipeline.contracts.public_text_guard import public_text_quality


def test_public_text_guard_rejects_web_chrome_artifacts():
    dirty = "!(/_next/static/media/logo.abb103bc.png) 产品 ! 产品 ADP 知识引擎 资源 登录"

    result = public_text_quality(dirty)

    assert result["ok"] is False
    assert result["severity"] == "reject"
    assert "markdown_image" in result["reasons"]
    assert "next_static_asset" in result["reasons"]


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
