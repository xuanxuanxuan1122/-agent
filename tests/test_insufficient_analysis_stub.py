"""P0 guardrail: when LLM analysis yields zero usable claims, the report must
be a short honest stub, not a fluent-but-vacuous deterministic long report."""
from __future__ import annotations

from rag_pipeline.flows.report.full_report import (
    _build_insufficient_stub_markdown,
    _insufficient_analysis_delivery_action,
    _insufficient_analysis_signal,
    _sync_analysis_repair_priorities_to_evidence_package,
)

_TEMPLATE_FINGERPRINT = "是把事实转成判断的核心连接点"


def _writer_report(*, final_source: str, usable_claims: int) -> dict:
    return {
        "structured_analysis": {
            "analysis_stage_diagnostics": {
                "uses_llm_analysis": final_source == "llm_evidence_analysis",
                "final_analysis_source": final_source,
                "llm_usable_claim_count": usable_claims,
                "llm_analysis_status": "invalid_output" if usable_claims == 0 else "success",
                "quality_path_degradation_reason": "invalid_output" if usable_claims == 0 else "",
            }
        },
        "source_registry": [
            {"title": "Official A source", "url": "https://a.example.com", "source_level": "A"},
            {"title": "Named research B", "url": "https://b.example.com", "source_level": "B"},
            {"title": "Low-grade C", "url": "https://c.example.com", "source_level": "C"},
        ],
        "qa_result": {"blocking_followups": [{"message": "缺少官方口径的市场规模数据"}]},
    }


def test_zero_usable_claims_triggers_short_honest_stub():
    wr = _writer_report(final_source="deterministic_rebuild", usable_claims=0)
    sig = _insufficient_analysis_signal(wr)
    assert sig["insufficient"] is True

    stub = _build_insufficient_stub_markdown("AI Agent企业落地机会与风险", wr, sig["diagnostics"])
    # short + honest, never the vacuous template filler
    assert len(stub) < 1500
    assert _TEMPLATE_FINGERPRINT not in stub
    assert "未达可发布" in stub
    assert "已掌握的来源" in stub and "主要缺口" in stub
    # prefers A/B sources and surfaces the real gap
    assert "Official A source" in stub and "Named research B" in stub
    assert "缺少官方口径的市场规模数据" in stub


def test_real_llm_claims_do_not_trigger_stub():
    wr = _writer_report(final_source="llm_evidence_analysis", usable_claims=8)
    assert _insufficient_analysis_signal(wr)["insufficient"] is False


def test_zero_usable_claims_keeps_existing_fact_backed_formal_report():
    wr = _writer_report(final_source="deterministic_rebuild", usable_claims=0)
    markdown = "\n".join(
        [
            "# AI Agent企业落地机会与风险",
            "",
            "## 已有证据能够支持的事实观察",
            "公开资料显示，企业 AI 应用已经从试点工具进入流程自动化探索阶段[1]。这一判断只能作为方向性观察，不能直接推导出市场规模。",
            "",
            "## 证据边界",
            "当前材料仍缺少统一口径的市场规模、渗透率和失败案例，因此报告应标记为内部研判而非 Clean 发布。",
            "在这种状态下，报告可以保留事实观察、证据边界和下一步补证任务，但不能把方向性观察包装成强结论。",
            "例如，企业流程自动化探索可以说明需求正在形成，却不能单独证明整体市场规模、商业化速度或竞争格局已经确定。",
            "因此正文应把已验证事实、可写判断和不可写判断分开呈现：已验证事实进入事实观察，可写判断进入方向性研判，不可写判断进入补正清单。",
            "这种交付形态比短稿更适合内部阅读，因为它保留了当前证据能支持的上下文，也明确告诉读者哪些结论仍然不能发布。",
            "同时，报告必须继续保留引用和来源附录，避免在降级交付时把硬事实变成无来源陈述。",
            "后续补证应优先围绕市场规模口径、渗透率、采购/落地案例、失败或延期案例以及监管约束展开。",
            "只要这些缺口没有补齐，该报告就只能作为 formal_scored 或 internal_review 使用，不能进入 Clean publishable。",
            "",
            "## 来源附录",
            "- [1] Official A source | https://a.example.com",
        ]
    )

    action = _insufficient_analysis_delivery_action(markdown, wr)

    assert action["mode"] == "limited_evidence_formal_report"
    assert action["replace_with_stub"] is False
    assert action["report_status"] == "formal_scored"
    assert action["delivery_tier"] == "limited_evidence_formal_report"


def test_zero_usable_claims_still_uses_stub_for_empty_or_tiny_report():
    wr = _writer_report(final_source="deterministic_rebuild", usable_claims=0)

    action = _insufficient_analysis_delivery_action("# Empty\n\n需要补证。", wr)

    assert action["mode"] == "insufficient_analysis_stub"
    assert action["replace_with_stub"] is True


def test_full_report_syncs_analysis_repair_priorities_to_evidence_package():
    evidence_package = {"evidence_gap_ledger": []}
    structured_analysis = {
        "evidence_repair_priorities": [
            {
                "schema_version": "claim_support_repair_priority_v1",
                "gap_id": "gap-direct",
                "claim_id": "claim-direct",
                "gap_type": "claim_semantic_support_mismatch",
            }
        ]
    }

    summary = _sync_analysis_repair_priorities_to_evidence_package(evidence_package, structured_analysis)

    assert summary["added_gap_count"] == 1
    assert evidence_package["evidence_gap_ledger"][0]["gap_id"] == "gap-direct"
    assert evidence_package["evidence_gap_ledger"][0]["allowed_for_writing"] is False


def test_missing_diagnostics_never_false_triggers_stub():
    # Defensive: with no analysis diagnostics at all, the source defaults to a
    # non-deterministic label so the stub must NOT fire on absent data.
    assert _insufficient_analysis_signal({})["insufficient"] is False
