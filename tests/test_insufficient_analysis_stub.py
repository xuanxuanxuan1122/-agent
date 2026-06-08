"""P0 guardrail: when LLM analysis yields zero usable claims, the report must
be a short honest stub, not a fluent-but-vacuous deterministic long report."""
from __future__ import annotations

from rag_pipeline.flows.report.full_report import (
    _build_insufficient_stub_markdown,
    _insufficient_analysis_signal,
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


def test_missing_diagnostics_never_false_triggers_stub():
    # Defensive: with no analysis diagnostics at all, the source defaults to a
    # non-deterministic label so the stub must NOT fire on absent data.
    assert _insufficient_analysis_signal({})["insufficient"] is False
