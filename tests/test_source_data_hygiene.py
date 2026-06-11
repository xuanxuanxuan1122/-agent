"""Regressions for the upstream dirty-data factories (P-1) and the review
layer writing diagnostics into writer-facing fields (P0).

Live evidence: a filename suffix "-240825" became metric value -240825, the
"26" in "published on July 26" became source_check=26, the IQS lane role
technology_product became a metric name, and the semantic-judge English note
was rendered verbatim into a published report body.
"""

from rag_pipeline.agents.brain_agent import _source_by_text, _structured_evidence_to_raw_points
from rag_pipeline.agents.chapter_evidence_builder import (
    _internal_metric_name,
    _metric_value_carries_meaning,
    _report_fact_sentence,
)
from rag_pipeline.agents.claim_builder_agent import _section_title_from_claim, _writer_safe_boundary_items
from rag_pipeline.agents.evidence_merger import (
    NUMERIC_RE,
    _clean_unit_text,
    _numeric_match_is_artifact,
    _numeric_norm,
    _strip_document_noise,
)


def test_filename_date_suffix_is_not_a_metric_value():
    text = "爱分析报告：智慧灯塔，照亮企业Al Agent实施明路-240825-研报"
    matches = [m for m in NUMERIC_RE.finditer(text) if m.group("number")]
    assert matches
    assert all(_numeric_match_is_artifact(text, m) for m in matches)


def test_bare_long_integers_are_artifacts_but_real_metrics_survive():
    bare = "市场规模为5084889"
    match = NUMERIC_RE.search(bare)
    assert _numeric_match_is_artifact(bare, match) is True
    real = "2025年中国生成式AI企业应用市场规模将达到629亿元"
    kept = [
        m for m in NUMERIC_RE.finditer(real)
        if m.group("unit") and not _numeric_match_is_artifact(real, m)
    ]
    assert [m.group("number") for m in kept] == ["629"]


def test_numeric_norm_never_emits_unknown_unit():
    for value in ("26", "240825", "50套", "471亿美元"):
        _, unit_key = _numeric_norm(value)
        assert unit_key != "unknown"
    assert _clean_unit_text("unknown") == ""
    assert _clean_unit_text("percent") == "percent"


def test_document_noise_strips_credentials_and_contacts():
    noisy = (
        "摘要：深 ### 超配（维持） ## 度 研 ## 究 2025 年 8 月 28 日 卢芷心 "
        "SAC 执业证书编号： S0340524100001 电话：0769-22119297 邮箱： luzhixin@dgzq.com.cn 行业核心观点不变"
    )
    cleaned = _strip_document_noise(noisy)
    assert "S0340524100001" not in cleaned
    assert "0769-22119297" not in cleaned
    assert "dgzq.com.cn" not in cleaned
    assert "###" not in cleaned
    assert "行业核心观点不变" in cleaned


def test_raw_points_do_not_brand_qualitative_lines_with_role_metric():
    points = _structured_evidence_to_raw_points(
        "【竞争对比】企业级AI Agent开发平台和教程资源日趋丰富，形成了活跃的技术开发生态，市场投入持续加大",
        sources=[],
        dimension="dim",
        proof_role="technology_product",
    )
    assert points
    for point in points:
        assert point["metric"] == ""
        assert point["proof_role"] == "technology_product"
        assert point["tag"] == "竞争对比"
        assert point["source_binding"] == "unbound"


def test_fuzzy_source_binding_requires_strong_overlap_and_is_flagged():
    sources = [
        {"title": "另一份无关报告", "snippet": "新能源汽车出口数据", "url": "https://example.org/a"},
    ]
    # Two shared numbers used to reach the old threshold of 5 and stitch the
    # line onto an unrelated source.
    weak_line = "2025年某领域增长12%相关讨论"
    assert _source_by_text(sources, weak_line) == {}


def test_internal_metric_names_never_render_as_indicator_sentences():
    for name in ("source_check", "technology_product", "竞争对比", "政策目标", "关键事实", "数据点"):
        assert _internal_metric_name(name) is True
    assert _internal_metric_name("市场规模") is False
    item = {
        "metric": "source_check",
        "value": "26",
        "fact": "2025世界人工智能大会发布全球AI治理行动计划，标志着AI治理进入国际协调新阶段",
        "source_level": "A",
        "source_url": "https://www.fmprc.gov.cn/x",
    }
    sentence = _report_fact_sentence(item, item["fact"])
    assert "source_check" not in sentence
    assert "26" not in sentence or "治理" in sentence


def test_unit_enum_never_leaks_into_metric_sentence():
    item = {
        "metric": "市场规模",
        "value": "471亿美元",
        "unit": "currency_usd",
        "scope": "中国",
        "fact": "据报告，中国市场规模为471亿美元，保持高速增长态势",
        "source_level": "B",
        "source_url": "https://example.org/r",
    }
    sentence = _report_fact_sentence(item, item["fact"])
    assert "currency_usd" not in sentence
    assert "471亿美元" in sentence


def test_metric_value_meaning_gate():
    assert _metric_value_carries_meaning("471亿美元", "") is True
    assert _metric_value_carries_meaning("12%", "") is True
    assert _metric_value_carries_meaning("26", "") is False
    assert _metric_value_carries_meaning("-240825", "") is False
    assert _metric_value_carries_meaning("88", "percent") is True


def test_writer_safe_boundary_items_drop_pipeline_diagnostics():
    items = [
        "semantic judge found only partial support; keep as cautious directional analysis until stronger evidence is bound",
        "metric fields incomplete: unit; use only as a directional signal until repaired",
        "风险案例集中于农业等非办公室Agent场景，需限制外推范围",
    ]
    safe = _writer_safe_boundary_items(items)
    assert safe == ["风险案例集中于农业等非办公室Agent场景，需限制外推范围"]


def test_section_title_from_claim_is_complete_phrase():
    assert _section_title_from_claim(
        "技术成熟度是影响AI Agent进入生产流程的关键变量，它直接关系到工具调用、权限、安全和部署稳定性。"
    ) == "技术成熟度是影响AI Agent进入生产流程的关键变量"
    assert _section_title_from_claim("短") == ""
