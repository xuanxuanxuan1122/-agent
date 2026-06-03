import asyncio
import inspect
import json

import pytest
import run_full_report
from rag_pipeline.flows.report import reformatter_agent as reformatter_module
from reformatter_agent import (
    CITATION_DENSITY_RULES,
    _quality_issues,
    build_reformatter_payload,
    clean_reformatted_report,
    validate_reformatted_report,
)
from run_full_report import has_legacy_decision_sections


def test_reformatter_strips_fixed_micro_template_labels():
    markdown = """
# 新能源汽车新型材料行业深度研究报告

## 第二章 市场规模与增速

### 2.1 章节判断
新能源汽车新型材料的行情需要先看需求和价格是否同向。

### 2.2 关键事实速览
| 维度 | 核心数据 | 来源 |
| --- | --- | --- |
| 市场规模 | 无可用数据 | 无 |

### 2.3 证据深读

#### 2.3.1 全球口径

**可引用事实**
当前材料显示样本仍然不足。

**机制与边界**
价格、订单和产能口径不一致时，结论不能外推。

**进入综合决策章的变量**
TAM、订单、毛利率。

### 2.4 本章结论
本章只能确认观察边界。
"""

    cleaned = clean_reformatted_report(markdown, sources=[])

    forbidden = [
        "章节判断",
        "关键事实速览",
        "证据深读",
        "全球口径",
        "可引用事实",
        "机制与边界",
        "进入综合决策章的变量",
        "本章结论",
    ]
    for phrase in forbidden:
        assert phrase not in cleaned
    assert "### 2.1" not in cleaned
    assert "新能源汽车新型材料的行情需要先看需求和价格是否同向" in cleaned
    assert has_legacy_decision_sections(cleaned) is False


def test_reformatter_validation_blocks_fixed_micro_template_labels():
    markdown = """
# 报告
## 第二章 市场规模与增速
### 2.1 章节判断
正文。
## 数据来源
"""

    validation = validate_reformatted_report(markdown, [], None)

    assert validation["passed"] is False
    assert any("章节判断" in item for item in validation["forbidden_hits"])


def test_reformatter_prompt_and_quality_gate_are_not_bound_to_old_chapter_numbers():
    source = inspect.getsource(_quality_issues)

    assert "第二章" not in source
    assert "第五章" not in source
    assert "第二章至第六章" not in CITATION_DENSITY_RULES
    assert "每章至少" not in CITATION_DENSITY_RULES
    assert not hasattr(run_full_report, "has_v3_subsections")


def test_reformatter_quality_gate_catches_internal_labels_in_any_body_chapter():
    markdown = """
# 报告
## 第七章 用户采购节奏为什么变慢
正文先解释采购预算和认证周期。
### 7.1 关键事实速览
这里不应该保留旧模板标题。
## 数据来源
"""

    quality = _quality_issues(markdown)

    assert quality["weak_chapter_judgments"]


def test_reformatter_payload_omits_empty_legacy_dimensions():
    payload = build_reformatter_payload(
        {
            "topic": "机器人渠道变化",
            "dimensions": {
                "市场规模与增速": [],
                "定价与渠道验证": [
                    {
                        "text": "经销渠道正在从一次性销售转向租赁和服务合同。",
                        "source": "1",
                    }
                ],
            },
            "sources": [{"id": "1", "title": "渠道调研", "url": "https://example.com"}],
        }
    )

    assert "市场规模与增速" not in payload["evidence_json"]
    assert "定价与渠道验证" in payload["evidence_json"]

def test_reformatter_payload_uses_compact_json():
    payload = build_reformatter_payload(
        {
            "topic": "payload compactness",
            "dimensions": {
                "market": [
                    {"text": "fact one with a numeric signal 10%", "source": "1"},
                    {"text": "fact two with a numeric signal 20%", "source": "2"},
                ],
            },
            "sources": [
                {"id": "1", "title": "Source One", "url": "https://example.com/1"},
                {"id": "2", "title": "Source Two", "url": "https://example.com/2"},
            ],
        },
        max_facts_per_dimension=2,
    )

    assert "\n" not in payload["evidence_json"]
    assert json.loads(payload["evidence_json"])["evidence_items"]


def test_reformatter_payload_applies_global_evidence_cap(monkeypatch):
    monkeypatch.setenv("REPORT_REFORMATTER_MAX_EVIDENCE_ITEMS", "3")
    clean_evidence = {
        "topic": "global cap",
        "dimensions": {
            "market": [{"text": f"market fact {index} 10%", "source": str(index)} for index in range(1, 5)],
            "policy": [{"text": f"policy fact {index} 20%", "source": str(index + 10)} for index in range(1, 5)],
        },
        "sources": [{"id": str(index), "title": f"Source {index}", "url": "https://example.com"} for index in range(1, 20)],
    }

    payload = build_reformatter_payload(clean_evidence, max_facts_per_dimension=4)
    evidence_items = json.loads(payload["evidence_json"])["evidence_items"]

    assert len(evidence_items) == 3


def test_reformatter_blocks_oversized_initial_payload_before_llm(monkeypatch):
    monkeypatch.setenv("REPORT_REFORMATTER_MAX_PAYLOAD_CHARS", "500")
    called = False

    async def fail_if_called(**kwargs):
        nonlocal called
        called = True
        raise AssertionError("LLM should not be called for an oversized reformatter payload")

    monkeypatch.setattr(reformatter_module, "_generate_reformatter_text", fail_if_called)
    clean_evidence = {
        "topic": "oversized reformatter payload",
        "dimensions": {
            "market": [
                {
                    "text": "very long evidence " * 80,
                    "source": "1",
                }
            ],
        },
        "sources": [{"id": "1", "title": "Source One", "url": "https://example.com/1"}],
    }

    with pytest.raises(RuntimeError, match="reformatter_context_too_large"):
        asyncio.run(reformatter_module.run_reformatter(clean_evidence))
    assert called is False


def test_reformatter_polish_omits_full_evidence_by_default(monkeypatch):
    monkeypatch.setenv("REPORT_REFORMATTER_MIN_BODY_CHARS", "1000")
    monkeypatch.setenv("REPORT_REFORMATTER_AUTO_EXPAND_ANALYSIS", "false")
    monkeypatch.setenv("REPORT_REFORMATTER_QUALITY_PASSES", "1")
    monkeypatch.delenv("REPORT_REFORMATTER_POLISH_INCLUDE_FULL_EVIDENCE", raising=False)
    secret_evidence = "UNIQUE_EVIDENCE_SHOULD_NOT_BE_REPEATED_IN_POLISH"
    calls = []

    async def fake_generate(**kwargs):
        calls.append(kwargs["user_content"])
        if len(calls) == 1:
            return "# Report\n\n## Body\nShort body [1]."
        return "# Report\n\n## Body\n" + ("Expanded analysis with the existing citation [1]. " * 80)

    monkeypatch.setattr(reformatter_module, "_generate_reformatter_text", fake_generate)
    clean_evidence = {
        "topic": "light polish",
        "dimensions": {
            "market": [
                {
                    "text": secret_evidence,
                    "source": "1",
                }
            ],
        },
        "sources": [{"id": "1", "title": "Source One", "url": "https://example.com/1"}],
    }

    asyncio.run(reformatter_module.run_reformatter(clean_evidence, quality_passes=1, max_tokens=8000))

    assert len(calls) == 2
    assert secret_evidence in calls[0]
    assert secret_evidence not in calls[1]
    assert "evidence_items" not in calls[1]
