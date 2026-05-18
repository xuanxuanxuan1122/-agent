import inspect

import run_full_report
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
