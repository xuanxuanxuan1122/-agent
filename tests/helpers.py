from __future__ import annotations

from typing import Any, Dict, List

from rag_pipeline.agents.writer_agent_clean import INDUSTRY_DIMENSIONS


DIMENSION_PAYLOADS = {
    "市场规模与增速": ("市场规模", "100亿元", "政府统计公报"),
    "竞争格局": ("市场份额", "35%", "上市公司公告"),
    "政策与监管环境": ("政策信号", "补贴目录", "农业农村部通知"),
    "技术路线与产业链": ("技术指标", "95%", "企业白皮书"),
    "资本动态": ("融资金额", "5亿元", "交易所公告"),
}


def sample_items() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for dimension_index, dimension in enumerate(INDUSTRY_DIMENSIONS, start=1):
        metric, value, title = DIMENSION_PAYLOADS[dimension]
        for item_index, source_level in enumerate(["A", "B"], start=1):
            evidence_id = f"{dimension_index}-{item_index}"
            items.append(
                {
                    "evidence_id": evidence_id,
                    "dimension": dimension,
                    "data_point": f"{dimension}核心事实{item_index}，{metric}为{value}，可用于企业决策。",
                    "metric": metric,
                    "value": value,
                    "source_level": source_level,
                    "evidence_role": "core" if item_index == 1 else "supporting",
                    "semantic_status": "ok",
                    "confidence": 0.82,
                    "source": {
                        "title": f"{title}{item_index}",
                        "url": f"https://example.com/{dimension_index}/{item_index}",
                        "date": "2026-05-08",
                        "credibility": source_level,
                    },
                }
            )
    return items


def sample_evidence_package() -> Dict[str, Any]:
    return {"analysis_ready_evidence": sample_items()}


def sample_structured_analysis() -> Dict[str, Any]:
    items = sample_items()
    return {
        "evidence_analyses": items,
        "chapter_thesis": {
            dimension: f"{dimension}的主线是用A/B级证据建立可验证判断，但仍需保留统计口径和时间窗口。"
            for dimension in INDUSTRY_DIMENSIONS
        },
    }

