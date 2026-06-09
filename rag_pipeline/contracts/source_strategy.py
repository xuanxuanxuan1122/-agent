from __future__ import annotations

from typing import Any, Dict, Iterable, List


ROLE_SOURCE_STRATEGIES: Dict[str, Dict[str, List[str]]] = {
    "metric": {
        "source_priority": ["official_data", "market_research", "annual_report", "survey", "pdf"],
        "query_enhancers": [
            "report",
            "研报",
            "survey",
            "调研",
            "pdf",
            "PDF",
            "annual report",
            "年报",
            "statistics",
            "统计",
            "methodology",
            "测算口径",
            "公报",
            "白皮书",
            "市场规模",
        ],
    },
    "source_check": {
        "source_priority": ["official_data", "filing_company", "exchange_announcement", "investor_relations"],
        "query_enhancers": ["official", "官方", "announcement", "公告", "annual report", "年报", "prospectus", "招股书", "exchange", "交易所"],
    },
    "filing": {
        "source_priority": ["filing_company", "official_data", "exchange_announcement", "investor_relations"],
        "query_enhancers": [
            "annual report",
            "年报",
            "announcement",
            "公告",
            "prospectus",
            "招股书",
            "exchange",
            "交易所",
            "investor relations",
            "投资者关系",
            "财报",
        ],
    },
    "counter": {
        "source_priority": ["counter_evidence", "news_event", "market_research", "regulatory"],
        "query_enhancers": [
            "failure",
            "cost",
            "ROI unclear",
            "失败案例",
            "成本过高",
            "ROI不明",
            "security",
            "安全",
            "compliance",
            "合规",
            "cancellation",
            "客户流失",
            "delay",
            "延期",
        ],
    },
    "case": {
        "source_priority": ["customer_case", "company_disclosure", "procurement", "filing_company"],
        "query_enhancers": ["customer case", "客户案例", "deployment", "落地案例", "procurement", "采购", "tender", "中标", "announcement", "合作公告", "部署"],
    },
    "customer_case": {
        "source_priority": ["customer_case", "company_disclosure", "procurement", "filing_company"],
        "query_enhancers": ["customer case", "客户案例", "deployment", "落地案例", "procurement", "采购", "tender", "中标", "announcement", "合作公告", "部署"],
    },
    "technology": {
        "source_priority": ["technical_standard", "patent", "product_doc", "academic"],
        "query_enhancers": ["standard", "标准", "patent", "专利", "technical documentation", "技术文档", "benchmark", "基准测试", "paper", "论文"],
    },
    "technology_product": {
        "source_priority": ["technical_standard", "patent", "product_doc", "academic"],
        "query_enhancers": ["standard", "标准", "patent", "专利", "technical documentation", "技术文档", "benchmark", "基准测试", "paper", "论文"],
    },
    "support": {
        "source_priority": ["market_research", "official_data", "news_event"],
        "query_enhancers": ["report", "报告", "official", "官方", "research", "研究", "source", "来源"],
    },
}


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return []


def _dedupe(values: Iterable[Any], *, limit: int = 12) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def source_strategy_for_role(proof_role: Any, *, overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    role = str(proof_role or "support").strip().lower() or "support"
    base = ROLE_SOURCE_STRATEGIES.get(role, ROLE_SOURCE_STRATEGIES["support"])
    overrides = dict(overrides or {})
    source_priority = _dedupe([*_as_list(overrides.get("source_priority")), *base["source_priority"]], limit=16)
    query_enhancers = _dedupe([*_as_list(overrides.get("query_enhancers")), *base["query_enhancers"]], limit=24)
    return {
        "strategy_version": "source_strategy_v2_bilingual",
        "proof_role": role,
        "source_priority": source_priority,
        "query_enhancers": query_enhancers,
    }
