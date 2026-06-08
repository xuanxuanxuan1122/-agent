from __future__ import annotations

from typing import Any, Dict, Iterable, List


ROLE_SOURCE_STRATEGIES: Dict[str, Dict[str, List[str]]] = {
    "metric": {
        "source_priority": ["official_data", "market_research", "annual_report", "survey", "pdf"],
        "query_enhancers": ["report", "survey", "pdf", "annual report", "statistics", "methodology"],
    },
    "source_check": {
        "source_priority": ["official_data", "filing_company", "exchange_announcement", "investor_relations"],
        "query_enhancers": ["official", "announcement", "annual report", "prospectus", "exchange"],
    },
    "filing": {
        "source_priority": ["filing_company", "official_data", "exchange_announcement", "investor_relations"],
        "query_enhancers": ["annual report", "announcement", "prospectus", "exchange", "investor relations"],
    },
    "counter": {
        "source_priority": ["counter_evidence", "news_event", "market_research", "regulatory"],
        "query_enhancers": ["failure", "cost", "ROI unclear", "security", "compliance", "cancellation", "delay"],
    },
    "case": {
        "source_priority": ["customer_case", "company_disclosure", "procurement", "filing_company"],
        "query_enhancers": ["customer case", "deployment", "procurement", "tender", "announcement"],
    },
    "customer_case": {
        "source_priority": ["customer_case", "company_disclosure", "procurement", "filing_company"],
        "query_enhancers": ["customer case", "deployment", "procurement", "tender", "announcement"],
    },
    "technology": {
        "source_priority": ["technical_standard", "patent", "product_doc", "academic"],
        "query_enhancers": ["standard", "patent", "technical documentation", "benchmark", "paper"],
    },
    "technology_product": {
        "source_priority": ["technical_standard", "patent", "product_doc", "academic"],
        "query_enhancers": ["standard", "patent", "technical documentation", "benchmark", "paper"],
    },
    "support": {
        "source_priority": ["market_research", "official_data", "news_event"],
        "query_enhancers": ["report", "official", "research", "source"],
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
    source_priority = _dedupe([*_as_list(overrides.get("source_priority")), *base["source_priority"]])
    query_enhancers = _dedupe([*_as_list(overrides.get("query_enhancers")), *base["query_enhancers"]])
    return {
        "strategy_version": "source_strategy_v1",
        "proof_role": role,
        "source_priority": source_priority,
        "query_enhancers": query_enhancers,
    }
