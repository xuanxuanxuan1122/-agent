from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence


QUERY_CONTRACT_VERSION = "0.1.0"

ROLE_LANE_PRIORITY = {
    "metric": ["official_data", "market_research"],
    "source_check": ["official_data", "filing_company", "market_research"],
    "filing": ["filing_company", "official_data"],
    "counter": ["news_event", "market_research", "customer_case"],
    "case": ["customer_case", "filing_company", "news_event"],
    "support": ["market_research", "official_data"],
}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _compact(value: Any, max_chars: int = 160) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "..."


def _dedupe(values: Sequence[Any], *, limit: int = 20) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        text = _compact(value, 120)
        if not text:
            continue
        key = re.sub(r"\s+", "", text.lower())
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def _split_terms(value: Any) -> List[str]:
    if isinstance(value, (list, tuple, set)):
        terms: List[str] = []
        for item in value:
            terms.extend(_split_terms(item))
        return terms
    return [
        item.strip(" ,;:!?，。；：！？、()（）[]【】{}《》\"'")
        for item in re.split(r"[\s,;:!?，。；：！？、/\\|()\[\]{}（）【】《》\"']+", str(value or ""))
        if item.strip()
    ]


@dataclass(frozen=True)
class QueryBuilder:
    max_terms: int = 18
    max_chars: int = 160

    def compose(self, parts: Sequence[Any]) -> str:
        terms = _dedupe([term for part in parts for term in _split_terms(part)], limit=self.max_terms)
        query = ""
        for term in terms:
            candidate = f"{query} {term}".strip()
            if len(candidate) > self.max_chars:
                continue
            query = candidate
        return query

    def lane_targets(self, task: Dict[str, Any], *, configured_lanes: Sequence[str] | None = None) -> List[str]:
        explicit = _dedupe(_as_list(task.get("lane_targets") or task.get("lanes")), limit=3)
        valid = set(configured_lanes or [])
        explicit = [lane for lane in explicit if not valid or lane in valid]
        if explicit:
            return explicit[:3]
        proof_role = str(task.get("proof_role") or task.get("evidence_type") or "support").strip().lower()
        lanes = list(ROLE_LANE_PRIORITY.get(proof_role, ROLE_LANE_PRIORITY["support"]))
        if task.get("counter_evidence") and "news_event" not in lanes:
            lanes.insert(0, "news_event")
        return [lane for lane in lanes if not valid or lane in valid][:3] or ["market_research"]

    def package(
        self,
        task: Dict[str, Any],
        *,
        query: str = "",
        lane_type: str = "",
        lane_focus: str = "",
    ) -> Dict[str, Any]:
        query_text = query or self.compose(
            [
                task.get("query"),
                task.get("topic_seed_terms"),
                task.get("must_have_terms"),
                task.get("evidence_goal"),
                task.get("hypothesis_statement"),
            ]
        )
        return {
            "query_contract_version": QUERY_CONTRACT_VERSION,
            "query": query_text,
            "must_have_terms": _dedupe(_as_list(task.get("must_have_terms")), limit=12),
            "forbidden_terms": _dedupe(_as_list(task.get("forbidden_terms")), limit=12),
            "proof_role": str(task.get("proof_role") or task.get("evidence_type") or "support").strip().lower(),
            "evidence_type": str(task.get("evidence_type") or task.get("proof_role") or "").strip().lower(),
            "topic_seed_terms": _dedupe(_as_list(task.get("topic_seed_terms")), limit=12),
            "lane_type": lane_type,
            "lane_focus": lane_focus,
            "source_priority": _dedupe(_as_list(task.get("source_priority")), limit=8),
        }


def build_query_package(
    task: Dict[str, Any],
    *,
    query: str = "",
    lane_type: str = "",
    lane_focus: str = "",
) -> Dict[str, Any]:
    return QueryBuilder().package(task, query=query, lane_type=lane_type, lane_focus=lane_focus)
