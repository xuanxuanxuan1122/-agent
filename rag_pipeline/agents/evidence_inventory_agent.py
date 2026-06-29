from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _text(value: Any) -> str:
    return str(value or "").strip()


def _compact(value: Any, limit: int = 360) -> str:
    text = " ".join(_text(value).split())
    if len(text) <= limit:
        return text
    return text[: max(20, limit - 3)].rstrip() + "..."


def _curated_items(payload: Dict[str, Any] | Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        for key in ("curated_evidence", "items", "notes"):
            items = [item for item in _as_list(payload.get(key)) if isinstance(item, dict)]
            if items:
                return items
        return []
    return [item for item in _as_list(payload) if isinstance(item, dict)]


def _cluster_key(item: Dict[str, Any]) -> str:
    explicit = _text(item.get("cluster_key"))
    if explicit:
        return explicit
    fact_type = _text(item.get("fact_type")).lower()
    for prefix in ("market", "policy", "risk", "case", "technology", "competition"):
        if fact_type.startswith(prefix):
            return prefix
    usable_for = {str(value or "").lower() for value in _as_list(item.get("usable_for"))}
    if {"market_size", "commercialization"} & usable_for:
        return "market"
    if "policy" in usable_for:
        return "policy"
    if {"risk", "counter", "boundary"} & usable_for:
        return "risk"
    if {"case", "demand"} & usable_for:
        return "case"
    if {"technology", "supply_chain"} & usable_for:
        return "technology"
    return "context"


def _strength_rank(value: Any) -> int:
    return {"strong": 4, "moderate": 3, "directional": 2, "weak": 1}.get(_text(value).lower(), 1)


def _dominant_strength(items: Iterable[Dict[str, Any]]) -> str:
    values = [_text(item.get("claim_strength_hint")).lower() or "weak" for item in items]
    if not values:
        return "weak"
    return max(values, key=_strength_rank)


def _best_source_level(items: Iterable[Dict[str, Any]]) -> str:
    levels = [_text(item.get("source_level")).upper() or "UNKNOWN" for item in items]
    order = {"A": 5, "B": 4, "C": 3, "D": 2, "UNKNOWN": 1}
    return max(levels or ["UNKNOWN"], key=lambda value: order.get(value, 1))


def _analysis_brief(cluster_key: str, items: Sequence[Dict[str, Any]]) -> str:
    facts = [_compact(item.get("clean_fact"), 140) for item in items if _text(item.get("clean_fact"))]
    if not facts:
        return ""
    lead = facts[0]
    if len(facts) == 1:
        return lead
    return f"{lead} Additional signals: {'; '.join(facts[1:3])}"


def build_evidence_inventory(
    curated_payload: Dict[str, Any] | Sequence[Dict[str, Any]],
    *,
    query: str = "",
    max_clusters: int = 24,
    max_ids_per_cluster: int = 80,
) -> Dict[str, Any]:
    """Build a compact, analysis-facing inventory from curated evidence notes."""

    items = _curated_items(curated_payload)
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for item in items:
        evidence_id = _text(item.get("evidence_id"))
        clean_fact = _text(item.get("clean_fact"))
        if not evidence_id or not clean_fact:
            continue
        grouped.setdefault(_cluster_key(item), []).append(item)

    inventories: List[Dict[str, Any]] = []
    for cluster_key, cluster_items in sorted(grouped.items(), key=lambda pair: len(pair[1]), reverse=True)[:max_clusters]:
        fact_type_counts = Counter(_text(item.get("fact_type")) or "context_signal" for item in cluster_items)
        usable_counts = Counter(
            str(value or "").strip()
            for item in cluster_items
            for value in _as_list(item.get("usable_for"))
            if str(value or "").strip()
        )
        requirement_ids = sorted({_text(item.get("requirement_id")) for item in cluster_items if _text(item.get("requirement_id"))})
        chapter_ids = sorted({_text(item.get("chapter_id")) for item in cluster_items if _text(item.get("chapter_id"))})
        usable_evidence_ids = [_text(item.get("evidence_id")) for item in cluster_items if _text(item.get("evidence_id"))][
            :max_ids_per_cluster
        ]
        limitations = []
        for item in cluster_items:
            for limitation in _as_list(item.get("limitations")):
                text = _compact(limitation, 120)
                if text and text not in limitations:
                    limitations.append(text)
                if len(limitations) >= 6:
                    break
            if len(limitations) >= 6:
                break
        inventory = {
            "schema_version": "evidence_inventory_item_v1",
            "inventory_id": f"INV-{cluster_key}",
            "cluster_key": cluster_key,
            "requirement_ids": requirement_ids,
            "requirement_id": requirement_ids[0] if requirement_ids else "",
            "chapter_ids": chapter_ids,
            "chapter_id": chapter_ids[0] if chapter_ids else "",
            "curated_evidence_count": len(cluster_items),
            "usable_evidence_ids": usable_evidence_ids,
            "fact_type_counts": dict(fact_type_counts),
            "usable_for_counts": dict(usable_counts),
            "strongest_available_level": _best_source_level(cluster_items),
            "dominant_strength": _dominant_strength(cluster_items),
            "analysis_brief": _analysis_brief(cluster_key, cluster_items),
            "limitations": limitations,
            "suggested_analysis_direction": list(usable_counts.keys())[:8] or list(fact_type_counts.keys())[:8],
        }
        inventories.append(inventory)

    return {
        "schema_version": "evidence_inventory_v1",
        "status": "ready" if inventories else "insufficient",
        "query": query,
        "input_curated_evidence_count": len(items),
        "inventory_count": len(inventories),
        "inventories": inventories,
        "inventories_by_cluster": {item["cluster_key"]: item for item in inventories},
    }
