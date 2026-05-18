from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence


AGENT_NAME = "evidence_synthesizer"
AGENT_DESCRIPTION = "Evidence Synthesizer Agent. Builds lightweight evidence graphs before claim building."


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _compact(value: Any, max_chars: int = 220) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "..."


def _dedupe(values: Iterable[Any], *, limit: int = 8) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        text = _compact(value, 140)
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


def _cluster_key(item: Dict[str, Any]) -> str:
    metric = _compact(item.get("metric"), 80)
    if metric:
        return metric
    fact = _compact(item.get("fact"), 80)
    return fact[:24] or "关键事实"


def run_evidence_synthesizer(
    *,
    chapter_evidence_packages: Optional[Sequence[Dict[str, Any]]] = None,
    llm_client: Any = None,
) -> Dict[str, Any]:
    del llm_client
    graph: Dict[str, Any] = {"agent": AGENT_NAME, "chapters": [], "conflicts": []}
    for package in list(chapter_evidence_packages or []):
        if not isinstance(package, dict):
            continue
        buckets: Dict[str, List[Dict[str, Any]]] = {}
        for item in _as_list(package.get("core_evidence")):
            if isinstance(item, dict):
                buckets.setdefault(_cluster_key(item), []).append(item)
        clusters: List[Dict[str, Any]] = []
        for index, (topic, items) in enumerate(buckets.items(), start=1):
            values = _dedupe([item.get("value") for item in items], limit=8)
            refs = _dedupe([item.get("ref") or item.get("evidence_id") for item in items], limit=8)
            conflict = len([value for value in values if value]) > 1
            cluster = {
                "cluster_id": f"{package.get('chapter_id')}_CL_{index:02d}",
                "chapter_id": package.get("chapter_id"),
                "topic": topic,
                "values": values,
                "evidence_refs": refs,
                "support_strength": "high" if len(items) >= 3 else "medium" if len(items) >= 2 else "low",
                "conflict_type": "口径差异" if conflict else "none",
                "use": "support" if refs else "clue",
            }
            clusters.append(cluster)
            if conflict:
                graph["conflicts"].append(cluster)
        graph["chapters"].append(
            {
                "chapter_id": package.get("chapter_id"),
                "chapter_title": package.get("chapter_title"),
                "clusters": clusters,
            }
        )
    return graph
