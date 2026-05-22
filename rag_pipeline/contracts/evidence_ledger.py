from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Sequence


LEDGER_VERSION = "0.1.0"


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _compact(value: Any, max_chars: int = 360) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "..."


def _dedupe(values: Iterable[Any], *, limit: int = 50) -> List[str]:
    result: List[str] = []
    seen = set()
    iterable = [values] if isinstance(values, str) else (values or [])
    for value in iterable:
        text = _compact(value, 180)
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


def _source_id(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    match = re.search(r"\[(\d{1,5})\]", text)
    if match:
        return match.group(1)
    if text.isdigit():
        return text
    return text.strip("[]")


def _source_map(sources: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for source in sources:
        if not isinstance(source, dict):
            continue
        keys = [
            source.get("id"),
            source.get("ref"),
            source.get("source_id"),
            source.get("citation_ref"),
        ]
        for key in keys:
            source_id = _source_id(key)
            if source_id:
                result[source_id] = source
    return result


def _level(value: Any) -> str:
    text = str(value or "").strip().upper()
    return text if text in {"A", "B", "C", "D"} else ""


def _level_score(level: str) -> float:
    return {"A": 0.92, "B": 0.78, "C": 0.56, "D": 0.28}.get(_level(level), 0.46)


def _source_type_bonus(source_type: str) -> float:
    text = str(source_type or "").strip().lower()
    if text in {"official", "policy", "filing", "financial_report", "annual_report"}:
        return 0.07
    if text in {"research", "academic", "association"}:
        return 0.05
    if text in {"news", "media"}:
        return 0.02
    if text in {"self_media", "unknown"}:
        return -0.06
    return 0.0


def _confidence(item: Dict[str, Any], source: Dict[str, Any]) -> float:
    source_type = str(item.get("source_type") or source.get("source_type") or "").strip()
    level = _level(
        item.get("credibility_level")
        or item.get("source_level")
        or source.get("credibility_level")
        or source.get("credibility")
    )
    score = _level_score(level) + _source_type_bonus(source_type)
    text = str(item.get("text") or item.get("fact") or "")
    if re.search(r"\d|%|同比|环比|cagr", text, re.I):
        score += 0.04
    if item.get("period") or item.get("time"):
        score += 0.02
    if item.get("scope"):
        score += 0.02
    if item.get("metric") and item.get("value"):
        score += 0.03
    if not (item.get("source") or item.get("source_id") or item.get("source_ref")):
        score -= 0.12
    return round(max(0.0, min(0.98, score)), 4)


def _fact_signature(text: str) -> str:
    compact = re.sub(r"\[\d{1,5}\]", "", str(text or ""))
    compact = re.sub(r"[^\w\u4e00-\u9fff]+", "", compact.lower())
    compact = re.sub(r"\d+(?:\.\d+)?", "#", compact)
    return compact[:120]


def build_evidence_ledger(clean_evidence: Dict[str, Any]) -> List[Dict[str, Any]]:
    clean_evidence = _as_dict(clean_evidence)
    sources = [_as_dict(item) for item in _as_list(clean_evidence.get("sources")) if isinstance(item, dict)]
    sources_by_id = _source_map(sources)
    entries: List[Dict[str, Any]] = []
    dimensions = _as_dict(clean_evidence.get("dimensions"))
    for dimension, items in dimensions.items():
        for item in _as_list(items):
            if not isinstance(item, dict):
                continue
            source_id = _source_id(item.get("source") or item.get("source_id") or item.get("source_ref") or item.get("ref"))
            source = sources_by_id.get(source_id, {})
            fact_text = _compact(item.get("text") or item.get("fact") or item.get("fact_text"), 900)
            if not fact_text:
                continue
            evidence_id = _compact(item.get("evidence_id") or item.get("id") or f"EV-{len(entries) + 1:04d}", 80)
            citation_id = f"[{source_id}]" if source_id and source_id.isdigit() else source_id
            confidence = _confidence(item, source)
            evidence_card = _as_dict(item.get("evidence_card"))
            entries.append(
                {
                    "ledger_id": f"L{len(entries) + 1:04d}",
                    "ledger_version": LEDGER_VERSION,
                    "evidence_id": evidence_id,
                    "fact_signature": _fact_signature(fact_text),
                    "fact_text": fact_text,
                    "dimension": _compact(dimension, 160),
                    "source_id": source_id,
                    "citation_id": citation_id,
                    "source_title": _compact(source.get("title") or source.get("name") or item.get("source_title"), 220),
                    "source_url": _compact(source.get("url") or item.get("url"), 500),
                    "source_type": _compact(item.get("source_type") or source.get("source_type"), 80),
                    "source_level": _level(
                        item.get("credibility_level")
                        or item.get("source_level")
                        or source.get("credibility_level")
                        or source.get("credibility")
                    ),
                    "source_quality": _compact(item.get("source_quality"), 80),
                    "confidence_score": evidence_card.get("confidence_score") or item.get("confidence_score") or confidence,
                    "allowed_use": _compact(item.get("allowed_use") or evidence_card.get("allowed_use"), 80),
                    "usage_tier": _compact(item.get("usage_tier"), 80),
                    "appendix_only": bool(item.get("appendix_only")),
                    "used_in_body": str(item.get("source_type") or "") == "report_citation" or bool(item.get("used_in_body")),
                    "chain": {
                        "search_task_id": _compact(item.get("search_task_id") or item.get("task_id"), 120),
                        "chapter_id": _compact(item.get("chapter_id"), 120),
                        "claim_id": _compact(item.get("claim_id"), 120),
                        "hypothesis_id": _compact(item.get("hypothesis_id"), 120),
                        "proof_role": _compact(item.get("proof_role") or evidence_card.get("proof_role"), 80),
                        "evidence_type": _compact(item.get("evidence_type"), 80),
                        "source_stage": _compact(item.get("source_stage") or item.get("origin_query_source"), 80),
                        "citation_ref": _compact(item.get("source_ref") or item.get("citation_ref") or item.get("ref"), 80),
                    },
                    "fact_metadata": {
                        "metric": _compact(item.get("metric") or _as_dict(evidence_card.get("metric_definition")).get("metric"), 160),
                        "value": _compact(item.get("value") or _as_dict(evidence_card.get("metric_definition")).get("value"), 160),
                        "unit": _compact(item.get("unit") or _as_dict(evidence_card.get("metric_definition")).get("unit"), 80),
                        "period": _compact(item.get("period") or item.get("time") or evidence_card.get("period"), 120),
                        "scope": _compact(item.get("scope") or evidence_card.get("scope"), 160),
                    },
                }
            )
    return entries


def build_evidence_groups(entries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        signature = str(entry.get("fact_signature") or "").strip()
        if not signature:
            continue
        key = f"{entry.get('dimension') or ''}|{signature}"
        grouped.setdefault(key, []).append(entry)
    groups: List[Dict[str, Any]] = []
    for index, items in enumerate(grouped.values(), start=1):
        source_ids = _dedupe(item.get("source_id") for item in items)
        source_levels = _dedupe(item.get("source_level") for item in items)
        confidences = [float(item.get("confidence_score") or 0.0) for item in items]
        combined = 1.0 - math.prod(max(0.0, 1.0 - score) for score in confidences) if confidences else 0.0
        groups.append(
            {
                "group_id": f"EG{index:04d}",
                "dimension": items[0].get("dimension"),
                "representative_fact": items[0].get("fact_text"),
                "supporting_ledger_ids": [item.get("ledger_id") for item in items],
                "supporting_source_ids": source_ids,
                "source_levels": source_levels,
                "source_count": len(source_ids),
                "multi_source_verified": len(source_ids) >= 2,
                "confidence_score": round(min(0.99, combined), 4),
            }
        )
    groups.sort(key=lambda item: (int(item.get("source_count") or 0), float(item.get("confidence_score") or 0.0)), reverse=True)
    return groups


def summarize_evidence_ledger(entries: Sequence[Dict[str, Any]], groups: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    level_distribution: Dict[str, int] = {}
    dimension_distribution: Dict[str, int] = {}
    source_ids = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        level = str(entry.get("source_level") or "unknown")
        dimension = str(entry.get("dimension") or "unknown")
        level_distribution[level] = level_distribution.get(level, 0) + 1
        dimension_distribution[dimension] = dimension_distribution.get(dimension, 0) + 1
        if entry.get("source_id"):
            source_ids.add(str(entry.get("source_id")))
    multi_source_groups = [item for item in groups if isinstance(item, dict) and item.get("multi_source_verified")]
    return {
        "ledger_version": LEDGER_VERSION,
        "ledger_entry_count": len([item for item in entries if isinstance(item, dict)]),
        "unique_source_count": len(source_ids),
        "used_in_body_count": len([item for item in entries if isinstance(item, dict) and item.get("used_in_body")]),
        "multi_source_group_count": len(multi_source_groups),
        "source_level_distribution": level_distribution,
        "dimension_distribution": dimension_distribution,
    }


def attach_evidence_ledger(clean_evidence: Dict[str, Any]) -> Dict[str, Any]:
    clean = dict(_as_dict(clean_evidence))
    entries = build_evidence_ledger(clean)
    groups = build_evidence_groups(entries)
    metadata = dict(_as_dict(clean.get("metadata")))
    metadata["evidence_ledger_summary"] = summarize_evidence_ledger(entries, groups)
    clean["metadata"] = metadata
    clean["evidence_ledger"] = entries
    clean["evidence_groups"] = groups
    return clean
