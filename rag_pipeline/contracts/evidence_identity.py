from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Sequence, Set


AMBIGUOUS_ALIAS = "__AMBIGUOUS__"


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return []


def _text(value: Any) -> str:
    return str(value or "").strip()


def _dedupe(values: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(item for item in values if item))


def normalize_evidence_ref_key(value: Any) -> str:
    text = _text(value)
    if not text:
        return ""
    text = text.strip("[](){} \t\r\n")
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.lower()


def canonical_evidence_id(item: Dict[str, Any]) -> str:
    return _text(
        item.get("canonical_evidence_id")
        or item.get("evidence_id")
        or item.get("fact_id")
        or item.get("id")
        or item.get("ref")
    )


def evidence_aliases(item: Dict[str, Any]) -> List[str]:
    aliases: List[str] = []
    for key in (
        "aliases",
        "alias_ids",
        "legacy_ids",
        "legacy_evidence_ids",
        "evidence_refs",
        "used_evidence_ids",
        "supporting_evidence_refs",
        "supporting_fact_refs",
        "used_fact_refs",
    ):
        aliases.extend(_text(value) for value in _as_list(item.get(key)))
    for key in ("evidence_id", "fact_id", "id", "ref", "canonical_evidence_id"):
        value = _text(item.get(key))
        if value:
            aliases.append(value)
    return _dedupe(aliases)


def _put_alias(alias_map: Dict[str, str], alias: str, canonical: str) -> None:
    key = normalize_evidence_ref_key(alias)
    if not key:
        return
    current = alias_map.get(key)
    if current and current != canonical:
        alias_map[key] = AMBIGUOUS_ALIAS
        return
    if current == AMBIGUOUS_ALIAS:
        return
    alias_map[key] = canonical


def build_evidence_alias_map(fact_cards: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    alias_map: Dict[str, str] = {}
    for raw in list(fact_cards or []):
        item = _as_dict(raw)
        canonical = canonical_evidence_id(item)
        if not canonical:
            continue
        _put_alias(alias_map, canonical, canonical)
        for alias in evidence_aliases(item):
            _put_alias(alias_map, alias, canonical)
    return alias_map


def canonicalize_evidence_id(raw_id: Any, alias_map: Dict[str, str]) -> str:
    key = normalize_evidence_ref_key(raw_id)
    if not key:
        return ""
    value = _text(_as_dict(alias_map).get(key))
    if value == AMBIGUOUS_ALIAS:
        return ""
    return value


def ambiguous_evidence_refs(refs: Sequence[Any], alias_map: Dict[str, str]) -> Set[str]:
    result: Set[str] = set()
    for ref in refs:
        key = normalize_evidence_ref_key(ref)
        if key and _as_dict(alias_map).get(key) == AMBIGUOUS_ALIAS:
            result.add(_text(ref))
    return result


def resolve_evidence_refs(refs: Sequence[Any], alias_map: Dict[str, str]) -> Dict[str, Any]:
    resolved: List[str] = []
    unresolved: List[str] = []
    ambiguous: List[str] = []
    alias_resolved: List[Dict[str, str]] = []
    seen_resolved: Set[str] = set()
    seen_raw: Set[str] = set()

    for raw in list(refs or []):
        raw_text = _text(raw)
        if not raw_text or raw_text in seen_raw:
            continue
        seen_raw.add(raw_text)
        key = normalize_evidence_ref_key(raw_text)
        target = _as_dict(alias_map).get(key)
        if target == AMBIGUOUS_ALIAS:
            ambiguous.append(raw_text)
            continue
        if not target:
            unresolved.append(raw_text)
            continue
        if target not in seen_resolved:
            resolved.append(target)
            seen_resolved.add(target)
        if normalize_evidence_ref_key(target) != key:
            alias_resolved.append({"raw_ref": raw_text, "canonical_ref": target})

    return {
        "resolved_fact_ids": resolved,
        "unresolved_refs": unresolved,
        "ambiguous_refs": ambiguous,
        "alias_resolved_refs": alias_resolved,
        "total_refs": len(seen_raw),
        "resolved_ref_count": len(resolved),
        "unresolved_ref_count": len(unresolved),
        "ambiguous_ref_count": len(ambiguous),
        "alias_resolved_ref_count": len(alias_resolved),
    }
