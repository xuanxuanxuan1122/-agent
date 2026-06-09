from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

from rag_pipeline.contracts.evidence_identity import (
    build_evidence_alias_map,
    canonical_evidence_id,
    resolve_evidence_refs,
)


FACT_REF_FIELDS = [
    "fact_ids",
    "used_fact_refs",
    "evidence_refs",
    "supporting_fact_refs",
    "supporting_evidence_refs",
    "used_evidence_ids",
    "required_evidence_refs",
]

SOURCE_REF_FIELDS = [
    "source_ids",
    "source_id",
    "source_ref",
    "citation_ref",
]

REQUIREMENT_REF_FIELDS = [
    "requirement_ids",
    "evidence_requirement_ids",
    "requirement_id",
    "goal_id",
]


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    if value in (None, ""):
        return []
    return [value]


def _text(value: Any) -> str:
    return str(value or "").strip()


def _dedupe(values: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(item for item in values if item))


def _collect_fields(payload: Dict[str, Any], fields: Sequence[str]) -> Dict[str, List[str]]:
    lineage = _as_dict(payload.get("lineage"))
    nested_payload = _as_dict(payload.get("payload"))
    result: Dict[str, List[str]] = {}
    for field in fields:
        values = [
            *_as_list(payload.get(field)),
            *_as_list(lineage.get(field)),
            *_as_list(nested_payload.get(field)),
        ]
        cleaned = _dedupe(_text(value) for value in values)
        if cleaned:
            result[field] = cleaned
    return result


def _flatten(field_map: Dict[str, List[str]]) -> List[str]:
    values: List[str] = []
    for field_values in field_map.values():
        values.extend(field_values)
    return _dedupe(values)


def _fact_lookup(fact_cards: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for raw in list(fact_cards or []):
        item = _as_dict(raw)
        canonical = canonical_evidence_id(item)
        if canonical:
            lookup[canonical] = item
    return lookup


def _source_ids_from_facts(fact_ids: Sequence[str], fact_cards: Sequence[Dict[str, Any]]) -> List[str]:
    lookup = _fact_lookup(fact_cards)
    return _dedupe(
        _text(_as_dict(lookup.get(fact_id)).get("source_id") or _as_dict(lookup.get(fact_id)).get("run_source_id"))
        for fact_id in fact_ids
    )


def _requirement_ids_from_facts(fact_ids: Sequence[str], fact_cards: Sequence[Dict[str, Any]]) -> List[str]:
    lookup = _fact_lookup(fact_cards)
    values: List[str] = []
    for fact_id in fact_ids:
        fact = _as_dict(lookup.get(fact_id))
        values.extend(_as_list(fact.get("requirement_ids")))
        values.extend(_as_list(fact.get("requirement_id")))
        values.extend(_as_list(fact.get("evidence_requirement_ids")))
    return _dedupe(_text(value) for value in values)


def _canonicalize_refs(values: Sequence[Any], alias_map: Dict[str, str]) -> List[str]:
    result: List[str] = []
    for raw in list(values or []):
        text = _text(raw)
        if not text:
            continue
        result.append(_text(alias_map.get(text) or alias_map.get(text.lower()) or text))
    return _dedupe(result)


def normalize_claim_refs(
    payload: Dict[str, Any],
    *,
    alias_map: Dict[str, str] | None = None,
    fact_cards: Sequence[Dict[str, Any]] | None = None,
    source_alias_map: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    item = dict(_as_dict(payload))
    facts = [fact for fact in list(fact_cards or []) if isinstance(fact, dict)]
    evidence_alias_map = dict(alias_map or build_evidence_alias_map(facts))
    source_map = dict(source_alias_map or {})

    legacy_fact_fields = _collect_fields(item, FACT_REF_FIELDS)
    legacy_source_fields = _collect_fields(item, SOURCE_REF_FIELDS)
    legacy_requirement_fields = _collect_fields(item, REQUIREMENT_REF_FIELDS)

    raw_fact_refs = _flatten(legacy_fact_fields)
    if evidence_alias_map:
        fact_resolution = resolve_evidence_refs(raw_fact_refs, evidence_alias_map)
    else:
        fact_resolution = {
            "resolved_fact_ids": raw_fact_refs,
            "unresolved_refs": [],
            "ambiguous_refs": [],
            "alias_resolved_refs": [],
            "total_refs": len(raw_fact_refs),
            "resolved_ref_count": len(raw_fact_refs),
            "unresolved_ref_count": 0,
            "ambiguous_ref_count": 0,
            "alias_resolved_ref_count": 0,
        }
    fact_ids = _dedupe(_text(value) for value in fact_resolution.get("resolved_fact_ids", []))
    source_ids = _canonicalize_refs(_flatten(legacy_source_fields), source_map)
    source_ids = _dedupe([*source_ids, *_source_ids_from_facts(fact_ids, facts)])
    fact_requirement_ids = _requirement_ids_from_facts(fact_ids, facts)
    explicit_requirement_ids = _dedupe(
        value
        for field in ("requirement_ids", "evidence_requirement_ids", "requirement_id")
        for value in legacy_requirement_fields.get(field, [])
    )
    legacy_requirement_ids = _dedupe(value for value in legacy_requirement_fields.get("goal_id", []))
    requirement_ids = fact_requirement_ids or explicit_requirement_ids or legacy_requirement_ids

    normalized = {
        **item,
        "fact_ids": fact_ids,
        "evidence_refs": fact_ids,
        "supporting_evidence_refs": fact_ids,
        "used_fact_refs": fact_ids,
        "source_ids": source_ids,
        "requirement_ids": requirement_ids,
        "legacy_ref_fields": {
            **{key: value for key, value in legacy_fact_fields.items() if key != "fact_ids"},
            **{key: value for key, value in legacy_source_fields.items() if key != "source_ids"},
            **{key: value for key, value in legacy_requirement_fields.items() if key != "requirement_ids"},
        },
        "ref_resolution": fact_resolution,
        "unresolved_refs": list(fact_resolution.get("unresolved_refs") or []),
        "ambiguous_refs": list(fact_resolution.get("ambiguous_refs") or []),
    }
    return normalized
