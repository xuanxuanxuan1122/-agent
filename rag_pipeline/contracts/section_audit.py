from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping

from .claim_roles import classify_claim_unit_roles


SECTION_AUDIT_VERSION = "section_audit_v1"


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value in (None, "", {}, ()):
        return []
    return [value]


def _text(value: Any) -> str:
    return str(value or "").strip()


def _dedupe(values: Iterable[Any]) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        text = _text(value)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _stable_slug(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", _text(value))
    return text.strip("-") or "unknown"


def _mapping_by_id(items: Any, id_keys: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    if isinstance(items, Mapping):
        return {
            _text(key): _as_dict(value)
            for key, value in items.items()
            if _text(key) and isinstance(value, dict)
        }
    result: Dict[str, Dict[str, Any]] = {}
    for item in _as_list(items):
        payload = _as_dict(item)
        item_id = next((_text(payload.get(key)) for key in id_keys if _text(payload.get(key))), "")
        if item_id:
            result[item_id] = payload
    return result


def _claim_refs(claim: Dict[str, Any]) -> List[str]:
    return _dedupe(
        [
            *_as_list(claim.get("fact_ids")),
            *_as_list(claim.get("used_fact_refs")),
            *_as_list(claim.get("evidence_refs")),
            *_as_list(claim.get("supporting_fact_refs")),
            *_as_list(claim.get("supporting_evidence_refs")),
            *_as_list(claim.get("used_evidence_ids")),
        ]
    )


def _section_claim_ids(section: Dict[str, Any]) -> List[str]:
    return _dedupe([section.get("claim_id"), *_as_list(section.get("claim_ids"))])


def _section_fact_refs(section: Dict[str, Any]) -> List[str]:
    return _dedupe(
        [
            *_as_list(section.get("used_fact_refs")),
            *_as_list(section.get("fact_ids")),
            *_as_list(section.get("evidence_refs")),
            *_as_list(section.get("supporting_fact_refs")),
        ]
    )


def _requirement_ids(section: Dict[str, Any], claims: List[Dict[str, Any]], facts: List[Dict[str, Any]]) -> List[str]:
    return _dedupe(
        [
            section.get("requirement_id"),
            *_as_list(section.get("requirement_ids")),
            *[req for claim in claims for req in _as_list(claim.get("requirement_ids"))],
            *[fact.get("requirement_id") for fact in facts],
        ]
    )


def _source_present(fact: Dict[str, Any]) -> bool:
    source = _as_dict(fact.get("source"))
    return bool(
        _text(fact.get("source_id"))
        or _text(fact.get("source_url"))
        or _text(fact.get("url"))
        or _text(fact.get("citation_ref"))
        or _text(source.get("url"))
        or _text(source.get("id"))
        or _text(source.get("ref"))
    )


def _metric_missing_fields(facts: List[Dict[str, Any]]) -> List[str]:
    metric_facts = [
        fact
        for fact in facts
        if _text(fact.get("metric"))
        or _text(fact.get("value"))
        or "metric" in _text(fact.get("proof_role")).lower()
        or "metric" in _text(_as_dict(fact.get("evidence_card")).get("proof_role")).lower()
    ]
    if not metric_facts:
        return []
    missing: List[str] = []
    for field in ("metric", "value", "unit", "period"):
        if not any(_text(fact.get(field)) for fact in metric_facts):
            missing.append(field)
    if not any(_source_present(fact) for fact in metric_facts):
        missing.append("source")
    return missing


def _has_boundary_text(claims: List[Dict[str, Any]]) -> bool:
    for claim in claims:
        if _text(claim.get("limitation_boundary") or claim.get("counter_boundary") or claim.get("counter_evidence")):
            return True
    return False


def _gap_id(section_id: str, gap_type: str, requirement_id: str) -> str:
    return f"SA-{_stable_slug(section_id)}-{_stable_slug(gap_type)}-{_stable_slug(requirement_id)}"


def _gap_payload(
    *,
    section_id: str,
    chapter_id: str,
    requirement_id: str,
    gap_type: str,
    missing: List[str],
    severity: str,
    proof_role: str,
    required_fields: List[str],
    finding_type: str,
) -> Dict[str, Any]:
    return {
        "gap_id": _gap_id(section_id, gap_type, requirement_id),
        "requirement_id": requirement_id,
        "chapter_id": chapter_id,
        "section_id": section_id,
        "gap_type": gap_type,
        "severity": severity,
        "missing": missing,
        "status": "open",
        "retry_plan": {
            "source_stage": "section_audit",
            "section_audit_version": SECTION_AUDIT_VERSION,
            "finding_type": finding_type,
            "proof_role": proof_role,
            "required_fields": required_fields,
            "allowed_for_writing": False,
        },
    }


def audit_section_claim_roles(section: Dict[str, Any], claim_units: Any, fact_cards: Any = None) -> Dict[str, Any]:
    section_payload = _as_dict(section)
    claim_map = _mapping_by_id(claim_units, ("claim_id", "id"))
    fact_map = _mapping_by_id(fact_cards, ("fact_id", "evidence_id", "ref", "id"))
    claim_ids = _section_claim_ids(section_payload)
    selected_claims = [claim_map[claim_id] for claim_id in claim_ids if claim_id in claim_map]
    if not claim_ids and claim_map:
        selected_claims = list(claim_map.values())

    fact_refs = _dedupe(
        [
            *_section_fact_refs(section_payload),
            *[ref for claim in selected_claims for ref in _claim_refs(claim)],
        ]
    )
    selected_facts = [fact_map[ref] for ref in fact_refs if ref in fact_map]

    normalized_claims: List[Dict[str, Any]] = []
    for claim in selected_claims:
        if _as_list(claim.get("claim_roles")) and _text(claim.get("primary_claim_role")):
            normalized_claims.append(claim)
            continue
        role_result = classify_claim_unit_roles(claim, fact_map)
        normalized_claims.append({**claim, **role_result})

    all_roles = _dedupe(role for claim in normalized_claims for role in _as_list(claim.get("claim_roles")))
    section_id = _text(section_payload.get("section_id") or section_payload.get("id"))
    chapter_id = _text(section_payload.get("chapter_id"))
    req_ids = _requirement_ids(section_payload, normalized_claims, selected_facts)
    requirement_id = req_ids[0] if req_ids else ""
    findings: List[Dict[str, Any]] = []
    score_gaps: List[Dict[str, Any]] = []

    strong_core_claims = [
        claim
        for claim in normalized_claims
        if "core_claim" in _as_list(claim.get("claim_roles"))
        and _text(claim.get("claim_strength") or claim.get("claim_status")).lower()
        in {"strong", "decision_ready", "moderate", "medium"}
    ]
    if strong_core_claims and not ({"boundary_claim", "counter_claim"} & set(all_roles)) and not _has_boundary_text(normalized_claims):
        finding = {
            "type": "section_missing_counter_boundary",
            "section_id": section_id,
            "requirement_id": requirement_id,
            "severity": "medium",
            "claim_ids": [_text(claim.get("claim_id") or claim.get("id")) for claim in strong_core_claims],
        }
        findings.append(finding)
        score_gaps.append(
            _gap_payload(
                section_id=section_id,
                chapter_id=chapter_id,
                requirement_id=requirement_id,
                gap_type="counter_boundary_missing",
                missing=["counter_boundary"],
                severity="medium",
                proof_role="counter",
                required_fields=["source"],
                finding_type=finding["type"],
            )
        )

    metric_missing = _metric_missing_fields(selected_facts)
    if "metric_claim" in all_roles and metric_missing:
        finding = {
            "type": "section_metric_missing_fields",
            "section_id": section_id,
            "requirement_id": requirement_id,
            "severity": "blocking",
            "missing": metric_missing,
        }
        findings.append(finding)
        score_gaps.append(
            _gap_payload(
                section_id=section_id,
                chapter_id=chapter_id,
                requirement_id=requirement_id,
                gap_type="metric_scope_period_unit_incomplete",
                missing=metric_missing,
                severity="blocking",
                proof_role="metric",
                required_fields=["metric", "value", "unit", "period", "source"],
                finding_type=finding["type"],
            )
        )

    for claim in normalized_claims:
        if not _as_list(claim.get("requirement_ids")) and requirement_id:
            findings.append(
                {
                    "type": "section_claim_missing_requirement_lineage",
                    "section_id": section_id,
                    "requirement_id": requirement_id,
                    "severity": "medium",
                    "claim_id": _text(claim.get("claim_id") or claim.get("id")),
                }
            )

    status = "needs_repair" if score_gaps else ("warning" if findings else "pass")
    return {
        "section_audit_version": SECTION_AUDIT_VERSION,
        "status": status,
        "section_id": section_id,
        "chapter_id": chapter_id,
        "requirement_ids": req_ids,
        "claim_roles": all_roles,
        "findings": findings,
        "missing_claim_roles": [],
        "score_gaps": score_gaps,
    }
