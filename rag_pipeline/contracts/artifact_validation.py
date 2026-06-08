from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence

from rag_pipeline.cache.artifact_models import UNUSABLE_FACT_STATUSES, USABLE_FACT_STATUSES, as_dict


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


WRITER_FORBIDDEN_KEYS = {
    "diagnostic_gap",
    "raw_html",
    "raw_page",
    "raw_pages",
    "retry_plan",
    "score_gaps",
    "search_snippet",
    "search_snippets",
    "section_draft",
}

STRENGTH_RANK = {
    "none": 0,
    "weak": 1,
    "limited": 1,
    "limited_evidence": 1,
    "directional": 2,
    "moderate": 3,
    "medium": 3,
    "strong": 4,
    "high": 4,
    "definitive": 5,
}

NUMERIC_OR_DATE_RE = re.compile(r"\b(?:19|20)\d{2}\b|\b\d+(?:\.\d+)?%?\b")
COMPANY_NAME_RE = re.compile(
    r"\b[A-Z][A-Za-z0-9&.-]*(?:\s+[A-Z][A-Za-z0-9&.-]*){0,4}\s+"
    r"(?:Inc|Corp|Corporation|Company|Ltd|Limited|LLC|PLC|Group|Holdings|Technologies|Technology|Systems|Bank)\b"
)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return []


def _valid_result(errors: Iterable[str], warnings: Iterable[str] = ()) -> ValidationResult:
    error_list = list(dict.fromkeys(errors))
    warning_list = list(dict.fromkeys(warnings))
    return ValidationResult(ok=not error_list, errors=error_list, warnings=warning_list)


def _fact_id(fact: Dict[str, Any]) -> str:
    return _clean(fact.get("fact_id") or fact.get("id") or fact.get("ref"))


def _source_id(source: Dict[str, Any]) -> str:
    return _clean(
        source.get("run_source_id")
        or source.get("source_id")
        or source.get("ref")
        or source.get("id")
        or source.get("canonical_source_id")
    )


def _claim_id(claim: Dict[str, Any]) -> str:
    return _clean(claim.get("claim_id") or claim.get("id") or claim.get("ref"))


def _fact_refs(value: Dict[str, Any]) -> List[str]:
    payload = as_dict(value.get("payload"))
    refs = (
        _as_list(value.get("fact_ids"))
        or _as_list(value.get("used_fact_refs"))
        or _as_list(value.get("evidence_refs"))
        or _as_list(payload.get("fact_ids"))
        or _as_list(payload.get("used_fact_refs"))
        or _as_list(payload.get("evidence_refs"))
    )
    return [_clean(item) for item in refs if _clean(item)]


def _claim_refs(section: Dict[str, Any]) -> List[str]:
    payload = as_dict(section.get("payload"))
    refs = _as_list(section.get("claim_ids")) or _as_list(payload.get("claim_ids"))
    if not refs and (section.get("claim_id") or payload.get("claim_id")):
        refs = [section.get("claim_id") or payload.get("claim_id")]
    return [_clean(item) for item in refs if _clean(item)]


def _strength_exceeds(value: str, ceiling: str) -> bool:
    strength = _clean(value).lower()
    limit = _clean(ceiling).lower()
    if not strength or not limit:
        return False
    return STRENGTH_RANK.get(strength, 0) > STRENGTH_RANK.get(limit, 0)


def validate_fact_card_for_context(fact_card: Dict[str, Any], source: Dict[str, Any]) -> ValidationResult:
    errors: List[str] = []
    status = _clean(fact_card.get("status")).lower()
    if status not in USABLE_FACT_STATUSES:
        errors.append("fact_card_unusable_status")
    if not _clean(fact_card.get("fact")):
        errors.append("fact_card_missing_fact_text")
    if not source:
        errors.append("source_missing")
    else:
        fact_source_id = _clean(fact_card.get("source_id"))
        source_ids = {
            _source_id(source),
            _clean(source.get("run_source_id")),
            _clean(source.get("canonical_source_id")),
        }
        if fact_source_id and fact_source_id not in source_ids:
            errors.append("fact_source_mismatch")
    return _valid_result(errors)


def validate_claim_unit_lineage(
    claim_unit: Dict[str, Any],
    fact_cards: Sequence[Dict[str, Any]],
) -> ValidationResult:
    errors: List[str] = []
    facts_by_id = {_fact_id(item): item for item in fact_cards if _fact_id(item)}
    refs = _fact_refs(claim_unit)
    payload = as_dict(claim_unit.get("payload"))
    requirement_ids = (
        _as_list(claim_unit.get("requirement_ids"))
        or _as_list(payload.get("requirement_ids"))
        or _as_list(as_dict(claim_unit.get("lineage")).get("requirement_ids"))
    )
    requirement_ids = [_clean(item) for item in requirement_ids if _clean(item)]
    if not refs:
        errors.append("claim_unit_missing_fact_refs")
    for ref in refs:
        fact = facts_by_id.get(ref)
        if not fact:
            errors.append("claim_unit_references_missing_fact")
            continue
        if _clean(fact.get("status")).lower() not in USABLE_FACT_STATUSES:
            errors.append("claim_unit_references_unusable_fact")
    if not requirement_ids:
        errors.append("claim_unit_missing_requirement_ids")
    if _strength_exceeds(
        claim_unit.get("claim_strength") or claim_unit.get("strength") or "",
        claim_unit.get("claim_strength_ceiling") or "",
    ):
        errors.append("claim_strength_exceeds_ceiling")
    return _valid_result(errors)


def validate_section_lineage(
    section: Dict[str, Any],
    claim_units: Sequence[Dict[str, Any]],
    fact_cards: Sequence[Dict[str, Any]],
    sources: Sequence[Dict[str, Any]],
) -> ValidationResult:
    errors: List[str] = []
    claims_by_id = {_claim_id(item): item for item in claim_units if _claim_id(item)}
    facts_by_id = {_fact_id(item): item for item in fact_cards if _fact_id(item)}
    source_ids = {
        source_id
        for source in sources
        for source_id in (
            _source_id(source),
            _clean(source.get("run_source_id")),
            _clean(source.get("canonical_source_id")),
        )
        if source_id
    }

    fact_refs = _fact_refs(section)
    if not fact_refs:
        errors.append("section_missing_fact_refs")
    for ref in fact_refs:
        fact = facts_by_id.get(ref)
        if not fact:
            errors.append("section_references_missing_fact")
            continue
        status = _clean(fact.get("status")).lower()
        if status in UNUSABLE_FACT_STATUSES or status not in USABLE_FACT_STATUSES:
            errors.append("section_references_unusable_fact")
        fact_source_id = _clean(fact.get("source_id"))
        if fact_source_id and source_ids and fact_source_id not in source_ids:
            errors.append("section_fact_source_missing")

    for claim_ref in _claim_refs(section):
        if claim_ref not in claims_by_id:
            errors.append("section_references_missing_claim")
    return _valid_result(errors)


def _walk_forbidden_keys(value: Any) -> List[str]:
    found: List[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key) in WRITER_FORBIDDEN_KEYS:
                found.append(str(key))
            found.extend(_walk_forbidden_keys(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_walk_forbidden_keys(child))
    return found


def _claim_texts(view: Dict[str, Any]) -> List[str]:
    texts: List[str] = []
    for claim in _as_list(view.get("claim_units")):
        if isinstance(claim, dict):
            text = _clean(claim.get("claim") or claim.get("text") or as_dict(claim.get("payload")).get("claim"))
            if text:
                texts.append(text)
    return texts


def _allowed_fact_text(view: Dict[str, Any]) -> str:
    parts: List[str] = []
    for fact in _as_list(view.get("usable_fact_cards")):
        if not isinstance(fact, dict):
            continue
        parts.append(_clean(fact.get("fact")))
        parts.append(_clean(fact.get("metric")))
        parts.append(_clean(fact.get("value")))
        parts.append(_clean(fact.get("period")))
        payload = as_dict(fact.get("payload"))
        parts.append(_clean(payload.get("fact")))
        parts.append(_clean(payload.get("value")))
        parts.append(_clean(payload.get("period")))
    return "\n".join(part for part in parts if part)


def _unbound_numeric_or_dates(view: Dict[str, Any]) -> List[str]:
    allowed = _allowed_fact_text(view)
    missing: List[str] = []
    for text in _claim_texts(view):
        for token in NUMERIC_OR_DATE_RE.findall(text):
            if token not in allowed:
                missing.append(token)
    return missing


def _unbound_company_names(view: Dict[str, Any]) -> List[str]:
    allowed = _allowed_fact_text(view)
    missing: List[str] = []
    for text in _claim_texts(view):
        for token in COMPANY_NAME_RE.findall(text):
            if token not in allowed:
                missing.append(token)
    return missing


def validate_context_view(view: Dict[str, Any], task_type: str) -> ValidationResult:
    errors: List[str] = []
    warnings: List[str] = []
    task = _clean(task_type or view.get("task")).lower()
    if task in {"write_section", "writer", "section_writer"}:
        if _walk_forbidden_keys(view):
            errors.append("writer_context_contains_forbidden_field")
        if _unbound_numeric_or_dates(view):
            errors.append("writer_context_unbound_numeric_or_date")
        if _unbound_company_names(view):
            errors.append("writer_context_unbound_company_name")
        for fact in _as_list(view.get("usable_fact_cards")):
            if isinstance(fact, dict) and _clean(fact.get("status")).lower() not in USABLE_FACT_STATUSES:
                errors.append("writer_context_contains_unusable_fact")
    if _clean(view.get("status")).lower() == "insufficient" and _clean(view.get("instruction")) != "do_not_infer":
        warnings.append("insufficient_context_should_do_not_infer")
    return _valid_result(errors, warnings)
