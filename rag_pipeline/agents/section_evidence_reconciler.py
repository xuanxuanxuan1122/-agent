from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence


PUBLIC_EVIDENCE_COLLECTIONS = (
    "core_evidence",
    "supporting_evidence",
    "metric_evidence",
    "case_evidence",
    "counter_evidence",
    "directional_evidence",
    "sample_evidence",
)

NON_WRITABLE_ALLOWED_USES = {
    "rejected",
    "not_allowed",
    "diagnostic_only",
    "clue_only",
    "followup_only",
    "not_for_writing",
}

NON_WRITABLE_READINESS = {
    "blocked",
    "followup_only",
    "clue_only",
    "diagnostic_only",
    "rejected",
}


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _norm_id(value: Any) -> str:
    return re.sub(r"[\s_\-./]+", "", str(value or "").strip().lower())


def _dedupe(values: Iterable[Any], *, limit: int = 12) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
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


def _citation_ref(item: Dict[str, Any]) -> str:
    for key in ("source_ref", "citation_ref"):
        value = str(item.get(key) or "").strip()
        if re.fullmatch(r"\[\d{1,3}\]", value):
            return value
    source_id = str(item.get("source_id") or "").strip()
    if re.fullmatch(r"\d{1,3}", source_id):
        return f"[{source_id}]"
    return str(item.get("ref") or item.get("evidence_id") or item.get("id") or "").strip()


def _canonical_ref(item: Dict[str, Any]) -> str:
    return str(item.get("evidence_id") or item.get("ref") or item.get("id") or _citation_ref(item)).strip()


def _ref_keys(item: Dict[str, Any]) -> set[str]:
    source_id = str(item.get("source_id") or "").strip()
    keys = {
        str(item.get("evidence_id") or "").strip(),
        str(item.get("id") or "").strip(),
        str(item.get("ref") or "").strip(),
        str(item.get("source_ref") or "").strip(),
        str(item.get("citation_ref") or "").strip(),
        _citation_ref(item),
    }
    if source_id:
        keys.add(source_id)
        if re.fullmatch(r"\d{1,3}", source_id):
            keys.add(f"[{source_id}]")
    return {key for key in keys if key}


def _public_fact_text(item: Dict[str, Any]) -> str:
    quality = _as_dict(item.get("public_fact_quality"))
    card = _as_dict(item.get("public_fact_card") or quality.get("public_fact_card"))
    return str(
        card.get("fact")
        or card.get("distilled_fact")
        or card.get("object")
        or item.get("distilled_fact")
        or item.get("fact")
        or item.get("clean_fact")
        or ""
    ).strip()


def _clean_evidence_item(item: Dict[str, Any]) -> bool:
    quality = _as_dict(item.get("public_fact_quality"))
    if quality and not bool(quality.get("eligible_for_report")):
        return False
    if item.get("public_text_allowed") is False:
        return False
    allowed = str(item.get("allowed_use") or "").strip().lower()
    if allowed in NON_WRITABLE_ALLOWED_USES:
        return False
    readiness = str(item.get("analysis_readiness") or item.get("evidence_readiness") or "").strip().lower()
    if readiness in NON_WRITABLE_READINESS:
        return False
    if not _public_fact_text(item):
        return False
    return bool(_canonical_ref(item) or _citation_ref(item))


def _clean_items_for_package(package: Dict[str, Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    seen = set()
    for collection in PUBLIC_EVIDENCE_COLLECTIONS:
        for item in _as_list(package.get(collection)):
            if not isinstance(item, dict) or not _clean_evidence_item(item):
                continue
            key = _canonical_ref(item) or _citation_ref(item)
            if not key or key in seen:
                continue
            seen.add(key)
            items.append(item)
    return items


def _find_items_by_refs(items: Sequence[Dict[str, Any]], refs: Sequence[Any]) -> List[Dict[str, Any]]:
    wanted = {str(ref or "").strip() for ref in refs if str(ref or "").strip()}
    if not wanted:
        return []
    matched: List[Dict[str, Any]] = []
    seen = set()
    for item in items:
        if not _ref_keys(item).intersection(wanted):
            continue
        ref = _canonical_ref(item)
        if ref in seen:
            continue
        seen.add(ref)
        matched.append(item)
    return matched


def _reconcile_section(section: Dict[str, Any], clean_items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    next_section = dict(section)
    planned_refs = _dedupe(
        _as_list(next_section.get("planned_required_evidence_refs"))
        or _as_list(next_section.get("required_evidence_refs")),
        limit=12,
    )
    already_bound = _dedupe(_as_list(next_section.get("bound_evidence_refs")), limit=12)
    valid_bound_items = _find_items_by_refs(clean_items, already_bound)
    valid_planned_items = _find_items_by_refs(clean_items, planned_refs)
    valid_refs = _dedupe([_canonical_ref(item) for item in valid_bound_items + valid_planned_items], limit=8)
    matched_keys = set()
    for item in valid_bound_items + valid_planned_items:
        matched_keys.update(_ref_keys(item))
    dropped_refs = [ref for ref in planned_refs if ref not in matched_keys]
    rebound_pairs: List[Dict[str, str]] = []
    if valid_refs:
        bound_refs = valid_refs
        status = "bound"
        action = "write"
    elif clean_items:
        rebound_item = clean_items[0]
        rebound_ref = _canonical_ref(rebound_item)
        bound_refs = [rebound_ref] if rebound_ref else []
        status = "rebound" if bound_refs else "unbound"
        action = "write" if bound_refs else "recompose_outline"
        if bound_refs:
            for planned in planned_refs or [""]:
                rebound_pairs.append(
                    {
                        "from_ref": planned,
                        "to_ref": rebound_ref,
                        "reason": "same_chapter_clean_fact_fallback",
                    }
                )
    else:
        bound_refs = []
        status = "invalid_only" if planned_refs else "unbound"
        action = "reanalyze_existing" if planned_refs else "recompose_outline"

    next_section["planned_required_evidence_refs"] = planned_refs
    next_section["bound_evidence_refs"] = bound_refs
    next_section["dropped_required_evidence_refs"] = dropped_refs
    next_section["rebound_evidence_refs"] = rebound_pairs
    next_section["ref_binding_status"] = status
    next_section["ref_binding_action"] = action
    next_section["valid_required_evidence_refs"] = valid_refs
    next_section["evidence_backed"] = bool(bound_refs)
    if not bound_refs:
        next_section["public_fact_claim_allowed"] = False
    return next_section


def _metrics_for_layout(sections: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    planned_count = sum(len(_as_list(section.get("planned_required_evidence_refs"))) for section in sections)
    valid_count = sum(len(_as_list(section.get("valid_required_evidence_refs"))) for section in sections)
    dropped_count = sum(len(_as_list(section.get("dropped_required_evidence_refs"))) for section in sections)
    rebound_count = sum(len(_as_list(section.get("rebound_evidence_refs"))) for section in sections)
    unbound_count = sum(1 for section in sections if str(section.get("ref_binding_status") or "") == "unbound")
    invalid_only_count = sum(1 for section in sections if str(section.get("ref_binding_status") or "") == "invalid_only")
    bound_count = sum(1 for section in sections if _as_list(section.get("bound_evidence_refs")))
    total = len([section for section in sections if isinstance(section, dict)])
    return {
        "section_count": total,
        "planned_required_ref_count": planned_count,
        "valid_required_ref_count": valid_count,
        "dropped_required_ref_count": dropped_count,
        "rebound_ref_count": rebound_count,
        "bound_section_count": bound_count,
        "unbound_section_count": unbound_count,
        "invalid_only_section_count": invalid_only_count,
        "valid_required_ref_rate": (valid_count / planned_count) if planned_count else 0.0,
        "rebound_success_rate": (rebound_count / dropped_count) if dropped_count else 0.0,
        "unbound_section_rate": (unbound_count / total) if total else 0.0,
    }


def reconcile_section_evidence_refs(
    *,
    micro_layouts: Optional[Sequence[Dict[str, Any]]] = None,
    chapter_evidence_packages: Optional[Sequence[Dict[str, Any]]] = None,
    structured_analysis: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    del structured_analysis
    packages_by_id = {
        _norm_id(package.get("chapter_id")): package
        for package in list(chapter_evidence_packages or [])
        if isinstance(package, dict) and str(package.get("chapter_id") or "").strip()
    }
    result: List[Dict[str, Any]] = []
    for layout in list(micro_layouts or []):
        if not isinstance(layout, dict):
            continue
        next_layout = dict(layout)
        package = packages_by_id.get(_norm_id(next_layout.get("chapter_id"))) or {}
        clean_items = _clean_items_for_package(package)
        sections = [
            _reconcile_section(section, clean_items)
            for section in _as_list(next_layout.get("sections"))
            if isinstance(section, dict)
        ]
        next_layout["sections"] = sections
        next_layout["ref_binding_metrics"] = _metrics_for_layout(sections)
        next_layout["section_evidence_reconciled"] = True
        result.append(next_layout)
    return result
