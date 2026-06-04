from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

from rag_pipeline.cache.artifact_models import USABLE_FACT_STATUSES, as_dict
from rag_pipeline.cache.artifact_store import ArtifactStore, default_artifact_store


FORBIDDEN_WRITER_KEYS = {
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


def _clean_scalar(value: Any) -> str:
    return str(value or "").strip()


def _stable_unique(values: Iterable[Any]) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        text = _clean_scalar(value)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _store() -> ArtifactStore:
    return default_artifact_store()


def _fact_context_item(fact: Dict[str, Any]) -> Dict[str, Any]:
    payload = as_dict(fact.get("payload"))
    return {
        "fact_id": fact.get("fact_id"),
        "requirement_id": fact.get("requirement_id") or payload.get("requirement_id"),
        "source_id": fact.get("source_id") or payload.get("source_id"),
        "fact": fact.get("fact") or payload.get("fact") or payload.get("claim_text"),
        "metric": fact.get("metric") or payload.get("metric"),
        "value": fact.get("value") or payload.get("value"),
        "unit": fact.get("unit") or payload.get("unit"),
        "period": fact.get("period") or payload.get("period"),
        "scope": fact.get("scope") or payload.get("scope"),
        "allowed_use": fact.get("allowed_use") or payload.get("allowed_use"),
        "analysis_role": fact.get("analysis_role") or payload.get("analysis_role"),
        "source_level": fact.get("source_level") or payload.get("source_level"),
        "status": fact.get("status"),
    }


def _requirement_context_item(requirement: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "requirement_id": requirement.get("requirement_id"),
        "chapter_id": requirement.get("chapter_id"),
        "hypothesis_id": requirement.get("hypothesis_id"),
        "proof_role": requirement.get("proof_role"),
        "required_fields": requirement.get("required_fields") or [],
        "min_source_level": requirement.get("min_source_level"),
        "claim_strength_ceiling": requirement.get("claim_strength_ceiling"),
        "freshness_required": bool(requirement.get("freshness_required")),
        "max_cache_age_hours": requirement.get("max_cache_age_hours"),
        "status": requirement.get("status"),
        "missing": requirement.get("missing") or [],
    }


def _source_context_item(source: Dict[str, Any]) -> Dict[str, Any]:
    payload = as_dict(source.get("payload"))
    return {
        "run_source_id": source.get("run_source_id"),
        "canonical_source_id": source.get("canonical_source_id"),
        "canonical_url": source.get("canonical_url") or payload.get("canonical_url") or payload.get("url"),
        "title": source.get("title") or payload.get("title"),
        "publisher": source.get("publisher") or payload.get("publisher"),
        "published_at": source.get("published_at") or payload.get("published_at"),
        "source_type": source.get("source_type") or payload.get("source_type"),
        "source_level": source.get("source_level") or payload.get("source_level"),
        "verification_status": source.get("verification_status") or payload.get("verification_status"),
        "content_hash": source.get("content_hash") or payload.get("content_hash"),
        "status": source.get("run_source_status") or source.get("status"),
    }


def _source_registry_slice(store: ArtifactStore, run_id: str, source_ids: Sequence[Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    seen = set()
    for source_id in _stable_unique(source_ids):
        source = store.resolve_run_source(run_id, source_id)
        if not source:
            continue
        run_source_id = source.get("run_source_id") or source_id
        if run_source_id in seen:
            continue
        seen.add(run_source_id)
        items.append(_source_context_item(source))
    return items


def _missing_from_requirements(requirements: Sequence[Dict[str, Any]]) -> List[Any]:
    missing: List[Any] = []
    for requirement in requirements:
        for item in requirement.get("missing") or []:
            missing.append(item)
    return missing


def _analysis_status(facts: Sequence[Dict[str, Any]], requirements: Sequence[Dict[str, Any]]) -> str:
    if not facts:
        return "insufficient"
    if any(str(req.get("status") or "").lower() == "insufficient" for req in requirements):
        return "insufficient"
    return "ready"


def build_analysis_context_view(
    run_id: str,
    requirement_id: str | None = None,
    chapter_id: str | None = None,
) -> Dict[str, Any]:
    """Return a model-facing evidence slice for analysis only."""

    store = _store()
    requirements = [
        _requirement_context_item(item)
        for item in store.list_evidence_requirements(
            run_id,
            requirement_id=requirement_id or "",
            chapter_id=chapter_id or "",
        )
    ]
    req_ids = {item["requirement_id"] for item in requirements if item.get("requirement_id")}
    if requirement_id:
        req_ids.add(requirement_id)

    if req_ids:
        facts = []
        for req_id in sorted(req_ids):
            facts.extend(
                store.list_fact_cards(
                    run_id,
                    requirement_id=req_id,
                    statuses=USABLE_FACT_STATUSES,
                )
            )
    else:
        facts = store.list_fact_cards(run_id, statuses=USABLE_FACT_STATUSES)

    usable_facts = [_fact_context_item(item) for item in facts]
    status = _analysis_status(usable_facts, requirements)
    view: Dict[str, Any] = {
        "task": "analysis",
        "run_id": run_id,
        "requirement_id": requirement_id or "",
        "chapter_id": chapter_id or "",
        "status": status,
        "instruction": "do_not_infer" if status == "insufficient" else "use_only_validated_fact_cards",
        "requirements": requirements,
        "usable_fact_cards": usable_facts,
        "source_registry_slice": _source_registry_slice(store, run_id, [item.get("source_id") for item in usable_facts]),
        "missing": _missing_from_requirements(requirements),
        "cache_boundary": "fact_cards_only",
    }
    return view


def _claim_context_item(claim: Dict[str, Any], allowed_fact_ids: Sequence[str]) -> Dict[str, Any]:
    payload = as_dict(claim.get("payload"))
    fact_ids = [item for item in claim.get("fact_ids") or payload.get("fact_ids") or [] if item in allowed_fact_ids]
    source_ids = claim.get("source_ids") or payload.get("source_ids") or []
    return {
        "claim_id": claim.get("claim_id") or payload.get("claim_id"),
        "claim": claim.get("claim") or claim.get("text") or payload.get("claim") or payload.get("text"),
        "requirement_ids": claim.get("requirement_ids") or payload.get("requirement_ids") or [],
        "fact_ids": fact_ids,
        "source_ids": source_ids,
        "claim_strength": claim.get("claim_strength") or payload.get("claim_strength") or payload.get("strength"),
        "claim_strength_ceiling": claim.get("claim_strength_ceiling") or payload.get("claim_strength_ceiling"),
        "limitation_boundary": (
            claim.get("limitation_boundary")
            or claim.get("boundary")
            or payload.get("limitation_boundary")
            or payload.get("boundary")
        ),
        "status": claim.get("status"),
    }


def build_writer_context_view(run_id: str, section_id: str) -> Dict[str, Any]:
    """Return the narrowed claim/fact slice a writer is allowed to use."""

    store = _store()
    section = store.get_section(run_id, section_id)
    if not section:
        return {
            "task": "write_section",
            "run_id": run_id,
            "section_id": section_id,
            "status": "insufficient",
            "instruction": "do_not_infer",
            "claim_units": [],
            "usable_fact_cards": [],
            "used_fact_refs": [],
            "source_registry_slice": [],
        }

    claim_units = store.list_claim_units(run_id, claim_ids=section.get("claim_ids") or [])
    requested_refs = _stable_unique(
        list(section.get("used_fact_refs") or [])
        + [ref for claim in claim_units for ref in (claim.get("fact_ids") or [])]
    )
    facts = store.list_fact_cards(run_id, fact_ids=requested_refs, statuses=USABLE_FACT_STATUSES)
    usable_facts = [_fact_context_item(item) for item in facts]
    allowed_fact_ids = _stable_unique(item.get("fact_id") for item in usable_facts)
    source_ids = _stable_unique(
        [item.get("source_id") for item in usable_facts]
        + [source_id for claim in claim_units for source_id in (claim.get("source_ids") or [])]
    )
    claims = [_claim_context_item(item, allowed_fact_ids) for item in claim_units]
    requirements = [
        _requirement_context_item(store.get_evidence_requirement(run_id, req_id))
        for req_id in section.get("requirement_ids") or []
        if req_id
    ]
    status = "ready" if claims and usable_facts else "insufficient"
    return {
        "task": "write_section",
        "run_id": run_id,
        "section_id": section_id,
        "status": status,
        "instruction": "do_not_infer" if status == "insufficient" else "write_only_from_claim_units_and_fact_refs",
        "requirements": [item for item in requirements if item.get("requirement_id")],
        "claim_units": claims,
        "usable_fact_cards": usable_facts,
        "used_fact_refs": allowed_fact_ids,
        "source_registry_slice": _source_registry_slice(store, run_id, source_ids),
        "section_boundary": section.get("boundary") or section.get("limitation_boundary") or "",
        "evidence_backed": bool(section.get("evidence_backed")),
        "cache_boundary": "no_raw_pages_no_diagnostics",
    }


def build_repair_context_view(
    run_id: str,
    gap_id: str | None = None,
    requirement_id: str | None = None,
) -> Dict[str, Any]:
    """Return score gaps and retry seeds, without quoteable evidence text."""

    store = _store()
    gaps = store.list_score_gaps(
        run_id,
        gap_id=gap_id or "",
        requirement_id=requirement_id or "",
        statuses=["open", "needs_repair", "insufficient"],
    )
    requirements = []
    if requirement_id:
        requirement = store.get_evidence_requirement(run_id, requirement_id)
        if requirement:
            requirements.append(_requirement_context_item(requirement))

    view_gaps = [
        {
            "gap_id": gap.get("gap_id"),
            "requirement_id": gap.get("requirement_id"),
            "chapter_id": gap.get("chapter_id"),
            "section_id": gap.get("section_id"),
            "gap_type": gap.get("gap_type"),
            "severity": gap.get("severity"),
            "missing_fields": gap.get("missing") or [],
            "recommended_search_task_seed": as_dict(gap.get("retry_plan")).get("next_search_task")
            or as_dict(gap.get("retry_plan")).get("query_seed")
            or "",
            "status": gap.get("status"),
        }
        for gap in gaps
    ]
    return {
        "task": "repair",
        "run_id": run_id,
        "gap_id": gap_id or "",
        "requirement_id": requirement_id or "",
        "status": "ready" if view_gaps else "insufficient",
        "instruction": "find_missing_evidence_do_not_write_body_text",
        "requirements": requirements,
        "score_gaps": view_gaps,
        "cache_boundary": "diagnostics_only_no_fact_text",
    }
