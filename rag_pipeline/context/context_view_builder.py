from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, Iterable, List, Sequence

from rag_pipeline.cache.artifact_models import USABLE_FACT_STATUSES, as_dict
from rag_pipeline.cache.artifact_store import ArtifactStore, default_artifact_store
from rag_pipeline.contracts.repair_dispatcher import dispatch_repair_seed


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


def _as_context_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if _clean_scalar(value):
        return [_clean_scalar(value)]
    return []


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

    section_claim_ids = section.get("claim_ids") or []
    claim_units = store.list_claim_units(run_id, claim_ids=section_claim_ids) if section_claim_ids else []
    requested_refs = (
        _stable_unique(
            list(section.get("used_fact_refs") or [])
            + [ref for claim in claim_units for ref in (claim.get("fact_ids") or [])]
        )
        if section_claim_ids
        else []
    )
    facts = store.list_fact_cards(run_id, fact_ids=requested_refs, statuses=USABLE_FACT_STATUSES) if requested_refs else []
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
        statuses=["open", "needs_repair", "insufficient", "still_insufficient", "live_search_required"],
    )
    requirements = []
    if requirement_id:
        requirement = store.get_evidence_requirement(run_id, requirement_id)
        if requirement:
            requirements.append(_requirement_context_item(requirement))

    requirement_lookup = {
        item.get("requirement_id"): item
        for item in requirements
        if item.get("requirement_id")
    }
    for gap in gaps:
        req_id = _clean_scalar(gap.get("requirement_id"))
        if req_id and req_id not in requirement_lookup:
            requirement = _requirement_context_item(store.get_evidence_requirement(run_id, req_id))
            if requirement.get("requirement_id"):
                requirement_lookup[req_id] = requirement
    view_gaps = [_repair_gap_context_item(gap, requirement_lookup.get(_clean_scalar(gap.get("requirement_id")), {})) for gap in gaps]
    view_gaps = sorted(
        view_gaps,
        key=lambda item: (
            -_context_int(item.get("repair_priority_score")),
            _clean_scalar(item.get("gap_id")),
        ),
    )
    repair_task_seeds = [item["repair_task_seed"] for item in view_gaps if item.get("repair_task_seed")]
    status = "ready" if view_gaps else "insufficient"
    return {
        "task": "repair",
        "run_id": run_id,
        "gap_id": gap_id or "",
        "requirement_id": requirement_id or "",
        "status": status,
        "instruction": "find_missing_evidence_do_not_write_body_text" if status == "ready" else "do_not_infer",
        "requirements": requirements,
        "score_gaps": view_gaps,
        "repair_task_seeds": repair_task_seeds,
        "search_task_schedule": _repair_search_task_schedule(run_id, repair_task_seeds),
        "cache_boundary": "diagnostics_only_no_fact_text",
    }


def _repair_gap_context_item(gap: Dict[str, Any], requirement: Dict[str, Any] | None = None) -> Dict[str, Any]:
    requirement = requirement or {}
    retry_plan = as_dict(gap.get("retry_plan"))
    query_terms = _stable_unique(_as_context_list(retry_plan.get("query_terms")))
    search_seed = (
        retry_plan.get("next_search_task")
        or retry_plan.get("query_seed")
        or " ".join(query_terms)
        or ""
    )
    required_fields = _as_context_list(retry_plan.get("required_fields") or gap.get("missing") or requirement.get("required_fields") or [])
    required_source_level = _as_context_list(
        retry_plan.get("required_source_level")
        or retry_plan.get("min_source_level")
        or requirement.get("min_source_level")
    )
    priority_score, priority_reason = _repair_priority(retry_plan, gap, required_fields=required_fields)
    item = {
        "gap_id": gap.get("gap_id"),
        "requirement_id": gap.get("requirement_id"),
        "chapter_id": gap.get("chapter_id"),
        "section_id": gap.get("section_id"),
        "gap_type": gap.get("gap_type"),
        "severity": gap.get("severity"),
        "missing_fields": gap.get("missing") or [],
        "recommended_search_task_seed": search_seed,
        "proof_role": retry_plan.get("proof_role") or requirement.get("proof_role") or "",
        "required_fields": required_fields,
        "required_source_level": required_source_level,
        "lane_targets": _as_context_list(retry_plan.get("lane_targets")),
        "query_terms": query_terms,
        "blocking_gaps": _as_context_list(retry_plan.get("blocking_gaps")),
        "current_evidence_refs": _as_context_list(retry_plan.get("current_evidence_refs")),
        "current_insufficiency": retry_plan.get("current_insufficiency") or "",
        "repair_route": retry_plan.get("repair_route") or "",
        "repair_priority": retry_plan.get("repair_priority") or "",
        "repair_priority_score": priority_score,
        "repair_priority_reason": priority_reason,
        "source_stage": retry_plan.get("source_stage") or "",
        "freshness_required": bool(retry_plan.get("freshness_required") or requirement.get("freshness_required")),
        "max_cache_age_hours": retry_plan.get("max_cache_age_hours") or requirement.get("max_cache_age_hours"),
        "allowed_for_writing": False,
        "status": gap.get("status"),
    }
    item["repair_task_seed"] = _repair_task_seed(item, retry_plan)
    return item


def _context_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _repair_priority(
    retry_plan: Dict[str, Any],
    gap: Dict[str, Any],
    *,
    required_fields: Sequence[Any],
) -> tuple[int, str]:
    score = 0
    reasons: List[str] = []
    severity = _clean_scalar(gap.get("severity")).lower()
    status = _clean_scalar(gap.get("status")).lower()
    gap_type = _clean_scalar(gap.get("gap_type")).lower()
    source_stage = _clean_scalar(retry_plan.get("source_stage")).lower()
    proof_role = _clean_scalar(retry_plan.get("proof_role")).lower()
    fields = {_clean_scalar(item).lower() for item in required_fields if _clean_scalar(item)}

    severity_score = {
        "fatal": 100,
        "blocking": 80,
        "high": 60,
        "medium": 30,
        "warning": 15,
        "low": 5,
    }.get(severity, 0)
    if severity_score:
        score += severity_score
        reasons.append(f"severity:{severity}")

    status_score = {
        "live_search_required": 25,
        "still_insufficient": 15,
        "needs_repair": 10,
        "insufficient": 10,
        "open": 5,
    }.get(status, 0)
    if status_score:
        score += status_score
        reasons.append(f"status:{status}")

    if source_stage == "section_audit":
        score += 45
        reasons.append("source_stage:section_audit")
    elif source_stage.startswith("final_audit"):
        score += 35
        reasons.append("source_stage:final_audit")

    if gap_type == "metric_scope_period_unit_incomplete" or {"unit", "period"} & fields:
        score += 35
        reasons.append("gap_type:metric_scope_period_unit_incomplete")
    elif gap_type == "counter_boundary_missing" or proof_role == "counter":
        score += 25
        reasons.append("gap_type:counter_boundary_missing")
    elif "source" in gap_type or "citation" in gap_type:
        score += 20
        reasons.append("gap_type:source_trace")

    if proof_role in {"metric", "counter", "source_check", "filing"}:
        score += 10
        reasons.append(f"proof_role:{proof_role}")
    if _clean_scalar(gap.get("section_id")):
        score += 5
        reasons.append("section_bound")

    return score, ", ".join(_stable_unique(reasons))


def _compact_query_text(parts: Sequence[Any], *, max_chars: int = 240) -> str:
    tokens: List[str] = []
    seen = set()
    for part in parts:
        text = _clean_scalar(part)
        if not text:
            continue
        normalized = " ".join(text.split()).lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        tokens.append(" ".join(text.split()))
    query = " ".join(tokens).strip()
    if len(query) <= max_chars:
        return query
    return query[:max_chars].rstrip()


def _repair_preferred_source_patterns(proof_role: Any, lane_targets: Sequence[Any]) -> List[str]:
    role = _clean_scalar(proof_role).lower()
    patterns: List[str] = _stable_unique([str(item) for item in lane_targets if _clean_scalar(item)])
    if role == "metric":
        patterns.extend(["official_data", "market_research", "survey", "pdf", "annual_report"])
    elif role in {"source_check", "filing"}:
        patterns.extend(["official_data", "filing_company", "exchange_announcement", "investor_relations"])
    elif role == "counter":
        patterns.extend(["counter_evidence", "failure", "cost", "roi_unclear", "security", "compliance"])
    elif role == "case":
        patterns.extend(["customer_case", "company_disclosure", "procurement", "filing_company"])
    else:
        patterns.extend(["market_research", "official_data"])
    return _stable_unique(patterns)[:8]


def _repair_reject_if(proof_role: Any, required_fields: Sequence[Any]) -> List[str]:
    role = _clean_scalar(proof_role).lower()
    fields = {_clean_scalar(item).lower() for item in required_fields if _clean_scalar(item)}
    reject = ["snippet_only", "no_source_url", "marketing_copy_only"]
    if role in {"metric", "source_check", "filing"} or {"period", "date", "source"} & fields:
        reject.append("no_date")
    if role == "metric" or {"metric", "value", "unit", "period"} & fields:
        reject.extend(["missing_metric_value", "missing_unit", "missing_period"])
    if role == "counter":
        reject.append("support_only_counter_missing")
    return _stable_unique(reject)


def _repair_success_criteria(proof_role: Any, required_fields: Sequence[Any]) -> str:
    role = _clean_scalar(proof_role).lower()
    fields = [_clean_scalar(item) for item in required_fields if _clean_scalar(item)]
    if role == "metric" or {"metric", "value", "unit", "period", "source"}.issubset(set(fields)):
        return "Only count as repaired when metric/value/unit/period/source are all present and traceable to the page source."
    if role == "counter":
        return "Only count as repaired when the result provides traceable counter/risk evidence rather than support-only evidence."
    if role in {"source_check", "filing"}:
        return "Only count as repaired when an authoritative original source, filing, announcement, or research source is traceable by URL."
    if fields:
        return f"Only count as repaired when required fields are present: {', '.join(fields)}."
    return "Only count as repaired when the missing evidence can be traced to a concrete source URL."


def _repair_task_seed(gap_item: Dict[str, Any], retry_plan: Dict[str, Any]) -> Dict[str, Any]:
    status = _clean_scalar(gap_item.get("status"))
    previous_result_count = _context_int(retry_plan.get("result_count"))
    previous_signal_count = _context_int(retry_plan.get("signal_count"))
    cache_hit_count = _context_int(retry_plan.get("cache_hit_count"))
    live_refresh_required = status == "live_search_required" or _context_int(retry_plan.get("live_refresh_required_count")) > 0
    cache_seed_available = bool(cache_hit_count > 0 or live_refresh_required)
    required_fields = _as_context_list(gap_item.get("required_fields"))
    proof_role = gap_item.get("proof_role") or ""
    lane_targets = _as_context_list(gap_item.get("lane_targets"))
    preferred_source_patterns = _repair_preferred_source_patterns(proof_role, lane_targets)
    query = _compact_query_text(
        [
            gap_item.get("recommended_search_task_seed"),
            " ".join(_as_context_list(gap_item.get("query_terms"))),
            gap_item.get("current_insufficiency"),
            gap_item.get("gap_type"),
            proof_role,
            " ".join(required_fields),
            " ".join(preferred_source_patterns[:3]),
        ]
    )
    if not query:
        query = _compact_query_text([gap_item.get("gap_type"), " ".join(required_fields), gap_item.get("requirement_id")])
    seed = {
        "schema_version": "repair_task_seed_v2",
        "query": query,
        "agent": retry_plan.get("agent") or "iqs",
        "gap_id": gap_item.get("gap_id"),
        "requirement_id": gap_item.get("requirement_id"),
        "chapter_id": gap_item.get("chapter_id"),
        "section_id": gap_item.get("section_id"),
        "gap_type": gap_item.get("gap_type"),
        "repair_status": status,
        "proof_role": proof_role,
        "required_fields": required_fields,
        "required_source_level": _as_context_list(gap_item.get("required_source_level")),
        "lane_targets": lane_targets,
        "success_criteria": retry_plan.get("success_criteria") or _repair_success_criteria(proof_role, required_fields),
        "reject_if": _as_context_list(retry_plan.get("reject_if")) or _repair_reject_if(proof_role, required_fields),
        "preferred_source_patterns": _as_context_list(retry_plan.get("preferred_source_patterns")) or preferred_source_patterns,
        "freshness_required": bool(gap_item.get("freshness_required") or live_refresh_required),
        "max_cache_age_hours": gap_item.get("max_cache_age_hours") or retry_plan.get("max_cache_age_hours"),
        "blocking_gaps": _as_context_list(gap_item.get("blocking_gaps")),
        "targets_gap": gap_item.get("current_insufficiency") or gap_item.get("gap_type") or "",
        "evidence_goal": gap_item.get("current_insufficiency") or gap_item.get("gap_type") or "",
        "repair_route": gap_item.get("repair_route") or "",
        "repair_priority_score": gap_item.get("repair_priority_score"),
        "repair_priority_reason": gap_item.get("repair_priority_reason"),
        "source_stage": gap_item.get("source_stage") or "",
        "source": "repair_context_view",
        "allowed_for_writing": False,
        "cache_seed_available": cache_seed_available,
        "live_refresh_required": bool(live_refresh_required),
        "avoid_repeating_failed_query": bool(status == "still_insufficient" or (previous_result_count > 0 and previous_signal_count <= 0)),
        "previous_result_count": previous_result_count,
        "previous_signal_count": previous_signal_count,
        "cache_hit_count": cache_hit_count,
    }
    return dispatch_repair_seed(seed, failed_queries=_as_context_list(retry_plan.get("failed_queries") or retry_plan.get("avoid_queries")))


def _repair_cache_lookup_key(seed: Dict[str, Any]) -> str:
    payload = {
        "schema_version": "repair_search_task_cache_key_v1",
        "requirement_id": seed.get("requirement_id"),
        "gap_id": seed.get("gap_id"),
        "proof_role": seed.get("proof_role"),
        "required_fields": _as_context_list(seed.get("required_fields")),
        "query": seed.get("query"),
        "source_stage": seed.get("source_stage"),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return "repair:" + hashlib.sha256(encoded).hexdigest()


def _safe_task_id_part(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", _clean_scalar(value)).strip("-")
    return text[:48] or "gap"


def _repair_search_task(seed: Dict[str, Any], *, index: int) -> Dict[str, Any]:
    cache_scope = {
        "requirement_id": seed.get("requirement_id") or "",
        "gap_id": seed.get("gap_id") or "",
        "proof_role": seed.get("proof_role") or "",
        "required_fields": _as_context_list(seed.get("required_fields")),
    }
    cache_lookup_key = _repair_cache_lookup_key(seed)
    return {
        "schema_version": "repair_search_task_v1",
        "task_id": f"RT-{_safe_task_id_part(seed.get('gap_id'))}-{cache_lookup_key[-10:]}",
        "task_index": index,
        "query": seed.get("query") or "",
        "agent": seed.get("agent") or "iqs",
        "gap_id": seed.get("gap_id") or "",
        "requirement_id": seed.get("requirement_id") or "",
        "chapter_id": seed.get("chapter_id") or "",
        "section_id": seed.get("section_id") or "",
        "gap_type": seed.get("gap_type") or "",
        "repair_status": seed.get("repair_status") or "",
        "proof_role": seed.get("proof_role") or "",
        "required_fields": cache_scope["required_fields"],
        "required_source_level": _as_context_list(seed.get("required_source_level")),
        "lane_targets": _as_context_list(seed.get("lane_targets")),
        "success_criteria": seed.get("success_criteria") or "",
        "reject_if": _as_context_list(seed.get("reject_if")),
        "preferred_source_patterns": _as_context_list(seed.get("preferred_source_patterns")),
        "source_priority": _as_context_list(seed.get("source_priority")),
        "repair_route": seed.get("repair_route") or "",
        "required_field_focus": seed.get("required_field_focus") or "",
        "freshness_required": bool(seed.get("freshness_required") or seed.get("live_refresh_required")),
        "max_cache_age_hours": seed.get("max_cache_age_hours"),
        "live_refresh_required": bool(seed.get("live_refresh_required")),
        "avoid_repeating_failed_query": bool(seed.get("avoid_repeating_failed_query")),
        "cache_seed_available": bool(seed.get("cache_seed_available")),
        "cache_hit_count": _context_int(seed.get("cache_hit_count")),
        "previous_result_count": _context_int(seed.get("previous_result_count")),
        "previous_signal_count": _context_int(seed.get("previous_signal_count")),
        "cache_scope": cache_scope,
        "cache_lookup_key": cache_lookup_key,
        "source_stage": seed.get("source_stage") or "",
        "allowed_for_writing": False,
    }


def _repair_search_task_schedule(run_id: str, seeds: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    tasks = [_repair_search_task(seed, index=index) for index, seed in enumerate(seeds, start=1) if seed.get("query")]
    return {
        "schema_version": "repair_search_task_schedule_v1",
        "run_id": run_id,
        "source": "repair_context_view",
        "cache_boundary": "ledger_score_gaps_only_no_fact_text",
        "task_count": len(tasks),
        "tasks": tasks,
    }
