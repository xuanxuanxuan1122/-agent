from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from rag_pipeline.contracts.claim_roles import classify_claim_unit_roles
from rag_pipeline.contracts.section_audit import audit_section_claim_roles
from rag_pipeline.runtime_cache import json_safe_default

from .artifact_store import ArtifactStore


FACT_LIST_KEYS = (
    "analysis_ready_evidence",
    "clean_evidence_list",
    "normalized_evidence",
    "core_evidence",
    "supporting_evidence",
    "directional_evidence",
    "metric_evidence",
    "clue_evidence",
    "appendix_evidence",
    "raw_data_points",
    "fact_cards",
    "extracted_fact_cards",
)

RESEARCH_REFLECTION_ARTIFACT_TYPE = "research_reflection_memo"
RESEARCH_REFLECTION_STAGE = "research_reflection"


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _text(value: Any) -> str:
    return str(value or "").strip()


def _first_text(*values: Any) -> str:
    for value in values:
        text = _text(value)
        if text:
            return text
    return ""


def _int_value(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _dedupe(values: Iterable[Any], *, limit: int = 200) -> List[str]:
    result: List[str] = []
    seen: Set[str] = set()
    for value in values:
        text = _text(value)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=json_safe_default)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stable_id(prefix: str, payload: Any, *, length: int = 14) -> str:
    return f"{prefix}-{_json_hash(payload)[:length]}"


def _lineage_list(payload: Dict[str, Any], key: str) -> List[str]:
    lineage = _as_dict(payload.get("lineage"))
    return _dedupe(
        [
            *_as_list(payload.get(key)),
            *_as_list(lineage.get(key)),
        ]
    )


def _requirement_ids(payload: Dict[str, Any]) -> List[str]:
    return _dedupe(
        [
            payload.get("requirement_id"),
            payload.get("evidence_requirement_id"),
            payload.get("evidence_goal_id"),
            payload.get("goal_id"),
            *_as_list(payload.get("requirement_ids")),
            *_as_list(payload.get("evidence_requirement_ids")),
            *_as_list(payload.get("evidence_goal_ids")),
            *_lineage_list(payload, "requirement_ids"),
        ]
    )


def _fact_refs(payload: Dict[str, Any]) -> List[str]:
    return _dedupe(
        [
            *_as_list(payload.get("fact_ids")),
            *_as_list(payload.get("used_fact_refs")),
            *_as_list(payload.get("evidence_refs")),
            *_as_list(payload.get("supporting_fact_refs")),
            *_as_list(payload.get("supporting_evidence_refs")),
            *_as_list(payload.get("used_evidence_ids")),
            *_lineage_list(payload, "fact_ids"),
        ],
        limit=80,
    )


def _source_ids(payload: Dict[str, Any]) -> List[str]:
    return _dedupe(
        [
            payload.get("source_id"),
            payload.get("source_ref"),
            payload.get("citation_ref"),
            *_as_list(payload.get("source_ids")),
            *_as_list(payload.get("source_refs")),
            *_lineage_list(payload, "source_ids"),
        ],
        limit=80,
    )


def _source_run_ref(source: Dict[str, Any], index: int) -> str:
    ref = _first_text(
        source.get("run_source_id"),
        source.get("source_id"),
        source.get("id"),
        source.get("ref"),
        source.get("citation_ref"),
        source.get("source_ref"),
    )
    if ref:
        return ref
    return f"SRC-{index:03d}"


def _source_from_fact(fact: Dict[str, Any], source_id: str) -> Dict[str, Any]:
    source = dict(_as_dict(fact.get("source")))
    source.setdefault("id", source_id)
    source.setdefault("source_id", source_id)
    for target, keys in {
        "canonical_url": ("canonical_url", "source_url", "url"),
        "title": ("source_title", "title", "document_title"),
        "publisher": ("publisher", "source_publisher"),
        "published_at": ("published_at", "date", "source_date"),
        "source_level": ("source_level",),
        "verification_status": ("source_verification_status", "verification_status"),
        "content_hash": ("content_hash",),
    }.items():
        if source.get(target):
            continue
        value = _first_text(*(fact.get(key) for key in keys))
        if value:
            source[target] = value
    return source


def _fact_id(item: Dict[str, Any], index: int) -> str:
    return _first_text(
        item.get("fact_id"),
        item.get("evidence_id"),
        item.get("source_ref"),
        item.get("ref"),
        item.get("id"),
    ) or f"EV-{index:04d}"


def _fact_text(item: Dict[str, Any]) -> str:
    public_card = _as_dict(item.get("public_fact_card")) or _as_dict(_as_dict(item.get("public_fact_quality")).get("public_fact_card"))
    return _first_text(
        item.get("fact"),
        item.get("distilled_fact"),
        item.get("evidence_text"),
        item.get("text"),
        item.get("claim_text"),
        public_card.get("fact"),
        public_card.get("object"),
    )


def _fact_status(item: Dict[str, Any]) -> str:
    status = _text(item.get("status")).lower()
    if status in {"validated", "admissible", "rejected", "stale", "superseded", "appendix_only"}:
        return status
    if _text(item.get("allowed_use")).lower() == "appendix_only":
        return "appendix_only"
    quality = _as_dict(item.get("public_fact_quality"))
    if quality and quality.get("eligible_for_report") is False:
        return "rejected"
    return "validated"


def _iter_fact_candidates(writer_package: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    evidence_package = _as_dict(writer_package.get("evidence_package"))
    structured_analysis = _as_dict(writer_package.get("structured_analysis"))
    for key in FACT_LIST_KEYS:
        for item in _as_list(evidence_package.get(key)):
            if isinstance(item, dict):
                yield item
        for item in _as_list(structured_analysis.get(key)):
            if isinstance(item, dict):
                yield item
    for chapter in _as_list(writer_package.get("chapter_evidence_packages")):
        if not isinstance(chapter, dict):
            continue
        for key in FACT_LIST_KEYS:
            for item in _as_list(chapter.get(key)):
                if isinstance(item, dict):
                    yield item
        analysis = _as_dict(chapter.get("chapter_analysis"))
        for item in _as_list(analysis.get("fact_cards")):
            if isinstance(item, dict):
                merged = {**item, "chapter_id": item.get("chapter_id") or chapter.get("chapter_id")}
                yield merged


def _iter_requirements(writer_package: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    evidence_package = _as_dict(writer_package.get("evidence_package"))
    structured_analysis = _as_dict(writer_package.get("structured_analysis"))
    contract = _as_dict(evidence_package.get("report_contract"))
    req_payload = _as_dict(contract.get("evidence_requirements"))
    for requirement in _as_list(req_payload.get("requirements")):
        if isinstance(requirement, dict):
            yield requirement
    for chapter in _as_list(_as_dict(writer_package.get("report_blueprint")).get("chapters")):
        if not isinstance(chapter, dict):
            continue
        for requirement in _as_list(chapter.get("evidence_requirements")):
            if isinstance(requirement, dict):
                yield {**requirement, "chapter_id": requirement.get("chapter_id") or chapter.get("chapter_id")}
    for plan in _iter_research_plans(writer_package, evidence_package, structured_analysis):
        for key in ("requirements", "evidence_requirements", "evidence_goals"):
            for requirement in _as_list(plan.get(key)):
                if not isinstance(requirement, dict):
                    continue
                requirement_id = _first_text(
                    requirement.get("requirement_id"),
                    requirement.get("evidence_requirement_id"),
                    requirement.get("goal_id"),
                    requirement.get("evidence_goal_id"),
                )
                if requirement_id:
                    yield {**requirement, "requirement_id": requirement_id}
        for task in _as_list(plan.get("search_tasks")):
            if not isinstance(task, dict):
                continue
            requirement_id = _first_text(task.get("requirement_id"), task.get("evidence_requirement_id"), task.get("evidence_goal_id"))
            if requirement_id:
                yield {**task, "requirement_id": requirement_id}


def _iter_research_plans(*containers: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    seen: Set[str] = set()
    for container in containers:
        if not isinstance(container, dict):
            continue
        for plan in (
            _as_dict(container.get("research_plan")),
            _as_dict(_as_dict(container.get("metadata")).get("research_plan")),
        ):
            if not plan:
                continue
            key = _json_hash(plan)
            if key in seen:
                continue
            seen.add(key)
            yield plan


def _iter_search_tasks(writer_package: Dict[str, Any], writer_report: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    evidence_package = _as_dict(writer_package.get("evidence_package"))
    structured_analysis = _as_dict(writer_package.get("structured_analysis"))
    for container in (
        evidence_package,
        writer_package,
        writer_report,
    ):
        schedule = _as_dict(container.get("search_task_schedule"))
        for key in ("tasks", "search_tasks", "scheduled_tasks"):
            for item in _as_list(schedule.get(key)):
                if isinstance(item, dict):
                    yield item
        for item in _as_list(container.get("search_tasks")):
            if isinstance(item, dict):
                yield item
    for plan in _iter_research_plans(writer_package, evidence_package, structured_analysis, writer_report):
        for item in _as_list(plan.get("search_tasks")):
            if isinstance(item, dict):
                yield item


def _iter_claim_units(writer_package: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    structured = _as_dict(writer_package.get("structured_analysis"))
    for item in _as_list(writer_package.get("argument_units")):
        if isinstance(item, dict):
            yield item
    for item in _as_list(structured.get("claim_units")):
        if isinstance(item, dict):
            yield item


def _iter_sections(writer_package: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for chapter in _as_list(writer_package.get("chapter_packages")):
        if not isinstance(chapter, dict):
            continue
        chapter_id = _text(chapter.get("chapter_id"))
        for index, section in enumerate(_as_list(chapter.get("sections")), start=1):
            if isinstance(section, dict):
                yield {**section, "chapter_id": section.get("chapter_id") or chapter_id, "_section_index": index}


def record_stage_snapshot_artifact(
    store: ArtifactStore,
    *,
    run_id: str,
    stage_name: str,
    payload: Any,
    snapshot_result: Dict[str, Any],
) -> Dict[str, Any]:
    storage_uri = _text(snapshot_result.get("full_payload_path"))
    status = "validated" if snapshot_result.get("stored") else "snapshot_failed"
    result = store.record_artifact(
        run_id=run_id,
        stage=stage_name,
        artifact_type=stage_name,
        payload=payload,
        status=status,
        storage_uri=storage_uri,
        lineage={
            "stage_snapshot": True,
            "replayable": bool(snapshot_result.get("replayable")),
            "manifest_path": _text(snapshot_result.get("manifest_path")),
            "reason": _text(snapshot_result.get("reason")),
        },
    )
    return {
        "artifact_id": result.artifact_id,
        "status": result.status,
        "storage_uri": result.storage_uri,
        "content_hash": result.output_hash,
    }


def _normalize_proof_role(value: Any) -> str:
    role = _text(value).lower()
    if role in {"required_proof_role", "evidence_type"}:
        return ""
    if "counter" in role or "反证" in role or "risk" in role:
        return "counter"
    if "metric" in role or "data" in role or "指标" in role or "统计" in role:
        return "metric"
    if "case" in role or "customer" in role or "案例" in role:
        return "case"
    if "source" in role or "official" in role or "filing" in role or "来源" in role:
        return "source_check"
    if "expert" in role or "专家" in role:
        return "expert"
    if "support" in role or "supporting" in role:
        return "support"
    return role


def _proof_role_from_payload(payload: Dict[str, Any]) -> str:
    card = _as_dict(payload.get("evidence_card")) or _as_dict(payload.get("public_fact_card"))
    quality_card = _as_dict(_as_dict(payload.get("public_fact_quality")).get("public_fact_card"))
    role = _normalize_proof_role(
        _first_text(
            payload.get("proof_role"),
            payload.get("required_proof_role"),
            payload.get("evidence_type"),
            payload.get("analysis_role"),
            payload.get("evidence_role"),
            card.get("proof_role"),
            card.get("claim_type"),
            quality_card.get("proof_role"),
        )
    )
    if role:
        return role
    gap_type = _text(payload.get("gap_type") or payload.get("type"))
    if "counter" in gap_type:
        return "counter"
    if "metric" in gap_type or "scope_period_unit" in gap_type:
        return "metric"
    if "case" in gap_type:
        return "case"
    if "source" in gap_type or "citation" in gap_type:
        return "source_check"
    return ""


def _chapter_id_from_payload(payload: Dict[str, Any]) -> str:
    return _first_text(
        payload.get("chapter_id"),
        payload.get("dimension_id"),
        payload.get("section_chapter_id"),
        _as_dict(payload.get("lineage")).get("chapter_id"),
    )


def _chapter_keys_from_payload(payload: Dict[str, Any]) -> List[str]:
    lineage = _as_dict(payload.get("lineage"))
    return _dedupe(
        [
            payload.get("chapter_id"),
            payload.get("dimension_id"),
            payload.get("section_chapter_id"),
            payload.get("chapter_title"),
            payload.get("dimension_name"),
            payload.get("chapter_question"),
            lineage.get("chapter_id"),
            lineage.get("chapter_title"),
        ],
        limit=12,
    )


def _build_requirement_lookup(requirements: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_id: Set[str] = set()
    by_chapter_role: Dict[tuple[str, str], List[str]] = {}
    for requirement in requirements:
        if not isinstance(requirement, dict):
            continue
        requirement_id = _first_text(requirement.get("requirement_id"), requirement.get("evidence_requirement_id"))
        if not requirement_id:
            continue
        by_id.add(requirement_id)
        role = _proof_role_from_payload(requirement) or _normalize_proof_role(requirement.get("proof_role")) or "support"
        for chapter_id in _chapter_keys_from_payload(requirement):
            if not role:
                continue
            key = (chapter_id, role)
            by_chapter_role[key] = _dedupe([*by_chapter_role.get(key, []), requirement_id], limit=20)
    return {"by_id": by_id, "by_chapter_role": by_chapter_role}


def _infer_requirement_ids(
    payload: Dict[str, Any],
    *,
    requirement_lookup: Dict[str, Any],
    task_requirement_lookup: Dict[str, str],
) -> List[str]:
    known_ids: Set[str] = set(requirement_lookup.get("by_id") or set())
    req_ids = [req_id for req_id in _requirement_ids(payload) if not known_ids or req_id in known_ids]
    task_id = _first_text(payload.get("task_id"), payload.get("search_task_id"), payload.get("source_task_id"))
    if task_id and task_requirement_lookup.get(task_id):
        req_ids.append(task_requirement_lookup[task_id])
    role = _proof_role_from_payload(payload)
    by_chapter_role = _as_dict(requirement_lookup.get("by_chapter_role"))
    if role:
        for chapter_id in _chapter_keys_from_payload(payload):
            req_ids.extend(_as_list(by_chapter_role.get((chapter_id, role))))
    return _dedupe(req_ids, limit=20)


def ingest_writer_package_artifacts(
    store: ArtifactStore,
    *,
    run_id: str,
    writer_package: Dict[str, Any],
    writer_report: Optional[Dict[str, Any]] = None,
    final_audit_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    writer_report = writer_report or _as_dict(writer_package.get("writer_report"))
    summary = {
        "enabled": True,
        "run_id": run_id,
        "requirement_count": 0,
        "search_task_count": 0,
        "source_count": 0,
        "fact_card_count": 0,
        "claim_unit_count": 0,
        "section_count": 0,
        "score_gap_count": 0,
        "score_gap_status_update_count": 0,
        "research_reflection_artifact_count": 0,
        "research_reflection_seed_count": 0,
        "lineage_edge_count": 0,
        "lineage_edge_total": 0,
    }
    edge_keys: Set[tuple[str, str, str, str, str]] = set()

    def add_edge(from_type: str, from_id: Any, to_type: str, to_id: Any, relation: str = "derived") -> None:
        source_id = _text(from_id)
        target_id = _text(to_id)
        if not source_id or not target_id:
            return
        key = (from_type, source_id, to_type, target_id, relation)
        if key in edge_keys:
            return
        edge_keys.add(key)
        if store.add_lineage_edge(run_id, from_type, source_id, to_type, target_id, relation):
            summary["lineage_edge_count"] += 1

    requirement_records: List[Dict[str, Any]] = []
    seen_requirement_ids: Set[str] = set()
    for requirement in _iter_requirements(writer_package):
        requirement_id = _first_text(requirement.get("requirement_id"), requirement.get("evidence_requirement_id"))
        if not requirement_id or requirement_id in seen_requirement_ids:
            continue
        seen_requirement_ids.add(requirement_id)
        requirement_records.append({**requirement, "requirement_id": requirement_id})
        store.upsert_evidence_requirement(
            run_id=run_id,
            requirement_id=requirement_id,
            chapter_id=_text(requirement.get("chapter_id")),
            hypothesis_id=_text(requirement.get("hypothesis_id")),
            proof_role=_text(requirement.get("proof_role")) or "support",
            required_fields=_as_list(requirement.get("required_fields")),
            min_source_level=requirement.get("min_source_level") or requirement.get("required_source_levels") or requirement.get("source_level"),
            claim_strength_ceiling=_text(requirement.get("claim_strength_ceiling")),
            freshness_required=bool(requirement.get("freshness_required")),
            max_cache_age_hours=requirement.get("max_cache_age_hours"),
            status=_text(requirement.get("status")) or "open",
            missing=_as_list(requirement.get("missing")),
        )
        summary["requirement_count"] += 1
    requirement_lookup = _build_requirement_lookup(requirement_records)
    research_reflection_memo = _research_reflection_memo_from_payloads(writer_package, writer_report)
    research_reflection_artifact_id = ""
    if research_reflection_memo:
        research_reflection_artifact_id = _record_research_reflection_artifact(
            store,
            run_id=run_id,
            memo=research_reflection_memo,
            add_edge=add_edge,
        )
        summary["research_reflection_artifact_count"] = 1 if research_reflection_artifact_id else 0
        summary["research_reflection_seed_count"] = len(
            [
                seed
                for seed in _as_list(research_reflection_memo.get("next_search_task_seeds"))
                if isinstance(seed, dict)
            ]
        )

    for index, source in enumerate(_as_list(writer_package.get("source_registry")), start=1):
        if not isinstance(source, dict):
            continue
        run_source_id = _source_run_ref(source, index)
        store.upsert_source(run_id=run_id, run_source_id=run_source_id, source=source)
        summary["source_count"] += 1

    task_requirement_lookup: Dict[str, str] = {}
    for index, task in enumerate(_iter_search_tasks(writer_package, writer_report), start=1):
        task_id = _first_text(task.get("task_id"), task.get("search_task_id"), task.get("id")) or f"ST-{index:04d}"
        req_ids = _infer_requirement_ids(task, requirement_lookup=requirement_lookup, task_requirement_lookup={})
        req_id = req_ids[0] if req_ids else _first_text(task.get("requirement_id"), task.get("evidence_requirement_id"))
        if req_id:
            task_requirement_lookup[task_id] = req_id
        store.record_artifact(
            run_id=run_id,
            stage="search_task",
            artifact_type="search_task",
            payload=task,
            status=_text(task.get("status")) or "scheduled",
            requirement_id=req_id,
            lineage={"search_task_id": task_id},
        )
        if req_id:
            add_edge("requirement", req_id, "search_task", task_id, "plans")
        summary["search_task_count"] += 1

    seen_facts: Set[str] = set()
    # fact-layer lineage maps: used to backfill requirement/source ids onto
    # claim_units/sections whose own payloads (and the chapter/role lookups)
    # do not carry requirement_ids. The graph already links facts to both
    # requirements (requires) and claims (supports); this projects that link
    # down onto the denormalized columns so the ledger rows are queryable.
    fact_requirement_map: Dict[str, List[str]] = {}
    fact_source_map: Dict[str, str] = {}
    fact_payload_map: Dict[str, Dict[str, Any]] = {}
    for index, item in enumerate(_iter_fact_candidates(writer_package), start=1):
        fact_id = _fact_id(item, index)
        if fact_id in seen_facts:
            continue
        fact_text = _fact_text(item)
        if not fact_text:
            continue
        seen_facts.add(fact_id)
        source_id = _first_text(
            item.get("source_id"),
            item.get("source_ref"),
            item.get("citation_ref"),
            _as_dict(item.get("source")).get("source_id"),
            _as_dict(item.get("source")).get("id"),
            _as_dict(item.get("source")).get("ref"),
        ) or f"SRC-{fact_id}"
        if _as_dict(item.get("source")) or item.get("source_url") or item.get("url"):
            store.upsert_source(run_id=run_id, run_source_id=source_id, source=_source_from_fact(item, source_id))
        req_ids = _infer_requirement_ids(
            item,
            requirement_lookup=requirement_lookup,
            task_requirement_lookup=task_requirement_lookup,
        )
        if req_ids:
            fact_requirement_map[fact_id] = req_ids
        if source_id:
            fact_source_map[fact_id] = source_id
        fact_payload = {
            **item,
            "fact_id": fact_id,
            "evidence_id": fact_id,
            "requirement_id": req_ids[0] if req_ids else _text(item.get("requirement_id")),
            "source_id": source_id,
            "analysis_role": _text(item.get("analysis_role") or item.get("proof_role")),
            "proof_role": _text(item.get("proof_role") or _as_dict(item.get("evidence_card")).get("proof_role")),
            "status": _fact_status(item),
        }
        fact_payload_map[fact_id] = fact_payload
        store.upsert_fact_card(
            run_id=run_id,
            fact_id=fact_id,
            requirement_id=req_ids[0] if req_ids else "",
            source_id=source_id,
            fact=fact_text,
            metric=_text(item.get("metric")),
            value=_text(item.get("value")),
            unit=_text(item.get("unit")),
            period=_text(item.get("period")),
            scope=_text(item.get("scope")),
            allowed_use=_text(item.get("allowed_use")),
            analysis_eligible=bool(item.get("analysis_eligible")),
            analysis_role=_text(item.get("analysis_role") or item.get("proof_role")),
            source_level=_text(item.get("source_level")),
            status=_fact_status(item),
            payload=item,
        )
        for req_id in req_ids:
            add_edge("requirement", req_id, "fact_card", fact_id, "requires")
        add_edge("source", source_id, "fact_card", fact_id, "supports")
        search_task_id = _first_text(item.get("search_task_id"), item.get("task_id"))
        if search_task_id:
            add_edge("search_task", search_task_id, "fact_card", fact_id, "found")
        summary["fact_card_count"] += 1

    seen_claims: Set[str] = set()
    claim_requirement_map: Dict[str, List[str]] = {}
    claim_payload_map: Dict[str, Dict[str, Any]] = {}
    for index, claim in enumerate(_iter_claim_units(writer_package), start=1):
        claim_id = _first_text(claim.get("claim_id"), claim.get("id")) or f"CL-{index:04d}"
        if claim_id in seen_claims:
            continue
        seen_claims.add(claim_id)
        fact_ids = _fact_refs(claim)
        req_ids = _infer_requirement_ids(
            claim,
            requirement_lookup=requirement_lookup,
            task_requirement_lookup=task_requirement_lookup,
        )
        if not req_ids:
            req_ids = _dedupe(
                [rid for fact_id in fact_ids for rid in fact_requirement_map.get(fact_id, [])],
                limit=20,
            )
        source_ids = _source_ids(claim)
        if not source_ids:
            source_ids = _dedupe(
                [fact_source_map[fact_id] for fact_id in fact_ids if fact_source_map.get(fact_id)],
                limit=20,
            )
        claim_payload = {
            **claim,
            "claim_id": claim_id,
            "requirement_ids": req_ids,
            "fact_ids": fact_ids,
            "source_ids": source_ids,
        }
        if not _as_list(claim_payload.get("claim_roles")):
            claim_payload = {
                **claim_payload,
                **classify_claim_unit_roles(claim_payload, fact_payload_map),
            }
        claim_requirement_map[claim_id] = req_ids
        claim_payload_map[claim_id] = claim_payload
        store.upsert_claim_unit(
            run_id=run_id,
            claim_id=claim_id,
            payload=claim_payload,
            requirement_ids=req_ids,
            fact_ids=fact_ids,
            source_ids=source_ids,
            status=_text(claim.get("status")) or "validated",
        )
        for req_id in req_ids:
            add_edge("requirement", req_id, "claim_unit", claim_id, "constrains")
        for fact_id in fact_ids:
            add_edge("fact_card", fact_id, "claim_unit", claim_id, "supports")
        summary["claim_unit_count"] += 1

    seen_sections: Set[str] = set()
    for section in _iter_sections(writer_package):
        section_id = _first_text(section.get("section_id"), section.get("id"), section.get("claim_id"))
        if not section_id:
            section_id = f"SEC-{_text(section.get('chapter_id')) or 'chapter'}-{int(section.get('_section_index') or 0):02d}"
        if section_id in seen_sections:
            continue
        seen_sections.add(section_id)
        claim_ids = _dedupe([section.get("claim_id"), *_as_list(section.get("claim_ids"))], limit=40)
        fact_ids = _fact_refs(section)
        # P5: if the section did not carry claim_ids (the composer often fails to
        # propagate them — the "N claim_units but sections.claim_ids empty" gap),
        # recover the binding from fact overlap so the section is provably tied to
        # the claim_units it consumes, not just to raw facts.
        if not claim_ids and fact_ids:
            fact_id_set = set(fact_ids)
            claim_ids = _dedupe(
                [
                    cid
                    for cid, payload in claim_payload_map.items()
                    if fact_id_set.intersection(set(_as_list(payload.get("fact_ids"))))
                ],
                limit=40,
            )
        req_ids = _infer_requirement_ids(
            section,
            requirement_lookup=requirement_lookup,
            task_requirement_lookup=task_requirement_lookup,
        )
        for claim_id in claim_ids:
            req_ids = _dedupe([*req_ids, *claim_requirement_map.get(claim_id, [])], limit=80)
        if not req_ids:
            req_ids = _dedupe(
                [rid for fact_id in fact_ids for rid in fact_requirement_map.get(fact_id, [])],
                limit=80,
            )
        section_payload = {
            **section,
            "section_id": section_id,
            "requirement_ids": req_ids,
            "claim_ids": claim_ids,
            "used_fact_refs": fact_ids,
        }
        store.upsert_section(
            run_id=run_id,
            section_id=section_id,
            payload=section_payload,
            requirement_ids=req_ids,
            claim_ids=claim_ids,
            used_fact_refs=fact_ids,
            evidence_backed=bool(section.get("evidence_backed") or fact_ids),
            status=_text(section.get("status")) or "validated",
        )
        section_audit = audit_section_claim_roles(section_payload, claim_payload_map, fact_payload_map)
        for gap in _as_list(section_audit.get("score_gaps")):
            item = _as_dict(gap)
            gap_id = _first_text(item.get("gap_id")) or _stable_gap_id(item, {"source": "section_audit"}, _gap_type(item, "section_audit"))
            requirement_id = _first_text(item.get("requirement_id"))
            store.upsert_score_gap(
                run_id=run_id,
                gap_id=gap_id,
                requirement_id=requirement_id,
                chapter_id=_first_text(item.get("chapter_id"), section_payload.get("chapter_id")),
                section_id=_first_text(item.get("section_id"), section_id),
                gap_type=_first_text(item.get("gap_type")) or "section_audit_gap",
                severity=_first_text(item.get("severity")),
                missing=_as_list(item.get("missing")),
                retry_plan=_as_dict(item.get("retry_plan")),
                status=_text(item.get("status")) or "open",
            )
            if requirement_id:
                add_edge("requirement", requirement_id, "score_gap", gap_id, "gap")
            add_edge("section", section_id, "score_gap", gap_id, "gap")
            summary["score_gap_count"] += 1
        for req_id in req_ids:
            add_edge("requirement", req_id, "section", section_id, "rendered_in")
        for claim_id in claim_ids:
            add_edge("claim_unit", claim_id, "section", section_id, "renders")
        for fact_id in fact_ids:
            add_edge("fact_card", fact_id, "section", section_id, "cited_by")
        summary["section_count"] += 1

    gap_count = _ingest_score_gaps(
        store,
        run_id=run_id,
        writer_package=writer_package,
        writer_report=writer_report,
        final_audit_result=final_audit_result,
        requirement_lookup=requirement_lookup,
        task_requirement_lookup=task_requirement_lookup,
        research_reflection_artifact_id=research_reflection_artifact_id,
        add_edge=add_edge,
    )
    summary["score_gap_count"] += gap_count
    summary["score_gap_status_update_count"] += _apply_score_gap_status_updates(
        store,
        run_id=run_id,
        writer_package=writer_package,
        writer_report=writer_report,
    )
    # ``lineage_edge_count`` is the number of edges *newly inserted on this call*
    # (0 on idempotent re-ingest). ``lineage_edge_total`` is the real edge count
    # for the run, so summaries/replays report the persisted graph rather than 0.
    summary["lineage_edge_total"] = store.count_lineage_edges(run_id)
    return summary


def _iter_quality_gap_payloads(
    writer_package: Dict[str, Any],
    writer_report: Dict[str, Any],
    final_audit_result: Optional[Dict[str, Any]],
    *,
    research_reflection_artifact_id: str = "",
) -> Iterable[Dict[str, Any]]:
    yield from _iter_evidence_gap_ledger_payloads(writer_package, writer_report)
    yield from _iter_research_reflection_gap_payloads(
        writer_package,
        writer_report,
        research_reflection_artifact_id=research_reflection_artifact_id,
    )
    qa = _as_dict(writer_report.get("qa_result"))
    for key in (
        "quality_findings",
        "errors",
        "warnings",
        "blocking_followups",
        "blocking_evidence_repair_followups",
        "blocking_content_repair_followups",
        "required_followups",
    ):
        for index, item in enumerate(_as_list(qa.get(key)), start=1):
            yield {"source": f"qa_result.{key}", "index": index, "payload": item}
    for key in ("layout_diagnostics", "ref_lineage_diagnostics", "source_claim_support"):
        payload = _as_dict(writer_report.get(key)) or _as_dict(writer_package.get(key))
        if payload and str(payload.get("status") or "").lower() not in {"", "ok", "passed", "success"}:
            yield {"source": key, "index": 1, "payload": payload}
    final_audit = _as_dict(final_audit_result) or _as_dict(writer_report.get("final_audit_result"))
    for container_key, list_key in (("audit", "critical_findings"), ("deterministic_audit", "findings")):
        container = _as_dict(final_audit.get(container_key))
        for index, item in enumerate(_as_list(container.get(list_key)), start=1):
            yield {"source": f"final_audit.{container_key}.{list_key}", "index": index, "payload": item}


def _iter_evidence_gap_ledger_payloads(
    writer_package: Dict[str, Any],
    writer_report: Dict[str, Any],
) -> Iterable[Dict[str, Any]]:
    containers = (
        ("writer_package.evidence_package", _as_dict(writer_package.get("evidence_package"))),
        ("writer_package.structured_analysis", _as_dict(writer_package.get("structured_analysis"))),
        ("writer_package", writer_package),
        ("writer_report.evidence_package", _as_dict(writer_report.get("evidence_package"))),
        ("writer_report.structured_analysis", _as_dict(writer_report.get("structured_analysis"))),
        ("writer_report", writer_report),
    )
    seen: Set[str] = set()
    for source_name, container in containers:
        if not container:
            continue
        for key in ("evidence_gap_ledger", "score_gap_ledger", "claim_repair_priorities"):
            for index, item in enumerate(_as_list(container.get(key)), start=1):
                if not isinstance(item, dict):
                    continue
                gap_key = _first_text(item.get("gap_id"), item.get("id")) or _json_hash(item)
                if gap_key in seen:
                    continue
                seen.add(gap_key)
                yield {
                    "source": f"{source_name}.{key}",
                    "index": index,
                    "payload": item,
                    "artifact_class": "claim_repair_priority" if key == "claim_repair_priorities" else "evidence_gap_ledger",
                }
        synthesis = _as_dict(container.get("llm_analysis_synthesis"))
        for index, item in enumerate(_as_list(synthesis.get("evidence_repair_priorities")), start=1):
            if not isinstance(item, dict):
                continue
            if str(item.get("schema_version") or "") != "claim_support_repair_priority_v1":
                continue
            gap_key = _first_text(item.get("gap_id"), item.get("id")) or _json_hash(item)
            if gap_key in seen:
                continue
            seen.add(gap_key)
            yield {
                "source": f"{source_name}.llm_analysis_synthesis.evidence_repair_priorities",
                "index": index,
                "payload": item,
                "artifact_class": "claim_repair_priority",
            }


def _research_reflection_memo_from_payloads(
    writer_package: Dict[str, Any],
    writer_report: Dict[str, Any],
) -> Dict[str, Any]:
    package = _as_dict(writer_package)
    report = _as_dict(writer_report)
    package_structured = _as_dict(package.get("structured_analysis"))
    report_structured = _as_dict(report.get("structured_analysis"))
    package_insight = _as_dict(package.get("report_insight_package")) or _as_dict(
        package_structured.get("report_insight_package")
    )
    report_insight = _as_dict(report.get("report_insight_package")) or _as_dict(
        report_structured.get("report_insight_package")
    )
    render_artifacts = _as_dict(report.get("render_artifacts")) or _as_dict(package.get("render_artifacts"))
    render_structured = _as_dict(render_artifacts.get("structured_analysis"))
    evidence_package = _as_dict(package.get("evidence_package")) or _as_dict(report.get("evidence_package"))

    for candidate in (
        _as_dict(report.get("research_reflection_memo")),
        _as_dict(report_insight.get("research_reflection_memo")),
        _as_dict(report_structured.get("research_reflection_memo")),
        _as_dict(render_artifacts.get("research_reflection_memo")),
        _as_dict(render_structured.get("research_reflection_memo")),
        _as_dict(_as_dict(render_structured.get("report_insight_package")).get("research_reflection_memo")),
        _as_dict(package.get("research_reflection_memo")),
        _as_dict(package_insight.get("research_reflection_memo")),
        _as_dict(package_structured.get("research_reflection_memo")),
        _as_dict(evidence_package.get("research_reflection_memo")),
    ):
        if candidate and (
            _text(candidate.get("schema_version")).startswith("research_reflection_memo")
            or candidate.get("next_search_task_seeds") is not None
            or candidate.get("coverage_by_requirement") is not None
        ):
            return candidate
    return {}


def _research_reflection_requirement_ids(memo: Dict[str, Any]) -> List[str]:
    values: List[Any] = []
    for key in ("requirement_ids", "open_requirement_ids", "blocked_requirement_ids"):
        values.extend(_as_list(memo.get(key)))
    for item in _as_list(memo.get("coverage_by_requirement")):
        if isinstance(item, dict):
            values.append(item.get("requirement_id"))
    for seed in _as_list(memo.get("next_search_task_seeds")):
        if isinstance(seed, dict):
            values.append(seed.get("requirement_id"))
    return _dedupe(values, limit=200)


def _record_research_reflection_artifact(
    store: ArtifactStore,
    *,
    run_id: str,
    memo: Dict[str, Any],
    add_edge: Any,
) -> str:
    result = store.record_artifact(
        run_id=run_id,
        stage=RESEARCH_REFLECTION_STAGE,
        artifact_type=RESEARCH_REFLECTION_ARTIFACT_TYPE,
        payload=memo,
        status=_text(memo.get("status")) or "recorded",
        schema_version=_text(memo.get("schema_version")) or "research_reflection_memo_v1",
        output_hash=_json_hash(memo),
        lineage={
            "allowed_for_writing": bool(memo.get("allowed_for_writing")),
            "write_mode": _text(memo.get("write_mode")),
            "seed_count": len(_as_list(memo.get("next_search_task_seeds"))),
        },
    )
    for requirement_id in _research_reflection_requirement_ids(memo):
        add_edge("requirement", requirement_id, "artifact", result.artifact_id, "research_reflection")
    return result.artifact_id


def _iter_research_reflection_gap_payloads(
    writer_package: Dict[str, Any],
    writer_report: Dict[str, Any],
    *,
    research_reflection_artifact_id: str = "",
) -> Iterable[Dict[str, Any]]:
    memo = _research_reflection_memo_from_payloads(writer_package, writer_report)
    seen: Set[str] = set()
    for index, seed in enumerate(_as_list(memo.get("next_search_task_seeds")), start=1):
        if not isinstance(seed, dict):
            continue
        gap_key = _first_text(seed.get("gap_id"), seed.get("id")) or _json_hash(seed)
        if gap_key in seen:
            continue
        seen.add(gap_key)
        status = _first_text(
            seed.get("status"),
            seed.get("repair_status"),
            "live_search_required" if seed.get("live_refresh_required") else "",
        ) or "still_insufficient"
        yield {
            "source": f"{RESEARCH_REFLECTION_ARTIFACT_TYPE}.next_search_task_seeds",
            "index": index,
            "payload": {
                **seed,
                "source": _text(seed.get("source")) or RESEARCH_REFLECTION_ARTIFACT_TYPE,
                "status": status,
                "repair_status": _text(seed.get("repair_status")) or status,
                "allowed_for_writing": False,
            },
            "artifact_class": RESEARCH_REFLECTION_ARTIFACT_TYPE,
            "artifact_id": research_reflection_artifact_id,
        }


def _gap_text(payload: Any) -> str:
    item = _as_dict(payload)
    return _first_text(
        item.get("message"),
        item.get("reason"),
        item.get("suggested_fix"),
        item.get("finding"),
        item.get("description"),
        payload if isinstance(payload, str) else "",
    )


def _gap_type(payload: Any, source: str) -> str:
    item = _as_dict(payload)
    value = _first_text(
        item.get("gap_type"),
        item.get("type"),
        item.get("finding_category"),
        item.get("qa_category"),
        item.get("reason"),
    )
    if value:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")[:80] or "quality_gap"
    return source.replace(".", "_")


def _gap_requirement_id(item: Dict[str, Any]) -> str:
    return _first_text(
        item.get("requirement_id"),
        item.get("evidence_requirement_id"),
        item.get("mandatory_proof_id"),
    )


def _gap_missing_fields(item: Dict[str, Any], payload: Any) -> List[Any]:
    missing = _as_list(item.get("missing")) or _as_list(item.get("missing_fields"))
    if missing:
        return missing
    required_fields = _as_list(item.get("required_fields"))
    if required_fields:
        return required_fields
    text = _gap_text(payload)
    return [text] if text else []


def _gap_retry_plan(item: Dict[str, Any], *, source_stage: str) -> Dict[str, Any]:
    retry_plan = dict(_as_dict(item.get("retry_plan")))
    query_seed = _first_text(
        item.get("next_search_task"),
        item.get("query_seed"),
        item.get("query"),
        item.get("suggested_query"),
    )
    if query_seed:
        retry_plan.setdefault("next_search_task", query_seed)
    query_terms = _as_list(item.get("query_terms")) or _as_list(item.get("topic_terms"))
    if query_terms:
        retry_plan.setdefault("query_terms", query_terms)
        retry_plan.setdefault("query_seed", " ".join(_text(term) for term in query_terms if _text(term)))
    for key in (
        "schema_version",
        "proof_role",
        "required_proof_role",
        "required_source_level",
        "required_source_levels",
        "required_fields",
        "lane_targets",
        "current_evidence_refs",
        "blocking_gaps",
        "repair_route",
        "repair_priority",
        "can_iqs_repair",
        "affected_metric_fields",
        "root_cause",
        "success_criteria",
        "reject_if",
        "preferred_source_patterns",
        "freshness_required",
        "max_cache_age_hours",
        "avoid_repeating_failed_query",
        "live_refresh_required",
        "failed_queries",
        "repair_status",
        "source_priority",
    ):
        value = item.get(key)
        if value not in (None, "", [], {}):
            target_key = "proof_role" if key == "required_proof_role" else key
            if target_key == "required_source_levels":
                target_key = "required_source_level"
            retry_plan.setdefault(target_key, value)
    insufficiency = _first_text(
        item.get("why_current_evidence_insufficient"),
        item.get("targets_gap"),
        item.get("evidence_goal"),
        item.get("reason"),
    )
    if insufficiency:
        retry_plan.setdefault("current_insufficiency", insufficiency)
    retry_plan.setdefault("source_stage", source_stage)
    retry_plan.setdefault("allowed_for_writing", False)
    return retry_plan


def _stable_gap_id(item: Dict[str, Any], raw: Dict[str, Any], gap_type: str) -> str:
    explicit = _first_text(item.get("gap_id"), item.get("id"))
    if explicit:
        return explicit
    stable_payload = {
        "source": _text(raw.get("source")),
        "requirement_id": _gap_requirement_id(item),
        "chapter_id": _first_text(item.get("chapter_id")),
        "section_id": _first_text(item.get("section_id"), item.get("block_id")),
        "hypothesis_id": _first_text(item.get("hypothesis_id")),
        "gap_type": gap_type,
        "proof_role": _first_text(item.get("proof_role"), item.get("required_proof_role"), item.get("evidence_type")),
    }
    return _stable_id("GAP", stable_payload)


def _ingest_score_gaps(
    store: ArtifactStore,
    *,
    run_id: str,
    writer_package: Dict[str, Any],
    writer_report: Dict[str, Any],
    final_audit_result: Optional[Dict[str, Any]],
    requirement_lookup: Dict[str, Any],
    task_requirement_lookup: Dict[str, str],
    add_edge: Any,
    research_reflection_artifact_id: str = "",
) -> int:
    count = 0
    for raw in _iter_quality_gap_payloads(
        writer_package,
        writer_report,
        final_audit_result,
        research_reflection_artifact_id=research_reflection_artifact_id,
    ):
        payload = raw.get("payload")
        item = _as_dict(payload)
        gap_type = _gap_type(payload, _text(raw.get("source")))
        gap_id = _stable_gap_id(item, raw, gap_type)
        req_ids = _infer_requirement_ids(
            {**item, "gap_type": item.get("gap_type") or gap_type},
            requirement_lookup=requirement_lookup,
            task_requirement_lookup=task_requirement_lookup,
        )
        requirement_id = req_ids[0] if req_ids else _gap_requirement_id(item)
        section_id = _first_text(item.get("section_id"), item.get("block_id"))
        chapter_id = _first_text(item.get("chapter_id"))
        missing = _gap_missing_fields(item, payload)
        retry_plan = _gap_retry_plan(item, source_stage=_text(raw.get("artifact_class")) or _text(raw.get("source")) or "quality_gap")
        store.upsert_score_gap(
            run_id=run_id,
            gap_id=gap_id,
            requirement_id=requirement_id,
            chapter_id=chapter_id,
            section_id=section_id,
            gap_type=gap_type,
            severity=_first_text(item.get("severity"), item.get("level")),
            missing=missing,
            retry_plan=retry_plan,
            status=_first_text(item.get("status"), item.get("repair_status")) or "open",
        )
        if requirement_id:
            add_edge("requirement", requirement_id, "score_gap", gap_id, "gap")
        if section_id:
            add_edge("section", section_id, "score_gap", gap_id, "gap")
        artifact_id = _first_text(raw.get("artifact_id"), item.get("artifact_id"))
        if artifact_id:
            add_edge("artifact", artifact_id, "score_gap", gap_id, "suggests_repair")
        count += 1
    return count


STATUS_RANK = {
    "open": 0,
    "needs_repair": 0,
    "insufficient": 0,
    "pending": 0,
    "still_insufficient": 10,
    "live_search_required": 20,
    "cache_satisfied": 30,
    "evidence_found": 40,
}


def _iter_evidence_cache_gap_status_updates(
    writer_package: Dict[str, Any],
    writer_report: Dict[str, Any],
) -> Iterable[Dict[str, Any]]:
    containers = (
        _as_dict(writer_report),
        _as_dict(writer_report.get("metadata")),
        _as_dict(writer_package),
        _as_dict(writer_package.get("metadata")),
        _as_dict(writer_package.get("evidence_package")),
        _as_dict(_as_dict(writer_package.get("evidence_package")).get("metadata")),
    )
    seen: Set[str] = set()
    for container in containers:
        summary = _as_dict(container.get("evidence_cache_summary"))
        by_gap = _as_dict(summary.get("by_gap"))
        for gap_id, raw in by_gap.items():
            item = _as_dict(raw)
            normalized_gap_id = _text(gap_id)
            if not normalized_gap_id or normalized_gap_id in seen:
                continue
            seen.add(normalized_gap_id)
            cache_hit_count = _int_value(item.get("cache_hit_count"))
            live_required_count = _int_value(item.get("live_refresh_required_count"))
            cache_only_count = _int_value(item.get("cache_only_skip_count"))
            if cache_hit_count <= 0:
                continue
            status = "live_search_required" if live_required_count > 0 and cache_only_count <= 0 else "cache_satisfied"
            yield {
                "gap_id": normalized_gap_id,
                "status": status,
                "source": "evidence_cache_summary",
                "requirement_id": _first_text(item.get("requirement_id")),
                "cache_hit_count": cache_hit_count,
                "cache_only_skip_count": cache_only_count,
                "live_refresh_required_count": live_required_count,
            }


def _iter_repair_gap_ledger_status_updates(
    writer_package: Dict[str, Any],
    writer_report: Dict[str, Any],
) -> Iterable[Dict[str, Any]]:
    containers = (
        _as_dict(writer_report),
        _as_dict(writer_package),
    )
    for container in containers:
        for trace_key in ("evidence_preflight_trace", "layout_refinement_trace", "post_qa_repair_trace", "loop_trace"):
            for trace_index, trace in enumerate(_as_list(container.get(trace_key)), start=1):
                trace_payload = _as_dict(trace)
                for item in _as_list(trace_payload.get("gap_ledger")):
                    payload = _as_dict(item)
                    gap_id = _first_text(payload.get("gap_id"))
                    raw_status = _text(payload.get("status"))
                    if not gap_id or raw_status not in {"evidence_found", "searched_no_signal"}:
                        continue
                    yield {
                        "gap_id": gap_id,
                        "status": "evidence_found" if raw_status == "evidence_found" else "still_insufficient",
                        "source": "repair_gap_ledger",
                        "trace_key": trace_key,
                        "trace_index": trace_index,
                        "signal_count": _int_value(payload.get("signal_count")),
                        "result_count": _int_value(payload.get("result_count")),
                        "max_signal_score": _int_value(payload.get("max_signal_score")),
                    }


def _score_gap_locked_by_final_audit(gap: Dict[str, Any]) -> bool:
    retry_plan = _as_dict(gap.get("retry_plan"))
    source_stage = _text(retry_plan.get("source_stage"))
    return source_stage.startswith("final_audit.")


def _apply_score_gap_status_updates(
    store: ArtifactStore,
    *,
    run_id: str,
    writer_package: Dict[str, Any],
    writer_report: Dict[str, Any],
) -> int:
    updates: Dict[str, Dict[str, Any]] = {}
    for update in (
        *_iter_evidence_cache_gap_status_updates(writer_package, writer_report),
        *_iter_repair_gap_ledger_status_updates(writer_package, writer_report),
    ):
        gap_id = _text(update.get("gap_id"))
        status = _text(update.get("status"))
        if not gap_id or not status:
            continue
        existing = updates.get(gap_id)
        if existing and STATUS_RANK.get(_text(existing.get("status")), 0) > STATUS_RANK.get(status, 0):
            continue
        updates[gap_id] = update

    count = 0
    for gap_id, update in updates.items():
        rows = store.list_score_gaps(run_id, gap_id=gap_id)
        if not rows:
            continue
        gap = rows[0]
        if _score_gap_locked_by_final_audit(gap):
            continue
        current_status = _text(gap.get("status")) or "open"
        next_status = _text(update.get("status"))
        if STATUS_RANK.get(next_status, 0) < STATUS_RANK.get(current_status, 0):
            continue
        retry_plan = {
            **_as_dict(gap.get("retry_plan")),
            "repair_status_source": _text(update.get("source")),
        }
        for key in (
            "cache_hit_count",
            "cache_only_skip_count",
            "live_refresh_required_count",
            "signal_count",
            "result_count",
            "max_signal_score",
            "trace_key",
            "trace_index",
        ):
            value = update.get(key)
            if value not in (None, "", [], {}):
                retry_plan[key] = value
        store.upsert_score_gap(
            run_id=run_id,
            gap_id=gap_id,
            requirement_id=_first_text(gap.get("requirement_id"), update.get("requirement_id")),
            chapter_id=_text(gap.get("chapter_id")),
            section_id=_text(gap.get("section_id")),
            gap_type=_text(gap.get("gap_type")) or "quality_gap",
            severity=_text(gap.get("severity")),
            missing=_as_list(gap.get("missing")),
            retry_plan=retry_plan,
            status=next_status,
        )
        if next_status != current_status:
            count += 1
    return count
