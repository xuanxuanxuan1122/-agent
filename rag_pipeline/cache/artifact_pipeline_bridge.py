from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

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
            *_as_list(payload.get("requirement_ids")),
            *_as_list(payload.get("evidence_requirement_ids")),
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


def _iter_search_tasks(writer_package: Dict[str, Any], writer_report: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for container in (
        _as_dict(writer_package.get("evidence_package")),
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
        "lineage_edge_count": 0,
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
        store.add_lineage_edge(run_id, from_type, source_id, to_type, target_id, relation)
        summary["lineage_edge_count"] += 1

    for requirement in _iter_requirements(writer_package):
        requirement_id = _first_text(requirement.get("requirement_id"), requirement.get("evidence_requirement_id"))
        if not requirement_id:
            continue
        store.upsert_evidence_requirement(
            run_id=run_id,
            requirement_id=requirement_id,
            chapter_id=_text(requirement.get("chapter_id")),
            hypothesis_id=_text(requirement.get("hypothesis_id")),
            proof_role=_text(requirement.get("proof_role")) or "support",
            required_fields=_as_list(requirement.get("required_fields")),
            min_source_level=requirement.get("min_source_level") or requirement.get("source_level"),
            claim_strength_ceiling=_text(requirement.get("claim_strength_ceiling")),
            freshness_required=bool(requirement.get("freshness_required")),
            max_cache_age_hours=requirement.get("max_cache_age_hours"),
            status=_text(requirement.get("status")) or "open",
            missing=_as_list(requirement.get("missing")),
        )
        summary["requirement_count"] += 1

    for index, source in enumerate(_as_list(writer_package.get("source_registry")), start=1):
        if not isinstance(source, dict):
            continue
        run_source_id = _source_run_ref(source, index)
        store.upsert_source(run_id=run_id, run_source_id=run_source_id, source=source)
        summary["source_count"] += 1

    for index, task in enumerate(_iter_search_tasks(writer_package, writer_report), start=1):
        task_id = _first_text(task.get("task_id"), task.get("search_task_id"), task.get("id")) or f"ST-{index:04d}"
        req_id = _first_text(task.get("requirement_id"), task.get("evidence_requirement_id"))
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
        req_ids = _requirement_ids(item)
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
    for index, claim in enumerate(_iter_claim_units(writer_package), start=1):
        claim_id = _first_text(claim.get("claim_id"), claim.get("id")) or f"CL-{index:04d}"
        if claim_id in seen_claims:
            continue
        seen_claims.add(claim_id)
        req_ids = _requirement_ids(claim)
        fact_ids = _fact_refs(claim)
        store.upsert_claim_unit(
            run_id=run_id,
            claim_id=claim_id,
            payload=claim,
            requirement_ids=req_ids,
            fact_ids=fact_ids,
            source_ids=_source_ids(claim),
            status=_text(claim.get("status")) or "validated",
        )
        for req_id in req_ids:
            add_edge("requirement", req_id, "claim_unit", claim_id, "constrains")
        for fact_id in fact_ids:
            add_edge("fact_card", fact_id, "claim_unit", claim_id, "supports")
        summary["claim_unit_count"] += 1

    seen_sections: Set[str] = set()
    claim_req_lookup = {}
    for claim in _iter_claim_units(writer_package):
        claim_id = _first_text(claim.get("claim_id"), claim.get("id"))
        if claim_id:
            claim_req_lookup[claim_id] = _requirement_ids(claim)
    for section in _iter_sections(writer_package):
        section_id = _first_text(section.get("section_id"), section.get("id"), section.get("claim_id"))
        if not section_id:
            section_id = f"SEC-{_text(section.get('chapter_id')) or 'chapter'}-{int(section.get('_section_index') or 0):02d}"
        if section_id in seen_sections:
            continue
        seen_sections.add(section_id)
        claim_ids = _dedupe([section.get("claim_id"), *_as_list(section.get("claim_ids"))], limit=40)
        req_ids = _requirement_ids(section)
        for claim_id in claim_ids:
            req_ids = _dedupe([*req_ids, *claim_req_lookup.get(claim_id, [])], limit=80)
        fact_ids = _fact_refs(section)
        store.upsert_section(
            run_id=run_id,
            section_id=section_id,
            payload=section,
            requirement_ids=req_ids,
            claim_ids=claim_ids,
            used_fact_refs=fact_ids,
            evidence_backed=bool(section.get("evidence_backed") or fact_ids),
            status=_text(section.get("status")) or "validated",
        )
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
        add_edge=add_edge,
    )
    summary["score_gap_count"] += gap_count
    return summary


def _iter_quality_gap_payloads(
    writer_package: Dict[str, Any],
    writer_report: Dict[str, Any],
    final_audit_result: Optional[Dict[str, Any]],
) -> Iterable[Dict[str, Any]]:
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


def _ingest_score_gaps(
    store: ArtifactStore,
    *,
    run_id: str,
    writer_package: Dict[str, Any],
    writer_report: Dict[str, Any],
    final_audit_result: Optional[Dict[str, Any]],
    add_edge: Any,
) -> int:
    count = 0
    for raw in _iter_quality_gap_payloads(writer_package, writer_report, final_audit_result):
        payload = raw.get("payload")
        item = _as_dict(payload)
        gap_type = _gap_type(payload, _text(raw.get("source")))
        gap_id = _first_text(item.get("gap_id"), item.get("id")) or _stable_id("GAP", raw)
        requirement_id = _first_text(item.get("requirement_id"), item.get("evidence_requirement_id"))
        section_id = _first_text(item.get("section_id"), item.get("block_id"))
        chapter_id = _first_text(item.get("chapter_id"))
        missing = _as_list(item.get("missing")) or _as_list(item.get("missing_fields"))
        if not missing:
            text = _gap_text(payload)
            missing = [text] if text else []
        retry_plan = _as_dict(item.get("retry_plan"))
        if not retry_plan and _first_text(item.get("next_search_task"), item.get("query_seed")):
            retry_plan = {
                "next_search_task": _first_text(item.get("next_search_task"), item.get("query_seed")),
            }
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
            status=_text(item.get("status")) or "open",
        )
        if requirement_id:
            add_edge("requirement", requirement_id, "score_gap", gap_id, "gap")
        if section_id:
            add_edge("section", section_id, "score_gap", gap_id, "gap")
        count += 1
    return count
