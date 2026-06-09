from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Sequence, Set

from rag_pipeline.contracts.evidence_identity import build_evidence_alias_map, canonical_evidence_id
from rag_pipeline.contracts.metric_asset import build_metric_assets
from rag_pipeline.contracts.ref_normalizer import normalize_claim_refs
from rag_pipeline.runtime_cache import json_safe_default


QUALITY_CONVERSION_SUMMARY_VERSION = "quality_conversion_summary_v1"

EVIDENCE_STAGE_KEYS = {
    "normalized": "normalized_evidence",
    "clean": "clean_evidence_list",
    "analysis_ready": "analysis_ready_evidence",
    "core": "core_evidence",
    "supporting": "supporting_evidence",
    "clue": "clue_evidence",
    "appendix": "appendix_evidence",
    "rejected_sample": "rejected_evidence_sample",
}

STRICT_CLOSED_GAP_STATUSES = {
    "closed",
    "resolved",
    "repaired",
    "sufficiency_passed",
    "claim_bound",
    "section_rewritten",
    "citation_resolved",
    "metric_table_ready",
}
SIGNAL_ONLY_GAP_STATUSES = {"evidence_found", "cache_satisfied"}
OPEN_GAP_STATUSES = {"open", "needs_repair", "insufficient", "still_insufficient", "live_search_required"}


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if value in (None, ""):
        return []
    return [value]


def _text(value: Any) -> str:
    return str(value or "").strip()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return default


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _stable_hash(value: Any, *, length: int = 14) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=json_safe_default)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def _dedupe(values: Iterable[Any], *, limit: int = 500) -> List[str]:
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


def _first_dict(*values: Any) -> Dict[str, Any]:
    for value in values:
        payload = _as_dict(value)
        if payload:
            return payload
    return {}


def _source_registry(package: Dict[str, Any], report: Dict[str, Any], evidence_package: Dict[str, Any]) -> List[Dict[str, Any]]:
    render_artifacts = _as_dict(report.get("render_artifacts"))
    candidates = (
        _as_list(package.get("source_registry"))
        or _as_list(report.get("source_registry"))
        or _as_list(render_artifacts.get("source_registry"))
        or _as_list(evidence_package.get("source_registry"))
        or _as_list(evidence_package.get("sources"))
    )
    return [item for item in candidates if isinstance(item, dict)]


def _evidence_id(item: Dict[str, Any]) -> str:
    return canonical_evidence_id(item)


def _stage_items(evidence_package: Dict[str, Any], stage: str) -> List[Dict[str, Any]]:
    return [item for item in _as_list(evidence_package.get(stage)) if isinstance(item, dict)]


def _distinct_evidence(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ids: Set[str] = set()
    missing = 0
    for item in items:
        evidence_id = _evidence_id(item)
        if evidence_id:
            ids.add(evidence_id)
        else:
            missing += 1
    return {"ids": ids, "missing": missing, "count": len(ids) + missing}


def _increment_slice(bucket: Dict[str, Dict[str, Any]], key: str, field: str, amount: int = 1) -> None:
    if not key:
        return
    payload = bucket.setdefault(key, {})
    payload[field] = _safe_int(payload.get(field)) + amount


def _evidence_funnel(evidence_package: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"schema_version": "evidence_conversion_funnel_v1"}
    stage_counts: Dict[str, int] = {}
    missing_counts: Dict[str, int] = {}
    for label, key in EVIDENCE_STAGE_KEYS.items():
        distinct = _distinct_evidence(_stage_items(evidence_package, key))
        stage_counts[label] = int(distinct["count"])
        missing_counts[label] = int(distinct["missing"])
        summary[f"{label}_count"] = stage_counts[label]
        summary[f"{label}_missing_canonical_id_count"] = missing_counts[label]
    denominator = stage_counts.get("clean") or stage_counts.get("normalized") or 0
    summary["analysis_ready_rate"] = _ratio(stage_counts.get("analysis_ready", 0), denominator)
    summary["core_rate"] = _ratio(stage_counts.get("core", 0), denominator)
    return summary


def _all_fact_cards(evidence_package: Dict[str, Any], package: Dict[str, Any], structured_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    cards: List[Dict[str, Any]] = []
    for key in (
        "analysis_ready_evidence",
        "fact_cards",
        "fact_card_pool",
        "evidence_cards",
        "clean_evidence_list",
        "normalized_evidence",
        "core_evidence",
        "supporting_evidence",
    ):
        cards.extend(item for item in _as_list(evidence_package.get(key)) if isinstance(item, dict))
    cards.extend(item for item in _as_list(package.get("fact_cards")) if isinstance(item, dict))
    cards.extend(item for item in _as_list(structured_analysis.get("core_facts")) if isinstance(item, dict))
    seen: Set[str] = set()
    unique: List[Dict[str, Any]] = []
    for item in cards:
        key = _evidence_id(item) or f"missing-{_stable_hash(item)}"
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _claim_id(claim: Dict[str, Any], index: int) -> str:
    return _text(claim.get("claim_id") or claim.get("id")) or f"CL-{_stable_hash([index, claim.get('claim') or claim.get('text')])}"


def _collect_claims(structured_analysis: Dict[str, Any], package: Dict[str, Any], report: Dict[str, Any]) -> List[Dict[str, Any]]:
    render_artifacts = _as_dict(report.get("render_artifacts"))
    raw_claims: List[Dict[str, Any]] = []
    raw_claims.extend(item for item in _as_list(structured_analysis.get("claim_units")) if isinstance(item, dict))
    raw_claims.extend(item for item in _as_list(package.get("argument_units")) if isinstance(item, dict))
    raw_claims.extend(item for item in _as_list(render_artifacts.get("argument_units")) if isinstance(item, dict))
    seen: Set[str] = set()
    claims: List[Dict[str, Any]] = []
    for index, claim in enumerate(raw_claims, start=1):
        claim_key = _claim_id(claim, index)
        if claim_key in seen:
            continue
        seen.add(claim_key)
        claims.append({**claim, "claim_id": claim_key})
    return claims


def _collect_sections(package: Dict[str, Any], report: Dict[str, Any]) -> List[Dict[str, Any]]:
    render_artifacts = _as_dict(report.get("render_artifacts"))
    containers = [
        *_as_list(package.get("chapter_packages")),
        *_as_list(render_artifacts.get("chapter_packages")),
        *_as_list(package.get("micro_layouts")),
        *_as_list(render_artifacts.get("micro_layouts")),
    ]
    sections: List[Dict[str, Any]] = []
    for container in containers:
        payload = _as_dict(container)
        for section in _as_list(payload.get("sections")):
            if isinstance(section, dict):
                sections.append({**section, "chapter_id": _text(section.get("chapter_id") or payload.get("chapter_id"))})
    return sections


def _section_bindings(sections: Sequence[Dict[str, Any]], fact_cards: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    claim_ids: Set[str] = set()
    fact_ids: Set[str] = set()
    for section in sections:
        claim_ids.update(_text(item) for item in [section.get("claim_id"), *_as_list(section.get("claim_ids"))] if _text(item))
        normalized = normalize_claim_refs(section, fact_cards=fact_cards)
        fact_ids.update(_as_list(normalized.get("fact_ids")))
    return {"claim_ids": claim_ids, "fact_ids": fact_ids}


def _claim_forbidden(claim: Dict[str, Any]) -> bool:
    values = {
        _text(claim.get("allowed_use")).lower(),
        _text(claim.get("writing_permission")).lower(),
        _text(claim.get("status")).lower(),
        _text(claim.get("claim_strength")).lower(),
    }
    return bool(
        values
        & {
            "not_allowed_until_repaired",
            "repair_needed",
            "rejected",
            "stale",
            "superseded",
            "diagnostic_only",
            "clue",
            "appendix_only",
            "context_only",
        }
    )


def _claim_funnel(
    structured_analysis: Dict[str, Any],
    package: Dict[str, Any],
    report: Dict[str, Any],
    fact_cards: Sequence[Dict[str, Any]],
    by_chapter: Dict[str, Dict[str, Any]],
    by_requirement: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    claims = _collect_claims(structured_analysis, package, report)
    section_binding = _section_bindings(_collect_sections(package, report), fact_cards)
    counts = Counter()
    alias_map = build_evidence_alias_map(fact_cards)
    for claim in claims:
        normalized = normalize_claim_refs(claim, alias_map=alias_map, fact_cards=fact_cards)
        fact_ids = _as_list(normalized.get("fact_ids"))
        source_ids = _as_list(normalized.get("source_ids"))
        requirement_ids = _as_list(normalized.get("requirement_ids"))
        chapter_id = _text(claim.get("chapter_id"))
        bound = bool(fact_ids and source_ids and requirement_ids)
        renderable = bool(bound and not _claim_forbidden({**claim, **normalized}))
        section_bound = bool(
            renderable
            and (
                _text(claim.get("claim_id")) in section_binding["claim_ids"]
                or bool(set(fact_ids) & set(section_binding["fact_ids"]))
            )
        )
        counts["total_claim_count"] += 1
        counts["bound_claim_count"] += int(bound)
        counts["unbound_claim_count"] += int(not bound)
        counts["renderable_claim_count"] += int(renderable)
        counts["section_bound_claim_count"] += int(section_bound)
        counts["diagnostic_only_claim_count"] += int(bound and not renderable)
        counts["downgraded_claim_count"] += int(bool(claim.get("downgraded") or claim.get("claim_strength_downgraded") or claim.get("metric_missing_fields")))
        for req_id in requirement_ids:
            _increment_slice(by_requirement, req_id, "claim_count")
            if bound:
                _increment_slice(by_requirement, req_id, "bound_claim_count")
            if renderable:
                _increment_slice(by_requirement, req_id, "renderable_claim_count")
        if chapter_id:
            _increment_slice(by_chapter, chapter_id, "claim_count")
            if bound:
                _increment_slice(by_chapter, chapter_id, "bound_claim_count")
            if renderable:
                _increment_slice(by_chapter, chapter_id, "renderable_claim_count")
    total = counts["total_claim_count"]
    counts["bound_claim_rate"] = _ratio(counts["bound_claim_count"], total)
    counts["renderable_claim_rate"] = _ratio(counts["renderable_claim_count"], total)
    counts["section_bound_claim_rate"] = _ratio(counts["section_bound_claim_count"], counts["renderable_claim_count"])
    return {"schema_version": "claim_conversion_funnel_v1", **dict(counts)}


def _metric_funnel(
    fact_cards: Sequence[Dict[str, Any]],
    source_registry: Sequence[Dict[str, Any]],
    by_chapter: Dict[str, Dict[str, Any]],
    by_requirement: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    assets = build_metric_assets(fact_cards, source_registry)
    missing_counts: Counter[str] = Counter()
    complete = 0
    table_ready = 0
    for asset in assets:
        missing = _as_list(asset.get("missing_fields"))
        missing_counts.update(missing)
        complete += int(bool(asset.get("complete")))
        table_ready += int(bool(asset.get("table_ready")))
        req_id = _text(asset.get("requirement_id"))
        chapter_id = _text(asset.get("chapter_id"))
        if req_id:
            _increment_slice(by_requirement, req_id, "metric_candidate_count")
            if asset.get("complete"):
                _increment_slice(by_requirement, req_id, "complete_metric_count")
            if asset.get("table_ready"):
                _increment_slice(by_requirement, req_id, "table_ready_metric_count")
        if chapter_id:
            _increment_slice(by_chapter, chapter_id, "metric_candidate_count")
            if asset.get("complete"):
                _increment_slice(by_chapter, chapter_id, "complete_metric_count")
            if asset.get("table_ready"):
                _increment_slice(by_chapter, chapter_id, "table_ready_metric_count")
    return {
        "schema_version": "metric_conversion_funnel_v1",
        "metric_candidate_count": len(assets),
        "complete_metric_count": complete,
        "table_ready_metric_count": table_ready,
        "metric_completion_rate": _ratio(complete, len(assets)),
        "metric_to_table_ready_rate": _ratio(table_ready, len(assets)),
        "metric_missing_field_counts": dict(missing_counts),
        "metric_assets": assets[:50],
    }


def _citation_audit(package: Dict[str, Any], report: Dict[str, Any]) -> Dict[str, Any]:
    render_artifacts = _as_dict(report.get("render_artifacts"))
    return _first_dict(
        package.get("final_citation_audit"),
        report.get("final_citation_audit"),
        render_artifacts.get("final_citation_audit"),
        _as_dict(package.get("writer_report")).get("final_citation_audit"),
    )


def _citation_funnel(package: Dict[str, Any], report: Dict[str, Any], renderable_claim_count: int) -> Dict[str, Any]:
    audit = _citation_audit(package, report)
    citationless = _safe_int(audit.get("citationless_factual_removed_count")) or (
        _safe_int(audit.get("citationless_factual_sentence_removed_count"))
        + _safe_int(audit.get("citationless_factual_bullet_removed_count"))
        + _safe_int(audit.get("citationless_short_factual_line_removed_count"))
    )
    unresolved_removed = _safe_int(audit.get("final_unresolved_citation_removed_count"))
    unsupported_removed = _safe_int(audit.get("unsupported_claim_removed_count"))
    public_leak_removed = _safe_int(audit.get("public_narrative_leak_removed_count"))
    final_deleted = citationless + unresolved_removed + unsupported_removed + public_leak_removed
    return {
        "schema_version": "citation_conversion_funnel_v1",
        "citationless_factual_removed_count": citationless,
        "final_unresolved_citation_removed_count": unresolved_removed,
        "unsupported_claim_removed_count": unsupported_removed,
        "public_narrative_leak_removed_count": public_leak_removed,
        "final_deleted_fact_count": final_deleted,
        "citation_deletion_rate": _ratio(final_deleted, final_deleted + renderable_claim_count),
        "citation_rebind_required": bool(audit.get("citation_rebind_required")),
        "final_citation_reconciliation_status": _text(audit.get("final_citation_reconciliation_status") or audit.get("status")),
    }


def _score_gaps(package: Dict[str, Any], report: Dict[str, Any], evidence_package: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        item
        for item in [
            *_as_list(package.get("score_gaps")),
            *_as_list(report.get("score_gaps")),
            *_as_list(evidence_package.get("evidence_gap_ledger")),
            *_as_list(_as_dict(package.get("structured_analysis")).get("evidence_gap_ledger")),
        ]
        if isinstance(item, dict)
    ]


def _repair_funnel(package: Dict[str, Any], report: Dict[str, Any], evidence_package: Dict[str, Any]) -> Dict[str, Any]:
    gaps = _score_gaps(package, report, evidence_package)
    status_counts = Counter(_text(gap.get("status")).lower() or "unknown" for gap in gaps)
    strict_closed = sum(status_counts[status] for status in STRICT_CLOSED_GAP_STATUSES)
    signal_only = sum(status_counts[status] for status in SIGNAL_ONLY_GAP_STATUSES)
    open_count = sum(status_counts[status] for status in OPEN_GAP_STATUSES)
    attempted = len(gaps)
    return {
        "schema_version": "repair_conversion_funnel_v1",
        "attempted_gap_count": attempted,
        "strict_closed_gap_count": strict_closed,
        "signal_only_gap_count": signal_only,
        "open_gap_count": open_count,
        "strict_closure_rate": _ratio(strict_closed, attempted),
        "by_gap_status": dict(status_counts),
    }


def _evidence_slices(evidence_package: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    by_chapter: Dict[str, Dict[str, Any]] = defaultdict(dict)
    by_requirement: Dict[str, Dict[str, Any]] = defaultdict(dict)
    for item in _stage_items(evidence_package, "analysis_ready_evidence"):
        req_id = _text(item.get("requirement_id") or (_as_list(item.get("requirement_ids")) or [""])[0])
        chapter_id = _text(item.get("chapter_id"))
        proof_role = _text(item.get("proof_role") or item.get("analysis_role")) or "unknown"
        if req_id:
            _increment_slice(by_requirement, req_id, "analysis_ready_count")
            roles = by_requirement[req_id].setdefault("by_proof_role", {})
            roles[proof_role] = _safe_int(roles.get(proof_role)) + 1
        if chapter_id:
            _increment_slice(by_chapter, chapter_id, "analysis_ready_count")
            roles = by_chapter[chapter_id].setdefault("by_proof_role", {})
            roles[proof_role] = _safe_int(roles.get(proof_role)) + 1
    return {"by_chapter": by_chapter, "by_requirement": by_requirement}


def _recommendations(
    evidence_funnel: Dict[str, Any],
    claim_funnel: Dict[str, Any],
    metric_funnel: Dict[str, Any],
    citation_funnel: Dict[str, Any],
    repair_funnel: Dict[str, Any],
    by_requirement: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    recommendations: List[Dict[str, Any]] = []
    if float(evidence_funnel.get("analysis_ready_rate") or 0.0) < 0.18:
        recommendations.append(
            {
                "type": "low_analysis_ready_rate",
                "severity": "medium",
                "action": "improve_evidence_admission_or_field_completion",
            }
        )
    if float(claim_funnel.get("bound_claim_rate") or 0.0) < 0.8:
        recommendations.append(
            {
                "type": "low_bound_claim_rate",
                "severity": "high",
                "action": "targeted_claim_rebuild",
                "affected_requirements": [
                    req_id
                    for req_id, payload in by_requirement.items()
                    if _safe_int(payload.get("claim_count")) and _safe_int(payload.get("bound_claim_count")) < _safe_int(payload.get("claim_count"))
                ][:20],
            }
        )
    if _safe_int(metric_funnel.get("complete_metric_count")) < 8 and _safe_int(metric_funnel.get("metric_candidate_count")):
        recommendations.append(
            {
                "type": "low_complete_metric_count",
                "severity": "high",
                "action": "metric_repair_search",
                "missing_fields": dict(_as_dict(metric_funnel.get("metric_missing_field_counts"))),
            }
        )
    if float(citation_funnel.get("citation_deletion_rate") or 0.0) > 0.05 or citation_funnel.get("citation_rebind_required"):
        recommendations.append(
            {
                "type": "high_citation_deletion_rate",
                "severity": "high",
                "action": "citation_rebind_or_section_rewrite",
            }
        )
    if _safe_int(repair_funnel.get("attempted_gap_count")) and float(repair_funnel.get("strict_closure_rate") or 0.0) < 0.4:
        recommendations.append(
            {
                "type": "low_repair_strict_closure_rate",
                "severity": "medium",
                "action": "repair_dispatcher_tuning",
            }
        )
    if not recommendations:
        recommendations.append({"type": "conversion_stable", "severity": "info", "action": "keep_monitoring"})
    return recommendations


def build_quality_conversion_summary(
    *,
    evidence_package: Dict[str, Any] | None = None,
    structured_analysis: Dict[str, Any] | None = None,
    writer_report: Dict[str, Any] | None = None,
    writer_package: Dict[str, Any] | None = None,
    artifact_store: Any = None,
    run_id: str = "",
) -> Dict[str, Any]:
    del artifact_store, run_id  # ledger-mode is intentionally deferred for v1.
    package = _as_dict(writer_package)
    report = _as_dict(writer_report) or _as_dict(package.get("writer_report"))
    render_artifacts = _as_dict(report.get("render_artifacts"))
    evidence = (
        _as_dict(evidence_package)
        or _as_dict(package.get("evidence_package"))
        or _as_dict(render_artifacts.get("evidence_package"))
    )
    analysis = (
        _as_dict(structured_analysis)
        or _as_dict(package.get("structured_analysis"))
        or _as_dict(render_artifacts.get("structured_analysis"))
    )
    source_registry = _source_registry(package, report, evidence)
    fact_cards = _all_fact_cards(evidence, package, analysis)
    slices = _evidence_slices(evidence)
    by_chapter: Dict[str, Dict[str, Any]] = defaultdict(dict, slices["by_chapter"])
    by_requirement: Dict[str, Dict[str, Any]] = defaultdict(dict, slices["by_requirement"])

    evidence_funnel = _evidence_funnel(evidence)
    claim_funnel = _claim_funnel(analysis, package, report, fact_cards, by_chapter, by_requirement)
    metric_funnel = _metric_funnel(fact_cards, source_registry, by_chapter, by_requirement)
    citation_funnel = _citation_funnel(package, report, _safe_int(claim_funnel.get("renderable_claim_count")))
    repair_funnel = _repair_funnel(package, report, evidence)
    recommendations = _recommendations(
        evidence_funnel,
        claim_funnel,
        metric_funnel,
        citation_funnel,
        repair_funnel,
        by_requirement,
    )

    return {
        "schema_version": QUALITY_CONVERSION_SUMMARY_VERSION,
        "mode": "package",
        "evidence_funnel": evidence_funnel,
        "claim_funnel": claim_funnel,
        "metric_funnel": metric_funnel,
        "citation_funnel": citation_funnel,
        "repair_funnel": repair_funnel,
        "by_chapter": {key: dict(value) for key, value in by_chapter.items()},
        "by_requirement": {key: dict(value) for key, value in by_requirement.items()},
        "recommendations": recommendations,
    }
