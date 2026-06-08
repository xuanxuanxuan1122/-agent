"""Deterministic research reflection memo.

The memo is a diagnostic artifact: it helps the pipeline decide what is known,
what remains weak, and what the next search task should target. It must not be
treated as quoteable evidence by the writer.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence

from .repair_dispatcher import dispatch_repair_seed


SCHEMA_VERSION = "research_reflection_memo_v1"


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    if value in (None, ""):
        return []
    return [value]


def _text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _dedupe(values: Iterable[Any], *, limit: int = 20) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        text = _text(value)
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def _compact(value: Any, max_chars: int = 240) -> str:
    text = _text(value)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


def _health(payload: Dict[str, Any]) -> Dict[str, Any]:
    summary = _as_dict(payload.get("summary"))
    metadata = _as_dict(payload.get("metadata"))
    return (
        _as_dict(payload.get("evidence_health_summary"))
        or _as_dict(summary.get("evidence_health_summary"))
        or _as_dict(metadata.get("evidence_health_summary"))
    )


def _gap_items(*payloads: Dict[str, Any]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for payload in payloads:
        for key in (
            "evidence_gap_ledger",
            "score_gaps",
            "score_gap_ledger",
            "missing_proof_standards",
            "required_followups",
        ):
            result.extend(item for item in _as_list(_as_dict(payload).get(key)) if isinstance(item, dict))
        deep_eval = _as_dict(_as_dict(payload).get("deep_evaluation"))
        result.extend(item for item in _as_list(deep_eval.get("required_followups")) if isinstance(item, dict))
    seen = set()
    unique: List[Dict[str, Any]] = []
    for index, item in enumerate(result, start=1):
        key = (
            _text(item.get("gap_id")),
            _text(item.get("requirement_id")),
            _text(item.get("gap_type") or item.get("type")),
            _text(item.get("chapter_id")),
            _text(item.get("section_id")),
        )
        if not any(key):
            key = (f"gap-{index}", "", "", "", "")
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _fact_cards(payload: Dict[str, Any], *, limit: int = 12) -> List[Dict[str, Any]]:
    cards: List[Dict[str, Any]] = []
    for key in (
        "analysis_ready_evidence",
        "validated_fact_cards",
        "fact_cards",
        "clean_fact_cards",
        "core_evidence",
        "public_fact_cards",
    ):
        cards.extend(item for item in _as_list(payload.get(key)) if isinstance(item, dict))
    seen = set()
    result: List[Dict[str, Any]] = []
    for card in cards:
        ref = _text(card.get("evidence_id") or card.get("fact_id") or card.get("ref") or card.get("id"))
        if not ref:
            continue
        if ref in seen:
            continue
        seen.add(ref)
        result.append(card)
        if len(result) >= limit:
            break
    return result


def _finding_refs(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    for card in _fact_cards(payload):
        lineage = _as_dict(card.get("lineage"))
        findings.append(
            {
                "evidence_id": _text(card.get("evidence_id") or card.get("fact_id") or card.get("ref") or card.get("id")),
                "requirement_id": _text(card.get("requirement_id") or lineage.get("requirement_id")),
                "chapter_id": _text(card.get("chapter_id") or lineage.get("chapter_id")),
                "source_id": _text(card.get("source_id") or lineage.get("source_id")),
                "source_level": _text(card.get("source_level")).upper(),
                "proof_role": _text(card.get("proof_role") or card.get("fact_type") or card.get("analysis_role")),
                "allowed_use": _text(card.get("allowed_use") or card.get("evidence_use_level")),
                "allowed_for_writing": False,
            }
        )
    return findings


def _coverage_from_gaps(gaps: Sequence[Dict[str, Any]], coverage_matrix: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_requirement: Dict[str, Dict[str, Any]] = {}
    for row in coverage_matrix:
        if not isinstance(row, dict):
            continue
        req_id = _text(row.get("requirement_id") or row.get("goal_id") or row.get("hypothesis_id"))
        if not req_id:
            continue
        by_requirement[req_id] = {
            "requirement_id": req_id,
            "status": "blocked" if _as_list(row.get("blocking_gaps")) else ("ready" if row.get("decision_ready") else "partial"),
            "proof_role": _text(row.get("proof_role")),
            "gap_ids": [],
            "blocking_gaps": _dedupe(row.get("blocking_gaps") or []),
            "missing_fields": [],
            "current_evidence_refs": _dedupe(row.get("current_evidence_refs") or row.get("evidence_refs") or []),
        }
    for gap in gaps:
        req_id = _text(gap.get("requirement_id") or gap.get("goal_id") or gap.get("hypothesis_id"))
        if not req_id:
            continue
        retry_plan = _as_dict(gap.get("retry_plan"))
        item = by_requirement.setdefault(
            req_id,
            {
                "requirement_id": req_id,
                "status": "partial",
                "proof_role": _text(gap.get("proof_role") or retry_plan.get("proof_role")),
                "gap_ids": [],
                "blocking_gaps": [],
                "missing_fields": [],
                "current_evidence_refs": [],
            },
        )
        gap_id = _text(gap.get("gap_id"))
        if gap_id:
            item["gap_ids"] = _dedupe([*item["gap_ids"], gap_id])
        gap_type = _text(gap.get("gap_type") or gap.get("type"))
        if gap_type:
            item["blocking_gaps"] = _dedupe([*item["blocking_gaps"], gap_type])
        missing = _as_list(gap.get("missing") or retry_plan.get("required_fields"))
        item["missing_fields"] = _dedupe([*item["missing_fields"], *missing])
        refs = _as_list(retry_plan.get("current_evidence_refs") or gap.get("current_evidence_refs"))
        item["current_evidence_refs"] = _dedupe([*item["current_evidence_refs"], *refs])
        severity = _text(gap.get("severity")).lower()
        status = _text(gap.get("status")).lower()
        if severity in {"fatal", "blocking", "high"} or status in {"still_insufficient", "live_search_required", "needs_repair"}:
            item["status"] = "blocked"
    return sorted(by_requirement.values(), key=lambda item: item.get("requirement_id") or "")


def _source_patterns(proof_role: str, lane_targets: Sequence[Any]) -> List[str]:
    role = proof_role.lower()
    patterns = _dedupe(lane_targets, limit=8)
    if role == "metric":
        patterns.extend(["official_data", "market_research", "survey", "pdf", "annual_report"])
    elif role in {"filing", "source_check"}:
        patterns.extend(["official_data", "filing_company", "exchange_announcement", "investor_relations"])
    elif role == "counter":
        patterns.extend(["counter_evidence", "failure", "cost", "roi_unclear", "security", "compliance"])
    elif role == "case":
        patterns.extend(["customer_case", "company_disclosure", "procurement"])
    else:
        patterns.extend(["market_research", "official_data"])
    return _dedupe(patterns, limit=8)


def _repair_seed_from_gap(gap: Dict[str, Any]) -> Dict[str, Any]:
    retry_plan = _as_dict(gap.get("retry_plan"))
    proof_role = _text(gap.get("proof_role") or retry_plan.get("proof_role"))
    required_fields = _dedupe(gap.get("missing") or retry_plan.get("required_fields") or [], limit=8)
    lane_targets = _dedupe(retry_plan.get("lane_targets") or gap.get("lane_targets") or [], limit=8)
    patterns = _source_patterns(proof_role, lane_targets)
    query_parts = [
        retry_plan.get("next_search_task"),
        retry_plan.get("query_seed"),
        " ".join(_as_list(retry_plan.get("query_terms"))),
        gap.get("current_insufficiency"),
        gap.get("gap_type") or gap.get("type"),
        proof_role,
        " ".join(required_fields),
        " ".join(patterns[:3]),
    ]
    query = _compact(" ".join(_text(item) for item in query_parts if _text(item)), 260)
    if not query:
        query = _compact(" ".join([_text(gap.get("requirement_id")), _text(gap.get("gap_type") or gap.get("type"))]), 260)
    seed = {
        "schema_version": "repair_task_seed_v2",
        "query": query,
        "gap_id": _text(gap.get("gap_id")),
        "requirement_id": _text(gap.get("requirement_id")),
        "chapter_id": _text(gap.get("chapter_id")),
        "section_id": _text(gap.get("section_id")),
        "gap_type": _text(gap.get("gap_type") or gap.get("type")),
        "repair_status": _text(gap.get("status")),
        "proof_role": proof_role,
        "required_fields": required_fields,
        "required_source_level": _dedupe(retry_plan.get("required_source_level") or retry_plan.get("min_source_level") or [], limit=4),
        "lane_targets": lane_targets,
        "preferred_source_patterns": patterns,
        "success_criteria": retry_plan.get("success_criteria")
        or ("Only count as repaired when missing fields are traceable: " + ", ".join(required_fields) if required_fields else "Only count as repaired when evidence is traceable to a concrete source URL."),
        "reject_if": _dedupe(
            [
                "snippet_only",
                "no_source_url",
                "marketing_copy_only",
                *(
                    ("no_date",)
                    if proof_role in {"metric", "filing", "source_check"}
                    or "period" in {field.lower() for field in required_fields}
                    else ()
                ),
            ],
            limit=8,
        ),
        "allowed_for_writing": False,
        "source": "research_reflection_memo",
        "avoid_repeating_failed_query": bool(_as_list(retry_plan.get("failed_queries")) or _text(gap.get("status")) == "still_insufficient"),
        "live_refresh_required": _text(gap.get("status")) == "live_search_required",
    }
    return dispatch_repair_seed(seed, failed_queries=_as_list(retry_plan.get("failed_queries") or retry_plan.get("avoid_queries")))


def _next_search_task_seeds(gaps: Sequence[Dict[str, Any]], *, limit: int = 8) -> List[Dict[str, Any]]:
    scored: List[tuple[int, Dict[str, Any]]] = []
    for gap in gaps:
        retry_plan = _as_dict(gap.get("retry_plan"))
        severity = _text(gap.get("severity")).lower()
        status = _text(gap.get("status")).lower()
        proof_role = _text(gap.get("proof_role") or retry_plan.get("proof_role")).lower()
        score = {
            "fatal": 100,
            "blocking": 80,
            "high": 60,
            "medium": 30,
        }.get(severity, 10)
        if status in {"live_search_required", "still_insufficient", "needs_repair"}:
            score += 25
        if proof_role in {"metric", "counter", "filing", "source_check"}:
            score += 15
        scored.append((score, gap))
    scored.sort(key=lambda item: (-item[0], _text(item[1].get("gap_id"))))
    return [_repair_seed_from_gap(gap) for _, gap in scored[:limit]]


def _weak_claims(structured_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    claims: List[Dict[str, Any]] = []
    for unit in _as_list(structured_analysis.get("claim_units")):
        if not isinstance(unit, dict):
            continue
        strength = _text(unit.get("claim_strength") or unit.get("claim_status")).lower()
        metric_status = _text(unit.get("metric_completeness_status")).lower()
        use_level = _text(unit.get("evidence_use_level") or unit.get("allowed_use")).lower()
        if strength not in {"directional", "weak", "limited_evidence"} and metric_status != "incomplete" and use_level not in {"directional_signal", "clue_only"}:
            continue
        claims.append(
            {
                "claim_id": _text(unit.get("claim_id") or unit.get("id")),
                "requirement_ids": _dedupe(unit.get("requirement_ids") or [], limit=6),
                "fact_ids": _dedupe(unit.get("fact_ids") or unit.get("evidence_refs") or [], limit=8),
                "claim_strength": strength or "directional",
                "evidence_use_level": use_level,
                "metric_completeness_status": metric_status,
                "metric_missing_fields": _dedupe(unit.get("metric_missing_fields") or [], limit=8),
                "allowed_for_writing": False,
            }
        )
    return claims[:12]


def _status_and_mode(health: Dict[str, Any], gaps: Sequence[Dict[str, Any]], finding_refs: Sequence[Dict[str, Any]]) -> tuple[str, str, bool, bool, str]:
    analysis_ready = _int(health.get("analysis_ready_count")) or len(finding_refs)
    verified_ab = _int(health.get("distinct_verified_ab_source_count"))
    traceable_ab = _int(health.get("traceable_ab_source_count"))
    publishable_gate = health.get("publishable_evidence_gate_passed") is True
    severe_gaps = [
        gap for gap in gaps
        if _text(gap.get("severity")).lower() in {"fatal", "blocking", "high"}
        or _text(gap.get("status")).lower() in {"still_insufficient", "live_search_required"}
    ]
    if publishable_gate and (verified_ab or traceable_ab) and not severe_gaps:
        return "sufficient", "publishable_draft", True, True, "write_from_validated_claims"
    if analysis_ready > 0:
        return "limited", "limited_review_draft", True, False, "write_with_boundaries_do_not_infer"
    if gaps:
        return "insufficient", "short_honest_draft", False, False, "do_not_infer"
    return "empty", "do_not_write", False, False, "do_not_infer"


def build_research_reflection_memo(
    evidence_package: Dict[str, Any],
    structured_analysis: Dict[str, Any] | None = None,
    writer_report: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build a compact deterministic memo for research orchestration."""

    evidence_package = _as_dict(evidence_package)
    structured_analysis = _as_dict(structured_analysis)
    writer_report = _as_dict(writer_report)
    health = _health(evidence_package) or _as_dict(writer_report.get("evidence_health_summary"))
    gaps = _gap_items(evidence_package, structured_analysis, writer_report)
    coverage = _coverage_from_gaps(gaps, _as_list(evidence_package.get("coverage_matrix") or writer_report.get("coverage_matrix")))
    finding_refs = _finding_refs(evidence_package)
    status, write_mode, enough_to_write, enough_for_publishable, instruction = _status_and_mode(health, gaps, finding_refs)
    gap_type_counts = Counter(_text(gap.get("gap_type") or gap.get("type") or "unknown") for gap in gaps)
    proof_role_counts = Counter(
        _text(gap.get("proof_role") or _as_dict(gap.get("retry_plan")).get("proof_role") or "unknown")
        for gap in gaps
    )
    missing_fields = _dedupe(
        field
        for gap in gaps
        for field in _as_list(gap.get("missing") or _as_dict(gap.get("retry_plan")).get("required_fields"))
    )
    memo = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "instruction": instruction,
        "write_mode": write_mode,
        "enough_to_write": enough_to_write,
        "enough_for_publishable": enough_for_publishable,
        "allowed_for_writing": False,
        "cache_boundary": "diagnostic_memo_no_quoteable_fact_text",
        "known_finding_refs": finding_refs,
        "coverage_by_requirement": coverage,
        "ambiguous_or_weak_claims": _weak_claims(structured_analysis),
        "missing_fields": missing_fields,
        "gap_type_counts": dict(gap_type_counts),
        "proof_role_counts": dict(proof_role_counts),
        "next_search_task_seeds": _next_search_task_seeds(gaps),
        "health_summary": {
            "analysis_ready_count": _int(health.get("analysis_ready_count")) or len(finding_refs),
            "analysis_ready_ab_count": _int(health.get("analysis_ready_ab_count")),
            "traceable_ab_source_count": _int(health.get("traceable_ab_source_count")),
            "distinct_verified_ab_source_count": _int(health.get("distinct_verified_ab_source_count")),
            "publishable_evidence_gate_passed": bool(health.get("publishable_evidence_gate_passed")),
        },
    }
    return memo
