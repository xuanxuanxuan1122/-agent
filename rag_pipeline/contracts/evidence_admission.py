from __future__ import annotations

from typing import Any, Dict, Iterable, List


PUBLISHABLE_SOURCE_LEVELS = {"A", "B"}
REJECT_STATUSES = {"rejected", "stale", "superseded", "invalid", "error"}
BAD_PAGE_STATUSES = {"snippet_only", "login_required", "http_error", "navigation_page", "marketing_copy_only"}
VERIFIED_SOURCE_STATUSES = {
    "readpage_verified",
    "document_verified",
    "manual_verified",
    "source_verified",
    "verified",
}
GENERIC_METRIC_NAMES = {"", "unknown", "key_fact", "fact", "metric", "indicator", "关键事实", "鍏抽敭浜嬪疄"}


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


def _dedupe(items: Iterable[str]) -> List[str]:
    result: List[str] = []
    seen = set()
    for item in items:
        value = _text(item)
        key = value.lower()
        if not value or key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _source_level(card: Dict[str, Any]) -> str:
    return _text(card.get("source_level") or _as_dict(card.get("source")).get("source_level")).upper() or "unknown"


def _verification_status(card: Dict[str, Any]) -> str:
    return _text(
        card.get("source_verification_status")
        or card.get("verification_status")
        or _as_dict(card.get("source")).get("verification_status")
    ).lower()


def _proof_role(card: Dict[str, Any]) -> str:
    return _text(card.get("proof_role") or card.get("fact_type") or card.get("analysis_role")).lower()


def _allowed_use(card: Dict[str, Any]) -> str:
    return _text(card.get("allowed_use") or card.get("writing_permission")).lower()


def _metric_name(card: Dict[str, Any]) -> str:
    metric = _text(card.get("metric") or card.get("indicator") or card.get("variable"))
    return "" if metric.lower() in GENERIC_METRIC_NAMES else metric


def _is_metric_evidence(card: Dict[str, Any]) -> bool:
    proof_role = _proof_role(card)
    if proof_role == "metric":
        return True
    return bool(_metric_name(card) and _text(card.get("value") or card.get("display_value") or card.get("numeric_value")))


def _value_carries_unit(card: Dict[str, Any]) -> bool:
    value = _text(card.get("value") or card.get("display_value") or card.get("numeric_value")).lower()
    return bool(value and ("%" in value or "percent" in value or "percentage point" in value or "百分点" in value or "百分比" in value))


def _metric_missing_fields(card: Dict[str, Any]) -> List[str]:
    missing: List[str] = []
    if not _metric_name(card):
        missing.append("metric")
    if not _text(card.get("value") or card.get("display_value") or card.get("numeric_value")):
        missing.append("value")
    if not _text(card.get("unit") or card.get("numeric_unit")) and not _value_carries_unit(card):
        missing.append("unit")
    if not _text(card.get("period") or card.get("scope") or card.get("time_or_scope") or card.get("date")):
        missing.append("period")
    source = _as_dict(card.get("source"))
    if not _text(card.get("source_url") or card.get("source_ref") or card.get("source_title") or source.get("url")):
        missing.append("source")
    return missing


def decide_evidence_admission(fact_card: Dict[str, Any]) -> Dict[str, Any]:
    """Return one unified admission decision for evidence consumption.

    The decision is deliberately conservative and diagnostic: downstream
    stages can consume the verdict without re-running a different local policy.
    """

    card = _as_dict(fact_card)
    status = _text(card.get("status") or card.get("fact_status")).lower()
    source_level = _source_level(card)
    verification_status = _verification_status(card)
    allowed_use = _allowed_use(card)
    proof_role = _proof_role(card)
    page_status = _text(card.get("page_status") or card.get("readpage_status")).lower()
    reasons: List[str] = []
    repair_action = "none"

    if status in REJECT_STATUSES:
        reasons.append(f"status_{status}")
    if verification_status in BAD_PAGE_STATUSES or page_status in BAD_PAGE_STATUSES:
        reasons.append("page_not_usable")
    if allowed_use in {"appendix_only", "clue"}:
        reasons.append(f"allowed_use_{allowed_use}")

    metric_missing: List[str] = []
    if _is_metric_evidence(card):
        metric_missing = _metric_missing_fields(card)
        if metric_missing:
            reasons.append("metric_fields_incomplete")
            repair_action = "repair_metric_fields"

    if any(reason.startswith("status_") or reason == "page_not_usable" for reason in reasons):
        verdict = "reject"
        public_use = "not_allowed"
        confidence = 0.95
    elif "allowed_use_appendix_only" in reasons or allowed_use == "appendix_only":
        verdict = "appendix_only"
        public_use = "appendix_only"
        confidence = 0.85
    elif "metric_fields_incomplete" in reasons:
        verdict = "directional"
        public_use = "cautious_with_boundary"
        confidence = 0.55
    elif source_level in PUBLISHABLE_SOURCE_LEVELS and (
        verification_status in VERIFIED_SOURCE_STATUSES or _text(card.get("source_url") or _as_dict(card.get("source")).get("url"))
    ):
        verdict = "publishable"
        public_use = "writing"
        confidence = 0.8 if source_level == "B" else 0.9
    else:
        verdict = "directional"
        public_use = "cautious_with_boundary"
        reasons.append("source_level_or_verification_limited")
        confidence = 0.5

    return {
        "schema_version": "evidence_admission_decision_v1",
        "evidence_id": _text(card.get("evidence_id") or card.get("fact_id") or card.get("id")),
        "verdict": verdict,
        "reasons": _dedupe(reasons),
        "allowed_use": public_use,
        "repair_action": repair_action,
        "confidence": confidence,
        "source_level": source_level,
        "proof_role": proof_role or "unknown",
        "metric_missing_fields": metric_missing,
    }


def summarize_evidence_admission(fact_cards: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    decisions = [decide_evidence_admission(card) for card in fact_cards if isinstance(card, dict)]
    verdict_counts: Dict[str, int] = {}
    repair_counts: Dict[str, int] = {}
    for decision in decisions:
        verdict = _text(decision.get("verdict")) or "unknown"
        repair = _text(decision.get("repair_action")) or "none"
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        if repair != "none":
            repair_counts[repair] = repair_counts.get(repair, 0) + 1
    return {
        "schema_version": "evidence_admission_summary_v1",
        "input_count": len(decisions),
        "verdict_counts": verdict_counts,
        "repair_action_counts": repair_counts,
        "decisions": decisions,
    }
