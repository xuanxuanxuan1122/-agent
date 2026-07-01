from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def repair_action_for_review_suggestion(suggestion: Mapping[str, Any]) -> str:
    """Choose the execution route for a diagnostic review suggestion.

    Review layers are advisory. This function deliberately returns an action
    for downstream analysis/writer/repair agents; it never changes public text.
    """

    payload = _as_dict(dict(suggestion))
    issue_type = _text(payload.get("issue_type") or payload.get("type")).lower()
    suggested = _text(payload.get("suggested_action")).lower()
    if suggested in {"search_more", "reanalyze_existing", "recompose_outline", "rewrite_with_caveat"}:
        return suggested

    if issue_type in {
        "low_claim_conversion",
        "claim_conversion_low",
        "evidence_many_claim_few",
        "analysis_underconverted",
        "needs_reanalysis",
        "section_ref_binding_invalid_only",
    }:
        return "reanalyze_existing"
    if issue_type in {
        "chapter_mismatch",
        "outline_mismatch",
        "plan_final_chapter_mismatch",
        "chapter_thin_with_claim_elsewhere",
        "section_ref_binding_unbound",
    }:
        return "recompose_outline"
    if issue_type in {
        "single_source",
        "weak_source",
        "claim_overreach",
        "source_depth_weak",
        "body_short",
        "chapter_thin",
        "missing_counter",
    }:
        return "rewrite_with_caveat"
    if issue_type in {
        "missing_metric",
        "metric_missing",
        "missing_case",
        "case_evidence_missing",
        "missing_policy",
        "missing_counter_evidence",
        "counter_evidence_missing",
        "missing_required_field",
    }:
        return "search_more"
    return "rewrite_with_caveat"


def make_review_suggestion(
    *,
    issue_type: str,
    severity: str = "warning",
    target: Mapping[str, Any] | None = None,
    suggested_action: str = "",
    detail: Mapping[str, Any] | None = None,
    source_stage: str = "",
    message: str = "",
) -> Dict[str, Any]:
    action = _text(suggested_action) or repair_action_for_review_suggestion({"issue_type": issue_type})
    payload: Dict[str, Any] = {
        "schema_version": "review_suggestion_v1",
        "issue_type": _text(issue_type) or "quality_observation",
        "severity": _text(severity) or "warning",
        "target": dict(_as_dict(target)),
        "suggested_action": action,
        "diagnostic_only": True,
        "must_not_render": True,
        "public_text_allowed": False,
        "not_for_public_text": True,
        "executor_should_decide": True,
    }
    if source_stage:
        payload["source_stage"] = _text(source_stage)
    if message:
        payload["message"] = _text(message)
    if detail:
        payload["detail"] = dict(_as_dict(detail))
    return payload


def review_suggestion_from_issue(issue: Mapping[str, Any], *, source_stage: str = "") -> Dict[str, Any]:
    payload = _as_dict(dict(issue))
    issue_type = _text(payload.get("issue_type") or payload.get("type") or payload.get("gap_type"))
    target = {
        key: payload.get(key)
        for key in ("chapter_id", "section_id", "claim_id", "requirement_id", "gap_id", "table_id")
        if payload.get(key) not in (None, "", [])
    }
    return make_review_suggestion(
        issue_type=issue_type or "quality_observation",
        severity=_text(payload.get("severity") or payload.get("level") or "warning"),
        target=target,
        suggested_action=_text(payload.get("suggested_action")),
        detail=payload,
        source_stage=source_stage or _text(payload.get("source_stage") or payload.get("source")),
        message=_text(payload.get("message") or payload.get("reason")),
    )


def review_suggestion_to_required_followup(
    suggestion: Mapping[str, Any],
    *,
    source_stage: str = "",
) -> Dict[str, Any]:
    payload = _as_dict(dict(suggestion))
    if payload.get("schema_version") != "review_suggestion_v1":
        payload = review_suggestion_from_issue(payload, source_stage=source_stage)
    target = _as_dict(payload.get("target"))
    issue_type = _text(payload.get("issue_type") or payload.get("type") or "quality_observation").lower()
    action = repair_action_for_review_suggestion(payload)
    detail = _as_dict(payload.get("detail"))
    followup: Dict[str, Any] = {
        "schema_version": "review_suggestion_required_followup_v1",
        "type": issue_type,
        "issue_type": issue_type,
        "repair_action": action,
        "repair_route": action,
        "source": source_stage or _text(payload.get("source_stage") or payload.get("source")) or "review_suggestion",
        "source_stage": source_stage or _text(payload.get("source_stage") or payload.get("source")) or "review_suggestion",
        "severity": _text(payload.get("severity") or "warning"),
        "chapter_id": _text(target.get("chapter_id") or payload.get("chapter_id")),
        "section_id": _text(target.get("section_id") or payload.get("section_id")),
        "claim_id": _text(target.get("claim_id") or payload.get("claim_id")),
        "requirement_id": _text(target.get("requirement_id") or payload.get("requirement_id")),
        "gap_id": _text(target.get("gap_id") or payload.get("gap_id")),
        "message": _text(payload.get("message") or detail.get("message") or detail.get("reason")),
        "diagnostic_only": True,
        "must_not_render": True,
        "public_text_allowed": False,
        "not_for_public_text": True,
        "executor_should_decide": True,
        "allowed_for_writing": False,
        "priority": "high" if action in {"reanalyze_existing", "recompose_outline", "search_more"} else "medium",
    }
    for key in (
        "query",
        "suggested_query",
        "proof_role",
        "required_fields",
        "required_source_level",
        "lane_targets",
        "success_criteria",
        "reject_if",
    ):
        value = payload.get(key, detail.get(key))
        if value not in (None, "", [], {}):
            followup[key] = value
    return {key: value for key, value in followup.items() if value not in ("", [], {})}


def review_suggestions_to_required_followups(
    suggestions: Iterable[Mapping[str, Any]],
    *,
    source_stage: str = "",
    limit: int = 40,
) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    seen = set()
    for raw in suggestions or []:
        if not isinstance(raw, Mapping):
            continue
        followup = review_suggestion_to_required_followup(raw, source_stage=source_stage)
        key = (
            followup.get("repair_action"),
            followup.get("issue_type"),
            followup.get("chapter_id"),
            followup.get("section_id"),
            followup.get("claim_id"),
            followup.get("requirement_id"),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(followup)
        if len(result) >= limit:
            break
    return result
