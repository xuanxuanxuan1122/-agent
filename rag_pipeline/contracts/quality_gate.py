from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _dedupe(values: Sequence[Any]) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _has_evidence_reason(reasons: Sequence[str]) -> bool:
    for reason in reasons:
        text = reason.lower()
        if any(token in text for token in ("evidence", "source", "proof", "citation", "metric", "table", "ab_source")):
            return True
        if "followup" in text and not any(token in text for token in ("content", "rewrite", "text")):
            return True
    return False


def _contract_repair_reasons(contract: Dict[str, Any]) -> List[str]:
    reasons: List[str] = []
    for issue in _as_list(contract.get("contract_issues")):
        issue = _as_dict(issue)
        issue_type = str(issue.get("type") or "").strip()
        severity = str(issue.get("severity") or "").strip()
        if issue_type and severity in {"error", "fatal"}:
            reasons.append(issue_type)
    return _dedupe(reasons)


def build_quality_gate_state(
    *,
    writer_status: str,
    writer_not_ready: bool,
    writer_pending_repair_reasons: Sequence[str],
    reformatter_result: Optional[Dict[str, Any]] = None,
    report_contract: Optional[Dict[str, Any]] = None,
    writer_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    reasons = _dedupe([str(item) for item in list(writer_pending_repair_reasons or [])])
    reformatter = _as_dict(reformatter_result)
    repair_plan = _as_dict(reformatter.get("repair_plan")) or _as_dict(reformatter.get("reformatter_repair_plan"))
    repair_status = str(repair_plan.get("status") or "").strip()
    post_reformatter_trace = _as_list(reformatter.get("post_reformatter_repair_trace"))
    post_reformatter_last = _as_dict(post_reformatter_trace[-1]) if post_reformatter_trace else {}
    reformatter_classification = _as_dict(reformatter.get("reformatter_repair_classification")) or _as_dict(
        repair_plan.get("reformatter_repair_classification")
    )
    reformatter_repair_state = {
        "status": (
            "repaired"
            if bool(reformatter.get("success")) and bool(reformatter.get("output_written")) and post_reformatter_trace
            else "still_failed"
            if post_reformatter_trace and str(reformatter.get("status") or "") in {"repair_required", "validation_failed", "failed"}
            else "not_run"
        ),
        "primary_route": post_reformatter_last.get("primary_route") or reformatter.get("clean_report_repair_route") or reformatter_classification.get("primary_route") or "",
        "stop_reason": post_reformatter_last.get("stop_reason") or "",
        "quality_gain": bool(post_reformatter_last.get("improved")),
    }
    contract = _as_dict(report_contract)
    writer = _as_dict(writer_report)
    post_qa_repair = _as_dict(writer.get("post_qa_repair"))
    post_qa_repair_status = str(post_qa_repair.get("status") or "").strip()
    post_qa_repair_stop_reason = str(post_qa_repair.get("stop_reason") or "").strip()
    contract_repair_reasons = _contract_repair_reasons(contract)
    if contract_repair_reasons:
        reasons = _dedupe([*reasons, *contract_repair_reasons])
    repair_routes: List[Dict[str, Any]] = []
    post_qa_evidence_exhausted = post_qa_repair_status in {"no_new_evidence_signal", "no_signal", "evidence_exhausted", "manual_review"}
    post_qa_evidence_exhausted = post_qa_evidence_exhausted or post_qa_repair_stop_reason in {
        "signal_found_but_no_quality_gain",
        "no_new_evidence_signal",
    }

    if writer_not_ready:
        status = "blocked"
        state = "evidence_insufficient"
        next_action = "evidence_refinement"
        loop_target = "evidence_refinement"
        reasons = _dedupe([*reasons, "writer_not_ready"])
    elif reasons:
        evidence_related = _has_evidence_reason(reasons)
        if evidence_related and post_qa_evidence_exhausted:
            status = "needs_degrade_or_manual_review"
            state = "post_qa_evidence_refinement_exhausted"
            next_action = "manual_review"
            loop_target = "manual_review"
            reasons = _dedupe([*reasons, "post_qa_no_new_evidence_signal"])
        else:
            status = "needs_evidence" if evidence_related else "needs_rewrite"
            state = "qa_failed_needs_evidence" if evidence_related else "qa_failed_needs_rewrite"
            next_action = "evidence_refinement" if evidence_related else "rewrite"
            loop_target = next_action
    elif bool(reformatter.get("success")) and bool(reformatter.get("output_written")):
        status = "publishable"
        state = "publishable"
        next_action = "publish"
        loop_target = ""
    elif repair_status == "needs_evidence_refinement":
        status = "needs_evidence"
        state = "reformatter_failed_needs_evidence"
        next_action = "evidence_refinement"
        loop_target = "evidence_refinement"
        reasons = _dedupe([*reasons, *[str(item) for item in _as_list(repair_plan.get("reasons"))]])
    elif repair_status == "needs_citation_repair":
        status = "needs_evidence"
        state = "reformatter_failed_needs_citation_repair"
        next_action = "citation_repair"
        loop_target = "citation_repair"
        reasons = _dedupe([*reasons, "citation_repair_required"])
    elif repair_status == "needs_text_repair":
        status = "needs_rewrite"
        state = "reformatter_failed_needs_rewrite"
        next_action = "rewrite"
        loop_target = "rewrite"
        reasons = _dedupe([*reasons, *[str(item) for item in _as_list(repair_plan.get("text_repair_reasons"))]])
    elif repair_status == "needs_degrade_or_manual_review":
        status = "needs_degrade_or_manual_review"
        state = "degrade_or_manual_review"
        next_action = "degrade_or_manual_review"
        loop_target = "manual_review"
    elif str(reformatter.get("status") or "") in {"failed", "repair_required", "validation_failed"}:
        status = "needs_review"
        state = "reformatter_failed_needs_review"
        next_action = "manual_review"
        loop_target = "manual_review"
        if reformatter.get("error"):
            reasons = _dedupe([*reasons, str(reformatter.get("error"))])
    elif str(writer_status or "").strip().lower() == "final":
        status = "ready_for_reformatter"
        state = "ready_for_reformatter"
        next_action = "reformatter"
        loop_target = "reformatter"
    else:
        status = "needs_review"
        state = "writer_draft_needs_review"
        next_action = "manual_review"
        loop_target = "manual_review"

    if status == "needs_evidence":
        repair_routes.append({"from": state, "to": "evidence_refinement", "reason": "evidence_gap_or_source_gap"})
    if status == "needs_rewrite":
        repair_routes.append({"from": state, "to": "rewrite", "reason": "text_logic_or_structure_gap"})
    if status in {"needs_review", "needs_degrade_or_manual_review"}:
        repair_routes.append({"from": state, "to": next_action, "reason": "automatic_repair_not_sufficient"})
    if next_action == "citation_repair":
        repair_routes.append({"from": state, "to": "citation_repair", "reason": "citation_or_source_rebinding_required"})

    evidence_required = status == "needs_evidence" or next_action in {"evidence_refinement", "citation_repair"}
    rewrite_required = status == "needs_rewrite" or next_action == "rewrite"
    manual_review_required = next_action in {"manual_review", "degrade_or_manual_review"}
    degrade_allowed = bool(repair_plan.get("degrade_allowed", True))
    if evidence_required:
        degrade_allowed = False
    if post_qa_evidence_exhausted and manual_review_required:
        degrade_allowed = True
    if status == "publishable":
        degrade_allowed = False

    return {
        "status": status,
        "state": state,
        "next_action": next_action,
        "loop_target": loop_target,
        "writer_status": str(writer_status or ""),
        "writer_pending_repair_reasons": reasons,
        "reformatter_status": str(reformatter.get("status") or ""),
        "reformatter_repair_status": repair_status,
        "blocking_reasons": reasons,
        "repair_routes": repair_routes,
        "evidence_required": evidence_required,
        "rewrite_required": rewrite_required,
        "manual_review_required": manual_review_required,
        "degrade_allowed": degrade_allowed,
        "publishable": status == "publishable",
        "contract_version": str(contract.get("contract_version") or ""),
        "quality_thresholds": _as_dict(contract.get("quality_thresholds")),
        "contract_repair_reasons": contract_repair_reasons,
        "post_qa_repair_status": post_qa_repair_status,
        "post_qa_repair_stop_reason": post_qa_repair_stop_reason,
        "reformatter_repair_state": reformatter_repair_state,
    }
