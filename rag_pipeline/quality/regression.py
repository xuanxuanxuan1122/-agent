from __future__ import annotations

import math
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from rag_pipeline.contracts.evidence_admission import summarize_evidence_admission


CLOSED_GAP_STATUSES = {"closed", "resolved", "repaired", "evidence_found", "cache_satisfied"}
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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return default


def _first_dict(*values: Any) -> Dict[str, Any]:
    for value in values:
        payload = _as_dict(value)
        if payload:
            return payload
    return {}


def _writer_report(package: Dict[str, Any], writer_report: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return _as_dict(writer_report) or _as_dict(package.get("writer_report"))


def _analysis_diag(package: Dict[str, Any], writer_report: Dict[str, Any]) -> Dict[str, Any]:
    structured = _as_dict(package.get("structured_analysis")) or _as_dict(writer_report.get("structured_analysis"))
    return _first_dict(
        package.get("analysis_stage_diagnostics"),
        structured.get("analysis_stage_diagnostics"),
        _as_dict(writer_report.get("render_artifacts")).get("analysis_stage_diagnostics"),
    )


def _quality_score(report: Dict[str, Any]) -> int:
    qa = _as_dict(report.get("qa_result"))
    clean_gate = _as_dict(report.get("clean_gate"))
    return _safe_int(report.get("quality_score") or qa.get("quality_score") or clean_gate.get("quality_score"))


def _fatal_types(final_audit: Dict[str, Any]) -> List[str]:
    types: List[str] = []
    for item in [*_as_list(final_audit.get("critical_findings")), *_as_list(final_audit.get("citation_issues"))]:
        payload = _as_dict(item)
        severity = _text(payload.get("severity")).lower()
        if severity and severity not in {"fatal", "blocking", "high"}:
            continue
        types.append(_text(payload.get("type") or payload.get("issue_type") or payload.get("issue") or "unknown"))
    return [item for item in types if item]


def _run_metrics(package: Dict[str, Any], report: Dict[str, Any]) -> Dict[str, Any]:
    direct = _first_dict(package.get("run_metrics"), report.get("run_metrics"), package.get("usage"), report.get("usage"))
    total_tokens = _safe_int(
        direct.get("total_tokens")
        or direct.get("token_count")
        or direct.get("tokens")
        or package.get("total_tokens")
        or report.get("total_tokens")
    )
    duration_seconds = _safe_float(
        direct.get("duration_seconds")
        or direct.get("elapsed_seconds")
        or direct.get("wall_seconds")
        or package.get("duration_seconds")
        or report.get("duration_seconds")
    )
    cost_usd = _safe_float(direct.get("cost_usd") or direct.get("estimated_cost_usd") or package.get("cost_usd") or report.get("cost_usd"))
    for payload in (
        _as_dict(package.get("final_audit_result")),
        _as_dict(report.get("final_audit_result")),
        _as_dict(package.get("structured_analysis")).get("analysis_stage_diagnostics"),
        _as_dict(report.get("structured_analysis")).get("analysis_stage_diagnostics"),
    ):
        data = _as_dict(payload)
        usage = _as_dict(data.get("usage") or data.get("_llm_usage") or data.get("llm_usage"))
        total_tokens += _safe_int(usage.get("total_tokens")) if usage else 0
        semantic_usage = _as_dict(data.get("llm_semantic_judge_usage"))
        total_tokens += _safe_int(semantic_usage.get("total_tokens")) if semantic_usage else 0
    return {
        "total_tokens": total_tokens,
        "duration_seconds": duration_seconds,
        "cost_usd": cost_usd,
    }


def _metadata(package: Dict[str, Any], report: Dict[str, Any]) -> Dict[str, Any]:
    return _first_dict(package.get("metadata"), report.get("metadata"))


def _handoff_contract_summary(package: Dict[str, Any], report: Dict[str, Any]) -> Dict[str, Any]:
    render_artifacts = _as_dict(report.get("render_artifacts"))
    metadata = _metadata(package, report)
    stage_card = _first_dict(
        package.get("stage_quality_card"),
        report.get("stage_quality_card"),
        render_artifacts.get("stage_quality_card"),
        metadata.get("stage_quality_card"),
    )
    return _first_dict(
        package.get("handoff_contract_summary"),
        report.get("handoff_contract_summary"),
        render_artifacts.get("handoff_contract_summary"),
        metadata.get("handoff_contract_summary"),
        stage_card.get("handoff"),
    )


def _score_gaps(package: Dict[str, Any], report: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        _as_dict(item)
        for item in (
            _as_list(package.get("score_gaps"))
            or _as_list(report.get("score_gaps"))
            or _as_list(_as_dict(package.get("evidence_package")).get("evidence_gap_ledger"))
        )
        if isinstance(item, dict)
    ]


def _repair_selection_summary(package: Dict[str, Any], report: Dict[str, Any]) -> Dict[str, Any]:
    metadata = _metadata(package, report)
    raw_metadata = _as_dict(_as_dict(package.get("raw_output")).get("metadata"))
    render_artifacts = _as_dict(report.get("render_artifacts"))
    return _first_dict(
        metadata.get("repair_task_selection_summary"),
        raw_metadata.get("repair_task_selection_summary"),
        package.get("repair_task_selection_summary"),
        report.get("repair_task_selection_summary"),
        render_artifacts.get("repair_task_selection_summary"),
    )


def _repair_result_summaries(report: Dict[str, Any], package: Dict[str, Any]) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for item in [
        *_as_list(report.get("post_qa_repair_trace")),
        *_as_list(package.get("post_qa_repair_trace")),
        report.get("post_qa_repair"),
        package.get("post_qa_repair"),
    ]:
        payload = _as_dict(item)
        summary = _as_dict(payload.get("repair_result_summary")) or payload
        if summary:
            summaries.append(summary)
    return summaries


def summarize_repair_effectiveness(
    *,
    writer_package: Dict[str, Any],
    writer_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    package = _as_dict(writer_package)
    report = _writer_report(package, writer_report)
    gaps = _score_gaps(package, report)
    selection = _repair_selection_summary(package, report)
    selected_count = _safe_int(selection.get("task_count")) or sum(
        _safe_int(value) for value in _as_dict(selection.get("by_proof_role") or selection.get("by_reason")).values()
    )

    status_counts: Counter[str] = Counter()
    gap_types: Counter[str] = Counter()
    for gap in gaps:
        status = _text(gap.get("status")).lower() or "unknown"
        status_counts[status] += 1
        gap_type = _text(gap.get("gap_type") or gap.get("type")) or "unknown"
        gap_types[gap_type] += 1

    closed_count = sum(status_counts[status] for status in CLOSED_GAP_STATUSES)
    open_count = sum(status_counts[status] for status in OPEN_GAP_STATUSES)
    attempted_count = max(selected_count, len(gaps), closed_count + open_count)

    result_counter: Counter[str] = Counter()
    for summary in _repair_result_summaries(report, package):
        for key in ("new_usable_evidence_count", "new_ab_source_count", "signal_count", "empty_success_count"):
            result_counter[key] += _safe_int(summary.get(key))

    closure_rate = closed_count / attempted_count if attempted_count else 0.0
    if attempted_count <= 0:
        effectiveness_status = "not_run"
    elif closed_count >= attempted_count:
        effectiveness_status = "effective"
    elif closed_count > 0 or result_counter["new_usable_evidence_count"] > 0:
        effectiveness_status = "partial"
    else:
        effectiveness_status = "no_signal"

    return {
        "schema_version": "repair_effectiveness_v1",
        "attempted_gap_count": attempted_count,
        "selected_repair_task_count": selected_count,
        "closed_gap_count": closed_count,
        "open_gap_count": open_count,
        "closure_rate": closure_rate,
        "new_usable_evidence_count": result_counter["new_usable_evidence_count"],
        "new_ab_source_count": result_counter["new_ab_source_count"],
        "signal_count": result_counter["signal_count"],
        "empty_success_count": result_counter["empty_success_count"],
        "by_gap_status": dict(status_counts),
        "by_gap_type": dict(gap_types),
        "effectiveness_status": effectiveness_status,
    }


def build_run_quality_snapshot(
    writer_package: Dict[str, Any],
    writer_report: Optional[Dict[str, Any]] = None,
    *,
    final_status: str = "",
) -> Dict[str, Any]:
    package = _as_dict(writer_package)
    report = _writer_report(package, writer_report)
    metadata = _metadata(package, report)
    analysis_diag = _analysis_diag(package, report)
    correctness_summary = _as_dict(analysis_diag.get("correctness_filter_summary"))
    final_audit = _as_dict(package.get("final_audit_result")) or _as_dict(report.get("final_audit_result"))
    run_metrics = _run_metrics(package, report)
    handoff = _handoff_contract_summary(package, report)
    handoff_failed_contracts = [
        _text(item)
        for item in _as_list(handoff.get("failed_contracts"))
        if _text(item)
    ]
    evidence_package = _as_dict(package.get("evidence_package")) or _as_dict(report.get("evidence_package"))
    fact_cards = (
        _as_list(evidence_package.get("analysis_ready_evidence"))
        or _as_list(evidence_package.get("clean_evidence_list"))
        or _as_list(package.get("fact_cards"))
    )

    return {
        "schema_version": "run_quality_snapshot_v1",
        "run_id": _text(metadata.get("run_id") or package.get("run_id") or report.get("run_id")),
        "topic_id": _text(metadata.get("topic_id") or metadata.get("query_id") or package.get("topic_id") or "unknown"),
        "domain": _text(metadata.get("domain") or metadata.get("industry") or "unknown"),
        "query": _text(metadata.get("query") or package.get("query") or report.get("query")),
        "final_status": _text(final_status or package.get("final_status") or report.get("final_status")),
        "report_status": _text(report.get("report_status")),
        "quality_score": _quality_score(report),
        "final_audit_blocked": bool(final_audit.get("blocked")),
        "final_audit_status": _text(final_audit.get("status")),
        "fatal_types": _fatal_types(final_audit),
        "formal_report_written": bool(
            _as_dict(package.get("report_delivery_status")).get("formal_report_written")
            or report.get("formal_report_written")
            or report.get("formal_report_path")
        ),
        "usable_claim_count": _safe_int(analysis_diag.get("llm_usable_claim_count")),
        "dropped_claim_count": _safe_int(analysis_diag.get("llm_dropped_claim_count")),
        "thin_report_risk": bool(correctness_summary.get("thin_report_risk")),
        "correctness_filter_summary": correctness_summary,
        "estimated_chars": _safe_int(report.get("estimated_chars") or len(_text(report.get("report_markdown")))),
        "total_tokens": run_metrics["total_tokens"],
        "duration_seconds": run_metrics["duration_seconds"],
        "cost_usd": run_metrics["cost_usd"],
        "repair_effectiveness": summarize_repair_effectiveness(writer_package=package, writer_report=report),
        "evidence_admission_summary": summarize_evidence_admission(fact_cards),
        "handoff_contract_summary": handoff,
        "handoff_failed_contracts": handoff_failed_contracts,
        "handoff_failed_contract_count": len(handoff_failed_contracts),
    }


def _is_publishable(snapshot: Dict[str, Any], *, min_publish_score: int) -> bool:
    report_status = _text(snapshot.get("report_status")).lower()
    if report_status == "diagnostic_only":
        return False
    return bool(
        not snapshot.get("final_audit_blocked")
        and _safe_int(snapshot.get("quality_score")) >= min_publish_score
        and (snapshot.get("formal_report_written") or report_status in {"final_clean", "formal_scored", "final"})
    )


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stddev(values: List[float]) -> float:
    if not values:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def summarize_topic_regression(
    snapshots: Iterable[Dict[str, Any]],
    *,
    min_publish_score: int = 70,
    min_pass_rate: float = 0.8,
    max_score_stddev: float = 5.0,
    max_tokens_per_run: int = 0,
    max_duration_seconds: int = 0,
) -> Dict[str, Any]:
    runs = [_as_dict(item) for item in snapshots if isinstance(item, dict)]
    scores = [_safe_float(item.get("quality_score")) for item in runs]
    tokens = [_safe_float(item.get("total_tokens")) for item in runs]
    durations = [_safe_float(item.get("duration_seconds")) for item in runs]
    costs = [_safe_float(item.get("cost_usd")) for item in runs]
    pass_count = sum(1 for item in runs if _is_publishable(item, min_publish_score=min_publish_score))
    pass_rate = pass_count / len(runs) if runs else 0.0
    score_mean = round(_mean(scores), 2)
    score_stddev = round(_stddev(scores), 2)
    fatal_counter: Counter[str] = Counter()
    handoff_counter: Counter[str] = Counter()
    for item in runs:
        fatal_counter.update(_as_list(item.get("fatal_types")))
        handoff_counter.update(_as_list(item.get("handoff_failed_contracts")))
    thin_count = sum(1 for item in runs if bool(item.get("thin_report_risk")))

    recommended_actions: List[str] = []
    if pass_rate < min_pass_rate:
        recommended_actions.append("raise_pass_rate")
    if score_stddev > max_score_stddev:
        recommended_actions.append("reduce_run_to_run_variance")
    if fatal_counter:
        recommended_actions.append("fix_recurring_fatal_findings")
    if thin_count:
        recommended_actions.append("inspect_filter_stacking")
    if handoff_counter:
        recommended_actions.append("inspect_handoff_contracts")
    if (max_tokens_per_run and any(value > max_tokens_per_run for value in tokens)) or (
        max_duration_seconds and any(value > max_duration_seconds for value in durations)
    ):
        recommended_actions.append("reduce_cost_or_latency")
    if not recommended_actions:
        recommended_actions.append("keep_monitoring")

    return {
        "schema_version": "topic_quality_regression_summary_v1",
        "topic_id": _text(runs[0].get("topic_id")) if runs else "unknown",
        "domain": _text(runs[0].get("domain")) if runs else "unknown",
        "run_count": len(runs),
        "pass_count": pass_count,
        "pass_rate": pass_rate,
        "score_mean": score_mean,
        "score_stddev": score_stddev,
        "score_min": min(scores) if scores else 0,
        "score_max": max(scores) if scores else 0,
        "score_range": round((max(scores) - min(scores)) if scores else 0, 2),
        "token_mean": round(_mean(tokens), 2),
        "token_max": max(tokens) if tokens else 0,
        "duration_seconds_mean": round(_mean(durations), 2),
        "duration_seconds_max": max(durations) if durations else 0,
        "cost_usd_total": round(sum(costs), 4),
        "cost_usd_mean": round(_mean(costs), 4),
        "fatal_type_counts": dict(fatal_counter),
        "handoff_failed_contract_counts": dict(handoff_counter),
        "thin_report_risk_count": thin_count,
        "stability_status": (
            "stable"
            if pass_rate >= min_pass_rate and score_stddev <= max_score_stddev and not fatal_counter and not handoff_counter
            else "unstable"
        ),
        "recommended_actions": recommended_actions,
        "runs": runs,
    }


def validate_golden_topic_suite(
    topics: Iterable[Dict[str, Any]],
    *,
    min_domains: int = 3,
    min_repeat_count: int = 2,
) -> Dict[str, Any]:
    normalized = [_as_dict(item) for item in topics if isinstance(item, dict)]
    domains = {
        _text(item.get("domain") or item.get("industry") or "unknown")
        for item in normalized
        if _text(item.get("domain") or item.get("industry") or "unknown") != "unknown"
    }
    planned_run_count = sum(max(1, _safe_int(item.get("repeat_count"), 1)) for item in normalized)
    low_repeat_topics = [
        _text(item.get("topic_id") or item.get("query") or f"topic_{index + 1}")
        for index, item in enumerate(normalized)
        if max(1, _safe_int(item.get("repeat_count"), 1)) < min_repeat_count
    ]
    missing_query_topics = [
        _text(item.get("topic_id") or f"topic_{index + 1}")
        for index, item in enumerate(normalized)
        if not _text(item.get("query"))
    ]
    issues: List[str] = []
    if len(domains) < min_domains:
        issues.append("insufficient_domain_coverage")
    if low_repeat_topics:
        issues.append("repeat_count_too_low")
    if missing_query_topics:
        issues.append("missing_query")
    if not normalized:
        issues.append("empty_suite")
    return {
        "schema_version": "golden_topic_suite_validation_v1",
        "status": "ready" if not issues else "not_ready",
        "topic_count": len(normalized),
        "domain_count": len(domains),
        "domains": sorted(domains),
        "planned_run_count": planned_run_count,
        "min_domains": min_domains,
        "min_repeat_count": min_repeat_count,
        "low_repeat_topics": low_repeat_topics,
        "missing_query_topics": missing_query_topics,
        "issues": issues,
    }


def load_quality_snapshots_from_paths(paths: Iterable[str | Path]) -> List[Dict[str, Any]]:
    snapshots: List[Dict[str, Any]] = []
    for path_value in paths:
        path = Path(path_value)
        payload = json.loads(path.read_text(encoding="utf-8"))
        package = _as_dict(payload.get("writer_package")) or _as_dict(payload)
        writer_report = _as_dict(payload.get("writer_report"))
        snapshot = build_run_quality_snapshot(package, writer_report=writer_report)
        if not snapshot.get("run_id"):
            snapshot["run_id"] = path.stem
        snapshot["source_path"] = str(path)
        snapshots.append(snapshot)
    return snapshots


def summarize_quality_regression_suite(
    snapshots: Iterable[Dict[str, Any]],
    *,
    min_publish_score: int = 70,
    min_pass_rate: float = 0.8,
    max_score_stddev: float = 5.0,
    max_tokens_per_run: int = 0,
    max_duration_seconds: int = 0,
) -> Dict[str, Any]:
    runs = [_as_dict(item) for item in snapshots if isinstance(item, dict)]
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for item in runs:
        topic_id = _text(item.get("topic_id")) or "unknown"
        grouped.setdefault(topic_id, []).append(item)
    topic_summaries = [
        summarize_topic_regression(
            topic_runs,
            min_publish_score=min_publish_score,
            min_pass_rate=min_pass_rate,
            max_score_stddev=max_score_stddev,
            max_tokens_per_run=max_tokens_per_run,
            max_duration_seconds=max_duration_seconds,
        )
        for topic_id, topic_runs in sorted(grouped.items())
    ]
    unstable_topics = [item["topic_id"] for item in topic_summaries if item.get("stability_status") != "stable"]
    return {
        "schema_version": "quality_regression_suite_summary_v1",
        "topic_count": len(topic_summaries),
        "run_count": len(runs),
        "overall_status": "stable" if topic_summaries and not unstable_topics else "unstable",
        "unstable_topics": unstable_topics,
        "topics": topic_summaries,
    }
