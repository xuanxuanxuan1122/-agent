from __future__ import annotations

import json
import logging
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

VALID_STATUSES = {"ok", "warning", "error", "degraded", "skipped"}
SENSITIVE_KEY_FRAGMENTS = (
    "api_key",
    "apikey",
    "authorization",
    "password",
    "secret",
    "token",
    "prompt",
    "raw_html",
    "raw_text",
    "raw_content",
    "maintext",
    "page_content",
)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, min_value: int = 0, max_value: int = 10_000) -> int:
    try:
        value = int(str(os.getenv(name, default)).strip())
    except (TypeError, ValueError):
        value = default
    return max(min_value, min(max_value, value))


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if value is None:
        return []
    return [value]


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return default


def _is_sensitive_key(key: Any) -> bool:
    text = str(key or "").strip().lower()
    return any(fragment in text for fragment in SENSITIVE_KEY_FRAGMENTS)


def _sanitize(value: Any, *, key: str = "", depth: int = 0) -> Any:
    if _is_sensitive_key(key):
        return "[redacted]"
    if depth > 6:
        return "[truncated]"
    if isinstance(value, dict):
        return {str(k): _sanitize(v, key=str(k), depth=depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize(item, depth=depth + 1) for item in value[:50]]
    if isinstance(value, tuple):
        return [_sanitize(item, depth=depth + 1) for item in list(value)[:50]]
    if isinstance(value, str):
        text = value.strip()
        if len(text) > 600:
            return text[:600] + "...[truncated]"
        return text
    return value


def _sample(values: Iterable[Any], limit: int) -> List[str]:
    output: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        output.append(text[:160])
        if len(output) >= limit:
            break
    return output


def trace_enabled() -> bool:
    return _env_flag("RUN_TRACE_ENABLED", True)


class RunTraceContext:
    """Append-only JSONL trace writer for a single report run.

    The context is deliberately fail-open: all filesystem failures are logged
    as warnings and never propagate into the report pipeline.
    """

    def __init__(self, *, run_id: str, output_dir: Path | str, base_name: Optional[str] = None) -> None:
        self.run_id = str(run_id or "run").strip() or "run"
        self.output_dir = Path(output_dir)
        self.base_name = str(base_name or self.run_id).strip() or self.run_id
        self.trace_path = self.output_dir / f"{self.base_name}.trace.jsonl"
        self.summary_path = self.output_dir / f"{self.base_name}.trace_summary.md"
        self.sample_limit = _env_int("RUN_TRACE_SAMPLE_LIMIT", 5, min_value=0, max_value=50)
        self.enabled = trace_enabled()
        self._seq = 0
        self._events: List[Dict[str, Any]] = []

    def emit(
        self,
        *,
        stage: str,
        event: str = "completed",
        status: str = "ok",
        duration_ms: int | float = 0,
        input_count: int | float = 0,
        output_count: int | float = 0,
        drop_count: int | float = 0,
        reason_counts: Optional[Dict[str, Any]] = None,
        sample_ids: Optional[Iterable[Any]] = None,
        message: str = "",
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        normalized_status = str(status or "ok").strip().lower()
        if normalized_status not in VALID_STATUSES:
            normalized_status = "warning"
        self._seq += 1
        payload = {
            "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
            "run_id": self.run_id,
            "seq": self._seq,
            "stage": str(stage or "unknown").strip() or "unknown",
            "event": str(event or "completed").strip() or "completed",
            "status": normalized_status,
            "duration_ms": _safe_int(duration_ms),
            "input_count": _safe_int(input_count),
            "output_count": _safe_int(output_count),
            "drop_count": _safe_int(drop_count),
            "reason_counts": _sanitize(reason_counts or {}),
            "sample_ids": _sample(sample_ids or [], self.sample_limit),
            "message": str(message or "").strip(),
            "diagnostics": _sanitize(diagnostics or {}),
        }
        self._events.append(payload)
        if not self.enabled:
            return payload
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with self.trace_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        except Exception as exc:  # pragma: no cover - filesystem failures must not block delivery.
            logger.warning("Run trace write failed", extra={"error": str(exc), "run_id": self.run_id})
        return payload

    def write_summary(self, *, final_status: str = "") -> Path:
        if not self.enabled or not _env_flag("RUN_TRACE_WRITE_SUMMARY", True):
            return self.summary_path
        lines = _render_summary(self._events, run_id=self.run_id, final_status=final_status)
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.summary_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        except Exception as exc:  # pragma: no cover - filesystem failures must not block delivery.
            logger.warning("Run trace summary write failed", extra={"error": str(exc), "run_id": self.run_id})
        return self.summary_path


def _render_summary(events: List[Dict[str, Any]], *, run_id: str, final_status: str = "") -> List[str]:
    status = final_status or (events[-1].get("status") if events else "unknown")
    lines = [
        f"# Run Trace Summary: {run_id}",
        "",
        "## Run Overview",
        f"- run_id: {run_id}",
        f"- final_status: {status}",
        f"- event_count: {len(events)}",
        "",
        "## Stage Funnel",
        "| Stage | Input | Output | Drop | Status |",
        "|---|---:|---:|---:|---|",
    ]
    for item in events:
        lines.append(
            f"| {item.get('stage')} | {_safe_int(item.get('input_count'))} | {_safe_int(item.get('output_count'))} | {_safe_int(item.get('drop_count'))} | {item.get('status')} |"
        )

    reason_counter: Counter[str] = Counter()
    for item in events:
        for reason, count in _as_dict(item.get("reason_counts")).items():
            reason_counter[str(reason)] += _safe_int(count, 1)
    lines.extend(["", "## Top Failure Reasons"])
    if reason_counter:
        for reason, count in reason_counter.most_common(10):
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- None recorded.")

    lines.extend(["", "## Actionable Next Checks"])
    checks = _actionable_checks(events)
    lines.extend(f"- {item}" for item in checks)
    return lines


def _actionable_checks(events: List[Dict[str, Any]]) -> List[str]:
    checks: List[str] = []
    for item in events:
        stage = str(item.get("stage") or "")
        input_count = _safe_int(item.get("input_count"))
        output_count = _safe_int(item.get("output_count"))
        drop_count = _safe_int(item.get("drop_count"))
        status = str(item.get("status") or "")
        if stage == "web_result_filter" and input_count > 0 and output_count == 0:
            checks.append(f"web_result_filter accepted 0 of {input_count}; inspect task filter thresholds and reason_counts.")
        elif stage == "llm_analysis" and input_count > 0 and output_count == 0:
            checks.append("llm_analysis produced no usable claims; inspect validation issue counts and cached raw outputs.")
        elif stage == "layout_planning" and drop_count > output_count:
            checks.append("layout_planning dropped more blocks than it rendered; inspect block affinity and claim/block matching.")
        elif stage == "citation_manifest" and status in {"warning", "error", "degraded"}:
            checks.append("citation_manifest is not ok; inspect missing refs, orphan citations, and excluded cited sources.")
    if not checks:
        checks.append("No immediate trace blockers detected.")
    return checks[:8]


def _source_level_counts(items: Iterable[Any]) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for item in items:
        payload = _as_dict(item)
        level = str(payload.get("source_level") or _as_dict(payload.get("source")).get("source_level") or "").strip() or "unknown"
        counts[level] += 1
    return dict(counts)


def _first_existing_dict(*values: Any) -> Dict[str, Any]:
    for value in values:
        payload = _as_dict(value)
        if payload:
            return payload
    return {}


def _fact_extractor_diag(writer_package: Dict[str, Any], writer_report: Dict[str, Any]) -> Dict[str, Any]:
    metadata = _as_dict(writer_package.get("metadata"))
    raw_output = _as_dict(writer_package.get("raw_output"))
    evidence_package = _as_dict(writer_package.get("evidence_package"))
    render_artifacts = _as_dict(writer_report.get("render_artifacts"))
    return _first_existing_dict(
        writer_package.get("fact_extractor"),
        writer_package.get("readpage_fact_extractor"),
        writer_report.get("fact_extractor"),
        metadata.get("readpage_fact_extractor"),
        raw_output.get("fact_extractor"),
        _as_dict(evidence_package.get("metadata")).get("readpage_fact_extractor"),
        _as_dict(render_artifacts.get("metadata")).get("readpage_fact_extractor"),
    )


def _analysis_diag(writer_package: Dict[str, Any], writer_report: Dict[str, Any]) -> Dict[str, Any]:
    structured = _as_dict(writer_package.get("structured_analysis")) or _as_dict(writer_report.get("structured_analysis"))
    return _first_existing_dict(
        writer_package.get("analysis_stage_diagnostics"),
        structured.get("analysis_stage_diagnostics"),
        _as_dict(writer_report.get("render_artifacts")).get("analysis_stage_diagnostics"),
    )


def _body_rewrite_diag(writer_package: Dict[str, Any]) -> Dict[str, Any]:
    global_diag: Dict[str, Any] = {}
    totals: Counter[str] = Counter()
    for chapter in _as_list(writer_package.get("chapter_packages")):
        payload = _as_dict(chapter)
        body_global = _as_dict(payload.get("body_rewrite_global"))
        if body_global and not global_diag:
            global_diag = body_global
        body = _as_dict(payload.get("body_rewrite"))
        for key in (
            "submitted_count",
            "called_count",
            "success_count",
            "cache_hit_count",
            "fallback_count",
            "rejected_count",
            "skipped_count",
            "budget_exhausted_count",
        ):
            totals[key] += _safe_int(body.get(key))
        for section in _as_list(payload.get("sections")):
            section_body = _as_dict(_as_dict(section).get("body_rewrite"))
            status = str(_as_dict(section).get("body_rewrite_status") or section_body.get("status") or "").strip()
            if status:
                totals[f"{status}_section_count"] += 1
    if global_diag:
        return global_diag
    return dict(totals)


def _chapter_narrative_diag(writer_package: Dict[str, Any], writer_report: Dict[str, Any]) -> Dict[str, Any]:
    render_artifacts = _as_dict(writer_report.get("render_artifacts")) or _as_dict(writer_package.get("render_artifacts"))
    direct = _first_existing_dict(
        writer_package.get("chapter_narrative"),
        writer_report.get("chapter_narrative"),
        render_artifacts.get("chapter_narrative"),
    )
    if direct:
        return direct

    totals: Counter[str] = Counter()
    for chapter in _as_list(writer_package.get("chapter_packages")) or _as_list(render_artifacts.get("chapter_packages")):
        payload = _as_dict(chapter)
        status = str(payload.get("chapter_narrative_status") or "").strip()
        if status:
            totals[f"{status}_chapter_count"] += 1
        for section in _as_list(payload.get("sections")):
            section_payload = _as_dict(section)
            section_status = str(section_payload.get("chapter_narrative_status") or "").strip()
            if section_status:
                totals[f"{section_status}_section_count"] += 1
    return dict(totals)


def _raw_output_payload(writer_package: Dict[str, Any], writer_report: Dict[str, Any]) -> Dict[str, Any]:
    return _first_existing_dict(
        writer_package.get("raw_output"),
        _as_dict(writer_report.get("render_artifacts")).get("raw_output"),
        _as_dict(writer_package.get("evidence_package")).get("raw_output"),
    )


def _raw_output_metadata(raw_output: Dict[str, Any], writer_package: Dict[str, Any]) -> Dict[str, Any]:
    return _first_existing_dict(
        raw_output.get("metadata"),
        writer_package.get("metadata"),
        _as_dict(writer_package.get("writer_report")).get("metadata"),
    )


def _repair_diagnostics(
    *,
    raw_metadata: Dict[str, Any],
    raw_output: Dict[str, Any],
    writer_package: Dict[str, Any],
    writer_report: Dict[str, Any],
    quality_posture: Dict[str, Any],
) -> Dict[str, Any]:
    metadata = _as_dict(writer_package.get("metadata"))
    render_artifacts = _as_dict(writer_report.get("render_artifacts"))
    selected = _first_existing_dict(
        raw_metadata.get("repair_task_selection_summary"),
        raw_output.get("repair_task_selection_summary"),
        metadata.get("repair_task_selection_summary"),
        writer_package.get("repair_task_selection_summary"),
        render_artifacts.get("repair_task_selection_summary"),
    )
    planned = _first_existing_dict(
        raw_metadata.get("openai_web_search_summary"),
        raw_output.get("openai_web_search_summary"),
        metadata.get("openai_web_search_summary"),
        writer_package.get("openai_web_search_summary"),
        render_artifacts.get("openai_web_search_summary"),
    )
    if not selected and not planned:
        return {}
    selected_by_reason = _as_dict(selected.get("by_proof_role")) or _as_dict(selected.get("by_reason"))
    planned_by_reason = (
        _as_dict(planned.get("last_planned_by_proof_role"))
        or _as_dict(planned.get("planned_by_proof_role"))
        or _as_dict(planned.get("by_proof_role"))
    )
    actual_count = (
        _safe_int(planned.get("gap_repair_task_count"))
        or _safe_int(planned.get("planned_count"))
        or _safe_int(selected.get("post_policy_task_count"))
    )
    selected_count = _safe_int(selected.get("task_count")) or sum(_safe_int(value) for value in selected_by_reason.values())
    budget_reason = str(planned.get("last_skip_reason") or selected.get("last_skip_reason") or "").strip().lower()
    budget_exhausted = bool(planned.get("budget_exhausted") or selected.get("budget_exhausted") or "budget" in budget_reason)
    return {
        "selected_repair_task_count": selected_count,
        "repair_task_count": actual_count or selected_count,
        "selected_repair_task_count_by_reason": selected_by_reason,
        "repair_task_count_by_reason": planned_by_reason or selected_by_reason,
        "repair_budget_exhausted": budget_exhausted,
        "last_skip_reason": planned.get("last_skip_reason") or selected.get("last_skip_reason") or "",
        "deep_budget_exhausted_count": selected.get("deep_budget_exhausted_count") or planned.get("deep_budget_exhausted_count") or 0,
        "self_refine_disabled_reason": "quality_posture" if _as_dict(quality_posture.get("disabled")).get("self_refine") else "",
        "selected_summary": selected,
        "planned_summary": planned,
    }


def write_run_trace_from_package(
    *,
    run_id: str,
    output_dir: Path | str,
    writer_package: Dict[str, Any],
    writer_report: Optional[Dict[str, Any]] = None,
    final_status: str = "",
    base_name: Optional[str] = None,
) -> Dict[str, Any]:
    if not trace_enabled():
        return {"enabled": False, "trace_path": "", "summary_path": ""}

    package = _as_dict(writer_package)
    report = _as_dict(writer_report) or _as_dict(package.get("writer_report"))
    context = RunTraceContext(run_id=run_id, output_dir=Path(output_dir), base_name=base_name)
    execution_mode = str(package.get("report_execution_mode") or report.get("report_execution_mode") or "").strip()
    quality_mode = bool(package.get("quality_mode") or report.get("quality_mode"))
    context.emit(
        stage="run_start",
        event="started",
        status="ok",
        message="Report run trace started",
        diagnostics={"report_execution_mode": execution_mode, "quality_mode": quality_mode},
    )

    raw_output = _raw_output_payload(package, report)
    raw_metadata = _raw_output_metadata(raw_output, package)
    quality_posture = _as_dict(package.get("quality_posture")) or _as_dict(report.get("quality_posture"))
    query_plan = _as_list(raw_metadata.get("query_plan"))
    search_tasks = _as_list(raw_metadata.get("search_tasks"))
    if query_plan or search_tasks:
        lane_counts: Counter[str] = Counter()
        for task in search_tasks:
            payload = _as_dict(task)
            lane = str(payload.get("lane_type") or payload.get("proof_role") or payload.get("source") or "unknown").strip() or "unknown"
            lane_counts[lane] += 1
        context.emit(
            stage="search_plan",
            status="ok" if search_tasks or query_plan else "skipped",
            input_count=len(query_plan),
            output_count=len(search_tasks) or len(query_plan),
            diagnostics={
                "lane_counts": dict(lane_counts),
                "query_plan_count": len(query_plan),
                "search_task_count": len(search_tasks),
                "quality_posture": quality_posture,
            },
        )

    query_rewrite_diag = (
        _as_dict(raw_metadata.get("query_rewrite_diagnostics"))
        or _as_dict(raw_output.get("query_rewrite_diagnostics"))
        or _as_dict(package.get("query_rewrite_diagnostics"))
    )
    if query_rewrite_diag or quality_posture:
        query_rewrite_disabled = bool(_as_dict(quality_posture.get("disabled")).get("query_rewrite"))
        self_refine_disabled = bool(_as_dict(quality_posture.get("disabled")).get("self_refine"))
        context.emit(
            stage="query_rewrite",
            status="warning" if query_rewrite_diag.get("query_rewrite_budget_exhausted") else ("skipped" if query_rewrite_disabled else "ok"),
            input_count=query_rewrite_diag.get("query_rewrite_call_count", 0),
            output_count=query_rewrite_diag.get("query_rewrite_call_count", 0),
            drop_count=1 if query_rewrite_diag.get("query_rewrite_budget_exhausted") else 0,
            reason_counts={
                "budget_exhausted": 1 if query_rewrite_diag.get("query_rewrite_budget_exhausted") else 0,
                "self_refine_disabled": 1 if self_refine_disabled else 0,
            },
            diagnostics={
                **query_rewrite_diag,
                "self_refine_disabled_reason": "quality_posture" if self_refine_disabled else "",
                "quality_posture": quality_posture,
            },
        )

    repair_diag = _repair_diagnostics(
        raw_metadata=raw_metadata,
        raw_output=raw_output,
        writer_package=package,
        writer_report=report,
        quality_posture=quality_posture,
    )
    if repair_diag:
        context.emit(
            stage="evidence_repair",
            status="warning" if repair_diag.get("repair_budget_exhausted") else "ok",
            input_count=repair_diag.get("selected_repair_task_count"),
            output_count=repair_diag.get("repair_task_count"),
            drop_count=repair_diag.get("deep_budget_exhausted_count") or 0,
            reason_counts=_as_dict(repair_diag.get("repair_task_count_by_reason")),
            diagnostics=repair_diag,
        )

    quality = _as_dict(raw_output.get("quality_processing")) or _as_dict(raw_metadata.get("quality_processing"))
    if quality:
        raw_count = _safe_int(quality.get("raw_count"))
        accepted = _safe_int(quality.get("normalized_count") or quality.get("deduped_count"))
        filtered = _safe_int(quality.get("task_filtered_count"))
        context.emit(
            stage="iqs_search",
            status="ok" if raw_count else "warning",
            input_count=len(search_tasks) or len(query_plan),
            output_count=raw_count,
            diagnostics={"raw_count": raw_count, "errors": _as_list(raw_output.get("errors"))[:5]},
        )
        context.emit(
            stage="web_result_filter",
            status="ok" if accepted else ("warning" if raw_count else "skipped"),
            input_count=raw_count,
            output_count=accepted,
            drop_count=filtered,
            reason_counts=_as_dict(quality.get("task_filter_reasons")),
            diagnostics=quality,
        )

    auto_readpage = _as_dict(raw_metadata.get("auto_readpage"))
    if auto_readpage:
        context.emit(
            stage="readpage",
            status="ok" if _safe_int(auto_readpage.get("failed")) == 0 else "warning",
            input_count=auto_readpage.get("attempted"),
            output_count=auto_readpage.get("succeeded"),
            drop_count=auto_readpage.get("failed"),
            sample_ids=_as_list(auto_readpage.get("urls")),
            diagnostics={"errors": _as_list(auto_readpage.get("errors"))[:5], "enabled": auto_readpage.get("enabled")},
        )

    fact_diag = _fact_extractor_diag(package, report)
    if fact_diag:
        context.emit(
            stage="fact_extractor",
            status="ok" if _safe_int(fact_diag.get("llm_error_count")) == 0 else "warning",
            input_count=fact_diag.get("attempted"),
            output_count=fact_diag.get("fact_card_count"),
            drop_count=fact_diag.get("rejected_span_count"),
            reason_counts={
                "invalid_metric": fact_diag.get("invalid_metric_count", 0),
                "llm_error": fact_diag.get("llm_error_count", 0),
                "cache_hit": fact_diag.get("cache_hit_count", 0),
            },
            diagnostics=fact_diag,
        )

    evidence_package = _as_dict(package.get("evidence_package")) or _as_dict(_as_dict(report.get("render_artifacts")).get("evidence_package"))
    if evidence_package:
        raw_items = _as_list(evidence_package.get("raw_data_points")) or _as_list(evidence_package.get("raw_evidence"))
        analysis_ready = _as_list(evidence_package.get("analysis_ready_evidence"))
        clean_items = _as_list(evidence_package.get("clean_evidence_list"))
        context.emit(
            stage="evidence_merge",
            status="ok" if analysis_ready or clean_items else "warning",
            input_count=len(raw_items),
            output_count=len(analysis_ready) or len(clean_items),
            drop_count=max(0, len(raw_items) - (len(analysis_ready) or len(clean_items))),
            diagnostics={
                "analysis_ready_count": len(analysis_ready),
                "clean_evidence_count": len(clean_items),
                "source_level_counts": _source_level_counts(analysis_ready or clean_items),
            },
        )

    chapter_packages = _as_list(package.get("chapter_packages")) or _as_list(_as_dict(report.get("render_artifacts")).get("chapter_packages"))
    if chapter_packages:
        rendered_sections = sum(len(_as_list(_as_dict(chapter).get("sections"))) for chapter in chapter_packages)
        evidence_backed = sum(
            1
            for chapter in chapter_packages
            for section in _as_list(_as_dict(chapter).get("sections"))
            if _as_dict(section).get("evidence_backed")
        )
        context.emit(
            stage="layout_planning",
            status="ok" if rendered_sections else "warning",
            input_count=sum(len(_as_list(_as_dict(chapter).get("planned_blocks"))) for chapter in chapter_packages),
            output_count=rendered_sections,
            drop_count=sum(len(_as_list(_as_dict(chapter).get("dropped_blocks"))) for chapter in chapter_packages),
            diagnostics={"chapter_count": len(chapter_packages), "evidence_backed_section_count": evidence_backed},
        )

    analysis_diag = _analysis_diag(package, report)
    if analysis_diag:
        issue_counts = _as_dict(analysis_diag.get("llm_validation_issue_counts"))
        input_count = (
            _safe_int(analysis_diag.get("llm_input_chapter_count"))
            or _safe_int(analysis_diag.get("llm_raw_chapter_count"))
            or _safe_int(analysis_diag.get("input_chapter_count"))
        )
        output_count = _safe_int(analysis_diag.get("llm_usable_claim_count"))
        dropped = _safe_int(analysis_diag.get("llm_dropped_claim_count"))
        context.emit(
            stage="llm_analysis",
            status="ok" if output_count else ("warning" if analysis_diag.get("llm_analysis_attempted") else "skipped"),
            input_count=input_count,
            output_count=output_count,
            drop_count=dropped,
            reason_counts=issue_counts,
            diagnostics={
                "llm_analysis_status": analysis_diag.get("llm_analysis_status"),
                "final_analysis_source": analysis_diag.get("final_analysis_source"),
                "llm_valid_chapter_count": analysis_diag.get("llm_valid_chapter_count"),
                "llm_failed_chapter_count": analysis_diag.get("llm_failed_chapter_count"),
                "quality_path_degraded": analysis_diag.get("quality_path_degraded"),
                "llm_validation_issue_counts": issue_counts,
            },
        )

    body_diag = _body_rewrite_diag(package)
    if body_diag:
        context.emit(
            stage="body_rewrite",
            status="ok" if _safe_int(body_diag.get("success_count")) or _safe_int(body_diag.get("cache_hit_count")) else "skipped",
            input_count=body_diag.get("submitted_count") or body_diag.get("called_count"),
            output_count=_safe_int(body_diag.get("success_count")) + _safe_int(body_diag.get("cache_hit_count")),
            drop_count=_safe_int(body_diag.get("fallback_count")) + _safe_int(body_diag.get("rejected_count")),
            reason_counts=_as_dict(body_diag.get("failure_reasons")),
            diagnostics=body_diag,
        )

    narrative_diag = _chapter_narrative_diag(package, report)
    if narrative_diag:
        success_count = _safe_int(narrative_diag.get("success_count")) + _safe_int(narrative_diag.get("cache_hit_count"))
        fallback_count = _safe_int(narrative_diag.get("fallback_count")) + _safe_int(narrative_diag.get("rejected_count"))
        skipped_reason = str(narrative_diag.get("skipped_reason") or "").strip()
        status = "ok" if success_count and not fallback_count else ("warning" if success_count or fallback_count else "skipped")
        context.emit(
            stage="chapter_narrative",
            status=status,
            input_count=narrative_diag.get("attempted_count") or narrative_diag.get("input_chapter_count"),
            output_count=success_count,
            drop_count=fallback_count,
            reason_counts={
                **_as_dict(narrative_diag.get("rejected_reasons")),
                **_as_dict(narrative_diag.get("failure_reasons")),
                **({"skipped": 1} if skipped_reason else {}),
            },
            sample_ids=_as_list(narrative_diag.get("failed_chapter_ids")),
            diagnostics=narrative_diag,
        )

    public_narrative_diag = (
        _as_dict(package.get("public_narrative_leak_audit"))
        or _as_dict(report.get("public_narrative_leak_audit"))
        or _as_dict(_as_dict(report.get("render_artifacts")).get("public_narrative_leak_audit"))
    )
    if public_narrative_diag:
        remaining = _safe_int(public_narrative_diag.get("public_narrative_leak_remaining_count"))
        removed = _safe_int(public_narrative_diag.get("public_narrative_leak_removed_count"))
        skipped_blocks = _safe_int(public_narrative_diag.get("skipped_global_block_count"))
        context.emit(
            stage="public_narrative_gate",
            status="ok" if remaining == 0 else "warning",
            input_count=public_narrative_diag.get("public_narrative_leak_input_count"),
            output_count=max(0, _safe_int(public_narrative_diag.get("public_narrative_leak_input_count")) - removed),
            drop_count=removed + skipped_blocks,
            reason_counts=_as_dict(public_narrative_diag.get("public_narrative_leak_reason_counts")),
            sample_ids=[
                _as_dict(item).get("reason") or _as_dict(item).get("text")
                for item in _as_list(public_narrative_diag.get("public_narrative_leak_examples"))
            ],
            diagnostics=public_narrative_diag,
        )

    citation = _first_existing_dict(package.get("citation_manifest"), _as_dict(report.get("render_artifacts")).get("citation_manifest"))
    if citation:
        missing = _as_list(citation.get("missing_evidence_refs"))
        excluded = _as_list(citation.get("excluded_cited_sources"))
        status = str(citation.get("citation_manifest_status") or "ok").strip().lower()
        context.emit(
            stage="citation_manifest",
            status="ok" if status == "ok" and not missing and not excluded else "warning",
            input_count=len(_as_dict(citation.get("section_citation_refs"))),
            output_count=len(_as_list(citation.get("appendix_sources"))),
            drop_count=len(missing) + len(excluded) + _safe_int(citation.get("orphan_citation_count")),
            reason_counts={
                "missing_evidence_ref": len(missing),
                "excluded_cited_source": len(excluded),
                "orphan_citation": citation.get("orphan_citation_count", 0),
            },
            sample_ids=missing,
            diagnostics=citation,
        )

    final_audit = _as_dict(package.get("final_audit_result")) or _as_dict(report.get("final_audit_result"))
    if final_audit:
        context.emit(
            stage="final_audit",
            status="error" if final_audit.get("blocked") else "ok",
            input_count=1,
            output_count=0 if final_audit.get("blocked") else 1,
            drop_count=1 if final_audit.get("blocked") else 0,
            reason_counts={"blocked": 1 if final_audit.get("blocked") else 0},
            diagnostics=final_audit,
        )

    delivery = _as_dict(package.get("report_delivery_status"))
    context.emit(
        stage="writer",
        status="ok" if delivery.get("formal_report_written") or report.get("formal_report_path") else "warning",
        input_count=1,
        output_count=int(bool(delivery.get("formal_report_written") or report.get("formal_report_path"))),
        drop_count=0,
        diagnostics={
            "formal_report_written": delivery.get("formal_report_written"),
            "score_report_written": delivery.get("score_report_written"),
            "quality_score": delivery.get("quality_score") or report.get("quality_score"),
            "clean_report_written": delivery.get("clean_report_written"),
            "clean_report_eligible": delivery.get("clean_report_eligible") or report.get("clean_report_eligible"),
        },
    )

    context.write_summary(final_status=final_status or "completed")
    return {
        "enabled": True,
        "run_id": context.run_id,
        "trace_path": str(context.trace_path),
        "summary_path": str(context.summary_path),
        "event_count": len(context._events),
    }
