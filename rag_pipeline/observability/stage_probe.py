from __future__ import annotations

import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from rag_pipeline.observability.blueprint_evidence_alignment import build_blueprint_evidence_alignment
from rag_pipeline.observability.dataflow_inspector import build_dataflow_report, render_dataflow_summary
from rag_pipeline.observability.evidence_claim_conversion import build_evidence_claim_conversion_monitor
from rag_pipeline.observability.stage_contracts import validate_stage_packets

try:  # Root hygiene is diagnostic-only; stage probes must stay fail-open.
    from rag_pipeline.observability.data_root_hygiene import (
        inspect_cache_hygiene as _inspect_cache_hygiene,
        inspect_evidence_analysis_hygiene as _inspect_evidence_analysis_hygiene,
        inspect_source_identity_hygiene as _inspect_source_identity_hygiene,
    )
except Exception:  # pragma: no cover - optional diagnostics only.
    _inspect_cache_hygiene = None
    _inspect_evidence_analysis_hygiene = None
    _inspect_source_identity_hygiene = None

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "stage_probe_packet_v1"
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


def stage_probe_enabled() -> bool:
    return _env_flag("STAGE_PROBE_ENABLED", True)


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
    if text in {"total_tokens", "input_tokens", "output_tokens", "prompt_tokens", "completion_tokens"}:
        return False
    return any(fragment in text for fragment in SENSITIVE_KEY_FRAGMENTS)


def _sanitize(value: Any, *, key: str = "", depth: int = 0) -> Any:
    if _is_sensitive_key(key):
        return "[redacted]"
    if depth > 5:
        return "[truncated]"
    if isinstance(value, dict):
        return {str(k): _sanitize(v, key=str(k), depth=depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize(item, depth=depth + 1) for item in value[:40]]
    if isinstance(value, tuple):
        return [_sanitize(item, depth=depth + 1) for item in list(value)[:40]]
    if isinstance(value, str):
        text = value.strip()
        if len(text) > 500:
            return text[:500] + "...[truncated]"
        return text
    return value


def _first_dict(*values: Any) -> Dict[str, Any]:
    for value in values:
        payload = _as_dict(value)
        if payload:
            return payload
    return {}


def _nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def _nested_value(item: Dict[str, Any], field: str) -> Any:
    payload: Any = item
    for part in str(field or "").split("."):
        if not part:
            return None
        payload = _as_dict(payload).get(part)
        if payload is None:
            return None
    return payload


def _has_any(item: Dict[str, Any], fields: Sequence[str]) -> bool:
    return any(_nonempty(_nested_value(item, field) if "." in str(field) else item.get(field)) for field in fields)


def _public_body_char_count(markdown: str) -> int:
    text = str(markdown or "")
    if not text.strip():
        return 0
    text = text.split("\n## Sources", 1)[0]
    text = text.split("\n## 参考来源", 1)[0]
    text = text.split("\n## 来源", 1)[0]
    text = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    return len("".join(text.split()))


def _pick_id(item: Any, fields: Sequence[str]) -> str:
    payload = _as_dict(item)
    for field in fields:
        value = _nested_value(payload, field) if "." in str(field) else payload.get(field)
        if _nonempty(value):
            if isinstance(value, (list, tuple)):
                return str(value[0])
            return str(value)
    return ""


def _sample_ids(items: Iterable[Any], fields: Sequence[str], *, limit: int = 8) -> List[str]:
    output: List[str] = []
    seen: set[str] = set()
    for item in items:
        identifier = _pick_id(item, fields).strip()
        if not identifier or identifier in seen:
            continue
        seen.add(identifier)
        output.append(identifier[:160])
        if len(output) >= limit:
            break
    return output


def _id_coverage(items: Sequence[Any], field_map: Dict[str, Sequence[str]]) -> tuple[Dict[str, float], Dict[str, Dict[str, int]]]:
    denominator = len(items)
    coverage: Dict[str, float] = {}
    details: Dict[str, Dict[str, int]] = {}
    for label, fields in field_map.items():
        if denominator <= 0:
            coverage[label] = 1.0
            details[label] = {"present_count": 0, "missing_count": 0, "total_count": 0}
            continue
        present = 0
        for item in items:
            payload = _as_dict(item)
            if _has_any(payload, fields):
                present += 1
        missing = denominator - present
        coverage[label] = round(present / denominator, 4)
        details[label] = {"present_count": present, "missing_count": missing, "total_count": denominator}
    return coverage, details


def _source_level_counts(items: Iterable[Any]) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for item in items:
        payload = _as_dict(item)
        source = _as_dict(payload.get("source"))
        level = str(payload.get("source_level") or source.get("source_level") or "").strip() or "unknown"
        counts[level] += 1
    return dict(counts)


def _section_is_public_candidate(section: Dict[str, Any]) -> bool:
    if section.get("public_render") is False or section.get("omit_from_report"):
        return False
    status = str(section.get("composition_status") or section.get("body_composition_status") or "").strip().lower()
    return status not in {"dropped", "dropped_unreferenced_section", "omitted"}


def section_binding_diagnostics(sections: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    valid_sections = [
        section
        for section in list(sections or [])
        if isinstance(section, dict) and _section_is_public_candidate(section)
    ]
    missing = [
        str(section.get("section_id") or section.get("id") or section.get("claim_id") or "").strip()
        for section in valid_sections
        if not _has_any(section, ("used_fact_refs", "evidence_refs", "citation_refs", "fact_ids"))
    ]
    missing = [item for item in missing if item]
    section_count = len(valid_sections)
    backed = max(0, section_count - len(missing))
    return {
        "section_count": section_count,
        "evidence_backed_section_count": backed,
        "section_binding_rate": round(backed / section_count, 4) if section_count else 0.0,
        "missing_used_fact_refs": len(missing),
        "missing_examples": missing[:10],
    }


def _packet(
    *,
    run_id: str,
    stage: str,
    input_count: int,
    output_count: int,
    drop_count: Optional[int] = None,
    status: str = "ok",
    reason_counts: Optional[Dict[str, Any]] = None,
    id_coverage: Optional[Dict[str, float]] = None,
    id_coverage_details: Optional[Dict[str, Dict[str, int]]] = None,
    input_ids: Optional[Iterable[Any]] = None,
    output_ids: Optional[Iterable[Any]] = None,
    cache: Optional[Dict[str, Any]] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    safe_input = max(0, _safe_int(input_count))
    safe_output = max(0, _safe_int(output_count))
    safe_drop = max(0, _safe_int(drop_count if drop_count is not None else safe_input - safe_output))
    normalized_status = str(status or "ok").strip().lower()
    if normalized_status not in {"ok", "warning", "error", "degraded", "skipped"}:
        normalized_status = "warning"
    diag = _as_dict(diagnostics).copy()
    if id_coverage_details:
        diag["id_coverage_details"] = id_coverage_details
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": str(run_id or "run").strip() or "run",
        "stage": str(stage or "unknown").strip() or "unknown",
        "status": normalized_status,
        "input_count": safe_input,
        "output_count": safe_output,
        "drop_count": safe_drop,
        "reason_counts": _sanitize(reason_counts or {}),
        "id_coverage": id_coverage or {},
        "lineage": {
            "input_ids_sample": [str(item)[:160] for item in list(input_ids or [])[:8] if str(item or "").strip()],
            "output_ids_sample": [str(item)[:160] for item in list(output_ids or [])[:8] if str(item or "").strip()],
        },
        "cache": _sanitize(cache or {}),
        "diagnostics": _sanitize(diag),
        "diagnostic_only": True,
        "must_not_render": True,
        "public_text_allowed": False,
    }


def _raw_output_payload(writer_package: Dict[str, Any], writer_report: Dict[str, Any]) -> Dict[str, Any]:
    return _first_dict(
        writer_package.get("raw_output"),
        _as_dict(writer_report.get("render_artifacts")).get("raw_output"),
        _as_dict(writer_package.get("evidence_package")).get("raw_output"),
    )


def _raw_metadata(raw_output: Dict[str, Any], writer_package: Dict[str, Any]) -> Dict[str, Any]:
    return _first_dict(raw_output.get("metadata"), writer_package.get("metadata"), _as_dict(writer_package.get("writer_report")).get("metadata"))


def _structured_analysis(writer_package: Dict[str, Any], writer_report: Dict[str, Any]) -> Dict[str, Any]:
    return _first_dict(
        writer_package.get("structured_analysis"),
        writer_report.get("structured_analysis"),
        _as_dict(writer_report.get("render_artifacts")).get("structured_analysis"),
    )


def _chapter_packages(writer_package: Dict[str, Any], writer_report: Dict[str, Any]) -> List[Any]:
    return _as_list(writer_package.get("chapter_packages")) or _as_list(_as_dict(writer_report.get("render_artifacts")).get("chapter_packages"))


def _sections(chapter_packages: Iterable[Any]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for chapter in chapter_packages:
        payload = _as_dict(chapter)
        for section in _as_list(payload.get("sections")):
            section_payload = _as_dict(section).copy()
            if "chapter_id" not in section_payload and payload.get("chapter_id"):
                section_payload["chapter_id"] = payload.get("chapter_id")
            output.append(section_payload)
    return output


def _usable_claim_units(claim_units: Sequence[Any]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for item in claim_units:
        payload = _as_dict(item)
        has_source = _has_any(payload, ("source_ids", "citation_refs", "sources"))
        has_fact = _has_any(payload, ("fact_ids", "used_fact_refs", "evidence_refs", "basis_fact_ids"))
        if has_source and has_fact:
            output.append(payload)
    return output


def _evidence_items(evidence_package: Dict[str, Any]) -> tuple[List[Any], List[Any], List[Any]]:
    raw_items = _as_list(evidence_package.get("raw_data_points")) or _as_list(evidence_package.get("raw_evidence"))
    clean_items = _as_list(evidence_package.get("clean_evidence_list"))
    analysis_ready = _as_list(evidence_package.get("analysis_ready_evidence")) or clean_items
    return raw_items, clean_items, analysis_ready


def _source_identity_items(evidence_package: Dict[str, Any], clean_items: Sequence[Any], analysis_ready: Sequence[Any]) -> List[Any]:
    source_registry = _as_list(evidence_package.get("source_registry"))
    if source_registry:
        return source_registry
    return list(analysis_ready or clean_items)


def _fact_extractor_diag(writer_package: Dict[str, Any], writer_report: Dict[str, Any], raw_output: Dict[str, Any], evidence_package: Dict[str, Any]) -> Dict[str, Any]:
    return _first_dict(
        writer_package.get("fact_extractor"),
        writer_package.get("readpage_fact_extractor"),
        raw_output.get("fact_extractor"),
        raw_output.get("readpage_fact_extractor"),
        _as_dict(raw_output.get("metadata")).get("readpage_fact_extractor"),
        _as_dict(raw_output.get("metadata")).get("fact_extractor"),
        _as_dict(evidence_package.get("metadata")).get("readpage_fact_extractor"),
        writer_report.get("fact_extractor"),
        _as_dict(writer_report.get("metadata")).get("readpage_fact_extractor"),
        _as_dict(writer_report.get("render_artifacts")).get("fact_extractor"),
        _as_dict(_as_dict(writer_report.get("render_artifacts")).get("metadata")).get("readpage_fact_extractor"),
    )


def readpage_stage_diagnostics(
    *,
    metadata: Dict[str, Any],
    raw_output: Optional[Dict[str, Any]] = None,
    evidence_package: Optional[Dict[str, Any]] = None,
    writer_report: Optional[Dict[str, Any]] = None,
    fact_extractor_diag: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    raw_output = _as_dict(raw_output)
    evidence_package = _as_dict(evidence_package)
    writer_report = _as_dict(writer_report)
    evidence_metadata = _as_dict(evidence_package.get("metadata"))
    report_metadata = _as_dict(writer_report.get("metadata"))
    render_metadata = _as_dict(_as_dict(writer_report.get("render_artifacts")).get("metadata"))
    auto_readpage = _first_dict(
        metadata.get("auto_readpage"),
        _as_dict(raw_output.get("metadata")).get("auto_readpage"),
        evidence_metadata.get("auto_readpage"),
        report_metadata.get("auto_readpage"),
        render_metadata.get("auto_readpage"),
    )
    if auto_readpage:
        attempted = _safe_int(auto_readpage.get("attempted") or auto_readpage.get("attempted_count"))
        succeeded = _safe_int(auto_readpage.get("succeeded") or auto_readpage.get("success_count"))
        failed = _safe_int(auto_readpage.get("failed") or auto_readpage.get("error_count")) or max(0, attempted - succeeded)
        return {
            "source": "auto_readpage",
            "attempted": attempted,
            "succeeded": succeeded,
            "failed": failed,
            "enabled": auto_readpage.get("enabled"),
            "urls": _as_list(auto_readpage.get("urls")),
            "errors": _as_list(auto_readpage.get("errors"))[:5],
        }
    coverage = _first_dict(
        metadata.get("readpage_coverage"),
        _as_dict(raw_output.get("metadata")).get("readpage_coverage"),
        evidence_metadata.get("readpage_coverage"),
        report_metadata.get("readpage_coverage"),
        render_metadata.get("readpage_coverage"),
        metadata.get("evidence_health_summary"),
        evidence_metadata.get("evidence_health_summary"),
    )
    if coverage:
        attempted = _safe_int(coverage.get("readpage_attempted") or coverage.get("attempted") or coverage.get("attempted_count"))
        succeeded = _safe_int(coverage.get("readpage_succeeded") or coverage.get("succeeded") or coverage.get("success_count"))
        failed = _safe_int(coverage.get("failed") or coverage.get("error_count")) or max(0, attempted - succeeded)
        return {
            "source": "readpage_coverage",
            "attempted": attempted,
            "succeeded": succeeded,
            "failed": failed,
            "enabled": coverage.get("enabled"),
            "urls": _as_list(coverage.get("urls")),
            "errors": _as_list(coverage.get("errors"))[:5],
        }
    fact_diag = _as_dict(fact_extractor_diag)
    if fact_diag:
        attempted = _safe_int(fact_diag.get("readpage_attempted") or fact_diag.get("attempted") or fact_diag.get("attempted_count"))
        succeeded = _safe_int(fact_diag.get("readpage_succeeded") or fact_diag.get("success_count") or fact_diag.get("fact_card_count"))
        if attempted or succeeded:
            return {
                "source": "readpage_fact_extractor",
                "attempted": attempted or succeeded,
                "succeeded": succeeded,
                "failed": max(0, (attempted or succeeded) - succeeded),
                "enabled": fact_diag.get("enabled"),
                "urls": _as_list(fact_diag.get("urls")),
                "errors": _as_list(fact_diag.get("errors"))[:5],
            }
    return {}


def _body_rewrite_diag(chapter_items: Sequence[Any]) -> Dict[str, Any]:
    totals: Counter[str] = Counter()
    found = False
    for chapter in chapter_items:
        payload = _as_dict(chapter)
        for source in (_as_dict(payload.get("body_rewrite_global")), _as_dict(payload.get("body_rewrite"))):
            if source:
                found = True
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
                    totals[key] += _safe_int(source.get(key))
        for section in _as_list(payload.get("sections")):
            section_payload = _as_dict(section)
            source = _as_dict(section_payload.get("body_rewrite"))
            if source:
                found = True
                for key in ("success_count", "cache_hit_count", "fallback_count", "rejected_count", "skipped_count"):
                    totals[key] += _safe_int(source.get(key))
            status = str(section_payload.get("body_rewrite_status") or source.get("status") or "").strip()
            if status:
                found = True
                totals[f"{status}_section_count"] += 1
    return dict(totals) if found else {}


def _chapter_narrative_diag(writer_package: Dict[str, Any], writer_report: Dict[str, Any]) -> Dict[str, Any]:
    render_artifacts = _as_dict(writer_package.get("render_artifacts")) or _as_dict(writer_report.get("render_artifacts"))
    direct = _first_dict(
        writer_package.get("chapter_narrative"),
        writer_report.get("chapter_narrative"),
        render_artifacts.get("chapter_narrative"),
    )
    if direct:
        return direct
    totals: Counter[str] = Counter()
    found = False
    for chapter in _as_list(writer_package.get("chapter_packages")) or _as_list(render_artifacts.get("chapter_packages")):
        chapter_payload = _as_dict(chapter)
        chapter_status = str(chapter_payload.get("chapter_narrative_status") or "").strip()
        if chapter_status:
            found = True
            totals[f"{chapter_status}_chapter_count"] += 1
        for section in _as_list(chapter_payload.get("sections")):
            status = str(_as_dict(section).get("chapter_narrative_status") or "").strip()
            if status:
                found = True
                totals[f"{status}_section_count"] += 1
    return dict(totals) if found else {}


def _cache_hit_count_from_mapping(payload: Dict[str, Any]) -> int:
    total = 0
    for key, value in payload.items():
        if str(key).endswith("cache_hit_count") or str(key) == "cache_hit_count":
            total += _safe_int(value)
        elif str(key).endswith("_llm_cache_hit_count"):
            total += _safe_int(value)
    return total


def _collect_cache_payloads(writer_package: Dict[str, Any], writer_report: Dict[str, Any]) -> List[Dict[str, Any]]:
    package = _as_dict(writer_package)
    report = _as_dict(writer_report)
    evidence_package = _as_dict(package.get("evidence_package")) or _as_dict(report.get("evidence_package"))
    raw_output = _raw_output_payload(package, report)
    metadata = _raw_metadata(raw_output, package)
    candidates = [
        _as_dict(_as_dict(evidence_package.get("metadata")).get("evidence_cache_store")),
        _as_dict(metadata.get("evidence_cache_summary")),
        _as_dict(metadata.get("evidence_cache_store")),
        _as_dict(raw_output.get("evidence_cache_summary")),
        _as_dict(package.get("evidence_cache_summary")),
        _as_dict(report.get("evidence_cache_summary")),
    ]
    return [item for item in candidates if item]


def _merge_cache_hygiene(cache_payloads: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not _inspect_cache_hygiene:
        return {}
    summaries = [_inspect_cache_hygiene(payload) for payload in cache_payloads if payload]
    if not summaries:
        return {}
    reason_counts: Counter[str] = Counter()
    merged = {
        "schema_version": "data_root_cache_hygiene_v1",
        "cache_hit_count": 0,
        "cache_miss_count": 0,
        "stale_count": 0,
        "polluted_count": 0,
        "quarantined_count": 0,
        "dirty_hit_count": 0,
        "reason_counts": {},
        "diagnostic_only": True,
        "must_not_render": True,
        "public_text_allowed": False,
    }
    samples: List[Any] = []
    for summary in summaries:
        for key in ("cache_hit_count", "cache_miss_count", "stale_count", "polluted_count", "quarantined_count", "dirty_hit_count"):
            merged[key] += _safe_int(summary.get(key))
        reason_counts.update(_as_dict(summary.get("reason_counts")))
        samples.extend(_as_list(summary.get("dirty_samples")))
    merged["reason_counts"] = dict(reason_counts)
    if samples:
        merged["dirty_samples"] = samples[:8]
    return merged


def _append_cache_summary(packets: List[Dict[str, Any]], *, run_id: str, writer_package: Optional[Dict[str, Any]] = None, writer_report: Optional[Dict[str, Any]] = None) -> None:
    cache_hits = 0
    cache_misses = 0
    cache_sources: Dict[str, int] = {}
    for packet in packets:
        stage = str(packet.get("stage") or "unknown")
        cache_payload = _as_dict(packet.get("cache"))
        diag = _as_dict(packet.get("diagnostics"))
        stage_hits = _safe_int(cache_payload.get("cache_hit_count")) + _cache_hit_count_from_mapping(diag)
        stage_misses = _safe_int(cache_payload.get("cache_miss_count")) + _safe_int(diag.get("cache_miss_count"))
        if stage_hits:
            cache_sources[stage] = cache_sources.get(stage, 0) + stage_hits
        cache_hits += stage_hits
        cache_misses += stage_misses
    cache_hygiene = _merge_cache_hygiene(_collect_cache_payloads(writer_package or {}, writer_report or {}))
    cache_reason_counts = _as_dict(cache_hygiene.get("reason_counts"))
    if cache_hits or cache_misses or cache_hygiene:
        packets.append(
            _packet(
                run_id=run_id,
                stage="cache_summary",
                status="warning" if _safe_int(cache_hygiene.get("polluted_count")) or _safe_int(cache_hygiene.get("dirty_hit_count")) else "ok",
                input_count=cache_hits + cache_misses,
                output_count=cache_hits,
                drop_count=cache_misses,
                reason_counts=cache_reason_counts,
                cache={"cache_hit_count": cache_hits, "cache_miss_count": cache_misses},
                diagnostics={
                    "total_cache_hit_count": cache_hits,
                    "total_cache_miss_count": cache_misses,
                    "cache_hit_count_by_stage": cache_sources,
                    "cache_root_hygiene": cache_hygiene,
                },
            )
        )


def build_stage_probe_packets(
    *,
    run_id: str,
    writer_package: Dict[str, Any],
    writer_report: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Build diagnostic-only packets that describe data transfer across stages."""

    package = _as_dict(writer_package)
    report = _as_dict(writer_report) or _as_dict(package.get("writer_report"))
    raw_output = _raw_output_payload(package, report)
    metadata = _raw_metadata(raw_output, package)
    evidence_package = _as_dict(package.get("evidence_package")) or _as_dict(_as_dict(report.get("render_artifacts")).get("evidence_package"))
    top_level_chapter_packages = _as_list(package.get("chapter_evidence_packages"))
    if top_level_chapter_packages and not _as_list(evidence_package.get("chapter_evidence_packages")):
        evidence_package = {
            **evidence_package,
            "chapter_evidence_packages": top_level_chapter_packages,
        }
    report_blueprint = _first_dict(
        package.get("report_blueprint"),
        package.get("blueprint"),
        metadata.get("report_blueprint"),
        metadata.get("blueprint"),
        raw_output.get("report_blueprint"),
        raw_output.get("blueprint"),
        _as_dict(report.get("render_artifacts")).get("report_blueprint"),
    )
    packets: List[Dict[str, Any]] = []

    query_plan = _as_list(metadata.get("query_plan"))
    search_tasks = _as_list(metadata.get("search_tasks"))
    if query_plan or search_tasks:
        coverage, details = _id_coverage(search_tasks, {"task_id": ("task_id", "id"), "requirement_id": ("requirement_id",), "proof_role": ("proof_role", "lane_type")})
        packets.append(
            _packet(
                run_id=run_id,
                stage="search_plan",
                status="ok" if search_tasks else "warning",
                input_count=len(query_plan),
                output_count=len(search_tasks) or len(query_plan),
                id_coverage=coverage,
                id_coverage_details=details,
                input_ids=_sample_ids(query_plan, ("query", "task")),
                output_ids=_sample_ids(search_tasks, ("task_id", "query")),
                diagnostics={"query_plan_count": len(query_plan), "search_task_count": len(search_tasks)},
            )
        )

    if report_blueprint and evidence_package:
        try:
            alignment = build_blueprint_evidence_alignment(
                report_blueprint=report_blueprint,
                evidence_package=evidence_package,
            )
            warning_count = (
                _safe_int(alignment.get("chapter_starved_count"))
                + _safe_int(alignment.get("chapter_overloaded_count"))
                + _safe_int(alignment.get("chapter_misaligned_count"))
            )
            packets.append(
                _packet(
                    run_id=run_id,
                    stage="blueprint_evidence_alignment",
                    status="warning" if warning_count else "ok",
                    input_count=_safe_int(alignment.get("chapter_count")),
                    output_count=_safe_int(alignment.get("chapter_count")) - _safe_int(alignment.get("chapter_starved_count")),
                    drop_count=_safe_int(alignment.get("chapter_starved_count")),
                    reason_counts=_as_dict(alignment.get("warnings_by_type")),
                    diagnostics=alignment,
                )
            )
        except Exception as exc:  # pragma: no cover - diagnostics must never break report generation.
            logger.debug("blueprint evidence alignment probe failed: %s", exc)

    query_rewrite_diag = (
        _as_dict(metadata.get("query_rewrite_diagnostics"))
        or _as_dict(raw_output.get("query_rewrite_diagnostics"))
        or _as_dict(package.get("query_rewrite_diagnostics"))
    )
    if query_rewrite_diag:
        call_count = _safe_int(query_rewrite_diag.get("query_rewrite_call_count") or query_rewrite_diag.get("call_count"))
        rewritten_count = _safe_int(query_rewrite_diag.get("rewritten_query_count") or query_rewrite_diag.get("output_query_count")) or call_count
        cache_hits = _safe_int(query_rewrite_diag.get("query_rewrite_cache_hit_count") or query_rewrite_diag.get("cache_hit_count"))
        budget_exhausted = bool(query_rewrite_diag.get("query_rewrite_budget_exhausted") or query_rewrite_diag.get("budget_exhausted"))
        packets.append(
            _packet(
                run_id=run_id,
                stage="query_rewrite",
                status="warning" if budget_exhausted else ("ok" if call_count or cache_hits else "skipped"),
                input_count=call_count,
                output_count=rewritten_count,
                drop_count=1 if budget_exhausted else max(0, call_count - rewritten_count),
                reason_counts={"budget_exhausted": 1 if budget_exhausted else 0},
                cache={"cache_hit_count": cache_hits},
                diagnostics=query_rewrite_diag,
            )
        )

    repair_selection = _first_dict(
        metadata.get("repair_task_selection_summary"),
        raw_output.get("repair_task_selection_summary"),
        package.get("repair_task_selection_summary"),
        _as_dict(package.get("metadata")).get("repair_task_selection_summary"),
    )
    if repair_selection:
        selected_count = _safe_int(repair_selection.get("task_count")) or sum(_safe_int(value) for value in _as_dict(repair_selection.get("by_proof_role")).values())
        output_count = (
            _safe_int(repair_selection.get("post_policy_task_count"))
            or _safe_int(repair_selection.get("repair_task_count"))
            or _safe_int(repair_selection.get("planned_count"))
            or selected_count
        )
        exhausted_count = _safe_int(repair_selection.get("deep_budget_exhausted_count"))
        packets.append(
            _packet(
                run_id=run_id,
                stage="evidence_repair",
                status="warning" if exhausted_count else ("ok" if output_count else "skipped"),
                input_count=selected_count,
                output_count=output_count,
                drop_count=max(0, selected_count - output_count) + exhausted_count,
                reason_counts=_as_dict(repair_selection.get("by_proof_role")) or _as_dict(repair_selection.get("by_reason")),
                cache={"cache_hit_count": _safe_int(repair_selection.get("cache_hit_count"))},
                diagnostics=repair_selection,
            )
        )

    quality = _as_dict(raw_output.get("quality_processing")) or _as_dict(metadata.get("quality_processing"))
    if quality:
        raw_count = _safe_int(quality.get("raw_count"))
        normalized_count = _safe_int(quality.get("normalized_count") or quality.get("deduped_count"))
        filtered_count = _safe_int(quality.get("task_filtered_count"))
        packets.append(
            _packet(
                run_id=run_id,
                stage="web_result_filter",
                status="ok" if normalized_count else ("warning" if raw_count else "skipped"),
                input_count=raw_count,
                output_count=normalized_count,
                drop_count=filtered_count or max(0, raw_count - normalized_count),
                reason_counts=_as_dict(quality.get("task_filter_reasons")),
                diagnostics=quality,
            )
        )

    fact_diag = _fact_extractor_diag(package, report, raw_output, evidence_package)
    readpage_diag = readpage_stage_diagnostics(
        metadata=metadata,
        raw_output=raw_output,
        evidence_package=evidence_package,
        writer_report=report,
        fact_extractor_diag=fact_diag,
    )
    if readpage_diag:
        packets.append(
            _packet(
                run_id=run_id,
                stage="readpage",
                status="ok" if _safe_int(readpage_diag.get("failed")) == 0 else "warning",
                input_count=readpage_diag.get("attempted"),
                output_count=readpage_diag.get("succeeded"),
                drop_count=readpage_diag.get("failed"),
                output_ids=_as_list(readpage_diag.get("urls")),
                diagnostics={
                    "source": readpage_diag.get("source"),
                    "enabled": readpage_diag.get("enabled"),
                    "errors": _as_list(readpage_diag.get("errors"))[:5],
                },
            )
        )

    if fact_diag:
        attempted = _safe_int(fact_diag.get("attempted") or fact_diag.get("attempted_count"))
        fact_card_count = _safe_int(fact_diag.get("fact_card_count") or fact_diag.get("success_count"))
        rejected_count = _safe_int(fact_diag.get("rejected_span_count") or fact_diag.get("rejected_count"))
        error_count = _safe_int(fact_diag.get("llm_error_count") or fact_diag.get("error_count"))
        packets.append(
            _packet(
                run_id=run_id,
                stage="fact_extractor",
                status="warning" if error_count or rejected_count else ("ok" if fact_card_count else "skipped"),
                input_count=attempted,
                output_count=fact_card_count,
                drop_count=rejected_count + error_count,
                reason_counts={
                    "invalid_metric": fact_diag.get("invalid_metric_count", 0),
                    "llm_error": error_count,
                    "rejected_span": rejected_count,
                },
                cache={"cache_hit_count": _safe_int(fact_diag.get("cache_hit_count"))},
                diagnostics=fact_diag,
            )
        )

    raw_items, clean_items, analysis_ready = _evidence_items(evidence_package)
    if raw_items or clean_items or analysis_ready:
        source_identity_items = _source_identity_items(evidence_package, clean_items, analysis_ready)
        evidence_root_hygiene = _inspect_evidence_analysis_hygiene(analysis_ready) if _inspect_evidence_analysis_hygiene else {}
        source_identity_hygiene = _inspect_source_identity_hygiene(source_identity_items) if _inspect_source_identity_hygiene else {}
        coverage, details = _id_coverage(
            analysis_ready,
            {
                "fact_id": ("fact_id", "evidence_id", "id"),
                "chapter_id": ("chapter_id", "lineage.chapter_id"),
                "requirement_id": ("requirement_id", "requirement_ids", "lineage.requirement_id"),
                "search_task_id": ("search_task_id", "task_id", "lineage.search_task_id"),
                "source_id": ("source_id", "source_ref", "canonical_source_id", "lineage.source_id"),
                "proof_role": ("proof_role", "lineage.proof_role"),
                "analysis_role": ("analysis_role", "lineage.analysis_role"),
            },
        )
        packets.append(
            _packet(
                run_id=run_id,
                stage="evidence_merge",
                status="ok" if analysis_ready else "warning",
                input_count=len(raw_items),
                output_count=len(analysis_ready),
                drop_count=max(0, len(raw_items) - len(analysis_ready)),
                id_coverage=coverage,
                id_coverage_details=details,
                input_ids=_sample_ids(raw_items, ("raw_id", "id", "evidence_id")),
                output_ids=_sample_ids(analysis_ready, ("fact_id", "evidence_id", "id")),
                reason_counts=_as_dict(_as_dict(evidence_root_hygiene).get("reason_counts")),
                diagnostics={
                    "raw_count": len(raw_items),
                    "clean_evidence_count": len(clean_items),
                    "analysis_ready_count": len(analysis_ready),
                    "source_level_counts": _source_level_counts(analysis_ready),
                    "evidence_root_hygiene": evidence_root_hygiene,
                    "source_identity_hygiene": source_identity_hygiene,
                },
            )
        )

    structured = _structured_analysis(package, report)
    claim_units = _as_list(structured.get("claim_units"))
    analysis_diag = _as_dict(structured.get("analysis_stage_diagnostics")) or _as_dict(package.get("analysis_stage_diagnostics"))
    if analysis_diag or claim_units:
        raw_claim_count = _safe_int(analysis_diag.get("llm_raw_claim_count")) or len(claim_units)
        usable_count = _safe_int(analysis_diag.get("llm_usable_claim_count")) or len(claim_units)
        packets.append(
            _packet(
                run_id=run_id,
                stage="llm_analysis",
                status="ok" if usable_count else ("warning" if raw_claim_count else "skipped"),
                input_count=len(analysis_ready) if analysis_ready else _safe_int(analysis_diag.get("llm_input_chapter_count")),
                output_count=usable_count,
                drop_count=max(0, raw_claim_count - usable_count),
                reason_counts=_as_dict(analysis_diag.get("llm_validation_issue_counts")),
                diagnostics=analysis_diag,
            )
        )

    if claim_units:
        usable_claims = _usable_claim_units(claim_units)
        coverage, details = _id_coverage(
            claim_units,
            {
                "claim_id": ("claim_id", "id"),
                "requirement_ids": ("requirement_ids", "requirement_id"),
                "fact_ids": ("fact_ids", "used_fact_refs", "evidence_refs", "basis_fact_ids"),
                "source_ids": ("source_ids", "citation_refs", "sources"),
            },
        )
        packets.append(
            _packet(
                run_id=run_id,
                stage="claim_builder",
                status="ok" if usable_claims else "warning",
                input_count=len(claim_units),
                output_count=len(usable_claims),
                drop_count=max(0, len(claim_units) - len(usable_claims)),
                id_coverage=coverage,
                id_coverage_details=details,
                input_ids=_sample_ids(claim_units, ("claim_id", "id")),
                output_ids=_sample_ids(usable_claims, ("claim_id", "id")),
                diagnostics={"usable_claim_count": len(usable_claims), "raw_claim_count": len(claim_units)},
            )
        )

    chapter_items = _chapter_packages(package, report)
    section_items = _sections(chapter_items)
    if section_items:
        public_section_items = [
            section
            for section in section_items
            if isinstance(section, dict) and _section_is_public_candidate(section)
        ]
        binding_diag = section_binding_diagnostics(section_items)
        usable_sections = [
            section
            for section in public_section_items
            if _has_any(section, ("used_fact_refs", "evidence_refs", "citation_refs", "fact_ids"))
        ]
        coverage, details = _id_coverage(
            public_section_items,
            {
                "section_id": ("section_id", "id"),
                "claim_id": ("claim_id", "claim_ids"),
                "requirement_ids": ("requirement_ids", "requirement_id"),
                "used_fact_refs": ("used_fact_refs", "evidence_refs", "citation_refs", "fact_ids"),
            },
        )
        packets.append(
            _packet(
                run_id=run_id,
                stage="section_builder",
                status="ok" if usable_sections else "warning",
                input_count=len(public_section_items),
                output_count=len(usable_sections),
                drop_count=max(0, len(public_section_items) - len(usable_sections)),
                id_coverage=coverage,
                id_coverage_details=details,
                input_ids=_sample_ids(public_section_items, ("section_id", "id", "claim_id")),
                output_ids=_sample_ids(usable_sections, ("section_id", "id", "claim_id")),
                diagnostics={
                    "chapter_count": len(chapter_items),
                    "section_count": len(public_section_items),
                    "all_section_count": len(section_items),
                    "usable_section_count": len(usable_sections),
                    **binding_diag,
                },
            )
        )

    if evidence_package or claim_units or section_items:
        try:
            conversion_monitor = build_evidence_claim_conversion_monitor(
                writer_package=package,
                writer_report=report,
            )
            loss_reason_counts = _as_dict(conversion_monitor.get("loss_reason_counts"))
            loss_count = sum(_safe_int(value) for value in loss_reason_counts.values())
            totals = _as_dict(conversion_monitor.get("totals"))
            cache_summary = {
                "analysis_shard_cache_hit_count": _safe_int(totals.get("analysis_shard_cache_hit_count")),
                "analysis_shard_cache_miss_count": _safe_int(totals.get("analysis_shard_cache_miss_count")),
                "analysis_shard_output_cache_stored_count": _safe_int(totals.get("analysis_shard_output_cache_stored_count")),
                "analysis_shard_cache_saved_llm_call_count": _safe_int(totals.get("analysis_shard_cache_saved_llm_call_count")),
            }
            packets.append(
                _packet(
                    run_id=run_id,
                    stage="evidence_claim_conversion",
                    status="warning" if loss_count else "ok",
                    input_count=_safe_int(totals.get("analysis_ready_facts")),
                    output_count=_safe_int(totals.get("backed_sections")),
                    drop_count=loss_count,
                    reason_counts=loss_reason_counts,
                    cache=cache_summary,
                    diagnostics=conversion_monitor,
                )
            )
        except Exception as exc:  # pragma: no cover - diagnostics must never block report generation.
            logger.debug("evidence claim conversion monitor failed: %s", exc)

    body_diag = _body_rewrite_diag(chapter_items)
    if body_diag:
        submitted = _safe_int(body_diag.get("submitted_count") or body_diag.get("called_count"))
        output_count = _safe_int(body_diag.get("success_count")) + _safe_int(body_diag.get("cache_hit_count"))
        drop_count = _safe_int(body_diag.get("fallback_count")) + _safe_int(body_diag.get("rejected_count")) + _safe_int(body_diag.get("budget_exhausted_count"))
        packets.append(
            _packet(
                run_id=run_id,
                stage="body_rewrite",
                status="ok" if output_count else ("warning" if drop_count else "skipped"),
                input_count=submitted,
                output_count=output_count,
                drop_count=drop_count,
                reason_counts={"fallback": body_diag.get("fallback_count", 0), "rejected": body_diag.get("rejected_count", 0), "budget_exhausted": body_diag.get("budget_exhausted_count", 0)},
                cache={"cache_hit_count": _safe_int(body_diag.get("cache_hit_count"))},
                diagnostics=body_diag,
            )
        )

    narrative_diag = _chapter_narrative_diag(package, report)
    if narrative_diag:
        attempted = _safe_int(narrative_diag.get("attempted_count") or narrative_diag.get("input_chapter_count"))
        output_count = _safe_int(narrative_diag.get("success_count")) + _safe_int(narrative_diag.get("cache_hit_count"))
        drop_count = _safe_int(narrative_diag.get("fallback_count")) + _safe_int(narrative_diag.get("rejected_count"))
        packets.append(
            _packet(
                run_id=run_id,
                stage="chapter_narrative",
                status="ok" if output_count else ("warning" if drop_count else "skipped"),
                input_count=attempted,
                output_count=output_count,
                drop_count=drop_count,
                reason_counts={**_as_dict(narrative_diag.get("rejected_reasons")), **_as_dict(narrative_diag.get("failure_reasons"))},
                cache={"cache_hit_count": _safe_int(narrative_diag.get("cache_hit_count"))},
                diagnostics=narrative_diag,
            )
        )

    public_gate = _first_dict(
        package.get("public_narrative_leak_audit"),
        report.get("public_narrative_leak_audit"),
        _as_dict(report.get("render_artifacts")).get("public_narrative_leak_audit"),
    )
    if public_gate:
        input_count = _safe_int(public_gate.get("public_narrative_leak_input_count"))
        removed = _safe_int(public_gate.get("public_narrative_leak_removed_count"))
        skipped = _safe_int(public_gate.get("skipped_global_block_count"))
        remaining = _safe_int(public_gate.get("public_narrative_leak_remaining_count"))
        packets.append(
            _packet(
                run_id=run_id,
                stage="public_narrative_gate",
                status="ok" if remaining == 0 else "warning",
                input_count=input_count,
                output_count=max(0, input_count - removed - skipped),
                drop_count=removed + skipped,
                reason_counts=_as_dict(public_gate.get("public_narrative_leak_reason_counts")),
                diagnostics=public_gate,
            )
        )

    table_packages = (
        _as_list(package.get("table_packages"))
        or _as_list(report.get("table_packages"))
        or _as_list(_as_dict(report.get("render_artifacts")).get("table_packages"))
        or _as_list(_as_dict(package.get("render_artifacts")).get("table_packages"))
    )
    if table_packages:
        rendered_tables = [
            table
            for table in table_packages
            if isinstance(table, dict) and table.get("should_render") and not table.get("appendix_only")
        ]
        reason_counts: Counter[str] = Counter()
        for table in table_packages:
            payload = _as_dict(table)
            if payload in rendered_tables:
                continue
            reasons = _as_list(payload.get("reject_reasons")) or _as_list(payload.get("invalid_reasons")) or ["not_rendered"]
            for reason in reasons:
                reason_counts[str(reason or "not_rendered")] += 1
        coverage, details = _id_coverage(table_packages, {"table_id": ("table_id", "id"), "chapter_id": ("chapter_id",), "source_refs": ("source_refs", "citation_refs", "evidence_refs")})
        packets.append(
            _packet(
                run_id=run_id,
                stage="table_builder",
                status="ok" if rendered_tables else "warning",
                input_count=len(table_packages),
                output_count=len(rendered_tables),
                drop_count=max(0, len(table_packages) - len(rendered_tables)),
                reason_counts=dict(reason_counts),
                id_coverage=coverage,
                id_coverage_details=details,
                input_ids=_sample_ids(table_packages, ("table_id", "id")),
                output_ids=_sample_ids(rendered_tables, ("table_id", "id")),
                diagnostics={"table_package_count": len(table_packages), "rendered_table_count": len(rendered_tables)},
            )
        )

    qa_result = _first_dict(package.get("qa_result"), report.get("qa_result"), report.get("validation"))
    if qa_result:
        passed = bool(qa_result.get("passed"))
        errors = _as_list(qa_result.get("errors")) + _as_list(qa_result.get("issues"))
        warnings = _as_list(qa_result.get("warnings"))
        packets.append(
            _packet(
                run_id=run_id,
                stage="qa_score",
                status="ok" if passed else "warning",
                input_count=1,
                output_count=1 if passed else 0,
                drop_count=0 if passed else 1,
                reason_counts={"error": len(errors), "warning": len(warnings), "rewrite_required": 1 if qa_result.get("rewrite_required") else 0},
                diagnostics={
                    "passed": passed,
                    "quality_score": qa_result.get("quality_score"),
                    "error_count": len(errors),
                    "warning_count": len(warnings),
                    "rewrite_required": bool(qa_result.get("rewrite_required")),
                    "clean_gate": _as_dict(qa_result.get("clean_gate")),
                    "render_gate": _as_dict(qa_result.get("render_gate")),
                },
            )
        )

    final_citation_audit = _first_dict(
        package.get("final_citation_audit"),
        report.get("final_citation_audit"),
        _as_dict(report.get("render_artifacts")).get("final_citation_audit"),
    )
    if final_citation_audit:
        body_refs = _as_list(final_citation_audit.get("final_body_citation_refs"))
        appendix_refs = _as_list(final_citation_audit.get("final_appendix_refs"))
        missing_refs = _as_list(final_citation_audit.get("final_missing_appendix_refs"))
        removed_count = _safe_int(final_citation_audit.get("final_unresolved_citation_removed_count")) + _safe_int(final_citation_audit.get("final_duplicate_citation_removed_count"))
        factual_without_refs = _safe_int(final_citation_audit.get("factual_body_without_citations_count"))
        status_text = str(final_citation_audit.get("final_citation_reconciliation_status") or "").strip().lower()
        drop_count = len(missing_refs) + removed_count + factual_without_refs
        packets.append(
            _packet(
                run_id=run_id,
                stage="final_citation_audit",
                status="ok" if status_text in {"", "ok"} and drop_count == 0 else "warning",
                input_count=len(body_refs),
                output_count=len(appendix_refs),
                drop_count=drop_count,
                reason_counts={
                    "missing_appendix_ref": len(missing_refs),
                    "removed_unresolved_or_duplicate": removed_count,
                    "factual_body_without_citation": factual_without_refs,
                },
                input_ids=body_refs,
                output_ids=appendix_refs,
                diagnostics=final_citation_audit,
            )
        )

    source_gate = _first_dict(
        package.get("source_claim_support"),
        report.get("source_claim_support"),
        _as_dict(report.get("render_artifacts")).get("source_claim_support"),
    )
    if source_gate:
        checked = _safe_int(source_gate.get("checked_source_count") or source_gate.get("input_source_count") or source_gate.get("cited_source_count"))
        supported = _safe_int(source_gate.get("supported_source_count") or source_gate.get("output_source_count") or source_gate.get("retained_source_count"))
        omitted = _safe_int(source_gate.get("empty_chapter_omitted_after_source_gate_count"))
        packets.append(
            _packet(
                run_id=run_id,
                stage="source_gate",
                status="ok" if supported else "warning",
                input_count=checked,
                output_count=supported,
                drop_count=max(0, checked - supported) + omitted,
                reason_counts={"empty_chapter_omitted": omitted},
                diagnostics=source_gate,
            )
        )

    reformatter = _first_dict(
        package.get("reformatter_result"),
        report.get("reformatter_result"),
        package.get("reformatter"),
        report.get("reformatter"),
    )
    if reformatter:
        attempted = _safe_int(reformatter.get("attempted")) or int(bool(reformatter.get("status")))
        passed = bool(reformatter.get("passed")) or str(reformatter.get("status") or "").strip().lower() in {"passed", "ok", "completed"}
        repair_attempts = _safe_int(reformatter.get("repair_attempt_count") or reformatter.get("polish_attempt_count"))
        packets.append(
            _packet(
                run_id=run_id,
                stage="reformatter",
                status="ok" if passed else "warning",
                input_count=attempted,
                output_count=1 if passed else 0,
                drop_count=0 if passed else 1,
                reason_counts={"repair_attempt": repair_attempts},
                diagnostics=reformatter,
            )
        )

    artifact_ledger = _first_dict(package.get("artifact_ledger"), report.get("artifact_ledger"))
    if artifact_ledger:
        artifact_count = _safe_int(
            artifact_ledger.get("artifact_count")
            or artifact_ledger.get("synced_artifact_count")
            or artifact_ledger.get("stored_artifact_count")
        )
        status_text = str(artifact_ledger.get("status") or "").strip().lower()
        packets.append(
            _packet(
                run_id=run_id,
                stage="artifact_ledger",
                status="ok" if artifact_ledger.get("enabled") and status_text in {"", "ok", "completed", "clean_blocked", "review_required"} else "warning",
                input_count=artifact_count,
                output_count=artifact_count,
                drop_count=0,
                diagnostics=artifact_ledger,
            )
        )

    snapshot_index = _as_list(package.get("stage_snapshot_index")) or _as_list(report.get("stage_snapshot_index"))
    if snapshot_index:
        replayable = [item for item in snapshot_index if _as_dict(item).get("replayable")]
        packets.append(
            _packet(
                run_id=run_id,
                stage="stage_snapshot",
                status="ok" if replayable else "warning",
                input_count=len(snapshot_index),
                output_count=len(replayable),
                drop_count=max(0, len(snapshot_index) - len(replayable)),
                reason_counts={"not_replayable": max(0, len(snapshot_index) - len(replayable))},
                input_ids=_sample_ids(snapshot_index, ("stage_name", "stage")),
                output_ids=_sample_ids(replayable, ("stage_name", "stage")),
                diagnostics={"snapshot_count": len(snapshot_index), "replayable_count": len(replayable)},
            )
        )

    score_gaps = _as_list(package.get("score_gaps")) or _as_list(report.get("score_gaps"))
    if score_gaps:
        status_counts: Counter[str] = Counter()
        for gap in score_gaps:
            status_counts[str(_as_dict(gap).get("status") or "unknown")] += 1
        closed = status_counts.get("closed", 0) + status_counts.get("evidence_found", 0) + status_counts.get("resolved", 0)
        packets.append(
            _packet(
                run_id=run_id,
                stage="score_gap_ledger",
                status="warning" if status_counts.get("still_insufficient", 0) else "ok",
                input_count=len(score_gaps),
                output_count=closed,
                drop_count=max(0, len(score_gaps) - closed),
                reason_counts=dict(status_counts),
                input_ids=_sample_ids(score_gaps, ("gap_id", "requirement_id")),
                output_ids=_sample_ids([gap for gap in score_gaps if str(_as_dict(gap).get("status") or "") in {"closed", "evidence_found", "resolved"}], ("gap_id", "requirement_id")),
                diagnostics={"gap_count": len(score_gaps), "status_counts": dict(status_counts)},
            )
        )

    citation = _first_dict(package.get("citation_manifest"), _as_dict(report.get("render_artifacts")).get("citation_manifest"))
    if citation:
        missing = _as_list(citation.get("missing_evidence_refs"))
        excluded = _as_list(citation.get("excluded_cited_sources"))
        packets.append(
            _packet(
                run_id=run_id,
                stage="citation_manifest",
                status="ok" if not missing and not excluded else "warning",
                input_count=len(_as_dict(citation.get("section_citation_refs"))),
                output_count=len(_as_list(citation.get("appendix_sources"))),
                drop_count=len(missing) + len(excluded) + _safe_int(citation.get("orphan_citation_count")),
                reason_counts={
                    "missing_evidence_ref": len(missing),
                    "excluded_cited_source": len(excluded),
                    "orphan_citation": citation.get("orphan_citation_count", 0),
                },
                output_ids=_sample_ids(_as_list(citation.get("appendix_sources")), ("source_id", "source_ref", "url")),
                diagnostics=citation,
            )
        )

    final_audit = _first_dict(package.get("final_audit_result"), report.get("final_audit_result"))
    if final_audit:
        packets.append(
            _packet(
                run_id=run_id,
                stage="final_audit",
                status="error" if final_audit.get("blocked") else "ok",
                input_count=1,
                output_count=0 if final_audit.get("blocked") else 1,
                drop_count=1 if final_audit.get("blocked") else 0,
                reason_counts={"blocked": 1 if final_audit.get("blocked") else 0},
                diagnostics=final_audit,
            )
        )

    delivery = _as_dict(package.get("report_delivery_status"))
    has_report = bool(delivery.get("formal_report_written") or report.get("formal_report_path") or report.get("quality_score") is not None)
    report_health = _first_dict(package.get("report_health"), report.get("report_health"))
    report_markdown = str(report.get("report_markdown") or package.get("report_markdown") or "")
    body_char_count = _safe_int(report_health.get("body_char_count")) or _public_body_char_count(report_markdown)
    target_body_chars = _safe_int(report_health.get("target_body_chars") or report.get("target_body_chars") or package.get("target_body_chars"))
    packets.append(
        _packet(
            run_id=run_id,
            stage="writer",
            status="ok" if has_report else "warning",
            input_count=1,
            output_count=1 if has_report else 0,
            drop_count=0 if has_report else 1,
            diagnostics={
                "formal_report_written": delivery.get("formal_report_written"),
                "quality_score": delivery.get("quality_score") or report.get("quality_score"),
                "final_audit_status": _as_dict(report.get("final_audit_result")).get("status") or _as_dict(package.get("final_audit_result")).get("status"),
                "final_body_char_count": body_char_count,
                "target_body_chars": target_body_chars,
                "body_char_gap": max(0, target_body_chars - body_char_count) if target_body_chars else 0,
            },
        )
    )

    _append_cache_summary(packets, run_id=run_id, writer_package=package, writer_report=report)
    return packets


def write_stage_probe_from_package(
    *,
    run_id: str,
    output_dir: Path | str,
    writer_package: Dict[str, Any],
    writer_report: Optional[Dict[str, Any]] = None,
    base_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Write stage-probe JSONL and dataflow summary sidecars.

    This function is fail-open and diagnostic-only. It must never block report
    generation, mutate the writer package, or create content intended for the
    public report body.
    """

    if not stage_probe_enabled():
        return {"enabled": False, "stage_probe_path": "", "dataflow_summary_path": "", "packet_count": 0}
    safe_run_id = str(run_id or "run").strip() or "run"
    safe_base_name = str(base_name or safe_run_id).strip() or safe_run_id
    root = Path(output_dir)
    jsonl_path = root / f"{safe_base_name}.stage_probe.jsonl"
    summary_path = root / f"{safe_base_name}.dataflow_summary.md"
    try:
        root.mkdir(parents=True, exist_ok=True)
        packets = build_stage_probe_packets(run_id=safe_run_id, writer_package=writer_package, writer_report=writer_report)
        validation = validate_stage_packets(packets)
        report = build_dataflow_report(packets)
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for packet in packets:
                handle.write(json.dumps(packet, ensure_ascii=False, sort_keys=True) + "\n")
        summary_path.write_text(render_dataflow_summary(report), encoding="utf-8")
        module_probe_result: Dict[str, Any] = {
            "enabled": False,
            "module_probe_path": "",
            "lineage_graph_path": "",
            "health_metrics_path": "",
            "event_count": 0,
        }
        try:
            from rag_pipeline.observability.module_probe_writer import write_module_probe_from_package

            module_probe_result = write_module_probe_from_package(
                run_id=safe_run_id,
                output_dir=root,
                writer_package=writer_package,
                writer_report=writer_report,
                base_name=safe_base_name,
                stage_packets=packets,
            )
        except Exception as exc:  # pragma: no cover - nested observability must not block report generation.
            logger.warning("Module probe sidecar failed", extra={"run_id": safe_run_id, "error": str(exc)})
            module_probe_result = {
                "enabled": False,
                "module_probe_path": "",
                "lineage_graph_path": "",
                "health_metrics_path": "",
                "event_count": 0,
                "error": str(exc),
            }
        return {
            "enabled": True,
            "run_id": safe_run_id,
            "stage_probe_path": str(jsonl_path),
            "dataflow_summary_path": str(summary_path),
            "packet_count": len(packets),
            "validation": validation,
            "module_probe_enabled": bool(module_probe_result.get("enabled")),
            "module_probe_path": str(module_probe_result.get("module_probe_path") or ""),
            "lineage_graph_path": str(module_probe_result.get("lineage_graph_path") or ""),
            "health_metrics_path": str(module_probe_result.get("health_metrics_path") or ""),
            "module_probe_event_count": _safe_int(module_probe_result.get("event_count")),
        }
    except Exception as exc:  # pragma: no cover - observability must not block report generation.
        logger.warning("Stage probe write failed", extra={"run_id": safe_run_id, "error": str(exc)})
        return {
            "enabled": False,
            "run_id": safe_run_id,
            "stage_probe_path": str(jsonl_path),
            "dataflow_summary_path": str(summary_path),
            "packet_count": 0,
            "error": str(exc),
        }
