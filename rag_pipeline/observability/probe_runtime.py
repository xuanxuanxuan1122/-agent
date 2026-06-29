from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

from rag_pipeline.observability.module_probe_models import as_dict, safe_int, safe_rate


ROOT_HYGIENE_KEYS = (
    ("source", "source_root_hygiene"),
    ("evidence", "evidence_root_hygiene"),
    ("cache", "cache_root_hygiene"),
    ("source_identity", "source_identity_hygiene"),
)


def load_runtime_events(path: Path | str) -> List[Dict[str, Any]]:
    event_path = Path(path)
    if not event_path.exists():
        return []
    events: List[Dict[str, Any]] = []
    for line in event_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            item = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            events.append(item)
    return events


def _transform_events(events: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [event for event in events if str(event.get("event_type") or "") == "transform_result"]


def _hygiene_issue_count(scope: str, payload: Dict[str, Any]) -> int:
    if scope == "cache":
        return (
            safe_int(payload.get("dirty_hit_count"))
            + safe_int(payload.get("polluted_count"))
            + safe_int(payload.get("quarantined_count"))
        )
    return safe_int(payload.get("dirty_item_count")) + sum(
        safe_int(value) for value in as_dict(payload.get("missing_lineage_counts")).values()
    )


def _merge_hygiene_payload(target: Dict[str, Any], scope: str, payload: Dict[str, Any]) -> None:
    if not payload:
        return
    if scope == "cache":
        for key in ("cache_hit_count", "cache_miss_count", "stale_count", "polluted_count", "quarantined_count", "dirty_hit_count"):
            target[key] = safe_int(target.get(key)) + safe_int(payload.get(key))
    else:
        for key in ("item_count", "dirty_item_count", "clean_item_count"):
            target[key] = safe_int(target.get(key)) + safe_int(payload.get(key))
        missing = target.setdefault("missing_lineage_counts", {})
        for key, value in as_dict(payload.get("missing_lineage_counts")).items():
            missing[str(key)] = safe_int(missing.get(str(key))) + safe_int(value)


def _summarize_root_hygiene(transforms: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    by_stage: Dict[str, Dict[str, Any]] = {}
    issue_count_by_stage: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    earliest_issue_stage = ""
    for event in transforms:
        stage = str(event.get("stage") or "unknown")
        diagnostics = as_dict(event.get("diagnostics"))
        for scope, key in ROOT_HYGIENE_KEYS:
            payload = as_dict(diagnostics.get(key))
            if not payload:
                continue
            stage_entry = by_stage.setdefault(stage, {})
            scope_entry = stage_entry.setdefault(scope, {})
            _merge_hygiene_payload(scope_entry, scope, payload)
            for reason, count in as_dict(payload.get("reason_counts")).items():
                reason_counts[f"{scope}:{reason}"] += safe_int(count, 1)
            issue_count = _hygiene_issue_count(scope, payload)
            if issue_count > 0:
                issue_count_by_stage[stage] += issue_count
                if not earliest_issue_stage:
                    earliest_issue_stage = stage
    return {
        "schema_version": "root_hygiene_summary_v1",
        "status": "warning" if issue_count_by_stage else "ok",
        "earliest_issue_stage": earliest_issue_stage,
        "issue_count_by_stage": dict(issue_count_by_stage),
        "reason_counts": dict(reason_counts.most_common(30)),
        "by_stage": by_stage,
        "diagnostic_only": True,
        "must_not_render": True,
        "public_text_allowed": False,
    }


def summarize_runtime_events(path: Path | str) -> Dict[str, Any]:
    events = load_runtime_events(path)
    transforms = _transform_events(events)
    stage_counts: Dict[str, Dict[str, int]] = {}
    reason_counts: Counter[str] = Counter()
    cache_hit_count = 0
    metrics: Dict[str, int] = {
        "readpage_attempted": 0,
        "readpage_succeeded": 0,
        "final_body_char_count": 0,
        "target_body_chars": 0,
        "llm_raw_claim_count": 0,
        "llm_usable_claim_count": 0,
    }
    for event in transforms:
        stage = str(event.get("stage") or "unknown")
        diagnostics = as_dict(event.get("diagnostics"))
        row = stage_counts.setdefault(stage, {"input_count": 0, "output_count": 0, "drop_count": 0})
        row["input_count"] += safe_int(as_dict(event.get("input")).get("count"))
        row["output_count"] += safe_int(as_dict(event.get("output")).get("count"))
        row["drop_count"] += safe_int(as_dict(event.get("drop")).get("count"))
        cache_hit_count += safe_int(as_dict(event.get("cache")).get("hit_count"))
        for reason, count in as_dict(as_dict(event.get("drop")).get("reason_counts")).items():
            reason_counts[str(reason)] += safe_int(count, 1)
        health = (
            as_dict(diagnostics.get("evidence_health_summary"))
            or as_dict(diagnostics.get("readpage_coverage"))
            or diagnostics
        )
        metrics["readpage_attempted"] += safe_int(
            health.get("readpage_attempted")
            or health.get("attempted")
            or health.get("attempted_count")
        )
        metrics["readpage_succeeded"] += safe_int(
            health.get("readpage_succeeded")
            or health.get("succeeded")
            or health.get("success_count")
        )
        body_chars = safe_int(
            diagnostics.get("final_body_char_count")
            or diagnostics.get("body_char_count")
            or as_dict(diagnostics.get("report_health")).get("body_char_count")
        )
        if body_chars:
            metrics["final_body_char_count"] = max(metrics["final_body_char_count"], body_chars)
        target_chars = safe_int(
            diagnostics.get("target_body_chars")
            or as_dict(diagnostics.get("report_health")).get("target_body_chars")
        )
        if target_chars:
            metrics["target_body_chars"] = max(metrics["target_body_chars"], target_chars)
        if stage in {"analysis_agent", "llm_analysis"}:
            metrics["llm_raw_claim_count"] += safe_int(diagnostics.get("llm_raw_claim_count"))
            metrics["llm_usable_claim_count"] += safe_int(diagnostics.get("llm_usable_claim_count"))

    def stage_rate(*stages: str) -> float:
        input_count = 0
        output_count = 0
        for stage in stages:
            row = stage_counts.get(stage, {})
            input_count += safe_int(row.get("input_count"))
            output_count += safe_int(row.get("output_count"))
        return safe_rate(output_count, input_count)

    readpage_success_rate = stage_rate("readpage")
    if safe_int(stage_counts.get("readpage", {}).get("input_count")) <= 0 and metrics["readpage_attempted"] > 0:
        readpage_success_rate = safe_rate(metrics["readpage_succeeded"], metrics["readpage_attempted"])
    analysis_output_count = safe_int(stage_counts.get("analysis_agent", {}).get("output_count")) + safe_int(stage_counts.get("llm_analysis", {}).get("output_count"))
    evidence_ready_count = safe_int(stage_counts.get("evidence_merger", {}).get("output_count")) + safe_int(stage_counts.get("evidence_merge", {}).get("output_count"))
    fact_to_claim_rate = safe_rate(analysis_output_count, evidence_ready_count or (safe_int(stage_counts.get("analysis_agent", {}).get("input_count")) + safe_int(stage_counts.get("llm_analysis", {}).get("input_count"))))
    llm_raw_claim_count = metrics["llm_raw_claim_count"] or analysis_output_count
    llm_usable_claim_count = metrics["llm_usable_claim_count"] or analysis_output_count
    llm_claim_acceptance_rate = safe_rate(llm_usable_claim_count, llm_raw_claim_count)
    claim_binding_rate = stage_rate("claim_builder")

    return {
        "schema_version": "runtime_probe_summary_v1",
        "event_count": len(events),
        "transform_event_count": len(transforms),
        "stage_counts": stage_counts,
        "rates": {
            "readpage_success_rate": readpage_success_rate,
            "fact_extraction_rate": stage_rate("fact_extractor"),
            "analysis_ready_rate": stage_rate("evidence_merger", "evidence_merge"),
            "claim_conversion_rate": fact_to_claim_rate,
            "fact_to_claim_rate": fact_to_claim_rate,
            "llm_claim_acceptance_rate": llm_claim_acceptance_rate,
            "claim_binding_rate": claim_binding_rate,
            "bound_claim_rate": claim_binding_rate,
            "section_binding_rate": stage_rate("section_builder"),
            "repair_success_rate": stage_rate("evidence_repair"),
        },
        "metrics": metrics,
        "cache_hit_count": cache_hit_count,
        "top_reason_counts": dict(reason_counts.most_common(20)),
        "root_hygiene": _summarize_root_hygiene(transforms),
        "diagnostic_only": True,
        "must_not_render": True,
        "public_text_allowed": False,
    }
