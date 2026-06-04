"""Phase 0 观测：聚合各缓存的命中/规模为一个只读 cache_report。

设计原则（见 docs/CACHE_OPTIMIZATION_PLAN.md §0）：
- 纯只读聚合，不修改任何既有 payload；
- 每个数据源各自 try/except，单源失败不影响整体（fail-open）；
- 产物用于阶段间 diff，证明后续改动「全流程跑通、无回归」。
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


SCHEMA_VERSION = 1


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def cache_report_enabled() -> bool:
    raw = os.getenv("CACHE_REPORT_SIDECAR_ENABLED")
    if raw is None:
        return True
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


def _evidence_cache_section() -> Dict[str, Any]:
    try:
        from rag_pipeline.cache.evidence_cache import (
            evidence_cache_activity_summary,
            get_evidence_cache,
        )

        return {
            "activity": _as_dict(evidence_cache_activity_summary()),
            "stats": _as_dict(get_evidence_cache().stats()),
        }
    except Exception as exc:  # fail-open
        return {"error": str(exc)}


def _stage_snapshot_section(run_id: str, stage_snapshot_index: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    try:
        index: List[Dict[str, Any]] = [dict(_as_dict(item)) for item in _as_list(stage_snapshot_index)]
        if not index:
            from rag_pipeline.cache.stage_snapshot_cache import list_stage_snapshots

            for manifest in list_stage_snapshots(run_id):
                manifest = _as_dict(manifest)
                index.append(
                    {
                        "stage_name": manifest.get("stage_name"),
                        "stored": True,
                        "replayable": bool(manifest.get("replayable")),
                        "full_payload_bytes": manifest.get("full_payload_bytes"),
                    }
                )
        stored = [item for item in index if item.get("stored")]
        replayable = [item for item in index if item.get("replayable")]
        return {
            "count": len(index),
            "stored_count": len(stored),
            "replayable_count": len(replayable),
            "stages": [str(item.get("stage_name") or "") for item in index],
            "index": index,
        }
    except Exception as exc:  # fail-open
        return {"error": str(exc)}


def _trusted_source_section() -> Dict[str, Any]:
    try:
        from rag_pipeline.cache.trusted_source_cache import trusted_source_stats

        return _as_dict(trusted_source_stats())
    except Exception as exc:  # fail-open
        return {"error": str(exc)}


def _topic_bundle_section(topic_bundle: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    bundle = _as_dict(topic_bundle)
    section: Dict[str, Any] = {
        "hit": bool(bundle.get("hit")),
        "used_for_skip_search": bool(bundle.get("used_for_skip_search")),
        "seedable": bool(bundle.get("seedable")),
        "summary_only": bool(bundle.get("summary_only")),
        "seed_evidence_count": int(bundle.get("seed_evidence_count") or 0),
        "preflight_status": str(_as_dict(bundle.get("preflight")).get("status") or ""),
    }
    try:
        from rag_pipeline.cache.topic_bundle_cache import topic_bundle_cache_root

        root = topic_bundle_cache_root()
        if root.exists():
            section["stored_bundle_count"] = sum(1 for child in root.iterdir() if child.is_dir())
        else:
            section["stored_bundle_count"] = 0
    except Exception as exc:  # fail-open
        section["fs_error"] = str(exc)
    return section


def build_cache_report(
    run_id: str,
    *,
    query: str = "",
    stage_snapshot_index: Optional[List[Dict[str, Any]]] = None,
    topic_bundle: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """聚合一个只读快照；任意子项失败都不会抛出。"""
    report: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "run_id": run_id,
        "query": query,
        "evidence_cache": _evidence_cache_section(),
        "stage_snapshots": _stage_snapshot_section(run_id, stage_snapshot_index),
        "trusted_source": _trusted_source_section(),
        "topic_bundle": _topic_bundle_section(topic_bundle),
    }
    if extra:
        report["extra"] = _as_dict(extra)
    return report


def write_cache_report(
    run_id: str,
    output_dir: Any,
    *,
    base_name: str = "",
    query: str = "",
    stage_snapshot_index: Optional[List[Dict[str, Any]]] = None,
    topic_bundle: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """构建并写一个 sidecar 文件 `<base_name>.cache_report.json`；写失败也不抛。"""
    report = build_cache_report(
        run_id,
        query=query,
        stage_snapshot_index=stage_snapshot_index,
        topic_bundle=topic_bundle,
        extra=extra,
    )
    try:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{base_name or run_id}.cache_report.json"
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        report["_path"] = str(out_path)
    except Exception as exc:  # fail-open
        report["_write_error"] = str(exc)
    return report
