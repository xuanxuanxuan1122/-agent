from __future__ import annotations

import gzip
import json
import os
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from rag_pipeline.runtime_cache import json_safe_default


SCHEMA_VERSION = 1
DEFAULT_CACHE_PATH = "output/cache/stage_snapshots"
REPLAYABLE_STAGES = {
    "evidence_package",
    "chapter_evidence_packages",
    "structured_analysis",
    "argument_units",
    "chapter_packages",
    "table_packages",
    "writer_report",
    "qa_result",
    "final_audit_result",
}


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


def _env_int(name: str, default: int, *, min_value: int = 0, max_value: int = 100000) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        return default
    return max(min_value, min(max_value, value))


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_cache_path(raw_path: str, default: str) -> Path:
    path = Path((raw_path or default).strip() or default)
    if path.is_absolute():
        return path
    return _project_root() / path


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_name(value: Any, *, max_chars: int = 80) -> str:
    text = re.sub(r"[^\w\u4e00-\u9fff.-]+", "_", str(value or "").strip(), flags=re.U)
    text = re.sub(r"_+", "_", text).strip("._")
    return (text or "snapshot")[:max_chars]


def stage_snapshot_enabled() -> bool:
    return _env_flag("STAGE_SNAPSHOT_CACHE_ENABLED", True)


def stage_snapshot_cache_root() -> Path:
    raw = os.getenv("STAGE_SNAPSHOT_CACHE_PATH", DEFAULT_CACHE_PATH).strip() or DEFAULT_CACHE_PATH
    return _resolve_cache_path(raw, DEFAULT_CACHE_PATH)


def _save_full_payload() -> bool:
    return _env_flag("STAGE_SNAPSHOT_SAVE_FULL_PAYLOAD", True)


def _compress_large_payload() -> bool:
    return _env_flag("STAGE_SNAPSHOT_COMPRESS_LARGE_PAYLOAD", True)


def _full_payload_replayable_only() -> bool:
    """When on, only replayable stages get their full payload persisted.

    The full-payload files are only consumable by a (future) replay/partial-rerun
    driver — they are never read back by the live pipeline. A non-replayable
    stage's full payload therefore can't drive anything, so this lever lets prod
    keep just the (small) manifest/diagnostics for those stages and skip the
    multi-MB payload write. Default off → behaviour unchanged."""
    return _env_flag("STAGE_SNAPSHOT_FULL_PAYLOAD_REPLAYABLE_ONLY", False)


def _max_payload_bytes() -> int:
    return _env_int("STAGE_SNAPSHOT_MAX_PAYLOAD_MB", 80, min_value=1, max_value=4096) * 1024 * 1024


def _json_bytes(payload: Any) -> bytes:
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=json_safe_default)
    return text.encode("utf-8")


def _write_payload(path: Path, payload: Any) -> Dict[str, Any]:
    data = _json_bytes(payload)
    too_large = len(data) > _max_payload_bytes()
    if _compress_large_payload() and len(data) > 1024 * 1024:
        gz_path = path.with_suffix(path.suffix + ".gz")
        gz_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(gz_path, "wb") as file:
            file.write(data)
        return {"path": str(gz_path), "compressed": True, "bytes": len(data), "too_large": too_large}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"path": str(path), "compressed": False, "bytes": len(data), "too_large": too_large}


def _read_payload(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as file:
            return json.loads(file.read().decode("utf-8"))
    return json.loads(path.read_text(encoding="utf-8"))


class StageSnapshotCorruptError(Exception):
    """Raised when a stage snapshot file exists but cannot be deserialized."""


def summarize_payload(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict):
        summary: Dict[str, Any] = {"type": "dict", "keys": sorted(str(key) for key in payload.keys())[:40]}
        for key, value in payload.items():
            if isinstance(value, list):
                summary[f"{key}_count"] = len(value)
            elif isinstance(value, dict):
                summary[f"{key}_keys"] = sorted(str(item) for item in value.keys())[:20]
        return summary
    if isinstance(payload, list):
        return {"type": "list", "count": len(payload)}
    return {"type": type(payload).__name__, "present": payload is not None}


def snapshot_is_replayable(stage_name: str, payload: Any) -> bool:
    stage = str(stage_name or "").strip()
    if stage not in REPLAYABLE_STAGES:
        return False
    if stage == "evidence_package":
        return not evidence_package_replay_missing(payload)
    if stage in {"writer_report", "qa_result", "final_audit_result"}:
        return bool(_as_dict(payload))
    if stage in {"chapter_evidence_packages", "argument_units", "chapter_packages", "table_packages"}:
        return bool(_as_list(payload))
    if stage == "structured_analysis":
        return bool(_as_dict(payload))
    return False


def evidence_package_replay_missing(payload: Any) -> List[str]:
    package = _as_dict(payload)
    evidence_package = _as_dict(package.get("evidence_package")) or package
    blueprint = (
        _as_dict(package.get("report_blueprint"))
        or _as_dict(evidence_package.get("report_blueprint"))
        or _as_dict(_as_dict(evidence_package.get("metadata")).get("report_blueprint"))
        or _as_dict(evidence_package.get("report_plan"))
        or _as_dict(_as_dict(evidence_package.get("metadata")).get("report_plan"))
    )
    chapters = (
        _as_list(blueprint.get("chapters"))
        or _as_list(evidence_package.get("chapter_plan"))
        or _as_list(_as_dict(evidence_package.get("metadata")).get("chapter_plan"))
    )
    evidence_items = (
        _as_list(evidence_package.get("analysis_ready_evidence"))
        or _as_list(evidence_package.get("clean_evidence_list"))
        or _as_list(evidence_package.get("normalized_evidence"))
        or _as_list(evidence_package.get("raw_data_points"))
    )
    sources = _as_list(evidence_package.get("source_registry")) or _as_list(evidence_package.get("sources"))
    missing: List[str] = []
    if not chapters:
        missing.append("report_blueprint_or_chapter_plan")
    if not evidence_items:
        missing.append("writable_evidence")
    if not sources:
        missing.append("source_registry")
    return missing


def write_stage_snapshot(
    stage_name: str,
    run_id: str,
    payload: Any,
    summary: Optional[Dict[str, Any]] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not stage_snapshot_enabled():
        return {"enabled": False, "stored": False, "stage_name": stage_name, "run_id": run_id}
    stage = _safe_name(stage_name)
    run = _safe_name(run_id, max_chars=120)
    stage_dir = stage_snapshot_cache_root() / run / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    replayable = snapshot_is_replayable(stage, payload)
    payload_info: Dict[str, Any] = {}
    full_payload_skipped = ""
    if _save_full_payload():
        if replayable or not _full_payload_replayable_only():
            payload_info = _write_payload(stage_dir / "payload.json", payload)
        else:
            full_payload_skipped = "non_replayable_payload_skipped"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage_name": stage,
        "run_id": run,
        "created_at": _now_iso(),
        "input_summary": _as_dict(summary),
        "output_summary": summarize_payload(payload),
        "diagnostics": _as_dict(diagnostics),
        "full_payload_path": payload_info.get("path", ""),
        "full_payload_compressed": bool(payload_info.get("compressed")),
        "full_payload_bytes": payload_info.get("bytes", 0),
        "full_payload_too_large": bool(payload_info.get("too_large")),
        "full_payload_skipped": full_payload_skipped,
        "replayable": replayable,
    }
    if stage == "evidence_package":
        manifest["replayable_missing"] = evidence_package_replay_missing(payload)
    (stage_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=json_safe_default),
        encoding="utf-8",
    )
    return {"enabled": True, "stored": True, **manifest}


def load_stage_snapshot(run_id: str, stage_name: str) -> Dict[str, Any]:
    stage_dir = stage_snapshot_cache_root() / _safe_name(run_id, max_chars=120) / _safe_name(stage_name)
    manifest_path = stage_dir / "manifest.json"
    if not manifest_path.exists():
        return {"status": "missing", "run_id": run_id, "stage_name": stage_name}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        return {
            "status": "corrupt",
            "run_id": run_id,
            "stage_name": stage_name,
            "manifest_path": str(manifest_path),
            "error": str(exc),
        }
    # A manifest can legitimately carry no full payload (full-payload saving
    # disabled, or replayable-only mode skipped a non-replayable stage). Guard
    # against the empty string — Path("") is Path("."), which exists and would
    # be misread as a corrupt payload.
    raw_payload_path = str(manifest.get("full_payload_path") or "").strip()
    payload_path = Path(raw_payload_path) if raw_payload_path else None
    payload = None
    payload_error: Optional[str] = None
    if payload_path is not None and payload_path.exists():
        try:
            payload = _read_payload(payload_path)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError, gzip.BadGzipFile, EOFError) as exc:
            payload_error = str(exc)
    if payload_error is not None:
        return {
            "status": "corrupt",
            "run_id": run_id,
            "stage_name": stage_name,
            "manifest": manifest,
            "manifest_path": str(manifest_path),
            "payload_path": str(payload_path or ""),
            "error": payload_error,
        }
    return {"status": "loaded", "manifest": manifest, "payload": payload}


def list_stage_snapshots(run_id: str) -> List[Dict[str, Any]]:
    run_dir = stage_snapshot_cache_root() / _safe_name(run_id, max_chars=120)
    if not run_dir.exists():
        return []
    snapshots: List[Dict[str, Any]] = []
    for manifest_path in sorted(run_dir.glob("*/manifest.json")):
        try:
            snapshots.append(json.loads(manifest_path.read_text(encoding="utf-8")))
        except Exception:
            continue
    return snapshots


def stage_snapshot_retention_max_runs() -> int:
    return _env_int("STAGE_SNAPSHOT_RETENTION_MAX_RUNS", 100, min_value=0, max_value=1_000_000)


def stage_snapshot_retention_max_age_days() -> int:
    return _env_int("STAGE_SNAPSHOT_RETENTION_MAX_AGE_DAYS", 0, min_value=0, max_value=100_000)


def prune_stage_snapshots(
    *,
    max_runs: Optional[int] = None,
    max_age_days: Optional[int] = None,
    keep_run_id: str = "",
) -> Dict[str, Any]:
    """Delete old per-run snapshot directories so the cache does not grow without
    bound (each run leaves a ~MB tree and nothing here is reaped otherwise).

    Keeps the newest ``max_runs`` run directories (by mtime) and drops anything
    older than ``max_age_days``; ``0`` disables that limit. ``keep_run_id`` (the
    in-flight run) is never deleted regardless of caps. Fail-open: any IO error
    is swallowed so retention never blocks report delivery.
    """
    max_runs = stage_snapshot_retention_max_runs() if max_runs is None else max(0, int(max_runs))
    max_age_days = stage_snapshot_retention_max_age_days() if max_age_days is None else max(0, int(max_age_days))
    result: Dict[str, Any] = {"deleted_count": 0, "deleted": [], "kept_count": 0, "errors": 0}
    if max_runs <= 0 and max_age_days <= 0:
        result["status"] = "disabled"
        return result
    root = stage_snapshot_cache_root()
    if not root.exists():
        result["status"] = "no_cache"
        return result
    keep_name = _safe_name(keep_run_id, max_chars=120) if keep_run_id else ""
    try:
        run_dirs = [child for child in root.iterdir() if child.is_dir()]
    except OSError as exc:
        return {**result, "status": "list_failed", "error": str(exc)}
    run_dirs.sort(key=lambda child: _safe_mtime(child), reverse=True)
    now = time.time()
    kept = 0
    for child in run_dirs:
        if keep_name and child.name == keep_name:
            continue
        kept += 1
        over_count = max_runs > 0 and kept > max_runs
        too_old = max_age_days > 0 and (now - _safe_mtime(child)) > max_age_days * 86400
        if not (over_count or too_old):
            continue
        kept -= 1
        try:
            shutil.rmtree(child)
            result["deleted_count"] += 1
            if len(result["deleted"]) < 50:
                result["deleted"].append(child.name)
        except OSError:
            result["errors"] += 1
    result["kept_count"] = kept
    result["status"] = "pruned"
    return result


def _safe_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0
