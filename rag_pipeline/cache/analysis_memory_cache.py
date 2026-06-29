from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from rag_pipeline.runtime_cache import json_safe_default

from .artifact_paths import artifact_object_root, env_flag, safe_path_part
from .artifact_store import ArtifactStore


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _text(value: Any) -> str:
    return str(value or "").strip()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=json_safe_default)


def _hash_payload(value: Any) -> str:
    return hashlib.sha256(_json_dumps(value).encode("utf-8")).hexdigest()


def _default_cache_root() -> Path:
    raw = os.getenv("ANALYSIS_MEMORY_CACHE_PATH")
    if raw:
        path = Path(raw)
        return path if path.is_absolute() else Path.cwd() / path
    return artifact_object_root() / "analysis_memory"


def _curated_items(evidence_package: Dict[str, Any]) -> List[Dict[str, Any]]:
    curated = _as_dict(evidence_package.get("curated_evidence"))
    return [item for item in _as_list(curated.get("curated_evidence")) if isinstance(item, dict)]


def _inventory_items(evidence_package: Dict[str, Any]) -> List[Dict[str, Any]]:
    inventory = _as_dict(evidence_package.get("evidence_inventory"))
    return [item for item in _as_list(inventory.get("inventories")) if isinstance(item, dict)]


def _ids(values: Iterable[Any]) -> List[str]:
    result: List[str] = []
    seen: set[str] = set()
    for value in values:
        text = _text(value)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def build_analysis_shards(evidence_package: Dict[str, Any], *, max_evidence_per_shard: int = 80) -> List[Dict[str, Any]]:
    """Build compact per-cluster analysis inputs from inventory + curated evidence."""

    curated_by_id = {
        _text(item.get("evidence_id")): item
        for item in _curated_items(evidence_package)
        if _text(item.get("evidence_id"))
    }
    shards: List[Dict[str, Any]] = []
    for inventory in _inventory_items(evidence_package):
        cluster_key = _text(inventory.get("cluster_key")) or "context"
        usable_ids = _ids(_as_list(inventory.get("usable_evidence_ids")))[:max_evidence_per_shard]
        curated_slice = [curated_by_id[evidence_id] for evidence_id in usable_ids if evidence_id in curated_by_id]
        if not curated_slice:
            continue
        shard_payload = {
            "schema_version": "analysis_shard_input_v1",
            "cluster_key": cluster_key,
            "chapter_id": _text(inventory.get("chapter_id")),
            "requirement_id": _text(inventory.get("requirement_id")),
            "inventory_id": _text(inventory.get("inventory_id")),
            "analysis_brief": _text(inventory.get("analysis_brief")),
            "dominant_strength": _text(inventory.get("dominant_strength")),
            "strongest_available_level": _text(inventory.get("strongest_available_level")),
            "curated_evidence_ids": [_text(item.get("evidence_id")) for item in curated_slice],
            "curated_evidence": curated_slice,
            "limitations": _as_list(inventory.get("limitations")),
            "suggested_analysis_direction": _as_list(inventory.get("suggested_analysis_direction")),
        }
        shard_payload["input_hash"] = _hash_payload(
            {
                "cluster_key": shard_payload["cluster_key"],
                "inventory": {
                    key: shard_payload.get(key)
                    for key in (
                        "chapter_id",
                        "requirement_id",
                        "analysis_brief",
                        "dominant_strength",
                        "strongest_available_level",
                    )
                },
                "curated_evidence": curated_slice,
            }
        )
        shards.append(shard_payload)
    return shards


def _write_json(path: Path, payload: Any) -> Dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False, indent=2, default=json_safe_default).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"path": str(path), "bytes": len(data), "content_hash": hashlib.sha256(data).hexdigest()}


def persist_analysis_memory_cache(
    evidence_package: Dict[str, Any],
    *,
    run_id: str,
    cache_root: Optional[Path] = None,
    artifact_store: Optional[ArtifactStore] = None,
) -> Dict[str, Any]:
    """Persist evidence inventory and shard inputs as temporary analysis memory.

    The cache stores only curated/inventory inputs. It does not store final prose
    or audit text, so downstream agents can use it as context without treating
    review diagnostics as facts.
    """

    run = safe_path_part(run_id or _text(evidence_package.get("run_id")) or "run")
    root = cache_root or _default_cache_root()
    inventory = _as_dict(evidence_package.get("evidence_inventory"))
    if not inventory:
        return {"status": "skipped", "reason": "missing_evidence_inventory", "run_id": run_id}

    shards = build_analysis_shards(evidence_package)
    run_dir = root / "runs" / run
    inventory_info = _write_json(run_dir / "evidence_inventory.json", inventory)
    shard_paths: List[str] = []
    shard_infos: List[Dict[str, Any]] = []
    for shard in shards:
        cluster = safe_path_part(shard.get("cluster_key") or shard.get("inventory_id") or "cluster")
        info = _write_json(run_dir / "analysis_shards" / f"{cluster}.json", shard)
        shard_paths.append(info["path"])
        shard_infos.append({"cluster_key": shard.get("cluster_key"), **info})

    store = artifact_store
    if store is None and env_flag("ARTIFACT_LEDGER_ENABLED", True):
        store = ArtifactStore()

    inventory_artifact_id = ""
    shard_artifact_ids: List[str] = []
    if store is not None and run_id:
        inventory_result = store.record_artifact(
            run_id=run_id,
            stage="evidence_inventory",
            artifact_type="evidence_inventory",
            payload=inventory,
            status=str(inventory.get("status") or "ready"),
            schema_version=str(inventory.get("schema_version") or "evidence_inventory_v1"),
            content_hash=inventory_info["content_hash"],
            storage_uri=inventory_info["path"],
            storage_bytes=int(inventory_info["bytes"]),
            lineage={"source": "curated_evidence"},
        )
        inventory_artifact_id = inventory_result.artifact_id
        for shard, info in zip(shards, shard_infos):
            result = store.record_artifact(
                run_id=run_id,
                stage="analysis_shard_input",
                artifact_type="analysis_shard_input",
                payload=shard,
                status="ready",
                schema_version="analysis_shard_input_v1",
                input_hash=str(shard.get("input_hash") or ""),
                content_hash=str(info.get("content_hash") or ""),
                storage_uri=str(info.get("path") or ""),
                storage_bytes=int(info.get("bytes") or 0),
                lineage={
                    "inventory_id": shard.get("inventory_id"),
                    "cluster_key": shard.get("cluster_key"),
                    "chapter_id": shard.get("chapter_id"),
                    "requirement_id": shard.get("requirement_id"),
                    "curated_evidence_ids": shard.get("curated_evidence_ids"),
                },
            )
            shard_artifact_ids.append(result.artifact_id)

    return {
        "schema_version": "analysis_memory_cache_v1",
        "status": "stored",
        "run_id": run_id,
        "inventory_cache_path": inventory_info["path"],
        "analysis_shard_cache_paths": shard_paths,
        "analysis_shard_count": len(shards),
        "inventory_artifact_id": inventory_artifact_id,
        "analysis_shard_artifact_ids": shard_artifact_ids,
    }


def persist_analysis_shard_output(
    evidence_package: Dict[str, Any],
    *,
    chapter_payload: Dict[str, Any],
    analysis_result: Dict[str, Any],
    run_id: str,
    cache_root: Optional[Path] = None,
    artifact_store: Optional[ArtifactStore] = None,
    model: str = "",
    prompt_version: str = "",
) -> Dict[str, Any]:
    """Persist one cluster/chapter LLM analysis output keyed by shard input hash."""

    shard = _as_dict(chapter_payload.get("analysis_shard"))
    input_hash = _text(shard.get("input_hash"))
    if not input_hash:
        return {"status": "skipped", "reason": "missing_analysis_shard_input_hash", "run_id": run_id}
    if not run_id:
        return {"status": "skipped", "reason": "missing_run_id", "input_hash": input_hash}

    run = safe_path_part(run_id)
    cluster = safe_path_part(shard.get("cluster_key") or chapter_payload.get("chapter_id") or "cluster")
    payload = {
        "schema_version": "analysis_shard_output_v1",
        "run_id": run_id,
        "cluster_key": _text(shard.get("cluster_key")),
        "chapter_id": _text(chapter_payload.get("chapter_id") or shard.get("chapter_id")),
        "requirement_id": _text(shard.get("requirement_id")),
        "inventory_id": _text(shard.get("inventory_id")),
        "input_hash": input_hash,
        "prompt_version": prompt_version,
        "model": model,
        "curated_evidence_ids": _as_list(shard.get("curated_evidence_ids")),
        "analysis_result": analysis_result,
    }
    root = cache_root or _default_cache_root()
    path = root / "runs" / run / "analysis_shard_outputs" / f"{cluster}_{input_hash[:16]}.json"
    output_info = _write_json(path, payload)

    store = artifact_store
    if store is None and env_flag("ARTIFACT_LEDGER_ENABLED", True):
        store = ArtifactStore()

    artifact_id = ""
    if store is not None:
        result = store.record_artifact(
            run_id=run_id,
            stage="analysis_shard_output",
            artifact_type="analysis_shard_output",
            payload=payload,
            status="ready",
            schema_version="analysis_shard_output_v1",
            prompt_version=prompt_version,
            model=model,
            input_hash=input_hash,
            content_hash=output_info["content_hash"],
            storage_uri=output_info["path"],
            storage_bytes=int(output_info["bytes"]),
            lineage={
                "inventory_id": payload.get("inventory_id"),
                "cluster_key": payload.get("cluster_key"),
                "chapter_id": payload.get("chapter_id"),
                "requirement_id": payload.get("requirement_id"),
                "curated_evidence_ids": payload.get("curated_evidence_ids"),
            },
        )
        artifact_id = result.artifact_id

    return {
        "schema_version": "analysis_shard_output_cache_v1",
        "status": "stored",
        "run_id": run_id,
        "input_hash": input_hash,
        "output_cache_path": output_info["path"],
        "artifact_id": artifact_id,
    }


def load_analysis_shard_output(
    *,
    chapter_payload: Dict[str, Any],
    run_id: str,
    cache_root: Optional[Path] = None,
    model: str = "",
    prompt_version: str = "",
) -> Dict[str, Any]:
    """Load a shard output only when the full replay contract matches."""

    shard = _as_dict(chapter_payload.get("analysis_shard"))
    input_hash = _text(shard.get("input_hash"))
    if not input_hash:
        return {"status": "miss", "reason": "missing_analysis_shard_input_hash"}
    if not run_id:
        return {"status": "miss", "reason": "missing_run_id", "input_hash": input_hash}
    run = safe_path_part(run_id)
    cluster = safe_path_part(shard.get("cluster_key") or chapter_payload.get("chapter_id") or "cluster")
    root = cache_root or _default_cache_root()
    path = root / "runs" / run / "analysis_shard_outputs" / f"{cluster}_{input_hash[:16]}.json"
    if not path.exists():
        return {"status": "miss", "reason": "missing_file", "input_hash": input_hash, "output_cache_path": str(path)}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "miss", "reason": "invalid_json", "error": str(exc), "output_cache_path": str(path)}

    expected = {
        "schema_version": "analysis_shard_output_v1",
        "input_hash": input_hash,
        "model": model,
        "prompt_version": prompt_version,
    }
    actual = {key: _text(payload.get(key)) for key in expected}
    if actual != expected:
        return {
            "status": "miss",
            "reason": "contract_mismatch",
            "expected": expected,
            "actual": actual,
            "output_cache_path": str(path),
        }
    return {
        "schema_version": "analysis_shard_output_cache_v1",
        "status": "hit",
        "run_id": run_id,
        "input_hash": input_hash,
        "output_cache_path": str(path),
        "analysis_result": _as_dict(payload.get("analysis_result")),
        "model": model,
        "prompt_version": prompt_version,
    }
