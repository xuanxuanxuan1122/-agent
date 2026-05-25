from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import urlparse

from rag_pipeline.runtime_cache import json_safe_default


SCHEMA_VERSION = 1
DEFAULT_CACHE_PATH = "output/cache/topic_bundles"
EVIDENCE_LAYER_KEYS = (
    "core_evidence",
    "supporting_evidence",
    "metric_evidence",
    "case_evidence",
    "counter_evidence",
    "directional_evidence",
    "sample_evidence",
)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, min_value: int = 1, max_value: int = 3650) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        return default
    return max(min_value, min(max_value, value))


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


def _parse_time(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _compact_text(value: Any, *, max_chars: int = 240) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "…"


def _normalize_topic_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _slug(value: Any, *, max_chars: int = 48) -> str:
    text = _normalize_topic_text(value)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return (text or "topic")[:max_chars].strip("_") or "topic"


def _hash_payload(payload: Dict[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=json_safe_default)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def topic_bundle_enabled() -> bool:
    return _env_flag("TOPIC_BUNDLE_CACHE_ENABLED", True)


def topic_bundle_allow_skip_search() -> bool:
    return _env_flag("TOPIC_BUNDLE_CACHE_ALLOW_SKIP_SEARCH", False)


def topic_bundle_reuse_analysis() -> bool:
    return _env_flag("TOPIC_BUNDLE_CACHE_REUSE_ANALYSIS", False)


def topic_bundle_require_hydrated_evidence() -> bool:
    return _env_flag("TOPIC_BUNDLE_CACHE_REQUIRE_HYDRATED_EVIDENCE", True)


def topic_bundle_cache_root() -> Path:
    raw = os.getenv("TOPIC_BUNDLE_CACHE_PATH", DEFAULT_CACHE_PATH).strip() or DEFAULT_CACHE_PATH
    path = Path(raw)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def build_topic_key(
    query: str,
    research_plan: Optional[Dict[str, Any]] = None,
    report_blueprint: Optional[Dict[str, Any]] = None,
) -> str:
    """Build a stable topic key.

    The query is always the primary scope so a report can be restored before a
    new planning pass exists. Optional plan/blueprint fields add scope when the
    caller has them, and load_topic_bundle can still find query-compatible
    aliases by scanning manifests.
    """

    plan = _as_dict(research_plan)
    blueprint = _as_dict(report_blueprint)
    family = str(
        blueprint.get("report_family")
        or blueprint.get("report_type")
        or plan.get("report_family")
        or plan.get("report_type")
        or ""
    ).strip()
    geography = str(plan.get("geography") or blueprint.get("geography") or "").strip()
    time_scope = str(plan.get("time_scope") or blueprint.get("time_scope") or plan.get("date_range") or "").strip()
    scope_payload = {
        "query": _normalize_topic_text(query),
        "family": _normalize_topic_text(family),
        "geography": _normalize_topic_text(geography),
        "time_scope": _normalize_topic_text(time_scope),
    }
    digest = _hash_payload(scope_payload)[:12]
    prefix = _slug(query)
    return f"{prefix}__{digest}"


def _json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_safe_default),
        encoding="utf-8",
    )


def _json_read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _source_url(source: Dict[str, Any]) -> str:
    return str(source.get("url") or source.get("source_url") or source.get("link") or "").strip()


def _source_domain(source: Dict[str, Any]) -> str:
    url = _source_url(source)
    try:
        return urlparse(url).netloc.lower().removeprefix("www.")
    except Exception:
        return ""


def _has_document_ref(source: Dict[str, Any]) -> bool:
    return bool(
        str(
            source.get("document_id")
            or source.get("doc_id")
            or source.get("document_ref")
            or source.get("page_ref")
            or source.get("source_ref")
            or ""
        ).strip()
    )


def _is_fake_or_placeholder(payload: Dict[str, Any]) -> bool:
    joined = " ".join(
        str(item or "")
        for item in (
            payload.get("url"),
            payload.get("source_url"),
            payload.get("title"),
            payload.get("source_title"),
            payload.get("publisher"),
            payload.get("source"),
            payload.get("fact"),
            payload.get("clean_fact"),
        )
    ).lower()
    if "example.gov" in joined or "example.com" in joined:
        return True
    if "official data shows ai agent adoption reached 50% in 2025" in joined:
        return True
    title = str(payload.get("title") or payload.get("source_title") or "").strip().lower()
    publisher = str(payload.get("publisher") or payload.get("source") or "").strip()
    return title == "official" and not publisher and not _source_url(payload)


def _is_title_only_source(source: Dict[str, Any]) -> bool:
    title = str(source.get("title") or source.get("source_title") or "").strip()
    return bool(title) and not _source_url(source) and not _has_document_ref(source)


def _source_traceable(source: Dict[str, Any]) -> bool:
    if _is_fake_or_placeholder(source):
        return False
    if _source_url(source):
        return True
    if not _has_document_ref(source):
        return False
    fields = [
        bool(str(source.get("title") or source.get("source_title") or "").strip()),
        bool(str(source.get("publisher") or source.get("source") or "").strip()),
        bool(str(source.get("date") or source.get("published_at") or "").strip()),
    ]
    return sum(fields) >= 2


def _source_registry_from_package(evidence_package: Dict[str, Any], source_registry: Optional[Sequence[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    candidates = (
        [item for item in list(source_registry or []) if isinstance(item, dict)]
        or [item for item in _as_list(evidence_package.get("source_registry")) if isinstance(item, dict)]
        or [item for item in _as_list(evidence_package.get("sources")) if isinstance(item, dict)]
    )
    return [dict(item) for item in candidates]


def _analysis_ready_count(evidence_package: Dict[str, Any]) -> int:
    summary = _as_dict(evidence_package.get("summary"))
    health = (
        _as_dict(evidence_package.get("evidence_health_summary"))
        or _as_dict(summary.get("evidence_health_summary"))
        or _as_dict(_as_dict(evidence_package.get("metadata")).get("evidence_health_summary"))
    )
    for value in (
        health.get("analysis_ready_count"),
        summary.get("analysis_ready_count"),
        len(_as_list(evidence_package.get("analysis_ready_evidence"))),
    ):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            continue
    return 0


def _full_evidence_items(evidence_package: Dict[str, Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for key in ("analysis_ready_evidence", "clean_evidence_list", "normalized_evidence", "raw_data_points"):
        value = evidence_package.get(key)
        if isinstance(value, list):
            items.extend([dict(item) for item in value if isinstance(item, dict)])
    return items


def _has_full_evidence_list(evidence_package: Dict[str, Any]) -> bool:
    return bool(_full_evidence_items(_as_dict(evidence_package)))


def _is_count_sample_summary(value: Any) -> bool:
    payload = _as_dict(value)
    return bool(payload and "count" in payload and "sample" in payload)


def is_compacted_evidence_package(evidence_package: Dict[str, Any]) -> bool:
    package = _as_dict(evidence_package)
    if str(package.get("payload_mode") or "").strip().lower() == "summary":
        return True
    return any(
        _is_count_sample_summary(package.get(key))
        for key in ("analysis_ready_evidence", "clean_evidence_list", "normalized_evidence", "raw_data_points")
    )


def _chapter_package_has_hydrated_item(package: Dict[str, Any]) -> bool:
    return any(
        any(isinstance(item, dict) for item in _as_list(package.get(key)))
        for key in EVIDENCE_LAYER_KEYS
    )


def _iter_chapter_evidence_items(chapter_packages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for package in chapter_packages:
        payload = _as_dict(package)
        for key in EVIDENCE_LAYER_KEYS:
            items.extend([dict(item) for item in _as_list(payload.get(key)) if isinstance(item, dict)])
    return items


def _asset_ref_values(item: Dict[str, Any]) -> List[str]:
    refs: List[str] = []
    for key in ("evidence_id", "id", "ref", "source_ref", "citation_ref", "url", "source_url", "document_ref", "document_id"):
        value = str(_as_dict(item).get(key) or "").strip()
        if value:
            refs.append(value.lower())
    return refs


def _chapter_hydrated_count(chapter_packages: Sequence[Dict[str, Any]]) -> int:
    total = 0
    for package in chapter_packages:
        if not isinstance(package, dict):
            continue
        for key in EVIDENCE_LAYER_KEYS:
            total += len([item for item in _as_list(package.get(key)) if isinstance(item, dict)])
    return total


def _chapter_count_only_warning(chapter_packages: Sequence[Dict[str, Any]]) -> bool:
    if not chapter_packages:
        return False
    hydrated = _chapter_hydrated_count(chapter_packages)
    if hydrated > 0:
        return False
    for package in chapter_packages:
        counts = _as_dict(_as_dict(package).get("evidence_counts"))
        if any(str(counts.get(key) or "").strip() not in {"", "0", "0.0"} for key in EVIDENCE_LAYER_KEYS):
            return True
        if any(str(_as_dict(package).get(f"{key}_count") or "").strip() not in {"", "0", "0.0"} for key in EVIDENCE_LAYER_KEYS):
            return True
    return False


def bundle_completeness_summary(
    evidence_package: Dict[str, Any],
    chapter_evidence_packages: Sequence[Dict[str, Any]],
    source_registry: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    package = _as_dict(evidence_package)
    chapters = [dict(item) for item in list(chapter_evidence_packages or []) if isinstance(item, dict)]
    registry = [dict(item) for item in list(source_registry or []) if isinstance(item, dict)]
    compacted = is_compacted_evidence_package(package)
    full_evidence_count = len(_full_evidence_items(package))
    summary_only = bool(compacted and full_evidence_count <= 0)
    traceable_source_count = sum(1 for item in registry if _source_traceable(_as_dict(item)))
    hydrated_chapter_count = sum(1 for item in chapters if _chapter_package_has_hydrated_item(item))
    hydrated_evidence_count = _chapter_hydrated_count(chapters)
    return {
        "payload_mode": str(package.get("payload_mode") or ("summary" if compacted else "full")),
        "is_compacted": compacted,
        "summary_only": summary_only,
        "has_full_analysis_ready_evidence": isinstance(package.get("analysis_ready_evidence"), list)
        and bool(_as_list(package.get("analysis_ready_evidence"))),
        "has_full_clean_evidence_list": isinstance(package.get("clean_evidence_list"), list)
        and bool(_as_list(package.get("clean_evidence_list"))),
        "has_full_normalized_evidence": isinstance(package.get("normalized_evidence"), list)
        and bool(_as_list(package.get("normalized_evidence"))),
        "has_full_raw_data_points": isinstance(package.get("raw_data_points"), list)
        and bool(_as_list(package.get("raw_data_points"))),
        "full_evidence_count": full_evidence_count,
        "analysis_ready_count": _analysis_ready_count(package),
        "source_count": len(registry),
        "traceable_source_count": traceable_source_count,
        "chapter_count": len(chapters),
        "hydrated_chapter_count": hydrated_chapter_count,
        "hydrated_chapter_evidence_count": hydrated_evidence_count,
        "chapter_count_only_warning": _chapter_count_only_warning(chapters),
        "seedable": bool(full_evidence_count > 0 and traceable_source_count > 0 and not summary_only),
    }


def _evidence_source_payload(item: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(item)
    if "url" not in payload and item.get("source_url"):
        payload["url"] = item.get("source_url")
    if "title" not in payload and item.get("source_title"):
        payload["title"] = item.get("source_title")
    if "publisher" not in payload and item.get("source"):
        payload["publisher"] = item.get("source")
    return payload


def _source_mismatch_suspected(payload: Dict[str, Any]) -> bool:
    return bool(
        payload.get("source_title_url_mismatch_suspected")
        or payload.get("title_url_mismatch_suspected")
        or payload.get("source_mismatch_suspected")
        or payload.get("source_registry_mismatch")
    )


def validate_bundle_assets(
    *,
    evidence_package: Dict[str, Any],
    source_registry: Sequence[Dict[str, Any]],
    chapter_evidence_packages: Sequence[Dict[str, Any]],
    structured_analysis: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    package = _as_dict(evidence_package)
    chapters = [dict(item) for item in list(chapter_evidence_packages or []) if isinstance(item, dict)]
    registry = [dict(item) for item in list(source_registry or []) if isinstance(item, dict)]
    completeness = bundle_completeness_summary(package, chapters, registry)
    reasons: List[str] = []
    polluted = False
    if completeness.get("summary_only"):
        return {
            "status": "summary_only",
            "seedable": False,
            "usable_for_skip_search": False,
            "summary_only": True,
            "reasons": ["summary_only_not_reusable"],
            "completeness": completeness,
        }
    if not registry:
        reasons.append("source_registry_missing")
    for source in registry[:1000]:
        payload = _as_dict(source)
        if _is_fake_or_placeholder(payload):
            reasons.append("fake_or_placeholder_source")
            polluted = True
            break
    for source in registry[:1000]:
        payload = _as_dict(source)
        if _is_title_only_source(payload):
            reasons.append("title_only_source")
            polluted = True
            break
    for source in registry[:1000]:
        if _source_mismatch_suspected(_as_dict(source)):
            reasons.append("source_title_url_mismatch")
            polluted = True
            break
    evidence_items = _full_evidence_items(package) + _iter_chapter_evidence_items(chapters)
    full_ref_values = set()
    for item in _full_evidence_items(package):
        full_ref_values.update(_asset_ref_values(_as_dict(item)))
    for source in registry:
        full_ref_values.update(_asset_ref_values(_as_dict(source)))
    unresolved_chapter_refs = 0
    for item in _iter_chapter_evidence_items(chapters):
        payload = _as_dict(item)
        refs = _asset_ref_values(payload)
        if refs and not any(ref in full_ref_values for ref in refs) and not _source_traceable(_evidence_source_payload(payload)):
            unresolved_chapter_refs += 1
    for item in evidence_items[:3000]:
        payload = _evidence_source_payload(_as_dict(item))
        if _is_fake_or_placeholder(payload):
            reasons.append("fake_or_placeholder_evidence")
            polluted = True
            break
    for item in evidence_items[:3000]:
        payload = _evidence_source_payload(_as_dict(item))
        if _is_title_only_source(payload):
            reasons.append("title_only_evidence")
            polluted = True
            break
    for item in evidence_items[:3000]:
        if _source_mismatch_suspected(_as_dict(item)):
            reasons.append("evidence_source_mismatch")
            polluted = True
            break
    if registry and not any(_source_traceable(source) for source in registry):
        reasons.append("no_traceable_source")
        polluted = True
    if polluted:
        return {
            "status": "polluted",
            "seedable": False,
            "usable_for_skip_search": False,
            "summary_only": False,
            "reasons": sorted(set(reasons)),
            "completeness": completeness,
        }
    seedable = bool(completeness.get("seedable"))
    hydrated = int(completeness.get("hydrated_chapter_evidence_count") or 0)
    analysis_rebuild_required = _structured_analysis_rebuild_required(_as_dict(structured_analysis))
    if not seedable:
        reasons.append("full_evidence_missing")
        status = "partial"
    elif topic_bundle_require_hydrated_evidence() and hydrated <= 0:
        reasons.append("chapter_evidence_not_hydrated")
        status = "partial"
    elif unresolved_chapter_refs > 0:
        reasons.append("unresolved_chapter_evidence_refs")
        status = "partial"
    elif analysis_rebuild_required:
        reasons.append("analysis_rebuild_required")
        status = "partial"
    else:
        status = "usable"
    return {
        "status": status,
        "seedable": seedable,
        "usable_for_skip_search": bool(status == "usable" and seedable and hydrated > 0),
        "summary_only": False,
        "reasons": sorted(set(reasons)),
        "analysis_rebuild_required": analysis_rebuild_required,
        "unresolved_chapter_evidence_ref_count": unresolved_chapter_refs,
        "completeness": completeness,
    }


def _structured_analysis_rebuild_required(structured_analysis: Dict[str, Any]) -> bool:
    analysis = _as_dict(structured_analysis)
    status = str(_as_dict(analysis.get("analysis_depth_quality")).get("status") or analysis.get("status") or "").strip().lower()
    if status in {"needs_rewrite", "invalid", "polluted"}:
        return True
    text = json.dumps(analysis, ensure_ascii=False, default=json_safe_default)[:120000]
    return bool(
        re.search(
            r"正文只能写成|本章只能写成|本章可写成|建议补证|建议避免|evidence_cards|EV-\d+|claim_status",
            text,
            flags=re.I,
        )
    )


def _bundle_manifest_summary(
    *,
    query: str,
    topic_key: str,
    evidence_package: Dict[str, Any],
    source_registry: Sequence[Dict[str, Any]],
    chapter_evidence_packages: Sequence[Dict[str, Any]],
    structured_analysis: Dict[str, Any],
    report_blueprint: Dict[str, Any],
    stage: str,
    existing_manifest: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    health = (
        _as_dict(evidence_package.get("evidence_health_summary"))
        or _as_dict(_as_dict(evidence_package.get("summary")).get("evidence_health_summary"))
        or _as_dict(_as_dict(evidence_package.get("metadata")).get("evidence_health_summary"))
    )
    existing = _as_dict(existing_manifest)
    completeness = bundle_completeness_summary(evidence_package, chapter_evidence_packages, source_registry)
    now = _now_iso()
    return {
        "schema_version": SCHEMA_VERSION,
        "topic_key": topic_key,
        "query": query,
        "query_normalized": _normalize_topic_text(query),
        "report_family": str(report_blueprint.get("report_family") or report_blueprint.get("report_type") or ""),
        "created_at": existing.get("created_at") or now,
        "updated_at": now,
        "stage": stage,
        "stored_from": stage,
        "cache_mode": os.getenv("TOPIC_BUNDLE_CACHE_MODE", "seed_or_skip"),
        "max_age_days": _env_int("TOPIC_BUNDLE_CACHE_MAX_AGE_DAYS", 30, min_value=1, max_value=3650),
        "completeness": completeness,
        "summary_only": bool(completeness.get("summary_only")),
        "seedable": bool(completeness.get("seedable")),
        "evidence_health_summary": health,
        "source_count": len(source_registry),
        "traceable_source_count": sum(1 for item in source_registry if _source_traceable(_as_dict(item))),
        "analysis_ready_count": _analysis_ready_count(evidence_package),
        "chapter_count": len(chapter_evidence_packages),
        "hydrated_chapter_evidence_count": _chapter_hydrated_count(chapter_evidence_packages),
        "chapter_count_only_warning": _chapter_count_only_warning(chapter_evidence_packages),
        "analysis_rebuild_required": _structured_analysis_rebuild_required(structured_analysis),
        "last_hit_at": existing.get("last_hit_at") or "",
        "hit_count": int(existing.get("hit_count") or 0),
    }


def store_topic_bundle(
    *,
    query: str,
    research_plan: Optional[Dict[str, Any]] = None,
    report_blueprint: Optional[Dict[str, Any]] = None,
    evidence_package: Optional[Dict[str, Any]] = None,
    structured_analysis: Optional[Dict[str, Any]] = None,
    source_registry: Optional[Sequence[Dict[str, Any]]] = None,
    chapter_evidence_packages: Optional[Sequence[Dict[str, Any]]] = None,
    micro_layouts: Optional[Sequence[Dict[str, Any]]] = None,
    table_packages: Optional[Sequence[Dict[str, Any]]] = None,
    writer_report: Optional[Dict[str, Any]] = None,
    stage: str = "writer",
    stored_from: str = "",
) -> Dict[str, Any]:
    if not topic_bundle_enabled():
        return {"enabled": False, "stored": False, "reason": "disabled"}
    package = _as_dict(evidence_package)
    if not package:
        return {"enabled": True, "stored": False, "reason": "missing_evidence_package"}
    blueprint = _as_dict(report_blueprint) or _as_dict(_as_dict(writer_report).get("report_blueprint"))
    topic_key = build_topic_key(query, research_plan, blueprint)
    bundle_dir = topic_bundle_cache_root() / topic_key
    existing_manifest: Dict[str, Any] = {}
    manifest_path = bundle_dir / "manifest.json"
    if manifest_path.exists():
        try:
            existing_manifest = _as_dict(_json_read(manifest_path))
        except Exception:
            existing_manifest = {}
    registry = _source_registry_from_package(package, source_registry)
    chapters = [dict(item) for item in list(chapter_evidence_packages or []) if isinstance(item, dict)]
    if not chapters:
        chapters = [dict(item) for item in _as_list(package.get("chapter_evidence_packages")) if isinstance(item, dict)]
    analysis = _as_dict(structured_analysis)
    layouts = [dict(item) for item in list(micro_layouts or []) if isinstance(item, dict)]
    tables = [dict(item) for item in list(table_packages or []) if isinstance(item, dict)]
    manifest = _bundle_manifest_summary(
        query=query,
        topic_key=topic_key,
        evidence_package=package,
        source_registry=registry,
        chapter_evidence_packages=chapters,
        structured_analysis=analysis,
        report_blueprint=blueprint,
        stage=stored_from or stage,
        existing_manifest=existing_manifest,
    )
    validation = validate_bundle_assets(
        evidence_package=package,
        source_registry=registry,
        chapter_evidence_packages=chapters,
        structured_analysis=analysis,
    )
    manifest["validation"] = validation
    manifest["stored_from"] = stored_from or stage
    if validation.get("status") == "summary_only":
        existing_has_full_bundle = bool(
            existing_manifest
            and not bool(_as_dict(existing_manifest.get("completeness")).get("summary_only"))
            and int(_as_dict(existing_manifest.get("completeness")).get("full_evidence_count") or 0) > 0
        )
        manifest["status"] = "summary_only_not_reusable"
        manifest["stored"] = False
        manifest["reason"] = "summary_only_not_reusable"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        if not existing_has_full_bundle:
            _json_write(manifest_path, manifest)
        return {
            "enabled": True,
            "stored": False,
            "manifest_written": not existing_has_full_bundle,
            "topic_key": topic_key,
            "path": str(bundle_dir),
            "stage": stage,
            "stored_from": stored_from or stage,
            "reason": "summary_only_not_reusable",
            "summary_only": True,
            "seedable": False,
            "existing_full_bundle_preserved": existing_has_full_bundle,
            "completeness": manifest.get("completeness"),
        }
    bundle_dir.mkdir(parents=True, exist_ok=True)
    _json_write(manifest_path, manifest)
    _json_write(bundle_dir / "source_registry.json", registry)
    _json_write(bundle_dir / "evidence_package.json", package)
    _json_write(bundle_dir / "chapter_evidence_packages.json", chapters)
    _json_write(bundle_dir / "structured_analysis.json", analysis)
    _json_write(
        bundle_dir / "layout_and_tables.json",
        {
            "report_blueprint": blueprint,
            "micro_layouts": layouts,
            "table_packages": tables,
            "table_validation_summary": _as_dict(_as_dict(writer_report).get("table_validation_summary")),
        },
    )
    return {
        "enabled": True,
        "stored": True,
        "topic_key": topic_key,
        "path": str(bundle_dir),
        "stage": stage,
        "stored_from": stored_from or stage,
        "analysis_rebuild_required": bool(manifest.get("analysis_rebuild_required")),
        "analysis_ready_count": manifest.get("analysis_ready_count"),
        "hydrated_chapter_evidence_count": manifest.get("hydrated_chapter_evidence_count"),
        "source_count": manifest.get("source_count"),
        "summary_only": bool(manifest.get("summary_only")),
        "seedable": bool(manifest.get("seedable")),
        "validation_status": validation.get("status"),
    }


def _read_bundle_dir(bundle_dir: Path) -> Dict[str, Any]:
    manifest = _as_dict(_json_read(bundle_dir / "manifest.json"))
    layout_and_tables = _as_dict(_json_read(bundle_dir / "layout_and_tables.json")) if (bundle_dir / "layout_and_tables.json").exists() else {}
    return {
        "manifest": manifest,
        "source_registry": _as_list(_json_read(bundle_dir / "source_registry.json")) if (bundle_dir / "source_registry.json").exists() else [],
        "evidence_package": _as_dict(_json_read(bundle_dir / "evidence_package.json")) if (bundle_dir / "evidence_package.json").exists() else {},
        "chapter_evidence_packages": _as_list(_json_read(bundle_dir / "chapter_evidence_packages.json")) if (bundle_dir / "chapter_evidence_packages.json").exists() else [],
        "structured_analysis": _as_dict(_json_read(bundle_dir / "structured_analysis.json")) if (bundle_dir / "structured_analysis.json").exists() else {},
        "report_blueprint": _as_dict(layout_and_tables.get("report_blueprint")),
        "micro_layouts": _as_list(layout_and_tables.get("micro_layouts")),
        "table_packages": _as_list(layout_and_tables.get("table_packages")),
        "layout_and_tables": layout_and_tables,
    }


def _candidate_bundle_dirs(query: str, topic_key: str) -> List[Path]:
    root = topic_bundle_cache_root()
    exact = root / topic_key
    candidates: List[Path] = []
    if exact.exists():
        candidates.append(exact)
    query_norm = _normalize_topic_text(query)
    if root.exists() and query_norm:
        for manifest_path in root.glob("*/manifest.json"):
            bundle_dir = manifest_path.parent
            if bundle_dir in candidates:
                continue
            try:
                manifest = _as_dict(_json_read(manifest_path))
            except Exception:
                continue
            if str(manifest.get("query_normalized") or "") == query_norm:
                candidates.append(bundle_dir)
    def _candidate_updated_at(path: Path) -> str:
        try:
            return str(_as_dict(_json_read(path / "manifest.json")).get("updated_at") or "")
        except Exception:
            return ""

    candidates.sort(key=_candidate_updated_at, reverse=True)
    return candidates


def load_topic_bundle(
    query: str,
    research_plan: Optional[Dict[str, Any]] = None,
    report_blueprint: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not topic_bundle_enabled():
        return {"enabled": False, "found": False, "reason": "disabled"}
    topic_key = build_topic_key(query, research_plan, report_blueprint)
    errors: List[str] = []
    for bundle_dir in _candidate_bundle_dirs(query, topic_key):
        try:
            bundle = _read_bundle_dir(bundle_dir)
            manifest = _as_dict(bundle.get("manifest"))
            manifest["last_hit_at"] = _now_iso()
            manifest["hit_count"] = int(manifest.get("hit_count") or 0) + 1
            _json_write(bundle_dir / "manifest.json", manifest)
            bundle["manifest"] = manifest
            return {
                "enabled": True,
                "found": True,
                "topic_key": str(manifest.get("topic_key") or bundle_dir.name),
                "path": str(bundle_dir),
                "bundle": bundle,
            }
        except Exception as exc:
            errors.append(f"{bundle_dir}: {exc}")
    return {"enabled": True, "found": False, "topic_key": topic_key, "errors": errors}


def _bundle_pollution_reasons(bundle: Dict[str, Any]) -> List[str]:
    validation = validate_bundle_assets(
        evidence_package=_as_dict(bundle.get("evidence_package")),
        source_registry=_as_list(bundle.get("source_registry")),
        chapter_evidence_packages=_as_list(bundle.get("chapter_evidence_packages")),
        structured_analysis=_as_dict(bundle.get("structured_analysis")),
    )
    if validation.get("status") == "polluted":
        return [str(item) for item in _as_list(validation.get("reasons"))]
    return []


def preflight_topic_bundle(
    bundle_result: Dict[str, Any],
    *,
    query: str = "",
    research_plan: Optional[Dict[str, Any]] = None,
    report_blueprint: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not _as_dict(bundle_result).get("enabled", True):
        return {"enabled": False, "status": "disabled", "can_skip_search": False}
    if not bundle_result.get("found"):
        return {
            "enabled": True,
            "status": "missing",
            "topic_key": bundle_result.get("topic_key") or build_topic_key(query, research_plan, report_blueprint),
            "can_skip_search": False,
        }
    bundle = _as_dict(bundle_result.get("bundle"))
    manifest = _as_dict(bundle.get("manifest"))
    topic_key = str(bundle_result.get("topic_key") or manifest.get("topic_key") or "")
    evidence_package = _as_dict(bundle.get("evidence_package"))
    validation = validate_bundle_assets(
        evidence_package=evidence_package,
        source_registry=_as_list(bundle.get("source_registry")),
        chapter_evidence_packages=_as_list(bundle.get("chapter_evidence_packages")),
        structured_analysis=_as_dict(bundle.get("structured_analysis")),
    )
    completeness = _as_dict(validation.get("completeness"))
    if int(manifest.get("schema_version") or 0) != SCHEMA_VERSION:
        return {
            "enabled": True,
            "status": "incompatible",
            "topic_key": topic_key,
            "path": bundle_result.get("path"),
            "stored_from": manifest.get("stored_from") or manifest.get("stage"),
            "can_skip_search": False,
            "can_seed_evidence": bool(validation.get("seedable")),
            "seedable": bool(validation.get("seedable")),
            "usable_for_skip_search": False,
            "summary_only": bool(validation.get("summary_only")),
            "completeness": completeness,
            "reason": "schema_version_mismatch",
        }
    if bool(manifest.get("summary_only")) or str(manifest.get("status") or "").strip() == "summary_only_not_reusable":
        return {
            "enabled": True,
            "status": "summary_only",
            "topic_key": topic_key,
            "path": bundle_result.get("path"),
            "stored_from": manifest.get("stored_from") or manifest.get("stage"),
            "can_skip_search": False,
            "can_seed_evidence": False,
            "seedable": False,
            "usable_for_skip_search": False,
            "summary_only": True,
            "reasons": _as_list(manifest.get("reasons")) or ["summary_only_not_reusable"],
            "reason": "summary_only_not_reusable",
            "completeness": _as_dict(manifest.get("completeness")) or completeness,
        }
    if validation.get("status") == "summary_only":
        return {
            "enabled": True,
            "status": "summary_only",
            "topic_key": topic_key,
            "path": bundle_result.get("path"),
            "stored_from": manifest.get("stored_from") or manifest.get("stage"),
            "can_skip_search": False,
            "can_seed_evidence": False,
            "seedable": False,
            "usable_for_skip_search": False,
            "summary_only": True,
            "reasons": _as_list(validation.get("reasons")) or ["summary_only_not_reusable"],
            "reason": "summary_only_not_reusable",
            "completeness": completeness,
        }
    if validation.get("status") == "polluted":
        return {
            "enabled": True,
            "status": "polluted",
            "topic_key": topic_key,
            "path": bundle_result.get("path"),
            "can_skip_search": False,
            "can_seed_evidence": False,
            "seedable": False,
            "usable_for_skip_search": False,
            "summary_only": False,
            "reasons": _as_list(validation.get("reasons")),
            "completeness": completeness,
        }
    updated = _parse_time(manifest.get("updated_at") or manifest.get("created_at"))
    age_days = None
    if updated:
        age_days = max(0.0, (datetime.now(timezone.utc) - updated).total_seconds() / 86400.0)
    max_age = _env_int("TOPIC_BUNDLE_CACHE_MAX_AGE_DAYS", int(manifest.get("max_age_days") or 30), min_value=1, max_value=3650)
    if age_days is not None and age_days > max_age:
        return {
            "enabled": True,
            "status": "stale",
            "topic_key": topic_key,
            "path": bundle_result.get("path"),
            "stored_from": manifest.get("stored_from") or manifest.get("stage"),
            "age_days": round(age_days, 2),
            "max_age_days": max_age,
            "can_skip_search": False,
            "can_seed_evidence": bool(validation.get("seedable")),
            "seedable": bool(validation.get("seedable")),
            "usable_for_skip_search": False,
            "summary_only": False,
            "completeness": completeness,
        }
    analysis_ready = _analysis_ready_count(evidence_package)
    hydrated = _chapter_hydrated_count(_as_list(bundle.get("chapter_evidence_packages")))
    if validation.get("status") == "partial" and "chapter_evidence_not_hydrated" in _as_list(validation.get("reasons")):
        return {
            "enabled": True,
            "status": "partial",
            "topic_key": topic_key,
            "path": bundle_result.get("path"),
            "stored_from": manifest.get("stored_from") or manifest.get("stage"),
            "can_skip_search": False,
            "can_seed_evidence": bool(validation.get("seedable")),
            "seedable": bool(validation.get("seedable")),
            "usable_for_skip_search": False,
            "summary_only": False,
            "reason": "chapter_evidence_not_hydrated",
            "reasons": _as_list(validation.get("reasons")),
            "analysis_ready_count": analysis_ready,
            "hydrated_chapter_evidence_count": hydrated,
            "completeness": completeness,
        }
    if not validation.get("seedable"):
        return {
            "enabled": True,
            "status": "partial",
            "topic_key": topic_key,
            "path": bundle_result.get("path"),
            "stored_from": manifest.get("stored_from") or manifest.get("stage"),
            "can_skip_search": False,
            "can_seed_evidence": False,
            "seedable": False,
            "usable_for_skip_search": False,
            "summary_only": False,
            "reason": "analysis_ready_evidence_missing",
            "reasons": _as_list(validation.get("reasons")),
            "completeness": completeness,
        }
    analysis_rebuild_required = bool(manifest.get("analysis_rebuild_required")) or _structured_analysis_rebuild_required(_as_dict(bundle.get("structured_analysis")))
    usable_for_skip_search = bool(validation.get("usable_for_skip_search")) and not analysis_rebuild_required
    can_skip = bool(topic_bundle_allow_skip_search() and usable_for_skip_search)
    status = "usable" if usable_for_skip_search else "partial"
    return {
        "enabled": True,
        "status": status,
        "topic_key": topic_key,
        "path": bundle_result.get("path"),
        "stored_from": manifest.get("stored_from") or manifest.get("stage"),
        "can_skip_search": can_skip,
        "can_seed_evidence": True,
        "seedable": True,
        "usable_for_skip_search": usable_for_skip_search,
        "summary_only": False,
        "analysis_ready_count": analysis_ready,
        "hydrated_chapter_evidence_count": hydrated,
        "analysis_rebuild_required": analysis_rebuild_required,
        "reuse_analysis": bool(topic_bundle_reuse_analysis() and not analysis_rebuild_required),
        "source_count": manifest.get("source_count"),
        "traceable_source_count": manifest.get("traceable_source_count"),
        "age_days": round(age_days, 2) if age_days is not None else None,
        "reasons": _as_list(validation.get("reasons")),
        "completeness": completeness,
    }


def bundle_to_writer_inputs(
    bundle_result: Dict[str, Any],
    *,
    preflight: Optional[Dict[str, Any]] = None,
    reuse_analysis: Optional[bool] = None,
) -> Dict[str, Any]:
    bundle = _as_dict(bundle_result.get("bundle"))
    manifest = _as_dict(bundle.get("manifest"))
    preflight_payload = _as_dict(preflight)
    if reuse_analysis is None:
        reuse_analysis = bool(preflight_payload.get("reuse_analysis")) if preflight_payload else topic_bundle_reuse_analysis()
    structured_analysis = _as_dict(bundle.get("structured_analysis")) if reuse_analysis else {}
    evidence_package = _as_dict(bundle.get("evidence_package"))
    seed_evidence = _full_evidence_items(evidence_package)
    return {
        "query": manifest.get("query") or "",
        "topic_key": manifest.get("topic_key") or bundle_result.get("topic_key") or "",
        "path": bundle_result.get("path") or "",
        "manifest": manifest,
        "evidence_package": evidence_package,
        "seed_evidence": seed_evidence,
        "seed_evidence_count": len(seed_evidence),
        "source_registry": _as_list(bundle.get("source_registry")),
        "chapter_evidence_packages": _as_list(bundle.get("chapter_evidence_packages")),
        "structured_analysis": structured_analysis,
        "report_blueprint": _as_dict(bundle.get("report_blueprint")),
        "micro_layouts": _as_list(bundle.get("micro_layouts")),
        "table_packages": _as_list(bundle.get("table_packages")),
        "layout_and_tables": _as_dict(bundle.get("layout_and_tables")),
        "preflight": preflight_payload,
        "analysis_rebuild_required": not bool(structured_analysis),
    }
