from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, Iterable, List


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


def _first_dict(*values: Any) -> Dict[str, Any]:
    for value in values:
        payload = _as_dict(value)
        if payload:
            return payload
    return {}


def _stable_unique(values: Iterable[Any]) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        text = _text(value)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _chapter_id(item: Dict[str, Any], default: str = "", aliases: Dict[str, str] | None = None) -> str:
    lineage = _as_dict(item.get("lineage"))
    metadata = _as_dict(item.get("metadata"))
    raw = _text(
        item.get("chapter_id")
        or item.get("dimension_id")
        or lineage.get("chapter_id")
        or metadata.get("chapter_id")
        or default
    )
    return (aliases or {}).get(raw, raw)


def _evidence_id(item: Dict[str, Any]) -> str:
    return _text(item.get("fact_id") or item.get("evidence_id") or item.get("id") or item.get("ref"))


def _claim_id(item: Dict[str, Any]) -> str:
    return _text(item.get("claim_id") or item.get("id"))


def _ids(item: Dict[str, Any], *keys: str) -> List[str]:
    values: List[Any] = []
    for key in keys:
        value = item.get(key)
        if isinstance(value, (list, tuple)):
            values.extend(_as_list(value))
        elif _text(value):
            values.append(value)
    return _stable_unique(values)


def _line_level_unresolved_refs(claim: Dict[str, Any]) -> List[str]:
    refs = _ids(claim, "unresolved_refs", "ambiguous_refs")
    ref_resolution = _as_dict(claim.get("ref_resolution"))
    refs.extend(_ids(ref_resolution, "unresolved_refs", "ambiguous_refs"))
    result: List[str] = []
    for ref in _stable_unique(refs):
        key = re.sub(r"[\s_]+", "-", ref.strip().lower())
        if re.fullmatch(r".+-l\d+", key):
            result.append(ref)
    return result


def _analysis_ready_items(evidence_package: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ("analysis_ready_evidence", "analysis_ready_facts", "fact_cards", "clean_facts"):
        items = [item for item in _as_list(evidence_package.get(key)) if isinstance(item, dict)]
        if items:
            return items
    return []


def _curated_evidence_items(evidence_package: Dict[str, Any]) -> List[Dict[str, Any]]:
    curated = _as_dict(evidence_package.get("curated_evidence"))
    for key in ("curated_evidence", "items", "notes"):
        items = [item for item in _as_list(curated.get(key)) if isinstance(item, dict)]
        if items:
            return items
    return []


def _curated_evidence_id_set(items: Iterable[Dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for item in items:
        for value in _ids(item, "fact_id", "evidence_id", "id", "ref", "merged_evidence_ids"):
            ids.add(value)
    return ids


def _inventory_items(evidence_package: Dict[str, Any]) -> List[Dict[str, Any]]:
    inventory = _as_dict(evidence_package.get("evidence_inventory"))
    return [item for item in _as_list(inventory.get("inventories")) if isinstance(item, dict)]


def _analysis_shard_items(evidence_package: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [item for item in _as_list(evidence_package.get("analysis_shards")) if isinstance(item, dict)]


def _inventory_evidence_id_set(items: Iterable[Dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for item in items:
        for value in _as_list(item.get("usable_evidence_ids")):
            text = _text(value)
            if text:
                ids.add(text)
    return ids


def _claim_units(package: Dict[str, Any], report: Dict[str, Any]) -> List[Dict[str, Any]]:
    analysis = _first_dict(
        package.get("structured_analysis"),
        _as_dict(report.get("render_artifacts")).get("structured_analysis"),
        report.get("structured_analysis"),
    )
    return [item for item in _as_list(analysis.get("claim_units")) if isinstance(item, dict)]


def _structured_analysis(package: Dict[str, Any], report: Dict[str, Any]) -> Dict[str, Any]:
    return _first_dict(
        package.get("structured_analysis"),
        _as_dict(report.get("render_artifacts")).get("structured_analysis"),
        report.get("structured_analysis"),
    )


def _analysis_stage_diagnostics(package: Dict[str, Any], report: Dict[str, Any]) -> Dict[str, Any]:
    structured = _structured_analysis(package, report)
    return _first_dict(
        package.get("analysis_stage_diagnostics"),
        structured.get("analysis_stage_diagnostics"),
        _as_dict(report.get("render_artifacts")).get("analysis_stage_diagnostics"),
        report.get("analysis_stage_diagnostics"),
    )


def _analysis_shard_cache_summary(package: Dict[str, Any], report: Dict[str, Any]) -> Dict[str, Any]:
    diagnostics = _analysis_stage_diagnostics(package, report)
    items: List[Dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    miss_reasons: Counter[str] = Counter()
    for raw in _as_list(diagnostics.get("llm_chapter_results")):
        chapter_result = _as_dict(raw)
        cache = _as_dict(chapter_result.get("analysis_shard_output_cache"))
        if not cache:
            continue
        status = _text(cache.get("status")) or "unknown"
        reason = _text(cache.get("reason"))
        status_counts[status] += 1
        if status == "miss" and reason:
            miss_reasons[reason] += 1
        items.append(
            {
                "chapter_id": _text(chapter_result.get("chapter_id")),
                "status": status,
                "reason": reason,
                "input_hash": _text(cache.get("input_hash")),
                "output_cache_path": _text(cache.get("output_cache_path")),
            }
        )
    return {
        "items": items,
        "status_counts": dict(status_counts),
        "miss_reason_counts": dict(miss_reasons),
        "hit_count": int(status_counts.get("hit") or 0),
        "miss_count": int(status_counts.get("miss") or 0),
        "stored_count": int(status_counts.get("stored") or 0),
        "saved_llm_call_count": int(status_counts.get("hit") or 0),
    }


def _chapter_alias_map(package: Dict[str, Any], report: Dict[str, Any]) -> Dict[str, str]:
    render_artifacts = _as_dict(report.get("render_artifacts"))
    recomposition = _first_dict(
        package.get("chapter_recomposition"),
        report.get("chapter_recomposition"),
        render_artifacts.get("chapter_recomposition"),
    )
    chapter_lists = [
        _as_list(package.get("final_chapters")),
        _as_list(report.get("final_chapters")),
        _as_list(render_artifacts.get("final_chapters")),
        _as_list(recomposition.get("final_chapters")),
    ]
    aliases: Dict[str, str] = {}
    ambiguous: set[str] = set()
    for chapters in chapter_lists:
        for raw in chapters:
            chapter = _as_dict(raw)
            canonical = _text(chapter.get("chapter_id") or chapter.get("final_chapter_id"))
            if not canonical:
                continue
            candidates = _stable_unique(
                [
                    canonical,
                    *_as_list(chapter.get("chapter_id_aliases")),
                    *_as_list(chapter.get("source_plan_chapter_ids")),
                    chapter.get("source_plan_chapter_id"),
                    chapter.get("plan_chapter_id"),
                    chapter.get("original_chapter_id"),
                ]
            )
            for candidate in candidates:
                existing = aliases.get(candidate)
                if existing and existing != canonical:
                    ambiguous.add(candidate)
                    continue
                aliases[candidate] = canonical
    for candidate in ambiguous:
        aliases.pop(candidate, None)
    return aliases


def _sections(package: Dict[str, Any], report: Dict[str, Any], aliases: Dict[str, str] | None = None) -> List[Dict[str, Any]]:
    sections: List[Dict[str, Any]] = []
    for chapter in _as_list(package.get("chapter_packages")):
        chapter_payload = _as_dict(chapter)
        chapter_id = _chapter_id(chapter_payload, aliases=aliases)
        for raw in _as_list(chapter_payload.get("sections")):
            section = dict(_as_dict(raw))
            if chapter_id and not _chapter_id(section, aliases=aliases):
                section["chapter_id"] = chapter_id
            if section:
                sections.append(section)
    render_artifacts = _as_dict(report.get("render_artifacts"))
    for raw in _as_list(package.get("sections")) or _as_list(report.get("sections")) or _as_list(render_artifacts.get("sections")):
        section = _as_dict(raw)
        if section:
            sections.append(section)
    return sections


def _chapter_bucket(chapters: Dict[str, Dict[str, Any]], chapter_id: str) -> Dict[str, Any]:
    key = chapter_id or "missing"
    return chapters.setdefault(
        key,
        {
            "chapter_id": key,
            "analysis_ready_facts": 0,
            "analysis_input_cards": 0,
            "raw_claims": 0,
            "usable_claims": 0,
            "bound_claims": 0,
            "backed_sections": 0,
            "loss_reasons": {},
        },
    )


def _increment_loss(bucket: Dict[str, Any], totals: Counter[str], reason: str) -> None:
    losses = bucket.setdefault("loss_reasons", {})
    losses[reason] = int(losses.get(reason) or 0) + 1
    totals[reason] += 1


def build_evidence_claim_conversion_monitor(
    *,
    writer_package: Dict[str, Any],
    writer_report: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Explain how analysis-ready evidence becomes claims and backed sections."""

    package = _as_dict(writer_package)
    report = _as_dict(writer_report) or _as_dict(package.get("writer_report"))
    evidence_package = _first_dict(
        package.get("evidence_package"),
        _as_dict(report.get("render_artifacts")).get("evidence_package"),
    )
    chapter_aliases = _chapter_alias_map(package, report)
    evidence_items = _analysis_ready_items(evidence_package)
    curated_items = _curated_evidence_items(evidence_package)
    inventory_items = _inventory_items(evidence_package)
    analysis_shards = _analysis_shard_items(evidence_package)
    shard_cache_summary = _analysis_shard_cache_summary(package, report)
    claims = _claim_units(package, report)
    sections = _sections(package, report, chapter_aliases)
    chapters: Dict[str, Dict[str, Any]] = {}
    loss_counts: Counter[str] = Counter()

    for item in evidence_items:
        chapter_id = _chapter_id(item, aliases=chapter_aliases)
        bucket = _chapter_bucket(chapters, chapter_id)
        bucket["analysis_ready_facts"] += 1
        if chapter_id:
            bucket["analysis_input_cards"] += 1
        else:
            _increment_loss(bucket, loss_counts, "no_chapter_match")
        if not _ids(item, "requirement_ids", "requirement_id"):
            _increment_loss(bucket, loss_counts, "no_requirement_id")
        if not _ids(item, "source_ids", "source_id"):
            _increment_loss(bucket, loss_counts, "no_source_ref")

    bound_claim_ids = set()
    claim_fact_ids = set()
    unresolved_l_ref_count = 0
    for claim in claims:
        chapter_id = _chapter_id(claim, aliases=chapter_aliases)
        bucket = _chapter_bucket(chapters, chapter_id)
        bucket["raw_claims"] += 1
        bucket["usable_claims"] += 1
        fact_ids = _ids(claim, "fact_ids", "used_evidence_ids", "used_fact_refs", "evidence_refs")
        source_ids = _ids(claim, "source_ids")
        if fact_ids and source_ids:
            bucket["bound_claims"] += 1
            claim_id = _claim_id(claim)
            if claim_id:
                bound_claim_ids.add(claim_id)
            claim_fact_ids.update(fact_ids)
        else:
            _increment_loss(bucket, loss_counts, "claim_unbound")
        unresolved_l_refs = _line_level_unresolved_refs(claim)
        if unresolved_l_refs:
            unresolved_l_ref_count += len(unresolved_l_refs)
            for _ in unresolved_l_refs:
                _increment_loss(bucket, loss_counts, "unresolved_L_ref")

    for section in sections:
        chapter_id = _chapter_id(section, aliases=chapter_aliases)
        bucket = _chapter_bucket(chapters, chapter_id)
        used_refs = _ids(section, "used_fact_refs", "evidence_refs", "fact_ids")
        claim_ids = _ids(section, "claim_ids", "claim_id")
        backed = bool(used_refs or (set(claim_ids) & bound_claim_ids))
        if backed:
            bucket["backed_sections"] += 1
        else:
            _increment_loss(bucket, loss_counts, "section_unbacked")

    curated_ids = _curated_evidence_id_set(curated_items)
    curated_used_ids = curated_ids & claim_fact_ids
    curated_count = len(curated_items)
    inventory_ids = _inventory_evidence_id_set(inventory_items)
    inventory_used_ids = inventory_ids & claim_fact_ids
    totals = {
        "analysis_ready_facts": len(evidence_items),
        "curated_evidence_count": curated_count,
        "curated_evidence_id_count": len(curated_ids),
        "curated_used_in_claim_count": len(curated_used_ids),
        "curated_to_claim_rate": round(len(curated_used_ids) / max(1, len(curated_ids)), 4),
        "inventory_cluster_count": len(inventory_items),
        "inventory_evidence_id_count": len(inventory_ids),
        "inventory_used_in_claim_count": len(inventory_used_ids),
        "inventory_to_claim_rate": round(len(inventory_used_ids) / max(1, len(inventory_ids)), 4),
        "analysis_shard_count": len(analysis_shards),
        "analysis_shard_cache_hit_count": shard_cache_summary["hit_count"],
        "analysis_shard_cache_miss_count": shard_cache_summary["miss_count"],
        "analysis_shard_output_cache_stored_count": shard_cache_summary["stored_count"],
        "analysis_shard_cache_saved_llm_call_count": shard_cache_summary["saved_llm_call_count"],
        "analysis_input_cards": sum(int(item.get("analysis_input_cards") or 0) for item in chapters.values()),
        "raw_claims": len(claims),
        "usable_claims": len(claims),
        "bound_claims": sum(int(item.get("bound_claims") or 0) for item in chapters.values()),
        "backed_sections": sum(int(item.get("backed_sections") or 0) for item in chapters.values()),
        "claim_fact_id_count": len(claim_fact_ids),
        "unresolved_L_ref_count": unresolved_l_ref_count,
    }
    return {
        "schema_version": "evidence_claim_conversion_monitor_v1",
        "diagnostic_only": True,
        "must_not_render": True,
        "public_text_allowed": False,
        "totals": totals,
        "analysis_shard_cache": shard_cache_summary,
        "loss_reason_counts": dict(loss_counts),
        "chapters": chapters,
    }
