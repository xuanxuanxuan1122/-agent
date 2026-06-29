from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Sequence

from rag_pipeline.contracts.repair_dispatcher import dispatch_repair_seed


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return default


def _chapter_id(item: Dict[str, Any], fallback: str = "") -> str:
    metadata = _as_dict(item.get("metadata"))
    return str(
        item.get("chapter_id")
        or item.get("dimension_id")
        or item.get("chapter")
        or metadata.get("chapter_id")
        or fallback
    ).strip()


def _evidence_id(item: Dict[str, Any]) -> str:
    return str(item.get("fact_id") or item.get("evidence_id") or item.get("id") or item.get("ref") or "").strip()


def _role(item: Dict[str, Any]) -> str:
    return str(item.get("proof_role") or item.get("analysis_role") or item.get("role") or "support").strip().lower() or "support"


def _text_terms(value: Any, *, max_terms: int = 16) -> List[str]:
    text = str(value or "").strip().lower()
    if not text:
        return []
    terms: List[str] = []
    for match in re.findall(r"[a-z0-9][a-z0-9_\-]{1,}|[\u4e00-\u9fff]{2,}", text, flags=re.I):
        if match not in terms:
            terms.append(match)
        if len(match) >= 4:
            for index in range(0, max(0, len(match) - 1)):
                gram = match[index : index + 2]
                if gram and gram not in terms:
                    terms.append(gram)
        if len(terms) >= max_terms:
            break
    return terms[:max_terms]


def _chapter_terms(chapter: Dict[str, Any]) -> List[str]:
    values = [
        chapter.get("chapter_title"),
        chapter.get("title"),
        chapter.get("chapter_question"),
        chapter.get("core_question"),
        chapter.get("question"),
    ]
    terms: List[str] = []
    for value in values:
        for term in _text_terms(value):
            if term not in terms:
                terms.append(term)
    return terms[:18]


def _item_text(item: Dict[str, Any]) -> str:
    fields = [
        item.get("title"),
        item.get("fact"),
        item.get("content"),
        item.get("summary"),
        item.get("claim"),
        item.get("text"),
        item.get("metric"),
        item.get("source_title"),
    ]
    return " ".join(str(value or "") for value in fields).lower()


def _collect_items(evidence_package: Dict[str, Any]) -> List[Dict[str, Any]]:
    buckets = [
        "analysis_ready_facts",
        "clean_facts",
        "fact_cards",
        "evidence_fact_cards",
        "facts",
        "evidence_items",
    ]
    chapter_buckets = [
        "analysis_ready_facts",
        "fact_cards",
        "core_evidence",
        "supporting_evidence",
        "metric_evidence",
        "case_evidence",
        "counter_evidence",
        "directional_evidence",
        "sample_evidence",
        "table_evidence",
        "clue_evidence",
    ]
    items: List[Dict[str, Any]] = []
    seen = set()
    for bucket in buckets:
        for raw in _as_list(evidence_package.get(bucket)):
            item = _as_dict(raw)
            if not item:
                continue
            key = _evidence_id(item) or f"{bucket}:{len(items)}"
            if key in seen:
                continue
            seen.add(key)
            copied = dict(item)
            copied["_alignment_bucket"] = bucket
            items.append(copied)
    for raw_chapter in _as_list(evidence_package.get("chapter_evidence_packages")):
        chapter = _as_dict(raw_chapter)
        if not chapter:
            continue
        chapter_id = _chapter_id(chapter)
        for bucket in chapter_buckets:
            for raw in _as_list(chapter.get(bucket)):
                item = _as_dict(raw)
                if not item:
                    continue
                key = _evidence_id(item) or f"chapter:{chapter_id}:{bucket}:{len(items)}"
                if key in seen:
                    continue
                seen.add(key)
                copied = dict(item)
                copied.setdefault("chapter_id", chapter_id)
                copied.setdefault("dimension_id", chapter_id)
                copied["_alignment_bucket"] = bucket
                items.append(copied)
    return items


def _search_task_counts(evidence_package: Dict[str, Any]) -> Dict[str, int]:
    tasks = (
        _as_list(evidence_package.get("search_tasks"))
        or _as_list(_as_dict(evidence_package.get("metadata")).get("search_tasks"))
        or _as_list(_as_dict(evidence_package.get("raw_output")).get("search_tasks"))
    )
    counts: Dict[str, int] = {}
    for raw in tasks:
        task = _as_dict(raw)
        chapter_id = _chapter_id(task)
        if chapter_id:
            counts[chapter_id] = counts.get(chapter_id, 0) + 1
    return counts


def _overlap_count(terms: Sequence[str], text: str) -> int:
    normalized = re.sub(r"\s+", "", str(text or "").lower())
    return sum(1 for term in terms if term and term in normalized)


def _repair_seed_for_chapter(chapter: Dict[str, Any], payload: Dict[str, Any], warning: str) -> Dict[str, Any]:
    chapter_id = str(payload.get("chapter_id") or "").strip()
    title = str(payload.get("chapter_title") or "").strip()
    terms = _as_list(payload.get("chapter_terms"))[:5]
    query = " ".join(str(item) for item in [title, *terms, "evidence source"] if str(item or "").strip())
    seed = {
        "schema_version": "repair_task_seed_v2",
        "query": query,
        "agent": "iqs",
        "gap_id": f"blueprint:{chapter_id}:{warning}",
        "chapter_id": chapter_id,
        "requirement_id": str(chapter.get("requirement_id") or chapter.get("evidence_requirement_id") or "").strip(),
        "gap_type": warning,
        "repair_status": "still_insufficient",
        "proof_role": "support",
        "missing_fields": ["fact", "source_ref"],
        "required_fields": ["fact", "source_ref"],
        "lane_targets": ["market_research", "official_data", "customer_case"],
        "success_criteria": "Only count as repaired when this chapter has traceable evidence that matches the chapter question.",
        "reject_if": ["snippet_only", "no_source_url", "off_topic"],
        "preferred_source_patterns": ["market_research", "official_data", "customer_case", "news"],
        "targets_gap": warning,
        "evidence_goal": f"Repair chapter evidence coverage for {title or chapter_id}",
        "source_stage": "blueprint_evidence_alignment",
        "allowed_for_writing": False,
        "diagnostic_only": True,
        "must_not_render": True,
        "public_text_allowed": False,
    }
    return dispatch_repair_seed(seed)


def build_blueprint_evidence_alignment(
    *,
    report_blueprint: Dict[str, Any],
    evidence_package: Dict[str, Any],
    min_claimable_facts: int = 1,
    overload_ratio: float = 0.65,
) -> Dict[str, Any]:
    """Build a diagnostic-only chapter coverage view before analysis/writing."""

    blueprint = _as_dict(report_blueprint)
    package = _as_dict(evidence_package)
    chapters = [_as_dict(item) for item in _as_list(blueprint.get("chapters")) if _as_dict(item)]
    items = _collect_items(package)
    search_counts = _search_task_counts(package)
    total_items = len(items)
    total_claimable = max(0, total_items)
    chapter_payload: Dict[str, Dict[str, Any]] = {}

    for index, chapter in enumerate(chapters, start=1):
        chapter_id = _chapter_id(chapter, f"ch_{index:02d}")
        terms = _chapter_terms(chapter)
        matched = [item for item in items if _chapter_id(item) == chapter_id]
        role_distribution: Dict[str, int] = {}
        matched_term_count = 0
        for item in matched:
            role = _role(item)
            role_distribution[role] = role_distribution.get(role, 0) + 1
            if _overlap_count(terms, _item_text(item)) > 0:
                matched_term_count += 1

        fact_count = len(matched)
        warnings: List[str] = []
        if fact_count < min_claimable_facts:
            warnings.append("chapter_starved")
        if total_claimable > 0 and len(chapters) > 1 and fact_count / total_claimable >= overload_ratio:
            warnings.append("chapter_overloaded")
        if fact_count > 0 and terms and matched_term_count == 0:
            warnings.append("chapter_misaligned")

        status = "starved" if "chapter_starved" in warnings else ("warning" if warnings else "ok")
        chapter_payload[chapter_id] = {
            "chapter_id": chapter_id,
            "chapter_title": str(chapter.get("chapter_title") or chapter.get("title") or "").strip(),
            "status": status,
            "warnings": warnings,
            "search_task_count": search_counts.get(chapter_id, 0),
            "clean_fact_count": sum(1 for item in matched if item.get("_alignment_bucket") == "clean_facts"),
            "analysis_ready_count": sum(1 for item in matched if item.get("_alignment_bucket") == "analysis_ready_facts"),
            "claimable_fact_count": fact_count,
            "matched_after_relevance": fact_count,
            "hydrated_card_count": fact_count,
            "role_distribution": role_distribution,
            "chapter_terms": terms[:8],
            "matched_term_count": matched_term_count,
            "sample_evidence_ids": [_evidence_id(item) for item in matched if _evidence_id(item)][:8],
        }

    warnings_by_type = {
        "chapter_starved": 0,
        "chapter_overloaded": 0,
        "chapter_misaligned": 0,
    }
    repair_task_seeds: List[Dict[str, Any]] = []
    for payload in chapter_payload.values():
        for warning in _as_list(payload.get("warnings")):
            if warning in warnings_by_type:
                warnings_by_type[warning] += 1
            if warning in {"chapter_starved", "chapter_misaligned"}:
                chapter = next(
                    (
                        item
                        for item in chapters
                        if _chapter_id(item) == str(payload.get("chapter_id") or "").strip()
                    ),
                    {},
                )
                repair_task_seeds.append(_repair_seed_for_chapter(chapter, payload, warning))

    return {
        "schema_version": "blueprint_evidence_alignment_v1",
        "diagnostic_only": True,
        "must_not_render": True,
        "public_text_allowed": False,
        "chapter_count": len(chapters),
        "evidence_count": total_items,
        "chapter_starved_count": warnings_by_type["chapter_starved"],
        "chapter_overloaded_count": warnings_by_type["chapter_overloaded"],
        "chapter_misaligned_count": warnings_by_type["chapter_misaligned"],
        "warnings_by_type": warnings_by_type,
        "repair_task_seeds": repair_task_seeds,
        "repair_task_seed_count": len(repair_task_seeds),
        "chapters": chapter_payload,
    }
