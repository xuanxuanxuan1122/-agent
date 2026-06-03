from __future__ import annotations

import copy
import re
from typing import Any, Dict, Iterable, List, Sequence

EVIDENCE_COLLECTION_KEYS = (
    "core_evidence",
    "supporting_evidence",
    "metric_evidence",
    "case_evidence",
    "counter_evidence",
    "directional_evidence",
    "sample_evidence",
    "evidence_items",
    "table_evidence",
    "analysis_ready_evidence",
    "clean_evidence_list",
    "normalized_evidence",
    "raw_data_points",
)


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _dedupe(values: Iterable[Any], *, limit: int = 200) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def _clean_source_scalar(value: Any) -> str:
    if isinstance(value, (dict, list, tuple, set)):
        return ""
    text = re.sub(r"\s+", " ", str(value or "").strip())
    if not text:
        return ""
    if text.startswith(("{", "[")) and re.search(r"['\"](?:title|url|source_url|ref)['\"]\s*:", text):
        return ""
    if len(text) > 240 and re.search(r"['\"](?:title|url|source_url|ref)['\"]\s*:", text):
        return ""
    return text


def _first_source_scalar(*values: Any) -> str:
    for value in values:
        text = _clean_source_scalar(value)
        if text:
            return text
    return ""


def _sanitize_source_metadata(source: Dict[str, Any]) -> None:
    for key in ("title", "source_title", "publisher", "source"):
        if key not in source:
            continue
        cleaned = _clean_source_scalar(source.get(key))
        if cleaned:
            source[key] = cleaned
        else:
            source.pop(key, None)


def normalize_citation_ref(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    match = re.fullmatch(r"\[?\s*(\d{1,5})\s*\]?", text)
    if match:
        return f"[{match.group(1)}]"
    return text


def _source_identity_refs(source: Dict[str, Any]) -> set[str]:
    refs = {
        str(source.get("ref") or "").strip(),
        str(source.get("id") or "").strip(),
        str(source.get("evidence_id") or "").strip(),
        str(source.get("source_ref") or "").strip(),
        str(source.get("citation_ref") or "").strip(),
        str(source.get("document_id") or "").strip(),
        str(source.get("doc_id") or "").strip(),
    }
    for key in ("evidence_refs", "used_fact_refs", "source_refs", "refs"):
        refs.update(str(item or "").strip() for item in _as_list(source.get(key)))
    normalized = set(ref for ref in refs if ref)
    normalized.update(normalize_citation_ref(ref) for ref in list(normalized))
    normalized.discard("")
    return normalized


def _placeholder_url(url: str) -> bool:
    text = str(url or "").strip().lower()
    return bool(re.search(r"(?:^|[/:.])example\.(?:com|gov)(?:[/:]|$)", text))


def _source_traceable(source: Dict[str, Any]) -> bool:
    url = str(source.get("url") or source.get("source_url") or "").strip()
    doc_id = str(source.get("document_id") or source.get("doc_id") or source.get("page_ref") or "").strip()
    return bool((url and not _placeholder_url(url)) or doc_id)


def _source_exclusion_reason(source: Dict[str, Any]) -> str:
    title = str(source.get("title") or source.get("source_title") or "").strip()
    url = str(source.get("url") or source.get("source_url") or "").strip()
    blob = " ".join(str(source.get(key) or "") for key in ("title", "source_title", "summary", "snippet", "url", "source_url")).lower()
    if re.search(r"404|not\s*found|页面未找到|页面不存在", blob, flags=re.I):
        return "dead_link"
    if _placeholder_url(url) or "official statistics show" in blob:
        return "fake_or_placeholder_source"
    if source.get("source_title_url_mismatch_suspected"):
        return "source_mismatch"
    if title and not _source_traceable(source):
        return "title_only_source"
    if not _source_traceable(source):
        return "untraceable_source"
    return ""


def _source_lookup(source_registry: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for source in list(source_registry or []):
        if not isinstance(source, dict):
            continue
        for ref in _source_identity_refs(source):
            lookup.setdefault(ref, source)
    return lookup


def _source_merge_key(source: Dict[str, Any]) -> str:
    for key in ("url", "source_url", "document_ref", "document_id", "doc_id", "page_ref"):
        text = str(source.get(key) or "").strip().lower()
        if text:
            return f"{key}:{text}"
    for key in ("ref", "source_ref", "citation_ref"):
        normalized = normalize_citation_ref(source.get(key))
        if re.fullmatch(r"\[\d{1,5}\]", normalized or ""):
            return f"ref:{normalized}"
    ref = str(source.get("ref") or source.get("evidence_id") or source.get("source_ref") or source.get("citation_ref") or "").strip()
    return f"ref:{ref}" if ref else ""


def _source_location_key(source: Dict[str, Any]) -> str:
    url = _first_source_scalar(source.get("url"), source.get("source_url")).lower()
    if url:
        return f"url:{url}"
    doc = _first_source_scalar(
        source.get("document_ref"),
        source.get("document_id"),
        source.get("doc_id"),
        source.get("page_ref"),
    ).lower()
    return f"doc:{doc}" if doc else ""


def _can_merge_sources_by_ref(existing: Dict[str, Any], incoming: Dict[str, Any]) -> bool:
    existing_location = _source_location_key(existing)
    incoming_location = _source_location_key(incoming)
    if existing_location and incoming_location and existing_location != incoming_location:
        return False
    return True


_GENERIC_SOURCE_TITLE_RE = re.compile(
    r"^(?:"
    r"official(?:\s+(?:ai\s+agent\s+)?(?:statistics|data|source|report|disclosure))?"
    r"|official\s+statistics\s+show"
    r"|iqs\s*(?:source|来源)"
    r"|source"
    r"|[\w.-]+\s+source"
    r")$",
    flags=re.I,
)


def _source_title_value(source: Dict[str, Any]) -> str:
    return _first_source_scalar(source.get("title"), source.get("source_title"))


def _generic_source_title(title: Any) -> bool:
    text = re.sub(r"\s+", " ", str(title or "").strip())
    if not text:
        return True
    return bool(_GENERIC_SOURCE_TITLE_RE.fullmatch(text))


def _specific_source_title(source: Dict[str, Any]) -> str:
    for key in ("title", "source_title"):
        candidate = _clean_source_scalar(source.get(key))
        if candidate and not _generic_source_title(candidate):
            return candidate
    return ""


def _merge_source_titles(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    candidate = _specific_source_title(source)
    if not candidate:
        return
    title = str(target.get("title") or "").strip()
    source_title = str(target.get("source_title") or "").strip()
    if not title or _generic_source_title(title):
        if title and title != candidate:
            target.setdefault("original_title", title)
            target["generic_title_replaced"] = True
        target["title"] = candidate
    if not source_title or _generic_source_title(source_title):
        if source_title and source_title != candidate:
            target.setdefault("original_source_title", source_title)
            target["generic_title_replaced"] = True
        target["source_title"] = candidate


def _merge_aliases(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    aliases: List[Any] = []
    for key in ("ref", "id", "evidence_id", "source_ref", "citation_ref", "document_id", "doc_id"):
        aliases.append(source.get(key))
    for key in ("evidence_refs", "used_fact_refs", "source_refs", "refs"):
        aliases.extend(_as_list(source.get(key)))
    existing = _as_list(target.get("evidence_refs")) + _as_list(target.get("source_refs")) + _as_list(target.get("refs"))
    merged = _dedupe([*existing, *aliases], limit=200)
    if merged:
        target["evidence_refs"] = merged
        target["source_refs"] = merged


_METRIC_SOURCE_FIELDS = (
    "metric",
    "indicator",
    "value",
    "metric_value",
    "unit",
    "metric_unit",
    "period",
    "time_or_scope",
    "fact_type",
)


def _metric_fact_from_source_payload(source: Dict[str, Any]) -> Dict[str, Any]:
    value = str(source.get("value") or source.get("metric_value") or "").strip()
    unit = str(source.get("unit") or source.get("metric_unit") or "").strip()
    period = str(source.get("period") or source.get("time_or_scope") or "").strip()
    source_ref = str(
        source.get("source_ref")
        or source.get("citation_ref")
        or source.get("ref")
        or source.get("evidence_id")
        or ""
    ).strip()
    if not value or not source_ref or not (unit or period):
        return {}
    return {
        "evidence_id": str(source.get("evidence_id") or source.get("ref") or "").strip(),
        "source_ref": source_ref,
        "metric": str(source.get("metric") or source.get("indicator") or "").strip(),
        "value": value,
        "unit": unit,
        "period": period,
        "time_or_scope": str(source.get("time_or_scope") or "").strip(),
        "fact_type": str(source.get("fact_type") or "").strip() or "metric",
    }


def _merge_metric_facts(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    facts: List[Dict[str, Any]] = []
    for item in [*_as_list(target.get("metric_facts")), *_as_list(source.get("metric_facts"))]:
        if isinstance(item, dict):
            facts.append(item)
    source_metric = _metric_fact_from_source_payload(source)
    if source_metric:
        facts.append(source_metric)
    if not facts:
        return
    seen = set()
    merged: List[Dict[str, Any]] = []
    for fact in facts:
        key = (
            str(fact.get("evidence_id") or "").strip(),
            str(fact.get("source_ref") or "").strip(),
            str(fact.get("metric") or "").strip(),
            str(fact.get("value") or "").strip(),
            str(fact.get("unit") or "").strip(),
            str(fact.get("period") or fact.get("time_or_scope") or "").strip(),
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(dict(fact))
    target["metric_facts"] = merged


def merge_source_registries(*registries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    by_key: Dict[str, Dict[str, Any]] = {}
    by_ref: Dict[str, Dict[str, Any]] = {}

    def remember_refs(source: Dict[str, Any]) -> None:
        for ref in _source_identity_refs(source):
            normalized = normalize_citation_ref(ref)
            if re.fullmatch(r"\[\d{1,5}\]", normalized or ""):
                by_ref.setdefault(normalized, source)

    for registry in registries:
        for source in list(registry or []):
            if not isinstance(source, dict):
                continue
            copied = copy.deepcopy(source)
            _sanitize_source_metadata(copied)
            key = _source_merge_key(copied)
            if not key:
                continue
            existing = by_key.get(key)
            if existing is None:
                for ref in _source_identity_refs(copied):
                    normalized = normalize_citation_ref(ref)
                    if re.fullmatch(r"\[\d{1,5}\]", normalized or "") and normalized in by_ref:
                        candidate = by_ref[normalized]
                        if not _can_merge_sources_by_ref(candidate, copied):
                            continue
                        existing = candidate
                        break
            if existing is None:
                _merge_source_titles(copied, copied)
                _merge_aliases(copied, copied)
                _merge_metric_facts(copied, copied)
                by_key[key] = copied
                merged.append(copied)
                remember_refs(copied)
                continue
            _merge_aliases(existing, copied)
            _merge_metric_facts(existing, copied)
            _merge_source_titles(existing, copied)
            for field in (
                "url",
                "source_url",
                "title",
                "source_title",
                "date",
                "publisher",
                "source_level",
                "source_verification_status",
                *_METRIC_SOURCE_FIELDS,
            ):
                if not existing.get(field) and copied.get(field):
                    existing[field] = copied.get(field)
            by_key.setdefault(key, existing)
            remember_refs(existing)
    return merged


def _source_entry_from_evidence(item: Dict[str, Any]) -> Dict[str, Any]:
    nested_source = _as_dict(item.get("source"))
    evidence_id = str(item.get("evidence_id") or item.get("ref") or "").strip()
    source_ref = str(item.get("source_ref") or item.get("citation_ref") or "").strip()
    normalized_source_ref = normalize_citation_ref(source_ref)
    ref = normalized_source_ref if re.fullmatch(r"\[\d{1,5}\]", normalized_source_ref or "") else evidence_id or source_ref
    url = str(
        item.get("source_url")
        or item.get("url")
        or item.get("link")
        or nested_source.get("source_url")
        or nested_source.get("url")
        or nested_source.get("link")
        or ""
    ).strip()
    document_ref = str(
        item.get("document_ref")
        or item.get("document_id")
        or item.get("doc_id")
        or nested_source.get("document_ref")
        or nested_source.get("document_id")
        or nested_source.get("doc_id")
        or ""
    ).strip()
    title = _first_source_scalar(
        item.get("source_title"),
        item.get("title"),
        nested_source.get("source_title"),
        nested_source.get("title"),
        item.get("publisher"),
        nested_source.get("publisher"),
    )
    if not ref or not (url or document_ref or re.fullmatch(r"\[\d{1,5}\]", normalized_source_ref or "")):
        return {}
    entry = {
        "ref": ref,
        "evidence_id": evidence_id or ref,
        "source_ref": source_ref or ref,
        "citation_ref": str(item.get("citation_ref") or "").strip(),
        "url": url,
        "source_url": url,
        "document_ref": document_ref,
        "document_id": document_ref,
        "title": title,
        "source_title": title,
        "date": item.get("date") or item.get("publish_date") or item.get("published_at"),
        "publisher": _first_source_scalar(item.get("publisher"), item.get("source_publisher"), nested_source.get("publisher")),
        "source_level": item.get("source_level") or item.get("source_tier") or item.get("source_grade"),
        "source_verification_status": item.get("source_verification_status") or item.get("verification_status"),
    }
    for field in _METRIC_SOURCE_FIELDS:
        if item.get(field) not in (None, ""):
            entry[field] = item.get(field)
    entry["evidence_refs"] = _dedupe(
        [
            entry.get("ref"),
            entry.get("evidence_id"),
            entry.get("source_ref"),
            item.get("citation_ref"),
            *(_as_list(item.get("evidence_refs"))),
            *(_as_list(item.get("used_fact_refs"))),
        ],
        limit=50,
    )
    metric_fact = _metric_fact_from_source_payload(entry)
    if metric_fact:
        entry["metric_facts"] = [metric_fact]
    return entry


def _iter_evidence_items_from_container(container: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for key in EVIDENCE_COLLECTION_KEYS:
        for item in _as_list(container.get(key)):
            if isinstance(item, dict):
                yield item
    chapter_evidence = _as_dict(container.get("chapter_evidence"))
    for bucket in chapter_evidence.values():
        for item in _as_list(bucket):
            if isinstance(item, dict):
                yield item


def evidence_source_entries_from_package(
    *,
    evidence_package: Dict[str, Any] | None = None,
    chapter_evidence_packages: Sequence[Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    containers: List[Dict[str, Any]] = []
    package = _as_dict(evidence_package)
    if package:
        containers.append(package)
        containers.extend(
            item
            for item in _as_list(package.get("chapter_evidence_packages"))
            if isinstance(item, dict)
        )
    containers.extend(item for item in list(chapter_evidence_packages or []) if isinstance(item, dict))
    for container in containers:
        for item in _iter_evidence_items_from_container(container):
            entry = _source_entry_from_evidence(item)
            if entry:
                entries.append(entry)
    return merge_source_registries(entries)


def _section_refs(section: Dict[str, Any]) -> List[str]:
    refs: List[Any] = []
    for key in ("citation_refs", "used_fact_refs", "evidence_refs", "supporting_evidence_refs"):
        refs.extend(_as_list(section.get(key)))
    for key in ("claim", "reasoning", "mechanism", "counter_evidence", "actionable"):
        refs.extend(f"[{match}]" for match in re.findall(r"\[(\d{1,5})\]", str(section.get(key) or "")))
    return _dedupe(refs, limit=24)


def _claim_refs(claim: Dict[str, Any]) -> List[str]:
    refs: List[Any] = []
    for key in ("used_evidence_ids", "used_fact_refs", "evidence_refs", "supporting_evidence_refs", "supporting_evidence"):
        refs.extend(_as_list(claim.get(key)))
    return _dedupe(refs, limit=24)


def _iter_sections(chapters: Sequence[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for chapter in list(chapters or []):
        if not isinstance(chapter, dict):
            continue
        for section in _as_list(chapter.get("sections")):
            if isinstance(section, dict):
                yield section


def _public_source(source: Dict[str, Any], public_ref: str, index: int) -> Dict[str, Any]:
    copied = copy.deepcopy(source)
    original_ref = str(copied.get("ref") or "").strip()
    if original_ref and original_ref != public_ref:
        copied.setdefault("original_ref", original_ref)
    copied["ref"] = public_ref
    copied["source_id"] = f"SRC-{index:03d}"
    return copied


def build_citation_manifest(
    *,
    chapters: Sequence[Dict[str, Any]],
    claim_units: Sequence[Dict[str, Any]] | None = None,
    source_registry: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    lookup = _source_lookup(source_registry)
    section_citation_refs: Dict[str, List[str]] = {}
    evidence_to_citation: Dict[str, str] = {}
    appendix_sources: List[Dict[str, Any]] = []
    source_key_to_public: Dict[str, str] = {}
    missing_refs: List[str] = []
    excluded_cited_sources: List[Dict[str, Any]] = []
    filtered_missing_refs: List[str] = []
    filtered_cited_sources: List[Dict[str, Any]] = []
    used_refs: List[str] = []
    first_section_by_chapter: Dict[str, str] = {}
    section_ids = set()
    for chapter in list(chapters or []):
        chapter_id = str(_as_dict(chapter).get("chapter_id") or "").strip()
        for section in _as_list(_as_dict(chapter).get("sections")):
            if not isinstance(section, dict):
                continue
            section_id = str(section.get("section_id") or "").strip()
            if section_id:
                section_ids.add(section_id)
                if chapter_id:
                    first_section_by_chapter.setdefault(chapter_id, section_id)

    def assign(ref: str) -> str:
        text = str(ref or "").strip()
        if not text:
            return ""
        if text in evidence_to_citation:
            return evidence_to_citation[text]
        source = lookup.get(text) or lookup.get(normalize_citation_ref(text))
        if not source:
            filtered_missing_refs.append(text)
            return ""
        reason = _source_exclusion_reason(source)
        if reason:
            filtered_cited_sources.append(
                {
                    "ref": source.get("ref") or source.get("id") or text,
                    "title": source.get("title") or source.get("source_title"),
                    "url": source.get("url") or source.get("source_url"),
                    "reason": reason,
                }
            )
            return ""
        source_identity = _source_merge_key(source) or "|".join(sorted(_source_identity_refs(source)))
        public_ref = source_key_to_public.get(source_identity)
        if not public_ref:
            public_ref = f"[{len(appendix_sources) + 1}]"
            source_key_to_public[source_identity] = public_ref
            appendix_sources.append(_public_source(source, public_ref, len(appendix_sources) + 1))
        evidence_to_citation[text] = public_ref
        evidence_to_citation[normalize_citation_ref(text)] = public_ref
        for alias in _source_identity_refs(source):
            evidence_to_citation.setdefault(alias, public_ref)
        return public_ref

    for section in _iter_sections(chapters):
        section_id = str(section.get("section_id") or "").strip() or f"section_{len(section_citation_refs) + 1}"
        refs = _section_refs(section)
        used_refs.extend(refs)
        citations = _dedupe([assign(ref) for ref in refs], limit=8)
        if citations:
            section_citation_refs[section_id] = citations

    for claim in list(claim_units or []):
        claim_dict = _as_dict(claim)
        refs = _claim_refs(claim_dict)
        used_refs.extend(refs)
        citations = _dedupe([assign(ref) for ref in refs], limit=8)
        for ref in refs:
            assign(ref)
        section_id = str(claim_dict.get("section_id") or "").strip()
        if not section_id or section_id not in section_ids:
            section_id = first_section_by_chapter.get(str(claim_dict.get("chapter_id") or "").strip(), "")
        if section_id and section_id in section_ids and citations:
            section_citation_refs[section_id] = _dedupe(
                [*_as_list(section_citation_refs.get(section_id)), *citations],
                limit=8,
            )

    missing_refs = _dedupe(missing_refs, limit=50)
    filtered_missing_refs = _dedupe(filtered_missing_refs, limit=50)
    deduped_excluded: List[Dict[str, Any]] = []
    seen_excluded = set()
    for item in excluded_cited_sources:
        if not item:
            continue
        key = (
            str(item.get("ref") or ""),
            str(item.get("url") or ""),
            str(item.get("title") or ""),
            str(item.get("reason") or ""),
        )
        if key in seen_excluded:
            continue
        seen_excluded.add(key)
        deduped_excluded.append(dict(item))
    excluded_cited_sources = deduped_excluded

    filtered_excluded: List[Dict[str, Any]] = []
    seen_filtered = set()
    for item in filtered_cited_sources:
        if not item:
            continue
        key = (
            str(item.get("ref") or ""),
            str(item.get("url") or ""),
            str(item.get("title") or ""),
            str(item.get("reason") or ""),
        )
        if key in seen_filtered:
            continue
        seen_filtered.add(key)
        filtered_excluded.append(dict(item))
    filtered_cited_sources = filtered_excluded

    if not appendix_sources and used_refs:
        missing_refs = filtered_missing_refs
        excluded_cited_sources = filtered_cited_sources
        status = "blocked"
    elif not used_refs:
        status = "warning"
    else:
        status = "ok"
    return {
        "citation_manifest_status": status,
        "section_citation_refs": section_citation_refs,
        "evidence_to_citation": evidence_to_citation,
        "appendix_sources": appendix_sources,
        "missing_evidence_refs": missing_refs,
        "missing_source_ref_count": len(missing_refs),
        "excluded_cited_sources": excluded_cited_sources,
        "excluded_source_count": len(excluded_cited_sources),
        "filtered_unresolved_refs": filtered_missing_refs,
        "filtered_unresolved_ref_count": len(filtered_missing_refs),
        "filtered_cited_sources": filtered_cited_sources,
        "filtered_cited_source_count": len(filtered_cited_sources),
        "orphan_citation_count": 0,
        "used_evidence_ref_count": len(_dedupe(used_refs, limit=10000)),
    }


def attach_manifest_citations(
    chapters: Sequence[Dict[str, Any]],
    manifest: Dict[str, Any],
) -> List[Dict[str, Any]]:
    section_map = _as_dict(manifest.get("section_citation_refs"))
    result: List[Dict[str, Any]] = []
    for chapter in list(chapters or []):
        if not isinstance(chapter, dict):
            continue
        copied = copy.deepcopy(chapter)
        sections = []
        for section in _as_list(copied.get("sections")):
            if not isinstance(section, dict):
                continue
            section_id = str(section.get("section_id") or "").strip()
            section_copy = dict(section)
            citations = _as_list(section_map.get(section_id))
            if citations:
                section_copy["citation_refs"] = citations
            sections.append(section_copy)
        copied["sections"] = sections
        result.append(copied)
    return result


def manifest_appendix_sources(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [dict(source) for source in _as_list(_as_dict(manifest).get("appendix_sources")) if isinstance(source, dict)]
