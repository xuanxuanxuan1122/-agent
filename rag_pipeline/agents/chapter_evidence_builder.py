from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


CORE_LIMIT = 6
SUPPORT_LIMIT = 8
COUNTER_LIMIT = 4
METRIC_LIMIT = 5
CASE_LIMIT = 4
DIRECTIONAL_LIMIT = 3

EVIDENCE_LAYER_KEYS = (
    "core_evidence",
    "supporting_evidence",
    "metric_evidence",
    "case_evidence",
    "counter_evidence",
    "directional_evidence",
    "sample_evidence",
)

CHAPTER_MATCH_MIN_SCORE = 18

GENERIC_CHAPTER_TERMS = {
    "agent",
    "agents",
    "ai",
    "智能体",
    "人工智能",
    "发展",
    "市场",
    "行业",
    "生态",
    "需求",
    "技术",
    "产品",
    "数据",
    "来源",
    "证据",
    "分析",
    "报告",
    "如何",
    "哪些",
    "什么",
}


BAD_FACT_PATTERNS = [
    r"\u519c\u4e1a\u4eba\u5de5\u667a\u80fd",
    r"\u6570\u636e\u6295\u6bd2",
    r"\u7eba\u7ec7",
    r"\u667a\u80fd\u624b\u673a",
    r"Scribd",
    r"SEO",
    r"example\.(?:com|gov|org)",
    r"Official data shows AI agent adoption reached 50%",
    r"^URL[:\uff1a]",
]


LOW_QUALITY_SOURCE_PATTERNS = [
    r"twitter\.com|x\.com|instagram\.com|facebook\.com",
    r"baike\.baidu\.com|baijiahao\.baidu\.com",
    r"blog\.csdn\.net|cnblogs\.com|juejin\.cn",
    r"fxbaogao\.com|sgpjbg\.com|jazzyear\.com",
]


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, dict) and isinstance(value.get("sample"), list):
        return list(value.get("sample") or [])
    return list(value) if isinstance(value, list) else []


def _compact(value: Any, max_chars: int = 320) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "..."


def _fact_text(item: Dict[str, Any]) -> str:
    return _compact(
        item.get("fact")
        or item.get("clean_fact")
        or item.get("content")
        or item.get("evidence")
        or item.get("summary")
        or item.get("answer")
        or item.get("text"),
        900,
    )


def _bad_fact_text(text: str) -> bool:
    if not str(text or "").strip():
        return True
    return any(re.search(pattern, text, flags=re.I) for pattern in BAD_FACT_PATTERNS)


def _source_url(item: Dict[str, Any]) -> str:
    source = _as_dict(item.get("source"))
    return str(
        item.get("source_url")
        or item.get("url")
        or item.get("link")
        or source.get("url")
        or source.get("source_url")
        or ""
    ).strip()


def _source_level(item: Dict[str, Any]) -> str:
    return str(item.get("source_level") or item.get("credibility") or item.get("source_grade") or "C").strip().upper()


def _traceable(item: Dict[str, Any]) -> bool:
    return bool(
        _source_url(item)
        or str(item.get("document_id") or item.get("doc_id") or item.get("page_ref") or "").strip()
    )


def _source_identity_bad(item: Dict[str, Any]) -> bool:
    haystack = " ".join(
        str(item.get(key) or "")
        for key in (
            "source_url",
            "url",
            "link",
            "title",
            "source_title",
            "publisher",
            "source",
            "source_ref",
            "citation_ref",
            "ref",
        )
    )
    if item.get("source_title_url_mismatch_suspected"):
        return True
    if re.search(r"\bIQS\s*来源\b|^IQS来源$", haystack, flags=re.I):
        return True
    if re.search(r"example\.(?:com|gov|org)", haystack, flags=re.I):
        return True
    if not _traceable(item):
        return True
    return False


def _low_quality_source(item: Dict[str, Any]) -> bool:
    haystack = " ".join(
        str(item.get(key) or "")
        for key in ("source_url", "url", "title", "source_title", "publisher", "source")
    )
    source_type = str(item.get("source_type") or item.get("type") or "").strip().lower()
    if source_type in {"self_media", "social", "forum", "wiki", "seo", "search_page", "aggregator"}:
        return True
    if _source_level(item) == "D":
        return True
    if _source_identity_bad(item):
        return True
    return any(re.search(pattern, haystack, flags=re.I) for pattern in LOW_QUALITY_SOURCE_PATTERNS)


def _invalid_metric(item: Dict[str, Any]) -> bool:
    metric = str(item.get("metric") or item.get("indicator") or "").strip()
    value = str(item.get("value") or item.get("display_value") or item.get("numeric_value") or "").strip()
    fact = _fact_text(item)
    metric_lower = metric.lower()
    if str(item.get("metric_validation_status") or "").strip().lower() == "invalid":
        return True
    if metric_lower in {"source_check", "status", "http_status", "response_code"} and re.fullmatch(r"[1-5]\d{2}", value):
        return True
    if re.search(r"\bsource_check\s*[:=]\s*[1-5]\d{2}\b", fact, flags=re.I):
        return True
    if value and re.fullmatch(r"-?\d{1,3}(?:\.0)?", value) and re.search(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|T\d{1,2}:\d{2}", fact):
        return True
    if metric in {"\u5173\u952e\u4e8b\u5b9e", "\u653f\u7b56\u76d1\u7ba1", "\u653f\u7b56\u76ee\u6807"} and re.fullmatch(r"-?\d{1,3}(?:\.0)?", value):
        return True
    if re.search(r"\u653f\u7b56|\u76ee\u6807|\u76d1\u7ba1", metric) and re.match(r"-\d", value):
        return True
    if re.search(r"\u6210\u672c", metric) and (re.search(r"\u5bb6$", value) or not fact):
        return True
    if re.search(r"\u5e02\u573a\u89c4\u6a21|\u878d\u8d44", metric) and re.search(r"%", value):
        return True
    return False


def _metric_ready(item: Dict[str, Any]) -> bool:
    metric = str(item.get("metric") or item.get("indicator") or "").strip()
    value = str(item.get("value") or item.get("display_value") or item.get("numeric_value") or "").strip()
    if not metric or not value or _invalid_metric(item):
        return False
    return bool(re.search(r"\d", value) or _as_list(item.get("numeric_values")))


def _ref_values(item: Dict[str, Any]) -> List[str]:
    values: List[str] = []
    for key in ("evidence_id", "id", "ref", "source_ref", "citation_ref", "source_id", "document_id"):
        value = str(item.get(key) or "").strip()
        if value:
            values.append(value)
    url = _source_url(item)
    if url:
        values.append(url)
    return values


def _normalize_ref(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return re.sub(r"^\[|\]$", "", text).strip().lower()


def _evidence_ref(item: Dict[str, Any]) -> str:
    for key in ("source_ref", "citation_ref", "ref", "evidence_id", "id"):
        value = str(item.get(key) or "").strip()
        if value:
            return value
    return _source_url(item)


def _dedupe_items(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        key = _normalize_ref(_evidence_ref(item)) or re.sub(r"\W+", "", _fact_text(item).lower())[:120]
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _source_keys(item: Dict[str, Any]) -> List[str]:
    source = _as_dict(item.get("source"))
    values = []
    for key in ("ref", "source_ref", "citation_ref", "source_id", "id"):
        values.append(str(item.get(key) or source.get(key) or "").strip())
    url = _source_url(item)
    if url:
        values.append(url)
    source_id = str(item.get("source_id") or source.get("source_id") or "").strip()
    if re.fullmatch(r"\d{1,3}", source_id):
        values.append(f"[{source_id}]")
    return [_normalize_ref(value) for value in values if _normalize_ref(value)]


def _source_registry_lookup(source_registry: Optional[Sequence[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    registry = [dict(source) for source in list(source_registry or []) if isinstance(source, dict)]
    title_hosts: Dict[str, set[str]] = {}
    for source in registry:
        title = re.sub(r"\s+", " ", str(source.get("title") or source.get("source_title") or "").strip()).lower()
        url = str(source.get("url") or source.get("source_url") or "").strip().lower()
        host_match = re.search(r"https?://([^/]+)", url)
        host = host_match.group(1) if host_match else ""
        if title and host:
            title_hosts.setdefault(title, set()).add(host)
    for source in registry:
        title = re.sub(r"\s+", " ", str(source.get("title") or source.get("source_title") or "").strip()).lower()
        if title and len(title_hosts.get(title, set())) > 1:
            source["source_title_url_mismatch_suspected"] = True
        for key in _source_keys(source):
            lookup.setdefault(key, source)
    return lookup


def _registry_source_for_item(item: Dict[str, Any], lookup: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    for key in _source_keys(item):
        if key in lookup:
            return lookup[key]
    return {}


def _normalize_item(item: Dict[str, Any], source_lookup: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    copied = dict(item)
    registry_source = _registry_source_for_item(copied, source_lookup or {})
    if registry_source:
        copied["source_registry_ref"] = registry_source.get("ref") or copied.get("source_registry_ref")
        copied["source_title_url_mismatch_suspected"] = bool(
            copied.get("source_title_url_mismatch_suspected")
            or registry_source.get("source_title_url_mismatch_suspected")
        )
        for target, candidates in {
            "source_url": ("url", "source_url"),
            "source_title": ("title", "source_title"),
            "publisher": ("publisher", "source"),
            "source_level": ("source_level", "credibility"),
            "source_type": ("source_type", "type"),
        }.items():
            if copied.get(target):
                continue
            for candidate in candidates:
                value = registry_source.get(candidate)
                if value:
                    copied[target] = value
                    break
    copied["fact"] = _fact_text(copied)
    copied.setdefault("ref", copied.get("evidence_id") or copied.get("source_ref") or copied.get("citation_ref") or "")
    copied.setdefault("source_ref", copied.get("source_ref") or copied.get("citation_ref") or copied.get("ref") or "")
    copied.setdefault("source_level", _source_level(copied))
    copied["source_traceable"] = _traceable(copied)
    copied["metric_ready"] = _metric_ready(copied)
    return copied


def _seed_items(evidence_package: Dict[str, Any], source_lookup: Optional[Dict[str, Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    seeds: List[Dict[str, Any]] = []
    for key in ("analysis_ready_evidence", "clean_evidence_list", "normalized_evidence", "raw_data_points"):
        for item in _as_list(evidence_package.get(key)):
            if not isinstance(item, dict):
                continue
            normalized = _normalize_item(item, source_lookup)
            fact = _fact_text(normalized)
            if _bad_fact_text(fact) or _invalid_metric(normalized) or _low_quality_source(normalized):
                continue
            seeds.append(normalized)
    return _dedupe_items(seeds)


def _items_from_existing_chapter(
    chapter: Dict[str, Any],
    chapter_id: str,
    source_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for key in (
        "core_evidence",
        "supporting_evidence",
        "counter_evidence",
        "metric_evidence",
        "case_evidence",
        "directional_evidence",
        "sample_evidence",
    ):
        for item in _as_list(chapter.get(key)):
            if not isinstance(item, dict):
                continue
            fact = _fact_text(item)
            if _bad_fact_text(fact) or _invalid_metric(item):
                continue
            copied = dict(item)
            copied.setdefault("chapter_id", chapter_id)
            normalized = _normalize_item(copied, source_lookup)
            if _low_quality_source(normalized):
                continue
            items.append(normalized)
    return _dedupe_items(items)


def _lookup_by_ref(seeds: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for item in seeds:
        for ref in _ref_values(item):
            key = _normalize_ref(ref)
            if key and key not in lookup:
                lookup[key] = item
    return lookup


def _text_key(value: Any) -> str:
    return re.sub(r"[\s\W_]+", "", str(value or "").lower())


def _chapter_terms(chapter: Dict[str, Any]) -> List[str]:
    text = " ".join(
        str(chapter.get(key) or "")
        for key in ("chapter_title", "chapter_question", "title", "unit_title", "core_question")
    )
    terms = set()
    for token in re.findall(r"[A-Za-z][A-Za-z0-9+.-]{2,}|[\u4e00-\u9fff]{2,}", text):
        if token.lower() in GENERIC_CHAPTER_TERMS:
            continue
        terms.add(token.lower())
    return list(terms)


def _with_binding(item: Dict[str, Any], *, reason: str, score: int, chapter_id: str) -> Dict[str, Any]:
    copied = dict(item)
    copied["binding_reason"] = reason
    copied["binding_score"] = score
    copied["bound_chapter_id"] = chapter_id
    copied.setdefault("chapter_id", chapter_id)
    return copied


def _hydrated_layer_count(package: Dict[str, Any]) -> int:
    return sum(len(_as_list(package.get(key))) for key in EVIDENCE_LAYER_KEYS)


def _chapter_identity(chapter: Dict[str, Any], index: int) -> Tuple[str, str, str]:
    chapter_id = str(chapter.get("chapter_id") or chapter.get("unit_id") or chapter.get("id") or f"ch_{index:02d}").strip()
    title = str(chapter.get("chapter_title") or chapter.get("unit_title") or chapter.get("title") or f"\u7ae0\u8282 {index}").strip()
    question = str(chapter.get("chapter_question") or chapter.get("core_question") or chapter.get("chapter_role") or title).strip()
    return chapter_id, title, question


def _diagnostic_payload(evidence_analysis_by_chapter: Dict[str, Any], chapter_id: str, title: str) -> Dict[str, Any]:
    for key in (chapter_id, title):
        payload = _as_dict(evidence_analysis_by_chapter.get(key))
        if payload:
            return payload
    title_key = _text_key(title)
    for key, payload in evidence_analysis_by_chapter.items():
        if title_key and (_text_key(key) in title_key or title_key in _text_key(key)):
            return _as_dict(payload)
    return {}


def _refs_from_diagnostics(payload: Dict[str, Any]) -> List[str]:
    refs: List[str] = []
    for key in ("sample_evidence_refs", "evidence_refs", "source_refs", "claim_ready_evidence_refs"):
        refs.extend(str(ref or "").strip() for ref in _as_list(payload.get(key)) if str(ref or "").strip())
    return refs


def _item_chapter_score(item: Dict[str, Any], chapter: Dict[str, Any], chapter_id: str, title: str, terms: Sequence[str]) -> int:
    score = 0
    dim = str(item.get("chapter_id") or item.get("dimension") or item.get("hypothesis_id") or item.get("dimension_id") or "").strip()
    if dim == chapter_id:
        score += 100
    dim_key = _text_key(dim)
    title_key = _text_key(title)
    if dim_key and title_key and (dim_key in title_key or title_key in dim_key):
        score += 80
    fact_key = _text_key(_fact_text(item))
    for term in terms:
        if _text_key(term) and _text_key(term) in fact_key:
            score += 6
    metric = str(item.get("metric") or item.get("indicator") or "").lower()
    if any(term in metric for term in terms):
        score += 10
    return score


def _binding_reason(score: int) -> str:
    if score >= 100:
        return "chapter_id"
    if score >= 80:
        return "dimension_title"
    if score >= CHAPTER_MATCH_MIN_SCORE:
        return "chapter_terms"
    return "unmatched"


def _role_text(item: Dict[str, Any]) -> str:
    return " ".join(
        str(item.get(key) or "")
        for key in ("evidence_role", "proof_role", "allowed_use", "metric", "indicator", "fact", "content", "evidence", "summary")
    )


def _is_counter(item: Dict[str, Any]) -> bool:
    return bool(re.search(r"\u98ce\u9669|\u53cd\u8bc1|\u5931\u8d25|\u4e0b\u964d|\u4e0d\u53ca\u9884\u671f|\u8fb9\u754c|counter|risk", _role_text(item), flags=re.I))


def _is_case(item: Dict[str, Any]) -> bool:
    return bool(re.search(r"\u5ba2\u6237|\u6848\u4f8b|\u8ba2\u5355|\u4e2d\u6807|\u91c7\u8d2d|\u843d\u5730|\u90e8\u7f72|case", _role_text(item), flags=re.I))


def _is_core(item: Dict[str, Any]) -> bool:
    level = _source_level(item)
    allowed = str(item.get("allowed_use") or item.get("evidence_role") or "").lower()
    return level in {"A", "B"} and _traceable(item) and not _is_counter(item) and ("clue" not in allowed)


def _rank_items(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def score(item: Dict[str, Any]) -> Tuple[int, str]:
        level = _source_level(item)
        level_score = {"A": 40, "B": 30, "C": 15}.get(level, 0)
        return (
            level_score
            + (20 if _traceable(item) else 0)
            + (15 if _metric_ready(item) else 0)
            + min(len(_fact_text(item)) // 80, 8),
            _evidence_ref(item),
        )

    return sorted(_dedupe_items(items), key=score, reverse=True)


def _layer_evidence(items: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    ranked = _rank_items(items)
    core = [item for item in ranked if _is_core(item)][:CORE_LIMIT]
    used = {_normalize_ref(_evidence_ref(item)) for item in core}
    counters = [item for item in ranked if _is_counter(item) and _normalize_ref(_evidence_ref(item)) not in used][:COUNTER_LIMIT]
    used.update(_normalize_ref(_evidence_ref(item)) for item in counters)
    metrics = [item for item in ranked if _metric_ready(item)][:METRIC_LIMIT]
    used.update(_normalize_ref(_evidence_ref(item)) for item in metrics)
    cases = [item for item in ranked if _is_case(item)][:CASE_LIMIT]
    used.update(_normalize_ref(_evidence_ref(item)) for item in cases)
    directional = [
        item
        for item in ranked
        if _normalize_ref(_evidence_ref(item)) not in used
        and _traceable(item)
        and (
            _source_level(item) == "C"
            or str(item.get("allowed_use") or "").strip().lower() == "directional_signal"
        )
    ][:DIRECTIONAL_LIMIT]
    used.update(_normalize_ref(_evidence_ref(item)) for item in directional)
    supporting = [
        item
        for item in ranked
        if _normalize_ref(_evidence_ref(item)) not in used
        and _traceable(item)
        and _source_level(item) in {"A", "B"}
    ][:SUPPORT_LIMIT]
    used.update(_normalize_ref(_evidence_ref(item)) for item in supporting)
    def mark_layer(layer_items: Sequence[Dict[str, Any]], layer: str, role: str, allowed_use: str = "") -> List[Dict[str, Any]]:
        marked: List[Dict[str, Any]] = []
        for item in layer_items:
            copied = dict(item)
            copied["chapter_evidence_layer"] = layer
            copied.setdefault("evidence_role", role)
            if allowed_use:
                copied.setdefault("allowed_use", allowed_use)
            elif _source_level(copied) == "C":
                copied.setdefault("allowed_use", "directional_signal")
            marked.append(copied)
        return marked

    core = mark_layer(core, "core_evidence", "core_claim", "core_claim")
    supporting = mark_layer(supporting, "supporting_evidence", "supporting")
    counters = mark_layer(counters, "counter_evidence", "counter", "counter_evidence")
    metrics = mark_layer(metrics, "metric_evidence", "supporting", "supporting")
    cases = mark_layer(cases, "case_evidence", "supporting", "directional_signal")
    directional = mark_layer(directional, "directional_evidence", "clue", "directional_signal")
    sample = _dedupe_items([*core, *supporting, *metrics, *cases, *counters, *directional])[:12]
    return {
        "core_evidence": core,
        "supporting_evidence": supporting,
        "counter_evidence": counters,
        "metric_evidence": metrics,
        "case_evidence": cases,
        "directional_evidence": directional,
        "sample_evidence": sample,
    }


def _existing_chapters(report_blueprint: Dict[str, Any], existing: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    existing_by_id = {
        str(item.get("chapter_id") or "").strip(): dict(item)
        for item in existing
        if isinstance(item, dict) and str(item.get("chapter_id") or "").strip()
    }
    chapters = []
    for index, chapter in enumerate(_as_list(report_blueprint.get("chapters")), start=1):
        if not isinstance(chapter, dict):
            continue
        chapter_id, title, question = _chapter_identity(chapter, index)
        payload = existing_by_id.get(chapter_id, {})
        payload.update({"chapter_id": chapter_id, "chapter_title": title, "chapter_question": question})
        chapters.append(payload)
    if chapters:
        return chapters
    return [dict(item) for item in existing if isinstance(item, dict)]


def build_chapter_evidence_packages_from_evidence_package(
    *,
    report_blueprint: Dict[str, Any],
    evidence_package: Dict[str, Any],
    existing_chapter_evidence_packages: Optional[Sequence[Dict[str, Any]]] = None,
    source_registry: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Build full per-chapter evidence packages from the central evidence pool.

    This is intentionally deterministic and conservative: it never upgrades
    evidence quality, and it records unresolved refs instead of silently
    discarding them.
    """

    evidence_package = _as_dict(evidence_package)
    source_lookup = _source_registry_lookup(source_registry)
    evidence_analysis_by_chapter = _as_dict(evidence_package.get("evidence_analysis_by_chapter"))
    seeds = _seed_items(evidence_package, source_lookup)
    lookup = _lookup_by_ref(seeds)
    chapters = _existing_chapters(_as_dict(report_blueprint), list(existing_chapter_evidence_packages or []))
    if not chapters:
        return []
    result: List[Dict[str, Any]] = []
    for index, chapter in enumerate(chapters, start=1):
        chapter = dict(chapter)
        chapter_id, title, question = _chapter_identity(chapter, index)
        diagnostics = _diagnostic_payload(evidence_analysis_by_chapter, chapter_id, title)
        resolved: List[Dict[str, Any]] = []
        unresolved_refs: List[str] = []
        for ref in _refs_from_diagnostics(diagnostics):
            item = lookup.get(_normalize_ref(ref))
            if item:
                resolved.append(_with_binding(item, reason="evidence_analysis_ref", score=120, chapter_id=chapter_id))
            else:
                unresolved_refs.append(ref)
        terms = _chapter_terms({"chapter_title": title, "chapter_question": question})
        scored = [
            (score, _with_binding(item, reason=_binding_reason(score), score=score, chapter_id=chapter_id))
            for item in seeds
            for score in [_item_chapter_score(item, chapter, chapter_id, title, terms)]
            if score >= CHAPTER_MATCH_MIN_SCORE
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        existing_items = [
            _with_binding(item, reason="existing_chapter_package", score=110, chapter_id=chapter_id)
            for item in _items_from_existing_chapter(chapter, chapter_id, source_lookup)
        ]
        matched = _dedupe_items([*resolved, *existing_items, *[item for _, item in scored]])
        layered = _layer_evidence(matched)
        hydrated_count = sum(len(layered.get(key, [])) for key in EVIDENCE_LAYER_KEYS)
        binding_reasons: Dict[str, int] = {}
        for item in matched:
            reason = str(item.get("binding_reason") or "unknown")
            binding_reasons[reason] = binding_reasons.get(reason, 0) + 1
        metadata = _as_dict(chapter.get("metadata"))
        metadata["chapter_evidence_rebuilt"] = True
        metadata["hydrated_evidence"] = bool(hydrated_count)
        metadata["hydrated_evidence_count"] = hydrated_count
        metadata["source_pool_size"] = len(seeds)
        metadata["existing_chapter_evidence_count"] = len(existing_items)
        metadata["matched_evidence_count"] = len(matched)
        metadata["binding_reasons"] = binding_reasons
        metadata["unresolved_evidence_refs"] = unresolved_refs
        metadata["evidence_binding_counts"] = {
            key: len(value)
            for key, value in layered.items()
            if isinstance(value, list)
        }
        chapter.update(layered)
        chapter.update(
            {
                "chapter_id": chapter_id,
                "chapter_title": title,
                "chapter_question": question,
                "metadata": metadata,
                "unresolved_evidence_refs": unresolved_refs,
                "hydrated_evidence": bool(hydrated_count),
                "hydrated_evidence_count": hydrated_count,
                "binding_reasons": binding_reasons,
                "core_evidence_count": len(layered["core_evidence"]),
                "supporting_evidence_count": len(layered["supporting_evidence"]),
                "metric_evidence_count": len(layered["metric_evidence"]),
                "counter_evidence_count": len(layered["counter_evidence"]),
                "case_evidence_count": len(layered["case_evidence"]),
                "directional_evidence_count": len(layered["directional_evidence"]),
                "unresolved_evidence_ref_count": len(unresolved_refs),
            }
        )
        result.append(chapter)
    return result
