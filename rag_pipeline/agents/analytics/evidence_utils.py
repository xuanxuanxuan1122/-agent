from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .analytics_contracts import as_dict, as_list, compact, dedupe


ENTITY_BLOCKLIST_RE = re.compile(
    r"(?:^LOOK\s*[~\-_=#]+$|"
    r"[~]{4,}|"
    r"^\d{4}年.{0,50}(?:报告|分析|研判|前瞻|展望|白皮书|研究)|"
    r"(?:报告|分析|研判|前瞻|展望|白皮书|研究报告)$)"
)
SHORT_ENTITY_RE = re.compile(
    r"^(?:[\u4e00-\u9fffA-Za-z0-9&.\-（）()·]{2,28})$"
)


def text_blob(*values: Any) -> str:
    return " ".join(str(value or "") for value in values)


def contains_any(text: str, terms: Sequence[str]) -> bool:
    lower = str(text or "").lower()
    return any(str(term or "").lower() in lower for term in terms if str(term or "").strip())


def first_term(text: str, terms: Sequence[str]) -> str:
    lower = str(text or "").lower()
    for term in terms:
        term_text = str(term or "").strip()
        if term_text and term_text.lower() in lower:
            return term_text
    return ""


def parse_number(value: Any) -> Optional[float]:
    text = str(value or "").replace(",", "")
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def extract_year(*values: Any) -> Optional[int]:
    text = text_blob(*values)
    years = [int(item) for item in re.findall(r"\b(20\d{2}|19\d{2})\b", text)]
    if not years:
        return None
    return max(years)


def source_ref(item: Dict[str, Any]) -> str:
    for value in (
        item.get("evidence_ref"),
        item.get("source_ref"),
        item.get("citation_ref"),
        item.get("ref"),
        item.get("evidence_id"),
        *as_list(item.get("source_refs")),
    ):
        text = str(value or "").strip()
        if text:
            return text
    return ""


def source_level(item: Dict[str, Any]) -> str:
    return compact(item.get("source_level") or as_dict(item.get("source")).get("level"), 20)


def evidence_text(item: Dict[str, Any]) -> str:
    raw = as_dict(item.get("raw"))
    return compact(
        item.get("fact")
        or item.get("clean_fact")
        or item.get("content")
        or item.get("text")
        or item.get("summary")
        or raw.get("fact")
        or raw.get("text")
        or text_blob(item.get("metric_name") or item.get("metric"), item.get("value"), item.get("subject")),
        420,
    )


def is_valid_entity_subject(value: Any) -> bool:
    text = compact(value, 80)
    if not text:
        return False
    if len(text) > 28:
        return False
    if ENTITY_BLOCKLIST_RE.search(text):
        return False
    if "行业" in text and not re.search(r"(协会|研究院|科学院|信通院)$", text):
        return False
    return bool(SHORT_ENTITY_RE.match(text))


def evidence_subject(item: Dict[str, Any], *, fallback: str = "") -> str:
    raw = as_dict(item.get("raw"))
    for value in (
        item.get("company"),
        item.get("enterprise"),
        item.get("entity"),
        item.get("subject"),
        raw.get("company"),
        raw.get("enterprise"),
        raw.get("entity"),
        raw.get("subject"),
        fallback,
    ):
        subject = compact(value, 80)
        if is_valid_entity_subject(subject):
            return subject
    return ""


def iter_candidate_items(
    *,
    chapter_evidence_packages: Sequence[Dict[str, Any]],
    metric_normalization_table: Sequence[Dict[str, Any]] | None = None,
    collections: Sequence[str] = ("core_evidence", "supporting_evidence", "table_evidence", "evidence_items"),
) -> Iterable[Dict[str, Any]]:
    for item in list(metric_normalization_table or []):
        if isinstance(item, dict):
            yield item
    for package in list(chapter_evidence_packages or []):
        if not isinstance(package, dict):
            continue
        chapter_id = str(package.get("chapter_id") or "")
        chapter_title = str(package.get("chapter_title") or "")
        chapter_question = str(package.get("chapter_question") or "")
        for collection in collections:
            for item in as_list(package.get(collection)):
                if isinstance(item, dict):
                    yield {
                        **item,
                        "chapter_id": item.get("chapter_id") or chapter_id,
                        "chapter_title": item.get("chapter_title") or chapter_title,
                        "chapter_question": item.get("chapter_question") or chapter_question,
                    }


def row_dedupe_key(row: Dict[str, Any], keys: Sequence[str]) -> str:
    return re.sub(r"\s+", "", "|".join(str(row.get(key) or "").lower() for key in keys))


def refs_from_items(items: Sequence[Dict[str, Any]], *, limit: int = 30) -> List[str]:
    refs: List[str] = []
    for item in items:
        refs.extend(as_list(item.get("evidence_refs")))
        ref = item.get("evidence_ref")
        if ref:
            refs.append(ref)
    return dedupe(refs, limit=limit)
