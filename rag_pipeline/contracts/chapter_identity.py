from __future__ import annotations

import re
from typing import Any, Dict, List


_CANONICAL_CHAPTER_ID_RE = re.compile(r"ch_\d{2,3}$", re.I)


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if value in (None, ""):
        return []
    return [value]


def _norm(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "").strip().lower())


def _canonical_id(value: Any) -> str:
    raw = str(value or "").strip()
    if _CANONICAL_CHAPTER_ID_RE.fullmatch(raw):
        return raw.lower()
    return ""


def _append_alias(aliases: List[str], value: Any) -> None:
    text = str(value or "").strip()
    if text and text not in aliases:
        aliases.append(text)


def _chapter_items(blueprint: Dict[str, Any], research_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for source in (blueprint, research_plan):
        for item in _as_list(_as_dict(source).get("chapters")):
            if isinstance(item, dict):
                items.append(item)
    return items


def _hypothesis_aliases_by_index(research_plan: Dict[str, Any]) -> Dict[int, List[str]]:
    aliases: Dict[int, List[str]] = {}
    for index, hypothesis in enumerate(_as_list(_as_dict(research_plan).get("hypotheses"))):
        item = _as_dict(hypothesis)
        values = [
            item.get("hypothesis_id"),
            item.get("id"),
            item.get("statement"),
            item.get("hypothesis"),
            item.get("title"),
        ]
        aliases[index] = [str(value).strip() for value in values if str(value or "").strip()]
    return aliases


def build_chapter_identity_map(
    *,
    blueprint: Dict[str, Any] | None = None,
    research_plan: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Return canonical chapter IDs plus every safe alias for lookup.

    Only stable ``ch_XX`` IDs are canonical. Hypothesis IDs, dimension names,
    and raw titles are aliases so they can be resolved before reaching LLM
    analysis, but they should never become public analysis chapter IDs.
    """

    blueprint_dict = _as_dict(blueprint)
    research_plan_dict = _as_dict(research_plan)
    hypotheses_by_index = _hypothesis_aliases_by_index(research_plan_dict)
    canonical: Dict[str, Dict[str, Any]] = {}
    alias_to_id: Dict[str, str] = {}

    for index, item in enumerate(_chapter_items(blueprint_dict, research_plan_dict)):
        chapter_id = _canonical_id(item.get("chapter_id") or item.get("id"))
        if not chapter_id:
            continue
        title = str(item.get("chapter_title") or item.get("title") or "").strip()
        question = str(
            item.get("chapter_question")
            or item.get("core_question")
            or item.get("question")
            or title
            or chapter_id
        ).strip()
        aliases: List[str] = []
        for value in (
            chapter_id,
            item.get("chapter_id"),
            item.get("id"),
            title,
            question,
            item.get("hypothesis_id"),
            item.get("dimension_id"),
            item.get("dimension"),
            item.get("dimension_name"),
            *_as_list(item.get("chapter_id_aliases")),
            *_as_list(item.get("aliases")),
            *hypotheses_by_index.get(index, []),
        ):
            _append_alias(aliases, value)
        canonical[chapter_id] = {
            "chapter_id": chapter_id,
            "chapter_title": title or question or chapter_id,
            "chapter_question": question or title or chapter_id,
            "chapter_id_aliases": aliases,
        }
        for alias in aliases:
            alias_key = _norm(alias)
            if alias_key:
                alias_to_id.setdefault(alias_key, chapter_id)

    return {"canonical": canonical, "alias_to_id": alias_to_id}


def resolve_canonical_chapter_id(identity: Dict[str, Any], raw_value: Any) -> str:
    raw = str(raw_value or "").strip()
    if not raw:
        return ""
    canonical = _as_dict(identity.get("canonical"))
    direct = _canonical_id(raw)
    if direct and direct in canonical:
        return direct
    return str(_as_dict(identity.get("alias_to_id")).get(_norm(raw)) or "")


def canonical_chapter_ids(identity: Dict[str, Any]) -> List[str]:
    return list(_as_dict(identity.get("canonical")).keys())


def canonical_chapter_payload(identity: Dict[str, Any], chapter_id: str) -> Dict[str, Any]:
    return dict(_as_dict(_as_dict(identity.get("canonical")).get(chapter_id)))
