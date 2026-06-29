from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterable, List, Sequence


INTERNAL_PUBLIC_TEXT_RE = re.compile(
    r"(?:claim_unit|score_gap|diagnostic(?:_only)?|source_check|QA|Clean|fatal|repair|EV-\d|事实说明|证据链|补证建议)",
    re.I,
)


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _env_int(name: str, default: int, *, min_value: int = 0, max_value: int = 100) -> int:
    raw = os.getenv(name)
    try:
        value = int(str(raw).strip()) if raw not in (None, "") else default
    except (TypeError, ValueError):
        value = default
    return max(min_value, min(max_value, value))


def _dedupe(values: Iterable[Any], *, limit: int = 12) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        text = _text(value)
        if not text:
            continue
        key = re.sub(r"\W+", "", text.lower())[:160]
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def _fact_text(item: Any) -> str:
    if isinstance(item, dict):
        card = _as_dict(item.get("public_fact_card")) or _as_dict(_as_dict(item.get("public_fact_quality")).get("public_fact_card"))
        return _text(
            card.get("distilled_fact")
            or item.get("distilled_fact")
            or item.get("public_fact")
            or item.get("fact")
            or item.get("summary")
            or item.get("content")
            or item.get("text")
        )
    return _text(item)


def _fact_refs(item: Any) -> List[str]:
    if not isinstance(item, dict):
        return []
    return _dedupe(
        [
            item.get("evidence_id"),
            item.get("fact_id"),
            item.get("id"),
            item.get("source_ref"),
            item.get("ref"),
            *_as_list(item.get("evidence_refs")),
            *_as_list(item.get("used_fact_refs")),
        ],
        limit=8,
    )


def _citation_refs(item: Any) -> List[str]:
    if not isinstance(item, dict):
        return []
    refs = [
        item.get("citation_ref"),
        item.get("source_ref"),
        item.get("ref"),
        *_as_list(item.get("citation_refs")),
    ]
    return _dedupe([ref for ref in refs if re.match(r"^\[\d+\]$", _text(ref))], limit=8)


def _section_citation_refs(section: Dict[str, Any]) -> List[str]:
    direct = _citation_refs(section)
    if direct:
        return direct
    plan = _as_dict(section.get("section_plan"))
    return _dedupe([ref for ref in _as_list(plan.get("used_fact_refs")) if re.match(r"^\[\d+\]$", _text(ref))], limit=8)


def _usable_public_fact(text: str) -> bool:
    if not text or len(text) < 12:
        return False
    if INTERNAL_PUBLIC_TEXT_RE.search(text):
        return False
    lowered = text.lower()
    if any(token in lowered for token in ("login", "captcha", "403 forbidden", "404 not found", "http error")):
        return False
    return True


def _collect_chapter_digest_items(chapter: Dict[str, Any], *, limit: int = 18) -> List[Dict[str, Any]]:
    collections = [
        *_as_list(chapter.get("chapter_fact_digest")),
        *_as_list(chapter.get("core_evidence")),
        *_as_list(chapter.get("supporting_evidence")),
        *_as_list(chapter.get("sample_evidence")),
    ]
    for section in _as_list(chapter.get("sections")):
        if not isinstance(section, dict) or section.get("omit_from_report"):
            continue
        section_refs = _as_list(section.get("used_fact_refs")) or _as_list(section.get("evidence_refs"))
        section_citations = _section_citation_refs(section)
        for fact in _as_list(section.get("supporting_facts")):
            collections.append(
                {
                    "distilled_fact": fact,
                    "evidence_refs": section_refs,
                    "citation_refs": section_citations,
                }
            )

    items: List[Dict[str, Any]] = []
    seen = set()
    for raw in collections:
        text = _fact_text(raw)
        if not _usable_public_fact(text):
            continue
        key = re.sub(r"\W+", "", text.lower())[:180]
        if key in seen:
            continue
        seen.add(key)
        refs = _fact_refs(raw)
        citations = _citation_refs(raw)
        if not refs and not citations:
            continue
        items.append(
            {
                "text": text,
                "used_fact_refs": refs or citations,
                "citation_refs": citations,
                "fact_type": _text(_as_dict(raw).get("fact_type") or _as_dict(raw).get("proof_role") or "signal"),
            }
        )
        if len(items) >= limit:
            break
    return items


def _digest_title(chapter_title: str, index: int) -> str:
    if index == 1:
        return f"{chapter_title}的公开信号"
    if index == 2:
        return f"{chapter_title}的场景含义"
    return f"{chapter_title}的边界观察"


def _digest_paragraph(*, chapter_title: str, facts: Sequence[str]) -> str:
    first = facts[0] if facts else ""
    rest = "；".join(facts[1:3])
    if rest:
        return (
            f"围绕{chapter_title}，公开材料已经提供了若干可以进入正文的观察信号：{first}。"
            f"同时，{rest}。这些材料可以把章节叙事落到主体行动、场景变化和产业链传导上："
            "先看哪些主体开始行动，再看这些行动如何影响相关人群、组织安排和后续决策。"
        )
    return (
        f"围绕{chapter_title}，公开材料已经提供了可以进入正文的观察信号：{first}。"
        "它说明章节判断不能只停留在事实罗列，而要写出相关主体为什么行动、哪些环节受到影响，以及这些变化会怎样改变后续判断。"
    )


def build_public_evidence_digest_sections(chapter: Dict[str, Any]) -> List[Dict[str, Any]]:
    chapter = _as_dict(chapter)
    if chapter.get("omit_from_report"):
        return []
    max_sections = _env_int("REPORT_PUBLIC_EVIDENCE_DIGEST_MAX_SECTIONS_PER_CHAPTER", 2, min_value=0, max_value=6)
    if max_sections <= 0:
        return []
    items = _collect_chapter_digest_items(
        chapter,
        limit=_env_int("REPORT_PUBLIC_EVIDENCE_DIGEST_MAX_FACTS_PER_CHAPTER", 12, min_value=3, max_value=40),
    )
    if len(items) < _env_int("REPORT_PUBLIC_EVIDENCE_DIGEST_MIN_FACTS", 2, min_value=1, max_value=10):
        return []

    chapter_id = _text(chapter.get("chapter_id")) or "chapter"
    chapter_title = _text(chapter.get("chapter_title")) or "本章"
    sections: List[Dict[str, Any]] = []
    chunk_size = _env_int("REPORT_PUBLIC_EVIDENCE_DIGEST_FACTS_PER_SECTION", 3, min_value=2, max_value=6)
    for index in range(max_sections):
        chunk = items[index * chunk_size : (index + 1) * chunk_size]
        if not chunk:
            break
        refs = _dedupe([ref for item in chunk for ref in _as_list(item.get("used_fact_refs"))], limit=10)
        citations = _dedupe([ref for item in chunk for ref in _as_list(item.get("citation_refs"))], limit=10)
        if not refs:
            continue
        facts = [_text(item.get("text")) for item in chunk if _text(item.get("text"))]
        paragraph = _digest_paragraph(chapter_title=chapter_title, facts=facts)
        if INTERNAL_PUBLIC_TEXT_RE.search(paragraph):
            continue
        section_id = f"{chapter_id}_public_digest_{index + 1:02d}"
        title = _digest_title(chapter_title, index + 1)
        sections.append(
            {
                "section_id": section_id,
                "section_title": title,
                "claim": paragraph,
                "reasoning": paragraph,
                "mechanism": paragraph,
                "counter_evidence": "公开披露通常更容易呈现已经启动的事项，实际进展仍会受到资金、审批、执行成本和使用门槛的共同影响。",
                "actionable": "判断重点转向主体行动是否持续、场景是否扩大、影响路径是否更清晰。",
                "supporting_facts": facts,
                "evidence_refs": refs,
                "used_fact_refs": refs,
                "citation_refs": citations,
                "render_blocks": [{"type": "paragraph", "label": "", "text": paragraph}],
                "public_render": True,
                "public_text_allowed": True,
                "diagnostic_only": False,
                "must_not_render": False,
                "evidence_backed": True,
                "claim_strength": "directional",
                "block_type": "public_evidence_digest",
                "writing_mode": "directional_observation",
                "public_digest_generated": True,
            }
        )
    return sections
