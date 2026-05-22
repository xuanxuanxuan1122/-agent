from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence


AGENT_NAME = "chapter_argument_agent"
AGENT_DESCRIPTION = "Chapter Argument Agent. Builds structured chapter packages from public argument units and tables."
BAD_FACT_PATTERNS = [
    r"联网分析\s*Agent\s*失败",
    r"Retrieval\.TestUserQueryExceeded",
    r"query\s+exceed(?:ed)?\s+the\s+limit",
    r"目前更像局部信号",
]


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _compact(value: Any, max_chars: int = 260) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "..."


def _dedupe(values: Iterable[Any], *, limit: int = 8) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        text = _compact(value, 120)
        if not text:
            continue
        key = re.sub(r"\s+", "", text.lower())
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def _is_bad_public_fact(value: Any) -> bool:
    text = str(value or "")
    if not text:
        return True
    return any(re.search(pattern, text, flags=re.I) for pattern in BAD_FACT_PATTERNS)


TOKEN_PROFILE_INT_DEFAULTS = {
    "lean": {"REPORT_CHAPTER_FACT_DIGEST_LIMIT": 4},
    "balanced": {"REPORT_CHAPTER_FACT_DIGEST_LIMIT": 8},
    "deep": {"REPORT_CHAPTER_FACT_DIGEST_LIMIT": 18},
}


def _profile_default(name: str, default: int) -> int:
    if os.getenv(name) is not None:
        return default
    profile = str(os.getenv("REPORT_TOKEN_PROFILE", "balanced") or "balanced").strip().lower()
    return TOKEN_PROFILE_INT_DEFAULTS.get(profile, TOKEN_PROFILE_INT_DEFAULTS["balanced"]).get(name, default)


def _env_int(name: str, default: int, *, min_value: int = 0, max_value: int = 10_000) -> int:
    default = _profile_default(name, default)
    try:
        value = int(os.getenv(name, str(default)) or default)
    except (TypeError, ValueError):
        value = default
    return max(min_value, min(max_value, value))


def _by_chapter(items: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    result: Dict[str, List[Dict[str, Any]]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        chapter_id = str(item.get("chapter_id") or "")
        result.setdefault(chapter_id, []).append(item)
    return result


def public_argument_units(units: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        unit
        for unit in list(units or [])
        if isinstance(unit, dict)
        and unit.get("public_render") is True
        and not unit.get("omit_from_report")
    ]


def _public_tables(tables: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        table
        for table in list(tables or [])
        if isinstance(table, dict)
        and table.get("should_render")
        and not table.get("appendix_only")
    ]


def _lead(chapter: Dict[str, Any], units: Sequence[Dict[str, Any]]) -> str:
    if units:
        claim = _compact(units[0].get("claim"), 240)
        if claim:
            return claim
    question = _compact(chapter.get("chapter_question"), 180)
    if question:
        return f"围绕“{question}”，讨论从现实信号进入变量关系，再回到结论成立的条件。"
    return ""


SECTION_TITLE_BY_BLOCK_TYPE = {
    "thesis": "核心判断与证据边界",
    "argument": "关键事实与判断依据",
    "metric_reconciliation": "指标口径与可比性",
    "risk_trigger": "反向信号与判断边界",
    "verification_checklist": "后续验证指标",
    "case_argument": "案例验证与适用范围",
    "customer_painpoint_matrix": "需求场景与付费约束",
    "competitive_positioning": "竞争格局与差异化变量",
    "technology_maturity": "技术成熟度与产业化约束",
    "unit_economics": "商业化质量与经济性",
}


def _public_section_title(unit: Dict[str, Any], chapter: Dict[str, Any], *, index: int) -> str:
    raw = _compact(unit.get("section_title") or "", 120)
    chapter_title = _compact(chapter.get("chapter_title") or chapter.get("title") or "", 120)
    chapter_question = _compact(chapter.get("chapter_question") or chapter.get("chapter_role") or "", 120)
    block_type = str(unit.get("block_type") or unit.get("layout_section_role") or "").strip()
    fallback = SECTION_TITLE_BY_BLOCK_TYPE.get(block_type) or ("关键事实与判断依据" if index == 1 else "判断边界与后续验证")
    if not raw:
        return fallback
    raw_key = re.sub(r"\s+", "", raw)
    title_key = re.sub(r"\s+", "", chapter_title)
    question_key = re.sub(r"\s+", "", chapter_question)
    if raw_key and raw_key in {title_key, question_key}:
        return fallback
    if len(raw) > 42 and (title_key.startswith(raw_key[:16]) or raw_key.startswith(title_key[:16])):
        return fallback
    return raw


def _section_from_unit(unit: Dict[str, Any], chapter: Dict[str, Any], *, index: int) -> Dict[str, Any]:
    evidence_refs = _as_list(unit.get("evidence_refs"))
    supporting_facts = [
        _compact(item, 260)
        for item in _as_list(unit.get("supporting_facts"))
        if str(item or "").strip()
    ][:3]
    render_blocks = _as_list(unit.get("render_blocks"))
    if not render_blocks:
        render_blocks = [
            {"type": "paragraph", "label": "关键判断", "text": unit.get("claim") or ""},
            {"type": "paragraph", "label": "事实锚点", "text": "；".join(supporting_facts)},
            {"type": "paragraph", "label": "证据依据", "text": unit.get("reasoning") or ""},
            {"type": "paragraph", "label": "边界", "text": unit.get("counter_evidence") or ""},
            {"type": "paragraph", "label": "含义", "text": unit.get("decision_implication") or unit.get("actionable") or ""},
            {"type": "evidence_list", "label": "关键证据", "evidence_refs": evidence_refs},
        ]
    return {
        "section_id": unit.get("section_id"),
        "section_title": _public_section_title(unit, chapter, index=index),
        "block_type": unit.get("block_type") or unit.get("layout_section_role") or "",
        "claim": unit.get("claim") or "",
        "reasoning": unit.get("reasoning") or "",
        "mechanism": unit.get("mechanism") or unit.get("reasoning") or "",
        "counter_evidence": unit.get("counter_evidence") or "",
        "actionable": unit.get("actionable") or "",
        "decision_implication": unit.get("decision_implication") or unit.get("actionable") or "",
        "what_to_verify_next": _as_list(unit.get("what_to_verify_next")),
        "supporting_facts": supporting_facts,
        "confidence": unit.get("confidence") or "medium",
        "evidence_refs": evidence_refs,
        "render_blocks": render_blocks,
        "public_render": True,
    }


def _chapter_fact_digest(evidence_package: Dict[str, Any]) -> List[str]:
    limit = _env_int("REPORT_CHAPTER_FACT_DIGEST_LIMIT", 18, min_value=0, max_value=80)
    if limit <= 0:
        return []
    facts: List[str] = []
    for collection in ("core_evidence", "supporting_evidence", "sample_evidence", "table_evidence", "clue_evidence"):
        for item in _as_list(evidence_package.get(collection)):
            if not isinstance(item, dict):
                continue
            if item.get("metric_validation_status") == "invalid":
                continue
            fact = _compact(item.get("fact") or item.get("clean_fact") or item.get("content") or item.get("summary"), 320)
            if _is_bad_public_fact(fact):
                continue
            metric = _compact(item.get("metric") or item.get("indicator"), 80)
            value = _compact(item.get("value") or item.get("display_value"), 80)
            if fact:
                facts.append(fact)
            elif metric and value:
                facts.append(f"{metric}: {value}")
    return _dedupe(facts, limit=limit)


def _omitted_chapter_package(
    chapter: Dict[str, Any],
    *,
    chapter_id: str,
    index: int,
    evidence_package: Dict[str, Any],
    raw_units: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "agent": AGENT_NAME,
        "chapter_id": chapter_id,
        "chapter_title": chapter.get("chapter_title") or chapter.get("title") or f"章节 {index}",
        "chapter_question": chapter.get("chapter_question") or chapter.get("chapter_role") or "",
        "lead": "",
        "sections": [],
        "table_packages": [],
        "chapter_summary": {"key_takeaway": "", "confidence": "insufficient", "next_actions": []},
                "evidence_gaps": _as_list(evidence_package.get("missing_evidence")),
                "evidence_quality_summary": _as_dict(evidence_package.get("evidence_quality_summary")),
                "missing_proof_standards": _as_list(evidence_package.get("missing_proof_standards")),
                "omit_from_report": True,
        "internal_reason": "no_public_argument_or_table",
        "dropped_sections": [
            {
                "section_id": unit.get("section_id"),
                "reason": unit.get("internal_reason") or unit.get("quality_status") or "not_public",
            }
            for unit in raw_units
            if isinstance(unit, dict)
        ],
    }


def run_chapter_argument_agent(
    *,
    report_blueprint: Optional[Dict[str, Any]] = None,
    micro_layouts: Optional[Sequence[Dict[str, Any]]] = None,
    argument_units: Optional[Sequence[Dict[str, Any]]] = None,
    table_packages: Optional[Sequence[Dict[str, Any]]] = None,
    chapter_evidence_packages: Optional[Sequence[Dict[str, Any]]] = None,
    llm_client: Any = None,
) -> List[Dict[str, Any]]:
    del llm_client, micro_layouts
    report_blueprint = _as_dict(report_blueprint)
    all_units = [item for item in list(argument_units or []) if isinstance(item, dict)]
    all_units_by_chapter = _by_chapter(all_units)
    units_by_chapter = _by_chapter(public_argument_units(all_units))
    tables_by_chapter = _by_chapter([item for item in list(table_packages or []) if isinstance(item, dict)])
    evidence_by_chapter = {
        str(package.get("chapter_id") or ""): package
        for package in list(chapter_evidence_packages or [])
        if isinstance(package, dict)
    }

    packages: List[Dict[str, Any]] = []
    for index, chapter in enumerate(_as_list(report_blueprint.get("chapters")), start=1):
        if not isinstance(chapter, dict):
            continue
        chapter_id = str(chapter.get("chapter_id") or f"chapter_{index}")
        units = units_by_chapter.get(chapter_id, [])
        raw_units = all_units_by_chapter.get(chapter_id, [])
        evidence_package = _as_dict(evidence_by_chapter.get(chapter_id))
        public_tables = _public_tables(tables_by_chapter.get(chapter_id, []))
        sections = [_section_from_unit(unit, chapter, index=section_index) for section_index, unit in enumerate(units, start=1)]

        if not sections and not public_tables:
            packages.append(
                _omitted_chapter_package(
                    chapter,
                    chapter_id=chapter_id,
                    index=index,
                    evidence_package=evidence_package,
                    raw_units=raw_units,
                )
            )
            continue

        key_takeaway = _compact(sections[0].get("claim") if sections else "", 240)
        next_actions = _dedupe([section.get("actionable") for section in sections], limit=5)
        mechanisms = _dedupe([section.get("mechanism") for section in sections], limit=3)
        counter_evidence = _dedupe([section.get("counter_evidence") for section in sections], limit=3)
        what_to_verify_next = _dedupe(
            [
                item
                for section in sections
                for item in _as_list(section.get("what_to_verify_next"))
            ],
            limit=5,
        )
        packages.append(
            {
                "agent": AGENT_NAME,
                "chapter_id": chapter_id,
                "chapter_title": chapter.get("chapter_title") or chapter.get("title") or f"章节 {index}",
                "chapter_question": chapter.get("chapter_question") or chapter.get("chapter_role") or "",
                "lead": _lead(chapter, units),
                "sections": sections,
                "table_packages": public_tables,
                "chapter_summary": {
                    "key_takeaway": key_takeaway,
                    "confidence": sections[0].get("confidence") if sections else "medium",
                    "mechanisms": mechanisms,
                    "counter_evidence": counter_evidence,
                    "next_actions": next_actions,
                    "what_to_verify_next": what_to_verify_next,
                },
                "chapter_fact_digest": _chapter_fact_digest(evidence_package),
                "evidence_gaps": _as_list(evidence_package.get("missing_evidence")),
                "evidence_quality_summary": _as_dict(evidence_package.get("evidence_quality_summary")),
                "missing_proof_standards": _as_list(evidence_package.get("missing_proof_standards")),
                "omit_from_report": False,
                "dropped_sections": [
                    {
                        "section_id": unit.get("section_id"),
                        "reason": unit.get("internal_reason") or unit.get("quality_status") or "not_public",
                    }
                    for unit in raw_units
                    if isinstance(unit, dict) and unit not in units
                ],
            }
        )
    return packages
