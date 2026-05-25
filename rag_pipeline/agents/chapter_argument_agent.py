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
BAD_FACT_PATTERNS.extend(
    [
        r"\u4e0d\u662f\u5355\u70b9\u4e8b\u5b9e\u9898",
        r"\u4f9b\u7ed9\u7ea6\u675f",
        r"\u4ef7\u683c\u4fee\u590d",
        r"\u5e93\u5b58\u4e0b\u964d",
        r"\u8ba2\u5355\u786e\u8ba4",
        r"\u519c\u4e1a\u4eba\u5de5\u667a\u80fd",
        r"\u6570\u636e\u6295\u6bd2",
        r"Scribd",
        r"\u53d1\u73b0\u62a5\u544a",
        r"\u7eba\u7ec7",
        r"\u667a\u80fd\u624b\u673a",
        r"SEO",
        r"(?:\u6210\u672c|\u5173\u952e\u4e8b\u5b9e|\u653f\u7b56\u76d1\u7ba1|\u653f\u7b56\u76ee\u6807)\s*[:\uff1a]\s*-?\d{1,3}(?:\.\d+)?%?",
        r"\u53ef\u590d\u6838\u6765\u6e90\u8d8a\u72ec\u7acb",
        r"\u76ee\u524d\u7ed3\u8bba\u4ecd\u53d7",
        r"\u4f18\u5148\u590d\u6838\u53ef\u8ffd\u6eaf\u6765\u6e90",
        r"\u672c\u7ae0\u5173\u6ce8",
        r"\u672c\u8282\u56f4\u7ed5",
        r"\u4e0d\u5177\u6709\u636e\u4ee5\u53d1\u884c\u80a1\u7968\u7684\u6cd5\u5f8b\u6548\u529b",
        r"\u4ec5\u4f9b\u9884\u5148\u62ab\u9732",
        r"\u6295\u8d44\u8005\u5e94\u5f53\u4ee5\u6b63\u5f0f\u516c",
        r"\u539f\u6587\u94fe\u63a5",
        r"\u539f\u6587\u51fa\u5904",
        r"\u4e0b\u8f7d",
        r"\u9644\u4e0b\u8f7d",
        r"\u7b2c\s*\d+\s*\u8f6e",
        r"picture\s*\[\d+\s*x\s*\d+\]\s*intentionally\s*omitted",
        r"\u8d2d\u7269\u8f66|\u6211\u7684\u8ba2\u5355|\u514d\u8d39\u6ce8\u518c|\u62a5\u544a\u670d\u52a1\u70ed\u7ebf",
        r"URL[:\uff1a]",
    ]
)

TEMPLATE_SENTENCE_PATTERNS = [
    r"[^。\n]*\u53ef\u590d\u6838\u6765\u6e90\u8d8a\u72ec\u7acb[^。\n]*(?:。|$)",
    r"[^。\n]*\u76ee\u524d\u7ed3\u8bba\u4ecd\u53d7[^。\n]*(?:。|$)",
    r"[^。\n]*\u4f18\u5148\u590d\u6838\u53ef\u8ffd\u6eaf\u6765\u6e90[^。\n]*(?:。|$)",
    r"[^。\n]*\u672c\u7ae0\u5173\u6ce8[^。\n]*(?:。|$)",
    r"[^。\n]*\u672c\u8282\u56f4\u7ed5[^。\n]*(?:。|$)",
]

BAD_SECTION_TITLE_PATTERNS = [
    r"\u6536\u5165\u3001\u5229\u6da6\u4e0e\u73b0\u91d1\u6d41\u8d28\u91cf",
    r"\u5355\u4f4d\u7ecf\u6d4e\u6a21\u578b",
    r"\u6295\u8d44\u4f18\u5148\u7ea7\u77e9\u9635",
]

PUBLIC_EVIDENCE_COLLECTIONS = (
    "core_evidence",
    "supporting_evidence",
    "metric_evidence",
    "case_evidence",
    "counter_evidence",
    "directional_evidence",
    "sample_evidence",
    "table_evidence",
    "clue_evidence",
)


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


def _clean_public_text(value: Any, max_chars: int = 900) -> str:
    text = _compact(value, max_chars)
    for pattern in TEMPLATE_SENTENCE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.I)
    if any(re.search(pattern, text, flags=re.I) for pattern in BAD_FACT_PATTERNS):
        return ""
    return re.sub(r"\s{2,}", " ", text).strip()


def _clean_fact_anchor(value: Any, max_chars: int = 220) -> str:
    text = _clean_public_text(value, max_chars * 2)
    if not text:
        return ""
    text = re.sub(r"\[[Pp][Dd][Ff]\]\s*", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\bRead page[:：]?\s*", "", text, flags=re.I)
    text = re.sub(r"\bcontent description[:：]?\s*", "", text, flags=re.I)
    text = re.sub(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:[+-]\d{2}:\d{2})?", "", text)
    text = re.sub(r"\s*[；;]\s*", "；", text)
    parts: List[str] = []
    seen = set()
    for part in re.split(r"[；;。]\s*", text):
        part = _compact(part, max_chars)
        if not part or _is_bad_public_fact(part):
            continue
        key = re.sub(r"\W+", "", part.lower())[:80]
        if key in seen:
            continue
        seen.add(key)
        parts.append(part)
        if len(parts) >= 2:
            break
    return "；".join(parts)[:max_chars].rstrip("；,， ")


def _is_bad_section_title(value: Any) -> bool:
    text = str(value or "")
    return any(re.search(pattern, text, flags=re.I) for pattern in BAD_SECTION_TITLE_PATTERNS)


def _invalid_metric_item(item: Dict[str, Any]) -> bool:
    metric = str(item.get("metric") or item.get("indicator") or "").strip()
    value = str(item.get("value") or item.get("display_value") or item.get("numeric_value") or "").strip()
    fact = str(item.get("fact") or item.get("clean_fact") or item.get("content") or item.get("summary") or "").strip()
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


def _citation_ref_from_evidence(item: Dict[str, Any]) -> str:
    for key in ("source_ref", "citation_ref"):
        value = str(item.get(key) or "").strip()
        if re.fullmatch(r"\[\d{1,3}\]", value):
            return value
    source_id = str(item.get("source_id") or "").strip()
    if re.fullmatch(r"\d{1,3}", source_id):
        return f"[{source_id}]"
    ref = str(item.get("ref") or "").strip()
    if re.fullmatch(r"\[\d{1,3}\]", ref):
        return ref
    return ref or str(item.get("evidence_id") or "").strip()


def _collections_for_layout_section(layout_section: Dict[str, Any]) -> List[str]:
    block_type = str(layout_section.get("block_type") or layout_section.get("output_type") or "").strip()
    if block_type == "metric_reconciliation":
        return ["metric_evidence", "core_evidence", "supporting_evidence"]
    if block_type in {"competitive_positioning", "customer_painpoint_matrix", "unit_economics", "case_comparison"}:
        return ["case_evidence", "core_evidence", "supporting_evidence", "directional_evidence"]
    if block_type == "technology_maturity":
        return ["metric_evidence", "core_evidence", "supporting_evidence", "directional_evidence"]
    if block_type in {"risk_trigger", "verification_checklist", "scenario_analysis"}:
        return ["counter_evidence", "core_evidence", "supporting_evidence", "directional_evidence"]
    return list(PUBLIC_EVIDENCE_COLLECTIONS)


def _facts_from_collections(evidence_package: Dict[str, Any], collections: Sequence[str], *, limit: int = 4) -> List[str]:
    facts: List[str] = []
    seen = set()
    for collection in collections:
        for item in _as_list(evidence_package.get(collection)):
            if not isinstance(item, dict):
                continue
            if _invalid_metric_item(item):
                continue
            fact = _clean_fact_anchor(
                item.get("fact") or item.get("clean_fact") or item.get("content") or item.get("summary"),
                220,
            )
            if not fact:
                metric = _compact(item.get("metric") or item.get("indicator"), 80)
                value = _compact(item.get("value") or item.get("display_value"), 80)
                if metric and value:
                    fact = f"{metric}: {value}"
            if not fact or _is_bad_public_fact(fact):
                continue
            key = re.sub(r"\s+", "", fact.lower())[:140]
            if key in seen:
                continue
            seen.add(key)
            facts.append(fact)
            if len(facts) >= limit:
                return facts
    return facts


def _refs_from_collections(evidence_package: Dict[str, Any], collections: Sequence[str], *, limit: int = 6) -> List[str]:
    refs: List[str] = []
    for collection in collections:
        for item in _as_list(evidence_package.get(collection)):
            if not isinstance(item, dict):
                continue
            ref = _citation_ref_from_evidence(item)
            if ref:
                refs.append(ref)
            if len(refs) >= limit * 2:
                break
        if len(refs) >= limit * 2:
            break
    return _dedupe(refs, limit=limit)


def _facts_for_refs(
    evidence_package: Dict[str, Any],
    refs: Sequence[Any],
    collections: Sequence[str],
    *,
    limit: int = 3,
) -> List[str]:
    wanted = {str(ref or "").strip() for ref in refs if str(ref or "").strip()}
    if not wanted:
        return []
    facts: List[str] = []
    seen = set()
    for collection in collections:
        for item in _as_list(evidence_package.get(collection)):
            if not isinstance(item, dict) or _invalid_metric_item(item):
                continue
            item_refs = {
                str(item.get("ref") or "").strip(),
                str(item.get("evidence_id") or "").strip(),
                str(item.get("source_ref") or "").strip(),
                str(item.get("citation_ref") or "").strip(),
                _citation_ref_from_evidence(item),
                *[str(ref or "").strip() for ref in _as_list(item.get("source_refs"))],
            }
            item_refs = {ref for ref in item_refs if ref}
            if not wanted.intersection(item_refs):
                continue
            fact = _clean_fact_anchor(
                item.get("fact") or item.get("clean_fact") or item.get("content") or item.get("summary"),
                220,
            )
            if not fact:
                metric = _compact(item.get("metric") or item.get("indicator"), 80)
                value = _compact(item.get("value") or item.get("display_value"), 80)
                if metric and value:
                    fact = f"{metric}: {value}"
            if not fact or _is_bad_public_fact(fact):
                continue
            key = re.sub(r"\s+", "", fact.lower())[:140]
            if key in seen:
                continue
            seen.add(key)
            facts.append(fact)
            if len(facts) >= limit:
                return facts
    return facts


def _lead(chapter: Dict[str, Any], units: Sequence[Dict[str, Any]]) -> str:
    if units:
        claim = _clean_public_text(units[0].get("claim"), 240)
        if claim:
            return claim
    question = _compact(chapter.get("chapter_question"), 180)
    if question:
        return f"{question}需要结合可核验事实、机制约束和边界条件判断。"
    return ""


SECTION_TITLE_BY_BLOCK_TYPE = {
    "thesis": "核心观察",
    "argument": "事实依据",
    "metric_reconciliation": "指标口径与可比性",
    "risk_trigger": "边界条件",
    "verification_checklist": "后续观察变量",
    "case_argument": "案例事实",
    "customer_painpoint_matrix": "需求与付费证据",
    "competitive_positioning": "竞争变量",
    "technology_maturity": "技术变量与约束",
    "unit_economics": "商业化证据",
}


def _public_section_title(unit: Dict[str, Any], chapter: Dict[str, Any], *, index: int, layout_section: Optional[Dict[str, Any]] = None) -> str:
    layout_section = _as_dict(layout_section)
    raw = _clean_public_text(layout_section.get("section_title") or layout_section.get("title") or unit.get("section_title") or "", 120)
    chapter_title = _compact(chapter.get("chapter_title") or chapter.get("title") or "", 120)
    chapter_question = _compact(chapter.get("chapter_question") or chapter.get("chapter_role") or "", 120)
    block_type = str(layout_section.get("block_type") or layout_section.get("output_type") or unit.get("block_type") or unit.get("layout_section_role") or "").strip()
    fallback = SECTION_TITLE_BY_BLOCK_TYPE.get(block_type) or ("关键事实与判断依据" if index == 1 else "判断边界与后续验证")
    if _is_bad_section_title(fallback):
        fallback = "关键事实与判断依据" if index == 1 else "判断边界与后续验证"
    canonical_title = SECTION_TITLE_BY_BLOCK_TYPE.get(block_type)
    if block_type == "thesis" and canonical_title:
        return fallback
    if raw and canonical_title:
        raw_title_key = re.sub(r"\s+", "", raw).lower()
        canonical_title_key = re.sub(r"\s+", "", canonical_title).lower()
        known_title_keys = {
            re.sub(r"\s+", "", title).lower()
            for title in SECTION_TITLE_BY_BLOCK_TYPE.values()
            if title
        }
        if raw_title_key in known_title_keys and raw_title_key != canonical_title_key:
            return fallback
    if not raw or _is_bad_section_title(raw):
        return fallback
    raw_key = re.sub(r"\s+", "", raw)
    title_key = re.sub(r"\s+", "", chapter_title)
    question_key = re.sub(r"\s+", "", chapter_question)
    if raw_key and raw_key in {title_key, question_key}:
        return fallback
    if len(raw) > 42 and (title_key.startswith(raw_key[:16]) or raw_key.startswith(title_key[:16])):
        return fallback
    return raw


def _section_from_unit(
    unit: Dict[str, Any],
    chapter: Dict[str, Any],
    *,
    index: int,
    layout_section: Optional[Dict[str, Any]] = None,
    evidence_package: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    layout_section = _as_dict(layout_section)
    evidence_package = _as_dict(evidence_package)
    block_type = str(layout_section.get("block_type") or layout_section.get("output_type") or unit.get("block_type") or unit.get("layout_section_role") or "").strip()
    output_type = str(layout_section.get("output_type") or block_type or unit.get("output_type") or "").strip()
    section_role = str(layout_section.get("section_role") or unit.get("layout_section_role") or block_type or "").strip()
    collections = _collections_for_layout_section({"block_type": block_type, "output_type": output_type}) if block_type else list(PUBLIC_EVIDENCE_COLLECTIONS)
    evidence_refs = _as_list(unit.get("evidence_refs")) or _as_list(layout_section.get("required_evidence_refs"))
    if not evidence_refs and evidence_package:
        evidence_refs = _refs_from_collections(evidence_package, collections, limit=6)
    supporting_facts = [
        cleaned
        for item in _as_list(unit.get("supporting_facts"))
        for cleaned in [_clean_fact_anchor(item, 220)]
        if cleaned
    ][:3]
    if not supporting_facts and evidence_package:
        supporting_facts = _facts_for_refs(evidence_package, evidence_refs, collections, limit=3)
    if not supporting_facts and evidence_package and evidence_refs:
        supporting_facts = _facts_for_refs(evidence_package, evidence_refs, list(PUBLIC_EVIDENCE_COLLECTIONS), limit=3)
    if not supporting_facts and evidence_package and block_type:
        supporting_facts = _facts_from_collections(evidence_package, collections, limit=3)
    claim = _clean_public_text(unit.get("claim") or "", 420)
    if not claim and supporting_facts:
        claim = supporting_facts[0]
    reasoning = _clean_public_text(unit.get("reasoning") or "", 720)
    mechanism = _clean_public_text(unit.get("mechanism") or unit.get("reasoning") or "", 720)
    counter_evidence = _clean_public_text(unit.get("counter_evidence") or "", 520)
    actionable = _clean_public_text(unit.get("actionable") or "", 420)
    decision_implication = _clean_public_text(unit.get("decision_implication") or unit.get("actionable") or "", 520)
    render_blocks = _as_list(unit.get("render_blocks"))
    if not render_blocks:
        render_blocks = [
            {"type": "paragraph", "label": "关键判断", "text": claim},
            {"type": "paragraph", "label": "事实锚点", "text": "；".join(supporting_facts)},
            {"type": "paragraph", "label": "证据依据", "text": reasoning},
            {"type": "paragraph", "label": "边界", "text": counter_evidence},
            {"type": "paragraph", "label": "含义", "text": decision_implication},
            {"type": "evidence_list", "label": "关键证据", "evidence_refs": evidence_refs},
        ]
    cleaned_blocks: List[Dict[str, Any]] = []
    for block in render_blocks:
        if not isinstance(block, dict):
            continue
        cleaned_block = dict(block)
        if "text" in cleaned_block:
            label = str(cleaned_block.get("label") or "")
            if "事实锚点" in label:
                fact_parts = [
                    cleaned
                    for part in re.split(r"[；;。]\s*", str(cleaned_block.get("text") or ""))
                    for cleaned in [_clean_fact_anchor(part, 220)]
                    if cleaned
                ][:3]
                cleaned_block["text"] = "；".join(_dedupe(fact_parts, limit=3))
            else:
                cleaned_block["text"] = _clean_public_text(cleaned_block.get("text"), 900)
            if cleaned_block.get("type") == "paragraph" and not cleaned_block["text"]:
                continue
        cleaned_blocks.append(cleaned_block)
    evidence_backed = bool(evidence_refs and supporting_facts)
    return {
        "section_id": layout_section.get("section_id") or unit.get("section_id"),
        "section_title": _public_section_title(unit, chapter, index=index, layout_section=layout_section),
        "block_type": block_type,
        "output_type": output_type,
        "section_role": section_role,
        "required_evidence_refs": _as_list(layout_section.get("required_evidence_refs")),
        "claim": claim,
        "reasoning": reasoning,
        "mechanism": mechanism,
        "counter_evidence": counter_evidence,
        "actionable": actionable,
        "decision_implication": decision_implication,
        "what_to_verify_next": _as_list(unit.get("what_to_verify_next")),
        "supporting_facts": supporting_facts,
        "confidence": unit.get("confidence") or "medium",
        "evidence_refs": evidence_refs,
        "render_blocks": cleaned_blocks,
        "public_render": True,
        "layout_generated": bool(unit.get("layout_generated")),
        "evidence_backed": evidence_backed,
        "observation_only": not evidence_backed,
        "layout_match_score": unit.get("layout_match_score"),
        "layout_match_reason": unit.get("layout_match_reason"),
    }


def _section_from_layout(
    layout_section: Dict[str, Any],
    chapter: Dict[str, Any],
    *,
    index: int,
    evidence_package: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    layout_section = _as_dict(layout_section)
    required_refs = _as_list(layout_section.get("required_evidence_refs"))
    collections = _collections_for_layout_section(layout_section)
    derived_refs = _dedupe([*required_refs, *_refs_from_collections(evidence_package, collections, limit=6)], limit=6)
    facts = _facts_from_collections(evidence_package, collections, limit=3) or _chapter_fact_digest(evidence_package)[:2]
    block_type = str(layout_section.get("block_type") or layout_section.get("output_type") or "").strip()
    text = _clean_public_text(
        "；".join(facts)
        or _as_dict(_as_list(layout_section.get("render_blocks"))[0] if _as_list(layout_section.get("render_blocks")) else {}).get("text")
        or "",
        420,
    )
    if not derived_refs and not text:
        return None
    section_title = _public_section_title({}, chapter, index=index, layout_section=layout_section)
    claim = text or f"{section_title}需要等待更多可核验证据后再形成明确判断。"
    evidence_backed = bool(derived_refs and facts)
    return {
        "section_id": layout_section.get("section_id"),
        "section_title": section_title,
        "block_type": block_type,
        "output_type": layout_section.get("output_type") or block_type,
        "section_role": layout_section.get("section_role") or block_type,
        "required_evidence_refs": required_refs,
        "claim": claim,
        "reasoning": text,
        "mechanism": text,
        "counter_evidence": "",
        "actionable": "",
        "decision_implication": "",
        "what_to_verify_next": [],
        "supporting_facts": facts,
        "confidence": "low",
        "evidence_refs": derived_refs,
        "render_blocks": [
            {"type": "paragraph", "label": "观察判断", "text": claim},
            {"type": "paragraph", "label": "事实锚点", "text": "；".join(facts)},
        ],
        "public_render": True,
        "layout_generated": True,
        "evidence_backed": evidence_backed,
        "observation_only": not evidence_backed,
        "layout_match_score": 0,
        "layout_match_reason": "layout_fallback",
    }


def _layout_by_chapter(micro_layouts: Optional[Sequence[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    return {
        str(layout.get("chapter_id") or "").strip(): dict(layout)
        for layout in list(micro_layouts or [])
        if isinstance(layout, dict) and str(layout.get("chapter_id") or "").strip()
    }


def _layout_sections(layout: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [dict(section) for section in _as_list(_as_dict(layout).get("sections")) if isinstance(section, dict)]


def _unit_layout_match_score(unit: Dict[str, Any], layout_section: Dict[str, Any]) -> int:
    section_id = str(layout_section.get("section_id") or "").strip()
    block_type = str(layout_section.get("block_type") or layout_section.get("output_type") or "").strip()
    title = re.sub(r"\s+", "", str(layout_section.get("section_title") or layout_section.get("title") or "").strip())
    unit_section_id = str(unit.get("section_id") or "").strip()
    unit_block = str(unit.get("block_type") or unit.get("output_type") or unit.get("layout_section_role") or "").strip()
    unit_title = re.sub(r"\s+", "", str(unit.get("section_title") or unit.get("question") or "").strip())
    if section_id and unit_section_id == section_id:
        return 100
    if block_type and unit_block == block_type:
        return 80
    if title and unit_title and (title == unit_title or title in unit_title or unit_title in title):
        return 50
    required_refs = {str(ref or "").strip() for ref in _as_list(layout_section.get("required_evidence_refs")) if str(ref or "").strip()}
    unit_refs = {str(ref or "").strip() for ref in _as_list(unit.get("evidence_refs")) if str(ref or "").strip()}
    if required_refs and unit_refs and required_refs.intersection(unit_refs):
        return 30
    return 0


def _pop_unit_for_layout(available_units: List[Dict[str, Any]], layout_section: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    best_index = -1
    best_score = 0
    for index, unit in enumerate(available_units):
        score = _unit_layout_match_score(unit, layout_section)
        if score > best_score:
            best_score = score
            best_index = index
    if best_index >= 0 and best_score > 0:
        return available_units.pop(best_index)
    return None


def _section_duplicate_key(section: Dict[str, Any]) -> str:
    refs = ",".join(sorted(str(ref or "").strip() for ref in _as_list(section.get("evidence_refs")) if str(ref or "").strip()))
    claim = re.sub(r"\s+", "", str(section.get("claim") or "").strip().lower())
    return f"{claim[:180]}|{refs}"


def _chapter_fact_digest(evidence_package: Dict[str, Any]) -> List[str]:
    limit = _env_int("REPORT_CHAPTER_FACT_DIGEST_LIMIT", 18, min_value=0, max_value=80)
    if limit <= 0:
        return []
    facts: List[str] = []
    for collection in PUBLIC_EVIDENCE_COLLECTIONS:
        for item in _as_list(evidence_package.get(collection)):
            if not isinstance(item, dict):
                continue
            if _invalid_metric_item(item):
                continue
            fact = _clean_fact_anchor(
                item.get("fact") or item.get("clean_fact") or item.get("content") or item.get("summary"),
                220,
            )
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
    del llm_client
    report_blueprint = _as_dict(report_blueprint)
    layout_by_chapter = _layout_by_chapter(micro_layouts)
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
        layout_sections = _layout_sections(_as_dict(layout_by_chapter.get(chapter_id)))
        available_units = [dict(unit) for unit in units]
        dropped_sections: List[Dict[str, Any]] = []
        sections: List[Dict[str, Any]] = []
        seen_section_keys = set()
        if layout_sections:
            for section_index, layout_section in enumerate(layout_sections, start=1):
                unit = _pop_unit_for_layout(available_units, layout_section)
                if unit:
                    section = _section_from_unit(
                        unit,
                        chapter,
                        index=section_index,
                        layout_section=layout_section,
                        evidence_package=evidence_package,
                    )
                else:
                    section = _section_from_layout(
                        layout_section,
                        chapter,
                        index=section_index,
                        evidence_package=evidence_package,
                    )
                if not section or not (section.get("claim") or section.get("reasoning") or section.get("supporting_facts")):
                    dropped_sections.append(
                        {
                            "section_id": layout_section.get("section_id"),
                            "block_type": layout_section.get("block_type") or layout_section.get("output_type"),
                            "reason": "layout_section_without_public_evidence",
                            "source": "micro_layout",
                        }
                    )
                    continue
                duplicate_key = _section_duplicate_key(section)
                if duplicate_key and duplicate_key in seen_section_keys:
                    dropped_sections.append(
                        {
                            "section_id": section.get("section_id"),
                            "block_type": section.get("block_type"),
                            "reason": "duplicate_claim_and_refs",
                            "source": "chapter_argument",
                        }
                    )
                    continue
                seen_section_keys.add(duplicate_key)
                sections.append(section)
        for section_index, unit in enumerate(available_units, start=len(sections) + 1):
            section = _section_from_unit(unit, chapter, index=section_index, evidence_package=evidence_package)
            if not (section.get("claim") or section.get("reasoning") or section.get("supporting_facts")):
                continue
            duplicate_key = _section_duplicate_key(section)
            if duplicate_key and duplicate_key in seen_section_keys:
                dropped_sections.append(
                    {
                        "section_id": section.get("section_id"),
                        "block_type": section.get("block_type"),
                        "reason": "duplicate_claim_and_refs",
                        "source": "chapter_argument",
                    }
                )
                continue
            seen_section_keys.add(duplicate_key)
            sections.append(section)

        omitted_observation_sections: List[Dict[str, Any]] = []
        evidence_sections = [
            section
            for section in sections
            if not (section.get("observation_only") and not section.get("evidence_backed"))
        ]
        observation_sections = [
            section
            for section in sections
            if section.get("observation_only") and not section.get("evidence_backed")
        ]
        if evidence_sections:
            omitted_observation_sections = observation_sections
            sections = evidence_sections
        elif observation_sections:
            sections = [observation_sections[0]]
            sections[0]["force_render_observation"] = True
            omitted_observation_sections = observation_sections[1:]

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

        evidence_backed_section_count = len([section for section in sections if section.get("evidence_backed")])
        observation_section_count = len([section for section in sections if section.get("observation_only")])
        lead_text = "" if sections and not evidence_backed_section_count else _lead(chapter, units)
        key_takeaway = _clean_public_text(sections[0].get("claim") if sections else "", 240)
        next_actions = _dedupe([_clean_public_text(section.get("actionable"), 220) for section in sections], limit=5)
        mechanisms = _dedupe([_clean_public_text(section.get("mechanism"), 320) for section in sections], limit=3)
        counter_evidence = _dedupe([_clean_public_text(section.get("counter_evidence"), 260) for section in sections], limit=3)
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
                "lead": lead_text,
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
                "layout_sections": layout_sections,
                "effective_section_count": evidence_backed_section_count,
                "observation_section_count": observation_section_count,
                "omitted_observation_sections": [
                    {
                        "section_id": section.get("section_id"),
                        "block_type": section.get("block_type"),
                        "section_title": section.get("section_title"),
                        "reason": "observation_only_without_evidence",
                    }
                    for section in omitted_observation_sections
                ],
                "dropped_sections": dropped_sections + [
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
