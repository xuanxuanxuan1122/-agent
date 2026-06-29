from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence


ROLE_ORDER = {
    "setup": 0,
    "evidence_progression": 1,
    "mechanism": 2,
    "business_implication": 3,
    "constraint": 4,
}

STRENGTH_RANK = {
    "strong": 4,
    "moderate": 3,
    "directional": 2,
    "limited_evidence": 2,
    "weak": 1,
    "contextual": 1,
}

RISK_RE = re.compile(
    r"risk|constraint|counter|failure|delay|uncertain|limitation|compliance|security|"
    r"\u98ce\u9669|\u7ea6\u675f|\u4e0d\u786e\u5b9a|\u5ef6\u671f|\u5408\u89c4|\u5b89\u5168|\u9650\u5236",
    re.I,
)
CASE_RE = re.compile(
    r"case|pilot|customer|deployment|adoption|project|procurement|order|"
    r"\u6848\u4f8b|\u8bd5\u70b9|\u5ba2\u6237|\u843d\u5730|\u9879\u76ee|\u91c7\u8d2d|\u8ba2\u5355|\u573a\u666f",
    re.I,
)
MECHANISM_RE = re.compile(
    r"mechanism|technology|workflow|product|capability|supply|chain|"
    r"\u673a\u5236|\u6280\u672f|\u6d41\u7a0b|\u4ea7\u54c1|\u80fd\u529b|\u4f9b\u7ed9|\u4ea7\u4e1a\u94fe",
    re.I,
)
BUSINESS_RE = re.compile(
    r"commercial|market|revenue|profit|business|monetization|demand|competition|"
    r"\u5546\u4e1a|\u5e02\u573a|\u6536\u5165|\u5229\u6da6|\u9700\u6c42|\u7ade\u4e89|\u683c\u5c40",
    re.I,
)


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return []


def _text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _dedupe(values: Iterable[Any], *, limit: int = 128) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        text = _text(value)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def _claim_id(unit: Mapping[str, Any], index: int) -> str:
    return _text(unit.get("claim_id") or unit.get("id")) or f"claim_{index}"


def _claim_text(unit: Mapping[str, Any]) -> str:
    return _text(unit.get("claim") or unit.get("judgment") or unit.get("conclusion"))


def _refs(unit: Mapping[str, Any], *keys: str) -> List[str]:
    values: List[Any] = []
    for key in keys:
        values.extend(_as_list(unit.get(key)))
    return _dedupe(values)


def _strength(unit: Mapping[str, Any]) -> str:
    return _text(unit.get("claim_strength") or unit.get("claim_status")).lower() or "directional"


def _role_for_claim(unit: Mapping[str, Any], *, index_in_chapter: int) -> str:
    haystack = " ".join(
        _text(unit.get(key))
        for key in (
            "cluster_key",
            "chapter_id",
            "recommended_chapter",
            "analysis_role",
            "proof_role",
            "section_role",
            "block_type",
            "claim",
        )
    )
    strength = _strength(unit)
    if RISK_RE.search(haystack) or strength in {"weak", "contextual"}:
        return "constraint"
    if index_in_chapter == 0 and strength in {"strong", "moderate", "directional"}:
        return "setup"
    if CASE_RE.search(haystack):
        return "evidence_progression"
    if MECHANISM_RE.search(haystack):
        return "mechanism"
    if BUSINESS_RE.search(haystack):
        return "business_implication"
    return "evidence_progression"


def _dominant_strength(units: Sequence[Mapping[str, Any]]) -> str:
    if not units:
        return "weak"
    strongest = max(units, key=lambda item: STRENGTH_RANK.get(_strength(item), 0))
    return _strength(strongest)


def _chapter_claim_ids(chapter: Mapping[str, Any]) -> List[str]:
    ids = _dedupe(_as_list(chapter.get("claim_ids")))
    if ids:
        return ids
    ids = []
    for section in _as_list(chapter.get("section_plan")):
        if not isinstance(section, dict):
            continue
        ids.extend(_as_list(section.get("claim_ids")) or [_text(section.get("claim_id"))])
    return _dedupe(ids)


def _transition_in(role: str) -> str:
    return {
        "setup": "\u8fd9\u4e00\u7ae0\u5148\u628a\u95ee\u9898\u843d\u5230\u4e3b\u4f53\u884c\u52a8\u548c\u5e02\u573a\u4fe1\u53f7\u4e0a\u3002",
        "evidence_progression": "\u5728\u57fa\u672c\u5224\u65ad\u4e4b\u540e\uff0c\u9700\u8981\u628a\u591a\u4e2a\u573a\u666f\u4fe1\u53f7\u653e\u5728\u4e00\u8d77\u770b\u3002",
        "mechanism": "\u5173\u952e\u4e0d\u53ea\u662f\u5355\u70b9\u4e8b\u4ef6\uff0c\u800c\u662f\u5b83\u5982\u4f55\u6539\u53d8\u4ea7\u4e1a\u8fd0\u884c\u673a\u5236\u3002",
        "business_implication": "\u843d\u5230\u5546\u4e1a\u5316\u5c42\u9762\uff0c\u5f71\u54cd\u4f1a\u4f53\u73b0\u5728\u9700\u6c42\u3001\u4f9b\u7ed9\u548c\u7ade\u4e89\u8282\u594f\u4e0a\u3002",
        "constraint": "\u4f46\u8282\u594f\u4ecd\u7136\u53d6\u51b3\u4e8e\u8d44\u91d1\u3001\u4ea4\u4ed8\u3001\u5ba2\u6237\u91c7\u7528\u548c\u76d1\u7ba1\u8fb9\u754c\u3002",
    }.get(role, "")


def _transition_out(role: str) -> str:
    return {
        "setup": "\u56e0\u6b64\u540e\u7eed\u5206\u6790\u8981\u7ee7\u7eed\u770b\u4fe1\u53f7\u662f\u5426\u6269\u5c55\u5230\u53ef\u590d\u5236\u573a\u666f\u3002",
        "evidence_progression": "\u8fd9\u4f7f\u5f97\u5224\u65ad\u91cd\u70b9\u4ece\u662f\u5426\u6709\u4fe1\u53f7\uff0c\u8f6c\u5411\u4fe1\u53f7\u80fd\u5426\u91cd\u590d\u548c\u6269\u5c55\u3002",
        "mechanism": "\u673a\u5236\u5c42\u7684\u53d8\u5316\u51b3\u5b9a\u4e86\u8fd9\u4e9b\u4fe1\u53f7\u662f\u77ed\u671f\u4e8b\u4ef6\u8fd8\u662f\u957f\u671f\u8d8b\u52bf\u3002",
        "business_implication": "\u5546\u4e1a\u542b\u4e49\u6700\u7ec8\u8981\u56de\u5230\u4ed8\u8d39\u610f\u613f\u3001\u5c65\u7ea6\u80fd\u529b\u548c\u5229\u6da6\u7a7a\u95f4\u3002",
        "constraint": "\u6240\u4ee5\u7ed3\u8bba\u5e94\u8be5\u4fdd\u7559\u8fb9\u754c\uff0c\u628a\u53ef\u89c2\u5bdf\u4fe1\u53f7\u548c\u786e\u5b9a\u6027\u7ed3\u8bba\u533a\u5206\u5f00\u3002",
    }.get(role, "")


def _writing_goal(role: str) -> str:
    return {
        "setup": "establish the chapter thesis from the strongest public signal",
        "evidence_progression": "combine related evidence instead of rendering one claim per paragraph",
        "mechanism": "explain why the evidence changes industry structure or operating logic",
        "business_implication": "connect the evidence to demand, supply, competition, and commercialization",
        "constraint": "centralize caveats and limits without repeating diagnostic language",
    }.get(role, "write a coherent public paragraph")


def _group_claims_for_chapter(
    chapter: Mapping[str, Any],
    chapter_units: Sequence[Dict[str, Any]],
) -> List[List[Dict[str, Any]]]:
    if not chapter_units:
        return []
    groups: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    current_role = ""
    current_sources: set[str] = set()
    for index, unit in enumerate(chapter_units):
        role = _role_for_claim(unit, index_in_chapter=index)
        sources = set(_refs(unit, "source_ids", "source_refs", "citation_refs"))
        can_merge = (
            current
            and role == current_role
            and role not in {"setup", "constraint"}
            and (
                not sources
                or not current_sources
                or bool(sources & current_sources)
                or _text(unit.get("cluster_key")) == _text(current[-1].get("cluster_key"))
            )
        )
        if not can_merge:
            if current:
                groups.append(current)
            current = [unit]
            current_role = role
            current_sources = set(sources)
        else:
            current.append(unit)
            current_sources.update(sources)
    if current:
        groups.append(current)
    return groups


def _paragraph_plan(
    *,
    chapter_id: str,
    paragraph_index: int,
    units: Sequence[Dict[str, Any]],
    role: str,
) -> Dict[str, Any]:
    main = units[0]
    claim_ids = [_claim_id(unit, index) for index, unit in enumerate(units, start=1)]
    fact_ids = _dedupe(
        ref
        for unit in units
        for ref in _refs(unit, "fact_ids", "used_fact_refs", "evidence_refs", "supporting_evidence")
    )
    source_ids = _dedupe(
        ref
        for unit in units
        for ref in _refs(unit, "source_ids", "source_refs", "citation_refs", "used_source_ids")
    )
    paragraph_id = f"{chapter_id}_p{paragraph_index:02d}"
    return {
        "paragraph_id": paragraph_id,
        "chapter_id": chapter_id,
        "role": role,
        "main_claim_id": claim_ids[0],
        "supporting_claim_ids": claim_ids[1:],
        "claim_ids": claim_ids,
        "fact_ids": fact_ids,
        "source_ids": source_ids,
        "dominant_strength": _dominant_strength(units),
        "transition_in": _transition_in(role),
        "transition_out": _transition_out(role),
        "writing_goal": _writing_goal(role),
        "must_not_render": True,
        "diagnostic_only": True,
        "public_text_allowed": False,
    }


def build_chapter_narrative_plan(
    *,
    final_chapters: Sequence[Mapping[str, Any]],
    claim_units: Sequence[Mapping[str, Any]],
    claim_clusters: Sequence[Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    del claim_clusters
    units_by_id: Dict[str, Dict[str, Any]] = {}
    for index, unit in enumerate(claim_units, start=1):
        payload = dict(_as_dict(dict(unit)))
        cid = _claim_id(payload, index)
        if not cid or not _claim_text(payload):
            continue
        payload["claim_id"] = cid
        if payload.get("omit_from_report") or payload.get("public_render") is False:
            continue
        units_by_id[cid] = payload

    chapter_plans: List[Dict[str, Any]] = []
    paragraph_plan_by_claim_id: Dict[str, Dict[str, Any]] = {}
    total_claims = 0
    paragraph_count = 0
    merged_claim_count = 0

    for chapter in final_chapters:
        chapter_payload = _as_dict(dict(chapter))
        chapter_id = _text(chapter_payload.get("chapter_id")) or f"chapter_{len(chapter_plans) + 1}"
        wanted_ids = _chapter_claim_ids(chapter_payload)
        chapter_units = [units_by_id[cid] for cid in wanted_ids if cid in units_by_id]
        if not chapter_units:
            continue
        groups = _group_claims_for_chapter(chapter_payload, chapter_units)
        paragraph_plans: List[Dict[str, Any]] = []
        for paragraph_index, group in enumerate(groups, start=1):
            role = _role_for_claim(group[0], index_in_chapter=paragraph_index - 1)
            plan = _paragraph_plan(
                chapter_id=chapter_id,
                paragraph_index=paragraph_index,
                units=group,
                role=role,
            )
            paragraph_plans.append(plan)
            paragraph_count += 1
            total_claims += len(group)
            if len(group) > 1:
                merged_claim_count += len(group) - 1
            for cid in plan["claim_ids"]:
                paragraph_plan_by_claim_id[cid] = plan
        chapter_plans.append(
            {
                "chapter_id": chapter_id,
                "chapter_title": _text(chapter_payload.get("chapter_title")) or chapter_id,
                "chapter_thesis": _claim_text(chapter_units[0]),
                "writing_mode": _text(chapter_payload.get("writing_mode")) or "core_chapter",
                "narrative_arc": [item["role"] for item in paragraph_plans],
                "paragraph_plans": paragraph_plans,
                "metrics": {
                    "claim_count": len(chapter_units),
                    "paragraph_count": len(paragraph_plans),
                    "merged_claim_count": sum(max(0, len(item["claim_ids"]) - 1) for item in paragraph_plans),
                    "avg_claims_per_paragraph": len(chapter_units) / max(1, len(paragraph_plans)),
                },
            }
        )

    return {
        "schema_version": "chapter_narrative_plan_v1",
        "diagnostic_only": True,
        "must_not_render": True,
        "public_text_allowed": False,
        "chapter_plans": chapter_plans,
        "paragraph_plan_by_claim_id": paragraph_plan_by_claim_id,
        "metrics": {
            "chapter_count": len(chapter_plans),
            "claim_count": total_claims,
            "paragraph_count": paragraph_count,
            "merged_claim_count": merged_claim_count,
            "avg_claims_per_paragraph": total_claims / max(1, paragraph_count),
        },
    }


def apply_narrative_plan_to_claim_units(
    claim_units: Sequence[Mapping[str, Any]],
    narrative_plan: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    plan_by_claim_id = {
        _text(claim_id): _as_dict(plan)
        for claim_id, plan in _as_dict(narrative_plan.get("paragraph_plan_by_claim_id")).items()
        if _text(claim_id)
    }
    claim_text_by_id = {
        _claim_id(unit, index): _claim_text(unit)
        for index, unit in enumerate(claim_units, start=1)
        if _claim_text(unit)
    }
    enriched: List[Dict[str, Any]] = []
    for index, unit in enumerate(claim_units, start=1):
        payload = dict(_as_dict(dict(unit)))
        cid = _claim_id(payload, index)
        payload["claim_id"] = cid
        plan = plan_by_claim_id.get(cid)
        if not plan:
            enriched.append(payload)
            continue
        main_claim_id = _text(plan.get("main_claim_id"))
        supporting_claim_ids = _dedupe(_as_list(plan.get("supporting_claim_ids")))
        payload["paragraph_plan_id"] = _text(plan.get("paragraph_id"))
        payload["narrative_role"] = _text(plan.get("role"))
        payload["paragraph_claim_ids"] = _dedupe(_as_list(plan.get("claim_ids")))
        payload["paragraph_main_claim_id"] = main_claim_id
        payload["paragraph_supporting_claim_ids"] = supporting_claim_ids
        payload["narrative_transition_in"] = _text(plan.get("transition_in"))
        payload["narrative_transition_out"] = _text(plan.get("transition_out"))
        payload["narrative_writing_goal"] = _text(plan.get("writing_goal"))
        payload["narrative_do_not_render"] = True
        payload["narrative_plan_public_text_allowed"] = False
        if cid == main_claim_id:
            payload["fact_ids"] = _dedupe(
                [
                    *_as_list(payload.get("fact_ids")),
                    *_as_list(payload.get("used_fact_refs")),
                    *_as_list(payload.get("evidence_refs")),
                    *_as_list(plan.get("fact_ids")),
                ]
            )
            payload["source_ids"] = _dedupe(
                [
                    *_as_list(payload.get("source_ids")),
                    *_as_list(payload.get("source_refs")),
                    *_as_list(payload.get("citation_refs")),
                    *_as_list(plan.get("source_ids")),
                ]
            )
            payload["used_fact_refs"] = _dedupe([*_as_list(payload.get("used_fact_refs")), *payload["fact_ids"]])
            payload["evidence_refs"] = _dedupe([*_as_list(payload.get("evidence_refs")), *payload["fact_ids"]])
            payload["narrative_supporting_claims"] = [
                claim_text_by_id[item]
                for item in supporting_claim_ids
                if item in claim_text_by_id and claim_text_by_id[item]
            ]
        else:
            payload["omit_from_report"] = True
            payload["public_render"] = False
            payload["narrative_merged_into_claim_id"] = main_claim_id
        enriched.append(payload)
    return enriched


def _section_title_for_plan(plan: Mapping[str, Any], original: Mapping[str, Any] | None = None) -> str:
    original = _as_dict(original)
    explicit = _text(original.get("section_title") or original.get("title"))
    if explicit:
        return explicit
    role = _text(plan.get("role"))
    return {
        "setup": "\u6838\u5fc3\u5224\u65ad\u4e0e\u95ee\u9898\u843d\u70b9",
        "evidence_progression": "\u591a\u4e2a\u573a\u666f\u4fe1\u53f7\u7684\u8fde\u7eed\u6027",
        "mechanism": "\u4ece\u5355\u70b9\u4e8b\u4ef6\u5230\u4ea7\u4e1a\u673a\u5236",
        "business_implication": "\u5546\u4e1a\u5316\u542b\u4e49\u4e0e\u7ade\u4e89\u8282\u594f",
        "constraint": "\u8282\u594f\u8fb9\u754c\u4e0e\u5f85\u9a8c\u8bc1\u95ee\u9898",
    }.get(role, "\u5173\u952e\u4fe1\u53f7\u4e0e\u516c\u5f00\u8bc1\u636e")


def apply_narrative_plan_to_final_chapters(
    final_chapters: Sequence[Mapping[str, Any]],
    narrative_plan: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    chapter_plan_by_id = {
        _text(item.get("chapter_id")): _as_dict(item)
        for item in _as_list(narrative_plan.get("chapter_plans"))
        if isinstance(item, dict) and _text(item.get("chapter_id"))
    }
    rewritten: List[Dict[str, Any]] = []
    for chapter in final_chapters:
        payload = dict(_as_dict(dict(chapter)))
        chapter_id = _text(payload.get("chapter_id"))
        chapter_plan = chapter_plan_by_id.get(chapter_id)
        if not chapter_plan:
            rewritten.append(payload)
            continue
        old_sections_by_claim_id: Dict[str, Dict[str, Any]] = {}
        for section in _as_list(payload.get("section_plan")):
            if not isinstance(section, dict):
                continue
            claim_id = _text(section.get("claim_id"))
            if claim_id:
                old_sections_by_claim_id[claim_id] = dict(section)
        section_plan: List[Dict[str, Any]] = []
        for index, paragraph in enumerate(_as_list(chapter_plan.get("paragraph_plans")), start=1):
            if not isinstance(paragraph, dict):
                continue
            main_claim_id = _text(paragraph.get("main_claim_id"))
            original = old_sections_by_claim_id.get(main_claim_id, {})
            section_plan.append(
                {
                    **original,
                    "section_id": _text(paragraph.get("paragraph_id")) or f"{chapter_id}_p{index:02d}",
                    "claim_id": main_claim_id,
                    "claim_ids": _dedupe(_as_list(paragraph.get("claim_ids"))),
                    "supporting_claim_ids": _dedupe(_as_list(paragraph.get("supporting_claim_ids"))),
                    "section_title": _section_title_for_plan(paragraph, original),
                    "required_evidence_refs": _dedupe(_as_list(paragraph.get("fact_ids"))),
                    "source_ids": _dedupe(_as_list(paragraph.get("source_ids"))),
                    "narrative_role": _text(paragraph.get("role")),
                    "paragraph_plan_id": _text(paragraph.get("paragraph_id")),
                    "narrative_transition_in": _text(paragraph.get("transition_in")),
                    "narrative_transition_out": _text(paragraph.get("transition_out")),
                    "narrative_plan_diagnostic_only": True,
                    "narrative_plan_public_text_allowed": False,
                    "block_type": original.get("block_type") or original.get("output_type") or "integrated_signal",
                    "output_type": original.get("output_type") or original.get("block_type") or "integrated_signal",
                    "selection_reason": "claim_narrative_paragraph_plan",
                }
            )
        if section_plan:
            payload["section_plan"] = section_plan
            payload["narrative_paragraph_count"] = len(section_plan)
            payload["narrative_claim_merge_count"] = sum(
                max(0, len(_as_list(section.get("claim_ids"))) - 1) for section in section_plan
            )
            payload["layout_policy"] = {
                **_as_dict(payload.get("layout_policy")),
                "block_selection_source": "claim_narrative_paragraphs",
            }
        rewritten.append(payload)
    return rewritten
