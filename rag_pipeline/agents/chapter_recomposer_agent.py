from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from rag_pipeline.contracts.report_intent import (
    PROFESSION_EDUCATION_INTENT,
    contains_commercial_frame_term,
    is_profession_education_employment_topic,
    profession_cluster_title,
)
from .public_report_sanitizer import remove_hard_industry_templates

CLUSTER_TITLE_HINTS = {
    "market": "市场空间与资源变化",
    "policy": "政策环境与约束边界",
    "case": "场景样本与执行进展",
    "risk": "风险约束与反向条件",
    "competition": "参与主体与格局变化",
    "technology": "技术能力与应用边界",
    "customer": "用户需求与使用约束",
    "financial": "资金投入与结果验证",
    "ecosystem": "生态协作与基础条件",
    "contextual_claim": "背景变化与外部信号",
    "metric_claim": "量化信号与需求验证",
    "case_claim": "场景样本与落地进展",
    "core_claim": "核心判断与方向选择",
    "mechanism_claim": "传导机制与能力重构",
    "counter_boundary_claim": "风险边界与反向约束",
    "limitations": "限制与待验证问题",
}

ROLE_CLUSTER_TITLE_KEYS = {
    "contextual_claim",
    "metric_claim",
    "case_claim",
    "core_claim",
    "mechanism_claim",
    "counter_boundary_claim",
}

STRENGTH_RANK = {
    "strong": 4,
    "moderate": 3,
    "directional": 2,
    "limited_evidence": 2,
    "weak": 1,
    "contextual": 1,
}

REVIEW_STRENGTH_RANK = {
    "strong": 4,
    "decision_ready": 4,
    "moderate": 3,
    "medium": 3,
    "limited_evidence": 2,
    "directional": 2,
    "contextual": 1,
    "weak": 1,
    "unsupported": 0,
}


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


def _maybe_list(value: Any) -> List[Any]:
    values = _as_list(value)
    if values:
        return values
    if value in (None, "", []):
        return []
    return [value]


def _text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


_TITLE_CONNECTOR_RE = re.compile(
    r"^(.{4,36}?)(?:应从|需要从|应采取|正在被|正在向|正在|正加速|以|通过|围绕|取决于|而不是|不是|需要|应当|应该)"
)


def _looks_incomplete_title(value: Any) -> bool:
    text = _text(value)
    if not text:
        return True
    if re.search(r"(?:\.{3}|…|(?:\.\s*){2,})$", text):
        return True
    if text.count("“") > text.count("”") or text.count("《") > text.count("》"):
        return True
    if text.endswith(("和", "与", "及", "或", "、", "的", "在", "向", "从", "以", "把", "被", "+")):
        return True
    if re.search(r"[A-Z]$", text) and not text.endswith(("AI", "API", "SaaS")):
        return True
    return False


def _public_title(value: Any, *, max_chars: int = 34) -> str:
    text = remove_hard_industry_templates(_text(value)).strip(" ？?！!。.；;")
    if not text:
        return ""
    connector = _TITLE_CONNECTOR_RE.search(text)
    if connector:
        title = connector.group(1).strip(" ？?！!。.；;，,、：:")
        if 4 <= len(title) <= max_chars and not _looks_incomplete_title(title):
            return title
    if len(text) <= max_chars and not _looks_incomplete_title(text):
        return text

    candidates: List[str] = []
    head = re.split(r"[，,；;：:？?。.!！]", text, 1)[0].strip()
    if head and head != text:
        candidates.append(head)
    for separator in ("，", "、", "；", "：", ",", ";", ":"):
        if separator in text:
            candidates.append(text.split(separator, 1)[0])

    for candidate in candidates:
        title = candidate.strip(" ？?！!。.；;，,、：:")
        if 4 <= len(title) <= max_chars and not _looks_incomplete_title(title):
            return title

    fallback = text[:max_chars].strip(" ？?！!。.；;，,、：:")
    return "" if _looks_incomplete_title(fallback) else fallback


def _norm_key(value: Any) -> str:
    text = _text(value).lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", text).strip("_")
    return text or "general"


def _dedupe(values: Iterable[Any], *, limit: int = 64) -> List[str]:
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


def _chapter_list(plan_blueprint: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [item for item in _as_list(_as_dict(dict(plan_blueprint)).get("chapters")) if isinstance(item, dict)]


def mark_plan_blueprint(plan_blueprint: Mapping[str, Any]) -> Dict[str, Any]:
    payload = dict(_as_dict(dict(plan_blueprint)))
    payload["blueprint_role"] = "research_plan"
    payload["final_outline_locked"] = False
    return payload


def _claim_units(structured_analysis: Mapping[str, Any]) -> List[Dict[str, Any]]:
    payload = _as_dict(dict(structured_analysis))
    units = [item for item in _as_list(payload.get("claim_units")) if isinstance(item, dict)]
    if units:
        return units
    for chapter in _as_list(payload.get("chapter_synthesis")):
        chapter_payload = _as_dict(chapter)
        for key in ("claim_units", "claims", "argument_units"):
            units.extend(item for item in _as_list(chapter_payload.get(key)) if isinstance(item, dict))
    return units


def _refs_for_claim(unit: Mapping[str, Any]) -> Dict[str, List[str]]:
    payload = _as_dict(dict(unit))
    fact_ids = _dedupe(
        [
            *_as_list(payload.get("fact_ids")),
            *_as_list(payload.get("evidence_refs")),
            *_as_list(payload.get("used_fact_refs")),
            *_as_list(payload.get("supporting_evidence")),
        ],
        limit=16,
    )
    source_ids = _dedupe(
        [
            *_as_list(payload.get("source_ids")),
            *_as_list(payload.get("citation_refs")),
            *_as_list(payload.get("source_refs")),
        ],
        limit=16,
    )
    return {"fact_ids": fact_ids, "source_ids": source_ids}


def _can_anchor(unit: Mapping[str, Any]) -> bool:
    payload = _as_dict(dict(unit))
    if payload.get("can_anchor_section") is False:
        return False
    if payload.get("omit_from_report") or payload.get("public_render") is False:
        return False
    if not _text(payload.get("claim") or payload.get("judgment")):
        return False
    refs = _refs_for_claim(payload)
    return bool(refs["fact_ids"] or refs["source_ids"])


def _cluster_key(unit: Mapping[str, Any]) -> str:
    payload = _as_dict(dict(unit))
    explicit = _text(payload.get("cluster_key"))
    if explicit:
        return _norm_key(explicit)
    haystack = " ".join(
        _text(payload.get(key))
        for key in ("recommended_chapter", "chapter_id", "analysis_role", "proof_role", "claim")
    ).lower()
    for key in CLUSTER_TITLE_HINTS:
        if key in haystack:
            return key
    if any(token in haystack for token in ("policy", "regulation", "standard", "subsidy")):
        return "policy"
    if any(token in haystack for token in ("market", "demand", "commercial", "revenue", "order")):
        return "market"
    if any(token in haystack for token in ("risk", "constraint", "counter", "failure")):
        return "risk"
    return _norm_key(payload.get("recommended_chapter") or payload.get("chapter_id") or "general")


def _cluster_title(key: str, units: Sequence[Dict[str, Any]]) -> str:
    return _cluster_title_with_plan(key, units, plan_title_by_id={})


def _looks_like_placeholder_title(value: Any) -> bool:
    text = _text(value)
    if not text:
        return True
    if text != remove_hard_industry_templates(text):
        return True
    if re.fullmatch(r"(?:ch|chapter|cluster)[_\-\s]*\d{1,3}", text, flags=re.I):
        return True
    compact = re.sub(r"\s+", "", text)
    generic_titles = {
        "\u5e02\u573a\u7a7a\u95f4\u5230\u5e95\u6709\u591a\u5927",
        "\u73a9\u5bb6\u683c\u5c40\u5728\u54ea\u91cc",
        "\u5546\u4e1a\u5316\u4fe1\u53f7\u662f\u5426\u6e05\u6670",
        "\u98ce\u9669\u8fb9\u754c\u5728\u54ea\u91cc",
        "\u5173\u952e\u6307\u6807\u5982\u4f55\u53d8\u5316",
        "\u53cd\u5411\u4fe1\u53f7\u5982\u4f55\u5f71\u54cd\u5224\u65ad",
        "\u9700\u6c42\u9a8c\u8bc1",
        "\u5e02\u573a\u4fe1\u53f7",
        "\u6848\u4f8b\u4fe1\u53f7",
        "\u98ce\u9669\u4fe1\u53f7",
    }
    return compact in generic_titles


def _claim_section_title(unit: Mapping[str, Any]) -> str:
    payload = _as_dict(dict(unit))
    for key in ("section_title", "recommended_chapter", "question", "dimension"):
        title = _text(payload.get(key))
        if title and not _looks_like_placeholder_title(title):
            return _public_title(title, max_chars=34) or title[:80]
    claim = _text(payload.get("claim") or payload.get("judgment") or payload.get("conclusion"))
    if not claim:
        return "Claim-driven section"
    first = re.split(r"[。；;！？!?]", claim, maxsplit=1)[0].strip()
    return _public_title(first or claim, max_chars=34) or (first or claim)[:80]


def _stable_claim_id(unit: Mapping[str, Any], index: int) -> str:
    payload = _as_dict(dict(unit))
    existing = _text(payload.get("claim_id") or payload.get("id"))
    if existing:
        return existing
    chapter_key = _norm_key(payload.get("chapter_id") or payload.get("recommended_chapter") or payload.get("cluster_key") or "claim")
    return f"{chapter_key}_claim_{index}"


def _review_suggested_strength(unit: Mapping[str, Any]) -> str:
    candidates: List[str] = []
    for suggestion in _as_list(unit.get("claim_review_suggestions")):
        if not isinstance(suggestion, dict):
            continue
        strength = _text(suggestion.get("suggested_claim_strength")).lower()
        if strength:
            candidates.append(strength)
    if not candidates:
        return ""
    return min(candidates, key=lambda value: REVIEW_STRENGTH_RANK.get(value, 0))


def _apply_review_constraints(unit: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(unit)
    semantic_status = _text(payload.get("semantic_judge_status")).lower()
    suggested_strength = _review_suggested_strength(payload)
    if semantic_status in {"partial", "adjacent", "unsupported"} and not suggested_strength:
        suggested_strength = "directional" if semantic_status in {"partial", "adjacent"} else "weak"
    if suggested_strength:
        current_strength = _text(payload.get("claim_strength") or payload.get("claim_status")).lower()
        if not current_strength or REVIEW_STRENGTH_RANK.get(current_strength, 0) > REVIEW_STRENGTH_RANK.get(suggested_strength, 0):
            payload["claim_strength"] = suggested_strength
            payload["claim_strength_reviewed_from"] = current_strength
            payload["claim_review_strength_applied"] = True
    if semantic_status in {"adjacent", "unsupported"}:
        payload["must_not_use_as_core_view"] = True
        payload["writing_mode_hint"] = "limitations" if semantic_status == "unsupported" else "directional_observation"
        payload["section_role"] = "boundary_context"
        payload["block_type"] = "risk_trigger"
    elif semantic_status == "partial":
        payload["writing_mode_hint"] = "directional_observation"
    return payload


def normalize_claim_units(claim_units: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for index, unit in enumerate(claim_units, start=1):
        payload = _apply_review_constraints(dict(_as_dict(dict(unit))))
        if not _text(payload.get("claim") or payload.get("judgment") or payload.get("conclusion")):
            continue
        claim_id = _stable_claim_id(payload, index)
        payload["claim_id"] = claim_id
        if not _text(payload.get("section_id")):
            safe = re.sub(r"[^0-9A-Za-z_\-]+", "_", claim_id).strip("_") or f"claim_{index}"
            payload["section_id"] = f"{safe}_section"
        if not _text(payload.get("section_title")):
            payload["section_title"] = _claim_section_title(payload)
        normalized.append(payload)
    return normalized


def _cluster_title_with_plan(
    key: str,
    units: Sequence[Dict[str, Any]],
    *,
    plan_title_by_id: Mapping[str, str],
    report_intent: str = "",
) -> str:
    if report_intent == PROFESSION_EDUCATION_INTENT:
        fallback_text = " ".join(
            _text(unit.get(field))
            for unit in units
            for field in ("claim", "recommended_chapter", "chapter_title", "section_title", "question", "dimension")
        )
        return profession_cluster_title(key, fallback_text)
    if key == "limitations":
        return CLUSTER_TITLE_HINTS["limitations"]
    source_plan_ids = _dedupe((_text(unit.get("chapter_id")) for unit in units if _text(unit.get("chapter_id"))), limit=8)
    if key in ROLE_CLUSTER_TITLE_KEYS and key in CLUSTER_TITLE_HINTS and len(source_plan_ids) != 1:
        return CLUSTER_TITLE_HINTS[key]
    for unit in units:
        for field in ("recommended_chapter", "chapter_title", "section_title", "question", "dimension"):
            title = _text(unit.get(field))
            if title and not _looks_like_placeholder_title(title):
                return _public_title(title, max_chars=34) or title
        plan_title = _text(plan_title_by_id.get(_text(unit.get("chapter_id"))))
        if plan_title and not _looks_like_placeholder_title(plan_title):
            return _public_title(plan_title, max_chars=34) or plan_title
    if key in CLUSTER_TITLE_HINTS:
        return CLUSTER_TITLE_HINTS[key]
    for unit in units:
        title = _text(unit.get("recommended_chapter"))
        if title and not _looks_like_placeholder_title(title):
            return _public_title(title, max_chars=34) or title
    return _text(key).replace("_", " ").title() or "Evidence-Driven Analysis"


def _dominant_strength(units: Sequence[Dict[str, Any]]) -> str:
    strengths = [_text(unit.get("claim_strength") or unit.get("claim_status")).lower() for unit in units]
    strengths = [item for item in strengths if item]
    if not strengths:
        return "directional"
    return max(strengths, key=lambda item: STRENGTH_RANK.get(item, 0))


def _writing_mode(strength: str, claim_count: int) -> str:
    if strength in {"strong", "moderate"} and claim_count >= 1:
        return "core_chapter"
    if strength in {"directional", "limited_evidence"}:
        return "directional_observation"
    return "limitations"


def _is_thin_observation_cluster(cluster: Mapping[str, Any]) -> bool:
    strength = _text(cluster.get("dominant_strength")).lower()
    mode = _text(cluster.get("writing_mode")).lower()
    try:
        claim_count = int(cluster.get("claim_count") or len(_as_list(cluster.get("claim_ids"))))
    except (TypeError, ValueError):
        claim_count = len(_as_list(cluster.get("claim_ids")))
    return claim_count <= 1 and (
        mode in {"directional_observation", "limitations"}
        or strength in {"directional", "limited_evidence", "weak", "contextual"}
    )


def build_claim_clusters(
    claim_units: Sequence[Mapping[str, Any]],
    *,
    plan_blueprint: Mapping[str, Any] | None = None,
    report_intent: str = "",
) -> List[Dict[str, Any]]:
    plan_chapters = _chapter_list(_as_dict(plan_blueprint))
    plan_ids = {_text(chapter.get("chapter_id")) for chapter in plan_chapters}
    plan_title_by_id = {
        _text(chapter.get("chapter_id")): _text(
            chapter.get("chapter_title")
            or chapter.get("chapter_question")
            or chapter.get("title")
            or chapter.get("core_question")
        )
        for chapter in plan_chapters
        if _text(chapter.get("chapter_id"))
    }
    plan_order_by_id = {
        _text(chapter.get("chapter_id")): index
        for index, chapter in enumerate(plan_chapters, start=1)
        if _text(chapter.get("chapter_id"))
    }
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for unit in normalize_claim_units(claim_units):
        payload = dict(_as_dict(dict(unit)))
        if not _can_anchor(payload):
            continue
        key = _cluster_key(payload)
        explicit_key = _norm_key(payload.get("cluster_key"))
        chapter_id = _text(payload.get("chapter_id"))
        recommended = _text(payload.get("recommended_chapter") or payload.get("chapter_title"))
        plan_title = _text(plan_title_by_id.get(chapter_id))
        if (
            explicit_key in ROLE_CLUSTER_TITLE_KEYS
            and chapter_id in plan_ids
            and recommended
            and plan_title
            and _public_title_key(recommended) == _public_title_key(plan_title)
        ):
            key = _norm_key(chapter_id)
        payload["cluster_key"] = key
        grouped[key].append(payload)

    clusters: List[Dict[str, Any]] = []
    for index, (key, units) in enumerate(grouped.items(), start=1):
        fact_ids = _dedupe((ref for unit in units for ref in _refs_for_claim(unit)["fact_ids"]), limit=80)
        source_ids = _dedupe((ref for unit in units for ref in _refs_for_claim(unit)["source_ids"]), limit=80)
        requirement_ids = _dedupe((req for unit in units for req in _maybe_list(unit.get("requirement_ids") or unit.get("requirement_id"))), limit=80)
        plan_chapter_ids = _dedupe((unit.get("chapter_id") for unit in units if _text(unit.get("chapter_id"))), limit=24)
        plan_chapter_titles = _dedupe((plan_title_by_id.get(item) for item in plan_chapter_ids if plan_title_by_id.get(item)), limit=24)
        plan_order = min((plan_order_by_id[item] for item in plan_chapter_ids if item in plan_order_by_id), default=10_000 + index)
        strength = _dominant_strength(units)
        mode = _writing_mode(strength, len(units))
        suggested = "write" if mode in {"core_chapter", "directional_observation"} else "downgrade"
        if mode == "limitations":
            suggested = "merge"
        claim_sections = []
        for offset, unit in enumerate(units, start=1):
            refs = _refs_for_claim(unit)
            boundary_section = bool(unit.get("must_not_use_as_core_view"))
            section_title = _claim_section_title(unit)
            if report_intent == PROFESSION_EDUCATION_INTENT and contains_commercial_frame_term(section_title):
                section_title = profession_cluster_title(key, unit.get("claim") or unit.get("judgment") or unit.get("conclusion"))
            claim_sections.append(
                {
                    "claim_id": unit.get("claim_id"),
                    "section_title": section_title,
                    "required_evidence_refs": refs["fact_ids"] or refs["source_ids"],
                    "must_not_use_as_core_view": boundary_section,
                    "writing_mode": unit.get("writing_mode_hint") or mode,
                    "matched_llm_claim": {
                        "claim_id": unit.get("claim_id"),
                        "claim": unit.get("claim") or unit.get("judgment") or unit.get("conclusion"),
                        "chapter_id": unit.get("chapter_id"),
                        "cluster_key": key,
                        "fact_ids": refs["fact_ids"],
                        "source_ids": refs["source_ids"],
                        "requirement_ids": _maybe_list(unit.get("requirement_ids") or unit.get("requirement_id")),
                        "claim_strength": unit.get("claim_strength"),
                        "evidence_status": unit.get("evidence_status"),
                        "semantic_judge_status": unit.get("semantic_judge_status"),
                        "must_not_use_as_core_view": boundary_section,
                    },
                    "section_role": unit.get("section_role") or ("boundary_context" if boundary_section else "claim_driven_argument"),
                    "block_type": unit.get("block_type") or ("risk_trigger" if boundary_section else "integrated_signal"),
                }
            )
        clusters.append(
            {
                "cluster_id": f"cluster_{key or index}",
                "cluster_key": key,
                "cluster_title": _cluster_title_with_plan(key, units, plan_title_by_id=plan_title_by_id, report_intent=report_intent),
                "claim_ids": _dedupe((unit.get("claim_id") for unit in units)),
                "fact_ids": fact_ids,
                "source_ids": source_ids,
                "requirement_ids": requirement_ids,
                "source_plan_chapter_ids": plan_chapter_ids,
                "source_plan_chapter_titles": plan_chapter_titles,
                "plan_order": plan_order,
                "claim_sections": claim_sections,
                "dominant_strength": strength if strength in STRENGTH_RANK else "directional",
                "writing_mode": mode,
                "suggested_action": suggested,
                "claim_count": len(units),
                "recomposition_action": "kept" if any(item in plan_ids for item in plan_chapter_ids) else "new",
            }
        )
    clusters.sort(
        key=lambda cluster: (
            int(cluster.get("plan_order") or 10_000),
            str(cluster.get("cluster_title") or cluster.get("cluster_id") or ""),
        )
    )
    return clusters


def _merge_thin_observation_clusters(clusters: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(clusters) <= 1:
        return [dict(cluster) for cluster in clusters]
    strong_clusters = [dict(cluster) for cluster in clusters if not _is_thin_observation_cluster(cluster)]
    thin_clusters = [dict(cluster) for cluster in clusters if _is_thin_observation_cluster(cluster)]
    if not strong_clusters or not thin_clusters:
        return [dict(cluster) for cluster in clusters]
    merged_sections = [
        section
        for cluster in thin_clusters
        for section in _as_list(cluster.get("claim_sections"))
        if isinstance(section, dict)
    ]
    merged = {
        "cluster_id": "cluster_limitations",
        "cluster_key": "limitations",
        "cluster_title": CLUSTER_TITLE_HINTS["limitations"],
        "claim_ids": _dedupe((claim for cluster in thin_clusters for claim in _as_list(cluster.get("claim_ids"))), limit=120),
        "fact_ids": _dedupe((ref for cluster in thin_clusters for ref in _as_list(cluster.get("fact_ids"))), limit=120),
        "source_ids": _dedupe((ref for cluster in thin_clusters for ref in _as_list(cluster.get("source_ids"))), limit=120),
        "requirement_ids": _dedupe((ref for cluster in thin_clusters for ref in _as_list(cluster.get("requirement_ids"))), limit=120),
        "source_plan_chapter_ids": _dedupe(
            (ref for cluster in thin_clusters for ref in _as_list(cluster.get("source_plan_chapter_ids"))),
            limit=80,
        ),
        "claim_sections": merged_sections,
        "dominant_strength": _dominant_strength(
            [
                {"claim_strength": cluster.get("dominant_strength")}
                for cluster in thin_clusters
                if isinstance(cluster, dict)
            ]
        ),
        "writing_mode": "limitations",
        "suggested_action": "write",
        "claim_count": sum(len(_as_list(cluster.get("claim_ids"))) for cluster in thin_clusters),
        "recomposition_action": "merged",
        "merged_cluster_ids": _dedupe((cluster.get("cluster_id") for cluster in thin_clusters), limit=80),
    }
    return [*strong_clusters, merged]


def _public_title_key(value: Any) -> str:
    text = _text(value).lower()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[.。…]+$", "", text)
    return text


def _merge_duplicate_title_clusters(clusters: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    order: List[str] = []
    for cluster in clusters:
        item = dict(cluster)
        source_plan_ids = _dedupe(_as_list(item.get("source_plan_chapter_ids")), limit=4)
        key = f"plan:{source_plan_ids[0]}" if len(source_plan_ids) == 1 else ""
        if not key and _text(item.get("cluster_key")):
            key = f"cluster:{_text(item.get('cluster_key'))}"
        if not key:
            key = _public_title_key(item.get("cluster_title") or item.get("cluster_id"))
        if not key:
            key = _text(item.get("cluster_id")) or f"cluster_{len(order) + 1}"
        if key not in grouped:
            order.append(key)
        grouped[key].append(item)

    merged: List[Dict[str, Any]] = []
    for key in order:
        items = sorted(
            grouped[key],
            key=lambda item: (
                int(item.get("plan_order") or 10_000),
                -STRENGTH_RANK.get(_text(item.get("dominant_strength")).lower(), 0),
                _text(item.get("cluster_id")),
            ),
        )
        if len(items) == 1:
            merged.append(dict(items[0]))
            continue
        claim_sections = [
            section
            for item in items
            for section in _as_list(item.get("claim_sections"))
            if isinstance(section, dict)
        ]
        strength = _dominant_strength(
            [{"claim_strength": item.get("dominant_strength")} for item in items]
        )
        modes = {_text(item.get("writing_mode")).lower() for item in items}
        mode = "core_chapter" if "core_chapter" in modes else _writing_mode(strength, sum(len(_as_list(item.get("claim_ids"))) for item in items))
        recomposition_actions = {_text(item.get("recomposition_action")).lower() for item in items}
        merged.append(
            {
                **items[0],
                "cluster_id": _text(items[0].get("cluster_id")) or f"cluster_{key}",
                "cluster_title": (
                    _text(_as_list(items[0].get("source_plan_chapter_titles"))[0])
                    if _as_list(items[0].get("source_plan_chapter_titles"))
                    else items[0].get("cluster_title")
                ),
                "claim_ids": _dedupe((claim for item in items for claim in _as_list(item.get("claim_ids"))), limit=160),
                "fact_ids": _dedupe((ref for item in items for ref in _as_list(item.get("fact_ids"))), limit=160),
                "source_ids": _dedupe((ref for item in items for ref in _as_list(item.get("source_ids"))), limit=160),
                "requirement_ids": _dedupe((ref for item in items for ref in _as_list(item.get("requirement_ids"))), limit=160),
                "source_plan_chapter_ids": _dedupe(
                    (ref for item in items for ref in _as_list(item.get("source_plan_chapter_ids"))),
                    limit=120,
                ),
                "source_plan_chapter_titles": _dedupe(
                    (ref for item in items for ref in _as_list(item.get("source_plan_chapter_titles"))),
                    limit=120,
                ),
                "claim_sections": claim_sections,
                "dominant_strength": strength if strength in STRENGTH_RANK else "directional",
                "writing_mode": mode,
                "suggested_action": "write" if mode in {"core_chapter", "directional_observation"} else "merge",
                "claim_count": sum(len(_as_list(item.get("claim_ids"))) for item in items),
                "recomposition_action": "merged" if len(items) > 1 or "merged" in recomposition_actions else items[0].get("recomposition_action", "kept"),
                "merged_cluster_ids": _dedupe((item.get("cluster_id") for item in items), limit=120),
                "plan_order": min((int(item.get("plan_order") or 10_000) for item in items), default=10_000),
            }
        )
    return merged


def _fallback_final_chapter_id(cluster: Mapping[str, Any], index: int) -> str:
    cluster_id = _text(cluster.get("cluster_id")) or f"cluster_{index}"
    return "CH_" + re.sub(r"^cluster_?", "", cluster_id, flags=re.I)


def _final_chapter_id(cluster: Mapping[str, Any], index: int, used_ids: set[str]) -> str:
    source_plan_ids = _as_list(cluster.get("source_plan_chapter_ids"))
    candidate = _text(source_plan_ids[0]) if len(source_plan_ids) == 1 else ""
    if not candidate:
        candidate = _fallback_final_chapter_id(cluster, index)
    if candidate in used_ids:
        candidate = _fallback_final_chapter_id(cluster, index)
    used_ids.add(candidate)
    return candidate


def _chapter_id_aliases(chapter_id: str, cluster: Mapping[str, Any]) -> List[str]:
    return _dedupe(
        [
            chapter_id,
            cluster.get("cluster_id"),
            *_as_list(cluster.get("source_plan_chapter_ids")),
        ],
        limit=24,
    )


def _final_chapters_from_clusters(clusters: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    chapters: List[Dict[str, Any]] = []
    used_ids: set[str] = set()
    merged_clusters = _merge_thin_observation_clusters(_merge_duplicate_title_clusters(clusters))
    for index, cluster in enumerate(merged_clusters, start=1):
        chapter_id = _final_chapter_id(cluster, index, used_ids)
        claim_ids = _as_list(cluster.get("claim_ids"))
        claim_sections = [item for item in _as_list(cluster.get("claim_sections")) if isinstance(item, dict)]
        chapters.append(
            {
                "chapter_id": chapter_id,
                "chapter_title": cluster.get("cluster_title") or f"Chapter {index}",
                "chapter_question": cluster.get("cluster_title") or f"Chapter {index}",
                "core_question": cluster.get("cluster_title") or f"Chapter {index}",
                "chapter_role": "claim_driven_final_chapter",
                "order": index,
                "claim_ids": claim_ids,
                "fact_ids": _as_list(cluster.get("fact_ids")),
                "source_ids": _as_list(cluster.get("source_ids")),
                "requirement_ids": _as_list(cluster.get("requirement_ids")),
                "source_plan_chapter_ids": _as_list(cluster.get("source_plan_chapter_ids")),
                "chapter_id_aliases": _chapter_id_aliases(chapter_id, cluster),
                "section_plan": [
                    {
                        **section,
                        "section_id": f"{chapter_id}_sec_{offset:02d}",
                        "claim_id": section.get("claim_id") or claim_id,
                        "section_role": section.get("section_role") or "claim_driven_argument",
                        "block_type": section.get("block_type") or "integrated_signal",
                        "output_type": section.get("output_type") or section.get("block_type") or "integrated_signal",
                    }
                    for offset, (claim_id, section) in enumerate(
                        zip(claim_ids, claim_sections or [{"claim_id": claim_id} for claim_id in claim_ids]),
                        start=1,
                    )
                ],
                "recomposition_action": cluster.get("recomposition_action") or "new",
                "writing_mode": cluster.get("writing_mode") or "core_chapter",
                "selection": {"source": "claim_first_recomposer", "order": index},
                "layout_policy": {
                    "preferred_blocks": ["claim_argument", "evidence", "mechanism", "boundary", "implication"],
                    "block_selection_source": "claim_units",
                },
            }
        )
    return chapters


def recompose_chapters_from_claims(
    *,
    plan_blueprint: Mapping[str, Any] | None,
    structured_analysis: Mapping[str, Any] | None,
    evidence_package: Mapping[str, Any] | None = None,
    query: str = "",
) -> Dict[str, Any]:
    del evidence_package
    plan = mark_plan_blueprint(_as_dict(plan_blueprint))
    report_intent = (
        PROFESSION_EDUCATION_INTENT
        if is_profession_education_employment_topic(query, plan)
        else _text(plan.get("report_intent"))
    )
    units = normalize_claim_units(_claim_units(_as_dict(structured_analysis)))
    clusters = build_claim_clusters(units, plan_blueprint=plan, report_intent=report_intent)
    final_chapters = _final_chapters_from_clusters(clusters)
    plan_ids = [_text(chapter.get("chapter_id")) for chapter in _chapter_list(plan)]
    covered_plan_ids = {
        item
        for cluster in clusters
        for item in _as_list(cluster.get("source_plan_chapter_ids"))
        if _text(item)
    }
    dropped = [item for item in plan_ids if item and item not in covered_plan_ids]
    anchored_claim_ids = {item for cluster in clusters for item in _as_list(cluster.get("claim_ids"))}
    total_claim_count = len([unit for unit in units if _text(unit.get("claim") or unit.get("judgment"))])
    transfer_rate = round(len(anchored_claim_ids) / total_claim_count, 4) if total_claim_count else 0.0
    status = "claim_first" if final_chapters else "fallback_to_plan"
    recomposed_blueprint = {
        **plan,
        "blueprint_role": "final_outline",
        "final_outline_locked": True,
        "layout_strategy": {
            **_as_dict(plan.get("layout_strategy")),
            "source": "claim_first_recomposer",
            "plan_chapters_are_research_reference": True,
        },
        "research_object": plan.get("research_object") or _text(query),
        "report_intent": report_intent,
        "claim_clusters": clusters,
        "final_chapters": final_chapters,
        "chapters": final_chapters if final_chapters else _chapter_list(plan),
    }
    return {
        "schema_version": "chapter_recomposition_v1",
        "status": status,
        "plan_blueprint": plan,
        "claim_clusters": clusters,
        "final_chapters": final_chapters,
        "normalized_claim_units": units,
        "report_blueprint": recomposed_blueprint if final_chapters else {},
        "dropped_plan_chapter_ids": dropped,
        "metrics": {
            "plan_chapter_count": len(plan_ids),
            "final_chapter_count": len(final_chapters),
            "claim_unit_count": total_claim_count,
            "bound_claim_count": len(anchored_claim_ids),
            "claim_cluster_count": len(clusters),
            "claim_to_final_chapter_transfer_rate": transfer_rate,
            "reanalyze_existing_recommended": bool(total_claim_count > 0 and len(anchored_claim_ids) <= max(1, total_claim_count // 3)),
            "recompose_outline_recommended": bool(final_chapters and dropped),
        },
    }
