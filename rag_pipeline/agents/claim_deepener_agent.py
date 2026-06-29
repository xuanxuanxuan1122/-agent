from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Sequence

from .public_narrative_bridge import build_public_bridge_pack


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _dedupe(values: Iterable[Any], *, limit: int = 8) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        text = _text(value)
        if not text:
            continue
        key = re.sub(r"\W+", "", text.lower())[:140]
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def _fact_id(item: Dict[str, Any]) -> str:
    return _text(item.get("evidence_id") or item.get("fact_id") or item.get("id") or item.get("source_ref") or item.get("ref"))


def _fact_text(item: Dict[str, Any]) -> str:
    card = _as_dict(item.get("public_fact_card")) or _as_dict(_as_dict(item.get("public_fact_quality")).get("public_fact_card"))
    return _text(
        card.get("distilled_fact")
        or item.get("distilled_fact")
        or item.get("public_fact")
        or item.get("fact")
        or item.get("summary")
        or item.get("content")
    )


def _source_id(item: Dict[str, Any]) -> str:
    return _text(item.get("source_id") or item.get("canonical_source_id") or item.get("source_ref") or item.get("ref"))


def _evidence_index(chapter_evidence_packages: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    collections = (
        "core_evidence",
        "supporting_evidence",
        "metric_evidence",
        "case_evidence",
        "counter_evidence",
        "risk_evidence",
        "evidence_items",
        "sample_evidence",
    )
    for package in list(chapter_evidence_packages or []):
        if not isinstance(package, dict):
            continue
        for key in collections:
            for raw in _as_list(package.get(key)):
                if not isinstance(raw, dict):
                    continue
                fact_id = _fact_id(raw)
                fact = _fact_text(raw)
                if not fact_id or not fact:
                    continue
                payload = {
                    **raw,
                    "chapter_id": raw.get("chapter_id") or package.get("chapter_id"),
                    "chapter_title": raw.get("chapter_title") or package.get("chapter_title"),
                    "distilled_fact": fact,
                }
                for alias in _dedupe([fact_id, raw.get("source_ref"), raw.get("ref"), raw.get("citation_ref")], limit=6):
                    index.setdefault(alias, payload)
    return index


def _unit_refs(unit: Dict[str, Any]) -> List[str]:
    return _dedupe(
        [
            *_as_list(unit.get("used_fact_refs")),
            *_as_list(unit.get("evidence_refs")),
            *_as_list(unit.get("used_evidence_ids")),
            *_as_list(unit.get("supporting_evidence_refs")),
        ],
        limit=8,
    )


def _not_allowed_until_repaired(unit: Dict[str, Any]) -> bool:
    for suggestion in _as_list(unit.get("claim_review_suggestions")):
        if not isinstance(suggestion, dict):
            continue
        repair = _as_dict(suggestion.get("repair_priority"))
        permission = _text(repair.get("writing_permission") or suggestion.get("suggested_writing_permission"))
        action = _text(repair.get("recommended_action") or suggestion.get("recommended_action"))
        if repair.get("allowed_for_writing") is False:
            return True
        if permission in {"not_allowed_until_repaired", "repair_before_publication"}:
            return True
        if action == "repair_evidence_binding_then_rebuild_claim":
            return True
    return False


def _unit_fact_texts(unit: Dict[str, Any], evidence_by_ref: Dict[str, Dict[str, Any]]) -> List[str]:
    ref_facts = [_fact_text(evidence_by_ref.get(ref, {})) for ref in _unit_refs(unit)]
    inline_facts = [
        _text(item.get("distilled_fact") or item.get("fact") or item.get("summary"))
        if isinstance(item, dict)
        else _text(item)
        for item in [
            *_as_list(unit.get("evidence_basis")),
            *_as_list(unit.get("supporting_facts")),
            *_as_list(unit.get("fact_chain")),
        ]
    ]
    return _dedupe([*ref_facts, *inline_facts], limit=6)


def _prefers_chinese(*values: Any) -> bool:
    text = " ".join(_text(value) for value in values if _text(value))
    return len(re.findall(r"[\u4e00-\u9fff]", text)) >= 6


def _claim_headline(value: Any, *, max_chars: int = 42) -> str:
    text = _text(value)
    if not text:
        return ""
    head = re.split(r"[\u3002\uff1b;.!?\uff01\uff1f\n]", text, 1)[0].strip(" \uff0c\uff0c\u201c\u201d\"'")
    if not head:
        return ""
    return head[:max_chars].rstrip(" \uff0c,\uff1a:")


def build_claim_depth_pack(unit: Dict[str, Any], *, evidence_by_ref: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    unit = _as_dict(unit)
    if _not_allowed_until_repaired(unit):
        return {}
    existing = _as_dict(unit.get("claim_depth_pack"))
    if existing:
        return existing
    claim = _text(unit.get("claim") or unit.get("judgment") or unit.get("conclusion"))
    if not claim:
        return {}
    refs = _unit_refs(unit)
    facts = _unit_fact_texts(unit, evidence_by_ref)
    if not refs or not facts:
        return {}
    reasoning = _text(unit.get("reasoning") or unit.get("mechanism") or unit.get("reasoning_chain"))
    implication = _text(unit.get("decision_implication") or unit.get("actionable") or unit.get("implication"))
    boundary = _text(unit.get("limitation_boundary") or unit.get("counter_evidence") or unit.get("boundary"))
    source_ids = _dedupe([_source_id(evidence_by_ref.get(ref, {})) for ref in refs], limit=8)
    fact_summary = "; ".join(facts[:3])
    chinese = _prefers_chinese(claim, fact_summary)
    claim_head = _claim_headline(claim)
    if chinese:
        fallback_evidence = f"公开材料提到，{fact_summary}。"
        fallback_mechanism = "材料的价值在于把可观察活动、应用场景和执行条件连接起来，而不是只提供背景描述。"
        fallback_segmentation = "直接动作比泛化描述更接近真实落地，仍需验证的外推部分则应保留为阶段性观察。"
        fallback_implication = "该信号可以解释需求、供给或监管条件如何变化，结论范围仍限制在已引用来源覆盖的场景内。"
        fallback_boundary = "结论仍受已引用来源覆盖范围限制，不能外推为超出证据边界的强结论。"
    else:
        fallback_evidence = f"The available evidence supports this judgement through these cited facts: {fact_summary}."
        fallback_mechanism = "The cited facts should be interpreted as a link between observable activity, adoption context, and practical execution conditions."
        fallback_segmentation = "Directly observed actions carry more weight than background context, so the signal is directional rather than market-wide proof."
        fallback_implication = "This signal helps explain how demand, supply, or governance conditions may change while remaining bounded by the cited sources."
        fallback_boundary = "The conclusion remains limited to the cited source scope and should not be extended beyond that scope."
    if chinese and not _text(unit.get("segmentation")):
        fallback_segmentation = "直接动作比泛化描述更接近真实落地，仍需验证的外推部分则保留为阶段性观察。"
    bridge_pack = build_public_bridge_pack(
        claim=claim,
        evidence_texts=facts,
        block_type=str(unit.get("block_type") or unit.get("section_role") or unit.get("analysis_role") or ""),
        claim_strength=str(unit.get("claim_strength") or ""),
        boundary=boundary,
    )
    if not _text(unit.get("segmentation")) and _text(bridge_pack.get("mechanism_bridge")):
        fallback_segmentation = _text(bridge_pack.get("mechanism_bridge"))
    if not implication and _text(bridge_pack.get("implication_bridge")):
        fallback_implication = _text(bridge_pack.get("implication_bridge"))
    if not boundary and _text(bridge_pack.get("boundary_bridge")):
        fallback_boundary = _text(bridge_pack.get("boundary_bridge"))
    return {
        "schema_version": "claim_depth_pack_v1",
        "claim_id": _text(unit.get("claim_id") or unit.get("id")),
        "judgement": claim,
        "evidence_chain": reasoning or fallback_evidence,
        "mechanism": reasoning or fallback_mechanism,
        "segmentation": _text(unit.get("segmentation"))
        or fallback_segmentation,
        "implication": implication or fallback_implication,
        "boundary": boundary or fallback_boundary,
        "used_fact_refs": refs,
        "used_source_ids": source_ids,
        "diagnostic_only": False,
        "must_not_render": False,
        "public_text_allowed": True,
    }


def enrich_claim_units_with_depth_packs(
    claim_units: Sequence[Dict[str, Any]],
    *,
    chapter_evidence_packages: Sequence[Dict[str, Any]] = (),
) -> Dict[str, Any]:
    evidence_by_ref = _evidence_index(chapter_evidence_packages)
    enriched: List[Dict[str, Any]] = []
    generated = 0
    preserved = 0
    missing = 0
    for raw in list(claim_units or []):
        if not isinstance(raw, dict):
            continue
        unit = dict(raw)
        if _as_dict(unit.get("claim_depth_pack")):
            preserved += 1
            enriched.append(unit)
            continue
        pack = build_claim_depth_pack(unit, evidence_by_ref=evidence_by_ref)
        if pack:
            unit = {**unit, "claim_depth_pack": pack}
            generated += 1
        else:
            missing += 1
        enriched.append(unit)
    total = len(enriched)
    return {
        "claim_units": enriched,
        "diagnostics": {
            "schema_version": "claim_depth_pack_diagnostics_v1",
            "claim_depth_pack_count": generated + preserved,
            "claim_depth_pack_generated_count": generated,
            "claim_depth_pack_preserved_count": preserved,
            "claim_depth_pack_missing_count": missing,
            "claim_depth_coverage_rate": ((generated + preserved) / total) if total else 0.0,
        },
    }
