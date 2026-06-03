from __future__ import annotations

from typing import Any, Dict, List, Sequence

from .report_contracts import ClaimUnit, as_list, normalize_evidence_refs


BLOCK_TYPES = {
    "metric_reconciliation",
    "case_comparison",
    "customer_painpoint_matrix",
    "technology_maturity",
    "risk_trigger",
    "verification_checklist",
    "scenario_analysis",
    "competitive_positioning",
    "unit_economics",
    "integrated_signal",
}


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _claim_dict(claim: Any) -> Dict[str, Any]:
    if isinstance(claim, ClaimUnit):
        return claim.to_legacy_dict()
    return _as_dict(claim)


def _claim_refs(claim: Dict[str, Any]) -> List[str]:
    return normalize_evidence_refs(claim)


def _claim_basis(claim: Dict[str, Any]) -> List[str]:
    values = as_list(claim.get("evidence_basis")) or as_list(claim.get("supporting_facts")) or as_list(claim.get("fact_chain"))
    return [_text(item) for item in values if _text(item)]


def claim_is_renderable(claim: Any) -> bool:
    payload = _claim_dict(claim)
    if payload.get("omit_from_report"):
        return False
    if payload.get("public_render") is False:
        return False
    return bool(_text(payload.get("claim") or payload.get("judgment")) and _claim_refs(payload) and _claim_basis(payload))


def fallback_block_for_claim(claim: Any) -> str:
    payload = _claim_dict(claim)
    explicit = [
        _text(payload.get("fact_type")),
        _text(payload.get("proof_role")),
        _text(payload.get("block_type")),
        _text(payload.get("layout_section_role")),
    ]
    if not any(explicit) and not as_list(payload.get("block_affinity")):
        return "integrated_signal"
    blob = " ".join(
        [
            *explicit,
            _text(payload.get("claim_strength")),
            " ".join(_claim_basis(payload)),
        ]
    ).lower()
    if any(token in blob for token in ("metric", "market", "size", "growth", "price", "revenue", "规模", "增速", "价格", "收入")):
        return "metric_reconciliation"
    if any(token in blob for token in ("technology", "standard", "security", "permission", "workflow", "tool", "技术", "标准", "安全", "权限")):
        return "technology_maturity"
    if any(token in blob for token in ("risk", "counter", "failure", "cost", "风险", "反证", "失败", "成本")):
        return "risk_trigger"
    if any(token in blob for token in ("competition", "player", "ecosystem", "竞争", "玩家", "生态")):
        return "competitive_positioning"
    if any(token in blob for token in ("case", "customer", "deployment", "scenario", "客户", "案例", "部署", "场景")):
        return "case_comparison"
    return "integrated_signal"


def claim_supported_block_types(claim: Any) -> List[str]:
    payload = _claim_dict(claim)
    if not claim_is_renderable(payload):
        return []
    result: List[str] = []
    for value in [payload.get("block_type"), payload.get("layout_section_role"), payload.get("output_type")]:
        text = _text(value)
        if text in BLOCK_TYPES and text not in result:
            result.append(text)
    for value in as_list(payload.get("block_affinity")):
        text = _text(value)
        if text in BLOCK_TYPES and text not in result:
            result.append(text)
    fact_type = _text(payload.get("fact_type") or payload.get("proof_role")).lower()
    mapped = {
        "metric": "metric_reconciliation",
        "market_metric": "metric_reconciliation",
        "financial_metric": "metric_reconciliation",
        "case": "case_comparison",
        "customer_case": "case_comparison",
        "technology": "technology_maturity",
        "technology_product": "technology_maturity",
        "standard": "technology_maturity",
        "counter": "risk_trigger",
        "risk": "risk_trigger",
    }.get(fact_type)
    if mapped and mapped not in result:
        result.append(mapped)
    fallback = fallback_block_for_claim(payload)
    if fallback not in result:
        result.append(fallback)
    if "integrated_signal" not in result:
        result.append("integrated_signal")
    return result


def _block_id(block: Dict[str, Any], index: int) -> str:
    return _text(block.get("block_id") or block.get("section_id") or f"block_{index}")


def match_claims_to_blocks(
    chapter_id: str,
    claims: Sequence[Any],
    blocks: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    chapter_id = _text(chapter_id)
    renderable_claims = [
        _claim_dict(claim)
        for claim in claims or []
        if claim_is_renderable(claim) and (not chapter_id or _text(_claim_dict(claim).get("chapter_id")) == chapter_id)
    ]
    matches: Dict[str, Dict[str, Any]] = {}
    used_claim_indexes: set[int] = set()
    for block_index, block in enumerate(blocks or [], start=1):
        block = _as_dict(block)
        block_type = _text(block.get("block_type") or block.get("output_type"))
        if not block_type:
            continue
        best_index = -1
        best_score = 0
        for claim_index, claim in enumerate(renderable_claims):
            if claim_index in used_claim_indexes:
                continue
            supported = claim_supported_block_types(claim)
            score = 0
            if block_type in supported:
                score = 100 if supported and supported[0] == block_type else 80
            elif block_type == "integrated_signal":
                score = 50
            if score > best_score:
                best_score = score
                best_index = claim_index
        if best_index >= 0:
            used_claim_indexes.add(best_index)
            matches[_block_id(block, block_index)] = renderable_claims[best_index]
    unmatched = [claim for index, claim in enumerate(renderable_claims) if index not in used_claim_indexes]
    return {
        "matches": matches,
        "matched_count": len(matches),
        "unmatched_count": len(unmatched),
        "unmatched_claims": unmatched,
    }


def claims_by_chapter(claims: Sequence[Any]) -> Dict[str, List[Dict[str, Any]]]:
    result: Dict[str, List[Dict[str, Any]]] = {}
    for claim in claims or []:
        payload = _claim_dict(claim)
        chapter_id = _text(payload.get("chapter_id"))
        if not chapter_id:
            continue
        result.setdefault(chapter_id, []).append(payload)
    return result
