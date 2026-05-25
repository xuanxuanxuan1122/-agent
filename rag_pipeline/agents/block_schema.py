from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Sequence


BLOCK_REGISTRY: Dict[str, Dict[str, Any]] = {
    "thesis": {"label": "本章结论", "renderer": "render_thesis_block", "roles": ["support", "metric"]},
    "evidence_matrix": {"label": "事实依据", "renderer": "render_evidence_matrix", "roles": ["support", "metric"]},
    "metric_reconciliation": {"label": "指标口径与可比性", "renderer": "render_metric_reconciliation", "roles": ["metric"]},
    "mechanism_chain": {"label": "影响路径与约束", "renderer": "render_mechanism_chain", "roles": ["support", "policy_original"]},
    "policy_timeline": {"label": "政策节点与执行节奏", "renderer": "render_policy_timeline", "roles": ["policy_original", "official_data"]},
    "stakeholder_map": {"label": "相关主体与影响分化", "renderer": "render_stakeholder_map", "roles": ["case", "official_data"]},
    "value_chain_map": {"label": "价值链位置与瓶颈", "renderer": "render_value_chain_map", "roles": ["metric", "company_filing"]},
    "case_comparison": {"label": "代表性案例对比", "renderer": "render_case_comparison", "roles": ["case", "customer_case"]},
    "customer_painpoint_matrix": {"label": "客户场景与采购约束", "renderer": "render_customer_painpoint_matrix", "roles": ["customer_case", "case"]},
    "competitive_positioning": {"label": "竞争位置与替代压力", "renderer": "render_competitive_positioning", "roles": ["case", "metric"]},
    "technology_maturity": {"label": "技术成熟度与量产边界", "renderer": "render_technology_maturity", "roles": ["technology_product", "case"]},
    "unit_economics": {"label": "商业化证据", "renderer": "render_unit_economics", "roles": ["financial_metric", "filing"]},
    "scenario_analysis": {"label": "情景分层与结论弹性", "renderer": "render_scenario_analysis", "roles": ["metric", "counter"]},
    "risk_trigger": {"label": "反向信号与失效条件", "renderer": "render_risk_trigger", "roles": ["counter"]},
    "verification_checklist": {"label": "后续验证重点", "renderer": "render_verification_checklist", "roles": ["support", "counter"]},
}


MODULE_DEFAULT_BLOCKS: Dict[str, List[str]] = {
    "industry_definition": ["thesis", "evidence_matrix", "verification_checklist"],
    "market_size": ["thesis", "metric_reconciliation", "evidence_matrix", "risk_trigger"],
    "demand_driver": ["thesis", "evidence_matrix", "scenario_analysis", "risk_trigger"],
    "industry_chain": ["thesis", "value_chain_map", "mechanism_chain", "risk_trigger"],
    "technology": ["thesis", "technology_maturity", "case_comparison", "risk_trigger"],
    "customer": ["thesis", "customer_painpoint_matrix", "case_comparison", "risk_trigger"],
    "business_model": ["thesis", "unit_economics", "case_comparison", "risk_trigger"],
    "competition": ["thesis", "competitive_positioning", "case_comparison", "risk_trigger"],
    "policy": ["policy_timeline", "mechanism_chain", "stakeholder_map", "risk_trigger"],
    "capital": ["thesis", "metric_reconciliation", "scenario_analysis", "risk_trigger"],
    "risk": ["thesis", "risk_trigger", "verification_checklist"],
    "entry_strategy": ["thesis", "case_comparison", "scenario_analysis", "verification_checklist"],
    "timeline": ["policy_timeline", "mechanism_chain", "risk_trigger"],
    "stakeholder_map": ["stakeholder_map", "mechanism_chain", "risk_trigger"],
    "transmission_chain": ["mechanism_chain", "stakeholder_map", "risk_trigger"],
    "beneficiary_loser": ["stakeholder_map", "value_chain_map", "risk_trigger"],
    "financial_quality": ["thesis", "unit_economics", "metric_reconciliation", "risk_trigger"],
}


EVIDENCE_ROLE_BLOCKS: Dict[str, str] = {
    "policy": "policy_timeline",
    "policy_original": "policy_timeline",
    "official_data": "evidence_matrix",
    "metric": "metric_reconciliation",
    "market_price": "metric_reconciliation",
    "financial_metric": "unit_economics",
    "filing": "unit_economics",
    "company_filing": "unit_economics",
    "case": "case_comparison",
    "customer_case": "customer_painpoint_matrix",
    "technology_product": "technology_maturity",
    "counter": "risk_trigger",
    "counter_evidence": "risk_trigger",
}


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _dedupe(values: Iterable[Any], *, limit: int = 8) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
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


def block_definition(block_type: str) -> Dict[str, Any]:
    return dict(BLOCK_REGISTRY.get(str(block_type or "").strip(), BLOCK_REGISTRY["thesis"]))


def valid_block_types() -> List[str]:
    return list(BLOCK_REGISTRY.keys())


def _evidence_roles_from_package(evidence_package: Dict[str, Any] | None = None) -> List[str]:
    roles: List[str] = []
    package = _as_dict(evidence_package)
    for collection in ("core_evidence", "supporting_evidence", "table_evidence", "clue_evidence"):
        for item in _as_list(package.get(collection)):
            if not isinstance(item, dict):
                continue
            roles.append(str(item.get("proof_role") or item.get("evidence_role") or item.get("role") or "").strip().lower())
            level = str(item.get("source_level") or "").strip().upper()
            if level in {"A", "B"}:
                roles.append("support")
    return _dedupe(roles, limit=20)


def select_blocks_for_chapter(
    chapter: Dict[str, Any],
    *,
    profile: Dict[str, Any] | None = None,
    evidence_package: Dict[str, Any] | None = None,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    chapter = _as_dict(chapter)
    profile = _as_dict(profile)
    layout_policy = _as_dict(chapter.get("layout_policy"))
    preferred = _as_list(layout_policy.get("preferred_blocks"))
    optional = _as_list(layout_policy.get("optional_blocks"))
    module_keys = _as_list(chapter.get("module_keys")) or _as_list(chapter.get("source_template_keys"))
    evidence_mix = _as_list(chapter.get("required_evidence_mix"))
    evidence_roles = _evidence_roles_from_package(evidence_package)

    candidates: List[str] = []
    candidates.extend(preferred)
    for module_key in module_keys:
        candidates.extend(MODULE_DEFAULT_BLOCKS.get(str(module_key), []))
    for role in [*evidence_mix, *evidence_roles]:
        mapped = EVIDENCE_ROLE_BLOCKS.get(str(role or "").strip().lower())
        if mapped:
            candidates.append(mapped)
    candidates.extend(optional)
    candidates.append("thesis")
    if any(str(role).lower() in {"counter", "counter_evidence", "risk"} for role in [*evidence_mix, *evidence_roles]):
        candidates.append("risk_trigger")
    candidates.append("verification_checklist")
    selected = _dedupe(candidates, limit=max(3, limit))

    blocks: List[Dict[str, Any]] = []
    for index, block_type in enumerate(selected, start=1):
        definition = block_definition(block_type)
        roles = _as_list(definition.get("roles"))
        blocks.append(
            {
                "block_id": f"{chapter.get('chapter_id') or 'chapter'}_b{index}",
                "block_type": block_type,
                "role": block_type,
                "title": definition.get("label") or block_type,
                "required_evidence_roles": roles,
                "min_evidence_refs": 1 if block_type in {"risk_trigger", "verification_checklist"} else 2,
                "renderer": definition.get("renderer") or "render_thesis_block",
                "public_render": True,
            }
        )
    return blocks


def block_types_from_layout(layout: Dict[str, Any]) -> List[str]:
    return [
        str(_as_dict(block).get("block_type") or "").strip()
        for block in _as_list(_as_dict(layout).get("blocks"))
        if str(_as_dict(block).get("block_type") or "").strip()
    ]
