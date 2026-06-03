from __future__ import annotations

import re
import os
from typing import Any, Dict, Iterable, List, Sequence

from .layout_claim_matcher import claim_supported_block_types, match_claims_to_blocks


BLOCK_REGISTRY: Dict[str, Dict[str, Any]] = {
    "thesis": {"label": "本章结论", "renderer": "render_thesis_block", "roles": ["support", "metric"]},
    "evidence_matrix": {"label": "事实依据", "renderer": "render_evidence_matrix", "roles": ["support", "metric"]},
    "signal_validation": {"label": "可验证信号", "renderer": "render_signal_validation", "roles": ["support", "case"]},
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
    "integrated_signal": {"label": "缁煎悎淇″彿", "renderer": "render_signal_validation", "roles": ["support", "case"]},
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
    "technology": "technology_maturity",
    "technology_product": "technology_maturity",
    "standard": "technology_maturity",
    "counter": "risk_trigger",
    "risk": "risk_trigger",
    "counter_evidence": "risk_trigger",
}


EVIDENCE_COLLECTIONS: Sequence[str] = (
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

PUBLIC_BODY_EVIDENCE_COLLECTIONS: Sequence[str] = (
    "core_evidence",
    "supporting_evidence",
    "metric_evidence",
    "case_evidence",
    "counter_evidence",
    "directional_evidence",
    "sample_evidence",
)

GENERIC_BLOCK_TYPES = {"thesis", "argument", "evidence_matrix"}

COMMERCIAL_RE = re.compile(
    r"营收|收入|利润|毛利|现金流|亏损|费用率|付费|收费|价格|续约|订单|采购|客单价|"
    r"revenue|profit|margin|pricing|price|paid|renewal|order|contract|procurement",
    re.I,
)
METRIC_RE = re.compile(r"指标|规模|增速|CAGR|TAM|SAM|SOM|市场|金额|亿元|亿美元|%|metric|growth|size", re.I)
CASE_RE = re.compile(r"案例|客户|采购|中标|订单|部署|落地|场景|试点|case|customer|deployment|procurement", re.I)
TECH_RE = re.compile(r"技术|标准|工具调用|权限|安全|可靠|部署|产品|专利|模型|agent|workflow|standard|patent|security|tool", re.I)
COUNTER_RE = re.compile(r"风险|反证|失败|约束|边界|合规|责任|安全|成本|counter|risk|failure|constraint", re.I)
COMPETITION_RE = re.compile(r"竞争|玩家|渠道|生态|份额|替代|平台|厂商|competition|player|channel|ecosystem", re.I)


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


def _evidence_items(evidence_package: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    package = _as_dict(evidence_package)
    return [
        item
        for collection in EVIDENCE_COLLECTIONS
        for item in _as_list(package.get(collection))
        if isinstance(item, dict)
    ]


def _public_evidence_items(evidence_package: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    package = _as_dict(evidence_package)
    for collection in PUBLIC_BODY_EVIDENCE_COLLECTIONS:
        for item in _as_list(package.get(collection)):
            if not isinstance(item, dict):
                continue
            quality = _as_dict(item.get("public_fact_quality"))
            card = _as_dict(item.get("public_fact_card") or quality.get("public_fact_card"))
            if quality and quality.get("eligible_for_report") is False:
                continue
            if (
                quality.get("eligible_for_report")
                or card.get("distilled_fact")
                or item.get("distilled_fact")
                or item.get("clean_fact")
                or item.get("fact")
            ):
                result.append(item)
    return result


def _item_blob(item: Dict[str, Any]) -> str:
    quality = _as_dict(item.get("public_fact_quality"))
    card = _as_dict(item.get("public_fact_card") or quality.get("public_fact_card"))
    fields = [
        item.get("proof_role"),
        item.get("evidence_role"),
        item.get("role"),
        item.get("intent"),
        item.get("source_type"),
        item.get("source_family"),
        item.get("metric"),
        item.get("indicator"),
        item.get("fact"),
        item.get("clean_fact"),
        item.get("distilled_fact"),
        item.get("title"),
        item.get("summary"),
        item.get("content"),
        card.get("fact_type"),
        card.get("analysis_variable"),
        " ".join(str(value or "") for value in _as_list(card.get("block_affinity"))),
    ]
    return " ".join(str(field or "") for field in fields)


def _has_explicit_block_affinity(items: Sequence[Dict[str, Any]]) -> bool:
    for item in items:
        quality = _as_dict(item.get("public_fact_quality"))
        card = _as_dict(item.get("public_fact_card") or quality.get("public_fact_card"))
        if _as_list(item.get("block_affinity")) or _as_list(card.get("block_affinity")):
            return True
    return False


def _collection_has_items(package: Dict[str, Any], *collections: str) -> bool:
    return any(_as_list(package.get(collection)) for collection in collections)


def _card_payload(item: Dict[str, Any]) -> Dict[str, Any]:
    quality = _as_dict(item.get("public_fact_quality"))
    card = _as_dict(item.get("public_fact_card") or quality.get("public_fact_card"))
    merged = {**item, **card}
    if quality and quality.get("eligible_for_report") is False:
        return {}
    fact = str(
        merged.get("distilled_fact")
        or merged.get("clean_fact")
        or merged.get("fact")
        or merged.get("summary")
        or ""
    ).strip()
    if not fact:
        return {}
    merged["distilled_fact"] = fact
    return merged


def _card_affinity(card: Dict[str, Any]) -> set[str]:
    return {
        str(item or "").strip()
        for item in _as_list(card.get("block_affinity"))
        if str(item or "").strip()
    }


def _card_blob(card: Dict[str, Any]) -> str:
    return " ".join(
        str(card.get(key) or "")
        for key in ("fact_type", "proof_role", "variable", "analysis_variable", "metric", "indicator", "distilled_fact")
    )


def _card_text(card: Dict[str, Any], *keys: str) -> str:
    return " ".join(str(card.get(key) or "").strip() for key in keys).strip()


def _source_ref(card: Dict[str, Any]) -> str:
    return str(card.get("source_ref") or card.get("citation_ref") or card.get("ref") or "").strip()


def _missing_fields_for_block(block_type: str, card: Dict[str, Any]) -> List[str]:
    missing: List[str] = []
    if not _source_ref(card):
        missing.append("source_ref")
    if block_type == "metric_reconciliation":
        if not _card_text(card, "subject", "company", "entity"):
            missing.append("subject")
        if not _card_text(card, "time_or_scope", "period", "scope", "date"):
            missing.append("time_or_scope")
        if not _card_text(card, "value", "display_value", "numeric_value") and not re.search(r"\d", str(card.get("distilled_fact") or "")):
            missing.append("value")
    elif block_type in {"case_comparison", "customer_painpoint_matrix", "unit_economics", "competitive_positioning"}:
        if not _card_text(card, "subject", "company", "entity"):
            missing.append("subject")
        if not (
            _card_text(card, "action_or_signal", "action", "signal")
            or _card_text(card, "variable", "analysis_variable")
            or re.search(r"deploy|customer|workflow|client|case|订单|客户|场景|部署|落地|采购|付费", str(card.get("distilled_fact") or ""), re.I)
        ):
            missing.append("case_or_action_signal")
    elif block_type == "technology_maturity":
        if not (
            _card_text(card, "variable", "analysis_variable")
            or re.search(r"tool|standard|security|permission|workflow|model|技术|标准|权限|安全|可靠|集成", str(card.get("distilled_fact") or ""), re.I)
        ):
            missing.append("technology_variable")
    elif block_type in {"risk_trigger", "verification_checklist", "scenario_analysis"}:
        if not (
            _card_text(card, "variable", "analysis_variable")
            or re.search(r"risk|counter|failure|cost|security|责任|风险|失败|成本|安全|反证", str(card.get("distilled_fact") or ""), re.I)
        ):
            missing.append("risk_variable")
    return missing


def can_render_block_from_evidence(block_type: str, evidence_package: Dict[str, Any] | None = None) -> Dict[str, Any]:
    package = _as_dict(evidence_package)
    block_type = str(block_type or "").strip()
    matches: List[Dict[str, Any]] = []
    matching_collections: List[str] = []
    skipped_reasons: List[Dict[str, Any]] = []
    for collection in PUBLIC_BODY_EVIDENCE_COLLECTIONS:
        for item in _as_list(package.get(collection)):
            if not isinstance(item, dict):
                continue
            card = _card_payload(item)
            if not card:
                continue
            fact_type = str(card.get("fact_type") or card.get("proof_role") or "").strip()
            affinity = _card_affinity(card)
            blob = _card_blob(card)
            matched = False
            if block_type == "metric_reconciliation":
                matched = collection == "metric_evidence" or fact_type == "metric"
            elif block_type == "unit_economics":
                matched = (
                    block_type in affinity
                    and fact_type in {"case", "customer_case", "metric", "financial_metric", "market_metric"}
                )
            elif block_type in {"case_comparison", "customer_painpoint_matrix"}:
                matched = collection == "case_evidence" or fact_type in {"case", "customer_case", "directional"} or block_type in affinity
            elif block_type == "competitive_positioning":
                matched = fact_type in {"case", "customer_case", "directional"} or block_type in affinity or COMPETITION_RE.search(blob)
            elif block_type == "technology_maturity":
                matched = fact_type in {"technology", "technology_product", "standard"} or block_type in affinity or TECH_RE.search(blob)
            elif block_type in {"risk_trigger", "verification_checklist", "scenario_analysis"}:
                matched = collection == "counter_evidence" or fact_type in {"counter", "risk"} or block_type in affinity
            elif block_type == "signal_validation":
                matched = bool(affinity)
            elif block_type in {"policy_timeline", "mechanism_chain", "stakeholder_map", "value_chain_map", "integrated_signal", "thesis"}:
                matched = True
            else:
                matched = True
            if matched:
                missing = _missing_fields_for_block(block_type, card)
                if missing:
                    if block_type == "metric_reconciliation" and {"subject", "time_or_scope"}.intersection(missing):
                        reason = "missing_metric_subject_or_scope"
                    elif "source_ref" in missing:
                        reason = "missing_source_ref"
                    else:
                        reason = "missing_required_fact_fields"
                    skipped_reasons.append(
                        {
                            "evidence_id": card.get("evidence_id") or card.get("ref"),
                            "collection": collection,
                            "reason": reason,
                            "missing_fields": missing,
                        }
                    )
                    continue
                matches.append(card)
                matching_collections.append(collection)
    matching_collections = _dedupe(matching_collections, limit=8)
    if matches:
        return {
            "can_render": True,
            "matching_fact_card_count": len(matches),
            "matching_collection": matching_collections[0] if matching_collections else "",
            "matching_collections": matching_collections,
            "reason": "matching_fact_card",
        }
    if _public_evidence_items(package) and block_type == "integrated_signal":
        return {
            "can_render": True,
            "matching_fact_card_count": len(_public_evidence_items(package)),
            "matching_collection": "public_evidence",
            "matching_collections": ["public_evidence"],
            "reason": "fallback_integrated_signal",
        }
    if skipped_reasons:
        first = skipped_reasons[0]
        return {
            "can_render": False,
            "matching_fact_card_count": 0,
            "matching_collection": "",
            "matching_collections": [],
            "reason": first.get("reason") or "missing_required_fact_fields",
            "missing_fields": first.get("missing_fields") or [],
            "skipped_fact_examples": skipped_reasons[:5],
        }
    return {
        "can_render": False,
        "matching_fact_card_count": 0,
        "matching_collection": "",
        "matching_collections": [],
        "missing_fields": [],
        "skipped_fact_examples": [],
        "reason": "missing_matching_fact_card",
    }


def _block_evidence_fit_score(block_type: str, evidence_package: Dict[str, Any] | None = None) -> int:
    package = _as_dict(evidence_package)
    items = _public_evidence_items(package)
    if not items:
        return 0
    blob = " ".join(_item_blob(item) for item in items)
    block_type = str(block_type or "").strip()
    feasibility = can_render_block_from_evidence(block_type, package)
    if feasibility.get("can_render"):
        if block_type == "metric_reconciliation":
            return 90
        if block_type == "unit_economics":
            return 90
        if block_type in {"case_comparison", "customer_painpoint_matrix", "competitive_positioning", "technology_maturity", "risk_trigger", "verification_checklist", "scenario_analysis"}:
            return 80
        if block_type in {"policy_timeline", "mechanism_chain", "stakeholder_map", "value_chain_map"}:
            return 55
        if block_type == "signal_validation":
            return 45
        if block_type == "integrated_signal":
            return 40
        if block_type == "thesis":
            return 25
    if block_type == "metric_reconciliation":
        return 0
    if block_type == "unit_economics":
        return 0
    if block_type in {"case_comparison", "customer_painpoint_matrix"}:
        return 0
    if block_type == "competitive_positioning":
        return 0
    if block_type == "technology_maturity":
        return 0
    if block_type in {"risk_trigger", "verification_checklist", "scenario_analysis"}:
        return 0
    if block_type in {"policy_timeline", "mechanism_chain", "stakeholder_map", "value_chain_map"}:
        return 55 if items else 0
    if block_type == "signal_validation":
        return 45 if _has_explicit_block_affinity(items) and _collection_has_items(package, "core_evidence", "supporting_evidence", "directional_evidence", "case_evidence") else 0
    if block_type == "evidence_matrix":
        return 35 if _collection_has_items(package, "core_evidence", "supporting_evidence", "directional_evidence") else 0
    if block_type == "integrated_signal":
        return 40 if items else 0
    if block_type == "thesis":
        return 25 if items else 0
    return 30 if items else 0


def _selection_reason(block_type: str, score: int) -> str:
    if score <= 0:
        return "no_matching_public_evidence"
    if block_type == "unit_economics":
        return "commercial_evidence_present"
    if block_type == "metric_reconciliation":
        return "metric_evidence_present"
    if block_type in {"case_comparison", "customer_painpoint_matrix"}:
        return "case_or_customer_evidence_present"
    if block_type == "competitive_positioning":
        return "competition_or_case_evidence_present"
    if block_type == "technology_maturity":
        return "technology_or_standard_evidence_present"
    if block_type in {"risk_trigger", "verification_checklist", "scenario_analysis"}:
        return "counter_or_risk_evidence_present"
    if block_type == "signal_validation":
        return "directional_or_supporting_evidence_present"
    if block_type == "evidence_matrix":
        return "generic_evidence_matrix_demoted"
    return "evidence_available"


def block_definition(block_type: str) -> Dict[str, Any]:
    return dict(BLOCK_REGISTRY.get(str(block_type or "").strip(), BLOCK_REGISTRY["thesis"]))


def valid_block_types() -> List[str]:
    return list(BLOCK_REGISTRY.keys())


def _evidence_roles_from_package(evidence_package: Dict[str, Any] | None = None) -> List[str]:
    roles: List[str] = []
    for item in _evidence_items(evidence_package):
        roles.append(str(item.get("proof_role") or item.get("evidence_role") or item.get("role") or "").strip().lower())
        quality = _as_dict(item.get("public_fact_quality"))
        card = _as_dict(item.get("public_fact_card") or quality.get("public_fact_card"))
        fact_type = str(card.get("fact_type") or "").strip().lower()
        if fact_type:
            roles.append(fact_type)
        for affinity in _as_list(card.get("block_affinity")):
            text = str(affinity or "").strip().lower()
            if text:
                roles.append(text)
        level = str(item.get("source_level") or "").strip().upper()
        if level in {"A", "B"}:
            roles.append("support")
    return _dedupe(roles, limit=20)


def select_blocks_for_chapter(
    chapter: Dict[str, Any],
    *,
    profile: Dict[str, Any] | None = None,
    evidence_package: Dict[str, Any] | None = None,
    claim_units_by_chapter: Dict[str, Sequence[Dict[str, Any]]] | None = None,
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
    chapter_id = str(chapter.get("chapter_id") or _as_dict(evidence_package).get("chapter_id") or "").strip()
    chapter_claims = list(_as_dict(claim_units_by_chapter).get(chapter_id, []))

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
    raw_selected = _dedupe(candidates, limit=max(8, limit * 3))
    scored: List[Dict[str, Any]] = []
    candidate_records: List[Dict[str, Any]] = []
    for block_type in raw_selected:
        if block_type not in BLOCK_REGISTRY:
            continue
        score = _block_evidence_fit_score(block_type, evidence_package)
        if block_type == "evidence_matrix" and score > 0:
            # Generic evidence matrices are useful internally, but the public
            # report should get a variable-specific signal block instead.
            signal_score = _block_evidence_fit_score("signal_validation", evidence_package)
            if signal_score > 0:
                block_type = "signal_validation"
                score = max(score, signal_score)
            else:
                score = 0
        record = {
            "block_type": block_type,
            "score": score,
            "selection_reason": _selection_reason(block_type, score),
        }
        if score > 0:
            scored.append(record)
        elif block_type not in {"thesis", "unit_economics"}:
            candidate_records.append({**record, "render_plan": "candidate", "candidate_reason": "missing_matching_fact_card"})
    public_items = _public_evidence_items(evidence_package)
    if public_items and scored and all(record.get("block_type") == "thesis" for record in scored):
        scored = []
    if any(record.get("block_type") != "thesis" and int(record.get("score") or 0) > 0 for record in scored):
        scored = [record for record in scored if record.get("block_type") != "thesis"]
    if public_items and not scored:
        scored.append({"block_type": "integrated_signal", "score": 40, "selection_reason": "fallback_integrated_signal"})
    existing_scored = {str(record.get("block_type") or "") for record in scored}
    for claim in chapter_claims:
        for block_type in claim_supported_block_types(claim):
            if block_type not in BLOCK_REGISTRY or block_type in existing_scored:
                continue
            existing_scored.add(block_type)
            scored.append(
                {
                    "block_type": block_type,
                    "score": 85 if block_type != "integrated_signal" else 60,
                    "selection_reason": "llm_claim_supported",
                    "matched_by_llm_claim": True,
                }
            )
            break

    selected_records: List[Dict[str, Any]] = []
    seen_blocks = set()
    try:
        max_must_blocks = max(1, min(3, int(os.getenv("REPORT_MAX_MUST_BLOCKS_PER_CHAPTER", "2"))))
    except ValueError:
        max_must_blocks = 2
    for record in sorted(scored, key=lambda item: (item["score"], item["block_type"] not in GENERIC_BLOCK_TYPES), reverse=True):
        block_type = record["block_type"]
        if block_type in seen_blocks:
            continue
        if block_type == "verification_checklist" and not any(str(role).lower() in {"counter", "counter_evidence", "risk"} for role in [*evidence_mix, *evidence_roles]):
            continue
        seen_blocks.add(block_type)
        selected_records.append(record)
        if len(selected_records) >= min(max_must_blocks, max(1, limit)):
            break

    blocks: List[Dict[str, Any]] = []
    for index, record in enumerate(selected_records, start=1):
        block_type = record["block_type"]
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
                "block_evidence_fit_score": record.get("score", 0),
                "selection_reason": record.get("selection_reason") or _selection_reason(block_type, int(record.get("score") or 0)),
                "render_plan": "must_render",
                "matched_by_llm_claim": bool(record.get("matched_by_llm_claim")),
            }
        )
    if chapter_claims and blocks:
        diagnostics = match_claims_to_blocks(chapter_id, chapter_claims, blocks)
        blocks[0]["claim_layout_match_diagnostics"] = {
            "llm_claim_to_block_match_count": diagnostics.get("matched_count", 0),
            "llm_claim_unmatched_count": diagnostics.get("unmatched_count", 0),
        }
    for record in candidate_records[: max(0, limit * 2)]:
        block_type = record["block_type"]
        definition = block_definition(block_type)
        roles = _as_list(definition.get("roles"))
        blocks.append(
            {
                "block_id": f"{chapter.get('chapter_id') or 'chapter'}_candidate_{len(blocks) + 1}",
                "block_type": block_type,
                "role": block_type,
                "title": definition.get("label") or block_type,
                "required_evidence_roles": roles,
                "min_evidence_refs": 1,
                "renderer": definition.get("renderer") or "render_thesis_block",
                "public_render": False,
                "block_evidence_fit_score": 0,
                "selection_reason": record.get("selection_reason") or "no_matching_public_evidence",
                "render_plan": "candidate",
                "candidate_reason": record.get("candidate_reason") or "missing_matching_fact_card",
            }
        )
    return blocks


def block_types_from_layout(layout: Dict[str, Any]) -> List[str]:
    return [
        str(_as_dict(block).get("block_type") or "").strip()
        for block in _as_list(_as_dict(layout).get("blocks"))
        if str(_as_dict(block).get("block_type") or "").strip()
    ]
