from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List


DEFAULT_PROFILE = "industry_scan_report"
GENERIC_PROFILE_HINTS = {"dynamic_research_report", "dynamic_research", "topic_report", "industry_scan_report"}


REPORT_PROFILES: Dict[str, Dict[str, Any]] = {
    "industry_scan_report": {
        "name": "industry_scan_report",
        "aliases": ["industry_scan", "industry_deep", "dynamic_research_report"],
        "keywords": ["行业", "市场", "机会", "增长", "产业", "格局", "空间"],
        "narrative_spines": ["definition_to_opportunity", "demand_supply_risk"],
        "candidate_modules": ["industry_definition", "market_size", "demand_driver", "industry_chain", "competition", "risk", "entry_strategy"],
        "required_evidence_roles": ["metric", "support", "counter"],
        "optional_evidence_roles": ["case", "technology_product", "company_filing"],
        "front_blocks": ["executive_summary", "key_judgments", "key_data"],
        "back_blocks": ["strategic_options", "risk_triggers", "verification_checklist", "appendix"],
        "max_body_chapters": 6,
        "min_body_chapters": 3,
        "module_order": ["industry_definition", "market_size", "demand_driver", "industry_chain", "competition", "technology", "risk", "entry_strategy"],
        "quality_contract": {
            "must_have_blocks": ["thesis", "evidence_matrix", "risk_trigger"],
            "must_have_evidence_roles": ["metric", "support", "counter"],
        },
    },
    "market_entry_report": {
        "name": "market_entry_report",
        "aliases": ["market_entry"],
        "keywords": ["进入", "切入", "立项", "市场进入", "落地", "BD", "渠道", "客户"],
        "narrative_spines": ["problem_to_entry_decision", "demand_to_channel_to_risk"],
        "candidate_modules": ["market_size", "demand_driver", "customer", "competition", "policy", "business_model", "entry_strategy", "risk"],
        "required_evidence_roles": ["metric", "case", "customer_case", "counter"],
        "optional_evidence_roles": ["policy_original", "financial_metric"],
        "front_blocks": ["executive_summary", "entry_decision_snapshot", "key_data"],
        "back_blocks": ["entry_recommendation", "risk_triggers", "verification_checklist", "appendix"],
        "max_body_chapters": 6,
        "min_body_chapters": 4,
        "module_order": ["market_size", "demand_driver", "customer", "competition", "policy", "business_model", "entry_strategy", "risk"],
        "quality_contract": {
            "must_have_blocks": ["thesis", "evidence_matrix", "case_comparison", "risk_trigger"],
            "must_have_evidence_roles": ["metric", "case", "counter"],
        },
    },
    "company_due_diligence_report": {
        "name": "company_due_diligence_report",
        "aliases": ["company_due_diligence", "company_dd", "due_diligence"],
        "keywords": ["公司", "尽调", "是否值得投", "投资尽调", "财务", "客户结构", "商业模式", "法务"],
        "narrative_spines": ["investment_thesis_to_risk"],
        "candidate_modules": ["business_model", "customer", "competition", "financial_quality", "policy", "risk", "entry_strategy"],
        "required_evidence_roles": ["filing", "customer_case", "financial_metric", "counter"],
        "optional_evidence_roles": ["case", "market_research"],
        "front_blocks": ["deal_snapshot", "investment_conclusion"],
        "back_blocks": ["red_flags", "dd_checklist", "appendix"],
        "max_body_chapters": 7,
        "min_body_chapters": 4,
        "module_order": ["business_model", "customer", "financial_quality", "competition", "policy", "risk", "entry_strategy"],
        "quality_contract": {
            "must_have_blocks": ["thesis", "unit_economics", "case_comparison", "risk_trigger"],
            "must_have_evidence_roles": ["filing", "financial_metric", "customer_case", "counter"],
        },
    },
    "competitor_analysis_report": {
        "name": "competitor_analysis_report",
        "aliases": ["competitor_analysis", "competitive_analysis"],
        "keywords": ["竞品", "对手", "相比", "竞争", "份额", "定位", "渠道差距", "品牌差距"],
        "narrative_spines": ["positioning_to_gap_to_action"],
        "candidate_modules": ["competition", "customer", "business_model", "market_size", "technology", "risk", "entry_strategy"],
        "required_evidence_roles": ["case", "metric", "company_filing", "counter"],
        "optional_evidence_roles": ["customer_case", "market_research"],
        "front_blocks": ["executive_summary", "competitive_snapshot"],
        "back_blocks": ["strategic_options", "risk_triggers", "appendix"],
        "max_body_chapters": 6,
        "min_body_chapters": 4,
        "module_order": ["competition", "customer", "business_model", "technology", "market_size", "risk", "entry_strategy"],
        "quality_contract": {
            "must_have_blocks": ["competitive_positioning", "case_comparison", "risk_trigger"],
            "must_have_evidence_roles": ["case", "metric", "counter"],
        },
    },
    "policy_impact_report": {
        "name": "policy_impact_report",
        "aliases": ["policy_impact", "policy_report"],
        "keywords": ["政策", "监管", "法规", "补贴", "出口管制", "制裁", "关税", "影响"],
        "narrative_spines": ["policy_original_to_transmission"],
        "candidate_modules": ["policy", "stakeholder_map", "transmission_chain", "industry_chain", "beneficiary_loser", "timeline", "risk"],
        "required_evidence_roles": ["policy_original", "official_data", "counter"],
        "optional_evidence_roles": ["case", "market_research", "company_filing"],
        "front_blocks": ["policy_summary", "impact_judgment"],
        "back_blocks": ["execution_risks", "monitoring_indicators", "appendix"],
        "max_body_chapters": 6,
        "min_body_chapters": 3,
        "module_order": ["policy", "timeline", "transmission_chain", "stakeholder_map", "industry_chain", "beneficiary_loser", "risk"],
        "quality_contract": {
            "must_have_blocks": ["policy_timeline", "mechanism_chain", "stakeholder_map", "risk_trigger"],
            "must_have_evidence_roles": ["policy_original", "official_data", "counter"],
        },
    },
    "consumer_market_report": {
        "name": "consumer_market_report",
        "aliases": ["consumer_market", "consumer_research"],
        "keywords": ["消费", "用户", "品牌", "渠道", "复购", "心智", "客群", "品类"],
        "narrative_spines": ["consumer_need_to_channel_to_brand"],
        "candidate_modules": ["market_size", "demand_driver", "customer", "competition", "business_model", "risk", "entry_strategy"],
        "required_evidence_roles": ["metric", "customer_case", "case", "counter"],
        "optional_evidence_roles": ["market_research", "company_filing"],
        "front_blocks": ["executive_summary", "consumer_opportunity_snapshot"],
        "back_blocks": ["product_opportunity", "risk_triggers", "verification_checklist", "appendix"],
        "max_body_chapters": 6,
        "min_body_chapters": 4,
        "module_order": ["market_size", "customer", "demand_driver", "competition", "business_model", "risk", "entry_strategy"],
        "quality_contract": {
            "must_have_blocks": ["customer_painpoint_matrix", "competitive_positioning", "risk_trigger"],
            "must_have_evidence_roles": ["metric", "customer_case", "counter"],
        },
    },
    "supply_chain_report": {
        "name": "supply_chain_report",
        "aliases": ["supply_chain_research"],
        "keywords": ["供应链", "产业链", "瓶颈", "产能", "成本", "重构", "友岸", "国产替代", "物流链路"],
        "narrative_spines": ["bottleneck_to_resilience_to_opportunity"],
        "candidate_modules": ["industry_chain", "policy", "technology", "competition", "customer", "risk", "entry_strategy"],
        "required_evidence_roles": ["metric", "case", "company_filing", "counter"],
        "optional_evidence_roles": ["policy_original", "technology_product"],
        "front_blocks": ["executive_summary", "supply_chain_snapshot", "key_data"],
        "back_blocks": ["resilience_options", "risk_triggers", "monitoring_indicators", "appendix"],
        "max_body_chapters": 6,
        "min_body_chapters": 4,
        "module_order": ["industry_chain", "policy", "technology", "competition", "customer", "risk", "entry_strategy"],
        "quality_contract": {
            "must_have_blocks": ["value_chain_map", "mechanism_chain", "risk_trigger"],
            "must_have_evidence_roles": ["metric", "case", "counter"],
        },
    },
    "technology_trend_report": {
        "name": "technology_trend_report",
        "aliases": ["technology_trend", "product_research_report"],
        "keywords": ["技术", "路线", "趋势", "产品", "成熟度", "替代", "AI", "Agent", "研发"],
        "narrative_spines": ["technology_maturity_to_adoption"],
        "candidate_modules": ["technology", "customer", "industry_chain", "business_model", "competition", "risk", "entry_strategy"],
        "required_evidence_roles": ["technology_product", "case", "counter"],
        "optional_evidence_roles": ["metric", "market_research"],
        "front_blocks": ["executive_summary", "technology_readiness_snapshot"],
        "back_blocks": ["adoption_path", "risk_triggers", "verification_checklist", "appendix"],
        "max_body_chapters": 6,
        "min_body_chapters": 3,
        "module_order": ["technology", "customer", "industry_chain", "business_model", "competition", "risk", "entry_strategy"],
        "quality_contract": {
            "must_have_blocks": ["technology_maturity", "case_comparison", "risk_trigger"],
            "must_have_evidence_roles": ["technology_product", "case", "counter"],
        },
    },
    "investment_memo": {
        "name": "investment_memo",
        "aliases": ["investment", "investment_memo_report"],
        "keywords": ["投资", "买入", "估值", "机会", "值得", "memo", "配置"],
        "narrative_spines": ["thesis_to_variant_view_to_risk"],
        "candidate_modules": ["market_size", "demand_driver", "competition", "business_model", "capital", "risk", "entry_strategy"],
        "required_evidence_roles": ["metric", "filing", "financial_metric", "counter"],
        "optional_evidence_roles": ["case", "market_research"],
        "front_blocks": ["investment_conclusion", "key_judgments", "key_data"],
        "back_blocks": ["risk_triggers", "monitoring_indicators", "appendix"],
        "max_body_chapters": 7,
        "min_body_chapters": 4,
        "module_order": ["market_size", "demand_driver", "competition", "business_model", "capital", "risk", "entry_strategy"],
        "quality_contract": {
            "must_have_blocks": ["thesis", "metric_reconciliation", "scenario_analysis", "risk_trigger"],
            "must_have_evidence_roles": ["metric", "financial_metric", "counter"],
        },
    },
    "briefing_note": {
        "name": "briefing_note",
        "aliases": ["briefing", "brief"],
        "keywords": ["简报", "briefing", "概览", "速览", "快报"],
        "narrative_spines": ["facts_to_judgment_to_watchlist"],
        "candidate_modules": ["executive_summary", "policy", "market_size", "demand_driver", "risk"],
        "required_evidence_roles": ["support", "counter"],
        "optional_evidence_roles": ["metric", "case"],
        "front_blocks": ["briefing_summary", "key_judgments"],
        "back_blocks": ["monitoring_indicators", "appendix"],
        "max_body_chapters": 4,
        "min_body_chapters": 2,
        "module_order": ["policy", "market_size", "demand_driver", "industry_chain", "risk"],
        "quality_contract": {
            "must_have_blocks": ["thesis", "verification_checklist"],
            "must_have_evidence_roles": ["support", "counter"],
        },
    },
}


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _text_blob(*values: Any) -> str:
    parts: List[str] = []
    for value in values:
        if isinstance(value, dict):
            parts.extend(str(v or "") for v in value.values())
        elif isinstance(value, list):
            parts.extend(str(item or "") for item in value)
        else:
            parts.append(str(value or ""))
    return " ".join(parts)


def profile_names() -> List[str]:
    return list(REPORT_PROFILES.keys())


def get_report_profile(name: str | None = None) -> Dict[str, Any]:
    key = str(name or "").strip()
    for profile_name, profile in REPORT_PROFILES.items():
        if key == profile_name or key in _as_list(profile.get("aliases")):
            return dict(profile)
    return dict(REPORT_PROFILES[DEFAULT_PROFILE])


def select_report_profile(query: str, research_plan: Dict[str, Any] | None = None, report_plan: Dict[str, Any] | None = None) -> Dict[str, Any]:
    plan = _as_dict(research_plan)
    report_plan = _as_dict(report_plan)
    explicit = str(
        plan.get("report_profile")
        or plan.get("report_family")
        or plan.get("research_type")
        or report_plan.get("report_family")
        or report_plan.get("report_type")
        or ""
    ).strip()
    explicit_profile = get_report_profile(explicit)
    if explicit and explicit_profile.get("name") != DEFAULT_PROFILE:
        return explicit_profile

    blob = _text_blob(
        query,
        plan.get("research_type"),
        plan.get("report_family"),
        plan.get("decision_context"),
        plan.get("research_object"),
        [item.get("statement") for item in _as_list(plan.get("hypotheses")) if isinstance(item, dict)],
        [item.get("question") for item in _as_list(plan.get("evidence_goals")) if isinstance(item, dict)],
    )
    query_text = str(query or "")
    if (
        re.search(r"AI|人工智能|大模型|生成式AI|AIGC", query_text, flags=re.I)
        and re.search(r"行业|产业|市场|焦虑|机遇|机会|发展|格局", query_text)
        and not re.search(r"AI\s*Agent|技术路线|产品形态|研发|成熟度|替代路径", query_text, flags=re.I)
    ):
        return dict(REPORT_PROFILES["industry_scan_report"])
    priority_rules = [
        ("company_due_diligence_report", r"尽调|due\s*diligence|是否值得.*投|投资尽调"),
        ("policy_impact_report", r"政策|监管|法规|补贴|出口管制|制裁|关税"),
        ("supply_chain_report", r"供应链|产业链|物流链路|瓶颈|重构|友岸|国产替代"),
        ("competitor_analysis_report", r"竞品|竞争对手|和竞品相比|渠道.*心智|心智.*差距"),
        ("consumer_market_report", r"消费|用户|品牌|品类|宠物食品|复购|客群"),
        ("technology_trend_report", r"技术趋势|技术路线|AI\s*Agent|产品形态|成熟度|替代路径"),
        ("investment_memo", r"投资机会|是否值得|买入|估值|配置机会"),
        ("market_entry_report", r"市场进入|进入机会|切入|立项|销售\s*BD|BD"),
    ]
    for profile_name, pattern in priority_rules:
        if re.search(pattern, blob, flags=re.I):
            return dict(REPORT_PROFILES[profile_name])
    scores: Dict[str, int] = {}
    for name, profile in REPORT_PROFILES.items():
        score = 0
        for keyword in _as_list(profile.get("keywords")):
            if keyword and re.search(re.escape(str(keyword)), blob, flags=re.I):
                score += 3
        for module in _as_list(profile.get("candidate_modules")):
            if module and re.search(re.escape(str(module)), blob, flags=re.I):
                score += 1
        if explicit and explicit not in GENERIC_PROFILE_HINTS and (name in explicit or explicit in _as_list(profile.get("aliases"))):
            score += 8
        scores[name] = score
    selected = max(scores, key=lambda key: scores[key])
    if scores[selected] <= 0:
        selected = DEFAULT_PROFILE
    return dict(REPORT_PROFILES[selected])


def default_report_shell(profile: Dict[str, Any] | str) -> Dict[str, Any]:
    profile = get_report_profile(profile) if isinstance(profile, str) else dict(profile)
    return {
        "front_blocks": list(_as_list(profile.get("front_blocks"))),
        "body_policy": "selected_chapters",
        "back_blocks": list(_as_list(profile.get("back_blocks"))),
    }


def quality_contract_for_profile(profile: Dict[str, Any] | str) -> Dict[str, Any]:
    profile = get_report_profile(profile) if isinstance(profile, str) else dict(profile)
    return dict(_as_dict(profile.get("quality_contract")))


def dedupe_keep_order(values: Iterable[Any], *, limit: int = 20) -> List[str]:
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
