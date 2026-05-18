from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Sequence


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _compact(value: Any, max_chars: int = 260) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "..."


def _walk_text(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _walk_text(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_text(item)
    elif value is not None:
        yield str(value)


def _blob(*values: Any) -> str:
    return " ".join(_walk_text([*values])).lower()


def _item_text(item: Dict[str, Any]) -> str:
    source = _as_dict(item.get("source"))
    raw = _as_dict(item.get("raw"))
    fields = [
        item.get("fact"),
        item.get("claim"),
        item.get("metric"),
        item.get("value"),
        item.get("period"),
        item.get("scope"),
        item.get("subject"),
        item.get("company"),
        item.get("entity"),
        item.get("source_family"),
        item.get("proof_role"),
        item.get("evidence_type"),
        source.get("title"),
        source.get("url"),
        source.get("publisher"),
        raw,
    ]
    return _blob(*fields)


def _metric_text(metric: Dict[str, Any]) -> str:
    return _blob(
        metric.get("metric_name"),
        metric.get("metric"),
        metric.get("value"),
        metric.get("unit"),
        metric.get("scope"),
        metric.get("period"),
        metric.get("source_title"),
        metric.get("evidence_ref"),
        metric.get("missing_fields"),
    )


def _source_level(value: Dict[str, Any]) -> str:
    source = _as_dict(value.get("source"))
    return str(
        value.get("source_level")
        or value.get("source_level_ab")
        or source.get("source_level")
        or source.get("credibility")
        or ""
    ).strip().upper()


def _source_family(value: Dict[str, Any]) -> str:
    source = _as_dict(value.get("source"))
    text = _blob(
        value.get("source_family"),
        value.get("source_type"),
        value.get("evidence_type"),
        value.get("proof_role"),
        source.get("source_type"),
        source.get("type"),
        source.get("title"),
        source.get("url"),
    )
    if any(term in text for term in ["official", "government", "regulator", "stats", "association", "whitepaper", "gov", "政府", "监管", "协会", "白皮书"]):
        return "official"
    if any(term in text for term in ["filing", "annual_report", "financial_report", "10-k", "8-k", "prospectus", "exchange", "公告", "财报", "年报", "交易所"]):
        return "filing"
    if any(term in text for term in ["research", "brokerage", "consulting", "industry_report", "研报", "研究报告", "券商"]):
        return "research_report"
    if any(term in text for term in ["case", "customer", "procurement", "order", "contract", "tender", "客户", "采购", "订单", "合同"]):
        return "case"
    if any(term in text for term in ["news", "media", "article", "新闻", "媒体"]):
        return "news"
    return "unknown"


def _strong_source_match(value: Dict[str, Any]) -> bool:
    level = _source_level(value)
    family = _source_family(value)
    allowed_use = str(value.get("allowed_use") or "").strip().lower()
    if level in {"A", "B"}:
        return True
    if family in {"official", "filing", "research_report"} and level not in {"C", "D"}:
        return True
    if allowed_use in {"core_claim", "supporting"} and level not in {"C", "D"}:
        return True
    return False


def _strong_metric_match(metric: Dict[str, Any]) -> bool:
    if _as_list(metric.get("missing_fields")):
        return False
    level = _source_level(metric)
    family = _source_family(metric)
    if level in {"A", "B"}:
        return True
    if family in {"official", "filing", "research_report"} and level not in {"C", "D"}:
        return True
    return False


def _contains_any(text: str, terms: Sequence[str]) -> bool:
    lowered = text.lower()
    return any(str(term or "").lower() in lowered for term in terms if str(term or "").strip())


def _matches_groups(text: str, groups: Sequence[Sequence[str]]) -> bool:
    if not groups:
        return False
    return all(_contains_any(text, group) for group in groups if group)


def _proof_applies(
    proof: Dict[str, Any],
    hypothesis: Optional[Dict[str, Any]],
    context: Optional[Any] = None,
) -> bool:
    terms = [str(item) for item in _as_list(proof.get("applies_to_terms")) if str(item or "").strip()]
    if not terms:
        return True
    if hypothesis:
        return _contains_any(_blob(hypothesis), terms)
    if context is not None:
        return _contains_any(_blob(context), terms)
    return True


PROOF_PROFILE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "tech_geopolitics": {
        "profile_id": "tech_geopolitics",
        "label": "Tech geopolitics and supply-chain reconnect",
        "trigger_terms": [
            "musk",
            "elon",
            "马斯克",
            "cook",
            "库克",
            "jensen",
            "huang",
            "黄仁勋",
            "nvidia",
            "英伟达",
            "apple",
            "苹果",
            "tesla",
            "特斯拉",
            "中美",
            "科技新均衡",
            "出口管制",
            "export control",
            "h20",
            "h200",
            "blackwell",
        ],
        "mandatory_proofs": [
            {
                "proof_id": "nvidia_export_control_status",
                "label": "NVIDIA export-control and China license status",
                "severity": "high",
                "required": True,
                "applies_to_terms": ["nvidia", "英伟达", "黄仁勋", "算力", "ai芯片", "ai 芯片", "出口管制"],
                "match_groups": [
                    ["nvidia", "英伟达", "黄仁勋"],
                    ["h20", "h200", "blackwell", "export control", "license", "bis", "8-k", "10-k", "出口管制", "出口许可", "许可证"],
                ],
                "query": "NVIDIA H20 H200 Blackwell China export license BIS 8-K 10-K data center revenue",
                "lane_targets": ["filing_company", "official_data", "news_event"],
                "source_priority": ["filing", "official", "tier1_media"],
                "proof_role": "metric",
                "evidence_type": "filing",
            },
            {
                "proof_id": "nvidia_china_business_data",
                "label": "NVIDIA China revenue, shipment, impairment or order data",
                "severity": "high",
                "required": True,
                "applies_to_terms": ["nvidia", "英伟达", "黄仁勋", "算力", "ai芯片", "ai 芯片"],
                "match_groups": [
                    ["nvidia", "英伟达"],
                    ["china", "中国", "data center", "数据中心", "h20", "revenue", "收入", "charge", "费用", "减值", "shipment", "出货", "order", "订单"],
                ],
                "query": "NVIDIA China data center revenue H20 charge order shipment official filing",
                "lane_targets": ["filing_company", "market_research"],
                "source_priority": ["filing", "official", "research_report"],
                "proof_role": "metric",
                "evidence_type": "metric",
            },
            {
                "proof_id": "apple_china_sales_supply_chain",
                "label": "Apple Greater China sales and China supply-chain dependence",
                "severity": "high",
                "required": True,
                "applies_to_terms": ["apple", "苹果", "库克", "消费电子", "供应链"],
                "match_groups": [
                    ["apple", "苹果", "库克"],
                    ["greater china", "大中华", "china", "中国", "net sales", "净销售", "10-k", "annual report", "供应链", "supplier"],
                ],
                "query": "Apple 2025 10-K Greater China net sales supply chain China suppliers",
                "lane_targets": ["filing_company", "official_data", "market_research"],
                "source_priority": ["filing", "official", "research_report"],
                "proof_role": "metric",
                "evidence_type": "filing",
            },
            {
                "proof_id": "tesla_china_factory_sales",
                "label": "Tesla Shanghai production, export, sales and margin signal",
                "severity": "high",
                "required": True,
                "applies_to_terms": ["tesla", "特斯拉", "马斯克", "上海工厂", "汽车", "新能源"],
                "match_groups": [
                    ["tesla", "特斯拉", "马斯克"],
                    ["shanghai", "上海", "china", "中国", "delivery", "deliveries", "production", "产量", "销量", "出口", "10-k", "annual report", "gross margin", "毛利"],
                ],
                "query": "Tesla 2025 10-K Shanghai factory China deliveries export gross margin",
                "lane_targets": ["filing_company", "official_data", "market_research"],
                "source_priority": ["filing", "official", "research_report"],
                "proof_role": "metric",
                "evidence_type": "metric",
            },
            {
                "proof_id": "china_cloud_capex_demand",
                "label": "China cloud and operator AI capex or procurement demand",
                "severity": "medium",
                "required": True,
                "applies_to_terms": ["算力", "云厂商", "云", "数据中心", "ai芯片", "ai 芯片", "nvidia", "英伟达"],
                "match_groups": [
                    ["capex", "资本开支", "采购", "订单", "算力", "数据中心", "云厂商", "阿里", "腾讯", "百度", "字节", "运营商"],
                    ["gpu", "ai", "算力", "服务器", "芯片", "采购", "订单"],
                ],
                "query": "China cloud AI capex GPU procurement Alibaba Tencent Baidu ByteDance operators",
                "lane_targets": ["filing_company", "market_research", "customer_case"],
                "source_priority": ["filing", "research_report", "official"],
                "proof_role": "metric",
                "evidence_type": "metric",
            },
            {
                "proof_id": "domestic_compute_substitution",
                "label": "Domestic AI compute substitution progress",
                "severity": "medium",
                "required": True,
                "applies_to_terms": ["国产替代", "算力", "ai芯片", "ai 芯片", "芯片", "半导体", "nvidia", "英伟达"],
                "match_groups": [
                    ["昇腾", "华为", "寒武纪", "海光", "壁仞", "国产替代", "domestic", "localization"],
                    ["订单", "出货", "适配", "采购", "服务器", "算力", "ai芯片", "ai 芯片"],
                ],
                "query": "China domestic AI chip substitution Huawei Ascend Cambricon Hygon orders shipments",
                "lane_targets": ["filing_company", "market_research", "customer_case"],
                "source_priority": ["filing", "research_report", "official"],
                "proof_role": "counter",
                "evidence_type": "counter",
            },
            {
                "proof_id": "policy_counter_trigger",
                "label": "Policy trigger or counter-signal for limited reconnect",
                "severity": "medium",
                "required": True,
                "applies_to_terms": ["中美", "政策", "出口管制", "再连接", "科技新均衡", "贸易", "监管"],
                "match_groups": [
                    ["bis", "commerce department", "商务部", "出口管制", "许可", "关税", "监管", "数据合规", "policy", "政策"],
                    ["触发", "限制", "license", "rule", "规则", "风险", "counter", "反向"],
                ],
                "query": "US Commerce BIS China AI chip export control license tariff policy trigger",
                "lane_targets": ["official_data", "news_event", "filing_company"],
                "source_priority": ["official", "filing", "tier1_media"],
                "proof_role": "counter",
                "evidence_type": "counter",
            },
        ],
        "required_tables": [
            "company_china_exposure_matrix",
            "industry_impact_matrix",
            "investment_mapping",
            "counter_trigger_monitor",
        ],
    },
    "generic_industry": {
        "profile_id": "generic_industry",
        "label": "Generic multi-industry research proof pack",
        "trigger_terms": [],
        "mandatory_proofs": [
            {
                "proof_id": "official_or_filing_baseline",
                "label": "Official, filing, association or authoritative research baseline",
                "severity": "high",
                "required": True,
                "match_groups": [["official", "filing", "annual report", "10-k", "公告", "财报", "年报", "协会", "政府", "监管", "白皮书", "研究报告"]],
                "query": "official filing annual report association industry data market size",
                "lane_targets": ["official_data", "filing_company", "market_research"],
                "source_priority": ["official", "filing", "research_report"],
                "proof_role": "metric",
                "evidence_type": "data",
            },
            {
                "proof_id": "hard_metric_baseline",
                "label": "Hard data for price, order, capacity, revenue, shipment or margin",
                "severity": "high",
                "required": True,
                "match_groups": [["price", "价格", "order", "订单", "capacity", "产能", "revenue", "收入", "shipment", "出货", "margin", "毛利", "capex", "资本开支", "market size", "市场规模"]],
                "query": "price order capacity revenue shipment margin capex market size official data",
                "lane_targets": ["official_data", "filing_company", "market_research"],
                "source_priority": ["filing", "official", "research_report"],
                "proof_role": "metric",
                "evidence_type": "metric",
            },
            {
                "proof_id": "counter_trigger_baseline",
                "label": "Counter-signal, risk trigger or falsification condition",
                "severity": "medium",
                "required": True,
                "match_groups": [["risk", "风险", "counter", "反向", "trigger", "触发", "decline", "下降", "delay", "延迟", "监管", "限制", "失败"]],
                "query": "counter evidence risk trigger decline delay regulatory restriction",
                "lane_targets": ["news_event", "filing_company", "market_research"],
                "source_priority": ["official", "filing", "research_report"],
                "proof_role": "counter",
                "evidence_type": "counter",
            },
        ],
        "required_tables": [
            "key_metric_matrix",
            "industry_impact_matrix",
            "investment_mapping",
            "counter_trigger_monitor",
        ],
    },
}


def select_research_proof_profile(
    *,
    query: str = "",
    research_plan: Optional[Dict[str, Any]] = None,
    report_blueprint: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    corpus = _blob(query, research_plan or {}, report_blueprint or {})
    best_profile_id = "generic_industry"
    best_score = 0
    for profile_id, profile in PROOF_PROFILE_REGISTRY.items():
        if profile_id == "generic_industry":
            continue
        score = sum(1 for term in _as_list(profile.get("trigger_terms")) if str(term or "").lower() in corpus)
        if score > best_score:
            best_score = score
            best_profile_id = profile_id
    profile = deepcopy(PROOF_PROFILE_REGISTRY[best_profile_id])
    profile["selection_score"] = best_score
    profile["selected_by"] = "trigger_terms" if best_score else "fallback"
    return profile


def mandatory_proof_checks(
    profile: Dict[str, Any],
    evidence_items: Sequence[Dict[str, Any]],
    metric_rows: Optional[Sequence[Dict[str, Any]]] = None,
    *,
    hypothesis: Optional[Dict[str, Any]] = None,
    context: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    usable_evidence = [item for item in evidence_items if isinstance(item, dict) and not item.get("excluded")]
    usable_metrics = [item for item in list(metric_rows or []) if isinstance(item, dict)]
    for proof in _as_list(profile.get("mandatory_proofs")):
        proof = _as_dict(proof)
        if not _proof_applies(proof, hypothesis, context):
            continue
        matched_refs: List[str] = []
        matched_metric_refs: List[str] = []
        weak_refs: List[str] = []
        match_groups = _as_list(proof.get("match_groups"))
        for item in usable_evidence:
            text = _item_text(item)
            if _matches_groups(text, match_groups):
                ref = str(item.get("ref") or item.get("evidence_id") or item.get("source_ref") or "").strip()
                ref = ref or _compact(_as_dict(item.get("source")).get("title"), 80)
                if _strong_source_match(item):
                    matched_refs.append(ref)
                else:
                    weak_refs.append(ref)
        for metric in usable_metrics:
            text = _metric_text(metric)
            if _matches_groups(text, match_groups):
                ref = str(metric.get("evidence_ref") or metric.get("source_ref") or "").strip()
                ref = ref or _compact(metric.get("metric_name") or metric.get("metric"), 80)
                if _strong_metric_match(metric):
                    matched_metric_refs.append(ref)
                else:
                    weak_refs.append(ref)
        refs = [ref for ref in dict.fromkeys([*matched_refs, *matched_metric_refs]) if ref]
        weak_refs = [ref for ref in dict.fromkeys(weak_refs) if ref and ref not in refs]
        status = "found" if refs else ("weak_found" if weak_refs else "missing")
        checks.append(
            {
                "proof_id": proof.get("proof_id"),
                "label": proof.get("label"),
                "status": status,
                "severity": proof.get("severity") or "medium",
                "required": bool(proof.get("required", True)),
                "matched_refs": refs[:8],
                "weak_matched_refs": weak_refs[:8],
                "query": proof.get("query"),
                "lane_targets": _as_list(proof.get("lane_targets")),
                "source_priority": _as_list(proof.get("source_priority")),
                "proof_role": proof.get("proof_role") or "support",
                "evidence_type": proof.get("evidence_type") or "data",
                "blocking_gap": "mandatory_proof_missing",
            }
        )
    return checks


def missing_mandatory_proofs(checks: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        item
        for item in checks
        if isinstance(item, dict)
        and item.get("status") in {"missing", "weak_found"}
        and item.get("required")
        and str(item.get("severity") or "").lower() in {"high", "medium"}
    ]


def build_mandatory_proof_followups(
    checks: Sequence[Dict[str, Any]],
    *,
    hypothesis: Optional[Dict[str, Any]] = None,
    profile: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    hypothesis_id = _as_dict(hypothesis).get("hypothesis_id")
    hypothesis_statement = _compact(
        _as_dict(hypothesis).get("hypothesis_statement") or _as_dict(hypothesis).get("statement") or "",
        120,
    )
    for check in missing_mandatory_proofs(checks):
        query = _compact(check.get("query") or check.get("label"), 220)
        if not query:
            continue
        tasks.append(
            {
                "query": query,
                "agent": "iqs",
                "targets_gap": check.get("label") or hypothesis_statement,
                "dimension_name": check.get("label"),
                "evidence_goal": check.get("label"),
                "hypothesis_id": hypothesis_id,
                "hypothesis_statement": hypothesis_statement,
                "proof_profile_id": _as_dict(profile).get("profile_id"),
                "mandatory_proof_id": check.get("proof_id"),
                "proof_role": check.get("proof_role") or "support",
                "counter_evidence": str(check.get("proof_role") or "").lower() == "counter",
                "evidence_type": check.get("evidence_type") or "data",
                "lane_targets": _as_list(check.get("lane_targets")) or ["official_data", "filing_company", "market_research"],
                "source_priority": _as_list(check.get("source_priority")) or ["official", "filing", "research_report"],
                "blocking_gaps": ["mandatory_proof_missing"],
                "missing_mandatory_proofs": [check],
                "priority": 0 if str(check.get("severity") or "").lower() == "high" else 1,
            }
        )
    return tasks


def research_maturity(
    *,
    report_text: str = "",
    coverage_rows: Sequence[Dict[str, Any]] = (),
    table_count: int = 0,
    metric_rows: int = 0,
    complete_metric_rows: int = 0,
) -> Dict[str, Any]:
    checks = [
        check
        for row in coverage_rows
        if isinstance(row, dict)
        for check in _as_list(row.get("mandatory_proof_checks"))
        if isinstance(check, dict)
    ]
    total = len(checks)
    found = len([check for check in checks if check.get("status") == "found"])
    missing = missing_mandatory_proofs(checks)
    text = str(report_text or "")
    investment_mapping_present = bool(
        re.search(r"A股|港股|美股|受益链条|投资优先级|投资启示|标的|催化剂|触发器|investment|beneficiary|catalyst", text, re.I)
    )
    trigger_monitor_present = bool(re.search(r"触发器|跟踪指标|监控表|反向信号|风险触发|watchlist|monitor", text, re.I))
    coverage_ratio = (found / total) if total else 0.0
    score = 35
    score += int(30 * coverage_ratio)
    score += 10 if complete_metric_rows else (5 if metric_rows else 0)
    score += 10 if table_count >= 2 else (5 if table_count else 0)
    score += 10 if investment_mapping_present else 0
    score += 5 if trigger_monitor_present else 0
    score = max(0, min(100, score))
    if score >= 82 and not missing:
        level = "professional_research"
    elif score >= 70:
        level = "investment_usable_draft"
    elif score >= 55:
        level = "evidence_draft"
    else:
        level = "framework_draft"
    return {
        "level": level,
        "score": score,
        "mandatory_proof_total": total,
        "mandatory_proof_found": found,
        "mandatory_proof_missing": len(missing),
        "mandatory_proof_coverage": round(coverage_ratio, 3) if total else None,
        "investment_mapping_present": investment_mapping_present,
        "trigger_monitor_present": trigger_monitor_present,
        "table_count": table_count,
        "metric_rows": metric_rows,
        "complete_metric_rows": complete_metric_rows,
        "missing_mandatory_proofs": missing[:10],
    }
