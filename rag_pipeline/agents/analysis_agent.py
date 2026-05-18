from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, TypedDict

try:
    from .evidence_merger import get_dynamic_dimensions
except Exception:  # pragma: no cover - script mode fallback
    from evidence_merger import get_dynamic_dimensions  # type: ignore


AGENT_NAME = "analysis_agent"
AGENT_DESCRIPTION = "Dynamic Research Claim Builder. Converts evidence packages into claim units for the writer."


class AnalysisAgentState(TypedDict, total=False):
    query: str
    evidence_package: Dict[str, Any]
    structured_analysis: Dict[str, Any]
    answer_text: str
    raw_output: Dict[str, Any]
    metadata: Dict[str, Any]
    errors: List[str]


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _compact(value: Any, max_chars: int = 260) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "..."


def _dedupe(values: List[Any]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _normalize_key(value: Any) -> str:
    text = re.sub(r"\s+", "", str(value or "").lower())
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", text)


def _overlaps(left: Any, right: Any) -> bool:
    left_key = _normalize_key(left)
    right_key = _normalize_key(right)
    if not left_key or not right_key:
        return False
    if left_key in right_key or right_key in left_key:
        return True
    overlap = set(left_key) & set(right_key)
    return len(overlap) >= max(2, min(len(left_key), len(right_key)) // 3)


def _research_plan(evidence_package: Dict[str, Any]) -> Dict[str, Any]:
    return _as_dict(evidence_package.get("research_plan")) or _as_dict(_as_dict(evidence_package.get("metadata")).get("research_plan"))


def _analysis_dimensions(evidence_package: Dict[str, Any]) -> List[str]:
    research_plan = _research_plan(evidence_package)
    dimensions = get_dynamic_dimensions(research_plan)
    for dimension in _as_dict(evidence_package.get("per_dimension")).keys():
        text = str(dimension or "").strip()
        if text and text not in dimensions:
            dimensions.append(text)
    for item in _as_list(evidence_package.get("analysis_ready_evidence")) + _as_list(evidence_package.get("clean_evidence_list")):
        if isinstance(item, dict):
            text = str(item.get("dimension_name") or item.get("evidence_goal") or item.get("dimension") or "").strip()
            if text and text not in dimensions:
                dimensions.append(text)
    return dimensions or ["综合研究问题"]


def _fact_text(item: Dict[str, Any]) -> str:
    for key in ("fact", "clean_fact", "content", "clean_content", "answer", "claim", "takeaway"):
        text = _compact(item.get(key), 260)
        if text:
            return text
    metric = _compact(item.get("metric"), 80)
    value = _compact(item.get("value"), 80)
    if metric and value:
        return f"{metric}: {value}"
    return ""


def _source_payload(item: Dict[str, Any]) -> Dict[str, Any]:
    source = _as_dict(item.get("source"))
    if source:
        return source
    key_sources = _as_list(item.get("key_sources"))
    for source_item in key_sources:
        if isinstance(source_item, dict):
            return source_item
    return {}


def _source_label(item: Dict[str, Any]) -> str:
    source = _source_payload(item)
    title = str(source.get("title") or source.get("source") or source.get("name") or "").strip()
    date = str(source.get("date") or source.get("period") or "").strip()
    return " | ".join(part for part in [title, date] if part)


def _confidence(item: Dict[str, Any]) -> float:
    try:
        value = float(item.get("confidence", 0.5))
    except (TypeError, ValueError):
        value = 0.5
    return max(0.0, min(1.0, value))


def _items_for_dimension(evidence_package: Dict[str, Any], dimension: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    payload = _as_dict(_as_dict(evidence_package.get("per_dimension")).get(dimension))
    for item in _as_list(payload.get("analysis_inputs")) + _as_list(payload.get("clean_facts")) + _as_list(payload.get("top_evidence")):
        if isinstance(item, dict):
            copied = dict(item)
            copied.setdefault("dimension", dimension)
            items.append(copied)
    for item in _as_list(evidence_package.get("analysis_ready_evidence")) + _as_list(evidence_package.get("clean_evidence_list")):
        if not isinstance(item, dict):
            continue
        item_dimension = str(item.get("dimension_name") or item.get("evidence_goal") or item.get("dimension") or "").strip()
        if item_dimension == dimension:
            items.append(dict(item))
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for item in items:
        key = (_fact_text(item), _source_label(item))
        if key in seen or not key[0]:
            continue
        seen.add(key)
        deduped.append(item)
    deduped.sort(key=lambda item: (_confidence(item), 1 if _source_label(item) else 0), reverse=True)
    return deduped
def _claim_units_from_synthesis(dimension_synthesis: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    units: List[Dict[str, Any]] = []
    for dimension, synthesis in dimension_synthesis.items():
        synthesis = _as_dict(synthesis)
        units.append(
            {
                "question": dimension,
                "claim": synthesis.get("takeaway") or "",
                "supporting_evidence": synthesis.get("evidence_ids") or [],
                "counter_evidence": synthesis.get("counter") or "",
                "reasoning": synthesis.get("mechanism") or synthesis.get("explain_why") or "",
                "mechanism": synthesis.get("mechanism") or "",
                "decision_implication": synthesis.get("decision_implication") or synthesis.get("verify_kpi") or "",
                "confidence": synthesis.get("confidence"),
                "dimension": dimension,
            }
        )
    return units


def _chapter_insights_from_synthesis(dimension_synthesis: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    insights: List[Dict[str, Any]] = []
    for index, (dimension, synthesis) in enumerate(dimension_synthesis.items(), start=1):
        synthesis = _as_dict(synthesis)
        insights.append(
            {
                "chapter_id": f"chapter_{index}",
                "chapter_question": dimension,
                "chapter_answer": synthesis.get("chapter_answer") or synthesis.get("takeaway") or "",
                "key_claims": [
                    {
                        "claim": synthesis.get("takeaway") or "",
                        "supporting_evidence": synthesis.get("evidence_ids") or [],
                        "mechanism": synthesis.get("mechanism") or "",
                        "counter_evidence": synthesis.get("counter") or "",
                        "decision_implication": synthesis.get("decision_implication") or "",
                        "confidence": synthesis.get("confidence"),
                        "what_to_verify_next": [synthesis.get("verify_kpi")],
                    }
                ],
                "decision_readiness": "ready" if _as_list(synthesis.get("evidence_ids")) else "needs_evidence",
                "blocking_gaps": [] if _as_list(synthesis.get("evidence_ids")) else ["evidence_missing"],
            }
        )
    return insights


def _analysis_source_level(item: Dict[str, Any]) -> str:
    return str(item.get("source_level") or _as_dict(item.get("source")).get("credibility") or "").strip().upper()


def _analysis_allowed_use(item: Dict[str, Any]) -> str:
    allowed = str(item.get("allowed_use") or _as_dict(item.get("evidence_card")).get("allowed_use") or "").strip()
    if allowed:
        return allowed
    level = _analysis_source_level(item)
    role = str(item.get("evidence_role") or "").strip().lower()
    if level in {"A", "B"} and role == "core":
        return "core_claim"
    if level in {"A", "B"} and role == "supporting":
        return "supporting"
    try:
        confidence = float(item.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    if level == "C" and confidence >= 0.55 and not item.get("appendix_only"):
        return "directional_signal"
    if level == "C":
        return "clue"
    return "appendix_only"


def _is_usable_for_claim(item: Dict[str, Any]) -> bool:
    allowed_use = _analysis_allowed_use(item)
    return (
        _analysis_source_level(item) in {"A", "B"} and allowed_use in {"core_claim", "supporting"}
    ) or allowed_use == "directional_signal"


def _evidence_strength(item: Dict[str, Any]) -> str:
    level = _analysis_source_level(item)
    allowed = _analysis_allowed_use(item)
    if level in {"A", "B"} and allowed == "core_claim":
        return "strong"
    if level in {"A", "B"} and allowed == "supporting":
        return "medium"
    if allowed == "directional_signal":
        return "directional"
    return "weak"


def _evidence_gap_tags(item: Dict[str, Any]) -> List[str]:
    level = _analysis_source_level(item)
    allowed = _analysis_allowed_use(item)
    gaps: List[str] = []
    if level not in {"A", "B"} and allowed != "directional_signal":
        gaps.append("needs_authoritative_source")
    if allowed in {"clue", "appendix_only"}:
        gaps.append("needs_corroboration")
    if not _source_label(item):
        gaps.append("source_metadata_missing")
    metric = _compact(item.get("metric"), 80)
    value = _compact(item.get("value"), 80)
    period = _compact(item.get("period") or _source_payload(item).get("date"), 80)
    if metric and not value:
        gaps.append("metric_value_missing")
    if (metric or value) and not period:
        gaps.append("metric_period_missing")
    if str(item.get("proof_role") or "").strip().lower() == "counter" and not _is_usable_for_claim(item):
        gaps.append("counter_needs_ab_source")
    return _dedupe(gaps)


def _followup_query_for_evidence(item: Dict[str, Any], *, dimension: str) -> str:
    gaps = set(_evidence_gap_tags(item))
    parts = [
        dimension,
        _compact(item.get("hypothesis_statement"), 80),
        _compact(item.get("metric"), 60),
        _compact(_fact_text(item), 90),
    ]
    if "needs_authoritative_source" in gaps or "needs_corroboration" in gaps:
        parts.extend(["官方", "公告", "财报", "协会", "权威研报"])
    if "metric_value_missing" in gaps or "metric_period_missing" in gaps:
        parts.extend(["指标口径", "数值", "期间", "单位", "范围"])
    if "counter_needs_ab_source" in gaps:
        parts.extend(["反证", "风险", "失败案例", "订单取消", "监管变化"])
    query = " ".join(part for part in parts if str(part or "").strip())
    return _compact(query, 220)


def _verification_questions(item: Dict[str, Any], *, dimension: str) -> List[str]:
    fact = _compact(_fact_text(item), 90)
    questions = [
        f"{dimension} 的这个信号是否有 A/B 级来源复核？",
        "同一口径下是否能找到时间、范围、单位一致的指标？",
    ]
    if fact:
        questions.insert(0, f"'{fact}' 能否被后续披露或第二来源验证？")
    if str(item.get("proof_role") or "").strip().lower() != "counter":
        questions.append("是否存在方向相反的反证或失败案例？")
    return _dedupe(questions)[:4]


def _evidence_card_from_item(item: Dict[str, Any], *, dimension: str, fact: str) -> Dict[str, Any]:
    card = _as_dict(item.get("evidence_card"))
    if card:
        return card
    source = _source_payload(item)
    level = _analysis_source_level(item) or "UNKNOWN"
    allowed = _analysis_allowed_use(item)
    return {
        "fact": fact,
        "source_level": level,
        "source_family": str(item.get("source_family") or "unknown"),
        "proof_role": str(item.get("proof_role") or ("counter" if item.get("counter_evidence") else "support")).strip().lower(),
        "directness": "direct" if item.get("metric") or item.get("value") else "indirect",
        "scope": str(item.get("scope") or item.get("dimension_name") or dimension or "").strip(),
        "period": str(item.get("period") or source.get("date") or "").strip(),
        "metric_definition": {
            "metric": item.get("metric"),
            "value": item.get("value"),
            "period": item.get("period") or source.get("date") or "",
        },
        "can_prove": [item.get("evidence_goal") or dimension],
        "cannot_prove": ["single-source conclusion", "industry-wide certainty", "investment priority without evidence bundle"],
        "inference_distance": "low" if allowed == "core_claim" else ("medium" if allowed == "supporting" else "high"),
        "contradictions": [],
        "allowed_use": allowed,
    }


def _evidence_analysis(item: Dict[str, Any], dimension: str, index: int) -> Dict[str, Any]:
    fact = _fact_text(item)
    source = _source_payload(item)
    evidence_id = str(item.get("evidence_id") or item.get("id") or f"EV-{index:04d}")
    card = _evidence_card_from_item(item, dimension=dimension, fact=fact)
    gaps = _evidence_gap_tags(item)
    verification_questions = _verification_questions(item, dimension=dimension)
    followup_query = _followup_query_for_evidence(item, dimension=dimension) if gaps else ""
    strength = _evidence_strength(item)
    return {
        "evidence_id": evidence_id,
        "dimension": dimension,
        "fact": fact,
        "writer_evidence": fact,
        "source": source,
        "source_label": _source_label(item),
        "confidence": _confidence(item),
        "hypothesis_id": item.get("hypothesis_id"),
        "hypothesis_statement": item.get("hypothesis_statement"),
        "proof_role": card.get("proof_role") or item.get("proof_role") or ("counter" if item.get("counter_evidence") else "support"),
        "source_level": card.get("source_level") or _analysis_source_level(item),
        "source_family": card.get("source_family") or item.get("source_family"),
        "metric": item.get("metric"),
        "value": item.get("value"),
        "allowed_use": card.get("allowed_use"),
        "evidence_card": card,
        "evidence_card_only": True,
        "evidence_strength": strength,
        "evidence_gaps": gaps,
        "verification_questions": verification_questions,
        "suggested_followup_query": followup_query,
        "claim": f"{dimension} 出现可观察信号，但结论强度取决于来源等级、指标口径和反证覆盖。" if fact else "",
        "reasoning": "该证据可用于建立事实链的一环；若要进入核心判断，需要与同口径指标、第二来源或反向案例交叉验证。",
        "mechanism": "先确认事实是否可复核，再判断它影响的是需求、供给、政策约束还是企业行为，最后评估能否外推为趋势。",
        "counter": "若后续 A/B 来源显示指标反向变化、企业动作未延续或出现失败案例，应下调该证据对结论的权重。",
        "decision_implication": "可作为正文分析素材；存在缺口时优先转入补证任务，而不是直接放大为强结论。",
        "analysis_depth": {
            "can_prove": card.get("can_prove") or [dimension],
            "cannot_prove": card.get("cannot_prove") or ["single-source conclusion"],
            "inference_distance": card.get("inference_distance"),
            "strength": strength,
            "gaps": gaps,
            "verification_questions": verification_questions,
            "suggested_followup_query": followup_query,
        },
    }


def _dimension_synthesis(dimension: str, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
    usable = [item for item in analyses if _is_usable_for_claim(item)]
    evidence_ids = [str(item.get("evidence_id")) for item in usable if item.get("evidence_id")][:12]
    usable_facts = [_compact(item.get("fact"), 120) for item in usable if str(item.get("fact") or "").strip()]
    all_gaps = _dedupe(
        [
            gap
            for item in analyses
            for gap in _as_list(item.get("evidence_gaps"))
        ]
    )
    followups = _dedupe(
        [
            item.get("suggested_followup_query")
            for item in analyses
            if str(item.get("suggested_followup_query") or "").strip()
        ]
    )
    first_fact = usable_facts[0] if usable_facts else _compact(_as_dict(analyses[0] if analyses else {}).get("fact"), 220)
    if usable:
        takeaway = f"{dimension} 已有 {len(usable)} 条可用于正文的信号，核心事实是：{first_fact}"
        mechanism = f"{dimension} 的分析应先看事实是否连续，再看它传导到需求、供给、政策约束或企业行为中的哪一环。"
        counter = "需要保留反证边界：若后续 A/B 来源显示同口径指标走弱、企业动作中断或监管条件收紧，结论应降级。"
        decision = "可进入正文作为分析主线，但应按来源等级和口径完整性区分强结论、方向性判断和待复核线索。"
    elif analyses:
        takeaway = f"{dimension} 目前只有线索或背景材料，尚不足以支撑强结论。"
        mechanism = "当前材料应先转化为补证任务，优先补 A/B 来源、指标口径和反向案例。"
        counter = "没有反证并不等于风险不存在；反证缺位本身应作为结论边界。"
        decision = "正文只能写成待验证方向，不能写成确定判断。"
    else:
        takeaway = ""
        mechanism = ""
        counter = ""
        decision = ""
    return {
        "takeaway": takeaway,
        "chapter_answer": takeaway,
        "fact": first_fact,
        "supporting_facts": usable_facts[:6],
        "explain_why": mechanism,
        "mechanism": mechanism,
        "inference": "强度取决于证据是否同时满足来源可信、口径完整、可被第二来源复核。",
        "counter": counter,
        "verify_kpi": "补齐 A/B 来源、同口径指标、时间范围、单位、反证案例",
        "decision_implication": decision,
        "evidence_ids": evidence_ids,
        "confidence": round(sum(float(item.get("confidence") or 0.0) for item in usable) / max(len(usable), 1), 3) if usable else 0.0,
        "limits": "；".join(all_gaps[:5]),
        "evidence_gap_tags": all_gaps,
        "followup_queries": followups[:6],
    }


def _hypothesis_insights(research_plan: Dict[str, Any], evidence_analyses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    insights: List[Dict[str, Any]] = []
    hypotheses = [item for item in _as_list(research_plan.get("hypotheses")) if isinstance(item, dict)]
    requirements = _as_dict(_as_dict(research_plan.get("evidence_coverage_requirements")).get("per_hypothesis"))
    required_ab = int(requirements.get("min_A_or_B_sources") if requirements.get("min_A_or_B_sources") not in {None, ""} else 1)
    required_counter = int(requirements.get("min_counter_sources") or 0)
    required_metric = int(requirements.get("min_metric_sources") or 0)
    required_case = int(requirements.get("min_case_sources") or 0)
    required_families = [str(item) for item in _as_list(requirements.get("source_diversity")) if str(item or "").strip()]
    for index, hypothesis in enumerate(hypotheses, start=1):
        hypothesis_id = str(hypothesis.get("hypothesis_id") or f"H{index}")
        statement = _compact(hypothesis.get("claim_to_test") or hypothesis.get("hypothesis_statement") or hypothesis.get("statement"), 260)
        relevant = [
            item
            for item in evidence_analyses
            if str(item.get("hypothesis_id") or "") == hypothesis_id
            or _overlaps(statement, item.get("dimension"))
            or _overlaps(statement, item.get("fact"))
        ]
        usable = [item for item in relevant if _is_usable_for_claim(item)]
        support = [item for item in usable if str(item.get("proof_role") or "").lower() != "counter"]
        counters = [item for item in usable if str(item.get("proof_role") or "").lower() == "counter"]
        metric_items = [
            item
            for item in usable
            if str(item.get("proof_role") or "").lower() == "metric" or bool(item.get("metric") or item.get("value"))
        ]
        case_items = [
            item
            for item in usable
            if str(item.get("proof_role") or "").lower() == "case" or str(item.get("source_family") or "") == "company/case"
        ]
        families = sorted({str(item.get("source_family") or "unknown") for item in usable})
        gaps: List[str] = []
        if len(usable) < required_ab:
            gaps.append("insufficient_ab_sources")
        if bool(hypothesis.get("counter_evidence_required", False)) and required_counter > 0 and len(counters) < required_counter:
            gaps.append("counter_evidence_missing")
        if required_metric > 0 and len(metric_items) < required_metric:
            gaps.append("metric_evidence_missing")
        if required_case > 0 and len(case_items) < required_case:
            gaps.append("case_evidence_missing")
        if required_families and not set(required_families).issubset(set(families)):
            gaps.append("source_diversity_missing")
        ready = not gaps and len(usable) >= max(1, required_ab)
        evidence_ids = [str(item.get("evidence_id")) for item in support if item.get("evidence_id")]
        counter_ids = [str(item.get("evidence_id")) for item in counters if item.get("evidence_id")]
        key_claims = []
        if ready:
            key_claims.append(
                {
                    "claim": statement,
                    "supporting_evidence": evidence_ids[:10],
                    "counter_evidence_refs": counter_ids[:6],
                    "mechanism": "该判断由指标、来源核验、客户/案例和反证证据束共同支撑。",
                    "counter_evidence": "反证已纳入判断边界；若后续A/B来源显示价格、订单、产能或客户验证反向变化，应下调结论。",
                    "decision_implication": "可进入正文核心判断，并用于进入/投资/产品布局优先级排序。",
                    "confidence": round(sum(float(item.get("confidence") or 0.0) for item in usable) / max(len(usable), 1), 3),
                    "what_to_verify_next": ["持续跟踪价格/毛利", "复核客户认证与订单", "监控产能过剩和替代路线"],
                }
            )
        insights.append(
            {
                "chapter_id": f"hypothesis_{index}",
                "hypothesis_id": hypothesis_id,
                "chapter_question": statement,
                "chapter_answer": statement if ready else "",
                "key_claims": key_claims,
                "decision_readiness": "ready" if ready else "needs_evidence",
                "blocking_gaps": gaps,
            }
        )
    return insights


def _gap_priority(gap: str) -> int:
    return {
        "insufficient_ab_sources": 0,
        "only_c_or_lower_sources": 1,
        "metric_evidence_missing": 2,
        "metric_definition_unfilled": 2,
        "metric_scope_period_unit_incomplete": 3,
        "counter_evidence_missing": 4,
        "case_evidence_missing": 5,
        "source_diversity_missing": 6,
        "needs_authoritative_source": 7,
        "needs_corroboration": 8,
    }.get(str(gap or ""), 20)


def _followup_for_gap(*, target: str, gap: str, hypothesis_id: str = "", dimension: str = "") -> Dict[str, Any]:
    query_parts = [target or dimension or hypothesis_id]
    proof_role = "support"
    evidence_type = "data"
    lane_targets = ["official_data", "filing_company", "market_research"]
    source_priority = ["official", "filing", "research_report"]
    if gap in {"insufficient_ab_sources", "only_c_or_lower_sources", "needs_authoritative_source"}:
        query_parts.extend(["官方", "公告", "财报", "协会", "权威研报", "A/B来源"])
    if gap in {"metric_evidence_missing", "metric_definition_unfilled", "metric_scope_period_unit_incomplete"}:
        query_parts.extend(["指标口径", "数值", "期间", "单位", "范围"])
        proof_role = "metric"
        evidence_type = "metric"
        lane_targets = ["official_data", "market_research"]
    if gap == "counter_evidence_missing":
        query_parts.extend(["反证", "风险", "失败案例", "价格下跌", "订单取消", "监管变化"])
        proof_role = "counter"
        evidence_type = "counter"
        lane_targets = ["news_event", "filing_company", "market_research"]
    if gap == "case_evidence_missing":
        query_parts.extend(["客户案例", "订单", "认证", "量产", "供应合同"])
        proof_role = "case"
        evidence_type = "case"
        lane_targets = ["customer_case", "filing_company"]
    if gap in {"source_diversity_missing", "needs_corroboration"}:
        query_parts.extend(["第二来源", "交叉验证", "官方", "公司披露"])
    query = _compact(" ".join(part for part in query_parts if str(part or "").strip()), 220)
    return {
        "query": query,
        "agent": "iqs",
        "targets_gap": target or dimension or hypothesis_id or gap,
        "dimension_name": dimension or target,
        "evidence_goal": target or dimension,
        "hypothesis_id": hypothesis_id,
        "hypothesis_statement": target,
        "proof_role": proof_role,
        "evidence_type": evidence_type,
        "lane_targets": lane_targets,
        "source_priority": source_priority,
        "blocking_gaps": [gap],
        "priority": _gap_priority(gap),
    }


def _evidence_refinement_plan(
    *,
    evidence_analyses: List[Dict[str, Any]],
    hypothesis_insights: List[Dict[str, Any]],
    dimension_synthesis: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    tasks: List[Dict[str, Any]] = []
    for chapter in hypothesis_insights:
        chapter = _as_dict(chapter)
        target = _compact(chapter.get("chapter_question") or chapter.get("hypothesis_statement"), 160)
        hypothesis_id = str(chapter.get("hypothesis_id") or "").strip()
        for gap in _as_list(chapter.get("blocking_gaps")):
            tasks.append(_followup_for_gap(target=target, gap=str(gap), hypothesis_id=hypothesis_id, dimension=target))
    for dimension, payload in dimension_synthesis.items():
        payload = _as_dict(payload)
        for gap in _as_list(payload.get("evidence_gap_tags")):
            tasks.append(_followup_for_gap(target=str(dimension), gap=str(gap), dimension=str(dimension)))
        for query in _as_list(payload.get("followup_queries")):
            query_text = _compact(query, 220)
            if query_text:
                tasks.append(
                    {
                        "query": query_text,
                        "agent": "iqs",
                        "targets_gap": str(dimension),
                        "dimension_name": str(dimension),
                        "evidence_goal": str(dimension),
                        "proof_role": "support",
                        "evidence_type": "data",
                        "blocking_gaps": ["needs_corroboration"],
                        "priority": _gap_priority("needs_corroboration"),
                    }
                )
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for task in sorted(tasks, key=lambda item: (int(item.get("priority") or 20), str(item.get("targets_gap") or ""), str(item.get("query") or ""))):
        key = (task.get("targets_gap"), task.get("proof_role"), task.get("query"))
        if key in seen or not str(task.get("query") or "").strip():
            continue
        seen.add(key)
        deduped.append(task)
    gap_counts: Dict[str, int] = {}
    for item in evidence_analyses:
        for gap in _as_list(item.get("evidence_gaps")):
            gap_text = str(gap or "")
            if gap_text:
                gap_counts[gap_text] = gap_counts.get(gap_text, 0) + 1
    for chapter in hypothesis_insights:
        for gap in _as_list(_as_dict(chapter).get("blocking_gaps")):
            gap_text = str(gap or "")
            if gap_text:
                gap_counts[gap_text] = gap_counts.get(gap_text, 0) + 1
    return {
        "status": "needs_refinement" if deduped else "sufficient_for_current_analysis",
        "gap_counts": dict(sorted(gap_counts.items(), key=lambda pair: (_gap_priority(pair[0]), pair[0]))),
        "follow_up_queries": deduped[:20],
        "top_priorities": deduped[:6],
    }


def build_fallback_analysis(evidence_package: Dict[str, Any]) -> Dict[str, Any]:
    evidence_package = _as_dict(evidence_package)
    research_plan = _research_plan(evidence_package)
    dimensions = _analysis_dimensions(evidence_package)
    evidence_analyses: List[Dict[str, Any]] = []
    refs_by_dimension: Dict[str, List[str]] = {}
    index = 1
    for dimension in dimensions:
        items = _items_for_dimension(evidence_package, dimension)
        refs_by_dimension[dimension] = []
        for item in items[:18]:
            analysis = _evidence_analysis(item, dimension, index)
            index += 1
            evidence_analyses.append(analysis)
            refs_by_dimension[dimension].append(str(analysis.get("evidence_id")))
    dimension_synthesis = {
        dimension: _dimension_synthesis(
            dimension,
            [item for item in evidence_analyses if str(item.get("dimension") or "") == dimension],
        )
        for dimension in dimensions
    }
    claim_units = _claim_units_from_synthesis(dimension_synthesis)
    core_facts = [
        {
            "dimension": item.get("dimension"),
            "fact": item.get("fact"),
            "evidence_ids": [item.get("evidence_id")],
            "confidence": item.get("confidence"),
        }
        for item in evidence_analyses
        if item.get("fact")
    ][: max(8, len(dimensions) * 3)]
    key_judgments = [
        {
            "judgment": unit.get("claim"),
            "supporting_dimensions": [unit.get("dimension")],
            "evidence_ids": unit.get("supporting_evidence") or [],
            "confidence": unit.get("confidence"),
            "decision_implication": unit.get("decision_implication"),
        }
        for unit in claim_units
    ]
    report_outline = [
        {
            "section": str(chapter.get("name") or chapter.get("title") or chapter.get("id") or ""),
            "dimension": ", ".join(_as_list(_as_dict(evidence_package.get("chapter_dim_mapping")).get(chapter.get("id")))),
            "evidence_ids": [],
        }
        for chapter in _as_list(evidence_package.get("chapter_plan"))
        if isinstance(chapter, dict)
    ] or [{"section": dimension, "dimension": dimension, "evidence_ids": refs_by_dimension.get(dimension, [])[:6]} for dimension in dimensions]
    hypothesis_insights = _hypothesis_insights(research_plan, evidence_analyses)
    hypothesis_key_judgments = [
        {
            "judgment": claim.get("claim"),
            "supporting_dimensions": [chapter.get("chapter_question")],
            "evidence_ids": claim.get("supporting_evidence") or [],
            "confidence": claim.get("confidence"),
            "decision_implication": claim.get("decision_implication"),
        }
        for chapter in hypothesis_insights
        for claim in _as_list(_as_dict(chapter).get("key_claims"))
        if isinstance(claim, dict) and str(claim.get("claim") or "").strip()
    ]
    if not any(str(item.get("judgment") or "").strip() for item in key_judgments):
        key_judgments = hypothesis_key_judgments
    chapter_insights = hypothesis_insights or _chapter_insights_from_synthesis(dimension_synthesis)
    evidence_refinement_plan = _evidence_refinement_plan(
        evidence_analyses=evidence_analyses,
        hypothesis_insights=hypothesis_insights,
        dimension_synthesis=dimension_synthesis,
    )
    report_insight_package = {
        "report_thesis": _compact(key_judgments[0].get("judgment") if key_judgments else "", 260),
        "executive_summary": {
            "one_sentence_answer": _compact(key_judgments[0].get("judgment") if key_judgments else "", 220),
            "top_3_judgments": key_judgments[:3],
            "what_changed": _dedupe([item.get("fact") for item in core_facts])[:5],
            "so_what": _dedupe([item.get("decision_implication") for item in key_judgments])[:5],
        },
        "chapters": chapter_insights,
        "decision_matrix": _as_list(_as_dict(evidence_package.get("decision_layer")).get("decision_matrix")),
        "risk_register": _as_list(_as_dict(evidence_package.get("risk_layer")).get("risk_items")),
        "evidence_refinement_plan": evidence_refinement_plan,
        "source_appendix": _as_list(evidence_package.get("source_registry")),
    }
    return {
        "analysis_type": "structured_analysis",
        "query": str(evidence_package.get("query") or ""),
        "research_plan": research_plan,
        "evidence_analyses": evidence_analyses,
        "dimension_synthesis": dimension_synthesis,
        "chapter_insights": chapter_insights,
        "hypothesis_insights": hypothesis_insights,
        "report_insight_package": report_insight_package,
        "claim_units": claim_units,
        "core_facts": core_facts,
        "key_judgments": key_judgments,
        "evidence_gap_analysis": [
            {
                "evidence_id": item.get("evidence_id"),
                "dimension": item.get("dimension"),
                "gaps": _as_list(item.get("evidence_gaps")),
                "verification_questions": _as_list(item.get("verification_questions")),
                "suggested_followup_query": item.get("suggested_followup_query"),
            }
            for item in evidence_analyses
            if _as_list(item.get("evidence_gaps")) or str(item.get("suggested_followup_query") or "").strip()
        ],
        "evidence_refinement_plan": evidence_refinement_plan,
        "counter_analyses": [
            {
                "dimension": dimension,
                "counter": payload.get("counter"),
                "verify_kpi": payload.get("verify_kpi"),
            }
            for dimension, payload in dimension_synthesis.items()
        ],
        "decision_layer": {
            "decision_context": research_plan.get("decision_context") or "",
            "research_type": research_plan.get("research_type") or "",
            "report_family": research_plan.get("report_family") or "",
            "next_actions": _dedupe([unit.get("decision_implication") for unit in claim_units])[:8],
        },
        "report_outline": report_outline,
        "metadata": {
            "agent": AGENT_NAME,
            "strategy": "dynamic_claim_builder",
            "dimension_count": len(dimensions),
            "evidence_analysis_count": len(evidence_analyses),
            "evidence_refinement_task_count": len(_as_list(evidence_refinement_plan.get("follow_up_queries"))),
        },
    }


def run_analysis_agent(
    evidence_package: Dict[str, Any],
    *,
    query: str = "",
    llm_config: Optional[Dict[str, Any]] = None,
) -> AnalysisAgentState:
    try:
        package = _as_dict(evidence_package)
        if query and not package.get("query"):
            package = {**package, "query": query}
        structured = build_fallback_analysis(package)
        return {
            "query": query or str(package.get("query") or ""),
            "evidence_package": package,
            "structured_analysis": structured,
            "answer_text": json.dumps({"structured_analysis": structured}, ensure_ascii=False, separators=(",", ":"), default=str),
            "raw_output": {
                "type": "structured_analysis",
                "source": "dynamic_claim_builder",
                "structured_analysis": structured,
            },
            "metadata": {
                "agent_name": AGENT_NAME,
                "agent_description": AGENT_DESCRIPTION,
                "agent_stage": "analyze_evidence",
                "handoff_ready": True,
            },
        }
    except Exception as exc:
        return {
            "query": query,
            "evidence_package": _as_dict(evidence_package),
            "structured_analysis": {},
            "answer_text": "",
            "errors": [str(exc)],
            "raw_output": {"type": "structured_analysis", "source": "failed", "error": str(exc)},
            "metadata": {
                "agent_name": AGENT_NAME,
                "agent_description": AGENT_DESCRIPTION,
                "agent_stage": "analyze_evidence",
                "handoff_ready": False,
            },
        }


def analysis_agent_tool(evidence_package: Dict[str, Any]) -> Dict[str, Any]:
    return _as_dict(run_analysis_agent(evidence_package).get("structured_analysis"))


def create_analysis_agent_tool():
    from langchain_core.tools import tool

    @tool("analysis_agent", description=AGENT_DESCRIPTION)
    def _analysis_agent(evidence_package: Dict[str, Any]) -> Dict[str, Any]:
        return analysis_agent_tool(evidence_package)

    return _analysis_agent


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=AGENT_DESCRIPTION)
    parser.add_argument("--input-json", required=True, help="Evidence package JSON file")
    args = parser.parse_args()
    with open(args.input_json, "r", encoding="utf-8") as file:
        package = json.load(file)
    state = run_analysis_agent(package)
    print(state.get("answer_text") or json.dumps(state, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
