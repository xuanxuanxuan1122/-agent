from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence


AGENT_NAME = "decision_synthesis_agent"
AGENT_DESCRIPTION = "Decision Synthesis Agent. Builds public final judgments and recommendations."


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
        text = _compact(value, 180)
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


def _decision_labels(report_family: str) -> Dict[str, str]:
    return {
        "industry_deep_report": {
            "judgment": "机会判断",
            "recommendation": "进入窗口",
            "watch": "验证指标",
            "abandon": "风险边界",
        },
        "company_due_diligence_report": {
            "judgment": "业务真实性",
            "recommendation": "投资结论",
            "watch": "客户与财务质量",
            "abandon": "重大风险事项",
        },
        "product_research": {
            "judgment": "用户痛点",
            "recommendation": "产品切入点",
            "watch": "ROI验证",
            "abandon": "竞品缺口不足",
        },
        "policy_impact_report": {
            "judgment": "政策影响",
            "recommendation": "应对动作",
            "watch": "执行与预算传导",
            "abandon": "政策落地不确定性",
        },
        "investment_memo": {
            "judgment": "核心假设",
            "recommendation": "投资建议",
            "watch": "触发器",
            "abandon": "放弃条件",
        },
    }.get(
        report_family,
        {
            "judgment": "核心判断",
            "recommendation": "策略建议",
            "watch": "观察指标",
            "abandon": "放弃条件",
        },
    )


def run_decision_synthesis_agent(
    *,
    report_blueprint: Optional[Dict[str, Any]] = None,
    chapter_packages: Optional[Sequence[Dict[str, Any]]] = None,
    argument_units: Optional[Sequence[Dict[str, Any]]] = None,
    table_packages: Optional[Sequence[Dict[str, Any]]] = None,
    llm_client: Any = None,
) -> Dict[str, Any]:
    del llm_client, table_packages
    report_blueprint = _as_dict(report_blueprint)
    report_family = str(report_blueprint.get("report_family") or "topic_report")
    labels = _decision_labels(report_family)
    public_chapters = [
        chapter
        for chapter in list(chapter_packages or [])
        if isinstance(chapter, dict) and not chapter.get("omit_from_report")
    ]
    public_units = [
        unit
        for unit in list(argument_units or [])
        if isinstance(unit, dict) and unit.get("public_render") is True and not unit.get("omit_from_report")
    ]
    chapter_summaries = [_as_dict(chapter.get("chapter_summary")) for chapter in public_chapters]
    claims = _dedupe(
        [summary.get("key_takeaway") for summary in chapter_summaries]
        + [unit.get("claim") for unit in public_units],
        limit=6,
    )
    recommendations = _dedupe(
        [action for summary in chapter_summaries for action in _as_list(summary.get("next_actions"))]
        + [unit.get("actionable") for unit in public_units],
        limit=6,
    )
    watchlist = _dedupe([unit.get("counter_evidence") for unit in public_units], limit=6)
    key_data_points = _dedupe(
        [unit.get("reasoning") for unit in public_units]
        + [unit.get("mechanism") for unit in public_units],
        limit=5,
    )
    abandon = _dedupe(
        [
            unit.get("counter_evidence")
            for unit in public_units
            if str(unit.get("confidence") or "").lower() in {"medium", "medium_low"}
        ],
        limit=3,
    )
    thesis = (
        "综合判断应建立在章节之间的共同变量上：需求是否真实兑现、供给约束是否能解释价格和利润、客户或订单是否形成持续样本、反向证据是否足以改变主线。"
        if claims
        else ""
    )
    scenario_base = claims[:3] or recommendations[:3]
    scenarios = [
        {
            "scenario": "上行情景",
            "condition": "关键指标同向改善，A/B层级来源之间口径一致，企业端订单或客户行为能够验证行业端数据。",
            "implication": "可以提高研究或资源投入优先级，但仍需保留反向触发器。",
        },
        {
            "scenario": "中性情景",
            "condition": "部分指标改善但尚未传导到订单、利润或客户行为，章节结论之间仍存在时间差。",
            "implication": "适合继续观察和小规模验证，不宜把单章信号外推为总判断。",
        },
        {
            "scenario": "下行情景",
            "condition": "价格、库存、需求、订单或监管变量出现反向变化，且反向样本覆盖更近时间窗口。",
            "implication": "应降低结论强度，收缩投入动作，优先重新核验核心假设。",
        },
    ] if scenario_base else []
    opportunity_level = "high" if len(claims) >= 4 and len(recommendations) >= 3 else "medium" if claims else "insufficient"
    priority_segments = [
        {
            "segment": _compact(chapter.get("chapter_title") or chapter.get("chapter_question"), 120),
            "priority": index,
            "reason": _compact(_as_dict(chapter.get("chapter_summary")).get("key_takeaway"), 180),
        }
        for index, chapter in enumerate(public_chapters[:5], start=1)
        if _compact(chapter.get("chapter_title") or chapter.get("chapter_question"), 120)
    ]
    validation_sequence = [
        {
            "step": index,
            "action": _compact(item, 180),
            "evidence_required": "A/B source + metric scope/period/unit + counter check",
        }
        for index, item in enumerate((recommendations or claims)[:6], start=1)
    ]
    upside_triggers = [
        {"trigger": _compact(item, 180), "action": "raise_priority"}
        for item in (key_data_points or claims)[:5]
    ]
    downgrade_triggers = [
        {"trigger": _compact(item, 180), "action": "downgrade_or_pause"}
        for item in (abandon or watchlist)[:5]
    ]
    return {
        "agent": AGENT_NAME,
        "report_family": report_family,
        "labels": labels,
        "decision_thesis": thesis,
        "opportunity_level": opportunity_level,
        "priority_segments": priority_segments,
        "validation_sequence": validation_sequence,
        "upside_triggers": upside_triggers,
        "downgrade_triggers": downgrade_triggers,
        "core_judgments": [
            {"label": labels["judgment"], "judgment": claim, "evidence_refs": []}
            for claim in claims[:5]
        ],
        "key_data_points": key_data_points,
        "recommendations": [
            {"label": labels["recommendation"], "recommendation": item}
            for item in recommendations[:5]
        ],
        "watchlist": [
            {"label": labels["watch"], "metric": item}
            for item in watchlist[:5]
        ],
        "abandon_conditions": [
            {"label": labels["abandon"], "condition": item}
            for item in abandon[:3]
        ],
        "scenario_analysis": scenarios,
        "strategic_implications": [
            {"label": "优先级", "text": item}
            for item in recommendations[:6]
        ],
        "confidence": "medium" if claims else "insufficient",
        "omit_from_report": not bool(claims or recommendations),
    }
