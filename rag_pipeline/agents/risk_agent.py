from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence


AGENT_NAME = "risk_agent"
AGENT_DESCRIPTION = "Risk Agent. Converts public claims, conflicts, and decision boundaries into traceable risk items."


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _compact(value: Any, max_chars: int = 220) -> str:
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


def _public_units(argument_units: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        unit
        for unit in list(argument_units or [])
        if isinstance(unit, dict) and unit.get("public_render") is True and not unit.get("omit_from_report")
    ]


def _risk_item(
    *,
    risk_type: str,
    description: Any,
    impact: str,
    severity: str,
    mitigation: Any,
    watch_metric: Any,
    hypothesis_id: Any = "",
    trigger: Any = "",
) -> Dict[str, Any]:
    return {
        "risk_type": risk_type,
        "hypothesis_id": str(hypothesis_id or "").strip(),
        "trigger": _compact(trigger or description, 160),
        "description": _compact(description, 220),
        "impact": _compact(impact, 220),
        "severity": severity,
        "mitigation": _compact(mitigation, 220),
        "watch_metric": _compact(watch_metric, 120),
    }


def run_risk_agent(
    *,
    chapter_packages: Optional[Sequence[Dict[str, Any]]] = None,
    argument_units: Optional[Sequence[Dict[str, Any]]] = None,
    evidence_conflicts: Optional[Sequence[Dict[str, Any]]] = None,
    decision_package: Optional[Dict[str, Any]] = None,
    llm_client: Any = None,
) -> Dict[str, Any]:
    del llm_client
    public_chapters = [
        chapter
        for chapter in list(chapter_packages or [])
        if isinstance(chapter, dict) and not chapter.get("omit_from_report")
    ]
    nested_units = [
        unit
        for chapter in public_chapters
        for unit in _as_list(chapter.get("argument_units"))
        if isinstance(unit, dict)
    ]
    public_units = _public_units([*(argument_units or []), *nested_units])
    decision_package = _as_dict(decision_package)
    if not public_chapters and not public_units and not _as_list(decision_package.get("abandon_conditions")):
        return {"agent": AGENT_NAME, "risk_items": [], "omit_from_report": True}

    risk_items: List[Dict[str, Any]] = []
    public_refs = {
        str(ref or "")
        for unit in public_units
        for ref in _as_list(unit.get("evidence_refs"))
        if str(ref or "").strip()
    }
    for conflict in list(evidence_conflicts or [])[:6]:
        if not isinstance(conflict, dict):
            continue
        refs = {
            str(ref or "")
            for ref in [*_as_list(conflict.get("evidence_refs")), *_as_list(conflict.get("evidence_ids"))]
            if str(ref or "").strip()
        }
        if refs and public_refs and not refs.intersection(public_refs):
            continue
        risk_items.append(
            _risk_item(
                risk_type="数据口径风险",
                hypothesis_id=conflict.get("hypothesis_id"),
                trigger=conflict.get("metric") or conflict.get("dimension") or "metric_scope_conflict",
                description=conflict.get("description") or "同一指标存在不同口径或数值。",
                impact="可能导致规模、增速、排序或进入窗口判断偏差。",
                severity="medium",
                mitigation="优先使用同口径、高等级、最新来源，并保留冲突记录。",
                watch_metric=conflict.get("metric") or "口径一致性",
            )
        )

    for unit in public_units:
        counter = _compact(unit.get("counter_evidence"), 180)
        if not counter:
            continue
        risk_items.append(
            _risk_item(
                risk_type="假设边界风险",
                hypothesis_id=unit.get("hypothesis_id"),
                trigger=unit.get("section_title") or unit.get("claim") or counter,
                description=counter,
                impact="若边界条件变化，相关机会排序和进入节奏需要下调。",
                severity="medium",
                mitigation=unit.get("actionable") or "设置阶段性验证门槛。",
                watch_metric=unit.get("section_title") or "核心假设验证",
            )
        )

    for condition in _as_list(decision_package.get("abandon_conditions"))[:3]:
        condition = _as_dict(condition)
        text = _compact(condition.get("condition"), 180)
        if not text:
            continue
        risk_items.append(
            _risk_item(
                risk_type="执行边界风险",
                hypothesis_id=condition.get("hypothesis_id"),
                trigger=text,
                description=text,
                impact="触发后应暂停投入或重新定义研究假设。",
                severity="high",
                mitigation="设置阶段性验证门槛，未达标不进入下一轮资源投入。",
                watch_metric=condition.get("label") or "放弃条件",
            )
        )

    severity_rank = {"high": 0, "medium": 1, "low": 2}
    ordered_risks = sorted(
        risk_items,
        key=lambda item: (
            severity_rank.get(str(item.get("severity") or "").lower(), 3),
            str(item.get("risk_type") or ""),
        ),
    )
    deduped = []
    for description in _dedupe([item.get("description") for item in ordered_risks], limit=8):
        for item in ordered_risks:
            if item.get("description") == description:
                deduped.append(item)
                break
    return {"agent": AGENT_NAME, "risk_items": deduped[:8], "omit_from_report": not bool(deduped)}
