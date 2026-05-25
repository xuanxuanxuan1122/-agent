from __future__ import annotations

import re
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    from .block_schema import select_blocks_for_chapter
except Exception:  # pragma: no cover - direct script mode fallback
    from block_schema import select_blocks_for_chapter  # type: ignore


AGENT_NAME = "micro_layout_agent"
AGENT_DESCRIPTION = "Micro Layout Agent. Chooses chapter-level expression structures after evidence binding."


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _compact(value: Any, max_chars: int = 220) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "..."


def _env_int(name: str, default: int, *, min_value: int = 0, max_value: int = 100) -> int:
    try:
        value = int(os.getenv(name, str(default)) or default)
    except (TypeError, ValueError):
        value = default
    return max(min_value, min(max_value, value))


def _dedupe(values: Iterable[Any], *, limit: int = 12) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        text = _compact(value, 160)
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


def _text_blob(package: Dict[str, Any]) -> str:
    parts = [
        package.get("chapter_title"),
        package.get("chapter_question"),
    ]
    for collection in ("core_evidence", "supporting_evidence", "sample_evidence", "table_evidence", "clue_evidence"):
        for item in _as_list(package.get(collection))[:8]:
            if isinstance(item, dict):
                parts.extend([item.get("fact"), item.get("metric"), item.get("value")])
    return " ".join(str(part or "") for part in parts)


def _evidence_shape(package: Dict[str, Any], chapter_blueprint: Optional[Dict[str, Any]] = None) -> set[str]:
    chapter_blueprint = _as_dict(chapter_blueprint)
    values: set[str] = set()
    for key in ("required_evidence_mix", "source_template_keys"):
        for value in _as_list(package.get(key)) + _as_list(chapter_blueprint.get(key)):
            text = str(value or "").strip().lower()
            if text:
                values.add(text)
    return values


def _layout_type(package: Dict[str, Any], chapter_blueprint: Optional[Dict[str, Any]] = None) -> str:
    if _as_list(package.get("conflicts")):
        return "metric_reconciliation"
    proof_roles = {
        str(item.get("proof_role") or "").strip().lower()
        for collection in ("core_evidence", "supporting_evidence", "sample_evidence", "table_evidence")
        for item in _as_list(package.get(collection))
        if isinstance(item, dict)
    }
    if "counter" in proof_roles:
        return "argument_with_boundary"
    if "case" in proof_roles:
        return "case_argument"
    shape = _evidence_shape(package, chapter_blueprint)
    if shape.intersection({"policy", "policy_regulation", "official_definition"}):
        return "transmission_chain"
    if shape.intersection({"technology", "technology_product", "industry_chain", "capacity", "cost"}):
        return "mechanism_map"
    if shape.intersection({"competition", "player", "customer_case"}):
        return "case_argument"
    if shape.intersection({"capital", "capital_market", "filing_company", "market_price", "transaction"}):
        return "signal_validation"
    if shape.intersection({"risk", "counter", "counter_evidence"}):
        return "argument_with_boundary"
    if "metric" in proof_roles or len(_as_list(package.get("table_evidence"))) >= 2:
        return "evidence_matrix"
    if len(_as_list(package.get("table_evidence"))) >= 2:
        return "evidence_matrix"
    return "argument_first"


def _table_type(layout_type: str, package: Dict[str, Any]) -> str:
    if layout_type == "metric_reconciliation":
        return "metric_reconciliation"
    proof_roles = {
        str(item.get("proof_role") or "").strip().lower()
        for item in _as_list(package.get("table_evidence"))
        if isinstance(item, dict)
    }
    blob = _text_blob(package).lower()
    if proof_roles.intersection({"metric", "financial_metric", "market_price"}) or re.search(r"cagr|tam|sam|som|market|size|growth|\d", blob):
        return "market_metric_table"
    if proof_roles.intersection({"technology_product", "technology"}) or re.search(r"technology|patent|roadmap|yield|mass production|chip|model|compute", blob):
        return "technology_roadmap"
    if proof_roles.intersection({"counter", "counter_evidence", "risk"}) or re.search(r"risk|delay|lawsuit|regulation|failure|downside", blob):
        return "risk_register"
    if proof_roles.intersection({"filing", "company_filing", "capital", "financial_metric"}) or re.search(r"investment|valuation|ipo|funding|revenue|margin", blob):
        return "investment_priority_table"
    if "case" in proof_roles:
        return "customer_painpoint_matrix"
    if proof_roles.intersection({"competition", "player", "customer_case"}) or re.search(r"competitor|competition|market share|player", blob):
        return "competitor_matrix"
    return "evidence_matrix"


def _table_role(table_type: str) -> str:
    table_type = str(table_type or "").strip()
    if table_type in {"risk_register"}:
        return "risk_boundary_table"
    if table_type in {"investment_priority_table"}:
        return "decision_support_table"
    if table_type in {"market_metric_table", "cagr_calculation", "metric_reconciliation"}:
        return "core_metric_table"
    if table_type in {"competitor_matrix", "technology_roadmap", "evidence_matrix"}:
        return "core_argument_table"
    return "supporting_table"


def _table_anchor_preferences(table_type: str) -> List[str]:
    table_type = str(table_type or "").strip()
    if table_type in {"market_metric_table", "cagr_calculation", "metric_reconciliation"}:
        return ["metric_reconciliation", "evidence_matrix", "thesis"]
    if table_type in {"competitor_matrix", "customer_painpoint_matrix"}:
        return ["competitive_positioning", "case_comparison", "evidence_matrix", "thesis"]
    if table_type in {"technology_roadmap", "technology_maturity"}:
        return ["technology_maturity", "mechanism_chain", "evidence_matrix", "thesis"]
    if table_type == "risk_register":
        return ["risk_trigger", "scenario_analysis", "verification_checklist"]
    if table_type == "investment_priority_table":
        return ["verification_checklist", "scenario_analysis", "thesis"]
    return ["evidence_matrix", "thesis"]


def _placement_slot(table_type: str, anchor_block_type: str) -> str:
    table_type = str(table_type or "").strip()
    anchor_block_type = str(anchor_block_type or "").strip()
    if table_type == "risk_register" or anchor_block_type == "risk_trigger":
        return "before_risk"
    if table_type == "investment_priority_table":
        return "before_decision"
    if table_type in {"technology_roadmap", "technology_maturity"} or anchor_block_type in {"technology_maturity", "mechanism_chain"}:
        return "after_mechanism"
    if table_type in {"market_metric_table", "cagr_calculation", "metric_reconciliation"}:
        return "after_evidence_matrix" if anchor_block_type != "thesis" else "after_thesis"
    if table_type in {"competitor_matrix", "customer_painpoint_matrix"}:
        return "after_evidence_matrix"
    return "after_evidence_matrix" if anchor_block_type != "thesis" else "after_thesis"


def _anchor_section_for_table(sections: Sequence[Dict[str, Any]], table_type: str) -> Dict[str, str]:
    preferences = _table_anchor_preferences(table_type)
    first_section: Dict[str, Any] = {}
    for section in sections:
        section = _as_dict(section)
        if not first_section:
            first_section = section
        block_type = str(section.get("block_type") or section.get("output_type") or "").strip()
        if block_type in preferences:
            return {
                "anchor_section_id": str(section.get("section_id") or "").strip(),
                "anchor_block_type": block_type,
            }
    if first_section:
        return {
            "anchor_section_id": str(first_section.get("section_id") or "").strip(),
            "anchor_block_type": str(first_section.get("block_type") or first_section.get("output_type") or "thesis").strip(),
        }
    return {"anchor_section_id": "", "anchor_block_type": ""}


def _table_planning(
    *,
    package: Dict[str, Any],
    chapter_blueprint: Dict[str, Any],
    layout_type: str,
    sections: Sequence[Dict[str, Any]],
    llm_client: Any = None,
) -> Dict[str, Any]:
    table_evidence = [item for item in _as_list(package.get("table_evidence")) if isinstance(item, dict)]
    all_items = _evidence_items(package)
    min_table_evidence = _env_int("REPORT_MIN_TABLE_EVIDENCE_PER_CHAPTER", 4, min_value=1, max_value=20)
    min_rows = _env_int("REPORT_MIN_TABLE_ROWS", 2, min_value=2, max_value=8)
    role_counts: Dict[str, int] = {}
    high_quality_refs = 0
    for item in all_items:
        role = str(item.get("proof_role") or item.get("evidence_role") or item.get("role") or "").strip().lower()
        if role:
            role_counts[role] = role_counts.get(role, 0) + 1
        if str(item.get("source_level") or "").strip().upper() in {"A", "B"} and (
            item.get("ref") or item.get("evidence_id") or item.get("source_ref") or _as_list(item.get("source_refs"))
        ):
            high_quality_refs += 1
    table_type = _table_type(layout_type, package)
    structural_need = bool(_as_list(package.get("conflicts"))) or layout_type in {"metric_reconciliation", "evidence_matrix", "mechanism_map", "case_argument", "argument_with_boundary"}
    comparison_need = any(role_counts.get(role, 0) >= 2 for role in ("metric", "financial_metric", "case", "technology_product", "counter", "filing"))
    enough_rows = len(table_evidence) >= min_table_evidence or high_quality_refs >= min_rows + 1
    need_table = bool(enough_rows and (structural_need or comparison_need or len(table_evidence) >= min_table_evidence + 1))
    why_table_needed = ""
    why_no_table = ""
    if need_table:
        why_table_needed = "本章存在多个可比对象或指标，表格能把口径、时间和判断差异放在同一视图里，直接支撑核心结论。"
    elif len(table_evidence) < min_table_evidence and high_quality_refs < min_rows + 1:
        why_no_table = "高质量可追溯证据不足，强行出表会把证据缺口伪装成结论。"
    elif not (structural_need or comparison_need):
        why_no_table = "本章主要解释机制和因果链，正文叙述比表格更能保留判断逻辑。"
    else:
        why_no_table = "现有证据放入表格后不会增加专业密度，优先保留在段落或附录线索中。"
    anchor = _anchor_section_for_table(sections, table_type)
    slot = _placement_slot(table_type, anchor.get("anchor_block_type", ""))
    return {
        "planner_source": "layout_rules",
        "llm_planning_requested": bool(llm_client),
        "need_table": need_table,
        "why_table_needed": why_table_needed,
        "why_no_table": why_no_table,
        "table_type": table_type,
        "table_role": _table_role(table_type),
        "placement_slot": slot,
        "anchor_section_id": anchor.get("anchor_section_id", ""),
        "anchor_block_type": anchor.get("anchor_block_type", ""),
        "minimum_rows": min_rows,
        "maximum_rows": _env_int("REPORT_MAX_BODY_TABLE_ROWS", 8, min_value=2, max_value=30),
        "render_priority": 80 if table_type in {"market_metric_table", "metric_reconciliation", "competitor_matrix", "technology_roadmap"} else 60,
        "required_evidence_roles": _dedupe([role for role, count in role_counts.items() if count], limit=6),
        "minimum_source_level": "B",
        "evidence_count": len(table_evidence),
        "high_quality_ref_count": high_quality_refs,
    }


def _has_financial_evidence(package: Dict[str, Any]) -> bool:
    financial_re = re.compile(r"营收|收入|利润|毛利|现金流|净利|亏损|费用率|PE|PS|估值|财报|filing|financial", re.I)
    for item in _evidence_items(package):
        blob = " ".join(
            str(item.get(key) or "")
            for key in ("proof_role", "evidence_role", "source_type", "metric", "fact", "title", "content", "summary")
        )
        if financial_re.search(blob):
            return True
    return False


_LAYOUT_SECTION_LABELS = {
    "argument_first": "核心判断与证据边界",
    "metric_reconciliation": "指标口径与可比性",
    "case_argument": "案例对照",
    "argument_with_boundary": "成立条件",
    "transmission_chain": "影响路径",
    "mechanism_map": "影响路径与约束",
    "signal_validation": "可验证信号",
    "evidence_matrix": "事实依据",
}


def _section_title(package: Dict[str, Any], layout_type: str) -> str:
    """Build the H3 subsection title.

    Previously prepended the full chapter_title to every label, e.g.
        "规模、增速和价格信号是否支持机会判断？的指标口径与可比性"
    which (a) read like a glued phrase and (b) blew up when chapter_title was
    actually a long research question. Now we just use the layout's canonical
    label so headings stay short and consistent across chapters.
    """
    label = _LAYOUT_SECTION_LABELS.get(layout_type)
    if label:
        return label
    question = _compact(package.get("chapter_question"), 120)
    if question:
        return question
    return "核心判断"


def _refs(items: Sequence[Dict[str, Any]], *, limit: int = 5) -> List[str]:
    return _dedupe([item.get("ref") or item.get("evidence_id") for item in items if isinstance(item, dict)], limit=limit)


def _evidence_items(package: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        item
        for collection in ("core_evidence", "supporting_evidence", "sample_evidence", "table_evidence", "clue_evidence")
        for item in _as_list(package.get(collection))
        if isinstance(item, dict)
    ]


def _refs_for_roles(package: Dict[str, Any], roles: Sequence[Any], *, fallback: Sequence[str], limit: int = 8) -> List[str]:
    wanted = {str(role or "").strip().lower() for role in roles if str(role or "").strip()}
    if not wanted:
        return list(fallback)[:limit]
    aliases = {
        "counter_evidence": "counter",
        "official_data": "support",
        "policy_original": "policy",
        "company_filing": "filing",
        "customer_case": "case",
        "technology_product": "technology",
        "financial_metric": "metric",
    }
    expanded = set(wanted)
    expanded.update(aliases.get(role, role) for role in wanted)
    refs: List[str] = []
    for item in _evidence_items(package):
        role_values = {
            str(item.get("proof_role") or "").strip().lower(),
            str(item.get("evidence_role") or "").strip().lower(),
            str(item.get("role") or "").strip().lower(),
            str(item.get("intent") or "").strip().lower(),
        }
        source_level = str(item.get("source_level") or "").strip().upper()
        if source_level in {"A", "B"}:
            role_values.add("support")
        text_blob = " ".join(str(item.get(key) or "") for key in ("evidence_type", "source_type", "metric", "fact", "title")).lower()
        if role_values.intersection(expanded) or any(role in text_blob for role in expanded if len(role) > 3):
            ref = str(item.get("ref") or item.get("evidence_id") or "").strip()
            if ref:
                refs.append(ref)
    return _dedupe(refs, limit=limit) or list(fallback)[:limit]


def _section_for_block(package: Dict[str, Any], block: Dict[str, Any], *, index: int, fallback_refs: Sequence[str]) -> Dict[str, Any]:
    chapter_id = str(package.get("chapter_id") or "chapter")
    block_type = str(block.get("block_type") or "thesis").strip()
    original_block_type = block_type
    if block_type == "unit_economics" and not _has_financial_evidence(package):
        block_type = "signal_validation"
    refs = _refs_for_roles(package, _as_list(block.get("required_evidence_roles")), fallback=fallback_refs, limit=8)
    raw_title = block.get("title") if original_block_type == block_type else ""
    title = _compact(raw_title or _section_title(package, block_type), 120)
    if block_type == "thesis":
        title = _section_title(package, _layout_type(package))
    text_by_type = {
        "policy_timeline": "按政策原文、执行节点和影响对象组织证据，避免只复述政策表态。",
        "value_chain_map": "按上游瓶颈、中游承接、下游需求和利润流向组织证据。",
        "customer_painpoint_matrix": "按采购主体、场景痛点、预算约束和替代方案组织证据。",
        "competitive_positioning": "按玩家位置、份额变化、壁垒和替代压力组织证据。",
        "technology_maturity": "按技术成熟度、量产验证、生态兼容和替代边界组织证据。",
        "unit_economics": "按收入质量、毛利、现金流和可持续性组织证据。",
        "risk_trigger": "把能够推翻章节判断的反向样本、指标恶化和执行偏差前置说明。",
        "verification_checklist": "把后续最需要跟踪的指标、来源和反证样本整理成验证清单。",
    }
    return {
        "section_id": str(block.get("block_id") or f"{chapter_id}_s{index}"),
        "section_title": title,
        "section_role": str(block.get("role") or block_type),
        "block_type": block_type,
        "required_evidence_refs": refs,
        "required_evidence_roles": _as_list(block.get("required_evidence_roles")),
        "output_type": block_type,
        "renderer": block.get("renderer"),
        "min_words": 180 if block_type in {"risk_trigger", "verification_checklist"} else 240,
        "render_blocks": [
            {
                "type": "paragraph",
                "label": "",
                "text": text_by_type.get(block_type) or package.get("chapter_question") or package.get("chapter_title") or "",
            },
            {
                "type": "evidence_list",
                "label": "关键证据",
                "evidence_refs": refs,
            },
        ],
    }


def run_micro_layout_agent(
    *,
    report_blueprint: Optional[Dict[str, Any]] = None,
    chapter_evidence_packages: Optional[Sequence[Dict[str, Any]]] = None,
    llm_client: Any = None,
) -> List[Dict[str, Any]]:
    """Create per-chapter expression plans from evidence packages."""

    report_blueprint = _as_dict(report_blueprint)
    blueprint_by_id = {
        str(chapter.get("chapter_id") or ""): chapter
        for chapter in _as_list(report_blueprint.get("chapters"))
        if isinstance(chapter, dict)
    }
    layouts: List[Dict[str, Any]] = []
    for package in list(chapter_evidence_packages or []):
        if not isinstance(package, dict):
            continue
        chapter_id = str(package.get("chapter_id") or f"chapter_{len(layouts) + 1}")
        chapter_blueprint = _as_dict(blueprint_by_id.get(chapter_id))
        block_plan = select_blocks_for_chapter(chapter_blueprint or package, profile=_as_dict(report_blueprint.get("layout_strategy")), evidence_package=package)
        layout_type = _layout_type(package, chapter_blueprint)
        core = [item for item in _as_list(package.get("core_evidence")) if isinstance(item, dict)]
        supporting = [item for item in _as_list(package.get("supporting_evidence")) if isinstance(item, dict)]
        conflicts = _as_list(package.get("conflicts"))
        missing = _as_list(package.get("missing_evidence"))
        core_refs = _refs(core[:8], limit=8)
        supporting_refs = _refs(supporting[:8], limit=8)
        fallback_refs = core_refs or supporting_refs
        sections: List[Dict[str, Any]] = [
            _section_for_block(package, block, index=index, fallback_refs=fallback_refs)
            for index, block in enumerate(block_plan, start=1)
        ]
        if conflicts and not any(section.get("block_type") == "risk_trigger" for section in sections):
            sections.append(
                {
                    "section_id": f"{chapter_id}_s2",
                    "section_title": "判断边界",
                    "section_role": "counter_or_boundary",
                    "required_evidence_refs": _dedupe(
                        [ref for conflict in conflicts for ref in _as_list(_as_dict(conflict).get("evidence_refs"))],
                        limit=6,
                    ),
                    "output_type": "boundary",
                    "min_words": 160,
                    "render_blocks": [
                        {
                            "type": "paragraph",
                            "label": "判断边界",
                            "text": "同一指标存在不同口径或相反信号时，正文只保留可解释的判断范围。",
                        }
                    ],
                }
            )
        elif missing and not any(section.get("block_type") == "verification_checklist" for section in sections):
            sections.append(
                {
                    "section_id": f"{chapter_id}_s2",
                    "section_title": "证据边界",
                    "section_role": "evidence_gap",
                    "required_evidence_refs": [],
                    "output_type": "gap",
                    "min_words": 120,
                    "render_blocks": [
                        {
                            "type": "paragraph",
                            "label": "判断边界",
                            "text": "本章只保留已经能被证据支持的判断，暂不把缺口扩写成结论。",
                        }
                    ],
                }
            )
        table_requests: List[Dict[str, Any]] = []
        if False and len(_as_list(package.get("table_evidence"))) >= int(os.getenv("REPORT_MIN_TABLE_EVIDENCE_PER_CHAPTER", "4")):
            table_type = _table_type(layout_type, package)
            table_requests.append(
                {
                    "table_id": f"{chapter_id}_t1",
                    "table_type": table_type,
                    "purpose": f"用结构化方式回答：{package.get('chapter_question') or package.get('chapter_title')}",
                    "required_evidence_refs": _refs(_as_list(package.get("table_evidence"))[:8], limit=8),
                }
            )
        table_plan = _table_planning(
            package=package,
            chapter_blueprint=chapter_blueprint,
            layout_type=layout_type,
            sections=sections,
            llm_client=llm_client,
        )
        table_requests = []
        if table_plan.get("need_table"):
            table_requests.append(
                {
                    "table_id": f"{chapter_id}_t1",
                    "need_table": True,
                    "table_type": table_plan.get("table_type") or "evidence_matrix",
                    "table_role": table_plan.get("table_role") or "core_argument_table",
                    "purpose": f"用结构化表格支撑本章判断：{package.get('chapter_question') or package.get('chapter_title')}",
                    "why_table_needed": table_plan.get("why_table_needed"),
                    "why_no_table": "",
                    "placement_slot": table_plan.get("placement_slot") or "after_evidence_matrix",
                    "anchor_section_id": table_plan.get("anchor_section_id"),
                    "anchor_block_type": table_plan.get("anchor_block_type"),
                    "required_evidence_roles": table_plan.get("required_evidence_roles") or [],
                    "minimum_source_level": table_plan.get("minimum_source_level") or "B",
                    "minimum_rows": table_plan.get("minimum_rows") or 2,
                    "maximum_rows": table_plan.get("maximum_rows") or 8,
                    "render_priority": table_plan.get("render_priority") or 50,
                    "fallback_if_invalid": "demote_to_narrative",
                    "required_evidence_refs": _refs(_as_list(package.get("table_evidence"))[:8], limit=8),
                }
            )
        followups = []
        for gap in missing:
            gap = _as_dict(gap)
            suggestion = _compact(gap.get("suggestion"), 180)
            if suggestion:
                followups.append(
                    {
                        "query": f"{chapter_blueprint.get('chapter_question') or package.get('chapter_question') or package.get('chapter_title')} {suggestion}",
                        "targets_gap": suggestion,
                        "agent": "iqs",
                    }
                )
        layouts.append(
            {
                "agent": AGENT_NAME,
                "chapter_id": chapter_id,
                "chapter_title": package.get("chapter_title"),
                "layout_type": layout_type,
                "blocks": block_plan,
                "sections": sections,
                "table_planning": table_plan,
                "table_requests": table_requests,
                "follow_up_queries": followups,
            }
        )
    return layouts
