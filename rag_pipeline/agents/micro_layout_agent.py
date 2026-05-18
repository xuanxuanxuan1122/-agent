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
    if "case" in proof_roles:
        return "customer_painpoint_matrix"
    return "evidence_matrix"


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
    "evidence_matrix": "关键事实对照",
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

    del llm_client
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
        if len(_as_list(package.get("table_evidence"))) >= int(os.getenv("REPORT_MIN_TABLE_EVIDENCE_PER_CHAPTER", "4")):
            table_type = _table_type(layout_type, package)
            table_requests.append(
                {
                    "table_id": f"{chapter_id}_t1",
                    "table_type": table_type,
                    "purpose": f"用结构化方式回答：{package.get('chapter_question') or package.get('chapter_title')}",
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
                "table_requests": table_requests,
                "follow_up_queries": followups,
            }
        )
    return layouts
