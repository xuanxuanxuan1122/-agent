from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Sequence

from .markdown_renderer import (
    collect_format_warnings,
    normalize_markdown_spacing,
    render_appendix,
    render_chapter_package,
    render_cover,
    render_decision_package,
    render_executive_summary,
    render_risk_package,
    strip_body_qa_leaks,
    strip_internal_layout_language,
)
from .public_report_sanitizer import has_internal_gap_language, rewrite_internal_gap_language, sanitize_public_markdown


AGENT_NAME = "final_writer_agent"
AGENT_DESCRIPTION = "Final Writer Agent. Only composes structured packages and renders Markdown."
PUBLIC_SECTION_KEYS = {
    "section_id",
    "section_title",
    "claim",
    "reasoning",
    "mechanism",
    "counter_evidence",
    "actionable",
    "decision_implication",
    "what_to_verify_next",
    "confidence",
    "evidence_refs",
    "render_blocks",
    "public_render",
}

SUMMARY_BLOCKS = {
    "executive_summary",
    "key_judgments",
    "deal_snapshot",
    "investment_conclusion",
    "policy_summary",
    "impact_judgment",
    "entry_decision_snapshot",
    "market_snapshot",
    "competitive_snapshot",
    "consumer_opportunity_snapshot",
    "supply_chain_snapshot",
    "technology_readiness_snapshot",
    "briefing_summary",
}

DECISION_BLOCKS = {
    "strategic_options",
    "entry_recommendation",
    "investment_recommendation",
    "product_opportunity",
    "resilience_options",
    "adoption_path",
}

RISK_BLOCKS = {"risk_triggers", "red_flags", "execution_risks"}
WATCHLIST_BLOCKS = {"verification_checklist", "monitoring_indicators", "dd_checklist"}

GLOBAL_BLOCK_TITLES = {
    "executive_summary": "核心观点与主要结论",
    "key_judgments": "关键判断",
    "key_data": "关键数据",
    "deal_snapshot": "交易速览",
    "investment_conclusion": "投资结论",
    "policy_summary": "政策摘要",
    "impact_judgment": "影响判断",
    "entry_decision_snapshot": "进入决策速览",
    "market_snapshot": "市场速览",
    "competitive_snapshot": "竞争速览",
    "consumer_opportunity_snapshot": "消费机会速览",
    "supply_chain_snapshot": "供应链速览",
    "technology_readiness_snapshot": "技术成熟度速览",
    "briefing_summary": "简报摘要",
    "strategic_options": "策略选择",
    "entry_recommendation": "进入建议",
    "investment_recommendation": "投资建议",
    "product_opportunity": "产品机会",
    "resilience_options": "韧性建设选项",
    "adoption_path": "落地路径",
    "risk_triggers": "风险提示与反向信号",
    "red_flags": "尽调红旗",
    "execution_risks": "执行风险",
    "verification_checklist": "验证清单",
    "monitoring_indicators": "监测指标",
    "dd_checklist": "尽调清单",
    "appendix": "研究口径与来源",
}


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def _source_appendix_enabled() -> bool:
    return _env_flag("REPORT_FINAL_WRITER_SOURCE_APPENDIX", False)


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _public_text(value: Any) -> str:
    text = rewrite_internal_gap_language(str(value or "").strip())
    return "" if has_internal_gap_language(text) else text


def _public_section(section: Dict[str, Any]) -> Dict[str, Any]:
    copied = {key: section.get(key) for key in PUBLIC_SECTION_KEYS if key in section}
    for key in ["section_title", "claim", "reasoning", "mechanism", "counter_evidence", "actionable", "decision_implication", "confidence"]:
        if key in copied:
            copied[key] = _public_text(copied.get(key))
    copied["what_to_verify_next"] = [
        item
        for item in (_public_text(value) for value in _as_list(copied.get("what_to_verify_next")))
        if item
    ]
    copied_blocks = []
    for block in (_as_dict(item) for item in _as_list(copied.get("render_blocks"))):
        if not str(block.get("type") or "").strip():
            continue
        block = dict(block)
        block["label"] = _public_text(block.get("label"))
        block["text"] = _public_text(block.get("text"))
        copied_blocks.append(block)
    copied["render_blocks"] = copied_blocks
    return copied


def _section_has_public_content(section: Dict[str, Any]) -> bool:
    visible = " ".join(
        str(section.get(key) or "")
        for key in ["section_title", "claim", "reasoning", "mechanism", "counter_evidence", "actionable", "decision_implication"]
    )
    visible = " ".join([visible, *[str(_as_dict(block).get("text") or "") for block in _as_list(section.get("render_blocks"))]])
    visible = " ".join([visible, *[str(item) for item in _as_list(section.get("what_to_verify_next"))]])
    return bool(visible.strip()) and not has_internal_gap_language(visible)


def _all_table_packages(chapter_packages: Sequence[Dict[str, Any]], explicit: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = [item for item in explicit if isinstance(item, dict)]
    seen = {str(item.get("table_id") or id(item)) for item in result}
    for chapter in chapter_packages:
        if not isinstance(chapter, dict):
            continue
        for table in _as_list(chapter.get("table_packages")):
            if not isinstance(table, dict):
                continue
            key = str(table.get("table_id") or id(table))
            if key in seen:
                continue
            seen.add(key)
            result.append(table)
    return result


def _renumber_sources_by_first_citation(
    markdown: str,
    source_registry: Sequence[Dict[str, Any]],
) -> tuple[str, List[Dict[str, Any]]]:
    sources = [dict(item) for item in list(source_registry or []) if isinstance(item, dict)]
    if not markdown or not sources:
        return markdown, sources
    by_ref = {str(source.get("ref") or "").strip(): source for source in sources if str(source.get("ref") or "").strip()}
    if not by_ref:
        return markdown, sources
    seen_refs: List[str] = []
    for match in re.finditer(r"\[(\d{1,3})\]", markdown):
        ref = f"[{match.group(1)}]"
        if ref in by_ref and ref not in seen_refs:
            seen_refs.append(ref)
    ordered_refs = seen_refs + [str(source.get("ref") or "").strip() for source in sources if str(source.get("ref") or "").strip() not in seen_refs]
    mapping = {old_ref: f"[{index}]" for index, old_ref in enumerate(ordered_refs, start=1)}

    def replace_ref(match: re.Match[str]) -> str:
        ref = f"[{match.group(1)}]"
        return mapping.get(ref, "")

    rewritten_markdown = re.sub(r"\[(\d{1,3})\]", replace_ref, markdown)
    ordered_sources: List[Dict[str, Any]] = []
    for index, old_ref in enumerate(ordered_refs, start=1):
        source = dict(by_ref.get(old_ref) or {})
        if not source:
            continue
        source["ref"] = f"[{index}]"
        source["source_id"] = f"SRC-{index:03d}"
        ordered_sources.append(source)
    return rewritten_markdown, ordered_sources


def should_render_chapter(chapter: Dict[str, Any]) -> bool:
    if chapter.get("omit_from_report"):
        return False

    lead = _public_text(chapter.get("lead"))

    sections = [
        public_section
        for item in _as_list(chapter.get("sections"))
        if isinstance(item, dict)
        and not item.get("omit_from_report")
        for public_section in [_public_section(item)]
        if _section_has_public_content(public_section)
    ]
    tables = [
        item
        for item in _as_list(chapter.get("table_packages"))
        if isinstance(item, dict) and item.get("should_render") and not item.get("appendix_only")
    ]
    return bool(lead or sections or tables)


def _public_chapter(chapter: Dict[str, Any]) -> Dict[str, Any]:
    copied = dict(chapter)
    copied["sections"] = [
        public_section
        for section in _as_list(chapter.get("sections"))
        if isinstance(section, dict)
        and not section.get("omit_from_report")
        for public_section in [_public_section(section)]
        if _section_has_public_content(public_section)
    ]
    copied["table_packages"] = [
        table
        for table in _as_list(chapter.get("table_packages"))
        if isinstance(table, dict) and table.get("should_render") and not table.get("appendix_only")
    ]
    copied["lead"] = _public_text(copied.get("lead"))
    return copied


def should_render_key_data(table_packages: Sequence[Dict[str, Any]]) -> bool:
    return any(
        isinstance(table, dict)
        and table.get("should_render")
        and not table.get("appendix_only")
        and len(_as_list(table.get("rows"))) >= 2
        for table in list(table_packages or [])
    )


def _as_block_list(value: Any, fallback: Sequence[str]) -> List[str]:
    blocks = [str(item or "").strip() for item in _as_list(value) if str(item or "").strip()]
    return blocks or list(fallback)


def _rename_first_h2(markdown: str, title: str) -> str:
    markdown = str(markdown or "").strip()
    title = str(title or "").strip()
    if not markdown or not title:
        return markdown
    if markdown.startswith("## "):
        return markdown.replace(markdown.splitlines()[0], f"## {title}", 1)
    return f"## {title}\n{markdown}"


def _render_key_data_block(title: str, decision_package: Dict[str, Any], table_packages: Sequence[Dict[str, Any]]) -> str:
    rows: List[str] = []
    for table in table_packages:
        table = _as_dict(table)
        if not table.get("should_render") or table.get("appendix_only"):
            continue
        for row in _as_list(table.get("rows"))[:1]:
            cells = _as_list(_as_dict(row).get("cells"))
            if cells:
                text = "；".join(str(cell or "").strip() for cell in cells[:3] if str(cell or "").strip())
                if text:
                    rows.append(text)
    if not rows:
        return ""
    return "\n".join([f"## {title}", *[f"- {item}" for item in rows[:6]]])


def _render_watchlist_block(title: str, decision_package: Dict[str, Any]) -> str:
    rows: List[str] = []
    for item in _as_list(decision_package.get("watchlist"))[:8]:
        metric = _public_text(_as_dict(item).get("metric"))
        if metric:
            rows.append(f"- {metric}")
    for item in _as_list(decision_package.get("abandon_conditions"))[:4]:
        condition = _public_text(_as_dict(item).get("condition"))
        if condition:
            rows.append(f"- 反证：{condition}")
    if not rows:
        return ""
    return "\n".join([f"## {title}", *rows])


def _render_global_block(
    block_key: str,
    *,
    title_override: str = "",
    decision_package: Dict[str, Any],
    risk_package: Dict[str, Any],
    appendix_package: Dict[str, Any],
    source_registry: Sequence[Dict[str, Any]],
    table_packages: Sequence[Dict[str, Any]],
    rendered_groups: set[str],
) -> str:
    key = str(block_key or "").strip()
    title = title_override or GLOBAL_BLOCK_TITLES.get(key, key.replace("_", " ").strip().title())
    if key in SUMMARY_BLOCKS:
        if "summary" in rendered_groups:
            return ""
        rendered_groups.add("summary")
        return _rename_first_h2(render_executive_summary(decision_package, table_packages), title)
    if key == "key_data":
        if "key_data" in rendered_groups or "summary" in rendered_groups:
            return ""
        rendered_groups.add("key_data")
        return _render_key_data_block(title, decision_package, table_packages)
    if key in DECISION_BLOCKS:
        if "decision" in rendered_groups:
            return ""
        rendered_groups.add("decision")
        return _rename_first_h2(render_decision_package(decision_package), title)
    if key in RISK_BLOCKS:
        if "risk" in rendered_groups:
            return ""
        rendered_groups.add("risk")
        return _rename_first_h2(render_risk_package(risk_package), title)
    if key in WATCHLIST_BLOCKS:
        if "watchlist" in rendered_groups:
            return ""
        rendered_groups.add("watchlist")
        return _render_watchlist_block(title, decision_package)
    if key == "appendix":
        if not _source_appendix_enabled():
            return ""
        if "appendix" in rendered_groups:
            return ""
        rendered_groups.add("appendix")
        return _rename_first_h2(render_appendix(source_registry, appendix_package), title)
    return ""


def run_final_writer_agent(
    *,
    query: str = "",
    report_blueprint: Optional[Dict[str, Any]] = None,
    chapter_packages: Optional[Sequence[Dict[str, Any]]] = None,
    table_packages: Optional[Sequence[Dict[str, Any]]] = None,
    decision_package: Optional[Dict[str, Any]] = None,
    risk_package: Optional[Dict[str, Any]] = None,
    appendix_package: Optional[Dict[str, Any]] = None,
    source_registry: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    report_blueprint = _as_dict(report_blueprint)
    chapter_packages = [item for item in list(chapter_packages or []) if isinstance(item, dict)]
    public_chapters = [_public_chapter(chapter) for chapter in chapter_packages if should_render_chapter(chapter)]
    table_packages = _all_table_packages(public_chapters, [item for item in list(table_packages or []) if isinstance(item, dict)])
    table_packages = [
        table
        for table in table_packages
        if isinstance(table, dict) and table.get("should_render") and not table.get("appendix_only")
    ]
    decision_package = _as_dict(decision_package)
    decision_package = {
        **decision_package,
        "chapter_syntheses": [
            {
                "chapter_title": chapter.get("chapter_title"),
                "chapter_question": chapter.get("chapter_question"),
                "chapter_summary": _as_dict(chapter.get("chapter_summary")),
            }
            for chapter in public_chapters
            if _as_dict(chapter.get("chapter_summary"))
        ],
    }
    risk_package = _as_dict(risk_package)
    appendix_package = _as_dict(appendix_package)
    source_registry = [item for item in list(source_registry or []) if isinstance(item, dict)]

    shell = _as_dict(report_blueprint.get("report_shell"))
    front_blocks = _as_block_list(shell.get("front_blocks"), ["executive_summary", "key_data"])
    back_blocks = _as_block_list(shell.get("back_blocks"), ["strategic_options", "risk_triggers"])
    summary_title_key = next(
        (block for block in front_blocks if block in SUMMARY_BLOCKS and block not in {"executive_summary", "key_judgments"}),
        front_blocks[0] if front_blocks else "executive_summary",
    )
    parts = [render_cover(query, report_blueprint)]
    rendered_groups: set[str] = set()
    front_section_titles: List[str] = []
    back_section_titles: List[str] = []
    for block_key in front_blocks:
        rendered = _render_global_block(
            block_key,
            title_override=GLOBAL_BLOCK_TITLES.get(summary_title_key, "") if block_key in SUMMARY_BLOCKS else "",
            decision_package=decision_package,
            risk_package=risk_package,
            appendix_package=appendix_package,
            source_registry=source_registry,
            table_packages=table_packages,
            rendered_groups=rendered_groups,
        )
        if rendered:
            parts.append(rendered)
            front_section_titles.append(GLOBAL_BLOCK_TITLES.get(summary_title_key if block_key in SUMMARY_BLOCKS else block_key, block_key))
    for index, chapter in enumerate(public_chapters, start=1):
        parts.append(
            render_chapter_package(
                chapter,
                index,
                previous_chapter=public_chapters[index - 2] if index > 1 else None,
                next_chapter=public_chapters[index] if index < len(public_chapters) else None,
            )
        )
    appendix_requested = False
    for block_key in back_blocks:
        if str(block_key or "").strip() == "appendix":
            appendix_requested = _source_appendix_enabled()
            continue
        rendered = _render_global_block(
            block_key,
            title_override="",
            decision_package=decision_package,
            risk_package=risk_package,
            appendix_package=appendix_package,
            source_registry=source_registry,
            table_packages=table_packages,
            rendered_groups=rendered_groups,
        )
        if rendered:
            parts.append(rendered)
            back_section_titles.append(GLOBAL_BLOCK_TITLES.get(block_key, block_key))
    body_markdown = "\n\n".join(part for part in parts if str(part or "").strip())
    body_markdown, source_registry = _renumber_sources_by_first_citation(body_markdown, source_registry)
    parts = [body_markdown]
    if appendix_requested and _source_appendix_enabled():
        rendered = _render_global_block(
            "appendix",
            title_override="",
            decision_package=decision_package,
            risk_package=risk_package,
            appendix_package=appendix_package,
            source_registry=source_registry,
            table_packages=table_packages,
            rendered_groups=rendered_groups,
        )
        if rendered:
            parts.append(rendered)
            back_section_titles.append(GLOBAL_BLOCK_TITLES.get("appendix", "appendix"))
    markdown = "\n\n".join(part for part in parts if str(part or "").strip())
    markdown = strip_internal_layout_language(markdown)
    markdown = strip_body_qa_leaks(markdown)
    markdown = normalize_markdown_spacing(markdown)
    markdown = sanitize_public_markdown(markdown)
    markdown = normalize_markdown_spacing(markdown)
    warnings = collect_format_warnings(markdown)
    return {
        "agent": AGENT_NAME,
        "report_markdown": markdown,
        "sections": [
            *front_section_titles,
            *[str(chapter.get("chapter_title") or "") for chapter in public_chapters],
            *back_section_titles,
        ],
        "source_count": len(source_registry),
        "source_registry": list(source_registry),
        "estimated_chars": len(markdown),
        "format_warnings": warnings,
    }
