from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterable, List, Sequence

from .public_report_sanitizer import has_internal_gap_language, rewrite_internal_gap_language


INTERNAL_LAYOUT_PHRASES = [
    "章节判断",
    "关键事实速览",
    "证据深读",
    "原文事实",
    "本章核心判断",
    "本章结论",
    "本章小结",
    "图表解读",
    "报告使用方式",
    "进入综合决策章的变量",
    "全球口径",
    "中国口径",
    "增速口径",
    "可引用事实",
    "机制与边界",
    "反证边界",
    "核心判断",
    "关键判断",
    "证据依据",
    "传导链条",
    "判断边界",
    "决策含义",
    "本章综合分析",
    "机制拆解与变量联动",
    "反证、边界与结论失效条件",
    "决策含义与后续观察优先级",
    "关联证据",
    "章节关系与参考分析",
]

INTERNAL_SECTION_TITLE_PATTERNS = [
    r"机制拆解",
    r"变量联动",
    r"反证",
    r"结论失效",
    r"决策含义",
    r"后续观察",
    r"全球口径",
    r"中国口径",
    r"增速口径",
    r"可引用事实",
    r"机制与边界",
    r"反证边界",
    r"核心论证",
    r"形成可验证判断",
    r"本章综合分析",
]

INTERNAL_BLOCK_LABELS = {
    "关键判断",
    "证据依据",
    "传导链条",
    "边界",
    "含义",
    "观察点",
    "关键证据",
    "判断边界",
    "核心判断",
    "主要结论",
    "资料支撑",
    "变量传导",
    "适用边界",
    "后续动作",
    "后续影响",
    "可引用事实",
    "机制与边界",
    "进入综合决策章的变量",
    "全球口径",
    "中国口径",
    "增速口径",
}

PUBLIC_TERM_REPLACEMENTS = {
    "本章核心判断": "主要结论",
    "核心判断": "主要结论",
    "关键判断": "主要结论",
    "证据依据": "资料支撑",
    "传导链条": "影响路径",
    "判断边界": "适用边界",
    "决策含义": "策略影响",
    "行动含义": "后续动作",
    "判断含义": "后续影响",
    "本章综合分析": "",
    "机制拆解与变量联动": "影响路径与约束关系",
    "反证、边界与结论失效条件": "反向条件与结论弹性",
    "决策含义与后续观察优先级": "策略影响与观察优先级",
}

PUBLIC_PROCESS_REWRITES = [
    (r"该信号需要同时穿过场景、主体和口径三层约束，才能从单点事实变成可复制结论。材料中已经出现的可观察事实是[:：]", "公开材料显示："),
    (r"当前可用事实包括[:：]", "公开材料显示："),
    (r"把反向触发器写入验证清单，并在新增证据改变口径时重新排序(?:章节)?结论。?", "后续应重点观察反向信号，并在口径变化时校准判断。"),
    (r"建议动作[:：]", "策略建议："),
    (r"材料中最有解释力的事实组合是[:：]", "公开材料显示："),
    (r"当前事实组合是[:：]", "公开材料显示："),
    (r"这些事实需要按供应链层级拆开理解[:：]", "可按供应链层级理解："),
    (r"围绕“([^”]+)”，讨论应从", r"围绕“\1”，分析可从"),
    (r"围绕“([^”]+)”，讨论从事实组合开始，再转入成立条件和相反情形。公开材料显示[:：]", r"围绕“\1”，分析先看已经出现的产业信号，再看成立条件和反向情形。公开材料显示："),
    (r"后续跟踪应集中在", "后续重点观察"),
    (r"后续跟踪的重点落在", "后续重点观察"),
    (r"后续跟踪集中在", "后续重点观察"),
    (r"章节结论才适合上升为全篇主线", "这一判断才更适合成为全文主线"),
    (r"章节结论才会进入全篇主线", "这一判断才会进入全文主线"),
    (r"章节结论", "判断"),
    (r"本章可用来源约\d+条[，。]?", ""),
    (r"A/B层级来源约\d+条[，。]?", ""),
    (r"反向或边界信号约\d+条[，。]?", ""),
    (r"来源层级分布为[^。；\n]*[。；]?", ""),
    (r"本章写作时应", ""),
    (r"分析需要先", ""),
    (r"当前最直接的支持点是[:：]", "材料显示："),
    (r"当前可用于判断的事实组合包括[:：]", "可以放在一起观察的事实包括："),
    (r"围绕“([^”]+)”形成可验证判断", r"\1"),
    (r"进入总判断", "进入全篇主线"),
    (r"补证任务", "持续观察项"),
    (r"公开表达采用相应边界", "结论保留相应边界"),
    (r"关联证据[:：][^\n。]*[。]?", ""),
    (r"(?<![A-Za-z0-9_])EV-\d+(?:-[A-Za-z0-9]+)?", ""),
]


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _compact(value: Any, max_chars: int = 500) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "..."


def _public_text(value: Any, max_chars: int = 500) -> str:
    text = rewrite_internal_gap_language(_compact(value, max_chars))
    for old, new in PUBLIC_TERM_REPLACEMENTS.items():
        text = text.replace(old, new)
    for pattern, replacement in PUBLIC_PROCESS_REWRITES:
        text = re.sub(pattern, replacement, text, flags=re.I)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"([。；]){2,}", r"\1", text)
    text = text.strip(" \t\r\n，；")
    return "" if has_internal_gap_language(text) else text


def _is_internal_section_title(value: Any) -> bool:
    text = _compact(value, 160)
    return bool(text and any(re.search(pattern, text) for pattern in INTERNAL_SECTION_TITLE_PATTERNS))


def _natural_transition(prefix: str, text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return ""
    if text.startswith(("因此", "所以", "不过", "但", "如果", "需要注意", "从")):
        return text
    return f"{prefix}{text}"


def _dedupe(values: Iterable[Any], *, limit: int = 20, max_chars: int = 240) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        text = _compact(value, max_chars)
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


TOKEN_PROFILE_INT_DEFAULTS = {
    "lean": {
        "REPORT_RENDER_BLOCK_MAX_CHARS": 1800,
        "REPORT_SECTION_CLAIM_MAX_CHARS": 700,
        "REPORT_SECTION_REASONING_MAX_CHARS": 1800,
        "REPORT_SECTION_COUNTER_MAX_CHARS": 900,
        "REPORT_SECTION_ACTION_MAX_CHARS": 900,
        "REPORT_SECTION_MECHANISM_MAX_CHARS": 1400,
        "REPORT_RENDER_FACT_DIGEST_LIMIT": 0,
    },
    "balanced": {
        "REPORT_RENDER_BLOCK_MAX_CHARS": 2400,
        "REPORT_SECTION_CLAIM_MAX_CHARS": 900,
        "REPORT_SECTION_REASONING_MAX_CHARS": 2400,
        "REPORT_SECTION_COUNTER_MAX_CHARS": 1200,
        "REPORT_SECTION_ACTION_MAX_CHARS": 1200,
        "REPORT_SECTION_MECHANISM_MAX_CHARS": 1800,
        "REPORT_RENDER_FACT_DIGEST_LIMIT": 0,
    },
    "deep": {
        "REPORT_RENDER_BLOCK_MAX_CHARS": 3200,
        "REPORT_SECTION_CLAIM_MAX_CHARS": 1100,
        "REPORT_SECTION_REASONING_MAX_CHARS": 3200,
        "REPORT_SECTION_COUNTER_MAX_CHARS": 1600,
        "REPORT_SECTION_ACTION_MAX_CHARS": 1600,
        "REPORT_SECTION_MECHANISM_MAX_CHARS": 2400,
        "REPORT_RENDER_FACT_DIGEST_LIMIT": 0,
    },
}


def _profile_default(name: str, default: int) -> int:
    if os.getenv(name) is not None:
        return default
    profile = str(os.getenv("REPORT_TOKEN_PROFILE", "balanced") or "balanced").strip().lower()
    return TOKEN_PROFILE_INT_DEFAULTS.get(profile, TOKEN_PROFILE_INT_DEFAULTS["balanced"]).get(name, default)


def _env_int(name: str, default: int, *, min_value: int = 0, max_value: int = 10_000) -> int:
    default = _profile_default(name, default)
    try:
        value = int(os.getenv(name, str(default)) or default)
    except (TypeError, ValueError):
        value = default
    return max(min_value, min(max_value, value))


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _line_key(value: Any) -> str:
    return re.sub(r"[\s，。；：:、,.!?！？“”\"'（）()《》]+", "", str(value or "")).lower()


def _dedupe_narrative_lines(lines: Sequence[str]) -> List[str]:
    result: List[str] = []
    seen = set()
    for line in lines:
        text = str(line or "")
        stripped = text.strip()
        if not stripped:
            if result and result[-1].strip():
                result.append(text)
            continue
        if stripped.startswith(("#", "|", "**", "- [")) or re.match(r"^\|?\s*-{3,}", stripped):
            result.append(text)
            continue
        key = _line_key(stripped)
        if len(key) >= 36 and key in seen:
            continue
        if len(key) >= 36:
            seen.add(key)
        result.append(text)
    while result and not result[-1].strip():
        result.pop()
    return result


def strip_internal_layout_language(text: str) -> str:
    result = str(text or "")
    for old, new in PUBLIC_TERM_REPLACEMENTS.items():
        result = result.replace(old, new)
    for pattern, replacement in PUBLIC_PROCESS_REWRITES:
        result = re.sub(pattern, replacement, result, flags=re.I)
    for phrase in INTERNAL_LAYOUT_PHRASES:
        result = result.replace(phrase, "")
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def strip_body_qa_leaks(text: str) -> str:
    result = str(text or "")
    result = re.sub(r"(?im)^\s*(QA|Self[- ]?check|Validation|质量检查)[:：].*$", "", result)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def normalize_markdown_spacing(text: str) -> str:
    result = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    result = re.sub(r"[ \t]+\n", "\n", result)
    result = result.replace("。；", "；").replace("；。", "。").replace("。。", "。")
    result = re.sub(r"\n{3,}", "\n\n", result)
    result = re.sub(r"(?m)^(#{1,6})\s*", r"\1 ", result)
    return result.strip() + ("\n" if result.strip() else "")


def _report_title(research_object: str) -> str:
    title = _compact(research_object, 120).strip(" #")
    if not title:
        title = "研究对象"
    if re.search(r"(报告|研究|简报)$", title):
        return f"# {title}"
    return f"# {title}研究报告"


def _cover_title_from_blueprint(query: str, report_blueprint: Dict[str, Any]) -> str:
    brief = _as_dict(report_blueprint.get("article_brief"))
    explicit = (
        report_blueprint.get("report_title")
        or report_blueprint.get("display_title")
        or brief.get("display_title")
        or brief.get("main_title")
    )
    title = _compact(explicit, 140).strip(" #")
    if title:
        return f"# {title}"
    research_object = str(report_blueprint.get("research_object") or query or "研究对象").strip()
    return _report_title(research_object)


def _cover_subtitle_from_blueprint(report_blueprint: Dict[str, Any]) -> str:
    brief = _as_dict(report_blueprint.get("article_brief"))
    subtitle = _compact(
        report_blueprint.get("report_subtitle")
        or report_blueprint.get("display_subtitle")
        or brief.get("display_subtitle")
        or brief.get("direction"),
        220,
    ).strip()
    subtitle = re.sub(r"^[—–-]{1,3}\s*", "", subtitle).strip()
    return f"——{subtitle}" if subtitle else ""


def render_cover(query: str, report_blueprint: Dict[str, Any]) -> str:
    research_object = str(report_blueprint.get("research_object") or query or "研究对象").strip()
    narrative = _public_text(report_blueprint.get("narrative"), 240)
    lines = [_cover_title_from_blueprint(query, report_blueprint), ""]
    subtitle = _cover_subtitle_from_blueprint(report_blueprint)
    if subtitle:
        lines.extend([subtitle, ""])
    if narrative:
        lines.append(f"研究主线：{narrative}")
    return "\n".join(_dedupe_narrative_lines(lines))


def _public_tables(table_packages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        table
        for table in list(table_packages or [])
        if isinstance(table, dict)
        and table.get("should_render")
        and not table.get("appendix_only")
        and len(_as_list(table.get("rows"))) >= 2
    ]


def render_executive_summary(decision_package: Dict[str, Any], table_packages: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    judgments = [_as_dict(item) for item in _as_list(decision_package.get("core_judgments"))]
    judgment_lines = []
    for item in judgments[:5]:
        judgment = _public_text(item.get("judgment"), 260)
        label = _compact(item.get("label"), 40)
        if label in INTERNAL_BLOCK_LABELS or _is_internal_section_title(label):
            label = ""
        if judgment:
            judgment_lines.append(f"- {label + '：' if label else ''}{judgment}")
    if judgment_lines:
        lines.extend(["## 核心观点与主要结论", *judgment_lines])

    key_rows = []
    for table in _public_tables(table_packages):
        for row in _as_list(table.get("rows"))[:1]:
            row = _as_dict(row)
            cells = _as_list(row.get("cells"))
            if cells:
                text = "；".join(_compact(cell, 60) for cell in cells[:3] if str(cell).strip())
                text = _public_text(text, 220)
                if text:
                    key_rows.append(text)
    if key_rows:
        if lines:
            lines.append("")
        lines.extend(["## 关键数据", *[f"- {item}" for item in _dedupe(key_rows, limit=5)]])
    return "\n".join(_dedupe_narrative_lines(lines))


def _markdown_table(headers: Sequence[Any], rows: Sequence[Sequence[Any]]) -> str:
    headers = [_public_text(str(header or "").replace("|", "/").strip(), 80) for header in headers]
    if not headers or not rows:
        return ""
    cleaned_rows: List[List[str]] = []
    for row in rows:
        cells = []
        for cell in row:
            text = str(cell or "").replace("\n", " ").replace("|", "/").strip()
            text = re.sub(r"第\s*\d+\s*轮\s*[｜|:：]\s*", "", text)
            text = re.sub(r"(?:竞争对比|政策监管|技术产业链|市场规模|成本|金额)\s*=\s*(?=；|;|$)", "", text)
            text = re.sub(r"(?:；\s*){2,}", "；", text).strip(" ；;，,")
            if re.search(r"(?:第\s*\d+\s*轮|openai_task_\d+)", text, flags=re.I):
                text = ""
            cells.append(_public_text(text, 220))
        cells = (cells + [""] * len(headers))[: len(headers)]
        if any(cell.strip() for cell in cells):
            cleaned_rows.append(cells)
    if not cleaned_rows:
        return ""
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for cells in cleaned_rows:
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _is_internal_table_header(header: Any) -> bool:
    text = str(header or "").strip().lower()
    return bool(re.search(r"(来源|引用|证据|evidence|source|ref)", text))


def _public_table_shape(headers: Sequence[Any], rows: Sequence[Sequence[Any]]) -> tuple[List[Any], List[List[Any]]]:
    keep_indices = [index for index, header in enumerate(headers) if not _is_internal_table_header(header)]
    public_headers = [headers[index] for index in keep_indices]
    public_rows = []
    for row in rows:
        row_values = list(row)
        public_rows.append([row_values[index] if index < len(row_values) else "" for index in keep_indices])
    return public_headers, public_rows


def _line_with_citations(text: str, evidence_refs: Sequence[Any]) -> str:
    suffix = _citation_suffix(evidence_refs)
    if not suffix or re.search(r"\[\d{1,3}\]\s*$", text):
        return text
    return text.rstrip("。；;，,") + "。" + suffix


def render_table_package(table: Dict[str, Any]) -> str:
    if not table.get("should_render") or table.get("appendix_only"):
        return ""
    headers = _as_list(table.get("headers"))
    row_objects = [_as_dict(row) for row in _as_list(table.get("rows")) if isinstance(row, dict)]
    rows = [_as_list(row.get("cells")) for row in row_objects]
    headers, rows = _public_table_shape(headers, rows)
    minimum_rows = 1 if str(table.get("table_type") or "") == "cagr_calculation" else 2
    if len(rows) < minimum_rows:
        return ""
    table_md = _markdown_table(headers, rows)
    if not table_md:
        return ""
    parts = [f"**{_compact(table.get('title'), 120)}**", "", table_md]
    decision_implication = _public_text(table.get("decision_implication"), 260)
    limitations = [
        item
        for item in (_public_text(value, 160) for value in _as_list(table.get("limitations"))[:1])
        if item
    ]
    takeaway = _public_text(table.get("takeaway"), 220)
    citation_refs = _as_list(table.get("evidence_refs")) or [
        ref
        for row in row_objects
        for ref in _as_list(row.get("evidence_refs"))
    ]
    if takeaway:
        parts.extend(["", _line_with_citations(f"这张表显示，{takeaway}", citation_refs)])
    if decision_implication and decision_implication != takeaway:
        parts.extend(["", f"判断含义：{decision_implication}"])
    if limitations:
        parts.extend(["", f"使用边界：{limitations[0]}"])
    return "\n".join(parts)


def render_evidence_inventory(evidence_refs: Sequence[Any]) -> List[str]:
    refs = _dedupe([str(ref or "").strip() for ref in evidence_refs if str(ref or "").strip()], limit=12)
    return [f"- {ref}" for ref in refs]


def _citation_suffix(evidence_refs: Sequence[Any], *, limit: int = 3) -> str:
    refs: List[str] = []
    for value in evidence_refs:
        text = str(value or "").strip()
        if not text:
            continue
        match = re.fullmatch(r"\[?(\d{1,3})\]?", text)
        if match:
            refs.append(f"[{match.group(1)}]")
            continue
        match = re.search(r"\[(\d{1,3})\]", text)
        if match:
            refs.append(f"[{match.group(1)}]")
    refs = _dedupe(refs, limit=limit)
    return "".join(refs)


def _append_citation_to_last_paragraph(lines: List[str], evidence_refs: Sequence[Any]) -> None:
    suffix = _citation_suffix(evidence_refs)
    if not suffix:
        return
    for index in range(len(lines) - 1, -1, -1):
        line = str(lines[index] or "").rstrip()
        if not line or line.startswith("#") or line.startswith("|") or re.match(r"^[:\-\s|]+$", line):
            continue
        if re.search(r"\[\d{1,3}\]\s*$", line):
            return
        lines[index] = line.rstrip("。；;，,") + "。" + suffix
        return


def render_dynamic_table(block: Dict[str, Any]) -> List[str]:
    headers = _as_list(block.get("headers"))
    rows = [_as_list(row) for row in _as_list(block.get("rows"))]
    table_md = _markdown_table(headers, rows)
    return [table_md] if table_md else []


def _paragraph_chunks(text: str, *, max_chars: int = 720, max_chunks: int = 5) -> List[str]:
    text = _public_text(text, max_chars * max_chunks)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    sentences = [item.strip() for item in re.split(r"(?<=[。；！？.!?])\s*", text) if item.strip()]
    if len(sentences) <= 1:
        return [text[index : index + max_chars].strip() for index in range(0, len(text), max_chars) if text[index : index + max_chars].strip()][:max_chunks]
    chunks: List[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) > max_chars:
            chunks.append(current.strip())
            current = sentence
        else:
            current = (current + sentence).strip()
        if len(chunks) >= max_chunks:
            break
    if current and len(chunks) < max_chunks:
        chunks.append(current.strip())
    return chunks[:max_chunks]


def _append_narrative_block(lines: List[str], title: str, text: str, *, max_chars: int) -> bool:
    chunks = _paragraph_chunks(text, max_chars=max_chars)
    if not chunks:
        return False
    if title:
        lines.extend(["", f"#### {title}"])
    lines.extend(chunks)
    return True


def render_chapter_deep_synthesis(chapter: Dict[str, Any]) -> List[str]:
    if not _env_flag("REPORT_RENDER_CHAPTER_DEEP_SYNTHESIS", False):
        return []
    summary = _as_dict(chapter.get("chapter_summary"))
    if not summary:
        return []
    title = _public_text(chapter.get("chapter_title"), 140)
    takeaway = _public_text(summary.get("key_takeaway"), 700)
    mechanisms = _dedupe([_public_text(item, 420) for item in _as_list(summary.get("mechanisms"))], limit=3)
    counters = _dedupe([_public_text(item, 420) for item in _as_list(summary.get("counter_evidence"))], limit=3)
    actions = _dedupe([_public_text(item, 420) for item in _as_list(summary.get("next_actions"))], limit=4)
    watch = _dedupe([_public_text(item, 260) for item in _as_list(summary.get("what_to_verify_next"))], limit=5)
    if not any([takeaway, mechanisms, counters, actions, watch]):
        return []
    lines = [""]
    if takeaway:
        lines.append(
            f"围绕“{title}”，本章最直接的判断是：{takeaway}"
            "这部分不单独替代全篇结论，而是说明当前事实能把判断推进到什么程度。"
        )
    if mechanisms:
        lines.append(
            "影响路径上，需要把事实之间的先后关系讲清楚："
            + "；".join(mechanisms)
            + "。这些关系成立时，章节结论才有继续外推的基础。"
        )
    if counters:
        lines.append(
            "结论需要保留的反向条件包括："
            + "；".join(counters)
            + "。这些条件出现时，本章判断应降级或重新校准。"
        )
    if actions:
        lines.append(
            "落到行动层面，本章对应的优先级是："
            + "；".join(actions)
            + "。这些动作应优先服务于缩小判断分歧，而不是扩大未经验证的假设。"
        )
    if watch:
        lines.append(
            "后续观察应聚焦这些触发器："
            + "；".join(watch)
            + "。这些触发器的价值在于让结论可以被复核、被更新，也可以在条件变化时及时收缩。"
        )
    return lines


def render_section(section: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    title = _compact(section.get("section_title"), 120)
    if title and not _is_internal_section_title(title):
        lines.append(f"### {title}")
    for block in _as_list(section.get("render_blocks")):
        block = _as_dict(block)
        block_type = str(block.get("type") or "").strip()
        raw_label = _compact(block.get("label"), 80)
        label = _public_text(block.get("label"), 80)
        text = _public_text(block.get("text"), _env_int("REPORT_RENDER_BLOCK_MAX_CHARS", 3200, min_value=800, max_value=8000))
        is_internal_label = raw_label in INTERNAL_BLOCK_LABELS or _is_internal_section_title(raw_label)
        if block_type == "paragraph":
            if not text:
                continue
            if label and not is_internal_label:
                lines.extend(["", f"#### {label}"])
                lines.extend(_paragraph_chunks(text))
                continue
            else:
                lines.extend(_paragraph_chunks(text))
        elif block_type == "evidence_list":
            continue
        elif block_type == "table":
            rendered = render_dynamic_table(block)
            if rendered:
                if label and not is_internal_label:
                    lines.append(label)
                lines.extend(rendered)
    return [line for line in lines if line.strip()]


def _fact_digest_chunks(facts: Sequence[str], *, chunk_size: int = 4) -> List[List[str]]:
    limit = _env_int("REPORT_RENDER_FACT_DIGEST_LIMIT", 0, min_value=0, max_value=40)
    if limit <= 0:
        return []
    cleaned = [
        item
        for item in (_public_text(value, 520) for value in facts)
        if item and not has_internal_gap_language(item)
    ]
    deduped = _dedupe(cleaned, limit=limit, max_chars=520)
    return [deduped[index : index + chunk_size] for index in range(0, len(deduped), chunk_size)]


def render_chapter_fact_digest(chapter: Dict[str, Any]) -> List[str]:
    chunks = _fact_digest_chunks(_as_list(chapter.get("chapter_fact_digest")), chunk_size=4)
    if not chunks:
        return []
    lines: List[str] = []
    if len(chunks) >= 1:
        lines.append(
            "本章首先需要合并观察这些事实："
            + "；".join(chunks[0])
            + "。它们共同决定本章判断的起点，而不是只作为材料罗列。"
        )
    if len(chunks) >= 2:
        lines.append(
            "进一步看，第二组事实用于校准时间、口径和适用范围："
            + "；".join(chunks[1])
            + "。如果这些信号在时间窗口和统计口径上可比，判断强度会提高；如果口径分化，正文只保留方向性结论。"
        )
    if len(chunks) >= 3:
        lines.append(
            "第三组事实更适合用来判断后续变化是否可持续："
            + "；".join(chunks[2])
            + "。这组组合越能被反复验证，本章越适合进入行动判断；否则应作为观察线索。"
        )
    if len(chunks) >= 4:
        lines.append(
            "剩余线索需要放在约束条件中理解："
            + "；".join(chunks[3])
            + "。它们未必能单独推出强结论，但可以帮助识别结论的适用范围、验证顺序和失效信号。"
        )
    if len(chunks) >= 5:
        lines.append(
            "如果把这些材料落到经营或产业链层面，还需要回答谁承担成本、谁形成预算、谁获得收益："
            + "；".join(chunks[4])
            + "。只有事实链同时覆盖需求、供给、价格或订单以及反向样本，正文才适合写成较强判断。"
        )
    if len(chunks) >= 6:
        lines.append(
            "因此，本章把事实分成两类处理：一类直接改变判断权重，另一类限定判断边界。"
            + "；".join(chunks[5])
            + "。前者决定结论能否上升为行动建议，后者决定建议需要附带哪些条件和验证动作。"
        )
    return [line for line in lines if line.strip()]


def _chapter_flow_intro(
    chapter: Dict[str, Any],
    *,
    index: int,
    previous_chapter: Dict[str, Any] | None = None,
) -> str:
    """Generate an opening line for a chapter.

    Previous implementation always emitted "本章聚焦"X"。围绕"Y",重点看事实能支持
    到哪一步,以及哪些条件会削弱判断。" for every chapter, which made the report
    feel templated. Now:
    - If the chapter already has a substantive `lead` field, return empty (the
      lead itself becomes the chapter opening).
    - Otherwise emit only the shortest contextual sentence — no boilerplate
      tail like "重点看事实能支持到哪一步".
    - Can be disabled entirely via REPORT_DISABLE_CHAPTER_INTRO=1.
    """
    if os.environ.get("REPORT_DISABLE_CHAPTER_INTRO", "").strip() in {"1", "true", "True"}:
        return ""
    lead = _public_text(chapter.get("lead"), 360)
    if lead:
        # The chapter already carries its own opening narrative; do not append a template.
        return ""
    title = _public_text(chapter.get("chapter_title"), 120)
    previous_title = _public_text(_as_dict(previous_chapter).get("chapter_title"), 120)
    if previous_title and title:
        return f"承接“{previous_title}”,本章关注“{title}”。"
    if title:
        return f"本章关注“{title}”。"
    return ""


def _chapter_transition(chapter: Dict[str, Any], next_chapter: Dict[str, Any] | None = None) -> str:
    """Inter-chapter transition sentence.

    Previously hard-coded as "由此,X给出了当前判断的成立条件;接下来的 Y 会继续
    检验这些条件是否能延续。" — which appeared at the end of every chapter and
    is the single biggest contributor to the "machine-stitched" feel of the
    rendered report.

    Now: disabled by default. Only emitted when REPORT_ENABLE_CHAPTER_TRANSITION=1
    explicitly opts back in. Even then, requires a real takeaway on the current
    chapter (not just a chapter title) to actually output.
    """
    if os.environ.get("REPORT_ENABLE_CHAPTER_TRANSITION", "0").strip() not in {"1", "true", "True"}:
        return ""
    title = _public_text(chapter.get("chapter_title"), 100)
    next_title = _public_text(_as_dict(next_chapter).get("chapter_title"), 100)
    takeaway = _public_text(_as_dict(chapter.get("chapter_summary")).get("key_takeaway"), 160)
    if not (title and next_title and takeaway):
        return ""
    return f"由此,“{title}”给出了当前判断的成立条件;接下来的“{next_title}”会继续检验这些条件是否能延续。"


def _final_action_phrase(value: str) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^后续重点跟踪", "", text)
    text = re.sub(r"^后续跟踪集中在", "", text)
    text = re.sub(r"^后续跟踪", "", text)
    text = re.sub(r"^重点跟踪", "", text)
    return text.strip(" ：，。；") or str(value or "").strip()


def _clause(value: str) -> str:
    return str(value or "").strip(" \t\r\n，。；")


def _table_slot(table: Dict[str, Any]) -> str:
    slot = str(table.get("placement_slot") or "").strip()
    return slot or "chapter_end"


def _table_render_priority(table: Dict[str, Any]) -> int:
    try:
        return int(table.get("render_priority") or 0)
    except (TypeError, ValueError):
        return 0


def _section_matches_table(section: Dict[str, Any], table: Dict[str, Any]) -> bool:
    section_id = str(section.get("section_id") or "").strip()
    block_type = str(section.get("block_type") or section.get("output_type") or "").strip()
    anchor_section_id = str(table.get("anchor_section_id") or "").strip()
    anchor_block_type = str(table.get("anchor_block_type") or "").strip()
    if anchor_section_id and section_id == anchor_section_id:
        return True
    if anchor_block_type and block_type == anchor_block_type:
        return True
    section_refs = {
        str(ref or "").strip()
        for ref in _as_list(section.get("evidence_refs")) + _as_list(section.get("required_evidence_refs"))
        if str(ref or "").strip()
    }
    table_refs = {str(ref or "").strip() for ref in _as_list(table.get("evidence_refs")) if str(ref or "").strip()}
    return bool(section_refs and table_refs and section_refs.intersection(table_refs))


def _slot_matches_section(slot: str, section: Dict[str, Any]) -> bool:
    block_type = str(section.get("block_type") or section.get("output_type") or "").strip()
    if slot == "after_thesis":
        return block_type == "thesis"
    if slot == "after_evidence_matrix":
        return block_type in {"evidence_matrix", "metric_reconciliation", "competitive_positioning", "case_comparison"}
    if slot == "after_mechanism":
        return block_type in {"mechanism_chain", "technology_maturity", "value_chain_map", "policy_timeline", "stakeholder_map"}
    if slot == "before_risk":
        return block_type == "risk_trigger"
    if slot == "before_decision":
        return block_type in {"verification_checklist", "scenario_analysis"}
    return False


def render_chapter_package(
    chapter: Dict[str, Any],
    index: int,
    *,
    previous_chapter: Dict[str, Any] | None = None,
    next_chapter: Dict[str, Any] | None = None,
) -> str:
    title = _compact(chapter.get("chapter_title") or f"章节 {index}", 120)
    lines = [f"## {index}. {title}"]
    lead = _public_text(chapter.get("lead"), 360)
    flow_intro = _chapter_flow_intro(chapter, index=index, previous_chapter=previous_chapter)
    if flow_intro:
        lines.append(flow_intro)
    if lead and lead != flow_intro:
        lines.append(lead)
    chapter_tables = [
        {"table": table, "rendered": rendered}
        for table in _as_list(chapter.get("table_packages"))
        if isinstance(table, dict)
        for rendered in [render_table_package(table)]
        if rendered
    ]
    chapter_tables.sort(key=lambda item: (-_table_render_priority(_as_dict(item.get("table"))), str(_as_dict(item.get("table")).get("table_id") or "")))
    placed_tables: set[str] = set()

    def append_tables(slot: str, *, section: Dict[str, Any] | None = None, limit: int = 1) -> None:
        appended = 0
        for item in chapter_tables:
            table = _as_dict(item.get("table"))
            table_id = str(table.get("table_id") or id(table))
            if table_id in placed_tables:
                continue
            table_slot = _table_slot(table)
            if table_slot != slot and not (section and _section_matches_table(section, table) and _slot_matches_section(slot, section)):
                continue
            if section and not (_section_matches_table(section, table) or _slot_matches_section(table_slot, section)):
                continue
            lines.extend(["", str(item.get("rendered") or "")])
            placed_tables.add(table_id)
            appended += 1
            if appended >= limit:
                return

    for section_index, section in enumerate(_as_list(chapter.get("sections")), start=1):
        section = _as_dict(section)
        if section.get("omit_from_report"):
            continue
        if _slot_matches_section("before_risk", section):
            append_tables("before_risk", section=section, limit=1)
        if _slot_matches_section("before_decision", section):
            append_tables("before_decision", section=section, limit=1)
        if _as_list(section.get("render_blocks")):
            rendered_section = render_section(section)
            if rendered_section:
                _append_citation_to_last_paragraph(rendered_section, _as_list(section.get("evidence_refs")))
                lines.append("")
                lines.extend(rendered_section)
                for slot in ("after_thesis", "after_evidence_matrix", "after_mechanism"):
                    if _slot_matches_section(slot, section):
                        append_tables(slot, section=section, limit=1)
                        break
            continue
        section_title = _compact(section.get("section_title"), 120)
        if section_title and not _is_internal_section_title(section_title):
            lines.extend(["", f"### {section_title}"])
        claim = _public_text(section.get("claim"), _env_int("REPORT_SECTION_CLAIM_MAX_CHARS", 1100, min_value=300, max_value=3000))
        reasoning = _public_text(section.get("reasoning"), _env_int("REPORT_SECTION_REASONING_MAX_CHARS", 3200, min_value=800, max_value=8000))
        counter = _public_text(section.get("counter_evidence"), _env_int("REPORT_SECTION_COUNTER_MAX_CHARS", 1600, min_value=400, max_value=5000))
        actionable = _public_text(section.get("actionable"), _env_int("REPORT_SECTION_ACTION_MAX_CHARS", 1600, min_value=400, max_value=5000))
        mechanism = _public_text(section.get("mechanism") or section.get("reasoning"), _env_int("REPORT_SECTION_MECHANISM_MAX_CHARS", 2400, min_value=600, max_value=6000))
        decision_implication = _public_text(section.get("decision_implication") or actionable, _env_int("REPORT_SECTION_ACTION_MAX_CHARS", 1600, min_value=400, max_value=5000))
        before_section_len = len(lines)
        if claim:
            lines.extend(_paragraph_chunks(claim, max_chars=700, max_chunks=2))
        if reasoning:
            lines.extend(_paragraph_chunks(reasoning, max_chars=760, max_chunks=3))
        if mechanism and mechanism != reasoning:
            lines.extend(_paragraph_chunks(mechanism, max_chars=700, max_chunks=2))
        elif mechanism and not reasoning:
            lines.extend(_paragraph_chunks(mechanism, max_chars=700, max_chunks=2))
        if counter:
            lines.extend(_paragraph_chunks(_natural_transition("同时，", counter), max_chars=680, max_chunks=2))
        if decision_implication:
            lines.extend(_paragraph_chunks(_natural_transition("落到行业含义上，", decision_implication), max_chars=680, max_chunks=2))
        if len(lines) > before_section_len:
            _append_citation_to_last_paragraph(lines, _as_list(section.get("evidence_refs")))
            for slot in ("after_thesis", "after_evidence_matrix", "after_mechanism"):
                if _slot_matches_section(slot, section):
                    append_tables(slot, section=section, limit=1)
                    break
    fact_digest = render_chapter_fact_digest(chapter)
    if fact_digest:
        lines.append("")
        lines.extend(fact_digest)
    deep_synthesis = render_chapter_deep_synthesis(chapter)
    if deep_synthesis:
        lines.extend(deep_synthesis)
    append_tables("chapter_end", limit=2)
    for item in chapter_tables:
        table = _as_dict(item.get("table"))
        table_id = str(table.get("table_id") or id(table))
        if table_id not in placed_tables:
            lines.extend(["", str(item.get("rendered") or "")])
            placed_tables.add(table_id)
    transition = _chapter_transition(chapter, next_chapter)
    if transition:
        lines.extend(["", transition])
    return "\n".join(_dedupe_narrative_lines(lines))


def render_final_reference_analysis(decision_package: Dict[str, Any]) -> List[str]:
    if not _env_flag("REPORT_RENDER_FINAL_REFERENCE_ANALYSIS", False):
        return []
    syntheses = [_as_dict(item) for item in _as_list(decision_package.get("chapter_syntheses"))]
    rows: List[str] = []
    visible_syntheses = syntheses[:8]
    for position, item in enumerate(visible_syntheses):
        title = _public_text(item.get("chapter_title"), 140)
        question = _public_text(item.get("chapter_question"), 180)
        prev_title = _public_text(_as_dict(visible_syntheses[position - 1]).get("chapter_title"), 140) if position > 0 else ""
        next_title = _public_text(_as_dict(visible_syntheses[position + 1]).get("chapter_title"), 140) if position + 1 < len(visible_syntheses) else ""
        summary = _as_dict(item.get("chapter_summary"))
        takeaway = _public_text(summary.get("key_takeaway"), 780)
        mechanisms = _dedupe([_clause(_public_text(value, 620)) for value in _as_list(summary.get("mechanisms"))], limit=3, max_chars=520)
        counters = _dedupe([_clause(_public_text(value, 620)) for value in _as_list(summary.get("counter_evidence"))], limit=3, max_chars=520)
        actions = _dedupe([_clause(_final_action_phrase(_public_text(value, 560))) for value in _as_list(summary.get("next_actions"))], limit=4, max_chars=460)
        watch = _dedupe([_clause(_public_text(value, 340)) for value in _as_list(summary.get("what_to_verify_next"))], limit=5, max_chars=320)
        if not any([takeaway, mechanisms, counters, actions, watch]):
            continue

        heading = title or question or "相关章节"
        paragraph_parts: List[str] = []
        if takeaway:
            paragraph_parts.append(
                f"“{heading}”对应的章节结论是：{takeaway}"
                "它的权重取决于能否和其他章节里的关键对象、指标、约束和反向样本互相解释。"
            )
        if mechanisms:
            paragraph_parts.append(
                "影响路径可以概括为："
                + "；".join(mechanisms)
                + "。这些环节衔接得越完整，章节结论越能进入全篇判断。"
            )
        if counters:
            paragraph_parts.append(
                "结论弹性主要来自这些反向条件："
                + "；".join(counters)
                + "。这些条件决定了结论在什么情况下增强、收缩或被重新解释。"
            )
        if actions:
            paragraph_parts.append(
                "资源配置会集中到："
                + "；".join(actions)
                + "。这样可以避免被孤立新闻、短期波动或单一案例牵引。"
            )
        if watch:
            paragraph_parts.append(
                "后续变化主要集中在："
                + "；".join(watch)
                + "。这些变量一旦变化，整篇报告的强弱排序和行动优先级也会随之变化。"
            )
        if prev_title or next_title:
            relation_parts = []
            if prev_title:
                relation_parts.append(f"它承接“{prev_title}”留下的变量条件")
            if next_title:
                relation_parts.append(f"也为“{next_title}”提供判断基础")
            paragraph_parts.append("放在章节顺序中看，" + "，".join(relation_parts) + "。")
        rows.append("".join(paragraph_parts))

    if not rows:
        return []
    chapter_titles = [_public_text(item.get("chapter_title"), 120) for item in visible_syntheses]
    chapter_chain = "、".join(title for title in chapter_titles if title)
    closing_lines = [
        "综合来看，最终结论的强弱取决于章节之间是否能相互解释：单个事实只能提供观察，多组事实在口径、时间和对象上相互支撑时，才适合进入更明确的行动判断。",
        "后续更新也应沿着同一顺序进行：先补足关键事实，再校准口径差异，最后观察反向条件是否出现。这样新增材料不会打散原有结构，而是能回到对应章节重新排序。"
    ]
    if chapter_chain:
        closing_lines.insert(
            0,
            f"从章节排列看，{chapter_chain}不是并列清单，而是一组逐步收束的判断。每一章承担什么位置，应由研究问题和证据决定。",
        )
    return [
        "",
        "### 全篇收束",
        *rows,
        *closing_lines,
    ]


def render_decision_package(decision_package: Dict[str, Any]) -> str:
    lines: List[str] = []
    thesis = _public_text(decision_package.get("decision_thesis"), 1200)
    if thesis:
        lines.extend(["## 综合判断与策略含义", thesis])
    recommendations = [_as_dict(item) for item in _as_list(decision_package.get("recommendations"))]
    public_recs = []
    for item in recommendations[:5]:
        text = _public_text(item.get("recommendation"), 700)
        label = _compact(item.get("label"), 40)
        if text:
            public_recs.append(f"- {label + '：' if label else ''}{text}")
    if public_recs:
        if lines:
            lines.extend(["", "### 结论如何落到动作", *public_recs])
        else:
            lines.extend(["## 综合判断与策略含义", *public_recs])
    scenarios = [_as_dict(item) for item in _as_list(decision_package.get("scenario_analysis"))]
    public_scenarios = []
    for item in scenarios[:3]:
        scenario = _public_text(item.get("scenario"), 80)
        condition = _public_text(item.get("condition"), 700)
        implication = _public_text(item.get("implication"), 700)
        if scenario and (condition or implication):
            public_scenarios.append(f"- **{scenario}**：{condition} {implication}".strip())
    if public_scenarios:
        if lines:
            lines.extend(["", "### 情景分层与结论弹性", *public_scenarios])
        else:
            lines.extend(["## 综合判断与策略含义", "### 情景分层与结论弹性", *public_scenarios])
    reference_lines = render_final_reference_analysis(decision_package)
    if reference_lines and lines:
        lines.extend(reference_lines)
    watchlist = [_as_dict(item) for item in _as_list(decision_package.get("watchlist"))]
    public_watch = []
    for item in watchlist[:5]:
        metric = _public_text(item.get("metric"), 520)
        if metric:
            public_watch.append(f"- {metric}")
    if public_watch and lines:
        lines.extend(["", "### 观察指标", *public_watch])
    abandon = [_as_dict(item) for item in _as_list(decision_package.get("abandon_conditions"))]
    public_abandon = []
    for item in abandon[:3]:
        condition = _public_text(item.get("condition"), 520)
        if condition:
            public_abandon.append(f"- {condition}")
    if public_abandon and lines:
        lines.extend(["", "### 放弃条件", *public_abandon])
    return "\n".join(lines)


def render_risk_package(risk_package: Dict[str, Any]) -> str:
    rows = []
    for item in _as_list(risk_package.get("risk_items"))[:8]:
        item = _as_dict(item)
        description = _public_text(item.get("description"), 220)
        if not description:
            continue
        risk_type = _compact(item.get("risk_type"), 80)
        mitigation = _public_text(item.get("mitigation"), 220)
        severity = _compact(item.get("severity"), 30)
        prefix = f"{risk_type}（{severity}）" if risk_type and severity else (risk_type or "风险事项")
        rows.append(f"- {prefix}：{description}")
        if mitigation:
            rows.append(f"  应对：{mitigation}")
    if not rows:
        return ""
    return "\n".join(["## 反向信号与风险触发", *rows])


def render_appendix(source_registry: Sequence[Dict[str, Any]], appendix_package: Dict[str, Any]) -> str:
    lines = ["## 研究口径与来源"]
    scope = _public_text(_as_dict(appendix_package).get("scope_note"), 260)
    if scope:
        lines.append(scope)
    coverage_rows = [_as_dict(item) for item in _as_list(_as_dict(appendix_package).get("coverage_matrix")) if isinstance(item, dict)]
    if coverage_rows and _env_flag("REPORT_RENDER_COVERAGE_MATRIX", False):
        rows = []
        for item in coverage_rows[:12]:
            rows.append(
                [
                    _compact(item.get("hypothesis_statement") or item.get("hypothesis_id"), 80),
                    f"{item.get('actual_ab_sources') or 0}/{item.get('required_ab_sources') or 0}",
                    str(item.get("counter_evidence_count") or 0),
                    f"{item.get('complete_metric_count') or 0}/{item.get('metric_count') or 0}",
                    "是" if item.get("decision_ready") else "否",
                    "、".join(_dedupe(_as_list(item.get("blocking_gaps")), limit=3)),
                ]
            )
        table_md = _markdown_table(["假设", "A/B来源", "反证", "指标口径", "可下判断", "待补"], rows)
        if table_md:
            lines.extend(["", "### 证据覆盖矩阵", table_md])
    metric_rows = [_as_dict(item) for item in _as_list(_as_dict(appendix_package).get("metric_normalization_table")) if isinstance(item, dict)]
    if metric_rows:
        rows = []
        for item in metric_rows[:12]:
            rows.append(
                [
                    _compact(item.get("metric_name"), 60),
                    _compact(item.get("subject"), 70),
                    _compact(item.get("scope"), 50),
                    _compact(item.get("period"), 40),
                    _compact(item.get("unit"), 24),
                    _compact(item.get("value"), 50),
                    _compact(item.get("source_level"), 20),
                ]
            )
        table_md = _markdown_table(["指标", "主体", "范围", "期间", "单位", "值", "来源等级"], rows)
        if table_md:
            lines.extend(["", "### 指标口径表", table_md])
    appendix_tables = [
        _as_dict(item)
        for item in _as_list(_as_dict(appendix_package).get("table_appendix_rows"))
        if isinstance(item, dict)
    ]
    if appendix_tables:
        for table in appendix_tables[:6]:
            headers = _as_list(table.get("headers"))
            rows = [_as_list(row) for row in _as_list(table.get("rows"))[:12]]
            headers, rows = _public_table_shape(headers, rows)
            table_md = _markdown_table(headers, rows)
            if not table_md:
                continue
            title = _compact(table.get("title") or "表格附录明细", 90)
            lines.extend(["", f"### {title}（附录明细）", table_md])
    if not source_registry:
        return "\n".join(lines) if len(lines) > 1 else ""
    for source in list(source_registry)[:50]:
        source = _as_dict(source)
        ref = str(source.get("ref") or "").strip()
        title = str(source.get("title") or "未命名来源").strip()
        date = str(source.get("date") or "").strip()
        url = str(source.get("url") or "").strip()
        suffix = " | ".join(part for part in [date, url] if part)
        lines.append(f"- {ref} {title}" + (f" | {suffix}" if suffix else ""))
    return "\n".join(lines)


def collect_format_warnings(markdown: str) -> List[str]:
    warnings: List[str] = []
    if not markdown.strip():
        warnings.append("report_markdown_empty")
    if len(re.findall(r"^# ", markdown, flags=re.M)) > 1:
        warnings.append("multiple_h1")
    if re.search(r"\n{4,}", markdown):
        warnings.append("excessive_blank_lines")
    return warnings
