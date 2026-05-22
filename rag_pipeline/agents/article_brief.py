from __future__ import annotations

import re
from typing import Any, Dict


TITLE_LABELS = (
    "主标题",
    "标题",
    "主题",
    "主主题",
    "文章主题",
    "文章主标题",
)
DIRECTION_LABELS = (
    "副标题",
    "方向",
    "报告方向",
    "具体方向",
    "文章方向",
    "研究方向",
)
LABEL_PATTERN = re.compile(
    rf"({'|'.join(map(re.escape, (*TITLE_LABELS, *DIRECTION_LABELS)))})\s*[:：]\s*"
    rf"(.*?)(?=(?:\s*(?:{'|'.join(map(re.escape, (*TITLE_LABELS, *DIRECTION_LABELS)))})\s*[:：])|$)",
    re.S,
)

SUBJECT_CUT_MARKERS = (
    "竞争格局",
    "市场格局",
    "产业格局",
    "技术瓶颈",
    "底层基础设施",
    "基础设施演进",
    "商业化路径",
    "商业化",
    "产业机会",
    "行业机会",
    "投资机会",
    "市场机会",
    "市场规模",
    "发展趋势",
    "趋势",
    "前景",
    "机会",
    "风险",
    "报告",
    "研究",
    "分析",
)
BROAD_AI_SUBJECT_RE = re.compile(
    r"^(?:中国|国内|全球|国产)?\s*(?:AI|人工智能|大模型|生成式AI|AIGC)\s*(?:行业|产业|市场|赛道)?$",
    re.I,
)


def _compact(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("\u3000", " ")).strip()


def _strip_label(value: Any, labels: tuple[str, ...]) -> str:
    text = _compact(value)
    if not text:
        return ""
    label_re = "|".join(map(re.escape, labels))
    return re.sub(rf"^(?:{label_re})\s*[:：]\s*", "", text, flags=re.I).strip()


def clean_main_title(value: Any) -> str:
    text = _strip_label(value, TITLE_LABELS)
    text = text.strip().strip("“”\"' ")
    if len(text) >= 2 and text[0] == "《" and text[-1] == "》":
        text = text[1:-1].strip()
    return _compact(text)


def clean_direction(value: Any) -> str:
    text = _strip_label(value, DIRECTION_LABELS)
    text = text.strip().strip("“”\"' ")
    text = re.sub(r"^[—–-]{1,3}\s*", "", text).strip()
    if len(text) >= 2 and text[0] == "《" and text[-1] == "》":
        text = text[1:-1].strip()
    return _compact(text)


def extract_research_subject(value: Any, *, fallback: Any = "") -> str:
    """Extract the concrete research object without binding to a fixed industry template."""
    text = clean_direction(value) or clean_direction(fallback)
    if not text:
        return ""
    text = re.sub(r"^\d{4}\s*年?", "", text).strip()
    text = re.sub(r"^(?:关于|围绕|聚焦)\s*", "", text).strip()
    text = re.sub(r"(?:企业行研|行业研究|行研|深度研究|研究)?(?:报告|文档)$", "", text).strip()
    first_clause = re.split(r"[，,；;。！？?]\s*", text, maxsplit=1)[0].strip()
    candidate_source = first_clause if len(first_clause) >= 3 else text
    candidate = candidate_source
    marker_positions = [
        candidate_source.find(marker)
        for marker in SUBJECT_CUT_MARKERS
        if candidate_source.find(marker) > 1
    ]
    if marker_positions:
        candidate = candidate_source[: min(marker_positions)].strip()
    candidate = re.sub(r"(?:当前|目前|未来|现在)$", "", candidate).strip()
    candidate = re.sub(r"(?:受阻|遇冷|爆发|破局)$", "", candidate).strip()
    candidate = candidate.strip("：: -—–")
    return _compact(candidate or candidate_source or text)


def is_broad_ai_subject(value: Any) -> bool:
    return bool(BROAD_AI_SUBJECT_RE.fullmatch(_compact(value)))


def parse_labeled_article_brief(raw_query: Any) -> Dict[str, str]:
    raw = str(raw_query or "").strip()
    if not raw:
        return {}
    result: Dict[str, str] = {}
    for match in LABEL_PATTERN.finditer(raw):
        label = _compact(match.group(1))
        value = match.group(2)
        if label in TITLE_LABELS:
            result["main_title"] = clean_main_title(value)
        elif label in DIRECTION_LABELS:
            result["direction"] = clean_direction(value)
    if result:
        return {key: value for key, value in result.items() if value}

    lines = [_compact(line) for line in raw.splitlines() if _compact(line)]
    if len(lines) >= 2:
        first = clean_main_title(lines[0])
        second = clean_direction(lines[1])
        if first and second and (lines[1].startswith(("—", "-", "副标题", "方向")) or "报告" in second):
            return {"main_title": first, "direction": second}
    return {}


def build_article_brief(
    *,
    raw_query: Any = "",
    title: Any = "",
    subtitle: Any = "",
    direction: Any = "",
    interactive_confirmed: bool = False,
) -> Dict[str, Any]:
    parsed = parse_labeled_article_brief(raw_query)
    explicit_title = clean_main_title(title)
    parsed_title = clean_main_title(parsed.get("main_title", ""))
    legacy_title = clean_main_title(raw_query)

    explicit_subtitle = clean_direction(subtitle)
    explicit_direction = clean_direction(direction)
    parsed_direction = clean_direction(parsed.get("direction", ""))

    main_title = explicit_title or parsed_title or (legacy_title if not parsed_direction else "")
    direction_text = explicit_direction or explicit_subtitle or parsed_direction
    display_subtitle = explicit_subtitle or parsed_direction or explicit_direction
    display_title = main_title or direction_text
    planning_query = direction_text or main_title or legacy_title
    if direction_text:
        planning_query_source = "direction"
    elif main_title:
        planning_query_source = "main_title"
    elif legacy_title:
        planning_query_source = "raw_query"
    else:
        planning_query_source = "empty"

    if explicit_title or explicit_subtitle or explicit_direction:
        parsed_from = "explicit_fields"
    elif parsed:
        parsed_from = "labeled_query"
    elif legacy_title:
        parsed_from = "legacy_query"
    else:
        parsed_from = "empty"

    return {
        "main_title": main_title,
        "direction": direction_text,
        "display_title": display_title,
        "display_subtitle": display_subtitle,
        "planning_query": planning_query,
        "planning_query_source": planning_query_source,
        "direction_missing": not bool(direction_text),
        "raw_query": str(raw_query or "").strip(),
        "parsed_from": parsed_from,
        "interactive_confirmed": bool(interactive_confirmed),
    }


def normalize_article_brief(value: Any, *, fallback_query: Any = "") -> Dict[str, Any]:
    if not isinstance(value, dict):
        return build_article_brief(raw_query=fallback_query)
    return build_article_brief(
        raw_query=value.get("raw_query") or fallback_query,
        title=value.get("main_title") or value.get("display_title"),
        subtitle=value.get("display_subtitle"),
        direction=value.get("direction") or value.get("article_direction"),
        interactive_confirmed=bool(value.get("interactive_confirmed", False)),
    )


def planning_query_from_brief(value: Any, *, fallback_query: Any = "") -> str:
    brief = normalize_article_brief(value, fallback_query=fallback_query)
    return str(brief.get("planning_query") or fallback_query or "").strip()
