from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, Iterable, List, Optional

try:
    from .layout_compiler import compile_report_layout
    from .universal_report_ontology import evidence_mix_for_modules, module_keys
except Exception:  # pragma: no cover - direct script mode fallback
    from layout_compiler import compile_report_layout  # type: ignore
    from universal_report_ontology import evidence_mix_for_modules, module_keys  # type: ignore


AGENT_NAME = "pre_layout_agent"
AGENT_DESCRIPTION = "Dynamic chapter blueprint normalizer. It standardizes planner chapters before search."
logger = logging.getLogger(__name__)

LEGACY_FIVE_TITLES = {
    "市场规模与增速",
    "竞争格局",
    "政策与监管环境",
    "技术路线与产业链",
    "资本动态",
}

DEFAULT_EVIDENCE_MIX = ["official_data", "market_research", "company_filing", "case", "counter_evidence"]


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _compact(value: Any, max_chars: int = 220) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "..."


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(str(raw).strip())
    except ValueError:
        return default


def _dedupe(values: Iterable[Any], *, limit: int = 20) -> List[str]:
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


def _questionize_title(title: str, core_question: str = "") -> str:
    title = _compact(title, 120)
    core_question = _compact(core_question, 120)
    if title.endswith(("？", "?")):
        return title
    if title in LEGACY_FIVE_TITLES:
        mapping = {
            "市场规模与增速": "当前规模、价格和增速能否支撑这个判断？",
            "竞争格局": "竞争强度和主要玩家会怎样改变结论？",
            "政策与监管环境": "政策和监管变量会怎样传导到结果？",
            "技术路线与产业链": "技术路线和产业链瓶颈会怎样影响落地？",
            "资本动态": "资本和交易信号是否真正验证商业化？",
        }
        return mapping.get(title, core_question or f"{title}需要回答的关键判断是什么？")
    if core_question and core_question.endswith(("？", "?")):
        return core_question
    if core_question:
        return f"{core_question}？"
    return f"{title}到底意味着什么？" if title else "研究对象的关键判断是什么？"


def _is_legacy_fixed_title(title: Any) -> bool:
    return str(title or "").strip() in LEGACY_FIVE_TITLES


# Trailing question-style fragments that, when present at the end of a heading,
# should be stripped so the H2 reads as a noun-phrase rather than a research
# question. Without this, every rendered chapter heading looks like:
#     "## 1. 规模、增速和价格信号是否支持机会判断？"
# which the user has flagged as the single biggest contributor to the report
# feeling templated.
_QUESTION_TAIL_RE = re.compile(
    r"("
    r"是否[支持成立兑现可持续可行]*?[？?]?$|"
    r"能否[一-鿿]{0,12}[？?]?$|"
    r"如何[一-鿿]{0,12}[？?]?$|"
    r"会怎样[一-鿿]{0,12}[？?]?$|"
    r"怎样[一-鿿]{0,12}[？?]?$|"
    r"有没有[一-鿿]{0,8}[？?]?$|"
    r"为什么[一-鿿]{0,12}[？?]?$|"
    r"会不会[一-鿿]{0,12}[？?]?$|"
    r"哪些[一-鿿]{0,10}[？?]?$|"
    r"吗[？?]?$|"
    r"[？?]+$"
    r")"
)


def _titleize_question(question: str) -> str:
    """Compress a research-style question into a noun-phrase heading.

    Examples:
        "规模、增速和价格信号是否支持机会判断？" -> "规模、增速与价格信号"
        "需求变化来自哪里,能否持续兑现？"        -> "需求变化来源与可持续性"
        "技术路线成熟度和替代路径会怎样影响落地？" -> "技术路线成熟度与替代路径"

    Returns empty string when nothing meaningful can be salvaged so the caller
    can keep the original input.
    """
    text = _compact(question, 200)
    if not text:
        return ""
    # Drop everything from the first interrogative marker onwards.
    cleaned = re.split(r"[,，;；]?\s*(?:能否|是否|如何|会怎样|怎样|有没有|为什么|会不会|哪些|吗)", text, maxsplit=1)[0]
    cleaned = _QUESTION_TAIL_RE.sub("", cleaned)
    cleaned = cleaned.strip(" ,，。:：;；?？")
    # Drop dangling connectors like trailing "和/与/的" left by the split.
    cleaned = re.sub(r"[和与及、的之]+$", "", cleaned).strip()
    if not cleaned or cleaned == text:
        return ""
    if len(cleaned) < 4:  # too short to be a meaningful heading; keep original
        return ""
    return cleaned


def _source_template_keys(raw: Dict[str, Any], research_plan: Dict[str, Any]) -> List[str]:
    explicit = _dedupe(_as_list(raw.get("source_template_keys") or raw.get("template_keys")), limit=8)
    valid = set(module_keys())
    filtered = [item for item in explicit if item in valid]
    if filtered:
        return filtered

    text = " ".join(
        str(part or "")
        for part in [
            raw.get("chapter_title"),
            raw.get("core_question"),
            raw.get("chapter_question"),
            raw.get("reason_to_include"),
            research_plan.get("query"),
        ]
    )
    signals = [
        ("industry_definition", r"定义|范围|边界|分类|是什么"),
        ("market_size", r"市场|规模|增速|行情|价格|空间|TAM|CAGR"),
        ("demand_driver", r"需求|订单|开工|销量|库存|采购|景气|边际"),
        ("industry_chain", r"产业链|上游|下游|供应链|利润|成本|产能"),
        ("technology", r"技术|产品|路线|性能|替代|专利|标准"),
        ("customer", r"客户|用户|场景|中标|采购|ROI|痛点"),
        ("business_model", r"商业模式|盈利|毛利|现金流|收费|变现"),
        ("competition", r"竞争|格局|玩家|份额|厂商|替代"),
        ("policy", r"政策|监管|法规|补贴|规划|目录|审批"),
        ("capital", r"融资|估值|并购|IPO|股价|市值|资本"),
        ("risk", r"风险|反证|触发器|下滑|过剩|价格战|不确定"),
        ("entry_strategy", r"进入|策略|投资|采购|立项|建议|动作"),
    ]
    matched = [key for key, pattern in signals if re.search(pattern, text, re.I)]
    return matched[:5] or ["market_size", "demand_driver", "risk"]


def _required_evidence_mix(raw: Dict[str, Any], template_keys: List[str]) -> List[str]:
    explicit = _dedupe(_as_list(raw.get("required_evidence_mix")), limit=10)
    if explicit:
        return explicit
    mix = evidence_mix_for_modules(template_keys)
    aliases = {
        "policy_original": "official_data",
        "official_definition": "official_data",
        "classification": "official_data",
        "standard": "official_data",
        "technical_standard": "technology_product",
        "patent": "technology_product",
        "product_doc": "technology_product",
        "procurement": "customer_case",
        "financial_metric": "company_filing",
        "forecast": "market_research",
        "market_data": "market_price",
        "counter": "counter_evidence",
    }
    normalized = [aliases.get(item, item) for item in mix]
    result = _dedupe([*normalized, *DEFAULT_EVIDENCE_MIX], limit=8)
    return result


def _goals_for_chapter(research_plan: Dict[str, Any], raw: Dict[str, Any], chapter_id: str, title: str) -> List[Dict[str, Any]]:
    direct = [dict(item) for item in _as_list(raw.get("evidence_goals")) if isinstance(item, dict)]
    if direct:
        return direct
    goals: List[Dict[str, Any]] = []
    raw_ids = {
        str(raw.get("chapter_id") or "").strip(),
        str(raw.get("dimension_id") or "").strip(),
        chapter_id,
    }
    raw_names = {
        str(raw.get("chapter_title") or "").strip(),
        str(raw.get("dimension_name") or raw.get("dimension") or "").strip(),
        title,
    }
    for goal in _as_list(research_plan.get("evidence_goals")):
        if not isinstance(goal, dict):
            continue
        goal_chapter_id = str(goal.get("chapter_id") or goal.get("dimension_id") or "").strip()
        goal_chapter_name = str(goal.get("chapter_title") or goal.get("dimension_name") or goal.get("dimension") or "").strip()
        if goal_chapter_id in raw_ids or goal_chapter_name in raw_names:
            goals.append(dict(goal))
    return goals


def _search_hints_for_chapter(research_plan: Dict[str, Any], chapter_id: str, title: str) -> List[Dict[str, Any]]:
    hints: List[Dict[str, Any]] = []
    for task in _as_list(research_plan.get("search_tasks")):
        if not isinstance(task, dict):
            continue
        task_chapter_id = str(task.get("chapter_id") or task.get("dimension_id") or "").strip()
        task_chapter_title = str(task.get("chapter_title") or task.get("dimension_name") or task.get("dimension") or "").strip()
        if task_chapter_id == chapter_id or task_chapter_title == title:
            hints.append(dict(task))
    return hints


def normalize_chapter(raw: Dict[str, Any], *, index: int, research_plan: Dict[str, Any]) -> Dict[str, Any]:
    raw = _as_dict(raw)
    fallback_id = f"ch_{index:02d}"
    chapter_id = str(raw.get("chapter_id") or raw.get("dimension_id") or raw.get("id") or fallback_id).strip()
    if not re.match(r"^ch[_-]?\d+", chapter_id, flags=re.I) and chapter_id.startswith(("dim_", "dimension_", "chapter_")):
        chapter_id = fallback_id

    title = _compact(
        raw.get("chapter_title")
        or raw.get("title")
        or raw.get("dimension_name")
        or raw.get("dimension")
        or raw.get("name")
        or f"研究问题 {index}",
        140,
    )
    core_question = _compact(
        raw.get("core_question")
        or raw.get("chapter_question")
        or raw.get("question")
        or raw.get("purpose")
        or title,
        220,
    )
    # IMPORTANT: do NOT promote a short noun-phrase title into a question. The
    # previous behaviour did exactly that, which is why every chapter heading
    # in the rendered report looked like a verbatim research question. Instead:
    #  - Legacy templated titles get rewritten to be more specific via _questionize_title
    #  - Question-shaped titles ("..." ending in 吗/？/?) get compressed to a
    #    noun-phrase via _titleize_question so the H2 reads as a heading, not a question.
    if _is_legacy_fixed_title(title):
        title = _questionize_title(title, core_question)
    if title.endswith(("？", "?")) or any(marker in title for marker in ("如何", "怎样", "怎么", "能否", "能不能", "是否", "有没有", "哪些", "为什么")):
        title = _titleize_question(title) or title

    template_keys = _source_template_keys(raw, research_plan)
    evidence_mix = _required_evidence_mix(raw, template_keys)
    min_total = int(raw.get("min_total_sources") or _env_int("REPORT_MIN_TOTAL_SOURCES_PER_CHAPTER", 12))
    min_ab = int(raw.get("min_ab_sources") or _env_int("REPORT_MIN_AB_SOURCES_PER_CHAPTER", 4))
    min_counter_default = 1 if any(item in evidence_mix for item in ["counter", "counter_evidence", "risk"]) else 0
    min_counter = int(raw.get("min_counter_sources") or _env_int("REPORT_MIN_COUNTER_SOURCES_PER_DECISION_CHAPTER", max(1, min_counter_default)))
    goals = _goals_for_chapter(research_plan, raw, chapter_id, title)

    return {
        "chapter_id": chapter_id or fallback_id,
        "chapter_title": title,
        "core_question": core_question,
        "chapter_question": core_question,
        "reason_to_include": _compact(raw.get("reason_to_include") or raw.get("chapter_role") or raw.get("purpose") or core_question, 220),
        "chapter_role": _compact(raw.get("chapter_role") or raw.get("role") or core_question, 160),
        "source_template_keys": template_keys,
        "required_evidence_mix": evidence_mix,
        "min_total_sources": max(12, min_total),
        "min_ab_sources": max(4, min_ab),
        "min_counter_sources": max(0, min_counter),
        "evidence_goals": goals,
        "search_task_hints": _search_hints_for_chapter(research_plan, chapter_id, title),
        "order": int(raw.get("order") or index),
    }


def _chapters_from_evidence_goals(research_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    chapters: Dict[str, Dict[str, Any]] = {}
    for index, goal in enumerate(_as_list(research_plan.get("evidence_goals")), start=1):
        if not isinstance(goal, dict):
            continue
        key = str(goal.get("chapter_id") or goal.get("dimension_id") or goal.get("dimension_name") or f"ch_{index:02d}").strip()
        title = str(goal.get("chapter_title") or goal.get("dimension_name") or goal.get("question") or key).strip()
        if key not in chapters:
            chapters[key] = {
                "chapter_id": key if key.startswith("ch_") else f"ch_{len(chapters) + 1:02d}",
                "chapter_title": title,
                "core_question": goal.get("question") or goal.get("evidence_goal") or title,
                "evidence_goals": [],
            }
        chapters[key]["evidence_goals"].append(dict(goal))
    return list(chapters.values())


def _chapters_from_hypotheses(research_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    chapters: List[Dict[str, Any]] = []
    for index, hypothesis in enumerate(_as_list(research_plan.get("hypotheses")), start=1):
        if not isinstance(hypothesis, dict):
            continue
        statement = _compact(hypothesis.get("statement") or hypothesis.get("hypothesis_statement") or hypothesis.get("claim_to_test"), 180)
        if not statement:
            continue
        chapters.append(
            {
                "chapter_id": f"ch_{index:02d}",
                "chapter_title": _questionize_title(statement, statement),
                "core_question": statement,
                "reason_to_include": hypothesis.get("decision_use") or "用于验证核心假设",
            }
        )
    return chapters


def _chapters_from_dimensions(research_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    chapters: List[Dict[str, Any]] = []
    for index, dimension in enumerate(_as_list(research_plan.get("dimensions")), start=1):
        if not isinstance(dimension, dict):
            continue
        name = _compact(dimension.get("dimension_name") or dimension.get("name") or dimension.get("dimension"), 120)
        if not name:
            continue
        purpose = _compact(dimension.get("purpose") or name, 180)
        chapters.append(
            {
                "chapter_id": f"ch_{index:02d}",
                "chapter_title": _questionize_title(name, purpose),
                "core_question": purpose,
                "dimension_id": dimension.get("dimension_id"),
                "dimension_name": name,
            }
        )
    return chapters


def _fallback_single_chapter(query: str) -> List[Dict[str, Any]]:
    object_label = _compact(query, 80) or "这个研究问题"
    return [
        {
            "chapter_id": "ch_01",
            "chapter_title": f"{object_label}首先需要判断什么？",
            "core_question": f"回答“{query or '研究问题'}”的关键事实、证伪条件和行动含义是什么？",
            "reason_to_include": "没有更细的 planner 章节时，保留一个问题驱动的最小章节。",
        }
    ]


def infer_narrative_from_chapters(chapters: List[Dict[str, Any]], research_object: str = "") -> str:
    titles = [str(item.get("chapter_title") or "").strip() for item in chapters[:4] if str(item.get("chapter_title") or "").strip()]
    object_label = research_object or "研究对象"
    if not titles:
        return f"围绕{object_label}形成问题驱动、证据约束、可证伪的判断链。"
    return f"围绕{object_label}，依次回答：" + "；".join(titles)


def _validate_blueprint(chapters: List[Dict[str, Any]]) -> Dict[str, Any]:
    issues: List[Dict[str, Any]] = []
    ids = set()
    for chapter in chapters:
        chapter_id = str(chapter.get("chapter_id") or "").strip()
        title = str(chapter.get("chapter_title") or "").strip()
        if chapter_id in ids:
            issues.append({"type": "duplicate_chapter_id", "chapter_id": chapter_id})
        ids.add(chapter_id)
        if title in LEGACY_FIVE_TITLES:
            issues.append({"type": "legacy_fixed_title", "chapter_title": title})
        if not chapter.get("core_question"):
            issues.append({"type": "missing_core_question", "chapter_id": chapter_id})
        if not _as_list(chapter.get("required_evidence_mix")):
            issues.append({"type": "missing_required_evidence_mix", "chapter_id": chapter_id})
    return {"passed": not issues, "issues": issues}


def run_pre_layout_agent(
    *,
    query: str = "",
    research_plan: Optional[Dict[str, Any]] = None,
    report_plan: Optional[Dict[str, Any]] = None,
    llm_client: Any = None,
) -> Dict[str, Any]:
    """Standardize a dynamic chapter blueprint before search expansion."""

    del llm_client
    plan = _as_dict(research_plan)
    report_plan = _as_dict(report_plan)
    if str(os.getenv("REPORT_ENABLE_LAYOUT_COMPILER", "true")).strip().lower() in {"1", "true", "yes", "on"}:
        try:
            compiled = compile_report_layout(query=query, research_plan=plan, report_plan=report_plan)
            if _as_list(compiled.get("chapters")):
                compiled["layout_validation"] = {
                    **_as_dict(compiled.get("layout_validation")),
                    "legacy_normalizer_validation": _validate_blueprint(_as_list(compiled.get("chapters"))),
                }
                return compiled
        except Exception as exc:
            logger.exception("Layout compiler failed", extra={"query": query})
            plan = {
                **plan,
                "layout_compiler_error": str(exc),
            }
    explicit_chapters = [dict(item) for item in _as_list(plan.get("chapters")) if isinstance(item, dict)]
    quality_rules = _as_dict(plan.get("quality_rules"))
    if quality_rules.get("chapters_come_from_hypotheses"):
        raw_chapters = (
            _chapters_from_hypotheses(plan)
            or _chapters_from_dimensions(plan)
            or _chapters_from_evidence_goals(plan)
            or explicit_chapters
            or _fallback_single_chapter(query)
        )
        chapter_source = "problem_framing_hypotheses"
    else:
        raw_chapters = (
            explicit_chapters
            or _chapters_from_evidence_goals(plan)
            or _chapters_from_hypotheses(plan)
            or _chapters_from_dimensions(plan)
            or _fallback_single_chapter(query)
        )
        chapter_source = "planner_chapters"

    chapters: List[Dict[str, Any]] = []
    used_ids = set()
    for index, raw in enumerate(raw_chapters, start=1):
        chapter = normalize_chapter(raw, index=index, research_plan=plan)
        if _is_legacy_fixed_title(chapter.get("chapter_title")):
            chapter["chapter_title"] = _questionize_title(chapter["chapter_title"], chapter.get("core_question", ""))
        chapter_id = str(chapter.get("chapter_id") or f"ch_{index:02d}").strip() or f"ch_{index:02d}"
        if chapter_id in used_ids:
            chapter_id = f"ch_{index:02d}"
        chapter["chapter_id"] = chapter_id
        used_ids.add(chapter_id)
        chapters.append(chapter)

    research_object = str(plan.get("research_object") or report_plan.get("research_object") or query or "").strip()
    blueprint = {
        "agent": AGENT_NAME,
        "report_family": str(plan.get("report_family") or report_plan.get("report_family") or "dynamic_research_report").strip()
        or "dynamic_research_report",
        "research_type": str(plan.get("research_type") or report_plan.get("research_type") or "").strip(),
        "research_object": research_object,
        "narrative": infer_narrative_from_chapters(chapters, research_object),
        "chapters": chapters,
        "dropped_template_sections": _as_list(plan.get("dropped_template_sections")),
        "quality_rules": {
            "forbid_legacy_five_dimensions": True,
            "forbid_fixed_micro_sections": True,
            "chapter_must_have_question": True,
            "chapter_source": chapter_source,
            "chapters_come_from_hypotheses": bool(quality_rules.get("chapters_come_from_hypotheses")),
            "layout_stage": "pre_search_dynamic_blueprint",
            "min_sources_per_chapter": _env_int("REPORT_MIN_TOTAL_SOURCES_PER_CHAPTER", 12),
            "min_ab_sources_per_chapter": _env_int("REPORT_MIN_AB_SOURCES_PER_CHAPTER", 4),
        },
    }
    blueprint["layout_validation"] = _validate_blueprint(chapters)
    return blueprint
