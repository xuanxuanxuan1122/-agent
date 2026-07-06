from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Sequence

from rag_pipeline.contracts.report_intent import (
    PROFESSION_EDUCATION_DOMAIN,
    PROFESSION_EDUCATION_INTENT,
    contains_commercial_frame_term,
    is_profession_education_employment_topic,
    profession_chapter_frames,
    profession_forbidden_terms,
    unique_text,
)


HIGH_STAKES_RE = re.compile(
    r"投资|尽调|并购|IPO|估值|买入|卖出|市场进入|进入|布局|值得|优先级|回报|投资价值|"
    r"investment|investor|due diligence|market entry|m&a|valuation|ipo",
    re.I,
)


def _requires_strong_proof(*values: Any) -> bool:
    text = " ".join(str(value or "") for value in values)
    return bool(HIGH_STAKES_RE.search(text))


def _env_int(name: str, default: int, *, min_value: int = 0, max_value: int = 10_000) -> int:
    try:
        value = int(os.getenv(name, str(default)) or default)
    except (TypeError, ValueError):
        value = default
    return max(min_value, min(max_value, value))


def _industry_report_family(value: Any) -> bool:
    family = str(value or "").strip().lower()
    return family in {"industry_deep_report", "deep_industry_report", "industry_report", "industry_scan_report"}


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", "none", "null", ""}:
        return False
    return default


@dataclass
class Hypothesis:
    hypothesis_id: str
    statement: str
    decision_use: str = "research"
    proof_standard: str = "medium"
    counter_evidence_required: bool = False
    required_source_levels: List[str] = field(default_factory=lambda: ["A", "B"])
    required_evidence_types: List[str] = field(default_factory=list)
    metric_definitions: List[Dict[str, Any]] = field(default_factory=list)
    falsification_triggers: List[str] = field(default_factory=list)
    claim_to_test: str = ""
    must_prove: List[str] = field(default_factory=list)
    must_disprove: List[str] = field(default_factory=list)
    required_sources: List[str] = field(default_factory=list)
    evidence_bundle: Dict[str, List[str]] = field(default_factory=dict)
    minimum_evidence_bundle: str = ""


@dataclass
class Chapter:
    chapter_id: str
    chapter_title: str
    core_question: str
    chapter_question: str = ""
    reason_to_include: str = ""
    source_template_keys: List[str] = field(default_factory=list)
    required_evidence_mix: List[str] = field(default_factory=list)
    min_total_sources: int = 12
    min_ab_sources: int = 2
    min_counter_sources: int = 1
    evidence_goals: List[Dict[str, Any]] = field(default_factory=list)
    search_tasks: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class EvidenceGoal:
    goal_id: str
    dimension_id: str
    dimension_name: str
    question: str
    expected_metrics: List[str] = field(default_factory=list)
    must_have_terms: List[str] = field(default_factory=list)
    forbidden_terms: List[str] = field(default_factory=list)
    source_priority: List[str] = field(default_factory=list)
    freshness: str = "normal"
    min_sources: int = 2
    evidence_type: str = "data"
    proof_role: str = "support"
    hypothesis_id: str = ""
    hypothesis_statement: str = ""
    proof_standard: str = "medium"
    counter_evidence_required: bool = False
    required_source_levels: List[str] = field(default_factory=lambda: ["A", "B"])
    metric_definitions: List[Dict[str, Any]] = field(default_factory=list)
    decision_use: str = "research"
    chapter_id: str = ""
    chapter_title: str = ""
    chapter_question: str = ""
    required_evidence_mix: List[str] = field(default_factory=list)
    lane_targets: List[str] = field(default_factory=list)


@dataclass
class SearchTask:
    task_id: str
    agent: str
    dimension_id: str
    dimension_name: str
    query: str
    evidence_goal: str
    intent: str
    search_options: Dict[str, Any] = field(default_factory=dict)
    must_have_terms: List[str] = field(default_factory=list)
    forbidden_terms: List[str] = field(default_factory=list)
    source_priority: List[str] = field(default_factory=list)
    retriever: str = ""
    hypothesis_id: str = ""
    hypothesis_statement: str = ""
    evidence_type: str = "data"
    lane_targets: List[str] = field(default_factory=list)
    counter_evidence: bool = False
    proof_role: str = "support"
    proof_standard: str = "medium"
    metric_definition: Dict[str, Any] = field(default_factory=dict)
    decision_use: str = "research"
    chapter_id: str = ""
    chapter_title: str = ""
    chapter_question: str = ""
    evidence_goal_id: str = ""
    required_evidence_mix: List[str] = field(default_factory=list)
    min_source_level: List[str] = field(default_factory=lambda: ["A", "B"])
    research_object: str = ""
    global_required_terms: List[str] = field(default_factory=list)
    topic_anchor_terms: List[str] = field(default_factory=list)
    topic_anchor_status: str = ""
    topic_anchor_repaired: bool = False
    topic_anchor_missing_before_repair: bool = False
    chapter_focus_terms: List[str] = field(default_factory=list)
    chapter_focus_status: str = ""
    chapter_focus_repaired: bool = False
    chapter_focus_missing_before_repair: bool = False
    report_intent: str = ""


@dataclass
class ResearchPlan:
    query: str
    research_type: str
    decision_context: str
    report_family: str
    research_object: str
    key_questions: List[str]
    hypotheses: List[Hypothesis]
    chapters: List[Chapter]
    dimensions: List[Dict[str, Any]]
    evidence_goals: List[EvidenceGoal]
    search_tasks: List[SearchTask]
    core_question: str = ""
    planning_query: str = ""
    article_direction: str = ""
    report_title: str = ""
    report_subtitle: str = ""
    article_brief: Dict[str, Any] = field(default_factory=dict)
    proof_standards: Dict[str, Any] = field(default_factory=dict)
    source_requirements: Dict[str, Any] = field(default_factory=dict)
    report_depth_target: str = "standard"
    source_strategy: List[Dict[str, Any]] = field(default_factory=list)
    problem_framing: Dict[str, Any] = field(default_factory=dict)
    evidence_coverage_requirements: Dict[str, Any] = field(default_factory=dict)
    output_format: str = "brief"
    global_forbidden_terms: List[str] = field(default_factory=list)
    global_required_terms: List[str] = field(default_factory=list)
    quality_rules: Dict[str, Any] = field(default_factory=dict)


def serialize_research_plan(plan: ResearchPlan) -> Dict[str, Any]:
    return asdict(plan)


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _string_list(value: Any) -> List[str]:
    if isinstance(value, str):
        raw_items = re.split(r"[,;，；\n]+", value)
    else:
        raw_items = _as_list(value)
    return [str(item).strip() for item in raw_items if str(item).strip()]


def _compact_text(value: Any, max_chars: int = 160) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "…"


def _compact_search_terms(value: Any, *, limit: int = 8) -> List[str]:
    terms = _string_list(value)
    result: List[str] = []
    seen = set()
    for term in terms:
        key = re.sub(r"\s+", "", term.lower())
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(_compact_text(term, 28))
        if len(result) >= limit:
            break
    return result


def _derive_global_required_terms(*values: Any) -> List[str]:
    text = " ".join(str(value or "") for value in values)
    terms: List[str] = []
    if re.search(r"中国|国内", text, re.I):
        terms.append("中国")
    if re.search(r"\bAI\b|人工智能|大模型|生成式|AIGC", text, re.I):
        terms.extend(["人工智能", "AI"])
    if re.search(r"新能源汽车|新能源车|动力电池|锂电", text):
        terms.extend(["新能源汽车", "动力电池"])
    if re.search(r"半导体|芯片|集成电路", text, re.I):
        terms.extend(["半导体", "芯片"])
    deduped: List[str] = []
    for term in terms:
        if term and term not in deduped:
            deduped.append(term)
    return deduped[:6]


_REAL_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_UNICODE_QUERY_CUT_RE = re.compile(
    r"(?:商业化|机会|风险|分析|报告|研究|研判|怎么看|如何|哪些|是否|当前|未来|现状|趋势|前景)",
    re.I,
)


def _has_real_cjk(value: Any) -> bool:
    return bool(_REAL_CJK_RE.search(str(value or "")))


def _strip_query_noise(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").replace("\u3000", " ")).strip()
    text = re.sub(r"[（(]\s*(?:20\d{2}|19\d{2})\s*[）)]", "", text).strip()
    text = re.sub(r"^(?:请|帮我|麻烦|分析一下|研究一下|看一下)\s*", "", text).strip()
    return text.strip(" \t\r\n,.;:!?，。；：！？")


def _unicode_query_subject(value: Any) -> str:
    text = _strip_query_noise(value)
    if not _has_real_cjk(text):
        return ""
    first_clause = re.split(r"[，。；：！？、\n\r]", text, maxsplit=1)[0].strip()
    candidate = first_clause or text
    cut = _UNICODE_QUERY_CUT_RE.search(candidate)
    if cut and cut.start() >= 2:
        candidate = candidate[: cut.start()].strip()
    candidate = candidate.strip(" -_—：:，,。.")
    if len(candidate) > 28:
        for suffix in ("产业链", "低空经济", "人工智能", "机器人", "新能源", "半导体", "芯片", "电池", "行业", "市场", "赛道", "经济"):
            idx = candidate.find(suffix)
            if idx >= 0:
                candidate = candidate[: idx + len(suffix)]
                break
    return candidate.strip()


def _unicode_query_anchor_terms(value: Any) -> List[str]:
    subject = _unicode_query_subject(value)
    if not subject:
        return []
    candidates: List[Any] = [subject]
    compact = re.sub(r"\s+", "", subject)
    if compact.startswith("中国") and len(compact) > 2:
        candidates.append(compact[2:])
    if compact.startswith("国内") and len(compact) > 2:
        candidates.append(compact[2:])
    # Keep high-signal subtopics from the original query. This makes topic
    # guards robust when the planner returns a verbose subject such as
    # "中国低空经济产业链" but downstream queries only need "低空经济".
    for pattern in (
        r"低空经济",
        r"eVTOL",
        r"无人机",
        r"通航",
        r"人工智能",
        r"AI\s*Agent",
        r"AIGC",
        r"大模型",
        r"机器人",
        r"半导体",
        r"芯片",
        r"动力电池",
        r"固态电池",
        r"储能",
    ):
        match = re.search(pattern, str(value or ""), re.I)
        if match:
            candidates.append(match.group(0))
    return _dedupe_limited_terms(candidates, limit=8, max_chars=32)


def _looks_placeholder_or_mojibake(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    compact = re.sub(r"\s+", "", text)
    if compact.count("?") >= max(3, len(compact) // 3):
        return True
    if "\ufffd" in compact:
        return True
    return False


def _topic_compatible_with_query_anchors(query_anchors: Sequence[str], candidate: Any) -> bool:
    if not query_anchors:
        return True
    text = re.sub(r"\s+", "", str(candidate or "")).lower()
    if not text or _looks_placeholder_or_mojibake(candidate):
        return False
    # A Chinese user query should not accept an unrelated ASCII-only topic from
    # the planner, e.g. "wind energy windpower" for "中国低空经济".
    if any(_has_real_cjk(anchor) for anchor in query_anchors) and not _has_real_cjk(text):
        return False
    for anchor in query_anchors:
        key = re.sub(r"\s+", "", str(anchor or "")).lower()
        if key and (key in text or text in key):
            return True
    cjk_tokens = {
        token
        for token in re.findall(r"[\u4e00-\u9fff]{2,}", text)
        if token not in {"中国", "国内", "全球", "行业", "市场", "产业链"}
    }
    for anchor in query_anchors:
        anchor_tokens = {
            token
            for token in re.findall(r"[\u4e00-\u9fff]{2,}", str(anchor or ""))
            if token not in {"中国", "国内", "全球", "行业", "市场", "产业链"}
        }
        if cjk_tokens & anchor_tokens:
            return True
    return False


_PLAN_GENERIC_DIMENSION_TERMS = (
    "商业化",
    "机会",
    "风险",
    "投资",
    "判断",
    "落地",
    "趋势",
    "格局",
)

_PLAN_EVIDENCE_DIMENSION_TERMS = (
    "政策",
    "产业链",
    "供应链",
    "应用场景",
    "市场规模",
    "投资判断",
    "客户案例",
    "财报",
    "公告",
    "招股书",
    "反证",
    "风险事件",
)

_PLAN_TOPIC_ANCHOR_GENERIC_TERMS = {
    "产业链",
    "供应链",
    "中国市场",
    "市场",
    "商业化",
    "政策",
    "应用场景",
    "投资判断",
    "风险",
    "机会",
    "研究",
    "分析",
    "判断",
    "市场规模",
    "需求",
    "数据",
    "统计",
    "口径",
    "来源",
}


def _dedupe_limited_terms(values: List[Any], *, limit: int = 8, max_chars: int = 32) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        text = text.strip(" ,;:!?，。；：！？、()（）[]【】{}《》\"'")
        if not text:
            continue
        key = re.sub(r"\s+", "", text.lower())
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(text[:max_chars].strip())
        if len(result) >= limit:
            break
    return result


def _term_in_plan_text(term: str, *values: Any) -> bool:
    text = re.sub(r"\s+", "", " ".join(str(value or "") for value in values)).lower()
    key = re.sub(r"\s+", "", str(term or "")).lower()
    return bool(key and key in text)


def _topic_anchor_variants(term: Any) -> List[str]:
    text = re.sub(r"\s+", " ", str(term or "")).strip()
    if not text:
        return []
    variants = [text]
    compact = re.sub(r"\s+", "", text)
    if compact.startswith("中国") and len(compact) > 2:
        variants.append(compact[2:])
    if compact.startswith("国内") and len(compact) > 2:
        variants.append(compact[2:])
    return _dedupe_limited_terms(variants, limit=3, max_chars=32)


def _is_generic_topic_anchor(term: Any) -> bool:
    compact = re.sub(r"\s+", "", str(term or "")).strip()
    if not compact:
        return True
    return compact in _PLAN_TOPIC_ANCHOR_GENERIC_TERMS


def _derive_topic_anchor_terms_for_plan(
    *,
    plan_query: str,
    research_object: str,
    global_required_terms: List[str],
    explicit_terms: List[str],
) -> List[str]:
    candidates: List[Any] = [*explicit_terms, research_object, *global_required_terms]
    anchors: List[Any] = []
    for value in candidates:
        if _is_generic_topic_anchor(value):
            continue
        anchors.extend(_topic_anchor_variants(value))
    if not anchors and re.search(r"\bAI\s*Agent\b|智能体|智能代理", plan_query, re.I):
        anchors.append("AI Agent")
    return _dedupe_limited_terms(anchors, limit=6, max_chars=32)


def _derive_plan_dimensions(plan_query: str, payload: Dict[str, Any], terms: Sequence[str]) -> List[str]:
    values: List[Any] = [plan_query, payload.get("query"), payload.get("research_object")]
    values.extend(_as_list(payload.get("key_questions")))
    values.extend(str(item.get("dimension_name") or item.get("name") or "") for item in _as_list(payload.get("dimensions")) if isinstance(item, dict))
    found = [term for term in terms if _term_in_plan_text(term, *values)]
    return _dedupe_limited_terms(found, limit=12, max_chars=24)


def _dict_list(value: Any) -> List[Dict[str, Any]]:
    return [dict(item) for item in _as_list(value) if isinstance(item, dict)]


def normalize_chapter(raw: Dict[str, Any], *, fallback_index: int = 1, query: str = "") -> Dict[str, Any]:
    chapter = _as_dict(raw)
    chapter_id = str(chapter.get("chapter_id") or chapter.get("id") or chapter.get("dimension_id") or f"ch_{fallback_index:02d}").strip()
    chapter_title = str(
        chapter.get("chapter_title")
        or chapter.get("title")
        or chapter.get("dimension_name")
        or chapter.get("dimension")
        or chapter.get("name")
        or query
        or f"Chapter {fallback_index}"
    ).strip()
    core_question = str(
        chapter.get("core_question")
        or chapter.get("chapter_question")
        or chapter.get("question")
        or chapter.get("purpose")
        or chapter_title
    ).strip()
    required_mix = _string_list(chapter.get("required_evidence_mix"))
    if not required_mix:
        required_mix = ["official_data", "market_research", "company_filing", "case", "counter_evidence"]
    return {
        "chapter_id": chapter_id,
        "chapter_title": chapter_title,
        "core_question": core_question,
        "chapter_question": str(chapter.get("chapter_question") or core_question).strip(),
        "reason_to_include": str(chapter.get("reason_to_include") or chapter.get("purpose") or "").strip(),
        "source_template_keys": _string_list(chapter.get("source_template_keys") or chapter.get("template_keys")),
        "required_evidence_mix": required_mix,
        "min_total_sources": int(chapter.get("min_total_sources") or 12),
        "min_ab_sources": int(chapter.get("min_ab_sources") or 2),
        "min_counter_sources": int(chapter.get("min_counter_sources") or 1),
        "key_chapter": _as_bool(chapter.get("key_chapter"), fallback_index <= 2),
        "chapter_evidence_contract": _as_dict(chapter.get("chapter_evidence_contract")),
        "evidence_goals": _dict_list(chapter.get("evidence_goals")),
        "search_tasks": _dict_list(chapter.get("search_tasks")),
    }


def _profession_chapters_from_frames(query: str) -> List[Dict[str, Any]]:
    chapters: List[Dict[str, Any]] = []
    for index, frame in enumerate(profession_chapter_frames(query), start=1):
        chapters.append(
            normalize_chapter(
                {
                    **frame,
                    "chapter_role": "profession_education_frame",
                    "reason_to_include": "该章节服务于就业变化、岗位能力和教育培养调整类报告的主线判断。",
                    "required_evidence_mix": ["official_data", "market_research", "case", "counter_evidence"],
                    "source_template_keys": ["employment_signal", "education_policy", "curriculum_case", "recruitment_signal"],
                    "min_total_sources": 4,
                    "min_ab_sources": 0,
                    "min_counter_sources": 1,
                },
                fallback_index=index,
                query=query,
            )
        )
        chapters[-1]["chapter_role"] = "profession_education_frame"
        chapters[-1]["final_outline_locked"] = False
    return chapters


def _profession_chapters_need_rewrite(chapters: Sequence[Dict[str, Any]]) -> bool:
    if not chapters:
        return True
    text = " ".join(
        str(chapter.get(field) or "")
        for chapter in chapters
        for field in ("chapter_title", "core_question", "chapter_question")
    )
    if contains_commercial_frame_term(text):
        return True
    core_hits = sum(
        1
        for chapter in chapters
        if re.search(r"就业|岗位|能力|培养|课程|高校|招聘|职业", str(chapter.get("chapter_title") or chapter.get("core_question") or ""))
    )
    return core_hits < max(2, min(4, len(chapters)))


def normalize_hypothesis(raw: Dict[str, Any], *, fallback_index: int = 1, query: str = "") -> Dict[str, Any]:
    payload = _as_dict(raw)
    hypothesis_id = str(payload.get("hypothesis_id") or payload.get("id") or f"H{fallback_index}").strip()
    statement = str(
        payload.get("statement")
        or payload.get("hypothesis_statement")
        or payload.get("hypothesis")
        or payload.get("question")
        or query
        or f"Hypothesis {fallback_index}"
    ).strip()
    proof_standard = str(payload.get("proof_standard") or payload.get("standard") or "medium").strip().lower()
    if proof_standard not in {"strong", "medium", "weak"}:
        proof_standard = "medium"
    decision_use = str(payload.get("decision_use") or "research").strip()
    strong_required = _requires_strong_proof(query, statement, decision_use)
    if strong_required:
        proof_standard = "strong"
    return {
        "hypothesis_id": hypothesis_id,
        "statement": statement,
        "hypothesis_statement": statement,
        "dimension_id": str(payload.get("dimension_id") or "").strip(),
        "dimension_name": str(payload.get("dimension_name") or "").strip(),
        "claim_to_test": str(payload.get("claim_to_test") or statement).strip(),
        "decision_use": decision_use,
        "proof_standard": proof_standard,
        "counter_evidence_required": bool(payload.get("counter_evidence_required", False) or strong_required),
        "required_source_levels": _string_list(payload.get("required_source_levels")) or ["A", "B"],
        "required_evidence_types": _string_list(payload.get("required_evidence_types")) or ["metric", "source_check", "case", "counter"],
        "metric_definitions": _dict_list(payload.get("metric_definitions")),
        "falsification_triggers": _string_list(payload.get("falsification_triggers")),
        "must_prove": _string_list(payload.get("must_prove") or payload.get("must_prove_terms")),
        "must_disprove": _string_list(payload.get("must_disprove") or payload.get("must_disprove_terms")),
        "required_sources": _string_list(payload.get("required_sources")),
        "evidence_bundle": {
            str(key): _string_list(value)
            for key, value in _as_dict(payload.get("evidence_bundle")).items()
            if _string_list(value)
        },
        "minimum_evidence_bundle": str(payload.get("minimum_evidence_bundle") or "").strip(),
    }


def _default_hypotheses(
    *,
    query: str,
    dimensions: List[Dict[str, Any]],
    tasks: List[Dict[str, Any]],
    key_questions: List[str],
) -> List[Dict[str, Any]]:
    seeds: List[Dict[str, Any]] = []
    for dimension in dimensions:
        statement = str(dimension.get("purpose") or dimension.get("dimension_name") or "").strip()
        if statement:
            seeds.append(
                {
                    "statement": statement,
                    "dimension_id": dimension.get("dimension_id"),
                    "dimension_name": dimension.get("dimension_name"),
                }
            )
    if not seeds:
        for task in tasks[:5]:
            statement = str(task.get("evidence_goal") or task.get("query") or "").strip()
            if statement:
                seeds.append({"statement": statement, "dimension_id": task.get("dimension_id"), "dimension_name": task.get("dimension_name")})
    if not seeds:
        for question in key_questions[:5]:
            seeds.append({"statement": question})
    if not seeds and query:
        seeds.append({"statement": query})

    hypotheses: List[Dict[str, Any]] = []
    strong_required = _requires_strong_proof(query)
    for index, seed in enumerate(seeds[:8], start=1):
        hypotheses.append(
            normalize_hypothesis(
                {
                    "hypothesis_id": f"H{index}",
                    "statement": seed.get("statement"),
                    "decision_use": "investment_or_market_entry" if strong_required else "research",
                    "proof_standard": "strong" if strong_required else "medium",
                    "counter_evidence_required": strong_required,
                    "required_source_levels": ["A", "B"],
                },
                fallback_index=index,
                query=query,
            )
            | {
                "dimension_id": str(seed.get("dimension_id") or "").strip(),
                "dimension_name": str(seed.get("dimension_name") or "").strip(),
            }
        )
    return hypotheses


def normalize_search_task(raw: Dict[str, Any], *, fallback_index: int = 1) -> Dict[str, Any]:
    task = _as_dict(raw)
    task_id = str(task.get("task_id") or f"dynamic_iqs_{fallback_index:03d}").strip()
    dimension_id = str(task.get("dimension_id") or task.get("chapter_id") or "").strip()
    dimension_name = str(task.get("dimension_name") or task.get("dimension") or "").strip()
    chapter_id = str(task.get("chapter_id") or dimension_id or "").strip()
    chapter_title = str(task.get("chapter_title") or task.get("chapter") or dimension_name or "").strip()
    chapter_question = str(task.get("chapter_question") or task.get("core_question") or "").strip()
    query = _compact_text(task.get("query") or task.get("text") or "", 80)
    evidence_goal = str(task.get("evidence_goal") or task.get("goal") or task.get("targets_gap") or "").strip()
    intent = str(task.get("intent") or "analysis").strip().lower()
    agent = str(task.get("agent") or "iqs").strip().lower()
    if agent not in {"iqs", "rag", "both", "all"} and not agent.startswith("iqs_"):
        agent = "iqs"
    retrieval_mode = str(task.get("retrieval_mode") or "").strip().lower()
    if retrieval_mode not in {"deep", "normal", "hybrid"}:
        retrieval_mode = ""
    return {
        "task_id": task_id,
        "agent": agent,
        "dimension_id": dimension_id,
        "dimension_name": dimension_name,
        "chapter_id": chapter_id,
        "chapter_title": chapter_title,
        "chapter_question": chapter_question,
        "query": query,
        "query_contract": _as_dict(task.get("query_contract")),
        "query_quality": _as_dict(task.get("query_quality")),
        "evidence_goal": evidence_goal,
        "targets_gap": str(task.get("targets_gap") or evidence_goal).strip(),
        "evidence_goal_id": str(task.get("evidence_goal_id") or task.get("goal_id") or "").strip(),
        "requirement_id": str(
            task.get("requirement_id")
            or task.get("evidence_requirement_id")
            or task.get("slot_id")
            or task.get("evidence_goal_id")
            or task.get("goal_id")
            or ""
        ).strip(),
        "gap_id": str(task.get("gap_id") or task.get("mandatory_proof_id") or task.get("proof_id") or task.get("targets_gap") or "").strip(),
        "gap_type": str(task.get("gap_type") or task.get("type") or "").strip(),
        "type": str(task.get("type") or "").strip(),
        "reason": str(task.get("reason") or "").strip(),
        "blocking_gaps": _string_list(task.get("blocking_gaps") or task.get("missing")),
        "required_fields": _string_list(task.get("required_fields")),
        "origin_node": str(task.get("origin_node") or "").strip(),
        "loop_name": str(task.get("loop_name") or "").strip(),
        "intent": intent,
        "search_options": _as_dict(task.get("search_options")),
        "must_have_terms": _compact_search_terms(task.get("must_have_terms"), limit=5),
        "forbidden_terms": _compact_search_terms(task.get("forbidden_terms"), limit=24),
        "source_priority": _compact_search_terms(task.get("source_priority"), limit=5),
        "retriever": str(task.get("retriever") or task.get("source_type") or "").strip(),
        "hypothesis_id": str(task.get("hypothesis_id") or "").strip(),
        "hypothesis_statement": str(task.get("hypothesis_statement") or task.get("hypothesis") or "").strip(),
        "claim_strength_ceiling": str(task.get("claim_strength_ceiling") or task.get("claim_strength") or "").strip().lower(),
        "evidence_type": str(task.get("evidence_type") or task.get("intent") or "data").strip().lower(),
        "lane_targets": _string_list(task.get("lane_targets") or task.get("lanes")),
        "counter_evidence": bool(task.get("counter_evidence") or str(task.get("proof_role") or "").strip().lower() == "counter"),
        "proof_role": str(task.get("proof_role") or ("counter" if task.get("counter_evidence") else "support")).strip().lower(),
        "proof_standard": str(task.get("proof_standard") or "medium").strip().lower(),
        "metric_definition": _as_dict(task.get("metric_definition")),
        "decision_use": str(task.get("decision_use") or "research").strip(),
        "required_evidence_mix": _string_list(task.get("required_evidence_mix")),
        "min_source_level": _string_list(task.get("min_source_level") or task.get("required_source_levels")) or ["A", "B"],
        "deep_search_variant": bool(task.get("deep_search_variant")),
        "prefer_deep": bool(task.get("prefer_deep")),
        "deep_reason": str(task.get("deep_reason") or "").strip(),
        "deep_status": str(task.get("deep_status") or "").strip(),
        "deep_skip_reason": str(task.get("deep_skip_reason") or "").strip(),
        "engineTypes": _string_list(task.get("engineTypes")),
        "retrieval_mode": retrieval_mode,
        "retrieval_reason": str(task.get("retrieval_reason") or "").strip(),
        "primary_provider": str(task.get("primary_provider") or "").strip(),
        "fallback_providers": _string_list(task.get("fallback_providers")),
        "provider": str(task.get("provider") or "").strip(),
        "repair_source": str(task.get("repair_source") or "").strip(),
        "gap_repair_round": task.get("gap_repair_round"),
        "scheduled_lane": str(task.get("scheduled_lane") or "").strip(),
        "scheduled_lane_type": str(task.get("scheduled_lane_type") or "").strip(),
        "lane_focus": str(task.get("lane_focus") or "").strip(),
        "allowed_domains": _string_list(task.get("allowed_domains")),
        "research_object": str(task.get("research_object") or "").strip(),
        "global_required_terms": _compact_search_terms(task.get("global_required_terms"), limit=6),
        "topic_anchor_terms": _compact_search_terms(task.get("topic_anchor_terms"), limit=6),
        "topic_anchor_status": str(task.get("topic_anchor_status") or "").strip(),
        "topic_anchor_repaired": _as_bool(task.get("topic_anchor_repaired"), False),
        "topic_anchor_missing_before_repair": _as_bool(task.get("topic_anchor_missing_before_repair"), False),
        "chapter_focus_terms": _compact_search_terms(task.get("chapter_focus_terms"), limit=6),
        "chapter_focus_status": str(task.get("chapter_focus_status") or "").strip(),
        "chapter_focus_repaired": _as_bool(task.get("chapter_focus_repaired"), False),
        "chapter_focus_missing_before_repair": _as_bool(task.get("chapter_focus_missing_before_repair"), False),
        "report_intent": str(task.get("report_intent") or "").strip(),
    }


def normalize_evidence_goal(raw: Dict[str, Any], *, fallback_index: int = 1) -> Dict[str, Any]:
    goal = _as_dict(raw)
    dimension_id = str(goal.get("dimension_id") or goal.get("chapter_id") or "").strip()
    dimension_name = str(goal.get("dimension_name") or goal.get("dimension") or goal.get("chapter_title") or "").strip()
    chapter_id = str(goal.get("chapter_id") or dimension_id or "").strip()
    chapter_title = str(goal.get("chapter_title") or dimension_name or "").strip()
    chapter_question = str(goal.get("chapter_question") or goal.get("core_question") or "").strip()
    return {
        "goal_id": str(goal.get("goal_id") or goal.get("id") or goal.get("question") or f"goal_{fallback_index:03d}").strip(),
        "dimension_id": dimension_id,
        "dimension_name": dimension_name,
        "chapter_id": chapter_id,
        "chapter_title": chapter_title,
        "chapter_question": chapter_question,
        "question": str(goal.get("question") or goal.get("evidence_goal") or "").strip(),
        "expected_metrics": _string_list(goal.get("expected_metrics")),
        "must_have_terms": _string_list(goal.get("must_have_terms")),
        "forbidden_terms": _string_list(goal.get("forbidden_terms")),
        "source_priority": _string_list(goal.get("source_priority")),
        "freshness": str(goal.get("freshness") or "normal").strip(),
        "min_sources": int(goal.get("min_sources") or 2),
        "evidence_type": str(goal.get("evidence_type") or "data").strip().lower(),
        "proof_role": str(goal.get("proof_role") or goal.get("role") or "").strip().lower(),
        "hypothesis_id": str(goal.get("hypothesis_id") or "").strip(),
        "hypothesis_statement": str(goal.get("hypothesis_statement") or goal.get("hypothesis") or "").strip(),
        "proof_standard": str(goal.get("proof_standard") or "medium").strip().lower(),
        "counter_evidence_required": bool(goal.get("counter_evidence_required", False)),
        "required_source_levels": _string_list(goal.get("required_source_levels")) or ["A", "B"],
        "metric_definitions": _dict_list(goal.get("metric_definitions")),
        "decision_use": str(goal.get("decision_use") or "research").strip(),
        "required_evidence_mix": _string_list(goal.get("required_evidence_mix")),
        "lane_targets": _string_list(goal.get("lane_targets") or goal.get("lanes")),
    }


_LANE_TYPES = {
    "official_data",
    "filing_company",
    "market_research",
    "news_event",
    "technology_product",
    "customer_case",
}


def _infer_lane_targets(task: Dict[str, Any]) -> List[str]:
    explicit = [item for item in _string_list(task.get("lane_targets") or task.get("lanes")) if item in _LANE_TYPES]
    if explicit:
        return explicit[:3]
    text = " ".join(
        [
            str(task.get("intent") or ""),
            str(task.get("evidence_type") or ""),
            str(task.get("proof_role") or ""),
            " ".join(_string_list(task.get("source_priority"))),
            str(task.get("query") or ""),
            str(task.get("evidence_goal") or ""),
        ]
    ).lower()
    scores = {
        "official_data": ["statistics", "data", "policy", "gov", "official", "stats", "统计", "政府", "监管", "协会"],
        "filing_company": ["filing", "company", "financial", "annual_report", "prospectus", "公告", "财报", "年报", "招股书"],
        "market_research": ["market", "research", "analysis", "consulting", "brokerage", "研报", "市场", "行业报告"],
        "news_event": ["news", "risk", "event", "counter", "tender", "order", "诉讼", "中标", "事故", "负面"],
        "technology_product": ["academic", "technology", "technical", "product", "patent", "论文", "专利", "技术", "产品"],
        "customer_case": ["case", "customer", "business", "application", "roi", "procurement", "客户", "案例", "采购"],
    }
    hits: Dict[str, int] = {}
    for lane, terms in scores.items():
        score = sum(1 for term in terms if term.lower() in text)
        if score:
            hits[lane] = score
    if str(task.get("proof_role") or "").lower() == "counter" or bool(task.get("counter_evidence")):
        hits["news_event"] = hits.get("news_event", 0) + 2
    if not hits:
        return ["official_data"] if str(task.get("proof_role") or "").lower() in {"metric", "source_check"} else ["market_research"]
    return sorted(hits, key=lambda lane: hits[lane], reverse=True)[:3]


def _support_task_for_hypothesis(hypothesis: Dict[str, Any], *, fallback_index: int) -> Dict[str, Any]:
    statement = str(hypothesis.get("statement") or hypothesis.get("hypothesis_statement") or "").strip()
    dimension_id = str(hypothesis.get("dimension_id") or f"hypothesis_{fallback_index}").strip()
    dimension_name = str(hypothesis.get("dimension_name") or statement or f"Hypothesis {fallback_index}").strip()
    return normalize_search_task(
        {
            "task_id": f"hypothesis_{hypothesis.get('hypothesis_id') or fallback_index}_support",
            "agent": "iqs",
            "dimension_id": dimension_id,
            "dimension_name": dimension_name,
            "query": f"{statement} official data market research evidence".strip(),
            "evidence_goal": statement,
            "intent": "data",
            "must_have_terms": _string_list(hypothesis.get("required_evidence_types")) or [term for term in [dimension_name] if term],
            "source_priority": ["official", "statistics", "research_report"],
            "hypothesis_id": hypothesis.get("hypothesis_id"),
            "hypothesis_statement": statement,
            "proof_role": "support",
            "proof_standard": hypothesis.get("proof_standard") or "medium",
            "evidence_type": "data",
            "lane_targets": ["official_data", "market_research"],
            "decision_use": hypothesis.get("decision_use") or "research",
        },
        fallback_index=fallback_index,
    )


def _counter_task_for_hypothesis(hypothesis: Dict[str, Any], support_task: Dict[str, Any], *, fallback_index: int) -> Dict[str, Any]:
    statement = str(hypothesis.get("statement") or hypothesis.get("hypothesis_statement") or support_task.get("evidence_goal") or "").strip()
    dimension_id = str(support_task.get("dimension_id") or hypothesis.get("dimension_id") or f"hypothesis_{fallback_index}").strip()
    dimension_name = str(support_task.get("dimension_name") or hypothesis.get("dimension_name") or statement or f"Hypothesis {fallback_index}").strip()
    return normalize_search_task(
        {
            "task_id": f"hypothesis_{hypothesis.get('hypothesis_id') or fallback_index}_counter",
            "agent": "iqs",
            "dimension_id": dimension_id,
            "dimension_name": dimension_name,
            "query": f"{statement} 反证 风险 失败案例 负面 替代方案 客户不买账",
            "evidence_goal": f"寻找可推翻或限制该判断的反向证据：{statement}",
            "intent": "risk",
            "must_have_terms": _string_list(support_task.get("must_have_terms"))[:2],
            "forbidden_terms": _string_list(support_task.get("forbidden_terms")),
            "source_priority": ["news", "risk", "case", "lawsuit", "customer"],
            "hypothesis_id": hypothesis.get("hypothesis_id"),
            "hypothesis_statement": statement,
            "proof_role": "counter",
            "counter_evidence": True,
            "proof_standard": hypothesis.get("proof_standard") or support_task.get("proof_standard") or "medium",
            "evidence_type": "risk",
            "lane_targets": ["news_event", "customer_case", "market_research"],
            "decision_use": hypothesis.get("decision_use") or support_task.get("decision_use") or "research",
        },
        fallback_index=fallback_index,
    )


_BUNDLE_ROLE_CONFIG = {
    "metric": {
        "intent": "statistics",
        "evidence_type": "metric",
        "lane_targets": ["official_data", "market_research"],
        "source_priority": ["official", "statistics", "association", "research_report"],
    },
    "source_check": {
        "intent": "source_check",
        "evidence_type": "source_check",
        "lane_targets": ["official_data", "filing_company", "market_research"],
        "source_priority": ["official", "filing", "annual_report", "association", "research_report"],
    },
    "case": {
        "intent": "case",
        "evidence_type": "case",
        "lane_targets": ["customer_case", "filing_company", "technology_product"],
        "source_priority": ["company", "filing", "customer", "procurement", "case"],
    },
    "technology_product": {
        "intent": "technology",
        "evidence_type": "technology_product",
        "lane_targets": ["technology_product", "official_data", "market_research"],
        "source_priority": ["product_doc", "technical_standard", "patent", "company_official", "whitepaper"],
    },
    "expert": {
        "intent": "research",
        "evidence_type": "expert",
        "lane_targets": ["market_research", "official_data"],
        "source_priority": ["brokerage", "association", "research_report", "whitepaper"],
    },
}


def _bundle_terms(hypothesis: Dict[str, Any], role: str) -> List[str]:
    bundle = _as_dict(hypothesis.get("evidence_bundle"))
    terms = _string_list(bundle.get(role))
    if terms:
        return terms[:8]
    if role == "counter":
        return _string_list(hypothesis.get("must_disprove"))[:8]
    if role in {"metric", "source_check", "case", "technology_product", "expert"}:
        return (_string_list(hypothesis.get("must_prove")) + _string_list(hypothesis.get("required_evidence_types")))[:8]
    return _string_list(hypothesis.get("must_prove"))[:8]


def _bundle_task_for_hypothesis(hypothesis: Dict[str, Any], role: str, *, fallback_index: int) -> Dict[str, Any]:
    statement = str(hypothesis.get("claim_to_test") or hypothesis.get("statement") or hypothesis.get("hypothesis_statement") or "").strip()
    dimension_id = str(hypothesis.get("dimension_id") or f"hypothesis_{hypothesis.get('hypothesis_id') or fallback_index}").strip()
    dimension_name = str(hypothesis.get("dimension_name") or statement or f"Hypothesis {fallback_index}").strip()
    config = dict(_BUNDLE_ROLE_CONFIG.get(role) or {})
    terms = _bundle_terms(hypothesis, role)
    role_hint = {
        "metric": "market size price capacity margin shipment penetration utilization",
        "source_check": "official filing annual report association brokerage verification",
        "case": "customer certification order mass production supply contract case",
        "technology_product": "product docs technology standard patent architecture benchmark",
        "expert": "brokerage association industry research expert view",
    }.get(role, role)
    return normalize_search_task(
        {
            "task_id": f"hypothesis_{hypothesis.get('hypothesis_id') or fallback_index}_{role}",
            "agent": "iqs",
            "dimension_id": dimension_id,
            "dimension_name": dimension_name,
            "query": " ".join(part for part in [statement, role_hint, " ".join(terms)] if part).strip(),
            "evidence_goal": f"{role}: {statement}",
            "intent": config.get("intent") or role,
            "must_have_terms": terms or [statement],
            "source_priority": config.get("source_priority") or [],
            "hypothesis_id": hypothesis.get("hypothesis_id"),
            "hypothesis_statement": statement,
            "proof_role": role,
            "proof_standard": hypothesis.get("proof_standard") or "strong",
            "evidence_type": config.get("evidence_type") or role,
            "lane_targets": config.get("lane_targets") or [],
            "decision_use": hypothesis.get("decision_use") or "research",
            "metric_definition": (_dict_list(hypothesis.get("metric_definitions")) or [{}])[0] if role == "metric" else {},
        },
        fallback_index=fallback_index,
    )


def _ensure_hypothesis_task_contract(tasks: List[Dict[str, Any]], hypotheses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    completed: List[Dict[str, Any]] = []
    for task in tasks:
        copied = dict(task)
        copied["lane_targets"] = _infer_lane_targets(copied)
        completed.append(copied)
    tasks_by_hypothesis: Dict[str, List[Dict[str, Any]]] = {}
    for task in completed:
        hypothesis_id = str(task.get("hypothesis_id") or "").strip()
        if hypothesis_id:
            tasks_by_hypothesis.setdefault(hypothesis_id, []).append(task)
    next_index = len(completed) + 1
    for hypothesis in hypotheses:
        hypothesis_id = str(hypothesis.get("hypothesis_id") or "").strip()
        if not hypothesis_id:
            continue
        bucket = tasks_by_hypothesis.get(hypothesis_id, [])
        if not bucket:
            support = _support_task_for_hypothesis(hypothesis, fallback_index=next_index)
            next_index += 1
            support["lane_targets"] = _infer_lane_targets(support)
            completed.append(support)
            bucket = [support]
        has_counter = any(str(task.get("proof_role") or "").lower() == "counter" or bool(task.get("counter_evidence")) for task in bucket)
        if bool(hypothesis.get("counter_evidence_required", False)) and not has_counter:
            counter = _counter_task_for_hypothesis(hypothesis, bucket[0], fallback_index=next_index)
            next_index += 1
            counter["lane_targets"] = _infer_lane_targets(counter)
            completed.append(counter)
            bucket.append(counter)
        roles_present = {str(task.get("proof_role") or "").strip().lower() for task in bucket if str(task.get("proof_role") or "").strip()}
        for required_role in ["metric", "source_check", "case", "technology_product", "expert"]:
            if required_role in roles_present:
                continue
            task = _bundle_task_for_hypothesis(hypothesis, required_role, fallback_index=next_index)
            next_index += 1
            task["lane_targets"] = _infer_lane_targets(task)
            completed.append(task)
            bucket.append(task)
            roles_present.add(required_role)
    return completed


def _dedupe_plan_items(items: List[Any], *, id_key: str, fallback_keys: List[str]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        explicit_id = str(item.get(id_key) or "").strip().lower()
        if explicit_id:
            key = (id_key, explicit_id)
        else:
            fallback = tuple(str(item.get(field) or "").strip().lower() for field in fallback_keys)
            if not any(fallback):
                deduped.append(item)
                continue
            key = ("fallback", fallback)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _chapter_trim_score(chapter: Dict[str, Any], query: str, index: int) -> tuple:
    text = " ".join(
        str(chapter.get(key) or "")
        for key in ("chapter_title", "core_question", "chapter_question", "reason_to_include")
    ).lower()
    query_terms = [part for part in re.split(r"\W+", str(query or "").lower()) if len(part) >= 3]
    query_overlap = sum(1 for term in query_terms[:12] if term and term in text)
    mix = {str(item or "").strip().lower() for item in _string_list(chapter.get("required_evidence_mix"))}
    proof_score = sum(
        1
        for wanted in ("official_data", "market_research", "company_filing", "case", "customer_case", "metric")
        if wanted in mix
    )
    nested_count = len(_as_list(chapter.get("evidence_goals"))) + len(_as_list(chapter.get("search_tasks")))
    return (
        1 if _as_bool(chapter.get("key_chapter")) else 0,
        min(query_overlap, 5),
        min(proof_score, 5),
        min(nested_count, 8),
        -index,
    )


def enforce_research_plan_chapter_limits(plan: Dict[str, Any], *, query: str = "") -> Dict[str, Any]:
    payload = dict(plan or {})
    if not _industry_report_family(payload.get("report_family")):
        return payload
    target_count = _env_int("REPORT_TARGET_CORE_CHAPTER_COUNT", 5, min_value=4, max_value=8)
    max_count = _env_int("REPORT_MAX_CORE_CHAPTER_COUNT", 6, min_value=4, max_value=10)
    chapters = [dict(item) for item in _as_list(payload.get("chapters")) if isinstance(item, dict)]
    original_chapter_ids = {
        str(item.get("chapter_id") or item.get("id") or "").strip()
        for item in chapters
        if str(item.get("chapter_id") or item.get("id") or "").strip()
    }
    diagnostics = dict(_as_dict(payload.get("planning_diagnostics")))
    diagnostics.setdefault("target_core_chapter_count", target_count)
    diagnostics.setdefault("max_core_chapter_count", max_count)
    diagnostics["original_chapter_count"] = len(chapters)
    if chapters:
        for index, chapter in enumerate(chapters, start=1):
            chapter["key_chapter"] = _as_bool(chapter.get("key_chapter"), index <= 2)
            contract = dict(_as_dict(chapter.get("chapter_evidence_contract")))
            contract.setdefault(
                "required_traceable_ab",
                _env_int("REPORT_KEY_CHAPTER_MIN_TRACEABLE_AB", 3, min_value=1, max_value=10)
                if _as_bool(chapter.get("key_chapter"))
                else _env_int("REPORT_MIN_CORE_AB_SOURCES_PER_CHAPTER", 2, min_value=1, max_value=10),
            )
            contract.setdefault("required_proof_roles", ["metric", "source_check", "case"] if _as_bool(chapter.get("key_chapter")) else ["source_check", "case"])
            chapter["chapter_evidence_contract"] = contract
        if len(chapters) > max_count:
            ranked = sorted(
                enumerate(chapters, start=1),
                key=lambda item: _chapter_trim_score(item[1], query or payload.get("query"), item[0]),
                reverse=True,
            )
            keep_ids = {
                str(chapter.get("chapter_id") or chapter.get("id") or index)
                for index, chapter in ranked[:max_count]
            }
            trimmed = [chapter for _, chapter in ranked[max_count:]]
            chapters = [
                chapter
                for chapter in chapters
                if str(chapter.get("chapter_id") or chapter.get("id") or "") in keep_ids
            ]
            diagnostics["trimmed_chapters"] = [
                {
                    "chapter_id": chapter.get("chapter_id") or chapter.get("id"),
                    "chapter_title": chapter.get("chapter_title") or chapter.get("title"),
                    "reason": "over_max_core_chapter_count",
                }
                for chapter in trimmed
            ]
    allowed_chapter_ids = {str(item.get("chapter_id") or item.get("id") or "").strip() for item in chapters if str(item.get("chapter_id") or item.get("id") or "").strip()}
    if allowed_chapter_ids:
        def keep_by_chapter(item: Dict[str, Any]) -> bool:
            chapter_id = str(item.get("chapter_id") or "").strip()
            if chapter_id:
                if chapter_id in original_chapter_ids:
                    return chapter_id in allowed_chapter_ids
                if chapter_id in allowed_chapter_ids:
                    return True
                dimension_id = str(item.get("dimension_id") or "").strip()
                if dimension_id and dimension_id == chapter_id and dimension_id not in original_chapter_ids:
                    return True
                return True
            dimension_id = str(item.get("dimension_id") or "").strip()
            if dimension_id and dimension_id in original_chapter_ids:
                return dimension_id in allowed_chapter_ids
            return True

        def mark_global_task(item: Any) -> Any:
            if not isinstance(item, dict):
                return item
            payload_item = dict(item)
            chapter_id = str(payload_item.get("chapter_id") or "").strip()
            dimension_id = str(payload_item.get("dimension_id") or "").strip()
            if chapter_id and chapter_id not in allowed_chapter_ids and chapter_id not in original_chapter_ids:
                payload_item.setdefault("scope", "global_evidence_task")
                payload_item.setdefault("report_level", True)
            elif not chapter_id and dimension_id and dimension_id not in original_chapter_ids:
                payload_item.setdefault("scope", "global_evidence_task")
                payload_item.setdefault("report_level", True)
            return payload_item

        payload["dimensions"] = [item for item in _as_list(payload.get("dimensions")) if not isinstance(item, dict) or keep_by_chapter(item)]
        payload["evidence_goals"] = [mark_global_task(item) for item in _as_list(payload.get("evidence_goals")) if not isinstance(item, dict) or keep_by_chapter(item)]
        payload["search_tasks"] = [mark_global_task(item) for item in _as_list(payload.get("search_tasks")) if not isinstance(item, dict) or keep_by_chapter(item)]
    payload["chapters"] = chapters
    payload["target_core_chapter_count"] = target_count
    payload["max_core_chapter_count"] = max_count
    payload["planning_diagnostics"] = diagnostics
    return payload


def normalize_research_plan(raw: Dict[str, Any], *, query: str = "") -> Dict[str, Any]:
    payload = _as_dict(raw)
    article_brief = _as_dict(payload.get("article_brief"))
    trusted_query = str(query or "").strip()
    plan_query = str(trusted_query or payload.get("planning_query") or article_brief.get("planning_query") or payload.get("query") or "").strip()
    article_direction = str(payload.get("article_direction") or article_brief.get("direction") or article_brief.get("display_subtitle") or "").strip()
    report_title = str(payload.get("report_title") or article_brief.get("display_title") or article_brief.get("main_title") or "").strip()
    report_subtitle = str(payload.get("report_subtitle") or article_brief.get("display_subtitle") or "").strip()
    plan_research_object = str(payload.get("research_object") or query or "").strip()
    original_query_anchor_terms = _unicode_query_anchor_terms(plan_query or query)
    topic_repaired_from_query = False
    if original_query_anchor_terms and not _topic_compatible_with_query_anchors(original_query_anchor_terms, plan_research_object):
        plan_research_object = original_query_anchor_terms[0]
        topic_repaired_from_query = True
    plan_global_required_terms = _string_list(payload.get("global_required_terms")) or _derive_global_required_terms(
        plan_query,
        plan_research_object,
    )
    if original_query_anchor_terms:
        plan_global_required_terms = _dedupe_limited_terms(
            [*original_query_anchor_terms, *plan_global_required_terms],
            limit=8,
            max_chars=32,
        )
    plan_topic_anchor_terms = _derive_topic_anchor_terms_for_plan(
        plan_query=plan_query,
        research_object=plan_research_object,
        global_required_terms=plan_global_required_terms,
        explicit_terms=_string_list(payload.get("topic_anchor_terms")),
    )
    if original_query_anchor_terms:
        compatible_plan_anchors = [
            term
            for term in plan_topic_anchor_terms
            if _topic_compatible_with_query_anchors(original_query_anchor_terms, term)
        ]
        plan_topic_anchor_terms = _dedupe_limited_terms(
            [*original_query_anchor_terms, *compatible_plan_anchors],
            limit=8,
            max_chars=32,
        )
    plan_generic_dimensions = _string_list(payload.get("generic_dimensions")) or _derive_plan_dimensions(
        plan_query,
        payload,
        _PLAN_GENERIC_DIMENSION_TERMS,
    )
    plan_evidence_dimensions = _string_list(payload.get("evidence_dimensions")) or _derive_plan_dimensions(
        plan_query,
        payload,
        _PLAN_EVIDENCE_DIMENSION_TERMS,
    )
    raw_research_domain = str(payload.get("research_domain") or _as_dict(payload.get("problem_framing")).get("research_domain") or "").strip()
    raw_report_intent = str(payload.get("report_intent") or _as_dict(payload.get("problem_framing")).get("report_intent") or "").strip()
    profession_plan = is_profession_education_employment_topic(
        plan_query,
        plan_research_object,
        plan_global_required_terms,
        raw_research_domain,
        raw_report_intent,
        payload.get("chapters"),
        payload.get("dimensions"),
        payload.get("evidence_goals"),
        payload.get("search_tasks"),
    )
    plan_research_domain = PROFESSION_EDUCATION_DOMAIN if profession_plan else raw_research_domain
    plan_report_intent = PROFESSION_EDUCATION_INTENT if profession_plan else raw_report_intent
    plan_global_forbidden_terms = _string_list(payload.get("global_forbidden_terms"))
    if profession_plan:
        plan_global_forbidden_terms = unique_text([*plan_global_forbidden_terms, *profession_forbidden_terms()], limit=80)
    raw_chapters = [item for item in _as_list(payload.get("chapters")) if isinstance(item, dict)]
    nested_tasks: List[Dict[str, Any]] = []
    nested_goals: List[Dict[str, Any]] = []
    for chapter in raw_chapters:
        chapter_payload = _as_dict(chapter)
        chapter_context = {
            "chapter_id": chapter_payload.get("chapter_id") or chapter_payload.get("id"),
            "chapter_title": chapter_payload.get("chapter_title") or chapter_payload.get("title"),
            "chapter_question": chapter_payload.get("core_question") or chapter_payload.get("chapter_question"),
        }
        for task in _as_list(chapter_payload.get("search_tasks")):
            if isinstance(task, dict):
                nested_tasks.append({**chapter_context, **task})
        for goal in _as_list(chapter_payload.get("evidence_goals")):
            if isinstance(goal, dict):
                nested_goals.append({**chapter_context, **goal})
    raw_task_items = _dedupe_plan_items(
        [*_as_list(payload.get("search_tasks")), *nested_tasks],
        id_key="task_id",
        fallback_keys=["query", "proof_role", "hypothesis_id", "chapter_id"],
    )
    tasks = [
        normalize_search_task(task, fallback_index=index)
        for index, task in enumerate(raw_task_items, start=1)
        if isinstance(task, dict)
    ]
    dimensions: List[Dict[str, Any]] = []
    seen = set()
    for raw_dim in _as_list(payload.get("dimensions")):
        if not isinstance(raw_dim, dict):
            continue
        name = str(raw_dim.get("dimension_name") or raw_dim.get("name") or raw_dim.get("dimension") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        dimensions.append(
            {
                "dimension_id": str(raw_dim.get("dimension_id") or raw_dim.get("id") or f"dim_{len(dimensions)+1}").strip(),
                "dimension_name": name,
                "purpose": str(raw_dim.get("purpose") or "").strip(),
                "must_have_terms": _string_list(raw_dim.get("must_have_terms")),
                "forbidden_terms": _string_list(raw_dim.get("forbidden_terms")),
                "hypothesis_id": str(raw_dim.get("hypothesis_id") or "").strip(),
            }
        )
    chapters = [
        normalize_chapter(item, fallback_index=index, query=plan_query)
        for index, item in enumerate(raw_chapters, start=1)
        if isinstance(item, dict)
    ]
    profession_rewritten_chapters = False
    legacy_profession_chapters: List[Dict[str, Any]] = []
    if profession_plan and _profession_chapters_need_rewrite(chapters):
        legacy_profession_chapters = [dict(item) for item in chapters]
        chapters = _profession_chapters_from_frames(plan_query)
        profession_rewritten_chapters = True
    if not chapters and dimensions:
        chapters = [
            normalize_chapter(
                {
                    "chapter_id": f"ch_{index:02d}",
                    "chapter_title": dimension.get("dimension_name"),
                    "core_question": dimension.get("purpose") or dimension.get("dimension_name"),
                    "dimension_id": dimension.get("dimension_id"),
                    "dimension_name": dimension.get("dimension_name"),
                    "source_template_keys": dimension.get("source_template_keys"),
                    "required_evidence_mix": dimension.get("required_evidence_mix"),
                },
                fallback_index=index,
                query=plan_query,
            )
            for index, dimension in enumerate(dimensions, start=1)
        ]
    if chapters and not dimensions:
        dimensions = [
            {
                "dimension_id": str(chapter.get("chapter_id") or f"ch_{index:02d}"),
                "dimension_name": str(chapter.get("chapter_title") or f"Chapter {index}"),
                "purpose": str(chapter.get("core_question") or chapter.get("chapter_question") or ""),
                "must_have_terms": [],
                "forbidden_terms": [],
                "hypothesis_id": "",
            }
            for index, chapter in enumerate(chapters, start=1)
        ]
    key_questions = _string_list(payload.get("key_questions"))
    hypotheses = [
        normalize_hypothesis(item, fallback_index=index, query=query)
        for index, item in enumerate(_as_list(payload.get("hypotheses")), start=1)
        if isinstance(item, dict)
    ]
    if not hypotheses:
        hypotheses = _default_hypotheses(query=str(payload.get("query") or query or ""), dimensions=dimensions, tasks=tasks, key_questions=key_questions)

    hypothesis_by_id = {str(item.get("hypothesis_id") or ""): item for item in hypotheses if str(item.get("hypothesis_id") or "")}
    hypothesis_by_dimension: Dict[str, Dict[str, Any]] = {}
    for item in hypotheses:
        dim_id = str(item.get("dimension_id") or "").strip()
        dim_name = str(item.get("dimension_name") or "").strip()
        if dim_id:
            hypothesis_by_dimension[dim_id] = item
        if dim_name:
            hypothesis_by_dimension[dim_name] = item

    chapter_by_id: Dict[str, Dict[str, Any]] = {}
    chapter_by_name: Dict[str, Dict[str, Any]] = {}
    for chapter in chapters:
        for key in [chapter.get("chapter_id"), chapter.get("dimension_id")]:
            text = str(key or "").strip()
            if text:
                chapter_by_id[text] = chapter
        for key in [chapter.get("chapter_title"), chapter.get("dimension_name"), chapter.get("core_question")]:
            text = str(key or "").strip()
            if text:
                chapter_by_name[text] = chapter

    def _inherit_chapter(payload_item: Dict[str, Any]) -> Dict[str, Any]:
        item = dict(payload_item)
        chapter = chapter_by_id.get(str(item.get("chapter_id") or "").strip())
        if not chapter:
            chapter = chapter_by_id.get(str(item.get("dimension_id") or "").strip())
        if not chapter:
            chapter = chapter_by_name.get(str(item.get("chapter_title") or "").strip())
        if not chapter:
            chapter = chapter_by_name.get(str(item.get("dimension_name") or item.get("dimension") or "").strip())
        if chapter:
            item["chapter_id"] = str(item.get("chapter_id") or chapter.get("chapter_id") or "").strip()
            item["chapter_title"] = str(item.get("chapter_title") or chapter.get("chapter_title") or "").strip()
            item["chapter_question"] = str(
                item.get("chapter_question") or chapter.get("chapter_question") or chapter.get("core_question") or ""
            ).strip()
            item["dimension_id"] = str(item.get("dimension_id") or chapter.get("chapter_id") or "").strip()
            item["dimension_name"] = str(item.get("dimension_name") or chapter.get("chapter_title") or "").strip()
            if "required_evidence_mix" in item:
                item["required_evidence_mix"] = _string_list(item.get("required_evidence_mix")) or _string_list(chapter.get("required_evidence_mix"))
        return item

    raw_goal_items = _dedupe_plan_items(
        [*_as_list(payload.get("evidence_goals")), *nested_goals],
        id_key="goal_id",
        fallback_keys=["question", "proof_role", "hypothesis_id", "chapter_id"],
    )
    normalized_goals = [
        normalize_evidence_goal(goal, fallback_index=index)
        for index, goal in enumerate(raw_goal_items, start=1)
        if isinstance(goal, dict)
    ]
    if not normalized_goals:
        normalized_goals = [
            normalize_evidence_goal(
                {
                    "goal_id": task.get("evidence_goal") or task.get("task_id"),
                    "dimension_id": task.get("dimension_id"),
                    "dimension_name": task.get("dimension_name"),
                    "question": task.get("evidence_goal"),
                    "expected_metrics": task.get("must_have_terms"),
                    "must_have_terms": task.get("must_have_terms"),
                    "forbidden_terms": task.get("forbidden_terms"),
                    "source_priority": task.get("source_priority"),
                    "freshness": "recent",
                    "min_sources": 2,
                    "evidence_type": task.get("evidence_type") or task.get("intent"),
                    "hypothesis_id": task.get("hypothesis_id"),
                    "proof_role": task.get("proof_role"),
                    "lane_targets": task.get("lane_targets") or task.get("lanes"),
                },
                fallback_index=index,
            )
            for index, task in enumerate(tasks, start=1)
            if task.get("evidence_goal")
        ]

    def _inherit_hypothesis(payload_item: Dict[str, Any]) -> Dict[str, Any]:
        item = dict(payload_item)
        hypothesis = hypothesis_by_id.get(str(item.get("hypothesis_id") or ""))
        if not hypothesis:
            hypothesis = hypothesis_by_dimension.get(str(item.get("dimension_id") or "")) or hypothesis_by_dimension.get(str(item.get("dimension_name") or ""))
        if not hypothesis and hypotheses:
            hypothesis = hypotheses[0]
        if hypothesis:
            item["hypothesis_id"] = str(item.get("hypothesis_id") or hypothesis.get("hypothesis_id") or "").strip()
            item["hypothesis_statement"] = str(item.get("hypothesis_statement") or hypothesis.get("statement") or "").strip()
            item["proof_standard"] = str(item.get("proof_standard") or hypothesis.get("proof_standard") or "medium").strip().lower()
            item["decision_use"] = str(item.get("decision_use") or hypothesis.get("decision_use") or "research").strip()
            if "required_source_levels" in item:
                item["required_source_levels"] = _string_list(item.get("required_source_levels")) or _string_list(hypothesis.get("required_source_levels")) or ["A", "B"]
            if "counter_evidence_required" in item:
                item["counter_evidence_required"] = bool(item.get("counter_evidence_required", hypothesis.get("counter_evidence_required", False)))
            if "metric_definitions" in item:
                item["metric_definitions"] = _dict_list(item.get("metric_definitions")) or _dict_list(hypothesis.get("metric_definitions"))
        return item

    def _attach_plan_topic(payload_item: Dict[str, Any]) -> Dict[str, Any]:
        item = dict(payload_item)
        topic_anchor = plan_topic_anchor_terms[0] if plan_topic_anchor_terms else plan_research_object
        if plan_research_object and not item.get("research_object"):
            item["research_object"] = plan_research_object
        if plan_topic_anchor_terms and not item.get("topic_anchor_terms"):
            item["topic_anchor_terms"] = plan_topic_anchor_terms
        if topic_anchor and item.get("query") and topic_anchor not in str(item.get("query") or ""):
            item["query"] = f"{topic_anchor} {str(item.get('query') or '').strip()}".strip()
        if plan_global_required_terms and not item.get("global_required_terms"):
            item["global_required_terms"] = plan_global_required_terms
        if plan_generic_dimensions and not item.get("generic_dimensions"):
            item["generic_dimensions"] = plan_generic_dimensions
        if plan_evidence_dimensions and not item.get("evidence_dimensions"):
            item["evidence_dimensions"] = plan_evidence_dimensions
        if plan_topic_anchor_terms:
            contract = _as_dict(item.get("query_contract"))
            contract.setdefault("topic_anchor_terms", plan_topic_anchor_terms)
            contract.setdefault("topic_anchor_required", True)
            item["query_contract"] = contract
        if plan_global_required_terms:
            existing_terms = _string_list(item.get("must_have_terms"))
            merged_terms: List[str] = []
            for term in [*plan_global_required_terms, *existing_terms]:
                if term and term not in merged_terms:
                    merged_terms.append(term)
            if merged_terms:
                item["must_have_terms"] = merged_terms[:8]
        query_text = str(item.get("query") or "").strip()
        role = str(item.get("proof_role") or item.get("intent") or "").strip().lower()
        role_hints = {
            "support": ["行业研究", "案例", "趋势"],
            "data": ["行业研究", "案例", "趋势"],
            "metric": ["指标", "数据", "统计"],
            "case": ["案例", "项目", "客户"],
            "counter": ["失败", "风险", "反向案例"],
            "source_check": ["原文", "来源", "口径"],
            "technology_product": ["技术路线", "产品方案", "标准"],
            "expert": ["研报", "白皮书", "专家"],
        }.get(role, ["行业研究", "案例", "趋势"])
        if query_text:
            repeated_anchor = bool(topic_anchor and query_text.count(str(topic_anchor)) >= 2)
            leaky_lane = bool(
                re.search(
                    r"official data|market research|market size price|official filing|customer certification|"
                    r"product docs|brokerage association|industry research expert|"
                    r"反证\s*风险\s*失败案例|客户不买账|替代方案|负面",
                    query_text,
                    flags=re.I,
                )
            )
            if repeated_anchor or leaky_lane or len(query_text) > 72 or (re.search(r"[:：]", query_text) and len(query_text) > 56):
                compact_terms = _dedupe_limited_terms(
                    [
                        topic_anchor,
                        *role_hints[:2],
                        *[
                            term
                            for term in _string_list(item.get("must_have_terms"))
                            if term and term != topic_anchor and len(term) <= 18 and not re.search(r"[:：]", term)
                        ],
                    ],
                    limit=7,
                    max_chars=18,
                )
                if compact_terms:
                    item["query"] = " ".join(compact_terms)
                    item["query_compacted_after_topic_anchor"] = True
            elif role_hints and not any(hint in query_text for hint in role_hints):
                item["query"] = " ".join([query_text, *role_hints[:2]]).strip()
                if len(item["query"]) > 96:
                    item["query"] = item["query"][:95].rstrip() + "…"
                item["query_role_hints_added"] = True
        return item

    tasks = [_attach_plan_topic(_inherit_hypothesis(_inherit_chapter(task))) for task in tasks]
    tasks = _ensure_hypothesis_task_contract(tasks, hypotheses)
    tasks = [_attach_plan_topic(_inherit_chapter(task)) for task in tasks]
    existing_goal_keys = {
        (str(goal.get("hypothesis_id") or ""), str(goal.get("question") or goal.get("goal_id") or ""))
        for goal in normalized_goals
    }
    for task in tasks:
        key = (str(task.get("hypothesis_id") or ""), str(task.get("evidence_goal") or ""))
        if not key[1] or key in existing_goal_keys:
            continue
        normalized_goals.append(
            normalize_evidence_goal(
                {
                    "goal_id": task.get("task_id"),
                    "dimension_id": task.get("dimension_id"),
                    "dimension_name": task.get("dimension_name"),
                    "question": task.get("evidence_goal"),
                    "expected_metrics": task.get("must_have_terms"),
                    "must_have_terms": task.get("must_have_terms"),
                    "forbidden_terms": task.get("forbidden_terms"),
                    "source_priority": task.get("source_priority"),
                    "freshness": "recent",
                    "min_sources": 2,
                    "evidence_type": task.get("evidence_type") or task.get("intent"),
                    "hypothesis_id": task.get("hypothesis_id"),
                    "hypothesis_statement": task.get("hypothesis_statement"),
                    "proof_standard": task.get("proof_standard"),
                    "decision_use": task.get("decision_use"),
                    "counter_evidence_required": bool(task.get("counter_evidence_required", False)),
                    "proof_role": task.get("proof_role"),
                    "lane_targets": task.get("lane_targets") or task.get("lanes"),
                },
                fallback_index=len(normalized_goals) + 1,
            )
        )
        existing_goal_keys.add(key)
    normalized_goals = [_inherit_hypothesis(_inherit_chapter(goal)) for goal in normalized_goals]
    normalized_plan = {
        "query": plan_query,
        "planning_query": plan_query,
        "raw_query": str(payload.get("raw_query") or article_brief.get("raw_query") or "").strip(),
        "article_brief": article_brief,
        "article_direction": article_direction,
        "report_title": report_title,
        "report_subtitle": report_subtitle,
        "research_type": str(payload.get("research_type") or "generic_topic").strip(),
        "research_domain": plan_research_domain,
        "report_intent": plan_report_intent,
        "decision_context": str(payload.get("decision_context") or "").strip(),
        "report_family": str(payload.get("report_family") or "dynamic_research_report").strip(),
        "research_object": plan_research_object,
        "topic_anchor_terms": plan_topic_anchor_terms,
        "generic_dimensions": plan_generic_dimensions,
        "evidence_dimensions": plan_evidence_dimensions,
        "core_question": str(payload.get("core_question") or payload.get("question") or query or "").strip(),
        "key_questions": key_questions,
        "hypotheses": hypotheses,
        "chapters": chapters,
        "dimensions": dimensions,
        "evidence_goals": normalized_goals,
        "search_tasks": [task for task in tasks if task.get("query")],
        "source_strategy": [item for item in _as_list(payload.get("source_strategy")) if isinstance(item, dict)],
        "problem_framing": _as_dict(payload.get("problem_framing")),
        "proof_standards": _as_dict(payload.get("proof_standards")),
        "source_requirements": _as_dict(payload.get("source_requirements")),
        "evidence_coverage_requirements": _as_dict(payload.get("evidence_coverage_requirements")),
        "report_depth_target": str(payload.get("report_depth_target") or "standard").strip(),
        "output_format": str(payload.get("output_format") or "brief").strip(),
        "global_forbidden_terms": plan_global_forbidden_terms,
        "global_required_terms": plan_global_required_terms,
        "quality_rules": _as_dict(payload.get("quality_rules")),
        "topic_repaired_from_query": topic_repaired_from_query,
        "profession_chapters_rewritten": profession_rewritten_chapters,
        "original_query_anchor_terms": original_query_anchor_terms,
        "legacy_planner_chapters": [
            *legacy_profession_chapters,
            *[dict(item) for item in _as_list(payload.get("legacy_planner_chapters")) if isinstance(item, dict)],
        ],
        "legacy_planner_dimensions": [dict(item) for item in _as_list(payload.get("legacy_planner_dimensions")) if isinstance(item, dict)],
        "legacy_planner_search_tasks": [dict(item) for item in _as_list(payload.get("legacy_planner_search_tasks")) if isinstance(item, dict)],
        "dropped_template_sections": [dict(item) for item in _as_list(payload.get("dropped_template_sections")) if isinstance(item, dict)],
    }
    return enforce_research_plan_chapter_limits(normalized_plan, query=plan_query)
