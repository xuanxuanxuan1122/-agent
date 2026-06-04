from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


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
        "forbidden_terms": _compact_search_terms(task.get("forbidden_terms"), limit=5),
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
    plan_query = str(payload.get("planning_query") or article_brief.get("planning_query") or payload.get("query") or query or "").strip()
    article_direction = str(payload.get("article_direction") or article_brief.get("direction") or article_brief.get("display_subtitle") or "").strip()
    report_title = str(payload.get("report_title") or article_brief.get("display_title") or article_brief.get("main_title") or "").strip()
    report_subtitle = str(payload.get("report_subtitle") or article_brief.get("display_subtitle") or "").strip()
    plan_research_object = str(payload.get("research_object") or query or "").strip()
    plan_global_required_terms = _string_list(payload.get("global_required_terms")) or _derive_global_required_terms(
        plan_query,
        plan_research_object,
    )
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
        if plan_research_object and not item.get("research_object"):
            item["research_object"] = plan_research_object
        if plan_research_object and item.get("query") and plan_research_object not in str(item.get("query") or ""):
            item["query"] = f"{plan_research_object} {str(item.get('query') or '').strip()}".strip()
        if plan_global_required_terms and not item.get("global_required_terms"):
            item["global_required_terms"] = plan_global_required_terms
        if plan_global_required_terms:
            existing_terms = _string_list(item.get("must_have_terms"))
            merged_terms: List[str] = []
            for term in [*plan_global_required_terms, *existing_terms]:
                if term and term not in merged_terms:
                    merged_terms.append(term)
            if merged_terms:
                item["must_have_terms"] = merged_terms[:8]
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
        "decision_context": str(payload.get("decision_context") or "").strip(),
        "report_family": str(payload.get("report_family") or "briefing_note").strip(),
        "research_object": plan_research_object,
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
        "global_forbidden_terms": _string_list(payload.get("global_forbidden_terms")),
        "global_required_terms": plan_global_required_terms,
        "quality_rules": _as_dict(payload.get("quality_rules")),
        "legacy_planner_chapters": [dict(item) for item in _as_list(payload.get("legacy_planner_chapters")) if isinstance(item, dict)],
        "legacy_planner_dimensions": [dict(item) for item in _as_list(payload.get("legacy_planner_dimensions")) if isinstance(item, dict)],
        "legacy_planner_search_tasks": [dict(item) for item in _as_list(payload.get("legacy_planner_search_tasks")) if isinstance(item, dict)],
        "dropped_template_sections": [dict(item) for item in _as_list(payload.get("dropped_template_sections")) if isinstance(item, dict)],
    }
    return enforce_research_plan_chapter_limits(normalized_plan, query=plan_query)
