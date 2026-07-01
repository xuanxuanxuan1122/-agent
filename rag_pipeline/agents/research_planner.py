from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, Optional

from ..config.search_config import (
    build_llm_config_for_task,
)
from ..search.memory import call_openai_compatible_json, llm_config_is_ready
from .article_brief import extract_research_subject, is_broad_ai_subject, normalize_article_brief, planning_query_from_brief
from .dynamic_search_schema import enforce_research_plan_chapter_limits, normalize_research_plan
from .problem_framing_agent import apply_problem_framing, run_problem_framing_agent
from .report_profile_registry import select_report_profile


logger = logging.getLogger(__name__)


RESEARCH_PLANNER_SYSTEM = """
你是企业研究报告的动态研究规划 Agent。
你不写报告，只负责把用户问题拆成研究维度、证据目标和搜索任务。

额外规划层要求：
- 同时返回 core_question、hypotheses、proof_standards、source_requirements 和 report_depth_target。
- 每个 hypothesis 必须包含 hypothesis_id、statement、decision_use、proof_standard、counter_evidence_required、required_source_levels、required_evidence_types、metric_definitions、falsification_triggers。
- 每个 evidence_goal 和 search_task 都必须携带 hypothesis_id、proof_standard、decision_use，以及它服务的来源/指标契约。
- 每个 search_task 必须包含 proof_role，可选值为 support、counter、metric、case、source_check。
- 每个 search_task 应包含 lane_targets，可选 official_data、filing_company、market_research、news_event、technology_product、customer_case。
- 每个重要 hypothesis 至少需要一个支持性任务和一个反向证据任务。
- 行业、生态、市场、发展趋势类请求默认 report_family 使用 industry_deep_report；除非用户明确要求政策影响分析。
- industry_deep_report 默认使用 4-6 个问题驱动核心章节（目标 5 个，硬上限 6 个）；通用行业模块只作为覆盖检查清单，不能作为固定章数。
- 1-2 个承载中心结论的章节应标记 key_chapter=true；关键章节需要比普通章节更强的可追溯证据。
- 行业证据应覆盖官方/协会/白皮书、市场研究、公司公告/财报、客户案例/订单、反向/风险证据。
- chapters 是一等对象，必须先输出问题驱动章节，再输出搜索任务。
- 通用报告模块只是候选池/覆盖清单，绝不能复制为固定章节标题。
- 每个 chapter 必须包含 chapter_id、chapter_title、core_question、reason_to_include、source_template_keys、required_evidence_mix、min_total_sources、min_ab_sources、min_counter_sources、evidence_goals、search_tasks。
- 每个 search_task 必须携带 chapter_id、chapter_title、chapter_question、evidence_goal_id、proof_role、lane_targets、min_source_level。
- 如果 article_brief.direction_missing 为 true，把 main title / planning_query 视为研究请求，动态推导具体研究方向、假设、章节和搜索任务；不要要求副标题，也不要退回固定五段式模板。

输出 JSON：
{
  "query": "...",
  "research_type": "industry_scan / market_entry / company_due_diligence / investment_memo / product_research / competitor_analysis / policy_impact / consumer_market / macro_topic / supply_chain_research / technology_trend / business_model_research",
  "report_family": "industry_deep_report / market_entry_report / company_due_diligence_report / investment_memo / product_research_report / competitor_analysis_report / policy_impact_report / consumer_market_report / macro_topic_report / supply_chain_report / briefing_note",
  "research_object": "...",
  "chapters": [
    {
      "chapter_id": "ch_01",
      "chapter_title": "一个由用户问题决定的判断型或问题型标题",
      "core_question": "本章要回答的核心问题",
      "reason_to_include": "为什么当前用户问题需要这一章",
      "source_template_keys": ["market_size", "demand_driver", "risk"],
      "required_evidence_mix": ["official_data", "market_research", "company_filing", "case", "counter_evidence"],
      "min_total_sources": 6,
      "min_ab_sources": 2,
      "min_counter_sources": 1,
      "evidence_goals": [],
      "search_tasks": []
    }
  ],
  "dimensions": [
    {
      "dimension_id": "...",
      "dimension_name": "...",
      "purpose": "...",
      "must_have_terms": [],
      "forbidden_terms": []
    }
  ],
  "evidence_goals": [
    {
      "goal_id": "...",
      "dimension_id": "...",
      "dimension_name": "...",
      "question": "...",
      "expected_metrics": [],
      "must_have_terms": [],
      "forbidden_terms": [],
      "source_priority": [],
      "freshness": "latest/recent/stable/historical",
      "min_sources": 2,
      "evidence_type": "data/policy/case/company/academic/news/filing"
    }
  ],
  "search_tasks": [
    {
      "task_id": "...",
      "agent": "iqs/rag/both",
      "dimension_id": "...",
      "dimension_name": "...",
      "query": "...",
      "evidence_goal": "...",
      "intent": "data/policy/company/case/academic/news/filing/statistics",
      "must_have_terms": [],
      "forbidden_terms": [],
      "source_priority": []
    }
  ],
  "global_forbidden_terms": [],
  "global_required_terms": [],
  "quality_rules": {}
}

硬约束：
1. 不要默认使用市场、竞争、政策、技术、资本五个维度。
2. 维度必须由用户问题决定。
3. 每个 search_task 只能服务一个明确证据目标。
4. 如果是人口、城市、宏观、政策、公司尽调、消费用户研究，不要生成融资、IPO、估值、市占率等无关搜索词。
5. 每个任务必须写 must_have_terms 和 forbidden_terms。
只返回 JSON。
""".strip()


RESEARCH_PLANNER_COMPACT_SYSTEM = """
你是研究规划 Agent，只返回紧凑且合法的 JSON。
不要写报告，不要输出 search_tasks，不要写长解释。
必须在 research_object 和 global_required_terms 中保留用户原始研究主题词；不得把中文主题替换成其他行业、猜测翻译或示例主题。
chapters 保持 4-6 个，单章 evidence_goals 最多 4 个。
每个 chapter 必须包含 chapter_id、chapter_title、core_question。
每个 evidence_goal 必须包含 goal_id、chapter_id、proof_role、required_fields、lane_targets。
proof_role 可选值：support、metric、case、counter、filing、source_check、technology_product。
只返回以下 schema：
{
  "query": "...",
  "research_type": "...",
  "report_family": "industry_deep_report|briefing_note|policy_impact_report|dynamic_research_report",
  "research_object": "...",
  "core_question": "...",
  "hypotheses": [{"hypothesis_id": "H1", "statement": "..."}],
  "chapters": [{"chapter_id": "ch_01", "chapter_title": "...", "core_question": "..."}],
  "evidence_goals": [{"goal_id": "...", "chapter_id": "ch_01", "proof_role": "metric", "required_fields": [], "lane_targets": []}],
  "global_required_terms": [],
  "global_forbidden_terms": [],
  "quality_rules": {}
}
""".strip()


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, min_value: int = 0, max_value: int = 1_000_000) -> int:
    try:
        value = int(str(os.getenv(name, str(default))).strip())
    except (TypeError, ValueError):
        value = default
    return min(max_value, max(min_value, value))


def _llm_config() -> Dict[str, Any]:
    return _planner_llm_config()


def _planner_llm_config(llm_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    config = _as_dict(llm_config) or dict(build_llm_config_for_task("planning"))
    min_output_tokens = _env_int(
        "REPORT_PLANNING_MAX_OUTPUT_TOKENS",
        8192,
        min_value=2048,
        max_value=64000,
    )
    try:
        configured_output_tokens = int(config.get("max_output_tokens") or 0)
    except (TypeError, ValueError):
        configured_output_tokens = 0
    if configured_output_tokens < min_output_tokens:
        config["max_output_tokens"] = min_output_tokens
    return config


def _planner_failure_reason(error: Any, diagnostic: Dict[str, Any]) -> str:
    finish_reason = str(diagnostic.get("finish_reason") or "").strip().lower()
    error_text = str(error or diagnostic.get("error") or "").strip().lower()
    if finish_reason == "length":
        return "json_truncated"
    if "not valid json" in error_text or "json" in error_text:
        return "invalid_json"
    if error_text:
        return "llm_error"
    return "unknown"


def _planner_payload(
    *,
    query: str,
    article_brief: Optional[Dict[str, Any]],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = {
        "query": query,
        "article_brief": _as_dict(article_brief),
        "current_year": datetime.now().year,
    }
    payload.update(_as_dict(extra))
    return payload


def _research_subject(query: str) -> str:
    text = re.sub(r"\s+", " ", str(query or "")).strip()
    text = re.sub(r"^(请问|帮我看看|帮我分析|分析一下|现在|当前|目前)\s*", "", text)
    text = re.sub(r"(企业行研|行业研究|行研|深度研究|研究)?(报告|文档)$", "", text).strip()
    text = re.sub(r"(怎么样|如何|有哪些|怎么看)[？?]?$", "", text).strip()
    text = re.sub(r"(焦虑与机遇|机遇与挑战)$", "", text).strip()
    specific = extract_research_subject(text)
    if specific and not is_broad_ai_subject(specific):
        return specific
    if re.search(r"(中国|国内).*(AI|人工智能)|(?:AI|人工智能).*(中国|国内)", text, re.I):
        return "中国人工智能行业"
    if re.search(r"\bAI\b|人工智能", text, re.I) and re.search(r"行业|产业|市场|赛道", text):
        return "人工智能行业"
    return text or str(query or "").strip()


def _topic_required_terms(query: str, research_object: str) -> list[str]:
    text = f"{query} {research_object}"
    terms: list[str] = []
    if re.search(r"中国|国内", text, re.I):
        terms.append("中国")
    if re.search(r"\bAI\b|人工智能|大模型|生成式", text, re.I):
        terms.extend(["人工智能", "AI"])
    if re.search(r"新能源汽车|新能源车|动力电池|锂电", text):
        terms.extend(["新能源汽车", "动力电池"])
    if re.search(r"半导体|芯片|集成电路", text, re.I):
        terms.extend(["半导体", "芯片"])
    if research_object and research_object not in terms:
        terms.append(research_object)
    deduped: list[str] = []
    for term in terms:
        if term and term not in deduped:
            deduped.append(term)
    return deduped[:6]


def _dynamic_seed_plan(
    query: str,
    problem_framing: Optional[Dict[str, Any]] = None,
    article_brief: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    brief = normalize_article_brief(article_brief, fallback_query=query) if article_brief else {}
    query = planning_query_from_brief(brief, fallback_query=query) if brief else str(query or "").strip()
    research_object = _research_subject(query)
    seed_plan = {
        "query": query,
        "planning_query": query,
        "article_brief": brief,
        "article_direction": str(brief.get("direction") or ""),
        "report_title": str(brief.get("display_title") or ""),
        "report_subtitle": str(brief.get("display_subtitle") or ""),
        "core_question": query,
        "research_type": "dynamic_research",
        "decision_context": "question_driven_research",
        "report_family": "dynamic_research_report",
        "research_object": research_object,
        "key_questions": [query] if query else [],
        "hypotheses": [],
        "chapters": [],
        "dimensions": [],
        "evidence_goals": [],
        "search_tasks": [],
        "source_strategy": [],
        "proof_standards": {
            "strong": {"required_ab_sources": 2, "counter_evidence_required": True, "metric_scope_period_unit_required": True},
            "medium": {"required_ab_sources": 1, "counter_evidence_required": False, "metric_scope_period_unit_required": True},
            "weak": {"required_ab_sources": 0, "counter_evidence_required": False, "appendix_or_followup_only": True},
        },
        "source_requirements": {
            "core_claim": ["A", "B"],
            "supporting_claim": ["A", "B"],
            "clue_only": ["C"],
            "appendix_only": ["D"],
        },
        "evidence_coverage_requirements": {
            "per_hypothesis": {
                "min_A_or_B_sources": 2,
                "min_counter_sources": 1,
                "min_metric_sources": 2,
                "min_case_sources": 1,
                "source_diversity": [],
            }
        },
        "report_depth_target": "deep",
        "output_format": "brief",
        "global_forbidden_terms": [],
        "global_required_terms": _topic_required_terms(query, research_object),
        "quality_rules": {
            "min_relevance_score": 0.45,
            "reject_cross_domain_pollution": True,
            "chapters_come_from_hypotheses": True,
            "disable_fixed_fallback_templates": True,
            "infer_directions_from_title_when_direction_missing": bool(brief.get("direction_missing")),
        },
    }
    framing = _as_dict(problem_framing)
    if not framing:
        try:
            framing = run_problem_framing_agent(query=query, article_brief=brief)
        except Exception:
            logger.exception("Problem framing fallback failed", extra={"query": query})
            framing = {}
    if framing:
        framed_plan = apply_problem_framing(seed_plan, framing)
        if framed_plan.get("chapters") or framed_plan.get("evidence_goals") or framed_plan.get("search_tasks"):
            return framed_plan
    return seed_plan


def _fallback_research_plan(query: str) -> Dict[str, Any]:
    """Backward-compatible wrapper; fixed fallback templates are no longer generated."""
    return _dynamic_seed_plan(query)


def _llm_research_plan(
    query: str,
    llm_config: Optional[Dict[str, Any]] = None,
    article_brief: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not _env_flag("BRAIN_ENABLE_LLM_RESEARCH_PLANNER", False):
        return {
            "_planner_llm_degraded": True,
            "_planner_degraded_reason": "llm_disabled",
            "_planning_status": "fallback_seed",
        }
    config = _planner_llm_config(llm_config)
    if not llm_config_is_ready(config):
        return {
            "_planner_llm_degraded": True,
            "_planner_degraded_reason": "llm_config_unavailable",
            "_planning_status": "fallback_seed",
        }

    def call_compact_planner(
        *,
        status: str,
        mode: str,
        extra_payload: Optional[Dict[str, Any]] = None,
        previous_diagnostic: Optional[Dict[str, Any]] = None,
        previous_reason: str = "",
    ) -> Dict[str, Any]:
        response = call_openai_compatible_json(
            config=config,
            system_prompt=RESEARCH_PLANNER_COMPACT_SYSTEM,
            user_payload=_planner_payload(query=query, article_brief=article_brief, extra=extra_payload),
        )
        payload = _as_dict(response.get("payload"))
        payload["_planner_llm_degraded"] = False
        payload["_planning_status"] = status
        payload["_planner_mode"] = mode
        if previous_reason:
            payload["_planner_degraded_reason"] = f"compact_retry_after_{previous_reason}"
        diagnostic = _as_dict(previous_diagnostic)
        if diagnostic:
            payload["_planner_finish_reason"] = str(diagnostic.get("finish_reason") or "").strip()
        llm_call = _as_dict(response.get("llm_call"))
        if diagnostic:
            llm_call = {
                **llm_call,
                "previous_planner_error": {
                    "reason": previous_reason,
                    "finish_reason": str(diagnostic.get("finish_reason") or "").strip(),
                    "error": str(diagnostic.get("error") or "")[:500],
                },
            }
        payload["_planner_llm_call"] = llm_call
        return payload

    if _env_flag("REPORT_PLANNING_COMPACT_FIRST", True):
        try:
            return call_compact_planner(
                status="ok",
                mode="compact_first",
                extra_payload={"compact_first": True},
            )
        except Exception as exc:
            diagnostic = _as_dict(getattr(exc, "diagnostic", {}))
            reason = _planner_failure_reason(str(exc), diagnostic)
            if reason in {"json_truncated", "invalid_json"} and _env_flag("REPORT_PLANNING_COMPACT_RETRY", True):
                try:
                    return call_compact_planner(
                        status="repaired",
                        mode="compact_retry",
                        extra_payload={
                            "compact_retry": True,
                            "previous_error": {
                                "reason": reason,
                                "finish_reason": str(diagnostic.get("finish_reason") or "").strip(),
                            },
                        },
                        previous_diagnostic=diagnostic,
                        previous_reason=reason,
                    )
                except Exception:
                    logger.exception("Compact-first LLM research planner retry failed", extra={"query": query})
            else:
                logger.exception("Compact-first LLM research planner failed", extra={"query": query})
            return {
                "_planner_llm_degraded": True,
                "_planner_llm_error": str(exc),
                "_planner_degraded_reason": reason,
                "_planner_finish_reason": str(diagnostic.get("finish_reason") or "").strip(),
                "_planning_status": "fallback_seed",
                "_planner_mode": "compact_first",
                "_planner_llm_call": diagnostic,
            }

    try:
        response = call_openai_compatible_json(
            config=config,
            system_prompt=RESEARCH_PLANNER_SYSTEM,
            user_payload=_planner_payload(query=query, article_brief=article_brief),
        )
    except Exception as exc:
        diagnostic = _as_dict(getattr(exc, "diagnostic", {}))
        reason = _planner_failure_reason(str(exc), diagnostic)
        if reason in {"json_truncated", "invalid_json"} and _env_flag("REPORT_PLANNING_COMPACT_RETRY", True):
            try:
                return call_compact_planner(
                    status="repaired",
                    mode="compact_retry",
                    extra_payload={
                        "compact_retry": True,
                        "previous_error": {
                            "reason": reason,
                            "finish_reason": str(diagnostic.get("finish_reason") or "").strip(),
                        },
                    },
                    previous_diagnostic=diagnostic,
                    previous_reason=reason,
                )
            except Exception:
                logger.exception("Compact LLM research planner retry failed", extra={"query": query})
        else:
            logger.exception("LLM research planner failed", extra={"query": query})
        return {
            "_planner_llm_degraded": True,
            "_planner_llm_error": str(exc),
            "_planner_degraded_reason": reason,
            "_planner_finish_reason": str(diagnostic.get("finish_reason") or "").strip(),
            "_planning_status": "fallback_seed",
            "_planner_llm_call": diagnostic,
        }
    payload = _as_dict(response.get("payload"))
    payload["_planner_llm_degraded"] = False
    payload["_planning_status"] = "ok"
    payload["_planner_mode"] = "full"
    payload["_planner_llm_call"] = _as_dict(response.get("llm_call"))
    return payload


def run_research_planner_agent(
    *,
    query: str,
    llm_config: Optional[Dict[str, Any]] = None,
    article_brief: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    brief = normalize_article_brief(article_brief, fallback_query=query) if article_brief else {}
    query = planning_query_from_brief(brief, fallback_query=query) if brief else str(query or "").strip()
    try:
        problem_framing = run_problem_framing_agent(query=query, llm_config=llm_config, article_brief=brief)
    except Exception:
        logger.exception("Problem framing failed in research planner", extra={"query": query})
        problem_framing = {}
    raw_llm_plan = _llm_research_plan(query, llm_config, brief)
    llm_plan = normalize_research_plan(raw_llm_plan, query=query)
    used_llm_plan = bool(
        (llm_plan.get("chapters") or llm_plan.get("dimensions"))
        and (llm_plan.get("search_tasks") or llm_plan.get("evidence_goals") or llm_plan.get("hypotheses"))
    )
    if used_llm_plan:
        plan = normalize_research_plan(apply_problem_framing(llm_plan, problem_framing), query=query)
    else:
        plan = normalize_research_plan(_dynamic_seed_plan(query, problem_framing, brief), query=query)
    if brief:
        plan = {
            **plan,
            "article_brief": brief,
            "article_direction": str(brief.get("direction") or ""),
            "report_title": str(brief.get("display_title") or ""),
            "report_subtitle": str(brief.get("display_subtitle") or ""),
            "planning_query": query,
            "query": query,
        }
    profile = select_report_profile(query, plan)
    llm_call = _as_dict(raw_llm_plan.get("_planner_llm_call"))
    llm_degraded = bool(raw_llm_plan.get("_planner_llm_degraded"))
    planner_finish_reason = str(raw_llm_plan.get("_planner_finish_reason") or llm_call.get("finish_reason") or "").strip()
    planner_degraded_reason = str(raw_llm_plan.get("_planner_degraded_reason") or "").strip()
    planner_mode = str(raw_llm_plan.get("_planner_mode") or "").strip()
    planning_status = str(raw_llm_plan.get("_planning_status") or "").strip()
    raw_planning_status = str(raw_llm_plan.get("_planning_status") or "").strip()
    if not used_llm_plan:
        planning_status = "fallback_seed"
        if not planner_degraded_reason:
            planner_degraded_reason = "llm_incomplete" if raw_llm_plan else "llm_disabled"
        llm_degraded = True
    elif raw_planning_status == "repaired":
        planning_status = "repaired"
        llm_degraded = False
    elif not planning_status:
        planning_status = "ok"
    plan = {
        **plan,
        "report_family": profile.get("name") or plan.get("report_family"),
        "report_profile": profile.get("name"),
        "planner_llm_degraded": llm_degraded,
        "planning_status": planning_status,
        "planner_degraded_reason": planner_degraded_reason,
        "planner_finish_reason": planner_finish_reason,
        "planner_mode": planner_mode,
        "allow_search": bool(plan.get("chapters") or plan.get("dimensions") or plan.get("evidence_goals")),
        "allow_full_report": planning_status in {"ok", "repaired"},
        "planner_llm_call": llm_call,
        "llm_calls": [llm_call] if llm_call else [],
        "layout_intent": {
            "profile": profile.get("name"),
            "narrative_spines": profile.get("narrative_spines") or [],
            "candidate_modules": profile.get("candidate_modules") or [],
            "front_blocks": profile.get("front_blocks") or [],
            "back_blocks": profile.get("back_blocks") or [],
        },
    }
    return enforce_research_plan_chapter_limits(plan, query=query)


def research_planner_tool(payload: Dict[str, Any]) -> Dict[str, Any]:
    return run_research_planner_agent(
        query=str(payload.get("query") or ""),
        llm_config=_as_dict(payload.get("llm_config")),
        article_brief=_as_dict(payload.get("article_brief")),
    )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Generate a dynamic research plan.")
    parser.add_argument("query")
    args = parser.parse_args()
    print(json.dumps(run_research_planner_agent(query=args.query), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
