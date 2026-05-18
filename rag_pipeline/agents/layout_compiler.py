from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

try:
    from .block_schema import select_blocks_for_chapter
    from .report_profile_registry import default_report_shell, select_report_profile
    from .universal_report_ontology import UNIVERSAL_REPORT_MODULES, evidence_mix_for_modules, module_by_key
except Exception:  # pragma: no cover - direct script mode fallback
    from block_schema import select_blocks_for_chapter  # type: ignore
    from report_profile_registry import default_report_shell, select_report_profile  # type: ignore
    from universal_report_ontology import UNIVERSAL_REPORT_MODULES, evidence_mix_for_modules, module_by_key  # type: ignore


AGENT_NAME = "layout_compiler"

MODULE_TITLE_HINTS = {
    "industry_definition": "研究对象的边界和分类应如何界定？",
    "market_size": "规模、增速和价格信号是否支持机会判断？",
    "demand_driver": "需求变化来自哪里，能否持续兑现？",
    "industry_chain": "产业链瓶颈、利润池和供应链位置如何变化？",
    "technology": "技术路线成熟度和替代路径会怎样影响落地？",
    "customer": "谁在购买，场景痛点和采购约束是什么？",
    "business_model": "商业模式、单位经济和现金流是否成立？",
    "competition": "主要玩家、壁垒和竞争强度如何改变结论？",
    "policy": "政策、监管或外部规则如何传导到行业结果？",
    "capital": "资本开支、融资和交易信号是否验证景气？",
    "risk": "哪些反证触发器会推翻当前判断？",
    "entry_strategy": "下一步进入、投资、采购或验证应如何排序？",
    "timeline": "关键事件按时间如何改变判断？",
    "stakeholder_map": "不同主体的利益、约束和行动会如何分化？",
    "transmission_chain": "冲击变量通过哪些链条传导到结果？",
    "beneficiary_loser": "哪些环节受益，哪些环节承压？",
    "financial_quality": "收入质量、利润质量和现金流能否支撑结论？",
}

MODULE_ROLE_HINTS = {
    "industry_definition": "scope_definition",
    "market_size": "market_validation",
    "demand_driver": "demand_validation",
    "industry_chain": "supply_chain_or_value_chain",
    "technology": "technology_validation",
    "customer": "customer_validation",
    "business_model": "business_model_validation",
    "competition": "competition_validation",
    "policy": "policy_transmission",
    "capital": "capital_signal_validation",
    "risk": "falsification_and_risk",
    "entry_strategy": "decision_and_action",
    "timeline": "event_sequence",
    "stakeholder_map": "stakeholder_mapping",
    "transmission_chain": "impact_transmission",
    "beneficiary_loser": "beneficiary_loser_split",
    "financial_quality": "financial_quality_validation",
}

SEMICONDUCTOR_MODULE_TITLE_HINTS = {
    "market_size": "AI算力与成熟制程双轨需求如何改变全球半导体规模和结构？",
    "demand_driver": "AI、汽车、工业和消费电子需求会如何影响国产芯片兑现节奏？",
    "industry_chain": "出口管制与区域化分工如何重塑设备、制造和封测链条？",
    "technology": "先进制程受限后，先进封装、Chiplet 和 RISC-V 能补上哪些能力？",
    "competition": "中国企业在哪些环节能形成替代优势，哪些环节仍被卡住？",
    "policy": "美国管制、盟友协同与关税政策会怎样改变企业布局？",
    "capital": "大基金、资本开支和并购整合如何影响产业集中度？",
    "risk": "哪些反证信号会削弱国产替代和供应链重构判断？",
    "entry_strategy": "中国芯片产业应优先投入哪些环节，哪些风险需要先验证？",
    "customer": "下游客户认证、订单持续性和替代成本如何决定国产化进度？",
    "business_model": "国产替代能否沉淀为收入质量、利润质量和现金流？",
    "financial_quality": "资本开支、价格压力和现金流质量会如何约束扩张？",
}


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _compact(value: Any, max_chars: int = 220) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "..."


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


def _text_blob(query: str, research_plan: Dict[str, Any]) -> str:
    parts: List[str] = [query, research_plan.get("research_object"), research_plan.get("decision_context"), research_plan.get("research_type")]
    for collection in ("hypotheses", "dimensions", "evidence_goals", "search_tasks", "chapters"):
        for item in _as_list(research_plan.get(collection)):
            if isinstance(item, dict):
                parts.extend(
                    [
                        item.get("statement"),
                        item.get("claim_to_test"),
                        item.get("dimension_name"),
                        item.get("question"),
                        item.get("chapter_title"),
                        item.get("core_question"),
                        item.get("query"),
                    ]
                )
    return " ".join(str(part or "") for part in parts)


def _is_semiconductor_topic(query: str, research_plan: Dict[str, Any]) -> bool:
    blob = " ".join(
        str(value or "")
        for value in [
            query,
            research_plan.get("research_object"),
            research_plan.get("core_question"),
            research_plan.get("research_type"),
            research_plan.get("report_family"),
        ]
    )
    return bool(
        re.search(r"半导体|芯片|集成电路|晶圆|光刻|EDA|封测|Chiplet|先进制程|成熟制程", blob, flags=re.I)
        and re.search(r"中美|美国|中国|全球|供应链|管制|制裁|国产|重构|机遇|挑战", blob, flags=re.I)
    )


def _topic_specific_module_title(module_key: str, query: str, research_plan: Dict[str, Any]) -> str:
    if _is_semiconductor_topic(query, research_plan):
        return SEMICONDUCTOR_MODULE_TITLE_HINTS.get(module_key, "")
    return ""


def _sanitize_query_echo_title(title: str, query: str, research_plan: Dict[str, Any]) -> str:
    title = _compact(title, 180)
    if not title:
        return ""
    if not _is_semiconductor_topic(query, research_plan):
        return title
    compact_query = _normalize_for_title_match(query)
    compact_title = _normalize_for_title_match(title)
    if compact_query and compact_query not in compact_title and len(title) <= 90:
        return title
    rules = [
        (r"真实需求|概念热度", "供应链重构是否已经形成真实需求和订单验证？"),
        (r"价格.*产能.*订单|盈利质量|行情", "价格、产能、订单和盈利质量是否支撑产业链机会？"),
        (r"商业化证据|概念或试点", "哪些环节已经商业化，哪些仍处在试点或概念阶段？"),
        (r"进入/投资/产品布局|投资.*优先级|采购.*优先级", "投资、采购和产业布局应按什么优先级推进？"),
        (r"政策|监管|外部规则", "美国管制、盟友协同与关税政策会怎样改变企业布局？"),
        (r"技术路线|替代路径", "先进制程受限后，先进封装、Chiplet 和 RISC-V 能补上哪些能力？"),
    ]
    for pattern, replacement in rules:
        if re.search(pattern, title):
            return replacement
    return "全球半导体供应链重构下，中国芯片产业的机会和约束是什么？"


def _normalize_for_title_match(value: Any) -> str:
    return re.sub(r"\W+", "", str(value or "").lower())


def _score_modules(query: str, research_plan: Dict[str, Any], profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    blob = _text_blob(query, research_plan)
    profile_modules = [str(item) for item in _as_list(profile.get("candidate_modules"))]
    module_order = {key: index for index, key in enumerate(_as_list(profile.get("module_order")))}
    scores: List[Dict[str, Any]] = []
    for module in UNIVERSAL_REPORT_MODULES:
        key = str(module.get("module_key") or "")
        if key == "executive_summary":
            continue
        score = 0
        reasons: List[str] = []
        if key in profile_modules:
            score += 4
            reasons.append("profile_candidate")
        for trigger in _as_list(module.get("triggers")):
            if trigger and re.search(re.escape(str(trigger)), blob, flags=re.I):
                score += 3
                reasons.append(f"trigger:{trigger}")
        for evidence_type in _as_list(module.get("evidence_types")):
            if evidence_type and re.search(re.escape(str(evidence_type)), blob, flags=re.I):
                score += 1
        if key in module_order:
            score += max(0, 3 - module_order[key] // 3)
        if score > 0:
            scores.append({"module_key": key, "score": score, "reasons": _dedupe(reasons, limit=6)})
    for key in profile_modules:
        if not any(item.get("module_key") == key for item in scores):
            scores.append({"module_key": key, "score": 2, "reasons": ["profile_minimum"]})
    scores.sort(key=lambda item: (-int(item.get("score") or 0), module_order.get(str(item.get("module_key")), 99), str(item.get("module_key"))))
    return scores


def _explicit_chapter_candidates(research_plan: Dict[str, Any], *, query: str = "") -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for item in _as_list(research_plan.get("chapters")):
        if not isinstance(item, dict):
            continue
        title = _compact(item.get("chapter_title") or item.get("title"), 140)
        question = _compact(item.get("core_question") or item.get("chapter_question") or item.get("question") or title, 220)
        if not title and not question:
            continue
        clean_title = _sanitize_query_echo_title(title or question, query, research_plan)
        clean_question = _sanitize_query_echo_title(question or clean_title, query, research_plan)
        candidates.append({**item, "chapter_title": clean_title, "core_question": clean_question or clean_title, "source": "planner_chapter"})
    return candidates


def _hypothesis_candidates(research_plan: Dict[str, Any], *, query: str = "", limit: int = 4) -> List[Dict[str, Any]]:
    if limit <= 0:
        return []
    candidates: List[Dict[str, Any]] = []
    for item in _as_list(research_plan.get("hypotheses")):
        if not isinstance(item, dict):
            continue
        statement = _compact(item.get("statement") or item.get("claim_to_test") or item.get("hypothesis_statement"), 160)
        if not statement:
            continue
        clean_title = _sanitize_query_echo_title(statement, query, research_plan)
        candidates.append(
            {
                "chapter_title": clean_title if clean_title.endswith(("？", "?")) else clean_title,
                "core_question": clean_title or statement,
                "chapter_role": item.get("decision_use") or "hypothesis_validation",
                "required_evidence_mix": _as_list(item.get("required_evidence_types")) or [],
                "source": "hypothesis",
            }
        )
        if len(candidates) >= limit:
            break
    return candidates


def _module_candidate(module_key: str, query: str, research_plan: Dict[str, Any], profile: Dict[str, Any], score_item: Dict[str, Any]) -> Dict[str, Any]:
    module = module_by_key(module_key)
    title = (
        _topic_specific_module_title(module_key, query, research_plan)
        or MODULE_TITLE_HINTS.get(module_key)
        or _compact(_as_list(module.get("core_questions"))[0] if _as_list(module.get("core_questions")) else module.get("label"), 120)
    )
    evidence_mix = evidence_mix_for_modules([module_key])
    profile_required = _as_list(profile.get("required_evidence_roles"))
    if module_key == "risk":
        evidence_mix = _dedupe([*evidence_mix, "counter_evidence"], limit=8)
    if not evidence_mix:
        evidence_mix = profile_required or ["official_data", "market_research", "counter_evidence"]
    return {
        "chapter_title": title,
        "core_question": title,
        "chapter_role": MODULE_ROLE_HINTS.get(module_key, module_key),
        "module_keys": [module_key],
        "source_template_keys": [module_key],
        "required_evidence_mix": _dedupe([*evidence_mix, *profile_required], limit=8),
        "selection_score": int(score_item.get("score") or 0),
        "selection_reason": _as_list(score_item.get("reasons")) or ["module_score"],
        "source": "module_score",
    }


def _merge_candidates(candidates: List[Dict[str, Any]], *, min_chapters: int, max_chapters: int) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen_keys = set()
    for candidate in candidates:
        title = _compact(candidate.get("chapter_title"), 140)
        modules = _as_list(candidate.get("module_keys")) or _as_list(candidate.get("source_template_keys"))
        key = re.sub(r"\s+", "", title.lower())
        module_key = ",".join(str(item) for item in modules)
        if candidate.get("source") in {"planner_chapter", "hypothesis"}:
            dedupe_key = key[:64]
        else:
            dedupe_key = module_key or key[:32]
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        merged.append(candidate)
        if len(merged) >= max_chapters:
            break
    if len(merged) < min_chapters:
        for candidate in candidates:
            if candidate in merged:
                continue
            merged.append(candidate)
            if len(merged) >= min_chapters:
                break
    return merged[:max_chapters]


def _order_chapters(chapters: List[Dict[str, Any]], profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    order = {key: index for index, key in enumerate(_as_list(profile.get("module_order")))}

    def sort_key(chapter: Dict[str, Any]) -> tuple[int, int]:
        modules = _as_list(chapter.get("module_keys")) or _as_list(chapter.get("source_template_keys"))
        module_index = min([order.get(str(module), 50) for module in modules] or [50])
        source_bias = 0 if chapter.get("source") == "planner_chapter" else 1
        return (source_bias, module_index)

    return sorted(chapters, key=sort_key)


def _normalize_chapter(
    raw: Dict[str, Any],
    *,
    index: int,
    query: str,
    profile: Dict[str, Any],
    research_plan: Dict[str, Any],
) -> Dict[str, Any]:
    chapter_id = str(raw.get("chapter_id") or raw.get("id") or f"ch_{index:02d}").strip() or f"ch_{index:02d}"
    if not re.match(r"^ch[_-]?\d+", chapter_id, flags=re.I):
        chapter_id = f"ch_{index:02d}"
    title = _compact(raw.get("chapter_title") or raw.get("title") or raw.get("core_question") or f"研究问题 {index}", 140)
    question = _compact(raw.get("core_question") or raw.get("chapter_question") or raw.get("question") or title, 220)
    title = _sanitize_query_echo_title(title, query, research_plan)
    question = _sanitize_query_echo_title(question, query, research_plan)
    module_keys = _dedupe(_as_list(raw.get("module_keys")) or _as_list(raw.get("source_template_keys")), limit=5)
    if not module_keys:
        module_keys = _dedupe(_as_list(raw.get("source_template_keys")), limit=5)
    evidence_mix = _dedupe(_as_list(raw.get("required_evidence_mix")) or evidence_mix_for_modules(module_keys), limit=10)
    if not evidence_mix:
        evidence_mix = _dedupe(_as_list(profile.get("required_evidence_roles")) + ["counter_evidence"], limit=8)
    chapter = {
        "chapter_id": chapter_id,
        "chapter_title": title,
        "core_question": question,
        "chapter_question": question,
        "reason_to_include": _compact(raw.get("reason_to_include") or raw.get("selection_reason") or question, 220),
        "chapter_role": _compact(raw.get("chapter_role") or MODULE_ROLE_HINTS.get(module_keys[0], "") if module_keys else raw.get("chapter_role") or question, 160),
        "module_keys": module_keys,
        "source_template_keys": module_keys,
        "required_evidence_mix": evidence_mix,
        "min_total_sources": int(raw.get("min_total_sources") or 8),
        "min_ab_sources": int(raw.get("min_ab_sources") or 2),
        "min_counter_sources": int(raw.get("min_counter_sources") or (1 if any("counter" in item for item in evidence_mix) else 0)),
        "evidence_goals": _as_list(raw.get("evidence_goals")),
        "search_task_hints": _as_list(raw.get("search_tasks") or raw.get("search_task_hints")),
        "selection": {
            "source": raw.get("source") or "compiled",
            "score": raw.get("selection_score"),
            "reason": raw.get("selection_reason"),
        },
        "order": index,
    }
    blocks = select_blocks_for_chapter(chapter, profile=profile, limit=5)
    chapter["layout_policy"] = {
        "preferred_blocks": [block.get("block_type") for block in blocks[:4]],
        "optional_blocks": [block.get("block_type") for block in blocks[4:]],
        "block_selection_source": "layout_compiler",
    }
    return chapter


def _narrative(profile: Dict[str, Any], chapters: List[Dict[str, Any]], research_object: str) -> str:
    spine = _as_list(profile.get("narrative_spines"))[0] if _as_list(profile.get("narrative_spines")) else "evidence_to_judgment"
    public_spines = {
        "definition_to_opportunity": "先界定研究对象，再判断需求、供给、机会与风险",
        "demand_supply_risk": "沿需求、供给、竞争和风险展开",
        "problem_to_entry_decision": "从真实需求走向进入窗口和行动排序",
        "demand_to_channel_to_risk": "沿需求、渠道、竞争和风险展开",
        "investment_thesis_to_risk": "从投资假设走向风险校准",
        "positioning_to_gap_to_action": "从竞争定位走向差距和行动选择",
        "policy_original_to_transmission": "从政策原文走向影响传导和风险边界",
        "consumer_need_to_channel_to_brand": "从用户需求走向渠道、品牌和复购",
        "bottleneck_to_resilience_to_opportunity": "从瓶颈约束走向韧性和机会",
        "technology_maturity_to_adoption": "从技术成熟度走向应用落地和商业化约束",
        "thesis_to_variant_view_to_risk": "从核心假设走向分歧情景和风险触发器",
        "facts_to_judgment_to_watchlist": "从事实归纳走向判断和观察清单",
        "evidence_to_judgment": "从证据归纳走向判断和边界",
    }
    spine_label = public_spines.get(spine, "从证据归纳走向判断和边界")
    titles = "；".join(_compact(chapter.get("chapter_title"), 80) for chapter in chapters[:5])
    return f"以{research_object or '研究对象'}为对象，{spine_label}：{titles}"


def validate_dynamic_blueprint(chapters: List[Dict[str, Any]], quality_rules: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    rules = _as_dict(quality_rules)
    issues: List[Dict[str, Any]] = []
    titles = [str(chapter.get("chapter_title") or "").strip() for chapter in chapters]
    fixed_words = ["真实需求", "概念热度", "商业化弹性", "价格、产能、订单", "进入/投资/产品布局"]
    for index, title in enumerate(titles, start=1):
        if any(word in title for word in fixed_words):
            issues.append({"type": "template_like_title", "chapter_index": index, "title": title})
        if len(title) > 90:
            issues.append({"type": "overlong_chapter_title", "chapter_index": index, "title": title, "severity": "warning"})
    if len(set(re.sub(r"\W+", "", title[:12]) for title in titles if title)) < max(1, len(titles) // 2):
        issues.append({"type": "chapter_titles_too_similar", "severity": "warning"})
    if rules.get("forbid_fixed_micro_sections"):
        block_sets = [
            tuple(_as_list(_as_dict(chapter.get("layout_policy")).get("preferred_blocks"))[:3])
            for chapter in chapters
        ]
        if len(block_sets) >= 3 and len(set(block_sets)) == 1:
            issues.append({"type": "fixed_micro_block_pattern", "severity": "warning"})
    blocking = [item for item in issues if item.get("severity") != "warning"]
    return {"passed": not blocking, "issues": issues, "blocking_count": len(blocking), "warning_count": len(issues) - len(blocking)}


def compile_report_layout(
    *,
    query: str = "",
    research_plan: Optional[Dict[str, Any]] = None,
    report_plan: Optional[Dict[str, Any]] = None,
    evidence_package: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    del evidence_package
    research_plan = _as_dict(research_plan)
    report_plan = _as_dict(report_plan)
    profile = select_report_profile(query, research_plan, report_plan)
    min_chapters = int(profile.get("min_body_chapters") or 3)
    max_chapters = int(profile.get("max_body_chapters") or 6)
    scores = _score_modules(query, research_plan, profile)

    explicit_candidates = _explicit_chapter_candidates(research_plan, query=query)
    use_explicit_first = bool(_as_dict(research_plan.get("quality_rules")).get("chapters_come_from_hypotheses"))
    candidates: List[Dict[str, Any]] = []
    if use_explicit_first and explicit_candidates:
        candidates.extend(explicit_candidates)
        if len(candidates) < min_chapters:
            for score_item in scores:
                candidates.append(_module_candidate(str(score_item.get("module_key")), query, research_plan, profile, score_item))
                if len(candidates) >= min_chapters:
                    break
    else:
        for score_item in scores:
            candidates.append(_module_candidate(str(score_item.get("module_key")), query, research_plan, profile, score_item))
        if len(candidates) < max_chapters:
            candidates.extend(explicit_candidates[: max(0, max_chapters - len(candidates))])
    candidates.extend(_hypothesis_candidates(research_plan, query=query, limit=max(0, min_chapters - len(candidates))))
    if not candidates:
        object_label = _compact(query, 80) or "这个研究问题"
        candidates.append(
            {
                "chapter_title": f"{object_label}首先需要判断什么？",
                "core_question": f"回答“{query or '研究问题'}”需要哪些事实、反证和行动含义？",
                "module_keys": ["risk"],
                "required_evidence_mix": ["support", "counter_evidence"],
                "source": "fallback",
            }
        )

    selected = _order_chapters(_merge_candidates(candidates, min_chapters=min_chapters, max_chapters=max_chapters), profile)
    chapters = [
        _normalize_chapter(raw, index=index, query=query, profile=profile, research_plan=research_plan)
        for index, raw in enumerate(selected, start=1)
    ]
    research_object = _compact(research_plan.get("research_object") or report_plan.get("research_object") or query or "研究对象", 160)
    quality_rules = {
        **_as_dict(research_plan.get("quality_rules")),
        "forbid_legacy_five_dimensions": True,
        "forbid_fixed_micro_sections": True,
        "profile_aware_contract": True,
        "chapter_must_have_question": True,
        "layout_stage": "compiled_dynamic_report_structure",
    }
    blueprint = {
        "agent": AGENT_NAME,
        "report_family": profile.get("name"),
        "research_type": research_plan.get("research_type") or profile.get("name"),
        "research_object": research_object,
        "layout_strategy": {
            "profile": profile.get("name"),
            "narrative_spine": _as_list(profile.get("narrative_spines"))[0] if _as_list(profile.get("narrative_spines")) else "evidence_to_judgment",
            "chapter_selection_policy": "profile_module_hypothesis_evidence_weighted",
            "max_body_chapters": max_chapters,
            "min_body_chapters": min_chapters,
            "module_scores": scores[:12],
        },
        "report_shell": default_report_shell(profile),
        "narrative": _narrative(profile, chapters, research_object),
        "chapters": chapters,
        "quality_rules": quality_rules,
        "layout_validation": validate_dynamic_blueprint(chapters, quality_rules),
        "dropped_template_sections": _as_list(research_plan.get("dropped_template_sections")),
    }
    return blueprint
