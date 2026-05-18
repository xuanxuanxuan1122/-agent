from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence


AGENT_NAME = "claim_builder_agent"
AGENT_DESCRIPTION = "Claim Builder Agent. Converts chapter evidence into claim/argument units."

ACTION_WORDS = ("优先", "避免", "验证", "跟踪", "排除", "补充", "设置")
WEAK_CLAIM_PREFIXES = ("已有可验证", "已有可核验", "已有可用证据", "当前证据")
BAD_CLAIM_PATTERNS = [
    r"已有可核验证据",
    r"已有可验证证据",
    r"证据不足",
    r"可作为判断输入",
    r"需结合来源等级",
    r"需结合.*时间范围",
    r"需结合.*口径边界",
    r"尚未发现足以推翻",
    r"继续补证",
]
CAUSE_WORDS = ("因为", "由于", "原因", "导致", "从而", "因此")
PUBLIC_BLOCKING_PATTERNS = [
    r"低置信",
    r"不能作为确定性结论",
    r"证据不足",
    r"暂无可核验",
    r"建议补充",
    r"A/B\s*级来源不足",
    r"权威来源交叉验证",
    r"needs_corroboration",
    r"insufficient",
    r"unsupported",
]
BAD_FACT_PATTERNS = [
    r"联网分析\s*Agent\s*失败",
    r"Retrieval\.TestUserQueryExceeded",
    r"query\s+exceed(?:ed)?\s+the\s+limit",
    r"目前更像局部信号",
    r"已通过\s*IQS\s*获取到联网证据",
    r"当前未启用或未成功调用大模型综合分析",
    r"先给出可核验的网页结果摘要",
    r"关键依据[:：]\s*\d+\.",
]

EVIDENCE_COLLECTIONS = (
    "core_evidence",
    "supporting_evidence",
    "sample_evidence",
    "table_evidence",
    "clue_evidence",
    "appendix_evidence",
    "evidence_items",
)

PUBLIC_EVIDENCE_COLLECTIONS = (
    "core_evidence",
    "supporting_evidence",
    "sample_evidence",
    "table_evidence",
    "clue_evidence",
)


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _compact(value: Any, max_chars: int = 260) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "..."


def _dedupe(values: Iterable[Any], *, limit: int = 8) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        text = _compact(value, 120)
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
        "REPORT_FACTS_PER_REASONING": 4,
        "REPORT_REASONING_FACT_TEXT_MAX_CHARS": 550,
        "REPORT_FACTS_PER_CHAPTER_ARGUMENTS": 8,
        "REPORT_ARGUMENT_UNITS_PER_CHAPTER": 4,
    },
    "balanced": {
        "REPORT_FACTS_PER_REASONING": 5,
        "REPORT_REASONING_FACT_TEXT_MAX_CHARS": 700,
        "REPORT_FACTS_PER_CHAPTER_ARGUMENTS": 12,
        "REPORT_ARGUMENT_UNITS_PER_CHAPTER": 4,
    },
    "deep": {
        "REPORT_FACTS_PER_REASONING": 6,
        "REPORT_REASONING_FACT_TEXT_MAX_CHARS": 900,
        "REPORT_FACTS_PER_CHAPTER_ARGUMENTS": 18,
        "REPORT_ARGUMENT_UNITS_PER_CHAPTER": 5,
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


def _line_key(value: Any) -> str:
    return re.sub(r"[\s，。；：:、,.!?！？“”\"'（）()《》]+", "", str(value or "")).lower()


def _has_bad_pattern(value: Any) -> bool:
    text = str(value or "")
    return any(re.search(pattern, text) for pattern in BAD_CLAIM_PATTERNS)


def _is_bad_public_fact(value: Any) -> bool:
    text = str(value or "")
    if not text:
        return True
    return any(re.search(pattern, text, flags=re.I) for pattern in BAD_FACT_PATTERNS)


def _support_profile(package: Dict[str, Any], refs: Sequence[str]) -> Dict[str, Any]:
    wanted = {str(ref or "").strip() for ref in refs if str(ref or "").strip()}
    items = [
        item
        for collection in EVIDENCE_COLLECTIONS
        for item in _as_list(package.get(collection))
        if isinstance(item, dict)
    ]
    matched = []
    for item in items:
        source_id = str(item.get("source_id") or "").strip()
        citation_ref = _citation_ref_from_evidence(item)
        item_refs = {
            str(item.get("ref") or "").strip(),
            str(item.get("evidence_id") or "").strip(),
            source_id,
            f"[{source_id}]" if re.fullmatch(r"\d{1,3}", source_id) else "",
            citation_ref,
            *[str(ref or "").strip() for ref in _as_list(item.get("source_refs"))],
        }
        if wanted and not wanted.intersection({ref for ref in item_refs if ref}):
            continue
        matched.append(item)
    levels: Dict[str, int] = {}
    roles: Dict[str, int] = {}
    allowed_uses: Dict[str, int] = {}
    contextual_ab_count = 0
    claim_ab_count = 0
    for item in matched:
        level = str(item.get("source_level") or _as_dict(item.get("source")).get("credibility") or "").strip().upper() or "UNKNOWN"
        levels[level] = levels.get(level, 0) + 1
        role = str(item.get("evidence_role") or item.get("role") or "").strip().lower() or "unknown"
        roles[role] = roles.get(role, 0) + 1
        allowed_use = str(item.get("allowed_use") or _as_dict(item.get("evidence_card")).get("allowed_use") or "").strip() or "unknown"
        allowed_uses[allowed_use] = allowed_uses.get(allowed_use, 0) + 1
        is_contextual = allowed_use in {"supporting_context", "contextual_support"} or str(item.get("usage_tier") or "") == "corroborated_context"
        if level in {"A", "B"} and is_contextual:
            contextual_ab_count += 1
        elif level in {"A", "B"}:
            claim_ab_count += 1
    ab_count = levels.get("A", 0) + levels.get("B", 0)
    cd_count = levels.get("C", 0) + levels.get("D", 0)
    core = roles.get("core", 0)
    supporting = roles.get("supporting", 0)
    clue = roles.get("clue", 0)
    directional = allowed_uses.get("directional_signal", 0)
    if core >= 1 or claim_ab_count >= 2:
        strength = "strong"
    elif claim_ab_count >= 1 or (supporting >= 1 and contextual_ab_count == 0):
        strength = "medium"
    elif contextual_ab_count >= 1 or directional >= 1:
        strength = "directional"
    elif clue >= 1:
        strength = "weak"
    else:
        strength = "unsupported"
    if strength == "strong" or claim_ab_count >= 2:
        grade = "high"
    elif strength == "medium" or (claim_ab_count == 1 and cd_count == 0):
        grade = "medium"
    elif claim_ab_count == 1 or contextual_ab_count >= 2:
        grade = "medium_low"
    else:
        grade = "low"
    return {
        "grade": grade,
        "source_level_distribution": levels,
        "ab_count": ab_count,
        "claim_ab_count": claim_ab_count,
        "contextual_ab_count": contextual_ab_count,
        "cd_count": cd_count,
        "matched_count": len(matched),
        "role_distribution": roles,
        "allowed_use_distribution": allowed_uses,
        "claim_strength": strength,
    }


def is_public_claim(unit: Dict[str, Any]) -> bool:
    if unit.get("omit_from_report"):
        return False

    claim_status = str(unit.get("claim_status") or "").lower()
    status = str(unit.get("quality_status") or "").lower()
    if status in {"unsupported", "invalid", "weak"}:
        return False
    if status == "insufficient" and claim_status not in {"directional", "context_only"}:
        return False

    source_quality = _as_dict(unit.get("source_quality"))
    strength = str(source_quality.get("claim_strength") or "").lower()
    if strength in {"weak", "unsupported"} and claim_status not in {"directional", "context_only"}:
        return False

    text = " ".join(
        str(unit.get(key) or "")
        for key in ["claim", "reasoning", "counter_evidence", "actionable"]
    )
    if any(re.search(pattern, text, re.I) for pattern in PUBLIC_BLOCKING_PATTERNS):
        return False

    if not _as_list(unit.get("evidence_refs")):
        return False

    return True


def _matches(unit: Dict[str, Any], package: Dict[str, Any]) -> bool:
    unit_chapter_id = str(unit.get("chapter_id") or "").strip()
    package_chapter_id = str(package.get("chapter_id") or "").strip()
    if unit_chapter_id and package_chapter_id:
        return unit_chapter_id == package_chapter_id

    fields = [
        unit.get("dimension"),
        unit.get("question"),
        unit.get("section_title"),
    ]
    targets = [
        package.get("chapter_id"),
        package.get("chapter_title"),
        package.get("chapter_question"),
    ]
    def _norm(value: Any) -> str:
        return re.sub(r"\s+", "", str(value or "").lower()).strip()

    def _bigrams(value: str) -> set[str]:
        chars = [char for char in value if char.strip()]
        return {"".join(chars[index : index + 2]) for index in range(max(0, len(chars) - 1))}

    normalized_fields = [_norm(item) for item in fields if str(item or "").strip()]
    normalized_targets = [_norm(item) for item in targets if str(item or "").strip()]
    if not normalized_fields or not normalized_targets:
        return False
    for field in normalized_fields:
        for target in normalized_targets:
            if field == target or field in target or target in field:
                return True
            if len(field) < 8 or len(target) < 8:
                continue
            field_bigrams = _bigrams(field)
            target_bigrams = _bigrams(target)
            if not field_bigrams or not target_bigrams:
                continue
            overlap = len(field_bigrams & target_bigrams)
            ratio = overlap / max(1, min(len(field_bigrams), len(target_bigrams)))
            if ratio >= 0.90:
                return True
    return False


def _structured_units(structured_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    units: List[Dict[str, Any]] = []
    insight_package = _as_dict(structured_analysis.get("report_insight_package"))
    for chapter in _as_list(insight_package.get("chapters")) + _as_list(structured_analysis.get("chapter_insights")):
        if not isinstance(chapter, dict):
            continue
        chapter_question = chapter.get("chapter_question") or chapter.get("chapter_answer")
        for index, claim_item in enumerate(_as_list(chapter.get("key_claims")), start=1):
            if not isinstance(claim_item, dict):
                continue
            claim = _compact(claim_item.get("claim"), 320)
            if not claim:
                continue
            units.append(
                {
                    "chapter_id": chapter.get("chapter_id"),
                    "dimension": chapter.get("chapter_question") or chapter.get("chapter_id"),
                    "question": chapter_question,
                    "section_title": chapter.get("chapter_question") or f"核心判断 {index}",
                    "fact": claim_item.get("supporting_fact") or claim_item.get("fact") or chapter.get("chapter_answer"),
                    "claim": claim,
                    "reasoning": claim_item.get("mechanism") or claim_item.get("reasoning") or "",
                    "mechanism": claim_item.get("mechanism") or "",
                    "counter_evidence": claim_item.get("counter_evidence") or "",
                    "decision_implication": claim_item.get("decision_implication") or "",
                    "actionable": claim_item.get("decision_implication") or "",
                    "confidence": claim_item.get("confidence"),
                    "supporting_evidence": _as_list(claim_item.get("supporting_evidence")) or _as_list(claim_item.get("evidence_refs")),
                    "evidence_refs": _as_list(claim_item.get("supporting_evidence")) or _as_list(claim_item.get("evidence_refs")),
                    "what_to_verify_next": _as_list(claim_item.get("what_to_verify_next")),
                }
            )
    for key in ("claim_units", "analysis_units"):
        for item in _as_list(structured_analysis.get(key)):
            if isinstance(item, dict):
                if not str(item.get("claim") or item.get("judgment") or item.get("conclusion") or "").strip():
                    continue
                units.append(dict(item))
    for item in _as_list(structured_analysis.get("evidence_analyses")):
        if not isinstance(item, dict):
            continue
        if item.get("evidence_card_only"):
            continue
        if str(item.get("allowed_use") or "").strip() not in {"core_claim"}:
            continue
        claim = _compact(item.get("claim") or item.get("takeaway") or item.get("judgment") or item.get("fact"), 260)
        if not claim:
            continue
        units.append(
            {
                "dimension": item.get("dimension"),
                "question": item.get("question") or item.get("dimension"),
                "fact": item.get("fact") or item.get("writer_evidence") or item.get("data_point"),
                "claim": claim,
                "reasoning": item.get("reasoning") or item.get("mechanism") or item.get("explain_why") or "",
                "mechanism": item.get("mechanism") or "",
                "counter_evidence": item.get("counter_evidence") or item.get("counter") or "",
                "decision_implication": item.get("decision_implication") or item.get("actionable") or "",
                "confidence": item.get("confidence"),
                "supporting_evidence": [item.get("evidence_id")] if item.get("evidence_id") else [],
            }
        )
    return units


def _question_for(package: Dict[str, Any], unit: Optional[Dict[str, Any]] = None) -> str:
    unit = _as_dict(unit)
    return _compact(
        unit.get("question")
        or unit.get("section_title")
        or package.get("chapter_question")
        or package.get("chapter_title")
        or "本章需要回答的关键问题",
        180,
    )


def _citation_ref_from_evidence(item: Dict[str, Any]) -> str:
    for key in ("source_ref", "citation_ref"):
        value = str(item.get(key) or "").strip()
        if re.fullmatch(r"\[\d{1,3}\]", value):
            return value
    source_id = str(item.get("source_id") or "").strip()
    if re.fullmatch(r"\d{1,3}", source_id):
        return f"[{source_id}]"
    ref = str(item.get("ref") or "").strip()
    if re.fullmatch(r"\[\d{1,3}\]", ref):
        return ref
    return ref or str(item.get("evidence_id") or "").strip()


def _source_refs_for_evidence_refs(package: Dict[str, Any], refs: Sequence[Any]) -> List[str]:
    wanted = {str(ref or "").strip() for ref in refs if str(ref or "").strip()}
    if not wanted:
        return []
    mapped: List[str] = []
    for collection in PUBLIC_EVIDENCE_COLLECTIONS:
        for item in _as_list(package.get(collection)):
            if not isinstance(item, dict):
                continue
            keys = {
                str(item.get("evidence_id") or "").strip(),
                str(item.get("ref") or "").strip(),
                str(item.get("source_id") or "").strip(),
                str(item.get("source_ref") or "").strip(),
                str(item.get("citation_ref") or "").strip(),
                _citation_ref_from_evidence(item),
            }
            if keys.intersection(wanted):
                mapped.append(_citation_ref_from_evidence(item))
    return _dedupe([ref for ref in mapped if ref], limit=8)


def _claim_from_fact(package: Dict[str, Any], fact: str) -> str:
    question = _question_for(package)
    title = _compact(package.get("chapter_title"), 80) or "本章"
    fact = _compact(fact, 180)
    if re.search(r"谁|付费|客户|用户|买单", question):
        return f"{title}的核心变量在付费主体，预算明确、使用频次稳定、替代成本足够高时，需求才会从兴趣转为采购。"
    if re.search(r"政策|监管|执行|影响", question + title):
        return f"{title}的关键不在政策表态本身，而在执行机制能否传导到预算、审批或采购动作。"
    if re.search(r"产品|机会|进入|切入", question + title):
        return f"{title}更适合作为阶段性切入口，而不是直接外推为整体市场机会。"
    if fact:
        return f"{title}当前材料能说明阶段性方向，但还不足以直接外推为整体机会；需要继续核验主体动作、指标连续性和反向样本。"
    return f"{title}目前更适合作为背景条件，结论强度取决于后续连续指标和相反样本的变化。"


def _directional_claim(package: Dict[str, Any], fact: str = "") -> str:
    title = _compact(package.get("chapter_title"), 80) or "本章"
    text = f"{title} {_question_for(package)}"
    if _looks_like_semiconductor_topic(package):
        if re.search(r"政策|监管|管制|关税|规则", text):
            return f"{title}需要从管制执行、许可豁免、盟友协同和企业迁移四条线共同判断，不能只看单一政策表态。"
        if re.search(r"技术|制程|封装|RISC|Chiplet|EDA|光刻", text, flags=re.I):
            return f"{title}的判断重点在先进制程约束与替代技术的实际承接能力，而不是把单点突破直接外推为系统性替代。"
        if re.search(r"产业链|供应链|瓶颈|利润池|位置", text):
            return f"{title}需要拆到设备、材料、制造、封测和下游客户之间的传导关系，才能判断供应链重构的真实方向。"
        if re.search(r"客户|采购|需求|场景", text):
            return f"{title}取决于客户认证、订单持续性和替代成本，只有需求端动作明确时，国产替代才会进入兑现阶段。"
        if re.search(r"竞争|玩家|壁垒", text):
            return f"{title}应区分成熟制程、封测、设备材料和先进算力等不同战场，各环节的壁垒和机会不能混在一起判断。"
        return f"{title}当前材料只能提供阶段性线索，还需要用高等级来源、企业行为和反向样本共同校准机会强度。"
    if fact:
        return _claim_from_fact(package, fact)
    return f"{title}需要更多可核验事实确认适用范围和决策优先级。"


def _normalize_claim(unit: Dict[str, Any], package: Dict[str, Any], fallback_fact: str) -> str:
    claim = _compact(unit.get("claim") or unit.get("judgment") or unit.get("conclusion"), 320)
    if not claim or claim.startswith(WEAK_CLAIM_PREFIXES) or _has_bad_pattern(claim):
        return _claim_from_fact(package, fallback_fact)
    return claim


def _ensure_reasoning(value: Any, package: Dict[str, Any], fallback_fact: str) -> str:
    text = _compact(value, 520)
    if not text or _has_bad_pattern(text):
        text = "该信号从事实线索转为可执行结论，取决于来源口径、适用场景和付费或执行主体是否同时成立。"
    elif not re.search(r"因为|由于|原因|导致|从而|因此", text):
        text = f"{text}"
    if fallback_fact and _line_key(fallback_fact)[:48] not in _line_key(text):
        text = f"{text} 材料中已经出现的可观察事实是：{_compact(fallback_fact, 160)}"
    return text


_TEMPLATE_COUNTER_FALLBACK = "样本、时间窗口和来源口径会影响结论强度；相反案例或更高等级来源出现时，原有结论会收缩。"
_TEMPLATE_ACTION_FALLBACK = "后续重点跟踪同口径指标、反向样本和执行进展，再根据连续变化安排资源投入。"


def _template_fallbacks_enabled() -> bool:
    """Template fallbacks are noisy and make the rendered report feel machine-stitched.
    They are disabled by default; set REPORT_TEMPLATE_FALLBACKS=1 to re-enable
    the historic always-output behaviour."""
    return os.environ.get("REPORT_TEMPLATE_FALLBACKS", "0").strip() in {"1", "true", "True"}


def _ensure_counter(value: Any) -> str:
    text = _compact(value, 340)
    if text and not _has_bad_pattern(text):
        return text
    return _TEMPLATE_COUNTER_FALLBACK if _template_fallbacks_enabled() else ""


def _ensure_actionable(value: Any) -> str:
    text = _compact(value, 340)
    if text:
        if not any(word in text for word in ACTION_WORDS):
            return f"后续重点跟踪{text}"
        return text
    return _TEMPLATE_ACTION_FALLBACK if _template_fallbacks_enabled() else ""


def _first_counter_from_package(package: Dict[str, Any]) -> str:
    """Pull a counter-evidence statement from the chapter package itself before
    falling back to a generic template. Looks at:
    - chapter-level counter_evidence list
    - first hypothesis' falsification_triggers / must_disprove
    - first item in counter_evidence collection on evidence_items
    """
    pkg = _as_dict(package)
    # 1. Chapter-level
    for candidate in _as_list(pkg.get("counter_evidence")) + _as_list(pkg.get("counter_signals")):
        if isinstance(candidate, str):
            text = _compact(candidate, 320)
            if text:
                return text
        elif isinstance(candidate, dict):
            text = _compact(candidate.get("statement") or candidate.get("fact") or candidate.get("text"), 320)
            if text:
                return text
    # 2. From hypotheses bound to this chapter
    for h in _as_list(pkg.get("hypotheses")):
        h = _as_dict(h)
        triggers = _as_list(h.get("falsification_triggers")) + _as_list(h.get("must_disprove"))
        for t in triggers:
            text = _compact(t, 320)
            if text:
                return text
    return ""


def _first_action_from_package(package: Dict[str, Any]) -> str:
    """Pull an action / next-step recommendation from the package before
    falling back to a generic template."""
    pkg = _as_dict(package)
    for candidate in (
        pkg.get("actionable"),
        _as_dict(pkg.get("chapter_summary")).get("next_action"),
        _as_dict(pkg.get("chapter_summary")).get("recommended_action"),
    ):
        text = _compact(candidate, 320)
        if text:
            return text
    for h in _as_list(pkg.get("hypotheses")):
        h = _as_dict(h)
        for key in ("decision_relevance", "next_action", "what_to_validate"):
            text = _compact(h.get(key), 320)
            if text:
                return text
    return ""


def _verification_metrics(package: Dict[str, Any], refs: Sequence[str]) -> List[str]:
    title = str(package.get("chapter_title") or "")
    if re.search(r"市场|规模|增速|增长", title):
        return ["同口径规模", "增速区间", "可服务市场边界"]
    if re.search(r"客户|用户|需求|付费", title):
        return ["付费主体", "采购频次", "替代成本"]
    if re.search(r"政策|监管", title):
        return ["执行部门", "预算/目录传导", "落地时间"]
    if re.search(r"技术|产品", title):
        return ["连续作业表现", "故障率", "ROI"]
    return ["来源等级", "反向案例", "可比口径"]


def _proof_gap_texts(package: Dict[str, Any]) -> List[str]:
    texts: List[str] = []
    for item in _as_list(package.get("missing_proof_standards")):
        item = _as_dict(item)
        hypothesis = _compact(item.get("hypothesis_statement") or item.get("hypothesis_id"), 100)
        gaps = "、".join(_dedupe(_as_list(item.get("blocking_gaps")), limit=4))
        if hypothesis or gaps:
            texts.append(f"{hypothesis}：{gaps}".strip("："))
    return _dedupe(texts, limit=5)


def _proof_followups(package: Dict[str, Any]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    seen = set()
    for item in _as_list(package.get("proof_follow_up_queries")) + _as_list(package.get("follow_up_queries")):
        if isinstance(item, dict):
            key = re.sub(r"\s+", "", str(item.get("query") or item.get("suggested_query") or "").lower())
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            result.append(item)
    return result[:8]


def _claim_status_from_support(support: Dict[str, Any], proof_gaps: Sequence[Any]) -> str:
    strength = str(support.get("claim_strength") or "").lower()
    if not proof_gaps and strength in {"strong", "medium"}:
        return "decision_ready"
    if strength in {"strong", "medium", "directional"}:
        return "directional"
    if int(support.get("matched_count") or 0) > 0:
        return "context_only"
    return "appendix_only"


def _public_verification_focus(package: Dict[str, Any], refs: Sequence[str]) -> List[str]:
    base = _verification_metrics(package, refs)
    gap_map = {
        "insufficient_ab_sources": "同一事实在不同来源中的一致性",
        "counter_evidence_missing": "相反样本或反向价格/订单变化",
        "metric_evidence_missing": "价格、库存、产能、销量等连续指标",
        "case_evidence_missing": "企业订单、客户验证或区域样本",
        "source_diversity_missing": "统计、产业链和企业侧口径差异",
        "metric_definition_unfilled": "指标范围、期间和单位",
        "metric_scope_period_unit_incomplete": "指标口径的可比性",
    }
    extras: List[str] = []
    for item in _as_list(package.get("missing_proof_standards")):
        for gap in _as_list(_as_dict(item).get("blocking_gaps")):
            label = gap_map.get(str(gap))
            if label:
                extras.append(label)
    return _dedupe([*base, *extras], limit=8)


def _unit_from_structured(unit: Dict[str, Any], package: Dict[str, Any], section_id: str, fallback_refs: Sequence[str]) -> Dict[str, Any]:
    raw_refs = _as_list(unit.get("supporting_evidence")) or _as_list(unit.get("evidence_refs")) or fallback_refs
    refs = _source_refs_for_evidence_refs(package, raw_refs) or _dedupe(raw_refs, limit=8)
    fallback_fact = _compact(unit.get("fact") or unit.get("supporting_fact") or "", 220)
    question = _question_for(package, unit)
    original_claim = _compact(unit.get("claim") or unit.get("judgment") or unit.get("conclusion"), 320)
    original_text = " ".join(
        _compact(unit.get(key), 320)
        for key in ("claim", "reasoning", "mechanism", "counter_evidence", "actionable", "decision_implication")
    )
    normalized_from_weak = bool(original_claim.startswith(WEAK_CLAIM_PREFIXES) or _has_bad_pattern(original_text))
    claim = _normalize_claim(unit, package, fallback_fact)
    support = _support_profile(package, refs)
    confidence = unit.get("confidence") or ("low" if support["grade"] == "low" else "medium")
    proof_gaps = _as_list(package.get("missing_proof_standards"))
    claim_status = _claim_status_from_support(support, proof_gaps)
    payload = {
        "agent": AGENT_NAME,
        "chapter_id": package.get("chapter_id"),
        "section_id": section_id,
        "question": question,
        "section_title": _compact(unit.get("section_title") or question, 160),
        "claim": claim,
        "reasoning": _ensure_reasoning(unit.get("reasoning") or unit.get("mechanism") or unit.get("explain_why"), package, fallback_fact),
        "counter_evidence": _ensure_counter(unit.get("counter_evidence") or unit.get("counter")),
        "actionable": _ensure_actionable(unit.get("actionable") or unit.get("decision_implication") or unit.get("next_action")),
        "mechanism": _compact(unit.get("mechanism") or unit.get("reasoning") or unit.get("explain_why"), 520),
        "what_to_verify_next": _dedupe(_as_list(unit.get("what_to_verify_next")) + _public_verification_focus(package, refs), limit=8),
        "confidence": confidence,
        "confidence_reason": "该判断按来源层级、指标口径和反向样本覆盖程度分级，公开表达采用相应边界。",
        "claim_status": claim_status,
        "supporting_evidence": _dedupe(refs, limit=8),
        "evidence_refs": _dedupe(refs, limit=8),
        "supporting_facts": _evidence_facts_from_package(package, limit=8),
        "verification_metrics": _public_verification_focus(package, refs),
        "source_quality": support,
        "rewrite_required": normalized_from_weak,
        "rewrite_reason": "weak_structured_unit" if normalized_from_weak else "",
        "proof_gaps": proof_gaps,
        "follow_up_queries": _proof_followups(package),
    }
    if not payload.get("mechanism"):
        payload["mechanism"] = payload.get("reasoning") or ""
    if support["grade"] == "low":
        payload = rewrite_weak_claim_unit(payload, package=package, reason="low_source_quality")
    if proof_gaps:
        payload["claim_status"] = "directional" if claim_status in {"decision_ready", "directional"} else "context_only"
        payload["omit_from_report"] = False
        payload["public_render"] = True
        payload["quality_status"] = "directional_with_boundary"
        payload["internal_reason"] = "missing_proof_bundle"
    return payload


def _unit_from_evidence(package: Dict[str, Any], section_id: str) -> Dict[str, Any]:
    proof_gaps = _as_list(package.get("missing_proof_standards"))
    core = [item for item in _as_list(package.get("core_evidence")) if isinstance(item, dict)]
    supporting = [
        item
        for collection in ("supporting_evidence", "sample_evidence")
        for item in _as_list(package.get(collection))
        if isinstance(item, dict)
    ]
    clue = [item for item in _as_list(package.get("clue_evidence")) if isinstance(item, dict)]
    candidates = core or supporting or clue
    first = candidates[0] if candidates else {}
    fact = _compact(first.get("fact"), 260)
    supporting_facts = _evidence_facts_from_package(package, limit=8)
    refs = _dedupe([_citation_ref_from_evidence(item) for item in (core + supporting + clue)[:8]], limit=8)
    question = _question_for(package)
    support = _support_profile(package, refs)
    strength = str(support.get("claim_strength") or "")
    if _is_bad_public_fact(fact):
        fact = ""
    if fact and strength in {"strong", "medium"} and support["grade"] != "low":
        claim = _claim_from_fact(package, fact)
        reasoning = f"公开材料显示：{fact}。这一事实需要结合应用场景、披露主体和统计口径理解；如果后续同口径数据、客户行为和反向样本继续同向，才适合支撑更强的行业判断。"
    elif fact and strength in {"directional", "weak"}:
        claim = _directional_claim(package, fact)
        reasoning = f"公开材料显示：{_compact(fact, 160)}。目前它更适合作为方向性信号，需要继续观察价格、库存、供给执行和下游需求是否连续同向。"
    elif fact:
        claim = f"{package.get('chapter_title') or '本章'}当前只能作为背景条件，用来限定后续判断边界。"
        reasoning = f"公开材料显示：{_compact(fact, 160)}。该事实可以说明局部状态，但还不能单独解释趋势是否持续。"
    else:
        claim = f"{package.get('chapter_title') or '本章'}目前更适合作为背景观察项。"
        reasoning = "目前公开信息尚未形成足够清晰的事实链，只能作为理解行业边界的背景条件。"
    claim_status = _claim_status_from_support(support, proof_gaps)
    if proof_gaps and claim_status == "decision_ready":
        claim_status = "directional"
    return {
        "agent": AGENT_NAME,
        "chapter_id": package.get("chapter_id"),
        "section_id": section_id,
        "question": question,
        "section_title": _compact(question, 160),
        "claim": claim,
        "reasoning": reasoning,
        "mechanism": reasoning,
        # Counter & actionable: route through _ensure_* helpers so they honour
        # REPORT_TEMPLATE_FALLBACKS. By default these are empty strings rather
        # than the historic boilerplate ("反向样本、可比口径和时间窗口会改变结论
        # 方向..." / "后续重点跟踪同口径指标..."), which previously appeared in
        # every section of every chapter.
        "counter_evidence": _ensure_counter(_first_counter_from_package(package)),
        "actionable": _ensure_actionable(_first_action_from_package(package)),
        "confidence": "medium" if fact and strength in {"strong", "medium"} and support["grade"] not in {"low", "medium_low"} else "low",
        "confidence_reason": "由本章证据数量、来源层级和反向样本覆盖程度决定。",
        "claim_status": claim_status,
        "supporting_evidence": refs,
        "evidence_refs": refs,
        "supporting_facts": supporting_facts,
        "verification_metrics": _public_verification_focus(package, refs),
        "what_to_verify_next": _public_verification_focus(package, refs),
        "proof_gaps": _as_list(package.get("missing_proof_standards")),
        "follow_up_queries": _proof_followups(package),
        "source_quality": support,
        "quality_status": "directional_with_boundary" if proof_gaps else "valid",
        "omit_from_report": claim_status == "appendix_only",
        "public_render": claim_status != "appendix_only",
        "internal_reason": "missing_proof_bundle" if proof_gaps else "",
    }


def _evidence_refs_from_package(package: Dict[str, Any], *, limit: int = 12) -> List[str]:
    return _dedupe(
        [
            _citation_ref_from_evidence(item)
            for collection in PUBLIC_EVIDENCE_COLLECTIONS
            for item in _as_list(package.get(collection))
            if isinstance(item, dict)
        ],
        limit=limit,
    )


def _evidence_facts_from_package(package: Dict[str, Any], *, limit: int = 5) -> List[str]:
    """Collect a deduped list of public-ready facts from one chapter package.

    Fix history:
    - Removed `f"{metric}: {value}，{fact}"` field concatenation that produced
      "市场规模: 12534亿元，市场规模: 12534亿元" style叠词 in the rendered report.
    - Added cross-collection dedup keyed by evidence_id / fact text so the same
      fact appearing in both core_evidence and supporting_evidence only renders once.
    """
    limit = _env_int("REPORT_FACTS_PER_ARGUMENT_UNIT", limit, min_value=3, max_value=40)
    facts: List[str] = []
    seen_keys: set[str] = set()

    def _fact_key(item: Dict[str, Any], fact_text: str) -> str:
        eid = str(item.get("evidence_id") or item.get("id") or "").strip()
        if eid:
            return f"eid::{eid}"
        # fallback: hash a compacted version of fact text so near-duplicates collapse
        normalized = re.sub(r"\s+", "", fact_text)[:160].lower()
        return f"text::{normalized}"

    for collection in PUBLIC_EVIDENCE_COLLECTIONS:
        for item in _as_list(package.get(collection)):
            if not isinstance(item, dict):
                continue
            if item.get("metric_validation_status") == "invalid":
                continue
            fact = _compact(item.get("fact") or item.get("clean_fact") or item.get("content") or item.get("summary"), 360)
            if _is_bad_public_fact(fact):
                continue
            metric = _compact(item.get("metric") or item.get("indicator"), 80)
            value = _compact(item.get("value") or item.get("display_value"), 80)
            # Prefer a self-contained natural-language fact; only fall back to
            # "metric: value" when no narrative fact is available. We never
            # concatenate metric+value+fact because the fact text usually already
            # contains the metric and value in natural form.
            if fact:
                rendered = fact
            elif metric and value:
                rendered = f"{metric}: {value}"
            else:
                continue
            key = _fact_key(item, rendered)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            facts.append(rendered)
            if len(facts) >= limit:
                return facts
    return facts


def _source_profile_text(package: Dict[str, Any]) -> str:
    return ""


def _package_topic_text(package: Dict[str, Any]) -> str:
    parts = [
        package.get("chapter_title"),
        package.get("chapter_question"),
        package.get("lead"),
        package.get("query"),
        _as_dict(package.get("metadata")).get("query"),
    ]
    return " ".join(str(part or "") for part in parts)


def _looks_like_semiconductor_topic(package: Dict[str, Any]) -> bool:
    text = _package_topic_text(package)
    return bool(
        re.search(r"半导体|芯片|集成电路|晶圆|光刻|EDA|封测|Chiplet|先进制程|成熟制程|科技博弈", text)
        and re.search(r"中美|美国|中国|全球|供应链|管制|制裁|国产|机遇|挑战|重构", text)
    )


def _long_reasoning(package: Dict[str, Any], *, lens: str, facts: Sequence[str]) -> str:
    title = _compact(package.get("chapter_title"), 120) or "本章问题"
    question = _compact(package.get("chapter_question"), 180) or title
    fact_limit = _env_int("REPORT_FACTS_PER_REASONING", 6, min_value=3, max_value=12)
    fact_text = _compact(
        "；".join(_dedupe(facts, limit=fact_limit)),
        _env_int("REPORT_REASONING_FACT_TEXT_MAX_CHARS", 900, min_value=300, max_value=2400),
    ) or "当前材料中的价格、供给、需求、订单、政策和反向样本需要放在同一条判断链上理解"
    profile = _source_profile_text(package)
    if _looks_like_semiconductor_topic(package):
        if lens == "mechanism":
            return (
                f"“{question}”不是普通景气周期题，而是出口管制、产业补贴、关键设备可得性、产能迁移和客户安全边界共同作用的结构性问题。"
                f"材料中最有解释力的事实组合是：{fact_text}。这些事实需要按供应链层级拆开理解：上游设备、EDA和材料决定先进制程上限；中游晶圆制造、封测和成熟制程决定短中期产能承接；下游AI、汽车、工业和消费电子客户决定国产替代能否变成真实订单。"
                f"{profile}只有政策限制、企业资本开支、客户导入和反向样本在同一方向上互相印证时，章节结论才适合上升为全篇主线。"
            )
        if lens == "boundary":
            return (
                f"这个判断的边界来自三组变量。第一是政策边界：出口管制、豁免、许可和盟友协同会直接改变设备、软件和高端芯片的可得性。第二是产业边界：成熟制程扩产、先进封装、国产设备验证和客户认证如果不能同步推进，局部突破很难转化为系统能力。第三是周期边界：资本开支过快、库存回升或价格下行，会削弱国产替代的利润质量。"
                f"围绕“{title}”，需要同时看正向材料和反向材料：{fact_text}。如果反向样本显示产能利用率下降、海外限制升级或客户验证停滞，结论应从确定性机会收缩为阶段性观察。"
                f"{profile}因此，本章重点不是简单判断乐观或悲观，而是说明机会能够兑现的条件、短板仍会卡住的位置，以及哪些信号会推翻原有判断。"
            )
        if lens == "decision":
            return (
                f"“{title}”的决策价值在于把供应链重构拆成可排序的产业机会。更靠近国产替代刚需、客户验证清晰、受出口管制冲击较小、财务质量可持续的环节，优先级应高于只停留在概念叙事的环节。"
                f"可以放在一起观察的事实包括：{fact_text}。它们分别回答哪些节点被外部限制重塑、哪些中国环节能承接订单、哪些短板仍依赖海外生态、以及扩产是否会带来过剩和盈利压力。"
                f"{profile}后续跟踪应集中在管制清单变化、晶圆厂资本开支、设备材料国产验证、封测和成熟制程订单、车规/工业客户导入，以及海外客户对中国供应链的信任变化。"
            )
        return (
            f"围绕“{question}”，讨论应从供应链节点开始，再转入技术约束、政策压力、客户导入和反向情形。当前事实组合是：{fact_text}。"
            f"{profile}这些事实如果能同时覆盖政策、技术、产能、订单和反证，结论可以更强；如果只覆盖单个企业或单条新闻，结论应保留为方向性判断。"
        )
    if lens == "mechanism":
        return (
            f"“{question}”不是单点事实题，而是变量之间能否形成同向传导的问题。上游约束决定供给弹性，中游价格和库存决定景气位置，下游订单、开工或采购行为决定需求是否真实兑现。"
            f"材料中最有解释力的事实组合是：{fact_text}。这些事实不能被简单相加，真正影响结论的是它们是否指向同一方向：价格修复伴随库存下降和订单确认时，信号强度会上升；价格变化只是短期扰动，而库存、产能利用率或客户预算没有跟上时，结论只能收缩到阶段性观察。"
            f"{profile}同一指标在不同来源中的口径、时间窗口、企业端披露和行业端统计，会共同改变最终结论的力度。"
        )
    if lens == "boundary":
        return (
            f"这个判断的边界主要来自三类变量。第一是时间边界：短期价格、新闻或订单信号可能领先于真实需求，也可能只是库存周期中的反弹。第二是口径边界：统计口径、企业披露口径和第三方研究口径如果覆盖范围不同，直接比较会放大或低估变化幅度。第三是反向样本：只要出现价格继续下行、库存重新累积、项目延期、客户认证慢于预期或政策执行弱于预期，原有判断就要降级。"
            f"围绕“{title}”，正向材料和反向材料需要放在同一个框架里观察：{fact_text}。正向材料只说明局部样本，而反向材料覆盖更广的行业或更近的时间窗口时，结论会转向保守口径；反向材料只是个别噪声，而核心指标持续改善时，结论强度才会上调。"
            f"{profile}因此，这一部分的重点不是单向乐观或悲观，而是结论成立的条件和失效的信号。"
        )
    if lens == "decision":
        return (
            f"“{title}”的价值不在于复述资料，而在于把资料转化为资源配置的优先顺序。事实链显示需求真实、供给约束可解释、价格或利润有持续性时，资源会更多流向客户、订单、产能和盈利弹性；事实链只停留在概念热度或单点新闻时，它更适合留在观察层。"
            f"可以放在一起观察的事实包括：{fact_text}。这些事实分别回答了有没有需求、谁在付款或采购、供给是否紧张、利润是否能留下来、反向风险是否足够强。只有这些问题形成一致方向，章节结论才会进入全篇主线。"
            f"{profile}后续跟踪的重点落在同口径高频指标、企业端订单或客户行为，以及能够推翻原有结论的反向样本。"
        )
    return (
        f"围绕“{question}”，讨论从事实组合开始，再转入成立条件和相反情形。当前事实组合是：{fact_text}。"
        f"{profile}这些事实来自不同类型来源且方向一致时，可以支撑较强结论；来源集中、口径不一致或缺少反向样本时，结论会保留边界。"
    )


def _deep_unit_from_package(
    package: Dict[str, Any],
    *,
    section_id: str,
    lens: str,
    title: str,
    facts: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    refs = _evidence_refs_from_package(package, limit=12)
    facts = list(facts or _evidence_facts_from_package(package, limit=8))
    support = _support_profile(package, refs)
    chapter_title = _compact(package.get("chapter_title"), 120) or "本章"
    fact_text = _compact("；".join(_dedupe(facts, limit=6)), 720)
    if not refs or not fact_text:
        return {
            "agent": AGENT_NAME,
            "chapter_id": package.get("chapter_id"),
            "section_id": section_id,
            "question": package.get("chapter_question") or package.get("chapter_title"),
            "section_title": title,
            "claim": f"{chapter_title}暂不适合作为独立结论展开。",
            "reasoning": "已披露信息还不能把变量关系讲完整，正文应避免把零散线索写成行业主线。",
            "mechanism": "",
            "counter_evidence": "如果后续出现更清晰的同口径指标或反向样本，本章判断需要重新校准。",
            "actionable": "可先保留为观察项，待关键指标连续披露后再进入正文主线。",
            "decision_implication": "可先保留为观察项，待关键指标连续披露后再进入正文主线。",
            "confidence": "low",
            "claim_status": "appendix_only",
            "supporting_evidence": refs,
            "evidence_refs": refs,
            "supporting_facts": list(facts),
            "verification_metrics": _public_verification_focus(package, refs),
            "what_to_verify_next": _public_verification_focus(package, refs),
            "source_quality": support,
            "quality_status": "appendix_only",
            "omit_from_report": True,
            "public_render": False,
            "internal_reason": "missing_refs_or_fact_text",
        }
    semiconductor_topic = _looks_like_semiconductor_topic(package)
    if lens == "mechanism" and semiconductor_topic:
        claim = f"{chapter_title}应放在“管制强度、技术可得性、产能位置、客户导入”四层关系中判断，而不是按单个新闻或单个企业外推。"
        counter = "如果出口管制边际放松、关键设备验证失败、成熟制程产能利用率下行，或客户认证慢于预期，供应链重构的方向和节奏都需要重新校准。"
        action = "后续跟踪集中在管制清单、设备材料国产验证、晶圆厂资本开支、封测和成熟制程订单，以及海外客户对中国供应链的采购态度。"
    elif lens == "mechanism":
        claim = f"{chapter_title}更适合放在“供给约束、需求兑现、价格利润、反向样本”四层关系中观察，而不是按单个事实外推。"
        counter = "这些变量在时间窗口或统计口径上不能对齐时，结论只保留方向性；更高等级来源给出相反数据时，原有传导关系会被重新校准。"
        action = "后续跟踪集中在能够同时解释价格、库存、订单和利润的指标组合，并把企业披露与行业统计放在同一口径下比较。"
    elif lens == "boundary" and semiconductor_topic:
        claim = f"{chapter_title}的结论强度取决于政策限制、技术短板、产能周期和客户信任是否同时改善。"
        counter = "边界条件包括管制升级、设备维护和交付受限、先进制程良率不达预期、成熟制程价格下行、库存回升、海外客户导入放缓。"
        action = "反向触发器出现时，结论应从确定性机会收缩为阶段性观察，并重新区分先进制程短板、成熟制程机会和封测/设计生态的可兑现程度。"
    elif lens == "boundary":
        claim = f"{chapter_title}的结论强度取决于反向样本是否足以改变主线。"
        counter = "边界条件包括价格继续走弱、库存回升、需求端项目延后、客户认证或采购节奏低于预期，以及政策或监管执行出现偏差。"
        action = "反向触发器出现时，结论会从强判断收缩为观察判断，并重新校准章节主线。"
    elif semiconductor_topic:
        claim = f"{chapter_title}的战略含义取决于哪些环节能把供应链安全需求转化为真实订单、技术验证和可持续盈利。"
        counter = "如果核心事实只说明政策支持或话题热度，却缺少客户导入、量产验证、产能利用率和财务质量支撑，就不宜直接推导为长期机会。"
        action = "优先观察国产设备材料验证、成熟制程和封测订单、车规/工业客户认证、先进封装生态，以及资本开支和产能利用率能否匹配需求。"
    else:
        claim = f"{chapter_title}最终会影响资源配置和研究优先级：多来源持续验证的环节权重更高，只具备话题热度的线索权重更低。"
        counter = "如果核心事实只能说明局部样本，或缺少客户、订单、价格、利润中的任一关键环节，就不宜直接推导为总体机会。"
        action = "付款主体、订单持续性、利润留存能力和反向样本强度，决定它能否进入投资、采购、产品立项或继续研究。"
    return {
        "agent": AGENT_NAME,
        "chapter_id": package.get("chapter_id"),
        "section_id": section_id,
        "question": package.get("chapter_question") or package.get("chapter_title"),
        "section_title": title,
        "claim": claim,
        "reasoning": _long_reasoning(package, lens=lens, facts=facts),
        "mechanism": _long_reasoning(package, lens="mechanism", facts=facts[:4]),
        "counter_evidence": counter,
        "actionable": action,
        "decision_implication": action,
        "confidence": "medium" if support.get("grade") in {"high", "medium"} else "low",
        "claim_status": "directional" if support.get("grade") in {"low", "medium_low"} else "decision_ready",
        "supporting_evidence": refs,
        "evidence_refs": refs,
        "supporting_facts": list(facts),
        "verification_metrics": _public_verification_focus(package, refs),
        "what_to_verify_next": _public_verification_focus(package, refs),
        "source_quality": support,
        "quality_status": "valid",
        "omit_from_report": False,
        "public_render": bool(refs),
    }


def _deep_units_from_package(package: Dict[str, Any], base_section_id: str) -> List[Dict[str, Any]]:
    chapter_id = str(package.get("chapter_id") or "chapter")
    semiconductor_topic = _looks_like_semiconductor_topic(package)
    mechanism_title = "供应链重构机制与变量联动" if semiconductor_topic else "机制拆解与变量联动"
    boundary_title = "反证、边界与结论失效条件"
    decision_title = "产业战略含义与后续观察优先级" if semiconductor_topic else "决策含义与后续观察优先级"
    verification_title = "哪些政策和产业信号值得继续验证" if semiconductor_topic else "哪些信号值得继续验证"
    economics_title = "受益环节、约束条件与兑现路径" if semiconductor_topic else "收益归属和商业化弹性在哪里"
    facts = _evidence_facts_from_package(
        package,
        limit=_env_int("REPORT_FACTS_PER_CHAPTER_ARGUMENTS", 18, min_value=6, max_value=48),
    )
    mechanism_facts = facts[:8] or facts
    boundary_facts = facts[4:12] or facts[:8] or facts
    decision_facts = facts[8:16] or facts[2:10] or facts
    units = [
        _unit_from_evidence(package, base_section_id),
        _deep_unit_from_package(package, section_id=f"{chapter_id}_mechanism", lens="mechanism", title=mechanism_title, facts=mechanism_facts),
        _deep_unit_from_package(package, section_id=f"{chapter_id}_boundary", lens="boundary", title=boundary_title, facts=boundary_facts),
        _deep_unit_from_package(package, section_id=f"{chapter_id}_decision", lens="decision", title=decision_title, facts=decision_facts),
    ]
    if _env_int("REPORT_ARGUMENT_UNITS_PER_CHAPTER", 5, min_value=4, max_value=6) >= 5:
        verification_facts = facts[12:20] or facts[6:14] or facts
        units.append(
            _deep_unit_from_package(
                package,
                section_id=f"{chapter_id}_verification",
                lens="verification",
                title=verification_title,
                facts=verification_facts,
            )
        )
    if _env_int("REPORT_ARGUMENT_UNITS_PER_CHAPTER", 5, min_value=4, max_value=6) >= 6:
        economics_facts = facts[16:24] or facts[8:16] or facts
        units.append(
            _deep_unit_from_package(
                package,
                section_id=f"{chapter_id}_economics",
                lens="economics",
                title=economics_title,
                facts=economics_facts,
            )
        )
    return units


def _lens_for_layout_section(section: Dict[str, Any]) -> str:
    block_type = str(section.get("block_type") or section.get("output_type") or "").strip()
    if block_type in {"policy_timeline", "mechanism_chain", "value_chain_map", "stakeholder_map", "technology_maturity"}:
        return "mechanism"
    if block_type in {"risk_trigger", "metric_reconciliation"}:
        return "boundary"
    if block_type in {"scenario_analysis", "verification_checklist"}:
        return "decision"
    if block_type in {"unit_economics", "customer_painpoint_matrix", "case_comparison", "competitive_positioning"}:
        return "economics"
    return "verification"


def _units_from_layout_sections(package: Dict[str, Any], sections: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    facts = _evidence_facts_from_package(
        package,
        limit=_env_int("REPORT_FACTS_PER_CHAPTER_ARGUMENTS", 18, min_value=6, max_value=48),
    )
    units: List[Dict[str, Any]] = []
    for index, section in enumerate([item for item in sections if isinstance(item, dict)], start=1):
        section_id = str(section.get("section_id") or f"{package.get('chapter_id')}_s{index}")
        title = _compact(section.get("section_title") or section.get("title") or package.get("chapter_question"), 160)
        required_refs = _as_list(section.get("required_evidence_refs"))
        if str(section.get("section_role") or "").strip() == "evidence_gap" and not required_refs:
            continue
        if index == 1 and str(section.get("block_type") or section.get("output_type") or "") == "thesis":
            unit = _unit_from_evidence(package, section_id)
            unit["section_title"] = title or unit.get("section_title")
        else:
            start = max(0, (index - 1) * 4)
            unit_facts = facts[start : start + 8] or facts
            unit = _deep_unit_from_package(
                package,
                section_id=section_id,
                lens=_lens_for_layout_section(section),
                title=title or "章节论证",
                facts=unit_facts,
            )
        if required_refs:
            mapped_refs = _source_refs_for_evidence_refs(package, required_refs) or _dedupe(required_refs, limit=8)
            unit["evidence_refs"] = mapped_refs
            unit["supporting_evidence"] = mapped_refs
            unit["source_quality"] = _support_profile(package, unit["evidence_refs"])
        unit["block_type"] = section.get("block_type") or section.get("output_type")
        unit["layout_section_role"] = section.get("section_role")
        units.append(unit)
    return units


def _argument_unit_issues(unit: Dict[str, Any]) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    claim = str(unit.get("claim") or "").strip()
    reasoning = str(unit.get("reasoning") or "").strip()
    counter = str(unit.get("counter_evidence") or "").strip()
    actionable = str(unit.get("actionable") or "").strip()
    refs = _as_list(unit.get("evidence_refs"))
    if not claim:
        issues.append({"type": "missing_claim"})
    elif claim.startswith(WEAK_CLAIM_PREFIXES) or _has_bad_pattern(claim):
        issues.append({"type": "weak_claim"})
    if not reasoning:
        issues.append({"type": "missing_reasoning"})
    elif _has_bad_pattern(reasoning):
        issues.append({"type": "weak_reasoning"})
    elif not any(word in reasoning for word in CAUSE_WORDS):
        issues.append({"type": "reasoning_missing_causal_chain", "severity": "warning"})
    if not counter:
        issues.append({"type": "missing_counter_evidence"})
    elif _has_bad_pattern(counter):
        issues.append({"type": "weak_counter_evidence"})
    if not actionable:
        issues.append({"type": "missing_actionable"})
    elif _has_bad_pattern(actionable):
        issues.append({"type": "weak_actionable"})
    elif not any(word in actionable for word in ACTION_WORDS):
        issues.append({"type": "actionable_missing_action_word"})
    if not refs:
        issues.append({"type": "missing_evidence_refs"})
    return issues


def rewrite_weak_claim_unit(unit: Dict[str, Any], *, package: Optional[Dict[str, Any]] = None, reason: str = "weak_claim") -> Dict[str, Any]:
    package = _as_dict(package)
    rewritten = dict(unit)
    fallback_fact = _compact(unit.get("fact") or unit.get("supporting_fact") or "", 180)
    question = _question_for(package, unit) if package else _compact(unit.get("question") or unit.get("section_title"), 180)
    title = _compact(package.get("chapter_title") if package else unit.get("section_title"), 80) or "本节"
    refs = _as_list(unit.get("evidence_refs"))
    source_quality = _as_dict(unit.get("source_quality")) or (_support_profile(package, refs) if package else {})
    if reason == "low_source_quality" or source_quality.get("grade") == "low":
        claim = f"{title}需要按连续指标和反向样本拆解，避免把单点信号直接外推为行业结论。"
        fact_clause = (
            f"；已披露的关键事实包括：{_compact(fallback_fact, 120)}"
            if fallback_fact and "？" not in fallback_fact and "?" not in fallback_fact and not _is_bad_public_fact(fallback_fact)
            else ""
        )
        if package and _looks_like_semiconductor_topic(package):
            reasoning = f"这一判断需要落到管制清单、设备材料验证、先进封装进展、成熟制程产能利用率和客户认证节奏上观察{fact_clause}。这些变量若同向改善，机会判断才更稳；若只出现单点进展，结论应停留在阶段性观察。"
            counter = "如果出口管制边际变化、关键设备或 EDA 替代验证失败、成熟制程价格下行，或客户导入慢于预期，本节判断需要收窄。"
            actionable = "后续重点跟踪 BIS/实体清单、CHIPS Act 执行、ASML/EDA/设备材料供应、SMIC/华虹/封测厂资本开支和客户认证进展。"
        else:
            reasoning = f"这一判断需要放在价格、库存、订单和政策执行的连续变化中验证{fact_clause}。如果这些变量持续同向，章节结论可以更明确；如果只是一两条孤立信号，则更适合写成阶段性观察。"
            counter = "样本、时间窗口和来源口径可能改变结论方向；价格、库存、订单或政策执行出现反向变化时，适用范围会收窄。"
            actionable = "后续重点跟踪同口径指标、反向样本和执行进展，再根据连续变化安排投资、采购或产品动作。"
        confidence = "low"
    else:
        claim = _claim_from_fact(package, fallback_fact) if package else f"{title}的结论强度取决于“{question}”能否被连续事实支撑。"
        reasoning = _ensure_reasoning(unit.get("reasoning"), package, fallback_fact) if package else "该结论需要同时满足场景、主体和可验证指标，才能从线索变成可执行结论。"
        counter = _ensure_counter(unit.get("counter_evidence"))
        actionable = _ensure_actionable(unit.get("actionable") or unit.get("decision_implication"))
        confidence = unit.get("confidence") or "medium"
    rewritten.update(
        {
            "claim": claim,
            "reasoning": reasoning,
            "counter_evidence": counter,
            "actionable": actionable,
            "confidence": confidence,
            "confidence_reason": rewritten.get("confidence_reason") or "已按判断-原因-边界-动作结构重写，并保留原 evidence_refs。",
            "rewrite_required": True,
            "rewrite_reason": reason,
            "verification_metrics": _as_list(unit.get("verification_metrics")) or (_verification_metrics(package, refs) if package else []),
            "claim_status": unit.get("claim_status") or ("directional" if confidence == "low" else "decision_ready"),
            "quality_status": unit.get("quality_status") or ("directional_with_boundary" if confidence == "low" else "valid"),
        }
    )
    return rewritten


def validate_argument_units(argument_units: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    invalid = []
    warnings = []
    for index, unit in enumerate(list(argument_units or [])):
        if not isinstance(unit, dict):
            invalid.append({"index": index, "type": "invalid_unit_type"})
            continue
        for issue in _argument_unit_issues(unit):
            payload = {"index": index, **issue}
            if issue.get("severity") == "warning":
                warnings.append(payload)
            else:
                invalid.append(payload)
    return {
        "passed": not invalid,
        "invalid_units": invalid,
        "warnings": warnings,
        "invalid_count": len(invalid),
        "warning_count": len(warnings),
    }


def run_claim_builder_agent(
    *,
    chapter_evidence_packages: Optional[Sequence[Dict[str, Any]]] = None,
    micro_layouts: Optional[Sequence[Dict[str, Any]]] = None,
    structured_analysis: Optional[Dict[str, Any]] = None,
    llm_client: Any = None,
) -> List[Dict[str, Any]]:
    del llm_client
    structured_analysis = _as_dict(structured_analysis)
    structured_units = _structured_units(structured_analysis)
    layout_by_id = {
        str(layout.get("chapter_id") or ""): layout
        for layout in list(micro_layouts or [])
        if isinstance(layout, dict)
    }
    argument_units: List[Dict[str, Any]] = []
    package_by_id: Dict[str, Dict[str, Any]] = {}
    for package in list(chapter_evidence_packages or []):
        if not isinstance(package, dict):
            continue
        chapter_id = str(package.get("chapter_id") or "")
        package_by_id[chapter_id] = package
        layout = _as_dict(layout_by_id.get(chapter_id))
        sections = [section for section in _as_list(layout.get("sections")) if isinstance(section, dict)]
        section_id = str(_as_dict(sections[0] if sections else {}).get("section_id") or f"{chapter_id}_s1")
        fallback_refs = _dedupe(
            [
                _citation_ref_from_evidence(item)
                for collection in ("core_evidence", "supporting_evidence", "sample_evidence", "clue_evidence")
                for item in _as_list(package.get(collection))
                if isinstance(item, dict)
            ],
            limit=8,
        )
        matched = [unit for unit in structured_units if _matches(unit, package)]
        if matched:
            for index, unit in enumerate(matched[:4], start=1):
                argument_units.append(_unit_from_structured(unit, package, section_id if index == 1 else f"{chapter_id}_s{index}", fallback_refs))
            if len(matched) < 3:
                layout_units = _units_from_layout_sections(package, sections[1:])
                argument_units.extend(layout_units or _deep_units_from_package(package, f"{chapter_id}_deep")[1:])
        else:
            layout_units = _units_from_layout_sections(package, sections)
            argument_units.extend(layout_units or _deep_units_from_package(package, section_id))
    cleaned: List[Dict[str, Any]] = []
    for unit in argument_units:
        package = _as_dict(package_by_id.get(str(unit.get("chapter_id") or "")))
        was_rewritten = bool(unit.get("rewrite_required"))
        original_issues = _argument_unit_issues(unit)
        if any(issue.get("severity") != "warning" for issue in original_issues):
            unit = rewrite_weak_claim_unit(unit, package=package, reason="contract_rewrite")
        final_issues = _argument_unit_issues(unit)
        blocking = [issue for issue in final_issues if issue.get("severity") != "warning"]
        unit["original_quality_issues"] = original_issues
        unit["quality_issues"] = final_issues
        existing_quality_status = str(unit.get("quality_status") or "")
        if blocking:
            unit["quality_status"] = "invalid"
        elif existing_quality_status in {"directional_with_boundary", "context_only"}:
            unit["quality_status"] = existing_quality_status
        else:
            unit["quality_status"] = "valid"
        unit["rewrite_required"] = was_rewritten or bool(original_issues)
        if not is_public_claim(unit):
            source_quality = _as_dict(unit.get("source_quality"))
            strength = str(source_quality.get("claim_strength") or "unsupported").lower()
            unit["omit_from_report"] = True
            unit["public_render"] = False
            unit["claim_status"] = unit.get("claim_status") or "appendix_only"
            unit["quality_status"] = "appendix_only" if unit["quality_status"] == "valid" else unit["quality_status"]
            unit["internal_reason"] = (
                "no_core_or_supporting_evidence"
                if strength in {"weak", "unsupported"}
                else "public_blocking_language_or_missing_refs"
            )
            unit.setdefault("follow_up_queries", _as_list(package.get("follow_up_queries")))
        else:
            unit["omit_from_report"] = False
            unit["public_render"] = True
        cleaned.append(unit)
    return cleaned
