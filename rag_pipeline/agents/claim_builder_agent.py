from __future__ import annotations

import ast
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    from rag_pipeline.contracts.evidence_quality import infer_claim_type
except Exception:  # pragma: no cover - direct script mode fallback
    infer_claim_type = None  # type: ignore

try:
    from rag_pipeline.agents.block_schema import can_render_block_from_evidence
except Exception:  # pragma: no cover - direct script mode fallback
    can_render_block_from_evidence = None  # type: ignore

try:
    from rag_pipeline.agents.layout_claim_matcher import claim_supported_block_types
except Exception:  # pragma: no cover - direct script mode fallback
    claim_supported_block_types = None  # type: ignore


AGENT_NAME = "claim_builder_agent"
AGENT_DESCRIPTION = "Claim Builder Agent. Converts chapter evidence into claim/argument units."

ACTION_WORDS = ("优先", "避免", "验证", "跟踪", "排除", "补充", "设置")
WEAK_CLAIM_PREFIXES = ("已有可验证", "已有可核验", "已有可用证据", "当前证据")
BAD_CLAIM_PATTERNS = [
    r"已有可核验证据",
    r"已有可验证证据",
    r"证据不足",
    r"正文\s*只能\s*写成",
    r"本章\s*只能\s*写成",
    r"本章\s*可\s*写成",
    r"本章\s*仍需\s*连续观察",
    r"建议避免",
    r"建议在后续版本中补充",
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
    r"正文\s*只能\s*写成",
    r"本章\s*只能\s*写成",
    r"本章\s*可\s*写成",
    r"本章\s*仍需\s*连续观察",
    r"建议避免",
    r"建议在后续版本中补充",
    r"暂无可核验",
    r"建议补充",
    r"A/B\s*级来源不足",
    r"权威来源交叉验证",
    r"needs_corroboration",
    r"insufficient",
    r"unsupported",
]
BAD_FACT_PATTERNS = [
    r"^\s*-?\d{2,6}(?:\.\d+)?\s*(?:$|[;,\.\u3002\uff1b\uff0c])",
    r"^\s*(?:fact|key fact|metric|source_check|status|policy target|competitive comparison|cost)\s*[:\uff1a]\s*-?\d{1,6}(?:\.\d+)?\b",
    r"^\s*(?:\u4e8b\u5b9e|\u5173\u952e\u4e8b\u5b9e|\u7ade\u4e89\u5bf9\u6bd4|\u653f\u7b56\u76ee\u6807|\u653f\u7b56\u76d1\u7ba1|\u6210\u672c)\s*[:\uff1a]\s*-?\d{1,6}(?:\.\d+)?\b",
    r"^\s*(?:\u5185\u5bb9\u8bf4\u660e|\u65f6\u95f4)\s*[:\uff1a]",
    r"联网分析\s*Agent\s*失败",
    r"Retrieval\.TestUserQueryExceeded",
    r"query\s+exceed(?:ed)?\s+the\s+limit",
    r"目前更像局部信号",
    r"已通过\s*IQS\s*获取到联网证据",
    r"当前未启用或未成功调用大模型综合分析",
    r"先给出可核验的网页结果摘要",
    r"关键依据[:：]\s*\d+\.",
    r"Skip to (?:content|main content)",
    r"picture intentionally omitted",
    r"Over the weekend",
    r"Futian district government",
    r"首页问\s*·\s*答|热搜公司|热搜词|登录注册",
    r"Caret right|View all products|Product\s+Documentation",
    r"^\s*(?:摘要|标题|来源)[:：]\s*$",
    r"\*\*==>\s*picture intentionally omitted\s*<==\*\*",
    r"登录\s+首页|上一篇|下一篇|分享到|AI帮你提炼|智能挖掘|智享会员|会员积分",
    r"^(?:事实|竞争对比|关键事实|政策目标)\s*[:：]\s*-?\d{1,3}(?:\.\d+)?\b",
    r"^\s*-?\d{1,3}(?:\.\d+)?\s*[;；，,]",
    r"以下是对整篇.*(?:深度分析|框架提炼)",
    r"问答[:：]?首页问|答云访谈|综合资讯投票",
]

EVIDENCE_COLLECTIONS = (
    "core_evidence",
    "supporting_evidence",
    "metric_evidence",
    "case_evidence",
    "counter_evidence",
    "directional_evidence",
    "sample_evidence",
    "table_evidence",
    "clue_evidence",
    "appendix_evidence",
    "evidence_items",
)

PUBLIC_EVIDENCE_COLLECTIONS = (
    "core_evidence",
    "supporting_evidence",
    "metric_evidence",
    "case_evidence",
    "counter_evidence",
    "directional_evidence",
    "sample_evidence",
    "table_evidence",
    "clue_evidence",
)

# Public-output guards added as unicode escapes so this file stays robust even
# when edited from Windows terminals with different code pages.
_PUBLIC_TEMPLATE_PATTERNS = [
    r"\u4e0d\u662f\u5355\u70b9\u4e8b\u5b9e\u9898",  # 不是单点事实题
    r"\u4f9b\u7ed9\u7ea6\u675f",
    r"\u9700\u6c42\u5151\u73b0",
    r"\u4ef7\u683c\u5229\u6da6",
    r"\u4ef7\u683c\u4fee\u590d",
    r"\u5e93\u5b58\u4e0b\u964d",
    r"\u8ba2\u5355\u786e\u8ba4",
    r"\u540e\u7eed\u91cd\u70b9\u89c2\u5bdf\u80fd\u591f\u540c\u65f6\u89e3\u91ca",
    r"\u6b63\u6587\s*\u53ea\u80fd\s*\u5199\u6210",
    r"\u6b63\u6587\s*\u5e94\s*\u628a",
    r"\u6b63\u6587\s*\u5e94\u5f53",
    r"\u672c\u7ae0\s*\u53ea\u80fd\s*\u5199\u6210",
    r"\u672c\u7ae0\s*\u53ef\s*\u5199\u6210",
    r"\u5efa\u8bae\u8865\u8bc1",
    r"\u5efa\u8bae\u907f\u514d",
    r"\u53ef\u590d\u6838\u6765\u6e90\u8d8a\u72ec\u7acb",
    r"\u76ee\u524d\u7ed3\u8bba\u4ecd\u53d7",
    r"\u4f18\u5148\u590d\u6838\u53ef\u8ffd\u6eaf\u6765\u6e90",
    r"\u672c\u7ae0\u5173\u6ce8",
    r"\u672c\u8282\u56f4\u7ed5",
    r"claim_status",
    r"evidence_cards",
]
BAD_CLAIM_PATTERNS.extend(_PUBLIC_TEMPLATE_PATTERNS)
PUBLIC_BLOCKING_PATTERNS.extend(_PUBLIC_TEMPLATE_PATTERNS)
BAD_FACT_PATTERNS.extend(
    [
        r"\u519c\u4e1a\u4eba\u5de5\u667a\u80fd",
        r"\u6570\u636e\u6295\u6bd2",
        r"Scribd",
        r"\u53d1\u73b0\u62a5\u544a",
        r"\u7eba\u7ec7",
        r"\u667a\u80fd\u624b\u673a",
        r"SEO",
        r"\u641c\u7d22\u7ed3\u679c",
        r"Read\s*page",
        r"\u7b2c\s*\d+\s*\u8f6e",
        r"picture\s*\[\d+\s*x\s*\d+\]\s*intentionally\s*omitted",
        r"\u8d2d\u7269\u8f66|\u6211\u7684\u8ba2\u5355|\u514d\u8d39\u6ce8\u518c|\u62a5\u544a\u670d\u52a1\u70ed\u7ebf",
        r"URL[:\uff1a]",
        r"(?:\u6210\u672c|\u5173\u952e\u4e8b\u5b9e|\u653f\u7b56\u76d1\u7ba1|\u653f\u7b56\u76ee\u6807)\s*[:\uff1a]\s*-?\d{1,3}(?:\.\d+)?%?",
    ]
)


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _parse_structured_string(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "[{":
        return value
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except Exception:
            continue
        if isinstance(parsed, (dict, list)):
            return parsed
    return value


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _analysis_quality_requires_evidence_rebuild(structured_analysis: Dict[str, Any]) -> bool:
    """Return True when claim_builder should switch into strict mode.

    Prefer the canonical hint produced by `analysis_agent.ensure_valid_structured_analysis`
    (`analysis_contract_status.should_force_strict_claim_building`). Falling back to the
    legacy local heuristic keeps behaviour stable for older payloads that
    don't yet include the field.
    """

    contract_status = _as_dict(structured_analysis.get("analysis_contract_status"))
    canonical = contract_status.get("should_force_strict_claim_building")
    if isinstance(canonical, bool):
        return canonical
    quality = _as_dict(structured_analysis.get("analysis_depth_quality"))
    status = str(quality.get("status") or "").strip().lower()
    repeated_ratio = _safe_float(quality.get("repeated_claim_ratio"), 0.0)
    title_as_claim_count = int(_safe_float(quality.get("title_as_claim_count"), 0.0))
    ref_mismatch_count = int(_safe_float(quality.get("evidence_ref_mismatch_count"), 0.0))
    return bool(
        status == "needs_rewrite"
        or repeated_ratio > 0.50
        or title_as_claim_count > 0
        or ref_mismatch_count > 0
    )


def _strict_quality_mode() -> bool:
    mode = str(os.getenv("REPORT_QUALITY_MODE") or os.getenv("QUALITY_MODE") or "balanced").strip().lower()
    if mode in {"strict", "hard", "deep_strict", "due_diligence", "investment_due_diligence"}:
        return True
    return str(os.getenv("STRICT_EVIDENCE_MODE") or "").strip().lower() in {"1", "true", "yes", "on", "strict"}


def _claim_type_for_unit(unit: Dict[str, Any]) -> str:
    explicit = str(unit.get("claim_type") or unit.get("conclusion_type") or "").strip().lower()
    if explicit:
        return explicit
    if infer_claim_type is not None:
        return str(infer_claim_type(unit) or "industry_analysis")
    return "industry_analysis"


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


def _clean_public_text(value: Any, max_chars: int = 900) -> str:
    text = _compact(value, max_chars)
    if not text:
        return ""
    for pattern in _PUBLIC_TEMPLATE_PATTERNS:
        text = re.sub(rf"[^。\n]*{pattern}[^。\n]*(?:。|$)", "", text, flags=re.I)
    if any(re.search(pattern, text, flags=re.I) for pattern in BAD_FACT_PATTERNS):
        return ""
    return re.sub(r"\s{2,}", " ", text).strip()


def _clean_argument_unit_public_fields(unit: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = dict(unit)
    text_fields = (
        "claim",
        "public_claim",
        "evidence_basis",
        "reasoning",
        "reasoning_chain",
        "mechanism",
        "counter_evidence",
        "limitation_boundary",
        "actionable",
        "decision_implication",
        "confidence_reason",
    )
    for key in text_fields:
        if key in cleaned:
            cleaned[key] = _clean_public_text(cleaned.get(key), 900)
    facts = [
        _clean_public_text(item, 260)
        for item in (_as_list(cleaned.get("supporting_facts")) + _as_list(cleaned.get("fact_chain")))
        if str(item or "").strip() and not _is_bad_public_fact(item)
    ]
    facts = _dedupe([item for item in facts if item], limit=8)
    if facts:
        cleaned["supporting_facts"] = facts
    if not cleaned.get("claim") and facts:
        cleaned["claim"] = facts[0]
        cleaned["public_claim"] = facts[0]
    render_blocks = []
    for block in _as_list(cleaned.get("render_blocks")):
        if not isinstance(block, dict):
            continue
        item = dict(block)
        if "text" in item:
            item["text"] = _clean_public_text(item.get("text"), 900)
            if item.get("type") == "paragraph" and not item["text"]:
                continue
        render_blocks.append(item)
    if render_blocks:
        cleaned["render_blocks"] = render_blocks
    return cleaned


def _invalid_metric_item(item: Dict[str, Any]) -> bool:
    """Reject metric-like fragments that are usually dates, URL ids, or bad parses."""
    if not isinstance(item, dict):
        return True
    if str(item.get("metric_validation_status") or "").strip().lower() == "invalid":
        return True
    metric = str(item.get("metric") or item.get("indicator") or "").strip()
    value = str(item.get("value") or item.get("display_value") or item.get("numeric_value") or "").strip()
    fact = str(item.get("fact") or item.get("clean_fact") or item.get("content") or item.get("summary") or "").strip()
    metric_lower = metric.lower()
    if not metric and not fact:
        return True
    if metric_lower in {"source_check", "status", "http_status", "response_code"} and re.fullmatch(r"[1-5]\d{2}", value):
        return True
    if re.search(r"\bsource_check\s*[:=]\s*[1-5]\d{2}\b", fact, flags=re.I):
        return True
    if value:
        if re.fullmatch(r"-?\d{1,3}(?:\.0)?", value) and re.search(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|T\d{1,2}:\d{2}", fact):
            return True
        if metric in {"\u5173\u952e\u4e8b\u5b9e", "\u653f\u7b56\u76d1\u7ba1", "\u653f\u7b56\u76ee\u6807"} and re.fullmatch(r"-?\d{1,3}(?:\.0)?", value):
            return True
        if re.search(r"\u653f\u7b56|\u76ee\u6807|\u76d1\u7ba1", metric) and re.match(r"-\d", value):
            return True
        if re.search(r"\u6210\u672c", metric) and (re.search(r"\u5bb6$", value) or not fact):
            return True
        if re.search(r"\u5e02\u573a\u89c4\u6a21", metric) and re.search(r"%", value):
            return True
        if re.search(r"\u878d\u8d44", metric) and re.search(r"%", value):
            return True
    if re.search(r"https?://|/\d{4}/\d{1,2}/\d{1,2}/|10462|10778|10876", value):
        return True
    return False


def _clean_public_fact_from_item(item: Dict[str, Any]) -> str:
    quality = _as_dict(item.get("public_fact_quality"))
    if quality and not bool(quality.get("eligible_for_report")):
        return ""
    card = _as_dict(item.get("public_fact_card") or quality.get("public_fact_card"))
    if not card and not quality:
        card = _legacy_fact_card_from_item(item)
    if not card:
        return ""
    card_fact = _compact(card.get("fact") or card.get("object"), 220)
    if card_fact and not _is_bad_public_fact(card_fact):
        return card_fact
    distilled = _compact(item.get("distilled_fact") or quality.get("distilled_fact"), 260)
    if distilled and not _is_bad_public_fact(distilled):
        return distilled
    if _invalid_metric_item(item):
        return ""
    fact = _compact(item.get("fact") or item.get("clean_fact") or item.get("content") or item.get("summary"), 260)
    if _is_bad_public_fact(fact):
        return ""
    metric = _compact(item.get("metric") or item.get("indicator"), 80)
    value = _compact(item.get("value") or item.get("display_value"), 80)
    if fact:
        return fact
    if metric and value:
        return f"{metric}: {value}"
    return ""


def _legacy_fact_card_from_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Hydrate old test/fixture evidence into the new fact-card shape."""
    fact = _compact(item.get("distilled_fact") or item.get("fact") or item.get("clean_fact") or item.get("summary"), 220)
    if not fact or _is_bad_public_fact(fact):
        return {}
    ref = _citation_ref_from_evidence(item) or str(item.get("ref") or item.get("evidence_id") or "").strip()
    if not ref:
        return {}
    role_text = " ".join(str(item.get(key) or "") for key in ("proof_role", "evidence_goal", "allowed_use", "metric", "indicator")).lower()
    if item.get("metric") or item.get("indicator") or "metric" in role_text:
        fact_type = "metric"
    elif re.search(r"counter|risk|风险|反证|失败", role_text, flags=re.I):
        fact_type = "counter"
    elif re.search(r"case|customer|客户|案例|中标|采购|落地", role_text, flags=re.I):
        fact_type = "case"
    elif re.search(r"technology|standard|技术|标准|产品", role_text, flags=re.I):
        fact_type = "technology"
    else:
        fact_type = "directional"
    source_level = str(item.get("source_level") or item.get("credibility") or "C").upper()
    strength = "moderate" if source_level in {"A", "B"} else "directional"
    block_affinity = {
        "metric": ["metric_reconciliation", "unit_economics"],
        "case": ["customer_painpoint_matrix", "case_comparison", "competitive_positioning"],
        "counter": ["risk_trigger", "scenario_analysis", "verification_checklist"],
        "technology": ["technology_maturity"],
    }.get(fact_type, ["thesis", "evidence_matrix"])
    return {
        "subject": _compact(item.get("subject") or item.get("publisher") or "source", 60),
        "action": "shows",
        "object": fact,
        "fact": fact,
        "time_or_scope": "",
        "variable": fact_type,
        "analysis_variable": fact_type,
        "block_affinity": block_affinity,
        "fact_type": fact_type,
        "source_ref": ref,
        "source_level": source_level,
        "claim_strength_hint": strength,
        "directional_only": strength == "directional",
    }


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
    directional_c_source_keys = set()
    matched_collections: Dict[int, str] = {}
    for collection in EVIDENCE_COLLECTIONS:
        for item in _as_list(package.get(collection)):
            if isinstance(item, dict):
                matched_collections[id(item)] = collection
    for item in matched:
        collection_name = matched_collections.get(id(item), "")
        level = str(item.get("source_level") or _as_dict(item.get("source")).get("credibility") or "").strip().upper() or "UNKNOWN"
        levels[level] = levels.get(level, 0) + 1
        role = str(item.get("evidence_role") or item.get("role") or "").strip().lower() or "unknown"
        roles[role] = roles.get(role, 0) + 1
        allowed_use = str(item.get("allowed_use") or _as_dict(item.get("evidence_card")).get("allowed_use") or "").strip() or "unknown"
        if collection_name in {"directional_evidence", "case_evidence"} and allowed_use == "unknown":
            allowed_use = "directional_signal"
        allowed_uses[allowed_use] = allowed_uses.get(allowed_use, 0) + 1
        is_contextual = allowed_use in {"supporting_context", "contextual_support"} or str(item.get("usage_tier") or "") == "corroborated_context"
        if level in {"A", "B"} and is_contextual:
            contextual_ab_count += 1
        elif level in {"A", "B"}:
            claim_ab_count += 1
        elif level in {"B", "C"} and allowed_use == "directional_signal":
            source = _as_dict(item.get("source"))
            key = re.sub(
                r"\s+",
                "",
                str(
                    item.get("source_id")
                    or item.get("source_ref")
                    or item.get("ref")
                    or source.get("url")
                    or source.get("title")
                    or item.get("source_title")
                    or ""
                ).strip().lower(),
            )
            if key:
                directional_c_source_keys.add(key)
    ab_count = levels.get("A", 0) + levels.get("B", 0)
    cd_count = levels.get("C", 0) + levels.get("D", 0)
    core = roles.get("core", 0)
    core_ab = len(
        [
            item
            for item in matched
            if str(item.get("source_level") or _as_dict(item.get("source")).get("credibility") or "").strip().upper() in {"A", "B"}
            and str(item.get("evidence_role") or item.get("role") or "").strip().lower() in {"core", "core_claim"}
        ]
    )
    supporting = roles.get("supporting", 0)
    clue = roles.get("clue", 0)
    directional = allowed_uses.get("directional_signal", 0)
    directional_c_distinct = len(directional_c_source_keys)
    if core_ab >= 1 or claim_ab_count >= 2:
        strength = "strong"
    elif claim_ab_count >= 1:
        strength = "medium"
    elif contextual_ab_count >= 1 or directional >= 1 or supporting >= 1 or directional_c_distinct >= 1:
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
        "core_ab_count": core_ab,
        "claim_ab_count": claim_ab_count,
        "contextual_ab_count": contextual_ab_count,
        "cd_count": cd_count,
        "directional_c_distinct_sources": directional_c_distinct,
        "matched_count": len(matched),
        "role_distribution": roles,
        "allowed_use_distribution": allowed_uses,
        "claim_strength": strength,
    }


def is_public_claim(unit: Dict[str, Any]) -> bool:
    if unit.get("omit_from_report"):
        return False

    if _analysis_claim_is_renderable(unit):
        return True

    claim_status = str(unit.get("claim_status") or "").lower()
    status = str(unit.get("quality_status") or "").lower()
    if status in {"unsupported", "invalid", "weak"}:
        return False
    if status == "insufficient" and claim_status not in {"directional", "directional_ready", "context_only"}:
        return False

    source_quality = _as_dict(unit.get("source_quality"))
    strength = str(source_quality.get("claim_strength") or "").lower()
    if strength in {"weak", "unsupported"} and claim_status not in {"directional", "directional_ready", "context_only"}:
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


def _analysis_claim_is_renderable(unit: Dict[str, Any]) -> bool:
    """Return True for already-admitted analysis claims with traceable support.

    The analysis layer is the single place where claim strength is decided.
    Downstream claim building should only reject hard contract failures (no
    claim, no refs, internal/public-blocking language), not recompute source
    strength from a possibly-empty chapter package and demote the claim to the
    appendix.
    """

    if unit.get("omit_from_report") or unit.get("public_render") is False:
        return False
    refs = _as_list(unit.get("evidence_refs")) or _as_list(unit.get("used_fact_refs"))
    facts = _as_list(unit.get("evidence_basis")) or _as_list(unit.get("supporting_facts"))
    claim = str(unit.get("claim") or unit.get("judgment") or "").strip()
    strength = str(unit.get("claim_strength") or "").strip().lower()
    role = str(unit.get("analysis_role") or "").strip().lower()
    has_analysis_shape = bool(
        unit.get("claim_id")
        or unit.get("source_support_map")
        or strength in {"strong", "moderate", "directional", "contextual", "limited_evidence"}
        or role in {"claimable", "directional", "contextual", "counter", "metric", "case", "technology"}
    )
    if not (has_analysis_shape and claim and refs and facts):
        return False
    text = " ".join(str(unit.get(key) or "") for key in ["claim", "reasoning", "counter_evidence", "actionable"])
    return not any(re.search(pattern, text, re.I) for pattern in PUBLIC_BLOCKING_PATTERNS)


def _unit_has_decision_support(unit: Dict[str, Any]) -> bool:
    source_quality = _as_dict(unit.get("source_quality"))
    allowed_uses = _as_dict(source_quality.get("allowed_use_distribution"))
    claim_type = str(unit.get("claim_type") or source_quality.get("claim_type") or _claim_type_for_unit(unit)).strip().lower()
    try:
        claim_ab_count = int(source_quality.get("claim_ab_count") or source_quality.get("ab_count") or 0)
    except (TypeError, ValueError):
        claim_ab_count = 0
    try:
        core_or_support = int(allowed_uses.get("core_claim") or 0) + int(allowed_uses.get("supporting") or 0)
    except (TypeError, ValueError):
        core_or_support = 0
    try:
        directional_c_count = int(source_quality.get("directional_c_distinct_sources") or allowed_uses.get("directional_signal") or 0)
    except (TypeError, ValueError):
        directional_c_count = 0
    if claim_ab_count > 0 or core_or_support > 0:
        return True
    return bool(not _strict_quality_mode() and claim_type != "hard_metric" and directional_c_count >= 2)


def _normalize_claim_binding_status(unit: Dict[str, Any], package: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(unit)
    refs = _as_list(normalized.get("evidence_refs"))
    if package and refs:
        normalized["source_quality"] = _support_profile(package, refs)
    claim_status = str(normalized.get("claim_status") or "").strip().lower()
    if claim_status in {"decision_ready", "core_claim"} and not _unit_has_decision_support(normalized):
        normalized["claim_status"] = "directional"
        normalized["quality_status"] = "directional_with_boundary"
        normalized["confidence"] = "low" if str(normalized.get("confidence") or "").strip().lower() in {"", "high"} else normalized.get("confidence")
        normalized["claim_downgraded_reason"] = "decision_ready_without_ab_source"
        normalized["rewrite_required"] = True
    elif claim_status in {"decision_ready", "core_claim"}:
        source_quality = _as_dict(normalized.get("source_quality"))
        try:
            ab_count = int(source_quality.get("claim_ab_count") or source_quality.get("ab_count") or 0)
            directional_c_count = int(source_quality.get("directional_c_distinct_sources") or 0)
        except (TypeError, ValueError):
            ab_count = 0
            directional_c_count = 0
        if ab_count <= 0 and directional_c_count >= 2 and _claim_type_for_unit(normalized) != "hard_metric":
            normalized["claim_status"] = "directional_ready"
            normalized["quality_status"] = normalized.get("quality_status") or "directional_with_boundary"
            normalized["claim_downgraded_reason"] = "directional_corroborated_without_ab_source"
    return normalized


def _norm_chapter_id(value: Any) -> str:
    """Normalize a chapter_id for cross-agent comparison.

    Different agents (analysis_agent, brain_agent, chapter_evidence_builder)
    sometimes store the same logical chapter id with different surface
    forms — `ch_01` vs `ch-01` vs `ch 01` vs `CH_01`. Strip whitespace,
    lower-case, and drop the connector characters so that all these forms
    compare equal.
    """

    return re.sub(r"[\s_\-./]+", "", str(value or "").strip().lower())


def _refs_from_structured_unit(unit: Dict[str, Any]) -> set[str]:
    refs: set[str] = set()
    for key in (
        "used_evidence_ids",
        "used_fact_refs",
        "supporting_fact_refs",
        "supporting_evidence_refs",
        "supporting_evidence",
        "evidence_refs",
    ):
        refs.update(str(ref or "").strip() for ref in _as_list(unit.get(key)) if str(ref or "").strip())
    source_support_map = _as_dict(unit.get("source_support_map"))
    for value in source_support_map.values():
        refs.update(str(ref or "").strip() for ref in _as_list(value) if str(ref or "").strip())
    return refs


def _refs_from_evidence_item(item: Dict[str, Any]) -> set[str]:
    refs = {
        str(item.get("evidence_id") or "").strip(),
        str(item.get("ref") or "").strip(),
        str(item.get("source_id") or "").strip(),
        str(item.get("source_ref") or "").strip(),
        str(item.get("citation_ref") or "").strip(),
        _citation_ref_from_evidence(item),
    }
    card = _as_dict(item.get("public_fact_card")) or _as_dict(item.get("fact_card"))
    refs.update(
        str(card.get(key) or "").strip()
        for key in ("evidence_id", "ref", "source_ref", "citation_ref", "source_id")
        if str(card.get(key) or "").strip()
    )
    return {ref for ref in refs if ref}


def _refs_from_package(package: Dict[str, Any]) -> set[str]:
    refs: set[str] = set()
    for collection in PUBLIC_EVIDENCE_COLLECTIONS:
        for item in _as_list(package.get(collection)):
            if isinstance(item, dict):
                refs.update(_refs_from_evidence_item(item))
    return refs


def _matches(unit: Dict[str, Any], package: Dict[str, Any]) -> bool:
    unit_chapter_id = str(unit.get("chapter_id") or "").strip()
    package_chapter_id = str(package.get("chapter_id") or "").strip()
    if unit_chapter_id and package_chapter_id:
        if unit_chapter_id == package_chapter_id:
            return True
        # Tolerate punctuation/whitespace differences between agents.
        if _norm_chapter_id(unit_chapter_id) == _norm_chapter_id(package_chapter_id):
            return True

    unit_refs = _refs_from_structured_unit(unit)
    package_refs = _refs_from_package(package)
    if unit_refs and package_refs and unit_refs.intersection(package_refs):
        return True

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
    chapters = [
        _parse_structured_string(item)
        for item in (_as_list(insight_package.get("chapters")) + _as_list(structured_analysis.get("chapter_insights")))
    ]
    for chapter in chapters:
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
                    "counter_evidence": claim_item.get("counter_evidence") or "；".join(str(item) for item in _as_list(chapter.get("counter_evidence_boundary"))[:3]),
                    "decision_implication": claim_item.get("decision_implication") or chapter.get("decision_implication") or "",
                    "actionable": claim_item.get("decision_implication") or chapter.get("decision_implication") or "",
                    "confidence": claim_item.get("confidence"),
                    "supporting_evidence": _as_list(claim_item.get("supporting_evidence")) or _as_list(claim_item.get("evidence_refs")),
                    "evidence_refs": _as_list(claim_item.get("supporting_evidence")) or _as_list(claim_item.get("evidence_refs")),
                    "what_to_verify_next": _as_list(claim_item.get("what_to_verify_next")),
                    "supporting_facts": _as_list(chapter.get("fact_chain")),
                    "mechanism_chain": _as_list(chapter.get("mechanism_chain")),
                    "counter_evidence_boundary": _as_list(chapter.get("counter_evidence_boundary")),
                }
            )
    for key in ("claim_units", "analysis_units"):
        for item in [_parse_structured_string(raw) for raw in _as_list(structured_analysis.get(key))]:
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


def _claim_from_fact(package: Dict[str, Any], fact: str) -> str:
    """Build a public claim from a distilled fact without material-list wording."""
    question = _question_for(package)
    fact = _compact(fact, 180)
    public_title = _compact(package.get("chapter_title"), 80) or "\u672c\u7ae0"
    context = f"{question} {public_title} {fact}"
    if not fact or _is_bad_public_fact(fact):
        return (
            f"{public_title}\u6682\u65f6\u53ea\u80fd\u4f5c\u4e3a\u5f85\u9a8c\u8bc1\u89c2\u5bdf\uff0c"
            "\u4e0d\u5e94\u6269\u5199\u4e3a\u786e\u5b9a\u6027\u884c\u4e1a\u7ed3\u8bba\u3002"
        )
    if re.search(r"demand|customer|case|purchase|pay|\u9700\u6c42|\u5ba2\u6237|\u6848\u4f8b|\u91c7\u8d2d|\u4ed8\u8d39|\u843d\u5730", context, flags=re.I):
        return (
            f"\u4ece\u843d\u5730\u548c\u9700\u6c42\u6837\u672c\u770b\uff0c{fact}\u3002"
            "\u8fd9\u66f4\u9002\u5408\u8bf4\u660e AI Agent \u5df2\u5728\u5c40\u90e8\u6d41\u7a0b\u4e2d\u51fa\u73b0\u9a8c\u8bc1\u4fe1\u53f7\uff0c"
            "\u4f46\u662f\u5426\u80fd\u591f\u6269\u5c55\u4e3a\u7a33\u5b9a\u5e02\u573a\u7a7a\u95f4\uff0c\u8fd8\u8981\u770b\u5ba2\u6237\u4ed8\u8d39\u3001\u590d\u7528\u9891\u7387\u548c ROI \u662f\u5426\u8fde\u7eed\u6210\u7acb\u3002"
        )
    if re.search(r"technology|standard|patent|maturity|product|\u6280\u672f|\u6807\u51c6|\u4e13\u5229|\u6210\u719f|\u4ea7\u54c1|\u5de5\u5177", context, flags=re.I):
        return (
            f"\u4ece\u6280\u672f\u548c\u4ea7\u54c1\u80fd\u529b\u770b\uff0c{fact}\u3002"
            "\u8fd9\u8868\u660e\u884c\u4e1a\u5224\u65ad\u4e0d\u80fd\u53ea\u770b\u6982\u5ff5\u70ed\u5ea6\uff0c"
            "\u8fd8\u9700\u8981\u62c6\u5206\u5de5\u5177\u8c03\u7528\u3001\u6743\u9650\u6cbb\u7406\u3001\u7a33\u5b9a\u6027\u548c\u96c6\u6210\u6210\u672c\u5bf9\u843d\u5730\u8282\u594f\u7684\u7ea6\u675f\u3002"
        )
    if re.search(r"risk|counter|failure|security|\u98ce\u9669|\u53cd\u8bc1|\u5931\u8d25|\u5b89\u5168|\u8fb9\u754c", context, flags=re.I):
        return (
            f"\u4ece\u98ce\u9669\u8fb9\u754c\u770b\uff0c{fact}\u3002"
            "\u8fd9\u610f\u5473\u7740 AI Agent \u7684\u91c7\u7528\u8282\u594f\u53d6\u51b3\u4e8e\u5b89\u5168\u3001\u8d23\u4efb\u5f52\u5c5e\u548c\u4eba\u5de5\u590d\u6838\u673a\u5236\u80fd\u5426\u4e0e\u4e1a\u52a1\u6d41\u7a0b\u540c\u6b65\u5efa\u7acb\u3002"
        )
    if re.search(r"market|size|growth|metric|revenue|scale|\u5e02\u573a|\u89c4\u6a21|\u589e\u901f|\u6307\u6807|\u6536\u5165|\u589e\u957f", context, flags=re.I):
        return (
            f"\u4ece\u89c4\u6a21\u548c\u6307\u6807\u53e3\u5f84\u770b\uff0c{fact}\u3002"
            "\u8fd9\u652f\u6301\u5bf9\u9636\u6bb5\u6027\u673a\u4f1a\u7684\u5224\u65ad\uff0c\u4f46\u4ecd\u9700\u533a\u5206\u6a21\u578b\u5382\u5546\u3001\u5e73\u53f0\u5de5\u5177\u548c\u5782\u76f4\u5e94\u7528\u4e4b\u95f4\u7684\u7edf\u8ba1\u8fb9\u754c\u3002"
        )
    return (
        f"\u8be5\u7ae0\u7684\u9636\u6bb5\u6027\u5224\u65ad\u5e94\u56f4\u7ed5\u8fd9\u4e00\u53ef\u6838\u9a8c\u4e8b\u5b9e\u5c55\u5f00\uff1a{fact}\u3002"
        "\u5b83\u53ef\u4ee5\u652f\u6491\u65b9\u5411\u6027\u5206\u6790\uff0c\u4f46\u7ed3\u8bba\u5f3a\u5ea6\u4ecd\u53d6\u51b3\u4e8e\u540e\u7eed\u72ec\u7acb\u6765\u6e90\u3001\u8fde\u7eed\u6307\u6807\u548c\u53cd\u5411\u6837\u672c\u7684\u4ea4\u53c9\u6821\u9a8c\u3002"
    )


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
    if (
        not proof_gaps
        and not _strict_quality_mode()
        and strength == "directional"
        and int(support.get("directional_c_distinct_sources") or 0) >= 2
    ):
        return "directional_ready"
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
    raw_refs = (
        _as_list(unit.get("used_evidence_ids"))
        or _as_list(unit.get("used_fact_refs"))
        or _as_list(unit.get("supporting_evidence_refs"))
        or _as_list(unit.get("supporting_evidence"))
        or _as_list(unit.get("evidence_refs"))
        or fallback_refs
    )
    refs = _source_refs_for_evidence_refs(package, raw_refs) or _dedupe(raw_refs, limit=8)
    supporting_facts = (
        _as_list(unit.get("evidence_basis"))
        or _as_list(unit.get("supporting_facts"))
        or _as_list(unit.get("fact_chain"))
    )
    reasoning_chain = _as_list(unit.get("reasoning_chain"))
    boundary_chain = _as_list(unit.get("limitation_boundary"))
    fallback_fact = _compact(
        unit.get("fact")
        or unit.get("supporting_fact")
        or (supporting_facts[0] if supporting_facts else ""),
        220,
    )
    question = _question_for(package, unit)
    original_claim = _compact(unit.get("claim") or unit.get("judgment") or unit.get("conclusion"), 320)
    original_text = " ".join(
        _compact(unit.get(key), 320)
        for key in ("claim", "reasoning", "mechanism", "counter_evidence", "actionable", "decision_implication")
    )
    normalized_from_weak = bool(original_claim.startswith(WEAK_CLAIM_PREFIXES) or _has_bad_pattern(original_text))
    claim = _normalize_claim(unit, package, fallback_fact)
    support = _support_profile(package, refs)
    analysis_strength = str(unit.get("claim_strength") or "").strip().lower()
    analysis_role = str(unit.get("analysis_role") or "").strip().lower()
    source_support_map = _as_dict(unit.get("source_support_map"))
    analysis_claim = bool(
        refs
        and supporting_facts
        and (
            unit.get("claim_id")
            or source_support_map
            or analysis_strength in {"strong", "moderate", "directional", "contextual", "limited_evidence"}
            or analysis_role in {"claimable", "directional", "contextual", "counter", "metric", "case", "technology"}
        )
    )
    if analysis_claim and support.get("claim_strength") in {"weak", "unsupported"}:
        support = {
            **support,
            "grade": "analysis_admitted",
            "claim_strength": analysis_strength or "directional",
        }
    confidence = unit.get("confidence") or ("low" if support["grade"] == "low" else "medium")
    proof_gaps = _as_list(package.get("missing_proof_standards"))
    claim_status = _claim_status_from_support(support, proof_gaps)
    if analysis_claim and claim_status == "appendix_only":
        claim_status = "directional" if analysis_strength in {"directional", "contextual", "limited_evidence", ""} else "decision_ready"
    payload = {
        "agent": AGENT_NAME,
        "claim_id": unit.get("claim_id") or unit.get("id") or "",
        "chapter_id": package.get("chapter_id"),
        "section_id": section_id,
        "question": question,
        "section_title": _compact(unit.get("section_title") or question, 160),
        "claim": claim,
        "reasoning": _ensure_reasoning(
            unit.get("reasoning")
            or unit.get("mechanism")
            or unit.get("explain_why")
            or " ".join(str(item) for item in reasoning_chain[:2]),
            package,
            fallback_fact,
        ),
        "counter_evidence": _ensure_counter(
            unit.get("counter_evidence")
            or unit.get("counter")
            or " ".join(str(item) for item in boundary_chain[:2])
        ),
        "actionable": _ensure_actionable(unit.get("actionable") or unit.get("decision_implication") or unit.get("next_action")),
        "mechanism": _compact(
            unit.get("mechanism")
            or unit.get("reasoning")
            or unit.get("explain_why")
            or " ".join(str(item) for item in reasoning_chain[:2]),
            520,
        ),
        "what_to_verify_next": _dedupe(_as_list(unit.get("what_to_verify_next")) + _public_verification_focus(package, refs), limit=8),
        "confidence": confidence,
        "confidence_reason": "该判断按来源层级、指标口径和反向样本覆盖程度分级，公开表达采用相应边界。",
        "claim_status": claim_status,
        "supporting_evidence": _dedupe(refs, limit=8),
        "evidence_refs": _dedupe(refs, limit=8),
        "used_fact_refs": _dedupe(refs, limit=8),
        "evidence_basis": _dedupe(supporting_facts, limit=8),
        "supporting_facts": _dedupe([*supporting_facts, *_evidence_facts_from_package(package, limit=8)], limit=8),
        "verification_metrics": _public_verification_focus(package, refs),
        "source_quality": support,
        "rewrite_required": normalized_from_weak,
        "rewrite_reason": "weak_structured_unit" if normalized_from_weak else "",
        "proof_gaps": proof_gaps,
        "follow_up_queries": _proof_followups(package),
    }
    source_quality = _as_dict(payload.get("source_quality"))
    original_status = str(unit.get("claim_status") or "").strip().lower()
    if original_status in {"decision_ready", "core_claim"} and int(source_quality.get("claim_ab_count") or source_quality.get("ab_count") or 0) <= 0:
        payload["claim_downgraded_reason"] = "decision_ready_without_ab_source"
    payload["block_type"] = _analysis_block_type_for_unit({**unit, **payload})
    for key in (
        "block_affinity",
        "fact_type",
        "proof_role",
        "layout_section_role",
        "claim_strength",
        "claim_strength_ceiling",
        "analysis_role",
        "source_support_map",
        "paragraph_seed",
        "hypothesis_id",
        "requirement_id",
        "requirement_ids",
        "lineage",
    ):
        value = unit.get(key)
        if value not in (None, "", []):
            payload[key] = value
    if not payload.get("block_type") and unit.get("block_type") not in (None, "", []):
        payload["block_type"] = str(unit.get("block_type") or "").strip()
    if not payload.get("mechanism"):
        payload["mechanism"] = payload.get("reasoning") or ""
    if support["grade"] == "low" and not analysis_claim:
        payload = rewrite_weak_claim_unit(payload, package=package, reason="low_source_quality")
    if proof_gaps:
        payload["claim_status"] = "directional" if claim_status in {"decision_ready", "directional", "directional_ready"} else "context_only"
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
        for collection in (
            "supporting_evidence",
            "metric_evidence",
            "case_evidence",
            "counter_evidence",
            "directional_evidence",
            "sample_evidence",
        )
        for item in _as_list(package.get(collection))
        if isinstance(item, dict)
    ]
    clue = [item for item in _as_list(package.get("clue_evidence")) if isinstance(item, dict)]
    candidates = [item for item in (core or supporting or clue) if _clean_public_fact_from_item(item)]
    first = candidates[0] if candidates else {}
    fact = _compact(_clean_public_fact_from_item(first), 260)
    supporting_facts = _evidence_facts_from_package(package, limit=8)
    refs = _dedupe([_citation_ref_from_evidence(item) for item in (core + supporting + clue)[:8]], limit=8)
    question = _question_for(package)
    support = _support_profile(package, refs)
    strength = str(support.get("claim_strength") or "")
    if _is_bad_public_fact(fact):
        fact = ""
    if fact and strength in {"strong", "medium"} and support["grade"] != "low":
        claim = _claim_from_fact(package, fact)
        reasoning = ""
    elif fact and strength in {"directional", "weak"}:
        claim = _directional_claim(package, fact)
        reasoning = ""
    elif fact:
        claim = f"{package.get('chapter_title') or '本章'}：{_compact(fact, 160)}。"
        reasoning = ""
    else:
        claim = ""
        reasoning = ""
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
            rendered = _clean_public_fact_from_item(item)
            if not rendered:
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
    clean_facts = [
        item
        for item in (_clean_public_text(value, 260) for value in facts)
        if item and not _has_bad_pattern(item)
    ]
    clean_facts = _dedupe(clean_facts, limit=fact_limit)
    if not clean_facts:
        return ""
    fact_text = _compact(
        "；".join(clean_facts),
        _env_int("REPORT_REASONING_FACT_TEXT_MAX_CHARS", 900, min_value=300, max_value=2400),
    )
    if lens == "boundary":
        return (
            f"围绕“{title}”，当前可验证材料首先说明：{fact_text}。"
            "这一判断仍受样本覆盖、来源独立性和后续反向案例约束。"
        )
    if lens == "decision":
        return (
            f"“{title}”对后续判断的价值在于把已出现的材料转成可跟踪变量：{fact_text}。"
            "如果这些变量持续同向，结论强度上调；若只停留在单点案例，则保持审慎。"
        )
    if lens == "mechanism":
        return (
            f"“{question}”需要从材料之间的变量关系判断。当前可用事实是：{fact_text}。"
            f"{profile}这些事实分别对应需求、供给、技术、客户或风险变量，只有变量之间方向一致时，章节判断才应上调。"
        )
    return f"当前可用于“{title}”的材料是：{fact_text}。结论强度取决于这些材料能否被更多独立来源和后续指标继续验证。"
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
    if not _looks_like_semiconductor_topic(package):
        if lens == "mechanism":
            return (
                f"从材料组合看，{fact_text}。这些信号需要分别落到需求来源、产品能力、客户导入和风险约束四个变量上判断。"
                f"{profile}如果这些变量在同一时间窗口内继续同向，判断可以上调；如果只是局部样本，则保留为审慎观察。"
            )
        if lens == "boundary":
            return (
                f"\u201c{title}\u201d\u7684\u8fb9\u754c\u6765\u81ea\u6837\u672c\u8986\u76d6\u3001\u6765\u6e90\u53ef\u6838\u9a8c\u6027\u548c\u6307\u6807\u53e3\u5f84\u3002"
                f"\u76ee\u524d\u53ef\u653e\u5728\u6b63\u6587\u4e2d\u7684\u4e8b\u5b9e\u662f\uff1a{fact_text}\u3002"
                "\u82e5\u540e\u7eed\u51fa\u73b0\u76f8\u53cd\u6848\u4f8b\u3001\u4ed8\u8d39\u6216\u590d\u8d2d\u4e0d\u8fbe\u9884\u671f\u3001\u6210\u672c\u4e0e\u5b89\u5168\u6cbb\u7406\u538b\u529b\u5347\u9ad8\uff0c"
                "\u672c\u8282\u5224\u65ad\u9700\u8981\u4ece\u5f3a\u7ed3\u8bba\u964d\u7ea7\u4e3a\u6709\u9650\u8bc1\u636e\u4e0b\u7684\u8d8b\u52bf\u89c2\u5bdf\u3002"
            )
        if lens == "decision":
            return (
                f"\u4ece\u884c\u4e1a\u5224\u65ad\u770b\uff0c\u201c{title}\u201d\u5e94\u8f6c\u5316\u4e3a\u51e0\u7c7b\u53ef\u8ddf\u8e2a\u7684\u4ea7\u4e1a\u53d8\u91cf\uff1a{fact_text}\u3002"
                "\u5f53\u8fd9\u4e9b\u53d8\u91cf\u540c\u65f6\u6307\u5411\u9700\u6c42\u771f\u5b9e\u3001\u5ba2\u6237\u5bfc\u5165\u3001\u6280\u672f\u53ef\u7528\u548c\u98ce\u9669\u53ef\u63a7\u65f6\uff0c"
                "\u62a5\u544a\u53ef\u4ee5\u63d0\u9ad8\u7ed3\u8bba\u5f3a\u5ea6\uff1b\u53ea\u6709\u5355\u70b9\u70ed\u5ea6\u6216\u5f31\u6765\u6e90\u65f6\uff0c\u5219\u5e94\u4fdd\u6301\u8bed\u6c14\u964d\u7ea7\u3002"
            )
        return (
            f"材料中可以用于判断的信号是：{fact_text}。这些事实可以支撑本节的审慎判断；结论能否上升为核心判断，取决于后续是否出现同口径、可交叉验证的连续材料。"
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
            f"“{title}”的价值不在于复述资料，而在于把资料转化为资源配置的优先顺序。当需求、供给、价格或利润材料持续互相印证时，资源会更多流向客户、订单、产能和盈利弹性；材料只停留在概念热度或单点新闻时，应保持审慎观察。"
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
    if not semiconductor_topic:
        if lens == "mechanism":
            claim = f"{chapter_title}\u5df2\u6709\u4e8b\u5b9e\u7ebf\u7d22\uff0c\u4f46\u9700\u56de\u5230\u9700\u6c42\u3001\u4ea7\u54c1\u80fd\u529b\u3001\u5ba2\u6237\u5bfc\u5165\u548c\u98ce\u9669\u7ea6\u675f\u7684\u4f20\u5bfc\u5173\u7cfb\u6765\u5224\u65ad\u3002"
            counter = "\u82e5\u540e\u7eed\u5ba2\u6237\u4ed8\u8d39\u3001\u7eed\u7ea6\u3001ROI\u6216\u5b89\u5168\u8d23\u4efb\u8bc1\u636e\u4e0d\u8db3\uff0c\u672c\u8282\u5224\u65ad\u5e94\u4fdd\u6301\u65b9\u5411\u6027\u3002"
            action = "\u540e\u7eed\u4f18\u5148\u8ffd\u8e2a\u5ba2\u6237\u5bfc\u5165\u3001\u590d\u8d2d\u3001\u6210\u672c\u53d8\u5316\u548c\u5b89\u5168\u6cbb\u7406\u6848\u4f8b\u3002"
        elif lens == "boundary":
            claim = f"{chapter_title}\u7684\u7ed3\u8bba\u8fb9\u754c\u53d6\u51b3\u4e8e\u8bc1\u636e\u662f\u5426\u72ec\u7acb\u3001\u53ef\u6838\u9a8c\uff0c\u4ee5\u53ca\u662f\u5426\u8986\u76d6\u53cd\u5411\u6837\u672c\u3002"
            counter = "\u5982\u679c\u5f31\u6765\u6e90\u3001\u641c\u7d22\u6458\u8981\u6216\u5c40\u90e8\u6848\u4f8b\u5360\u6bd4\u8fc7\u9ad8\uff0c\u7ed3\u8bba\u4e0d\u5e94\u4e0a\u5347\u4e3a\u786e\u5b9a\u6027\u5224\u65ad\u3002"
            action = "\u5728\u8bc4\u4f30\u7ed3\u8bba\u65f6\u540c\u6b65\u68c0\u67e5\u53ef\u8ffd\u6eaf\u6765\u6e90\u3001\u6307\u6807\u53e3\u5f84\u548c\u53cd\u8bc1\u6837\u672c\u3002"
        elif lens == "decision":
            claim = f"{chapter_title}\u53ef\u4f5c\u4e3a\u89c2\u5bdf\u4ea7\u4e1a\u4f18\u5148\u7ea7\u7684\u5207\u53e3\uff0c\u4f46\u5e94\u6309\u8bc1\u636e\u5f3a\u5ea6\u5206\u5c42\u8bfb\u53d6\u3002"
            counter = "\u82e5\u4e8b\u5b9e\u4ec5\u80fd\u8bc1\u660e\u8bdd\u9898\u70ed\u5ea6\uff0c\u5374\u4e0d\u80fd\u8bc1\u660e\u4ed8\u8d39\u3001\u843d\u5730\u6216\u6280\u672f\u7a33\u5b9a\u6027\uff0c\u5219\u4e0d\u5e94\u76f4\u63a5\u63a8\u5bfc\u4e3a\u5e02\u573a\u7a7a\u95f4\u3002"
            action = "\u628a\u5df2\u6709\u4e8b\u5b9e\u62c6\u6210\u9700\u6c42\u3001\u4f9b\u7ed9\u3001\u5546\u4e1a\u5316\u3001\u98ce\u9669\u56db\u7c7b\u53d8\u91cf\u7ee7\u7eed\u9a8c\u8bc1\u3002"
        else:
            claim = _claim_from_fact(package, facts[0] if facts else fact_text)
            counter = "\u8be5\u5224\u65ad\u7684\u8fb9\u754c\u5728\u4e8e\u539f\u6587\u6838\u9a8c\u548c\u53cd\u5411\u6837\u672c\u8986\u76d6\u662f\u5426\u8db3\u591f\u3002"
            action = "\u540e\u7eed\u89c2\u5bdf\u672c\u7ae0\u76f8\u5173\u7684\u6307\u6807\u53e3\u5f84\u3001\u4f01\u4e1a\u62ab\u9732\u548c\u5ba2\u6237\u6848\u4f8b\u3002"
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
    if lens == "mechanism" and semiconductor_topic:
        claim = f"{chapter_title}应放在“管制强度、技术可得性、产能位置、客户导入”四层关系中判断，而不是按单个新闻或单个企业外推。"
        counter = "如果出口管制边际放松、关键设备验证失败、成熟制程产能利用率下行，或客户认证慢于预期，供应链重构的方向和节奏都需要重新校准。"
        action = "后续跟踪集中在管制清单、设备材料国产验证、晶圆厂资本开支、封测和成熟制程订单，以及海外客户对中国供应链的采购态度。"
    elif lens == "mechanism":
        claim = f"{chapter_title}需要放在“供给约束、需求兑现、价格利润、反向样本”四层关系中观察，而不是按单个事实外推。"
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
    economics_title = "受益环节、约束条件与兑现路径" if semiconductor_topic else "商业化证据"
    facts = _evidence_facts_from_package(
        package,
        limit=_env_int("REPORT_FACTS_PER_CHAPTER_ARGUMENTS", 18, min_value=6, max_value=48),
    )
    support = _support_profile(package, _evidence_refs_from_package(package, limit=12))
    if str(support.get("grade") or "").lower() in {"low", "medium_low"} or len(facts) < 3:
        return [
            _unit_from_evidence(package, base_section_id),
            _deep_unit_from_package(
                package,
                section_id=f"{chapter_id}_boundary",
                lens="boundary",
                title=boundary_title,
                facts=facts[:6] or facts,
            ),
        ]
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


def _facts_from_collections(package: Dict[str, Any], collections: Sequence[str], *, limit: int) -> List[str]:
    facts: List[str] = []
    seen = set()
    for collection in collections:
        for item in _as_list(package.get(collection)):
            if not isinstance(item, dict):
                continue
            rendered = _clean_public_fact_from_item(item)
            if not rendered:
                continue
            key = re.sub(r"\s+", "", rendered.lower())[:160]
            if key in seen:
                continue
            seen.add(key)
            facts.append(rendered)
            if len(facts) >= limit:
                return facts
    return facts


def _collections_for_layout_section(section: Dict[str, Any]) -> List[str]:
    block_type = str(section.get("block_type") or section.get("output_type") or "").strip()
    if block_type == "metric_reconciliation":
        return ["metric_evidence", "core_evidence", "supporting_evidence"]
    if block_type in {"competitive_positioning", "customer_painpoint_matrix", "unit_economics", "case_comparison"}:
        return ["case_evidence", "core_evidence", "supporting_evidence", "directional_evidence"]
    if block_type == "technology_maturity":
        return ["metric_evidence", "core_evidence", "supporting_evidence", "directional_evidence"]
    if block_type in {"risk_trigger", "verification_checklist", "scenario_analysis"}:
        return ["counter_evidence", "core_evidence", "supporting_evidence", "directional_evidence"]
    return list(PUBLIC_EVIDENCE_COLLECTIONS)


def _strict_collections_for_layout_section(section: Dict[str, Any]) -> List[str]:
    block_type = str(section.get("block_type") or section.get("output_type") or "").strip()
    if block_type == "metric_reconciliation":
        return ["metric_evidence"]
    if block_type in {"competitive_positioning", "customer_painpoint_matrix", "unit_economics", "case_comparison"}:
        return ["case_evidence", "supporting_evidence", "directional_evidence"]
    if block_type == "technology_maturity":
        return ["metric_evidence", "supporting_evidence", "directional_evidence"]
    if block_type in {"risk_trigger", "verification_checklist", "scenario_analysis"}:
        return ["counter_evidence"]
    if block_type == "thesis":
        return ["core_evidence", "supporting_evidence", "directional_evidence"]
    return ["core_evidence", "supporting_evidence", "directional_evidence"]


def _fact_cards_from_package(package: Dict[str, Any], collections: Sequence[str], *, limit: int = 8) -> List[Dict[str, Any]]:
    cards: List[Dict[str, Any]] = []
    seen = set()
    for collection in collections:
        for item in _as_list(package.get(collection)):
            if not isinstance(item, dict):
                continue
            quality = _as_dict(item.get("public_fact_quality"))
            if quality and not bool(quality.get("eligible_for_report")):
                continue
            card = _as_dict(item.get("public_fact_card") or quality.get("public_fact_card"))
            if not card and not quality:
                card = _legacy_fact_card_from_item(item)
            if not card:
                continue
            card = dict(card)
            ref = str(card.get("source_ref") or _citation_ref_from_evidence(item) or item.get("source_ref") or "").strip()
            card["source_ref"] = ref
            fact = _compact(card.get("fact") or card.get("object"), 220)
            if not fact or _is_bad_public_fact(fact):
                continue
            card["fact"] = fact
            key = (ref, re.sub(r"\W+", "", fact.lower())[:120])
            if key in seen:
                continue
            seen.add(key)
            cards.append(card)
            if len(cards) >= limit:
                return cards
    return cards


def _analysis_cards_for_section(package: Dict[str, Any], section: Dict[str, Any], *, strict_layers: bool = True) -> List[Dict[str, Any]]:
    collections = _strict_collections_for_layout_section(section) if strict_layers else _collections_for_layout_section(section)
    cards = [
        card
        for card in _fact_cards_from_package(package, collections, limit=8)
        if _card_matches_layout_section(card, section)
    ][:4]
    if cards:
        return cards
    if strict_layers and can_render_block_from_evidence is not None:
        feasibility = can_render_block_from_evidence(str(section.get("block_type") or section.get("output_type") or ""), package)
        if feasibility.get("can_render"):
            cards = [
                card
                for card in _fact_cards_from_package(package, _collections_for_layout_section(section), limit=8)
                if _card_matches_layout_section(card, section)
            ][:4]
            if cards:
                return cards
    analysis = _as_dict(package.get("chapter_analysis"))
    allowed_types = {
        "metric_reconciliation": {"metric"},
        "technology_maturity": {"technology", "source_check"},
        "risk_trigger": {"counter"},
        "scenario_analysis": {"counter", "directional"},
        "competitive_positioning": {"case", "directional", "source_check"},
        "customer_painpoint_matrix": {"case", "directional"},
        "unit_economics": {"metric", "case"},
        "case_comparison": {"case", "directional"},
    }.get(str(section.get("block_type") or section.get("output_type") or "").strip(), set())
    for card in _as_list(analysis.get("fact_cards")):
        if not isinstance(card, dict):
            continue
        if not _card_matches_layout_section(card, section):
            continue
        if allowed_types and str(card.get("fact_type") or "") not in allowed_types:
            continue
        fact = _compact(card.get("fact") or card.get("object"), 220)
        if fact and not _is_bad_public_fact(fact):
            next_card = dict(card)
            next_card["fact"] = fact
            cards.append(next_card)
            if len(cards) >= 4:
                break
    return cards


def _card_matches_layout_section(card: Dict[str, Any], section: Dict[str, Any]) -> bool:
    block_type = str(section.get("block_type") or section.get("output_type") or "").strip()
    if not block_type or block_type in {"thesis", "evidence_matrix", "argument"}:
        return True
    affinity = {
        str(item or "").strip()
        for item in _as_list(card.get("block_affinity"))
        if str(item or "").strip()
    }
    if block_type in affinity:
        return True
    fact_type = str(card.get("fact_type") or "").strip()
    allowed = {
        "metric_reconciliation": {"metric"},
        "unit_economics": {"metric", "case"},
        "technology_maturity": {"technology", "source_check"},
        "risk_trigger": {"counter"},
        "scenario_analysis": {"counter", "directional"},
        "verification_checklist": {"counter"},
        "competitive_positioning": {"case", "source_check", "directional"},
        "customer_painpoint_matrix": {"case", "directional"},
        "case_comparison": {"case", "directional"},
    }.get(block_type, set())
    return bool(allowed and fact_type in allowed)


def _claim_for_block(block_type: str, facts: Sequence[str], cards: Sequence[Dict[str, Any]], strength: str) -> str:
    first = _compact(facts[0] if facts else "", 180)
    variable = _compact(_as_dict(cards[0] if cards else {}).get("analysis_variable") or _as_dict(cards[0] if cards else {}).get("variable"), 60)
    if variable.lower() in {
        "case",
        "metric",
        "counter",
        "source_check",
        "directional",
        "supporting",
        "fact",
        "关键事实",
        "事实",
        "案例",
        "指标",
        "线索",
    }:
        variable = ""
    if block_type == "metric_reconciliation":
        return f"{first} 说明市场空间判断已有可观察的数量信号。"
    if block_type == "technology_maturity":
        return f"{first} 显示落地瓶颈正在从模型能力转向可靠性、权限和集成成本。"
    if block_type in {"competitive_positioning", "customer_painpoint_matrix", "case_comparison"}:
        return f"{first} 表明需求验证正在从概念叙事转向具体客户、场景或采购动作。"
    if block_type in {"risk_trigger", "scenario_analysis", "verification_checklist"}:
        return f"{first} 提醒商业化节奏仍受成本、安全和责任边界约束。"
    if strength in {"directional", "weak"}:
        variable_text = variable or "相关场景"
        return f"{first} 指向{variable_text}已经出现可跟踪变化。"
    return f"{first}。"


def _reasoning_for_block(block_type: str, facts: Sequence[str], cards: Sequence[Dict[str, Any]], strength: str) -> str:
    first = _compact(facts[0] if facts else "", 180)
    second = _compact(facts[1] if len(facts) > 1 else "", 160)
    variable = _compact(_as_dict(cards[0] if cards else {}).get("analysis_variable") or _as_dict(cards[0] if cards else {}).get("variable"), 80)
    if not first:
        return ""
    if block_type == "metric_reconciliation":
        return f"数量信号的价值在于把需求热度转成可比较的市场变量；{first} 只能在主体、范围和期间清楚时用于判断空间上限。"
    if block_type == "technology_maturity":
        return f"技术信号需要落到可部署性上观察；{first} 更适合解释能力边界，而不是直接证明规模化落地已经完成。"
    if block_type in {"competitive_positioning", "customer_painpoint_matrix", "case_comparison"}:
        if second:
            return f"客户和玩家动作能验证落地质量；{first} 与 {second} 共同说明需求已出现，但仍要区分试点、采购和持续付费。"
        return f"客户和玩家动作能验证落地质量；{first} 说明已有场景线索，但还不能单独证明持续付费能力。"
    if block_type in {"risk_trigger", "scenario_analysis", "verification_checklist"}:
        return f"风险事实用于校准乐观判断；{first} 一旦扩大，会压低部署节奏、ROI 预期和责任分配的确定性。"
    if strength in {"directional", "weak"}:
        variable_text = variable or "这类动作"
        return f"{variable_text}的意义在于把讨论落到具体动作；{first} 说明相关变量已经可观察，但是否扩大取决于同类动作是否持续出现。"
    return f"{first} 将本章讨论从概念层拉到可观察变量，但结论仍取决于需求兑现、供给能力和商业化约束是否同向。"


def _boundary_for_block(block_type: str, cards: Sequence[Dict[str, Any]], strength: str) -> str:
    variable = _compact(_as_dict(cards[0] if cards else {}).get("analysis_variable") or _as_dict(cards[0] if cards else {}).get("variable"), 80)
    if block_type == "metric_reconciliation":
        return "如果指标缺少主体、范围、期间或来源引用，只能作为背景信息。"
    if block_type in {"competitive_positioning", "customer_painpoint_matrix", "case_comparison"}:
        return "如果案例停留在宣传、试点或单一客户，结论只能保留在场景线索层。"
    if block_type == "technology_maturity":
        return "如果缺少稳定性、安全、权限和集成成本证据，技术能力不能直接等同于规模化落地。"
    if block_type in {"risk_trigger", "scenario_analysis", "verification_checklist"}:
        return "如果反向样本增加或监管边界收紧，本章机会排序需要下调。"
    if strength in {"directional", "weak"}:
        return "边界在于样本范围和持续性；若后续没有同类动作，本段只保留为阶段性观察。"
    return ""


def _unit_from_chapter_analysis(package: Dict[str, Any], section: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    analysis = _as_dict(package.get("chapter_analysis"))
    cards = _analysis_cards_for_section(package, section, strict_layers=True)
    if not cards:
        return None
    facts = [_compact(card.get("fact") or card.get("object"), 170) for card in cards[:2]]
    facts = [fact for fact in facts if fact and not _is_bad_public_fact(fact)]
    refs = _dedupe([str(card.get("source_ref") or "").strip() for card in cards if str(card.get("source_ref") or "").strip()], limit=6)
    if not facts or not refs:
        return None
    strength = str(analysis.get("claim_strength") or cards[0].get("claim_strength_hint") or "directional")
    block_type = str(section.get("block_type") or section.get("output_type") or "").strip()
    claim = _claim_for_block(block_type, facts, cards, strength)
    reasoning = _reasoning_for_block(block_type, facts, cards, strength)
    boundary = _boundary_for_block(block_type, cards, strength)
    if not reasoning:
        return None
    return {
        "chapter_id": package.get("chapter_id"),
        "section_id": section.get("section_id"),
        "section_title": section.get("section_title"),
        "block_type": block_type,
        "output_type": section.get("output_type") or block_type,
        "layout_section_role": section.get("section_role") or block_type,
        "claim": claim,
        "public_claim": claim,
        "reasoning": _compact(reasoning, 420),
        "mechanism": _compact(reasoning, 420),
        "counter_evidence": _compact(boundary, 320),
        "actionable": "",
        "decision_implication": "",
        "evidence_basis": facts,
        "supporting_facts": facts,
        "used_fact_refs": refs,
        "evidence_refs": refs,
        "supporting_evidence": refs,
        "claim_strength": strength,
        "claim_status": "directional" if strength in {"directional", "weak"} else "decision_ready",
        "quality_status": "directional_with_boundary" if strength in {"directional", "weak"} else "valid",
        "evidence_backed": bool(refs and facts and reasoning),
        "observation_only": strength in {"directional", "weak"},
        "fact_card_to_block_match": True,
        "fact_card_count": len(cards),
    }


def _refs_from_collections(package: Dict[str, Any], collections: Sequence[str], *, limit: int = 8) -> List[str]:
    refs: List[str] = []
    for collection in collections:
        for item in _as_list(package.get(collection)):
            if not isinstance(item, dict):
                continue
            ref = _citation_ref_from_evidence(item)
            if ref:
                refs.append(ref)
            if len(refs) >= limit * 2:
                break
        if len(refs) >= limit * 2:
            break
    return _dedupe(refs, limit=limit)


def _refs_for_layout_section(package: Dict[str, Any], section: Dict[str, Any], *, limit: int = 8, strict_layers: bool = False) -> List[str]:
    explicit_refs = _as_list(section.get("required_evidence_refs"))
    if explicit_refs:
        return _source_refs_for_evidence_refs(package, explicit_refs) or _dedupe(explicit_refs, limit=limit)
    collections = _strict_collections_for_layout_section(section) if strict_layers else _collections_for_layout_section(section)
    return _refs_from_collections(package, collections, limit=limit)


def _facts_for_layout_section(package: Dict[str, Any], section: Dict[str, Any], *, index: int, fallback: Sequence[str], strict_layers: bool = False) -> List[str]:
    block_type = str(section.get("block_type") or section.get("output_type") or "").strip()
    if strict_layers:
        return _facts_from_collections(package, _strict_collections_for_layout_section(section), limit=8)
    if block_type:
        facts = _facts_from_collections(package, _collections_for_layout_section(section), limit=8)
    else:
        facts = []
    if facts:
        return facts
    if block_type == "metric_reconciliation":
        facts = _facts_from_collections(package, ["metric_evidence", "core_evidence", "supporting_evidence"], limit=8)
    elif block_type in {"competitive_positioning", "customer_painpoint_matrix", "unit_economics", "case_comparison"}:
        facts = _facts_from_collections(package, ["case_evidence", "core_evidence", "supporting_evidence", "directional_evidence"], limit=8)
    elif block_type == "technology_maturity":
        facts = _facts_from_collections(package, ["core_evidence", "supporting_evidence", "directional_evidence"], limit=8)
    elif block_type in {"risk_trigger", "verification_checklist", "scenario_analysis"}:
        facts = _facts_from_collections(package, ["counter_evidence", "core_evidence", "supporting_evidence", "directional_evidence"], limit=8)
    else:
        start = max(0, (index - 1) * 4)
        facts = list(fallback[start : start + 8] or fallback)
    return facts


def _units_from_layout_sections(package: Dict[str, Any], sections: Sequence[Dict[str, Any]], *, strict_layers: bool = False) -> List[Dict[str, Any]]:
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
            unit_facts = _facts_for_layout_section(package, section, index=index, fallback=facts, strict_layers=strict_layers)
            if strict_layers and not unit_facts:
                continue
            unit = _deep_unit_from_package(
                package,
                section_id=section_id,
                lens=_lens_for_layout_section(section),
                title=title or "章节论证",
                facts=unit_facts,
            )
        mapped_refs = _refs_for_layout_section(package, section, limit=8, strict_layers=strict_layers)
        if strict_layers and not mapped_refs:
            continue
        if mapped_refs:
            unit["evidence_refs"] = mapped_refs
            unit["supporting_evidence"] = mapped_refs
            unit["source_quality"] = _support_profile(package, unit["evidence_refs"])
        unit["block_type"] = section.get("block_type") or section.get("output_type")
        unit["layout_section_role"] = section.get("section_role")
        refs = _as_list(unit.get("evidence_refs"))
        facts_for_unit = _as_list(unit.get("supporting_facts"))
        evidence_backed = bool(refs and facts_for_unit)
        unit["layout_generated"] = True
        unit["evidence_backed"] = evidence_backed
        unit["observation_only"] = not evidence_backed
        units.append(unit)
    return units


def _structured_unit_match_score(unit: Dict[str, Any], section: Dict[str, Any]) -> int:
    section_id = str(section.get("section_id") or "").strip()
    block_type = str(section.get("block_type") or section.get("output_type") or "").strip()
    title = re.sub(r"\s+", "", str(section.get("section_title") or section.get("title") or "").strip().lower())
    unit_section_id = str(unit.get("section_id") or "").strip()
    unit_block = str(unit.get("block_type") or unit.get("output_type") or unit.get("layout_section_role") or "").strip()
    unit_title = re.sub(
        r"\s+",
        "",
        str(unit.get("section_title") or unit.get("question") or unit.get("dimension") or "").strip().lower(),
    )
    if section_id and unit_section_id == section_id:
        return 100
    if block_type and unit_block == block_type:
        return 80
    if title and unit_title and (title == unit_title or title in unit_title or unit_title in title):
        return 50
    required_refs = {str(ref or "").strip() for ref in _as_list(section.get("required_evidence_refs")) if str(ref or "").strip()}
    unit_refs = {
        str(ref or "").strip()
        for ref in (
            _as_list(unit.get("used_evidence_ids"))
            + _as_list(unit.get("used_fact_refs"))
            + _as_list(unit.get("supporting_evidence_refs"))
            + _as_list(unit.get("supporting_evidence"))
            + _as_list(unit.get("evidence_refs"))
        )
        if str(ref or "").strip()
    }
    matched_claim = _as_dict(section.get("matched_llm_claim"))
    matched_refs = {
        str(ref or "").strip()
        for ref in (
            _as_list(matched_claim.get("used_evidence_ids"))
            + _as_list(matched_claim.get("used_fact_refs"))
            + _as_list(matched_claim.get("supporting_evidence_refs"))
            + _as_list(matched_claim.get("supporting_evidence"))
            + _as_list(matched_claim.get("evidence_refs"))
        )
        if str(ref or "").strip()
    }
    if matched_refs and unit_refs and matched_refs.intersection(unit_refs):
        return 95
    if block_type and claim_supported_block_types is not None:
        try:
            if block_type in claim_supported_block_types(unit):
                return 85
        except Exception:
            pass
    if required_refs and unit_refs and required_refs.intersection(unit_refs):
        return 30
    return 0


def _pop_structured_unit_for_section(matched_units: List[Dict[str, Any]], section: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    best_index = -1
    best_score = 0
    for index, unit in enumerate(matched_units):
        score = _structured_unit_match_score(unit, section)
        if score > best_score:
            best_score = score
            best_index = index
    if best_index >= 0 and best_score > 0:
        return matched_units.pop(best_index)
    return None


def _layout_match_reason(score: int) -> str:
    if score >= 100:
        return "section_id"
    if score >= 80:
        return "block_type"
    if score >= 50:
        return "section_title"
    if score >= 30:
        return "evidence_ref"
    return "layout_fallback"


def _analysis_block_type_for_unit(unit: Dict[str, Any]) -> str:
    def _normalize_block_type_value(value: Any) -> str:
        text = str(value or "").strip().strip("\"'")
        match = re.fullmatch(r"\[\s*['\"]?([a-zA-Z0-9_]+)['\"]?\s*\]", text)
        if match:
            text = match.group(1)
        return text

    for key in ("block_type", "output_type", "layout_section_role"):
        value = _normalize_block_type_value(unit.get(key))
        if value:
            return value
    if claim_supported_block_types is not None:
        try:
            for block_type in claim_supported_block_types(unit):
                value = _normalize_block_type_value(block_type)
                if value:
                    return value
        except Exception:
            pass
    return "integrated_signal"


def _structured_unit_priority(unit: Dict[str, Any]) -> tuple[int, int, int]:
    strength = str(unit.get("claim_strength") or "").strip().lower()
    analysis_role = str(unit.get("analysis_role") or "").strip().lower()
    analysis_signals = int(
        bool(unit.get("claim_id") or unit.get("id"))
        + bool(_as_dict(unit.get("source_support_map")))
        + bool(analysis_role)
        + int(strength in {"strong", "moderate", "directional", "contextual", "limited_evidence"})
    )
    ref_count = len(_refs_from_structured_unit(unit))
    fact_count = len(_as_list(unit.get("evidence_basis")) + _as_list(unit.get("supporting_facts")))
    return (analysis_signals, ref_count, fact_count)


def _argument_unit_duplicate_key(unit: Dict[str, Any]) -> str:
    refs = ",".join(sorted(str(ref or "").strip() for ref in _as_list(unit.get("evidence_refs")) if str(ref or "").strip()))
    claim = re.sub(r"\s+", "", str(unit.get("claim") or "").strip().lower())
    return f"{claim[:180]}|{refs}"


def _argument_unit_issues(unit: Dict[str, Any]) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    claim = str(unit.get("claim") or "").strip()
    reasoning = str(unit.get("reasoning") or "").strip()
    counter = str(unit.get("counter_evidence") or "").strip()
    actionable = str(unit.get("actionable") or "").strip()
    refs = _as_list(unit.get("evidence_refs"))
    used_refs = _as_list(unit.get("used_fact_refs"))
    facts = _as_list(unit.get("evidence_basis")) + _as_list(unit.get("supporting_facts"))
    fact_card_backed = bool((refs or used_refs) and facts)
    strength = str(unit.get("claim_strength") or _as_dict(unit.get("source_quality")).get("claim_strength") or "").strip().lower()
    directional_unit = strength in {"directional", "weak", "limited", "limited_evidence"} or str(unit.get("claim_status") or "").strip().lower() in {"directional", "directional_ready", "context_only"}
    if not claim:
        issues.append({"type": "missing_claim"})
    elif claim.startswith(WEAK_CLAIM_PREFIXES) or _has_bad_pattern(claim):
        issues.append({"type": "weak_claim", "severity": "warning" if directional_unit and refs else "error"})
    if not reasoning:
        issues.append({"type": "missing_reasoning", "severity": "warning" if fact_card_backed else "error"})
    elif _has_bad_pattern(reasoning) or _is_bad_public_fact(reasoning):
        issues.append({"type": "weak_reasoning"})
    elif not any(word in reasoning for word in CAUSE_WORDS):
        issues.append({"type": "reasoning_missing_causal_chain", "severity": "warning"})
    if not counter:
        issues.append({"type": "missing_counter_evidence", "severity": "warning" if directional_unit or fact_card_backed else "error"})
    elif _has_bad_pattern(counter):
        issues.append({"type": "weak_counter_evidence"})
    if not actionable:
        issues.append({"type": "missing_actionable", "severity": "warning" if directional_unit or fact_card_backed else "error"})
    elif _has_bad_pattern(actionable):
        issues.append({"type": "weak_actionable"})
    elif not any(word in actionable for word in ACTION_WORDS):
        issues.append({"type": "actionable_missing_action_word"})
    if not refs:
        issues.append({"type": "missing_evidence_refs"})
    if any(_is_bad_public_fact(item) for item in _as_list(unit.get("supporting_facts"))):
        issues.append({"type": "invalid_supporting_fact"})
    return issues


def _fact_card_backed_unit(unit: Dict[str, Any]) -> bool:
    refs = _as_list(unit.get("evidence_refs")) or _as_list(unit.get("used_fact_refs"))
    facts = _as_list(unit.get("evidence_basis")) + _as_list(unit.get("supporting_facts"))
    return bool(refs and facts)


def _unit_needs_public_rewrite(unit: Dict[str, Any], issues: Sequence[Dict[str, Any]]) -> bool:
    if _analysis_claim_is_renderable(unit):
        return False
    blocking = [issue for issue in issues if issue.get("severity") != "warning"]
    if not blocking:
        return False
    if not _fact_card_backed_unit(unit):
        return True
    rewrite_types = {
        "missing_claim",
        "weak_claim",
        "weak_reasoning",
        "weak_counter_evidence",
        "weak_actionable",
        "missing_evidence_refs",
        "invalid_supporting_fact",
    }
    return any(str(issue.get("type") or "") in rewrite_types for issue in blocking)


def rewrite_weak_claim_unit(unit: Dict[str, Any], *, package: Optional[Dict[str, Any]] = None, reason: str = "weak_claim") -> Dict[str, Any]:
    package = _as_dict(package)
    rewritten = dict(unit)
    supporting_facts = [
        _compact(item, 150)
        for item in _as_list(unit.get("supporting_facts")) + _as_list(unit.get("fact_chain"))
        if str(item or "").strip() and not _is_bad_public_fact(item)
    ]
    fallback_fact = _compact(unit.get("fact") or unit.get("supporting_fact") or (supporting_facts[0] if supporting_facts else ""), 150)
    if _is_bad_public_fact(fallback_fact):
        fallback_fact = ""
    question = _question_for(package, unit) if package else _compact(unit.get("question") or unit.get("section_title"), 180)
    title = _compact(package.get("chapter_title") if package else unit.get("section_title"), 80) or "本节"
    refs = _as_list(unit.get("evidence_refs"))
    source_quality = _as_dict(unit.get("source_quality")) or (_support_profile(package, refs) if package else {})
    fact_phrase = fallback_fact or (supporting_facts[0] if supporting_facts else "")
    facts_for_basis = _dedupe([fact_phrase, *supporting_facts, *_evidence_facts_from_package(package, limit=3)], limit=2)
    basis = "；".join([item for item in facts_for_basis if item])
    strength = str(source_quality.get("claim_strength") or "").strip().lower()
    if basis:
        if strength in {"strong", "medium", "moderate"} and source_quality.get("grade") not in {"low", "medium_low"}:
            existing_claim = str(unit.get("claim") or "").strip()
            claim = existing_claim if existing_claim and not _has_bad_pattern(existing_claim) else (_claim_from_fact(package, fact_phrase) if package else f"{title}已有可用于判断的事实基础。")
            reasoning = ""
            confidence = unit.get("confidence") or "medium"
        else:
            existing_claim = str(unit.get("claim") or "").strip()
            claim = existing_claim if existing_claim and not _has_bad_pattern(existing_claim) else (_claim_from_fact(package, fact_phrase) if package else f"{title}出现可用于跟踪的方向性信号。")
            reasoning = ""
            confidence = "low"
    else:
        claim = ""
        reasoning = ""
        confidence = "low"
    counter = _ensure_counter(unit.get("counter_evidence"))
    actionable = _ensure_actionable(unit.get("actionable") or unit.get("decision_implication"))
    rewritten.update(
        {
            "claim": claim,
            "public_claim": claim,
            "evidence_basis": basis,
            "reasoning": reasoning,
            "reasoning_chain": reasoning,
            "mechanism": reasoning,
            "counter_evidence": counter,
            "limitation_boundary": counter,
            "actionable": actionable,
            "confidence": confidence,
            "confidence_reason": rewritten.get("confidence_reason") or "已按事实依据、变量解释、结论强度和边界条件重写。",
            "rewrite_required": True,
            "rewrite_reason": reason,
            "verification_metrics": _as_list(unit.get("verification_metrics")) or (_verification_metrics(package, refs) if package else []),
            "claim_status": "directional" if confidence == "low" else (unit.get("claim_status") or "decision_ready"),
            "quality_status": "directional_with_boundary" if confidence == "low" else (unit.get("quality_status") or "valid"),
            "supporting_facts": facts_for_basis,
            "claim_strength": strength or ("directional" if confidence == "low" else "moderate"),
            "evidence_backed": bool(refs and facts_for_basis),
            "observation_only": not bool(refs and facts_for_basis),
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


def _apply_section_metadata(unit: Dict[str, Any], section: Dict[str, Any], match_score: int) -> None:
    """Copy layout-section metadata onto a built argument unit.

    Previously the run loop assigned six identical fields twice (once after
    building the unit, once after a rewrite). Centralizing it avoids drift
    between the two copies and keeps `run_claim_builder_agent` readable.
    """

    section = _as_dict(section)
    unit["section_id"] = section.get("section_id") or unit.get("section_id")
    unit["section_title"] = section.get("section_title") or unit.get("section_title")
    unit["block_type"] = (
        section.get("block_type") or section.get("output_type") or unit.get("block_type")
    )
    unit["output_type"] = (
        section.get("output_type") or unit.get("output_type") or unit.get("block_type")
    )
    unit["layout_section_role"] = section.get("section_role") or unit.get("layout_section_role")
    unit["layout_match_score"] = match_score
    unit["layout_match_reason"] = _layout_match_reason(match_score)
    required_refs = _as_list(section.get("required_evidence_refs"))
    if required_refs and not _as_list(unit.get("evidence_refs")):
        unit["evidence_refs"] = required_refs
        unit["supporting_evidence"] = required_refs


def run_claim_builder_agent(
    *,
    chapter_evidence_packages: Optional[Sequence[Dict[str, Any]]] = None,
    micro_layouts: Optional[Sequence[Dict[str, Any]]] = None,
    structured_analysis: Optional[Dict[str, Any]] = None,
    llm_client: Any = None,
) -> List[Dict[str, Any]]:
    del llm_client
    structured_analysis = _as_dict(structured_analysis)
    force_evidence_rebuild = _analysis_quality_requires_evidence_rebuild(structured_analysis)
    # needs_rewrite means "rewrite polluted claims", not "discard the whole
    # analysis". Keeping structured units preserves fact_chain/reasoning_chain
    # and avoids falling back to generic chapter templates.
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
                for collection in PUBLIC_EVIDENCE_COLLECTIONS
                for item in _as_list(package.get(collection))
                if isinstance(item, dict)
            ],
            limit=8,
        )
        matched = sorted(
            [unit for unit in structured_units if _matches(unit, package)],
            key=_structured_unit_priority,
            reverse=True,
        )
        if sections:
            matched_pool = [dict(unit) for unit in matched]
            generated_keys = set()
            for index, section in enumerate(sections, start=1):
                if str(section.get("section_role") or "").strip() == "evidence_gap" and not _as_list(section.get("required_evidence_refs")):
                    continue
                analysis_unit = _unit_from_chapter_analysis(package, section)
                selected = _pop_structured_unit_for_section(matched_pool, section)
                match_score = _structured_unit_match_score(selected, section) if selected else 0
                if selected:
                    built_unit = _unit_from_structured(
                        selected,
                        package,
                        str(section.get("section_id") or f"{chapter_id}_s{index}"),
                        _as_list(section.get("required_evidence_refs")) or fallback_refs,
                    )
                elif analysis_unit:
                    built_unit = analysis_unit
                else:
                    if force_evidence_rebuild:
                        continue
                    layout_units = _units_from_layout_sections(package, [section], strict_layers=force_evidence_rebuild)
                    if not layout_units:
                        continue
                    built_unit = layout_units[0]
                _apply_section_metadata(built_unit, section, match_score)
                unit_issues = _argument_unit_issues(built_unit)
                if force_evidence_rebuild and _unit_needs_public_rewrite(built_unit, unit_issues):
                    built_unit = rewrite_weak_claim_unit(built_unit, package=package, reason="analysis_quality_rewrite")
                    _apply_section_metadata(built_unit, section, match_score)
                duplicate_key = _argument_unit_duplicate_key(built_unit)
                if duplicate_key and duplicate_key in generated_keys:
                    continue
                generated_keys.add(duplicate_key)
                argument_units.append(built_unit)
            extra_limit = _env_int("REPORT_EXTRA_LLM_CLAIMS_PER_CHAPTER", 4, min_value=0, max_value=8)
            extra_index = 0
            for unit in matched_pool:
                if extra_index >= max(0, extra_limit):
                    break
                if not _refs_from_structured_unit(unit):
                    continue
                extra_index += 1
                fallback_section = {
                    "section_id": f"{chapter_id}_llm_extra_{extra_index}",
                    "section_title": unit.get("section_title") or unit.get("question") or package.get("chapter_title"),
                    "block_type": _analysis_block_type_for_unit(unit),
                    "output_type": _analysis_block_type_for_unit(unit),
                    "section_role": "integrated_signal",
                }
                built_unit = _unit_from_structured(
                    unit,
                    package,
                    str(fallback_section["section_id"]),
                    fallback_refs,
                )
                built_unit["block_type"] = str(fallback_section["block_type"] or "integrated_signal")
                built_unit["layout_section_role"] = "integrated_signal"
                built_unit["llm_claim_block_fallback"] = True
                _apply_section_metadata(built_unit, fallback_section, 0)
                duplicate_key = _argument_unit_duplicate_key(built_unit)
                if duplicate_key and duplicate_key in generated_keys:
                    continue
                generated_keys.add(duplicate_key)
                argument_units.append(built_unit)
        else:
            if matched:
                for index, unit in enumerate(matched[:4], start=1):
                    built_unit = _unit_from_structured(unit, package, section_id if index == 1 else f"{chapter_id}_s{index}", fallback_refs)
                    unit_issues = _argument_unit_issues(built_unit)
                    if force_evidence_rebuild and _unit_needs_public_rewrite(built_unit, unit_issues):
                        built_unit = rewrite_weak_claim_unit(built_unit, package=package, reason="analysis_quality_rewrite")
                    argument_units.append(built_unit)
                if len(matched) < 3:
                    argument_units.extend(_deep_units_from_package(package, f"{chapter_id}_deep")[1:])
            else:
                analysis_unit = _unit_from_chapter_analysis(package, {"section_id": section_id, "block_type": "thesis", "section_title": package.get("chapter_title")})
                if analysis_unit:
                    argument_units.append(analysis_unit)
                elif force_evidence_rebuild:
                    continue
                else:
                    layout_units = _units_from_layout_sections(package, sections, strict_layers=force_evidence_rebuild)
                    argument_units.extend(layout_units or _deep_units_from_package(package, section_id))
    cleaned: List[Dict[str, Any]] = []
    repeated_claim_prefixes: set[tuple[str, str]] = set()
    for unit in argument_units:
        package = _as_dict(package_by_id.get(str(unit.get("chapter_id") or "")))
        unit = _normalize_claim_binding_status(unit, package)
        unit = _clean_argument_unit_public_fields(unit)
        was_rewritten = bool(unit.get("rewrite_required"))
        original_issues = _argument_unit_issues(unit)
        if _unit_needs_public_rewrite(unit, original_issues):
            unit = rewrite_weak_claim_unit(unit, package=package, reason="contract_rewrite")
            unit = _normalize_claim_binding_status(unit, package)
            unit = _clean_argument_unit_public_fields(unit)
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
        claim_prefix = re.sub(r"\s+", "", str(unit.get("claim") or "").strip().lower())[:96]
        chapter_key = str(unit.get("chapter_id") or "")
        repeated_claim = bool(force_evidence_rebuild and claim_prefix and (chapter_key, claim_prefix) in repeated_claim_prefixes)
        if claim_prefix:
            repeated_claim_prefixes.add((chapter_key, claim_prefix))
        if repeated_claim:
            issues = _as_list(unit.get("quality_issues"))
            issues.append({"type": "argument_unit_repetition_failed", "severity": "warning"})
            unit["quality_issues"] = issues
            unit["omit_from_report"] = True
            unit["public_render"] = False
            unit["observation_only"] = True
            unit["internal_reason"] = "argument_unit_repetition_failed"
            cleaned.append(unit)
            continue
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
