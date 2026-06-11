from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


CORE_LIMIT = 6
SUPPORT_LIMIT = 8
COUNTER_LIMIT = 4
METRIC_LIMIT = 5
CASE_LIMIT = 4
DIRECTIONAL_LIMIT = 3

EVIDENCE_LAYER_KEYS = (
    "core_evidence",
    "supporting_evidence",
    "metric_evidence",
    "case_evidence",
    "counter_evidence",
    "directional_evidence",
    "sample_evidence",
)

CHAPTER_MATCH_MIN_SCORE = 18

GENERIC_CHAPTER_TERMS = {
    "agent",
    "agents",
    "ai",
    "智能体",
    "人工智能",
    "发展",
    "市场",
    "行业",
    "生态",
    "需求",
    "技术",
    "产品",
    "数据",
    "来源",
    "证据",
    "分析",
    "报告",
    "如何",
    "哪些",
    "什么",
}

GENERIC_ENGLISH_CHAPTER_TERMS = {
    "and",
    "the",
    "which",
    "what",
    "when",
    "where",
    "why",
    "how",
    "with",
    "from",
    "into",
    "this",
    "that",
}


BAD_FACT_PATTERNS = [
    r"^\s*-?\d{2,6}(?:\.\d+)?\s*(?:$|[;,\.\u3002\uff1b\uff0c])",
    r"^\s*(?:fact|key fact|metric|source_check|status|policy target|competitive comparison|cost)\s*[:\uff1a]\s*-?\d{1,6}(?:\.\d+)?\b",
    r"^\s*(?:\u4e8b\u5b9e|\u5173\u952e\u4e8b\u5b9e|\u7ade\u4e89\u5bf9\u6bd4|\u653f\u7b56\u76ee\u6807|\u653f\u7b56\u76d1\u7ba1|\u6210\u672c)\s*[:\uff1a]\s*-?\d{1,6}(?:\.\d+)?\b",
    r"^\s*(?:\u5185\u5bb9\u8bf4\u660e|\u65f6\u95f4)\s*[:\uff1a]",
    r"\u4ec0\u4e48\u662f.*(?:AI\s*Agent|\u667a\u80fd\u4f53)",
    r"\u6700\u65b0\u62a5\u9053|\u539f\u521b#|##\s*|\u80a1\u7968\s*#",
    r"Google\s+Patents|CN\d{6,}[A-Z]?",
    r"\u519c\u4e1a\u4eba\u5de5\u667a\u80fd",
    r"\u6570\u636e\u6295\u6bd2",
    r"\u7eba\u7ec7",
    r"\u667a\u80fd\u624b\u673a",
    r"\u6982\u5ff5\u80a1|\u6da8\u505c|\u5343\u4ebf\u4ff1\u4e50\u90e8|\u591a\u80a1\u5f3a\u52bf",
    r"A\u80a1-\u7814\u62a5\u8be6\u60c5|\u624b\u673a\u65b0\u6d6a\u7f51|_\u624b\u673a\u65b0\u6d6a\u7f51",
    r"\u62db\u6807\u7f16\u53f7\s*\u70b9\u51fb\u67e5\u770b|\u62db\u6807\u4f30\u4ef7\s*\u70b9\u51fb\u67e5\u770b|\u62a5\u540d\u622a\u6b62\u65f6\u95f4\s*\u70b9\u51fb\u67e5\u770b|\u6295\u6807\u622a\u6b62\u65f6\u95f4\s*\u70b9\u51fb\u67e5\u770b",
    r"\u5b98\u65b9\u6296\u97f3\u3001\u5feb\u624b|\u4e00\u5e74\u591a\u5c11\u94b1|\u771f\u5b9e\u62a5\u4ef7|\u6743\u5a01\u6307\u5357",
    r"\u00a9\u8457\u4f5c\u6743|\u8f6c\u8f7d\u6388\u6743|\u5c06\u8ffd\u7a76\u6cd5\u5f8b\u8d23\u4efb",
    r"Scribd",
    r"SEO",
    r"example\.(?:com|gov|org)",
    r"Official data shows AI agent adoption reached 50%",
    r"^URL[:\uff1a]",
    r"Skip to (?:content|main content)",
    r"picture intentionally omitted",
    r"\*\*==>\s*picture intentionally omitted\s*<==\*\*",
    r"登录\s+首页|上一篇|下一篇|分享到|AI帮你提炼|智能挖掘|智享会员|会员积分",
    r"首页问\s*·\s*答|热搜公司|热搜词|登录注册",
    r"Caret right|View all products|Product\s+Documentation",
    r"^\s*(?:事实|竞争对比|关键事实|政策目标)\s*[:：]\s*-?\d{1,3}(?:\.\d+)?\b",
    r"^\s*-?\d{1,3}(?:\.\d+)?\s*[;；，,]",
    r"以下是对整篇.*(?:深度分析|框架提炼)",
]


LOW_QUALITY_SOURCE_PATTERNS = [
    r"twitter\.com|x\.com|instagram\.com|facebook\.com",
    r"baike\.baidu\.com|baijiahao\.baidu\.com",
    r"blog\.csdn\.net|cnblogs\.com|juejin\.cn",
    r"fxbaogao\.com|sgpjbg\.com|jazzyear\.com",
]


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, dict) and isinstance(value.get("sample"), list):
        return list(value.get("sample") or [])
    return list(value) if isinstance(value, list) else []


def _compact(value: Any, max_chars: int = 320) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "..."


def _fact_text(item: Dict[str, Any]) -> str:
    return _compact(
        item.get("distilled_fact")
        or item.get("fact")
        or item.get("clean_fact")
        or item.get("content")
        or item.get("evidence")
        or item.get("summary")
        or item.get("answer")
        or item.get("text"),
        900,
    )


def _bad_fact_text(text: str) -> bool:
    if not str(text or "").strip():
        return True
    return any(re.search(pattern, text, flags=re.I) for pattern in BAD_FACT_PATTERNS)


def _navigation_or_search_text(text: Any) -> bool:
    value = str(text or "").strip()
    return bool(
        re.search(
            r"(skip\s+to\s+(?:content|main content)|Product\s+Solutions\s+Resources|"
            r"login\s+contact\s+us|search\s+results?|related\s+articles?|"
            r"\u767b\u5f55|\u9996\u9875|\u5bfc\u822a|\u641c\u7d22|\u4e0b\u8f7d|\u76ee\u5f55)",
            value,
            flags=re.I,
        )
    )


def _source_host(url: str) -> str:
    match = re.search(r"https?://([^/]+)", str(url or "").strip().lower())
    return match.group(1) if match else ""


def _fact_type(item: Dict[str, Any]) -> str:
    role_text = _role_text(item).lower()
    if _metric_ready(item):
        return "metric"
    if _is_counter(item):
        return "counter"
    if _is_case(item):
        return "case"
    if re.search(r"technology|standard|patent|roadmap|maturity|技术|标准|专利|架构|路线|模型|平台|产品", role_text, flags=re.I):
        return "technology"
    if re.search(r"policy|regulation|official|filing|公告|监管|政策|年报|财报|披露", role_text, flags=re.I):
        return "source_check"
    return "directional"


def _topic_relevance(item: Dict[str, Any]) -> str:
    text = " ".join(
        str(item.get(key) or "")
        for key in ("fact", "clean_fact", "content", "summary", "title", "source_title", "metric", "indicator")
    )
    has_agent = bool(re.search(r"\bAI\s*Agent\b|agentic|智能体|企业级智能体|AI智能体", text, flags=re.I))
    has_general_ai = bool(re.search(r"人工智能|大模型|生成式AI|AI\s+", text, flags=re.I))
    if has_agent:
        return "strong"
    if has_general_ai:
        return "medium"
    return "weak"


def _distill_fact(item: Dict[str, Any], *, max_chars: int = 220) -> str:
    fact = _fact_text(item)
    fact = re.sub(r"<[^>]+>", "", fact)
    fact = re.sub(r"\[[Pp][Dd][Ff]\]\s*", "", fact)
    fact = re.sub(r"https?://\S+", "", fact)
    fact = re.sub(r"\b(?:Read page|content description)[:：]?\s*", "", fact, flags=re.I)
    fact = re.sub(r"^(?:摘要|标题|来源|事实|关键事实)[:：\s]*", "", fact).strip()
    fact = re.sub(r"^.{0,140}?（\s*）[:：]\s*", "", fact)
    fact = re.sub(r"^#{1,6}\s*", "", fact)
    fact = re.sub(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:[+-]\d{2}:\d{2})?", "", fact)
    fact = re.sub(r"\s+", " ", fact).strip(" ；;，,。")
    if not fact or _bad_fact_text(fact):
        metric = _compact(item.get("metric") or item.get("indicator"), 80)
        value = _compact(item.get("value") or item.get("display_value") or item.get("numeric_value"), 80)
        if (
            metric
            and value
            and not _invalid_metric(item)
            and not _internal_metric_name(metric)
            and _metric_value_carries_meaning(value, str(item.get("unit") or item.get("numeric_unit") or ""))
        ):
            fact = f"{metric}: {value}"
        else:
            return ""
    parts: List[str] = []
    seen = set()
    for part in re.split(r"[。；;\n]+", fact):
        part = _compact(part.strip(), max_chars)
        if not part or _bad_fact_text(part):
            continue
        if len(part) > 80 and re.search(r"登录|首页|上一篇|下一篇|分享到|Product|Solutions|Resources", part, flags=re.I):
            continue
        key = re.sub(r"\W+", "", part.lower())[:100]
        if key in seen:
            continue
        seen.add(key)
        parts.append(part)
        if len(parts) >= 2:
            break
    return "；".join(parts)[:max_chars].rstrip("；;，,。")


_UNIT_DISPLAY_MAP = {
    "percent": "%",
    "ratio": "",
    "count": "",
    "currency_cny": "",
    "currency_usd": "",
    "unknown": "",
    "": "",
}

# Pipeline role names, evidence tags, and validation fields that upstream
# extraction used to write into the metric slot. None of these are real
# indicators; rendering them produced body text like "technology_product为20"
# and "source_check为26（2025）".
_INTERNAL_METRIC_NAME_RE = re.compile(
    r"^(?:source_check|status|http_status|response_code|qualitative_fact|"
    r"technology_product|official_data|customer_case|filing|case|counter|support|metric|"
    r"事实|关键事实|竞争对比|政策目标|政策监管|技术产业链|数字数据|发展趋势|资本事件|数据点)$",
    re.I,
)


def _internal_metric_name(metric: str) -> bool:
    return bool(_INTERNAL_METRIC_NAME_RE.fullmatch(str(metric or "").strip()))


def _metric_value_carries_meaning(value: str, unit: str) -> bool:
    """A metric sentence is only renderable when the value reads as a real
    quantity: either the value text carries its own unit, or the unit field
    maps to a known display unit. Bare numbers (dates, ids, file suffixes)
    must fall through to prose."""

    text = str(value or "").strip()
    if not re.search(r"\d", text):
        return False
    if re.search(r"\d\s*(?:%|亿|万|千|元|美元|台|套|件|家|倍|人|个|吨|GWh|MWh|kWh|GW|MW|KW)", text, re.I):
        return True
    mapped = _UNIT_DISPLAY_MAP.get(str(unit or "").lower())
    return bool(mapped)


def _report_fact_sentence(item: Dict[str, Any], text: str, *, max_chars: int = 150) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    value = re.sub(r"https?://\S+", "", value)
    value = re.sub(r"\[[Pp][Dd][Ff]\]\s*", "", value)
    value = re.sub(r"\b(?:Read page|content description)[:：]?\s*", "", value, flags=re.I)
    value = value.strip(" \t\r\n：:；;，,。")
    metric = _compact(item.get("metric") or item.get("indicator"), 50)
    metric_value = _compact(item.get("value") or item.get("display_value") or item.get("numeric_value"), 40)
    unit = _compact(item.get("unit") or item.get("numeric_unit"), 20)
    scope = _compact(item.get("scope") or item.get("market_scope") or _time_or_scope(value), 40)
    subject = _compact(item.get("subject") or item.get("company") or item.get("entity"), 60)
    if (
        metric
        and metric_value
        and not _invalid_metric(item)
        and not _internal_metric_name(metric)
        and _metric_value_carries_meaning(metric_value, unit)
    ):
        prefix = f"{subject}的" if subject else ""
        normalized_unit = _UNIT_DISPLAY_MAP.get(unit.lower())
        if normalized_unit is None:
            # Unknown unit labels are internal enums (currency_usd leaked into
            # a published body as "471亿美元currency_usd"); never render them.
            normalized_unit = "" if re.fullmatch(r"[a-z0-9_]+", unit.lower() or "") else unit
        suffix = normalized_unit if normalized_unit and normalized_unit not in metric_value else ""
        tail = f"（{scope}）" if scope else ""
        metric_sentence = _compact(f"{prefix}{metric}为{metric_value}{suffix}{tail}", max_chars)
        # Extraction noise often leaves stub metric fields (metric="成本",
        # value="5") on facts whose prose is the real payload. Only replace the
        # prose when the metric sentence is substantial enough to survive the
        # downstream fact-card length gate; otherwise fall through to prose.
        if len(metric_sentence) >= 18:
            return metric_sentence
    for separator in ("：", ":"):
        if separator not in value:
            continue
        head, rest = value.split(separator, 1)
        head = head.strip()
        rest = rest.strip(" ：:；;，,。")
        if len(rest) >= 18 and (
            len(head) >= 6
            or re.search(r"报告|专题|研究|网站|Archives|Page|IDC|Gartner|政府|新闻|动态|快讯|虎嗅|研报|PDF", head, flags=re.I)
        ):
            value = rest
            break
    value = re.sub(r"\s*\|\s*", "，", value)
    value = re.sub(r"\.{2,}", "...", value)
    candidates = [
        part.strip(" ：:；;，,。")
        for part in re.split(r"[。；;\n]+", value)
        if part and part.strip(" ：:；;，,。")
    ]
    for part in candidates:
        if _bad_fact_text(part) or _navigation_or_search_text(part):
            continue
        if len(part) < 12:
            continue
        return _compact(part, max_chars)
    return _compact(value, max_chars)


def _time_or_scope(text: str) -> str:
    matches = re.findall(r"(?:20\d{2}|19\d{2})(?:[-/年]\d{1,2}(?:[-/月]\d{1,2}日?)?)?", text or "")
    if matches:
        return matches[0]
    if re.search(r"\u5168\u7403", text):
        return "\u5168\u7403"
    if re.search(r"\u4e2d\u56fd|\u56fd\u5185", text):
        return "\u4e2d\u56fd"
    return ""


PUBLISHER_TITLE_TOKENS = {
    "ijiwei",
    "36kr",
    "36氪",
    "sina",
    "sohu",
    "baidu",
    "zhihu",
    "爱集微",
    "财联社",
    "界面",
    "新浪",
    "腾讯",
    "网易",
    "搜狐",
    "百度",
    "知乎",
    "雪球",
    "虎嗅",
    "钛媒体",
    "亿欧",
    "证券时报",
    "证券日报",
    "中国证券网",
    "人民网",
    "新华社",
    "央视",
    "凤凰网",
    "公众号",
}
PUBLISHER_DOMAIN_RE = re.compile(
    r"\b(?:ijiwei|36kr|sina|qq|sohu|baidu|zhihu|caixin|cls|stcn|cnstock|xinhuanet|people|ifeng|netease)\b|[a-z0-9-]+\.(?:com|cn|net|org)",
    re.I,
)
METRIC_VARIABLE_ALLOWED_RE = re.compile(
    r"规模|增速|收入|利润|价格|成本|订单|采购|续约|复购|渗透率|份额|占比|ROI|客户|部署|流程|安全|治理|权限|可靠|竞争|玩家|场景|落地|商业化|技术",
    re.I,
)


def _looks_like_publisher_or_domain(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    lowered = text.lower()
    if any(token.lower() in lowered for token in PUBLISHER_TITLE_TOKENS):
        return True
    if PUBLISHER_DOMAIN_RE.search(lowered):
        return True
    return bool(re.search(r"[\u4e00-\u9fff]+-[a-z][a-z0-9-]{2,}", text, flags=re.I))


def _fact_subject(text: str, item: Dict[str, Any]) -> str:
    for key in ("subject", "company", "entity", "actor"):
        title = _compact(item.get(key), 60)
        if title and not _looks_like_publisher_or_domain(title):
            return title
    text = re.sub(r"^[#\s]+", "", text or "")
    for sep in ("：", ":", "，", ",", "。"):
        if sep in text:
            candidate = _compact(text.split(sep, 1)[0], 60)
            if 2 <= len(candidate) <= 60 and not _bad_fact_text(candidate) and not _looks_like_publisher_or_domain(candidate):
                return candidate
    candidate = _compact(text, 40)
    return "" if _looks_like_publisher_or_domain(candidate) else candidate


def _fact_action(text: str, fact_type: str) -> str:
    if fact_type == "metric":
        return "\u663e\u793a"
    if fact_type == "counter":
        return "\u63d0\u793a\u98ce\u9669"
    if re.search(r"\u53d1\u5e03|\u63a8\u51fa|\u4e0a\u7ebf|\u90e8\u7f72|\u91c7\u8d2d|\u4e2d\u6807|\u6295\u8d44", text or ""):
        return "\u843d\u5730"
    if fact_type == "technology":
        return "\u4f53\u73b0"
    if fact_type == "case":
        return "\u9a8c\u8bc1"
    return "\u8868\u660e"


def _fact_variable(fact_type: str) -> str:
    return {
        "metric": "\u89c4\u6a21\u4e0e\u589e\u901f",
        "case": "\u5ba2\u6237\u4e0e\u843d\u5730",
        "technology": "\u6280\u672f\u6210\u719f\u5ea6",
        "counter": "\u98ce\u9669\u8fb9\u754c",
        "source_check": "\u6765\u6e90\u6838\u9a8c",
    }.get(fact_type, "\u65b9\u5411\u6027\u4fe1\u53f7")


def _analysis_variable(item: Dict[str, Any], fact_type: str, text: str) -> str:
    metric = str(item.get("metric") or item.get("indicator") or "").strip()
    if metric and METRIC_VARIABLE_ALLOWED_RE.search(metric):
        return _compact(metric, 40)
    if fact_type == "case":
        return "\u5ba2\u6237\u843d\u5730\u4e0e\u5e94\u7528\u573a\u666f"
    if fact_type == "counter":
        if re.search(r"\u6210\u672c|ROI|\u4ef7\u503c|\u5546\u4e1a", text, flags=re.I):
            return "\u6210\u672c\u4e0e\u5546\u4e1a\u4ef7\u503c\u98ce\u9669"
        if re.search(r"\u5b89\u5168|\u6cbb\u7406|\u8d23\u4efb|\u53ef\u9760", text, flags=re.I):
            return "\u5b89\u5168\u6cbb\u7406\u4e0e\u8d23\u4efb\u8fb9\u754c"
        return "\u98ce\u9669\u89e6\u53d1\u6761\u4ef6"
    if fact_type == "technology":
        return "\u6280\u672f\u6210\u719f\u5ea6\u4e0e\u90e8\u7f72\u7ea6\u675f"
    if fact_type == "metric":
        return "\u6307\u6807\u53e3\u5f84\u4e0e\u53ef\u6bd4\u6027"
    if fact_type == "source_check":
        return "\u6765\u6e90\u53ef\u6838\u9a8c\u6027"
    if re.search(r"\u7ade\u4e89|\u73a9\u5bb6|\u6e20\u9053|\u5e73\u53f0|\u4f9b\u7ed9", text, flags=re.I):
        return "\u7ade\u4e89\u4e0e\u4f9b\u7ed9\u53d8\u91cf"
    if re.search(r"\u9700\u6c42|\u5ba2\u6237|\u4ed8\u8d39|\u4f7f\u7528", text, flags=re.I):
        return "\u9700\u6c42\u4e0e\u4ed8\u8d39\u53d8\u91cf"
    return _fact_variable(fact_type)


def _block_affinity(item: Dict[str, Any], fact_type: str, text: str) -> List[str]:
    role_text = _role_text(item)
    combined = f"{role_text} {text}"
    affinity: List[str] = []
    if fact_type == "metric" or re.search(r"\u89c4\u6a21|\u589e\u901f|ROI|\u6210\u672c|\u4ef7\u683c|\u8425\u6536|\u5229\u6da6|%|\u4ebf|\u4e07", combined, flags=re.I):
        affinity.extend(["metric_reconciliation", "unit_economics"])
    if fact_type == "technology" or re.search(r"\u6280\u672f|\u6807\u51c6|\u67b6\u6784|\u6a21\u578b|\u5de5\u5177|\u5e73\u53f0|\u90e8\u7f72|\u53ef\u9760|\u5b89\u5168|\u6cbb\u7406", combined, flags=re.I):
        affinity.append("technology_maturity")
    if fact_type == "counter" or re.search(r"\u98ce\u9669|\u5931\u8d25|\u53d6\u6d88|\u6210\u672c|\u8d23\u4efb|\u8fb9\u754c|guardrail|risk", combined, flags=re.I):
        affinity.extend(["risk_trigger", "scenario_analysis", "verification_checklist"])
    if fact_type == "case" or re.search(r"\u5ba2\u6237|\u6848\u4f8b|\u8ba2\u5355|\u4e2d\u6807|\u91c7\u8d2d|\u843d\u5730|\u5e94\u7528|\u5de5\u4f5c\u6d41|workflow|customer|case", combined, flags=re.I):
        affinity.extend(["customer_painpoint_matrix", "case_comparison", "competitive_positioning"])
    if re.search(r"\u7ade\u4e89|\u73a9\u5bb6|\u4f9b\u7ed9|\u6e20\u9053|\u5e73\u53f0|vendor|platform", combined, flags=re.I):
        affinity.append("competitive_positioning")
    if not affinity:
        affinity.extend(["thesis", "evidence_matrix"])
    result: List[str] = []
    for item_value in affinity:
        if item_value not in result:
            result.append(item_value)
    return result


def _claim_strength_hint(item: Dict[str, Any]) -> str:
    level = _source_level(item)
    verified = str(item.get("source_verification_status") or item.get("verification_status") or "").strip().lower()
    if level in {"A", "B"} and verified in {"readpage_verified", "document_verified"}:
        return "strong"
    if level in {"A", "B"}:
        return "moderate"
    if level == "C" and _traceable(item):
        return "directional"
    return "weak"


def _public_fact_card(item: Dict[str, Any], distilled: str) -> Dict[str, Any]:
    text = _report_fact_sentence(item, distilled, max_chars=150)
    if not text or len(text) < 18 or _bad_fact_text(text):
        return {}
    if re.search(r"\u4ec0\u4e48\u662f|\u6700\u65b0\u62a5\u9053|\u641c\u7d22|\u9996\u9875|Google\s+Patents|^\s*#", text, flags=re.I):
        return {}
    fact_type = _fact_type(item)
    ref = _evidence_ref(item) or str(item.get("source_ref") or item.get("citation_ref") or "").strip()
    source_level = _source_level(item)
    variable = _analysis_variable(item, fact_type, text)
    affinity = _block_affinity(item, fact_type, text)
    return {
        "subject": _fact_subject(text, item),
        "action": _fact_action(text, fact_type),
        "action_or_signal": _fact_action(text, fact_type),
        "object": text,
        "fact": text,
        "distilled_fact": text,
        "time_or_scope": _time_or_scope(text),
        "variable": variable,
        "analysis_variable": variable,
        "block_affinity": affinity,
        "fact_type": fact_type,
        "source_ref": ref,
        "source_level": source_level,
        "claim_strength_hint": _claim_strength_hint(item),
        "directional_only": source_level == "C" or _claim_strength_hint(item) in {"directional", "weak"},
    }


def _public_fact_quality(item: Dict[str, Any]) -> Dict[str, Any]:
    """Two-tier quality verdict for the public fact pool.

    Hard rejections are reserved for facts that can never be written (garbage
    text, untraceable, blocked hosts). Weak-but-real facts (suspicious metric
    fields, aggregator sources, identity doubts) are kept as directional-only
    cards instead of being discarded: dropping them is what previously starved
    6/8 chapters down to zero fact cards in fail-open rebuilds.
    """

    fact = _fact_text(item)
    distilled = _distill_fact(item)
    level = _source_level(item)
    url = _source_url(item)
    host = _source_host(url)
    traceable = _traceable(item)
    rejection: List[str] = []
    downgrade: List[str] = []
    if not distilled:
        rejection.append("empty_or_bad_fact")
    if _invalid_metric(item):
        # The metric fields are extraction garbage, but the fact text itself
        # may still be a usable qualitative signal; demote instead of drop.
        downgrade.append("invalid_metric")
    if _fact_type(item) == "metric" and (
        not str(item.get("subject") or item.get("company") or item.get("entity") or "").strip()
        or not (str(item.get("scope") or item.get("market_scope") or "").strip() or _time_or_scope(distilled))
    ):
        if len(distilled) >= 40:
            # The prose carries the fact and the metric fields are extraction
            # residue: keep the prose as a directional card.
            downgrade.append("no_subject_or_scope")
        else:
            # A bare metric fragment with no subject or scope is unusable.
            rejection.append("no_subject_or_scope")
    if _navigation_or_search_text(fact):
        rejection.append("navigation_text")
    if _bad_fact_text(fact):
        rejection.append("bad_fact_pattern")
    if _source_identity_bad(item):
        downgrade.append("source_identity_bad")
    if _low_quality_source(item):
        downgrade.append("low_quality_source")
    if _topic_relevance(item) == "weak":
        rejection.append("weak_topic_relevance")
    if level == "D":
        rejection.append("source_level_d")
    if not traceable:
        rejection.append("not_traceable")
    if host and re.search(r"(?:twitter|x|instagram|facebook|baike|baijiahao|csdn|cnblogs|juejin)\.", host, flags=re.I):
        rejection.append("blocked_host")
    fact_card = _public_fact_card(item, distilled) if not rejection else {}
    if fact_card and downgrade:
        if "invalid_metric" in downgrade or "no_subject_or_scope" in downgrade:
            # Never let an unverifiable metric masquerade as one downstream.
            if fact_card.get("fact_type") == "metric":
                fact_card["fact_type"] = "directional"
        fact_card["directional_only"] = True
        fact_card["claim_strength_hint"] = "directional"
        fact_card["downgrade_reasons"] = list(downgrade)
    if fact_card:
        required_missing = [
            key
            for key in ("subject", "action_or_signal", "variable", "distilled_fact", "source_ref", "source_level", "block_affinity")
            if not fact_card.get(key)
        ]
        if required_missing:
            rejection.append("fact_card_required_field_missing")
        if fact_card.get("fact_type") == "metric" and not fact_card.get("time_or_scope"):
            rejection.append("no_subject_or_scope")
    if not fact_card:
        rejection.append("fact_card_missing")
    eligible = not rejection and bool(fact_card)
    eligible_for_citation = bool(eligible and traceable and level in {"A", "B", "C"})
    return {
        "eligible_for_report": eligible,
        "eligible_for_citation": eligible_for_citation,
        "fact_type": _fact_type(item),
        "topic_relevance": _topic_relevance(item),
        "rejection_reason": rejection,
        "downgrade_reason": downgrade,
        "distilled_fact": distilled,
        "public_fact_card": fact_card,
    }


def _source_url(item: Dict[str, Any]) -> str:
    source = _as_dict(item.get("source"))
    return str(
        item.get("source_url")
        or item.get("url")
        or item.get("link")
        or source.get("url")
        or source.get("source_url")
        or ""
    ).strip()


def _source_level(item: Dict[str, Any]) -> str:
    return str(item.get("source_level") or item.get("credibility") or item.get("source_grade") or "C").strip().upper()


def _traceable(item: Dict[str, Any]) -> bool:
    return bool(
        _source_url(item)
        or str(item.get("document_id") or item.get("doc_id") or item.get("page_ref") or "").strip()
    )


def _source_identity_bad(item: Dict[str, Any]) -> bool:
    haystack = " ".join(
        str(item.get(key) or "")
        for key in (
            "source_url",
            "url",
            "link",
            "title",
            "source_title",
            "publisher",
            "source",
            "source_ref",
            "citation_ref",
            "ref",
        )
    )
    if item.get("source_title_url_mismatch_suspected"):
        return True
    if re.search(r"\bIQS\s*来源\b|^IQS来源$", haystack, flags=re.I):
        return True
    if re.search(r"example\.(?:com|gov|org)", haystack, flags=re.I):
        return True
    if not _traceable(item):
        return True
    return False


def _low_quality_source(item: Dict[str, Any]) -> bool:
    haystack = " ".join(
        str(item.get(key) or "")
        for key in ("source_url", "url", "title", "source_title", "publisher", "source")
    )
    source_type = str(item.get("source_type") or item.get("type") or "").strip().lower()
    if source_type in {"self_media", "social", "forum", "wiki", "seo", "search_page", "aggregator"}:
        return True
    if _source_level(item) == "D":
        return True
    if _source_identity_bad(item):
        return True
    return any(re.search(pattern, haystack, flags=re.I) for pattern in LOW_QUALITY_SOURCE_PATTERNS)


def _invalid_metric(item: Dict[str, Any]) -> bool:
    metric = str(item.get("metric") or item.get("indicator") or "").strip()
    value = str(item.get("value") or item.get("display_value") or item.get("numeric_value") or "").strip()
    unit = str(item.get("unit") or item.get("numeric_unit") or "").strip().lower()
    fact = _fact_text(item)
    metric_lower = metric.lower()
    if str(item.get("metric_validation_status") or "").strip().lower() == "invalid":
        return True
    if unit == "unknown" and (metric or value):
        # An unknown unit only invalidates an actual metric claim; plain text
        # facts routinely carry unit="unknown" from upstream normalization and
        # must not be culled wholesale for it.
        return True
    if metric_lower in {"source_check", "status", "http_status", "response_code"} and re.fullmatch(r"[1-5]\d{2}", value):
        return True
    if re.search(r"\bsource_check\s*[:=]\s*[1-5]\d{2}\b", fact, flags=re.I):
        return True
    if value and re.fullmatch(r"-?\d{1,3}(?:\.0)?", value) and re.search(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|T\d{1,2}:\d{2}", fact):
        return True
    if metric in {"\u5173\u952e\u4e8b\u5b9e", "\u653f\u7b56\u76d1\u7ba1", "\u653f\u7b56\u76ee\u6807"} and re.fullmatch(r"-?\d{1,3}(?:\.0)?", value):
        return True
    if re.search(r"\u653f\u7b56|\u76ee\u6807|\u76d1\u7ba1", metric) and re.match(r"-\d", value):
        return True
    if re.search(r"\u6210\u672c", metric) and (re.search(r"\u5bb6$", value) or not fact):
        return True
    if re.search(r"\u5e02\u573a\u89c4\u6a21|\u878d\u8d44", metric) and re.search(r"%", value):
        return True
    return False


def _metric_ready(item: Dict[str, Any]) -> bool:
    metric = str(item.get("metric") or item.get("indicator") or "").strip()
    value = str(item.get("value") or item.get("display_value") or item.get("numeric_value") or "").strip()
    if not metric or not value or _invalid_metric(item):
        return False
    return bool(re.search(r"\d", value) or _as_list(item.get("numeric_values")))


def _ref_values(item: Dict[str, Any]) -> List[str]:
    values: List[str] = []
    for key in ("evidence_id", "id", "ref", "source_ref", "citation_ref", "source_id", "document_id"):
        value = str(item.get(key) or "").strip()
        if value:
            values.append(value)
    url = _source_url(item)
    if url:
        values.append(url)
    return values


def _normalize_ref(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return re.sub(r"^\[|\]$", "", text).strip().lower()


def _evidence_ref(item: Dict[str, Any]) -> str:
    for key in ("source_ref", "citation_ref", "ref", "evidence_id", "id"):
        value = str(item.get(key) or "").strip()
        if value:
            return value
    return _source_url(item)


def _dedupe_items(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        key = _normalize_ref(_evidence_ref(item)) or re.sub(r"\W+", "", _fact_text(item).lower())[:120]
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _source_keys(item: Dict[str, Any]) -> List[str]:
    source = _as_dict(item.get("source"))
    values = []
    for key in ("ref", "source_ref", "citation_ref", "source_id", "id"):
        values.append(str(item.get(key) or source.get(key) or "").strip())
    url = _source_url(item)
    if url:
        values.append(url)
    source_id = str(item.get("source_id") or source.get("source_id") or "").strip()
    if re.fullmatch(r"\d{1,3}", source_id):
        values.append(f"[{source_id}]")
    return [_normalize_ref(value) for value in values if _normalize_ref(value)]


def _source_registry_lookup(source_registry: Optional[Sequence[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    registry = [dict(source) for source in list(source_registry or []) if isinstance(source, dict)]
    title_hosts: Dict[str, set[str]] = {}
    for source in registry:
        title = re.sub(r"\s+", " ", str(source.get("title") or source.get("source_title") or "").strip()).lower()
        url = str(source.get("url") or source.get("source_url") or "").strip().lower()
        host_match = re.search(r"https?://([^/]+)", url)
        host = host_match.group(1) if host_match else ""
        if title and host:
            title_hosts.setdefault(title, set()).add(host)
    for source in registry:
        title = re.sub(r"\s+", " ", str(source.get("title") or source.get("source_title") or "").strip()).lower()
        if title and len(title_hosts.get(title, set())) > 1:
            source["source_title_url_mismatch_suspected"] = True
        for key in _source_keys(source):
            lookup.setdefault(key, source)
    return lookup


def _registry_source_for_item(item: Dict[str, Any], lookup: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    for key in _source_keys(item):
        if key in lookup:
            return lookup[key]
    return {}


def _normalize_item(item: Dict[str, Any], source_lookup: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    copied = dict(item)
    registry_source = _registry_source_for_item(copied, source_lookup or {})
    if registry_source:
        copied["source_registry_ref"] = registry_source.get("ref") or copied.get("source_registry_ref")
        copied["source_title_url_mismatch_suspected"] = bool(
            copied.get("source_title_url_mismatch_suspected")
            or registry_source.get("source_title_url_mismatch_suspected")
        )
        for target, candidates in {
            "source_url": ("url", "source_url"),
            "source_title": ("title", "source_title"),
            "publisher": ("publisher", "source"),
            "source_level": ("source_level", "credibility"),
            "source_type": ("source_type", "type"),
        }.items():
            if copied.get(target):
                continue
            for candidate in candidates:
                value = registry_source.get(candidate)
                if value:
                    copied[target] = value
                    break
    copied["fact"] = _fact_text(copied)
    copied.setdefault("ref", copied.get("evidence_id") or copied.get("source_ref") or copied.get("citation_ref") or "")
    copied.setdefault("source_ref", copied.get("source_ref") or copied.get("citation_ref") or copied.get("ref") or "")
    copied.setdefault("source_level", _source_level(copied))
    copied["source_traceable"] = _traceable(copied)
    copied["metric_ready"] = _metric_ready(copied)
    quality = _public_fact_quality(copied)
    copied["public_fact_quality"] = quality
    copied["distilled_fact"] = quality.get("distilled_fact") or ""
    copied["public_fact_card"] = _as_dict(quality.get("public_fact_card"))
    copied["eligible_for_report"] = bool(quality.get("eligible_for_report"))
    copied["eligible_for_citation"] = bool(quality.get("eligible_for_citation"))
    copied["fact_type"] = quality.get("fact_type")
    return copied


def _normalized_seed_candidates(evidence_package: Dict[str, Any], source_lookup: Optional[Dict[str, Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for key in ("analysis_ready_evidence", "clean_evidence_list", "normalized_evidence", "raw_data_points"):
        for item in _as_list(evidence_package.get(key)):
            if not isinstance(item, dict):
                continue
            normalized = _normalize_item(item, source_lookup)
            normalized["source_collection"] = key
            candidates.append(normalized)
    return _dedupe_items(candidates)


def _seed_items(evidence_package: Dict[str, Any], source_lookup: Optional[Dict[str, Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    return [
        item
        for item in _normalized_seed_candidates(evidence_package, source_lookup)
        if bool(_as_dict(item.get("public_fact_quality")).get("eligible_for_report"))
    ]


def _public_filter_summary(candidates: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = 0
    eligible = 0
    citation_eligible = 0
    fact_cards = 0
    invalid_metric = 0
    downgraded = 0
    reasons: Dict[str, int] = {}
    downgrade_reasons: Dict[str, int] = {}
    fact_types: Dict[str, int] = {}
    for item in candidates:
        if not isinstance(item, dict):
            continue
        total += 1
        quality = _as_dict(item.get("public_fact_quality"))
        if quality.get("eligible_for_report"):
            eligible += 1
        if quality.get("eligible_for_citation"):
            citation_eligible += 1
        if _as_dict(quality.get("public_fact_card")):
            fact_cards += 1
        if "invalid_metric" in _as_list(quality.get("rejection_reason")) or "invalid_metric" in _as_list(quality.get("downgrade_reason")):
            invalid_metric += 1
        if _as_list(quality.get("downgrade_reason")):
            downgraded += 1
        fact_type = str(quality.get("fact_type") or "unknown")
        fact_types[fact_type] = fact_types.get(fact_type, 0) + 1
        for reason in _as_list(quality.get("rejection_reason")):
            reason_text = str(reason or "").strip() or "unknown"
            reasons[reason_text] = reasons.get(reason_text, 0) + 1
        for reason in _as_list(quality.get("downgrade_reason")):
            reason_text = str(reason or "").strip() or "unknown"
            downgrade_reasons[reason_text] = downgrade_reasons.get(reason_text, 0) + 1
    return {
        "candidate_fact_count": total,
        "eligible_fact_count": eligible,
        "eligible_citation_count": citation_eligible,
        "fact_card_count": fact_cards,
        "filtered_fact_count": max(0, total - eligible),
        "invalid_metric_filtered_count": invalid_metric,
        "downgraded_fact_count": downgraded,
        "rejection_reasons": reasons,
        "downgrade_reasons": downgrade_reasons,
        "fact_type_distribution": fact_types,
    }


def _items_from_existing_chapter(
    chapter: Dict[str, Any],
    chapter_id: str,
    source_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for key in (
        "core_evidence",
        "supporting_evidence",
        "counter_evidence",
        "metric_evidence",
        "case_evidence",
        "directional_evidence",
        "sample_evidence",
    ):
        for item in _as_list(chapter.get(key)):
            if not isinstance(item, dict):
                continue
            fact = _fact_text(item)
            if _bad_fact_text(fact) or _invalid_metric(item):
                continue
            copied = dict(item)
            copied.setdefault("chapter_id", chapter_id)
            normalized = _normalize_item(copied, source_lookup)
            quality = _as_dict(normalized.get("public_fact_quality"))
            if not bool(quality.get("eligible_for_report")):
                continue
            items.append(normalized)
    return _dedupe_items(items)


def _lookup_by_ref(seeds: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for item in seeds:
        for ref in _ref_values(item):
            key = _normalize_ref(ref)
            if key and key not in lookup:
                lookup[key] = item
    return lookup


def _text_key(value: Any) -> str:
    return re.sub(r"[\s\W_]+", "", str(value or "").lower())


def _chapter_terms(chapter: Dict[str, Any]) -> List[str]:
    text = " ".join(
        str(chapter.get(key) or "")
        for key in ("chapter_title", "chapter_question", "title", "unit_title", "core_question")
    )
    terms = set()
    for token in re.findall(r"[A-Za-z][A-Za-z0-9+.-]{2,}|[\u4e00-\u9fff]{2,}", text):
        if token.lower() in GENERIC_CHAPTER_TERMS or token.lower() in GENERIC_ENGLISH_CHAPTER_TERMS:
            continue
        terms.add(token.lower())
    return list(terms)


def _with_binding(item: Dict[str, Any], *, reason: str, score: int, chapter_id: str) -> Dict[str, Any]:
    copied = dict(item)
    copied["binding_reason"] = reason
    copied["binding_score"] = score
    copied["bound_chapter_id"] = chapter_id
    copied.setdefault("chapter_id", chapter_id)
    return copied


def _hydrated_layer_count(package: Dict[str, Any]) -> int:
    return sum(len(_as_list(package.get(key))) for key in EVIDENCE_LAYER_KEYS)


def _chapter_identity(chapter: Dict[str, Any], index: int) -> Tuple[str, str, str]:
    chapter_id = str(chapter.get("chapter_id") or chapter.get("unit_id") or chapter.get("id") or f"ch_{index:02d}").strip()
    title = str(chapter.get("chapter_title") or chapter.get("unit_title") or chapter.get("title") or f"\u7ae0\u8282 {index}").strip()
    question = str(chapter.get("chapter_question") or chapter.get("core_question") or chapter.get("chapter_role") or title).strip()
    return chapter_id, title, question


def _diagnostic_payload(evidence_analysis_by_chapter: Dict[str, Any], chapter_id: str, title: str) -> Dict[str, Any]:
    for key in (chapter_id, title):
        payload = _as_dict(evidence_analysis_by_chapter.get(key))
        if payload:
            return payload
    title_key = _text_key(title)
    for key, payload in evidence_analysis_by_chapter.items():
        if title_key and (_text_key(key) in title_key or title_key in _text_key(key)):
            return _as_dict(payload)
    return {}


def _refs_from_diagnostics(payload: Dict[str, Any]) -> List[str]:
    refs: List[str] = []
    for key in ("sample_evidence_refs", "evidence_refs", "source_refs", "claim_ready_evidence_refs"):
        refs.extend(str(ref or "").strip() for ref in _as_list(payload.get(key)) if str(ref or "").strip())
    return refs


def _item_chapter_score(item: Dict[str, Any], chapter: Dict[str, Any], chapter_id: str, title: str, terms: Sequence[str]) -> int:
    score = 0
    dim = str(item.get("chapter_id") or item.get("dimension") or item.get("hypothesis_id") or item.get("dimension_id") or "").strip()
    if dim == chapter_id:
        score += 100
    dim_key = _text_key(dim)
    title_key = _text_key(title)
    if dim_key and title_key and (dim_key in title_key or title_key in dim_key):
        score += 80
    fact_key = _text_key(_fact_text(item))
    for term in terms:
        if _text_key(term) and _text_key(term) in fact_key:
            score += 6
    metric = str(item.get("metric") or item.get("indicator") or "").lower()
    if any(term in metric for term in terms):
        score += 10
    return score


def _chapter_term_matches(item: Dict[str, Any], terms: Sequence[str]) -> List[str]:
    if not terms:
        return []
    evidence_key = _text_key(
        " ".join(
            str(item.get(key) or "")
            for key in (
                "fact",
                "distilled_fact",
                "clean_fact",
                "content",
                "summary",
                "title",
                "source_title",
                "metric",
                "indicator",
            )
        )
    )
    matches: List[str] = []
    for term in terms:
        term_key = _text_key(term)
        if term_key and term_key in evidence_key:
            matches.append(str(term))
    return matches


def _chapter_relevance_rejection_reason(item: Dict[str, Any], terms: Sequence[str]) -> str:
    """Reject generic source/directional facts that only match by chapter_id.

    Exact chapter_id is useful for recall, but it is not enough to prove that a
    generic source-check fact semantically supports the chapter. Metric/case/
    counter/technology facts keep their role-based path; broad source-check and
    directional facts need at least one non-generic chapter term match.
    """

    if not terms or _chapter_term_matches(item, terms):
        return ""
    fact_type = str(item.get("fact_type") or _as_dict(item.get("public_fact_quality")).get("fact_type") or _fact_type(item)).strip().lower()
    if fact_type in {"metric", "case", "counter", "technology"}:
        return ""
    if fact_type in {"source_check", "directional"}:
        return "weak_chapter_relevance"
    allowed_use = str(item.get("allowed_use") or "").strip().lower()
    if allowed_use in {"supporting", "core_claim"} and str(item.get("binding_reason") or "") == "chapter_id":
        return "weak_chapter_relevance"
    return ""


def _filter_chapter_relevant_items(items: Sequence[Dict[str, Any]], terms: Sequence[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    kept: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for item in items:
        reason = _chapter_relevance_rejection_reason(item, terms)
        if reason:
            copied = dict(item)
            copied["chapter_relevance_status"] = "rejected"
            copied["chapter_relevance_rejection_reason"] = reason
            rejected.append(copied)
        else:
            kept.append(item)
    return kept, rejected


def _binding_reason(score: int) -> str:
    if score >= 100:
        return "chapter_id"
    if score >= 80:
        return "dimension_title"
    if score >= CHAPTER_MATCH_MIN_SCORE:
        return "chapter_terms"
    return "unmatched"


def _role_text(item: Dict[str, Any]) -> str:
    return " ".join(
        str(item.get(key) or "")
        for key in ("evidence_role", "proof_role", "allowed_use", "metric", "indicator", "fact", "content", "evidence", "summary")
    )


def _is_counter(item: Dict[str, Any]) -> bool:
    return bool(re.search(r"\u98ce\u9669|\u53cd\u8bc1|\u5931\u8d25|\u4e0b\u964d|\u4e0d\u53ca\u9884\u671f|\u8fb9\u754c|counter|risk", _role_text(item), flags=re.I))


def _is_case(item: Dict[str, Any]) -> bool:
    return bool(re.search(r"\u5ba2\u6237|\u6848\u4f8b|\u8ba2\u5355|\u4e2d\u6807|\u91c7\u8d2d|\u843d\u5730|\u90e8\u7f72|case", _role_text(item), flags=re.I))


def _is_core(item: Dict[str, Any]) -> bool:
    level = _source_level(item)
    allowed = str(item.get("allowed_use") or item.get("evidence_role") or "").lower()
    return level in {"A", "B"} and _traceable(item) and not _is_counter(item) and ("clue" not in allowed)


def _rank_items(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def score(item: Dict[str, Any]) -> Tuple[int, str]:
        level = _source_level(item)
        level_score = {"A": 40, "B": 30, "C": 15}.get(level, 0)
        return (
            level_score
            + (20 if _traceable(item) else 0)
            + (15 if _metric_ready(item) else 0)
            + min(len(_fact_text(item)) // 80, 8),
            _evidence_ref(item),
        )

    return sorted(_dedupe_items(items), key=score, reverse=True)


def _layer_evidence(items: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    ranked = _rank_items(items)
    core = [item for item in ranked if _is_core(item)][:CORE_LIMIT]
    used = {_normalize_ref(_evidence_ref(item)) for item in core}
    counters = [item for item in ranked if _is_counter(item) and _normalize_ref(_evidence_ref(item)) not in used][:COUNTER_LIMIT]
    used.update(_normalize_ref(_evidence_ref(item)) for item in counters)
    metrics = [item for item in ranked if _metric_ready(item)][:METRIC_LIMIT]
    used.update(_normalize_ref(_evidence_ref(item)) for item in metrics)
    cases = [item for item in ranked if _is_case(item)][:CASE_LIMIT]
    used.update(_normalize_ref(_evidence_ref(item)) for item in cases)
    directional = [
        item
        for item in ranked
        if _normalize_ref(_evidence_ref(item)) not in used
        and _traceable(item)
        and (
            _source_level(item) == "C"
            or str(item.get("allowed_use") or "").strip().lower() == "directional_signal"
        )
    ][:DIRECTIONAL_LIMIT]
    used.update(_normalize_ref(_evidence_ref(item)) for item in directional)
    supporting = [
        item
        for item in ranked
        if _normalize_ref(_evidence_ref(item)) not in used
        and _traceable(item)
        and _source_level(item) in {"A", "B"}
    ][:SUPPORT_LIMIT]
    used.update(_normalize_ref(_evidence_ref(item)) for item in supporting)
    def mark_layer(layer_items: Sequence[Dict[str, Any]], layer: str, role: str, allowed_use: str = "") -> List[Dict[str, Any]]:
        marked: List[Dict[str, Any]] = []
        for item in layer_items:
            copied = dict(item)
            copied["chapter_evidence_layer"] = layer
            copied.setdefault("evidence_role", role)
            if allowed_use:
                copied.setdefault("allowed_use", allowed_use)
            elif _source_level(copied) == "C":
                copied.setdefault("allowed_use", "directional_signal")
            marked.append(copied)
        return marked

    core = mark_layer(core, "core_evidence", "core_claim", "core_claim")
    supporting = mark_layer(supporting, "supporting_evidence", "supporting")
    counters = mark_layer(counters, "counter_evidence", "counter", "counter_evidence")
    metrics = mark_layer(metrics, "metric_evidence", "supporting", "supporting")
    cases = mark_layer(cases, "case_evidence", "supporting", "directional_signal")
    directional = mark_layer(directional, "directional_evidence", "clue", "directional_signal")
    sample = _dedupe_items([*core, *supporting, *metrics, *cases, *counters, *directional])[:12]
    return {
        "core_evidence": core,
        "supporting_evidence": supporting,
        "counter_evidence": counters,
        "metric_evidence": metrics,
        "case_evidence": cases,
        "directional_evidence": directional,
        "sample_evidence": sample,
    }


def _fact_cards_from_layers(layered: Dict[str, List[Dict[str, Any]]], *, limit: int = 8) -> List[Dict[str, Any]]:
    cards: List[Dict[str, Any]] = []
    seen = set()
    for key in ("core_evidence", "metric_evidence", "case_evidence", "counter_evidence", "supporting_evidence", "directional_evidence"):
        for item in _as_list(layered.get(key)):
            if not isinstance(item, dict):
                continue
            card = _as_dict(item.get("public_fact_card") or _as_dict(item.get("public_fact_quality")).get("public_fact_card"))
            if not card:
                continue
            card = dict(card)
            card.setdefault("evidence_layer", key)
            ref = str(card.get("source_ref") or _evidence_ref(item) or "").strip()
            card["source_ref"] = ref
            dedupe_key = (ref, re.sub(r"\W+", "", str(card.get("fact") or "").lower())[:120])
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            cards.append(card)
            if len(cards) >= limit:
                return cards
    return cards


def _chapter_analysis_from_fact_cards(chapter: Dict[str, Any], layered: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    cards = _fact_cards_from_layers(layered, limit=8)
    if not cards:
        return {
            "chapter_analysis_valid": False,
            "fact_card_count": 0,
            "dropped_template_fallback_count": 1,
            "fact_cards": [],
        }
    strong_cards = [card for card in cards if str(card.get("claim_strength_hint") or "") in {"strong", "moderate"}]
    directional_cards = [card for card in cards if str(card.get("claim_strength_hint") or "") in {"directional", "weak"}]
    basis_cards = cards[:4]
    evidence_basis = [
        _compact(str(card.get("fact") or card.get("object") or ""), 180)
        for card in basis_cards
        if str(card.get("fact") or card.get("object") or "").strip()
    ]
    thesis_strength = "moderate" if strong_cards else "directional"
    variables: List[str] = []
    block_counts: Dict[str, int] = {}
    for card in cards:
        variable = str(card.get("analysis_variable") or card.get("variable") or "").strip()
        if variable and variable not in variables:
            variables.append(variable)
        for block in _as_list(card.get("block_affinity")):
            block_text = str(block or "").strip()
            if block_text:
                block_counts[block_text] = block_counts.get(block_text, 0) + 1
    return {
        "chapter_analysis_valid": True,
        "fact_card_count": len(cards),
        "directional_fact_card_count": len(directional_cards),
        "strong_fact_card_count": len(strong_cards),
        "chapter_thesis": "",
        "evidence_basis": evidence_basis,
        "mechanism": "",
        "boundary": "",
        "analysis_variables": variables[:4],
        "block_fact_card_counts": block_counts,
        "used_fact_refs": [card.get("source_ref") for card in cards if card.get("source_ref")],
        "claim_strength": thesis_strength,
        "fact_cards": cards,
    }


def _existing_chapters(report_blueprint: Dict[str, Any], existing: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    existing_by_id = {
        str(item.get("chapter_id") or "").strip(): dict(item)
        for item in existing
        if isinstance(item, dict) and str(item.get("chapter_id") or "").strip()
    }
    chapters = []
    for index, chapter in enumerate(_as_list(report_blueprint.get("chapters")), start=1):
        if not isinstance(chapter, dict):
            continue
        chapter_id, title, question = _chapter_identity(chapter, index)
        payload = existing_by_id.get(chapter_id, {})
        payload.update({"chapter_id": chapter_id, "chapter_title": title, "chapter_question": question})
        chapters.append(payload)
    if chapters:
        return chapters
    return [dict(item) for item in existing if isinstance(item, dict)]


def build_chapter_evidence_packages_from_evidence_package(
    *,
    report_blueprint: Dict[str, Any],
    evidence_package: Dict[str, Any],
    existing_chapter_evidence_packages: Optional[Sequence[Dict[str, Any]]] = None,
    source_registry: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Build full per-chapter evidence packages from the central evidence pool.

    This is intentionally deterministic and conservative: it never upgrades
    evidence quality, and it records unresolved refs instead of silently
    discarding them.
    """

    evidence_package = _as_dict(evidence_package)
    source_lookup = _source_registry_lookup(source_registry)
    evidence_analysis_by_chapter = _as_dict(evidence_package.get("evidence_analysis_by_chapter"))
    seed_candidates = _normalized_seed_candidates(evidence_package, source_lookup)
    public_filter_summary = _public_filter_summary(seed_candidates)
    seeds = [
        item
        for item in seed_candidates
        if bool(_as_dict(item.get("public_fact_quality")).get("eligible_for_report"))
    ]
    lookup = _lookup_by_ref(seeds)
    chapters = _existing_chapters(_as_dict(report_blueprint), list(existing_chapter_evidence_packages or []))
    if not chapters:
        return []
    result: List[Dict[str, Any]] = []
    for index, chapter in enumerate(chapters, start=1):
        chapter = dict(chapter)
        chapter_id, title, question = _chapter_identity(chapter, index)
        diagnostics = _diagnostic_payload(evidence_analysis_by_chapter, chapter_id, title)
        resolved: List[Dict[str, Any]] = []
        unresolved_refs: List[str] = []
        for ref in _refs_from_diagnostics(diagnostics):
            item = lookup.get(_normalize_ref(ref))
            if item:
                resolved.append(_with_binding(item, reason="evidence_analysis_ref", score=120, chapter_id=chapter_id))
            else:
                unresolved_refs.append(ref)
        terms = _chapter_terms({"chapter_title": title, "chapter_question": question})
        scored = [
            (score, _with_binding(item, reason=_binding_reason(score), score=score, chapter_id=chapter_id))
            for item in seeds
            for score in [_item_chapter_score(item, chapter, chapter_id, title, terms)]
            if score >= CHAPTER_MATCH_MIN_SCORE
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        existing_items = [
            _with_binding(item, reason="existing_chapter_package", score=110, chapter_id=chapter_id)
            for item in _items_from_existing_chapter(chapter, chapter_id, source_lookup)
        ]
        matched_before_relevance = _dedupe_items([*resolved, *existing_items, *[item for _, item in scored]])
        matched, chapter_relevance_rejected = _filter_chapter_relevant_items(matched_before_relevance, terms)
        layered = _layer_evidence(matched)
        chapter_analysis = _chapter_analysis_from_fact_cards(chapter, layered)
        hydrated_count = sum(len(layered.get(key, [])) for key in EVIDENCE_LAYER_KEYS)
        chapter_public_filter_summary = _public_filter_summary(matched)
        writable_fact_count = int(chapter_public_filter_summary.get("eligible_fact_count") or 0)
        eligible_citation_count = int(chapter_public_filter_summary.get("eligible_citation_count") or 0)
        binding_reasons: Dict[str, int] = {}
        for item in matched:
            reason = str(item.get("binding_reason") or "unknown")
            binding_reasons[reason] = binding_reasons.get(reason, 0) + 1
        layer_counts = {
            key: len(value)
            for key, value in layered.items()
            if isinstance(value, list)
        }
        binding_funnel = {
            "schema_version": "chapter_evidence_binding_funnel_v1",
            "candidate_fact_count": int(public_filter_summary.get("candidate_fact_count") or 0),
            "eligible_fact_count": int(public_filter_summary.get("eligible_fact_count") or 0),
            "filtered_fact_count": int(public_filter_summary.get("filtered_fact_count") or 0),
            "resolved_diagnostic_ref_count": len(resolved),
            "unresolved_ref_count": len(unresolved_refs),
            "scored_match_count": len(scored),
            "existing_chapter_evidence_count": len(existing_items),
            "matched_before_relevance_count": len(matched_before_relevance),
            "relevance_rejected_count": len(chapter_relevance_rejected),
            "matched_after_relevance_count": len(matched),
            "hydrated_evidence_count": hydrated_count,
            "writable_fact_count": writable_fact_count,
            "eligible_citation_count": eligible_citation_count,
            "binding_reasons": binding_reasons,
            "layer_counts": layer_counts,
        }
        metadata = _as_dict(chapter.get("metadata"))
        metadata["chapter_evidence_rebuilt"] = True
        metadata["hydrated_evidence"] = bool(hydrated_count)
        metadata["hydrated_evidence_count"] = hydrated_count
        metadata["source_pool_size"] = len(seeds)
        metadata["public_fact_filter_summary"] = chapter_public_filter_summary
        metadata["global_public_fact_filter_summary"] = public_filter_summary
        metadata["writable_fact_count"] = writable_fact_count
        metadata["eligible_citation_count"] = eligible_citation_count
        metadata["fact_card_count"] = int(chapter_analysis.get("fact_card_count") or 0)
        metadata["chapter_analysis_valid"] = bool(chapter_analysis.get("chapter_analysis_valid"))
        metadata["existing_chapter_evidence_count"] = len(existing_items)
        metadata["matched_evidence_count"] = len(matched)
        metadata["chapter_relevance_rejected_count"] = len(chapter_relevance_rejected)
        metadata["chapter_relevance_rejected_refs"] = [
            str(item.get("evidence_id") or item.get("ref") or item.get("id") or _evidence_ref(item)).strip()
            for item in chapter_relevance_rejected
            if str(item.get("evidence_id") or item.get("ref") or item.get("id") or _evidence_ref(item)).strip()
        ][:12]
        metadata["chapter_relevance_rejection_reasons"] = {
            reason: len([item for item in chapter_relevance_rejected if item.get("chapter_relevance_rejection_reason") == reason])
            for reason in sorted(
                {
                    str(item.get("chapter_relevance_rejection_reason") or "unknown")
                    for item in chapter_relevance_rejected
                }
            )
        }
        metadata["binding_reasons"] = binding_reasons
        metadata["unresolved_evidence_refs"] = unresolved_refs
        metadata["evidence_binding_counts"] = layer_counts
        metadata["evidence_binding_funnel"] = binding_funnel
        chapter.update(layered)
        chapter.update(
            {
                "chapter_id": chapter_id,
                "chapter_title": title,
                "chapter_question": question,
                "metadata": metadata,
                "unresolved_evidence_refs": unresolved_refs,
                "hydrated_evidence": bool(hydrated_count),
                "hydrated_evidence_count": hydrated_count,
                "binding_reasons": binding_reasons,
                "core_evidence_count": len(layered["core_evidence"]),
                "supporting_evidence_count": len(layered["supporting_evidence"]),
                "metric_evidence_count": len(layered["metric_evidence"]),
                "counter_evidence_count": len(layered["counter_evidence"]),
                "case_evidence_count": len(layered["case_evidence"]),
                "directional_evidence_count": len(layered["directional_evidence"]),
                "unresolved_evidence_ref_count": len(unresolved_refs),
                "evidence_binding_funnel": binding_funnel,
                "public_fact_filter_summary": chapter_public_filter_summary,
                "global_public_fact_filter_summary": public_filter_summary,
                "writable_fact_count": writable_fact_count,
                "eligible_citation_count": eligible_citation_count,
                "chapter_analysis": chapter_analysis,
                "fact_card_count": int(chapter_analysis.get("fact_card_count") or 0),
                "chapter_analysis_valid": bool(chapter_analysis.get("chapter_analysis_valid")),
            }
        )
        result.append(chapter)
    return result
