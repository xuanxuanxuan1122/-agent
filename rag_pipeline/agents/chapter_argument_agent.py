from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .report_contracts import ClaimUnit, EvidenceFactCard
from .layout_claim_matcher import claim_supported_block_types
from .section_composer import compose_section_paragraph
from .section_body_rewrite_agent import body_rewrite_max_sections, rewrite_sections_for_report


AGENT_NAME = "chapter_argument_agent"
AGENT_DESCRIPTION = "Chapter Argument Agent. Builds structured chapter packages from public argument units and tables."
BAD_FACT_PATTERNS = [
    r"^\s*-?\d{2,6}(?:\.\d+)?\s*(?:$|[;,\.\u3002\uff1b\uff0c])",
    r"^\s*(?:fact|key fact|metric|source_check|status|policy target|competitive comparison|cost)\s*[:\uff1a]\s*-?\d{1,6}(?:\.\d+)?\b",
    r"^\s*(?:\u4e8b\u5b9e|\u5173\u952e\u4e8b\u5b9e|\u7ade\u4e89\u5bf9\u6bd4|\u653f\u7b56\u76ee\u6807|\u653f\u7b56\u76d1\u7ba1|\u6210\u672c)\s*[:\uff1a]\s*-?\d{1,6}(?:\.\d+)?\b",
    r"^\s*(?:\u5185\u5bb9\u8bf4\u660e|\u65f6\u95f4)\s*[:\uff1a]",
    r"联网分析\s*Agent\s*失败",
    r"Retrieval\.TestUserQueryExceeded",
    r"query\s+exceed(?:ed)?\s+the\s+limit",
    r"目前更像局部信号",
    r"Skip to (?:content|main content)",
    r"picture intentionally omitted",
    r"Over the weekend",
    r"Futian district government",
    r"首页问\s*·\s*答|热搜公司|热搜词|登录注册",
    r"Caret right|View all products|Product\s+Documentation",
    r"\*\*==>\s*picture intentionally omitted\s*<==\*\*",
    r"登录\s+首页|上一篇|下一篇|分享到|AI帮你提炼|智能挖掘|智享会员|会员积分",
    r"^(?:事实|竞争对比|关键事实|政策目标)\s*[:：]\s*-?\d{1,3}(?:\.\d+)?\b",
    r"^\s*-?\d{1,3}(?:\.\d+)?\s*[;；，,]",
    r"以下是对整篇.*(?:深度分析|框架提炼)",
    r"问答[:：]?首页问|答云访谈|综合资讯投票",
]
BAD_FACT_PATTERNS.extend(
    [
        r"\u4e0d\u662f\u5355\u70b9\u4e8b\u5b9e\u9898",
        r"\u4f9b\u7ed9\u7ea6\u675f",
        r"\u4ef7\u683c\u4fee\u590d",
        r"\u5e93\u5b58\u4e0b\u964d",
        r"\u8ba2\u5355\u786e\u8ba4",
        r"\u519c\u4e1a\u4eba\u5de5\u667a\u80fd",
        r"\u6570\u636e\u6295\u6bd2",
        r"Scribd",
        r"\u53d1\u73b0\u62a5\u544a",
        r"\u7eba\u7ec7",
        r"\u667a\u80fd\u624b\u673a",
        r"SEO",
        r"(?:\u6210\u672c|\u5173\u952e\u4e8b\u5b9e|\u653f\u7b56\u76d1\u7ba1|\u653f\u7b56\u76ee\u6807)\s*[:\uff1a]\s*-?\d{1,3}(?:\.\d+)?%?",
        r"\u53ef\u590d\u6838\u6765\u6e90\u8d8a\u72ec\u7acb",
        r"\u76ee\u524d\u7ed3\u8bba\u4ecd\u53d7",
        r"\u4f18\u5148\u590d\u6838\u53ef\u8ffd\u6eaf\u6765\u6e90",
        r"\u672c\u7ae0\u5173\u6ce8",
        r"\u672c\u8282\u56f4\u7ed5",
        r"\u4e0d\u5177\u6709\u636e\u4ee5\u53d1\u884c\u80a1\u7968\u7684\u6cd5\u5f8b\u6548\u529b",
        r"\u4ec5\u4f9b\u9884\u5148\u62ab\u9732",
        r"\u6295\u8d44\u8005\u5e94\u5f53\u4ee5\u6b63\u5f0f\u516c",
        r"\u539f\u6587\u94fe\u63a5",
        r"\u539f\u6587\u51fa\u5904",
        r"\u4e0b\u8f7d",
        r"\u9644\u4e0b\u8f7d",
        r"\u7b2c\s*\d+\s*\u8f6e",
        r"picture\s*\[\d+\s*x\s*\d+\]\s*intentionally\s*omitted",
        r"\u8d2d\u7269\u8f66|\u6211\u7684\u8ba2\u5355|\u514d\u8d39\u6ce8\u518c|\u62a5\u544a\u670d\u52a1\u70ed\u7ebf",
        r"URL[:\uff1a]",
        r"\bOfficial\s+statistics\s+show\s+AI\s+agent\s+adoption\b",
        r"\bIf\s+later\s+A/B\s+sources\b",
        r"\bdowngrade\s+the\s+claim\b",
        r"\*\s*\u9996\u9875\s*\*\s*\u5feb\u8baf",
        r"\bProduct\s*\*\s*Solutions\s*\*\s*Pricing\b",
        r"\bResources\s*\*\s*About\b",
    ]
)

TEMPLATE_SENTENCE_PATTERNS = [
    r"[^。\n]*\u53ef\u590d\u6838\u6765\u6e90\u8d8a\u72ec\u7acb[^。\n]*(?:。|$)",
    r"[^。\n]*\u76ee\u524d\u7ed3\u8bba\u4ecd\u53d7[^。\n]*(?:。|$)",
    r"[^。\n]*\u4f18\u5148\u590d\u6838\u53ef\u8ffd\u6eaf\u6765\u6e90[^。\n]*(?:。|$)",
    r"[^。\n]*\u672c\u7ae0\u5173\u6ce8[^。\n]*(?:。|$)",
    r"[^。\n]*\u672c\u8282\u56f4\u7ed5[^。\n]*(?:。|$)",
    r"[^。\n]*\u8be5\u4fe1\u53f7\u53ef\u4f5c\u4e3a\u672c\u7ae0\u7684\u5ba1\u614e\u7ed3\u8bba[^。\n]*(?:。|$)",
    r"[^。\n]*\u8fd9\u4e9b\u4fe1\u606f\u5bf9\u5e94[^。\n]*\u53d8\u91cf[^。\n]*(?:。|$)",
    r"[^。\n]*\u8fb9\u754c\u5728\u4e8e\u6837\u672c\u662f\u5426\u4ee3\u8868\u4e3b\u6d41\u9700\u6c42[^。\n]*(?:。|$)",
    r"[^。\n]*\u672c\u7ae0\u4fe1\u53f7\u96c6\u4e2d\u5728[^。\n]*(?:。|$)",
    r"[^。\n]*\u4fe1\u53f7\u96c6\u4e2d\u5728[^。\n]*\u8be5\u4fe1\u53f7[^。\n]*(?:。|$)",
    r"[^。\n]*\u4e8b\u5b9e\u4f9d\u636e\u5305\u62ec[^。\n]*(?:。|$)",
    r"[^。\n]*\u53ef\u590d\u6838\u4e8b\u5b9e\u663e\u793a[^。\n]*(?:。|$)",
    r"[^。\n]*\u53ef\u6838\u9a8c\u4e8b\u5b9e\u663e\u793a[^。\n]*(?:。|$)",
    r"[^。\n]*\u82e5\u76f8\u53cd\u6837\u672c\u6216\u53e3\u5f84\u5dee\u5f02\u6269\u5927[^。\n]*(?:。|$)",
    r"[^。\n]*\u5206\u6790\u91cd\u70b9\u662f\u8fd9\u4e9b\u4e8b\u5b9e\u4e4b\u95f4\u662f\u5426\u6307\u5411\u540c\u4e00\u53d8\u91cf[^。\n]*(?:。|$)",
    r"[^。\n]*\u672c\u6bb5\u5224\u65ad\u9700\u8981\u6536\u7a84[^。\n]*(?:。|$)",
    r"[^\u3002\n]*\u4e0e\u672c\u7ae0\u5224\u65ad\u76f4\u63a5\u76f8\u5173[^\u3002\n]*(?:\u3002|$)",
    r"[^\u3002\n]*\u540e\u7eed\u5206\u6790\u9700\u8981[^\u3002\n]*(?:\u3002|$)",
    r"[^\u3002\n]*\u53ea\u6709\u4e3b\u4f53\u3001\u8303\u56f4\u548c\u671f\u95f4\u4e00\u81f4[^\u3002\n]*(?:\u3002|$)",
    r"[^\u3002\n]*\u5f71\u54cd\u7684\u662f\u5546\u4e1a\u5316\u6df1\u5ea6[^\u3002\n]*(?:\u3002|$)",
    r"[^\u3002\n]*\u5f71\u54cd\u7684\u662f\u5e02\u573a\u7a7a\u95f4\u5224\u65ad[^\u3002\n]*(?:\u3002|$)",
    r"[^\u3002\n]*\u4e0d\u80fd\u5916\u63a8\u4e3a\u666e\u904d\u4ed8\u8d39\u80fd\u529b[^\u3002\n]*(?:\u3002|$)",
]

BAD_SECTION_TITLE_PATTERNS = [
    r"\u6536\u5165\u3001\u5229\u6da6\u4e0e\u73b0\u91d1\u6d41\u8d28\u91cf",
    r"\u5355\u4f4d\u7ecf\u6d4e\u6a21\u578b",
    r"\u6295\u8d44\u4f18\u5148\u7ea7\u77e9\u9635",
]

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


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _compact(value: Any, max_chars: int = 260) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "..."


def _truncate_without_ellipsis(value: Any, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip(" ,，;；。")


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


def _is_bad_public_fact(value: Any) -> bool:
    text = str(value or "")
    if not text:
        return True
    return _is_snippet_like_public_text(text) or any(re.search(pattern, text, flags=re.I) for pattern in BAD_FACT_PATTERNS)


def _is_snippet_like_public_text(value: Any) -> bool:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return False
    if re.search(r"字体\s*[:：]\s*大\s*中\s*小", text):
        return True
    if re.search(r"国内垂直领域研报服务|以下为本次访谈实录|电子工程专辑|爱分析访谈", text):
        return True
    if re.search(r"AI\s*时代，唯一确定的是数据", text, flags=re.I):
        return True
    if re.match(r"^[^。；;]{6,90}[｜|][^。；;]{2,90}\s*-\s*[^:：。]{2,50}(?:[（(]20\d{2}[^）)]*[）)])?[:：]", text):
        return True
    if re.match(r"^[^。；;]{6,90}\s*-\s*[^:：。]{2,50}(?:[（(]20\d{2}[^）)]*[）)])?[:：]", text):
        return True
    if re.match(r"^(?:显示|为此|因此|当前|相关|本文)[，,：:]", text):
        return True
    if "..." in text or "…" in text:
        return True
    if re.search(r"[（(]\s*[）)]", text):
        return True
    if re.search(r"(?:落地与竞争信号|指标口径|技术成熟度|代表性案例对比|反向信号与失效条件)\s*[:：]", text):
        return True
    if re.match(r"^(?:近日|今日|日前|今年\s*\d+\s*月份?|过去\s*\d+\s*[天周月年]|一盆|一场|一句|一篇)", text):
        return True
    if re.match(r"^[^。；;]{6,48}-[^:：。]{2,24}[:：]\s*", text):
        return True
    if re.match(r"^[^。]*[“\"][^”\"。]{8,}[”\"][^。]*[:：]", text):
        return True
    if re.match(r"^[^。；;]{8,90}[:：]\s*", text) and len(text) > 90:
        return True
    chinese_count = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin_count = len(re.findall(r"[A-Za-z]", text))
    if len(text) > 140 and latin_count > 80 and chinese_count / max(1, chinese_count + latin_count) < 0.3:
        return True
    return False


def _clean_public_text(value: Any, max_chars: int = 900) -> str:
    raw_text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not raw_text:
        return ""
    text = raw_text
    for pattern in TEMPLATE_SENTENCE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""

    sentences = [
        item.strip()
        for item in re.split(r"(?<=[\u3002\uff1b\uff01\uff1f.!?;])\s*", text)
        if item.strip()
    ]
    if not sentences:
        sentences = [text]

    kept: List[str] = []
    for sentence in sentences:
        candidate = sentence.strip()
        if not candidate:
            continue
        if re.search(r"https?://|URL[:\uff1a]", candidate, flags=re.I):
            continue
        if _is_snippet_like_public_text(candidate):
            continue
        if any(re.search(pattern, candidate, flags=re.I) for pattern in BAD_FACT_PATTERNS):
            continue
        kept.append(candidate)

    cleaned = re.sub(r"\s{2,}", " ", " ".join(kept)).strip()
    if not cleaned:
        return ""
    return _truncate_without_ellipsis(cleaned, max_chars)


def _has_template_risk(value: Any) -> bool:
    text = str(value or "")
    if not text:
        return False
    return any(re.search(pattern, text, flags=re.I) for pattern in TEMPLATE_SENTENCE_PATTERNS)


def _clean_fact_anchor(value: Any, max_chars: int = 220) -> str:
    text = _compact(value, max_chars * 3)
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\[[Pp][Dd][Ff]\]\s*", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\bRead page[:：]?\s*", "", text, flags=re.I)
    text = re.sub(r"\bcontent description[:：]?\s*", "", text, flags=re.I)
    text = re.sub(r"^(?:摘要|关键事实|事实|标题|来源)[:：]\s*", "", text)
    text = re.sub(r"^(?:显示|为此|因此|当前|相关|本文)[，,：:]\s*", "", text)
    text = re.sub(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:[+-]\d{2}:\d{2})?", "", text)
    text = re.sub(r"\s*[；;]\s*", "；", text)
    text = re.sub(r"\s{2,}", " ", text).strip(" ；;，,。")
    if _is_bad_public_fact(text):
        return ""
    parts: List[str] = []
    seen = set()
    for part in re.split(r"[；;。]\s*", text):
        part = _compact(part, max_chars)
        if not part or _is_bad_public_fact(part):
            continue
        if part.count("*") >= 2 or len(re.findall(r"\b[A-Z][a-z]+(?:\s+\*)", part)) >= 2:
            continue
        if len(part) > 80 and re.search(r"\bNEWS\b|\bProduct\b|\bSolutions\b|\bResources\b", part, flags=re.I):
            continue
        key = re.sub(r"\W+", "", part.lower())[:80]
        if key in seen:
            continue
        seen.add(key)
        parts.append(part)
        if len(parts) >= 2:
            break
    return "；".join(parts)[:max_chars].rstrip("；,， ")


def _is_bad_section_title(value: Any) -> bool:
    text = str(value or "")
    if text in {"代表性案例对比", "反向信号与失效条件", "市场空间是否成立", "付费转化是否成立"}:
        return True
    lowered = text.lower()
    if any(token in lowered for token in ("official_me", "source_check", "proof_role", "block_type")):
        return True
    if re.search(r"\b(?:metric|counter|case_comparison|risk_trigger|unit_economics|metric_reconciliation)\b", lowered):
        return True
    if re.fullmatch(r"[a-z]+(?:_[a-z0-9]+){1,4}.*", lowered):
        return True
    return any(re.search(pattern, text, flags=re.I) for pattern in BAD_SECTION_TITLE_PATTERNS)


def _invalid_metric_item(item: Dict[str, Any]) -> bool:
    quality = _as_dict(item.get("public_fact_quality"))
    if quality and not bool(quality.get("eligible_for_report")):
        return True
    metric = str(item.get("metric") or item.get("indicator") or "").strip()
    value = str(item.get("value") or item.get("display_value") or item.get("numeric_value") or "").strip()
    fact = str(item.get("fact") or item.get("clean_fact") or item.get("content") or item.get("summary") or "").strip()
    metric_lower = metric.lower()
    if str(item.get("metric_validation_status") or "").strip().lower() == "invalid":
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


def _public_fact_from_item(item: Dict[str, Any], max_chars: int = 220) -> str:
    quality = _as_dict(item.get("public_fact_quality"))
    if quality and not bool(quality.get("eligible_for_report")):
        return ""
    card = _as_dict(item.get("public_fact_card") or quality.get("public_fact_card"))
    if not card:
        return ""
    card_fact = _clean_fact_anchor(
        card.get("distilled_fact")
        or card.get("public_fact")
        or card.get("fact")
        or card.get("object"),
        max_chars,
    )
    if card_fact:
        return card_fact
    distilled = _clean_fact_anchor(item.get("distilled_fact") or quality.get("distilled_fact"), max_chars)
    if distilled:
        return distilled
    fact = _clean_fact_anchor(item.get("fact") or item.get("clean_fact") or item.get("content") or item.get("summary"), max_chars)
    if fact:
        return fact
    metric = _compact(item.get("metric") or item.get("indicator"), 80)
    value = _compact(item.get("value") or item.get("display_value"), 80)
    if metric and value and not _invalid_metric_item(item):
        return f"{metric}: {value}"
    return ""


TOKEN_PROFILE_INT_DEFAULTS = {
    "lean": {"REPORT_CHAPTER_FACT_DIGEST_LIMIT": 4},
    "balanced": {"REPORT_CHAPTER_FACT_DIGEST_LIMIT": 8},
    "deep": {"REPORT_CHAPTER_FACT_DIGEST_LIMIT": 18},
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


def _by_chapter(items: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    result: Dict[str, List[Dict[str, Any]]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        chapter_id = str(item.get("chapter_id") or "")
        result.setdefault(chapter_id, []).append(item)
    return result


def public_argument_units(units: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        unit
        for unit in list(units or [])
        if isinstance(unit, dict)
        and unit.get("public_render") is True
        and not unit.get("omit_from_report")
    ]


def _public_tables(tables: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        table
        for table in list(tables or [])
        if isinstance(table, dict)
        and table.get("should_render")
        and not table.get("appendix_only")
    ]


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


def _collections_for_layout_section(layout_section: Dict[str, Any]) -> List[str]:
    block_type = str(layout_section.get("block_type") or layout_section.get("output_type") or "").strip()
    if block_type == "metric_reconciliation":
        return ["metric_evidence", "core_evidence", "supporting_evidence"]
    if block_type == "unit_economics":
        return ["metric_evidence", "case_evidence", "core_evidence", "supporting_evidence"]
    if block_type in {"competitive_positioning", "customer_painpoint_matrix", "case_comparison"}:
        return ["case_evidence", "core_evidence", "supporting_evidence", "directional_evidence"]
    if block_type == "technology_maturity":
        return ["core_evidence", "supporting_evidence", "directional_evidence", "metric_evidence"]
    if block_type in {"risk_trigger", "verification_checklist", "scenario_analysis"}:
        return ["counter_evidence", "core_evidence", "supporting_evidence", "directional_evidence"]
    if block_type in {"signal_validation", "evidence_matrix", "thesis", "argument"}:
        return ["core_evidence", "supporting_evidence", "directional_evidence", "case_evidence", "sample_evidence"]
    return list(PUBLIC_EVIDENCE_COLLECTIONS)


def _facts_from_collections(evidence_package: Dict[str, Any], collections: Sequence[str], *, limit: int = 4) -> List[str]:
    facts: List[str] = []
    seen = set()
    for collection in collections:
        for item in _as_list(evidence_package.get(collection)):
            if not isinstance(item, dict):
                continue
            if _invalid_metric_item(item):
                continue
            fact = _public_fact_from_item(item, 220)
            if not fact or _is_bad_public_fact(fact):
                continue
            key = re.sub(r"\s+", "", fact.lower())[:140]
            if key in seen:
                continue
            seen.add(key)
            facts.append(fact)
            if len(facts) >= limit:
                return facts
    return facts


def _refs_from_collections(evidence_package: Dict[str, Any], collections: Sequence[str], *, limit: int = 6) -> List[str]:
    refs: List[str] = []
    for collection in collections:
        for item in _as_list(evidence_package.get(collection)):
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


def _evidence_item_ref_keys(item: Dict[str, Any]) -> set[str]:
    keys = {
        str(item.get("ref") or "").strip(),
        str(item.get("evidence_id") or "").strip(),
        str(item.get("source_ref") or "").strip(),
        str(item.get("citation_ref") or "").strip(),
        _citation_ref_from_evidence(item),
    }
    keys.update(str(ref or "").strip() for ref in _as_list(item.get("source_refs")))
    return {key for key in keys if key}


def _fact_cards_from_package(
    evidence_package: Dict[str, Any],
    refs: Sequence[Any],
    collections: Sequence[str],
    *,
    layout_section: Optional[Dict[str, Any]] = None,
    limit: int = 4,
) -> List[EvidenceFactCard]:
    wanted = {str(ref or "").strip() for ref in refs if str(ref or "").strip()}
    cards: List[EvidenceFactCard] = []
    seen = set()
    layout_section = _as_dict(layout_section)
    for collection in collections:
        for item in _as_list(evidence_package.get(collection)):
            if not isinstance(item, dict) or _invalid_metric_item(item):
                continue
            if layout_section and not _item_matches_layout_section(item, layout_section):
                continue
            if wanted and not wanted.intersection(_evidence_item_ref_keys(item)):
                continue
            card = EvidenceFactCard.from_legacy_dict({**item, "chapter_id": evidence_package.get("chapter_id") or item.get("chapter_id")})
            if not card.is_valid_for_report:
                continue
            if card.evidence_id in seen:
                continue
            seen.add(card.evidence_id)
            cards.append(card)
            if len(cards) >= limit:
                return cards
    return cards


def _fact_type_for_block_type(block_type: str) -> str:
    block = str(block_type or "").strip()
    if block == "metric_reconciliation":
        return "metric"
    if block in {"case_comparison", "customer_painpoint_matrix", "unit_economics", "competitive_positioning"}:
        return "case"
    if block == "technology_maturity":
        return "technology"
    if block in {"risk_trigger", "verification_checklist", "scenario_analysis"}:
        return "counter"
    return "directional"


def _variable_for_block_type(block_type: str) -> str:
    block = str(block_type or "").strip()
    if block == "metric_reconciliation":
        return "市场指标"
    if block in {"case_comparison", "customer_painpoint_matrix"}:
        return "部署深度"
    if block == "unit_economics":
        return "商业化"
    if block == "technology_maturity":
        return "技术成熟度"
    if block in {"risk_trigger", "verification_checklist", "scenario_analysis"}:
        return "风险边界"
    if block == "competitive_positioning":
        return "竞争位置"
    return "章节信号"


def _fact_cards_from_unit_basis(
    unit: Dict[str, Any],
    chapter: Dict[str, Any],
    supporting_facts: Sequence[str],
    evidence_refs: Sequence[Any],
    block_type: str,
    *,
    limit: int = 4,
) -> List[EvidenceFactCard]:
    refs = _dedupe(
        [
            *list(evidence_refs or []),
            *_as_list(unit.get("used_fact_refs")),
            *_as_list(unit.get("used_evidence_ids")),
            *_as_list(unit.get("supporting_evidence_refs")),
        ],
        limit=limit,
    )
    if not refs:
        return []
    facts = [
        cleaned
        for value in list(supporting_facts or []) + _as_list(unit.get("evidence_basis")) + _as_list(unit.get("supporting_facts"))
        for cleaned in [_clean_fact_anchor(value, 220)]
        if cleaned and not _is_bad_public_fact(cleaned)
    ]
    facts = _dedupe(facts, limit=limit)
    if not facts:
        return []
    fact_type = _fact_type_for_block_type(block_type)
    cards: List[EvidenceFactCard] = []
    for index, fact in enumerate(facts[:limit]):
        ref = refs[min(index, len(refs) - 1)]
        card = EvidenceFactCard.from_legacy_dict(
            {
                "evidence_id": ref,
                "chapter_id": unit.get("chapter_id") or chapter.get("chapter_id"),
                "source_ref": ref,
                "source_level": unit.get("source_level") or unit.get("evidence_source_level") or "",
                "source_verification_status": unit.get("source_verification_status") or "",
                "proof_role": unit.get("proof_role") or unit.get("section_role") or block_type,
                "claim_strength": unit.get("claim_strength"),
                "public_fact_card": {
                    "subject": unit.get("subject") or unit.get("actor") or unit.get("company") or "",
                    "action_or_signal": unit.get("action_or_signal") or unit.get("signal") or "",
                    "variable": unit.get("variable") or unit.get("analysis_variable") or _variable_for_block_type(block_type),
                    "distilled_fact": fact,
                    "fact_type": fact_type,
                    "source_ref": ref,
                    "block_affinity": [block_type] if block_type else [],
                    "claim_strength_hint": unit.get("claim_strength") or "",
                },
                "public_fact_quality": {"eligible_for_report": True},
            }
        )
        if card.is_valid_for_report:
            cards.append(card)
    return cards


def _item_matches_layout_section(item: Dict[str, Any], layout_section: Dict[str, Any]) -> bool:
    block_type = str(layout_section.get("block_type") or layout_section.get("output_type") or "").strip()
    if not block_type:
        return True
    quality = _as_dict(item.get("public_fact_quality"))
    card = _as_dict(item.get("public_fact_card") or quality.get("public_fact_card"))
    affinity = {
        str(value or "").strip()
        for value in _as_list(card.get("block_affinity"))
        if str(value or "").strip()
    }
    if block_type in affinity:
        return True
    fact_type = str(card.get("fact_type") or quality.get("fact_type") or item.get("fact_type") or "").strip()
    blob = " ".join(
        str(item.get(key) or "")
        for key in ("proof_role", "evidence_role", "role", "source_type", "metric", "fact", "clean_fact", "distilled_fact", "title")
    )
    allowed = {
        "metric_reconciliation": {"metric"},
        "unit_economics": {"metric", "case", "source_check"},
        "technology_maturity": {"technology", "technology_product", "standard", "source_check"},
        "risk_trigger": {"counter"},
        "verification_checklist": {"counter"},
        "scenario_analysis": {"counter", "directional"},
        "competitive_positioning": {"case", "source_check", "directional"},
        "customer_painpoint_matrix": {"case", "directional"},
        "case_comparison": {"case", "directional"},
        "signal_validation": {"metric", "case", "directional", "source_check", "technology", "counter"},
        "evidence_matrix": {"metric", "case", "directional", "source_check", "technology", "counter"},
        "thesis": {"metric", "case", "directional", "source_check", "technology", "counter"},
        "argument": {"metric", "case", "directional", "source_check", "technology", "counter"},
    }.get(block_type, set())
    if allowed and fact_type in allowed:
        return True
    if block_type == "technology_maturity":
        return bool(re.search(r"技术|标准|工具调用|权限|安全|可靠|部署|产品|专利|模型|agent|workflow|standard|patent|security|tool", blob, flags=re.I))
    if block_type == "unit_economics":
        return bool(re.search(r"营收|收入|利润|毛利|现金流|付费|收费|价格|续约|订单|采购|revenue|pricing|renewal|order|procurement", blob, flags=re.I))
    if block_type == "competitive_positioning":
        return bool(re.search(r"竞争|玩家|渠道|生态|份额|替代|平台|厂商|competition|player|channel|ecosystem", blob, flags=re.I))
    if block_type in {"customer_painpoint_matrix", "case_comparison"}:
        return bool(re.search(r"案例|客户|采购|中标|订单|部署|落地|场景|case|customer|deployment|procurement", blob, flags=re.I))
    if block_type in {"signal_validation", "evidence_matrix", "thesis", "argument"}:
        return bool(card or quality.get("eligible_for_report") or _public_fact_from_item(item))
    return False


def _facts_and_refs_for_layout_section(
    evidence_package: Dict[str, Any],
    layout_section: Dict[str, Any],
    collections: Sequence[str],
    *,
    fact_limit: int = 3,
    ref_limit: int = 6,
) -> tuple[List[str], List[str]]:
    facts: List[str] = []
    refs: List[str] = []
    seen_facts = set()
    for collection in collections:
        for item in _as_list(evidence_package.get(collection)):
            if not isinstance(item, dict):
                continue
            if _invalid_metric_item(item) or not _item_matches_layout_section(item, layout_section):
                continue
            fact = _public_fact_from_item(item, 220)
            if fact and not _is_bad_public_fact(fact):
                key = re.sub(r"\s+", "", fact.lower())[:140]
                if key not in seen_facts and len(facts) < fact_limit:
                    seen_facts.add(key)
                    facts.append(fact)
            ref = _citation_ref_from_evidence(item)
            if ref:
                refs.append(ref)
            if len(facts) >= fact_limit and len(refs) >= ref_limit:
                return facts, _dedupe(refs, limit=ref_limit)
    return facts, _dedupe(refs, limit=ref_limit)


def _facts_for_refs(
    evidence_package: Dict[str, Any],
    refs: Sequence[Any],
    collections: Sequence[str],
    *,
    limit: int = 3,
) -> List[str]:
    wanted = {str(ref or "").strip() for ref in refs if str(ref or "").strip()}
    if not wanted:
        return []
    facts: List[str] = []
    seen = set()
    for collection in collections:
        for item in _as_list(evidence_package.get(collection)):
            if not isinstance(item, dict) or _invalid_metric_item(item):
                continue
            item_refs = {
                str(item.get("ref") or "").strip(),
                str(item.get("evidence_id") or "").strip(),
                str(item.get("source_ref") or "").strip(),
                str(item.get("citation_ref") or "").strip(),
                _citation_ref_from_evidence(item),
                *[str(ref or "").strip() for ref in _as_list(item.get("source_refs"))],
            }
            item_refs = {ref for ref in item_refs if ref}
            if not wanted.intersection(item_refs):
                continue
            fact = _public_fact_from_item(item, 220)
            if not fact or _is_bad_public_fact(fact):
                continue
            key = re.sub(r"\s+", "", fact.lower())[:140]
            if key in seen:
                continue
            seen.add(key)
            facts.append(fact)
            if len(facts) >= limit:
                return facts
    return facts


def _lead(chapter: Dict[str, Any], units: Sequence[Dict[str, Any]]) -> str:
    if units:
        claim = _clean_public_text(units[0].get("claim"), 240)
        if claim:
            return claim
    question = _compact(chapter.get("chapter_question"), 180)
    if question:
        return f"{question}需要结合可核验事实、机制约束和边界条件判断。"
    return ""


SECTION_TITLE_BY_BLOCK_TYPE = {
    "thesis": "核心观察",
    "argument": "事实依据",
    "metric_reconciliation": "指标口径与可比性",
    "risk_trigger": "边界条件",
    "verification_checklist": "后续观察变量",
    "case_argument": "案例事实",
    "customer_painpoint_matrix": "需求与付费证据",
    "competitive_positioning": "竞争变量",
    "technology_maturity": "技术变量与约束",
    "unit_economics": "商业化证据",
}

GENERIC_PUBLIC_SECTION_TITLES = {
    "事实依据",
    "商业化证据",
    "核心观察",
    "本章结论",
    "关键事实与判断依据",
    "判断边界与后续验证",
}
FORBIDDEN_PUBLIC_TITLE_TERMS = {"证据", "口径", "变量", "可验证信号", "判断依据"}


def _safe_section_title(value: Any) -> str:
    title = _clean_public_text(value, 120)
    generic_titles = {
        "事实依据",
        "商业化证据",
        "核心观察",
        "本章结论",
        "关键事实与判断依据",
        "判断边界与后续验证",
    }
    forbidden_terms = {
        "证据",
        "口径",
        "变量",
        "可验证信号",
        "判断依据",
    }
    if re.fullmatch(r"本节.{0,12}观察", title):
        return ""
    if not title or title in GENERIC_PUBLIC_SECTION_TITLES or title in generic_titles:
        return ""
    if _is_bad_section_title(title):
        return ""
    if any(term in title for term in FORBIDDEN_PUBLIC_TITLE_TERMS) or any(term in title for term in forbidden_terms):
        return ""
    if re.fullmatch(r"(?:ch|chapter|section)[_-]?\d{1,3}", title.strip(), flags=re.I):
        return ""
    return title


def _chapter_variable(chapter: Dict[str, Any], block_type: str) -> str:
    blob = " ".join(
        str(chapter.get(key) or "")
        for key in ("chapter_title", "title", "chapter_question", "core_question", "chapter_role")
    )
    if block_type == "metric_reconciliation" or re.search(r"规模|增速|指标|口径|市场", blob):
        return "市场空间"
    if block_type == "unit_economics" or re.search(r"付费|商业化|收入|订单|续约|采购", blob):
        return "付费转化"
    if block_type in {"customer_painpoint_matrix", "case_comparison"} or re.search(r"客户|场景|需求|部署|采购", blob):
        return "需求场景"
    if block_type == "competitive_positioning" or re.search(r"竞争|玩家|生态|渠道|替代", blob):
        return "玩家格局"
    if block_type == "technology_maturity" or re.search(r"技术|工具|权限|安全|部署|标准", blob):
        return "部署卡点"
    if block_type in {"risk_trigger", "verification_checklist", "scenario_analysis"} or re.search(r"风险|反证|边界|失败|治理", blob):
        return "失效条件"
    return "判断主线"


def _fallback_dynamic_section_title(chapter: Dict[str, Any], block_type: str, index: int) -> str:
    variable = _chapter_variable(chapter, block_type)
    if block_type == "metric_reconciliation":
        if "空间" in variable:
            return "市场空间到底有多大"
        return f"{variable}说明了什么"
    if block_type == "unit_economics":
        return f"{variable}是否成立"
    if block_type in {"customer_painpoint_matrix", "case_comparison"}:
        return f"{variable}在哪里发生"
    if block_type == "competitive_positioning":
        return "谁在占据入口"
    if block_type == "technology_maturity":
        return f"{variable}在哪里"
    if block_type in {"risk_trigger", "verification_checklist", "scenario_analysis"}:
        return f"{variable}会怎样改变结论"
    return f"{variable}是否已经成立" if index <= 2 else f"{variable}接下来怎么看"


def _public_section_title(unit: Dict[str, Any], chapter: Dict[str, Any], *, index: int, layout_section: Optional[Dict[str, Any]] = None) -> str:
    layout_section = _as_dict(layout_section)
    dynamic = _safe_section_title(
        layout_section.get("dynamic_section_title")
        or unit.get("dynamic_section_title")
        or layout_section.get("generated_section_title")
        or unit.get("generated_section_title")
    )
    if dynamic:
        return dynamic
    raw = _safe_section_title(layout_section.get("section_title") or layout_section.get("title") or unit.get("section_title") or "")
    chapter_title = _compact(chapter.get("chapter_title") or chapter.get("title") or "", 120)
    chapter_question = _compact(chapter.get("chapter_question") or chapter.get("chapter_role") or "", 120)
    block_type = str(layout_section.get("block_type") or layout_section.get("output_type") or unit.get("block_type") or unit.get("layout_section_role") or "").strip()
    fallback = _fallback_dynamic_section_title(chapter, block_type, index)
    if _is_bad_section_title(fallback):
        fallback = _fallback_dynamic_section_title(chapter, "", index)
    canonical_title = SECTION_TITLE_BY_BLOCK_TYPE.get(block_type)
    if raw and canonical_title:
        raw_title_key = re.sub(r"\s+", "", raw).lower()
        canonical_title_key = re.sub(r"\s+", "", canonical_title).lower()
        known_title_keys = {
            re.sub(r"\s+", "", title).lower()
            for title in [*SECTION_TITLE_BY_BLOCK_TYPE.values(), *GENERIC_PUBLIC_SECTION_TITLES]
            if title
        }
        if raw_title_key in known_title_keys and raw_title_key != canonical_title_key:
            return fallback
    if not raw or _is_bad_section_title(raw):
        return fallback
    raw_key = re.sub(r"\s+", "", raw)
    title_key = re.sub(r"\s+", "", chapter_title)
    question_key = re.sub(r"\s+", "", chapter_question)
    if raw_key and raw_key in {title_key, question_key}:
        return fallback
    if len(raw) > 42 and (title_key.startswith(raw_key[:16]) or raw_key.startswith(title_key[:16])):
        return fallback
    return raw


def _section_purpose_for_block(block_type: str) -> str:
    block_type = str(block_type or "").strip()
    if block_type in {"metric_reconciliation", "evidence_matrix"}:
        return "metric_or_market_judgment"
    if block_type in {"customer_painpoint_matrix", "case_comparison", "case_argument"}:
        return "customer_or_case_judgment"
    if block_type == "competitive_positioning":
        return "competitive_judgment"
    if block_type == "technology_maturity":
        return "technology_judgment"
    if block_type in {"risk_trigger", "verification_checklist", "scenario_analysis"}:
        return "risk_or_boundary_judgment"
    if block_type == "unit_economics":
        return "commercialization_judgment"
    return "chapter_judgment"


def _build_section_plan(
    *,
    title: str,
    block_type: str,
    evidence_refs: Sequence[Any],
    used_fact_refs: Sequence[Any],
    supporting_facts: Sequence[Any],
    claim_strength: Any,
    template_removed: bool,
    evidence_backed: bool,
) -> Dict[str, Any]:
    refs = _dedupe([*list(used_fact_refs or []), *list(evidence_refs or [])], limit=8)
    fact_count = len([item for item in supporting_facts if str(item or "").strip()])
    omit_reason = ""
    if template_removed:
        omit_reason = "template_risk"
    elif not refs:
        omit_reason = "missing_fact_refs"
    elif fact_count <= 0:
        omit_reason = "missing_distilled_fact"
    elif not evidence_backed:
        omit_reason = "not_evidence_backed"
    return {
        "public_title": title,
        "section_purpose": _section_purpose_for_block(block_type),
        "used_fact_refs": refs,
        "claim_strength": str(claim_strength or "").strip() or "directional",
        "paragraph_plan": "判断句 -> 关键事实 -> 机制解释 -> 边界条件",
        "omit_reason": omit_reason,
        "evidence_backed": bool(evidence_backed),
    }


def _section_from_unit(
    unit: Dict[str, Any],
    chapter: Dict[str, Any],
    *,
    index: int,
    layout_section: Optional[Dict[str, Any]] = None,
    evidence_package: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    layout_section = _as_dict(layout_section)
    evidence_package = _as_dict(evidence_package)
    block_type = str(layout_section.get("block_type") or layout_section.get("output_type") or unit.get("block_type") or unit.get("layout_section_role") or "").strip()
    original_block_type = block_type
    unit_block_type = str(unit.get("block_type") or unit.get("output_type") or unit.get("layout_section_role") or "").strip()
    if block_type == "metric_reconciliation" and unit_block_type and unit_block_type != block_type:
        supported_blocks = set(claim_supported_block_types(unit))
        if unit_block_type in supported_blocks or unit_block_type in {
            "case_comparison",
            "customer_painpoint_matrix",
            "competitive_positioning",
            "technology_maturity",
            "risk_trigger",
            "integrated_signal",
        }:
            block_type = unit_block_type
    output_type = str(layout_section.get("output_type") or block_type or unit.get("output_type") or "").strip()
    section_role = str(layout_section.get("section_role") or unit.get("layout_section_role") or block_type or "").strip()
    collections = _collections_for_layout_section({"block_type": block_type, "output_type": output_type}) if block_type else list(PUBLIC_EVIDENCE_COLLECTIONS)
    used_fact_refs = _as_list(unit.get("used_fact_refs"))
    evidence_refs = used_fact_refs or _as_list(unit.get("evidence_refs")) or _as_list(layout_section.get("required_evidence_refs"))
    if not evidence_refs and evidence_package:
        evidence_refs = _refs_from_collections(evidence_package, collections, limit=6)
    unit_supporting_facts = [
        cleaned
        for item in (_as_list(unit.get("evidence_basis")) + _as_list(unit.get("supporting_facts")))
        for cleaned in [_clean_fact_anchor(item, 220)]
        if cleaned
    ][:3]
    package_supporting_facts: List[str] = []
    if evidence_package:
        package_supporting_facts = _facts_for_refs(evidence_package, evidence_refs, collections, limit=3)
    if not package_supporting_facts and evidence_package and evidence_refs:
        package_supporting_facts = _facts_for_refs(evidence_package, evidence_refs, list(PUBLIC_EVIDENCE_COLLECTIONS), limit=3)
    if not package_supporting_facts and evidence_package and block_type:
        package_supporting_facts = _facts_from_collections(evidence_package, collections, limit=3)
    supporting_facts = _dedupe([*package_supporting_facts, *unit_supporting_facts], limit=3)
    fact_cards = (
        _fact_cards_from_package(
            evidence_package,
            evidence_refs,
            collections,
            layout_section=layout_section,
            limit=4,
        )
        if evidence_package
        else []
    )
    if not fact_cards and evidence_package and evidence_refs:
        fact_cards = _fact_cards_from_package(
            evidence_package,
            evidence_refs,
            collections,
            layout_section=None,
            limit=4,
        )
    if not fact_cards:
        fact_cards = _fact_cards_from_unit_basis(
            unit,
            chapter,
            supporting_facts,
            evidence_refs,
            block_type,
            limit=4,
        )
    raw_public_texts = [
        unit.get("claim"),
        unit.get("reasoning"),
        unit.get("mechanism"),
        unit.get("counter_evidence"),
        *list(_as_list(unit.get("evidence_basis"))),
        *list(_as_list(unit.get("supporting_facts"))),
    ]
    snippet_lead_dropped = any(_is_snippet_like_public_text(value) for value in raw_public_texts if str(value or "").strip())
    claim = _clean_public_text(unit.get("claim") or "", 420)
    if not claim and supporting_facts:
        claim = supporting_facts[0]
    reasoning = _clean_public_text(unit.get("reasoning") or "", 720)
    mechanism = _clean_public_text(unit.get("mechanism") or unit.get("reasoning") or "", 720)
    counter_evidence = _clean_public_text(unit.get("counter_evidence") or "", 520)
    actionable = _clean_public_text(unit.get("actionable") or "", 420)
    decision_implication = _clean_public_text(unit.get("decision_implication") or unit.get("actionable") or "", 520)
    if reasoning and _fact_text_key(reasoning) == _fact_text_key(claim):
        reasoning = ""
    if mechanism and _fact_text_key(mechanism) in {_fact_text_key(claim), _fact_text_key(reasoning)}:
        mechanism = ""
    if counter_evidence and _fact_text_key(counter_evidence) in {_fact_text_key(claim), _fact_text_key(reasoning), _fact_text_key(mechanism)}:
        counter_evidence = ""
    if actionable and _fact_text_key(actionable) in {_fact_text_key(claim), _fact_text_key(reasoning), _fact_text_key(mechanism), _fact_text_key(counter_evidence)}:
        actionable = ""
    if decision_implication and _fact_text_key(decision_implication) in {
        _fact_text_key(claim),
        _fact_text_key(reasoning),
        _fact_text_key(mechanism),
        _fact_text_key(counter_evidence),
        _fact_text_key(actionable),
    }:
        decision_implication = ""
    composition = compose_section_paragraph(
        fact_cards=fact_cards,
        claim_unit=ClaimUnit.from_legacy_dict(
            {
                **unit,
                "chapter_id": unit.get("chapter_id") or chapter.get("chapter_id"),
                "evidence_refs": evidence_refs,
                "used_fact_refs": used_fact_refs or evidence_refs,
                "evidence_basis": supporting_facts,
                "claim_strength": unit.get("claim_strength"),
                "block_type": block_type,
                "section_id": layout_section.get("section_id") or unit.get("section_id"),
            }
        ),
        block_type=block_type,
        chapter_question=str(chapter.get("chapter_question") or chapter.get("chapter_title") or ""),
    )
    if composition.get("composition_status") != "dropped":
        claim = str(composition.get("claim") or claim)
        reasoning = str(composition.get("reasoning") or reasoning)
        mechanism = str(composition.get("mechanism") or mechanism)
        counter_evidence = str(composition.get("counter_evidence") or counter_evidence)
        supporting_facts = _as_list(composition.get("supporting_facts")) or supporting_facts
        used_fact_refs = _as_list(composition.get("used_fact_refs")) or used_fact_refs
        evidence_refs = _dedupe([*evidence_refs, *used_fact_refs], limit=8)
        variable_explanation = str(composition.get("variable_explanation") or "").strip()
        claim_key = _fact_text_key(claim)
        reasoning_key = _fact_text_key(reasoning)
        if reasoning and claim_key and (reasoning_key == claim_key or claim_key in reasoning_key):
            reasoning = variable_explanation if variable_explanation and _fact_text_key(variable_explanation) != claim_key else ""
        mechanism_key = _fact_text_key(mechanism)
        reasoning_key = _fact_text_key(reasoning)
        duplicate_mechanism = bool(
            mechanism
            and (
                mechanism_key in {claim_key, reasoning_key}
                or (claim_key and claim_key in mechanism_key)
                or (reasoning_key and reasoning_key in mechanism_key)
            )
        )
        if duplicate_mechanism:
            if variable_explanation and _fact_text_key(variable_explanation) not in {claim_key, reasoning_key}:
                mechanism = variable_explanation
            else:
                mechanism = ""
        counter_key = _fact_text_key(counter_evidence)
        mechanism_key = _fact_text_key(mechanism)
        if counter_evidence and (
            counter_key in {claim_key, reasoning_key, mechanism_key}
            or (claim_key and claim_key in counter_key)
            or (reasoning_key and reasoning_key in counter_key)
            or (mechanism_key and mechanism_key in counter_key)
        ):
            counter_evidence = ""
    render_blocks = []
    if composition.get("composition_status") != "dropped":
        composed_text = str(composition.get("paragraph") or claim or "").strip()
        boundary_text = str(composition.get("counter_evidence") or "").strip()
        render_blocks = [{"type": "paragraph", "label": "", "text": composed_text}]
        # Boundary/counter text is part of the analysis contract, but rendering it
        # as a standalone public paragraph made reports read like a review memo
        # ("this case is limited...", "cannot represent the whole market...").
        # Keep it available on the section for score/audit, and only expose it
        # when explicitly requested for debugging.
        render_boundary = str(os.getenv("REPORT_RENDER_BOUNDARY_PARAGRAPH") or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if render_boundary and boundary_text and _fact_text_key(boundary_text) != _fact_text_key(composed_text):
            render_blocks.append({"type": "paragraph", "label": "", "text": boundary_text})
    if not render_blocks:
        render_blocks = _as_list(unit.get("render_blocks"))
    if not render_blocks:
        render_blocks = [
            {"type": "paragraph", "label": "关键判断", "text": claim},
            {"type": "paragraph", "label": "证据依据", "text": reasoning},
            {"type": "paragraph", "label": "边界", "text": counter_evidence},
            {"type": "paragraph", "label": "含义", "text": decision_implication},
            {"type": "evidence_list", "label": "关键证据", "evidence_refs": evidence_refs},
        ]
    cleaned_blocks: List[Dict[str, Any]] = []
    for block in render_blocks:
        if not isinstance(block, dict):
            continue
        cleaned_block = dict(block)
        if "text" in cleaned_block:
            label = str(cleaned_block.get("label") or "")
            if "事实锚点" in label or "事实依据" in label:
                fact_parts = [
                    cleaned
                    for part in re.split(r"[；;。]\s*", str(cleaned_block.get("text") or ""))
                    for cleaned in [_clean_fact_anchor(part, 220)]
                    if cleaned
                ][:3]
                cleaned_block["text"] = "；".join(_dedupe(fact_parts, limit=3))
            else:
                cleaned_block["text"] = _clean_public_text(cleaned_block.get("text"), 900)
            if cleaned_block.get("type") == "paragraph" and not cleaned_block["text"]:
                continue
            previous_block_keys = {
                _fact_text_key(existing.get("text"))
                for existing in cleaned_blocks
                if isinstance(existing, dict)
            }
            if _fact_text_key(cleaned_block["text"]) and _fact_text_key(cleaned_block["text"]) in previous_block_keys:
                continue
        cleaned_blocks.append(cleaned_block)
    fact_card_match = bool(unit.get("fact_card_to_block_match")) or not bool(unit.get("fact_card_count"))
    template_removed = bool(
        _has_template_risk(unit.get("claim"))
        or _has_template_risk(unit.get("reasoning"))
        or _has_template_risk(unit.get("counter_evidence"))
        or _has_template_risk(unit.get("mechanism"))
    )
    mechanism_for_evidence = mechanism or reasoning or str(composition.get("variable_explanation") or "").strip()
    if not mechanism and mechanism_for_evidence:
        mechanism = mechanism_for_evidence
    evidence_backed = bool(used_fact_refs and supporting_facts and mechanism_for_evidence and fact_card_match and not template_removed)
    section_title = _public_section_title(unit, chapter, index=index, layout_section=layout_section)
    section_plan = _build_section_plan(
        title=section_title,
        block_type=block_type,
        evidence_refs=evidence_refs,
        used_fact_refs=used_fact_refs,
        supporting_facts=supporting_facts,
        claim_strength=unit.get("claim_strength"),
        template_removed=template_removed,
        evidence_backed=evidence_backed,
    )
    return {
        "claim_id": unit.get("claim_id") or unit.get("id") or "",
        "section_id": layout_section.get("section_id") or unit.get("section_id"),
        "section_title": section_title,
        "section_plan": section_plan,
        "dynamic_section_title": layout_section.get("dynamic_section_title") or unit.get("dynamic_section_title"),
        "title_source": layout_section.get("title_source") or unit.get("title_source"),
        "title_variables": _as_list(layout_section.get("title_variables")) or _as_list(unit.get("title_variables")),
        "block_title_generation_failed": bool(layout_section.get("block_title_generation_failed") or unit.get("block_title_generation_failed")),
        "block_type": block_type,
        "output_type": output_type,
        "section_role": section_role,
        "required_evidence_refs": _as_list(layout_section.get("required_evidence_refs")),
        "claim": claim,
        "composed_paragraph": str(composition.get("paragraph") or "").strip(),
        "hypothesis_id": unit.get("hypothesis_id") or "",
        "requirement_id": unit.get("requirement_id") or "",
        "requirement_ids": _as_list(unit.get("requirement_ids")),
        "claim_strength_ceiling": unit.get("claim_strength_ceiling") or "",
        "lineage": _as_dict(unit.get("lineage")),
        "analysis_role": unit.get("analysis_role") or "",
        "source_support_map": _as_dict(unit.get("source_support_map")),
        "paragraph_seed": unit.get("paragraph_seed") or "",
        "reasoning": reasoning,
        "mechanism": mechanism,
        "counter_evidence": counter_evidence,
        "actionable": actionable,
        "decision_implication": decision_implication,
        "what_to_verify_next": _as_list(unit.get("what_to_verify_next")),
        "supporting_facts": supporting_facts,
        "confidence": unit.get("confidence") or "medium",
        "evidence_refs": evidence_refs,
        "used_fact_refs": used_fact_refs,
        "render_blocks": cleaned_blocks,
        "public_render": True,
        "layout_generated": bool(unit.get("layout_generated")),
        "evidence_backed": evidence_backed,
        "composition_status": composition.get("composition_status") or "legacy",
        "body_composition_status": composition.get("body_composition_status") or "fact_passthrough",
        "variable_explanation": composition.get("variable_explanation") or "",
        "composer_variable_explanation_count": composition.get("composer_variable_explanation_count") or 0,
        "composer_expansion_status": composition.get("composer_expansion_status") or "",
        "composer_target_section_chars": composition.get("composer_target_section_chars") or 0,
        "composer_paragraph_chars": composition.get("composer_paragraph_chars") or 0,
        "observation_only": not evidence_backed,
        "fact_card_to_block_match": fact_card_match,
        "template_section_removed": template_removed,
        "snippet_lead_dropped": snippet_lead_dropped and not (claim or reasoning or mechanism),
        "layout_match_score": unit.get("layout_match_score"),
        "layout_match_reason": unit.get("layout_match_reason"),
        "block_evidence_fit_score": layout_section.get("block_evidence_fit_score") or unit.get("block_evidence_fit_score"),
        "selection_reason": layout_section.get("selection_reason") or unit.get("selection_reason"),
        "block_type_demoted_from": original_block_type if original_block_type != block_type else "",
    }


def _section_from_layout(
    layout_section: Dict[str, Any],
    chapter: Dict[str, Any],
    *,
    index: int,
    evidence_package: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    layout_section = _as_dict(layout_section)
    required_refs = _as_list(layout_section.get("required_evidence_refs"))
    collections = _collections_for_layout_section(layout_section)
    facts, matched_refs = _facts_and_refs_for_layout_section(evidence_package, layout_section, collections, fact_limit=3, ref_limit=6)
    derived_refs = _dedupe([*required_refs, *matched_refs], limit=6)
    block_type = str(layout_section.get("block_type") or layout_section.get("output_type") or "").strip()
    fact_cards = _fact_cards_from_package(
        evidence_package,
        derived_refs,
        collections,
        layout_section=layout_section,
        limit=4,
    )
    raw_layout_text = (
        "；".join(facts)
        or _as_dict(_as_list(layout_section.get("render_blocks"))[0] if _as_list(layout_section.get("render_blocks")) else {}).get("text")
        or ""
    )
    snippet_lead_dropped = _is_snippet_like_public_text(raw_layout_text)
    text = _clean_public_text(
        raw_layout_text,
        420,
    )
    if not derived_refs or not facts:
        return None
    section_title = _public_section_title({}, chapter, index=index, layout_section=layout_section)
    composition = compose_section_paragraph(
        fact_cards=fact_cards,
        claim_unit=ClaimUnit(
            chapter_id=str(chapter.get("chapter_id") or ""),
            evidence_refs=derived_refs,
            claim_strength="directional",
            block_type=block_type,
            section_id=str(layout_section.get("section_id") or ""),
        ),
        block_type=block_type,
        chapter_question=str(chapter.get("chapter_question") or chapter.get("chapter_title") or ""),
    )
    if composition.get("composition_status") == "dropped":
        return None
    composed_paragraph = str(composition.get("paragraph") or "").strip()
    claim = str(composition.get("claim") or text)
    reasoning = str(composition.get("reasoning") or text)
    mechanism = str(composition.get("mechanism") or reasoning)
    counter_evidence = str(composition.get("counter_evidence") or "")
    supporting_facts = _as_list(composition.get("supporting_facts")) or facts
    used_fact_refs = _as_list(composition.get("used_fact_refs")) or derived_refs
    evidence_backed = bool(derived_refs and facts)
    section_plan = _build_section_plan(
        title=section_title,
        block_type=block_type,
        evidence_refs=derived_refs,
        used_fact_refs=used_fact_refs,
        supporting_facts=supporting_facts,
        claim_strength="directional",
        template_removed=False,
        evidence_backed=evidence_backed,
    )
    return {
        "section_id": layout_section.get("section_id"),
        "section_title": section_title,
        "section_plan": section_plan,
        "dynamic_section_title": layout_section.get("dynamic_section_title"),
        "title_source": layout_section.get("title_source"),
        "title_variables": _as_list(layout_section.get("title_variables")),
        "block_title_generation_failed": bool(layout_section.get("block_title_generation_failed")),
        "block_type": block_type,
        "output_type": layout_section.get("output_type") or block_type,
        "section_role": layout_section.get("section_role") or block_type,
        "required_evidence_refs": required_refs,
        "claim": claim,
        "composed_paragraph": composed_paragraph,
        "reasoning": reasoning,
        "mechanism": mechanism,
        "counter_evidence": counter_evidence,
        "actionable": "",
        "decision_implication": "",
        "what_to_verify_next": [],
        "supporting_facts": supporting_facts,
        "used_fact_refs": used_fact_refs,
        "confidence": "low",
        "evidence_refs": derived_refs,
        "render_blocks": [
            {"type": "paragraph", "label": "", "text": composed_paragraph or claim},
        ],
        "public_render": True,
        "layout_generated": True,
        "evidence_backed": evidence_backed,
        "composition_status": composition.get("composition_status") or "legacy",
        "body_composition_status": composition.get("body_composition_status") or "fact_passthrough",
        "variable_explanation": composition.get("variable_explanation") or "",
        "composer_variable_explanation_count": composition.get("composer_variable_explanation_count") or 0,
        "composer_expansion_status": composition.get("composer_expansion_status") or "",
        "composer_target_section_chars": composition.get("composer_target_section_chars") or 0,
        "composer_paragraph_chars": composition.get("composer_paragraph_chars") or 0,
        "observation_only": not evidence_backed,
        "snippet_lead_dropped": snippet_lead_dropped and not text,
        "layout_match_score": 0,
        "layout_match_reason": "layout_fallback",
        "block_evidence_fit_score": layout_section.get("block_evidence_fit_score"),
        "selection_reason": layout_section.get("selection_reason"),
    }


def _layout_by_chapter(micro_layouts: Optional[Sequence[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    return {
        str(layout.get("chapter_id") or "").strip(): dict(layout)
        for layout in list(micro_layouts or [])
        if isinstance(layout, dict) and str(layout.get("chapter_id") or "").strip()
    }


def _layout_sections(layout: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [dict(section) for section in _as_list(_as_dict(layout).get("sections")) if isinstance(section, dict)]


def _unit_layout_match_score(unit: Dict[str, Any], layout_section: Dict[str, Any]) -> int:
    section_id = str(layout_section.get("section_id") or "").strip()
    block_type = str(layout_section.get("block_type") or layout_section.get("output_type") or "").strip()
    title = re.sub(r"\s+", "", str(layout_section.get("section_title") or layout_section.get("title") or "").strip())
    unit_section_id = str(unit.get("section_id") or "").strip()
    unit_block = str(unit.get("block_type") or unit.get("output_type") or unit.get("layout_section_role") or "").strip()
    unit_title = re.sub(r"\s+", "", str(unit.get("section_title") or unit.get("question") or "").strip())
    supported_blocks = claim_supported_block_types(unit)
    if section_id and unit_section_id == section_id:
        return 100
    if block_type and unit_block == block_type:
        return 80
    if block_type and block_type in supported_blocks:
        return 75
    if block_type == "integrated_signal" and supported_blocks:
        return 60
    if title and unit_title and (title == unit_title or title in unit_title or unit_title in title):
        return 50
    required_refs = {str(ref or "").strip() for ref in _as_list(layout_section.get("required_evidence_refs")) if str(ref or "").strip()}
    unit_refs = {str(ref or "").strip() for ref in _as_list(unit.get("evidence_refs")) if str(ref or "").strip()}
    if required_refs and unit_refs and required_refs.intersection(unit_refs):
        return 30
    return 0


def _pop_unit_for_layout(available_units: List[Dict[str, Any]], layout_section: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    best_index = -1
    best_score = 0
    for index, unit in enumerate(available_units):
        score = _unit_layout_match_score(unit, layout_section)
        if score > best_score:
            best_score = score
            best_index = index
    if best_index >= 0 and best_score > 0:
        return available_units.pop(best_index)
    return None


def _section_duplicate_key(section: Dict[str, Any]) -> str:
    refs = ",".join(sorted(str(ref or "").strip() for ref in _as_list(section.get("evidence_refs")) if str(ref or "").strip()))
    claim = re.sub(r"\s+", "", str(section.get("claim") or "").strip().lower())
    return f"{claim[:180]}|{refs}"


def _section_reference_keys(section: Dict[str, Any]) -> set[str]:
    refs = [
        *list(_as_list(section.get("used_fact_refs"))),
        *list(_as_list(section.get("evidence_refs"))),
        *list(_as_list(section.get("required_evidence_refs"))),
    ]
    return {str(ref or "").strip() for ref in refs if str(ref or "").strip()}


def _analysis_section_can_reuse_refs(section: Dict[str, Any]) -> bool:
    """Allow admitted LLM analysis claims to reuse a source for a new angle.

    Reusing the same cited fact is acceptable when the section is traceable to
    a distinct ClaimUnit. Duplicate claim+ref combinations are still removed
    by `_section_duplicate_key`, and repeated fact text is still checked below.
    """

    return bool(
        (section.get("claim_id") or "_llm_extra_" in str(section.get("section_id") or ""))
        and (
            section.get("source_support_map")
            or section.get("llm_claim_block_fallback")
            or section.get("analysis_role")
            or section.get("paragraph_seed")
            or "_llm_extra_" in str(section.get("section_id") or "")
        )
    )


def _fact_text_key(value: Any) -> str:
    text = re.sub(r"\s+", "", str(value or "").strip().lower())
    text = re.sub(r"[，,。；;：:、（）()\"“”‘’'`]+", "", text)
    if len(text) < 12:
        return ""
    return text[:120]


def _section_fact_text_keys(section: Dict[str, Any]) -> set[str]:
    values: List[Any] = [
        section.get("claim"),
        section.get("reasoning"),
        section.get("mechanism"),
        section.get("counter_evidence"),
        section.get("actionable"),
        section.get("decision_implication"),
        *list(_as_list(section.get("supporting_facts"))),
    ]
    for block in _as_list(section.get("render_blocks")):
        if isinstance(block, dict):
            values.append(block.get("text"))
    return {key for value in values for key in [_fact_text_key(value)] if key}


def _chapter_fact_digest(evidence_package: Dict[str, Any]) -> List[str]:
    limit = _env_int("REPORT_CHAPTER_FACT_DIGEST_LIMIT", 18, min_value=0, max_value=80)
    if limit <= 0:
        return []
    facts: List[str] = []
    for collection in PUBLIC_EVIDENCE_COLLECTIONS:
        for item in _as_list(evidence_package.get(collection)):
            if not isinstance(item, dict):
                continue
            if _invalid_metric_item(item):
                continue
            fact = _public_fact_from_item(item, 220)
            if _is_bad_public_fact(fact):
                continue
            if fact:
                facts.append(fact)
    return _dedupe(facts, limit=limit)


def _omitted_chapter_package(
    chapter: Dict[str, Any],
    *,
    chapter_id: str,
    index: int,
    evidence_package: Dict[str, Any],
    raw_units: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "agent": AGENT_NAME,
        "chapter_id": chapter_id,
        "chapter_title": chapter.get("chapter_title") or chapter.get("title") or f"章节 {index}",
        "chapter_question": chapter.get("chapter_question") or chapter.get("chapter_role") or "",
        "lead": "",
        "sections": [],
        "table_packages": [],
        "chapter_summary": {"key_takeaway": "", "confidence": "insufficient", "next_actions": []},
                "evidence_gaps": _as_list(evidence_package.get("missing_evidence")),
                "evidence_quality_summary": _as_dict(evidence_package.get("evidence_quality_summary")),
                "missing_proof_standards": _as_list(evidence_package.get("missing_proof_standards")),
                "omit_from_report": True,
        "internal_reason": "no_public_argument_or_table",
        "dropped_sections": [
            {
                "section_id": unit.get("section_id"),
                "reason": unit.get("internal_reason") or unit.get("quality_status") or "not_public",
            }
            for unit in raw_units
            if isinstance(unit, dict)
        ],
    }


def run_chapter_argument_agent(
    *,
    report_blueprint: Optional[Dict[str, Any]] = None,
    micro_layouts: Optional[Sequence[Dict[str, Any]]] = None,
    argument_units: Optional[Sequence[Dict[str, Any]]] = None,
    table_packages: Optional[Sequence[Dict[str, Any]]] = None,
    chapter_evidence_packages: Optional[Sequence[Dict[str, Any]]] = None,
    llm_client: Any = None,
) -> List[Dict[str, Any]]:
    del llm_client
    report_blueprint = _as_dict(report_blueprint)
    layout_by_chapter = _layout_by_chapter(micro_layouts)
    all_units = [item for item in list(argument_units or []) if isinstance(item, dict)]
    all_units_by_chapter = _by_chapter(all_units)
    units_by_chapter = _by_chapter(public_argument_units(all_units))
    tables_by_chapter = _by_chapter([item for item in list(table_packages or []) if isinstance(item, dict)])
    evidence_by_chapter = {
        str(package.get("chapter_id") or ""): package
        for package in list(chapter_evidence_packages or [])
        if isinstance(package, dict)
    }

    packages: List[Dict[str, Any]] = []
    for index, chapter in enumerate(_as_list(report_blueprint.get("chapters")), start=1):
        if not isinstance(chapter, dict):
            continue
        chapter_id = str(chapter.get("chapter_id") or f"chapter_{index}")
        units = units_by_chapter.get(chapter_id, [])
        raw_units = all_units_by_chapter.get(chapter_id, [])
        evidence_package = _as_dict(evidence_by_chapter.get(chapter_id))
        public_tables = _public_tables(tables_by_chapter.get(chapter_id, []))
        layout_sections = _layout_sections(_as_dict(layout_by_chapter.get(chapter_id)))
        available_units = [dict(unit) for unit in units]
        dropped_sections: List[Dict[str, Any]] = []
        sections: List[Dict[str, Any]] = []
        seen_section_keys = set()
        rendered_evidence_refs: set[str] = set()
        rendered_fact_keys: set[str] = set()
        if layout_sections:
            for section_index, layout_section in enumerate(layout_sections, start=1):
                unit = _pop_unit_for_layout(available_units, layout_section)
                if unit:
                    section = _section_from_unit(
                        unit,
                        chapter,
                        index=section_index,
                        layout_section=layout_section,
                        evidence_package=evidence_package,
                    )
                else:
                    section = _section_from_layout(
                        layout_section,
                        chapter,
                        index=section_index,
                        evidence_package=evidence_package,
                    )
                if section and section.get("template_section_removed"):
                    dropped_sections.append(
                        {
                            "section_id": section.get("section_id") or layout_section.get("section_id"),
                            "block_type": section.get("block_type") or layout_section.get("block_type") or layout_section.get("output_type"),
                            "reason": "template_section_removed",
                            "source": "chapter_argument",
                        }
                    )
                    continue
                if not section or not (section.get("claim") or section.get("reasoning") or section.get("supporting_facts")):
                    dropped_sections.append(
                        {
                            "section_id": layout_section.get("section_id"),
                            "block_type": layout_section.get("block_type") or layout_section.get("output_type"),
                            "reason": "snippet_lead_dropped" if section and section.get("snippet_lead_dropped") else "layout_section_without_public_evidence",
                            "source": "micro_layout",
                        }
                    )
                    continue
                duplicate_key = _section_duplicate_key(section)
                if duplicate_key and duplicate_key in seen_section_keys:
                    dropped_sections.append(
                        {
                            "section_id": section.get("section_id"),
                            "block_type": section.get("block_type"),
                            "reason": "duplicate_claim_and_refs",
                            "source": "chapter_argument",
                        }
                    )
                    continue
                section_refs = _section_reference_keys(section)
                repeated_refs = section_refs.intersection(rendered_evidence_refs)
                if repeated_refs and not _analysis_section_can_reuse_refs(section):
                    dropped_sections.append(
                        {
                            "section_id": section.get("section_id"),
                            "block_type": section.get("block_type"),
                            "reason": "repeated_evidence_id_within_chapter",
                            "source": "chapter_argument",
                            "repeated_refs": sorted(repeated_refs),
                        }
                    )
                    continue
                section_fact_keys = _section_fact_text_keys(section)
                repeated_fact_keys = section_fact_keys.intersection(rendered_fact_keys)
                if repeated_fact_keys and not _analysis_section_can_reuse_refs(section):
                    dropped_sections.append(
                        {
                            "section_id": section.get("section_id"),
                            "block_type": section.get("block_type"),
                            "reason": "repeated_fact_within_chapter",
                            "source": "chapter_argument",
                        }
                    )
                    continue
                seen_section_keys.add(duplicate_key)
                rendered_evidence_refs.update(section_refs)
                rendered_fact_keys.update(section_fact_keys)
                sections.append(section)
        for section_index, unit in enumerate(available_units, start=len(sections) + 1):
            section = _section_from_unit(unit, chapter, index=section_index, evidence_package=evidence_package)
            if not (section.get("claim") or section.get("reasoning") or section.get("supporting_facts")):
                continue
            duplicate_key = _section_duplicate_key(section)
            if duplicate_key and duplicate_key in seen_section_keys:
                dropped_sections.append(
                    {
                        "section_id": section.get("section_id"),
                        "block_type": section.get("block_type"),
                        "reason": "duplicate_claim_and_refs",
                        "source": "chapter_argument",
                    }
                )
                continue
            section_refs = _section_reference_keys(section)
            repeated_refs = section_refs.intersection(rendered_evidence_refs)
            if repeated_refs and not _analysis_section_can_reuse_refs(section):
                dropped_sections.append(
                    {
                        "section_id": section.get("section_id"),
                        "block_type": section.get("block_type"),
                        "reason": "repeated_evidence_id_within_chapter",
                        "source": "chapter_argument",
                        "repeated_refs": sorted(repeated_refs),
                    }
                )
                continue
            section_fact_keys = _section_fact_text_keys(section)
            repeated_fact_keys = section_fact_keys.intersection(rendered_fact_keys)
            if repeated_fact_keys and not _analysis_section_can_reuse_refs(section):
                dropped_sections.append(
                    {
                        "section_id": section.get("section_id"),
                        "block_type": section.get("block_type"),
                        "reason": "repeated_fact_within_chapter",
                        "source": "chapter_argument",
                    }
                )
                continue
            seen_section_keys.add(duplicate_key)
            rendered_evidence_refs.update(section_refs)
            rendered_fact_keys.update(section_fact_keys)
            sections.append(section)

        omitted_observation_sections: List[Dict[str, Any]] = []
        evidence_sections = [
            section
            for section in sections
            if not (section.get("observation_only") and not section.get("evidence_backed"))
        ]
        observation_sections = [
            section
            for section in sections
            if section.get("observation_only") and not section.get("evidence_backed")
        ]
        if evidence_sections:
            omitted_observation_sections = observation_sections
            sections = evidence_sections
        elif observation_sections:
            sections = [observation_sections[0]]
            sections[0]["force_render_observation"] = True
            omitted_observation_sections = observation_sections[1:]

        if not sections and not public_tables:
            packages.append(
                _omitted_chapter_package(
                    chapter,
                    chapter_id=chapter_id,
                    index=index,
                    evidence_package=evidence_package,
                    raw_units=raw_units,
                )
            )
            continue

        evidence_backed_section_count = len([section for section in sections if section.get("evidence_backed")])
        observation_section_count = len([section for section in sections if section.get("observation_only")])
        # The first public section already carries the chapter argument. Repeating it as a
        # lead makes the final report look templated and doubles weak directional claims.
        lead_text = "" if sections else _lead(chapter, units)
        key_takeaway = _clean_public_text(sections[0].get("claim") if sections else "", 240)
        next_actions = _dedupe([_clean_public_text(section.get("actionable"), 220) for section in sections], limit=5)
        mechanisms = _dedupe([_clean_public_text(section.get("mechanism"), 320) for section in sections], limit=3)
        counter_evidence = _dedupe([_clean_public_text(section.get("counter_evidence"), 260) for section in sections], limit=3)
        what_to_verify_next = _dedupe(
            [
                item
                for section in sections
                for item in _as_list(section.get("what_to_verify_next"))
            ],
            limit=5,
        )
        packages.append(
            {
                "agent": AGENT_NAME,
                "chapter_id": chapter_id,
                "chapter_title": chapter.get("chapter_title") or chapter.get("title") or f"章节 {index}",
                "chapter_question": chapter.get("chapter_question") or chapter.get("chapter_role") or "",
                "lead": lead_text,
                "sections": sections,
                "table_packages": public_tables,
                "chapter_summary": {
                    "key_takeaway": key_takeaway,
                    "confidence": sections[0].get("confidence") if sections else "medium",
                    "mechanisms": mechanisms,
                    "counter_evidence": counter_evidence,
                    "next_actions": next_actions,
                    "what_to_verify_next": what_to_verify_next,
                },
                "chapter_fact_digest": _chapter_fact_digest(evidence_package),
                "evidence_gaps": _as_list(evidence_package.get("missing_evidence")),
                "evidence_quality_summary": _as_dict(evidence_package.get("evidence_quality_summary")),
                "missing_proof_standards": _as_list(evidence_package.get("missing_proof_standards")),
                "omit_from_report": False,
                "layout_sections": layout_sections,
                "effective_section_count": evidence_backed_section_count,
                "observation_section_count": observation_section_count,
                "omitted_observation_sections": [
                    {
                        "section_id": section.get("section_id"),
                        "block_type": section.get("block_type"),
                        "section_title": section.get("section_title"),
                        "reason": "observation_only_without_evidence",
                    }
                    for section in omitted_observation_sections
                ],
                "dropped_sections": dropped_sections + [
                    {
                        "section_id": unit.get("section_id"),
                        "reason": unit.get("internal_reason") or unit.get("quality_status") or "not_public",
                    }
                    for unit in raw_units
                    if isinstance(unit, dict) and unit not in units
                ],
            }
        )
    # Body rewrite (section-level paragraph polish) and chapter narrative (P4,
    # cross-section weaving) target different layers and stack cleanly. The old
    # gate disabled body_rewrite whenever P4 was enabled, but P4 itself can be
    # skipped downstream (e.g. insufficient_cited_sections), leaving both polish
    # passes off. ``rewrite_sections_for_report`` already short-circuits when
    # ``REPORT_ENABLE_LLM_BODY_REWRITE`` is off, so invoking it unconditionally is
    # safe and gives the operator independent control of each polish pass.
    if packages:
        packages, _body_rewrite_diagnostics = rewrite_sections_for_report(
            chapter_packages=packages,
            max_llm_calls=body_rewrite_max_sections(),
        )
        for package in packages:
            sections = [section for section in _as_list(package.get("sections")) if isinstance(section, dict)]
            if not sections or package.get("omit_from_report"):
                continue
            summary = _as_dict(package.get("chapter_summary"))
            summary["key_takeaway"] = _clean_public_text(sections[0].get("claim") or "", 240)
            summary["mechanisms"] = _dedupe([_clean_public_text(section.get("mechanism"), 320) for section in sections], limit=3)
            package["chapter_summary"] = summary
    return packages
