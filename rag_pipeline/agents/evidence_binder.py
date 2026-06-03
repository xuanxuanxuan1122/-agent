from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

try:
    from .research_proof_registry import (
        build_mandatory_proof_followups,
        mandatory_proof_checks,
        missing_mandatory_proofs,
        select_research_proof_profile,
    )
    from rag_pipeline.contracts.evidence_quality import apply_evidence_quality_contract, infer_claim_type
except Exception:  # pragma: no cover - direct script mode fallback
    from research_proof_registry import (  # type: ignore
        build_mandatory_proof_followups,
        mandatory_proof_checks,
        missing_mandatory_proofs,
        select_research_proof_profile,
    )
    try:
        from evidence_quality import apply_evidence_quality_contract, infer_claim_type  # type: ignore
    except Exception:  # pragma: no cover - minimal direct script fallback
        def infer_claim_type(item: Dict[str, Any]) -> str:  # type: ignore
            text = " ".join(str(value or "") for value in item.values()).lower()
            if any(term in text for term in {"market size", "market share", "revenue", "profit", "cagr"}):
                return "hard_metric"
            return "industry_analysis"
        apply_evidence_quality_contract = None  # type: ignore


AGENT_NAME = "evidence_binder"
AGENT_DESCRIPTION = "Evidence Binder Agent. Normalizes sources and binds evidence to report chapters."
REJECTED_ROLES = {"rejected", "spam", "irrelevant", "blacklisted", "exclude"}
REJECTED_STATUSES = {"rejected", "spam", "irrelevant", "blacklisted", "exclude"}
WEAK_SEMANTIC_STATUSES = {"weak", "weak_relevance", "appendix"}
PROOF_STANDARD_BY_REPORT_MODE = {
    "quick_market_scan": {
        "min_ab_sources": 1,
        "min_counter_sources": 0,
        "min_metric_sources": 1,
        "min_case_sources": 0,
        "allow_directional_c_sources": True,
        "require_counter": False,
    },
    "deep_industry_report": {
        "min_ab_sources": 2,
        "min_counter_sources": 1,
        "min_metric_sources": 1,
        "min_case_sources": 0,
        "allow_directional_c_sources": True,
        "require_counter": True,
    },
    "investment_due_diligence": {
        "min_ab_sources": 2,
        "min_counter_sources": 1,
        "min_metric_sources": 1,
        "min_case_sources": 1,
        "allow_directional_c_sources": False,
        "require_counter": True,
    },
}
AI_MATERIAL_TOPIC_TERMS = [
    "AI", "人工智能", "算力", "服务器", "GPU", "ASIC", "HBM",
    "半导体", "封装", "PCB", "铜箔", "电子布", "液冷", "导热",
    "光模块", "硅光", "磷化铟", "玻璃基板", "光刻胶", "靶材",
    "电子化学品", "新材料", "材料研发", "AI4S",
]
OFF_TOPIC_TERMS = [
    "五粮液", "白酒", "保险公司", "银行金融科技", "央企红利指数",
    "A股成交额", "铜矿泥石流", "会计差错更正",
]
FAILURE_TEXT_PATTERNS = [
    r"联网分析\s*Agent\s*失败",
    r"Retrieval\.TestUserQueryExceeded",
    r"Retrieval\.",
    r"TestUserQueryExceeded",
    r"query\s+exceed(?:ed)?\s+the\s+limit",
    r"\bTraceback\s*\(",
    r"\bTimeoutError\b",
    r"\bConnectionError\b",
    r"\bHTTPError\b",
]
GENERIC_METRIC_NAMES = {"", "数据指标", "数据点", "关键数据", "比例/增速", "占比", "比例", "增速"}
LOW_CREDIBILITY_DOMAINS = {
    "sohu.com",
    "m.sohu.com",
    "uc.cn",
    "ucweb.com",
    "baijiahao.baidu.com",
    "baike.baidu.com",
    "wk.baidu.com",
    "wenku.baidu.com",
    "jingyan.baidu.com",
    "tieba.baidu.com",
    "zhidao.baidu.com",
    "51baogao.cn",
    "m.51baogao.cn",
    "chinabaogao.com",
    "m.chinabaogao.com",
    "chinairn.com",
    "leetcode.cn",
}
LOW_CREDIBILITY_URL_HINTS = [
    "baidu.com/s?",
    "baidu.com/link?",
    "m.baidu.com",
    "landing",
    "ucbrowser",
]


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _quality_mode() -> str:
    return str(os.getenv("REPORT_QUALITY_MODE") or os.getenv("QUALITY_MODE") or "balanced").strip().lower()


def _strict_quality_mode() -> bool:
    mode = _quality_mode()
    if mode in {"speed", "fast", "loose", "draft", "balanced", "quick_market_scan"}:
        return False
    if mode in {"strict", "deep_strict", "due_diligence", "investment_due_diligence"}:
        return True
    raw = os.getenv("STRICT_EVIDENCE_MODE")
    if raw is None:
        return False
    return str(raw).strip().lower() in {"1", "true", "yes", "on", "strict"}


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return default


TOKEN_PROFILE_INT_DEFAULTS = {
    "lean": {
        "REPORT_MAX_CORE_EVIDENCE_PER_CHAPTER": 12,
        "REPORT_MAX_SUPPORTING_EVIDENCE_PER_CHAPTER": 18,
        "REPORT_MAX_APPENDIX_EVIDENCE_PER_CHAPTER": 24,
    },
    "balanced": {
        "REPORT_MAX_CORE_EVIDENCE_PER_CHAPTER": 18,
        "REPORT_MAX_SUPPORTING_EVIDENCE_PER_CHAPTER": 24,
        "REPORT_MAX_APPENDIX_EVIDENCE_PER_CHAPTER": 48,
    },
    "deep": {
        "REPORT_MAX_CORE_EVIDENCE_PER_CHAPTER": 24,
        "REPORT_MAX_SUPPORTING_EVIDENCE_PER_CHAPTER": 32,
        "REPORT_MAX_APPENDIX_EVIDENCE_PER_CHAPTER": 80,
    },
}


def _profile_default(name: str, default: int) -> int:
    if os.getenv(name) is not None:
        return default
    profile = str(os.getenv("REPORT_TOKEN_PROFILE", "balanced") or "balanced").strip().lower()
    return TOKEN_PROFILE_INT_DEFAULTS.get(profile, TOKEN_PROFILE_INT_DEFAULTS["balanced"]).get(name, default)


def _profiled_env_int(name: str, default: int) -> int:
    return _env_int(name, _profile_default(name, default))


def _directional_c_min_confidence() -> float:
    raw = os.getenv("REPORT_DIRECTIONAL_C_MIN_CONFIDENCE", "0.55")
    try:
        return max(0.0, min(1.0, float(raw)))
    except (TypeError, ValueError):
        return 0.55


def _can_use_directional_c(*, level: str, confidence: float, role: str, semantic_status: str, explicit_appendix_only: bool) -> bool:
    if _strict_quality_mode() or explicit_appendix_only:
        return False
    if str(level or "").upper() != "C":
        return False
    if role in REJECTED_ROLES or semantic_status in REJECTED_STATUSES or semantic_status in WEAK_SEMANTIC_STATUSES:
        return False
    return confidence >= _directional_c_min_confidence()


def _evidence_source_key(item: Dict[str, Any]) -> str:
    source = _as_dict(item.get("source"))
    raw_url = str(source.get("url") or source.get("source_url") or item.get("source_url") or item.get("url") or "").strip()
    if raw_url:
        parsed = urlparse(raw_url)
        domain = parsed.netloc.lower().removeprefix("www.")
        path = re.sub(r"/+$", "", parsed.path or "")
        query = f"?{parsed.query}" if parsed.query else ""
        if domain:
            return f"url:{domain}{path}{query}".lower()
    document_ref = str(
        source.get("document_id")
        or source.get("doc_id")
        or source.get("page_ref")
        or item.get("document_id")
        or item.get("doc_id")
        or item.get("page_ref")
        or ""
    ).strip().lower()
    if document_ref:
        return f"doc:{document_ref}"
    title = re.sub(r"\s+", "", str(source.get("title") or item.get("source_title") or "").strip().lower())
    publisher = re.sub(r"\s+", "", str(source.get("publisher") or source.get("source") or item.get("source_text") or "").strip().lower())
    date = re.sub(r"\s+", "", str(source.get("date") or item.get("date") or item.get("period") or "").strip().lower())
    if title:
        return f"title:{title}|{publisher}|{date}"
    for value in [item.get("source_id"), source.get("id"), item.get("source_ref"), item.get("ref")]:
        text = re.sub(r"\s+", "", str(value or "").strip().lower())
        if text:
            return f"id:{text}"
    return ""


def _distinct_source_count(items: Sequence[Dict[str, Any]]) -> int:
    keys = {_evidence_source_key(item) for item in items if isinstance(item, dict)}
    return len({key for key in keys if key})


def _evidence_source_domain_key(item: Dict[str, Any]) -> str:
    source = _as_dict(item.get("source"))
    raw_url = str(source.get("url") or source.get("source_url") or item.get("source_url") or item.get("url") or "").strip()
    if raw_url:
        parsed = urlparse(raw_url)
        domain = parsed.netloc.lower().removeprefix("www.")
        if domain:
            return f"domain:{domain}"
    return _evidence_source_key(item)


def _distinct_source_domain_count(items: Sequence[Dict[str, Any]]) -> int:
    keys = {_evidence_source_domain_key(item) for item in items if isinstance(item, dict)}
    return len({key for key in keys if key})


VERIFIED_SOURCE_STATUSES = {"readpage_verified", "document_verified"}
DOCUMENT_SOURCE_RE = re.compile(
    r"(\.pdf(?:$|\?)|annual[-_ ]?report|financial[-_ ]?report|filing|prospectus|"
    r"announcement|disclosure|standard|whitepaper|policy|regulation|official|gov\.|\.gov|exchange)",
    re.I,
)


def _source_verification_status(item: Dict[str, Any]) -> str:
    source = _as_dict(item.get("source"))
    explicit = str(
        item.get("source_verification_status")
        or source.get("source_verification_status")
        or item.get("verification_status")
        or source.get("verification_status")
        or ""
    ).strip().lower()
    if explicit in {"search_result_only", "readpage_verified", "document_verified", "inaccessible"}:
        return explicit
    raw_url = str(source.get("url") or source.get("source_url") or item.get("source_url") or item.get("url") or "").strip()
    document_ref = str(
        source.get("document_id")
        or source.get("doc_id")
        or source.get("page_ref")
        or item.get("document_id")
        or item.get("doc_id")
        or item.get("page_ref")
        or ""
    ).strip()
    if not raw_url and not document_ref:
        return "inaccessible"
    source_text = " ".join(
        str(value or "")
        for value in [raw_url, source.get("source_type"), source.get("title"), item.get("source_type"), item.get("source_family")]
    )
    if document_ref or DOCUMENT_SOURCE_RE.search(source_text):
        return "document_verified"
    if bool(
        source.get("readpage_verified")
        or source.get("auto_readpage")
        or source.get("readpage_priority")
        or item.get("readpage_verified")
        or item.get("auto_readpage")
        or item.get("readpage_priority")
    ):
        return "readpage_verified"
    for key in ("mainText", "main_text", "markdown", "content", "text", "quote", "page_content"):
        if str(source.get(key) or item.get(key) or "").strip():
            return "readpage_verified"
    return "search_result_only"


def _has_verified_source(item: Dict[str, Any]) -> bool:
    return _source_verification_status(item) in VERIFIED_SOURCE_STATUSES


def _claim_type_for_hypothesis(hypothesis: Dict[str, Any], goals: Sequence[Dict[str, Any]]) -> str:
    related_goal_text = " ".join(
        str(goal.get("question") or goal.get("evidence_goal") or goal.get("dimension_name") or "")
        for goal in goals
        if str(goal.get("hypothesis_id") or "").strip() == str(hypothesis.get("hypothesis_id") or "").strip()
    )
    return infer_claim_type(
        {
            "claim_type": hypothesis.get("claim_type") or hypothesis.get("conclusion_type"),
            "proof_role": hypothesis.get("proof_role"),
            "evidence_type": hypothesis.get("evidence_type"),
            "metric_definitions": hypothesis.get("metric_definitions"),
            "metric": " ".join(str(_as_dict(item).get("metric") or "") for item in _as_list(hypothesis.get("metric_definitions"))),
            "hypothesis_statement": hypothesis.get("statement") or hypothesis.get("hypothesis_statement"),
            "evidence_goal": related_goal_text,
            "dimension_name": hypothesis.get("dimension_name"),
        }
    )


def _directional_c_threshold_for_claim(claim_type: str) -> int:
    if claim_type == "forecast_judgment":
        return 3
    if claim_type in {"industry_analysis", "product_event", "case_signal"}:
        return 2
    return 10_000


def _optional_int(payload: Dict[str, Any], key: str, default: int) -> int:
    value = payload.get(key)
    if key not in payload or value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _compact(value: Any, max_chars: int = 260) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "..."


def _dedupe(values: Iterable[Any], *, limit: int = 100) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        text = _compact(value, 240)
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


def _source_text(source: Dict[str, Any]) -> str:
    title = str(source.get("title") or source.get("source") or source.get("name") or "").strip()
    date = str(source.get("date") or source.get("period") or "").strip()
    url = str(source.get("url") or source.get("link") or "").strip()
    return " | ".join(part for part in [title, date, url] if part) or "未命名来源"


def _fact_text(item: Dict[str, Any]) -> str:
    for key in (
        "fact",
        "clean_fact",
        "content",
        "clean_content",
        "data_point",
        "writer_evidence",
        "claim",
        "takeaway",
        "answer",
        "conclusion",
    ):
        text = _compact(item.get(key), 420)
        if text:
            return text
    metric = _compact(item.get("metric") or item.get("indicator"), 80)
    value = _compact(item.get("value") or item.get("display_value"), 120)
    if metric and value:
        return f"{metric}: {value}"
    return ""


def _contains_failure_text(value: Any) -> bool:
    text = str(value or "")
    if not text:
        return False
    return any(re.search(pattern, text, flags=re.I) for pattern in FAILURE_TEXT_PATTERNS)


def _is_failed_evidence_item(item: Dict[str, Any]) -> bool:
    if not isinstance(item, dict):
        return False
    status_blob = " ".join(
        str(item.get(key) or "")
        for key in ("status", "error", "exception", "message", "answer", "fact", "content")
    )
    if _contains_failure_text(status_blob):
        return True
    try:
        return _contains_failure_text(json.dumps(item, ensure_ascii=False))
    except (TypeError, ValueError):
        return False


def _metric_value_issue(metric: Any, value: Any) -> str:
    metric_text = str(metric or "").strip()
    value_text = str(value or "").strip()
    if not metric_text or not value_text or not re.search(r"\d", value_text):
        return ""
    combined = f"{metric_text} {value_text}"
    has_percent = bool(re.search(r"[%％]|pct|百分点", value_text, flags=re.I))
    has_currency = bool(re.search(r"亿元|万元|万美元|亿美元|万亿元|人民币|美元|欧元|日元|元", value_text))
    has_count = bool(re.search(r"\d[\d,.]*\s*(家|台|辆|片|个|座|条|起|项|人|次|份|宗|例)", value_text))
    ratio_metric = bool(re.search(r"份额|占比|比例|比重|渗透率|国产化率|良品率|毛利率|净利率|率$|CAGR|增速|增长率", metric_text, flags=re.I))
    money_metric = bool(re.search(r"营收|收入|净利润|利润|销售额|金额|投资额|融资额|交易额|资本开支|市场规模|规模|估值", metric_text))
    count_metric = bool(re.search(r"数量|企业数|公司数|客户数|订单数|产线数|项目数|出货量|销量|产量", metric_text))
    if ratio_metric and has_currency:
        return "ratio_metric_with_currency_value"
    if ratio_metric and has_count and not count_metric:
        return "ratio_metric_with_count_value"
    if money_metric and has_count and not count_metric:
        return "money_metric_with_count_value"
    if money_metric and has_percent and not re.search(r"增速|增长|率|CAGR|同比|环比|变化|变动", metric_text):
        return "money_metric_with_percent_value"
    if re.search(r"市场规模|规模", metric_text) and has_percent and not re.search(r"增速|增长|CAGR|占比|份额|比例", metric_text):
        return "market_size_with_percent_value"
    if re.search(r"市场份额|份额|占比", metric_text) and has_currency:
        return "share_metric_with_currency_value"
    if re.search(r"价格|单价", metric_text) and has_percent and not re.search(r"涨|降|跌|增|减|变动|变化|同比|环比|折扣", combined):
        return "price_metric_with_plain_percent_value"
    return ""


def _fact_is_metric_pair_only(fact: str, metric: Any, value: Any) -> bool:
    metric_text = str(metric or "").strip()
    value_text = str(value or "").strip()
    if not fact or not metric_text or not value_text:
        return False
    key = _normalize_key(fact)
    pair_key = _normalize_key(f"{metric_text}:{value_text}")
    loose_pair_key = _normalize_key(f"{metric_text}{value_text}")
    if key in {pair_key, loose_pair_key}:
        return True
    return len(fact) <= 48 and metric_text in fact and value_text in fact


def _unanchored_financial_metric_issue(fact: str, metric: Any, value: Any) -> str:
    metric_text = str(metric or "").strip()
    if not _fact_is_metric_pair_only(fact, metric, value):
        return ""
    if re.search(r"营收|营业收入|收入|净利润|归母净利润|利润|毛利率|净利率|现金流|市值|股价|融资额|估值", metric_text):
        return "unanchored_financial_metric_pair"
    return ""


def _dimension_text(item: Dict[str, Any]) -> str:
    return (
        str(item.get("dimension_name") or "").strip()
        or str(item.get("dimension") or "").strip()
        or str(item.get("evidence_goal") or "").strip()
        or str(item.get("question") or "").strip()
        or "核心研究问题"
    )


def _source_payload(item: Dict[str, Any]) -> Dict[str, Any]:
    source = _as_dict(item.get("source"))
    if source:
        return source
    source_text = str(item.get("source") or "").strip()
    if source_text:
        return {"title": source_text, "name": source_text}
    for candidate in _as_list(item.get("key_sources")):
        if isinstance(candidate, dict):
            return dict(candidate)
        candidate_text = str(candidate or "").strip()
        if candidate_text:
            return {"title": candidate_text, "name": candidate_text}
    return {}


def _source_level(item: Dict[str, Any]) -> str:
    source = _source_payload(item)
    return str(item.get("source_level") or source.get("credibility") or source.get("source_level") or "").strip().upper()


def _source_domain(source: Dict[str, Any]) -> str:
    raw_url = str(source.get("url") or source.get("link") or "").strip()
    domain = urlparse(raw_url).netloc.lower().removeprefix("www.")
    return domain


def score_source(item: Dict[str, Any]) -> Dict[str, Any]:
    source = _source_payload(item)
    title = " ".join(
        str(source.get(key) or item.get(key) or "")
        for key in ["title", "source", "name", "source_type", "url"]
    ).lower()
    domain = _source_domain(source)
    text = f"{domain} {title}"

    level = _source_level(item)
    source_type = str(source.get("source_type") or source.get("type") or "").strip().lower()
    reason = "upstream_source_level"
    score = {"A": 0.95, "B": 0.78, "C": 0.46, "D": 0.12}.get(level, 0.0)
    official_signal = (
        domain.endswith(".gov.cn")
        or domain.endswith(".gov")
        or any(term in text for term in ["stats.gov", "ndrc", "miit", "mof.gov", "pbc.gov", "csrc", "统计局", "政府", "监管"])
    )
    filing_signal = any(term in text for term in ["sse.com.cn", "szse.cn", "hkexnews", "cninfo", "annual report", "prospectus", "公告", "年报", "招股书", "交易所"])
    authoritative_signal = any(term in text for term in ["reuters", "bloomberg", "caixin", "yicai", "协会", "白皮书", "券商", "研究报告", "研报"])
    low_credibility_signal = (
        domain in LOW_CREDIBILITY_DOMAINS
        or any(domain.endswith("." + item) for item in LOW_CREDIBILITY_DOMAINS)
        or any(hint in text for hint in LOW_CREDIBILITY_URL_HINTS)
        or any(term in text for term in ["zhihu", "csdn", "baijiahao", "tieba", "weibo", "toutiao", "wenku", "doc88", "文库", "论坛", "搜狐", "百家号", "UC"])
    )

    if official_signal:
        level, score, source_type, reason = "A", 0.95, source_type or "government_official", "official_domain_or_title"
    elif filing_signal:
        level, score, source_type, reason = "A", 0.92, source_type or "filing_or_exchange", "filing_or_exchange_signal"
    elif authoritative_signal and level not in {"A"}:
        level, score, source_type, reason = "B", 0.78, source_type or "authoritative_secondary", "authoritative_secondary_signal"
    elif low_credibility_signal and level not in {"A", "B"}:
        level, score, source_type, reason = "D", 0.12, source_type or "low_credibility_web", "low_credibility_domain_or_title"
    elif not level:
        if official_signal:
            level, score, source_type, reason = "A", 0.95, source_type or "government_official", "official_domain_or_title"
        elif filing_signal:
            level, score, source_type, reason = "A", 0.92, source_type or "filing_or_exchange", "filing_or_exchange_signal"
        elif authoritative_signal:
            level, score, source_type, reason = "B", 0.78, source_type or "authoritative_secondary", "authoritative_secondary_signal"
        elif low_credibility_signal:
            level, score, source_type, reason = "D", 0.12, source_type or "low_credibility_web", "low_credibility_domain_or_title"
        else:
            level, score, source_type, reason = "C", 0.46, source_type or "general_web", "default_general_web"

    return {
        "source_level": level or "C",
        "source_type": source_type,
        "source_score": score or 0.46,
        "source_reason": reason,
        "can_support_core_claim": (level or "C") in {"A", "B"},
    }


def _source_family_from_item(item: Dict[str, Any]) -> str:
    source = _source_payload(item)
    source_type = str(source.get("source_type") or item.get("source_type") or "").strip().lower()
    text = " ".join(
        [
            source_type,
            str(source.get("title") or "").lower(),
            str(source.get("url") or "").lower(),
            str(item.get("evidence_type") or "").lower(),
            str(item.get("proof_role") or "").lower(),
        ]
    )
    if source_type in {"official", "government", "financial_report", "annual_report", "prospectus", "exchange"}:
        return "official/filing"
    if source_type in {"research", "academic", "industry_report", "association", "consulting", "market_research", "brokerage", "whitepaper", "authoritative_secondary"}:
        return "research/association"
    if any(term in text for term in ["customer", "case", "procurement", "order", "contract", "client", "tender"]):
        return "company/case"
    if source_type in {"media", "news"}:
        return "news/secondary"
    return "unknown"


def _confidence(item: Dict[str, Any]) -> float:
    try:
        value = float(item.get("confidence", 0.5))
    except (TypeError, ValueError):
        value = 0.5
    return max(0.0, min(1.0, value))


def _source_quality_score(level: str, confidence: float) -> float:
    base = {
        "A": 0.95,
        "B": 0.78,
        "C": 0.46,
        "D": 0.12,
    }.get(str(level or "").upper(), 0.35)
    return round(max(0.0, min(1.0, base * 0.75 + confidence * 0.25)), 4)


def _evidence_role(item: Dict[str, Any]) -> str:
    return str(item.get("evidence_role") or item.get("role") or item.get("use_in_report") or "").strip().lower()


def _canonical_role(item: Dict[str, Any]) -> str:
    role = _evidence_role(item)
    if role == "exclude":
        return "rejected"
    if role:
        return role
    level = _source_level(item)
    if level == "A":
        return "core"
    if level == "B":
        return "supporting"
    if level in {"C", "D"}:
        return "clue"
    return "appendix"


def _level_distribution(items: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    distribution: Dict[str, int] = {"A": 0, "B": 0, "C": 0, "D": 0, "UNKNOWN": 0}
    for item in items:
        if not isinstance(item, dict):
            continue
        level = str(item.get("source_level") or _source_level(item) or "").strip().upper()
        if level not in distribution:
            level = "UNKNOWN"
        distribution[level] += 1
    return {key: value for key, value in distribution.items() if value}


def _normalize_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"(?<=\d)[,，](?=\d{3}(?:\D|$))", "", text)
    text = re.sub(r"\s+", "", text)
    return re.sub(r"[^0-9a-z_\u4e00-\u9fff.%％+\-]+", "", text)


def _meaningful_overlap(left: Any, right: Any, *, min_chars: int = 2) -> bool:
    left_key = _normalize_key(left)
    right_key = _normalize_key(right)
    if not left_key or not right_key:
        return False
    left_numbers = set(re.findall(r"[+\-]?\d+(?:\.\d+)?%?", left_key))
    right_numbers = set(re.findall(r"[+\-]?\d+(?:\.\d+)?%?", right_key))
    if (left_numbers or right_numbers) and not (left_numbers & right_numbers):
        return False
    if left_key in right_key or right_key in left_key:
        return True
    overlap = set(left_key) & set(right_key)
    return len(overlap) >= max(min_chars, min(len(left_key), len(right_key)) // 3)


def _term_in_text(term: str, text: str) -> bool:
    term = str(term or "").strip().lower()
    if not term:
        return False
    if re.fullmatch(r"[a-z0-9]{1,3}", term):
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text, re.I))
    return term in text


def _topic_anchor_groups(topic_text: str) -> List[List[str]]:
    groups: List[List[str]] = []
    if re.search(r"\bAI\b|人工智能|大模型|生成式|AIGC", topic_text, re.I):
        groups.append(["人工智能", "ai", "aigc", "大模型", "生成式ai", "生成式人工智能"])
    if re.search(r"中国|国内", topic_text, re.I):
        groups.append(["中国", "国内", "china", "chinese"])
    if re.search(r"新能源汽车|新能源车|动力电池|锂电", topic_text):
        groups.append(["新能源汽车", "新能源车", "动力电池", "锂电"])
    if re.search(r"半导体|芯片|集成电路", topic_text, re.I):
        groups.append(["半导体", "芯片", "集成电路", "semiconductor", "chip"])
    return groups


def _evidence_topic_text(evidence: Dict[str, Any]) -> str:
    source = _as_dict(evidence.get("source"))
    return " ".join(
        str(value or "")
        for value in [
            evidence.get("fact"),
            evidence.get("clean_fact"),
            evidence.get("content"),
            evidence.get("summary"),
            evidence.get("metric"),
            evidence.get("value"),
            source.get("title"),
            source.get("quote"),
            source.get("url"),
        ]
    ).lower()


def evidence_matches_report_topic(evidence: Dict[str, Any], report_blueprint: Dict[str, Any]) -> bool:
    text = _evidence_topic_text(evidence)
    blueprint_text = json.dumps(
        {
            "research_object": report_blueprint.get("research_object"),
            "narrative": report_blueprint.get("narrative"),
            "chapters": [
                {
                    "chapter_title": _as_dict(chapter).get("chapter_title"),
                    "chapter_question": _as_dict(chapter).get("chapter_question"),
                }
                for chapter in _as_list(report_blueprint.get("chapters"))
            ],
        },
        ensure_ascii=False,
    )
    is_ai_material_report = (
        ("AI" in blueprint_text or "人工智能" in blueprint_text)
        and ("材料" in blueprint_text or "新材料" in blueprint_text)
    )
    if is_ai_material_report:
        if any(term in text for term in OFF_TOPIC_TERMS):
            return False
        return any(term in text for term in AI_MATERIAL_TOPIC_TERMS)
    for group in _topic_anchor_groups(blueprint_text):
        if not any(_term_in_text(term, text) for term in group):
            return False
    return True


@dataclass
class SourceRegistry:
    sources: List[Dict[str, Any]] = field(default_factory=list)
    _seen: Dict[str, str] = field(default_factory=dict)

    def register(self, source: Dict[str, Any]) -> str:
        source = _as_dict(source)
        key = _normalize_key(source.get("url") or _source_text(source))
        if not key:
            key = _normalize_key(_source_text(source))
        if key in self._seen:
            return self._seen[key]
        ref_index = len(self.sources) + 1
        ref = f"[{ref_index}]"
        payload = {
            "source_id": f"SRC-{ref_index:03d}",
            "ref": ref,
            "title": str(source.get("title") or source.get("source") or source.get("name") or "未命名来源").strip(),
            "date": str(source.get("date") or source.get("period") or "").strip(),
            "url": str(source.get("url") or source.get("link") or "").strip(),
            "source_type": str(source.get("source_type") or source.get("type") or "").strip(),
            "credibility": str(source.get("credibility") or source.get("source_level") or "").strip(),
        }
        self.sources.append(payload)
        self._seen[key] = ref
        return ref

    def footnotes(self) -> List[str]:
        return [
            f"{source['ref']} {_source_text(source)}"
            for source in self.sources
        ]


def _rewrite_inline_numeric_citations(text: str, source_ref: str) -> str:
    if not source_ref:
        return text
    rewritten = re.sub(r"\[(?:id:)?\d{1,4}\]", source_ref, str(text or ""), flags=re.I)
    rewritten = re.sub(rf"(?:{re.escape(source_ref)}\s*){{2,}}", source_ref, rewritten)
    rewritten = re.sub(rf"{re.escape(source_ref)}\s*(?:、|,|，|and|和)\s*{re.escape(source_ref)}", source_ref, rewritten, flags=re.I)
    rewritten = re.sub(rf"{re.escape(source_ref)}(?:\s|[;；/&?])+{re.escape(source_ref)}", source_ref, rewritten, flags=re.I)
    return rewritten


def _append_raw(result: List[Dict[str, Any]], item: Dict[str, Any], *, source_hint: str = "", dimension: str = "") -> None:
    if _is_failed_evidence_item(item):
        return
    fact = _fact_text(item)
    if not fact:
        return
    if _contains_failure_text(fact):
        return
    copied = dict(item)
    if dimension:
        copied.setdefault("dimension", dimension)
    if source_hint:
        copied.setdefault("source_hint", source_hint)
    result.append(copied)


def collect_raw_evidence(
    *,
    evidence_package: Optional[Dict[str, Any]] = None,
    structured_analysis: Optional[Dict[str, Any]] = None,
    child_outputs: Optional[Dict[str, Dict[str, Any]]] = None,
    evidence_pool: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    evidence_package = _as_dict(evidence_package)
    structured_analysis = _as_dict(structured_analysis)
    result: List[Dict[str, Any]] = []

    for dimension, payload in _as_dict(evidence_package.get("per_dimension")).items():
        payload = _as_dict(payload)
        for item in (
            _as_list(payload.get("analysis_inputs"))
            + _as_list(payload.get("clean_facts"))
            + _as_list(payload.get("top_evidence"))
        ):
            if isinstance(item, dict):
                _append_raw(result, item, source_hint="evidence_package.per_dimension", dimension=str(dimension))

    for item in (
        _as_list(evidence_package.get("core_evidence"))
        + _as_list(evidence_package.get("supporting_evidence"))
        + _as_list(evidence_package.get("clue_evidence"))
        + _as_list(evidence_package.get("appendix_evidence"))
        + _as_list(evidence_package.get("analysis_ready_evidence"))
        + _as_list(evidence_package.get("clean_evidence_list"))
    ):
        if isinstance(item, dict):
            _append_raw(result, item, source_hint="evidence_package")

    for item in list(evidence_pool or []):
        if not isinstance(item, dict):
            continue
        _append_raw(result, item, source_hint="evidence_pool")
        for point in _as_list(item.get("raw_data_points")):
            if isinstance(point, dict):
                copied = dict(point)
                copied.setdefault("dimension", _dimension_text(item))
                _append_raw(result, copied, source_hint="evidence_pool.raw_data_points")

    for child_name, child in _as_dict(child_outputs).items():
        if not isinstance(child, dict):
            continue
        for point in _as_list(child.get("raw_data_points")):
            if isinstance(point, dict):
                copied = dict(point)
                copied.setdefault("key_sources", child.get("key_sources") or [])
                _append_raw(result, copied, source_hint=f"{child_name}.raw_data_points")
        if child.get("answer"):
            _append_raw(
                result,
                {
                    "answer": child.get("answer"),
                    "key_sources": child.get("key_sources") or [],
                    "dimension": child.get("dimension") or child.get("agent") or "核心研究问题",
                },
                source_hint=f"{child_name}.answer",
            )

    for item in _as_list(structured_analysis.get("evidence_analyses")):
        if isinstance(item, dict):
            _append_raw(result, item, source_hint="structured_analysis.evidence_analyses")

    return result


def normalize_and_register_sources(raw_items: Sequence[Dict[str, Any]], registry: SourceRegistry) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    seen = set()
    for index, item in enumerate(raw_items, start=1):
        if not isinstance(item, dict):
            continue
        if _is_failed_evidence_item(item):
            continue
        fact = _fact_text(item)
        if not fact:
            continue
        if _contains_failure_text(fact):
            continue
        metric_raw = _compact(item.get("metric") or item.get("indicator"), 80)
        value_raw = _compact(item.get("value") or item.get("display_value"), 120)
        if metric_raw in GENERIC_METRIC_NAMES and value_raw and re.search(r"\d", value_raw):
            if _fact_is_metric_pair_only(fact, metric_raw, value_raw):
                continue
            metric_raw = ""
        metric_issue = _metric_value_issue(metric_raw, value_raw)
        if not metric_issue:
            metric_issue = _unanchored_financial_metric_issue(fact, metric_raw, value_raw)
        if metric_issue and _fact_is_metric_pair_only(fact, metric_raw, value_raw):
            continue
        source = _source_payload(item)
        source_score = score_source(item)
        if source:
            source = {
                **source,
                "credibility": source.get("credibility") or source_score.get("source_level"),
                "source_level": source.get("source_level") or source_score.get("source_level"),
                "source_type": source.get("source_type") or source_score.get("source_type"),
            }
        source_ref = registry.register(source) if source else ""
        fact = _rewrite_inline_numeric_citations(fact, source_ref)
        key = (_normalize_key(fact), source_ref or _normalize_key(_source_text(source)))
        if key in seen:
            continue
        seen.add(key)
        explicit_id = str(item.get("evidence_id") or item.get("id") or "").strip()
        evidence_id = explicit_id if explicit_id.startswith("EV-") else f"EV-{len(normalized) + 1:03d}"
        role = _canonical_role(item)
        level = str(source_score.get("source_level") or _source_level(item)).strip().upper()
        if _strict_quality_mode() and level in {"C", "D"} and role not in REJECTED_ROLES:
            role = "clue"
        if role in {"", "appendix"} and level == "A":
            role = "core"
        elif role in {"", "appendix"} and level == "B":
            role = "supporting"
        elif role in {"", "appendix"} and level in {"C", "D"}:
            role = "clue"
        semantic_status = str(item.get("semantic_status") or "").strip().lower()
        explicit_appendix_only = bool(item.get("appendix_only"))
        confidence = _confidence(item)
        directional_c = _can_use_directional_c(
            level=level,
            confidence=confidence,
            role=role,
            semantic_status=semantic_status,
            explicit_appendix_only=explicit_appendix_only,
        )
        appendix_only = (
            explicit_appendix_only
            or level == "D"
            or role in {"appendix", "appendix_only"}
            or (role == "clue" and not directional_c)
            or semantic_status in WEAK_SEMANTIC_STATUSES
        )
        excluded = role in REJECTED_ROLES or semantic_status in REJECTED_STATUSES
        quality_score = _source_quality_score(level, confidence)
        allowed_use = str(item.get("allowed_use") or _as_dict(item.get("evidence_card")).get("allowed_use") or "").strip()
        if not allowed_use:
            if level in {"A", "B"} and role == "core" and not appendix_only:
                allowed_use = "core_claim"
            elif level in {"A", "B"} and role in {"core", "supporting"} and not appendix_only:
                allowed_use = "supporting"
            elif directional_c:
                allowed_use = "directional_signal"
            elif level == "C":
                allowed_use = "clue"
            else:
                allowed_use = "appendix_only"
        evidence_card = {
            **_as_dict(item.get("evidence_card")),
            "fact": fact,
            "source_level": level,
            "source_family": _source_family_from_item({**item, "source": source}),
            "proof_role": str(item.get("proof_role") or "").strip().lower() or "support",
            "period": _compact(item.get("period") or item.get("date") or _as_dict(source).get("date"), 80),
            "allowed_use": allowed_use,
        }
        normalized_item = {
            "evidence_id": evidence_id,
            "ref": evidence_id,
            "source_refs": [source_ref] if source_ref else [],
            "source_ref": source_ref,
            "source": source,
            "source_text": _source_text(source) if source else "",
            "source_level": level,
            "source_family": evidence_card.get("source_family"),
            "source_type": source_score.get("source_type"),
            "source_score": source_score.get("source_score"),
            "source_reason": source_score.get("source_reason"),
            "can_support_core_claim": source_score.get("can_support_core_claim"),
            "confidence": confidence,
            "evidence_quality_score": quality_score,
            "dimension": _dimension_text(item),
            "evidence_goal": _compact(item.get("evidence_goal") or item.get("question"), 220),
            "chapter_id": str(item.get("chapter_id") or _as_dict(item.get("search_task")).get("chapter_id") or "").strip(),
            "chapter_title": _compact(item.get("chapter_title") or _as_dict(item.get("search_task")).get("chapter_title"), 160),
            "chapter_question": _compact(item.get("chapter_question") or _as_dict(item.get("search_task")).get("chapter_question"), 220),
            "evidence_goal_id": str(item.get("evidence_goal_id") or _as_dict(item.get("search_task")).get("evidence_goal_id") or "").strip(),
            "task_id": str(item.get("task_id") or "").strip(),
            "hypothesis_id": str(item.get("hypothesis_id") or "").strip(),
            "hypothesis_statement": _compact(item.get("hypothesis_statement") or item.get("hypothesis"), 260),
            "proof_role": str(item.get("proof_role") or "").strip().lower(),
            "claim_type": str(item.get("claim_type") or item.get("conclusion_type") or "").strip().lower(),
            "evidence_type": str(item.get("evidence_type") or item.get("intent") or "").strip().lower(),
            "fact": fact,
            "metric": "" if metric_issue else metric_raw,
            "value": "" if metric_issue else value_raw,
            "metric_validation_status": "invalid" if metric_issue else "valid",
            "metric_validation_issue": metric_issue,
            "period": _compact(item.get("period") or item.get("date") or _as_dict(source).get("date"), 80),
            "evidence_role": role,
            "allowed_use": allowed_use,
            "evidence_card": evidence_card,
            "source_hint": str(item.get("source_hint") or "").strip(),
            "appendix_only": appendix_only,
            "followup_seed": bool(item.get("followup_seed")) or role == "clue" or level == "D",
            "can_support_claim_if_corrobated": (bool(item.get("can_support_claim_if_corrobated")) or level == "C") and not _strict_quality_mode(),
            "usage_tier": str(item.get("usage_tier") or ("directional_signal" if directional_c else "")).strip(),
            "excluded": excluded,
            "raw": item,
        }
        if apply_evidence_quality_contract is not None:
            normalized_item = apply_evidence_quality_contract(
                normalized_item,
                strict_quality=_strict_quality_mode(),
                directional_c_min_confidence=_directional_c_min_confidence(),
            )
            normalized_item["excluded"] = excluded
            normalized_item["raw"] = item
        normalized.append(normalized_item)
    normalized.sort(key=lambda item: (item.get("excluded"), item.get("appendix_only"), -float(item.get("confidence") or 0.0)))
    return normalized


def _evidence_is_essential(item: Dict[str, Any]) -> bool:
    if item.get("excluded"):
        return False
    level = str(item.get("source_level") or "").strip().upper()
    role = str(item.get("evidence_role") or "").strip().lower()
    proof_role = str(item.get("proof_role") or "").strip().lower()
    evidence_type = str(item.get("evidence_type") or "").strip().lower()
    if level in {"A", "B"}:
        return True
    if role in {"core", "supporting"} and not item.get("appendix_only"):
        return True
    if proof_role in {"counter", "metric", "case"} or evidence_type in {"counter", "metric", "case"}:
        return True
    if item.get("can_support_core_claim"):
        return True
    return False


def _evidence_rank(item: Dict[str, Any]) -> Tuple[int, float, str]:
    if item.get("excluded"):
        base = 90
    elif item.get("appendix_only"):
        base = 60
    else:
        base = 20
    level = str(item.get("source_level") or "").strip().upper()
    level_bonus = {"A": -20, "B": -12, "C": 5, "D": 15}.get(level, 10)
    if _evidence_is_essential(item):
        base -= 30
    confidence = float(item.get("confidence") or 0.0)
    return (base + level_bonus, -confidence, str(item.get("evidence_id") or ""))


def compress_normalized_items_for_reporting(normalized_items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items = [item for item in normalized_items if isinstance(item, dict)]
    if _strict_quality_mode() or not _env_flag("REPORT_ENABLE_EVIDENCE_CLUSTER_COMPRESSION", True):
        return items
    max_items = _env_int("REPORT_MAX_NORMALIZED_EVIDENCE", 1200)
    if max_items <= 0 or len(items) <= max_items:
        return items
    per_cluster_limit = max(1, _env_int("REPORT_EVIDENCE_CLUSTER_MAX_ITEMS", 4))
    essential = [item for item in items if _evidence_is_essential(item)]
    optional = [item for item in items if not _evidence_is_essential(item)]
    kept: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    cluster_counts: Dict[Tuple[str, str, str, str], int] = {}

    def keep(item: Dict[str, Any], *, enforce_cluster: bool) -> None:
        evidence_id = str(item.get("evidence_id") or id(item))
        if evidence_id in seen_ids:
            return
        cluster_key = (
            _normalize_key(item.get("fact"))[:180],
            str(item.get("source_ref") or item.get("source_family") or "").lower(),
            str(item.get("proof_role") or "").lower(),
            str(item.get("hypothesis_id") or item.get("evidence_goal_id") or "").lower(),
        )
        if enforce_cluster and cluster_counts.get(cluster_key, 0) >= per_cluster_limit:
            return
        kept.append(item)
        seen_ids.add(evidence_id)
        cluster_counts[cluster_key] = cluster_counts.get(cluster_key, 0) + 1

    for item in sorted(essential, key=_evidence_rank):
        keep(item, enforce_cluster=False)
    remaining_budget = max(0, max_items - len(kept))
    if remaining_budget <= 0:
        return kept
    for item in sorted(optional, key=_evidence_rank):
        if len(kept) >= max_items:
            break
        keep(item, enforce_cluster=True)
    return kept


def _chapter_match_score(item: Dict[str, Any], chapter: Dict[str, Any], goals: Sequence[Dict[str, Any]]) -> int:
    chapter_id = str(chapter.get("chapter_id") or "").strip()
    chapter_title = str(chapter.get("chapter_title") or chapter.get("title") or "").strip()
    if chapter_id and chapter_id in {
        str(item.get("chapter_id") or "").strip(),
        str(item.get("dimension_id") or "").strip(),
        str(_as_dict(item.get("raw")).get("chapter_id") or "").strip(),
    }:
        return 100
    if chapter_title and chapter_title in {
        str(item.get("chapter_title") or "").strip(),
        str(item.get("dimension") or "").strip(),
        str(_as_dict(item.get("raw")).get("chapter_title") or "").strip(),
    }:
        return 80
    fields = [
        chapter.get("chapter_id"),
        chapter.get("chapter_title"),
        chapter.get("dimension"),
        chapter.get("chapter_question"),
        chapter.get("chapter_role"),
    ]
    chapter_tokens = {_normalize_key(value) for value in fields if str(value or "").strip()}
    goal_texts = []
    for goal in goals:
        if isinstance(goal, dict):
            goal_texts.extend([goal.get("goal_id"), goal.get("question"), goal.get("evidence_goal")])
        else:
            goal_texts.append(goal)
    chapter_tokens.update(_normalize_key(value) for value in goal_texts if str(value or "").strip())
    item_fields = [
        item.get("dimension"),
        item.get("evidence_goal"),
        item.get("task_id"),
        item.get("fact"),
        item.get("metric"),
    ]
    item_tokens = [_normalize_key(value) for value in item_fields if str(value or "").strip()]
    score = 0
    for token in item_tokens:
        if not token:
            continue
        for chapter_token in chapter_tokens:
            if not chapter_token:
                continue
            if token == chapter_token:
                score += 6
            elif token in chapter_token or chapter_token in token:
                score += 2
    return score


def _split_evidence_items(
    items: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    core: List[Dict[str, Any]] = []
    supporting: List[Dict[str, Any]] = []
    clue: List[Dict[str, Any]] = []
    table: List[Dict[str, Any]] = []
    appendix: List[Dict[str, Any]] = []
    for item in items:
        if item.get("excluded"):
            continue
        role = str(item.get("evidence_role") or "").strip().lower()
        level = str(item.get("source_level") or "").strip().upper()
        allowed_use = str(item.get("allowed_use") or "").strip()
        is_directional = allowed_use == "directional_signal" and level == "C" and not item.get("appendix_only")
        if is_directional:
            supporting.append(item)
            if item.get("metric") or item.get("value") or re.search(r"\d", str(item.get("fact") or "")):
                table.append(item)
            continue
        if role == "core" and level != "D" and not item.get("appendix_only"):
            if _strict_quality_mode() and level not in {"A", "B"}:
                clue.append(item)
                appendix.append(item)
                continue
            core.append(item)
            if item.get("metric") or item.get("value") or re.search(r"\d", str(item.get("fact") or "")):
                table.append(item)
            continue
        if role == "supporting":
            if _strict_quality_mode() and level not in {"A", "B"}:
                clue.append(item)
                appendix.append(item)
                continue
            supporting.append(item)
            if level not in {"D"} and not item.get("appendix_only") and (
                item.get("metric") or item.get("value") or re.search(r"\d", str(item.get("fact") or ""))
            ):
                table.append(item)
            continue
        if role == "clue" or level == "D":
            clue.append(item)
            appendix.append(item)
            continue
        if item.get("appendix_only") or role in {"appendix", "appendix_only"}:
            appendix.append(item)
            continue
        supporting.append(item)
        if item.get("metric") or item.get("value") or re.search(r"\d", str(item.get("fact") or "")):
            table.append(item)
    core_limit = _profiled_env_int("REPORT_MAX_CORE_EVIDENCE_PER_CHAPTER", 24)
    supporting_limit = _profiled_env_int("REPORT_MAX_SUPPORTING_EVIDENCE_PER_CHAPTER", 32)
    appendix_limit = _profiled_env_int("REPORT_MAX_APPENDIX_EVIDENCE_PER_CHAPTER", 80)
    try:
        table_budget = int(os.getenv("REPORT_MAX_BODY_TABLES", "6") or 6)
    except (TypeError, ValueError):
        table_budget = 6
    table_limit = max(18, min(60, table_budget * 5))
    return core[:core_limit], supporting[:supporting_limit], clue[:appendix_limit], table[:table_limit], appendix[:appendix_limit]


def _package_quality_summary(
    *,
    candidates: Sequence[Dict[str, Any]],
    selected: Sequence[Dict[str, Any]],
    core: Sequence[Dict[str, Any]],
    supporting: Sequence[Dict[str, Any]],
    clue: Sequence[Dict[str, Any]],
    table: Sequence[Dict[str, Any]],
    appendix: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    rejected = [item for item in selected if isinstance(item, dict) and item.get("excluded")]
    core_levels = _level_distribution(core)
    all_levels = _level_distribution(selected)
    core_ab = core_levels.get("A", 0) + core_levels.get("B", 0)
    core_cd = core_levels.get("C", 0) + core_levels.get("D", 0)
    scores = [float(item.get("evidence_quality_score") or 0.0) for item in core if isinstance(item, dict)]
    quality_score = round(sum(scores) / max(len(scores), 1), 4) if scores else 0.0
    return {
        "candidate_count": len([item for item in candidates if isinstance(item, dict)]),
        "kept_count": len([item for item in selected if isinstance(item, dict) and not item.get("excluded")]),
        "rejected_count": len(rejected),
        "core_evidence_count": len(core),
        "supporting_evidence_count": len(supporting),
        "clue_evidence_count": len(clue),
        "table_evidence_count": len(table),
        "appendix_only_count": len(appendix),
        "source_level_distribution": all_levels,
        "core_source_level_distribution": core_levels,
        "core_ab_source_count": core_ab,
        "core_cd_source_count": core_cd,
        "evidence_quality_score": quality_score,
    }


def _detect_conflicts(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for item in items:
        metric = _normalize_key(item.get("metric") or item.get("dimension"))
        if not metric:
            continue
        value = _compact(item.get("value"), 80)
        if not value:
            continue
        buckets.setdefault(metric, []).append(item)
    conflicts: List[Dict[str, Any]] = []
    for metric, bucket in buckets.items():
        values = _dedupe([item.get("value") for item in bucket], limit=8)
        if len(values) <= 1:
            continue
        conflicts.append(
            {
                "conflict_id": f"CF-{len(conflicts) + 1:03d}",
                "metric": metric,
                "values": values,
                "evidence_refs": [item.get("ref") for item in bucket if item.get("ref")],
                "description": "同一指标存在不同口径或数值，需要在正文中说明边界。",
            }
        )
    return conflicts


def normalize_metric_table(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    metrics: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict) or item.get("excluded"):
            continue
        if item.get("metric_validation_status") == "invalid":
            continue
        if _is_failed_evidence_item(item):
            continue
        metric_name = _compact(item.get("metric") or item.get("indicator"), 80)
        value = _compact(item.get("value") or item.get("display_value"), 120)
        fact = str(item.get("fact") or "")
        if metric_name in GENERIC_METRIC_NAMES and value and re.search(r"\d", value):
            continue
        if _contains_failure_text(fact):
            continue
        if _metric_value_issue(metric_name, value):
            continue
        if not metric_name and not value and not re.search(r"\d", fact):
            continue
        raw = _as_dict(item.get("raw"))
        period = _compact(item.get("period") or raw.get("period") or raw.get("date"), 80)
        if not period:
            match = re.search(r"20\d{2}(?:\s*[Ee]|\s*年|\s*Q[1-4]|\s*H[12])?(?:\s*[-~至]\s*20\d{2}(?:\s*[Ee]|\s*年)?)?", fact)
            period = _compact(match.group(0), 80) if match else ""
        unit = _compact(raw.get("unit"), 40)
        if not unit:
            unit_pattern = r"(亿元|万元|万美元|亿美元|人民币|美元|万台|台|辆|吨|%|pct|CAGR|GW|GWh|MW|平方米|亩)"
            unit_match = re.search(unit_pattern, value, re.I) if value else None
            if not unit_match:
                unit_match = re.search(rf"\d+(?:\.\d+)?\s*{unit_pattern}", fact, re.I)
            unit = unit_match.group(1) if unit_match and unit_match.lastindex else (unit_match.group(0) if unit_match else "")
        scope = _compact(raw.get("scope") or raw.get("region"), 80)
        if not scope:
            scope_match = re.search(r"(中国|全球|美国|欧洲|日本|东南亚|亚太|华东|华南|华北|一线城市|二线城市|全国|海外)", fact)
            scope = scope_match.group(0) if scope_match else ""
        calculation_method = _compact(raw.get("calculation_method") or raw.get("method"), 80)
        if not calculation_method:
            if re.search(r"预测|预计|20\d{2}\s*[Ee]\b|estimate|forecast", fact, re.I):
                calculation_method = "预测/估算"
            elif value or metric_name:
                calculation_method = "原文披露"
        raw_subject = _compact(raw.get("subject") or raw.get("company") or raw.get("entity"), 120)
        subject = raw_subject or _compact(item.get("subject") or item.get("company") or item.get("dimension") or item.get("evidence_goal"), 120)
        source_level = str(item.get("source_level") or "").strip().upper()
        public_ready = bool(subject and metric_name and value and scope and period and unit and source_level in {"A", "B"})
        metrics.append(
            {
                "metric_id": f"MET-{len(metrics) + 1:03d}",
                "metric_name": metric_name or _compact(item.get("dimension") or item.get("evidence_goal"), 80),
                "subject": subject,
                "scope": scope,
                "period": period,
                "unit": unit,
                "value": value or _compact(fact, 160),
                "calculation_method": calculation_method,
                "public_key_data_ready": public_ready,
                "口径完整": bool(scope and period and unit),
                "missing_fields": [
                    field
                    for field, field_value in {
                        "subject": subject,
                        "metric": metric_name,
                        "value": value,
                        "scope": scope,
                        "period": period,
                        "unit": unit,
                        "source_level_ab": source_level if source_level in {"A", "B"} else "",
                    }.items()
                    if not field_value
                ],
                "source_ref": item.get("source_ref"),
                "evidence_ref": item.get("ref") or item.get("evidence_id"),
                "source_level": source_level,
                "confidence": item.get("confidence"),
                "hypothesis_id": item.get("hypothesis_id"),
            }
        )
    return metrics


def _item_matches_hypothesis(item: Dict[str, Any], hypothesis: Dict[str, Any], goals: Sequence[Dict[str, Any]]) -> bool:
    hypothesis_id = str(hypothesis.get("hypothesis_id") or "").strip()
    if hypothesis_id and str(item.get("hypothesis_id") or "").strip() == hypothesis_id:
        return True
    h_text = _normalize_key(hypothesis.get("statement") or hypothesis.get("hypothesis_statement"))
    item_text = _normalize_key(" ".join(str(item.get(key) or "") for key in ["dimension", "evidence_goal", "fact", "metric"]))
    if h_text and item_text and (h_text in item_text or item_text in h_text):
        return True
    hypothesis_terms = _as_list(hypothesis.get("evidence_goal_ids")) + _as_list(hypothesis.get("metric_definitions"))
    for term in hypothesis_terms:
        if isinstance(term, dict):
            term = " ".join(str(term.get(key) or "") for key in ("metric_name", "subject", "scope"))
        if _meaningful_overlap(term, item_text, min_chars=3):
            return True
    for goal in goals:
        if not isinstance(goal, dict):
            continue
        if str(goal.get("hypothesis_id") or "").strip() != hypothesis_id:
            continue
        goal_terms = [
            goal.get("dimension_id"),
            goal.get("dimension_name"),
            goal.get("question"),
            goal.get("goal_id"),
            goal.get("evidence_goal"),
        ]
        if any(_meaningful_overlap(term, item_text, min_chars=3) for term in goal_terms):
            return True
        goal_text = _normalize_key(" ".join(str(goal.get(key) or "") for key in ["dimension_id", "dimension_name", "question", "goal_id"]))
        if goal_text and item_text and (goal_text in item_text or item_text in goal_text):
            return True
    return False


def _report_proof_mode(research_plan: Dict[str, Any], hypothesis: Dict[str, Any]) -> str:
    explicit = str(
        research_plan.get("report_mode")
        or research_plan.get("proof_mode")
        or os.getenv("REPORT_PROOF_MODE")
        or ""
    ).strip().lower()
    family = str(research_plan.get("report_family") or research_plan.get("report_type") or "").strip().lower()
    deep_family = "deep" in family or "industry_deep_report" in family
    if explicit in PROOF_STANDARD_BY_REPORT_MODE:
        if explicit == "quick_market_scan" and deep_family and not _env_flag("REPORT_ALLOW_QUICK_PROOF_FOR_DEEP", False):
            return "deep_industry_report"
        return explicit
    if any(term in family for term in ["due", "dd", "尽调", "投资"]):
        return "investment_due_diligence"
    if any(term in family for term in ["deep", "深度"]):
        return "deep_industry_report"
    proof_standard = str(hypothesis.get("proof_standard") or "").strip().lower()
    if proof_standard == "strong" and _strict_quality_mode():
        return "deep_industry_report"
    return "quick_market_scan"


def _coverage_profile(research_plan: Dict[str, Any], hypothesis: Dict[str, Any]) -> Dict[str, Any]:
    return dict(PROOF_STANDARD_BY_REPORT_MODE[_report_proof_mode(research_plan, hypothesis)])


def _hypothesis_has_metric_need(hypothesis: Dict[str, Any]) -> bool:
    if _as_list(hypothesis.get("metric_definitions")):
        return True
    required_types = {str(item or "").strip().lower() for item in _as_list(hypothesis.get("required_evidence_types"))}
    if {"metric", "data", "indicator"}.intersection(required_types):
        return True
    bundle = _as_dict(hypothesis.get("evidence_bundle"))
    return bool(_as_list(bundle.get("metric_terms")) or _as_list(hypothesis.get("must_prove")))


def build_coverage_matrix(
    *,
    research_plan: Dict[str, Any],
    normalized_items: Sequence[Dict[str, Any]],
    metric_table: Sequence[Dict[str, Any]],
    report_blueprint: Optional[Dict[str, Any]] = None,
    query: str = "",
    proof_profile: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    hypotheses = [item for item in _as_list(research_plan.get("hypotheses")) if isinstance(item, dict)]
    goals = [item for item in _as_list(research_plan.get("evidence_goals")) if isinstance(item, dict)]
    if not hypotheses:
        return []
    proof_profile = _as_dict(proof_profile) or select_research_proof_profile(
        query=query,
        research_plan=research_plan,
        report_blueprint=_as_dict(report_blueprint),
    )
    matrix: List[Dict[str, Any]] = []
    for hypothesis in hypotheses:
        hypothesis_id = str(hypothesis.get("hypothesis_id") or "").strip()
        relevant = [
            item
            for item in normalized_items
            if isinstance(item, dict) and not item.get("excluded") and _item_matches_hypothesis(item, hypothesis, goals)
        ]
        levels = _level_distribution(relevant)
        usable = [
            item
            for item in relevant
            if not item.get("appendix_only")
            and str(item.get("source_level") or "").strip().upper() in {"A", "B"}
            and str(item.get("allowed_use") or "").strip() in {"core_claim", "supporting", ""}
            and str(item.get("analysis_readiness") or "").strip() in {"", "decision_ready"}
        ]
        profile = _coverage_profile(research_plan, hypothesis)
        directional = [
            item
            for item in relevant
            if profile.get("allow_directional_c_sources")
            and not item.get("appendix_only")
            and str(item.get("source_level") or "").strip().upper() == "C"
            and str(item.get("allowed_use") or "").strip() == "directional_signal"
        ]
        usable_for_direction = usable + directional
        usable_levels = _level_distribution(usable)
        ab_count = usable_levels.get("A", 0) + usable_levels.get("B", 0)
        ab_candidates = [
            item
            for item in relevant
            if not item.get("appendix_only")
            and str(item.get("source_level") or "").strip().upper() in {"A", "B"}
            and str(item.get("allowed_use") or "").strip() in {"core_claim", "supporting", "supporting_context", ""}
        ]
        distinct_ab_source_count = _distinct_source_count(ab_candidates)
        traceable_ab_candidates = [
            item
            for item in ab_candidates
            if _evidence_source_key(item) and _source_verification_status(item) != "inaccessible"
        ]
        verified_ab_candidates = [item for item in traceable_ab_candidates if _has_verified_source(item)]
        distinct_traceable_ab_source_count = _distinct_source_count(traceable_ab_candidates)
        distinct_verified_ab_source_count = _distinct_source_count(verified_ab_candidates)
        distinct_primary_source_count = _distinct_source_count(
            [
                item
                for item in verified_ab_candidates
                if str(item.get("source_level") or "").strip().upper() == "A"
                or str(item.get("source_family") or "").strip().lower() in {"official/filing", "research/association"}
            ]
        )
        directional_count = len(directional)
        directional_distinct_count = _distinct_source_domain_count(directional)
        claim_type = _claim_type_for_hypothesis(hypothesis, goals)
        hard_metric_claim = claim_type == "hard_metric"
        strict_mode = _strict_quality_mode()
        counter_items = [
            item
            for item in usable_for_direction
            if str(item.get("proof_role") or "").lower() == "counter"
            or bool(_as_dict(item.get("raw")).get("counter_evidence"))
        ]
        distinct_counter_source_count = _distinct_source_count(
            [item for item in counter_items if _evidence_source_key(item) and _source_verification_status(item) != "inaccessible"]
        )
        distinct_verified_counter_source_count = _distinct_source_count([item for item in counter_items if _has_verified_source(item)])
        counter_count = distinct_verified_counter_source_count
        counter_signal_count = len(
            [
                item
                for item in counter_items
                if str(item.get("proof_role") or "").lower() == "counter"
                or bool(_as_dict(item.get("raw")).get("counter_evidence"))
            ]
        )
        counter_clue_count = len(
            [
                item
                for item in relevant
                if str(item.get("proof_role") or "").lower() == "counter"
                and str(item.get("source_level") or "").strip().upper() == "C"
            ]
        )
        metric_source_count = len(
            [
                item
                for item in usable_for_direction
                if str(item.get("proof_role") or item.get("evidence_type") or "").lower() == "metric"
                or bool(item.get("metric") or item.get("value"))
            ]
        )
        metric_gap_items = [
            item
            for item in usable_for_direction
            if _as_list(item.get("metric_proof_gaps") or _as_dict(item.get("evidence_card")).get("metric_proof_gaps"))
        ]
        case_source_count = len(
            [
                item
                for item in usable_for_direction
                if str(item.get("proof_role") or item.get("evidence_type") or "").lower() == "case"
                or str(item.get("source_family") or "") == "company/case"
            ]
        )
        source_families = sorted(
            {
                str(item.get("source_family") or "unknown")
                for item in usable_for_direction
                if str(item.get("source_family") or "unknown")
            }
        )
        hypothesis_goals = [goal for goal in goals if str(goal.get("hypothesis_id") or "").strip() == hypothesis_id]
        coverage_requirements = _as_dict(_as_dict(research_plan.get("evidence_coverage_requirements")).get("per_hypothesis"))
        required_sources = _optional_int(coverage_requirements, "min_A_or_B_sources", int(profile.get("min_ab_sources") or 0))
        required_sources = _optional_int(coverage_requirements, "min_ab_sources", required_sources)
        required_counter = _optional_int(coverage_requirements, "min_counter_sources", int(profile.get("min_counter_sources") or 0))
        metric_default = int(profile.get("min_metric_sources") or 0) if _hypothesis_has_metric_need(hypothesis) else 0
        required_metric = _optional_int(coverage_requirements, "min_metric_sources", metric_default)
        required_case = _optional_int(coverage_requirements, "min_case_sources", int(profile.get("min_case_sources") or 0))
        required_diversity = [str(item) for item in _as_list(coverage_requirements.get("source_diversity")) if str(item or "").strip()]
        required_levels = _as_list(hypothesis.get("required_source_levels")) or ["A", "B"]
        for goal in hypothesis_goals:
            try:
                goal_min_sources = int(goal.get("min_sources") or 0)
            except (TypeError, ValueError):
                goal_min_sources = 0
            if _report_proof_mode(research_plan, hypothesis) != "quick_market_scan":
                required_sources = max(required_sources, goal_min_sources)
        missing: List[str] = []
        degradation_notes: List[str] = []
        ab_requirement_satisfied = distinct_verified_ab_source_count >= required_sources
        directional_ready = False
        if not ab_requirement_satisfied and not strict_mode and not hard_metric_claim:
            c_threshold = _directional_c_threshold_for_claim(claim_type)
            if distinct_traceable_ab_source_count >= 1:
                ab_requirement_satisfied = True
                degradation_notes.append("verified_ab_shortfall_traceable_ab_substituted")
            elif directional_distinct_count >= c_threshold:
                ab_requirement_satisfied = True
                directional_ready = True
                degradation_notes.append("directional_corroborated_from_independent_c_sources")
        if not ab_requirement_satisfied:
            missing.append("insufficient_ab_sources")
        counter_required = bool(hypothesis.get("counter_evidence_required", profile.get("require_counter", False)))
        if counter_required and required_counter > 0 and counter_count < required_counter:
            if not strict_mode and not hard_metric_claim and not bool(hypothesis.get("counter_evidence_required")) and (distinct_traceable_ab_source_count or directional_distinct_count >= 2):
                degradation_notes.append("counter_evidence_missing_advisory")
            else:
                missing.append("counter_evidence_missing")
        if required_metric > 0 and metric_source_count < required_metric:
            if not strict_mode and not hard_metric_claim and (distinct_traceable_ab_source_count or directional_distinct_count >= 2):
                degradation_notes.append("metric_evidence_degraded_for_non_hard_claim")
            else:
                missing.append("metric_evidence_missing")
        if metric_gap_items:
            if hard_metric_claim or strict_mode:
                if "metric_scope_period_unit_incomplete" not in missing:
                    missing.append("metric_scope_period_unit_incomplete")
            else:
                degradation_notes.append("metric_scope_period_unit_incomplete_advisory")
        if required_case > 0 and case_source_count < required_case:
            missing.append("case_evidence_missing")
        if required_diversity and not set(required_diversity).issubset(set(source_families)):
            missing.append("source_diversity_missing")
        if levels and not (levels.get("A", 0) + levels.get("B", 0)) and not directional_count:
            missing.append("only_c_or_lower_sources")
        hypothesis_metrics = [
            metric
            for metric in metric_table
            if isinstance(metric, dict)
            and (
                str(metric.get("hypothesis_id") or "").strip() == hypothesis_id
                or any(str(metric.get("evidence_ref") or "") in _as_list(item.get("source_refs")) or str(metric.get("evidence_ref") or "") == str(item.get("ref") or item.get("evidence_id") or "") for item in relevant)
            )
        ]
        if not hypothesis_metrics and _as_list(hypothesis.get("metric_definitions")):
            if not strict_mode and not hard_metric_claim and (distinct_traceable_ab_source_count or directional_distinct_count >= 2):
                degradation_notes.append("metric_definition_unfilled_advisory")
            else:
                missing.append("metric_definition_unfilled")
        incomplete_metrics = [metric for metric in hypothesis_metrics if _as_list(metric.get("missing_fields"))]
        if incomplete_metrics:
            if not strict_mode and not hard_metric_claim and (distinct_traceable_ab_source_count or directional_distinct_count >= 2):
                degradation_notes.append("metric_scope_period_unit_incomplete_advisory")
            else:
                missing.append("metric_scope_period_unit_incomplete")
        proof_checks = mandatory_proof_checks(
            proof_profile,
            relevant,
            hypothesis_metrics,
            hypothesis=hypothesis,
        )
        missing_proofs = missing_mandatory_proofs(proof_checks)
        if missing_proofs and "mandatory_proof_missing" not in missing:
            if not strict_mode and not hard_metric_claim and (distinct_traceable_ab_source_count or directional_distinct_count >= _directional_c_threshold_for_claim(claim_type)):
                degradation_notes.append("mandatory_proof_missing_advisory")
            else:
                missing.append("mandatory_proof_missing")
        if not missing and directional_ready:
            claim_status = "directional_ready"
        elif not missing:
            claim_status = "decision_ready"
        elif distinct_traceable_ab_source_count or directional_count:
            claim_status = "directional"
        else:
            claim_status = "context_only"
        if claim_status == "decision_ready":
            readiness_level = "decision_ready"
        elif claim_status == "directional_ready":
            readiness_level = "directional_ready"
        elif missing and not (distinct_traceable_ab_source_count or directional_count):
            readiness_level = "blocked"
        else:
            readiness_level = "context_only"
        matrix.append(
            {
                "hypothesis_id": hypothesis_id,
                "hypothesis_statement": hypothesis.get("statement") or hypothesis.get("hypothesis_statement"),
                "proof_standard": hypothesis.get("proof_standard") or "medium",
                "report_proof_mode": _report_proof_mode(research_plan, hypothesis),
                "required_source_levels": required_levels,
                "claim_type": claim_type,
                "required_ab_sources": required_sources,
                "actual_ab_sources": ab_count,
                "distinct_ab_source_count": distinct_ab_source_count,
                "distinct_traceable_ab_source_count": distinct_traceable_ab_source_count,
                "distinct_verified_ab_source_count": distinct_verified_ab_source_count,
                "distinct_primary_source_count": distinct_primary_source_count,
                "directional_c_sources": directional_count,
                "directional_c_distinct_sources": directional_distinct_count,
                "evidence_degradation_notes": degradation_notes,
                "counter_evidence_count": counter_count,
                "counter_signal_count": counter_signal_count,
                "distinct_counter_source_count": distinct_counter_source_count,
                "distinct_verified_counter_source_count": distinct_verified_counter_source_count,
                "counter_clue_count": counter_clue_count,
                "metric_source_count": metric_source_count,
                "case_source_count": case_source_count,
                "source_diversity": source_families,
                "required_source_diversity": required_diversity,
                "metric_count": len(hypothesis_metrics),
                "complete_metric_count": len([metric for metric in hypothesis_metrics if not _as_list(metric.get("missing_fields"))]),
                "metric_proof_gap_count": len(metric_gap_items),
                "metric_proof_gaps": sorted(
                    {
                        str(gap)
                        for item in metric_gap_items
                        for gap in _as_list(item.get("metric_proof_gaps") or _as_dict(item.get("evidence_card")).get("metric_proof_gaps"))
                        if str(gap or "").strip()
                    }
                ),
                "proof_profile_id": proof_profile.get("profile_id"),
                "mandatory_proof_checks": proof_checks,
                "missing_mandatory_proofs": missing_proofs,
                "evidence_refs": [item.get("ref") or item.get("evidence_id") for item in usable_for_direction[:12] if item.get("ref") or item.get("evidence_id")],
                "source_level_distribution": levels,
                "usable_source_level_distribution": usable_levels,
                "usable_with_directional_source_level_distribution": _level_distribution(usable_for_direction),
                "claim_status": claim_status,
                "readiness_level": readiness_level,
                "decision_ready": not missing,
                "blocking_gaps": missing,
            }
        )
    return matrix


def _proof_followup_query(item: Dict[str, Any]) -> Dict[str, Any]:
    gaps = _as_list(item.get("blocking_gaps"))
    hypothesis = _compact(item.get("hypothesis_statement") or item.get("hypothesis_id"), 120)
    query_parts = [hypothesis]
    source_priority = ["official", "filing", "research_report"]
    proof_role = "support"
    evidence_type = "data"
    missing_proofs = [proof for proof in _as_list(item.get("missing_mandatory_proofs")) if isinstance(proof, dict)]
    if "mandatory_proof_missing" in gaps and missing_proofs:
        primary_proof = _as_dict(missing_proofs[0])
        query_parts.extend([primary_proof.get("query"), primary_proof.get("label")])
        proof_role = str(primary_proof.get("proof_role") or proof_role)
        evidence_type = str(primary_proof.get("evidence_type") or evidence_type)
        if _as_list(primary_proof.get("source_priority")):
            source_priority = [
                str(value)
                for value in _as_list(primary_proof.get("source_priority"))
                if str(value or "").strip()
            ]
    if "insufficient_ab_sources" in gaps:
        query_parts.extend(["A/B来源", "官方", "公告", "财报", "协会", "权威研报"])
    if "counter_evidence_missing" in gaps:
        query_parts.extend(["反证", "失败案例", "风险", "投诉", "事故", "撤单"])
        proof_role = "counter"
        evidence_type = "counter"
    if "metric_definition_unfilled" in gaps or "metric_scope_period_unit_incomplete" in gaps:
        query_parts.extend(["指标口径", "市场规模", "增速", "单位", "期间", "范围"])
        if evidence_type != "counter":
            evidence_type = "data"
    if "metric_evidence_missing" in gaps:
        query_parts.extend(["metric", "price", "capacity", "margin", "shipment", "penetration"])
        if proof_role == "support":
            proof_role = "metric"
            evidence_type = "metric"
    if "case_evidence_missing" in gaps:
        query_parts.extend(["customer", "certification", "order", "mass production", "supply contract"])
        if proof_role == "support":
            proof_role = "case"
            evidence_type = "case"
    if "source_diversity_missing" in gaps or "only_c_or_lower_sources" in gaps:
        query_parts.extend(["official", "filing", "annual report", "association", "brokerage research", "company case"])
    query = _compact(" ".join(part for part in query_parts if str(part or "").strip()), 220)
    lane_targets = {
        "counter": ["news_event", "filing_company", "market_research"],
        "metric": ["official_data", "market_research"],
        "case": ["customer_case", "filing_company"],
    }.get(proof_role, ["official_data", "filing_company", "market_research"])
    if missing_proofs and _as_list(missing_proofs[0].get("lane_targets")):
        lane_targets = [
            str(value)
            for value in _as_list(missing_proofs[0].get("lane_targets"))
            if str(value or "").strip()
        ]
    return {
        "query": query,
        "agent": "iqs",
        "targets_gap": hypothesis or str(item.get("hypothesis_id") or ""),
        "dimension_name": hypothesis,
        "evidence_goal": hypothesis,
        "hypothesis_id": item.get("hypothesis_id"),
        "hypothesis_statement": item.get("hypothesis_statement"),
        "proof_role": proof_role,
        "counter_evidence": proof_role == "counter",
        "evidence_type": evidence_type,
        "lane_targets": lane_targets,
        "source_priority": source_priority,
        "blocking_gaps": gaps,
        "missing_mandatory_proofs": missing_proofs[:5],
        "proof_profile_id": item.get("proof_profile_id"),
    }


def _proof_gap_priority(gap: str) -> int:
    return {
        "mandatory_proof_missing": 0,
        "insufficient_ab_sources": 0,
        "insufficient_ab_core_sources": 0,
        "only_c_or_lower_sources": 1,
        "metric_evidence_missing": 2,
        "metric_definition_unfilled": 2,
        "metric_scope_period_unit_incomplete": 3,
        "counter_evidence_missing": 4,
        "insufficient_counter_sources": 4,
        "case_evidence_missing": 5,
        "source_diversity_missing": 6,
        "needs_corroboration": 7,
    }.get(str(gap or ""), 20)


def build_evidence_refinement_plan(
    *,
    coverage_matrix: Sequence[Dict[str, Any]],
    chapter_packages: Sequence[Dict[str, Any]],
    proof_follow_up_queries: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    tasks: List[Dict[str, Any]] = []
    gap_counts: Dict[str, int] = {}

    def add_task(task: Dict[str, Any], *, source: str) -> None:
        task = dict(_as_dict(task))
        query = _compact(task.get("query") or task.get("follow_up_query"), 220)
        if not query:
            return
        gaps = [str(gap) for gap in _as_list(task.get("blocking_gaps")) if str(gap or "").strip()]
        for gap in gaps:
            gap_counts[gap] = gap_counts.get(gap, 0) + 1
        priority = min([_proof_gap_priority(gap) for gap in gaps] or [20])
        tasks.append(
            {
                **task,
                "query": query,
                "agent": str(task.get("agent") or "iqs"),
                "source": source,
                "priority": int(task.get("priority") if task.get("priority") not in {None, ""} else priority),
            }
        )

    for row in coverage_matrix:
        row = _as_dict(row)
        gaps = _as_list(row.get("blocking_gaps"))
        for gap in gaps:
            gap_counts[str(gap)] = gap_counts.get(str(gap), 0) + 1
        for task in build_mandatory_proof_followups(
            _as_list(row.get("mandatory_proof_checks")),
            hypothesis=row,
            profile={"profile_id": row.get("proof_profile_id")},
        ):
            add_task(task, source="mandatory_proof_profile")
    for item in proof_follow_up_queries:
        if isinstance(item, dict):
            add_task(item, source="coverage_matrix")

    for package in chapter_packages:
        package = _as_dict(package)
        for item in _as_list(package.get("proof_follow_up_queries")):
            if isinstance(item, dict):
                add_task(item, source="chapter_proof_gap")
        for item in _as_list(package.get("missing_evidence")):
            item = _as_dict(item)
            query = item.get("follow_up_query")
            if not query:
                continue
            gap = str(item.get("type") or "needs_corroboration")
            add_task(
                {
                    "query": query,
                    "agent": "iqs",
                    "targets_gap": package.get("chapter_title") or package.get("chapter_id"),
                    "dimension_name": package.get("chapter_title"),
                    "evidence_goal": item.get("suggestion") or package.get("chapter_question"),
                    "blocking_gaps": [gap],
                    "proof_role": "counter" if "counter" in gap else "support",
                    "evidence_type": "counter" if "counter" in gap else "data",
                },
                source="chapter_missing_evidence",
            )

    seen = set()
    deduped: List[Dict[str, Any]] = []
    for task in sorted(tasks, key=lambda item: (int(item.get("priority") or 20), str(item.get("targets_gap") or ""), str(item.get("query") or ""))):
        key = (
            str(task.get("targets_gap") or ""),
            str(task.get("proof_role") or ""),
            str(task.get("query") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(task)
    return {
        "status": "needs_refinement" if deduped else "sufficient_for_current_analysis",
        "gap_counts": dict(sorted(gap_counts.items(), key=lambda pair: (_proof_gap_priority(pair[0]), pair[0]))),
        "follow_up_queries": deduped[:30],
        "top_priorities": deduped[:8],
    }


def _followup_query_from_clue(evidence: Dict[str, Any], chapter: Dict[str, Any]) -> str:
    fact = _compact(evidence.get("fact"), 120)
    metric = _compact(evidence.get("metric"), 60)
    title = _compact(_as_dict(evidence.get("source")).get("title"), 80)
    chapter_text = _compact(chapter.get("chapter_title") or chapter.get("chapter_question"), 80)
    seed = " ".join(part for part in [chapter_text, metric, fact, title] if part)
    return _compact(f"{seed} 官方 公告 财报 招股书 政府 协会 权威研报 验证", 220)


def bind_evidence_to_chapters(
    normalized_items: Sequence[Dict[str, Any]],
    chapters: Sequence[Dict[str, Any]],
    research_evidence_goals: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not chapters:
        chapters = [{"chapter_id": "core_question", "chapter_title": "核心研究问题", "chapter_question": "核心研究问题"}]
    packages: List[Dict[str, Any]] = []
    assigned: set[str] = set()
    goals = [goal for goal in research_evidence_goals if isinstance(goal, dict)]
    for chapter_index, chapter in enumerate(chapters, start=1):
        chapter = _as_dict(chapter)
        chapter_goals = _as_list(chapter.get("evidence_goals")) or goals
        scored: List[Tuple[int, Dict[str, Any]]] = []
        for item in normalized_items:
            score = _chapter_match_score(item, chapter, chapter_goals)
            if score > 0:
                scored.append((score, item))
        if not scored and chapter_index == 1:
            scored = [(1, item) for item in normalized_items if str(item.get("evidence_id")) not in assigned]
        scored.sort(key=lambda pair: (pair[0], -float(pair[1].get("confidence") or 0.0)), reverse=True)
        candidate_items = [item for _, item in scored]
        evidence_items: List[Dict[str, Any]] = []
        chapter_limit = int(os.getenv("REPORT_MAX_EVIDENCE_PER_CHAPTER", "100"))
        for _, item in scored:
            evidence_id = str(item.get("evidence_id") or "")
            if evidence_id in assigned and len(evidence_items) >= 3:
                continue
            evidence_items.append(item)
            assigned.add(evidence_id)
            if len(evidence_items) >= chapter_limit:
                break
        core, supporting, clue, table, appendix = _split_evidence_items(evidence_items)
        quality_summary = _package_quality_summary(
            candidates=candidate_items,
            selected=evidence_items,
            core=core,
            supporting=supporting,
            clue=clue,
            table=table,
            appendix=appendix,
        )
        min_sources = max(
            int(chapter.get("min_total_sources") or 0),
            int(os.getenv("REPORT_MIN_TOTAL_SOURCES_PER_CHAPTER", "12")),
        )
        for goal in chapter_goals:
            if isinstance(goal, dict) and goal.get("min_sources"):
                try:
                    goal_min_sources = int(goal.get("min_sources") or min_sources)
                except (TypeError, ValueError):
                    goal_min_sources = min_sources
                min_sources = max(min_sources, goal_min_sources)
        missing: List[Dict[str, Any]] = []
        for clue_item in clue[:10]:
            missing.append(
                {
                    "type": "needs_corroboration",
                    "seed": clue_item.get("fact"),
                    "evidence_ref": clue_item.get("ref") or clue_item.get("evidence_id"),
                    "source_level": clue_item.get("source_level"),
                    "follow_up_query": _followup_query_from_clue(clue_item, chapter),
                }
            )
        if len(core) < min_sources:
            missing.append(
                {
                    "type": "insufficient_core_evidence",
                    "required": min_sources,
                    "actual": len(core),
                    "suggestion": "补充更高等级、可核验、能直接回答章节问题的证据。",
                }
            )
        min_ab_core_sources = max(
            int(chapter.get("min_ab_sources") or 0),
            _env_int("REPORT_MIN_AB_SOURCES_PER_CHAPTER", 4 if _strict_quality_mode() else 2),
        )
        if int(quality_summary.get("core_ab_source_count") or 0) < min_ab_core_sources:
            missing.append(
                {
                    "type": "insufficient_ab_core_sources",
                    "required": min_ab_core_sources,
                    "actual": int(quality_summary.get("core_ab_source_count") or 0),
                    "suggestion": "核心判断缺少 A/B 级来源；优先补充公告、财报、政府/协会统计、交易所披露或权威研报原文。",
                    "follow_up_query": f"{chapter.get('chapter_title') or chapter.get('chapter_question') or ''} 官方 公告 财报 协会 统计 权威研报",
                }
            )
        min_counter_sources = max(
            int(chapter.get("min_counter_sources") or 0),
            int(os.getenv("REPORT_MIN_COUNTER_SOURCES_PER_DECISION_CHAPTER", "1")),
        )
        counter_count = len(
            [
                item
                for item in evidence_items
                if str(item.get("proof_role") or "").strip().lower() == "counter"
                or bool(item.get("counter_evidence"))
            ]
        )
        if min_counter_sources and counter_count < min_counter_sources:
            missing.append(
                {
                    "type": "insufficient_counter_sources",
                    "required": min_counter_sources,
                    "actual": counter_count,
                    "suggestion": "本章缺少反证或风险证据；优先补充需求不及预期、价格下跌、产能过剩、失败案例或监管变化。",
                    "follow_up_query": f"{chapter.get('chapter_title') or chapter.get('chapter_question') or ''} 风险 反证 产能过剩 价格下跌 需求不及预期",
                }
            )
        packages.append(
            {
                "chapter_id": str(chapter.get("chapter_id") or f"chapter_{chapter_index}"),
                "chapter_title": str(chapter.get("chapter_title") or chapter.get("title") or f"章节 {chapter_index}"),
                "chapter_question": str(chapter.get("chapter_question") or chapter.get("chapter_role") or "").strip(),
                "required_evidence_mix": _as_list(chapter.get("required_evidence_mix")),
                "min_total_sources": min_sources,
                "min_ab_sources": min_ab_core_sources,
                "min_counter_sources": min_counter_sources,
                "source_count": len([item for item in evidence_items if not item.get("excluded")]),
                "ab_source_count": int(quality_summary.get("core_ab_source_count") or 0),
                "counter_source_count": counter_count,
                "evidence_items": evidence_items,
                "core_evidence": core,
                "supporting_evidence": supporting,
                "clue_evidence": clue,
                "table_evidence": table,
                "appendix_evidence": appendix,
                "conflicts": _detect_conflicts(evidence_items),
                "missing_evidence": missing,
                "evidence_quality_summary": quality_summary,
                "follow_up_queries": [
                    item.get("follow_up_query")
                    for item in missing
                    if isinstance(item, dict) and str(item.get("follow_up_query") or "").strip()
                ],
            }
        )
    return packages


def build_materials_payload_from_packages(result: Dict[str, Any]) -> Dict[str, Any]:
    materials: Dict[str, List[Dict[str, Any]]] = {}
    for package in _as_list(result.get("chapter_evidence_packages")):
        if not isinstance(package, dict):
            continue
        key = str(package.get("chapter_title") or package.get("chapter_id") or "核心研究问题")
        materials[key] = [item for item in _as_list(package.get("evidence_items")) if isinstance(item, dict)]
    return {
        "materials": materials,
        "sources": _as_list(result.get("source_registry")),
        "footnotes": _as_list(result.get("footnotes")),
        "metadata": {
            "dimension_count": len(materials),
            "source_count": len(_as_list(result.get("source_registry"))),
        },
    }


def run_evidence_binder(
    *,
    research_plan: Optional[Dict[str, Any]] = None,
    report_blueprint: Optional[Dict[str, Any]] = None,
    evidence_package: Optional[Dict[str, Any]] = None,
    structured_analysis: Optional[Dict[str, Any]] = None,
    child_outputs: Optional[Dict[str, Dict[str, Any]]] = None,
    evidence_pool: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    research_plan = _as_dict(research_plan)
    report_blueprint = _as_dict(report_blueprint)
    raw_items = collect_raw_evidence(
        evidence_package=evidence_package,
        structured_analysis=structured_analysis,
        child_outputs=child_outputs,
        evidence_pool=evidence_pool,
    )
    registry = SourceRegistry()
    normalized = normalize_and_register_sources(raw_items, registry)
    for item in normalized:
        if not isinstance(item, dict):
            continue
        if not evidence_matches_report_topic(item, report_blueprint):
            item["evidence_role"] = "rejected"
            item["semantic_status"] = "irrelevant"
            item["semantic_reason"] = "off_topic_for_report"
            item["appendix_only"] = False
            item["excluded"] = True
    normalized_before_compression_count = len(normalized)
    normalized = compress_normalized_items_for_reporting(normalized)
    chapter_packages = bind_evidence_to_chapters(
        normalized,
        _as_list(report_blueprint.get("chapters")),
        _as_list(research_plan.get("evidence_goals")),
    )
    metric_table = normalize_metric_table(normalized)
    research_proof_profile = select_research_proof_profile(
        query=str(report_blueprint.get("research_object") or report_blueprint.get("topic") or ""),
        research_plan=research_plan,
        report_blueprint=report_blueprint,
    )
    coverage_matrix = build_coverage_matrix(
        research_plan=research_plan,
        normalized_items=normalized,
        metric_table=metric_table,
        report_blueprint=report_blueprint,
        proof_profile=research_proof_profile,
    )
    report_mandatory_proof_checks = mandatory_proof_checks(
        research_proof_profile,
        normalized,
        metric_table,
        context={"research_plan": research_plan, "report_blueprint": report_blueprint},
    )
    coverage_by_hypothesis = {
        str(item.get("hypothesis_id") or ""): item
        for item in coverage_matrix
        if isinstance(item, dict) and str(item.get("hypothesis_id") or "")
    }
    proof_follow_up_queries = [
        _proof_followup_query(item)
        for item in coverage_matrix
        if isinstance(item, dict) and _as_list(item.get("blocking_gaps"))
    ]
    proof_follow_up_queries.extend(
        build_mandatory_proof_followups(
            report_mandatory_proof_checks,
            profile=research_proof_profile,
        )
    )
    proof_followup_by_hypothesis = {
        str(item.get("hypothesis_id") or ""): item
        for item in proof_follow_up_queries
        if isinstance(item, dict) and str(item.get("hypothesis_id") or "")
    }
    for package in chapter_packages:
        if not isinstance(package, dict):
            continue
        refs = {
            str(item.get("ref") or item.get("evidence_id") or "")
            for item in _as_list(package.get("evidence_items"))
            if isinstance(item, dict)
        }
        package["coverage_matrix"] = [
            item
            for item in coverage_matrix
            if set(str(ref or "") for ref in _as_list(item.get("evidence_refs"))).intersection(refs)
        ]
        package["missing_proof_standards"] = [
            {
                "hypothesis_id": item.get("hypothesis_id"),
                "hypothesis_statement": item.get("hypothesis_statement"),
                "blocking_gaps": item.get("blocking_gaps"),
                "proof_profile_id": item.get("proof_profile_id"),
                "missing_mandatory_proofs": item.get("missing_mandatory_proofs"),
            }
            for item in package["coverage_matrix"]
            if _as_list(item.get("blocking_gaps"))
        ]
        package_proof_followups = [
            proof_followup_by_hypothesis[str(item.get("hypothesis_id") or "")]
            for item in package["missing_proof_standards"]
            if str(item.get("hypothesis_id") or "") in proof_followup_by_hypothesis
        ]
        if package_proof_followups:
            package["proof_follow_up_queries"] = package_proof_followups
            existing_queries = [
                item
                for item in _as_list(package.get("follow_up_queries"))
                if isinstance(item, dict) or str(item or "").strip()
            ]
            package["follow_up_queries"] = existing_queries + package_proof_followups
    evidence_refinement_plan = build_evidence_refinement_plan(
        coverage_matrix=coverage_matrix,
        chapter_packages=chapter_packages,
        proof_follow_up_queries=proof_follow_up_queries,
    )
    return {
        "agent": AGENT_NAME,
        "source_registry": registry.sources,
        "footnotes": registry.footnotes(),
        "normalized_evidence": normalized,
        "chapter_evidence_packages": chapter_packages,
        "source_quality_map": {
            str(item.get("source_ref") or item.get("evidence_id") or ""): {
                "source_level": item.get("source_level"),
                "source_type": item.get("source_type"),
                "source_score": item.get("source_score"),
                "source_reason": item.get("source_reason"),
                "can_support_core_claim": item.get("can_support_core_claim"),
            }
            for item in normalized
            if isinstance(item, dict)
        },
        "metric_normalization_table": metric_table,
        "research_proof_profile": research_proof_profile,
        "mandatory_proof_checks": report_mandatory_proof_checks,
        "coverage_matrix": coverage_matrix,
        "hypothesis_evidence_map": {
            hypothesis_id: [
                item.get("ref") or item.get("evidence_id")
                for item in normalized
                if str(item.get("hypothesis_id") or "") == hypothesis_id and (item.get("ref") or item.get("evidence_id"))
            ]
            for hypothesis_id in coverage_by_hypothesis
        },
        "missing_proof_standards": [
            {
                "hypothesis_id": item.get("hypothesis_id"),
                "hypothesis_statement": item.get("hypothesis_statement"),
                "blocking_gaps": item.get("blocking_gaps"),
                "proof_profile_id": item.get("proof_profile_id"),
                "missing_mandatory_proofs": item.get("missing_mandatory_proofs"),
                "follow_up_query": proof_followup_by_hypothesis.get(str(item.get("hypothesis_id") or "")),
            }
            for item in coverage_matrix
            if _as_list(item.get("blocking_gaps"))
        ],
        "proof_follow_up_queries": proof_follow_up_queries,
        "evidence_refinement_plan": evidence_refinement_plan,
        "metadata": {
            "raw_evidence_count": len(raw_items),
            "normalized_evidence_count": len(normalized),
            "normalized_evidence_before_compression_count": normalized_before_compression_count,
            "evidence_compression_enabled": (
                _env_flag("REPORT_ENABLE_EVIDENCE_CLUSTER_COMPRESSION", True)
                and not _strict_quality_mode()
                and normalized_before_compression_count > len(normalized)
            ),
            "candidate_count": len(raw_items),
            "kept_count": len([item for item in normalized if not item.get("excluded")]),
            "rejected_count": len([item for item in normalized if item.get("excluded")]),
            "core_candidate_source_level_distribution": _level_distribution([item for item in normalized if not item.get("appendix_only") and not item.get("excluded")]),
            "source_level_distribution": _level_distribution(normalized),
            "appendix_only_count": len([item for item in normalized if item.get("appendix_only") and not item.get("excluded")]),
            "excluded_count": len([item for item in normalized if item.get("excluded")]),
            "source_count": len(registry.sources),
            "chapter_package_count": len(chapter_packages),
            "metric_count": len(metric_table),
            "coverage_matrix_count": len(coverage_matrix),
            "missing_proof_standard_count": len([item for item in coverage_matrix if _as_list(item.get("blocking_gaps"))]),
            "mandatory_proof_missing_count": len(missing_mandatory_proofs(report_mandatory_proof_checks)),
            "evidence_refinement_task_count": len(_as_list(evidence_refinement_plan.get("follow_up_queries"))),
        },
    }
