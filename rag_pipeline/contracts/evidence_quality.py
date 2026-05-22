from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse


QUALITY_CONTRACT_VERSION = "0.1.0"

REJECTED_STATUSES = {"rejected", "spam", "irrelevant", "blacklisted", "exclude"}
LOW_QUALITY_ROLES = {"clue", "appendix", "appendix_only"}
CORE_ROLES = {"core", "core_claim"}
SUPPORTING_ROLES = {
    "support",
    "supporting",
    "metric",
    "case",
    "counter",
    "filing",
    "source_check",
    "technology_product",
}
GENERIC_METRIC_NAMES = {"", "unknown", "key_fact", "fact", "metric", "indicator", "关键事实", "定性事实"}

SOURCE_TYPE_TO_LEVEL = {
    "official": "A",
    "government": "A",
    "policy": "A",
    "filing": "A",
    "financial_report": "A",
    "annual_report": "A",
    "prospectus": "A",
    "exchange": "A",
    "research": "B",
    "academic": "B",
    "industry_report": "B",
    "association": "B",
    "consulting": "B",
    "market_research": "B",
    "brokerage": "B",
    "whitepaper": "B",
    "company_announcement": "A",
    "company_official": "B",
    "product_doc": "B",
    "technical_standard": "B",
    "patent": "B",
    "authoritative_secondary": "B",
    "media": "C",
    "news": "C",
    "self_media": "D",
    "ugc": "D",
    "unknown": "C",
    "": "C",
}

SOURCE_LEVEL_SCORE = {"A": 0.92, "B": 0.78, "C": 0.56, "D": 0.28}

SOURCE_TYPE_TO_TIER = {
    "official": "A1",
    "government": "A1",
    "policy": "A1",
    "filing": "A1",
    "financial_report": "A1",
    "annual_report": "A1",
    "prospectus": "A1",
    "exchange": "A1",
    "company_announcement": "A2",
    "company_official": "A2",
    "product_doc": "A2",
    "technical_standard": "B1",
    "patent": "B1",
    "academic": "B1",
    "research": "B1",
    "industry_report": "B1",
    "association": "B1",
    "consulting": "B2",
    "market_research": "B2",
    "brokerage": "B2",
    "whitepaper": "B2",
    "authoritative_secondary": "B2",
    "media": "C1",
    "news": "C1",
    "unknown": "C2",
    "self_media": "D",
    "ugc": "D",
    "": "C2",
}

HARD_METRIC_PROOF_ROLES = {"metric", "filing", "official_data", "source_check", "financial"}
PRODUCT_EVENT_SOURCE_TYPES = {"company_announcement", "company_official", "product_doc", "technical_standard", "patent"}
CLAIM_TYPES = {
    "hard_metric",
    "industry_analysis",
    "product_event",
    "case_signal",
    "forecast_judgment",
}
HARD_METRIC_TERMS = {
    "market size",
    "market share",
    "revenue",
    "profit",
    "valuation",
    "shipment",
    "penetration",
    "growth rate",
    "cagr",
    "\u5e02\u573a\u89c4\u6a21",
    "\u5e02\u573a\u4efd\u989d",
    "\u8425\u6536",
    "\u5229\u6da6",
    "\u4f30\u503c",
    "\u51fa\u8d27",
    "\u6e17\u900f\u7387",
    "\u589e\u901f",
    "\u589e\u957f\u7387",
    "\u76d1\u7ba1",
    "\u653f\u7b56",
    "\u8d22\u62a5",
}
PRODUCT_EVENT_TERMS = {
    "launch",
    "release",
    "feature",
    "product",
    "version",
    "\u4e0a\u7ebf",
    "\u53d1\u5e03",
    "\u529f\u80fd",
    "\u4ea7\u54c1",
    "\u7248\u672c",
    "\u5b98\u65b9",
}
CASE_SIGNAL_TERMS = {
    "customer",
    "case",
    "order",
    "contract",
    "tender",
    "\u5ba2\u6237",
    "\u6848\u4f8b",
    "\u8ba2\u5355",
    "\u5408\u540c",
    "\u4e2d\u6807",
    "\u5546\u4e1a\u5316",
}
FORECAST_TERMS = {
    "forecast",
    "estimate",
    "outlook",
    "expected",
    "will",
    "\u9884\u6d4b",
    "\u9884\u8ba1",
    "\u5c55\u671b",
    "\u5224\u65ad",
    "\u8d8b\u52bf",
    "\u63a8\u6f14",
}


def infer_claim_type(item: Dict[str, Any]) -> str:
    explicit = str(item.get("claim_type") or item.get("conclusion_type") or item.get("claim_kind") or "").strip().lower()
    if explicit in CLAIM_TYPES:
        return explicit
    source = _source_payload(item)
    source_type = str(source.get("source_type") or "").strip().lower()
    proof_role = _proof_role(item)
    text = " ".join(
        str(value or "")
        for value in [
            proof_role,
            item.get("evidence_type"),
            item.get("intent"),
            item.get("metric"),
            item.get("dimension"),
            item.get("dimension_name"),
            item.get("evidence_goal"),
            item.get("hypothesis_statement"),
            item.get("claim"),
            item.get("content"),
            item.get("clean_fact"),
            _as_dict(item.get("search_task")).get("proof_role"),
            _as_dict(item.get("search_task")).get("evidence_goal"),
            _as_dict(item.get("search_task")).get("dimension_name"),
        ]
    ).lower()
    has_number = bool(item.get("value") or item.get("numeric_values") or item.get("numeric_value") or re.search(r"\d", text))
    if proof_role in HARD_METRIC_PROOF_ROLES or any(term in text for term in HARD_METRIC_TERMS):
        return "hard_metric"
    if has_number and any(term in text for term in {"share", "size", "revenue", "profit", "\u4efd\u989d", "\u89c4\u6a21", "\u8425\u6536", "\u5229\u6da6"}):
        return "hard_metric"
    if source_type in PRODUCT_EVENT_SOURCE_TYPES or any(term in text for term in PRODUCT_EVENT_TERMS):
        return "product_event"
    if any(term in text for term in CASE_SIGNAL_TERMS):
        return "case_signal"
    if any(term in text for term in FORECAST_TERMS):
        return "forecast_judgment"
    return "industry_analysis"


def is_hard_metric_claim(item: Dict[str, Any]) -> bool:
    return infer_claim_type(item) == "hard_metric"


def _source_subtier(item: Dict[str, Any], level: str) -> str:
    source = _source_payload(item)
    source_type = str(source.get("source_type") or "").strip().lower()
    if level == "A":
        return "A"
    if source_type in {"consulting", "market_research", "brokerage", "whitepaper", "authoritative_secondary"}:
        return "B+"
    if source_type in PRODUCT_EVENT_SOURCE_TYPES:
        return "B+"
    if source_type in {"research", "academic", "industry_report", "association"}:
        return "B"
    if source_type in {"media", "news"}:
        return "C+"
    return level or "C"


def _source_tier(item: Dict[str, Any], level: str) -> str:
    source = _source_payload(item)
    source_type = str(source.get("source_type") or "").strip().lower()
    url = str(source.get("url") or "").strip().lower()
    title = str(source.get("title") or "").strip().lower()
    text = f"{source_type} {url} {title}"
    if level == "D":
        return "D"
    if any(fragment in text for fragment in ("baijiahao", "zhihu", "weibo", "xueqiu", "toutiao", "wenku", "doc88", "docin")):
        return "D"
    if source_type in SOURCE_TYPE_TO_TIER:
        return SOURCE_TYPE_TO_TIER[source_type]
    if level == "A":
        return "A1" if (url.endswith(".gov") or ".gov." in url or "sec.gov" in url or "cninfo" in url) else "A2"
    if level == "B":
        return "B1" if any(term in text for term in ("association", "academic", "research", "standard", "patent", "协会", "标准", "专利")) else "B2"
    if level == "C":
        return "C1" if any(term in text for term in ("media", "news", "reuters", "bloomberg", "caixin", "新闻", "媒体")) else "C2"
    return SOURCE_TYPE_TO_TIER.get(source_type, "C2")


def _evidence_grade_note(source_level: str, source_subtier: str, claim_type: str, allowed_use: str) -> str:
    if source_level == "A":
        return "primary_or_official_evidence"
    if source_level == "B":
        return "industry_grade_core_support" if claim_type != "hard_metric" else "ab_support_for_hard_metric"
    if source_level == "C" and claim_type == "hard_metric":
        return "c_level_hard_metric_corroboration_only"
    if source_level == "C" and allowed_use == "directional_signal":
        return f"{source_subtier.lower()}_directional_signal"
    if source_level == "D":
        return "appendix_only_low_credibility"
    return "appendix_or_context"


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _compact(value: Any, max_chars: int = 360) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "..."


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clip(value: Any, default: float = 0.0) -> float:
    return max(0.0, min(1.0, _safe_float(value, default)))


def _dedupe(values: List[Any], *, limit: int = 20) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
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


def _source_payload(item: Dict[str, Any]) -> Dict[str, Any]:
    source = _as_dict(item.get("source"))
    return {
        "title": str(source.get("title") or item.get("source_title") or item.get("title") or "").strip(),
        "url": str(source.get("url") or source.get("source_url") or item.get("url") or item.get("source_url") or "").strip(),
        "date": str(source.get("date") or item.get("date") or item.get("period") or "").strip(),
        "source_type": str(source.get("source_type") or item.get("source_type") or item.get("source_family") or "").strip().lower(),
        "credibility": str(source.get("credibility") or source.get("source_level") or item.get("source_level") or "").strip().upper(),
    }


def _has_traceable_source(item: Dict[str, Any]) -> bool:
    source = _source_payload(item)
    if bool(item.get("fake_or_placeholder_source") or source.get("fake_or_placeholder_source")):
        return False
    url = str(source.get("url") or item.get("source_url") or item.get("url") or "").strip()
    if url and not re.search(r"example\.(?:com|gov|org)", url, re.I):
        return True
    document_ref = str(
        source.get("document_id")
        or source.get("doc_id")
        or source.get("page_ref")
        or item.get("document_id")
        or item.get("doc_id")
        or item.get("page_ref")
        or ""
    ).strip()
    return bool(document_ref)


def _metric_unit_present(item: Dict[str, Any], fact: str) -> bool:
    if str(item.get("unit") or item.get("numeric_unit") or "").strip():
        return True
    completeness = _as_dict(item.get("metric_completeness"))
    return bool(str(completeness.get("unit") or "").strip())
    return bool(re.search(r"%|pct|bps|x|times|usd|rmb|yuan|dollar|billion|million|trillion|元|亿元|万元|美元|台|套|吨|gwh|mwh|kwh", text, re.I))


def _metric_proof_gaps(item: Dict[str, Any], claim_type: str, fact: str, period: str) -> List[str]:
    if claim_type != "hard_metric":
        return []
    gaps: List[str] = []
    metric = str(item.get("metric") or "").strip()
    value = str(item.get("value") or item.get("numeric_value") or "").strip()
    if not metric or metric.lower() in GENERIC_METRIC_NAMES:
        gaps.append("metric")
    if not value and not item.get("numeric_values") and not re.search(r"\d", fact):
        gaps.append("value")
    if not period:
        gaps.append("period")
    if not str(item.get("scope") or "").strip():
        gaps.append("scope")
    if not _metric_unit_present(item, fact):
        gaps.append("unit")
    if not _has_traceable_source(item):
        gaps.append("source")
    return _dedupe(gaps, limit=8)


def _evidence_fit_score(
    *,
    source_level: str,
    allowed_use: str,
    directness: str,
    metric_proof_gaps: List[str],
    traceable: bool,
) -> float:
    score = SOURCE_LEVEL_SCORE.get(source_level, 0.46)
    if allowed_use == "core_claim":
        score += 0.05
    elif allowed_use == "directional_signal":
        score -= 0.08
    elif allowed_use in {"clue", "appendix_only", "rejected"}:
        score -= 0.18
    if directness == "direct":
        score += 0.06
    elif directness == "clue":
        score -= 0.10
    if traceable:
        score += 0.05
    else:
        score -= 0.18
    score -= min(0.36, 0.06 * len(metric_proof_gaps))
    return round(max(0.0, min(1.0, score)), 4)


def _analysis_readiness(
    *,
    source_level: str,
    allowed_use: str,
    claim_type: str,
    metric_proof_gaps: List[str],
    semantic_status: str,
    traceable: bool,
) -> str:
    if semantic_status in REJECTED_STATUSES or allowed_use == "rejected":
        return "blocked"
    if source_level == "D":
        return "followup_only"
    if claim_type == "hard_metric" and metric_proof_gaps:
        return "context_only" if source_level in {"A", "B"} and traceable else "followup_only"
    if allowed_use in {"core_claim", "supporting"} and source_level in {"A", "B"} and traceable:
        return "decision_ready"
    if allowed_use == "directional_signal":
        return "directional_ready"
    if allowed_use in {"clue", "appendix_only"}:
        return "followup_only"
    return "context_only"


def _source_level(item: Dict[str, Any]) -> str:
    source = _source_payload(item)
    url = str(source.get("url") or item.get("url") or item.get("source_url") or "").lower()
    title = str(source.get("title") or item.get("title") or "").lower()
    text = f"{url} {title}"
    if any(fragment in text for fragment in ("caifuhao.eastmoney", "guba.eastmoney", "mguba.eastmoney", "baijiahao", "toutiao", "zhihu", "xueqiu", "weibo", "sohu", "book118", "docin", "doc88", "wenku.baidu")):
        return "D"
    if any(fragment in text for fragment in ("view.inews.qq.com", "finance.sina.com.cn", "news.10jqka.com.cn", "news.futunn.com")):
        return "C"
    explicit = str(
        item.get("source_level")
        or item.get("credibility_level")
        or source.get("credibility")
        or ""
    ).strip().upper()
    if explicit in {"A", "B", "C", "D"}:
        return explicit
    return SOURCE_TYPE_TO_LEVEL.get(str(source.get("source_type") or "").strip().lower(), "C")


def _source_family(item: Dict[str, Any]) -> str:
    source = _source_payload(item)
    source_type = str(source.get("source_type") or "").strip().lower()
    url = source.get("url", "").lower()
    title = source.get("title", "").lower()
    text = f"{source_type} {url} {title}"
    if source_type in {"official", "government", "policy", "filing", "financial_report", "annual_report", "prospectus", "exchange", "company_announcement"}:
        return "official/filing"
    if source_type in {"research", "academic", "industry_report", "association", "technical_standard", "patent"}:
        return "research/association"
    if any(term in text for term in ("customer", "case", "procurement", "order", "contract", "client", "tender")):
        return "company/case"
    if source_type in {"media", "news", "consulting"}:
        return "news/secondary"
    domain = urlparse(source.get("url", "")).netloc.lower()
    if domain.endswith(".gov.cn") or "gov." in domain:
        return "official/filing"
    return "unknown"


def _proof_role(item: Dict[str, Any]) -> str:
    role = str(
        item.get("proof_role")
        or _as_dict(item.get("search_task")).get("proof_role")
        or item.get("evidence_type")
        or item.get("evidence_role")
        or item.get("role")
        or ""
    ).strip().lower()
    if role in {"core", "core_claim"}:
        return "support"
    if role in {"supporting"}:
        return "support"
    if item.get("counter_evidence"):
        return "counter"
    if role in {"technology_product"}:
        return role
    metric = str(item.get("metric") or "").strip()
    has_specific_metric = bool(metric and metric.lower() not in GENERIC_METRIC_NAMES)
    if has_specific_metric or item.get("value") or item.get("numeric_values"):
        return "metric"
    return role or "support"


def _evidence_role(item: Dict[str, Any], proof_role: str) -> str:
    role = str(item.get("evidence_role") or item.get("role") or "").strip().lower()
    if role == "exclude":
        return "rejected"
    if role:
        return role
    if proof_role in {"metric", "source_check", "support", "case", "counter", "filing", "technology_product"}:
        return "supporting"
    return "appendix"


def _directness(item: Dict[str, Any], proof_role: str, source_level: str, evidence_role: str) -> str:
    has_metric = bool(item.get("metric") or item.get("value") or item.get("numeric_values") or item.get("numeric_value"))
    direct = proof_role in {"metric", "case", "source_check", "counter", "filing", "technology_product"} or has_metric
    if source_level in {"A", "B"} and direct and evidence_role not in LOW_QUALITY_ROLES:
        return "direct"
    if direct:
        return "indirect"
    return "clue" if source_level in {"C", "D"} or evidence_role in LOW_QUALITY_ROLES else "indirect"


def _clean_value_text(value: Any) -> str:
    text = re.sub(r"\s+", "", str(value or "").strip())
    return text.strip(" ,;:!?，。；：！？")


@dataclass(frozen=True)
class EvidenceNormalizer:
    max_content_chars: int = 520
    max_fact_chars: int = 260

    def clean_content(self, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        text = re.sub(r"\[[Ii][Dd]:[^\]]+\]", "", text)
        text = re.sub(r"\[\d{1,5}\]", "", text)
        text = re.sub(r"\s+", " ", text)
        return _compact(text.strip(" ,;:!?，。；：！？"), self.max_content_chars)

    def clean_metric_name(self, value: Any, content: Any = "") -> str:
        metric = re.sub(r"\s+", "", str(value or "").strip())
        if metric.lower() in GENERIC_METRIC_NAMES:
            metric = ""
        if not metric:
            content_text = str(content or "")
            if re.search(r"CAGR|复合增速|年均增速", content_text, re.I):
                return "CAGR"
            if re.search(r"同比|增速|增长", content_text):
                return "增速"
            if re.search(r"市场规模|规模", content_text):
                return "市场规模"
            return "关键事实"
        return metric

    def clean_fact_period(self, item: Dict[str, Any]) -> str:
        source = _source_payload(item)
        if source.get("date"):
            return source["date"]
        text = str(item.get("content") or item.get("clean_content") or item.get("fact") or "")
        match = re.search(r"(20\d{2}(?:[-—~至]\d{2,4})?年?|20\d{2}Q[1-4]|近\d+年|未来\d+年)", text)
        return match.group(1) if match else ""

    def clean_fact_description(self, item: Dict[str, Any]) -> str:
        content = self.clean_content(item.get("clean_fact") or item.get("clean_content") or item.get("fact") or item.get("content"))
        if content:
            return _compact(content, self.max_fact_chars)
        metric = self.clean_metric_name(item.get("metric"), item.get("content"))
        value = _clean_value_text(item.get("value") or item.get("numeric_value"))
        if metric and value and metric != "关键事实":
            return _compact(f"{metric}={value}", self.max_fact_chars)
        return _compact(value, self.max_fact_chars)

    def normalize(self, item: Dict[str, Any]) -> Dict[str, Any]:
        copied = dict(item or {})
        copied["clean_content"] = self.clean_content(copied.get("clean_content") or copied.get("content") or copied.get("fact"))
        copied["metric"] = self.clean_metric_name(copied.get("metric"), copied.get("clean_content") or copied.get("content"))
        copied["period"] = str(copied.get("period") or self.clean_fact_period(copied)).strip()
        copied["clean_fact"] = self.clean_fact_description(copied)
        return copied


@dataclass(frozen=True)
class EvidenceClassifier:
    strict_quality: bool = False
    directional_c_min_confidence: float = 0.55

    def classify(self, item: Dict[str, Any]) -> Dict[str, Any]:
        copied = dict(item or {})
        source_level = _source_level(copied)
        proof_role = _proof_role(copied)
        evidence_role = _evidence_role(copied, proof_role)
        semantic_status = str(copied.get("semantic_status") or "ok").strip().lower()
        confidence = _clip(copied.get("confidence"), 0.0)
        claim_type = infer_claim_type(copied)
        source_subtier = _source_subtier(copied, source_level)
        source_tier = _source_tier(copied, source_level)

        if semantic_status in REJECTED_STATUSES or evidence_role in REJECTED_STATUSES:
            evidence_role = "rejected"
            allowed_use = "rejected"
            appendix_only = False
            enterprise_usable = False
            followup_seed = False
            usage_tier = "rejected"
            directness = "clue"
        elif source_level == "D":
            evidence_role = "clue"
            allowed_use = "appendix_only"
            appendix_only = True
            enterprise_usable = False
            followup_seed = True
            usage_tier = "clue_low_quality"
            directness = "clue"
        elif source_level == "C":
            c_floor = self.directional_c_min_confidence
            if claim_type in {"product_event", "case_signal", "industry_analysis"}:
                c_floor = min(c_floor, 0.50)
            directional = (
                not self.strict_quality
                and confidence >= c_floor
                and semantic_status not in {"weak", "weak_relevance", "appendix"}
            )
            allowed_use = "directional_signal" if directional else "clue"
            appendix_only = not directional
            enterprise_usable = directional
            followup_seed = not directional
            usage_tier = "directional_signal" if directional else "appendix_or_corroboration"
            if evidence_role not in LOW_QUALITY_ROLES and directional:
                evidence_role = "supporting"
            else:
                evidence_role = "clue"
            directness = _directness(copied, proof_role, source_level, evidence_role)
        elif source_level in {"A", "B"} and evidence_role in CORE_ROLES:
            allowed_use = "core_claim"
            appendix_only = False
            enterprise_usable = True
            followup_seed = False
            usage_tier = "core"
            directness = _directness(copied, proof_role, source_level, evidence_role)
        elif source_level in {"A", "B"} and (evidence_role in SUPPORTING_ROLES or proof_role in SUPPORTING_ROLES):
            evidence_role = "supporting" if evidence_role not in CORE_ROLES else evidence_role
            allowed_use = "supporting"
            appendix_only = False
            enterprise_usable = True
            followup_seed = False
            usage_tier = "supporting"
            directness = _directness(copied, proof_role, source_level, evidence_role)
        else:
            evidence_role = "appendix"
            allowed_use = "appendix_only"
            appendix_only = True
            enterprise_usable = False
            followup_seed = True
            usage_tier = "appendix_only"
            directness = "clue"

        base = SOURCE_LEVEL_SCORE.get(source_level, 0.46)
        confidence_score = round(max(confidence, base * 0.75) if confidence else base, 4)
        inference_distance = "low" if allowed_use == "core_claim" else ("medium" if allowed_use == "supporting" else "high")
        grade_note = _evidence_grade_note(source_level, source_subtier, claim_type, allowed_use)
        fact = str(copied.get("clean_fact") or copied.get("fact") or copied.get("clean_content") or copied.get("content") or "").strip()
        period = str(copied.get("period") or _source_payload(copied).get("date") or "").strip()
        metric_proof_gaps = _metric_proof_gaps(copied, claim_type, fact, period)
        traceable = _has_traceable_source(copied)
        evidence_fit_score = _evidence_fit_score(
            source_level=source_level,
            allowed_use=allowed_use,
            directness=directness,
            metric_proof_gaps=metric_proof_gaps,
            traceable=traceable,
        )
        analysis_readiness = _analysis_readiness(
            source_level=source_level,
            allowed_use=allowed_use,
            claim_type=claim_type,
            metric_proof_gaps=metric_proof_gaps,
            semantic_status=semantic_status,
            traceable=traceable,
        )
        card = {
            **_as_dict(copied.get("evidence_card")),
            "fact": fact,
            "source_level": source_level,
            "source_subtier": source_subtier,
            "source_tier": source_tier,
            "source_family": _source_family(copied),
            "claim_type": claim_type,
            "evidence_grade_note": grade_note,
            "proof_role": proof_role,
            "directness": directness,
            "scope": str(copied.get("scope") or copied.get("dimension_name") or copied.get("dimension") or "").strip(),
            "period": period,
            "metric_definition": {
                "metric": copied.get("metric"),
                "value": copied.get("value"),
                "period": period,
                "unit": copied.get("numeric_unit"),
            },
            "can_prove": _dedupe([
                copied.get("evidence_goal"),
                copied.get("hypothesis_statement"),
                copied.get("dimension_name"),
                copied.get("dimension"),
                proof_role,
            ], limit=5),
            "cannot_prove": _dedupe([
                "industry-wide conclusion without cross-source bundle",
                "investment priority without counter-evidence",
                "market certainty from a single source",
            ], limit=5),
            "inference_distance": inference_distance,
            "contradictions": _as_list(_as_dict(copied.get("evidence_card")).get("contradictions")),
            "allowed_use": allowed_use,
            "confidence_score": confidence_score,
            "evidence_fit_score": evidence_fit_score,
            "metric_proof_gaps": metric_proof_gaps,
            "analysis_readiness": analysis_readiness,
            "quality_contract_version": QUALITY_CONTRACT_VERSION,
        }
        can_support_if_corrobated = allowed_use == "directional_signal" and claim_type != "hard_metric"
        return {
            "source_level": source_level,
            "source_subtier": source_subtier,
            "source_tier": source_tier,
            "claim_type": claim_type,
            "proof_role": proof_role,
            "evidence_role": evidence_role,
            "semantic_status": semantic_status or "ok",
            "directness": directness,
            "allowed_use": allowed_use,
            "usage_tier": usage_tier,
            "appendix_only": appendix_only,
            "enterprise_usable": enterprise_usable,
            "followup_seed": followup_seed,
            "can_support_claim_if_corrobated": can_support_if_corrobated,
            "can_support_industry_analysis": bool(source_level in {"A", "B"} or can_support_if_corrobated),
            "evidence_grade_note": grade_note,
            "confidence_score": confidence_score,
            "evidence_fit_score": evidence_fit_score,
            "metric_proof_gaps": metric_proof_gaps,
            "analysis_readiness": analysis_readiness,
            "evidence_card": card,
        }


def normalize_evidence(item: Dict[str, Any], *, normalizer: Optional[EvidenceNormalizer] = None) -> Dict[str, Any]:
    return (normalizer or EvidenceNormalizer()).normalize(item)


def classify_evidence(
    item: Dict[str, Any],
    *,
    strict_quality: bool = False,
    directional_c_min_confidence: float = 0.55,
) -> Dict[str, Any]:
    return EvidenceClassifier(
        strict_quality=strict_quality,
        directional_c_min_confidence=directional_c_min_confidence,
    ).classify(item)


def apply_evidence_quality_contract(
    item: Dict[str, Any],
    *,
    strict_quality: bool = False,
    directional_c_min_confidence: float = 0.55,
) -> Dict[str, Any]:
    normalized = normalize_evidence(item)
    classification = classify_evidence(
        normalized,
        strict_quality=strict_quality,
        directional_c_min_confidence=directional_c_min_confidence,
    )
    merged = {**normalized, **classification}
    merged["quality_contract_version"] = QUALITY_CONTRACT_VERSION
    return merged
