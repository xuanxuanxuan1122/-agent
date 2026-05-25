from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

from rag_pipeline.telemetry.token_usage import record_llm_usage


PROVIDER_NAME = "openai_web"
CHILD_AGENT_NAME = "openai_web_search_agent"
DEFAULT_RESPONSES_URL = "https://api.openai.com/v1/responses"
DEFAULT_MODEL = "gpt-5.5"

AUTHORITATIVE_DOMAINS = [
    "stats.gov.cn",
    "ndrc.gov.cn",
    "miit.gov.cn",
    "mof.gov.cn",
    "pbc.gov.cn",
    "csrc.gov.cn",
    "samr.gov.cn",
    "sse.com.cn",
    "szse.cn",
    "hkexnews.hk",
    "cninfo.com.cn",
    "sec.gov",
]
KNOWN_TARGET_DOMAIN_PATTERNS: Tuple[Tuple[str, str], ...] = (
    (r"\bnist\b|national institute of standards|ai risk management framework|ai rmf", "nist.gov"),
    (r"\bsec\b|10-k|10-q|annual report|edgar", "sec.gov"),
    (r"\bstats\.gov\.cn\b|national bureau of statistics|statistics bureau", "stats.gov.cn"),
    (r"\bmiit\b|ministry of industry and information technology", "miit.gov.cn"),
    (r"\bndrc\b|national development and reform commission", "ndrc.gov.cn"),
    (r"\bcsrc\b|china securities regulatory commission", "csrc.gov.cn"),
    (r"\bsse\b|shanghai stock exchange", "sse.com.cn"),
    (r"\bszse\b|shenzhen stock exchange", "szse.cn"),
    (r"\bcninfo\b", "cninfo.com.cn"),
    (r"\bhkex\b|hong kong exchange", "hkexnews.hk"),
)
RESEARCH_DOMAINS = [
    "idc.com",
    "counterpointresearch.com",
    "omdia.tech.informa.com",
    "canalys.com",
    "gartner.com",
    "mckinsey.com",
    "bcg.com",
    "deloitte.com",
    "pwc.com",
]
HEAD_MEDIA_DOMAINS = [
    "reuters.com",
    "bloomberg.com",
    "caixin.com",
    "yicai.com",
    "stcn.com",
    "cs.com.cn",
    "21jingji.com",
    "thepaper.cn",
]


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, min_value: int = 0, max_value: int = 10_000) -> int:
    try:
        value = int(str(os.getenv(name, str(default))).strip())
    except (TypeError, ValueError):
        value = default
    return min(max_value, max(min_value, value))


def _env_csv(name: str, default: str = "") -> List[str]:
    raw = os.getenv(name, default)
    return [item.strip().lower() for item in re.split(r"[,;\s]+", str(raw or "")) if item.strip()]


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _compact_text(value: Any, max_chars: int = 900) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "..."


def _clip(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(0.0, min(1.0, number))


def _domain(url: Any) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""
    try:
        return urlparse(raw).netloc.lower().removeprefix("www.")
    except Exception:
        return ""


def _dedupe_strings(values: Sequence[str], *, limit: int = 24) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        text = str(value or "").strip().lower().removeprefix("www.")
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def _domains_from_text(text: str, *, limit: int = 12) -> List[str]:
    """Infer only explicit or highly specific target domains."""

    haystack = str(text or "").lower()
    domains: List[str] = []
    for pattern, domain in KNOWN_TARGET_DOMAIN_PATTERNS:
        if re.search(pattern, haystack, flags=re.I):
            domains.append(domain)
    for match in re.finditer(r"\b(?:https?://)?(?:www\.)?([a-z0-9-]+(?:\.[a-z0-9-]+)+)\b", haystack, flags=re.I):
        domain = match.group(1).lower().removeprefix("www.")
        if domain in {"example.com", "example.gov"}:
            continue
        domains.append(domain)
    return _dedupe_strings(domains, limit=limit)


def _explicit_domains_from_text(text: str, *, limit: int = 12) -> List[str]:
    haystack = str(text or "").lower()
    domains: List[str] = []
    for match in re.finditer(r"\b(?:https?://)?(?:www\.)?([a-z0-9-]+(?:\.[a-z0-9-]+)+)\b", haystack, flags=re.I):
        domain = match.group(1).lower().removeprefix("www.")
        if domain in {"example.com", "example.gov"}:
            continue
        domains.append(domain)
    return _dedupe_strings(domains, limit=limit)


def openai_web_search_enabled() -> bool:
    if not _env_flag("OPENAI_WEB_SEARCH_ENABLED", True):
        return False
    mode = str(os.getenv("OPENAI_WEB_SEARCH_MODE", "gap_repair") or "gap_repair").strip().lower()
    if mode not in {"gap_repair", "repair", "cross_check", "always"}:
        return False
    return bool(str(os.getenv("OPENAI_API_KEY") or "").strip())


def openai_web_search_config() -> Dict[str, Any]:
    return {
        "enabled": openai_web_search_enabled(),
        "api_key": str(os.getenv("OPENAI_API_KEY") or "").strip(),
        "model": str(os.getenv("OPENAI_WEB_SEARCH_MODEL") or DEFAULT_MODEL).strip() or DEFAULT_MODEL,
        "url": str(os.getenv("OPENAI_RESPONSES_URL") or DEFAULT_RESPONSES_URL).strip() or DEFAULT_RESPONSES_URL,
        "timeout": _env_int("OPENAI_WEB_SEARCH_TIMEOUT_SECONDS", 180, min_value=5, max_value=300),
        "external_web_access": _env_flag("OPENAI_WEB_SEARCH_EXTERNAL_WEB_ACCESS", True),
        "tool_choice": "required",
        "reasoning_effort": str(os.getenv("OPENAI_WEB_SEARCH_REASONING_EFFORT") or "").strip().lower(),
    }


def allowed_domains_for_task(search_task: Optional[Dict[str, Any]] = None) -> List[str]:
    explicit = _env_csv("OPENAI_WEB_SEARCH_ALLOWED_DOMAINS")
    if explicit:
        return _dedupe_strings(explicit, limit=50)

    task = _as_dict(search_task)
    task_domains = _dedupe_strings([str(item) for item in _as_list(task.get("allowed_domains"))], limit=50)
    if task_domains:
        return task_domains

    source_priority = " ".join(str(item).lower() for item in _as_list(task.get("source_priority")))
    proof_role = str(task.get("proof_role") or task.get("evidence_type") or "").lower()
    text = " ".join(
        str(value or "")
        for value in [
            source_priority,
            proof_role,
            task.get("query"),
            task.get("evidence_goal"),
            task.get("targets_gap"),
            " ".join(str(item) for item in _as_list(task.get("must_have_terms"))),
        ]
    )
    if not _env_flag("OPENAI_WEB_SEARCH_INFER_ALLOWED_DOMAINS", False):
        return _explicit_domains_from_text(text, limit=24)
    return _domains_from_text(text, limit=24)


LOW_CREDIBILITY_DOMAINS = (
    "caifuhao.eastmoney",
    "guba.eastmoney",
    "mguba.eastmoney",
    "baijiahao",
    "toutiao",
    "zhihu",
    "baike.baidu",
    "xueqiu",
    "weibo",
    "sohu",
    "book118",
    "docin",
    "doc88",
    "renrendoc",
    "wenku.baidu",
)

NEWS_AGGREGATOR_DOMAINS = (
    "view.inews.qq.com",
    "kuaixun",
    "finance.sina.com.cn",
    "news.10jqka.com.cn",
    "news.futunn.com",
    "finance.sina.cn",
    "k.sina.cn",
    "m.163.com",
    "view.inews.qq.com",
    "eastmoney.com",
    "allsearch",
    "search?",
)


def _source_type_from_source(source: Dict[str, Any]) -> str:
    domain = _domain(source.get("url"))
    title = str(source.get("title") or "").lower()
    text = f"{domain} {title}"
    if any(fragment in text for fragment in LOW_CREDIBILITY_DOMAINS):
        return "self_media"
    if any(fragment in text for fragment in NEWS_AGGREGATOR_DOMAINS):
        return "media"
    explicit = str(source.get("source_type") or source.get("type") or "").strip().lower()
    if explicit:
        return explicit
    if any(fragment in text for fragment in ("sec.gov", "cninfo", "sse.com", "szse", "hkexnews", "annual report", "filing")):
        return "financial_report"
    if domain.endswith(".gov.cn") or domain.endswith(".gov") or "gov." in domain:
        return "official"
    if any(fragment in text for fragment in ("idc", "counterpoint", "omdia", "canalys", "gartner", "research", "whitepaper", "brokerage", "consulting")):
        return "research"
    if any(fragment in domain for fragment in HEAD_MEDIA_DOMAINS):
        return "media"
    if any(fragment in domain for fragment in ("zhihu", "weibo", "baijiahao", "xueqiu", "reddit", "quora")):
        return "self_media"
    return "unknown"


def _source_level(source_type: str) -> str:
    if source_type in {"official", "government", "policy", "financial_report", "annual_report", "prospectus", "exchange"}:
        return "A"
    if source_type in {"research", "academic", "industry_report", "association", "consulting", "market_research", "brokerage", "whitepaper", "authoritative_secondary"}:
        return "B"
    if source_type in {"self_media", "ugc"}:
        return "D"
    return "C"


def _extract_number(value: Any) -> str:
    text = str(value or "")
    match = re.search(r"-?\d+(?:,\d{3})*(?:\.\d+)?\s*(?:%|pct|bps|x|times|usd|rmb|yuan|dollars|billion|million|trillion)?", text, flags=re.I)
    return re.sub(r"\s+", "", match.group(0)) if match else ""


def _infer_metric(text: Any, fallback: Any = "") -> str:
    raw_fallback = str(fallback or "").strip()
    if raw_fallback:
        return raw_fallback
    content = str(text or "")
    for metric, pattern in [
        ("market_size", r"market size|tam|规模|市场"),
        ("growth", r"growth|yoy|cagr|同比|增速|增长"),
        ("market_share", r"market share|share|份额|市占率"),
        ("revenue", r"revenue|sales|营收|收入"),
        ("profit", r"profit|margin|利润|净利"),
        ("funding", r"funding|financing|融资"),
        ("valuation", r"valuation|估值"),
        ("policy_target", r"policy|target|standard|framework|政策|标准"),
    ]:
        if re.search(pattern, content, flags=re.I):
            return metric
    return "metric_fact" if _extract_number(content) else "qualitative_fact"


def _safe_json_from_text(text: str) -> Dict[str, Any]:
    cleaned = str(text or "").strip()
    if not cleaned:
        return {}
    try:
        payload = json.loads(cleaned)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            payload = json.loads(cleaned[start : end + 1])
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _collect_output_text(response: Dict[str, Any]) -> str:
    if str(response.get("output_text") or "").strip():
        return str(response.get("output_text") or "").strip()
    parts: List[str] = []
    for output in _as_list(response.get("output")):
        if not isinstance(output, dict):
            continue
        for content in _as_list(output.get("content")):
            if isinstance(content, dict):
                text = content.get("text") or content.get("output_text")
                if text:
                    parts.append(str(text))
        if output.get("type") == "message" and isinstance(output.get("text"), str):
            parts.append(str(output.get("text")))
    return "\n".join(part.strip() for part in parts if part and str(part).strip()).strip()


def _walk_dicts(value: Any) -> List[Dict[str, Any]]:
    found: List[Dict[str, Any]] = []
    if isinstance(value, dict):
        found.append(value)
        for child in value.values():
            found.extend(_walk_dicts(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_walk_dicts(child))
    return found


def _collect_sources(response: Dict[str, Any], payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_sources: List[Dict[str, Any]] = []
    for item in _walk_dicts(response):
        if str(item.get("type") or "") == "url_citation" and item.get("url"):
            raw_sources.append(item)
        if item.get("url") and (item.get("title") or item.get("source") or item.get("snippet")):
            raw_sources.append(item)
    for item in _as_list(payload.get("sources")):
        if isinstance(item, dict) and (item.get("url") or item.get("title")):
            raw_sources.append(item)
    for evidence in _as_list(payload.get("evidence")):
        if not isinstance(evidence, dict):
            continue
        if evidence.get("source_url") or evidence.get("url"):
            raw_sources.append(
                {
                    "url": evidence.get("source_url") or evidence.get("url"),
                    "title": evidence.get("source_title") or evidence.get("title") or evidence.get("source"),
                    "date": evidence.get("date") or evidence.get("period"),
                }
            )

    sources: List[Dict[str, Any]] = []
    seen = set()
    for raw in raw_sources:
        source = {
            "title": str(raw.get("title") or raw.get("source") or raw.get("name") or raw.get("url") or "Untitled").strip(),
            "url": str(raw.get("url") or raw.get("source_url") or "").strip(),
            "date": str(raw.get("date") or raw.get("published_at") or raw.get("publishedTime") or "").strip(),
            "snippet": _compact_text(raw.get("snippet") or raw.get("summary") or raw.get("text") or "", max_chars=900),
            "provider": PROVIDER_NAME,
        }
        key = (source["url"].lower(), source["title"].lower())
        if key in seen or not source["url"]:
            continue
        seen.add(key)
        source_type = _source_type_from_source(source)
        source["source_type"] = source_type
        source["source_level"] = _source_level(source_type)
        source["credibility"] = source["source_level"]
        source["id"] = len(sources)
        source["source_id"] = len(sources)
        sources.append(source)
    return sources


def _source_for_evidence(evidence: Dict[str, Any], sources: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    source_url = str(evidence.get("source_url") or evidence.get("url") or "").strip()
    source_title = str(evidence.get("source_title") or evidence.get("source") or evidence.get("title") or "").strip()
    if source_url:
        for source in sources:
            if str(source.get("url") or "").strip().lower() == source_url.lower():
                return dict(source)
    if source_title:
        lowered = source_title.lower()
        for source in sources:
            if lowered and lowered in str(source.get("title") or "").lower():
                return dict(source)
    source_ids = _as_list(evidence.get("source_ids"))
    if evidence.get("source_id") is not None:
        source_ids.append(evidence.get("source_id"))
    for source_id in source_ids:
        for source in sources:
            if str(source.get("id")) == str(source_id) or str(source.get("source_id")) == str(source_id):
                return dict(source)
    return dict(sources[0]) if sources else {}


def _evidence_items_from_payload(payload: Dict[str, Any], text: str) -> List[Dict[str, Any]]:
    del text
    items = [item for item in _as_list(payload.get("evidence")) if isinstance(item, dict)]
    if items:
        return items
    return []
    generated: List[Dict[str, Any]] = []
    for line in re.split(r"[\n\r]+", str(text or "")):
        cleaned = re.sub(r"^\s*[-*鈥d.銆?锛塢+\s*", "", line).strip()
        if len(cleaned) < 18:
            continue
        if not re.search(r"\d", cleaned):
            continue
        generated.append({"claim": cleaned})
        if len(generated) >= 8:
            break
    return generated


def normalize_openai_web_response(
    response: Dict[str, Any],
    *,
    query: str,
    search_task: Optional[Dict[str, Any]] = None,
    targets_gap: str = "",
    round_number: int = 1,
) -> Dict[str, Any]:
    output_text = _collect_output_text(response)
    payload = _safe_json_from_text(output_text)
    sources = _collect_sources(response, payload)
    search_task = _as_dict(search_task)
    repair_source = str(search_task.get("repair_source") or "openai_web_gap_repair").strip()
    retrieval_mode = str(search_task.get("retrieval_mode") or "openai_repair").strip()
    for source in sources:
        source.setdefault("repair_source", repair_source)
        source.setdefault("retrieval_mode", retrieval_mode)
    raw_points: List[Dict[str, Any]] = []
    confidence_values: List[float] = []
    for evidence in _evidence_items_from_payload(payload, output_text):
        claim = _compact_text(
            evidence.get("claim")
            or evidence.get("fact")
            or evidence.get("content")
            or evidence.get("evidence")
            or evidence.get("summary"),
            max_chars=900,
        )
        if not claim:
            continue
        source = _source_for_evidence(evidence, sources)
        if not str(source.get("url") or "").strip():
            continue
        value = str(evidence.get("value") or evidence.get("numeric_value") or "").strip() or _extract_number(claim)
        metric = _infer_metric(claim, evidence.get("metric") or evidence.get("indicator"))
        source_type = str(source.get("source_type") or _source_type_from_source(source)).strip()
        source_level = str(source.get("source_level") or _source_level(source_type)).strip()
        confidence = _clip(evidence.get("confidence"), 0.68 if source_level in {"A", "B"} else 0.52)
        confidence_values.append(confidence)
        raw_points.append(
            {
                "dimension": evidence.get("dimension") or search_task.get("dimension_name") or targets_gap,
                "metric": metric,
                "value": value,
                "period": str(evidence.get("period") or evidence.get("date") or source.get("date") or "").strip(),
                "evidence": claim,
                "source": str(source.get("title") or source.get("url") or "OpenAI web source").strip(),
                "source_title": str(source.get("title") or "").strip(),
                "source_url": str(source.get("url") or "").strip(),
                "date": str(source.get("date") or "").strip(),
                "source_type": source_type,
                "source_level": source_level,
                "provider": PROVIDER_NAME,
                "repair_source": repair_source,
                "retrieval_mode": retrieval_mode,
                "confidence": confidence,
                "search_task": dict(search_task),
                "citation_ids": _as_list(evidence.get("source_ids")),
            }
        )

    summary = str(payload.get("summary") or payload.get("answer") or output_text or "").strip()
    source_candidates: List[Dict[str, Any]] = []
    if sources and not raw_points:
        for source in sources:
            if not str(source.get("url") or "").strip():
                continue
            source_type = str(source.get("source_type") or _source_type_from_source(source)).strip()
            source_level = str(source.get("source_level") or _source_level(source_type)).strip()
            source_candidates.append(
                {
                    "title": str(source.get("title") or source.get("url") or "").strip(),
                    "url": str(source.get("url") or "").strip(),
                    "date": str(source.get("date") or "").strip(),
                    "source_type": source_type,
                    "source_level": source_level,
                    "provider": PROVIDER_NAME,
                    "repair_source": repair_source,
                    "retrieval_mode": retrieval_mode,
                    "candidate_only": True,
                    "candidate_reason": "url_citation_without_structured_evidence",
                    "search_task": dict(search_task),
                }
            )
    avg_confidence = round(sum(confidence_values) / max(len(confidence_values), 1), 4) if confidence_values else 0.0
    status = "success" if sources and raw_points else "failed"
    evidence_gaps = _as_list(payload.get("evidence_gap")) or _as_list(payload.get("limitations"))
    domain_task = {**_as_dict(search_task)}
    domain_task.setdefault("query", query)
    allowed_domains = allowed_domains_for_task(domain_task)
    lowered_summary = f"{summary} {' '.join(str(item) for item in evidence_gaps)}".lower()
    failure_reason = ""
    if status == "failed":
        avg_confidence = 0.0
        if allowed_domains and re.search(r"allowed[- ]domain|domain constraint|no qualifying|restricted to the allowed domains", lowered_summary):
            failure_reason = "openai_web_domain_filter_too_strict"
        elif sources and not raw_points:
            failure_reason = "openai_web_raw_evidence_missing"
        elif not sources:
            failure_reason = "openai_web_source_url_missing"
        else:
            failure_reason = "openai_web_no_usable_evidence"
    return {
        "answer": summary,
        "confidence": avg_confidence,
        "key_sources": list(sources),
        "raw_data_points": raw_points,
        "source_candidates": source_candidates,
        "limitations": {
            "provider": PROVIDER_NAME,
            "coverage": "OpenAI web search gap-repair evidence",
            "model": response.get("model") or "",
            "evidence_gap": evidence_gaps,
            "failure_reason": failure_reason,
            "allowed_domains": allowed_domains,
            "source_candidate_count": len(source_candidates),
        },
        "status": status,
        "search_task": dict(search_task),
        "used": status in {"success", "partial"} and avg_confidence >= 0.2,
        "note": f"OpenAI web search returned {len(sources)} sources and {len(raw_points)} evidence points.",
        "provider": PROVIDER_NAME,
        "metadata": {
            "round": round_number,
            "query": query,
            "source_count": len(sources),
            "raw_data_point_count": len(raw_points),
            "source_candidate_count": len(source_candidates),
            "retrieval_mode": retrieval_mode,
            "repair_source": repair_source,
        },
    }


def build_openai_web_prompt(query: str, search_task: Optional[Dict[str, Any]] = None, targets_gap: str = "") -> str:
    task = _as_dict(search_task)
    compact_retry = _env_flag("OPENAI_WEB_SEARCH_COMPACT_PROMPT", True) or bool(task.get("openai_retry_compact_prompt"))
    if compact_retry:
        keep_keys = {
            "chapter_id",
            "chapter_title",
            "proof_role",
            "root_cause",
            "evidence_goal",
            "targets_gap",
            "query",
            "allowed_domains",
        }
        compact_task = {key: task.get(key) for key in keep_keys if task.get(key) not in (None, "", [])}
        payload = {
            "original_query": _compact_text(query, max_chars=240),
            "targets_gap": _compact_text(targets_gap or task.get("targets_gap") or task.get("root_cause"), max_chars=160),
            "task": compact_task,
        }
        return (
            "Fill this evidence gap for an industry research report. Search the web and return JSON only. "
            "Prefer official, company, regulator, exchange, association, reputable research, or major media sources. "
            "Every evidence item must have source_url. Do not invent facts.\n"
            "JSON: {\"summary\":\"...\",\"evidence\":[{\"claim\":\"fact with number/unit/period if any\","
            "\"metric\":\"...\",\"value\":\"...\",\"period\":\"...\",\"source_title\":\"...\",\"source_url\":\"https://...\","
            "\"confidence\":0.0}],\"evidence_gap\":[\"...\"]}\n"
            f"Task: {json.dumps(payload, ensure_ascii=False, default=str)}"
        )
    payload = {
        "original_query": query,
        "targets_gap": targets_gap,
        "evidence_goal": task.get("evidence_goal"),
        "proof_role": task.get("proof_role") or task.get("evidence_type"),
        "source_priority": _as_list(task.get("source_priority")),
        "must_have_terms": _as_list(task.get("must_have_terms")),
        "forbidden_terms": _as_list(task.get("forbidden_terms")),
    }
    return (
        "You are filling evidence gaps for an investment/industry research report. "
        "Run web search and return only JSON. Prefer primary sources, regulator/official pages, "
        "exchange filings, company announcements, industry associations, reputable research firms, "
        "and major financial media. Do not invent facts.\n\n"
        "Return this JSON shape:\n"
        "{"
        "\"summary\":\"brief grounded summary\","
        "\"evidence\":[{\"claim\":\"full evidence sentence with original number/unit/period\","
        "\"metric\":\"metric name\",\"value\":\"original value\",\"period\":\"date or period\","
        "\"source_title\":\"source title\",\"source_url\":\"https://...\",\"confidence\":0.0}],"
        "\"evidence_gap\":[\"remaining gap\"]"
        "}\n\n"
        f"Task payload: {json.dumps(payload, ensure_ascii=False, default=str)}"
    )


def build_openai_web_request_payload(
    query: str,
    *,
    search_task: Optional[Dict[str, Any]] = None,
    targets_gap: str = "",
) -> Dict[str, Any]:
    config = openai_web_search_config()
    tool: Dict[str, Any] = {
        "type": "web_search",
        "external_web_access": bool(config["external_web_access"]),
    }
    domain_task = {**_as_dict(search_task)}
    domain_task.setdefault("query", query)
    allowed_domains = allowed_domains_for_task(domain_task)
    if allowed_domains:
        tool["filters"] = {"allowed_domains": allowed_domains}
    payload: Dict[str, Any] = {
        "model": config["model"],
        "tools": [tool],
        "tool_choice": config["tool_choice"],
        "include": ["web_search_call.action.sources"],
        "input": build_openai_web_prompt(query, search_task=search_task, targets_gap=targets_gap),
    }
    if config.get("reasoning_effort"):
        payload["reasoning"] = {"effort": config["reasoning_effort"]}
    return payload


def call_openai_web_search(
    query: str,
    *,
    search_task: Optional[Dict[str, Any]] = None,
    targets_gap: str = "",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    config = openai_web_search_config()
    if not config["enabled"]:
        raise RuntimeError("OpenAI web search is disabled or OPENAI_API_KEY is not configured.")
    payload = build_openai_web_request_payload(query, search_task=search_task, targets_gap=targets_gap)
    request = urllib.request.Request(
        str(config["url"]),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.build_opener().open(request, timeout=float(config["timeout"])) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI web search failed with HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenAI web search failed: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        snippet = raw[:1000].replace("\n", " ")
        raise RuntimeError(f"OpenAI web search returned non-JSON response: {snippet}") from exc
    elapsed_seconds = round(time.perf_counter() - started, 2)
    token_event = record_llm_usage(
        usage=data.get("usage", {}),
        provider="openai_responses",
        model=str(config["model"]),
        task="openai_web_search",
        profile="openai_web",
        api="openai_web_search_responses",
        elapsed_ms=max(0, int(elapsed_seconds * 1000)),
    )
    return data, {
        "request": payload,
        "elapsed_seconds": elapsed_seconds,
        "token_usage_event": token_event or {},
    }


def run_openai_web_search_child(
    *,
    query: str,
    search_task: Optional[Dict[str, Any]] = None,
    targets_gap: str = "",
    round_number: int = 1,
) -> Dict[str, Any]:
    try:
        response, meta = call_openai_web_search(query, search_task=search_task, targets_gap=targets_gap)
    except RuntimeError as exc:
        text = str(exc).lower()
        if "timed out" not in text and "timeout" not in text:
            raise
        retry_task = {**_as_dict(search_task), "openai_retry_compact_prompt": True}
        retry_query = _compact_text(
            " ".join(
                str(value or "")
                for value in [
                    query,
                    retry_task.get("chapter_title"),
                    retry_task.get("proof_role"),
                    retry_task.get("root_cause"),
                    retry_task.get("evidence_goal"),
                ]
            ),
            max_chars=260,
        )
        try:
            response, meta = call_openai_web_search(retry_query or query, search_task=retry_task, targets_gap=targets_gap)
            meta = {**meta, "retry_with_compact_prompt": True, "primary_timeout_error": str(exc)}
            search_task = retry_task
            query = retry_query or query
        except RuntimeError as retry_exc:
            raise RuntimeError(f"{exc}; retry_with_compact_prompt failed: {retry_exc}") from retry_exc
    child = normalize_openai_web_response(
        response,
        query=query,
        search_task=search_task,
        targets_gap=targets_gap,
        round_number=round_number,
    )
    child["limitations"] = {**_as_dict(child.get("limitations")), "request_meta": meta}
    return child
