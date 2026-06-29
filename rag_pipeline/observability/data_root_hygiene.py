from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence
from urllib.parse import urlparse

from rag_pipeline.contracts.evidence_quality import (
    BLOCKING_CONTENT_SHAPE_ISSUES,
    NON_CLAIM_ALLOWED_USES,
    NON_CLAIM_ANALYSIS_READINESS,
)


LINEAGE_FIELDS = ("requirement_id", "source_id", "search_task_id")

TEXT_FIELDS = (
    "fact",
    "distilled_fact",
    "clean_fact",
    "summary",
    "evidence",
    "finding",
    "text",
    "content",
    "mainText",
    "main_text",
    "snippet",
    "title",
    "source_title",
)

GENERIC_METRIC_NAMES = {
    "数据指标",
    "定性事实",
    "关键事实",
    "数据点",
    "source_check",
    "technology_product",
    "http_status",
    "response_code",
}
WEB_CHROME_RE = re.compile(
    r"(skip\s+to\s+content|login|sign\s+in|cookie|privacy\s+policy|terms\s+of\s+use|"
    r"please\s+enable\s+javascript|checking\s+your\s+browser|request\s+a\s+demo|book\s+a\s+demo|"
    r"请登录|登录后|注册\s*登录|隐私政策|版权声明|导航菜单)",
    re.I,
)
PAGE_SHELL_RE = re.compile(
    (
        "目\\s*录.{0,120}(?:/|01).{0,120}(?:02|03).{0,120}(?:04|07)"
        "|个人信息.{0,100}我的订单.{0,100}我的优惠券"
        "|我的下载.{0,100}我的上传.{0,100}我的订阅"
        "|回到首页.{0,100}搜索.{0,100}(?:VIP|付费社群|报告详情)"
        "|my\\s+orders.{0,100}my\\s+coupons.{0,100}my\\s+downloads"
    ),
    re.I | re.S,
)

MALFORMED_METRIC_RE = re.compile(r"在\s*\d{3,4}\s*为\s*[-+]?\d+(?:\.\d+)?\s*%?")
SOURCE_HEADING_FRAGMENT_RE = re.compile(r"(转化愿景为现实|愿景.{0,12}现实).{0,40}引言")
INTERNAL_DIAGNOSTIC_RE = re.compile(
    r"(semantic\s+judge|source_check|metric\s+fields\s+incomplete|not_allowed_until_repaired|"
    r"evidence_refs|coverage_matrix|diagnostic_only|补证|证据不足)",
    re.I,
)
LONG_BARE_INTEGER_RE = re.compile(r"^[-+]?\d{5,}$")
URL_OR_FILE_RE = re.compile(r"(https?://|www\.|\.pdf\b|\.html?\b|[_/-]\d{5,})", re.I)
GENERIC_SOURCE_TITLE_RE = re.compile(
    r"^(?:"
    r"source|iqs\s*source|official(?:\s+(?:ai\s+agent\s+)?(?:statistics|data|source|report|disclosure))?"
    r"|official\s+statistics\s+show|www\.[\w.-]+|[\w.-]+\.(?:com|cn|org|gov|net)"
    r")$",
    re.I,
)


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if value is None:
        return []
    return [value]


def _scalar(value: Any) -> str:
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return ""
    return str(value).strip()


def _safe_int(value: Any) -> int:
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


def _join_text(item: Mapping[str, Any], fields: Sequence[str] = TEXT_FIELDS) -> str:
    parts: List[str] = []
    source = _as_dict(item.get("source"))
    for field in fields:
        text = _scalar(item.get(field))
        if text:
            parts.append(text)
        source_text = _scalar(source.get(field))
        if source_text:
            parts.append(source_text)
    return " ".join(parts)


def _reason_counter(text: str) -> Counter[str]:
    reasons: Counter[str] = Counter()
    if not text:
        return reasons
    if WEB_CHROME_RE.search(text):
        reasons["web_chrome_or_login"] += 1
    if PAGE_SHELL_RE.search(text):
        reasons["page_shell_or_toc_fragment"] += 1
    if MALFORMED_METRIC_RE.search(text):
        reasons["malformed_metric_span"] += 1
    if SOURCE_HEADING_FRAGMENT_RE.search(text):
        reasons["source_heading_fragment"] += 1
    if INTERNAL_DIAGNOSTIC_RE.search(text):
        reasons["internal_diagnostic_text"] += 1
    return reasons


def _metric_reasons(item: Mapping[str, Any]) -> Counter[str]:
    reasons: Counter[str] = Counter()
    metric = _scalar(item.get("metric") or item.get("metric_name") or item.get("indicator"))
    value = _scalar(item.get("value") or item.get("metric_value"))
    unit = _scalar(item.get("unit") or item.get("metric_unit")).lower()
    if metric and metric.strip() in GENERIC_METRIC_NAMES:
        reasons["generic_metric_name"] += 1
    compact_value = re.sub(r"\s+", "", value)
    if compact_value and (LONG_BARE_INTEGER_RE.fullmatch(compact_value) or URL_OR_FILE_RE.search(value)):
        reasons["artifact_like_value"] += 1
    if unit in {"unknown", "none", "null", "n/a"}:
        reasons["unknown_unit"] += 1
    return reasons


def _analysis_gate_reasons(item: Mapping[str, Any]) -> Counter[str]:
    reasons: Counter[str] = Counter()
    allowed_use = _scalar(item.get("allowed_use") or _as_dict(item.get("evidence_card")).get("allowed_use")).lower()
    readiness = _scalar(item.get("analysis_readiness") or _as_dict(item.get("evidence_card")).get("analysis_readiness")).lower()
    shape_issues = {
        _scalar(issue)
        for issue in _as_list(item.get("content_shape_issues") or _as_dict(item.get("evidence_card")).get("content_shape_issues"))
        if _scalar(issue)
    }
    if allowed_use in NON_CLAIM_ALLOWED_USES:
        reasons["diagnostic_only_in_analysis_ready" if allowed_use == "diagnostic_only" else "non_claim_allowed_use_in_analysis_ready"] += 1
    if readiness in NON_CLAIM_ANALYSIS_READINESS:
        reasons["clue_only_in_analysis_ready" if readiness == "clue_only" else "non_claim_readiness_in_analysis_ready"] += 1
    if shape_issues & BLOCKING_CONTENT_SHAPE_ISSUES:
        reasons["shape_issue_in_analysis_ready"] += 1
    return reasons


def _source_type_counts(items: Iterable[Any]) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for item in items:
        payload = _as_dict(item)
        source = _as_dict(payload.get("source"))
        source_type = _scalar(payload.get("source_type") or source.get("source_type")).lower()
        if source_type:
            counts[source_type] += 1
    return dict(counts)


def _summary(items: List[Dict[str, Any]], reason_counts: Counter[str], dirty_flags: List[bool]) -> Dict[str, Any]:
    dirty_count = sum(1 for value in dirty_flags if value)
    clean_count = max(0, len(items) - dirty_count)
    return {
        "schema_version": "data_root_hygiene_v1",
        "item_count": len(items),
        "dirty_item_count": dirty_count,
        "clean_item_count": clean_count,
        "reason_counts": dict(reason_counts),
        "source_type_counts": _source_type_counts(items),
        "diagnostic_only": True,
        "must_not_render": True,
        "public_text_allowed": False,
    }


def inspect_source_hygiene(items: Iterable[Any]) -> Dict[str, Any]:
    """Inspect raw source/page/search-result payloads for upstream dirtiness.

    This is diagnostic-only. It deliberately does not reject policy, official,
    or media sources by type; only concrete dirty text patterns are counted.
    """

    payloads = [_as_dict(item) for item in items if isinstance(item, dict)]
    reasons: Counter[str] = Counter()
    dirty_flags: List[bool] = []
    samples: List[Dict[str, Any]] = []
    for item in payloads:
        item_reasons = _reason_counter(_join_text(item))
        dirty = bool(item_reasons)
        dirty_flags.append(dirty)
        reasons.update(item_reasons)
        if dirty and len(samples) < 8:
            samples.append(
                {
                    "url": _scalar(item.get("url") or item.get("source_url")),
                    "source_id": _scalar(item.get("source_id") or item.get("source_ref")),
                    "reasons": sorted(item_reasons),
                }
            )
    summary = _summary(payloads, reasons, dirty_flags)
    summary["dirty_samples"] = samples
    return summary


def inspect_evidence_analysis_hygiene(items: Iterable[Any]) -> Dict[str, Any]:
    """Inspect analysis-ready evidence/fact-card payloads before claim writing."""

    payloads = [_as_dict(item) for item in items if isinstance(item, dict)]
    reasons: Counter[str] = Counter()
    dirty_flags: List[bool] = []
    missing_lineage: Counter[str] = Counter()
    samples: List[Dict[str, Any]] = []
    for item in payloads:
        item_reasons = _reason_counter(_join_text(item))
        item_reasons.update(_metric_reasons(item))
        item_reasons.update(_analysis_gate_reasons(item))
        for field in LINEAGE_FIELDS:
            if not _scalar(item.get(field)):
                missing_lineage[field] += 1
        dirty = bool(item_reasons)
        dirty_flags.append(dirty)
        reasons.update(item_reasons)
        if dirty and len(samples) < 8:
            samples.append(
                {
                    "evidence_id": _scalar(item.get("evidence_id") or item.get("fact_id") or item.get("id")),
                    "source_id": _scalar(item.get("source_id") or item.get("source_ref")),
                    "requirement_id": _scalar(item.get("requirement_id")),
                    "reasons": sorted(item_reasons),
                }
            )
    summary = _summary(payloads, reasons, dirty_flags)
    summary["missing_lineage_counts"] = dict(missing_lineage)
    summary["dirty_samples"] = samples
    return summary


def _source_title(item: Mapping[str, Any]) -> str:
    source = _as_dict(item.get("source"))
    return _scalar(item.get("title") or item.get("source_title") or source.get("title") or source.get("source_title"))


def _source_url(item: Mapping[str, Any]) -> str:
    source = _as_dict(item.get("source"))
    return _scalar(item.get("url") or item.get("source_url") or source.get("url") or source.get("source_url"))


def _host(url: str) -> str:
    parsed = urlparse(str(url or ""))
    return parsed.netloc.lower().removeprefix("www.")


def _normalized_title(title: str) -> str:
    return re.sub(r"\s+", " ", str(title or "").strip()).lower()


def _generic_source_title(title: str) -> bool:
    text = _normalized_title(title)
    if not text:
        return True
    return bool(GENERIC_SOURCE_TITLE_RE.fullmatch(text))


def inspect_source_identity_hygiene(items: Iterable[Any]) -> Dict[str, Any]:
    """Detect source identity contamination such as one title reused across unrelated URLs.

    This is diagnostic-only. It targets title/url binding corruption and does
    not classify source quality by publisher type.
    """

    payloads = [_as_dict(item) for item in items if isinstance(item, dict)]
    reasons: Counter[str] = Counter()
    dirty_flags = [False for _ in payloads]
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for index, item in enumerate(payloads):
        title = _source_title(item)
        url = _source_url(item)
        host = _host(url)
        key = _normalized_title(title)
        if not title or not url or _generic_source_title(title):
            continue
        groups[key].append({"index": index, "title": title, "url": url, "host": host})

    suspicious_groups: List[Dict[str, Any]] = []
    for group in groups.values():
        hosts = sorted({entry["host"] for entry in group if entry.get("host")})
        urls = sorted({entry["url"] for entry in group if entry.get("url")})
        if len(hosts) < 3 and len(urls) < 4:
            continue
        for entry in group:
            dirty_flags[entry["index"]] = True
        count = len(group)
        reasons["same_title_many_hosts"] += count
        reasons["source_title_url_mismatch"] += count
        reasons["fallback_title_reused"] += count
        suspicious_groups.append(
            {
                "title": group[0]["title"],
                "item_count": count,
                "host_count": len(hosts),
                "url_count": len(urls),
                "hosts": hosts[:12],
                "sample_urls": urls[:8],
            }
        )

    summary = _summary(payloads, reasons, dirty_flags)
    summary["suspicious_title_groups"] = suspicious_groups[:8]
    summary["dirty_samples"] = [
        {
            "title": group.get("title"),
            "hosts": group.get("hosts", [])[:4],
            "reasons": ["same_title_many_hosts", "source_title_url_mismatch", "fallback_title_reused"],
        }
        for group in suspicious_groups[:8]
    ]
    return summary


def inspect_cache_hygiene(payload: Any) -> Dict[str, Any]:
    """Summarize cache health without reading cached content into the model."""

    data = _as_dict(payload)
    reasons: Counter[str] = Counter()
    hit_items = [_as_dict(item) for item in _as_list(data.get("hit_items") or data.get("hits")) if isinstance(item, dict)]
    polluted_count = _safe_int(data.get("polluted_count") or data.get("polluted_hit_count"))
    quarantined_count = _safe_int(data.get("quarantined_count") or data.get("quarantine_count"))
    stale_count = _safe_int(data.get("stale_count"))
    dirty_hit_count = 0
    dirty_samples: List[Dict[str, Any]] = []
    for item in hit_items:
        status = _scalar(item.get("status") or item.get("cache_status")).lower()
        polluted = bool(item.get("polluted") or item.get("is_polluted")) or status in {"polluted", "quarantined", "poisoned"}
        if polluted:
            dirty_hit_count += 1
            if len(dirty_samples) < 8:
                dirty_samples.append(
                    {
                        "cache_key": _scalar(item.get("cache_key") or item.get("key")),
                        "status": status,
                        "source_url": _scalar(item.get("source_url") or item.get("url")),
                    }
                )
    if polluted_count:
        reasons["cache_polluted_count"] = polluted_count
    if quarantined_count:
        reasons["cache_quarantined_count"] = quarantined_count
    if stale_count:
        reasons["cache_stale_count"] = stale_count
    if dirty_hit_count:
        reasons["cache_hit_polluted"] = dirty_hit_count
    return {
        "schema_version": "data_root_cache_hygiene_v1",
        "cache_hit_count": _safe_int(data.get("cache_hit_count") or data.get("hit_count")),
        "cache_miss_count": _safe_int(data.get("cache_miss_count") or data.get("miss_count")),
        "stale_count": stale_count,
        "polluted_count": polluted_count,
        "quarantined_count": quarantined_count,
        "dirty_hit_count": dirty_hit_count,
        "reason_counts": dict(reasons),
        "dirty_samples": dirty_samples,
        "diagnostic_only": True,
        "must_not_render": True,
        "public_text_allowed": False,
    }
