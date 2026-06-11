from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
from urllib.parse import urlparse


_LOCK = threading.Lock()
_SOURCE_LEVEL_RANK = {"A": 4, "B": 3, "C": 2, "D": 1}
_TRUSTED_SOURCE_TYPES = {
    "official",
    "government",
    "regulator",
    "filing",
    "company_filing",
    "financial_report",
    "annual_report",
    "exchange",
    "company_official",
    "standard",
    "technical_standard",
    "association",
    "research",
    "market_research",
    "consulting",
    "whitepaper",
    "industry_report",
    "brokerage",
    "think_tank",
}
_BLOCKED_SOURCE_TYPES = {
    "media",
    "news",
    "aggregator",
    "news_aggregator",
    "self_media",
    "wiki",
    "encyclopedia",
    "blog",
    "forum",
    "social",
    "search",
    "search_result",
}
_GENERIC_CACHE_MATCH_TERMS = {
    "ab",
    "analysis",
    "case",
    "check",
    "claim",
    "claims",
    "company",
    "content",
    "data",
    "evidence",
    "fact",
    "facts",
    "filing",
    "gap",
    "gaps",
    "industry",
    "market",
    "metric",
    "metrics",
    "missing",
    "official",
    "official_data",
    "proof",
    "public",
    "report",
    "reports",
    "research",
    "source",
    "source_check",
    "source_quality",
    "sources",
    "statistics",
    "statistic",
    "support",
}
_TRUSTED_DOMAIN_FRAGMENTS = (
    ".gov",
    "gov.cn",
    "sec.gov",
    "nist.gov",
    "stats.gov.cn",
    "miit.gov.cn",
    "ndrc.gov.cn",
    "csrc.gov.cn",
    "cninfo.com.cn",
    "sse.com.cn",
    "szse.cn",
    "hkexnews.hk",
    "idc.com",
    "counterpointresearch.com",
    "omdia.tech.informa.com",
    "canalys.com",
    "gartner.com",
    "mckinsey.com",
    "bcg.com",
    "deloitte.com",
    "pwc.com",
)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, min_value: int, max_value: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        return default
    return max(min_value, min(max_value, value))


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


def _path() -> Path:
    raw = os.getenv("TRUSTED_SOURCE_CACHE_PATH", "output/cache/trusted_sources.jsonl").strip()
    path = Path(raw)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _enabled() -> bool:
    return _env_flag("TRUSTED_SOURCE_CACHE_ENABLED", True)


def trusted_source_stats() -> Dict[str, Any]:
    """只读统计，供 cache_report 聚合（fail-open）。"""
    try:
        path = _path()
        if not path.exists():
            return {"enabled": _enabled(), "entry_count": 0, "path": str(path)}
        return {"enabled": _enabled(), "entry_count": len(_load_entries(path)), "path": str(path)}
    except Exception as exc:
        return {"enabled": _enabled(), "error": str(exc)}


def _compact_text(value: Any, *, max_chars: int = 1200) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "..."


def _hash_payload(payload: Dict[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _tokenize(value: Any) -> List[str]:
    text = str(value or "").lower()
    tokens = re.findall(r"[\w\u4e00-\u9fff]{2,}", text)
    return [token for token in tokens if token not in {"http", "https", "www"}]


def _distinctive_cache_terms(terms: Sequence[str]) -> set[str]:
    distinctive: set[str] = set()
    for term in terms:
        token = str(term or "").strip().lower()
        if not token or token in _GENERIC_CACHE_MATCH_TERMS:
            continue
        if len(token) <= 2 and not re.search(r"[\u4e00-\u9fff]", token):
            continue
        distinctive.add(token)
    return distinctive


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().removeprefix("www.")
    except Exception:
        return ""


def _source_level(value: Any) -> str:
    level = str(value or "").strip().upper()
    return level if level in _SOURCE_LEVEL_RANK else ""


_PLACEHOLDER_TITLES = {"official ai agent statistics"}
_PLACEHOLDER_TEXT = "official data shows ai agent adoption reached 50% in 2025"


def _scalar_text(value: Any) -> str:
    if isinstance(value, (dict, list, tuple, set)):
        return ""
    return str(value or "").strip()


def _looks_like_error_or_page_shell(text: Any, *, title: Any = "") -> bool:
    body = re.sub(r"\s+", " ", _scalar_text(text)).strip()
    header = re.sub(r"\s+", " ", _scalar_text(title)).strip()
    lowered = f"{header} {body}".lower()
    if not lowered.strip():
        return False
    if any(
        marker in lowered
        for marker in (
            "403 forbidden",
            "404 not found",
            "页面未找到",
            "页面找不到",
            "访问的页面不存在",
            "内容不存在或被删除",
            "permission denied",
            "access denied",
        )
    ):
        return True
    if re.match(r"^(url|source_url)\s*[:：]\s*https?://\S+\s*$", body, flags=re.I):
        return True
    if re.match(r"^(时间|date)\s*[:：]\s*[^。；;]{1,80}$", body, flags=re.I):
        return True
    if body.lower().startswith(("摘要：skip to main content", "skip to main content")):
        return True
    if "linkedin is better on the app" in lowered and len(body) < 600:
        return True
    return False


def _is_fake_or_placeholder(source: Dict[str, Any], text: str = "") -> bool:
    joined = " ".join(
        str(item or "")
        for item in [
            source.get("url"),
            source.get("source_url"),
            source.get("title"),
            source.get("publisher"),
            source.get("source"),
            text,
        ]
    ).lower()
    if "example.gov" in joined or "example.com" in joined:
        return True
    if "official data shows ai agent adoption reached 50% in 2025" in joined:
        return True
    title = str(source.get("title") or "").strip().lower()
    publisher = str(source.get("publisher") or source.get("source") or "").strip()
    if title in _PLACEHOLDER_TITLES:
        return True
    return title == "official" and not publisher


def _traceable(source: Dict[str, Any]) -> bool:
    if _is_fake_or_placeholder(source):
        return False
    if str(source.get("url") or source.get("source_url") or "").strip():
        return True
    doc_ref = str(source.get("document_id") or source.get("doc_id") or source.get("page_ref") or "").strip()
    if not doc_ref:
        return False
    fields = [
        bool(str(source.get("title") or "").strip()),
        bool(str(source.get("publisher") or source.get("source") or "").strip()),
        bool(str(source.get("date") or "").strip()),
    ]
    return sum(fields) >= 2


def _trusted_source_family_allowed(source_family: str, source: Dict[str, Any], domain: str) -> bool:
    family = str(source_family or "").strip().lower()
    source_type = str(source.get("source_type") or source.get("type") or "").strip().lower()
    text = f"{family} {source_type} {domain} {source.get('url') or source.get('source_url') or ''}".lower()
    if any(blocked in text for blocked in _BLOCKED_SOURCE_TYPES):
        return False
    if family in _TRUSTED_SOURCE_TYPES or source_type in _TRUSTED_SOURCE_TYPES:
        return True
    return any(fragment in domain for fragment in _TRUSTED_DOMAIN_FRAGMENTS)


def _iter_package_evidence(package: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for key in ("analysis_ready_evidence", "clean_evidence_list", "core_evidence", "supporting_evidence"):
        for item in _as_list(package.get(key)):
            if isinstance(item, dict):
                yield dict(item)
    for point in _as_list(package.get("raw_data_points")):
        if isinstance(point, dict):
            yield dict(point)


def _entry_from_evidence(query: str, item: Dict[str, Any], *, report_id: str = "", run_id: str = "") -> Optional[Dict[str, Any]]:
    source = dict(_as_dict(item.get("source")))
    source_url = _scalar_text(source.get("url") or source.get("source_url") or item.get("source_url") or item.get("url"))
    source_title = _scalar_text(source.get("title") or item.get("source_title") or item.get("title"))
    source_publisher = _scalar_text(source.get("publisher") or source.get("source") or item.get("publisher"))
    if source_url and not source.get("url"):
        source["url"] = source_url
    if source_title and not source.get("title"):
        source["title"] = source_title
    if source_publisher and not source.get("publisher"):
        source["publisher"] = source_publisher
    source_level = _source_level(item.get("source_level") or source.get("source_level"))
    fact = str(item.get("fact") or item.get("clean_fact") or item.get("content") or item.get("fact_description") or item.get("evidence") or "").strip()
    metric = str(item.get("metric") or item.get("metric_name") or item.get("indicator") or "").strip()
    value = str(item.get("value") or item.get("numeric_value") or "").strip()
    proof_role = str(item.get("proof_role") or item.get("evidence_type") or item.get("role") or "").strip().lower()
    allowed_use = str(item.get("allowed_use") or _as_dict(item.get("evidence_card")).get("allowed_use") or "").strip().lower()
    if not source_level or _SOURCE_LEVEL_RANK.get(source_level, 0) < _SOURCE_LEVEL_RANK["B"]:
        return None
    if not _traceable(source):
        return None
    if _is_fake_or_placeholder(source, fact):
        return None
    if _looks_like_error_or_page_shell(fact, title=source_title):
        return None
    if not (fact or metric or value):
        return None
    if proof_role not in {"metric", "source_check", "case", "customer_case", "counter", "filing", "company_filing", "technology_product", "support", ""}:
        proof_role = "source_check"
    source_family = str(item.get("source_family") or source.get("source_family") or source.get("source_type") or item.get("source_type") or "").strip()
    source_domain = _domain(source_url)
    if not _trusted_source_family_allowed(source_family, source, source_domain):
        return None
    chapter_id = str(item.get("chapter_id") or item.get("dimension_id") or item.get("hypothesis_id") or "").strip()
    chapter_title = str(item.get("chapter_title") or item.get("dimension_name") or item.get("hypothesis_statement") or "").strip()
    entry_id = "trusted:" + _hash_payload(
        {
            "url": source_url,
            "title": source_title,
            "fact": fact[:360],
            "metric": metric,
            "period": str(item.get("period") or source.get("date") or ""),
            "value": value,
        }
    )
    terms: List[str] = []
    for value_for_terms in (
        query,
        fact,
        metric,
        source_title,
        source_publisher,
        source_url,
        proof_role,
        chapter_id,
        chapter_title,
    ):
        terms.extend(_tokenize(value_for_terms))
    terms = list(dict.fromkeys(terms))[:80]
    return {
        "id": entry_id,
        "topic_key": " ".join(terms[:20]),
        "topic_terms": terms,
        "query": query,
        "report_id": report_id,
        "run_id": run_id,
        "source_url": source_url,
        "source_domain": source_domain,
        "title": source_title,
        "publisher": source_publisher,
        "date": str(source.get("date") or item.get("period") or item.get("date") or "").strip(),
        "source_level": source_level,
        "source_family": source_family,
        "source_type": source_family,
        "proof_role": proof_role or ("metric" if metric and value else "source_check"),
        "allowed_use": allowed_use or ("core_claim" if source_level in {"A", "B"} else "directional_signal"),
        "supported_chapters": [item for item in [chapter_id, chapter_title] if item],
        "last_verified_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "traceability_status": "traceable",
        "fact_description": _compact_text(fact, max_chars=1200),
        "metric_name": metric,
        "value": value,
        "unit": str(item.get("unit") or item.get("numeric_unit") or "").strip(),
        "period": str(item.get("period") or source.get("date") or item.get("date") or "").strip(),
        "confidence_score": float(item.get("confidence") or item.get("confidence_score") or 0.75),
        "raw": {
            **item,
            "source": source,
            "source_level": source_level,
            "source_type": source_family,
            "proof_role": proof_role or ("metric" if metric and value else "source_check"),
            "evidence_origin": "trusted_source_cache",
        },
    }


def _load_entries(path: Path, *, max_entries: int = 20000) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    entries: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    entries.append(payload)
    except OSError:
        return []
    if len(entries) > max_entries:
        return entries[-max_entries:]
    return entries


def store_trusted_sources_from_package(
    *,
    query: str,
    evidence_package: Dict[str, Any],
    report_id: str = "",
    run_id: str = "",
) -> Dict[str, Any]:
    if not _enabled():
        return {"enabled": False, "stored_count": 0, "skipped_count": 0, "path": str(_path())}
    package = _as_dict(evidence_package)
    path = _path()
    entries: List[Dict[str, Any]] = []
    skipped = 0
    for item in _iter_package_evidence(package):
        entry = _entry_from_evidence(query, item, report_id=report_id, run_id=run_id)
        if not entry:
            skipped += 1
            continue
        entries.append(entry)
    if not entries:
        return {"enabled": True, "stored_count": 0, "skipped_count": skipped, "path": str(path)}
    with _LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        existing_ids = {str(item.get("id") or "") for item in _load_entries(path, max_entries=50000)}
        new_entries = [entry for entry in entries if str(entry.get("id") or "") not in existing_ids]
        if new_entries:
            with path.open("a", encoding="utf-8") as handle:
                for entry in new_entries:
                    handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True, default=str) + "\n")
    return {
        "enabled": True,
        "stored_count": len(new_entries),
        "skipped_count": skipped + max(0, len(entries) - len(new_entries)),
        "path": str(path),
    }


def _match_score(entry: Dict[str, Any], terms: Sequence[str], task: Dict[str, Any]) -> float:
    distinctive_terms = _distinctive_cache_terms(terms)
    distinctive_hits = 0
    haystack = " ".join(
        str(item or "").lower()
        for item in [
            entry.get("topic_key"),
            entry.get("source_url"),
            entry.get("title"),
            entry.get("publisher"),
            entry.get("fact_description"),
            entry.get("metric_name"),
            entry.get("proof_role"),
            " ".join(_as_list(entry.get("supported_chapters"))),
        ]
    )
    score = 0.0
    for term in terms:
        token = str(term or "").lower()
        if not token:
            continue
        if token in haystack:
            score += 1.0
            if token in distinctive_terms:
                distinctive_hits += 1
    task_role = str(task.get("proof_role") or task.get("evidence_type") or "").strip().lower()
    entry_role = str(entry.get("proof_role") or "").strip().lower()
    if task_role and entry_role:
        if task_role == entry_role:
            score += 3.0
        elif task_role == "source_check" and entry_role in {"source_check", "metric", "filing", "support"}:
            score += 1.5
        else:
            score -= 1.0
    task_chapter = str(task.get("chapter_id") or task.get("dimension_id") or "").strip()
    if task_chapter and task_chapter in _as_list(entry.get("supported_chapters")):
        score += 2.0
    if distinctive_terms and distinctive_hits <= 0:
        return 0.0
    return score


def lookup_trusted_sources(
    repair_task: Dict[str, Any],
    *,
    min_source_level: Sequence[str] | str = ("A", "B"),
    required_fields: Optional[Sequence[str]] = None,
    max_hits: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if not _enabled():
        return []
    task = _as_dict(repair_task)
    values: List[Any] = [
        task.get("query"),
        task.get("suggested_query"),
        task.get("targets_gap"),
        task.get("evidence_goal"),
        task.get("chapter_title"),
        task.get("proof_role"),
        task.get("proof_standard"),
    ]
    values.extend(_as_list(task.get("topic_terms")))
    terms = list(dict.fromkeys(token for value in values for token in _tokenize(value)))[:80]
    if not terms:
        return []
    levels = [str(item).strip().upper() for item in (min_source_level if isinstance(min_source_level, (list, tuple, set)) else [min_source_level]) if str(item).strip()]
    minimum_rank = min((_SOURCE_LEVEL_RANK.get(level, 0) for level in levels), default=_SOURCE_LEVEL_RANK["B"])
    required = {str(item).strip().lower() for item in list(required_fields or []) if str(item).strip()}
    max_hits = max_hits or _env_int("TRUSTED_SOURCE_CACHE_MAX_HITS_PER_TASK", 4, min_value=1, max_value=50)
    entries = _load_entries(_path(), max_entries=_env_int("TRUSTED_SOURCE_CACHE_MAX_ENTRIES_SCAN", 20000, min_value=100, max_value=100000))
    hits: List[Dict[str, Any]] = []
    for entry in entries:
        if _SOURCE_LEVEL_RANK.get(_source_level(entry.get("source_level")), 0) < minimum_rank:
            continue
        if str(entry.get("traceability_status") or "").strip().lower() != "traceable":
            continue
        raw = _as_dict(entry.get("raw"))
        source = _as_dict(raw.get("source")) or {
            "url": entry.get("source_url"),
            "title": entry.get("title"),
            "publisher": entry.get("publisher"),
            "date": entry.get("date"),
        }
        if _is_fake_or_placeholder(source, str(entry.get("fact_description") or "")):
            continue
        if _looks_like_error_or_page_shell(entry.get("fact_description"), title=entry.get("title")):
            continue
        if "source" in required and not str(entry.get("source_url") or "").strip():
            continue
        if "metric" in required and not str(entry.get("metric_name") or "").strip():
            continue
        if "value" in required and not str(entry.get("value") or "").strip():
            continue
        if "period" in required and not str(entry.get("period") or entry.get("date") or "").strip():
            continue
        score = _match_score(entry, terms, task)
        if score <= 0:
            continue
        hits.append(
            {
                "evidence_id": entry.get("id"),
                "match_score": score,
                "source_level": entry.get("source_level"),
                "source_type": entry.get("source_type") or entry.get("source_family"),
                "proof_role": entry.get("proof_role"),
                "allowed_use": entry.get("allowed_use"),
                "confidence_score": entry.get("confidence_score") or 0.75,
                "fact_description": entry.get("fact_description"),
                "metric_name": entry.get("metric_name"),
                "value": entry.get("value"),
                "unit": entry.get("unit"),
                "period": entry.get("period") or entry.get("date"),
                "source_url": entry.get("source_url"),
                "source_domain": entry.get("source_domain"),
                "raw": raw or entry,
                "trusted_source_cache": True,
            }
        )
    hits.sort(key=lambda item: float(item.get("match_score") or 0.0), reverse=True)
    return hits[:max_hits]
