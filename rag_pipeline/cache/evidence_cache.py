from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import urlparse

from rag_pipeline.runtime_cache import json_safe_default

logger = logging.getLogger(__name__)

_CACHE_VERSION = 1
_CACHE: Optional["EvidenceCache"] = None
_CACHE_LOCK = threading.Lock()
_ACTIVITY_LOCK = threading.Lock()
_ACTIVITY: Dict[str, int] = {
    "search_hit": 0,
    "search_negative_hit": 0,
    "search_store": 0,
    "search_negative_store": 0,
    "search_bypass": 0,
    "evidence_hit": 0,
    "evidence_store": 0,
    "evidence_bypass": 0,
    "skipped_deep_count": 0,
    "stale_count": 0,
    "error_count": 0,
}
_BYPASS_REASONS: Dict[str, int] = {}

_SOURCE_LEVEL_RANK = {"A": 4, "B": 3, "C": 2, "D": 1, "UNKNOWN": 0, "": 0}
_REJECTED_ROLES = {"rejected", "spam", "irrelevant", "blacklisted", "exclude"}
_APPENDIX_ONLY_USES = {"appendix_only", "rejected"}
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


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, min_value: int = 0, max_value: int = 10_000_000) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return min(max_value, max(min_value, value))


def _now() -> float:
    return time.time()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=json_safe_default)


def _json_loads(value: Any, default: Any) -> Any:
    if not value:
        return copy.deepcopy(default)
    try:
        return json.loads(str(value))
    except Exception:
        return copy.deepcopy(default)


def _hash_payload(payload: Dict[str, Any]) -> str:
    raw = _json_dumps(payload)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _compact_text(value: Any, max_chars: int = 12000) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "..."


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _normalized_list(value: Any) -> List[str]:
    if isinstance(value, str):
        raw_items = re.split(r"[,，;；|]+", value)
    elif isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    else:
        raw_items = []
    result: List[str] = []
    seen = set()
    for item in raw_items:
        text = str(item or "").strip().lower()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _first_option_value(options: Dict[str, Any], task: Dict[str, Any], names: Sequence[str]) -> str:
    for source in (task, options):
        for name in names:
            if name in source and source.get(name) not in (None, ""):
                value = source.get(name)
                if isinstance(value, bool):
                    return "true" if value else "false"
                if isinstance(value, (list, tuple, set)):
                    return ",".join(sorted(str(item).strip().lower() for item in value if str(item).strip()))
                return str(value).strip()
    return ""


def _normalize_query(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip().lower())
    text = re.sub(r"[\"'`]+", "", text)
    return text[:500]


def _tokenize(value: Any) -> List[str]:
    text = str(value or "").lower()
    raw_tokens = re.findall(r"[a-z0-9][a-z0-9._/-]{1,}|[\u4e00-\u9fff]{2,}", text, re.I)
    tokens: List[str] = []
    seen = set()
    for token in raw_tokens:
        cleaned = token.strip("._/- ")
        if len(cleaned) < 2 or cleaned in seen:
            continue
        seen.add(cleaned)
        tokens.append(cleaned)
        if len(tokens) >= 80:
            break
    return tokens


def _distinctive_cache_terms(terms: Sequence[str]) -> set[str]:
    distinctive: set[str] = set()
    for term in terms:
        text = str(term or "").strip().lower()
        if not text or text in _GENERIC_CACHE_MATCH_TERMS:
            continue
        if len(text) <= 2 and not re.search(r"[\u4e00-\u9fff]", text):
            continue
        distinctive.add(text)
    return distinctive


def _domain(url: Any) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""
    try:
        return urlparse(raw).netloc.lower()
    except Exception:
        return ""


def _source_type_from_text(source: Dict[str, Any], fallback: str = "") -> str:
    text = " ".join(
        [
            str(source.get("title") or ""),
            str(source.get("publisher") or ""),
            str(source.get("url") or source.get("source_url") or ""),
        ]
    ).lower()
    if re.search(r"(caifuhao\.eastmoney|guba\.eastmoney|mguba\.eastmoney|baijiahao|toutiao|zhihu|xueqiu|weibo|sohu|book118|docin|doc88|renrendoc|wenku\.baidu)", text):
        return "self_media"
    if re.search(r"(view\.inews\.qq\.com|kuaixun|finance\.sina\.com\.cn|news\.10jqka\.com\.cn|news\.futunn\.com|eastmoney\.com)", text):
        return "news"
    explicit = str(source.get("source_type") or source.get("type") or fallback or "").strip().lower()
    if explicit:
        return explicit
    if re.search(r"(gov\.|\.gov|stats\.gov|miit|ndrc|samr|official|监管|统计局|政府|标准|专利)", text):
        return "official"
    if re.search(r"(10-k|annual report|filing|cninfo|sse\.com|szse|sec\.gov|公告|年报|财报)", text):
        return "financial_report"
    if re.search(r"(research|report|whitepaper|idc|counterpoint|omdia|dscc|canalys|协会|研报|白皮书)", text):
        return "research"
    if re.search(r"(news|reuters|bloomberg|caixin|媒体|新闻)", text):
        return "news"
    return "unknown"


def _source_level(source_type: str, explicit: Any = "") -> str:
    source_type = str(source_type or "").strip().lower()
    if source_type in {"self_media", "ugc"}:
        return "D"
    if source_type in {"media", "news", "unknown"}:
        return "C"
    level = str(explicit or "").strip().upper()
    if level in _SOURCE_LEVEL_RANK:
        return level
    if source_type in {"official", "government", "financial_report", "annual_report", "prospectus", "exchange", "patent", "standard"}:
        return "A"
    if source_type in {"research", "academic", "industry_report", "association", "consulting", "market_research", "brokerage", "whitepaper", "authoritative_secondary", "company_announcement", "company_official", "product_doc"}:
        return "B"
    return "C"


def _ttl_seconds(source_type: str, payload: Optional[Dict[str, Any]] = None, *, negative: bool = False) -> int:
    if negative:
        return _env_int("EVIDENCE_CACHE_NEGATIVE_TTL_HOURS", 6, min_value=1, max_value=720) * 3600
    payload = _as_dict(payload)
    if str(payload.get("timeRange") or "").strip() in {"OneDay", "OneWeek"}:
        return 2 * 86400
    source_type = str(source_type or "").strip().lower()
    if source_type in {"news", "media", "event"}:
        return 2 * 86400
    if source_type in {"financial_report", "annual_report", "filing", "announcement", "exchange"}:
        return 90 * 86400
    if source_type in {"policy", "regulation", "official", "government", "standard", "patent"}:
        return 180 * 86400
    if source_type in {"research", "industry_report", "association", "market_research"}:
        return 14 * 86400
    return 7 * 86400


_HARD_SEARCH_ERROR_PATTERNS = (
    "timeout",
    "timed out",
    "permission",
    "quota",
    "unauthorized",
    "forbidden",
    "auth",
    "ssl",
    "connection",
    "connect",
    "network",
    "max retries",
    "failed",
    "exception",
)


def _search_errors_are_hard(errors: Sequence[Any]) -> bool:
    for error in errors:
        text = str(error or "").strip().lower()
        if not text:
            continue
        if any(pattern in text for pattern in _HARD_SEARCH_ERROR_PATTERNS):
            return True
    return False


_SOURCE_TYPE_ALIASES: Dict[str, set[str]] = {
    "official_data": {"official", "government", "policy", "regulation", "standard", "patent"},
    "official": {"official", "government", "policy", "regulation", "standard", "patent"},
    "policy": {"official", "government", "policy", "regulation", "standard"},
    "regulation": {"official", "government", "policy", "regulation", "standard"},
    "standard": {"official", "government", "standard"},
    "filing": {"financial_report", "annual_report", "filing", "announcement", "exchange"},
    "filing_company": {"financial_report", "annual_report", "filing", "announcement", "exchange"},
    "company_filing": {"financial_report", "annual_report", "filing", "announcement", "exchange"},
    "announcement": {"financial_report", "annual_report", "filing", "announcement", "exchange"},
    "market_research": {"research", "industry_report", "association", "academic"},
    "industry_research": {"research", "industry_report", "association", "academic"},
    "research": {"research", "industry_report", "association", "academic"},
    "technology_product": {"patent", "standard", "official", "government", "research", "industry_report", "academic", "financial_report"},
    "iqs_lane_5": {"patent", "standard", "official", "government", "research", "industry_report", "academic", "financial_report"},
    "patent": {"patent", "standard", "official", "government", "research"},
    "news_event": {"news", "media"},
    "news": {"news", "media"},
}


def _source_type_hints(values: Sequence[str]) -> set[str]:
    hints: set[str] = set()
    for value in values:
        text = str(value or "").strip().lower()
        if not text:
            continue
        if text in _SOURCE_TYPE_ALIASES:
            hints.update(_SOURCE_TYPE_ALIASES[text])
        if any(token in text for token in ("official", "gov", "policy", "regulation", "standard", "政府", "政策", "监管", "标准", "专利", "统计")):
            hints.update(_SOURCE_TYPE_ALIASES["official_data"])
        if any(token in text for token in ("filing", "annual", "announcement", "financial", "财报", "公告", "年报", "披露", "交易所")):
            hints.update(_SOURCE_TYPE_ALIASES["filing"])
        if any(token in text for token in ("research", "report", "association", "研报", "研究", "协会", "咨询", "白皮书")):
            hints.update(_SOURCE_TYPE_ALIASES["market_research"])
        if any(token in text for token in ("technology", "product", "patent", "technical", "技术", "专利", "产品", "量产", "良率")):
            hints.update(_SOURCE_TYPE_ALIASES["technology_product"])
        if any(token in text for token in ("news", "event", "新闻", "事件", "媒体", "风险")):
            hints.update(_SOURCE_TYPE_ALIASES["news_event"])
    return hints


def _task_source_type_hints(task: Dict[str, Any]) -> set[str]:
    values: List[str] = []
    for key in ("lane_targets", "source_priority", "required_source_types", "source_types"):
        values.extend(_normalized_list(task.get(key)))
    for key in ("lane_type", "scheduled_lane_type", "source_type", "proof_role", "evidence_type"):
        value = str(task.get(key) or "").strip()
        if value:
            values.append(value)
    return _source_type_hints(values)


def _row_matches_source_type(row: sqlite3.Row, raw: Dict[str, Any], required_types: set[str]) -> bool:
    if not required_types:
        return True
    source = _as_dict(raw.get("source"))
    row_type = str(row["source_type"] or raw.get("source_type") or source.get("source_type") or "").strip().lower()
    if row_type in required_types:
        return True
    text = " ".join(
        [
            row_type,
            str(row["source_domain"] or ""),
            str(row["source_url"] or source.get("url") or ""),
            str(source.get("title") or ""),
            str(source.get("publisher") or ""),
        ]
    ).lower()
    inferred = _source_type_hints([text])
    return bool(inferred.intersection(required_types))


def record_cache_activity(**increments: int) -> None:
    with _ACTIVITY_LOCK:
        for key, value in increments.items():
            if not value:
                continue
            _ACTIVITY[key] = int(_ACTIVITY.get(key, 0)) + int(value)


def record_cache_bypass(reason: str, *, search: bool = False, evidence: bool = False) -> None:
    reason_key = re.sub(r"\s+", "_", str(reason or "unknown").strip().lower()) or "unknown"
    with _ACTIVITY_LOCK:
        if search:
            _ACTIVITY["search_bypass"] = int(_ACTIVITY.get("search_bypass", 0)) + 1
        if evidence:
            _ACTIVITY["evidence_bypass"] = int(_ACTIVITY.get("evidence_bypass", 0)) + 1
        _BYPASS_REASONS[reason_key] = int(_BYPASS_REASONS.get(reason_key, 0)) + 1


def evidence_cache_activity_summary() -> Dict[str, Any]:
    activity = {}
    bypass_reasons = {}
    with _ACTIVITY_LOCK:
        activity = dict(_ACTIVITY)
        bypass_reasons = dict(_BYPASS_REASONS)
    stats: Dict[str, Any] = {}
    try:
        stats = get_evidence_cache().stats()
    except Exception:
        stats = {"enabled": False}
    return {**activity, "bypass_reasons": bypass_reasons, "store": stats}


def _cache_enabled() -> bool:
    return _env_flag("EVIDENCE_CACHE_ENABLED", True)


def _cache_path() -> Path:
    raw = os.getenv("EVIDENCE_CACHE_PATH", "output/cache/evidence_cache.sqlite")
    path = Path(raw)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


class EvidenceCache:
    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or _cache_path()
        self._init_lock = threading.Lock()
        self._initialized = False

    def enabled(self) -> bool:
        return _cache_enabled()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.path), timeout=2.0)
        conn.row_factory = sqlite3.Row
        journal_mode = str(os.getenv("EVIDENCE_CACHE_SQLITE_JOURNAL_MODE", "MEMORY") or "MEMORY").strip().upper()
        if journal_mode not in {"WAL", "MEMORY", "OFF", "DELETE", "TRUNCATE", "PERSIST"}:
            journal_mode = "MEMORY"
        try:
            conn.execute(f"PRAGMA journal_mode={journal_mode}")
        except sqlite3.OperationalError:
            conn.execute("PRAGMA journal_mode=MEMORY")
        return conn

    def _ensure_schema(self) -> None:
        if self._initialized or not self.enabled():
            return
        with self._init_lock:
            if self._initialized:
                return
            conn = self._connect()
            try:
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS search_cache (
                        cache_key TEXT PRIMARY KEY,
                        query_key TEXT NOT NULL,
                        query_text TEXT NOT NULL,
                        engine_type TEXT,
                        time_range TEXT,
                        contents TEXT,
                        phase TEXT,
                        proof_role TEXT,
                        lane_targets_json TEXT,
                        response_json TEXT NOT NULL,
                        search_trace_json TEXT,
                        negative INTEGER DEFAULT 0,
                        created_at REAL NOT NULL,
                        expires_at REAL NOT NULL,
                        hit_count INTEGER DEFAULT 0,
                        last_hit_at REAL
                    )
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_search_expires ON search_cache(expires_at)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_search_query ON search_cache(query_key)")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS evidence_cache (
                        evidence_id TEXT PRIMARY KEY,
                        topic_key TEXT NOT NULL,
                        query_terms_json TEXT,
                        source_url TEXT,
                        source_domain TEXT,
                        source_level TEXT,
                        source_type TEXT,
                        proof_role TEXT,
                        allowed_use TEXT,
                        confidence_score REAL,
                        fact_description TEXT,
                        metric_name TEXT,
                        value TEXT,
                        unit TEXT,
                        period TEXT,
                        raw_json TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        expires_at REAL NOT NULL,
                        hit_count INTEGER DEFAULT 0,
                        last_hit_at REAL
                    )
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_evidence_expires ON evidence_cache(expires_at)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_evidence_topic ON evidence_cache(topic_key)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_evidence_source ON evidence_cache(source_domain)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_evidence_role ON evidence_cache(proof_role, source_level)")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS evidence_lineage (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        evidence_id TEXT NOT NULL,
                        search_key TEXT,
                        report_id TEXT,
                        run_id TEXT,
                        query TEXT,
                        task_id TEXT,
                        gap_id TEXT,
                        loop_name TEXT,
                        created_at REAL NOT NULL
                    )
                    """
                )
                conn.execute("PRAGMA user_version = %d" % _CACHE_VERSION)
                conn.commit()
            finally:
                conn.close()
            self._initialized = True

    def _safe(self, fn, default: Any) -> Any:
        if not self.enabled():
            return default
        try:
            self._ensure_schema()
            return fn()
        except (sqlite3.Error, OSError, ValueError, TypeError) as exc:
            record_cache_activity(error_count=1)
            logger.debug("Evidence cache bypassed", extra={"error": str(exc)})
            return default

    def search_key(self, query: str, search_options: Dict[str, Any], search_task: Optional[Dict[str, Any]] = None) -> str:
        options = _as_dict(search_options)
        task = _as_dict(search_task) or _as_dict(options.get("search_task"))
        payload = {
            "query": _normalize_query(query),
            "engineType": str(options.get("engineType") or "").strip(),
            "timeRange": str(options.get("timeRange") or "").strip(),
            "contents": str(options.get("contents") or "").strip(),
            "phase": str(options.get("phase") or "").strip(),
            "numResults": _first_option_value(options, task, ("numResults", "num_results", "topK", "top_k", "limit")),
            "maxQueries": _first_option_value(options, task, ("maxQueries", "max_queries")),
            "maxSearchTasks": _first_option_value(options, task, ("maxSearchTasks", "max_search_tasks")),
            "enableBatchSearch": _first_option_value(options, task, ("enableBatchSearch", "enable_batch_search")),
            "proof_role": str(task.get("proof_role") or options.get("proof_role") or "").strip().lower(),
            "lane_targets": sorted(_normalized_list(task.get("lane_targets") or options.get("lane_targets"))),
            "source_priority": sorted(_normalized_list(task.get("source_priority") or options.get("source_priority"))),
            "must_have_terms": sorted(_normalized_list(task.get("must_have_terms") or options.get("must_have_terms"))),
            "forbidden_terms": sorted(_normalized_list(task.get("forbidden_terms") or options.get("forbidden_terms"))),
        }
        return "search:" + _hash_payload(payload)

    def lookup_search(self, query: str, search_options: Dict[str, Any], search_task: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        if not _env_flag("EVIDENCE_CACHE_SEARCH_READ_ENABLED", True):
            record_cache_bypass("persistent_search_read_disabled", search=True)
            return None
        if not _env_flag("IQS_SEARCH_CACHE_ENABLED", True):
            record_cache_bypass("iqs_search_cache_disabled", search=True)
            return None
        options = _as_dict(search_options)
        if str(options.get("disable_cache") or "").strip().lower() in {"1", "true", "yes", "on"}:
            record_cache_bypass("request_disable_cache", search=True)
            return None
        if str(options.get("cache_ttl_seconds") or "").strip() == "0":
            record_cache_bypass("request_cache_ttl_zero", search=True)
            return None
        key = self.search_key(query, search_options, search_task)
        now = _now()

        def _lookup() -> Optional[Dict[str, Any]]:
            with self._connect() as conn:
                row = conn.execute("SELECT * FROM search_cache WHERE cache_key = ?", (key,)).fetchone()
                if not row:
                    return None
                if float(row["expires_at"] or 0) <= now:
                    conn.execute("DELETE FROM search_cache WHERE cache_key = ?", (key,))
                    record_cache_activity(stale_count=1)
                    return None
                conn.execute(
                    "UPDATE search_cache SET hit_count = hit_count + 1, last_hit_at = ? WHERE cache_key = ?",
                    (now, key),
                )
                payload = _json_loads(row["response_json"], {})
                if not isinstance(payload, dict):
                    return None
                cache_meta = _as_dict(payload.get("cache"))
                payload["cache"] = {
                    **cache_meta,
                    "enabled": True,
                    "hit": True,
                    "layer": "search_cache",
                    "persistent": True,
                    "negative": bool(row["negative"]),
                    "cache_key": key,
                }
                record_cache_activity(search_negative_hit=1 if row["negative"] else 0, search_hit=0 if row["negative"] else 1)
                return payload

        return self._safe(_lookup, None)

    def store_search(self, query: str, search_options: Dict[str, Any], search_task: Optional[Dict[str, Any]], payload: Dict[str, Any]) -> None:
        if not _env_flag("EVIDENCE_CACHE_WRITE_ENABLED", True):
            return
        if not _env_flag("IQS_SEARCH_CACHE_ENABLED", True):
            record_cache_bypass("iqs_search_cache_disabled", search=True)
            return
        options = _as_dict(search_options)
        if str(options.get("disable_cache") or "").strip().lower() in {"1", "true", "yes", "on"}:
            record_cache_bypass("request_disable_cache", search=True)
            return
        if str(options.get("cache_ttl_seconds") or "").strip() == "0":
            record_cache_bypass("request_cache_ttl_zero", search=True)
            return
        if not isinstance(payload, dict):
            return
        errors = _as_list(payload.get("errors"))
        results = _as_list(payload.get("results"))
        if errors and not results and _search_errors_are_hard(errors):
            return
        key = self.search_key(query, search_options, search_task)
        negative = not bool(results)
        now = _now()
        source_type = "unknown"
        for item in results:
            raw = _as_dict(item)
            source_type = _source_type_from_text(raw, str(raw.get("source_type") or raw.get("type") or ""))
            if source_type != "unknown":
                break
        ttl = _ttl_seconds(source_type, search_options, negative=negative)
        stored_payload = self._trim_payload(payload)
        task = _as_dict(search_task) or _as_dict(options.get("search_task"))

        def _store() -> None:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO search_cache (
                        cache_key, query_key, query_text, engine_type, time_range, contents, phase,
                        proof_role, lane_targets_json, response_json, search_trace_json, negative,
                        created_at, expires_at, hit_count, last_hit_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL)
                    ON CONFLICT(cache_key) DO UPDATE SET
                        response_json=excluded.response_json,
                        search_trace_json=excluded.search_trace_json,
                        negative=excluded.negative,
                        expires_at=excluded.expires_at,
                        created_at=excluded.created_at
                    """,
                    (
                        key,
                        _normalize_query(query),
                        str(query or ""),
                        str(options.get("engineType") or ""),
                        str(options.get("timeRange") or ""),
                        str(options.get("contents") or ""),
                        str(options.get("phase") or ""),
                        str(task.get("proof_role") or options.get("proof_role") or ""),
                        _json_dumps(_as_list(task.get("lane_targets") or options.get("lane_targets"))),
                        _json_dumps(stored_payload),
                        _json_dumps(_as_list(payload.get("search_trace"))),
                        1 if negative else 0,
                        now,
                        now + ttl,
                    ),
                )
            record_cache_activity(search_negative_store=1 if negative else 0, search_store=0 if negative else 1)

        self._safe(_store, None)

    def _trim_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        max_chars = _env_int("EVIDENCE_CACHE_MAX_TEXT_CHARS_PER_RESULT", 12000, min_value=1000, max_value=120000)

        def _trim(value: Any) -> Any:
            if isinstance(value, dict):
                return {key: _trim(inner) for key, inner in value.items()}
            if isinstance(value, list):
                return [_trim(item) for item in value]
            if isinstance(value, str):
                return _compact_text(value, max_chars=max_chars)
            return value

        return _trim(copy.deepcopy(payload))

    def lookup_evidence(
        self,
        repair_task: Dict[str, Any],
        *,
        min_source_level: Sequence[str] | str = ("A", "B", "C"),
        required_fields: Optional[Sequence[str]] = None,
        max_hits: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not _env_flag("EVIDENCE_CACHE_EVIDENCE_READ_ENABLED", True):
            record_cache_bypass("persistent_evidence_read_disabled", evidence=True)
            return []
        task = _as_dict(repair_task)
        max_hits = max_hits or _env_int("EVIDENCE_CACHE_MAX_HITS_PER_TASK", 6, min_value=1, max_value=50)
        required = {str(item).strip().lower() for item in list(required_fields or _as_list(task.get("required_fields"))) if str(item).strip()}
        proof_role = str(task.get("proof_role") or task.get("evidence_type") or "").strip().lower()
        min_levels = [str(item).strip().upper() for item in (min_source_level if isinstance(min_source_level, (list, tuple, set)) else [min_source_level]) if str(item).strip()]
        minimum_rank = min((_SOURCE_LEVEL_RANK.get(level, 0) for level in min_levels), default=0)
        required_source_types = _task_source_type_hints(task)
        terms = self._task_terms(task)
        if not terms:
            return []
        now = _now()

        def _lookup() -> List[Dict[str, Any]]:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT * FROM evidence_cache
                    WHERE expires_at > ?
                    ORDER BY confidence_score DESC, created_at DESC
                    LIMIT 300
                    """,
                    (now,),
                ).fetchall()
                hits: List[Dict[str, Any]] = []
                hit_ids: List[str] = []
                for row in rows:
                    if _SOURCE_LEVEL_RANK.get(str(row["source_level"] or "").upper(), 0) < minimum_rank:
                        continue
                    row_role = str(row["proof_role"] or "").strip().lower()
                    if proof_role and row_role and proof_role != row_role:
                        if not (proof_role == "source_check" and row_role in {"source_check", "metric", "filing", "official_data", "support"}):
                            continue
                    raw = _json_loads(row["raw_json"], {})
                    if not isinstance(raw, dict):
                        continue
                    if not _row_matches_source_type(row, raw, required_source_types):
                        continue
                    if not self._required_fields_satisfied(row, raw, required):
                        continue
                    score = self._match_score(row, raw, terms)
                    if score <= 0:
                        continue
                    item = {
                        "evidence_id": row["evidence_id"],
                        "match_score": score,
                        "source_level": row["source_level"],
                        "source_type": row["source_type"],
                        "proof_role": row["proof_role"],
                        "allowed_use": row["allowed_use"],
                        "confidence_score": row["confidence_score"],
                        "fact_description": row["fact_description"],
                        "metric_name": row["metric_name"],
                        "value": row["value"],
                        "unit": row["unit"],
                        "period": row["period"],
                        "source_url": row["source_url"],
                        "source_domain": row["source_domain"],
                        "raw": raw,
                    }
                    hits.append(item)
                    hit_ids.append(str(row["evidence_id"]))
                    if len(hits) >= max_hits:
                        break
                if hit_ids:
                    conn.executemany(
                        "UPDATE evidence_cache SET hit_count = hit_count + 1, last_hit_at = ? WHERE evidence_id = ?",
                        [(now, evidence_id) for evidence_id in hit_ids],
                    )
                    record_cache_activity(evidence_hit=len(hit_ids))
                return hits

        return self._safe(_lookup, [])

    def _task_terms(self, task: Dict[str, Any]) -> List[str]:
        values: List[Any] = []
        for key in (
            "query",
            "suggested_query",
            "targets_gap",
            "hypothesis_statement",
            "dimension_name",
            "evidence_goal",
            "chapter_title",
            "proof_standard",
            "proof_role",
        ):
            values.append(task.get(key))
        values.extend(_as_list(task.get("topic_terms")))
        values.extend(_as_list(task.get("must_have_terms")))
        values.extend(_as_list(task.get("blocking_gaps")))
        tokens: List[str] = []
        seen = set()
        for value in values:
            for token in _tokenize(value):
                if token not in seen:
                    seen.add(token)
                    tokens.append(token)
        return tokens[:80]

    def _required_fields_satisfied(self, row: sqlite3.Row, raw: Dict[str, Any], required: set[str]) -> bool:
        if not required:
            return True
        source = _as_dict(raw.get("source"))
        source_url = str(row["source_url"] or raw.get("source_url") or source.get("url") or "").strip()
        source_name = str(source.get("title") or source.get("publisher") or raw.get("source_title") or raw.get("publisher") or "").strip()
        values = {
            "metric": row["metric_name"] or raw.get("metric") or raw.get("metric_name"),
            "period": row["period"] or raw.get("period") or source.get("date"),
            "unit": row["unit"] or raw.get("unit") or raw.get("numeric_unit"),
            "source": source_url or row["source_domain"] or source_name,
            "value": row["value"] or raw.get("value") or raw.get("numeric_value"),
        }
        for field in required:
            if field in {"source_url", "source_ref", "source"}:
                if not values["source"]:
                    return False
                continue
            if field in {"metric", "indicator"} and not values["metric"]:
                return False
            if field == "period" and not values["period"]:
                return False
            if field == "unit" and not (values["unit"] or re.search(r"%|pct|元|美元|台|件|套|million|billion|bn", str(values["value"] or ""), re.I)):
                return False
            if field in {"value", "number"} and not values["value"]:
                return False
        return True

    def _match_score(self, row: sqlite3.Row, raw: Dict[str, Any], terms: Sequence[str]) -> float:
        query_terms = set(str(item).lower() for item in _json_loads(row["query_terms_json"], []) if str(item).strip())
        distinctive_terms = _distinctive_cache_terms(terms)
        distinctive_hits = 0
        haystack = " ".join(
            [
                str(row["topic_key"] or ""),
                str(row["fact_description"] or ""),
                str(row["metric_name"] or ""),
                str(row["source_domain"] or ""),
                str(row["source_type"] or ""),
                str(row["proof_role"] or ""),
                str(raw.get("content") or raw.get("fact") or raw.get("clean_fact") or ""),
                str(_as_dict(raw.get("source")).get("title") or ""),
            ]
        ).lower()
        score = 0.0
        for term in terms:
            term = str(term or "").lower()
            if not term:
                continue
            if term in query_terms:
                score += 2.0
                if term in distinctive_terms:
                    distinctive_hits += 1
            elif term in haystack:
                score += 1.0
                if term in distinctive_terms:
                    distinctive_hits += 1
        if distinctive_terms and distinctive_hits <= 0:
            return 0.0
        return score

    def store_evidence_from_package(
        self,
        *,
        query: str,
        evidence_package: Dict[str, Any],
        report_id: str = "",
        run_id: str = "",
    ) -> Dict[str, Any]:
        if not _env_flag("EVIDENCE_CACHE_WRITE_ENABLED", True):
            return {"stored_count": 0, "skipped_count": 0, "enabled": False}
        package = _as_dict(evidence_package)
        items = self._evidence_items_from_package(package)
        now = _now()
        stored = 0
        skipped = 0

        def _store() -> Dict[str, Any]:
            nonlocal stored, skipped
            with self._connect() as conn:
                for item in items:
                    record = self._record_from_evidence(query, item)
                    if not record:
                        skipped += 1
                        continue
                    conn.execute(
                        """
                        INSERT INTO evidence_cache (
                            evidence_id, topic_key, query_terms_json, source_url, source_domain,
                            source_level, source_type, proof_role, allowed_use, confidence_score,
                            fact_description, metric_name, value, unit, period, raw_json,
                            created_at, expires_at, hit_count, last_hit_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL)
                        ON CONFLICT(evidence_id) DO UPDATE SET
                            topic_key=excluded.topic_key,
                            query_terms_json=excluded.query_terms_json,
                            source_level=excluded.source_level,
                            source_type=excluded.source_type,
                            proof_role=excluded.proof_role,
                            allowed_use=excluded.allowed_use,
                            confidence_score=excluded.confidence_score,
                            fact_description=excluded.fact_description,
                            metric_name=excluded.metric_name,
                            value=excluded.value,
                            unit=excluded.unit,
                            period=excluded.period,
                            raw_json=excluded.raw_json,
                            expires_at=excluded.expires_at,
                            created_at=excluded.created_at
                        """,
                        (
                            record["evidence_id"],
                            record["topic_key"],
                            _json_dumps(record["query_terms"]),
                            record["source_url"],
                            record["source_domain"],
                            record["source_level"],
                            record["source_type"],
                            record["proof_role"],
                            record["allowed_use"],
                            record["confidence_score"],
                            record["fact_description"],
                            record["metric_name"],
                            record["value"],
                            record["unit"],
                            record["period"],
                            _json_dumps(record["raw"]),
                            now,
                            now + _ttl_seconds(record["source_type"], record["raw"]),
                        ),
                    )
                    conn.execute(
                        """
                        INSERT INTO evidence_lineage (
                            evidence_id, search_key, report_id, run_id, query, task_id, gap_id, loop_name, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            record["evidence_id"],
                            "",
                            report_id,
                            run_id,
                            query,
                            str(record["raw"].get("task_id") or ""),
                            str(record["raw"].get("gap_id") or _as_dict(record["raw"].get("search_task")).get("gap_id") or ""),
                            str(record["raw"].get("loop_name") or _as_dict(record["raw"].get("search_task")).get("loop_name") or ""),
                            now,
                        ),
                    )
                    stored += 1
            record_cache_activity(evidence_store=stored)
            return {"stored_count": stored, "skipped_count": skipped, "enabled": True}

        return self._safe(_store, {"stored_count": 0, "skipped_count": len(items), "enabled": False})

    def _evidence_items_from_package(self, package: Dict[str, Any]) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        for key in ("analysis_ready_evidence", "clean_evidence_list", "core_evidence", "supporting_evidence", "clue_evidence"):
            candidates.extend([dict(item) for item in _as_list(package.get(key)) if isinstance(item, dict)])
        for point in _as_list(package.get("raw_data_points")):
            if isinstance(point, dict):
                candidates.append(dict(point))
        for item in _as_list(_as_dict(package.get("summary")).get("evidence_samples")):
            if isinstance(item, dict):
                candidates.append(dict(item))
        seen = set()
        result: List[Dict[str, Any]] = []
        for item in candidates:
            key = _hash_payload(
                {
                    "url": str(_as_dict(item.get("source")).get("url") or item.get("source_url") or item.get("url") or ""),
                    "fact": str(item.get("fact") or item.get("clean_fact") or item.get("content") or item.get("fact_description") or "")[:240],
                    "metric": str(item.get("metric") or item.get("metric_name") or ""),
                    "period": str(item.get("period") or _as_dict(item.get("source")).get("date") or ""),
                }
            )
            if key in seen:
                continue
            seen.add(key)
            result.append(item)
        return result

    def _record_from_evidence(self, query: str, item: Dict[str, Any]) -> Dict[str, Any]:
        source = dict(_as_dict(item.get("source")))
        source_url = str(source.get("url") or item.get("source_url") or item.get("url") or "").strip()
        source_title = str(source.get("title") or item.get("source_title") or item.get("title") or "").strip()
        source_publisher = str(source.get("publisher") or item.get("publisher") or "").strip()
        if source_url and not source.get("url"):
            source["url"] = source_url
        if source_title and not source.get("title"):
            source["title"] = source_title
        if source_publisher and not source.get("publisher"):
            source["publisher"] = source_publisher
        source_type = _source_type_from_text(source, str(item.get("source_type") or ""))
        source_level = _source_level(source_type, item.get("source_level"))
        role = str(item.get("evidence_role") or item.get("role") or "").strip().lower()
        allowed_use = str(item.get("allowed_use") or _as_dict(item.get("evidence_card")).get("allowed_use") or "").strip().lower()
        semantic_status = str(item.get("semantic_status") or "").strip().lower()
        if semantic_status in _REJECTED_ROLES or role in _REJECTED_ROLES:
            return {}
        if source_level == "D" or allowed_use in _APPENDIX_ONLY_USES:
            return {}
        fact = str(item.get("fact") or item.get("clean_fact") or item.get("content") or item.get("fact_description") or "").strip()
        metric = str(item.get("metric") or item.get("metric_name") or item.get("indicator") or "").strip()
        value = str(item.get("value") or item.get("numeric_value") or "").strip()
        if not (source_url or source_title or source_publisher):
            return {}
        if not (fact or metric or value):
            return {}
        proof_role = str(item.get("proof_role") or item.get("evidence_type") or "").strip().lower() or ("metric" if metric and value else "source_check")
        period = str(item.get("period") or source.get("date") or item.get("date") or "").strip()
        unit = str(item.get("unit") or item.get("numeric_unit") or "").strip()
        domain = _domain(source_url)
        terms: List[str] = []
        for value_for_terms in (
            query,
            item.get("query"),
            fact,
            metric,
            source.get("title"),
            source_url,
            item.get("dimension_name"),
            item.get("evidence_goal"),
            proof_role,
        ):
            terms.extend(_tokenize(value_for_terms))
        terms = list(dict.fromkeys(terms))[:80]
        evidence_id = "ev:" + _hash_payload(
            {
                "url": source_url,
                "fact": fact[:360],
                "metric": metric,
                "period": period,
                "value": value,
            }
        )
        raw = copy.deepcopy(item)
        raw.setdefault("source_level", source_level)
        raw.setdefault("source_type", source_type)
        raw.setdefault("proof_role", proof_role)
        raw["source"] = source or {"url": source_url, "title": source_title, "publisher": source_publisher, "source_type": source_type}
        return {
            "evidence_id": evidence_id,
            "topic_key": _normalize_query(" ".join(terms[:20])),
            "query_terms": terms,
            "source_url": source_url,
            "source_domain": domain,
            "source_level": source_level,
            "source_type": source_type,
            "proof_role": proof_role,
            "allowed_use": allowed_use or ("core_claim" if source_level in {"A", "B"} else "directional_signal"),
            "confidence_score": float(item.get("confidence") or item.get("confidence_score") or 0.0),
            "fact_description": _compact_text(fact, max_chars=1200),
            "metric_name": metric,
            "value": value,
            "unit": unit,
            "period": period,
            "raw": raw,
        }

    def prune_expired(self) -> Dict[str, int]:
        now = _now()

        def _prune() -> Dict[str, int]:
            with self._connect() as conn:
                search_deleted = conn.execute("DELETE FROM search_cache WHERE expires_at <= ?", (now,)).rowcount
                evidence_deleted = conn.execute("DELETE FROM evidence_cache WHERE expires_at <= ?", (now,)).rowcount
                return {"search_deleted": int(search_deleted or 0), "evidence_deleted": int(evidence_deleted or 0)}

        return self._safe(_prune, {"search_deleted": 0, "evidence_deleted": 0})

    def stats(self) -> Dict[str, Any]:
        def _stats() -> Dict[str, Any]:
            with self._connect() as conn:
                search_count = conn.execute("SELECT COUNT(*) FROM search_cache WHERE expires_at > ?", (_now(),)).fetchone()[0]
                evidence_count = conn.execute("SELECT COUNT(*) FROM evidence_cache WHERE expires_at > ?", (_now(),)).fetchone()[0]
                negative_count = conn.execute("SELECT COUNT(*) FROM search_cache WHERE negative = 1 AND expires_at > ?", (_now(),)).fetchone()[0]
                return {
                    "enabled": True,
                    "path": str(self.path),
                    "search_count": int(search_count),
                    "evidence_count": int(evidence_count),
                    "negative_count": int(negative_count),
                }

        if not self.enabled():
            return {"enabled": False, "path": str(self.path)}
        return self._safe(_stats, {"enabled": False, "path": str(self.path)})


def get_evidence_cache() -> EvidenceCache:
    global _CACHE
    with _CACHE_LOCK:
        if _CACHE is None or _CACHE.path != _cache_path():
            _CACHE = EvidenceCache(_cache_path())
        return _CACHE


def lookup_search(query: str, search_options: Dict[str, Any], search_task: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    return get_evidence_cache().lookup_search(query, search_options, search_task)


def store_search(query: str, search_options: Dict[str, Any], search_task: Optional[Dict[str, Any]], payload: Dict[str, Any]) -> None:
    get_evidence_cache().store_search(query, search_options, search_task, payload)


def lookup_evidence(
    repair_task: Dict[str, Any],
    *,
    min_source_level: Sequence[str] | str = ("A", "B", "C"),
    required_fields: Optional[Sequence[str]] = None,
    max_hits: Optional[int] = None,
) -> List[Dict[str, Any]]:
    return get_evidence_cache().lookup_evidence(
        repair_task,
        min_source_level=min_source_level,
        required_fields=required_fields,
        max_hits=max_hits,
    )


def store_evidence_from_package(
    *,
    query: str,
    evidence_package: Dict[str, Any],
    report_id: str = "",
    run_id: str = "",
) -> Dict[str, Any]:
    return get_evidence_cache().store_evidence_from_package(
        query=query,
        evidence_package=evidence_package,
        report_id=report_id,
        run_id=run_id,
    )
