from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import urlparse

from ..config.search_config import build_llm_config_for_task, build_llm_config_from_profile
from ..search.memory import call_openai_compatible_json, llm_config_is_ready, normalize_llm_config
from rag_pipeline.contracts.public_text_guard import public_text_quality
from rag_pipeline.contracts.quality_gate_policy import public_signal_mode
from rag_pipeline.contracts.repair_dispatcher import rejected_span_repair_summary

try:  # Observability must never make extraction unavailable.
    from rag_pipeline.observability.probe_api import emit_transform as _probe_emit_transform
    from rag_pipeline.observability.probe_context import current_probe_context_from_env as _current_probe_context_from_env
    from rag_pipeline.observability.data_root_hygiene import inspect_source_hygiene as _inspect_source_hygiene
except Exception:  # pragma: no cover - optional diagnostics only.
    _probe_emit_transform = None
    _current_probe_context_from_env = None
    _inspect_source_hygiene = None


SCHEMA_VERSION = "readpage_fact_card_v2"

_RUN_BUDGET_STATE: Dict[str, int] = {}

INTERNAL_OR_CLAIM_PATTERNS = re.compile(
    "|".join(
        [
            r"\u8bc1\u636e\u4e0d\u8db3",
            r"\u5efa\u8bae\u8865\u8bc1",
            r"\u540e\u7eed\u5206\u6790\u9700\u8981",
            r"\u672c\u7ae0\u5e94\u5199\u6210",
            r"\u6b63\u6587\u5e94\u4ee5",
            r"\u65b9\u5411\u6027\u89c2\u5bdf",
            r"needs?\s+more\s+evidence",
            r"insufficient\s+evidence",
        ]
    ),
    re.I,
)

NAVIGATION_PATTERNS = re.compile(
    r"(skip to (?:content|main content)|login|sign in|privacy policy|terms of use|cookie|subscribe|"
    r"\u9996\u9875|\u767b\u5f55|\u6ce8\u518c|\u9690\u79c1|\u7248\u6743|\u4e0b\u8f7d|\u539f\u6587\u94fe\u63a5|\u76ee\u5f55|"
    r"\u6570\u636e\u4e2d\u5fc3|\u5168\u7403\u8d22\u7ecf\u5feb\u8baf|\u8d22\u7ecf\u5feb\u8baf|"
    r"\u884c\u60c5\u4e2d\u5fc3|Choice\s*\u6570\u636e|\u4e1c\u65b9\u8d22\u5bcc|\u81ea\u9009\u80a1|\u80a1\u5427)",
    re.I,
)
MARKDOWN_LINK_CLUSTER_RE = re.compile(
    r"(?:\[[^\]]{0,80}\]\((?:https?:)?//[^)]+|/[^)]+\).*){2,}"
    r"|/newstatic/images/logo|(?:^|/|\\)logo\.(?:gif|png|jpg|jpeg|webp|svg)\b",
    re.I,
)

LOW_QUALITY_HOST_RE = re.compile(
    r"(?:twitter|x|instagram|facebook|baike|baijiahao|csdn|cnblogs|juejin|wenku|doc88|book118)\.",
    re.I,
)

# HTTP error pages / bot walls / sales chrome that scraping captures verbatim.
# These are unambiguous non-facts; length-independent because error sentences
# (e.g. "If the problem continues, contact the site owner.") are long enough to
# slip past the navigation filter and end up as evidence in the report.
ERROR_PAGE_PATTERNS = re.compile(
    r"this page (?:isn['’]?t|is not) working"
    r"|if the problem continues"
    r"|contact the site owner"
    r"|http error \d"
    r"|took too long to respond"
    r"|your connection is not private"
    r"|attackers might be trying to steal"
    r"|can['’]?t be reached"
    r"|verify (?:you are|you['’]?re) (?:a )?human"
    r"|checking your browser before"
    r"|please enable javascript"
    r"|(?:book|request) a demo"
    r"|(?:accept all|we use) cookies",
    re.I,
)

# Structural markup scraping captured as a "fact": headings, table cells/headers,
# and financial-report table markers (e.g. ``<h5>Company Name</h5>`` or
# ``<th>[Table_StockNameRptType] ...</th>``) are titles/structure, not verifiable
# facts. The LLM analyst correctly abstains when a chapter's evidence is mostly
# these, which is exactly why most chapters produced zero claims — so reject them
# at extraction before they ever reach the analysis stage.
STRUCTURAL_MARKUP_RE = re.compile(
    r"^\s*<\s*/?\s*(?:h[1-6]|th|td|tr|thead|tbody|table|title|caption|nav|ul|ol|li)\b"
    r"|\[Table_[A-Za-z]"
    r"|</?(?:th|td|tr|thead|tbody|table)\b",
    re.I,
)

# Institutional "how we publish" boilerplate that scraping captures from agency
# pages (e.g. "国家统计局通过官方网站、数据发布库、《中国统计年鉴》、两微一端等渠道
# 发布统计数据"). It is off-topic for an industry report yet binds to chapters and
# gets glued into paragraphs with mismatched citations -> FinalAudit "引用来源与内容
# 不匹配" fatal. These markers are specific enough to be near-zero false-positive.
INSTITUTIONAL_BOILERPLATE_RE = re.compile(
    r"两微一端"
    r"|数据发布库"
    r"|统计出版物"
    r"|《?中国统计年鉴》?"
    r"|通过[^。]{0,40}(?:官方网站|新闻发布会)[^。]{0,40}发布统计数据"
    r"|满足不同用户群体获取统计数据",
)


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, min_value: int = 0, max_value: int = 100_000) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(min_value, min(max_value, value))


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if value in (None, ""):
        return []
    return [value]


def _compact_text(value: Any, max_chars: int = 6000) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "..."


def _page_text(page: Dict[str, Any]) -> str:
    return str(
        page.get("mainText")
        or page.get("markdown")
        or page.get("content")
        or page.get("text")
        or page.get("summary")
        or page.get("snippet")
        or ""
    ).strip()


def _source_url(page: Dict[str, Any], fallback: str = "") -> str:
    return str(page.get("url") or page.get("source_url") or page.get("link") or fallback or "").strip()


def _source_ref(page: Dict[str, Any], fallback: str = "") -> str:
    for key in ("source_id", "source_ref", "id"):
        if key in page and page.get(key) is not None:
            value = str(page.get(key)).strip()
            if value:
                return value
    return str(fallback or "").strip()


_BUDGET_STATE_MAX_KEYS = 32


def _budget_key() -> str:
    return str(os.getenv("REPORT_STAGE_SNAPSHOT_RUN_ID") or os.getenv("REPORT_RUN_ID") or "default").strip() or "default"


def _budget_limit() -> int:
    return _env_int("READPAGE_FACT_EXTRACTOR_MAX_CALLS_PER_REPORT", 40, min_value=0, max_value=10_000)


def _budget_used() -> int:
    return int(_RUN_BUDGET_STATE.get(_budget_key(), 0))


def reset_budget(run_id: Optional[str] = None) -> None:
    """Drop the in-process budget counter for a given run_id (default: current).

    Long-running services keep `_RUN_BUDGET_STATE` alive across reports.
    Without a reset hook the dict grows forever as new run_ids arrive.
    `brain_agent` should call this when a report finishes.
    """

    key = (run_id or _budget_key()).strip() or _budget_key()
    _RUN_BUDGET_STATE.pop(key, None)


def _emit_fact_extractor_probe(
    result: Dict[str, Any],
    *,
    task: Dict[str, Any],
    selected_page_count: int,
    raw_page_count: Optional[int] = None,
    page_results: Optional[Sequence[Dict[str, Any]]] = None,
) -> None:
    """Emit a diagnostic-only runtime probe for fact extraction.

    This deliberately does not return or mutate pipeline data. The probe is a
    sidecar health signal for tracing where evidence is lost.
    """

    if _probe_emit_transform is None or _current_probe_context_from_env is None:
        return
    try:
        probe = _current_probe_context_from_env()
        if probe is None:
            return
        fact_cards = [item for item in _as_list(result.get("fact_cards")) if isinstance(item, dict)]
        requirement_filled = sum(1 for item in fact_cards if str(item.get("requirement_id") or "").strip())
        source_filled = sum(1 for item in fact_cards if str(item.get("source_ref") or item.get("source_url") or "").strip())
        search_task_id = str(task.get("task_id") or task.get("search_task_id") or "").strip()
        status = str(result.get("status") or "")
        source_hygiene = _inspect_source_hygiene(page_results or _as_list(result.get("source_pages") or result.get("page_results") or task.get("page_results"))) if _inspect_source_hygiene else {}
        source_dirty_count = int(source_hygiene.get("dirty_item_count") or 0) if isinstance(source_hygiene, dict) else 0
        source_reason_counts = dict(source_hygiene.get("reason_counts") or {}) if isinstance(source_hygiene, dict) else {}
        _probe_emit_transform(
            probe,
            stage="fact_extractor",
            module="readpage_fact_extractor_agent",
            input_count=selected_page_count if raw_page_count is None else raw_page_count,
            output_count=int(result.get("fact_card_count") or len(fact_cards)),
            drop_count=int(result.get("rejected_span_count") or 0) + int(result.get("llm_error_count") or 0),
            status="ok" if status in {"success", "partial_budget_exhausted"} else "warning",
            reason_counts={
                "rejected_span": int(result.get("rejected_span_count") or 0),
                "invalid_metric": int(result.get("invalid_metric_count") or 0),
                "llm_error": int(result.get("llm_error_count") or 0),
                "budget_exhausted": 1 if result.get("budget_exhausted") else 0,
                "fallback_used": 1 if result.get("fallback_used") else 0,
                "no_readpage_text": 1 if status == "no_readpage_text" else 0,
                "source_dirty_item": source_dirty_count,
                **{f"source_{key}": value for key, value in source_reason_counts.items()},
            },
            id_coverage={
                "requirement_id": (requirement_filled / len(fact_cards)) if fact_cards else (1.0 if str(task.get("requirement_id") or "").strip() else 0.0),
                "search_task_id": 1.0 if search_task_id else 0.0,
                "source_id": (source_filled / len(fact_cards)) if fact_cards else 0.0,
            },
            cache={"cache_hit_count": int(result.get("cache_hit_count") or 0)},
            metrics={
                "attempted": int(result.get("attempted") or 0),
                "selected_page_count": max(0, int(selected_page_count or 0)),
                "success_count": int(result.get("success_count") or 0),
                "budget_used": int(result.get("budget_used") or 0),
                "budget_limit": int(result.get("budget_limit") or 0),
            },
            diagnostics={
                "requirement_id": task.get("requirement_id") or "",
                "search_task_id": search_task_id,
                "gap_id": task.get("gap_id") or "",
                "proof_role": task.get("proof_role") or task.get("evidence_type") or "",
                "fallback_used": bool(result.get("fallback_used")),
                "budget_exhausted": bool(result.get("budget_exhausted")),
                "status": status,
                "source_root_hygiene": source_hygiene,
            },
        )
    except Exception:
        return


def _evict_old_budget_keys() -> None:
    """Prune the in-memory budget map when too many run_ids have accumulated.

    Keeps the most recent N entries (insertion order) — guards against the
    map growing without bound when callers forget to call `reset_budget`.
    """

    if len(_RUN_BUDGET_STATE) <= _BUDGET_STATE_MAX_KEYS:
        return
    overflow = len(_RUN_BUDGET_STATE) - _BUDGET_STATE_MAX_KEYS
    for stale_key in list(_RUN_BUDGET_STATE.keys())[:overflow]:
        _RUN_BUDGET_STATE.pop(stale_key, None)


def _try_consume_budget() -> tuple[bool, bool]:
    limit = _budget_limit()
    if limit <= 0:
        # `limit == 0` means "feature disabled", not "out of budget".
        # Signal not-consumed + not-exhausted so callers fall back to the
        # deterministic path without flagging budget_exhausted in diagnostics.
        return False, False
    key = _budget_key()
    used = int(_RUN_BUDGET_STATE.get(key, 0))
    if used >= limit:
        return False, True
    _RUN_BUDGET_STATE[key] = used + 1
    _evict_old_budget_keys()
    return True, False


def _source_level(page: Dict[str, Any]) -> str:
    value = str(page.get("source_level") or page.get("credibility") or page.get("source_grade") or "").strip().upper()
    return value if value in {"A", "B", "C", "D"} else "C"


def _verification_status(page: Dict[str, Any]) -> str:
    explicit = str(page.get("source_verification_status") or page.get("verification_status") or "").strip().lower()
    if explicit in {"search_result_only", "readpage_verified", "document_verified", "inaccessible"}:
        return explicit
    url = _source_url(page)
    if re.search(r"\.pdf(?:$|\?)|annual-report|filing|disclosure|announcement", url, flags=re.I):
        return "document_verified"
    return "readpage_verified" if _page_text(page) else "search_result_only"


def _hash_text(value: str) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8", errors="ignore")).hexdigest()


def _cache_root() -> Path:
    raw = os.getenv("READPAGE_FACT_EXTRACTOR_CACHE_PATH", "output/cache/readpage_fact_extractor").strip()
    path = Path(raw)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _cache_key(page: Dict[str, Any], *, proof_role: str, query: str) -> str:
    url = _source_url(page)
    text_hash = _hash_text(_page_text(page))[:20]
    role = re.sub(r"[^A-Za-z0-9_-]+", "_", str(proof_role or "unknown"))[:40]
    query_hash = _hash_text(query)[:12]
    return _hash_text(f"{SCHEMA_VERSION}|{url}|{text_hash}|{role}|{query_hash}")


def _chunk_enabled() -> bool:
    return _env_flag("READPAGE_FACT_EXTRACTOR_CHUNK_ENABLED", True)


def _chunk_trigger_chars() -> int:
    return _env_int("READPAGE_FACT_EXTRACTOR_CHUNK_TRIGGER_CHARS", 1800, min_value=500, max_value=30_000)


def _chunk_max_chars() -> int:
    return _env_int("READPAGE_FACT_EXTRACTOR_CHUNK_MAX_CHARS", 1800, min_value=500, max_value=12_000)


def _chunk_max_fact_cards() -> int:
    return _env_int("READPAGE_FACT_EXTRACTOR_CHUNK_MAX_FACT_CARDS", 3, min_value=1, max_value=12)


def _chunk_max_fact_chars() -> int:
    return _env_int("READPAGE_FACT_EXTRACTOR_CHUNK_MAX_FACT_CHARS", 220, min_value=80, max_value=600)


def _max_chunks_per_page() -> int:
    return _env_int("READPAGE_FACT_EXTRACTOR_MAX_CHUNKS_PER_PAGE", 3, min_value=1, max_value=12)


def _chunk_cache_key(page: Dict[str, Any], chunk: Dict[str, Any], *, proof_role: str, query: str) -> str:
    url = _source_url(page)
    page_hash = _hash_text(_page_text(page))[:20]
    chunk_hash = _hash_text(str(chunk.get("chunk_text") or ""))[:20]
    role = re.sub(r"[^A-Za-z0-9_-]+", "_", str(proof_role or "unknown"))[:40]
    query_hash = _hash_text(query)[:12]
    chunk_index = str(chunk.get("chunk_index") or "")
    return _hash_text(f"{SCHEMA_VERSION}|chunk|{url}|{page_hash}|{chunk_hash}|{chunk_index}|{role}|{query_hash}")


def _load_cache(key: str) -> Optional[Dict[str, Any]]:
    if not _env_flag("READPAGE_FACT_EXTRACTOR_CACHE_ENABLED", True):
        return None
    path = _cache_root() / f"{key}.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _store_cache(key: str, payload: Dict[str, Any]) -> None:
    if not _env_flag("READPAGE_FACT_EXTRACTOR_CACHE_ENABLED", True):
        return
    try:
        root = _cache_root()
        root.mkdir(parents=True, exist_ok=True)
        (root / f"{key}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        return


def _build_llm_config() -> Dict[str, Any]:
    profile = str(os.getenv("READPAGE_FACT_EXTRACTOR_MODEL_PROFILE") or "").strip()
    if profile:
        config = build_llm_config_from_profile(
            profile,
            default_timeout=float(_env_int("READPAGE_FACT_EXTRACTOR_TIMEOUT_SECONDS", 90, min_value=10, max_value=300)),
        )
    else:
        config = build_llm_config_for_task("readpage_fact_extractor")
    config = dict(config or {})
    config["timeout"] = float(_env_int("READPAGE_FACT_EXTRACTOR_TIMEOUT_SECONDS", int(float(config.get("timeout") or 90)), min_value=10, max_value=300))
    config["task_name"] = "readpage_fact_extractor"
    return config


def _system_prompt() -> str:
    return (
        "你负责从已读取网页正文中抽取结构化公开 fact cards，用于中文行业研究。"
        "只能使用输入的 page_text 和 source 元数据，不得使用搜索摘要、缓存摘要或模型常识。"
        "不要写报告，不要给补证建议，不要生成页面原文不支持的结论；只返回严格 JSON。"
        "每张 fact card 必须包含 subject、action_or_signal、variable、distilled_fact、fact_type、"
        "source_url 或 source_ref、source_level、source_verification_status、proof_role、block_affinity、"
        "claim_strength_hint。metric 类型必须包含 metric、value、unit、period 或 time_or_scope、source。"
        "如果 search_task.required_fields 中任一字段无法从当前 page_text 和 source 元数据填出，应把该片段放入 rejected_spans。"
        "counter 证据只能作为反证/风险证据，不能用于支撑正向强结论。"
        "拒绝导航、登录页、下载提示、SEO 文案、搜索结果摘要、HTTP 错误页、没有可追溯事实的营销话术，"
        "以及“证据不足、建议补证”等诊断话术。"
        "如果输入包含 chunk 对象，fact_cards 数量不得超过 chunk.max_fact_cards，distilled_fact 长度不得超过 chunk.max_fact_chars；"
        "优先抽取完整指标、订单/合同、客户部署、政策目标和具体风险约束。"
        "public-signal 模式下，社交平台、百科、论坛内容只能抽为可追溯的方向性信号，不能作为强事实。"
    )


def _user_payload(*, query: str, page: Dict[str, Any], search_task: Dict[str, Any], max_chars: int) -> Dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "query": query,
        "search_task": {
            "task_id": search_task.get("task_id"),
            "search_task_id": search_task.get("search_task_id") or search_task.get("task_id"),
            "requirement_id": search_task.get("requirement_id"),
            "gap_id": search_task.get("gap_id"),
            "chapter_id": search_task.get("chapter_id") or search_task.get("dimension_id"),
            "section_id": search_task.get("section_id"),
            "dimension_id": search_task.get("dimension_id"),
            "proof_role": search_task.get("proof_role") or search_task.get("evidence_type"),
            "evidence_goal": search_task.get("evidence_goal"),
            "required_fields": _as_list(search_task.get("required_fields")),
            "required_source_level": _as_list(search_task.get("required_source_level") or search_task.get("min_source_level")),
            "success_criteria": search_task.get("success_criteria"),
            "reject_if": _as_list(search_task.get("reject_if")),
            "must_have_terms": _as_list(search_task.get("must_have_terms")),
            "forbidden_terms": _as_list(search_task.get("forbidden_terms")),
        },
        "source": {
            "source_ref": _source_ref(page),
            "title": page.get("title"),
            "url": _source_url(page),
            "publisher": page.get("publisher") or page.get("source") or page.get("site"),
            "date": page.get("date") or page.get("publishedTime"),
            "source_level": _source_level(page),
            "source_verification_status": _verification_status(page),
        },
        "page_text": _compact_text(_page_text(page), max_chars=max_chars),
    }
    if page.get("_chunk_index") is not None:
        payload["chunk"] = {
            "chunk_index": page.get("_chunk_index"),
            "chunk_heading": page.get("_chunk_heading") or "",
            "chunk_text_chars": len(_page_text(page)),
            "max_fact_cards": _chunk_max_fact_cards(),
            "max_fact_chars": _chunk_max_fact_chars(),
            "max_rejected_spans": 3,
            "instruction": "只从当前 chunk 抽取事实，不得根据被省略的页面部分推断",
        }
    return payload


def _normalize_affinity(value: Any, fact_type: str, proof_role: str) -> List[str]:
    values = [str(item).strip() for item in _as_list(value) if str(item or "").strip()]
    if not values:
        role = f"{fact_type} {proof_role}".lower()
        if "metric" in role:
            values = ["metric_reconciliation"]
        elif "technology" in role or "standard" in role:
            values = ["technology_maturity"]
        elif "risk" in role or "counter" in role:
            values = ["risk_trigger"]
        elif "case" in role or "customer" in role:
            values = ["case_comparison", "customer_painpoint_matrix"]
        else:
            values = ["evidence_matrix"]
    seen: List[str] = []
    for item in values:
        if item and item not in seen:
            seen.append(item)
    return seen


def _looks_bad_text(text: str) -> bool:
    value = str(text or "").strip()
    if len(value) < 18:
        return True
    if INTERNAL_OR_CLAIM_PATTERNS.search(value):
        return True
    if ERROR_PAGE_PATTERNS.search(value):
        return True
    if STRUCTURAL_MARKUP_RE.search(value):
        return True
    if MARKDOWN_LINK_CLUSTER_RE.search(value):
        return True
    if INSTITUTIONAL_BOILERPLATE_RE.search(value):
        return True
    if NAVIGATION_PATTERNS.search(value) and (len(value) < 260 or MARKDOWN_LINK_CLUSTER_RE.search(value)):
        return True
    if re.search(r"^\s*(?:https?://|www\.)", value, flags=re.I):
        return True
    if re.search(r"^\s*[-*]?\s*\[[^\]]+\]\((?:https?:)?//[^)]+\)\s*$", value, flags=re.I):
        return True
    return False


FALLBACK_METRIC_SIGNAL_RE = re.compile(
    r"\d+(?:,\d{3})*(?:\.\d+)?\s*(?:"
    r"%|percent|percentage points?|"
    r"billion yuan|million yuan|trillion yuan|billion usd|million usd|usd|"
    r"\u4e07\u4ebf\u5143|\u4ebf\u5143|\u4e07\u5143|\u4ebf\u7f8e\u5143|\u4e07\u7f8e\u5143|"
    r"\u4e07\u53f0|\u53f0|\u5bb6|\u8d77|\u4e2a"
    r")",
    re.I,
)


FALLBACK_TOPIC_SIGNAL_RE = re.compile(
    r"market size|market scale|forecast|predict|growth|financing|deployment|policy|"
    r"\u5e02\u573a\u89c4\u6a21|\u9884\u6d4b|\u540c\u6bd4|\u589e\u957f|\u878d\u8d44|"
    r"\u653f\u7b56|\u884c\u52a8\u65b9\u6848|\u5e94\u7528\u573a\u666f|\u4ea7\u4e1a\u94fe|"
    r"\u4e0a\u5e02\u4f01\u4e1a|\u4ea7\u503c|\u8ba2\u5355",
    re.I,
)


FALLBACK_TABLE_OR_HEADER_RE = re.compile(
    r"(?:\|[-:\s]{2,}\|)|"
    r"(?:证券研究报告|行业研究[·・]|核心观点|优于大市|维持评级|证券分析师|"
    r"投资评级|评级说明|分析师声明|免责声明)",
    re.I,
)


def _looks_bad_fallback_candidate(text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return True
    pipe_count = value.count("|")
    if pipe_count >= 4:
        return True
    if FALLBACK_TABLE_OR_HEADER_RE.search(value):
        return True
    if len(value) > 180 and pipe_count >= 2:
        return True
    compact = re.sub(r"\s+", "", value)
    if compact and compact.count("?") >= max(6, len(compact) // 4):
        return True
    return False


def _fallback_candidate_score(text: str, *, proof_role: str, position: int) -> int:
    score = 0
    if FALLBACK_METRIC_SIGNAL_RE.search(text):
        score += 8
    if FALLBACK_TOPIC_SIGNAL_RE.search(text):
        score += 4
    if str(proof_role or "").strip().lower() == "metric" and FALLBACK_METRIC_SIGNAL_RE.search(text):
        score += 4
    if re.search(r"(?:19|20)\d{2}", text):
        score += 2
    length = len(text)
    if 50 <= length <= 260:
        score += 2
    if re.search(r"!\[[^\]]*\]\(|\[[^\]]+\]\(https?://|https?://", text, flags=re.I):
        score -= 6
    if _looks_bad_fallback_candidate(text):
        score -= 20
    if NAVIGATION_PATTERNS.search(text):
        score -= 4
    if MARKDOWN_LINK_CLUSTER_RE.search(text):
        score -= 20
    return score - min(position, 20)


def _fallback_candidate_sentences(text: str, *, proof_role: str) -> List[str]:
    candidates = re.split(r"(?<=[。！？!?])\s*|(?<=[銆傦紒锛?!?])\s+|\n+", text)
    scored: List[tuple[int, int, str]] = []
    for position, sentence in enumerate(candidates):
        cleaned = _compact_text(sentence, max_chars=220)
        if _looks_bad_fallback_candidate(cleaned):
            continue
        if _looks_bad_text(cleaned):
            continue
        if public_text_quality(cleaned).get("severity") == "reject":
            continue
        scored.append((_fallback_candidate_score(cleaned, proof_role=proof_role, position=position), position, cleaned))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in scored]


_CHUNK_HEADING_RE = re.compile(r"^\s*(?:#{1,6}\s+.+|\*\*[^*\n]{2,120}\*\*)\s*$")


def _clean_chunk_heading(value: str) -> str:
    heading = re.sub(r"^\s*#{1,6}\s*", "", str(value or "")).strip()
    heading = re.sub(r"^\*\*|\*\*$", "", heading).strip()
    return _compact_text(heading, max_chars=120)


def _flush_chunk(chunks: List[Dict[str, Any]], *, lines: List[str], heading: str, position: int, proof_role: str) -> None:
    text = _compact_text("\n".join(line for line in lines if str(line or "").strip()), max_chars=_chunk_max_chars())
    if not text or _looks_bad_text(text):
        return
    score = _fallback_candidate_score(text, proof_role=proof_role, position=position)
    if heading and re.search(r"(market|size|scale|metric|data|revenue|growth|case|risk|policy|trend|规模|市场|数据|增速|案例|风险|政策|趋势)", heading, flags=re.I):
        score += 5
    chunks.append(
        {
            "chunk_index": 0,
            "chunk_heading": heading,
            "chunk_text": text,
            "chunk_score": score,
            "source_position": position,
        }
    )


def _split_page_into_chunks(page: Dict[str, Any], *, proof_role: str) -> List[Dict[str, Any]]:
    text = _page_text(page)
    if not text:
        return []
    max_chars = _chunk_max_chars()
    chunks: List[Dict[str, Any]] = []
    buffer: List[str] = []
    heading = ""
    position = 0
    for raw_line in text.splitlines():
        line = str(raw_line or "").strip()
        if not line:
            continue
        is_heading = bool(_CHUNK_HEADING_RE.match(line))
        if is_heading and buffer:
            _flush_chunk(chunks, lines=buffer, heading=heading, position=position, proof_role=proof_role)
            position += 1
            buffer = []
        if is_heading:
            heading = _clean_chunk_heading(line)
            buffer.append(line)
        else:
            buffer.append(line)
        if len("\n".join(buffer)) >= max_chars:
            _flush_chunk(chunks, lines=buffer, heading=heading, position=position, proof_role=proof_role)
            position += 1
            buffer = []
    if buffer:
        _flush_chunk(chunks, lines=buffer, heading=heading, position=position, proof_role=proof_role)
    if not chunks:
        compact = _compact_text(text, max_chars=max_chars)
        if compact and not _looks_bad_text(compact):
            chunks.append(
                {
                    "chunk_index": 0,
                    "chunk_heading": str(page.get("title") or ""),
                    "chunk_text": compact,
                    "chunk_score": _fallback_candidate_score(compact, proof_role=proof_role, position=0),
                    "source_position": 0,
                }
            )
    chunks.sort(key=lambda item: (-int(item.get("chunk_score") or 0), int(item.get("source_position") or 0)))
    selected = chunks[: _max_chunks_per_page()]
    for index, chunk in enumerate(selected, start=1):
        chunk["chunk_index"] = index
    return selected


def _chunk_page_for_llm(page: Dict[str, Any], chunk: Dict[str, Any]) -> Dict[str, Any]:
    chunk_page = dict(page)
    chunk_page["content"] = str(chunk.get("chunk_text") or "")
    chunk_page["mainText"] = str(chunk.get("chunk_text") or "")
    chunk_page["_chunk_index"] = chunk.get("chunk_index")
    chunk_page["_chunk_heading"] = chunk.get("chunk_heading") or ""
    return chunk_page


def _apply_chunk_metadata(payload: Dict[str, Any], chunk: Dict[str, Any]) -> Dict[str, Any]:
    chunk_index = int(chunk.get("chunk_index") or 0)
    chunk_heading = str(chunk.get("chunk_heading") or "")
    chunk_text = str(chunk.get("chunk_text") or "")
    cards = [item for item in _as_list(payload.get("fact_cards")) if isinstance(item, dict)][:_chunk_max_fact_cards()]
    payload["fact_cards"] = cards
    for card in cards:
        card["chunk_index"] = chunk_index
        card["chunk_heading"] = chunk_heading
        card["extraction_mode"] = "chunk_llm"
        span = _compact_text(
            card.get("distilled_fact") or card.get("fact") or card.get("action_or_signal") or chunk_text,
            max_chars=_chunk_max_fact_chars(),
        )
        card["chunk_evidence_span"] = span
        evidence_id = str(card.get("evidence_id") or "").strip()
        if evidence_id and not re.search(r"-C\d{2}$", evidence_id):
            card["evidence_id"] = f"{evidence_id}-C{chunk_index:02d}"
            card["ref"] = card["evidence_id"]
        public_card = _as_dict(card.get("public_fact_card"))
        if public_card:
            public_card.update(
                {
                    "chunk_index": chunk_index,
                    "chunk_heading": chunk_heading,
                    "chunk_evidence_span": span,
                    "extraction_mode": "chunk_llm",
                }
            )
            if card.get("evidence_id"):
                public_card["evidence_id"] = card.get("evidence_id")
                public_card["ref"] = card.get("ref")
            card["public_fact_card"] = public_card
        quality = _as_dict(card.get("public_fact_quality"))
        if isinstance(quality.get("public_fact_card"), dict):
            quality["public_fact_card"] = copy.deepcopy(public_card or card)
            card["public_fact_quality"] = quality
    return payload


def _dedupe_fact_cards(cards: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    deduped: List[Dict[str, Any]] = []
    for card in cards:
        key = _hash_text(
            "|".join(
                [
                    str(card.get("source_url") or card.get("source_ref") or ""),
                    str(card.get("distilled_fact") or card.get("fact") or ""),
                    str(card.get("value") or ""),
                    str(card.get("period") or card.get("time_or_scope") or ""),
                ]
            ).lower()
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(card)
    return deduped


def _metric_missing_fields(card: Dict[str, Any], required_fields: Optional[Sequence[Any]] = None) -> List[str]:
    required = {
        str(item or "").strip().lower()
        for item in _as_list(list(required_fields or []))
        if str(item or "").strip()
    }
    if not required:
        required = {"subject", "period", "value", "unit"}
    missing: List[str] = []
    if "subject" in required and not str(card.get("subject") or "").strip():
        missing.append("subject")
    if "metric" in required and not str(card.get("metric") or card.get("variable") or "").strip():
        missing.append("metric")
    if ("period" in required or "scope" in required or "date" in required) and not (
        str(card.get("period") or "").strip()
        or str(card.get("scope") or "").strip()
        or str(card.get("time_or_scope") or "").strip()
        or str(card.get("date") or "").strip()
    ):
        missing.append("period")
    if "value" in required and not str(card.get("value") or "").strip():
        missing.append("value")
    value_text = str(card.get("value") or "").strip().lower()
    value_carries_unit = bool(
        value_text
        and (
            "%"
            in value_text
            or "percent" in value_text
            or "percentage point" in value_text
            or "百分点" in value_text
            or "百分比" in value_text
        )
    )
    if "unit" in required and not str(card.get("unit") or "").strip() and not value_carries_unit:
        missing.append("unit")
    if "source" in required and not (
        str(card.get("source_url") or "").strip()
        or str(card.get("source_ref") or "").strip()
        or str(card.get("source") or "").strip()
    ):
        missing.append("source")
    return missing


def _metric_text_for_inference(card: Dict[str, Any]) -> str:
    return _compact_text(
        " ".join(
            str(card.get(key) or "")
            for key in (
                "distilled_fact",
                "fact",
                "action_or_signal",
                "summary",
                "metric",
                "variable",
                "value",
            )
        ),
        max_chars=1200,
    )


def _infer_metric_period_from_text(text: str) -> str:
    match = re.search(r"\b((?:19|20)\d{2})(?:\s*(?:年|年度|财年|calendar year|fiscal year|fy)?)\b", text, flags=re.I)
    return match.group(1) if match else ""


METRIC_UNIT_PATTERN = (
    r"%|percent|percentage points?|"
    r"billion yuan|million yuan|trillion yuan|yuan|rmb|"
    r"billion usd|million usd|usd|dollars?|"
    r"users?|customers?|companies|enterprises|units?|shipments?|"
    r"\u4ebf\u5143|\u4e07\u5143|\u5143|\u4ebf\u7f8e\u5143|\u4e07\u7f8e\u5143|\u7f8e\u5143|"
    r"\u4e07\u4eba|\u4ebf\u4eba|\u4eba|\u4e07\u6237|\u6237|\u5bb6|\u53f0|\u5957"
)


def _infer_metric_value_from_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text or "").strip()
    match = re.search(rf"\b(\d+(?:,\d{{3}})*(?:\.\d+)?)\s*(?:{METRIC_UNIT_PATTERN})", normalized, flags=re.I)
    if match:
        return match.group(1).replace(",", "")
    percent = re.search(r"\b(\d+(?:\.\d+)?)\s*%", normalized)
    return percent.group(1) if percent else ""


def _infer_metric_unit_from_text(text: str, value: Any) -> str:
    value_text = str(value or "").strip()
    if re.search(r"%|percent|percentage point|百分点|百分比", value_text, flags=re.I):
        return "%"
    normalized = re.sub(r"\s+", " ", text or "").strip()
    if value_text:
        value_re = re.escape(value_text).replace(r"\ ", r"\s*")
        match = re.search(rf"{value_re}\s*({METRIC_UNIT_PATTERN})", normalized, flags=re.I)
        if match:
            return match.group(1).strip()
    match = re.search(rf"\b\d+(?:\.\d+)?\s*({METRIC_UNIT_PATTERN})", normalized, flags=re.I)
    return match.group(1).strip() if match else ""


def _infer_metric_name_from_text(text: str, card: Dict[str, Any]) -> str:
    lower = str(text or "").lower()
    patterns = [
        (r"\bmarket size\b|\bmarket scale\b", "market size"),
        (r"\bmarket share\b", "market share"),
        (r"\badoption rate\b|\bpenetration rate\b", "adoption rate"),
        (r"\bcagr\b|\bcompound annual growth rate\b", "CAGR"),
        (r"\bgrowth rate\b|\byoy\b|year[- ]over[- ]year", "growth rate"),
        (r"\brevenue\b|\bsales\b", "revenue"),
        (r"\bshipments?\b", "shipments"),
        (r"\busers?\b|\bcustomers?\b", "user count"),
        (r"\bprice\b|\bpricing\b", "price"),
        (r"\broi\b|return on investment", "ROI"),
    ]
    for pattern, name in patterns:
        if re.search(pattern, lower, flags=re.I):
            return name
    action = _compact_text(card.get("action_or_signal"), max_chars=80)
    action = re.split(r"\b(?:reached|was|were|is|are|stood at|amounted to)\b|[:：,;，；]", action, maxsplit=1, flags=re.I)[0]
    return _compact_text(action, max_chars=48)


def _repair_metric_fields_from_text(card: Dict[str, Any]) -> Dict[str, Any]:
    text = _metric_text_for_inference(card)
    if not str(card.get("value") or "").strip():
        value = _infer_metric_value_from_text(text)
        if value:
            card["value"] = value
    if not str(card.get("period") or card.get("time_or_scope") or "").strip():
        period = _infer_metric_period_from_text(text)
        if period:
            card["period"] = period
            card.setdefault("time_or_scope", period)
    if not str(card.get("unit") or "").strip():
        unit = _infer_metric_unit_from_text(text, card.get("value"))
        if unit:
            card["unit"] = unit
    if not str(card.get("metric") or card.get("variable") or "").strip():
        metric = _infer_metric_name_from_text(text, card)
        if metric:
            card["metric"] = metric
            card.setdefault("variable", metric)
    return card


def _rejection(reason: str, card: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "reason": reason,
        "text": _compact_text(card.get("distilled_fact") or card.get("fact") or card.get("action_or_signal"), max_chars=220),
        "source_url": card.get("source_url"),
        "source_ref": card.get("source_ref"),
    }


def _validated_card(
    raw_card: Dict[str, Any],
    *,
    source_url: str,
    source_ref: str,
    source_level: str,
    verification_status: str,
    proof_role: str,
    chapter_id: str = "",
    search_task: Optional[Dict[str, Any]] = None,
    index: int = 0,
) -> tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    card = copy.deepcopy(raw_card)
    rejected: List[Dict[str, Any]] = []
    task = _as_dict(search_task)
    runtime_values = {
        "source_url": str(source_url or card.get("source_url") or "").strip(),
        "source_ref": str(source_ref or card.get("source_ref") or "").strip(),
        "proof_role": str(proof_role or task.get("proof_role") or task.get("evidence_type") or card.get("proof_role") or "").strip(),
        "chapter_id": str(chapter_id or task.get("chapter_id") or task.get("dimension_id") or card.get("chapter_id") or "").strip(),
        "section_id": str(task.get("section_id") or card.get("section_id") or "").strip(),
        "requirement_id": str(task.get("requirement_id") or card.get("requirement_id") or "").strip(),
        "gap_id": str(task.get("gap_id") or card.get("gap_id") or "").strip(),
        "search_task_id": str(task.get("search_task_id") or task.get("task_id") or card.get("search_task_id") or "").strip(),
    }
    for key in ("task_id", "dimension_id", "dimension_name", "evidence_goal", "hypothesis_id"):
        if key in task:
            runtime_values[key] = task.get(key)
    cached_context: Dict[str, Any] = {}
    for key, runtime_value in runtime_values.items():
        if runtime_value in (None, ""):
            continue
        old_value = card.get(key)
        if old_value not in (None, "") and str(old_value) != str(runtime_value):
            cached_context[key] = old_value
        card[key] = runtime_value
    if cached_context:
        card["cached_context"] = {**_as_dict(card.get("cached_context")), **cached_context}
    card_source_url = str(card.get("source_url") or "").strip()
    card_source_ref = str(card.get("source_ref") or "").strip()
    card["source_url"] = card_source_url
    card["source_ref"] = card_source_ref
    card["source_level"] = str(card.get("source_level") or source_level or "C").strip().upper()
    card["source_verification_status"] = str(card.get("source_verification_status") or verification_status or "readpage_verified").strip()
    card["proof_role"] = str(card.get("proof_role") or "").strip()
    fact_type = str(card.get("fact_type") or card.get("proof_role") or "case").strip().lower()
    card["fact_type"] = fact_type
    distilled = str(card.get("distilled_fact") or card.get("fact") or card.get("action_or_signal") or "").strip()
    card["distilled_fact"] = distilled
    card["fact"] = distilled
    card["clean_fact"] = distilled
    card["block_affinity"] = _normalize_affinity(card.get("block_affinity"), fact_type, card["proof_role"])
    card["claim_strength_hint"] = str(card.get("claim_strength_hint") or ("strong" if card["source_level"] in {"A", "B"} and card["source_verification_status"] in {"readpage_verified", "document_verified"} else "directional")).strip()
    if fact_type == "metric":
        card = _repair_metric_fields_from_text(card)

    if not (card_source_url or card_source_ref):
        rejected.append(_rejection("missing_source_ref", card))
    if card_source_url:
        host = urlparse(card_source_url).netloc.lower()
        if LOW_QUALITY_HOST_RE.search(host) and not public_signal_mode():
            rejected.append(_rejection("low_quality_source", card))
        if re.search(r"example\.(?:com|gov|org)", card_source_url, flags=re.I):
            rejected.append(_rejection("fake_or_placeholder_source", card))
    guard = public_text_quality(distilled)
    if guard.get("severity") == "reject":
        rejected.append({**_rejection("dirty_public_text", card), "guard_reasons": guard.get("reasons", [])})
    if _looks_bad_text(distilled):
        reason = "internal_or_claim_like_text" if INTERNAL_OR_CLAIM_PATTERNS.search(distilled) else "navigation_or_low_quality_text"
        rejected.append(_rejection(reason, card))
    if fact_type == "metric":
        missing = _metric_missing_fields(card, task.get("required_fields"))
        if missing:
            rejected.append({**_rejection("metric_missing_scope_or_period", card), "missing_fields": missing})
    required_source_levels = {str(item or "").strip().upper() for item in _as_list(task.get("required_source_level") or task.get("min_source_level")) if str(item or "").strip()}
    if required_source_levels and card["source_level"] not in required_source_levels:
        card["source_level_gap"] = {
            "required": sorted(required_source_levels),
            "actual": card["source_level"],
        }
        card["claim_strength_hint"] = "directional"
        card["allowed_use"] = "directional_signal"
        card["evidence_use_level"] = "directional_signal"
        card["limitation_boundary"] = _as_list(card.get("limitation_boundary")) + [
            "来源等级低于任务偏好，只能作为方向性信号，不能单独支撑强结论。"
        ]
    if card.get("source_title_url_mismatch_suspected"):
        rejected.append(_rejection("source_mismatch", card))

    if rejected:
        rejected.sort(key=lambda item: 0 if str(item.get("reason") or "").startswith("metric_") else 1)
        return None, rejected

    card.setdefault("metric", card.get("variable"))
    card.setdefault("source", card.get("source_title") or card_source_url or card_source_ref)
    card.setdefault("source_title", card.get("source_title") or card.get("title") or "")
    card.setdefault("url", card_source_url)
    card.setdefault("evidence_origin", "readpage_fact_extractor")
    card.setdefault("extraction_schema_version", SCHEMA_VERSION)
    card.setdefault("source_verified", card["source_verification_status"] in {"readpage_verified", "document_verified"})
    role = str(card.get("proof_role") or "").strip().lower()
    if role == "counter" or fact_type == "counter":
        card.setdefault("allowed_use", "counter")
    elif card["source_level"] in {"A", "B"} and card["source_verification_status"] in {"readpage_verified", "document_verified"}:
        card.setdefault("allowed_use", "supporting")
    else:
        card.setdefault("allowed_use", "directional_signal")
    card.setdefault("public_fact_card", copy.deepcopy(card))
    card.setdefault("public_fact_quality", {"eligible_for_report": True, "eligible_for_citation": True, "public_fact_card": copy.deepcopy(card)})
    # Make evidence_id collision-resistant: encode the (task_id, proof_role)
    # tuple alongside the source so the same page seen by two different
    # search tasks (or two different proof roles) gets distinct ids.
    task_slot = re.sub(r"[^A-Za-z0-9]+", "", str(task.get("task_id") or task.get("dimension_id") or ""))[:12] or "x"
    role_slot = re.sub(r"[^A-Za-z0-9]+", "", str(card.get("proof_role") or task.get("proof_role") or ""))[:8] or "any"
    source_slot = card_source_ref or _hash_text(card_source_url)[:8] or "src"
    card.setdefault(
        "evidence_id",
        f"RFC-{source_slot}-{task_slot}-{role_slot}-{index + 1}",
    )
    card.setdefault("ref", card["evidence_id"])
    for key in (
        "task_id",
        "search_task_id",
        "requirement_id",
        "gap_id",
        "section_id",
        "dimension_id",
        "dimension_name",
        "evidence_goal",
        "must_have_terms",
        "forbidden_terms",
        "source_priority",
        "required_fields",
        "required_source_level",
        "success_criteria",
        "reject_if",
        "hypothesis_id",
    ):
        if key in task and key not in card:
            card[key] = task.get(key)
    return card, []


def _cacheable_extractor_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    runtime_keys = {
        "source_ref",
        "ref",
        "evidence_id",
        "chapter_id",
        "section_id",
        "requirement_id",
        "gap_id",
        "search_task_id",
        "dimension_id",
        "dimension_name",
        "task_id",
        "evidence_goal",
        "must_have_terms",
        "forbidden_terms",
        "source_priority",
        "required_fields",
        "required_source_level",
        "success_criteria",
        "reject_if",
        "hypothesis_id",
        "search_task",
        "cached_context",
        "public_fact_card",
        "public_fact_quality",
    }
    cards: List[Dict[str, Any]] = []
    for item in _as_list(payload.get("fact_cards")):
        if not isinstance(item, dict):
            continue
        card = {key: copy.deepcopy(value) for key, value in item.items() if key not in runtime_keys}
        cards.append(card)
    return {
        "schema_version": SCHEMA_VERSION,
        "fact_cards": cards,
        "rejected_spans": copy.deepcopy(_as_list(payload.get("rejected_spans"))),
    }


def _store_fallback_cache_if_useful(key: str, payload: Dict[str, Any]) -> None:
    if not _env_flag("READPAGE_FACT_EXTRACTOR_CACHE_FALLBACK_ENABLED", True):
        return
    if not _as_list(payload.get("fact_cards")):
        return
    cache_payload = _cacheable_extractor_payload(payload)
    cache_payload["cache_origin"] = "fallback_fact_extractor"
    _store_cache(key, cache_payload)


def validate_extracted_fact_payload(
    payload: Dict[str, Any],
    *,
    source_url: str = "",
    source_ref: str = "",
    source_level: str = "C",
    verification_status: str = "readpage_verified",
    proof_role: str = "",
    chapter_id: str = "",
    search_task: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    fact_cards: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for index, item in enumerate(_as_list(_as_dict(payload).get("fact_cards"))):
        if not isinstance(item, dict):
            rejected.append({"reason": "non_dict_fact_card", "text": str(item)[:200]})
            continue
        card, card_rejections = _validated_card(
            item,
            source_url=source_url,
            source_ref=source_ref,
            source_level=source_level,
            verification_status=verification_status,
            proof_role=proof_role,
            chapter_id=chapter_id,
            search_task=search_task,
            index=index,
        )
        if card:
            fact_cards.append(card)
        rejected.extend(card_rejections)
    for item in _as_list(_as_dict(payload).get("rejected_spans")):
        if isinstance(item, dict):
            rejected.append(item)
    return {
        "fact_cards": fact_cards,
        "rejected_spans": rejected,
        "invalid_metric_count": sum(1 for item in rejected if str(item.get("reason") or "").startswith("metric_")),
    }


def _extract_fact_cards_from_chunks(
    *,
    config: Dict[str, Any],
    query: str,
    page: Dict[str, Any],
    search_task: Dict[str, Any],
    proof_role: str,
    max_chars: int,
    result: Dict[str, Any],
    trigger_reason: str,
) -> Dict[str, Any]:
    chunks = _split_page_into_chunks(page, proof_role=proof_role)
    if not chunks:
        return {"fact_cards": [], "rejected_spans": [], "invalid_metric_count": 0}
    result["chunk_mode_used"] = True
    result["chunk_trigger_reason"] = str(result.get("chunk_trigger_reason") or trigger_reason)
    result["chunk_count"] += len(chunks)
    collected_cards: List[Dict[str, Any]] = []
    collected_rejected: List[Dict[str, Any]] = []
    invalid_metric_count = 0
    for chunk in chunks:
        result["chunk_attempted"] += 1
        chunk_page = _chunk_page_for_llm(page, chunk)
        chunk_key = _chunk_cache_key(page, chunk, proof_role=proof_role, query=query)
        cached = _load_cache(chunk_key)
        if cached:
            result["chunk_cache_hit_count"] += 1
            validated = validate_extracted_fact_payload(
                _as_dict(cached),
                source_url=_source_url(page),
                source_ref=_source_ref(page),
                source_level=_source_level(page),
                verification_status=_verification_status(page),
                proof_role=proof_role,
                chapter_id=str(search_task.get("chapter_id") or search_task.get("dimension_id") or ""),
                search_task=search_task,
            )
            validated = _apply_chunk_metadata(validated, chunk)
        else:
            consumed_budget, budget_exhausted = _try_consume_budget()
            if not consumed_budget:
                result["budget_exhausted"] = bool(budget_exhausted)
                break
            try:
                response = call_openai_compatible_json(
                    config=config,
                    system_prompt=_system_prompt(),
                    user_payload=_user_payload(
                        query=query,
                        page=chunk_page,
                        search_task=search_task,
                        max_chars=min(max_chars, _chunk_max_chars()),
                    ),
                )
                payload = _as_dict(response.get("payload"))
                validated = validate_extracted_fact_payload(
                    payload,
                    source_url=_source_url(page),
                    source_ref=_source_ref(page),
                    source_level=_source_level(page),
                    verification_status=_verification_status(page),
                    proof_role=proof_role,
                    chapter_id=str(search_task.get("chapter_id") or search_task.get("dimension_id") or ""),
                    search_task=search_task,
                )
                validated = _apply_chunk_metadata(validated, chunk)
                _store_cache(chunk_key, _cacheable_extractor_payload(validated))
            except Exception as exc:
                result["llm_error_count"] += 1
                result["errors"].append(str(exc))
                continue
        cards = [item for item in _as_list(validated.get("fact_cards")) if isinstance(item, dict)]
        rejected = [item for item in _as_list(validated.get("rejected_spans")) if isinstance(item, dict)]
        if cards:
            result["chunk_success_count"] += 1
            result["chunk_fact_card_count"] += len(cards)
            collected_cards.extend(cards)
        collected_rejected.extend(rejected)
        invalid_metric_count += int(validated.get("invalid_metric_count") or 0)
    return {
        "fact_cards": _dedupe_fact_cards(collected_cards),
        "rejected_spans": collected_rejected,
        "invalid_metric_count": invalid_metric_count,
    }


def _fallback_subject_from_sentence(sentence: str) -> str:
    value = _compact_text(sentence, max_chars=80)
    value = re.sub(r"^[#\s]+", "", value)
    value = re.sub(r"^[^:：]{4,60}[:：]\s*", "", value)
    for sep in ("，", "。", "；", ";", ",", ":"):
        if sep in value:
            candidate = _compact_text(value.split(sep, 1)[0], max_chars=32)
            if 2 <= len(candidate) <= 32 and not _looks_bad_text(candidate):
                return candidate
    candidate = _compact_text(value, max_chars=24)
    if re.search(r"[a-z0-9-]+\.(?:com|cn|net|org)|\b(?:ijiwei|36kr|sina|sohu|baidu|zhihu)\b", candidate, flags=re.I):
        return ""
    return candidate


def _fallback_fact_cards_from_page(page: Dict[str, Any], *, search_task: Dict[str, Any], proof_role: str) -> Dict[str, Any]:
    text = _page_text(page)
    source_url = _source_url(page)
    source_ref = _source_ref(page)
    payload_cards: List[Dict[str, Any]] = []
    for cleaned in _fallback_candidate_sentences(text, proof_role=proof_role):
        subject = _fallback_subject_from_sentence(cleaned)
        payload_cards.append(
            {
                "subject": subject,
                "action_or_signal": cleaned,
                "variable": str(proof_role or "case"),
                "time_or_scope": "",
                "distilled_fact": cleaned,
                "fact_type": "metric" if re.search(r"\d+(?:\.\d+)?\s*(?:%|亿元|亿美元|CAGR)", cleaned, flags=re.I) else (proof_role or "case"),
                "source_url": source_url,
                "source_ref": source_ref,
                "source_level": _source_level(page),
                "source_verification_status": _verification_status(page),
                "proof_role": proof_role,
                "block_affinity": _normalize_affinity([], proof_role, proof_role),
                "claim_strength_hint": "directional",
            }
        )
        if len(payload_cards) >= 2:
            break
    return validate_extracted_fact_payload(
        {"fact_cards": payload_cards},
        source_url=source_url,
        source_ref=source_ref,
        source_level=_source_level(page),
        verification_status=_verification_status(page),
        proof_role=proof_role,
        chapter_id=str(search_task.get("chapter_id") or search_task.get("dimension_id") or ""),
        search_task=search_task,
    )


def _zero_valid_fallback_fact_cards_from_page(page: Dict[str, Any], *, search_task: Dict[str, Any], proof_role: str) -> Dict[str, Any]:
    fallback = _fallback_fact_cards_from_page(page, search_task=search_task, proof_role=proof_role)
    if _as_list(fallback.get("fact_cards")):
        return fallback
    text = _page_text(page)
    source_url = _source_url(page)
    source_ref = _source_ref(page)
    sentence = ""
    for candidate in re.split(r"(?<=[。！？!?])\s*|\n+", text):
        cleaned = _compact_text(candidate, max_chars=260)
        if (
            cleaned
            and not _looks_bad_fallback_candidate(cleaned)
            and not _looks_bad_text(cleaned)
            and public_text_quality(cleaned).get("severity") != "reject"
        ):
            sentence = cleaned
            break
    if not sentence:
        sentence = _compact_text(text, max_chars=260)
    if not sentence or _looks_bad_fallback_candidate(sentence) or _looks_bad_text(sentence):
        return fallback
    raw_card = {
        "subject": _fallback_subject_from_sentence(sentence) or str(page.get("title") or proof_role or "source fact")[:80],
        "action_or_signal": sentence,
        "variable": str(proof_role or "case"),
        "time_or_scope": "",
        "distilled_fact": sentence,
        "fact_type": str(proof_role or "case"),
        "source_url": source_url,
        "source_ref": source_ref,
        "source_level": _source_level(page),
        "source_verification_status": _verification_status(page),
        "proof_role": proof_role,
        "block_affinity": _normalize_affinity([], proof_role, proof_role),
        "claim_strength_hint": "directional",
        "extraction_mode": "zero_valid_rule_fallback",
    }
    return validate_extracted_fact_payload(
        {"fact_cards": [raw_card]},
        source_url=source_url,
        source_ref=source_ref,
        source_level=_source_level(page),
        verification_status=_verification_status(page),
        proof_role=proof_role,
        chapter_id=str(search_task.get("chapter_id") or search_task.get("dimension_id") or ""),
        search_task=search_task,
    )


def extract_fact_cards_from_pages(
    *,
    query: str,
    page_results: Sequence[Dict[str, Any]],
    search_task: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    task = _as_dict(search_task)
    proof_role = str(task.get("proof_role") or task.get("evidence_type") or "").strip()
    enabled = _env_flag("READPAGE_FACT_EXTRACTOR_ENABLED", True)
    max_pages = _env_int("READPAGE_FACT_EXTRACTOR_MAX_PAGES_PER_TASK", 4, min_value=0, max_value=20)
    max_chars = _env_int("READPAGE_FACT_EXTRACTOR_MAX_CHARS_PER_PAGE", 6000, min_value=500, max_value=30_000)
    start_budget_used = _budget_used()
    selected_pages = [dict(item) for item in page_results if isinstance(item, dict) and _page_text(item)][:max_pages]
    result: Dict[str, Any] = {
        "enabled": enabled,
        "schema_version": SCHEMA_VERSION,
        "attempted": 0,
        "success_count": 0,
        "fact_card_count": 0,
        "rejected_span_count": 0,
        "invalid_metric_count": 0,
        "cache_hit_count": 0,
        "llm_error_count": 0,
        "fallback_used": False,
        "zero_valid_fallback_used": False,
        "whole_page_failed": False,
        "chunk_mode_used": False,
        "chunk_trigger_reason": "",
        "chunk_count": 0,
        "chunk_attempted": 0,
        "chunk_success_count": 0,
        "chunk_fact_card_count": 0,
        "chunk_cache_hit_count": 0,
        "fact_cards": [],
        "rejected_spans": [],
        "rejected_span_repair_summary": {},
        "errors": [],
        "model": "",
        "budget_limit": _budget_limit(),
        "budget_used": 0,
        "budget_exhausted": False,
    }
    if not selected_pages:
        result["status"] = "no_readpage_text"
        _emit_fact_extractor_probe(result, task=task, selected_page_count=0, raw_page_count=len(page_results or []), page_results=page_results)
        return result
    if not enabled:
        result["status"] = "disabled"
        _emit_fact_extractor_probe(result, task=task, selected_page_count=len(selected_pages), raw_page_count=len(page_results or []), page_results=page_results)
        return result

    config = _build_llm_config()
    ready = bool(llm_config_is_ready(config))
    result["model"] = str(normalize_llm_config(config).get("model") or "")
    for page in selected_pages:
        result["attempted"] += 1
        key = _cache_key(page, proof_role=proof_role, query=query)
        cached = _load_cache(key)
        if cached:
            result["cache_hit_count"] += 1
            validated = validate_extracted_fact_payload(
                _as_dict(cached),
                source_url=_source_url(page),
                source_ref=_source_ref(page),
                source_level=_source_level(page),
                verification_status=_verification_status(page),
                proof_role=proof_role,
                chapter_id=str(task.get("chapter_id") or task.get("dimension_id") or ""),
                search_task=task,
            )
        else:
            if ready:
                validated: Optional[Dict[str, Any]] = None
                tried_chunk = False
                if _chunk_enabled() and len(_page_text(page)) >= _chunk_trigger_chars():
                    tried_chunk = True
                    chunk_validated = _extract_fact_cards_from_chunks(
                        config=config,
                        query=query,
                        page=page,
                        search_task=task,
                        proof_role=proof_role,
                        max_chars=max_chars,
                        result=result,
                        trigger_reason="page_too_long",
                    )
                    if _as_list(chunk_validated.get("fact_cards")):
                        validated = chunk_validated
                if validated is None:
                    consumed_budget, budget_exhausted = _try_consume_budget()
                    if not consumed_budget:
                        result["budget_exhausted"] = bool(budget_exhausted)
                        validated = _fallback_fact_cards_from_page(page, search_task=task, proof_role=proof_role)
                        _store_fallback_cache_if_useful(key, validated)
                        result["fallback_used"] = True
                        cards = _as_list(validated.get("fact_cards"))
                        if cards:
                            result["success_count"] += 1
                            result["fact_cards"].extend([item for item in cards if isinstance(item, dict)])
                        rejected = [item for item in _as_list(validated.get("rejected_spans")) if isinstance(item, dict)]
                        result["rejected_spans"].extend(rejected)
                        result["invalid_metric_count"] += int(validated.get("invalid_metric_count") or 0)
                        continue
                    try:
                        response = call_openai_compatible_json(
                            config=config,
                            system_prompt=_system_prompt(),
                            user_payload=_user_payload(query=query, page=page, search_task=task, max_chars=max_chars),
                        )
                        payload = _as_dict(response.get("payload"))
                        validated = validate_extracted_fact_payload(
                            payload,
                            source_url=_source_url(page),
                            source_ref=_source_ref(page),
                            source_level=_source_level(page),
                            verification_status=_verification_status(page),
                            proof_role=proof_role,
                            chapter_id=str(task.get("chapter_id") or task.get("dimension_id") or ""),
                            search_task=task,
                        )
                        if not _as_list(validated.get("fact_cards")) and _env_flag("READPAGE_FACT_EXTRACTOR_ZERO_VALID_FALLBACK_ENABLED", True):
                            fallback_validated = _zero_valid_fallback_fact_cards_from_page(page, search_task=task, proof_role=proof_role)
                            if _as_list(fallback_validated.get("fact_cards")):
                                validated = fallback_validated
                                _store_fallback_cache_if_useful(key, validated)
                                result["fallback_used"] = True
                                result["zero_valid_fallback_used"] = True
                            else:
                                _store_cache(key, _cacheable_extractor_payload(validated))
                        else:
                            _store_cache(key, _cacheable_extractor_payload(validated))
                    except Exception as exc:
                        result["llm_error_count"] += 1
                        result["whole_page_failed"] = True
                        result["errors"].append(str(exc))
                        if _chunk_enabled() and not tried_chunk and len(_page_text(page)) >= _chunk_trigger_chars():
                            chunk_validated = _extract_fact_cards_from_chunks(
                                config=config,
                                query=query,
                                page=page,
                                search_task=task,
                                proof_role=proof_role,
                                max_chars=max_chars,
                                result=result,
                                trigger_reason="llm_error",
                            )
                            if _as_list(chunk_validated.get("fact_cards")):
                                validated = chunk_validated
                        if validated is None:
                            validated = _fallback_fact_cards_from_page(page, search_task=task, proof_role=proof_role)
                            _store_fallback_cache_if_useful(key, validated)
                            result["fallback_used"] = True
            else:
                validated = _fallback_fact_cards_from_page(page, search_task=task, proof_role=proof_role)
                _store_fallback_cache_if_useful(key, validated)
                result["fallback_used"] = True
        cards = _as_list(validated.get("fact_cards"))
        if cards:
            result["success_count"] += 1
            result["fact_cards"].extend([item for item in cards if isinstance(item, dict)])
        rejected = [item for item in _as_list(validated.get("rejected_spans")) if isinstance(item, dict)]
        result["rejected_spans"].extend(rejected)
        result["invalid_metric_count"] += int(validated.get("invalid_metric_count") or 0)
    if (
        not result["fact_cards"]
        and ready
        and _env_flag("READPAGE_FACT_EXTRACTOR_ZERO_VALID_FALLBACK_ENABLED", True)
    ):
        for page in selected_pages:
            fallback_validated = _zero_valid_fallback_fact_cards_from_page(page, search_task=task, proof_role=proof_role)
            cards = [item for item in _as_list(fallback_validated.get("fact_cards")) if isinstance(item, dict)]
            if not cards:
                continue
            result["fact_cards"].extend(cards)
            result["success_count"] += 1
            result["fallback_used"] = True
            result["zero_valid_fallback_used"] = True
            result["invalid_metric_count"] += int(fallback_validated.get("invalid_metric_count") or 0)
            break
    result["fact_card_count"] = len(result["fact_cards"])
    result["rejected_span_count"] = len(result["rejected_spans"])
    result["rejected_span_repair_summary"] = rejected_span_repair_summary(
        [item for item in result["rejected_spans"] if isinstance(item, dict)],
        search_task=task,
    )
    result["budget_used"] = max(0, _budget_used() - start_budget_used)
    if result["budget_exhausted"] and result["fact_card_count"]:
        result["status"] = "partial_budget_exhausted"
    elif result["budget_exhausted"]:
        result["status"] = "budget_exhausted"
    else:
        result["status"] = "success" if result["fact_card_count"] else ("fallback_empty" if result["fallback_used"] else "no_valid_fact_cards")
    _emit_fact_extractor_probe(result, task=task, selected_page_count=len(selected_pages), raw_page_count=len(page_results or []), page_results=page_results)
    return result
