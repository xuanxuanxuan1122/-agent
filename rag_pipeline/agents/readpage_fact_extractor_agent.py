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


SCHEMA_VERSION = "readpage_fact_card_v1"

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
    r"\u9996\u9875|\u767b\u5f55|\u6ce8\u518c|\u9690\u79c1|\u7248\u6743|\u4e0b\u8f7d|\u539f\u6587\u94fe\u63a5|\u76ee\u5f55)",
    re.I,
)

LOW_QUALITY_HOST_RE = re.compile(
    r"(?:twitter|x|instagram|facebook|baike|baijiahao|csdn|cnblogs|juejin|wenku|doc88|book118)\.",
    re.I,
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
        "You extract structured public fact cards from verified web page body text for industry research. "
        "Use only the supplied page text and source metadata. Do not write a report, do not give repair advice, "
        "and do not create claims unsupported by the page. Return strict JSON only. "
        "Each fact card must include subject, action_or_signal, variable, distilled_fact, fact_type, "
        "source_url or source_ref, source_level, source_verification_status, proof_role, block_affinity, "
        "claim_strength_hint. Metric cards must include subject, scope or time_or_scope, value, unit, and source. "
        "Reject navigation, login, download notices, SEO text, search-result summaries, social/wiki/forum text, "
        "and diagnostic phrases such as evidence is insufficient or suggest repair."
    )


def _user_payload(*, query: str, page: Dict[str, Any], search_task: Dict[str, Any], max_chars: int) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "query": query,
        "search_task": {
            "task_id": search_task.get("task_id"),
            "chapter_id": search_task.get("chapter_id") or search_task.get("dimension_id"),
            "dimension_id": search_task.get("dimension_id"),
            "proof_role": search_task.get("proof_role") or search_task.get("evidence_type"),
            "evidence_goal": search_task.get("evidence_goal"),
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
    if NAVIGATION_PATTERNS.search(value) and len(value) < 120:
        return True
    if re.search(r"^\s*(?:https?://|www\.)", value, flags=re.I):
        return True
    return False


def _metric_missing_fields(card: Dict[str, Any]) -> List[str]:
    missing: List[str] = []
    if not str(card.get("subject") or "").strip():
        missing.append("subject")
    if not (str(card.get("scope") or "").strip() or str(card.get("time_or_scope") or "").strip()):
        missing.append("scope_or_period")
    if not str(card.get("value") or "").strip():
        missing.append("value")
    if not str(card.get("unit") or "").strip():
        missing.append("unit")
    return missing


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

    if not (card_source_url or card_source_ref):
        rejected.append(_rejection("missing_source_ref", card))
    if card_source_url:
        host = urlparse(card_source_url).netloc.lower()
        if LOW_QUALITY_HOST_RE.search(host):
            rejected.append(_rejection("low_quality_source", card))
        if re.search(r"example\.(?:com|gov|org)", card_source_url, flags=re.I):
            rejected.append(_rejection("fake_or_placeholder_source", card))
    if _looks_bad_text(distilled):
        reason = "internal_or_claim_like_text" if INTERNAL_OR_CLAIM_PATTERNS.search(distilled) else "navigation_or_low_quality_text"
        rejected.append(_rejection(reason, card))
    if fact_type == "metric":
        missing = _metric_missing_fields(card)
        if missing:
            rejected.append({**_rejection("metric_missing_scope_or_period", card), "missing_fields": missing})
    if card.get("source_title_url_mismatch_suspected"):
        rejected.append(_rejection("source_mismatch", card))

    if rejected:
        return None, rejected

    card.setdefault("metric", card.get("variable"))
    card.setdefault("source", card.get("source_title") or card_source_url or card_source_ref)
    card.setdefault("source_title", card.get("source_title") or card.get("title") or "")
    card.setdefault("url", card_source_url)
    card.setdefault("evidence_origin", "readpage_fact_extractor")
    card.setdefault("extraction_schema_version", SCHEMA_VERSION)
    card.setdefault("source_verified", card["source_verification_status"] in {"readpage_verified", "document_verified"})
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
        "dimension_id",
        "dimension_name",
        "evidence_goal",
        "must_have_terms",
        "forbidden_terms",
        "source_priority",
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
        "dimension_id",
        "dimension_name",
        "task_id",
        "evidence_goal",
        "must_have_terms",
        "forbidden_terms",
        "source_priority",
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
    candidates = re.split(r"(?<=[。！？.!?])\s+|\n+", text)
    payload_cards: List[Dict[str, Any]] = []
    for sentence in candidates:
        cleaned = _compact_text(sentence, max_chars=220)
        if _looks_bad_text(cleaned):
            continue
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
        "fact_cards": [],
        "rejected_spans": [],
        "errors": [],
        "model": "",
        "budget_limit": _budget_limit(),
        "budget_used": 0,
        "budget_exhausted": False,
    }
    if not selected_pages:
        result["status"] = "no_readpage_text"
        return result
    if not enabled:
        result["status"] = "disabled"
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
                consumed_budget, budget_exhausted = _try_consume_budget()
                if not consumed_budget:
                    result["budget_exhausted"] = bool(budget_exhausted)
                    validated = _fallback_fact_cards_from_page(page, search_task=task, proof_role=proof_role)
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
                    _store_cache(key, _cacheable_extractor_payload(validated))
                except Exception as exc:
                    result["llm_error_count"] += 1
                    result["errors"].append(str(exc))
                    validated = _fallback_fact_cards_from_page(page, search_task=task, proof_role=proof_role)
                    result["fallback_used"] = True
            else:
                validated = _fallback_fact_cards_from_page(page, search_task=task, proof_role=proof_role)
                result["fallback_used"] = True
        cards = _as_list(validated.get("fact_cards"))
        if cards:
            result["success_count"] += 1
            result["fact_cards"].extend([item for item in cards if isinstance(item, dict)])
        rejected = [item for item in _as_list(validated.get("rejected_spans")) if isinstance(item, dict)]
        result["rejected_spans"].extend(rejected)
        result["invalid_metric_count"] += int(validated.get("invalid_metric_count") or 0)
    result["fact_card_count"] = len(result["fact_cards"])
    result["rejected_span_count"] = len(result["rejected_spans"])
    result["budget_used"] = max(0, _budget_used() - start_budget_used)
    if result["budget_exhausted"] and result["fact_card_count"]:
        result["status"] = "partial_budget_exhausted"
    elif result["budget_exhausted"]:
        result["status"] = "budget_exhausted"
    else:
        result["status"] = "success" if result["fact_card_count"] else ("fallback_empty" if result["fallback_used"] else "no_valid_fact_cards")
    return result
