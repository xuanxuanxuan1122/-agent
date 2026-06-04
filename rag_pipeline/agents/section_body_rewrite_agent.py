from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..config.search_config import build_llm_config_for_task, build_llm_config_from_profile
from ..search.memory import call_openai_compatible_json, llm_config_is_ready


PROMPT_VERSION = "section_body_rewrite_v1"
DEFAULT_CACHE_PATH = Path("output/cache/section_body_rewrite")
FORBIDDEN_RE = re.compile(
    r"QA\s*failed|Clean\s*资格|fatal|EV-|evidence_cards|URL:|"
    r"补证|建议补证|证据不足|内部诊断|质量总分|Clean",
    re.I,
)


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, min_value: int = 0, max_value: int = 10_000) -> int:
    try:
        value = int(float(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        value = default
    return max(min_value, min(max_value, value))


def body_rewrite_enabled() -> bool:
    return _env_flag("REPORT_ENABLE_LLM_BODY_REWRITE", False)


def body_rewrite_cache_enabled() -> bool:
    return _env_flag("REPORT_BODY_REWRITE_CACHE_ENABLED", True)


def body_rewrite_max_sections() -> int:
    value = _env_int("REPORT_BODY_REWRITE_MAX_SECTIONS", 12, min_value=0, max_value=200)
    quality_mode = str(os.getenv("REPORT_QUALITY_MODE") or "").strip().lower()
    replay_mode = str(os.getenv("REPORT_REPLAY_EXECUTION_MODE") or "").strip().lower()
    if quality_mode == "high" or replay_mode == "quality_llm_replay":
        value = max(value, 24)
    return value


def body_rewrite_max_elapsed_seconds() -> int:
    return _env_int("REPORT_BODY_REWRITE_MAX_ELAPSED_SECONDS", 120, min_value=0, max_value=3600)


def body_rewrite_concurrency() -> int:
    return _env_int("REPORT_BODY_REWRITE_CONCURRENCY", 3, min_value=1, max_value=8)


def body_rewrite_target_section_chars() -> int:
    return _env_int("REPORT_BODY_REWRITE_TARGET_SECTION_CHARS", 650, min_value=0, max_value=5000)


def body_rewrite_max_expansion_ratio() -> float:
    raw = os.getenv("REPORT_BODY_REWRITE_MAX_EXPANSION_RATIO", "").strip()
    try:
        value = float(raw) if raw else 5.0
    except Exception:
        value = 5.0
    return max(1.0, min(5.0, value))


def _cache_root() -> Path:
    raw = os.getenv("REPORT_BODY_REWRITE_CACHE_PATH", "").strip()
    return Path(raw) if raw else DEFAULT_CACHE_PATH


def _cache_key(section: Dict[str, Any], facts: Sequence[Dict[str, Any]], chapter_question: str) -> str:
    payload = {
        "prompt_version": PROMPT_VERSION,
        "section_id": section.get("section_id"),
        "chapter_id": section.get("chapter_id"),
        "block_type": section.get("block_type"),
        "fact_refs": _required_fact_refs(section, facts),
        "composer_paragraph": _original_paragraph(section),
        "chapter_question": chapter_question,
        "target_chars": body_rewrite_target_section_chars(),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()


def _cache_path(key: str) -> Path:
    return _cache_root() / f"{key}.json"


def _load_cache(key: str) -> Optional[Dict[str, Any]]:
    if not body_rewrite_cache_enabled():
        return None
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _write_cache(key: str, payload: Dict[str, Any]) -> None:
    if not body_rewrite_cache_enabled():
        return
    path = _cache_path(key)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_path, path)
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        return


def _dedupe(values: Iterable[Any], *, limit: int = 20) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        text = _text(value)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def _original_paragraph(section: Dict[str, Any]) -> str:
    text = _text(section.get("composed_paragraph"))
    if text:
        return text
    for block in _as_list(section.get("render_blocks")):
        payload = _as_dict(block)
        if str(payload.get("type") or "") == "paragraph":
            text = _text(payload.get("text"))
            if text:
                return text
    for key in ("paragraph", "reasoning", "claim", "mechanism"):
        text = _text(section.get(key))
        if text:
            return text
    return ""


def _required_fact_refs(section: Dict[str, Any], facts: Sequence[Dict[str, Any]]) -> List[str]:
    section_fact_refs = _as_list(section.get("used_fact_refs"))
    if not section_fact_refs:
        section_fact_refs = [
            ref for ref in _as_list(section.get("evidence_refs"))
            if not re.fullmatch(r"\[\d{1,5}\]", _text(ref))
        ]
    refs = _dedupe(
        [
            *section_fact_refs,
            *[item.get("evidence_id") for item in facts if isinstance(item, dict)],
        ],
        limit=12,
    )
    return refs


def _required_citation_refs(section: Dict[str, Any], facts: Sequence[Dict[str, Any]]) -> List[str]:
    def citation(value: Any) -> str:
        text = _text(value)
        return text if re.fullmatch(r"\[\d{1,5}\]", text) else ""

    return _dedupe(
        [
            *_as_list(section.get("citation_refs")),
            *[citation(item.get("source_ref")) for item in facts if isinstance(item, dict)],
        ],
        limit=12,
    )


def _numbers(text: str) -> set[str]:
    cleaned = re.sub(r"\[\d{1,5}\]", " ", str(text or ""))
    return set(re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?%?", cleaned))


def _source_text(section: Dict[str, Any], facts: Sequence[Dict[str, Any]]) -> str:
    fact_text = " ".join(
        _text(item.get(key))
        for item in facts
        if isinstance(item, dict)
        for key in ("distilled_fact", "fact", "subject", "variable", "value", "time_or_scope")
    )
    return " ".join(
        [
            _original_paragraph(section),
            _text(section.get("claim")),
            _text(section.get("reasoning")),
            _text(section.get("mechanism")),
            _text(section.get("counter_evidence")),
            fact_text,
            " ".join(_required_citation_refs(section, facts)),
        ]
    )


def _fallback(
    section: Dict[str, Any],
    facts: Sequence[Dict[str, Any]],
    *,
    status: str = "fallback",
    reason: str = "",
    model: str = "",
) -> Dict[str, Any]:
    return {
        "status": status,
        "paragraph": _original_paragraph(section),
        "original_paragraph": _original_paragraph(section),
        "failure_reason": reason,
        "model": model,
        "input_ref_count": len(_required_fact_refs(section, facts)),
        "output_ref_count": 0,
        "used_fact_refs": _required_fact_refs(section, facts),
        "citation_refs": _required_citation_refs(section, facts),
        "llm_called": False,
        "cache_hit": False,
    }


def ensure_public_render_blocks(section: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(section or {})
    blocks = [dict(block) for block in _as_list(payload.get("render_blocks")) if isinstance(block, dict)]
    if any(str(block.get("type") or "") == "paragraph" and _text(block.get("text")) for block in blocks):
        return payload

    paragraph = ""
    for key in ("composed_paragraph", "paragraph", "mechanism", "reasoning", "claim"):
        paragraph = _text(payload.get(key))
        if paragraph:
            break
    if not paragraph:
        return payload

    non_paragraph_blocks = [block for block in blocks if str(block.get("type") or "") != "paragraph"]
    payload["render_blocks"] = [{"type": "paragraph", "label": "", "text": paragraph}, *non_paragraph_blocks]
    payload["render_blocks_recovered"] = True
    return payload


def _validate_candidate(
    *,
    paragraph: str,
    used_fact_refs: Sequence[Any],
    citation_refs: Sequence[Any],
    section: Dict[str, Any],
    facts: Sequence[Dict[str, Any]],
) -> Tuple[bool, str]:
    original = _original_paragraph(section)
    required_fact_refs = set(_required_fact_refs(section, facts))
    output_fact_refs = set(_dedupe(used_fact_refs))
    if required_fact_refs and not required_fact_refs.issubset(output_fact_refs):
        return False, "missing_required_refs"
    required_citations = set(_required_citation_refs(section, facts))
    output_citations = set(_dedupe(citation_refs))
    if required_citations and not required_citations.issubset(output_citations):
        return False, "missing_required_citations"
    if not paragraph:
        return False, "empty_paragraph"
    if FORBIDDEN_RE.search(paragraph):
        return False, "forbidden_public_text"
    allowed_numbers = _numbers(_source_text(section, facts))
    new_numbers = _numbers(paragraph) - allowed_numbers
    if new_numbers:
        return False, "new_numeric_claim"
    original_len = len(re.sub(r"\s+", "", original))
    candidate_len = len(re.sub(r"\s+", "", paragraph))
    min_safe_chars = _env_int("REPORT_BODY_REWRITE_MIN_ACCEPT_CHARS", 80, min_value=20, max_value=2000)
    try:
        min_ratio = float(os.getenv("REPORT_BODY_REWRITE_MIN_COMPRESSION_RATIO", "0.35") or "0.35")
    except ValueError:
        min_ratio = 0.35
    if original_len >= 30 and candidate_len < min_safe_chars and candidate_len < int(original_len * min_ratio):
        return False, "output_too_short"
    max_ratio = body_rewrite_max_expansion_ratio()
    if original_len >= 30 and candidate_len > int(original_len * max_ratio):
        return False, "output_too_long"
    return True, ""


def _llm_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    result = dict(config or build_llm_config_for_task("body_rewrite"))
    timeout = _env_int("REPORT_BODY_REWRITE_TIMEOUT_SECONDS", 90, min_value=1, max_value=600)
    if timeout:
        result["timeout"] = timeout
    return result


def _fallback_llm_configs(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    embedded = config.get("fallback_config") if isinstance(config, dict) else None
    if isinstance(embedded, dict):
        configs.append(dict(embedded))
    profile_values = [
        os.getenv("REPORT_BODY_REWRITE_FALLBACK_PROFILES", ""),
    ]
    seen = {
        (
            str(config.get("profile") or "").strip().lower(),
            str(config.get("model") or "").strip().lower(),
        )
    }
    for raw in profile_values:
        for profile in re.split(r"[,;]\s*", str(raw or "").strip()):
            profile = profile.strip()
            if not profile:
                continue
            try:
                candidate = build_llm_config_from_profile(profile, default_timeout=float(config.get("timeout") or 90))
            except Exception:
                continue
            key = (
                str(candidate.get("profile") or "").strip().lower(),
                str(candidate.get("model") or "").strip().lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            configs.append(candidate)
    return [candidate for candidate in configs if llm_config_is_ready(candidate)]


def _system_prompt() -> str:
    return (
        "You are a section-level industry research writing editor. Rewrite only the given "
        "composer paragraph into a polished Chinese industry-research paragraph. Use only "
        "the provided facts. Do not add companies, numbers, sources, claims, or citations. "
        "If target_chars is provided, expand toward that length with mechanism, industry meaning, "
        "and boundary analysis that is grounded only in the supplied facts. "
        "Do not change claim strength. Preserve all used_fact_refs and citation_refs exactly. "
        "Return JSON only: {\"paragraph\":\"...\", \"used_fact_refs\":[...], \"citation_refs\":[...]}."
    )


def rewrite_section_body(
    *,
    section: Dict[str, Any],
    facts: Sequence[Dict[str, Any]],
    chapter_question: str = "",
    llm_config: Optional[Dict[str, Any]] = None,
    allow_llm_call: bool = True,
) -> Dict[str, Any]:
    section = _as_dict(section)
    facts = [dict(item) for item in facts if isinstance(item, dict)]
    if not body_rewrite_enabled():
        return _fallback(section, facts, status="skipped", reason="disabled")
    if str(section.get("body_composition_status") or "") not in {"composed", "composed_directional"}:
        return _fallback(section, facts, status="skipped", reason="not_composed")
    if not _required_fact_refs(section, facts):
        return _fallback(section, facts, status="skipped", reason="missing_used_fact_refs")

    key = _cache_key(section, facts, chapter_question)
    cached = _load_cache(key)
    if cached:
        valid, reason = _validate_candidate(
            paragraph=_text(cached.get("paragraph")),
            used_fact_refs=_as_list(cached.get("used_fact_refs")),
            citation_refs=_as_list(cached.get("citation_refs")),
            section=section,
            facts=facts,
        )
        if valid:
            return {
                "status": "cached",
                "paragraph": _text(cached.get("paragraph")),
                "original_paragraph": _original_paragraph(section),
                "failure_reason": "",
                "model": _text(cached.get("model")),
                "input_ref_count": len(_required_fact_refs(section, facts)),
                "output_ref_count": len(_as_list(cached.get("used_fact_refs"))),
                "used_fact_refs": _dedupe(cached.get("used_fact_refs") or []),
                "citation_refs": _dedupe(cached.get("citation_refs") or []),
                "llm_called": False,
                "cache_hit": True,
            }
        return _fallback(section, facts, status="rejected", reason=reason)

    if not allow_llm_call:
        return _fallback(section, facts, status="skipped", reason="budget_exhausted")

    config = _llm_config(llm_config)
    model = _text(config.get("model") or config.get("profile"))
    if not llm_config_is_ready(config):
        return _fallback(section, facts, reason="llm_config_not_ready", model=model)

    payload = {
        "chapter_question": chapter_question,
        "section_title": section.get("section_title"),
        "block_type": section.get("block_type"),
        "claim_strength": section.get("claim_strength") or section.get("confidence"),
        "composer_paragraph": _original_paragraph(section),
        "target_chars": body_rewrite_target_section_chars(),
        "facts": facts,
        "used_fact_refs": _required_fact_refs(section, facts),
        "citation_refs": _required_citation_refs(section, facts),
    }
    fallback_used = False
    attempted_errors: List[str] = []
    response: Dict[str, Any] = {}
    active_model = model
    for attempt_index, attempt_config in enumerate([config, *_fallback_llm_configs(config)]):
        active_model = _text(attempt_config.get("model") or attempt_config.get("profile"))
        try:
            response = call_openai_compatible_json(
                config=attempt_config,
                system_prompt=_system_prompt(),
                user_payload=payload,
            )
            fallback_used = attempt_index > 0
            break
        except Exception as exc:
            attempted_errors.append(f"{active_model or 'unknown'}:{type(exc).__name__}")
            continue
    if not response:
        failed = _fallback(section, facts, reason=f"llm_error:{attempted_errors[-1].split(':')[-1] if attempted_errors else 'Exception'}", model=model)
        failed["llm_called"] = True
        if attempted_errors:
            failed["llm_error_attempts"] = attempted_errors
        return failed

    result = _as_dict(response.get("payload"))
    paragraph = _text(result.get("paragraph"))
    used_fact_refs = _as_list(result.get("used_fact_refs"))
    citation_refs = _as_list(result.get("citation_refs"))
    valid, reason = _validate_candidate(
        paragraph=paragraph,
        used_fact_refs=used_fact_refs,
        citation_refs=citation_refs,
        section=section,
        facts=facts,
    )
    if not valid:
        rejected = _fallback(section, facts, status="rejected", reason=reason, model=active_model)
        rejected["llm_called"] = True
        rejected["fallback_used"] = fallback_used
        return rejected
    output = {
        "status": "rewritten",
        "paragraph": paragraph,
        "original_paragraph": _original_paragraph(section),
        "failure_reason": "",
        "model": active_model,
        "input_ref_count": len(_required_fact_refs(section, facts)),
        "output_ref_count": len(_dedupe(used_fact_refs)),
        "used_fact_refs": _dedupe(used_fact_refs),
        "citation_refs": _dedupe(citation_refs),
        "llm_called": True,
        "cache_hit": False,
        "fallback_used": fallback_used,
    }
    _write_cache(key, output)
    return output


def rewrite_sections_for_chapter(
    *,
    sections: Sequence[Dict[str, Any]],
    chapter_question: str = "",
    llm_config: Optional[Dict[str, Any]] = None,
    max_llm_calls: int = 12,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    diagnostics = {
        "enabled": body_rewrite_enabled(),
        "called_count": 0,
        "success_count": 0,
        "cache_hit_count": 0,
        "rejected_count": 0,
        "fallback_count": 0,
        "skipped_count": 0,
        "failure_reasons": {},
        "output_too_long_allowed_count": 0,
    }
    if not diagnostics["enabled"]:
        return [dict(section) for section in sections], diagnostics

    rewritten_sections: List[Dict[str, Any]] = []
    started_at = time.monotonic()
    max_elapsed = body_rewrite_max_elapsed_seconds()
    for section in sections:
        payload = ensure_public_render_blocks(dict(section))
        facts = [
            {
                "evidence_id": ref,
                "distilled_fact": fact,
                "source_ref": ref,
            }
            for ref, fact in zip(_as_list(payload.get("used_fact_refs")), _as_list(payload.get("supporting_facts")))
        ]
        elapsed_exhausted = bool(max_elapsed and (time.monotonic() - started_at) >= max_elapsed)
        allow_call = diagnostics["called_count"] < max_llm_calls and not elapsed_exhausted
        result = rewrite_section_body(
            section=payload,
            facts=facts,
            chapter_question=chapter_question,
            llm_config=llm_config,
            allow_llm_call=allow_call,
        )
        status = str(result.get("status") or "")
        if result.get("llm_called"):
            diagnostics["called_count"] += 1
        if result.get("cache_hit"):
            diagnostics["cache_hit_count"] += 1
        if status in {"rewritten", "cached"}:
            diagnostics["success_count"] += 1
            paragraph = _text(result.get("paragraph"))
            payload["claim"] = paragraph
            payload["reasoning"] = paragraph
            payload["mechanism"] = paragraph
            payload["render_blocks"] = [{"type": "paragraph", "label": "", "text": paragraph}]
        elif status == "rejected":
            diagnostics["rejected_count"] += 1
        elif status == "skipped":
            diagnostics["skipped_count"] += 1
        else:
            diagnostics["fallback_count"] += 1
        reason = str(result.get("failure_reason") or "")
        if reason:
            diagnostics["failure_reasons"][reason] = diagnostics["failure_reasons"].get(reason, 0) + 1
        payload["body_rewrite_status"] = status
        payload["body_rewrite"] = result
        if status not in {"rewritten", "cached"}:
            payload = ensure_public_render_blocks(payload)
        rewritten_sections.append(payload)
    return rewritten_sections, diagnostics


def _facts_for_section(section: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "evidence_id": ref,
            "distilled_fact": fact,
            "source_ref": ref,
        }
        for ref, fact in zip(_as_list(section.get("used_fact_refs")), _as_list(section.get("supporting_facts")))
    ]


def _new_diagnostics(*, enabled: Optional[bool] = None) -> Dict[str, Any]:
    return {
        "enabled": body_rewrite_enabled() if enabled is None else bool(enabled),
        "called_count": 0,
        "submitted_count": 0,
        "success_count": 0,
        "cache_hit_count": 0,
        "rejected_count": 0,
        "fallback_count": 0,
        "skipped_count": 0,
        "budget_exhausted_count": 0,
        "inflight_dedup_count": 0,
        "output_too_long_allowed_count": 0,
        "max_llm_calls": 0,
        "elapsed_seconds": 0.0,
        "concurrency": body_rewrite_concurrency(),
        "failure_reasons": {},
    }


def _record_result(diagnostics: Dict[str, Any], result: Dict[str, Any]) -> None:
    status = str(result.get("status") or "")
    if result.get("llm_called"):
        diagnostics["called_count"] += 1
    if result.get("cache_hit"):
        diagnostics["cache_hit_count"] += 1
    if status in {"rewritten", "cached"}:
        diagnostics["success_count"] += 1
    elif status == "rejected":
        diagnostics["rejected_count"] += 1
    elif status == "skipped":
        diagnostics["skipped_count"] += 1
    else:
        diagnostics["fallback_count"] += 1
    reason = str(result.get("failure_reason") or "")
    if reason:
        diagnostics["failure_reasons"][reason] = diagnostics["failure_reasons"].get(reason, 0) + 1
    if status in {"rewritten", "cached"}:
        original_len = len(re.sub(r"\s+", "", str(result.get("original_paragraph") or "")))
        rewritten_len = len(re.sub(r"\s+", "", str(result.get("paragraph") or "")))
        if original_len >= 30 and rewritten_len > int(original_len * 1.8):
            diagnostics["output_too_long_allowed_count"] = int(diagnostics.get("output_too_long_allowed_count") or 0) + 1
    if reason == "budget_exhausted":
        diagnostics["budget_exhausted_count"] += 1


def _merge_diagnostics(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    for key in (
        "called_count",
        "submitted_count",
        "success_count",
        "cache_hit_count",
        "rejected_count",
        "fallback_count",
        "skipped_count",
        "budget_exhausted_count",
        "inflight_dedup_count",
        "output_too_long_allowed_count",
    ):
        target[key] = int(target.get(key) or 0) + int(source.get(key) or 0)
    target["elapsed_seconds"] = max(float(target.get("elapsed_seconds") or 0.0), float(source.get("elapsed_seconds") or 0.0))
    target["concurrency"] = max(int(target.get("concurrency") or 1), int(source.get("concurrency") or 1))
    target["max_llm_calls"] = max(int(target.get("max_llm_calls") or 0), int(source.get("max_llm_calls") or 0))
    target["enabled"] = bool(target.get("enabled") or source.get("enabled"))
    for reason, count in _as_dict(source.get("failure_reasons")).items():
        target["failure_reasons"][reason] = target["failure_reasons"].get(reason, 0) + int(count or 0)


def _apply_rewrite_result(section: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    payload = ensure_public_render_blocks(dict(section))
    status = str(result.get("status") or "")
    if status in {"rewritten", "cached"}:
        paragraph = _text(result.get("paragraph"))
        payload["claim"] = paragraph
        payload["reasoning"] = paragraph
        payload["mechanism"] = paragraph
        payload["render_blocks"] = [{"type": "paragraph", "label": "", "text": paragraph}]
    else:
        payload = ensure_public_render_blocks(payload)
    payload["body_rewrite_status"] = status
    payload["body_rewrite"] = result
    return payload


def _cached_result_for_section(
    *,
    section: Dict[str, Any],
    facts: Sequence[Dict[str, Any]],
    chapter_question: str,
) -> Optional[Dict[str, Any]]:
    key = _cache_key(section, facts, chapter_question)
    cached = _load_cache(key)
    if not cached:
        return None
    valid, reason = _validate_candidate(
        paragraph=_text(cached.get("paragraph")),
        used_fact_refs=_as_list(cached.get("used_fact_refs")),
        citation_refs=_as_list(cached.get("citation_refs")),
        section=section,
        facts=facts,
    )
    if not valid:
        return _fallback(section, facts, status="rejected", reason=reason)
    return {
        "status": "cached",
        "paragraph": _text(cached.get("paragraph")),
        "original_paragraph": _original_paragraph(section),
        "failure_reason": "",
        "model": _text(cached.get("model")),
        "input_ref_count": len(_required_fact_refs(section, facts)),
        "output_ref_count": len(_as_list(cached.get("used_fact_refs"))),
        "used_fact_refs": _dedupe(cached.get("used_fact_refs") or []),
        "citation_refs": _dedupe(cached.get("citation_refs") or []),
        "llm_called": False,
        "cache_hit": True,
    }


def _preflight_result_for_section(
    *,
    section: Dict[str, Any],
    facts: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not body_rewrite_enabled():
        return _fallback(section, facts, status="skipped", reason="disabled")
    if str(section.get("body_composition_status") or "") not in {"composed", "composed_directional"}:
        return _fallback(section, facts, status="skipped", reason="not_composed")
    if not _required_fact_refs(section, facts):
        return _fallback(section, facts, status="skipped", reason="missing_used_fact_refs")
    return None


def rewrite_sections_for_report(
    *,
    chapter_packages: Sequence[Dict[str, Any]],
    llm_config: Optional[Dict[str, Any]] = None,
    max_llm_calls: int = 12,
    concurrency: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    diagnostics = _new_diagnostics()
    max_llm_calls = max(0, int(max_llm_calls or 0))
    concurrency = max(1, min(8, int(concurrency or body_rewrite_concurrency())))
    diagnostics["concurrency"] = concurrency
    diagnostics["max_llm_calls"] = max_llm_calls

    packages = [dict(package) for package in chapter_packages if isinstance(package, dict)]
    if not diagnostics["enabled"]:
        return packages, diagnostics

    started_at = time.monotonic()
    max_elapsed = body_rewrite_max_elapsed_seconds()
    chapter_diagnostics = [_new_diagnostics(enabled=True) for _ in packages]
    for item in chapter_diagnostics:
        item["concurrency"] = concurrency
        item["max_llm_calls"] = max_llm_calls

    section_positions: List[Tuple[int, int, Dict[str, Any], List[Dict[str, Any]], str, str]] = []
    rewritten_sections_by_chapter: List[List[Dict[str, Any]]] = []
    for chapter_index, package in enumerate(packages):
        chapter_question = _text(package.get("chapter_question") or package.get("chapter_title"))
        sections = [dict(section) for section in _as_list(package.get("sections")) if isinstance(section, dict)]
        rewritten_sections_by_chapter.append(sections)
        for section_index, section in enumerate(sections):
            facts = _facts_for_section(section)
            preflight = _preflight_result_for_section(section=section, facts=facts)
            if preflight:
                rewritten_sections_by_chapter[chapter_index][section_index] = _apply_rewrite_result(section, preflight)
                _record_result(chapter_diagnostics[chapter_index], preflight)
                continue
            cached = _cached_result_for_section(section=section, facts=facts, chapter_question=chapter_question)
            if cached:
                rewritten_sections_by_chapter[chapter_index][section_index] = _apply_rewrite_result(section, cached)
                _record_result(chapter_diagnostics[chapter_index], cached)
                continue
            key = _cache_key(section, facts, chapter_question)
            section_positions.append((chapter_index, section_index, section, facts, chapter_question, key))

    futures: Dict[Future, str] = {}
    key_to_positions: Dict[str, List[Tuple[int, int, Dict[str, Any], List[Dict[str, Any]]]]] = {}
    queue: List[Tuple[str, int, int, Dict[str, Any], List[Dict[str, Any]], str]] = []
    submitted_keys: set[str] = set()
    for chapter_index, section_index, section, facts, chapter_question, key in section_positions:
        if key in key_to_positions:
            key_to_positions[key].append((chapter_index, section_index, section, facts))
            chapter_diagnostics[chapter_index]["inflight_dedup_count"] += 1
            continue
        if len(submitted_keys) >= max_llm_calls:
            result = _fallback(section, facts, status="skipped", reason="budget_exhausted")
            rewritten_sections_by_chapter[chapter_index][section_index] = _apply_rewrite_result(section, result)
            _record_result(chapter_diagnostics[chapter_index], result)
            continue
        key_to_positions[key] = [(chapter_index, section_index, section, facts)]
        submitted_keys.add(key)
        queue.append((key, chapter_index, section_index, section, facts, chapter_question))

    def elapsed_exhausted() -> bool:
        return bool(max_elapsed and (time.monotonic() - started_at) >= max_elapsed)

    def mark_budget_exhausted(items: Sequence[Tuple[str, int, int, Dict[str, Any], List[Dict[str, Any]], str]]) -> None:
        for key, chapter_index, section_index, section, facts, _chapter_question in items:
            result = _fallback(section, facts, status="skipped", reason="budget_exhausted")
            for dup_chapter_index, dup_section_index, dup_section, dup_facts in key_to_positions.get(key, []):
                dup_result = result if dup_section is section else _fallback(dup_section, dup_facts, status="skipped", reason="budget_exhausted")
                rewritten_sections_by_chapter[dup_chapter_index][dup_section_index] = _apply_rewrite_result(dup_section, dup_result)
                _record_result(chapter_diagnostics[dup_chapter_index], dup_result)

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        pending_queue = list(queue)
        while pending_queue and len(futures) < concurrency and not elapsed_exhausted():
            key, chapter_index, _section_index, section, facts, chapter_question = pending_queue.pop(0)
            future = executor.submit(
                rewrite_section_body,
                section=section,
                facts=facts,
                chapter_question=chapter_question,
                llm_config=llm_config,
                allow_llm_call=True,
            )
            futures[future] = key
            chapter_diagnostics[chapter_index]["submitted_count"] += 1

        while futures:
            done, _not_done = wait(set(futures.keys()), timeout=0.05, return_when=FIRST_COMPLETED)
            if not done:
                continue
            for future in done:
                key = futures.pop(future)
                positions = key_to_positions.get(key, [])
                try:
                    result = future.result()
                except Exception as exc:
                    first_section = positions[0][2] if positions else {}
                    first_facts = positions[0][3] if positions else []
                    result = _fallback(first_section, first_facts, reason=f"llm_error:{type(exc).__name__}")
                    result["llm_called"] = True
                for dup_chapter_index, dup_section_index, dup_section, dup_facts in positions:
                    dup_result = result if dup_section is positions[0][2] else dict(result)
                    if dup_section is not positions[0][2]:
                        dup_result["llm_called"] = False
                        dup_result["inflight_dedup"] = True
                    if dup_result.get("status") not in {"rewritten", "cached"} and dup_section is not positions[0][2]:
                        dup_result = _fallback(dup_section, dup_facts, status=str(result.get("status") or "fallback"), reason=str(result.get("failure_reason") or ""))
                        dup_result["llm_called"] = False
                        dup_result["inflight_dedup"] = True
                    rewritten_sections_by_chapter[dup_chapter_index][dup_section_index] = _apply_rewrite_result(dup_section, dup_result)
                    _record_result(chapter_diagnostics[dup_chapter_index], dup_result)
            while pending_queue and len(futures) < concurrency and not elapsed_exhausted():
                key, chapter_index, _section_index, section, facts, chapter_question = pending_queue.pop(0)
                future = executor.submit(
                    rewrite_section_body,
                    section=section,
                    facts=facts,
                    chapter_question=chapter_question,
                    llm_config=llm_config,
                    allow_llm_call=True,
                )
                futures[future] = key
                chapter_diagnostics[chapter_index]["submitted_count"] += 1
        if pending_queue:
            mark_budget_exhausted(pending_queue)
        for future in list(futures.keys()):
            key = futures.pop(future)
            positions = key_to_positions.get(key, [])
            try:
                result = future.result(timeout=0)
            except Exception:
                for dup_chapter_index, dup_section_index, dup_section, dup_facts in positions:
                    dup_result = _fallback(dup_section, dup_facts, status="skipped", reason="budget_exhausted")
                    rewritten_sections_by_chapter[dup_chapter_index][dup_section_index] = _apply_rewrite_result(dup_section, dup_result)
                    _record_result(chapter_diagnostics[dup_chapter_index], dup_result)
                continue
            for dup_chapter_index, dup_section_index, dup_section, dup_facts in positions:
                dup_result = result if dup_section is positions[0][2] else dict(result)
                if dup_section is not positions[0][2]:
                    dup_result["llm_called"] = False
                    dup_result["inflight_dedup"] = True
                if dup_result.get("status") not in {"rewritten", "cached"} and dup_section is not positions[0][2]:
                    dup_result = _fallback(dup_section, dup_facts, status=str(result.get("status") or "fallback"), reason=str(result.get("failure_reason") or ""))
                    dup_result["llm_called"] = False
                    dup_result["inflight_dedup"] = True
                rewritten_sections_by_chapter[dup_chapter_index][dup_section_index] = _apply_rewrite_result(dup_section, dup_result)
                _record_result(chapter_diagnostics[dup_chapter_index], dup_result)

    for index, package in enumerate(packages):
        package["sections"] = rewritten_sections_by_chapter[index]
        chapter_diagnostics[index]["elapsed_seconds"] = round(time.monotonic() - started_at, 3)
        package["body_rewrite"] = chapter_diagnostics[index]
        _merge_diagnostics(diagnostics, chapter_diagnostics[index])
    diagnostics["elapsed_seconds"] = round(time.monotonic() - started_at, 3)
    if packages:
        packages[0]["body_rewrite_global"] = diagnostics
    return packages, diagnostics
