from __future__ import annotations

import json
import os
import re
import hashlib
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4

from ..config.search_config import DEFAULT_EXTERNAL_API_TRUST_ENV, DEFAULT_LLM_DISABLE_THINKING
from ..telemetry.context_budget import (
    ContextBudgetExceededError,
    assert_llm_input_budget,
    compact_json,
)
from ..telemetry.token_usage import record_llm_usage
from .models import Turn


_PRONOUN_RE = re.compile(r"(?:^|\b)(it|they|them|that|this|he|she|its|their)\b|[它那个这该其他们她们该公司该行业该技术]", re.I)


_UNAVAILABLE_LLM_KEYS: set[str] = set()


class LLMCallError(RuntimeError):
    """RuntimeError with non-secret call diagnostics attached."""

    def __init__(self, message: str, *, diagnostic: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.diagnostic = dict(diagnostic or {})


def generate_turn_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]


def _json_default(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def _normalize_bool_flag(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _build_url_opener(*, trust_env: bool) -> urllib.request.OpenerDirector:
    if trust_env:
        return urllib.request.build_opener()
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def normalize_llm_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    config = dict(config or {})
    max_output_tokens = 0
    try:
        max_output_tokens = max(0, int(config.get("max_output_tokens") or 0))
    except (TypeError, ValueError):
        max_output_tokens = 0
    model = str(config.get("model") or "").strip()
    disable_thinking = _normalize_bool_flag(config.get("disable_thinking"), DEFAULT_LLM_DISABLE_THINKING)
    if max_output_tokens <= 0 and model.lower().startswith("deepseek-v4-pro"):
        try:
            max_output_tokens = max(
                512,
                int(os.getenv("RAG_DEEPSEEK_V4_PRO_DEFAULT_MAX_OUTPUT_TOKENS", "2048")),
            )
        except ValueError:
            max_output_tokens = 2048
    return {
        "provider": str(config.get("provider") or "openai_compatible").strip().lower() or "openai_compatible",
        "url": str(config.get("url") or "").strip(),
        "api_key": str(config.get("api_key") or "").strip(),
        "model": model,
        "timeout": float(config.get("timeout") or 30.0),
        "trust_env": _normalize_bool_flag(config.get("trust_env"), DEFAULT_EXTERNAL_API_TRUST_ENV),
        "disable_thinking": disable_thinking,
        "reasoning_effort": str(config.get("reasoning_effort") or "").strip().lower(),
        "max_output_tokens": max_output_tokens,
        "api_mode": str(config.get("api_mode") or "").strip().lower(),
        "task_name": str(config.get("task_name") or config.get("task") or "").strip(),
        "profile": str(config.get("profile") or "").strip(),
    }


def llm_config_is_ready(config: Optional[Dict[str, Any]]) -> bool:
    normalized = normalize_llm_config(config)
    return bool(normalized["url"] and normalized["api_key"] and normalized["model"])


def normalize_openai_compatible_chat_url(url: str) -> str:
    cleaned = str(url or "").strip().rstrip("/")
    if not cleaned:
        return ""
    if cleaned.endswith("/chat/completions"):
        return cleaned
    if cleaned.endswith("/v1"):
        return f"{cleaned}/chat/completions"
    return cleaned


def normalize_openai_responses_url(url: str) -> str:
    cleaned = str(url or "").strip().rstrip("/")
    if not cleaned:
        return ""
    if cleaned.endswith("/responses"):
        return cleaned
    if cleaned.endswith("/chat/completions"):
        return f"{cleaned[: -len('/chat/completions')]}/responses"
    if cleaned.endswith("/v1"):
        return f"{cleaned}/responses"
    return cleaned


def model_uses_default_temperature_only(model: str) -> bool:
    return str(model or "").strip().lower().startswith("gpt-5")


def _model_uses_default_temperature_only(model: str) -> bool:
    return model_uses_default_temperature_only(model)


def should_use_openai_responses_api(normalized: Dict[str, Any]) -> bool:
    mode = str(normalized.get("api_mode") or "").strip().lower()
    if mode in {"responses", "openai_responses"}:
        return True
    model = str(normalized.get("model") or "").strip().lower()
    return model.startswith("gpt-5.5")


def _llm_circuit_key(normalized: Dict[str, Any]) -> str:
    return "|".join(
        [
            str(normalized.get("provider") or ""),
            str(normalized.get("url") or "").rstrip("/"),
            str(normalized.get("model") or ""),
            str(normalized.get("profile") or ""),
        ]
    )


def _is_gpt55_config(normalized: Dict[str, Any]) -> bool:
    model = str(normalized.get("model") or "").strip().lower()
    profile = str(normalized.get("profile") or "").strip().lower()
    return model.startswith("gpt-5.5") or profile == "gpt-5.5"


def _fallback_config_for(config: Dict[str, Any], normalized: Dict[str, Any]) -> Dict[str, Any]:
    if not _is_gpt55_config(normalized):
        return {}
    fallback = config.get("fallback_config") if isinstance(config, dict) else None
    if not isinstance(fallback, dict):
        return {}
    fallback_normalized = normalize_llm_config(fallback)
    if not llm_config_is_ready(fallback_normalized):
        return {}
    if _llm_circuit_key(fallback_normalized) == _llm_circuit_key(normalized):
        return {}
    return dict(fallback)


def _llm_error_summary(exc: Exception) -> str:
    diagnostic = getattr(exc, "diagnostic", None)
    if isinstance(diagnostic, dict) and diagnostic.get("error"):
        return str(diagnostic.get("error") or "")[:1200]
    return str(exc)[:1200]


def _mark_primary_unavailable(normalized: Dict[str, Any], exc: Exception) -> None:
    del exc
    if _is_gpt55_config(normalized):
        _UNAVAILABLE_LLM_KEYS.add(_llm_circuit_key(normalized))


def _annotate_fallback_result(
    result: Dict[str, Any],
    *,
    primary: Dict[str, Any],
    fallback: Dict[str, Any],
    primary_error: str,
    primary_skipped: bool = False,
) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return result
    llm_call = dict(result.get("llm_call") or {})
    llm_call.update(
        {
            "fallback_used": True,
            "primary_skipped": bool(primary_skipped),
            "primary_profile": str(primary.get("profile") or ""),
            "primary_model": str(primary.get("model") or ""),
            "fallback_profile": str(fallback.get("profile") or ""),
            "fallback_model": str(fallback.get("model") or ""),
            "primary_error": str(primary_error or "")[:1200],
        }
    )
    result["llm_call"] = llm_call
    result["fallback_used"] = True
    return result


def _raise_fallback_error(primary_exc: Exception, fallback_exc: Exception, *, primary: Dict[str, Any], fallback: Dict[str, Any]) -> None:
    primary_error = _llm_error_summary(primary_exc)
    fallback_error = _llm_error_summary(fallback_exc)
    diagnostic = {
        "status": "failed",
        "fallback_used": True,
        "primary_profile": str(primary.get("profile") or ""),
        "primary_model": str(primary.get("model") or ""),
        "primary_error": primary_error,
        "fallback_profile": str(fallback.get("profile") or ""),
        "fallback_model": str(fallback.get("model") or ""),
        "fallback_error": fallback_error,
        "error": f"primary={primary_error}; fallback={fallback_error}",
    }
    raise LLMCallError(
        f"Primary LLM failed and fallback LLM failed. primary={primary_error}; fallback={fallback_error}",
        diagnostic=diagnostic,
    ) from fallback_exc


def apply_openai_compatible_reasoning_flags(payload: Dict[str, Any], normalized: Dict[str, Any]) -> None:
    if not normalized.get("disable_thinking"):
        return
    model = str(normalized.get("model") or "").strip().lower()
    if model.startswith("deepseek-v4"):
        payload["thinking"] = {"type": "disabled"}
        return
    payload["enable_thinking"] = False


def _apply_openai_compatible_reasoning_flags(payload: Dict[str, Any], normalized: Dict[str, Any]) -> None:
    apply_openai_compatible_reasoning_flags(payload, normalized)


def _apply_openai_responses_reasoning(payload: Dict[str, Any], normalized: Dict[str, Any]) -> None:
    effort = str(normalized.get("reasoning_effort") or "").strip().lower()
    if effort:
        payload["reasoning"] = {"effort": effort}


def _record_llm_usage_event(
    *,
    normalized: Dict[str, Any],
    usage: Any,
    api: str,
    started_at: float,
) -> None:
    try:
        record_llm_usage(
            usage=usage,
            provider=str(normalized.get("provider") or ""),
            model=str(normalized.get("model") or ""),
            task=str(normalized.get("task_name") or ""),
            profile=str(normalized.get("profile") or ""),
            api=api,
            elapsed_ms=max(0, int((time.perf_counter() - started_at) * 1000)),
        )
    except Exception:
        return


def _llm_call_diagnostic(
    *,
    normalized: Dict[str, Any],
    api: str,
    started_at: float,
    status: str,
    usage: Any = None,
    error: str = "",
) -> Dict[str, Any]:
    return {
        "task": str(normalized.get("task_name") or ""),
        "profile": str(normalized.get("profile") or ""),
        "model": str(normalized.get("model") or ""),
        "api": str(api or ""),
        "status": str(status or ""),
        "error": str(error or "")[:1200],
        "usage": usage if isinstance(usage, dict) else {},
        "elapsed_ms": max(0, int((time.perf_counter() - started_at) * 1000)),
    }


def _raise_llm_call_error(
    exc: Exception,
    *,
    normalized: Dict[str, Any],
    api: str,
    started_at: float,
) -> None:
    if isinstance(exc, LLMCallError):
        raise exc
    diagnostic = _llm_call_diagnostic(
        normalized=normalized,
        api=api,
        started_at=started_at,
        status="failed",
        error=str(exc),
    )
    raise LLMCallError(str(exc), diagnostic=diagnostic) from exc


def _post_llm_json(
    *,
    normalized: Dict[str, Any],
    url: str,
    payload: Dict[str, Any],
    error_prefix: str,
) -> Dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {normalized['api_key']}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    opener = _build_url_opener(trust_env=normalized["trust_env"])
    try:
        with opener.open(request, timeout=normalized["timeout"]) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{error_prefix} failed with HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{error_prefix} failed: {exc}") from exc
    return json.loads(raw)


def collect_openai_response_text(response: Dict[str, Any]) -> str:
    if str(response.get("output_text") or "").strip():
        return str(response.get("output_text") or "").strip()
    parts: List[str] = []
    for output in response.get("output") or []:
        if not isinstance(output, dict):
            continue
        for content in output.get("content") or []:
            if isinstance(content, dict):
                text = content.get("text") or content.get("output_text")
                if text:
                    parts.append(str(text))
        if output.get("type") == "message" and isinstance(output.get("text"), str):
            parts.append(str(output.get("text")))
    return "\n".join(part.strip() for part in parts if str(part or "").strip()).strip()


def _chat_message_text(message: Dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, list):
        content = "\n".join(str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in content)
    return str(content or "").strip()


def _deepseek_reasoning_content_chars(message: Dict[str, Any]) -> int:
    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, list):
        reasoning = "\n".join(str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in reasoning)
    return len(str(reasoning or "").strip())


def _should_retry_deepseek_empty_content(normalized: Dict[str, Any], message: Dict[str, Any]) -> bool:
    model = str(normalized.get("model") or "").strip().lower()
    return (
        model.startswith("deepseek-v4-pro")
        and not bool(normalized.get("disable_thinking"))
        and _deepseek_reasoning_content_chars(message) > 0
    )


def _empty_chat_content_error(normalized: Dict[str, Any], message: Dict[str, Any], choice: Dict[str, Any]) -> str:
    model = str(normalized.get("model") or "")
    reasoning_chars = _deepseek_reasoning_content_chars(message)
    finish_reason = str(choice.get("finish_reason") or "")
    if model.lower().startswith("deepseek-v4-pro") and reasoning_chars > 0:
        return (
            "LLM response content is empty after DeepSeek returned reasoning_content. "
            "Increase max_output_tokens or disable thinking for this call. "
            f"finish_reason={finish_reason}; reasoning_content_chars={reasoning_chars}"
        )
    return "LLM response content is empty."


def _retry_deepseek_with_disabled_thinking(
    *,
    normalized: Dict[str, Any],
    payload: Dict[str, Any],
    api: str,
    started_at: float,
    first_choice: Dict[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    retry_normalized = dict(normalized)
    retry_normalized["disable_thinking"] = True
    retry_payload = dict(payload)
    retry_payload["thinking"] = {"type": "disabled"}
    first_message = first_choice.get("message") or {}
    first_reasoning_chars = _deepseek_reasoning_content_chars(first_message)
    data = _post_llm_json(
        normalized=retry_normalized,
        url=normalize_openai_compatible_chat_url(retry_normalized["url"]),
        payload=retry_payload,
        error_prefix="LLM request",
    )
    _record_llm_usage_event(
        normalized=retry_normalized,
        usage=data.get("usage", {}),
        api=f"{api}_empty_content_retry",
        started_at=started_at,
    )
    llm_call = _llm_call_diagnostic(
        normalized=retry_normalized,
        api=api,
        started_at=started_at,
        status="success",
        usage=data.get("usage", {}),
    )
    llm_call.update(
        {
            "retry_after_empty_content": True,
            "retry_disable_thinking": True,
            "first_finish_reason": str(first_choice.get("finish_reason") or ""),
            "first_reasoning_content_chars": first_reasoning_chars,
        }
    )
    return data, retry_normalized, llm_call


def call_openai_responses_json(
    *,
    normalized: Dict[str, Any],
    system_prompt: str,
    user_payload: Dict[str, Any],
) -> Dict[str, Any]:
    api = "openai_responses_json"
    input_payload = {
        "response_format_instruction": "Return a valid JSON object. The response must be json.",
        "payload": user_payload,
    }
    user_content = compact_json(input_payload)
    max_output_tokens = int(normalized.get("max_output_tokens") or 0)
    context_budget = assert_llm_input_budget(
        normalized_config=normalized,
        system_prompt=system_prompt,
        user_content=user_content,
        api=api,
        max_output_tokens=max_output_tokens,
    )
    payload: Dict[str, Any] = {
        "model": normalized["model"],
        "instructions": system_prompt,
        "input": user_content,
        "text": {"format": {"type": "json_object"}},
    }
    if max_output_tokens > 0:
        payload["max_output_tokens"] = max_output_tokens
    _apply_openai_responses_reasoning(payload, normalized)
    started_at = time.perf_counter()
    try:
        data = _post_llm_json(
            normalized=normalized,
            url=normalize_openai_responses_url(normalized["url"]),
            payload=payload,
            error_prefix="OpenAI Responses request",
        )
    except Exception as exc:
        _raise_llm_call_error(exc, normalized=normalized, api=api, started_at=started_at)
    _record_llm_usage_event(
        normalized=normalized,
        usage=data.get("usage", {}),
        api=api,
        started_at=started_at,
    )
    llm_call = _llm_call_diagnostic(
        normalized=normalized,
        api=api,
        started_at=started_at,
        status="success",
        usage=data.get("usage", {}),
    )
    llm_call["context_budget"] = context_budget
    content = collect_openai_response_text(data)
    if not content:
        diagnostic = {**llm_call, "status": "failed", "error": "OpenAI Responses content is empty."}
        raise LLMCallError("OpenAI Responses content is empty.", diagnostic=diagnostic)
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        diagnostic = {**llm_call, "status": "failed", "error": f"OpenAI Responses output is not valid JSON: {content[:400]}"}
        raise LLMCallError(diagnostic["error"], diagnostic=diagnostic) from exc
    if not isinstance(parsed, dict):
        diagnostic = {**llm_call, "status": "failed", "error": "OpenAI Responses JSON root must be an object."}
        raise LLMCallError("OpenAI Responses JSON root must be an object.", diagnostic=diagnostic)
    return {
        "payload": parsed,
        "usage": data.get("usage", {}),
        "raw_response": data,
        "request_payload": user_payload,
        "llm_call": llm_call,
    }


def _call_openai_compatible_text_once(
    *,
    config: Dict[str, Any],
    system_prompt: str,
    user_content: str,
    temperature: float = 0.2,
    max_tokens: int = 16000,
) -> Dict[str, Any]:
    normalized = normalize_llm_config(config)
    if normalized["provider"] != "openai_compatible":
        raise RuntimeError(f"Unsupported LLM provider: {normalized['provider']}")
    if not llm_config_is_ready(normalized):
        raise RuntimeError("LLM config is incomplete.")

    max_output_tokens = int(normalized.get("max_output_tokens") or 0) or int(max_tokens or 0)
    if should_use_openai_responses_api(normalized):
        api = "openai_responses_text"
        context_budget = assert_llm_input_budget(
            normalized_config=normalized,
            system_prompt=system_prompt,
            user_content=user_content,
            api=api,
            max_output_tokens=max_output_tokens,
        )
        payload: Dict[str, Any] = {
            "model": normalized["model"],
            "instructions": system_prompt,
            "input": user_content,
        }
        if max_output_tokens > 0:
            payload["max_output_tokens"] = max_output_tokens
        _apply_openai_responses_reasoning(payload, normalized)
        started_at = time.perf_counter()
        try:
            data = _post_llm_json(
                normalized=normalized,
                url=normalize_openai_responses_url(normalized["url"]),
                payload=payload,
                error_prefix="OpenAI Responses request",
            )
        except Exception as exc:
            _raise_llm_call_error(exc, normalized=normalized, api=api, started_at=started_at)
        _record_llm_usage_event(
            normalized=normalized,
            usage=data.get("usage", {}),
            api=api,
            started_at=started_at,
        )
        llm_call = _llm_call_diagnostic(
            normalized=normalized,
            api=api,
            started_at=started_at,
            status="success",
            usage=data.get("usage", {}),
        )
        llm_call["context_budget"] = context_budget
        text = collect_openai_response_text(data)
        if not text:
            diagnostic = {**llm_call, "status": "failed", "error": "OpenAI Responses content is empty."}
            raise LLMCallError("OpenAI Responses content is empty.", diagnostic=diagnostic)
        return {"text": text, "usage": data.get("usage", {}), "raw_response": data, "llm_call": llm_call}

    api = "openai_compatible_chat_text"
    context_budget = assert_llm_input_budget(
        normalized_config=normalized,
        system_prompt=system_prompt,
        user_content=user_content,
        api=api,
        max_output_tokens=max_output_tokens,
    )
    payload = {
        "model": normalized["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    }
    if max_output_tokens > 0:
        payload["max_tokens"] = max_output_tokens
    if not model_uses_default_temperature_only(normalized["model"]):
        payload["temperature"] = temperature
    apply_openai_compatible_reasoning_flags(payload, normalized)
    started_at = time.perf_counter()
    try:
        data = _post_llm_json(
            normalized=normalized,
            url=normalize_openai_compatible_chat_url(normalized["url"]),
            payload=payload,
            error_prefix="LLM request",
        )
    except Exception as exc:
        _raise_llm_call_error(exc, normalized=normalized, api=api, started_at=started_at)
    _record_llm_usage_event(
        normalized=normalized,
        usage=data.get("usage", {}),
        api=api,
        started_at=started_at,
    )
    llm_call = _llm_call_diagnostic(
        normalized=normalized,
        api=api,
        started_at=started_at,
        status="success",
        usage=data.get("usage", {}),
    )
    llm_call["context_budget"] = context_budget
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("LLM response did not include choices.")
    choice = choices[0]
    message = choice.get("message") or {}
    text = _chat_message_text(message)
    if not text:
        if _should_retry_deepseek_empty_content(normalized, message):
            try:
                data, normalized, llm_call = _retry_deepseek_with_disabled_thinking(
                    normalized=normalized,
                    payload=payload,
                    api=api,
                    started_at=started_at,
                    first_choice=choice,
                )
            except Exception as exc:
                _raise_llm_call_error(exc, normalized=normalized, api=api, started_at=started_at)
            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError("LLM response did not include choices.")
            choice = choices[0]
            message = choice.get("message") or {}
            text = _chat_message_text(message)
        if not text:
            error = _empty_chat_content_error(normalized, message, choice)
            diagnostic = {**llm_call, "status": "failed", "error": error}
            raise LLMCallError(error, diagnostic=diagnostic)
    return {"text": text, "usage": data.get("usage", {}), "raw_response": data, "llm_call": llm_call}


def call_openai_compatible_text(
    *,
    config: Dict[str, Any],
    system_prompt: str,
    user_content: str,
    temperature: float = 0.2,
    max_tokens: int = 16000,
) -> Dict[str, Any]:
    normalized = normalize_llm_config(config)
    fallback_config = _fallback_config_for(config, normalized)
    fallback_normalized = normalize_llm_config(fallback_config) if fallback_config else {}
    circuit_key = _llm_circuit_key(normalized)
    if fallback_config and circuit_key in _UNAVAILABLE_LLM_KEYS:
        result = _call_openai_compatible_text_once(
            config=fallback_config,
            system_prompt=system_prompt,
            user_content=user_content,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return _annotate_fallback_result(
            result,
            primary=normalized,
            fallback=fallback_normalized,
            primary_error="primary_model_marked_unavailable",
            primary_skipped=True,
        )
    try:
        return _call_openai_compatible_text_once(
            config=config,
            system_prompt=system_prompt,
            user_content=user_content,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except ContextBudgetExceededError:
        raise
    except Exception as primary_exc:
        if not fallback_config:
            raise
        _mark_primary_unavailable(normalized, primary_exc)
        try:
            result = _call_openai_compatible_text_once(
                config=fallback_config,
                system_prompt=system_prompt,
                user_content=user_content,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as fallback_exc:
            _raise_fallback_error(primary_exc, fallback_exc, primary=normalized, fallback=fallback_normalized)
        return _annotate_fallback_result(
            result,
            primary=normalized,
            fallback=fallback_normalized,
            primary_error=_llm_error_summary(primary_exc),
        )


def _call_openai_compatible_json_once(
    *,
    config: Dict[str, Any],
    system_prompt: str,
    user_payload: Dict[str, Any],
) -> Dict[str, Any]:
    normalized = normalize_llm_config(config)
    if normalized["provider"] != "openai_compatible":
        raise RuntimeError(f"Unsupported LLM provider: {normalized['provider']}")
    if not llm_config_is_ready(normalized):
        raise RuntimeError("LLM config is incomplete.")
    if should_use_openai_responses_api(normalized):
        return call_openai_responses_json(
            normalized=normalized,
            system_prompt=system_prompt,
            user_payload=user_payload,
        )

    api = "openai_compatible_chat_json"
    user_content = compact_json(user_payload)
    context_budget = assert_llm_input_budget(
        normalized_config=normalized,
        system_prompt=system_prompt,
        user_content=user_content,
        api=api,
        max_output_tokens=int(normalized.get("max_output_tokens") or 0),
    )
    payload = {
        "model": normalized["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "response_format": {"type": "json_object"},
    }
    if int(normalized.get("max_output_tokens") or 0) > 0:
        payload["max_tokens"] = int(normalized["max_output_tokens"])
    if not model_uses_default_temperature_only(normalized["model"]):
        payload["temperature"] = 0.1
    apply_openai_compatible_reasoning_flags(payload, normalized)
    started_at = time.perf_counter()
    try:
        data = _post_llm_json(
            normalized=normalized,
            url=normalize_openai_compatible_chat_url(normalized["url"]),
            payload=payload,
            error_prefix="LLM request",
        )
    except Exception as exc:
        _raise_llm_call_error(exc, normalized=normalized, api=api, started_at=started_at)
    _record_llm_usage_event(
        normalized=normalized,
        usage=data.get("usage", {}),
        api=api,
        started_at=started_at,
    )
    llm_call = _llm_call_diagnostic(
        normalized=normalized,
        api=api,
        started_at=started_at,
        status="success",
        usage=data.get("usage", {}),
    )
    llm_call["context_budget"] = context_budget
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("LLM response did not include choices.")
    choice = choices[0]
    message = choice.get("message") or {}
    content = _chat_message_text(message)
    if not content:
        if _should_retry_deepseek_empty_content(normalized, message):
            try:
                data, normalized, llm_call = _retry_deepseek_with_disabled_thinking(
                    normalized=normalized,
                    payload=payload,
                    api=api,
                    started_at=started_at,
                    first_choice=choice,
                )
            except Exception as exc:
                _raise_llm_call_error(exc, normalized=normalized, api=api, started_at=started_at)
            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError("LLM response did not include choices.")
            choice = choices[0]
            message = choice.get("message") or {}
            content = _chat_message_text(message)
        if not content:
            error = _empty_chat_content_error(normalized, message, choice)
            diagnostic = {**llm_call, "status": "failed", "error": error}
            raise LLMCallError(error, diagnostic=diagnostic)
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        diagnostic = {**llm_call, "status": "failed", "error": f"LLM response is not valid JSON: {content[:400]}"}
        raise LLMCallError(diagnostic["error"], diagnostic=diagnostic) from exc
    if not isinstance(payload, dict):
        diagnostic = {**llm_call, "status": "failed", "error": "LLM response JSON root must be an object."}
        raise LLMCallError("LLM response JSON root must be an object.", diagnostic=diagnostic)
    return {
        "payload": payload,
        "usage": data.get("usage", {}),
        "raw_response": data,
        "request_payload": user_payload,
        "llm_call": llm_call,
    }


def call_openai_compatible_json(
    *,
    config: Dict[str, Any],
    system_prompt: str,
    user_payload: Dict[str, Any],
) -> Dict[str, Any]:
    normalized = normalize_llm_config(config)
    fallback_config = _fallback_config_for(config, normalized)
    fallback_normalized = normalize_llm_config(fallback_config) if fallback_config else {}
    circuit_key = _llm_circuit_key(normalized)
    if fallback_config and circuit_key in _UNAVAILABLE_LLM_KEYS:
        result = _call_openai_compatible_json_once(
            config=fallback_config,
            system_prompt=system_prompt,
            user_payload=user_payload,
        )
        return _annotate_fallback_result(
            result,
            primary=normalized,
            fallback=fallback_normalized,
            primary_error="primary_model_marked_unavailable",
            primary_skipped=True,
        )
    try:
        return _call_openai_compatible_json_once(
            config=config,
            system_prompt=system_prompt,
            user_payload=user_payload,
        )
    except ContextBudgetExceededError:
        raise
    except Exception as primary_exc:
        if not fallback_config:
            raise
        _mark_primary_unavailable(normalized, primary_exc)
        try:
            result = _call_openai_compatible_json_once(
                config=fallback_config,
                system_prompt=system_prompt,
                user_payload=user_payload,
            )
        except Exception as fallback_exc:
            _raise_fallback_error(primary_exc, fallback_exc, primary=normalized, fallback=fallback_normalized)
        return _annotate_fallback_result(
            result,
            primary=normalized,
            fallback=fallback_normalized,
            primary_error=_llm_error_summary(primary_exc),
        )


class ConversationMemory:
    def __init__(self, store_dir: Path | str, max_turns: int = 10, summary_trigger: int = 6):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.max_turns = max(1, int(max_turns))
        self.summary_trigger = max(2, int(summary_trigger))

    def _session_path(self, session_id: str) -> Path:
        raw = str(session_id or "").strip()
        safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", raw) or "default"
        digest = hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:12] if raw else "empty"
        return self.store_dir / f"{safe[:80]}-{digest}.json"

    def load_session(self, session_id: str) -> Dict[str, Any]:
        if not session_id:
            return {"session_id": "", "summary": "", "turns": []}
        path = self._session_path(session_id)
        if not path.exists():
            return {"session_id": session_id, "summary": "", "turns": []}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {"session_id": session_id, "summary": "", "turns": []}
        turns = data.get("turns", [])
        if not isinstance(turns, list):
            turns = []
        return {
            "session_id": session_id,
            "summary": str(data.get("summary") or "").strip(),
            "turns": turns,
        }

    def save_session(self, session_id: str, summary: str, turns: Iterable[Dict[str, Any]]) -> None:
        if not session_id:
            return
        payload = {
            "session_id": session_id,
            "summary": str(summary or "").strip(),
            "turns": list(turns),
        }
        self._session_path(session_id).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )

    def get_recent_turns(self, session_id: str, limit: int = 4) -> List[Dict[str, Any]]:
        session = self.load_session(session_id)
        return list(session.get("turns", []))[-max(1, int(limit)) :]

    def get_recent_retrieved_chunk_uids(self, session_id: str, limit: int = 1) -> List[str]:
        turns = self.get_recent_turns(session_id, limit=limit)
        values: List[str] = []
        seen = set()
        for turn in reversed(turns):
            for field in ("evidence_chunk_uids", "retrieved_chunk_uids"):
                for uid in turn.get(field, []) or []:
                    cleaned = str(uid or "").strip()
                    if cleaned and cleaned not in seen:
                        seen.add(cleaned)
                        values.append(cleaned)
        return values

    def append_turn(
        self,
        session_id: str,
        turn: Turn,
        *,
        llm_config: Optional[Dict[str, Any]] = None,
        llm_trace_collector: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if not session_id:
            return
        session = self.load_session(session_id)
        turns = list(session.get("turns", []))
        turns.append(turn.to_dict())
        summary = str(session.get("summary") or "").strip()

        if len(turns) > self.max_turns:
            overflow = turns[:-self.max_turns]
            turns = turns[-self.max_turns :]
            summary = summarize_archived_turns(
                current_summary=summary,
                archived_turns=overflow,
                llm_config=llm_config,
                llm_trace_collector=llm_trace_collector,
            )

        self.save_session(session_id, summary=summary, turns=turns)


def build_turn_summary(query: str, answer: str) -> str:
    answer = " ".join(str(answer or "").split())
    if not answer:
        return str(query or "").strip()[:180]
    return f"Q: {str(query or '').strip()} | A: {answer[:220]}"


def summarize_archived_turns(
    *,
    current_summary: str,
    archived_turns: List[Dict[str, Any]],
    llm_config: Optional[Dict[str, Any]] = None,
    llm_trace_collector: Optional[List[Dict[str, Any]]] = None,
) -> str:
    if not archived_turns:
        return current_summary

    compact_items = []
    for turn in archived_turns:
        compact_items.append(
            {
                "query": str(turn.get("user_query") or "").strip(),
                "standalone_query": str(turn.get("standalone_query") or "").strip(),
                "answer_status": str(turn.get("answer_status") or "").strip(),
                "summary": str(turn.get("summary") or "").strip(),
            }
        )

    if llm_config_is_ready(llm_config):
        system_prompt = (
            "你负责总结企业检索问答中的历史对话轮次。"
            "只返回 JSON，且只能包含一个键：summary。"
            "请把归档轮次压缩成简洁、事实性的会话记忆，保留实体、约束条件和仍未解决的追问。"
        )
        user_payload = {
            "current_summary": current_summary,
            "archived_turns": compact_items,
        }
        try:
            response = call_openai_compatible_json(config=llm_config or {}, system_prompt=system_prompt, user_payload=user_payload)
            payload = response.get("payload", {})
            summary = str(payload.get("summary") or "").strip()
            if summary:
                if llm_trace_collector is not None:
                    llm_trace_collector.append(
                        {
                            "type": "memory_summary",
                            "request": user_payload,
                            "response": payload,
                            "usage": response.get("usage", {}),
                        }
                    )
                return summary
        except Exception as exc:
            if llm_trace_collector is not None:
                llm_trace_collector.append(
                    {
                        "type": "memory_summary",
                        "request": user_payload,
                        "error": str(exc),
                    }
                )

    lines = [part for part in [current_summary] if part]
    lines.extend(item["summary"] or item["query"] for item in compact_items if item["summary"] or item["query"])
    merged = " | ".join(lines)
    return merged[:800]


def build_contextualization_result(
    *,
    query: str,
    standalone_query: str,
    memory_summary: str,
    recent_turns: List[Dict[str, Any]],
    reused_chunk_uids: List[str],
    source: str,
    note: str = "",
    llm_call: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "original_query": str(query or "").strip(),
        "standalone_query": str(standalone_query or query or "").strip(),
        "memory_summary": memory_summary,
        "recent_turns": recent_turns,
        "reused_chunk_uids": reused_chunk_uids,
        "source": source,
        "note": note,
        "llm_call": llm_call or {},
    }


def contextualize_query(
    *,
    query: str,
    session_id: str,
    memory: ConversationMemory,
    llm_config: Optional[Dict[str, Any]],
    history_turns: int = 4,
) -> Dict[str, Any]:
    query = str(query or "").strip()
    if not session_id:
        return build_contextualization_result(
            query=query,
            standalone_query=query,
            memory_summary="",
            recent_turns=[],
            reused_chunk_uids=[],
            source="disabled",
            note="no_session_id",
        )

    session = memory.load_session(session_id)
    recent_turns = list(session.get("turns", []))[-max(1, int(history_turns)) :]
    memory_summary = str(session.get("summary") or "").strip()
    reused_chunk_uids = memory.get_recent_retrieved_chunk_uids(session_id, limit=1)

    if not recent_turns:
        return build_contextualization_result(
            query=query,
            standalone_query=query,
            memory_summary=memory_summary,
            recent_turns=[],
            reused_chunk_uids=[],
            source="cold_start",
            note="no_prior_turns",
        )

    heuristic_followup = bool(_PRONOUN_RE.search(query))
    if llm_config_is_ready(llm_config):
        system_prompt = (
            "你负责把企业检索问答中的追问改写成可独立检索的查询。"
            "不要回答问题，只消解指代并保留原有约束。"
            "只返回 JSON，包含这些键：standalone_query、reuse_last_turn、note。"
        )
        recent_payload = [
            {
                "user_query": str(turn.get("user_query") or "").strip(),
                "standalone_query": str(turn.get("standalone_query") or "").strip(),
                "summary": str(turn.get("summary") or "").strip(),
                "answer_status": str(turn.get("answer_status") or "").strip(),
            }
            for turn in recent_turns
        ]
        user_payload = {
            "current_query": query,
            "memory_summary": memory_summary,
            "recent_turns": recent_payload,
        }
        try:
            response = call_openai_compatible_json(config=llm_config or {}, system_prompt=system_prompt, user_payload=user_payload)
            payload = response.get("payload", {})
            standalone_query = str(payload.get("standalone_query") or query).strip() or query
            reuse_last_turn = bool(payload.get("reuse_last_turn", False))
            if reuse_last_turn:
                reused_chunk_uids = memory.get_recent_retrieved_chunk_uids(session_id, limit=1)
            return build_contextualization_result(
                query=query,
                standalone_query=standalone_query,
                memory_summary=memory_summary,
                recent_turns=recent_payload,
                reused_chunk_uids=reused_chunk_uids,
                source="llm",
                note=str(payload.get("note") or "").strip(),
                llm_call={
                    "type": "contextualize_query",
                    "request": user_payload,
                    "response": payload,
                    "usage": response.get("usage", {}),
                },
            )
        except Exception as exc:
            pass

    if heuristic_followup and recent_turns:
        last_turn = recent_turns[-1]
        anchor = str(last_turn.get("standalone_query") or last_turn.get("user_query") or "").strip()
        standalone_query = f"{anchor} ; follow-up: {query}".strip(" ;")
        return build_contextualization_result(
            query=query,
            standalone_query=standalone_query,
            memory_summary=memory_summary,
            recent_turns=recent_turns,
            reused_chunk_uids=reused_chunk_uids,
            source="heuristic",
            note="pronoun_follow_up",
        )

    return build_contextualization_result(
        query=query,
        standalone_query=query,
        memory_summary=memory_summary,
        recent_turns=recent_turns,
        reused_chunk_uids=[],
        source="passthrough",
        note="history_present_but_no_rewrite_needed",
    )
