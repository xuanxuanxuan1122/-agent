from __future__ import annotations

import json
import math
import os
import re
from typing import Any, Dict, Optional


_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


class ContextBudgetExceededError(RuntimeError):
    """Raised before an LLM request when the estimated input exceeds budget."""

    def __init__(self, diagnostic: Dict[str, Any]) -> None:
        self.diagnostic = dict(diagnostic)
        task = str(diagnostic.get("task") or "unknown")
        estimated_input_tokens = int(diagnostic.get("estimated_input_tokens") or 0)
        max_input_tokens = int(diagnostic.get("max_input_tokens") or 0)
        input_chars = int(diagnostic.get("input_chars") or 0)
        max_input_chars = int(diagnostic.get("max_input_chars") or 0)
        super().__init__(
            "llm_context_budget_exceeded: "
            f"task={task} estimated_input_tokens={estimated_input_tokens} "
            f"max_input_tokens={max_input_tokens} input_chars={input_chars} "
            f"max_input_chars={max_input_chars}"
        )


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def estimate_input_tokens(text: str) -> int:
    text = str(text or "")
    if not text:
        return 0
    cjk_chars = len(_CJK_RE.findall(text))
    non_cjk_chars = max(0, len(text) - cjk_chars)
    non_cjk_tokens = int(math.ceil(non_cjk_chars / 4)) if non_cjk_chars else 0
    return cjk_chars + non_cjk_tokens


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(str(raw).strip())
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(str(raw).strip())
    except ValueError:
        return default


def _scope_key(value: Any) -> str:
    key = re.sub(r"[^A-Za-z0-9]+", "_", str(value or "").strip()).strip("_").upper()
    return key


def _scoped_env_int(base_name: str, *, task: str = "", profile: str = "", model: str = "") -> int:
    keys = [_scope_key(task), _scope_key(profile), _scope_key(model)]
    for key in keys:
        if not key:
            continue
        value = _env_int(f"{base_name}_{key}", 0)
        if value > 0:
            return value
        suffix = base_name.removeprefix("RAG_LLM_")
        value = _env_int(f"RAG_LLM_{key}_{suffix}", 0)
        if value > 0:
            return value
    return _env_int(base_name, 0)


def _model_context_window_tokens(model: str) -> int:
    lower_model = str(model or "").strip().lower()
    if "8k" in lower_model:
        return 8_192
    if "16k" in lower_model:
        return 16_384
    if "32k" in lower_model:
        return 32_768
    if "64k" in lower_model:
        return 65_536
    if "128k" in lower_model:
        return 131_072
    return _env_int("RAG_LLM_DEFAULT_CONTEXT_WINDOW_TOKENS", 128_000)


def _input_token_budget(
    normalized_config: Dict[str, Any],
    *,
    max_output_tokens: int = 0,
) -> Dict[str, int]:
    task = str(normalized_config.get("task_name") or normalized_config.get("task") or "").strip()
    profile = str(normalized_config.get("profile") or "").strip()
    model = str(normalized_config.get("model") or "").strip()
    max_input_tokens = _scoped_env_int("RAG_LLM_MAX_INPUT_TOKENS", task=task, profile=profile, model=model)
    context_window_tokens = _scoped_env_int(
        "RAG_LLM_CONTEXT_WINDOW_TOKENS",
        task=task,
        profile=profile,
        model=model,
    ) or _model_context_window_tokens(model)
    reserve_output_tokens = _scoped_env_int(
        "RAG_LLM_OUTPUT_RESERVE_TOKENS",
        task=task,
        profile=profile,
        model=model,
    )
    reserve_output_tokens = max(
        reserve_output_tokens,
        int(normalized_config.get("max_output_tokens") or 0),
        int(max_output_tokens or 0),
        1_024,
    )
    budget_ratio = min(0.98, max(0.10, _env_float("RAG_LLM_INPUT_BUDGET_RATIO", 0.90)))
    if max_input_tokens <= 0:
        max_input_tokens = int(max(0, context_window_tokens - reserve_output_tokens) * budget_ratio)
    return {
        "max_input_tokens": max_input_tokens,
        "context_window_tokens": context_window_tokens,
        "reserve_output_tokens": reserve_output_tokens,
    }


def llm_input_budget_diagnostic(
    *,
    normalized_config: Dict[str, Any],
    system_prompt: str,
    user_content: str,
    api: str,
    max_output_tokens: int = 0,
) -> Dict[str, Any]:
    task = str(normalized_config.get("task_name") or normalized_config.get("task") or "").strip()
    profile = str(normalized_config.get("profile") or "").strip()
    model = str(normalized_config.get("model") or "").strip()
    system_prompt = str(system_prompt or "")
    user_content = str(user_content or "")
    input_chars = len(system_prompt) + len(user_content)
    estimated_input_tokens = estimate_input_tokens(system_prompt) + estimate_input_tokens(user_content)
    token_budget = _input_token_budget(normalized_config, max_output_tokens=max_output_tokens)
    max_input_chars = _scoped_env_int("RAG_LLM_MAX_INPUT_CHARS", task=task, profile=profile, model=model)
    return {
        "type": "llm_context_budget",
        "status": "ok",
        "api": api,
        "task": task,
        "profile": profile,
        "model": model,
        "input_chars": input_chars,
        "max_input_chars": max_input_chars,
        "estimated_input_tokens": estimated_input_tokens,
        **token_budget,
    }


def assert_llm_input_budget(
    *,
    normalized_config: Dict[str, Any],
    system_prompt: str,
    user_content: str,
    api: str,
    max_output_tokens: int = 0,
) -> Dict[str, Any]:
    diagnostic = llm_input_budget_diagnostic(
        normalized_config=normalized_config,
        system_prompt=system_prompt,
        user_content=user_content,
        api=api,
        max_output_tokens=max_output_tokens,
    )
    if not _env_flag("RAG_LLM_CONTEXT_BUDGET_ENABLED", True):
        return {**diagnostic, "status": "disabled"}
    reasons = []
    max_input_tokens = int(diagnostic.get("max_input_tokens") or 0)
    if max_input_tokens > 0 and int(diagnostic["estimated_input_tokens"]) > max_input_tokens:
        reasons.append("estimated_input_tokens")
    max_input_chars = int(diagnostic.get("max_input_chars") or 0)
    if max_input_chars > 0 and int(diagnostic["input_chars"]) > max_input_chars:
        reasons.append("input_chars")
    if reasons:
        blocked = {**diagnostic, "status": "blocked", "reasons": reasons}
        _record_context_budget(blocked)
        raise ContextBudgetExceededError(blocked)
    _record_context_budget(diagnostic)
    return diagnostic


def _record_context_budget(diagnostic: Dict[str, Any]) -> None:
    try:
        from .token_usage import record_llm_context_budget

        record_llm_context_budget(diagnostic)
    except Exception:
        return
