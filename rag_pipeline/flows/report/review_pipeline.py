from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, Optional

from .llm_review_agent import build_structured_review, llm_review_structured
from .review_agent import rule_based_review


logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, min_value: int = 0, max_value: int = 100_000) -> int:
    try:
        value = int(os.getenv(name, str(default)) or default)
    except (TypeError, ValueError):
        value = default
    return max(min_value, min(max_value, value))


def _env_float(name: str, default: float, *, min_value: float = 0.0, max_value: float = 2.0) -> float:
    try:
        value = float(os.getenv(name, str(default)) or default)
    except (TypeError, ValueError):
        value = default
    return max(min_value, min(max_value, value))


def _message_content(message: Any) -> tuple[str, str]:
    role = ""
    content = ""
    if isinstance(message, dict):
        role = str(message.get("role") or "")
        content = str(message.get("content") or "")
        return role, content
    role = str(getattr(message, "type", "") or getattr(message, "role", "") or "")
    content = str(getattr(message, "content", "") or message or "")
    return role, content


class OpenAICompatibleReviewClient:
    """Tiny async adapter so ReviewAgent Stage2 can use the shared LLM config."""

    async def ainvoke(self, messages: Any) -> str:
        system_parts = []
        user_parts = []
        for message in list(messages or []):
            role, content = _message_content(message)
            if not content.strip():
                continue
            if role.lower() in {"system"}:
                system_parts.append(content)
            else:
                user_parts.append(content)
        system_prompt = "\n\n".join(system_parts).strip()
        user_content = "\n\n".join(user_parts).strip()
        if not user_content:
            return ""
        from .reformatter_agent import _chat_text_with_openai_compatible
        from rag_pipeline.config.search_config import build_llm_config_for_task

        return await asyncio.to_thread(
            _chat_text_with_openai_compatible,
            config=dict(build_llm_config_for_task("review_stage2")),
            system_prompt=system_prompt,
            user_content=user_content,
            temperature=_env_float("REPORT_LLM_REVIEW_TEMPERATURE", 0.15, min_value=0.0, max_value=1.5),
            max_tokens=_env_int("REPORT_LLM_REVIEW_MAX_TOKENS", 14000, min_value=2000, max_value=64000),
        )


def _default_review_client() -> Optional[OpenAICompatibleReviewClient]:
    if not _env_flag("REPORT_ENABLE_OPENAI_COMPATIBLE_REVIEW", True):
        return None
    try:
        from rag_pipeline.config.search_config import build_llm_config_for_task
        from rag_pipeline.search.memory import llm_config_is_ready

        if not llm_config_is_ready(dict(build_llm_config_for_task("review_stage2"))):
            return None
    except Exception:
        return None
    return OpenAICompatibleReviewClient()


def _count_total_fixes(audit_log: Dict[str, Any]) -> int:
    return (
        len(audit_log.get("leak_patterns_removed") or [])
        + int(audit_log.get("duplicate_bullets_removed") or 0)
        + int(audit_log.get("duplicate_paragraphs_removed") or 0)
        + int(audit_log.get("truncated_or_meaningless_removed") or 0)
        + int(audit_log.get("empty_bullets_removed") or 0)
        + int(audit_log.get("empty_risks_removed") or 0)
        + len(audit_log.get("empty_sections_filled") or [])
    )


async def run_review_pipeline(
    writer_output: str,
    llm_client: Any = None,
    skip_llm_review: bool = False,
    evidence: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    logger.info("[ReviewAgent] Stage1 规则审查开始")
    stage1_output, audit_log = rule_based_review(writer_output)
    total_fixes = _count_total_fixes(audit_log)
    logger.info(
        "[ReviewAgent] Stage1 完成: 泄露文本=%s处, 重复bullet=%s处, 重复段落=%s处, 空风险=%s处",
        len(audit_log.get("leak_patterns_removed") or []),
        audit_log.get("duplicate_bullets_removed") or 0,
        audit_log.get("duplicate_paragraphs_removed") or 0,
        audit_log.get("empty_risks_removed") or 0,
    )

    if not skip_llm_review and llm_client is None:
        llm_client = _default_review_client()

    if skip_llm_review or llm_client is None:
        structured_review = build_structured_review(
            original_report=writer_output,
            revised_report=stage1_output,
            evidence=evidence,
            llm_used=False,
            stage2_skipped=True,
            stage2_reason="skip_llm_review=True" if skip_llm_review else "openai_compatible_review_config_not_ready",
        )
        return {
            "final_report": stage1_output,
            "stage1_audit": audit_log,
            "stage2_skipped": True,
            "stage2_reason": "skip_llm_review=True" if skip_llm_review else "openai_compatible_review_config_not_ready",
            "structured_review": structured_review,
            "total_fixes": total_fixes,
        }

    logger.info("[ReviewAgent] Stage2 LLM精修开始")
    try:
        structured_review = await llm_review_structured(stage1_output, llm_client, evidence=evidence)
        stage2_output = str(structured_review.get("revised_report") or stage1_output)
        return {
            "final_report": stage2_output,
            "stage1_audit": audit_log,
            "stage2_skipped": bool(structured_review.get("stage2_skipped")),
            "stage2_reason": str(structured_review.get("stage2_reason") or ""),
            "structured_review": structured_review,
            "total_fixes": total_fixes,
        }
    except Exception as exc:
        logger.exception("[ReviewAgent] Stage2 失败，回退到 Stage1 结果: %s", exc)
        structured_review = build_structured_review(
            original_report=writer_output,
            revised_report=stage1_output,
            evidence=evidence,
            llm_used=False,
            stage2_skipped=True,
            stage2_reason=str(exc),
            errors=[str(exc)],
        )
        return {
            "final_report": stage1_output,
            "stage1_audit": audit_log,
            "stage2_skipped": True,
            "stage2_reason": str(exc),
            "structured_review": structured_review,
            "total_fixes": total_fixes,
        }


def run_review_pipeline_sync(
    writer_output: str,
    llm_client: Any = None,
    skip_llm_review: bool = False,
    evidence: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return asyncio.run(
        run_review_pipeline(
            writer_output=writer_output,
            llm_client=llm_client,
            skip_llm_review=skip_llm_review,
            evidence=evidence,
        )
    )
