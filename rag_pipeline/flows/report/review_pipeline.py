from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict

from .llm_review_agent import llm_review
from .review_agent import rule_based_review


logger = logging.getLogger(__name__)


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
    skip_llm_review: bool = True,
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

    if skip_llm_review or llm_client is None:
        return {
            "final_report": stage1_output,
            "stage1_audit": audit_log,
            "stage2_skipped": True,
            "stage2_reason": "skip_llm_review=True" if skip_llm_review else "llm_client is None",
            "total_fixes": total_fixes,
        }

    logger.info("[ReviewAgent] Stage2 LLM精修开始")
    try:
        stage2_output = await llm_review(stage1_output, llm_client)
        return {
            "final_report": stage2_output,
            "stage1_audit": audit_log,
            "stage2_skipped": False,
            "stage2_reason": "",
            "total_fixes": total_fixes,
        }
    except Exception as exc:
        logger.exception("[ReviewAgent] Stage2 失败，回退到 Stage1 结果: %s", exc)
        return {
            "final_report": stage1_output,
            "stage1_audit": audit_log,
            "stage2_skipped": True,
            "stage2_reason": str(exc),
            "total_fixes": total_fixes,
        }


def run_review_pipeline_sync(
    writer_output: str,
    llm_client: Any = None,
    skip_llm_review: bool = True,
) -> Dict[str, Any]:
    return asyncio.run(
        run_review_pipeline(
            writer_output=writer_output,
            llm_client=llm_client,
            skip_llm_review=skip_llm_review,
        )
    )
