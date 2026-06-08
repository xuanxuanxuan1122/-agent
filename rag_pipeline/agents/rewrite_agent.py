from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..config.search_config import (
    build_llm_config_for_task,
)
from ..search.memory import call_openai_compatible_json, llm_config_is_ready
from .qa_agent import INTERNAL_LABELS
from .public_report_sanitizer import SAFE_PUBLIC_TERMS, rewrite_internal_gap_language, sanitize_public_markdown


AGENT_NAME = "rewrite_agent"
AGENT_DESCRIPTION = "Rewrite Agent. Fixes expression and formatting without changing facts."
logger = logging.getLogger(__name__)


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _llm_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if config:
        return _as_dict(config)
    return dict(build_llm_config_for_task("qa"))


def _qa_instructions(qa_result: Optional[Dict[str, Any]], rewrite_instructions: Optional[Sequence[Any]]) -> List[str]:
    qa = _as_dict(qa_result)
    instructions: List[str] = [str(item).strip() for item in _as_list(rewrite_instructions) if str(item).strip()]
    instructions.extend(str(item).strip() for item in _as_list(qa.get("rewrite_instructions")) if str(item).strip())
    for issue in [*_as_list(qa.get("errors")), *_as_list(qa.get("issues"))]:
        item = _as_dict(issue)
        issue_type = str(item.get("type") or item.get("code") or "").strip()
        message = str(item.get("message") or item.get("detail") or item).strip()
        if issue_type or message:
            instructions.append(" | ".join(part for part in [issue_type, message] if part))
    deep = _as_dict(qa.get("deep_evaluation"))
    for item in _as_list(deep.get("rewrite_instructions")):
        text = str(item or "").strip()
        if text:
            instructions.append(text)
    return _dedupe(instructions, limit=30)


def _compact_issue(issue: Any, *, message_chars: int = 240) -> Dict[str, Any]:
    payload = _as_dict(issue)
    message = str(payload.get("message") or payload.get("detail") or payload.get("text") or "").strip()
    if len(message) > message_chars:
        message = message[: message_chars - 1].rstrip() + "..."
    compact: Dict[str, Any] = {}
    for key in ("type", "code", "severity", "qa_category", "chapter_id", "section_id"):
        value = payload.get(key)
        if value:
            compact[key] = value
    if message:
        compact["message"] = message
    return compact


def _compact_qa_for_llm(qa_result: Dict[str, Any], *, item_limit: int = 30) -> Dict[str, Any]:
    """Strip qa_result to the fields the rewriter actually needs.

    Full qa_result carries evidence_health_summary, chapter_packages, dimension
    scores, etc., which can push the LLM input past the context budget (observed
    ~360k tokens vs ~113k cap). The rewriter only needs the actionable findings
    plus rewrite_instructions, so this returns a compacted shell.
    """
    payload = _as_dict(qa_result)
    deep = _as_dict(payload.get("deep_evaluation"))
    deep_instructions = [
        str(item or "").strip()
        for item in _as_list(deep.get("rewrite_instructions"))
        if str(item or "").strip()
    ]
    deep_compact: Dict[str, Any] = {}
    if deep_instructions:
        deep_compact["rewrite_instructions"] = deep_instructions[:item_limit]
    deep_findings = [
        _compact_issue(item)
        for item in _as_list(deep.get("issues"))[:item_limit]
        if isinstance(item, dict)
    ]
    if deep_findings:
        deep_compact["issues"] = deep_findings
    compact: Dict[str, Any] = {
        "errors": [_compact_issue(item) for item in _as_list(payload.get("errors"))[:item_limit]],
        "warnings": [_compact_issue(item) for item in _as_list(payload.get("warnings"))[:item_limit]],
        "rewrite_instructions": [
            str(item or "").strip()
            for item in _as_list(payload.get("rewrite_instructions"))
            if str(item or "").strip()
        ][:item_limit],
    }
    quality_findings = _as_list(payload.get("quality_findings"))
    if quality_findings:
        compact["quality_findings"] = [
            _compact_issue(item) for item in quality_findings[:item_limit]
        ]
    if deep_compact:
        compact["deep_evaluation"] = deep_compact
    return compact


def _dedupe(values: Iterable[str], *, limit: int = 30) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        key = re.sub(r"\s+", "", str(value or "").lower())
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(str(value).strip())
        if len(result) >= limit:
            break
    return result


def _strip_internal_labels(text: str) -> str:
    value = str(text or "")
    safe_terms = set(SAFE_PUBLIC_TERMS)
    for label in INTERNAL_LABELS:
        if label in safe_terms:
            continue
        value = value.replace(label, "")
    value = re.sub(r"(?m)^\s*(?:QA|Validation|质量检查|璐ㄩ噺妫€鏌?).*$", "", value)
    return value


def _apply_instruction_rewrites(text: str, instructions: Sequence[str]) -> str:
    value = str(text or "")
    joined = "\n".join(str(item or "") for item in instructions)
    if re.search(r"internal|内部|黑名单|forbidden|gap|缺口", joined, re.I):
        value = rewrite_internal_gap_language(value)
    if re.search(r"table|表格|source header|资料来源|引用", joined, re.I):
        value = re.sub(r"(?m)^\|[^\n]*(?:引用|来源|资料来源|判断用途)[^\n]*\|\n(?:\|[^\n]*\|\n?)+", "", value)
    if re.search(r"empty heading|空标题|empty section", joined, re.I):
        value = re.sub(r"(?m)^#{2,4}\s+[^\n]+\n(?=\s*(?:#{1,4}\s+|$))", "", value)
    return value


SEMANTIC_DOWNGRADE_RE = re.compile(
    r"proof|evidence|coverage|counter|insufficient|missing|unsupported|overclaim|"
    r"证明|证据|覆盖|反证|缺口|缺少|不足|强结论|过强|降级|方向性|边界|不确定",
    re.I,
)


def _needs_semantic_downgrade(instructions: Sequence[str]) -> bool:
    return bool(SEMANTIC_DOWNGRADE_RE.search("\n".join(str(item or "") for item in instructions)))


def _downgrade_semantic_claims(text: str) -> str:
    value = str(text or "")
    replacements = [
        (r"已经证明", "初步显示"),
        (r"足以证明", "可以作为初步线索支持"),
        (r"能够证明", "可以作为线索支持"),
        (r"明确证明", "显示出"),
        (r"明确表明", "显示出"),
        (r"清晰表明", "显示出"),
        (r"确定性结论", "阶段性判断"),
        (r"确定性机会", "阶段性机会"),
        (r"必然", "可能"),
        (r"一定会", "可能会"),
        (r"一定能", "可能能"),
        (r"全面领先", "在部分维度领先"),
        (r"全面改善", "在部分指标上改善"),
        (r"显著优于", "在可见指标上优于"),
        (r"显著改善", "有所改善"),
        (r"必须立即", "可优先评估"),
        (r"应当立即", "可优先评估"),
        (r"应该立即", "可优先评估"),
        (r"支持可优先评估进入", "支持进一步评估是否进入"),
        (r"支持可优先评估", "支持进一步评估"),
    ]
    for pattern, replacement in replacements:
        value = re.sub(pattern, replacement, value)
    if "现有公开材料更适合支持方向性观察" not in value and re.search(r"阶段性判断|初步显示|线索支持|方向性", value):
        paragraphs = value.split("\n\n")
        insert_at = 1 if paragraphs and paragraphs[0].lstrip().startswith("#") else 0
        boundary = "现有公开材料更适合支持方向性观察；涉及规模、盈利、份额或政策传导的判断，仍需要同口径数据、较高等级来源和反向样本继续验证。"
        paragraphs.insert(min(insert_at + 1, len(paragraphs)), boundary)
        value = "\n\n".join(paragraphs)
    return value


def _citation_ids(text: str) -> set[str]:
    return set(re.findall(r"\[(\d{1,3})\]", str(text or "")))


def _llm_rewrite(
    *,
    markdown: str,
    qa_result: Dict[str, Any],
    instructions: Sequence[str],
    llm_config: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    if not _env_flag("REPORT_ENABLE_LLM_REWRITE", False):
        return None
    config = _llm_config(llm_config)
    if not llm_config_is_ready(config):
        return None
    system_prompt = """
你是行业研究报告的终稿重写 Agent。只能做表达修复、结论降级和结构整理，不得新增事实、数字、公司、政策或来源。

要求：
1. 保留原文已有事实和引用编号，不能把没有引用的新增事实写进正文。
2. 如果 QA 指出证据不足、反证缺失、覆盖不足或证明强度不足，把强结论改成方向性观察、阶段性判断或需验证事项。
3. 删除内部过程标签、QA 标记、证据缺口黑话和空标题。
4. 不得新增未在原文中出现的数字、公司名、日期、政策名或来源；证据不足时宁可缩短，不要扩写。
5. 输出 JSON：{"markdown":"...","change_summary":["..."]}。
""".strip()
    # qa_result is the dominant source of payload bloat (it can carry the full
    # evidence_health_summary, chapter packages and dimension scores). The
    # rewriter only needs the actionable findings + instructions, so we send a
    # compacted shell and leave the markdown intact. If the markdown alone still
    # exceeds the LLM's context budget, the upstream context_budget guard will
    # block the call deterministically rather than producing a partial rewrite.
    compact_qa = _compact_qa_for_llm(qa_result)
    try:
        response = call_openai_compatible_json(
            config=config,
            system_prompt=system_prompt,
            user_payload={
                "markdown": markdown,
                "qa_result": compact_qa,
                "rewrite_instructions": list(instructions)[:30],
            },
        )
    except Exception:
        logger.exception("LLM rewrite failed")
        return None
    payload = _as_dict(response.get("payload"))
    candidate = str(payload.get("markdown") or "").strip()
    if not candidate:
        return None
    old_citations = _citation_ids(markdown)
    new_citations = _citation_ids(candidate)
    if old_citations and len(new_citations & old_citations) < max(1, int(len(old_citations) * 0.7)):
        logger.warning("LLM rewrite rejected because citation retention is too low")
        return None
    old_len = len(re.sub(r"\s+", "", markdown))
    new_len = len(re.sub(r"\s+", "", candidate))
    if old_len >= 1000 and new_len < int(old_len * 0.45):
        logger.warning("LLM rewrite rejected because output is too short")
        return None
    return candidate


def run_rewrite_agent(
    *,
    report_markdown: str = "",
    qa_result: Optional[Dict[str, Any]] = None,
    rewrite_instructions: Optional[Sequence[Any]] = None,
    llm_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    original = str(report_markdown or "")
    instructions = _qa_instructions(qa_result, rewrite_instructions)
    text = _strip_internal_labels(original)
    text = _apply_instruction_rewrites(text, instructions)
    if _needs_semantic_downgrade(instructions):
        text = _downgrade_semantic_claims(text)
    llm_text = _llm_rewrite(
        markdown=text,
        qa_result=_as_dict(qa_result),
        instructions=instructions,
        llm_config=llm_config,
    )
    if llm_text:
        text = llm_text
    text = sanitize_public_markdown(text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if text:
        text += "\n"
    return {
        "agent": AGENT_NAME,
        "report_markdown": text,
        "rewrite_instructions": instructions,
        "qa_issue_count": len(_as_list(_as_dict(qa_result).get("issues"))),
        "changed": text != original,
    }
