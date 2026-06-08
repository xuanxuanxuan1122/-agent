from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence


LLM_REVIEW_SCHEMA_VERSION = "0.2.0"


LLM_REVIEW_SYSTEM_PROMPT = """你是一个行业研究报告的审稿编辑，只能做最小化、可校验的清理。

## 你的任务
对输入的报告文本进行以下修复，直接输出修复后的 Markdown 片段，不要输出任何解释。不要自由重写整篇报告。

### 必须修复
1. 删除所有推理过程文字
- 删除任何以"这一信息的价值"、"后续应补充"、"该证据的核心口径"开头的句子
- 删除任何包含"时间口径为20XX-XX"的句子
- 删除任何解读模板文字，判断原则是：这句话描述的是"分析方法"而非"事实结论"

2. 修复空白章节
- 如果某个 ### 或 #### 标题下方没有内容，删除该空标题；不要添加“材料不足”“后续补充”等占位说明。

3. 修复风险触发器重复
- 如果风险触发器列表中有多条完全相同的内容，合并为一条。

4. 修复截断引用
- 如果某个 bullet 中有引用标签，但前面的文字用 ... 被截断，删除该 bullet。

5. 修复无意义 bullet
- 删除内容只有字段名+数字的 bullet，例如 "- 估值0[12]"、"- 市场规模2[12]"。

### 禁止操作
- 不要修改任何数据数字
- 不要添加报告中没有的新事实
- 不要新增公司名、日期、政策名、来源编号或引用
- 不要根据 QA 或审稿意见自行补事实
- 不要改变报告结构，包括章节顺序和标题层级
- 不要删除有实质内容的段落
- 如果证据不足，只能删除或降级表达，不能扩写

### 输出格式
直接输出修复后的完整 Markdown 报告，从 # 标题开始。"""


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _compact(value: Any, max_chars: int = 260) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "..."


def _fingerprint(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "").lower())


def _valid_source_ids(evidence: Optional[Dict[str, Any]]) -> set[str]:
    evidence = _as_dict(evidence)
    ids: set[str] = set()
    for source in _as_list(evidence.get("sources")):
        if not isinstance(source, dict):
            continue
        for key in ("id", "ref", "source_id", "citation_ref"):
            value = str(source.get(key) or "").strip()
            if not value:
                continue
            match = re.search(r"\[(\d{1,5})\]", value)
            ids.add(match.group(1) if match else value.strip("[]"))
    return ids


def split_markdown_for_review(markdown: str, *, max_chars: int = 12000) -> List[Dict[str, Any]]:
    """Split a report into bounded markdown chunks without losing headings."""

    text = str(markdown or "").strip()
    if not text:
        return []
    max_chars = max(1200, int(max_chars or 12000))
    blocks: List[str] = []
    current: List[str] = []
    for line in text.splitlines(keepends=True):
        if re.match(r"^#{1,3}\s+\S", line) and current:
            blocks.append("".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        blocks.append("".join(current).strip())

    chunks: List[Dict[str, Any]] = []
    active = ""
    for block in blocks:
        if not block:
            continue
        if len(block) > max_chars:
            for part in _split_long_block(block, max_chars=max_chars):
                if active:
                    chunks.append(_chunk_payload(active, len(chunks) + 1))
                    active = ""
                chunks.append(_chunk_payload(part, len(chunks) + 1))
            continue
        if active and len(active) + len(block) + 2 > max_chars:
            chunks.append(_chunk_payload(active, len(chunks) + 1))
            active = block
        else:
            active = f"{active}\n\n{block}".strip() if active else block
    if active:
        chunks.append(_chunk_payload(active, len(chunks) + 1))
    return chunks


def _split_long_block(block: str, *, max_chars: int) -> List[str]:
    parts: List[str] = []
    current = ""
    for paragraph in re.split(r"\n{2,}", str(block or "")):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) > max_chars:
            if current:
                parts.append(current.strip())
                current = ""
            for start in range(0, len(paragraph), max_chars):
                parts.append(paragraph[start : start + max_chars].strip())
            continue
        if current and len(current) + len(paragraph) + 2 > max_chars:
            parts.append(current.strip())
            current = paragraph
        else:
            current = f"{current}\n\n{paragraph}".strip() if current else paragraph
    if current:
        parts.append(current.strip())
    return parts


def _chunk_payload(text: str, index: int) -> Dict[str, Any]:
    heading = ""
    match = re.search(r"(?m)^#{1,3}\s+(.+?)\s*$", text)
    if match:
        heading = match.group(1).strip()
    return {
        "chunk_id": f"review_chunk_{index:03d}",
        "heading": heading,
        "input_chars": len(text),
        "text": text,
    }


def detect_review_findings(report_text: str, evidence: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    text = str(report_text or "")
    findings: List[Dict[str, Any]] = []
    findings.extend(_detect_empty_sections(text))
    findings.extend(_detect_truncated_citations(text))
    findings.extend(_detect_meaningless_bullets(text))
    findings.extend(_detect_duplicate_paragraphs(text))
    findings.extend(_detect_invalid_citations(text, evidence))
    findings.extend(_detect_possible_logic_gaps(text))
    return findings


def _detect_empty_sections(text: str) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if not re.match(r"^#{2,4}\s+\S", line.strip()):
            continue
        body_lines: List[str] = []
        for following in lines[index + 1 :]:
            if re.match(r"^#{1,4}\s+\S", following.strip()):
                break
            if following.strip():
                body_lines.append(following.strip())
        if not body_lines:
            findings.append(
                {
                    "type": "empty_section",
                    "severity": "medium",
                    "line": index + 1,
                    "message": "章节标题下缺少正文内容",
                    "evidence_required": False,
                    "rewrite_required": True,
                }
            )
    return findings


def _detect_truncated_citations(text: str) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if re.search(r"\.\.\.\s*(?:\[\d{1,5}\])?\s*$", stripped) or re.search(r"…\s*(?:\[\d{1,5}\])?\s*$", stripped):
            findings.append(
                {
                    "type": "truncated_citation",
                    "severity": "high",
                    "line": line_no,
                    "message": "存在疑似截断的引用句",
                    "evidence_required": False,
                    "rewrite_required": True,
                }
            )
    return findings


def _detect_meaningless_bullets(text: str) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    pattern = re.compile(r"^\s*[-*+]\s*(?:[\u4e00-\u9fffA-Za-z]{1,12})?\s*-?\d+(?:\.\d+)?%?\s*(?:\[\d{1,5}\])?\s*$")
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern.match(line):
            findings.append(
                {
                    "type": "meaningless_bullet",
                    "severity": "medium",
                    "line": line_no,
                    "message": "项目符号缺少可读事实",
                    "evidence_required": False,
                    "rewrite_required": True,
                }
            )
    return findings


def _detect_duplicate_paragraphs(text: str) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    seen: Dict[str, int] = {}
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("|") or len(line) < 28:
            continue
        key = _fingerprint(re.sub(r"\[\d{1,5}\]", "", line))
        if key in seen:
            findings.append(
                {
                    "type": "duplicate_paragraph",
                    "severity": "low",
                    "line": line_no,
                    "first_line": seen[key],
                    "message": "正文段落重复",
                    "evidence_required": False,
                    "rewrite_required": True,
                }
            )
        else:
            seen[key] = line_no
    return findings


def _detect_invalid_citations(text: str, evidence: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    valid_ids = _valid_source_ids(evidence)
    if not valid_ids:
        return []
    findings: List[Dict[str, Any]] = []
    cited = list(dict.fromkeys(re.findall(r"\[(\d{1,5})\]", text)))
    invalid = [item for item in cited if item not in valid_ids]
    for source_id in invalid[:20]:
        findings.append(
            {
                "type": "invalid_citation",
                "severity": "high",
                "source_id": source_id,
                "message": f"正文引用 [{source_id}] 不在证据来源列表中",
                "evidence_required": True,
                "rewrite_required": True,
            }
        )
    return findings


def _detect_possible_logic_gaps(text: str) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if len(stripped) < 20 or re.search(r"\[\d{1,5}\]", stripped):
            continue
        if re.search(r"(因此|由此|说明|意味着|可以判断|可见|核心结论是|\btherefore\b|\bimplies\b|\bsuggests\b|\bindicates\b)", stripped, re.I):
            findings.append(
                {
                    "type": "uncited_inference",
                    "severity": "medium",
                    "line": line_no,
                    "message": "存在没有正文引用支撑的推断句",
                    "evidence_required": True,
                    "rewrite_required": True,
                }
            )
    return findings[:20]


def build_structured_review(
    *,
    original_report: str,
    revised_report: Optional[str] = None,
    evidence: Optional[Dict[str, Any]] = None,
    chunks: Optional[Sequence[Dict[str, Any]]] = None,
    llm_used: bool = False,
    stage2_skipped: bool = False,
    stage2_reason: str = "",
    errors: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    revised = str(revised_report if revised_report is not None else original_report or "")
    findings = detect_review_findings(revised, evidence)
    citation_issues = [item for item in findings if str(item.get("type") or "").endswith("citation") or item.get("type") == "invalid_citation"]
    logic_issues = [item for item in findings if item.get("type") in {"uncited_inference", "duplicate_paragraph"}]
    evidence_followups = [
        {
            "type": item.get("type"),
            "line": item.get("line"),
            "source_id": item.get("source_id"),
            "suggested_action": "补充可核验证据或删除该推断",
        }
        for item in findings
        if item.get("evidence_required")
    ]
    rewrite_required = any(bool(item.get("rewrite_required")) for item in findings)
    evidence_required = any(bool(item.get("evidence_required")) for item in findings)
    high_count = sum(1 for item in findings if item.get("severity") == "high")
    medium_count = sum(1 for item in findings if item.get("severity") == "medium")
    low_count = sum(1 for item in findings if item.get("severity") == "low")
    score = max(0, 100 - high_count * 18 - medium_count * 9 - low_count * 3)
    status = "passed"
    if evidence_required:
        status = "needs_evidence"
    elif rewrite_required:
        status = "needs_rewrite"
    return {
        "schema_version": LLM_REVIEW_SCHEMA_VERSION,
        "status": status,
        "passed": status == "passed",
        "quality_score": score,
        "llm_used": bool(llm_used),
        "stage2_skipped": bool(stage2_skipped),
        "stage2_reason": str(stage2_reason or ""),
        "original_chars": len(str(original_report or "")),
        "revised_chars": len(revised),
        "chunk_count": len(list(chunks or [])),
        "chunks": [
            {
                "chunk_id": item.get("chunk_id"),
                "heading": item.get("heading"),
                "input_chars": item.get("input_chars"),
                "output_chars": len(str(item.get("output_text") or item.get("text") or "")),
            }
            for item in list(chunks or [])
            if isinstance(item, dict)
        ],
        "findings": findings,
        "citation_issues": citation_issues,
        "logic_issues": logic_issues,
        "evidence_followups": evidence_followups,
        "rewrite_required": rewrite_required,
        "evidence_required": evidence_required,
        "errors": [str(item) for item in list(errors or []) if str(item or "").strip()],
        "revised_report": revised,
    }


async def _invoke_review_chunk(chunk_text: str, llm_client: Any) -> str:
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=LLM_REVIEW_SYSTEM_PROMPT),
            HumanMessage(content=f"请修复以下报告片段，保持 Markdown 结构：\n\n{chunk_text}"),
        ]
    except Exception:
        messages = [
            {"role": "system", "content": LLM_REVIEW_SYSTEM_PROMPT},
            {"role": "user", "content": f"请修复以下报告片段，保持 Markdown 结构：\n\n{chunk_text}"},
        ]
    response = await llm_client.ainvoke(messages)
    return str(getattr(response, "content", response) or chunk_text)


async def llm_review_structured(
    report_text: str,
    llm_client: Any = None,
    *,
    evidence: Optional[Dict[str, Any]] = None,
    max_chunk_chars: int = 12000,
) -> Dict[str, Any]:
    chunks = split_markdown_for_review(report_text, max_chars=max_chunk_chars)
    if llm_client is None:
        return build_structured_review(
            original_report=report_text,
            revised_report=report_text,
            evidence=evidence,
            chunks=chunks,
            llm_used=False,
            stage2_skipped=True,
            stage2_reason="llm_client is None",
        )

    revised_chunks: List[str] = []
    errors: List[str] = []
    for chunk in chunks or [{"chunk_id": "review_chunk_001", "heading": "", "input_chars": len(report_text or ""), "text": report_text}]:
        try:
            output_text = await _invoke_review_chunk(str(chunk.get("text") or ""), llm_client)
            chunk["output_text"] = output_text
            revised_chunks.append(output_text)
        except Exception as exc:
            errors.append(f"{chunk.get('chunk_id')}: {exc}")
            fallback_text = str(chunk.get("text") or "")
            chunk["output_text"] = fallback_text
            revised_chunks.append(fallback_text)
    revised_report = "\n\n".join(part.strip() for part in revised_chunks if str(part or "").strip()).strip()
    return build_structured_review(
        original_report=report_text,
        revised_report=revised_report or report_text,
        evidence=evidence,
        chunks=chunks,
        llm_used=True,
        stage2_skipped=bool(errors),
        stage2_reason="; ".join(errors[:3]),
        errors=errors,
    )


async def llm_review(report_text: str, llm_client: Any) -> str:
    result = await llm_review_structured(report_text, llm_client)
    return str(result.get("revised_report") or report_text)
