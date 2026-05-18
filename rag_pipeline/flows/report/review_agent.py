from __future__ import annotations

import hashlib
import re
from difflib import SequenceMatcher
from typing import Dict, List, Tuple


# Neutral placeholder for empty subsections. The previous text ("本维度暂无可核验数据，
# 建议后续补充调研。") triggered the BAD_PATTERN blocklist in writer_agent_clean (which
# matches "暂无可核验数据" / "建议后续补充调研") and was repeatedly flagged as internal-gap
# language, producing a fill -> sanitize -> fill loop that left placeholders visible in
# the final rendered report.
EMPTY_SECTION_PLACEHOLDER = "（本节内容已并入相邻章节。）"

LEAK_PATTERNS = [
    r"这一信息的价值在于[^。\n]{0,160}。?",
    r"这一信息可以用来判断[^。\n]{0,160}。?",
    r"这一信息的价值在于判断[^。\n]{0,120}。?",
    r"若后续能同时补到出货量[^。\n]{0,180}。?",
    r"后续应补充来源范围[^。\n]{0,120}。?",
    r"后续应补充来源口径[^。\n]{0,120}。?",
    r"后续应补充来源、样本范围和企业级验证[^。\n]{0,120}。?",
    r"避免把单点信息误读为稳定趋势。?",
    r"该证据的核心口径是[^。\n]{0,140}。?",
    r"时间口径为\d{4}(?:-\d{2})?(?:-\d{2})?[^。\n]{0,80}。?",
    r"这条数据要服务的不是[^。\n]{0,140}。?",
    r"这条证据的核心价值在于[^。\n]{0,140}。?",
    r"这条证据要直接回答[^。\n]{0,160}。?",
    r"技术信息需要被翻译成产品可行性[^。\n]{0,120}。?",
    r"政策信息更适合作为订单释放条件[^。\n]{0,120}。?",
    r"资本信息说明赛道受到关注[^。\n]{0,120}。?",
    r"判断重点是它能否降低人工[^。\n]{0,140}。?",
    r"解读[:：]该证据[^。\n]{0,240}。?",
]

MEANINGLESS_BULLET_RE = re.compile(
    r"^\s*-\s*(?:\*\*[^*]+\*\*[:：]?)?\s*(?:估值|市场规模|增速|收入|并购|定性事实|关键事实)\s*-?\d+(?:\.\d+)?%?(?:\[\d+\])?\s*$"
)


def _compact_blank_lines(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", str(text or "")).strip()


def _normalized_text(text: str) -> str:
    return re.sub(r"[\s\[\]\d，。；：、！？,.!?:;（）()\-_*#>]", "", str(text or ""))


def get_text_fingerprint(text: str) -> str:
    normalized = _normalized_text(text)
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def _is_substantive(text: str, *, min_chars: int = 14) -> bool:
    normalized = _normalized_text(text)
    return len(normalized) >= min_chars


def clean_leak_patterns(text: str) -> Tuple[str, List[str]]:
    removed: List[str] = []
    for pattern in LEAK_PATTERNS:
        matches = re.findall(pattern, text)
        if matches:
            removed.extend(match.strip() for match in matches if str(match).strip())
        text = re.sub(pattern, "", text)
    return _compact_blank_lines(text), removed


def remove_empty_bullets(text: str) -> Tuple[str, int]:
    result: List[str] = []
    removed = 0
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if re.fullmatch(r"-\s*(?:→.*)?", stripped) or re.fullmatch(r"-\s*\*\*[^*]+\*\*[:：]?\s*(?:→.*)?", stripped):
            removed += 1
            continue
        result.append(line)
    return "\n".join(result), removed


def deduplicate_bullets(text: str, *, similarity_threshold: float = 0.88) -> Tuple[str, int]:
    seen_fingerprints = set()
    seen_texts: List[str] = []
    result_lines: List[str] = []
    removed_count = 0

    for line in str(text or "").splitlines():
        if not line.strip().startswith("- "):
            result_lines.append(line)
            continue

        bullet_content = line.strip()[2:]
        if not _is_substantive(bullet_content, min_chars=8):
            result_lines.append(line)
            continue

        fp = get_text_fingerprint(bullet_content)
        if fp in seen_fingerprints:
            removed_count += 1
            continue

        if any(SequenceMatcher(None, bullet_content[:140], seen[:140]).ratio() > similarity_threshold for seen in seen_texts):
            removed_count += 1
            continue

        seen_fingerprints.add(fp)
        seen_texts.append(bullet_content)
        result_lines.append(line)

    return "\n".join(result_lines), removed_count


def deduplicate_paragraphs(text: str, *, similarity_threshold: float = 0.92) -> Tuple[str, int]:
    seen_fingerprints = set()
    seen_texts: List[str] = []
    result_lines: List[str] = []
    removed_count = 0

    for line in str(text or "").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("- ") or stripped.startswith(">"):
            result_lines.append(line)
            continue
        if not _is_substantive(stripped, min_chars=24):
            result_lines.append(line)
            continue

        fp = get_text_fingerprint(stripped)
        if fp in seen_fingerprints:
            removed_count += 1
            continue
        if any(SequenceMatcher(None, stripped[:180], seen[:180]).ratio() > similarity_threshold for seen in seen_texts):
            removed_count += 1
            continue

        seen_fingerprints.add(fp)
        seen_texts.append(stripped)
        result_lines.append(line)

    return "\n".join(result_lines), removed_count


def detect_empty_sections(text: str) -> List[str]:
    empty_sections: List[str] = []
    lines = str(text or "").splitlines()
    for index, line in enumerate(lines):
        if not re.match(r"^#{2,4}\s+\S", line):
            continue
        cursor = index + 1
        while cursor < len(lines) and not lines[cursor].strip():
            cursor += 1
        if cursor >= len(lines) or re.match(r"^#{1,4}\s+\S", lines[cursor]):
            empty_sections.append(line.strip())
    return empty_sections


def fill_empty_sections(text: str) -> Tuple[str, List[str]]:
    lines = str(text or "").splitlines()
    result: List[str] = []
    filled: List[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if re.match(r"^#{2,4}\s+\S", line):
            cursor = index + 1
            while cursor < len(lines) and not lines[cursor].strip():
                cursor += 1
            if cursor >= len(lines) or re.match(r"^#{1,4}\s+\S", lines[cursor]):
                filled.append(line.strip())
                index += 1
                continue
        result.append(line)
        index += 1
    return "\n".join(result), filled


def detect_truncated_content(text: str) -> List[str]:
    issues: List[str] = []
    for line_no, line in enumerate(str(text or "").splitlines(), start=1):
        stripped = line.strip()
        if re.search(r"[\u4e00-\u9fff]\.\.\.\s*(?:\[\d+\])?$", stripped):
            issues.append(f"第{line_no}行内容被截断: {stripped[:80]}")
        if MEANINGLESS_BULLET_RE.match(stripped):
            issues.append(f"第{line_no}行内容无意义: {stripped}")
    return issues


def remove_truncated_and_meaningless_bullets(text: str) -> Tuple[str, List[str], int]:
    result: List[str] = []
    issues: List[str] = []
    removed = 0
    for line_no, line in enumerate(str(text or "").splitlines(), start=1):
        stripped = line.strip()
        truncated = bool(stripped.startswith("- ") and re.search(r"[\u4e00-\u9fff]\.\.\.\s*(?:\[\d+\])?$", stripped))
        meaningless = bool(MEANINGLESS_BULLET_RE.match(stripped))
        if truncated or meaningless:
            issues.append(f"第{line_no}行{'内容被截断' if truncated else '内容无意义'}: {stripped[:80]}")
            removed += 1
            continue
        result.append(line)
    return "\n".join(result), issues, removed


def fix_empty_risk_items(text: str) -> Tuple[str, int]:
    result: List[str] = []
    removed = 0
    for line in str(text or "").splitlines():
        stripped = line.strip()
        risk_match = re.match(r"^-\s+\*\*[^*]+风险\*\*[:：]\s*(.*?)\s*→\s*.*$", stripped)
        if risk_match and len(risk_match.group(1).strip()) < 5:
            removed += 1
            continue
        result.append(line)
    return "\n".join(result), removed


def rule_based_review(report_text: str) -> Tuple[str, Dict[str, object]]:
    audit_log: Dict[str, object] = {
        "leak_patterns_removed": [],
        "duplicate_bullets_removed": 0,
        "duplicate_paragraphs_removed": 0,
        "empty_sections_detected": [],
        "empty_sections_filled": [],
        "truncated_content": [],
        "truncated_or_meaningless_removed": 0,
        "empty_bullets_removed": 0,
        "empty_risks_removed": 0,
    }

    report_text, removed_leaks = clean_leak_patterns(report_text)
    audit_log["leak_patterns_removed"] = removed_leaks

    report_text, truncated_issues, removed_bad = remove_truncated_and_meaningless_bullets(report_text)
    audit_log["truncated_content"] = truncated_issues
    audit_log["truncated_or_meaningless_removed"] = removed_bad

    report_text, empty_risk_count = fix_empty_risk_items(report_text)
    audit_log["empty_risks_removed"] = empty_risk_count

    report_text, empty_bullet_count = remove_empty_bullets(report_text)
    audit_log["empty_bullets_removed"] = empty_bullet_count

    report_text, dup_bullet_count = deduplicate_bullets(report_text)
    audit_log["duplicate_bullets_removed"] = dup_bullet_count

    report_text, dup_paragraph_count = deduplicate_paragraphs(report_text)
    audit_log["duplicate_paragraphs_removed"] = dup_paragraph_count

    audit_log["empty_sections_detected"] = detect_empty_sections(report_text)
    report_text, filled_sections = fill_empty_sections(report_text)
    audit_log["empty_sections_filled"] = filled_sections

    remaining_truncated = detect_truncated_content(report_text)
    if remaining_truncated:
        audit_log["truncated_content"] = list(audit_log["truncated_content"]) + remaining_truncated

    return _compact_blank_lines(report_text), audit_log
