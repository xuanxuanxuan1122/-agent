from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple


_METRIC_FRAGMENT_RE = re.compile(
    r"^\s*[\w\u4e00-\u9fff/+\-\s]{1,24}\s*[:：]\s*[\d,.]+(?:\s*(?:%|％|亿元|万元|亿美元|家|个|年|台|套|pct))?\s*[。.]?\s*$",
    re.I,
)
_SEMICOLON_FRAGMENT_RE = re.compile(
    r"^\s*[\w\u4e00-\u9fff/+\-\s]{1,16}\s*[;；]\s*(?:20\d{2}年?|Q[1-4]|[\d,.]+(?:%|％)?)\s*$",
    re.I,
)
_BARE_TIME_OR_VALUE_RE = re.compile(r"^\s*(?:20\d{2}年?|Q[1-4]|[\d,.]+(?:%|％|亿元|万元|亿美元|家|个|台|套)?)\s*$", re.I)
_URL_OR_REF_RE = re.compile(r"https?://|www\.|(?<![A-Za-z0-9_])EV-\d+|evidence_cards?|claim_status", re.I)
_INTERNAL_RE = re.compile(
    r"证据不足|建议补证|正文应以|后续验证|可追溯来源继续校准|只能形成方向性观察|"
    r"Clean\s*资格|QA\s*failed|fatal|source appendix|table_validation|"
    r"本章应写成|本章只能写成|正文只能写成|建议避免",
    re.I,
)
_ACTION_RE = re.compile(
    r"正在|转向|取决|推动|说明|显示|表明|意味着|集中|制约|改善|扩大|收缩|验证|进入|形成|决定|"
    r"支撑|反映|提升|下降|分化|变化|约束|来自|依赖|受.+影响|需要|仍|但|而|"
    r"shows?|indicates?|suggests?|depends?|moves?|shifts?|drives?",
    re.I,
)
_SOURCE_TITLE_RE = re.compile(r"^[^。；;]{4,50}[-—_][^。；;:：]{2,24}[:：]")


def _compact(value: Any, max_chars: int = 260) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "..."


def summary_judgment_text(item: Any) -> str:
    if isinstance(item, dict):
        return _compact(item.get("judgment") or item.get("claim") or item.get("text") or item.get("summary"), 260)
    return _compact(item, 260)


def is_valid_summary_judgment(value: Any) -> bool:
    text = summary_judgment_text(value)
    if not text:
        return False
    if _URL_OR_REF_RE.search(text) or _INTERNAL_RE.search(text):
        return False
    if _METRIC_FRAGMENT_RE.match(text) or _SEMICOLON_FRAGMENT_RE.match(text) or _BARE_TIME_OR_VALUE_RE.match(text):
        return False
    if _SOURCE_TITLE_RE.match(text):
        return False
    if len(re.sub(r"[^\w\u4e00-\u9fff]", "", text)) < 14:
        return False
    if not _ACTION_RE.search(text):
        return False
    # A useful executive judgment should not be only a source/title/table cell.
    if text.count("；") >= 3 and not re.search(r"但|而|因为|取决|说明|显示|意味着|therefore|because", text, re.I):
        return False
    return True


def sanitize_summary_judgments(
    items: Sequence[Any],
    *,
    max_items: int = 3,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    valid: List[Dict[str, Any]] = []
    filtered_examples: List[str] = []
    seen = set()
    for item in list(items or []):
        payload = dict(item) if isinstance(item, dict) else {"judgment": item}
        text = summary_judgment_text(payload)
        key = re.sub(r"\s+", "", text).lower()
        if not text or key in seen:
            continue
        seen.add(key)
        if not is_valid_summary_judgment(text):
            if len(filtered_examples) < 5:
                filtered_examples.append(_compact(text, 120))
            continue
        payload["judgment"] = text
        valid.append(payload)
        if len(valid) >= max_items:
            break
    diagnostics = {
        "executive_summary_valid_judgment_count": len(valid),
        "executive_summary_filtered_judgment_count": max(0, len(seen) - len(valid)),
        "executive_summary_omitted_low_quality": not bool(valid),
        "filtered_summary_examples": filtered_examples,
    }
    return valid, diagnostics
