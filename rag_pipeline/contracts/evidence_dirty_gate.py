from __future__ import annotations

import re
from typing import Any, Dict, List


PAGE_SHELL_RE = re.compile(
    r"(login|sign\s*in|cookie|privacy\s*policy|terms\s*of\s*use|"
    r"please\s+enable\s+javascript|checking\s+your\s+browser|request\s+a\s+demo|"
    r"book\s+a\s+demo|跳转|登录|注册|隐私政策|用户协议|无障碍浏览)",
    re.I,
)

INTERNAL_MARKER_RE = re.compile(
    r"\b(claim_unit|score_gap|diagnostic_only|analysis_ready_exclusion_reason|llm_semantic_judge)\b",
    re.I,
)

PLACEHOLDER_URL_RE = re.compile(r"(example\.(?:com|org|net)|placeholder|localhost|127\.0\.0\.1)", re.I)


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _text(item: Dict[str, Any]) -> str:
    source = _as_dict(item.get("source"))
    return " ".join(
        str(value or "")
        for value in (
            item.get("fact"),
            item.get("clean_fact"),
            item.get("content"),
            item.get("distilled_fact"),
            item.get("metric"),
            item.get("value"),
            source.get("title"),
            source.get("url"),
            item.get("source_url"),
        )
    )


def _source_url(item: Dict[str, Any]) -> str:
    source = _as_dict(item.get("source"))
    return str(item.get("source_url") or source.get("url") or source.get("source_url") or "").strip()


def _has_source_ref(item: Dict[str, Any]) -> bool:
    source = _as_dict(item.get("source"))
    return bool(
        _source_url(item)
        or str(item.get("source_ref") or item.get("source_id") or item.get("run_source_id") or "").strip()
        or str(source.get("source_ref") or source.get("document_id") or source.get("page_ref") or "").strip()
    )


def _looks_like_pure_title_or_number(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return True
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?%?", compact):
        return True
    return bool(len(compact) < 12 and not re.search(r"[。；，,;:：]", text))


def evaluate_dirty_gate(item: Dict[str, Any]) -> Dict[str, Any]:
    """Block only data that should never reach public analysis.

    Weak source level, single source, and incomplete metric fields are warnings.
    They should be handled by analysis/writing as boundaries, not by hard
    deletion before the model can reason over the evidence.
    """

    text = _text(item)
    url = _source_url(item)
    reasons: List[str] = []
    warnings: List[str] = []

    if not _has_source_ref(item):
        reasons.append("missing_source_ref")
    if url and PLACEHOLDER_URL_RE.search(url):
        reasons.append("placeholder_or_fake_url")
    if PAGE_SHELL_RE.search(text):
        reasons.append("page_shell_or_login")
    if INTERNAL_MARKER_RE.search(text):
        reasons.append("internal_diagnostic_marker")
    if _looks_like_pure_title_or_number(
        str(item.get("fact") or item.get("clean_fact") or item.get("distilled_fact") or item.get("content") or "")
    ):
        warnings.append("thin_or_title_like_text")

    metric = str(item.get("metric") or "").strip()
    value = str(item.get("value") or "").strip()
    unit = str(item.get("unit") or "").strip()
    period = str(item.get("period") or "").strip()
    if metric and (not value or not unit or not period):
        warnings.append("metric_incomplete")

    status = "blocked" if reasons else "allowed"
    return {
        "schema_version": "dirty_gate_v1",
        "status": status,
        "reasons": reasons,
        "warnings": warnings,
        "diagnostic_only": status == "blocked",
        "must_not_render": status == "blocked",
        "public_text_allowed": status != "blocked",
    }
