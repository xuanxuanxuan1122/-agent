from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

from rag_pipeline.config.search_config import build_llm_config_for_task
from rag_pipeline.search.memory import call_openai_compatible_json, llm_config_is_ready, normalize_llm_config


FINAL_AUDIT_SYSTEM_PROMPT = """You are the final audit model for a Chinese investment and industry research report.

Audit the report for publishability. Be strict. Do not rewrite the report.
Return only a valid JSON object with this shape:
{
  "status": "pass|warning|fatal",
  "overall_score": 0,
  "critical_findings": [
    {
      "type": "unsupported_claim|evidence_gap|citation_issue|data_conflict|logic_jump|risk_understated|scope_issue",
      "severity": "low|medium|high|fatal",
      "message": "short finding",
      "evidence_hint": "where to inspect",
      "suggested_fix": "concrete fix"
    }
  ],
  "unsupported_claims": [],
  "citation_issues": [],
  "scope_or_method_issues": [],
  "risk_section_feedback": [],
  "publish_recommendation": "publish|publish_with_caveats|hold",
  "summary": "short audit summary"
}

Use fatal only when the report should not be delivered as clean output without human repair.
Focus on missing sources, weak evidence, conflicting metric scope, inconsistent time windows,
investment conclusions that overreach the evidence, and risk sections that are too weak.
Also treat leaked internal pipeline markers (for example ch_01, policy_summary in a non-policy report,
绗?杞?traces, malformed metric tables, or source appendix gaps) as fatal unless clearly intentional.
"""


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, min_value: int = 0, max_value: int = 1_000_000) -> int:
    try:
        value = int(str(os.getenv(name, str(default))).strip())
    except (TypeError, ValueError):
        value = default
    return min(max_value, max(min_value, value))


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _compact_text(value: Any, max_chars: int = 600) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."


def _truncate_report(markdown: str, max_chars: int) -> Dict[str, Any]:
    text = str(markdown or "").strip()
    if len(text) <= max_chars:
        return {"text": text, "truncated": False, "original_chars": len(text), "included_chars": len(text)}
    head_chars = int(max_chars * 0.7)
    tail_chars = max_chars - head_chars
    truncated = (
        text[:head_chars].rstrip()
        + "\n\n[FINAL_AUDIT_TRUNCATED_MIDDLE]\n\n"
        + text[-tail_chars:].lstrip()
    )
    return {
        "text": truncated,
        "truncated": True,
        "original_chars": len(text),
        "included_chars": len(truncated),
    }


def _source_snapshot(clean_evidence: Optional[Dict[str, Any]], *, limit: int = 80) -> List[Dict[str, Any]]:
    sources = _as_list(_as_dict(clean_evidence).get("sources"))
    snapshot: List[Dict[str, Any]] = []
    for source in sources[:limit]:
        item = _as_dict(source)
        snapshot.append(
            {
                "id": item.get("id") or item.get("source_id") or item.get("ref"),
                "title": _compact_text(item.get("title") or item.get("source") or item.get("name"), 220),
                "url": str(item.get("url") or item.get("source_url") or "").strip(),
                "date": str(item.get("date") or item.get("published_at") or "").strip(),
                "source_type": str(item.get("source_type") or item.get("type") or "").strip(),
                "source_level": str(item.get("source_level") or item.get("credibility") or "").strip(),
            }
        )
    return snapshot


def _validation_snapshot(validation: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    payload = _as_dict(validation)
    keys = [
        "passed",
        "hard_pass",
        "quality_score",
        "fatal_blockers",
        "repair_blockers",
        "advisory_issues",
        "invalid_citations",
        "citation_count",
        "unique_cited_source_count",
        "source_pool_count",
        "body_chars_without_sources",
    ]
    return {key: payload.get(key) for key in keys if key in payload}


def _is_fatal_audit(payload: Dict[str, Any]) -> bool:
    if str(payload.get("status") or "").strip().lower() == "fatal":
        return True
    if str(payload.get("publish_recommendation") or "").strip().lower() == "hold":
        return True
    for item in _as_list(payload.get("critical_findings")):
        if str(_as_dict(item).get("severity") or "").strip().lower() == "fatal":
            return True
    return False


PUBLIC_EV_ID_RE = re.compile(r"(?<![A-Za-z0-9_])EV-\d+(?:-[A-Za-z0-9]+)?")
PUBLIC_CHAPTER_ID_RE = re.compile(r"(?<![A-Za-z0-9_])ch_\d{1,3}(?![A-Za-z0-9_])", re.I)
SOURCE_APPENDIX_RE = re.compile(
    r"(?mi)^##+\s*(?:数据来源|资料来源|研究口径与来源|参考来源|来源附录|Sources|References)(?:[^\n]*)?$"
)


FAKE_EVIDENCE_TEXT = "official data shows ai agent adoption reached 50% in 2025"


def _is_placeholder_url(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return bool(re.search(r"(?:^|[/:.])example\.(?:com|gov)(?:[/:]|$)", text))


def _is_fake_or_placeholder_source(source: Dict[str, Any]) -> bool:
    url = str(source.get("url") or source.get("source_url") or "").strip()
    title = str(source.get("title") or source.get("source_title") or source.get("name") or "").strip().lower()
    publisher = str(source.get("publisher") or source.get("source") or "").strip()
    if _is_placeholder_url(url):
        return True
    if title == "official" and not publisher and not url:
        return True
    source_text = " ".join(str(source.get(key) or "") for key in ("title", "source_title", "summary", "snippet")).lower()
    return FAKE_EVIDENCE_TEXT in source_text


def _source_candidates(package: Dict[str, Any], clean_evidence: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for source in _as_list(_as_dict(package.get("writer_report")).get("source_registry")):
        if isinstance(source, dict):
            candidates.append(source)
    for source in _as_list(package.get("source_registry")):
        if isinstance(source, dict):
            candidates.append(source)
    for source in _as_list(_as_dict(clean_evidence).get("sources")):
        if isinstance(source, dict):
            candidates.append(source)
    return candidates


def _empty_markdown_table_lines(markdown: str) -> List[int]:
    lines = str(markdown or "").splitlines()
    empty_tables: List[int] = []
    for index in range(len(lines) - 1):
        header = lines[index].strip()
        separator = lines[index + 1].strip()
        if not (header.startswith("|") and header.endswith("|") and separator.startswith("|") and re.match(r"^\|\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|\s*$", separator)):
            continue
        next_line = lines[index + 2].strip() if index + 2 < len(lines) else ""
        if not (next_line.startswith("|") and next_line.endswith("|")):
            empty_tables.append(index + 1)
    return empty_tables


def _dedupe_findings(findings: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    seen = set()
    for item in list(findings or []):
        if not isinstance(item, dict):
            continue
        examples = _as_list(item.get("examples"))
        example_key = ""
        if examples:
            first = _as_dict(examples[0])
            example_key = str(first.get("ref") or first.get("title") or "")[:120]
        key = (
            str(item.get("type") or ""),
            str(item.get("severity") or ""),
            str(item.get("message") or "")[:180],
            example_key,
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def run_deterministic_audit(
    *,
    report_markdown: str,
    validation: Optional[Dict[str, Any]] = None,
    clean_evidence: Optional[Dict[str, Any]] = None,
    writer_package_payload: Optional[Dict[str, Any]] = None,
    query: str = "",
) -> Dict[str, Any]:
    del validation, query
    text = str(report_markdown or "")
    package = _as_dict(writer_package_payload)
    writer_report = _as_dict(package.get("writer_report"))
    report_family = str(
        writer_report.get("report_type")
        or _as_dict(package.get("report_blueprint")).get("report_family")
        or _as_dict(writer_report.get("report_blueprint")).get("report_family")
        or ""
    ).strip().lower()
    findings: List[Dict[str, Any]] = []

    checks = [
        ("internal_evidence_cards", r"\bevidence_cards?\b"),
        ("internal_round_trace", r"(?:第\s*\d+\s*轮|绗.\s*\d+\s*杞.)"),
        ("internal_draft_instruction", r"(?:当前卡片|正文只能写成|本章只能写成|本章应写成|本章可以作为|本章可写成|本章可作为|建议写成|建议避免|适合写成|后续版本中补充|褰撳墠鍗＄墖|鏈珷搴斿啓鎴恷鏈珷鍙互浣滀负|鏈珷鍙綔涓簗寤鸿鍐欐垚|閫傚悎鍐欐垚)"),
        ("policy_profile_leak", r"\bpolicy_summary\b"),
    ]
    for finding_type, pattern in checks:
        if finding_type == "policy_profile_leak" and "policy" in report_family:
            continue
        hits = re.findall(pattern, text, flags=re.I)
        if hits:
            findings.append({"type": finding_type, "severity": "fatal", "message": f"Public report contains internal marker: {finding_type}", "count": len(hits)})

    ev_hits = PUBLIC_EV_ID_RE.findall(text)
    if ev_hits:
        findings.append({"type": "internal_evidence_id", "severity": "fatal", "message": "Public report contains EV-* evidence ids.", "examples": ev_hits[:5]})
    chapter_hits = PUBLIC_CHAPTER_ID_RE.findall(text)
    if chapter_hits:
        findings.append({"type": "internal_chapter_id", "severity": "fatal", "message": "Public report contains internal chapter ids.", "examples": chapter_hits[:5]})

    empty_tables = _empty_markdown_table_lines(text)
    if empty_tables:
        findings.append({"type": "empty_markdown_table", "severity": "fatal", "message": "Markdown table has a header/separator but no data rows.", "lines": empty_tables[:8]})

    if re.search(r"\[\d{1,3}\]", text) and not SOURCE_APPENDIX_RE.search(text):
        findings.append({"type": "missing_sources_appendix", "severity": "fatal", "message": "Report cites numbered sources but has no source appendix."})

    if _is_placeholder_url(text) or FAKE_EVIDENCE_TEXT in text.lower():
        findings.append({"type": "fake_or_placeholder_evidence", "severity": "fatal", "message": "Report contains placeholder source URLs or fake example evidence text."})

    title_only_sources = []
    fake_sources = []
    for source in _source_candidates(package, clean_evidence):
        if _is_fake_or_placeholder_source(source):
            fake_sources.append(
                {
                    "ref": source.get("ref") or source.get("id"),
                    "title": _compact_text(source.get("title") or source.get("source_title"), 160),
                    "url": source.get("url") or source.get("source_url"),
                }
            )
        url = str(source.get("url") or source.get("source_url") or "").strip()
        doc_id = str(source.get("document_id") or source.get("doc_id") or source.get("page_ref") or "").strip()
        title = str(source.get("title") or source.get("source_title") or "").strip()
        publisher = str(source.get("publisher") or source.get("source") or "").strip()
        date = str(source.get("date") or source.get("published_at") or "").strip()
        stable_local = bool(doc_id and sum(bool(item) for item in [publisher, title, date]) >= 2)
        if title and not url and not stable_local:
            title_only_sources.append({"ref": source.get("ref") or source.get("id"), "title": _compact_text(title, 160)})
    if title_only_sources:
        findings.append({"type": "title_only_source", "severity": "fatal", "message": "Source registry contains title-only sources that are not independently traceable.", "examples": title_only_sources[:8]})
    if fake_sources:
        findings.append({"type": "fake_or_placeholder_source", "severity": "fatal", "message": "Source registry contains placeholder or fake sources.", "examples": fake_sources[:8]})

    findings = _dedupe_findings(findings)
    fatal = any(str(item.get("severity") or "").lower() == "fatal" for item in findings)
    return {
        "status": "fatal" if fatal else "pass",
        "fatal": fatal,
        "findings": findings,
        "summary": "deterministic blockers found" if fatal else "deterministic audit passed",
    }


def _normalize_audit_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(payload or {})
    status = str(result.get("status") or "").strip().lower()
    if status not in {"pass", "warning", "fatal"}:
        status = "fatal" if _is_fatal_audit(result) else "warning"
    result["status"] = status
    recommendation = str(result.get("publish_recommendation") or "").strip().lower()
    if recommendation not in {"publish", "publish_with_caveats", "hold"}:
        recommendation = "hold" if status == "fatal" else "publish_with_caveats" if status == "warning" else "publish"
    result["publish_recommendation"] = recommendation
    try:
        score = int(float(result.get("overall_score") or 0))
    except (TypeError, ValueError):
        score = 0
    result["overall_score"] = max(0, min(100, score))
    for key in (
        "critical_findings",
        "unsupported_claims",
        "citation_issues",
        "scope_or_method_issues",
        "risk_section_feedback",
    ):
        result[key] = _as_list(result.get(key))
    result["summary"] = _compact_text(result.get("summary"), 1200)
    return result


def run_final_audit(
    *,
    report_markdown: str,
    validation: Optional[Dict[str, Any]] = None,
    clean_evidence: Optional[Dict[str, Any]] = None,
    writer_package_payload: Optional[Dict[str, Any]] = None,
    query: str = "",
) -> Dict[str, Any]:
    if not _env_flag("REPORT_ENABLE_FINAL_AUDIT", True):
        return {"enabled": False, "success": True, "status": "skipped", "skipped_reason": "disabled"}

    package = _as_dict(writer_package_payload)
    deterministic_audit = run_deterministic_audit(
        report_markdown=report_markdown,
        validation=validation,
        clean_evidence=clean_evidence,
        writer_package_payload=package,
        query=query,
    )
    blocking = _env_flag("REPORT_FINAL_AUDIT_BLOCKING", True)
    if deterministic_audit.get("fatal") and blocking:
        return {
            "enabled": True,
            "success": True,
            "status": "fatal",
            "blocked": True,
            "blocking": blocking,
            "deterministic_audit": deterministic_audit,
            "audit": {
                "status": "fatal",
                "overall_score": 0,
                "critical_findings": _as_list(deterministic_audit.get("findings")),
                "publish_recommendation": "hold",
                "summary": deterministic_audit.get("summary"),
            },
        }

    config = dict(build_llm_config_for_task("final_audit"))
    normalized = normalize_llm_config(config)
    if not llm_config_is_ready(normalized):
        return {
            "enabled": True,
            "success": False,
            "status": "skipped",
            "blocked": False,
            "blocking": blocking,
            "skipped_reason": "config_missing",
            "model": normalized.get("model") or "",
            "deterministic_audit": deterministic_audit,
        }

    max_chars = _env_int("REPORT_FINAL_AUDIT_MAX_CHARS", 200000, min_value=4000, max_value=1_000_000)
    report_payload = _truncate_report(report_markdown, max_chars)
    user_payload = {
        "query": query,
        "report_markdown": report_payload["text"],
        "report_truncation": {
            "truncated": report_payload["truncated"],
            "original_chars": report_payload["original_chars"],
            "included_chars": report_payload["included_chars"],
        },
        "reformatter_validation": _validation_snapshot(validation),
        "source_snapshot": _source_snapshot(clean_evidence),
        "evidence_handoff_diagnostics": _as_dict(package.get("evidence_handoff_diagnostics")),
        "qa_blocker_summary": _as_dict(package.get("qa_blocker_summary")),
        "quality_gate_state": _as_dict(package.get("quality_gate_state")),
    }
    try:
        response = call_openai_compatible_json(
            config=config,
            system_prompt=FINAL_AUDIT_SYSTEM_PROMPT,
            user_payload=user_payload,
        )
        audit_payload = _normalize_audit_payload(_as_dict(response.get("payload")))
        fatal = _is_fatal_audit(audit_payload) or bool(deterministic_audit.get("fatal"))
        return {
            "enabled": True,
            "success": True,
            "status": "fatal" if deterministic_audit.get("fatal") else audit_payload["status"],
            "blocked": bool(blocking and fatal),
            "blocking": blocking,
            "model": normalized.get("model") or "",
            "reasoning_effort": normalized.get("reasoning_effort") or "",
            "max_output_tokens": int(normalized.get("max_output_tokens") or 0),
            "audit": audit_payload,
            "deterministic_audit": deterministic_audit,
            "usage": response.get("usage") or {},
            "llm_call": _as_dict(response.get("llm_call")),
            "report_truncation": user_payload["report_truncation"],
        }
    except Exception as exc:
        diagnostic = _as_dict(getattr(exc, "diagnostic", {}))
        return {
            "enabled": True,
            "success": False,
            "status": "failed",
            "blocked": bool(blocking and deterministic_audit.get("fatal")),
            "blocking": blocking,
            "model": normalized.get("model") or "",
            "reasoning_effort": normalized.get("reasoning_effort") or "",
            "error": str(exc),
            "deterministic_audit": deterministic_audit,
            "llm_call": diagnostic,
            "report_truncation": user_payload["report_truncation"],
        }
