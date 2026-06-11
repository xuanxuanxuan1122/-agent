from __future__ import annotations

import os
import re
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from rag_pipeline.config.search_config import build_llm_config_for_task
from rag_pipeline.contracts.quality_gate_policy import quality_gate_mode, quality_gates_isolated
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
      "requirement_id": "optional requirement id",
      "gap_id": "optional gap id",
      "section_id": "optional section id",
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
When a finding maps to a known gap, include requirement_id, gap_id, and section_id so the repair ledger can create the next search task.
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


def _final_audit_mode() -> str:
    raw = str(
        os.getenv("REPORT_FINAL_AUDIT_MODE")
        or os.getenv("REPORT_DELIVERY_MODE")
        or "draft_complete"
    ).strip().lower()
    if raw in {"strict", "publish", "publish_strict", "publication", "release"}:
        return "publish_strict"
    if raw in {"audit_only", "observe", "diagnostic", "diagnostic_only"}:
        return "audit_only"
    return "draft_complete"


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


def _compact_retry_plan_for_audit(value: Any) -> Dict[str, Any]:
    payload = _as_dict(value)
    allowed = {
        "query",
        "query_terms",
        "proof_role",
        "required_fields",
        "required_source_level",
        "lane_targets",
        "success_criteria",
        "reject_if",
        "current_insufficiency",
        "repair_route",
        "source_stage",
        "live_refresh_required_count",
        "result_count",
        "signal_count",
    }
    return {
        key: payload.get(key)
        for key in allowed
        if payload.get(key) not in (None, "", [], {})
    }


def _compact_score_gaps_for_audit(package: Dict[str, Any], writer_report: Dict[str, Any], *, limit: int = 40) -> List[Dict[str, Any]]:
    raw_gaps = (
        _as_list(package.get("score_gaps"))
        or _as_list(package.get("score_gap_ledger"))
        or _as_list(writer_report.get("score_gaps"))
        or _as_list(_as_dict(package.get("evidence_package")).get("evidence_gap_ledger"))
    )
    result: List[Dict[str, Any]] = []
    for raw in raw_gaps[:limit]:
        item = _as_dict(raw)
        if not item:
            continue
        result.append(
            {
                key: value
                for key, value in {
                    "gap_id": item.get("gap_id") or item.get("id"),
                    "requirement_id": item.get("requirement_id"),
                    "chapter_id": item.get("chapter_id"),
                    "section_id": item.get("section_id"),
                    "gap_type": item.get("gap_type") or item.get("type"),
                    "status": item.get("status"),
                    "severity": item.get("severity"),
                    "missing": _as_list(item.get("missing")) or _as_list(item.get("missing_fields")),
                    "retry_plan": _compact_retry_plan_for_audit(item.get("retry_plan")),
                    "message": _compact_text(item.get("message") or item.get("reason") or item.get("why_current_evidence_insufficient"), 260),
                }.items()
                if value not in (None, "", [], {})
            }
        )
    return result


def _compact_final_citation_audit(value: Any) -> Dict[str, Any]:
    payload = _as_dict(value)
    keys = [
        "final_citation_reconciliation_status",
        "final_body_citation_refs",
        "final_appendix_refs",
        "final_missing_appendix_refs",
        "final_unresolved_citation_refs",
        "final_unresolved_citation_removed_count",
        "final_duplicate_citation_removed_count",
        "factual_body_without_citations_count",
        "citationless_fact_examples",
    ]
    return {key: payload.get(key) for key in keys if payload.get(key) not in (None, "", [], {})}


def _is_fatal_audit(payload: Dict[str, Any]) -> bool:
    if str(payload.get("status") or "").strip().lower() == "fatal":
        return True
    if str(payload.get("publish_recommendation") or "").strip().lower() == "hold":
        return True
    for item in _as_list(payload.get("critical_findings")):
        if str(_as_dict(item).get("severity") or "").strip().lower() == "fatal":
            return True
    return False


DELIVERY_BLOCKER_TYPES = {
    "internal_evidence_cards",
    "internal_round_trace",
    "internal_draft_instruction",
    "policy_profile_leak",
    "internal_evidence_id",
    "internal_chapter_id",
    "empty_markdown_table",
    "missing_sources_appendix",
    "fake_or_placeholder_evidence",
    "fake_or_placeholder_source",
    "title_only_source",
}


def _delivery_blocker_findings(findings: Any) -> List[Dict[str, Any]]:
    blockers: List[Dict[str, Any]] = []
    for raw in _as_list(findings):
        item = _as_dict(raw)
        if not item:
            continue
        if str(item.get("type") or "").strip() in DELIVERY_BLOCKER_TYPES:
            blockers.append(item)
    return blockers


def _deterministic_delivery_blockers(deterministic_audit: Dict[str, Any]) -> List[Dict[str, Any]]:
    return _delivery_blocker_findings(_as_list(_as_dict(deterministic_audit).get("findings")))


def _draft_complete_status(
    *,
    audit_payload: Dict[str, Any],
    deterministic_audit: Dict[str, Any],
    delivery_blockers: Sequence[Dict[str, Any]],
    isolated_gate: bool,
) -> str:
    if delivery_blockers:
        return "fatal"
    if isolated_gate:
        if bool(deterministic_audit.get("fatal")):
            return "fatal"
        return str(audit_payload.get("status") or "warning").strip().lower()
    if _is_fatal_audit(audit_payload) or bool(deterministic_audit.get("fatal")):
        return "warning"
    return str(audit_payload.get("status") or "warning").strip().lower()


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


def _normalize_source_ref(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    # Allow up to 5 digits so deep reports with >999 cited sources still
    # normalize correctly. Bracketed and bare-number forms are both accepted.
    match = re.fullmatch(r"\[?\s*(\d{1,5})\s*\]?", text)
    if match:
        return f"[{match.group(1)}]"
    return text


def _cited_refs(markdown: str) -> set[str]:
    return {f"[{match}]" for match in re.findall(r"\[(\d{1,5})\]", str(markdown or ""))}


def _source_ref_candidates(source: Dict[str, Any]) -> set[str]:
    refs: set[str] = set()
    for key in ("ref", "id", "source_id", "source_ref", "citation_ref"):
        ref = _normalize_source_ref(source.get(key))
        if ref:
            refs.add(ref)
    return refs


def _source_is_cited(source: Dict[str, Any], cited: set[str]) -> bool:
    refs = _source_ref_candidates(source)
    return bool(refs and cited and refs.intersection(cited))


def _source_supports_core_claim(source: Dict[str, Any]) -> bool:
    text = " ".join(
        str(source.get(key) or "").strip().lower()
        for key in (
            "allowed_use",
            "judgment_use",
            "claim_role",
            "evidence_role",
            "role",
            "usage_tier",
            "claim_strength",
        )
    )
    if any(token in text for token in ("core_claim", "core", "decision_ready", "strong")):
        return True
    for key in ("used_for_core_claim", "core_supporting_source", "supports_core_claim"):
        if bool(source.get(key)):
            return True
    return False


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
    citation_manifest = _as_dict(package.get("citation_manifest") or writer_report.get("citation_manifest"))
    final_citation_audit = _as_dict(package.get("final_citation_audit") or writer_report.get("final_citation_audit"))
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

    if citation_manifest:
        manifest_status = str(citation_manifest.get("citation_manifest_status") or "").strip().lower()
        if manifest_status == "blocked":
            findings.append(
                {
                    "type": "citation_manifest_blocked",
                    "severity": "fatal",
                    "message": "Citation manifest has unresolved or excluded cited sources.",
                    "missing_evidence_refs": _as_list(citation_manifest.get("missing_evidence_refs"))[:10],
                    "excluded_cited_sources": _as_list(citation_manifest.get("excluded_cited_sources"))[:8],
                    "orphan_citation_count": citation_manifest.get("orphan_citation_count") or 0,
                }
            )
    if final_citation_audit:
        final_missing = _as_list(final_citation_audit.get("final_missing_appendix_refs"))
        final_status = str(final_citation_audit.get("final_citation_reconciliation_status") or "").strip().lower()
        citationless_count = int(final_citation_audit.get("factual_body_without_citations_count") or 0)
        final_unresolved = _as_list(
            final_citation_audit.get("final_unresolved_citation_refs")
            or final_citation_audit.get("unresolved_citation_refs")
        )
        final_body_refs = [_normalize_source_ref(ref) for ref in _as_list(final_citation_audit.get("final_body_citation_refs"))]
        final_appendix_refs = [_normalize_source_ref(ref) for ref in _as_list(final_citation_audit.get("final_appendix_refs"))]
        appendix_ref_set = {ref for ref in final_appendix_refs if ref}
        body_refs_missing_from_appendix = [
            ref for ref in final_body_refs if ref and appendix_ref_set and ref not in appendix_ref_set
        ]
        if citationless_count:
            findings.append(
                {
                    "type": "citationless_factual_body",
                    "severity": "fatal",
                    "message": "Final rendered body contains factual claims without public citations.",
                    "count": citationless_count,
                    "examples": _as_list(final_citation_audit.get("citationless_fact_examples"))[:5],
                }
            )
        has_explicit_final_gap = bool(final_missing or final_unresolved or body_refs_missing_from_appendix)
        has_unknown_final_gap = (
            final_status == "blocked"
            and not citationless_count
            and not has_explicit_final_gap
            and not (final_body_refs and final_appendix_refs)
        )
        if has_explicit_final_gap or has_unknown_final_gap:
            findings.append(
                {
                    "type": "final_citation_gap",
                    "severity": "fatal",
                    "message": "Final rendered body citations and source appendix are inconsistent.",
                    "final_missing_appendix_refs": (final_missing or body_refs_missing_from_appendix)[:10],
                    "final_unresolved_citation_refs": final_unresolved[:10],
                    "final_body_citation_refs": final_body_refs[:20],
                    "final_appendix_refs": final_appendix_refs[:20],
                }
            )

    if re.search(r"\[\d{1,3}\]", text) and not SOURCE_APPENDIX_RE.search(text):
        findings.append({"type": "missing_sources_appendix", "severity": "fatal", "message": "Report cites numbered sources but has no source appendix."})

    if _is_placeholder_url(text) or FAKE_EVIDENCE_TEXT in text.lower():
        findings.append({"type": "fake_or_placeholder_evidence", "severity": "fatal", "message": "Report contains placeholder source URLs or fake example evidence text."})

    title_only_sources = []
    title_only_candidates = []
    fake_sources = []
    cited = _cited_refs(text)
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
            item = {"ref": source.get("ref") or source.get("id"), "title": _compact_text(title, 160)}
            if _source_is_cited(source, cited) or _source_supports_core_claim(source):
                title_only_sources.append(item)
            else:
                title_only_candidates.append(item)
    if title_only_sources:
        findings.append({"type": "title_only_source", "severity": "fatal", "message": "Cited or core-supporting source is title-only and not independently traceable.", "examples": title_only_sources[:8]})
    if title_only_candidates:
        findings.append({"type": "title_only_source_candidate", "severity": "medium", "message": "Source registry contains unused title-only candidates; they were not treated as clean blockers.", "examples": title_only_candidates[:8]})
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


def _current_audit_date() -> date:
    raw = str(os.getenv("REPORT_FINAL_AUDIT_CURRENT_DATE") or "").strip()
    if raw:
        try:
            return datetime.strptime(raw[:10], "%Y-%m-%d").date()
        except ValueError:
            pass
    return date.today()


def _extract_iso_dates(text: str) -> List[date]:
    dates: List[date] = []
    for match in re.findall(r"\b(20\d{2}-\d{1,2}-\d{1,2})\b", str(text or "")):
        try:
            dates.append(datetime.strptime(match, "%Y-%m-%d").date())
        except ValueError:
            continue
    return dates


def _false_future_date_finding(item: Dict[str, Any], *, current_date: date) -> bool:
    text = " ".join(str(value or "") for value in item.values())
    if not re.search(r"future[-\s]?dated|future date|in the future|未来|未来日期", text, flags=re.I):
        return False
    dates = _extract_iso_dates(text)
    if not dates:
        return False
    return all(value <= current_date for value in dates)


def _apply_final_audit_date_sanity(payload: Dict[str, Any], *, current_date: date) -> Dict[str, Any]:
    result = dict(payload or {})
    removed = 0
    for key in ("critical_findings", "citation_issues"):
        kept: List[Any] = []
        for item in _as_list(result.get(key)):
            finding = _as_dict(item)
            if finding and _false_future_date_finding(finding, current_date=current_date):
                removed += 1
                continue
            kept.append(item)
        result[key] = kept
    result["date_sanity_current_date"] = current_date.isoformat()
    result["date_sanity_removed_findings_count"] = removed
    if removed:
        fatal_findings = any(
            str(_as_dict(item).get("severity") or "").strip().lower() == "fatal"
            for item in [*_as_list(result.get("critical_findings")), *_as_list(result.get("citation_issues"))]
        )
        if not fatal_findings:
            has_findings = any(
                _as_list(result.get(key))
                for key in (
                    "critical_findings",
                    "unsupported_claims",
                    "citation_issues",
                    "scope_or_method_issues",
                    "risk_section_feedback",
                )
            )
            result["status"] = "warning" if has_findings else "pass"
            result["publish_recommendation"] = "publish_with_caveats" if has_findings else "publish"
            if not has_findings:
                result["summary"] = "Audit date sanity removed false future-date findings."
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
    writer_report = _as_dict(package.get("writer_report"))
    deterministic_audit = run_deterministic_audit(
        report_markdown=report_markdown,
        validation=validation,
        clean_evidence=clean_evidence,
        writer_package_payload=package,
        query=query,
    )
    isolated_gate = quality_gates_isolated()
    audit_mode = _final_audit_mode()
    delivery_blockers = _deterministic_delivery_blockers(deterministic_audit)
    blocking = _env_flag("REPORT_FINAL_AUDIT_BLOCKING", True) and not isolated_gate and audit_mode != "audit_only"
    deterministic_blocks = bool(deterministic_audit.get("fatal")) if audit_mode == "publish_strict" else bool(delivery_blockers)
    if deterministic_blocks and blocking:
        return {
            "enabled": True,
            "success": True,
            "status": "fatal",
            "blocked": True,
            "blocking": blocking,
            "final_audit_mode": audit_mode,
            "delivery_blockers": delivery_blockers or _as_list(deterministic_audit.get("findings")),
            "quality_fatal_observed": bool(deterministic_audit.get("fatal")) and not bool(delivery_blockers),
            "quality_gate_mode": quality_gate_mode(),
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
    min_output_tokens = _env_int(
        "REPORT_FINAL_AUDIT_MAX_OUTPUT_TOKENS",
        8192,
        min_value=2048,
        max_value=64000,
    )
    try:
        configured_output_tokens = int(config.get("max_output_tokens") or 0)
    except (TypeError, ValueError):
        configured_output_tokens = 0
    if configured_output_tokens < min_output_tokens:
        config["max_output_tokens"] = min_output_tokens
    normalized = normalize_llm_config(config)
    if not llm_config_is_ready(normalized):
        deterministic_fatal = bool(deterministic_audit.get("fatal"))
        audit_payload = {
            "status": "fatal" if deterministic_fatal else "skipped",
            "overall_score": 0,
            "critical_findings": _as_list(deterministic_audit.get("findings")),
            "publish_recommendation": "hold",
            "summary": deterministic_audit.get("summary"),
        } if deterministic_fatal else {}
        top_status = (
            "fatal"
            if audit_mode == "publish_strict" and deterministic_fatal
            else _draft_complete_status(
                audit_payload=audit_payload or {"status": "skipped"},
                deterministic_audit=deterministic_audit,
                delivery_blockers=delivery_blockers,
                isolated_gate=isolated_gate,
            )
        )
        return {
            "enabled": True,
            "success": False,
            "status": top_status,
            "blocked": bool(blocking and delivery_blockers),
            "blocking": blocking,
            "final_audit_mode": audit_mode,
            "delivery_blockers": delivery_blockers,
            "quality_fatal_observed": bool(deterministic_fatal and not delivery_blockers),
            "quality_gate_mode": quality_gate_mode(),
            "skipped_reason": "config_missing",
            "model": normalized.get("model") or "",
            "deterministic_audit": deterministic_audit,
            "audit": audit_payload,
        }

    max_chars = _env_int("REPORT_FINAL_AUDIT_MAX_CHARS", 200000, min_value=4000, max_value=1_000_000)
    report_payload = _truncate_report(report_markdown, max_chars)
    current_audit_date = _current_audit_date()
    user_payload = {
        "query": query,
        "current_date": current_audit_date.isoformat(),
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
        "score_gaps": _compact_score_gaps_for_audit(package, writer_report),
        "requirement_gap_summary": _as_dict(package.get("requirement_gap_summary") or writer_report.get("requirement_gap_summary")),
        "final_citation_audit": _compact_final_citation_audit(package.get("final_citation_audit") or writer_report.get("final_citation_audit")),
    }
    try:
        response = call_openai_compatible_json(
            config=config,
            system_prompt=FINAL_AUDIT_SYSTEM_PROMPT,
            user_payload=user_payload,
        )
        audit_payload = _apply_final_audit_date_sanity(
            _normalize_audit_payload(_as_dict(response.get("payload"))),
            current_date=current_audit_date,
        )
        fatal = _is_fatal_audit(audit_payload) or bool(deterministic_audit.get("fatal"))
        if audit_mode == "publish_strict":
            top_status = "fatal" if deterministic_audit.get("fatal") else audit_payload["status"]
            blocked = bool(blocking and fatal)
        else:
            top_status = _draft_complete_status(
                audit_payload=audit_payload,
                deterministic_audit=deterministic_audit,
                delivery_blockers=delivery_blockers,
                isolated_gate=isolated_gate,
            )
            blocked = bool(blocking and delivery_blockers)
        return {
            "enabled": True,
            "success": True,
            "status": top_status,
            "blocked": blocked,
            "blocking": blocking,
            "final_audit_mode": audit_mode,
            "delivery_blockers": delivery_blockers,
            "quality_fatal_observed": bool(fatal and not delivery_blockers),
            "quality_gate_mode": quality_gate_mode(),
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
            "blocked": bool(
                blocking
                and (
                    bool(deterministic_audit.get("fatal"))
                    if audit_mode == "publish_strict"
                    else bool(delivery_blockers)
                )
            ),
            "blocking": blocking,
            "final_audit_mode": audit_mode,
            "delivery_blockers": delivery_blockers,
            "quality_fatal_observed": bool(deterministic_audit.get("fatal") and not delivery_blockers),
            "quality_gate_mode": quality_gate_mode(),
            "model": normalized.get("model") or "",
            "reasoning_effort": normalized.get("reasoning_effort") or "",
            "error": str(exc),
            "deterministic_audit": deterministic_audit,
            "llm_call": diagnostic,
            "report_truncation": user_payload["report_truncation"],
        }
