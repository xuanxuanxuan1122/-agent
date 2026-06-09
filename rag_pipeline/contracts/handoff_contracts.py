from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence, Set

from rag_pipeline.agents.report_contracts import text_has_factual_claim
from rag_pipeline.contracts.evidence_identity import build_evidence_alias_map
from rag_pipeline.contracts.ref_normalizer import normalize_claim_refs


HANDOFF_CONTRACT_VERSION = "handoff_contracts_v1"

CITATION_RE = re.compile(r"\[(\d{1,5})\]")


@dataclass(frozen=True)
class HandoffValidationResult:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "summary": dict(self.summary),
            "contract_version": HANDOFF_CONTRACT_VERSION,
        }


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return []


def _text(value: Any) -> str:
    return str(value or "").strip()


def _unique(values: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(item for item in values if item))


def _result(errors: Iterable[str], warnings: Iterable[str], summary: Dict[str, Any]) -> HandoffValidationResult:
    error_list = _unique(errors)
    warning_list = _unique(warnings)
    return HandoffValidationResult(
        ok=not error_list,
        errors=error_list,
        warnings=warning_list,
        summary={"contract_version": HANDOFF_CONTRACT_VERSION, **summary},
    )


def _evidence_id(item: Dict[str, Any]) -> str:
    return _text(item.get("evidence_id") or item.get("fact_id") or item.get("id") or item.get("ref"))


def _claim_id(item: Dict[str, Any]) -> str:
    return _text(item.get("claim_id") or item.get("id") or item.get("ref"))


def _source_id(item: Dict[str, Any]) -> str:
    return _text(
        item.get("source_id")
        or item.get("run_source_id")
        or item.get("canonical_source_id")
        or item.get("ref")
        or item.get("id")
    )


def _source_registry(package_or_sources: Any) -> List[Dict[str, Any]]:
    if isinstance(package_or_sources, dict):
        candidates = (
            _as_list(package_or_sources.get("source_registry"))
            or _as_list(package_or_sources.get("sources"))
            or _as_list(_as_dict(package_or_sources.get("render_artifacts")).get("source_registry"))
        )
    else:
        candidates = _as_list(package_or_sources)
    return [item for item in candidates if isinstance(item, dict)]


def _source_keys(source: Dict[str, Any]) -> Set[str]:
    keys = {
        _source_id(source),
        _text(source.get("ref")),
        _text(source.get("source_ref")),
        _text(source.get("url") or source.get("source_url")),
        _text(source.get("title") or source.get("source")),
    }
    for ref in _as_list(source.get("evidence_refs")):
        keys.add(_text(ref))
    return {item for item in keys if item}


def _source_key_set(sources: Sequence[Dict[str, Any]]) -> Set[str]:
    keys: Set[str] = set()
    for source in sources:
        keys.update(_source_keys(source))
    return keys


def _analysis_evidence_items(evidence_package: Dict[str, Any]) -> List[Dict[str, Any]]:
    package = _as_dict(evidence_package)
    candidates: List[Any] = []
    for key in (
        "analysis_ready_evidence",
        "fact_cards",
        "fact_card_pool",
        "evidence_cards",
        "evidence_analyses",
        "items",
    ):
        candidates.extend(_as_list(package.get(key)))
    if not candidates:
        candidates.extend(_as_list(_as_dict(package.get("evidence_ledger")).get("items")))
    return [item for item in candidates if isinstance(item, dict)]


def _evidence_refs(item: Dict[str, Any]) -> List[str]:
    lineage = _as_dict(item.get("lineage"))
    payload = _as_dict(item.get("payload"))
    refs = (
        _as_list(item.get("fact_ids"))
        or _as_list(item.get("evidence_refs"))
        or _as_list(item.get("supporting_evidence_refs"))
        or _as_list(item.get("supporting_evidence"))
        or _as_list(item.get("used_evidence_ids"))
        or _as_list(lineage.get("fact_ids"))
        or _as_list(lineage.get("evidence_refs"))
        or _as_list(payload.get("fact_ids"))
        or _as_list(payload.get("evidence_refs"))
    )
    return _unique(_text(ref) for ref in refs)


def _requirement_ids(item: Dict[str, Any]) -> List[str]:
    lineage = _as_dict(item.get("lineage"))
    payload = _as_dict(item.get("payload"))
    return _unique(
        _text(ref)
        for ref in (
            _as_list(item.get("requirement_ids"))
            or _as_list(lineage.get("requirement_ids"))
            or _as_list(payload.get("requirement_ids"))
        )
    )


def _collect_claim_units(structured_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    analysis = _as_dict(structured_analysis)
    claims: List[Dict[str, Any]] = [item for item in _as_list(analysis.get("claim_units")) if isinstance(item, dict)]
    for chapter in _as_list(_as_dict(analysis.get("llm_analysis_synthesis")).get("chapter_synthesis")):
        if isinstance(chapter, dict):
            claims.extend(item for item in _as_list(chapter.get("claim_units")) if isinstance(item, dict))
    return claims


def _collect_repair_priorities(
    evidence_package: Dict[str, Any],
    structured_analysis: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    package = _as_dict(evidence_package)
    analysis = _as_dict(structured_analysis)
    candidates: List[Any] = []
    for source in (
        package,
        analysis,
        _as_dict(analysis.get("llm_analysis_synthesis")),
        _as_dict(_as_dict(analysis.get("llm_analysis_synthesis")).get("validation")),
    ):
        candidates.extend(_as_list(source.get("evidence_repair_priorities")))
        candidates.extend(_as_list(source.get("claim_repair_priorities")))
        candidates.extend(_as_list(source.get("evidence_gap_ledger")))
        candidates.extend(_as_list(source.get("score_gaps")))
    return [item for item in candidates if isinstance(item, dict)]


def _item_source_resolves(item: Dict[str, Any], source_keys: Set[str]) -> bool:
    nested = _as_dict(item.get("source"))
    direct_keys = {
        _text(item.get("source_id")),
        _text(item.get("source_ref")),
        _text(item.get("source_url")),
        _text(item.get("url")),
        _text(nested.get("source_id")),
        _text(nested.get("ref")),
        _text(nested.get("url") or nested.get("source_url")),
        _text(nested.get("title") or nested.get("source")),
    }
    direct_keys = {key for key in direct_keys if key}
    if direct_keys & source_keys:
        return True
    return bool(nested.get("url") or item.get("url") or item.get("source_url"))


def validate_evidence_package_for_analysis(evidence_package: Dict[str, Any]) -> HandoffValidationResult:
    package = _as_dict(evidence_package)
    evidence_items = _analysis_evidence_items(package)
    source_keys = _source_key_set(_source_registry(package))
    errors: List[str] = []
    warnings: List[str] = []
    missing_id_count = 0
    unresolved_source_count = 0
    public_candidate_count = 0
    missing_allowed_use_count = 0

    for item in evidence_items:
        if not _evidence_id(item):
            missing_id_count += 1
            errors.append("evidence_missing_evidence_id")
        allowed_use = _text(item.get("allowed_use") or item.get("usage") or item.get("analysis_role")).lower()
        if not allowed_use:
            missing_allowed_use_count += 1
            warnings.append("evidence_allowed_use_missing")
            continue
        if allowed_use in {"public", "admissible", "publishable", "directional", "core", "supporting"}:
            public_candidate_count += 1
            if not _item_source_resolves(item, source_keys):
                unresolved_source_count += 1
                errors.append("public_evidence_source_unresolved")

    if not evidence_items:
        warnings.append("analysis_evidence_empty")
    return _result(
        errors,
        warnings,
        {
            "stage": "evidence_package_for_analysis",
            "analysis_candidate_count": len(evidence_items),
            "public_candidate_count": public_candidate_count,
            "missing_allowed_use_count": missing_allowed_use_count,
            "missing_evidence_id_count": missing_id_count,
            "unresolved_public_source_count": unresolved_source_count,
            "source_registry_count": len(_source_registry(package)),
        },
    )


def validate_structured_analysis_for_writer(
    structured_analysis: Dict[str, Any],
    *,
    evidence_package: Dict[str, Any] | None = None,
) -> HandoffValidationResult:
    claims = _collect_claim_units(_as_dict(structured_analysis))
    fact_cards = _analysis_evidence_items(_as_dict(evidence_package))
    evidence_ids = {_evidence_id(item) for item in fact_cards if _evidence_id(item)}
    alias_map = build_evidence_alias_map(fact_cards)
    errors: List[str] = []
    warnings: List[str] = []
    missing_claim_id_count = 0
    missing_refs_count = 0
    unresolved_ref_count = 0
    ambiguous_ref_count = 0
    alias_resolved_ref_count = 0
    total_ref_count = 0
    resolved_ref_count = 0
    claims_with_resolved_refs_count = 0
    missing_requirement_count = 0
    missing_source_count = 0

    for claim in claims:
        if not _claim_id(claim):
            missing_claim_id_count += 1
            errors.append("claim_missing_claim_id")
        normalized = normalize_claim_refs(claim, alias_map=alias_map, fact_cards=fact_cards)
        refs = _as_list(normalized.get("fact_ids"))
        resolution = _as_dict(normalized.get("ref_resolution"))
        total_ref_count += int(resolution.get("total_refs") or len(refs) or 0)
        resolved_ref_count += int(resolution.get("resolved_ref_count") or 0)
        alias_resolved_ref_count += int(resolution.get("alias_resolved_ref_count") or 0)
        ambiguous_ref_count += int(resolution.get("ambiguous_ref_count") or 0)
        unresolved_ref_count += int(resolution.get("unresolved_ref_count") or 0)
        if ambiguous_ref_count:
            errors.append("claim_references_ambiguous_evidence_alias")
        if not refs:
            missing_refs_count += 1
            errors.append("claim_missing_fact_or_evidence_refs")
        elif not evidence_ids:
            claims_with_resolved_refs_count += 1
        else:
            resolved = [ref for ref in refs if ref in evidence_ids]
            if resolved:
                claims_with_resolved_refs_count += 1
            direct_unresolved_count = len(refs) - len(resolved)
            if direct_unresolved_count:
                unresolved_ref_count += direct_unresolved_count
                errors.append("claim_references_evidence_missing_from_package")
        if _as_list(normalized.get("unresolved_refs")):
            errors.append("claim_references_evidence_missing_from_package")
        if not _as_list(normalized.get("source_ids")):
            missing_source_count += 1
            warnings.append("claim_missing_source_ids")
        if not _requirement_ids(normalized):
            missing_requirement_count += 1
            warnings.append("claim_missing_requirement_ids")

    if not claims:
        warnings.append("structured_analysis_claims_empty")
    return _result(
        errors,
        warnings,
        {
            "stage": "structured_analysis_for_writer",
            "claim_count": len(claims),
            "claims_with_resolved_refs_count": claims_with_resolved_refs_count,
            "missing_claim_id_count": missing_claim_id_count,
            "missing_fact_or_evidence_refs_count": missing_refs_count,
            "unresolved_evidence_ref_count": unresolved_ref_count,
            "missing_source_ids_count": missing_source_count,
            "missing_requirement_ids_count": missing_requirement_count,
            "evidence_ref_resolution": {
                "total_refs": total_ref_count,
                "resolved_ref_count": resolved_ref_count,
                "unresolved_ref_count": unresolved_ref_count,
                "ambiguous_ref_count": ambiguous_ref_count,
                "alias_resolved_ref_count": alias_resolved_ref_count,
            },
        },
    )


def validate_repair_priorities_for_dispatch(
    evidence_package: Dict[str, Any],
    structured_analysis: Dict[str, Any] | None = None,
) -> HandoffValidationResult:
    priorities = _collect_repair_priorities(_as_dict(evidence_package), structured_analysis)
    errors: List[str] = []
    warnings: List[str] = []
    missing_gap_id_count = 0
    missing_gap_type_count = 0
    missing_route_count = 0
    unsafe_writing_count = 0
    dispatch_ready_count = 0

    for item in priorities:
        gap_id = _text(item.get("gap_id") or item.get("id") or item.get("claim_id"))
        gap_type = _text(item.get("gap_type") or item.get("type"))
        route = _text(item.get("repair_route") or item.get("route"))
        allowed = bool(item.get("allowed_for_writing", False))
        if not gap_id:
            missing_gap_id_count += 1
            errors.append("repair_priority_missing_gap_id")
        if not gap_type:
            missing_gap_type_count += 1
            errors.append("repair_priority_missing_gap_type")
        if not route:
            missing_route_count += 1
            errors.append("repair_priority_missing_repair_route")
        if allowed:
            unsafe_writing_count += 1
            errors.append("repair_priority_allowed_for_writing_must_be_false")
        if gap_id and gap_type and route and not allowed:
            dispatch_ready_count += 1
        if not _text(item.get("query") or item.get("suggested_query")):
            warnings.append("repair_priority_missing_query_seed")

    return _result(
        errors,
        warnings,
        {
            "stage": "repair_priorities_for_dispatch",
            "repair_priority_count": len(priorities),
            "dispatch_ready_count": dispatch_ready_count,
            "missing_gap_id_count": missing_gap_id_count,
            "missing_gap_type_count": missing_gap_type_count,
            "missing_repair_route_count": missing_route_count,
            "allowed_for_writing_violation_count": unsafe_writing_count,
        },
    )


def _markdown_lines(markdown: str) -> List[str]:
    return [line.strip() for line in str(markdown or "").splitlines() if line.strip()]


def _line_has_citation(line: str) -> bool:
    return bool(CITATION_RE.search(line))


def validate_writer_report_for_final(writer_report: Dict[str, Any]) -> HandoffValidationResult:
    report = _as_dict(writer_report)
    markdown = _text(report.get("report_markdown") or report.get("markdown") or report.get("answer_text"))
    source_registry = _source_registry(report)
    errors: List[str] = []
    warnings: List[str] = []
    citationless_factual_lines: List[str] = []

    for line in _markdown_lines(markdown):
        if line.startswith(("#", "|", "```")):
            continue
        if text_has_factual_claim(line) and not _line_has_citation(line):
            citationless_factual_lines.append(line[:180])
            errors.append("writer_factual_line_without_citation")

    if not markdown:
        errors.append("writer_report_missing_markdown")
    if not source_registry:
        warnings.append("writer_report_source_registry_empty")
    return _result(
        errors,
        warnings,
        {
            "stage": "writer_report_for_final",
            "markdown_char_count": len(markdown),
            "source_registry_count": len(source_registry),
            "citationless_factual_line_count": len(citationless_factual_lines),
            "citationless_factual_line_samples": citationless_factual_lines[:5],
        },
    )


def _manifest_items(citation_manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    manifest = _as_dict(citation_manifest)
    for key in ("items", "sources", "citations", "entries"):
        items = [item for item in _as_list(manifest.get(key)) if isinstance(item, dict)]
        if items:
            return items
    return []


def _citation_ref(item: Dict[str, Any]) -> str:
    raw = _text(item.get("ref") or item.get("citation") or item.get("citation_ref") or item.get("id"))
    if raw.isdigit():
        return f"[{raw}]"
    return raw


def validate_citation_reconciliation(
    *,
    markdown: str,
    citation_manifest: Dict[str, Any],
    source_registry: Sequence[Dict[str, Any]],
) -> HandoffValidationResult:
    manifest_items = _manifest_items(_as_dict(citation_manifest))
    manifest_by_ref = {_citation_ref(item): item for item in manifest_items if _citation_ref(item)}
    source_keys = _source_key_set([item for item in _as_list(source_registry) if isinstance(item, dict)])
    markdown_refs = _unique(f"[{match.group(1)}]" for match in CITATION_RE.finditer(str(markdown or "")))
    errors: List[str] = []
    warnings: List[str] = []
    missing_manifest_count = 0
    missing_source_count = 0
    resolved_ref_count = 0

    for ref in markdown_refs:
        item = manifest_by_ref.get(ref)
        if not item:
            missing_manifest_count += 1
            errors.append("citation_ref_missing_from_manifest")
            continue
        if _source_keys(item) & source_keys or not source_keys:
            resolved_ref_count += 1
        else:
            missing_source_count += 1
            errors.append("citation_manifest_source_unresolved")

    if markdown_refs and not manifest_items:
        warnings.append("citation_manifest_empty")
    return _result(
        errors,
        warnings,
        {
            "stage": "citation_reconciliation",
            "markdown_ref_count": len(markdown_refs),
            "manifest_ref_count": len(manifest_by_ref),
            "resolved_ref_count": resolved_ref_count,
            "missing_manifest_ref_count": missing_manifest_count,
            "unresolved_manifest_source_count": missing_source_count,
        },
    )


def build_handoff_contract_summary(
    *,
    evidence_package: Dict[str, Any],
    structured_analysis: Dict[str, Any],
    writer_report: Dict[str, Any] | None = None,
    markdown: str | None = None,
    citation_manifest: Dict[str, Any] | None = None,
    source_registry: Sequence[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    report = _as_dict(writer_report)
    sources = list(source_registry or _source_registry(report) or _source_registry(evidence_package))
    resolved_markdown = markdown if markdown is not None else _text(report.get("report_markdown"))
    resolved_manifest = _as_dict(citation_manifest) or _as_dict(report.get("citation_manifest"))
    results = {
        "evidence_to_analysis": validate_evidence_package_for_analysis(evidence_package).to_dict(),
        "analysis_to_writer": validate_structured_analysis_for_writer(
            structured_analysis,
            evidence_package=evidence_package,
        ).to_dict(),
        "repair_to_dispatch": validate_repair_priorities_for_dispatch(
            evidence_package,
            structured_analysis,
        ).to_dict(),
    }
    if report or resolved_markdown:
        writer_payload = {**report, "report_markdown": resolved_markdown, "source_registry": sources}
        results["writer_to_final"] = validate_writer_report_for_final(writer_payload).to_dict()
    if resolved_markdown or resolved_manifest:
        results["citation_reconciliation"] = validate_citation_reconciliation(
            markdown=resolved_markdown,
            citation_manifest=resolved_manifest,
            source_registry=sources,
        ).to_dict()
    failed = [name for name, payload in results.items() if not bool(_as_dict(payload).get("ok"))]
    warnings = [name for name, payload in results.items() if _as_list(_as_dict(payload).get("warnings"))]
    return {
        "schema_version": "handoff_contract_summary_v1",
        "contract_version": HANDOFF_CONTRACT_VERSION,
        "ok": not failed,
        "failed_contracts": failed,
        "warning_contracts": warnings,
        "results": results,
    }
