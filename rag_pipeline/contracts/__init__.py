"""Shared report-production contracts.

These helpers keep the report pipeline's handoffs explicit without forcing a
new runtime dependency or changing agent execution order.
"""

from .evidence_ledger import attach_evidence_ledger, build_evidence_ledger
from .evidence_admission import decide_evidence_admission, summarize_evidence_admission
from .evidence_identity import build_evidence_alias_map, canonicalize_evidence_id, resolve_evidence_refs
from .handoff_contracts import (
    build_handoff_contract_summary,
    validate_citation_reconciliation,
    validate_evidence_package_for_analysis,
    validate_repair_priorities_for_dispatch,
    validate_structured_analysis_for_writer,
    validate_writer_report_for_final,
)
from .evidence_quality import (
    EvidenceClassifier,
    EvidenceNormalizer,
    apply_evidence_quality_contract,
    classify_evidence,
    normalize_evidence,
)
from .claim_roles import classify_claim_unit_roles
from .quality_gate import build_quality_gate_state
from .query_builder import QueryBuilder, build_query_package
from .ref_normalizer import normalize_claim_refs
from .research_reflection import build_research_reflection_memo
from .report_contract import build_report_contract, build_report_contract_from_package
from .section_audit import audit_section_claim_roles
from .source_registry import pick_refs, renumber_sources_by_first_citation

__all__ = [
    "EvidenceClassifier",
    "EvidenceNormalizer",
    "attach_evidence_ledger",
    "apply_evidence_quality_contract",
    "audit_section_claim_roles",
    "build_evidence_alias_map",
    "build_evidence_ledger",
    "build_handoff_contract_summary",
    "build_quality_gate_state",
    "build_query_package",
    "build_research_reflection_memo",
    "build_report_contract",
    "build_report_contract_from_package",
    "classify_evidence",
    "classify_claim_unit_roles",
    "canonicalize_evidence_id",
    "decide_evidence_admission",
    "normalize_claim_refs",
    "normalize_evidence",
    "pick_refs",
    "QueryBuilder",
    "renumber_sources_by_first_citation",
    "resolve_evidence_refs",
    "summarize_evidence_admission",
    "validate_citation_reconciliation",
    "validate_evidence_package_for_analysis",
    "validate_repair_priorities_for_dispatch",
    "validate_structured_analysis_for_writer",
    "validate_writer_report_for_final",
]
