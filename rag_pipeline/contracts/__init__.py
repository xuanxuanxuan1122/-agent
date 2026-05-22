"""Shared report-production contracts.

These helpers keep the report pipeline's handoffs explicit without forcing a
new runtime dependency or changing agent execution order.
"""

from .evidence_ledger import attach_evidence_ledger, build_evidence_ledger
from .evidence_quality import (
    EvidenceClassifier,
    EvidenceNormalizer,
    apply_evidence_quality_contract,
    classify_evidence,
    normalize_evidence,
)
from .quality_gate import build_quality_gate_state
from .query_builder import QueryBuilder, build_query_package
from .report_contract import build_report_contract, build_report_contract_from_package
from .source_registry import pick_refs, renumber_sources_by_first_citation

__all__ = [
    "EvidenceClassifier",
    "EvidenceNormalizer",
    "attach_evidence_ledger",
    "apply_evidence_quality_contract",
    "build_evidence_ledger",
    "build_quality_gate_state",
    "build_query_package",
    "build_report_contract",
    "build_report_contract_from_package",
    "classify_evidence",
    "normalize_evidence",
    "pick_refs",
    "QueryBuilder",
    "renumber_sources_by_first_citation",
]
