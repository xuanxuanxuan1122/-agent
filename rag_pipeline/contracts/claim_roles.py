from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping


CLAIM_ROLE_CONTRACT_VERSION = "claim_roles_v1"

ROLE_ORDER = (
    "metric_claim",
    "counter_claim",
    "case_claim",
    "mechanism_claim",
    "technology_claim",
    "boundary_claim",
    "core_claim",
    "context_claim",
)


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value in (None, "", {}, ()):
        return []
    return [value]


def _text(value: Any) -> str:
    return str(value or "").strip()


def _lower(value: Any) -> str:
    return _text(value).lower()


def _dedupe(values: Iterable[Any]) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        text = _text(value)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _fact_map(fact_cards: Any) -> Dict[str, Dict[str, Any]]:
    if isinstance(fact_cards, Mapping):
        return {
            _text(key): _as_dict(value)
            for key, value in fact_cards.items()
            if _text(key) and isinstance(value, dict)
        }
    result: Dict[str, Dict[str, Any]] = {}
    for item in _as_list(fact_cards):
        card = _as_dict(item)
        fact_id = _text(card.get("fact_id") or card.get("evidence_id") or card.get("ref") or card.get("id"))
        if fact_id:
            result[fact_id] = card
    return result


def _refs_from_claim(claim_unit: Dict[str, Any]) -> List[str]:
    return _dedupe(
        [
            *_as_list(claim_unit.get("fact_ids")),
            *_as_list(claim_unit.get("used_fact_refs")),
            *_as_list(claim_unit.get("evidence_refs")),
            *_as_list(claim_unit.get("supporting_fact_refs")),
            *_as_list(claim_unit.get("supporting_evidence_refs")),
            *_as_list(claim_unit.get("used_evidence_ids")),
        ]
    )


def _has_any(text: str, markers: Iterable[str]) -> bool:
    return any(marker in text for marker in markers)


def _append_role(roles: List[str], reasons: List[str], role: str, reason: str) -> None:
    if role not in roles:
        roles.append(role)
    if reason and reason not in reasons:
        reasons.append(reason)


def _is_positive_support_role(role_text: str) -> bool:
    return any(
        marker in role_text
        for marker in (
            "support",
            "core",
            "metric",
            "case",
            "mechanism",
            "technology",
            "official",
            "claimable",
        )
    )


def classify_claim_unit_roles(claim_unit: Dict[str, Any], fact_cards: Any = None) -> Dict[str, Any]:
    """Classify a claim unit into writer-safe semantic roles.

    The result is intentionally compact and deterministic so later pipeline
    stages can audit whether a section has core support, metric support,
    counter/boundary coverage, and case/mechanism context without rereading
    large evidence payloads.
    """

    unit = _as_dict(claim_unit)
    facts = _fact_map(fact_cards)
    refs = _refs_from_claim(unit)
    cited_facts = [facts[ref] for ref in refs if ref in facts]
    roles: List[str] = []
    reasons: List[str] = []

    explicit_roles = [
        role
        for role in _as_list(unit.get("claim_roles"))
        if _text(role) in ROLE_ORDER
    ]
    for role in explicit_roles:
        _append_role(roles, reasons, _text(role), "explicit_claim_role")

    payload_text = " ".join(
        [
            _lower(unit.get("claim")),
            _lower(unit.get("analysis_role")),
            _lower(unit.get("proof_role")),
            _lower(unit.get("allowed_use")),
            _lower(unit.get("block_affinity")),
            " ".join(_lower(item) for item in _as_list(unit.get("block_affinity"))),
            _lower(unit.get("reasoning")),
            _lower(unit.get("mechanism")),
            " ".join(_lower(item) for item in _as_list(unit.get("reasoning_chain"))),
            _lower(unit.get("limitation_boundary")),
            _lower(unit.get("counter_boundary")),
            _lower(unit.get("counter_evidence")),
        ]
    )

    if _has_any(
        payload_text,
        (
            "metric",
            "indicator",
            "survey",
            "penetration",
            "market size",
            "revenue",
            "rate",
            "percentage",
            "percent",
            "value",
            "unit",
            "period",
            "metric_reconciliation",
        ),
    ):
        _append_role(roles, reasons, "metric_claim", "claim_or_layout_requests_metric")
    if _has_any(payload_text, ("case", "customer", "deployment", "rollout", "implementation", "use case", "company")):
        _append_role(roles, reasons, "case_claim", "claim_or_layout_requests_case")
    if _has_any(payload_text, ("technology", "architecture", "stack", "model", "workflow", "orchestration", "integration")):
        _append_role(roles, reasons, "technology_claim", "claim_or_reasoning_describes_technology")
    if _has_any(payload_text, ("mechanism", "because", "explains", "drives", "reduces", "indicates", "reasoning", "handoff")):
        _append_role(roles, reasons, "mechanism_claim", "claim_has_mechanism_reasoning")
    if _has_any(payload_text, ("limited", "limitation", "boundary", "counter", "risk", "security", "compliance", "failure", "unclear", "cost")):
        _append_role(roles, reasons, "boundary_claim", "claim_has_boundary_or_risk_language")

    counter_only = False
    for fact in cited_facts:
        fact_role = " ".join(
            [
                _lower(fact.get("proof_role")),
                _lower(fact.get("analysis_role")),
                _lower(fact.get("allowed_use")),
                _lower(fact.get("fact_type")),
                _lower(_as_dict(fact.get("evidence_card")).get("proof_role")),
            ]
        )
        if fact.get("metric") or fact.get("value") or "metric" in fact_role:
            _append_role(roles, reasons, "metric_claim", "cited_fact_is_metric")
        if "counter" in fact_role or "risk" in fact_role:
            _append_role(roles, reasons, "counter_claim", "cited_fact_is_counter")
            _append_role(roles, reasons, "boundary_claim", "counter_fact_sets_boundary")
        if "case" in fact_role or "customer" in fact_role:
            _append_role(roles, reasons, "case_claim", "cited_fact_is_case")
        if "technology" in fact_role or "technical" in fact_role:
            _append_role(roles, reasons, "technology_claim", "cited_fact_is_technology")
        if fact_role and "counter" in fact_role and not _is_positive_support_role(fact_role):
            counter_only = True

    analysis_role = _lower(unit.get("analysis_role"))
    proof_role = _lower(unit.get("proof_role"))
    if "counter" in analysis_role or "counter" in proof_role:
        _append_role(roles, reasons, "counter_claim", "claim_role_is_counter")
        _append_role(roles, reasons, "boundary_claim", "counter_claim_sets_boundary")
        counter_only = True

    strength = _lower(unit.get("claim_strength") or unit.get("claim_status"))
    if strength in {"strong", "decision_ready", "moderate", "medium"} and "counter_claim" not in roles and not counter_only:
        _append_role(roles, reasons, "core_claim", "claim_strength_supports_core_use")

    if not roles:
        _append_role(roles, reasons, "context_claim", "no_claimable_role_detected")

    ordered_roles = [role for role in ROLE_ORDER if role in roles]
    primary = ordered_roles[0] if ordered_roles else "context_claim"
    return {
        "claim_role_contract_version": CLAIM_ROLE_CONTRACT_VERSION,
        "primary_claim_role": primary,
        "claim_roles": ordered_roles,
        "role_reasons": reasons[:12],
    }
