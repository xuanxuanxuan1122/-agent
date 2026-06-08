from __future__ import annotations

from typing import Any, Dict, List

from rag_pipeline.contracts.source_strategy import source_strategy_for_role


REQUIRED_FIELDS_BY_ROLE = {
    "metric": ["metric", "value", "unit", "period", "scope", "source_ref"],
    "source_check": ["source_ref", "source_title", "source_url"],
    "counter": ["counter_signal", "source_ref"],
    "case": ["company", "use_case", "deployment_scope", "source_ref"],
    "customer_case": ["company", "use_case", "deployment_scope", "source_ref"],
    "technology": ["capability", "constraint", "source_ref"],
    "technology_product": ["capability", "constraint", "source_ref"],
    "support": ["fact", "source_ref"],
}


GENERIC_FIELD_NAMES = {
    "market",
    "market_size",
    "trend",
    "competition",
    "policy",
    "technology",
    "capital",
    "risk",
}


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return []


def _role(value: Any) -> str:
    return str(value or "support").strip().lower() or "support"


def _source_levels(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value.strip().upper()] if value.strip() else []
    return [str(item or "").strip().upper() for item in _as_list(value) if str(item or "").strip()]


def _success_criteria(role: str, fields: List[str]) -> str:
    if role == "metric":
        return "Only pass when metric/value/unit/period/scope/source_ref are present and traceable to an A/B source."
    if role == "counter":
        return "Only pass when counter/risk evidence is traceable and not merely support evidence."
    if role in {"source_check", "filing"}:
        return "Only pass when the original source URL/title can be traced to an authoritative source."
    return f"Only pass when required fields are present: {', '.join(fields)}."


def validate_requirement_quality(requirement: Dict[str, Any]) -> Dict[str, Any]:
    item = dict(requirement or {})
    role = _role(item.get("proof_role"))
    required = [str(field or "").strip() for field in _as_list(item.get("required_fields")) if str(field or "").strip()]
    suggested = list(REQUIRED_FIELDS_BY_ROLE.get(role, REQUIRED_FIELDS_BY_ROLE["support"]))
    required_set = {field.lower() for field in required}
    suggested_set = {field.lower() for field in suggested}
    issues: List[Dict[str, Any]] = []
    if not required:
        issues.append({"type": "requirement_missing_required_fields", "severity": "high"})
    generic = sorted(required_set & GENERIC_FIELD_NAMES)
    if generic and len(required_set) <= 2:
        issues.append({"type": "requirement_fields_too_generic", "severity": "medium", "fields": generic})
    if role == "metric":
        missing = [field for field in suggested if field.lower() not in required_set]
        if missing:
            issues.append({"type": "metric_missing_required_fields", "severity": "high", "missing_fields": missing})
        levels = set(_source_levels(item.get("required_source_level") or item.get("min_source_level")))
        if not levels or not levels <= {"A", "B"}:
            issues.append({"type": "metric_requires_ab_source_level", "severity": "high"})
    if role == "counter" and "counter_signal" not in required_set:
        issues.append({"type": "counter_missing_counter_signal", "severity": "medium"})
    status = "pass" if not issues else "needs_repair"
    return {
        "quality_check_version": "requirement_quality_v1",
        "status": status,
        "issues": issues,
        "suggested_required_fields": suggested,
        "source_strategy": source_strategy_for_role(role),
        "success_criteria": _success_criteria(role, suggested),
        "reject_if": ["snippet_only", "no_source_url", "marketing_copy_only", "no_date"] if role == "metric" else ["snippet_only", "no_source_url", "marketing_copy_only"],
    }
