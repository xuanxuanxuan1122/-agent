from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, Iterable, List, Sequence

from rag_pipeline.contracts.public_text_guard import public_text_quality
from rag_pipeline.runtime_cache import json_safe_default


METRIC_ASSET_VERSION = "metric_asset_v1"
_PERCENT_RE = re.compile(r"\d+(?:\.\d+)?\s*%")
_VALUE_RE = re.compile(
    r"^\s*(?:约|超过|达到|大于|小于|不少于|不低于)?\s*"
    r"\d+(?:\.\d+)?\s*(?:[-~至]\s*\d+(?:\.\d+)?\s*)?"
    r"(?:%|pct|个百分点|亿美元|亿元|万亿元|美元|元|个|家|人|次|项|项目|平方米|万平方米|亿平方米|GB|TB|tokens?|CAGR)?\s*$",
    re.I,
)
_UNIT_RE = re.compile(
    r"^(?:%|pct|百分点|亿美元|亿元|万亿元|美元|元|个|家|人|次|项|项目|平方米|万平方米|亿平方米|CAGR|GB|TB|token|tokens)$",
    re.I,
)
_PERIOD_RE = re.compile(
    r"^(?:截至)?\s*\d{4}\s*(?:年)?(?:\s*(?:Q[1-4]|第[一二三四1234]季度|上半年|下半年))?"
    r"(?:\s*[-至到]\s*(?:\d{4}\s*(?:年)?(?:\s*(?:Q[1-4]|第[一二三四1234]季度|上半年|下半年))?))?\s*$",
    re.I,
)
_PUBLISHED_AT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", re.I)


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if value in (None, ""):
        return []
    return [value]


def _text(value: Any) -> str:
    return str(value or "").strip()


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


def _stable_hash(value: Any, *, length: int = 14) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=json_safe_default)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def _first_text(*values: Any) -> str:
    for value in values:
        text = _text(value)
        if text:
            return text
    return ""


def _source_id(card: Dict[str, Any]) -> str:
    source = _as_dict(card.get("source"))
    return _first_text(
        card.get("source_id"),
        card.get("source_ref"),
        card.get("citation_ref"),
        source.get("source_id"),
        source.get("id"),
        source.get("ref"),
    )


def _unit_from_value(value: str) -> str:
    if _PERCENT_RE.search(value):
        return "%"
    return ""


def _metric_candidate(card: Dict[str, Any]) -> bool:
    proof_role = _text(card.get("proof_role") or card.get("analysis_role")).lower()
    explicit_non_metric_roles = {
        "source_check",
        "official_data",
        "filing",
        "case",
        "counter",
        "risk",
        "technology_product",
        "boundary",
        "context",
        "support",
    }
    if proof_role in explicit_non_metric_roles and not _as_list(card.get("metric_missing_fields")):
        return False
    return bool(
        proof_role in {"metric", "market_data", "quantitative", "quant_metric"}
        or card.get("metric")
        or card.get("indicator")
        or card.get("value")
        or card.get("metric_value")
        or _as_list(card.get("metric_missing_fields"))
    )


def _missing_fields(asset: Dict[str, Any]) -> List[str]:
    missing: List[str] = []
    for field in ("metric", "value", "unit", "period", "source_id"):
        if not _text(asset.get(field)):
            missing.append(field)
    return missing


def metric_is_complete(metric_asset: Dict[str, Any]) -> bool:
    return not _missing_fields(_as_dict(metric_asset))


def _hard_public_reject(text: str) -> bool:
    guard = public_text_quality(text)
    reasons = set(_as_list(guard.get("reasons")))
    return bool(guard.get("severity") == "reject" and reasons - {"too_short_after_cleaning"})


def _valid_metric_value(value: str) -> bool:
    text = _text(value)
    if not text or len(text) > 60:
        return False
    if _hard_public_reject(text):
        return False
    if not re.search(r"\d", text):
        return False
    return bool(_VALUE_RE.match(text))


def _valid_metric_unit(unit: str, value: str = "") -> bool:
    text = _text(unit)
    if not text:
        return bool(_unit_from_value(value))
    if len(text) > 24:
        return False
    if _hard_public_reject(text):
        return False
    return bool(_UNIT_RE.match(text))


def _valid_metric_period(period: str) -> tuple[bool, str]:
    text = _text(period)
    if not text:
        return False, "missing_period"
    if _PUBLISHED_AT_RE.match(text):
        return False, "published_at_as_period"
    if len(text) > 40:
        return False, "invalid_period"
    if _hard_public_reject(text):
        return False, "dirty_period"
    return bool(_PERIOD_RE.match(text)), "invalid_period"


def _semantic_reject_reasons(asset: Dict[str, Any]) -> List[str]:
    reasons: List[str] = []
    if not _valid_metric_value(_text(asset.get("value"))):
        reasons.append("invalid_value")
    if not _valid_metric_unit(_text(asset.get("unit")), _text(asset.get("value"))):
        reasons.append("invalid_unit")
    period_ok, period_reason = _valid_metric_period(_text(asset.get("period")))
    if not period_ok:
        reasons.append(period_reason)
    combined = " ".join(_text(asset.get(key)) for key in ("metric", "value", "unit", "period", "scope"))
    guard = public_text_quality(combined)
    if guard.get("severity") == "reject":
        reasons.extend(f"dirty_{reason}" for reason in _as_list(guard.get("reasons")))
    return _dedupe(reasons)


def _table_allowed(asset: Dict[str, Any]) -> bool:
    status = _text(asset.get("status")).lower()
    allowed_use = _text(asset.get("allowed_use")).lower()
    return status not in {"rejected", "stale", "superseded"} and allowed_use not in {
        "rejected",
        "clue",
        "appendix_only",
        "not_allowed_until_repaired",
    }


def _metric_id(asset: Dict[str, Any]) -> str:
    return _first_text(
        asset.get("metric_id"),
        asset.get("fact_id"),
        asset.get("evidence_id"),
        f"MET-{_stable_hash([asset.get('metric'), asset.get('value'), asset.get('unit'), asset.get('period'), asset.get('source_id')])}",
    )


def build_metric_assets(
    fact_cards: Sequence[Dict[str, Any]],
    source_registry: Sequence[Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    del source_registry  # v1 derives source ids from fact cards; kept for API compatibility.
    assets: List[Dict[str, Any]] = []
    seen = set()
    for raw in list(fact_cards or []):
        card = _as_dict(raw)
        if not card or not _metric_candidate(card):
            continue
        value = _first_text(card.get("value"), card.get("metric_value"), card.get("number"), card.get("amount"))
        unit = _first_text(card.get("unit"), _unit_from_value(value))
        asset = {
            "schema_version": METRIC_ASSET_VERSION,
            "metric_id": "",
            "fact_id": _first_text(card.get("fact_id"), card.get("evidence_id"), card.get("id")),
            "evidence_id": _first_text(card.get("evidence_id"), card.get("fact_id"), card.get("id")),
            "requirement_id": _first_text(card.get("requirement_id"), *_as_list(card.get("requirement_ids"))),
            "chapter_id": _first_text(card.get("chapter_id")),
            "proof_role": _first_text(card.get("proof_role"), card.get("analysis_role")),
            "metric": _first_text(card.get("metric"), card.get("indicator"), card.get("name")),
            "value": value,
            "unit": unit,
            "period": _first_text(card.get("period"), card.get("year"), card.get("date")),
            "scope": _first_text(card.get("scope"), card.get("market_scope"), card.get("region")),
            "source_id": _source_id(card),
            "source_level": _first_text(card.get("source_level")),
            "allowed_use": _first_text(card.get("allowed_use")),
            "status": _first_text(card.get("status")),
        }
        asset["metric_id"] = _metric_id(asset)
        asset["missing_fields"] = _dedupe([*_as_list(card.get("metric_missing_fields")), *_missing_fields(asset)])
        asset["complete"] = not asset["missing_fields"]
        asset["syntactic_complete"] = bool(asset["complete"])
        semantic_reject_reasons = _semantic_reject_reasons(asset) if asset["complete"] else []
        asset["semantic_complete"] = bool(asset["complete"] and not semantic_reject_reasons)
        asset["reject_reasons"] = semantic_reject_reasons
        asset["table_ready"] = bool(asset["semantic_complete"] and _table_allowed(asset))
        if asset["metric_id"] in seen:
            continue
        seen.add(asset["metric_id"])
        assets.append(asset)
    return assets
