from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, Iterable, List, Sequence

from rag_pipeline.runtime_cache import json_safe_default


METRIC_ASSET_VERSION = "metric_asset_v1"
_PERCENT_RE = re.compile(r"\d+(?:\.\d+)?\s*%")


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
    return bool(
        proof_role == "metric"
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
        asset["table_ready"] = bool(asset["complete"] and _table_allowed(asset))
        if asset["metric_id"] in seen:
            continue
        seen.add(asset["metric_id"])
        assets.append(asset)
    return assets
