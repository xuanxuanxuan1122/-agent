from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

from .public_report_sanitizer import (
    has_internal_gap_language,
    remove_hard_industry_templates,
    rewrite_internal_gap_language,
)
from rag_pipeline.contracts.public_text_guard import public_text_quality


def _compact(value: Any, max_chars: int = 700) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "..."


def _public_text(value: Any, max_chars: int = 700) -> str:
    text = remove_hard_industry_templates(rewrite_internal_gap_language(_compact(value, max_chars)))
    if not text or has_internal_gap_language(text):
        return ""
    guard = public_text_quality(text)
    if guard.get("severity") == "reject":
        return ""
    return str(guard.get("cleaned") or text).strip()


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _claim_headline(value: Any, *, max_chars: int = 36) -> str:
    text = _public_text(value, 240)
    if not text:
        return ""
    head = re.split(r"[\u3002\uff1b;.!?\uff01\uff1f\n]", text, 1)[0].strip(" \uff0c,\uff1a:\u201c\u201d\"'")
    if not head:
        return ""
    return head[:max_chars].rstrip(" \uff0c,\uff1a:")


def _dedupe(values: Sequence[str], *, limit: int = 4) -> List[str]:
    result: List[str] = []
    seen: set[str] = set()
    for value in values:
        text = _public_text(value, 700)
        key = re.sub(r"\s+", "", text).lower()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def _strip_sentence_end(text: str) -> str:
    return re.sub(r"[\s。.!?！？；;,，]+$", "", str(text or "").strip())


def _evidence_context_bridge(head: str, family: str, facts: Sequence[str]) -> str:
    cleaned = [_strip_sentence_end(item) for item in facts if _strip_sentence_end(item)]
    if not cleaned:
        return ""
    fact_text = "；".join(cleaned)
    return f"已引用证据显示，{fact_text}。"


def _block_family(block_type: Any) -> str:
    value = str(block_type or "").strip().lower()
    if value in {"case_comparison", "customer_painpoint_matrix"}:
        return "case"
    if value in {"metric_reconciliation", "market_size", "growth_metric"}:
        return "metric"
    if value in {"risk_trigger", "boundary", "counter_evidence"}:
        return "risk"
    if value in {"technology_maturity", "technology_readiness"}:
        return "technology"
    return "general"


def _mechanism_bridge(head: str, family: str) -> str:
    return ""


def build_public_bridge_pack(
    *,
    claim: str,
    evidence_texts: Sequence[Any] | None = None,
    block_type: str | None = None,
    claim_strength: str | None = None,
    boundary: str | None = None,
) -> Dict[str, Any]:
    head = _claim_headline(claim)
    family = _block_family(block_type)
    facts = _dedupe([str(item or "") for item in _as_list(evidence_texts)], limit=3)
    strength = str(claim_strength or "directional").strip().lower() or "directional"

    evidence_context = _public_text(_evidence_context_bridge(head, family, facts), 1200) if facts else ""

    mechanism = _public_text(_mechanism_bridge(head, family), 900)
    boundary_text = _public_text(boundary, 700)
    if not boundary_text:
        if strength in {"weak", "directional"}:
            boundary_text = _public_text(
                "目前仍属于方向性结论，只有更多同类动作连续出现，结论强度才会继续上调。",
                700,
            )
        else:
            boundary_text = _public_text(
                "边界仍是已引用来源能够覆盖的场景，超过该范围的外推需要更多材料印证。",
                700,
            )

    implication = ""

    return {
        "schema_version": "public_narrative_bridge_v1",
        "claim_head": head,
        "block_family": family,
        "evidence_context": evidence_context,
        "mechanism_bridge": mechanism,
        "boundary_bridge": boundary_text,
        "implication_bridge": implication,
        "template_keys": [
            f"{family}:mechanism:{re.sub(r'\\s+', '', head).lower()[:48]}",
            f"{family}:implication:{re.sub(r'\\s+', '', head).lower()[:48]}",
        ],
    }
