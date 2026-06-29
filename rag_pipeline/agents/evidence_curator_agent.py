from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Sequence

from rag_pipeline.contracts.evidence_dirty_gate import evaluate_dirty_gate


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _compact(value: Any, limit: int = 420) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:limit].rstrip()


def _source(item: Dict[str, Any]) -> Dict[str, Any]:
    return _as_dict(item.get("source"))


def _source_url(item: Dict[str, Any]) -> str:
    source = _source(item)
    return str(item.get("source_url") or source.get("url") or source.get("source_url") or "").strip()


def _source_level(item: Dict[str, Any]) -> str:
    return str(item.get("source_level") or _source(item).get("source_level") or "C").strip().upper() or "C"


def _fact_text(item: Dict[str, Any]) -> str:
    return _compact(
        item.get("clean_fact")
        or item.get("distilled_fact")
        or item.get("fact")
        or item.get("content")
        or item.get("evidence"),
        520,
    )


def _dedupe_key(item: Dict[str, Any]) -> str:
    text = re.sub(r"\s+", "", _fact_text(item)).lower()
    source = _source_url(item).lower().rstrip("/")
    return hashlib.sha1(f"{source}|{text}".encode("utf-8")).hexdigest()


def _fact_type(item: Dict[str, Any]) -> str:
    text = " ".join(
        str(value or "")
        for value in (
            item.get("proof_role"),
            item.get("analysis_role"),
            item.get("metric"),
            item.get("fact_type"),
            _fact_text(item),
        )
    )
    if re.search(r"市场|规模|出货|份额|增速|价格|成本|订单|亿元|万台|%", text):
        return "market_signal"
    if re.search(r"政策|监管|标准|目录|补贴|方案", text):
        return "policy_signal"
    if re.search(r"风险|失败|下滑|取消|约束|瓶颈|挑战", text):
        return "risk_signal"
    if re.search(r"客户|订单|采购|中标|部署|应用|案例|场景", text):
        return "case_signal"
    if re.search(r"技术|算法|控制|传感器|续航|供应链|国产化", text):
        return "technology_signal"
    return "context_signal"


def _topic_fit(item: Dict[str, Any], query: str) -> str:
    query_text = str(query or "").lower()
    fact_text = " ".join(
        str(value or "")
        for value in (
            _fact_text(item),
            item.get("source_title"),
            _source(item).get("title"),
            _source_url(item),
        )
    ).lower()
    low_altitude_query = bool(re.search(r"低空|evtol|无人机|通航|空域|low[-\s]?altitude", query_text))
    if not low_altitude_query:
        return "direct"
    if re.search(r"低空|evtol|无人机|无人驾驶飞行|通航|直升机|飞行器|航空器|空域|航线|飞行服务|低空旅游|低空物流", fact_text):
        return "direct"
    if re.search(r"人形机器人|机器人|储能|数据要素|算力|大模型|半导体|光伏|新能源车", fact_text):
        return "off_topic"
    if re.search(r"政策|监管|标准|基础设施|供应链|商业化|应用场景|安全|保险", fact_text):
        return "related"
    return "background"


def _usable_for(fact_type: str) -> List[str]:
    mapping = {
        "market_signal": ["market_size", "competition", "commercialization"],
        "policy_signal": ["policy", "risk", "commercialization"],
        "risk_signal": ["risk", "boundary", "counter"],
        "case_signal": ["case", "demand", "commercialization"],
        "technology_signal": ["technology", "supply_chain", "commercialization"],
        "context_signal": ["background", "trend"],
    }
    return mapping.get(fact_type, ["background"])


def _claim_strength_hint(item: Dict[str, Any], fact_type: str) -> str:
    level = _source_level(item)
    allowed = str(item.get("allowed_use") or "").strip().lower()
    if allowed == "directional_signal" or level in {"C", "D"}:
        return "directional"
    if level in {"A", "B"} and fact_type in {"market_signal", "policy_signal", "case_signal", "technology_signal"}:
        return "moderate"
    return "directional"


def _limitations(item: Dict[str, Any]) -> List[str]:
    limitations: List[str] = []
    level = _source_level(item)
    if level in {"C", "D"}:
        limitations.append("来源可用于方向性分析，不应单独支撑强结论。")
    metric = str(item.get("metric") or "").strip()
    if metric and (not str(item.get("unit") or "").strip() or not str(item.get("period") or "").strip()):
        limitations.append("指标字段不完整，正文需说明口径或使用谨慎措辞。")
    if item.get("source_level_gap"):
        limitations.append("来源等级低于原任务偏好，建议作为辅助信号。")
    return limitations or ["需要结合其他证据交叉判断。"]


def _curated_note(item: Dict[str, Any], merged_ids: List[str], *, query: str = "") -> Dict[str, Any]:
    fact_type = _fact_type(item)
    strength = _claim_strength_hint(item, fact_type)
    source = _source(item)
    topic_fit = _topic_fit(item, query)
    usable_for = ["background"] if topic_fit == "off_topic" else _usable_for(fact_type)
    evidence_use_level = "background_only" if topic_fit == "off_topic" else ("directional_signal" if strength == "directional" else "analysis_signal")
    return {
        "schema_version": "curated_evidence_v1",
        "evidence_id": str(item.get("evidence_id") or (merged_ids[0] if merged_ids else "")).strip(),
        "merged_evidence_ids": merged_ids,
        "source_id": str(item.get("source_id") or item.get("run_source_id") or source.get("source_id") or "").strip(),
        "source_url": _source_url(item),
        "source_title": _compact(source.get("title") or item.get("source_title"), 160),
        "source_level": _source_level(item),
        "clean_fact": _fact_text(item),
        "fact_type": fact_type,
        "topic_fit": topic_fit,
        "usable_for": usable_for,
        "evidence_use_level": evidence_use_level,
        "claim_strength_hint": strength,
        "can_support_claim": topic_fit != "off_topic",
        "limitations": _limitations(item),
        "must_not_use_as": ["official_statistic"] if strength == "directional" else [],
        "dirty": False,
        "lineage": _as_dict(item.get("lineage")),
        "requirement_id": str(item.get("requirement_id") or "").strip(),
        "chapter_id": str(item.get("chapter_id") or _as_dict(item.get("lineage")).get("chapter_id") or "").strip(),
        "proof_role": str(item.get("proof_role") or "").strip(),
    }


def curate_evidence_batch(
    evidence_items: Sequence[Dict[str, Any]],
    *,
    query: str = "",
    max_items: int = 240,
) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    dirty_blocked: List[Dict[str, Any]] = []
    input_count = 0
    for item in evidence_items:
        if not isinstance(item, dict):
            continue
        input_count += 1
        dirty_gate = evaluate_dirty_gate(item)
        if dirty_gate["status"] == "blocked":
            dirty_blocked.append({"evidence_id": item.get("evidence_id"), "dirty_gate": dirty_gate})
            continue
        groups.setdefault(_dedupe_key(item), []).append(item)

    curated: List[Dict[str, Any]] = []
    deduped_count = 0
    for grouped_items in groups.values():
        representative = max(
            grouped_items,
            key=lambda item: (
                1 if _source_level(item) in {"A", "B"} else 0,
                len(_fact_text(item)),
            ),
        )
        merged_ids = [
            str(item.get("evidence_id") or "").strip()
            for item in grouped_items
            if str(item.get("evidence_id") or "").strip()
        ]
        deduped_count += max(0, len(merged_ids) - 1)
        note = _curated_note(representative, merged_ids, query=query)
        if note["clean_fact"]:
            curated.append(note)

    curated.sort(
        key=lambda item: (
            1 if item["source_level"] in {"A", "B"} else 0,
            1 if item["claim_strength_hint"] == "moderate" else 0,
            len(item["clean_fact"]),
        ),
        reverse=True,
    )
    curated = curated[:max_items]
    return {
        "schema_version": "evidence_curator_result_v1",
        "status": "ready" if curated else "insufficient",
        "query": query,
        "input_count": input_count,
        "curated_evidence_count": len(curated),
        "dirty_blocked_count": len(dirty_blocked),
        "deduped_count": deduped_count,
        "curated_evidence": curated,
        "dirty_blocked": dirty_blocked[:20],
    }
