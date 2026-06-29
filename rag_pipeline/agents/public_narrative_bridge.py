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
    if family == "case":
        tail = (
            "这些主体和动作让抽象判断落到具体场景、责任分工和执行条件中，"
            "比单纯的话题讨论更接近可复核材料。"
        )
    elif family == "metric":
        tail = (
            "这些数字需要放到可比较的指标口径中，"
            "才能区分规模信号、阶段性波动和单一来源估算。"
        )
    elif family == "risk":
        tail = (
            "这些约束让机会判断不再只看增长空间，"
            "还要同时看安全、成本、责任和执行门槛。"
        )
    elif family == "technology":
        tail = (
            "这些执行条件比单纯的能力展示更关键，"
            "因为实际影响取决于适用场景、协作成本和持续运行要求。"
        )
    else:
        tail = (
            "这些主体、场景和约束让抽象趋势落到具体产业动作里，"
            "也让机会判断有了可继续跟踪的现实抓手。"
        )
    return f"公开材料提到，{fact_text}。{tail}"


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
    if family == "case":
        return (
            "关键不在概念热度，而在材料是否呈现明确主体、具体场景和连续动作。"
            "一旦相关动作从单个样本扩展到更多地区、机构或人群，判断就会从孤立现象转向更稳定的趋势；"
            "如果样本仍少，它只能说明早期变化正在发生，结论强度仍要看后续材料是否重复出现。"
            "这比单看宣传表述更能判断变化是否具有重复性。"
        )
    if family == "metric":
        return (
            "指标需要回到统计口径、覆盖范围和时间窗口里理解。"
            "口径一致的数字能够校准市场空间、增长速度或渗透节奏；"
            "口径不一致时，数字更像参照坐标，不能直接推出行业总量。"
        )
    if family == "risk":
        return (
            "风险线索的作用是识别机会兑现前的触发条件和约束。"
            "当安全、成本、责任或监管压力放大时，原本的增长预期需要收缩；"
            "这不是否定全部机会，而是给机会判断设置边界。"
        )
    if family == "technology":
        return (
            "实际影响取决于可靠性、执行成本和场景适配是否同步改善。"
            "能力展示只能证明可能性存在，只有进入稳定流程并承担持续运行要求，"
            "才更接近可复核的应用变化。"
        )
    return (
        "判断这类变化时，重点是主体行动是否持续、影响路径是否清楚、"
        "约束条件是否可解释。相关材料如果能说明谁在行动、改变了哪些流程、"
        "又受到哪些外部条件限制，就能更具体地解释岗位任务、组织安排或资源配置的变化；"
        "如果这些关系尚不完整，结论就应保留在明确场景和时间窗口内。"
    )


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

    implication = _public_text(
        "这一变化需要放回具体对象和工作流程中理解：谁在行动、哪些任务被改变、"
        "主体行动是否持续、影响路径是否清楚、约束条件是否可解释，"
        "共同决定它对岗位任务、组织安排和资源配置的影响强度。"
        "当这些关系能够连续出现时，相关变化就不只是孤立现象，"
        "而是可以解释真实工作流程如何调整的持续信号。",
        900,
    )

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
