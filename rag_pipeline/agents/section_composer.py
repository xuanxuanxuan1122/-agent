from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterable, List, Sequence

from .report_contracts import ClaimUnit, EvidenceFactCard


def _env_int(name: str, default: int, *, min_value: int = 0, max_value: int = 100000) -> int:
    try:
        value = int(str(os.getenv(name, "")).strip() or default)
    except Exception:
        value = default
    return max(min_value, min(max_value, value))


def _text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _dedupe(values: Iterable[str], *, limit: int = 4) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        text = _text(value)
        if not text:
            continue
        key = re.sub(r"\W+", "", text.lower())[:120]
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def _lens(block_type: str) -> str:
    block = str(block_type or "").strip()
    if block == "metric_reconciliation":
        return "metric"
    if block in {"case_comparison", "customer_painpoint_matrix"}:
        return "case"
    if block == "technology_maturity":
        return "technology"
    if block in {"risk_trigger", "verification_checklist", "scenario_analysis"}:
        return "risk"
    if block == "competitive_positioning":
        return "competition"
    if block == "unit_economics":
        return "commercial"
    return "general"


EMPTY_SUBJECTS = {
    "为此",
    "近日",
    "当前",
    "目前",
    "本文",
    "这类",
    "相关",
    "材料",
    "报告",
    "资料",
}


def _public_subject(value: str, fallback: str) -> str:
    subject = _text(value)
    if subject in EMPTY_SUBJECTS or len(subject) <= 1:
        return fallback
    if len(subject) > 24:
        subject = re.split(r"[，,；;：:。.!！?？]", subject, 1)[0].strip()
    if subject in EMPTY_SUBJECTS or len(subject) <= 1:
        return fallback
    return subject[:24]


def _is_snippet_like(text: str) -> bool:
    if not text:
        return True
    if re.search(r"字体\s*[:：]\s*大\s*中\s*小", text):
        return True
    if re.search(r"国内垂直领域研报服务|以下为本次访谈实录|电子工程专辑|爱分析访谈", text):
        return True
    if re.search(r"AI\s*时代，唯一确定的是数据", text, flags=re.I):
        return True
    if re.match(r"^[^。；;]{6,90}[｜|][^。；;]{2,90}\s*-\s*[^:：。]{2,50}(?:[（(]20\d{2}[^）)]*[）)])?[:：]", text):
        return True
    if re.match(r"^[^。；;]{6,90}\s*-\s*[^:：。]{2,50}(?:[（(]20\d{2}[^）)]*[）)])?[:：]", text):
        return True
    if re.match(r"^(?:显示|为此|因此|当前|相关|本文)[，,：:]", text):
        return True
    if re.fullmatch(r"(?:Publication|Published|Release|Updated)\s+date\s*[:：]\s*[\w\s,./-]{4,40}", text, flags=re.I):
        return True
    if re.fullmatch(r"(?:\d{1,2}\s+[A-Za-z]{3,9}\s+20\d{2}|20\d{2}[-/]\d{1,2}[-/]\d{1,2})", text):
        return True
    if "..." in text or "…" in text:
        return True
    if re.match(r"^[^。；;]{6,60}-[^:：。]{2,30}[:：]", text):
        return True
    if re.match(r"^(?:近日|今日|日前|今年\s*\d+\s*月份?|过去\s*\d+\s*[天周月年]|一盆|一句|一篇|Over the weekend)\b", text):
        return True
    if re.match(r"^[A-Za-z _-]{3,30}\s*[:：]\s*\d+(?:\.\d+)?%?$", text):
        return True
    if re.search(r"https?://|Skip to content|Product Documentation|picture intentionally omitted", text, flags=re.I):
        return True
    chinese = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    return bool(len(text) > 220 and latin > 160 and chinese / max(1, chinese + latin) < 0.25)


def _clean_analysis_basis_text(value: Any, *, max_chars: int = 260) -> str:
    text = _text(value)
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:[+-]\d{2}:\d{2})?", "", text)
    text = re.sub(r"\s{2,}", " ", text).strip(" ，,；;。")
    if not text or _is_snippet_like(text):
        return ""
    return _compact(text, max_chars) if len(text) > max_chars else text


def _card_matches_lens(card: EvidenceFactCard, lens: str) -> bool:
    affinity = {str(item or "").strip() for item in card.block_affinity if str(item or "").strip()}
    if lens == "metric":
        return card.fact_type == "metric" or "metric_reconciliation" in affinity
    if lens == "case":
        return card.fact_type in {"case", "customer_case", "directional"} or affinity.intersection({"case_comparison", "customer_painpoint_matrix"})
    if lens == "technology":
        blob = " ".join([card.fact_type, card.variable, card.distilled_fact]).lower()
        return card.fact_type in {"technology", "technology_product", "standard"} or "technology_maturity" in affinity or any(
            token in blob for token in ("tool", "security", "standard", "permission", "workflow", "技术", "标准", "权限", "安全")
        )
    if lens == "risk":
        blob = " ".join([card.fact_type, card.variable, card.distilled_fact]).lower()
        return card.fact_type in {"counter", "risk"} or "risk_trigger" in affinity or any(
            token in blob for token in ("risk", "counter", "failure", "安全", "风险", "失败", "成本", "责任")
        )
    if lens == "commercial":
        blob = " ".join([card.fact_type, card.variable, card.distilled_fact]).lower()
        return card.fact_type in {"metric", "case"} or any(
            token in blob for token in ("revenue", "pricing", "order", "procurement", "收入", "付费", "订单", "采购", "续约")
        )
    if lens == "competition":
        blob = " ".join([card.fact_type, card.variable, card.distilled_fact]).lower()
        return card.fact_type in {"case", "directional"} or any(
            token in blob for token in ("competition", "player", "ecosystem", "竞争", "玩家", "生态", "渠道")
        )
    return True


def _valid_cards(cards: Sequence[EvidenceFactCard], lens: str) -> List[EvidenceFactCard]:
    result: List[EvidenceFactCard] = []
    for card in cards:
        if not card.is_valid_for_report:
            continue
        if not card.source_ref:
            continue
        if _is_snippet_like(card.distilled_fact):
            continue
        if not _card_matches_lens(card, lens):
            continue
        result.append(card)
    return result


def _metric_sentence(card: EvidenceFactCard) -> str:
    subject = _public_subject(card.subject, "相关主体")
    variable = card.variable or "关键指标"
    value = card.value
    if value and card.unit and card.unit not in value:
        value = f"{value}{card.unit}"
    period = card.time_or_scope
    fact = card.distilled_fact.rstrip("。.")
    if value:
        prefix = f"{subject}的{variable}"
        if period:
            return f"{prefix}在{period}为{value}，这一指标用于判断市场空间和兑现节奏。"
        return f"{prefix}为{value}，这一指标用于判断市场空间和兑现节奏。"
    return f"{fact}，这一事实用于校准{variable}的规模和可比性。"


def _case_sentence(card: EvidenceFactCard) -> str:
    subject = _public_subject(card.subject, "相关玩家")
    fact = card.distilled_fact.rstrip("。.")
    variable = card.variable or "落地深度"
    return f"{subject}的动作显示{variable}已经出现可观察样本；{fact}。这类样本的意义在于验证需求是否从试用进入具体流程。"


def _technology_sentence(card: EvidenceFactCard) -> str:
    fact = card.distilled_fact.rstrip("。.")
    variable = card.variable or "技术成熟度"
    return f"{fact}。它对应的关键变量是{variable}，会影响工具调用、权限、安全和部署稳定性。"


def _risk_sentence(card: EvidenceFactCard) -> str:
    fact = card.distilled_fact.rstrip("。.")
    variable = card.variable or "风险边界"
    return f"{fact}。这个反向样本提示{variable}仍可能改变结论强度，需要把商业化判断限制在已验证场景内。"


def _boundary_sentence(lens: str, card: EvidenceFactCard, strength: str) -> str:
    variable = card.variable or card.time_or_scope or ""
    if lens == "metric":
        return "边界在于指标的主体、范围、期间和统计口径是否保持一致。"
    if lens in {"case", "commercial", "competition"}:
        if variable:
            return f"边界在于{variable}是否能从单点样本延伸到更多客户、流程和付费链路。"
        return "边界在于样本是否能延伸到更多客户、流程和付费链路。"
    if lens == "technology":
        if variable:
            return f"边界在于{variable}是否同时满足可靠性、权限、安全和集成成本要求。"
        return "边界在于可靠性、权限、安全和集成成本是否同时满足部署要求。"
    if lens == "risk":
        if variable:
            return f"触发条件是{variable}进一步扩大，进而压低部署节奏、ROI 预期或责任分配确定性。"
        return "触发条件是反向样本继续扩大，进而压低部署节奏、ROI 预期或责任分配确定性。"
    if strength in {"directional", "weak"} and variable:
        return f"边界在于{variable}能否在更多可追溯样本中重复出现。"
    return ""


def _variable_explanation(lens: str, card: EvidenceFactCard) -> str:
    variable = card.variable or card.action_or_signal or card.time_or_scope
    if not variable:
        return ""
    if lens == "metric":
        return f"这说明{variable}不是单个数值本身，而是需求空间、商业化节奏和口径可比性的共同信号。"
    if lens in {"case", "commercial"}:
        return f"这说明{variable}的关键不只是出现案例，而是案例是否具备部署深度、客户复制和付费链路。"
    if lens == "competition":
        return f"这说明{variable}会影响玩家分化、渠道控制和生态入口的判断。"
    if lens == "technology":
        return f"这说明{variable}会影响可靠性、权限、安全和集成成本，进而决定能否进入生产流程。"
    if lens == "risk":
        return f"这说明{variable}是推翻或削弱本章判断的触发条件。"
    return f"这说明{variable}已经成为影响本章判断的核心变量。"


def _general_sentence(card: EvidenceFactCard, lens: str) -> str:
    if lens == "metric":
        return _metric_sentence(card)
    if lens in {"case", "commercial", "competition"}:
        return _case_sentence(card)
    if lens == "technology":
        return _technology_sentence(card)
    if lens == "risk":
        return _risk_sentence(card)
    fact = card.distilled_fact.rstrip("。.")
    variable = card.variable or "章节判断"
    return f"{fact}。这一事实用于判断{variable}是否具备持续性。"


def _claim_analysis_parts(claim_unit: ClaimUnit) -> List[str]:
    parts: List[str] = []
    values: List[Any] = [
        claim_unit.claim,
        claim_unit.paragraph_seed,
        *[_clean_analysis_basis_text(item) for item in claim_unit.evidence_basis[:2]],
        claim_unit.reasoning_chain,
        _clean_analysis_basis_text(claim_unit.limitation_boundary),
    ]
    for value in values:
        text = _text(value)
        if not text or _is_snippet_like(text):
            continue
        parts.append(text)
    return _dedupe(parts, limit=6)


def _select_cards_for_claim(cards: Sequence[EvidenceFactCard], claim_unit: ClaimUnit, *, limit: int = 3) -> List[EvidenceFactCard]:
    by_ref = {card.evidence_id: card for card in cards if card.evidence_id}
    selected: List[EvidenceFactCard] = []
    for ref in claim_unit.evidence_refs:
        card = by_ref.get(ref)
        if card and card not in selected:
            selected.append(card)
        if len(selected) >= limit:
            return selected
    for card in cards:
        if card not in selected:
            selected.append(card)
        if len(selected) >= limit:
            break
    return selected


def _join_public_sentences(parts: Sequence[str]) -> str:
    return " ".join(part.strip() for part in _dedupe(parts, limit=16) if part.strip()).strip()


def _compact_len(text: str) -> int:
    return len(re.sub(r"\s+", "", str(text or "")))


def _section_target_chars() -> int:
    return _env_int("REPORT_COMPOSER_TARGET_SECTION_CHARS", 450, min_value=120, max_value=1200)


def _expansion_sentences(
    *,
    lens: str,
    claim_unit: ClaimUnit,
    selected: Sequence[EvidenceFactCard],
    chapter_question: str,
    strength: str,
) -> List[str]:
    first = selected[0]
    subject = _public_subject(first.subject, "相关主体")
    variable = first.variable or first.action_or_signal or first.time_or_scope or "本章变量"
    clean_basis = [_clean_analysis_basis_text(item) for item in claim_unit.evidence_basis]
    fact_basis = _dedupe([card.distilled_fact for card in selected] + [item for item in clean_basis if item], limit=4)
    facts_text = "；".join(fact_basis[:2])
    if lens == "metric":
        mechanism = f"从分析口径看，{variable}需要同时观察主体、范围、期间和单位是否一致；只有这些口径能够对齐，指标才适合用来判断市场空间、需求弹性和商业化节奏。"
        implication = f"因此，本节不把单个数值当成完整结论，而是把{subject}相关指标放在需求兑现、付费链路和竞争进入节奏之间交叉理解。"
    elif lens in {"case", "commercial"}:
        mechanism = f"从机制上看，{variable}的价值不在于出现一个案例，而在于案例是否进入真实流程、是否需要权限和系统集成、是否能形成可重复的客户使用路径。"
        implication = f"因此，{subject}的样本更适合作为落地深度信号，而不是直接外推为全行业成熟；后续判断应继续围绕部署深度、客户复制和付费链路展开。"
    elif lens == "technology":
        mechanism = f"从技术成熟度看，{variable}会同时影响可靠性、权限治理、安全边界和集成成本；这些变量决定产品能否从演示环境进入生产流程。"
        implication = f"因此，本节把技术事实转化为落地约束：能力本身只是入口，稳定运行、责任划分和系统兼容才决定商业化速度。"
    elif lens == "risk":
        mechanism = f"从风险边界看，{variable}一旦扩大，就可能压低部署节奏、ROI 预期或责任分配确定性，从而改变原有机会判断的强度。"
        implication = f"因此，风险事实不应被放在附录里处理，而应进入正文成为约束条件，帮助区分已经验证的机会和仍需谨慎对待的假设。"
    elif lens == "competition":
        mechanism = f"从竞争结构看，{variable}会影响玩家分化、渠道控制、生态入口和客户迁移成本；同一事实在不同玩家手中可能代表不同的壁垒。"
        implication = f"因此，本节关注的不只是参与者数量，而是哪些能力真正沉淀为客户入口、交付能力和可持续优势。"
    else:
        mechanism = f"从本章问题看，{variable}是把事实转成判断的核心连接点；它决定这些证据究竟只是背景信息，还是已经能支撑方向性分析。"
        implication = f"因此，本节会把事实、机制和边界放在同一个判断框架中，而不是只复述来源材料。"
    boundary = _clean_analysis_basis_text(claim_unit.limitation_boundary) if claim_unit.limitation_boundary else ""
    if not boundary:
        if strength in {"strong", "moderate", "decision_ready", "core_claim"}:
            boundary = f"边界在于这些事实仍需要与更多来源中的同类变量相互印证，尤其要观察{variable}是否能持续出现在不同客户、不同场景或不同时间窗口。"
        else:
            boundary = f"由于该判断属于方向性分析，结论应限定在已引用事实能够覆盖的场景内，重点看{variable}后续是否重复出现并形成更稳定的证据链。"
    question_sentence = f"放回章节问题看，{chapter_question} 的回答不能只依赖单点事实，而要看上述机制能否连续支撑需求、商业化或风险判断。" if chapter_question else ""
    fact_sentence = f"已有事实链显示：{facts_text}。" if facts_text else ""
    return [fact_sentence, mechanism, implication, boundary, question_sentence]


def _expand_to_target(
    *,
    base_parts: Sequence[str],
    lens: str,
    claim_unit: ClaimUnit,
    selected: Sequence[EvidenceFactCard],
    chapter_question: str,
    strength: str,
) -> tuple[str, str]:
    target = _section_target_chars()
    parts = [part for part in base_parts if _text(part)]
    status = "base"
    if _compact_len(_join_public_sentences(parts)) < target:
        for sentence in _expansion_sentences(
            lens=lens,
            claim_unit=claim_unit,
            selected=selected,
            chapter_question=chapter_question,
            strength=strength,
        ):
            if sentence:
                parts.append(sentence)
            if _compact_len(_join_public_sentences(parts)) >= target:
                status = "expanded"
                break
    paragraph = _join_public_sentences(parts)
    if _compact_len(paragraph) >= target:
        status = "expanded"
    elif len(selected) < 2 and not claim_unit.reasoning_chain:
        status = "insufficient_facts"
    return paragraph, status


def compose_section_paragraph(
    *,
    fact_cards: Sequence[EvidenceFactCard],
    claim_unit: ClaimUnit | None = None,
    block_type: str = "",
    chapter_question: str = "",
) -> Dict[str, Any]:
    lens = _lens(block_type)
    claim_unit = claim_unit or ClaimUnit()
    valid = _valid_cards(list(fact_cards), lens)
    if not valid:
        return {
            "composition_status": "dropped",
            "body_composition_status": "dropped",
            "paragraph": "",
            "claim": "",
            "reasoning": "",
            "mechanism": "",
            "counter_evidence": "",
            "used_fact_refs": [],
            "supporting_facts": [],
            "variable_explanation": "",
            "composer_variable_explanation_count": 0,
            "omit_reason": "no_valid_fact_card",
        }

    selected = _select_cards_for_claim(valid, claim_unit, limit=3)
    sentences = [_general_sentence(card, lens) for card in selected]
    variable_explanation = _variable_explanation(lens, selected[0])
    refs = _dedupe([card.evidence_id for card in selected if card.evidence_id], limit=4)
    clean_basis = [_clean_analysis_basis_text(item) for item in claim_unit.evidence_basis]
    facts = _dedupe([*(card.distilled_fact for card in selected), *[item for item in clean_basis if item]], limit=6)
    strength = (claim_unit.claim_strength or selected[0].claim_strength_hint or "").lower()
    has_strong_source = any(card.source_level.upper() in {"A", "B"} for card in selected)
    if strength in {"strong", "moderate", "decision_ready", "core_claim"} or has_strong_source:
        status = "composed"
    else:
        status = "composed_directional"
    paragraph, expansion_status = _expand_to_target(
        base_parts=[*_claim_analysis_parts(claim_unit), *sentences[:2], variable_explanation],
        lens=lens,
        claim_unit=claim_unit,
        selected=selected,
        chapter_question=chapter_question,
        strength=strength,
    )
    if claim_unit.claim and not _is_snippet_like(claim_unit.claim):
        claim = claim_unit.claim
    else:
        claim = sentences[0]
    mechanism = claim_unit.reasoning_chain if claim_unit.reasoning_chain and not _is_snippet_like(claim_unit.reasoning_chain) else paragraph
    boundary = ""
    if claim_unit.limitation_boundary and not _is_snippet_like(claim_unit.limitation_boundary):
        boundary = claim_unit.limitation_boundary
    elif lens == "risk":
        boundary = sentences[0]
    elif strength not in {"strong", "moderate", "decision_ready", "core_claim"}:
        boundary = _boundary_sentence(lens, selected[0], strength)
    elif chapter_question:
        boundary = _boundary_sentence(lens, selected[0], strength)
    return {
        "composition_status": status,
        "body_composition_status": "composed",
        "paragraph": paragraph,
        "claim": claim,
        "reasoning": paragraph,
        "mechanism": mechanism,
        "counter_evidence": boundary,
        "used_fact_refs": refs,
        "supporting_facts": facts,
        "variable_explanation": variable_explanation,
        "composer_variable_explanation_count": 1 if variable_explanation else 0,
        "composer_expansion_status": expansion_status,
        "composer_target_section_chars": _section_target_chars(),
        "composer_paragraph_chars": _compact_len(paragraph),
        "claim_strength": claim_unit.claim_strength or selected[0].claim_strength_hint or ("moderate" if has_strong_source else "directional"),
        "omit_reason": "",
    }
