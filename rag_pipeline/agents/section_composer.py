from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterable, List, Sequence

from .public_report_sanitizer import remove_hard_industry_templates
from .report_contracts import ClaimUnit, EvidenceFactCard


def _env_int(name: str, default: int, *, min_value: int = 0, max_value: int = 100000) -> int:
    try:
        value = int(str(os.getenv(name, "")).strip() or default)
    except Exception:
        value = default
    return max(min_value, min(max_value, value))


def _env_flag(name: str, default: bool = False) -> bool:
    raw = str(os.getenv(name, "")).strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


GENERIC_PUBLIC_SUBJECTS = {"", "相关主体", "相关企业", "相关机构", "主体", "相关场景", "这一场景"}
GENERIC_PUBLIC_VARIABLES = {"", "章节信号", "本章变量", "相关变量", "关键事实", "具体进展"}


def _claim_focus_text(value: Any) -> str:
    text = _text(value)
    if not text:
        return ""
    text = re.split(r"[。；;,.，]", text, maxsplit=1)[0].strip()
    for marker in (
        "呈现",
        "面临",
        "正通过",
        "正加速",
        "已启动",
        "加速",
        "正在",
        "已经",
        "将会",
        "将",
        "开始",
        "进入",
        "推动",
        "成为",
        "需要",
        "显示",
        "说明",
        "会",
    ):
        if marker in text:
            left = text.split(marker, 1)[0].strip()
            if 2 <= len(left) <= 24:
                return left
    return text[:24].strip()


def _public_focus_from_claim(claim_unit: "ClaimUnit", fallback: Any = "") -> str:
    focus = _claim_focus_text(getattr(claim_unit, "claim", ""))
    if focus:
        return focus
    return _claim_focus_text(fallback)


def _public_variable(value: Any, *, fallback: str = "这一变化") -> str:
    variable = _text(value)
    if not variable:
        return fallback
    if variable in GENERIC_PUBLIC_VARIABLES or "章节信号" in variable:
        return fallback
    return variable


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return []


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


def _fact_norm(value: Any) -> str:
    return re.sub(r"\W+", "", _text(value).lower())


def _duplicates_known_fact(text: str, known_facts: Sequence[str]) -> bool:
    normalized = _fact_norm(text)
    if len(normalized) < 24:
        return False
    for fact in known_facts:
        fact_norm = _fact_norm(fact)
        if len(fact_norm) < 24:
            continue
        if normalized == fact_norm or fact_norm in normalized:
            return True
    return False


def _would_repeat_known_fact(parts: Sequence[str], sentence: str, known_facts: Sequence[str]) -> bool:
    existing = _fact_norm(" ".join(parts))
    if not existing:
        return False
    candidate = _fact_norm(sentence)
    if not candidate:
        return False
    for fact in known_facts:
        fact_norm = _fact_norm(fact)
        if len(fact_norm) < 24:
            continue
        if fact_norm in existing and fact_norm in candidate:
            return True
    return False


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


_CLAIM_LIKE_SUBJECT_RE = re.compile(
    r"(呈现|正在|已经|催生|重塑|提供|影响|表明|显示|成为|需要|转向|改变|证明|印证|支撑|加速|整合|推出|纳入|倒逼|主导|聚焦)"
)


def _public_topic_label(subject: str, lens: str) -> str:
    text = _public_subject(subject, "相关场景")
    if text in GENERIC_PUBLIC_SUBJECTS:
        return "相关场景"
    if len(text) >= 16 and _CLAIM_LIKE_SUBJECT_RE.search(text):
        if lens == "metric":
            return "相关指标"
        if lens == "technology":
            return "技术应用"
        if lens == "risk":
            return "风险变量"
        if lens == "competition":
            return "竞争关系"
        return "相关场景"
    return text


def _is_snippet_like(text: str) -> bool:
    if not text:
        return True
    if re.search(r"####\s*(?:\u5b57\u53f7|font|share)", text, flags=re.I):
        return True
    if re.search(r"\u5b57\u53f7\s*[-\u2014]\s*\u5927\s*[-\u2014]\s*\u4e2d\s*[-\u2014]\s*\u5c0f\s*\u5206\u4eab", text):
        return True
    if re.search(r"!\s*\[\s*\]\s*\(", text):
        return True
    if re.search(r"[_-]\u65f6\u653f\u8981\u95fb[_-].*(?:####|!\s*\[\s*\]\s*\()", text):
        return True
    if re.search(r"字体\s*[:：]\s*大\s*中\s*小", text):
        return True
    if re.search(r"字体\s*[:：]\s*\(?\s*javascript", text, flags=re.I):
        return True
    if re.search(r"(?:记者|主持人)\s*[:：].{0,220}(?:请问|谢谢|提问)", text):
        return True
    if re.search(r"国内垂直领域研报服务|以下为本次访谈实录|电子工程专辑|爱分析访谈", text):
        return True
    if re.search(r"现印发给你们，请认真贯彻执行|请认真贯彻执行|附件\s*\d+[^。；;\n]{0,120}征求意见稿", text):
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


_LEADING_ARTIFACT_LABEL_RE = re.compile(
    r"^(?:相关进展|出货/部署|出货量|数据指标|定性事实|指标|样本|事实)\s*[:：]\s*\d+\s*"
)


def _public_fact_text(value: Any) -> str:
    text = remove_hard_industry_templates(_text(value)).strip()
    text = _LEADING_ARTIFACT_LABEL_RE.sub("", text)
    text = re.sub(r"《[^》]{2,120}》已经[^。；;\n]{0,120}(?:现印发给你们，请认真贯彻执行|请认真贯彻执行)[。；;]?", "", text)
    text = re.sub(r"附件\s*\d+[^。；;\n]{0,160}(?:征求意见稿|关键时期)[^。；;\n]{0,160}[。；;]?", "", text)
    text = re.sub(r"\s+", " ", text).strip(" 。；;")
    return text


def _clean_analysis_basis_text(value: Any, *, max_chars: int = 260) -> str:
    text = _text(value)
    if not text:
        return ""
    if re.search(
        r"\b(?:semantic\s+judge|metric\s+fields\s+incomplete|not_allowed_until_repaired|currency_(?:usd|cny)"
        r"|writer_advice|expand_claim_writing|body_short|must_not_render|diagnostic_only|public_text_allowed)\b",
        text,
        flags=re.I,
    ):
        return ""
    if re.search(r"\b(?:source_check|http_status|response_code|status_code)\s*(?:=|:|\u4e3a)", text, flags=re.I):
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
        clean_fact = _public_fact_text(card.distilled_fact)
        if not clean_fact or _is_snippet_like(card.distilled_fact) or _is_snippet_like(clean_fact):
            continue
        if not _card_matches_lens(card, lens):
            continue
        result.append(card)
    return result


def _content_shape_issues(card: EvidenceFactCard) -> set[str]:
    raw = card.raw if isinstance(card.raw, dict) else {}
    values: List[Any] = []
    for payload in (
        raw,
        raw.get("public_fact_card") if isinstance(raw.get("public_fact_card"), dict) else {},
        raw.get("evidence_card") if isinstance(raw.get("evidence_card"), dict) else {},
        raw.get("analysis_input") if isinstance(raw.get("analysis_input"), dict) else {},
    ):
        if not isinstance(payload, dict):
            continue
        issues = payload.get("content_shape_issues")
        if isinstance(issues, str):
            values.append(issues)
        elif isinstance(issues, (list, tuple, set)):
            values.extend(issues)
    return {str(item or "").strip() for item in values if str(item or "").strip()}


def _metric_sentence(card: EvidenceFactCard) -> str:
    subject = _public_subject(card.subject, "相关主体")
    variable = card.variable or "关键指标"
    shape_issues = _content_shape_issues(card)
    value = "" if "generic_metric_name" in shape_issues else card.value
    unit = "" if str(card.unit or "").strip().lower() in {"currency_usd", "currency_cny", "money", "unknown"} else card.unit
    if value and unit and unit not in value:
        value = f"{value}{unit}"
    period = card.time_or_scope
    fact = _public_fact_text(card.distilled_fact).rstrip("。.")
    if value:
        prefix = f"{subject}的{variable}"
        if period:
            return f"{prefix}在{period}为{value}，这一指标需要放回主体、范围和时间窗口中理解。"
        return f"{prefix}为{value}，这一指标需要结合统计范围和同口径材料理解。"
    return f"{fact}，这一事实用于校准{variable}的范围、时间和可比性。"


def _case_sentence(card: EvidenceFactCard) -> str:
    subject = _public_topic_label(card.subject, "case")
    fact = _public_fact_text(card.distilled_fact).rstrip("。.")
    variable = _public_variable(card.variable or card.action_or_signal, fallback="这一变化")
    if subject in GENERIC_PUBLIC_SUBJECTS or subject == "相关场景" or variable in GENERIC_PUBLIC_VARIABLES:
        return f"{fact}。相关变化已经出现具体动作，可继续观察{variable}是否进入实际执行。"
    return f"{fact}。这使{subject}的{variable}从概念描述转向具体动作和实践训练。"


def _technology_sentence(card: EvidenceFactCard) -> str:
    fact = _public_fact_text(card.distilled_fact).rstrip("。.")
    variable = card.variable or "技术成熟度"
    return f"{fact}。{variable}的实际影响取决于可靠性、成本、适用场景和执行条件能否同时成立。"


def _risk_sentence(card: EvidenceFactCard) -> str:
    fact = _public_fact_text(card.distilled_fact).rstrip("。.")
    variable = card.variable or "风险边界"
    return f"{fact}。这类风险信号会影响{variable}，也会改变相关结论的适用范围。"


def _boundary_sentence(lens: str, card: EvidenceFactCard, strength: str) -> str:
    variable = _public_variable(card.variable or card.time_or_scope or "", fallback="")
    if lens == "metric":
        return "指标解释需要同时看主体、范围、期间和统计口径是否保持一致。"
    if lens in {"case", "commercial", "competition"}:
        if variable:
            return f"{variable}还需要结合更多地区、主体和场景样本判断其普遍性。"
        return "相关样本还需要结合更多地区、主体和场景判断其普遍性。"
    if lens == "technology":
        if variable:
            return f"{variable}的实际影响取决于可靠性、权限、安全和集成成本能否同时成立。"
        return "实际影响取决于可靠性、权限、安全和集成成本能否同时成立。"
    if lens == "risk":
        if variable:
            return f"触发条件是{variable}进一步扩大，进而改变执行节奏、成本预期或责任分配确定性。"
        return "触发条件是反向样本继续扩大，进而改变执行节奏、成本预期或责任分配确定性。"
    if strength in {"directional", "weak"} and variable:
        return f"{variable}仍处在样本积累阶段，更适合支持审慎的阶段性判断。"
    return ""


def _variable_explanation(lens: str, card: EvidenceFactCard) -> str:
    variable = _public_variable(card.variable or card.action_or_signal or card.time_or_scope, fallback="这一变化")
    if not variable:
        return ""
    if lens == "metric":
        return f"{variable}的解释力来自主体范围、期间和统计口径是否一致。"
    if lens in {"case", "commercial"}:
        return f"{variable}已经有具体样本支撑，影响范围取决于后续是否继续投入资源并形成重复动作。"
    if lens == "competition":
        return f"这说明{variable}会影响玩家分化、渠道控制和生态入口的判断。"
    if lens == "technology":
        return f"这说明{variable}需要同时观察可靠性、成本、场景适配和执行条件，避免把单点能力直接外推为整体结论。"
    if lens == "risk":
        return f"这说明{variable}是推翻或削弱本章判断的触发条件。"
    return f"{variable}需要结合来源范围、场景深度和时间窗口一起判断，避免把背景信息直接外推为结论。"


def _fact_anchor_phrase(value: Any, *, max_chars: int = 80) -> str:
    text = _public_fact_text(value).strip("銆?.!?锛? ")
    if not text:
        return ""
    if re.search(r"[A-Za-z]", text):
        match = re.search(
            r"(?:in|into|for|within)\s+(?:a|an|the)?\s*([^.;。]{12,90})",
            text,
            flags=re.I,
        )
        if match:
            return _text(match.group(1))[:max_chars].strip(" ,.;")
        words = text.split()
        return " ".join(words[-min(8, len(words)):])[:max_chars].strip(" ,.;")
    parts = [
        _text(part)
        for part in re.split(r"[，,；;。]|显示|提到|开始|均|已经|正在", text)
        if _text(part)
    ]
    for part in reversed(parts):
        if 6 <= len(part) <= max_chars:
            return part
    return text[-max_chars:]


def _general_sentence(card: EvidenceFactCard, lens: str) -> str:
    if lens == "metric":
        return _metric_sentence(card)
    if lens in {"case", "commercial", "competition"}:
        return _case_sentence(card)
    if lens == "technology":
        return _technology_sentence(card)
    if lens == "risk":
        return _risk_sentence(card)
    fact = _public_fact_text(card.distilled_fact).rstrip("。.")
    return f"{fact}。"


def _claim_analysis_parts(claim_unit: ClaimUnit, *, known_facts: Sequence[str] = ()) -> List[str]:
    parts: List[str] = []
    raw = claim_unit.raw if isinstance(claim_unit.raw, dict) else {}
    narrative_supporting = [
        _clean_analysis_basis_text(item, max_chars=320)
        for item in _as_list(raw.get("narrative_supporting_claims"))[:3]
    ]
    role_bridge = ""
    # narrative_supporting claims are already public-facing prose. Adding a
    # generic bridge here makes the report read like an internal analysis trace.
    values: List[Any] = [
        claim_unit.claim,
        claim_unit.paragraph_seed,
        role_bridge,
        *narrative_supporting,
        *[_clean_analysis_basis_text(item) for item in claim_unit.evidence_basis[:2]],
        claim_unit.reasoning_chain,
        # limitation_boundary is internal-only; keep it out of the public paragraph.
    ]
    for value in values:
        text = _text(value)
        if not text or _is_snippet_like(text):
            continue
        if _duplicates_known_fact(text, known_facts):
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
    normalized = [
        _ensure_sentence_terminal(part.strip())
        for part in _dedupe(parts, limit=24)
        if part.strip()
    ]
    return _repair_missing_sentence_punctuation(remove_hard_industry_templates("".join(normalized).strip()))


_PUBLIC_SENTENCE_TERMINAL_RE = re.compile(r"(?:[。！？!?；;：:]|[）)]|[】》]|(?:\[\d{1,5}\])+)$")


def _ensure_sentence_terminal(text: str) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    if _PUBLIC_SENTENCE_TERMINAL_RE.search(value):
        return value
    if re.search(r"[\u4e00-\u9fffA-Za-z0-9]$", value):
        return f"{value}。"
    return value


def _compact_len(text: str) -> int:
    return len(re.sub(r"\s+", "", str(text or "")))


def _near_section_target(text: str, *, ratio: float = 0.95) -> bool:
    target = _section_target_chars()
    return _compact_len(text) >= max(1, int(target * ratio))


def _repair_missing_sentence_punctuation(text: str) -> str:
    cleaned = str(text or "")
    cleaned = re.sub(
        r"([\u4e00-\u9fff])\s+(?=(?:政策|结论|边界|但|不过|同时|这一|这种|该|因此)(?:为|是|仍|需|需要|具有|可以|不能|会|将))",
        r"\1。",
        cleaned,
    )
    return cleaned


def _section_target_chars() -> int:
    return _env_int("REPORT_COMPOSER_TARGET_SECTION_CHARS", 450, min_value=120, max_value=1200)


def _expand_to_target_enabled() -> bool:
    raw = os.getenv("REPORT_COMPOSER_EXPAND_TO_TARGET")
    if raw is not None and str(raw).strip():
        return _env_flag("REPORT_COMPOSER_EXPAND_TO_TARGET", False)
    blueprint_source = str(os.getenv("REPORT_BLUEPRINT_SOURCE", "") or "").strip().lower()
    return blueprint_source in {"claim_first", "claims", "analysis_claims"}


WRITER_ADVICE_EXPANSION_ACTIONS = {
    "expand_claim_writing",
    "rewrite_with_caveat",
    "keep_as_directional",
}


def _writer_advice_actions(claim_unit: ClaimUnit) -> List[Dict[str, Any]]:
    raw = claim_unit.raw if isinstance(claim_unit.raw, dict) else {}
    actions = raw.get("writer_advice_actions")
    if not isinstance(actions, list):
        return []
    return [dict(item) for item in actions if isinstance(item, dict)]


def _writer_advice_requires_expansion(claim_unit: ClaimUnit) -> bool:
    for item in _writer_advice_actions(claim_unit):
        action = str(item.get("action") or item.get("suggested_action") or "").strip().lower()
        if action in WRITER_ADVICE_EXPANSION_ACTIONS:
            return True
    return False


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
    fact_basis = _dedupe([_public_fact_text(card.distilled_fact) for card in selected] + [item for item in clean_basis if item], limit=4)
    facts_text = "；".join(fact_basis[:2])
    fact_reference = facts_text or (_public_fact_text(selected[0].distilled_fact) if selected else "")
    focus = _public_focus_from_claim(claim_unit, fact_reference)
    if subject in GENERIC_PUBLIC_SUBJECTS and focus:
        subject = focus
    subject = _public_topic_label(subject, lens)
    if subject in GENERIC_PUBLIC_SUBJECTS and focus:
        subject = _public_subject(focus, "本章主题")
    if variable in GENERIC_PUBLIC_VARIABLES or "章节信号" in variable:
        variable = focus or "相关变化"
    if variable == subject:
        variable = "这一变化"
    outcome_sentence = ""
    if lens == "metric":
        mechanism = f"{variable}的解释力取决于{subject}对应的统计范围、时间窗口和单位口径；这些条件越清楚，越能判断相关变化是否具备持续性。"
        implication = f"相关数值更适合作为判断{subject}变化节奏的依据，而不是孤立的规模描述。"
    elif lens in {"case", "commercial"}:
        if subject == "相关场景":
            mechanism = f"{variable}之所以重要，是因为公开材料已经出现具体动作或业务安排，能够反映需求、供给或组织配置的真实变化。"
        else:
            mechanism = f"{variable}之所以重要，是因为它已经落到{subject}的具体动作或业务安排上，能够反映需求、供给或组织配置的真实变化。"
        implication = "如果类似动作持续增加，相关变化会更容易沉淀为稳定的产业节奏；如果仍集中在少数主体，则更应关注场景差异和复制条件。"
    elif lens == "technology":
        mechanism = f"{variable}会同时影响可靠性、权限治理、安全边界和集成成本，这些条件决定相关能力能否进入稳定使用。"
        implication = f"技术能力本身只是入口，稳定运行、责任划分和系统兼容才决定实际影响。"
    elif lens == "risk":
        mechanism = f"{variable}一旦扩大，就可能改变执行节奏、成本预期或责任分配确定性，从而压低原有判断的强度。"
        implication = f"风险事实会直接影响机会判断，帮助区分已经验证的进展和仍需谨慎对待的假设。"
    elif lens == "competition":
        mechanism = f"{variable}会影响主体分化、资源控制、协作入口和切换成本；同一事实在不同主体中可能代表不同能力。"
        implication = f"竞争判断不只看参与者数量，更要看哪些能力能够沉淀为组织能力、执行能力和可持续优势。"
    else:
        mechanism = f"{variable}的行业含义在于，它会改变{subject}的任务边界、能力组合和评价标准。"
        implication = "如果这种变化持续进入课程、岗位、工具和组织流程，它对就业结构和培养方式的影响会更稳定。"
    # limitation_boundary is internal-only; the public boundary sentence is generated
    # below from the claim's strength, never from the internal analysis field.
    boundary = ""
    if not boundary:
        if strength in {"strong", "moderate", "decision_ready", "core_claim"}:
            boundary = f"{variable}仍需要结合更多公开披露和连续结果确认，避免把少数主体的动作外推为全部市场。"
        else:
            boundary = f"现阶段关于{variable}的材料更适合支撑阶段性判断，不能直接推出全行业已经完成同样变化。"
    context_label = "本章主题" if subject in GENERIC_PUBLIC_SUBJECTS else subject
    fact_anchor = _fact_anchor_phrase(fact_reference)
    anchor_sentence = (
        f"{fact_anchor}是观察{variable}的具体切口，它把本章判断落到课程、岗位、工具或业务流程的实际变化上。"
        if fact_anchor
        else ""
    )
    if lens == "technology":
        context_sentence = (
            f"在{context_label}中，{variable}会首先影响算力供给、系统集成、权限治理和自动化边界。"
            if chapter_question
            else ""
        )
        depth_sentence = "相关投入只有转化为稳定系统、可控权限、明确成本结构和可复核业务结果，才会真正改变应用边界。"
        subject_sentence = (
            f"{subject}的变化路径更可能表现为先扩大基础设施和工具接入，再影响流程自动化、责任划分和岗位能力边界。"
        )
        comparison_sentence = (
            f"{variable}正在进入技术供给、业务系统和使用门槛的交叉位置。"
        )
        pathway_sentence = (
            "投入规模本身不是终点，真正影响来自稳定运行、权限治理、系统兼容和实际流程改造。"
        )
        industry_sentence = (
            f"从技术扩散看，{variable}会提高对可靠性、成本可控性和系统接入能力的要求，弱化只看单点工具能力的判断。"
        )
        enterprise_sentence = (
            "对学校、企业和服务机构来说，后续差异会体现在工具接入深度、数据治理能力和真实业务场景适配能力。"
        )
        operation_sentence = (
            "落到执行层面，相关投入需要形成清晰的系统边界、流程责任和可量化效率改善。"
        )
        execution_sentence = (
            "一旦进入系统改造、权限配置、数据治理或流程自动化，能力建设就会更接近真实业务需求。"
        )
    elif lens == "risk":
        context_sentence = (
            f"在{context_label}中，{variable}会首先影响责任划分、复核要求、异常处理和合规边界。"
            if chapter_question
            else ""
        )
        depth_sentence = "风险只有在真实流程中反复出现并形成责任分工、处置机制时，才会持续改变岗位和组织安排。"
        subject_sentence = (
            f"{subject}的变化路径更可能表现为先暴露执行风险，再影响权限配置、复核流程和责任承担方式。"
        )
        comparison_sentence = (
            f"{variable}正在影响自动化工具、人工复核和组织责任之间的边界。"
        )
        pathway_sentence = (
            "风险提示只有进入审批、复核、异常处理和责任追踪流程，才会转化为持续约束。"
        )
        industry_sentence = (
            f"从风险约束看，{variable}会提高对可解释性、审计留痕和人工兜底能力的要求。"
        )
        enterprise_sentence = (
            "对学校、企业和专业服务机构来说，后续差异会体现在风险识别训练、复核机制和责任边界设计能力。"
        )
        operation_sentence = (
            "落到执行层面，相关组织需要把风险要求转化为明确流程、权限规则和复核标准。"
        )
        execution_sentence = (
            "一旦进入审计、财务共享、内控或合规流程，风险边界就会直接影响岗位能力要求。"
        )
    else:
        context_sentence = (
            f"对{context_label}而言，{variable}会首先影响课程设置、工具采用、岗位分工或服务交付的先后顺序。"
            if chapter_question
            else ""
        )
        depth_sentence = "只有这些变化进入连续招聘、课程方案、采购项目或业务流程，才会从零散变化沉淀为可比较的产业节奏。"
        subject_sentence = (
            f"{subject}的变化路径更可能表现为先调整工具和训练内容，再改变岗位分工、评价标准和服务交付方式。"
        )
        comparison_sentence = (
            f"{variable}开始同时影响供给端能力建设和需求端筛选标准，课程、工具和岗位要求被拉到同一条能力链上。"
        )
        pathway_sentence = (
            "实际影响会沿着课程、工具、岗位和业务流程逐步展开，不同地区、院校和企业的推进节奏会形成分层差异。"
        )
        industry_sentence = (
            f"从供需关系看，{variable}会提高对稳定交付、合规边界和效率改善的要求，弱化只会工具操作的低阶能力。"
        )
        enterprise_sentence = (
            "对学校、企业和培训机构来说，后续竞争会体现在课程迭代速度、真实案例沉淀和业务场景适配能力。"
        )
        operation_sentence = (
            "落到执行层面，这些变化需要体现为明确预算、岗位描述、课程模块或项目交付。"
        )
        execution_sentence = (
            "一旦进入课程、招聘、采购或流程改造，能力建设就会更接近真实需求。"
        )
        outcome_sentence = "这会把能力训练、工具使用和岗位场景连接得更紧。"
    evidence_sentence = (
        f"材料中“{fact_reference}”这一事实提供了具体切口，可以观察{variable}在公开材料中的实际落点。"
        if fact_reference
        else ""
    )
    risk_sentence = (
        "如果相关动作没有形成连续结果，结论仍应保留为审慎判断。"
    )
    synthesis_subject = subject if subject not in GENERIC_PUBLIC_SUBJECTS else (focus or "这些公开样本")
    synthesis_sentence = f"因此，{synthesis_subject}可以作为观察{variable}扩散速度的早期样本，但还不能直接替代更大范围的统计结论。"
    limit_sentence = (
        f"需要保留的边界是：当前材料能够说明{variable}带来的方向变化，但不同地区、企业规模、院校类型或岗位层级仍可能呈现差异。"
    )
    education_sentence = f"在教育培养与用工衔接中，{variable}会同时影响课程训练、工具熟练度、岗位筛选和业务流程协作。"
    school_sentence = "对学校而言，重点不只是新增一门工具课程，而是把真实业务场景、数据处理训练和风险复核要求放进连续培养链条。"
    enterprise_sentence_public = "对企业而言，招聘要求会更强调能否把财务规则、数据口径和业务系统连接起来，而不是只完成单项核算任务。"
    talent_sentence = f"因此，{subject}的变化会把教育培养、岗位职责和组织数字化进度连在一起，形成更具体的人才能力重排。"
    return [
        mechanism,
        implication,
        subject_sentence,
        anchor_sentence,
        context_sentence,
        depth_sentence,
        comparison_sentence,
        pathway_sentence,
        industry_sentence,
        enterprise_sentence,
        operation_sentence,
        execution_sentence,
        outcome_sentence,
        education_sentence,
        school_sentence,
        enterprise_sentence_public,
        talent_sentence,
    ]


def _expand_to_target(
    *,
    base_parts: Sequence[str],
    lens: str,
    claim_unit: ClaimUnit,
    selected: Sequence[EvidenceFactCard],
    chapter_question: str,
    strength: str,
    force_expand: bool = False,
) -> tuple[str, str]:
    target = _section_target_chars()
    parts = [part for part in base_parts if _text(part)]
    known_facts = [_public_fact_text(card.distilled_fact) for card in selected if _text(card.distilled_fact)]
    status = "base"
    added_expansion = False
    expansion_enabled = force_expand or _expand_to_target_enabled()
    if not expansion_enabled:
        paragraph = _join_public_sentences(parts)
        if _compact_len(paragraph) < target:
            status = "base_no_expand" if paragraph else "insufficient_facts"
        return paragraph, status
    if _compact_len(_join_public_sentences(parts)) < target:
        for sentence in _expansion_sentences(
            lens=lens,
            claim_unit=claim_unit,
            selected=selected,
            chapter_question=chapter_question,
            strength=strength,
        ):
            if sentence:
                if _would_repeat_known_fact(parts, sentence, known_facts):
                    continue
                parts.append(sentence)
                added_expansion = True
            if _compact_len(_join_public_sentences(parts)) >= target:
                status = "expanded"
                break
    paragraph = _join_public_sentences(parts)
    if _compact_len(paragraph) >= target:
        status = "expanded"
    elif len(selected) < 2 and not claim_unit.reasoning_chain:
        status = "insufficient_facts"
    elif added_expansion:
        status = "expanded"
    if force_expand and added_expansion:
        status = "writer_advice_expanded"
    return paragraph, status


def _claim_depth_pack(claim_unit: ClaimUnit) -> Dict[str, Any]:
    pack = claim_unit.raw.get("claim_depth_pack") if isinstance(claim_unit.raw, dict) else {}
    if not isinstance(pack, dict):
        return {}
    if pack.get("diagnostic_only") or pack.get("must_not_render") or pack.get("public_text_allowed") is False:
        return {}
    return pack


def _claim_depth_pack_parts(
    pack: Dict[str, Any],
    *,
    claim_unit: ClaimUnit,
    selected: Sequence[EvidenceFactCard],
) -> List[str]:
    known_facts = [_public_fact_text(card.distilled_fact) for card in selected if card.distilled_fact]
    parts: List[str] = []
    for key in ("judgement", "judgment", "evidence_chain", "mechanism", "segmentation", "implication"):
        text = _clean_analysis_basis_text(pack.get(key), max_chars=520)
        if not text:
            continue
        if key in {"judgement", "judgment"} and claim_unit.claim and _fact_norm(text) == _fact_norm(claim_unit.claim):
            continue
        if _would_repeat_known_fact(parts, text, known_facts):
            continue
        parts.append(text)
    return _dedupe(parts, limit=7)


def _paragraph_from_claim_depth_pack(
    pack: Dict[str, Any],
    *,
    claim_unit: ClaimUnit,
    selected: Sequence[EvidenceFactCard],
    fallback_parts: Sequence[str],
    lens: str,
    chapter_question: str,
    strength: str,
    force_expand: bool = False,
) -> tuple[str, str]:
    parts = [claim_unit.claim] if claim_unit.claim and not _is_snippet_like(claim_unit.claim) else []
    parts.extend(_claim_depth_pack_parts(pack, claim_unit=claim_unit, selected=selected))
    if len(parts) < 3:
        parts.extend(fallback_parts)
    status = "claim_depth_pack"
    known_facts = [_public_fact_text(card.distilled_fact) for card in selected if card.distilled_fact]
    skip_generic_depth_expansion = bool(pack.get("generic_depth_fallback"))
    if (
        (force_expand or _expand_to_target_enabled())
        and not skip_generic_depth_expansion
        and _compact_len(_join_public_sentences(parts)) < _section_target_chars()
    ):
        added_expansion = False
        for sentence in _expansion_sentences(
            lens=lens,
            claim_unit=claim_unit,
            selected=selected,
            chapter_question=chapter_question,
            strength=strength,
        ):
            if not sentence or _would_repeat_known_fact(parts, sentence, known_facts):
                continue
            parts.append(sentence)
            added_expansion = True
            if _compact_len(_join_public_sentences(parts)) >= _section_target_chars():
                break
        paragraph = _join_public_sentences(_dedupe(parts, limit=24))
        if _near_section_target(paragraph):
            status = "claim_depth_pack_writer_advice_expanded" if force_expand else "claim_depth_pack_expanded"
        elif added_expansion:
            status = "claim_depth_pack_writer_advice_partial" if force_expand else "claim_depth_pack_partial"
    return _join_public_sentences(_dedupe(parts, limit=24)), status


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
    if lens == "general" and (claim_unit.claim or claim_unit.reasoning_chain):
        variable_explanation = ""
    refs = _dedupe([card.evidence_id for card in selected if card.evidence_id], limit=4)
    clean_basis = [_clean_analysis_basis_text(item) for item in claim_unit.evidence_basis]
    facts = _dedupe([*(_public_fact_text(card.distilled_fact) for card in selected), *[item for item in clean_basis if item]], limit=6)
    strength = (claim_unit.claim_strength or selected[0].claim_strength_hint or "").lower()
    has_strong_source = any(card.source_level.upper() in {"A", "B"} for card in selected)
    if strength in {"strong", "moderate", "decision_ready", "core_claim"} or has_strong_source:
        status = "composed"
    else:
        status = "composed_directional"
    synthetic_basis_cards = bool(selected) and all(
        bool(getattr(card, "raw", {}).get("synthetic_from_unit_basis")) for card in selected
    )
    analysis_parts = _claim_analysis_parts(
        claim_unit,
        known_facts=[] if synthetic_basis_cards else [_public_fact_text(card.distilled_fact) for card in selected if card.distilled_fact],
    )
    has_public_analysis = bool(
        _text(claim_unit.paragraph_seed)
        or _text(claim_unit.reasoning_chain)
        or _as_list((claim_unit.raw or {}).get("narrative_supporting_claims") if isinstance(claim_unit.raw, dict) else [])
    )
    base_parts = [*analysis_parts]
    if not has_public_analysis:
        base_parts.extend(sentences[:2])
    depth_pack = _claim_depth_pack(claim_unit)
    writer_advice_expand = _writer_advice_requires_expansion(claim_unit)
    if depth_pack:
        paragraph, expansion_status = _paragraph_from_claim_depth_pack(
            depth_pack,
            claim_unit=claim_unit,
            selected=selected,
            fallback_parts=base_parts,
            lens=lens,
            chapter_question=chapter_question,
            strength=strength,
            force_expand=writer_advice_expand,
        )
    else:
        paragraph, expansion_status = _expand_to_target(
            base_parts=base_parts,
            lens=lens,
            claim_unit=claim_unit,
            selected=selected,
            chapter_question=chapter_question,
            strength=strength,
            force_expand=writer_advice_expand,
        )
    if claim_unit.claim and not _is_snippet_like(claim_unit.claim):
        claim = claim_unit.claim
    else:
        claim = sentences[0]
    mechanism = claim_unit.reasoning_chain if claim_unit.reasoning_chain and not _is_snippet_like(claim_unit.reasoning_chain) else paragraph
    # limitation_boundary is an internal analysis field (diagnostic caveats such as
    # "这一判断的价值在于… / 需交叉验证"). It must not be spliced into the public body
    # counter_evidence — only a real risk-lens fact or a generated boundary sentence may.
    boundary = ""
    if lens == "risk":
        boundary = sentences[0]
    claim = remove_hard_industry_templates(claim)
    mechanism = remove_hard_industry_templates(mechanism)
    boundary = remove_hard_industry_templates(boundary)
    variable_explanation = remove_hard_industry_templates(variable_explanation)
    paragraph = remove_hard_industry_templates(paragraph)
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
        "evidence_use_level": claim_unit.evidence_use_level,
        "writing_permission": claim_unit.writing_permission,
        "metric_completeness_status": claim_unit.metric_completeness_status,
        "metric_missing_fields": list(claim_unit.metric_missing_fields),
        "omit_reason": "",
    }
