from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, Iterable, List, Sequence


def as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _first_text(*values: Any) -> str:
    for value in values:
        text = _text(value)
        if text:
            return text
    return ""


def _refs_from_values(values: Iterable[Any]) -> List[str]:
    refs: List[str] = []
    seen = set()
    for value in values:
        if isinstance(value, dict):
            candidates: Sequence[Any] = (
                value.get("evidence_id"),
                value.get("ref"),
                value.get("source_ref"),
                value.get("citation_ref"),
            )
        elif isinstance(value, list):
            candidates = value
        else:
            candidates = (value,)
        for candidate in candidates:
            text = _text(candidate)
            if not text or text in seen:
                continue
            seen.add(text)
            refs.append(text)
    return refs


def normalize_evidence_refs(payload: Dict[str, Any]) -> List[str]:
    values: List[Any] = []
    for key in (
        "used_evidence_ids",
        "used_fact_refs",
        "citation_refs",
        "evidence_refs",
        "supporting_evidence_refs",
        "supporting_evidence",
        "required_evidence_refs",
        "source_refs",
    ):
        value = payload.get(key)
        if isinstance(value, list):
            values.extend(value)
        elif value not in (None, ""):
            values.append(value)
    return _refs_from_values(values)


def normalize_requirement_ids(payload: Dict[str, Any]) -> List[str]:
    values: List[Any] = []
    for key in (
        "requirement_ids",
        "requirement_id",
        "evidence_requirement_ids",
        "evidence_requirement_id",
        "required_slot_ids",
        "slot_id",
    ):
        value = as_dict(payload.get("lineage")).get(key) if key in as_dict(payload.get("lineage")) else payload.get(key)
        if isinstance(value, list):
            values.extend(value)
        elif value not in (None, ""):
            values.append(value)
    return _refs_from_values(values)


def _first_requirement_id(payload: Dict[str, Any]) -> str:
    ids = normalize_requirement_ids(payload)
    return ids[0] if ids else ""


def _lineage_list(payload: Dict[str, Any], key: str) -> List[str]:
    lineage = as_dict(payload.get("lineage"))
    value = lineage.get(key)
    if isinstance(value, list):
        return _refs_from_values(value)
    if value not in (None, ""):
        return _refs_from_values([value])
    return []


def _source_identity_refs(source: Dict[str, Any]) -> List[str]:
    refs: List[Any] = [
        source.get("ref"),
        source.get("id"),
        source.get("evidence_id"),
        source.get("source_ref"),
        source.get("citation_ref"),
        source.get("document_id"),
        source.get("doc_id"),
        source.get("url"),
        source.get("source_url"),
    ]
    for key in ("evidence_refs", "used_fact_refs", "source_refs", "refs"):
        refs.extend(as_list(source.get(key)))
    return _refs_from_values(refs)


def build_source_ref_lookup(source_registry: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for index, source in enumerate(source_registry or [], start=1):
        if not isinstance(source, dict):
            continue
        public_ref = _first_text(source.get("ref"), source.get("source_ref"), source.get("evidence_id"), f"[{index}]")
        payload = {"source_ref": public_ref, "source": source}
        for ref in _source_identity_refs(source):
            lookup.setdefault(ref, payload)
    return lookup


def resolve_evidence_source_ref(ref: Any, source_registry: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    text = _text(ref)
    if not text:
        return {"resolved": False, "input_ref": "", "source_ref": "", "reason": "empty_ref"}
    lookup = build_source_ref_lookup(source_registry)
    match = lookup.get(text)
    if match:
        return {
            "resolved": True,
            "input_ref": text,
            "source_ref": match.get("source_ref") or text,
            "reason": "matched_source_registry",
            "source": match.get("source") or {},
        }
    return {"resolved": False, "input_ref": text, "source_ref": "", "reason": "unresolved_ref"}


def filter_resolvable_evidence_refs(refs: Sequence[Any], source_registry: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    resolved_refs: List[str] = []
    filtered_refs: List[Dict[str, Any]] = []
    seen = set()
    for ref in refs or []:
        result = resolve_evidence_source_ref(ref, source_registry)
        if result.get("resolved"):
            source_ref = _text(result.get("source_ref"))
            if source_ref and source_ref not in seen:
                seen.add(source_ref)
                resolved_refs.append(source_ref)
        else:
            filtered_refs.append({"ref": _text(ref), "reason": result.get("reason") or "unresolved_ref"})
    return {
        "resolved_refs": resolved_refs,
        "filtered_refs": filtered_refs,
        "filtered_unresolved_ref_count": len(filtered_refs),
    }


FACTUAL_CLAIM_RE = re.compile(
    r"(?:"
    r"\d{4}\s*年|"
    r"\d+(?:\.\d+)?\s*(?:%|亿|万亿|万|条|家|美元|元|人民币|美元|CAGR)|"
    r"CAGR|market\s+size|forecast|revenue|funding|"
    r"市场规模|融资|收入|营收|预测|白皮书|报告|发布|财报|公告|政策|补贴|"
    r"OpenAI|Microsoft|Google|Salesforce|AWS|Anthropic|Gartner|IDC|"
    r"阿里|百度|腾讯|华为|字节|九科信息|机构|企业"
    r")",
    re.I,
)

NON_FACTUAL_TRANSITION_RE = re.compile(
    r"^(?:因此|同时|总体来看|换言之|从这个角度看|这意味着|进一步看|由此看)[^。.!?？]{0,80}[。.!?？]?$"
)

# Sentences that describe HOW the chapter should be analyzed rather than
# asserting a fact. They often mention facts as examples (so they trip
# ``FACTUAL_CLAIM_RE``) but contain no testable claim that would require its
# own citation. Treating them as ``non-factual`` keeps the citationless-fact
# auditor from flagging framing/process language as a publication blocker.
NON_FACTUAL_FRAMING_RE = re.compile(
    r"(?:的判断应以.{0,40}为事实锚点"
    r"|确认本章的事实起点"
    r"|的关键不在政策表态本身"
    r"|需要按连续指标和反向样本拆解"
    r"|避免把单点信号直接外推"
    r"|目前更适合作为背景条件"
    r"|目前只有线索或背景材料.{0,20}尚不足以支撑强结论"
    r"|结论强度取决于后续连续指标"
    r"|分析先看已经出现的产业信号"
    r"|这些事实来自不同类型来源且方向一致时"
    r"|结论会保留边界"
    r"|结论强度取决于2026"
    r"|后续重点跟踪同口径指标"
    r"|后续重点跟踪适合写成带适用边界的正文判断"
    r"|围绕.{0,10}的(?:判断|分析|结论)需要"
    r"|落到行业含义上.{0,20}更有价值的观察顺序"
    r"|这一信息是局部样本还是可迁移趋势"
    # Methodology / boundary / risk-list paragraphs that describe how to
    # interpret evidence rather than assert a specific claim.
    r"|样本.{0,10}时间窗口.{0,10}来源口径可能改变结论方向"
    r"|这个判断的边界主要来自.{0,20}变量"
    r"|时间边界.{0,40}口径边界.{0,40}反向样本"
    r"|边界条件包括"
    r"|没有反证并不等于风险不存在"
    r"|反证缺位本身应作为结论边界"
    r"|结论强度取决于"
    r"|价格.{0,10}库存.{0,10}订单"
    r"|客户认证.{0,10}采购节奏"
    r"|尚不足以支撑强结论)"
)


def _iter_public_text_values(payload: Dict[str, Any]) -> Iterable[str]:
    for key in (
        "claim",
        "judgment",
        "reasoning",
        "mechanism",
        "counter_evidence",
        "decision_implication",
        "section_title",
        "chapter_title",
    ):
        text = _text(payload.get(key))
        if text:
            yield text
    for key in ("supporting_facts", "evidence_basis", "fact_cards", "facts"):
        for item in as_list(payload.get(key)):
            if isinstance(item, dict):
                text = " ".join(
                    _text(item.get(field))
                    for field in (
                        "distilled_fact",
                        "public_fact",
                        "fact",
                        "source_title",
                        "title",
                        "value",
                        "unit",
                        "period",
                        "time_or_scope",
                    )
                    if _text(item.get(field))
                )
                if text:
                    yield text
            else:
                text = _text(item)
                if text:
                    yield text
    for block in as_list(payload.get("render_blocks")):
        if isinstance(block, dict):
            text = _text(block.get("text") or block.get("paragraph"))
            if text:
                yield text


def text_has_factual_claim(text: Any) -> bool:
    value = _text(text)
    if not value:
        return False
    if NON_FACTUAL_TRANSITION_RE.match(value):
        return False
    if NON_FACTUAL_FRAMING_RE.search(value):
        return False
    return bool(FACTUAL_CLAIM_RE.search(value))


def section_has_factual_claim(section: Dict[str, Any]) -> bool:
    payload = as_dict(section)
    if payload.get("non_factual_transition"):
        return False
    return any(text_has_factual_claim(value) for value in _iter_public_text_values(payload))


def resolve_section_source_refs(section: Dict[str, Any], source_registry: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    payload = as_dict(section)
    refs = normalize_evidence_refs(payload)
    result = filter_resolvable_evidence_refs(refs, source_registry)
    factual = section_has_factual_claim(payload)
    resolved_refs = list(result.get("resolved_refs") or [])
    filtered_refs = list(result.get("filtered_refs") or [])
    status = "ok"
    reason = ""
    if factual and not resolved_refs:
        status = "blocked"
        reason = "factual_section_without_resolved_ref"
    elif filtered_refs:
        status = "warning"
        reason = "some_refs_unresolved"
    return {
        "status": status,
        "reason": reason,
        "has_factual_claim": factual,
        "has_resolved_ref": bool(resolved_refs),
        "input_refs": refs,
        "resolved_refs": resolved_refs,
        "filtered_refs": filtered_refs,
        "filtered_unresolved_ref_count": len(filtered_refs),
    }


def normalize_chapter_id(payload: Dict[str, Any]) -> str:
    return _first_text(
        payload.get("chapter_id"),
        payload.get("dimension_id"),
        payload.get("hypothesis_id"),
        payload.get("chapter_ref"),
    )


@dataclass(frozen=True)
class EvidenceFactCard:
    evidence_id: str = ""
    chapter_id: str = ""
    hypothesis_id: str = ""
    requirement_id: str = ""
    subject: str = ""
    action_or_signal: str = ""
    variable: str = ""
    value: str = ""
    unit: str = ""
    time_or_scope: str = ""
    distilled_fact: str = ""
    fact_type: str = ""
    source_ref: str = ""
    source_level: str = ""
    verification_status: str = ""
    proof_role: str = ""
    analysis_role: str = ""
    analysis_eligible: bool = False
    allowed_use: str = ""
    block_affinity: List[str] = field(default_factory=list)
    claim_strength_hint: str = ""
    source_id: str = ""
    search_task_id: str = ""
    lineage: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_legacy_dict(cls, payload: Dict[str, Any]) -> "EvidenceFactCard":
        payload = as_dict(payload)
        quality = as_dict(payload.get("public_fact_quality"))
        nested_card = (
            as_dict(payload.get("public_fact_card"))
            or as_dict(quality.get("public_fact_card"))
            or as_dict(as_dict(payload.get("evidence_card")).get("public_fact_card"))
        )
        merged = {**payload, **nested_card}
        source_ref = _first_text(
            merged.get("source_ref"),
            merged.get("citation_ref"),
            merged.get("ref"),
        )
        if not source_ref and "source_id" in merged:
            source_ref = _text(merged.get("source_id"))
        search_task = as_dict(merged.get("search_task"))
        requirement_id = _first_requirement_id(merged)
        hypothesis_id = _first_text(merged.get("hypothesis_id"), search_task.get("hypothesis_id"))
        evidence_id = _first_text(merged.get("evidence_id"), merged.get("id"), merged.get("ref"))
        chapter_id = normalize_chapter_id(merged)
        source_id = _first_text(merged.get("source_id"), merged.get("source_ref"), merged.get("citation_ref"), source_ref)
        search_task_id = _first_text(merged.get("search_task_id"), search_task.get("task_id"), search_task.get("id"))
        lineage = {
            **as_dict(merged.get("lineage")),
            "chapter_id": chapter_id,
            "hypothesis_id": hypothesis_id,
            "requirement_id": requirement_id,
            "fact_id": evidence_id,
            "source_id": source_id,
            "search_task_id": search_task_id,
        }
        lineage = {key: value for key, value in lineage.items() if value not in (None, "", [])}
        affinity_raw = merged.get("block_affinity")
        if isinstance(affinity_raw, str):
            affinity = [affinity_raw] if affinity_raw.strip() else []
        else:
            affinity = [_text(item) for item in as_list(affinity_raw) if _text(item)]
        return cls(
            evidence_id=evidence_id,
            chapter_id=chapter_id,
            hypothesis_id=hypothesis_id,
            requirement_id=requirement_id,
            subject=_first_text(merged.get("subject"), merged.get("company"), merged.get("entity")),
            action_or_signal=_first_text(merged.get("action_or_signal"), merged.get("action"), merged.get("signal")),
            variable=_first_text(merged.get("variable"), merged.get("analysis_variable"), merged.get("metric"), merged.get("indicator")),
            value=_first_text(merged.get("value"), merged.get("display_value"), merged.get("numeric_value")),
            unit=_first_text(merged.get("unit"), merged.get("numeric_unit")),
            time_or_scope=_first_text(merged.get("time_or_scope"), merged.get("period"), merged.get("scope"), merged.get("date")),
            distilled_fact=_first_text(
                merged.get("distilled_fact"),
                merged.get("public_fact"),
                merged.get("clean_fact"),
                merged.get("fact"),
                merged.get("content"),
                merged.get("summary"),
            ),
            fact_type=_first_text(merged.get("fact_type"), quality.get("fact_type"), merged.get("proof_role")),
            source_ref=source_ref,
            source_level=_first_text(merged.get("source_level"), merged.get("credibility")),
            verification_status=_first_text(merged.get("source_verification_status"), merged.get("verification_status")),
            proof_role=_first_text(merged.get("proof_role"), merged.get("evidence_goal"), merged.get("role")),
            analysis_role=_first_text(merged.get("analysis_role"), quality.get("analysis_role")),
            analysis_eligible=bool(merged.get("analysis_eligible") if "analysis_eligible" in merged else quality.get("analysis_eligible")),
            allowed_use=_first_text(merged.get("allowed_use"), quality.get("allowed_use")),
            block_affinity=affinity,
            claim_strength_hint=_first_text(merged.get("claim_strength_hint"), merged.get("claim_strength")),
            source_id=source_id,
            search_task_id=search_task_id,
            lineage=lineage,
            raw=payload,
            diagnostics={"eligible_for_report": quality.get("eligible_for_report")},
        )

    @property
    def is_valid_for_report(self) -> bool:
        return bool(self.evidence_id and self.chapter_id and self.distilled_fact)

    def to_legacy_dict(self) -> Dict[str, Any]:
        card = {
            "subject": self.subject,
            "action_or_signal": self.action_or_signal,
            "variable": self.variable,
            "value": self.value,
            "unit": self.unit,
            "time_or_scope": self.time_or_scope,
            "distilled_fact": self.distilled_fact,
            "fact_type": self.fact_type,
            "source_ref": self.source_ref,
            "source_level": self.source_level,
            "source_verification_status": self.verification_status,
            "proof_role": self.proof_role,
            "block_affinity": list(self.block_affinity),
            "claim_strength_hint": self.claim_strength_hint,
        }
        return {
            **self.raw,
            "evidence_id": self.evidence_id,
            "chapter_id": self.chapter_id,
            "hypothesis_id": self.hypothesis_id,
            "requirement_id": self.requirement_id,
            "analysis_role": self.analysis_role,
            "analysis_eligible": self.analysis_eligible,
            "allowed_use": self.allowed_use,
            "source_id": self.source_id,
            "search_task_id": self.search_task_id,
            "lineage": dict(self.lineage),
            "public_fact_card": card,
            "public_fact_quality": {"eligible_for_report": self.is_valid_for_report, "public_fact_card": card},
        }


@dataclass(frozen=True)
class ClaimUnit:
    claim_id: str = ""
    chapter_id: str = ""
    hypothesis_id: str = ""
    requirement_ids: List[str] = field(default_factory=list)
    claim: str = ""
    evidence_refs: List[str] = field(default_factory=list)
    evidence_basis: List[str] = field(default_factory=list)
    reasoning_chain: str = ""
    limitation_boundary: str = ""
    claim_strength: str = "directional"
    claim_strength_ceiling: str = ""
    analysis_role: str = ""
    source_support_map: Dict[str, List[str]] = field(default_factory=dict)
    paragraph_seed: str = ""
    block_type: str = ""
    section_id: str = ""
    lineage: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_legacy_dict(cls, payload: Dict[str, Any]) -> "ClaimUnit":
        payload = as_dict(payload)
        basis = payload.get("evidence_basis") or payload.get("supporting_facts") or payload.get("fact_chain") or []
        evidence_refs = normalize_evidence_refs(payload)
        requirement_ids = normalize_requirement_ids(payload)
        lineage = {
            **as_dict(payload.get("lineage")),
            "chapter_id": normalize_chapter_id(payload),
            "hypothesis_id": _first_text(payload.get("hypothesis_id"), as_dict(payload.get("lineage")).get("hypothesis_id")),
            "requirement_ids": requirement_ids or _lineage_list(payload, "requirement_ids"),
            "fact_ids": _lineage_list(payload, "fact_ids") or evidence_refs,
            "source_ids": _lineage_list(payload, "source_ids"),
            "search_task_ids": _lineage_list(payload, "search_task_ids"),
        }
        lineage = {key: value for key, value in lineage.items() if value not in (None, "", [])}
        return cls(
            claim_id=_first_text(payload.get("claim_id"), payload.get("id")),
            chapter_id=normalize_chapter_id(payload),
            hypothesis_id=_first_text(payload.get("hypothesis_id"), as_dict(payload.get("lineage")).get("hypothesis_id")),
            requirement_ids=requirement_ids,
            claim=_first_text(payload.get("claim"), payload.get("judgment"), payload.get("takeaway"), payload.get("core_answer")),
            evidence_refs=evidence_refs,
            evidence_basis=[_text(item) for item in as_list(basis) if _text(item)],
            reasoning_chain=_first_text(payload.get("reasoning_chain"), payload.get("reasoning"), payload.get("mechanism")),
            limitation_boundary=_first_text(payload.get("limitation_boundary"), payload.get("counter_evidence"), payload.get("boundary")),
            claim_strength=_first_text(payload.get("claim_strength"), payload.get("strength"), payload.get("claim_status")) or "directional",
            claim_strength_ceiling=_first_text(payload.get("claim_strength_ceiling"), as_dict(payload.get("lineage")).get("claim_strength_ceiling")),
            analysis_role=_first_text(payload.get("analysis_role"), payload.get("allowed_use"), payload.get("proof_role")),
            source_support_map={
                key: [ref for ref in normalize_evidence_refs({"evidence_refs": value}) if ref]
                for key, value in as_dict(payload.get("source_support_map")).items()
            },
            paragraph_seed=_first_text(payload.get("paragraph_seed")),
            block_type=_first_text(payload.get("block_type"), payload.get("layout_section_role"), payload.get("output_type")),
            section_id=_first_text(payload.get("section_id"), payload.get("id")),
            lineage=lineage,
            raw=payload,
        )

    def to_legacy_dict(self) -> Dict[str, Any]:
        return {
            **self.raw,
            "claim_id": self.claim_id,
            "chapter_id": self.chapter_id,
            "hypothesis_id": self.hypothesis_id,
            "requirement_ids": list(self.requirement_ids),
            "claim": self.claim,
            "evidence_refs": list(self.evidence_refs),
            "used_fact_refs": list(self.evidence_refs),
            "supporting_evidence_refs": list(self.evidence_refs),
            "evidence_basis": list(self.evidence_basis),
            "reasoning": self.reasoning_chain,
            "mechanism": self.reasoning_chain,
            "counter_evidence": self.limitation_boundary,
            "claim_strength": self.claim_strength,
            "claim_strength_ceiling": self.claim_strength_ceiling,
            "analysis_role": self.analysis_role,
            "source_support_map": {key: list(value) for key, value in self.source_support_map.items()},
            "paragraph_seed": self.paragraph_seed,
            "block_type": self.block_type,
            "section_id": self.section_id,
            "lineage": dict(self.lineage),
        }


@dataclass(frozen=True)
class ChapterInsight:
    chapter_id: str = ""
    chapter_question: str = ""
    claim_units: List[ClaimUnit] = field(default_factory=list)
    fact_chain: List[str] = field(default_factory=list)
    evidence_health: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_legacy_dict(cls, payload: Dict[str, Any]) -> "ChapterInsight":
        payload = as_dict(payload)
        raw_units = as_list(payload.get("claim_units")) or as_list(payload.get("key_claims")) or as_list(payload.get("key_judgments"))
        return cls(
            chapter_id=normalize_chapter_id(payload),
            chapter_question=_first_text(payload.get("chapter_question"), payload.get("question"), payload.get("chapter_title")),
            claim_units=[ClaimUnit.from_legacy_dict(item) for item in raw_units if isinstance(item, dict)],
            fact_chain=[_text(item) for item in as_list(payload.get("fact_chain")) if _text(item)],
            evidence_health=as_dict(payload.get("evidence_health")),
            raw=payload,
        )

    def to_legacy_dict(self) -> Dict[str, Any]:
        return {
            **self.raw,
            "chapter_id": self.chapter_id,
            "chapter_question": self.chapter_question,
            "claim_units": [unit.to_legacy_dict() for unit in self.claim_units],
            "key_claims": [unit.to_legacy_dict() for unit in self.claim_units],
            "fact_chain": list(self.fact_chain),
            "evidence_health": dict(self.evidence_health),
        }


@dataclass(frozen=True)
class ReportSection:
    section_id: str = ""
    chapter_id: str = ""
    block_type: str = ""
    section_title: str = ""
    claim: str = ""
    paragraph: str = ""
    evidence_refs: List[str] = field(default_factory=list)
    supporting_facts: List[str] = field(default_factory=list)
    evidence_backed: bool = False
    composition_status: str = "legacy"
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_legacy_dict(cls, payload: Dict[str, Any]) -> "ReportSection":
        payload = as_dict(payload)
        return cls(
            section_id=_first_text(payload.get("section_id"), payload.get("id")),
            chapter_id=normalize_chapter_id(payload),
            block_type=_first_text(payload.get("block_type"), payload.get("output_type"), payload.get("section_role")),
            section_title=_first_text(payload.get("section_title"), payload.get("title")),
            claim=_first_text(payload.get("claim"), payload.get("judgment")),
            paragraph=_first_text(payload.get("composed_paragraph"), payload.get("paragraph")),
            evidence_refs=normalize_evidence_refs(payload),
            supporting_facts=[_text(item) for item in as_list(payload.get("supporting_facts")) if _text(item)],
            evidence_backed=bool(payload.get("evidence_backed")),
            composition_status=_first_text(payload.get("composition_status")) or "legacy",
            raw=payload,
        )

    def to_legacy_dict(self) -> Dict[str, Any]:
        return {
            **self.raw,
            "section_id": self.section_id,
            "chapter_id": self.chapter_id,
            "block_type": self.block_type,
            "section_title": self.section_title,
            "claim": self.claim,
            "composed_paragraph": self.paragraph,
            "evidence_refs": list(self.evidence_refs),
            "used_fact_refs": list(self.evidence_refs),
            "supporting_facts": list(self.supporting_facts),
            "evidence_backed": self.evidence_backed,
            "composition_status": self.composition_status,
        }
