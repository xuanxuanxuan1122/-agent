from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence


GENERIC_ENTITY_TOKENS = {
    "ai",
    "api",
    "agent",
    "agents",
    "enterprise",
    "enterprises",
    "workflow",
    "workflows",
    "market",
    "markets",
    "price",
    "prices",
    "pricing",
    "cost",
    "costs",
    "revenue",
    "roi",
    "cloud",
    "model",
    "models",
    "data",
    "report",
    "reports",
    "source",
    "sources",
}

GENERIC_CHINESE_ANCHORS = {
    "企业级",
    "相关",
    "数据",
    "报告",
    "行业",
    "市场",
    "来源",
    "统计",
    "部门",
    "官网",
    "官方",
    "通过",
    "发布",
    "说明",
    "正在",
    "开始",
    "指出",
}

CHINESE_MATERIAL_PHRASES = (
    "竞争格局",
    "场景落地",
    "渠道生态",
    "厂商分化",
    "交付能力",
    "部署模式",
    "客户案例",
    "工作流部署",
    "流程自动化",
    "商业化",
    "付费能力",
    "单位经济",
    "风险边界",
    "反向证据",
    "安全合规",
    "投资优先级",
    "市场规模",
    "渗透率",
    "增长率",
    "使用率",
    "采购",
    "中标",
)

NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:\d+(?:\.\d+)?\s*/\s*\d+(?:\.\d+)?|\d+(?:\.\d+)?\s*(?:%|亿元|亿美元|万|亿|倍|x|X)?)(?![A-Za-z0-9])"
)
ENGLISH_TOKEN_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:[.-][A-Za-z0-9]+)*\b")
CHINESE_RUN_RE = re.compile(r"[\u4e00-\u9fff]{2,}")
CHINESE_NUMERAL_CHARS = "\u96f6\u4e00\u4e8c\u4e24\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u767e\u70b9"
CHINESE_PERCENT_RE = re.compile("\u767e\u5206\u4e4b([" + CHINESE_NUMERAL_CHARS + r"\d.]+)")
CHINESE_DIGIT_VALUES = {
    "\u96f6": 0,
    "\u4e00": 1,
    "\u4e8c": 2,
    "\u4e24": 2,
    "\u4e09": 3,
    "\u56db": 4,
    "\u4e94": 5,
    "\u516d": 6,
    "\u4e03": 7,
    "\u516b": 8,
    "\u4e5d": 9,
}

# Override the legacy pattern above: the old unit suffix list contained mojibake
# and downstream support checks must compare exact numeric tokens, not substrings.
NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?:"
    r"\d+(?:\.\d+)?\s*/\s*\d+(?:\.\d+)?"
    r"|"
    r"\d+(?:\.\d+)?\s*"
    r"(?:%|个百分点|个点|倍|x|X|万元|亿元|万|亿|美元|人民币|元|家|个|人|次|GB|TB|tokens?)?"
    r")"
    r"(?![A-Za-z0-9])"
)


@dataclass(frozen=True)
class ClaimSupportResult:
    status: str
    unsupported_terms: List[str] = field(default_factory=list)
    unsupported_numbers: List[str] = field(default_factory=list)
    unsupported_entities: List[str] = field(default_factory=list)

    @property
    def supported(self) -> bool:
        return self.status == "supported"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "unsupported_terms": list(self.unsupported_terms),
            "unsupported_numbers": list(self.unsupported_numbers),
            "unsupported_entities": list(self.unsupported_entities),
        }


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


def _norm(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).lower()


def _compact_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _dedupe(items: Iterable[str]) -> List[str]:
    result: List[str] = []
    seen = set()
    for item in items:
        value = str(item or "").strip()
        key = value.lower()
        if not value or key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _format_number(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return f"{value:.12g}"


def _parse_chinese_integer(value: str) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text == "\u5341":
        return 10
    if "\u767e" in text:
        left, _, right = text.partition("\u767e")
        hundreds = CHINESE_DIGIT_VALUES.get(left, 1 if not left else None)
        if hundreds is None:
            return None
        tail = _parse_chinese_integer(right) if right else 0
        return hundreds * 100 + (tail or 0)
    if "\u5341" in text:
        left, _, right = text.partition("\u5341")
        tens = CHINESE_DIGIT_VALUES.get(left, 1 if not left else None)
        ones = CHINESE_DIGIT_VALUES.get(right, 0 if not right else None)
        if tens is None or ones is None:
            return None
        return tens * 10 + ones
    if len(text) == 1:
        return CHINESE_DIGIT_VALUES.get(text)
    total = 0
    for char in text:
        digit = CHINESE_DIGIT_VALUES.get(char)
        if digit is None:
            return None
        total = total * 10 + digit
    return total


def _parse_chinese_number(value: str) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return float(text)
    integer_text, dot, decimal_text = text.partition("\u70b9")
    integer = _parse_chinese_integer(integer_text)
    if integer is None:
        return None
    if not dot:
        return float(integer)
    decimals: List[str] = []
    for char in decimal_text:
        digit = CHINESE_DIGIT_VALUES.get(char)
        if digit is None:
            return None
        decimals.append(str(digit))
    if not decimals:
        return float(integer)
    return float(f"{integer}.{''.join(decimals)}")


def _plain_numeric_value(value: str) -> float | None:
    token = _norm(value).replace(",", "")
    if token.endswith("%"):
        token = token[:-1]
    if not re.fullmatch(r"\d+(?:\.\d+)?", token):
        return None
    return float(token)


def _number_variants(value: str) -> set[str]:
    token = _norm(value).replace(",", "")
    variants = {token} if token else set()
    if "/" in token:
        return variants
    numeric = _plain_numeric_value(token)
    if numeric is None:
        return variants
    if token.endswith("%"):
        variants.add(f"{_format_number(numeric)}%")
        variants.add(_format_number(numeric / 100))
        return variants
    variants.add(_format_number(numeric))
    if 0 < numeric <= 1:
        variants.add(f"{_format_number(numeric * 100)}%")
    return variants


def _claim_numbers(claim: str) -> List[str]:
    values = []
    for match in CHINESE_PERCENT_RE.finditer(claim):
        parsed = _parse_chinese_number(str(match.group(1) or "").rstrip(".;\u3002\uff1b,\uff0c"))
        if parsed is not None:
            values.append(f"{_format_number(parsed)}%")
    for match in NUMBER_RE.finditer(claim):
        value = re.sub(r"\s+", "", match.group(0))
        if not value:
            continue
        # Years alone are often scope markers; keep them, but ignore tiny single
        # ordinals that usually describe prose structure rather than evidence.
        if re.fullmatch(r"\d", value):
            continue
        values.append(value)
    return _dedupe(values)


def _number_token_set(text: str) -> set[str]:
    tokens: set[str] = set()
    for number in _claim_numbers(_compact_text(text)):
        tokens.update(_number_variants(number))
    return tokens


def claim_has_numeric_terms(claim: Any) -> bool:
    return bool(_claim_numbers(_compact_text(claim)))


def _material_english_entities(claim: str) -> List[str]:
    entities: List[str] = []
    tokens = ENGLISH_TOKEN_RE.findall(claim)
    for index, token in enumerate(tokens):
        lowered = token.lower().strip(".")
        if lowered in GENERIC_ENTITY_TOKENS:
            continue
        if len(lowered) < 3:
            continue
        has_case_signal = any(ch.isupper() for ch in token[1:])
        has_digit = any(ch.isdigit() for ch in token)
        has_separator = "." in token or "-" in token
        is_all_caps = token.isupper() and len(token) > 3
        # Avoid treating the first ordinary capitalized word in an English
        # sentence as an entity. Brands/products normally carry internal caps,
        # digits, separators, or all-caps form.
        if has_case_signal or has_digit or has_separator or is_all_caps:
            entities.append(token.strip("."))
            continue
        if index > 0 and token[:1].isupper():
            entities.append(token.strip("."))
    return _dedupe(entities)


def _material_chinese_anchors(claim: str) -> List[str]:
    text = _compact_text(claim)
    anchors: List[str] = []
    for phrase in CHINESE_MATERIAL_PHRASES:
        if phrase in text:
            anchors.append(phrase)
    phrase_anchors = _dedupe(anchors)
    if len(phrase_anchors) >= 2:
        return phrase_anchors[:8]
    for run in CHINESE_RUN_RE.findall(text):
        if len(run) < 4:
            continue
        for size in (6, 5, 4):
            for index in range(0, max(0, len(run) - size + 1)):
                token = run[index : index + size]
                if token in GENERIC_CHINESE_ANCHORS:
                    continue
                if any(generic in token for generic in GENERIC_CHINESE_ANCHORS) and size <= 4:
                    continue
                anchors.append(token)
    # Keep phrase anchors first and cap the fallback ngrams to avoid noisy
    # unsupported-term payloads. At least two material anchors are enough to
    # catch off-topic qualitative claims without turning this into full NLP.
    return _dedupe(anchors)[:8]


def _unsupported_chinese_anchors(claim: str, evidence_text: str) -> List[str]:
    anchors = _material_chinese_anchors(claim)
    if not anchors:
        return []
    evidence_key = _norm(evidence_text)
    unsupported = [anchor for anchor in anchors if _norm(anchor) not in evidence_key]
    if len(unsupported) >= 2:
        return unsupported
    return []


def _evidence_text(card: Dict[str, Any]) -> str:
    parts: List[str] = []
    public_card = _as_dict(card.get("public_fact_card"))
    source = _as_dict(card.get("source"))
    for payload in (card, public_card, source):
        for key in (
            "distilled_fact",
            "fact",
            "clean_fact",
            "content",
            "summary",
            "title",
            "source_title",
            "metric",
            "indicator",
            "value",
            "unit",
            "period",
            "time_or_scope",
            "source_url",
            "url",
        ):
            value = payload.get(key)
            if value not in (None, ""):
                parts.append(_compact_text(value))
    for key in ("supporting_facts", "evidence_basis"):
        parts.extend(_compact_text(item) for item in _as_list(card.get(key)) if str(item or "").strip())
    return " ".join(part for part in parts if part)


def _is_metric_card(card: Dict[str, Any]) -> bool:
    fact_type = str(card.get("fact_type") or card.get("proof_role") or "").strip().lower()
    if fact_type == "metric" or "metric" in fact_type:
        return True
    return bool(str(card.get("metric") or card.get("indicator") or "").strip())


def _metric_missing_fields(card: Dict[str, Any]) -> List[str]:
    missing: List[str] = []
    if not str(card.get("metric") or card.get("indicator") or "").strip():
        missing.append("metric")
    if not str(card.get("value") or card.get("display_value") or card.get("numeric_value") or "").strip():
        missing.append("value")
    value_text = str(card.get("value") or card.get("display_value") or card.get("numeric_value") or "").strip().lower()
    value_carries_unit = bool(
        value_text
        and (
            "%"
            in value_text
            or "percent" in value_text
            or "percentage point" in value_text
            or "百分点" in value_text
            or "百分比" in value_text
        )
    )
    if not str(card.get("unit") or card.get("numeric_unit") or "").strip() and not value_carries_unit:
        missing.append("unit")
    if not (
        str(card.get("period") or "").strip()
        or str(card.get("scope") or "").strip()
        or str(card.get("time_or_scope") or "").strip()
        or str(card.get("date") or "").strip()
    ):
        missing.append("period")
    if not (
        str(card.get("source_url") or "").strip()
        or str(card.get("source_ref") or "").strip()
        or str(card.get("source_title") or "").strip()
        or str(card.get("source") or "").strip()
        or str(_as_dict(card.get("source")).get("url") or "").strip()
    ):
        missing.append("source")
    return missing


def incomplete_metric_cards_for_numeric_claim(claim: Any, fact_cards: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not claim_has_numeric_terms(claim):
        return []
    gaps: List[Dict[str, Any]] = []
    for card in fact_cards:
        payload = _as_dict(card)
        if not _is_metric_card(payload):
            continue
        missing = _metric_missing_fields(payload)
        if missing:
            gaps.append(
                {
                    "evidence_id": str(payload.get("evidence_id") or payload.get("id") or "").strip(),
                    "missing_fields": missing,
                }
            )
    return gaps


def validate_claim_supported_by_facts(claim: Any, fact_cards: Sequence[Dict[str, Any]]) -> ClaimSupportResult:
    claim_text = _compact_text(claim)
    cards = [_as_dict(card) for card in fact_cards if isinstance(card, dict)]
    if not claim_text or not cards:
        return ClaimSupportResult(status="unsupported", unsupported_terms=["no_cited_fact_cards"])
    evidence_text = " ".join(_evidence_text(card) for card in cards)
    evidence_norm = _norm(evidence_text)
    evidence_numbers = _number_token_set(evidence_text)
    unsupported_numbers = [
        number
        for number in _claim_numbers(claim_text)
        if not _number_variants(number).intersection(evidence_numbers)
    ]
    unsupported_entities = [
        entity
        for entity in _material_english_entities(claim_text)
        if _norm(entity) not in evidence_norm
    ]
    unsupported_chinese_anchors = _unsupported_chinese_anchors(claim_text, evidence_text)
    unsupported_terms = _dedupe([*unsupported_numbers, *unsupported_entities, *unsupported_chinese_anchors])
    if unsupported_terms:
        return ClaimSupportResult(
            status="unsupported",
            unsupported_terms=unsupported_terms,
            unsupported_numbers=unsupported_numbers,
            unsupported_entities=unsupported_entities,
        )
    return ClaimSupportResult(status="supported")
