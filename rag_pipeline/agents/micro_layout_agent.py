from __future__ import annotations

import json
import re
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    from .block_schema import select_blocks_for_chapter
    from .layout_claim_matcher import claims_by_chapter, match_claims_to_blocks
    from .report_contracts import normalize_evidence_refs
    from ..config.search_config import build_llm_config_for_task
    from ..search.memory import call_openai_compatible_json, llm_config_is_ready
except Exception:  # pragma: no cover - direct script mode fallback
    from block_schema import select_blocks_for_chapter  # type: ignore
    from layout_claim_matcher import claims_by_chapter, match_claims_to_blocks  # type: ignore
    from report_contracts import normalize_evidence_refs  # type: ignore
    build_llm_config_for_task = None  # type: ignore
    call_openai_compatible_json = None  # type: ignore
    llm_config_is_ready = None  # type: ignore


AGENT_NAME = "micro_layout_agent"
AGENT_DESCRIPTION = "Micro Layout Agent. Chooses chapter-level expression structures after evidence binding."


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _compact(value: Any, max_chars: int = 220) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "..."


def _env_int(name: str, default: int, *, min_value: int = 0, max_value: int = 100) -> int:
    try:
        value = int(os.getenv(name, str(default)) or default)
    except (TypeError, ValueError):
        value = default
    return max(min_value, min(max_value, value))


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _dedupe(values: Iterable[Any], *, limit: int = 12) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        text = _compact(value, 160)
        if not text:
            continue
        key = re.sub(r"\s+", "", text.lower())
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def _text_blob(package: Dict[str, Any]) -> str:
    parts = [
        package.get("chapter_title"),
        package.get("chapter_question"),
    ]
    for collection in ("core_evidence", "supporting_evidence", "sample_evidence", "table_evidence", "clue_evidence"):
        for item in _as_list(package.get(collection))[:8]:
            if isinstance(item, dict):
                parts.extend([item.get("fact"), item.get("metric"), item.get("value")])
    return " ".join(str(part or "") for part in parts)


def _evidence_shape(package: Dict[str, Any], chapter_blueprint: Optional[Dict[str, Any]] = None) -> set[str]:
    chapter_blueprint = _as_dict(chapter_blueprint)
    values: set[str] = set()
    for key in ("required_evidence_mix", "source_template_keys"):
        for value in _as_list(package.get(key)) + _as_list(chapter_blueprint.get(key)):
            text = str(value or "").strip().lower()
            if text:
                values.add(text)
    return values


def _layout_type(package: Dict[str, Any], chapter_blueprint: Optional[Dict[str, Any]] = None) -> str:
    if _as_list(package.get("conflicts")):
        return "metric_reconciliation"
    proof_roles = {
        str(item.get("proof_role") or "").strip().lower()
        for collection in ("core_evidence", "supporting_evidence", "sample_evidence", "table_evidence")
        for item in _as_list(package.get(collection))
        if isinstance(item, dict)
    }
    if "counter" in proof_roles:
        return "argument_with_boundary"
    if "case" in proof_roles:
        return "case_argument"
    shape = _evidence_shape(package, chapter_blueprint)
    if shape.intersection({"policy", "policy_regulation", "official_definition"}):
        return "transmission_chain"
    if shape.intersection({"technology", "technology_product", "industry_chain", "capacity", "cost"}):
        return "mechanism_map"
    if shape.intersection({"competition", "player", "customer_case"}):
        return "case_argument"
    if shape.intersection({"capital", "capital_market", "filing_company", "market_price", "transaction"}):
        return "signal_validation"
    if shape.intersection({"risk", "counter", "counter_evidence"}):
        return "argument_with_boundary"
    if "metric" in proof_roles or len(_as_list(package.get("table_evidence"))) >= 2:
        return "evidence_matrix"
    if len(_as_list(package.get("table_evidence"))) >= 2:
        return "evidence_matrix"
    return "argument_first"


def _table_type(layout_type: str, package: Dict[str, Any]) -> str:
    if layout_type == "metric_reconciliation":
        return "metric_reconciliation"
    proof_roles = {
        str(item.get("proof_role") or "").strip().lower()
        for item in _as_list(package.get("table_evidence"))
        if isinstance(item, dict)
    }
    blob = _text_blob(package).lower()
    if proof_roles.intersection({"metric", "financial_metric", "market_price"}) or re.search(r"cagr|tam|sam|som|market|size|growth|\d", blob):
        return "market_metric_table"
    if proof_roles.intersection({"technology_product", "technology"}) or re.search(r"technology|patent|roadmap|yield|mass production|chip|model|compute", blob):
        return "technology_roadmap"
    if proof_roles.intersection({"counter", "counter_evidence", "risk"}) or re.search(r"risk|delay|lawsuit|regulation|failure|downside", blob):
        return "risk_register"
    if proof_roles.intersection({"filing", "company_filing", "capital", "financial_metric"}) or re.search(r"investment|valuation|ipo|funding|revenue|margin", blob):
        return "investment_priority_table"
    if "case" in proof_roles:
        return "customer_painpoint_matrix"
    if proof_roles.intersection({"competition", "player", "customer_case"}) or re.search(r"competitor|competition|market share|player", blob):
        return "competitor_matrix"
    return "evidence_matrix"


def _table_role(table_type: str) -> str:
    table_type = str(table_type or "").strip()
    if table_type in {"risk_register"}:
        return "risk_boundary_table"
    if table_type in {"investment_priority_table"}:
        return "decision_support_table"
    if table_type in {"market_metric_table", "cagr_calculation", "metric_reconciliation"}:
        return "core_metric_table"
    if table_type in {"competitor_matrix", "technology_roadmap", "evidence_matrix"}:
        return "core_argument_table"
    return "supporting_table"


def _table_anchor_preferences(table_type: str) -> List[str]:
    table_type = str(table_type or "").strip()
    if table_type in {"market_metric_table", "cagr_calculation", "metric_reconciliation"}:
        return ["metric_reconciliation", "evidence_matrix", "thesis"]
    if table_type in {"competitor_matrix", "customer_painpoint_matrix"}:
        return ["competitive_positioning", "case_comparison", "evidence_matrix", "thesis"]
    if table_type in {"technology_roadmap", "technology_maturity"}:
        return ["technology_maturity", "mechanism_chain", "evidence_matrix", "thesis"]
    if table_type == "risk_register":
        return ["risk_trigger", "scenario_analysis", "verification_checklist"]
    if table_type == "investment_priority_table":
        return ["verification_checklist", "scenario_analysis", "thesis"]
    return ["evidence_matrix", "thesis"]


def _placement_slot(table_type: str, anchor_block_type: str) -> str:
    table_type = str(table_type or "").strip()
    anchor_block_type = str(anchor_block_type or "").strip()
    if table_type == "risk_register" or anchor_block_type == "risk_trigger":
        return "before_risk"
    if table_type == "investment_priority_table":
        return "before_decision"
    if table_type in {"technology_roadmap", "technology_maturity"} or anchor_block_type in {"technology_maturity", "mechanism_chain"}:
        return "after_mechanism"
    if table_type in {"market_metric_table", "cagr_calculation", "metric_reconciliation"}:
        return "after_evidence_matrix" if anchor_block_type != "thesis" else "after_thesis"
    if table_type in {"competitor_matrix", "customer_painpoint_matrix"}:
        return "after_evidence_matrix"
    return "after_evidence_matrix" if anchor_block_type != "thesis" else "after_thesis"


def _anchor_section_for_table(sections: Sequence[Dict[str, Any]], table_type: str) -> Dict[str, str]:
    preferences = _table_anchor_preferences(table_type)
    first_section: Dict[str, Any] = {}
    for section in sections:
        section = _as_dict(section)
        if not first_section:
            first_section = section
        block_type = str(section.get("block_type") or section.get("output_type") or "").strip()
        if block_type in preferences:
            return {
                "anchor_section_id": str(section.get("section_id") or "").strip(),
                "anchor_block_type": block_type,
            }
    if first_section:
        return {
            "anchor_section_id": str(first_section.get("section_id") or "").strip(),
            "anchor_block_type": str(first_section.get("block_type") or first_section.get("output_type") or "thesis").strip(),
        }
    return {"anchor_section_id": "", "anchor_block_type": ""}


def _table_planning(
    *,
    package: Dict[str, Any],
    chapter_blueprint: Dict[str, Any],
    layout_type: str,
    sections: Sequence[Dict[str, Any]],
    llm_client: Any = None,
) -> Dict[str, Any]:
    table_evidence = [item for item in _as_list(package.get("table_evidence")) if isinstance(item, dict)]
    all_items = _evidence_items(package)
    min_table_evidence = _env_int("REPORT_MIN_TABLE_EVIDENCE_PER_CHAPTER", 4, min_value=1, max_value=20)
    min_rows = _env_int("REPORT_MIN_TABLE_ROWS", 2, min_value=2, max_value=8)
    role_counts: Dict[str, int] = {}
    high_quality_refs = 0
    for item in all_items:
        role = str(item.get("proof_role") or item.get("evidence_role") or item.get("role") or "").strip().lower()
        if role:
            role_counts[role] = role_counts.get(role, 0) + 1
        if str(item.get("source_level") or "").strip().upper() in {"A", "B"} and (
            item.get("ref") or item.get("evidence_id") or item.get("source_ref") or _as_list(item.get("source_refs"))
        ):
            high_quality_refs += 1
    table_type = _table_type(layout_type, package)
    structural_need = bool(_as_list(package.get("conflicts"))) or layout_type in {"metric_reconciliation", "evidence_matrix", "mechanism_map", "case_argument", "argument_with_boundary"}
    comparison_need = any(role_counts.get(role, 0) >= 2 for role in ("metric", "financial_metric", "case", "technology_product", "counter", "filing"))
    enough_rows = len(table_evidence) >= min_table_evidence or high_quality_refs >= min_rows + 1
    need_table = bool(enough_rows and (structural_need or comparison_need or len(table_evidence) >= min_table_evidence + 1))
    why_table_needed = ""
    why_no_table = ""
    if need_table:
        why_table_needed = "本章存在多个可比对象或指标，表格能把口径、时间和判断差异放在同一视图里，直接支撑核心结论。"
    elif len(table_evidence) < min_table_evidence and high_quality_refs < min_rows + 1:
        why_no_table = "高质量可追溯证据不足，强行出表会把证据缺口伪装成结论。"
    elif not (structural_need or comparison_need):
        why_no_table = "本章主要解释机制和因果链，正文叙述比表格更能保留判断逻辑。"
    else:
        why_no_table = "现有证据放入表格后不会增加专业密度，优先保留在段落或附录线索中。"
    anchor = _anchor_section_for_table(sections, table_type)
    slot = _placement_slot(table_type, anchor.get("anchor_block_type", ""))
    return {
        "planner_source": "layout_rules",
        "llm_planning_requested": bool(llm_client),
        "need_table": need_table,
        "why_table_needed": why_table_needed,
        "why_no_table": why_no_table,
        "table_type": table_type,
        "table_role": _table_role(table_type),
        "placement_slot": slot,
        "anchor_section_id": anchor.get("anchor_section_id", ""),
        "anchor_block_type": anchor.get("anchor_block_type", ""),
        "minimum_rows": min_rows,
        "maximum_rows": _env_int("REPORT_MAX_BODY_TABLE_ROWS", 8, min_value=2, max_value=30),
        "render_priority": 80 if table_type in {"market_metric_table", "metric_reconciliation", "competitor_matrix", "technology_roadmap"} else 60,
        "required_evidence_roles": _dedupe([role for role, count in role_counts.items() if count], limit=6),
        "minimum_source_level": "B",
        "evidence_count": len(table_evidence),
        "high_quality_ref_count": high_quality_refs,
    }


def _has_financial_evidence(package: Dict[str, Any]) -> bool:
    financial_re = re.compile(r"营收|收入|利润|毛利|现金流|净利|亏损|费用率|PE|PS|估值|财报|filing|financial", re.I)
    for item in _evidence_items(package):
        blob = " ".join(
            str(item.get(key) or "")
            for key in ("proof_role", "evidence_role", "source_type", "metric", "fact", "title", "content", "summary")
        )
        if financial_re.search(blob):
            return True
    return False


GENERIC_SECTION_TITLES = {"事实依据", "商业化证据", "核心观察", "本章结论"}
FORBIDDEN_EXACT_PUBLIC_TITLES = {
    "代表性案例对比",
    "反向信号与失效条件",
    "市场空间是否成立",
    "付费转化是否成立",
}
TITLE_FORBIDDEN_TERMS = {
    "证据",
    "口径",
    "变量",
    "可验证信号",
    "核心观察",
    "事实依据",
    "商业化证据",
    "判断依据",
}
LOW_SPECIFIC_VARIABLES = {
    "竞争变量",
    "指标口径",
    "付费验证",
    "需求场景",
    "验证变量",
    "可验证信号",
    "关键事实",
    "事实",
    "case",
    "metric",
    "support",
    "source_check",
    "directional",
}
INTERNAL_ROLE_TOKENS = {
    "official_me",
    "source_check",
    "counter",
    "metric",
    "proof_role",
    "block_type",
    "evidence_matrix",
    "unit_economics",
    "risk_trigger",
    "case_comparison",
    "metric_reconciliation",
}
PUBLISHER_TITLE_TOKENS = {
    "ijiwei",
    "36kr",
    "36氪",
    "sina",
    "sohu",
    "baidu",
    "zhihu",
    "爱集微",
    "财联社",
    "界面",
    "新浪",
    "腾讯",
    "网易",
    "搜狐",
    "百度",
    "知乎",
    "雪球",
    "虎嗅",
    "钛媒体",
    "亿欧",
    "证券时报",
    "证券日报",
    "中国证券网",
    "人民网",
    "新华社",
    "央视",
    "凤凰网",
    "公众号",
}
PUBLISHER_DOMAIN_RE = re.compile(
    r"\b(?:ijiwei|36kr|sina|qq|sohu|baidu|zhihu|caixin|cls|stcn|cnstock|xinhuanet|people|ifeng|netease)\b|[a-z0-9-]+\.(?:com|cn|net|org)",
    re.I,
)
ALLOWED_TITLE_VARIABLE_RE = re.compile(
    r"规模|增速|收入|利润|价格|成本|订单|采购|续约|复购|渗透率|份额|占比|ROI|客户|部署|流程|安全|治理|权限|可靠|竞争|玩家|场景|落地|商业化|技术",
    re.I,
)
BLOCK_LENS_PHRASE = {
    "case_comparison": "落地",
    "customer_painpoint_matrix": "落地",
    "competitive_positioning": "竞争",
    "unit_economics": "商业化",
    "metric_reconciliation": "指标",
    "technology_maturity": "技术",
    "risk_trigger": "边界",
    "scenario_analysis": "情景",
    "verification_checklist": "复核",
    "evidence_matrix": "判断",
    "signal_validation": "判断",
    "argument": "判断",
    "thesis": "判断",
}
BLOCK_DEFAULT_TITLES = {
    "case_comparison": "落地信号是否清晰",
    "customer_painpoint_matrix": "落地信号是否清晰",
    "competitive_positioning": "玩家动作如何变化",
    "unit_economics": "商业化信号是否清晰",
    "metric_reconciliation": "关键指标如何变化",
    "technology_maturity": "技术约束在哪里",
    "risk_trigger": "反向信号如何影响判断",
    "scenario_analysis": "情景边界如何变化",
    "verification_checklist": "复核重点在哪里",
    "evidence_matrix": "判断依据是否成立",
    "signal_validation": "判断依据是否成立",
    "argument": "判断依据是否成立",
    "thesis": "判断依据是否成立",
}
TITLE_TEMPLATE_TAIL_KEYWORDS = {
    "landing": {"落地", "部署", "应用场景"},
    "entrance": {"入口", "渠道"},
    "revenue": {"收入", "营收", "收益", "付费", "商业化", "出货", "部署", "销量", "规模", "市场规模"},
    "metric": {"指标", "口径"},
    "bottleneck": {"卡点", "瓶颈"},
    "boundary": {"边界", "条件"},
    "where": {"发生", "落地", "在哪里"},
    "conclusion": {"结论", "判断", "推翻"},
    "established": {"成立", "信号"},
    "competition": {"格局", "竞争格局"},
}

COMMERCIAL_RE = re.compile(r"营收|收入|利润|毛利|现金流|付费|收费|价格|续约|订单|采购|客户采购|revenue|pricing|renewal|order|procurement", re.I)
DEMAND_RE = re.compile(r"需求|客户|场景|部署|采购|流程|试点|adoption|workflow|deployment|customer", re.I)
COMPETITION_RE = re.compile(r"竞争|玩家|渠道|生态|份额|替代|平台|厂商|competition|player|channel|ecosystem", re.I)
TECH_RE = re.compile(r"技术|工具调用|权限|安全|可靠|部署|标准|产品|专利|模型|agent|workflow|standard|patent|security|tool", re.I)
METRIC_RE = re.compile(r"规模|增速|口径|指标|市场|金额|CAGR|TAM|SAM|SOM|growth|metric|market|size", re.I)
RISK_RE = re.compile(r"风险|反证|失败|边界|合规|责任|安全|成本|约束|counter|risk|failure|constraint", re.I)


_LAYOUT_SECTION_LABELS = {
    "argument_first": "核心判断与证据边界",
    "metric_reconciliation": "指标口径与可比性",
    "case_argument": "案例对照",
    "argument_with_boundary": "成立条件",
    "transmission_chain": "影响路径",
    "mechanism_map": "影响路径与约束",
    "signal_validation": "可验证信号",
    "evidence_matrix": "可验证信号",
}


def _public_fact_card(item: Dict[str, Any]) -> Dict[str, Any]:
    quality = _as_dict(item.get("public_fact_quality"))
    return _as_dict(item.get("public_fact_card") or quality.get("public_fact_card"))


def _title_key(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "").strip().lower())


def _looks_internal_role(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    lowered = text.lower()
    if lowered in INTERNAL_ROLE_TOKENS:
        return True
    if any(token in lowered for token in INTERNAL_ROLE_TOKENS):
        return True
    return bool(re.fullmatch(r"[a-z]+(?:_[a-z0-9]+){1,4}", lowered))


def _looks_like_publisher_or_domain(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    lowered = text.lower()
    if any(token.lower() in lowered for token in PUBLISHER_TITLE_TOKENS):
        return True
    if PUBLISHER_DOMAIN_RE.search(lowered):
        return True
    return bool(re.search(r"[\u4e00-\u9fff]+-[a-z][a-z0-9-]{2,}", text, flags=re.I))


def _clean_title_token(value: Any, *, max_chars: int = 14) -> str:
    raw = re.sub(r"\s+", " ", str(value or "")).strip(" -_/:：，。；;")
    if not raw:
        return ""
    if _looks_like_publisher_or_domain(raw):
        return ""
    if re.search(r"\.\.\.|…|要闻|快讯|日讯|新闻|报道称|转载|赛道|冷水|（）|\(\)|Over the weekend|我们也|本文|原文|下载|报告|白皮书|商业帝国|价值及应用|今年\s*\d|\d+\s*月份", raw, flags=re.I):
        return ""
    chinese_count = len(re.findall(r"[\u4e00-\u9fff]", raw))
    if chinese_count > 16 and re.search(r"[，。；：？！、]", raw):
        return ""
    text = _compact(raw, max_chars).strip(" -_/:：，。；;")
    if "..." in text or "…" in text:
        return ""
    if not text or text in GENERIC_SECTION_TITLES or text in LOW_SPECIFIC_VARIABLES:
        return ""
    if _looks_internal_role(text) or _title_has_forbidden_terms(text):
        return ""
    return text


def _block_default_title(block_type: str) -> str:
    return BLOCK_DEFAULT_TITLES.get(block_type, "关键判断如何变化")


def _contains_tail_keyword(value: str, keyword_group: str) -> bool:
    text = str(value or "")
    return any(keyword and keyword in text for keyword in TITLE_TEMPLATE_TAIL_KEYWORDS.get(keyword_group, set()))


def _safe_title_template(value: str, template: str, keyword_group: str, block_type: str) -> str:
    clean = _clean_title_token(value, max_chars=12)
    if not clean or _contains_tail_keyword(clean, keyword_group):
        return _block_default_title(block_type)
    return _compact(template.format(value=clean), 24)


def _short_title_subject(value: Any) -> str:
    raw = re.sub(r"\s+", "", str(value or "")).strip(" ：:，,。；;")
    if not raw:
        return ""
    if re.search(r"[？?！!。；;]", raw):
        return ""
    if "，" in raw or "," in raw or "、" in raw:
        raw = re.split(r"[，,、]", raw, 1)[0]
    if len(re.findall(r"[\u4e00-\u9fff]", raw)) > 8:
        return ""
    return _clean_title_token(raw, max_chars=8)


def _subject_from_items(items: Sequence[Dict[str, Any]]) -> str:
    for item in items:
        if not isinstance(item, dict):
            continue
        card = _public_fact_card(item)
        for key in ("subject", "company", "entity", "actor"):
            subject = _clean_title_token(card.get(key) or item.get(key), max_chars=18)
            if subject:
                return subject
    return ""


def _variable_from_items(items: Sequence[Dict[str, Any]]) -> str:
    for item in items:
        if not isinstance(item, dict):
            continue
        card = _public_fact_card(item)
        for key in ("analysis_variable", "variable", "fact_type", "proof_role"):
            variable = _clean_title_token(card.get(key) or item.get(key), max_chars=14)
            if variable:
                return variable
    return ""


def _evidence_text(item: Dict[str, Any]) -> str:
    card = _public_fact_card(item)
    return " ".join(
        str(value or "")
        for value in (
            item.get("proof_role"),
            item.get("evidence_role"),
            item.get("role"),
            item.get("source_type"),
            item.get("metric"),
            item.get("fact"),
            item.get("clean_fact"),
            item.get("distilled_fact"),
            item.get("title"),
            card.get("fact_type"),
            card.get("analysis_variable"),
            " ".join(str(v or "") for v in _as_list(card.get("block_affinity"))),
        )
    )


def _items_for_block(package: Dict[str, Any], block_type: str, *, limit: int = 8) -> List[Dict[str, Any]]:
    block_type = str(block_type or "").strip()
    if block_type == "metric_reconciliation":
        collections = ("metric_evidence", "table_evidence")
        pattern = METRIC_RE
    elif block_type == "unit_economics":
        collections = ("metric_evidence", "case_evidence", "core_evidence", "supporting_evidence")
        pattern = COMMERCIAL_RE
    elif block_type in {"case_comparison", "customer_painpoint_matrix"}:
        collections = ("case_evidence", "supporting_evidence", "directional_evidence")
        pattern = DEMAND_RE
    elif block_type == "competitive_positioning":
        collections = ("case_evidence", "supporting_evidence", "directional_evidence", "core_evidence")
        pattern = COMPETITION_RE
    elif block_type == "technology_maturity":
        collections = ("core_evidence", "supporting_evidence", "directional_evidence", "metric_evidence")
        pattern = TECH_RE
    elif block_type in {"risk_trigger", "verification_checklist", "scenario_analysis"}:
        collections = ("counter_evidence", "directional_evidence", "core_evidence", "supporting_evidence")
        pattern = RISK_RE
    else:
        collections = ("core_evidence", "supporting_evidence", "directional_evidence", "sample_evidence")
        pattern = re.compile(r".+")
    items: List[Dict[str, Any]] = []
    for collection in collections:
        for item in _as_list(package.get(collection)):
            if not isinstance(item, dict):
                continue
            if pattern.search(_evidence_text(item)):
                items.append(item)
            if len(items) >= limit:
                return items
    return items


def _chapter_blob(package: Dict[str, Any], items: Sequence[Dict[str, Any]]) -> str:
    parts = [package.get("chapter_title"), package.get("chapter_question")]
    parts.extend(_evidence_text(item) for item in items[:8])
    return " ".join(str(part or "") for part in parts)


def _chapter_context_variable(package: Dict[str, Any]) -> str:
    blob = " ".join(str(package.get(key) or "") for key in ("chapter_title", "chapter_question"))
    if re.search(r"真实需求|市场空间|概念热度|需求", blob):
        return "需求验证"
    if re.search(r"竞争格局|玩家|能力|渠道|成本", blob):
        return "玩家能力"
    if re.search(r"商业化|试点|付费|订单|客户|收入", blob):
        return "商业化进展"
    if re.search(r"技术|供应|监管|替代|约束|机会排序", blob):
        return "约束变量"
    if re.search(r"规模|增速|价格|机会判断|市场", blob):
        return "规模价格"
    return ""


def _analysis_variable(package: Dict[str, Any], block_type: str, items: Sequence[Dict[str, Any]]) -> str:
    context = _chapter_context_variable(package)
    cards = [_public_fact_card(item) for item in items if isinstance(item, dict)]
    for card in cards:
        variable = _clean_title_token(card.get("analysis_variable") or card.get("variable"), max_chars=12)
        if variable and not ALLOWED_TITLE_VARIABLE_RE.search(variable):
            variable = ""
        if variable and variable not in GENERIC_SECTION_TITLES:
            if variable in LOW_SPECIFIC_VARIABLES and context:
                return context
            return variable
    subject = _subject_from_items(items)
    if subject:
        return subject
    blob = _chapter_blob(package, items)
    if block_type == "metric_reconciliation" or METRIC_RE.search(blob):
        if context and re.search(r"规模|价格|市场", context):
            return context
        if re.search(r"规模|市场|TAM|SAM|SOM", blob, flags=re.I):
            return "市场规模"
        if re.search(r"增速|增长|CAGR", blob, flags=re.I):
            return "增速口径"
        return "指标口径"
    if block_type == "unit_economics" or COMMERCIAL_RE.search(blob):
        if context and re.search(r"商业化|需求|玩家", context):
            return context
        if re.search(r"续约|复购|renewal", blob, flags=re.I):
            return "续约信号"
        if re.search(r"订单|采购|contract|order|procurement", blob, flags=re.I):
            return "订单采购"
        return "付费场景"
    if block_type in {"case_comparison", "customer_painpoint_matrix"} or DEMAND_RE.search(blob):
        if context and re.search(r"需求|商业化", context):
            return context
        if re.search(r"流程|workflow", blob, flags=re.I):
            return "流程部署"
        if re.search(r"客户|customer", blob, flags=re.I):
            return "客户落地"
        return "需求场景"
    if block_type == "competitive_positioning" or COMPETITION_RE.search(blob):
        if context and re.search(r"玩家|竞争|商业化", context):
            return context
        if re.search(r"生态|ecosystem", blob, flags=re.I):
            return "生态位置"
        if re.search(r"渠道|channel", blob, flags=re.I):
            return "渠道动作"
        return "玩家分层"
    if block_type == "technology_maturity" or TECH_RE.search(blob):
        if context and re.search(r"约束|技术", context):
            return context
        if re.search(r"权限|安全|security", blob, flags=re.I):
            return "权限与安全"
        if re.search(r"工具调用|tool", blob, flags=re.I):
            return "工具调用"
        return "技术成熟度"
    if block_type in {"risk_trigger", "verification_checklist", "scenario_analysis"} or RISK_RE.search(blob):
        if context and re.search(r"约束|风险|规模|商业化", context):
            return context
        if re.search(r"安全|合规|责任", blob):
            return "治理边界"
        return "风险边界"
    return context


def _title_from_variable(variable: str, block_type: str) -> str:
    raw_variable = str(variable or "").strip()
    if raw_variable and (re.search(r"[？?！!。；;，,]", raw_variable) or len(re.findall(r"[\u4e00-\u9fff]", raw_variable)) > 10):
        return _block_default_title(block_type)
    if raw_variable and (raw_variable in LOW_SPECIFIC_VARIABLES or not ALLOWED_TITLE_VARIABLE_RE.search(raw_variable)):
        return _block_default_title(block_type)
    variable = _clean_title_token(variable, max_chars=12)
    if not variable:
        return _block_default_title(block_type) if raw_variable else ""
    if not ALLOWED_TITLE_VARIABLE_RE.search(variable):
        return _block_default_title(block_type)
    if re.search(r"竞争对比|玩家能力|竞争", variable):
        if block_type == "metric_reconciliation":
            return "玩家差异能否量化"
        if block_type == "unit_economics":
            return "玩家能力能否变现"
        return "玩家动作有哪些差异"
    if re.search(r"技术产业链|技术|部署卡点", variable):
        if block_type in {"case_comparison", "customer_painpoint_matrix"}:
            return "哪些技术环节开始落地"
        return "部署卡点在哪里"
    if re.search(r"商业化进展|付费转化|付费", variable):
        if block_type == "unit_economics":
            return "商业化走到哪一步"
        return "哪些场景先出现付费"
    if re.search(r"需求验证|需求", variable):
        if block_type == "unit_economics":
            return "需求能否转成付费"
        return "需求是否走出试用"
    if re.search(r"规模价格", variable):
        if block_type in {"case_comparison", "customer_painpoint_matrix"}:
            return "规模和价格如何支撑判断"
        return "规模和价格是否支撑机会"
    if re.search(r"现金流", variable):
        return "现金流能否支撑投入"
    if re.search(r"政策监管|监管", variable):
        return "监管会怎样改变边界"
    if re.search(r"成本", variable):
        return "成本压力会不会改路径"
    if block_type == "metric_reconciliation":
        if re.search(r"规模|市场|空间", variable):
            return "市场空间到底有多大"
        if re.search(r"增速|增长", variable):
            return "增长能否持续"
        if re.search(r"价格|成本", variable):
            return "价格变化意味着什么"
        return _safe_title_template(variable, "{value}如何影响判断", "conclusion", block_type)
    if block_type == "unit_economics":
        if re.search(r"订单|采购", variable):
            return "订单能否证明需求"
        if re.search(r"续约|复购", variable):
            return "续约是否开始出现"
        if re.search(r"客户|落地|场景", variable):
            return "客户落地能否变现"
        if re.search(r"需求", variable):
            return "需求能否转成付费"
        return _safe_title_template(variable, "{value}能否转成收入", "revenue", block_type)
    if block_type in {"case_comparison", "customer_painpoint_matrix"}:
        if re.search(r"客户|落地", variable):
            return "客户在哪些场景落地"
        if re.search(r"流程|部署", variable):
            return "流程部署走到哪一步"
        return _safe_title_template(variable, "{value}在哪里发生", "where", block_type)
    if block_type == "competitive_positioning":
        if re.search(r"玩家|竞争", variable):
            return "谁在占据入口"
        if re.search(r"生态|渠道", variable):
            return "生态入口由谁掌握"
        return _safe_title_template(variable, "{value}如何改变格局", "competition", block_type)
    if block_type == "technology_maturity":
        if re.search(r"权限|安全|治理", variable):
            return "安全和权限卡在哪里"
        if re.search(r"工具|调用", variable):
            return "工具调用是否足够可靠"
        return _safe_title_template(variable, "{value}还有哪些卡点", "bottleneck", block_type)
    if block_type in {"risk_trigger", "verification_checklist", "scenario_analysis"}:
        if not re.search(r"风险|反证|失败|边界|安全|治理|合规|责任|成本|约束", variable):
            return _block_default_title(block_type)
        return _safe_title_template(variable, "{value}会怎样改变判断", "conclusion", block_type)
    return _safe_title_template(variable, "{value}是否已经成立", "established", block_type)


def _title_from_subject(subject: str, variable: str, block_type: str) -> str:
    subject = _short_title_subject(subject)
    variable = _clean_title_token(variable, max_chars=12)
    if not subject:
        return ""
    if block_type in {"case_comparison", "customer_painpoint_matrix"}:
        return _safe_title_template(subject, "{value}落地到哪一步", "landing", block_type)
    if block_type == "competitive_positioning":
        return _safe_title_template(subject, "{value}占据什么入口", "entrance", block_type)
    if block_type == "unit_economics":
        return _safe_title_template(subject, "{value}能否转成收入", "revenue", block_type)
    if block_type == "metric_reconciliation":
        return _safe_title_template(subject, "{value}指标说明什么", "metric", block_type)
    if block_type == "technology_maturity":
        return _safe_title_template(subject, "{value}卡点在哪里", "bottleneck", block_type)
    if block_type in {"risk_trigger", "verification_checklist", "scenario_analysis"}:
        return _safe_title_template(subject, "{value}提示什么边界", "boundary", block_type)
    if variable:
        return _compact(f"{subject}{variable}进展", 24)
    return _compact(f"{subject}说明什么", 24)


def _title_has_forbidden_terms(title: Any) -> bool:
    text = str(title or "")
    if text in FORBIDDEN_EXACT_PUBLIC_TITLES:
        return True
    if _looks_internal_role(text):
        return True
    if _looks_like_publisher_or_domain(text):
        return True
    return any(term in text for term in TITLE_FORBIDDEN_TERMS)


def _naturalize_title(title: str, package: Dict[str, Any], block_type: str) -> str:
    title = _compact(title, 24).strip(" ：:，,。")
    if not title or title in GENERIC_SECTION_TITLES or _title_has_forbidden_terms(title):
        variable = _chapter_context_variable(package) or _analysis_variable(package, block_type, _items_for_block(package, block_type))
        title = _title_from_variable(variable, block_type)
    title = _compact(title, 24).strip(" ：:，,。")
    if _title_has_forbidden_terms(title):
        title = ""
    return title


def generate_dynamic_section_title(chapter: Dict[str, Any], block_type: str, evidence_items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    variable = _analysis_variable(chapter, block_type, evidence_items)
    if not variable and block_type in {"evidence_matrix", "thesis", "argument", "signal_validation"}:
        return {
            "dynamic_section_title": "",
            "title_source": "dropped",
            "title_variables": [],
            "block_title_generation_failed": True,
        }
    subject = _subject_from_items(evidence_items)
    subject_title = _title_from_subject(subject, variable, block_type)
    title = _naturalize_title(subject_title or _title_from_variable(variable, block_type), chapter, block_type)
    if not title or title in GENERIC_SECTION_TITLES:
        return {
            "dynamic_section_title": "",
            "title_source": "dropped",
            "title_variables": [variable] if variable else [],
            "block_title_generation_failed": True,
        }
    return {
        "dynamic_section_title": title,
        "title_source": "dynamic",
        "title_variables": [variable] if variable else [],
        "block_title_generation_failed": False,
    }


def _section_title(package: Dict[str, Any], layout_type: str, evidence_items: Sequence[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    payload = generate_dynamic_section_title(package, layout_type, list(evidence_items or []))
    if payload.get("dynamic_section_title"):
        return payload
    context = _chapter_context_variable(package)
    if context and evidence_items:
        title = _naturalize_title(_title_from_variable(context, layout_type), package, layout_type)
        return {
            "dynamic_section_title": _compact(title, 24),
            "title_source": "dynamic",
            "title_variables": [context],
            "block_title_generation_failed": False,
        }
    return payload


def _title_llm_enabled(llm_client: Any = None) -> bool:
    if llm_client is not None:
        return True
    return _env_flag("REPORT_ENABLE_LLM_SECTION_TITLES", False)


def _section_title_payload_for_llm(package: Dict[str, Any], sections: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    payload_sections: List[Dict[str, Any]] = []
    for section in sections:
        section = _as_dict(section)
        items = _items_for_block(package, str(section.get("block_type") or section.get("output_type") or ""), limit=3)
        facts = []
        for item in items[:3]:
            card = _public_fact_card(item)
            facts.append(
                _compact(
                    card.get("fact")
                    or item.get("distilled_fact")
                    or item.get("fact")
                    or item.get("title"),
                    120,
                )
            )
        payload_sections.append(
            {
                "section_id": section.get("section_id"),
                "current_title": section.get("section_title"),
                "block_type": section.get("block_type") or section.get("output_type"),
                "facts": [fact for fact in facts if fact],
            }
        )
    return {
        "chapter_title": package.get("chapter_title"),
        "chapter_question": package.get("chapter_question"),
        "sections": payload_sections,
        "rules": [
            "Return natural Chinese report subtitles, not internal labels.",
            "Do not use these words: 证据, 口径, 变量, 可验证信号, 核心观察, 事实依据, 商业化证据.",
            "Each title should be 8-18 Chinese characters and should not repeat within the chapter.",
        ],
    }


def _polish_section_titles_with_llm(package: Dict[str, Any], sections: Sequence[Dict[str, Any]], *, llm_client: Any = None) -> List[Dict[str, Any]]:
    if not sections or not _title_llm_enabled(llm_client):
        return list(sections)
    if llm_client is not None:
        # The local pipeline mostly uses provider configs rather than direct
        # clients. Keep direct clients as a future extension instead of
        # guessing sync/async behavior inside this synchronous agent.
        return list(sections)
    if build_llm_config_for_task is None or call_openai_compatible_json is None or llm_config_is_ready is None:
        return list(sections)
    try:
        config = build_llm_config_for_task("layout_title")
        if not llm_config_is_ready(config):
            return list(sections)
        response = call_openai_compatible_json(
            config=config,
            system_prompt=(
                "你是中文行研报告编辑，只负责把章节小标题改得自然、专业、可读。"
                "不要新增事实，不要写解释，只返回 JSON。"
            ),
            user_payload=_section_title_payload_for_llm(package, sections),
        )
        payload = _as_dict(response.get("payload"))
        title_map = {
            str(item.get("section_id") or ""): _compact(item.get("title"), 24)
            for item in _as_list(payload.get("sections"))
            if isinstance(item, dict)
        }
    except Exception:
        return list(sections)
    polished: List[Dict[str, Any]] = []
    seen = set()
    for section in sections:
        section = dict(section)
        section_id = str(section.get("section_id") or "")
        proposed = _naturalize_title(title_map.get(section_id) or "", package, str(section.get("block_type") or section.get("output_type") or ""))
        if proposed and proposed not in seen:
            section["section_title"] = proposed
            section["dynamic_section_title"] = proposed
            section["title_source"] = "llm"
            seen.add(proposed)
        polished.append(section)
    return polished


def _candidate_titles_for_section(package: Dict[str, Any], section: Dict[str, Any]) -> List[str]:
    block_type = str(section.get("block_type") or section.get("output_type") or "")
    items = _items_for_block(package, block_type)
    subject = _short_title_subject(_subject_from_items(items))
    raw_variable = _variable_from_items(items) or _chapter_context_variable(package)
    variable = "" if re.search(r"[，,。；;？?！!]", str(raw_variable or "")) else raw_variable
    candidates = [
        section.get("section_title"),
        section.get("dynamic_section_title"),
        _title_from_subject(subject, variable, block_type),
        _title_from_variable(variable, block_type),
    ]
    if subject and variable:
        variable = _clean_title_token(variable, max_chars=8)
        candidates.extend(
            [
                _compact(f"{subject}{variable}进展", 24),
                _compact(f"{variable}里的{subject}", 24),
            ]
        )
    chapter_hint = _chapter_context_variable(package) or _short_title_subject(
        package.get("chapter_title") or package.get("chapter_question")
    )
    if chapter_hint:
        if block_type in {"case_comparison", "customer_painpoint_matrix"}:
            candidates.append(_compact(f"{chapter_hint}的落地信号", 24))
        elif block_type in {"risk_trigger", "verification_checklist", "scenario_analysis"}:
            candidates.append(_compact(f"{chapter_hint}的边界", 24))
        elif block_type == "technology_maturity":
            candidates.append(_compact(f"{chapter_hint}的技术卡点", 24))
        else:
            candidates.append(_block_default_title(block_type))
    else:
        candidates.append(_block_default_title(block_type))
    return [_naturalize_title(str(candidate or ""), package, block_type) for candidate in candidates]


def _dedupe_report_section_titles(
    package: Dict[str, Any],
    sections: Sequence[Dict[str, Any]],
    seen_report_titles: set[str],
    dropped_sections: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    for index, raw_section in enumerate(sections, start=1):
        section = dict(raw_section)
        title = _naturalize_title(section.get("section_title") or section.get("dynamic_section_title") or "", package, str(section.get("block_type") or section.get("output_type") or ""))
        title_key = _title_key(title)
        needs_rewrite = not title or title_key in seen_report_titles or title in FORBIDDEN_EXACT_PUBLIC_TITLES or _title_has_forbidden_terms(title)
        rewritten = False
        if needs_rewrite:
            for candidate in _candidate_titles_for_section(package, section):
                candidate_key = _title_key(candidate)
                if candidate and candidate_key and candidate_key not in seen_report_titles and not _title_has_forbidden_terms(candidate):
                    title = candidate
                    title_key = candidate_key
                    rewritten = True
                    break
        if not title or not title_key or title_key in seen_report_titles or _title_has_forbidden_terms(title):
            section["omit_from_report"] = True
            section["repeated_title_dropped"] = True
            dropped_sections.append(
                {
                    "section_id": section.get("section_id") or f"{package.get('chapter_id')}_s{index}",
                    "block_type": section.get("block_type") or section.get("output_type"),
                    "reason": "repeated_title_dropped",
                    "source": "micro_layout",
                    "previous_title": section.get("section_title") or section.get("dynamic_section_title"),
                }
            )
            continue
        seen_report_titles.add(title_key)
        section["section_title"] = title
        section["dynamic_section_title"] = title
        if rewritten:
            section["repeated_title_rewritten"] = True
            section["title_source"] = "deduped"
        kept.append(section)
    return kept


def _refs(items: Sequence[Dict[str, Any]], *, limit: int = 5) -> List[str]:
    return _dedupe([item.get("ref") or item.get("evidence_id") for item in items if isinstance(item, dict)], limit=limit)


def _evidence_items(package: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        item
        for collection in (
            "core_evidence",
            "supporting_evidence",
            "metric_evidence",
            "case_evidence",
            "counter_evidence",
            "directional_evidence",
            "sample_evidence",
            "table_evidence",
            "clue_evidence",
        )
        for item in _as_list(package.get(collection))
        if isinstance(item, dict)
    ]


def _refs_for_roles(package: Dict[str, Any], roles: Sequence[Any], *, fallback: Sequence[str], limit: int = 8) -> List[str]:
    wanted = {str(role or "").strip().lower() for role in roles if str(role or "").strip()}
    if not wanted:
        return list(fallback)[:limit]
    aliases = {
        "counter_evidence": "counter",
        "official_data": "support",
        "policy_original": "policy",
        "company_filing": "filing",
        "customer_case": "case",
        "technology_product": "technology",
        "financial_metric": "metric",
    }
    expanded = set(wanted)
    expanded.update(aliases.get(role, role) for role in wanted)
    refs: List[str] = []
    for item in _evidence_items(package):
        role_values = {
            str(item.get("proof_role") or "").strip().lower(),
            str(item.get("evidence_role") or "").strip().lower(),
            str(item.get("role") or "").strip().lower(),
            str(item.get("intent") or "").strip().lower(),
        }
        source_level = str(item.get("source_level") or "").strip().upper()
        if source_level in {"A", "B"}:
            role_values.add("support")
        text_blob = " ".join(str(item.get(key) or "") for key in ("evidence_type", "source_type", "metric", "fact", "title")).lower()
        if role_values.intersection(expanded) or any(role in text_blob for role in expanded if len(role) > 3):
            ref = str(item.get("ref") or item.get("evidence_id") or "").strip()
            if ref:
                refs.append(ref)
    return _dedupe(refs, limit=limit) or list(fallback)[:limit]


def _section_for_block(package: Dict[str, Any], block: Dict[str, Any], *, index: int, fallback_refs: Sequence[str]) -> Dict[str, Any]:
    chapter_id = str(package.get("chapter_id") or "chapter")
    block_type = str(block.get("block_type") or "thesis").strip()
    original_block_type = block_type
    if block_type == "unit_economics" and not _has_financial_evidence(package):
        block_type = "signal_validation"
    evidence_items = _items_for_block(package, block_type)
    refs = _refs_for_roles(package, _as_list(block.get("required_evidence_roles")), fallback=fallback_refs, limit=8)
    title_payload = _section_title(package, block_type, evidence_items)
    title = _compact(title_payload.get("dynamic_section_title") or "", 120)
    if block_type == "thesis":
        title_payload = _section_title(package, _layout_type(package), evidence_items)
        title = _compact(title_payload.get("dynamic_section_title") or title, 120)
    text_by_type = {
        "policy_timeline": "按政策原文、执行节点和影响对象组织证据，避免只复述政策表态。",
        "value_chain_map": "按上游瓶颈、中游承接、下游需求和利润流向组织证据。",
        "customer_painpoint_matrix": "按采购主体、场景痛点、预算约束和替代方案组织证据。",
        "competitive_positioning": "按玩家位置、份额变化、壁垒和替代压力组织证据。",
        "technology_maturity": "按技术成熟度、量产验证、生态兼容和替代边界组织证据。",
        "unit_economics": "按收入质量、毛利、现金流和可持续性组织证据。",
        "risk_trigger": "把能够推翻章节判断的反向样本、指标恶化和执行偏差前置说明。",
        "verification_checklist": "把后续最需要跟踪的指标、来源和反证样本整理成验证清单。",
    }
    return {
        "section_id": str(block.get("block_id") or f"{chapter_id}_s{index}"),
        "section_title": title,
        "dynamic_section_title": title,
        "title_source": title_payload.get("title_source") or ("dynamic" if title else "dropped"),
        "title_variables": title_payload.get("title_variables") or [],
        "block_title_generation_failed": bool(title_payload.get("block_title_generation_failed")) or not bool(title),
        "section_role": str(block.get("role") or block_type),
        "block_type": block_type,
        "original_block_type": original_block_type,
        "required_evidence_refs": refs,
        "required_evidence_roles": _as_list(block.get("required_evidence_roles")),
        "output_type": block_type,
        "renderer": block.get("renderer"),
        "block_evidence_fit_score": block.get("block_evidence_fit_score"),
        "selection_reason": block.get("selection_reason"),
        "omit_from_report": not bool(title),
        "min_words": 180 if block_type in {"risk_trigger", "verification_checklist"} else 240,
        "render_blocks": [
            {
                "type": "paragraph",
                "label": "",
                "text": text_by_type.get(block_type) or package.get("chapter_question") or package.get("chapter_title") or "",
            },
            {
                "type": "evidence_list",
                "label": "鍏抽敭璇佹嵁",
                "evidence_refs": refs,
            },
        ],
    }


def _claim_basis_for_section(claim: Dict[str, Any]) -> List[str]:
    values = (
        _as_list(claim.get("evidence_basis"))
        or _as_list(claim.get("supporting_facts"))
        or _as_list(claim.get("fact_chain"))
    )
    return _dedupe(values, limit=8)


def _claim_title_for_section(package: Dict[str, Any], block_type: str, claim: Dict[str, Any]) -> str:
    for value in (
        claim.get("section_title"),
        claim.get("dynamic_section_title"),
        claim.get("dimension"),
        claim.get("question"),
    ):
        title = _naturalize_title(str(value or ""), package, block_type)
        if title:
            return _compact(title, 24)
    return _naturalize_title(_block_default_title(block_type), package, block_type)


def _enrich_section_with_matched_claim(
    package: Dict[str, Any],
    section: Dict[str, Any],
    claim: Dict[str, Any],
) -> Dict[str, Any]:
    if not claim:
        return section
    section = dict(section)
    block_type = str(section.get("block_type") or section.get("output_type") or "integrated_signal")
    refs = normalize_evidence_refs(claim)
    if refs:
        section["required_evidence_refs"] = _dedupe(
            [*_as_list(section.get("required_evidence_refs")), *refs],
            limit=8,
        )
        section["matched_llm_claim_refs"] = refs
    basis = _claim_basis_for_section(claim)
    if basis:
        section["matched_llm_claim_facts"] = basis
    section["matched_llm_claim"] = dict(claim)
    section["matched_by_llm_claim"] = True
    section["selection_reason"] = section.get("selection_reason") or "llm_claim_supported"
    if section.get("block_title_generation_failed") or not section.get("section_title"):
        title = _claim_title_for_section(package, block_type, claim)
        section["section_title"] = title
        section["dynamic_section_title"] = title
        section["title_source"] = "llm_claim"
        section["block_title_generation_failed"] = False
        section["omit_from_report"] = False
    section["render_blocks"] = [
        {
            "type": "paragraph",
            "label": "",
            "text": _compact(claim.get("claim") or claim.get("judgment") or "", 320),
        },
        {
            "type": "evidence_list",
            "label": "",
            "evidence_refs": refs,
        },
    ]
    return section
    if False:
        return {
        "section_id": str(block.get("block_id") or f"{chapter_id}_s{index}"),
        "section_title": title,
        "dynamic_section_title": title,
        "title_source": title_payload.get("title_source") or ("dynamic" if title else "dropped"),
        "title_variables": title_payload.get("title_variables") or [],
        "block_title_generation_failed": bool(title_payload.get("block_title_generation_failed")) or not bool(title),
        "section_role": str(block.get("role") or block_type),
        "block_type": block_type,
        "original_block_type": original_block_type,
        "required_evidence_refs": refs,
        "required_evidence_roles": _as_list(block.get("required_evidence_roles")),
        "output_type": block_type,
        "renderer": block.get("renderer"),
        "block_evidence_fit_score": block.get("block_evidence_fit_score"),
        "selection_reason": block.get("selection_reason"),
        "omit_from_report": not bool(title),
        "min_words": 180 if block_type in {"risk_trigger", "verification_checklist"} else 240,
        "render_blocks": [
            {
                "type": "paragraph",
                "label": "",
                "text": text_by_type.get(block_type) or package.get("chapter_question") or package.get("chapter_title") or "",
            },
            {
                "type": "evidence_list",
                "label": "关键证据",
                "evidence_refs": refs,
            },
        ],
        }


def run_micro_layout_agent(
    *,
    report_blueprint: Optional[Dict[str, Any]] = None,
    chapter_evidence_packages: Optional[Sequence[Dict[str, Any]]] = None,
    structured_analysis: Optional[Dict[str, Any]] = None,
    claim_units_by_chapter: Optional[Dict[str, Sequence[Dict[str, Any]]]] = None,
    llm_client: Any = None,
) -> List[Dict[str, Any]]:
    """Create per-chapter expression plans from evidence packages."""

    report_blueprint = _as_dict(report_blueprint)
    structured_analysis = _as_dict(structured_analysis)
    claim_units_by_chapter = _as_dict(claim_units_by_chapter) or claims_by_chapter(_as_list(structured_analysis.get("claim_units")))
    blueprint_by_id = {
        str(chapter.get("chapter_id") or ""): chapter
        for chapter in _as_list(report_blueprint.get("chapters"))
        if isinstance(chapter, dict)
    }
    layouts: List[Dict[str, Any]] = []
    seen_report_titles: set[str] = set()
    for package in list(chapter_evidence_packages or []):
        if not isinstance(package, dict):
            continue
        chapter_id = str(package.get("chapter_id") or f"chapter_{len(layouts) + 1}")
        chapter_blueprint = _as_dict(blueprint_by_id.get(chapter_id))
        block_plan = select_blocks_for_chapter(
            chapter_blueprint or package,
            profile=_as_dict(report_blueprint.get("layout_strategy")),
            evidence_package=package,
            claim_units_by_chapter=claim_units_by_chapter,
        )
        layout_type = _layout_type(package, chapter_blueprint)
        core = [item for item in _as_list(package.get("core_evidence")) if isinstance(item, dict)]
        supporting = [item for item in _as_list(package.get("supporting_evidence")) if isinstance(item, dict)]
        conflicts = _as_list(package.get("conflicts"))
        missing = _as_list(package.get("missing_evidence"))
        core_refs = _refs(core[:8], limit=8)
        supporting_refs = _refs(supporting[:8], limit=8)
        fallback_refs = core_refs or supporting_refs
        public_block_plan = [
            block
            for block in block_plan
            if str(_as_dict(block).get("render_plan") or "must_render") == "must_render"
            and _as_dict(block).get("public_render") is not False
        ]
        claim_layout_match_diagnostics = match_claims_to_blocks(
            chapter_id,
            _as_list(claim_units_by_chapter.get(chapter_id)),
            public_block_plan,
        )
        matched_block_ids = {
            str(block_id or "")
            for block_id in _as_dict(claim_layout_match_diagnostics.get("matches")).keys()
            if str(block_id or "")
        }
        has_claim_context = any(_as_list(value) for value in _as_dict(claim_units_by_chapter).values())
        demoted_public_blocks: List[Dict[str, Any]] = []
        if has_claim_context and public_block_plan:
            chapter_claim_units = _as_list(claim_units_by_chapter.get(chapter_id))
            if not chapter_claim_units:
                demoted_public_blocks = [
                    {**_as_dict(block), "render_plan": "candidate", "candidate_reason": "no_chapter_llm_claim"}
                    for block in public_block_plan
                ]
                public_block_plan = []
            elif matched_block_ids:
                filtered_public_block_plan = []
                seen_matched_claim_refs: set[str] = set()
                for index, block in enumerate(public_block_plan, start=1):
                    block_payload = _as_dict(block)
                    block_id = str(block_payload.get("block_id") or block_payload.get("section_id") or f"block_{index}")
                    if block_id in matched_block_ids:
                        matched_claim = _as_dict(_as_dict(claim_layout_match_diagnostics.get("matches")).get(block_id))
                        claim_refs = set(normalize_evidence_refs(matched_claim))
                        if claim_refs and seen_matched_claim_refs.intersection(claim_refs):
                            demoted_public_blocks.append(
                                {
                                    **block_payload,
                                    "render_plan": "candidate",
                                    "candidate_reason": "overlapping_llm_claim_refs",
                                }
                            )
                            continue
                        filtered_public_block_plan.append(block)
                        seen_matched_claim_refs.update(claim_refs)
                    else:
                        demoted_public_blocks.append({**block_payload, "render_plan": "candidate", "candidate_reason": "no_matching_llm_claim"})
                if len(filtered_public_block_plan) > 1:
                    has_specific_block = any(
                        str(_as_dict(block).get("block_type") or _as_dict(block).get("output_type") or "") != "integrated_signal"
                        for block in filtered_public_block_plan
                    )
                    if has_specific_block:
                        specific_public_block_plan: List[Dict[str, Any]] = []
                        for block in filtered_public_block_plan:
                            block_payload = _as_dict(block)
                            block_type = str(block_payload.get("block_type") or block_payload.get("output_type") or "")
                            if block_type == "integrated_signal":
                                demoted_public_blocks.append(
                                    {
                                        **block_payload,
                                        "render_plan": "candidate",
                                        "candidate_reason": "integrated_signal_redundant_with_specific_claim_block",
                                    }
                                )
                                continue
                            specific_public_block_plan.append(block)
                        filtered_public_block_plan = specific_public_block_plan
                public_block_plan = filtered_public_block_plan
            elif public_block_plan:
                demoted_public_blocks = [
                    {**_as_dict(block), "render_plan": "candidate", "candidate_reason": "no_matching_llm_claim"}
                    for block in public_block_plan
                ]
                public_block_plan = []
            claim_layout_match_diagnostics = match_claims_to_blocks(
                chapter_id,
                _as_list(claim_units_by_chapter.get(chapter_id)),
                public_block_plan,
            )
        candidate_blocks = [
            block
            for block in block_plan
            if str(_as_dict(block).get("render_plan") or "") == "candidate"
            or _as_dict(block).get("public_render") is False
        ]
        candidate_blocks.extend(demoted_public_blocks)
        matched_claims_by_block = _as_dict(claim_layout_match_diagnostics.get("matches"))
        raw_sections = []
        for index, block in enumerate(public_block_plan, start=1):
            section = _section_for_block(package, block, index=index, fallback_refs=fallback_refs)
            block_id = str(_as_dict(block).get("block_id") or _as_dict(block).get("section_id") or f"block_{index}")
            matched_claim = _as_dict(matched_claims_by_block.get(block_id))
            if matched_claim:
                section = _enrich_section_with_matched_claim(package, section, matched_claim)
            raw_sections.append(section)
        dropped_sections: List[Dict[str, Any]] = [
            {
                "section_id": section.get("section_id"),
                "block_type": section.get("block_type") or section.get("output_type"),
                "reason": "block_title_generation_failed" if section.get("block_title_generation_failed") else "dropped_generic_block",
                "source": "micro_layout",
                "title_source": section.get("title_source"),
                "selection_reason": section.get("selection_reason"),
            }
            for section in raw_sections
            if section.get("omit_from_report")
        ]
        sections: List[Dict[str, Any]] = [section for section in raw_sections if not section.get("omit_from_report")]
        sections = _polish_section_titles_with_llm(package, sections, llm_client=llm_client)
        sections = _dedupe_report_section_titles(package, sections, seen_report_titles, dropped_sections)
        if conflicts and not any(section.get("block_type") == "risk_trigger" for section in sections):
            sections.append(
                {
                    "section_id": f"{chapter_id}_s2",
                    "section_title": "判断边界",
                    "section_role": "counter_or_boundary",
                    "required_evidence_refs": _dedupe(
                        [ref for conflict in conflicts for ref in _as_list(_as_dict(conflict).get("evidence_refs"))],
                        limit=6,
                    ),
                    "output_type": "boundary",
                    "min_words": 160,
                    "render_blocks": [
                        {
                            "type": "paragraph",
                            "label": "判断边界",
                            "text": "同一指标存在不同口径或相反信号时，正文只保留可解释的判断范围。",
                        }
                    ],
                }
            )
        elif missing and not any(section.get("block_type") == "verification_checklist" for section in sections):
            # Evidence gaps are score diagnostics. They should not become
            # empty public sections that downstream agents immediately drop.
            pass
        table_requests: List[Dict[str, Any]] = []
        if False and len(_as_list(package.get("table_evidence"))) >= int(os.getenv("REPORT_MIN_TABLE_EVIDENCE_PER_CHAPTER", "4")):
            table_type = _table_type(layout_type, package)
            table_requests.append(
                {
                    "table_id": f"{chapter_id}_t1",
                    "table_type": table_type,
                    "purpose": f"用结构化方式回答：{package.get('chapter_question') or package.get('chapter_title')}",
                    "required_evidence_refs": _refs(_as_list(package.get("table_evidence"))[:8], limit=8),
                }
            )
        table_plan = _table_planning(
            package=package,
            chapter_blueprint=chapter_blueprint,
            layout_type=layout_type,
            sections=sections,
            llm_client=llm_client,
        )
        table_requests = []
        if table_plan.get("need_table"):
            table_requests.append(
                {
                    "table_id": f"{chapter_id}_t1",
                    "need_table": True,
                    "table_type": table_plan.get("table_type") or "evidence_matrix",
                    "table_role": table_plan.get("table_role") or "core_argument_table",
                    "purpose": f"用结构化表格支撑本章判断：{package.get('chapter_question') or package.get('chapter_title')}",
                    "why_table_needed": table_plan.get("why_table_needed"),
                    "why_no_table": "",
                    "placement_slot": table_plan.get("placement_slot") or "after_evidence_matrix",
                    "anchor_section_id": table_plan.get("anchor_section_id"),
                    "anchor_block_type": table_plan.get("anchor_block_type"),
                    "required_evidence_roles": table_plan.get("required_evidence_roles") or [],
                    "minimum_source_level": table_plan.get("minimum_source_level") or "B",
                    "minimum_rows": table_plan.get("minimum_rows") or 2,
                    "maximum_rows": table_plan.get("maximum_rows") or 8,
                    "render_priority": table_plan.get("render_priority") or 50,
                    "fallback_if_invalid": "demote_to_narrative",
                    "required_evidence_refs": _refs(_as_list(package.get("table_evidence"))[:8], limit=8),
                }
            )
        followups = []
        for gap in missing:
            gap = _as_dict(gap)
            suggestion = _compact(gap.get("suggestion"), 180)
            if suggestion:
                followups.append(
                    {
                        "query": f"{chapter_blueprint.get('chapter_question') or package.get('chapter_question') or package.get('chapter_title')} {suggestion}",
                        "targets_gap": suggestion,
                        "agent": "iqs",
                    }
                )
        layouts.append(
            {
                "agent": AGENT_NAME,
                "chapter_id": chapter_id,
                "chapter_title": package.get("chapter_title"),
                "layout_type": layout_type,
                "blocks": block_plan,
                "must_render_blocks": public_block_plan,
                "candidate_blocks": candidate_blocks,
                "sections": sections,
                "dropped_sections": dropped_sections,
                "claim_layout_match_diagnostics": {
                    "llm_claim_to_block_match_count": claim_layout_match_diagnostics.get("matched_count", 0),
                    "llm_claim_unmatched_count": claim_layout_match_diagnostics.get("unmatched_count", 0),
                    "must_block_matched_by_llm_claim_count": claim_layout_match_diagnostics.get("matched_count", 0),
                    "must_block_dropped_no_matching_claim_count": max(
                        0,
                        len(public_block_plan) - int(claim_layout_match_diagnostics.get("matched_count", 0) or 0),
                    ),
                },
                "table_planning": table_plan,
                "table_requests": table_requests,
                "follow_up_queries": followups,
            }
        )
    return layouts
