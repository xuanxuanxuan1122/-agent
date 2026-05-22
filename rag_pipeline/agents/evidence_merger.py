from __future__ import annotations

import hashlib
import json
import os
import re
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

from rag_pipeline.cache.evidence_cache import store_evidence_from_package
from rag_pipeline.cache.trusted_source_cache import store_trusted_sources_from_package

try:
    from rag_pipeline.contracts.evidence_quality import apply_evidence_quality_contract, infer_claim_type
except Exception:  # pragma: no cover - direct script mode fallback
    apply_evidence_quality_contract = None  # type: ignore
    infer_claim_type = None  # type: ignore


MERGER_NAME = "evidence_merger"

INDUSTRY_DIMENSIONS = [
    "综合研究问题",
]

CHAPTER_PLAN = [
    {"id": "ch1", "name": "执行摘要", "writer": "summarizer"},
    {"id": "ch2", "name": "动态证据与判断", "writer": "writer"},
    {"id": "ch3", "name": "结论、风险与验证清单", "writer": "conclusion_writer"},
]

CHAPTER_DIM_MAPPING = {
    "ch2": list(INDUSTRY_DIMENSIONS),
    "ch3": list(INDUSTRY_DIMENSIONS),
}

DIMENSION_ALIASES: Dict[str, str] = {}

DIMENSION_KEYWORDS = {
    "综合研究问题": [],
}

AGENT_LABELS = {
    "rag": "RAG",
    "industry_rag_agent": "RAG",
    "iqs": "IQS",
    "web_analysis_agent": "IQS",
    "openai_web": "OPENAI_WEB",
    "openai_web_search_agent": "OPENAI_WEB",
    **{f"iqs_lane_{index}": f"IQS Lane {index}" for index in range(1, 7)},
    **{f"iqs_lane_{index}_agent": f"IQS Lane {index}" for index in range(1, 7)},
    "fin": "FIN",
    "fin_agent": "FIN",
    "fin_data_agent": "FIN",
}

AGENT_DIMENSIONS = {
}

DEFAULT_DOMAIN_BLACKLIST = [
    "4399.com",
    "3dmgame.com",
    "gamersky.com",
    "17173.com",
    "ali213.net",
    "duowan.com",
    "youxi",
    "zxxk.com",
    "21cnjy.com",
    "zujuan.com",
    "doc88.com",
    "docin.com",
    "51test.net",
    "zhidao.baidu.com",
    "baike.baidu.com",
    "baijiahao.baidu.com",
    "wenku.baidu.com",
    "51baogao.cn",
    "chinabaogao.com",
    "chinairn.com",
    "leetcode.cn",
]

OFFICIAL_DOMAIN_HINTS = [
    ".gov.cn",
    "gov.cn",
    "stats.gov.cn",
    "ndrc.gov.cn",
    "miit.gov.cn",
    "mof.gov.cn",
    "pbc.gov.cn",
    "csrc.gov.cn",
    "samr.gov.cn",
    "sse.com.cn",
    "szse.cn",
    "hkexnews.hk",
    "cninfo.com.cn",
    "sec.gov",
]

HEAD_MEDIA_DOMAIN_HINTS = [
    "reuters.com",
    "bloomberg.com",
    "caixin.com",
    "yicai.com",
    "stcn.com",
    "cs.com.cn",
    "21jingji.com",
    "36kr.com",
    "thepaper.cn",
]

SELF_MEDIA_DOMAIN_HINTS = [
    "zhihu.com",
    "baijiahao.baidu.com",
    "weibo.com",
    "toutiao.com",
    "sohu.com",
    "163.com/dy",
    "mp.weixin.qq.com",
    "book118.com",
    "renrendoc.com",
    "docin.com",
    "doc88.com",
    "wk.baidu.com",
    "wenku.baidu.com",
    "xueqiu.com",
    "mguba.eastmoney.com",
]

SEARCH_OR_AGGREGATOR_HINTS = [
    "baidu.com/s?",
    "google.com/search",
    "bing.com/search",
    "mofcom.gov.cn/allsearch",
    "allsearch",
    "search?",
    "view.inews.qq.com",
    "k.sina.cn",
    "finance.sina.cn",
    "m.163.com",
    "caifuhao.eastmoney.com",
    "guba.eastmoney.com",
]

NUMERIC_RE = re.compile(
    r"(?P<prefix>约|超过|超|达到|达|为|同比|环比|预计|亏损|盈利|增长|下降)?\s*"
    r"(?P<number>-?\d+(?:,\d{3})*(?:\.\d+)?)\s*"
    r"(?P<unit>%|pct|个百分点|万亿元|千亿元|百亿元|亿元|千万元|百万元|万元|亿美元|千万美元|百万美元|万美元|美元/桶|元/吨|美元|"
    r"万吨|吨|GWh|MWh|kWh|万千瓦|GW|MW|KW|亿|万台套|万台|万套|万件|台|套|件|个|人|家|倍|元)?",
    re.I,
)

METRIC_HINTS = [
    ("市场规模", ["市场规模", "规模"]),
    ("CAGR", ["CAGR", "复合增速", "复合增长率"]),
    ("增速", ["增速", "同比", "增长率", "增长"]),
    ("渗透率", ["渗透率"]),
    ("出货量", ["出货", "出货量"]),
    ("销量", ["销量", "销售量"]),
    ("市场份额", ["市场份额", "市占率", "份额"]),
    ("CR3", ["CR3"]),
    ("CR5", ["CR5"]),
    ("融资金额", ["融资金额", "融资额", "融资"]),
    ("估值", ["估值"]),
    ("股价", ["股价"]),
    ("市值", ["市值"]),
    ("营收", ["营收", "营业收入", "收入"]),
    ("净利润", ["净利润", "归母净利润"]),
    ("亏损", ["亏损"]),
    ("毛利率", ["毛利率"]),
    ("现金流", ["现金流"]),
    ("政策目标", ["政策目标", "目标"]),
]

META_FIELD_LABELS = ["背景", "解释", "报告用途"]
GENERIC_METRIC_NAMES = {"", "数据指标", "数据点", "关键数据", "比例/增速", "占比"}
EVIDENCE_ROLES = {
    "core": "可进入正文核心判断",
    "supporting": "可辅助支撑正文判断",
    "clue": "线索证据，需要交叉验证",
    "appendix": "只进附录，不支撑核心判断",
    "rejected": "真正剔除",
}
REJECTED_STATUSES = {"rejected", "spam", "irrelevant", "blacklisted", "exclude"}
REJECTED_ROLES = {"rejected", "spam", "irrelevant", "blacklisted", "exclude"}
QUALITATIVE_KEEP_RE = re.compile(
    r"认证|客户|订单|中标|公告|试点|示范|补贴|目录|量产|投产|扩产|供货|合作|政策|标准|审批|注册|招股书|年报|财报|专利|论文|签约|采购|入围|供应链|验证",
    re.I,
)
CONTAMINATION_TEXT_PATTERNS = [
    r"因此可用于支撑：[^。\n]{0,140}的阶段判断、机会排序或风险提示，但必须保留口径边界",
    r"未来\s*\d+[-–—]\d+\s*个月应跟踪该口径是否被更多权威来源[^。\n]{0,180}",
    r"该条数据不能单独证明完整行业趋势[^。\n]{0,140}",
    r"若持续增强，可提高相关章节判断权重",
    r"说明[^。]{0,30}已经有可引用口径[^。\n]{0,80}",
    r"证明强度为[高中低]，需要结合来源类型和时间口径使用",
    r"这条数据把[^。]{0,50}从抽象判断推进到可追踪证据[^。\n]{0,120}",
    r"提供了后续横向比较、建模或访谈验证的锚点",
    r"后续应补充来源范围、样本口径和企业级验证[^。\n]{0,100}",
    r"避免把单点信息误读为稳定趋势",
    r"来源类型为[^。\n]{0,60}，来源为[^。\n]{0,120}，时间口径为[^。\n]{0,80}，证据置信度\d+(?:\.\d+)?",
]


def _compact_text(value: Any, max_chars: int = 600) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "..."


def _clean_evidence_content(value: Any, *, max_chars: int = 520) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"【[^】]{1,40}】", "", text)
    text = re.sub(r"\[id:[^\]]+\]", "", text, flags=re.I)
    text = re.sub(r"\[\d{1,3}\]", "", text)
    for pattern in CONTAMINATION_TEXT_PATTERNS:
        text = re.sub(pattern, "", text)
    for label in META_FIELD_LABELS:
        text = re.sub(rf"[；;。]?\s*{label}\s*[:：][^；;。]*", "", text)
    text = re.sub(r"^\s*(?:核心数字|核心结论|事实描述|结论)\s*[:：]\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return _compact_text(text.strip(" ，,；;。"), max_chars=max_chars)


def _clean_metric_name(value: Any, content: Any = "") -> str:
    metric = re.sub(r"\s+", "", str(value or "").strip())
    if metric in GENERIC_METRIC_NAMES:
        metric = _infer_metric(content, "")
    if metric in GENERIC_METRIC_NAMES:
        return "关键事实"
    return metric


def _clean_value_text(value: Any) -> str:
    text = re.sub(r"\s+", "", str(value or "").strip())
    text = text.strip(" ，,；;。")
    if _is_plain_small_number(text):
        return ""
    return text


def _is_plain_small_number(value: Any) -> bool:
    text = str(value or "").strip()
    if not re.fullmatch(r"-?\d+(?:\.\d+)?", text):
        return False
    try:
        return abs(float(text)) < 10
    except ValueError:
        return False


def should_discard_evidence(fact_dict: Dict[str, Any]) -> bool:
    value = _clean_value_text(fact_dict.get("value"))
    name = re.sub(
        r"\s+",
        "",
        str(fact_dict.get("name") or fact_dict.get("metric") or fact_dict.get("indicator") or "").strip(),
    )
    content = _clean_evidence_content(
        fact_dict.get("clean_fact") or fact_dict.get("fact") or fact_dict.get("content"),
        max_chars=520,
    )
    if re.match(r"^年(?:[，,]|全球|中国|智能|农业|行业|市场)", content):
        return True
    normalized_content = re.sub(r"\s+", "", content)
    compact_pair = f"{name}{value}"
    is_field_artifact = (
        not normalized_content
        or normalized_content == compact_pair
        or len(normalized_content) <= 12
        or bool(re.fullmatch(r"(?:定性事实|关键事实)-?\d+(?:\.\d+)?%?", normalized_content))
    )
    if "定性事实" in name or "定性事实" in normalized_content:
        if QUALITATIVE_KEEP_RE.search(content):
            return False
        return True
    if "市场规模" in name and "%" in value and is_field_artifact:
        return True
    if "增速" in name and re.search(r"亿|万元|万美元|美元|元", value):
        return True
    if "估值" in name and re.fullmatch(r"-?0+(?:\.0+)?", value):
        return True
    if _is_plain_small_number(value) and is_field_artifact:
        return True
    if is_field_artifact and re.fullmatch(r"(?:并购|融资|估值|市场规模|增速)\d+(?:\.\d+)?", normalized_content):
        return True
    return False


def _clean_fact_description(evidence: Dict[str, Any]) -> str:
    content = _clean_evidence_content(evidence.get("content"), max_chars=520)
    metric = _clean_metric_name(evidence.get("metric"), content)
    value = _clean_value_text(evidence.get("value"))
    if content:
        return _compact_text(content, max_chars=260)
    if should_discard_evidence({"metric": metric, "value": value, "content": content}):
        return ""
    if metric and value and metric != "关键事实":
        description = f"{metric}为{value}"
        if re.search(r"\d", description):
            return _compact_text(description, max_chars=260)
    if value:
        return _compact_text(f"{metric}{value}" if metric else value, max_chars=260)
    return ""


def _clean_fact_period(evidence: Dict[str, Any]) -> str:
    source = _as_dict(evidence.get("source"))
    date = str(source.get("date") or "").strip()
    if date:
        return date
    content = str(evidence.get("content") or "")
    match = re.search(r"(20\d{2}(?:[-—~至]\d{2,4})?年?|20\d{2}Q[1-4]|近\d+年|未来\d+年)", content)
    return match.group(1) if match else ""


def _analysis_lens_for_dimension(dimension: Any) -> str:
    return "把单点事实拆成它能证明的结论、不能证明的边界、商业含义和后续验证动作。"


def _source_quality_payload(source: Dict[str, Any], confidence: Any, conflict_flag: Any = False) -> Dict[str, Any]:
    source_type = str(source.get("source_type") or "").strip() or "unknown"
    title = str(source.get("title") or "未命名来源").strip()
    date = str(source.get("date") or "").strip() or "未标注日期"
    return {
        "source_type": source_type,
        "title": title,
        "date": date,
        "confidence": _clip(confidence),
        "conflict_flag": bool(conflict_flag),
    }


def _analysis_input_for_evidence(evidence: Dict[str, Any], *, fact_text: Any = "") -> Dict[str, Any]:
    source = _as_dict(evidence.get("source"))
    data_point = _compact_text(fact_text or evidence.get("clean_fact") or evidence.get("clean_content") or evidence.get("content"), max_chars=360)
    dimension = str(evidence.get("dimension") or "").strip()
    return {
        "analysis_status": "pending_llm_data_analysis",
        "data_point": data_point,
        "dimension": dimension,
        "metric": str(evidence.get("metric") or "").strip(),
        "metric_kind": str(evidence.get("metric_kind") or "").strip(),
        "value": str(evidence.get("value") or "").strip(),
        "period": str(evidence.get("period") or _clean_fact_period(evidence) or source.get("date") or "").strip(),
        "numeric_values": list(evidence.get("numeric_values") or []),
        "source_level": str(evidence.get("source_level") or "").strip(),
        "evidence_role": str(evidence.get("evidence_role") or "").strip(),
        "allowed_use": str(evidence.get("allowed_use") or "").strip(),
        "evidence_card": _as_dict(evidence.get("evidence_card")),
        "semantic_status": str(evidence.get("semantic_status") or "").strip(),
        "semantic_reason": str(evidence.get("semantic_reason") or "").strip(),
        "source": {
            "title": str(source.get("title") or "未命名来源").strip(),
            "url": str(source.get("url") or source.get("source_url") or "").strip(),
            "date": str(source.get("date") or "").strip(),
            "source_type": str(source.get("source_type") or "").strip(),
        },
        "analysis_lens": _analysis_lens_for_dimension(dimension),
        "task": {
            "task_id": str(evidence.get("task_id") or "").strip(),
            "dimension_id": str(evidence.get("dimension_id") or "").strip(),
            "dimension_name": str(evidence.get("dimension_name") or "").strip(),
            "evidence_goal": str(evidence.get("evidence_goal") or "").strip(),
            "task_relevance_score": evidence.get("task_relevance_score"),
        },
        "analysis_contract": {
            "proof": "提炼这条数据能够支持的事实判断。",
            "new_information": "说明它为当前维度补充了哪类信息。",
            "future_relevance": "说明它对商业化、投资优先级或风险识别的影响。",
            "boundary": "保留来源、时间和统计口径边界。",
        },
        "source_quality": _source_quality_payload(source, evidence.get("confidence"), evidence.get("conflict_flag")),
    }


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _strict_quality_mode() -> bool:
    mode = str(os.getenv("REPORT_QUALITY_MODE") or os.getenv("QUALITY_MODE") or "balanced").strip().lower()
    if mode in {"speed", "fast", "loose", "draft", "balanced", "quick_market_scan"}:
        return False
    if mode in {"strict", "deep_strict", "due_diligence", "investment_due_diligence"}:
        return True
    raw = os.getenv("STRICT_EVIDENCE_MODE")
    if raw is None:
        return False
    return str(raw).strip().lower() in {"1", "true", "yes", "on", "strict"}


def _directional_c_min_confidence() -> float:
    raw = os.getenv("REPORT_DIRECTIONAL_C_MIN_CONFIDENCE", "0.55")
    try:
        return max(0.0, min(1.0, float(raw)))
    except (TypeError, ValueError):
        return 0.55


def _source_family(evidence: Dict[str, Any]) -> str:
    source = _as_dict(evidence.get("source"))
    source_type = str(source.get("source_type") or evidence.get("source_type") or "").strip().lower()
    url = str(source.get("url") or "").lower()
    title = str(source.get("title") or "").lower()
    text = f"{source_type} {url} {title}"
    if source_type in {"official", "government", "financial_report", "annual_report", "prospectus", "exchange"}:
        return "official/filing"
    if source_type in {"research", "academic", "industry_report", "association", "consulting", "market_research", "brokerage", "whitepaper", "authoritative_secondary"}:
        return "research/association"
    if any(term in text for term in ["customer", "case", "procurement", "order", "contract", "client", "tender"]):
        return "company/case"
    if source_type in {"media", "news"}:
        return "news/secondary"
    return "unknown"


def _evidence_card_for(evidence: Dict[str, Any]) -> Dict[str, Any]:
    level = str(evidence.get("source_level") or "").strip().upper() or "UNKNOWN"
    role = str(evidence.get("evidence_role") or "").strip().lower()
    proof_role = str(evidence.get("proof_role") or _as_dict(evidence.get("search_task")).get("proof_role") or "").strip().lower()
    semantic_status = str(evidence.get("semantic_status") or "").strip().lower()
    has_metric = bool(evidence.get("metric") or evidence.get("value") or evidence.get("numeric_values"))
    directness = "direct" if proof_role in {"metric", "case", "source_check", "counter"} or has_metric else "indirect"
    if level in {"C", "D", "UNKNOWN"} or role in {"clue", "appendix"}:
        directness = "clue" if directness != "direct" else "indirect"
    if level in {"A", "B"} and role == "core" and directness == "direct" and semantic_status not in REJECTED_STATUSES:
        allowed_use = "core_claim"
        inference_distance = "low"
    elif level in {"A", "B"} and role in {"core", "supporting"} and semantic_status not in REJECTED_STATUSES:
        allowed_use = "supporting"
        inference_distance = "medium"
    elif level == "C" and semantic_status not in REJECTED_STATUSES and not evidence.get("appendix_only"):
        allowed_use = "directional_signal"
        inference_distance = "high"
    elif level == "C" and semantic_status not in REJECTED_STATUSES:
        allowed_use = "clue"
        inference_distance = "high"
    else:
        allowed_use = "appendix_only"
        inference_distance = "high"
    period = str(evidence.get("period") or _clean_fact_period(evidence) or _as_dict(evidence.get("source")).get("date") or "").strip()
    metric_definition = {
        "metric": str(evidence.get("metric") or "").strip(),
        "value": str(evidence.get("value") or "").strip(),
        "scope": str(evidence.get("scope") or "").strip(),
        "period": period,
        "unit": str(evidence.get("numeric_unit") or "").strip(),
    }
    fact = str(evidence.get("clean_fact") or evidence.get("clean_content") or evidence.get("content") or "").strip()
    return {
        "fact": fact,
        "source_level": level,
        "source_family": _source_family(evidence),
        "proof_role": proof_role or "support",
        "directness": directness,
        "scope": str(evidence.get("scope") or evidence.get("dimension_name") or evidence.get("dimension") or "").strip(),
        "period": period,
        "metric_definition": metric_definition,
        "can_prove": [item for item in [evidence.get("evidence_goal"), evidence.get("dimension_name"), proof_role] if str(item or "").strip()],
        "cannot_prove": [
            "industry-wide conclusion without cross-source bundle",
            "investment priority without counter-evidence",
            "market certainty from a single source",
        ],
        "inference_distance": inference_distance,
        "contradictions": [],
        "allowed_use": allowed_use,
    }


def get_dynamic_dimensions(research_plan: Optional[Dict[str, Any]] = None) -> List[str]:
    plan = _as_dict(research_plan)
    dimensions: List[str] = []
    seen = set()
    for raw in _as_list(plan.get("chapters")):
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("chapter_title") or raw.get("title") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        dimensions.append(name)
    for raw in _as_list(plan.get("dimensions")):
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("dimension_name") or raw.get("name") or raw.get("dimension") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        dimensions.append(name)
    return dimensions or ["综合研究问题"]


def _task_from_research_plan(task_id: str, research_plan: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not task_id:
        return {}
    for task in _as_list(_as_dict(research_plan).get("search_tasks")):
        if isinstance(task, dict) and str(task.get("task_id") or "") == task_id:
            return dict(task)
    return {}


def _task_payload_from_item(item: Dict[str, Any], research_plan: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    plan = _as_dict(research_plan)
    task = _as_dict(item.get("search_task"))
    if not task:
        task = _task_from_research_plan(str(item.get("task_id") or ""), plan)
    if not task:
        task = {
            "task_id": item.get("task_id"),
            "chapter_id": item.get("chapter_id"),
            "chapter_title": item.get("chapter_title"),
            "chapter_question": item.get("chapter_question"),
            "dimension_id": item.get("dimension_id"),
            "dimension_name": item.get("dimension_name"),
            "evidence_goal": item.get("evidence_goal") or item.get("targets_gap"),
            "must_have_terms": _as_list(item.get("must_have_terms")),
            "forbidden_terms": _as_list(item.get("forbidden_terms")),
            "source_priority": _as_list(item.get("source_priority")),
            "hypothesis_id": item.get("hypothesis_id"),
            "hypothesis_statement": item.get("hypothesis_statement"),
            "proof_role": item.get("proof_role"),
            "proof_standard": item.get("proof_standard"),
            "evidence_type": item.get("evidence_type"),
            "claim_type": item.get("claim_type"),
            "conclusion_type": item.get("conclusion_type"),
            "counter_evidence": item.get("counter_evidence"),
        }
    cleaned: Dict[str, Any] = {}
    for key, value in task.items():
        if value is None or value == "" or value == []:
            continue
        cleaned[key] = value
    if plan:
        if plan.get("research_object") and not cleaned.get("research_object"):
            cleaned["research_object"] = plan.get("research_object")
        if plan.get("global_required_terms") and not cleaned.get("global_required_terms"):
            cleaned["global_required_terms"] = _as_list(plan.get("global_required_terms"))
        if plan.get("query") and not cleaned.get("plan_query"):
            cleaned["plan_query"] = plan.get("query")
    return cleaned


def _dynamic_dimension_from_item(item: Dict[str, Any], research_plan: Optional[Dict[str, Any]] = None) -> str:
    for key in ["chapter_title", "dimension_name", "dimension"]:
        value = str(item.get(key) or "").strip()
        if value:
            return value
    task = _task_payload_from_item(item, research_plan)
    value = str(task.get("chapter_title") or task.get("dimension_name") or task.get("dimension") or "").strip()
    if value:
        return value
    task_id = str(item.get("task_id") or "").strip()
    task_from_plan = _task_from_research_plan(task_id, research_plan)
    return str(task_from_plan.get("chapter_title") or task_from_plan.get("dimension_name") or "").strip()


def build_dynamic_chapter_plan(dimensions: Sequence[str]) -> List[Dict[str, Any]]:
    chapters = [{"id": "ch1", "name": "执行摘要", "writer": "summarizer"}]
    for index, dimension in enumerate(dimensions, start=2):
        chapters.append({"id": f"ch{index}", "name": str(dimension), "writer": "writer"})
    chapters.append({"id": f"ch{len(chapters)+1}", "name": "结论与建议", "writer": "conclusion_writer"})
    return chapters


def build_dynamic_chapter_mapping(dimensions: Sequence[str]) -> Dict[str, List[str]]:
    mapping = {f"ch{index}": [str(dimension)] for index, dimension in enumerate(dimensions, start=2)}
    mapping[f"ch{len(dimensions)+2}"] = [str(dimension) for dimension in dimensions]
    return mapping


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clip(value: Any, default: float = 0.0) -> float:
    return max(0.0, min(1.0, _safe_float(value, default)))


def _domain(url: Any) -> str:
    parsed = urlparse(str(url or ""))
    return parsed.netloc.lower().removeprefix("www.")


def _env_csv(name: str) -> List[str]:
    raw = os.getenv(name, "")
    return [item.strip().lower() for item in re.split(r"[,;\s]+", raw) if item.strip()]


def _hash_id(*parts: Any) -> str:
    raw = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]


def _normalize_dimension(value: Any, research_plan: Optional[Dict[str, Any]] = None) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = DIMENSION_ALIASES.get(text, text)
    dynamic_dimensions = get_dynamic_dimensions(research_plan)
    if text in dynamic_dimensions:
        return text
    if text in INDUSTRY_DIMENSIONS:
        return text
    return text if research_plan and text else ""


def _infer_dimension(
    text: str,
    *,
    explicit: Any = "",
    agent: str = "",
    child_agent: str = "",
    research_plan: Optional[Dict[str, Any]] = None,
) -> str:
    explicit_dimension = _normalize_dimension(explicit, research_plan)
    lowered = str(text or "").lower()
    dynamic_dimensions = get_dynamic_dimensions(research_plan)
    if explicit_dimension and explicit_dimension not in INDUSTRY_DIMENSIONS:
        return explicit_dimension
    if research_plan:
        dynamic_scores: Dict[str, int] = {}
        for raw in _as_list(_as_dict(research_plan).get("dimensions")):
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("dimension_name") or raw.get("name") or "").strip()
            if not name:
                continue
            terms = [name, *[str(item) for item in _as_list(raw.get("must_have_terms"))]]
            dynamic_scores[name] = sum(1 for term in terms if term and term.lower() in lowered)
        if dynamic_scores:
            best_dynamic = max(dynamic_scores, key=lambda item: dynamic_scores[item])
            if dynamic_scores.get(best_dynamic, 0) > 0:
                return best_dynamic
    scores = {
        dimension: sum(1 for keyword in keywords if keyword.lower() in lowered)
        for dimension, keywords in DIMENSION_KEYWORDS.items()
    }
    best_dimension = max(scores, key=lambda item: scores[item])
    best_count = scores.get(best_dimension, 0)
    if explicit_dimension:
        explicit_count = scores.get(explicit_dimension, 0)
        if best_count >= explicit_count + 2 and best_count >= 2:
            return best_dimension
        return explicit_dimension
    for key in [agent, child_agent]:
        mapped = AGENT_DIMENSIONS.get(str(key or ""))
        if mapped:
            return mapped
    if best_count > 0:
        return best_dimension
    return dynamic_dimensions[0] if dynamic_dimensions else "综合研究问题"


def _agent_label(agent: Any, child_agent: Any = "") -> str:
    for key in [str(agent or ""), str(child_agent or "")]:
        label = AGENT_LABELS.get(key)
        if label:
            return label
    return str(agent or child_agent or "UNKNOWN").upper()


def _source_type(source: Dict[str, Any]) -> str:
    url = str(source.get("url") or source.get("source_url") or "")
    domain = _domain(url)
    title = str(source.get("title") or "")
    text = f"{domain} {url} {title}".lower()
    if any(hint in text for hint in SELF_MEDIA_DOMAIN_HINTS):
        return "self_media"
    if any(hint in text for hint in SEARCH_OR_AGGREGATOR_HINTS):
        return "media"
    explicit = str(source.get("source_type") or source.get("type") or "").strip().lower()
    if explicit:
        return explicit
    if any(hint in text for hint in OFFICIAL_DOMAIN_HINTS):
        if any(hint in text for hint in ["cninfo", "sse.com", "szse", "hkexnews", "sec.gov"]):
            return "financial_report"
        return "official"
    if any(hint in text for hint in HEAD_MEDIA_DOMAIN_HINTS):
        return "media"
    if re.search(r"(研究院|协会|学会|白皮书|报告|招股书|年报|财报)", title):
        return "research"
    return "unknown"


def _source_rank(source_type: str) -> int:
    return {
        "official": 5,
        "financial_report": 5,
        "research": 4,
        "media": 3,
        "unknown": 2,
        "self_media": 1,
    }.get(source_type, 2)


_RETRIEVAL_SCORE_FIELDS = (
    "web_final_score",
    "web_rerank_score",
    "api_rerank_score",
    "local_rerank_score",
    "task_relevance_score",
    "task_term_score",
    "retrieval_relevance_score",
)


def _source_level_rank(value: Any) -> int:
    return {"A": 5, "B": 4, "C": 3, "D": 1}.get(str(value or "").strip().upper(), 0)


def _retrieval_relevance_score(*items: Any) -> float:
    for item in items:
        payload = _as_dict(item)
        for key in _RETRIEVAL_SCORE_FIELDS:
            if key in payload and str(payload.get(key) or "").strip() != "":
                return _clip(payload.get(key), 0.0)
    return 0.0


def _copy_retrieval_score_fields(target: Dict[str, Any], *items: Any) -> None:
    for item in items:
        payload = _as_dict(item)
        for key in (
            "web_final_score",
            "web_rerank_score",
            "web_rerank_rank",
            "api_rerank_score",
            "api_rerank_rank",
            "local_rerank_score",
            "task_term_score",
            "task_relevance_score",
            "lexical_relevance_score",
            "credibility_score",
        ):
            if key in payload and payload.get(key) is not None:
                target[key] = payload.get(key)
    score = _retrieval_relevance_score(target, *items)
    if score > 0:
        target["retrieval_relevance_score"] = score


def _source_quality_rank(item: Dict[str, Any]) -> int:
    source = _as_dict(item.get("source"))
    level_rank = _source_level_rank(item.get("source_level"))
    if level_rank:
        return level_rank
    return _source_rank(str(source.get("source_type") or ""))


def _parse_date(value: Any, *, current_date: datetime) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    patterns = [
        r"(?P<year>20\d{2})[-/.年](?P<month>\d{1,2})[-/.月](?P<day>\d{1,2})",
        r"(?P<year>20\d{2})[-/.年](?P<month>\d{1,2})",
        r"(?P<year>20\d{2})Q(?P<quarter>[1-4])",
        r"(?P<year>20\d{2})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if not match:
            continue
        year = int(match.group("year"))
        if match.groupdict().get("quarter"):
            month = int(match.group("quarter")) * 3
            return datetime(year, month, 1)
        month = int(match.groupdict().get("month") or 12)
        day = int(match.groupdict().get("day") or 1)
        try:
            return datetime(year, min(max(month, 1), 12), min(max(day, 1), 28))
        except ValueError:
            return None
    return None


def _timeliness(source: Dict[str, Any], *, current_date: datetime) -> str:
    parsed = _parse_date(source.get("date") or source.get("period"), current_date=current_date)
    if not parsed:
        return "unknown"
    age_days = max(0, (current_date - parsed).days)
    if age_days <= 180:
        return "fresh"
    if age_days <= 730:
        return "recent"
    return "dated"


def _timeliness_rank(value: str) -> int:
    return {"fresh": 3, "recent": 2, "unknown": 1, "dated": 0}.get(value, 1)


def _has_number(text: Any) -> bool:
    return any(_valid_numeric_match(match) for match in NUMERIC_RE.finditer(str(text or "")))


def _valid_numeric_match(match: re.Match[str]) -> bool:
    unit = str(match.group("unit") or "")
    number = float(str(match.group("number") or "0").replace(",", ""))
    if unit:
        return True
    return not (1900 <= abs(number) <= 2099 and number.is_integer())


def _extract_numeric_values(text: Any) -> List[str]:
    values: List[str] = []
    for match in NUMERIC_RE.finditer(str(text or "")):
        if not _valid_numeric_match(match):
            continue
        value = f"{match.group('number')}{match.group('unit') or ''}"
        if value not in values:
            values.append(value)
        if len(values) >= 5:
            break
    return values


def _numeric_norm(value: Any) -> Tuple[Optional[float], str]:
    text = str(value or "")
    match = NUMERIC_RE.search(text)
    if not match or not _valid_numeric_match(match):
        return None, ""
    number = float(str(match.group("number") or "0").replace(",", ""))
    unit = str(match.group("unit") or "").lower()
    if unit in {"%", "pct", "个百分点"}:
        return number, "percent"
    if "美元" in unit:
        if "万" in unit:
            number *= 10_000
        if "亿" in unit:
            number *= 100_000_000
        return number, "currency_usd"
    if unit in {"台", "套", "件", "家", "万台", "万套", "万件", "万台套"}:
        if "万" in unit:
            number *= 10_000
        return number, "count"
    if "元" in unit or "亿" in unit or "万" in unit:
        if "万亿" in unit:
            number *= 1_000_000_000_000
        elif "亿" in unit:
            number *= 100_000_000
        elif "万" in unit:
            number *= 10_000
        return number, "currency_cny" if "元" in unit or "亿" in unit or "万" in unit else "count"
    if unit == "倍":
        return number, "ratio"
    return number, "unknown"


PERCENT_KEYS = {"percent", "percentage", "ratio_percent"}
MONEY_KEYS = {"currency_cny", "currency_usd", "money"}
COUNT_KEYS = {"count", "quantity"}


def _unit_family(value: Any, unit_key: Any = "") -> str:
    text = str(value or "")
    key = str(unit_key or "").lower()
    if key in PERCENT_KEYS or re.search(r"%|pct|百分点", text, re.I):
        return "percent"
    if key in MONEY_KEYS or re.search(r"亿元|万元|亿美元|万美元|人民币|美元|元", text):
        return "money"
    if key in COUNT_KEYS or re.search(r"万台套|万台|万套|台|套|件|亩|公顷|项目|家", text):
        return "count"
    if key == "ratio" or "倍" in text:
        return "ratio"
    if re.search(r"20\d{2}[-年]", text):
        return "date"
    return "unknown"


def _metric_kind(metric: Any, content: Any = "", dimension: Any = "") -> str:
    text = " ".join([str(metric or ""), str(content or ""), str(dimension or "")])
    if re.search(r"CAGR|复合增速|复合增长率", text, re.I):
        return "growth"
    if re.search(r"增速|同比|环比|增长率|下降率|渗透率", text):
        return "growth"
    if re.search(r"市占率|市场份额|份额|CR3|CR5|集中度", text, re.I):
        return "share"
    if re.search(r"市场规模|市场空间|TAM|SAM|SOM|规模", text, re.I):
        return "market_size"
    if re.search(r"出货|销量|交付|部署|装机|保有量|应用.*台", text):
        return "shipment"
    if re.search(r"营收|收入|净利润|利润|毛利|现金流|亏损", text):
        return "financial"
    if re.search(r"融资|估值|IPO|并购|市值|股价", text, re.I):
        return "capital"
    if re.search(r"政策|补贴|目录|标准|规划|通知|方案", text):
        return "policy_signal"
    if re.search(r"技术|算法|传感器|控制器|BOM|成本|作业效率|故障率", text, re.I):
        return "tech_signal"
    return "unknown"


def _validate_metric_semantics(metric: Any, value: Any, content: Any, dimension: Any, unit_key: Any = "") -> Dict[str, Any]:
    kind = _metric_kind(metric, content, dimension)
    family = _unit_family(value, unit_key)
    metric_text = str(metric or "").strip()

    if kind in {"growth", "share"} and family not in {"percent", "ratio"}:
        if family == "money":
            content_text = str(content or "")
            return {
                "status": "reclassified",
                "metric": "市场规模" if "规模" in content_text else "经营数据",
                "metric_kind": "market_size" if "规模" in content_text else "financial",
                "reason": f"{metric_text} 不应使用金额单位，已重分类",
            }
        if family == "count":
            return {
                "status": "reclassified",
                "metric": "出货/部署",
                "metric_kind": "shipment",
                "reason": f"{metric_text} 不应使用数量单位，已重分类为出货/部署",
            }
        return {
            "status": "exclude",
            "metric": metric_text,
            "metric_kind": kind,
            "reason": f"{metric_text} 缺少百分比或比例单位",
        }

    if kind == "market_size" and family == "count":
        return {
            "status": "reclassified",
            "metric": "出货/部署",
            "metric_kind": "shipment",
            "reason": "市场规模使用数量单位，已重分类为出货/部署",
        }

    if kind == "shipment" and family == "money":
        return {
            "status": "reclassified",
            "metric": "市场规模",
            "metric_kind": "market_size",
            "reason": "出货/部署使用金额单位，已重分类为市场规模",
        }

    if kind in {"financial", "capital"} and family in {"date", "unknown"} and not re.search(r"\d", str(value or "")):
        return {
            "status": "exclude",
            "metric": metric_text,
            "metric_kind": kind,
            "reason": "资本/经营指标缺少有效金额、比例或数值",
        }

    return {
        "status": "ok",
        "metric": metric_text or "关键事实",
        "metric_kind": kind,
        "reason": "",
    }


def _metric_key(metric: Any) -> str:
    text = re.sub(r"\s+", "", str(metric or "").lower())
    text = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", text)
    return text[:60]


def _infer_metric(text: Any, fallback: str = "") -> str:
    if str(fallback or "").strip():
        return str(fallback).strip()
    source_text = str(text or "")
    for metric, keywords in METRIC_HINTS:
        if any(keyword.lower() in source_text.lower() for keyword in keywords):
            return metric
    values = _extract_numeric_values(source_text)
    return "数据指标" if values else "定性事实"


def _source_from_raw_point(point: Dict[str, Any], fallback_source: Dict[str, Any]) -> Dict[str, Any]:
    source_url = str(point.get("source_url") or point.get("url") or point.get("source") or fallback_source.get("url") or "").strip()
    title = str(point.get("source_title") or point.get("source_name") or fallback_source.get("title") or point.get("source") or "").strip()
    return {
        "title": title or "未命名来源",
        "url": source_url,
        "date": str(point.get("date") or point.get("period") or fallback_source.get("date") or "").strip(),
        "quote": _compact_text(point.get("evidence") or fallback_source.get("quote") or "", max_chars=420),
        "source_type": str(point.get("source_type") or point.get("type") or fallback_source.get("source_type") or fallback_source.get("type") or "").strip(),
        "publisher": str(point.get("publisher") or point.get("source_publisher") or fallback_source.get("publisher") or fallback_source.get("source") or "").strip(),
        "source": str(point.get("publisher") or point.get("source_publisher") or fallback_source.get("publisher") or fallback_source.get("source") or "").strip(),
        "document_id": str(point.get("document_id") or point.get("doc_id") or fallback_source.get("document_id") or fallback_source.get("doc_id") or "").strip(),
        "page_ref": str(point.get("page_ref") or fallback_source.get("page_ref") or "").strip(),
    }


def _citation_ids_from_text(text: str) -> List[str]:
    return re.findall(r"\[(?:id\s*[:：]\s*)?(\d+)\]", str(text or ""), flags=re.I)


def _source_from_item(item: Dict[str, Any], *, citation_text: str = "") -> Dict[str, Any]:
    sources = [source for source in _as_list(item.get("key_sources")) if isinstance(source, dict)]
    citation_ids = _citation_ids_from_text(citation_text)
    for citation_id in citation_ids:
        for source in sources:
            source_id = source.get("id") if source.get("id") is not None else source.get("source_id")
            if str(source_id).strip() == str(citation_id).strip():
                return dict(source)
    if sources:
        return dict(sources[0])
    return {
        "title": str(item.get("child_agent") or item.get("agent") or "未命名来源"),
        "url": "",
        "date": "",
        "quote": "",
    }


def _is_blacklisted_source(source: Dict[str, Any], blacklist: Sequence[str]) -> bool:
    url = str(source.get("url") or source.get("source_url") or "").lower()
    domain = _domain(url)
    title = str(source.get("title") or "").lower()
    haystack = f"{domain} {url} {title}"
    return any(fragment and fragment.lower() in haystack for fragment in blacklist)


def _confidence_for_evidence(
    *,
    base_confidence: Any,
    has_numeric: bool,
    source_type: str,
    timeliness: str,
) -> float:
    confidence = _clip(base_confidence, 0.35)
    if has_numeric:
        confidence += 0.15
    if source_type in {"official", "filing", "financial_report", "company_announcement"}:
        confidence += 0.18
    elif source_type in {"research", "consulting", "market_research", "brokerage", "whitepaper", "authoritative_secondary", "technical_standard", "patent", "product_doc"}:
        confidence += 0.12
    elif source_type == "media":
        confidence += 0.06
    elif source_type == "self_media":
        confidence -= 0.15
    if timeliness == "fresh":
        confidence += 0.10
    elif timeliness == "recent":
        confidence += 0.04
    elif timeliness == "dated":
        confidence -= 0.08
    return round(_clip(confidence), 4)


SOURCE_TYPE_TO_LEVEL = {
    "official": "A",
    "government": "A",
    "filing": "A",
    "financial_report": "A",
    "annual_report": "A",
    "prospectus": "A",
    "exchange": "A",
    "company_announcement": "A",
    "research": "B",
    "academic": "B",
    "industry_report": "B",
    "association": "B",
    "consulting": "B",
    "market_research": "B",
    "brokerage": "B",
    "whitepaper": "B",
    "company_official": "B",
    "product_doc": "B",
    "technical_standard": "B",
    "patent": "B",
    "authoritative_secondary": "B",
    "media": "C",
    "news": "C",
    "self_media": "D",
    "ugc": "D",
    "unknown": "C",
    "": "C",
}

FAKE_EVIDENCE_TEXT = "official data shows ai agent adoption reached 50% in 2025"
PLACEHOLDER_SOURCE_HOSTS = {"example.com", "www.example.com", "example.gov", "www.example.gov"}


def _source_url_host(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw if re.match(r"^[a-z][a-z0-9+.-]*://", raw, flags=re.I) else f"https://{raw}")
    return str(parsed.netloc or parsed.path.split("/")[0] or "").strip().lower()


def _is_placeholder_source_url(value: Any) -> bool:
    host = _source_url_host(value)
    return bool(host and (host in PLACEHOLDER_SOURCE_HOSTS or host.endswith(".example.com") or host.endswith(".example.gov")))


def _is_generic_official_source(source: Dict[str, Any]) -> bool:
    title = str(source.get("title") or source.get("name") or "").strip().lower()
    publisher = str(source.get("publisher") or source.get("source") or source.get("organization") or "").strip()
    url = str(source.get("url") or source.get("source_url") or "").strip()
    return title == "official" and not publisher and not url


def _evidence_text_for_fake_check(evidence: Dict[str, Any]) -> str:
    source = _as_dict(evidence.get("source"))
    parts = [
        evidence.get("fact"),
        evidence.get("clean_fact"),
        evidence.get("content"),
        evidence.get("summary"),
        source.get("title"),
        source.get("publisher"),
        source.get("source"),
        source.get("url"),
        source.get("source_url"),
    ]
    return " ".join(str(item or "") for item in parts).strip().lower()


def _is_fake_or_placeholder_source(source: Dict[str, Any], evidence: Optional[Dict[str, Any]] = None) -> bool:
    payload = _as_dict(source)
    if _is_placeholder_source_url(payload.get("url") or payload.get("source_url")):
        return True
    if _is_generic_official_source(payload):
        return True
    if evidence is not None and FAKE_EVIDENCE_TEXT in _evidence_text_for_fake_check(evidence):
        return True
    return False


def _is_title_only_source_payload(source: Dict[str, Any], evidence: Optional[Dict[str, Any]] = None) -> bool:
    payload = _as_dict(source)
    evidence = _as_dict(evidence)
    title = str(payload.get("title") or payload.get("name") or evidence.get("source_title") or "").strip()
    url = str(payload.get("url") or payload.get("source_url") or evidence.get("source_url") or evidence.get("url") or "").strip()
    document_ref = str(
        payload.get("document_id")
        or payload.get("doc_id")
        or payload.get("page_ref")
        or evidence.get("document_id")
        or evidence.get("doc_id")
        or evidence.get("page_ref")
        or ""
    ).strip()
    return bool(title and not url and not document_ref)


def _source_level_for_evidence(evidence: Dict[str, Any]) -> str:
    source = _as_dict(evidence.get("source"))
    if _is_fake_or_placeholder_source(source, evidence) or _is_title_only_source_payload(source, evidence):
        return "D"
    source_type = _source_type(source) or str(evidence.get("source_type") or "").strip().lower()
    return SOURCE_TYPE_TO_LEVEL.get(source_type, "C")


def _assign_evidence_role(evidence: Dict[str, Any]) -> str:
    semantic_status = str(evidence.get("semantic_status") or "").strip().lower()
    if semantic_status in REJECTED_STATUSES:
        return "rejected"
    level = _source_level_for_evidence(evidence)
    confidence = _safe_float(evidence.get("confidence"), 0.0)
    task_score = _safe_float(evidence.get("task_relevance_score"), 0.0)
    text = " ".join(
        [
            str(evidence.get("content") or ""),
            str(evidence.get("clean_fact") or ""),
            str(evidence.get("metric") or ""),
            str(evidence.get("value") or ""),
        ]
    )
    has_numeric = bool(evidence.get("numeric_values")) or bool(re.search(r"\d", text))
    has_verifiable_qualitative = bool(QUALITATIVE_KEEP_RE.search(text))

    if level == "A" and confidence >= 0.55:
        return "core"
    if level == "B" and confidence >= 0.50:
        return "core" if has_numeric or has_verifiable_qualitative else "supporting"
    if level == "C":
        if _strict_quality_mode():
            return "clue"
        if confidence >= 0.45 and task_score >= 0.55:
            return "supporting"
        return "clue"
    if level == "D":
        return "clue"
    if has_verifiable_qualitative or has_numeric:
        return "clue"
    return "appendix"


def _source_level_distribution(items: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    distribution: Dict[str, int] = {"A": 0, "B": 0, "C": 0, "D": 0, "UNKNOWN": 0}
    for item in items:
        if not isinstance(item, dict):
            continue
        level = str(item.get("source_level") or "").strip().upper()
        if level not in distribution:
            level = "UNKNOWN"
        distribution[level] += 1
    return {key: value for key, value in distribution.items() if value}


GENERIC_GOAL_TOKENS = {
    "数据",
    "统计",
    "口径",
    "来源",
    "权威",
    "报告",
    "研究",
    "市场",
    "行业",
    "官方",
    "原文",
    "公告",
    "分析",
    "补齐",
    "寻找",
    "验证",
}


def _term_in_text(term: str, text: str) -> bool:
    term = str(term or "").strip().lower()
    if not term:
        return False
    if re.fullmatch(r"[a-z0-9]{1,3}", term):
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text, re.I))
    return term in text


def _topic_anchor_groups(task: Dict[str, Any]) -> List[List[str]]:
    topic_text = " ".join(
        str(value or "")
        for value in [
            task.get("plan_query"),
            task.get("research_object"),
            task.get("query"),
            " ".join(str(item) for item in _as_list(task.get("global_required_terms"))),
        ]
    )
    groups: List[List[str]] = []
    if re.search(r"\bAI\b|人工智能|大模型|生成式|AIGC", topic_text, re.I):
        groups.append(["人工智能", "ai", "aigc", "大模型", "生成式ai", "生成式人工智能"])
    if re.search(r"中国|国内", topic_text, re.I):
        groups.append(["中国", "国内", "china", "chinese"])
    if re.search(r"新能源汽车|新能源车|动力电池|锂电", topic_text):
        groups.append(["新能源汽车", "新能源车", "动力电池", "锂电"])
    if re.search(r"半导体|芯片|集成电路", topic_text, re.I):
        groups.append(["半导体", "芯片", "集成电路", "semiconductor", "chip"])
    return groups


def _missing_topic_anchor_groups(task: Dict[str, Any], evidence_text: str) -> List[List[str]]:
    missing: List[List[str]] = []
    for group in _topic_anchor_groups(task):
        if not any(_term_in_text(term, evidence_text) for term in group):
            missing.append(group)
    return missing


def task_acceptance_filter(evidence: Dict[str, Any], task: Dict[str, Any]) -> Dict[str, Any]:
    if not task:
        return {
            "accepted": True,
            "relevance_score": 1.0,
            "role_hint": "candidate",
            "reason": "no_task_filter",
        }
    evidence_text = " ".join(
        str(evidence.get(key) or "")
        for key in [
            "content",
            "clean_content",
            "clean_fact",
            "metric",
            "value",
        ]
    )
    source = _as_dict(evidence.get("source"))
    evidence_text = " ".join(
        [
            evidence_text,
            str(source.get("title") or ""),
            str(source.get("url") or ""),
        ]
    )
    evidence_text = re.sub(r"\s+", "", evidence_text.lower())
    readable_text = " ".join(
        [
            str(evidence.get("content") or ""),
            str(evidence.get("clean_content") or ""),
            str(evidence.get("clean_fact") or ""),
            str(evidence.get("metric") or ""),
            str(evidence.get("value") or ""),
            str(source.get("title") or ""),
            str(source.get("url") or ""),
        ]
    ).lower()

    must_terms = [str(item).strip().lower() for item in _as_list(task.get("must_have_terms")) if str(item).strip()]
    forbidden_terms = [str(item).strip().lower() for item in _as_list(task.get("forbidden_terms")) if str(item).strip()]
    source_priority = [str(item).strip().lower() for item in _as_list(task.get("source_priority")) if str(item).strip()]

    matched_must = [term for term in must_terms if term and term in evidence_text]
    must_ratio = len(matched_must) / max(len(must_terms), 1)
    forbidden_hit = any(term and term in evidence_text for term in forbidden_terms)

    dimension_name = str(task.get("dimension_name") or "").strip().lower()
    dimension_hit = bool(dimension_name and re.sub(r"\s+", "", dimension_name) in evidence_text)

    evidence_goal = str(task.get("evidence_goal") or "").strip().lower()
    goal_tokens = [
        token
        for token in re.split(r"[\s,，；;、/]+", evidence_goal)
        if len(token) >= 2 and token not in GENERIC_GOAL_TOKENS
    ]
    goal_hit = any(token in evidence_text for token in goal_tokens[:12])

    has_number = bool(re.search(r"\d", str(evidence.get("content") or evidence.get("value") or "")))
    source_text = " ".join(
        [
            str(source.get("title") or ""),
            str(source.get("url") or ""),
            str(source.get("source_type") or ""),
            str(evidence.get("source_level") or ""),
        ]
    ).lower()
    source_priority_hit = any(term and term in source_text for term in source_priority)
    missing_topic_groups = _missing_topic_anchor_groups(task, readable_text)

    score = 0.0
    score += 0.40 * must_ratio
    score += 0.20 if goal_hit else 0.0
    score += 0.15 if dimension_hit else 0.0
    score += 0.10 if has_number else 0.0
    score += 0.15 if source_priority_hit else 0.0
    if must_terms and must_ratio >= 0.5:
        score += 0.15
    if forbidden_hit:
        score -= 0.45
    score = max(0.0, min(1.0, score))
    if missing_topic_groups:
        return {
            "accepted": False,
            "relevance_score": round(min(score, 0.2), 4),
            "role_hint": "rejected",
            "reason": "topic_anchor_missing",
            "matched_terms": matched_must,
            "missing_topic_groups": missing_topic_groups,
        }
    if forbidden_hit:
        return {
            "accepted": False,
            "relevance_score": round(score, 4),
            "role_hint": "rejected",
            "reason": "forbidden_terms_hit",
            "matched_terms": matched_must,
        }
    threshold = 0.42
    if must_terms and not matched_must and not (dimension_hit or goal_hit):
        threshold = 0.58
    if score >= threshold:
        return {
            "accepted": True,
            "relevance_score": round(score, 4),
            "role_hint": "candidate",
            "reason": "task_relevance_pass",
            "matched_terms": matched_must,
        }
    return {
        "accepted": False,
        "relevance_score": round(score, 4),
        "role_hint": "weak_clue",
        "reason": "low_task_relevance_keep_as_clue",
        "matched_terms": matched_must,
    }


def _build_evidence(
    *,
    raw_id: str,
    item: Dict[str, Any],
    content: str,
    dimension: str,
    metric: str,
    value: str,
    source: Dict[str, Any],
    current_date: datetime,
    research_plan: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    agent = str(item.get("agent") or "")
    child_agent = str(item.get("child_agent") or "")
    numeric_values = _extract_numeric_values(" ".join([content, value]))
    has_numeric = bool(value and re.search(r"\d", value)) or bool(numeric_values)
    source_type = _source_type(source)
    time_bucket = _timeliness(source, current_date=current_date)
    confidence = _confidence_for_evidence(
        base_confidence=item.get("confidence"),
        has_numeric=has_numeric,
        source_type=source_type,
        timeliness=time_bucket,
    )
    evidence_id = f"EV-{raw_id}"
    s_grade = bool(has_numeric and source_type in {"official", "financial_report", "research", "media"} and confidence >= 0.55)
    normalized_value, unit_key = _numeric_norm(value or (numeric_values[0] if numeric_values else ""))
    task_payload = _task_payload_from_item(item, research_plan)
    evidence = {
        "evidence_id": evidence_id,
        "dimension": dimension,
        "content": _compact_text(content, max_chars=900),
        "metric": metric,
        "metric_key": _metric_key(metric),
        "value": value or (numeric_values[0] if numeric_values else ""),
        "numeric_values": numeric_values,
        "numeric_value": normalized_value,
        "numeric_unit": unit_key,
        "unit": item.get("unit") or item.get("numeric_unit") or unit_key,
        "source": {
            "title": str(source.get("title") or "未命名来源").strip(),
            "url": str(source.get("url") or source.get("source_url") or "").strip(),
            "date": str(source.get("date") or "").strip(),
            "quote": _compact_text(source.get("quote") or "", max_chars=420),
            "source_type": source_type,
            "publisher": str(source.get("publisher") or source.get("source") or "").strip(),
            "source": str(source.get("source") or source.get("publisher") or "").strip(),
            "document_id": str(source.get("document_id") or source.get("doc_id") or "").strip(),
            "page_ref": str(source.get("page_ref") or "").strip(),
        },
        "timeliness": time_bucket,
        "confidence": confidence,
        "agent": _agent_label(agent, child_agent),
        "agent_key": agent,
        "child_agent": child_agent,
        "round": item.get("round") or 1,
        "s_grade": s_grade,
        "conflict_flag": False,
        "fact_key": "",
        "trace": {
            "query": _compact_text(item.get("query"), max_chars=140),
            "targets_gap": str(item.get("targets_gap") or ""),
            "raw_pool_id": str(item.get("pool_id") or ""),
        },
    }
    _copy_retrieval_score_fields(evidence, item, source)
    _copy_retrieval_score_fields(evidence["source"], item, source)
    if evidence.get("retrieval_relevance_score") is not None:
        evidence["trace"]["retrieval_relevance_score"] = evidence.get("retrieval_relevance_score")
    if task_payload:
        evidence.update(
            {
                "task_id": task_payload.get("task_id"),
                "chapter_id": task_payload.get("chapter_id"),
                "chapter_title": task_payload.get("chapter_title"),
                "chapter_question": task_payload.get("chapter_question"),
                "dimension_id": task_payload.get("dimension_id"),
                "dimension_name": task_payload.get("dimension_name"),
                "evidence_goal": task_payload.get("evidence_goal"),
                "evidence_goal_id": task_payload.get("evidence_goal_id"),
                "must_have_terms": _as_list(task_payload.get("must_have_terms")),
                "forbidden_terms": _as_list(task_payload.get("forbidden_terms")),
                "source_priority": _as_list(task_payload.get("source_priority")),
                "hypothesis_id": task_payload.get("hypothesis_id"),
                "hypothesis_statement": task_payload.get("hypothesis_statement"),
                "proof_role": task_payload.get("proof_role"),
                "proof_standard": task_payload.get("proof_standard"),
                "evidence_type": task_payload.get("evidence_type"),
                "claim_type": task_payload.get("claim_type") or task_payload.get("conclusion_type"),
                "conclusion_type": task_payload.get("conclusion_type") or task_payload.get("claim_type"),
                "counter_evidence": bool(task_payload.get("counter_evidence")),
                "search_task": task_payload,
                "research_object": task_payload.get("research_object"),
                "global_required_terms": _as_list(task_payload.get("global_required_terms")),
            }
        )
    semantic = _validate_metric_semantics(
        metric=evidence.get("metric"),
        value=evidence.get("value"),
        content=evidence.get("content"),
        dimension=evidence.get("dimension"),
        unit_key=evidence.get("numeric_unit"),
    )
    if semantic.get("status") == "reclassified":
        evidence["metric"] = semantic.get("metric") or evidence["metric"]
        evidence["metric_key"] = _metric_key(evidence["metric"])
    evidence["metric_kind"] = semantic.get("metric_kind") or "unknown"
    evidence["semantic_status"] = semantic.get("status") or "ok"
    if str(evidence["semantic_status"]).strip().lower() == "exclude":
        evidence["semantic_status"] = "rejected"
    evidence["semantic_reason"] = semantic.get("reason") or ""
    evidence["source_level"] = _source_level_for_evidence(evidence)
    task_acceptance = task_acceptance_filter(evidence, task_payload)
    evidence["task_relevance_score"] = task_acceptance.get("relevance_score")
    evidence["task_accepted"] = bool(task_acceptance.get("accepted"))
    evidence["task_acceptance_reason"] = str(task_acceptance.get("reason") or "")
    evidence["task_matched_terms"] = list(task_acceptance.get("matched_terms") or [])
    evidence["task_role_hint"] = str(task_acceptance.get("role_hint") or "")
    if str(evidence.get("semantic_status") or "").strip().lower() in REJECTED_STATUSES:
        evidence["evidence_role"] = "rejected"
    elif task_payload and not evidence["task_accepted"]:
        reason = str(task_acceptance.get("reason") or "")
        if task_acceptance.get("role_hint") == "rejected":
            evidence["semantic_status"] = "rejected"
            evidence["semantic_reason"] = reason
            evidence["evidence_role"] = "rejected"
        else:
            evidence["semantic_status"] = "weak_relevance"
            evidence["semantic_reason"] = reason
            evidence["evidence_role"] = "clue"
            evidence["appendix_only"] = True
            evidence["followup_seed"] = True
    else:
        evidence["evidence_role"] = _assign_evidence_role(evidence)
    if evidence.get("evidence_role") == "clue":
        evidence.setdefault("appendix_only", True)
        evidence.setdefault("followup_seed", True)
    elif evidence.get("evidence_role") == "appendix":
        evidence.setdefault("appendix_only", True)
    elif evidence.get("evidence_role") == "rejected":
        evidence.setdefault("appendix_only", False)
        evidence.setdefault("followup_seed", False)
    else:
        evidence.setdefault("appendix_only", False)
        evidence.setdefault("followup_seed", False)
    source_level = str(evidence.get("source_level") or "").strip().upper()
    role = _canonical_role(evidence.get("evidence_role"))
    if source_level == "D":
        evidence["evidence_role"] = "clue" if role != "rejected" else "rejected"
        evidence["appendix_only"] = role != "rejected"
        evidence["enterprise_usable"] = False
        evidence["followup_seed"] = role != "rejected"
        evidence["usage_tier"] = "clue_low_quality" if role != "rejected" else "rejected"
    elif source_level == "C" and role != "rejected":
        if _strict_quality_mode():
            evidence["evidence_role"] = "clue"
            role = "clue"
        confidence = float(evidence.get("confidence") or 0.0)
        directional = (
            (not _strict_quality_mode())
            and confidence >= _directional_c_min_confidence()
            and str(evidence.get("semantic_status") or "").strip().lower() not in {"weak", "weak_relevance", "appendix"}
        )
        evidence["appendix_only"] = not directional
        evidence["enterprise_usable"] = directional
        evidence["can_support_claim_if_corrobated"] = not _strict_quality_mode()
        evidence["usage_tier"] = "directional_signal" if directional else "appendix_or_corroboration"
    elif role in {"core", "supporting"}:
        evidence["appendix_only"] = False
        evidence["enterprise_usable"] = True
        evidence["usage_tier"] = role
    elif role == "appendix":
        evidence["appendix_only"] = True
        evidence["enterprise_usable"] = False
        evidence["usage_tier"] = "appendix_only"
    evidence["clean_content"] = _clean_evidence_content(evidence["content"], max_chars=520)
    evidence["clean_fact"] = _clean_fact_description(evidence)
    evidence["period"] = _clean_fact_period(evidence)
    evidence["evidence_card"] = _evidence_card_for(evidence)
    evidence["allowed_use"] = str(_as_dict(evidence.get("evidence_card")).get("allowed_use") or "")
    if apply_evidence_quality_contract is not None:
        evidence = apply_evidence_quality_contract(
            evidence,
            strict_quality=_strict_quality_mode(),
            directional_c_min_confidence=_directional_c_min_confidence(),
        )
    evidence["analysis_input"] = _analysis_input_for_evidence(evidence, fact_text=evidence["clean_fact"])
    return evidence


def _iter_answer_lines(answer: Any, *, max_lines: int = 40) -> Iterable[str]:
    seen = set()
    for raw_line in re.split(r"[\n\r]+", str(answer or "")):
        line = re.sub(r"^\s*[-*•\d.、\)）]+\s*", "", raw_line).strip()
        if not line:
            continue
        if line.startswith("本次使用") or line.startswith("主要来源") or line.startswith("来源"):
            continue
        if len(line) < 18 and not _has_number(line):
            continue
        key = re.sub(r"\s+", "", line.lower())[:120]
        if key in seen:
            continue
        seen.add(key)
        yield line
        if len(seen) >= max_lines:
            return


def _children_to_pool(children: Dict[str, Dict[str, Any]], original_query: str) -> List[Dict[str, Any]]:
    pool: List[Dict[str, Any]] = []
    for child_name, child in children.items():
        if not isinstance(child, dict):
            continue
        agent = str(child.get("role_key") or child_name)
        pool.append(
            {
                "round": 1,
                "agent": agent,
                "child_agent": child_name,
                "query": original_query,
                "targets_gap": str(child.get("dimension") or "初始问题"),
                "status": str(child.get("status") or "failed"),
                "confidence": child.get("confidence"),
                "answer": str(child.get("answer") or ""),
                "key_sources": list(child.get("key_sources") or []),
                "limitations": _as_dict(child.get("limitations")),
                "raw_data_points": list(child.get("raw_data_points") or []),
                "rerank_diagnostics": _as_dict(child.get("rerank_diagnostics")),
                "data_gap": list(child.get("data_gap") or []),
                "dimension_name": child.get("dimension"),
                "dynamic_tasks": list(child.get("dynamic_tasks") or []),
            }
        )
    return pool


def _pool_item_signal_reason(item: Dict[str, Any]) -> str:
    if not isinstance(item, dict):
        return "invalid_item"
    status = str(item.get("status") or "").strip().lower()
    if status in {"failed", "error", "timeout", "cancelled"}:
        return "failed_followup_result"
    if status and status not in {"success", "partial"}:
        return f"status_{status}"
    key_sources = _as_list(item.get("key_sources"))
    raw_points = _as_list(item.get("raw_data_points"))
    answer = str(item.get("answer") or "").strip()
    valid_sources = [
        source
        for source in key_sources
        if (
            isinstance(source, dict)
            and (
                str(source.get("url") or source.get("source_url") or "").strip()
                or str(source.get("title") or "").strip()
                or str(source.get("snippet") or source.get("summary") or "").strip()
            )
        )
        or (not isinstance(source, dict) and len(str(source or "").strip()) >= 12)
    ]
    valid_points = [
        point
        for point in raw_points
        if (
            isinstance(point, dict)
            and (
                str(point.get("metric") or point.get("indicator") or "").strip()
                or str(point.get("value") or point.get("numeric_value") or "").strip()
                or str(point.get("source_url") or point.get("url") or point.get("source") or "").strip()
            )
        )
        or (not isinstance(point, dict) and len(str(point or "").strip()) >= 12)
    ]
    if valid_sources or valid_points:
        return "usable_signal"
    weak_negative = bool(re.search(r"(没有找到|未找到|暂无|无相关|不足以|无法确认|未能确认)", answer))
    sourced_or_numeric_answer = bool(
        len(answer) >= 80
        and not weak_negative
        and (
            re.search(r"\[\d{1,4}\]|https?://|来源|公告|财报|年报|统计|协会|专利|标准|report|filing|source", answer, re.I)
            or re.search(r"\d+(?:\.\d+)?\s*(?:%|亿美元|亿元|万台|百万|million|bn|billion)", answer, re.I)
        )
    )
    if sourced_or_numeric_answer:
        return "usable_signal"
    if status in {"success", "partial"} and not answer and not key_sources and not raw_points:
        return "empty_success_payload"
    if answer:
        return "weak_or_unsourced_payload"
    return "empty_payload"


def _filter_source_pool_for_merge(source_pool: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    usable: List[Dict[str, Any]] = []
    reason_counts: Dict[str, int] = {}
    for item in list(source_pool or []):
        if not isinstance(item, dict):
            reason_counts["invalid_item"] = reason_counts.get("invalid_item", 0) + 1
            continue
        reason = _pool_item_signal_reason(item)
        if reason == "usable_signal":
            usable.append(item)
        else:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return usable, {
        "source_pool_input_count": len([item for item in list(source_pool or []) if isinstance(item, dict)]),
        "source_pool_usable_count": len(usable),
        "source_pool_filtered_count": sum(reason_counts.values()),
        "source_pool_filter_reasons": reason_counts,
    }


def normalize_evidence_items(
    evidence_pool: Sequence[Dict[str, Any]],
    *,
    current_date: Optional[datetime] = None,
    domain_blacklist: Optional[Sequence[str]] = None,
    research_plan: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    current_date = current_date or datetime.now()
    blacklist = list(domain_blacklist or DEFAULT_DOMAIN_BLACKLIST) + _env_csv("EVIDENCE_MERGER_DOMAIN_BLACKLIST")
    normalized: List[Dict[str, Any]] = []
    filtered_noise = 0
    raw_count = 0
    score_candidates = list(_iter_retrieval_score_candidates(evidence_pool))
    source_pool_rerank_input_count = len(score_candidates)
    source_pool_rerank_score_count = sum(1 for item in score_candidates if _has_explicit_rerank_score(item))
    source_pool_rerank_returned_count = sum(
        1
        for item in score_candidates
        if "web_rerank_score" in item and str(item.get("web_rerank_score") or "").strip() != ""
    )
    child_rerank_input_count = sum(
        int(_as_dict(_as_dict(item).get("rerank_diagnostics")).get("input_count") or 0)
        for item in list(evidence_pool or [])
        if isinstance(item, dict)
    )
    child_rerank_returned_count = sum(
        int(_as_dict(_as_dict(item).get("rerank_diagnostics")).get("returned_count") or 0)
        for item in list(evidence_pool or [])
        if isinstance(item, dict)
    )
    for pool_index, item in enumerate(evidence_pool, start=1):
        if not isinstance(item, dict) or str(item.get("status") or "") == "failed":
            continue
        agent = str(item.get("agent") or "")
        child_agent = str(item.get("child_agent") or "")
        fallback_source = _source_from_item(item)
        raw_points = _as_list(item.get("raw_data_points"))
        for point_index, point in enumerate(raw_points, start=1):
            if not isinstance(point, dict):
                continue
            metric = _infer_metric(point.get("evidence") or point, str(point.get("metric") or ""))
            value = str(point.get("value") or "").strip()
            content = _compact_text(
                point.get("evidence") or f"{metric}：{value}",
                max_chars=900,
            )
            qualitative_candidate = bool(QUALITATIVE_KEEP_RE.search(content)) or "定性事实" in str(metric or "")
            if not content or (not value and not _has_number(content) and not qualitative_candidate):
                continue
            source = _source_from_raw_point(point, fallback_source)
            raw_count += 1
            if _is_blacklisted_source(source, blacklist):
                filtered_noise += 1
                continue
            dimension = _dynamic_dimension_from_item(point, research_plan) or _dynamic_dimension_from_item(item, research_plan) or _infer_dimension(
                " ".join([content, metric, value]),
                explicit=point.get("dimension") or item.get("dimension"),
                agent=agent,
                child_agent=child_agent,
                research_plan=research_plan,
            )
            point_with_task = {**item, **point}
            normalized.append(
                _build_evidence(
                    raw_id=f"{pool_index:02d}-{point_index:02d}",
                    item=point_with_task,
                    content=content,
                    dimension=dimension,
                    metric=metric,
                    value=value,
                    source=source,
                    current_date=current_date,
                    research_plan=research_plan,
                )
            )

        answer = str(item.get("answer") or "").strip()
        for line_index, line in enumerate(_iter_answer_lines(answer), start=1):
            if raw_points and not _citation_ids_from_text(line):
                continue
            source = _source_from_item(item, citation_text=line)
            raw_count += 1
            if _is_blacklisted_source(source, blacklist):
                filtered_noise += 1
                continue
            metric = _infer_metric(line)
            values = _extract_numeric_values(line)
            dimension = _dynamic_dimension_from_item(item, research_plan) or _infer_dimension(
                line,
                explicit=item.get("dimension"),
                agent=agent,
                child_agent=child_agent,
                research_plan=research_plan,
            )
            normalized.append(
                _build_evidence(
                    raw_id=f"{pool_index:02d}-L{line_index:02d}",
                    item=item,
                    content=line,
                    dimension=dimension,
                    metric=metric,
                    value=values[0] if values else "",
                    source=source,
                    current_date=current_date,
                    research_plan=research_plan,
                )
            )
    metadata = {
        "raw_evidence_count": raw_count,
        "normalized_count": len(normalized),
        "filtered_noise_count": filtered_noise,
        "source_level_distribution": _source_level_distribution(normalized),
        "candidate_count": raw_count,
        "kept_count": len([item for item in normalized if str(item.get("evidence_role") or "").lower() not in REJECTED_ROLES]),
        "rejected_count": len([item for item in normalized if str(item.get("evidence_role") or "").lower() in REJECTED_ROLES]) + filtered_noise,
        "appendix_only_count": len([item for item in normalized if item.get("appendix_only") or str(item.get("evidence_role") or "") in {"clue", "appendix"}]),
        "core_candidate_count": len([item for item in normalized if str(item.get("evidence_role") or "") in {"core", "supporting"}]),
        "domain_blacklist": blacklist,
        "source_pool_score_candidate_count": source_pool_rerank_input_count,
        "rerank_input_count": child_rerank_input_count or source_pool_rerank_input_count,
        "rerank_returned_count": child_rerank_returned_count or source_pool_rerank_returned_count,
        "source_pool_rerank_input_count": source_pool_rerank_input_count,
        "source_pool_rerank_score_count": source_pool_rerank_score_count,
        "source_pool_rerank_returned_count": source_pool_rerank_returned_count,
    }
    return normalized, metadata


def _dedupe_key(evidence: Dict[str, Any]) -> str:
    metric_key = str(evidence.get("metric_key") or "")
    value = re.sub(r"\s+", "", str(evidence.get("value") or "").lower())
    source = _as_dict(evidence.get("source"))
    period = re.sub(r"\s+", "", str(source.get("date") or "").lower())
    source_key = re.sub(
        r"\s+",
        "",
        " ".join(str(source.get(key) or "").lower() for key in ["url", "title", "source_type"]),
    )[:180]
    if metric_key and value:
        return "|".join([str(evidence.get("dimension") or ""), metric_key, value, period, source_key])
    content = re.sub(r"[^\w\u4e00-\u9fff]+", "", str(evidence.get("content") or "").lower())
    return "|".join([str(evidence.get("dimension") or ""), content[:160], source_key])


def _quality_key(evidence: Dict[str, Any]) -> Tuple[int, int, int, float, float, int]:
    return (
        1 if evidence.get("numeric_values") else 0,
        _timeliness_rank(str(evidence.get("timeliness") or "")),
        _source_quality_rank(evidence),
        _safe_float(evidence.get("confidence"), 0.0),
        _retrieval_relevance_score(evidence, evidence.get("source")),
        len(str(evidence.get("content") or "")),
    )


def dedupe_evidence(evidence_items: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in evidence_items:
        copied = dict(item)
        copied["fact_key"] = _dedupe_key(copied)
        buckets[copied["fact_key"]].append(copied)
    deduped: List[Dict[str, Any]] = []
    for items in buckets.values():
        deduped.append(max(items, key=_quality_key))
    deduped.sort(key=_quality_key, reverse=True)
    return deduped, max(0, len(evidence_items) - len(deduped))


def _clean_fact_key(fact: Dict[str, Any]) -> str:
    source = _as_dict(fact.get("source"))
    source_key = re.sub(
        r"\s+",
        "",
        " ".join(str(source.get(key) or "").lower() for key in ["url", "title", "date"]),
    )
    metric = re.sub(r"\s+", "", str(fact.get("metric") or "").lower())
    value = re.sub(r"\s+", "", str(fact.get("value") or "").lower())
    period = re.sub(r"\s+", "", str(fact.get("period") or "").lower())
    description = re.sub(r"[^\w\u4e00-\u9fff]+", "", str(fact.get("fact") or "").lower())
    if value:
        return "|".join([str(fact.get("dimension") or ""), metric, value, period, source_key])
    return "|".join([str(fact.get("dimension") or ""), description[:160], source_key])


def build_clean_facts(evidence_items: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in evidence_items:
        if not isinstance(item, dict):
            continue
        if str(item.get("semantic_status") or "").strip().lower() in REJECTED_STATUSES:
            continue
        if should_discard_evidence(item):
            continue
        fact_text = str(item.get("clean_fact") or _clean_fact_description(item)).strip()
        if not fact_text:
            continue
        source = _as_dict(item.get("source"))
        clean_source = {
            "title": str(source.get("title") or "未命名来源").strip(),
            "url": str(source.get("url") or source.get("source_url") or "").strip(),
            "date": str(source.get("date") or "").strip(),
            "source_type": str(source.get("source_type") or "").strip(),
            "publisher": str(source.get("publisher") or source.get("source") or "").strip(),
            "source": str(source.get("source") or source.get("publisher") or "").strip(),
            "document_id": str(source.get("document_id") or source.get("doc_id") or "").strip(),
            "page_ref": str(source.get("page_ref") or "").strip(),
        }
        _copy_retrieval_score_fields(clean_source, source, item)
        fact = {
            "evidence_id": item.get("evidence_id"),
            "dimension": item.get("dimension"),
            "fact": fact_text,
            "metric": _clean_metric_name(item.get("metric"), item.get("clean_content") or item.get("content")),
            "value": _clean_value_text(item.get("value")),
            "period": str(item.get("period") or _clean_fact_period(item) or "").strip(),
            "unit": str(item.get("unit") or item.get("numeric_unit") or "").strip(),
            "source": clean_source,
            "numeric_values": list(item.get("numeric_values") or []),
            "confidence": _clip(item.get("confidence"), 0.0),
            "agent": item.get("agent"),
            "s_grade": bool(item.get("s_grade")),
            "conflict_flag": bool(item.get("conflict_flag")),
            "source_level": str(item.get("source_level") or "").strip(),
            "evidence_role": str(item.get("evidence_role") or "").strip(),
            "semantic_status": str(item.get("semantic_status") or "").strip(),
            "semantic_reason": str(item.get("semantic_reason") or "").strip(),
            "metric_kind": str(item.get("metric_kind") or "").strip(),
            "task_id": str(item.get("task_id") or "").strip(),
            "dimension_id": str(item.get("dimension_id") or "").strip(),
            "dimension_name": str(item.get("dimension_name") or "").strip(),
            "evidence_goal": str(item.get("evidence_goal") or "").strip(),
            "hypothesis_id": str(item.get("hypothesis_id") or "").strip(),
            "hypothesis_statement": str(item.get("hypothesis_statement") or "").strip(),
            "proof_role": str(item.get("proof_role") or "").strip(),
            "proof_standard": str(item.get("proof_standard") or "").strip(),
            "evidence_type": str(item.get("evidence_type") or "").strip(),
            "claim_type": str(item.get("claim_type") or item.get("conclusion_type") or "").strip(),
            "conclusion_type": str(item.get("conclusion_type") or item.get("claim_type") or "").strip(),
            "task_relevance_score": item.get("task_relevance_score"),
            "retrieval_relevance_score": _retrieval_relevance_score(item, source),
            "web_final_score": item.get("web_final_score") or source.get("web_final_score"),
            "web_rerank_score": item.get("web_rerank_score") or source.get("web_rerank_score"),
            "web_rerank_rank": item.get("web_rerank_rank") or source.get("web_rerank_rank"),
            "task_term_score": item.get("task_term_score") or source.get("task_term_score"),
            "task_accepted": item.get("task_accepted"),
            "task_acceptance_reason": str(item.get("task_acceptance_reason") or "").strip(),
            "appendix_only": bool(item.get("appendix_only")),
            "enterprise_usable": bool(item.get("enterprise_usable")),
            "followup_seed": bool(item.get("followup_seed")),
            "can_support_claim_if_corrobated": bool(item.get("can_support_claim_if_corrobated")),
            "allowed_use": str(item.get("allowed_use") or _as_dict(item.get("evidence_card")).get("allowed_use") or "").strip(),
            "evidence_card": _as_dict(item.get("evidence_card")),
            "source_subtier": str(item.get("source_subtier") or "").strip(),
            "evidence_grade_note": str(item.get("evidence_grade_note") or "").strip(),
            "usage_tier": str(item.get("usage_tier") or "").strip(),
        }
        fact["analysis_input"] = _analysis_input_for_evidence({**item, **fact}, fact_text=fact_text)
        if should_discard_evidence(fact):
            continue
        buckets[_clean_fact_key(fact)].append(fact)

    facts: List[Dict[str, Any]] = []
    for items in buckets.values():
        facts.append(
            max(
                items,
                key=lambda item: (
                    _source_quality_rank(item),
                    _clip(item.get("confidence"), 0.0),
                    _retrieval_relevance_score(item, item.get("source")),
                    1 if item.get("numeric_values") else 0,
                    len(str(item.get("fact") or "")),
                ),
            )
        )
    facts.sort(
        key=lambda item: (
            _clip(item.get("confidence"), 0.0),
            1 if item.get("numeric_values") else 0,
            _source_quality_rank(item),
            _retrieval_relevance_score(item, item.get("source")),
        ),
        reverse=True,
    )
    return facts, max(0, sum(len(items) for items in buckets.values()) - len(facts))


def _chapter_fact_key(fact: Dict[str, Any]) -> str:
    metric = re.sub(r"\s+", "", str(fact.get("metric") or "").lower())
    value = re.sub(r"\s+", "", str(fact.get("value") or "").lower())
    period = re.sub(r"\s+", "", str(fact.get("period") or "").lower())
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "", str(fact.get("fact") or "").lower())
    if value:
        return "|".join([str(fact.get("dimension") or ""), metric, value, period])
    return "|".join([str(fact.get("dimension") or ""), text[:180]])


def _chapter_fact_quality(fact: Dict[str, Any]) -> Tuple[float, int, int, float, int, int]:
    source = _as_dict(fact.get("source"))
    parsed_date = _parse_date(fact.get("period") or source.get("date"), current_date=datetime.now())
    date_rank = parsed_date.toordinal() if parsed_date else 0
    return (
        _clip(fact.get("confidence"), 0.0),
        1 if fact.get("numeric_values") else 0,
        _source_quality_rank(fact),
        _retrieval_relevance_score(fact, source),
        date_rank,
        len(str(fact.get("fact") or "")),
    )


def _clean_chapter_fact(fact: Dict[str, Any]) -> Dict[str, Any]:
    source = _as_dict(fact.get("source"))
    clean_source = {
        "title": str(source.get("title") or "未命名来源").strip(),
        "url": str(source.get("url") or source.get("source_url") or "").strip(),
        "date": str(source.get("date") or "").strip(),
        "source_type": str(source.get("source_type") or "").strip(),
        "publisher": str(source.get("publisher") or source.get("source") or "").strip(),
        "source": str(source.get("source") or source.get("publisher") or "").strip(),
        "document_id": str(source.get("document_id") or source.get("doc_id") or "").strip(),
        "page_ref": str(source.get("page_ref") or "").strip(),
    }
    _copy_retrieval_score_fields(clean_source, source, fact)
    cleaned = {
        "evidence_id": str(fact.get("evidence_id") or "").strip(),
        "fact": str(fact.get("fact") or "").strip(),
        "time": str(fact.get("period") or source.get("date") or "").strip(),
        "source": clean_source,
        "dimension": str(fact.get("dimension") or "").strip(),
        "metric": str(fact.get("metric") or "").strip(),
        "value": str(fact.get("value") or "").strip(),
        "unit": str(fact.get("unit") or fact.get("numeric_unit") or "").strip(),
        "numeric_values": list(fact.get("numeric_values") or []),
        "confidence": _clip(fact.get("confidence"), 0.0),
        "s_grade": bool(fact.get("s_grade")),
        "conflict_flag": bool(fact.get("conflict_flag")),
        "source_level": str(fact.get("source_level") or "").strip(),
        "evidence_role": str(fact.get("evidence_role") or "").strip(),
        "semantic_status": str(fact.get("semantic_status") or "").strip(),
        "semantic_reason": str(fact.get("semantic_reason") or "").strip(),
        "metric_kind": str(fact.get("metric_kind") or "").strip(),
        "task_id": str(fact.get("task_id") or "").strip(),
        "dimension_id": str(fact.get("dimension_id") or "").strip(),
        "dimension_name": str(fact.get("dimension_name") or "").strip(),
        "evidence_goal": str(fact.get("evidence_goal") or "").strip(),
        "hypothesis_id": str(fact.get("hypothesis_id") or "").strip(),
        "hypothesis_statement": str(fact.get("hypothesis_statement") or "").strip(),
        "proof_role": str(fact.get("proof_role") or "").strip(),
        "proof_standard": str(fact.get("proof_standard") or "").strip(),
        "evidence_type": str(fact.get("evidence_type") or "").strip(),
        "claim_type": str(fact.get("claim_type") or fact.get("conclusion_type") or "").strip(),
        "conclusion_type": str(fact.get("conclusion_type") or fact.get("claim_type") or "").strip(),
        "task_relevance_score": fact.get("task_relevance_score"),
        "retrieval_relevance_score": _retrieval_relevance_score(fact, source),
        "web_final_score": fact.get("web_final_score") or source.get("web_final_score"),
        "web_rerank_score": fact.get("web_rerank_score") or source.get("web_rerank_score"),
        "web_rerank_rank": fact.get("web_rerank_rank") or source.get("web_rerank_rank"),
        "task_term_score": fact.get("task_term_score") or source.get("task_term_score"),
        "task_acceptance_reason": str(fact.get("task_acceptance_reason") or "").strip(),
        "appendix_only": bool(fact.get("appendix_only")),
        "enterprise_usable": bool(fact.get("enterprise_usable")),
        "followup_seed": bool(fact.get("followup_seed")),
        "can_support_claim_if_corrobated": bool(fact.get("can_support_claim_if_corrobated")),
        "allowed_use": str(fact.get("allowed_use") or _as_dict(fact.get("evidence_card")).get("allowed_use") or "").strip(),
        "evidence_card": _as_dict(fact.get("evidence_card")),
        "source_subtier": str(fact.get("source_subtier") or "").strip(),
        "evidence_grade_note": str(fact.get("evidence_grade_note") or "").strip(),
        "usage_tier": str(fact.get("usage_tier") or "").strip(),
    }
    cleaned["analysis_input"] = _as_dict(fact.get("analysis_input")) or _analysis_input_for_evidence(cleaned, fact_text=cleaned["fact"])
    return cleaned

def build_chapter_evidence(
    clean_facts: Sequence[Dict[str, Any]],
    *,
    chapter_dim_mapping: Optional[Dict[str, List[str]]] = None,
    max_per_chapter: int = 28,
) -> Dict[str, List[Dict[str, Any]]]:
    mapping = chapter_dim_mapping or CHAPTER_DIM_MAPPING
    facts = [fact for fact in clean_facts if isinstance(fact, dict) and str(fact.get("fact") or "").strip()]
    chapter_evidence: Dict[str, List[Dict[str, Any]]] = {}
    for chapter_id, dimensions in mapping.items():
        allowed = set(dimensions)
        candidates = [fact for fact in facts if str(fact.get("dimension") or "") in allowed]
        buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for fact in candidates:
            buckets[_chapter_fact_key(fact)].append(fact)
        deduped = [max(items, key=_chapter_fact_quality) for items in buckets.values()]
        deduped.sort(key=_chapter_fact_quality, reverse=True)
        chapter_evidence[chapter_id] = [_clean_chapter_fact(fact) for fact in deduped[:max_per_chapter]]
    return chapter_evidence


def detect_conflicts(evidence_items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for item in evidence_items:
        metric_key = str(item.get("metric_key") or "")
        numeric_value = item.get("numeric_value")
        unit = str(item.get("numeric_unit") or "")
        if not metric_key or numeric_value is None or not unit:
            continue
        groups[(str(item.get("dimension") or ""), metric_key, unit)].append(item)

    conflicts: List[Dict[str, Any]] = []
    for (dimension, _, unit), items in groups.items():
        if len(items) < 2:
            continue
        for left_index, left in enumerate(items):
            for right in items[left_index + 1 :]:
                left_value = _safe_float(left.get("numeric_value"))
                right_value = _safe_float(right.get("numeric_value"))
                denominator = max(min(abs(left_value), abs(right_value)), 1e-9)
                delta_ratio = abs(left_value - right_value) / denominator
                if delta_ratio <= 0.20:
                    continue
                left["conflict_flag"] = True
                right["conflict_flag"] = True
                metric = str(left.get("metric") or right.get("metric") or "同一指标")
                conflicts.append(
                    {
                        "conflict_id": f"CF-{_hash_id(left.get('evidence_id'), right.get('evidence_id'))}",
                        "dimension": dimension,
                        "metric": metric,
                        "unit": unit,
                        "delta_ratio": round(delta_ratio, 4),
                        "evidence_ids": [left.get("evidence_id"), right.get("evidence_id")],
                        "values": [
                            {
                                "value": left.get("value"),
                                "source": left.get("source"),
                                "agent": left.get("agent"),
                            },
                            {
                                "value": right.get("value"),
                                "source": right.get("source"),
                                "agent": right.get("agent"),
                            },
                        ],
                        "resolution": "保留冲突，不裁决；交给 Analysis Agent 判断引用口径。",
                    }
                )
                if len(conflicts) >= 50:
                    return conflicts
    return conflicts


def _dimension_coverage(evidence_items: Sequence[Dict[str, Any]]) -> float:
    if not evidence_items:
        return 0.0
    s_grade_count = sum(1 for item in evidence_items if item.get("s_grade"))
    numeric_count = sum(1 for item in evidence_items if item.get("numeric_values"))
    source_types = {str(_as_dict(item.get("source")).get("source_type") or "") for item in evidence_items}
    agents = {str(item.get("agent") or "") for item in evidence_items}
    score = 0.20
    score += min(0.25, len(evidence_items) * 0.05)
    score += min(0.25, numeric_count * 0.08)
    score += min(0.20, s_grade_count * 0.10)
    score += min(0.10, len(agents) * 0.03)
    if source_types & {"official", "financial_report", "research"}:
        score += 0.10
    return round(_clip(score), 4)


def _canonical_role(value: Any) -> str:
    role = str(value or "").strip().lower()
    if role == "exclude":
        return "rejected"
    return role or "appendix"


def _new_filter_funnel(metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    metadata = _as_dict(metadata)
    raw_count = int(metadata.get("raw_evidence_count") or metadata.get("raw_pool_count") or metadata.get("source_pool_count") or 0)
    filtered_noise = int(metadata.get("filtered_noise_count") or 0)
    return {
        "raw_pool_count": raw_count,
        "normalized_count": int(metadata.get("normalized_count") or 0),
        "after_blacklist_count": max(0, raw_count - filtered_noise),
        "after_content_cleaning_count": 0,
        "task_accepted_count": 0,
        "task_weak_clue_count": 0,
        "semantic_rejected_count": 0,
        "role_distribution": {
            "core": 0,
            "supporting": 0,
            "clue": 0,
            "appendix": 0,
            "rejected": 0,
        },
        "analysis_ready_count": 0,
        "appendix_only_count": 0,
        "followup_seed_count": 0,
        "reject_reasons": {},
    }


def update_filter_funnel(funnel: Dict[str, Any], evidence: Dict[str, Any]) -> None:
    role = _canonical_role(evidence.get("evidence_role") or evidence.get("role"))
    role_distribution = _as_dict(funnel.get("role_distribution"))
    role_distribution[role] = role_distribution.get(role, 0) + 1
    funnel["role_distribution"] = role_distribution

    if evidence.get("task_accepted"):
        funnel["task_accepted_count"] += 1
    elif role == "clue" or str(evidence.get("semantic_status") or "").strip().lower() == "weak_relevance":
        funnel["task_weak_clue_count"] += 1

    semantic_status = str(evidence.get("semantic_status") or "").strip().lower()
    if role == "rejected" or semantic_status in REJECTED_STATUSES:
        funnel["semantic_rejected_count"] += 1

    if evidence.get("appendix_only") or role in {"clue", "appendix"}:
        funnel["appendix_only_count"] += 1

    if evidence.get("followup_seed"):
        funnel["followup_seed_count"] += 1

    if role == "rejected":
        reason = str(evidence.get("semantic_reason") or evidence.get("task_acceptance_reason") or "unknown")
        reject_reasons = _as_dict(funnel.get("reject_reasons"))
        reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
        funnel["reject_reasons"] = reject_reasons


def build_filter_funnel(
    metadata: Dict[str, Any],
    *,
    evidence_items: Sequence[Dict[str, Any]],
    clean_evidence_list: Sequence[Dict[str, Any]],
    analysis_ready_evidence: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    funnel = _new_filter_funnel(metadata)
    for item in evidence_items:
        if isinstance(item, dict):
            update_filter_funnel(funnel, item)
    filtered_noise = int(_as_dict(metadata).get("filtered_noise_count") or 0)
    if filtered_noise:
        funnel["semantic_rejected_count"] += filtered_noise
        reject_reasons = _as_dict(funnel.get("reject_reasons"))
        reject_reasons["domain_blacklist"] = reject_reasons.get("domain_blacklist", 0) + filtered_noise
        funnel["reject_reasons"] = reject_reasons
    funnel["after_content_cleaning_count"] = len([item for item in clean_evidence_list if isinstance(item, dict)])
    funnel["analysis_ready_count"] = len([item for item in analysis_ready_evidence if isinstance(item, dict)])
    return funnel


def _has_explicit_rerank_score(item: Any) -> bool:
    payload = _as_dict(item)
    return any(
        key in payload and str(payload.get(key) or "").strip() != ""
        for key in ("web_final_score", "web_rerank_score", "api_rerank_score", "local_rerank_score", "retrieval_relevance_score")
    )


def _iter_retrieval_score_candidates(items: Sequence[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for item in list(items or []):
        if not isinstance(item, dict):
            continue
        yield item
        source = _as_dict(item.get("source"))
        if source:
            yield source
        for key in ("key_sources", "raw_data_points", "search_results", "page_results"):
            for nested in _as_list(item.get(key)):
                if isinstance(nested, dict):
                    yield nested


def _retrieval_score_count(items: Sequence[Dict[str, Any]]) -> int:
    return sum(
        1
        for item in list(items or [])
        if isinstance(item, dict)
        and (_has_explicit_rerank_score(item) or _has_explicit_rerank_score(_as_dict(item.get("source"))))
    )


def build_rerank_diagnostics(
    metadata: Dict[str, Any],
    *,
    evidence_items: Sequence[Dict[str, Any]],
    clean_evidence_list: Sequence[Dict[str, Any]],
    analysis_ready_evidence: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    metadata = _as_dict(metadata)
    input_count = int(metadata.get("rerank_input_count") or metadata.get("source_pool_rerank_input_count") or 0)
    returned_count = int(metadata.get("rerank_returned_count") or metadata.get("source_pool_rerank_returned_count") or 0)
    normalized_with_score = _retrieval_score_count(evidence_items)
    clean_with_score = _retrieval_score_count(clean_evidence_list)
    analysis_ready_with_score = _retrieval_score_count(analysis_ready_evidence)
    if not input_count:
        input_count = int(metadata.get("source_pool_score_candidate_count") or normalized_with_score)
    if not returned_count:
        returned_count = int(metadata.get("source_pool_rerank_score_count") or normalized_with_score)
    clean_scores = [
        _retrieval_relevance_score(item, _as_dict(item).get("source"))
        for item in clean_evidence_list
        if isinstance(item, dict) and _retrieval_relevance_score(item, _as_dict(item).get("source")) > 0
    ]
    top_sources: List[Dict[str, Any]] = []
    seen_sources = set()
    for item in sorted(
        [item for item in clean_evidence_list if isinstance(item, dict)],
        key=lambda payload: _retrieval_relevance_score(payload, _as_dict(payload).get("source")),
        reverse=True,
    ):
        score = _retrieval_relevance_score(item, _as_dict(item).get("source"))
        if score <= 0:
            continue
        source = _as_dict(item.get("source"))
        key = (str(source.get("url") or "").strip(), str(source.get("title") or "").strip())
        if key in seen_sources:
            continue
        seen_sources.add(key)
        top_sources.append(
            {
                "title": str(source.get("title") or "").strip(),
                "url": str(source.get("url") or "").strip(),
                "score": round(score, 4),
                "source_level": str(item.get("source_level") or "").strip(),
            }
        )
        if len(top_sources) >= 8:
            break
    return {
        "rerank_input_count": input_count,
        "rerank_returned_count": returned_count,
        "normalized_evidence_with_rerank_score_count": normalized_with_score,
        "evidence_with_rerank_score_count": clean_with_score,
        "analysis_ready_with_rerank_score_count": analysis_ready_with_score,
        "avg_retrieval_relevance_score": round(sum(clean_scores) / max(len(clean_scores), 1), 4) if clean_scores else 0.0,
        "top_rerank_sources": top_sources,
        "rerank_score_lost_after_normalization": max(0, returned_count - normalized_with_score),
    }


def build_filter_funnel_by_chapter(
    metadata: Dict[str, Any],
    *,
    evidence_items: Sequence[Dict[str, Any]],
    clean_evidence_list: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    funnel: Dict[str, Dict[str, Any]] = {}

    def bucket_for(item: Dict[str, Any]) -> Dict[str, Any]:
        chapter = str(
            item.get("chapter_id")
            or item.get("hypothesis_id")
            or item.get("dimension_id")
            or item.get("dimension")
            or item.get("dimension_name")
            or "unmapped"
        ).strip() or "unmapped"
        bucket = funnel.setdefault(
            chapter,
            {
                "input_count": 0,
                "kept_count": 0,
                "reasons": {
                    "topic_anchor_missing": 0,
                    "low_directness": 0,
                    "metric_invalid": 0,
                    "missing_source_ref": 0,
                    "appendix_only": 0,
                    "duplicate": 0,
                    "failed_followup_result": 0,
                },
            },
        )
        return bucket

    for item in list(evidence_items or []):
        if not isinstance(item, dict):
            continue
        bucket_for(item)["input_count"] += 1

    filtered_reasons = _as_dict(_as_dict(metadata).get("source_pool_filter_reasons"))
    if filtered_reasons:
        bucket = funnel.setdefault(
            "followup_results",
            {
                "input_count": int(_as_dict(metadata).get("source_pool_input_count") or 0),
                "kept_count": int(_as_dict(metadata).get("source_pool_usable_count") or 0),
                "reasons": {
                    "topic_anchor_missing": 0,
                    "low_directness": 0,
                    "metric_invalid": 0,
                    "missing_source_ref": 0,
                    "appendix_only": 0,
                    "duplicate": 0,
                    "failed_followup_result": 0,
                },
            },
        )
        for reason, count in filtered_reasons.items():
            normalized_reason = "failed_followup_result" if "failed" in str(reason) or "timeout" in str(reason) else str(reason)
            bucket["reasons"][normalized_reason] = int(bucket["reasons"].get(normalized_reason, 0)) + int(count or 0)

    for fact in list(clean_evidence_list or []):
        if not isinstance(fact, dict):
            continue
        bucket = bucket_for(fact)
        bucket["kept_count"] += 1
        reasons = _as_dict(bucket.get("reasons"))
        source = _as_dict(fact.get("source"))
        card = _as_dict(fact.get("evidence_card"))
        role = _canonical_role(fact.get("evidence_role"))
        directness = str(card.get("directness") or fact.get("directness") or "").strip().lower()
        metric = str(fact.get("metric") or "").strip()
        value = str(fact.get("value") or "").strip()
        if not str(source.get("url") or source.get("title") or source.get("source") or "").strip():
            reasons["missing_source_ref"] = int(reasons.get("missing_source_ref", 0)) + 1
        if fact.get("appendix_only") or role in {"clue", "appendix"}:
            reasons["appendix_only"] = int(reasons.get("appendix_only", 0)) + 1
        if directness in {"low", "indirect", "weak"} or str(fact.get("semantic_status") or "").strip().lower() == "weak_relevance":
            reasons["low_directness"] = int(reasons.get("low_directness", 0)) + 1
        if metric and not (value or _has_number(str(fact.get("content") or fact.get("fact") or ""))):
            reasons["metric_invalid"] = int(reasons.get("metric_invalid", 0)) + 1
        if str(fact.get("task_acceptance_reason") or "").strip() in {"topic_anchor_missing", "no_topic_anchor"}:
            reasons["topic_anchor_missing"] = int(reasons.get("topic_anchor_missing", 0)) + 1
        bucket["reasons"] = reasons

    duplicate_count = int(_as_dict(metadata).get("deduped_count") or 0)
    if duplicate_count:
        bucket = funnel.setdefault(
            "dedupe",
            {
                "input_count": duplicate_count,
                "kept_count": 0,
                "reasons": {
                    "topic_anchor_missing": 0,
                    "low_directness": 0,
                    "metric_invalid": 0,
                    "missing_source_ref": 0,
                    "appendix_only": 0,
                    "duplicate": duplicate_count,
                    "failed_followup_result": 0,
                },
            },
        )
        bucket["reasons"]["duplicate"] = int(bucket["reasons"].get("duplicate", 0)) + duplicate_count
    return funnel


def _source_traceability_payload(fact: Dict[str, Any]) -> Dict[str, Any]:
    source = _as_dict(fact.get("source"))
    url = str(source.get("url") or source.get("source_url") or fact.get("source_url") or "").strip()
    title = str(source.get("title") or source.get("name") or fact.get("source_title") or "").strip()
    publisher = str(source.get("publisher") or source.get("source") or fact.get("source_text") or "").strip()
    source_ref = str(fact.get("source_ref") or source.get("ref") or source.get("id") or "").strip()
    document_ref = str(
        source.get("document_id")
        or source.get("doc_id")
        or source.get("page_ref")
        or fact.get("document_id")
        or fact.get("doc_id")
        or fact.get("page_ref")
        or ""
    ).strip()
    stable_local_ref = bool(document_ref and sum(bool(item) for item in [title, publisher, source_ref]) >= 2)
    fake_source = _is_fake_or_placeholder_source(source, fact)
    has_source_ref = bool((url or stable_local_ref) and not fake_source)
    return {
        "has_source_ref": has_source_ref,
        "has_url": bool(url and not fake_source),
        "has_title": bool(title),
        "has_publisher": bool(publisher),
        "has_stable_local_ref": stable_local_ref,
        "fake_or_placeholder_source": fake_source,
        "document_ref": document_ref,
        "source_ref": source_ref,
        "source_url": url,
        "source_title": title,
        "publisher": publisher,
    }


def _source_traceability_status(traceability: Dict[str, Any]) -> str:
    if bool(_as_dict(traceability).get("fake_or_placeholder_source")):
        return "fake_or_placeholder_source"
    if bool(_as_dict(traceability).get("has_source_ref")):
        return "traceable"
    if bool(_as_dict(traceability).get("has_title")) and not bool(_as_dict(traceability).get("has_url")):
        return "title_only"
    return "missing_metadata"


def _source_registry_key(source: Dict[str, Any], traceability: Dict[str, Any]) -> str:
    url = str(source.get("url") or source.get("source_url") or traceability.get("source_url") or "").strip().lower()
    if url:
        return f"url:{url}"
    document_ref = str(
        source.get("document_id")
        or source.get("doc_id")
        or source.get("page_ref")
        or traceability.get("document_ref")
        or ""
    ).strip().lower()
    if document_ref:
        return f"doc:{document_ref}"
    title = re.sub(r"\s+", "", str(source.get("title") or traceability.get("source_title") or "").strip().lower())
    publisher = re.sub(r"\s+", "", str(source.get("publisher") or source.get("source") or traceability.get("publisher") or "").strip().lower())
    date = re.sub(r"\s+", "", str(source.get("date") or "").strip().lower())
    return f"title:{title}|{publisher}|{date}" if title else ""


def _build_source_registry(clean_evidence_list: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    registry: List[Dict[str, Any]] = []
    index_by_key: Dict[str, Dict[str, Any]] = {}
    for fact in list(clean_evidence_list or []):
        if not isinstance(fact, dict):
            continue
        source = _as_dict(fact.get("source"))
        traceability = _source_traceability_payload(fact)
        fake_source = bool(traceability.get("fake_or_placeholder_source"))
        key = _source_registry_key(source, traceability)
        if not key:
            continue
        entry = index_by_key.get(key)
        if entry is None:
            ref = str(source.get("ref") or source.get("id") or fact.get("source_ref") or f"[{len(registry) + 1}]").strip()
            if ref and not ref.startswith("[") and ref.isdigit():
                ref = f"[{ref}]"
            entry = {
                "ref": ref or f"[{len(registry) + 1}]",
                "id": str(source.get("id") or fact.get("source_id") or len(registry) + 1),
                "title": str(source.get("title") or traceability.get("source_title") or "").strip(),
                "url": str(source.get("url") or source.get("source_url") or traceability.get("source_url") or "").strip(),
                "publisher": str(source.get("publisher") or source.get("source") or traceability.get("publisher") or "").strip(),
                "source": str(source.get("source") or source.get("publisher") or traceability.get("publisher") or "").strip(),
                "date": str(source.get("date") or "").strip(),
                "source_type": str(source.get("source_type") or "").strip(),
                "source_level": "D" if fake_source else str(fact.get("source_level") or source.get("source_level") or "").strip().upper(),
                "document_id": str(source.get("document_id") or source.get("doc_id") or "").strip(),
                "page_ref": str(source.get("page_ref") or "").strip(),
                "traceability_status": _source_traceability_status(traceability),
                "has_source_ref": bool(traceability.get("has_source_ref")),
                "fake_or_placeholder_source": fake_source,
                "evidence_refs": [],
                "evidence_count": 0,
            }
            registry.append(entry)
            index_by_key[key] = entry
        evidence_ref = str(fact.get("evidence_id") or "").strip()
        if evidence_ref and evidence_ref not in entry["evidence_refs"]:
            entry["evidence_refs"].append(evidence_ref)
        entry["evidence_count"] = int(entry.get("evidence_count") or 0) + 1
        fact["source_ref"] = entry["ref"]
        source["ref"] = entry["ref"]
        source["id"] = entry["id"]
        source["traceability_status"] = entry["traceability_status"]
        source["fake_or_placeholder_source"] = bool(entry.get("fake_or_placeholder_source"))
        fact["source"] = source
    return registry


def _source_registry_summary(source_registry: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    registry = [item for item in list(source_registry or []) if isinstance(item, dict)]
    traceable = [
        item
        for item in registry
        if not bool(item.get("fake_or_placeholder_source"))
        and (
            bool(item.get("has_source_ref"))
            or str(item.get("traceability_status") or "").strip().lower() == "traceable"
            or str(item.get("url") or item.get("source_url") or "").strip()
        )
    ]
    title_only = [
        item
        for item in registry
        if str(item.get("traceability_status") or "").strip().lower() == "title_only"
        or (
            str(item.get("title") or "").strip()
            and not str(item.get("url") or item.get("source_url") or "").strip()
            and not str(item.get("document_id") or item.get("doc_id") or item.get("page_ref") or "").strip()
        )
    ]
    traceable_ab = [
        item
        for item in traceable
        if str(item.get("source_level") or "").strip().upper() in {"A", "B"}
    ]
    fake_sources = [item for item in registry if bool(item.get("fake_or_placeholder_source"))]
    return {
        "source_registry_count": len(registry),
        "traceable_source_count": len(traceable),
        "title_only_source_count": len(title_only),
        "fake_or_placeholder_source_count": len(fake_sources),
        "traceable_ab_source_count": len(traceable_ab),
        "source_registry_sample": registry[:8],
    }


def _raw_data_points_for_package(clean_evidence_list: Sequence[Dict[str, Any]], *, limit: int = 500) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    for fact in list(clean_evidence_list or []):
        if not isinstance(fact, dict):
            continue
        points.append(
            {
                "evidence_id": fact.get("evidence_id"),
                "chapter_id": fact.get("chapter_id") or fact.get("hypothesis_id") or fact.get("dimension_id"),
                "dimension": fact.get("dimension") or fact.get("dimension_name"),
                "metric": fact.get("metric"),
                "value": fact.get("value"),
                "period": fact.get("period"),
                "source_url": _as_dict(fact.get("source")).get("url"),
                "source_ref": fact.get("source_ref") or _as_dict(fact.get("source")).get("ref"),
                "source_level": fact.get("source_level"),
                "evidence": _compact_text(fact.get("fact"), max_chars=420),
            }
        )
        if len(points) >= limit:
            break
    return points


def _evidence_health_inconsistencies(health: Dict[str, Any]) -> List[Dict[str, Any]]:
    payload = _as_dict(health)
    issues: List[Dict[str, Any]] = []
    if int(_safe_float(payload.get("analysis_ready_ab_count"), 0.0)) > 0 and int(_safe_float(payload.get("source_registry_count"), 0.0)) <= 0:
        issues.append({"type": "analysis_ready_ab_without_source_registry"})
    if int(_safe_float(payload.get("analysis_ready_count"), 0.0)) > 0 and int(_safe_float(payload.get("raw_data_point_count"), 0.0)) <= 0 and int(_safe_float(payload.get("normalized_evidence_count"), 0.0)) <= 0:
        issues.append({"type": "analysis_ready_without_raw_or_normalized_evidence"})
    if int(_safe_float(payload.get("traceable_ab_source_count"), 0.0)) > int(_safe_float(payload.get("ab_source_count"), 0.0)):
        issues.append({"type": "traceable_ab_exceeds_ab_source_count"})
    return issues


def _build_evidence_health_summary(
    *,
    metadata: Dict[str, Any],
    evidence_items: Sequence[Dict[str, Any]],
    clean_evidence_list: Sequence[Dict[str, Any]],
    analysis_ready_evidence: Sequence[Dict[str, Any]],
    source_registry: Sequence[Dict[str, Any]],
    evidence_analysis_summary: Dict[str, Any],
    evidence_gap_ledger: Sequence[Dict[str, Any]],
    source_distribution: Dict[str, Any],
    readpage_coverage: Dict[str, Any],
    core_traceable_ab_by_chapter: Dict[str, int],
    publishable_evidence_gate: Dict[str, Any],
) -> Dict[str, Any]:
    registry_summary = _source_registry_summary(source_registry)
    ab_source_count = int(_safe_float(source_distribution.get("A"), 0.0)) + int(_safe_float(source_distribution.get("B"), 0.0))
    health = {
        "raw_data_point_count": int(_safe_float(metadata.get("raw_evidence_count"), 0.0)),
        "normalized_evidence_count": int(_safe_float(metadata.get("normalized_count"), 0.0)) or len(evidence_items or []),
        "evidence_count": len(evidence_items or []),
        "clean_fact_count": len(clean_evidence_list or []),
        "analysis_ready_count": len(analysis_ready_evidence or []),
        "analysis_ready_ab_count": len(
            [
                item
                for item in list(analysis_ready_evidence or [])
                if isinstance(item, dict) and str(item.get("source_level") or "").strip().upper() in {"A", "B"}
            ]
        ),
        "ab_source_count": ab_source_count,
        "traceable_ab_source_count": registry_summary.get("traceable_ab_source_count", 0),
        "core_traceable_ab_by_chapter": dict(core_traceable_ab_by_chapter or {}),
        "readpage_attempted": int(_safe_float(readpage_coverage.get("attempted"), 0.0)),
        "readpage_succeeded": int(_safe_float(readpage_coverage.get("succeeded"), 0.0)),
        "blocking_gap_count": len([item for item in list(evidence_gap_ledger or []) if str(_as_dict(item).get("severity") or "") == "blocking"]),
        "publishable_evidence_gate_passed": bool(_as_dict(publishable_evidence_gate).get("passed")),
        "source_level_distribution": dict(source_distribution or {}),
        "evidence_gap_type_distribution": _as_dict(evidence_analysis_summary).get("gap_type_distribution") or {},
        **registry_summary,
    }
    inconsistencies = _evidence_health_inconsistencies(health)
    health["inconsistent"] = bool(inconsistencies)
    health["inconsistencies"] = inconsistencies
    return health


def _metric_completeness_payload(fact: Dict[str, Any]) -> Dict[str, Any]:
    source_trace = _source_traceability_payload(fact)
    metric = str(fact.get("metric") or fact.get("metric_name") or "").strip()
    value = str(fact.get("value") or fact.get("numeric_value") or "").strip()
    text = str(fact.get("fact") or fact.get("clean_fact") or fact.get("content") or "").strip()
    period = str(fact.get("period") or _clean_fact_period(fact) or _as_dict(fact.get("source")).get("date") or "").strip()
    unit = str(fact.get("numeric_unit") or fact.get("unit") or "").strip()
    if not unit:
        unit_match = re.search(r"(亿美元|亿元|万美元|万元|亿台|万台|百万台|%|百分点|倍|家|项)", text)
        unit = unit_match.group(1) if unit_match else ""
    missing: List[str] = []
    if not metric:
        missing.append("metric")
    if not value and not _has_number(text):
        missing.append("value")
    if not unit:
        missing.append("unit")
    if not period:
        missing.append("period")
    if not source_trace.get("has_source_ref"):
        missing.append("source")
    has_metric_signal = bool(metric or value or _has_number(text))
    return {
        "has_metric_signal": has_metric_signal,
        "complete": bool(has_metric_signal and not missing),
        "missing_fields": missing,
        "metric": metric,
        "value": value,
        "unit": unit,
        "period": period,
        "has_source": bool(source_trace.get("has_source_ref")),
    }


def _evidence_analysis_contract_for(fact: Dict[str, Any]) -> Dict[str, Any]:
    card = _as_dict(fact.get("evidence_card"))
    allowed_use = str(fact.get("allowed_use") or card.get("allowed_use") or "").strip()
    source_level = str(fact.get("source_level") or card.get("source_level") or "").strip().upper()
    role = _canonical_role(fact.get("evidence_role"))
    semantic_status = str(fact.get("semantic_status") or "").strip().lower()
    appendix_only = bool(fact.get("appendix_only")) or role in {"appendix", "clue", "rejected"}
    directness = str(card.get("directness") or fact.get("directness") or "").strip().lower()
    source_trace = _source_traceability_payload(fact)
    metric_completeness = _metric_completeness_payload(fact)
    proof_role = str(fact.get("proof_role") or card.get("proof_role") or _as_dict(fact.get("search_task")).get("proof_role") or "").strip().lower()
    claim_type = str(fact.get("claim_type") or card.get("claim_type") or "").strip().lower()
    if not claim_type and infer_claim_type is not None:
        claim_type = str(infer_claim_type(fact) or "").strip().lower()
    claim_type = claim_type or "industry_analysis"
    metric_proof_gaps = [
        str(item)
        for item in _as_list(fact.get("metric_proof_gaps") or card.get("metric_proof_gaps"))
        if str(item or "").strip()
    ]
    if claim_type == "hard_metric" and not metric_proof_gaps:
        metric_proof_gaps = [
            str(item)
            for item in _as_list(metric_completeness.get("missing_fields"))
            if str(item or "").strip()
        ]
    if not allowed_use and semantic_status not in REJECTED_STATUSES:
        if source_level in {"A", "B"} and role == "core" and (proof_role in {"metric", "source_check", "filing", "official_data", "counter"} or metric_completeness.get("has_metric_signal")):
            allowed_use = "core_claim"
        elif source_level in {"A", "B"} and role in {"core", "supporting"}:
            allowed_use = "supporting"
        elif source_level == "C":
            allowed_use = "directional_signal"

    can_support: List[str] = []
    cannot_support: List[str] = []
    repair_need: List[str] = []

    if source_level in {"A", "B"} and allowed_use == "core_claim" and source_trace.get("has_source_ref"):
        can_support.extend(["core_claim", "supporting_claim", "decision_implication"])
    elif source_level in {"A", "B"} and allowed_use in {"supporting", "supporting_context"} and source_trace.get("has_source_ref"):
        can_support.append("supporting_claim")
        cannot_support.append("standalone_core_claim")
    elif source_level == "C" and allowed_use in {"directional_signal", "clue"}:
        can_support.append("directional_claim")
        if claim_type == "hard_metric":
            cannot_support.extend(["core_claim", "quantified_table", "investment_decision"])
        else:
            can_support.append("corroborated_industry_claim")
            cannot_support.extend(["standalone_core_claim", "quantified_table", "investment_decision"])
    else:
        cannot_support.extend(["core_claim", "clean_report_claim"])

    if metric_completeness.get("complete") and source_level in {"A", "B"}:
        can_support.append("quantified_table")
    elif metric_completeness.get("has_metric_signal") and not metric_completeness.get("complete"):
        cannot_support.append("quantified_table")
        repair_need.append("metric_scope_period_unit_incomplete")
    if claim_type == "hard_metric" and metric_proof_gaps:
        can_support = [item for item in can_support if item not in {"core_claim", "decision_implication", "quantified_table"}]
        if source_level in {"A", "B"} and source_trace.get("has_source_ref"):
            can_support.append("supporting_claim")
        cannot_support.extend(["complete_hard_metric_claim", "quantified_table", "investment_decision"])
        repair_need.append("metric_scope_period_unit_incomplete")

    if not source_trace.get("has_source_ref"):
        repair_need.append("source_trace_missing")
        cannot_support.append("clean_report_claim")
    if directness in {"low", "indirect", "weak", "clue"} and "core_claim" in can_support:
        can_support = [item for item in can_support if item != "core_claim"]
        can_support.append("supporting_claim")
        cannot_support.append("direct_core_claim")
    if semantic_status in REJECTED_STATUSES or role == "rejected" or appendix_only:
        if allowed_use not in {"supporting_context"}:
            can_support = [item for item in can_support if item in {"directional_claim"}]
            cannot_support.extend(["clean_report_claim", "core_claim"])
            repair_need.append("replace_with_traceable_evidence")

    if "core_claim" in can_support:
        claim_scope = "core_claim"
        proof_strength = "strong"
    elif "supporting_claim" in can_support:
        claim_scope = "supporting"
        proof_strength = "medium"
    elif "directional_claim" in can_support:
        claim_scope = "directional"
        proof_strength = "directional"
    else:
        claim_scope = "appendix_only"
        proof_strength = "weak"

    return {
        "can_support": sorted(set(can_support)),
        "cannot_support": sorted(set(cannot_support)),
        "claim_scope": claim_scope,
        "claim_type": claim_type,
        "metric_completeness": metric_completeness,
        "metric_proof_gaps": sorted(set(metric_proof_gaps)),
        "source_traceability": source_trace,
        "proof_strength": proof_strength,
        "repair_need": sorted(set(repair_need)),
    }


def _public_fact_payload(fact: Dict[str, Any]) -> Dict[str, Any]:
    analysis_contract = _evidence_analysis_contract_for(fact)
    return {
        "evidence_id": str(fact.get("evidence_id") or "").strip(),
        "dimension": str(fact.get("dimension") or "").strip(),
        "fact": str(fact.get("fact") or "").strip(),
        "metric": str(fact.get("metric") or "").strip(),
        "value": str(fact.get("value") or "").strip(),
        "period": str(fact.get("period") or "").strip(),
        "source": _as_dict(fact.get("source")),
        "confidence": _clip(fact.get("confidence"), 0.0),
        "s_grade": bool(fact.get("s_grade")),
        "conflict_flag": bool(fact.get("conflict_flag")),
        "source_level": str(fact.get("source_level") or "").strip(),
        "source_tier": str(fact.get("source_tier") or _as_dict(fact.get("evidence_card")).get("source_tier") or "").strip(),
        "evidence_role": _canonical_role(fact.get("evidence_role")),
        "allowed_use": str(fact.get("allowed_use") or _as_dict(fact.get("evidence_card")).get("allowed_use") or "").strip(),
        "evidence_fit_score": fact.get("evidence_fit_score") or _as_dict(fact.get("evidence_card")).get("evidence_fit_score"),
        "metric_proof_gaps": _as_list(fact.get("metric_proof_gaps") or _as_dict(fact.get("evidence_card")).get("metric_proof_gaps")),
        "analysis_readiness": str(fact.get("analysis_readiness") or _as_dict(fact.get("evidence_card")).get("analysis_readiness") or "").strip(),
        "evidence_card": _as_dict(fact.get("evidence_card")),
        "semantic_status": str(fact.get("semantic_status") or "").strip(),
        "semantic_reason": str(fact.get("semantic_reason") or "").strip(),
        "metric_kind": str(fact.get("metric_kind") or "").strip(),
        "task_id": str(fact.get("task_id") or "").strip(),
        "dimension_id": str(fact.get("dimension_id") or "").strip(),
        "dimension_name": str(fact.get("dimension_name") or "").strip(),
        "evidence_goal": str(fact.get("evidence_goal") or "").strip(),
        "task_relevance_score": fact.get("task_relevance_score"),
        "retrieval_relevance_score": _retrieval_relevance_score(fact, fact.get("source")),
        "web_final_score": fact.get("web_final_score") or _as_dict(fact.get("source")).get("web_final_score"),
        "web_rerank_score": fact.get("web_rerank_score") or _as_dict(fact.get("source")).get("web_rerank_score"),
        "web_rerank_rank": fact.get("web_rerank_rank") or _as_dict(fact.get("source")).get("web_rerank_rank"),
        "task_term_score": fact.get("task_term_score") or _as_dict(fact.get("source")).get("task_term_score"),
        "task_accepted": fact.get("task_accepted"),
        "task_acceptance_reason": str(fact.get("task_acceptance_reason") or "").strip(),
        "appendix_only": bool(fact.get("appendix_only")),
        "enterprise_usable": bool(fact.get("enterprise_usable")),
        "followup_seed": bool(fact.get("followup_seed")),
        "can_support_claim_if_corrobated": bool(fact.get("can_support_claim_if_corrobated")),
        "usage_tier": str(fact.get("usage_tier") or "").strip(),
        **analysis_contract,
        "analysis_input": _as_dict(fact.get("analysis_input")) or _analysis_input_for_evidence(fact, fact_text=fact.get("fact")),
    }


def _chapter_key_for_fact(fact: Dict[str, Any]) -> str:
    return str(
        fact.get("chapter_id")
        or fact.get("hypothesis_id")
        or fact.get("dimension_id")
        or fact.get("dimension")
        or fact.get("dimension_name")
        or "unmapped"
    ).strip() or "unmapped"


def _gap_terms_from_text(*values: Any, max_terms: int = 6) -> List[str]:
    terms: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        parts = [part.strip() for part in re.split(r"[,，;；、\s]+", text) if part.strip()]
        if not parts:
            parts = [text]
        for part in parts:
            cleaned = part[:36]
            if cleaned and cleaned not in terms:
                terms.append(cleaned)
            if len(terms) >= max_terms:
                return terms
    return terms


def _evidence_analysis_by_chapter(clean_evidence_list: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    chapters: Dict[str, Dict[str, Any]] = {}
    for fact in list(clean_evidence_list or []):
        if not isinstance(fact, dict):
            continue
        chapter_id = _chapter_key_for_fact(fact)
        payload = _public_fact_payload(fact)
        contract = {
            "can_support": _as_list(payload.get("can_support")),
            "claim_scope": str(payload.get("claim_scope") or ""),
            "claim_type": str(payload.get("claim_type") or ""),
            "metric_completeness": _as_dict(payload.get("metric_completeness")),
            "source_traceability": _as_dict(payload.get("source_traceability")),
            "proof_strength": str(payload.get("proof_strength") or ""),
            "repair_need": _as_list(payload.get("repair_need")),
        }
        bucket = chapters.setdefault(
            chapter_id,
            {
                "chapter_id": chapter_id,
                "chapter_title": str(fact.get("dimension") or fact.get("dimension_name") or chapter_id),
                "evidence_count": 0,
                "core_ab_source_count": 0,
                "core_traceable_ab_source_count": 0,
                "core_traceable_ab_source_keys": set(),
                "metric_ready_count": 0,
                "counter_signal_count": 0,
                "traceable_counter_source_count": 0,
                "traceable_counter_source_keys": set(),
                "case_signal_count": 0,
                "claim_ready_evidence_count": 0,
                "directional_only_count": 0,
                "directional_c_claim_count": 0,
                "directional_c_source_keys": set(),
                "source_trace_missing_count": 0,
                "metric_incomplete_count": 0,
                "hard_metric_incomplete_count": 0,
                "evidence_gap_types": [],
                "advisory_gap_types": [],
                "sample_evidence_refs": [],
            },
        )
        bucket["evidence_count"] += 1
        level = str(payload.get("source_level") or "").strip().upper()
        can_support = set(str(item) for item in _as_list(contract.get("can_support")))
        has_source = bool(_as_dict(contract.get("source_traceability")).get("has_source_ref"))
        if level in {"A", "B"} and has_source and can_support.intersection({"core_claim", "supporting_claim"}):
            bucket["core_ab_source_count"] += 1
            source_key = _source_registry_key(_as_dict(payload.get("source")), _as_dict(contract.get("source_traceability"))) or _source_identity(payload) or str(payload.get("evidence_id") or "")
            if source_key:
                bucket["core_traceable_ab_source_keys"].add(source_key)
        if bool(_as_dict(contract.get("metric_completeness")).get("complete")) and level in {"A", "B"}:
            bucket["metric_ready_count"] += 1
        role_text = " ".join(
            str(value or "").lower()
            for value in [
                fact.get("proof_role"),
                fact.get("evidence_role"),
                _as_dict(fact.get("evidence_card")).get("proof_role"),
                fact.get("metric_kind"),
                fact.get("semantic_reason"),
            ]
        )
        if any(token in role_text for token in ["counter", "risk", "反证", "风险"]):
            bucket["counter_signal_count"] += 1
            if has_source:
                source_key = _source_registry_key(_as_dict(payload.get("source")), _as_dict(contract.get("source_traceability"))) or _source_identity(payload) or str(payload.get("evidence_id") or "")
                if source_key:
                    bucket["traceable_counter_source_keys"].add(source_key)
        if any(token in role_text for token in ["case", "customer", "procurement", "order", "tender", "客户", "案例", "采购", "中标", "订单"]):
            bucket["case_signal_count"] += 1
        if can_support.intersection({"core_claim", "supporting_claim"}):
            bucket["claim_ready_evidence_count"] += 1
        if contract.get("claim_scope") == "directional":
            bucket["directional_only_count"] += 1
        if level == "C" and has_source and "corroborated_industry_claim" in can_support:
            bucket["directional_c_claim_count"] += 1
            source_key = _source_identity(payload) or _source_identity(fact)
            if source_key:
                bucket["directional_c_source_keys"].add(source_key)
        if not has_source:
            bucket["source_trace_missing_count"] += 1
        if "metric_scope_period_unit_incomplete" in _as_list(contract.get("repair_need")):
            bucket["metric_incomplete_count"] += 1
            if str(contract.get("claim_type") or "").strip().lower() == "hard_metric":
                bucket["hard_metric_incomplete_count"] += 1
        ref = str(payload.get("evidence_id") or _as_dict(payload.get("source")).get("ref") or "").strip()
        if ref and len(bucket["sample_evidence_refs"]) < 8:
            bucket["sample_evidence_refs"].append(ref)

    for bucket in chapters.values():
        gaps: List[str] = []
        advisory: List[str] = []
        directional_c_sources = len(bucket.get("directional_c_source_keys") or set())
        claim_ready = int(bucket.get("claim_ready_evidence_count") or 0)
        directional_only = int(bucket.get("directional_only_count") or 0)
        core_ab = int(bucket.get("core_ab_source_count") or 0)
        balanced_directional_ready = bool(
            not _strict_quality_mode()
            and core_ab <= 0
            and claim_ready <= 0
            and directional_only > 0
            and directional_c_sources >= 2
        )
        bucket["directional_c_distinct_source_count"] = directional_c_sources
        bucket["core_traceable_ab_source_count"] = len(bucket.get("core_traceable_ab_source_keys") or set())
        bucket["traceable_counter_source_count"] = len(bucket.get("traceable_counter_source_keys") or set())
        bucket["balanced_directional_ready"] = balanced_directional_ready
        bucket.pop("directional_c_source_keys", None)
        bucket.pop("core_traceable_ab_source_keys", None)
        bucket.pop("traceable_counter_source_keys", None)
        if core_ab <= 0:
            if balanced_directional_ready:
                advisory.append("directional_corroborated_no_ab")
            else:
                gaps.append("insufficient_ab_sources")
        if int(bucket.get("metric_incomplete_count") or 0) > 0 and int(bucket.get("metric_ready_count") or 0) <= 0:
            if _strict_quality_mode() or int(bucket.get("hard_metric_incomplete_count") or 0) > 0:
                gaps.append("metric_scope_period_unit_incomplete")
            else:
                advisory.append("metric_scope_period_unit_incomplete")
        if int(bucket.get("source_trace_missing_count") or 0) > 0:
            gaps.append("source_trace_missing")
        if claim_ready <= 0 and directional_only > 0 and not balanced_directional_ready:
            gaps.append("directional_only_evidence")
        if int(bucket.get("counter_signal_count") or 0) <= 0:
            gaps.append("counter_evidence_missing")
        bucket["evidence_gap_types"] = gaps
        bucket["advisory_gap_types"] = advisory
    return chapters


def _gap_payload(
    *,
    chapter: Dict[str, Any],
    gap_type: str,
    severity: str,
    required_proof_role: str,
    required_fields: Sequence[str],
    lane_targets: Sequence[str],
    current_refs: Sequence[str],
    reason: str,
    root_cause: str = "",
    affected_metric_fields: Optional[Sequence[str]] = None,
    repair_priority: str = "",
    can_openai_repair: Optional[bool] = None,
) -> Dict[str, Any]:
    chapter_id = str(chapter.get("chapter_id") or "unmapped")
    query_terms = _gap_terms_from_text(chapter.get("chapter_title"), gap_type)
    return {
        "gap_id": _hash_id("evidence_gap", chapter_id, gap_type)[:16],
        "chapter_id": chapter_id,
        "claim_id": "",
        "gap_type": gap_type,
        "type": gap_type,
        "severity": severity,
        "required_proof_role": required_proof_role,
        "proof_role": required_proof_role,
        "required_source_level": ["A", "B"] if gap_type not in {"counter_evidence_missing"} else ["A", "B", "C"],
        "required_fields": list(required_fields),
        "current_evidence_refs": list(current_refs or []),
        "why_current_evidence_insufficient": reason,
        "repair_route": "evidence_search" if severity in {"blocking", "advisory"} else "rewrite",
        "query_terms": query_terms,
        "topic_terms": query_terms,
        "lane_targets": list(lane_targets),
        "targets_gap": reason,
        "source": "evidence_gap_ledger",
        "root_cause": root_cause or gap_type,
        "affected_metric_fields": list(affected_metric_fields or []),
        "repair_priority": repair_priority or ("high" if severity == "blocking" else "medium"),
        "can_openai_repair": bool(can_openai_repair) if can_openai_repair is not None else severity in {"blocking", "advisory"},
    }


def _evidence_gap_ledger_from_analysis(chapter_analysis: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    ledger: List[Dict[str, Any]] = []
    for chapter in chapter_analysis.values():
        refs = _as_list(chapter.get("sample_evidence_refs"))
        gaps = set(str(item) for item in _as_list(chapter.get("evidence_gap_types")))
        if "insufficient_ab_sources" in gaps:
            ledger.append(
                _gap_payload(
                    chapter=chapter,
                    gap_type="insufficient_ab_sources",
                    severity="blocking",
                    required_proof_role="source_check",
                    required_fields=["source"],
                    lane_targets=["official_data", "filing_company", "market_research"],
                    current_refs=refs,
                    reason="核心章节缺少可追溯的 A/B 来源支撑。",
                    root_cause="ab_source_insufficient",
                    repair_priority="high",
                )
            )
        if "metric_scope_period_unit_incomplete" in gaps:
            ledger.append(
                _gap_payload(
                    chapter=chapter,
                    gap_type="metric_scope_period_unit_incomplete",
                    severity="blocking",
                    required_proof_role="metric",
                    required_fields=["metric", "value", "unit", "period", "source"],
                    lane_targets=["official_data", "market_research", "filing_company"],
                    current_refs=refs,
                    reason="存在指标信号但 metric/value/unit/period/source 不完整，不能进入量化表格或强结论。",
                    root_cause="metric_incomplete",
                    affected_metric_fields=["metric", "value", "unit", "period", "source"],
                    repair_priority="high",
                )
            )
        if "source_trace_missing" in gaps:
            ledger.append(
                _gap_payload(
                    chapter=chapter,
                    gap_type="source_trace_missing",
                    severity="blocking",
                    required_proof_role="source_check",
                    required_fields=["source"],
                    lane_targets=["official_data", "market_research"],
                    current_refs=refs,
                    reason="部分证据缺少 URL、标题、发布方或 source_ref，无法进入 Clean report。",
                    root_cause="source_trace_missing",
                    repair_priority="high",
                )
            )
        if "directional_only_evidence" in gaps:
            ledger.append(
                _gap_payload(
                    chapter=chapter,
                    gap_type="directional_only_evidence",
                    severity="blocking",
                    required_proof_role="source_check",
                    required_fields=["source"],
                    lane_targets=["official_data", "market_research"],
                    current_refs=refs,
                    reason="当前证据只能支撑方向性判断，不能支撑核心结论。",
                    root_cause="ab_source_insufficient",
                    repair_priority="high",
                )
            )
        if "counter_evidence_missing" in gaps:
            ledger.append(
                _gap_payload(
                    chapter=chapter,
                    gap_type="counter_evidence_missing",
                    severity="advisory",
                    required_proof_role="counter",
                    required_fields=["source"],
                    lane_targets=["news_event", "market_research"],
                    current_refs=refs,
                    reason="章节缺少反证、风险边界或失败案例，建议补充以提升决策质量。",
                    root_cause="counter_missing",
                    repair_priority="medium",
                )
            )
    return ledger


def _evidence_analysis_summary(chapter_analysis: Dict[str, Dict[str, Any]], ledger: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    gap_dist: Dict[str, int] = {}
    severity_dist: Dict[str, int] = {}
    chapter_advisory_dist: Dict[str, int] = {}
    for gap in list(ledger or []):
        payload = _as_dict(gap)
        gap_type = str(payload.get("gap_type") or payload.get("type") or "unknown")
        severity = str(payload.get("severity") or "unknown")
        gap_dist[gap_type] = gap_dist.get(gap_type, 0) + 1
        severity_dist[severity] = severity_dist.get(severity, 0) + 1
    for chapter in chapter_analysis.values():
        for gap_type in _as_list(_as_dict(chapter).get("advisory_gap_types")):
            key = str(gap_type or "unknown")
            chapter_advisory_dist[key] = chapter_advisory_dist.get(key, 0) + 1
    return {
        "chapter_count": len(chapter_analysis),
        "total_evidence_count": sum(int(_safe_float(item.get("evidence_count"), 0.0)) for item in chapter_analysis.values()),
        "total_core_ab_source_count": sum(int(_safe_float(item.get("core_ab_source_count"), 0.0)) for item in chapter_analysis.values()),
        "total_core_traceable_ab_source_count": sum(int(_safe_float(item.get("core_traceable_ab_source_count"), 0.0)) for item in chapter_analysis.values()),
        "total_metric_ready_count": sum(int(_safe_float(item.get("metric_ready_count"), 0.0)) for item in chapter_analysis.values()),
        "total_counter_signal_count": sum(int(_safe_float(item.get("counter_signal_count"), 0.0)) for item in chapter_analysis.values()),
        "total_traceable_counter_source_count": sum(int(_safe_float(item.get("traceable_counter_source_count"), 0.0)) for item in chapter_analysis.values()),
        "total_case_signal_count": sum(int(_safe_float(item.get("case_signal_count"), 0.0)) for item in chapter_analysis.values()),
        "total_claim_ready_evidence_count": sum(int(_safe_float(item.get("claim_ready_evidence_count"), 0.0)) for item in chapter_analysis.values()),
        "total_directional_only_count": sum(int(_safe_float(item.get("directional_only_count"), 0.0)) for item in chapter_analysis.values()),
        "total_directional_c_distinct_source_count": sum(int(_safe_float(item.get("directional_c_distinct_source_count"), 0.0)) for item in chapter_analysis.values()),
        "balanced_directional_ready_chapter_count": sum(1 for item in chapter_analysis.values() if bool(item.get("balanced_directional_ready"))),
        "blocking_gap_count": severity_dist.get("blocking", 0),
        "advisory_gap_count": severity_dist.get("advisory", 0),
        "gap_type_distribution": gap_dist,
        "chapter_advisory_gap_type_distribution": chapter_advisory_dist,
        "severity_distribution": severity_dist,
    }


def _distribution(values: Iterable[Any]) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for value in values:
        key = str(value or "unknown").strip() or "unknown"
        result[key] = result.get(key, 0) + 1
    return result


def _core_ab_by_chapter(chapter_analysis: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    return {
        str(chapter_id): int(_safe_float(_as_dict(payload).get("core_ab_source_count"), 0.0))
        for chapter_id, payload in chapter_analysis.items()
    }


def _core_traceable_ab_by_chapter(chapter_analysis: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    return {
        str(chapter_id): int(
            _safe_float(
                _as_dict(payload).get("core_traceable_ab_source_count"),
                0.0,
            )
        )
        for chapter_id, payload in chapter_analysis.items()
    }


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", "none", "null", ""}:
        return False
    return default


def _chapter_contracts_from_research_plan(research_plan: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    plan = _as_dict(research_plan)
    contracts: Dict[str, Dict[str, Any]] = {}
    for index, raw in enumerate(_as_list(plan.get("chapters")), start=1):
        chapter = _as_dict(raw)
        chapter_id = str(chapter.get("chapter_id") or chapter.get("id") or f"ch_{index:02d}").strip()
        if not chapter_id:
            continue
        key_chapter = _as_bool(chapter.get("key_chapter"), index <= 2)
        contract = dict(_as_dict(chapter.get("chapter_evidence_contract")))
        required_ab = int(
            _safe_float(
                contract.get("required_traceable_ab")
                or chapter.get("required_traceable_ab")
                or chapter.get("min_ab_sources")
                or (
                    _report_env_int("REPORT_KEY_CHAPTER_MIN_TRACEABLE_AB", 3, min_value=1, max_value=10)
                    if key_chapter
                    else _report_env_int("REPORT_MIN_CORE_AB_SOURCES_PER_CHAPTER", 2, min_value=1, max_value=10)
                ),
                0.0,
            )
        )
        required_ab = max(
            _report_env_int("REPORT_KEY_CHAPTER_MIN_TRACEABLE_AB", 3, min_value=1, max_value=10)
            if key_chapter
            else _report_env_int("REPORT_MIN_CORE_AB_SOURCES_PER_CHAPTER", 2, min_value=1, max_value=10),
            required_ab,
        )
        required_roles = [
            str(item or "").strip().lower()
            for item in _as_list(contract.get("required_proof_roles"))
            if str(item or "").strip()
        ]
        if not required_roles:
            required_roles = ["metric", "source_check", "case"] if key_chapter else ["source_check", "case"]
        contracts[chapter_id] = {
            "chapter_id": chapter_id,
            "chapter_title": str(chapter.get("chapter_title") or chapter.get("title") or chapter_id).strip(),
            "key_chapter": key_chapter,
            "required_traceable_ab": required_ab,
            "required_proof_roles": required_roles,
        }
    return contracts


def _default_chapter_contracts(chapter_analysis: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    contracts: Dict[str, Dict[str, Any]] = {}
    for index, (chapter_id, payload) in enumerate(chapter_analysis.items(), start=1):
        key_chapter = index <= 2
        contracts[str(chapter_id)] = {
            "chapter_id": str(chapter_id),
            "chapter_title": str(_as_dict(payload).get("chapter_title") or chapter_id),
            "key_chapter": key_chapter,
            "required_traceable_ab": _report_env_int("REPORT_KEY_CHAPTER_MIN_TRACEABLE_AB", 3, min_value=1, max_value=10)
            if key_chapter
            else _report_env_int("REPORT_MIN_CORE_AB_SOURCES_PER_CHAPTER", 2, min_value=1, max_value=10),
            "required_proof_roles": ["metric", "source_check", "case"] if key_chapter else ["source_check", "case"],
        }
    return contracts


def _readpage_coverage_from_evidence(clean_evidence_list: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    attempted = 0
    succeeded = 0
    for fact in clean_evidence_list:
        if not isinstance(fact, dict):
            continue
        source = _as_dict(fact.get("source"))
        if fact.get("auto_readpage") or source.get("auto_readpage") or source.get("readpage_priority"):
            attempted += 1
            if str(fact.get("fact") or fact.get("clean_fact") or fact.get("content") or "").strip():
                succeeded += 1
    return {"attempted": attempted, "succeeded": succeeded}


def _source_pool_readpage_coverage(source_pool: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    attempted = 0
    succeeded = 0
    for raw in list(source_pool or []):
        item = _as_dict(raw)
        if not item:
            continue
        metadata = _as_dict(item.get("metadata"))
        auto_meta = _as_dict(metadata.get("auto_readpage") or item.get("auto_readpage"))
        page_attempted = 0
        page_succeeded = 0
        for page in _as_list(item.get("page_results")):
            if not isinstance(page, dict):
                continue
            if page.get("auto_readpage") or page.get("readpage_priority") or page.get("url") or page.get("source_url"):
                page_attempted += 1
                if str(
                    page.get("content")
                    or page.get("markdown")
                    or page.get("text")
                    or page.get("summary")
                    or ""
                ).strip():
                    page_succeeded += 1
        attempted += max(page_attempted, int(_safe_float(auto_meta.get("attempted"), 0.0)))
        succeeded += max(page_succeeded, int(_safe_float(auto_meta.get("succeeded"), 0.0)))
    return {"attempted": attempted, "succeeded": succeeded}


def _merge_readpage_coverage(primary: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, int]:
    return {
        "attempted": max(int(_safe_float(primary.get("attempted"), 0.0)), int(_safe_float(extra.get("attempted"), 0.0))),
        "succeeded": max(int(_safe_float(primary.get("succeeded"), 0.0)), int(_safe_float(extra.get("succeeded"), 0.0))),
    }


def _publishable_evidence_gate(
    *,
    chapter_analysis: Dict[str, Dict[str, Any]],
    readpage_coverage: Dict[str, Any],
    evidence_gap_ledger: Sequence[Dict[str, Any]],
    core_traceable_ab_by_chapter: Optional[Dict[str, int]] = None,
    chapter_contracts: Optional[Dict[str, Dict[str, Any]]] = None,
    evidence_analysis_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    min_core_ab = _report_env_int("REPORT_MIN_CORE_AB_SOURCES_PER_CHAPTER", 2, min_value=1, max_value=20)
    key_min_ab = _report_env_int("REPORT_KEY_CHAPTER_MIN_TRACEABLE_AB", 3, min_value=1, max_value=20)
    min_counter_sources = _report_env_int("REPORT_MIN_COUNTER_SOURCES_PER_REPORT", 2, min_value=0, max_value=20)
    require_readpage = _report_env_flag("REPORT_REQUIRE_READPAGE_EVIDENCE", True)
    core_ab = _core_ab_by_chapter(chapter_analysis)
    traceable_core_ab = dict(core_traceable_ab_by_chapter or _core_traceable_ab_by_chapter(chapter_analysis))
    contracts = dict(chapter_contracts or _default_chapter_contracts(chapter_analysis))
    weak_chapters = [
        {
            "chapter_id": chapter_id,
            "chapter_title": _as_dict(contracts.get(chapter_id)).get("chapter_title") or _as_dict(chapter_analysis.get(chapter_id)).get("chapter_title"),
            "key_chapter": _as_bool(_as_dict(contracts.get(chapter_id)).get("key_chapter")),
            "core_traceable_ab_source_count": int(_safe_float(traceable_core_ab.get(chapter_id), 0.0)),
            "core_ab_source_count": int(_safe_float(core_ab.get(chapter_id), 0.0)),
            "required": int(_safe_float(_as_dict(contracts.get(chapter_id)).get("required_traceable_ab"), key_min_ab if _as_bool(_as_dict(contracts.get(chapter_id)).get("key_chapter")) else min_core_ab)),
        }
        for chapter_id in sorted(set(contracts) | set(traceable_core_ab))
        if int(_safe_float(traceable_core_ab.get(chapter_id), 0.0))
        < int(_safe_float(_as_dict(contracts.get(chapter_id)).get("required_traceable_ab"), key_min_ab if _as_bool(_as_dict(contracts.get(chapter_id)).get("key_chapter")) else min_core_ab))
    ]
    blocking_gap_count = len([item for item in evidence_gap_ledger if str(_as_dict(item).get("severity") or "") == "blocking"])
    readpage_succeeded = int(_safe_float(readpage_coverage.get("succeeded"), 0.0))
    summary = _as_dict(evidence_analysis_summary)
    traceable_counter_count = int(_safe_float(summary.get("total_traceable_counter_source_count"), 0.0))
    blocking_reasons: List[Dict[str, Any]] = []
    if weak_chapters:
        blocking_reasons.append({"type": "chapter_core_ab_below_minimum", "chapters": weak_chapters[:12]})
    if min_counter_sources > 0 and traceable_counter_count < min_counter_sources:
        blocking_reasons.append(
            {
                "type": "report_counter_sources_below_minimum",
                "actual": traceable_counter_count,
                "required": min_counter_sources,
                "scope": os.getenv("REPORT_COUNTER_SCOPE", "report_level") or "report_level",
            }
        )
    if require_readpage and readpage_succeeded <= 0:
        blocking_reasons.append({"type": "readpage_evidence_missing", "readpage_coverage": readpage_coverage})
    if blocking_gap_count > 0:
        blocking_reasons.append({"type": "blocking_evidence_gaps", "count": blocking_gap_count})
    return {
        "passed": not blocking_reasons,
        "blocking_reasons": blocking_reasons,
        "min_core_ab_sources_per_chapter": min_core_ab,
        "key_chapter_min_traceable_ab": key_min_ab,
        "min_counter_sources_per_report": min_counter_sources,
        "require_readpage": require_readpage,
        "core_traceable_ab_by_chapter": traceable_core_ab,
        "chapter_contracts": contracts,
    }


def _delivery_policy() -> str:
    raw = os.getenv("REPORT_DELIVERY_POLICY") or "three_tier"
    policy = str(raw or "").strip().lower()
    if policy in {"three_tier", "three-tier", "tiered", "balanced"}:
        return "three_tier"
    if policy in {"strict", "block", "diagnostic_only"}:
        return "strict"
    if policy in {"draft_first", "always_draft", "loose"}:
        return "draft_first"
    return "three_tier"


def _delivery_gate(
    *,
    publishable_gate: Dict[str, Any],
    evidence_count: int,
    clean_fact_count: int,
    analysis_ready_count: int,
    analysis_ready_ab_count: int,
    source_distribution: Dict[str, Any],
    readpage_coverage: Dict[str, Any],
    core_ab_by_chapter: Dict[str, int],
    evidence_gap_ledger: Sequence[Dict[str, Any]],
    traceable_ab_source_count: Optional[int] = None,
    core_traceable_ab_by_chapter: Optional[Dict[str, int]] = None,
    evidence_health_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    policy = _delivery_policy()
    publishable = bool(_as_dict(publishable_gate).get("passed"))
    ab_source_count = int(_safe_float(source_distribution.get("A"), 0.0)) + int(_safe_float(source_distribution.get("B"), 0.0))
    traceable_ab_count = int(_safe_float(traceable_ab_source_count, 0.0))
    traceable_by_chapter = dict(core_traceable_ab_by_chapter or {})
    health = _as_dict(evidence_health_summary)
    if health.get("inconsistent"):
        publishable = False
    readpage_succeeded = int(_safe_float(readpage_coverage.get("succeeded"), 0.0))
    blocking_reasons = list(_as_list(_as_dict(publishable_gate).get("blocking_reasons")))
    review_reasons = list(blocking_reasons)
    severe_reasons: List[Dict[str, Any]] = []
    if evidence_count <= 0 or clean_fact_count <= 0:
        severe_reasons.append({"type": "no_clean_evidence", "evidence_count": evidence_count, "clean_fact_count": clean_fact_count})
    if analysis_ready_count <= 0 and ab_source_count <= 0 and readpage_succeeded <= 0:
        severe_reasons.append(
            {
                "type": "no_usable_evidence_signal",
                "analysis_ready_count": analysis_ready_count,
                "ab_source_count": ab_source_count,
                "readpage_succeeded": readpage_succeeded,
            }
        )
    if not core_ab_by_chapter and analysis_ready_ab_count <= 0 and ab_source_count <= 0:
        severe_reasons.append({"type": "no_chapter_ab_signal"})
    if health.get("inconsistent"):
        severe_reasons.append({"type": "evidence_health_summary_inconsistent", "details": _as_list(health.get("inconsistencies"))})
    blocking_gap_count = len([item for item in evidence_gap_ledger if str(_as_dict(item).get("severity") or "") == "blocking"])
    if blocking_gap_count and analysis_ready_count <= 0:
        severe_reasons.append({"type": "blocking_gaps_without_analysis_ready_evidence", "count": blocking_gap_count})

    if publishable:
        tier = "publishable_clean"
        review_reasons = []
        severe_reasons = []
    elif policy == "strict":
        tier = "diagnostic_only"
    elif severe_reasons and policy != "draft_first":
        tier = "diagnostic_only"
    else:
        tier = "limited_review_draft"

    return {
        "policy": policy,
        "tier": tier,
        "publishable": tier == "publishable_clean",
        "draft_allowed": tier in {"publishable_clean", "limited_review_draft"},
        "diagnostic_only": tier == "diagnostic_only",
        "exhausted": False,
        "blocking_reasons": severe_reasons if tier == "diagnostic_only" else [],
        "review_reasons": review_reasons if tier == "limited_review_draft" else [],
        "publishable_gate": publishable_gate,
        "evidence_signal": {
            "evidence_count": evidence_count,
            "clean_fact_count": clean_fact_count,
            "analysis_ready_count": analysis_ready_count,
            "analysis_ready_ab_count": analysis_ready_ab_count,
            "ab_source_count": ab_source_count,
            "traceable_ab_source_count": traceable_ab_count,
            "core_traceable_ab_by_chapter": traceable_by_chapter,
            "readpage_succeeded": readpage_succeeded,
            "evidence_health_summary_inconsistent": bool(health.get("inconsistent")),
        },
    }


def _build_evidence_preflight_summary(
    *,
    chapter_analysis: Dict[str, Dict[str, Any]],
    chapter_contracts: Dict[str, Dict[str, Any]],
    evidence_health_summary: Dict[str, Any],
    evidence_analysis_summary: Dict[str, Any],
    evidence_gap_ledger: Sequence[Dict[str, Any]],
    publishable_evidence_gate: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    health = _as_dict(evidence_health_summary)
    analysis_summary = _as_dict(evidence_analysis_summary)
    contracts = dict(chapter_contracts or _default_chapter_contracts(chapter_analysis))
    by_chapter: Dict[str, Dict[str, Any]] = {}
    clean_blocking_reasons: List[Dict[str, Any]] = []
    for chapter_id in sorted(set(contracts) | set(chapter_analysis)):
        contract = _as_dict(contracts.get(chapter_id))
        payload = _as_dict(chapter_analysis.get(chapter_id))
        required_ab = int(_safe_float(contract.get("required_traceable_ab"), _report_env_int("REPORT_MIN_CORE_AB_SOURCES_PER_CHAPTER", 2, min_value=1, max_value=20)))
        actual_ab = int(_safe_float(payload.get("core_traceable_ab_source_count"), 0.0))
        missing_roles: List[str] = []
        if actual_ab < required_ab:
            missing_roles.append("source_check")
        required_roles = {str(item or "").strip().lower() for item in _as_list(contract.get("required_proof_roles")) if str(item or "").strip()}
        if "metric" in required_roles and int(_safe_float(payload.get("metric_ready_count"), 0.0)) <= 0:
            missing_roles.append("metric")
        if "case" in required_roles and int(_safe_float(payload.get("case_signal_count"), 0.0)) <= 0:
            missing_roles.append("case")
        if missing_roles:
            clean_blocking_reasons.append(
                {
                    "type": "chapter_evidence_contract_gap",
                    "chapter_id": chapter_id,
                    "chapter_title": contract.get("chapter_title") or payload.get("chapter_title") or chapter_id,
                    "key_chapter": _as_bool(contract.get("key_chapter")),
                    "required_traceable_ab": required_ab,
                    "actual_traceable_ab": actual_ab,
                    "missing_proof_roles": sorted(set(missing_roles)),
                    "repairable_by_openai": True,
                }
            )
        by_chapter[chapter_id] = {
            "chapter_id": chapter_id,
            "chapter_title": contract.get("chapter_title") or payload.get("chapter_title") or chapter_id,
            "key_chapter": _as_bool(contract.get("key_chapter")),
            "required_traceable_ab": required_ab,
            "actual_traceable_ab": actual_ab,
            "metric_ready_count": int(_safe_float(payload.get("metric_ready_count"), 0.0)),
            "case_signal_count": int(_safe_float(payload.get("case_signal_count"), 0.0)),
            "missing_proof_roles": sorted(set(missing_roles)),
            "repairable_by_openai": bool(missing_roles),
        }
    min_counter = _report_env_int("REPORT_MIN_COUNTER_SOURCES_PER_REPORT", 2, min_value=0, max_value=20)
    actual_counter = int(_safe_float(analysis_summary.get("total_traceable_counter_source_count"), 0.0))
    if min_counter > 0 and actual_counter < min_counter:
        clean_blocking_reasons.append(
            {
                "type": "report_counter_sources_below_minimum",
                "chapter_id": "report",
                "proof_role": "counter",
                "required": min_counter,
                "actual": actual_counter,
                "repairable_by_openai": True,
            }
        )
    for reason in _as_list(_as_dict(publishable_evidence_gate).get("blocking_reasons")):
        payload = _as_dict(reason)
        reason_type = str(payload.get("type") or "").strip()
        if reason_type in {"chapter_core_ab_below_minimum", "report_counter_sources_below_minimum"}:
            continue
        clean_blocking_reasons.append({**payload, "repairable_by_openai": reason_type in {"readpage_evidence_missing", "blocking_evidence_gaps"}})
    analysis_ready_count = int(_safe_float(health.get("analysis_ready_count"), 0.0))
    clean_fact_count = int(_safe_float(health.get("clean_fact_count"), 0.0))
    health_inconsistent = bool(health.get("inconsistent"))
    repairable = [item for item in clean_blocking_reasons if bool(_as_dict(item).get("repairable_by_openai"))]
    diagnostic_only = bool(health_inconsistent or (analysis_ready_count <= 0 and clean_fact_count <= 0))
    return {
        "enabled": _report_env_flag("REPORT_ENABLE_EVIDENCE_PREFLIGHT", True),
        "ready_for_clean_writer": bool(_as_dict(publishable_evidence_gate).get("passed")) and not health_inconsistent,
        "needs_gap_repair": bool(repairable) and not diagnostic_only,
        "review_draft_allowed": bool(not diagnostic_only and (analysis_ready_count > 0 or clean_fact_count > 0)),
        "diagnostic_only": diagnostic_only,
        "by_chapter": by_chapter,
        "clean_blocking_reasons": clean_blocking_reasons,
        "repair_attempts": _as_dict(metadata).get("gap_attempt_summary") or {},
        "openai_web_repair_summary": _as_dict(metadata).get("openai_web_search_summary") or {},
        "counter_scope": os.getenv("REPORT_COUNTER_SCOPE", "report_level") or "report_level",
        "required_counter_sources_per_report": min_counter,
        "actual_counter_sources_per_report": actual_counter,
        "evidence_gap_count": len([item for item in list(evidence_gap_ledger or []) if isinstance(item, dict)]),
    }


def _report_env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _report_env_int(name: str, default: int, *, min_value: int = 0, max_value: int = 10_000) -> int:
    try:
        value = int(os.getenv(name, str(default)) or default)
    except (TypeError, ValueError):
        value = default
    return max(min_value, min(max_value, value))


def _fact_has_decision_signal(fact: Dict[str, Any]) -> bool:
    text = " ".join(
        str(fact.get(key) or "")
        for key in ("fact", "clean_fact", "content", "metric", "value", "metric_kind")
    )
    return bool(
        fact.get("s_grade")
        or fact.get("metric")
        or fact.get("value")
        or fact.get("numeric_values")
        or _has_number(text)
    )


def _source_identity(fact: Dict[str, Any]) -> str:
    source = _as_dict(fact.get("source"))
    raw = str(source.get("url") or source.get("title") or source.get("source") or fact.get("source_text") or "").strip()
    return re.sub(r"\s+", "", raw.lower())[:140] or str(fact.get("evidence_id") or id(fact))


def _context_fact_key(fact: Dict[str, Any]) -> str:
    text = str(fact.get("fact") or fact.get("clean_fact") or fact.get("content") or "").strip()
    metric = str(fact.get("metric") or "").strip()
    value = str(fact.get("value") or "").strip()
    dimension = str(fact.get("dimension") or fact.get("dimension_name") or "").strip()
    raw = "|".join([dimension[:80], metric[:40], value[:40], text[:160]])
    return re.sub(r"\s+", "", raw.lower())[:220]


def _is_context_support_candidate(fact: Dict[str, Any]) -> bool:
    if not isinstance(fact, dict) or not str(fact.get("fact") or fact.get("clean_fact") or fact.get("content") or "").strip():
        return False
    role = _canonical_role(fact.get("evidence_role"))
    semantic_status = str(fact.get("semantic_status") or "").strip().lower()
    if role == "rejected" or semantic_status in REJECTED_STATUSES:
        return False
    level = str(fact.get("source_level") or "").strip().upper()
    if level not in {"A", "B"}:
        return False
    if not _fact_has_decision_signal(fact):
        return False
    if bool(fact.get("conflict_flag")) and not fact.get("s_grade"):
        return False
    score = fact.get("task_relevance_score")
    try:
        if score is not None and float(score) < float(os.getenv("REPORT_CONTEXT_MIN_TASK_RELEVANCE", "0.10")):
            return False
    except (TypeError, ValueError):
        pass
    return role in {"clue", "supporting", "appendix"} or bool(fact.get("appendix_only"))


def promote_corroborated_context_facts(clean_evidence_list: Sequence[Dict[str, Any]]) -> int:
    """Lift strong A/B clue facts into bounded supporting context, not core claims."""
    if not _report_env_flag("REPORT_ENABLE_CORROBORATED_CONTEXT", True):
        return 0
    per_dimension_limit = _report_env_int("REPORT_CONTEXT_SUPPORTING_PER_DIMENSION", 24, min_value=0, max_value=200)
    total_limit = _report_env_int("REPORT_CONTEXT_SUPPORTING_TOTAL", 160, min_value=0, max_value=1000)
    per_source_cap = _report_env_int("REPORT_CONTEXT_SUPPORTING_PER_SOURCE", 4, min_value=1, max_value=50)
    if per_dimension_limit <= 0 or total_limit <= 0:
        return 0

    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for fact in clean_evidence_list:
        if _is_context_support_candidate(fact):
            dimension = str(fact.get("dimension") or fact.get("dimension_name") or "default").strip() or "default"
            buckets[dimension].append(fact)

    promoted = 0
    seen_keys: set[str] = set()
    for dimension, facts in buckets.items():
        source_counts: Dict[str, int] = defaultdict(int)
        selected = 0
        ranked = sorted(
            facts,
            key=lambda item: (
                1 if str(item.get("source_level") or "").upper() == "A" else 0,
                1 if item.get("s_grade") else 0,
                _clip(item.get("confidence"), 0.0),
                1 if item.get("metric") or item.get("value") else 0,
            ),
            reverse=True,
        )
        for fact in ranked:
            if promoted >= total_limit or selected >= per_dimension_limit:
                break
            key = _context_fact_key(fact)
            if key in seen_keys:
                continue
            source_key = _source_identity(fact)
            if source_counts[source_key] >= per_source_cap:
                continue
            seen_keys.add(key)
            source_counts[source_key] += 1
            selected += 1
            promoted += 1
            fact["evidence_role"] = "supporting"
            fact["appendix_only"] = False
            fact["enterprise_usable"] = True
            fact["usage_tier"] = "corroborated_context"
            fact["allowed_use"] = "supporting_context"
            fact["context_supporting"] = True
            fact["can_support_claim_if_corrobated"] = True
            previous_status = str(fact.get("semantic_status") or "").strip()
            previous_reason = str(fact.get("semantic_reason") or "").strip()
            fact["semantic_status"] = "context_support"
            fact["semantic_reason"] = "; ".join(
                item
                for item in [
                    previous_reason,
                    f"promoted_from_{previous_status or 'clue'}_for_context_only",
                ]
                if item
            )
            card = dict(_as_dict(fact.get("evidence_card")))
            card["allowed_use"] = "supporting_context"
            card["directness"] = card.get("directness") or "indirect"
            card["inference_distance"] = "medium_high"
            cannot_prove = list(_as_list(card.get("cannot_prove")))
            if "standalone core conclusion" not in cannot_prove:
                cannot_prove.append("standalone core conclusion")
            card["cannot_prove"] = cannot_prove
            fact["evidence_card"] = card
            fact["analysis_input"] = _analysis_input_for_evidence(fact, fact_text=fact.get("fact"))
    return promoted


def _layer_clean_evidence(clean_evidence_list: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    layered = {
        "core_evidence": [],
        "supporting_evidence": [],
        "clue_evidence": [],
        "appendix_evidence": [],
    }
    for fact in clean_evidence_list:
        if not isinstance(fact, dict) or not str(fact.get("fact") or "").strip():
            continue
        payload = _public_fact_payload(fact)
        role = _canonical_role(payload.get("evidence_role"))
        if role == "core":
            layered["core_evidence"].append(payload)
        elif role == "supporting":
            layered["supporting_evidence"].append(payload)
        elif role == "clue":
            layered["clue_evidence"].append(payload)
        elif role == "rejected":
            continue
        else:
            layered["appendix_evidence"].append(payload)
    return layered


def _rejected_evidence_sample(evidence_items: Sequence[Dict[str, Any]], *, limit: int = 20) -> List[Dict[str, Any]]:
    sample: List[Dict[str, Any]] = []
    for evidence in evidence_items:
        if not isinstance(evidence, dict):
            continue
        role = _canonical_role(evidence.get("evidence_role"))
        semantic_status = str(evidence.get("semantic_status") or "").strip().lower()
        if role != "rejected" and semantic_status not in REJECTED_STATUSES:
            continue
        sample.append(
            {
                "reason": evidence.get("semantic_reason") or evidence.get("task_acceptance_reason"),
                "source": evidence.get("source"),
                "content": str(evidence.get("content") or evidence.get("clean_fact") or "")[:160],
            }
        )
        if len(sample) >= limit:
            break
    return sample


def build_evidence_package(
    *,
    original_query: str = "",
    evidence_items: Sequence[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]] = None,
    top_k: int = 18,
    research_plan: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    top_k = int(os.getenv("EVIDENCE_PACKAGE_TOP_K") or max(int(top_k or 0), 48))
    dimensions = get_dynamic_dimensions(research_plan)
    evidence_by_dimension: Dict[str, List[Dict[str, Any]]] = {dimension: [] for dimension in dimensions}
    for item in evidence_items:
        dimension = _normalize_dimension(item.get("dimension"), research_plan) or _dynamic_dimension_from_item(item, research_plan)
        if not dimension:
            dimension = dimensions[0] if dimensions else "综合研究问题"
        if dimension not in evidence_by_dimension:
            evidence_by_dimension[dimension] = []
            dimensions.append(dimension)
        copied = dict(item)
        copied["dimension"] = dimension
        evidence_by_dimension[dimension].append(copied)

    conflicts = detect_conflicts([item for items in evidence_by_dimension.values() for item in items])
    conflicts_by_dimension: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for conflict in conflicts:
        conflicts_by_dimension[str(conflict.get("dimension") or "")].append(conflict)

    per_dimension: Dict[str, Dict[str, Any]] = {}
    clean_evidence_list: List[Dict[str, Any]] = []
    clean_duplicate_count = 0
    for dimension in dimensions:
        items = sorted(evidence_by_dimension.get(dimension, []), key=lambda item: (1 if item.get("s_grade") else 0, *_quality_key(item)), reverse=True)
        clean_facts, dimension_clean_duplicates = build_clean_facts(items)
        clean_duplicate_count += dimension_clean_duplicates
        clean_evidence_list.extend(clean_facts)
        per_dimension_limit = max(15, min(80, top_k + 12))
        per_dimension[dimension] = {
            "top_evidence": items[: max(8, min(60, top_k))],
            "clean_facts": clean_facts[:per_dimension_limit],
            "analysis_inputs": [
                _as_dict(fact.get("analysis_input")) or _analysis_input_for_evidence(fact, fact_text=fact.get("fact"))
                for fact in clean_facts[:per_dimension_limit]
            ],
            "s_grade_count": sum(1 for item in items if item.get("s_grade")),
            "conflicts": conflicts_by_dimension.get(dimension, []),
            "coverage_score": _dimension_coverage(items),
            "evidence_count": len(items),
            "agent_coverage": sorted({str(item.get("agent") or "") for item in items if item.get("agent")}),
        }

    coverage_scores = {dimension: payload["coverage_score"] for dimension, payload in per_dimension.items()}
    overall_coverage = round(sum(coverage_scores.values()) / max(len(dimensions), 1), 4)
    weakest_dimension = min(coverage_scores.items(), key=lambda item: item[1])[0] if coverage_scores else ""
    covered_dimensions = sum(1 for score in coverage_scores.values() if score > 0)
    chapter_plan = build_dynamic_chapter_plan(dimensions)
    chapter_dim_mapping = build_dynamic_chapter_mapping(dimensions)
    promoted_context_count = promote_corroborated_context_facts(clean_evidence_list)
    source_registry = _build_source_registry(clean_evidence_list)
    chapter_evidence = build_chapter_evidence(clean_evidence_list, chapter_dim_mapping=chapter_dim_mapping)
    analysis_ready_evidence = [
        _public_fact_payload(fact)
        for fact in clean_evidence_list
        if str(fact.get("fact") or "").strip()
        and _canonical_role(fact.get("evidence_role")) in {"core", "supporting"}
        and not fact.get("appendix_only")
        and str(fact.get("allowed_use") or _as_dict(fact.get("evidence_card")).get("allowed_use") or "") in {"core_claim", "supporting", "supporting_context", ""}
    ]
    rerank_diagnostics = build_rerank_diagnostics(
        _as_dict(metadata),
        evidence_items=evidence_items,
        clean_evidence_list=clean_evidence_list,
        analysis_ready_evidence=analysis_ready_evidence,
    )
    layered_evidence = _layer_clean_evidence(clean_evidence_list)
    filter_funnel = build_filter_funnel(
        _as_dict(metadata),
        evidence_items=evidence_items,
        clean_evidence_list=clean_evidence_list,
        analysis_ready_evidence=analysis_ready_evidence,
    )
    filter_funnel_by_chapter = build_filter_funnel_by_chapter(
        _as_dict(metadata),
        evidence_items=evidence_items,
        clean_evidence_list=clean_evidence_list,
    )
    evidence_analysis_by_chapter = _evidence_analysis_by_chapter(clean_evidence_list)
    evidence_gap_ledger = _evidence_gap_ledger_from_analysis(evidence_analysis_by_chapter)
    evidence_analysis_summary = _evidence_analysis_summary(evidence_analysis_by_chapter, evidence_gap_ledger)
    source_distribution = _source_level_distribution(clean_evidence_list)
    source_family_distribution = _distribution(_source_family(item) for item in clean_evidence_list if isinstance(item, dict))
    readpage_coverage = _merge_readpage_coverage(
        _readpage_coverage_from_evidence(clean_evidence_list),
        _as_dict(metadata).get("source_pool_readpage_coverage") or {},
    )
    core_ab_by_chapter = _core_ab_by_chapter(evidence_analysis_by_chapter)
    core_traceable_ab_by_chapter = _core_traceable_ab_by_chapter(evidence_analysis_by_chapter)
    chapter_contracts = _chapter_contracts_from_research_plan(_as_dict(research_plan)) or _default_chapter_contracts(evidence_analysis_by_chapter)
    publishable_evidence_gate = _publishable_evidence_gate(
        chapter_analysis=evidence_analysis_by_chapter,
        readpage_coverage=readpage_coverage,
        evidence_gap_ledger=evidence_gap_ledger,
        core_traceable_ab_by_chapter=core_traceable_ab_by_chapter,
        chapter_contracts=chapter_contracts,
        evidence_analysis_summary=evidence_analysis_summary,
    )
    analysis_ready_ab_count = len(
        [
            item
            for item in analysis_ready_evidence
            if str(item.get("source_level") or "").strip().upper() in {"A", "B"}
        ]
    )
    analysis_ready_metric_count = len(
        [
            item
            for item in analysis_ready_evidence
            if str(item.get("metric") or "").strip()
            and str(item.get("value") or "").strip()
            and (
                str(item.get("period") or "").strip()
                or str(_as_dict(item.get("source")).get("date") or "").strip()
            )
            and str(_as_dict(item.get("source")).get("url") or _as_dict(item.get("source")).get("title") or "").strip()
        ]
    )
    core_chapters_with_signal = len(
        [
            payload
            for payload in per_dimension.values()
            if any(str(item.get("source_level") or "").strip().upper() in {"A", "B"} for item in payload.get("clean_facts", []))
        ]
    )
    ready_for_analysis = bool(
        len(analysis_ready_evidence) > 0
        and covered_dimensions > 0
        and overall_coverage >= 0.30
        and (analysis_ready_ab_count > 0 or analysis_ready_metric_count > 0 or core_chapters_with_signal > 0)
    )
    evidence_health_summary = _build_evidence_health_summary(
        metadata=_as_dict(metadata),
        evidence_items=evidence_items,
        clean_evidence_list=clean_evidence_list,
        analysis_ready_evidence=analysis_ready_evidence,
        source_registry=source_registry,
        evidence_analysis_summary=evidence_analysis_summary,
        evidence_gap_ledger=evidence_gap_ledger,
        source_distribution=source_distribution,
        readpage_coverage=readpage_coverage,
        core_traceable_ab_by_chapter=core_traceable_ab_by_chapter,
        publishable_evidence_gate=publishable_evidence_gate,
    )
    missing_traceable_ab_fields = [
        {
            "chapter_id": str(chapter_id),
            "core_ab_source_count": int(_safe_float(_as_dict(payload).get("core_ab_source_count"), 0.0)),
        }
        for chapter_id, payload in evidence_analysis_by_chapter.items()
        if "core_traceable_ab_source_count" not in _as_dict(payload)
        and int(_safe_float(_as_dict(payload).get("core_ab_source_count"), 0.0)) > 0
    ]
    if missing_traceable_ab_fields:
        inconsistencies = list(_as_list(evidence_health_summary.get("inconsistencies")))
        inconsistencies.append(
            {
                "type": "traceable_ab_field_missing_with_ab_signal",
                "chapters": missing_traceable_ab_fields[:12],
            }
        )
        evidence_health_summary["inconsistent"] = True
        evidence_health_summary["inconsistencies"] = inconsistencies
    delivery_gate = _delivery_gate(
        publishable_gate=publishable_evidence_gate,
        evidence_count=len(evidence_items),
        clean_fact_count=len(clean_evidence_list),
        analysis_ready_count=len(analysis_ready_evidence),
        analysis_ready_ab_count=analysis_ready_ab_count,
        source_distribution=source_distribution,
        readpage_coverage=readpage_coverage,
        core_ab_by_chapter=core_ab_by_chapter,
        evidence_gap_ledger=evidence_gap_ledger,
        traceable_ab_source_count=evidence_health_summary.get("traceable_ab_source_count"),
        core_traceable_ab_by_chapter=core_traceable_ab_by_chapter,
        evidence_health_summary=evidence_health_summary,
    )
    evidence_preflight_summary = _build_evidence_preflight_summary(
        chapter_analysis=evidence_analysis_by_chapter,
        chapter_contracts=chapter_contracts,
        evidence_health_summary=evidence_health_summary,
        evidence_analysis_summary=evidence_analysis_summary,
        evidence_gap_ledger=evidence_gap_ledger,
        publishable_evidence_gate=publishable_evidence_gate,
        metadata=_as_dict(metadata),
    )
    evidence_health_summary["delivery_tier"] = delivery_gate.get("tier")
    evidence_health_summary["delivery_publishable"] = bool(delivery_gate.get("publishable"))
    evidence_health_summary["evidence_preflight"] = {
        "ready_for_clean_writer": bool(evidence_preflight_summary.get("ready_for_clean_writer")),
        "needs_gap_repair": bool(evidence_preflight_summary.get("needs_gap_repair")),
        "review_draft_allowed": bool(evidence_preflight_summary.get("review_draft_allowed")),
        "diagnostic_only": bool(evidence_preflight_summary.get("diagnostic_only")),
    }
    raw_data_points = _raw_data_points_for_package(clean_evidence_list)
    return {
        "package_type": "evidence_package",
        "query": original_query,
        "research_plan": _as_dict(research_plan),
        "chapter_plan": chapter_plan,
        "chapter_dim_mapping": chapter_dim_mapping,
        "chapter_evidence": chapter_evidence,
        "core_evidence": layered_evidence["core_evidence"],
        "supporting_evidence": layered_evidence["supporting_evidence"],
        "clue_evidence": layered_evidence["clue_evidence"],
        "appendix_evidence": layered_evidence["appendix_evidence"],
        "rejected_evidence_sample": _rejected_evidence_sample(evidence_items),
        "analysis_ready_evidence": analysis_ready_evidence,
        "normalized_evidence": list(evidence_items or []),
        "raw_data_points": raw_data_points,
        "source_registry": source_registry,
        "sources": source_registry,
        "evidence_health_summary": evidence_health_summary,
        "evidence_preflight_summary": evidence_preflight_summary,
        "source_registry_summary": _source_registry_summary(source_registry),
        "evidence_analysis_by_chapter": evidence_analysis_by_chapter,
        "chapter_evidence_diagnostics": evidence_analysis_by_chapter,
        "evidence_analysis_summary": evidence_analysis_summary,
        "evidence_gap_ledger": evidence_gap_ledger,
        "rerank_diagnostics": rerank_diagnostics,
        "filter_funnel": filter_funnel,
        "evidence_filter_funnel_by_chapter": filter_funnel_by_chapter,
        "per_dimension": per_dimension,
        "clean_evidence_list": clean_evidence_list,
        "summary": {
            "overall_coverage": overall_coverage,
            "weakest_dimension": weakest_dimension,
            "conflict_count": len(conflicts),
            "ready_for_analysis": ready_for_analysis,
            "dimension_count": len(dimensions),
            "covered_dimension_count": covered_dimensions,
            "evidence_count": len(evidence_items),
            "clean_fact_count": len(clean_evidence_list),
            "analysis_ready_count": len(analysis_ready_evidence),
            "analysis_ready_ab_count": analysis_ready_ab_count,
            "analysis_ready_metric_count": analysis_ready_metric_count,
            "claim_ready_evidence_count": evidence_analysis_summary.get("total_claim_ready_evidence_count"),
            "metric_ready_count": evidence_analysis_summary.get("total_metric_ready_count"),
            "evidence_gap_count": len(evidence_gap_ledger),
            "blocking_evidence_gap_count": evidence_analysis_summary.get("blocking_gap_count"),
            "evidence_gap_type_distribution": evidence_analysis_summary.get("gap_type_distribution"),
            "core_chapters_with_signal": core_chapters_with_signal,
            "promoted_context_count": promoted_context_count,
            "source_level_distribution": source_distribution,
            "source_family_distribution": source_family_distribution,
            "core_ab_by_chapter": core_ab_by_chapter,
            "core_traceable_ab_by_chapter": core_traceable_ab_by_chapter,
            "readpage_coverage": readpage_coverage,
            "evidence_health_summary": evidence_health_summary,
            "source_registry_summary": _source_registry_summary(source_registry),
            "openai_web_repair_summary": _as_dict(metadata).get("openai_web_search_summary", {}),
            "publishable_evidence_gate": publishable_evidence_gate,
            "evidence_preflight_summary": evidence_preflight_summary,
            "delivery_gate": delivery_gate,
            "delivery_tier": delivery_gate.get("tier"),
            "core_candidate_count": len([item for item in clean_evidence_list if str(item.get("evidence_role") or "") in {"core", "supporting"}]),
            "appendix_only_count": filter_funnel.get("appendix_only_count"),
            "clue_count": len(layered_evidence["clue_evidence"]),
            "rejected_count": filter_funnel.get("semantic_rejected_count"),
            "role_distribution": filter_funnel.get("role_distribution"),
            "source_pool_filter_reasons": _as_dict(metadata).get("source_pool_filter_reasons", {}),
            "rerank_diagnostics": rerank_diagnostics,
        },
        "metadata": {
            "merger": MERGER_NAME,
            "strategy": "deterministic_rules_to_analysis_ready_evidence",
            "cleaner": "atomic_fact_cleaning_source_dedupe_analysis_input_generation",
            "writer_contract": "writer_must_use_analysis_outputs_not_raw_evidence_for_judgment",
            "chapter_evidence_count": {chapter_id: len(items) for chapter_id, items in chapter_evidence.items()},
            "clean_deduped_count": clean_duplicate_count,
            "promoted_context_count": promoted_context_count,
            "dynamic_dimensions": dimensions,
            "research_plan": _as_dict(research_plan),
            "rerank_diagnostics": rerank_diagnostics,
            "evidence_health_summary": evidence_health_summary,
            "evidence_preflight_summary": evidence_preflight_summary,
            "source_registry_summary": _source_registry_summary(source_registry),
            **dict(metadata or {}),
        },
    }


def merge_evidence_package(
    *,
    original_query: str = "",
    evidence_pool: Optional[Sequence[Dict[str, Any]]] = None,
    children: Optional[Dict[str, Dict[str, Any]]] = None,
    current_date: Optional[datetime] = None,
    top_k: int = 18,
    domain_blacklist: Optional[Sequence[str]] = None,
    research_plan: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    source_pool = list(evidence_pool or [])
    if not source_pool and children:
        source_pool = _children_to_pool(children, original_query)
    filtered_source_pool, source_filter_meta = _filter_source_pool_for_merge(source_pool)

    normalized, normalize_meta = normalize_evidence_items(
        filtered_source_pool,
        current_date=current_date,
        domain_blacklist=domain_blacklist,
        research_plan=research_plan,
    )
    deduped, duplicate_count = dedupe_evidence(normalized)
    metadata = {
        **normalize_meta,
        **source_filter_meta,
        "deduped_count": duplicate_count,
        "source_pool_count": len(source_pool),
        "source_pool_readpage_coverage": _source_pool_readpage_coverage(filtered_source_pool),
    }
    package = build_evidence_package(
        original_query=original_query,
        evidence_items=deduped,
        metadata=metadata,
        top_k=top_k,
        research_plan=research_plan,
    )
    cache_summary = store_evidence_from_package(
        query=original_query,
        evidence_package=package,
        report_id=str(_as_dict(metadata).get("report_id") or ""),
        run_id=str(_as_dict(metadata).get("run_id") or ""),
    )
    trusted_source_cache_summary = store_trusted_sources_from_package(
        query=original_query,
        evidence_package=package,
        report_id=str(_as_dict(metadata).get("report_id") or ""),
        run_id=str(_as_dict(metadata).get("run_id") or ""),
    )
    package.setdefault("metadata", {})
    package["metadata"]["evidence_cache_store"] = cache_summary
    package["metadata"]["trusted_source_cache_store"] = trusted_source_cache_summary
    return package


def evidence_merger_tool(payload: Dict[str, Any]) -> Dict[str, Any]:
    return merge_evidence_package(
        original_query=str(payload.get("query") or payload.get("original_query") or ""),
        evidence_pool=[item for item in _as_list(payload.get("evidence_pool")) if isinstance(item, dict)],
        children=_as_dict(payload.get("children")),
        research_plan=_as_dict(payload.get("research_plan")),
    )


def create_evidence_merger_tool():
    """Return a LangChain-compatible deterministic merger tool."""

    from langchain_core.tools import tool

    @tool("evidence_merger", description="Deterministically merge child-agent evidence into an evidence_package.")
    def _evidence_merger(payload: Dict[str, Any]) -> Dict[str, Any]:
        return evidence_merger_tool(payload)

    return _evidence_merger
