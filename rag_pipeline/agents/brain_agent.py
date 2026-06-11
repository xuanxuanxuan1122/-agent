from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import operator
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError, as_completed
from datetime import datetime
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict

from langgraph.graph import END, START, StateGraph

from ..config.search_config import (
    build_llm_config_for_task,
)
from ..logging_utils import configure_pipeline_logging
from ..runtime_cache import json_safe_default
from ..search.engine import build_arg_parser as build_rag_arg_parser
from ..search.memory import call_openai_compatible_json, llm_config_is_ready, normalize_llm_config
from rag_pipeline.cache.evidence_cache import lookup_evidence as lookup_cached_evidence
from rag_pipeline.cache.evidence_cache import record_cache_activity
from rag_pipeline.cache.trusted_source_cache import lookup_trusted_sources
from rag_pipeline.contracts.query_builder import build_query_package
from rag_pipeline.contracts.repair_dispatcher import dispatch_repair_seed
from .analysis_agent import run_analysis_agent
from .article_brief import normalize_article_brief, planning_query_from_brief
from .dynamic_search_schema import normalize_search_task
from .evidence_merger import merge_evidence_package
from .pre_layout_agent import run_pre_layout_agent
from .rag_agent import namespace_to_overrides, run_rag_agent
from .research_planner import run_research_planner_agent
from .web_analysis_agent import run_web_analysis_agent
from .writer_agent import run_writer_agent


AGENT_NAME = "brain_agent"
logger = logging.getLogger(__name__)
AGENT_DESCRIPTION = (
    "企业研究多智能体系统的大脑 Agent。负责理解用户问题，调用 Research Planner 生成动态研究维度和搜索任务，"
    "并调度 IQS 联网检索 Worker 获取对应证据。本地 RAG 默认不进入主流程，可通过环境变量重新启用。"
)


def _progress_enabled() -> bool:
    raw = os.getenv("PIPELINE_PROGRESS", "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _short(value: Any, *, max_chars: int = 110) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)] + "…"


def _progress(stage: str, message: str, **fields: Any) -> None:
    if not _progress_enabled():
        return
    suffix = " ".join(f"{key}={_short(value, max_chars=80)}" for key, value in fields.items() if value not in (None, ""))
    line = f"[{datetime.now().strftime('%H:%M:%S')}] [{stage}] {message}"
    if suffix:
        line = f"{line} {suffix}"
    try:
        print(line, file=sys.stderr, flush=True)
    except (OSError, ValueError):
        logger.debug("Progress output skipped because stderr is unavailable.", exc_info=True)

_WEB_INTENT_RE = re.compile(
    r"(联网|网上|网页|搜索|搜一下|最新|近期|当前|现在|今日|今天|昨日|昨天|新闻|快讯|实时|行情|股价|价格|政策|监管|财报|公告|融资|并购|指数|利率|汇率)",
    re.I,
)
_LOCAL_INTENT_RE = re.compile(
    r"(本地|知识库|资料库|内部资料|文档|材料|文件|可可资本|基金|被投|投资项目|尽调|访谈|RAG|已有资料|这些资料)",
    re.I,
)
_INDUSTRY_RESEARCH_RE = re.compile(
    r"(行业|产业|市场|趋势|格局|竞争|产业链|价值链|商业模式|盈利|毛利|风险|机器人|人工智能|AI|新能源|半导体|医药|消费|制造)",
    re.I,
)
_GROWTH_FINANCE_RE = re.compile(
    r"("
    r"企业发展|公司发展|业务发展|战略|增长|转型|扩张|出海|降本增效|组织能力|管理|经营|商业化|产品化|市场进入|获客|渠道|品牌|"
    r"个人发展|职业发展|成长|升职|加薪|跳槽|副业|创业|打工人|普通人|致富|赚钱|财富|收入|变现|机会|"
    r"金融|投资|融资|理财|资产|股票|基金|债券|期货|股权|估值|资本|并购|上市|IPO|财务|现金流|利率|汇率|宏观|经济"
    r")",
    re.I,
)
_MARKET_DATA_RE = re.compile(
    r"(实时行情|实时股价|股价|股票行情|A股|港股|美股|Ticker|K线|涨跌幅|市值|外汇|汇率|USDCNY|EURUSD|期货|黄金|铜|螺纹钢|全球股市|全球指数|纳斯达克|道琼斯|标普|恒生)",
    re.I,
)

IQS_ROLE_CONFIGS: Dict[str, Dict[str, str]] = {
    f"iqs_lane_{index}": {
        "node": f"iqs_lane_{index}_agent",
        "state": f"iqs_lane_{index}_state",
        "child": f"iqs_lane_{index}_agent",
        "label": f"IQS Lane {index}",
        "dimension": "",
        "focus": "按 Research Planner 生成的 search_task 执行检索。",
    }
    for index in range(1, 7)
}
IQS_ROLE_ORDER = list(IQS_ROLE_CONFIGS.keys())

IQS_LANE_TYPE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "official_data": {
        "label": "Official/Data Lane",
        "source_priority": ["gov", "stats", "statistics", "association", "whitepaper", "regulator", "official"],
        "intents": ["statistics", "policy", "data"],
        "query_terms": ["official", "statistics", "government", "association"],
    },
    "filing_company": {
        "label": "Filing/Company Lane",
        "source_priority": ["annual_report", "prospectus", "exchange", "filing", "company", "cninfo", "公告", "财报"],
        "intents": ["filing", "company", "financial", "finance"],
        "query_terms": ["annual report", "filing", "announcement", "company"],
    },
    "market_research": {
        "label": "Market/Research Lane",
        "source_priority": ["research_report", "consulting", "brokerage", "industry_database", "market_research", "研报"],
        "intents": ["market", "analysis", "statistics", "data"],
        "query_terms": ["market research", "industry report", "market size"],
    },
    "news_event": {
        "label": "News/Event Lane",
        "source_priority": ["news", "event", "tender", "order", "lawsuit", "accident", "policy_implementation", "中标", "订单"],
        "intents": ["news", "risk", "event"],
        "query_terms": ["news", "event", "order", "tender"],
    },
    "technology_product": {
        "label": "Technology/Product Lane",
        "source_priority": ["paper", "patent", "product_doc", "technical_standard", "technology", "product", "专利", "技术标准"],
        "intents": ["academic", "technology", "technical", "product", "技术", "产品", "专利"],
        "query_terms": ["technology", "product", "patent", "standard", "技术", "专利", "折叠屏", "铰链", "良率"],
    },
    "customer_case": {
        "label": "Customer/Case Lane",
        "source_priority": ["case", "customer", "procurement", "roi", "application", "客户", "案例", "采购"],
        "intents": ["case", "customer", "business", "application"],
        "query_terms": ["customer case", "procurement", "ROI", "application"],
    },
}
IQS_LANE_TO_ROLE = {
    "official_data": "iqs_lane_1",
    "filing_company": "iqs_lane_2",
    "market_research": "iqs_lane_3",
    "news_event": "iqs_lane_4",
    "technology_product": "iqs_lane_5",
    "customer_case": "iqs_lane_6",
}
IQS_ROLE_CONFIGS = {
    role_key: {
        "node": f"{role_key}_agent",
        "state": f"{role_key}_state",
        "child": f"{role_key}_agent",
        "label": config["label"],
        "lane_type": lane_type,
        "dimension": "",
        "focus": f"Evidence-type retrieval: {config['label']}",
    }
    for lane_type, role_key in IQS_LANE_TO_ROLE.items()
    for config in [IQS_LANE_TYPE_CONFIGS[lane_type]]
}
IQS_ROLE_ORDER = list(IQS_ROLE_CONFIGS.keys())

TECH_LANE_HINT_RE = re.compile(
    r"(技术瓶颈|折叠屏|铰链|UTG|良率|专利|量产|供应链|材料|芯片|模型|算力|端侧|视觉|多模态|"
    r"technology|patent|yield|mass production|supply chain|chip|compute|multimodal)",
    re.I,
)
COMPANY_LANE_HINT_RE = re.compile(
    r"(公司|财报|年报|公告|招股书|港交所|交易所|披露|收入|毛利率|订单|SEC|10-K|annual report|filing|prospectus)",
    re.I,
)
INITIAL_LANE_PRIORITY = [
    "official_data",
    "market_research",
    "technology_product",
    "filing_company",
    "news_event",
    "customer_case",
]
INDUSTRY_REPORT_FAMILIES = {
    "industry_deep_report",
    "industry_scan_report",
    "deep_industry_report",
    "industry_report",
}
AI_AGENT_TOPIC_RE = re.compile(
    r"(AI\s*Agent|Agentic\s*AI|autonomous\s+agent|workflow\s+automation|multi[-\s]?agent|"
    r"智能体|智能代理|自主智能体|代理式AI|工作流自动化|智能体生态|Agent生态)",
    re.I,
)
PROOF_ROLES = {"metric", "support", "counter", "case", "filing", "source_check", "technology_product"}


class BrainAgentState(TypedDict, total=False):
    messages: List[Dict[str, Any]]
    query: str
    article_brief: Dict[str, Any]
    route: str
    route_reason: str
    query_analysis: Dict[str, Any]
    research_plan: Dict[str, Any]
    session_id: str
    args_overrides: Dict[str, Any]
    web_search_options: Dict[str, Any]
    enable_web_analysis: bool
    enable_llm_merge: bool
    enable_followup_loop: bool
    supervisor_max_loops: int
    supervisor_min_coverage_gain: float
    supervisor_max_followup_queries: int
    layout_max_refinement_rounds: int
    output_mode: str
    parallel_raw_output: bool
    topic_bundle_seed: Dict[str, Any]
    deadline_ts: float
    timeout_context: Dict[str, Any]
    fail_open_on_timeout: bool
    max_wall_seconds: int
    local_state: Dict[str, Any]
    web_state: Dict[str, Any]
    iqs_lane_1_state: Dict[str, Any]
    iqs_lane_2_state: Dict[str, Any]
    iqs_lane_3_state: Dict[str, Any]
    iqs_lane_4_state: Dict[str, Any]
    iqs_lane_5_state: Dict[str, Any]
    iqs_lane_6_state: Dict[str, Any]
    evidence_package: Dict[str, Any]
    structured_analysis: Dict[str, Any]
    report_blueprint: Dict[str, Any]
    search_tasks: List[Dict[str, Any]]
    search_task_schedule: Dict[str, Any]
    lane_coverage: Dict[str, Any]
    chapter_evidence_packages: List[Dict[str, Any]]
    evidence_graph: Dict[str, Any]
    micro_layouts: List[Dict[str, Any]]
    table_packages: List[Dict[str, Any]]
    argument_units: List[Dict[str, Any]]
    chapter_packages: List[Dict[str, Any]]
    decision_package: Dict[str, Any]
    risk_package: Dict[str, Any]
    appendix_package: Dict[str, Any]
    writer_report: Dict[str, Any]
    qa_result: Dict[str, Any]
    package_quality_report: Dict[str, Any]
    answer_text: str
    raw_output: Dict[str, Any]
    metadata: Dict[str, Any]
    agent_trace: Annotated[List[Dict[str, Any]], operator.add]
    errors: Annotated[List[str], operator.add]


def _message_content(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("content") or "").strip()
    return str(getattr(message, "content", "") or "").strip()


def _message_role(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role") or message.get("type") or "").strip().lower()
    return str(getattr(message, "type", "") or getattr(message, "role", "") or "").strip().lower()


def extract_query_from_state(state: BrainAgentState) -> str:
    explicit_query = str(state.get("query") or "").strip()
    if explicit_query:
        return explicit_query
    for message in reversed(list(state.get("messages") or [])):
        role = _message_role(message)
        if role in {"user", "human"} or not role:
            content = _message_content(message)
            if content:
                return content
    return ""


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(
    name: str,
    default: int,
    *,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
) -> int:
    raw = os.getenv(name)
    if raw is None:
        value = default
    else:
        try:
            value = int(str(raw).strip())
        except ValueError:
            value = default
    if min_value is not None:
        value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value


def _iqs_query_max_chars() -> int:
    configured = _env_int("BRAIN_IQS_QUERY_MAX_CHARS", _env_int("IQS_QUERY_MAX_CHARS", 56))
    return max(32, min(96, configured))


def _iqs_query_max_bytes() -> int:
    configured = _env_int("BRAIN_IQS_QUERY_MAX_BYTES", _env_int("IQS_QUERY_MAX_BYTES", 64))
    return max(40, min(180, configured))


def _iqs_query_fits(value: str, *, max_chars: Optional[int] = None, max_bytes: Optional[int] = None) -> bool:
    text = str(value or "")
    return len(text) <= (max_chars or _iqs_query_max_chars()) and len(text.encode("utf-8")) <= (max_bytes or _iqs_query_max_bytes())


def _trim_iqs_query(value: str, *, max_chars: Optional[int] = None, max_bytes: Optional[int] = None) -> str:
    char_limit = max_chars or _iqs_query_max_chars()
    byte_limit = max_bytes or _iqs_query_max_bytes()
    tokens: List[str] = []
    for token in _split_iqs_terms(value):
        candidate = " ".join(tokens + [token]).strip()
        if _iqs_query_fits(candidate, max_chars=char_limit, max_bytes=byte_limit):
            tokens.append(token)
            continue
        if tokens:
            continue
        trimmed = ""
        for char in token:
            next_value = trimmed + char
            if len(next_value) > char_limit or len(next_value.encode("utf-8")) > byte_limit:
                break
            trimmed = next_value
        if trimmed:
            return trimmed.strip()
    return " ".join(tokens).strip()


def _compact_iqs_term(value: Any, *, max_chars: int = 14) -> str:
    text = re.sub(r"https?://\S+", " ", str(value or ""))
    text = re.sub(r"[\r\n\t,;，。；、：:!?！？\[\]【】()（）{}<>《》\"']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].strip()


def _split_iqs_terms(value: Any) -> List[str]:
    text = re.sub(r"https?://\S+", " ", str(value or ""))
    text = re.sub(r"[\r\n\t,;，。；、：:!?！？\[\]【】()（）{}<>《》\"']", " ", text)
    return [part for part in re.split(r"\s+", text.strip()) if part]


def _compact_iqs_terms(values: Sequence[Any], *, max_terms: int = 4, max_chars: int = 14) -> List[str]:
    terms: List[str] = []
    seen = set()
    for raw in values:
        for part in _split_iqs_terms(raw):
            term = _compact_iqs_term(part, max_chars=max_chars)
            key = re.sub(r"\s+", "", term.lower())
            if not key or key in seen:
                continue
            seen.add(key)
            terms.append(term)
            if len(terms) >= max_terms:
                return terms
    return terms


def _topic_seed_terms(query: str, chapter: Optional[Dict[str, Any]] = None, goal: Optional[Dict[str, Any]] = None) -> List[str]:
    text = " ".join(
        [
            str(query or ""),
            str((chapter or {}).get("chapter_title") or ""),
            str((chapter or {}).get("core_question") or ""),
            str((goal or {}).get("question") or (goal or {}).get("evidence_goal") or ""),
        ]
    )
    lower_text = text.lower()
    chip_context = any(
        needle in text
        for needle in ["半导体", "芯片", "晶圆", "封测", "先进封装", "光刻", "EDA", "Chiplet", "GPU", "ASIC"]
    ) or any(needle in lower_text for needle in ["semiconductor", "chiplet", "wafer", "gpu", "asic", "eda", "asml"])
    seeds: List[str] = []

    def add(term: str, *needles: str) -> None:
        if term in seeds:
            return
        if not needles or any(needle and needle.lower() in lower_text for needle in needles):
            seeds.append(term)

    # Keep named actors ahead of generic policy/industry seeds.  Otherwise long
    # report topics collapse into broad queries such as "中美政策 数据 统计".
    if AI_AGENT_TOPIC_RE.search(text):
        add("AI Agent", "AI Agent", "智能体", "智能代理", "Agent生态")
        add("智能体", "智能体", "智能代理", "自主智能体")
        add("Agentic AI", "Agentic AI", "代理式AI", "autonomous agent")
        add("workflow automation", "workflow automation", "工作流自动化")
        add("enterprise AI agents", "enterprise", "企业", "商业化", "客户")
    add("马斯克", "马斯克", "musk")
    add("库克", "库克", "tim cook")
    add("黄仁勋", "黄仁勋", "jensen huang")
    add("特斯拉", "特斯拉", "tesla", "马斯克", "musk")
    add("苹果", "苹果", "apple", "库克", "tim cook")
    add("英伟达", "英伟达", "nvidia", "黄仁勋", "jensen huang")
    if any(term in seeds for term in {"马斯克", "库克", "黄仁勋", "特斯拉", "苹果", "英伟达"}) and (
        "访" in text or "再连接" in text or "中美" in text
    ):
        add("高管访华")

    add("半导体", "半导体", "芯片")
    add("芯片", "芯片")
    if chip_context and any(needle in lower_text for needle in ["ai", "人工智能", "gpu", "asic"]):
        add("AI芯片", "AI", "人工智能", "GPU", "ASIC")
    elif not AI_AGENT_TOPIC_RE.search(text):
        add("人工智能", "AI", "人工智能")
    add("供应链", "供应链", "产业链")
    add("中美科技" if chip_context else "中美政策", "中美", "美国", "出口管制")
    add("中国芯片" if chip_context else "中国市场", "中国", "国产")
    add("先进制程", "先进制程", "制程")
    add("EDA", "EDA")
    add("光刻设备", "光刻", "ASML")
    add("设备材料", "设备", "材料")
    add("封测", "封测")
    add("成熟制程", "成熟制程")
    add("先进封装", "先进封装", "Chiplet")
    if not seeds:
        seeds.extend(_compact_iqs_terms([query], max_terms=2, max_chars=16))
    return seeds[:6]


def _compose_iqs_query(parts: Sequence[Any], *, max_chars: Optional[int] = None) -> str:
    char_limit = max_chars or _iqs_query_max_chars()
    byte_limit = _iqs_query_max_bytes()
    tokens: List[str] = []
    seen = set()
    for part in parts:
        if isinstance(part, (list, tuple, set)):
            chunks = list(part)
        else:
            chunks = _split_iqs_terms(part)
        for chunk in chunks:
            for raw_token in _split_iqs_terms(chunk):
                token = _compact_iqs_term(raw_token, max_chars=14)
                key = re.sub(r"\s+", "", token.lower())
                if not key or key in seen:
                    continue
                candidate = " ".join(tokens + [token]).strip()
                if not _iqs_query_fits(candidate, max_chars=char_limit, max_bytes=byte_limit):
                    continue
                seen.add(key)
                tokens.append(token)
    return _trim_iqs_query(" ".join(tokens).strip(), max_chars=char_limit, max_bytes=byte_limit)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(str(raw).strip())
    except ValueError:
        return default


def _local_rag_enabled() -> bool:
    if os.getenv("BRAIN_ENABLE_LOCAL_RAG") is not None:
        return _env_flag("BRAIN_ENABLE_LOCAL_RAG", False)
    if os.getenv("REPORT_ENABLE_LOCAL_RAG") is not None:
        return _env_flag("REPORT_ENABLE_LOCAL_RAG", False)
    return False


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


def _continuous_evidence_loop_mode() -> bool:
    raw = os.getenv("REPORT_CONTINUOUS_EVIDENCE_LOOP")
    if raw is not None:
        return str(raw).strip().lower() in {"1", "true", "yes", "on", "strict"}
    return _strict_quality_mode()


def _supervisor_coverage_target() -> float:
    return max(0.5, min(0.98, _env_float("BRAIN_SUPERVISOR_COVERAGE_TARGET", 0.8)))


def _effective_iqs_lane_task_limit() -> int:
    configured = _env_int("BRAIN_MAX_TASKS_PER_IQS_LANE", 48)
    if _strict_quality_mode():
        return max(configured, _env_int("BRAIN_STRICT_MAX_TASKS_PER_IQS_LANE", 96))
    if _continuous_evidence_loop_mode():
        return max(configured, _env_int("BRAIN_CONTINUOUS_MAX_TASKS_PER_IQS_LANE", 12))
    return configured


def _effective_queries_per_agent() -> int:
    configured = _env_int("BRAIN_QUERY_ANALYSIS_MAX_QUERIES_PER_AGENT", 12)
    if _strict_quality_mode():
        return max(configured, _env_int("BRAIN_STRICT_MAX_QUERIES_PER_AGENT", 16))
    if _continuous_evidence_loop_mode():
        return max(configured, _env_int("BRAIN_CONTINUOUS_MAX_QUERIES_PER_AGENT", 8))
    return configured


def route_query(query: str, forced_route: str = "auto") -> tuple[str, str]:
    forced = str(forced_route or "auto").strip().lower()
    if forced in {"local", "web", "both", "all"}:
        if forced in {"local", "both", "all"} and not _local_rag_enabled():
            return "web", f"Local RAG is disabled for the main flow; route={forced} was mapped to web/IQS-only."
        return forced, f"用户显式指定 route={forced}"

    text = str(query or "").strip()
    web_intent = bool(_WEB_INTENT_RE.search(text))
    local_intent = bool(_LOCAL_INTENT_RE.search(text))
    industry_intent = bool(_INDUSTRY_RESEARCH_RE.search(text))
    growth_finance_intent = bool(_GROWTH_FINANCE_RE.search(text))
    market_data_intent = bool(_MARKET_DATA_RE.search(text))

    if not _local_rag_enabled():
        if market_data_intent:
            return "web", "Local RAG is disabled; market/data requests use IQS/web evidence only."
        if web_intent or local_intent or industry_intent or growth_finance_intent:
            return "web", "Local RAG is disabled for the main flow; using IQS/web evidence lanes only."
        return "web", "Local RAG is disabled for the main flow; using IQS/web evidence lanes only."
    if market_data_intent and (growth_finance_intent or industry_intent or web_intent or local_intent):
        return "both", "问题包含金融行情/实时数值需求，由 IQS 负责最新数据检索，并结合 RAG 做背景交叉验证"
    if market_data_intent:
        return "web", "问题主要需要股票、外汇、期货、指数或企业财务等最新数值数据，由 IQS 联网检索"
    if growth_finance_intent:
        return "both", "问题涉及企业发展、个人发展或金融投资，适合本地知识库与联网信息并行交叉分析"
    if web_intent and (local_intent or industry_intent):
        return "both", "问题同时需要最新外部信息和行研/本地证据交叉验证"
    if web_intent:
        return "web", "问题主要需要联网搜索、实时信息或公开网页证据"
    if industry_intent:
        return "both", "问题属于研究分析场景，默认并行调用本地 RAG 与 Planner 生成的动态 IQS 检索任务。"
    return "local", "问题优先使用本地知识库进行证据约束回答"


def _unique_strings(items: Sequence[Any], *, max_items: int = 5) -> List[str]:
    values: List[str] = []
    seen = set()
    for item in items:
        text = re.sub(r"\s+", " ", str(item or "").strip())
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        values.append(text)
        if len(values) >= max(1, max_items):
            break
    return values


def _env_csv(name: str, default: str = "") -> List[str]:
    raw = os.getenv(name, default)
    values: List[str] = []
    for part in re.split(r"[,，\s]+", str(raw or "")):
        text = part.strip().lower()
        if text and text not in values:
            values.append(text)
    return values


def _lane_type_for_role(role_key: str) -> str:
    role = str(role_key or "").strip().lower()
    for lane_type, mapped_role in IQS_LANE_TO_ROLE.items():
        if mapped_role == role:
            return lane_type
    return ""


def _role_for_lane_type_safe(lane_type: str) -> str:
    return IQS_LANE_TO_ROLE.get(str(lane_type or "").strip().lower(), "")


RETRIEVAL_MODES = {"deep", "normal", "hybrid"}
DEEP_DEFAULT_LANE_TYPES = {"official_data", "filing_company", "market_research", "technology_product"}
HYBRID_DEFAULT_LANE_TYPES = {"customer_case"}
DEEP_PROOF_ROLES = {"metric", "source_check", "filing", "official_data", "technology_product", "counter"}
AUTHORITY_DOCUMENT_RE = re.compile(
    r"(policy|regulation|regulator|official|government|gov|filing|annual report|prospectus|"
    r"announcement|disclosure|standard|statistics|whitepaper|association|"
    r"政策|法规|监管|官方|政府|公告|披露|财报|年报|招股书|标准|统计|白皮书|协会|原文)",
    re.I,
)
FRESHNESS_RE = re.compile(
    r"(latest|recent|news|today|yesterday|price|stock|quote|breaking|event|"
    r"最新|近期|新闻|今日|今天|昨日|昨天|价格|股价|行情|快讯|事件|动态)",
    re.I,
)


def _retrieval_routing_enabled() -> bool:
    return _env_flag("BRAIN_ENABLE_RETRIEVAL_MODE_ROUTING", True)


def _normalize_retrieval_mode(value: Any) -> str:
    mode = str(value or "").strip().lower()
    return mode if mode in RETRIEVAL_MODES else ""


def _retrieval_task_text(task: Dict[str, Any]) -> str:
    return " ".join(
        str(value or "")
        for value in [
            task.get("query"),
            task.get("evidence_goal"),
            task.get("targets_gap"),
            task.get("proof_role"),
            task.get("evidence_type"),
            " ".join(str(item) for item in _as_list(task.get("source_priority"))),
            " ".join(str(item) for item in _as_list(task.get("required_evidence_mix"))),
            " ".join(str(item) for item in _as_list(task.get("blocking_gaps"))),
        ]
    ).lower()


def _retrieval_route_for_task(task: Dict[str, Any], lane_type: str = "") -> Dict[str, Any]:
    explicit_mode = _normalize_retrieval_mode(task.get("retrieval_mode"))
    if explicit_mode and not _retrieval_routing_enabled():
        return {
            "retrieval_mode": explicit_mode,
            "retrieval_reason": task.get("retrieval_reason") or "explicit_retrieval_mode",
            "primary_provider": task.get("primary_provider") or ("iqs_deep" if explicit_mode in {"deep", "hybrid"} else "iqs_normal"),
            "fallback_providers": _as_list(task.get("fallback_providers")),
        }

    lane = str(lane_type or task.get("scheduled_lane_type") or "").strip().lower()
    text = _retrieval_task_text(task)
    role = str(task.get("proof_role") or task.get("evidence_type") or "").strip().lower()
    deep_requested = bool(task.get("deep_search_variant") or task.get("prefer_deep") or role in DEEP_PROOF_ROLES)
    authoritative = bool(AUTHORITY_DOCUMENT_RE.search(text))
    freshness = bool(FRESHNESS_RE.search(text))

    if explicit_mode and explicit_mode != "normal":
        mode = explicit_mode
        reason = task.get("retrieval_reason") or "explicit_retrieval_mode"
    elif lane in DEEP_DEFAULT_LANE_TYPES:
        mode = "deep"
        reason = f"lane_default:{lane}"
    elif lane in HYBRID_DEFAULT_LANE_TYPES:
        mode = "hybrid"
        reason = f"lane_default:{lane}"
    elif lane == "news_event":
        if authoritative or deep_requested:
            mode = "hybrid"
            reason = "news_event_authoritative_source"
        else:
            mode = "normal"
            reason = "news_event_freshness"
    elif deep_requested and not freshness:
        mode = "deep"
        reason = f"proof_role:{role or 'deep_required'}"
    else:
        mode = explicit_mode or "normal"
        reason = task.get("retrieval_reason") or ("freshness_or_breadth" if freshness else "default_normal_search")

    if mode == "deep":
        primary_provider = "iqs_deep"
        fallback_providers = ["iqs_normal"]
    elif mode == "hybrid":
        primary_provider = "iqs_deep"
        fallback_providers = ["iqs_normal"]
    else:
        primary_provider = "iqs_normal"
        fallback_providers = []

    return {
        "retrieval_mode": mode,
        "retrieval_reason": str(reason or "").strip(),
        "primary_provider": str(task.get("primary_provider") or primary_provider).strip(),
        "fallback_providers": _as_list(task.get("fallback_providers")) or fallback_providers,
    }


def _apply_retrieval_routing_to_task(task: Dict[str, Any], lane_type: str = "") -> Dict[str, Any]:
    copied = dict(task)
    route = _retrieval_route_for_task(copied, lane_type=lane_type)
    mode = str(route.get("retrieval_mode") or "normal").strip().lower()
    copied["retrieval_mode"] = mode
    copied["retrieval_reason"] = route.get("retrieval_reason") or copied.get("retrieval_reason") or ""
    copied["primary_provider"] = route.get("primary_provider") or copied.get("primary_provider") or ""
    copied["fallback_providers"] = _unique_strings(_as_list(route.get("fallback_providers")), max_items=4)
    if mode in {"deep", "hybrid"}:
        copied["prefer_deep"] = True
        if not str(copied.get("deep_reason") or "").strip():
            copied["deep_reason"] = copied.get("retrieval_reason") or "retrieval_routing"
        if not _as_list(copied.get("engineTypes")):
            copied["engineTypes"] = _deep_repair_engines()
    elif mode == "normal":
        copied["prefer_deep"] = False
    return copied


def _query_wants_technology_lane(text: str) -> bool:
    return bool(TECH_LANE_HINT_RE.search(str(text or "")))


def _query_wants_company_lane(text: str) -> bool:
    return bool(COMPANY_LANE_HINT_RE.search(str(text or "")))


def _select_quality_first_initial_lanes(
    *,
    query: str,
    agents: Sequence[str],
    dynamic_tasks: Sequence[Dict[str, Any]],
    research_plan: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Choose a smaller first-wave IQS lane set without dropping core lanes blindly."""
    original_roles = [role for role in list(agents or []) if role in IQS_ROLE_CONFIGS]
    if not _env_flag("BRAIN_IQS_INITIAL_LANE_QUALITY_FIRST", True) or not original_roles:
        return {
            "enabled": False,
            "selected_roles": original_roles,
            "selected_lane_types": [_lane_type_for_role(role) for role in original_roles],
            "skipped_lane_types": [],
        }

    text_parts = [str(query or "")]
    for task in list(dynamic_tasks or [])[:80]:
        if not isinstance(task, dict):
            continue
        text_parts.extend(
            [
                str(task.get("query") or ""),
                str(task.get("chapter_title") or ""),
                str(task.get("hypothesis_statement") or ""),
                str(task.get("evidence_goal") or ""),
            ]
        )
    combined_text = " ".join(text_parts)
    plan = _as_dict(research_plan)
    report_family = str(
        plan.get("report_family")
        or plan.get("report_profile")
        or plan.get("report_mode")
        or ""
    ).strip().lower()
    force_industry_lanes = report_family in INDUSTRY_REPORT_FAMILIES
    force_required_lanes = _env_flag("BRAIN_IQS_INITIAL_FORCE_REQUIRED_LANES", True)
    top_n = _env_int("BRAIN_IQS_INITIAL_LANE_TOP_N", 6, min_value=1, max_value=len(IQS_LANE_TO_ROLE))
    if force_required_lanes or force_industry_lanes:
        top_n = max(top_n, min(6, len(IQS_LANE_TO_ROLE)))
    disabled = set(_env_csv("BRAIN_IQS_INITIAL_DISABLED_LANES", ""))
    if force_required_lanes or force_industry_lanes:
        disabled.difference_update({"official_data", "market_research", "news_event", "technology_product", "filing_company", "customer_case"})
    enable_customer = force_industry_lanes or _env_flag("BRAIN_IQS_INITIAL_ENABLE_CUSTOMER_CASE", True)

    wanted: List[str] = ["official_data", "market_research", "news_event", "technology_product", "filing_company"]
    if _query_wants_technology_lane(combined_text):
        wanted.append("technology_product")
    if _query_wants_company_lane(combined_text):
        wanted.append("filing_company")
    if enable_customer:
        wanted.append("customer_case")

    available_lane_types = [_lane_type_for_role(role) for role in original_roles]
    selected: List[str] = []
    for lane_type in INITIAL_LANE_PRIORITY:
        if lane_type not in wanted and lane_type != "official_data":
            continue
        if lane_type in disabled and lane_type != "official_data":
            continue
        if lane_type in available_lane_types and lane_type not in selected:
            selected.append(lane_type)
        if len(selected) >= top_n:
            break

    if "official_data" in available_lane_types and "official_data" not in selected:
        selected.insert(0, "official_data")
    if not selected:
        selected = available_lane_types[:top_n]
    selected = selected[:top_n]
    selected_roles = [_role_for_lane_type_safe(lane) for lane in selected if _role_for_lane_type_safe(lane)]
    skipped_lane_types = [lane for lane in available_lane_types if lane and lane not in selected]
    return {
        "enabled": True,
        "quality_first": True,
        "top_n": top_n,
        "selected_lane_types": selected,
        "selected_roles": selected_roles,
        "skipped_lane_types": skipped_lane_types,
        "disabled_lane_types": sorted(disabled),
        "technology_lane_selected": "technology_product" in selected,
        "filing_lane_selected": "filing_company" in selected,
        "customer_case_initial_enabled": enable_customer and "customer_case" in selected,
    }


def _route_agents(route: str) -> List[str]:
    route = str(route or "local").strip().lower()
    rag_agents = ["rag"] if _local_rag_enabled() else []
    if route == "all":
        return [*rag_agents, *IQS_ROLE_ORDER]
    if route == "both":
        return [*rag_agents, *IQS_ROLE_ORDER]
    if route == "web":
        return list(IQS_ROLE_ORDER)
    if route == "local":
        return rag_agents or list(IQS_ROLE_ORDER)
    return rag_agents or list(IQS_ROLE_ORDER)


REPORT_TYPE_PROFILES: Dict[str, Dict[str, Any]] = {
    "dynamic": {
        "report_type": "dynamic_research",
        "report_name": "动态研究报告",
        "chapter_structure": [],
        "iqs_dimensions": [],
        "core_value": "按 Research Planner 的任务与证据目标组织输出",
    },
}


def classify_report_type(query: str) -> Dict[str, Any]:
    return {
        "report_type": "dynamic_research",
        "report_name": "动态研究报告",
        "chapter_structure": [],
        "iqs_dimensions": [],
        "core_value": "按 Research Planner 的任务与证据目标组织输出",
    }


def _report_plan_from_research_plan(research_plan: Dict[str, Any], query: str) -> Dict[str, Any]:
    chapters = [
        str(item.get("chapter_title") or item.get("title") or "").strip()
        for item in _as_list(_as_dict(research_plan).get("chapters"))
        if isinstance(item, dict) and str(item.get("chapter_title") or item.get("title") or "").strip()
    ]
    dimensions = chapters or [
        str(item.get("dimension_name") or item.get("name") or item.get("dimension") or "").strip()
        for item in _as_list(_as_dict(research_plan).get("dimensions"))
        if isinstance(item, dict) and str(item.get("dimension_name") or item.get("name") or item.get("dimension") or "").strip()
    ]
    report_family = str(_as_dict(research_plan).get("report_family") or "dynamic_research").strip()
    return {
        "report_type": report_family,
        "report_name": report_family,
        "research_type": _as_dict(research_plan).get("research_type"),
        "research_object": _as_dict(research_plan).get("research_object") or query,
        "planning_query": _as_dict(research_plan).get("planning_query") or _as_dict(research_plan).get("query") or query,
        "article_direction": _as_dict(research_plan).get("article_direction"),
        "article_brief": _as_dict(research_plan).get("article_brief"),
        "report_title": _as_dict(research_plan).get("report_title"),
        "report_subtitle": _as_dict(research_plan).get("report_subtitle"),
        "chapter_structure": dimensions,
        "iqs_dimensions": [],
        "core_value": "按 Research Planner 的任务与证据目标组织输出",
    }


def _role_queries_for_report_type(query: str, current_year: int, report_type: str) -> Dict[str, Sequence[str]]:
    return {}


def build_dynamic_iqs_tasks(query_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    plan = _as_dict(query_analysis.get("research_plan"))
    tasks: List[Dict[str, Any]] = []
    for index, raw in enumerate(_as_list(plan.get("search_tasks")), start=1):
        if not isinstance(raw, dict):
            continue
        task = normalize_search_task(raw, fallback_index=index)
        if task.get("query"):
            tasks.append(task)
    return tasks


def _mix_to_lane_targets(required_mix: Sequence[Any], proof_role: str) -> List[str]:
    lanes: List[str] = []
    role_priority = {
        "metric": ["official_data", "market_research"],
        "support": ["market_research", "official_data"],
        "source_check": ["official_data", "filing_company", "market_research"],
        "case": ["customer_case", "filing_company", "news_event"],
        "counter": ["news_event", "market_research", "customer_case"],
        "filing": ["filing_company", "official_data"],
        "technology_product": ["technology_product", "official_data", "market_research"],
    }
    mapping = {
        "official_data": "official_data",
        "official": "official_data",
        "market_price": "market_research",
        "inventory": "market_research",
        "market_research": "market_research",
        "company_filing": "filing_company",
        "filing": "filing_company",
        "news_event": "news_event",
        "counter": "news_event",
        "counter_evidence": "news_event",
        "risk": "news_event",
        "technology_product": "technology_product",
        "technology": "technology_product",
        "product_doc": "technology_product",
        "technical_standard": "technology_product",
        "patent": "technology_product",
        "case": "customer_case",
        "customer_case": "customer_case",
        "procurement": "customer_case",
    }
    def add_lane(lane: str) -> None:
        if lane and lane not in lanes:
            lanes.append(lane)

    for lane in role_priority.get(str(proof_role or "").strip().lower(), []):
        add_lane(lane)
    for item in required_mix:
        add_lane(mapping.get(str(item or "").strip().lower(), ""))
    return lanes[:3] or ["market_research"]


def _intent_for_proof_role(proof_role: str) -> str:
    return {
        "metric": "statistics",
        "support": "analysis",
        "counter": "risk",
        "case": "case",
        "filing": "filing",
        "source_check": "source_check",
        "technology_product": "technology",
    }.get(proof_role, "analysis")


_SEARCH_TERM_HINTS = (
    "新能源汽车",
    "新能源车",
    "新型材料",
    "电池材料",
    "功能材料",
    "动力电池",
    "轻量化材料",
    "结构材料",
    "市场行情",
    "市场规模",
    "出货量",
    "渗透率",
    "价格",
    "毛利率",
    "产能利用率",
    "产能",
    "订单",
    "客户认证",
    "量产",
    "需求",
)
_SEARCH_TERM_STOPWORDS = {
    "现在",
    "当前",
    "哪些",
    "是否",
    "怎么",
    "如何",
    "本章",
    "核心",
    "问题",
    "回答",
    "判断",
    "向好",
    "确定性",
    "短期",
    "放量",
}


def _append_unique_text(items: List[str], value: Any, *, max_items: int = 12) -> None:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = text.strip(" ,;:!?，。；：！？、()（）[]【】{}《》\"'")
    if not text or text in _SEARCH_TERM_STOPWORDS or text in items:
        return
    if len(text) < 2 or len(text) > 24:
        return
    items.append(text)
    del items[max_items:]


def _anchor_terms_from_text(value: Any, *, max_terms: int = 10) -> List[str]:
    text = re.sub(r"\s+", "", str(value or "")).strip()
    if not text:
        return []
    terms: List[str] = []
    for phrase in _SEARCH_TERM_HINTS:
        if phrase in text:
            _append_unique_text(terms, phrase, max_items=max_terms)
    parts = re.split(
        r"[\s,;:!?，。；：！？、/\\|()\[\]{}（）【】《》\"']+|"
        r"(?:是否|哪些|怎么|如何|当前|现在|其中|对于|关于|以及|或者|并且|同时|只有|判断为|"
        r"本章|核心|问题|回答|中|里|的|与|和|及|比|更|才|可|向好|确定性|短期|放量)",
        text,
    )
    for part in parts:
        _append_unique_text(terms, part, max_items=max_terms)
    return terms[:max_terms]


def _search_terms_from_value(value: Any, *, max_terms: int = 10) -> List[str]:
    raw = re.sub(r"\s+", " ", str(value or "")).strip()
    if not raw:
        return []
    terms: List[str] = []
    if len(raw) <= 18 and not re.search(r"[?？。；;:：]", raw):
        _append_unique_text(terms, raw, max_items=max_terms)
    for term in _anchor_terms_from_text(raw, max_terms=max_terms):
        _append_unique_text(terms, term, max_items=max_terms)
    return terms[:max_terms]


def _terms_for_chapter(chapter: Dict[str, Any], goal: Dict[str, Any]) -> List[str]:
    terms: List[str] = []
    for value in _as_list(goal.get("must_have_terms")) + _as_list(goal.get("expected_metrics")):
        for text in _search_terms_from_value(value):
            if text and text not in terms:
                terms.append(text)
    for value in [chapter.get("chapter_title"), chapter.get("core_question"), chapter.get("chapter_question")]:
        for text in _search_terms_from_value(value):
            if text and text not in terms:
                terms.append(text)
    return terms[:8]


_SEARCH_REQUIRED_FIELDS_BY_ROLE: Dict[str, List[str]] = {
    "metric": ["metric", "value", "unit", "period", "scope", "source_ref"],
    "source_check": ["source_ref", "source_title", "source_url"],
    "counter": ["counter_signal", "source_ref"],
    "case": ["company", "use_case", "deployment_scope", "source_ref"],
    "customer_case": ["company", "use_case", "deployment_scope", "source_ref"],
    "technology_product": ["capability", "constraint", "source_ref"],
    "filing": ["company", "filing_type", "period", "source_ref"],
    "support": ["fact", "source_ref"],
}


def _required_fields_for_proof_role(proof_role: Any) -> List[str]:
    role = str(proof_role or "").strip().lower()
    return list(_SEARCH_REQUIRED_FIELDS_BY_ROLE.get(role) or _SEARCH_REQUIRED_FIELDS_BY_ROLE["support"])


_SEARCH_FIELD_QUERY_TERMS: Dict[str, str] = {
    "metric": "metric",
    "value": "value",
    "unit": "unit",
    "period": "period",
    "scope": "scope",
    "source": "source",
    "source_ref": "source",
    "source_url": "source",
    "source_title": "source",
    "company": "company",
    "use_case": "use case",
    "deployment_scope": "deployment",
    "counter_signal": "risk",
    "filing_type": "filing",
}


_SEARCH_LANE_QUERY_TERMS: Dict[str, str] = {
    "official_data": "official",
    "market_research": "research report",
    "customer_case": "customer case",
    "counter_evidence": "risk",
    "filing_company": "filing",
    "filing": "filing",
    "association": "association",
    "technology_product": "product",
    "technical_standard": "standard",
}


def _search_query_contract_terms(
    *,
    proof_role: str,
    required_fields: Sequence[Any],
    lane_targets: Sequence[Any],
    source_priority: Sequence[Any],
    max_terms: int = 7,
) -> Dict[str, Any]:
    field_terms: List[str] = []
    source_terms: List[str] = []

    role_terms = {
        "metric": ["metric"],
        "source_check": ["source"],
        "counter": ["risk"],
        "case": ["customer case"],
        "filing": ["filing"],
        "technology_product": ["product"],
        "support": ["evidence"],
    }.get(proof_role, ["evidence"])

    for field in required_fields:
        term = _SEARCH_FIELD_QUERY_TERMS.get(str(field or "").strip().lower())
        if term:
            _append_unique_text(field_terms, term, max_items=5)

    for lane in list(lane_targets or []) + list(source_priority or []):
        term = _SEARCH_LANE_QUERY_TERMS.get(str(lane or "").strip().lower())
        if term:
            _append_unique_text(source_terms, term, max_items=4)

    query_terms: List[str] = []
    for term in [*role_terms, *source_terms, *field_terms]:
        _append_unique_text(query_terms, term, max_items=max_terms)

    return {
        "schema_version": "search_query_contract_v2",
        "proof_role": proof_role,
        "required_fields": [str(item) for item in required_fields if str(item or "").strip()],
        "field_query_terms": field_terms,
        "source_query_terms": source_terms,
        "query_terms": query_terms,
    }


def _claim_strength_ceiling_for_goal(goal: Dict[str, Any], proof_role: str) -> str:
    explicit = str(goal.get("claim_strength_ceiling") or goal.get("claim_strength") or "").strip().lower()
    if explicit:
        return explicit
    min_levels = [str(item or "").strip().upper() for item in _as_list(goal.get("required_source_levels") or goal.get("min_source_level"))]
    if proof_role in {"metric", "source_check"} and any(level in {"A", "B"} for level in min_levels or ["B"]):
        return "moderate"
    if "A" in min_levels:
        return "moderate"
    return "directional"


def _requirement_id_for_goal(goal: Dict[str, Any]) -> str:
    return str(
        goal.get("requirement_id")
        or goal.get("evidence_requirement_id")
        or goal.get("slot_id")
        or goal.get("goal_id")
        or ""
    ).strip()


def build_evidence_goals_for_chapter(chapter: Dict[str, Any], research_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    chapter_id = str(chapter.get("chapter_id") or "").strip()
    title = str(chapter.get("chapter_title") or "").strip()
    core_question = str(chapter.get("core_question") or chapter.get("chapter_question") or title).strip()
    existing = [dict(item) for item in _as_list(chapter.get("evidence_goals")) if isinstance(item, dict)]
    if not existing:
        for goal in _as_list(research_plan.get("evidence_goals")):
            if not isinstance(goal, dict):
                continue
            if str(goal.get("chapter_id") or goal.get("dimension_id") or "").strip() == chapter_id:
                existing.append(dict(goal))
            elif str(goal.get("chapter_title") or goal.get("dimension_name") or "").strip() == title:
                existing.append(dict(goal))
    if not existing:
        role_specs = [
            ("metric", "用指标、时间、范围和单位回答本章核心问题"),
            ("support", "寻找能直接支撑本章判断的高等级证据"),
            ("counter", "寻找可推翻或收窄本章判断边界的反向证据"),
            ("case", "寻找公司、客户、订单、采购或落地案例"),
            ("source_check", "核验关键事实的来源口径与权威出处"),
        ]
        existing = [
            {
                "goal_id": f"{chapter_id}_{role}",
                "question": f"{core_question}：{description}",
                "evidence_goal": f"{core_question}：{description}",
                "proof_role": role,
                "min_sources": max(1, int(chapter.get("min_ab_sources") or 2)) if role in {"metric", "source_check"} else 1,
            }
            for role, description in role_specs
        ]
    goals: List[Dict[str, Any]] = []
    for index, goal in enumerate(existing, start=1):
        copied = dict(goal)
        proof_role = str(copied.get("proof_role") or copied.get("role") or "").strip().lower()
        if proof_role not in PROOF_ROLES:
            evidence_type = str(copied.get("evidence_type") or copied.get("intent") or "").lower()
            proof_role = (
                "counter"
                if "risk" in evidence_type or "counter" in evidence_type
                else "case"
                if evidence_type in {"case", "customer_case", "procurement"}
                else "technology_product"
                if evidence_type in {"technology", "technology_product", "product_doc", "technical_standard", "patent"}
                else "filing"
                if evidence_type in {"filing", "company_filing", "annual_report", "prospectus"}
                else "source_check"
                if evidence_type in {"source_check", "official_verification"}
                else "metric"
                if evidence_type in {"data", "statistics", "metric", "market_data", "official_data"}
                else "support"
            )
        copied["goal_id"] = str(copied.get("goal_id") or copied.get("id") or f"{chapter_id}_goal_{index:02d}")
        copied["chapter_id"] = chapter_id
        copied["chapter_title"] = title
        copied["chapter_question"] = core_question
        copied["dimension_id"] = copied.get("dimension_id") or chapter_id
        copied["dimension_name"] = copied.get("dimension_name") or title
        copied["question"] = str(copied.get("question") or copied.get("evidence_goal") or core_question).strip()
        copied["proof_role"] = proof_role
        copied["required_evidence_mix"] = _as_list(copied.get("required_evidence_mix")) or _as_list(chapter.get("required_evidence_mix"))
        copied["lane_targets"] = _as_list(copied.get("lane_targets") or copied.get("lanes")) or _mix_to_lane_targets(copied["required_evidence_mix"], proof_role)
        copied["required_source_levels"] = _as_list(copied.get("required_source_levels")) or ["A", "B"]
        copied["requirement_id"] = _requirement_id_for_goal(copied) or copied["goal_id"]
        copied["required_fields"] = _as_list(copied.get("required_fields")) or _required_fields_for_proof_role(proof_role)
        copied["claim_strength_ceiling"] = _claim_strength_ceiling_for_goal(copied, proof_role)
        copied["min_sources"] = int(copied.get("min_sources") or (chapter.get("min_ab_sources") if proof_role in {"metric", "source_check"} else 1) or 1)
        goals.append(copied)
    roles_present = {str(goal.get("proof_role") or "").strip().lower() for goal in goals}
    supplemental_specs = [
        ("metric", "补齐本章关键指标、时间、范围和单位"),
        ("support", "补齐本章直接支撑证据"),
        ("counter", "补齐本章反证、风险和判断边界"),
        ("case", "补齐本章案例、订单、客户或采购证据"),
        ("technology_product", "补齐本章技术、产品、专利、标准或产品文档证据"),
        ("source_check", "补齐本章来源核验和权威出处"),
    ]
    for role, description in supplemental_specs:
        if role in roles_present:
            continue
        goals.append(
            {
                "goal_id": f"{chapter_id}_{role}",
                "requirement_id": f"{chapter_id}_{role}",
                "chapter_id": chapter_id,
                "chapter_title": title,
                "chapter_question": core_question,
                "dimension_id": chapter_id,
                "dimension_name": title,
                "question": f"{core_question}：{description}",
                "proof_role": role,
                "required_evidence_mix": _as_list(chapter.get("required_evidence_mix")),
                "lane_targets": _mix_to_lane_targets(_as_list(chapter.get("required_evidence_mix")), role),
                "required_source_levels": ["A", "B"],
                "required_fields": _required_fields_for_proof_role(role),
                "claim_strength_ceiling": _claim_strength_ceiling_for_goal({"required_source_levels": ["A", "B"]}, role),
                "min_sources": max(1, int(chapter.get("min_ab_sources") or 2)) if role in {"metric", "source_check"} else 1,
            }
        )
    return goals


def build_search_tasks_for_goal(
    *,
    chapter: Dict[str, Any],
    goal: Dict[str, Any],
    research_plan: Dict[str, Any],
) -> List[Dict[str, Any]]:
    chapter_id = str(chapter.get("chapter_id") or "").strip()
    title = str(chapter.get("chapter_title") or "").strip()
    core_question = str(chapter.get("core_question") or chapter.get("chapter_question") or title).strip()
    query = str(research_plan.get("query") or "").strip()
    goal_text = str(goal.get("question") or goal.get("evidence_goal") or core_question).strip()
    proof_role = str(goal.get("proof_role") or "support").strip().lower()
    if proof_role not in PROOF_ROLES:
        proof_role = "support"
    required_mix = _as_list(goal.get("required_evidence_mix")) or _as_list(chapter.get("required_evidence_mix"))
    explicit_lanes = _as_list(goal.get("lane_targets") or goal.get("lanes"))
    lanes = explicit_lanes[:3] if explicit_lanes else _mix_to_lane_targets(required_mix, proof_role)
    terms = _terms_for_chapter(chapter, goal)
    source_priority = _as_list(goal.get("source_priority"))
    research_object = str(research_plan.get("research_object") or "").strip()
    global_required_terms = _as_list(research_plan.get("global_required_terms"))
    query_hint = {
        "metric": "数据 统计 口径",
        "support": "权威 来源 报告",
        "counter": "风险 反证 失败",
        "case": "案例 客户 采购",
        "filing": "企业公告 财报 招股书",
        "source_check": "官方 原文 公告",
        "technology_product": "技术 产品 专利 标准",
    }[proof_role]
    topic_terms = _topic_seed_terms(query, chapter, goal)[:3]
    required_fields = _as_list(goal.get("required_fields")) or _required_fields_for_proof_role(proof_role)
    query_contract = _search_query_contract_terms(
        proof_role=proof_role,
        required_fields=required_fields,
        lane_targets=lanes,
        source_priority=source_priority,
    )
    query_focus = _compact_iqs_terms([research_object, *global_required_terms, *terms], max_terms=2, max_chars=16)
    base_query = _compose_iqs_query([topic_terms[:3], query_contract["query_terms"], query_focus, query_hint])
    task = {
        "task_id": f"{chapter_id}_{str(goal.get('goal_id') or proof_role).replace(' ', '_')}_{proof_role}",
        "agent": "iqs",
        "chapter_id": chapter_id,
        "chapter_title": title,
        "chapter_question": core_question,
        "dimension_id": chapter_id,
        "dimension_name": title,
        "query": base_query,
        "evidence_goal_id": goal.get("goal_id"),
        "evidence_goal": goal_text,
        "intent": _intent_for_proof_role(proof_role),
        "must_have_terms": terms,
        "forbidden_terms": _as_list(goal.get("forbidden_terms")) + _as_list(research_plan.get("global_forbidden_terms")),
        "source_priority": source_priority,
        "lane_targets": lanes,
        "min_source_level": _as_list(goal.get("required_source_levels")) or ["A", "B"],
        "required_evidence_mix": required_mix,
        "proof_role": proof_role,
        "requirement_id": _requirement_id_for_goal(goal),
        "required_fields": required_fields,
        "claim_strength_ceiling": goal.get("claim_strength_ceiling") or _claim_strength_ceiling_for_goal(goal, proof_role),
        "query_contract": {
            **query_contract,
            "requirement_id": _requirement_id_for_goal(goal),
            "lane_targets": lanes,
            "source_priority": source_priority,
        },
        "counter_evidence": proof_role == "counter",
        "research_object": research_object,
        "global_required_terms": global_required_terms,
        "hypothesis_id": goal.get("hypothesis_id"),
        "hypothesis_statement": goal.get("hypothesis_statement"),
        "proof_standard": goal.get("proof_standard") or "medium",
        "decision_use": goal.get("decision_use") or research_plan.get("decision_context") or "research",
        "evidence_type": goal.get("evidence_type") or proof_role,
    }
    tasks = [normalize_search_task(task)]
    report_family = str(
        research_plan.get("report_family")
        or research_plan.get("report_profile")
        or research_plan.get("report_mode")
        or ""
    ).strip().lower()
    force_deep_for_industry = (
        _env_flag("BRAIN_FORCE_DEEP_SEARCH_FOR_INDUSTRY", True)
        and report_family in {"industry_deep_report", "industry_scan_report", "deep_industry_report", "industry_report"}
    )
    if _env_flag("BRAIN_ENABLE_DEEP_SEARCH_VARIANTS", True) or force_deep_for_industry:
        deep_hint = {
            "metric": "官方统计 原始表",
            "support": "协会 白皮书 研报",
            "counter": "失败案例 价格下行",
            "case": "客户认证 中标",
            "filing": "年报 公告 招股书",
            "source_check": "发布机构 披露日期",
            "technology_product": "产品文档 技术标准 专利",
        }[proof_role]
        deep_task = {
            **task,
            "task_id": f"{task['task_id']}_deep",
            "query": _compose_iqs_query([base_query, deep_hint]),
            "evidence_goal": f"{goal_text}；补充交叉验证、反证和原始口径",
            "deep_search_variant": True,
            "prefer_deep": True,
            "deep_reason": "industry_deep_report_evidence_mix" if force_deep_for_industry else "deep_search_variant",
            "engineTypes": _deep_repair_engines(),
            "retrieval_mode": "deep",
            "retrieval_reason": "industry_deep_report_evidence_mix" if force_deep_for_industry else "deep_search_variant",
            "primary_provider": "iqs_deep",
            "fallback_providers": ["iqs_normal"],
            "source_priority": _as_list(source_priority) + ["official", "filing", "association", "research_report"],
        }
        tasks.append(normalize_search_task(deep_task))
    return tasks


def expand_search_tasks_from_chapters(research_plan: Dict[str, Any], report_blueprint: Dict[str, Any]) -> Dict[str, Any]:
    tasks: List[Dict[str, Any]] = []
    goals: List[Dict[str, Any]] = []
    for chapter in _as_list(_as_dict(report_blueprint).get("chapters")):
        if not isinstance(chapter, dict):
            continue
        chapter_goals = build_evidence_goals_for_chapter(chapter, research_plan)
        goals.extend(chapter_goals)
        for goal in chapter_goals:
            tasks.extend(build_search_tasks_for_goal(chapter=chapter, goal=goal, research_plan=research_plan))
    updated = dict(research_plan)
    updated["chapters"] = _as_list(_as_dict(report_blueprint).get("chapters"))
    updated["evidence_goals"] = goals
    normalized_tasks: List[Dict[str, Any]] = []
    research_object = str(updated.get("research_object") or "").strip()
    global_required_terms = _as_list(updated.get("global_required_terms"))
    for task in tasks:
        copied = dict(task)
        if research_object and not copied.get("research_object"):
            copied["research_object"] = research_object
        if global_required_terms and not copied.get("global_required_terms"):
            copied["global_required_terms"] = global_required_terms
        normalized_tasks.append(copied)
    updated["search_tasks"] = normalized_tasks
    return updated


def _task_group_key(task: Dict[str, Any]) -> str:
    intent = str(task.get("intent") or "").strip().lower()
    if intent:
        return intent
    source_priority = [str(item).strip().lower() for item in _as_list(task.get("source_priority")) if str(item).strip()]
    return source_priority[0] if source_priority else "general"


def _role_for_lane_type(lane_type: str) -> str:
    cleaned = str(lane_type or "").strip().lower()
    if cleaned in IQS_LANE_TO_ROLE:
        return IQS_LANE_TO_ROLE[cleaned]
    if cleaned in IQS_ROLE_CONFIGS:
        return cleaned
    return ""


def _infer_lane_types_for_task(task: Dict[str, Any]) -> List[str]:
    explicit = [
        str(item).strip().lower()
        for item in _as_list(task.get("lane_targets") or task.get("lanes"))
        if str(item).strip()
    ]
    inferred = [item for item in explicit if item in IQS_LANE_TO_ROLE or item in IQS_ROLE_CONFIGS]
    if inferred:
        return inferred[:3]

    text = " ".join(
        [
            str(task.get("intent") or ""),
            str(task.get("evidence_type") or ""),
            str(task.get("proof_role") or ""),
            " ".join(str(item) for item in _as_list(task.get("source_priority"))),
            str(task.get("query") or ""),
            str(task.get("evidence_goal") or ""),
        ]
    ).lower()
    technical_needles = (
        "技术瓶颈",
        "技术",
        "专利",
        "铰链",
        "折叠屏",
        "柔性屏",
        "utg",
        "oled",
        "良率",
        "量产",
        "验证",
        "hinge",
        "foldable",
        "crease",
        "yield",
        "patent",
        "technical",
        "technology",
    )
    lane_scores: Dict[str, int] = {}
    for lane_type, config in IQS_LANE_TYPE_CONFIGS.items():
        score = 0
        for term in list(config.get("intents") or []) + list(config.get("source_priority") or []):
            if str(term).lower() and str(term).lower() in text:
                score += 2
        for term in list(config.get("query_terms") or []):
            if str(term).lower() and str(term).lower() in text:
                score += 1
        if score:
            lane_scores[lane_type] = score
    if any(needle in text for needle in technical_needles):
        lane_scores["technology_product"] = max(lane_scores.get("technology_product", 0), 8)
    if not lane_scores:
        return ["official_data"] if str(task.get("proof_role") or "").lower() in {"metric", "source_check"} else ["market_research"]
    ordered = sorted(lane_scores, key=lambda item: lane_scores[item], reverse=True)
    if str(task.get("counter_evidence") or "").lower() in {"1", "true", "yes"} or str(task.get("proof_role") or "").lower() == "counter":
        if "news_event" not in ordered:
            ordered.append("news_event")
    return ordered[:3]


def _task_for_lane(task: Dict[str, Any], lane_type: str, role_key: str) -> Dict[str, Any]:
    copied = dict(task)
    config = IQS_LANE_TYPE_CONFIGS.get(lane_type, {})
    query_terms = [str(item) for item in list(config.get("query_terms") or [])[:2] if str(item).strip()]
    query = str(copied.get("query") or "").strip()
    if query_terms and not any(term.lower() in query.lower() for term in query_terms):
        copied["query"] = _compose_iqs_query([query, query_terms])
    else:
        copied["query"] = _compose_iqs_query([query])
    copied["scheduled_lane_type"] = lane_type
    copied["scheduled_lane"] = role_key
    copied["lane_focus"] = config.get("label") or lane_type
    copied.setdefault("lane_targets", [lane_type])
    source_priority = [str(item) for item in _as_list(copied.get("source_priority")) if str(item).strip()]
    for item in list(config.get("source_priority") or [])[:4]:
        if item not in source_priority:
            source_priority.append(str(item))
    copied["source_priority"] = source_priority[:6]
    copied = _apply_retrieval_routing_to_task(copied, lane_type=lane_type)
    copied["query_package"] = build_query_package(
        copied,
        query=str(copied.get("query") or ""),
        lane_type=lane_type,
        lane_focus=str(copied.get("lane_focus") or ""),
    )
    return copied


def assign_tasks_to_iqs_lanes(tasks: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    assigned: Dict[str, List[Dict[str, Any]]] = {lane: [] for lane in IQS_ROLE_ORDER}
    for raw in tasks:
        if not isinstance(raw, dict):
            continue
        task = dict(raw)
        preferred = str(task.get("agent") or "").strip().lower()
        if preferred in IQS_ROLE_CONFIGS:
            lane_type = _lane_type_for_role(preferred)
            assigned[preferred].append(_task_for_lane(task, lane_type, preferred) if lane_type else _apply_retrieval_routing_to_task(task))
            continue
        lane_types = _infer_lane_types_for_task(task)
        for lane_type in lane_types:
            role_key = _role_for_lane_type(lane_type)
            if role_key:
                assigned[role_key].append(_task_for_lane(task, lane_type, role_key))
    return assigned


def _dynamic_role_query(task: Dict[str, Any]) -> str:
    query = str(task.get("query") or "").strip()
    if not query:
        return ""
    return _compose_iqs_query([query])


def _proof_role_rank_for_lane(task: Dict[str, Any]) -> int:
    lane_type = str(task.get("scheduled_lane_type") or "").strip().lower()
    proof_role = str(task.get("proof_role") or "").strip().lower()
    evidence_type = str(task.get("evidence_type") or "").strip().lower()
    role = proof_role or evidence_type or "support"
    priority = {
        "official_data": ["metric", "source_check", "filing", "support", "counter", "case"],
        "filing_company": ["filing", "source_check", "metric", "support", "case", "counter"],
        "market_research": ["metric", "source_check", "filing", "support", "counter", "case"],
        "news_event": ["counter", "case", "support", "source_check", "metric", "filing"],
        "technology_product": ["technology_product", "source_check", "metric", "filing", "support", "case", "counter"],
        "customer_case": ["case", "metric", "source_check", "filing", "counter", "support"],
    }
    ordered = priority.get(lane_type) or ["metric", "source_check", "filing", "support", "counter", "case"]
    return ordered.index(role) if role in ordered else len(ordered)


def _task_role_key(task: Dict[str, Any]) -> str:
    return str(task.get("proof_role") or task.get("evidence_type") or "").strip().lower()


def _task_budget_group_key(task: Dict[str, Any], index: int) -> str:
    return str(
        task.get("chapter_id")
        or task.get("dimension_id")
        or task.get("hypothesis_id")
        or task.get("chapter_title")
        or f"task_{index}"
    ).strip()


def _task_budget_dedupe_key(task: Dict[str, Any], index: int) -> str:
    lane_type = str(task.get("scheduled_lane_type") or task.get("lane_type") or "").strip().lower()
    group_key = _task_budget_group_key(task, index)
    role = _task_role_key(task) or "support"
    return "|".join([lane_type, group_key, role])


def _task_budget_preference(task: Dict[str, Any], index: int) -> tuple:
    retrieval_mode = str(task.get("retrieval_mode") or "").strip().lower()
    deep_bonus = 0 if retrieval_mode in {"deep", "hybrid"} or bool(task.get("prefer_deep")) else 1
    query = str(task.get("query") or "")
    has_required_terms = 0 if _as_list(task.get("must_have_terms")) else 1
    is_blocking = 0 if _is_blocking_dropped_task(task) else 1
    return (
        _proof_role_rank_for_lane(task),
        is_blocking,
        deep_bonus,
        has_required_terms,
        min(len(query), 240),
        index,
    )


def _dedupe_lane_tasks_for_budget(lane_tasks: Sequence[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    grouped: Dict[str, List[tuple[int, Dict[str, Any]]]] = {}
    passthrough: List[Dict[str, Any]] = []
    for index, raw in enumerate(lane_tasks or []):
        if not isinstance(raw, dict):
            continue
        task = dict(raw)
        key = _task_budget_dedupe_key(task, index)
        if not key.strip("|"):
            passthrough.append(task)
            continue
        grouped.setdefault(key, []).append((index, task))
    deduped: List[tuple[int, Dict[str, Any]]] = []
    merged_out: List[Dict[str, Any]] = []
    for key, items in grouped.items():
        winner_index, winner = min(items, key=lambda item: _task_budget_preference(item[1], item[0]))
        deduped.append((winner_index, winner))
        for index, task in items:
            if index == winner_index:
                continue
            merged_out.append(
                {
                    **task,
                    "drop_reason": "deduped_by_chapter_proof_role",
                    "deduped_into_task_id": winner.get("task_id"),
                    "dedupe_key": key,
                }
            )
    deduped.sort(key=lambda item: item[0])
    return [task for _, task in deduped] + passthrough, merged_out


def _select_best_group_task(
    group: List[tuple[int, Dict[str, Any]]],
    roles: Sequence[str],
) -> Optional[tuple[int, Dict[str, Any]]]:
    wanted = {str(role or "").strip().lower() for role in roles if str(role or "").strip()}
    candidates = [item for item in group if _task_role_key(item[1]) in wanted]
    if not candidates:
        return None
    return min(candidates, key=lambda item: (_proof_role_rank_for_lane(item[1]), item[0]))


def _select_lane_tasks_for_budget(lane_tasks: Sequence[Dict[str, Any]], limit: int) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    tasks = [dict(task) for task in lane_tasks if isinstance(task, dict)]
    if limit <= 0:
        return [], tasks
    if len(tasks) <= limit:
        return tasks, []

    indexed = [(index, task) for index, task in enumerate(tasks)]
    grouped: Dict[str, List[tuple[int, Dict[str, Any]]]] = {}
    group_order: List[str] = []
    for index, task in indexed:
        group_key = _task_budget_group_key(task, index)
        if group_key not in grouped:
            grouped[group_key] = []
            group_order.append(group_key)
        grouped[group_key].append((index, task))

    for group in grouped.values():
        group.sort(key=lambda item: (_proof_role_rank_for_lane(item[1]), item[0]))

    selected: List[tuple[int, Dict[str, Any]]] = []
    selected_indices: set[int] = set()

    # Reserve scarce lane budget for proof-bearing tasks first.  This keeps
    # each chapter from losing the exact evidence roles QA later requires.
    for roles in (
        ("metric",),
        ("source_check",),
        ("filing", "company_filing"),
        ("case", "customer_case"),
        ("counter",),
        ("technology_product", "technical_standard", "product_doc"),
    ):
        if len(selected) >= limit:
            break
        for group_key in group_order:
            if len(selected) >= limit:
                break
            group = grouped.get(group_key) or []
            picked = _select_best_group_task(group, roles)
            if not picked or picked[0] in selected_indices:
                continue
            selected.append(picked)
            selected_indices.add(picked[0])
            group.remove(picked)

    while len(selected) < limit:
        progressed = False
        for group_key in group_order:
            group = grouped.get(group_key) or []
            if not group:
                continue
            picked = group.pop(0)
            selected.append(picked)
            selected_indices.add(picked[0])
            progressed = True
            if len(selected) >= limit:
                break
        if not progressed:
            break

    selected.sort(key=lambda item: item[0])
    dropped = [(index, task) for index, task in indexed if index not in selected_indices]
    return [task for _, task in selected], [task for _, task in dropped]


def _count_tasks_by_key(tasks: Sequence[Dict[str, Any]], key: str, *, fallback_key: str = "") -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for task in tasks:
        if not isinstance(task, dict):
            continue
        value = str(task.get(key) or (task.get(fallback_key) if fallback_key else "") or "unknown").strip() or "unknown"
        counts[value] = counts.get(value, 0) + 1
    return counts


def _is_blocking_dropped_task(task: Dict[str, Any]) -> bool:
    proof_role = str(task.get("proof_role") or task.get("evidence_type") or "").strip().lower()
    lane_type = str(task.get("scheduled_lane_type") or task.get("dropped_lane_type") or task.get("dropped_lane") or "").strip().lower()
    lane_targets = {str(item or "").strip().lower() for item in _as_list(task.get("lane_targets"))}
    return bool(
        proof_role in {"metric", "source_check", "filing", "company_filing", "counter", "case", "technology_product", "customer_case"}
        or lane_type in {"customer_case", "technology_product"}
        or lane_targets.intersection({"customer_case", "technology_product"})
    )


def _blocking_dropped_task_count(tasks: Sequence[Dict[str, Any]]) -> int:
    return len([task for task in tasks if isinstance(task, dict) and _is_blocking_dropped_task(task)])


def _apply_global_task_budget(
    scheduled_tasks: Sequence[Dict[str, Any]],
    budget: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """P2 hard global cap on total scheduled search tasks.

    The per-lane caps are multiplicative (lanes x per-lane x rounds) with a strict
    floor, so they cannot guarantee a total ceiling — this is the single hard gate.
    Selection is round-robin across lanes so the cap preserves cross-lane / proof-role
    coverage instead of starving later lanes by a flat truncation.

    Returns ``(kept, overflow)``; ``overflow`` carries a ``global_task_budget`` drop
    reason so the caller can record them as dropped tasks.
    """
    from collections import OrderedDict

    by_lane: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
    for task in scheduled_tasks:
        by_lane.setdefault(str(task.get("scheduled_lane") or ""), []).append(task)
    queues = [list(tasks) for tasks in by_lane.values()]
    kept: List[Dict[str, Any]] = []
    while len(kept) < budget and any(queues):
        for queue in queues:
            if not queue:
                continue
            kept.append(queue.pop(0))
            if len(kept) >= budget:
                break
    overflow = [
        {
            **task,
            "drop_reason": "global_task_budget",
            "dropped_lane": task.get("scheduled_lane"),
            "dropped_lane_type": task.get("scheduled_lane_type"),
        }
        for queue in queues
        for task in queue
    ]
    return kept, overflow


def _summarize_dropped_search_tasks(tasks: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_chapter: Dict[str, int] = {}
    by_proof_role: Dict[str, int] = {}
    by_lane_type: Dict[str, int] = {}
    examples: List[Dict[str, Any]] = []
    blocking_examples: List[Dict[str, Any]] = []
    blocking_count = 0
    for task in tasks:
        if not isinstance(task, dict):
            continue
        chapter_id = str(task.get("chapter_id") or task.get("hypothesis_id") or "unknown").strip() or "unknown"
        proof_role = str(task.get("proof_role") or task.get("evidence_type") or "unknown").strip() or "unknown"
        lane_type = str(task.get("scheduled_lane_type") or task.get("dropped_lane_type") or task.get("dropped_lane") or "unknown").strip() or "unknown"
        by_chapter[chapter_id] = by_chapter.get(chapter_id, 0) + 1
        by_proof_role[proof_role] = by_proof_role.get(proof_role, 0) + 1
        by_lane_type[lane_type] = by_lane_type.get(lane_type, 0) + 1
        is_blocking = _is_blocking_dropped_task(task)
        if is_blocking:
            blocking_count += 1
        if len(examples) < 12:
            example = {
                "task_id": task.get("task_id"),
                "chapter_id": task.get("chapter_id"),
                "hypothesis_id": task.get("hypothesis_id"),
                "proof_role": proof_role,
                "evidence_type": task.get("evidence_type"),
                "scheduled_lane": task.get("scheduled_lane"),
                "scheduled_lane_type": task.get("scheduled_lane_type"),
                "dropped_lane": task.get("dropped_lane"),
                "drop_reason": task.get("drop_reason"),
                "blocking": is_blocking,
                "query": _compact_text(task.get("query"), max_chars=160),
            }
            examples.append(example)
            if is_blocking and len(blocking_examples) < 8:
                blocking_examples.append(example)
    return {
        "dropped_count": len([task for task in tasks if isinstance(task, dict)]),
        "blocking_dropped_count": blocking_count,
        "by_chapter": by_chapter,
        "by_proof_role": by_proof_role,
        "by_lane_type": by_lane_type,
        "examples": examples,
        "blocking_examples": blocking_examples,
    }


def build_query_analysis(query: str, route: str, article_brief: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    brief = normalize_article_brief(article_brief, fallback_query=query) if article_brief else {}
    query = planning_query_from_brief(brief, fallback_query=query) if brief else str(query or "").strip()
    agents = _route_agents(route)
    max_queries = _effective_queries_per_agent()
    max_tasks_per_lane = _effective_iqs_lane_task_limit()
    research_plan = run_research_planner_agent(query=query, llm_config=build_llm_config("planning"), article_brief=brief)
    report_plan = _report_plan_from_research_plan(research_plan, query)
    report_blueprint = run_pre_layout_agent(
        query=query,
        research_plan=research_plan,
        report_plan=report_plan,
    )
    research_plan = expand_search_tasks_from_chapters(research_plan, report_blueprint)
    agent_queries: Dict[str, List[str]] = {"rag": []}
    agent_tasks: Dict[str, List[Dict[str, Any]]] = {}
    scheduled_tasks: List[Dict[str, Any]] = []
    dropped_tasks: List[Dict[str, Any]] = []
    for role_key in IQS_ROLE_ORDER:
        agent_queries[role_key] = []
        agent_tasks[role_key] = []

    if "rag" in agents:
        agent_queries["rag"] = _unique_strings(
            [
                query,
                f"从本地知识库提取「{query}」相关的行业框架、历史案例、内部资料和关键判断",
                f"查找「{query}」相关的盈利、亏损、成本、现金流、商业模式等可支撑报告的数据",
            ],
            max_items=max_queries,
        )
    dynamic_tasks = build_dynamic_iqs_tasks({"research_plan": research_plan})
    initial_lane_selection = _select_quality_first_initial_lanes(
        query=query,
        agents=agents,
        dynamic_tasks=dynamic_tasks,
        research_plan=research_plan,
    )
    selected_iqs_roles = set(_as_list(initial_lane_selection.get("selected_roles")))
    if initial_lane_selection.get("enabled"):
        agents = [
            *[agent for agent in agents if agent == "rag"],
            *[role for role in IQS_ROLE_ORDER if role in selected_iqs_roles],
        ]
    assigned_tasks = assign_tasks_to_iqs_lanes(dynamic_tasks)
    if any(role_key in agents for role_key in IQS_ROLE_ORDER) and any(assigned_tasks.values()):
        deduped_tasks: List[Dict[str, Any]] = []
        for role_key in IQS_ROLE_ORDER:
            if role_key not in agents:
                dropped_tasks.extend(
                    [
                        {
                            **task,
                            "dropped_lane": role_key,
                            "dropped_lane_type": task.get("scheduled_lane_type") or _lane_type_for_role(role_key),
                            "drop_reason": "initial_lane_not_selected",
                        }
                        for task in assigned_tasks.get(role_key, [])
                    ]
                )
                continue
            lane_tasks = assigned_tasks.get(role_key, [])
            lane_tasks, deduped = _dedupe_lane_tasks_for_budget(lane_tasks)
            deduped_tasks.extend(
                [
                    {
                        **task,
                        "deduped_lane": role_key,
                        "deduped_lane_type": task.get("scheduled_lane_type"),
                    }
                    for task in deduped
                ]
            )
            tasks, dropped = _select_lane_tasks_for_budget(lane_tasks, max_tasks_per_lane)
            agent_tasks[role_key] = tasks
            scheduled_tasks.extend([{**task, "scheduled_lane": role_key} for task in tasks])
            dropped_tasks.extend(
                [
                    {
                        **task,
                        "dropped_lane": role_key,
                        "dropped_lane_type": task.get("scheduled_lane_type"),
                        "drop_reason": "max_tasks_per_lane",
                    }
                    for task in dropped
                ]
            )
            agent_queries[role_key] = _unique_strings([_dynamic_role_query(task) for task in tasks], max_items=max_queries)
    else:
        deduped_tasks = []
    # P2: hard global budget on total search tasks. Per-lane caps multiply and
    # cannot bound the total (the explosion that drags the run into timeout
    # fail-open). Default 48 = a generous safety ceiling that prevents the
    # 140-task explosion while leaving headroom; set ``BRAIN_GLOBAL_MAX_SEARCH_TASKS``
    # lower (e.g. 32, the validated thin value) or 0 to disable entirely.
    global_task_budget = _env_int("BRAIN_GLOBAL_MAX_SEARCH_TASKS", 48, min_value=0, max_value=400)
    global_budget_applied = False
    if global_task_budget > 0 and len(scheduled_tasks) > global_task_budget:
        scheduled_tasks, _budget_overflow = _apply_global_task_budget(scheduled_tasks, global_task_budget)
        dropped_tasks.extend(_budget_overflow)
        global_budget_applied = True
        # Rebuild the per-lane dispatch structures (agent_tasks / agent_queries are
        # what actually drive IQS calls) so the cap cuts real search work, not just
        # the reported count.
        capped_by_lane: Dict[str, List[Dict[str, Any]]] = {}
        for task in scheduled_tasks:
            capped_by_lane.setdefault(str(task.get("scheduled_lane") or ""), []).append(task)
        for role_key in list(agent_tasks.keys()):
            kept_lane_tasks = capped_by_lane.get(role_key, [])
            agent_tasks[role_key] = kept_lane_tasks
            agent_queries[role_key] = _unique_strings(
                [_dynamic_role_query(task) for task in kept_lane_tasks], max_items=max_queries
            )
    dropped_summary = _summarize_dropped_search_tasks(dropped_tasks)
    dropped_by_lane = _count_tasks_by_key(dropped_tasks, "scheduled_lane_type", fallback_key="dropped_lane_type")
    scheduled_by_lane = _count_tasks_by_key(scheduled_tasks, "scheduled_lane_type")
    recommended_min_tasks_per_lane = {
        lane: int(scheduled_by_lane.get(lane, 0)) + int(dropped_by_lane.get(lane, 0))
        for lane in sorted(set(scheduled_by_lane) | set(dropped_by_lane))
        if int(dropped_by_lane.get(lane, 0)) > 0
    }
    related_questions = _unique_strings(
        [
            f"{query} 的核心结论是什么？",
            f"{query} 有哪些可以写进报告的关键数据？",
            f"{query} 的盈利、亏损、估值或行情数据是否足够？",
            f"{query} 还缺哪些证据需要补充？",
        ],
        max_items=5,
    )
    return {
        "original_query": query,
        "article_brief": brief,
        "planning_query": query,
        "route": route,
        "report_plan": report_plan,
        "report_blueprint": report_blueprint,
        "research_plan": research_plan,
        "research_type": research_plan.get("research_type"),
        "dynamic_dimensions": research_plan.get("dimensions"),
        "dynamic_search_tasks": dynamic_tasks,
        "search_task_schedule": {
            "max_tasks_per_lane": max_tasks_per_lane,
            "global_task_budget": global_task_budget,
            "global_budget_applied": global_budget_applied,
            "scheduled_tasks": scheduled_tasks,
            "dropped_tasks": dropped_tasks,
            "deduped_tasks": deduped_tasks,
            "scheduled_count": len(scheduled_tasks),
            "dropped_count": len(dropped_tasks),
            "deduped_count": len(deduped_tasks),
            "scheduled_by_lane": scheduled_by_lane,
            "scheduled_by_proof_role": _count_tasks_by_key(scheduled_tasks, "proof_role"),
            "scheduled_by_retrieval_mode": _count_tasks_by_key(scheduled_tasks, "retrieval_mode"),
            "scheduled_by_primary_provider": _count_tasks_by_key(scheduled_tasks, "primary_provider"),
            "dropped_by_lane": dropped_by_lane,
            "dropped_by_proof_role": _count_tasks_by_key(dropped_tasks, "proof_role"),
            "deduped_by_lane": _count_tasks_by_key(deduped_tasks, "scheduled_lane_type", fallback_key="deduped_lane_type"),
            "deduped_by_proof_role": _count_tasks_by_key(deduped_tasks, "proof_role"),
            "dropped_blocking_count": _blocking_dropped_task_count(dropped_tasks),
            "dropped_summary": dropped_summary,
            "recommended_min_tasks_per_lane": recommended_min_tasks_per_lane,
            "initial_lane_selection": initial_lane_selection,
        },
        "target_agents": agents,
        "related_questions": related_questions,
        "agent_queries": {key: value for key, value in agent_queries.items() if value},
        "agent_tasks": {key: value for key, value in agent_tasks.items() if value},
    }


def _queries_for_agent(state: BrainAgentState, agent_key: str) -> List[str]:
    analysis = _as_dict(state.get("query_analysis"))
    agent_queries = _as_dict(analysis.get("agent_queries"))
    queries = _unique_strings(agent_queries.get(agent_key) or [], max_items=_effective_queries_per_agent())
    if queries:
        return queries
    query = str(state.get("query") or "").strip()
    return [query] if query else []


def _tasks_for_agent(state: BrainAgentState, agent_key: str) -> List[Dict[str, Any]]:
    analysis = _as_dict(state.get("query_analysis"))
    agent_tasks = _as_dict(analysis.get("agent_tasks"))
    tasks = [
        normalize_search_task(task, fallback_index=index)
        for index, task in enumerate(_as_list(agent_tasks.get(agent_key)), start=1)
        if isinstance(task, dict)
    ]
    return [task for task in tasks if task.get("query")]


def _search_options_for_task(state: BrainAgentState, task: Dict[str, Any], phase: str) -> Dict[str, Any]:
    options = _iqs_search_options_for_phase(state, phase)
    research_plan = _as_dict(_as_dict(state.get("query_analysis")).get("research_plan")) or _as_dict(state.get("research_plan"))
    task_options = _as_dict(task.get("search_options"))
    options.update(task_options)
    options["session_id"] = str(state.get("session_id") or options.get("session_id") or "").strip()
    options["research_plan"] = research_plan
    options["search_task"] = copy.deepcopy(task)
    options["task_id"] = task.get("task_id")
    options["dimension_id"] = task.get("dimension_id")
    options["dimension_name"] = task.get("dimension_name")
    options["chapter_id"] = task.get("chapter_id")
    options["chapter_title"] = task.get("chapter_title")
    options["chapter_question"] = task.get("chapter_question")
    options["evidence_goal"] = task.get("evidence_goal")
    options["evidence_goal_id"] = task.get("evidence_goal_id")
    options["must_have_terms"] = _as_list(task.get("must_have_terms"))
    options["forbidden_terms"] = _as_list(task.get("forbidden_terms"))
    options["source_priority"] = _as_list(task.get("source_priority"))
    options["research_object"] = task.get("research_object") or research_plan.get("research_object")
    options["global_required_terms"] = _as_list(task.get("global_required_terms")) or _as_list(research_plan.get("global_required_terms"))
    options["hypothesis_id"] = task.get("hypothesis_id")
    options["hypothesis_statement"] = task.get("hypothesis_statement")
    options["proof_role"] = task.get("proof_role")
    options["proof_standard"] = task.get("proof_standard")
    options["evidence_type"] = task.get("evidence_type")
    options["lane_targets"] = _as_list(task.get("lane_targets"))
    options["retrieval_mode"] = task.get("retrieval_mode") or "normal"
    options["retrieval_reason"] = task.get("retrieval_reason")
    options["primary_provider"] = task.get("primary_provider")
    options["fallback_providers"] = _as_list(task.get("fallback_providers"))
    options["provider"] = task.get("provider")
    if _as_list(task.get("allowed_domains")):
        options["allowed_domains"] = _as_list(task.get("allowed_domains"))
    options["prefer_deep"] = bool(task.get("prefer_deep"))
    options["deep_reason"] = task.get("deep_reason")
    options["deep_status"] = task.get("deep_status")
    if _as_list(task.get("engineTypes")):
        options["engineTypes"] = _as_list(task.get("engineTypes"))
    if options.get("prefer_deep"):
        options["enable_batch_search"] = False
    options["min_source_level"] = _as_list(task.get("min_source_level"))
    options["required_evidence_mix"] = _as_list(task.get("required_evidence_mix"))
    options["scheduled_lane_type"] = task.get("scheduled_lane_type")
    options["counter_evidence"] = task.get("counter_evidence")
    options["decision_use"] = task.get("decision_use")
    if (
        phase == "followup"
        and _env_flag("BRAIN_FOLLOWUP_ADAPTIVE_SEARCH_BUDGET", True)
        and not _strict_quality_mode()
        and not _continuous_evidence_loop_mode()
    ):
        high_priority = _followup_priority(task) <= 15
        caps = {
            "max_queries": _env_int("BRAIN_FOLLOWUP_FAST_MAX_QUERIES", 4),
            "max_search_tasks": _env_int("BRAIN_FOLLOWUP_FAST_MAX_SEARCH_TASKS", 10),
            "results_per_query": _env_int("BRAIN_FOLLOWUP_FAST_RESULTS_PER_QUERY", 80),
            "rerank_top_k": _env_int("BRAIN_FOLLOWUP_FAST_RERANK_TOP_K", 24),
            "rerank_max_docs": _env_int("BRAIN_FOLLOWUP_FAST_RERANK_MAX_DOCS", 60),
            "rerank_prefilter_max_docs": _env_int("BRAIN_FOLLOWUP_FAST_RERANK_PREFILTER_MAX_DOCS", 60),
        }
        for key, cap in caps.items():
            if high_priority and key in {"max_queries", "max_search_tasks"}:
                continue
            try:
                current = int(options.get(key) or cap)
            except (TypeError, ValueError):
                current = cap
            options[key] = max(1, min(current, max(1, cap)))
    if (
        phase == "initial"
        and _env_flag("BRAIN_INITIAL_LANE_ADAPTIVE_SEARCH_BUDGET", True)
        and not _strict_quality_mode()
    ):
        role = str(task.get("proof_role") or task.get("evidence_type") or "").strip().lower()
        high_priority = role in {"metric", "source_check", "filing"}
        caps = {
            "max_queries": _env_int("BRAIN_INITIAL_LANE_MAX_QUERIES", 4 if high_priority else 3),
            "max_search_tasks": _env_int("BRAIN_INITIAL_LANE_MAX_SEARCH_TASKS", 10 if high_priority else 8),
            "results_per_query": _env_int("BRAIN_INITIAL_LANE_RESULTS_PER_QUERY", 50 if high_priority else 40),
            "rerank_top_k": _env_int("BRAIN_INITIAL_LANE_RERANK_TOP_K", 18 if high_priority else 14),
            "rerank_max_docs": _env_int("BRAIN_INITIAL_LANE_RERANK_MAX_DOCS", 60 if high_priority else 45),
            "rerank_prefilter_max_docs": _env_int("BRAIN_INITIAL_LANE_RERANK_PREFILTER_MAX_DOCS", 60 if high_priority else 45),
        }
        for key, cap in caps.items():
            try:
                current = int(options.get(key) or cap)
            except (TypeError, ValueError):
                current = cap
            options[key] = max(1, min(current, max(1, cap)))
        if _env_flag("BRAIN_INITIAL_LANE_DISABLE_SELF_REFINE", True):
            options["enable_self_refine"] = False
    return options


def _iqs_search_options_for_phase(state: BrainAgentState, phase: str) -> Dict[str, Any]:
    options = dict(state.get("web_search_options") or {})
    options["search_profile"] = phase
    options.setdefault("enable_self_refine", _env_flag("IQS_ENABLE_SELF_REFINE", True))
    options.setdefault("enable_batch_search", True)
    if phase == "initial":
        options.setdefault("max_queries", _env_int("IQS_INITIAL_MAX_QUERIES", 12))
        options.setdefault("max_search_tasks", _env_int("IQS_INITIAL_MAX_SEARCH_TASKS", 80))
        options.setdefault("results_per_query", _env_int("IQS_INITIAL_RESULTS_PER_QUERY", 100))
        options.setdefault("rerank_top_k", _env_int("IQS_INITIAL_RERANK_TOP_K", 80))
        options.setdefault("rerank_max_docs", _env_int("IQS_INITIAL_RERANK_MAX_DOCS", 240))
        options.setdefault("rerank_prefilter_max_docs", _env_int("IQS_INITIAL_RERANK_PREFILTER_MAX_DOCS", 160))
    elif phase == "followup":
        options.setdefault("max_queries", _env_int("IQS_FOLLOWUP_MAX_QUERIES", 10))
        options.setdefault("max_search_tasks", _env_int("IQS_FOLLOWUP_MAX_SEARCH_TASKS", 48))
        options.setdefault("results_per_query", _env_int("IQS_FOLLOWUP_RESULTS_PER_QUERY", 100))
        options.setdefault("rerank_top_k", _env_int("IQS_FOLLOWUP_RERANK_TOP_K", 70))
        options.setdefault("rerank_max_docs", _env_int("IQS_FOLLOWUP_RERANK_MAX_DOCS", 180))
        options.setdefault("rerank_prefilter_max_docs", _env_int("IQS_FOLLOWUP_RERANK_PREFILTER_MAX_DOCS", 160))
    if _strict_quality_mode():
        floors = {
            "max_queries": 12 if phase == "initial" else 10,
            "max_search_tasks": 80 if phase == "initial" else 48,
            "results_per_query": 100,
            "rerank_top_k": 80 if phase == "initial" else 70,
            "rerank_max_docs": 240 if phase == "initial" else 180,
            "rerank_prefilter_max_docs": 160,
        }
        for key, floor in floors.items():
            try:
                options[key] = max(int(options.get(key) or 0), floor)
            except (TypeError, ValueError):
                options[key] = floor
        options["enable_self_refine"] = True
    elif _continuous_evidence_loop_mode():
        floors = {
            "max_queries": 6 if phase == "initial" else 6,
            "max_search_tasks": 24 if phase == "initial" else 18,
            "results_per_query": 80,
            "rerank_top_k": 40 if phase == "initial" else 24,
            "rerank_max_docs": 100 if phase == "initial" else 60,
            "rerank_prefilter_max_docs": 100 if phase == "initial" else 60,
        }
        for key, floor in floors.items():
            try:
                options[key] = max(int(options.get(key) or 0), floor)
            except (TypeError, ValueError):
                options[key] = floor
        options["enable_self_refine"] = True
    return options


def build_llm_config(task_name: str = "writer") -> Dict[str, Any]:
    return dict(build_llm_config_for_task(task_name))


def prepare_query_node(state: BrainAgentState) -> BrainAgentState:
    article_brief = normalize_article_brief(state.get("article_brief"), fallback_query=state.get("query")) if state.get("article_brief") else {}
    query = planning_query_from_brief(article_brief, fallback_query=extract_query_from_state(state)) if article_brief else extract_query_from_state(state)
    if not query:
        return {
            "errors": ["查询不能为空"],
            "metadata": {
                **dict(state.get("metadata") or {}),
                "agent_name": AGENT_NAME,
                "framework": "langgraph",
                "agent_stage": "decompose_query",
            },
        }
    _progress("brain", "收到问题，准备研究规划", query=query)
    return {
        "query": query,
        "article_brief": article_brief,
        "metadata": {
            **dict(state.get("metadata") or {}),
            "article_brief": article_brief,
            "agent_name": AGENT_NAME,
            "agent_description": AGENT_DESCRIPTION,
            "framework": "langgraph",
            "children": (
                (["industry_rag_agent"] if _local_rag_enabled() else [])
                + [IQS_ROLE_CONFIGS[key]["child"] for key in IQS_ROLE_ORDER]
            ),
            "agent_stage": "decompose_query",
        },
    }


def route_node(state: BrainAgentState) -> BrainAgentState:
    if state.get("errors"):
        return {}
    started = time.perf_counter()
    _progress("brain", "路由与动态研究规划开始", query=state.get("query"))
    route, reason = route_query(str(state.get("query") or ""), str(state.get("route") or os.getenv("BRAIN_AGENT_ROUTE", "auto")))
    article_brief = normalize_article_brief(state.get("article_brief"), fallback_query=state.get("query")) if state.get("article_brief") else {}
    query_analysis = build_query_analysis(str(state.get("query") or ""), route, article_brief=article_brief)
    research_plan = _as_dict(query_analysis.get("research_plan"))
    report_blueprint = _as_dict(query_analysis.get("report_blueprint")) or run_pre_layout_agent(
        query=str(state.get("query") or ""),
        research_plan=research_plan,
        report_plan=_as_dict(query_analysis.get("report_plan")),
    )
    schedule = _as_dict(query_analysis.get("search_task_schedule"))
    scheduled_total = len(_as_list(schedule.get("scheduled_tasks")))
    dropped_total = len(_as_list(schedule.get("dropped_tasks")))
    _progress(
        "brain",
        "路由与研究规划完成",
        route=route,
        chapters=len(_as_list(report_blueprint.get("chapters"))),
        tasks=scheduled_total,
        dropped=dropped_total,
        elapsed=f"{time.perf_counter() - started:.1f}s",
    )
    return {
        "route": route,
        "route_reason": reason,
        "article_brief": article_brief,
        "query_analysis": query_analysis,
        "research_plan": research_plan,
        "report_blueprint": report_blueprint,
        "search_tasks": _as_list(research_plan.get("search_tasks")),
        "search_task_schedule": _as_dict(query_analysis.get("search_task_schedule")),
        "agent_trace": [
            {
                "agent": AGENT_NAME,
                "stage": "route",
                "route": route,
                "reason": reason,
                "research_type": research_plan.get("research_type"),
                "report_blueprint_chapters": len(_as_list(report_blueprint.get("chapters"))),
            }
        ],
        "metadata": {
            **dict(state.get("metadata") or {}),
            "agent_stage": "route",
            "route": route,
            "route_reason": reason,
            "query_analysis": query_analysis,
            "report_blueprint": report_blueprint,
            "search_task_schedule": _as_dict(query_analysis.get("search_task_schedule")),
        },
    }


def run_local_rag_agent_node(state: BrainAgentState) -> BrainAgentState:
    if not _local_rag_enabled():
        _progress("rag", "Local RAG disabled; skipped")
        return {
            "agent_trace": [
                {
                    "agent": "industry_rag_agent",
                    "stage": "child_agent",
                    "status": "skipped",
                    "reason": "local_rag_disabled",
                }
            ]
        }
    queries = _queries_for_agent(state, "rag")
    if not queries:
        return {}

    node_started = time.perf_counter()
    _progress("rag", "本地 RAG 开始", queries=len(queries))
    outputs: BrainAgentState = {}
    errors: List[str] = []
    query_results: List[Dict[str, Any]] = []
    evidence: List[Dict[str, Any]] = []
    answer_blocks: List[str] = []
    confidences: List[float] = []

    for index, query in enumerate(queries, start=1):
        query_started = time.perf_counter()
        _progress("rag", "检索开始", index=f"{index}/{len(queries)}", query=query)
        try:
            child_state = run_rag_agent(
                query,
                session_id=str(state.get("session_id") or "").strip(),
                args_overrides=dict(state.get("args_overrides") or {}),
            )
            query_results.append({"query": query, "state": child_state})
            child_answer = str(child_state.get("answer_text") or "").strip()
            raw_output = _as_dict(child_state.get("raw_output"))
            answer_payload = _as_dict(raw_output.get("answer"))
            confidences.append(_clip_confidence(answer_payload.get("confidence"), 0.0))
            child_evidence = [item for item in list(raw_output.get("evidence") or child_state.get("evidence") or []) if isinstance(item, dict)]
            evidence.extend(child_evidence)
            child_errors = [str(item) for item in child_state.get("errors") or [] if str(item).strip()]
            if child_answer and (child_evidence or not child_errors):
                answer_blocks.append(f"子问题：{query}\n{child_answer}")
            errors.extend([f"本地 RAG 子智能体：{item}" for item in child_errors])
            _progress(
                "rag",
                "检索完成",
                index=f"{index}/{len(queries)}",
                evidence=len(child_evidence),
                errors=len(child_errors),
                elapsed=f"{time.perf_counter() - query_started:.1f}s",
            )
        except Exception as exc:
            logger.exception("Local RAG child failed", extra={"query": query})
            errors.append(f"本地 RAG 子智能体失败（{query}）：{exc}")
            _progress("rag", "检索失败", index=f"{index}/{len(queries)}", error=exc, elapsed=f"{time.perf_counter() - query_started:.1f}s")

    avg_conf = round(sum(confidences) / max(len([item for item in confidences if item > 0]), 1), 4) if confidences else 0.0
    answer_text = "\n\n".join(answer_blocks).strip()
    if query_results or answer_text or evidence:
        outputs["local_state"] = {
            "answer_text": answer_text,
            "query_results": query_results,
            "raw_output": {
                "answer": {
                    "status": "answered" if answer_text and evidence else "insufficient_evidence",
                    "confidence": avg_conf,
                    "answer": answer_text,
                    "gaps": [],
                    "conflicts": [],
                },
                "evidence": evidence,
            },
            "evidence": evidence,
        }

    all_errors = errors
    if all_errors:
        outputs["errors"] = all_errors
    outputs["agent_trace"] = [
        {
            "agent": "industry_rag_agent",
            "stage": "child_agent",
            "status": "error" if all_errors else "ok",
            "reason": state.get("route_reason", ""),
        }
    ]
    _progress("rag", "本地 RAG 结束", evidence=len(evidence), elapsed=f"{time.perf_counter() - node_started:.1f}s")
    return outputs


def run_web_analysis_agent_node(state: BrainAgentState) -> BrainAgentState:
    queries = _queries_for_agent(state, "iqs")
    if not queries:
        return {}

    node_started = time.perf_counter()
    _progress("iqs", "通用 IQS 联网分析开始", queries=len(queries))
    outputs: BrainAgentState = {}
    errors: List[str] = []
    query_results: List[Dict[str, Any]] = []
    answer_blocks: List[str] = []
    search_results: List[Dict[str, Any]] = []
    page_results: List[Dict[str, Any]] = []
    key_sources: List[Dict[str, Any]] = []
    confidences: List[float] = []
    conclusion_blocks: List[str] = []
    evidence_blocks: List[str] = []
    inference_blocks: List[str] = []
    evidence_gaps: List[Any] = []

    for index, query in enumerate(queries, start=1):
        if _deadline_exceeded(state, min_remaining=1.0):
            timeout_payload = _deadline_timeout_payload(state, stage="web_analysis_agent")
            errors.append("联网分析达到报告墙钟预算，已停止提交新的 IQS 子问题。")
            outputs.setdefault("metadata", {})["live_timeout"] = timeout_payload
            break
        query_started = time.perf_counter()
        _progress("iqs", "通用联网检索开始", index=f"{index}/{len(queries)}", query=query)
        try:
            child_state = run_web_analysis_agent(
                query,
                search_options=_iqs_search_options_for_phase(state, "initial"),
                enable_llm_analysis=bool(state.get("enable_web_analysis", _env_flag("BRAIN_WEB_ENABLE_LLM_ANALYSIS", True))),
            )
            query_results.append({"query": query, "state": child_state})
            child_answer = str(child_state.get("answer_text") or "").strip()
            if child_answer:
                answer_blocks.append(f"子问题：{query}\n{child_answer}")
            raw_output = _as_dict(child_state.get("raw_output"))
            synthesis = _as_dict(raw_output.get("synthesis"))
            payload = _as_dict(synthesis.get("structured_payload"))
            confidences.append(_clip_confidence(payload.get("confidence") if payload.get("confidence") is not None else synthesis.get("confidence"), 0.0))
            search_results.extend([item for item in list(raw_output.get("search_results") or []) if isinstance(item, dict)])
            page_results.extend([item for item in list(raw_output.get("page_results") or []) if isinstance(item, dict)])
            key_sources.extend([item for item in list(payload.get("key_sources") or synthesis.get("key_sources") or []) if isinstance(item, dict)])
            errors.extend([f"联网分析子智能体：{item}" for item in child_state.get("errors") or []])
            _progress(
                "iqs",
                "通用联网检索完成",
                index=f"{index}/{len(queries)}",
                results=len(_as_list(raw_output.get("search_results"))),
                sources=len(_as_list(payload.get("key_sources") or synthesis.get("key_sources"))),
                elapsed=f"{time.perf_counter() - query_started:.1f}s",
            )
        except Exception as exc:
            logger.exception("Generic IQS child failed", extra={"query": query})
            errors.append(f"联网分析子智能体失败（{query}）：{exc}")
            _progress("iqs", "通用联网检索失败", index=f"{index}/{len(queries)}", error=exc, elapsed=f"{time.perf_counter() - query_started:.1f}s")

    avg_conf = round(sum(confidences) / max(len([item for item in confidences if item > 0]), 1), 4) if confidences else 0.0
    answer_text = "\n\n".join(answer_blocks).strip()
    if query_results or answer_text or search_results or page_results:
        outputs["web_state"] = {
            "answer_text": answer_text,
            "query_results": query_results,
            "raw_output": {
                "query": str(state.get("query") or ""),
                "search_options": dict(state.get("web_search_options") or {}),
                "search_results": search_results,
                "page_results": page_results,
                "synthesis": {
                    "type": "web_analysis_synthesis",
                    "source": "multi_query",
                    "structured_payload": {
                        "answer": {
                            "conclusion": "联网子问题结果如下",
                            "evidence": answer_text,
                            "inference": None,
                            "evidence_gap": [],
                        },
                        "confidence": avg_conf,
                        "key_sources": key_sources,
                        "limitations": {"data_recency": "多子问题联网结果", "coverage": "按问题拆解检索", "conflicts": None},
                    },
                    "confidence": avg_conf,
                    "key_sources": key_sources,
                    "limitations": {"data_recency": "多子问题联网结果", "coverage": "按问题拆解检索", "conflicts": None},
                },
            },
        }

    all_errors = errors
    if all_errors:
        outputs["errors"] = all_errors
    outputs["agent_trace"] = [
        {
            "agent": "web_analysis_agent",
            "stage": "child_agent",
            "status": "error" if all_errors else "ok",
            "reason": state.get("route_reason", ""),
        }
    ]
    _progress("iqs", "通用 IQS 联网分析结束", results=len(search_results), elapsed=f"{time.perf_counter() - node_started:.1f}s")
    return outputs


def _run_iqs_lane_task(
    *,
    state: BrainAgentState,
    config: Dict[str, Any],
    role_key: str,
    index: int,
    total: int,
    task: Dict[str, Any],
) -> Dict[str, Any]:
    query = str(task.get("query") or "").strip()
    task_started = time.perf_counter()
    _progress(
        "iqs-lane",
        "lane task started",
        lane=role_key,
        index=f"{index}/{total}",
        role=task.get("proof_role") or task.get("evidence_type"),
        query=query,
    )
    if _deadline_exceeded(state, min_remaining=1.0):
        return {
            "index": index,
            "status": "cancelled",
            "cancel_reason": "deadline_exceeded",
            "errors": [f"{config['label']} skipped because the report deadline was reached"],
            "live_timeout": _deadline_timeout_payload(state, stage=f"{role_key}_task"),
        }
    try:
        child_state = run_web_analysis_agent(
            query,
            search_options=_search_options_for_task(state, task, "initial"),
            enable_llm_analysis=bool(state.get("enable_web_analysis", _env_flag("BRAIN_WEB_ENABLE_LLM_ANALYSIS", True))),
        )
        child_answer = str(child_state.get("answer_text") or "").strip()
        answer_block = ""
        if child_answer:
            dimension_name = str(task.get("dimension_name") or task.get("dimension") or "").strip()
            answer_block = (
                f"research_dimension: {dimension_name or 'dynamic_task'}\n"
                f"evidence_task: {task.get('evidence_goal') or ''}\n"
                f"query: {query}\n{child_answer}"
            )
        raw_output = _as_dict(child_state.get("raw_output"))
        child_metadata = _as_dict(child_state.get("metadata"))
        readpage_meta = _as_dict(child_metadata.get("auto_readpage"))
        synthesis = _as_dict(raw_output.get("synthesis"))
        payload = _as_dict(synthesis.get("structured_payload"))
        answer_payload = _as_dict(payload.get("answer"))
        normalized_task_child = normalize_web_child_output(child_state, route="web", errors=[])
        task_points = [dict(point) for point in _as_list(normalized_task_child.get("raw_data_points")) if isinstance(point, dict)]
        for point in task_points:
            point["task_id"] = task.get("task_id")
            point["chapter_id"] = task.get("chapter_id")
            point["chapter_title"] = task.get("chapter_title")
            point["chapter_question"] = task.get("chapter_question")
            point["dimension_id"] = task.get("dimension_id")
            point["dimension_name"] = task.get("dimension_name") or task.get("dimension")
            point["evidence_goal"] = task.get("evidence_goal")
            point["evidence_goal_id"] = task.get("evidence_goal_id")
            point["must_have_terms"] = _as_list(task.get("must_have_terms"))
            point["forbidden_terms"] = _as_list(task.get("forbidden_terms"))
            point["source_priority"] = _as_list(task.get("source_priority"))
            point["research_object"] = task.get("research_object")
            point["global_required_terms"] = _as_list(task.get("global_required_terms"))
            point["search_task"] = dict(task)
        task_result = {
            "task": task,
            "status": normalized_task_child.get("status"),
            "confidence": normalized_task_child.get("confidence"),
            "data_points": len(task_points),
            "sources": len(_as_list(normalized_task_child.get("key_sources"))),
            "readpage_attempted": int(readpage_meta.get("attempted") or 0),
            "readpage_succeeded": int(readpage_meta.get("succeeded") or len(_as_list(raw_output.get("page_results")))),
        }
        errors = [f"{config['label']}: {item}" for item in child_state.get("errors") or []]
        _progress(
            "iqs-lane",
            "lane task completed",
            lane=role_key,
            index=f"{index}/{total}",
            points=len(task_points),
            sources=len(_as_list(normalized_task_child.get("key_sources"))),
            elapsed=f"{time.perf_counter() - task_started:.1f}s",
        )
        confidence_value = payload.get("confidence") if payload.get("confidence") is not None else synthesis.get("confidence")
        return {
            "index": index,
            "query_result": {"query": query, "task": task, "state": child_state},
            "answer_block": answer_block,
            "conclusion": str(answer_payload.get("conclusion") or "").strip(),
            "evidence": str(answer_payload.get("evidence") or "").strip(),
            "inference": str(answer_payload.get("inference") or "").strip(),
            "evidence_gaps": _as_list(answer_payload.get("evidence_gap")),
            "confidence": _clip_confidence(confidence_value, 0.0),
            "search_results": [item for item in list(raw_output.get("search_results") or []) if isinstance(item, dict)],
            "page_results": [item for item in list(raw_output.get("page_results") or []) if isinstance(item, dict)],
            "key_sources": [item for item in list(payload.get("key_sources") or synthesis.get("key_sources") or []) if isinstance(item, dict)],
            "raw_data_points": task_points,
            "task_result": task_result,
            "errors": errors,
        }
    except Exception as exc:
        logger.exception("IQS lane task failed", extra={"query": query, "lane": role_key})
        error = f"{config['label']} failed ({query}): {exc}"
        _progress(
            "iqs-lane",
            "lane task failed",
            lane=role_key,
            index=f"{index}/{total}",
            error=exc,
            elapsed=f"{time.perf_counter() - task_started:.1f}s",
        )
        return {"index": index, "errors": [error]}

def _run_iqs_role_agent_node(state: BrainAgentState, role_key: str) -> BrainAgentState:
    config = IQS_ROLE_CONFIGS[role_key]
    role_tasks = _tasks_for_agent(state, role_key)
    work_items = role_tasks
    if not work_items:
        return {}

    node_started = time.perf_counter()
    _progress("iqs-lane", "证据 Lane 开始", lane=role_key, label=config["label"], tasks=len(work_items))
    outputs: BrainAgentState = {}
    errors: List[str] = []
    query_results: List[Dict[str, Any]] = []
    answer_blocks: List[str] = []
    search_results: List[Dict[str, Any]] = []
    page_results: List[Dict[str, Any]] = []
    key_sources: List[Dict[str, Any]] = []
    confidences: List[float] = []
    conclusion_blocks: List[str] = []
    evidence_blocks: List[str] = []
    inference_blocks: List[str] = []
    evidence_gaps: List[Any] = []
    raw_data_points: List[Dict[str, Any]] = []
    task_results: List[Dict[str, Any]] = []

    indexed_tasks = [
        (index, task)
        for index, task in enumerate(work_items, start=1)
        if str(task.get("query") or "").strip()
    ]
    if _deadline_exceeded(state, min_remaining=1.0):
        timeout_payload = _deadline_timeout_payload(state, stage=f"{role_key}_lane")
        return {
            config["state"]: {
                "answer_text": "",
                "query_results": [],
                "raw_output": {
                    "query": str(state.get("query") or ""),
                    "role_key": role_key,
                    "dynamic_tasks": work_items,
                    "task_results": [],
                    "raw_data_points": [],
                    "lane_coverage": {
                        "scheduled": len(work_items),
                        "completed_task_count": 0,
                        "cancelled_task_count": len(indexed_tasks),
                        "execution_status": "deadline_exceeded",
                        "status": "deadline_exceeded",
                    },
                    "live_timeout": timeout_payload,
                },
                "metadata": {
                    "role_key": role_key,
                    "role_label": config["label"],
                    "status": "deadline_exceeded",
                    "live_timeout": timeout_payload,
                },
                "raw_data_points": [],
            },
            "errors": [f"{config['label']} skipped because the report deadline was reached"],
        }
    lane_workers = max(1, min(_env_int("BRAIN_IQS_LANE_PARALLEL_WORKERS", 4), len(indexed_tasks) or 1))
    task_timeout = max(0.0, _env_float("BRAIN_IQS_LANE_TASK_TIMEOUT_SECONDS", 180.0))
    remaining_for_lane = _deadline_remaining_seconds(state)
    if remaining_for_lane != float("inf"):
        task_timeout = max(1.0, min(task_timeout or remaining_for_lane, remaining_for_lane))
    task_payloads: List[Dict[str, Any]] = []
    early_stop_state: Dict[str, Any] = {"early_stopped": False, "early_stop_reason": ""}
    if lane_workers <= 1 or len(indexed_tasks) <= 1:
        for position, (index, task) in enumerate(indexed_tasks):
            if _deadline_exceeded(state, min_remaining=1.0):
                for cancelled_index, _ in indexed_tasks[position:]:
                    task_payloads.append(
                        {
                            "index": cancelled_index,
                            "status": "cancelled",
                            "cancel_reason": "deadline_exceeded",
                            "errors": [f"{config['label']} skipped because the report deadline was reached"],
                        }
                    )
                break
            if not task_timeout:
                task_payloads.append(
                    _run_iqs_lane_task(
                        state=state,
                        config=config,
                        role_key=role_key,
                        index=index,
                        total=len(work_items),
                        task=task,
                    )
                )
                early_stop_state = _lane_early_stop_decision(task_payloads, started_at=node_started)
                if early_stop_state.get("early_stopped"):
                    for cancelled_index, _ in indexed_tasks[position + 1 :]:
                        task_payloads.append(
                            {
                                "index": cancelled_index,
                                "status": "cancelled",
                                "cancel_reason": "cancelled_by_early_stop",
                            }
                        )
                    break
                continue
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(
                _run_iqs_lane_task,
                state=state,
                config=config,
                role_key=role_key,
                index=index,
                total=len(work_items),
                task=task,
            )
            try:
                task_payloads.append(future.result(timeout=task_timeout))
            except FutureTimeoutError:
                future.cancel()
                task_payloads.append(
                    {
                        "index": index,
                        "errors": [f"{config['label']} task timed out after {task_timeout:.0f}s"],
                    }
                )
            except Exception as exc:
                task_payloads.append(
                    {
                        "index": index,
                        "errors": [f"{config['label']} task failed: {exc}"],
                    }
                )
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
            early_stop_state = _lane_early_stop_decision(task_payloads, started_at=node_started)
            if early_stop_state.get("early_stopped"):
                for cancelled_index, _ in indexed_tasks[position + 1 :]:
                    task_payloads.append(
                        {
                            "index": cancelled_index,
                            "status": "cancelled",
                            "cancel_reason": "cancelled_by_early_stop",
                        }
                    )
                break
    else:
        _progress(
            "iqs-lane",
            "lane parallel tasks",
            lane=role_key,
            workers=lane_workers,
            tasks=len(indexed_tasks),
        )
        executor = ThreadPoolExecutor(max_workers=lane_workers)
        future_map = {
            executor.submit(
                _run_iqs_lane_task,
                state=state,
                config=config,
                role_key=role_key,
                index=index,
                total=len(work_items),
                task=task,
            ): index
            for index, task in indexed_tasks
        }
        completed: set[Any] = set()

        def _cancel_pending(reason: str) -> None:
            for pending_future, pending_index in future_map.items():
                if pending_future in completed:
                    continue
                pending_future.cancel()
                completed.add(pending_future)
                task_payloads.append(
                    {"index": pending_index, "status": "cancelled", "cancel_reason": reason}
                )

        try:
            wall_timeout = None
            if task_timeout:
                waves = max(1, math.ceil(len(indexed_tasks) / max(lane_workers, 1)))
                wall_timeout = task_timeout * waves + max(30.0, min(task_timeout, 60.0))
            # Hard global deadline: cap the lane wait by the remaining run budget so
            # the parallel branch never blocks past the deadline (per-lane wall_timeout
            # alone did not enforce the run deadline -> tasks ran on past it).
            deadline_remaining = _deadline_remaining_seconds(state)
            if deadline_remaining != float("inf") and deadline_remaining > 0:
                wall_timeout = deadline_remaining if wall_timeout is None else min(wall_timeout, deadline_remaining)
            for future in as_completed(future_map, timeout=wall_timeout):
                completed.add(future)
                try:
                    task_payloads.append(future.result())
                except Exception as exc:
                    task_payloads.append(
                        {
                            "index": future_map[future],
                            "errors": [f"{config['label']} parallel task failed: {exc}"],
                        }
                    )
                # Stop the moment the run deadline is hit: cancel everything still
                # pending instead of continuing to collect past the deadline.
                if _deadline_exceeded(state, min_remaining=1.0):
                    _cancel_pending("deadline_exceeded")
                    break
                early_stop_state = _lane_early_stop_decision(task_payloads, started_at=node_started)
                if early_stop_state.get("early_stopped"):
                    _cancel_pending("cancelled_by_early_stop")
                    break
        except FutureTimeoutError:
            # as_completed hit the (deadline-capped) wall timeout -> hard stop.
            reason = "deadline_exceeded" if _deadline_exceeded(state, min_remaining=1.0) else "lane_wall_timeout"
            _cancel_pending(reason)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    for payload in sorted(task_payloads, key=lambda item: int(item.get("index") or 0)):
        query_result = payload.get("query_result")
        if isinstance(query_result, dict):
            query_results.append(query_result)
        if payload.get("answer_block"):
            answer_blocks.append(str(payload.get("answer_block")))
        if payload.get("conclusion"):
            conclusion_blocks.append(str(payload.get("conclusion")).strip())
        if payload.get("evidence"):
            evidence_blocks.append(str(payload.get("evidence")).strip())
        if payload.get("inference"):
            inference_blocks.append(str(payload.get("inference")).strip())
        evidence_gaps.extend(_as_list(payload.get("evidence_gaps")))
        if "confidence" in payload:
            confidences.append(_clip_confidence(payload.get("confidence"), 0.0))
        search_results.extend([item for item in _as_list(payload.get("search_results")) if isinstance(item, dict)])
        page_results.extend([item for item in _as_list(payload.get("page_results")) if isinstance(item, dict)])
        key_sources.extend([item for item in _as_list(payload.get("key_sources")) if isinstance(item, dict)])
        raw_data_points.extend([item for item in _as_list(payload.get("raw_data_points")) if isinstance(item, dict)])
        task_result = payload.get("task_result")
        if isinstance(task_result, dict):
            task_results.append(task_result)
        errors.extend(str(item) for item in _as_list(payload.get("errors")) if str(item).strip())

    avg_conf = round(sum(confidences) / max(len([item for item in confidences if item > 0]), 1), 4) if confidences else 0.0
    succeeded = len(
        [
            item
            for item in task_results
            if str(item.get("status") or "") in {"success", "partial"}
            and (int(item.get("data_points") or 0) > 0 or int(item.get("sources") or 0) > 0)
        ]
    )
    lane_signal_counts = _lane_payload_signal_counts(task_payloads)
    early_stop_state = {**lane_signal_counts, **early_stop_state}
    coverage = {
        "scheduled": len(work_items),
        "succeeded": succeeded,
        "failed": int(lane_signal_counts.get("failed_task_count") or 0),
        "failed_task_count": int(lane_signal_counts.get("failed_task_count") or 0),
        "planned_task_count": len(work_items),
        "completed_task_count": int(lane_signal_counts.get("completed_task_count") or 0),
        "timed_out_task_count": int(lane_signal_counts.get("timed_out_task_count") or 0),
        "cancelled_task_count": int(lane_signal_counts.get("cancelled_task_count") or 0),
        "usable_source_count": int(lane_signal_counts.get("usable_source_count") or 0),
        "ab_source_count": int(lane_signal_counts.get("ab_source_count") or 0),
        "true_a_source_count": int(lane_signal_counts.get("true_a_source_count") or 0),
        "core_ab_source_count": int(lane_signal_counts.get("core_ab_source_count") or 0),
        "valid_metric_count": int(lane_signal_counts.get("valid_metric_count") or 0),
        "readpage_attempted": sum(int(_as_dict(_as_dict(payload).get("task_result")).get("readpage_attempted") or 0) for payload in task_payloads),
        "readpage_succeeded": sum(int(_as_dict(_as_dict(payload).get("task_result")).get("readpage_succeeded") or 0) for payload in task_payloads),
        "early_stopped": bool(early_stop_state.get("early_stopped")),
        "early_stop_reason": early_stop_state.get("early_stop_reason") or "",
        "raw_data_points": len(raw_data_points),
        "search_results": len(search_results),
        "page_results": len(page_results),
        "key_sources": len(key_sources),
    }
    completed_count = int(coverage.get("completed_task_count") or 0)
    timed_out_count = int(coverage.get("timed_out_task_count") or 0)
    failed_count = int(coverage.get("failed_task_count") or 0)
    cancelled_count = int(coverage.get("cancelled_task_count") or 0)
    if timed_out_count >= len(indexed_tasks) and completed_count <= 0:
        execution_status = "timed_out"
    elif completed_count + timed_out_count + failed_count + cancelled_count < len(indexed_tasks):
        execution_status = "partial"
    elif timed_out_count or failed_count or cancelled_count:
        execution_status = "partial"
    else:
        execution_status = "completed"
    coverage["execution_status"] = execution_status
    coverage["status"] = execution_status
    answer_text = "\n\n".join(answer_blocks).strip()
    outputs[config["state"]] = {
        "answer_text": answer_text,
        "query_results": query_results,
        "raw_output": {
            "query": str(state.get("query") or ""),
            "role_key": role_key,
            "role_label": config["label"],
            "dimension": config["dimension"],
            "focus": config["focus"],
            "dynamic_tasks": work_items,
            "task_results": task_results,
            "raw_data_points": raw_data_points,
            "lane_coverage": coverage,
            "search_options": dict(state.get("web_search_options") or {}),
            "search_results": search_results,
            "page_results": page_results,
            "errors": errors,
            "task_payload_summary": [
                {
                    "index": payload.get("index"),
                    "status": payload.get("status"),
                    "errors": _as_list(payload.get("errors")),
                    "cancel_reason": payload.get("cancel_reason"),
                }
                for payload in task_payloads
            ],
            "synthesis": {
                "type": "web_analysis_role_synthesis",
                "source": role_key,
                "structured_payload": {
                    "answer": {
                        "conclusion": conclusion_blocks[0] if conclusion_blocks else f"{config['dimension']}联网结果如下",
                        "evidence": "\n".join(item for item in evidence_blocks if item).strip() or answer_text,
                        "inference": "\n".join(item for item in inference_blocks if item).strip() or None,
                        "evidence_gap": evidence_gaps,
                    },
                    "confidence": avg_conf,
                    "key_sources": key_sources,
                    "limitations": {
                        "data_recency": "角色化 IQS 联网结果",
                        "coverage": config["focus"],
                        "conflicts": None,
                    },
                },
                "confidence": avg_conf,
                "key_sources": key_sources,
                "limitations": {"data_recency": "角色化 IQS 联网结果", "coverage": config["focus"], "conflicts": None},
            },
        },
        "metadata": {
            "role_key": role_key,
            "role_label": config["label"],
            "dimension": config["dimension"],
            "focus": config["focus"],
            "dynamic_tasks": work_items,
            "task_results": task_results,
            "lane_coverage": coverage,
            "status": execution_status,
        },
        "raw_data_points": raw_data_points,
    }

    if errors:
        outputs["errors"] = errors
    outputs["agent_trace"] = [
        {
            "agent": config["child"],
            "stage": "child_agent",
            "status": "error" if errors else "ok",
            "reason": "dynamic_iqs_lane",
        }
    ]
    _progress(
        "iqs-lane",
        "证据 Lane 结束",
        lane=role_key,
        succeeded=succeeded,
        raw_points=len(raw_data_points),
        search_results=len(search_results),
        elapsed=f"{time.perf_counter() - node_started:.1f}s",
    )
    return outputs


def run_iqs_lane_1_agent_node(state: BrainAgentState) -> BrainAgentState:
    return _run_iqs_role_agent_node(state, "iqs_lane_1")


def run_iqs_lane_2_agent_node(state: BrainAgentState) -> BrainAgentState:
    return _run_iqs_role_agent_node(state, "iqs_lane_2")


def run_iqs_lane_3_agent_node(state: BrainAgentState) -> BrainAgentState:
    return _run_iqs_role_agent_node(state, "iqs_lane_3")


def run_iqs_lane_4_agent_node(state: BrainAgentState) -> BrainAgentState:
    return _run_iqs_role_agent_node(state, "iqs_lane_4")


def run_iqs_lane_5_agent_node(state: BrainAgentState) -> BrainAgentState:
    return _run_iqs_role_agent_node(state, "iqs_lane_5")


def run_iqs_lane_6_agent_node(state: BrainAgentState) -> BrainAgentState:
    return _run_iqs_role_agent_node(state, "iqs_lane_6")


def select_child_agents(state: BrainAgentState) -> List[str]:
    if state.get("errors"):
        return ["merge_outputs"]
    route = str(state.get("route") or "local").strip().lower()
    analysis_agents = _as_list(_as_dict(state.get("query_analysis")).get("target_agents"))
    agents = [str(agent).strip().lower() for agent in analysis_agents if str(agent).strip()] or _route_agents(route)
    nodes: List[str] = []
    if "rag" in agents:
        nodes.append("industry_rag_agent")
    nodes.extend(IQS_ROLE_CONFIGS[key]["node"] for key in IQS_ROLE_ORDER if key in agents)
    return nodes or ["merge_outputs"]


def _child_answer(child_state: Optional[Dict[str, Any]]) -> str:
    if not child_state:
        return ""
    return str(child_state.get("answer_text") or "").strip()


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _topic_bundle_seed_from_state(state: Dict[str, Any]) -> Dict[str, Any]:
    return _as_dict(state.get("topic_bundle_seed"))


def _evidence_identity_key(item: Dict[str, Any]) -> str:
    payload = _as_dict(item)
    for key in ("url", "source_url", "document_ref", "document_id", "source_ref", "evidence_id", "ref"):
        value = str(payload.get(key) or "").strip().lower()
        if value:
            return f"{key}:{value}"
    text = str(payload.get("fact") or payload.get("clean_fact") or payload.get("summary") or payload.get("title") or "").strip()
    return f"fact:{_stable_short_hash(text, length=20)}" if text else ""


def _topic_bundle_seed_evidence_from_state(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    seed = _topic_bundle_seed_from_state(state)
    topic_key = str(seed.get("topic_key") or "").strip()
    path = str(seed.get("path") or "").strip()
    result: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in _as_list(seed.get("seed_evidence")):
        payload = dict(_as_dict(item))
        if not payload:
            continue
        fact_text = str(payload.get("evidence") or payload.get("fact") or payload.get("clean_fact") or payload.get("content") or payload.get("summary") or "").strip()
        if fact_text:
            payload.setdefault("evidence", fact_text)
            payload.setdefault("fact", fact_text)
            payload.setdefault("content", fact_text)
        source = _as_dict(payload.get("source"))
        if source:
            if source.get("url") or source.get("source_url"):
                payload.setdefault("source_url", source.get("url") or source.get("source_url"))
            if source.get("title"):
                payload.setdefault("source_title", source.get("title"))
            if source.get("publisher") or source.get("source"):
                payload.setdefault("publisher", source.get("publisher") or source.get("source"))
        payload.setdefault("evidence_origin", "topic_bundle_cache")
        payload.setdefault("origin", "topic_bundle_cache")
        if topic_key:
            payload.setdefault("topic_bundle_key", topic_key)
        if path:
            payload.setdefault("topic_bundle_path", path)
        key = _evidence_identity_key(payload)
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        result.append(payload)
    return result


def _source_from_seed_evidence(item: Dict[str, Any]) -> Dict[str, Any]:
    payload = _as_dict(item)
    source = _as_dict(payload.get("source"))
    return {
        "title": str(source.get("title") or payload.get("source_title") or payload.get("title") or "").strip(),
        "url": str(source.get("url") or source.get("source_url") or payload.get("source_url") or payload.get("url") or "").strip(),
        "publisher": str(source.get("publisher") or source.get("source") or payload.get("publisher") or payload.get("source") or "").strip(),
        "source": str(source.get("source") or source.get("publisher") or payload.get("source") or payload.get("publisher") or "").strip(),
        "date": str(source.get("date") or payload.get("date") or payload.get("period") or "").strip(),
        "source_level": str(payload.get("source_level") or source.get("source_level") or "").strip().upper(),
        "source_verification_status": str(payload.get("source_verification_status") or source.get("source_verification_status") or "").strip(),
        "source_verified": bool(payload.get("source_verified") or source.get("source_verified")),
        "evidence_origin": "topic_bundle_cache",
    }


def _topic_bundle_seed_pool_item(state: Dict[str, Any]) -> Dict[str, Any]:
    seed = _topic_bundle_seed_from_state(state)
    seed_items = _topic_bundle_seed_evidence_from_state(state)
    if not seed_items:
        return {}
    sources: List[Dict[str, Any]] = []
    seen_sources: set[str] = set()
    for item in seed_items:
        source = _source_from_seed_evidence(item)
        key = str(source.get("url") or source.get("title") or "").strip().lower()
        if not key or key in seen_sources:
            continue
        seen_sources.add(key)
        sources.append(source)
        if len(sources) >= 40:
            break
    return {
        "round": 0,
        "agent": "topic_bundle_cache",
        "child_agent": "topic_bundle_cache",
        "query": str(state.get("query") or ""),
        "targets_gap": "topic_bundle_seed",
        "status": "success",
        "confidence": 0.72,
        "answer": "Topic bundle seed evidence loaded for live merge.",
        "key_sources": sources,
        "raw_data_points": seed_items,
        "limitations": {"cache_seed": True, "requires_live_regrading": True},
        "evidence_origin": "topic_bundle_cache",
        "topic_bundle_key": seed.get("topic_key"),
        "topic_bundle_path": seed.get("path"),
    }


def _merge_topic_seed_with_live_evidence(state: Dict[str, Any], evidence_pool: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seed_pool_item = _topic_bundle_seed_pool_item(state)
    if not seed_pool_item:
        return [dict(item) for item in list(evidence_pool or []) if isinstance(item, dict)]
    merged: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in [seed_pool_item] + [dict(item) for item in list(evidence_pool or []) if isinstance(item, dict)]:
        key = _evidence_identity_key(item)
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        merged.append(dict(item))
    return merged


def _topic_bundle_seed_summary(state: Dict[str, Any], merged_evidence_pool: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    seed = _topic_bundle_seed_from_state(state)
    seed_count = len(_topic_bundle_seed_evidence_from_state(state))
    if not seed and seed_count <= 0:
        return {}
    return {
        "enabled": bool(seed),
        "topic_key": seed.get("topic_key"),
        "path": seed.get("path"),
        "seed_evidence_count": seed_count,
        "merged_evidence_pool_count": len(list(merged_evidence_pool or [])),
        "preflight_status": _as_dict(seed.get("preflight")).get("status"),
    }


def _store_topic_bundle_from_brain(
    *,
    state: Dict[str, Any],
    evidence_package: Dict[str, Any],
    structured_analysis: Optional[Dict[str, Any]] = None,
    writer_report: Optional[Dict[str, Any]] = None,
    stage: str,
) -> Dict[str, Any]:
    try:
        from rag_pipeline.cache.topic_bundle_cache import store_topic_bundle

        report_blueprint = (
            _as_dict(_as_dict(writer_report).get("report_blueprint"))
            or _as_dict(state.get("report_blueprint"))
            or _as_dict(_as_dict(state.get("query_analysis")).get("report_blueprint"))
            or _as_dict(_as_dict(state.get("query_analysis")).get("report_plan"))
        )
        chapter_packages = (
            _as_list(_as_dict(writer_report).get("chapter_evidence_packages"))
            or _as_list(evidence_package.get("chapter_evidence_packages"))
            or _as_list(state.get("chapter_evidence_packages"))
        )
        micro_layouts = _as_list(_as_dict(writer_report).get("micro_layouts")) or _as_list(state.get("micro_layouts"))
        table_packages = _as_list(_as_dict(writer_report).get("table_packages")) or _as_list(state.get("table_packages"))
        return store_topic_bundle(
            query=str(state.get("query") or ""),
            research_plan=_research_plan_from_state(state),
            report_blueprint=report_blueprint,
            evidence_package=evidence_package,
            structured_analysis=_as_dict(structured_analysis),
            source_registry=_as_list(_as_dict(writer_report).get("source_registry")) or _as_list(evidence_package.get("source_registry")) or _as_list(evidence_package.get("sources")),
            chapter_evidence_packages=chapter_packages,
            micro_layouts=micro_layouts,
            table_packages=table_packages,
            writer_report=_as_dict(writer_report),
            stage=stage,
            stored_from=stage,
        )
    except Exception as exc:  # pragma: no cover - cache must never block report runs.
        return {"enabled": True, "stored": False, "reason": "store_failed", "error": str(exc), "stored_from": stage}


def _write_stage_snapshot_from_brain(
    *,
    state: Dict[str, Any],
    stage_name: str,
    payload: Any,
    summary: Optional[Dict[str, Any]] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    run_id = str(state.get("stage_snapshot_run_id") or os.getenv("REPORT_STAGE_SNAPSHOT_RUN_ID") or "").strip()
    if not run_id:
        return {"enabled": False, "stored": False, "reason": "missing_run_id", "stage_name": stage_name}
    try:
        from rag_pipeline.cache.stage_snapshot_cache import write_stage_snapshot

        return write_stage_snapshot(
            stage_name=stage_name,
            run_id=run_id,
            payload=payload,
            summary=summary,
            diagnostics=diagnostics,
        )
    except Exception as exc:  # pragma: no cover - snapshots are diagnostic only.
        return {"enabled": True, "stored": False, "reason": "store_failed", "stage_name": stage_name, "error": str(exc)}


def _emit_pre_writer_snapshots(
    state: Dict[str, Any],
    evidence_package: Dict[str, Any],
    structured_analysis: Dict[str, Any],
) -> None:
    """Persist the pre-writer evidence + analysis snapshots and attach the result to evidence_package.metadata.

    Centralised here so that `merge_outputs_node` does not have to repeat the
    same six-line snapshot recipe in each of its branches. Mutates
    `evidence_package` in place.
    """

    chapter_evidence_packages = _as_list(evidence_package.get("chapter_evidence_packages"))
    if not chapter_evidence_packages:
        try:
            from rag_pipeline.agents.chapter_evidence_builder import build_chapter_evidence_packages_from_evidence_package

            report_blueprint = (
                _as_dict(state.get("report_blueprint"))
                or _as_dict(_as_dict(state.get("query_analysis")).get("report_blueprint"))
                or _as_dict(evidence_package.get("report_blueprint"))
                or _as_dict(_as_dict(evidence_package.get("metadata")).get("report_blueprint"))
                or _as_dict(evidence_package.get("report_plan"))
                or _as_dict(_as_dict(evidence_package.get("metadata")).get("report_plan"))
            )
            source_registry = _as_list(evidence_package.get("source_registry")) or _as_list(evidence_package.get("sources"))
            if report_blueprint:
                chapter_evidence_packages = build_chapter_evidence_packages_from_evidence_package(
                    report_blueprint=report_blueprint,
                    evidence_package=evidence_package,
                    existing_chapter_evidence_packages=[],
                    source_registry=source_registry,
                )
                if chapter_evidence_packages:
                    evidence_package["chapter_evidence_packages"] = chapter_evidence_packages
        except Exception as exc:  # pragma: no cover - snapshot rebuild must never block live runs.
            evidence_package.setdefault("metadata", {})["chapter_evidence_snapshot_error"] = str(exc)

    evidence_snapshot_store = _write_stage_snapshot_from_brain(
        state=state,
        stage_name="evidence_package",
        payload=evidence_package,
        summary={"stored_from": "brain_full_payload", "phase": "pre_writer"},
    )
    chapter_snapshot_store = _write_stage_snapshot_from_brain(
        state=state,
        stage_name="chapter_evidence_packages",
        payload=chapter_evidence_packages,
        summary={
            "stored_from": "brain_full_payload",
            "phase": "pre_writer",
            "chapter_evidence_package_count": len(chapter_evidence_packages),
        },
        diagnostics={
            "chapter_binding_status": "available" if chapter_evidence_packages else "missing_or_unrebuildable",
        },
    )
    analysis_snapshot_store = _write_stage_snapshot_from_brain(
        state=state,
        stage_name="structured_analysis",
        payload=structured_analysis,
        summary={
            "stored_from": "brain_full_payload",
            "phase": "pre_writer",
            "chapter_evidence_package_count": len(chapter_evidence_packages),
        },
        diagnostics={
            "chapter_binding_status": "available" if chapter_evidence_packages else "missing_or_unrebuildable",
        },
    )
    if evidence_snapshot_store.get("stored") or chapter_snapshot_store.get("stored") or analysis_snapshot_store.get("stored"):
        evidence_package.setdefault("metadata", {})["stage_snapshot_store"] = {
            "evidence_package": evidence_snapshot_store,
            "chapter_evidence_packages": chapter_snapshot_store,
            "structured_analysis": analysis_snapshot_store,
        }


def _emit_post_writer_snapshot(state: Dict[str, Any], writer_report: Dict[str, Any]) -> None:
    """Persist the writer-output snapshot and attach the result to writer_report.

    Mutates `writer_report` in place.
    """

    writer_snapshot_store = _write_stage_snapshot_from_brain(
        state=state,
        stage_name="writer_report",
        payload=writer_report,
        summary={"stored_from": "writer_full_payload", "phase": "post_writer"},
    )
    if writer_snapshot_store.get("stored"):
        writer_report["stage_snapshot_store"] = writer_snapshot_store


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _deadline_ts_from_state(state: Dict[str, Any]) -> float:
    value = _as_dict(state.get("timeout_context")).get("deadline_ts") or state.get("deadline_ts")
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _deadline_remaining_seconds(state: Dict[str, Any]) -> float:
    deadline_ts = _deadline_ts_from_state(state)
    if deadline_ts <= 0:
        return float("inf")
    return deadline_ts - time.perf_counter()


def _deadline_exceeded(state: Dict[str, Any], *, min_remaining: float = 0.0) -> bool:
    remaining = _deadline_remaining_seconds(state)
    return remaining != float("inf") and remaining <= float(min_remaining or 0.0)


def _analysis_deadline_ts_for_state(state: Dict[str, Any]) -> Optional[float]:
    """Deadline handed to run_analysis_agent: the report deadline minus a writer
    reserve, so the renderer/body-rewrite stages still fit before the wall clock
    runs out instead of discovering the timeout only after analysis returns."""
    deadline_ts = _deadline_ts_from_state(state)
    if deadline_ts <= 0:
        return None
    reserve = 0.0
    try:
        reserve = float(os.getenv("BRAIN_ANALYSIS_WRITER_RESERVE_SECONDS", "240") or 240)
    except (TypeError, ValueError):
        reserve = 240.0
    return deadline_ts - max(0.0, reserve)


def _deadline_timeout_payload(state: Dict[str, Any], *, stage: str) -> Dict[str, Any]:
    context = _as_dict(state.get("timeout_context"))
    return {
        "timeout_triggered": True,
        "timeout_stage": stage,
        "deadline_ts": _deadline_ts_from_state(state),
        "remaining_seconds": max(0.0, _deadline_remaining_seconds(state)),
        "live_deadline_seconds": int(context.get("max_seconds") or state.get("max_wall_seconds") or 0),
        "fail_open_on_timeout": bool(context.get("fail_open_on_timeout", state.get("fail_open_on_timeout", True))),
    }


def _stable_short_hash(*values: Any, length: int = 16) -> str:
    raw = "|".join(str(value or "") for value in values)
    return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:length]


def _clip_confidence(value: Any, default: float = 0.0) -> float:
    return round(max(0.0, min(1.0, _safe_float(value, default))), 4)


def _compact_text(value: Any, max_chars: int = 1200) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    if len(text) <= max_chars:
        return text
    return text[: max(20, max_chars - 3)].rstrip() + "..."


def _normalize_identity_text(value: Any, *, max_chars: int = 160) -> str:
    text = _compact_text(value, max_chars=max_chars).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _followup_gap_id(item: Dict[str, Any]) -> str:
    payload = _as_dict(item)
    for key in (
        "gap_id",
        "mandatory_proof_id",
        "proof_id",
        "hypothesis_id",
        "evidence_goal_id",
        "task_id",
    ):
        value = _normalize_identity_text(payload.get(key), max_chars=120)
        if value:
            return value
    target = _normalize_identity_text(
        payload.get("targets_gap")
        or payload.get("dimension_name")
        or payload.get("dimension")
        or payload.get("evidence_goal")
        or payload.get("type")
        or payload.get("reason"),
        max_chars=120,
    )
    query = _normalize_identity_text(payload.get("query") or payload.get("suggested_query"), max_chars=120)
    if target and query:
        return f"{target}|{query}"
    return target or query


def _followup_task_base_key(task: Dict[str, Any]) -> str:
    search_task = _as_dict(task.get("search_task"))
    gap_id = _followup_gap_id(search_task) or _followup_gap_id(task)
    query = _normalize_identity_text(task.get("query"), max_chars=220)
    target = _normalize_identity_text(task.get("targets_gap") or search_task.get("targets_gap") or search_task.get("dimension_name"), max_chars=120)
    return f"{gap_id}|{target}|{query}"


def _dedupe_followup_tasks(tasks: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = [dict(task) for task in list(tasks or []) if isinstance(task, dict)]
    specific_iqs_bases = {
        _followup_task_base_key(task)
        for task in normalized
        if str(task.get("agent") or "").strip().lower() in IQS_ROLE_CONFIGS
    }
    result: List[Dict[str, Any]] = []
    seen_exact = set()
    for task in normalized:
        agent = str(task.get("agent") or "").strip().lower()
        base_key = _followup_task_base_key(task)
        if agent == "iqs" and base_key in specific_iqs_bases:
            continue
        exact_key = (base_key, agent)
        if exact_key in seen_exact:
            continue
        seen_exact.add(exact_key)
        result.append(task)
    return result


def _json_text(value: Any, max_chars: int = 900) -> str:
    if value in (None, "", [], {}):
        return ""
    if isinstance(value, str):
        return _compact_text(value, max_chars=max_chars)
    return _compact_text(json.dumps(value, ensure_ascii=False, default=json_safe_default), max_chars=max_chars)


def _payload_mode(name: str, default: str = "summary") -> str:
    value = str(os.getenv(name, default) or default).strip().lower()
    return value if value in {"summary", "full"} else default


def _brain_full_payloads() -> bool:
    if _env_flag("BRAIN_RETURN_FULL_ARTIFACTS", False):
        return True
    return _payload_mode("BRAIN_STATE_PAYLOAD_MODE", "summary") == "full"


def _compact_mapping_for_state(value: Dict[str, Any], *, max_items: int = 20, max_chars: int = 220) -> Dict[str, Any]:
    compacted: Dict[str, Any] = {}
    for key, item in list(value.items())[:max_items]:
        if item in (None, "", [], {}):
            continue
        if isinstance(item, dict):
            compacted[str(key)] = {
                str(sub_key): _compact_text(sub_value, max_chars=min(max_chars, 160))
                for sub_key, sub_value in list(item.items())[:8]
                if sub_value not in (None, "", [], {})
            }
        elif isinstance(item, list):
            compacted[str(key)] = [_compact_text(entry, max_chars=120) for entry in item[:6] if str(entry or "").strip()]
        else:
            compacted[str(key)] = _compact_text(item, max_chars=max_chars)
    return compacted


def _compact_health_summary_for_state(value: Dict[str, Any], *, max_items: int = 80, max_chars: int = 220) -> Dict[str, Any]:
    compacted: Dict[str, Any] = {}
    for key, item in list(_as_dict(value).items())[:max_items]:
        if item in (None, "", [], {}):
            continue
        if isinstance(item, (bool, int, float)):
            compacted[str(key)] = item
        elif isinstance(item, dict):
            nested: Dict[str, Any] = {}
            for sub_key, sub_value in list(item.items())[:40]:
                if sub_value in (None, "", [], {}):
                    continue
                if isinstance(sub_value, (bool, int, float)):
                    nested[str(sub_key)] = sub_value
                else:
                    nested[str(sub_key)] = _compact_text(sub_value, max_chars=min(max_chars, 160))
            compacted[str(key)] = nested
        elif isinstance(item, list):
            compacted[str(key)] = [
                entry if isinstance(entry, (bool, int, float)) else _compact_text(entry, max_chars=120)
                for entry in item[:20]
                if entry not in (None, "", [], {})
            ]
        else:
            compacted[str(key)] = _compact_text(item, max_chars=max_chars)
    return compacted


def _summarize_sequence(values: Sequence[Any], *, sample: int = 5, max_chars: int = 180) -> Dict[str, Any]:
    items = list(values or [])
    return {
        "count": len(items),
        "sample": [
            _compact_mapping_for_state(item, max_items=10, max_chars=max_chars) if isinstance(item, dict) else _compact_text(item, max_chars=max_chars)
            for item in items[:sample]
        ],
    }


def _compact_evidence_package_for_state(evidence_package: Dict[str, Any]) -> Dict[str, Any]:
    summary = _as_dict(evidence_package.get("summary"))
    metadata = _as_dict(evidence_package.get("metadata"))
    normalized = _as_list(evidence_package.get("normalized_evidence"))
    raw_data_points = _as_list(evidence_package.get("raw_data_points"))
    evidence_analysis_summary = _as_dict(evidence_package.get("evidence_analysis_summary"))
    evidence_analysis_by_chapter = _as_dict(evidence_package.get("evidence_analysis_by_chapter"))
    chapter_evidence_diagnostics = _as_dict(evidence_package.get("chapter_evidence_diagnostics")) or evidence_analysis_by_chapter
    gap_ledger = _as_list(evidence_package.get("evidence_gap_ledger"))
    health_summary = (
        _as_dict(evidence_package.get("evidence_health_summary"))
        or _as_dict(summary.get("evidence_health_summary"))
        or _as_dict(metadata.get("evidence_health_summary"))
    )
    source_registry = _as_list(evidence_package.get("source_registry")) or _as_list(evidence_package.get("sources"))
    source_registry_summary = (
        _as_dict(evidence_package.get("source_registry_summary"))
        or _as_dict(summary.get("source_registry_summary"))
        or _as_dict(metadata.get("source_registry_summary"))
    )
    compact_gap_fields = [
        "gap_id",
        "chapter_id",
        "claim_id",
        "gap_type",
        "type",
        "severity",
        "required_proof_role",
        "proof_role",
        "required_fields",
        "query_terms",
        "lane_targets",
        "why_current_evidence_insufficient",
    ]
    compact_ledger = [
        {
            key: item.get(key)
            for key in compact_gap_fields
            if isinstance(item, dict) and key in item
        }
        for item in gap_ledger[:80]
        if isinstance(item, dict)
    ]
    compact_chapters = {
        str(chapter_id): _compact_mapping_for_state(_as_dict(payload), max_items=18, max_chars=160)
        for chapter_id, payload in list(chapter_evidence_diagnostics.items())[:40]
        if isinstance(payload, dict)
    }
    compact_summary = _compact_mapping_for_state(summary, max_items=20, max_chars=180)
    for key in (
        "readpage_coverage",
        "publishable_evidence_gate",
        "delivery_gate",
        "evidence_health_summary",
        "source_registry_summary",
    ):
        if key in summary:
            compact_summary[key] = _compact_mapping_for_state(_as_dict(summary.get(key)), max_items=40, max_chars=180)
    if health_summary:
        compact_summary["evidence_health_summary"] = _compact_health_summary_for_state(health_summary, max_items=80, max_chars=180)
    if source_registry_summary:
        compact_summary["source_registry_summary"] = _compact_mapping_for_state(source_registry_summary, max_items=30, max_chars=180)
    compact_metadata = _compact_mapping_for_state(metadata, max_items=20, max_chars=160)
    if health_summary:
        compact_metadata["evidence_health_summary"] = _compact_health_summary_for_state(health_summary, max_items=80, max_chars=180)
    if source_registry_summary:
        compact_metadata["source_registry_summary"] = _compact_mapping_for_state(source_registry_summary, max_items=30, max_chars=180)
    return {
        "payload_mode": "summary",
        "summary": compact_summary,
        "metadata": compact_metadata,
        "normalized_evidence": _summarize_sequence(normalized, sample=8, max_chars=160),
        "raw_data_points": _summarize_sequence(raw_data_points, sample=8, max_chars=160),
        "evidence_analysis_summary": _compact_mapping_for_state(evidence_analysis_summary, max_items=30, max_chars=180),
        "evidence_health_summary": _compact_health_summary_for_state(health_summary, max_items=80, max_chars=180),
        "source_registry_summary": _compact_mapping_for_state(source_registry_summary, max_items=20, max_chars=180),
        "evidence_gap_ledger": compact_ledger,
        "evidence_gap_ledger_count": len(gap_ledger),
        "evidence_analysis_by_chapter": compact_chapters,
        "chapter_evidence_diagnostics": compact_chapters,
        "analysis_ready_ab_count": summary.get("analysis_ready_ab_count"),
        "metric_ready_count": summary.get("metric_ready_count"),
        "blocking_evidence_gap_count": summary.get("blocking_evidence_gap_count"),
        "gap_type_distribution": summary.get("evidence_gap_type_distribution"),
        "source_registry_count": health_summary.get("source_registry_count") or source_registry_summary.get("source_registry_count") or len(source_registry),
        "traceable_ab_source_count": health_summary.get("traceable_ab_source_count"),
        "raw_data_point_count": health_summary.get("raw_data_point_count") or len(raw_data_points),
        "normalized_evidence_count": health_summary.get("normalized_evidence_count") or len(normalized),
        "source_count": health_summary.get("source_registry_count") or source_registry_summary.get("source_registry_count") or len(source_registry),
    }


def _compact_structured_analysis_for_state(structured_analysis: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "payload_mode": "summary",
        "analysis_type": structured_analysis.get("analysis_type"),
        "report_plan": _compact_mapping_for_state(_as_dict(structured_analysis.get("report_plan")), max_items=30, max_chars=180),
        "research_plan": _compact_mapping_for_state(_as_dict(structured_analysis.get("research_plan")), max_items=30, max_chars=180),
        "structured_analysis": _compact_mapping_for_state(_as_dict(structured_analysis.get("structured_analysis")), max_items=30, max_chars=180),
        "report_insight_package": _compact_mapping_for_state(_as_dict(structured_analysis.get("report_insight_package")), max_items=30, max_chars=180),
        "chapter_evidence_diagnostics": _compact_mapping_for_state(_as_dict(structured_analysis.get("chapter_evidence_diagnostics")), max_items=40, max_chars=160),
        "evidence_analysis_summary": _compact_mapping_for_state(_as_dict(structured_analysis.get("evidence_analysis_summary")), max_items=30, max_chars=180),
        "evidence_gap_ledger": _summarize_sequence(_as_list(structured_analysis.get("evidence_gap_ledger")), sample=12, max_chars=180),
        "analysis_depth_quality": _compact_mapping_for_state(_as_dict(structured_analysis.get("analysis_depth_quality")), max_items=20, max_chars=180),
        "analysis_stage_diagnostics": _compact_mapping_for_state(_as_dict(structured_analysis.get("analysis_stage_diagnostics")), max_items=20, max_chars=180),
        "llm_analysis_synthesis": _compact_mapping_for_state(_as_dict(structured_analysis.get("llm_analysis_synthesis")), max_items=16, max_chars=180),
    }


def _compact_writer_report_for_state(writer_report: Dict[str, Any]) -> Dict[str, Any]:
    keep_keys = [
        "report_markdown",
        "report_type",
        "report_status",
        "delivery_tier",
        "delivery_gate",
        "draft_mode",
        "message",
        "estimated_chars",
        "estimated_body_chars",
        "target_body_chars",
        "source_registry",
        "footnotes",
        "validation",
        "qa_result",
        "package_quality_report",
        "layout_plan",
        "report_blueprint",
        "search_tasks",
        "search_task_schedule",
        "lane_coverage",
        "pipeline_payload_mode",
        "pipeline_artifact_summary",
        "config_warnings",
        "delivery_blockers",
        "required_followups",
        "qa_pending_repair",
        "qa_pending_repair_reasons",
        "reformatter_preflight",
        "post_qa_repair",
        "topic_bundle_cache_store",
        "render_artifacts",
        "metadata",
    ]
    compacted = {key: writer_report.get(key) for key in keep_keys if key in writer_report}
    compacted["payload_mode"] = "summary"
    return compacted


def _compact_children_for_state(children: Dict[str, Any]) -> Dict[str, Any]:
    compacted: Dict[str, Any] = {}
    for name, child in children.items():
        child_dict = _as_dict(child)
        compacted[name] = {
            "status": child_dict.get("status") or _as_dict(child_dict.get("metadata")).get("status"),
            "confidence": child_dict.get("confidence"),
            "answer": _compact_text(child_dict.get("answer") or child_dict.get("answer_text"), max_chars=500),
            "key_sources": _summarize_sequence(_as_list(child_dict.get("key_sources")), sample=5, max_chars=160),
            "evidence_count": len(_as_list(child_dict.get("evidence"))),
            "raw_data_point_count": len(_as_list(child_dict.get("raw_data_points"))),
        }
    return compacted


def compact_children_for_llm_merge(children: Dict[str, Any]) -> Dict[str, Any]:
    """Small, deterministic child summary for supervisor merge prompts."""

    compacted: Dict[str, Any] = {}
    for name, child in _as_dict(children).items():
        child_dict = _as_dict(child)
        raw_output = _as_dict(child_dict.get("raw_output"))
        metadata = _as_dict(child_dict.get("metadata"))
        lane_coverage = _as_dict(raw_output.get("lane_coverage")) or _as_dict(metadata.get("lane_coverage"))
        evidence_package = _as_dict(raw_output.get("evidence_package")) or _as_dict(child_dict.get("evidence_package"))
        health = (
            _as_dict(evidence_package.get("evidence_health_summary"))
            or _as_dict(_as_dict(evidence_package.get("summary")).get("evidence_health_summary"))
            or _as_dict(_as_dict(evidence_package.get("metadata")).get("evidence_health_summary"))
        )
        source_candidates = (
            _as_list(child_dict.get("key_sources"))
            + _as_list(raw_output.get("key_sources"))
            + _as_list(raw_output.get("page_results"))
            + _as_list(raw_output.get("search_results"))
        )
        sources: List[Dict[str, Any]] = []
        seen_urls: set[str] = set()
        for source in source_candidates:
            payload = _as_dict(source)
            if not payload:
                continue
            url = str(payload.get("url") or payload.get("source_url") or "").strip()
            title = str(payload.get("title") or payload.get("source_title") or "").strip()
            key = url or title
            if not key or key in seen_urls:
                continue
            seen_urls.add(key)
            sources.append(
                {
                    "title": _compact_text(title, max_chars=140),
                    "url": _compact_text(url, max_chars=180),
                    "source_level": payload.get("source_level") or payload.get("credibility"),
                    "date": payload.get("date") or payload.get("published_at"),
                }
            )
            if len(sources) >= 8:
                break
        compacted[str(name)] = {
            "status": child_dict.get("status") or metadata.get("status") or raw_output.get("status"),
            "answer": _compact_text(child_dict.get("answer") or child_dict.get("answer_text"), max_chars=800),
            "query_result_count": len(_as_list(child_dict.get("query_results")) or _as_list(raw_output.get("query_results"))),
            "evidence_count": len(_as_list(child_dict.get("evidence")) or _as_list(raw_output.get("evidence"))),
            "raw_data_point_count": len(_as_list(child_dict.get("raw_data_points")) or _as_list(raw_output.get("raw_data_points"))),
            "top_sources": sources,
            "lane_coverage": _compact_mapping_for_state(lane_coverage, max_items=20, max_chars=120),
            "evidence_health_summary": _compact_health_summary_for_state(health, max_items=40, max_chars=120),
            "errors": [_compact_text(item, max_chars=220) for item in _as_list(child_dict.get("errors"))[:6]],
        }
    return compacted


def _state_payload(value: Any, kind: str) -> Any:
    if _brain_full_payloads():
        return value
    if kind == "evidence_package":
        return _compact_evidence_package_for_state(_as_dict(value))
    if kind == "structured_analysis":
        return _compact_structured_analysis_for_state(_as_dict(value))
    if kind == "writer_report":
        return _compact_writer_report_for_state(_as_dict(value))
    if kind == "children":
        return _compact_children_for_state(_as_dict(value))
    if isinstance(value, dict):
        return _compact_mapping_for_state(value, max_items=30, max_chars=180)
    if isinstance(value, list):
        return _summarize_sequence(value, sample=8, max_chars=180)
    return value


_HANDOFF_ITEM_DROP_KEYS = {
    "raw",
    "raw_html",
    "html",
    "embedding",
    "vector",
    "tokens",
}


def _handoff_text_limit(key: str) -> int:
    if key in {"fact", "clean_fact", "content", "text", "summary"}:
        return 1200
    if key in {"analysis_input", "evidence_card", "source"}:
        return 1800
    return 600


def _slim_handoff_value(value: Any, *, key: str = "", depth: int = 0) -> Any:
    if depth > 6:
        return _compact_text(value, max_chars=240)
    if isinstance(value, dict):
        compacted: Dict[str, Any] = {}
        for item_key, item_value in value.items():
            text_key = str(item_key)
            if text_key in _HANDOFF_ITEM_DROP_KEYS:
                continue
            compacted[text_key] = _slim_handoff_value(item_value, key=text_key, depth=depth + 1)
        return compacted
    if isinstance(value, list):
        return [_slim_handoff_value(item, key=key, depth=depth + 1) for item in value[:80]]
    if isinstance(value, str):
        limit = _handoff_text_limit(key)
        return value if len(value) <= limit else _compact_text(value, max_chars=limit)
    return value


def _handoff_source_id(item: Dict[str, Any]) -> str:
    source = _as_dict(item.get("source"))
    for key in ("source_ref", "source_id", "source", "citation_ref", "ref"):
        value = item.get(key)
        if value:
            return str(value).strip().strip("[]")
    for key in ("id", "source_id", "ref"):
        value = source.get(key)
        if value:
            return str(value).strip().strip("[]")
    return ""


def _handoff_item_key(item: Dict[str, Any]) -> str:
    evidence_id = str(item.get("evidence_id") or item.get("ref") or item.get("id") or "").strip()
    if evidence_id:
        return f"id:{evidence_id}"
    source_id = _handoff_source_id(item)
    text = str(item.get("fact") or item.get("clean_fact") or item.get("content") or item.get("text") or "").strip()
    return f"{source_id}|{re.sub(r'\\s+', '', text.lower())[:220]}"


def _select_handoff_items(
    *groups: Sequence[Dict[str, Any]],
    limit: int,
    priority_source_ids: Optional[set[str]] = None,
    preserve_first_group: bool = False,
) -> List[Dict[str, Any]]:
    priority_source_ids = priority_source_ids or set()
    selected: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def add(item: Dict[str, Any], *, force: bool = False) -> None:
        if not isinstance(item, dict):
            return
        key = _handoff_item_key(item)
        if not key or key in seen:
            return
        if len(selected) >= limit and not force:
            return
        seen.add(key)
        selected.append(_slim_handoff_value(item))

    start_index = 0
    if preserve_first_group and groups:
        for item in list(groups[0] or []):
            add(item, force=True)
        start_index = 1
    for group in groups[start_index:]:
        cited = []
        other = []
        for item in list(group or []):
            if _handoff_source_id(item) in priority_source_ids:
                cited.append(item)
            else:
                other.append(item)
        for item in cited:
            add(item, force=len(selected) < limit)
        for item in other:
            add(item)
            if len(selected) >= limit:
                break
    preserved_count = len(list(groups[0] or [])) if preserve_first_group and groups else 0
    return selected[: max(limit, preserved_count)]


def _writer_cited_source_ids(writer_report: Dict[str, Any]) -> set[str]:
    markdown = str(_as_dict(writer_report).get("report_markdown") or "")
    return {item.strip("[]") for item in re.findall(r"\[(\d{1,3})\]", markdown)}


def _source_registry_for_handoff(evidence_package: Dict[str, Any], writer_report: Dict[str, Any]) -> List[Dict[str, Any]]:
    sources = (
        _as_list(_as_dict(writer_report).get("source_registry"))
        or _as_list(_as_dict(writer_report).get("sources"))
        or _as_list(evidence_package.get("source_registry"))
        or _as_list(evidence_package.get("sources"))
    )
    return [_slim_handoff_value(source) for source in sources if isinstance(source, dict)]


def _per_dimension_handoff(
    evidence_package: Dict[str, Any],
    *,
    priority_source_ids: set[str],
    per_dimension_limit: int,
) -> Dict[str, Dict[str, Any]]:
    per_dimension: Dict[str, Dict[str, Any]] = {}
    for dimension, payload in _as_dict(evidence_package.get("per_dimension")).items():
        if not isinstance(payload, dict):
            continue
        clean_facts = _as_list(payload.get("clean_facts"))
        top_evidence = _as_list(payload.get("top_evidence"))
        per_dimension[str(dimension)] = {
            "clean_facts": _select_handoff_items(
                clean_facts,
                top_evidence,
                limit=per_dimension_limit,
                priority_source_ids=priority_source_ids,
            ),
            "evidence_count": payload.get("evidence_count"),
            "coverage_score": payload.get("coverage_score"),
            "s_grade_count": payload.get("s_grade_count"),
            "conflicts": _slim_handoff_value(_as_list(payload.get("conflicts"))[:20]),
        }
    return per_dimension


def _chapter_evidence_handoff(
    evidence_package: Dict[str, Any],
    *,
    priority_source_ids: set[str],
    per_chapter_limit: int,
) -> Dict[str, List[Dict[str, Any]]]:
    chapters: Dict[str, List[Dict[str, Any]]] = {}
    for chapter_id, items in _as_dict(evidence_package.get("chapter_evidence")).items():
        chapters[str(chapter_id)] = _select_handoff_items(
            _as_list(items),
            limit=per_chapter_limit,
            priority_source_ids=priority_source_ids,
        )
    return chapters


def _reformatter_evidence_package_for_handoff(
    evidence_package: Dict[str, Any],
    writer_report: Dict[str, Any],
    structured_analysis: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    package = _as_dict(evidence_package)
    if not package:
        return {}
    writer = _as_dict(writer_report)
    cited_source_ids = _writer_cited_source_ids(writer)
    clean_limit = _env_int("REPORT_REFORMATTER_HANDOFF_MAX_CLEAN_FACTS", 600, min_value=50, max_value=5000)
    per_dimension_limit = _env_int("REPORT_REFORMATTER_HANDOFF_MAX_FACTS_PER_DIMENSION", 100, min_value=20, max_value=1000)
    per_chapter_limit = _env_int("REPORT_REFORMATTER_HANDOFF_MAX_FACTS_PER_CHAPTER", 120, min_value=20, max_value=1000)
    analysis_ready = _as_list(package.get("analysis_ready_evidence"))
    clean_facts = _as_list(package.get("clean_evidence_list"))
    selected_clean = _select_handoff_items(
        analysis_ready,
        clean_facts,
        limit=max(clean_limit, len(analysis_ready)),
        priority_source_ids=cited_source_ids,
        preserve_first_group=True,
    )
    source_registry = _source_registry_for_handoff(package, writer)
    return {
        "package_type": "reformatter_evidence_package",
        "payload_mode": "handoff",
        "query": package.get("query"),
        "research_plan": _slim_handoff_value(_as_dict(package.get("research_plan"))),
        "report_plan": _slim_handoff_value(_as_dict(package.get("report_plan"))),
        "chapter_plan": _slim_handoff_value(_as_list(package.get("chapter_plan"))[:40]),
        "chapter_dim_mapping": _slim_handoff_value(_as_dict(package.get("chapter_dim_mapping"))),
        "summary": _slim_handoff_value(_as_dict(package.get("summary"))),
        "metadata": {
            **_slim_handoff_value(_as_dict(package.get("metadata"))),
            "source": "runtime_full_evidence_package",
            "handoff_clean_fact_count": len(selected_clean),
            "handoff_analysis_ready_count": len(analysis_ready),
            "handoff_source_count": len(source_registry),
            "handoff_cited_source_count": len(cited_source_ids),
        },
        "sources": source_registry,
        "source_registry": source_registry,
        "clean_evidence_list": selected_clean,
        "analysis_ready_evidence": _select_handoff_items(
            analysis_ready,
            limit=max(clean_limit, len(analysis_ready)),
            priority_source_ids=cited_source_ids,
            preserve_first_group=True,
        ),
        "per_dimension": _per_dimension_handoff(
            package,
            priority_source_ids=cited_source_ids,
            per_dimension_limit=per_dimension_limit,
        ),
        "chapter_evidence": _chapter_evidence_handoff(
            package,
            priority_source_ids=cited_source_ids,
            per_chapter_limit=per_chapter_limit,
        ),
        "core_evidence": _select_handoff_items(_as_list(package.get("core_evidence")), limit=clean_limit, priority_source_ids=cited_source_ids),
        "supporting_evidence": _select_handoff_items(_as_list(package.get("supporting_evidence")), limit=clean_limit, priority_source_ids=cited_source_ids),
        "clue_evidence": _select_handoff_items(_as_list(package.get("clue_evidence")), limit=clean_limit, priority_source_ids=cited_source_ids),
        "appendix_evidence": _select_handoff_items(_as_list(package.get("appendix_evidence")), limit=clean_limit, priority_source_ids=cited_source_ids),
        "evidence_analysis_summary": _slim_handoff_value(_as_dict(package.get("evidence_analysis_summary"))),
        "evidence_analysis_by_chapter": _slim_handoff_value(_as_dict(package.get("evidence_analysis_by_chapter"))),
        "chapter_evidence_diagnostics": _slim_handoff_value(_as_dict(package.get("chapter_evidence_diagnostics"))),
        "evidence_gap_ledger": _slim_handoff_value(_as_list(package.get("evidence_gap_ledger"))[:120]),
        "structured_analysis_summary": _slim_handoff_value(_as_dict(structured_analysis or {}).get("evidence_analysis_summary")),
    }


def _child_error_messages(errors: Sequence[str], keywords: Sequence[str]) -> List[str]:
    selected: List[str] = []
    for item in errors:
        text = str(item or "").strip()
        if text and any(keyword in text for keyword in keywords):
            selected.append(text)
    return selected


def _evidence_relevance(score: Any) -> str:
    numeric = _safe_float(score, 0.0)
    if numeric >= 0.75:
        return "high"
    if numeric >= 0.35:
        return "medium"
    return "low"


def _normalize_rag_sources(evidence: Sequence[Dict[str, Any]], max_items: int = 5) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    for index, item in enumerate(evidence[:max_items], start=1):
        if not isinstance(item, dict):
            continue
        citation = _as_dict(item.get("citation"))
        source_id = str(item.get("id") or f"E{index}").strip()
        doc_title = str(item.get("doc_title") or citation.get("doc_title") or "").strip()
        section_title = str(item.get("section_title") or citation.get("section_title") or "").strip()
        title = " / ".join(part for part in [doc_title, section_title] if part) or source_id
        score = item.get("evidence_score")
        sources.append(
            {
                "id": source_id,
                "title": title,
                "url": "",
                "date": str(item.get("date") or citation.get("date") or "").strip(),
                "source_file": str(item.get("source_file") or citation.get("source_file") or "").strip(),
                "chunk_uid": str(item.get("chunk_uid") or citation.get("chunk_uid") or "").strip(),
                "quote": _compact_text(item.get("quote") or "", max_chars=320),
                "relevance": _evidence_relevance(score),
                "score": _safe_float(score, 0.0),
            }
        )
    return sources


def normalize_rag_child_output(
    local_state: Optional[Dict[str, Any]],
    *,
    route: str,
    errors: Sequence[str],
) -> Dict[str, Any]:
    child_errors = _child_error_messages(errors, ["本地 RAG", "RAG Agent"])
    if not local_state:
        rag_enabled = _local_rag_enabled()
        scheduled = rag_enabled and route in {"local", "both", "all"}
        return {
            "answer": "",
            "confidence": 0.0,
            "key_sources": [],
            "limitations": {
                "failure_reason": (
                    "Local RAG is disabled for the main flow."
                    if not rag_enabled
                    else "本地 RAG 子智能体未返回结果。"
                )
            },
            "status": "skipped" if not rag_enabled else "failed",
            "used": False,
            "note": (
                "Local RAG is disabled for the main flow."
                if not rag_enabled
                else "当前路由应调度本地 RAG 但未获得结果。"
                if scheduled
                else "当前路由未调度本地 RAG。"
            ),
        }

    raw_output = _as_dict(local_state.get("raw_output"))
    answer_payload = _as_dict(raw_output.get("answer"))
    answer_text = str(answer_payload.get("answer") or local_state.get("answer_text") or "").strip()
    evidence = [item for item in list(raw_output.get("evidence") or local_state.get("evidence") or []) if isinstance(item, dict)]
    confidence = _clip_confidence(answer_payload.get("confidence"), 0.0)
    answer_status = str(answer_payload.get("status") or "").strip()
    refusal_reason = str(answer_payload.get("refusal_reason") or "").strip()

    failure_answer = answer_text.startswith("RAG Agent 失败")
    if child_errors and (failure_answer or not (answer_text or evidence)):
        status = "failed"
    elif answer_status in {"answered", "conflicted"} and confidence >= 0.2 and not child_errors:
        status = "success"
    elif answer_text or evidence:
        status = "partial"
    else:
        status = "failed"
    if status == "failed":
        confidence = 0.0

    limitations = {
        "status": answer_status,
        "refusal_reason": refusal_reason,
        "gaps": _as_list(answer_payload.get("gaps")),
        "conflicts": _as_list(answer_payload.get("conflicts")),
        "review_status": str(answer_payload.get("review_status") or "").strip(),
        "review_issues": _as_list(answer_payload.get("review_issues")),
        "errors": child_errors,
    }
    note_parts = []
    if answer_status:
        note_parts.append(f"RAG 状态：{answer_status}")
    if refusal_reason:
        note_parts.append(f"拒答/降级原因：{refusal_reason}")
    if child_errors:
        note_parts.append("；".join(child_errors[:2]))
    if not note_parts:
        note_parts.append("已提取本地知识库证据。")

    return {
        "answer": answer_text,
        "confidence": confidence,
        "key_sources": _normalize_rag_sources(evidence),
        "limitations": limitations,
        "status": status,
        "used": status in {"success", "partial"} and confidence >= 0.2,
        "note": "；".join(note_parts),
    }


_RETRIEVAL_SCORE_FIELDS = (
    "web_final_score",
    "web_rerank_score",
    "web_rerank_rank",
    "task_term_score",
    "task_relevance_score",
    "lexical_relevance_score",
    "credibility_score",
)


def _retrieval_relevance_from_item(item: Dict[str, Any]) -> float:
    for key in ("web_final_score", "web_rerank_score", "task_relevance_score", "task_term_score"):
        if key in item:
            return _clip_confidence(item.get(key), 0.0)
    return 0.0


def _copy_retrieval_scores(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    for key in _RETRIEVAL_SCORE_FIELDS:
        if key in source and source.get(key) is not None:
            target[key] = source.get(key)
    relevance = _retrieval_relevance_from_item(source)
    if relevance > 0:
        target["retrieval_relevance_score"] = relevance


def _normalize_web_sources(raw_output: Dict[str, Any], payload: Dict[str, Any], max_items: int = 30) -> List[Dict[str, Any]]:
    payload_sources = [item for item in _as_list(payload.get("key_sources")) if isinstance(item, dict)]
    combined_results = list(raw_output.get("search_results") or []) + list(raw_output.get("page_results") or [])
    page_verified_urls = {
        str(item.get("url") or "").strip()
        for item in list(raw_output.get("page_results") or [])
        if isinstance(item, dict)
        and str(item.get("url") or "").strip()
        and str(item.get("mainText") or item.get("markdown") or item.get("text") or item.get("content") or item.get("summary") or "").strip()
    }

    def verification_status(source: Dict[str, Any], raw: Dict[str, Any]) -> str:
        explicit = str(raw.get("source_verification_status") or source.get("source_verification_status") or "").strip().lower()
        if explicit in {"search_result_only", "readpage_verified", "document_verified", "inaccessible"}:
            return explicit
        url = str(source.get("url") or raw.get("url") or "").strip()
        source_text = " ".join(str(value or "") for value in [url, source.get("title"), raw.get("source_type"), raw.get("origin_intent")])
        if re.search(r"\.pdf(?:$|\?)|annual[-_ ]?report|filing|prospectus|announcement|disclosure|standard|policy|regulation|official|gov\.|\.gov|exchange", source_text, re.I):
            return "document_verified"
        if url and url in page_verified_urls:
            return "readpage_verified"
        if str(raw.get("mainText") or raw.get("markdown") or raw.get("text") or raw.get("content") or "").strip():
            return "readpage_verified"
        return "search_result_only" if url else "inaccessible"

    score_lookup: Dict[tuple[str, str], Dict[str, Any]] = {}
    for item in combined_results:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        title = str(item.get("title") or "").strip()
        if url or title:
            score_lookup[(url, title)] = item
        if url:
            score_lookup[(url, "")] = item
    normalized: List[Dict[str, Any]] = []
    seen = set()
    if payload_sources:
        for item in payload_sources[:max_items]:
            source = {
                "id": item.get("id"),
                "title": str(item.get("title") or "").strip(),
                "url": str(item.get("url") or "").strip(),
                "date": str(item.get("date") or "").strip(),
                "relevance": str(item.get("relevance") or "medium").strip() or "medium",
                "snippet": _compact_text(
                    item.get("snippet") or item.get("summary") or item.get("mainText") or item.get("content"),
                    max_chars=900,
                ),
            }
            _copy_retrieval_scores(source, item)
            matched_raw = score_lookup.get((source["url"], source["title"]), {}) or score_lookup.get((source["url"], ""), {})
            _copy_retrieval_scores(source, matched_raw)
            source["source_verification_status"] = verification_status(source, matched_raw or item)
            source["source_verified"] = source["source_verification_status"] in {"readpage_verified", "document_verified"}
            key = (source["url"], source["title"])
            if key not in seen:
                seen.add(key)
                normalized.append(source)
            if len(normalized) >= max_items:
                return normalized

    sources: List[Dict[str, Any]] = []
    for index, item in enumerate(combined_results):
        if not isinstance(item, dict):
            continue
        source_id = item.get("source_id")
        if source_id is None:
            source_id = index
        source = {
            "id": source_id,
            "title": str(item.get("title") or "Untitled").strip(),
            "url": str(item.get("url") or "").strip(),
            "date": str(item.get("publishedTime") or item.get("date") or "").strip(),
            "relevance": str(item.get("relevance") or item.get("credibility_level") or "medium").strip() or "medium",
            "snippet": _compact_text(item.get("mainText") or item.get("snippet") or item.get("summary") or item.get("content"), max_chars=900),
        }
        _copy_retrieval_scores(source, item)
        source["source_verification_status"] = verification_status(source, item)
        source["source_verified"] = source["source_verification_status"] in {"readpage_verified", "document_verified"}
        key = (source["url"], source["title"])
        if key in seen:
            continue
        seen.add(key)
        sources.append(source)
        if len(normalized) + len(sources) >= max_items:
            break
    return normalized + sources


_IQS_EVIDENCE_VALUE_RE = re.compile(
    r"(?P<value>(?:约|超过|超|达到|达|为|同比|环比|预计|亏损|盈利)?\s*\d+(?:\.\d+)?\s*(?:%|pct|亿元|万亿元|亿美元|万元|亿|万台|万套|万件|台|套|件|家|倍|元|美元))",
    re.I,
)


def _source_by_citation_id(sources: Sequence[Dict[str, Any]], citation_ids: Sequence[str]) -> Dict[str, Any]:
    if citation_ids:
        wanted = {str(item).strip() for item in citation_ids if str(item).strip()}
        for source in sources:
            source_id = source.get("id")
            if source_id is None:
                source_id = source.get("source_id")
            if str(source_id).strip() in wanted:
                return source
    return {}


def _extract_citation_ids(text: str) -> List[str]:
    return re.findall(r"\[(?:id\s*[:：]\s*)?(\d+)\]", str(text or ""), flags=re.I)


def _source_match_score(source: Dict[str, Any], evidence_text: str) -> int:
    haystack = " ".join(
        str(source.get(key) or "")
        for key in ["title", "snippet", "summary", "mainText", "content", "quote", "url"]
    ).lower()
    needle = str(evidence_text or "").lower()
    if not haystack or not needle:
        return 0
    score = 0
    title = str(source.get("title") or "").strip().lower()
    if title and title in needle:
        score += 8
    for number in set(re.findall(r"\d+(?:\.\d+)?", needle)):
        if number and number in haystack:
            score += 3
    tokens = [
        token
        for token in re.findall(r"[a-zA-Z0-9]{2,}|[\u4e00-\u9fff]{2,}", needle)
        if len(token) >= 2
    ]
    for token in set(tokens[:80]):
        if token in haystack:
            score += 1
    return score


def _source_by_text(sources: Sequence[Dict[str, Any]], evidence_text: str) -> Dict[str, Any]:
    ranked = sorted(
        ((source, _source_match_score(source, evidence_text)) for source in sources),
        key=lambda item: item[1],
        reverse=True,
    )
    # Token co-occurrence is a weak signal: at the old threshold of 5 (two
    # shared numbers were enough) evidence lines were routinely stitched onto
    # the wrong source, producing citations whose prose names one publication
    # while the appendix resolves to another. Prefer leaving a line unbound
    # over fabricating its provenance.
    threshold = _env_int("BRAIN_SOURCE_TEXT_MATCH_MIN_SCORE", 8, min_value=1, max_value=100)
    if ranked and ranked[0][1] >= threshold:
        return ranked[0][0]
    return {}


def _source_for_evidence_line(
    sources: Sequence[Dict[str, Any]],
    citation_ids: Sequence[str],
    evidence_text: str,
) -> Dict[str, Any]:
    return _source_by_citation_id(sources, citation_ids) or _source_by_text(sources, evidence_text)


def _structured_evidence_to_raw_points(
    evidence_text: Any,
    *,
    sources: Sequence[Dict[str, Any]],
    dimension: str = "",
    confidence: float = 0.0,
    proof_role: str = "",
    max_items: int = 24,
) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    seen = set()
    role = str(proof_role or "").strip().lower()
    qualitative_roles = {"policy", "case", "technology", "technology_product", "counter", "source_check", "filing", "official_data", "customer_case"}
    for raw_line in re.split(r"[\n\r]+", str(evidence_text or "")):
        line = re.sub(r"^\s*[-*•\d.、\)）]+\s*", "", raw_line).strip()
        if not line or line.startswith("本次使用") or line.startswith("主要来源"):
            continue
        citation_ids = _extract_citation_ids(line)
        clean = re.sub(r"\s*\[[^\[\]]+\]", "", line).strip()
        clean = re.sub(r"\s+", " ", clean)
        if len(clean) < 18:
            continue
        has_number = bool(re.search(r"\d", clean))
        if not has_number and role not in qualitative_roles:
            continue
        tag_match = re.search(r"【([^】]+)】", clean)
        tag = tag_match.group(1).strip() if tag_match else ""
        source = _source_by_citation_id(sources, citation_ids)
        source_binding = "citation_id" if source else ""
        if not source:
            source = _source_by_text(sources, clean)
            source_binding = "fuzzy_text" if source else "unbound"
        line_period = _extract_period(clean)
        value_matches = list(_IQS_EVIDENCE_VALUE_RE.finditer(clean))
        if not value_matches:
            value_matches = [None]  # type: ignore[list-item]
        for value_match in value_matches[:4]:
            value = re.sub(r"\s+", "", value_match.group("value")) if value_match else ""
            local_context = clean[max(0, value_match.start() - 24) : min(len(clean), value_match.end() + 32)] if value_match else clean
            prefix_context = clean[max(0, value_match.start() - 16) : value_match.start()] if value_match else ""
            prefix_years = re.findall(r"20\d{2}年?", prefix_context)
            period = (prefix_years[-1] if prefix_years else "") or _extract_period(local_context) or line_period
            # Qualitative lines must not masquerade as metrics: tags like
            # 【竞争对比】and lane roles like technology_product used to be
            # written into the metric field and later rendered as fake
            # indicator sentences ("technology_product为20"). The tag and the
            # proof role keep their own fields below.
            metric = _infer_metric_from_context(clean, value) if value else ""
            key = (dimension, metric, value, clean[:100], str(source.get("url") or ""))
            if key in seen:
                continue
            seen.add(key)
            points.append(
                {
                    "dimension": dimension or _infer_dimension_from_text(clean),
                    "metric": metric,
                    "value": value,
                    "period": period or str(source.get("date") or "").strip(),
                    "source": str(source.get("title") or source.get("url") or "IQS来源").strip(),
                    "source_title": str(source.get("title") or "").strip(),
                    "source_url": str(source.get("url") or "").strip(),
                    "date": str(source.get("date") or "").strip(),
                    "evidence": _compact_text(clean, max_chars=900),
                    "tag": tag,
                    "confidence": confidence,
                    "citation_ids": citation_ids,
                    "proof_role": role,
                    "source_binding": source_binding,
                    "source_binding_fuzzy": source_binding == "fuzzy_text",
                    "source_verification_status": str(source.get("source_verification_status") or ("readpage_verified" if source.get("readpage_verified") else "search_result_only")).strip(),
                    "source_verified": bool(source.get("source_verified") or str(source.get("source_verification_status") or "").strip() in {"readpage_verified", "document_verified"}),
                }
            )
            _copy_retrieval_scores(points[-1], source)
            if len(points) >= max_items:
                return points
    return points


def normalize_web_child_output(
    web_state: Optional[Dict[str, Any]],
    *,
    route: str,
    errors: Sequence[str],
) -> Dict[str, Any]:
    child_errors = _child_error_messages(errors, ["联网分析", "IQS", "web_analysis"])
    if not web_state:
        scheduled = route in {"web", "both", "all"}
        return {
            "answer": "",
            "confidence": 0.0,
            "key_sources": [],
            "limitations": {"failure_reason": "联网分析子智能体未返回结果。"},
            "status": "failed",
            "used": False,
            "note": "当前路由应调度联网分析但未获得结果。" if scheduled else "当前路由未调度联网分析。",
        }

    raw_output = _as_dict(web_state.get("raw_output"))
    synthesis = _as_dict(raw_output.get("synthesis"))
    payload = _as_dict(synthesis.get("structured_payload"))
    answer_payload = _as_dict(payload.get("answer"))
    answer_text = str(web_state.get("answer_text") or "").strip()
    if answer_payload:
        pieces = [
            answer_payload.get("conclusion"),
            answer_payload.get("evidence"),
            answer_payload.get("inference"),
        ]
        structured_text = "\n".join(str(item).strip() for item in pieces if str(item or "").strip())
        if structured_text:
            answer_text = structured_text

    search_results = list(raw_output.get("search_results") or [])
    page_results = list(raw_output.get("page_results") or [])
    source_count = len(search_results) + len(page_results)
    raw_confidence = payload.get("confidence")
    if raw_confidence is None:
        raw_confidence = synthesis.get("confidence")
    confidence = _clip_confidence(raw_confidence, 0.0)
    if confidence == 0.0 and source_count:
        confidence = _clip_confidence(min(0.55, 0.25 + 0.05 * min(source_count, 6)))

    partial_errors = _as_list(_as_dict(web_state.get("metadata")).get("partial_errors"))
    synthesis_source = str(synthesis.get("source") or "").strip()
    failure_answer = answer_text.startswith("联网分析 Agent 失败")
    if child_errors and (failure_answer or source_count == 0):
        status = "failed"
    elif source_count == 0:
        status = "failed"
    elif child_errors or partial_errors or synthesis.get("error") or synthesis_source == "fallback_extractive":
        status = "partial"
    elif confidence >= 0.2:
        status = "success"
    else:
        status = "partial"
    if status == "failed":
        confidence = 0.0

    quality = _as_dict(raw_output.get("quality_processing"))
    rerank_diagnostics = _as_dict(quality.get("rerank"))
    limitations = _as_dict(payload.get("limitations"))
    if not limitations:
        limitations = {
            "data_recency": str(_as_dict(raw_output.get("search_options")).get("timeRange") or "未限定时间范围").strip(),
            "coverage": f"IQS 返回 {source_count} 条候选；精排后保留 {quality.get('final_count', source_count)} 条。" if source_count else "未返回可用来源。",
            "conflicts": None,
        }
    if answer_payload.get("evidence_gap"):
        limitations = {**limitations, "evidence_gap": answer_payload.get("evidence_gap")}
    search_options = _as_dict(raw_output.get("search_options"))
    search_task = _as_dict(search_options.get("search_task"))
    normalized_sources = _normalize_web_sources(raw_output, payload)
    extracted_fact_cards = [dict(item) for item in _as_list(raw_output.get("extracted_fact_cards")) if isinstance(item, dict)]
    fact_extractor = _as_dict(raw_output.get("fact_extractor"))
    regex_fallback_point_count = 0
    regex_fallback_used = False
    extractor_empty_without_regex_points = False
    if status == "failed" or failure_answer:
        raw_data_points = []
    elif extracted_fact_cards:
        raw_data_points = []
        for index, item in enumerate(extracted_fact_cards, start=1):
            point = dict(item)
            point.setdefault("evidence_origin", "readpage_fact_extractor")
            point.setdefault("extraction_schema_version", "readpage_fact_card_v2")
            point.setdefault("confidence", confidence)
            point.setdefault("evidence", point.get("distilled_fact") or point.get("fact") or point.get("clean_fact"))
            point.setdefault("content", point.get("distilled_fact") or point.get("fact") or point.get("clean_fact"))
            point.setdefault("clean_fact", point.get("distilled_fact") or point.get("fact") or point.get("content"))
            point.setdefault("fact", point.get("clean_fact") or point.get("content"))
            point.setdefault("metric", point.get("metric") or point.get("variable") or point.get("fact_type") or point.get("proof_role"))
            point.setdefault("source", point.get("source_title") or point.get("source_url") or point.get("source_ref") or "readpage_fact_extractor")
            point.setdefault("source_title", point.get("source_title") or point.get("title") or "")
            point.setdefault("source_url", point.get("source_url") or point.get("url") or "")
            point.setdefault("source_verification_status", point.get("source_verification_status") or "readpage_verified")
            point.setdefault("source_verified", str(point.get("source_verification_status") or "").strip() in {"readpage_verified", "document_verified"})
            point.setdefault("ref", point.get("ref") or point.get("evidence_id") or f"RFC-{index}")
            point.setdefault("evidence_id", point.get("evidence_id") or point.get("ref") or f"RFC-{index}")
            raw_data_points.append(point)
    else:
        raw_data_points = _structured_evidence_to_raw_points(
            answer_payload.get("evidence") or answer_text,
            sources=normalized_sources,
            dimension=str(raw_output.get("dimension") or "").strip(),
            confidence=confidence,
            proof_role=str(search_task.get("proof_role") or search_task.get("evidence_type") or "").strip(),
        )
        regex_fallback_point_count = len(raw_data_points)
        regex_fallback_used = regex_fallback_point_count > 0
        extractor_empty_without_regex_points = not regex_fallback_used
    limitations = {
        **limitations,
        "fact_extractor": {
            **fact_extractor,
            "regex_fallback_used": regex_fallback_used,
            "regex_fallback_point_count": regex_fallback_point_count,
            "extractor_empty_without_regex_points": bool(
                status != "failed"
                and not failure_answer
                and not extracted_fact_cards
                and extractor_empty_without_regex_points
            ),
        },
        "errors": child_errors,
        "partial_errors": partial_errors,
    }
    if search_task:
        for point in raw_data_points:
            point["task_id"] = search_task.get("task_id")
            point["dimension_id"] = search_task.get("dimension_id")
            point["dimension_name"] = search_task.get("dimension_name") or search_task.get("dimension")
            point["evidence_goal"] = search_task.get("evidence_goal")
            point["must_have_terms"] = _as_list(search_task.get("must_have_terms"))
            point["forbidden_terms"] = _as_list(search_task.get("forbidden_terms"))
            point["source_priority"] = _as_list(search_task.get("source_priority"))
            point["search_task"] = dict(search_task)

    note_parts = []
    if synthesis_source:
        note_parts.append(f"联网综合来源：{synthesis_source}")
    if child_errors:
        note_parts.append("；".join(child_errors[:2]))
    if partial_errors:
        note_parts.append("部分搜索任务降级。")
    if not note_parts:
        note_parts.append("已提取联网公开证据。")

    return {
        "answer": answer_text,
        "confidence": confidence,
        "key_sources": normalized_sources,
        "limitations": limitations,
        "status": status,
        "used": status in {"success", "partial"} and confidence >= 0.2,
        "note": "；".join(note_parts),
        "raw_data_points": raw_data_points,
        "rerank_diagnostics": rerank_diagnostics,
    }


def normalize_iqs_role_child_output(
    role_key: str,
    role_state: Optional[Dict[str, Any]],
    *,
    route: str,
    errors: Sequence[str],
) -> Dict[str, Any]:
    config = IQS_ROLE_CONFIGS[role_key]
    child = normalize_web_child_output(role_state, route=route, errors=errors)
    role_raw_points = [dict(item) for item in _as_list(_as_dict(role_state).get("raw_data_points")) if isinstance(item, dict)]
    if role_raw_points and child.get("status") != "failed" and child.get("used"):
        child["raw_data_points"] = role_raw_points
    child["role_key"] = role_key
    role_metadata = _as_dict(_as_dict(role_state).get("metadata"))
    dynamic_tasks = _as_list(role_metadata.get("dynamic_tasks"))
    if not dynamic_tasks:
        dynamic_tasks = _as_list(_as_dict(_as_dict(role_state).get("raw_output")).get("dynamic_tasks"))
    dimensions = _unique_strings(
        [str(_as_dict(task).get("dimension_name") or "") for task in _as_list(dynamic_tasks) if isinstance(task, dict)],
        max_items=6,
    )
    child["dimension"] = " / ".join(dimensions) if dimensions else "动态检索任务"
    child["dynamic_tasks"] = _as_list(dynamic_tasks)
    child["label"] = config["label"]
    note = str(child.get("note") or "").strip()
    child["note"] = f"{config['label']}；{note}" if note else config["label"]
    return child


def aggregate_iqs_role_children(children: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    role_children = []
    for role_key in IQS_ROLE_ORDER:
        child_name = IQS_ROLE_CONFIGS[role_key]["child"]
        child = children.get(child_name) or {}
        if child:
            role_children.append((role_key, child))
    if not role_children:
        return {
            "answer": "",
            "confidence": 0.0,
            "key_sources": [],
            "limitations": {"coverage": "未调度"},
            "status": "failed",
            "used": False,
            "note": "未调度 IQS 动态检索任务。",
        }

    answer_parts: List[str] = []
    sources: List[Dict[str, Any]] = []
    confidences: List[float] = []
    statuses: List[str] = []
    gaps: List[Any] = []
    covered_dimensions: List[str] = []
    for role_key, child in role_children:
        config = IQS_ROLE_CONFIGS[role_key]
        answer = str(child.get("answer") or "").strip()
        if answer:
            answer_parts.append(f"【{config['label']}】\n{answer}")
        sources.extend([item for item in _as_list(child.get("key_sources")) if isinstance(item, dict)])
        confidences.append(_clip_confidence(child.get("confidence"), 0.0))
        statuses.append(str(child.get("status") or "failed"))
        limitations = _as_dict(child.get("limitations"))
        gaps.extend(_as_list(limitations.get("evidence_gap")))
        gaps.extend(_as_list(limitations.get("gaps")))
        for task in _as_list(child.get("dynamic_tasks")):
            name = str(_as_dict(task).get("dimension_name") or "").strip()
            if name and name not in covered_dimensions:
                covered_dimensions.append(name)
        child_dimension = str(child.get("dimension") or "").strip()
        if child_dimension and child_dimension not in covered_dimensions:
            covered_dimensions.append(child_dimension)

    source_seen = set()
    deduped_sources: List[Dict[str, Any]] = []
    for source in sources:
        key = (str(source.get("url") or ""), str(source.get("title") or ""))
        if key in source_seen:
            continue
        source_seen.add(key)
        deduped_sources.append(source)
        if len(deduped_sources) >= 30:
            break

    success_count = len([status for status in statuses if status == "success"])
    partial_count = len([status for status in statuses if status == "partial"])
    status = "success" if success_count >= 3 else "partial" if success_count or partial_count else "failed"
    confidence = round(sum(confidences) / max(len([item for item in confidences if item > 0]), 1), 4) if confidences else 0.0
    if status == "failed":
        confidence = 0.0
    return {
        "answer": "\n\n".join(answer_parts).strip(),
        "confidence": confidence,
        "key_sources": deduped_sources,
        "limitations": {
            "data_recency": "5个 IQS 角色化子智能体并发检索结果",
            "coverage": "、".join(covered_dimensions) if covered_dimensions else "动态 IQS 任务检索结果",
            "conflicts": None,
            "evidence_gap": gaps,
        },
        "status": status,
        "used": status in {"success", "partial"} and confidence >= 0.2,
        "note": f"已汇总 {len(role_children)} 个 IQS 角色化子智能体输出。",
    }


def _has_conflict_marker(value: Any) -> bool:
    text = json.dumps(value, ensure_ascii=False, default=json_safe_default) if isinstance(value, (dict, list)) else str(value or "")
    if not text:
        return False
    if re.search(r"(矛盾|冲突|分歧|不一致|不同口径|相互矛盾|conflict)", text, re.I):
        lowered = text.strip().lower()
        return lowered not in {"none", "null", "无", "无冲突", "若无则填 null"}
    return False


def _source_is_authoritative(sources: Sequence[Dict[str, Any]]) -> bool:
    authority_terms = [
        "gov.cn",
        "stats.gov",
        "miit.gov",
        "ndrc.gov",
        "mofcom.gov",
        "csrc.gov",
        "reuters",
        "bloomberg",
        "caixin",
        "wind",
        "国家统计局",
        "工信部",
        "发改委",
        "证监会",
        "交易所",
    ]
    text = json.dumps(list(sources or []), ensure_ascii=False, default=json_safe_default).lower()
    return any(term.lower() in text for term in authority_terms)


def _has_dated_web_source(web_child: Dict[str, Any]) -> bool:
    for item in web_child.get("key_sources") or []:
        if isinstance(item, dict) and str(item.get("date") or "").strip():
            return True
    data_recency = _as_dict(web_child.get("limitations")).get("data_recency")
    return bool(str(data_recency or "").strip())


def _infer_conflict_priority(rag_child: Dict[str, Any], web_child: Dict[str, Any]) -> tuple[str, str]:
    if _has_dated_web_source(web_child):
        return "web_preferred", "联网来源带有时间信息，本地知识库更适合作历史对比。"
    if _source_is_authoritative(web_child.get("key_sources") or []):
        return "web_preferred", "联网来源包含政府、官方或头部媒体信息，权威性更高。"
    if _source_is_authoritative(rag_child.get("key_sources") or []):
        return "rag_preferred", "本地知识库来源更权威，暂优先采信 RAG。"
    return "unresolved", "当前来源权威性和时效性不足以裁定分歧，需补充核验。"


def _build_agent_trace(children: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    trace: List[Dict[str, Any]] = []
    ordered_agents = ["industry_rag_agent", *[IQS_ROLE_CONFIGS[key]["child"] for key in IQS_ROLE_ORDER], "web_analysis_agent"]
    for agent_name in ordered_agents:
        if agent_name not in children:
            continue
        child = children.get(agent_name) or {}
        trace.append(
            {
                "agent": agent_name,
                "status": str(child.get("status") or "failed"),
                "confidence": _clip_confidence(child.get("confidence"), 0.0),
                "used": bool(child.get("used")),
                "note": str(child.get("note") or "").strip(),
            }
        )
    return trace


def _tagged_insight(tag: str, child: Dict[str, Any], missing_text: str) -> str:
    answer = _compact_text(child.get("answer"), max_chars=1600)
    if answer:
        return f"{tag} {answer}"
    return missing_text


def _first_claim(text: str, max_chars: int = 180) -> str:
    for line in str(text or "").splitlines():
        cleaned = re.sub(r"^(核心判断|结论|关键判断)[:：]\s*", "", line.strip())
        if cleaned:
            return _compact_text(cleaned, max_chars=max_chars)
    return ""


def _calculate_supervisor_confidence(rag_child: Dict[str, Any], web_child: Dict[str, Any], *, has_conflict: bool) -> float:
    rag_status = str(rag_child.get("status") or "failed")
    web_status = str(web_child.get("status") or "failed")
    rag_conf = _clip_confidence(rag_child.get("confidence"), 0.0)
    web_conf = _clip_confidence(web_child.get("confidence"), 0.0)

    if (rag_status == "failed" and web_status == "failed") or (rag_conf < 0.2 and web_conf < 0.2):
        return 0.0
    if rag_status == "success" and web_status == "success":
        if has_conflict:
            return round(min(0.70, (rag_conf + web_conf) / 2), 4)
        return round(min(0.95, (rag_conf + web_conf) / 2 + 0.10), 4)
    if rag_status == "success" and web_status != "success":
        return round(rag_conf * 0.85, 4)
    if web_status == "success" and rag_status != "success":
        return round(web_conf * 0.85, 4)

    usable_conf = max(conf for conf in [rag_conf, web_conf] if conf >= 0.2)
    return round(usable_conf * 0.65, 4)


def _calculate_supervisor_confidence_v2(children: Dict[str, Dict[str, Any]], *, has_conflict: bool) -> float:
    """Confidence helper for the multi-lane supervisor path."""
    usable: List[float] = []
    for child in children.values():
        if not isinstance(child, dict):
            continue
        status = str(child.get("status") or "failed").strip().lower()
        confidence = _clip_confidence(child.get("confidence"), 0.0)
        if status == "success":
            usable.append(confidence)
        elif status == "partial" and confidence >= 0.2:
            usable.append(confidence * 0.75)
    if not usable:
        return 0.0
    best = max(usable)
    average = sum(usable) / len(usable)
    combined = best * 0.65 + average * 0.35
    if len(usable) >= 2:
        combined += 0.05
    if has_conflict:
        combined = min(combined, 0.70)
    return round(min(0.95, combined), 4)


def _collect_evidence_gaps(children: Dict[str, Dict[str, Any]], *, confidence: float, has_conflict: bool) -> List[Dict[str, str]]:
    gaps: List[Dict[str, str]] = []
    rag_child = children.get("industry_rag_agent") or {}
    web_child = children.get("web_analysis_agent") or {}

    for agent_name, child, missing_from, dimension, suggestion in [
        ("industry_rag_agent", rag_child, "rag", "本地知识库证据", "补充或重新同步本地行业资料、尽调报告、研报切片后再检索。"),
        ("web_analysis_agent", web_child, "web", "联网公开信息", "补充 IQS 联网搜索，优先查官方、媒体、财报和政策来源。"),
    ]:
        status = str(child.get("status") or "failed")
        note = str(child.get("note") or "")
        if "未调度" in note:
            continue
        if status == "failed":
            gaps.append({"dimension": dimension, "missing_from": missing_from, "suggestion": suggestion})
        elif status == "partial":
            gaps.append({"dimension": f"{dimension}完整性", "missing_from": missing_from, "suggestion": suggestion})

    for item in _as_list(_as_dict(rag_child.get("limitations")).get("gaps")):
        if item:
            gaps.append({"dimension": _compact_text(item, 80), "missing_from": "rag", "suggestion": "围绕该缺口补充本地证据后重新检索。"})
    for item in _as_list(_as_dict(web_child.get("limitations")).get("evidence_gap")):
        if item:
            gaps.append({"dimension": _compact_text(item, 80), "missing_from": "web", "suggestion": "围绕该缺口发起补充联网搜索。"})

    if has_conflict:
        gaps.append({"dimension": "本地与联网结论分歧", "missing_from": "both", "suggestion": "补充更权威或更新的来源，核验同一指标的口径和时间。"})
    if confidence > 0 and not gaps:
        gaps.append({"dimension": "关键数据复核", "missing_from": "both", "suggestion": "进入 Analysis Agent 前保留原始来源，并抽样核验关键日期、数值和引用。"})

    deduped: List[Dict[str, str]] = []
    seen = set()
    for item in gaps:
        key = (item["dimension"], item["missing_from"])
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped


INDUSTRY_DIMENSIONS = [
    "综合研究问题",
]

DIMENSION_KEYWORDS = {
    "综合研究问题": [],
}

PRODUCT_COMMERCIAL_GAP_SPECS: List[Dict[str, Any]] = []


def _normalize_query_text(query: str) -> str:
    return re.sub(r"\s+", "", str(query or "").strip().lower())


def _has_specific_data(text: str) -> bool:
    return bool(re.search(r"(\d{4}|20\d{2}Q[1-4]|\d+(?:\.\d+)?\s*(?:%|亿元|万亿元|亿美元|万台|台|家|倍|pct))", text, re.I))


def _item_to_evidence_text(item: Dict[str, Any], max_chars: int = 1400) -> str:
    sources = item.get("key_sources") or []
    source_text = []
    for source in sources[:5]:
        if not isinstance(source, dict):
            continue
        source_text.append(
            " ".join(
                str(part or "").strip()
                for part in [
                    source.get("title"),
                    source.get("date"),
                    source.get("url"),
                    source.get("quote"),
                    source.get("source_file"),
                ]
                if str(part or "").strip()
            )
        )
    return _compact_text(
        " ".join(
            part
            for part in [
                str(item.get("query") or ""),
                str(item.get("answer") or ""),
                _json_text(item.get("raw_data_points"), max_chars=700),
                " ".join(source_text),
                _json_text(item.get("limitations"), max_chars=500),
            ]
            if part
        ),
        max_chars=max_chars,
    )


def coverage_units_from_state(state: Optional[BrainAgentState]) -> List[Dict[str, Any]]:
    state = state or {}
    blueprint = _as_dict(state.get("report_blueprint"))
    chapters = [chapter for chapter in _as_list(blueprint.get("chapters")) if isinstance(chapter, dict)]
    if chapters:
        return [
            {
                "unit_id": str(chapter.get("chapter_id") or f"ch_{index:02d}"),
                "unit_title": str(chapter.get("chapter_title") or f"章节 {index}"),
                "core_question": str(chapter.get("core_question") or chapter.get("chapter_question") or ""),
                "required_evidence_mix": _as_list(chapter.get("required_evidence_mix")),
                "min_total_sources": int(chapter.get("min_total_sources") or _env_int("REPORT_MIN_TOTAL_SOURCES_PER_CHAPTER", 6)),
                "min_ab_sources": int(chapter.get("min_ab_sources") or _env_int("REPORT_MIN_AB_SOURCES_PER_CHAPTER", 2)),
                "min_counter_sources": int(chapter.get("min_counter_sources") or 0),
            }
            for index, chapter in enumerate(chapters, start=1)
        ]
    plan = _as_dict(state.get("research_plan")) or _as_dict(_as_dict(state.get("query_analysis")).get("research_plan"))
    plan_chapters = [chapter for chapter in _as_list(plan.get("chapters")) if isinstance(chapter, dict)]
    if plan_chapters:
        fake_state: BrainAgentState = {"report_blueprint": {"chapters": plan_chapters}}  # type: ignore[typeddict-item]
        return coverage_units_from_state(fake_state)
    dimensions = [item for item in _as_list(plan.get("dimensions")) if isinstance(item, dict)]
    if dimensions:
        return [
            {
                "unit_id": str(item.get("dimension_id") or f"dim_{index}"),
                "unit_title": str(item.get("dimension_name") or item.get("name") or f"研究问题 {index}"),
                "core_question": str(item.get("purpose") or ""),
                "required_evidence_mix": ["official_data", "market_research", "counter_evidence"],
                "min_total_sources": _env_int("REPORT_MIN_TOTAL_SOURCES_PER_CHAPTER", 6),
                "min_ab_sources": _env_int("REPORT_MIN_AB_SOURCES_PER_CHAPTER", 2),
                "min_counter_sources": 0,
            }
            for index, item in enumerate(dimensions, start=1)
        ]
    return [
        {
            "unit_id": "core_question",
            "unit_title": "综合研究问题",
            "core_question": str(state.get("query") or ""),
            "required_evidence_mix": ["official_data", "market_research", "counter_evidence"],
            "min_total_sources": _env_int("REPORT_MIN_TOTAL_SOURCES_PER_CHAPTER", 6),
            "min_ab_sources": _env_int("REPORT_MIN_AB_SOURCES_PER_CHAPTER", 2),
            "min_counter_sources": 0,
        }
    ]


def _pool_item_matches_unit(item: Dict[str, Any], unit: Dict[str, Any]) -> bool:
    unit_id = str(unit.get("unit_id") or "").strip()
    unit_title = str(unit.get("unit_title") or "").strip()
    search_task = _as_dict(item.get("search_task"))
    if unit_id and unit_id in {
        str(item.get("chapter_id") or "").strip(),
        str(item.get("dimension_id") or "").strip(),
        str(search_task.get("chapter_id") or "").strip(),
        str(search_task.get("dimension_id") or "").strip(),
    }:
        return True
    if unit_title and unit_title in {
        str(item.get("chapter_title") or "").strip(),
        str(item.get("dimension_name") or "").strip(),
        str(search_task.get("chapter_title") or "").strip(),
        str(search_task.get("dimension_name") or "").strip(),
        str(item.get("targets_gap") or "").strip(),
    }:
        return True
    text = _item_to_evidence_text(item, max_chars=1600).lower()
    token = str(unit_title or unit.get("core_question") or "").lower()
    return bool(token and (token in text or text in token))


def _source_levels_from_pool_item(item: Dict[str, Any]) -> List[str]:
    levels: List[str] = []
    for source in _as_list(item.get("key_sources")):
        if isinstance(source, dict):
            level = str(source.get("credibility") or source.get("source_level") or source.get("level") or "").strip().upper()
            if level:
                levels.append(level)
    for point in _as_list(item.get("raw_data_points")):
        if isinstance(point, dict):
            level = str(point.get("source_level") or _as_dict(point.get("source")).get("credibility") or "").strip().upper()
            if level:
                levels.append(level)
    task = _as_dict(item.get("search_task"))
    if str(task.get("proof_role") or "").lower() in {"metric", "source_check"} and item.get("status") != "failed":
        levels.append("B")
    return levels


def _unit_coverage_score_from_pool(unit: Dict[str, Any], evidence_pool: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    matched = [
        item
        for item in evidence_pool
        if isinstance(item, dict) and str(item.get("status") or "") != "failed" and _pool_item_matches_unit(item, unit)
    ]
    levels = [level for item in matched for level in _source_levels_from_pool_item(item)]
    ab_count = len([level for level in levels if level in {"A", "B"}])
    counter_count = len(
        [
            item
            for item in matched
            if str(_as_dict(item.get("search_task")).get("proof_role") or item.get("proof_role") or "").lower() == "counter"
            or "反证" in _item_to_evidence_text(item, max_chars=500)
            or "风险" in _item_to_evidence_text(item, max_chars=500)
        ]
    )
    total_required = max(1, int(unit.get("min_total_sources") or 6))
    ab_required = max(1, int(unit.get("min_ab_sources") or 2))
    counter_required = max(0, int(unit.get("min_counter_sources") or 0))
    has_specific = any(_has_specific_data(_item_to_evidence_text(item)) for item in matched)
    score_parts = [
        min(len(matched) / total_required, 1.0),
        min(ab_count / ab_required, 1.0),
        1.0 if counter_required == 0 else min(counter_count / counter_required, 1.0),
        1.0 if has_specific else 0.4 if matched else 0.0,
    ]
    score = round(sum(score_parts) / len(score_parts), 4)
    gaps: List[str] = []
    if len(matched) < total_required:
        gaps.append("missing_total_sources")
    if ab_count < ab_required:
        gaps.append("missing_ab_sources")
    if counter_count < counter_required:
        gaps.append("missing_counter")
    if not has_specific:
        gaps.append("missing_metric_scope_period_unit")
    return {
        "score": score,
        "reason": f"matched={len(matched)}/{total_required}; A/B={ab_count}/{ab_required}; counter={counter_count}/{counter_required}",
        "source_count": len(matched),
        "ab_source_count": ab_count,
        "counter_source_count": counter_count,
        "missing": gaps,
    }


def build_followup_queries_for_chapter(chapter: Dict[str, Any], coverage_gap: Dict[str, Any]) -> List[Dict[str, Any]]:
    question = str(chapter.get("core_question") or chapter.get("unit_title") or "").strip()
    chapter_id = str(chapter.get("unit_id") or chapter.get("chapter_id") or "").strip()
    title = str(chapter.get("unit_title") or chapter.get("chapter_title") or "").strip()
    missing = set(_as_list(coverage_gap.get("missing")))
    queries: List[Dict[str, Any]] = []
    if "missing_total_sources" in missing or "missing_ab_sources" in missing:
        queries.append(
            {
                "query": f"{question or title} 官方 统计 协会 白皮书 数据",
                "agent": "iqs_lane_1",
                "targets_gap": title,
                "chapter_id": chapter_id,
                "chapter_title": title,
                "chapter_question": question,
                "proof_role": "source_check",
                "lane_targets": ["official_data"],
            }
        )
        queries.append(
            {
                "query": f"{question or title} 年报 公告 招股书 公司 披露",
                "agent": "iqs_lane_2",
                "targets_gap": title,
                "chapter_id": chapter_id,
                "chapter_title": title,
                "chapter_question": question,
                "proof_role": "source_check",
                "lane_targets": ["filing_company"],
            }
        )
    if "missing_metric_scope_period_unit" in missing:
        queries.append(
            {
                "query": f"{question or title} 数据 统计 口径 年度 单位 指标",
                "agent": "iqs_lane_1",
                "targets_gap": title,
                "chapter_id": chapter_id,
                "chapter_title": title,
                "chapter_question": question,
                "proof_role": "metric",
                "lane_targets": ["official_data", "market_research"],
            }
        )
    if "missing_counter" in missing:
        queries.append(
            {
                "query": f"{question or title} 风险 反证 产能过剩 价格下跌 需求不及预期",
                "agent": "iqs_lane_4",
                "targets_gap": title,
                "chapter_id": chapter_id,
                "chapter_title": title,
                "chapter_question": question,
                "proof_role": "counter",
                "lane_targets": ["news_event", "market_research"],
            }
        )
    return queries


def _item_to_signal_text(item: Dict[str, Any], max_chars: int = 1400) -> str:
    sources = item.get("key_sources") or []
    source_text = []
    for source in sources[:5]:
        if not isinstance(source, dict):
            continue
        source_text.append(
            " ".join(
                str(part or "").strip()
                for part in [
                    source.get("title"),
                    source.get("date"),
                    source.get("quote"),
                    source.get("source_file"),
                ]
                if str(part or "").strip()
            )
        )
    return _compact_text(
        " ".join(
            part
            for part in [
                str(item.get("answer") or ""),
                _json_text(item.get("raw_data_points"), max_chars=800),
                " ".join(source_text),
                _json_text(item.get("limitations"), max_chars=400),
            ]
            if part
        ),
        max_chars=max_chars,
    )


def _product_commercial_gaps_from_pool(original_query: str, evidence_pool: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
    return []


def build_initial_evidence_pool(
    *,
    original_query: str,
    children: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    pool: List[Dict[str, Any]] = []
    child_agent_pairs: List[tuple[str, str]] = [("industry_rag_agent", "rag")]
    child_agent_pairs.extend((IQS_ROLE_CONFIGS[key]["child"], key) for key in IQS_ROLE_ORDER)
    has_role_iqs = any(children.get(IQS_ROLE_CONFIGS[key]["child"]) for key in IQS_ROLE_ORDER)
    if not has_role_iqs:
        child_agent_pairs.append(("web_analysis_agent", "iqs"))

    for child_name, agent in child_agent_pairs:
        child = children.get(child_name) or {}
        note_text = str(child.get("note") or "")
        status = str(child.get("status") or "failed").strip().lower()
        if status in {"skipped", "disabled"} or "未调度" in note_text or "已停用" in note_text or "Local RAG is disabled" in note_text:
            continue
        pool.append(
            {
                "round": 1,
                "agent": agent,
                "child_agent": child_name,
                "query": original_query,
                "targets_gap": "初始问题",
                "status": status,
                "confidence": _clip_confidence(child.get("confidence"), 0.0),
                "answer": str(child.get("answer") or "").strip(),
                "key_sources": list(child.get("key_sources") or []),
                "limitations": _as_dict(child.get("limitations")),
                "note": str(child.get("note") or "").strip(),
                "raw_data_points": list(child.get("raw_data_points") or []),
                "data_gap": list(child.get("data_gap") or []),
            }
        )
    return pool


def summarize_evidence_pool(evidence_pool: Sequence[Dict[str, Any]], *, max_chars: int = 1500) -> str:
    buckets: Dict[str, List[tuple[int, Dict[str, Any], str]]] = {dimension: [] for dimension in INDUSTRY_DIMENSIONS}
    for item in evidence_pool:
        if not isinstance(item, dict) or str(item.get("status") or "") not in {"success", "partial"}:
            continue
        text = _item_to_evidence_text(item)
        if not text:
            continue
        matched = []
        for dimension, keywords in DIMENSION_KEYWORDS.items():
            if not keywords or any(keyword.lower() in text.lower() for keyword in keywords):
                matched.append(dimension)
        if not matched:
            matched = ["综合研究问题"]
        priority = 2 if _has_specific_data(text) else 1
        priority += 1 if item.get("agent") == "iqs" or item.get("agent") in IQS_ROLE_CONFIGS else 0
        for dimension in matched:
            buckets[dimension].append((priority, item, text))

    lines: List[str] = []
    uncovered: List[str] = []
    for dimension in INDUSTRY_DIMENSIONS:
        lines.append(f"【{dimension}】")
        candidates = sorted(
            buckets.get(dimension, []),
            key=lambda value: (value[0], _safe_float(value[1].get("confidence"), 0.0)),
            reverse=True,
        )
        if not candidates:
            lines.append("- 无实质性证据")
            uncovered.append(f"{dimension}：无实质性证据")
            continue
        for _, item, text in candidates[:3]:
            agent_key = str(item.get("agent") or "")
            source_label = IQS_ROLE_CONFIGS.get(agent_key, {}).get("label") or ("IQS" if agent_key == "iqs" else "RAG")
            round_label = item.get("round") or 1
            query = _compact_text(item.get("query"), max_chars=90)
            snippet = _compact_text(text, max_chars=230)
            lines.append(f"- [{source_label}·第{round_label}轮] {query}：{snippet}")
        if not any(_has_specific_data(text) for _, _, text in candidates):
            uncovered.append(f"{dimension}：仅有模糊线索，缺少具体数字或时间范围")
    if uncovered:
        lines.append("【未覆盖维度】")
        for item in uncovered:
            lines.append(f"- {item}")
    return _compact_text("\n".join(lines), max_chars=max_chars)


def _dimension_score_from_pool(dimension: str, evidence_pool: Sequence[Dict[str, Any]]) -> tuple[float, str]:
    keywords = DIMENSION_KEYWORDS.get(dimension, [])
    matched_texts: List[str] = []
    for item in evidence_pool:
        if not isinstance(item, dict) or str(item.get("status") or "") == "failed":
            continue
        text = _item_to_evidence_text(item)
        if not keywords or any(keyword.lower() in text.lower() for keyword in keywords):
            matched_texts.append(text)
    if not matched_texts:
        return 0.0, "当前证据池没有覆盖该维度。"
    if any(_has_specific_data(text) for text in matched_texts):
        return 1.0, "已有带时间、数字或明确来源的实质性证据。"
    return 0.5, "已有相关线索，但缺少具体数字、时间范围或权威来源。"


def _fallback_followup_agent(dimension: str, score: float) -> str:
    return "both"


def _fallback_followup_queries(
    *,
    original_query: str,
    gaps: Sequence[Dict[str, Any]],
    previous_queries: Sequence[str],
    max_queries: int,
    coverage_units: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, str]]:
    year = datetime.now().year
    templates = {
        "综合研究问题": [
            "{query} {year} 可核验证据 来源 口径 时间范围",
            "{query} {year} 反向证据 风险 关键假设 验证清单",
        ],
    }
    units_by_title = {
        str(unit.get("unit_title") or ""): unit
        for unit in list(coverage_units or [])
        if isinstance(unit, dict)
    }
    seen = {_normalize_query_text(query) for query in previous_queries}
    followups: List[Dict[str, str]] = []
    ordered_gaps = sorted(
        [gap for gap in gaps if isinstance(gap, dict)],
        key=lambda gap: 0 if str(gap.get("gap_key") or "").strip() else 1,
    )
    for gap in ordered_gaps:
        severity = str(gap.get("severity") or "").strip()
        if severity not in {"critical", "moderate"}:
            continue
        dimension = str(gap.get("dimension") or "").strip()
        unit = units_by_title.get(dimension) or {}
        if unit:
            for item in build_followup_queries_for_chapter(unit, gap):
                query = str(item.get("query") or "").strip()
                key = _normalize_query_text(query)
                if not query or key in seen:
                    continue
                seen.add(key)
                followups.append(item)
                if len(followups) >= max(1, max_queries):
                    return followups
            continue
        selected_templates = templates.get(dimension) or templates["综合研究问题"]
        for template in selected_templates[:2]:
            query = template.format(query=original_query, year=year)
            key = _normalize_query_text(query)
            if key in seen:
                continue
            seen.add(key)
            score = 0.0 if severity == "critical" else 0.5
            followups.append(
                {
                    "query": query,
                    "agent": _fallback_followup_agent(dimension, score),
                    "targets_gap": dimension,
                }
            )
            if len(followups) >= max(1, max_queries):
                return followups
    return followups


def _normalize_coverage_evaluation(
    payload: Dict[str, Any],
    *,
    fallback: Dict[str, Any],
    coverage_units: Optional[Sequence[Dict[str, Any]]] = None,
    loop_number: int,
    max_loops: int,
    prev_coverage_score: float,
    min_gain: float,
    original_query: str,
    previous_queries: Sequence[str],
    max_followup_queries: int,
) -> Dict[str, Any]:
    payload = _as_dict(payload)
    coverage_units = [unit for unit in list(coverage_units or []) if isinstance(unit, dict)]
    unit_titles = [str(unit.get("unit_title") or unit.get("unit_id") or "") for unit in coverage_units]
    fallback_scores = _as_dict(fallback.get("chapter_scores")) or _as_dict(fallback.get("dimension_scores"))
    product_gap_dimensions = {
        str(item.get("dimension") or "")
        for item in _as_list(fallback.get("knowledge_gaps"))
        if isinstance(item, dict) and item.get("gap_key")
    }
    raw_scores = _as_dict(payload.get("chapter_scores")) or _as_dict(payload.get("dimension_scores")) or fallback_scores
    dimension_scores: Dict[str, Dict[str, Any]] = {}
    score_units = unit_titles or INDUSTRY_DIMENSIONS
    for dimension in score_units:
        raw_item = _as_dict(raw_scores.get(dimension))
        raw_score = raw_item.get("score", fallback_scores.get(dimension, {}).get("score", 0.0))
        score = _safe_float(raw_score, 0.0)
        if dimension in product_gap_dimensions:
            score = min(score, _safe_float(_as_dict(fallback_scores.get(dimension)).get("score"), score))
        if score not in {0.0, 0.5, 1.0}:
            score = 1.0 if score >= 0.75 else 0.5 if score >= 0.25 else 0.0
        dimension_scores[dimension] = {
            "score": score,
            "reason": str(raw_item.get("reason") or fallback_scores.get(dimension, {}).get("reason") or "").strip(),
        }

    coverage_score = round(sum(item["score"] for item in dimension_scores.values()) / max(len(score_units), 1), 4)
    knowledge_gaps = [item for item in _as_list(payload.get("knowledge_gaps")) if isinstance(item, dict)]
    if not knowledge_gaps:
        knowledge_gaps = list(fallback.get("knowledge_gaps") or [])
    else:
        existing_gap_keys = {
            (
                str(item.get("dimension") or ""),
                str(item.get("gap_key") or ""),
                str(item.get("reason") or ""),
            )
            for item in knowledge_gaps
            if isinstance(item, dict)
        }
        for fallback_gap in _as_list(fallback.get("knowledge_gaps")):
            if not isinstance(fallback_gap, dict):
                continue
            key = (
                str(fallback_gap.get("dimension") or ""),
                str(fallback_gap.get("gap_key") or ""),
                str(fallback_gap.get("reason") or ""),
            )
            if key in existing_gap_keys:
                continue
            if fallback_gap.get("gap_key"):
                knowledge_gaps.append(dict(fallback_gap))
                existing_gap_keys.add(key)
    normalized_gaps: List[Dict[str, str]] = []
    for item in knowledge_gaps:
        dimension = str(item.get("dimension") or "").strip()
        if dimension not in score_units:
            continue
        score = dimension_scores[dimension]["score"]
        severity = str(item.get("severity") or "").strip()
        if severity not in {"critical", "moderate", "minor"}:
            severity = "critical" if score == 0.0 else "moderate" if score == 0.5 else "minor"
        if score < 1.0:
            normalized_gap = {
                "dimension": dimension,
                "reason": _compact_text(item.get("reason") or dimension_scores[dimension]["reason"], max_chars=160),
                "severity": severity,
            }
            if _as_list(item.get("missing")):
                normalized_gap["missing"] = _as_list(item.get("missing"))
            gap_key = str(item.get("gap_key") or "").strip()
            if gap_key:
                normalized_gap["gap_key"] = gap_key
            normalized_gaps.append(normalized_gap)

    coverage_target = _supervisor_coverage_target()
    if coverage_score >= coverage_target:
        is_sufficient = True
        stop_reason = "sufficient"
    elif loop_number >= max_loops:
        is_sufficient = False
        stop_reason = "max_loops_reached_with_gaps"
    elif loop_number > 1 and coverage_score - prev_coverage_score <= min_gain:
        is_sufficient = coverage_score >= coverage_target
        stop_reason = "coverage_stalled" if is_sufficient else "coverage_stalled_with_gaps"
    elif normalized_gaps and all(gap["severity"] == "minor" for gap in normalized_gaps):
        is_sufficient = True
        stop_reason = "sufficient"
    else:
        is_sufficient = False
        stop_reason = None

    followups = [item for item in _as_list(payload.get("follow_up_queries")) if isinstance(item, dict)]
    normalized_followups: List[Dict[str, str]] = []
    seen = {_normalize_query_text(query) for query in previous_queries}
    for item in followups:
        if len(normalized_followups) >= max(1, max_followup_queries):
            break
        query = str(item.get("query") or "").strip()
        agent = str(item.get("agent") or "").strip().lower()
        target = str(item.get("targets_gap") or "").strip() or "综合研究问题"
        if not query or agent not in {"rag", "iqs", "both", "all", *IQS_ROLE_ORDER}:
            continue
        key = _normalize_query_text(query)
        if key in seen:
            continue
        seen.add(key)
        normalized_followups.append(
            {
                "query": query,
                "agent": agent,
                "targets_gap": target,
                "gap_id": item.get("gap_id") or _followup_gap_id(item),
                "chapter_id": item.get("chapter_id"),
                "chapter_title": item.get("chapter_title"),
                "chapter_question": item.get("chapter_question"),
                "proof_role": item.get("proof_role"),
                "lane_targets": _as_list(item.get("lane_targets")),
            }
        )
    if not is_sufficient and not normalized_followups:
        normalized_followups = _fallback_followup_queries(
            original_query=original_query,
            gaps=normalized_gaps,
            previous_queries=previous_queries,
            max_queries=max_followup_queries,
            coverage_units=coverage_units,
        )

    return {
        "coverage_score": coverage_score,
        "dimension_scores": dimension_scores,
        "chapter_scores": dimension_scores,
        "is_sufficient": bool(is_sufficient),
        "stop_reason": stop_reason,
        "knowledge_gaps": normalized_gaps,
        "follow_up_queries": [] if is_sufficient else normalized_followups,
    }


def evaluate_coverage_fallback(
    *,
    original_query: str,
    evidence_pool: Sequence[Dict[str, Any]],
    coverage_units: Optional[Sequence[Dict[str, Any]]] = None,
    loop_number: int,
    max_loops: int,
    prev_coverage_score: float,
    min_gain: float,
    previous_queries: Sequence[str],
    max_followup_queries: int,
) -> Dict[str, Any]:
    dimension_scores: Dict[str, Dict[str, Any]] = {}
    gaps: List[Dict[str, str]] = []
    units = [unit for unit in list(coverage_units or []) if isinstance(unit, dict)]
    if units:
        for unit in units:
            dimension = str(unit.get("unit_title") or unit.get("unit_id") or "").strip()
            profile = _unit_coverage_score_from_pool(unit, evidence_pool)
            score = _safe_float(profile.get("score"), 0.0)
            reason = str(profile.get("reason") or "")
            dimension_scores[dimension] = {**profile, "score": score, "reason": reason}
            if score < 1.0:
                gaps.append(
                    {
                        "dimension": dimension,
                        "chapter_id": str(unit.get("unit_id") or ""),
                        "reason": reason,
                        "severity": "critical" if score < 0.5 else "moderate",
                        "missing": _as_list(profile.get("missing")),
                    }
                )
    else:
        for dimension in INDUSTRY_DIMENSIONS:
            score, reason = _dimension_score_from_pool(dimension, evidence_pool)
            dimension_scores[dimension] = {"score": score, "reason": reason}
            if score < 1.0:
                gaps.append(
                    {
                        "dimension": dimension,
                        "reason": reason,
                        "severity": "critical" if score == 0.0 else "moderate",
                    }
                )
    for gap in _product_commercial_gaps_from_pool(original_query, evidence_pool):
        dimension = str(gap.get("dimension") or "")
        if dimension in dimension_scores:
            dimension_scores[dimension]["score"] = min(_safe_float(dimension_scores[dimension].get("score"), 0.0), 0.5)
            base_reason = str(dimension_scores[dimension].get("reason") or "")
            dimension_scores[dimension]["reason"] = _compact_text(
                f"{base_reason}；{gap.get('reason')}" if base_reason else gap.get("reason"),
                max_chars=220,
            )
        gaps.append(gap)
    fallback = {
        "coverage_score": round(sum(item["score"] for item in dimension_scores.values()) / max(len(dimension_scores), 1), 4),
        "dimension_scores": dimension_scores,
        "chapter_scores": dimension_scores,
        "knowledge_gaps": gaps,
        "follow_up_queries": [],
    }
    return _normalize_coverage_evaluation(
        fallback,
        fallback=fallback,
        coverage_units=coverage_units,
        loop_number=loop_number,
        max_loops=max_loops,
        prev_coverage_score=prev_coverage_score,
        min_gain=min_gain,
        original_query=original_query,
        previous_queries=previous_queries,
        max_followup_queries=max_followup_queries,
    )


def evaluate_coverage_with_llm(
    *,
    original_query: str,
    evidence_pool_summary: str,
    coverage_units: Optional[Sequence[Dict[str, Any]]] = None,
    loop_number: int,
    max_loops: int,
    prev_coverage_score: float,
    min_gain: float,
    fallback: Dict[str, Any],
    previous_queries: Sequence[str],
    max_followup_queries: int,
) -> Dict[str, Any]:
    llm_config = build_llm_config("coverage_eval")
    if not llm_config_is_ready(llm_config):
        raise RuntimeError("Supervisor 覆盖率评估的大模型配置不完整。")
    user_payload = {
        "original_query": original_query,
        "loop_number": loop_number,
        "max_loops": max_loops,
        "evidence_pool_summary": evidence_pool_summary,
        "coverage_units": list(coverage_units or []),
        "prev_coverage_score": prev_coverage_score,
    }
    response = call_openai_compatible_json(
        config=llm_config,
        system_prompt=SUPERVISOR_COVERAGE_SYSTEM_PROMPT,
        user_payload=user_payload,
    )
    payload = _as_dict(response.get("payload"))
    if not payload:
        raise RuntimeError("Supervisor 覆盖率评估为空。")
    evaluation = _normalize_coverage_evaluation(
        payload,
        fallback=fallback,
        coverage_units=coverage_units,
        loop_number=loop_number,
        max_loops=max_loops,
        prev_coverage_score=prev_coverage_score,
        min_gain=min_gain,
        original_query=original_query,
        previous_queries=previous_queries,
        max_followup_queries=max_followup_queries,
    )
    evaluation["llm_call"] = _as_dict(response.get("llm_call"))
    evaluation["llm_degraded"] = False
    return evaluation


def _child_output_to_pool_item(
    *,
    round_number: int,
    agent: str,
    query: str,
    targets_gap: str,
    child: Dict[str, Any],
    search_task: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if agent == "rag":
        child_agent = "industry_rag_agent"
    elif agent in IQS_ROLE_CONFIGS:
        child_agent = IQS_ROLE_CONFIGS[agent]["child"]
    else:
        child_agent = "web_analysis_agent"
    task = _as_dict(search_task)
    item = {
        "round": round_number,
        "agent": agent,
        "child_agent": child_agent,
        "query": query,
        "targets_gap": targets_gap,
        "status": str(child.get("status") or "failed"),
        "confidence": _clip_confidence(child.get("confidence"), 0.0),
        "answer": str(child.get("answer") or "").strip(),
        "key_sources": list(child.get("key_sources") or []),
        "source_candidates": list(child.get("source_candidates") or []),
        "limitations": _as_dict(child.get("limitations")),
        "note": str(child.get("note") or "").strip(),
        "raw_data_points": list(child.get("raw_data_points") or []),
        "data_gap": list(child.get("data_gap") or []),
        "evidence_origin": "local_rag" if agent == "rag" else "live_search",
        "live_verified": bool(agent != "rag" and str(child.get("status") or "").strip().lower() in {"success", "partial"}),
    }
    if task:
        item.update(
            {
                "task_id": task.get("task_id"),
                "gap_id": task.get("gap_id") or _followup_gap_id(task),
                "requirement_id": task.get("requirement_id") or task.get("evidence_requirement_id"),
                "dimension_id": task.get("dimension_id"),
                "dimension_name": task.get("dimension_name"),
                "chapter_id": task.get("chapter_id"),
                "chapter_title": task.get("chapter_title"),
                "chapter_question": task.get("chapter_question"),
                "evidence_goal": task.get("evidence_goal"),
                "evidence_goal_id": task.get("evidence_goal_id"),
                "must_have_terms": _as_list(task.get("must_have_terms")),
                "forbidden_terms": _as_list(task.get("forbidden_terms")),
                "source_priority": _as_list(task.get("source_priority")),
                "proof_role": task.get("proof_role"),
                "required_fields": _as_list(task.get("required_fields")),
                "blocking_gaps": _as_list(task.get("blocking_gaps")),
                "retrieval_mode": task.get("retrieval_mode"),
                "retrieval_reason": task.get("retrieval_reason"),
                "primary_provider": task.get("primary_provider"),
                "fallback_providers": _as_list(task.get("fallback_providers")),
                "provider": task.get("provider"),
                "repair_source": task.get("repair_source"),
                "cache_seed_available": bool(task.get("cache_seed_available")),
                "live_refresh_required": bool(task.get("live_refresh_required")),
                "search_task": copy.deepcopy(task),
            }
        )
        for source in item["key_sources"]:
            if isinstance(source, dict):
                source.setdefault("evidence_origin", item["evidence_origin"])
                source.setdefault("live_verified", item["live_verified"])
                source.setdefault("cache_seed", False)
                source.setdefault("provider", task.get("provider") or task.get("primary_provider") or agent)
                source.setdefault("retrieval_mode", task.get("retrieval_mode"))
                if task.get("repair_source"):
                    source.setdefault("repair_source", task.get("repair_source"))
        for source in item["source_candidates"]:
            if isinstance(source, dict):
                source.setdefault("candidate_only", True)
                source.setdefault("evidence_origin", item["evidence_origin"])
                source.setdefault("live_verified", False)
                source.setdefault("provider", task.get("provider") or task.get("primary_provider") or agent)
                source.setdefault("retrieval_mode", task.get("retrieval_mode"))
                if task.get("repair_source"):
                    source.setdefault("repair_source", task.get("repair_source"))
        for point in item["raw_data_points"]:
            if isinstance(point, dict):
                point.setdefault("evidence_origin", item["evidence_origin"])
                point.setdefault("live_verified", item["live_verified"])
                point.setdefault("cache_seed", False)
                point.setdefault("provider", task.get("provider") or task.get("primary_provider") or agent)
                point.setdefault("retrieval_mode", task.get("retrieval_mode"))
                if task.get("repair_source"):
                    point.setdefault("repair_source", task.get("repair_source"))
                point.setdefault("task_id", task.get("task_id"))
                point.setdefault("dimension_id", task.get("dimension_id"))
                point.setdefault("dimension_name", task.get("dimension_name"))
                point.setdefault("chapter_id", task.get("chapter_id"))
                point.setdefault("chapter_title", task.get("chapter_title"))
                point.setdefault("chapter_question", task.get("chapter_question"))
                point.setdefault("evidence_goal", task.get("evidence_goal"))
                point.setdefault("evidence_goal_id", task.get("evidence_goal_id"))
                point.setdefault("must_have_terms", _as_list(task.get("must_have_terms")))
                point.setdefault("forbidden_terms", _as_list(task.get("forbidden_terms")))
                point.setdefault("source_priority", _as_list(task.get("source_priority")))
                point.setdefault("search_task", copy.deepcopy(task))
    deep_search = _as_dict(child.get("deep_search"))
    if deep_search:
        item["deep_search"] = deep_search
    elif task.get("prefer_deep"):
        item["deep_search"] = {
            "prefer_deep": True,
            "deep_reason": task.get("deep_reason"),
            "deep_status": task.get("deep_status"),
        }
    return item


def _summarize_deep_search_trace(search_trace: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    traces = [_as_dict(item) for item in list(search_trace or []) if isinstance(item, dict)]
    deep_traces = [
        item
        for item in traces
        if item.get("prefer_deep")
        or str(item.get("primary_engine") or item.get("engineType") or _as_dict(item.get("primary_options")).get("engineType") or "").strip() == "Deep"
    ]
    if not deep_traces:
        return {}
    return {
        "prefer_deep": True,
        "deep_reason": next((item.get("deep_reason") for item in deep_traces if item.get("deep_reason")), ""),
        "deep_task_count": len(deep_traces),
        "deep_signal_count": len([item for item in deep_traces if int(_safe_float(item.get("primary_count"), 0.0)) > 0]),
        "deep_fallback_used_count": len([item for item in deep_traces if item.get("fallback_used")]),
        "deep_unavailable": any(bool(item.get("deep_unavailable")) for item in deep_traces),
        "deep_exhausted": any(bool(item.get("deep_exhausted")) for item in deep_traces),
        "fallback_chain": next((_as_list(item.get("fallback_chain")) for item in deep_traces if _as_list(item.get("fallback_chain"))), []),
    }


def _state_metadata(state: BrainAgentState) -> Dict[str, Any]:
    metadata = state.get("metadata")
    if isinstance(metadata, dict):
        return metadata
    metadata = {}
    state["metadata"] = metadata
    return metadata


def _child_agent_for_followup_agent(agent: str) -> str:
    if agent == "rag":
        return "industry_rag_agent"
    return IQS_ROLE_CONFIGS.get(agent, {}).get("child", "web_analysis_agent")


def _run_single_followup(
    *,
    agent: str,
    query: str,
    targets_gap: str,
    round_number: int,
    state: BrainAgentState,
    search_task: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if agent == "rag" and not _local_rag_enabled():
        child = {
            "status": "skipped",
            "confidence": 0.0,
            "answer": "",
            "key_sources": [],
            "limitations": {"failure_reason": "Local RAG is disabled for the main flow."},
            "note": "Local RAG follow-up skipped.",
            "raw_data_points": [],
            "data_gap": [],
        }
        return _child_output_to_pool_item(
            round_number=round_number,
            agent=agent,
            query=query,
            targets_gap=targets_gap,
            child=child,
            search_task=search_task,
        )
    if agent == "rag":
        try:
            local_state = run_rag_agent(
                query,
                session_id=str(state.get("session_id") or "").strip(),
                args_overrides=dict(state.get("args_overrides") or {}),
            )
            child_errors = [f"本地 RAG 子智能体：{item}" for item in (local_state.get("errors") or [])]
            child = normalize_rag_child_output(local_state, route="local", errors=child_errors)
        except Exception as exc:
            logger.exception("RAG followup failed", extra={"query": query, "round": round_number})
            child = normalize_rag_child_output(None, route="local", errors=[f"本地 RAG 补充检索失败：{exc}"])
    elif agent in IQS_ROLE_CONFIGS:
        config = IQS_ROLE_CONFIGS[agent]
        try:
            web_state = run_web_analysis_agent(
                query,
                search_options=_search_options_for_task(state, _as_dict(search_task) or {"query": query, "agent": agent, "dimension_name": targets_gap, "evidence_goal": targets_gap}, "followup"),
                enable_llm_analysis=bool(state.get("enable_web_analysis", _env_flag("BRAIN_WEB_ENABLE_LLM_ANALYSIS", True))),
            )
            child_errors = [f"{config['label']}：{item}" for item in (web_state.get("errors") or [])]
            child = normalize_iqs_role_child_output(agent, web_state, route="web", errors=child_errors)
            child["deep_search"] = _summarize_deep_search_trace(_as_list(_as_dict(web_state.get("raw_output")).get("search_trace")))
        except Exception as exc:
            logger.exception("IQS role followup failed", extra={"query": query, "agent": agent, "round": round_number})
            child = normalize_iqs_role_child_output(agent, None, route="web", errors=[f"{config['label']}补充检索失败：{exc}"])
    else:
        try:
            web_state = run_web_analysis_agent(
                query,
                search_options=_search_options_for_task(state, _as_dict(search_task) or {"query": query, "agent": agent, "dimension_name": targets_gap, "evidence_goal": targets_gap}, "followup"),
                enable_llm_analysis=bool(state.get("enable_web_analysis", _env_flag("BRAIN_WEB_ENABLE_LLM_ANALYSIS", True))),
            )
            child_errors = [f"联网分析子智能体：{item}" for item in (web_state.get("errors") or [])]
            child = normalize_web_child_output(web_state, route="web", errors=child_errors)
            child["deep_search"] = _summarize_deep_search_trace(_as_list(_as_dict(web_state.get("raw_output")).get("search_trace")))
        except Exception as exc:
            logger.exception("Web followup failed", extra={"query": query, "round": round_number})
            child = normalize_web_child_output(None, route="web", errors=[f"联网分析补充检索失败：{exc}"])
    return _child_output_to_pool_item(
        round_number=round_number,
        agent=agent,
        query=query,
        targets_gap=targets_gap,
        child=child,
        search_task=search_task,
    )


def run_followup_queries(
    *,
    follow_up_queries: Sequence[Dict[str, Any]],
    round_number: int,
    state: BrainAgentState,
) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    local_rag_enabled = _local_rag_enabled()
    valid_agents = {"iqs", "both", "all", *IQS_ROLE_ORDER}
    if local_rag_enabled:
        valid_agents.add("rag")
    for item in follow_up_queries:
        if not isinstance(item, dict):
            continue
        if _is_non_evidence_followup(_as_dict(item)):
            continue
        query = str(item.get("query") or "").strip()
        agent = str(item.get("agent") or "").strip().lower()
        targets_gap = str(item.get("targets_gap") or "").strip()
        if agent == "rag" and not local_rag_enabled:
            agent = "iqs"
        if not query or agent not in valid_agents:
            continue
        lane_agents = []
        for lane in _as_list(item.get("lane_targets")):
            lane_key = str(lane or "").strip().lower()
            role_key = IQS_LANE_TO_ROLE.get(lane_key) or (lane_key if lane_key in IQS_ROLE_CONFIGS else "")
            if role_key and role_key not in lane_agents:
                lane_agents.append(role_key)
        if agent in {"iqs", "both", "all"} and targets_gap in INDUSTRY_DIMENSIONS and _env_flag("BRAIN_DIMENSIONAL_IQS_FOLLOWUP", True):
            agent = _fallback_followup_agent(targets_gap, 0.0)
        if agent == "iqs" and lane_agents:
            agents = lane_agents
        elif agent in {"both", "all"}:
            agents = ([ "rag" ] if local_rag_enabled else []) + (lane_agents or IQS_ROLE_ORDER)
        else:
            agents = [agent]
        for routed_agent in agents:
            search_task = normalize_search_task(
                {
                    **dict(item),
                    "gap_id": item.get("gap_id") or _followup_gap_id(item),
                    "agent": routed_agent,
                    "query": query,
                    "dimension_name": item.get("dimension_name") or item.get("dimension") or targets_gap,
                    "evidence_goal": item.get("evidence_goal") or targets_gap,
                },
                fallback_index=len(tasks) + 1,
            )
            lane_type = _lane_type_for_role(routed_agent)
            if lane_type:
                search_task["scheduled_lane"] = routed_agent
                search_task["scheduled_lane_type"] = lane_type
                search_task["lane_targets"] = [lane_type]
                search_task = _apply_retrieval_routing_to_task(search_task, lane_type=lane_type)
            tasks.append({"query": query, "agent": routed_agent, "targets_gap": targets_gap, "search_task": search_task})
    tasks = _dedupe_followup_tasks(tasks)
    tasks = _select_high_value_repair_tasks(tasks, state=state, round_number=round_number)
    tasks = _apply_lane_circuit_breaker_to_tasks(tasks, state)
    if tasks and not _strict_quality_mode():
        if _continuous_evidence_loop_mode():
            max_tasks = _env_int(
                "BRAIN_CONTINUOUS_FOLLOWUP_MAX_TASKS_PER_ROUND",
                _env_int("BRAIN_FOLLOWUP_MAX_TASKS_PER_ROUND", 0),
            )
        else:
            max_tasks = _env_int("BRAIN_FOLLOWUP_MAX_TASKS_PER_ROUND", 0)
        if max_tasks > 0 and len(tasks) > max_tasks:
            before_count = len(tasks)
            tasks = sorted(
                tasks,
                key=lambda task: (
                    _followup_priority(_as_dict(task.get("search_task"))),
                    _followup_target_key(_as_dict(task.get("search_task"))),
                    str(task.get("agent") or ""),
                ),
            )[:max_tasks]
            _progress("followup", "补证任务裁剪", before=before_count, after=len(tasks), limit=max_tasks)
    tasks = _filter_exhausted_gap_tasks(tasks, state=state)
    _record_gap_attempts(tasks, state=state, round_number=round_number)
    cache_results, tasks, cache_only_skipped = _apply_evidence_cache_to_followup_tasks(tasks, state=state, round_number=round_number)
    tasks = _apply_deep_repair_policy_to_tasks(tasks, state=state, round_number=round_number)
    live_refresh_attempted_count = len(
        [
            task
            for task in tasks
            if _as_dict(task.get("search_task")).get("cache_seed_available")
            and _as_dict(task.get("search_task")).get("live_refresh_required")
        ]
    )
    if live_refresh_attempted_count or cache_only_skipped:
        metadata = dict(state.get("metadata") or {})
        summary = dict(metadata.get("evidence_cache_summary") or {})
        summary["live_refresh_attempted_count"] = int(_safe_float(summary.get("live_refresh_attempted_count"), 0.0)) + live_refresh_attempted_count
        metadata["evidence_cache_summary"] = summary
        state["metadata"] = metadata
    if not tasks:
        return cache_results

    started = time.perf_counter()
    max_workers = max(1, min(_env_int("BRAIN_FOLLOWUP_PARALLEL_WORKERS", 4), len(tasks)))
    deep_count = len([task for task in tasks if _as_dict(task.get("search_task")).get("prefer_deep")])
    _progress("followup", "补证任务开始", tasks=len(tasks), deep=deep_count, workers=max_workers, round=round_number)
    results: List[Dict[str, Any]] = []
    task_timeout = max(0.0, _env_float("BRAIN_FOLLOWUP_TASK_TIMEOUT_SECONDS", 180.0))
    executor = ThreadPoolExecutor(max_workers=max_workers)
    future_map = {
        executor.submit(
            _run_single_followup,
            agent=task["agent"],
            query=task["query"],
            targets_gap=task["targets_gap"],
            round_number=round_number,
            state=state,
            search_task=_as_dict(task.get("search_task")),
        ): task
        for task in tasks
    }
    completed: set[Any] = set()
    try:
        for future in as_completed(future_map, timeout=task_timeout or None):
            completed.add(future)
            try:
                results.append(future.result())
            except Exception as exc:
                task = future_map[future]
                results.append(
                    {
                        "round": round_number,
                        "agent": task["agent"],
                        "child_agent": _child_agent_for_followup_agent(task["agent"]),
                        "query": task["query"],
                        "targets_gap": task["targets_gap"],
                        "status": "failed",
                        "confidence": 0.0,
                        "answer": "",
                        "key_sources": [],
                        "limitations": {"failure_reason": str(exc)},
                        "note": f"补充检索执行失败：{exc}",
                        "search_task": _as_dict(task.get("search_task")),
                    }
                )
    except FutureTimeoutError:
        for future, task in future_map.items():
            if future in completed:
                continue
            future.cancel()
            results.append(
                {
                    "round": round_number,
                    "agent": task["agent"],
                    "child_agent": _child_agent_for_followup_agent(task["agent"]),
                    "query": task["query"],
                    "targets_gap": task["targets_gap"],
                    "status": "failed",
                    "confidence": 0.0,
                    "answer": "",
                    "key_sources": [],
                    "limitations": {"failure_reason": f"timeout after {task_timeout:.0f}s"},
                    "note": f"补充检索执行超时：{task_timeout:.0f}s",
                    "search_task": _as_dict(task.get("search_task")),
                }
            )
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    _record_deep_repair_unavailable_from_results(state, results)
    combined_results = [*cache_results, *results]
    _record_gap_attempt_results(combined_results, state=state)
    live_refresh_signal_count = len(
        [
            item
            for item in results
            if _as_dict(item.get("search_task")).get("cache_seed_available")
            and _as_dict(item.get("search_task")).get("live_refresh_required")
            and _followup_result_signal_score(_as_dict(item)) >= 2
        ]
    )
    if live_refresh_attempted_count or live_refresh_signal_count:
        metadata = dict(state.get("metadata") or {})
        summary = dict(metadata.get("evidence_cache_summary") or {})
        summary["live_refresh_signal_count"] = int(_safe_float(summary.get("live_refresh_signal_count"), 0.0)) + live_refresh_signal_count
        cache_seed_draft_count = max(0, live_refresh_attempted_count - live_refresh_signal_count)
        summary["cache_seed_used_for_draft_count"] = int(_safe_float(summary.get("cache_seed_used_for_draft_count"), 0.0)) + cache_seed_draft_count
        summary["cache_live_refresh_miss_count"] = int(_safe_float(summary.get("cache_live_refresh_miss_count"), 0.0)) + cache_seed_draft_count
        metadata["evidence_cache_summary"] = summary
        state["metadata"] = metadata
    success_count = len([item for item in combined_results if str(item.get("status") or "") in {"success", "partial"}])
    _progress(
        "followup",
        "补证任务结束",
        success=success_count,
        total=len(combined_results),
        cache_hits=len(cache_results),
        cache_only_skipped=len(cache_only_skipped),
        live_refresh_attempted=live_refresh_attempted_count,
        live_refresh_signal=live_refresh_signal_count,
        elapsed=f"{time.perf_counter() - started:.1f}s",
    )
    return combined_results


def _layout_followup_queries_from_writer_report(writer_report: Dict[str, Any], *, max_queries: int) -> List[Dict[str, Any]]:
    layout_plan = _as_dict(writer_report.get("layout_plan"))
    queries: List[Dict[str, Any]] = []

    def add_query(item: Dict[str, Any]) -> None:
        payload = _normalize_followup_payload(item)
        if not payload or _is_non_evidence_followup(payload):
            return
        queries.append(payload)

    for item in _as_list(layout_plan.get("global_follow_up_queries")):
        if isinstance(item, dict):
            add_query(item)
    refinement_plan = _as_dict(writer_report.get("evidence_refinement_plan"))
    for item in _as_list(refinement_plan.get("top_priorities")) + _as_list(refinement_plan.get("follow_up_queries")):
        if isinstance(item, dict):
            add_query(item)
    reflection = _research_reflection_memo_from_writer_report(writer_report)
    for item in _as_list(reflection.get("next_search_task_seeds")):
        if isinstance(item, dict):
            add_query({**item, "source": item.get("source") or "research_reflection_memo"})
    for item in _as_list(writer_report.get("required_followups")):
        item = _as_dict(item)
        payload = _normalize_followup_payload(item)
        if not payload:
            continue
        payload.setdefault("agent", "iqs")
        payload.setdefault("targets_gap", item.get("hypothesis_statement") or item.get("type"))
        payload.setdefault("evidence_goal", item.get("hypothesis_statement") or item.get("type"))
        payload.setdefault("dimension_name", item.get("hypothesis_statement") or item.get("type"))
        add_query(payload)
    qa_required = _as_list(_as_dict(_as_dict(writer_report.get("qa_result")).get("deep_evaluation")).get("required_followups"))
    for item in qa_required:
        item = _as_dict(item)
        payload = _normalize_followup_payload(item)
        if not payload:
            continue
        payload.setdefault("agent", "iqs")
        payload.setdefault("targets_gap", item.get("hypothesis_statement") or item.get("type"))
        payload.setdefault("evidence_goal", item.get("hypothesis_statement") or item.get("type"))
        payload.setdefault("dimension_name", item.get("hypothesis_statement") or item.get("type"))
        add_query(payload)
    if not queries:
        for gap in _as_list(layout_plan.get("layout_gaps")):
            gap_item = _as_dict(gap)
            missing = " ".join(str(value) for value in _as_list(gap_item.get("missing")) if str(value).strip())
            dimension = str(gap_item.get("dimension") or "").strip()
            if not missing and not dimension:
                continue
            queries.append(
                {
                    "query": f"{dimension} {missing} 行业数据 证据",
                    "agent": _fallback_followup_agent(dimension, 0.0),
                    "targets_gap": dimension,
                }
            )
    queries.sort(
        key=lambda item: (
            _followup_priority(item),
            _followup_target_key(item),
            str(item.get("query") or ""),
        )
    )
    normalized: List[Dict[str, Any]] = []
    seen = set()
    target_counts: Dict[str, int] = {}
    per_target_limit = max(1, _env_int("BRAIN_FOLLOWUP_MAX_PER_TARGET", 2))
    valid_agents = {"rag", "iqs", "both", "all", *IQS_ROLE_ORDER}
    for item in queries:
        query = _compact_text(item.get("query"), max_chars=220)
        agent = str(item.get("agent") or "").strip().lower()
        targets_gap = _compact_text(item.get("targets_gap"), max_chars=80)
        if not query:
            continue
        if agent not in valid_agents:
            agent = _fallback_followup_agent(targets_gap, 0.0)
        key = (query, agent, targets_gap)
        if key in seen:
            continue
        seen.add(key)
        target_key = _followup_target_key(item)
        if target_counts.get(target_key, 0) >= per_target_limit:
            continue
        target_counts[target_key] = target_counts.get(target_key, 0) + 1
        normalized.append(
            {
                "query": query,
                "agent": agent,
                "targets_gap": targets_gap,
                "gap_id": item.get("gap_id") or _followup_gap_id(item),
                "dimension_name": item.get("dimension_name") or item.get("dimension") or targets_gap,
                "evidence_goal": item.get("evidence_goal") or item.get("reason") or targets_gap,
                "source_priority": _as_list(item.get("source_priority")),
                "lane_targets": _as_list(item.get("lane_targets")),
                "blocking_gaps": _as_list(item.get("blocking_gaps")),
                "hypothesis_id": item.get("hypothesis_id"),
                "hypothesis_statement": item.get("hypothesis_statement"),
                "proof_profile_id": item.get("proof_profile_id"),
                "mandatory_proof_id": item.get("mandatory_proof_id") or item.get("proof_id"),
                "requirement_id": item.get("requirement_id"),
                "chapter_id": item.get("chapter_id"),
                "section_id": item.get("section_id"),
                "gap_type": item.get("gap_type") or item.get("type"),
                "repair_status": item.get("repair_status"),
                "missing_mandatory_proofs": _as_list(item.get("missing_mandatory_proofs")),
                "proof_role": item.get("proof_role"),
                "proof_standard": item.get("proof_standard"),
                "evidence_type": item.get("evidence_type"),
                "required_evidence_mix": _as_list(item.get("required_evidence_mix")),
                "required_fields": _as_list(item.get("required_fields")),
                "required_source_level": _as_list(item.get("required_source_level")),
                "success_criteria": item.get("success_criteria"),
                "reject_if": _as_list(item.get("reject_if")),
                "preferred_source_patterns": _as_list(item.get("preferred_source_patterns")),
                "freshness_required": bool(item.get("freshness_required")),
                "max_cache_age_hours": item.get("max_cache_age_hours"),
                "source_stage": item.get("source_stage"),
                "source": item.get("source"),
                "allowed_for_writing": bool(item.get("allowed_for_writing", False)),
                "live_refresh_required": bool(item.get("live_refresh_required")),
                "avoid_repeating_failed_query": bool(item.get("avoid_repeating_failed_query")),
                "failed_queries": _as_list(item.get("failed_queries") or item.get("avoid_queries")),
                "counter_evidence": item.get("counter_evidence"),
            }
        )
        if len(normalized) >= max(1, max_queries):
            break
    return normalized


def _attach_report_plan(evidence_package: Dict[str, Any], structured_analysis: Dict[str, Any], report_plan: Dict[str, Any]) -> None:
    if not report_plan:
        return
    evidence_package["report_plan"] = report_plan
    evidence_package.setdefault("metadata", {})
    evidence_package["metadata"]["report_plan"] = report_plan
    structured_analysis["report_plan"] = report_plan


def _research_plan_from_state(state: BrainAgentState) -> Dict[str, Any]:
    return _as_dict(state.get("research_plan")) or _as_dict(_as_dict(state.get("query_analysis")).get("research_plan"))


def _attach_research_plan(evidence_package: Dict[str, Any], structured_analysis: Dict[str, Any], research_plan: Dict[str, Any]) -> None:
    if not research_plan:
        return
    evidence_package["research_plan"] = research_plan
    evidence_package.setdefault("metadata", {})
    evidence_package["metadata"]["research_plan"] = research_plan
    structured_analysis["research_plan"] = research_plan


def _sync_analysis_repair_priorities_to_evidence_package(
    evidence_package: Dict[str, Any],
    structured_analysis: Dict[str, Any],
) -> Dict[str, Any]:
    """Expose analysis-generated repair gaps where repair loops already read.

    Analysis validators can create claim repair priorities after the evidence
    package has been built. Post-QA repair and evidence preflight primarily read
    ``evidence_package.evidence_gap_ledger``, so mirror these analysis gaps back
    into the package before writer/repair stages run.
    """

    package = _as_dict(evidence_package)
    analysis = _as_dict(structured_analysis)
    existing_ledger = [item for item in _as_list(package.get("evidence_gap_ledger")) if isinstance(item, dict)]
    existing_repair = [item for item in _as_list(package.get("evidence_repair_priorities")) if isinstance(item, dict)]
    seen = {
        str(item.get("gap_id") or item.get("id") or item.get("claim_id") or "").strip()
        for item in [*existing_ledger, *existing_repair]
        if str(item.get("gap_id") or item.get("id") or item.get("claim_id") or "").strip()
    }
    added = 0

    candidates: List[Dict[str, Any]] = []
    for item in [
        *_as_list(analysis.get("evidence_repair_priorities")),
        *_as_list(analysis.get("claim_repair_priorities")),
        *_as_list(analysis.get("evidence_gap_ledger")),
        *_as_list(_as_dict(analysis.get("llm_analysis_synthesis")).get("evidence_repair_priorities")),
        *_as_list(_as_dict(_as_dict(analysis.get("llm_analysis_synthesis")).get("validation")).get("claim_repair_priorities")),
    ]:
        if isinstance(item, dict):
            candidates.append(item)

    for candidate in candidates:
        gap_key = str(candidate.get("gap_id") or candidate.get("id") or candidate.get("claim_id") or "").strip()
        if gap_key and gap_key in seen:
            continue
        copied = dict(candidate)
        copied.setdefault("repair_route", "evidence_search")
        copied.setdefault("source_stage", copied.get("source_stage") or "analysis_claim_support")
        copied.setdefault("status", "open")
        copied.setdefault("allowed_for_writing", False)
        copied.setdefault("source", "structured_analysis_repair_priority")
        if not str(copied.get("gap_id") or "").strip():
            copied["gap_id"] = _stable_short_hash(
                "analysis_repair_gap",
                copied.get("claim_id"),
                copied.get("gap_type"),
                copied.get("type"),
            )
            gap_key = str(copied["gap_id"])
        existing_ledger.append(copied)
        if str(copied.get("schema_version") or "") == "claim_support_repair_priority_v1":
            existing_repair.append(copied)
        seen.add(str(gap_key or copied.get("gap_id") or copied.get("claim_id") or ""))
        added += 1

    if added:
        package["evidence_gap_ledger"] = existing_ledger
        package["evidence_repair_priorities"] = existing_repair
        package.setdefault("metadata", {})["analysis_repair_sync"] = {
            "added_gap_count": added,
            "evidence_gap_ledger_count": len(existing_ledger),
            "evidence_repair_priority_count": len(existing_repair),
        }
    return {
        "added_gap_count": added,
        "evidence_gap_ledger_count": len(existing_ledger),
        "evidence_repair_priority_count": len(existing_repair),
    }


def _lane_coverage_from_state(state: BrainAgentState) -> Dict[str, Any]:
    coverage: Dict[str, Any] = {}
    query_analysis = _as_dict(state.get("query_analysis"))
    schedule = _as_dict(state.get("search_task_schedule")) or _as_dict(query_analysis.get("search_task_schedule"))
    agent_tasks = _as_dict(query_analysis.get("agent_tasks"))
    scheduled_tasks = [task for task in _as_list(schedule.get("scheduled_tasks")) if isinstance(task, dict)]
    dropped_tasks = [task for task in _as_list(schedule.get("dropped_tasks")) if isinstance(task, dict)]
    for role_key in IQS_ROLE_ORDER:
        config = IQS_ROLE_CONFIGS[role_key]
        lane_type = _lane_type_for_role(role_key)
        planned_tasks = [task for task in _as_list(agent_tasks.get(role_key)) if isinstance(task, dict)]
        if not planned_tasks:
            planned_tasks = [
                task
                for task in scheduled_tasks
                if str(task.get("scheduled_lane") or "").strip().lower() == role_key
                or str(task.get("scheduled_lane_type") or "").strip().lower() == lane_type
            ]
        planned_count = len(planned_tasks)
        role_state = _as_dict(state.get(config["state"]))  # type: ignore[literal-required]
        raw_output = _as_dict(role_state.get("raw_output"))
        lane = _as_dict(raw_output.get("lane_coverage")) or _as_dict(_as_dict(role_state.get("metadata")).get("lane_coverage"))
        if lane:
            execution_status = str(lane.get("execution_status") or lane.get("status") or "").strip() or "completed"
            coverage[role_key] = {
                **lane,
                "planned_task_count": int(_safe_float(lane.get("planned_task_count"), planned_count)),
                "scheduled": int(_safe_float(lane.get("scheduled"), planned_count)),
                "failed_task_count": int(_safe_float(lane.get("failed_task_count") or lane.get("failed"), 0.0)),
                "execution_status": execution_status,
                "status": lane.get("status") or execution_status,
            }
        else:
            missing_status = "missing_state" if planned_count else "missing"
            coverage[role_key] = {
                "status": missing_status,
                "execution_status": missing_status,
                "planned_task_count": planned_count,
                "scheduled": planned_count,
                "succeeded": 0,
                "failed": 0,
                "failed_task_count": 0,
                "completed_task_count": 0,
                "timed_out_task_count": 0,
                "cancelled_task_count": 0,
                "raw_data_points": 0,
                "search_results": 0,
                "page_results": 0,
                "key_sources": 0,
                "scheduled_lane_type": lane_type,
            }
        lane_dropped = [
            task
            for task in dropped_tasks
            if str(task.get("scheduled_lane_type") or task.get("dropped_lane_type") or "").strip().lower() == lane_type
            or str(task.get("dropped_lane") or "").strip().lower() == role_key
        ]
        coverage[role_key]["dropped_task_count"] = len(lane_dropped)
        coverage[role_key]["dropped_blocking_count"] = _blocking_dropped_task_count(lane_dropped)
        coverage[role_key].setdefault("readpage_attempted", coverage[role_key].get("page_results", 0))
        coverage[role_key].setdefault("readpage_succeeded", coverage[role_key].get("page_results", 0))
        coverage[role_key].setdefault("true_a_source_count", 0)
        coverage[role_key].setdefault("core_ab_source_count", coverage[role_key].get("ab_source_count", 0))
    return coverage


def _lane_health_summary_from_coverage(lane_coverage: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for role_key, raw in _as_dict(lane_coverage).items():
        lane = _as_dict(raw)
        scheduled = int(_safe_float(lane.get("scheduled"), 0.0))
        timed_out = int(_safe_float(lane.get("timed_out_task_count"), 0.0))
        usable = int(_safe_float(lane.get("usable_source_count") or lane.get("key_sources"), 0.0))
        timeout_rate = round(timed_out / max(scheduled, 1), 4) if scheduled else 0.0
        degraded = bool((timed_out >= 2 and usable <= 0) or (scheduled > 0 and timeout_rate >= 0.6))
        lane_type = _lane_type_for_role(role_key)
        fallback = {
            "filing_company": ["official_data"],
            "market_research": ["official_data"],
            "news_event": ["market_research"],
            "customer_case": [],
            "technology_product": ["official_data", "market_research"],
            "official_data": [],
        }.get(lane_type, [])
        summary[role_key] = {
            "lane_type": lane_type,
            "status": "degraded" if degraded else str(lane.get("status") or "completed"),
            "disabled_for_low_priority": degraded,
            "disabled_reason": "timeout_exhausted" if degraded else "",
            "fallback_lanes": fallback if degraded else [],
            "scheduled": scheduled,
            "completed": int(_safe_float(lane.get("completed_task_count"), 0.0)),
            "timed_out": timed_out,
            "cancelled": int(_safe_float(lane.get("cancelled_task_count"), 0.0)),
            "usable_source_count": usable,
            "ab_source_count": int(_safe_float(lane.get("ab_source_count"), 0.0)),
            "core_ab_source_count": int(_safe_float(lane.get("core_ab_source_count") or lane.get("ab_source_count"), 0.0)),
            "true_a_source_count": int(_safe_float(lane.get("true_a_source_count"), 0.0)),
            "valid_metric_count": int(_safe_float(lane.get("valid_metric_count"), 0.0)),
            "readpage_attempted": int(_safe_float(lane.get("readpage_attempted"), 0.0)),
            "readpage_succeeded": int(_safe_float(lane.get("readpage_succeeded"), 0.0)),
            "dropped_blocking_count": int(_safe_float(lane.get("dropped_blocking_count"), 0.0)),
            "timeout_rate": timeout_rate,
            "early_stopped": bool(lane.get("early_stopped")),
            "early_stop_reason": lane.get("early_stop_reason") or "",
        }
    return summary


def _lane_health_summary_from_state(state: BrainAgentState) -> Dict[str, Any]:
    return _lane_health_summary_from_coverage(_lane_coverage_from_state(state))


def _retrieval_mode_for_summary(task: Dict[str, Any]) -> str:
    mode = _normalize_retrieval_mode(task.get("retrieval_mode"))
    if mode:
        return mode
    return "deep" if task.get("prefer_deep") else "normal"


def _provider_for_summary(task: Dict[str, Any]) -> str:
    provider = str(task.get("primary_provider") or task.get("provider") or "").strip().lower()
    if provider:
        return provider
    return "iqs_deep" if _retrieval_mode_for_summary(task) in {"deep", "hybrid"} else "iqs_normal"


def _scheduled_tasks_from_state_for_summary(state: BrainAgentState) -> List[Dict[str, Any]]:
    query_analysis = _as_dict(state.get("query_analysis"))
    schedule = _as_dict(state.get("search_task_schedule")) or _as_dict(query_analysis.get("search_task_schedule"))
    tasks = [dict(task) for task in _as_list(schedule.get("scheduled_tasks")) if isinstance(task, dict)]
    if tasks:
        return tasks
    agent_tasks = _as_dict(query_analysis.get("agent_tasks"))
    for role_key, lane_tasks in agent_tasks.items():
        if role_key not in IQS_ROLE_CONFIGS:
            continue
        lane_type = _lane_type_for_role(role_key)
        for task in _as_list(lane_tasks):
            if isinstance(task, dict):
                tasks.append({**task, "scheduled_lane": role_key, "scheduled_lane_type": task.get("scheduled_lane_type") or lane_type})
    return tasks


def _increment_count(mapping: Dict[str, int], key: Any, amount: int = 1) -> None:
    name = str(key or "unknown").strip().lower() or "unknown"
    mapping[name] = int(mapping.get(name, 0)) + int(amount)


def _retrieval_strategy_summary_from_state(
    state: BrainAgentState,
    *,
    lane_coverage: Dict[str, Any],
    package_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    scheduled_tasks = _scheduled_tasks_from_state_for_summary(state)
    by_lane: Dict[str, Dict[str, Any]] = {}
    by_mode: Dict[str, int] = {}
    by_provider: Dict[str, int] = {}
    totals = {
        "scheduled_task_count": 0,
        "deep_task_count": 0,
        "normal_task_count": 0,
        "hybrid_task_count": 0,
        "success_count": 0,
        "failed_task_count": 0,
        "timed_out_task_count": 0,
        "fallback_count": 0,
        "ab_source_count": 0,
    }

    for task in scheduled_tasks:
        mode = _retrieval_mode_for_summary(task)
        provider = _provider_for_summary(task)
        lane_type = str(task.get("scheduled_lane_type") or _lane_type_for_role(str(task.get("scheduled_lane") or "")) or "unassigned").strip().lower()
        lane_summary = by_lane.setdefault(
            lane_type,
            {
                "scheduled_task_count": 0,
                "deep_task_count": 0,
                "normal_task_count": 0,
                "hybrid_task_count": 0,
                "success_count": 0,
                "failed_task_count": 0,
                "timed_out_task_count": 0,
                "fallback_count": 0,
                "ab_source_count": 0,
            },
        )
        lane_summary["scheduled_task_count"] += 1
        totals["scheduled_task_count"] += 1
        if mode in {"deep", "normal", "hybrid"}:
            lane_summary[f"{mode}_task_count"] += 1
            totals[f"{mode}_task_count"] += 1
        _increment_count(by_mode, mode)
        _increment_count(by_provider, provider)
        if _as_list(task.get("fallback_providers")):
            lane_summary["fallback_count"] += 1
            totals["fallback_count"] += 1

    for lane_key, coverage in _as_dict(lane_coverage).items():
        lane = _as_dict(coverage)
        lane_type = str(lane.get("scheduled_lane_type") or _lane_type_for_role(str(lane_key)) or lane_key or "unknown").strip().lower()
        lane_summary = by_lane.setdefault(
            lane_type,
            {
                "scheduled_task_count": 0,
                "deep_task_count": 0,
                "normal_task_count": 0,
                "hybrid_task_count": 0,
                "success_count": 0,
                "failed_task_count": 0,
                "timed_out_task_count": 0,
                "fallback_count": 0,
                "ab_source_count": 0,
            },
        )
        success = int(_safe_float(lane.get("succeeded") or lane.get("completed_task_count") or lane.get("success_count"), 0.0))
        failed = int(_safe_float(lane.get("failed_task_count") or lane.get("failed"), 0.0))
        timed_out = int(_safe_float(lane.get("timed_out_task_count") or lane.get("timed_out"), 0.0))
        ab_sources = int(_safe_float(lane.get("ab_source_count"), 0.0))
        lane_summary["success_count"] = max(int(lane_summary.get("success_count") or 0), success)
        lane_summary["failed_task_count"] = max(int(lane_summary.get("failed_task_count") or 0), failed)
        lane_summary["timed_out_task_count"] = max(int(lane_summary.get("timed_out_task_count") or 0), timed_out)
        lane_summary["ab_source_count"] = max(int(lane_summary.get("ab_source_count") or 0), ab_sources)

    for lane_summary in by_lane.values():
        totals["success_count"] += int(lane_summary.get("success_count") or 0)
        totals["failed_task_count"] += int(lane_summary.get("failed_task_count") or 0)
        totals["timed_out_task_count"] += int(lane_summary.get("timed_out_task_count") or 0)
        totals["ab_source_count"] += int(lane_summary.get("ab_source_count") or 0)

    package_summary = _as_dict(package_summary)
    source_distribution = _as_dict(package_summary.get("source_level_distribution"))
    package_ab = int(_safe_float(source_distribution.get("A"), 0.0)) + int(_safe_float(source_distribution.get("B"), 0.0))
    if package_ab:
        totals["ab_source_count"] = max(totals["ab_source_count"], package_ab)

    metadata = _as_dict(state.get("metadata"))

    return {
        "policy": "iqs_deep_core_iqs_normal_breadth",
        "scheduled_by_mode": by_mode,
        "scheduled_by_provider": by_provider,
        "totals": totals,
        "by_lane": by_lane,
        "iqs_deep_repair_summary": _as_dict(metadata.get("iqs_deep_repair")),
    }


def _annotate_evidence_package_runtime(
    evidence_package: Dict[str, Any],
    *,
    lane_coverage: Dict[str, Any],
    state: BrainAgentState,
) -> Dict[str, Any]:
    if not isinstance(evidence_package, dict):
        return evidence_package
    summary = dict(_as_dict(evidence_package.get("summary")))
    metadata = _as_dict(state.get("metadata"))
    readpage_attempted = sum(
        int(_safe_float(_as_dict(lane).get("readpage_attempted") or _as_dict(lane).get("page_results"), 0.0))
        for lane in _as_dict(lane_coverage).values()
    )
    readpage_succeeded = sum(
        int(_safe_float(_as_dict(lane).get("readpage_succeeded") or _as_dict(lane).get("page_results"), 0.0))
        for lane in _as_dict(lane_coverage).values()
    )
    true_a_source_count = sum(
        int(_safe_float(_as_dict(lane).get("true_a_source_count"), 0.0))
        for lane in _as_dict(lane_coverage).values()
    )
    retrieval_strategy_summary = _retrieval_strategy_summary_from_state(
        state,
        lane_coverage=lane_coverage,
        package_summary=summary,
    )
    health_summary = dict(
        _as_dict(evidence_package.get("evidence_health_summary"))
        or _as_dict(summary.get("evidence_health_summary"))
        or _as_dict(_as_dict(evidence_package.get("metadata")).get("evidence_health_summary"))
    )
    runtime_issues: List[Dict[str, Any]] = []
    for lane_name, lane_payload in _as_dict(lane_coverage).items():
        lane = _as_dict(lane_payload)
        scheduled = int(_safe_float(lane.get("scheduled") or lane.get("scheduled_count") or lane.get("planned_task_count"), 0.0))
        timed_out = int(_safe_float(lane.get("timed_out_task_count") or lane.get("timed_out"), 0.0))
        usable = int(_safe_float(lane.get("usable_source_count") or lane.get("key_sources") or lane.get("raw_data_points"), 0.0))
        status = str(lane.get("execution_status") or "").strip()
        if scheduled > 0 and timed_out > 0 and usable <= 0:
            runtime_issues.append(
                {
                    "type": "lane_timeout",
                    "root_cause": "lane_timeout",
                    "lane": lane_name,
                    "scheduled": scheduled,
                    "timed_out_task_count": timed_out,
                }
            )
        if scheduled > 0 and status == "missing_state":
            runtime_issues.append(
                {
                    "type": "missing_lane_state",
                    "root_cause": "missing_lane_state",
                    "lane": lane_name,
                    "scheduled": scheduled,
                }
            )
    if readpage_attempted > 0 and readpage_succeeded <= 0:
        runtime_issues.append(
            {
                "type": "readpage_evidence_missing",
                "root_cause": "readpage_failed",
                "readpage_coverage": {"attempted": readpage_attempted, "succeeded": readpage_succeeded},
            }
        )
    if runtime_issues:
        existing_ledger = [item for item in _as_list(evidence_package.get("evidence_gap_ledger")) if isinstance(item, dict)]
        seen_runtime_keys = {
            (
                str(item.get("source") or ""),
                str(item.get("root_cause") or item.get("type") or ""),
                str(item.get("lane") or ""),
            )
            for item in existing_ledger
        }
        for issue in runtime_issues:
            key = ("runtime_lane_coverage", str(issue.get("root_cause") or issue.get("type") or ""), str(issue.get("lane") or ""))
            if key in seen_runtime_keys:
                continue
            existing_ledger.append(
                {
                    "gap_id": _stable_short_hash("runtime_evidence_gap", issue.get("root_cause"), issue.get("lane")),
                    "chapter_id": "runtime",
                    "gap_type": issue.get("type"),
                    "type": issue.get("type"),
                    "severity": "blocking",
                    "root_cause": issue.get("root_cause") or issue.get("type"),
                    "repair_priority": "high",
                    "can_iqs_repair": issue.get("root_cause") != "missing_lane_state",
                    "required_proof_role": "source_check",
                    "required_fields": ["source"],
                    "lane_targets": [issue.get("lane")] if issue.get("lane") else [],
                    "why_current_evidence_insufficient": "Runtime retrieval did not produce enough verifiable evidence.",
                    "source": "runtime_lane_coverage",
                    **issue,
                }
            )
        evidence_package["evidence_gap_ledger"] = existing_ledger
        summary["evidence_gap_count"] = len(existing_ledger)
        summary["blocking_evidence_gap_count"] = len([item for item in existing_ledger if str(_as_dict(item).get("severity") or "") == "blocking"])
    summary["readpage_coverage"] = {
        **_as_dict(summary.get("readpage_coverage")),
        "attempted": readpage_attempted,
        "succeeded": readpage_succeeded,
    }
    summary["true_a_source_count"] = true_a_source_count
    summary["retrieval_strategy_summary"] = retrieval_strategy_summary
    health_summary.update(
        {
            "readpage_attempted": readpage_attempted,
            "readpage_succeeded": readpage_succeeded,
            "lane_timeout_issue_count": len([item for item in runtime_issues if item.get("root_cause") == "lane_timeout"]),
            "missing_lane_state_issue_count": len([item for item in runtime_issues if item.get("root_cause") == "missing_lane_state"]),
            "runtime_evidence_issues": runtime_issues,
            "blocking_gap_count": summary.get("blocking_evidence_gap_count", health_summary.get("blocking_gap_count", 0)),
        }
    )
    gate = dict(_as_dict(summary.get("publishable_evidence_gate")))
    if gate:
        reasons = [
            reason
            for reason in _as_list(gate.get("blocking_reasons"))
            if not (
                _as_dict(reason).get("type") == "readpage_evidence_missing"
                and readpage_succeeded > 0
            )
        ]
        if gate.get("require_readpage") and readpage_succeeded <= 0 and not any(_as_dict(reason).get("type") == "readpage_evidence_missing" for reason in reasons):
            reasons.append({"type": "readpage_evidence_missing", "readpage_coverage": summary["readpage_coverage"]})
        gate["blocking_reasons"] = reasons
        gate["passed"] = not reasons
        summary["publishable_evidence_gate"] = gate
    delivery_gate = dict(_as_dict(summary.get("delivery_gate")))
    source_distribution = _as_dict(summary.get("source_level_distribution"))
    ab_source_count = int(_safe_float(source_distribution.get("A"), 0.0)) + int(_safe_float(source_distribution.get("B"), 0.0))
    analysis_ready_count = int(_safe_float(summary.get("analysis_ready_count"), 0.0))
    evidence_count = int(_safe_float(summary.get("evidence_count"), 0.0))
    gap_attempt_summary = _as_dict(metadata.get("gap_attempt_summary"))
    exhausted = bool(gap_attempt_summary.get("evidence_exhausted"))
    scheduled_lanes = [
        lane
        for lane in _as_dict(lane_coverage).values()
        if int(_safe_float(_as_dict(lane).get("scheduled") or _as_dict(lane).get("scheduled_count"), 0.0)) > 0
    ]
    all_page_zero = bool(
        scheduled_lanes
        and all(int(_safe_float(_as_dict(lane).get("page_results") or _as_dict(lane).get("page_result_count"), 0.0)) <= 0 for lane in scheduled_lanes)
    )
    publishable = bool(gate.get("passed"))
    severe_reasons: List[Dict[str, Any]] = []
    if evidence_count <= 0:
        severe_reasons.append({"type": "no_evidence"})
    if analysis_ready_count <= 0 and ab_source_count <= 0 and int(summary["readpage_coverage"].get("succeeded") or 0) <= 0:
        severe_reasons.append({"type": "no_usable_evidence_signal"})
    if all_page_zero and analysis_ready_count <= 0:
        severe_reasons.append({"type": "page_results_all_zero"})
    if publishable:
        tier = "publishable_clean"
    elif severe_reasons:
        tier = "diagnostic_only"
    else:
        tier = "limited_review_draft"
    delivery_gate.update(
        {
            "policy": delivery_gate.get("policy") or "three_tier",
            "tier": tier,
            "publishable": tier == "publishable_clean",
            "draft_allowed": tier in {"publishable_clean", "limited_review_draft"},
            "diagnostic_only": tier == "diagnostic_only",
            "exhausted": exhausted,
            "blocking_reasons": severe_reasons if tier == "diagnostic_only" else _as_list(delivery_gate.get("blocking_reasons")),
            "review_reasons": [] if tier == "publishable_clean" else _as_list(gate.get("blocking_reasons")) or _as_list(delivery_gate.get("review_reasons")),
            "gap_attempt_summary": gap_attempt_summary,
        }
    )
    summary["delivery_gate"] = delivery_gate
    summary["delivery_tier"] = tier
    health_summary["delivery_tier"] = tier
    health_summary["delivery_publishable"] = tier == "publishable_clean"
    health_summary["publishable_evidence_gate_passed"] = bool(gate.get("passed"))
    preflight = dict(
        _as_dict(evidence_package.get("evidence_preflight_summary"))
        or _as_dict(summary.get("evidence_preflight_summary"))
        or _as_dict(metadata.get("evidence_preflight_summary"))
    )
    if preflight:
        blocking = list(_as_list(preflight.get("clean_blocking_reasons")))
        for issue in runtime_issues:
            blocking.append(
                {
                    "type": issue.get("type") or issue.get("root_cause"),
                    "root_cause": issue.get("root_cause") or issue.get("type"),
                    "chapter_id": issue.get("chapter_id") or "runtime",
                    "proof_role": "source_check",
                    "repairable_by_search": issue.get("root_cause") != "missing_lane_state",
                    **issue,
                }
            )
        preflight.update(
            {
                "ready_for_clean_writer": tier == "publishable_clean",
                "needs_gap_repair": bool([item for item in blocking if bool(_as_dict(item).get("repairable_by_search"))]) and tier != "publishable_clean",
                "review_draft_allowed": tier in {"publishable_clean", "limited_review_draft"},
                "diagnostic_only": tier == "diagnostic_only",
                "clean_blocking_reasons": blocking,
                "gap_attempt_summary": gap_attempt_summary,
            }
        )
        summary["evidence_preflight_summary"] = preflight
        health_summary["evidence_preflight"] = {
            "ready_for_clean_writer": bool(preflight.get("ready_for_clean_writer")),
            "needs_gap_repair": bool(preflight.get("needs_gap_repair")),
            "review_draft_allowed": bool(preflight.get("review_draft_allowed")),
            "diagnostic_only": bool(preflight.get("diagnostic_only")),
        }
    summary["evidence_health_summary"] = health_summary
    evidence_package["summary"] = summary
    evidence_package["evidence_health_summary"] = health_summary
    if preflight:
        evidence_package["evidence_preflight_summary"] = preflight
    evidence_package.setdefault("metadata", {})
    evidence_package["metadata"]["readpage_coverage"] = summary["readpage_coverage"]
    evidence_package["metadata"]["retrieval_strategy_summary"] = retrieval_strategy_summary
    evidence_package["metadata"]["delivery_gate"] = delivery_gate
    evidence_package["metadata"]["evidence_health_summary"] = health_summary
    if preflight:
        evidence_package["metadata"]["evidence_preflight_summary"] = preflight
    return evidence_package


def _task_is_core_evidence_repair(task: Dict[str, Any]) -> bool:
    payload = _as_dict(task)
    markers = _followup_marker_values(payload)
    role = str(payload.get("proof_role") or payload.get("evidence_type") or "").strip().lower()
    decision_use = str(payload.get("decision_use") or "").strip().lower()
    if markers.intersection(DEEP_REPAIR_BLOCKING_GAPS | CORE_CACHE_REFRESH_BLOCKING_GAPS):
        return True
    if role in {"metric", "source_check", "filing", "official_data"}:
        return True
    return decision_use in {"core_claim", "decision_ready", "investment_or_market_entry"}


def _apply_lane_circuit_breaker_to_tasks(tasks: Sequence[Dict[str, Any]], state: BrainAgentState) -> List[Dict[str, Any]]:
    if not tasks or not _env_flag("BRAIN_IQS_LANE_CIRCUIT_BREAKER_ENABLED", True):
        return [dict(task) for task in list(tasks or []) if isinstance(task, dict)]
    health = _lane_health_summary_from_state(state)
    if not health:
        return [dict(task) for task in list(tasks or []) if isinstance(task, dict)]
    updated: List[Dict[str, Any]] = []
    skipped = 0
    rerouted = 0
    for raw in list(tasks or []):
        task = dict(_as_dict(raw))
        agent = str(task.get("agent") or "").strip().lower()
        if agent not in IQS_ROLE_CONFIGS:
            updated.append(task)
            continue
        lane_state = _as_dict(health.get(agent))
        if not lane_state.get("disabled_for_low_priority"):
            updated.append(task)
            continue
        search_task = dict(_as_dict(task.get("search_task")))
        core_task = _task_is_core_evidence_repair(search_task)
        fallback_roles = [
            _role_for_lane_type_safe(lane)
            for lane in _as_list(lane_state.get("fallback_lanes"))
            if _role_for_lane_type_safe(lane)
        ]
        if core_task and fallback_roles:
            fallback_role = fallback_roles[0]
            search_task["lane_circuit_breaker"] = {
                "from_agent": agent,
                "to_agent": fallback_role,
                "reason": lane_state.get("disabled_reason") or "timeout_exhausted",
            }
            search_task["scheduled_lane"] = fallback_role
            search_task["scheduled_lane_type"] = _lane_type_for_role(fallback_role)
            search_task["lane_targets"] = [_lane_type_for_role(fallback_role)]
            search_task = _apply_retrieval_routing_to_task(search_task, lane_type=str(search_task.get("scheduled_lane_type") or ""))
            task["agent"] = fallback_role
            task["search_task"] = search_task
            updated.append(task)
            rerouted += 1
        elif core_task:
            updated.append(task)
        else:
            skipped += 1
    if skipped or rerouted:
        metadata = _state_metadata(state)
        summary = dict(metadata.get("lane_circuit_breaker_summary") or {})
        summary["skipped_low_priority_count"] = int(_safe_float(summary.get("skipped_low_priority_count"), 0.0)) + skipped
        summary["rerouted_core_task_count"] = int(_safe_float(summary.get("rerouted_core_task_count"), 0.0)) + rerouted
        metadata["lane_circuit_breaker_summary"] = summary
    return updated


def _attach_lane_health_to_writer_report(
    writer_report: Dict[str, Any],
    *,
    lane_coverage: Dict[str, Any],
    state: Optional[BrainAgentState] = None,
) -> Dict[str, Any]:
    copied = dict(_as_dict(writer_report))
    lane_health = _lane_health_summary_from_coverage(lane_coverage)
    copied["lane_health_summary"] = lane_health
    if state is not None:
        metadata = _as_dict(state.get("metadata"))
        if metadata.get("lane_circuit_breaker_summary"):
            copied["lane_circuit_breaker_summary"] = metadata.get("lane_circuit_breaker_summary")
        if metadata.get("repair_task_selection_summary"):
            copied["repair_task_selection_summary"] = metadata.get("repair_task_selection_summary")
    return copied


_FACT_EXTRACTOR_SUM_KEYS = (
    "attempted",
    "success_count",
    "fact_card_count",
    "rejected_span_count",
    "invalid_metric_count",
    "cache_hit_count",
    "llm_error_count",
    "regex_fallback_point_count",
    "budget_used",
)
_FACT_EXTRACTOR_BOOL_KEYS = (
    "regex_fallback_used",
    "fallback_used",
    "budget_exhausted",
    "extractor_empty_without_regex_points",
)


def _looks_like_fact_extractor_diagnostics(payload: Any) -> bool:
    payload = _as_dict(payload)
    if not payload:
        return False
    return any(key in payload for key in (*_FACT_EXTRACTOR_SUM_KEYS, *_FACT_EXTRACTOR_BOOL_KEYS, "budget_limit"))


def _fact_extractor_payload_from_container(container: Any) -> Dict[str, Any]:
    payload = _as_dict(container)
    if not payload:
        return {}
    candidates = (
        payload,
        payload.get("fact_extractor"),
        payload.get("readpage_fact_extractor"),
        _as_dict(payload.get("metadata")).get("readpage_fact_extractor"),
        _as_dict(payload.get("metadata")).get("fact_extractor"),
        _as_dict(payload.get("raw_output")).get("fact_extractor"),
        _as_dict(payload.get("raw_output")).get("readpage_fact_extractor"),
        _as_dict(_as_dict(payload.get("raw_output")).get("metadata")).get("readpage_fact_extractor"),
        _as_dict(_as_dict(payload.get("raw_output")).get("metadata")).get("fact_extractor"),
        _as_dict(payload.get("limitations")).get("fact_extractor"),
        _as_dict(payload.get("limitations")).get("readpage_fact_extractor"),
    )
    for candidate in candidates:
        candidate_dict = _as_dict(candidate)
        if _looks_like_fact_extractor_diagnostics(candidate_dict):
            return candidate_dict
    return {}


def _aggregate_readpage_fact_extractor_diagnostics(
    children: Optional[Dict[str, Any]] = None,
    *,
    extra_payloads: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """Aggregate one diagnostics payload per child/container.

    The score report intentionally reads fixed diagnostics paths only. This
    helper closes the production gap by lifting per-child extractor telemetry
    into the evidence package / writer report without recursively counting
    chapter-level fact-card summaries.
    """

    payloads: List[Dict[str, Any]] = []
    for child in _as_dict(children).values():
        candidate = _fact_extractor_payload_from_container(child)
        if candidate:
            payloads.append(candidate)
    for payload in _as_list(extra_payloads):
        candidate = _fact_extractor_payload_from_container(payload)
        if candidate:
            payloads.append(candidate)
    if not payloads:
        return {}

    totals: Dict[str, Any] = {
        "attempted": 0,
        "success_count": 0,
        "fact_card_count": 0,
        "rejected_span_count": 0,
        "invalid_metric_count": 0,
        "cache_hit_count": 0,
        "llm_error_count": 0,
        "regex_fallback_point_count": 0,
        "budget_used": 0,
        "budget_limit": 0,
        "regex_fallback_used": False,
        "fallback_used": False,
        "budget_exhausted": False,
        "extractor_empty_without_regex_points": False,
        "statuses": [],
        "models": [],
    }
    for payload in payloads:
        for key in _FACT_EXTRACTOR_SUM_KEYS:
            totals[key] = int(totals.get(key) or 0) + int(_safe_float(payload.get(key), 0.0))
        if payload.get("budget_limit") not in (None, ""):
            totals["budget_limit"] = max(int(totals.get("budget_limit") or 0), int(_safe_float(payload.get("budget_limit"), 0.0)))
        for key in _FACT_EXTRACTOR_BOOL_KEYS:
            totals[key] = bool(totals.get(key) or payload.get(key))
        status = str(payload.get("status") or "").strip()
        if status and status not in totals["statuses"]:
            totals["statuses"].append(status)
        model = str(payload.get("model") or "").strip()
        if model and model not in totals["models"]:
            totals["models"].append(model)
    return totals


def _attach_readpage_fact_extractor_diagnostics(
    evidence_package: Dict[str, Any],
    writer_report: Optional[Dict[str, Any]] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> None:
    diagnostics = _as_dict(diagnostics)
    if not diagnostics:
        return
    evidence_package.setdefault("metadata", {})["readpage_fact_extractor"] = dict(diagnostics)
    if writer_report is None:
        return
    writer_report["fact_extractor"] = dict(diagnostics)
    writer_report.setdefault("metadata", {})["readpage_fact_extractor"] = dict(diagnostics)
    writer_report.setdefault("render_artifacts", {}).setdefault("metadata", {})["readpage_fact_extractor"] = dict(diagnostics)


def _writer_pipeline_state_fields(writer_report: Dict[str, Any]) -> Dict[str, Any]:
    writer_report = _as_dict(writer_report)
    report_blueprint = _as_dict(writer_report.get("report_blueprint"))
    if not _brain_full_payloads():
        result = {
            "report_blueprint": report_blueprint,
            "search_tasks": _as_list(writer_report.get("search_tasks"))
            or _as_list(_as_dict(writer_report.get("report_plan")).get("search_tasks"))
            or _as_list(report_blueprint.get("search_tasks")),
            "layout_plan": _as_dict(writer_report.get("layout_plan")),
            "chapter_evidence_packages": _as_list(writer_report.get("chapter_evidence_packages")),
            "evidence_graph": _as_dict(writer_report.get("evidence_graph")),
            "micro_layouts": _as_list(writer_report.get("micro_layouts")),
            "table_packages": _as_list(writer_report.get("table_packages")),
            "argument_units": _as_list(writer_report.get("argument_units")),
            "chapter_packages": _as_list(writer_report.get("chapter_packages")),
            "render_artifacts": _as_dict(writer_report.get("render_artifacts")),
            "decision_package": _as_dict(writer_report.get("decision_package")),
            "risk_package": _as_dict(writer_report.get("risk_package")),
            "appendix_package": _as_dict(writer_report.get("appendix_package")),
            "qa_result": _as_dict(writer_report.get("qa_result")),
            "package_quality_report": _as_dict(writer_report.get("package_quality_report")),
            "pipeline_payload_mode": writer_report.get("pipeline_payload_mode") or "summary",
            "pipeline_artifact_summary": _as_dict(writer_report.get("pipeline_artifact_summary")),
        }
        schedule = _as_dict(writer_report.get("search_task_schedule"))
        if schedule:
            result["search_task_schedule"] = schedule
        lane_coverage = _as_dict(writer_report.get("lane_coverage"))
        if lane_coverage:
            result["lane_coverage"] = lane_coverage
        return result
    result = {
        "report_blueprint": report_blueprint,
        "search_tasks": _as_list(writer_report.get("search_tasks"))
        or _as_list(_as_dict(writer_report.get("report_plan")).get("search_tasks"))
        or _as_list(report_blueprint.get("search_tasks")),
        "layout_plan": _as_dict(writer_report.get("layout_plan")),
        "chapter_evidence_packages": _as_list(writer_report.get("chapter_evidence_packages")),
        "evidence_graph": _as_dict(writer_report.get("evidence_graph")),
        "micro_layouts": _as_list(writer_report.get("micro_layouts")),
        "table_packages": _as_list(writer_report.get("table_packages")),
        "argument_units": _as_list(writer_report.get("argument_units")),
        "chapter_packages": _as_list(writer_report.get("chapter_packages")),
        "render_artifacts": _as_dict(writer_report.get("render_artifacts")),
        "decision_package": _as_dict(writer_report.get("decision_package")),
        "risk_package": _as_dict(writer_report.get("risk_package")),
        "appendix_package": _as_dict(writer_report.get("appendix_package")),
        "qa_result": _as_dict(writer_report.get("qa_result")),
        "package_quality_report": _as_dict(writer_report.get("package_quality_report")),
    }
    schedule = _as_dict(writer_report.get("search_task_schedule"))
    if schedule:
        result["search_task_schedule"] = schedule
    lane_coverage = _as_dict(writer_report.get("lane_coverage"))
    if lane_coverage:
        result["lane_coverage"] = lane_coverage
    return result


def _writer_quality_snapshot(writer_report: Dict[str, Any]) -> Dict[str, Any]:
    validation = _as_dict(writer_report.get("validation"))
    layout_plan = _as_dict(writer_report.get("layout_plan"))
    preflight_plan = _as_dict(_as_dict(writer_report.get("reformatter_preflight")).get("repair_plan"))
    errors = _as_list(validation.get("errors"))
    warnings = _as_list(validation.get("warnings"))
    gaps = _as_list(layout_plan.get("layout_gaps"))
    required_followups = _as_list(writer_report.get("required_followups")) or _as_list(
        _as_dict(validation.get("deep_evaluation")).get("required_followups")
    )
    clean_repair_count = sum(
        [
            1 if writer_report.get("qa_pending_repair") else 0,
            1 if validation.get("repair_required") else 0,
            len(_as_list(validation.get("repair_followups"))),
            len(_as_list(validation.get("evidence_repair_followups"))),
            len(_as_list(validation.get("content_repair_followups"))),
            len(required_followups),
        ]
    )
    render_gate = _as_dict(_as_dict(writer_report.get("qa_result")).get("render_gate"))
    render_repair_count = len(_as_list(render_gate.get("blockers")))
    return {
        "status": str(writer_report.get("report_status") or ""),
        "passed": bool(validation.get("passed")),
        "quality_score": int(_safe_float(validation.get("quality_score"), 0.0)),
        "reformatter_preflight_status": str(preflight_plan.get("status") or ""),
        "pending_repair_count": render_repair_count,
        "clean_repair_count": clean_repair_count,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "layout_gap_count": len(gaps),
        "estimated_chars": int(_safe_float(writer_report.get("estimated_chars"), 0.0)),
    }


def _reformatter_preflight_rank(status: str) -> int:
    status = str(status or "").strip()
    if status == "passed":
        return 4
    if status == "needs_text_repair":
        return 3
    if not status:
        return 2
    if status == "needs_degrade_or_manual_review":
        return 1
    if status == "needs_evidence_refinement":
        return 0
    return 1


def _writer_quality_key(writer_report: Dict[str, Any]) -> tuple:
    snapshot = _writer_quality_snapshot(writer_report)
    return (
        1 if snapshot["status"] == "final" else 0,
        1 if snapshot["passed"] else 0,
        -int(snapshot.get("pending_repair_count") or 0),
        _reformatter_preflight_rank(str(snapshot.get("reformatter_preflight_status") or "")),
        snapshot["quality_score"],
        -snapshot["error_count"],
        -snapshot["layout_gap_count"],
        min(snapshot["estimated_chars"], 60000),
        -snapshot["warning_count"],
    )


def _followup_result_signal_score(item: Dict[str, Any]) -> int:
    if not isinstance(item, dict):
        return 0
    if str(item.get("status") or "").strip().lower() not in {"success", "partial"}:
        return 0
    key_sources = _as_list(item.get("key_sources"))
    raw_points = _as_list(item.get("raw_data_points"))
    answer = str(item.get("answer") or "").strip()
    weak_negative = bool(re.search(r"(没有找到|未找到|暂无|无相关|不足以|无法确认|未能确认)", answer))
    score = 0
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
    sourced_answer = bool(
        len(answer) >= 80
        and not weak_negative
        and (
            re.search(r"\[\d{1,4}\]|https?://|来源|公告|财报|年报|统计|协会|专利|标准|report|filing|source", answer, re.I)
            or re.search(r"\d+(?:\.\d+)?\s*(?:%|亿美元|亿元|万台|百万|million|bn|billion)", answer, re.I)
        )
    )
    if valid_sources:
        score += min(3, len(valid_sources)) + 1
    if valid_points:
        score += min(3, len(valid_points)) + 1
    if sourced_answer:
        score += 1
    return score


def _followup_result_has_signal(results: Sequence[Dict[str, Any]]) -> bool:
    for item in results:
        if _followup_result_signal_score(_as_dict(item)) >= 2:
            return True
    return False


def _followup_signal_diagnostics(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    signal_count = 0
    empty_success_count = 0
    failed_count = 0
    partial_count = 0
    no_signal_reasons: Dict[str, int] = {}
    for raw in list(results or []):
        item = _as_dict(raw)
        status = str(item.get("status") or "").strip().lower()
        score = _followup_result_signal_score(item)
        if score >= 2:
            signal_count += 1
            continue
        if status == "failed":
            failed_count += 1
            reason = str(_as_dict(item.get("limitations")).get("failure_reason") or item.get("note") or "failed")
            reason = _compact_text(reason, max_chars=100) or "failed"
            no_signal_reasons[reason] = no_signal_reasons.get(reason, 0) + 1
            continue
        if status == "partial":
            partial_count += 1
        if status in {"success", "partial"}:
            has_payload = bool(
                _as_list(item.get("key_sources"))
                or _as_list(item.get("raw_data_points"))
                or str(item.get("answer") or "").strip()
            )
            if not has_payload:
                empty_success_count += 1
                no_signal_reasons["empty_success_payload"] = no_signal_reasons.get("empty_success_payload", 0) + 1
            else:
                no_signal_reasons["weak_or_unsourced_payload"] = no_signal_reasons.get("weak_or_unsourced_payload", 0) + 1
        elif status:
            no_signal_reasons[status] = no_signal_reasons.get(status, 0) + 1
    return {
        "signal_count": signal_count,
        "empty_success_count": empty_success_count,
        "failed_count": failed_count,
        "partial_count": partial_count,
        "no_signal_reasons": [
            {"reason": reason, "count": count}
            for reason, count in sorted(no_signal_reasons.items(), key=lambda item: (-item[1], item[0]))[:12]
        ],
    }


def _substantive_followup_results(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Only results with real evidence signal should enter the analysis evidence pool."""
    usable: List[Dict[str, Any]] = []
    for raw in list(results or []):
        item = _as_dict(raw)
        if item and _followup_result_signal_score(item) >= 2:
            usable.append(item)
    return usable


def _source_level_from_source(source: Dict[str, Any]) -> str:
    text = " ".join(
        str(source.get(key) or "")
        for key in ("source_type", "publisher", "domain", "url", "title", "snippet")
    ).lower()
    if re.search(r"(caifuhao\.eastmoney|guba\.eastmoney|mguba\.eastmoney|baijiahao|toutiao|zhihu|xueqiu|weibo|sohu|book118|docin|doc88|wenku\.baidu)", text):
        return "D"
    if re.search(r"(view\.inews\.qq\.com|finance\.sina|news\.10jqka|eastmoney\.com|futunn\.com)", text):
        return "C"
    level = str(source.get("source_level") or source.get("level") or source.get("tier") or "").strip().upper()
    if level in {"A", "B", "C", "D"}:
        return level
    if re.search(r"(official|filing|company_announcement|annual report|10-k|10-q|sec\.gov|cninfo|sse\.com|szse|hkexnews|公告|年报|财报|监管|统计局)", text):
        return "A"
    if re.search(r"(research|association|technical_standard|patent|product_doc|company_official|研报|咨询|market report|counterpoint|idc|omdia|dscc|canalys|gartner|mckinsey|bcg|deloitte|pwc|协会|白皮书|标准|专利)", text):
        return "B"
    return ""


def _payload_source_key(source: Dict[str, Any]) -> str:
    return str(
        source.get("url")
        or source.get("source_url")
        or source.get("title")
        or source.get("publisher")
        or source.get("domain")
        or ""
    ).strip().lower()


def _lane_payload_signal_counts(payloads: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    source_keys: set[str] = set()
    ab_keys: set[str] = set()
    metric_count = 0
    page_count = 0
    true_a_keys: set[str] = set()
    completed_roles: set[str] = set()
    completed_chapters: set[str] = set()
    completed = 0
    timed_out = 0
    cancelled = 0
    failed = 0
    for raw in list(payloads or []):
        payload = _as_dict(raw)
        if not payload:
            continue
        status = str(payload.get("status") or "").strip().lower()
        errors = " ".join(str(item) for item in _as_list(payload.get("errors"))).lower()
        if status == "cancelled":
            cancelled += 1
            continue
        if "timed out" in errors or "timeout" in errors:
            timed_out += 1
            continue
        has_signal = bool(
            payload.get("query_result")
            or payload.get("answer_block")
            or payload.get("task_result")
            or payload.get("search_results")
            or payload.get("key_sources")
            or payload.get("raw_data_points")
            or payload.get("page_results")
        )
        if has_signal:
            completed += 1
            task_result = _as_dict(payload.get("task_result"))
            task = _as_dict(task_result.get("task")) or task_result
            if not task:
                query_result = _as_dict(payload.get("query_result"))
                task = _as_dict(query_result.get("task"))
            if not task:
                task = _as_dict(payload.get("search_task"))
            role = str(task.get("proof_role") or task.get("evidence_type") or "").strip().lower()
            chapter = str(task.get("chapter_id") or task.get("dimension_id") or "").strip()
            if role:
                completed_roles.add(role)
            if chapter:
                completed_chapters.add(chapter)
        elif errors:
            failed += 1
        page_items = [_as_dict(item) for item in _as_list(payload.get("page_results")) if isinstance(item, dict)]
        page_count += len(page_items)
        for source in _as_list(payload.get("key_sources")) + _as_list(payload.get("search_results")) + page_items:
            item = _as_dict(source)
            if not item:
                continue
            key = _payload_source_key(item)
            if not key:
                continue
            source_keys.add(key)
            level_value = _source_level_from_source(item)
            if level_value in {"A", "B"}:
                ab_keys.add(key)
            if level_value == "A":
                true_a_keys.add(key)
        for point in _as_list(payload.get("raw_data_points")):
            item = _as_dict(point)
            if not item:
                continue
            if (
                str(item.get("metric") or item.get("metric_name") or item.get("indicator") or "").strip()
                and str(item.get("value") or item.get("numeric_value") or "").strip()
                and str(item.get("source") or item.get("source_url") or item.get("url") or "").strip()
            ):
                metric_count += 1
    return {
        "completed_task_count": completed,
        "timed_out_task_count": timed_out,
        "cancelled_task_count": cancelled,
        "failed_task_count": failed,
        "usable_source_count": len(source_keys),
        "ab_source_count": len(ab_keys),
        "core_ab_source_count": len(ab_keys),
        "true_a_source_count": len(true_a_keys),
        "valid_metric_count": metric_count,
        "page_result_count": page_count,
        "completed_role_count": len(completed_roles),
        "completed_chapter_count": len(completed_chapters),
    }


def _lane_early_stop_decision(payloads: Sequence[Dict[str, Any]], *, started_at: float) -> Dict[str, Any]:
    counts = _lane_payload_signal_counts(payloads)
    elapsed = time.perf_counter() - started_at
    if not _env_flag("BRAIN_IQS_LANE_EARLY_STOP_ENABLED", True):
        return {**counts, "early_stopped": False, "early_stop_reason": ""}
    min_seconds = _env_int("BRAIN_IQS_LANE_EARLY_STOP_MIN_SECONDS", 45, min_value=0, max_value=3600)
    min_completed = _env_int("BRAIN_IQS_LANE_EARLY_STOP_MIN_COMPLETED_TASKS", 2, min_value=1, max_value=100)
    min_ab = _env_int("BRAIN_IQS_LANE_EARLY_STOP_MIN_AB_SOURCES", 2, min_value=1, max_value=100)
    min_sources = _env_int("BRAIN_IQS_LANE_EARLY_STOP_MIN_USABLE_SOURCES", 5, min_value=1, max_value=200)
    min_pages = _env_int("BRAIN_IQS_LANE_EARLY_STOP_MIN_PAGE_RESULTS", 1, min_value=0, max_value=100)
    min_roles = _env_int("BRAIN_IQS_LANE_EARLY_STOP_MIN_ROLES", 2, min_value=0, max_value=20)
    min_chapters = _env_int("BRAIN_IQS_LANE_EARLY_STOP_MIN_CHAPTERS", 2, min_value=0, max_value=50)
    if elapsed < min_seconds or counts["completed_task_count"] < min_completed:
        return {**counts, "early_stopped": False, "early_stop_reason": ""}
    if _env_flag("BRAIN_IQS_LANE_EARLY_STOP_REQUIRE_READPAGE", True) and counts["page_result_count"] < min_pages:
        return {**counts, "early_stopped": False, "early_stop_reason": ""}
    if counts["completed_role_count"] < min_roles or counts["completed_chapter_count"] < min_chapters:
        return {**counts, "early_stopped": False, "early_stop_reason": ""}
    reason = ""
    if counts["ab_source_count"] >= min_ab:
        reason = "enough_ab_sources"
    elif counts["ab_source_count"] >= 1 and counts["valid_metric_count"] >= 1:
        reason = "ab_source_and_metric_found"
    elif counts["usable_source_count"] >= min_sources:
        reason = "enough_usable_sources"
    return {**counts, "early_stopped": bool(reason), "early_stop_reason": reason}


def _followup_new_ab_source_count(results: Sequence[Dict[str, Any]]) -> int:
    count = 0
    seen: set[str] = set()
    for result in list(results or []):
        for raw_source in _as_list(_as_dict(result).get("key_sources")):
            source = _as_dict(raw_source)
            if not source:
                continue
            level = _source_level_from_source(source)
            if level not in {"A", "B"}:
                continue
            key = str(source.get("url") or source.get("source_url") or source.get("title") or source.get("publisher") or "").strip().lower()
            if not key:
                key = str(id(source))
            if key in seen:
                continue
            seen.add(key)
            count += 1
    return count


def _followup_new_metric_ready_count(results: Sequence[Dict[str, Any]]) -> int:
    count = 0
    seen: set[str] = set()
    for result in list(results or []):
        payload = _as_dict(result)
        result_sources = _as_list(payload.get("key_sources"))
        candidates: List[Dict[str, Any]] = []
        for key in ("raw_data_points", "data_points", "metrics", "metric_points"):
            candidates.extend([_as_dict(item) for item in _as_list(payload.get(key)) if isinstance(item, dict)])
        candidates.append(payload)
        for point in candidates:
            metric = str(point.get("metric") or point.get("metric_name") or point.get("name") or "").strip()
            value = str(point.get("value") or point.get("numeric_value") or point.get("amount") or "").strip()
            unit = str(point.get("unit") or point.get("numeric_unit") or "").strip()
            period = str(point.get("period") or point.get("date") or point.get("time_period") or "").strip()
            source = _as_dict(point.get("source"))
            has_source = bool(source.get("url") or source.get("title") or source.get("publisher") or result_sources)
            if not (metric and value and unit and period and has_source):
                continue
            key = re.sub(r"\s+", "", "|".join([metric, value, unit, period]).lower())[:180]
            if key in seen:
                continue
            seen.add(key)
            count += 1
    return count


def _followup_deep_trace(result: Dict[str, Any]) -> Dict[str, Any]:
    item = _as_dict(result)
    trace = _as_dict(item.get("deep_search"))
    if trace:
        return trace
    task = _as_dict(item.get("search_task"))
    return {
        "prefer_deep": bool(task.get("prefer_deep")),
        "deep_reason": task.get("deep_reason"),
        "deep_status": task.get("deep_status"),
    }


def _repair_result_summary(
    results: Sequence[Dict[str, Any]],
    *,
    usable_results: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    diagnostics = _followup_signal_diagnostics(results)
    usable = list(usable_results) if usable_results is not None else _substantive_followup_results(results)
    deep_results = [_as_dict(item) for item in list(results or []) if _as_dict(_followup_deep_trace(_as_dict(item))).get("prefer_deep")]
    deep_signal_count = len([item for item in deep_results if _followup_result_signal_score(item) >= 2])
    deep_fallback_used_count = len([item for item in deep_results if _as_dict(_followup_deep_trace(item)).get("fallback_used")])
    cache_seed_results = [_as_dict(item) for item in list(results or []) if _as_dict(item).get("cache_seed")]
    live_refresh_required = [_as_dict(item) for item in cache_seed_results if _as_dict(item).get("live_refresh_required")]
    live_refresh_results = [
        _as_dict(item)
        for item in list(results or [])
        if _as_dict(_as_dict(item).get("search_task")).get("cache_seed_available")
        and _as_dict(_as_dict(item).get("search_task")).get("live_refresh_required")
    ]
    live_refresh_signal_count = len([item for item in live_refresh_results if _followup_result_signal_score(item) >= 2])
    new_ab_source_count = _followup_new_ab_source_count(usable)
    new_metric_ready_count = _followup_new_metric_ready_count(usable)
    resolved_gap_ids = {
        _followup_gap_id(_as_dict(item).get("search_task") or _as_dict(item))
        for item in list(usable or [])
        if _followup_result_signal_score(_as_dict(item)) >= 2 and _followup_gap_id(_as_dict(item).get("search_task") or _as_dict(item))
    }
    return {
        "attempted_result_count": len([item for item in list(results or []) if isinstance(item, dict)]),
        "signal_count": int(diagnostics.get("signal_count") or 0),
        "empty_success_count": int(diagnostics.get("empty_success_count") or 0),
        "failed_count": int(diagnostics.get("failed_count") or 0),
        "partial_count": int(diagnostics.get("partial_count") or 0),
        "new_usable_evidence_count": len(usable),
        "new_ab_source_count": new_ab_source_count,
        "new_metric_ready_count": new_metric_ready_count,
        "resolved_gap_count": len(resolved_gap_ids),
        "deep_signal_count": deep_signal_count,
        "deep_no_signal_count": max(0, len(deep_results) - deep_signal_count),
        "deep_fallback_used_count": deep_fallback_used_count,
        "cache_seed_count": len(cache_seed_results),
        "live_refresh_required_count": len(live_refresh_required),
        "live_refresh_attempted_count": len(live_refresh_results),
        "live_refresh_signal_count": live_refresh_signal_count,
        "cache_seed_used_for_draft_count": max(0, len(live_refresh_results) - live_refresh_signal_count),
        "no_signal_reasons": _as_list(diagnostics.get("no_signal_reasons")),
        "proof_delta_summary": {
            "signal_count": int(diagnostics.get("signal_count") or 0),
            "new_usable_evidence_count": len(usable),
            "new_ab_source_count": new_ab_source_count,
            "new_metric_ready_count": new_metric_ready_count,
            "resolved_gap_count": len(resolved_gap_ids),
            "claim_binding_delta": 0,
            "table_ready_delta": new_metric_ready_count,
            "quality_gain": bool(new_ab_source_count > 0 or new_metric_ready_count > 0 or resolved_gap_ids),
        },
    }


def _followup_search_task_trace(item: Dict[str, Any]) -> Dict[str, Any]:
    task = _as_dict(_as_dict(item).get("search_task"))
    if not task:
        return {}
    return {
        "task_id": task.get("task_id"),
        "proof_role": task.get("proof_role"),
        "evidence_type": task.get("evidence_type"),
        "blocking_gaps": _as_list(task.get("blocking_gaps")),
        "lane_targets": _as_list(task.get("lane_targets")),
        "prefer_deep": task.get("prefer_deep"),
        "deep_status": task.get("deep_status"),
        "deep_reason": task.get("deep_reason"),
        "deep_skip_reason": task.get("deep_skip_reason"),
        "cache_seed_available": task.get("cache_seed_available"),
        "live_refresh_required": task.get("live_refresh_required"),
    }


def _trace_followup_result(item: Dict[str, Any]) -> Dict[str, Any]:
    item = _as_dict(item)
    trace = {
        "agent": item.get("agent"),
        "child_agent": item.get("child_agent"),
        "query": item.get("query"),
        "targets_gap": item.get("targets_gap"),
        "status": item.get("status"),
        "confidence": item.get("confidence"),
    }
    search_task = _followup_search_task_trace(item)
    if search_task:
        trace["search_task"] = search_task
    if item.get("cache_seed"):
        trace["cache_seed"] = True
        trace["live_refresh_required"] = item.get("live_refresh_required")
    deep_trace = _as_dict(_followup_deep_trace(item))
    if deep_trace:
        trace["deep_search"] = {
            "prefer_deep": deep_trace.get("prefer_deep"),
            "primary_engine": deep_trace.get("primary_engine"),
            "fallback_used": deep_trace.get("fallback_used"),
            "deep_unavailable": deep_trace.get("deep_unavailable"),
            "deep_exhausted": deep_trace.get("deep_exhausted"),
        }
    return trace


def _repair_task_summary_after_policy(
    tasks: Sequence[Dict[str, Any]],
    results: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    pre_summary = _repair_task_summary(tasks)
    post_tasks = [
        {"search_task": _as_dict(_as_dict(item).get("search_task"))}
        for item in list(results or [])
        if _as_dict(_as_dict(item).get("search_task"))
    ]
    post_summary = _repair_task_summary(post_tasks) if post_tasks else {}
    summary = dict(pre_summary)
    summary["pre_deep_task_count"] = int(_safe_float(pre_summary.get("deep_task_count"), 0.0))
    summary["pre_deep_skipped_count"] = int(_safe_float(pre_summary.get("deep_skipped_count"), 0.0))
    summary["post_policy_task_count"] = int(_safe_float(post_summary.get("task_count"), 0.0)) if post_summary else 0
    summary["post_deep_task_count"] = int(_safe_float(post_summary.get("deep_task_count"), 0.0)) if post_summary else 0
    summary["post_deep_skipped_count"] = int(_safe_float(post_summary.get("deep_skipped_count"), 0.0)) if post_summary else 0
    summary["post_deep_budget_exhausted_count"] = (
        int(_safe_float(post_summary.get("deep_budget_exhausted_count"), 0.0)) if post_summary else 0
    )
    if post_summary:
        summary["deep_task_count"] = post_summary.get("deep_task_count", summary.get("deep_task_count", 0))
        summary["deep_skipped_count"] = post_summary.get("deep_skipped_count", summary.get("deep_skipped_count", 0))
        summary["deep_budget_exhausted_count"] = post_summary.get(
            "deep_budget_exhausted_count",
            summary.get("deep_budget_exhausted_count", 0),
        )
    return summary


def _evidence_package_snapshot(package: Dict[str, Any]) -> Dict[str, Any]:
    summary = _as_dict(_as_dict(package).get("summary"))
    source_dist = _as_dict(summary.get("source_level_distribution"))
    return {
        "evidence_count": int(_safe_float(summary.get("evidence_count"), 0.0)),
        "clean_fact_count": int(_safe_float(summary.get("clean_fact_count"), 0.0)),
        "analysis_ready_count": int(_safe_float(summary.get("analysis_ready_count"), 0.0)),
        "core_candidate_count": int(_safe_float(summary.get("core_candidate_count"), 0.0)),
        "appendix_only_count": int(_safe_float(summary.get("appendix_only_count"), 0.0)),
        "ab_source_count": int(_safe_float(source_dist.get("A"), 0.0)) + int(_safe_float(source_dist.get("B"), 0.0)),
        "ready_for_analysis": bool(summary.get("ready_for_analysis")),
    }


def _evidence_delta_summary(before_package: Dict[str, Any], after_package: Dict[str, Any]) -> Dict[str, Any]:
    before = _evidence_package_snapshot(before_package)
    after = _evidence_package_snapshot(after_package)
    return {
        "before": before,
        "after": after,
        "evidence_delta": after["evidence_count"] - before["evidence_count"],
        "clean_fact_delta": after["clean_fact_count"] - before["clean_fact_count"],
        "analysis_ready_delta": after["analysis_ready_count"] - before["analysis_ready_count"],
        "ab_source_delta": after["ab_source_count"] - before["ab_source_count"],
        "ready_for_analysis_changed": before["ready_for_analysis"] != after["ready_for_analysis"],
    }


def _writer_blocker_snapshot(writer_report: Dict[str, Any]) -> Dict[str, Any]:
    report = _as_dict(writer_report)
    package_quality = _as_dict(report.get("package_quality_report"))
    qa_result = _as_dict(report.get("qa_result"))
    deep_eval = _as_dict(qa_result.get("deep_evaluation"))
    chapter_packages = _as_list(report.get("chapter_evidence_packages"))
    coverage_matrix = _as_list(report.get("coverage_matrix"))
    missing_proofs = _as_list(report.get("missing_proof_standards"))
    package_errors = _as_list(package_quality.get("errors")) + _as_list(package_quality.get("blocking_errors"))
    required_followups = _as_list(report.get("required_followups")) + _as_list(deep_eval.get("required_followups"))
    core_ab_source_count = 0
    chapter_core_evidence_count = 0
    for chapter in chapter_packages:
        summary = _as_dict(_as_dict(chapter).get("evidence_summary"))
        chapter_core_ab = int(_safe_float(summary.get("core_ab_source_count"), 0.0))
        core_ab_source_count += chapter_core_ab
        if chapter_core_ab > 0 or _as_list(_as_dict(chapter).get("core_evidence")):
            chapter_core_evidence_count += 1
    claim_binding_error_keys = {
        (str(_as_dict(item).get("type") or ""), str(_as_dict(item).get("path") or ""))
        for item in package_errors
        if str(_as_dict(item).get("type") or "").strip() == "core_claim_without_ab_source"
    }
    claim_binding_error_count = len(claim_binding_error_keys)
    missing_proof_count = len(missing_proofs)
    if coverage_matrix:
        missing_proof_count += sum(1 for row in coverage_matrix if _as_list(_as_dict(row).get("blocking_gaps")))
    return {
        "writer_status": str(report.get("report_status") or ""),
        "package_error_count": len(package_errors),
        "claim_binding_error_count": claim_binding_error_count,
        "missing_proof_count": missing_proof_count,
        "core_ab_source_count": core_ab_source_count,
        "chapter_core_evidence_count": chapter_core_evidence_count,
        "required_followup_count": len(required_followups),
        "qa_passed": bool(qa_result.get("passed")),
        "pending_repair_count": len(_as_list(report.get("pending_repair_reasons"))) + len(required_followups),
    }


def _blocker_delta(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "package_error_delta": int(_safe_float(after.get("package_error_count"), 0.0)) - int(_safe_float(before.get("package_error_count"), 0.0)),
        "claim_binding_error_delta": int(_safe_float(after.get("claim_binding_error_count"), 0.0)) - int(_safe_float(before.get("claim_binding_error_count"), 0.0)),
        "claim_binding_delta": int(_safe_float(before.get("claim_binding_error_count"), 0.0)) - int(_safe_float(after.get("claim_binding_error_count"), 0.0)),
        "missing_proof_delta": int(_safe_float(after.get("missing_proof_count"), 0.0)) - int(_safe_float(before.get("missing_proof_count"), 0.0)),
        "core_ab_source_delta": int(_safe_float(after.get("core_ab_source_count"), 0.0)) - int(_safe_float(before.get("core_ab_source_count"), 0.0)),
        "chapter_core_evidence_delta": int(_safe_float(after.get("chapter_core_evidence_count"), 0.0)) - int(_safe_float(before.get("chapter_core_evidence_count"), 0.0)),
        "required_followup_delta": int(_safe_float(after.get("required_followup_count"), 0.0)) - int(_safe_float(before.get("required_followup_count"), 0.0)),
        "qa_passed_changed": bool(after.get("qa_passed")) != bool(before.get("qa_passed")),
    }


def _repair_quality_gain(
    before_report: Dict[str, Any],
    after_report: Dict[str, Any],
    before_package: Dict[str, Any],
    after_package: Dict[str, Any],
) -> Dict[str, Any]:
    before_blockers = _writer_blocker_snapshot(before_report)
    after_blockers = _writer_blocker_snapshot(after_report)
    blocker_delta = _blocker_delta(before_blockers, after_blockers)
    evidence_delta = _evidence_delta_summary(before_package, after_package)
    has_quality_gain = any(
        [
            int(_safe_float(blocker_delta.get("package_error_delta"), 0.0)) < 0,
            int(_safe_float(blocker_delta.get("claim_binding_delta"), 0.0)) > 0,
            int(_safe_float(blocker_delta.get("core_ab_source_delta"), 0.0)) > 0,
            int(_safe_float(blocker_delta.get("chapter_core_evidence_delta"), 0.0)) > 0,
            int(_safe_float(evidence_delta.get("ab_source_delta"), 0.0)) > 0,
        ]
    )
    next_route = "continue" if has_quality_gain else "manual_review"
    if (
        not has_quality_gain
        and int(_safe_float(after_blockers.get("claim_binding_error_count"), 0.0)) > 0
        and int(_safe_float(_evidence_package_snapshot(after_package).get("ab_source_count"), 0.0)) > 0
    ):
        next_route = "claim_rebuild"
    proof_delta_summary = {
        "package_error_delta": blocker_delta.get("package_error_delta"),
        "core_ab_source_delta": blocker_delta.get("core_ab_source_delta"),
        "claim_binding_delta": blocker_delta.get("claim_binding_delta"),
        "chapter_core_evidence_delta": blocker_delta.get("chapter_core_evidence_delta"),
        "analysis_ready_delta": evidence_delta.get("analysis_ready_delta"),
        "ab_source_delta": evidence_delta.get("ab_source_delta"),
        "quality_gain": has_quality_gain,
    }
    claim_binding_feedback = {
        "claim_binding_error_count": after_blockers.get("claim_binding_error_count"),
        "available_ab_source_count": _evidence_package_snapshot(after_package).get("ab_source_count"),
        "route": next_route if next_route == "claim_rebuild" else "",
        "reason": "ab_evidence_available_but_not_bound" if next_route == "claim_rebuild" else "",
    }
    return {
        "has_quality_gain": has_quality_gain,
        "quality_gain": has_quality_gain,
        "blocker_delta": blocker_delta,
        "evidence_delta_summary": evidence_delta,
        "proof_delta_summary": proof_delta_summary,
        "claim_binding_feedback_summary": claim_binding_feedback,
        "package_error_delta": blocker_delta.get("package_error_delta"),
        "core_ab_source_delta": blocker_delta.get("core_ab_source_delta"),
        "claim_binding_delta": blocker_delta.get("claim_binding_delta"),
        "chapter_core_evidence_delta": blocker_delta.get("chapter_core_evidence_delta"),
        "next_route": next_route,
    }


def _claim_rebuild_context_from_package(
    *,
    reason: str,
    writer_report: Dict[str, Any],
    evidence_package: Dict[str, Any],
) -> Dict[str, Any]:
    blockers = _writer_blocker_snapshot(writer_report)
    if int(_safe_float(blockers.get("claim_binding_error_count"), 0.0)) <= 0:
        return {}
    package_snapshot = _evidence_package_snapshot(evidence_package)
    if int(_safe_float(package_snapshot.get("ab_source_count"), 0.0)) <= 0:
        return {}
    package_quality = _as_dict(writer_report.get("package_quality_report"))
    package_errors = _as_list(package_quality.get("blocking_errors")) or _as_list(package_quality.get("errors"))
    target_claims = [
        {
            "path": _as_dict(item).get("path"),
            "message": _as_dict(item).get("message"),
        }
        for item in package_errors
        if str(_as_dict(item).get("type") or "") == "core_claim_without_ab_source"
    ][:8]
    return {
        "reason": reason,
        "target_claims": target_claims,
        "available_ab_source_count": package_snapshot.get("ab_source_count"),
        "instruction": "A/B evidence is available in the package; rebuild argument units so core claims bind to eligible sources instead of searching again.",
    }


def _candidate_not_better_reasons(before: Dict[str, Any], after: Dict[str, Any]) -> List[str]:
    reasons: List[str] = []
    if int(_safe_float(after.get("package_error_count"), 0.0)) >= int(_safe_float(before.get("package_error_count"), 0.0)):
        reasons.append("package_errors_not_reduced")
    if int(_safe_float(after.get("claim_binding_error_count"), 0.0)) >= int(_safe_float(before.get("claim_binding_error_count"), 0.0)):
        reasons.append("claim_bindings_not_improved")
    if int(_safe_float(after.get("missing_proof_count"), 0.0)) >= int(_safe_float(before.get("missing_proof_count"), 0.0)):
        reasons.append("missing_proofs_not_reduced")
    if int(_safe_float(after.get("core_ab_source_count"), 0.0)) <= int(_safe_float(before.get("core_ab_source_count"), 0.0)):
        reasons.append("core_ab_sources_not_increased")
    if bool(before.get("qa_passed")) and not bool(after.get("qa_passed")):
        reasons.append("qa_regressed")
    return reasons or ["writer_quality_key_not_improved"]


def _gap_ledger_from_followups(
    followups: Sequence[Dict[str, Any]],
    results: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    ledger: Dict[str, Dict[str, Any]] = {}
    for followup in followups:
        if not isinstance(followup, dict):
            continue
        gap_id = _followup_gap_id(followup)
        if not gap_id:
            continue
        ledger.setdefault(
            gap_id,
            {
                "gap_id": gap_id,
                "targets_gap": followup.get("targets_gap") or followup.get("dimension_name") or followup.get("evidence_goal"),
                "query_count": 0,
                "result_count": 0,
                "signal_count": 0,
                "max_signal_score": 0,
                "status": "pending",
            },
        )
        ledger[gap_id]["query_count"] += 1
    for result in results:
        if not isinstance(result, dict):
            continue
        search_task = _as_dict(result.get("search_task"))
        gap_id = _followup_gap_id(search_task) or _followup_gap_id(result)
        if not gap_id:
            continue
        item = ledger.setdefault(
            gap_id,
            {
                "gap_id": gap_id,
                "targets_gap": result.get("targets_gap") or search_task.get("targets_gap") or search_task.get("dimension_name"),
                "query_count": 0,
                "result_count": 0,
                "signal_count": 0,
                "max_signal_score": 0,
                "status": "pending",
            },
        )
        score = _followup_result_signal_score(result)
        item["result_count"] += 1
        item["max_signal_score"] = max(int(item.get("max_signal_score") or 0), score)
        if score >= 2:
            item["signal_count"] += 1
    for item in ledger.values():
        if int(item.get("signal_count") or 0) > 0:
            item["status"] = "evidence_found"
        elif int(item.get("result_count") or 0) > 0:
            item["status"] = "searched_no_signal"
        else:
            item["status"] = "pending"
    return sorted(ledger.values(), key=lambda item: (str(item.get("status") or ""), str(item.get("gap_id") or "")))


NON_EVIDENCE_FOLLOWUP_TYPES = {
    "report_body_below_target_chars",
    "body_below_target_chars",
    "report_length_below_target",
    "insufficient_body_length",
    "body_table_contains_source_header",
    "table_contains_source_header",
    "source_header_leaked",
    "markdown_source_header",
    "format_issue",
    "formatting_issue",
}

SOFT_NON_EVIDENCE_FOLLOWUP_TYPES = {
    "mechanism_explanation_weak",
    "section_reasoning_weak",
    "section_too_short",
    "argument_thin",
    "writing_depth_weak",
}

EVIDENCE_FOLLOWUP_HINTS = {
    "missing_proof_standard",
    "missing_proof_standards",
    "mandatory_proof_missing",
    "insufficient_ab_sources",
    "insufficient_ab_core_sources",
    "insufficient_counter_sources",
    "needs_corroboration",
    "metric_evidence_missing",
    "case_evidence_missing",
    "counter_evidence_missing",
    "missing_metric",
    "missing_case",
    "missing_counter",
    "proof",
    "source",
    "evidence",
    "corroboration",
    "counter",
    "metric",
    "case",
}

DEEP_REPAIR_BLOCKING_GAPS = {
    "insufficient_ab_sources",
    "insufficient_ab_core_sources",
    "mandatory_proof_missing",
    "metric_scope_period_unit_incomplete",
    "core_claim_without_ab_source",
    "citation_source_missing",
    "invalid_citation",
    "missing_source_ref",
}

DEEP_REPAIR_PROOF_ROLES = {"source_check", "metric", "filing", "official_data"}
DEEP_REPAIR_LOOPS = {
    "evidence_preflight",
    "post_qa_repair",
    "reformatter_preflight",
    "layout_followup",
    "supervisor_coverage",
}
CORE_CACHE_REFRESH_BLOCKING_GAPS = {
    "insufficient_ab_sources",
    "insufficient_ab_core_sources",
    "core_claim_without_ab_source",
    "mandatory_proof_missing",
    "metric_scope_period_unit_incomplete",
    "citation_source_missing",
}
CORE_CACHE_REFRESH_PROOF_ROLES = {"source_check", "metric", "filing", "official_data"}
CORE_CACHE_REFRESH_LOOPS = {
    "evidence_preflight",
    "post_qa_repair",
    "reformatter_preflight",
    "supervisor_coverage",
}


def _deep_repair_enabled() -> bool:
    return _env_flag("IQS_DEEP_REPAIR_ENABLED", True)


def _deep_repair_engines() -> List[str]:
    raw = os.getenv("IQS_DEEP_REPAIR_ENGINE_TYPES", "Deep,LiteAdvanced,GenericAdvanced,Generic")
    engines: List[str] = []
    for part in re.split(r"[,，\s]+", str(raw or "")):
        engine = part.strip()
        if engine and engine not in engines:
            engines.append(engine)
    if "Deep" not in engines:
        engines.insert(0, "Deep")
    return engines or ["Deep", "LiteAdvanced", "GenericAdvanced", "Generic"]


def _deep_repair_gap_query_key(task: Dict[str, Any]) -> str:
    gap = _followup_gap_id(task) or _followup_target_key(task)
    query = re.sub(r"\s+", " ", str(task.get("query") or task.get("suggested_query") or "").strip().lower())
    return f"{gap}|{query}"


def _deep_repair_reason(task: Dict[str, Any]) -> str:
    if not _deep_repair_enabled():
        return ""
    payload = _as_dict(task)
    if payload.get("prefer_deep") is True:
        return str(payload.get("deep_reason") or "explicit_prefer_deep")
    markers = _followup_marker_values(payload)
    gaps = {str(item or "").strip().lower() for item in _as_list(payload.get("blocking_gaps")) if str(item or "").strip()}
    gaps.update(marker for marker in markers if marker in DEEP_REPAIR_BLOCKING_GAPS)
    if gaps.intersection(DEEP_REPAIR_BLOCKING_GAPS):
        return "blocking_gap:" + ",".join(sorted(gaps.intersection(DEEP_REPAIR_BLOCKING_GAPS))[:3])
    role = str(payload.get("proof_role") or payload.get("evidence_type") or "").strip().lower()
    if role in DEEP_REPAIR_PROOF_ROLES:
        return f"proof_role:{role}"
    loop_name = str(payload.get("loop_name") or "").strip().lower()
    origin_node = str(payload.get("origin_node") or "").strip().lower()
    if loop_name in DEEP_REPAIR_LOOPS:
        return f"repair_loop:{loop_name}"
    if origin_node in {"evidence_binder", "writer_qa", "reformatter_preflight"}:
        return f"origin_node:{origin_node}"
    return ""


def _deep_repair_budget_metadata(state: BrainAgentState) -> Dict[str, Any]:
    metadata = dict(state.get("metadata") or {})
    state["metadata"] = metadata
    deep_state = dict(metadata.get("iqs_deep_repair") or {})
    metadata["iqs_deep_repair"] = deep_state
    deep_state.setdefault("used_count", 0)
    deep_state.setdefault("seen_gap_queries", [])
    deep_state.setdefault("deep_unavailable", False)
    return deep_state


def _deep_repair_loop_limit(loop_name: str, default: int) -> int:
    loop = str(loop_name or "").strip().lower()
    env_by_loop = {
        "evidence_preflight": ("IQS_DEEP_REPAIR_EVIDENCE_PREFLIGHT_MAX", 5),
        "post_qa_repair": ("IQS_DEEP_REPAIR_POST_QA_MAX", 6),
        "reformatter_preflight": ("IQS_DEEP_REPAIR_REFORMATTER_PREFLIGHT_MAX", 4),
        "layout_followup": ("IQS_DEEP_REPAIR_LAYOUT_MAX", 2),
    }
    name, fallback = env_by_loop.get(loop, ("IQS_DEEP_REPAIR_MAX_TASKS_PER_ROUND", default))
    return _env_int(name, fallback, min_value=0, max_value=50)


def _apply_deep_repair_policy_to_tasks(
    tasks: Sequence[Dict[str, Any]],
    *,
    state: BrainAgentState,
    round_number: int,
) -> List[Dict[str, Any]]:
    if not tasks:
        return []
    round_limit = max(0, _env_int("IQS_DEEP_REPAIR_MAX_TASKS_PER_ROUND", 6))
    report_limit = max(0, _env_int("IQS_DEEP_REPAIR_MAX_TASKS_PER_REPORT", 24))
    deep_state = _deep_repair_budget_metadata(state)
    report_used = int(_safe_float(deep_state.get("used_count"), 0.0))
    deep_unavailable = bool(deep_state.get("deep_unavailable"))
    seen_gap_queries = set(str(item) for item in _as_list(deep_state.get("seen_gap_queries")))
    round_used = 0
    loop_round_used: Dict[str, int] = {}
    updated: List[Dict[str, Any]] = []
    for task in tasks:
        copied = dict(task)
        search_task = dict(_as_dict(copied.get("search_task")))
        reason = _deep_repair_reason(search_task)
        gap_query_key = _deep_repair_gap_query_key(search_task)
        loop_name = str(search_task.get("loop_name") or "unknown").strip().lower() or "unknown"
        loop_limit = _deep_repair_loop_limit(loop_name, round_limit)
        deep_status = "not_selected"
        markers = _followup_marker_values(search_task)
        if (
            not _strict_quality_mode()
            and round_number >= 3
            and markers.intersection({"insufficient_ab_sources", "insufficient_ab_core_sources", "core_claim_without_ab_source"})
            and str(search_task.get("claim_type") or search_task.get("conclusion_type") or "").strip().lower() != "hard_metric"
        ):
            search_task["min_source_level"] = ["A", "B", "C"]
            search_task["balanced_evidence_degradation"] = "after_two_rounds_allow_b_or_c_corroboration"
            source_priority: List[str] = []
            for value in [*_as_list(search_task.get("source_priority")), "research", "market_research", "authoritative_media"]:
                text = str(value or "").strip()
                if text and text not in source_priority:
                    source_priority.append(text)
            search_task["source_priority"] = source_priority[:12]
        if not reason:
            search_task["prefer_deep"] = False
            search_task.setdefault("retrieval_mode", "normal")
            search_task.setdefault("retrieval_reason", "deep_repair_not_required")
            search_task.setdefault("primary_provider", "iqs_normal")
        elif deep_unavailable:
            deep_status = "deep_unavailable"
            search_task["prefer_deep"] = False
            search_task["deep_skip_reason"] = deep_status
            search_task.setdefault("retrieval_mode", "normal")
            search_task.setdefault("retrieval_reason", deep_status)
            search_task.setdefault("primary_provider", "iqs_normal")
        elif report_used >= report_limit:
            deep_status = "report_budget_exhausted"
            search_task["prefer_deep"] = False
            search_task["deep_skip_reason"] = deep_status
            search_task.setdefault("retrieval_mode", "normal")
            search_task.setdefault("retrieval_reason", deep_status)
            search_task.setdefault("primary_provider", "iqs_normal")
        elif round_used >= round_limit:
            deep_status = "round_budget_exhausted"
            search_task["prefer_deep"] = False
            search_task["deep_skip_reason"] = deep_status
            search_task.setdefault("retrieval_mode", "normal")
            search_task.setdefault("retrieval_reason", deep_status)
            search_task.setdefault("primary_provider", "iqs_normal")
        elif loop_round_used.get(loop_name, 0) >= loop_limit:
            deep_status = "loop_budget_exhausted"
            search_task["prefer_deep"] = False
            search_task["deep_skip_reason"] = deep_status
            search_task.setdefault("retrieval_mode", "normal")
            search_task.setdefault("retrieval_reason", deep_status)
            search_task.setdefault("primary_provider", "iqs_normal")
        elif gap_query_key in seen_gap_queries:
            deep_status = "duplicate_gap_query"
            search_task["prefer_deep"] = False
            search_task["deep_skip_reason"] = deep_status
            search_task.setdefault("retrieval_mode", "normal")
            search_task.setdefault("retrieval_reason", deep_status)
            search_task.setdefault("primary_provider", "iqs_normal")
        else:
            deep_status = "selected"
            search_task["prefer_deep"] = True
            search_task["deep_reason"] = reason
            search_task["deep_round"] = round_number
            search_task["engineTypes"] = _deep_repair_engines()
            search_task["retrieval_mode"] = "deep"
            search_task["retrieval_reason"] = reason
            search_task["primary_provider"] = "iqs_deep"
            search_task["fallback_providers"] = ["iqs_normal"]
            search_task.setdefault("search_options", {})
            task_options = dict(_as_dict(search_task.get("search_options")))
            task_options.update(
                {
                    "prefer_deep": True,
                    "deep_reason": reason,
                    "engineTypes": _deep_repair_engines(),
                    "enable_batch_search": False,
                }
            )
            search_task["search_options"] = task_options
            seen_gap_queries.add(gap_query_key)
            round_used += 1
            loop_round_used[loop_name] = loop_round_used.get(loop_name, 0) + 1
            report_used += 1
        search_task["deep_status"] = deep_status
        copied["search_task"] = search_task
        updated.append(copied)
    deep_state["used_count"] = report_used
    deep_state["seen_gap_queries"] = sorted(seen_gap_queries)[-200:]
    deep_state["last_round_loop_used"] = loop_round_used
    return updated


def _record_deep_repair_unavailable_from_results(state: BrainAgentState, results: Sequence[Dict[str, Any]]) -> None:
    unavailable_traces: List[Dict[str, Any]] = []
    for item in results or []:
        trace = _as_dict(_as_dict(item).get("deep_search"))
        if not trace:
            continue
        if trace.get("deep_unavailable"):
            unavailable_traces.append(trace)
    if not unavailable_traces:
        return
    deep_state = _deep_repair_budget_metadata(state)
    deep_state["deep_unavailable"] = True
    deep_state["deep_unavailable_round_reason"] = "permission_or_quota_error"
    deep_state["deep_unavailable_examples"] = [
        {
            "query": item.get("query"),
            "deep_reason": item.get("deep_reason"),
            "primary_engine": item.get("primary_engine"),
            "errors": _as_list(item.get("errors"))[:3],
        }
        for item in unavailable_traces[:3]
    ]


def _followup_marker_values(item: Dict[str, Any]) -> set[str]:
    payload = _as_dict(item)
    nested = _as_dict(payload.get("follow_up_query"))
    markers: List[Any] = []
    for source in (payload, nested):
        markers.extend(
            [
                source.get("type"),
                source.get("reason"),
                source.get("targets_gap"),
                source.get("gap_type"),
                source.get("proof_role"),
                source.get("evidence_type"),
                source.get("decision_use"),
            ]
        )
        markers.extend(_as_list(source.get("blocking_gaps")))
        markers.extend(_as_list(source.get("missing")))
    result: set[str] = set()
    for marker in markers:
        text = re.sub(r"\s+", "_", str(marker or "").strip().lower())
        if text:
            result.add(text)
    return result


def _minimum_cache_source_levels_for_task(task: Dict[str, Any]) -> List[str]:
    markers = _followup_marker_values(task)
    if markers.intersection(
        {
            "insufficient_ab_sources",
            "insufficient_ab_core_sources",
            "mandatory_proof_missing",
            "core_claim_without_ab_source",
        }
    ):
        try:
            repair_round = int(task.get("gap_repair_round") or task.get("round") or 0)
        except (TypeError, ValueError):
            repair_round = 0
        claim_type = str(task.get("claim_type") or task.get("conclusion_type") or "").strip().lower()
        if not _strict_quality_mode() and repair_round >= 3 and claim_type != "hard_metric":
            return ["A", "B", "C"]
        return ["A", "B"]
    return ["A", "B", "C"]


def _required_cache_fields_for_task(task: Dict[str, Any]) -> List[str]:
    required = _as_list(task.get("required_fields"))
    role = str(task.get("proof_role") or task.get("evidence_type") or "").strip().lower()
    markers = _followup_marker_values(task)
    if role == "metric" or "metric_scope_period_unit_incomplete" in markers:
        required = [*required, "metric", "period", "unit", "source"]
    if any(
        marker in markers
        for marker in {
            "citation_source_missing",
            "missing_source_ref",
            "invalid_citation",
            "insufficient_ab_sources",
            "insufficient_ab_core_sources",
            "mandatory_proof_missing",
            "core_claim_without_ab_source",
        }
    ):
        required = [*required, "source"]
    deduped: List[str] = []
    for item in required:
        text = str(item or "").strip().lower()
        if text and text not in deduped:
            deduped.append(text)
    return deduped


def _cache_live_refresh_required(task: Dict[str, Any]) -> bool:
    if not _env_flag("EVIDENCE_CACHE_CORE_LIVE_REFRESH_ENABLED", True):
        return False
    payload = _as_dict(task)
    markers = _followup_marker_values(payload)
    if markers.intersection(CORE_CACHE_REFRESH_BLOCKING_GAPS):
        return True
    role = str(payload.get("proof_role") or payload.get("evidence_type") or "").strip().lower()
    loop_name = str(payload.get("loop_name") or "").strip().lower()
    if role in CORE_CACHE_REFRESH_PROOF_ROLES and loop_name in CORE_CACHE_REFRESH_LOOPS:
        return True
    if str(payload.get("decision_use") or "").strip().lower() in {"core_claim", "decision_ready", "investment_or_market_entry"}:
        return True
    return False


def _repair_lineage_from_task(task: Dict[str, Any]) -> Dict[str, Any]:
    payload = _as_dict(task)
    return {
        "gap_id": payload.get("gap_id") or _followup_gap_id(payload),
        "requirement_id": payload.get("requirement_id") or payload.get("evidence_requirement_id") or "",
        "proof_role": payload.get("proof_role") or payload.get("evidence_type") or "",
        "required_fields": _required_cache_fields_for_task(payload) or _as_list(payload.get("required_fields")),
        "blocking_gaps": _as_list(payload.get("blocking_gaps")),
        "loop_name": payload.get("loop_name") or "",
        "origin_node": payload.get("origin_node") or "",
    }


def _cache_hit_followup_result(
    *,
    task: Dict[str, Any],
    routed_task: Dict[str, Any],
    hits: Sequence[Dict[str, Any]],
    round_number: int,
    live_refresh_required: bool = False,
) -> Dict[str, Any]:
    key_sources: List[Dict[str, Any]] = []
    raw_points: List[Dict[str, Any]] = []
    answer_parts: List[str] = []
    lineage = _repair_lineage_from_task(task)
    for hit in list(hits or []):
        raw = _as_dict(hit.get("raw"))
        source = _as_dict(raw.get("source"))
        origin = "trusted_source_cache" if bool(hit.get("trusted_source_cache")) else "cache"
        source_url = str(hit.get("source_url") or source.get("url") or raw.get("source_url") or "").strip()
        title = str(source.get("title") or hit.get("source_domain") or source_url or "cached evidence").strip()
        source_item = {
            "title": title,
            "url": source_url,
            "source_level": hit.get("source_level") or raw.get("source_level"),
            "source_type": hit.get("source_type") or raw.get("source_type"),
            "snippet": _compact_text(hit.get("fact_description") or raw.get("fact") or raw.get("content"), max_chars=260),
            "evidence_origin": origin,
            "cache_seed": True,
            "live_verified": False,
            "gap_id": lineage.get("gap_id"),
            "requirement_id": lineage.get("requirement_id"),
        }
        if source_item not in key_sources:
            key_sources.append(source_item)
        metric = str(hit.get("metric_name") or raw.get("metric") or "").strip()
        value = str(hit.get("value") or raw.get("value") or raw.get("numeric_value") or "").strip()
        if metric or value:
            raw_points.append(
                {
                    "metric": metric,
                    "value": value,
                    "unit": hit.get("unit") or raw.get("unit") or raw.get("numeric_unit"),
                    "period": hit.get("period") or raw.get("period"),
                    "source_url": source_url,
                    "source_level": hit.get("source_level") or raw.get("source_level"),
                    "evidence": _compact_text(hit.get("fact_description") or raw.get("fact") or raw.get("content"), max_chars=420),
                    "cache_evidence_id": hit.get("evidence_id"),
                    "evidence_origin": origin,
                    "cache_seed": True,
                    "live_verified": False,
                    "gap_id": lineage.get("gap_id"),
                    "requirement_id": lineage.get("requirement_id"),
                }
            )
        fact_text = _compact_text(hit.get("fact_description") or raw.get("fact") or raw.get("content"), max_chars=320)
        if fact_text:
            answer_parts.append(fact_text)
    return {
        "round": round_number,
        "agent": routed_task.get("agent"),
        "child_agent": IQS_ROLE_CONFIGS.get(str(routed_task.get("agent") or ""), {}).get("child", "web_analysis_agent"),
        "query": routed_task.get("query"),
        "targets_gap": routed_task.get("targets_gap"),
        **lineage,
        "status": "success",
        "confidence": max([float(_safe_float(hit.get("confidence_score"), 0.55)) for hit in list(hits or [])] or [0.55]),
        "answer": "；".join(answer_parts[:4]),
        "key_sources": key_sources,
        "raw_data_points": raw_points,
        "limitations": {"cache_hit": True, "cache_layer": "evidence_cache"},
        "note": "补证任务命中本地 evidence cache，作为候选证据种子使用。" if live_refresh_required else "补证任务命中本地 evidence cache，跳过联网检索。",
        "search_task": task,
        "evidence_origin": "cache",
        "cache_seed": True,
        "live_verified": False,
        "live_refresh_required": bool(live_refresh_required),
        "cache": {
            "hit": True,
            "layer": "evidence_cache",
            "hit_count": len(list(hits or [])),
            "cache_seed": True,
            "live_refresh_required": bool(live_refresh_required),
            "source_levels": sorted({str(hit.get("source_level") or "") for hit in list(hits or []) if str(hit.get("source_level") or "")}),
            "gap_id": lineage.get("gap_id"),
            "requirement_id": lineage.get("requirement_id"),
            "proof_role": lineage.get("proof_role"),
            "required_fields": lineage.get("required_fields"),
        },
    }


def _evidence_cache_gap_summary(cache_results: Sequence[Dict[str, Any]], cache_only_skipped: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    cache_only_gap_ids = {
        _repair_lineage_from_task(_as_dict(item).get("search_task") or _as_dict(item)).get("gap_id")
        for item in list(cache_only_skipped or [])
        if isinstance(item, dict)
    }
    result: Dict[str, Dict[str, Any]] = {}
    for item in list(cache_results or []):
        payload = _as_dict(item)
        gap_id = str(payload.get("gap_id") or "").strip()
        if not gap_id:
            continue
        entry = dict(result.get(gap_id) or {})
        entry["requirement_id"] = entry.get("requirement_id") or str(payload.get("requirement_id") or "")
        entry["proof_role"] = entry.get("proof_role") or str(payload.get("proof_role") or "")
        entry["required_fields"] = entry.get("required_fields") or _as_list(payload.get("required_fields"))
        entry["cache_hit_count"] = int(_safe_float(entry.get("cache_hit_count"), 0.0)) + 1
        if payload.get("live_refresh_required"):
            entry["live_refresh_required_count"] = int(_safe_float(entry.get("live_refresh_required_count"), 0.0)) + 1
        else:
            entry.setdefault("live_refresh_required_count", 0)
        if gap_id in cache_only_gap_ids:
            entry["cache_only_skip_count"] = int(_safe_float(entry.get("cache_only_skip_count"), 0.0)) + 1
        else:
            entry.setdefault("cache_only_skip_count", 0)
        result[gap_id] = entry
    return result


def _is_fake_cache_hit(hit: Dict[str, Any]) -> bool:
    raw = _as_dict(hit.get("raw"))
    source = _as_dict(raw.get("source"))
    text = " ".join(
        str(value or "").lower()
        for value in [
            hit.get("source_url"),
            hit.get("fact_description"),
            hit.get("source_domain"),
            raw.get("source_url"),
            raw.get("fact"),
            raw.get("content"),
            source.get("url"),
            source.get("title"),
        ]
    )
    return bool(
        "example.gov" in text
        or "example.com" in text
        or "official data shows ai agent adoption reached 50% in 2025" in text
    )


def _apply_evidence_cache_to_followup_tasks(
    tasks: Sequence[Dict[str, Any]],
    *,
    state: BrainAgentState,
    round_number: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    cache_results: List[Dict[str, Any]] = []
    remaining: List[Dict[str, Any]] = []
    cache_only_skipped: List[Dict[str, Any]] = []
    for routed_task in list(tasks or []):
        task = _as_dict(routed_task.get("search_task"))
        agent = str(routed_task.get("agent") or "").strip().lower()
        if agent == "rag" or not task or _is_non_evidence_followup(task):
            remaining.append(dict(routed_task))
            continue
        hits = lookup_cached_evidence(
            task,
            min_source_level=_minimum_cache_source_levels_for_task(task),
            required_fields=_required_cache_fields_for_task(task),
            max_hits=_env_int("EVIDENCE_CACHE_MAX_HITS_PER_TASK", 6, min_value=1, max_value=50),
        )
        trusted_hits = lookup_trusted_sources(
            task,
            min_source_level=_minimum_cache_source_levels_for_task(task),
            required_fields=_required_cache_fields_for_task(task),
            max_hits=_env_int("TRUSTED_SOURCE_CACHE_MAX_HITS_PER_TASK", 4, min_value=1, max_value=50),
        )
        if trusted_hits:
            hits = list(hits or []) + [hit for hit in trusted_hits if isinstance(hit, dict)]
        if not hits:
            remaining.append(dict(routed_task))
            continue
        fake_hits = [hit for hit in list(hits or []) if isinstance(hit, dict) and _is_fake_cache_hit(hit)]
        hits = [hit for hit in list(hits or []) if isinstance(hit, dict) and not _is_fake_cache_hit(hit)]
        if fake_hits:
            metadata = dict(state.get("metadata") or {})
            summary = dict(metadata.get("evidence_cache_summary") or {})
            summary["fake_or_placeholder_cache_hit_count"] = int(_safe_float(summary.get("fake_or_placeholder_cache_hit_count"), 0.0)) + len(fake_hits)
            metadata["evidence_cache_summary"] = summary
            state["metadata"] = metadata
        if not hits:
            remaining.append(dict(routed_task))
            continue
        live_refresh_required = _cache_live_refresh_required(task)
        if _deep_repair_reason(task) and not live_refresh_required:
            record_cache_activity(skipped_deep_count=1)
        cache_results.append(
            _cache_hit_followup_result(
                task=task,
                routed_task=_as_dict(routed_task),
                hits=hits,
                round_number=round_number,
                live_refresh_required=live_refresh_required,
            )
        )
        if live_refresh_required:
            live_task = dict(routed_task)
            live_search_task = dict(task)
            live_search_task["cache_seed_available"] = True
            live_search_task["cache_seed_hit_count"] = len(list(hits or []))
            live_search_task["live_refresh_required"] = True
            live_task["search_task"] = live_search_task
            remaining.append(live_task)
        else:
            cache_only_skipped.append(dict(routed_task))
    if cache_results:
        metadata = dict(state.get("metadata") or {})
        summary = dict(metadata.get("evidence_cache_summary") or {})
        by_gap = dict(summary.get("by_gap") or {})
        for gap_id, gap_summary in _evidence_cache_gap_summary(cache_results, cache_only_skipped).items():
            existing = dict(by_gap.get(gap_id) or {})
            merged = {
                **existing,
                **{key: value for key, value in gap_summary.items() if value not in (None, "", [], {})},
            }
            for counter_key in ("cache_hit_count", "live_refresh_required_count", "cache_only_skip_count"):
                merged[counter_key] = int(_safe_float(existing.get(counter_key), 0.0)) + int(_safe_float(gap_summary.get(counter_key), 0.0))
            by_gap[gap_id] = merged
        live_required_count = len([item for item in cache_results if item.get("live_refresh_required")])
        cache_only_count = len(cache_only_skipped)
        summary["evidence_hit"] = int(_safe_float(summary.get("evidence_hit"), 0.0)) + len(cache_results)
        summary["cache_seed_count"] = int(_safe_float(summary.get("cache_seed_count"), 0.0)) + len(cache_results)
        summary["live_refresh_required_count"] = int(_safe_float(summary.get("live_refresh_required_count"), 0.0)) + live_required_count
        summary["cache_only_skip_count"] = int(_safe_float(summary.get("cache_only_skip_count"), 0.0)) + cache_only_count
        summary["skipped_network_tasks"] = int(_safe_float(summary.get("skipped_network_tasks"), 0.0)) + cache_only_count
        if by_gap:
            summary["by_gap"] = by_gap
        metadata["evidence_cache_summary"] = summary
        state["metadata"] = metadata
    return cache_results, remaining, cache_only_skipped


def _marker_text(markers: set[str]) -> str:
    return " ".join(sorted(markers))


def _has_evidence_followup_hint(item: Dict[str, Any], markers: Optional[set[str]] = None) -> bool:
    markers = markers if markers is not None else _followup_marker_values(item)
    text = _marker_text(markers)
    if any(hint in markers or hint in text for hint in EVIDENCE_FOLLOWUP_HINTS):
        return True
    if _as_list(item.get("source_priority")) or _as_list(item.get("required_evidence_mix")):
        return True
    return bool(item.get("proof_role") or item.get("hypothesis_id") or item.get("hypothesis_statement"))


def _normalize_followup_payload(item: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(_as_dict(item))
    nested = payload.get("follow_up_query")
    if isinstance(nested, dict):
        payload.update({key: value for key, value in nested.items() if value not in (None, "", [])})
    elif str(nested or "").strip() and not str(payload.get("query") or payload.get("suggested_query") or "").strip():
        payload["query"] = nested
    query_value = payload.get("query") or payload.get("suggested_query")
    if isinstance(query_value, dict):
        nested_query = _as_dict(query_value)
        payload.update({key: value for key, value in nested_query.items() if value not in (None, "", [])})
        query_value = payload.get("query") or payload.get("suggested_query")
    query = _compact_text(query_value, max_chars=220)
    if not query:
        return {}
    payload["query"] = query
    return payload


def _is_non_evidence_followup(item: Dict[str, Any]) -> bool:
    if not _env_flag("BRAIN_FOLLOWUP_SKIP_NON_EVIDENCE_GAPS", True):
        return False
    markers = _followup_marker_values(item)
    if markers.intersection(NON_EVIDENCE_FOLLOWUP_TYPES):
        return True
    if markers.intersection(SOFT_NON_EVIDENCE_FOLLOWUP_TYPES) and not _has_evidence_followup_hint(item, markers):
        return True
    return False


def _followup_priority(item: Dict[str, Any]) -> int:
    markers = _followup_marker_values(_as_dict(item))
    text = _marker_text(markers)
    if any(marker in markers or marker in text for marker in {"core_claim_without_ab_source"}):
        return 0
    if any(marker in markers or marker in text for marker in {"missing_proof_standard", "missing_proof_standards"}):
        return 0
    if any(marker in markers or marker in text for marker in {"mandatory_proof_missing"}):
        return 0
    if any(marker in markers or marker in text for marker in {"insufficient_ab_sources", "insufficient_ab_core_sources"}):
        return 2
    if any(marker in markers or marker in text for marker in {"metric_scope_period_unit_incomplete", "citation_source_missing", "invalid_citation", "missing_source_ref"}):
        return 4
    if "counter" in text:
        return 8
    if "metric" in text:
        return 10
    if "case" in text:
        return 12
    if any(marker in markers or marker in text for marker in {"needs_corroboration", "corroboration", "source", "proof", "evidence"}):
        return 15
    return 30


def _high_value_repair_dedupe_key(task: Dict[str, Any]) -> str:
    payload = _as_dict(task.get("search_task")) or _as_dict(task)
    query = re.sub(r"\s+", " ", str(payload.get("query") or task.get("query") or "").strip().lower())
    return "|".join(
        [
            str(payload.get("loop_name") or "").strip().lower(),
            _followup_gap_id(payload),
            str(payload.get("hypothesis_id") or "").strip().lower(),
            str(payload.get("chapter_id") or "").strip().lower(),
            str(payload.get("proof_role") or payload.get("evidence_type") or "").strip().lower(),
            query,
        ]
    )


def _task_chapter_key_for_budget(task: Dict[str, Any], index: int) -> str:
    payload = _as_dict(task.get("search_task")) or _as_dict(task)
    return str(
        payload.get("chapter_id")
        or payload.get("hypothesis_id")
        or payload.get("mandatory_proof_id")
        or payload.get("targets_gap")
        or f"task_{index}"
    ).strip().lower()


def _repair_max_tasks_per_chapter() -> int:
    if os.getenv("BRAIN_REPAIR_MAX_TASKS_PER_CHAPTER") is not None:
        return _env_int("BRAIN_REPAIR_MAX_TASKS_PER_CHAPTER", 2, min_value=1, max_value=20)
    mode = str(os.getenv("REPORT_QUALITY_MODE") or os.getenv("QUALITY_MODE") or "balanced").strip().lower()
    if mode in {"speed", "fast", "loose", "draft", "quick_market_scan"}:
        return 1
    if mode in {"strict", "deep", "high", "deep_strict", "due_diligence", "investment_due_diligence"}:
        return 4
    return 2


def _select_high_value_repair_tasks(
    tasks: Sequence[Dict[str, Any]],
    *,
    state: BrainAgentState,
    round_number: int,
) -> List[Dict[str, Any]]:
    if not tasks or not _env_flag("BRAIN_REPAIR_HIGH_VALUE_SELECTION_ENABLED", True):
        return [dict(task) for task in list(tasks or []) if isinstance(task, dict)]
    normalized = [dict(task) for task in list(tasks or []) if isinstance(task, dict)]
    if not normalized:
        return []
    has_post_qa = any(str(_as_dict(_as_dict(task).get("search_task")).get("loop_name") or "").strip().lower() == "post_qa_repair" for task in normalized)
    max_tasks = _env_int(
        "BRAIN_POST_QA_REPAIR_MAX_EVIDENCE_TASKS" if has_post_qa else "BRAIN_REPAIR_MAX_TASKS_PER_ROUND",
        8 if has_post_qa else 10,
        min_value=1,
        max_value=80,
    )
    per_chapter = _repair_max_tasks_per_chapter()
    sorted_tasks = sorted(
        normalized,
        key=lambda task: (
            _followup_priority(_as_dict(task.get("search_task")) or _as_dict(task)),
            _followup_target_key(_as_dict(task.get("search_task")) or _as_dict(task)),
            str(task.get("agent") or ""),
            str(task.get("query") or ""),
        ),
    )
    selected: List[Dict[str, Any]] = []
    seen: set[str] = set()
    by_chapter: Dict[str, int] = {}
    skipped_duplicate = 0
    skipped_budget = 0
    skipped_per_chapter = 0
    for index, task in enumerate(sorted_tasks):
        payload = _as_dict(task.get("search_task")) or _as_dict(task)
        if _is_non_evidence_followup(payload):
            skipped_budget += 1
            continue
        key = _high_value_repair_dedupe_key(task)
        if key in seen:
            skipped_duplicate += 1
            continue
        chapter_key = _task_chapter_key_for_budget(task, index)
        if by_chapter.get(chapter_key, 0) >= per_chapter:
            skipped_per_chapter += 1
            continue
        if len(selected) >= max_tasks:
            skipped_budget += 1
            continue
        seen.add(key)
        by_chapter[chapter_key] = by_chapter.get(chapter_key, 0) + 1
        selected.append(task)
    if len(selected) != len(normalized):
        metadata = _state_metadata(state)
        summary = dict(metadata.get("repair_task_selection_summary") or {})
        summary.update(
            {
                "round_number": round_number,
                "before_count": len(normalized),
                "after_count": len(selected),
                "max_tasks": max_tasks,
                "max_tasks_per_chapter": per_chapter,
                "skipped_duplicate_count": skipped_duplicate,
                "skipped_budget_count": skipped_budget,
                "skipped_per_chapter_count": skipped_per_chapter,
            }
        )
        metadata["repair_task_selection_summary"] = summary
        _progress("followup", "高价值补证任务筛选", before=len(normalized), after=len(selected), round=round_number)
    return selected


def _followup_target_key(item: Dict[str, Any]) -> str:
    for key in (
        "mandatory_proof_id",
        "proof_id",
        "hypothesis_id",
        "evidence_goal_id",
        "task_id",
        "targets_gap",
        "dimension_name",
        "dimension",
        "type",
        "reason",
    ):
        value = _compact_text(item.get(key), max_chars=120)
        if value:
            return re.sub(r"\s+", " ", value).lower()
    return re.sub(r"\s+", " ", _compact_text(item.get("query"), max_chars=120)).lower()


def _followup_query_key(item: Dict[str, Any]) -> str:
    query = re.sub(r"\s+", " ", str(item.get("query") or "").strip()).lower()
    agent = str(item.get("agent") or "").strip().lower()
    if agent == "all":
        agent = "both"
    target = re.sub(r"\s+", " ", str(item.get("targets_gap") or item.get("dimension_name") or "").strip()).lower()
    gap_id = _followup_gap_id(item)
    requirement_id = str(item.get("requirement_id") or "").strip().lower()
    lanes = ",".join(
        sorted(
            {
                str(lane or "").strip().lower()
                for lane in _as_list(item.get("lane_targets"))
                if str(lane or "").strip()
            }
        )
    )
    proof_role = str(item.get("proof_role") or item.get("evidence_type") or "").strip().lower()
    required_fields = ",".join(
        sorted(
            {
                str(field or "").strip().lower()
                for field in _as_list(item.get("required_fields"))
                if str(field or "").strip()
            }
        )
    )
    return f"{agent}|{requirement_id}|{gap_id}|{target}|{proof_role}|{required_fields}|{lanes}|{query}"


def _gap_attempt_key(task: Dict[str, Any]) -> str:
    payload = _as_dict(task.get("search_task")) or _as_dict(task)
    return (
        _followup_gap_id(payload)
        or str(payload.get("hypothesis_id") or "").strip().lower()
        or str(payload.get("chapter_id") or "").strip().lower()
        or _followup_target_key(payload)
        or re.sub(r"\s+", " ", str(payload.get("query") or "").strip().lower())[:160]
    )


def _gap_attempt_summary_state(state: BrainAgentState) -> Dict[str, Any]:
    metadata = _state_metadata(state)
    summary = dict(metadata.get("gap_attempt_summary") or {})
    summary.setdefault("max_attempts_per_gap", _env_int("BRAIN_GAP_MAX_ATTEMPTS", 3, min_value=1, max_value=20))
    summary.setdefault("max_consecutive_no_gain_rounds", _env_int("BRAIN_GAP_MAX_NO_GAIN_ROUNDS", 2, min_value=1, max_value=10))
    summary.setdefault("gaps", {})
    metadata["gap_attempt_summary"] = summary
    return summary


def _filter_exhausted_gap_tasks(tasks: Sequence[Dict[str, Any]], *, state: BrainAgentState) -> List[Dict[str, Any]]:
    if not _env_flag("BRAIN_GAP_FUSE_ENABLED", True):
        return [dict(task) for task in list(tasks or []) if isinstance(task, dict)]
    summary = _gap_attempt_summary_state(state)
    max_attempts = int(_safe_float(summary.get("max_attempts_per_gap"), 3.0))
    max_no_gain = int(_safe_float(summary.get("max_consecutive_no_gain_rounds"), 2.0))
    gaps = dict(_as_dict(summary.get("gaps")))
    selected: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for task in list(tasks or []):
        if not isinstance(task, dict):
            continue
        key = _gap_attempt_key(task)
        item = dict(_as_dict(gaps.get(key)))
        attempts = int(_safe_float(item.get("attempt_count"), 0.0))
        no_gain = int(_safe_float(item.get("consecutive_no_gain_rounds"), 0.0))
        exhausted = bool(item.get("exhausted")) or attempts >= max_attempts or no_gain >= max_no_gain
        if exhausted:
            item.update(
                {
                    "gap_key": key,
                    "exhausted": True,
                    "exhausted_reason": "max_attempts" if attempts >= max_attempts else "consecutive_no_quality_gain",
                    "skipped_after_exhausted_count": int(_safe_float(item.get("skipped_after_exhausted_count"), 0.0)) + 1,
                }
            )
            gaps[key] = item
            skipped.append(task)
            continue
        selected.append(dict(task))
    if skipped:
        summary["evidence_exhausted"] = True
        summary["skipped_exhausted_task_count"] = int(_safe_float(summary.get("skipped_exhausted_task_count"), 0.0)) + len(skipped)
        summary["exhausted_gap_count"] = len([item for item in gaps.values() if _as_dict(item).get("exhausted")])
        summary["gaps"] = gaps
        _state_metadata(state)["gap_attempt_summary"] = summary
        _progress("followup", "补证熔断跳过已耗尽 gap", skipped=len(skipped), remaining=len(selected))
    return selected


def _source_ab_count_from_result(result: Dict[str, Any]) -> int:
    seen: set[str] = set()
    for source in _as_list(result.get("key_sources")) + _as_list(result.get("search_results")) + _as_list(result.get("page_results")):
        item = _as_dict(source)
        if not item:
            continue
        level = _source_level_from_source(item)
        if level not in {"A", "B"}:
            continue
        key = _payload_source_key(item)
        if key:
            seen.add(key)
    for point in _as_list(result.get("raw_data_points")):
        item = _as_dict(point)
        level = str(item.get("source_level") or item.get("level") or "").strip().upper()
        if level not in {"A", "B"}:
            continue
        key = str(item.get("source_url") or item.get("url") or item.get("source") or "").strip().lower()
        if key:
            seen.add(key)
    return len(seen)


def _readpage_success_count_from_result(result: Dict[str, Any]) -> int:
    count = 0
    for page in _as_list(result.get("page_results")):
        item = _as_dict(page)
        if str(item.get("content") or item.get("markdown") or item.get("text") or item.get("summary") or "").strip():
            count += 1
    return count


def _record_gap_attempts(tasks: Sequence[Dict[str, Any]], *, state: BrainAgentState, round_number: int) -> None:
    if not _env_flag("BRAIN_GAP_FUSE_ENABLED", True):
        return
    summary = _gap_attempt_summary_state(state)
    gaps = dict(_as_dict(summary.get("gaps")))
    for task in list(tasks or []):
        if not isinstance(task, dict):
            continue
        key = _gap_attempt_key(task)
        item = dict(_as_dict(gaps.get(key)))
        item["gap_key"] = key
        item["attempt_count"] = int(_safe_float(item.get("attempt_count"), 0.0)) + 1
        item["last_round"] = round_number
        item["last_query"] = _as_dict(task.get("search_task")).get("query") or task.get("query")
        item["agents"] = _unique_strings([*_as_list(item.get("agents")), task.get("agent")], max_items=12)
        gaps[key] = item
    summary["gaps"] = gaps
    summary["attempted_task_count"] = int(_safe_float(summary.get("attempted_task_count"), 0.0)) + len([task for task in tasks if isinstance(task, dict)])
    _state_metadata(state)["gap_attempt_summary"] = summary


def _record_gap_attempt_results(results: Sequence[Dict[str, Any]], *, state: BrainAgentState) -> None:
    if not _env_flag("BRAIN_GAP_FUSE_ENABLED", True):
        return
    summary = _gap_attempt_summary_state(state)
    max_attempts = int(_safe_float(summary.get("max_attempts_per_gap"), 3.0))
    max_no_gain = int(_safe_float(summary.get("max_consecutive_no_gain_rounds"), 2.0))
    gaps = dict(_as_dict(summary.get("gaps")))
    for result in list(results or []):
        if not isinstance(result, dict):
            continue
        task = _as_dict(result.get("search_task")) or result
        key = _gap_attempt_key(task)
        item = dict(_as_dict(gaps.get(key)))
        ab_count = _source_ab_count_from_result(result)
        readpage_count = _readpage_success_count_from_result(result)
        signal_score = _followup_result_signal_score(result)
        quality_gain = bool(ab_count > 0 or readpage_count > 0 or signal_score >= 3)
        item["gap_key"] = key
        item["result_count"] = int(_safe_float(item.get("result_count"), 0.0)) + 1
        item["signal_count"] = int(_safe_float(item.get("signal_count"), 0.0)) + (1 if signal_score >= 2 else 0)
        item["ab_source_count"] = int(_safe_float(item.get("ab_source_count"), 0.0)) + ab_count
        item["readpage_success_count"] = int(_safe_float(item.get("readpage_success_count"), 0.0)) + readpage_count
        item["quality_gain_count"] = int(_safe_float(item.get("quality_gain_count"), 0.0)) + (1 if quality_gain else 0)
        item["consecutive_no_gain_rounds"] = 0 if quality_gain else int(_safe_float(item.get("consecutive_no_gain_rounds"), 0.0)) + 1
        attempts = int(_safe_float(item.get("attempt_count"), 0.0))
        if attempts >= max_attempts or int(_safe_float(item.get("consecutive_no_gain_rounds"), 0.0)) >= max_no_gain:
            item["exhausted"] = True
            item["exhausted_reason"] = "max_attempts" if attempts >= max_attempts else "consecutive_no_quality_gain"
        gaps[key] = item
    exhausted_count = len([item for item in gaps.values() if _as_dict(item).get("exhausted")])
    summary["gaps"] = gaps
    summary["exhausted_gap_count"] = exhausted_count
    summary["evidence_exhausted"] = bool(exhausted_count)
    _state_metadata(state)["gap_attempt_summary"] = summary


def _dedupe_followups(
    followups: Sequence[Dict[str, Any]],
    seen_keys: set[str],
) -> tuple[List[Dict[str, Any]], int]:
    deduped: List[Dict[str, Any]] = []
    skipped = 0
    for item in followups:
        if not isinstance(item, dict):
            continue
        key = _followup_query_key(item)
        if not key or key in seen_keys:
            skipped += 1
            continue
        seen_keys.add(key)
        deduped.append(item)
    return deduped, skipped


def _repair_contract_reject_if(proof_role: Any, required_fields: Sequence[Any]) -> List[str]:
    role = str(proof_role or "").strip().lower()
    fields = {str(item or "").strip().lower() for item in _as_list(list(required_fields)) if str(item or "").strip()}
    reject = ["snippet_only", "no_source_url", "marketing_copy_only"]
    if role in {"metric", "source_check", "filing"} or {"period", "date", "source"} & fields:
        reject.append("no_date")
    if role == "metric" or {"metric", "value", "unit", "period"} & fields:
        reject.extend(["missing_metric_value", "missing_unit", "missing_period"])
    if role == "counter":
        reject.append("support_only_counter_missing")
    return _unique_strings(reject, max_items=12)


def _repair_contract_success_criteria(proof_role: Any, required_fields: Sequence[Any]) -> str:
    role = str(proof_role or "").strip().lower()
    fields = [str(item or "").strip() for item in _as_list(list(required_fields)) if str(item or "").strip()]
    if role == "metric" or {"metric", "value", "unit", "period", "source"}.issubset({item.lower() for item in fields}):
        return "Only count as repaired when metric/value/unit/period/source are all present and traceable to the page source."
    if role == "counter":
        return "Only count as repaired when the result provides traceable counter/risk evidence rather than support-only evidence."
    if role in {"source_check", "filing"}:
        return "Only count as repaired when an authoritative original source, filing, announcement, or research source is traceable by URL."
    if fields:
        return f"Only count as repaired when required fields are present: {', '.join(fields)}."
    return "Only count as repaired when the missing evidence can be traced to a concrete source URL."


def _repair_contract_source_patterns(proof_role: Any, lane_targets: Sequence[Any]) -> List[str]:
    role = str(proof_role or "").strip().lower()
    patterns = _unique_strings([str(item or "").strip() for item in _as_list(list(lane_targets))], max_items=8)
    if role == "metric":
        patterns.extend(["official_data", "market_research", "survey", "pdf", "annual_report"])
    elif role in {"source_check", "filing"}:
        patterns.extend(["official_data", "filing_company", "exchange_announcement", "investor_relations"])
    elif role == "counter":
        patterns.extend(["counter_evidence", "failure", "cost", "roi_unclear", "security", "compliance"])
    elif role == "case":
        patterns.extend(["customer_case", "company_disclosure", "procurement", "filing_company"])
    else:
        patterns.extend(["market_research", "official_data"])
    return _unique_strings(patterns, max_items=10)


def _repair_payload_from_gap_ledger(item: Dict[str, Any]) -> Dict[str, Any]:
    gap = dict(_as_dict(item))
    gap_type = str(gap.get("gap_type") or gap.get("type") or "").strip()
    role = str(gap.get("required_proof_role") or gap.get("proof_role") or gap.get("evidence_type") or "source_check").strip()
    terms = _as_list(gap.get("query_terms")) or _as_list(gap.get("topic_terms"))
    if not terms:
        terms = _compact_iqs_terms(
            [
                gap.get("chapter_title"),
                gap.get("chapter_id"),
                gap.get("why_current_evidence_insufficient"),
                gap_type,
            ],
            max_terms=6,
            max_chars=18,
        )
    role_terms: List[str] = []
    if gap_type == "metric_scope_period_unit_incomplete" or role == "metric":
        role_terms = ["指标", "数值", "单位", "周期", "来源"]
        role = "metric"
    elif gap_type in {"insufficient_ab_sources", "core_claim_without_ab_source", "source_trace_missing", "citation_source_missing"}:
        role_terms = ["官方", "原文", "披露", "来源"]
        role = role or "source_check"
    elif gap_type == "counter_evidence_missing" or role == "counter":
        role_terms = ["风险", "延期", "失败", "监管", "下调"]
        role = "counter"
    elif gap_type == "mandatory_proof_missing":
        role_terms = ["原文", "统计", "标准", "专利", "公司披露"]
    query = _compose_iqs_query([terms, role_terms, _as_list(gap.get("source_priority"))], max_chars=96)
    if not query:
        query = _post_qa_repair_query_from_item({**gap, "proof_role": role, "topic_terms": terms})
    required_fields = _as_list(gap.get("required_fields")) or _required_fields_for_proof_role(role)
    lane_targets = _as_list(gap.get("lane_targets"))
    payload = {
        **gap,
        "schema_version": gap.get("schema_version") or "repair_task_seed_v2",
        "type": gap_type or gap.get("type") or "evidence_gap",
        "query": query,
        "suggested_query": query,
        "targets_gap": gap.get("targets_gap") or gap.get("why_current_evidence_insufficient") or gap_type,
        "evidence_goal": gap.get("evidence_goal") or gap.get("why_current_evidence_insufficient") or gap_type,
        "dimension_name": gap.get("dimension_name") or gap.get("chapter_title") or gap.get("chapter_id"),
        "hypothesis_id": gap.get("hypothesis_id") or gap.get("chapter_id"),
        "blocking_gaps": _unique_strings([gap_type, *_as_list(gap.get("blocking_gaps"))], max_items=6),
        "proof_role": role or "source_check",
        "evidence_type": gap.get("evidence_type") or role or "source_check",
        "lane_targets": lane_targets,
        "required_fields": required_fields,
        "required_source_level": _as_list(gap.get("required_source_level") or gap.get("min_source_level")),
        "success_criteria": gap.get("success_criteria") or _repair_contract_success_criteria(role, required_fields),
        "reject_if": _as_list(gap.get("reject_if")) or _repair_contract_reject_if(role, required_fields),
        "preferred_source_patterns": _as_list(gap.get("preferred_source_patterns")) or _repair_contract_source_patterns(role, lane_targets),
        "freshness_required": bool(gap.get("freshness_required") or gap.get("live_refresh_required")),
        "max_cache_age_hours": gap.get("max_cache_age_hours"),
        "source": gap.get("source") or "evidence_gap_ledger",
        "agent": gap.get("agent") or "iqs",
        "allowed_for_writing": False,
    }
    return payload


def _repair_task_from_item(
    item: Dict[str, Any],
    *,
    origin_node: str,
    loop_name: str,
) -> Dict[str, Any]:
    raw_item = _as_dict(item)
    if raw_item.get("gap_type") and not str(raw_item.get("query") or raw_item.get("suggested_query") or "").strip():
        raw_item = _repair_payload_from_gap_ledger(raw_item)
    payload = _normalize_followup_payload(raw_item)
    if not payload or _is_non_evidence_followup(payload):
        return {}
    normalized = _post_qa_repair_followup_payload(payload) or payload
    query = _compact_text(normalized.get("query") or payload.get("query"), max_chars=220)
    if not query:
        return {}
    task = {
        **normalized,
        "schema_version": normalized.get("schema_version") or payload.get("schema_version") or "repair_task_seed_v2",
        "query": query,
        "gap_id": normalized.get("gap_id") or _followup_gap_id(normalized) or _followup_target_key(normalized),
        "origin_node": origin_node,
        "loop_name": loop_name,
        "hypothesis_id": normalized.get("hypothesis_id") or payload.get("hypothesis_id") or payload.get("mandatory_proof_id"),
        "blocking_gaps": _as_list(normalized.get("blocking_gaps")) or _as_list(payload.get("blocking_gaps")),
        "proof_role": normalized.get("proof_role") or payload.get("proof_role") or payload.get("evidence_type") or "source_check",
        "lane_targets": _as_list(normalized.get("lane_targets")),
        "required_fields": _as_list(normalized.get("required_fields")),
        "required_source_level": _as_list(normalized.get("required_source_level") or payload.get("required_source_level")),
        "success_criteria": normalized.get("success_criteria") or payload.get("success_criteria") or "",
        "reject_if": _as_list(normalized.get("reject_if") or payload.get("reject_if")),
        "preferred_source_patterns": _as_list(normalized.get("preferred_source_patterns") or payload.get("preferred_source_patterns")),
        "freshness_required": bool(normalized.get("freshness_required") or payload.get("freshness_required")),
        "max_cache_age_hours": normalized.get("max_cache_age_hours") or payload.get("max_cache_age_hours"),
        "agent": normalized.get("agent") or payload.get("agent") or "iqs",
    }
    if not task["required_fields"] and str(task.get("proof_role") or "").strip().lower() == "metric":
        task["required_fields"] = ["metric", "value", "unit", "period", "source"]
    if not task["success_criteria"]:
        task["success_criteria"] = _repair_contract_success_criteria(task.get("proof_role"), task["required_fields"])
    if not task["reject_if"]:
        task["reject_if"] = _repair_contract_reject_if(task.get("proof_role"), task["required_fields"])
    if not task["preferred_source_patterns"]:
        task["preferred_source_patterns"] = _repair_contract_source_patterns(task.get("proof_role"), task["lane_targets"])
    return dispatch_repair_seed(task, failed_queries=_as_list(task.get("failed_queries") or task.get("avoid_queries")))


def _repair_seen_keys_for_state(state: Dict[str, Any]) -> set[str]:
    runtime = state.get("_repair_runtime")
    if not isinstance(runtime, dict):
        runtime = {}
        state["_repair_runtime"] = runtime
    seen = runtime.get("seen_followup_keys")
    if isinstance(seen, set):
        return seen
    if isinstance(seen, list):
        seen_set = {str(item) for item in seen if str(item or "").strip()}
    else:
        seen_set = set()
    runtime["seen_followup_keys"] = seen_set
    return seen_set


REPAIR_CONTEXT_SEED_ALLOWED_KEYS = {
    "schema_version",
    "query",
    "suggested_query",
    "agent",
    "task_id",
    "gap_id",
    "requirement_id",
    "chapter_id",
    "section_id",
    "gap_type",
    "type",
    "repair_status",
    "proof_role",
    "evidence_type",
    "required_fields",
    "required_source_level",
    "lane_targets",
    "source_priority",
    "success_criteria",
    "reject_if",
    "preferred_source_patterns",
    "freshness_required",
    "max_cache_age_hours",
    "blocking_gaps",
    "targets_gap",
    "evidence_goal",
    "repair_route",
    "repair_priority_score",
    "repair_priority_reason",
    "source_stage",
    "source",
    "cache_seed_available",
    "live_refresh_required",
    "avoid_repeating_failed_query",
    "previous_result_count",
    "previous_signal_count",
    "cache_hit_count",
    "cache_lookup_key",
    "cache_scope",
    "allowed_for_writing",
}


def _sanitize_repair_context_seed(seed: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        key: value
        for key, value in dict(_as_dict(seed)).items()
        if key in REPAIR_CONTEXT_SEED_ALLOWED_KEYS and value not in (None, "", [])
    }
    query = _compact_text(payload.get("query") or payload.get("suggested_query"), max_chars=220)
    if not query:
        return {}
    payload["query"] = query
    payload["source"] = payload.get("source") or "repair_context_view"
    payload["agent"] = payload.get("agent") or "iqs"
    payload["allowed_for_writing"] = False
    return payload


def _repair_tasks_from_context_view(
    view: Dict[str, Any],
    *,
    origin_node: str,
    loop_name: str,
    max_tasks: int,
    seen_keys: Optional[set[str]] = None,
) -> tuple[List[Dict[str, Any]], int]:
    if max_tasks <= 0:
        return [], 0
    schedule_tasks = [
        item
        for item in _as_list(_as_dict(_as_dict(view).get("search_task_schedule")).get("tasks"))
        if isinstance(item, dict)
    ]
    source_items = schedule_tasks or [
        item
        for item in _as_list(_as_dict(view).get("repair_task_seeds"))
        if isinstance(item, dict)
    ]
    seeds = [
        _sanitize_repair_context_seed(item)
        for item in source_items
    ]
    return _repair_tasks_from_items(
        [item for item in seeds if item],
        origin_node=origin_node,
        loop_name=loop_name,
        max_tasks=max_tasks,
        seen_keys=seen_keys,
    )


def _repair_context_run_id_from_state(state: Dict[str, Any]) -> str:
    return str(
        state.get("artifact_ledger_run_id")
        or state.get("stage_snapshot_run_id")
        or os.getenv("REPORT_STAGE_SNAPSHOT_RUN_ID")
        or ""
    ).strip()


def _sync_evidence_gap_ledger_to_artifact_ledger(
    *,
    state: Dict[str, Any],
    evidence_package: Dict[str, Any],
    structured_analysis: Optional[Dict[str, Any]] = None,
    writer_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not _env_flag("BRAIN_SYNC_EVIDENCE_GAPS_TO_LEDGER", True):
        return {"status": "disabled", "reason": "gap_ledger_sync_disabled"}
    run_id = _repair_context_run_id_from_state(state)
    if not run_id:
        return {"status": "disabled", "reason": "missing_run_id"}
    package = _as_dict(evidence_package)
    gaps = [item for item in _as_list(package.get("evidence_gap_ledger")) if isinstance(item, dict)]
    if not gaps:
        return {"status": "skipped", "reason": "no_evidence_gap_ledger"}
    try:
        from rag_pipeline.cache.artifact_pipeline_bridge import ingest_writer_package_artifacts
        from rag_pipeline.cache.artifact_store import default_artifact_store

        store = default_artifact_store()
        if not store.enabled():
            return {"status": "disabled", "reason": "artifact_ledger_disabled"}
        query = str(state.get("query") or package.get("query") or "").strip()
        research_plan = _research_plan_from_state(state) or _as_dict(package.get("research_plan"))
        store.upsert_run(run_id=run_id, query=query, report_type="full_report", status="running")
        package_payload = {**package, "evidence_gap_ledger": gaps}
        structured_payload = dict(_as_dict(structured_analysis))
        if research_plan:
            package_payload.setdefault("research_plan", research_plan)
            package_payload.setdefault("metadata", {})
            package_payload["metadata"] = {**_as_dict(package_payload.get("metadata")), "research_plan": research_plan}
            structured_payload.setdefault("research_plan", research_plan)
        writer_package = {
            "query": query,
            "research_plan": research_plan,
            "evidence_package": package_payload,
            "structured_analysis": structured_payload,
            "source_registry": _as_list(package.get("source_registry")) or _as_list(package.get("sources")),
        }
        summary = ingest_writer_package_artifacts(
            store,
            run_id=run_id,
            writer_package=writer_package,
            writer_report=_as_dict(writer_report),
        )
        return {"status": "synced", **_as_dict(summary)}
    except Exception as exc:  # pragma: no cover - ledger sync is an optional accelerator.
        return {"status": "error", "reason": "gap_ledger_sync_failed", "error": str(exc)}


def _sync_writer_artifacts_to_artifact_ledger(
    *,
    state: Dict[str, Any],
    evidence_package: Dict[str, Any],
    structured_analysis: Optional[Dict[str, Any]] = None,
    writer_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not _env_flag("BRAIN_SYNC_WRITER_ARTIFACTS_TO_LEDGER", True):
        return {"status": "disabled", "reason": "writer_artifact_sync_disabled"}
    run_id = _repair_context_run_id_from_state(state)
    if not run_id:
        return {"status": "disabled", "reason": "missing_run_id"}
    report = _as_dict(writer_report)
    if not report:
        return {"status": "skipped", "reason": "missing_writer_report"}
    writer_fields = _writer_pipeline_state_fields(report)
    if not (_as_list(writer_fields.get("argument_units")) or _as_list(writer_fields.get("chapter_packages"))):
        return {"status": "skipped", "reason": "missing_writer_claim_section_artifacts"}
    try:
        from rag_pipeline.cache.artifact_pipeline_bridge import ingest_writer_package_artifacts
        from rag_pipeline.cache.artifact_store import default_artifact_store

        store = default_artifact_store()
        if not store.enabled():
            return {"status": "disabled", "reason": "artifact_ledger_disabled"}
        package = _as_dict(evidence_package)
        query = str(state.get("query") or package.get("query") or "").strip()
        research_plan = _research_plan_from_state(state) or _as_dict(package.get("research_plan"))
        store.upsert_run(run_id=run_id, query=query, report_type="full_report", status="running")
        writer_package = {
            "query": query,
            "research_plan": research_plan,
            "evidence_package": package,
            "structured_analysis": _as_dict(structured_analysis),
            "source_registry": _as_list(report.get("source_registry"))
            or _as_list(package.get("source_registry"))
            or _as_list(package.get("sources")),
            **writer_fields,
            "writer_report": report,
        }
        summary = ingest_writer_package_artifacts(
            store,
            run_id=run_id,
            writer_package=writer_package,
            writer_report=report,
        )
        return {"status": "synced", **_as_dict(summary)}
    except Exception as exc:  # pragma: no cover - optional ledger sync must not block writer.
        return {"status": "error", "reason": "writer_artifact_sync_failed", "error": str(exc)}


def _ledger_repair_items_from_state(
    *,
    state: Dict[str, Any],
    max_tasks: int,
    seen_keys: Optional[set[str]],
) -> tuple[List[Dict[str, Any]], Dict[str, Any], int]:
    if not _env_flag("BRAIN_ENABLE_LEDGER_REPAIR_CONTEXT", True):
        return [], {"status": "disabled", "reason": "ledger_repair_context_disabled"}, 0
    run_id = _repair_context_run_id_from_state(state)
    if not run_id:
        return [], {"status": "disabled", "reason": "missing_run_id"}, 0
    try:
        from rag_pipeline.context.context_view_builder import build_repair_context_view

        view = build_repair_context_view(run_id)
        tasks, skipped = _repair_tasks_from_context_view(
            view,
            origin_node="artifact_ledger",
            loop_name="ledger_repair",
            max_tasks=max_tasks,
            seen_keys=seen_keys,
        )
        return tasks, _as_dict(view), skipped
    except Exception as exc:  # pragma: no cover - repair context is an optional accelerator.
        return [], {"status": "error", "reason": "repair_context_view_failed", "error": str(exc)}, 0


def _repair_tasks_from_items(
    items: Sequence[Dict[str, Any]],
    *,
    origin_node: str,
    loop_name: str,
    max_tasks: int,
    seen_keys: Optional[set[str]] = None,
) -> tuple[List[Dict[str, Any]], int]:
    if max_tasks <= 0:
        return [], 0
    seen = seen_keys if seen_keys is not None else set()
    tasks: List[Dict[str, Any]] = []
    skipped = 0
    for item in list(items or []):
        if not isinstance(item, dict):
            continue
        task = _repair_task_from_item(item, origin_node=origin_node, loop_name=loop_name)
        if not task:
            skipped += 1
            continue
        key = _followup_query_key(task)
        if key in seen:
            skipped += 1
            continue
        seen.add(key)
        tasks.append(task)
        if len(tasks) >= max_tasks:
            break
    return tasks, skipped


def _repair_task_summary(tasks: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_loop: Dict[str, int] = {}
    by_origin: Dict[str, int] = {}
    by_role: Dict[str, int] = {}
    by_lane: Dict[str, int] = {}
    deep_task_count = 0
    deep_skipped_count = 0
    deep_budget_exhausted_count = 0
    for task in list(tasks or []):
        item = _as_dict(task)
        if item.get("search_task"):
            item = _as_dict(item.get("search_task"))
        loop = str(item.get("loop_name") or "unknown")
        origin = str(item.get("origin_node") or "unknown")
        role = str(item.get("proof_role") or "source_check")
        by_loop[loop] = by_loop.get(loop, 0) + 1
        by_origin[origin] = by_origin.get(origin, 0) + 1
        by_role[role] = by_role.get(role, 0) + 1
        if item.get("prefer_deep"):
            deep_task_count += 1
        elif item.get("deep_skip_reason"):
            deep_skipped_count += 1
            if "budget" in str(item.get("deep_skip_reason") or ""):
                deep_budget_exhausted_count += 1
        for lane in _as_list(item.get("lane_targets")) or [item.get("agent") or "iqs"]:
            key = str(lane or "").strip() or "iqs"
            by_lane[key] = by_lane.get(key, 0) + 1
    return {
        "task_count": len([item for item in list(tasks or []) if isinstance(item, dict)]),
        "by_loop": by_loop,
        "by_origin": by_origin,
        "by_proof_role": by_role,
        "by_lane": by_lane,
        "deep_task_count": deep_task_count,
        "deep_skipped_count": deep_skipped_count,
        "deep_budget_exhausted_count": deep_budget_exhausted_count,
    }


def _post_qa_repair_needed(writer_report: Dict[str, Any], *, respect_env: bool = True) -> bool:
    if respect_env and not _env_flag("BRAIN_ENABLE_POST_QA_REPAIR", False):
        return False
    report = _as_dict(writer_report)
    if str(report.get("report_status") or "").strip().lower() in {"final", "final_clean"}:
        return False
    reflection = _research_reflection_memo_from_writer_report(report)
    if _as_list(reflection.get("next_search_task_seeds")):
        return True
    qa = _as_dict(report.get("qa_result")) or _as_dict(report.get("validation"))
    render_gate = _as_dict(qa.get("render_gate"))
    if _as_list(render_gate.get("blockers")):
        return True
    deep = _as_dict(qa.get("deep_evaluation"))
    return any(
        [
            bool(_as_list(report.get("required_followups"))),
            bool(_as_list(qa.get("repair_followups"))),
            bool(_as_list(qa.get("evidence_repair_followups"))),
            bool(_as_list(qa.get("content_repair_followups"))),
            bool(_as_list(deep.get("required_followups"))),
            bool(_as_list(report.get("review_evidence_followups"))),
            bool(_as_list(report.get("review_logic_issues"))),
        ]
    )


def _research_reflection_memo_from_writer_report(writer_report: Dict[str, Any]) -> Dict[str, Any]:
    report = _as_dict(writer_report)
    structured = _as_dict(report.get("structured_analysis"))
    insight = _as_dict(report.get("report_insight_package")) or _as_dict(structured.get("report_insight_package"))
    render_artifacts = _as_dict(report.get("render_artifacts"))
    render_structured = _as_dict(render_artifacts.get("structured_analysis"))
    return (
        _as_dict(report.get("research_reflection_memo"))
        or _as_dict(insight.get("research_reflection_memo"))
        or _as_dict(structured.get("research_reflection_memo"))
        or _as_dict(render_artifacts.get("research_reflection_memo"))
        or _as_dict(render_structured.get("research_reflection_memo"))
        or _as_dict(_as_dict(render_structured.get("report_insight_package")).get("research_reflection_memo"))
    )


def _post_qa_repair_topic_terms(payload: Dict[str, Any]) -> List[str]:
    explicit = _as_list(payload.get("topic_terms")) or _as_list(payload.get("must_have_terms"))
    terms: List[Any] = list(explicit)
    seed_text = " ".join(
        str(payload.get(key) or "")
        for key in (
            "hypothesis_statement",
            "targets_gap",
            "evidence_goal",
            "dimension_name",
            "query",
            "suggested_query",
        )
    )
    if seed_text:
        terms.extend(_topic_seed_terms(seed_text))
    lower = seed_text.lower()
    if any(token in seed_text for token in ("折叠屏", "折叠", "柔性屏")) or "fold" in lower:
        terms.extend(["折叠屏", "iPhone Fold", "铰链", "柔性OLED"])
    if any(token in seed_text for token in ("苹果", "iPhone")) or "apple" in lower:
        terms.extend(["苹果", "iPhone"])
    if any(token in seed_text for token in ("瓶颈", "技术", "良率")):
        terms.extend(["技术瓶颈", "良率"])
    if not terms:
        terms.append(payload.get("type") or "核心证据")
    return _compact_iqs_terms(terms, max_terms=6, max_chars=18)


def _post_qa_repair_hypothesis_terms(payload: Dict[str, Any]) -> List[str]:
    hypothesis_id = str(payload.get("hypothesis_id") or payload.get("chapter_id") or "").strip().upper()
    seed_text = " ".join(
        str(payload.get(key) or "")
        for key in (
            "hypothesis_statement",
            "targets_gap",
            "evidence_goal",
            "dimension_name",
            "query",
            "suggested_query",
        )
    )
    lower = seed_text.lower()
    terms: List[str] = []
    if hypothesis_id in {"H1", "CH_01", "CH01"} or any(token in seed_text for token in ("需求", "出货", "渗透", "换机")):
        terms.extend(["需求", "出货量", "渗透率", "换机", "IDC", "Counterpoint", "DSCC"])
    if hypothesis_id in {"H2", "CH_02", "CH02"} or any(token in seed_text for token in ("价格", "产能", "订单", "盈利", "利润")):
        terms.extend(["价格", "产能", "订单", "利润率", "供应商", "财报", "公告"])
    if hypothesis_id in {"H3", "CH_03", "CH03"} or any(token in seed_text for token in ("商业化", "量产", "验证", "客户", "认证")):
        terms.extend(["量产验证", "客户认证", "商业化", "供应链", "良率", "技术标准"])
    if hypothesis_id in {"H4", "CH_04", "CH04"} or any(token in seed_text for token in ("反证", "风险", "放弃", "边界")):
        terms.extend(["反证", "风险事件", "延期", "失败案例", "放弃条件"])
    if any(token in seed_text for token in ("折叠屏", "折叠", "柔性屏")) or "fold" in lower:
        terms.extend(["折叠屏", "iPhone Fold", "铰链", "UTG"])
    if any(token in seed_text for token in ("苹果", "iPhone")) or "apple" in lower:
        terms.extend(["苹果", "iPhone"])
    return _compact_iqs_terms(terms, max_terms=8, max_chars=18)


def _post_qa_repair_gap_terms(payload: Dict[str, Any]) -> List[str]:
    markers = _followup_marker_values(payload)
    text = _marker_text(markers)
    terms: List[str] = []
    if "insufficient_ab_sources" in markers or "insufficient_ab_core_sources" in markers:
        terms.extend(["A/B来源", "官方", "公告", "财报", "权威研报"])
    if "metric_scope_period_unit_incomplete" in markers or "metric" in text:
        terms.extend(["指标口径", "周期", "单位", "数据"])
    if "mandatory_proof_missing" in markers or "proof" in text:
        terms.extend(["原文", "协会", "统计", "专利", "标准", "公司披露"])
    if "counter" in text:
        terms.extend(["反证", "风险"])
    if not terms and str(payload.get("type") or "") == "missing_proof_standard":
        terms.extend(["A/B来源", "指标口径", "官方", "公告"])
    return _compact_iqs_terms(terms, max_terms=8, max_chars=14)


def _repair_role_query_terms(payload: Dict[str, Any]) -> List[str]:
    role = str(payload.get("proof_role") or payload.get("evidence_type") or "").strip().lower()
    markers = _followup_marker_values(payload)
    terms: List[str] = []
    if role == "metric" or "metric_scope_period_unit_incomplete" in markers:
        terms.extend(
            [
                str(payload.get("metric_name") or payload.get("metric") or "").strip(),
                str(payload.get("period") or payload.get("time_period") or "").strip(),
                str(payload.get("unit") or "").strip(),
                "指标",
                "周期",
                "单位",
                "来源",
            ]
        )
    elif role in {"filing", "company_filing"}:
        terms.extend(["年报", "公告", "财报", "交易所", "披露"])
    elif role in {"source_check", "official_data"} or markers.intersection({"insufficient_ab_sources", "core_claim_without_ab_source", "citation_source_missing"}):
        terms.extend(["官方", "原文", "披露", "来源"])
    elif role in {"technology_product", "technical", "product"}:
        terms.extend(["良率", "量产", "专利", "标准", "白皮书", "供应商"])
    elif "counter" in _marker_text(markers):
        terms.extend(["失败", "延期", "监管", "事故", "下调", "诉讼", "风险事件"])
    return _compact_iqs_terms(terms, max_terms=8, max_chars=16)


def _post_qa_repair_query_from_item(item: Dict[str, Any]) -> str:
    payload = _as_dict(item)
    markers = _followup_marker_values(payload)
    role_terms = _repair_role_query_terms(payload)
    should_rebuild = bool(
        str(payload.get("type") or "") == "missing_proof_standard"
        or role_terms
        or markers.intersection(
            {
                "insufficient_ab_sources",
                "insufficient_ab_core_sources",
                "metric_scope_period_unit_incomplete",
                "mandatory_proof_missing",
                "core_claim_without_ab_source",
                "citation_source_missing",
            }
        )
    )
    if should_rebuild:
        return _compose_iqs_query(
            [
                _post_qa_repair_topic_terms(payload),
                _post_qa_repair_hypothesis_terms(payload),
                role_terms or _post_qa_repair_gap_terms(payload),
                _as_list(payload.get("source_priority")),
            ],
            max_chars=96,
        )
    if str(payload.get("suggested_query") or "").strip():
        return _compose_iqs_query([_topic_seed_terms(str(payload.get("suggested_query") or ""))], max_chars=96)
    return _compose_iqs_query(
        [
            _post_qa_repair_topic_terms(payload),
            _post_qa_repair_hypothesis_terms(payload),
            _post_qa_repair_gap_terms(payload),
        ],
        max_chars=96,
    )


def _post_qa_repair_followup_payload(item: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(_as_dict(item))
    if payload.get("gap_type") and not str(payload.get("query") or payload.get("suggested_query") or "").strip():
        payload = _repair_payload_from_gap_ledger(payload)
    markers = _followup_marker_values(payload)
    preserve_structured_gap_query = (
        str(payload.get("source") or "") == "evidence_gap_ledger"
        and str(payload.get("query") or "").strip()
    )
    if not preserve_structured_gap_query and (
        str(payload.get("type") or "") == "missing_proof_standard" or markers.intersection(
        {"insufficient_ab_sources", "insufficient_ab_core_sources", "metric_scope_period_unit_incomplete", "mandatory_proof_missing"}
        )
    ):
        payload["query"] = _post_qa_repair_query_from_item(payload)
    elif not preserve_structured_gap_query and not str(payload.get("query") or payload.get("suggested_query") or "").strip():
        payload["query"] = _post_qa_repair_query_from_item(payload)
    normalized = _normalize_followup_payload(payload)
    if not normalized or _is_non_evidence_followup(normalized):
        return {}
    target = (
        normalized.get("hypothesis_statement")
        or normalized.get("targets_gap")
        or normalized.get("evidence_goal")
        or normalized.get("dimension_name")
        or normalized.get("type")
    )
    normalized.setdefault("agent", "iqs")
    normalized.setdefault("targets_gap", _compact_text(target, max_chars=100))
    normalized.setdefault("evidence_goal", _compact_text(target, max_chars=140))
    normalized.setdefault("dimension_name", _compact_text(target, max_chars=100))
    normalized.setdefault("topic_terms", _post_qa_repair_topic_terms(normalized))
    if str(payload.get("source") or "") == "evidence_gap_ledger" and _as_list(payload.get("query_terms")):
        normalized["topic_terms"] = _as_list(payload.get("query_terms"))
    needs_ab_sources = bool(markers.intersection({"insufficient_ab_sources", "insufficient_ab_core_sources"}))
    needs_metric_fields = bool(markers.intersection({"metric_scope_period_unit_incomplete"}))
    needs_mandatory_proof = str(normalized.get("type") or "") == "missing_proof_standard" or bool(markers.intersection({"mandatory_proof_missing"}))
    if needs_ab_sources:
        normalized["lane_targets"] = ["official_data", "filing_company", "market_research"]
        normalized["source_priority"] = ["官方", "公告", "财报", "协会", "统计", "权威研报"]
        normalized.setdefault("proof_role", "source_check")
        normalized.setdefault("evidence_type", "source_check")
    elif needs_metric_fields:
        normalized["lane_targets"] = ["official_data", "market_research", "filing_company"]
        normalized["source_priority"] = ["官方", "统计", "协会", "财报", "公告", "权威研报"]
        normalized.setdefault("proof_role", "metric")
        normalized.setdefault("evidence_type", "market_data")
    elif needs_mandatory_proof and not _as_list(normalized.get("source_priority")):
        normalized["source_priority"] = ["官方", "公告", "财报", "协会", "统计", "权威研报"]
        normalized.setdefault("lane_targets", ["official_data", "filing_company", "technology_product", "market_research"])
    if needs_metric_fields:
        normalized["required_fields"] = _unique_strings(
            [*_as_list(normalized.get("required_fields")), "metric", "period", "unit", "source"],
            max_items=8,
        )
        normalized.setdefault("proof_role", "metric")
    if needs_mandatory_proof and "technology_product" not in _as_list(normalized.get("lane_targets")):
        if any(term in " ".join(_as_list(normalized.get("topic_terms")) + [_compact_text(normalized.get("query"), max_chars=160)]) for term in ("技术", "专利", "铰链", "折叠屏", "良率", "UTG")):
            normalized["lane_targets"] = _unique_strings([*_as_list(normalized.get("lane_targets")), "technology_product"], max_items=5)
    return normalized


def _post_qa_repair_sort_key(item: Dict[str, Any]) -> tuple:
    type_name = str(_as_dict(item).get("type") or "").strip()
    dropped_penalty = 20 if type_name == "search_tasks_dropped" else 0
    return (
        _followup_priority(item) + dropped_penalty,
        _followup_target_key(item),
        str(item.get("query") or ""),
    )


def _post_qa_repair_plan(writer_report: Dict[str, Any], *, max_queries: int) -> Dict[str, Any]:
    report = _as_dict(writer_report)
    qa = _as_dict(report.get("qa_result")) or _as_dict(report.get("validation"))
    deep = _as_dict(qa.get("deep_evaluation"))
    evidence_candidates: List[Dict[str, Any]] = []
    rewrite_reasons: List[Dict[str, Any]] = []
    skipped_non_evidence: List[Dict[str, Any]] = []

    def add_candidate(item: Any, *, default_source: str) -> None:
        payload = dict(_as_dict(item))
        if not payload:
            return
        payload.setdefault("source", default_source)
        markers = _followup_marker_values(payload)
        if markers.intersection(NON_EVIDENCE_FOLLOWUP_TYPES):
            rewrite_reasons.append(
                {
                    "type": payload.get("type") or default_source,
                    "source": default_source,
                    "required": payload.get("required"),
                    "actual": payload.get("actual"),
                    "priority": payload.get("priority"),
                }
            )
            skipped_non_evidence.append({"type": payload.get("type") or default_source, "source": default_source})
            return
        normalized = _post_qa_repair_followup_payload(payload)
        if normalized:
            evidence_candidates.append(normalized)
        elif payload.get("type") or payload.get("reason"):
            rewrite_reasons.append({"type": payload.get("type") or payload.get("reason"), "source": default_source})

    for item in _as_list(report.get("evidence_gap_ledger")):
        payload = _as_dict(item)
        route = str(payload.get("repair_route") or "").strip()
        gap_type = str(payload.get("gap_type") or payload.get("type") or "").strip()
        if route in {"rewrite", "claim_rebuild"} or gap_type == "evidence_available_but_not_bound":
            rewrite_reasons.append(
                {
                    "type": gap_type or route or "evidence_gap",
                    "source": "evidence_gap_ledger",
                    "repair_route": route or "claim_rebuild",
                    "chapter_id": payload.get("chapter_id"),
                }
            )
            continue
        add_candidate(payload, default_source="evidence_gap_ledger")
    for item in _as_list(report.get("required_followups")):
        add_candidate(item, default_source="writer_required_followups")
    for item in _as_list(deep.get("required_followups")):
        add_candidate(item, default_source="qa_required_followups")
    for item in _as_list(qa.get("evidence_repair_followups")):
        add_candidate(item, default_source="qa_evidence_repair_followups")
    for item in _as_list(qa.get("repair_followups")):
        add_candidate(item, default_source="qa_repair_followups")
    for item in _as_list(report.get("review_evidence_followups")):
        add_candidate(item, default_source="review_evidence_followups")
    for item in _as_list(qa.get("content_repair_followups")):
        payload = _as_dict(item)
        rewrite_reasons.append({"type": payload.get("type") or "content_repair", "source": "qa_content_repair_followups"})
    for item in _as_list(report.get("review_logic_issues")):
        payload = _as_dict(item)
        rewrite_reasons.append({"type": payload.get("type") or "review_logic_issue", "source": "review_logic_issues"})

    evidence_candidates.sort(key=_post_qa_repair_sort_key)
    deduped, skipped_duplicates = _dedupe_followups(evidence_candidates, set())
    evidence_followups = deduped[: max(0, int(max_queries or 0))]
    return {
        "status": "planned" if evidence_followups or rewrite_reasons else "no_repair_tasks",
        "evidence_followups": evidence_followups,
        "rewrite_required": bool(rewrite_reasons),
        "rewrite_reasons": rewrite_reasons[:12],
        "skipped_duplicate_followups": skipped_duplicates + max(0, len(deduped) - len(evidence_followups)),
        "skipped_non_evidence": skipped_non_evidence[:12],
        "max_queries": max_queries,
    }


def _attach_post_qa_repair(writer_report: Dict[str, Any], trace: Dict[str, Any]) -> Dict[str, Any]:
    copied = dict(_as_dict(writer_report))
    copied["post_qa_repair"] = {
        "status": trace.get("status"),
        "enabled": bool(trace.get("enabled")),
        "evidence_followup_count": len(_as_list(_as_dict(trace.get("plan")).get("evidence_followups"))),
        "rewrite_required": bool(_as_dict(trace.get("plan")).get("rewrite_required")),
        "has_signal": trace.get("has_signal"),
        "signal_count": trace.get("signal_count"),
        "empty_success_count": trace.get("empty_success_count"),
        "failed_count": trace.get("failed_count"),
        "no_signal_reasons": _as_list(trace.get("no_signal_reasons"))[:8],
        "is_best": trace.get("is_best"),
        "stop_reason": trace.get("stop_reason"),
    }
    return copied


def _post_qa_repair_context(plan: Dict[str, Any], trace: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "source": "post_qa_repair",
        "status": trace.get("status") or "running",
        "rewrite_required": bool(plan.get("rewrite_required")),
        "rewrite_reasons": _as_list(plan.get("rewrite_reasons")),
        "evidence_followups": _as_list(plan.get("evidence_followups")),
        "instruction": "Use the repaired evidence and rewrite weak sections. Do not publish unsupported claims.",
    }


def _preflight_repair_items_from_binder(binder_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    plan = _as_dict(binder_result.get("evidence_refinement_plan"))
    items = [item for item in _as_list(plan.get("top_priorities")) if isinstance(item, dict)]
    if len(items) < 6:
        existing = {_followup_query_key(item) for item in items}
        for item in _as_list(plan.get("follow_up_queries")):
            if not isinstance(item, dict):
                continue
            key = _followup_query_key(item)
            if key in existing:
                continue
            existing.add(key)
            items.append(item)
            if len(items) >= 12:
                break
    if items:
        return items
    for proof in _as_list(binder_result.get("missing_proof_standards")):
        payload = _as_dict(_as_dict(proof).get("follow_up_query"))
        if payload:
            items.append(payload)
    return items


def _run_evidence_preflight_round(
    *,
    state: BrainAgentState,
    children: Dict[str, Dict[str, Any]],
    evidence_pool: Sequence[Dict[str, Any]],
    evidence_package: Dict[str, Any],
    structured_analysis: Dict[str, Any],
    report_plan: Dict[str, Any],
    query: str,
    max_followups: int,
    started: float,
) -> Dict[str, Any]:
    trace: Dict[str, Any] = {
        "round": "evidence_preflight",
        "source": "evidence_binder",
        "enabled": _env_flag("BRAIN_ENABLE_EVIDENCE_PREFLIGHT_LOOP", True),
        "status": "not_started",
    }
    current_pool = [item for item in list(evidence_pool or []) if isinstance(item, dict)]
    current_package = _as_dict(evidence_package)
    if not trace["enabled"]:
        trace["status"] = "disabled"
        trace["stop_reason"] = "evidence_preflight_disabled"
        return {
            "evidence_pool": current_pool,
            "evidence_package": current_package,
            "evidence_preflight_trace": [trace],
            "updated": False,
        }
    try:
        from .evidence_binder import run_evidence_binder
    except Exception as exc:  # pragma: no cover - optional bridge.
        trace["status"] = "binder_unavailable"
        trace["stop_reason"] = "binder_import_failed"
        trace["error"] = str(exc)
        return {
            "evidence_pool": current_pool,
            "evidence_package": current_package,
            "evidence_preflight_trace": [trace],
            "updated": False,
        }
    try:
        binder_result = run_evidence_binder(
            research_plan=_research_plan_from_state(state),
            report_blueprint=_as_dict(state.get("report_blueprint"))
            or _as_dict(_as_dict(state.get("query_analysis")).get("report_blueprint")),
            evidence_package=current_package,
            structured_analysis=_as_dict(structured_analysis),
            child_outputs=children,
            evidence_pool=current_pool,
        )
    except Exception as exc:  # pragma: no cover - defensive preflight.
        logger.exception("Evidence preflight binder failed", extra={"query": query})
        trace["status"] = "binder_failed"
        trace["stop_reason"] = "binder_failed"
        trace["error"] = str(exc)
        return {
            "evidence_pool": current_pool,
            "evidence_package": current_package,
            "evidence_preflight_trace": [trace],
            "updated": False,
        }

    binder_plan = _as_dict(binder_result.get("evidence_refinement_plan"))
    package_gap_ledger = [
        item
        for item in _as_list(current_package.get("evidence_gap_ledger"))
        if isinstance(item, dict) and str(item.get("repair_route") or "evidence_search") == "evidence_search"
    ]
    package_gap_ledger.sort(key=_post_qa_repair_sort_key)
    ledger_gap_sync = _sync_evidence_gap_ledger_to_artifact_ledger(
        state=state,
        evidence_package=current_package,
        structured_analysis=_as_dict(structured_analysis),
    )
    max_tasks = max(1, min(max_followups, _env_int("BRAIN_EVIDENCE_PREFLIGHT_MAX_FOLLOWUPS", 6, min_value=1, max_value=12)))
    seen_keys = _repair_seen_keys_for_state(state)
    ledger_tasks, ledger_view, ledger_skipped = _ledger_repair_items_from_state(
        state=state,
        max_tasks=max_tasks,
        seen_keys=seen_keys,
    )
    ledger_tasks = [
        {
            **_as_dict(task),
            "origin_node": _as_dict(task).get("origin_node") or "artifact_ledger",
            "loop_name": _as_dict(task).get("loop_name") or "ledger_repair",
        }
        for task in ledger_tasks
        if isinstance(task, dict)
    ]
    raw_items = [*package_gap_ledger, *_preflight_repair_items_from_binder(binder_result)]
    remaining_task_budget = max(0, max_tasks - len(ledger_tasks))
    if remaining_task_budget > 0:
        binder_tasks, binder_skipped = _repair_tasks_from_items(
            raw_items,
            origin_node="evidence_binder",
            loop_name="evidence_preflight",
            max_tasks=remaining_task_budget,
            seen_keys=seen_keys,
        )
    else:
        binder_tasks, binder_skipped = [], 0
    tasks = [*ledger_tasks, *binder_tasks]
    skipped = ledger_skipped + binder_skipped
    trace.update(
        {
            "binder_status": binder_plan.get("status"),
            "coverage_gap_counts": _as_dict(binder_plan.get("gap_counts")),
            "missing_proof_count": len(_as_list(binder_result.get("missing_proof_standards"))),
            "evidence_gap_ledger_count": len(package_gap_ledger),
            "ledger_gap_sync": ledger_gap_sync,
            "ledger_repair_view_status": _as_dict(ledger_view).get("status"),
            "ledger_repair_seed_count": len(ledger_tasks),
            "ledger_repair_skipped_count": ledger_skipped,
            "attempted_task_count": len(tasks),
            "repair_task_summary": _repair_task_summary(tasks),
            "skipped_task_count": skipped,
            "follow_up_queries": tasks,
        }
    )
    if not tasks:
        trace["status"] = "no_tasks"
        trace["stop_reason"] = "no_preflight_repair_tasks"
        return {
            "evidence_pool": current_pool,
            "evidence_package": current_package,
            "evidence_preflight_trace": [trace],
            "updated": False,
        }

    before_package = current_package
    before_pool_size = len(current_pool)
    state["evidence_package"] = current_package
    _progress("writer", "Evidence preflight 补证开始", followups=len(tasks))
    followup_results = run_followup_queries(follow_up_queries=tasks, round_number=1, state=state)
    usable_results = _substantive_followup_results(followup_results)
    result_summary = _repair_result_summary(followup_results, usable_results=usable_results)
    trace.update(
        {
            "repair_task_summary": _repair_task_summary_after_policy(tasks, followup_results),
            "repair_result_summary": result_summary,
            "signal_count": result_summary.get("signal_count"),
            "empty_success_count": result_summary.get("empty_success_count"),
            "failed_count": result_summary.get("failed_count"),
            "new_usable_evidence_count": result_summary.get("new_usable_evidence_count"),
            "new_ab_source_count": result_summary.get("new_ab_source_count"),
            "gap_ledger": _gap_ledger_from_followups(tasks, followup_results),
            "followup_results": [_trace_followup_result(item) for item in followup_results if isinstance(item, dict)],
        }
    )
    if not usable_results:
        trace["status"] = "no_signal"
        trace["stop_reason"] = "no_new_evidence_signal"
        trace["evidence_pool_size_before"] = before_pool_size
        trace["evidence_pool_size_after"] = before_pool_size
        _progress("writer", "Evidence preflight 补证停止", reason=trace["stop_reason"], elapsed=f"{time.perf_counter() - started:.1f}s")
        return {
            "evidence_pool": current_pool,
            "evidence_package": current_package,
            "evidence_preflight_trace": [trace],
            "updated": False,
        }

    current_pool.extend(usable_results)
    current_package = merge_evidence_package(
        original_query=query,
        evidence_pool=current_pool,
        children=children,
        research_plan=_research_plan_from_state(state),
    )
    current_package = _annotate_evidence_package_runtime(current_package, lane_coverage=_lane_coverage_from_state(state), state=state)
    state["evidence_package"] = current_package
    if report_plan:
        current_package["report_plan"] = report_plan
        current_package.setdefault("metadata", {})
        current_package["metadata"]["report_plan"] = report_plan
    trace["status"] = "completed"
    trace["stop_reason"] = "one_round_completed"
    trace["evidence_pool_size_before"] = before_pool_size
    trace["evidence_pool_size_after"] = len(current_pool)
    trace["evidence_delta_summary"] = _evidence_delta_summary(before_package, current_package)
    _progress("writer", "Evidence preflight 补证完成", usable=len(usable_results), elapsed=f"{time.perf_counter() - started:.1f}s")
    return {
        "evidence_pool": current_pool,
        "evidence_package": current_package,
        "evidence_preflight_trace": [trace],
        "updated": True,
    }


def _run_post_qa_repair_round(
    *,
    state: BrainAgentState,
    children: Dict[str, Dict[str, Any]],
    best: Dict[str, Any],
    report_plan: Dict[str, Any],
    query: str,
    search_task_schedule: Dict[str, Any],
    lane_coverage: Dict[str, Any],
    max_followups: int,
    started: float,
    respect_env: bool = False,
) -> Dict[str, Any]:
    best_report = _as_dict(best.get("writer_report"))
    trace: Dict[str, Any] = {
        "round": "post_qa",
        "source": "post_qa_repair",
        "enabled": True if not respect_env else _env_flag("BRAIN_ENABLE_POST_QA_REPAIR", False),
        "has_signal": None,
        "quality_before": _writer_quality_snapshot(best_report),
    }
    if not trace["enabled"]:
        trace["status"] = "disabled_by_default"
        trace["stop_reason"] = "disabled_by_default"
        best["writer_report"] = _attach_post_qa_repair(best_report, trace)
        return {**best, "post_qa_repair_trace": [trace]}
    if not _post_qa_repair_needed(best_report, respect_env=respect_env):
        trace["status"] = "not_needed"
        return {**best, "post_qa_repair_trace": [trace]}

    plan = _post_qa_repair_plan(best_report, max_queries=max_followups)
    trace.update({"enabled": True, "plan": plan})
    max_tasks = max(0, int(max_followups or 0))
    seen_keys = _repair_seen_keys_for_state(state)
    writer_artifact_ledger_sync = _sync_writer_artifacts_to_artifact_ledger(
        state=state,
        evidence_package=_as_dict(best.get("evidence_package")),
        structured_analysis=_as_dict(best.get("structured_analysis")),
        writer_report=best_report,
    )
    ledger_followups, ledger_view, ledger_skipped = _ledger_repair_items_from_state(
        state=state,
        max_tasks=max_tasks,
        seen_keys=seen_keys,
    )
    ledger_followups = [
        {
            **_as_dict(task),
            "origin_node": _as_dict(task).get("origin_node") or "artifact_ledger",
            "loop_name": _as_dict(task).get("loop_name") or "ledger_repair",
        }
        for task in ledger_followups
        if isinstance(task, dict)
    ]
    remaining_task_budget = max(0, max_tasks - len(ledger_followups))
    if remaining_task_budget > 0:
        qa_followups, qa_skipped = _repair_tasks_from_items(
            [item for item in _as_list(plan.get("evidence_followups")) if isinstance(item, dict)],
            origin_node="writer_qa",
            loop_name="post_qa_repair",
            max_tasks=remaining_task_budget,
            seen_keys=seen_keys,
        )
    else:
        qa_followups, qa_skipped = [], 0
    evidence_followups = [*ledger_followups, *qa_followups]
    skipped_repair_tasks = ledger_skipped + qa_skipped
    trace["repair_task_summary"] = _repair_task_summary(evidence_followups)
    trace["skipped_repair_task_count"] = skipped_repair_tasks
    trace["writer_artifact_ledger_sync"] = writer_artifact_ledger_sync
    trace["ledger_repair_view_status"] = _as_dict(ledger_view).get("status")
    trace["ledger_repair_seed_count"] = len(ledger_followups)
    trace["ledger_repair_skipped_count"] = ledger_skipped
    rewrite_required = bool(plan.get("rewrite_required"))
    if not evidence_followups and not rewrite_required:
        trace["status"] = "no_repair_tasks"
        trace["stop_reason"] = "no_repair_tasks"
        best["writer_report"] = _attach_post_qa_repair(best_report, trace)
        return {**best, "post_qa_repair_trace": [trace]}

    current_evidence_pool = [item for item in _as_list(best.get("evidence_pool")) if isinstance(item, dict)]
    current_evidence_package = _as_dict(best.get("evidence_package"))
    current_structured_analysis = _as_dict(best.get("structured_analysis"))
    current_analysis_state = _as_dict(best.get("analysis_state"))
    has_signal: Optional[bool] = None
    followup_results: List[Dict[str, Any]] = []
    gap_ledger: List[Dict[str, Any]] = []
    before_blockers = _writer_blocker_snapshot(best_report)
    before_package_for_quality = current_evidence_package

    if evidence_followups:
        before_pool_size = len(current_evidence_pool)
        before_package = current_evidence_package
        _progress("writer", "Post-QA 补证开始", followups=len(evidence_followups))
        followup_results = run_followup_queries(
            follow_up_queries=evidence_followups,
            round_number=max(1, int(_layout_refinement_round_count(_as_list(best.get("layout_refinement_trace"))) or 0) + 1),
            state=state,
        )
        has_signal = _followup_result_has_signal(followup_results)
        usable_followup_results = _substantive_followup_results(followup_results)
        repair_result_summary = _repair_result_summary(followup_results, usable_results=usable_followup_results)
        gap_ledger = _gap_ledger_from_followups(evidence_followups, followup_results)
        trace.update(
            {
                "repair_task_summary": _repair_task_summary_after_policy(evidence_followups, followup_results),
                "evidence_pool_size_before": before_pool_size,
                "attempted_task_count": len(evidence_followups),
                "followup_results": [_trace_followup_result(item) for item in followup_results if isinstance(item, dict)],
                "gap_ledger": gap_ledger,
                "has_signal": has_signal,
                "repair_result_summary": repair_result_summary,
                "signal_count": repair_result_summary.get("signal_count"),
                "empty_success_count": repair_result_summary.get("empty_success_count"),
                "failed_count": repair_result_summary.get("failed_count"),
                "partial_count": repair_result_summary.get("partial_count"),
                "new_usable_evidence_count": repair_result_summary.get("new_usable_evidence_count"),
                "new_ab_source_count": repair_result_summary.get("new_ab_source_count"),
                "no_signal_reasons": repair_result_summary.get("no_signal_reasons"),
            }
        )
        if not usable_followup_results:
            trace["status"] = "no_new_evidence_signal"
            trace["stop_reason"] = "no_new_evidence_signal"
            best["writer_report"] = _attach_post_qa_repair(best_report, trace)
            _progress("writer", "Post-QA 补证停止", reason=trace["stop_reason"], elapsed=f"{time.perf_counter() - started:.1f}s")
            return {**best, "post_qa_repair_trace": [trace]}

        current_evidence_pool.extend(usable_followup_results)
        current_evidence_package = merge_evidence_package(
            original_query=query,
            evidence_pool=current_evidence_pool,
            children=children,
            research_plan=_research_plan_from_state(state),
        )
        current_evidence_package = _annotate_evidence_package_runtime(current_evidence_package, lane_coverage=lane_coverage, state=state)
        if report_plan:
            current_evidence_package["report_plan"] = report_plan
            current_evidence_package.setdefault("metadata", {})
            current_evidence_package["metadata"]["report_plan"] = report_plan
        try:
            current_analysis_state = run_analysis_agent(current_evidence_package, query=query, llm_config=build_llm_config("decision"))
        except TypeError as exc:
            if "llm_config" not in str(exc):
                raise
            current_analysis_state = run_analysis_agent(current_evidence_package, query=query)
        current_structured_analysis = _as_dict(current_analysis_state.get("structured_analysis"))
        _attach_report_plan(current_evidence_package, current_structured_analysis, report_plan)
        _attach_research_plan(current_evidence_package, current_structured_analysis, _research_plan_from_state(state))
        _sync_analysis_repair_priorities_to_evidence_package(current_evidence_package, current_structured_analysis)
        trace["evidence_pool_size_after"] = len(current_evidence_pool)
        trace["evidence_delta_summary"] = _evidence_delta_summary(before_package, current_evidence_package)

    repaired_structured_analysis = copy.deepcopy(current_structured_analysis)
    repaired_report_plan = copy.deepcopy(report_plan)
    repair_context = _post_qa_repair_context(plan, trace)
    repaired_structured_analysis["post_qa_repair_context"] = repair_context
    if isinstance(repaired_report_plan, dict):
        repaired_report_plan["post_qa_repair_context"] = repair_context
    claim_rebuild_context = _claim_rebuild_context_from_package(
        reason="ab_evidence_available_but_not_bound",
        writer_report=best_report,
        evidence_package=current_evidence_package,
    )
    if claim_rebuild_context:
        repaired_structured_analysis["claim_rebuild_context"] = claim_rebuild_context
        if isinstance(repaired_report_plan, dict):
            repaired_report_plan["claim_rebuild_context"] = claim_rebuild_context

    current_writer_state = run_writer_agent(
        query=query,
        child_outputs=children,
        evidence_pool=current_evidence_pool,
        evidence_package=current_evidence_package,
        structured_analysis=repaired_structured_analysis,
        report_plan=repaired_report_plan,
        report_blueprint=_as_dict(state.get("report_blueprint")),
        search_task_schedule=search_task_schedule,
        lane_coverage=lane_coverage,
    )
    current_writer_report = _attach_lane_health_to_writer_report(
        _as_dict(current_writer_state.get("writer_report")),
        lane_coverage=lane_coverage,
        state=state,
    )
    current_writer_report = _attach_reformatter_preflight_feedback(
        query=query,
        writer_report=current_writer_report,
        evidence_package=current_evidence_package,
        chapter_evidence_packages=_as_list(current_evidence_package.get("chapter_evidence_packages"))
        or _as_list(current_writer_report.get("chapter_evidence_packages")),
    )
    improved_best = _writer_quality_key(current_writer_report) > _writer_quality_key(best_report)
    after_blockers = _writer_blocker_snapshot(current_writer_report)
    repair_quality_gain = _repair_quality_gain(
        best_report,
        current_writer_report,
        before_package_for_quality,
        current_evidence_package,
    )
    trace.update(
        {
            "status": "completed" if str(current_writer_report.get("report_status") or "").strip().lower() in {"final", "final_clean"} else "still_requires_review",
            "quality_after": _writer_quality_snapshot(current_writer_report),
            "blocker_before": before_blockers,
            "blocker_after": after_blockers,
            "blocker_delta": _blocker_delta(before_blockers, after_blockers),
            "repair_quality_gain": repair_quality_gain,
            "claim_rebuild_context": claim_rebuild_context,
            "is_best": improved_best,
        }
    )
    if improved_best:
        current_writer_report = _attach_post_qa_repair(current_writer_report, trace)
        _progress("writer", "Post-QA 补正完成", status=trace["status"], best=True, elapsed=f"{time.perf_counter() - started:.1f}s")
        return {
            "writer_state": current_writer_state,
            "writer_report": current_writer_report,
            "evidence_pool": list(current_evidence_pool),
            "evidence_package": current_evidence_package,
            "structured_analysis": repaired_structured_analysis,
            "analysis_state": current_analysis_state,
            "layout_refinement_trace": _as_list(best.get("layout_refinement_trace")),
            "evidence_preflight_trace": _as_list(best.get("evidence_preflight_trace")),
            "initial_writer_report": best.get("initial_writer_report"),
            "post_qa_repair_trace": [trace],
        }

    trace["candidate_not_better_reasons"] = _candidate_not_better_reasons(before_blockers, after_blockers)
    if has_signal and not repair_quality_gain.get("has_quality_gain"):
        trace["status"] = "manual_review"
        trace["manual_review_required"] = True
        trace["stop_reason"] = "signal_found_but_no_quality_gain"
        trace["repair_route"] = repair_quality_gain.get("next_route")
    else:
        trace["stop_reason"] = "candidate_not_better"
    best["writer_report"] = _attach_post_qa_repair(best_report, trace)
    _progress("writer", "Post-QA 补正完成", status=trace["status"], best=False, elapsed=f"{time.perf_counter() - started:.1f}s")
    return {**best, "post_qa_repair_trace": [trace]}


def _layout_refinement_round_count(trace: Sequence[Dict[str, Any]]) -> int:
    return len([item for item in trace if isinstance(item, dict) and isinstance(item.get("round"), int) and int(item.get("round") or 0) > 0])


def _last_loop_item(trace: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    for item in reversed(list(trace or [])):
        if isinstance(item, dict):
            return item
    return {}


def _loop_health_from_trace(trace: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    last = _last_loop_item(trace)
    result_summary = _as_dict(last.get("repair_result_summary"))
    return {
        "status": last.get("status") or last.get("stop_reason") or ("not_run" if not trace else "completed"),
        "attempted_task_count": int(_safe_float(last.get("attempted_task_count"), 0.0)),
        "signal_count": int(_safe_float(last.get("signal_count") or result_summary.get("signal_count"), 0.0)),
        "empty_success_count": int(_safe_float(last.get("empty_success_count") or result_summary.get("empty_success_count"), 0.0)),
        "failed_count": int(_safe_float(last.get("failed_count") or result_summary.get("failed_count"), 0.0)),
        "new_usable_evidence_count": int(_safe_float(last.get("new_usable_evidence_count") or result_summary.get("new_usable_evidence_count"), 0.0)),
        "new_ab_source_count": int(_safe_float(last.get("new_ab_source_count") or result_summary.get("new_ab_source_count"), 0.0)),
        "blocker_delta": _as_dict(last.get("blocker_delta")),
        "repair_quality_gain": _as_dict(last.get("repair_quality_gain")),
        "stop_reason": last.get("stop_reason"),
    }


def build_loop_health_summary(
    *,
    supervisor_trace: Sequence[Dict[str, Any]],
    evidence_preflight_trace: Sequence[Dict[str, Any]],
    layout_refinement_trace: Sequence[Dict[str, Any]],
    post_qa_repair_trace: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "supervisor_coverage": _loop_health_from_trace(supervisor_trace),
        "evidence_preflight": _loop_health_from_trace(evidence_preflight_trace),
        "layout_refinement": _loop_health_from_trace(layout_refinement_trace),
        "post_qa_repair": _loop_health_from_trace(post_qa_repair_trace),
    }


def _attach_reformatter_preflight_feedback(
    *,
    query: str,
    writer_report: Dict[str, Any],
    evidence_package: Dict[str, Any],
    chapter_evidence_packages: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    if not _env_flag("BRAIN_ENABLE_REFORMATTER_PREFLIGHT_LOOP", True):
        return writer_report
    report_markdown = str(_as_dict(writer_report).get("report_markdown") or "").strip()
    if not report_markdown:
        return writer_report
    try:
        from rag_pipeline.flows.report.evidence_extractor import extract_clean_evidence_from_package
        from rag_pipeline.flows.report.reformatter_agent import build_reformatter_repair_plan, validate_reformatted_report
    except Exception as exc:  # pragma: no cover - optional feedback bridge.
        logger.debug("Reformatter preflight imports unavailable", extra={"error": str(exc)})
        return writer_report
    package = {
        "query": query,
        "evidence_package": evidence_package,
        "chapter_evidence_packages": list(chapter_evidence_packages or []),
        "writer_report": writer_report,
    }
    try:
        clean_evidence = extract_clean_evidence_from_package(package)
        validation = validate_reformatted_report(report_markdown, _as_list(clean_evidence.get("sources")), clean_evidence)
        repair_plan = build_reformatter_repair_plan(
            validation,
            clean_evidence,
            topic=query,
            max_queries=_env_int("BRAIN_REFORMATTER_PREFLIGHT_MAX_FOLLOWUPS", 6, min_value=1, max_value=20),
        )
    except Exception as exc:  # pragma: no cover - defensive bridge.
        logger.debug("Reformatter preflight failed", extra={"error": str(exc)})
        return writer_report
    copied = dict(writer_report)
    copied["reformatter_preflight"] = {
        "validation": validation,
        "repair_plan": repair_plan,
    }
    if repair_plan.get("status") == "needs_evidence_refinement":
        existing = [item for item in _as_list(copied.get("required_followups")) if isinstance(item, dict)]
        copied["required_followups"] = [*existing, *[item for item in _as_list(repair_plan.get("follow_up_queries")) if isinstance(item, dict)]]
    return copied


def run_writer_with_layout_refinement(
    *,
    state: BrainAgentState,
    children: Dict[str, Dict[str, Any]],
    evidence_pool: Sequence[Dict[str, Any]],
    evidence_package: Dict[str, Any],
    structured_analysis: Dict[str, Any],
    report_plan: Dict[str, Any],
    analysis_state: Dict[str, Any],
) -> Dict[str, Any]:
    query = str(state.get("query") or "")
    started = time.perf_counter()
    _progress("writer", "Writer 流水线开始", evidence_items=len(list(evidence_pool or [])))
    current_evidence_pool = [item for item in list(evidence_pool or []) if isinstance(item, dict)]
    current_evidence_package = evidence_package
    current_structured_analysis = structured_analysis
    current_analysis_state = analysis_state
    search_task_schedule = _as_dict(state.get("search_task_schedule")) or _as_dict(_as_dict(state.get("query_analysis")).get("search_task_schedule"))
    lane_coverage = _lane_coverage_from_state(state)
    supervisor_followups = int(state.get("supervisor_max_followup_queries") or _env_int("BRAIN_SUPERVISOR_MAX_FOLLOWUP_QUERIES", 4))
    layout_followups = _env_int("BRAIN_LAYOUT_MAX_FOLLOWUP_QUERIES", 6)
    max_followups = max(
        1,
        min(
            20,
            max(supervisor_followups, layout_followups),
        ),
    )
    if _deadline_exceeded(state, min_remaining=1.0):
        preflight_result = {
            "updated": False,
            "evidence_preflight_trace": [
                {
                    "stop_reason": "deadline_exceeded",
                    "live_timeout": _deadline_timeout_payload(state, stage="writer_preflight"),
                }
            ],
        }
    else:
        preflight_result = _run_evidence_preflight_round(
            state=state,
            children=children,
            evidence_pool=current_evidence_pool,
            evidence_package=current_evidence_package,
            structured_analysis=current_structured_analysis,
            report_plan=report_plan,
            query=query,
            max_followups=max_followups,
            started=started,
        )
    evidence_preflight_trace = _as_list(preflight_result.get("evidence_preflight_trace"))
    if preflight_result.get("updated"):
        current_evidence_pool = [item for item in _as_list(preflight_result.get("evidence_pool")) if isinstance(item, dict)]
        current_evidence_package = _as_dict(preflight_result.get("evidence_package")) or current_evidence_package
        current_analysis_state = run_analysis_agent(current_evidence_package, query=query, llm_config=build_llm_config("decision"))
        current_structured_analysis = _as_dict(current_analysis_state.get("structured_analysis"))
        _attach_report_plan(current_evidence_package, current_structured_analysis, report_plan)
        _attach_research_plan(current_evidence_package, current_structured_analysis, _research_plan_from_state(state))
        _sync_analysis_repair_priorities_to_evidence_package(current_evidence_package, current_structured_analysis)
    current_writer_state = run_writer_agent(
        query=query,
        child_outputs=children,
        evidence_pool=current_evidence_pool,
        evidence_package=current_evidence_package,
        structured_analysis=current_structured_analysis,
        report_plan=report_plan,
        report_blueprint=_as_dict(state.get("report_blueprint")),
        search_task_schedule=search_task_schedule,
        lane_coverage=lane_coverage,
    )
    current_writer_report = _attach_lane_health_to_writer_report(
        _as_dict(current_writer_state.get("writer_report")),
        lane_coverage=lane_coverage,
        state=state,
    )
    current_writer_report = _attach_reformatter_preflight_feedback(
        query=query,
        writer_report=current_writer_report,
        evidence_package=current_evidence_package,
        chapter_evidence_packages=_as_list(current_evidence_package.get("chapter_evidence_packages"))
        or _as_list(current_writer_report.get("chapter_evidence_packages")),
    )
    _progress(
        "writer",
        "Writer 首轮完成",
        chars=len(str(current_writer_report.get("report_markdown") or "")),
        qa=_as_dict(current_writer_report.get("qa_result")).get("passed"),
        elapsed=f"{time.perf_counter() - started:.1f}s",
    )
    initial_writer_report = current_writer_report
    enable_layout_followup = (
        bool(state.get("enable_followup_loop", _env_flag("BRAIN_ENABLE_FOLLOWUP_LOOP", True)))
        and _env_flag("BRAIN_ENABLE_LAYOUT_FOLLOWUP_LOOP", True)
    )
    max_rounds = max(0, min(6, _env_int("BRAIN_LAYOUT_MAX_REFINEMENT_ROUNDS", 3)))
    if "layout_max_refinement_rounds" in state:
        try:
            max_rounds = max(0, min(6, int(state.get("layout_max_refinement_rounds") or max_rounds)))
        except Exception:
            logger.exception("Invalid layout_max_refinement_rounds", extra={"value": state.get("layout_max_refinement_rounds")})
    seen_followup_keys = _repair_seen_keys_for_state(state)
    layout_seen_followup_keys: set[str] = set()
    before_seen_followup_keys = set(seen_followup_keys)
    writer_artifact_ledger_sync = _sync_writer_artifacts_to_artifact_ledger(
        state=state,
        evidence_package=current_evidence_package,
        structured_analysis=current_structured_analysis,
        writer_report=current_writer_report,
    )
    ledger_followups, ledger_view, ledger_skipped = _ledger_repair_items_from_state(
        state=state,
        max_tasks=max_followups,
        seen_keys=seen_followup_keys,
    )
    ledger_followups = [
        {
            **_as_dict(task),
            "origin_node": _as_dict(task).get("origin_node") or "artifact_ledger",
            "loop_name": _as_dict(task).get("loop_name") or "ledger_repair",
        }
        for task in ledger_followups
        if isinstance(task, dict)
    ]
    layout_seen_followup_keys.update(seen_followup_keys - before_seen_followup_keys)
    remaining_task_budget = max(0, max_followups - len(ledger_followups))
    followups = _layout_followup_queries_from_writer_report(current_writer_report, max_queries=max_followups * 4)
    if remaining_task_budget > 0:
        before_seen_followup_keys = set(seen_followup_keys)
        layout_followups, layout_skipped_duplicate_followups = _repair_tasks_from_items(
            followups,
            origin_node="writer_layout",
            loop_name="layout_followup",
            max_tasks=remaining_task_budget,
            seen_keys=seen_followup_keys,
        )
        layout_seen_followup_keys.update(seen_followup_keys - before_seen_followup_keys)
    else:
        layout_followups, layout_skipped_duplicate_followups = [], 0
    new_followups = [*ledger_followups, *layout_followups]
    skipped_duplicate_followups = ledger_skipped + layout_skipped_duplicate_followups
    best = {
        "writer_state": current_writer_state,
        "writer_report": current_writer_report,
        "evidence_pool": list(current_evidence_pool),
        "evidence_package": current_evidence_package,
        "structured_analysis": current_structured_analysis,
        "analysis_state": current_analysis_state,
    }
    trace: List[Dict[str, Any]] = [
        {
            "round": 0,
            "source": "layout_planner",
            "quality": _writer_quality_snapshot(current_writer_report),
            "layout_gaps": _as_list(_as_dict(current_writer_report.get("layout_plan")).get("layout_gaps"))[:12],
            "follow_up_queries": new_followups,
            "repair_task_summary": _repair_task_summary(new_followups),
            "skipped_duplicate_followups": skipped_duplicate_followups,
            "writer_artifact_ledger_sync": writer_artifact_ledger_sync,
            "ledger_repair_view_status": _as_dict(ledger_view).get("status"),
            "ledger_repair_seed_count": len(ledger_followups),
            "ledger_repair_skipped_count": ledger_skipped,
            "enabled": enable_layout_followup,
            "max_rounds": max_rounds,
            "max_followups_per_round": max_followups,
        }
    ]
    if _deadline_exceeded(state, min_remaining=1.0):
        trace[0]["stop_reason"] = "deadline_exceeded"
        trace[0]["live_timeout"] = _deadline_timeout_payload(state, stage="writer_layout_refinement")
        _progress("writer", "Layout 补证跳过", reason="deadline_exceeded", elapsed=f"{time.perf_counter() - started:.1f}s")
        return {
            **best,
            "layout_refinement_trace": trace,
            "initial_writer_report": initial_writer_report,
            "evidence_preflight_trace": evidence_preflight_trace,
        }
    if not enable_layout_followup or not new_followups or max_rounds <= 0:
        trace[0]["stop_reason"] = "layout_followup_disabled_or_no_queries"
        _progress("writer", "Layout 补证跳过", reason=trace[0]["stop_reason"], elapsed=f"{time.perf_counter() - started:.1f}s")
        best_with_trace = {
            **best,
            "layout_refinement_trace": trace,
            "initial_writer_report": initial_writer_report,
            "evidence_preflight_trace": evidence_preflight_trace,
        }
        repaired = _run_post_qa_repair_round(
            state=state,
            children=children,
            best=best_with_trace,
            report_plan=report_plan,
            query=query,
            search_task_schedule=search_task_schedule,
            lane_coverage=lane_coverage,
            max_followups=max_followups,
            started=started,
            respect_env=True,
        )
        _progress("writer", "Writer 流水线结束", stop=trace[0]["stop_reason"], elapsed=f"{time.perf_counter() - started:.1f}s")
        return repaired

    pending_followups = new_followups
    stop_reason = "max_rounds_reached"
    for round_number in range(1, max_rounds + 1):
        if _deadline_exceeded(state, min_remaining=1.0):
            trace.append(
                {
                    "round": round_number,
                    "source": "layout_followup",
                    "quality": _writer_quality_snapshot(_as_dict(best.get("writer_report"))),
                    "is_best": False,
                    "attempted_task_count": 0,
                    "follow_up_queries": [],
                    "stop_reason": "deadline_exceeded",
                    "live_timeout": _deadline_timeout_payload(state, stage="writer_layout_followup"),
                }
            )
            stop_reason = "deadline_exceeded"
            break
        round_started = time.perf_counter()
        _progress("writer", "Layout 补证轮次开始", round=round_number, followups=len(pending_followups))
        before_pool_size = len(current_evidence_pool)
        before_package = current_evidence_package
        before_blockers = _writer_blocker_snapshot(_as_dict(best.get("writer_report")))
        active_followups = list(pending_followups)
        followup_results = run_followup_queries(follow_up_queries=active_followups, round_number=round_number, state=state)
        usable_followup_results = _substantive_followup_results(followup_results)
        current_evidence_pool.extend(usable_followup_results)
        has_signal = _followup_result_has_signal(followup_results)
        repair_result_summary = _repair_result_summary(followup_results, usable_results=usable_followup_results)
        gap_ledger = _gap_ledger_from_followups(active_followups, followup_results)
        if not usable_followup_results:
            trace.append(
                {
                    "round": round_number,
                    "source": "layout_followup",
                    "quality": _writer_quality_snapshot(_as_dict(best.get("writer_report"))),
                    "is_best": False,
                    "attempted_task_count": len(active_followups),
                    "evidence_pool_size_before": before_pool_size,
                    "evidence_pool_size_after": len(current_evidence_pool),
                    "has_signal": has_signal,
                    "repair_task_summary": _repair_task_summary_after_policy(active_followups, followup_results),
                    "repair_result_summary": repair_result_summary,
                    "signal_count": repair_result_summary.get("signal_count"),
                    "empty_success_count": repair_result_summary.get("empty_success_count"),
                    "failed_count": repair_result_summary.get("failed_count"),
                    "new_usable_evidence_count": repair_result_summary.get("new_usable_evidence_count"),
                    "new_ab_source_count": repair_result_summary.get("new_ab_source_count"),
                    "attempted_follow_up_queries": active_followups,
                    "follow_up_queries": [],
                    "skipped_duplicate_followups": 0,
                    "gap_ledger": gap_ledger,
                    "followup_results": [_trace_followup_result(item) for item in followup_results if isinstance(item, dict)],
                    "stop_reason": "no_new_evidence_signal",
                }
            )
            stop_reason = "no_new_evidence_signal"
            break
        current_evidence_package = merge_evidence_package(
            original_query=query,
            evidence_pool=current_evidence_pool,
            children=children,
            research_plan=_research_plan_from_state(state),
        )
        current_evidence_package = _annotate_evidence_package_runtime(current_evidence_package, lane_coverage=lane_coverage, state=state)
        if report_plan:
            current_evidence_package["report_plan"] = report_plan
            current_evidence_package.setdefault("metadata", {})
            current_evidence_package["metadata"]["report_plan"] = report_plan
        current_analysis_state = run_analysis_agent(current_evidence_package, query=query, llm_config=build_llm_config("decision"))
        current_structured_analysis = _as_dict(current_analysis_state.get("structured_analysis"))
        _attach_report_plan(current_evidence_package, current_structured_analysis, report_plan)
        _attach_research_plan(current_evidence_package, current_structured_analysis, _research_plan_from_state(state))
        _sync_analysis_repair_priorities_to_evidence_package(current_evidence_package, current_structured_analysis)
        current_writer_state = run_writer_agent(
            query=query,
            child_outputs=children,
            evidence_pool=current_evidence_pool,
            evidence_package=current_evidence_package,
            structured_analysis=current_structured_analysis,
            report_plan=report_plan,
            report_blueprint=_as_dict(state.get("report_blueprint")),
            search_task_schedule=search_task_schedule,
            lane_coverage=lane_coverage,
        )
        current_writer_report = _attach_lane_health_to_writer_report(
            _as_dict(current_writer_state.get("writer_report")),
            lane_coverage=lane_coverage,
            state=state,
        )
        current_writer_report = _attach_reformatter_preflight_feedback(
            query=query,
            writer_report=current_writer_report,
            evidence_package=current_evidence_package,
            chapter_evidence_packages=_as_list(current_evidence_package.get("chapter_evidence_packages"))
            or _as_list(current_writer_report.get("chapter_evidence_packages")),
        )
        after_blockers = _writer_blocker_snapshot(current_writer_report)
        blocker_delta = _blocker_delta(before_blockers, after_blockers)
        repair_quality_gain = _repair_quality_gain(
            _as_dict(best.get("writer_report")),
            current_writer_report,
            before_package,
            current_evidence_package,
        )
        _progress(
            "writer",
            "Layout 补证轮次完成",
            round=round_number,
            evidence_before=before_pool_size,
            evidence_after=len(current_evidence_pool),
            next_followups=len(pending_followups),
            elapsed=f"{time.perf_counter() - round_started:.1f}s",
        )
        improved_best = _writer_quality_key(current_writer_report) > _writer_quality_key(_as_dict(best.get("writer_report")))
        if improved_best:
            best = {
                "writer_state": current_writer_state,
                "writer_report": current_writer_report,
                "evidence_pool": list(current_evidence_pool),
                "evidence_package": current_evidence_package,
                "structured_analysis": current_structured_analysis,
                "analysis_state": current_analysis_state,
            }
        candidate_followups = _layout_followup_queries_from_writer_report(current_writer_report, max_queries=max_followups * 4)
        before_seen_followup_keys = set(seen_followup_keys)
        pending_followups, skipped_duplicate_followups = _repair_tasks_from_items(
            candidate_followups,
            origin_node="writer_layout",
            loop_name="layout_followup",
            max_tasks=max_followups,
            seen_keys=seen_followup_keys,
        )
        layout_seen_followup_keys.update(seen_followup_keys - before_seen_followup_keys)
        trace.append(
            {
                "round": round_number,
                "source": "layout_followup",
                "quality": _writer_quality_snapshot(current_writer_report),
                "is_best": improved_best,
                "attempted_task_count": len(active_followups),
                "evidence_pool_size_before": before_pool_size,
                "evidence_pool_size_after": len(current_evidence_pool),
                "has_signal": has_signal,
                "repair_task_summary": _repair_task_summary_after_policy(active_followups, followup_results),
                "repair_result_summary": repair_result_summary,
                "signal_count": repair_result_summary.get("signal_count"),
                "empty_success_count": repair_result_summary.get("empty_success_count"),
                "failed_count": repair_result_summary.get("failed_count"),
                "new_usable_evidence_count": repair_result_summary.get("new_usable_evidence_count"),
                "new_ab_source_count": repair_result_summary.get("new_ab_source_count"),
                "evidence_delta_summary": _evidence_delta_summary(before_package, current_evidence_package),
                "blocker_before": before_blockers,
                "blocker_after": after_blockers,
                "blocker_delta": blocker_delta,
                "repair_quality_gain": repair_quality_gain,
                "candidate_not_better_reasons": [] if improved_best else _candidate_not_better_reasons(before_blockers, after_blockers),
                "attempted_follow_up_queries": active_followups,
                "follow_up_queries": pending_followups,
                "skipped_duplicate_followups": skipped_duplicate_followups,
                "gap_ledger": gap_ledger,
                "followup_results": [_trace_followup_result(item) for item in followup_results if isinstance(item, dict)],
                "layout_gaps": _as_list(_as_dict(current_writer_report.get("layout_plan")).get("layout_gaps"))[:12],
            }
        )
        if not pending_followups:
            stop_reason = "no_new_layout_followup_queries"
            break
        if not usable_followup_results:
            stop_reason = "no_new_evidence_signal"
            break
        if (not improved_best) and has_signal and not repair_quality_gain.get("has_quality_gain"):
            stop_reason = "signal_found_but_no_quality_gain"
            break

    trace.append(
        {
            "round": "final",
            "source": "layout_refinement_summary",
            "stop_reason": stop_reason,
            "best_quality": _writer_quality_snapshot(_as_dict(best.get("writer_report"))),
            "total_rounds": len([item for item in trace if isinstance(item.get("round"), int) and item.get("round", 0) > 0]),
            "unique_followup_queries": len(layout_seen_followup_keys),
        }
    )
    best_with_trace = {
        **best,
        "layout_refinement_trace": trace,
        "initial_writer_report": initial_writer_report,
        "evidence_preflight_trace": evidence_preflight_trace,
    }
    repaired = _run_post_qa_repair_round(
        state=state,
        children=children,
        best=best_with_trace,
        report_plan=report_plan,
        query=query,
        search_task_schedule=search_task_schedule,
        lane_coverage=lane_coverage,
        max_followups=max_followups,
        started=started,
        respect_env=True,
    )
    _progress("writer", "Writer 流水线结束", stop=stop_reason, elapsed=f"{time.perf_counter() - started:.1f}s")
    return repaired


def aggregate_children_from_evidence_pool(evidence_pool: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    children: Dict[str, Dict[str, Any]] = {}
    agent_pairs: List[tuple[str, str, str]] = [("rag", "industry_rag_agent", "本地 RAG")]
    agent_pairs.extend((key, IQS_ROLE_CONFIGS[key]["child"], IQS_ROLE_CONFIGS[key]["label"]) for key in IQS_ROLE_ORDER)
    agent_pairs.append(("iqs", "web_analysis_agent", "联网 IQS"))
    for agent, child_name, label in agent_pairs:
        items = [item for item in evidence_pool if item.get("agent") == agent]
        usable = [
            item
            for item in items
            if str(item.get("status") or "") in {"success", "partial"}
            and (str(item.get("answer") or "").strip() or item.get("raw_data_points"))
        ]
        if not items:
            children[child_name] = {
                "answer": "",
                "confidence": 0.0,
                "key_sources": [],
                "limitations": {"coverage": "未调度"},
                "status": "failed",
                "used": False,
                "note": "未调度该子 Agent。",
            }
            continue
        if not usable:
            children[child_name] = {
                "answer": "",
                "confidence": 0.0,
                "key_sources": [],
                "limitations": {"rounds": items},
                "status": "failed",
                "used": False,
                "note": "多轮检索均未获得可用结论。",
            }
            continue
        answer_lines = []
        for item in usable:
            round_number = item.get("round") or 1
            query = _compact_text(item.get("query"), max_chars=120)
            if item.get("raw_data_points"):
                answer = _compact_text(
                    "；".join(
                        f"{point.get('metric')}={point.get('value')} {point.get('change') or ''}"
                        for point in list(item.get("raw_data_points") or [])[:8]
                        if isinstance(point, dict)
                    ),
                    max_chars=900,
                )
            else:
                answer = _compact_text(item.get("answer"), max_chars=900)
            answer_lines.append(f"第{round_number}轮｜{query}：{answer}")
        sources: List[Dict[str, Any]] = []
        seen_sources = set()
        for item in usable:
            for source in item.get("key_sources") or []:
                if not isinstance(source, dict):
                    continue
                key = (
                    str(source.get("url") or ""),
                    str(source.get("source_file") or ""),
                    str(source.get("chunk_uid") or ""),
                    str(source.get("title") or ""),
                )
                if key in seen_sources:
                    continue
                seen_sources.add(key)
                copied = dict(source)
                copied.setdefault("round", item.get("round") or 1)
                copied.setdefault("query", item.get("query") or "")
                sources.append(copied)
                if len(sources) >= 30:
                    break
            if len(sources) >= 30:
                break
        confidences = [_clip_confidence(item.get("confidence"), 0.0) for item in usable]
        status = "success" if any(str(item.get("status")) == "success" for item in usable) else "partial"
        children[child_name] = {
            "answer": "\n".join(answer_lines),
            "confidence": round(sum(confidences) / max(len(confidences), 1), 4),
            "key_sources": sources,
            "raw_data_points": [point for item in usable for point in list(item.get("raw_data_points") or []) if isinstance(point, dict)],
            "data_gap": [gap for item in items for gap in list(item.get("data_gap") or []) if isinstance(gap, dict)],
            "limitations": {
                "rounds": len({item.get("round") for item in items}),
                "queries": [item.get("query") for item in items],
                "partial_or_failed": [item for item in items if str(item.get("status")) != "success"],
            },
            "status": status,
            "used": True,
            "note": f"已聚合 {len(usable)} 条{label}证据。",
        }
    role_children = {IQS_ROLE_CONFIGS[key]["child"]: children.get(IQS_ROLE_CONFIGS[key]["child"], {}) for key in IQS_ROLE_ORDER}
    if any(child.get("used") or child.get("status") in {"success", "partial"} for child in role_children.values()):
        children["web_analysis_agent"] = aggregate_iqs_role_children(children)
    return children


def _pool_item_quality(item: Dict[str, Any]) -> float:
    status = str(item.get("status") or "").strip()
    status_score = 3.0 if status == "success" else 1.5 if status == "partial" else 0.0
    confidence = _clip_confidence(item.get("confidence"), 0.0)
    data_bonus = min(1.0, 0.18 * len([point for point in list(item.get("raw_data_points") or []) if isinstance(point, dict)]))
    source_bonus = min(0.7, 0.08 * len([source for source in list(item.get("key_sources") or []) if isinstance(source, dict)]))
    answer_bonus = 0.35 if _has_specific_data(str(item.get("answer") or "")) else 0.0
    recency_bonus = min(0.12, 0.02 * _safe_float(item.get("round"), 1.0))
    return status_score + confidence + data_bonus + source_bonus + answer_bonus + recency_bonus


def _dedupe_dict_items(items: Sequence[Dict[str, Any]], key_fields: Sequence[str], *, max_items: int = 12) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        key = tuple(str(item.get(field) or "") for field in key_fields)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(dict(item))
        if len(deduped) >= max_items:
            break
    return deduped


def _render_iqs_refined_answer(items: Sequence[Dict[str, Any]], *, max_items: int = 4) -> str:
    lines: List[str] = []
    seen = set()
    for item in items:
        answer = _limit_multiline_text(item.get("answer"), max_chars=1100)
        if not answer:
            continue
        answer_lines: List[str] = []
        seen_answer_lines = set()
        for raw_line in answer.splitlines():
            line = raw_line.strip()
            key = _normalize_query_text(line)
            if line and key not in seen_answer_lines:
                seen_answer_lines.add(key)
                answer_lines.append(line)
        answer = "\n".join(answer_lines).strip()
        compact_key = _normalize_query_text(answer[:260])
        if compact_key in seen:
            continue
        seen.add(compact_key)
        round_number = item.get("round") or 1
        query = _compact_text(item.get("query"), max_chars=120)
        lines.append(f"补证结果{len(lines) + 1}｜第{round_number}轮｜{query}\n{answer}")
        if len(lines) >= max_items:
            break
    return "\n\n".join(lines).strip()


def aggregate_best_children_from_evidence_pool(evidence_pool: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    children: Dict[str, Dict[str, Any]] = {}
    agent_pairs: List[tuple[str, str, str]] = [("rag", "industry_rag_agent", "本地 RAG")]
    agent_pairs.extend((key, IQS_ROLE_CONFIGS[key]["child"], IQS_ROLE_CONFIGS[key]["label"]) for key in IQS_ROLE_ORDER)
    agent_pairs.append(("iqs", "web_analysis_agent", "联网 IQS"))
    for agent, child_name, label in agent_pairs:
        items = [item for item in evidence_pool if isinstance(item, dict) and item.get("agent") == agent]
        usable = [
            item
            for item in items
            if str(item.get("status") or "") in {"success", "partial"}
            and (str(item.get("answer") or "").strip() or item.get("raw_data_points"))
        ]
        if not items:
            children[child_name] = {
                "answer": "",
                "confidence": 0.0,
                "key_sources": [],
                "limitations": {"coverage": "未调度"},
                "status": "failed",
                "used": False,
                "note": "未调度该子 Agent。",
                "raw_data_points": [],
                "data_gap": [],
            }
            continue
        if not usable:
            children[child_name] = {
                "answer": "",
                "confidence": 0.0,
                "key_sources": [],
                "limitations": {"rounds": items},
                "status": "failed",
                "used": False,
                "note": f"{label} 多轮自补证后仍未获得可用结论。",
                "raw_data_points": [],
                "data_gap": [gap for item in items for gap in list(item.get("data_gap") or []) if isinstance(gap, dict)],
            }
            continue

        ranked = sorted(usable, key=_pool_item_quality, reverse=True)
        ranked_for_answer = ranked
        if agent == "iqs" or agent in IQS_ROLE_CONFIGS:
            followup_ranked = [
                item
                for item in sorted(usable, key=lambda value: (_safe_float(value.get("round"), 1.0), _pool_item_quality(value)), reverse=True)
                if _safe_float(item.get("round"), 1.0) > 1 and str(item.get("answer") or "").strip()
            ]
            if followup_ranked:
                ranked_for_answer = followup_ranked
        best = ranked_for_answer[0]
        raw_points = _dedupe_dict_items(
            [point for item in ranked for point in list(item.get("raw_data_points") or []) if isinstance(point, dict)],
            ["metric", "value", "period", "source_url"],
            max_items=30,
        )
        sources = _dedupe_dict_items(
            [source for item in ranked for source in list(item.get("key_sources") or []) if isinstance(source, dict)],
            ["url", "source_file", "chunk_uid", "title"],
            max_items=30,
        )
        confidence = max(_clip_confidence(item.get("confidence"), 0.0) for item in usable)
        successful_count = len([item for item in usable if str(item.get("status") or "") == "success"])
        best_round = best.get("round") or 1
        answer = str(best.get("answer") or "").strip()
        if agent == "iqs" or agent in IQS_ROLE_CONFIGS:
            answer = _render_iqs_refined_answer(ranked_for_answer) or answer
        best_gaps = [gap for gap in list(best.get("data_gap") or []) if isinstance(gap, dict)]
        children[child_name] = {
            "answer": answer,
            "confidence": confidence,
            "key_sources": sources,
            "raw_data_points": raw_points,
            "data_gap": best_gaps,
            "limitations": {
                "rounds": len({item.get("round") for item in items}),
                "best_round": best_round,
                "best_query": best.get("query"),
                "queries": [item.get("query") for item in items],
                "partial_or_failed": [item for item in items if str(item.get("status")) != "success"],
            },
            "status": "success" if successful_count else "partial",
            "used": True,
            "note": (
                f"{label} 已完成自补证，优先综合补证轮结果；当前最佳轮次为第 {best_round} 轮，共 {len(items)} 次检索。"
                if agent == "iqs" or agent in IQS_ROLE_CONFIGS
                else f"{label} 已完成自补证，从 {len(items)} 次检索中选取第 {best_round} 轮最佳结果。"
            ),
        }
    role_children = {IQS_ROLE_CONFIGS[key]["child"]: children.get(IQS_ROLE_CONFIGS[key]["child"], {}) for key in IQS_ROLE_ORDER}
    if any(child.get("used") or child.get("status") in {"success", "partial"} for child in role_children.values()):
        children["web_analysis_agent"] = aggregate_iqs_role_children(children)
    return children


def _enabled_self_refine_agents(route: str, query_analysis: Dict[str, Any]) -> List[str]:
    configured = str(os.getenv("BRAIN_AGENT_TEXT_SELF_REFINE_AGENTS", "iqs") or "")
    wanted = {item.strip().lower() for item in configured.split(",") if item.strip()}
    if "web" in wanted or "iqs" in wanted:
        wanted.update(IQS_ROLE_ORDER)
    scheduled = set(_as_list(_as_dict(query_analysis).get("target_agents")) or _route_agents(route))
    ordered = [agent for agent in ["rag", *IQS_ROLE_ORDER, "iqs"] if agent in scheduled and agent in wanted]
    return ordered


def _pool_text(evidence_pool: Sequence[Dict[str, Any]], *, max_chars: int = 6000) -> str:
    parts: List[str] = []
    for item in evidence_pool:
        if not isinstance(item, dict):
            continue
        parts.append(str(item.get("query") or ""))
        parts.append(str(item.get("answer") or ""))
        parts.append(_json_text(item.get("key_sources"), max_chars=900))
        parts.append(_json_text(item.get("raw_data_points"), max_chars=900))
        parts.append(_json_text(item.get("data_gap"), max_chars=900))
    return _compact_text("\n".join(part for part in parts if part), max_chars=max_chars)


def _agent_previous_queries(evidence_pool: Sequence[Dict[str, Any]], agent: str) -> List[str]:
    return [str(item.get("query") or "").strip() for item in evidence_pool if isinstance(item, dict) and item.get("agent") == agent and str(item.get("query") or "").strip()]


def _should_self_refine_agent(agent: str, child: Dict[str, Any]) -> bool:
    status = str(child.get("status") or "").strip()
    confidence = _clip_confidence(child.get("confidence"), 0.0)
    if status != "success" or confidence < 0.68:
        return True
    gaps = _collect_child_gaps(child, max_items=2)
    if gaps:
        return True
    if agent == "iqs" or agent in IQS_ROLE_CONFIGS:
        return len(_as_list(child.get("key_sources"))) < 3 or not _has_specific_data(str(child.get("answer") or ""))
    return False


def _rank_self_refine_agents(target_agents: Sequence[str], children: Dict[str, Dict[str, Any]], max_agents: int) -> List[str]:
    scored: List[tuple[float, str]] = []
    for agent in target_agents:
        if agent in IQS_ROLE_CONFIGS:
            child_name = IQS_ROLE_CONFIGS[agent]["child"]
        else:
            child_name = "web_analysis_agent" if agent == "iqs" else "industry_rag_agent"
        child = children.get(child_name) or {}
        if not _should_self_refine_agent(agent, child):
            continue
        status = str(child.get("status") or "failed")
        confidence = _clip_confidence(child.get("confidence"), 0.0)
        source_count = len(_as_list(child.get("key_sources")))
        gap_count = len(_collect_child_gaps(child, max_items=5))
        priority = 0.0
        priority += 3.0 if status == "failed" else 2.0 if status == "partial" else 0.5
        priority += max(0.0, 1.0 - confidence)
        priority += min(1.0, gap_count * 0.25)
        priority += 0.5 if source_count < 3 else 0.0
        if agent in IQS_ROLE_CONFIGS:
            priority += 0.2
        scored.append((priority, agent))
    scored.sort(reverse=True)
    return [agent for _, agent in scored[: max(1, max_agents)]]


def _rank_self_refine_agents_from_merger(
    *,
    original_query: str,
    evidence_pool: Sequence[Dict[str, Any]],
    target_agents: Sequence[str],
    children: Dict[str, Dict[str, Any]],
    max_agents: int,
    research_plan: Optional[Dict[str, Any]] = None,
) -> List[str]:
    selected: List[str] = []
    try:
        package = merge_evidence_package(original_query=original_query, evidence_pool=evidence_pool, research_plan=research_plan)
        per_dimension = _as_dict(package.get("per_dimension"))
        ranked_dimensions = sorted(
            (
                (
                    _safe_float(_as_dict(payload).get("coverage_score"), 0.0),
                    str(dimension),
                )
                for dimension, payload in per_dimension.items()
            ),
            key=lambda item: item[0],
        )
        for score, dimension in ranked_dimensions:
            if score >= _env_float("BRAIN_AGENT_TEXT_SELF_REFINE_COVERAGE_TARGET", 0.60):
                continue
            agent = _fallback_followup_agent(dimension, score)
            if agent not in target_agents or agent not in IQS_ROLE_CONFIGS:
                continue
            child_name = IQS_ROLE_CONFIGS[agent]["child"]
            if _should_self_refine_agent(agent, children.get(child_name) or {}) and agent not in selected:
                selected.append(agent)
            if len(selected) >= max(1, max_agents):
                return selected
    except Exception:
        logger.exception("Self-refine agent selection failed", extra={"query": original_query})
        selected = []
    fallback = _rank_self_refine_agents(target_agents, children, max_agents)
    for agent in fallback:
        if agent not in selected:
            selected.append(agent)
        if len(selected) >= max(1, max_agents):
            break
    return selected


def _build_iqs_self_refine_queries(
    *,
    original_query: str,
    child: Dict[str, Any],
    previous_queries: Sequence[str],
    max_queries: int,
    role_key: str = "iqs",
) -> List[Dict[str, str]]:
    year = datetime.now().year
    seen = {_normalize_query_text(query) for query in previous_queries}
    gaps = _collect_child_gaps(child, max_items=4)
    role_config = IQS_ROLE_CONFIGS.get(role_key, {})
    role_focus = str(role_config.get("focus") or "补充 Research Planner 指定证据目标")
    role_dimension = str(role_config.get("dimension") or "动态检索任务")
    templates = [
        f"{original_query} {year} {role_dimension} {role_focus}",
        f"{original_query} {year} {role_dimension} 可核验事实 来源 口径 时间范围",
    ]
    for gap in gaps:
        templates.insert(0, f"{original_query} {gap} {year} {role_dimension} 可核验来源")
    followups: List[Dict[str, str]] = []
    for query in templates:
        key = _normalize_query_text(query)
        if key in seen:
            continue
        seen.add(key)
        followups.append({"query": query, "agent": role_key, "targets_gap": f"{role_dimension}自补证"})
        if len(followups) >= max(1, max_queries):
            break
    return followups


def build_agent_self_refine_queries(
    *,
    agent: str,
    original_query: str,
    evidence_pool: Sequence[Dict[str, Any]],
    children: Dict[str, Dict[str, Any]],
    previous_queries: Sequence[str],
    max_queries: int,
) -> List[Dict[str, str]]:
    if agent in IQS_ROLE_CONFIGS:
        child_name = IQS_ROLE_CONFIGS[agent]["child"]
    else:
        child_name = "web_analysis_agent" if agent == "iqs" else "industry_rag_agent"
    child = children.get(child_name) or {}
    if not _should_self_refine_agent(agent, child):
        return []
    if agent == "iqs" or agent in IQS_ROLE_CONFIGS:
        return _build_iqs_self_refine_queries(
            original_query=original_query,
            child=child,
            previous_queries=previous_queries,
            max_queries=max_queries,
            role_key=agent,
        )
    return []


def run_agent_text_self_refinement(
    *,
    state: BrainAgentState,
    initial_children: Dict[str, Dict[str, Any]],
    route: str,
) -> Dict[str, Any]:
    original_query = str(state.get("query") or "").strip()
    started = time.perf_counter()
    evidence_pool = build_initial_evidence_pool(original_query=original_query, children=initial_children)
    query_analysis = _as_dict(state.get("query_analysis"))
    if not _env_flag("BRAIN_AGENT_TEXT_SELF_REFINE", True):
        _progress("self-refine", "Agent Text 自精炼关闭", evidence=len(evidence_pool))
        return {
            "children": aggregate_best_children_from_evidence_pool(evidence_pool),
            "evidence_pool": evidence_pool,
            "self_refine_trace": [],
        }

    target_agents = _enabled_self_refine_agents(route, query_analysis)
    max_rounds = max(0, min(4, _env_int("BRAIN_AGENT_TEXT_SELF_REFINE_LOOPS", 2)))
    max_queries = max(1, min(4, _env_int("BRAIN_AGENT_TEXT_SELF_REFINE_MAX_QUERIES", 2)))
    max_agents = max(1, min(len(target_agents) or 1, _env_int("BRAIN_AGENT_TEXT_SELF_REFINE_MAX_AGENTS", 2)))
    trace: List[Dict[str, Any]] = []
    _progress("self-refine", "Agent Text 自精炼开始", rounds=max_rounds, agents=max_agents, queries_per_agent=max_queries)
    for offset in range(max_rounds):
        round_number = offset + 2
        children = aggregate_best_children_from_evidence_pool(evidence_pool)
        round_agents = _rank_self_refine_agents_from_merger(
            original_query=original_query,
            evidence_pool=evidence_pool,
            target_agents=target_agents,
            children=children,
            max_agents=max_agents,
            research_plan=_research_plan_from_state(state),
        )
        followups: List[Dict[str, str]] = []
        for agent in round_agents:
            followups.extend(
                build_agent_self_refine_queries(
                    agent=agent,
                    original_query=original_query,
                    evidence_pool=evidence_pool,
                    children=children,
                    previous_queries=_agent_previous_queries(evidence_pool, agent),
                    max_queries=max_queries,
                )
            )
        if not followups:
            trace.append({"round": round_number, "follow_up_queries": [], "stop_reason": "no_agent_gap"})
            _progress("self-refine", "自精炼无补证缺口，提前结束", round=round_number)
            break
        _progress("self-refine", "自精炼补证开始", round=round_number, followups=len(followups))
        followup_results = run_followup_queries(follow_up_queries=followups, round_number=round_number, state=state)
        evidence_pool.extend(followup_results)
        trace.append(
            {
                "round": round_number,
                "follow_up_queries": followups,
                "results": [
                    {
                        "agent": item.get("agent"),
                        "query": item.get("query"),
                        "status": item.get("status"),
                        "confidence": item.get("confidence"),
                        "data_points": len(list(item.get("raw_data_points") or [])),
                        "sources": len(list(item.get("key_sources") or [])),
                    }
                    for item in followup_results
                ],
            }
        )
        _progress("self-refine", "自精炼补证完成", round=round_number, evidence=len(evidence_pool))
    _progress("self-refine", "Agent Text 自精炼结束", evidence=len(evidence_pool), elapsed=f"{time.perf_counter() - started:.1f}s")
    return {
        "children": aggregate_best_children_from_evidence_pool(evidence_pool),
        "evidence_pool": evidence_pool,
        "self_refine_trace": trace,
    }


FINANCIAL_TERMS = [
    "营收",
    "收入",
    "净利润",
    "利润",
    "亏损",
    "毛利率",
    "净利率",
    "EBITDA",
    "现金流",
    "成本",
    "费用",
    "市值",
    "估值",
    "融资",
    "并购",
    "IPO",
    "股价",
]


def _infer_dimension_from_text(text: str) -> str:
    lowered = text.lower()
    best_dimension = "其他数据"
    best_count = 0
    for dimension, keywords in DIMENSION_KEYWORDS.items():
        count = sum(1 for keyword in keywords if keyword.lower() in lowered)
        if count > best_count:
            best_dimension = dimension
            best_count = count
    return best_dimension


def _infer_metric_from_context(context: str, value: str) -> str:
    if "亏损" in value:
        return "亏损"
    value_index = context.find(value)
    if value_index < 0:
        numeric = re.search(r"\d+(?:\.\d+)?", value)
        value_index = context.find(numeric.group(0)) if numeric else -1
    if value_index >= 0:
        near_left = context[max(0, value_index - 36):value_index]
        near_right = context[value_index:min(len(context), value_index + 42)]
        if "%" in value and ("市场份额" in near_left + near_right or "市占率" in near_left + near_right or "份额" in near_right):
            return "市场份额"
        nearby_terms = [
            ("市场规模", "市场规模"),
            ("市场空间", "市场规模"),
            ("融资", "融资额"),
            ("出货", "出货量"),
            ("交付", "交付量"),
            ("营收", "营收"),
            ("收入", "收入"),
            ("净利润", "净利润"),
            ("亏损", "亏损"),
            ("估值", "估值"),
        ]
        nearest_label = ""
        nearest_pos = -1
        for term, label in nearby_terms:
            pos = near_left.rfind(term)
            if pos > nearest_pos:
                nearest_pos = pos
                nearest_label = label
        if nearest_label:
            return nearest_label
        nearest_term = ""
        nearest_pos = -1
        for term in FINANCIAL_TERMS + ["市场规模", "增速", "市占率", "市场份额", "份额", "占比", "出货量", "出货", "案例占比", "年入", "销量", "价格", "渗透率", "CAGR", "融资额", "融资"]:
            pos = context.lower().rfind(term.lower(), 0, value_index + 1)
            if pos > nearest_pos:
                nearest_pos = pos
                nearest_term = term
        if nearest_term:
            return nearest_term
    if any(term.lower() in context.lower() for term in FINANCIAL_TERMS):
        for term in FINANCIAL_TERMS:
            if term.lower() in context.lower():
                return term
    metric_terms = ["市场规模", "增速", "市占率", "市场份额", "份额", "占比", "出货量", "出货", "案例占比", "年入", "销量", "价格", "渗透率", "CAGR", "融资额", "融资", "估值"]
    for term in metric_terms:
        if term.lower() in context.lower():
            return term
    if "%" in value:
        return "比例/增速"
    if "亿元" in value or "万亿元" in value or "亿美元" in value:
        return "金额"
    return "数据点"


def _extract_period(context: str) -> str:
    match = re.search(r"(20\d{2}(?:年|Q[1-4]|H[12])?|近\d+个月|近三年|过去\d+年|第[一二三四1-4]季度)", context, re.I)
    return match.group(1) if match else ""


def _extract_citations(text: str) -> str:
    citations = re.findall(r"\[(?:E\d+|\d+|W\d+|P\d+)\]", text)
    deduped: List[str] = []
    for citation in citations:
        if citation not in deduped:
            deduped.append(citation)
    return "".join(deduped[:4])


def _source_label_from_pool_item(item: Dict[str, Any], context: str = "") -> str:
    agent = "IQS" if item.get("agent") == "iqs" else "RAG"
    round_number = item.get("round") or 1
    citations = _extract_citations(context or str(item.get("answer") or ""))
    return f"{agent}·第{round_number}轮{citations}"


def extract_data_points(evidence_pool: Sequence[Dict[str, Any]], *, max_items: int = 12) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    seen = set()
    value_pattern = re.compile(
        r"(?P<value>(?:约|超过|超|达到|达|为|同比|环比|预计|亏损|盈利)?\s*\d+(?:\.\d+)?\s*(?:%|pct|亿元|万亿元|亿美元|万元|亿|万台|万套|万件|台|套|件|家|倍|元|美元))",
        re.I,
    )
    for item in evidence_pool:
        if not isinstance(item, dict) or str(item.get("status") or "") == "failed":
            continue
        used_raw_points = False
        for raw_point in list(item.get("raw_data_points") or []):
            if not isinstance(raw_point, dict):
                continue
            used_raw_points = True
            metric = str(raw_point.get("metric") or "").strip()
            value = str(raw_point.get("value") or "").strip()
            if not metric or not value or value == "N/A":
                continue
            key = (metric, value, str(raw_point.get("period") or ""), str(raw_point.get("source_url") or ""))
            if key in seen:
                continue
            seen.add(key)
            points.append(
                {
                    "dimension": str(raw_point.get("dimension") or "金融行情"),
                    "metric": metric,
                    "value": value,
                    "change": raw_point.get("change"),
                    "period": str(raw_point.get("period") or ""),
                    "entity": str(raw_point.get("raw_symbol") or item.get("query") or ""),
                    "source": _source_label_from_pool_item(item),
                    "source_url": str(raw_point.get("source_url") or raw_point.get("source") or ""),
                    "evidence": _compact_text(f"{metric}：{value} {raw_point.get('change') or ''}", max_chars=140),
                    "confidence": _clip_confidence(raw_point.get("confidence"), item.get("confidence") or 0.0),
                }
            )
            if len(points) >= max_items:
                return points
        if used_raw_points:
            continue
        source_text = " ".join(
            " ".join(str(source.get(key) or "") for key in ["title", "date"])
            for source in list(item.get("key_sources") or [])[:3]
            if isinstance(source, dict)
        )
        answer_text = str(item.get("answer") or "").strip()
        if not answer_text:
            answer_text = " ".join(str(source.get("quote") or "") for source in list(item.get("key_sources") or [])[:3] if isinstance(source, dict))
        text = _compact_text(" ".join([str(item.get("query") or ""), answer_text, source_text]), max_chars=2000)
        if not text:
            continue
        for match in value_pattern.finditer(text):
            value = re.sub(r"\s+", "", match.group("value") or "")
            if not value or not re.search(r"\d", value):
                continue
            start = max(0, match.start() - 70)
            end = min(len(text), match.end() + 90)
            context = _compact_text(text[start:end], max_chars=140)
            metric = _infer_metric_from_context(context, value)
            key = (metric, value, _extract_period(context), _source_label_from_pool_item(item, context))
            if key in seen:
                continue
            seen.add(key)
            dimension = _infer_dimension_from_text(context)
            if any(term.lower() in metric.lower() for term in FINANCIAL_TERMS):
                dimension = "财务与盈利"
            point = {
                "dimension": dimension,
                "metric": metric,
                "value": value,
                "period": _extract_period(context),
                "entity": _compact_text(str(item.get("query") or ""), max_chars=80),
                "source": _source_label_from_pool_item(item, context),
                "evidence": context,
                "confidence": _clip_confidence(item.get("confidence"), 0.0),
            }
            points.append(point)
            if len(points) >= max_items:
                return points
    return points


def extract_financial_points(data_points: Sequence[Dict[str, Any]], *, max_items: int = 5) -> List[Dict[str, Any]]:
    financial: List[Dict[str, Any]] = []
    for point in data_points:
        metric = str(point.get("metric") or "")
        dimension = str(point.get("dimension") or "")
        if dimension == "财务与盈利" or any(term.lower() in metric.lower() for term in FINANCIAL_TERMS):
            financial.append(dict(point))
            if len(financial) >= max_items:
                break
    return financial


def extract_commercial_data_points(evidence_pool: Sequence[Dict[str, Any]], *, max_items: int = 16) -> List[Dict[str, Any]]:
    source_items = [
        item
        for item in evidence_pool
        if isinstance(item, dict)
        and item.get("agent") in {"iqs", "rag"}
        and str(item.get("status") or "") != "failed"
    ]
    points = extract_data_points(source_items, max_items=max_items * 2)
    commercial_terms = [
        "市场规模",
        "融资",
        "融资额",
        "出货",
        "出货量",
        "市场份额",
        "市占率",
        "营收",
        "收入",
        "净利润",
        "利润",
        "亏损",
        "毛利率",
        "现金流",
        "估值",
        "销量",
        "渗透率",
        "增速",
    ]
    selected: List[Dict[str, Any]] = []
    seen = set()
    for point in points:
        metric = str(point.get("metric") or "")
        evidence = str(point.get("evidence") or "")
        if not any(term.lower() in (metric + evidence).lower() for term in commercial_terms):
            continue
        value = str(point.get("value") or "").strip()
        if not value or not re.search(r"\d", value):
            continue
        copied = dict(point)
        copied["dimension"] = str(copied.get("dimension") or "商业数据")
        copied["source"] = str(copied.get("source") or "IQS/RAG证据抽取")
        copied["freshness"] = "evidence_extracted"
        copied["market"] = "商业数据"
        normalized_value = re.sub(r"[^\d.%]+", "", str(copied.get("value") or ""))
        key = (copied.get("metric"), normalized_value or copied.get("value"))
        if key in seen:
            continue
        seen.add(key)
        selected.append(copied)
        if len(selected) >= max_items:
            break
    return selected


def build_key_evidence(evidence_pool: Sequence[Dict[str, Any]], *, max_items: int = 3) -> List[Dict[str, Any]]:
    evidence: List[Dict[str, Any]] = []
    for item in evidence_pool:
        if not isinstance(item, dict) or str(item.get("status") or "") == "failed":
            continue
        answer = _compact_text(item.get("answer"), max_chars=220)
        if not answer:
            continue
        evidence.append(
            {
                "claim": answer,
                "source": _source_label_from_pool_item(item, answer),
                "query": _compact_text(item.get("query"), max_chars=100),
                "confidence": _clip_confidence(item.get("confidence"), 0.0),
            }
        )
        if len(evidence) >= max_items:
            break
    return evidence


def build_report_package(decision: Dict[str, Any], evidence_pool: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    answer_payload = _as_dict(decision.get("answer"))
    review = _as_dict(decision.get("supervisor_review"))
    data_points = extract_data_points(evidence_pool, max_items=8)
    financial_points = extract_financial_points(data_points)
    conclusion = str(answer_payload.get("conclusion") or "证据不足，无法输出有效判断").strip()
    next_action = str(decision.get("next_action") or "").strip()
    data_gaps: List[str] = []
    if not data_points:
        conclusion = "当前证据以定性判断为主，缺少可直接入报告的量化数据"
        if next_action == "complete":
            next_action = "needs_more_search"
    if not financial_points:
        data_gaps.append("缺少营收、净利润、亏损、毛利率、现金流等财务指标，后续报告不能直接写盈利/亏损判断。")
    for gap in _as_list(decision.get("evidence_gap")):
        if isinstance(gap, dict):
            text = str(gap.get("dimension") or "").strip()
            suggestion = str(gap.get("suggestion") or "").strip()
            if text:
                data_gaps.append(f"{text}：{suggestion}" if suggestion else text)
    deduped_gaps: List[str] = []
    for gap in data_gaps:
        if gap and gap not in deduped_gaps:
            deduped_gaps.append(gap)

    return {
        "conclusion": conclusion,
        "confidence": _clip_confidence(decision.get("confidence"), 0.0),
        "coverage_score": _safe_float(review.get("coverage_score"), 0.0) if review else None,
        "financial_data": financial_points[:5],
        "key_data": data_points[:6],
        "qualitative_evidence": [] if data_points else build_key_evidence(evidence_pool),
        "data_gaps": deduped_gaps[:5],
        "next_action": next_action,
    }


def render_brief_report_text(report_package: Dict[str, Any]) -> str:
    lines = [f"结论：{report_package.get('conclusion') or '证据不足，无法输出有效判断'}"]
    financial = list(report_package.get("financial_data") or [])
    key_data = list(report_package.get("key_data") or [])
    if financial:
        lines.append("")
        lines.append("关键财务数据：")
        for item in financial[:5]:
            period = f"（{item.get('period')}）" if item.get("period") else ""
            lines.append(f"- {item.get('metric')}：{item.get('value')}{period}；来源：{item.get('source')}；依据：{item.get('evidence')}")
    if key_data:
        lines.append("")
        lines.append("关键数据：")
        for item in key_data[:6]:
            period = f"（{item.get('period')}）" if item.get("period") else ""
            lines.append(f"- {item.get('dimension')}｜{item.get('metric')}：{item.get('value')}{period}；来源：{item.get('source')}")
    gaps = list(report_package.get("data_gaps") or [])
    if gaps:
        lines.append("")
        lines.append("数据缺口：")
        for item in gaps[:4]:
            lines.append(f"- {item}")
    return "\n".join(lines).strip()


def _format_gap_item(item: Any) -> str:
    if isinstance(item, dict):
        metric = str(item.get("metric") or item.get("dimension") or item.get("title") or "数据缺口").strip()
        reason = _compact_text(item.get("reason") or item.get("missing_from") or item.get("note") or "", max_chars=220)
        suggestion = _compact_text(item.get("suggestion") or item.get("next_step") or "", max_chars=160)
        parts = [metric]
        if reason:
            parts.append(reason)
        if suggestion:
            parts.append(f"建议：{suggestion}")
        return "；".join(parts)
    return str(item or "").strip()


def _collect_child_gaps(child: Dict[str, Any], *, max_items: int = 5) -> List[str]:
    gaps: List[str] = []
    limitations = _as_dict(child.get("limitations"))
    candidates: List[Any] = []
    candidates.extend(_as_list(child.get("data_gap")))
    candidates.extend(_as_list(limitations.get("data_gap")))
    candidates.extend(_as_list(limitations.get("evidence_gap")))
    candidates.extend(_as_list(limitations.get("gaps")))
    candidates.extend(_as_list(limitations.get("errors")))
    for item in candidates:
        text = _format_gap_item(item)
        if text and text not in gaps:
            gaps.append(text)
        if len(gaps) >= max_items:
            break
    return gaps


def _limit_multiline_text(value: Any, *, max_chars: int = 2400) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    if len(text) <= max_chars:
        return text
    return text[: max(20, max_chars - 3)].rstrip() + "..."


def _render_source_lines(sources: Sequence[Dict[str, Any]], *, max_items: int = 30) -> List[str]:
    lines: List[str] = []
    seen = set()
    for source in sources:
        title = str(source.get("title") or f"来源{len(lines) + 1}").strip()
        date = str(source.get("date") or "").strip()
        url = str(source.get("url") or source.get("source_file") or "").strip()
        key = (title, date, url)
        if key in seen:
            continue
        seen.add(key)
        score = source.get("score")
        detail_parts = []
        if date:
            detail_parts.append(date)
        if score not in (None, ""):
            detail_parts.append(f"score={_safe_float(score):.3f}")
        if url:
            detail_parts.append(url)
        detail = f"（{'；'.join(detail_parts)}）" if detail_parts else ""
        lines.append(f"{len(lines) + 1}. {title}{detail}")
        if len(lines) >= max_items:
            break
    return lines


def _render_data_lines(data_points: Sequence[Dict[str, Any]], *, max_items: int = 30) -> List[str]:
    lines: List[str] = []
    seen = set()
    for item in data_points:
        metric = str(item.get("metric") or "").strip()
        value = str(item.get("value") or "").strip()
        change = str(item.get("change") or "").strip()
        period = str(item.get("period") or "").strip()
        source = str(item.get("source") or item.get("source_url") or "").strip()
        evidence = _compact_text(item.get("evidence") or "", max_chars=120)
        key = (metric, value, period)
        if key in seen:
            continue
        seen.add(key)
        suffix = f"；变化：{change}" if change and change.lower() != "none" else ""
        period_text = f"；时间：{period}" if period else ""
        source_text = f"；来源：{source}" if source else ""
        evidence_text = f"；依据：{evidence}" if evidence else ""
        lines.append(f"- {metric or '数据指标'}：{value or 'N/A'}{suffix}{period_text}{source_text}{evidence_text}")
        if len(lines) >= max_items:
            break
    return lines


def _format_evidence_refs(value: Any, *, max_items: int = 4) -> str:
    refs: List[str] = []
    for ref in _as_list(value):
        text = str(ref or "").strip()
        if not text or text in refs:
            continue
        refs.append(text)
        if len(refs) >= max_items:
            break
    return f" [{', '.join(refs)}]" if refs else ""


def _humanize_structured_gap(reason: Any, *, dimension: str = "") -> str:
    text = _compact_text(reason, max_chars=180)
    if not text or "S 级证据" in text or "S级证据" in text:
        prefix = f"{dimension}相关" if dimension else "该维度"
        return f"{prefix}权威定量证据仍需补充，建议进一步核验官方统计、公司公告或权威研究报告。"
    return text


def _render_structured_processing_lines(
    *,
    evidence_package: Optional[Dict[str, Any]] = None,
    structured_analysis: Optional[Dict[str, Any]] = None,
    analysis_meta: Optional[Dict[str, Any]] = None,
    writer_report: Optional[Dict[str, Any]] = None,
    writer_meta: Optional[Dict[str, Any]] = None,
    max_judgments: int = 3,
    max_gaps: int = 3,
) -> List[str]:
    evidence_package = _as_dict(evidence_package)
    structured_analysis = _as_dict(structured_analysis)
    analysis_meta = _as_dict(analysis_meta)
    writer_report = _as_dict(writer_report)
    writer_meta = _as_dict(writer_meta)
    if not evidence_package and not structured_analysis:
        return []

    lines: List[str] = []
    lines.append("\u7ed3\u6784\u5316\u5904\u7406\u7ed3\u679c")

    summary = _as_dict(evidence_package.get("summary"))
    if summary:
        overall = _safe_float(summary.get("overall_coverage"), 0.0)
        evidence_count = int(_safe_float(summary.get("evidence_count"), 0.0))
        conflict_count = int(_safe_float(summary.get("conflict_count"), 0.0))
        weakest_dimension = str(summary.get("weakest_dimension") or "").strip()
        ready = "\u662f" if bool(summary.get("ready_for_analysis")) else "\u5426"
        detail = (
            f"\u8986\u76d6\u5ea6={overall:.2f}; "
            f"\u8bc1\u636e\u6570={evidence_count}; "
            f"\u51b2\u7a81={conflict_count}; "
            f"\u53ef\u5206\u6790={ready}"
        )
        if weakest_dimension:
            detail += f"; \u6700\u5f31\u7ef4\u5ea6={weakest_dimension}"
        lines.append(f"Evidence Merger: {detail}")

    analysis_source = str(analysis_meta.get("source") or "").strip()
    if analysis_source:
        lines.append(f"Analysis Agent: source={analysis_source}")

    if writer_report:
        writer_source = str(writer_meta.get("source") or _as_dict(writer_report.get("metadata")).get("strategy") or "").strip()
        source_count = int(_safe_float(writer_report.get("source_count"), 0.0))
        estimated_chars = int(_safe_float(writer_report.get("estimated_chars"), 0.0))
        sections = len(_as_list(writer_report.get("sections")))
        lines.append(
            "Writer Agent: "
            f"source={writer_source or 'analysis_first_writer_using_evidence_analyses'}; "
            f"sections={sections}; sources={source_count}; chars={estimated_chars}"
        )

    judgments = [item for item in _as_list(structured_analysis.get("key_judgments")) if isinstance(item, dict)]
    if judgments:
        lines.append("\u6838\u5fc3\u5224\u65ad:")
        for index, item in enumerate(judgments[:max_judgments], start=1):
            judgment_type = str(item.get("type") or "\u6838\u5fc3\u5224\u65ad").strip()
            judgment = _compact_text(item.get("judgment") or item.get("conclusion") or "", max_chars=280)
            if not judgment:
                continue
            confidence = _safe_float(item.get("confidence"), 0.0)
            lines.append(f"{index}. {judgment_type}: {judgment}; confidence={confidence:.2f}")

    facts = [item for item in _as_list(structured_analysis.get("core_facts")) if isinstance(item, dict)]
    facts_by_dimension: Dict[str, Dict[str, Any]] = {}
    for item in facts:
        dimension = str(item.get("dimension") or "").strip()
        fact = _compact_text(item.get("fact") or "", max_chars=220)
        if not dimension or not fact or dimension in facts_by_dimension:
            continue
        facts_by_dimension[dimension] = item
    if facts_by_dimension:
        lines.append("\u6838\u5fc3\u4e8b\u5b9e\u9aa8\u67b6:")
        for dimension, item in list(facts_by_dimension.items())[:5]:
            fact = _compact_text(item.get("fact") or "", max_chars=220)
            confidence = _safe_float(item.get("confidence"), 0.0)
            lines.append(f"- {dimension}: {fact}; confidence={confidence:.2f}")

    gaps = [item for item in _as_list(structured_analysis.get("evidence_gaps")) if isinstance(item, dict)]
    if gaps:
        lines.append("\u5f85\u8865\u8bc1\u636e:")
        for item in gaps[:max_gaps]:
            dimension = str(item.get("dimension") or "").strip()
            reason = _humanize_structured_gap(item.get("reason") or item.get("suggestion") or "", dimension=dimension)
            coverage = _safe_float(item.get("coverage_score"), 0.0)
            label = f"{dimension}: " if dimension else ""
            lines.append(f"- {label}{reason}; coverage={coverage:.2f}")

    return lines


def render_agent_text_output(
    *,
    query_analysis: Dict[str, Any],
    children: Dict[str, Dict[str, Any]],
    evidence_pool: Sequence[Dict[str, Any]],
    evidence_package: Optional[Dict[str, Any]] = None,
    structured_analysis: Optional[Dict[str, Any]] = None,
    analysis_meta: Optional[Dict[str, Any]] = None,
    writer_report: Optional[Dict[str, Any]] = None,
    writer_meta: Optional[Dict[str, Any]] = None,
    include_child_details: bool = True,
    include_writer_report: bool = False,
) -> str:
    query_analysis = _as_dict(query_analysis)
    original_query = str(query_analysis.get("original_query") or "").strip()
    related_questions = _as_list(query_analysis.get("related_questions"))
    agent_queries = _as_dict(query_analysis.get("agent_queries"))
    report_plan = _as_dict(query_analysis.get("report_plan"))
    target_agents = set(_as_list(query_analysis.get("target_agents")))
    if not target_agents:
        target_agents = {"rag", *IQS_ROLE_ORDER}

    lines: List[str] = []
    lines.append("问题拆解")
    lines.append(f"原问题：{original_query}")
    if report_plan:
        lines.append(
            "报告类型："
            f"{report_plan.get('report_type') or ''} "
            f"{report_plan.get('report_name') or ''}；"
            f"核心价值：{report_plan.get('core_value') or ''}"
        )
        chapters = [str(item) for item in _as_list(report_plan.get("chapter_structure")) if str(item).strip()]
        if chapters:
            lines.append("章节结构：" + " → ".join(chapters))
    if related_questions:
        lines.append("相关高价值问题：")
        for index, question in enumerate(related_questions[:5], start=1):
            lines.append(f"{index}. {str(question).strip()}")
    lines.append("")
    lines.append("子 Agent 检索问题：")
    query_labels: List[tuple[str, str]] = [("rag", "RAG Agent")]
    query_labels.extend((key, IQS_ROLE_CONFIGS[key]["label"]) for key in IQS_ROLE_ORDER)
    query_labels.append(("iqs", "IQS Agent"))
    for agent_key, label in query_labels:
        queries = _as_list(agent_queries.get(agent_key))
        if agent_key not in target_agents and not queries:
            continue
        if not queries:
            continue
        lines.append(f"{label}：")
        for index, query in enumerate(queries, start=1):
            lines.append(f"{index}. {str(query).strip()}")

    structured_lines = _render_structured_processing_lines(
        evidence_package=evidence_package,
        structured_analysis=structured_analysis,
        analysis_meta=analysis_meta,
        writer_report=writer_report,
        writer_meta=writer_meta,
    )
    if structured_lines:
        lines.append("")
        lines.extend(structured_lines)

    agent_order = [
        ("rag", "industry_rag_agent", "RAG Agent（本地知识库）"),
    ]
    agent_order.extend(
        (key, IQS_ROLE_CONFIGS[key]["child"], f"{IQS_ROLE_CONFIGS[key]['label']}（联网搜索）") for key in IQS_ROLE_ORDER
    )
    agent_order.append(("iqs", "web_analysis_agent", "IQS Agent（联网汇总）"))
    for agent_key, child_name, label in agent_order:
        child = children.get(child_name) or {}
        note = str(child.get("note") or "").strip()
        if agent_key not in target_agents:
            continue
        status = str(child.get("status") or "unknown").strip()
        confidence = _clip_confidence(child.get("confidence"), 0.0)
        lines.append("")
        lines.append(f"【{label}｜补正后最佳结果】")
        lines.append(f"状态：{status}；置信度：{confidence:.2f}" + (f"；说明：{note}" if note else ""))

        data_points = [item for item in _as_list(child.get("raw_data_points")) if isinstance(item, dict)]
        sources = [item for item in _as_list(child.get("key_sources")) if isinstance(item, dict)]
        gaps = _collect_child_gaps(child)
        lines.append(f"摘要：数据点={len(data_points)}；来源={len(sources)}；缺口={len(gaps)}")
        if not include_child_details:
            continue

        if data_points:
            lines.append("数据点：")
            lines.extend(_render_data_lines(data_points))

        answer = _limit_multiline_text(child.get("answer"), max_chars=4000)
        if answer:
            lines.append("回答：")
            lines.append(answer)
        elif not data_points:
            lines.append("回答：未返回可用文字数据。")

        if sources:
            lines.append("主要来源：")
            lines.extend(_render_source_lines(sources))

        if gaps:
            lines.append("数据缺口：")
            for gap in gaps:
                lines.append(f"- {gap}")

    report_markdown = str(_as_dict(writer_report).get("report_markdown") or "").strip()
    if include_writer_report and report_markdown:
        lines.append("")
        lines.append("Writer Agent 报告")
        lines.append(report_markdown)

    if not any(str(item.get("answer") or "").strip() or item.get("raw_data_points") for item in evidence_pool if isinstance(item, dict)):
        lines.append("")
        lines.append("提示：本轮没有拿到可直接用于报告的文字或数值数据，请检查数据库、IQS Key 或扩大检索范围。")

    return "\n".join(line for line in lines if line is not None).strip()


def run_supervisor_evidence_loop(
    *,
    state: BrainAgentState,
    initial_children: Dict[str, Dict[str, Any]],
    route: str,
) -> Dict[str, Any]:
    started = time.perf_counter()
    original_query = str(state.get("query") or "").strip()
    enable_loop = bool(state.get("enable_followup_loop", _env_flag("BRAIN_ENABLE_FOLLOWUP_LOOP", True)))
    enable_llm_eval = _env_flag("BRAIN_ENABLE_LLM_COVERAGE_EVAL", True)
    max_loops = max(1, min(8, int(state.get("supervisor_max_loops") or _env_int("BRAIN_SUPERVISOR_MAX_LOOPS", 5))))
    min_gain = max(0.0, _safe_float(state.get("supervisor_min_coverage_gain"), _env_float("BRAIN_SUPERVISOR_MIN_COVERAGE_GAIN", 0.10)))
    max_followups = max(1, min(10, int(state.get("supervisor_max_followup_queries") or _env_int("BRAIN_SUPERVISOR_MAX_FOLLOWUP_QUERIES", 10))))
    coverage_target = _supervisor_coverage_target()

    evidence_pool = build_initial_evidence_pool(original_query=original_query, children=initial_children)
    coverage_units = coverage_units_from_state(state)
    previous_queries = [original_query]
    seen_followup_keys = _repair_seen_keys_for_state(state)
    loop_trace: List[Dict[str, Any]] = []
    evaluation_errors: List[str] = []
    prev_coverage = 0.0
    final_evaluation: Dict[str, Any] = {}
    final_summary = summarize_evidence_pool(evidence_pool)
    _progress(
        "coverage",
        "证据覆盖率闭环开始",
        loop=enable_loop,
        llm_eval=enable_llm_eval,
        max_loops=max_loops,
        initial_evidence=len(evidence_pool),
    )

    for loop_number in range(1, max_loops + 1):
        if _deadline_exceeded(state, min_remaining=1.0):
            timeout_payload = _deadline_timeout_payload(state, stage="supervisor_evidence_loop")
            loop_trace.append(
                {
                    "round": loop_number,
                    "coverage_score": prev_coverage,
                    "is_sufficient": False,
                    "stop_reason": "deadline_exceeded",
                    "knowledge_gaps": [],
                    "follow_up_queries": [],
                    "evidence_count": len(evidence_pool),
                    "live_timeout": timeout_payload,
                }
            )
            final_evaluation = {
                **_as_dict(final_evaluation),
                "is_sufficient": False,
                "stop_reason": "deadline_exceeded",
                "live_timeout": timeout_payload,
            }
            _progress("coverage", "覆盖率闭环停止", reason="deadline_exceeded")
            break
        final_summary = summarize_evidence_pool(evidence_pool)
        fallback_eval = evaluate_coverage_fallback(
            original_query=original_query,
            evidence_pool=evidence_pool,
            coverage_units=coverage_units,
            loop_number=loop_number,
            max_loops=max_loops,
            prev_coverage_score=prev_coverage,
            min_gain=min_gain,
            previous_queries=previous_queries,
            max_followup_queries=max_followups,
        )
        evaluation = fallback_eval
        if enable_llm_eval and enable_loop:
            try:
                evaluation = evaluate_coverage_with_llm(
                    original_query=original_query,
                    evidence_pool_summary=final_summary,
                    coverage_units=coverage_units,
                    loop_number=loop_number,
                    max_loops=max_loops,
                    prev_coverage_score=prev_coverage,
                    min_gain=min_gain,
                    fallback=fallback_eval,
                    previous_queries=previous_queries,
                    max_followup_queries=max_followups,
                )
            except Exception as exc:
                logger.exception("Coverage LLM evaluation failed", extra={"query": original_query, "round": loop_number})
                evaluation_errors.append(f"第{loop_number}轮覆盖率 LLM 评估降级：{exc}")
                diagnostic = _as_dict(getattr(exc, "diagnostic", {}))
                evaluation = {
                    **fallback_eval,
                    "is_sufficient": False if _env_flag("COVERAGE_LLM_FAILURE_BLOCKING", True) else bool(fallback_eval.get("is_sufficient")),
                    "stop_reason": "coverage_llm_eval_degraded",
                    "llm_degraded": True,
                    "degraded": True,
                    "llm_call": diagnostic,
                    "degraded_reason": str(exc),
                    "knowledge_gaps": [
                        *_as_list(fallback_eval.get("knowledge_gaps")),
                        {
                            "dimension": "coverage_eval",
                            "reason": "LLM coverage evaluation failed; fallback result cannot auto-publish.",
                            "severity": "critical",
                        },
                    ],
                }
        final_evaluation = evaluation
        followups = list(evaluation.get("follow_up_queries") or [])
        trace_item: Dict[str, Any] = {
            "round": loop_number,
            "coverage_score": evaluation.get("coverage_score", 0.0),
            "is_sufficient": bool(evaluation.get("is_sufficient")),
            "stop_reason": evaluation.get("stop_reason"),
            "knowledge_gaps": evaluation.get("knowledge_gaps", []),
            "follow_up_queries": followups,
            "evidence_count": len(evidence_pool),
            "llm_degraded": bool(evaluation.get("llm_degraded") or evaluation.get("degraded")),
            "llm_call": _as_dict(evaluation.get("llm_call")),
        }
        _progress(
            "coverage",
            "覆盖率评估完成",
            round=f"{loop_number}/{max_loops}",
            score=evaluation.get("coverage_score", 0.0),
            sufficient=bool(evaluation.get("is_sufficient")),
            followups=len(followups),
            stop=evaluation.get("stop_reason"),
        )

        if (not enable_loop) or evaluation.get("is_sufficient"):
            if not enable_loop and not trace_item.get("stop_reason"):
                trace_item["stop_reason"] = "followup_loop_disabled"
            _progress("coverage", "覆盖率闭环停止", reason=trace_item.get("stop_reason") or "sufficient")
            loop_trace.append(trace_item)
            break
        if not followups:
            coverage_score = _safe_float(evaluation.get("coverage_score"), 0.0)
            sufficient = coverage_score >= coverage_target
            evaluation = {**evaluation, "is_sufficient": sufficient, "stop_reason": "coverage_stalled" if sufficient else "coverage_stalled_with_gaps"}
            final_evaluation = evaluation
            trace_item["is_sufficient"] = sufficient
            trace_item["stop_reason"] = evaluation["stop_reason"]
            _progress("coverage", "覆盖率闭环停止", reason=evaluation["stop_reason"])
            loop_trace.append(trace_item)
            break

        if route == "local":
            followups = [{**item, "agent": "rag"} for item in followups]
        elif route == "web":
            followups = [
                {**item, "agent": item.get("agent") if str(item.get("agent") or "").lower() in IQS_ROLE_CONFIGS else "iqs"}
                for item in followups
            ]
        followups, skipped_duplicates = _repair_tasks_from_items(
            followups,
            origin_node="coverage_evaluation",
            loop_name="supervisor_coverage",
            max_tasks=max_followups,
            seen_keys=seen_followup_keys,
        )
        trace_item["follow_up_queries"] = followups
        trace_item["skipped_duplicate_followups"] = skipped_duplicates
        trace_item["repair_task_summary"] = _repair_task_summary(followups)
        if not followups:
            coverage_score = _safe_float(evaluation.get("coverage_score"), 0.0)
            sufficient = coverage_score >= coverage_target
            evaluation = {**evaluation, "is_sufficient": sufficient, "stop_reason": "duplicate_followups_exhausted" if sufficient else "duplicate_followups_exhausted_with_gaps"}
            final_evaluation = evaluation
            trace_item["is_sufficient"] = sufficient
            trace_item["stop_reason"] = evaluation["stop_reason"]
            _progress("coverage", "覆盖率闭环停止", reason=evaluation["stop_reason"])
            loop_trace.append(trace_item)
            break
        if _deadline_exceeded(state, min_remaining=1.0):
            timeout_payload = _deadline_timeout_payload(state, stage="supervisor_followup_queries")
            trace_item["stop_reason"] = "deadline_exceeded"
            trace_item["live_timeout"] = timeout_payload
            loop_trace.append(trace_item)
            final_evaluation = {**_as_dict(final_evaluation), "is_sufficient": False, "stop_reason": "deadline_exceeded", "live_timeout": timeout_payload}
            _progress("coverage", "覆盖率闭环停止", reason="deadline_exceeded")
            break
        before_pool_size = len(evidence_pool)
        followup_results = run_followup_queries(follow_up_queries=followups, round_number=loop_number + 1, state=state)
        for item in followups:
            previous_queries.append(str(item.get("query") or ""))
        usable_followup_results = _substantive_followup_results(followup_results)
        evidence_pool.extend(usable_followup_results)
        has_signal = _followup_result_has_signal(followup_results)
        repair_result_summary = _repair_result_summary(followup_results, usable_results=usable_followup_results)
        gap_ledger = _gap_ledger_from_followups(followups, followup_results)
        trace_item["repair_task_summary"] = _repair_task_summary_after_policy(followups, followup_results)
        trace_item["followup_results"] = [_trace_followup_result(item) for item in followup_results if isinstance(item, dict)]
        trace_item["gap_ledger"] = gap_ledger
        trace_item["followup_has_signal"] = has_signal
        trace_item["attempted_task_count"] = len(followups)
        trace_item["repair_result_summary"] = repair_result_summary
        trace_item["signal_count"] = repair_result_summary.get("signal_count")
        trace_item["empty_success_count"] = repair_result_summary.get("empty_success_count")
        trace_item["failed_count"] = repair_result_summary.get("failed_count")
        trace_item["new_usable_evidence_count"] = repair_result_summary.get("new_usable_evidence_count")
        trace_item["new_ab_source_count"] = repair_result_summary.get("new_ab_source_count")
        trace_item["evidence_delta"] = len(evidence_pool) - before_pool_size
        loop_trace.append(trace_item)
        if not has_signal:
            coverage_score = _safe_float(evaluation.get("coverage_score"), 0.0)
            sufficient = coverage_score >= coverage_target
            evaluation = {**evaluation, "is_sufficient": sufficient, "stop_reason": "coverage_exhausted" if sufficient else "coverage_exhausted_with_gaps"}
            final_evaluation = evaluation
            trace_item["is_sufficient"] = sufficient
            trace_item["stop_reason"] = evaluation["stop_reason"]
            _progress("coverage", "覆盖率闭环停止", reason=evaluation["stop_reason"])
            if (not sufficient) and _continuous_evidence_loop_mode() and loop_number < max_loops:
                trace_item["stop_reason"] = "no_new_followup_signal_retry"
                prev_coverage = coverage_score
                continue
            break
        prev_coverage = _safe_float(evaluation.get("coverage_score"), prev_coverage)

    if loop_trace and loop_trace[-1].get("followup_has_signal") and not _as_dict(final_evaluation).get("is_sufficient"):
        final_summary = summarize_evidence_pool(evidence_pool)
        final_evaluation = evaluate_coverage_fallback(
            original_query=original_query,
            evidence_pool=evidence_pool,
            coverage_units=coverage_units,
            loop_number=max_loops,
            max_loops=max_loops,
            prev_coverage_score=prev_coverage,
            min_gain=min_gain,
            previous_queries=previous_queries,
            max_followup_queries=max_followups,
        )
        loop_trace.append(
            {
                "round": "final_evaluation",
                "coverage_score": final_evaluation.get("coverage_score", 0.0),
                "is_sufficient": bool(final_evaluation.get("is_sufficient")),
                "stop_reason": final_evaluation.get("stop_reason"),
                "knowledge_gaps": final_evaluation.get("knowledge_gaps", []),
                "evidence_count": len(evidence_pool),
            }
        )

    aggregated_children = aggregate_children_from_evidence_pool(evidence_pool)
    _progress(
        "coverage",
        "证据覆盖率闭环完成",
        evidence=len(evidence_pool),
        rounds=len(loop_trace),
        elapsed=f"{time.perf_counter() - started:.1f}s",
    )
    return {
        "children": aggregated_children,
        "evidence_pool": evidence_pool,
        "evidence_pool_summary": final_summary,
        "coverage_evaluation": final_evaluation,
        "loop_trace": loop_trace,
        "evaluation_errors": evaluation_errors,
    }


def build_supervisor_decision(
    *,
    query: str,
    route: str,
    route_reason: str,
    children: Dict[str, Dict[str, Any]],
    errors: Sequence[str],
    coverage_evaluation: Optional[Dict[str, Any]] = None,
    loop_trace: Optional[Sequence[Dict[str, Any]]] = None,
    evidence_pool_summary: str = "",
) -> Dict[str, Any]:
    rag_child = children.get("industry_rag_agent") or {}
    web_child = children.get("web_analysis_agent") or {}
    local_rag_enabled = _local_rag_enabled()
    conflict_inputs = [
        _has_conflict_marker(web_child.get("answer")),
        _has_conflict_marker(_as_dict(web_child.get("limitations")).get("conflicts")),
    ]
    if local_rag_enabled:
        conflict_inputs.extend(
            [
                _has_conflict_marker(rag_child.get("answer")),
                _has_conflict_marker(_as_dict(rag_child.get("limitations")).get("conflicts")),
            ]
        )
    has_conflict = any(conflict_inputs)
    conflicts: List[Dict[str, str]] = []
    if has_conflict:
        if local_rag_enabled:
            priority, reason = _infer_conflict_priority(rag_child, web_child)
            conflicts.append(
                {
                    "dimension": "子 Agent 结论一致性",
                    "rag_view": _compact_text(rag_child.get("answer"), max_chars=260) or "RAG 未提供可比较结论",
                    "web_view": _compact_text(web_child.get("answer"), max_chars=260) or "WEB 未提供可比较结论",
                    "priority": priority,
                    "reason": reason,
                }
            )
        else:
            conflicts.append(
                {
                    "dimension": "联网来源内部一致性",
                    "rag_view": "Local RAG disabled for this run.",
                    "web_view": _compact_text(web_child.get("answer"), max_chars=260) or "WEB 未提供可比较结论",
                    "priority": "web_review_required",
                    "reason": "Local RAG is disabled; resolve conflicts within public/web evidence sources.",
                }
            )

    confidence = _calculate_supervisor_confidence(
        rag_child if local_rag_enabled else {"status": "skipped", "confidence": 0.0},
        web_child,
        has_conflict=has_conflict,
    )
    rag_status = str(rag_child.get("status") or "failed")
    web_status = str(web_child.get("status") or "failed")
    if confidence == 0.0:
        conclusion = "证据不足，无法输出有效判断"
        consensus = None
    elif not local_rag_enabled:
        if web_status == "success":
            conclusion = "联网公开证据可供进入行研分析"
        elif web_status == "partial":
            conclusion = "联网公开证据部分可用，带缺口进入行研分析"
        else:
            conclusion = "联网公开证据不足，需要继续补充检索"
        web_claim = _first_claim(str(web_child.get("answer") or ""))
        consensus = f"[WEB] {web_claim}" if web_claim else None
    elif rag_status == "success" and web_status == "success" and not has_conflict:
        conclusion = "本地与联网证据方向一致，可进入行研分析"
        rag_claim = _first_claim(str(rag_child.get("answer") or ""))
        web_claim = _first_claim(str(web_child.get("answer") or ""))
        consensus = "\n".join(part for part in [f"[RAG] {rag_claim}" if rag_claim else "", f"[WEB] {web_claim}" if web_claim else ""] if part) or None
    elif rag_status == "success" and web_status == "success":
        conclusion = "本地与联网证据存在分歧，需先核验口径"
        consensus = None
    elif rag_status == "success":
        conclusion = "本地知识库可供初判，但缺少联网交叉验证"
        consensus = None
    elif web_status == "success":
        conclusion = "联网证据可供初判，但缺少本地交叉验证"
        consensus = None
    else:
        conclusion = "证据有限，仅可作为补充线索"
        consensus = None

    coverage_payload = _as_dict(coverage_evaluation)
    coverage_score = _safe_float(coverage_payload.get("coverage_score"), 0.0)
    if coverage_payload and confidence > 0:
        stop_reason = str(coverage_payload.get("stop_reason") or "")
        if coverage_score >= _supervisor_coverage_target() and not has_conflict:
            conclusion = "证据覆盖率较高，可进入行研分析"
        elif stop_reason in {"max_loops_reached", "max_loops_reached_with_gaps"}:
            conclusion = "已达补证上限，带缺口进入行研分析"
        elif stop_reason in {"coverage_stalled", "coverage_stalled_with_gaps"}:
            conclusion = "补证提升停滞，带缺口进入行研分析"
        elif not coverage_payload.get("is_sufficient"):
            conclusion = "证据覆盖不足，需要继续补充检索"

    evidence_gap = _collect_evidence_gaps(children, confidence=confidence, has_conflict=has_conflict)
    for gap in _as_list(coverage_payload.get("knowledge_gaps")):
        if not isinstance(gap, dict):
            continue
        severity = str(gap.get("severity") or "").strip()
        dimension = str(gap.get("dimension") or "").strip()
        if not dimension:
            continue
        suggestion = "补充该维度的权威数据或明确时间口径。"
        if severity == "critical":
            suggestion = "优先补充该维度的权威来源、具体数字和时间范围。"
        elif severity == "moderate":
            suggestion = "补充更具体的数字、公司或政策文件以增强证据。"
        evidence_gap.append(
            {
                "dimension": dimension,
                "missing_from": "both",
                "suggestion": suggestion,
            }
        )
    deduped_gap: List[Dict[str, str]] = []
    seen_gap = set()
    for item in evidence_gap:
        key = (str(item.get("dimension") or ""), str(item.get("missing_from") or "both"))
        if key in seen_gap:
            continue
        seen_gap.add(key)
        deduped_gap.append(item)
    evidence_gap = deduped_gap

    if confidence == 0.0:
        next_action = "insufficient"
    elif coverage_payload and bool(coverage_payload.get("is_sufficient")):
        next_action = "complete"
    elif coverage_payload and not bool(coverage_payload.get("is_sufficient")):
        next_action = "needs_more_search"
    elif has_conflict or (
        any(str(child.get("status")) != "success" for child in [rag_child, web_child])
        if local_rag_enabled
        else web_status not in {"success", "partial"}
    ):
        next_action = "needs_more_search"
    else:
        next_action = "complete"

    return {
        "answer": {
            "conclusion": conclusion,
            "rag_insights": (
                _tagged_insight("[RAG]", rag_child, "本地知识库未返回可用结论。")
                if local_rag_enabled
                else "Local RAG disabled for this run."
            ),
            "web_insights": _tagged_insight("[WEB]", web_child, "联网搜索未返回可用结论。"),
            "consensus": consensus,
            "conflicts": conflicts,
        },
        "confidence": confidence,
        "agent_trace": _build_agent_trace(children),
        "evidence_gap": evidence_gap,
        "next_action": next_action,
        "supervisor_review": {
            "coverage_score": coverage_score if coverage_payload else None,
            "dimension_scores": coverage_payload.get("dimension_scores") if coverage_payload else {},
            "is_sufficient": coverage_payload.get("is_sufficient") if coverage_payload else None,
            "stop_reason": coverage_payload.get("stop_reason") if coverage_payload else None,
            "knowledge_gaps": coverage_payload.get("knowledge_gaps", []) if coverage_payload else [],
            "follow_up_queries": coverage_payload.get("follow_up_queries", []) if coverage_payload else [],
            "loop_trace": list(loop_trace or []),
            "evidence_pool_summary": evidence_pool_summary,
        },
    }


EVIDENCE_POOL_SUMMARY_PROMPT = """
## 证据摘要压缩

将以下多轮证据池压缩为结构化摘要，供 Supervisor 评估使用。

压缩规则：
1. 按 Planner 生成的动态证据目标归类，不要套用固定章节维度
2. 每个证据目标保留最关键的 2~3 条证据（优先保留可核验事实）
3. 标注证据来源类型（RAG/IQS）和轮次（第几轮获取）
4. 剔除重复内容，同一事实只保留一条（优先保留 IQS 的更新数据）
5. 总输出不超过 1500 字

输出格式：
【动态证据目标】
- [IQS·第1轮] 可核验事实、来源、时间范围与口径
- [RAG·第1轮] 内部知识库中与该证据目标相关的材料

【未覆盖证据目标】
- 某证据目标：无实质性证据或仅有模糊线索
""".strip()


SUPERVISOR_COVERAGE_SYSTEM_PROMPT = """
## 角色
你是 Research Compiler 的 Supervisor，负责判断当前证据是否足以支撑用户这一次研究任务，而不是套用固定行业报告维度。

## 输入
- 原始用户问题：user_payload.original_query
- 章节覆盖单元：user_payload.coverage_units
- 当前轮次：user_payload.loop_number
- 已收集证据：user_payload.evidence_pool_summary
- 上一轮覆盖率：user_payload.prev_coverage_score

## 评估原则
1. 先识别用户问题背后的决策、关键问题和证据目标。
2. 优先按 coverage_units 中的 chapter_id / unit_title 逐章评估，不要默认使用固定章节或固定行业五维。
3. 有可核验事实、来源、时间范围和口径说明可记为 1；只有线索记为 0.5；没有证据记为 0。
4. 若证据不足，只围绕缺失证据目标生成补充问题。

## 补充问题要求
- 问题必须具体，包含研究对象、缺口、来源类型或口径要求。
- agent 可以是 rag、iqs、both，或明确 IQS lane（iqs_lane_1 到 iqs_lane_6）。
- targets_gap 填动态证据目标名称。
- 如果能定位章节，补充问题携带 chapter_id、chapter_title、chapter_question。
- follow_up_queries 只能输出补证方向和 search task seed，不允许生成事实、结论或正文句子。
- 每个缺口必须映射 proof_role：metric/source_check/case/counter/support。
- metric 缺口必须显式 required_fields=["metric","value","unit","period","source"]。
- A/B 来源不足必须要求 lane_targets 包含 official_data、filing_company 或 market_research。
- counter 缺口必须搜索反向证据，不得写成支持性证据搜索。
- 输出 is_sufficient=false 时必须带 instruction="do_not_infer"。

## 输出格式（严格 JSON）
{
  "coverage_score": 0.0,
  "chapter_scores": {
    "章节标题": {"score": 0, "reason": "..."}
  },
  "is_sufficient": false,
  "instruction": "do_not_infer",
  "stop_reason": null,
  "knowledge_gaps": [
    {
      "dimension": "缺失的动态证据目标",
      "reason": "当前证据为什么不够",
      "severity": "critical | moderate | minor"
    }
  ],
  "follow_up_queries": [
    {
      "schema_version": "repair_task_seed_v2",
      "query": "补充搜索问题",
      "agent": "rag | iqs | both",
      "targets_gap": "对应动态证据目标",
      "gap_id": "可稳定复用的缺口ID",
      "requirement_id": "若可定位则填写",
      "chapter_id": "若可定位则填写",
      "section_id": "若可定位则填写",
      "gap_type": "metric_scope_period_unit_incomplete | insufficient_ab_sources | counter_evidence_missing | evidence_gap",
      "proof_role": "metric | source_check | case | counter | support",
      "required_fields": ["metric", "value", "unit", "period", "source"],
      "required_source_level": ["A", "B"],
      "lane_targets": ["official_data", "market_research"],
      "success_criteria": "只有补齐所需字段且来源可追溯才算修复",
      "reject_if": ["snippet_only", "no_date", "no_source_url", "marketing_copy_only"],
      "allowed_for_writing": false
    }
  ]
}
""".strip()


SUPERVISOR_SYSTEM_PROMPT = """
## 角色定位
你是行业研究多智能体系统的【Supervisor Agent】，系统的决策中枢。
你不只是汇总员，你负责：
  1. 评估两个子 Agent 输出的质量和可用性
  2. 整合证据，识别一致性与分歧
  3. 输出结构化决策包，供下游 Analysis Agent 直接使用
  4. 在证据不足时主动标注缺口并给出补充建议

## 系统位置
上游：industry_rag_agent（本地知识库）、web_analysis_agent（联网搜索、实时行情、财务与行业数值数据）
下游：Analysis Agent（将用你的输出生成行研框架分析）
原则：宁可输出"证据不足"，不可捏造或过度推断

---

## 输入格式
你将收到两个子 Agent 的 JSON 输出：
- industry_rag_agent 输出结构：{answer, confidence, key_sources, limitations}
- web_analysis_agent 输出结构：{answer, confidence, key_sources, limitations}
- 子 Agent 状态："success" | "partial" | "failed"
- 还会收到 supervisor_review：覆盖率评估、证据池摘要、多轮补充检索轨迹

---

## 整合规则（必须严格遵守）

### 规则 1：证据标注
- 所有来自 industry_rag_agent 的结论标注 [RAG]
- 所有来自 web_analysis_agent 的结论标注 [WEB]
- 子 Agent 的引用标记（如 [W0][P2]）必须原样保留，不得删改

### 规则 2：置信度计算
按以下规则计算最终 confidence（浮点数 0.0~1.0）：
  - 双 Agent 均成功 + 结论一致    → min(0.95, (rag_conf + web_conf) / 2 + 0.10)
  - 双 Agent 均成功 + 结论有分歧  → min(0.70, (rag_conf + web_conf) / 2)
  - 仅一个 Agent 成功             → 该 Agent 的 confidence × 0.85
  - 双 Agent 均失败或 confidence < 0.2 → 0.0，禁止输出实质性结论

### 规则 3：冲突处理优先级
当两个子 Agent 对同一维度结论不一致时：
  1. 时效性：IQS 数据发布时间 > RAG 数据 → 优先采信 WEB，但保留 RAG 作历史对比
  2. 权威性：政府/官方来源 > 媒体 > 分析师观点 → 权威来源优先
  3. 无法判断优先级 → 两者并列列出，标注待核验，不得自行裁定

### 规则 4：失败降级
  - 某 Agent 状态为 "failed" → 在 agent_trace 中记录失败原因，在 evidence_gap 中标注该维度缺口
  - 某 Agent 状态为 "partial" → 使用可用部分，在 limitations 中说明残缺程度
  - 双 Agent 均 failed → confidence = 0.0，answer.conclusion = "证据不足，无法输出有效判断"

### 规则 5：禁止行为
  - 不得引用子 Agent 未提供的数据
  - 不得将单一来源的结论表述为"行业普遍认为"
  - 不得删除子 Agent 原有的引用标记
  - 不得在 confidence > 0 的情况下将 evidence_gap 留空

---

## 输出格式（严格 JSON，禁止输出任何其他内容）

{
  "answer": {
    "conclusion":  "一句话核心判断（≤60字，必须有明确立场，不得模糊）",
    "rag_insights": "来自本地知识库的关键证据摘要，保留 [RAG][引用标记]",
    "web_insights": "来自联网搜索的关键证据摘要，保留 [WEB][引用标记]",
    "consensus":    "双方一致的结论（若无则填 null）",
    "conflicts": [
      {
        "dimension": "分歧所在维度",
        "rag_view":  "RAG 的结论",
        "web_view":  "WEB 的结论",
        "priority":  "rag_preferred | web_preferred | unresolved",
        "reason":    "优先选择的理由，或 unresolved 的原因"
      }
    ]
  },
  "confidence": 0.0,
  "agent_trace": [
    {
      "agent":      "industry_rag_agent | web_analysis_agent",
      "status":     "success | partial | failed",
      "confidence": 0.0,
      "used":       true,
      "note":       "说明（失败原因 / 部分使用的范围）"
    }
  ],
  "evidence_gap": [
    {
      "dimension":    "缺失的分析维度",
      "missing_from": "rag | web | both",
      "suggestion":   "建议补充的数据来源或搜索方向"
    }
  ],
  "next_action": "complete | needs_more_search | insufficient",
  "supervisor_review": {
    "coverage_score": 0.0,
    "dimension_scores": {},
    "is_sufficient": true,
    "stop_reason": "sufficient | max_loops_reached_with_gaps | coverage_stalled_with_gaps | followup_loop_disabled | null",
    "knowledge_gaps": [],
    "follow_up_queries": [],
    "loop_trace": [],
    "evidence_pool_summary": "证据池压缩摘要"
  }
}
""".strip()


def _normalize_supervisor_decision(
    payload: Dict[str, Any],
    *,
    fallback: Dict[str, Any],
    children: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    answer_payload = _as_dict(payload.get("answer"))
    fallback_answer = _as_dict(fallback.get("answer"))
    conflicts = [item for item in _as_list(answer_payload.get("conflicts")) if isinstance(item, dict)]
    if not conflicts:
        conflicts = [item for item in _as_list(fallback_answer.get("conflicts")) if isinstance(item, dict)]

    normalized_answer = {
        "conclusion": _compact_text(answer_payload.get("conclusion") or fallback_answer.get("conclusion"), max_chars=60),
        "rag_insights": str(answer_payload.get("rag_insights") or fallback_answer.get("rag_insights") or "").strip(),
        "web_insights": str(answer_payload.get("web_insights") or fallback_answer.get("web_insights") or "").strip(),
        "consensus": answer_payload.get("consensus") if answer_payload.get("consensus") not in ("", []) else fallback_answer.get("consensus"),
        "conflicts": conflicts,
    }

    confidence = _calculate_supervisor_confidence_v2(children, has_conflict=bool(conflicts))
    if confidence == 0.0:
        normalized_answer["conclusion"] = "证据不足，无法输出有效判断"

    agent_trace = [item for item in _as_list(payload.get("agent_trace")) if isinstance(item, dict)]
    if not agent_trace:
        agent_trace = list(fallback.get("agent_trace") or [])
    normalized_trace: List[Dict[str, Any]] = []
    for item in agent_trace:
        agent = str(item.get("agent") or "").strip()
        if agent not in {"industry_rag_agent", "web_analysis_agent"}:
            continue
        status = str(item.get("status") or "failed").strip()
        if status not in {"success", "partial", "failed"}:
            status = "failed"
        normalized_trace.append(
            {
                "agent": agent,
                "status": status,
                "confidence": _clip_confidence(item.get("confidence"), 0.0),
                "used": bool(item.get("used")),
                "note": str(item.get("note") or "").strip(),
            }
        )
    if len(normalized_trace) < 2:
        normalized_trace = _build_agent_trace(children)

    gaps = [item for item in _as_list(payload.get("evidence_gap")) if isinstance(item, dict)]
    if not gaps:
        gaps = list(fallback.get("evidence_gap") or [])
    if confidence > 0 and not gaps:
        gaps = _collect_evidence_gaps(children, confidence=confidence, has_conflict=bool(conflicts))
    normalized_gaps: List[Dict[str, str]] = []
    for item in gaps:
        missing_from = str(item.get("missing_from") or "both").strip()
        if missing_from not in {"rag", "web", "both"}:
            missing_from = "both"
        normalized_gaps.append(
            {
                "dimension": _compact_text(item.get("dimension"), max_chars=100),
                "missing_from": missing_from,
                "suggestion": _compact_text(item.get("suggestion"), max_chars=180),
            }
        )

    next_action = str(payload.get("next_action") or fallback.get("next_action") or "").strip()
    if next_action not in {"complete", "needs_more_search", "insufficient"}:
        if confidence == 0.0:
            next_action = "insufficient"
        elif bool(conflicts) or any(str((children.get(agent) or {}).get("status")) != "success" for agent in ["industry_rag_agent", "web_analysis_agent"]):
            next_action = "needs_more_search"
        else:
            next_action = "complete"

    fallback_review = _as_dict(fallback.get("supervisor_review"))
    payload_review = _as_dict(payload.get("supervisor_review"))
    supervisor_review = {**fallback_review, **payload_review}

    return {
        "answer": normalized_answer,
        "confidence": confidence,
        "agent_trace": normalized_trace,
        "evidence_gap": normalized_gaps,
        "next_action": next_action,
        "supervisor_review": supervisor_review,
    }


def merge_with_llm(
    *,
    query: str,
    route: str,
    route_reason: str,
    children: Dict[str, Dict[str, Any]],
    errors: Sequence[str],
    fallback_decision: Dict[str, Any],
    evidence_pool_summary: str = "",
    coverage_evaluation: Optional[Dict[str, Any]] = None,
    loop_trace: Optional[Sequence[Dict[str, Any]]] = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    llm_config = build_llm_config("evidence_merge")
    if not llm_config_is_ready(llm_config):
        raise RuntimeError("大脑 Agent 的大模型汇总配置不完整。")

    compact_children = compact_children_for_llm_merge(children)
    compact_loop_trace = [
        _compact_mapping_for_state(_as_dict(item), max_items=18, max_chars=180)
        for item in list(loop_trace or [])[:20]
        if isinstance(item, dict)
    ]
    user_payload = {
        "query": query,
        "route": route,
        "route_reason": route_reason,
        "child_outputs": compact_children,
        "child_errors": list(errors),
        "supervisor_review": {
            "coverage_evaluation": coverage_evaluation or {},
            "loop_trace": compact_loop_trace,
            "evidence_pool_summary": _compact_text(evidence_pool_summary, max_chars=4000),
        },
        "fallback_for_schema_reference": fallback_decision,
    }
    max_payload_chars = _env_int("BRAIN_MERGE_LLM_MAX_PAYLOAD_CHARS", 180000, min_value=20000, max_value=600000)
    payload_chars = len(json.dumps(user_payload, ensure_ascii=False, default=json_safe_default))
    if payload_chars > max_payload_chars:
        return fallback_decision, {
            "type": "brain_merge",
            "source": "supervisor_fallback_context_too_large",
            "skipped_reason": "merge_llm_skipped_context_too_large",
            "payload_chars": payload_chars,
            "max_payload_chars": max_payload_chars,
            "structured_payload": fallback_decision,
        }
    response = call_openai_compatible_json(
        config=llm_config,
        system_prompt=SUPERVISOR_SYSTEM_PROMPT,
        user_payload=user_payload,
    )
    payload = response.get("payload", {})
    if not isinstance(payload, dict) or not payload:
        raise RuntimeError("大脑 Agent 汇总回答为空。")
    decision = _normalize_supervisor_decision(payload, fallback=fallback_decision, children=children)
    return decision, {
        "type": "brain_merge",
        "source": "supervisor_llm",
        "model": normalize_llm_config(llm_config).get("model", ""),
        "usage": response.get("usage", {}),
        "structured_payload": decision,
        "llm_payload": payload,
    }


def merge_outputs_node(state: BrainAgentState) -> BrainAgentState:
    started = time.perf_counter()
    route = str(state.get("route") or "local")
    route_reason = str(state.get("route_reason") or "")
    errors = list(state.get("errors") or [])
    new_errors: List[str] = []
    enable_llm_merge = bool(state.get("enable_llm_merge", _env_flag("BRAIN_ENABLE_LLM_MERGE", True)))
    parallel_raw_output = bool(state.get("parallel_raw_output", _env_flag("BRAIN_PARALLEL_RAW_OUTPUT", False)))
    output_mode = str(state.get("output_mode") or os.getenv("BRAIN_OUTPUT_MODE", "writer_markdown")).strip().lower()
    query_analysis = _as_dict(state.get("query_analysis"))
    report_plan = _as_dict(query_analysis.get("report_plan"))
    research_plan = _research_plan_from_state(state)
    target_agents = set(_as_list(query_analysis.get("target_agents")) or _route_agents(route))
    lane_coverage = _lane_coverage_from_state(state)
    lane_health_summary = _lane_health_summary_from_coverage(lane_coverage)
    _progress("merge", "结构化汇总开始", route=route, output_mode=output_mode, children=len(target_agents))

    children = {
        "industry_rag_agent": normalize_rag_child_output(state.get("local_state"), route=route, errors=errors),
    }
    role_children: Dict[str, Dict[str, Any]] = {}
    for role_key in IQS_ROLE_ORDER:
        config = IQS_ROLE_CONFIGS[role_key]
        role_state = state.get(config["state"])  # type: ignore[literal-required]
        if role_key in target_agents or role_state:
            role_errors = _child_error_messages(errors, [config["label"], config["child"]])
            role_children[config["child"]] = normalize_iqs_role_child_output(
                role_key,
                role_state,
                route=route,
                errors=role_errors,
            )
    children.update(role_children)
    if role_children:
        children["web_analysis_agent"] = aggregate_iqs_role_children(children)
    else:
        children["web_analysis_agent"] = normalize_web_child_output(state.get("web_state"), route=route, errors=errors)
    if output_mode == "agent_text":
        loop_result = run_supervisor_evidence_loop(state=state, initial_children=children, route=route)
        children = loop_result["children"]
        fact_extractor_diagnostics = _aggregate_readpage_fact_extractor_diagnostics(children)
        loop_errors = list(loop_result.get("evaluation_errors") or [])
        new_errors.extend(loop_errors)
        all_errors = errors + new_errors
        evidence_pool = _merge_topic_seed_with_live_evidence(
            state,
            [item for item in _as_list(loop_result.get("evidence_pool")) if isinstance(item, dict)],
        )
        evidence_package = merge_evidence_package(
            original_query=str(state.get("query") or ""),
            evidence_pool=[item for item in list(evidence_pool or []) if isinstance(item, dict)],
            children=children,
            research_plan=research_plan,
        )
        topic_seed_summary = _topic_bundle_seed_summary(state, evidence_pool)
        if topic_seed_summary:
            evidence_package.setdefault("metadata", {})["topic_bundle_seed"] = topic_seed_summary
        evidence_package = _annotate_evidence_package_runtime(evidence_package, lane_coverage=lane_coverage, state=state)
        _attach_readpage_fact_extractor_diagnostics(evidence_package, diagnostics=fact_extractor_diagnostics)
        if report_plan:
            evidence_package["report_plan"] = report_plan
            evidence_package.setdefault("metadata", {})
            evidence_package["metadata"]["report_plan"] = report_plan
        analysis_state = run_analysis_agent(evidence_package, query=str(state.get("query") or ""), llm_config=build_llm_config("decision"))
        structured_analysis = _as_dict(analysis_state.get("structured_analysis"))
        if report_plan:
            structured_analysis["report_plan"] = report_plan
        _attach_research_plan(evidence_package, structured_analysis, research_plan)
        _sync_analysis_repair_priorities_to_evidence_package(evidence_package, structured_analysis)
        _emit_pre_writer_snapshots(state, evidence_package, structured_analysis)
        topic_bundle_store = _store_topic_bundle_from_brain(
            state=state,
            evidence_package=evidence_package,
            structured_analysis=structured_analysis,
            stage="brain_full_payload",
        )
        if topic_bundle_store:
            evidence_package.setdefault("metadata", {})["topic_bundle_cache_store"] = topic_bundle_store
        writer_bundle = run_writer_with_layout_refinement(
            state=state,
            children=children,
            evidence_pool=evidence_pool,
            evidence_package=evidence_package,
            structured_analysis=structured_analysis,
            report_plan=report_plan,
            analysis_state=analysis_state,
        )
        writer_state = _as_dict(writer_bundle.get("writer_state"))
        writer_report = _attach_lane_health_to_writer_report(
            _as_dict(writer_bundle.get("writer_report")),
            lane_coverage=lane_coverage,
            state=state,
        )
        evidence_pool = [item for item in _as_list(writer_bundle.get("evidence_pool")) if isinstance(item, dict)]
        evidence_package = _as_dict(writer_bundle.get("evidence_package")) or evidence_package
        structured_analysis = _as_dict(writer_bundle.get("structured_analysis")) or structured_analysis
        _attach_readpage_fact_extractor_diagnostics(evidence_package, writer_report, fact_extractor_diagnostics)
        writer_topic_bundle_store = _store_topic_bundle_from_brain(
            state=state,
            evidence_package=evidence_package,
            structured_analysis=structured_analysis,
            writer_report=writer_report,
            stage="writer_full_payload",
        )
        if writer_topic_bundle_store:
            writer_report["topic_bundle_cache_store"] = writer_topic_bundle_store
            evidence_package.setdefault("metadata", {})["topic_bundle_cache_store_writer"] = writer_topic_bundle_store
        _emit_post_writer_snapshot(state, writer_report)
        analysis_state = _as_dict(writer_bundle.get("analysis_state")) or analysis_state
        layout_refinement_trace = _as_list(writer_bundle.get("layout_refinement_trace"))
        evidence_preflight_trace = _as_list(writer_bundle.get("evidence_preflight_trace"))
        post_qa_repair_trace = _as_list(writer_bundle.get("post_qa_repair_trace"))
        loop_health_summary = build_loop_health_summary(
            supervisor_trace=_as_list(loop_result.get("loop_trace")),
            evidence_preflight_trace=evidence_preflight_trace,
            layout_refinement_trace=layout_refinement_trace,
            post_qa_repair_trace=post_qa_repair_trace,
        )
        analysis_raw_output = _as_dict(analysis_state.get("raw_output"))
        analysis_meta = _as_dict(analysis_raw_output.get("analysis"))
        analysis_errors = [str(item) for item in analysis_state.get("errors") or [] if str(item).strip()]
        if analysis_errors:
            additions = [f"Analysis Agent：{item}" for item in analysis_errors]
            new_errors.extend(additions)
            all_errors.extend(additions)
        writer_raw_output = _as_dict(writer_state.get("raw_output"))
        writer_meta = _as_dict(writer_raw_output.get("writer"))
        writer_errors = [str(item) for item in writer_state.get("errors") or [] if str(item).strip()]
        if writer_errors:
            additions = [f"Writer Agent：{item}" for item in writer_errors]
            new_errors.extend(additions)
            all_errors.extend(additions)
        writer_handoff_package = {
            "evidence_package": evidence_package,
            "structured_analysis": structured_analysis,
            "writer_report": writer_report,
            "writer_constraints": {
                "writer_can_use_iqs_materials_directly": False,
                "writer_must_use_analysis_outputs_for_judgment": True,
                "raw_evidence_is_source_anchor_only": True,
                "writer_can_only_use_provided_materials": True,
                "final_writer_consumes_structured_packages_only": True,
                "do_not_invent_facts": True,
                "cite_sources": True,
            },
        }
        answer_text = render_agent_text_output(
            query_analysis=query_analysis,
            children=children,
            evidence_pool=evidence_pool,
            evidence_package=evidence_package,
            structured_analysis=structured_analysis,
            analysis_meta=analysis_meta,
            writer_report=writer_report,
            writer_meta=writer_meta,
            include_child_details=_env_flag("BRAIN_AGENT_TEXT_INCLUDE_CHILD_DETAILS", False),
            include_writer_report=_env_flag("BRAIN_AGENT_TEXT_INCLUDE_WRITER_REPORT", True),
        )
        reformatter_evidence_package = _reformatter_evidence_package_for_handoff(
            evidence_package,
            writer_report,
            structured_analysis,
        )
        raw_output: Dict[str, Any] = {
            "query": state.get("query", ""),
            "route": route,
            "route_reason": route_reason,
            "query_analysis": _as_dict(state.get("query_analysis")),
            "search_task_schedule": _as_dict(state.get("search_task_schedule")) or _as_dict(query_analysis.get("search_task_schedule")),
            "lane_coverage": lane_coverage,
            "lane_health_summary": lane_health_summary,
            "graph_trace": state.get("agent_trace", []),
            "child_outputs": _state_payload(children, "children"),
            "evidence_pool": _state_payload(evidence_pool, "evidence_pool"),
            "evidence_package": _state_payload(evidence_package, "evidence_package"),
            "structured_analysis": _state_payload(structured_analysis, "structured_analysis"),
            "self_refine_trace": [],
            "coverage_evaluation": loop_result.get("coverage_evaluation", {}),
            "loop_trace": loop_result.get("loop_trace", []),
            "evidence_pool_summary": loop_result.get("evidence_pool_summary", ""),
            "loop_health_summary": loop_health_summary,
            "evidence_preflight_trace": evidence_preflight_trace,
            "layout_refinement_trace": layout_refinement_trace,
            "post_qa_repair_trace": post_qa_repair_trace,
            "merge": {
                "type": "agent_text_structured_processing",
                "source": "evidence_merger_analysis_agent",
                "analysis_source": analysis_meta.get("source"),
            },
            "analysis": analysis_meta,
            "analysis_errors": analysis_errors,
            "writer": writer_meta,
            "writer_errors": writer_errors,
            "writer_report": _state_payload(writer_report, "writer_report"),
            "writer_handoff_package": _state_payload(writer_handoff_package, "writer_handoff_package"),
            "output_mode": output_mode,
            "payload_mode": "full" if _brain_full_payloads() else "summary",
        }
        if fact_extractor_diagnostics:
            raw_output["fact_extractor"] = fact_extractor_diagnostics
            raw_output.setdefault("metadata", {})["readpage_fact_extractor"] = fact_extractor_diagnostics
        if parallel_raw_output:
            raw_output["local_state"] = state.get("local_state", {})
            raw_output["web_state"] = state.get("web_state", {})
            for role_key in IQS_ROLE_ORDER:
                config = IQS_ROLE_CONFIGS[role_key]
                raw_output[config["state"]] = state.get(config["state"], {})
        _progress(
            "merge",
            "结构化汇总完成",
            mode=output_mode,
            evidence=len(evidence_pool),
            elapsed=f"{time.perf_counter() - started:.1f}s",
        )
        return {
            "answer_text": answer_text,
            "raw_output": raw_output,
            "evidence_package": _state_payload(evidence_package, "evidence_package"),
            "reformatter_evidence_package": reformatter_evidence_package,
            "structured_analysis": _state_payload(structured_analysis, "structured_analysis"),
            "writer_report": _state_payload(writer_report, "writer_report"),
            "evidence_preflight_trace": _state_payload(evidence_preflight_trace, "evidence_preflight_trace"),
            "post_qa_repair_trace": _state_payload(post_qa_repair_trace, "post_qa_repair_trace"),
            "loop_health_summary": _state_payload(loop_health_summary, "loop_health_summary"),
            **_writer_pipeline_state_fields(writer_report),
            "lane_coverage": lane_coverage,
            "lane_health_summary": lane_health_summary,
            "errors": new_errors,
            "metadata": {
                **dict(state.get("metadata") or {}),
                "agent_stage": "merge_outputs",
                "merge_source": "agent_text_structured_processing",
                "analysis_source": analysis_meta.get("source"),
                "writer_source": writer_meta.get("source"),
                "readpage_fact_extractor": fact_extractor_diagnostics,
            "layout_refinement_rounds": _layout_refinement_round_count(layout_refinement_trace),
            "evidence_preflight_rounds": len(evidence_preflight_trace),
            "post_qa_repair_rounds": len(post_qa_repair_trace),
            "coverage_score": _as_dict(evidence_package.get("summary")).get("overall_coverage"),
                "next_action": "display_structured_agent_outputs",
            },
        }

    loop_result = run_supervisor_evidence_loop(state=state, initial_children=children, route=route)
    children = loop_result["children"]
    fact_extractor_diagnostics = _aggregate_readpage_fact_extractor_diagnostics(children)
    loop_errors = list(loop_result.get("evaluation_errors") or [])
    new_errors.extend(loop_errors)
    all_errors = errors + new_errors
    loop_evidence_pool = _merge_topic_seed_with_live_evidence(
        state,
        [item for item in list(loop_result.get("evidence_pool") or []) if isinstance(item, dict)],
    )
    evidence_package = merge_evidence_package(
        original_query=str(state.get("query") or ""),
        evidence_pool=loop_evidence_pool,
        children=children,
        research_plan=research_plan,
    )
    topic_seed_summary = _topic_bundle_seed_summary(state, loop_evidence_pool)
    if topic_seed_summary:
        evidence_package.setdefault("metadata", {})["topic_bundle_seed"] = topic_seed_summary
    evidence_package = _annotate_evidence_package_runtime(evidence_package, lane_coverage=lane_coverage, state=state)
    _attach_readpage_fact_extractor_diagnostics(evidence_package, diagnostics=fact_extractor_diagnostics)
    if report_plan:
        evidence_package["report_plan"] = report_plan
        evidence_package.setdefault("metadata", {})
        evidence_package["metadata"]["report_plan"] = report_plan
    analysis_state = run_analysis_agent(
        evidence_package,
        query=str(state.get("query") or ""),
        llm_config=build_llm_config("decision"),
        deadline_ts=_analysis_deadline_ts_for_state(state),
    )
    structured_analysis = _as_dict(analysis_state.get("structured_analysis"))
    if report_plan:
        structured_analysis["report_plan"] = report_plan
    _attach_research_plan(evidence_package, structured_analysis, research_plan)
    _sync_analysis_repair_priorities_to_evidence_package(evidence_package, structured_analysis)
    _emit_pre_writer_snapshots(state, evidence_package, structured_analysis)
    if _deadline_exceeded(state, min_remaining=1.0):
        timeout_payload = _deadline_timeout_payload(state, stage="merge_outputs_pre_writer")
        return {
            "answer_text": "",
            "raw_output": {
                "query": state.get("query", ""),
                "route": route,
                "route_reason": route_reason,
                "query_analysis": _as_dict(state.get("query_analysis")),
                "search_task_schedule": _as_dict(state.get("search_task_schedule")) or _as_dict(query_analysis.get("search_task_schedule")),
                "lane_coverage": lane_coverage,
                "child_outputs": _state_payload(children, "children"),
                "evidence_pool": _state_payload(loop_evidence_pool, "evidence_pool"),
                "evidence_package": evidence_package,
                "structured_analysis": structured_analysis,
                "output_mode": output_mode,
                "payload_mode": "deadline_partial",
                "live_timeout": timeout_payload,
            },
            "evidence_package": evidence_package,
            "structured_analysis": structured_analysis,
            "errors": [*new_errors, "Report deadline reached after evidence_package; fail-open rebuild is required."],
            "metadata": {
                **dict(state.get("metadata") or {}),
                "agent_stage": "merge_outputs",
                "merge_source": "deadline_partial_after_evidence_package",
                "live_timeout": timeout_payload,
                "readpage_fact_extractor": fact_extractor_diagnostics,
            },
        }
    topic_bundle_store = _store_topic_bundle_from_brain(
        state=state,
        evidence_package=evidence_package,
        structured_analysis=structured_analysis,
        stage="brain_full_payload",
    )
    if topic_bundle_store:
        evidence_package.setdefault("metadata", {})["topic_bundle_cache_store"] = topic_bundle_store
    analysis_errors = [str(item) for item in analysis_state.get("errors") or [] if str(item).strip()]
    if analysis_errors:
        additions = [f"Analysis Agent：{item}" for item in analysis_errors]
        new_errors.extend(additions)
        all_errors.extend(additions)
    fallback_decision = build_supervisor_decision(
        query=str(state.get("query") or ""),
        route=route,
        route_reason=route_reason,
        children=children,
        errors=all_errors,
        coverage_evaluation=loop_result.get("coverage_evaluation"),
        loop_trace=loop_result.get("loop_trace"),
        evidence_pool_summary=str(loop_result.get("evidence_pool_summary") or ""),
    )
    merge_meta: Dict[str, Any] = {
        "type": "brain_merge",
        "source": "supervisor_fallback",
        "structured_payload": fallback_decision,
    }
    decision = fallback_decision
    if enable_llm_merge and any(str(child.get("answer") or "").strip() for child in children.values()):
        try:
            decision, merge_meta = merge_with_llm(
                query=str(state.get("query") or ""),
                route=route,
                route_reason=route_reason,
                children=children,
                errors=all_errors,
                fallback_decision=fallback_decision,
                evidence_pool_summary=str(loop_result.get("evidence_pool_summary") or ""),
                coverage_evaluation=loop_result.get("coverage_evaluation"),
                loop_trace=loop_result.get("loop_trace"),
            )
        except Exception as exc:
            merge_meta = {**merge_meta, "error": str(exc), "source": "supervisor_fallback_after_llm_error"}
            error_text = f"Supervisor Merge LLM：{exc}"
            new_errors.append(error_text)
            all_errors.append(error_text)

    evidence_pool = loop_evidence_pool
    report_package = build_report_package(decision, evidence_pool)
    writer_bundle = run_writer_with_layout_refinement(
        state=state,
        children=children,
        evidence_pool=evidence_pool,
        evidence_package=evidence_package,
        structured_analysis=structured_analysis,
        report_plan=report_plan,
        analysis_state=analysis_state,
    )
    writer_state = _as_dict(writer_bundle.get("writer_state"))
    writer_report = _attach_lane_health_to_writer_report(
        _as_dict(writer_bundle.get("writer_report")),
        lane_coverage=lane_coverage,
        state=state,
    )
    evidence_pool = [item for item in _as_list(writer_bundle.get("evidence_pool")) if isinstance(item, dict)]
    evidence_package = _as_dict(writer_bundle.get("evidence_package")) or evidence_package
    structured_analysis = _as_dict(writer_bundle.get("structured_analysis")) or structured_analysis
    _attach_readpage_fact_extractor_diagnostics(evidence_package, writer_report, fact_extractor_diagnostics)
    writer_topic_bundle_store = _store_topic_bundle_from_brain(
        state=state,
        evidence_package=evidence_package,
        structured_analysis=structured_analysis,
        writer_report=writer_report,
        stage="writer_full_payload",
    )
    if writer_topic_bundle_store:
        writer_report["topic_bundle_cache_store"] = writer_topic_bundle_store
        evidence_package.setdefault("metadata", {})["topic_bundle_cache_store_writer"] = writer_topic_bundle_store
    _emit_post_writer_snapshot(state, writer_report)
    analysis_state = _as_dict(writer_bundle.get("analysis_state")) or analysis_state
    layout_refinement_trace = _as_list(writer_bundle.get("layout_refinement_trace"))
    evidence_preflight_trace = _as_list(writer_bundle.get("evidence_preflight_trace"))
    post_qa_repair_trace = _as_list(writer_bundle.get("post_qa_repair_trace"))
    loop_health_summary = build_loop_health_summary(
        supervisor_trace=_as_list(loop_result.get("loop_trace")),
        evidence_preflight_trace=evidence_preflight_trace,
        layout_refinement_trace=layout_refinement_trace,
        post_qa_repair_trace=post_qa_repair_trace,
    )
    report_package = build_report_package(decision, evidence_pool)
    writer_raw_output = _as_dict(writer_state.get("raw_output"))
    writer_meta = _as_dict(writer_raw_output.get("writer"))
    writer_errors = [str(item) for item in writer_state.get("errors") or [] if str(item).strip()]
    if writer_errors:
        additions = [f"Writer Agent：{item}" for item in writer_errors]
        new_errors.extend(additions)
        all_errors.extend(additions)
    writer_handoff_package = {
        "evidence_package": evidence_package,
        "structured_analysis": structured_analysis,
        "legacy_report_package": report_package,
        "writer_report": writer_report,
        "writer_constraints": {
            "writer_can_use_iqs_materials_directly": False,
            "writer_must_use_analysis_outputs_for_judgment": True,
            "raw_evidence_is_source_anchor_only": True,
            "writer_can_only_use_provided_materials": True,
            "final_writer_consumes_structured_packages_only": True,
            "do_not_invent_facts": True,
            "cite_sources": True,
        },
    }
    compact_json = not _brain_full_payloads()
    json_kwargs = {"ensure_ascii": False, "default": json_safe_default}
    if not compact_json:
        json_kwargs["indent"] = 2
    else:
        json_kwargs["separators"] = (",", ":")
    if output_mode == "supervisor_json":
        answer_text = json.dumps(decision, **json_kwargs)
    elif output_mode == "evidence_package_json":
        payload = {"evidence_package": _state_payload(evidence_package, "evidence_package") if compact_json else evidence_package}
        answer_text = json.dumps(payload, **json_kwargs)
    elif output_mode == "analysis_json":
        payload = {"structured_analysis": _state_payload(structured_analysis, "structured_analysis") if compact_json else structured_analysis}
        answer_text = json.dumps(payload, **json_kwargs)
    elif output_mode == "writer_markdown":
        answer_text = str(writer_report.get("report_markdown") or "")
    elif output_mode == "brief_text":
        answer_text = render_brief_report_text(report_package)
    else:
        payload = _state_payload(writer_handoff_package, "writer_handoff_package") if compact_json else writer_handoff_package
        answer_text = json.dumps(payload, **json_kwargs)
    raw_output: Dict[str, Any] = {
        "query": state.get("query", ""),
        "route": route,
        "route_reason": route_reason,
        "search_task_schedule": _as_dict(state.get("search_task_schedule")) or _as_dict(query_analysis.get("search_task_schedule")),
        "lane_coverage": lane_coverage,
        "lane_health_summary": lane_health_summary,
        "graph_trace": state.get("agent_trace", []),
        "child_outputs": _state_payload(children, "children"),
        "evidence_pool": _state_payload(evidence_pool, "evidence_pool"),
        "evidence_package": _state_payload(evidence_package, "evidence_package"),
        "structured_analysis": _state_payload(structured_analysis, "structured_analysis"),
        "evidence_pool_summary": loop_result.get("evidence_pool_summary", ""),
        "coverage_evaluation": loop_result.get("coverage_evaluation", {}),
        "loop_trace": loop_result.get("loop_trace", []),
        "loop_health_summary": loop_health_summary,
        "evidence_preflight_trace": evidence_preflight_trace,
        "layout_refinement_trace": layout_refinement_trace,
        "post_qa_repair_trace": post_qa_repair_trace,
        "merge": merge_meta,
        "analysis": _as_dict(analysis_state.get("raw_output")).get("analysis", {}),
        "writer": writer_meta,
        "supervisor_decision": decision,
        "report_package": report_package,
        "writer_report": _state_payload(writer_report, "writer_report"),
        "writer_handoff_package": _state_payload(writer_handoff_package, "writer_handoff_package"),
        "output_mode": output_mode,
        "payload_mode": "full" if _brain_full_payloads() else "summary",
    }
    if fact_extractor_diagnostics:
        raw_output["fact_extractor"] = fact_extractor_diagnostics
        raw_output.setdefault("metadata", {})["readpage_fact_extractor"] = fact_extractor_diagnostics
    if parallel_raw_output:
        raw_output["local_state"] = state.get("local_state", {})
        raw_output["web_state"] = state.get("web_state", {})
        for role_key in IQS_ROLE_ORDER:
            config = IQS_ROLE_CONFIGS[role_key]
            raw_output[config["state"]] = state.get(config["state"], {})

    reformatter_evidence_package = _reformatter_evidence_package_for_handoff(
        evidence_package,
        writer_report,
        structured_analysis,
    )
    _progress(
        "merge",
        "结构化汇总完成",
        mode=output_mode,
        evidence=len(evidence_pool),
        elapsed=f"{time.perf_counter() - started:.1f}s",
    )
    return {
        "answer_text": answer_text,
        "raw_output": raw_output,
        "evidence_package": _state_payload(evidence_package, "evidence_package"),
        "reformatter_evidence_package": reformatter_evidence_package,
        "structured_analysis": _state_payload(structured_analysis, "structured_analysis"),
        "writer_report": _state_payload(writer_report, "writer_report"),
        "evidence_preflight_trace": _state_payload(evidence_preflight_trace, "evidence_preflight_trace"),
        "post_qa_repair_trace": _state_payload(post_qa_repair_trace, "post_qa_repair_trace"),
        "loop_health_summary": _state_payload(loop_health_summary, "loop_health_summary"),
        **_writer_pipeline_state_fields(writer_report),
        "lane_coverage": lane_coverage,
        "lane_health_summary": lane_health_summary,
        "errors": new_errors,
        "metadata": {
            **dict(state.get("metadata") or {}),
            "agent_stage": "merge_outputs",
            "merge_source": merge_meta.get("source", "fallback"),
            "analysis_source": _as_dict(_as_dict(analysis_state.get("raw_output")).get("analysis")).get("source"),
            "writer_source": writer_meta.get("source"),
            "readpage_fact_extractor": fact_extractor_diagnostics,
            "layout_refinement_rounds": _layout_refinement_round_count(layout_refinement_trace),
            "evidence_preflight_rounds": len(evidence_preflight_trace),
            "post_qa_repair_rounds": len(post_qa_repair_trace),
            "supervisor_confidence": decision.get("confidence"),
            "coverage_score": _as_dict(evidence_package.get("summary")).get("overall_coverage"),
            "next_action": decision.get("next_action"),
        },
    }


def format_response_node(state: BrainAgentState) -> BrainAgentState:
    errors = list(state.get("errors") or [])
    answer_text = str(state.get("answer_text") or "").strip()
    if not answer_text:
        answer_text = "大脑 Agent 失败：" + (errors[-1] if errors else "没有生成可用回答。")

    messages = list(state.get("messages") or [])
    messages.append({"role": "assistant", "name": AGENT_NAME, "content": answer_text})
    return {
        "answer_text": answer_text,
        "messages": messages,
        "metadata": {
            **dict(state.get("metadata") or {}),
            "agent_stage": "format_response",
            "handoff_ready": bool(answer_text),
        },
    }


def create_brain_agent_graph(*, name: str = AGENT_NAME):
    builder = StateGraph(BrainAgentState)
    builder.add_node("decompose_query", prepare_query_node)
    builder.add_node("route", route_node)
    builder.add_node("industry_rag_agent", run_local_rag_agent_node)
    builder.add_node("web_analysis_agent", run_web_analysis_agent_node)
    builder.add_node("iqs_lane_1_agent", run_iqs_lane_1_agent_node)
    builder.add_node("iqs_lane_2_agent", run_iqs_lane_2_agent_node)
    builder.add_node("iqs_lane_3_agent", run_iqs_lane_3_agent_node)
    builder.add_node("iqs_lane_4_agent", run_iqs_lane_4_agent_node)
    builder.add_node("iqs_lane_5_agent", run_iqs_lane_5_agent_node)
    builder.add_node("iqs_lane_6_agent", run_iqs_lane_6_agent_node)
    builder.add_node("merge_outputs", merge_outputs_node)
    builder.add_node("format_response", format_response_node)
    builder.add_edge(START, "decompose_query")
    builder.add_edge("decompose_query", "route")
    builder.add_conditional_edges(
        "route",
        select_child_agents,
        {
            "industry_rag_agent": "industry_rag_agent",
            "web_analysis_agent": "web_analysis_agent",
            "iqs_lane_1_agent": "iqs_lane_1_agent",
            "iqs_lane_2_agent": "iqs_lane_2_agent",
            "iqs_lane_3_agent": "iqs_lane_3_agent",
            "iqs_lane_4_agent": "iqs_lane_4_agent",
            "iqs_lane_5_agent": "iqs_lane_5_agent",
            "iqs_lane_6_agent": "iqs_lane_6_agent",
            "merge_outputs": "merge_outputs",
        },
    )
    builder.add_edge("industry_rag_agent", "merge_outputs")
    builder.add_edge("web_analysis_agent", "merge_outputs")
    builder.add_edge("iqs_lane_1_agent", "merge_outputs")
    builder.add_edge("iqs_lane_2_agent", "merge_outputs")
    builder.add_edge("iqs_lane_3_agent", "merge_outputs")
    builder.add_edge("iqs_lane_4_agent", "merge_outputs")
    builder.add_edge("iqs_lane_5_agent", "merge_outputs")
    builder.add_edge("iqs_lane_6_agent", "merge_outputs")
    builder.add_edge("merge_outputs", "format_response")
    builder.add_edge("format_response", END)
    return builder.compile(name=name)


def run_brain_agent(
    query: str,
    *,
    messages: Optional[Sequence[Dict[str, Any]]] = None,
    article_brief: Optional[Dict[str, Any]] = None,
    session_id: str = "",
    route: str = "auto",
    args_overrides: Optional[Dict[str, Any]] = None,
    web_search_options: Optional[Dict[str, Any]] = None,
    enable_web_analysis: Optional[bool] = None,
    enable_llm_merge: Optional[bool] = None,
    enable_followup_loop: Optional[bool] = None,
    supervisor_max_loops: Optional[int] = None,
    supervisor_min_coverage_gain: Optional[float] = None,
    supervisor_max_followup_queries: Optional[int] = None,
    layout_max_refinement_rounds: Optional[int] = None,
    output_mode: Optional[str] = None,
    parallel_raw_output: Optional[bool] = None,
    topic_bundle_seed: Optional[Dict[str, Any]] = None,
    deadline_ts: Optional[float] = None,
    timeout_context: Optional[Dict[str, Any]] = None,
    fail_open_on_timeout: Optional[bool] = None,
) -> BrainAgentState:
    configure_pipeline_logging()
    started = time.perf_counter()
    graph = create_brain_agent_graph()
    brief = normalize_article_brief(article_brief, fallback_query=query) if article_brief else {}
    query = planning_query_from_brief(brief, fallback_query=query) if brief else str(query or "")
    initial_messages = list(messages or [])
    if query and not initial_messages:
        initial_messages.append({"role": "user", "content": query})
    state: BrainAgentState = {
        "query": query,
        "article_brief": brief,
        "messages": initial_messages,
        "session_id": session_id,
        "route": route,
        "args_overrides": dict(args_overrides or {}),
        "web_search_options": dict(web_search_options or {}),
    }
    if topic_bundle_seed:
        state["topic_bundle_seed"] = dict(topic_bundle_seed)
    if deadline_ts is not None:
        state["deadline_ts"] = float(deadline_ts)
    if timeout_context:
        state["timeout_context"] = dict(timeout_context)
    if fail_open_on_timeout is not None:
        state["fail_open_on_timeout"] = bool(fail_open_on_timeout)
    if enable_web_analysis is not None:
        state["enable_web_analysis"] = bool(enable_web_analysis)
    if enable_llm_merge is not None:
        state["enable_llm_merge"] = bool(enable_llm_merge)
    if enable_followup_loop is not None:
        state["enable_followup_loop"] = bool(enable_followup_loop)
    if supervisor_max_loops is not None:
        state["supervisor_max_loops"] = int(supervisor_max_loops)
    if supervisor_min_coverage_gain is not None:
        state["supervisor_min_coverage_gain"] = float(supervisor_min_coverage_gain)
    if supervisor_max_followup_queries is not None:
        state["supervisor_max_followup_queries"] = int(supervisor_max_followup_queries)
    if layout_max_refinement_rounds is not None:
        state["layout_max_refinement_rounds"] = int(layout_max_refinement_rounds)
    if output_mode is not None:
        state["output_mode"] = str(output_mode or "").strip()
    if parallel_raw_output is not None:
        state["parallel_raw_output"] = bool(parallel_raw_output)
    _progress(
        "brain",
        "主图运行开始",
        route=state.get("route"),
        output_mode=state.get("output_mode") or os.getenv("BRAIN_OUTPUT_MODE", "writer_markdown"),
        query=query,
    )
    graph_config = {"recursion_limit": max(25, _env_int("BRAIN_GRAPH_RECURSION_LIMIT", 80))}
    result = graph.invoke(state, config=graph_config)
    _progress(
        "brain",
        "主图运行完成",
        route=result.get("route") or state.get("route"),
        output_chars=len(str(result.get("answer_text") or "")),
        elapsed=f"{time.perf_counter() - started:.1f}s",
    )
    return result


def create_brain_agent_tool():
    """返回可被更上层 supervisor 调用的 LangChain 兼容工具。"""

    from langchain_core.tools import tool

    @tool("brain_agent", description=AGENT_DESCRIPTION)
    def _brain_agent(query: str) -> str:
        return str(run_brain_agent(query).get("answer_text") or "")

    return _brain_agent


def build_arg_parser() -> argparse.ArgumentParser:
    parser = build_rag_arg_parser()
    parser.description = "行业研究多智能体系统的大脑 Agent。"
    parser.add_argument("--route", choices=["auto", "local", "web", "both", "all"], default=os.getenv("BRAIN_AGENT_ROUTE", "auto"), help="大脑 Agent 路由策略。默认本地 RAG 关闭；local/both/all 会映射到 IQS-only，除非 BRAIN_ENABLE_LOCAL_RAG=1。")
    parser.add_argument("--web-engine-type", default="", help="联网子 Agent 的 IQS 搜索引擎类型/资源名；填 auto 时按意图使用 .env 中的资源池。")
    parser.add_argument("--web-time-range", choices=["NoLimit", "OneDay", "OneWeek", "OneMonth", "OneYear"], default="", help="联网子 Agent 的 IQS 搜索时间范围。")
    parser.add_argument("--web-contents", choices=["summary", "mainText"], default="", help="联网子 Agent 返回内容类型。")
    parser.add_argument("--web-num-results", type=int, default=0, help="联网子 Agent 搜索结果数量。")
    parser.add_argument("--web-timeout-ms", type=int, default=0, help="联网子 Agent 搜索超时时间。")
    parser.add_argument("--disable-web-llm-analysis", dest="enable_web_analysis", action="store_false", default=_env_flag("BRAIN_WEB_ENABLE_LLM_ANALYSIS", True), help="关闭联网子 Agent 的大模型综合分析。")
    parser.add_argument("--disable-brain-llm-merge", dest="enable_llm_merge", action="store_false", default=_env_flag("BRAIN_ENABLE_LLM_MERGE", True), help="关闭大脑 Agent 的大模型汇总。")
    parser.add_argument("--disable-followup-loop", dest="enable_followup_loop", action="store_false", default=_env_flag("BRAIN_ENABLE_FOLLOWUP_LOOP", True), help="关闭 Supervisor 覆盖率评估后的自动补充检索闭环。")
    parser.add_argument("--supervisor-max-loops", type=int, default=_env_int("BRAIN_SUPERVISOR_MAX_LOOPS", 3), help="Supervisor 补充检索最大轮数，默认 3。")
    parser.add_argument("--supervisor-min-coverage-gain", type=float, default=_env_float("BRAIN_SUPERVISOR_MIN_COVERAGE_GAIN", 0.10), help="每轮覆盖率最低提升阈值，低于或等于该值则停止补证。")
    parser.add_argument("--supervisor-max-followup-queries", type=int, default=_env_int("BRAIN_SUPERVISOR_MAX_FOLLOWUP_QUERIES", 4), help="每轮最多生成的补充问题数。")
    parser.add_argument("--layout-max-refinement-rounds", type=int, default=_env_int("BRAIN_LAYOUT_MAX_REFINEMENT_ROUNDS", 3), help="Layout Planner 驱动的补证与重写最大轮数，默认 3，上限 6。")
    parser.add_argument(
        "--output-mode",
        choices=[
            "agent_text",
            "evidence_package_json",
            "analysis_json",
            "report_json",
            "writer_markdown",
            "brief_text",
            "supervisor_json",
        ],
        default=os.getenv("BRAIN_OUTPUT_MODE", "writer_markdown"),
        help=(
            "agent_text=展示问题拆解、结构化处理摘要、子 Agent 摘要与 Writer 报告；"
            "evidence_package_json=输出 Evidence Merger 证据包；"
            "analysis_json=输出 Analysis Agent 分析骨架；"
            "report_json=输出 Writer 交接包；writer_markdown=输出 Writer Agent 完整 Markdown 报告；"
            "supervisor_json=调试完整 Supervisor JSON。"
        ),
    )
    parser.add_argument("--include-raw-child-states", dest="parallel_raw_output", action="store_true", default=_env_flag("BRAIN_PARALLEL_RAW_OUTPUT", False), help="在 raw_output 中附带 RAG 与 5 个 IQS 子 Agent 的原始状态，便于调试；默认回答仍保持精简报告数据包。")
    parser.add_argument("--disable-parallel-raw-output", dest="parallel_raw_output", action="store_false", help=argparse.SUPPRESS)
    return parser


def parse_query_from_args(args: argparse.Namespace) -> str:
    if args.query_text:
        return " ".join(str(item) for item in args.query_text).strip()
    return " ".join(str(item) for item in args.query).strip()


def web_options_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    if getattr(args, "web_engine_type", ""):
        values["engineType"] = args.web_engine_type
    if getattr(args, "web_time_range", ""):
        values["timeRange"] = args.web_time_range
    if getattr(args, "web_contents", ""):
        values["contents"] = args.web_contents
    if getattr(args, "web_num_results", 0):
        values["numResults"] = args.web_num_results
    if getattr(args, "web_timeout_ms", 0):
        values["timeout"] = args.web_timeout_ms
    return values


def brain_namespace_to_rag_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    values = namespace_to_overrides(args)
    for key in [
        "route",
        "web_engine_type",
        "web_time_range",
        "web_contents",
        "web_num_results",
        "web_timeout_ms",
        "enable_web_analysis",
        "enable_llm_merge",
        "enable_followup_loop",
        "supervisor_max_loops",
        "supervisor_min_coverage_gain",
        "supervisor_max_followup_queries",
        "layout_max_refinement_rounds",
        "output_mode",
        "parallel_raw_output",
    ]:
        values.pop(key, None)
    return values


def main() -> int:
    configure_pipeline_logging()
    args = build_arg_parser().parse_args()
    query = parse_query_from_args(args)
    if not query:
        query = input("请输入问题：").strip()
    if not query:
        raise RuntimeError("查询不能为空。")

    state = run_brain_agent(
        query,
        session_id=str(getattr(args, "session_id", "") or "").strip(),
        route=str(getattr(args, "route", "auto") or "auto"),
        args_overrides=brain_namespace_to_rag_overrides(args),
        web_search_options=web_options_from_args(args),
        enable_web_analysis=bool(getattr(args, "enable_web_analysis", True)),
        enable_llm_merge=bool(getattr(args, "enable_llm_merge", True)),
        enable_followup_loop=bool(getattr(args, "enable_followup_loop", True)),
        supervisor_max_loops=int(getattr(args, "supervisor_max_loops", 3) or 3),
        supervisor_min_coverage_gain=float(getattr(args, "supervisor_min_coverage_gain", 0.10) or 0.10),
        supervisor_max_followup_queries=int(getattr(args, "supervisor_max_followup_queries", 4) or 4),
        layout_max_refinement_rounds=int(getattr(args, "layout_max_refinement_rounds", 8) or 8),
        output_mode=str(getattr(args, "output_mode", "agent_text") or "agent_text"),
        parallel_raw_output=bool(getattr(args, "parallel_raw_output", False)),
    )
    if getattr(args, "json", False):
        print(json.dumps(state, ensure_ascii=False, indent=2, default=json_safe_default))
    else:
        print(str(state.get("answer_text") or "").strip())
    return 1 if state.get("errors") and not state.get("answer_text") else 0


if __name__ == "__main__":
    raise SystemExit(main())
