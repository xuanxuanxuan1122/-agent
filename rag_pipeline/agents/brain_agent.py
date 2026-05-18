from __future__ import annotations

import argparse
import copy
import json
import logging
import operator
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError, as_completed
from datetime import datetime
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict

from langgraph.graph import END, START, StateGraph

from ..config.search_config import (
    DEFAULT_LLM_SYNTHESIS_API_KEY,
    DEFAULT_LLM_SYNTHESIS_DISABLE_THINKING,
    DEFAULT_LLM_SYNTHESIS_MODEL,
    DEFAULT_LLM_SYNTHESIS_PROVIDER,
    DEFAULT_LLM_SYNTHESIS_TIMEOUT,
    DEFAULT_LLM_SYNTHESIS_URL,
)
from ..logging_utils import configure_pipeline_logging
from ..runtime_cache import json_safe_default
from ..search.engine import build_arg_parser as build_rag_arg_parser
from ..search.memory import call_openai_compatible_json, llm_config_is_ready, normalize_llm_config
from .analysis_agent import run_analysis_agent
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
    "并调度 RAG 与联网检索 Worker 获取对应证据。IQS Worker 不再代表固定的市场、竞争、政策、技术、资本角色。"
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
    print(line, file=sys.stderr, flush=True)

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
        "source_priority": ["paper", "patent", "product_doc", "technical_standard", "technology", "product", "专利"],
        "intents": ["academic", "technology", "technical", "product"],
        "query_terms": ["technology", "product", "patent", "standard"],
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


class BrainAgentState(TypedDict, total=False):
    messages: List[Dict[str, Any]]
    query: str
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


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(str(raw).strip())
    except ValueError:
        return default


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
    else:
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
        return forced, f"用户显式指定 route={forced}"

    text = str(query or "").strip()
    web_intent = bool(_WEB_INTENT_RE.search(text))
    local_intent = bool(_LOCAL_INTENT_RE.search(text))
    industry_intent = bool(_INDUSTRY_RESEARCH_RE.search(text))
    growth_finance_intent = bool(_GROWTH_FINANCE_RE.search(text))
    market_data_intent = bool(_MARKET_DATA_RE.search(text))

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


def _route_agents(route: str) -> List[str]:
    route = str(route or "local").strip().lower()
    if route == "all":
        return ["rag", *IQS_ROLE_ORDER]
    if route == "both":
        return ["rag", *IQS_ROLE_ORDER]
    if route == "web":
        return list(IQS_ROLE_ORDER)
    return ["rag"]


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
        if proof_role not in {"metric", "support", "counter", "case", "filing", "source_check"}:
            evidence_type = str(copied.get("evidence_type") or copied.get("intent") or "").lower()
            proof_role = (
                "counter"
                if "risk" in evidence_type or "counter" in evidence_type
                else "case"
                if evidence_type in {"case", "customer_case", "procurement"}
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
        copied["min_sources"] = int(copied.get("min_sources") or (chapter.get("min_ab_sources") if proof_role in {"metric", "source_check"} else 1) or 1)
        goals.append(copied)
    roles_present = {str(goal.get("proof_role") or "").strip().lower() for goal in goals}
    supplemental_specs = [
        ("metric", "补齐本章关键指标、时间、范围和单位"),
        ("support", "补齐本章直接支撑证据"),
        ("counter", "补齐本章反证、风险和判断边界"),
        ("case", "补齐本章案例、订单、客户或采购证据"),
        ("source_check", "补齐本章来源核验和权威出处"),
    ]
    for role, description in supplemental_specs:
        if role in roles_present:
            continue
        goals.append(
            {
                "goal_id": f"{chapter_id}_{role}",
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
    if proof_role not in {"metric", "support", "counter", "case", "filing", "source_check"}:
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
    }[proof_role]
    topic_terms = _topic_seed_terms(query, chapter, goal)[:3]
    query_focus = _compact_iqs_terms(terms, max_terms=1, max_chars=10)
    base_query = _compose_iqs_query([topic_terms, query_focus, query_hint])
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
    if _env_flag("BRAIN_ENABLE_DEEP_SEARCH_VARIANTS", True):
        deep_hint = {
            "metric": "官方统计 原始表",
            "support": "协会 白皮书 研报",
            "counter": "失败案例 价格下行",
            "case": "客户认证 中标",
            "filing": "年报 公告 招股书",
            "source_check": "发布机构 披露日期",
        }[proof_role]
        deep_task = {
            **task,
            "task_id": f"{task['task_id']}_deep",
            "query": _compose_iqs_query([base_query, deep_hint]),
            "evidence_goal": f"{goal_text}；补充交叉验证、反证和原始口径",
            "deep_search_variant": True,
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
    return copied


def assign_tasks_to_iqs_lanes(tasks: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    assigned: Dict[str, List[Dict[str, Any]]] = {lane: [] for lane in IQS_ROLE_ORDER}
    for raw in tasks:
        if not isinstance(raw, dict):
            continue
        task = dict(raw)
        preferred = str(task.get("agent") or "").strip().lower()
        if preferred in IQS_ROLE_CONFIGS:
            assigned[preferred].append(task)
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
        "official_data": ["metric", "source_check", "support", "filing", "counter", "case"],
        "filing_company": ["filing", "source_check", "case", "support", "metric", "counter"],
        "market_research": ["support", "metric", "counter", "source_check", "case", "filing"],
        "news_event": ["counter", "case", "support", "source_check", "metric", "filing"],
        "technology_product": ["support", "metric", "source_check", "case", "counter", "filing"],
        "customer_case": ["case", "counter", "support", "filing", "metric", "source_check"],
    }
    ordered = priority.get(lane_type) or ["support", "metric", "source_check", "counter", "case", "filing"]
    return ordered.index(role) if role in ordered else len(ordered)


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
        group_key = str(
            task.get("chapter_id")
            or task.get("dimension_id")
            or task.get("hypothesis_id")
            or task.get("chapter_title")
            or f"task_{index}"
        ).strip()
        if group_key not in grouped:
            grouped[group_key] = []
            group_order.append(group_key)
        grouped[group_key].append((index, task))

    for group in grouped.values():
        group.sort(key=lambda item: (_proof_role_rank_for_lane(item[1]), item[0]))

    selected: List[tuple[int, Dict[str, Any]]] = []
    while len(selected) < limit:
        progressed = False
        for group_key in group_order:
            group = grouped.get(group_key) or []
            if not group:
                continue
            selected.append(group.pop(0))
            progressed = True
            if len(selected) >= limit:
                break
        if not progressed:
            break

    selected_indices = {index for index, _ in selected}
    selected.sort(key=lambda item: item[0])
    dropped = [(index, task) for index, task in indexed if index not in selected_indices]
    return [task for _, task in selected], [task for _, task in dropped]


def build_query_analysis(query: str, route: str) -> Dict[str, Any]:
    query = str(query or "").strip()
    agents = _route_agents(route)
    max_queries = _effective_queries_per_agent()
    max_tasks_per_lane = _effective_iqs_lane_task_limit()
    research_plan = run_research_planner_agent(query=query, llm_config=build_llm_config())
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
    assigned_tasks = assign_tasks_to_iqs_lanes(dynamic_tasks)
    if any(role_key in agents for role_key in IQS_ROLE_ORDER) and any(assigned_tasks.values()):
        for role_key in IQS_ROLE_ORDER:
            if role_key not in agents:
                continue
            lane_tasks = assigned_tasks.get(role_key, [])
            tasks, dropped = _select_lane_tasks_for_budget(lane_tasks, max_tasks_per_lane)
            agent_tasks[role_key] = tasks
            scheduled_tasks.extend([{**task, "scheduled_lane": role_key} for task in tasks])
            dropped_tasks.extend([{**task, "dropped_lane": role_key, "drop_reason": "max_tasks_per_lane"} for task in dropped])
            agent_queries[role_key] = _unique_strings([_dynamic_role_query(task) for task in tasks], max_items=max_queries)
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
        "route": route,
        "report_plan": report_plan,
        "report_blueprint": report_blueprint,
        "research_plan": research_plan,
        "research_type": research_plan.get("research_type"),
        "dynamic_dimensions": research_plan.get("dimensions"),
        "dynamic_search_tasks": dynamic_tasks,
        "search_task_schedule": {
            "max_tasks_per_lane": max_tasks_per_lane,
            "scheduled_tasks": scheduled_tasks,
            "dropped_tasks": dropped_tasks,
            "scheduled_count": len(scheduled_tasks),
            "dropped_count": len(dropped_tasks),
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


def build_llm_config() -> Dict[str, Any]:
    return {
        "provider": DEFAULT_LLM_SYNTHESIS_PROVIDER,
        "url": DEFAULT_LLM_SYNTHESIS_URL,
        "api_key": DEFAULT_LLM_SYNTHESIS_API_KEY,
        "model": DEFAULT_LLM_SYNTHESIS_MODEL,
        "timeout": DEFAULT_LLM_SYNTHESIS_TIMEOUT,
        "disable_thinking": DEFAULT_LLM_SYNTHESIS_DISABLE_THINKING,
    }


def prepare_query_node(state: BrainAgentState) -> BrainAgentState:
    query = extract_query_from_state(state)
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
        "metadata": {
            **dict(state.get("metadata") or {}),
            "agent_name": AGENT_NAME,
            "agent_description": AGENT_DESCRIPTION,
            "framework": "langgraph",
            "children": ["industry_rag_agent", *[IQS_ROLE_CONFIGS[key]["child"] for key in IQS_ROLE_ORDER]],
            "agent_stage": "decompose_query",
        },
    }


def route_node(state: BrainAgentState) -> BrainAgentState:
    if state.get("errors"):
        return {}
    started = time.perf_counter()
    _progress("brain", "路由与动态研究规划开始", query=state.get("query"))
    route, reason = route_query(str(state.get("query") or ""), str(state.get("route") or os.getenv("BRAIN_AGENT_ROUTE", "auto")))
    query_analysis = build_query_analysis(str(state.get("query") or ""), route)
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
    lane_workers = max(1, min(_env_int("BRAIN_IQS_LANE_PARALLEL_WORKERS", 4), len(indexed_tasks) or 1))
    task_timeout = max(0.0, _env_float("BRAIN_IQS_LANE_TASK_TIMEOUT_SECONDS", 180.0))
    task_payloads: List[Dict[str, Any]] = []
    if lane_workers <= 1 or len(indexed_tasks) <= 1:
        for index, task in indexed_tasks:
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
        try:
            for future in as_completed(future_map, timeout=task_timeout or None):
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
        except FutureTimeoutError:
            for future, index in future_map.items():
                if future in completed:
                    continue
                future.cancel()
                task_payloads.append(
                    {
                        "index": index,
                        "errors": [f"{config['label']} parallel task timed out after {task_timeout:.0f}s"],
                    }
                )
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
    coverage = {
        "scheduled": len(work_items),
        "succeeded": succeeded,
        "failed": max(0, len(work_items) - succeeded),
        "raw_data_points": len(raw_data_points),
        "search_results": len(search_results),
        "page_results": len(page_results),
        "key_sources": len(key_sources),
    }
    answer_text = "\n\n".join(answer_blocks).strip()
    if query_results or answer_text or search_results or page_results:
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
    if route == "all":
        return ["industry_rag_agent", *[IQS_ROLE_CONFIGS[key]["node"] for key in IQS_ROLE_ORDER]]
    if route == "both":
        return ["industry_rag_agent", *[IQS_ROLE_CONFIGS[key]["node"] for key in IQS_ROLE_ORDER]]
    if route == "local":
        return ["industry_rag_agent"]
    if route == "web":
        return [IQS_ROLE_CONFIGS[key]["node"] for key in IQS_ROLE_ORDER]
    return ["merge_outputs"]


def _child_answer(child_state: Optional[Dict[str, Any]]) -> str:
    if not child_state:
        return ""
    return str(child_state.get("answer_text") or "").strip()


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clip_confidence(value: Any, default: float = 0.0) -> float:
    return round(max(0.0, min(1.0, _safe_float(value, default))), 4)


def _compact_text(value: Any, max_chars: int = 1200) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    if len(text) <= max_chars:
        return text
    return text[: max(20, max_chars - 3)].rstrip() + "..."


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
    return {
        "payload_mode": "summary",
        "summary": _compact_mapping_for_state(summary, max_items=20, max_chars=180),
        "metadata": _compact_mapping_for_state(metadata, max_items=20, max_chars=160),
        "normalized_evidence": _summarize_sequence(normalized, sample=8, max_chars=160),
        "raw_data_points": _summarize_sequence(raw_data_points, sample=8, max_chars=160),
        "source_count": len(_as_list(evidence_package.get("sources")) or _as_list(evidence_package.get("source_registry"))),
    }


def _compact_structured_analysis_for_state(structured_analysis: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "payload_mode": "summary",
        "analysis_type": structured_analysis.get("analysis_type"),
        "report_plan": _compact_mapping_for_state(_as_dict(structured_analysis.get("report_plan")), max_items=30, max_chars=180),
        "research_plan": _compact_mapping_for_state(_as_dict(structured_analysis.get("research_plan")), max_items=30, max_chars=180),
        "structured_analysis": _compact_mapping_for_state(_as_dict(structured_analysis.get("structured_analysis")), max_items=30, max_chars=180),
        "report_insight_package": _compact_mapping_for_state(_as_dict(structured_analysis.get("report_insight_package")), max_items=30, max_chars=180),
    }


def _compact_writer_report_for_state(writer_report: Dict[str, Any]) -> Dict[str, Any]:
    keep_keys = [
        "report_markdown",
        "report_type",
        "report_status",
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
        "delivery_blockers",
        "required_followups",
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
        scheduled = route in {"local", "both", "all"}
        return {
            "answer": "",
            "confidence": 0.0,
            "key_sources": [],
            "limitations": {"failure_reason": "本地 RAG 子智能体未返回结果。"},
            "status": "failed",
            "used": False,
            "note": "当前路由应调度本地 RAG 但未获得结果。" if scheduled else "当前路由未调度本地 RAG。",
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


def _normalize_web_sources(raw_output: Dict[str, Any], payload: Dict[str, Any], max_items: int = 30) -> List[Dict[str, Any]]:
    payload_sources = [item for item in _as_list(payload.get("key_sources")) if isinstance(item, dict)]
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
            key = (source["url"], source["title"])
            if key not in seen:
                seen.add(key)
                normalized.append(source)
            if len(normalized) >= max_items:
                return normalized

    sources: List[Dict[str, Any]] = []
    combined = list(raw_output.get("search_results") or []) + list(raw_output.get("page_results") or [])
    for index, item in enumerate(combined):
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
    if ranked and ranked[0][1] >= 5:
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
    max_items: int = 24,
) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    seen = set()
    for raw_line in re.split(r"[\n\r]+", str(evidence_text or "")):
        line = re.sub(r"^\s*[-*•\d.、\)）]+\s*", "", raw_line).strip()
        if not line or line.startswith("本次使用") or line.startswith("主要来源"):
            continue
        citation_ids = _extract_citation_ids(line)
        clean = re.sub(r"\s*\[[^\[\]]+\]", "", line).strip()
        clean = re.sub(r"\s+", " ", clean)
        if len(clean) < 18:
            continue
        if not re.search(r"\d", clean):
            continue
        tag_match = re.search(r"【([^】]+)】", clean)
        tag = tag_match.group(1).strip() if tag_match else ""
        source = _source_for_evidence_line(sources, citation_ids, clean)
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
            metric = _infer_metric_from_context(clean, value) if value else (tag or "事实")
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
                }
            )
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

    limitations = _as_dict(payload.get("limitations"))
    if not limitations:
        quality = _as_dict(raw_output.get("quality_processing"))
        limitations = {
            "data_recency": str(_as_dict(raw_output.get("search_options")).get("timeRange") or "未限定时间范围").strip(),
            "coverage": f"IQS 返回 {source_count} 条候选；精排后保留 {quality.get('final_count', source_count)} 条。" if source_count else "未返回可用来源。",
            "conflicts": None,
        }
    if answer_payload.get("evidence_gap"):
        limitations = {**limitations, "evidence_gap": answer_payload.get("evidence_gap")}
    limitations = {**limitations, "errors": child_errors, "partial_errors": partial_errors}
    normalized_sources = _normalize_web_sources(raw_output, payload)
    if status == "failed" or failure_answer:
        raw_data_points = []
    else:
        raw_data_points = _structured_evidence_to_raw_points(
            answer_payload.get("evidence") or answer_text,
            sources=normalized_sources,
            dimension=str(raw_output.get("dimension") or "").strip(),
            confidence=confidence,
        )
    search_options = _as_dict(raw_output.get("search_options"))
    search_task = _as_dict(search_options.get("search_task"))
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
        if "未调度" in note_text or "已停用" in note_text:
            continue
        pool.append(
            {
                "round": 1,
                "agent": agent,
                "child_agent": child_name,
                "query": original_query,
                "targets_gap": "初始问题",
                "status": str(child.get("status") or "failed"),
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
        if not isinstance(item, dict) or str(item.get("status") or "") == "failed":
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
    llm_config = build_llm_config()
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
    return _normalize_coverage_evaluation(
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
        "limitations": _as_dict(child.get("limitations")),
        "note": str(child.get("note") or "").strip(),
        "raw_data_points": list(child.get("raw_data_points") or []),
        "data_gap": list(child.get("data_gap") or []),
    }
    if task:
        item.update(
            {
                "task_id": task.get("task_id"),
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
                "search_task": copy.deepcopy(task),
            }
        )
        for point in item["raw_data_points"]:
            if isinstance(point, dict):
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
    return item


def _run_single_followup(
    *,
    agent: str,
    query: str,
    targets_gap: str,
    round_number: int,
    state: BrainAgentState,
    search_task: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
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
    valid_agents = {"rag", "iqs", "both", "all", *IQS_ROLE_ORDER}
    for item in follow_up_queries:
        query = str(item.get("query") or "").strip()
        agent = str(item.get("agent") or "").strip().lower()
        targets_gap = str(item.get("targets_gap") or "").strip()
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
            agents = ["rag", *(lane_agents or IQS_ROLE_ORDER)]
        else:
            agents = [agent]
        for routed_agent in agents:
            search_task = normalize_search_task(
                {
                    **dict(item),
                    "agent": routed_agent,
                    "query": query,
                    "dimension_name": item.get("dimension_name") or item.get("dimension") or targets_gap,
                    "evidence_goal": item.get("evidence_goal") or targets_gap,
                },
                fallback_index=len(tasks) + 1,
            )
            tasks.append({"query": query, "agent": routed_agent, "targets_gap": targets_gap, "search_task": search_task})
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
    if not tasks:
        return []

    started = time.perf_counter()
    max_workers = max(1, min(_env_int("BRAIN_FOLLOWUP_PARALLEL_WORKERS", 4), len(tasks)))
    _progress("followup", "补证任务开始", tasks=len(tasks), workers=max_workers, round=round_number)
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
                        "child_agent": (
                            "industry_rag_agent"
                            if task["agent"] == "rag"
                            else IQS_ROLE_CONFIGS.get(task["agent"], {}).get("child", "web_analysis_agent")
                        ),
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
                    "child_agent": (
                        "industry_rag_agent"
                        if task["agent"] == "rag"
                        else IQS_ROLE_CONFIGS.get(task["agent"], {}).get("child", "web_analysis_agent")
                    ),
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
    success_count = len([item for item in results if str(item.get("status") or "") in {"success", "partial"}])
    _progress("followup", "补证任务结束", success=success_count, total=len(results), elapsed=f"{time.perf_counter() - started:.1f}s")
    return results


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
                "dimension_name": item.get("dimension_name") or item.get("dimension") or targets_gap,
                "evidence_goal": item.get("evidence_goal") or item.get("reason") or targets_gap,
                "source_priority": _as_list(item.get("source_priority")),
                "lane_targets": _as_list(item.get("lane_targets")),
                "blocking_gaps": _as_list(item.get("blocking_gaps")),
                "hypothesis_id": item.get("hypothesis_id"),
                "hypothesis_statement": item.get("hypothesis_statement"),
                "proof_profile_id": item.get("proof_profile_id"),
                "mandatory_proof_id": item.get("mandatory_proof_id") or item.get("proof_id"),
                "missing_mandatory_proofs": _as_list(item.get("missing_mandatory_proofs")),
                "proof_role": item.get("proof_role"),
                "proof_standard": item.get("proof_standard"),
                "evidence_type": item.get("evidence_type"),
                "required_evidence_mix": _as_list(item.get("required_evidence_mix")),
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


def _lane_coverage_from_state(state: BrainAgentState) -> Dict[str, Any]:
    coverage: Dict[str, Any] = {}
    for role_key in IQS_ROLE_ORDER:
        config = IQS_ROLE_CONFIGS[role_key]
        role_state = _as_dict(state.get(config["state"]))  # type: ignore[literal-required]
        raw_output = _as_dict(role_state.get("raw_output"))
        lane = _as_dict(raw_output.get("lane_coverage")) or _as_dict(_as_dict(role_state.get("metadata")).get("lane_coverage"))
        if lane:
            coverage[role_key] = {**lane, "status": lane.get("status") or "completed"}
        else:
            coverage[role_key] = {
                "status": "missing",
                "scheduled": 0,
                "succeeded": 0,
                "failed": 0,
                "raw_data_points": 0,
                "search_results": 0,
                "page_results": 0,
                "key_sources": 0,
            }
    return coverage


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
    return {
        "status": str(writer_report.get("report_status") or ""),
        "passed": bool(validation.get("passed")),
        "quality_score": int(_safe_float(validation.get("quality_score"), 0.0)),
        "reformatter_preflight_status": str(preflight_plan.get("status") or ""),
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
        _reformatter_preflight_rank(str(snapshot.get("reformatter_preflight_status") or "")),
        snapshot["quality_score"],
        -snapshot["error_count"],
        -snapshot["layout_gap_count"],
        min(snapshot["estimated_chars"], 60000),
        -snapshot["warning_count"],
    )


def _followup_result_has_signal(results: Sequence[Dict[str, Any]]) -> bool:
    for item in results:
        if not isinstance(item, dict):
            continue
        if str(item.get("status") or "") in {"success", "partial"}:
            if str(item.get("answer") or "").strip() or item.get("raw_data_points") or item.get("key_sources"):
                return True
    return False


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
    if any(marker in markers or marker in text for marker in {"missing_proof_standard", "missing_proof_standards"}):
        return 0
    if any(marker in markers or marker in text for marker in {"mandatory_proof_missing"}):
        return 0
    if any(marker in markers or marker in text for marker in {"insufficient_ab_sources", "insufficient_ab_core_sources"}):
        return 5
    if "counter" in text:
        return 8
    if "metric" in text:
        return 10
    if "case" in text:
        return 12
    if any(marker in markers or marker in text for marker in {"needs_corroboration", "corroboration", "source", "proof", "evidence"}):
        return 15
    return 30


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
    target = re.sub(r"\s+", " ", str(item.get("targets_gap") or item.get("dimension_name") or "").strip()).lower()
    return f"{agent}|{target}|{query}"


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


def _layout_refinement_round_count(trace: Sequence[Dict[str, Any]]) -> int:
    return len([item for item in trace if isinstance(item, dict) and isinstance(item.get("round"), int) and int(item.get("round") or 0) > 0])


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
    current_writer_report = _as_dict(current_writer_state.get("writer_report"))
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
    supervisor_followups = int(state.get("supervisor_max_followup_queries") or _env_int("BRAIN_SUPERVISOR_MAX_FOLLOWUP_QUERIES", 4))
    layout_followups = _env_int("BRAIN_LAYOUT_MAX_FOLLOWUP_QUERIES", 6)
    max_followups = max(
        1,
        min(
            20,
            max(supervisor_followups, layout_followups),
        ),
    )
    seen_followup_keys = set()
    followups = _layout_followup_queries_from_writer_report(current_writer_report, max_queries=max_followups * 4)
    new_followups: List[Dict[str, Any]] = []
    for item in followups:
        key = (item.get("query"), item.get("agent"), item.get("targets_gap"))
        if key in seen_followup_keys:
            continue
        seen_followup_keys.add(key)
        new_followups.append(item)
        if len(new_followups) >= max_followups:
            break
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
            "enabled": enable_layout_followup,
            "max_rounds": max_rounds,
            "max_followups_per_round": max_followups,
        }
    ]
    if not enable_layout_followup or not new_followups or max_rounds <= 0:
        trace[0]["stop_reason"] = "layout_followup_disabled_or_no_queries"
        _progress("writer", "Layout 补证跳过", reason=trace[0]["stop_reason"], elapsed=f"{time.perf_counter() - started:.1f}s")
        return {
            **best,
            "layout_refinement_trace": trace,
        }

    pending_followups = new_followups
    stop_reason = "max_rounds_reached"
    for round_number in range(1, max_rounds + 1):
        round_started = time.perf_counter()
        _progress("writer", "Layout 补证轮次开始", round=round_number, followups=len(pending_followups))
        before_pool_size = len(current_evidence_pool)
        followup_results = run_followup_queries(follow_up_queries=pending_followups, round_number=round_number, state=state)
        current_evidence_pool.extend([item for item in followup_results if isinstance(item, dict)])
        has_signal = _followup_result_has_signal(followup_results)
        current_evidence_package = merge_evidence_package(
            original_query=query,
            evidence_pool=current_evidence_pool,
            children=children,
            research_plan=_research_plan_from_state(state),
        )
        if report_plan:
            current_evidence_package["report_plan"] = report_plan
            current_evidence_package.setdefault("metadata", {})
            current_evidence_package["metadata"]["report_plan"] = report_plan
        current_analysis_state = run_analysis_agent(current_evidence_package, query=query)
        current_structured_analysis = _as_dict(current_analysis_state.get("structured_analysis"))
        _attach_report_plan(current_evidence_package, current_structured_analysis, report_plan)
        _attach_research_plan(current_evidence_package, current_structured_analysis, _research_plan_from_state(state))
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
        current_writer_report = _as_dict(current_writer_state.get("writer_report"))
        current_writer_report = _attach_reformatter_preflight_feedback(
            query=query,
            writer_report=current_writer_report,
            evidence_package=current_evidence_package,
            chapter_evidence_packages=_as_list(current_evidence_package.get("chapter_evidence_packages"))
            or _as_list(current_writer_report.get("chapter_evidence_packages")),
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
        pending_followups = []
        for item in candidate_followups:
            key = (item.get("query"), item.get("agent"), item.get("targets_gap"))
            if key in seen_followup_keys:
                continue
            seen_followup_keys.add(key)
            pending_followups.append(item)
            if len(pending_followups) >= max_followups:
                break
        trace.append(
            {
                "round": round_number,
                "source": "layout_followup",
                "quality": _writer_quality_snapshot(current_writer_report),
                "is_best": improved_best,
                "evidence_pool_size_before": before_pool_size,
                "evidence_pool_size_after": len(current_evidence_pool),
                "has_signal": has_signal,
                "follow_up_queries": pending_followups,
                "followup_results": [
                    {
                        "agent": item.get("agent"),
                        "child_agent": item.get("child_agent"),
                        "query": item.get("query"),
                        "targets_gap": item.get("targets_gap"),
                        "status": item.get("status"),
                        "confidence": item.get("confidence"),
                    }
                    for item in followup_results
                ],
                "layout_gaps": _as_list(_as_dict(current_writer_report.get("layout_plan")).get("layout_gaps"))[:12],
            }
        )
        if not pending_followups:
            stop_reason = "no_new_layout_followup_queries"
            break
        if not has_signal:
            stop_reason = "no_new_evidence_signal"
            break

    trace.append(
        {
            "round": "final",
            "source": "layout_refinement_summary",
            "stop_reason": stop_reason,
            "best_quality": _writer_quality_snapshot(_as_dict(best.get("writer_report"))),
            "total_rounds": len([item for item in trace if isinstance(item.get("round"), int) and item.get("round", 0) > 0]),
            "unique_followup_queries": len(seen_followup_keys),
        }
    )
    _progress("writer", "Writer 流水线结束", stop=stop_reason, elapsed=f"{time.perf_counter() - started:.1f}s")
    return {
        **best,
        "layout_refinement_trace": trace,
        "initial_writer_report": initial_writer_report,
    }


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
    seen_followup_keys: set[str] = set()
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
        followups, skipped_duplicates = _dedupe_followups(followups, seen_followup_keys)
        trace_item["follow_up_queries"] = followups
        trace_item["skipped_duplicate_followups"] = skipped_duplicates
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
        followup_results = run_followup_queries(follow_up_queries=followups, round_number=loop_number + 1, state=state)
        for item in followups:
            previous_queries.append(str(item.get("query") or ""))
        evidence_pool.extend(followup_results)
        has_signal = _followup_result_has_signal(followup_results)
        trace_item["followup_results"] = [
            {
                "agent": item.get("agent"),
                "query": item.get("query"),
                "targets_gap": item.get("targets_gap"),
                "status": item.get("status"),
                "confidence": item.get("confidence"),
            }
            for item in followup_results
        ]
        trace_item["followup_has_signal"] = has_signal
        loop_trace.append(trace_item)
        if not has_signal:
            coverage_score = _safe_float(evaluation.get("coverage_score"), 0.0)
            sufficient = coverage_score >= coverage_target
            evaluation = {**evaluation, "is_sufficient": sufficient, "stop_reason": "no_new_followup_signal" if sufficient else "no_new_followup_signal_with_gaps"}
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
    has_conflict = any(
        [
            _has_conflict_marker(rag_child.get("answer")),
            _has_conflict_marker(web_child.get("answer")),
            _has_conflict_marker(_as_dict(rag_child.get("limitations")).get("conflicts")),
            _has_conflict_marker(_as_dict(web_child.get("limitations")).get("conflicts")),
        ]
    )
    conflicts: List[Dict[str, str]] = []
    if has_conflict:
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

    confidence = _calculate_supervisor_confidence(rag_child, web_child, has_conflict=has_conflict)
    rag_status = str(rag_child.get("status") or "failed")
    web_status = str(web_child.get("status") or "failed")
    if confidence == 0.0:
        conclusion = "证据不足，无法输出有效判断"
        consensus = None
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
    elif has_conflict or any(str(child.get("status")) != "success" for child in [rag_child, web_child]):
        next_action = "needs_more_search"
    else:
        next_action = "complete"

    return {
        "answer": {
            "conclusion": conclusion,
            "rag_insights": _tagged_insight("[RAG]", rag_child, "本地知识库未返回可用结论。"),
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

## 输出格式（严格 JSON）
{
  "coverage_score": 0.0,
  "chapter_scores": {
    "章节标题": {"score": 0, "reason": "..."}
  },
  "is_sufficient": false,
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
      "query": "补充搜索问题",
      "agent": "rag | iqs | both",
      "targets_gap": "对应动态证据目标"
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
    llm_config = build_llm_config()
    if not llm_config_is_ready(llm_config):
        raise RuntimeError("大脑 Agent 的大模型汇总配置不完整。")

    user_payload = {
        "query": query,
        "route": route,
        "route_reason": route_reason,
        "child_outputs": children,
        "child_errors": list(errors),
        "supervisor_review": {
            "coverage_evaluation": coverage_evaluation or {},
            "loop_trace": list(loop_trace or []),
            "evidence_pool_summary": evidence_pool_summary,
        },
        "fallback_for_schema_reference": fallback_decision,
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
        loop_errors = list(loop_result.get("evaluation_errors") or [])
        new_errors.extend(loop_errors)
        all_errors = errors + new_errors
        evidence_pool = [item for item in _as_list(loop_result.get("evidence_pool")) if isinstance(item, dict)]
        evidence_package = merge_evidence_package(
            original_query=str(state.get("query") or ""),
            evidence_pool=[item for item in list(evidence_pool or []) if isinstance(item, dict)],
            children=children,
            research_plan=research_plan,
        )
        if report_plan:
            evidence_package["report_plan"] = report_plan
            evidence_package.setdefault("metadata", {})
            evidence_package["metadata"]["report_plan"] = report_plan
        analysis_state = run_analysis_agent(evidence_package, query=str(state.get("query") or ""))
        structured_analysis = _as_dict(analysis_state.get("structured_analysis"))
        if report_plan:
            structured_analysis["report_plan"] = report_plan
        _attach_research_plan(evidence_package, structured_analysis, research_plan)
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
        writer_report = _as_dict(writer_bundle.get("writer_report"))
        evidence_pool = [item for item in _as_list(writer_bundle.get("evidence_pool")) if isinstance(item, dict)]
        evidence_package = _as_dict(writer_bundle.get("evidence_package")) or evidence_package
        structured_analysis = _as_dict(writer_bundle.get("structured_analysis")) or structured_analysis
        analysis_state = _as_dict(writer_bundle.get("analysis_state")) or analysis_state
        layout_refinement_trace = _as_list(writer_bundle.get("layout_refinement_trace"))
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
        raw_output: Dict[str, Any] = {
            "query": state.get("query", ""),
            "route": route,
            "route_reason": route_reason,
            "query_analysis": _as_dict(state.get("query_analysis")),
            "search_task_schedule": _as_dict(state.get("search_task_schedule")) or _as_dict(query_analysis.get("search_task_schedule")),
            "lane_coverage": lane_coverage,
            "graph_trace": state.get("agent_trace", []),
            "child_outputs": _state_payload(children, "children"),
            "evidence_pool": _state_payload(evidence_pool, "evidence_pool"),
            "evidence_package": _state_payload(evidence_package, "evidence_package"),
            "structured_analysis": _state_payload(structured_analysis, "structured_analysis"),
            "self_refine_trace": [],
            "coverage_evaluation": loop_result.get("coverage_evaluation", {}),
            "loop_trace": loop_result.get("loop_trace", []),
            "evidence_pool_summary": loop_result.get("evidence_pool_summary", ""),
            "layout_refinement_trace": layout_refinement_trace,
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
            "structured_analysis": _state_payload(structured_analysis, "structured_analysis"),
            "writer_report": _state_payload(writer_report, "writer_report"),
            **_writer_pipeline_state_fields(writer_report),
            "lane_coverage": lane_coverage,
            "errors": new_errors,
            "metadata": {
                **dict(state.get("metadata") or {}),
                "agent_stage": "merge_outputs",
                "merge_source": "agent_text_structured_processing",
                "analysis_source": analysis_meta.get("source"),
                "writer_source": writer_meta.get("source"),
                "layout_refinement_rounds": _layout_refinement_round_count(layout_refinement_trace),
                "coverage_score": _as_dict(evidence_package.get("summary")).get("overall_coverage"),
                "next_action": "display_structured_agent_outputs",
            },
        }

    loop_result = run_supervisor_evidence_loop(state=state, initial_children=children, route=route)
    children = loop_result["children"]
    loop_errors = list(loop_result.get("evaluation_errors") or [])
    new_errors.extend(loop_errors)
    all_errors = errors + new_errors
    evidence_package = merge_evidence_package(
        original_query=str(state.get("query") or ""),
        evidence_pool=[item for item in list(loop_result.get("evidence_pool") or []) if isinstance(item, dict)],
        children=children,
        research_plan=research_plan,
    )
    if report_plan:
        evidence_package["report_plan"] = report_plan
        evidence_package.setdefault("metadata", {})
        evidence_package["metadata"]["report_plan"] = report_plan
    analysis_state = run_analysis_agent(evidence_package, query=str(state.get("query") or ""))
    structured_analysis = _as_dict(analysis_state.get("structured_analysis"))
    if report_plan:
        structured_analysis["report_plan"] = report_plan
    _attach_research_plan(evidence_package, structured_analysis, research_plan)
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

    evidence_pool = [item for item in _as_list(loop_result.get("evidence_pool")) if isinstance(item, dict)]
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
    writer_report = _as_dict(writer_bundle.get("writer_report"))
    evidence_pool = [item for item in _as_list(writer_bundle.get("evidence_pool")) if isinstance(item, dict)]
    evidence_package = _as_dict(writer_bundle.get("evidence_package")) or evidence_package
    structured_analysis = _as_dict(writer_bundle.get("structured_analysis")) or structured_analysis
    analysis_state = _as_dict(writer_bundle.get("analysis_state")) or analysis_state
    layout_refinement_trace = _as_list(writer_bundle.get("layout_refinement_trace"))
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
        "graph_trace": state.get("agent_trace", []),
        "child_outputs": _state_payload(children, "children"),
        "evidence_pool": _state_payload(evidence_pool, "evidence_pool"),
        "evidence_package": _state_payload(evidence_package, "evidence_package"),
        "structured_analysis": _state_payload(structured_analysis, "structured_analysis"),
        "evidence_pool_summary": loop_result.get("evidence_pool_summary", ""),
        "coverage_evaluation": loop_result.get("coverage_evaluation", {}),
        "loop_trace": loop_result.get("loop_trace", []),
        "layout_refinement_trace": layout_refinement_trace,
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
        "structured_analysis": _state_payload(structured_analysis, "structured_analysis"),
        "writer_report": _state_payload(writer_report, "writer_report"),
        **_writer_pipeline_state_fields(writer_report),
        "lane_coverage": lane_coverage,
        "errors": new_errors,
        "metadata": {
            **dict(state.get("metadata") or {}),
            "agent_stage": "merge_outputs",
            "merge_source": merge_meta.get("source", "fallback"),
            "analysis_source": _as_dict(_as_dict(analysis_state.get("raw_output")).get("analysis")).get("source"),
            "writer_source": writer_meta.get("source"),
            "layout_refinement_rounds": _layout_refinement_round_count(layout_refinement_trace),
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
) -> BrainAgentState:
    configure_pipeline_logging()
    started = time.perf_counter()
    graph = create_brain_agent_graph()
    initial_messages = list(messages or [])
    if query and not initial_messages:
        initial_messages.append({"role": "user", "content": query})
    state: BrainAgentState = {
        "query": query,
        "messages": initial_messages,
        "session_id": session_id,
        "route": route,
        "args_overrides": dict(args_overrides or {}),
        "web_search_options": dict(web_search_options or {}),
    }
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
    parser.add_argument("--route", choices=["auto", "local", "web", "both", "all"], default=os.getenv("BRAIN_AGENT_ROUTE", "auto"), help="大脑 Agent 路由策略。all=RAG/IQS 并行。")
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

