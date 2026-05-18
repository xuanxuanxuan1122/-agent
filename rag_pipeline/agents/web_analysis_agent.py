from __future__ import annotations

import argparse
import copy
import difflib
import json
import logging
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TypedDict
from urllib.parse import urlparse

from langgraph.graph import END, START, StateGraph

from ..config.search_config import (
    DEFAULT_ENABLE_API_RERANK,
    DEFAULT_LLM_SYNTHESIS_API_KEY,
    DEFAULT_LLM_SYNTHESIS_DISABLE_THINKING,
    DEFAULT_LLM_SYNTHESIS_MODEL,
    DEFAULT_LLM_SYNTHESIS_PROVIDER,
    DEFAULT_LLM_SYNTHESIS_TIMEOUT,
    DEFAULT_LLM_SYNTHESIS_URL,
    DEFAULT_RERANK_API_KEY,
    DEFAULT_RERANK_MAX_CHARS_PER_DOC,
    DEFAULT_RERANK_MAX_DOCS,
    DEFAULT_RERANK_MODEL,
    DEFAULT_RERANK_PROVIDER,
    DEFAULT_RERANK_TIMEOUT,
    DEFAULT_RERANK_TOP_N,
    DEFAULT_RERANK_URL,
)
from ..search.engine import call_external_rerank_api
from ..search.memory import call_openai_compatible_json, llm_config_is_ready, normalize_llm_config
from ..runtime_cache import TTLCache, make_cache_key


AGENT_NAME = "web_analysis_agent"
AGENT_DESCRIPTION = (
    "基于阿里云 IQS Skills 的联网行研分析子智能体。适用于需要当前网页数据、新闻、政策更新、"
    "市场价格、财务数据、融资估值、发展趋势、公开网页或本地 Qdrant 知识库之外事实核验的问题。"
)
logger = logging.getLogger(__name__)


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
    suffix = " ".join(
        f"{key}={_short(value, max_chars=80)}"
        for key, value in fields.items()
        if value not in (None, "")
    )
    line = f"[{datetime.now().strftime('%H:%M:%S')}] [{stage}] {message}"
    if suffix:
        line = f"{line} {suffix}"
    print(line, file=sys.stderr, flush=True)

_URL_RE = re.compile(r"https?://[^\s)）>]+", re.I)
_WEB_ANALYSIS_CACHE = TTLCache(
    ttl_seconds=int(os.getenv("IQS_AGENT_CACHE_TTL_SECONDS", "300") or "0"),
    max_items=int(os.getenv("IQS_AGENT_CACHE_MAX_ITEMS", "96") or "0"),
)
_IQS_SEARCH_CACHE = TTLCache(
    ttl_seconds=int(os.getenv("IQS_SEARCH_CACHE_TTL_SECONDS", os.getenv("IQS_AGENT_CACHE_TTL_SECONDS", "300")) or "0"),
    max_items=int(os.getenv("IQS_SEARCH_CACHE_MAX_ITEMS", "256") or "0"),
)
_CURRENT_INFO_RE = re.compile(
    r"(现在|当前|今日|今天|昨日|昨天|最新|近期|新闻|快讯|实时|行情|股价|价格|政策|监管|财报|业绩|招股|公告|融资|并购|利率|汇率|指数)"
)
_IQS_QUOTA_ERROR_RE = re.compile(r"(Retrieval\.)?TestUserQueryExceeded|query exceed the limit|user query exceeded", re.I)
_YEAR_RE = re.compile(r"(今年|年度|年内|202[0-9]|203[0-9])")
_DATA_INTENT_RE = re.compile(r"(规模|增速|市占率|份额|销量|产量|收入|营收|利润|毛利|数据|统计|预测|CAGR|渗透率)", re.I)
_NEWS_INTENT_RE = re.compile(r"(最新|近期|新闻|快讯|今日|今天|昨日|昨天|动态|事件|融资|并购|投融资)", re.I)
_POLICY_INTENT_RE = re.compile(r"(政策|监管|补贴|标准|条例|办法|意见|通知|规划|发改委|工信部|证监会|央行|财政部)", re.I)
_FINANCE_INTENT_RE = re.compile(r"(财报|业绩|营收|利润|现金流|估值|IPO|上市|融资|并购|股价|股票|基金|债券|利率|汇率)", re.I)
_ANALYSIS_INTENT_RE = re.compile(r"(分析|机会|趋势|格局|竞争|产业链|价值链|商业模式|壁垒|风险|战略|增长|出海|转型)", re.I)

_HIGH_CREDIBILITY_DOMAINS = [
    "gov.cn",
    "stats.gov.cn",
    "ndrc.gov.cn",
    "miit.gov.cn",
    "mof.gov.cn",
    "pbc.gov.cn",
    "csrc.gov.cn",
    "sse.com.cn",
    "szse.cn",
    "hkexnews.hk",
    "cninfo.com.cn",
    "reuters.com",
    "bloomberg.com",
    "caixin.com",
    "yicai.com",
    "wind.com.cn",
]
_LOW_CREDIBILITY_HINTS = [
    "zhihu.com",
    "csdn.net",
    "baijiahao.baidu.com",
    "tieba.baidu.com",
    "weibo.com",
    "toutiao.com",
    "book118.com",
    "renrendoc.com",
    "docin.com",
    "doc88.com",
    "zhidao.baidu.com",
    "baike.baidu.com",
    "51baogao.cn",
    "chinabaogao.com",
    "chinairn.com",
    "leetcode.cn",
    "wk.baidu.com",
    "wenku.baidu.com",
    "xueqiu.com",
    "mguba.eastmoney.com",
    "自媒体",
    "公众号",
    "百家号",
    "贴吧",
]

_INTENT_ROUTE_ENV = {
    "数据型": "IQS_ENGINE_ROUTE_DATA",
    "新闻型": "IQS_ENGINE_ROUTE_NEWS",
    "政策型": "IQS_ENGINE_ROUTE_POLICY",
    "分析型": "IQS_ENGINE_ROUTE_ANALYSIS",
    "财报型": "IQS_ENGINE_ROUTE_FINANCE",
}
_INTENT_ROUTE_DEFAULTS = {
    "数据型": "LiteAdvanced,Generic",
    "新闻型": "Generic,LiteAdvanced",
    "政策型": "LiteAdvanced,Generic",
    "分析型": "LiteAdvanced,Generic",
    "财报型": "LiteAdvanced,Generic",
}


class WebAnalysisAgentState(TypedDict, total=False):
    messages: List[Dict[str, Any]]
    query: str
    urls: List[str]
    search_options: Dict[str, Any]
    enable_llm_analysis: bool
    search_results: List[Dict[str, Any]]
    page_results: List[Dict[str, Any]]
    answer_text: str
    raw_output: Dict[str, Any]
    metadata: Dict[str, Any]
    errors: List[str]


def _message_content(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("content") or "").strip()
    return str(getattr(message, "content", "") or "").strip()


def _message_role(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role") or message.get("type") or "").strip().lower()
    return str(getattr(message, "type", "") or getattr(message, "role", "") or "").strip().lower()


def extract_query_from_state(state: WebAnalysisAgentState) -> str:
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


def extract_urls(text: str) -> List[str]:
    seen = set()
    values: List[str] = []
    for match in _URL_RE.findall(str(text or "")):
        cleaned = match.rstrip("，。；;,.")
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            values.append(cleaned)
    return values


def discover_iqs_skill_dir() -> Path:
    candidates: List[Path] = []
    env_dir = os.getenv("ALIYUN_IQS_SKILL_DIR")
    if env_dir:
        candidates.append(Path(env_dir))

    here = Path(__file__).resolve()
    for parent in [*here.parents, Path.cwd(), Path.cwd().parent]:
        candidates.append(parent / ".agents" / "skills" / "alibabacloud-iqs-search")

    seen = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if str(resolved).lower() in seen:
            continue
        seen.add(str(resolved).lower())
        if (resolved / "scripts" / "search.mjs").is_file() and (resolved / "scripts" / "readpage.mjs").is_file():
            return resolved
    raise RuntimeError(
        "未找到 alibabacloud-iqs-search Skill。请将 ALIYUN_IQS_SKILL_DIR 设置为该 Skill 目录。"
    )


def iqs_api_key_is_configured() -> bool:
    return bool(os.getenv("ALIYUN_IQS_API_KEY"))


def infer_search_options(query: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    overrides = dict(overrides or {})
    is_current = bool(_CURRENT_INFO_RE.search(query))
    mentions_year = bool(_YEAR_RE.search(query))

    engine_type = str(overrides.get("engineType") or overrides.get("engine_type") or "").strip()
    if not engine_type:
        engine_type = os.getenv("IQS_SEARCH_ENGINE_TYPE", "").strip()
    engine_mode = "forced" if engine_type and engine_type.lower() != "auto" else "auto"
    if not engine_type or engine_type.lower() == "auto":
        engine_type = "Generic" if is_current else "LiteAdvanced"

    time_range = str(overrides.get("timeRange") or overrides.get("time_range") or "").strip()
    if not time_range:
        time_range = os.getenv("IQS_SEARCH_TIME_RANGE", "").strip()
    if not time_range:
        if re.search(r"(今日|今天|昨日|昨天|盘中|实时|快讯)", query):
            time_range = "OneWeek"
        elif is_current:
            time_range = "OneMonth"
        elif mentions_year:
            time_range = "OneYear"
        else:
            time_range = "NoLimit"
    if time_range not in {"NoLimit", "OneDay", "OneWeek", "OneMonth", "OneYear"}:
        time_range = "NoLimit"

    contents = str(overrides.get("contents") or os.getenv("IQS_SEARCH_CONTENTS", "mainText")).strip() or "mainText"
    if contents not in {"summary", "mainText"}:
        contents = "summary"

    try:
        num_results = int(overrides.get("numResults") or overrides.get("num_results") or os.getenv("IQS_SEARCH_NUM_RESULTS", "100"))
    except ValueError:
        num_results = 100
    num_results = min(100, max(1, num_results))

    try:
        timeout_ms = int(overrides.get("timeout") or overrides.get("timeout_ms") or os.getenv("IQS_SEARCH_TIMEOUT_MS", "20000"))
    except ValueError:
        timeout_ms = 20000
    timeout_ms = min(180000, max(1000, timeout_ms))

    category = str(overrides.get("category") or os.getenv("IQS_SEARCH_CATEGORY", "")).strip()

    values = {
        "engineType": engine_type,
        "engineMode": engine_mode,
        "timeRange": time_range,
        "contents": contents,
        "numResults": num_results,
        "timeout": timeout_ms,
    }
    if category:
        values["category"] = category
    return values


def _parse_script_error(stderr: str, stdout: str) -> str:
    for raw in (stderr, stdout):
        text = str(raw or "").strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return text[:800]
        if isinstance(payload, dict) and payload.get("error"):
            return str(payload.get("error"))
    return "IQS Skill 命令执行失败。"


def run_iqs_script(script_path: Path, args: Sequence[str], *, timeout_ms: int) -> Any:
    command = ["node", str(script_path), *list(args)]
    env = os.environ.copy()
    env["NO_COLOR"] = "1"
    try:
        completed = subprocess.run(
            command,
            cwd=str(script_path.parent.parent),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(5.0, timeout_ms / 1000 + 10),
        )
    except FileNotFoundError as exc:
        raise RuntimeError("阿里云 IQS Skills 需要 Node.js，但当前没有找到 `node` 命令。") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"IQS Skill 命令在 {timeout_ms}ms 后超时。") from exc

    if completed.returncode != 0:
        raise RuntimeError(_friendly_iqs_error(_parse_script_error(completed.stderr, completed.stdout)))
    raw = str(completed.stdout or "").strip()
    if not raw:
        raise RuntimeError("IQS Skill 返回了空输出。")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"IQS Skill 返回了非 JSON 输出：{raw[:800]}") from exc


def call_iqs_search(query: str, options: Dict[str, Any]) -> List[Dict[str, Any]]:
    query = _clean_search_query(query, max_chars=_iqs_query_max_chars())
    skill_dir = discover_iqs_skill_dir()
    script = skill_dir / "scripts" / "search.mjs"
    args = [
        "--query",
        query,
        "--engineType",
        str(options.get("engineType") or "LiteAdvanced"),
        "--timeRange",
        str(options.get("timeRange") or "NoLimit"),
        "--contents",
        str(options.get("contents") or "summary"),
        "--numResults",
        str(options.get("numResults") or 6),
        "--timeout",
        str(options.get("timeout") or 20000),
    ]
    if options.get("category"):
        args.extend(["--category", str(options.get("category"))])
    payload = run_iqs_script(script, args, timeout_ms=int(options.get("timeout") or 20000))
    if not isinstance(payload, list):
        raise RuntimeError("IQS 搜索输出的根结构必须是列表。")
    return [dict(item) for item in payload if isinstance(item, dict)]


def call_iqs_search_batch(requests: Sequence[Dict[str, Any]], *, timeout_ms: int, concurrency: int) -> List[Dict[str, Any]]:
    if not requests:
        return []
    skill_dir = discover_iqs_skill_dir()
    script = skill_dir / "scripts" / "search.mjs"
    args = [
        "--batchJson",
        json.dumps(list(requests), ensure_ascii=False, default=str),
        "--batchConcurrency",
        str(max(1, concurrency)),
    ]
    waves = (len(requests) + max(1, concurrency) - 1) // max(1, concurrency)
    payload = run_iqs_script(script, args, timeout_ms=max(timeout_ms, timeout_ms * waves + 5000))
    if not isinstance(payload, list):
        raise RuntimeError("IQS batch search output must be a list.")
    return [dict(item) for item in payload if isinstance(item, dict)]


def call_iqs_readpage(url: str, *, timeout_ms: int = 60000) -> Dict[str, Any]:
    skill_dir = discover_iqs_skill_dir()
    script = skill_dir / "scripts" / "readpage.mjs"
    args = [
        "--url",
        url,
        "--format",
        "markdown",
        "--timeout",
        str(timeout_ms),
        "--extractArticle",
        "true",
    ]
    payload = run_iqs_script(script, args, timeout_ms=timeout_ms)
    if not isinstance(payload, dict):
        raise RuntimeError("IQS 网页读取输出的根结构必须是对象。")
    return dict(payload)


def _compact_text(value: Any, max_chars: int = 700) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "..."


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, min_value: int = 1, max_value: int = 100) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return min(max_value, max(min_value, value))


def _iqs_query_max_chars() -> int:
    return _env_int("IQS_QUERY_MAX_CHARS", 56, min_value=32, max_value=96)


def _iqs_query_max_bytes() -> int:
    return _env_int("IQS_QUERY_MAX_BYTES", 64, min_value=40, max_value=180)


def _iqs_query_fits(text: str, *, max_chars: Optional[int] = None, max_bytes: Optional[int] = None) -> bool:
    value = str(text or "")
    return len(value) <= (max_chars or _iqs_query_max_chars()) and len(value.encode("utf-8")) <= (max_bytes or _iqs_query_max_bytes())


def _is_iqs_quota_error(value: Any) -> bool:
    return bool(_IQS_QUOTA_ERROR_RE.search(str(value or "")))


def _friendly_iqs_error(value: Any) -> str:
    text = str(value or "").strip()
    if _is_iqs_quota_error(text):
        return "IQS 检索额度或测试账号请求上限已触发（Retrieval.TestUserQueryExceeded）"
    return text


def _search_profile(options: Optional[Dict[str, Any]]) -> str:
    raw = ""
    if isinstance(options, dict):
        raw = str(options.get("search_profile") or options.get("searchProfile") or "").strip().lower()
    if not raw:
        raw = os.getenv("IQS_SEARCH_PROFILE", "").strip().lower()
    return raw if raw in {"initial", "followup"} else ""


def _explicit_int(options: Optional[Dict[str, Any]], keys: Sequence[str]) -> Optional[int]:
    if not isinstance(options, dict):
        return None
    for key in keys:
        if key not in options:
            continue
        try:
            return int(options.get(key))
        except (TypeError, ValueError):
            return None
    return None


def _profile_int(
    options: Optional[Dict[str, Any]],
    keys: Sequence[str],
    *,
    env_name: str,
    default: int,
    initial_env: str,
    initial_default: int,
    followup_env: str,
    followup_default: int,
    min_value: int = 1,
    max_value: int = 100,
) -> int:
    explicit = _explicit_int(options, keys)
    if explicit is not None:
        return min(max_value, max(min_value, explicit))
    profile = _search_profile(options)
    if profile == "initial":
        return _env_int(initial_env, initial_default, min_value=min_value, max_value=max_value)
    if profile == "followup":
        return _env_int(followup_env, followup_default, min_value=min_value, max_value=max_value)
    return _env_int(env_name, default, min_value=min_value, max_value=max_value)


def _option_flag(options: Optional[Dict[str, Any]], keys: Sequence[str], env_name: str, default: bool) -> bool:
    if isinstance(options, dict):
        for key in keys:
            if key in options:
                value = options.get(key)
                if isinstance(value, bool):
                    return value
                return str(value).strip().lower() in {"1", "true", "yes", "on"}
    return _env_flag(env_name, default)


def _option_text(options: Optional[Dict[str, Any]], keys: Sequence[str], env_name: str, default: str = "") -> str:
    if isinstance(options, dict):
        for key in keys:
            value = str(options.get(key) or "").strip()
            if value:
                return value
    return os.getenv(env_name, default).strip()


def _split_csv(value: Any) -> List[str]:
    seen = set()
    items: List[str] = []
    for part in re.split(r"[,，;\s]+", str(value or "")):
        cleaned = part.strip()
        if not cleaned or cleaned.lower() == "auto" or cleaned in seen:
            continue
        seen.add(cleaned)
        items.append(cleaned)
    return items


def _env_csv(name: str, default: str) -> List[str]:
    return _split_csv(os.getenv(name, default))


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _web_cache_allowed(search_options: Optional[Dict[str, Any]]) -> bool:
    options = _as_dict(search_options)
    if str(options.get("disable_cache") or "").strip().lower() in {"1", "true", "yes", "on"}:
        return False
    if str(options.get("cache_ttl_seconds") or "").strip() == "0":
        return False
    return _env_flag("IQS_AGENT_CACHE_ENABLED", True)


def _web_cache_key(
    *,
    query: str,
    messages: Sequence[Dict[str, Any]],
    urls: Sequence[str],
    search_options: Optional[Dict[str, Any]],
    enable_llm_analysis: bool,
) -> str:
    options = _as_dict(search_options)
    return make_cache_key(
        "web_analysis_agent",
        {
            "query": query,
            "messages": list(messages or []),
            "urls": list(urls or []),
            "search_options": options,
            "session_id": str(options.get("session_id") or "").strip(),
            "user_id": str(options.get("user_id") or "").strip(),
            "enable_llm_analysis": bool(enable_llm_analysis),
        },
    )


def _mark_web_cache_hit(state: "WebAnalysisAgentState") -> "WebAnalysisAgentState":
    cached = dict(state)
    cached["metadata"] = {
        **dict(cached.get("metadata") or {}),
        "cache_hit": "web_analysis_agent",
    }
    raw_output = dict(cached.get("raw_output") or {})
    if raw_output:
        raw_output["cache_hit"] = "web_analysis_agent"
        cached["raw_output"] = raw_output
    return cached


def _iqs_search_cache_allowed(search_options: Optional[Dict[str, Any]]) -> bool:
    options = _as_dict(search_options)
    if str(options.get("disable_cache") or "").strip().lower() in {"1", "true", "yes", "on"}:
        return False
    if str(options.get("cache_ttl_seconds") or "").strip() == "0":
        return False
    return _env_flag("IQS_SEARCH_CACHE_ENABLED", True)


def _iqs_search_cache_key(query: str, search_options: Dict[str, Any]) -> str:
    options = _as_dict(search_options)
    return make_cache_key(
        "iqs_optimized_search",
        {
            "query": query,
            "search_options": options,
            "session_id": str(options.get("session_id") or "").strip(),
            "user_id": str(options.get("user_id") or "").strip(),
        },
    )


def _search_task_dedupe_key(task: Dict[str, Any]) -> tuple[Any, ...]:
    text = re.sub(r"\s+", " ", str(task.get("text") or "").strip().lower())
    return (
        text,
        str(task.get("engineType") or "").strip(),
        str(task.get("timeRange") or "").strip(),
        str(task.get("contents") or "").strip(),
        tuple(str(item).strip().lower() for item in _as_list(task.get("must_have_terms")) if str(item).strip()),
        tuple(str(item).strip().lower() for item in _as_list(task.get("forbidden_terms")) if str(item).strip()),
        tuple(str(item).strip().lower() for item in _as_list(task.get("source_priority")) if str(item).strip()),
        str(task.get("proof_role") or "").strip().lower(),
        str(task.get("evidence_type") or "").strip().lower(),
    )


def _normalize_url(url: str) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw)
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/")
    return f"{parsed.scheme.lower()}://{netloc}{path}" if parsed.scheme and netloc else raw.rstrip("/")


def _domain(url: str) -> str:
    return urlparse(str(url or "")).netloc.lower().removeprefix("www.")


def _result_text(item: Dict[str, Any], *, max_chars: int = 1200) -> str:
    parts = [
        item.get("title"),
        item.get("snippet"),
        item.get("summary"),
        item.get("mainText"),
        item.get("content"),
    ]
    return _compact_text(" ".join(str(part or "") for part in parts), max_chars=max_chars)


def classify_query_intent(query: str) -> str:
    text = str(query or "")
    if _POLICY_INTENT_RE.search(text):
        return "政策型"
    if _FINANCE_INTENT_RE.search(text):
        return "财报型"
    if _NEWS_INTENT_RE.search(text):
        return "新闻型"
    if _DATA_INTENT_RE.search(text):
        return "数据型"
    if _ANALYSIS_INTENT_RE.search(text):
        return "分析型"
    return "分析型"


def _dynamic_intent_label(intent: Any, query: str = "") -> str:
    raw = str(intent or "").strip().lower()
    if raw in {"data", "statistics", "metric", "market"}:
        return "数据型"
    if raw in {"policy", "regulation"}:
        return "政策型"
    if raw in {"news", "risk"}:
        return "新闻型"
    if raw in {"filing", "finance", "financial"}:
        return "财报型"
    if raw in {"company", "case", "academic", "analysis", "technical_case"}:
        return "分析型"
    return classify_query_intent(query)


def _clean_search_query(text: str, *, max_chars: int = 80) -> str:
    max_chars = min(max_chars, _iqs_query_max_chars())
    max_bytes = _iqs_query_max_bytes()
    cleaned = re.sub(r"https?://[^\s)）>]+", " ", str(text or ""))
    cleaned = re.sub(r"[/、，。！？?；;：:\[\]【】（）(){}<>《》\"']", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if _iqs_query_fits(cleaned, max_chars=max_chars, max_bytes=max_bytes):
        return cleaned
    tokens: List[str] = []
    for token in cleaned.split():
        candidate = " ".join(tokens + [token]).strip()
        if not _iqs_query_fits(candidate, max_chars=max_chars, max_bytes=max_bytes):
            if tokens:
                continue
            trimmed = ""
            for char in token:
                next_value = trimmed + char
                if not _iqs_query_fits(next_value, max_chars=max_chars, max_bytes=max_bytes):
                    break
                trimmed = next_value
            return trimmed.strip()
        tokens.append(token)
    if tokens:
        return " ".join(tokens).strip()
    return ""


def _compact_query_terms(values: Sequence[Any], *, limit: int = 4, max_chars: int = 10) -> List[str]:
    terms: List[str] = []
    seen = set()
    for value in values:
        cleaned = re.sub(r"https?://[^\s)）>]+", " ", str(value or ""))
        cleaned = re.sub(r"[/、，。！？?；;：:\[\]【】（）(){}<>《》\"']", " ", cleaned)
        for part in re.split(r"\s+", cleaned.strip()):
            token = part.strip()[:max_chars].strip()
            key = token.lower()
            if not key or key in seen:
                continue
            seen.add(key)
            terms.append(token)
            if len(terms) >= limit:
                return terms
    return terms


def _time_scope_for_query(query: str, intent: str) -> str:
    current_year = datetime.now().year
    if re.search(r"202[0-9]|203[0-9]", query):
        return ""
    if intent == "新闻型":
        return f"{current_year} 最新 近三个月"
    if intent in {"政策型", "财报型", "数据型"}:
        return str(current_year)
    return f"{current_year} 最新"


def _route_options_for_intent(intent: str, base_options: Dict[str, Any]) -> Dict[str, Any]:
    options = dict(base_options or {})
    forced_engine = str(options.get("engineType") or "").strip() if options.get("engineMode") == "forced" else ""
    route_env = _INTENT_ROUTE_ENV.get(intent, "IQS_ENGINE_ROUTE_ANALYSIS")
    route_default = _INTENT_ROUTE_DEFAULTS.get(intent, "LiteAdvanced,Generic")
    engine_types = [forced_engine] if forced_engine else _env_csv(route_env, route_default)
    if not engine_types:
        engine_types = ["Generic" if intent == "新闻型" else "LiteAdvanced"]
    if intent == "新闻型":
        options["engineType"] = engine_types[0]
        options["timeRange"] = options.get("timeRange") if options.get("timeRange") != "NoLimit" else "OneMonth"
        options["contents"] = "summary"
    elif intent == "政策型":
        options["engineType"] = engine_types[0]
        options["timeRange"] = "OneYear" if options.get("timeRange") in {"", None, "NoLimit"} else options.get("timeRange")
        options["contents"] = "mainText"
    elif intent in {"数据型", "财报型"}:
        options["engineType"] = engine_types[0]
        options["timeRange"] = "OneYear" if options.get("timeRange") in {"", None, "NoLimit"} else options.get("timeRange")
        options["contents"] = "mainText"
    else:
        options["engineType"] = engine_types[0]
        options["contents"] = "mainText"
    if not options.get("timeRange"):
        options["timeRange"] = "NoLimit"
    options["engineTypes"] = engine_types
    return options


def build_rule_based_query_plan(query: str, base_options: Dict[str, Any]) -> List[Dict[str, Any]]:
    core = _clean_search_query(query)
    if not core:
        return []
    current_year = datetime.now().year
    primary_intent = classify_query_intent(query)
    candidates = [
        {"text": f"{core} {_time_scope_for_query(query, primary_intent)}".strip(), "intent": primary_intent, "time_scope": _time_scope_for_query(query, primary_intent), "source": "rule_primary"},
        {"text": f"{core} 可核验证据 来源 口径 时间范围 {current_year}", "intent": "数据型", "time_scope": str(current_year), "source": "rule_dynamic_evidence"},
        {"text": f"{core} 官方文件 公告 财报 统计数据 {current_year}", "intent": "财报型", "time_scope": str(current_year), "source": "rule_dynamic_source"},
        {"text": f"{core} 案例 事件 风险 反向证据 {current_year}", "intent": "新闻型" if _CURRENT_INFO_RE.search(query) else "分析型", "time_scope": str(current_year), "source": "rule_dynamic_counter"},
    ]
    max_queries = _profile_int(
        base_options,
        ["maxQueries", "max_queries"],
        env_name="IQS_MAX_QUERIES",
        default=6,
        initial_env="IQS_INITIAL_MAX_QUERIES",
        initial_default=6,
        followup_env="IQS_FOLLOWUP_MAX_QUERIES",
        followup_default=4,
        min_value=1,
        max_value=10,
    )
    seen = set()
    plan: List[Dict[str, Any]] = []
    for item in candidates:
        text = _clean_search_query(item.get("text", ""), max_chars=100)
        if not text or text in seen:
            continue
        seen.add(text)
        routed_options = _route_options_for_intent(str(item.get("intent") or "分析型"), base_options)
        plan.append(
            {
                **item,
                "text": text,
                "engineType": routed_options.get("engineType", "LiteAdvanced"),
                "engineTypes": routed_options.get("engineTypes", [routed_options.get("engineType", "LiteAdvanced")]),
                "timeRange": routed_options.get("timeRange", "NoLimit"),
                "contents": routed_options.get("contents", "summary"),
            }
        )
        if len(plan) >= max_queries:
            break
    return plan


def build_task_query_plan(query: str, base_options: Dict[str, Any]) -> List[Dict[str, Any]]:
    search_task = _as_dict(base_options.get("search_task"))
    if not search_task:
        return []
    task_query = _clean_search_query(search_task.get("query") or query, max_chars=120)
    if not task_query:
        return []
    intent = _dynamic_intent_label(search_task.get("intent"), task_query)
    routed_options = _route_options_for_intent(intent, base_options)
    current_year = datetime.now().year
    must_terms = _compact_query_terms(_as_list(search_task.get("must_have_terms")), limit=5, max_chars=10)
    forbidden_terms = _compact_query_terms(_as_list(search_task.get("forbidden_terms")), limit=5, max_chars=10)
    source_priority = _compact_query_terms(_as_list(search_task.get("source_priority")), limit=5, max_chars=14)
    base = task_query
    must = " ".join(must_terms[:2])
    variants_with_source = [
        (base, "primary", "dynamic_search_task"),
        (f"{base} {must} 数据 统计 口径 {current_year}", "数据型", "dynamic_metric"),
        (f"{base} 官方 统计局 协会 白皮书 政策 文件", "数据型", "dynamic_official_data"),
        (f"{base} 年报 公告 招股书 交易所 投资者关系", "财报型", "dynamic_filing_company"),
        (f"{base} 券商研报 行业报告 市场规模 增速", "分析型", "dynamic_market_research"),
        (f"{base} 客户 案例 订单 中标 采购 应用场景", "分析型", "dynamic_customer_case"),
        (f"{base} 风险 反证 下滑 过剩 价格战 失败案例", "新闻型", "dynamic_counter_evidence"),
    ]
    proof_role = str(search_task.get("proof_role") or "").strip().lower()
    if proof_role == "metric":
        variants_with_source = [item for item in variants_with_source if item[2] in {"dynamic_search_task", "dynamic_metric", "dynamic_official_data", "dynamic_market_research"}]
    elif proof_role == "source_check":
        variants_with_source = [item for item in variants_with_source if item[2] in {"dynamic_search_task", "dynamic_official_data", "dynamic_filing_company", "dynamic_market_research"}]
    elif proof_role == "case":
        variants_with_source = [item for item in variants_with_source if item[2] in {"dynamic_search_task", "dynamic_customer_case", "dynamic_filing_company"}]
    elif proof_role == "counter":
        variants_with_source = [item for item in variants_with_source if item[2] in {"dynamic_search_task", "dynamic_counter_evidence", "dynamic_market_research"}]
    if source_priority:
        variants_with_source.append((f"{base} {must} {' '.join(source_priority[:3])}", intent, "dynamic_source_priority"))
    plan: List[Dict[str, Any]] = []
    seen = set()
    for variant, variant_intent, variant_source in variants_with_source:
        text = _clean_search_query(variant, max_chars=120)
        if not text or text in seen:
            continue
        seen.add(text)
        item_intent = _dynamic_intent_label(variant_intent, text)
        item_options = _route_options_for_intent(item_intent, base_options)
        plan.append(
            {
                "text": text,
                "intent": item_intent,
                "time_scope": str(search_task.get("freshness") or ""),
                "source": variant_source,
                "engineType": item_options.get("engineType", "LiteAdvanced"),
                "engineTypes": item_options.get("engineTypes", [item_options.get("engineType", "LiteAdvanced")]),
                "timeRange": item_options.get("timeRange", "NoLimit"),
                "contents": item_options.get("contents", "summary"),
                "task_id": search_task.get("task_id"),
                "dimension_id": search_task.get("dimension_id"),
                "dimension_name": search_task.get("dimension_name"),
                "chapter_id": search_task.get("chapter_id"),
                "chapter_title": search_task.get("chapter_title"),
                "chapter_question": search_task.get("chapter_question"),
                "evidence_goal": search_task.get("evidence_goal"),
                "evidence_goal_id": search_task.get("evidence_goal_id"),
                "proof_role": search_task.get("proof_role"),
                "must_have_terms": must_terms,
                "forbidden_terms": forbidden_terms,
                "source_priority": source_priority,
                "search_task": dict(search_task),
            }
        )
    return plan[
        : _profile_int(
            base_options,
            ["maxQueries", "max_queries"],
            env_name="IQS_MAX_QUERIES",
            default=8,
            initial_env="IQS_INITIAL_MAX_QUERIES",
            initial_default=8,
            followup_env="IQS_FOLLOWUP_MAX_QUERIES",
            followup_default=6,
            min_value=1,
            max_value=10,
        )
    ]


def build_llm_query_plan(
    query: str,
    base_options: Dict[str, Any],
    research_plan: Optional[Dict[str, Any]] = None,
    search_task: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    if not _env_flag("IQS_ENABLE_LLM_QUERY_REWRITE", False):
        return []
    llm_config = build_llm_config()
    if not llm_config_is_ready(llm_config):
        return []
    dynamic_mode = bool(research_plan or search_task or base_options.get("research_plan") or base_options.get("search_task"))
    system_prompt = """
你是行研联网检索 Query 优化器。将用户问题改写为更适合搜索引擎的查询词，严格输出 JSON。
输出格式：
{"queries":[{"text":"优化后的查询词","intent":"数据型|新闻型|政策型|分析型|财报型","time_scope":"时间限定"}]}
规则：
1. 将自然语言改写为关键词组合。
2. 补充当前年份、最新、近三个月、Q1/Q2/Q3/Q4 等时间词。
3. 补充市场规模、增速、政策、监管、财报、融资、IPO、估值等限定词。
4. 复合问题拆成 5-8 个独立查询，尽量覆盖市场、竞争、政策、技术、资本、风险、趋势不同角度。
5. 每条查询尽量只聚焦一个子问题，避免把多个维度揉成一句话。
6. 优先生成能找到具体数字的查询：市场规模、CAGR、出货量、销量、市占率、营收、净利润、毛利率、亏损、现金流、融资金额、估值、股价、市值。
7. 如果原问题是竞争分析/投融资/公司经营类，额外补充企业名、年份、财报口径、政策口径、地区口径。
""".strip()
    if dynamic_mode:
        system_prompt = """
你是动态研究检索 Query Planner。
你会收到用户问题、research_plan 和可选 search_task。

输出 JSON：
{"queries":[
  {
    "text":"优化后的查询词",
    "intent":"数据型|新闻型|政策型|分析型|财报型",
    "dimension_id":"...",
    "dimension_name":"...",
    "evidence_goal":"...",
    "must_have_terms":[],
    "forbidden_terms":[],
    "source_priority":[],
    "time_scope":"..."
  }
]}

规则：
1. 如果提供 search_task，只围绕该 task 改写，不扩展到无关维度。
2. 如果 research_type 是 urban_population，不得加入市场规模、CAGR、市占率、融资、IPO、估值。
3. 如果 research_type 是 policy_research，优先找政策原文、解读、实施细则、影响主体，不要找市场规模。
4. 如果 research_type 是 company_due_diligence，优先找官网、公告、工商、财报、诉讼、融资。
5. 查询词必须包含 must_have_terms 中至少一个核心词。
6. 查询词不得包含 forbidden_terms。
7. 每条查询只服务一个 evidence_goal。
只返回 JSON。
""".strip()
    try:
        response = call_openai_compatible_json(
            config=llm_config,
            system_prompt=system_prompt,
            user_payload={
                "query": query,
                "current_year": datetime.now().year,
                "research_plan": research_plan or base_options.get("research_plan"),
                "search_task": search_task or base_options.get("search_task"),
            },
        )
    except Exception:
        logger.exception("LLM query rewrite failed", extra={"query": query})
        return []
    payload = response.get("payload") or {}
    values = payload.get("queries") if isinstance(payload, dict) else []
    if not isinstance(values, list):
        return []
    plan: List[Dict[str, Any]] = []
    for raw in values:
        if not isinstance(raw, dict):
            continue
        text = _clean_search_query(raw.get("text"), max_chars=100)
        if not text:
            continue
        intent = str(raw.get("intent") or classify_query_intent(text)).strip()
        if intent not in {"数据型", "新闻型", "政策型", "分析型", "财报型"}:
            intent = _dynamic_intent_label(intent, text)
        routed_options = _route_options_for_intent(intent, base_options)
        task = _as_dict(search_task or base_options.get("search_task"))
        plan.append(
            {
                "text": text,
                "intent": intent,
                "time_scope": str(raw.get("time_scope") or ""),
                "source": "llm_rewrite",
                "engineType": routed_options.get("engineType", "LiteAdvanced"),
                "engineTypes": routed_options.get("engineTypes", [routed_options.get("engineType", "LiteAdvanced")]),
                "timeRange": routed_options.get("timeRange", "NoLimit"),
                "contents": routed_options.get("contents", "summary"),
                "task_id": raw.get("task_id") or task.get("task_id"),
                "dimension_id": raw.get("dimension_id") or task.get("dimension_id"),
                "dimension_name": raw.get("dimension_name") or task.get("dimension_name"),
                "evidence_goal": raw.get("evidence_goal") or task.get("evidence_goal"),
                "must_have_terms": _as_list(raw.get("must_have_terms")) or _as_list(task.get("must_have_terms")),
                "forbidden_terms": _as_list(raw.get("forbidden_terms")) or _as_list(task.get("forbidden_terms")),
                "source_priority": _as_list(raw.get("source_priority")) or _as_list(task.get("source_priority")),
                "search_task": task,
            }
        )
    return plan[
        : _profile_int(
            base_options,
            ["maxQueries", "max_queries"],
            env_name="IQS_MAX_QUERIES",
            default=8,
            initial_env="IQS_INITIAL_MAX_QUERIES",
            initial_default=2,
            followup_env="IQS_FOLLOWUP_MAX_QUERIES",
            followup_default=2,
            min_value=1,
            max_value=8,
        )
    ]


def build_hyde_query(query: str, base_options: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not _env_flag("IQS_ENABLE_HYDE", False):
        return []
    llm_config = build_llm_config()
    if not llm_config_is_ready(llm_config):
        return []
    system_prompt = """
你是行研 RAG 的 HyDE 查询扩展器。假设你已经找到关于用户问题的完整行研报告，
写出一段 180-260 字的核心段落，尽量包含数据口径、时间范围、主体名称、政策/市场/竞争/技术/资本维度关键词，
并加入更容易触发行研语义召回的具体数字与趋势表达。
这段文字只用于语义检索，不是最终答案。严格输出 JSON：{"hyde":"..."}。
""".strip()
    try:
        response = call_openai_compatible_json(
            config=llm_config,
            system_prompt=system_prompt,
            user_payload={"query": query, "current_year": datetime.now().year},
        )
    except Exception:
        logger.exception("LLM HyDE query generation failed", extra={"query": query})
        return []
    payload = response.get("payload") or {}
    hyde = _clean_search_query(str(payload.get("hyde") or ""), max_chars=180) if isinstance(payload, dict) else ""
    if not hyde:
        return []
    routed_options = _route_options_for_intent("分析型", base_options)
    return [
        {
            "text": hyde,
            "intent": "分析型",
            "time_scope": str(datetime.now().year),
            "source": "llm_hyde",
            "engineType": routed_options.get("engineType", "LiteAdvanced"),
            "engineTypes": routed_options.get("engineTypes", [routed_options.get("engineType", "LiteAdvanced")]),
            "timeRange": routed_options.get("timeRange", "NoLimit"),
            "contents": "mainText",
        }
    ]


def build_query_plan(query: str, base_options: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not _env_flag("IQS_ENABLE_QUERY_OPTIMIZATION", True):
        intent = classify_query_intent(query)
        routed_options = _route_options_for_intent(intent, base_options)
        return [
            {
                "text": _clean_search_query(query),
                "intent": intent,
                "time_scope": "",
                "source": "raw",
                "engineType": routed_options.get("engineType", "LiteAdvanced"),
                "engineTypes": routed_options.get("engineTypes", [routed_options.get("engineType", "LiteAdvanced")]),
                "timeRange": routed_options.get("timeRange", "NoLimit"),
                "contents": routed_options.get("contents", "summary"),
            }
        ]
    search_task = _as_dict(base_options.get("search_task"))
    research_plan = _as_dict(base_options.get("research_plan"))
    if search_task:
        plan = (
            build_llm_query_plan(query, base_options, research_plan=research_plan, search_task=search_task)
            or build_task_query_plan(query, base_options)
        )
    else:
        plan = build_llm_query_plan(query, base_options, research_plan=research_plan) or build_rule_based_query_plan(query, base_options)
    hyde_plan = [] if search_task else build_hyde_query(query, base_options)
    if hyde_plan:
        plan = [*plan[:1], *hyde_plan, *plan[1:]]
    if not plan:
        return []
    seen = set()
    deduped = []
    for item in plan:
        text = str(item.get("text") or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(item)
    return deduped[
        : _profile_int(
            base_options,
            ["maxQueries", "max_queries"],
            env_name="IQS_MAX_QUERIES",
            default=8,
            initial_env="IQS_INITIAL_MAX_QUERIES",
            initial_default=2,
            followup_env="IQS_FOLLOWUP_MAX_QUERIES",
            followup_default=2,
            min_value=1,
            max_value=8,
        )
    ]


def _options_for_query_item(query_item: Dict[str, Any], base_options: Dict[str, Any]) -> Dict[str, Any]:
    options = dict(base_options or {})
    options["engineType"] = query_item.get("engineType") or options.get("engineType") or "LiteAdvanced"
    options["timeRange"] = query_item.get("timeRange") or options.get("timeRange") or "NoLimit"
    options["contents"] = query_item.get("contents") or options.get("contents") or "mainText"
    result_keys = ["resultsPerQuery", "results_per_query"]
    if not _search_profile(options):
        result_keys.extend(["numResults", "num_results"])
    options["numResults"] = _profile_int(
        options,
        result_keys,
        env_name="IQS_RESULTS_PER_QUERY",
        default=int(options.get("numResults") or 100),
        initial_env="IQS_INITIAL_RESULTS_PER_QUERY",
        initial_default=100,
        followup_env="IQS_FOLLOWUP_RESULTS_PER_QUERY",
        followup_default=100,
        min_value=1,
        max_value=100,
    )
    return options


def _search_request(text: str, options: Dict[str, Any]) -> Dict[str, Any]:
    text = _clean_search_query(text, max_chars=_iqs_query_max_chars())
    request = {
        "query": text,
        "engineType": str(options.get("engineType") or "LiteAdvanced"),
        "timeRange": str(options.get("timeRange") or "NoLimit"),
        "contents": str(options.get("contents") or "summary"),
        "numResults": int(options.get("numResults") or 100),
        "timeout": int(options.get("timeout") or 20000),
    }
    if options.get("category"):
        request["category"] = str(options.get("category"))
    return request


def _fallback_options_for_query(query_item: Dict[str, Any], primary_options: Dict[str, Any]) -> Dict[str, Any]:
    fallback_options = dict(primary_options)
    fallback_engines = _env_csv("IQS_FALLBACK_ENGINE_TYPES", "Generic,LiteAdvanced")
    primary_engine = str(primary_options.get("engineType") or "")
    fallback_engine = next((item for item in fallback_engines if item != primary_engine), "")

    if not fallback_engine:
        fallback_engine = "Generic" if primary_engine == "LiteAdvanced" else "LiteAdvanced"

    fallback_options["engineType"] = fallback_engine
    fallback_options["contents"] = primary_options.get("contents") or "mainText"

    intent = str(query_item.get("intent") or "").strip().lower()
    bounded_intents = {"新闻型", "政策型", "财报型", "news", "policy", "filing", "finance", "financial"}
    if fallback_options.get("timeRange") == "NoLimit" and any(item in intent for item in bounded_intents):
        fallback_options["timeRange"] = "OneYear"

    return fallback_options

def _should_use_fallback(results: Sequence[Dict[str, Any]], trace: Dict[str, Any], options: Dict[str, Any]) -> bool:
    if not _option_flag(options, ["enableFallback", "enable_fallback"], "IQS_ENABLE_FALLBACK", True):
        return False
    fallback_mode = _option_text(options, ["fallbackMode", "fallback_mode"], "IQS_FALLBACK_MODE", "empty_only").lower()
    if fallback_mode in {"never", "off", "false", "0"}:
        return False
    if trace.get("primary_error"):
        return True
    if fallback_mode in {"empty", "empty_only", "zero", "zero_only"}:
        return len(results) == 0
    min_results = _profile_int(
        options,
        ["minResultsPerQuery", "min_results_per_query"],
        env_name="IQS_MIN_RESULTS_PER_QUERY",
        default=2,
        initial_env="IQS_INITIAL_MIN_RESULTS_PER_QUERY",
        initial_default=1,
        followup_env="IQS_FOLLOWUP_MIN_RESULTS_PER_QUERY",
        followup_default=1,
        min_value=1,
        max_value=10,
    )
    return len(results) < min_results

def call_iqs_search_with_fallback(query_item: Dict[str, Any], base_options: Dict[str, Any]) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    text = _clean_search_query(str(query_item.get("text") or "").strip(), max_chars=_iqs_query_max_chars())
    primary_options = _options_for_query_item(query_item, base_options)
    trace = {
        "query": text,
        "intent": query_item.get("intent"),
        "source": query_item.get("source"),
        "primary_options": primary_options,
        "fallback_used": False,
        "errors": [],
    }
    for key in ["task_id", "dimension_id", "dimension_name", "evidence_goal", "must_have_terms", "forbidden_terms", "source_priority", "hypothesis_id", "hypothesis_statement", "proof_role", "proof_standard", "evidence_type", "lane_targets", "scheduled_lane_type", "scheduled_lane", "counter_evidence", "decision_use", "search_task"]:
        if key in query_item:
            trace[key] = query_item.get(key)
    try:
        results = call_iqs_search(text, primary_options)
        trace["primary_count"] = len(results)
        if not _should_use_fallback(results, trace, primary_options):
            return results, trace
        trace["errors"].append("主查询结果数量不足，启用降级查询")
    except Exception as exc:
        logger.exception("IQS primary search failed", extra={"query": text})
        results = []
        trace["primary_error"] = True
        error_text = _friendly_iqs_error(exc)
        trace["errors"].append(error_text)
        if _is_iqs_quota_error(error_text):
            trace["quota_exceeded"] = True
            return results, trace

    fallback_options = _fallback_options_for_query(query_item, primary_options)
    trace["fallback_used"] = True
    trace["fallback_options"] = fallback_options
    try:
        fallback_results = call_iqs_search(text, fallback_options)
        trace["fallback_count"] = len(fallback_results)
        return results + fallback_results, trace
    except Exception as exc:
        logger.exception("IQS fallback search failed", extra={"query": text})
        error_text = _friendly_iqs_error(exc)
        trace["errors"].append(error_text)
        if _is_iqs_quota_error(error_text):
            trace["quota_exceeded"] = True
        return results, trace


def _extend_raw_results(raw_results: List[Dict[str, Any]], results: Sequence[Dict[str, Any]], trace: Dict[str, Any]) -> None:
    for rank, result in enumerate(results, start=1):
        copied = dict(result)
        copied["origin_query"] = trace.get("query")
        copied["origin_intent"] = trace.get("intent")
        copied["origin_query_source"] = trace.get("source")
        copied["origin_rank"] = rank
        for key in ["task_id", "dimension_id", "dimension_name", "evidence_goal", "must_have_terms", "forbidden_terms", "source_priority", "hypothesis_id", "hypothesis_statement", "proof_role", "proof_standard", "evidence_type", "lane_targets", "scheduled_lane_type", "scheduled_lane", "counter_evidence", "decision_use", "search_task"]:
            if key in trace:
                copied[key] = trace.get(key)
        raw_results.append(copied)


def _run_iqs_search_tasks_batch(
    search_tasks: Sequence[Dict[str, Any]],
    base_options: Dict[str, Any],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    started = time.perf_counter()
    raw_results: List[Dict[str, Any]] = []
    search_trace: List[Dict[str, Any]] = []
    errors: List[str] = []
    prepared: List[Dict[str, Any]] = []
    requests: List[Dict[str, Any]] = []
    max_timeout = 0
    for index, query_item in enumerate(search_tasks):
        text = _clean_search_query(str(query_item.get("text") or "").strip(), max_chars=_iqs_query_max_chars())
        primary_options = _options_for_query_item(query_item, base_options)
        trace = {
            "query": text,
            "intent": query_item.get("intent"),
            "source": query_item.get("source"),
            "primary_options": primary_options,
            "fallback_used": False,
            "batch_index": index,
            "errors": [],
        }
        for key in ["task_id", "dimension_id", "dimension_name", "evidence_goal", "must_have_terms", "forbidden_terms", "source_priority", "hypothesis_id", "hypothesis_statement", "proof_role", "proof_standard", "evidence_type", "lane_targets", "scheduled_lane_type", "scheduled_lane", "counter_evidence", "decision_use", "search_task"]:
            if key in query_item:
                trace[key] = query_item.get(key)
        prepared.append({"query_item": query_item, "options": primary_options, "trace": trace, "results": []})
        requests.append(_search_request(text, primary_options))
        max_timeout = max(max_timeout, int(primary_options.get("timeout") or 20000))

    concurrency = _env_int("IQS_BATCH_CONCURRENCY", min(len(requests), 4), min_value=1, max_value=12)
    _progress(
        "iqs-batch",
        "批量搜索开始",
        requests=len(requests),
        concurrency=concurrency,
        timeout_ms=max_timeout or 20000,
    )
    responses = call_iqs_search_batch(requests, timeout_ms=max_timeout or 20000, concurrency=concurrency)
    responses_by_index = {int(item.get("index", index)): item for index, item in enumerate(responses)}
    fallback_prepared: List[Dict[str, Any]] = []
    fallback_requests: List[Dict[str, Any]] = []
    for index, item in enumerate(prepared):
        trace = item["trace"]
        response = responses_by_index.get(index, {})
        results: List[Dict[str, Any]] = []
        if response.get("ok", True):
            results = [dict(value) for value in list(response.get("results") or []) if isinstance(value, dict)]
            trace["primary_count"] = len(results)
        else:
            trace["primary_error"] = True
            error_text = _friendly_iqs_error(response.get("error") or "IQS batch search failed")
            trace["errors"].append(error_text)
            if _is_iqs_quota_error(error_text):
                trace["quota_exceeded"] = True
        item["results"] = results
        if trace.get("quota_exceeded"):
            _extend_raw_results(raw_results, results, trace)
            search_trace.append(trace)
            for error in trace.get("errors") or []:
                errors.append(f"{trace.get('query')}: {error}")
        elif _should_use_fallback(results, trace, item["options"]):
            fallback_options = _fallback_options_for_query(item["query_item"], item["options"])
            trace["fallback_used"] = True
            trace["fallback_options"] = fallback_options
            fallback_prepared.append({**item, "fallback_options": fallback_options})
            fallback_requests.append(_search_request(str(trace.get("query") or ""), fallback_options))
        else:
            if len(results) == 0 and not trace.get("primary_error"):
                trace["fallback_skipped"] = True
            _extend_raw_results(raw_results, results, trace)
            search_trace.append(trace)
            for error in trace.get("errors") or []:
                errors.append(f"{trace.get('query')}: {error}")

    if fallback_requests:
        fallback_timeout = max(int(item["fallback_options"].get("timeout") or 20000) for item in fallback_prepared)
        _progress(
            "iqs-batch",
            "fallback 批量搜索开始",
            requests=len(fallback_requests),
            concurrency=concurrency,
            timeout_ms=fallback_timeout,
        )
        fallback_responses = call_iqs_search_batch(fallback_requests, timeout_ms=fallback_timeout, concurrency=concurrency)
        fallback_by_index = {int(item.get("index", index)): item for index, item in enumerate(fallback_responses)}
        for index, item in enumerate(fallback_prepared):
            trace = item["trace"]
            results = list(item.get("results") or [])
            response = fallback_by_index.get(index, {})
            if response.get("ok", True):
                fallback_results = [dict(value) for value in list(response.get("results") or []) if isinstance(value, dict)]
                trace["fallback_count"] = len(fallback_results)
                results.extend(fallback_results)
            else:
                error_text = _friendly_iqs_error(response.get("error") or "IQS batch fallback failed")
                trace["errors"].append(error_text)
                if _is_iqs_quota_error(error_text):
                    trace["quota_exceeded"] = True
            _extend_raw_results(raw_results, results, trace)
            search_trace.append(trace)
            for error in trace.get("errors") or []:
                errors.append(f"{trace.get('query')}: {error}")
    _progress(
        "iqs-batch",
        "批量搜索完成",
        raw_results=len(raw_results),
        traces=len(search_trace),
        errors=len(errors),
        elapsed=f"{time.perf_counter() - started:.1f}s",
    )
    return raw_results, search_trace, errors


def credibility_score(item: Dict[str, Any]) -> float:
    url = str(item.get("url") or "")
    domain = _domain(url)
    text = " ".join([domain, str(item.get("title") or ""), str(item.get("snippet") or ""), str(item.get("summary") or "")]).lower()
    score = 0.50
    if any(pattern in domain for pattern in _HIGH_CREDIBILITY_DOMAINS):
        score += 0.25
    if domain.endswith(".gov") or ".gov." in domain:
        score += 0.20
    if any(pattern.lower() in text for pattern in _LOW_CREDIBILITY_HINTS):
        score -= 0.30
    return round(max(0.0, min(1.0, score)), 4)


def lexical_relevance_score(query: str, item: Dict[str, Any]) -> float:
    normalized_query = re.sub(r"[^\w%]+", " ", str(query or "").lower())
    terms = [term for term in normalized_query.split() if len(term) >= 2]
    if not terms:
        return 0.0
    title = str(item.get("title") or "").lower()
    body = " ".join(
        str(item.get(key) or "").lower()
        for key in ["snippet", "summary", "mainText", "content", "source"]
    )
    title_hits = sum(1 for term in terms if term in title)
    body_hits = sum(1 for term in terms if term in body)
    return round(min(1.0, (0.70 * title_hits + 0.30 * body_hits) / max(len(terms), 1)), 4)


def _task_from_context(item: Dict[str, Any], options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    option_task = _as_dict(_as_dict(options).get("search_task"))
    if option_task:
        return option_task
    return _as_dict(item.get("search_task"))


_TASK_TERM_HINTS = (
    "人工智能",
    "AI",
    "大模型",
    "生成式AI",
    "AIGC",
    "中国人工智能",
    "AI行业",
    "人工智能产业",
    "算力",
    "国产算力",
    "模型成本",
    "数据合规",
    "AI安全",
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
_TASK_TERM_STOPWORDS = {
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
    "数据",
    "统计",
    "口径",
    "来源",
    "权威",
    "官方",
    "报告",
    "研究",
    "分析",
    "向好",
    "确定性",
    "短期",
    "放量",
}


def _append_task_term(terms: List[str], seen: set[str], value: Any) -> None:
    term = re.sub(r"\s+", " ", str(value or "").strip().lower())
    term = term.strip(" ,;:!?，。；：！？、()（）[]【】{}《》\"'")
    if not term or term in seen or term in _TASK_TERM_STOPWORDS:
        return
    if len(term) < 2 or len(term) > 24:
        return
    seen.add(term)
    terms.append(term)


def _expand_task_term(value: Any) -> List[str]:
    raw = re.sub(r"\s+", " ", str(value or "")).strip()
    if not raw:
        return []
    expanded: List[str] = []
    seen: set[str] = set()
    if len(raw) <= 18 and not re.search(r"[?？。；;:：]", raw):
        _append_task_term(expanded, seen, raw)
    compact = re.sub(r"\s+", "", raw.lower())
    for phrase in _TASK_TERM_HINTS:
        if phrase.lower() in compact:
            _append_task_term(expanded, seen, phrase)
    parts = re.split(
        r"[\s,;:!?，。；：！？、/\\|()\[\]{}（）【】《》\"']+|"
        r"(?:是否|哪些|怎么|如何|当前|现在|其中|对于|关于|以及|或者|并且|同时|只有|判断为|"
        r"本章|核心|问题|回答|中|里|的|与|和|及|比|更|才|可|向好|确定性|短期|放量)",
        compact,
    )
    for part in parts:
        _append_task_term(expanded, seen, part)
    return expanded


def _task_terms(value: Any) -> List[str]:
    if isinstance(value, (list, tuple, set)):
        raw_values = list(value)
    elif value is None:
        raw_values = []
    else:
        raw_values = re.split(r"[,;，；\n]+", str(value))
    terms: List[str] = []
    seen = set()
    for raw in raw_values:
        for term in _expand_task_term(raw):
            if not term or term in seen:
                continue
            seen.add(term)
            terms.append(term)
    return terms


def _contains_any_term(text: str, terms: Sequence[str]) -> bool:
    return any(term and term in text for term in terms)


def _task_field_hit(text: str, values: Sequence[Any]) -> bool:
    terms: List[str] = []
    for value in values:
        terms.extend(_task_terms(value))
    return _contains_any_term(text, [term for term in terms if len(term) >= 2])


def _term_in_text(term: str, text: str) -> bool:
    term = str(term or "").strip().lower()
    if not term:
        return False
    if re.fullmatch(r"[a-z0-9]{1,3}", term):
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text, re.I))
    return term in text


def _topic_anchor_groups(task: Dict[str, Any], options: Optional[Dict[str, Any]]) -> List[List[str]]:
    opts = _as_dict(options)
    research_plan = _as_dict(opts.get("research_plan"))
    topic_text = " ".join(
        str(value or "")
        for value in [
            opts.get("query"),
            research_plan.get("query"),
            research_plan.get("research_object"),
            task.get("research_object"),
            task.get("query"),
            " ".join(str(item) for item in _as_list(research_plan.get("global_required_terms"))),
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


def _missing_topic_anchor_groups(task: Dict[str, Any], options: Optional[Dict[str, Any]], text: str) -> List[List[str]]:
    missing: List[List[str]] = []
    for group in _topic_anchor_groups(task, options):
        if not any(_term_in_text(term, text) for term in group):
            missing.append(group)
    return missing


def task_acceptance_filter(item: Dict[str, Any], options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    task = _task_from_context(item, options)
    if not task:
        return {
            "accepted": True,
            "relevance_score": 0.0,
            "role_hint": "candidate",
            "reason": "no_task_filter",
            "matched_terms": [],
        }

    text = _result_text(item, max_chars=2000).lower()
    domain = _domain(str(item.get("url") or "")).lower()
    must_terms = _task_terms(task.get("must_have_terms"))
    forbidden_terms = _task_terms(task.get("forbidden_terms"))
    source_priority = _task_terms(task.get("source_priority"))
    lexical = lexical_relevance_score(str(task.get("query") or task.get("evidence_goal") or ""), item)
    matched_must = [term for term in must_terms if term in text]
    forbidden_hit = [term for term in forbidden_terms if term in text]
    missing_topic_groups = _missing_topic_anchor_groups(task, options, text)

    must_ratio = len(matched_must) / max(len(must_terms), 1) if must_terms else 0.0
    source_hit = bool(source_priority and any(term in text or term in domain for term in source_priority))
    goal_hit = _task_field_hit(text, [task.get("evidence_goal"), task.get("query")])
    dimension_hit = _task_field_hit(text, [task.get("dimension_name"), task.get("dimension_id")])
    has_number = bool(re.search(r"\d|%|cagr|yoy|qoq", text, re.I))

    score = 0.0
    if must_terms:
        score += 0.30 * must_ratio
        score += 0.20 * lexical
    else:
        score += 0.35 * lexical
    if source_hit:
        score += 0.15
    if goal_hit:
        score += 0.15
    if dimension_hit:
        score += 0.10
    if has_number:
        score += 0.10
    if float(item.get("credibility_score", 0.0) or 0.0) >= 0.65:
        score += 0.05
    score = min(1.0, score)

    if forbidden_hit:
        return {
            "accepted": False,
            "relevance_score": round(score, 4),
            "role_hint": "rejected",
            "reason": "forbidden_terms_hit",
            "matched_terms": matched_must,
        }
    if missing_topic_groups:
        return {
            "accepted": False,
            "relevance_score": round(min(score, 0.2), 4),
            "role_hint": "rejected",
            "reason": "topic_anchor_missing",
            "matched_terms": matched_must,
            "missing_topic_groups": missing_topic_groups,
        }
    if must_terms and not matched_must and lexical < 0.08 and not (source_hit and has_number):
        return {
            "accepted": False,
            "relevance_score": round(score, 4),
            "role_hint": "rejected",
            "reason": "must_terms_missing",
            "matched_terms": matched_must,
        }

    threshold = 0.45
    if source_priority:
        threshold = 0.40
    if has_number and (goal_hit or dimension_hit):
        threshold = 0.35
    if must_terms and not matched_must:
        threshold = max(threshold, 0.60)

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
        "role_hint": "rejected",
        "reason": "low_task_relevance_reject",
        "matched_terms": matched_must,
    }


def task_term_score(item: Dict[str, Any], options: Optional[Dict[str, Any]]) -> float:
    task = _task_from_context(item, options)
    if not task:
        return 0.0

    text = _result_text(item, max_chars=1500).lower()
    must_terms = _task_terms(task.get("must_have_terms"))
    forbidden_terms = _task_terms(task.get("forbidden_terms"))

    if forbidden_terms and any(term in text for term in forbidden_terms):
        return -1.0
    if must_terms:
        matched = sum(1 for term in must_terms if term in text)
        if not matched:
            lexical = lexical_relevance_score(str(task.get("query") or task.get("evidence_goal") or ""), item)
            return min(0.35, max(0.0, lexical * 0.5))
        return min(1.0, matched / max(len(must_terms), 1))
    return 0.0


def deduplicate_web_results(results: Sequence[Dict[str, Any]], *, threshold: float = 0.88) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    seen_urls = set()
    for item in results:
        url_key = _normalize_url(str(item.get("url") or ""))
        if url_key and url_key in seen_urls:
            continue
        text = _result_text(item, max_chars=500)
        duplicate = False
        for previous in kept:
            previous_text = _result_text(previous, max_chars=500)
            title = str(item.get("title") or "")
            previous_title = str(previous.get("title") or "")
            title_ratio = difflib.SequenceMatcher(None, title, previous_title).ratio() if title and previous_title else 0.0
            text_ratio = difflib.SequenceMatcher(None, text, previous_text).ratio() if text and previous_text else 0.0
            if text_ratio >= threshold and title_ratio >= 0.95:
                duplicate = True
                break
        if duplicate:
            continue
        if url_key:
            seen_urls.add(url_key)
        kept.append(dict(item))
    return kept


def rerank_web_results(
    query: str,
    results: Sequence[Dict[str, Any]],
    *,
    top_k: Optional[int] = None,
    options: Optional[Dict[str, Any]] = None,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    started = time.perf_counter()
    top_k = top_k or _profile_int(
        options,
        ["rerankTopK", "rerank_top_k"],
        env_name="IQS_RERANK_TOP_K",
        default=60,
        initial_env="IQS_INITIAL_RERANK_TOP_K",
        initial_default=60,
        followup_env="IQS_FOLLOWUP_RERANK_TOP_K",
        followup_default=50,
        min_value=1,
        max_value=100,
    )
    max_docs = _profile_int(
        options,
        ["rerankMaxDocs", "rerank_max_docs"],
        env_name="IQS_RERANK_MAX_DOCS",
        default=150,
        initial_env="IQS_INITIAL_RERANK_MAX_DOCS",
        initial_default=150,
        followup_env="IQS_FOLLOWUP_RERANK_MAX_DOCS",
        followup_default=100,
        min_value=1,
        max_value=200,
    )
    prefilter_docs = _profile_int(
        options,
        ["rerankPrefilterMaxDocs", "rerank_prefilter_max_docs"],
        env_name="IQS_RERANK_PREFILTER_MAX_DOCS",
        default=100,
        initial_env="IQS_INITIAL_RERANK_PREFILTER_MAX_DOCS",
        initial_default=100,
        followup_env="IQS_FOLLOWUP_RERANK_PREFILTER_MAX_DOCS",
        followup_default=100,
        min_value=1,
        max_value=100,
    )
    candidates = [dict(item) for item in results[:max_docs]]
    for index, item in enumerate(candidates):
        item["credibility_score"] = credibility_score(item)
        item["origin_rank_score"] = max(0.0, 1.0 - (index * 0.025))
        item["lexical_relevance_score"] = lexical_relevance_score(query, item)
        item["web_prefilter_score"] = round(
            (0.45 * float(item.get("credibility_score", 0.0)))
            + (0.35 * float(item.get("lexical_relevance_score", 0.0)))
            + (0.20 * float(item.get("origin_rank_score", 0.0))),
            6,
        )
    candidates.sort(
        key=lambda item: (
            float(item.get("web_prefilter_score", 0.0)),
            float(item.get("credibility_score", 0.0)),
        ),
        reverse=True,
    )
    prepared = candidates[: min(max_docs, prefilter_docs)]
    api_rerank_enabled = _env_flag("RAG_ENABLE_API_RERANK", DEFAULT_ENABLE_API_RERANK)
    meta: Dict[str, Any] = {
        "enabled": bool(api_rerank_enabled and DEFAULT_RERANK_API_KEY),
        "provider": DEFAULT_RERANK_PROVIDER if api_rerank_enabled else "",
        "model": DEFAULT_RERANK_MODEL if api_rerank_enabled else "",
        "pre_filter_input_count": len(candidates),
        "pre_filter_limit": prefilter_docs,
        "input_count": len(prepared),
        "top_k": top_k,
        "max_docs": max_docs,
    }
    if prepared and api_rerank_enabled and DEFAULT_RERANK_API_KEY:
        max_chars_per_doc = _env_int("IQS_RERANK_MAX_CHARS_PER_DOC", DEFAULT_RERANK_MAX_CHARS_PER_DOC, min_value=500, max_value=5000)
        documents = [_result_text(item, max_chars=max_chars_per_doc) for item in prepared]
        _progress(
            "iqs-rerank",
            "API 重排开始",
            input=len(documents),
            top_k=min(top_k, len(documents)),
            max_chars=max_chars_per_doc,
        )
        try:
            rerank_items = call_external_rerank_api(
                query=query,
                documents=documents,
                provider=DEFAULT_RERANK_PROVIDER,
                api_url=DEFAULT_RERANK_URL,
                api_key=DEFAULT_RERANK_API_KEY,
                model=DEFAULT_RERANK_MODEL,
                top_n=min(top_k, len(documents)),
                timeout=DEFAULT_RERANK_TIMEOUT,
            )
            for rank, rerank_item in enumerate(rerank_items, start=1):
                idx = int(rerank_item.get("index", -1))
                if 0 <= idx < len(prepared):
                    prepared[idx]["web_rerank_score"] = float(rerank_item.get("relevance_score", 0.0) or 0.0)
                    prepared[idx]["web_rerank_rank"] = rank
            meta["returned_count"] = len(rerank_items)
            _progress(
                "iqs-rerank",
                "API 重排完成",
                returned=len(rerank_items),
                elapsed=f"{time.perf_counter() - started:.1f}s",
            )
        except Exception as exc:
            meta["error"] = str(exc)
            _progress("iqs-rerank", "API 重排失败，改用本地分数", error=exc)
    for item in prepared:
        rerank_score = float(item.get("web_rerank_score", 0.0) or 0.0)
        credibility = float(item.get("credibility_score", 0.0) or 0.0)
        lexical = float(item.get("lexical_relevance_score", 0.0) or 0.0)
        origin = float(item.get("origin_rank_score", 0.0) or 0.0)
        task_score = task_term_score(item, options)
        item["task_term_score"] = round(task_score, 4)
        if task_score < 0:
            item["web_final_score"] = -1
        else:
            item["web_final_score"] = round(
                (0.40 * rerank_score)
                + (0.20 * credibility)
                + (0.15 * lexical)
                + (0.15 * task_score)
                + (0.10 * origin),
                6,
            )
    prepared.sort(key=lambda item: (float(item.get("web_final_score", 0.0)), float(item.get("credibility_score", 0.0))), reverse=True)
    selected = [
        item
        for item in prepared
        if float(item.get("credibility_score", 0.0)) >= 0.35
        and float(item.get("web_final_score", 0.0)) >= 0.25
    ][:top_k]
    meta["task_score_rejected_count"] = sum(1 for item in prepared if float(item.get("task_term_score", 0.0)) < 0)
    meta["output_count"] = len(selected)
    _progress(
        "iqs-rerank",
        "重排完成",
        input=len(results),
        prepared=len(prepared),
        selected=len(selected),
        rejected=meta["task_score_rejected_count"],
        elapsed=f"{time.perf_counter() - started:.1f}s",
    )
    return selected, meta


def process_web_results(
    query: str,
    raw_results: Sequence[Dict[str, Any]],
    *,
    options: Optional[Dict[str, Any]] = None,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    task_filter_reasons: Dict[str, int] = {}
    for index, raw in enumerate(raw_results):
        if not isinstance(raw, dict):
            continue
        item = dict(raw)
        item["raw_rank"] = index + 1
        item["credibility_score"] = credibility_score(item)
        acceptance = task_acceptance_filter(item, options)
        item["task_filter"] = acceptance
        item["task_relevance_score"] = acceptance.get("relevance_score", 0.0)
        if not acceptance.get("accepted", False):
            reason = str(acceptance.get("reason") or "task_rejected")
            task_filter_reasons[reason] = task_filter_reasons.get(reason, 0) + 1
            continue
        normalized.append(item)
    deduped = deduplicate_web_results(normalized, threshold=float(os.getenv("IQS_DEDUP_THRESHOLD", "0.88")))
    reranked, rerank_meta = rerank_web_results(query, deduped, options=options)
    return reranked, {
        "raw_count": len(raw_results),
        "normalized_count": len(normalized),
        "task_filtered_count": sum(task_filter_reasons.values()),
        "task_filter_reasons": task_filter_reasons,
        "deduped_count": len(deduped),
        "rerank": rerank_meta,
    }


def process_web_results_with_top_k(
    query: str,
    raw_results: Sequence[Dict[str, Any]],
    *,
    top_k: int,
    options: Optional[Dict[str, Any]] = None,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    task_filter_reasons: Dict[str, int] = {}
    for index, raw in enumerate(raw_results):
        if not isinstance(raw, dict):
            continue
        item = dict(raw)
        item["raw_rank"] = index + 1
        item["credibility_score"] = credibility_score(item)
        acceptance = task_acceptance_filter(item, options)
        item["task_filter"] = acceptance
        item["task_relevance_score"] = acceptance.get("relevance_score", 0.0)
        if not acceptance.get("accepted", False):
            reason = str(acceptance.get("reason") or "task_rejected")
            task_filter_reasons[reason] = task_filter_reasons.get(reason, 0) + 1
            continue
        normalized.append(item)
    deduped = deduplicate_web_results(normalized, threshold=float(os.getenv("IQS_DEDUP_THRESHOLD", "0.88")))
    reranked, rerank_meta = rerank_web_results(query, deduped, top_k=top_k, options=options)
    return reranked, {
        "raw_count": len(raw_results),
        "normalized_count": len(normalized),
        "task_filtered_count": sum(task_filter_reasons.values()),
        "task_filter_reasons": task_filter_reasons,
        "deduped_count": len(deduped),
        "rerank": rerank_meta,
    }


def run_iqs_optimized_search(query: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    started = time.perf_counter()
    incoming_options = dict(options or {})
    base_options = {**incoming_options, **infer_search_options(query, incoming_options)}
    cache_key = ""
    if _iqs_search_cache_allowed(base_options):
        cache_key = _iqs_search_cache_key(query, base_options)
        cached_payload = _IQS_SEARCH_CACHE.get(cache_key)
        if cached_payload:
            _progress("iqs-search", "命中 IQS 搜索缓存", query=query)
            cached_copy = copy.deepcopy(cached_payload)
            cached_copy["cache"] = {**_as_dict(cached_copy.get("cache")), "enabled": True, "hit": True}
            return cached_copy
    query_plan = build_query_plan(query, base_options)
    search_tasks: List[Dict[str, Any]] = []
    seen_task_keys: set[tuple[Any, ...]] = set()
    duplicate_task_count = 0
    max_tasks = _profile_int(
        base_options,
        ["maxSearchTasks", "max_search_tasks"],
        env_name="IQS_MAX_SEARCH_TASKS",
        default=36,
        initial_env="IQS_INITIAL_MAX_SEARCH_TASKS",
        initial_default=36,
        followup_env="IQS_FOLLOWUP_MAX_SEARCH_TASKS",
        followup_default=18,
        min_value=1,
        max_value=80,
    )
    for query_item in query_plan:
        raw_engine_types = query_item.get("engineTypes")
        if isinstance(raw_engine_types, list):
            engine_types = _split_csv(",".join(str(item) for item in raw_engine_types))
        else:
            engine_types = _split_csv(raw_engine_types)
        engine_types = engine_types or [str(query_item.get("engineType") or "LiteAdvanced")]
        for engine_type in engine_types:
            task_item = dict(query_item)
            task_item["engineType"] = engine_type
            task_item["engineTypes"] = [engine_type]
            task_key = _search_task_dedupe_key(task_item)
            if task_key in seen_task_keys:
                duplicate_task_count += 1
                continue
            seen_task_keys.add(task_key)
            search_tasks.append(task_item)
            if len(search_tasks) >= max_tasks:
                break
        if len(search_tasks) >= max_tasks:
            break
    raw_results: List[Dict[str, Any]] = []
    search_trace: List[Dict[str, Any]] = []
    errors: List[str] = []
    _progress(
        "iqs-search",
        "Query Plan 完成",
        plan_items=len(query_plan),
        search_tasks=len(search_tasks),
        duplicates=duplicate_task_count,
        max_tasks=max_tasks,
        batch=_option_flag(base_options, ["enableBatchSearch", "enable_batch_search"], "IQS_ENABLE_BATCH_SEARCH", True),
        query=query,
    )
    if search_tasks:
        if _option_flag(base_options, ["enableBatchSearch", "enable_batch_search"], "IQS_ENABLE_BATCH_SEARCH", True):
            try:
                raw_results, search_trace, errors = _run_iqs_search_tasks_batch(search_tasks, base_options)
                processed_results, quality_meta = process_web_results(query, raw_results, options=base_options)
                _progress(
                    "iqs-search",
                    "搜索处理完成",
                    raw_results=len(raw_results),
                    processed=len(processed_results),
                    errors=len(errors),
                    elapsed=f"{time.perf_counter() - started:.1f}s",
                )
                payload = {
                    "query": query,
                    "search_options": base_options,
                    "query_plan": query_plan,
                    "search_tasks": search_tasks,
                    "search_trace": search_trace,
                    "quality_processing": quality_meta,
                    "results": processed_results,
                    "errors": errors,
                    "cache": {
                        "enabled": bool(cache_key),
                        "hit": False,
                        "deduped_duplicate_search_tasks": duplicate_task_count,
                    },
                }
                if cache_key and not errors:
                    _IQS_SEARCH_CACHE.set(cache_key, copy.deepcopy(payload))
                return payload
            except Exception as exc:
                errors.append(f"IQS batch search failed, fallback to parallel subprocess: {exc}")
                _progress("iqs-search", "批量搜索失败，切换并发子进程", error=exc)
        max_workers = min(len(search_tasks), _env_int("IQS_PARALLEL_WORKERS", 5, min_value=1, max_value=12))
        _progress("iqs-search", "并发子进程搜索开始", tasks=len(search_tasks), workers=max_workers)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(call_iqs_search_with_fallback, item, base_options): item for item in search_tasks}
            for future in as_completed(futures):
                try:
                    results, trace = future.result()
                    search_trace.append(trace)
                    for rank, result in enumerate(results, start=1):
                        copied = dict(result)
                        copied["origin_query"] = trace.get("query")
                        copied["origin_intent"] = trace.get("intent")
                        copied["origin_query_source"] = trace.get("source")
                        copied["origin_rank"] = rank
                        for key in ["task_id", "dimension_id", "dimension_name", "evidence_goal", "must_have_terms", "forbidden_terms", "source_priority", "hypothesis_id", "hypothesis_statement", "proof_role", "proof_standard", "evidence_type", "lane_targets", "scheduled_lane_type", "scheduled_lane", "counter_evidence", "decision_use", "search_task"]:
                            if key in trace:
                                copied[key] = trace.get(key)
                        raw_results.append(copied)
                    for item in trace.get("errors") or []:
                        errors.append(f"{trace.get('query')}：{item}")
                except Exception as exc:
                    item = futures[future]
                    errors.append(f"{item.get('text')}：{exc}")
    processed_results, quality_meta = process_web_results(query, raw_results, options=base_options)
    _progress(
        "iqs-search",
        "搜索处理完成",
        raw_results=len(raw_results),
        processed=len(processed_results),
        errors=len(errors),
        elapsed=f"{time.perf_counter() - started:.1f}s",
    )
    payload = {
        "query": query,
        "search_options": base_options,
        "query_plan": query_plan,
        "search_tasks": search_tasks,
        "search_trace": search_trace,
        "quality_processing": quality_meta,
        "results": processed_results,
        "errors": errors,
        "cache": {
            "enabled": bool(cache_key),
            "hit": False,
            "deduped_duplicate_search_tasks": duplicate_task_count,
        },
    }
    if cache_key and not errors:
        _IQS_SEARCH_CACHE.set(cache_key, copy.deepcopy(payload))
    return payload


def assign_source_ids(results: Sequence[Dict[str, Any]], pages: Sequence[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    search_items: List[Dict[str, Any]] = []
    for index, item in enumerate(results):
        copied = dict(item)
        copied["source_id"] = index
        search_items.append(copied)
    page_items: List[Dict[str, Any]] = []
    offset = len(search_items)
    for index, item in enumerate(pages):
        copied = dict(item)
        copied["source_id"] = offset + index
        page_items.append(copied)
    return search_items, page_items


def build_llm_config() -> Dict[str, Any]:
    return {
        "provider": DEFAULT_LLM_SYNTHESIS_PROVIDER,
        "url": DEFAULT_LLM_SYNTHESIS_URL,
        "api_key": DEFAULT_LLM_SYNTHESIS_API_KEY,
        "model": DEFAULT_LLM_SYNTHESIS_MODEL,
        "timeout": DEFAULT_LLM_SYNTHESIS_TIMEOUT,
        "disable_thinking": DEFAULT_LLM_SYNTHESIS_DISABLE_THINKING,
    }


def build_web_payload(
    *,
    query: str,
    search_results: Sequence[Dict[str, Any]],
    page_results: Sequence[Dict[str, Any]],
    search_options: Dict[str, Any],
) -> Dict[str, Any]:
    sources = []
    for item in search_results:
        sources.append(
            {
                "id": item.get("source_id"),
                "round": item.get("research_round") or item.get("round") or "initial",
                "origin_query": item.get("origin_query") or item.get("follow_up_query"),
                "title": item.get("title"),
                "url": item.get("url"),
                "date": item.get("publishedTime"),
                "snippet": _compact_text(item.get("mainText") or item.get("snippet") or item.get("summary"), 1800),
                "relevance": item.get("relevance"),
            }
        )
    for item in page_results:
        sources.append(
            {
                "id": item.get("source_id"),
                "round": item.get("research_round") or item.get("round") or "page_read",
                "origin_query": item.get("origin_query") or item.get("url"),
                "title": item.get("title"),
                "url": item.get("url"),
                "date": "",
                "snippet": _compact_text(item.get("content"), 2200),
                "relevance": "",
            }
        )
    return {
        "query": query,
        "sources": sources,
    }


def render_structured_web_answer(payload: Dict[str, Any]) -> str:
    answer_payload = payload.get("answer") if isinstance(payload.get("answer"), dict) else {}
    conclusion = str(answer_payload.get("conclusion") or "").strip()
    evidence = str(answer_payload.get("evidence") or "").strip()
    inference = answer_payload.get("inference")
    evidence_gap = answer_payload.get("evidence_gap")
    limitations = payload.get("limitations") if isinstance(payload.get("limitations"), dict) else {}
    next_steps = payload.get("next_steps") if isinstance(payload.get("next_steps"), list) else []

    lines: List[str] = []
    if conclusion:
        lines.append(f"核心判断：{conclusion}")
    if evidence:
        lines.extend(["", "关键依据：", evidence])
    if inference:
        inference_text = str(inference).strip()
        if inference_text:
            if not inference_text.startswith("推断："):
                inference_text = "推断：" + inference_text
            lines.extend(["", inference_text])
    if isinstance(evidence_gap, list) and evidence_gap:
        lines.extend(["", "证据缺口："])
        for item in evidence_gap:
            cleaned = str(item or "").strip()
            if cleaned:
                lines.append(f"- {cleaned}")
    if limitations:
        lines.extend(["", "限制说明："])
        for label, key in [("数据时效", "data_recency"), ("覆盖范围", "coverage"), ("来源矛盾", "conflicts")]:
            value = limitations.get(key)
            if value:
                lines.append(f"- {label}：{value}")
    if next_steps:
        lines.extend(["", "建议补充："])
        for item in next_steps:
            cleaned = str(item or "").strip()
            if cleaned:
                lines.append(f"- {cleaned}")
    return "\n".join(lines).strip()


def synthesize_with_llm(
    *,
    query: str,
    search_results: Sequence[Dict[str, Any]],
    page_results: Sequence[Dict[str, Any]],
    search_options: Dict[str, Any],
) -> tuple[str, Dict[str, Any]]:
    llm_config = build_llm_config()
    if not llm_config_is_ready(llm_config):
        raise RuntimeError("大模型综合分析配置不完整。")

    system_prompt = """
## 角色定位
你是行业研究多智能体系统中的【联网分析子智能体】。
- 上游：接收 Supervisor 传入的结构化 query 和 IQS 检索结果（网页摘要列表）
- 下游：你的 JSON 输出将被 Writer Agent 用于报告撰写、被 QA Agent 用于事实核验
- 原则：宁可输出"证据不足"，不可捏造事实

## 输入格式
IQS 会传入如下结构，sources[i] 的 index 即引用编号：
{
  "query": "...",
  "sources": [
    {"id": 0, "title": "...", "url": "...", "date": "...", "snippet": "..."},
    {"id": 1, ...}
  ]
}

## 分析要求
1. 核心判断优先：先给一句话结论，再展开依据，每条事实必须带 [id] 引用（id 对应 sources 数组下标）
2. 三类信息严格区分：
   - 【事实】有来源支撑的陈述，必须带引用
   - 【推断】基于证据的合理推论，需标注"推断："前缀
   - 【证据缺口】检索结果未覆盖的重要维度，必须列出
3. 数据优先级：优先抽取可直接支撑行研报告的具体数字数据，包括市场规模、增速/CAGR、出货量/销量/产量、市占率/份额、营收、净利润、毛利率、亏损、现金流、融资金额、估值、股价、市值、政策金额或目标年份。
4. 数字保真：凡是 sources 中出现的数字，必须尽量保留原始单位、时间范围和统计口径；不得自行换算、四舍五入或合并不同口径。
5. 趋势表达：趋势判断必须基于数字或明确事件支撑，写入 inference；没有数字支撑时只能列为证据缺口。
6. 关键依据格式：evidence 中优先输出【数字数据】、【发展趋势】、【竞争对比】、【政策监管】、【技术产业链】、【资本事件】等标签；每条事实必须是“可直接写进行研正文”的完整素材，不要只给孤立数字。
   推荐格式：
   “【数字数据】指标/事件：数值或变化（时间/统计口径）；背景：这条数据衡量什么市场/主体/环节；解释：它说明需求、竞争、成本、政策或资本的什么变化；报告用途：可用于哪类小节或判断。[id]”
   硬规则：evidence 的每一行末尾至少保留一个来源编号，格式可以是 [id:11] 或 [11]；无法确定来源编号的内容不要写入 evidence，改写入 evidence_gap。
7. 单条素材长度：每条 evidence 尽量写 80-180 个中文字符。必须保留来源原始数字、单位、时间范围、统计口径、主体名称和场景，避免“市场规模：100亿元”这种过短表达。
8. 时效性标注：股价、行情、政策、财报类数据，必须在 limitations 中说明数据日期范围。
9. 内容完整度：如果来源足够，evidence 至少写 14-20 条事实，尽量覆盖“数据现状、变化原因、主体/企业、应用场景、产业链环节、风险缺口”，不要只写两三条概括性结论。
10. 来源展示：key_sources 尽量保留 15-25 条最相关来源，优先选官方、财报、头部媒体和原始公告。
11. 置信度降级规则（必须遵守）：
   - sources 数量 = 0 → confidence = 0.0，answer 说明无法回答
   - sources 数量 = 1 → confidence ≤ 0.4
   - 最新 source 发布超过 180 天 → confidence 自动 -0.1
   - 来源为官方政府/头部媒体 → confidence +0.1（上限 1.0）
12. 来源矛盾处理：若多个来源数据相互矛盾，在 answer 中明确列出分歧，不得自行裁定

## 输出格式（严格 JSON，不输出任何其他内容）
{
  "answer": {
    "conclusion": "一句话核心判断（≤200字）",
    "evidence": "多条可直接写作的长素材，每条带标签和 [id:来源编号] 或 [来源编号] 引用，用 \\n 分隔；每条包含数据/事件、背景、解释、报告用途，尽量不少于 8 条",
    "inference": "基于数字和事件证据的发展趋势推断（若无则填 null）",
    "evidence_gap": ["缺口1", "缺口2"]
  },
  "confidence": 0.0,
  "key_sources": [
    {
      "id": 0,
      "title": "...",
      "url": "...",
      "date": "...",
      "relevance": "high|medium|low"
    }
  ],
  "limitations": {
    "data_recency": "数据时效说明，如：最新数据截至2025Q3",
    "coverage": "来源覆盖范围说明，尽量写清覆盖了哪些维度和哪些来源类型",
    "conflicts": "来源间矛盾说明（若无则填 null）"
  },
  "next_steps": ["建议补充来源1", "建议补充来源2"]
}
""".strip()
    user_payload = build_web_payload(
        query=query,
        search_results=search_results,
        page_results=page_results,
        search_options=search_options,
    )
    response = call_openai_compatible_json(
        config=llm_config,
        system_prompt=system_prompt,
        user_payload=user_payload,
    )
    payload = response.get("payload", {})
    answer = render_structured_web_answer(payload)
    if not answer:
        raise RuntimeError("大模型联网分析回答为空。")
    return answer, {
        "type": "web_analysis_synthesis",
        "source": "llm",
        "model": normalize_llm_config(llm_config).get("model", ""),
        "usage": response.get("usage", {}),
        "structured_payload": payload,
        "confidence": payload.get("confidence"),
        "key_sources": payload.get("key_sources", []),
        "limitations": payload.get("limitations", {}),
        "next_steps": payload.get("next_steps", []),
    }


def _text_items(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        parts = []
        for key in ["dimension", "reason", "suggestion", "query", "source", "metric", "title"]:
            if value.get(key):
                parts.append(str(value.get(key)))
        if not parts:
            parts = [str(item) for item in value.values() if item]
        return [" ".join(parts)] if parts else []
    if isinstance(value, (list, tuple)):
        items: List[str] = []
        for item in value:
            items.extend(_text_items(item))
        return items
    return [str(value)]


def build_follow_up_queries_from_payload(
    *,
    query: str,
    payload: Dict[str, Any],
    max_queries: int,
) -> List[str]:
    answer_payload = payload.get("answer") if isinstance(payload.get("answer"), dict) else {}
    raw_items: List[str] = []
    raw_items.extend(_text_items(answer_payload.get("evidence_gap")))
    raw_items.extend(_text_items(payload.get("next_steps")))
    limitations = payload.get("limitations") if isinstance(payload.get("limitations"), dict) else {}
    raw_items.extend(_text_items(limitations.get("coverage")))

    year = datetime.now().year
    seen = set()
    queries: List[str] = []
    generic_patterns = re.compile(r"^(缺口\d+|建议补充来源\d+|null|none|无)$", re.I)
    for item in raw_items:
        item_text = re.sub(r"\s+", " ", str(item or "").strip())
        if not item_text or generic_patterns.match(item_text):
            continue
        search_text = _clean_search_query(f"{query} {item_text} {year} 具体数据 权威来源", max_chars=120)
        key = search_text.lower()
        if not search_text or key in seen:
            continue
        seen.add(key)
        queries.append(search_text)
        if len(queries) >= max(1, max_queries):
            break
    return queries


def run_follow_up_iqs_searches(
    *,
    original_query: str,
    follow_up_queries: Sequence[str],
    search_options: Dict[str, Any],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    results: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []
    errors: List[str] = []
    queries = [str(item or "").strip() for item in follow_up_queries if str(item or "").strip()]
    if not queries:
        return results, traces, errors

    max_workers = min(len(queries), _env_int("IQS_SELF_REFINE_PARALLEL_WORKERS", 3, min_value=1, max_value=8))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(run_iqs_optimized_search, query, search_options): query for query in queries}
        for future in as_completed(future_map):
            follow_query = future_map[future]
            try:
                search_state = future.result()
                traces.append(
                    {
                        "query": follow_query,
                        "result_count": len(list(search_state.get("results") or [])),
                        "query_plan": search_state.get("query_plan", []),
                        "quality_processing": search_state.get("quality_processing", {}),
                        "errors": list(search_state.get("errors") or []),
                    }
                )
                for item in list(search_state.get("results") or []):
                    if not isinstance(item, dict):
                        continue
                    copied = dict(item)
                    copied["research_round"] = "follow_up"
                    copied["follow_up_query"] = follow_query
                    copied.setdefault("origin_query", follow_query)
                    results.append(copied)
                errors.extend(f"{follow_query}：{item}" for item in list(search_state.get("errors") or []))
            except Exception as exc:
                errors.append(f"{follow_query}：{exc}")
                traces.append({"query": follow_query, "result_count": 0, "errors": [str(exc)]})
    return results, traces, errors


def synthesize_refined_with_llm(
    *,
    query: str,
    initial_answer: str,
    initial_payload: Dict[str, Any],
    follow_up_queries: Sequence[str],
    search_results: Sequence[Dict[str, Any]],
    page_results: Sequence[Dict[str, Any]],
    search_options: Dict[str, Any],
) -> tuple[str, Dict[str, Any]]:
    llm_config = build_llm_config()
    if not llm_config_is_ready(llm_config):
        raise RuntimeError("大模型综合分析配置不完整。")

    system_prompt = """
## 角色定位
你是行业研究多智能体系统中的【联网分析子智能体】。
你已经完成第一轮 IQS 分析，并根据证据缺口完成第二轮补充检索。

## 任务
将第一轮结论、补充检索问题、两轮来源证据合并为一个最终 JSON。

## 合并要求
1. 最终结论必须以两轮证据共同支撑；若第二轮补齐了缺口，要更新 conclusion/evidence/evidence_gap。
2. 所有事实必须引用当前 sources 的 [id]，不得沿用 first_analysis 里的旧引用。
3. 优先保留具体数字：市场规模、增速/CAGR、出货量、销量、市占率、营收、净利润、毛利率、亏损、现金流、融资额、估值、股价、市值。
4. 若第一轮与第二轮来源存在口径冲突，必须在 limitations.conflicts 中说明，不得自行裁定。
5. evidence 中按【数字数据】、【发展趋势】、【竞争对比】、【政策监管】、【技术产业链】、【资本事件】组织，每条事实一行；每条都要写清背景、统计/事件口径、解释含义和报告用途，避免只列数字。
6. 如果补充检索后仍缺少关键数据，在 evidence_gap 和 next_steps 中保留，但不要输出泛泛的“建议补充来源1”。
7. evidence 的每一行末尾至少保留一个来源编号，格式可以是 [id:11] 或 [11]；无法确定来源编号的内容不要写入 evidence，改写入 evidence_gap。
8. 如果两轮材料足够，evidence 至少保留 14-20 条事实，单条尽量 80-180 个中文字符，key_sources 尽量保留 15-25 条。

## 输出格式（严格 JSON，不输出任何其他内容）
{
  "answer": {
    "conclusion": "一句话核心判断（≤200字）",
    "evidence": "多条可直接写作的长素材，每条带标签和 [id:来源编号] 或 [来源编号] 引用，用 \\n 分隔；每条包含数据/事件、背景、解释、报告用途，尽量不少于 8 条",
    "inference": "基于数字和事件证据的发展趋势推断（若无则填 null）",
    "evidence_gap": ["缺口1", "缺口2"]
  },
  "confidence": 0.0,
  "key_sources": [
    {"id": 0, "title": "...", "url": "...", "date": "...", "relevance": "high|medium|low"}
  ],
  "limitations": {
    "data_recency": "数据时效说明",
    "coverage": "两轮检索覆盖范围说明，尽量写清覆盖了哪些维度和哪些来源类型",
    "conflicts": "来源间矛盾说明（若无则填 null）"
  },
  "next_steps": ["仍需补充的具体来源或数据口径"]
}
""".strip()

    user_payload = build_web_payload(
        query=query,
        search_results=search_results,
        page_results=page_results,
        search_options=search_options,
    )
    user_payload["first_analysis"] = {
        "answer_text": _compact_text(initial_answer, 3000),
        "structured_payload": initial_payload,
    }
    user_payload["follow_up_queries"] = list(follow_up_queries)
    response = call_openai_compatible_json(
        config=llm_config,
        system_prompt=system_prompt,
        user_payload=user_payload,
    )
    payload = response.get("payload", {})
    answer = render_structured_web_answer(payload)
    if not answer:
        raise RuntimeError("大模型二次合并回答为空。")
    return answer, {
        "type": "web_analysis_refined_synthesis",
        "source": "llm_self_refine",
        "model": normalize_llm_config(llm_config).get("model", ""),
        "usage": response.get("usage", {}),
        "structured_payload": payload,
        "confidence": payload.get("confidence"),
        "key_sources": payload.get("key_sources", []),
        "limitations": payload.get("limitations", {}),
        "next_steps": payload.get("next_steps", []),
        "follow_up_queries": list(follow_up_queries),
    }


def format_sources(search_results: Sequence[Dict[str, Any]], page_results: Sequence[Dict[str, Any]], *, max_items: int = 30) -> str:
    lines = ["本次使用的联网来源："]
    count = 0
    for item in search_results:
        if count >= max_items:
            break
        count += 1
        source_id = item.get("source_id")
        if source_id is None:
            source_id = count - 1
        title = str(item.get("title") or "Untitled").strip()
        url = str(item.get("url") or "").strip()
        published = str(item.get("publishedTime") or "").strip()
        snippet = _compact_text(item.get("mainText") or item.get("snippet") or item.get("summary"), 500)
        lines.append(f"{count}. [{source_id}] {title}")
        if url:
            lines.append(f"   URL：{url}")
        if published:
            lines.append(f"   时间：{published}")
        if snippet:
            lines.append(f"   摘要：{snippet}")
    for item in page_results:
        if count >= max_items:
            break
        count += 1
        source_id = item.get("source_id")
        if source_id is None:
            source_id = count - 1
        title = str(item.get("title") or "Read page").strip()
        url = str(item.get("url") or "").strip()
        content = _compact_text(item.get("content"), 650)
        lines.append(f"{count}. [{source_id}] {title}")
        if url:
            lines.append(f"   URL：{url}")
        if content:
            lines.append(f"   摘要：{content}")
    return "\n".join(lines)


def _fallback_evidence_tag(text: str, query: str = "") -> str:
    haystack = f"{query} {text}"
    if re.search(r"\d+(?:\.\d+)?\s*(?:%|pct|亿元|万亿元|亿美元|万元|亿|万台|台|套|件|家|倍|元|美元)", haystack, re.I):
        return "数字数据"
    if re.search(r"(市占率|份额|竞争|头部|排名|企业|公司|玩家|对比)", haystack, re.I):
        return "竞争对比"
    if re.search(r"(政策|监管|补贴|标准|规划|通知|意见|办法|政府)", haystack, re.I):
        return "政策监管"
    if re.search(r"(技术|产业链|上游|下游|零部件|成本|路线|商业化|国产替代)", haystack, re.I):
        return "技术产业链"
    if re.search(r"(融资|IPO|并购|估值|市值|股价|财报|营收|利润|亏损)", haystack, re.I):
        return "资本事件"
    if re.search(r"(趋势|增长|提升|下降|加速|放缓|爆发|转向)", haystack, re.I):
        return "发展趋势"
    return "事实"


def _fallback_report_use(tag: str) -> str:
    mapping = {
        "数字数据": "可用于对应证据目标的关键事实和口径说明",
        "竞争对比": "可用于供给结构、参与者对比或替代方案分析",
        "政策监管": "可用于政策、监管或执行机制相关证据目标",
        "技术产业链": "可用于技术、交付、成本或供应约束相关证据目标",
        "资本事件": "可用于财务、交易事件或风险验证相关证据目标",
        "发展趋势": "可用于趋势判断和章节结论",
    }
    return mapping.get(tag, "可用于背景事实、章节导语或证据补充")


def _format_fallback_material_line(
    *,
    title: str,
    body: str,
    source_id: Any,
    query: str,
    date: str = "",
) -> str:
    body = _compact_text(body, 520)
    tag = _fallback_evidence_tag(f"{title} {body}", query)
    date_text = f"（{date}）" if date else ""
    report_use = _fallback_report_use(tag)
    return (
        f"【{tag}】{title}{date_text}：{body}；"
        f"内容说明：该来源补充了“{query}”相关的事实背景、主体或指标口径；"
        f"报告用途：{report_use}。 [{source_id}]"
    )


def build_fallback_answer(
    *,
    query: str,
    search_results: Sequence[Dict[str, Any]],
    page_results: Sequence[Dict[str, Any]],
    search_options: Dict[str, Any],
    llm_error: str = "",
) -> str:
    if not search_results and not page_results:
        return "未获取到可用公开网页结果，暂不形成联网侧判断。"

    lines = [
        "可核验网页线索如下，正文判断应以来源质量、时间窗口和同口径交叉验证为准。",
        "",
        "关键依据：",
    ]
    for index, item in enumerate(list(search_results)[:8], start=1):
        source_id = item.get("source_id")
        if source_id is None:
            source_id = index - 1
        title = str(item.get("title") or "Untitled").strip()
        snippet = _compact_text(item.get("mainText") or item.get("snippet") or item.get("summary"), 520)
        published = str(item.get("publishedTime") or "").strip()
        if snippet:
            lines.append(
                f"{index}. "
                + _format_fallback_material_line(
                    title=title,
                    body=snippet,
                    source_id=source_id,
                    query=query,
                    date=published,
                )
            )
        else:
            lines.append(f"{index}. 【事实】{title}：该来源标题与检索主题相关，但摘要不足，正文不单独采用该条判断。 [{source_id}]")
    offset = len(list(search_results)[:8])
    for index, item in enumerate(list(page_results)[:5], start=1):
        source_id = item.get("source_id")
        if source_id is None:
            source_id = offset + index - 1
        title = str(item.get("title") or "Read page").strip()
        content = _compact_text(item.get("content"), 520)
        lines.append(
            f"{offset + index}. "
            + _format_fallback_material_line(
                title=title,
                body=content,
                source_id=source_id,
                query=query,
            )
        )

    lines.extend(
        [
            "",
            f"检索口径：engineType={search_options.get('engineType')}, timeRange={search_options.get('timeRange')}, contents={search_options.get('contents')}",
        ]
    )
    if llm_error:
        lines.append(f"分析降级原因：{llm_error}")
    lines.extend(["", format_sources(search_results, page_results)])
    return "\n".join(lines).strip()


def prepare_query_node(state: WebAnalysisAgentState) -> WebAnalysisAgentState:
    query = extract_query_from_state(state)
    urls = list(state.get("urls") or [])
    urls.extend(extract_urls(query))
    seen_urls = []
    seen = set()
    for url in urls:
        cleaned = str(url or "").strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            seen_urls.append(cleaned)
    if not query and not seen_urls:
        return {
            "errors": list(state.get("errors") or []) + ["查询或 URL 不能为空"],
            "metadata": {
                **dict(state.get("metadata") or {}),
                "agent_name": AGENT_NAME,
                "framework": "langgraph",
                "agent_stage": "prepare_query",
            },
        }
    incoming_options = dict(state.get("search_options") or {})
    return {
        "query": query,
        "urls": seen_urls,
        "search_options": {**incoming_options, **infer_search_options(query, incoming_options)},
        "metadata": {
            **dict(state.get("metadata") or {}),
            "agent_name": AGENT_NAME,
            "agent_description": AGENT_DESCRIPTION,
            "framework": "langgraph",
            "provider": "aliyun_iqs_skills",
            "capabilities": [
                "web_search",
                "page_reading",
                "current_information_analysis",
                "fact_verification",
            ],
            "agent_stage": "prepare_query",
        },
    }


def iqs_research_node(state: WebAnalysisAgentState) -> WebAnalysisAgentState:
    if state.get("errors"):
        return {}
    if not iqs_api_key_is_configured():
        return {
            "errors": list(state.get("errors") or [])
            + [
                "ALIYUN_IQS_API_KEY 未配置。请把它写入项目根目录的 .env。"
            ],
            "metadata": {
                **dict(state.get("metadata") or {}),
                "agent_stage": "iqs_research",
            },
        }

    query = str(state.get("query") or "").strip()
    urls = list(state.get("urls") or [])
    search_options = dict(state.get("search_options") or infer_search_options(query))
    search_results: List[Dict[str, Any]] = []
    page_results: List[Dict[str, Any]] = []
    errors: List[str] = []
    optimized_search: Dict[str, Any] = {}
    started = time.perf_counter()
    _progress("web-agent", "IQS 检索阶段开始", query=query, url_count=len(urls))

    if query:
        try:
            optimized_search = run_iqs_optimized_search(query, search_options)
            search_results = list(optimized_search.get("results") or [])
            for item in search_results:
                if isinstance(item, dict):
                    item.setdefault("research_round", "initial")
                    item.setdefault("origin_query", query)
            errors.extend(str(item) for item in optimized_search.get("errors") or [])
        except Exception as exc:
            logger.exception("IQS optimized search failed", extra={"query": query})
            errors.append(f"搜索失败：{exc}")
    for url in urls[:3]:
        try:
            _progress("web-agent", "读取指定网页", url=url)
            page_results.append(call_iqs_readpage(url, timeout_ms=60000))
        except Exception as exc:
            logger.exception("IQS readpage failed", extra={"url": url})
            errors.append(f"读取网页失败（{url}）：{exc}")

    search_results, page_results = assign_source_ids(search_results, page_results)
    output: WebAnalysisAgentState = {
        "search_results": search_results,
        "page_results": page_results,
        "metadata": {
            **dict(state.get("metadata") or {}),
            "agent_stage": "iqs_research",
            "result_count": len(search_results),
            "page_count": len(page_results),
            "query_plan": optimized_search.get("query_plan", []),
            "search_tasks": optimized_search.get("search_tasks", []),
            "search_trace": optimized_search.get("search_trace", []),
            "quality_processing": optimized_search.get("quality_processing", {}),
        },
    }
    if errors and not search_results and not page_results:
        output["errors"] = list(state.get("errors") or []) + errors
    elif errors:
        output["metadata"] = {**dict(output["metadata"]), "partial_errors": errors}
    _progress(
        "web-agent",
        "IQS 检索阶段完成",
        search_results=len(search_results),
        page_results=len(page_results),
        errors=len(errors),
        elapsed=f"{time.perf_counter() - started:.1f}s",
    )
    return output


def synthesize_analysis_node(state: WebAnalysisAgentState) -> WebAnalysisAgentState:
    if state.get("errors"):
        return {}
    started = time.perf_counter()
    query = str(state.get("query") or "").strip()
    search_results = list(state.get("search_results") or [])
    page_results = list(state.get("page_results") or [])
    search_options = dict(state.get("search_options") or {})
    enable_llm = bool(state.get("enable_llm_analysis", True))

    llm_meta: Dict[str, Any] = {"type": "web_analysis_synthesis", "source": "fallback_extractive"}
    llm_error = ""
    _progress(
        "web-agent",
        "联网分析综合开始",
        search_results=len(search_results),
        page_results=len(page_results),
        llm=enable_llm,
    )
    if enable_llm and (search_results or page_results):
        try:
            answer_text, llm_meta = synthesize_with_llm(
                query=query,
                search_results=search_results,
                page_results=page_results,
                search_options=search_options,
            )
        except Exception as exc:
            logger.exception("LLM web synthesis failed", extra={"query": query})
            llm_error = str(exc)
            answer_text = build_fallback_answer(
                query=query,
                search_results=search_results,
                page_results=page_results,
                search_options=search_options,
                llm_error=llm_error,
            )
            llm_meta = {**llm_meta, "error": llm_error}
    else:
        answer_text = build_fallback_answer(
            query=query,
            search_results=search_results,
            page_results=page_results,
            search_options=search_options,
        )

    refinement_meta: Dict[str, Any] = {
        "enabled": bool(enable_llm and _option_flag(search_options, ["enableSelfRefine", "enable_self_refine"], "IQS_ENABLE_SELF_REFINE", True)),
        "follow_up_queries": [],
        "search_trace": [],
        "errors": [],
        "combined_quality_processing": {},
    }
    if isinstance(llm_meta.get("structured_payload"), dict):
        initial_answer_payload = llm_meta["structured_payload"].get("answer") if isinstance(llm_meta["structured_payload"].get("answer"), dict) else {}
        refinement_meta["initial_analysis"] = {
            "confidence": llm_meta.get("confidence"),
            "evidence_gap": initial_answer_payload.get("evidence_gap", []),
            "next_steps": llm_meta.get("next_steps", []),
        }
    if refinement_meta["enabled"] and llm_meta.get("source") == "llm":
        initial_payload = llm_meta.get("structured_payload") if isinstance(llm_meta.get("structured_payload"), dict) else {}
        follow_up_queries = build_follow_up_queries_from_payload(
            query=query,
            payload=initial_payload,
            max_queries=_env_int("IQS_SELF_REFINE_MAX_QUERIES", 6, min_value=1, max_value=10),
        )
        refinement_meta["follow_up_queries"] = follow_up_queries
        if follow_up_queries:
            _progress("web-agent", "联网自修补检索开始", followups=len(follow_up_queries))
            follow_up_results, follow_up_trace, follow_up_errors = run_follow_up_iqs_searches(
                original_query=query,
                follow_up_queries=follow_up_queries,
                search_options=search_options,
            )
            refinement_meta["search_trace"] = follow_up_trace
            refinement_meta["errors"] = follow_up_errors
            if follow_up_results:
                combined_raw = [dict(item) for item in search_results] + [dict(item) for item in follow_up_results]
                combined_query = " ".join([query, *follow_up_queries])
                final_top_k = _env_int("IQS_SELF_REFINE_FINAL_TOP_K", 60, min_value=1, max_value=100)
                combined_results, combined_quality = process_web_results_with_top_k(
                    combined_query,
                    combined_raw,
                    top_k=final_top_k,
                    options=search_options,
                )
                preserved_initial = [dict(item) for item in search_results[: min(len(search_results), max(3, final_top_k // 2))]]
                merged_results: List[Dict[str, Any]] = []
                seen_keys = set()
                for item in [*preserved_initial, *combined_results]:
                    key = _normalize_url(str(item.get("url") or "")) or str(item.get("title") or "").strip().lower()
                    if key and key in seen_keys:
                        continue
                    if key:
                        seen_keys.add(key)
                    merged_results.append(dict(item))
                    if len(merged_results) >= final_top_k:
                        break
                combined_results = merged_results
                combined_quality["preserved_initial_count"] = len([item for item in combined_results if item.get("research_round") == "initial"])
                for item in combined_results:
                    if isinstance(item, dict):
                        item.setdefault("research_round", "combined")
                combined_results, combined_pages = assign_source_ids(combined_results, page_results)
                try:
                    refined_answer, refined_meta = synthesize_refined_with_llm(
                        query=query,
                        initial_answer=answer_text,
                        initial_payload=initial_payload,
                        follow_up_queries=follow_up_queries,
                        search_results=combined_results,
                        page_results=combined_pages,
                        search_options=search_options,
                    )
                    answer_text = refined_answer
                    llm_meta = refined_meta
                    search_results = combined_results
                    page_results = combined_pages
                    refinement_meta["combined_quality_processing"] = combined_quality
                    refinement_meta["final_result_count"] = len(combined_results)
                except Exception as exc:
                    refinement_meta["errors"] = list(refinement_meta.get("errors") or []) + [f"二次合并失败：{exc}"]
            _progress(
                "web-agent",
                "联网自修补检索完成",
                followup_results=len(follow_up_results),
                errors=len(follow_up_errors),
            )

    if answer_text and "本次使用的联网来源：" not in answer_text:
        answer_text = answer_text.rstrip() + "\n\n" + format_sources(search_results, page_results)

    _progress(
        "web-agent",
        "联网分析综合完成",
        source=llm_meta.get("source", "fallback_extractive"),
        results=len(search_results),
        self_refine=bool(refinement_meta.get("follow_up_queries")),
        elapsed=f"{time.perf_counter() - started:.1f}s",
    )
    return {
        "answer_text": answer_text,
        "raw_output": {
            "query": query,
            "search_options": search_options,
            "query_plan": (state.get("metadata") or {}).get("query_plan", []),
            "search_tasks": (state.get("metadata") or {}).get("search_tasks", []),
            "search_trace": (state.get("metadata") or {}).get("search_trace", []),
            "quality_processing": (state.get("metadata") or {}).get("quality_processing", {}),
            "search_results": search_results,
            "page_results": page_results,
            "synthesis": llm_meta,
            "self_refinement": refinement_meta,
        },
        "metadata": {
            **dict(state.get("metadata") or {}),
            "agent_stage": "synthesize_analysis",
            "grounding_mode": llm_meta.get("source", "fallback_extractive"),
            "llm_model": llm_meta.get("model", ""),
            "self_refinement": refinement_meta,
        },
    }


def format_agent_response_node(state: WebAnalysisAgentState) -> WebAnalysisAgentState:
    errors = list(state.get("errors") or [])
    if errors:
        answer_text = "联网分析 Agent 失败：" + errors[-1]
    else:
        answer_text = str(state.get("answer_text") or "").strip() or "当前没有生成可用的联网分析。"

    messages = list(state.get("messages") or [])
    messages.append(
        {
            "role": "assistant",
            "name": AGENT_NAME,
            "content": answer_text,
        }
    )
    return {
        "answer_text": answer_text,
        "messages": messages,
        "metadata": {
            **dict(state.get("metadata") or {}),
            "agent_stage": "format_response",
            "handoff_ready": not bool(errors),
        },
    }


def create_web_analysis_agent_graph(*, name: str = AGENT_NAME):
    builder = StateGraph(WebAnalysisAgentState)
    builder.add_node("prepare_query", prepare_query_node)
    builder.add_node("iqs_research", iqs_research_node)
    builder.add_node("synthesize_analysis", synthesize_analysis_node)
    builder.add_node("format_response", format_agent_response_node)
    builder.add_edge(START, "prepare_query")
    builder.add_edge("prepare_query", "iqs_research")
    builder.add_edge("iqs_research", "synthesize_analysis")
    builder.add_edge("synthesize_analysis", "format_response")
    builder.add_edge("format_response", END)
    return builder.compile(name=name)


def run_web_analysis_agent(
    query: str,
    *,
    messages: Optional[Sequence[Dict[str, Any]]] = None,
    urls: Optional[Sequence[str]] = None,
    search_options: Optional[Dict[str, Any]] = None,
    enable_llm_analysis: bool = True,
) -> WebAnalysisAgentState:
    started = time.perf_counter()
    _progress("web-agent", "联网分析 Agent 启动", query=query)
    graph = create_web_analysis_agent_graph()
    initial_messages = list(messages or [])
    if query and not initial_messages:
        initial_messages.append({"role": "user", "content": query})
    state: WebAnalysisAgentState = {
        "query": query,
        "messages": initial_messages,
        "urls": list(urls or []),
        "search_options": dict(search_options or {}),
        "enable_llm_analysis": enable_llm_analysis,
    }
    cache_key = ""
    if _web_cache_allowed(search_options):
        cache_key = _web_cache_key(
            query=query,
            messages=initial_messages,
            urls=list(urls or []),
            search_options=search_options,
            enable_llm_analysis=enable_llm_analysis,
        )
        cached_state = _WEB_ANALYSIS_CACHE.get(cache_key)
        if cached_state:
            _progress("web-agent", "命中联网分析缓存", query=query)
            return _mark_web_cache_hit(cached_state)

    result = graph.invoke(state)
    if cache_key and not result.get("errors"):
        _WEB_ANALYSIS_CACHE.set(cache_key, result)
    _progress(
        "web-agent",
        "联网分析 Agent 完成",
        search_results=len(result.get("search_results") or []),
        page_results=len(result.get("page_results") or []),
        errors=len(result.get("errors") or []),
        elapsed=f"{time.perf_counter() - started:.1f}s",
    )
    return result


def web_analysis_tool(query: str, **kwargs: Any) -> str:
    """供 supervisor 或 planner 智能体调用的工具式入口。"""

    result = run_web_analysis_agent(query, search_options=kwargs)
    return str(result.get("answer_text") or "")


def create_web_analysis_tool():
    """返回可被 supervisor 风格多智能体图调用的 LangChain 兼容工具。"""

    from langchain_core.tools import tool

    @tool("web_analysis_agent", description=AGENT_DESCRIPTION)
    def _web_analysis_agent(query: str) -> str:
        return web_analysis_tool(query)

    return _web_analysis_agent


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="基于阿里云 IQS Skills 的 LangGraph 联网分析 Agent。")
    parser.add_argument("query", nargs="*", help="需要当前联网信息的问题。")
    parser.add_argument("-q", "--query-text", "--query", nargs="+", help="显式传入查询文本。")
    parser.add_argument("--url", action="append", default=[], help="读取并分析指定 URL，可重复传入。")
    parser.add_argument("--engine-type", default="", help="IQS 搜索引擎类型/资源名；填 auto 时按意图使用 .env 中的资源池。")
    parser.add_argument("--time-range", choices=["NoLimit", "OneDay", "OneWeek", "OneMonth", "OneYear"], default="", help="IQS 搜索时间范围。")
    parser.add_argument("--contents", choices=["summary", "mainText"], default="", help="IQS 返回内容类型。")
    parser.add_argument("--category", default="", help="透传给阿里云 IQS 的 category 参数。")
    parser.add_argument("--num-results", type=int, default=0, help="搜索结果数量，范围 1-100。")
    parser.add_argument("--timeout-ms", type=int, default=0, help="IQS 搜索超时时间，单位毫秒。")
    parser.add_argument("--disable-query-optimization", action="store_true", help="关闭 Query 重写/拆解，直接使用原始问题搜索。")
    parser.add_argument("--max-queries", type=int, default=0, help="Query 优化后最多并行搜索的子查询数量。")
    parser.add_argument("--results-per-query", type=int, default=0, help="每个子查询的 IQS 返回数量。")
    parser.add_argument("--rerank-top-k", type=int, default=0, help="联网结果 rerank 后保留数量。")
    parser.add_argument("--disable-web-rerank", action="store_true", help="关闭联网搜索结果的外部 rerank。")
    parser.add_argument("--no-llm", dest="enable_llm_analysis", action="store_false", default=True, help="关闭大模型综合分析，仅返回抽取式网页摘要。")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出完整 Agent 状态。")
    return parser


def parse_query_from_args(args: argparse.Namespace) -> str:
    if args.query_text:
        return " ".join(str(item) for item in args.query_text).strip()
    return " ".join(str(item) for item in args.query).strip()


def search_options_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    options: Dict[str, Any] = {}
    if args.engine_type:
        options["engineType"] = args.engine_type
    if args.time_range:
        options["timeRange"] = args.time_range
    if args.contents:
        options["contents"] = args.contents
    if args.category:
        options["category"] = args.category
    if args.num_results:
        options["numResults"] = args.num_results
    if args.timeout_ms:
        options["timeout"] = args.timeout_ms
    if args.disable_query_optimization:
        os.environ["IQS_ENABLE_QUERY_OPTIMIZATION"] = "0"
    if args.max_queries:
        os.environ["IQS_MAX_QUERIES"] = str(args.max_queries)
    if args.results_per_query:
        os.environ["IQS_RESULTS_PER_QUERY"] = str(args.results_per_query)
    if args.rerank_top_k:
        os.environ["IQS_RERANK_TOP_K"] = str(args.rerank_top_k)
    if args.disable_web_rerank:
        os.environ["RAG_ENABLE_API_RERANK"] = "0"
    return options


def main() -> int:
    args = build_arg_parser().parse_args()
    query = parse_query_from_args(args)
    urls = list(args.url or [])
    if not query and not urls:
        query = input("请输入联网查询：").strip()
    state = run_web_analysis_agent(
        query,
        urls=urls,
        search_options=search_options_from_args(args),
        enable_llm_analysis=bool(args.enable_llm_analysis),
    )
    if args.json:
        print(json.dumps(state, ensure_ascii=False, indent=2, default=str))
    else:
        print(str(state.get("answer_text") or "").strip())
    return 1 if state.get("errors") else 0


if __name__ == "__main__":
    raise SystemExit(main())
