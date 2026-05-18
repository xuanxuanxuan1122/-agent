from __future__ import annotations

import os
import re
from pathlib import Path

from ..ingest.embedding_qdrant import DEFAULT_BGE_M3_MODEL_PATH, DEFAULT_ENABLE_SPARSE_VECTORS


def env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def env_first_nonempty(*names: str, default: str = "") -> str:
    for name in names:
        raw = os.getenv(name)
        if raw is not None and str(raw).strip():
            return str(raw).strip()
    return default


def normalize_llm_profile(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", str(value or "").strip().upper()).strip("_")


def llm_profile_env_name(profile: str, field: str) -> str:
    profile_key = normalize_llm_profile(profile)
    field_key = normalize_llm_profile(field)
    return f"RAG_LLM_PROFILE_{profile_key}_{field_key}" if profile_key and field_key else ""


def env_profile_value(profile: str, field: str, *fallback_names: str, default: str = "") -> str:
    profile_key = normalize_llm_profile(profile)
    if profile_key:
        raw = os.getenv(llm_profile_env_name(profile_key, field))
        if raw is not None:
            return str(raw).strip()
        return ""
    return env_first_nonempty(*fallback_names, default=default)


def env_profile_timeout(profile: str, field: str, fallback_name: str, default: float) -> float:
    profile_key = normalize_llm_profile(profile)
    raw = os.getenv(llm_profile_env_name(profile_key, field)) if profile_key else None
    if raw is None or not str(raw).strip():
        raw = os.getenv(fallback_name, str(default))
    try:
        return float(str(raw).strip())
    except ValueError:
        return default


def env_profile_flag(profile: str, field: str, default: bool) -> bool:
    profile_key = normalize_llm_profile(profile)
    raw = os.getenv(llm_profile_env_name(profile_key, field)) if profile_key else None
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


PIPELINE_ROOT = Path(__file__).resolve().parents[2]
HF_CACHE_DIR = Path(os.getenv("RAG_HF_CACHE_DIR", str(PIPELINE_ROOT / ".hf_cache")))
DEFAULT_TRACE_DIR = Path(os.getenv("RAG_TRACE_DIR", str(PIPELINE_ROOT / "traces")))
DEFAULT_MEMORY_STORE_DIR = Path(os.getenv("RAG_MEMORY_STORE_DIR", str(PIPELINE_ROOT / "sessions")))

DEFAULT_TOP_K = max(1, int(os.getenv("RAG_TOP_K", "5")))
DEFAULT_PREVIEW_CHARS = 320
DEFAULT_CANDIDATE_MULTIPLIER = max(1, int(os.getenv("RAG_CANDIDATE_MULTIPLIER", "5")))
DEFAULT_QUERY_VARIANTS = 3
DEFAULT_MAX_PER_DOCUMENT = 10
DEFAULT_RRF_K = 60
DEFAULT_DENSE_RRF_WEIGHT = float(os.getenv("RAG_DENSE_RRF_WEIGHT", "1.00"))
DEFAULT_SPARSE_RRF_WEIGHT = float(os.getenv("RAG_SPARSE_RRF_WEIGHT", "0.90"))
DEFAULT_FLAT_CHILD_RRF_WEIGHT = float(os.getenv("RAG_FLAT_CHILD_RRF_WEIGHT", "0.95"))
DEFAULT_PARENT_RRF_WEIGHT = float(os.getenv("RAG_PARENT_RRF_WEIGHT", "0.80"))
DEFAULT_ENABLE_SPARSE_RETRIEVAL = DEFAULT_ENABLE_SPARSE_VECTORS
DEFAULT_KEEP_EMBEDDER_LOADED = env_flag("RAG_KEEP_EMBEDDER_LOADED", True)
DEFAULT_PARALLEL_HIERARCHICAL_RETRIEVAL = env_flag("RAG_PARALLEL_HIERARCHICAL_RETRIEVAL", True)
DEFAULT_QUERY_HNSW_EF = max(0, int(os.getenv("QDRANT_QUERY_HNSW_EF", "96")))
DEFAULT_QUERY_EXACT = env_flag("QDRANT_QUERY_EXACT", False)
DEFAULT_QUERY_QUANTIZATION_RESCORE = env_flag("QDRANT_QUERY_QUANTIZATION_RESCORE", True)
DEFAULT_QUERY_QUANTIZATION_OVERSAMPLING = max(
    1.0,
    float(os.getenv("QDRANT_QUERY_QUANTIZATION_OVERSAMPLING", "1.0")),
)

DEFAULT_RERANK_PROVIDER = os.getenv("RAG_RERANK_PROVIDER", "cohere")
DEFAULT_RERANK_URL = os.getenv("RAG_RERANK_URL", "https://dashscope.aliyuncs.com/compatible-api/v1/reranks")
DEFAULT_RERANK_API_KEY = os.getenv("RAG_RERANK_API_KEY", "").strip()
DEFAULT_RERANK_MODEL = os.getenv("RAG_RERANK_MODEL", "qwen3-rerank")
DEFAULT_RERANK_TOP_N = int(os.getenv("RAG_RERANK_TOP_N", "5"))
DEFAULT_RERANK_MAX_DOCS = int(os.getenv("RAG_RERANK_MAX_DOCS", "25"))
DEFAULT_RERANK_MAX_CHARS_PER_DOC = int(os.getenv("RAG_RERANK_MAX_CHARS_PER_DOC", "2000"))
DEFAULT_RERANK_TIMEOUT = float(os.getenv("RAG_RERANK_TIMEOUT", "30"))
DEFAULT_ENABLE_API_RERANK = env_flag("RAG_ENABLE_API_RERANK", False)
DEFAULT_EXTERNAL_API_TRUST_ENV = env_flag("RAG_EXTERNAL_API_TRUST_ENV", False)
DEFAULT_LLM_ACTIVE_PROFILE = os.getenv("RAG_LLM_ACTIVE_PROFILE", "").strip()
DEFAULT_LLM_EXECUTION_PROFILE = (
    os.getenv("RAG_LLM_EXECUTION_PROFILE", DEFAULT_LLM_ACTIVE_PROFILE).strip()
    or DEFAULT_LLM_ACTIVE_PROFILE
)
DEFAULT_LLM_DISABLE_THINKING = env_profile_flag(
    DEFAULT_LLM_ACTIVE_PROFILE,
    "DISABLE_THINKING",
    env_flag("RAG_LLM_DISABLE_THINKING", True),
)
DEFAULT_ENABLE_LOCAL_RERANK = env_flag("RAG_ENABLE_LOCAL_RERANK", False)
DEFAULT_LOCAL_RERANK_MODEL_PATH = os.getenv("RAG_LOCAL_RERANK_MODEL_PATH", "").strip()
DEFAULT_LOCAL_RERANK_MAX_DOCS = int(os.getenv("RAG_LOCAL_RERANK_MAX_DOCS", "25"))
DEFAULT_LOCAL_RERANK_BATCH_SIZE = max(1, int(os.getenv("RAG_LOCAL_RERANK_BATCH_SIZE", "8")))
DEFAULT_ENABLE_BGE_DENSE_RETRIEVAL = env_flag(
    "RAG_ENABLE_BGE_DENSE_RETRIEVAL",
    bool(DEFAULT_BGE_M3_MODEL_PATH),
)
DEFAULT_ENABLE_BGE_SPARSE_RETRIEVAL = env_flag(
    "RAG_ENABLE_BGE_SPARSE_RETRIEVAL",
    bool(DEFAULT_BGE_M3_MODEL_PATH),
)

DEFAULT_ENABLE_QUERY_PLANNER = env_flag("RAG_ENABLE_QUERY_PLANNER", True)
DEFAULT_PLANNER_HARD_FILTERS = env_flag("RAG_PLANNER_HARD_FILTERS", False)
DEFAULT_ENABLE_LLM_PLANNER = env_flag("RAG_ENABLE_LLM_PLANNER", False)
DEFAULT_LLM_PLANNER_PROVIDER = (
    env_profile_value(
        DEFAULT_LLM_ACTIVE_PROFILE,
        "PROVIDER",
        "RAG_LLM_PLANNER_PROVIDER",
        default="openai_compatible",
    )
    or "openai_compatible"
).strip().lower()
DEFAULT_LLM_PLANNER_URL = env_profile_value(DEFAULT_LLM_ACTIVE_PROFILE, "URL", "RAG_LLM_PLANNER_URL")
DEFAULT_LLM_PLANNER_API_KEY = env_profile_value(DEFAULT_LLM_ACTIVE_PROFILE, "API_KEY", "RAG_LLM_PLANNER_API_KEY")
DEFAULT_LLM_PLANNER_MODEL = env_profile_value(DEFAULT_LLM_ACTIVE_PROFILE, "MODEL", "RAG_LLM_PLANNER_MODEL")
DEFAULT_LLM_PLANNER_TIMEOUT = env_profile_timeout(DEFAULT_LLM_ACTIVE_PROFILE, "TIMEOUT", "RAG_LLM_PLANNER_TIMEOUT", 30)
DEFAULT_LLM_PLANNER_MAX_QUERIES = max(4, int(os.getenv("RAG_LLM_PLANNER_MAX_QUERIES", "8")))

DEFAULT_ANSWER_MODE = os.getenv("RAG_ANSWER_MODE", "grounded")
DEFAULT_EVIDENCE_TOP_K = max(3, int(os.getenv("RAG_EVIDENCE_TOP_K", "5")))
DEFAULT_MIN_EVIDENCE = max(1, int(os.getenv("RAG_MIN_EVIDENCE", "2")))
DEFAULT_MIN_EVIDENCE_SCORE = max(0.0, float(os.getenv("RAG_MIN_EVIDENCE_SCORE", "0.55")))
DEFAULT_MAX_ANSWER_CLAIMS = max(1, int(os.getenv("RAG_MAX_ANSWER_CLAIMS", "4")))
DEFAULT_ANSWER_SHOW_EVIDENCE = env_flag("RAG_ANSWER_SHOW_EVIDENCE", True)
DEFAULT_ANSWER_EVIDENCE_TOP_K = max(1, int(os.getenv("RAG_ANSWER_EVIDENCE_TOP_K", "5")))
DEFAULT_CORE_EVIDENCE_TOP_K = max(2, int(os.getenv("RAG_CORE_EVIDENCE_TOP_K", "3")))
DEFAULT_SUPPORT_EVIDENCE_TOP_K = max(DEFAULT_CORE_EVIDENCE_TOP_K, int(os.getenv("RAG_SUPPORT_EVIDENCE_TOP_K", "5")))
DEFAULT_LLM_CONTEXT_MAX_TOKENS = max(800, int(os.getenv("RAG_LLM_CONTEXT_MAX_TOKENS", "6000")))
DEFAULT_LLM_CONTEXT_MAX_TOKENS_PER_EVIDENCE = max(120, int(os.getenv("RAG_LLM_CONTEXT_MAX_TOKENS_PER_EVIDENCE", "900")))
DEFAULT_LLM_CONTEXT_DEDUP_THRESHOLD = min(
    0.99,
    max(0.50, float(os.getenv("RAG_LLM_CONTEXT_DEDUP_THRESHOLD", "0.86"))),
)

DEFAULT_LLM_SYNTHESIS_PROVIDER = (
    env_profile_value(
        DEFAULT_LLM_EXECUTION_PROFILE,
        "PROVIDER",
        "RAG_LLM_SYNTHESIS_PROVIDER",
        default=DEFAULT_LLM_PLANNER_PROVIDER,
    )
    or "openai_compatible"
).strip().lower()
DEFAULT_LLM_SYNTHESIS_URL = env_profile_value(
    DEFAULT_LLM_EXECUTION_PROFILE,
    "URL",
    "RAG_LLM_SYNTHESIS_URL",
    "RAG_LLM_REFLECTION_URL",
    "RAG_LLM_PLANNER_URL",
    default=DEFAULT_LLM_PLANNER_URL,
)
DEFAULT_LLM_SYNTHESIS_API_KEY = env_profile_value(
    DEFAULT_LLM_EXECUTION_PROFILE,
    "API_KEY",
    "RAG_LLM_SYNTHESIS_API_KEY",
    "RAG_LLM_REFLECTION_API_KEY",
    "RAG_LLM_PLANNER_API_KEY",
    default=DEFAULT_LLM_PLANNER_API_KEY,
)
DEFAULT_LLM_SYNTHESIS_MODEL = env_profile_value(
    DEFAULT_LLM_EXECUTION_PROFILE,
    "MODEL",
    "RAG_LLM_SYNTHESIS_MODEL",
    "RAG_LLM_REFLECTION_MODEL",
    "RAG_LLM_PLANNER_MODEL",
    default=DEFAULT_LLM_PLANNER_MODEL,
)
DEFAULT_LLM_SYNTHESIS_TIMEOUT = env_profile_timeout(
    DEFAULT_LLM_EXECUTION_PROFILE,
    "TIMEOUT",
    "RAG_LLM_SYNTHESIS_TIMEOUT",
    DEFAULT_LLM_PLANNER_TIMEOUT,
)
DEFAULT_LLM_SYNTHESIS_DISABLE_THINKING = env_profile_flag(
    DEFAULT_LLM_EXECUTION_PROFILE,
    "DISABLE_THINKING",
    env_flag("RAG_LLM_SYNTHESIS_DISABLE_THINKING", DEFAULT_LLM_DISABLE_THINKING),
)
DEFAULT_ENABLE_LLM_SYNTHESIS = env_flag(
    "RAG_ENABLE_LLM_SYNTHESIS",
    bool(DEFAULT_LLM_SYNTHESIS_URL and DEFAULT_LLM_SYNTHESIS_API_KEY and DEFAULT_LLM_SYNTHESIS_MODEL),
)
DEFAULT_ENABLE_ANSWER_REVIEW = env_flag("RAG_ENABLE_ANSWER_REVIEW", True)
DEFAULT_ENABLE_LLM_ANSWER_REVIEW = env_flag(
    "RAG_ENABLE_LLM_ANSWER_REVIEW",
    bool(DEFAULT_LLM_SYNTHESIS_URL and DEFAULT_LLM_SYNTHESIS_API_KEY and DEFAULT_LLM_SYNTHESIS_MODEL),
)
DEFAULT_LLM_ANSWER_REVIEW_PROVIDER = (
    env_profile_value(
        DEFAULT_LLM_ACTIVE_PROFILE,
        "PROVIDER",
        "RAG_LLM_ANSWER_REVIEW_PROVIDER",
        default=DEFAULT_LLM_SYNTHESIS_PROVIDER,
    )
    or "openai_compatible"
).strip().lower()
DEFAULT_LLM_ANSWER_REVIEW_URL = env_profile_value(
    DEFAULT_LLM_ACTIVE_PROFILE,
    "URL",
    "RAG_LLM_ANSWER_REVIEW_URL",
    default=DEFAULT_LLM_SYNTHESIS_URL,
)
DEFAULT_LLM_ANSWER_REVIEW_API_KEY = env_profile_value(
    DEFAULT_LLM_ACTIVE_PROFILE,
    "API_KEY",
    "RAG_LLM_ANSWER_REVIEW_API_KEY",
    default=DEFAULT_LLM_SYNTHESIS_API_KEY,
)
DEFAULT_LLM_ANSWER_REVIEW_MODEL = env_profile_value(
    DEFAULT_LLM_ACTIVE_PROFILE,
    "MODEL",
    "RAG_LLM_ANSWER_REVIEW_MODEL",
    default=DEFAULT_LLM_SYNTHESIS_MODEL,
)
DEFAULT_LLM_ANSWER_REVIEW_TIMEOUT = env_profile_timeout(
    DEFAULT_LLM_ACTIVE_PROFILE,
    "TIMEOUT",
    "RAG_LLM_ANSWER_REVIEW_TIMEOUT",
    DEFAULT_LLM_SYNTHESIS_TIMEOUT,
)
DEFAULT_ENABLE_REFLECTION = env_flag("RAG_ENABLE_REFLECTION", True)
DEFAULT_REFLECTION_MAX_HOPS = max(1, int(os.getenv("RAG_REFLECTION_MAX_HOPS", "3")))
DEFAULT_REFLECTION_OVERLAP_THRESHOLD = min(
    1.0,
    max(0.0, float(os.getenv("RAG_REFLECTION_OVERLAP_THRESHOLD", "0.80"))),
)
DEFAULT_ENABLE_LLM_REFLECTION = env_flag(
    "RAG_ENABLE_LLM_REFLECTION",
    bool(DEFAULT_LLM_SYNTHESIS_URL and DEFAULT_LLM_SYNTHESIS_API_KEY and DEFAULT_LLM_SYNTHESIS_MODEL),
)
DEFAULT_LLM_REFLECTION_PROVIDER = (
    env_profile_value(
        DEFAULT_LLM_ACTIVE_PROFILE,
        "PROVIDER",
        "RAG_LLM_REFLECTION_PROVIDER",
        default=DEFAULT_LLM_SYNTHESIS_PROVIDER,
    )
    or "openai_compatible"
).strip().lower()
DEFAULT_LLM_REFLECTION_URL = env_profile_value(
    DEFAULT_LLM_ACTIVE_PROFILE,
    "URL",
    "RAG_LLM_REFLECTION_URL",
    default=DEFAULT_LLM_SYNTHESIS_URL,
)
DEFAULT_LLM_REFLECTION_API_KEY = env_profile_value(
    DEFAULT_LLM_ACTIVE_PROFILE,
    "API_KEY",
    "RAG_LLM_REFLECTION_API_KEY",
    default=DEFAULT_LLM_SYNTHESIS_API_KEY,
)
DEFAULT_LLM_REFLECTION_MODEL = env_profile_value(
    DEFAULT_LLM_ACTIVE_PROFILE,
    "MODEL",
    "RAG_LLM_REFLECTION_MODEL",
    default=DEFAULT_LLM_SYNTHESIS_MODEL,
)
DEFAULT_LLM_REFLECTION_TIMEOUT = env_profile_timeout(
    DEFAULT_LLM_ACTIVE_PROFILE,
    "TIMEOUT",
    "RAG_LLM_REFLECTION_TIMEOUT",
    DEFAULT_LLM_SYNTHESIS_TIMEOUT,
)

DEFAULT_ENABLE_MEMORY = env_flag("RAG_ENABLE_MEMORY", True)
DEFAULT_ENABLE_CONTEXTUALIZER = env_flag(
    "RAG_ENABLE_CONTEXTUALIZER",
    bool(DEFAULT_LLM_SYNTHESIS_URL and DEFAULT_LLM_SYNTHESIS_API_KEY and DEFAULT_LLM_SYNTHESIS_MODEL),
)
DEFAULT_CONTEXT_HISTORY_TURNS = max(1, int(os.getenv("RAG_CONTEXT_HISTORY_TURNS", "4")))
DEFAULT_MEMORY_MAX_TURNS = max(DEFAULT_CONTEXT_HISTORY_TURNS, int(os.getenv("RAG_MEMORY_MAX_TURNS", "10")))
DEFAULT_MEMORY_SUMMARY_TRIGGER = max(DEFAULT_CONTEXT_HISTORY_TURNS + 1, int(os.getenv("RAG_MEMORY_SUMMARY_TRIGGER", "6")))

DEFAULT_TRACE_ENABLED = env_flag("RAG_TRACE_ENABLED", True)
DEFAULT_TRACE_TOP_K = max(1, int(os.getenv("RAG_TRACE_TOP_K", "10")))

DEFAULT_RETRIEVAL_INSTRUCTION = (
    "指令：为有证据约束的 RAG 回答检索最相关的行业研究证据。"
    "优先召回包含市场规模、增速、渗透率、供需、竞争格局、产业链/价值链、商业模式、价格、毛利/盈利能力、政策、风险、数据来源和时间口径的段落。"
    "除非宣传性表述能够直接支撑事实判断，否则不要优先召回宣传性话术。\n查询：{query}"
)
DEFAULT_POLICY_INSTRUCTION = (
    "指令：检索能够直接支撑合规、准确回答的政策条款和流程性段落。\n查询：{query}"
)
DEFAULT_CODE_INSTRUCTION = (
    "指令：检索能够直接解决工程问题的代码、API、错误处理和技术文档段落。\n查询：{query}"
)


for required_dir in [HF_CACHE_DIR, DEFAULT_TRACE_DIR, DEFAULT_MEMORY_STORE_DIR]:
    required_dir.mkdir(parents=True, exist_ok=True)
