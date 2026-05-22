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


def llm_task_profile_env_name(task_name: str) -> str:
    task_key = normalize_llm_profile(task_name)
    return f"RAG_MODEL_{task_key}_PROFILE" if task_key else ""


def env_profile_value(profile: str, field: str, *fallback_names: str, default: str = "") -> str:
    profile_key = normalize_llm_profile(profile)
    if profile_key:
        raw = os.getenv(llm_profile_env_name(profile_key, field))
        if raw is not None and str(raw).strip():
            return str(raw).strip()
        if not fallback_names and not default:
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


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(str(raw).strip())
    except ValueError:
        return default


def _env_int_optional(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return None
    try:
        return max(1, int(str(raw).strip()))
    except ValueError:
        return None


def _config_is_ready(config: dict[str, object]) -> bool:
    return bool(
        str(config.get("url") or "").strip()
        and str(config.get("api_key") or "").strip()
        and str(config.get("model") or "").strip()
    )


GPT55_QUALITY_TASKS = {
    "planning",
    "coverage_eval",
    "qa",
    "review_stage2",
    "reflection",
    "reformatter",
    "final_audit",
}
GPT55_QUALITY_PROFILE = "gpt-5.5"
GPT55_FALLBACK_PROFILE = os.getenv("RAG_LLM_GPT55_FALLBACK_PROFILE", "deepseek-v4-pro").strip()


def build_llm_config_from_profile(profile: str, *, default_timeout: float = 180.0) -> dict[str, object]:
    """Build an OpenAI-compatible LLM config from a named RAG_LLM_PROFILE_* block."""

    profile = str(profile or "").strip()
    provider = env_profile_value(profile, "PROVIDER", default="openai_compatible") or "openai_compatible"
    timeout_env = llm_profile_env_name(profile, "TIMEOUT")
    reasoning_effort = env_profile_value(profile, "REASONING_EFFORT").lower()
    max_output_tokens = _env_int_optional(llm_profile_env_name(profile, "MAX_OUTPUT_TOKENS"))
    config: dict[str, object] = {
        "provider": provider.strip().lower(),
        "url": env_profile_value(profile, "URL"),
        "api_key": env_profile_value(profile, "API_KEY"),
        "model": env_profile_value(profile, "MODEL"),
        "timeout": _env_float(timeout_env, default_timeout) if timeout_env else default_timeout,
        "profile": profile,
        "disable_thinking": env_profile_flag(
            profile,
            "DISABLE_THINKING",
            env_flag("RAG_LLM_DISABLE_THINKING", True),
        ),
    }
    if reasoning_effort:
        config["reasoning_effort"] = reasoning_effort
    if max_output_tokens:
        config["max_output_tokens"] = max_output_tokens
    return config


def _config_uses_gpt55(config: dict[str, object]) -> bool:
    model = str(config.get("model") or "").strip().lower()
    profile = str(config.get("profile") or "").strip().lower()
    return model.startswith("gpt-5.5") or profile == GPT55_QUALITY_PROFILE


def _attach_gpt55_fallback(config: dict[str, object], *, task_name: str = "") -> dict[str, object]:
    """Attach a DeepSeek fallback to GPT-5.5 configs without changing callers."""

    config = dict(config or {})
    if not _config_uses_gpt55(config):
        return config
    fallback_profile = os.getenv("RAG_LLM_GPT55_FALLBACK_PROFILE", GPT55_FALLBACK_PROFILE).strip()
    if not fallback_profile or fallback_profile.lower() == str(config.get("profile") or "").strip().lower():
        return config
    fallback_config = build_llm_config_from_profile(
        fallback_profile,
        default_timeout=float(config.get("timeout") or DEFAULT_LLM_SYNTHESIS_TIMEOUT),
    )
    if not _config_is_ready(fallback_config):
        return config
    fallback_config["task_name"] = task_name or str(config.get("task_name") or "")
    fallback_config["fallback_for_profile"] = str(config.get("profile") or "")
    fallback_config.pop("forced_quality_profile", None)
    config["fallback_profile"] = fallback_profile
    config["fallback_config"] = fallback_config
    return config


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


def build_legacy_synthesis_llm_config() -> dict[str, object]:
    """Return the historical synthesis config used before functional model routing."""

    config: dict[str, object] = {
        "provider": DEFAULT_LLM_SYNTHESIS_PROVIDER,
        "url": DEFAULT_LLM_SYNTHESIS_URL,
        "api_key": DEFAULT_LLM_SYNTHESIS_API_KEY,
        "model": DEFAULT_LLM_SYNTHESIS_MODEL,
        "timeout": DEFAULT_LLM_SYNTHESIS_TIMEOUT,
        "profile": DEFAULT_LLM_EXECUTION_PROFILE or DEFAULT_LLM_ACTIVE_PROFILE or "legacy_synthesis",
        "disable_thinking": DEFAULT_LLM_SYNTHESIS_DISABLE_THINKING,
    }
    reasoning_effort = env_first_nonempty("RAG_LLM_SYNTHESIS_REASONING_EFFORT").lower()
    max_output_tokens = _env_int_optional("RAG_LLM_SYNTHESIS_MAX_OUTPUT_TOKENS")
    if reasoning_effort:
        config["reasoning_effort"] = reasoning_effort
    if max_output_tokens:
        config["max_output_tokens"] = max_output_tokens
    return config


def build_llm_config_for_task(task_name: str) -> dict[str, object]:
    """Resolve an LLM config for a functional task.

    The resolver reads RAG_MODEL_{TASK}_PROFILE first. If the selected profile
    is absent or missing URL/API key/model, it falls back to the existing
    synthesis config so older deployments keep working.
    """

    task = str(task_name or "").strip()
    if env_flag("RAG_FORCE_GPT55_QUALITY_TASKS", True) and normalize_llm_profile(task).lower() in {
        normalize_llm_profile(item).lower() for item in GPT55_QUALITY_TASKS
    }:
        config = build_llm_config_from_profile(GPT55_QUALITY_PROFILE, default_timeout=DEFAULT_LLM_SYNTHESIS_TIMEOUT)
        if _config_is_ready(config):
            config["task_name"] = task
            config["forced_quality_profile"] = True
            return _attach_gpt55_fallback(config, task_name=task)

    task_env_name = llm_task_profile_env_name(task)
    profile = os.getenv(task_env_name, "").strip() if task_env_name else ""
    if profile:
        config = build_llm_config_from_profile(profile, default_timeout=DEFAULT_LLM_SYNTHESIS_TIMEOUT)
        if _config_is_ready(config):
            config["task_name"] = task
            return _attach_gpt55_fallback(config, task_name=task)
    config = build_legacy_synthesis_llm_config()
    config["task_name"] = task
    return _attach_gpt55_fallback(config, task_name=task)


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
