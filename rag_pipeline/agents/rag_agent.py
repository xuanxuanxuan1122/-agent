from __future__ import annotations

import argparse
import copy
import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, TypedDict

from langgraph.graph import END, START, StateGraph

from ..runtime_cache import TTLCache, make_cache_key
from ..search.engine import build_arg_parser, format_answer_only_output, run_search


AGENT_NAME = "industry_rag_agent"
AGENT_DESCRIPTION = (
    "行研 RAG 子智能体。适用于需要基于本地 Qdrant 知识库进行证据约束回答的问题，"
    "包括市场、趋势、公司、政策和尽调证据分析。优先输出更多可支撑报告的证据段、数据段和原文摘录。"
)

_FORCE_CPU_AFTER_CUDA_FAILURE = False
_RAG_AGENT_CACHE = TTLCache(
    ttl_seconds=int(os.getenv("RAG_AGENT_CACHE_TTL_SECONDS", "900") or "0"),
    max_items=int(os.getenv("RAG_AGENT_CACHE_MAX_ITEMS", "64") or "0"),
)
_CACHE_ARG_EXCLUDE = {"query", "query_text", "json", "session_id"}


class RagAgentState(TypedDict, total=False):
    messages: List[Dict[str, Any]]
    query: str
    session_id: str
    args_overrides: Dict[str, Any]
    answer_text: str
    raw_output: Dict[str, Any]
    evidence: List[Dict[str, Any]]
    trace_file: str
    timings: Dict[str, Any]
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


def extract_query_from_state(state: RagAgentState) -> str:
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


def build_rag_args(overrides: Optional[Dict[str, Any]] = None) -> argparse.Namespace:
    global _FORCE_CPU_AFTER_CUDA_FAILURE
    args = build_arg_parser().parse_args([])
    args.answer_only = True
    args.show_evidence = True
    args.enable_llm_synthesis = True
    args.json = False
    for key, value in dict(overrides or {}).items():
        if hasattr(args, key):
            setattr(args, key, value)
    if _FORCE_CPU_AFTER_CUDA_FAILURE and not dict(overrides or {}).get("device"):
        args.device = "cpu"
        args.dtype = "float32"
        if hasattr(args, "bge_m3_device"):
            args.bge_m3_device = "cpu"
        if hasattr(args, "bge_m3_use_fp16"):
            args.bge_m3_use_fp16 = False
    return args


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _args_cache_payload(args: argparse.Namespace) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for key, value in sorted(vars(args).items()):
        normalized_key = str(key).lower()
        if key in _CACHE_ARG_EXCLUDE or "api_key" in normalized_key:
            continue
        payload[key] = value
    return payload


def _rag_cache_key(query: str, args: argparse.Namespace) -> str:
    return make_cache_key(
        "rag_agent",
        {
            "query": query,
            "args": _args_cache_payload(args),
        },
    )


def _mark_cache_hit(state: RagAgentState, cache_name: str) -> RagAgentState:
    cached = dict(state)
    cached["metadata"] = {
        **dict(cached.get("metadata") or {}),
        "cache_hit": cache_name,
    }
    cached["timings"] = {
        **dict(cached.get("timings") or {}),
        "cache_hit": True,
    }
    raw_output = dict(cached.get("raw_output") or {})
    if raw_output:
        raw_output["cache_hit"] = cache_name
        cached["raw_output"] = raw_output
    return cached


def _is_cuda_embedding_error(error: BaseException | str) -> bool:
    text = str(error or "").lower()
    return "cuda" in text and ("embedding" in text or "qwen" in text or "kernel" in text or "out of memory" in text)


def _sanitize_rag_error(error: BaseException | str) -> str:
    text = str(error or "").strip()
    if _is_cuda_embedding_error(text):
        return (
            "本地 RAG 向量化在 CUDA 上失败，已触发 CPU 降级重试；"
            "如仍失败，请检查显卡驱动/CUDA 状态，或将 QWEN3_EMBEDDING_DEVICE=cpu 写入 .env。"
        )
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), text)
    first_line = re.sub(r"\s+", " ", first_line).strip()
    return first_line[:500]


def _build_success_state(
    *,
    state: RagAgentState,
    args: argparse.Namespace,
    raw_output: Dict[str, Any],
    fallback_note: str = "",
) -> RagAgentState:
    answer_text = format_answer_only_output(
        raw_output,
        show_evidence=bool(getattr(args, "show_evidence", True)),
        evidence_top_k=max(1, int(getattr(args, "answer_evidence_top_k", 12))),
    )
    metadata = {
        **dict(state.get("metadata") or {}),
        "agent_stage": "rag_core",
        "grounding_mode": (raw_output.get("answer") or {}).get("grounding_mode", ""),
        "llm_model": (raw_output.get("answer") or {}).get("llm_model", ""),
    }
    if fallback_note:
        metadata["runtime_fallback"] = fallback_note
    return {
        "answer_text": answer_text,
        "raw_output": raw_output,
        "evidence": list(raw_output.get("evidence") or []),
        "trace_file": str(raw_output.get("trace_file") or ""),
        "timings": dict(raw_output.get("timings") or {}),
        "metadata": metadata,
    }


def prepare_query_node(state: RagAgentState) -> RagAgentState:
    query = extract_query_from_state(state)
    if not query:
        return {
            "errors": list(state.get("errors") or []) + ["查询不能为空"],
            "metadata": {
                **dict(state.get("metadata") or {}),
                "agent_name": AGENT_NAME,
                "framework": "langgraph",
                "agent_stage": "prepare_query",
            },
        }
    return {
        "query": query,
        "metadata": {
            **dict(state.get("metadata") or {}),
            "agent_name": AGENT_NAME,
            "agent_description": AGENT_DESCRIPTION,
            "framework": "langgraph",
            "capabilities": [
                "industry_research_rag",
                "grounded_answering",
                "local_qdrant_retrieval",
            ],
            "agent_stage": "prepare_query",
        },
    }


def run_rag_core_node(state: RagAgentState) -> RagAgentState:
    global _FORCE_CPU_AFTER_CUDA_FAILURE
    if state.get("errors"):
        return {}
    query = str(state.get("query") or "").strip()
    args = build_rag_args(state.get("args_overrides") or {})
    if state.get("session_id"):
        args.session_id = str(state.get("session_id") or "").strip()
    cache_key = ""
    cache_allowed = _env_flag("RAG_AGENT_CACHE_ENABLED", True) and not str(getattr(args, "session_id", "") or "").strip()
    if cache_allowed:
        cache_key = _rag_cache_key(query, args)
        cached_state = _RAG_AGENT_CACHE.get(cache_key)
        if cached_state:
            return _mark_cache_hit(cached_state, "rag_agent")
    try:
        raw_output = run_search(args, query)
        success_state = _build_success_state(state=state, args=args, raw_output=raw_output)
        if cache_key:
            _RAG_AGENT_CACHE.set(cache_key, success_state)
        return success_state
    except Exception as exc:
        first_error = _sanitize_rag_error(exc)
        if _env_flag("RAG_CUDA_TO_CPU_FALLBACK", True) and _is_cuda_embedding_error(exc) and str(getattr(args, "device", "")).lower() != "cpu":
            _FORCE_CPU_AFTER_CUDA_FAILURE = True
            fallback_args = copy.copy(args)
            fallback_args.device = "cpu"
            fallback_args.dtype = "float32"
            if hasattr(fallback_args, "bge_m3_device"):
                fallback_args.bge_m3_device = "cpu"
            if hasattr(fallback_args, "bge_m3_use_fp16"):
                fallback_args.bge_m3_use_fp16 = False
            if hasattr(fallback_args, "no_embedder_cache"):
                fallback_args.no_embedder_cache = True
            try:
                raw_output = run_search(fallback_args, query)
                success_state = _build_success_state(
                    state=state,
                    args=fallback_args,
                    raw_output=raw_output,
                    fallback_note="CUDA embedding 失败后已自动降级到 CPU。",
                )
                if cache_key:
                    _RAG_AGENT_CACHE.set(cache_key, success_state)
                return success_state
            except Exception as fallback_exc:
                return {
                    "errors": list(state.get("errors") or [])
                    + [f"{first_error} CPU 降级重试仍失败：{_sanitize_rag_error(fallback_exc)}"],
                    "metadata": {
                        **dict(state.get("metadata") or {}),
                        "agent_stage": "rag_core",
                        "runtime_fallback": "cuda_to_cpu_failed",
                    },
                }
        return {
            "errors": list(state.get("errors") or []) + [first_error],
            "metadata": {
                **dict(state.get("metadata") or {}),
                "agent_stage": "rag_core",
            },
        }


def format_agent_response_node(state: RagAgentState) -> RagAgentState:
    errors = list(state.get("errors") or [])
    if errors:
        answer_text = "RAG Agent 失败：" + _sanitize_rag_error(errors[-1])
    else:
        answer_text = str(state.get("answer_text") or "").strip() or "当前没有生成可用答案。"

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


def create_rag_agent_graph(*, name: str = AGENT_NAME):
    builder = StateGraph(RagAgentState)
    builder.add_node("prepare_query", prepare_query_node)
    builder.add_node("rag_core", run_rag_core_node)
    builder.add_node("format_response", format_agent_response_node)
    builder.add_edge(START, "prepare_query")
    builder.add_edge("prepare_query", "rag_core")
    builder.add_edge("rag_core", "format_response")
    builder.add_edge("format_response", END)
    return builder.compile(name=name)


def run_rag_agent(
    query: str,
    *,
    messages: Optional[Sequence[Dict[str, Any]]] = None,
    session_id: str = "",
    args_overrides: Optional[Dict[str, Any]] = None,
) -> RagAgentState:
    graph = create_rag_agent_graph()
    initial_messages = list(messages or [])
    if query and not initial_messages:
        initial_messages.append({"role": "user", "content": query})
    state: RagAgentState = {
        "query": query,
        "messages": initial_messages,
        "session_id": session_id,
        "args_overrides": dict(args_overrides or {}),
    }
    return graph.invoke(state)


def rag_agent_tool(query: str, **kwargs: Any) -> str:
    """供 LangGraph supervisor 或其他智能体调用的工具式入口。"""

    result = run_rag_agent(query, args_overrides=kwargs)
    return str(result.get("answer_text") or "")


def create_rag_agent_tool():
    """返回可被 supervisor 风格多智能体图调用的 LangChain 兼容工具。"""

    from langchain_core.tools import tool

    @tool("industry_rag_agent", description=AGENT_DESCRIPTION)
    def _industry_rag_agent(query: str) -> str:
        return rag_agent_tool(query)

    return _industry_rag_agent


def namespace_to_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    values = vars(copy.copy(args))
    for key in ["query", "query_text", "json"]:
        values.pop(key, None)
    return values


def parse_query_from_args(args: argparse.Namespace) -> str:
    if args.query_text:
        return " ".join(str(item) for item in args.query_text).strip()
    return " ".join(str(item) for item in args.query).strip()


def main() -> int:
    args = build_arg_parser().parse_args()
    query = parse_query_from_args(args)
    if not query:
        query = input("Enter query: ").strip()
    if not query:
        raise RuntimeError("查询不能为空。")

    overrides = namespace_to_overrides(args)
    state = run_rag_agent(
        query,
        session_id=str(getattr(args, "session_id", "") or "").strip(),
        args_overrides=overrides,
    )
    if getattr(args, "json", False):
        print(json.dumps(state, ensure_ascii=False, indent=2, default=str))
    else:
        print(str(state.get("answer_text") or "").strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
