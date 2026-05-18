from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

PIPELINE_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = PIPELINE_ROOT.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from rag_pipeline.runtime_cache import json_safe_default
from rag_pipeline.logging_utils import configure_pipeline_logging
from rag_pipeline.agents.public_report_sanitizer import find_publication_blockers, sanitize_public_markdown


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key and not os.environ.get(key):
            os.environ[key] = value


def safe_filename(value: str, *, max_chars: int = 80) -> str:
    text = re.sub(r"\s+", "_", str(value or "").strip())
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", text)
    text = re.sub(r"_+", "_", text).strip("._")
    return (text or "report")[:max_chars]


def as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def json_default(value: Any) -> Any:
    return json_safe_default(value)


QUIET_STAGE_LOGS = False


class OverallProgress:
    def __init__(self, *, enabled: bool, stream: Any = sys.stderr, width: int = 30) -> None:
        self.enabled = enabled
        self.stream = stream
        self.width = max(10, width)
        self.started_at = time.perf_counter()
        self.percent = 0.0
        self.label = ""
        self._lock = threading.Lock()
        self._pulse_stop: Optional[threading.Event] = None
        self._pulse_thread: Optional[threading.Thread] = None
        self._last_len = 0
        self._interactive = bool(getattr(stream, "isatty", lambda: False)())
        self._heartbeat_seconds = max(1.0, float(os.getenv("REPORT_PROGRESS_HEARTBEAT_SECONDS", "10") or 10))
        self._pulse_expected_seconds = max(60.0, float(os.getenv("REPORT_PROGRESS_PULSE_EXPECTED_SECONDS", "3600") or 3600))

    def _line(self, percent: float, label: str) -> str:
        percent = max(0.0, min(100.0, percent))
        filled = int(round(self.width * percent / 100.0))
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = time.perf_counter() - self.started_at
        return f"[{bar}] {percent:5.1f}%  {label}  elapsed {elapsed:5.1f}s"

    def _render_locked(self, *, newline: bool = False) -> None:
        if not self.enabled:
            return
        line = self._line(self.percent, self.label)
        if self._interactive:
            pad = " " * max(0, self._last_len - len(line))
            print("\r" + line + pad, end="\n" if newline else "", file=self.stream, flush=True)
            self._last_len = len(line)
            return
        print(line, file=self.stream, flush=True)

    def stop_pulse(self) -> None:
        event = self._pulse_stop
        thread = self._pulse_thread
        if event is not None:
            event.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._pulse_stop = None
        self._pulse_thread = None

    def update(self, percent: float, label: str) -> None:
        self.stop_pulse()
        with self._lock:
            self.percent = max(self.percent, min(100.0, float(percent)))
            self.label = str(label or self.label or "running")
            self._render_locked()

    def pulse_to(self, percent: float, label: str) -> None:
        if not self.enabled:
            return
        self.stop_pulse()
        target = max(self.percent, min(99.0, float(percent)))
        with self._lock:
            self.label = str(label or self.label or "running")
            self._render_locked()
        event = threading.Event()
        self._pulse_stop = event
        pulse_started_at = time.perf_counter()
        pulse_start_percent = self.percent

        def _pulse() -> None:
            interval = 0.6 if self._interactive else self._heartbeat_seconds
            while not event.wait(interval):
                with self._lock:
                    if self.percent < target:
                        elapsed = max(0.0, time.perf_counter() - pulse_started_at)
                        estimated = pulse_start_percent + (target - pulse_start_percent) * min(1.0, elapsed / self._pulse_expected_seconds)
                        self.percent = min(target, max(self.percent, estimated))
                    self._render_locked()

        thread = threading.Thread(target=_pulse, daemon=True)
        self._pulse_thread = thread
        thread.start()

    def finish(self, label: str = "done") -> None:
        self.stop_pulse()
        with self._lock:
            self.percent = 100.0
            self.label = label
            self._render_locked(newline=True)


def log(*values: Any, force: bool = False, **kwargs: Any) -> None:
    if QUIET_STAGE_LOGS and not force:
        return
    print(*values, file=sys.stderr, **kwargs)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pretty = env_flag("REPORT_PRETTY_JSON", False)
    separators = None if pretty else (",", ":")
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2 if pretty else None, separators=separators, default=json_default),
        encoding="utf-8",
    )


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(finalize_public_report(str(text or "")).strip() + "\n", encoding="utf-8")


def finalize_public_report(markdown: str) -> str:
    cleaned = sanitize_public_markdown(str(markdown or ""))
    for _ in range(3):
        blockers = find_publication_blockers(cleaned)
        if not blockers:
            break
        blocked_lines = {int(item.get("line") or 0) for item in blockers}
        cleaned = "\n".join(
            line
            for line_no, line in enumerate(cleaned.splitlines(), start=1)
            if line_no not in blocked_lines
        )
        cleaned = sanitize_public_markdown(cleaned)
    remaining = find_publication_blockers(cleaned)
    if remaining:
        sample = "; ".join(str(item.get("text") or "")[:80] for item in remaining[:3])
        raise ValueError(f"publication blockers remain after sanitization: {sample}")
    return cleaned.strip()


def llm_runtime_status() -> Dict[str, Any]:
    try:
        from rag_pipeline.config import search_config as cfg
        from rag_pipeline.search.memory import llm_config_is_ready, normalize_llm_config

        synthesis = normalize_llm_config(
            {
                "provider": cfg.DEFAULT_LLM_SYNTHESIS_PROVIDER,
                "url": cfg.DEFAULT_LLM_SYNTHESIS_URL,
                "api_key": cfg.DEFAULT_LLM_SYNTHESIS_API_KEY,
                "model": cfg.DEFAULT_LLM_SYNTHESIS_MODEL,
                "timeout": cfg.DEFAULT_LLM_SYNTHESIS_TIMEOUT,
                "disable_thinking": getattr(cfg, "DEFAULT_LLM_SYNTHESIS_DISABLE_THINKING", False),
            }
        )
        return {
            "active_profile": os.environ.get("RAG_LLM_ACTIVE_PROFILE", "").strip(),
            "execution_profile": os.environ.get("RAG_LLM_EXECUTION_PROFILE", "").strip()
            or os.environ.get("RAG_LLM_ACTIVE_PROFILE", "").strip(),
            "synthesis_ready": bool(llm_config_is_ready(synthesis)),
            "synthesis_model": synthesis.get("model") or "",
            "synthesis_url_set": bool(synthesis.get("url")),
            "synthesis_api_key_set": bool(synthesis.get("api_key")),
            "synthesis_disable_thinking": bool(getattr(cfg, "DEFAULT_LLM_SYNTHESIS_DISABLE_THINKING", False)),
            "flags": {
                "rag_llm_planner": env_flag("RAG_ENABLE_LLM_PLANNER", False),
                "rag_llm_synthesis": env_flag("RAG_ENABLE_LLM_SYNTHESIS", False),
                "rag_llm_answer_review": env_flag("RAG_ENABLE_LLM_ANSWER_REVIEW", False),
                "rag_llm_reflection": env_flag("RAG_ENABLE_LLM_REFLECTION", False),
                "iqs_llm_query_rewrite": env_flag("IQS_ENABLE_LLM_QUERY_REWRITE", False),
                "iqs_hyde": env_flag("IQS_ENABLE_HYDE", False),
                "brain_llm_research_planner": env_flag("BRAIN_ENABLE_LLM_RESEARCH_PLANNER", False),
                "brain_web_llm_analysis": env_flag("BRAIN_WEB_ENABLE_LLM_ANALYSIS", False),
                "brain_llm_merge": env_flag("BRAIN_ENABLE_LLM_MERGE", False),
                "brain_llm_coverage_eval": env_flag("BRAIN_ENABLE_LLM_COVERAGE_EVAL", False),
                "report_llm_rewrite": env_flag("REPORT_ENABLE_LLM_REWRITE", False),
            },
        }
    except Exception as exc:
        return {"error": str(exc)}


def has_legacy_decision_sections(markdown: str) -> bool:
    return bool(
        re.search(
            r"章节判断|关键事实速览|证据深读|原文事实|行业形势含义|投资/产品判断|与上下章节的联动|战略含义与行动建议|"
            r"全球口径|中国口径|增速口径|可引用事实|机制与边界|进入综合决策章的变量|核心判断[:：]|机制拆解|反证边界|决策含义[:：]",
            str(markdown or ""),
        )
    )


def env_large_int(name: str, default: int, *, min_value: int = 0, max_value: int = 1_000_000) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        value = default
    return min(max_value, max(min_value, value))


def markdown_heading_count(markdown: str) -> int:
    return len(re.findall(r"(?m)^#{1,3}\s+\S+", str(markdown or "")))


def markdown_dense_chars(markdown: str) -> int:
    text = re.sub(r"(?m)^#{1,6}\s+.*$", "", str(markdown or ""))
    return len(re.sub(r"\s+", "", text))


def reformatter_structure_loss_reason(clean_report: str, writer_report: str) -> str:
    clean = str(clean_report or "").strip()
    writer = str(writer_report or "").strip()
    if not clean or not writer:
        return ""
    clean_chars = markdown_dense_chars(clean)
    writer_chars = markdown_dense_chars(writer)
    if writer_chars < 8000:
        return ""
    clean_headings = markdown_heading_count(clean)
    writer_headings = markdown_heading_count(writer)
    target_chars = env_large_int("REPORT_TARGET_BODY_CHARS", 0)
    allow_condense = env_flag("REPORT_REFORMATTER_ALLOW_STRUCTURAL_CONDENSE", True)
    min_ratio_percent = env_large_int("REPORT_REFORMATTER_MIN_DENSE_RATIO_PERCENT", 45, min_value=20, max_value=90)
    min_accept_chars = env_large_int("REPORT_REFORMATTER_MIN_ACCEPT_CHARS", 12000, min_value=2000, max_value=100000)
    if clean_chars < int(writer_chars * (min_ratio_percent / 100.0)) and clean_chars < min_accept_chars:
        return f"正文压缩过多 clean={clean_chars} writer={writer_chars}"
    if not allow_condense and writer_headings >= 10 and clean_headings < max(6, int(writer_headings * 0.55)):
        return f"章节层级丢失 clean_headings={clean_headings} writer_headings={writer_headings}"
    if (
        target_chars
        and not allow_condense
        and writer_chars >= int(target_chars * 0.70)
        and clean_chars < int(target_chars * 0.60)
    ):
        return f"未达到目标正文量 clean={clean_chars} target={target_chars}"
    return ""


def stage_status(state: Dict[str, Any]) -> Dict[str, bool]:
    raw_output = as_dict(state.get("raw_output"))
    writer_report = as_dict(state.get("writer_report")) or as_dict(raw_output.get("writer_report"))
    return {
        "question_analysis": bool(as_dict(state.get("query_analysis")) or as_dict(raw_output.get("query_analysis"))),
        "question_decomposition": bool(
            as_list(as_dict(state.get("query_analysis")).get("related_questions"))
            or as_dict(as_dict(state.get("query_analysis")).get("agent_queries"))
            or as_list(as_dict(raw_output.get("query_analysis")).get("related_questions"))
            or as_dict(as_dict(raw_output.get("query_analysis")).get("agent_queries"))
        ),
        "child_agents": bool(as_dict(raw_output.get("child_outputs"))),
        "evidence_merger": bool(as_dict(state.get("evidence_package")) or as_dict(raw_output.get("evidence_package"))),
        "analysis_agent": bool(as_dict(state.get("structured_analysis")) or as_dict(raw_output.get("structured_analysis"))),
        "writer_agent": bool(writer_report.get("report_markdown")),
    }


def missing_required_stages(status: Dict[str, bool]) -> List[str]:
    required = [
        "question_analysis",
        "question_decomposition",
        "child_agents",
        "evidence_merger",
        "analysis_agent",
        "writer_agent",
    ]
    return [name for name in required if not status.get(name)]


def _compact_error_text(value: Any, *, max_chars: int = 260) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."


def compact_errors(values: Iterable[Any], *, max_items: int = 8) -> List[str]:
    errors: List[str] = []
    for value in values:
        text = _compact_error_text(value)
        if text and text not in errors:
            errors.append(text)
        if len(errors) >= max_items:
            break
    return errors


def env_int(name: str, default: int, *, min_value: int = 1, max_value: int = 100) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        value = default
    return min(max_value, max(min_value, value))


def env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def normalize_llm_profile(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", str(value or "").strip().upper()).strip("_")


def llm_profile_env_name(profile: str, field: str) -> str:
    profile_key = normalize_llm_profile(profile)
    field_key = normalize_llm_profile(field)
    return f"RAG_LLM_PROFILE_{profile_key}_{field_key}" if profile_key and field_key else ""


def available_llm_profiles() -> List[str]:
    raw = os.environ.get("RAG_LLM_PROFILES", "qwen,deepseek-v4-pro")
    profiles = [item.strip() for item in raw.split(",") if item.strip()]
    for current in (
        str(os.environ.get("RAG_LLM_EXECUTION_PROFILE") or "").strip(),
        str(os.environ.get("RAG_LLM_ACTIVE_PROFILE") or "").strip(),
    ):
        if current and current not in profiles:
            profiles.insert(0, current)
    return profiles


def llm_profile_config_status(profile: str) -> Dict[str, str]:
    return {
        "provider": os.environ.get(llm_profile_env_name(profile, "PROVIDER"), "").strip(),
        "url": os.environ.get(llm_profile_env_name(profile, "URL"), "").strip(),
        "api_key": os.environ.get(llm_profile_env_name(profile, "API_KEY"), "").strip(),
        "model": os.environ.get(llm_profile_env_name(profile, "MODEL"), "").strip(),
        "timeout": os.environ.get(llm_profile_env_name(profile, "TIMEOUT"), "").strip(),
        "disable_thinking": os.environ.get(llm_profile_env_name(profile, "DISABLE_THINKING"), "").strip(),
    }


LLM_EXECUTION_PROFILE_ENV = "RAG_LLM_EXECUTION_PROFILE"
LLM_EXECUTION_CONFIG_PREFIX = "RAG_LLM_SYNTHESIS"


def apply_llm_profile_to_environment(profile: str) -> None:
    status = llm_profile_config_status(profile)
    provider = status.get("provider") or "openai_compatible"
    timeout = status.get("timeout") or "180"
    prefix = LLM_EXECUTION_CONFIG_PREFIX
    os.environ[f"{prefix}_PROVIDER"] = provider
    os.environ[f"{prefix}_URL"] = status["url"]
    os.environ[f"{prefix}_API_KEY"] = status["api_key"]
    os.environ[f"{prefix}_MODEL"] = status["model"]
    os.environ[f"{prefix}_TIMEOUT"] = timeout
    disable_thinking_env = f"{prefix}_DISABLE_THINKING"
    if status.get("disable_thinking"):
        os.environ[disable_thinking_env] = status["disable_thinking"]
    else:
        os.environ.pop(disable_thinking_env, None)


def select_llm_profile(args: argparse.Namespace) -> str:
    selected = str(
        args.llm_profile
        or os.environ.get(LLM_EXECUTION_PROFILE_ENV)
        or os.environ.get("RAG_LLM_ACTIVE_PROFILE")
        or ""
    ).strip()
    should_prompt = bool((args.select_llm or env_flag("REPORT_SELECT_LLM_PROFILE", False)) and not args.llm_profile)
    if should_prompt:
        if args.no_interactive_input:
            raise RuntimeError("--select-llm 需要 stdin；请改用 --llm-profile 指定模型。")
        profiles = available_llm_profiles()
        if not profiles:
            raise RuntimeError("RAG_LLM_PROFILES 为空，无法选择执行大模型。")
        print("可选执行大模型：", file=sys.stderr)
        for index, profile in enumerate(profiles, 1):
            status = llm_profile_config_status(profile)
            configured = all(status.get(field) for field in ("url", "api_key", "model"))
            state_tags = []
            if profile == selected:
                state_tags.append("当前默认")
            state_tags.append("已配置" if configured else "未配置")
            suffix = f" ({'，'.join(state_tags)})"
            model = status.get("model") or "model未填"
            print(f"  {index}. {profile} - {model}{suffix}", file=sys.stderr)
        choice = input(f"请选择执行大模型 [默认 {selected or profiles[0]}]：").strip()
        if choice:
            if choice.isdigit() and 1 <= int(choice) <= len(profiles):
                selected = profiles[int(choice) - 1]
            else:
                selected = choice
        elif not selected:
            selected = profiles[0]
    if selected:
        os.environ[LLM_EXECUTION_PROFILE_ENV] = selected
        status = llm_profile_config_status(selected)
        missing = [field for field in ("url", "api_key", "model") if not status.get(field)]
        if missing:
            env_names = ", ".join(llm_profile_env_name(selected, field) for field in missing)
            raise RuntimeError(f"LLM profile '{selected}' 配置不完整，请在 .env 填写：{env_names}")
        apply_llm_profile_to_environment(selected)
    return selected


def strict_quality_mode() -> bool:
    mode = str(os.environ.get("REPORT_QUALITY_MODE") or os.environ.get("QUALITY_MODE") or "strict").strip().lower()
    if mode in {"speed", "fast", "loose", "draft"}:
        return False
    raw = os.environ.get("STRICT_EVIDENCE_MODE")
    if raw is None:
        return True
    return str(raw).strip().lower() in {"1", "true", "yes", "on", "strict"}


def continuous_evidence_loop_mode() -> bool:
    raw = os.environ.get("REPORT_CONTINUOUS_EVIDENCE_LOOP")
    if raw is not None:
        return str(raw).strip().lower() in {"1", "true", "yes", "on", "strict"}
    return strict_quality_mode()


_STATE_LIST_LIMITS = {
    "messages": 2,
    "search_results": 40,
    "page_results": 10,
    "raw_data_points": 120,
    "evidence_pool": 160,
    "evidence": 120,
    "key_sources": 120,
    "search_trace": 80,
    "query_plan": 160,
    "search_tasks": 200,
    "follow_up_queries": 80,
    "followup_results": 80,
    "layout_refinement_trace": 40,
    "self_refine_trace": 40,
    "loop_trace": 40,
}


_STATE_TEXT_LIMITS = {
    "answer_text": 12000,
    "report_markdown": 12000,
    "content": 3000,
    "mainText": 3000,
    "snippet": 2000,
    "summary": 2000,
    "evidence": 6000,
    "raw_output": 6000,
}


def _state_list_limit(key: str) -> int:
    return _STATE_LIST_LIMITS.get(key, env_int("REPORT_STATE_MAX_LIST_ITEMS", 80, max_value=500))


def _state_text_limit(key: str) -> int:
    return _STATE_TEXT_LIMITS.get(key, env_int("REPORT_STATE_MAX_TEXT_CHARS", 6000, max_value=50000))


def compact_state_for_disk(value: Any, *, key: str = "", depth: int = 0) -> Any:
    """Keep debug state useful while avoiding hundreds of MB of repeated raw outputs."""

    if env_flag("REPORT_SAVE_FULL_STATE", False):
        return value
    if depth > 10:
        return "[state compacted: max depth reached]"
    if isinstance(value, dict):
        compacted: Dict[str, Any] = {}
        for item_key, item_value in value.items():
            text_key = str(item_key)
            compacted[text_key] = compact_state_for_disk(item_value, key=text_key, depth=depth + 1)
        return compacted
    if isinstance(value, list):
        limit = _state_list_limit(key)
        kept = [compact_state_for_disk(item, key=key, depth=depth + 1) for item in value[:limit]]
        if len(value) > limit:
            kept.append(
                {
                    "_truncated": True,
                    "original_count": len(value),
                    "kept_count": limit,
                    "hint": "Set REPORT_SAVE_FULL_STATE=1 to keep the complete debug snapshot.",
                }
            )
        return kept
    if isinstance(value, str):
        limit = _state_text_limit(key)
        if len(value) > limit:
            return (
                value[:limit]
                + f"\n\n[state compacted: text truncated from {len(value)} to {limit} chars; "
                + "full report is saved as Markdown and writer_package keeps evidence inputs]"
            )
        return value
    return value


def write_state_json(path: Path, payload: Dict[str, Any]) -> None:
    write_json(path, compact_state_for_disk(payload))


def full_report_iqs_options() -> Dict[str, Any]:
    """Use the active IQS profile instead of forcing the old quality-first funnel."""

    default_max_queries = env_int("IQS_INITIAL_MAX_QUERIES", 1, max_value=10)
    default_max_tasks = env_int("IQS_INITIAL_MAX_SEARCH_TASKS", 2, max_value=40)
    default_results = env_int("IQS_INITIAL_RESULTS_PER_QUERY", 20, max_value=100)
    default_top_k = env_int("IQS_INITIAL_RERANK_TOP_K", 8, max_value=80)
    default_max_docs = env_int("IQS_INITIAL_RERANK_MAX_DOCS", 30, max_value=100)

    options = {
        "search_profile": "initial",
        "max_queries": env_int("FULL_REPORT_IQS_MAX_QUERIES", default_max_queries, max_value=10),
        "max_search_tasks": env_int("FULL_REPORT_IQS_MAX_SEARCH_TASKS", default_max_tasks, max_value=40),
        "results_per_query": env_int("FULL_REPORT_IQS_RESULTS_PER_QUERY", default_results, max_value=100),
        "rerank_top_k": env_int("FULL_REPORT_IQS_RERANK_TOP_K", default_top_k, max_value=80),
        "rerank_max_docs": env_int("FULL_REPORT_IQS_RERANK_MAX_DOCS", default_max_docs, max_value=100),
        "rerank_prefilter_max_docs": env_int("FULL_REPORT_IQS_RERANK_PREFILTER_MAX_DOCS", default_max_docs, max_value=100),
        "enable_self_refine": env_flag("FULL_REPORT_IQS_ENABLE_SELF_REFINE", env_flag("IQS_ENABLE_SELF_REFINE", True)),
        "enable_batch_search": env_flag("IQS_ENABLE_BATCH_SEARCH", True),
    }
    if continuous_evidence_loop_mode():
        floors = {
            "max_queries": 6,
            "max_search_tasks": 24,
            "results_per_query": 80,
            "rerank_top_k": 40,
            "rerank_max_docs": 100,
            "rerank_prefilter_max_docs": 100,
        }
        for key, floor in floors.items():
            options[key] = max(int(options.get(key) or 0), floor)
        options["enable_self_refine"] = True
    return options


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full industry-research flow: question analysis, decomposition, "
            "RAG/IQS agents, Evidence Merger, Analysis Agent, and Writer Agent report generation."
        )
    )
    parser.add_argument("query", nargs="*", help="报告问题/主题，例如：智能机器人行业投资机会")
    parser.add_argument("--query", dest="query_option", default="", help="报告问题/主题。")
    parser.add_argument("--route", choices=["all", "both", "auto", "web", "local"], default="all", help="默认 all，确保 RAG 与 6 路 IQS 都参与。")
    parser.add_argument("--llm-profile", default="", help="选择本次报告执行大模型 profile，例如 qwen、deepseek-v4-pro。")
    parser.add_argument("--select-llm", action="store_true", help="运行前在终端列出 RAG_LLM_PROFILES 并交互选择执行大模型。")
    parser.add_argument("--output-dir", default=str(PIPELINE_ROOT / "output" / "full_reports"), help="状态和调试文件输出目录。")
    parser.add_argument("--session-id", default="", help="可选 session id。")
    parser.add_argument("--supervisor-max-loops", type=int, default=env_int("BRAIN_SUPERVISOR_MAX_LOOPS", 1, max_value=5), help="默认 3 轮补证，优先保证深度行研覆盖。")
    parser.add_argument("--supervisor-max-followup-queries", type=int, default=env_int("BRAIN_SUPERVISOR_MAX_FOLLOWUP_QUERIES", 2, max_value=10), help="每轮最多补充问题数。")
    parser.add_argument("--supervisor-min-coverage-gain", type=float, default=0.05, help="补证覆盖率提升阈值。")
    parser.add_argument("--include-raw-child-states", action="store_true", help="额外保存 RAG/IQS 原始状态，文件会更大。")
    parser.add_argument("--save-full-state", action="store_true", help="保存完整未压缩 state JSON；只建议定位深层调试问题时使用。")
    parser.add_argument("--no-interactive-input", action="store_true", help="没有传 query 时直接失败，不进入终端输入等待。")
    parser.add_argument("--print-report", action="store_true", help="兼容旧参数；报告正文现在默认输出到 stdout，不再生成 md 文件。")
    parser.add_argument("--skip-review", action="store_true", help="跳过 ReviewAgent 终审；默认启用规则审查。")
    parser.add_argument("--enable-llm-review", action="store_true", help="启用 ReviewAgent 的 LLM 精修层；默认只跑规则层。")
    parser.add_argument("--skip-reformatter", action="store_true", help="跳过 ReformatterAgent，回退到旧 WriterAgent/ReviewAgent 输出路径。")
    parser.add_argument("--reformatter-output", default="", help="可选：指定 ReformatterAgent 洁净报告输出路径。")
    parser.add_argument("--no-progress-bar", action="store_true", help="关闭整体进度条，恢复普通阶段日志。")
    parser.add_argument("--verbose-progress", action="store_true", help="保留内部详细进度日志；默认只显示整体进度条。")
    parser.add_argument("--allow-missing-stage", action="store_true", help="即使阶段产物缺失也返回 0；默认会失败退出。")
    return parser

def main() -> int:
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    load_dotenv(PIPELINE_ROOT / ".env")
    configure_pipeline_logging()
    args = build_arg_parser().parse_args()
    selected_llm_profile = select_llm_profile(args)
    progress_enabled = (not args.no_progress_bar) and env_flag("REPORT_PROGRESS_BAR", True)
    global QUIET_STAGE_LOGS
    QUIET_STAGE_LOGS = bool(progress_enabled and not args.verbose_progress)
    if progress_enabled and not args.verbose_progress:
        os.environ["PIPELINE_PROGRESS"] = "0"
    else:
        os.environ.setdefault("PIPELINE_PROGRESS", "1")
    progress = OverallProgress(enabled=progress_enabled)
    progress.update(1, "准备参数")
    if selected_llm_profile:
        log(f"[0/6] 执行大模型 profile: {selected_llm_profile}")

    if args.save_full_state:
        os.environ["REPORT_SAVE_FULL_STATE"] = "1"
    query = (args.query_option or " ".join(args.query)).strip()
    if not query:
        if args.no_interactive_input or env_flag("REPORT_NO_INTERACTIVE_INPUT", False):
            raise RuntimeError("Query cannot be empty. 请使用 --query 传入报告问题。")
        log("[0/6] 未检测到 --query，等待你在终端输入报告问题/主题；也可以用 --query \"你的问题\" 直接启动")
        try:
            query = input("请输入报告问题/主题：").strip()
        except EOFError as exc:
            raise RuntimeError("Query cannot be empty. 当前运行环境没有可用 stdin，请使用 --query 传入报告问题。") from exc
    if not query:
        raise RuntimeError("Query cannot be empty.")

    from rag_pipeline.agents.brain_agent import run_brain_agent

    pipeline_started = time.perf_counter()
    progress.update(5, "问题分析与任务规划")
    log("[1/6] 问题分析与拆解启动")
    log("[2/6] 调度 RAG 与 6 路 IQS Agent")
    log("[3/6] Evidence Merger / Analysis Agent / Writer Agent 将在 merge 阶段串行执行")

    progress.pulse_to(72, "检索 / 证据 / 正文生成")
    continuous_loop = continuous_evidence_loop_mode()
    state = run_brain_agent(
        query=query,
        route=args.route,
        session_id=args.session_id,
        web_search_options=full_report_iqs_options(),
        enable_web_analysis=env_flag("BRAIN_WEB_ENABLE_LLM_ANALYSIS", False),
        enable_llm_merge=env_flag("BRAIN_ENABLE_LLM_MERGE", False),
        enable_followup_loop=continuous_loop or env_flag("BRAIN_ENABLE_FOLLOWUP_LOOP", False),
        supervisor_max_loops=max(args.supervisor_max_loops, 5) if continuous_loop else args.supervisor_max_loops,
        supervisor_min_coverage_gain=args.supervisor_min_coverage_gain,
        supervisor_max_followup_queries=max(args.supervisor_max_followup_queries, 8) if continuous_loop else args.supervisor_max_followup_queries,
        layout_max_refinement_rounds=max(env_int("BRAIN_LAYOUT_MAX_REFINEMENT_ROUNDS", 3, max_value=6), 3) if continuous_loop else None,
        output_mode="writer_markdown",
        parallel_raw_output=bool(args.include_raw_child_states),
    )
    progress.update(72, "主体报告生成完成")
    log(f"[3/6] Brain 主流程完成，用时 {time.perf_counter() - pipeline_started:.1f}s")

    state_dict = dict(state or {})
    raw_output = as_dict(state_dict.get("raw_output"))
    writer_report = as_dict(state_dict.get("writer_report")) or as_dict(raw_output.get("writer_report"))
    report_markdown = str(writer_report.get("report_markdown") or state_dict.get("answer_text") or "").strip()
    writer_status = str(writer_report.get("report_status") or "").strip().lower()
    writer_not_ready = writer_status == "not_ready" or bool(writer_report.get("skip_reformatter"))
    writer_publishable = bool(report_markdown) and writer_status == "final" and not writer_not_ready
    if not report_markdown:
        reformatter_skip_reason = "no_report_markdown"
    elif writer_not_ready:
        reformatter_skip_reason = "writer_not_ready"
    elif not writer_publishable:
        reformatter_skip_reason = f"report_status_{writer_status or 'unknown'}"
    elif args.skip_reformatter:
        reformatter_skip_reason = "skip_reformatter_arg"
    else:
        reformatter_skip_reason = ""
    review_result: Dict[str, Any] = {}
    reformatter_result: Dict[str, Any] = {}

    if report_markdown and args.skip_reformatter and not args.skip_review and not writer_not_ready:
        from .review_pipeline import run_review_pipeline_sync

        progress.pulse_to(82, "ReviewAgent 审查")
        log("[5/6] ReviewAgent 审查报告中")
        review_result = run_review_pipeline_sync(
            writer_output=report_markdown,
            llm_client=None,
            skip_llm_review=not args.enable_llm_review,
        )
        report_markdown = finalize_public_report(str(review_result.get("final_report") or report_markdown))
        writer_report["report_markdown"] = report_markdown
        writer_report["review_audit"] = as_dict(review_result.get("stage1_audit"))
        writer_report["review_stage2_skipped"] = bool(review_result.get("stage2_skipped", True))
        writer_report["review_total_fixes"] = int(review_result.get("total_fixes") or 0)
        state_dict["writer_report"] = writer_report
        state_dict["answer_text"] = report_markdown

        audit = as_dict(review_result.get("stage1_audit"))
        log(f"  [ReviewAgent] 修复泄露文本: {len(as_list(audit.get('leak_patterns_removed')))} 处")
        log(f"  [ReviewAgent] 删除重复 bullet: {int(audit.get('duplicate_bullets_removed') or 0)} 处")
        log(f"  [ReviewAgent] 删除重复段落: {int(audit.get('duplicate_paragraphs_removed') or 0)} 处")
        log(f"  [ReviewAgent] 修复/填充空节: {len(as_list(audit.get('empty_sections_filled')))} 处")
        if review_result.get("stage2_skipped"):
            log(f"  [ReviewAgent] LLM 精修跳过: {review_result.get('stage2_reason') or 'not enabled'}")
        if as_list(audit.get("truncated_content")):
            log(f"  [WARN] 截断/无意义内容: {len(as_list(audit.get('truncated_content')))} 处")
        progress.update(82, "ReviewAgent 审查完成")

    progress.update(86, "写入报告文件")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir).resolve()
    base_name = f"{timestamp}_{safe_filename(query)}"
    state_path = output_dir / f"{base_name}.state.json"
    package_path = output_dir / f"{base_name}.writer_package.json"
    writer_md_path = output_dir / f"{base_name}.writer.md"

    state_dict["writer_package_path"] = str(package_path)
    if report_markdown and not writer_not_ready:
        report_markdown = finalize_public_report(report_markdown)
        writer_report["report_markdown"] = report_markdown
        state_dict["answer_text"] = report_markdown
        write_markdown(writer_md_path, report_markdown)
        writer_report["writer_markdown_path"] = str(writer_md_path)
        state_dict["writer_markdown_path"] = str(writer_md_path)
        state_dict["writer_report"] = writer_report
    write_state_json(state_path, state_dict)
    llm_status = llm_runtime_status()
    reformatter_result = {
        "enabled": bool(writer_publishable and not args.skip_reformatter),
        "status": "pending" if writer_publishable and not args.skip_reformatter else "skipped",
        "skipped_reason": reformatter_skip_reason,
        "llm_runtime": llm_status,
    }
    writer_package_payload = {
        "query": query,
        "stage_status": stage_status(state_dict),
        "llm_runtime": llm_status,
        "evidence_package": as_dict(state_dict.get("evidence_package")) or as_dict(raw_output.get("evidence_package")),
        "structured_analysis": as_dict(state_dict.get("structured_analysis")) or as_dict(raw_output.get("structured_analysis")),
        "report_blueprint": as_dict(state_dict.get("report_blueprint")) or as_dict(raw_output.get("report_blueprint")),
        "chapter_evidence_packages": as_list(state_dict.get("chapter_evidence_packages")) or as_list(raw_output.get("chapter_evidence_packages")),
        "micro_layouts": as_list(state_dict.get("micro_layouts")) or as_list(raw_output.get("micro_layouts")),
        "table_packages": as_list(state_dict.get("table_packages")) or as_list(raw_output.get("table_packages")),
        "argument_units": as_list(state_dict.get("argument_units")) or as_list(raw_output.get("argument_units")),
        "chapter_packages": as_list(state_dict.get("chapter_packages")) or as_list(raw_output.get("chapter_packages")),
        "writer_report": writer_report,
        "review_result": review_result,
        "reformatter_result": reformatter_result,
    }
    write_json(package_path, writer_package_payload)

    status = stage_status(state_dict)
    missing = missing_required_stages(status)
    errors = compact_errors(as_list(state_dict.get("errors")) + as_list(raw_output.get("writer_errors")) + as_list(raw_output.get("analysis_errors")))

    progress.update(90, "阶段产物校验")
    log("[4/6] 阶段校验")
    for name, ok in status.items():
        log(f"  - {name}: {'OK' if ok else 'MISSING'}")

    if not writer_publishable:
        reformatter_result = {
            **reformatter_result,
            "enabled": False,
            "success": False,
            "status": "skipped",
            "skipped_reason": reformatter_skip_reason,
        }
        writer_package_payload["reformatter_result"] = reformatter_result
        write_json(package_path, writer_package_payload)
        if writer_not_ready:
            log("[5/6] 证据门槛未达成，跳过 ReformatterAgent，只输出补证清单")
            progress.update(96, "已阻断正式报告")
        else:
            log(f"[5/6] WriterAgent 状态为 {writer_status or 'unknown'}，跳过 ReformatterAgent，保留 review draft")
            progress.update(96, "保留待复核草稿")
    elif not args.skip_reformatter:
        progress.pulse_to(96, "ReformatterAgent 清洗报告")
        log("[5/6] ReformatterAgent 从 writer_package 重写洁净报告")
        clean_output_path = Path(args.reformatter_output).resolve() if args.reformatter_output else package_path.with_name(package_path.name.replace(".writer_package.json", "_clean.md"))
        try:
            from .evidence_extractor import extract_clean_evidence
            from .reformatter_agent import build_reformatter_repair_plan, run_reformatter, validate_reformatted_report

            writer_v3_markdown = str(writer_report.get("report_markdown") or report_markdown or "").strip()
            reformatter_result = {
                **reformatter_result,
                "enabled": True,
                "success": False,
                "status": "started",
                "output_path": str(clean_output_path),
            }
            writer_package_payload["reformatter_result"] = reformatter_result
            write_json(package_path, writer_package_payload)
            clean_evidence = extract_clean_evidence(str(package_path))
            clean_report = asyncio.run(run_reformatter(clean_evidence, llm_client=None))
            fallback_reason = ""
            if has_legacy_decision_sections(clean_report) and writer_v3_markdown and not has_legacy_decision_sections(writer_v3_markdown):
                log("  [WARN] ReformatterAgent 输出含固定小节模板，自动回退到 WriterAgent 动态正文")
                fallback_reason = "legacy_decision_sections"
                clean_report = writer_v3_markdown
            structure_loss_reason = reformatter_structure_loss_reason(clean_report, writer_v3_markdown)
            if structure_loss_reason:
                log(f"  [WARN] ReformatterAgent 输出丢失篇幅或章节结构，自动回退到 WriterAgent 动态正文: {structure_loss_reason}")
                fallback_reason = structure_loss_reason
                clean_report = writer_v3_markdown
            clean_report = finalize_public_report(clean_report)
            validation = validate_reformatted_report(clean_report, as_list(clean_evidence.get("sources")), clean_evidence)
            repair_plan = build_reformatter_repair_plan(validation, clean_evidence, topic=query)
            repair_required = str(repair_plan.get("status") or "passed") != "passed"
            if (not validation.get("passed") or repair_required) and writer_v3_markdown:
                if repair_required and validation.get("passed"):
                    log(f"  [WARN] ReformatterAgent 仍需补正，已阻断 Clean report 写出: {repair_plan}")
                    fallback_reason = fallback_reason or "reformatter_repair_required"
                else:
                    log(f"  [WARN] ReformatterAgent 校验未通过，已阻断 Clean report 写出: {validation}")
                    fallback_reason = fallback_reason or "reformatter_validation_failed"
            if not validation.get("passed") or repair_required:
                fallback_report = ""
                fallback_validation: Dict[str, Any] = {}
                fallback_output_written = False
                if writer_v3_markdown:
                    fallback_report = finalize_public_report(writer_v3_markdown)
                    fallback_validation = validate_reformatted_report(fallback_report, as_list(clean_evidence.get("sources")), clean_evidence)
                    fallback_repair_plan = build_reformatter_repair_plan(fallback_validation, clean_evidence, topic=query)
                    clean_output_path.parent.mkdir(parents=True, exist_ok=True)
                    write_markdown(clean_output_path, fallback_report)
                    report_markdown = fallback_report.strip()
                    writer_report["reformatter_output_path"] = str(clean_output_path)
                    writer_report["reformatter_failed_validation"] = validation
                    writer_report["reformatter_validation"] = fallback_validation
                    writer_report["report_markdown"] = report_markdown
                    state_dict["writer_report"] = writer_report
                    state_dict["answer_text"] = report_markdown
                    fallback_output_written = True
                reformatter_result = {
                    "enabled": True,
                    "success": fallback_output_written,
                    "status": "fallback_writer" if fallback_output_written else ("repair_required" if repair_required else "validation_failed"),
                    "output_path": str(clean_output_path),
                    "output_written": fallback_output_written,
                    "validation": fallback_validation if fallback_output_written else validation,
                    "reformatter_validation": validation,
                    "fallback_validation": fallback_validation,
                    "repair_plan": fallback_repair_plan if fallback_output_written else repair_plan,
                    "reformatter_repair_plan": repair_plan,
                    "llm_runtime": llm_status,
                    "clean_evidence_count": int(as_dict(clean_evidence.get("metadata")).get("evidence_count") or 0),
                    "clean_body_chars_without_sources": int((fallback_validation if fallback_output_written else validation).get("body_chars_without_sources") or 0),
                    "clean_body_citation_count": int((fallback_validation if fallback_output_written else validation).get("citation_count") or 0),
                    "clean_body_unique_source_count": int((fallback_validation if fallback_output_written else validation).get("unique_cited_source_count") or 0),
                    "fallback_to_writer": bool(fallback_reason),
                    "fallback_reason": fallback_reason or ("reformatter_repair_required" if repair_required else "reformatter_validation_failed"),
                }
                writer_report.setdefault("reformatter_validation", validation)
                state_dict["writer_report"] = writer_report
                writer_package_payload["writer_report"] = writer_report
                writer_package_payload["reformatter_result"] = reformatter_result
                write_state_json(state_path, state_dict)
                write_json(package_path, writer_package_payload)
                if fallback_output_written:
                    log(f"  [WARN] ReformatterAgent 未达到 clean 标准，已写出 Writer 回退报告: {clean_output_path}")
                    progress.update(96, "ReformatterAgent 已回退")
                else:
                    log(f"  [WARN] ReformatterAgent 未达到 clean 标准，未写出 Clean report: {validation}")
                    progress.update(96, "ReformatterAgent 需补正")
            else:
                clean_output_path.parent.mkdir(parents=True, exist_ok=True)
                write_markdown(clean_output_path, clean_report)

                report_markdown = clean_report.strip()
                writer_report["reformatter_output_path"] = str(clean_output_path)
                writer_report["reformatter_validation"] = validation
                writer_report["report_markdown"] = report_markdown
                state_dict["writer_report"] = writer_report
                state_dict["answer_text"] = report_markdown
                reformatter_result = {
                    "enabled": True,
                    "success": True,
                    "status": "completed",
                    "output_path": str(clean_output_path),
                    "output_written": True,
                    "validation": validation,
                    "repair_plan": repair_plan,
                    "llm_runtime": llm_status,
                    "clean_evidence_count": int(as_dict(clean_evidence.get("metadata")).get("evidence_count") or 0),
                    "clean_body_chars_without_sources": int(validation.get("body_chars_without_sources") or 0),
                    "clean_body_citation_count": int(validation.get("citation_count") or 0),
                    "clean_body_unique_source_count": int(validation.get("unique_cited_source_count") or 0),
                    "fallback_to_writer": bool(fallback_reason),
                    "fallback_reason": fallback_reason,
                }
                writer_package_payload["writer_report"] = writer_report
                writer_package_payload["reformatter_result"] = reformatter_result
                write_state_json(state_path, state_dict)
                write_json(package_path, writer_package_payload)
                log(f"  - Clean report Markdown: {clean_output_path}")
                progress.update(96, "ReformatterAgent 清洗完成")
        except Exception as exc:
            reformatter_result = {
                **reformatter_result,
                "enabled": True,
                "success": False,
                "status": "failed",
                "output_path": str(clean_output_path),
                "output_written": False,
                "error": str(exc),
            }
            writer_package_payload["reformatter_result"] = reformatter_result
            write_json(package_path, writer_package_payload)
            log(f"  [WARN] ReformatterAgent 失败，保留 WriterAgent 输出: {exc}")
            if report_markdown and not args.skip_review:
                from .review_pipeline import run_review_pipeline_sync

                log("  [Fallback] ReviewAgent 审查 WriterAgent 报告中")
                review_result = run_review_pipeline_sync(
                    writer_output=report_markdown,
                    llm_client=None,
                    skip_llm_review=not args.enable_llm_review,
                )
                report_markdown = finalize_public_report(str(review_result.get("final_report") or report_markdown))
                writer_report["report_markdown"] = report_markdown
                writer_report["review_audit"] = as_dict(review_result.get("stage1_audit"))
                writer_report["review_stage2_skipped"] = bool(review_result.get("stage2_skipped", True))
                writer_report["review_total_fixes"] = int(review_result.get("total_fixes") or 0)
                state_dict["writer_report"] = writer_report
                state_dict["answer_text"] = report_markdown
                write_markdown(writer_md_path, report_markdown)
                writer_report["writer_markdown_path"] = str(writer_md_path)
                state_dict["writer_markdown_path"] = str(writer_md_path)
                writer_package_payload["writer_report"] = writer_report
                writer_package_payload["review_result"] = review_result
                write_state_json(state_path, state_dict)
                write_json(package_path, writer_package_payload)
            if report_markdown:
                report_markdown = finalize_public_report(report_markdown)
                fallback_output_written = False
                fallback_write_error = ""
                try:
                    clean_output_path.parent.mkdir(parents=True, exist_ok=True)
                    write_markdown(clean_output_path, report_markdown)
                    fallback_output_written = True
                except Exception as fallback_exc:  # pragma: no cover - filesystem edge case.
                    fallback_write_error = str(fallback_exc)
                writer_report["report_markdown"] = report_markdown
                if fallback_output_written:
                    writer_report["reformatter_output_path"] = str(clean_output_path)
                state_dict["writer_report"] = writer_report
                state_dict["answer_text"] = report_markdown
                writer_package_payload["writer_report"] = writer_report
                writer_package_payload["reformatter_result"] = {
                    **reformatter_result,
                    "fallback_to_writer": True,
                    "fallback_output_path": str(clean_output_path) if fallback_output_written else "",
                    "output_written": fallback_output_written,
                    "fallback_write_error": fallback_write_error,
                }
                write_state_json(state_path, state_dict)
                write_json(package_path, writer_package_payload)
                if fallback_output_written:
                    log(f"  [WARN] ReformatterAgent 失败，已写出 Writer 回退报告: {clean_output_path}")
                else:
                    log("  [WARN] ReformatterAgent 失败，已保留 Writer 草稿但不写出 Clean report")
            progress.update(96, "ReformatterAgent 已降级")

    reformatter_required = bool(writer_publishable and not args.skip_reformatter)
    reformatter_output_written = bool(as_dict(reformatter_result).get("output_written"))
    reformatter_blocked_clean = bool(reformatter_required and not reformatter_output_written)

    if writer_not_ready:
        log("[6/6] 正式报告已阻断，输出研究未完成与补证清单")
    elif not writer_publishable:
        log("[6/6] WriterAgent 输出为待复核草稿，已跳过 Clean report 写出")
    elif reformatter_blocked_clean:
        log("[6/6] ReformatterAgent 未产出 Clean report，已保留 Writer 草稿和失败状态")
    elif args.skip_reformatter:
        log("[5/6] 状态文件已生成，报告正文直接输出" if args.skip_review else "[5/6] ReviewAgent 与状态文件已完成，报告正文直接输出")
    else:
        log("[6/6] ReformatterAgent 与状态文件已完成，洁净报告正文直接输出")
    progress.update(98, "准备输出结果")
    final_incomplete = bool(missing and not args.allow_missing_stage)
    progress.finish("研究未完成" if writer_not_ready else ("Clean report 未生成" if reformatter_blocked_clean else ("流程不完整" if final_incomplete else "全流程完成")))
    log(f"  - Full state JSON: {state_path}", force=True)
    log(f"  - Writer package JSON: {package_path}", force=True)
    if report_markdown and not writer_not_ready:
        log(f"  - Writer Markdown: {writer_md_path}", force=True)
    clean_path = str(writer_report.get("reformatter_output_path") or "")
    if clean_path:
        log(f"  - Clean Markdown: {clean_path}", force=True)

    if errors:
        log("[WARN] 运行中存在非致命错误/降级：", force=True)
        for item in errors:
            log(f"  - {item}", force=True)

    final_stdout_allowed = bool(report_markdown and writer_publishable and not writer_not_ready and not reformatter_blocked_clean)
    if final_stdout_allowed:
        print(report_markdown)

    if final_incomplete:
        log("[6/6] 全流程执行不完整，以上阶段缺失。", force=True)
        return 2

    if writer_not_ready:
        log("[6/6] 检索或证据门槛未通过，正式报告未生成。", force=True)
        return 3 if env_flag("REPORT_NOT_READY_EXIT_NONZERO", True) else 0

    if not writer_publishable:
        log("[6/6] 流程完成但报告仍为待复核草稿，未发布 Clean report。", force=True)
        return 4 if env_flag("REPORT_REVIEW_REQUIRED_EXIT_NONZERO", False) else 0

    if reformatter_blocked_clean:
        log("[6/6] 流程完成但 Reformatter 未生成 Clean report；请查看 writer_package.reformatter_result。", force=True)
        return 5 if env_flag("REPORT_REFORMATTER_FAILURE_EXIT_NONZERO", False) else 0

    log("[6/6] 全流程执行完成。", force=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
