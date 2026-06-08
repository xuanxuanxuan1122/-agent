from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from rag_pipeline.quality.regression import (
    load_quality_snapshots_from_paths,
    summarize_quality_regression_suite,
)


RunnerResult = Dict[str, Any]
Runner = Callable[..., RunnerResult]


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


def _text(value: Any) -> str:
    return str(value or "").strip()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return default


def _slug(value: Any, fallback: str) -> str:
    text = _text(value).lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "-", text, flags=re.I)
    text = re.sub(r"-+", "-", text).strip("-")
    return text[:80] or fallback


def _writer_package_paths(output_dir: Path) -> List[Path]:
    if not output_dir.exists():
        return []
    return sorted(output_dir.glob("**/*writer_package*.json"))


def build_quality_execution_plan(
    topics: Iterable[Dict[str, Any]],
    *,
    output_root: str | Path,
    python_executable: str = sys.executable,
    route: str = "web",
    quality_mode: str = "",
    llm_profile: str = "",
    extra_args: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    root = Path(output_root)
    args = [str(item) for item in (extra_args or [])]
    plan: List[Dict[str, Any]] = []
    for topic_index, raw_topic in enumerate(topics):
        topic = _as_dict(raw_topic)
        query = _text(topic.get("query"))
        topic_id = _slug(topic.get("topic_id") or query, f"topic-{topic_index + 1}")
        domain = _text(topic.get("domain") or topic.get("industry") or "unknown")
        repeat_count = max(1, _safe_int(topic.get("repeat_count"), 1))
        topic_route = _text(topic.get("route") or route) or "web"
        topic_quality_mode = _text(topic.get("quality_mode") or quality_mode)
        topic_llm_profile = _text(topic.get("llm_profile") or llm_profile)
        for repeat_index in range(1, repeat_count + 1):
            output_dir = root / topic_id / f"run_{repeat_index:02d}"
            command = [
                str(python_executable),
                "-m",
                "rag_pipeline.flows.report.full_report",
                "--query",
                query,
                "--route",
                topic_route,
                "--output-dir",
                str(output_dir),
                "--no-interactive-input",
            ]
            if topic_llm_profile:
                command.extend(["--llm-profile", topic_llm_profile])
            command.extend(args)
            plan.append(
                {
                    "schema_version": "quality_regression_execution_task_v1",
                    "topic_id": topic_id,
                    "domain": domain,
                    "query": query,
                    "repeat_index": repeat_index,
                    "output_dir": str(output_dir),
                    "quality_mode": topic_quality_mode,
                    "route": topic_route,
                    "command": command,
                }
            )
    return plan


def _subprocess_runner(command: Sequence[str], *, cwd: str | Path, env: Dict[str, str], timeout: Optional[float]) -> RunnerResult:
    started = time.time()
    completed = subprocess.run(
        list(command),
        cwd=str(cwd),
        env=env,
        timeout=timeout,
        text=True,
        capture_output=True,
    )
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "duration_seconds": round(time.time() - started, 3),
    }


def _execution_env(base_env: Optional[Dict[str, str]], task: Dict[str, Any]) -> Dict[str, str]:
    env = dict(os.environ)
    if base_env:
        env.update({str(key): str(value) for key, value in base_env.items()})
    if task.get("quality_mode"):
        env["REPORT_QUALITY_MODE"] = _text(task.get("quality_mode"))
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("TQDM_DISABLE", "1")
    env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    env["QUALITY_REGRESSION_TOPIC_ID"] = _text(task.get("topic_id"))
    env["QUALITY_REGRESSION_DOMAIN"] = _text(task.get("domain"))
    env["QUALITY_REGRESSION_REPEAT_INDEX"] = str(task.get("repeat_index") or "")
    return env


def _snapshot_with_task_context(snapshot: Dict[str, Any], task: Dict[str, Any], path: Path) -> Dict[str, Any]:
    result = dict(snapshot)
    existing_topic = _text(result.get("topic_id"))
    existing_domain = _text(result.get("domain"))
    result["topic_id"] = existing_topic if existing_topic and existing_topic != "unknown" else _text(task.get("topic_id"))
    result["domain"] = existing_domain if existing_domain and existing_domain != "unknown" else _text(task.get("domain"))
    result["query"] = _text(result.get("query")) or _text(task.get("query"))
    result["repeat_index"] = _safe_int(task.get("repeat_index"), 0)
    result["output_dir"] = _text(task.get("output_dir"))
    result["source_path"] = str(path)
    return result


def run_quality_regression_execution(
    topics: Iterable[Dict[str, Any]],
    *,
    output_root: str | Path,
    runner: Optional[Runner] = None,
    python_executable: str = sys.executable,
    cwd: str | Path | None = None,
    timeout_seconds: Optional[float] = None,
    env_overrides: Optional[Dict[str, str]] = None,
    route: str = "web",
    quality_mode: str = "",
    llm_profile: str = "",
    extra_args: Optional[Sequence[str]] = None,
    dry_run: bool = False,
    min_publish_score: int = 70,
    min_pass_rate: float = 0.8,
    max_score_stddev: float = 5.0,
    max_tokens_per_run: int = 0,
    max_duration_seconds: int = 0,
) -> Dict[str, Any]:
    plan = build_quality_execution_plan(
        topics,
        output_root=output_root,
        python_executable=python_executable,
        route=route,
        quality_mode=quality_mode,
        llm_profile=llm_profile,
        extra_args=extra_args,
    )
    project_root = Path(cwd) if cwd is not None else Path(__file__).resolve().parents[2]
    active_runner = runner or _subprocess_runner
    execution_results: List[Dict[str, Any]] = []
    snapshots: List[Dict[str, Any]] = []
    for task in plan:
        command = [str(item) for item in _as_list(task.get("command"))]
        output_dir = Path(_text(task.get("output_dir")))
        output_dir.mkdir(parents=True, exist_ok=True)
        result: Dict[str, Any]
        if dry_run:
            result = {"returncode": None, "stdout": "", "stderr": "", "duration_seconds": 0, "dry_run": True}
        else:
            try:
                result = active_runner(
                    command,
                    cwd=project_root,
                    env=_execution_env(env_overrides, task),
                    timeout=timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                result = {
                    "returncode": -1,
                    "stdout": _text(getattr(exc, "stdout", "")),
                    "stderr": f"timeout_after_{timeout_seconds}_seconds",
                    "duration_seconds": timeout_seconds or 0,
                    "timed_out": True,
                }
        paths = _writer_package_paths(output_dir)
        task_snapshots = load_quality_snapshots_from_paths(paths) if paths else []
        contextual_snapshots = [
            _snapshot_with_task_context(snapshot, task, path)
            for snapshot, path in zip(task_snapshots, paths)
        ]
        snapshots.extend(contextual_snapshots)
        execution_results.append(
            {
                "schema_version": "quality_regression_execution_result_v1",
                "topic_id": task.get("topic_id"),
                "domain": task.get("domain"),
                "query": task.get("query"),
                "repeat_index": task.get("repeat_index"),
                "output_dir": task.get("output_dir"),
                "command": command,
                "returncode": result.get("returncode"),
                "duration_seconds": result.get("duration_seconds", 0),
                "stdout_tail": _text(result.get("stdout"))[-1200:],
                "stderr_tail": _text(result.get("stderr"))[-1200:],
                "status": "planned" if dry_run else ("completed" if result.get("returncode") == 0 else "failed"),
                "writer_package_paths": [str(path) for path in paths],
                "snapshot_count": len(contextual_snapshots),
            }
        )
    suite_summary = summarize_quality_regression_suite(
        snapshots,
        min_publish_score=min_publish_score,
        min_pass_rate=min_pass_rate,
        max_score_stddev=max_score_stddev,
        max_tokens_per_run=max_tokens_per_run,
        max_duration_seconds=max_duration_seconds,
    )
    completed_count = sum(1 for item in execution_results if item.get("status") == "completed")
    failed_count = sum(1 for item in execution_results if item.get("status") == "failed")
    return {
        "schema_version": "quality_regression_execution_v1",
        "execution_summary": {
            "planned_run_count": len(plan),
            "completed_run_count": completed_count,
            "failed_run_count": failed_count,
            "snapshot_count": len(snapshots),
        },
        "execution_plan": plan,
        "execution_results": execution_results,
        "snapshots": snapshots,
        "suite_summary": suite_summary,
    }
