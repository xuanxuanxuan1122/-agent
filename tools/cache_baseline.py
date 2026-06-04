"""Phase 0 验证护栏：跑一个 golden query 端到端，落基线（正文 + cache_report + 摘要）。

用途：在每个缓存优化阶段前后各跑一次，diff `baseline_summary.json` 与 cache_report，
证明改动「全流程跑通、无回归」。本工具只是驱动现有 run_full_report.py + 收割产物，
不改任何管线行为。

示例：
  python tools/cache_baseline.py --golden first --label before_phase1
  python tools/cache_baseline.py --query "中国农业机器人行业是否值得进入" --label adhoc
  python tools/cache_baseline.py --from-dir output/full_reports --label harvest_only   # 只收割不跑
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = PIPELINE_ROOT / "golden_cases" / "minimal_cases.json"
ENTRY = PIPELINE_ROOT / "run_full_report.py"


def _load_golden(case: str) -> Dict[str, Any]:
    data = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    cases: List[Dict[str, Any]] = list(data.get("cases") or [])
    if not cases:
        raise SystemExit(f"golden_cases 为空: {GOLDEN_PATH}")
    if case in {"first", "", None}:
        return cases[0]
    for item in cases:
        if str(item.get("case_id") or "") == case:
            return item
    try:
        return cases[int(case)]
    except (ValueError, IndexError):
        ids = ", ".join(str(item.get("case_id")) for item in cases)
        raise SystemExit(f"未找到 golden case '{case}'。可选: {ids}")


def _newest(directory: Path, pattern: str) -> Optional[Path]:
    matches = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _digest(cache_report: Dict[str, Any], report_chars: int) -> Dict[str, Any]:
    ev = cache_report.get("evidence_cache") or {}
    activity = ev.get("activity") or {}
    stats = ev.get("stats") or {}
    snaps = cache_report.get("stage_snapshots") or {}
    trusted = cache_report.get("trusted_source") or {}
    bundle = cache_report.get("topic_bundle") or {}
    return {
        "report_chars": report_chars,
        "evidence_cache": {
            "search_hit": activity.get("search_hit"),
            "search_store": activity.get("search_store"),
            "evidence_hit": activity.get("evidence_hit"),
            "evidence_store": activity.get("evidence_store"),
            "stale_count": activity.get("stale_count"),
            "error_count": activity.get("error_count"),
            "search_count": stats.get("search_count"),
            "evidence_count": stats.get("evidence_count"),
            "negative_count": stats.get("negative_count"),
        },
        "stage_snapshots": {
            "count": snaps.get("count"),
            "replayable_count": snaps.get("replayable_count"),
            "stages": snaps.get("stages"),
        },
        "trusted_source_entry_count": trusted.get("entry_count"),
        "topic_bundle": {
            "hit": bundle.get("hit"),
            "used_for_skip_search": bundle.get("used_for_skip_search"),
            "stored_bundle_count": bundle.get("stored_bundle_count"),
        },
    }


def _harvest(run_dir: Path, baseline_dir: Path, *, query: str, label: str) -> Dict[str, Any]:
    baseline_dir.mkdir(parents=True, exist_ok=True)
    cache_report_path = _newest(run_dir, "*.cache_report.json")
    report_md_path = _newest(run_dir, "*_report.md") or _newest(run_dir, "*.writer.md")
    state_path = _newest(run_dir, "*.state.json")

    cache_report: Dict[str, Any] = {}
    if cache_report_path and cache_report_path.exists():
        try:
            cache_report = json.loads(cache_report_path.read_text(encoding="utf-8"))
            shutil.copy2(cache_report_path, baseline_dir / "cache_report.json")
        except Exception as exc:
            cache_report = {"_read_error": str(exc), "_path": str(cache_report_path)}

    report_chars = 0
    if report_md_path and report_md_path.exists():
        try:
            text = report_md_path.read_text(encoding="utf-8")
            report_chars = len(text)
            (baseline_dir / "report_head.md").write_text(text[:4000], encoding="utf-8")
            shutil.copy2(report_md_path, baseline_dir / "report.md")
        except Exception:
            pass

    summary = {
        "label": label,
        "query": query,
        "harvested_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "cache_report_path": str(cache_report_path) if cache_report_path else "",
        "report_md_path": str(report_md_path) if report_md_path else "",
        "state_path": str(state_path) if state_path else "",
        "digest": _digest(cache_report, report_chars),
    }
    (baseline_dir / "baseline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 0 缓存基线工具")
    parser.add_argument("--query", default="", help="自定义报告问题。")
    parser.add_argument("--golden", default="", help="使用 golden case：case_id / 序号 / 'first'。")
    parser.add_argument("--label", default="", help="基线标签（目录名），默认时间戳。")
    parser.add_argument("--route", default="web", help="run_full_report 的 --route，默认 web。")
    parser.add_argument("--output-root", default="output/cache_baseline", help="基线输出根目录。")
    parser.add_argument("--from-dir", default="", help="只收割该目录里最新产物，不跑新报告。")
    parser.add_argument("--no-run", action="store_true", help="只收割 run-dir，不启动新报告。")
    parser.add_argument("--timeout", type=int, default=0, help="子进程超时秒数，0=不限。")
    parser.add_argument("extra", nargs="*", help="透传给 run_full_report.py 的额外参数（用 -- 分隔）。")
    args = parser.parse_args()

    label = args.label or datetime.now().strftime("%Y%m%d_%H%M%S")
    baseline_dir = (PIPELINE_ROOT / args.output_root / label).resolve()

    query = args.query.strip()
    if not query and args.golden:
        query = str(_load_golden(args.golden).get("query") or "").strip()

    # 只收割模式
    if args.from_dir or args.no_run:
        run_dir = Path(args.from_dir or (PIPELINE_ROOT / "output" / "full_reports")).resolve()
        if not run_dir.exists():
            raise SystemExit(f"run-dir 不存在: {run_dir}")
        summary = _harvest(run_dir, baseline_dir, query=query or "(harvest_only)", label=label)
        _print_summary(summary, baseline_dir)
        return 0

    if not query:
        raise SystemExit("需要 --query 或 --golden 指定报告问题。")

    run_dir = baseline_dir / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ENTRY),
        query,
        "--route",
        args.route,
        "--output-dir",
        str(run_dir),
        "--no-interactive-input",
    ] + list(args.extra)
    print(f"[cache_baseline] 运行: {' '.join(cmd)}")
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(PIPELINE_ROOT),
            timeout=args.timeout or None,
        )
    except subprocess.TimeoutExpired:
        print(f"[cache_baseline] 子进程超时（{args.timeout}s），仍尝试收割已产出。")
        completed = None

    if completed is not None and completed.returncode != 0:
        print(f"[cache_baseline][WARN] run_full_report 退出码 {completed.returncode}，仍尝试收割。")

    summary = _harvest(run_dir, baseline_dir, query=query, label=label)
    _print_summary(summary, baseline_dir)
    return 0


def _print_summary(summary: Dict[str, Any], baseline_dir: Path) -> None:
    digest = summary.get("digest") or {}
    print("\n========== cache_baseline ==========")
    print(f"label      : {summary.get('label')}")
    print(f"query      : {summary.get('query')}")
    print(f"baseline   : {baseline_dir}")
    print(f"report_md  : {summary.get('report_md_path') or '(none)'}")
    print(f"report_chars: {digest.get('report_chars')}")
    print(f"evidence_cache: {json.dumps(digest.get('evidence_cache'), ensure_ascii=False)}")
    print(f"snapshots  : {json.dumps(digest.get('stage_snapshots'), ensure_ascii=False)}")
    print(f"trusted    : {digest.get('trusted_source_entry_count')}")
    print(f"topic_bundle: {json.dumps(digest.get('topic_bundle'), ensure_ascii=False)}")
    if not summary.get("cache_report_path"):
        print("[WARN] 未发现 cache_report.json —— 确认本次运行的 full_report 已含 Phase 0 改动，"
              "且 CACHE_REPORT_SIDECAR_ENABLED 未关闭。")
    print("====================================\n")


if __name__ == "__main__":
    raise SystemExit(main())
