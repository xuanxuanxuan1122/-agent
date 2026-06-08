from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_pipeline.quality.executor import run_quality_regression_execution


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _load_topics(path: str) -> List[Dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    data = _as_dict(payload)
    topics = data.get("topics")
    if isinstance(topics, list):
        return [item for item in topics if isinstance(item, dict)]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Execute a cross-domain, repeated full-report quality regression suite and summarize stability."
    )
    parser.add_argument("--suite", required=True, help="JSON file: either a list of topics or {'topics': [...]} .")
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "output" / "quality_regression_runs"))
    parser.add_argument("--route", default="web")
    parser.add_argument("--quality-mode", default="high")
    parser.add_argument("--llm-profile", default="")
    parser.add_argument("--timeout-seconds", type=float, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--min-publish-score", type=int, default=70)
    parser.add_argument("--min-pass-rate", type=float, default=0.8)
    parser.add_argument("--max-score-stddev", type=float, default=5.0)
    parser.add_argument("--max-tokens-per-run", type=int, default=0)
    parser.add_argument("--max-duration-seconds", type=int, default=0)
    parser.add_argument("--output", default="", help="Optional JSON summary output path.")
    parser.add_argument("extra", nargs="*", help="Extra args passed to full_report.py, e.g. -- --supervisor-max-loops 1.")
    args = parser.parse_args()

    extra_args = list(args.extra)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    result = run_quality_regression_execution(
        _load_topics(args.suite),
        output_root=args.output_root,
        route=args.route,
        quality_mode=args.quality_mode,
        llm_profile=args.llm_profile,
        timeout_seconds=args.timeout_seconds or None,
        dry_run=args.dry_run,
        extra_args=extra_args,
        min_publish_score=args.min_publish_score,
        min_pass_rate=args.min_pass_rate,
        max_score_stddev=args.max_score_stddev,
        max_tokens_per_run=args.max_tokens_per_run,
        max_duration_seconds=args.max_duration_seconds,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    if args.dry_run:
        return 0
    return 0 if result.get("suite_summary", {}).get("overall_status") == "stable" else 1


if __name__ == "__main__":
    raise SystemExit(main())
