from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_pipeline.quality.regression import (
    load_quality_snapshots_from_paths,
    summarize_quality_regression_suite,
)


def _expand_inputs(values: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    for value in values:
        path = Path(value)
        if path.is_dir():
            paths.extend(sorted(path.glob("**/*writer_package*.json")))
        else:
            paths.append(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize multi-run report quality stability from writer_package JSON files.")
    parser.add_argument("inputs", nargs="+", help="writer_package JSON files or directories containing them.")
    parser.add_argument("--min-publish-score", type=int, default=70)
    parser.add_argument("--min-pass-rate", type=float, default=0.8)
    parser.add_argument("--max-score-stddev", type=float, default=5.0)
    parser.add_argument("--max-tokens-per-run", type=int, default=0)
    parser.add_argument("--max-duration-seconds", type=int, default=0)
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    args = parser.parse_args()

    paths = _expand_inputs(args.inputs)
    snapshots = load_quality_snapshots_from_paths(paths)
    summary = summarize_quality_regression_suite(
        snapshots,
        min_publish_score=args.min_publish_score,
        min_pass_rate=args.min_pass_rate,
        max_score_stddev=args.max_score_stddev,
        max_tokens_per_run=args.max_tokens_per_run,
        max_duration_seconds=args.max_duration_seconds,
    )
    text = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0 if summary.get("overall_status") == "stable" else 1


if __name__ == "__main__":
    raise SystemExit(main())
