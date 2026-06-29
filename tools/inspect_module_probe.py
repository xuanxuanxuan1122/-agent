from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_pipeline.observability.probe_runtime import summarize_runtime_events


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        item = json.loads(text)
        if isinstance(item, dict):
            events.append(item)
    return events


def _summarize_events(events: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    by_stage: Counter[str] = Counter()
    by_type: Counter[str] = Counter()
    drops: Counter[str] = Counter()
    cache_hits: Counter[str] = Counter()
    for event in events:
        stage = str(event.get("stage") or "unknown")
        event_type = str(event.get("event_type") or "unknown")
        by_stage[stage] += 1
        by_type[event_type] += 1
        if event_type != "transform_result":
            continue
        drop_count = int(_as_dict(event.get("drop")).get("count") or 0)
        if drop_count:
            drops[stage] += drop_count
        hit_count = int(_as_dict(event.get("cache")).get("hit_count") or 0)
        if hit_count:
            cache_hits[stage] += hit_count
    return {
        "event_count": sum(by_stage.values()),
        "event_count_by_stage": dict(by_stage),
        "event_count_by_type": dict(by_type),
        "drop_count_by_stage": dict(drops),
        "cache_hit_count_by_stage": dict(cache_hits),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect module_probe, lineage_graph, and health_metrics sidecars.")
    parser.add_argument("module_probe", help="Path to *.module_probe.jsonl.")
    parser.add_argument("--lineage", default="", help="Optional *.lineage_graph.json path.")
    parser.add_argument("--health", default="", help="Optional *.health_metrics.json path.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of Markdown.")
    args = parser.parse_args()

    module_path = Path(args.module_probe)
    if not module_path.exists():
        print(f"module_probe not found: {module_path}", file=sys.stderr)
        return 1
    events = _read_jsonl(module_path)
    summary: Dict[str, Any] = {"module_probe": _summarize_events(events)}
    summary["runtime"] = summarize_runtime_events(module_path)
    if args.lineage:
        lineage_path = Path(args.lineage)
        if lineage_path.exists():
            lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
            summary["lineage"] = {
                "node_count": len(lineage.get("nodes") or []),
                "edge_count": len(lineage.get("edges") or []),
                "coverage": lineage.get("coverage") or {},
            }
    if args.health:
        health_path = Path(args.health)
        if health_path.exists():
            health = json.loads(health_path.read_text(encoding="utf-8"))
            summary["health"] = {"rates": health.get("rates") or {}, "counts": health.get("counts") or {}}
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    print("# Module Probe Summary")
    print("")
    print("## Events")
    for stage, count in _as_dict(summary["module_probe"].get("event_count_by_stage")).items():
        print(f"- {stage}: {count}")
    print("")
    print("## Drops")
    drops = _as_dict(summary["module_probe"].get("drop_count_by_stage"))
    if drops:
        for stage, count in drops.items():
            print(f"- {stage}: {count}")
    else:
        print("- None recorded.")
    if "health" in summary:
        print("")
        print("## Core Rates")
        for key, value in _as_dict(summary["health"].get("rates")).items():
            print(f"- {key}: {value}")
    if "runtime" in summary:
        print("")
        print("## Runtime Rates")
        for key, value in _as_dict(summary["runtime"].get("rates")).items():
            print(f"- {key}: {value}")
        root_hygiene = _as_dict(summary["runtime"].get("root_hygiene"))
        if root_hygiene:
            print("")
            print("## Root Hygiene")
            print(f"- status: {root_hygiene.get('status')}")
            print(f"- earliest_issue_stage: {root_hygiene.get('earliest_issue_stage') or 'none'}")
            issue_counts = _as_dict(root_hygiene.get("issue_count_by_stage"))
            if issue_counts:
                for stage, count in issue_counts.items():
                    print(f"- {stage}: {count}")
            else:
                print("- no root hygiene issues recorded.")
    if "lineage" in summary:
        print("")
        print("## Lineage")
        print(f"- nodes: {summary['lineage'].get('node_count')}")
        print(f"- edges: {summary['lineage'].get('edge_count')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
