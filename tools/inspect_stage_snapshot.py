from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_pipeline.cache.stage_snapshot_cache import list_stage_snapshots, load_stage_snapshot, summarize_payload


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect diagnostic stage snapshots for one report run.")
    parser.add_argument("--run-id", required=True, help="Stage snapshot run id.")
    parser.add_argument("--stage", default="", help="Optional stage name to load. Omit to list manifests.")
    parser.add_argument("--include-payload-summary", action="store_true", help="Include compact payload summary for --stage.")
    parser.add_argument("--follow", action="store_true", help="Keep watching the run snapshot index.")
    parser.add_argument("--interval", type=float, default=5.0, help="Polling interval for --follow.")
    args = parser.parse_args()

    if args.follow and not args.stage:
        seen: set[str] = set()
        while True:
            snapshots = list_stage_snapshots(args.run_id)
            for manifest in snapshots:
                stage_name = str(_as_dict(manifest).get("stage_name") or "")
                if not stage_name or stage_name in seen:
                    continue
                seen.add(stage_name)
                summary = _as_dict(manifest).get("output_summary")
                byte_count = _as_dict(manifest).get("full_payload_bytes", 0)
                replayable = _as_dict(manifest).get("replayable")
                print(f"{stage_name:<28} replayable={replayable} bytes={byte_count} summary={summary}", flush=True)
            time.sleep(max(0.5, args.interval))

    if not args.stage:
        snapshots = list_stage_snapshots(args.run_id)
        print(json.dumps({"run_id": args.run_id, "snapshot_count": len(snapshots), "snapshots": snapshots}, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    loaded = load_stage_snapshot(args.run_id, args.stage)
    payload = _as_dict(loaded).get("payload")
    if args.include_payload_summary and payload is not None:
        loaded = dict(loaded)
        loaded["payload_summary"] = summarize_payload(payload)
        loaded.pop("payload", None)
    print(json.dumps(loaded, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    return 0 if _as_dict(loaded).get("status") in {"loaded", "missing"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
