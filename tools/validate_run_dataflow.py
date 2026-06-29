from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_pipeline.observability.dataflow_inspector import build_dataflow_report, render_dataflow_summary
from rag_pipeline.observability.stage_contracts import validate_stage_packets


def _read_jsonl(path: Path) -> List[Dict[str, object]]:
    packets: List[Dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        item = json.loads(text)
        if isinstance(item, dict):
            packets.append(item)
    return packets


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and summarize a stage_probe JSONL file.")
    parser.add_argument("stage_probe", help="Path to *.stage_probe.jsonl.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of Markdown.")
    parser.add_argument("--output", default="", help="Optional output file for the rendered summary.")
    args = parser.parse_args()

    path = Path(args.stage_probe)
    packets = _read_jsonl(path)
    validation = validate_stage_packets(packets)
    report = build_dataflow_report(packets)
    payload = {"validation": validation, "dataflow": report}

    if args.json:
        text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    else:
        text = render_dataflow_summary(report)
        if not validation.get("ok"):
            text += "\n## Packet Validation Errors\n"
            text += json.dumps(validation.get("errors"), ensure_ascii=False, indent=2) + "\n"

    if args.output:
        Path(args.output).write_text(text.rstrip() + "\n", encoding="utf-8")
    else:
        print(text.rstrip())
    return 0 if validation.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
