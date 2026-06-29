from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _format_packet(packet: Dict[str, Any]) -> str:
    stage = packet.get("stage") or "unknown"
    event_type = str(packet.get("event_type") or "").strip()
    status = packet.get("status") or "unknown"
    if "input_count" in packet:
        input_count = packet.get("input_count", 0)
        output_count = packet.get("output_count", 0)
        drop_count = packet.get("drop_count", 0)
    else:
        input_count = _as_dict(packet.get("input")).get("count", 0)
        output_count = _as_dict(packet.get("output")).get("count", 0)
        drop_count = _as_dict(packet.get("drop")).get("count", 0)
    gaps = []
    coverage_payload = _as_dict(packet.get("id_coverage")) or _as_dict(_as_dict(packet.get("output")).get("id_coverage"))
    for field, coverage in coverage_payload.items():
        try:
            ratio = float(coverage)
        except (TypeError, ValueError):
            continue
        if ratio < 1:
            gaps.append(f"{field}={ratio:.2f}")
    gap_text = f" lineage_gap[{', '.join(gaps)}]" if gaps else ""
    event_text = f"{event_type:<18} " if event_type else ""
    return f"{stage:<22} {event_text}{status:<8} in={input_count} out={output_count} drop={drop_count}{gap_text}"


def _read_packets(path: Path) -> Iterable[Dict[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            item = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            yield item


def _follow(path: Path, *, interval: float) -> None:
    seen = 0
    while True:
        if path.exists():
            packets = list(_read_packets(path))
            for packet in packets[seen:]:
                print(_format_packet(packet), flush=True)
            seen = len(packets)
        time.sleep(interval)


def main() -> int:
    parser = argparse.ArgumentParser(description="Print a compact live view of a trace or stage_probe JSONL file.")
    parser.add_argument("jsonl_path", help="Path to *.trace.jsonl or *.stage_probe.jsonl.")
    parser.add_argument("--follow", action="store_true", help="Keep watching the file for appended packets.")
    parser.add_argument("--interval", type=float, default=2.0, help="Polling interval for --follow.")
    args = parser.parse_args()

    path = Path(args.jsonl_path)
    if args.follow:
        _follow(path, interval=max(0.2, args.interval))
        return 0
    if not path.exists():
        print(f"trace/probe file not found: {path}", file=sys.stderr)
        return 1
    for packet in _read_packets(path):
        print(_format_packet(packet))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
