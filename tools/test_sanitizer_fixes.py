"""Offline driver for validating the recent sanitizer / renderer fixes.

Runs ``sanitize_public_markdown`` against the most recent ``*_clean.md`` and
prints a structured before/after diagnostic so we can see whether the diagnostic
table stripping, term rewrites and citation-heading fix actually trigger on the
real report. No external API calls, no network — pure local transformation.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_pipeline.agents.public_report_sanitizer import (  # noqa: E402
    PUBLIC_INTERNAL_TERM_REWRITES,
    public_narrative_leak_audit,
    sanitize_public_markdown,
)
from rag_pipeline.agents.final_writer_agent import _citationless_factual_segments  # noqa: E402


REPORT_PATH = ROOT / "output" / "full_reports" / "20260520_161059_AI_Agent生态发展报告：从工具到智能体的范式跃迁_clean.md"


DIAGNOSTIC_RE = re.compile(
    r"该指标须同时披露|进入正文判断|不会凭空补齐|缺口数据只作为"
    r"|指标口径表|市场指标与口径表|政策影响与风险登记表",
    re.I,
)
COMPETITIVE_SIGNAL_RE = re.compile(r"competitive signal", re.I)
CITATION_RE = re.compile(r"\[\d{1,3}\]")
SOURCE_HEADING_RE = re.compile(
    r"(?mi)^##+\s*(?:数据来源|资料来源|研究口径与来源|参考来源|来源附录|Sources|References)"
)


def measure(label: str, text: str) -> dict:
    diagnostic_lines = []
    citationless_factual = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("|"):
            continue
        if DIAGNOSTIC_RE.search(line):
            diagnostic_lines.append(line.strip()[:120])
        # A rough "looks-factual without [n]" heuristic, mirroring the auditor.
        if not CITATION_RE.search(line) and re.search(r"\d{2,}|%|亿|万|2024|2025|2026|采购|预算", line):
            citationless_factual += 1
    return {
        "label": label,
        "char_count": len(text),
        "line_count": text.count("\n") + 1,
        "diagnostic_line_count": len(diagnostic_lines),
        "diagnostic_samples": diagnostic_lines[:5],
        "competitive_signal_count": len(COMPETITIVE_SIGNAL_RE.findall(text)),
        "source_appendix_heading_present": bool(SOURCE_HEADING_RE.search(text)),
        "citationless_factual_count_estimate": citationless_factual,
    }


def diff_report(before: dict, after: dict) -> None:
    keys = [
        "char_count",
        "line_count",
        "diagnostic_line_count",
        "competitive_signal_count",
        "source_appendix_heading_present",
        "citationless_factual_count_estimate",
    ]
    print("\n=== before / after metrics ===")
    width = max(len(k) for k in keys)
    for key in keys:
        b = before[key]
        a = after[key]
        delta = ""
        if isinstance(b, int) and isinstance(a, int):
            delta = f"  delta={a - b:+d}"
        print(f"  {key.ljust(width)}  before={b}  after={a}{delta}")
    if before["diagnostic_samples"]:
        print("\n=== diagnostic samples seen in BEFORE ===")
        for sample in before["diagnostic_samples"]:
            print(f"  - {sample}")
    if after["diagnostic_samples"]:
        print("\n=== diagnostic samples STILL in AFTER ===")
        for sample in after["diagnostic_samples"]:
            print(f"  - {sample}")


def main() -> int:
    if not REPORT_PATH.exists():
        print(f"ERROR: report not found at {REPORT_PATH}")
        return 1

    raw = REPORT_PATH.read_text(encoding="utf-8")
    print(f"Loaded: {REPORT_PATH.name}  ({len(raw)} chars, {raw.count(chr(10))+1} lines)")

    # Inject a synthetic source appendix and a Competitive signal phrase to
    # validate the renderer-side fixes too, since the existing clean.md was
    # produced before the changes landed.
    augmented = raw
    if "Competitive signal" not in augmented:
        augmented += (
            "\n\n*Embedded test marker:* Competitive signal observed for AI Agent.\n"
        )
    if not SOURCE_HEADING_RE.search(augmented):
        augmented += "\n\n## 来源附录\n- [1] 测试来源 | 2026-04 | https://example.com/test\n"

    before = measure("before", augmented)
    cleaned = sanitize_public_markdown(augmented)
    after = measure("after", cleaned)

    diff_report(before, after)

    print("\n=== citationless factual segments (auditor reality check) ===")
    before_citationless = _citationless_factual_segments(augmented, limit=20)
    after_citationless = _citationless_factual_segments(cleaned, limit=20)
    print(f"  before: count={len(before_citationless)}")
    for sample in before_citationless[:6]:
        print(f"    - {sample[:140]}")
    print(f"  after:  count={len(after_citationless)}")
    for sample in after_citationless[:6]:
        print(f"    - {sample[:140]}")

    print("\n=== public_narrative_leak_audit on cleaned output ===")
    leak = public_narrative_leak_audit(cleaned)
    print(f"  blocker_count: {leak['blocker_count']}")
    print(f"  reason_counts: {leak['reason_counts']}")
    for example in leak["examples"][:5]:
        print(f"  - line {example['line']}: [{example['reason']}] {example['text'][:120]}")

    # Spot check: does the cleaned text retain the appendix and drop the diagnostic table?
    print("\n=== spot checks ===")
    print(f"  source_appendix retained:        {bool(SOURCE_HEADING_RE.search(cleaned))}")
    print(f"  '该指标须同时披露' removed:        {'该指标须同时披露' not in cleaned}")
    print(f"  '进入正文判断' removed:           {'进入正文判断' not in cleaned}")
    print(f"  '不会凭空补齐' removed:           {'不会凭空补齐' not in cleaned}")
    print(f"  'Competitive signal' rewritten:  {'Competitive signal' not in cleaned}")
    print(f"  '竞争信号' present (from rewrite): {'竞争信号' in cleaned}")
    print(f"  '市场指标与口径表' removed:       {'市场指标与口径表' not in cleaned}")

    # Show the AFTER text head/tail for human eyeball check.
    print("\n=== AFTER head (first 1200 chars) ===")
    print(cleaned[:1200])
    print("\n=== AFTER tail (last 800 chars) ===")
    print(cleaned[-800:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
