from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


PIPELINE_ROOT = Path(__file__).resolve().parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from evidence_extractor import REPORT_DIMENSIONS, clean_evidence_text, extract_clean_evidence
from reformatter_agent import run_reformatter, validate_reformatted_report


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
        value = value.strip().strip("\"'")
        if key and not os.environ.get(key):
            os.environ[key] = value


def build_evidence_from_messy_report(report_path: str) -> Dict[str, Any]:
    content = Path(report_path).read_text(encoding="utf-8")
    sources: List[Dict[str, Any]] = []
    for match in re.finditer(r"\[(\d+)\]\s+([^\n，]+)，([^\n，]*?)，([^\n，]*)(?:，([^\n]+))?", content):
        sources.append(
            {
                "id": int(match.group(1)),
                "title": match.group(2).strip(),
                "domain": match.group(3).strip(),
                "date": match.group(4).strip(),
                "url": (match.group(5) or "").strip(),
            }
        )

    dim_keywords = {
        "市场规模与增速": ["市场规模", "亿美元", "亿元", "CAGR", "增速", "增长率", "渗透率"],
        "竞争格局": ["竞争", "市场份额", "头部", "龙头", "集中度", "市占率"],
        "政策与监管环境": ["政策", "补贴", "监管", "奖励", "试点", "示范", "标准"],
        "技术路线与产业链": ["技术", "算法", "传感器", "大模型", "产业链", "上游", "下游"],
        "投融资与资本动态": ["融资", "估值", "并购", "IPO", "投资", "资本", "财报"],
    }
    dimensions = {dimension: [] for dimension in REPORT_DIMENSIONS}
    seen = set()
    for raw_fact in re.findall(r"([^。\n]{15,220}\[\d+\])", content):
        ref_match = re.search(r"\[(\d+)\]", raw_fact)
        fact = clean_evidence_text(raw_fact)
        key = re.sub(r"\s+", "", fact)[:120]
        if not fact or key in seen:
            continue
        seen.add(key)
        source_id = ref_match.group(1) if ref_match else "?"
        target_dimension = "市场规模与增速"
        for dimension, keywords in dim_keywords.items():
            if any(keyword in fact for keyword in keywords):
                target_dimension = dimension
                break
        dimensions[target_dimension].append({"text": fact, "source": source_id, "time": ""})

    title_match = re.search(r"#\s+(.+?)行业研究报告", content)
    topic = title_match.group(1).strip() if title_match else "未知主题"
    return {"topic": topic, "sources": sources, "dimensions": dimensions}


def latest_input(output_dir: Path) -> tuple[str, str]:
    packages = sorted(output_dir.glob("*.writer_package.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    if packages:
        return "package", str(packages[0])
    reports = sorted(output_dir.glob("*.md"), key=lambda item: item.stat().st_mtime, reverse=True)
    if reports:
        return "report", str(reports[0])
    return "", ""


async def main() -> int:
    load_dotenv(PIPELINE_ROOT / ".env")
    parser = argparse.ArgumentParser(description="ReformatterAgent standalone report cleaner")
    parser.add_argument("--package", help="writer_package.json path")
    parser.add_argument("--report", help="messy report .md path, fallback mode")
    parser.add_argument("--output", help="output markdown path")
    parser.add_argument("--stream", action="store_true", help="stream LangChain client output when a client is supplied by custom code")
    args = parser.parse_args()

    if not args.package and not args.report:
        kind, value = latest_input(PIPELINE_ROOT / "output" / "full_reports")
        if kind == "package":
            args.package = value
            print(f"自动选择最新 writer_package: {args.package}")
        elif kind == "report":
            args.report = value
            print(f"自动选择最新报告: {args.report}")
        else:
            print("未找到可处理的文件，请指定 --package 或 --report。")
            return 2

    if args.package:
        print(f"从 writer_package.json 提取证据: {args.package}")
        clean_evidence = extract_clean_evidence(args.package)
    else:
        print(f"从脏报告提取证据（备用模式）: {args.report}")
        clean_evidence = build_evidence_from_messy_report(args.report)

    print(f"主题: {clean_evidence.get('topic') or '未知主题'}")
    for dimension in REPORT_DIMENSIONS:
        print(f"  {dimension}: {len(clean_evidence.get('dimensions', {}).get(dimension, []))} 条证据")

    final_report = await run_reformatter(clean_evidence, llm_client=None, stream=bool(args.stream))
    validation = validate_reformatted_report(final_report, clean_evidence.get("sources", []), clean_evidence)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    topic_safe = re.sub(r'[<>:"/\\|?*\x00-\x1f\s]+', "_", str(clean_evidence.get("topic") or "report")).strip("_")[:40]
    output_path = Path(args.output) if args.output else PIPELINE_ROOT / "output" / "full_reports" / f"{timestamp}_{topic_safe}_clean.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(final_report, encoding="utf-8")
    print(f"洁净报告已保存: {output_path}")
    print(f"校验: {'OK' if validation.get('passed') else 'WARN'} {validation}")
    return 0 if validation.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
