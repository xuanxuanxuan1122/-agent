from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_pipeline.agents.claim_builder_agent import run_claim_builder_agent
from rag_pipeline.agents.chapter_argument_agent import run_chapter_argument_agent
from rag_pipeline.agents.final_writer_agent import run_final_writer_agent
from rag_pipeline.flows.report.final_audit_agent import run_final_audit
from rag_pipeline.flows.report.full_report import append_final_audit_note


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _quality_score(markdown: str, package: Dict[str, Any]) -> int:
    writer_report = _as_dict(package.get("writer_report"))
    for value in (
        writer_report.get("quality_score"),
        _as_dict(writer_report.get("validation")).get("quality_score"),
        _as_dict(writer_report.get("qa_result")).get("score"),
    ):
        try:
            return max(0, min(100, int(float(value))))
        except (TypeError, ValueError):
            continue
    match = re.search(r"质量总分[:：]\s*(\d{1,3})\s*/\s*100", markdown)
    return max(0, min(100, int(match.group(1)))) if match else 60


def _scorecard(markdown: str, package: Dict[str, Any]) -> str:
    health = _as_dict(package.get("evidence_health_summary"))
    score = _quality_score(markdown, package)
    clean = "是" if score >= 90 else "否"
    return "\n".join(
        [
            "## 报告质量评分与证据限制",
            "",
            f"- 质量总分：{score}/100",
            f"- Clean 资格：{clean}",
            f"- 可进入分析材料：{health.get('analysis_ready_count') or 0} 条",
            f"- 清洗后事实：{health.get('clean_fact_count') or 0} 条",
            f"- 可追溯 A/B 来源：{health.get('traceable_ab_source_count') or 0} 个",
            "- 说明：本文件由既有 writer_package 低成本重建，未重新执行联网检索。",
            "",
        ]
    )


def regenerate(package_path: Path, output_path: Path | None = None) -> Path:
    package = json.loads(package_path.read_text(encoding="utf-8"))
    chapter_evidence_packages = _as_list(package.get("chapter_evidence_packages"))
    micro_layouts = _as_list(package.get("micro_layouts"))
    table_packages = _as_list(package.get("table_packages"))
    structured_analysis = _as_dict(package.get("structured_analysis"))
    report_blueprint = _as_dict(package.get("report_blueprint"))
    source_registry = _as_list(package.get("source_registry")) or _as_list(_as_dict(package.get("writer_report")).get("source_registry"))

    argument_units = run_claim_builder_agent(
        chapter_evidence_packages=chapter_evidence_packages,
        micro_layouts=micro_layouts,
        structured_analysis=structured_analysis,
    )
    chapter_packages = run_chapter_argument_agent(
        report_blueprint=report_blueprint,
        micro_layouts=micro_layouts,
        argument_units=argument_units,
        table_packages=table_packages,
        chapter_evidence_packages=chapter_evidence_packages,
    )
    writer_output = run_final_writer_agent(
        query=str(package.get("query") or ""),
        report_blueprint=report_blueprint,
        chapter_packages=chapter_packages,
        table_packages=table_packages,
        decision_package=_as_dict(_as_dict(package.get("writer_report")).get("decision_package")),
        risk_package=_as_dict(_as_dict(package.get("writer_report")).get("risk_package")),
        appendix_package={
            **_as_dict(_as_dict(package.get("writer_report")).get("appendix_package")),
            "metric_normalization_table": _as_list(_as_dict(package.get("evidence_package")).get("metric_normalization_table")),
        },
        source_registry=source_registry,
    )
    markdown = str(writer_output.get("report_markdown") or "").strip()
    if not markdown:
        raise RuntimeError("Regenerated writer output is empty.")
    if "## 报告质量评分与证据限制" not in markdown:
        first_line, _, rest = markdown.partition("\n")
        markdown = f"{first_line}\n{_scorecard(markdown, package)}{rest.lstrip()}".strip()

    audit_package = {
        **package,
        "writer_report": {
            **_as_dict(package.get("writer_report")),
            "report_markdown": markdown,
            "source_registry": writer_output.get("source_registry") or [],
        },
        "source_registry": writer_output.get("source_registry") or [],
    }
    final_audit = run_final_audit(
        report_markdown=markdown,
        validation=_as_dict(_as_dict(package.get("writer_report")).get("validation")),
        clean_evidence=None,
        writer_package_payload=audit_package,
        query=str(package.get("query") or ""),
    )
    audited_markdown = append_final_audit_note(markdown, final_audit)
    if audited_markdown == markdown and final_audit.get("enabled"):
        audit = _as_dict(final_audit.get("audit"))
        audited_markdown = (
            f"{markdown}\n\n## 最终审查补充\n\n"
            f"- 最终审查状态：{final_audit.get('status') or audit.get('status') or 'unknown'}\n"
            f"- 洁净版资格：{'暂不建议自动交付' if final_audit.get('blocked') else '未发现阻断洁净版的问题'}\n"
            f"- 审查摘要：{audit.get('summary') or _as_dict(final_audit.get('deterministic_audit')).get('summary') or ''}"
        ).strip()
    markdown = audited_markdown

    output = output_path or package_path.with_name(package_path.name.replace(".writer_package.json", "_regenerated_report.md"))
    output.write_text(markdown, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate a formal report from an existing writer_package without rerunning retrieval.")
    parser.add_argument("writer_package", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = regenerate(args.writer_package.resolve(), args.output.resolve() if args.output else None)
    print(str(output))


if __name__ == "__main__":
    main()
