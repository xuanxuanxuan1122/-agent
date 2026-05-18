from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_pipeline.agents.writer_agent_clean import build_writer_report


GOLDEN_SAMPLES = [
    {
        "slug": "industry_robotics",
        "query": "智能农业机器人行业机会分析",
        "report_family": "industry_deep_report",
        "dimensions": [
            ("demand", "需求真实性", "谁会买单，需求是否刚性？"),
            ("supply", "供给缺口", "现有产品和服务能否满足核心场景？"),
            ("entry", "进入窗口", "什么场景适合作为优先切入口？"),
        ],
    },
    {
        "slug": "company_dd",
        "query": "某机器人公司是否值得投资",
        "report_family": "company_due_diligence_report",
        "dimensions": [
            ("business", "业务真实性", "公司的收入和客户是否真实可验证？"),
            ("financial", "财务质量", "财务表现是否支撑投资判断？"),
            ("risk", "风险事项", "是否存在需要排除的重大风险？"),
        ],
    },
    {
        "slug": "product_strawberry_robot",
        "query": "草莓采摘机器人产品机会",
        "report_family": "product_research_report",
        "dimensions": [
            ("user_pain", "用户痛点", "采摘场景中最刚性的痛点是什么？"),
            ("roi", "ROI验证", "客户是否能算出可接受的回收期？"),
            ("roadmap", "产品切入点", "短期应该切入哪个产品形态？"),
        ],
    },
    {
        "slug": "policy_low_altitude",
        "query": "低空经济政策对无人机配送的影响",
        "report_family": "policy_impact_report",
        "dimensions": [
            ("policy_terms", "政策条款", "政策具体改变了哪些准入或运营条件？"),
            ("transmission", "执行传导", "政策如何传导到预算、审批和订单？"),
            ("affected", "影响主体", "哪些主体会先受益或受约束？"),
        ],
    },
    {
        "slug": "consumer_entry",
        "query": "某消费品类市场进入机会",
        "report_family": "consumer_market_report",
        "dimensions": [
            ("consumer", "消费需求", "目标用户为什么会购买？"),
            ("channel", "渠道验证", "哪些渠道能低成本验证需求？"),
            ("positioning", "进入定位", "应该避开哪些同质化竞争？"),
        ],
    },
]


def _plan(sample: Dict[str, Any]) -> Dict[str, Any]:
    dimensions = [
        {
            "dimension_id": dim_id,
            "dimension_name": title,
            "purpose": question,
            "must_have_terms": [title],
            "forbidden_terms": [],
        }
        for dim_id, title, question in sample["dimensions"]
    ]
    hypotheses = [
        {
            "hypothesis_id": f"H{index}",
            "statement": title,
            "hypothesis_statement": f"{title}能够支撑“{sample['query']}”的核心判断。",
            "proof_standard": "medium",
            "counter_evidence_required": True,
            "required_source_levels": ["A", "B"],
            "required_evidence_types": ["official_data", "market_research", "customer_case"],
            "metric_definitions": [
                {
                    "metric_name": f"{title}验证项",
                    "subject": title,
                    "scope": "sample",
                    "period": "2026-05-09",
                    "unit": "signal",
                }
            ],
            "decision_use": "research",
            "evidence_goal_ids": [dim_id],
            "falsification_triggers": ["反向案例", "A/B来源不支持", "指标口径不可比"],
        }
        for index, (dim_id, title, question) in enumerate(sample["dimensions"], start=1)
    ]
    return {
        "query": sample["query"],
        "research_type": sample["slug"],
        "decision_context": "research",
        "report_family": sample["report_family"],
        "research_object": sample["query"],
        "core_question": sample["query"],
        "hypotheses": hypotheses,
        "proof_standards": {
            "strong": {"required_ab_sources": 2, "counter_evidence_required": True, "metric_scope_period_unit_required": True},
            "medium": {"required_ab_sources": 1, "counter_evidence_required": True, "metric_scope_period_unit_required": True},
            "weak": {"required_ab_sources": 0, "counter_evidence_required": False, "appendix_or_followup_only": True},
        },
        "source_requirements": ["official_data", "filing_company", "market_research", "news_event"],
        "report_depth_target": "deep",
        "dimensions": dimensions,
        "evidence_goals": [
            {
                "goal_id": dim_id,
                "dimension_id": dim_id,
                "dimension_name": title,
                "question": question,
                "expected_metrics": [title],
                "source_priority": ["official", "filing", "research_report"],
                "freshness": "recent",
                "min_sources": 2,
                "evidence_type": "data",
            }
            for dim_id, title, question in sample["dimensions"]
        ],
        "search_tasks": [
            {
                "task_id": f"{dim_id}_001",
                "agent": "iqs",
                "dimension_id": dim_id,
                "dimension_name": title,
                "query": f"{sample['query']} {title} {question}",
                "evidence_goal": question,
                "intent": "analysis",
                "must_have_terms": [title],
                "forbidden_terms": [],
                "source_priority": ["official", "filing", "research_report"],
            }
            for dim_id, title, question in sample["dimensions"]
        ],
    }


def _evidence(sample: Dict[str, Any], research_plan: Dict[str, Any]) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []
    index = 1
    for dim in research_plan["dimensions"]:
        title = dim["dimension_name"]
        for row in range(1, 4):
            items.append(
                {
                    "evidence_id": f"EV-{index:03d}",
                    "dimension": title,
                    "fact": f"{sample['query']}在“{title}”上存在第{row}个可验证信号，需要结合主体、场景和口径判断。",
                    "metric": f"{title}验证项",
                    "value": f"信号{row}",
                    "source_level": "A" if row == 1 else "B",
                    "confidence": 0.82 if row == 1 else 0.72,
                    "source": {
                        "title": f"{sample['query']} smoke source {index}",
                        "date": "2026-05-09",
                        "url": f"https://example.com/smoke/{sample['slug']}/{index}",
                        "credibility": "A" if row == 1 else "B",
                    },
                }
            )
            index += 1
    return {"research_plan": research_plan, "analysis_ready_evidence": items}


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def run_smoke(output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("REPORT_PIPELINE_PAYLOAD_MODE", "full")
    os.environ.setdefault("REPORT_DEBUG_PAYLOAD_MODE", "full")
    summary: Dict[str, Any] = {"output_dir": str(output_dir), "samples": []}
    for sample in GOLDEN_SAMPLES:
        sample_dir = output_dir / sample["slug"]
        sample_dir.mkdir(parents=True, exist_ok=True)
        research_plan = _plan(sample)
        search_task_schedule = {
            "max_tasks_per_lane": 8,
            "scheduled_tasks": [
                {**task, "scheduled_lane": task.get("agent") or "iqs"}
                for task in research_plan.get("search_tasks", [])
            ],
            "dropped_tasks": [],
            "scheduled_count": len(research_plan.get("search_tasks", [])),
            "dropped_count": 0,
        }
        evidence_package = _evidence(sample, research_plan)
        structured_analysis = {"research_plan": research_plan, "evidence_analyses": evidence_package["analysis_ready_evidence"]}
        writer_report = build_writer_report(
            query=sample["query"],
            evidence_package=evidence_package,
            structured_analysis=structured_analysis,
            report_plan={"report_family": sample["report_family"], "research_object": sample["query"]},
            search_task_schedule=search_task_schedule,
        )
        artifacts = {
            "01_research_plan.json": research_plan,
            "02_report_blueprint.json": writer_report.get("report_blueprint"),
            "03_search_task_schedule.json": writer_report.get("search_task_schedule"),
            "04_chapter_evidence_packages.json": writer_report.get("chapter_evidence_packages"),
            "05_micro_layouts.json": writer_report.get("micro_layouts"),
            "06_table_packages.json": writer_report.get("table_packages"),
            "07_argument_units.json": writer_report.get("argument_units"),
            "08_chapter_packages.json": writer_report.get("chapter_packages"),
            "09_package_quality_report.json": writer_report.get("package_quality_report"),
            "11_qa_result.json": writer_report.get("qa_result"),
            "12_pipeline_debug.json": writer_report.get("debug_snapshot"),
            "evidence_package.json": evidence_package,
            "decision_package.json": writer_report.get("decision_package"),
            "risk_package.json": writer_report.get("risk_package"),
        }
        for filename, payload in artifacts.items():
            _write_json(sample_dir / filename, payload)
        (sample_dir / "10_writer_report.md").write_text(str(writer_report.get("report_markdown") or ""), encoding="utf-8")
        (sample_dir / "writer_report.md").write_text(str(writer_report.get("report_markdown") or ""), encoding="utf-8")
        summary["samples"].append(
            {
                "slug": sample["slug"],
                "status": writer_report.get("report_status"),
                "qa_passed": bool((writer_report.get("qa_result") or {}).get("passed")),
                "package_quality_passed": bool((writer_report.get("package_quality_report") or {}).get("passed")),
                "estimated_chars": writer_report.get("estimated_chars"),
            }
        )
    _write_json(output_dir / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run offline split-pipeline smoke samples.")
    parser.add_argument(
        "--output-dir",
        default=f"output/split_pipeline_smoke/{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Directory to write smoke artifacts.",
    )
    args = parser.parse_args()
    summary = run_smoke(Path(args.output_dir))
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
