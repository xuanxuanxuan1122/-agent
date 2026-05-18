from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

from .analytics_contracts import as_dict, as_list
from .competitor_analysis_agent import run_competitor_analysis_agent
from .investor_insight_agent import run_investor_insight_agent
from .market_analytics_agent import run_market_analytics_agent
from .regulatory_impact_agent import run_regulatory_impact_agent
from .technology_roadmap_agent import run_technology_roadmap_agent


AGENT_NAME = "analytics_orchestrator"


def _enabled() -> bool:
    raw = os.getenv("REPORT_ENABLE_ANALYTICS", "true")
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _has_payload(output: Dict[str, Any]) -> bool:
    return bool(as_list(output.get("metrics")) or as_list(output.get("calculations")) or as_list(output.get("tables")))


def run_analytics_agents(
    *,
    report_blueprint: Optional[Dict[str, Any]] = None,
    chapter_evidence_packages: Optional[Sequence[Dict[str, Any]]] = None,
    evidence_graph: Optional[Dict[str, Any]] = None,
    metric_normalization_table: Optional[Sequence[Dict[str, Any]]] = None,
    coverage_matrix: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    if not _enabled():
        return []
    outputs: List[Dict[str, Any]] = []
    market_output = run_market_analytics_agent(
        report_blueprint=as_dict(report_blueprint),
        chapter_evidence_packages=list(chapter_evidence_packages or []),
        evidence_graph=as_dict(evidence_graph),
        metric_normalization_table=list(metric_normalization_table or []),
        coverage_matrix=list(coverage_matrix or []),
    )
    if _has_payload(market_output) or str(os.getenv("REPORT_INCLUDE_EMPTY_ANALYTICS", "false")).strip().lower() in {"1", "true", "yes", "on"}:
        outputs.append(market_output)
    agent_calls = [
        run_competitor_analysis_agent,
        run_regulatory_impact_agent,
        run_technology_roadmap_agent,
        run_investor_insight_agent,
    ]
    for agent_call in agent_calls:
        output = agent_call(
            report_blueprint=as_dict(report_blueprint),
            chapter_evidence_packages=list(chapter_evidence_packages or []),
            evidence_graph=as_dict(evidence_graph),
            metric_normalization_table=list(metric_normalization_table or []),
            coverage_matrix=list(coverage_matrix or []),
            market_analytics=outputs,
        )
        if _has_payload(output) or str(os.getenv("REPORT_INCLUDE_EMPTY_ANALYTICS", "false")).strip().lower() in {"1", "true", "yes", "on"}:
            outputs.append(output)
    return outputs
