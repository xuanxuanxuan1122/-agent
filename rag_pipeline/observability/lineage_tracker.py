from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

from rag_pipeline.observability.module_probe_models import as_dict, as_list

SCHEMA_VERSION = "lineage_graph_v1"


def _first_dict(*values: Any) -> Dict[str, Any]:
    for value in values:
        payload = as_dict(value)
        if payload:
            return payload
    return {}


def _raw_metadata(writer_package: Dict[str, Any], writer_report: Dict[str, Any]) -> Dict[str, Any]:
    raw_output = _first_dict(
        writer_package.get("raw_output"),
        as_dict(writer_report.get("render_artifacts")).get("raw_output"),
        as_dict(writer_package.get("evidence_package")).get("raw_output"),
    )
    return _first_dict(raw_output.get("metadata"), writer_package.get("metadata"), as_dict(writer_package.get("writer_report")).get("metadata"))


def _id_values(value: Any) -> List[str]:
    output: List[str] = []
    for item in as_list(value):
        text = str(item or "").strip()
        if text:
            output.append(text)
    if not output and isinstance(value, str) and value.strip():
        output.append(value.strip())
    return output


def _add_node(nodes: Dict[str, Dict[str, Any]], node_id: Any, node_type: str, **attrs: Any) -> None:
    text = str(node_id or "").strip()
    if not text:
        return
    existing = nodes.setdefault(text, {"id": text, "type": node_type})
    existing.setdefault("type", node_type)
    for key, value in attrs.items():
        if value not in (None, "", [], {}):
            existing[key] = value


def _add_edge(
    edges: Dict[Tuple[str, str, str], Dict[str, Any]],
    nodes: Dict[str, Dict[str, Any]],
    from_id: Any,
    to_id: Any,
    relation: str,
    *,
    from_type: str,
    to_type: str,
) -> None:
    source = str(from_id or "").strip()
    target = str(to_id or "").strip()
    if not source or not target:
        return
    _add_node(nodes, source, from_type)
    _add_node(nodes, target, to_type)
    edges[(source, target, relation)] = {
        "from": source,
        "to": target,
        "relation": relation,
        "from_type": from_type,
        "to_type": to_type,
    }


def _claim_units(writer_package: Dict[str, Any], writer_report: Dict[str, Any]) -> List[Dict[str, Any]]:
    structured = _first_dict(
        writer_package.get("structured_analysis"),
        writer_report.get("structured_analysis"),
        as_dict(writer_report.get("render_artifacts")).get("structured_analysis"),
    )
    return [as_dict(item) for item in as_list(structured.get("claim_units")) if isinstance(item, dict)]


def _sections(writer_package: Dict[str, Any], writer_report: Dict[str, Any]) -> List[Dict[str, Any]]:
    chapters = as_list(writer_package.get("chapter_packages")) or as_list(as_dict(writer_report.get("render_artifacts")).get("chapter_packages"))
    output: List[Dict[str, Any]] = []
    for chapter in chapters:
        chapter_payload = as_dict(chapter)
        for section in as_list(chapter_payload.get("sections")):
            section_payload = as_dict(section).copy()
            if "chapter_id" not in section_payload and chapter_payload.get("chapter_id"):
                section_payload["chapter_id"] = chapter_payload.get("chapter_id")
            output.append(section_payload)
    return output


def _facts(writer_package: Dict[str, Any], writer_report: Dict[str, Any]) -> List[Dict[str, Any]]:
    evidence_package = _first_dict(writer_package.get("evidence_package"), as_dict(writer_report.get("render_artifacts")).get("evidence_package"))
    return [
        as_dict(item)
        for item in (
            as_list(evidence_package.get("analysis_ready_evidence"))
            or as_list(evidence_package.get("clean_evidence_list"))
            or as_list(evidence_package.get("raw_data_points"))
        )
        if isinstance(item, dict)
    ]


def build_lineage_graph(
    *,
    run_id: str,
    writer_package: Dict[str, Any],
    writer_report: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build a diagnostic-only ID lineage graph from the writer package."""

    package = as_dict(writer_package)
    report = as_dict(writer_report) or as_dict(package.get("writer_report"))
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    metadata = _raw_metadata(package, report)

    for task in as_list(metadata.get("search_tasks")):
        payload = as_dict(task)
        task_id = str(payload.get("task_id") or payload.get("id") or payload.get("query") or "").strip()
        if not task_id:
            continue
        _add_node(nodes, task_id, "search_task", proof_role=payload.get("proof_role") or payload.get("lane_type"))
        for req_id in _id_values(payload.get("requirement_id") or payload.get("requirement_ids")):
            _add_edge(edges, nodes, req_id, task_id, "plans_search", from_type="requirement", to_type="search_task")

    for fact in _facts(package, report):
        fact_id = str(fact.get("fact_id") or fact.get("evidence_id") or fact.get("id") or "").strip()
        if not fact_id:
            continue
        _add_node(nodes, fact_id, "fact", source_level=fact.get("source_level"))
        for req_id in _id_values(fact.get("requirement_id") or fact.get("requirement_ids")):
            _add_edge(edges, nodes, req_id, fact_id, "satisfies_requirement", from_type="requirement", to_type="fact")
        for task_id in _id_values(fact.get("search_task_id") or fact.get("task_id")):
            _add_edge(edges, nodes, task_id, fact_id, "retrieves_fact", from_type="search_task", to_type="fact")
        for source_id in _id_values(fact.get("source_id") or fact.get("source_ref") or fact.get("canonical_source_id")):
            _add_edge(edges, nodes, source_id, fact_id, "provides_fact", from_type="source", to_type="fact")

    for claim in _claim_units(package, report):
        claim_id = str(claim.get("claim_id") or claim.get("id") or "").strip()
        if not claim_id:
            continue
        _add_node(nodes, claim_id, "claim", claim_strength=claim.get("claim_strength"))
        for req_id in _id_values(claim.get("requirement_ids") or claim.get("requirement_id")):
            _add_edge(edges, nodes, req_id, claim_id, "informs_claim", from_type="requirement", to_type="claim")
        for fact_id in _id_values(claim.get("fact_ids") or claim.get("evidence_refs") or claim.get("used_fact_refs")):
            _add_edge(edges, nodes, fact_id, claim_id, "supports_claim", from_type="fact", to_type="claim")
        for source_id in _id_values(claim.get("source_ids") or claim.get("citation_refs") or claim.get("sources")):
            _add_edge(edges, nodes, source_id, claim_id, "cited_by_claim", from_type="source", to_type="claim")

    for section in _sections(package, report):
        section_id = str(section.get("section_id") or section.get("id") or "").strip()
        if not section_id:
            continue
        _add_node(nodes, section_id, "section", chapter_id=section.get("chapter_id"))
        for claim_id in _id_values(section.get("claim_ids") or section.get("claim_id")):
            _add_edge(edges, nodes, claim_id, section_id, "renders_section", from_type="claim", to_type="section")
        for fact_id in _id_values(section.get("used_fact_refs") or section.get("evidence_refs") or section.get("fact_ids")):
            _add_edge(edges, nodes, fact_id, section_id, "used_in_section", from_type="fact", to_type="section")
        for req_id in _id_values(section.get("requirement_ids") or section.get("requirement_id")):
            _add_edge(edges, nodes, req_id, section_id, "shapes_section", from_type="requirement", to_type="section")

    for gap in as_list(package.get("score_gaps")) or as_list(report.get("score_gaps")):
        payload = as_dict(gap)
        gap_id = str(payload.get("gap_id") or payload.get("id") or "").strip()
        if not gap_id:
            continue
        _add_node(nodes, gap_id, "score_gap", status=payload.get("status"))
        for req_id in _id_values(payload.get("requirement_id") or payload.get("requirement_ids")):
            _add_edge(edges, nodes, req_id, gap_id, "has_score_gap", from_type="requirement", to_type="score_gap")

    edge_values = list(edges.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": str(run_id or "run").strip() or "run",
        "nodes": sorted(nodes.values(), key=lambda item: (str(item.get("type")), str(item.get("id")))),
        "edges": sorted(edge_values, key=lambda item: (str(item.get("from")), str(item.get("to")), str(item.get("relation")))),
        "coverage": {
            "node_count": len(nodes),
            "edge_count": len(edge_values),
            "requirement_to_search_task_edge_count": sum(1 for item in edge_values if item.get("relation") == "plans_search"),
            "fact_to_claim_edge_count": sum(1 for item in edge_values if item.get("relation") == "supports_claim"),
            "claim_to_section_edge_count": sum(1 for item in edge_values if item.get("relation") == "renders_section"),
            "score_gap_edge_count": sum(1 for item in edge_values if item.get("relation") == "has_score_gap"),
        },
        "diagnostic_only": True,
        "must_not_render": True,
        "public_text_allowed": False,
    }
