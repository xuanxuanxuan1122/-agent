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

from rag_pipeline.agents.analysis_agent import ensure_valid_structured_analysis, run_analysis_agent
from rag_pipeline.agents.chapter_evidence_builder import build_chapter_evidence_packages_from_evidence_package
from rag_pipeline.agents.claim_builder_agent import run_claim_builder_agent
from rag_pipeline.agents.chapter_argument_agent import run_chapter_argument_agent
from rag_pipeline.agents.micro_layout_agent import run_micro_layout_agent
from rag_pipeline.agents.final_writer_agent import run_final_writer_agent
from rag_pipeline.flows.report.final_audit_agent import run_final_audit
from rag_pipeline.flows.report.full_report import render_score_markdown, write_formal_markdown, write_score_markdown


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _renumber_public_chapter_headings(markdown: str) -> str:
    chapter_index = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal chapter_index
        chapter_index += 1
        title = re.sub(r"^\d+\.\s*", "", match.group(1).strip())
        return f"## {chapter_index}. {title}"

    return re.sub(r"^##\s+\d+\.\s+(.+?)\s*$", replace, markdown or "", flags=re.M)


def _compact_sample(value: Any) -> Any:
    if isinstance(value, dict) and isinstance(value.get("sample"), list):
        return list(value.get("sample") or [])
    return value


def _expand_compact_evidence_package(evidence_package: Dict[str, Any], *, query: str = "") -> Dict[str, Any]:
    package = dict(evidence_package or {})
    if query and not package.get("query"):
        package["query"] = query
    for key in ("raw_data_points", "normalized_evidence", "analysis_ready_evidence", "clean_evidence_list"):
        if key in package:
            package[key] = _compact_sample(package.get(key))
    return package


def _structured_analysis_has_public_material(structured_analysis: Dict[str, Any]) -> bool:
    analysis = _as_dict(structured_analysis)
    nested = _as_dict(analysis.get("structured_analysis"))
    candidates = [analysis, nested]
    for payload in candidates:
        if _as_list(payload.get("claim_units")):
            return True
        if _as_list(payload.get("chapter_insights")):
            return True
        if _as_dict(payload.get("dimension_synthesis")):
            return True
        if _as_list(payload.get("evidence_analyses")):
            return True
    return False


def _normalize_structured_analysis(
    package: Dict[str, Any],
    *,
    llm_config: Dict[str, Any] | None = None,
    force_analysis: bool = False,
) -> Dict[str, Any]:
    structured_analysis = _as_dict(package.get("structured_analysis"))
    nested = _as_dict(structured_analysis.get("structured_analysis"))
    if nested and not _structured_analysis_has_public_material(structured_analysis):
        structured_analysis = {**structured_analysis, **nested}
    evidence_package = _expand_compact_evidence_package(
        _as_dict(package.get("evidence_package")),
        query=str(package.get("query") or ""),
    )
    if _as_list(package.get("chapter_evidence_packages")) and not _as_list(evidence_package.get("chapter_evidence_packages")):
        evidence_package["chapter_evidence_packages"] = _as_list(package.get("chapter_evidence_packages"))
    if _structured_analysis_has_public_material(structured_analysis) and not force_analysis:
        normalized = ensure_valid_structured_analysis(
            structured_analysis,
            evidence_package,
            rebuild_reason="regenerate_package_analysis_contract",
        )
        package["analysis_rebuild_diagnostics"] = {
            "triggered": bool(normalized.get("analysis_rebuilt_from_evidence")),
            "source": "regenerate_structured_analysis_contract",
            "reason": "analysis_contract_validation",
            "analysis_contract_status": _as_dict(normalized.get("analysis_contract_status")),
            "analysis_stage_diagnostics": _as_dict(normalized.get("analysis_stage_diagnostics")),
        }
        return normalized
    analysis_state = run_analysis_agent(
        evidence_package,
        query=str(package.get("query") or ""),
        llm_config=llm_config,
    )
    rebuilt = ensure_valid_structured_analysis(
        _as_dict(analysis_state.get("structured_analysis")),
        evidence_package,
        rebuild_reason="regenerate_missing_structured_analysis_material",
    )
    package["analysis_rebuild_diagnostics"] = {
        "triggered": True,
        "source": "state_or_writer_package_evidence_package",
        "reason": "missing_structured_analysis_material",
        "analysis_errors": _as_list(analysis_state.get("errors")),
        "llm_analysis_status": _as_dict(analysis_state.get("metadata")).get("llm_analysis_status"),
    }
    return rebuilt


def _load_rebuild_package(input_path: Path) -> Dict[str, Any]:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    if input_path.name.endswith(".state.json"):
        sibling_package: Dict[str, Any] = {}
        sibling = input_path.with_name(input_path.name[: -len(".state.json")] + ".writer_package.json")
        if sibling.exists():
            try:
                sibling_package = json.loads(sibling.read_text(encoding="utf-8-sig"))
            except Exception:
                sibling_package = {}
        raw = _as_dict(data.get("raw_output"))
        writer_report = _as_dict(data.get("writer_report")) or _as_dict(raw.get("writer_report"))
        sibling_writer_report = _as_dict(sibling_package.get("writer_report"))
        evidence_package = (
            _as_dict(data.get("evidence_package"))
            or _as_dict(raw.get("evidence_package"))
            or _as_dict(_as_dict(raw.get("writer_handoff_package")).get("evidence_package"))
            or _as_dict(sibling_package.get("evidence_package"))
        )
        package = {
            "query": data.get("query") or raw.get("query") or "",
            "stage_status": data.get("stage_status") or {},
            "evidence_package": _expand_compact_evidence_package(evidence_package, query=str(data.get("query") or raw.get("query") or "")),
            "evidence_health_summary": _as_dict(data.get("evidence_health_summary")) or _as_dict(evidence_package.get("evidence_health_summary")),
            "source_registry": _as_list(data.get("source_registry")) or _as_list(writer_report.get("source_registry")) or _as_list(sibling_package.get("source_registry")) or _as_list(sibling_writer_report.get("source_registry")),
            "structured_analysis": _as_dict(data.get("structured_analysis")) or _as_dict(raw.get("structured_analysis")) or _as_dict(sibling_package.get("structured_analysis")),
            "report_blueprint": _as_dict(data.get("report_blueprint")) or _as_dict(raw.get("report_blueprint")) or _as_dict(writer_report.get("report_blueprint")) or _as_dict(sibling_package.get("report_blueprint")),
            "chapter_evidence_packages": _as_list(sibling_package.get("chapter_evidence_packages")) or _as_list(data.get("chapter_evidence_packages")) or _as_list(raw.get("chapter_evidence_packages")) or _as_list(writer_report.get("chapter_evidence_packages")),
            "micro_layouts": _as_list(data.get("micro_layouts")) or _as_list(raw.get("micro_layouts")) or _as_list(writer_report.get("micro_layouts")) or _as_list(sibling_package.get("micro_layouts")),
            "table_packages": _as_list(data.get("table_packages")) or _as_list(raw.get("table_packages")) or _as_list(writer_report.get("table_packages")) or _as_list(sibling_package.get("table_packages")),
            "argument_units": _as_list(data.get("argument_units")) or _as_list(raw.get("argument_units")) or _as_list(writer_report.get("argument_units")) or _as_list(sibling_package.get("argument_units")),
            "chapter_packages": _as_list(data.get("chapter_packages")) or _as_list(raw.get("chapter_packages")) or _as_list(writer_report.get("chapter_packages")) or _as_list(sibling_package.get("chapter_packages")),
            "writer_report": {**sibling_writer_report, **writer_report},
            "review_result": _as_dict(data.get("review_result")),
            "reformatter_result": _as_dict(data.get("reformatter_result")),
            "legacy_package_incomplete": False,
        }
        return package
    package = data
    package["evidence_package"] = _expand_compact_evidence_package(_as_dict(package.get("evidence_package")), query=str(package.get("query") or ""))
    if not _as_dict(package.get("evidence_package")):
        package["legacy_package_incomplete"] = True
    return package


def _default_output_path(input_path: Path) -> Path:
    name = input_path.name
    for suffix in (".writer_package.json", ".state.json"):
        if name.endswith(suffix):
            return input_path.with_name(name[: -len(suffix)] + "_regenerated_report.md")
    return input_path.with_name(input_path.stem + "_regenerated_report.md")


def _fact_text(item: Dict[str, Any]) -> str:
    return re.sub(
        r"\s+",
        " ",
        str(item.get("fact") or item.get("clean_fact") or item.get("content") or item.get("evidence") or item.get("summary") or "").strip(),
    )


def _bad_fact_text(text: str) -> bool:
    if not text:
        return True
    return any(
        re.search(pattern, text, flags=re.I)
        for pattern in [
            r"\u519c\u4e1a\u4eba\u5de5\u667a\u80fd",
            r"\u6570\u636e\u6295\u6bd2",
            r"Scribd",
            r"\u53d1\u73b0\u62a5\u544a",
            r"\u7eba\u7ec7",
            r"\u667a\u80fd\u624b\u673a",
            r"SEO",
            r"^URL[:\uff1a]",
        ]
    )


def _invalid_metric(item: Dict[str, Any]) -> bool:
    metric = str(item.get("metric") or item.get("indicator") or "").strip()
    value = str(item.get("value") or item.get("display_value") or item.get("numeric_value") or "").strip()
    fact = _fact_text(item)
    metric_lower = metric.lower()
    if str(item.get("metric_validation_status") or "").lower() == "invalid":
        return True
    if metric_lower in {"source_check", "status", "http_status", "response_code"} and re.fullmatch(r"[1-5]\d{2}", value):
        return True
    if re.search(r"\bsource_check\s*[:=]\s*[1-5]\d{2}\b", fact, flags=re.I):
        return True
    if value and re.fullmatch(r"-?\d{1,3}(?:\.0)?", value) and re.search(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|T\d{1,2}:\d{2}", fact):
        return True
    if metric in {"\u5173\u952e\u4e8b\u5b9e", "\u653f\u7b56\u76d1\u7ba1", "\u653f\u7b56\u76ee\u6807"} and re.fullmatch(r"-?\d{1,3}(?:\.0)?", value):
        return True
    if re.search(r"\u6210\u672c", metric) and (re.search(r"\u5bb6$", value) or not fact):
        return True
    if re.search(r"\u5e02\u573a\u89c4\u6a21|\u878d\u8d44", metric) and re.search(r"%", value):
        return True
    return False


def _chapter_match(item: Dict[str, Any], package: Dict[str, Any]) -> bool:
    cid = str(package.get("chapter_id") or "")
    title = str(package.get("chapter_title") or package.get("chapter_question") or "")
    dim = str(item.get("chapter_id") or item.get("dimension") or item.get("hypothesis_id") or "")
    if cid and dim and cid == dim:
        return True
    if title and dim and (dim in title or title in dim):
        return True
    return False


def _evidence_ref(item: Dict[str, Any]) -> str:
    for key in ("source_ref", "citation_ref", "ref", "evidence_id"):
        value = str(item.get(key) or "").strip()
        if value:
            return value
    return ""


def _normalize_seed_evidence(item: Dict[str, Any]) -> Dict[str, Any]:
    copied = dict(item)
    fact = _fact_text(copied)
    copied["fact"] = fact
    copied.setdefault("ref", copied.get("evidence_id") or copied.get("source_ref") or "")
    copied.setdefault("source_ref", copied.get("citation_ref") or copied.get("source_ref") or copied.get("ref") or "")
    copied.setdefault("source_level", copied.get("source_level") or copied.get("credibility") or "C")
    copied.setdefault("evidence_role", copied.get("evidence_role") or copied.get("role") or "supporting")
    copied.setdefault("allowed_use", copied.get("allowed_use") or "supporting")
    return copied


def _augment_chapter_evidence_packages(package: Dict[str, Any]) -> List[Dict[str, Any]]:
    existing = _as_list(package.get("chapter_evidence_packages"))
    rebuilt = build_chapter_evidence_packages_from_evidence_package(
        report_blueprint=_as_dict(package.get("report_blueprint")),
        evidence_package=_as_dict(package.get("evidence_package")),
        existing_chapter_evidence_packages=existing,
        source_registry=_as_list(package.get("source_registry")),
    )
    rebuilt_hydrated = _chapter_evidence_hydrated_count(rebuilt)
    existing_hydrated = _chapter_evidence_hydrated_count(existing)
    rebuilt_signal = _chapter_evidence_signal_count(rebuilt)
    existing_signal = _chapter_evidence_signal_count(existing)
    rebuilt_layered = _chapter_evidence_layered_count(rebuilt)
    existing_layered = _chapter_evidence_layered_count(existing)
    rebuilt_score = _chapter_evidence_selection_score(rebuilt)
    existing_score = _chapter_evidence_selection_score(existing)
    use_rebuilt = bool(
        rebuilt
        and rebuilt_hydrated > 0
        and (
            (rebuilt_layered > 0 and existing_layered <= 0)
            or rebuilt_score > existing_score
            or (rebuilt_hydrated >= existing_hydrated and rebuilt_signal >= existing_signal)
            or existing_hydrated <= 0
        )
    )
    if use_rebuilt:
        package["chapter_evidence_rebuild_diagnostics"] = {
            "status": "used_rebuilt_packages",
            "reason": "rebuilt_package_has_better_layered_signal_score",
            "rebuilt_signal_count": rebuilt_signal,
            "existing_signal_count": existing_signal,
            "rebuilt_hydrated_count": rebuilt_hydrated,
            "existing_hydrated_count": existing_hydrated,
            "rebuilt_layered_count": rebuilt_layered,
            "existing_layered_count": existing_layered,
            "rebuilt_selection_score": rebuilt_score,
            "existing_selection_score": existing_score,
        }
        return rebuilt
    package["chapter_evidence_rebuild_diagnostics"] = {
        "status": "kept_existing_packages",
        "reason": "existing_package_has_better_layered_signal_score",
        "rebuilt_signal_count": rebuilt_signal,
        "existing_signal_count": existing_signal,
        "rebuilt_hydrated_count": rebuilt_hydrated,
        "existing_hydrated_count": existing_hydrated,
        "rebuilt_layered_count": rebuilt_layered,
        "existing_layered_count": existing_layered,
        "rebuilt_selection_score": rebuilt_score,
        "existing_selection_score": existing_score,
    }
    return existing


def _chapter_evidence_hydrated_count(packages: List[Dict[str, Any]]) -> int:
    total = 0
    for package in packages or []:
        if not isinstance(package, dict):
            continue
        for key in (
            "core_evidence",
            "supporting_evidence",
            "metric_evidence",
            "case_evidence",
            "counter_evidence",
            "directional_evidence",
        ):
            total += len(_as_list(package.get(key)))
    return total


def _chapter_evidence_signal_count(packages: List[Dict[str, Any]]) -> int:
    total = 0
    for package in packages or []:
        if not isinstance(package, dict):
            continue
        counts = _as_dict(package.get("evidence_counts"))
        for key in (
            "core_evidence",
            "supporting_evidence",
            "metric_evidence",
            "case_evidence",
            "counter_evidence",
            "directional_evidence",
            "sample_evidence",
        ):
            total += len(_as_list(package.get(key)))
            try:
                total += int(float(package.get(f"{key}_count") or counts.get(key) or 0))
            except (TypeError, ValueError):
                pass
    return total


def _chapter_evidence_layered_count(packages: List[Dict[str, Any]]) -> int:
    total = 0
    for package in packages or []:
        if not isinstance(package, dict):
            continue
        counts = _as_dict(package.get("evidence_counts"))
        for key in ("metric_evidence", "case_evidence", "counter_evidence", "directional_evidence"):
            total += len(_as_list(package.get(key)))
            try:
                total += int(float(package.get(f"{key}_count") or counts.get(key) or 0))
            except (TypeError, ValueError):
                pass
    return total


def _chapter_evidence_selection_score(packages: List[Dict[str, Any]]) -> int:
    return (
        _chapter_evidence_layered_count(packages) * 3
        + _chapter_evidence_signal_count(packages) * 2
        + _chapter_evidence_hydrated_count(packages)
    )


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


def _micro_layouts_need_rebuild(micro_layouts: List[Dict[str, Any]]) -> bool:
    if not micro_layouts:
        return True
    generic_titles = {"事实依据", "商业化证据", "核心观察", "本章结论"}
    for layout in micro_layouts:
        for section in _as_list(_as_dict(layout).get("sections")):
            if not isinstance(section, dict):
                continue
            title = str(section.get("section_title") or section.get("title") or "").strip()
            if title in generic_titles:
                return True
            if not section.get("dynamic_section_title") and str(section.get("title_source") or "") != "dynamic":
                return True
    return False


def regenerate(package_path: Path, output_path: Path | None = None) -> Path:
    package = _load_rebuild_package(package_path)
    chapter_evidence_packages = _augment_chapter_evidence_packages(package)
    micro_layouts = _as_list(package.get("micro_layouts"))
    # Legacy table packages are often exactly where invalid metric parses live
    # (dates, URL ids, "cost: 40%", etc.). Regeneration should rebuild prose
    # first and keep table issues in the score report.
    table_packages: List[Dict[str, Any]] = []
    package["chapter_evidence_packages"] = chapter_evidence_packages
    package.setdefault("evidence_package", {})["chapter_evidence_packages"] = chapter_evidence_packages
    structured_analysis = _normalize_structured_analysis(package)
    package["structured_analysis"] = structured_analysis
    report_blueprint = _as_dict(package.get("report_blueprint"))
    source_registry = _as_list(package.get("source_registry")) or _as_list(_as_dict(package.get("writer_report")).get("source_registry"))
    if _micro_layouts_need_rebuild(micro_layouts):
        micro_layouts = run_micro_layout_agent(
            report_blueprint=report_blueprint,
            chapter_evidence_packages=chapter_evidence_packages,
            structured_analysis=structured_analysis,
        )
        package["micro_layouts"] = micro_layouts

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
        # Do not reuse legacy decision/risk packages: those were produced by
        # the polluted writer path and can re-inject invalid metrics such as
        # "cost: 40%" even after claim/chapter rebuild.
        decision_package={},
        risk_package={},
        appendix_package={
            **_as_dict(_as_dict(package.get("writer_report")).get("appendix_package")),
            "metric_normalization_table": _as_list(_as_dict(package.get("evidence_package")).get("metric_normalization_table")),
        },
        source_registry=source_registry,
        evidence_package=_as_dict(package.get("evidence_package")),
        chapter_evidence_packages=chapter_evidence_packages,
        claim_units=argument_units,
    )
    markdown = str(writer_output.get("report_markdown") or "").strip()
    if not markdown:
        raise RuntimeError("Regenerated writer output is empty.")
    markdown = _renumber_public_chapter_headings(markdown)
    markdown = re.sub(r"\n+##\s*报告质量评分与证据限制[\s\S]*?(?=\n+##\s|\Z)", "\n", markdown).strip()

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
    output = output_path or _default_output_path(package_path)
    render_artifacts = {
        "payload_mode": "full",
        "structured_analysis": structured_analysis,
        "chapter_evidence_packages": chapter_evidence_packages,
        "argument_units": argument_units,
        "chapter_packages": chapter_packages,
        "micro_layouts": micro_layouts,
        "table_packages": table_packages,
        "source_registry": writer_output.get("source_registry") or [],
    }
    writer_report = {
        **_as_dict(package.get("writer_report")),
        "report_markdown": markdown,
        "report_status": _as_dict(package.get("writer_report")).get("report_status") or "formal_scored",
        "source_registry": writer_output.get("source_registry") or [],
        "structured_analysis": structured_analysis,
        "chapter_evidence_packages": chapter_evidence_packages,
        "argument_units": argument_units,
        "chapter_packages": chapter_packages,
        "micro_layouts": micro_layouts,
        "table_packages": table_packages,
        "render_artifacts": render_artifacts,
    }
    score_path = output.with_name(output.name.replace("_report.md", "_score.md")) if output.name.endswith("_report.md") else output.with_name(output.stem + "_score.md")
    score_markdown = render_score_markdown(
        query=str(package.get("query") or ""),
        writer_report=writer_report,
        writer_package={
            **package,
            "writer_report": writer_report,
            "micro_layouts": micro_layouts,
            "table_packages": table_packages,
            "argument_units": argument_units,
            "chapter_packages": chapter_packages,
            "source_registry": writer_output.get("source_registry") or [],
            "analysis_rebuild_diagnostics": _as_dict(package.get("analysis_rebuild_diagnostics")),
        },
        final_audit_result=final_audit,
        reformatter_result={"enabled": False, "status": "skipped", "skipped_reason": "regenerate_report_only"},
    )
    write_formal_markdown(output, markdown)
    write_score_markdown(score_path, score_markdown)
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
