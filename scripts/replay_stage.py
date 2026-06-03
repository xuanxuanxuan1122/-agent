from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for candidate in (ROOT, SCRIPT_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from rag_pipeline.agents.chapter_argument_agent import run_chapter_argument_agent
from rag_pipeline.agents.claim_builder_agent import run_claim_builder_agent
from rag_pipeline.agents.citation_manifest import merge_source_registries
from rag_pipeline.agents.final_writer_agent import run_final_writer_agent
from rag_pipeline.agents.micro_layout_agent import run_micro_layout_agent
from rag_pipeline.cache.stage_snapshot_cache import list_stage_snapshots, load_stage_snapshot, stage_snapshot_cache_root
from rag_pipeline.config.search_config import build_llm_config_for_task
from rag_pipeline.flows.report.final_audit_agent import run_final_audit
from rag_pipeline.flows.report.full_report import render_score_markdown, write_formal_markdown, write_score_markdown
from rag_pipeline.observability.run_trace import write_run_trace_from_package
from regenerate_report_from_package import _augment_chapter_evidence_packages, _as_dict, _as_list, _normalize_structured_analysis


STAGE_ORDER = ["evidence_package", "chapter_evidence", "analysis", "claim", "chapter", "writer"]


def _snapshot_payload(run_id: str, stage_name: str) -> Any:
    loaded = load_stage_snapshot(run_id, stage_name)
    if loaded.get("status") != "loaded":
        return None
    return loaded.get("payload")


def _render_artifacts(writer_report: Dict[str, Any]) -> Dict[str, Any]:
    return _as_dict(writer_report.get("render_artifacts"))


def _merged_source_registry_for_replay(
    *,
    artifacts: Dict[str, Any],
    writer_report: Dict[str, Any],
    evidence_package: Dict[str, Any],
) -> List[Dict[str, Any]]:
    return merge_source_registries(
        _as_list(evidence_package.get("source_registry")),
        _as_list(evidence_package.get("sources")),
        _as_list(writer_report.get("source_registry")),
        _as_list(artifacts.get("source_registry")),
    )


def _load_replay_package(run_id: str) -> Dict[str, Any]:
    writer_report = _as_dict(_snapshot_payload(run_id, "writer_report"))
    artifacts = _render_artifacts(writer_report)
    evidence_package = (
        _as_dict(_snapshot_payload(run_id, "evidence_package"))
        or _as_dict(artifacts.get("evidence_package"))
        or _as_dict(writer_report.get("evidence_package"))
    )
    evidence_metadata = _as_dict(evidence_package.get("metadata"))
    report_blueprint = (
        _as_dict(artifacts.get("report_blueprint"))
        or _as_dict(writer_report.get("report_blueprint"))
        or _as_dict(evidence_package.get("report_blueprint"))
        or _as_dict(evidence_metadata.get("report_blueprint"))
        or _as_dict(evidence_package.get("report_plan"))
        or _as_dict(evidence_metadata.get("report_plan"))
    )
    source_registry = _merged_source_registry_for_replay(
        artifacts=artifacts,
        writer_report=writer_report,
        evidence_package=evidence_package,
    )
    package = {
        "query": writer_report.get("query") or evidence_package.get("query") or run_id,
        "evidence_package": evidence_package,
        "chapter_evidence_packages": (
            _as_list(_snapshot_payload(run_id, "chapter_evidence_packages"))
            or _as_list(artifacts.get("chapter_evidence_packages"))
            or _as_list(writer_report.get("chapter_evidence_packages"))
        ),
        "structured_analysis": (
            _as_dict(_snapshot_payload(run_id, "structured_analysis"))
            or _as_dict(artifacts.get("structured_analysis"))
            or _as_dict(writer_report.get("structured_analysis"))
        ),
        "argument_units": (
            _as_list(_snapshot_payload(run_id, "argument_units"))
            or _as_list(artifacts.get("argument_units"))
            or _as_list(writer_report.get("argument_units"))
        ),
        "chapter_packages": (
            _as_list(_snapshot_payload(run_id, "chapter_packages"))
            or _as_list(artifacts.get("chapter_packages"))
            or _as_list(writer_report.get("chapter_packages"))
        ),
        "table_packages": (
            _as_list(_snapshot_payload(run_id, "table_packages"))
            or _as_list(artifacts.get("table_packages"))
            or _as_list(writer_report.get("table_packages"))
        ),
        "micro_layouts": _as_list(artifacts.get("micro_layouts")) or _as_list(writer_report.get("micro_layouts")),
        "report_blueprint": report_blueprint,
        "source_registry": source_registry,
        "writer_report": writer_report,
        "stage_snapshot_run_id": run_id,
    }
    return package


def _stage_index(stage: str) -> int:
    normalized = "chapter_evidence" if stage == "chapter_evidence_packages" else stage
    if normalized not in STAGE_ORDER:
        raise ValueError(f"Unsupported --from stage: {stage}")
    return STAGE_ORDER.index(normalized)


def _rebuild_stage_for_replay(package: Dict[str, Any], from_stage: str) -> str:
    """Choose the earliest downstream stage that can be rebuilt with current code.

    Stage snapshots are useful as inputs, but replay should not reuse generated
    downstream artifacts such as argument units, chapter packages, or rendered
    markdown when enough upstream material exists to rebuild them. This keeps
    replay results aligned with the current implementation instead of stale
    cached writer output.
    """

    normalized = "chapter_evidence" if from_stage == "chapter_evidence_packages" else from_stage
    if normalized == "writer":
        if _as_dict(package.get("structured_analysis")) and _as_list(package.get("chapter_evidence_packages")):
            return "analysis"
        if _as_list(package.get("argument_units")) and _as_list(package.get("micro_layouts")):
            return "claim"
        return "chapter"
    if normalized == "chapter" and _as_list(package.get("argument_units")) and _as_list(package.get("micro_layouts")):
        return "claim"
    return normalized


def _drop_rebuilt_downstream_artifacts(package: Dict[str, Any], rebuild_stage: str) -> None:
    start = _stage_index(rebuild_stage)
    if start <= _stage_index("analysis"):
        package["micro_layouts"] = []
        package["argument_units"] = []
        package["chapter_packages"] = []
    elif start <= _stage_index("claim"):
        package["chapter_packages"] = []
    writer_report = _as_dict(package.get("writer_report"))
    if writer_report:
        package["writer_report"] = {**writer_report, "report_markdown": ""}


def replay_stage(
    *,
    run_id: str,
    from_stage: str,
    output_dir: Optional[Path] = None,
    allow_llm: bool = False,
    quality_mode: bool = False,
) -> Dict[str, Any]:
    allow_llm = bool(allow_llm or quality_mode)
    execution_mode = "quality_llm_replay" if quality_mode else ("llm_replay" if allow_llm else "deterministic_replay")
    previous_env = {
        "BRAIN_ENABLE_LLM_EVIDENCE_ANALYSIS": os.environ.get("BRAIN_ENABLE_LLM_EVIDENCE_ANALYSIS"),
        "REPORT_ENABLE_LLM_BODY_REWRITE": os.environ.get("REPORT_ENABLE_LLM_BODY_REWRITE"),
        "REPORT_BODY_REWRITE_MAX_SECTIONS": os.environ.get("REPORT_BODY_REWRITE_MAX_SECTIONS"),
        "REPORT_BODY_REWRITE_MAX_ELAPSED_SECONDS": os.environ.get("REPORT_BODY_REWRITE_MAX_ELAPSED_SECONDS"),
        "REPORT_BODY_REWRITE_CONCURRENCY": os.environ.get("REPORT_BODY_REWRITE_CONCURRENCY"),
        "REPORT_BODY_REWRITE_MAX_EXPANSION_RATIO": os.environ.get("REPORT_BODY_REWRITE_MAX_EXPANSION_RATIO"),
        "REPORT_BODY_REWRITE_TARGET_SECTION_CHARS": os.environ.get("REPORT_BODY_REWRITE_TARGET_SECTION_CHARS"),
        "REPORT_ENABLE_LLM_CHAPTER_NARRATIVE": os.environ.get("REPORT_ENABLE_LLM_CHAPTER_NARRATIVE"),
        "REPORT_TARGET_BODY_CHARS": os.environ.get("REPORT_TARGET_BODY_CHARS"),
        "REPORT_COMPOSER_TARGET_SECTION_CHARS": os.environ.get("REPORT_COMPOSER_TARGET_SECTION_CHARS"),
        "REPORT_RENDER_MIN_SECTION_CHARS": os.environ.get("REPORT_RENDER_MIN_SECTION_CHARS"),
        "REPORT_REPLAY_EXECUTION_MODE": os.environ.get("REPORT_REPLAY_EXECUTION_MODE"),
    }
    if quality_mode:
        os.environ["BRAIN_ENABLE_LLM_EVIDENCE_ANALYSIS"] = "true"
        os.environ["REPORT_ENABLE_LLM_BODY_REWRITE"] = "true"
        os.environ["REPORT_BODY_REWRITE_MAX_SECTIONS"] = "24"
        os.environ["REPORT_BODY_REWRITE_MAX_ELAPSED_SECONDS"] = "300"
        os.environ["REPORT_BODY_REWRITE_CONCURRENCY"] = "3"
        os.environ["REPORT_BODY_REWRITE_MAX_EXPANSION_RATIO"] = "5.0"
        os.environ["REPORT_BODY_REWRITE_TARGET_SECTION_CHARS"] = "850"
        os.environ["REPORT_ENABLE_LLM_CHAPTER_NARRATIVE"] = "true"
        os.environ["REPORT_TARGET_BODY_CHARS"] = "20000"
        os.environ["REPORT_COMPOSER_TARGET_SECTION_CHARS"] = "850"
        os.environ["REPORT_RENDER_MIN_SECTION_CHARS"] = "850"
        os.environ["REPORT_REPLAY_EXECUTION_MODE"] = execution_mode
    elif not allow_llm:
        os.environ["BRAIN_ENABLE_LLM_EVIDENCE_ANALYSIS"] = "false"
        os.environ["REPORT_REPLAY_EXECUTION_MODE"] = execution_mode
    else:
        os.environ["REPORT_REPLAY_EXECUTION_MODE"] = execution_mode
    try:
        return _replay_stage_impl(
            run_id=run_id,
            from_stage=from_stage,
            output_dir=output_dir,
            allow_llm=allow_llm,
            quality_mode=quality_mode,
            execution_mode=execution_mode,
        )
    finally:
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _replay_stage_impl(
    *,
    run_id: str,
    from_stage: str,
    output_dir: Optional[Path],
    allow_llm: bool,
    quality_mode: bool,
    execution_mode: str,
) -> Dict[str, Any]:
    package = _load_replay_package(run_id)
    if not _as_dict(package.get("evidence_package")):
        raise RuntimeError(f"No evidence_package snapshot found for run_id={run_id}")
    package["report_execution_mode"] = execution_mode
    package["quality_mode"] = bool(quality_mode)

    rebuild_stage = _rebuild_stage_for_replay(package, from_stage)
    _drop_rebuilt_downstream_artifacts(package, rebuild_stage)
    start = _stage_index(rebuild_stage)
    if start <= _stage_index("evidence_package"):
        package["chapter_evidence_packages"] = _augment_chapter_evidence_packages(package)
    if start <= _stage_index("analysis") and not _as_list(package.get("chapter_evidence_packages")):
        available = list_stage_snapshots(run_id)
        evidence_package = _as_dict(package.get("evidence_package"))
        missing = []
        if not _as_dict(package.get("report_blueprint")):
            missing.append("report_blueprint")
        if not (
            _as_list(evidence_package.get("analysis_ready_evidence"))
            or _as_list(evidence_package.get("clean_evidence_list"))
            or _as_list(evidence_package.get("normalized_evidence"))
            or _as_list(evidence_package.get("raw_data_points"))
        ):
            missing.append("writable_evidence")
        if not _as_list(package.get("source_registry")):
            missing.append("source_registry")
        raise RuntimeError(
            "Replay requires chapter evidence packages; rebuild failed. "
            f"missing={missing or ['chapter_evidence_packages']} "
            f"available_snapshots={[item.get('stage_name') for item in available]}"
        )

    should_run_analysis = start <= _stage_index("chapter_evidence") or (quality_mode and start <= _stage_index("analysis"))
    if should_run_analysis:
        llm_config = build_llm_config_for_task("decision") if allow_llm else None
        package["structured_analysis"] = _normalize_structured_analysis(
            package,
            llm_config=dict(llm_config or {}),
            force_analysis=bool(quality_mode),
        )
    if start <= _stage_index("analysis"):
        package["micro_layouts"] = run_micro_layout_agent(
            report_blueprint=_as_dict(package.get("report_blueprint")),
            chapter_evidence_packages=_as_list(package.get("chapter_evidence_packages")),
            structured_analysis=_as_dict(package.get("structured_analysis")),
        )
    if start <= _stage_index("analysis"):
        package["argument_units"] = run_claim_builder_agent(
            chapter_evidence_packages=_as_list(package.get("chapter_evidence_packages")),
            micro_layouts=_as_list(package.get("micro_layouts")),
            structured_analysis=_as_dict(package.get("structured_analysis")),
        )
    if start <= _stage_index("claim"):
        package["chapter_packages"] = run_chapter_argument_agent(
            report_blueprint=_as_dict(package.get("report_blueprint")),
            micro_layouts=_as_list(package.get("micro_layouts")),
            argument_units=_as_list(package.get("argument_units")),
            table_packages=_as_list(package.get("table_packages")),
            chapter_evidence_packages=_as_list(package.get("chapter_evidence_packages")),
        )

    writer_report = _as_dict(package.get("writer_report"))
    if not _as_list(package.get("chapter_packages")):
        raise RuntimeError("Replay refuses to reuse stale writer markdown without chapter_packages for rerendering.")
    writer_output = run_final_writer_agent(
        query=str(package.get("query") or ""),
        report_blueprint=_as_dict(package.get("report_blueprint")),
        chapter_packages=_as_list(package.get("chapter_packages")),
        table_packages=_as_list(package.get("table_packages")),
        decision_package={},
        risk_package={},
        appendix_package=_as_dict(writer_report.get("appendix_package")),
        source_registry=_as_list(package.get("source_registry")),
        evidence_package=_as_dict(package.get("evidence_package")),
        chapter_evidence_packages=_as_list(package.get("chapter_evidence_packages")),
        claim_units=_as_list(package.get("argument_units")),
        analysis_claim_units=_as_list(_as_dict(package.get("structured_analysis")).get("claim_units")),
    )
    writer_report = {
        **writer_report,
        "report_markdown": str(writer_output.get("report_markdown") or "").strip(),
        "report_status": writer_report.get("report_status") or "formal_scored",
        "report_execution_mode": execution_mode,
        "quality_mode": bool(quality_mode),
        "source_registry": writer_output.get("source_registry") or [],
        "citation_manifest": writer_output.get("citation_manifest") or {},
        "final_citation_audit": writer_output.get("final_citation_audit") or {},
        "source_claim_support": writer_output.get("source_claim_support") or {},
        "analysis_transfer": writer_output.get("analysis_transfer") or {},
        "ref_lineage_diagnostics": writer_output.get("ref_lineage_diagnostics") or {},
        "naturalness_cleanup": writer_output.get("naturalness_cleanup") or {},
        "public_narrative_leak_audit": writer_output.get("public_narrative_leak_audit") or {},
        "render_artifacts": {
            "payload_mode": "full",
            "evidence_package": package.get("evidence_package"),
            "chapter_evidence_packages": package.get("chapter_evidence_packages"),
            "structured_analysis": package.get("structured_analysis"),
            "argument_units": package.get("argument_units"),
            "chapter_packages": package.get("chapter_packages"),
            "table_packages": package.get("table_packages"),
            "micro_layouts": package.get("micro_layouts"),
            "source_registry": writer_output.get("source_registry") or [],
            "citation_manifest": writer_output.get("citation_manifest") or {},
            "final_citation_audit": writer_output.get("final_citation_audit") or {},
            "source_claim_support": writer_output.get("source_claim_support") or {},
            "analysis_transfer": writer_output.get("analysis_transfer") or {},
            "ref_lineage_diagnostics": writer_output.get("ref_lineage_diagnostics") or {},
            "public_narrative_leak_audit": writer_output.get("public_narrative_leak_audit") or {},
            "report_blueprint": package.get("report_blueprint"),
            "metadata": {
                "report_execution_mode": execution_mode,
                "quality_mode": bool(quality_mode),
            },
        },
    }
    markdown = str(writer_report.get("report_markdown") or "").strip()
    if not markdown:
        raise RuntimeError("Replay writer output is empty.")

    audit_package = {
        **package,
        "writer_report": writer_report,
        "source_registry": _as_list(writer_report.get("source_registry")),
        "stage_snapshot_replay": {
            "run_id": run_id,
            "from_stage": from_stage,
            "rebuild_stage": rebuild_stage,
            "quality_mode": bool(quality_mode),
            "execution_mode": execution_mode,
        },
        "report_execution_mode": execution_mode,
        "quality_mode": bool(quality_mode),
    }
    final_audit = run_final_audit(
        report_markdown=markdown,
        validation=_as_dict(writer_report.get("validation")),
        clean_evidence=None,
        writer_package_payload=audit_package,
        query=str(package.get("query") or ""),
    )
    score_writer_package = {
        **package,
        "writer_report": writer_report,
        "source_registry": _as_list(writer_report.get("source_registry")),
        "stage_snapshot_replay": {
            "run_id": run_id,
            "from_stage": from_stage,
            "rebuild_stage": rebuild_stage,
            "quality_mode": bool(quality_mode),
            "execution_mode": execution_mode,
        },
        "report_execution_mode": execution_mode,
        "quality_mode": bool(quality_mode),
        "final_audit_result": final_audit,
        "report_delivery_status": {
            "formal_report_written": True,
            "score_report_written": True,
            "clean_report_written": False,
        },
    }
    score = render_score_markdown(
        query=str(package.get("query") or ""),
        writer_report=writer_report,
        writer_package=score_writer_package,
        final_audit_result=final_audit,
        reformatter_result={"enabled": False, "status": "skipped", "skipped_reason": "stage_replay"},
    )

    base_dir = output_dir or (ROOT / "output" / "full_reports")
    base_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{run_id}_replay"
    report_path = base_dir / f"{stem}_report.md"
    score_path = base_dir / f"{stem}_score.md"
    write_formal_markdown(report_path, markdown)
    write_score_markdown(score_path, score)
    trace_result = write_run_trace_from_package(
        run_id=run_id,
        output_dir=base_dir,
        base_name=stem,
        writer_package=score_writer_package,
        writer_report=writer_report,
        final_status="replay_completed",
    )
    return {
        "run_id": run_id,
        "from_stage": from_stage,
        "stage_snapshot_replay": {
            "run_id": run_id,
            "from_stage": from_stage,
            "rebuild_stage": rebuild_stage,
            "quality_mode": bool(quality_mode),
            "execution_mode": execution_mode,
        },
        "report_execution_mode": execution_mode,
        "quality_mode": bool(quality_mode),
        "report_path": str(report_path),
        "score_path": str(score_path),
        "trace_path": str(trace_result.get("trace_path") or ""),
        "trace_summary_path": str(trace_result.get("summary_path") or ""),
        "snapshot_root": str(stage_snapshot_cache_root()),
        "available_snapshots": list_stage_snapshots(run_id),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay the report pipeline from a saved stage snapshot without rerunning retrieval.")
    parser.add_argument("--run-id", required=True, help="Stage snapshot run id, usually the report timestamp plus safe query.")
    parser.add_argument("--from", dest="from_stage", default="evidence_package", choices=STAGE_ORDER + ["chapter_evidence_packages"])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--allow-network", action="store_true", help="Accepted for compatibility; replay does not perform retrieval.")
    parser.add_argument("--allow-llm", action="store_true", help="Allow LLM-backed analysis if the local config enables it.")
    parser.add_argument(
        "--quality-mode",
        action="store_true",
        help="Run high-quality offline replay: allow LLM evidence analysis and enable section body rewrite without retrieval.",
    )
    args = parser.parse_args()
    result = replay_stage(
        run_id=args.run_id,
        from_stage=args.from_stage,
        output_dir=args.output_dir.resolve() if args.output_dir else None,
        allow_llm=bool(args.allow_llm),
        quality_mode=bool(args.quality_mode),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
