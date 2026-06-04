from __future__ import annotations

import sys
from pathlib import Path

from rag_pipeline.cache.stage_snapshot_cache import write_stage_snapshot
from rag_pipeline.flows.report.full_report import merge_source_registry_candidates, render_score_markdown

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import regenerate_report_from_package  # noqa: E402
import replay_stage as replay_stage_module  # noqa: E402
from replay_stage import _load_replay_package, replay_stage  # noqa: E402


def _quality_test_source_registry():
    return [
        {
            "ref": "S1",
            "title": "Verified AI Agent Workflow Source",
            "url": "https://example.org/workflow",
            "source_level": "A",
            "publisher": "Example Research",
            "traceability_status": "traceable",
        }
    ]


def test_replay_package_keeps_full_evidence_source_registry_when_writer_snapshot_is_compacted(tmp_path, monkeypatch):
    monkeypatch.setenv("STAGE_SNAPSHOT_CACHE_ENABLED", "true")
    monkeypatch.setenv("STAGE_SNAPSHOT_CACHE_PATH", str(tmp_path / "snapshots"))
    run_id = "20260101_000010_source_registry_merge"
    full_sources = [
        {
            "ref": "[1]",
            "title": "Source one",
            "url": "https://example.org/one",
            "evidence_refs": ["EV-1"],
            "source_level": "A",
        },
        {
            "ref": "[2]",
            "title": "Source two",
            "url": "https://example.org/two",
            "evidence_refs": ["EV-2"],
            "source_level": "B",
        },
    ]
    compact_writer_source = [
        {
            "ref": "[1]",
            "title": "Source one",
            "url": "https://example.org/one",
            "evidence_refs": ["EV-1"],
            "source_level": "A",
        }
    ]
    write_stage_snapshot(
        "evidence_package",
        run_id,
        {
            "query": "source registry merge",
            "source_registry": full_sources,
            "analysis_ready_evidence": [
                {"evidence_id": "EV-1", "source_ref": "[1]", "fact": "Fact one."},
                {"evidence_id": "EV-2", "source_ref": "[2]", "fact": "Fact two."},
            ],
        },
    )
    write_stage_snapshot(
        "writer_report",
        run_id,
        {
            "query": "source registry merge",
            "source_registry": compact_writer_source,
            "render_artifacts": {"source_registry": compact_writer_source},
        },
    )

    package = _load_replay_package(run_id)

    refs = {source.get("ref") for source in package["source_registry"]}
    assert {"[1]", "[2]"} <= refs
    assert any("EV-2" in (source.get("evidence_refs") or []) for source in package["source_registry"])


def test_full_report_source_registry_merge_does_not_let_compact_registry_hide_evidence_sources():
    full_sources = [
        {"ref": "[1]", "url": "https://example.org/one", "evidence_refs": ["EV-1"]},
        {"ref": "[2]", "url": "https://example.org/two", "evidence_refs": ["EV-2"]},
    ]
    compact_sources = [{"ref": "[1]", "url": "https://example.org/one", "evidence_refs": ["EV-1"]}]

    merged = merge_source_registry_candidates(full_sources, compact_sources)

    refs = {source.get("ref") for source in merged}
    assert {"[1]", "[2]"} <= refs
    assert any("EV-2" in (source.get("evidence_refs") or []) for source in merged)


def test_quality_mode_replay_reruns_analysis_with_llm_and_enables_rewrite(tmp_path, monkeypatch):
    monkeypatch.setenv("STAGE_SNAPSHOT_CACHE_ENABLED", "true")
    monkeypatch.setenv("STAGE_SNAPSHOT_CACHE_PATH", str(tmp_path / "snapshots"))
    monkeypatch.setenv("REPORT_ENABLE_FINAL_AUDIT", "false")
    monkeypatch.setenv("REPORT_ENABLE_LLM_BODY_REWRITE", "false")
    monkeypatch.setenv("BRAIN_ENABLE_LLM_EVIDENCE_ANALYSIS", "false")
    monkeypatch.setenv("RAG_MODEL_QA_PROFILE", "qwen")
    monkeypatch.setenv("RAG_MODEL_QUERY_REWRITE_PROFILE", "deepseek-v4-pro")
    run_id = "20260101_000001_quality_topic"
    source_registry = _quality_test_source_registry()
    report_blueprint = {
        "chapters": [
            {
                "chapter_id": "ch_01",
                "chapter_title": "Enterprise workflow adoption",
                "chapter_question": "Are AI Agent workflows entering enterprise operations?",
            }
        ]
    }
    evidence_item = {
        "evidence_id": "EV-1",
        "ref": "EV-1",
        "source_ref": "S1",
        "source_level": "A",
        "url": "https://example.org/workflow",
        "fact": "Enterprises are deploying AI Agent workflows in support and software engineering operations.",
        "distilled_fact": "Enterprises are deploying AI Agent workflows in support and software engineering operations.",
        "chapter_id": "ch_01",
        "source_verification_status": "readpage_verified",
        "public_fact_card": {
            "subject": "enterprise AI Agent workflows",
            "action_or_signal": "deployment",
            "variable": "workflow adoption",
            "source_ref": "S1",
            "block_affinity": "case_comparison",
        },
        "source": {
            "title": "Verified AI Agent Workflow Source",
            "url": "https://example.org/workflow",
            "source_verification_status": "readpage_verified",
        },
    }
    evidence_package = {
        "query": "AI Agent high quality replay",
        "analysis_ready_evidence": [evidence_item],
        "source_registry": source_registry,
        "report_blueprint": report_blueprint,
    }
    chapter_evidence = [
        {
            "chapter_id": "ch_01",
            "chapter_title": "Enterprise workflow adoption",
            "chapter_question": "Are AI Agent workflows entering enterprise operations?",
            "core_evidence": [evidence_item],
            "case_evidence": [evidence_item],
            "fact_card_count": 1,
        }
    ]
    stale_analysis = {
        "claim_units": [{"claim": "old deterministic claim", "evidence_refs": ["EV-OLD"]}],
        "chapter_insights": [{"chapter_id": "ch_01", "key_claims": ["old deterministic claim"]}],
        "evidence_analyses": [{"evidence_id": "EV-OLD", "chapter_id": "ch_01"}],
        "analysis_stage_diagnostics": {
            "uses_llm_analysis": False,
            "llm_analysis_status": "not_run",
            "final_analysis_source": "deterministic_rebuild",
        },
    }
    write_stage_snapshot("evidence_package", run_id, evidence_package)
    write_stage_snapshot("chapter_evidence_packages", run_id, chapter_evidence)
    write_stage_snapshot("structured_analysis", run_id, stale_analysis)
    write_stage_snapshot(
        "writer_report",
        run_id,
        {
            "query": "AI Agent high quality replay",
            "report_blueprint": report_blueprint,
            "source_registry": source_registry,
            "render_artifacts": {
                "report_blueprint": report_blueprint,
                "source_registry": source_registry,
                "evidence_package": evidence_package,
                "chapter_evidence_packages": chapter_evidence,
            },
        },
    )
    calls = []

    def fake_analysis(evidence_package_arg, *, query="", llm_config=None):
        import os

        calls.append(
            {
                "llm_config": dict(llm_config or {}),
                "llm_enabled": os.environ.get("BRAIN_ENABLE_LLM_EVIDENCE_ANALYSIS"),
                "rewrite_enabled": os.environ.get("REPORT_ENABLE_LLM_BODY_REWRITE"),
                "rewrite_max_sections": os.environ.get("REPORT_BODY_REWRITE_MAX_SECTIONS"),
                "rewrite_max_elapsed": os.environ.get("REPORT_BODY_REWRITE_MAX_ELAPSED_SECONDS"),
                "rewrite_max_expansion_ratio": os.environ.get("REPORT_BODY_REWRITE_MAX_EXPANSION_RATIO"),
                "rewrite_target_chars": os.environ.get("REPORT_BODY_REWRITE_TARGET_SECTION_CHARS"),
                "chapter_narrative_enabled": os.environ.get("REPORT_ENABLE_LLM_CHAPTER_NARRATIVE"),
                "chapter_narrative_max_chapters": os.environ.get("REPORT_CHAPTER_NARRATIVE_MAX_CHAPTERS"),
                "target_body_chars": os.environ.get("REPORT_TARGET_BODY_CHARS"),
                "composer_target_chars": os.environ.get("REPORT_COMPOSER_TARGET_SECTION_CHARS"),
                "render_min_section_chars": os.environ.get("REPORT_RENDER_MIN_SECTION_CHARS"),
                "qa_profile": os.environ.get("RAG_MODEL_QA_PROFILE"),
                "query_rewrite_profile": os.environ.get("RAG_MODEL_QUERY_REWRITE_PROFILE"),
            }
        )
        structured = {
            "claim_units": [
                {
                    "chapter_id": "ch_01",
                    "claim": "Enterprise AI Agent adoption is moving into workflow deployment.",
                    "claim_status": "decision_ready",
                    "claim_strength": "moderate",
                    "evidence_refs": ["EV-1"],
                    "supporting_evidence_refs": ["EV-1"],
                    "reasoning": "Support and software engineering workflows provide measurable operating contexts.",
                }
            ],
            "chapter_insights": [
                {
                    "chapter_id": "ch_01",
                    "chapter_title": "Enterprise workflow adoption",
                    "key_claims": ["Enterprise AI Agent adoption is moving into workflow deployment."],
                    "evidence_refs": ["EV-1"],
                }
            ],
            "evidence_analyses": [{"evidence_id": "EV-1", "chapter_id": "ch_01"}],
            "analysis_contract_status": {
                "structured_analysis_valid": True,
                "claim_unit_count": 1,
                "chapter_insight_count": 1,
                "evidence_analysis_count": 1,
                "analysis_rebuilt_from_evidence": False,
            },
            "analysis_stage_diagnostics": {
                "uses_llm_analysis": True,
                "llm_analysis_attempted": True,
                "llm_analysis_status": "success",
                "final_analysis_source": "llm_evidence_analysis",
                "quality_path_degraded": False,
                "llm_usable_claim_count": 1,
                "llm_dropped_claim_count": 0,
                "llm_usable_chapter_count": 1,
            },
        }
        return {"structured_analysis": structured, "metadata": {"llm_analysis_status": "success"}, "errors": []}

    monkeypatch.setattr(regenerate_report_from_package, "run_analysis_agent", fake_analysis)

    result = replay_stage(run_id=run_id, from_stage="chapter_evidence", output_dir=tmp_path / "out", quality_mode=True)

    assert calls
    assert calls[0]["llm_enabled"] == "true"
    assert calls[0]["rewrite_enabled"] == "true"
    assert calls[0]["rewrite_max_sections"] == "24"
    assert calls[0]["rewrite_max_elapsed"] == "900"
    assert calls[0]["rewrite_max_expansion_ratio"] == "5.0"
    assert calls[0]["rewrite_target_chars"] == "650"
    assert calls[0]["chapter_narrative_enabled"] == "true"
    assert calls[0]["chapter_narrative_max_chapters"] == "12"
    assert calls[0]["target_body_chars"] == "0"
    assert calls[0]["composer_target_chars"] == "550"
    assert calls[0]["render_min_section_chars"] == "0"
    assert calls[0]["qa_profile"] == "deepseek-v4-pro"
    assert calls[0]["query_rewrite_profile"] == "qwen"
    assert calls[0]["llm_config"]
    score_text = Path(result["score_path"]).read_text(encoding="utf-8")
    assert "report_execution_mode: quality_llm_replay" in score_text
    assert "quality_mode: True" in score_text
    assert "uses_llm_analysis: True" in score_text
    assert "final_analysis_source: llm_evidence_analysis" in score_text


def test_score_reports_quality_path_degradation():
    writer_report = {
        "report_status": "formal_scored",
        "quality_score": 70,
        "report_execution_mode": "live_quality_full",
        "quality_mode": True,
        "render_artifacts": {
            "payload_mode": "full",
            "chapter_packages": [{"chapter_id": "ch_01"}],
            "argument_units": [{"claim": "claim"}],
            "structured_analysis": {
                "analysis_contract_status": {"structured_analysis_valid": True},
                "analysis_stage_diagnostics": {
                    "uses_llm_analysis": False,
                    "llm_analysis_attempted": True,
                    "llm_analysis_status": "invalid_output",
                    "final_analysis_source": "deterministic_rebuild",
                    "quality_path_degraded": True,
                    "quality_path_degradation_reason": "invalid_output",
                },
            },
        },
    }
    score = render_score_markdown(
        query="quality path",
        writer_report=writer_report,
        writer_package={"writer_report": writer_report},
        final_audit_result={"blocked": False},
        reformatter_result={"enabled": False, "status": "skipped"},
    )
    assert "report_execution_mode: live_quality_full" in score
    assert "quality_mode: True" in score
    assert "llm_analysis_attempted: True" in score
    assert "quality_path_degraded: True" in score
    assert "quality_path_degradation_reason: invalid_output" in score


def test_replay_from_chapter_snapshot_writes_report_and_score(tmp_path, monkeypatch):
    monkeypatch.setenv("STAGE_SNAPSHOT_CACHE_ENABLED", "true")
    monkeypatch.setenv("STAGE_SNAPSHOT_CACHE_PATH", str(tmp_path / "snapshots"))
    monkeypatch.setenv("REPORT_ENABLE_FINAL_AUDIT", "false")
    run_id = "20260101_000000_test_topic"
    source_registry = [
        {
            "ref": "S1",
            "title": "Official AI Agent Platform Disclosure",
            "url": "https://www.salesforce.com/news/ai-agent-platform",
            "source_level": "B",
            "publisher": "Example Research",
            "traceability_status": "traceable",
        }
    ]
    report_blueprint = {
        "chapters": [
            {
                "chapter_id": "ch_01",
                "chapter_title": "Enterprise AI Agent Demand",
                "chapter_question": "Has enterprise AI Agent demand become observable?",
            }
        ],
        "report_shell": {"front_blocks": [], "back_blocks": ["appendix"]},
    }
    evidence_package = {
        "query": "AI Agent industry report",
        "analysis_ready_evidence": [
            {
                "ref": "E1",
                "source_ref": "S1",
                "source_level": "B",
                "url": "https://www.salesforce.com/news/ai-agent-platform",
                "fact": "Enterprises are testing AI Agent workflows in support, software engineering, and analytics.",
                "chapter_id": "ch_01",
            }
        ],
        "source_registry": source_registry,
    }
    chapter_packages = [
        {
            "chapter_id": "ch_01",
            "chapter_title": "Enterprise AI Agent Demand",
            "chapter_question": "Has enterprise AI Agent demand become observable?",
            "sections": [
                {
                    "section_id": "ch_01_s1",
                    "section_title": "Core Observation",
                    "claim": "Enterprise AI Agent demand is moving from concept discussion into workflow pilots.",
                    "reasoning": "Support, software engineering, and analytics are lower-risk workflows where ROI can be tested earlier.",
                    "supporting_facts": [
                        {
                            "evidence_id": "S1",
                            "source_ref": "S1",
                            "distilled_fact": "Enterprises are testing AI Agent workflows in support, software engineering, and analytics.",
                            "source_title": "Official AI Agent Platform Disclosure",
                        }
                    ],
                    "used_fact_refs": ["S1"],
                    "evidence_refs": ["S1"],
                    "render_blocks": [
                        {
                            "type": "paragraph",
                            "text": "Enterprise AI Agent demand is moving from concept discussion into workflow pilots.",
                        }
                    ],
                    "evidence_backed": True,
                    "claim_strength": "directional",
                }
            ],
        }
    ]
    write_stage_snapshot("evidence_package", run_id, evidence_package)
    write_stage_snapshot("chapter_packages", run_id, chapter_packages)
    write_stage_snapshot(
        "writer_report",
        run_id,
        {
            "query": "AI Agent industry report",
            "report_blueprint": report_blueprint,
            "source_registry": source_registry,
            "render_artifacts": {
                "report_blueprint": report_blueprint,
                "source_registry": source_registry,
                "evidence_package": evidence_package,
            },
        },
    )

    result = replay_stage(run_id=run_id, from_stage="chapter", output_dir=tmp_path / "out")

    report = Path(result["report_path"])
    score = Path(result["score_path"])
    assert report.exists()
    assert score.exists()
    report_text = report.read_text(encoding="utf-8")
    assert "Enterprise AI Agent demand" in report_text
    assert "质量总分" not in report_text
def test_replay_from_analysis_discards_stale_downstream_writer_markdown(tmp_path, monkeypatch):
    monkeypatch.setenv("STAGE_SNAPSHOT_CACHE_ENABLED", "true")
    monkeypatch.setenv("STAGE_SNAPSHOT_CACHE_PATH", str(tmp_path / "snapshots"))
    monkeypatch.setenv("REPORT_ENABLE_FINAL_AUDIT", "false")
    run_id = "20260101_000011_rebuild_downstream"
    report_blueprint = {"chapters": [{"chapter_id": "ch_01", "chapter_title": "Current analysis"}]}
    evidence_package = {
        "query": "rebuild downstream",
        "report_blueprint": report_blueprint,
        "analysis_ready_evidence": [{"evidence_id": "EV-1", "chapter_id": "ch_01", "fact": "Current fact."}],
        "source_registry": [{"ref": "S1", "url": "https://example.org/current", "title": "Current source"}],
    }
    structured_analysis = {
        "claim_units": [{"claim_id": "CL-1", "chapter_id": "ch_01", "claim": "Current claim.", "evidence_refs": ["EV-1"]}],
        "analysis_stage_diagnostics": {"final_analysis_source": "llm_partial_merged"},
    }
    write_stage_snapshot("evidence_package", run_id, evidence_package)
    write_stage_snapshot(
        "chapter_evidence_packages",
        run_id,
        [{"chapter_id": "ch_01", "chapter_title": "Current analysis", "core_evidence": evidence_package["analysis_ready_evidence"]}],
    )
    write_stage_snapshot("structured_analysis", run_id, structured_analysis)
    write_stage_snapshot(
        "writer_report",
        run_id,
        {
            "query": "rebuild downstream",
            "report_markdown": "OLD SNAPSHOT MARKDOWN SHOULD NOT BE USED",
            "report_blueprint": report_blueprint,
            "render_artifacts": {
                "report_blueprint": report_blueprint,
                "evidence_package": evidence_package,
                "structured_analysis": structured_analysis,
                "argument_units": [{"claim": "old unit"}],
                "chapter_packages": [{"chapter_id": "old"}],
            },
        },
    )

    monkeypatch.setattr(replay_stage_module, "run_micro_layout_agent", lambda **kwargs: [{"chapter_id": "ch_01", "sections": []}])
    monkeypatch.setattr(
        replay_stage_module,
        "run_claim_builder_agent",
        lambda **kwargs: [{"claim_id": "CL-1", "chapter_id": "ch_01", "claim": "Current claim.", "public_render": True}],
    )
    monkeypatch.setattr(
        replay_stage_module,
        "run_chapter_argument_agent",
        lambda **kwargs: [{"chapter_id": "ch_01", "sections": [{"claim_id": "CL-1", "claim": "Current claim."}]}],
    )
    monkeypatch.setattr(
        replay_stage_module,
        "run_final_writer_agent",
        lambda **kwargs: {
            "report_markdown": "NEW CURRENT WRITER MARKDOWN",
            "source_registry": [],
            "citation_manifest": {},
            "final_citation_audit": {"status": "ok"},
            "analysis_transfer": {"rendered_analysis_claim_count": 1},
        },
    )

    result = replay_stage(run_id=run_id, from_stage="analysis", output_dir=tmp_path / "out")

    report_text = Path(result["report_path"]).read_text(encoding="utf-8")
    assert "NEW CURRENT WRITER MARKDOWN" in report_text
    assert "OLD SNAPSHOT" not in report_text


def test_replay_from_analysis_discards_stale_chapter_packages(tmp_path, monkeypatch):
    monkeypatch.setenv("STAGE_SNAPSHOT_CACHE_ENABLED", "true")
    monkeypatch.setenv("STAGE_SNAPSHOT_CACHE_PATH", str(tmp_path / "snapshots"))
    monkeypatch.setenv("REPORT_ENABLE_FINAL_AUDIT", "false")
    run_id = "20260101_000013_rebuild_stale_chapters"
    report_blueprint = {"chapters": [{"chapter_id": "ch_01", "chapter_title": "Current analysis"}]}
    evidence_package = {
        "query": "rebuild stale chapters",
        "report_blueprint": report_blueprint,
        "analysis_ready_evidence": [{"evidence_id": "EV-1", "chapter_id": "ch_01", "fact": "Current fact."}],
        "source_registry": [{"ref": "S1", "url": "https://example.org/current", "title": "Current source"}],
    }
    structured_analysis = {
        "claim_units": [{"claim_id": "CL-1", "chapter_id": "ch_01", "claim": "Current claim.", "evidence_refs": ["EV-1"]}],
        "analysis_stage_diagnostics": {"final_analysis_source": "llm_partial_merged"},
    }
    write_stage_snapshot("evidence_package", run_id, evidence_package)
    write_stage_snapshot(
        "chapter_evidence_packages",
        run_id,
        [{"chapter_id": "ch_01", "chapter_title": "Current analysis", "core_evidence": evidence_package["analysis_ready_evidence"]}],
    )
    write_stage_snapshot("structured_analysis", run_id, structured_analysis)
    write_stage_snapshot("argument_units", run_id, [{"claim_id": "OLD", "chapter_id": "old", "claim": "Old unit."}])
    write_stage_snapshot("chapter_packages", run_id, [{"chapter_id": "old", "sections": [{"claim": "Old stale section."}]}])
    write_stage_snapshot(
        "writer_report",
        run_id,
        {
            "query": "rebuild stale chapters",
            "report_markdown": "OLD SNAPSHOT MARKDOWN SHOULD NOT BE USED",
            "report_blueprint": report_blueprint,
            "render_artifacts": {
                "report_blueprint": report_blueprint,
                "evidence_package": evidence_package,
                "structured_analysis": structured_analysis,
                "argument_units": [{"claim_id": "OLD", "chapter_id": "old", "claim": "Old unit."}],
                "chapter_packages": [{"chapter_id": "old", "sections": [{"claim": "Old stale section."}]}],
            },
        },
    )

    monkeypatch.setattr(replay_stage_module, "run_micro_layout_agent", lambda **kwargs: [{"chapter_id": "ch_01", "sections": []}])
    monkeypatch.setattr(
        replay_stage_module,
        "run_claim_builder_agent",
        lambda **kwargs: [{"claim_id": "CL-1", "chapter_id": "ch_01", "claim": "Current claim.", "public_render": True}],
    )
    chapter_calls = []

    def fake_chapter_argument(**kwargs):
        chapter_calls.append(kwargs)
        return [{"chapter_id": "ch_01", "sections": [{"claim_id": "CL-1", "claim": "Current claim."}]}]

    monkeypatch.setattr(replay_stage_module, "run_chapter_argument_agent", fake_chapter_argument)
    monkeypatch.setattr(
        replay_stage_module,
        "run_final_writer_agent",
        lambda **kwargs: {
            "report_markdown": "\n".join(section.get("claim", "") for chapter in kwargs.get("chapter_packages", []) for section in chapter.get("sections", [])),
            "source_registry": [],
            "citation_manifest": {},
            "final_citation_audit": {"status": "ok"},
            "analysis_transfer": {"rendered_analysis_claim_count": 1},
        },
    )

    result = replay_stage(run_id=run_id, from_stage="analysis", output_dir=tmp_path / "out")

    assert chapter_calls
    assert chapter_calls[0]["argument_units"][0]["claim_id"] == "CL-1"
    report_text = Path(result["report_path"]).read_text(encoding="utf-8")
    assert "Current claim" in report_text
    assert "Old stale section" not in report_text


def test_replay_from_writer_rerenders_instead_of_returning_stale_markdown(tmp_path, monkeypatch):
    monkeypatch.setenv("STAGE_SNAPSHOT_CACHE_ENABLED", "true")
    monkeypatch.setenv("STAGE_SNAPSHOT_CACHE_PATH", str(tmp_path / "snapshots"))
    monkeypatch.setenv("REPORT_ENABLE_FINAL_AUDIT", "false")
    run_id = "20260101_000012_writer_rerender"
    report_blueprint = {"chapters": [{"chapter_id": "ch_01", "chapter_title": "Current writer"}]}
    evidence_package = {
        "query": "writer rerender",
        "report_blueprint": report_blueprint,
        "analysis_ready_evidence": [{"evidence_id": "EV-1", "chapter_id": "ch_01", "fact": "Current fact."}],
        "source_registry": [{"ref": "S1", "url": "https://example.org/current", "title": "Current source"}],
    }
    chapter_packages = [{"chapter_id": "ch_01", "sections": [{"claim_id": "CL-1", "claim": "Current writer claim."}]}]
    argument_units = [{"claim_id": "CL-1", "chapter_id": "ch_01", "claim": "Current writer claim.", "public_render": True}]
    write_stage_snapshot("evidence_package", run_id, evidence_package)
    write_stage_snapshot(
        "writer_report",
        run_id,
        {
            "query": "writer rerender",
            "report_markdown": "OLD WRITER SNAPSHOT SHOULD NOT BE RETURNED",
            "report_blueprint": report_blueprint,
            "render_artifacts": {
                "report_blueprint": report_blueprint,
                "evidence_package": evidence_package,
                "chapter_evidence_packages": [{"chapter_id": "ch_01"}],
                "structured_analysis": {"claim_units": argument_units},
                "argument_units": argument_units,
                "chapter_packages": chapter_packages,
                "source_registry": evidence_package["source_registry"],
            },
        },
    )
    monkeypatch.setattr(replay_stage_module, "run_micro_layout_agent", lambda **kwargs: [{"chapter_id": "ch_01", "sections": []}])
    monkeypatch.setattr(
        replay_stage_module,
        "run_claim_builder_agent",
        lambda **kwargs: [{"claim_id": "CL-1", "chapter_id": "ch_01", "claim": "Current writer claim.", "public_render": True}],
    )
    monkeypatch.setattr(
        replay_stage_module,
        "run_chapter_argument_agent",
        lambda **kwargs: [{"chapter_id": "ch_01", "sections": [{"claim_id": "CL-1", "claim": "Current writer claim."}]}],
    )
    monkeypatch.setattr(
        replay_stage_module,
        "run_final_writer_agent",
        lambda **kwargs: {
            "report_markdown": "\n".join(section.get("claim", "") for chapter in kwargs.get("chapter_packages", []) for section in chapter.get("sections", [])),
            "source_registry": [],
            "citation_manifest": {},
            "final_citation_audit": {"status": "ok"},
        },
    )

    result = replay_stage(run_id=run_id, from_stage="writer", output_dir=tmp_path / "out")

    report_text = Path(result["report_path"]).read_text(encoding="utf-8")
    assert "Current writer claim" in report_text
    assert "OLD WRITER SNAPSHOT" not in report_text
