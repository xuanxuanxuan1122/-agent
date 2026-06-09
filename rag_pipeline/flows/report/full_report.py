from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

PIPELINE_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = PIPELINE_ROOT.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from rag_pipeline.runtime_cache import json_safe_default
from rag_pipeline.logging_utils import configure_pipeline_logging
from rag_pipeline.agents.public_report_sanitizer import find_publication_blockers, sanitize_public_markdown
from rag_pipeline.agents.report_health import build_report_health_card
from rag_pipeline.agents.summary_quality import sanitize_summary_judgments
from rag_pipeline.contracts.quality_gate import build_quality_gate_state
from rag_pipeline.observability.run_trace import write_run_trace_from_package


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
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key and not os.environ.get(key):
            os.environ[key] = value


def safe_filename(value: str, *, max_chars: int = 80) -> str:
    text = re.sub(r"\s+", "_", str(value or "").strip())
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", text)
    text = re.sub(r"_+", "_", text).strip("._")
    return (text or "report")[:max_chars]


def as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


REPAIR_TRACE_KEYS = (
    "evidence_preflight_trace",
    "layout_refinement_trace",
    "post_qa_repair_trace",
)


def repair_trace_payload_from_state(
    *,
    state_dict: Dict[str, Any],
    raw_output: Dict[str, Any],
    writer_report: Dict[str, Any],
) -> Dict[str, List[Any]]:
    containers = (
        as_dict(state_dict),
        as_dict(raw_output),
        as_dict(writer_report),
        as_dict(as_dict(raw_output).get("writer_report")),
    )
    payload: Dict[str, List[Any]] = {}
    for key in REPAIR_TRACE_KEYS:
        for container in containers:
            items = as_list(container.get(key))
            if items:
                payload[key] = items
                break
    return payload


def merge_source_registry_candidates(*registries: Any) -> List[Dict[str, Any]]:
    """Merge source registries without letting a compact writer snapshot hide the full evidence pool."""
    from rag_pipeline.agents.citation_manifest import merge_source_registries

    return merge_source_registries(
        *[
            [item for item in as_list(registry) if isinstance(item, dict)]
            for registry in registries
        ]
    )


def json_default(value: Any) -> Any:
    return json_safe_default(value)


def resolve_runtime_article_brief(
    *,
    raw_query: Any = "",
    title: Any = "",
    subtitle: Any = "",
    direction: Any = "",
    no_interactive: bool = False,
    input_fn: Any = input,
) -> Dict[str, Any]:
    from rag_pipeline.agents.article_brief import build_article_brief

    brief = build_article_brief(raw_query=raw_query, title=title, subtitle=subtitle, direction=direction)
    current_title = str(brief.get("display_title") or "").strip()
    current_direction = str(brief.get("direction") or "").strip()
    if no_interactive:
        if not str(brief.get("main_title") or "").strip():
            raise RuntimeError("Article title is required in non-interactive mode.")
        return {**brief, "interactive_confirmed": False}

    title_answer = str(input_fn(f"Report title [{current_title}]: ") or "").strip()
    resolved_title = title_answer or current_title
    direction_answer = str(input_fn(f"Report direction [{current_direction}]: ") or "").strip()
    resolved_direction = direction_answer or current_direction
    if not resolved_title:
        raise RuntimeError("Article title is required.")
    return build_article_brief(
        title=resolved_title,
        direction=resolved_direction,
        interactive_confirmed=True,
    )


QUIET_STAGE_LOGS = False


class OverallProgress:
    def __init__(self, *, enabled: bool, stream: Any = sys.stderr, width: int = 30) -> None:
        self.enabled = enabled
        self.stream = stream
        self.width = max(10, width)
        self.started_at = time.perf_counter()
        self.percent = 0.0
        self.label = ""
        self._lock = threading.Lock()
        self._pulse_stop: Optional[threading.Event] = None
        self._pulse_thread: Optional[threading.Thread] = None
        self._last_len = 0
        self._interactive = bool(getattr(stream, "isatty", lambda: False)())
        self._heartbeat_seconds = max(1.0, float(os.getenv("REPORT_PROGRESS_HEARTBEAT_SECONDS", "10") or 10))
        self._pulse_expected_seconds = max(60.0, float(os.getenv("REPORT_PROGRESS_PULSE_EXPECTED_SECONDS", "3600") or 3600))

    def _line(self, percent: float, label: str) -> str:
        percent = max(0.0, min(100.0, percent))
        filled = int(round(self.width * percent / 100.0))
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = time.perf_counter() - self.started_at
        return f"[{bar}] {percent:5.1f}%  {label}  elapsed {elapsed:5.1f}s"

    def _render_locked(self, *, newline: bool = False) -> None:
        if not self.enabled:
            return
        line = self._line(self.percent, self.label)
        if self._interactive:
            pad = " " * max(0, self._last_len - len(line))
            print("\r" + line + pad, end="\n" if newline else "", file=self.stream, flush=True)
            self._last_len = len(line)
            return
        print(line, file=self.stream, flush=True)

    def stop_pulse(self) -> None:
        event = self._pulse_stop
        thread = self._pulse_thread
        if event is not None:
            event.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._pulse_stop = None
        self._pulse_thread = None

    def update(self, percent: float, label: str) -> None:
        self.stop_pulse()
        with self._lock:
            self.percent = max(self.percent, min(100.0, float(percent)))
            self.label = str(label or self.label or "running")
            self._render_locked()

    def pulse_to(self, percent: float, label: str) -> None:
        if not self.enabled:
            return
        self.stop_pulse()
        target = max(self.percent, min(99.0, float(percent)))
        with self._lock:
            self.label = str(label or self.label or "running")
            self._render_locked()
        event = threading.Event()
        self._pulse_stop = event
        pulse_started_at = time.perf_counter()
        pulse_start_percent = self.percent

        def _pulse() -> None:
            interval = 0.6 if self._interactive else self._heartbeat_seconds
            while not event.wait(interval):
                with self._lock:
                    if self.percent < target:
                        elapsed = max(0.0, time.perf_counter() - pulse_started_at)
                        estimated = pulse_start_percent + (target - pulse_start_percent) * min(1.0, elapsed / self._pulse_expected_seconds)
                        self.percent = min(target, max(self.percent, estimated))
                    self._render_locked()

        thread = threading.Thread(target=_pulse, daemon=True)
        self._pulse_thread = thread
        thread.start()

    def finish(self, label: str = "done") -> None:
        self.stop_pulse()
        with self._lock:
            self.percent = 100.0
            self.label = label
            self._render_locked(newline=True)


def log(*values: Any, force: bool = False, **kwargs: Any) -> None:
    if QUIET_STAGE_LOGS and not force:
        return
    print(*values, file=sys.stderr, **kwargs)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pretty = env_flag("REPORT_PRETTY_JSON", False)
    separators = None if pretty else (",", ":")
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2 if pretty else None, separators=separators, default=json_default),
        encoding="utf-8",
    )


def write_stage_snapshot_safe(
    *,
    run_id: str,
    stage_name: str,
    payload: Any,
    summary: Optional[Dict[str, Any]] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    try:
        from rag_pipeline.cache.stage_snapshot_cache import write_stage_snapshot

        return write_stage_snapshot(
            stage_name=stage_name,
            run_id=run_id,
            payload=payload,
            summary=summary,
            diagnostics=diagnostics,
        )
    except Exception as exc:  # pragma: no cover - snapshots must never block report delivery.
        return {
            "enabled": True,
            "stored": False,
            "stage_name": stage_name,
            "run_id": run_id,
            "reason": "snapshot_write_failed",
            "error": str(exc),
        }


def init_artifact_ledger_run_safe(
    *,
    run_id: str,
    query: str,
    report_type: str = "full_report",
    freshness_policy: Optional[Dict[str, Any]] = None,
) -> tuple[Any, Dict[str, Any]]:
    try:
        from rag_pipeline.cache.artifact_store import default_artifact_store

        store = default_artifact_store()
        if not store.enabled():
            return None, {"enabled": False, "status": "disabled"}
        store.upsert_run(
            run_id=run_id,
            query=query,
            report_type=report_type,
            status="running",
            freshness_policy=freshness_policy or {},
        )
        return store, {
            "enabled": True,
            "status": "running",
            "ledger_path": str(store.path),
            "object_root": str(store.object_root),
        }
    except Exception as exc:  # pragma: no cover - artifact ledger must never block report delivery.
        return None, {"enabled": True, "status": "init_failed", "error": str(exc)}


def record_stage_snapshot_artifact_safe(
    store: Any,
    *,
    run_id: str,
    stage_name: str,
    payload: Any,
    snapshot_result: Dict[str, Any],
) -> Dict[str, Any]:
    if store is None:
        return {}
    try:
        from rag_pipeline.cache.artifact_pipeline_bridge import record_stage_snapshot_artifact

        return record_stage_snapshot_artifact(
            store,
            run_id=run_id,
            stage_name=stage_name,
            payload=payload,
            snapshot_result=snapshot_result,
        )
    except Exception as exc:  # pragma: no cover - artifact ledger must stay fail-open.
        return {"status": "record_failed", "error": str(exc)}


def sync_artifact_ledger_package_safe(
    store: Any,
    *,
    run_id: str,
    writer_package: Dict[str, Any],
    writer_report: Dict[str, Any],
    final_audit_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if store is None:
        return {}
    try:
        from rag_pipeline.cache.artifact_pipeline_bridge import ingest_writer_package_artifacts

        return ingest_writer_package_artifacts(
            store,
            run_id=run_id,
            writer_package=writer_package,
            writer_report=writer_report,
            final_audit_result=final_audit_result,
        )
    except Exception as exc:  # pragma: no cover - artifact ledger must stay fail-open.
        return {"enabled": True, "status": "sync_failed", "error": str(exc)}


def finish_artifact_ledger_run_safe(store: Any, *, run_id: str, query: str, status: str) -> Dict[str, Any]:
    if store is None:
        return {}
    try:
        store.upsert_run(run_id=run_id, query=query, report_type="full_report", status=status)
        return {"enabled": True, "status": status}
    except Exception as exc:  # pragma: no cover - artifact ledger must stay fail-open.
        return {"enabled": True, "status": "finish_failed", "error": str(exc)}


def attach_token_usage_snapshot(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from rag_pipeline.telemetry.token_usage import token_usage_payload

        usage_payload = token_usage_payload()
    except Exception as exc:  # pragma: no cover - telemetry must never block report delivery.
        payload["token_usage_error"] = str(exc)
        return payload
    summary = as_dict(usage_payload.get("token_usage_summary"))
    if summary.get("enabled") or int(summary.get("call_count") or 0) > 0:
        payload.update(usage_payload)
    return payload


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(finalize_public_report(str(text or "")).strip() + "\n", encoding="utf-8")


FORMAL_REPORT_PRIVATE_SENTENCE_PATTERNS = [
    "(?m)^\\s*[-*]?\\s*\\u62a5\\u544a\\u7ea7\\u68c0\\u7d22\\u7f3a\\u53e3[^\\n]*$",
    "(?m)^\\s*[-*]?\\s*[^\\n]{0,24}\\u68c0\\u7d22\\u7f3a\\u53e3[^\\n]*$",
    "(?m)^\\s*[-*]?\\s*[^\\n]{0,24}\\u8bc1\\u636e\\u7f3a\\u53e3[^\\n]*$",
    r"[^。\n]*正文应把[^。\n]*(?:。|$)",
    r"[^。\n]*正文应当[^。\n]*(?:。|$)",
    r"[^。\n]*本章应写成[^。\n]*(?:。|$)",
    r"[^。\n]*本章只能写成[^。\n]*(?:。|$)",
    r"[^。\n]*建议补证[^。\n]*(?:。|$)",
    r"[^。\n]*质量评分与证据限制[^。\n]*(?:。|$)",
    r"[^。\n]*可复核来源越独立[^。\n]*(?:。|$)",
    r"[^。\n]*目前结论仍受来源独立性[^。\n]*(?:。|$)",
    r"[^。\n]*优先复核可追溯来源[^。\n]*(?:。|$)",
    r"[^。\n]*本章关注[^。\n]*(?:。|$)",
    r"[^。\n]*本节围绕[^。\n]*(?:。|$)",
    r"[^。\n]*供给约束、需求兑现、价格利润、反向样本[^。\n]*(?:。|$)",
    r"[^。\n]*价格修复伴随库存下降[^。\n]*(?:。|$)",
    r"^#+\s*收入、利润与现金流质量\s*$",
    r"^#+\s*单位经济模型\s*$",
    r"^#+\s*投资优先级矩阵\s*$",
]


def strip_formal_report_private_sentences(markdown: str) -> str:
    text = str(markdown or "")
    for pattern in FORMAL_REPORT_PRIVATE_SENTENCE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.I)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def renumber_formal_chapter_headings(markdown: str) -> str:
    chapter_index = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal chapter_index
        chapter_index += 1
        title = re.sub(r"^\d+\.\s*", "", match.group(1).strip())
        return f"## {chapter_index}. {title}"

    return re.sub(r"^##\s+\d+\.\s+(.+?)\s*$", replace, markdown or "", flags=re.M)


def _filter_existing_executive_summary_block(markdown: str) -> str:
    lines = str(markdown or "").splitlines()
    output: List[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if not re.match(r"^##\s*核心观点与主要结论\s*$", line.strip()):
            output.append(line)
            index += 1
            continue
        heading = line
        block: List[str] = []
        index += 1
        while index < len(lines) and not re.match(r"^##\s+", lines[index].strip()):
            block.append(lines[index])
            index += 1
        bullet_payloads: List[Dict[str, Any]] = []
        passthrough: List[str] = []
        for item in block:
            stripped = item.strip()
            if not stripped:
                continue
            bullet_match = re.match(r"^[-*]\s*(.+?)\s*$", stripped)
            if bullet_match:
                bullet_payloads.append({"judgment": bullet_match.group(1)})
            else:
                passthrough.append(stripped)
        valid, _diag = sanitize_summary_judgments(bullet_payloads, max_items=5)
        valid_lines = [f"- {item.get('judgment')}" for item in valid]
        for item in passthrough:
            valid_extra, _ = sanitize_summary_judgments([{"judgment": item}], max_items=1)
            if valid_extra:
                valid_lines.append(str(valid_extra[0].get("judgment") or "").strip())
        if valid_lines:
            output.append(heading)
            output.extend(valid_lines)
    return "\n".join(output)


def finalize_formal_report(markdown: str) -> str:
    text = str(markdown or "")
    try:
        cleaned = _filter_existing_executive_summary_block(finalize_public_report(text))
        return renumber_formal_chapter_headings(strip_formal_report_private_sentences(cleaned))
    except ValueError:
        cleaned = sanitize_public_markdown(text)
        for _ in range(8):
            blockers = find_publication_blockers(cleaned)
            if not blockers:
                break
            blocked_lines = {int(item.get("line") or 0) for item in blockers}
            cleaned = "\n".join(
                line
                for line_no, line in enumerate(cleaned.splitlines(), start=1)
                if line_no not in blocked_lines
            )
            cleaned = sanitize_public_markdown(cleaned)
        cleaned = _filter_existing_executive_summary_block(cleaned)
        return renumber_formal_chapter_headings(strip_formal_report_private_sentences(cleaned)).strip()


_SOURCE_APPENDIX_HEADING_RE = re.compile(
    r"(?m)^##\s*(?:来源附录|数据来源列表|数据来源|研究口径与来源|参考来源|参考资料|Source Appendix|Sources)\b"
)


def _public_body_markdown_for_health(markdown: str) -> str:
    text = str(markdown or "")
    match = _SOURCE_APPENDIX_HEADING_RE.search(text)
    if match:
        return text[: match.start()].strip()
    return text.strip()


def _public_body_char_count(markdown: str) -> int:
    return len(re.sub(r"\s+", "", _public_body_markdown_for_health(markdown)))


def write_formal_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(finalize_formal_report(str(text or "")).strip() + "\n", encoding="utf-8")


def finalize_formal_report_and_refresh_audit(
    markdown: str,
    writer_report: Dict[str, Any],
) -> tuple[str, Dict[str, Any]]:
    """Finalize public markdown and keep writer diagnostics in the same state.

    ``write_formal_markdown`` applies a last public-report sanitizer.  If score
    rendering or FinalAudit inspect the pre-sanitized string, they can report
    citationless diagnostic lines that are no longer present in the actual
    ``*_report.md``.  This helper moves that finalization before audit/score and
    refreshes the final citation audit against the exact markdown that will be
    written.
    """

    finalized = finalize_formal_report(str(markdown or ""))
    report = dict(writer_report or {})
    render_artifacts = dict(as_dict(report.get("render_artifacts")))
    report["report_markdown"] = finalized
    try:
        from rag_pipeline.agents.final_writer_agent import _rewrite_final_markdown_with_reconciled_appendix

        citation_manifest = as_dict(report.get("citation_manifest")) or as_dict(render_artifacts.get("citation_manifest"))
        source_registry = (
            as_list(report.get("source_registry"))
            or as_list(render_artifacts.get("source_registry"))
            or as_list(report.get("sources"))
        )
        appendix_package = as_dict(report.get("appendix_package")) or as_dict(render_artifacts.get("appendix_package"))
        if citation_manifest or source_registry:
            finalized, refreshed_sources, refreshed_audit = _rewrite_final_markdown_with_reconciled_appendix(
                finalized,
                citation_manifest=citation_manifest,
                source_registry=source_registry,
                appendix_package=appendix_package,
            )
            report["report_markdown"] = finalized
            report["source_registry"] = refreshed_sources
            report["final_citation_audit"] = refreshed_audit
            render_artifacts["source_registry"] = refreshed_sources
            render_artifacts["final_citation_audit"] = refreshed_audit
            if citation_manifest:
                render_artifacts["citation_manifest"] = citation_manifest
            report["render_artifacts"] = render_artifacts
    except Exception as exc:  # pragma: no cover - diagnostics must not block report output.
        report.setdefault("finalization_warnings", []).append(f"final_citation_refresh_failed:{type(exc).__name__}")
    return str(report.get("report_markdown") or finalized).strip(), report


def write_score_markdown(path: Path, text: str) -> None:
    """Write the audit/score companion file without public-report sanitizing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(text or "").strip() + "\n", encoding="utf-8")


FINAL_AUDIT_PUBLIC_LABELS = {
    "unsupported_claim": "部分判断支撑不足",
    "evidence_gap": "部分证据链仍有缺口",
    "citation_issue": "引用或来源附录需要复核",
    "data_conflict": "数据口径存在冲突",
    "logic_jump": "存在推理跨度偏大的判断",
    "risk_understated": "风险提示不够充分",
    "scope_issue": "适用范围需要收窄",
    "missing_sources_appendix": "来源附录不完整",
    "title_only_source": "部分来源缺少可独立访问链接",
    "fake_or_placeholder_source": "来源登记中存在占位来源",
    "fake_or_placeholder_evidence": "正文存在占位证据痕迹",
    "internal_evidence_id": "正文存在内部证据编号",
    "internal_chapter_id": "正文存在内部章节编号",
    "empty_markdown_table": "存在空表或坏表",
}


def _public_audit_label(value: Any) -> str:
    text = str(value or "audit_finding").strip()
    return FINAL_AUDIT_PUBLIC_LABELS.get(text, text.replace("_", " "))


def final_audit_public_note(final_audit_result: Dict[str, Any]) -> str:
    result = as_dict(final_audit_result)
    if not result or not result.get("enabled"):
        return ""
    audit = as_dict(result.get("audit"))
    deterministic = as_dict(result.get("deterministic_audit"))
    findings = []
    seen_findings = set()
    for item in [*as_list(deterministic.get("findings")), *as_list(audit.get("critical_findings"))]:
        payload = as_dict(item)
        key = (
            str(payload.get("type") or ""),
            str(payload.get("severity") or ""),
            str(payload.get("message") or payload.get("suggested_fix") or "")[:180],
        )
        if key in seen_findings:
            continue
        seen_findings.add(key)
        findings.append(payload)
    status = str(result.get("status") or audit.get("status") or "").strip() or "unknown"
    blocked = bool(result.get("blocked"))
    lines = [
        "## 最终审查补充",
        "",
        f"- 最终审查状态：{status}",
        f"- 洁净版资格：{'暂不建议自动交付' if blocked else '未发现阻断洁净版的问题'}",
    ]
    summary = str(audit.get("summary") or deterministic.get("summary") or "").strip()
    if summary:
        lines.append(f"- 审查摘要：{summary[:220]}")
    if findings:
        lines.extend(["", "### 审查发现"])
        for item in findings[:8]:
            payload = as_dict(item)
            label = _public_audit_label(payload.get("type"))
            message = str(payload.get("message") or payload.get("suggested_fix") or "").strip()
            severity = str(payload.get("severity") or "").strip()
            detail = "；".join(part for part in [severity, message[:180]] if part)
            lines.append(f"- {label}" + (f"：{detail}" if detail else ""))
    return "\n".join(lines).strip()


def append_final_audit_note(markdown: str, final_audit_result: Dict[str, Any]) -> str:
    note = final_audit_public_note(final_audit_result)
    text = str(markdown or "").strip()
    if not note or not text or "## 最终审查补充" in text:
        return text
    return f"{text}\n\n{note}".strip()


def _score_from_writer_report(writer_report: Dict[str, Any]) -> Any:
    qa = as_dict(writer_report.get("qa_result"))
    clean_gate = as_dict(qa.get("clean_gate"))
    for value in (
        writer_report.get("quality_score"),
        qa.get("quality_score"),
        clean_gate.get("quality_score"),
    ):
        try:
            return max(0, min(100, int(float(value))))
        except (TypeError, ValueError):
            continue
    return 0


def _grade_from_score(score: Any) -> str:
    try:
        value = int(float(score))
    except (TypeError, ValueError):
        return "未计算"
    if value >= 90:
        return "可发布级"
    if value >= 75:
        return "高质量但需人工复核"
    if value >= 60:
        return "证据有限但可参考"
    return "强风险报告，仅供内部研判"


def _analysis_contract_diagnostics(writer_package: Dict[str, Any]) -> Dict[str, Any]:
    writer_report = as_dict(writer_package.get("writer_report"))
    render_artifacts = as_dict(writer_report.get("render_artifacts"))
    naturalness_cleanup = as_dict(writer_report.get("naturalness_cleanup"))
    chapter_narrative_diag = (
        as_dict(render_artifacts.get("chapter_narrative"))
        or as_dict(writer_package.get("chapter_narrative"))
        or as_dict(writer_report.get("chapter_narrative"))
    )
    structured = (
        as_dict(render_artifacts.get("structured_analysis"))
        or as_dict(writer_package.get("structured_analysis"))
        or as_dict(writer_report.get("structured_analysis"))
    )
    metadata = as_dict(writer_report.get("metadata"))
    contract = as_dict(structured.get("analysis_contract_status"))
    stage = {
        **as_dict(writer_package.get("analysis_stage_diagnostics")),
        **as_dict(metadata.get("analysis_stage_diagnostics")),
        **as_dict(structured.get("analysis_stage_diagnostics")),
    }
    render_full = bool(
        render_artifacts
        and str(render_artifacts.get("payload_mode") or "") == "full"
        and as_list(render_artifacts.get("chapter_packages"))
        and as_list(render_artifacts.get("argument_units"))
    )
    analysis_rebuilt = bool(
        structured.get("analysis_rebuilt_from_evidence")
        or contract.get("analysis_rebuilt_from_evidence")
        or stage.get("analysis_rebuilt_from_evidence")
    )
    uses_llm = stage.get("uses_llm_analysis")
    final_source = stage.get("final_analysis_source")
    if not final_source:
        if uses_llm is True:
            final_source = "llm_evidence_analysis"
        elif analysis_rebuilt:
            final_source = "deterministic_rebuild"
        else:
            final_source = "dynamic_claim_builder"
    deterministic_used = stage.get("deterministic_synthesis_used")
    if deterministic_used is None:
        deterministic_used = final_source != "llm_evidence_analysis"
    llm_attempted = stage.get("llm_analysis_attempted")
    if llm_attempted is None:
        llm_attempted = str(stage.get("llm_analysis_status") or "").strip() not in {"", "not_run", "disabled", "fallback_config_missing"}
    quality_degraded = stage.get("quality_path_degraded")
    if quality_degraded is None:
        quality_requested = bool(
            writer_report.get("quality_mode")
            or writer_package.get("quality_mode")
            or str(writer_report.get("report_execution_mode") or writer_package.get("report_execution_mode") or "").strip()
            in {"quality_llm_replay", "live_quality_full"}
        )
        quality_degraded = bool(quality_requested and final_source != "llm_evidence_analysis")
    return {
        "structured_analysis_valid": contract.get("structured_analysis_valid"),
        "analysis_rebuilt_from_evidence": analysis_rebuilt,
        "uses_llm_analysis": uses_llm,
        "llm_analysis_attempted": bool(llm_attempted),
        "llm_analysis_status": stage.get("llm_analysis_status"),
        "final_analysis_source": final_source,
        "deterministic_synthesis_used": deterministic_used,
        "quality_path_degraded": bool(quality_degraded),
        "quality_path_degradation_reason": stage.get("quality_path_degradation_reason") or stage.get("fallback_reason") or "",
        "llm_input_valid_ref_count": stage.get("llm_input_valid_ref_count", 0),
        "llm_usable_claim_count": stage.get("llm_usable_claim_count", 0),
        "llm_dropped_claim_count": stage.get("llm_dropped_claim_count", 0),
        "llm_usable_chapter_count": stage.get("llm_usable_chapter_count", 0),
        "llm_valid_chapter_count": stage.get("llm_valid_chapter_count", stage.get("llm_usable_chapter_count", 0)),
        "llm_failed_chapter_count": stage.get("llm_failed_chapter_count", 0),
        "llm_analysis_cache_hit_count": stage.get("llm_analysis_cache_hit_count", 0),
        "llm_raw_chapter_count": stage.get("llm_raw_chapter_count", 0),
        "llm_raw_claim_count": stage.get("llm_raw_claim_count", 0),
        "llm_validation_issue_counts": stage.get("llm_validation_issue_counts", {}),
        "llm_validation_issue_examples": stage.get("llm_validation_issue_examples", []),
        "llm_valid_claim_examples": stage.get("llm_valid_claim_examples", []),
        "llm_rejected_claim_examples": stage.get("llm_rejected_claim_examples", []),
        "llm_validation_status": stage.get("llm_validation_status"),
        "analysis_contract_issues": contract.get("issues") or stage.get("previous_contract_issues") or [],
        "analysis_rebuild_reasons": contract.get("quality_rebuild_reasons") or stage.get("analysis_rebuild_reasons") or [],
        "claim_unit_count": contract.get("claim_unit_count") or len(as_list(structured.get("claim_units"))),
        "chapter_insight_count": contract.get("chapter_insight_count")
        or len(as_list(structured.get("chapter_insights")))
        or len(as_list(as_dict(structured.get("report_insight_package")).get("chapters"))),
        "evidence_analysis_count": contract.get("evidence_analysis_count") or len(as_list(structured.get("evidence_analyses"))),
        "render_artifacts_full": render_full,
        "compacted_artifact_used": not render_full,
    }


def _insufficient_analysis_signal(writer_report: Dict[str, Any]) -> Dict[str, Any]:
    """Detect the "analysis produced no real LLM claims" (vacuous) condition.

    When the LLM analysis stage yields zero usable claims the pipeline falls back
    to a deterministic template rebuild that reads fluent but is vacuous. Rather
    than ship that long filler, we emit a short honest stub (P0 guardrail).

    Triggers only on the *positive* ``deterministic_rebuild`` + zero-usable-claim
    signal, so missing/partial diagnostics can never false-trigger the stub.
    """
    diag = _analysis_contract_diagnostics({"writer_report": writer_report})
    final_source = str(diag.get("final_analysis_source") or "")
    try:
        usable = int(float(diag.get("llm_usable_claim_count") or 0))
    except (TypeError, ValueError):
        usable = 0
    insufficient = final_source == "deterministic_rebuild" and usable <= 0
    return {"insufficient": insufficient, "diagnostics": diag}


_TEMPLATE_FILLER_FINGERPRINT = "是把事实转成判断的核心连接点"


def _insufficient_analysis_delivery_action(
    report_markdown: str,
    writer_report: Dict[str, Any],
) -> Dict[str, Any]:
    """Decide whether zero-usable-claim analysis should replace or downgrade.

    Zero usable LLM claims means "not Clean", not automatically "no formal
    report". Keep a fact-backed formal draft when the writer already produced a
    citation-bearing body with traceable sources; fall back to the short honest
    stub only when the public surface is too small or cannot be cited.
    """

    signal = _insufficient_analysis_signal(writer_report)
    if not signal.get("insufficient"):
        return {"mode": "normal", "replace_with_stub": False, "diagnostics": signal.get("diagnostics") or {}}

    text = str(report_markdown or "").strip()
    dense_chars = len(re.sub(r"\s+", "", _public_body_markdown_for_health(text)))
    citations = re.findall(r"\[\d{1,5}\]", text)
    render_artifacts = as_dict(writer_report.get("render_artifacts"))
    sources = merge_source_registry_candidates(
        as_list(writer_report.get("source_registry")),
        as_list(render_artifacts.get("source_registry")),
        as_list(writer_report.get("sources")),
    )
    traceable_sources = [
        source
        for source in sources
        if str(as_dict(source).get("url") or as_dict(source).get("canonical_url") or "").strip()
    ]
    has_source_appendix = bool(
        re.search(r"(?mi)^##\s*(来源附录|数据来源|参考来源|参考资料|Source Appendix|Sources)\b", text)
    )
    min_dense_chars = env_large_int(
        "REPORT_INSUFFICIENT_FORMAL_MIN_DENSE_CHARS",
        500,
        min_value=200,
        max_value=5000,
    )
    template_risk = _TEMPLATE_FILLER_FINGERPRINT in text or has_legacy_decision_sections(text)
    can_keep_formal = bool(
        dense_chars >= min_dense_chars
        and citations
        and (traceable_sources or has_source_appendix)
        and not template_risk
    )
    if can_keep_formal:
        return {
            "mode": "limited_evidence_formal_report",
            "replace_with_stub": False,
            "report_status": "formal_scored",
            "delivery_tier": "limited_evidence_formal_report",
            "diagnostics": signal.get("diagnostics") or {},
            "dense_chars": dense_chars,
            "citation_count": len(citations),
            "traceable_source_count": len(traceable_sources),
        }
    return {
        "mode": "insufficient_analysis_stub",
        "replace_with_stub": True,
        "report_status": "insufficient_analysis_stub",
        "delivery_tier": "insufficient_analysis_stub",
        "diagnostics": signal.get("diagnostics") or {},
        "dense_chars": dense_chars,
        "citation_count": len(citations),
        "traceable_source_count": len(traceable_sources),
    }


def _build_insufficient_stub_markdown(
    query: str,
    writer_report: Dict[str, Any],
    diagnostics: Dict[str, Any],
) -> str:
    """Short, honest "insufficient analysis" report.

    Instead of a long template-filler report, state plainly: the thesis we could
    not substantiate, the high-trust sources we *did* gather, and the concrete
    gaps. This protects trust when the evidence chain breaks.
    """
    reason = str(
        diagnostics.get("quality_path_degradation_reason")
        or diagnostics.get("llm_analysis_status")
        or "analysis_produced_no_usable_claims"
    )
    render_artifacts = as_dict(writer_report.get("render_artifacts"))
    sources = merge_source_registry_candidates(
        as_list(writer_report.get("source_registry")),
        as_list(render_artifacts.get("source_registry")),
        as_list(writer_report.get("sources")),
    )

    def _level(source: Any) -> str:
        return str(as_dict(source).get("source_level") or as_dict(source).get("credibility") or "").strip().upper()

    def _src_line(source: Any) -> str:
        item = as_dict(source)
        title = str(item.get("title") or item.get("name") or item.get("url") or item.get("canonical_url") or "").strip()
        url = str(item.get("url") or item.get("canonical_url") or "").strip()
        level = _level(item)
        tag = f"[{level}] " if level in {"A", "B", "C", "D"} else ""
        line = f"- {tag}{title}".rstrip()
        return f"{line} — {url}" if url else line

    preferred = [s for s in sources if _level(s) in {"A", "B"}] or sources
    chosen = [s for s in preferred if str(as_dict(s).get("title") or as_dict(s).get("url") or "").strip()][:6]

    qa = as_dict(writer_report.get("qa_result"))
    gap_msgs: List[str] = []
    for key in (
        "blocking_followups",
        "blocking_evidence_repair_followups",
        "blocking_content_repair_followups",
        "required_followups",
        "quality_findings",
    ):
        for item in as_list(qa.get(key)):
            item = as_dict(item)
            msg = str(
                item.get("message")
                or item.get("reason")
                or item.get("gap_type")
                or item.get("requirement_id")
                or ""
            ).strip()
            if msg and msg not in gap_msgs:
                gap_msgs.append(msg)

    lines = [
        f"# {query} — 研究简报（未达可发布）",
        "",
        "> **状态：分析证据不足，未生成可发布报告。**",
        "> 系统在证据链不足时不再硬凑正式长文，改为如实说明已掌握与缺失，避免输出看似正式实则空洞的报告。",
        f"> 触发原因：`{reason}`。",
        "",
        "## 已掌握的来源",
    ]
    lines += [_src_line(s) for s in chosen] if chosen else ["- （本次未沉淀到可信来源）"]
    lines += ["", "## 主要缺口"]
    if gap_msgs:
        lines += [f"- {m}" for m in gap_msgs[:8]]
    else:
        lines += ["- 关键章节缺少可绑定的高可信证据，无法形成可发布的方向性判断。"]
    lines += [
        "",
        "## 结论",
        "- 当前证据不足以支撑可发布的行业判断；需补齐上述缺口后重跑。",
        "- 完整诊断见同目录 `_score.md` 与 `.trace_summary.md`。",
    ]
    return "\n".join(lines).strip() + "\n"


def _executive_summary_diagnostics(writer_package: Dict[str, Any], writer_report: Dict[str, Any]) -> Dict[str, Any]:
    render_artifacts = as_dict(writer_report.get("render_artifacts"))
    structured = (
        as_dict(render_artifacts.get("structured_analysis"))
        or as_dict(writer_package.get("structured_analysis"))
        or as_dict(writer_report.get("structured_analysis"))
    )
    report_insight = as_dict(structured.get("report_insight_package"))
    executive_summary = as_dict(report_insight.get("executive_summary"))
    quality = {
        **as_dict(structured.get("executive_summary_quality")),
        **as_dict(report_insight.get("executive_summary_quality")),
    }
    decision_package = (
        as_dict(writer_report.get("decision_package"))
        or as_dict(writer_package.get("decision_package"))
        or as_dict(render_artifacts.get("decision_package"))
    )
    candidates: List[Dict[str, Any]] = []
    candidates.extend(as_dict(item) for item in as_list(decision_package.get("core_judgments")) if isinstance(item, dict))
    candidates.extend(as_dict(item) for item in as_list(executive_summary.get("top_3_judgments")) if isinstance(item, dict))
    if str(report_insight.get("report_thesis") or "").strip():
        candidates.append({"judgment": report_insight.get("report_thesis")})
    valid, computed = sanitize_summary_judgments(candidates, max_items=5)
    return {
        "executive_summary_valid_judgment_count": quality.get(
            "executive_summary_valid_judgment_count",
            computed.get("executive_summary_valid_judgment_count", len(valid)),
        ),
        "executive_summary_filtered_judgment_count": quality.get(
            "executive_summary_filtered_judgment_count",
            computed.get("executive_summary_filtered_judgment_count", 0),
        ),
        "executive_summary_fallback_used": bool(quality.get("executive_summary_fallback_used", False)),
        "executive_summary_omitted_low_quality": bool(
            quality.get("executive_summary_omitted_low_quality", computed.get("executive_summary_omitted_low_quality", False))
        ),
        "filtered_summary_examples": quality.get("filtered_summary_examples")
        or computed.get("filtered_summary_examples")
        or [],
    }


def _readpage_fact_extractor_diagnostics(writer_package: Dict[str, Any]) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = []
    candidate_ids: set[int] = set()
    candidate_fingerprints: set[str] = set()

    def add_candidate(payload: Any) -> None:
        payload = as_dict(payload)
        if not payload:
            return
        marker = id(payload)
        if marker in candidate_ids:
            return
        candidate_ids.add(marker)
        try:
            fingerprint = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        except Exception:
            fingerprint = repr(sorted(payload.items()))
        if fingerprint in candidate_fingerprints:
            return
        candidate_fingerprints.add(fingerprint)
        candidates.append(payload)

    writer_report = as_dict(writer_package.get("writer_report"))
    raw_output = as_dict(writer_package.get("raw_output"))
    metadata = as_dict(writer_package.get("metadata"))
    writer_metadata = as_dict(writer_report.get("metadata"))
    render_artifacts = as_dict(writer_report.get("render_artifacts"))
    evidence_package = as_dict(writer_package.get("evidence_package"))
    render_evidence_package = as_dict(render_artifacts.get("evidence_package"))
    for payload in (
        writer_package.get("fact_extractor"),
        writer_package.get("readpage_fact_extractor"),
        writer_report.get("fact_extractor"),
        writer_report.get("readpage_fact_extractor"),
        raw_output.get("fact_extractor"),
        raw_output.get("readpage_fact_extractor"),
        metadata.get("readpage_fact_extractor"),
        metadata.get("fact_extractor"),
        writer_metadata.get("readpage_fact_extractor"),
        writer_metadata.get("fact_extractor"),
        as_dict(evidence_package.get("metadata")).get("readpage_fact_extractor"),
        as_dict(evidence_package.get("metadata")).get("fact_extractor"),
        as_dict(render_evidence_package.get("metadata")).get("readpage_fact_extractor"),
        as_dict(render_evidence_package.get("metadata")).get("fact_extractor"),
        as_dict(render_artifacts.get("metadata")).get("readpage_fact_extractor"),
        as_dict(render_artifacts.get("metadata")).get("fact_extractor"),
    ):
        add_candidate(payload)
    if not candidates:
        return {}
    totals = {
        "attempted": 0,
        "success_count": 0,
        "fact_card_count": 0,
        "rejected_span_count": 0,
        "invalid_metric_count": 0,
        "cache_hit_count": 0,
        "llm_error_count": 0,
        "regex_fallback_point_count": 0,
        "extractor_empty_without_regex_points": False,
        "budget_limit": 0,
        "budget_used": 0,
        "budget_exhausted": False,
        "regex_fallback_used": False,
        "fallback_used": False,
        "statuses": [],
        "models": [],
    }
    for payload in candidates:
        for key in (
            "attempted",
            "success_count",
            "fact_card_count",
            "rejected_span_count",
            "invalid_metric_count",
            "cache_hit_count",
            "llm_error_count",
            "regex_fallback_point_count",
            "budget_used",
        ):
            try:
                totals[key] += int(float(payload.get(key) or 0))
            except (TypeError, ValueError):
                continue
        if payload.get("budget_limit") not in (None, ""):
            try:
                totals["budget_limit"] = max(int(totals.get("budget_limit") or 0), int(float(payload.get("budget_limit") or 0)))
            except (TypeError, ValueError):
                pass
        totals["regex_fallback_used"] = bool(totals["regex_fallback_used"] or payload.get("regex_fallback_used"))
        totals["extractor_empty_without_regex_points"] = bool(
            totals["extractor_empty_without_regex_points"] or payload.get("extractor_empty_without_regex_points")
        )
        totals["budget_exhausted"] = bool(totals["budget_exhausted"] or payload.get("budget_exhausted"))
        totals["fallback_used"] = bool(totals["fallback_used"] or payload.get("fallback_used"))
        status = str(payload.get("status") or "").strip()
        if status and status not in totals["statuses"]:
            totals["statuses"].append(status)
        model = str(payload.get("model") or "").strip()
        if model and model not in totals["models"]:
            totals["models"].append(model)
    return totals


def _coerce_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _layout_score_diagnostics(writer_package: Dict[str, Any]) -> Dict[str, Any]:
    writer_report = as_dict(writer_package.get("writer_report"))
    render_artifacts = as_dict(writer_report.get("render_artifacts"))
    naturalness_cleanup = as_dict(writer_report.get("naturalness_cleanup"))
    chapter_narrative_diag = (
        as_dict(render_artifacts.get("chapter_narrative"))
        or as_dict(writer_package.get("chapter_narrative"))
        or as_dict(writer_report.get("chapter_narrative"))
    )
    micro_layouts = (
        as_list(render_artifacts.get("micro_layouts"))
        or as_list(writer_package.get("micro_layouts"))
        or as_list(writer_report.get("micro_layouts"))
    )
    chapter_packages = (
        as_list(render_artifacts.get("chapter_packages"))
        or as_list(writer_package.get("chapter_packages"))
        or as_list(writer_report.get("chapter_packages"))
    )
    table_packages = (
        as_list(render_artifacts.get("table_packages"))
        or as_list(writer_package.get("table_packages"))
        or as_list(writer_report.get("table_packages"))
    )
    compacted_artifact_used = not bool(as_list(render_artifacts.get("chapter_packages")))
    chapter_by_id = {
        str(chapter.get("chapter_id") or "").strip(): chapter
        for chapter in chapter_packages
        if isinstance(chapter, dict) and str(chapter.get("chapter_id") or "").strip()
    }

    def _is_core_chapter_for_health(chapter: Dict[str, Any], layout: Dict[str, Any]) -> bool:
        for payload in (chapter, layout):
            if bool(payload.get("is_core_chapter") or payload.get("core_chapter") or payload.get("required_chapter")):
                return True
            marker = str(
                payload.get("chapter_priority")
                or payload.get("priority")
                or payload.get("chapter_type")
                or payload.get("importance")
                or ""
            ).strip().lower()
            if marker in {"core", "required", "must", "main"}:
                return True
        return False

    chapters: List[Dict[str, Any]] = []
    dropped_blocks: List[Dict[str, Any]] = []
    observation_only_blocks: List[Dict[str, Any]] = []
    evidence_backed_blocks: List[Dict[str, Any]] = []
    omitted_observation_blocks: List[Dict[str, Any]] = []
    template_removed_blocks: List[Dict[str, Any]] = []
    block_without_fact_card_blocks: List[Dict[str, Any]] = []
    fact_card_matched_blocks: List[Dict[str, Any]] = []
    rendered_section_titles: List[str] = []
    dynamic_section_title_count = 0
    block_title_generation_failed_count = 0
    dropped_generic_block_count = 0
    section_plan_valid_count = 0
    section_plan_dropped_count = 0
    natural_title_count = 0
    generic_title_filtered_count = 0
    repeated_title_rewrite_count = 0
    repeated_title_dropped_count = 0
    internal_role_title_filtered_count = 0
    weak_template_sentence_removed_count = 0
    publisher_in_title_count = 0
    publisher_in_subject_count = 0
    english_snippet_lead_count = 0
    empty_parens_in_body_count = 0
    snippet_lead_rewritten_count = 0
    snippet_lead_dropped_count = 0
    repeated_evidence_id_within_chapter_count = 0
    repeated_fact_demoted_count = 0
    snippet_like_text_dropped_count = 0
    metric_sentence_rewritten_count = 0
    traditional_chinese_normalized_count = 0
    chapter_omitted_no_evidence_count = 0
    core_chapter_omitted_no_evidence_count = 0
    optional_chapter_omitted_count = 0
    composer_variable_explanation_count = 0
    composed_section_count = 0
    fact_passthrough_section_count = 0
    body_rewrite_enabled = False
    body_rewrite_called_count = 0
    body_rewrite_success_count = 0
    body_rewrite_cache_hit_count = 0
    body_rewrite_rejected_count = 0
    body_rewrite_fallback_count = 0
    body_rewrite_skipped_count = 0
    body_rewrite_submitted_count = 0
    body_rewrite_budget_exhausted_count = 0
    body_rewrite_inflight_dedup_count = 0
    body_rewrite_elapsed_seconds = 0.0
    body_rewrite_concurrency = 0
    body_rewrite_failure_reasons: Dict[str, int] = {}
    must_render_block_count = 0
    candidate_block_count = 0
    rendered_must_block_count = 0
    llm_claim_to_block_match_count = 0
    llm_claim_unmatched_count = 0
    must_block_matched_by_llm_claim_count = 0
    must_block_dropped_no_matching_claim_count = 0
    publisher_or_domain_re = re.compile(
        r"爱集微|ijiwei|36kr|36氪|财联社|界面|新浪|腾讯|网易|搜狐|百度|知乎|雪球|虎嗅|钛媒体|亿欧|证券时报|证券日报|人民网|新华社|央视|凤凰网|[a-z0-9-]+\.(?:com|cn|net|org)",
        re.I,
    )
    generic_titles = {"事实依据", "商业化证据", "核心观察", "本章结论"}
    generic_titles.update({"事实依据", "商业化证据", "核心观察", "本章结论"})
    traditional_re = re.compile(r"[發佈體團業務軟證據場應與實驗轉進階價為單個對雲數電費戶產鏈]")
    rendered_block_count = 0
    scored_chapter_ids: set[str] = set()
    body_rewrite_global_seen = False
    for layout in micro_layouts:
        if not isinstance(layout, dict):
            continue
        chapter_id = str(layout.get("chapter_id") or "").strip()
        if chapter_id:
            scored_chapter_ids.add(chapter_id)
        block_records = [as_dict(block) for block in as_list(layout.get("blocks")) if isinstance(block, dict)]
        if "must_render_blocks" in layout:
            must_block_records = [as_dict(block) for block in as_list(layout.get("must_render_blocks")) if isinstance(block, dict)]
        else:
            must_block_records = [
                block
                for block in block_records
                if str(block.get("render_plan") or "must_render") == "must_render"
                and block.get("public_render") is not False
            ]
        if "candidate_blocks" in layout:
            candidate_block_records = [as_dict(block) for block in as_list(layout.get("candidate_blocks")) if isinstance(block, dict)]
        else:
            candidate_block_records = [
                block
                for block in block_records
                if str(block.get("render_plan") or "") == "candidate"
                or block.get("public_render") is False
            ]
        planned_blocks = [
            str(block.get("block_type") or "").strip()
            for block in must_block_records
            if str(block.get("block_type") or "").strip()
        ]
        candidate_blocks = [
            str(block.get("block_type") or "").strip()
            for block in candidate_block_records
            if str(block.get("block_type") or "").strip()
        ]
        claim_layout_diag = as_dict(layout.get("claim_layout_match_diagnostics"))
        llm_claim_to_block_match_count += _coerce_int(claim_layout_diag.get("llm_claim_to_block_match_count"))
        llm_claim_unmatched_count += _coerce_int(claim_layout_diag.get("llm_claim_unmatched_count"))
        must_block_matched_by_llm_claim_count += _coerce_int(
            claim_layout_diag.get("must_block_matched_by_llm_claim_count")
        )
        must_block_dropped_no_matching_claim_count += _coerce_int(
            claim_layout_diag.get("must_block_dropped_no_matching_claim_count")
        )
        must_render_block_count += len(planned_blocks)
        candidate_block_count += len(candidate_blocks)
        for dropped in as_list(layout.get("dropped_sections")):
            payload = as_dict(dropped)
            reason = str(payload.get("reason") or "")
            block_type = str(payload.get("block_type") or payload.get("output_type") or "").strip()
            dropped_blocks.append({"chapter_id": chapter_id, "block_type": block_type, "reason": reason or "layout_block_dropped"})
            if reason in {"block_title_generation_failed", "dropped_generic_block"}:
                dropped_generic_block_count += 1
                block_title_generation_failed_count += 1 if reason == "block_title_generation_failed" else 0
            if reason == "repeated_title_dropped":
                repeated_title_dropped_count += 1
            if reason == "internal_role_title_filtered":
                internal_role_title_filtered_count += 1
        chapter = as_dict(chapter_by_id.get(chapter_id))
        body_rewrite_global = as_dict(chapter.get("body_rewrite_global"))
        if body_rewrite_global and not body_rewrite_global_seen:
            body_rewrite_global_seen = True
            body_rewrite_enabled = True
            body_rewrite_submitted_count += _coerce_int(body_rewrite_global.get("submitted_count"))
            body_rewrite_called_count += _coerce_int(body_rewrite_global.get("called_count"))
            body_rewrite_success_count += _coerce_int(body_rewrite_global.get("success_count"))
            body_rewrite_cache_hit_count += _coerce_int(body_rewrite_global.get("cache_hit_count"))
            body_rewrite_rejected_count += _coerce_int(body_rewrite_global.get("rejected_count"))
            body_rewrite_fallback_count += _coerce_int(body_rewrite_global.get("fallback_count"))
            body_rewrite_skipped_count += _coerce_int(body_rewrite_global.get("skipped_count"))
            body_rewrite_budget_exhausted_count += _coerce_int(body_rewrite_global.get("budget_exhausted_count"))
            body_rewrite_inflight_dedup_count += _coerce_int(body_rewrite_global.get("inflight_dedup_count"))
            body_rewrite_concurrency = max(body_rewrite_concurrency, _coerce_int(body_rewrite_global.get("concurrency")))
            for reason, count in as_dict(body_rewrite_global.get("failure_reasons")).items():
                if reason:
                    body_rewrite_failure_reasons[str(reason)] = body_rewrite_failure_reasons.get(str(reason), 0) + _coerce_int(count)
            try:
                body_rewrite_elapsed_seconds = max(
                    body_rewrite_elapsed_seconds,
                    float(body_rewrite_global.get("elapsed_seconds") or 0.0),
                )
            except (TypeError, ValueError):
                pass
        if chapter.get("omit_from_report") or chapter.get("chapter_omitted_no_evidence") or (
            str(chapter.get("internal_reason") or "") == "no_public_argument_or_table"
            and not as_list(chapter.get("sections"))
            and not as_list(chapter.get("table_packages"))
        ):
            if _is_core_chapter_for_health(chapter, layout):
                core_chapter_omitted_no_evidence_count += 1
            else:
                optional_chapter_omitted_count += 1
        raw_rendered_sections = [as_dict(section) for section in as_list(chapter.get("sections")) if isinstance(section, dict)]
        rendered_sections = [
            section
            for section in raw_rendered_sections
            if not section.get("omit_from_report")
            and not (
                section.get("observation_only")
                and not section.get("evidence_backed")
                and not section.get("force_render_observation")
            )
        ]
        skipped_observation_sections = [
            section
            for section in raw_rendered_sections
            if section.get("observation_only")
            and not section.get("evidence_backed")
            and not section.get("force_render_observation")
        ]
        omitted_observation_sections = [
            as_dict(section)
            for section in as_list(chapter.get("omitted_observation_sections"))
            if isinstance(section, dict)
        ]
        rendered_blocks = [
            str(section.get("block_type") or section.get("output_type") or "").strip()
            for section in rendered_sections
            if str(section.get("block_type") or section.get("output_type") or "").strip()
        ]
        rendered_block_count += len(rendered_blocks)
        rendered_must_block_count += sum(1 for block in rendered_blocks if block in planned_blocks)
        rendered_quality = []
        for section in rendered_sections:
            block_type = str(section.get("block_type") or section.get("output_type") or "").strip()
            if not block_type:
                continue
            section_title = str(section.get("section_title") or "").strip()
            if section_title:
                rendered_section_titles.append(section_title)
                if publisher_or_domain_re.search(section_title):
                    publisher_in_title_count += 1
            section_plan = as_dict(section.get("section_plan"))
            if section_plan.get("public_title") and not section_plan.get("omit_reason"):
                section_plan_valid_count += 1
                natural_title_count += 1
            if section_plan.get("omit_reason"):
                section_plan_dropped_count += 1
            if section_title in generic_titles:
                generic_title_filtered_count += 1
            if section.get("repeated_title_rewritten"):
                repeated_title_rewrite_count += 1
            if section.get("internal_role_title_filtered"):
                internal_role_title_filtered_count += 1
            if section.get("dynamic_section_title") or str(section.get("title_source") or "") == "dynamic":
                dynamic_section_title_count += 1
            if section.get("block_title_generation_failed"):
                block_title_generation_failed_count += 1
            refs = as_list(section.get("evidence_refs")) or as_list(section.get("required_evidence_refs"))
            facts = as_list(section.get("supporting_facts"))
            text = " ".join(
                str(section.get(key) or "")
                for key in ("claim", "reasoning", "mechanism", "counter_evidence")
            )
            if re.search(r"[（(]\s*[）)]", text):
                empty_parens_in_body_count += 1
            chinese_count = len(re.findall(r"[\u4e00-\u9fff]", text))
            latin_count = len(re.findall(r"[A-Za-z]", text))
            if len(text) > 80 and latin_count > 50 and chinese_count / max(1, chinese_count + latin_count) < 0.3:
                english_snippet_lead_count += 1
            if publisher_or_domain_re.search(text):
                publisher_in_subject_count += 1
            if traditional_re.search(text):
                traditional_chinese_normalized_count += 1
            if block_type == "metric_reconciliation" and re.search(r"^[^。；;\n]{2,40}[:：]\s*[^。；;\n]{1,80}", text):
                metric_sentence_rewritten_count += 1
            if section.get("snippet_lead_rewritten"):
                snippet_lead_rewritten_count += 1
            if section.get("snippet_lead_dropped"):
                snippet_lead_dropped_count += 1
                snippet_like_text_dropped_count += 1
            evidence_backed = bool(section.get("evidence_backed")) or bool(refs and facts)
            composition_status = str(section.get("body_composition_status") or section.get("composition_status") or "").strip()
            if composition_status in {"composed", "composed_directional"}:
                composed_section_count += 1
            elif evidence_backed:
                fact_passthrough_section_count += 1
            composer_variable_explanation_count += _coerce_int(section.get("composer_variable_explanation_count"))
            body_rewrite = as_dict(section.get("body_rewrite"))
            body_rewrite_status = str(section.get("body_rewrite_status") or body_rewrite.get("status") or "").strip()
            if (body_rewrite_status or body_rewrite) and not body_rewrite_global_seen:
                body_rewrite_enabled = True
                if body_rewrite.get("llm_called"):
                    body_rewrite_called_count += 1
                if body_rewrite.get("cache_hit") or body_rewrite_status == "cached":
                    body_rewrite_cache_hit_count += 1
                if body_rewrite_status == "rewritten":
                    body_rewrite_success_count += 1
                elif body_rewrite_status == "rejected":
                    body_rewrite_rejected_count += 1
                elif body_rewrite_status == "fallback":
                    body_rewrite_fallback_count += 1
                elif body_rewrite_status == "skipped":
                    body_rewrite_skipped_count += 1
                failure_reason = str(body_rewrite.get("failure_reason") or "").strip()
                if failure_reason:
                    body_rewrite_failure_reasons[failure_reason] = body_rewrite_failure_reasons.get(failure_reason, 0) + 1
            if any(
                phrase in text
                for phrase in (
                    "只能形成初步信号",
                    "暂不宜外推",
                    "低强度判断",
                    "更多独立来源复核",
                    "更多客户样本或反向案例",
                )
            ):
                weak_template_sentence_removed_count += 1
            evidence_backed = bool(section.get("evidence_backed")) or bool(refs and facts)
            observation_only = bool(section.get("observation_only")) or (
                bool(section.get("layout_generated")) and not evidence_backed
            ) or "只保留为观察项" in text
            quality_item = {
                "chapter_id": chapter_id,
                "section_id": section.get("section_id"),
                "block_type": block_type,
                "evidence_backed": evidence_backed,
                "observation_only": observation_only,
                "layout_generated": bool(section.get("layout_generated")),
                "fact_card_to_block_match": bool(section.get("fact_card_to_block_match")),
                "template_section_removed": bool(section.get("template_section_removed")),
            }
            rendered_quality.append(quality_item)
            if evidence_backed:
                evidence_backed_blocks.append(quality_item)
            if quality_item.get("fact_card_to_block_match"):
                fact_card_matched_blocks.append(quality_item)
            if quality_item.get("template_section_removed"):
                template_removed_blocks.append(quality_item)
            if observation_only:
                observation_only_blocks.append(quality_item)
        omitted_blocks = []
        for section in omitted_observation_sections + skipped_observation_sections:
            block_type = str(section.get("block_type") or section.get("output_type") or "").strip()
            item = {
                "chapter_id": chapter_id,
                "section_id": section.get("section_id"),
                "block_type": block_type,
                "reason": section.get("reason") or "observation_only_without_evidence",
            }
            omitted_blocks.append(block_type or str(section.get("section_id") or "unknown"))
            omitted_observation_blocks.append(item)
        missing = []
        rendered_remaining = list(rendered_blocks)
        for block in planned_blocks:
            if block in rendered_remaining:
                rendered_remaining.remove(block)
                continue
            missing.append(block)
            dropped_blocks.append({"chapter_id": chapter_id, "block_type": block, "reason": "layout_block_dropped"})
        for dropped in as_list(chapter.get("dropped_sections")):
            payload = as_dict(dropped)
            reason = str(payload.get("reason") or "")
            block_type = str(payload.get("block_type") or payload.get("output_type") or "").strip()
            if reason == "template_section_removed":
                template_removed_blocks.append({"chapter_id": chapter_id, "section_id": payload.get("section_id"), "block_type": block_type, "reason": reason})
            if reason in {"layout_section_without_public_evidence", "block_without_fact_card", "layout_block_dropped"} and not (
                reason == "layout_section_without_public_evidence" and block_type in {"boundary", "gap", ""}
            ):
                block_without_fact_card_blocks.append({"chapter_id": chapter_id, "section_id": payload.get("section_id"), "block_type": block_type, "reason": reason})
            if reason in {"block_title_generation_failed", "dropped_generic_block"}:
                dropped_generic_block_count += 1
                block_title_generation_failed_count += 1 if reason == "block_title_generation_failed" else 0
            if reason == "repeated_title_dropped":
                repeated_title_dropped_count += 1
            if reason == "internal_role_title_filtered":
                internal_role_title_filtered_count += 1
            if reason == "repeated_evidence_id_within_chapter":
                repeated_evidence_id_within_chapter_count += 1
            if reason == "repeated_fact_within_chapter":
                repeated_fact_demoted_count += 1
            if reason == "snippet_lead_dropped":
                snippet_lead_dropped_count += 1
                snippet_like_text_dropped_count += 1
        chapters.append(
            {
                "chapter_id": chapter_id,
                "planned_blocks": planned_blocks,
                "candidate_blocks": candidate_blocks,
                "rendered_blocks": rendered_blocks,
                "dropped_blocks": missing,
                "matched_by_llm_claim": claim_layout_diag.get("llm_claim_to_block_match_count", 0),
                "unmatched_llm_claims": claim_layout_diag.get("llm_claim_unmatched_count", 0),
                "must_block_matched_by_llm_claim": claim_layout_diag.get("must_block_matched_by_llm_claim_count", 0),
                "must_block_dropped_no_matching_claim": claim_layout_diag.get(
                    "must_block_dropped_no_matching_claim_count",
                    0,
                ),
                "rendered_section_count": len(rendered_sections),
                "effective_section_count": len([item for item in rendered_quality if item.get("evidence_backed")]),
                "observation_section_count": len([item for item in rendered_quality if item.get("observation_only")]),
                "omitted_observation_blocks": omitted_blocks,
                "chapter_effective_section_ratio": round(
                    len([item for item in rendered_quality if item.get("evidence_backed")])
                    / max(1, len(rendered_quality)),
                    3,
                ),
                "rendered_quality": rendered_quality,
                "observation_only_blocks": [
                    item.get("block_type")
                    for item in rendered_quality
                    if item.get("observation_only")
                ],
                "evidence_backed_blocks": [
                    item.get("block_type")
                    for item in rendered_quality
                    if item.get("evidence_backed")
                ],
            }
        )
    for chapter in chapter_packages:
        if not isinstance(chapter, dict):
            continue
        chapter_id = str(chapter.get("chapter_id") or "").strip()
        if chapter_id in scored_chapter_ids:
            continue
        raw_rendered_sections = [as_dict(section) for section in as_list(chapter.get("sections")) if isinstance(section, dict)]
        rendered_sections = [
            section
            for section in raw_rendered_sections
            if not section.get("omit_from_report")
            and not (
                section.get("observation_only")
                and not section.get("evidence_backed")
                and not section.get("force_render_observation")
            )
        ]
        rendered_quality = []
        rendered_blocks = []
        for section in rendered_sections:
            block_type = str(section.get("block_type") or section.get("output_type") or "").strip()
            if not block_type:
                block_type = str(section.get("section_title") or "section").strip()
            section_title = str(section.get("section_title") or "").strip()
            if section_title:
                rendered_section_titles.append(section_title)
                if publisher_or_domain_re.search(section_title):
                    publisher_in_title_count += 1
            section_plan = as_dict(section.get("section_plan"))
            if section_plan.get("public_title") and not section_plan.get("omit_reason"):
                section_plan_valid_count += 1
                natural_title_count += 1
            if section_plan.get("omit_reason"):
                section_plan_dropped_count += 1
            if section_title in generic_titles:
                generic_title_filtered_count += 1
            if section.get("repeated_title_rewritten"):
                repeated_title_rewrite_count += 1
            if section.get("internal_role_title_filtered"):
                internal_role_title_filtered_count += 1
            if section.get("dynamic_section_title") or str(section.get("title_source") or "") == "dynamic":
                dynamic_section_title_count += 1
            if section.get("block_title_generation_failed"):
                block_title_generation_failed_count += 1
            refs = as_list(section.get("evidence_refs")) or as_list(section.get("required_evidence_refs"))
            facts = as_list(section.get("supporting_facts"))
            evidence_backed = bool(section.get("evidence_backed")) or bool(refs and facts)
            text = " ".join(
                str(section.get(key) or "")
                for key in ("claim", "reasoning", "mechanism", "counter_evidence")
            )
            if re.search(r"[（(]\s*[）)]", text):
                empty_parens_in_body_count += 1
            chinese_count = len(re.findall(r"[\u4e00-\u9fff]", text))
            latin_count = len(re.findall(r"[A-Za-z]", text))
            if len(text) > 80 and latin_count > 50 and chinese_count / max(1, chinese_count + latin_count) < 0.3:
                english_snippet_lead_count += 1
            if publisher_or_domain_re.search(text):
                publisher_in_subject_count += 1
            if traditional_re.search(text):
                traditional_chinese_normalized_count += 1
            if block_type == "metric_reconciliation" and re.search(r"^[^。；;\n]{2,40}[:：]\s*[^。；;\n]{1,80}", text):
                metric_sentence_rewritten_count += 1
            if section.get("snippet_lead_rewritten"):
                snippet_lead_rewritten_count += 1
            if section.get("snippet_lead_dropped"):
                snippet_lead_dropped_count += 1
                snippet_like_text_dropped_count += 1
            composition_status = str(section.get("body_composition_status") or section.get("composition_status") or "").strip()
            if composition_status in {"composed", "composed_directional"}:
                composed_section_count += 1
            elif evidence_backed:
                fact_passthrough_section_count += 1
            composer_variable_explanation_count += _coerce_int(section.get("composer_variable_explanation_count"))
            body_rewrite = as_dict(section.get("body_rewrite"))
            body_rewrite_status = str(section.get("body_rewrite_status") or body_rewrite.get("status") or "").strip()
            if (body_rewrite_status or body_rewrite) and not body_rewrite_global_seen:
                body_rewrite_enabled = True
                if body_rewrite.get("llm_called"):
                    body_rewrite_called_count += 1
                if body_rewrite.get("cache_hit") or body_rewrite_status == "cached":
                    body_rewrite_cache_hit_count += 1
                if body_rewrite_status == "rewritten":
                    body_rewrite_success_count += 1
                elif body_rewrite_status == "rejected":
                    body_rewrite_rejected_count += 1
                elif body_rewrite_status == "fallback":
                    body_rewrite_fallback_count += 1
                elif body_rewrite_status == "skipped":
                    body_rewrite_skipped_count += 1
                failure_reason = str(body_rewrite.get("failure_reason") or "").strip()
                if failure_reason:
                    body_rewrite_failure_reasons[failure_reason] = body_rewrite_failure_reasons.get(failure_reason, 0) + 1
            if any(
                phrase in text
                for phrase in (
                    "只能形成初步信号",
                    "暂不宜外推",
                    "低强度判断",
                    "更多独立来源复核",
                    "更多客户样本或反向案例",
                )
            ):
                weak_template_sentence_removed_count += 1
            observation_only = bool(section.get("observation_only")) or (
                bool(section.get("layout_generated")) and not evidence_backed
            )
            quality_item = {
                "chapter_id": chapter_id,
                "section_id": section.get("section_id"),
                "block_type": block_type,
                "evidence_backed": evidence_backed,
                "observation_only": observation_only,
                "layout_generated": bool(section.get("layout_generated")),
                "fact_card_to_block_match": bool(section.get("fact_card_to_block_match")),
                "template_section_removed": bool(section.get("template_section_removed")),
            }
            rendered_quality.append(quality_item)
            rendered_blocks.append(block_type)
            if evidence_backed:
                evidence_backed_blocks.append(quality_item)
            if quality_item.get("fact_card_to_block_match"):
                fact_card_matched_blocks.append(quality_item)
            if quality_item.get("template_section_removed"):
                template_removed_blocks.append(quality_item)
            if observation_only:
                observation_only_blocks.append(quality_item)
        for dropped in as_list(chapter.get("dropped_sections")):
            payload = as_dict(dropped)
            reason = str(payload.get("reason") or "")
            block_type = str(payload.get("block_type") or payload.get("output_type") or "").strip()
            if reason == "template_section_removed":
                template_removed_blocks.append({"chapter_id": chapter_id, "section_id": payload.get("section_id"), "block_type": block_type, "reason": reason})
            if reason in {"layout_section_without_public_evidence", "block_without_fact_card", "layout_block_dropped"} and not (
                reason == "layout_section_without_public_evidence" and block_type in {"boundary", "gap", ""}
            ):
                block_without_fact_card_blocks.append({"chapter_id": chapter_id, "section_id": payload.get("section_id"), "block_type": block_type, "reason": reason})
            if reason in {"block_title_generation_failed", "dropped_generic_block"}:
                dropped_generic_block_count += 1
                block_title_generation_failed_count += 1 if reason == "block_title_generation_failed" else 0
            if reason == "repeated_title_dropped":
                repeated_title_dropped_count += 1
            if reason == "internal_role_title_filtered":
                internal_role_title_filtered_count += 1
            if reason == "repeated_evidence_id_within_chapter":
                repeated_evidence_id_within_chapter_count += 1
            if reason == "repeated_fact_within_chapter":
                repeated_fact_demoted_count += 1
            if reason == "snippet_lead_dropped":
                snippet_lead_dropped_count += 1
                snippet_like_text_dropped_count += 1
        rendered_block_count += len(rendered_blocks)
        omitted_blocks = [
            str(section.get("block_type") or section.get("section_id") or "unknown")
            for section in as_list(chapter.get("omitted_observation_sections"))
            if isinstance(section, dict)
        ]
        chapters.append(
            {
                "chapter_id": chapter_id,
                "planned_blocks": [],
                "rendered_blocks": rendered_blocks,
                "dropped_blocks": [],
                "rendered_section_count": len(rendered_sections),
                "effective_section_count": len([item for item in rendered_quality if item.get("evidence_backed")]),
                "observation_section_count": len([item for item in rendered_quality if item.get("observation_only")]),
                "omitted_observation_blocks": omitted_blocks,
                "chapter_effective_section_ratio": round(
                    len([item for item in rendered_quality if item.get("evidence_backed")])
                    / max(1, len(rendered_quality)),
                    3,
                ),
                "rendered_quality": rendered_quality,
                "observation_only_blocks": [
                    item.get("block_type")
                    for item in rendered_quality
                    if item.get("observation_only")
                ],
                "evidence_backed_blocks": [
                    item.get("block_type")
                    for item in rendered_quality
                    if item.get("evidence_backed")
                ],
            }
        )
    skipped_tables = []
    invalid_table_candidates = []
    diagnostic_table_re = re.compile(r"后续影响|使用边界|该指标须|须同时披露|不会凭空补齐|进入正文判断|缺口数据|后续验证项", re.I)
    for table in table_packages:
        if not isinstance(table, dict):
            continue
        table_text = "\n".join(
            str(value or "")
            for value in [
                table.get("title"),
                table.get("takeaway"),
                table.get("decision_implication"),
                *as_list(table.get("limitations")),
                *as_list(table.get("headers")),
                *[
                    cell
                    for row in as_list(table.get("rows"))
                    for cell in as_list(as_dict(row).get("cells"))
                ],
            ]
        )
        if table.get("should_render") and diagnostic_table_re.search(table_text):
            invalid_table_candidates.append(
                {
                    "chapter_id": table.get("chapter_id"),
                    "table_id": table.get("table_id") or table.get("id"),
                    "anchor_block_type": table.get("anchor_block_type"),
                    "reason": "diagnostic_table_language",
                }
            )
        if table.get("should_render"):
            continue
        skipped_tables.append(
            {
                "chapter_id": table.get("chapter_id"),
                "table_id": table.get("table_id") or table.get("id"),
                "anchor_block_type": table.get("anchor_block_type"),
                "reason": ", ".join(str(item) for item in as_list(table.get("reject_reasons"))[:4])
                or str(table.get("validation_error") or table.get("status") or "not_rendered"),
            }
        )
    title_counter = Counter(title for title in rendered_section_titles if title)
    repeated_titles = {
        title: count
        for title, count in title_counter.items()
        if count > 1
    }
    repeated_section_title_count = sum(count - 1 for count in repeated_titles.values())
    generic_section_title_count = sum(1 for title in rendered_section_titles if title in generic_titles)
    title_repetition_findings = []
    if repeated_section_title_count > 2 or any(count > 2 for count in repeated_titles.values()):
        title_repetition_findings.append(
            {
                "issue_type": "template_section_title_repetition",
                "repeated_titles": repeated_titles,
            }
        )
    return {
        "micro_layout_count": len([item for item in micro_layouts if isinstance(item, dict)]),
        "chapter_package_count": len([item for item in chapter_packages if isinstance(item, dict)]),
        "rendered_block_count": rendered_block_count,
        "must_render_block_count": must_render_block_count,
        "candidate_block_count": candidate_block_count,
        "rendered_must_block_count": rendered_must_block_count,
        "llm_claim_to_block_match_count": llm_claim_to_block_match_count,
        "llm_claim_unmatched_count": llm_claim_unmatched_count,
        "must_block_matched_by_llm_claim_count": must_block_matched_by_llm_claim_count,
        "must_block_dropped_no_matching_claim_count": must_block_dropped_no_matching_claim_count,
        "dropped_block_count": len(dropped_blocks),
        "observation_only_block_count": len(observation_only_blocks),
        "evidence_backed_block_count": len(evidence_backed_blocks),
        "effective_section_count": len(evidence_backed_blocks),
        "omitted_observation_section_count": len(omitted_observation_blocks),
        "template_section_removed_count": len(template_removed_blocks),
        "block_without_fact_card_count": len(block_without_fact_card_blocks),
        "fact_card_to_block_match_count": len(fact_card_matched_blocks),
        "dropped_due_to_template_risk_count": len(template_removed_blocks),
        "generic_section_title_count": generic_section_title_count,
        "repeated_section_title_count": repeated_section_title_count,
        "dynamic_section_title_count": dynamic_section_title_count,
        "natural_title_count": natural_title_count,
        "generic_title_filtered_count": generic_title_filtered_count,
        "repeated_title_rewrite_count": repeated_title_rewrite_count,
        "repeated_title_dropped_count": repeated_title_dropped_count,
        "internal_role_title_filtered_count": internal_role_title_filtered_count,
        "weak_template_sentence_removed_count": weak_template_sentence_removed_count,
        "publisher_in_title_count": publisher_in_title_count,
        "publisher_in_subject_count": publisher_in_subject_count,
        "english_snippet_lead_count": english_snippet_lead_count,
        "empty_parens_in_body_count": (
            _coerce_int(naturalness_cleanup.get("residual_empty_parens_count"))
            if "residual_empty_parens_count" in naturalness_cleanup
            else max(
                0,
                empty_parens_in_body_count - _coerce_int(naturalness_cleanup.get("empty_parens_removed_count")),
            )
        ),
        "snippet_lead_rewritten_count": snippet_lead_rewritten_count,
        "snippet_lead_dropped_count": snippet_lead_dropped_count,
        "repeated_evidence_id_within_chapter_count": repeated_evidence_id_within_chapter_count,
        "repeated_fact_demoted_count": repeated_fact_demoted_count,
        "snippet_like_text_dropped_count": snippet_like_text_dropped_count
        + _coerce_int(naturalness_cleanup.get("residual_headline_dropped_count")),
        "metric_sentence_rewritten_count": metric_sentence_rewritten_count
        + _coerce_int(naturalness_cleanup.get("metric_sentence_rewritten_count")),
        "traditional_chinese_normalized_count": traditional_chinese_normalized_count
        + _coerce_int(naturalness_cleanup.get("traditional_chinese_normalized_count")),
        "ocr_artifact_normalized_count": _coerce_int(naturalness_cleanup.get("ocr_artifact_normalized_count")),
        "empty_parens_removed_count": _coerce_int(naturalness_cleanup.get("empty_parens_removed_count")),
        "truncated_punctuation_cleaned_count": _coerce_int(naturalness_cleanup.get("truncated_punctuation_cleaned_count")),
        "chapter_omitted_no_evidence_count": core_chapter_omitted_no_evidence_count,
        "core_chapter_omitted_no_evidence_count": core_chapter_omitted_no_evidence_count,
        "optional_chapter_omitted_count": optional_chapter_omitted_count,
        "block_drop_reason_examples": (block_without_fact_card_blocks or dropped_blocks)[:12],
        "composer_variable_explanation_count": composer_variable_explanation_count,
        "body_rewrite": {
            "enabled": body_rewrite_enabled,
            "called_count": body_rewrite_called_count,
            "submitted_count": body_rewrite_submitted_count,
            "success_count": body_rewrite_success_count,
            "cache_hit_count": body_rewrite_cache_hit_count,
            "rejected_count": body_rewrite_rejected_count,
            "fallback_count": body_rewrite_fallback_count,
            "skipped_count": body_rewrite_skipped_count,
            "budget_exhausted_count": body_rewrite_budget_exhausted_count,
            "inflight_dedup_count": body_rewrite_inflight_dedup_count,
            "elapsed_seconds": round(body_rewrite_elapsed_seconds, 3),
            "concurrency": body_rewrite_concurrency,
            "failure_reasons": body_rewrite_failure_reasons,
        },
        "chapter_narrative": chapter_narrative_diag,
        "body_composition_status": "fact_passthrough"
        if fact_passthrough_section_count and not composed_section_count
        else ("mixed" if fact_passthrough_section_count and composed_section_count else ("composed" if composed_section_count else "unknown")),
        "composed_section_count": composed_section_count,
        "fact_passthrough_section_count": fact_passthrough_section_count,
        "section_plan_valid_count": section_plan_valid_count,
        "section_plan_dropped_count": section_plan_dropped_count,
        "block_title_generation_failed_count": block_title_generation_failed_count,
        "dropped_generic_block_count": dropped_generic_block_count,
        "repeated_section_titles": repeated_titles,
        "title_repetition_findings": title_repetition_findings,
        "chapters": chapters,
        "dropped_blocks": dropped_blocks,
        "observation_only_blocks": observation_only_blocks,
        "evidence_backed_blocks": evidence_backed_blocks,
        "omitted_observation_blocks": omitted_observation_blocks,
        "template_removed_blocks": template_removed_blocks,
        "block_without_fact_card_blocks": block_without_fact_card_blocks,
        "skipped_tables": skipped_tables,
        "filtered_table_count": len(skipped_tables) + len(invalid_table_candidates),
        "invalid_table_count": len(invalid_table_candidates),
        "invalid_table_candidates": invalid_table_candidates,
        "compacted_artifact_used": compacted_artifact_used,
    }


def _chapter_evidence_input_diagnostics(writer_package: Dict[str, Any]) -> Dict[str, Any]:
    writer_report = as_dict(writer_package.get("writer_report"))
    render_artifacts = as_dict(writer_report.get("render_artifacts"))
    evidence_package = as_dict(render_artifacts.get("evidence_package")) or as_dict(writer_package.get("evidence_package"))
    chapters = (
        as_list(render_artifacts.get("chapter_evidence_packages"))
        or as_list(evidence_package.get("chapter_evidence_packages"))
        or as_list(writer_report.get("chapter_evidence_packages"))
        or as_list(writer_package.get("chapter_evidence_packages"))
    )
    compacted_artifact_used = not bool(as_list(render_artifacts.get("chapter_evidence_packages")))
    health = as_dict(writer_package.get("evidence_health_summary")) or as_dict(
        evidence_package.get("evidence_health_summary")
    )

    source_pool_count = 0
    for key in ("analysis_ready_evidence", "clean_evidence_list", "normalized_evidence", "raw_data_points"):
        value = evidence_package.get(key)
        if isinstance(value, dict) and isinstance(value.get("sample"), list):
            source_pool_count += len(value.get("sample") or [])
        elif isinstance(value, list):
            source_pool_count += len(value)
    if not source_pool_count:
        for key in ("analysis_ready_count", "clean_fact_count", "normalized_evidence_count", "raw_data_point_count"):
            try:
                source_pool_count = max(source_pool_count, int(float(health.get(key) or 0)))
            except (TypeError, ValueError):
                continue

    items: List[Dict[str, Any]] = []
    binding_failed = False
    total_unresolved_refs = 0

    def actual_layer_count(chapter: Dict[str, Any], key: str) -> int:
        return len(as_list(chapter.get(key)))

    def layer_count(chapter: Dict[str, Any], key: str) -> int:
        count_key = f"{key}_count"
        try:
            explicit = int(float(chapter.get(count_key) or 0))
        except (TypeError, ValueError):
            explicit = 0
        evidence_counts = as_dict(chapter.get("evidence_counts"))
        try:
            counted = int(float(evidence_counts.get(key) or 0))
        except (TypeError, ValueError):
            counted = 0
        return max(explicit, counted, len(as_list(chapter.get(key))))

    for chapter in chapters:
        if not isinstance(chapter, dict):
            continue
        core_count = layer_count(chapter, "core_evidence")
        supporting_count = layer_count(chapter, "supporting_evidence")
        metric_count = layer_count(chapter, "metric_evidence")
        counter_count = layer_count(chapter, "counter_evidence")
        case_count = layer_count(chapter, "case_evidence")
        directional_count = layer_count(chapter, "directional_evidence")
        actual_core_count = actual_layer_count(chapter, "core_evidence")
        actual_supporting_count = actual_layer_count(chapter, "supporting_evidence")
        actual_metric_count = actual_layer_count(chapter, "metric_evidence")
        actual_counter_count = actual_layer_count(chapter, "counter_evidence")
        actual_case_count = actual_layer_count(chapter, "case_evidence")
        actual_directional_count = actual_layer_count(chapter, "directional_evidence")
        hydrated_layer_count = (
            actual_core_count
            + actual_supporting_count
            + actual_metric_count
            + actual_counter_count
            + actual_case_count
            + actual_directional_count
        )
        sample_count = layer_count(chapter, "sample_evidence")
        unresolved_refs = as_list(chapter.get("unresolved_evidence_refs"))
        unresolved_count = int(chapter.get("unresolved_evidence_ref_count") or len(unresolved_refs) or 0)
        public_filter = as_dict(chapter.get("public_fact_filter_summary")) or as_dict(as_dict(chapter.get("metadata")).get("public_fact_filter_summary"))
        chapter_analysis = as_dict(chapter.get("chapter_analysis"))
        writable_fact_count = chapter.get("writable_fact_count")
        if writable_fact_count is None:
            writable_fact_count = public_filter.get("eligible_fact_count")
        citation_fact_count = chapter.get("eligible_citation_count")
        if citation_fact_count is None:
            citation_fact_count = public_filter.get("eligible_citation_count")
        total_unresolved_refs += unresolved_count
        has_bound_evidence = bool(hydrated_layer_count)
        if source_pool_count and not has_bound_evidence:
            binding_failed = True
        items.append(
            {
                "chapter_id": chapter.get("chapter_id") or chapter.get("id"),
                "chapter_title": chapter.get("chapter_title") or chapter.get("title"),
                "core_evidence_count": core_count,
                "supporting_evidence_count": supporting_count,
                "metric_evidence_count": metric_count,
                "counter_evidence_count": counter_count,
                "case_evidence_count": case_count,
                "directional_evidence_count": directional_count,
                "sample_evidence_count": sample_count,
                "hydrated_layer_item_count": hydrated_layer_count,
                "count_only_warning": bool(
                    not hydrated_layer_count
                    and (core_count or supporting_count or metric_count or counter_count or case_count or directional_count)
                ),
                "sample_only_warning": bool(sample_count and not hydrated_layer_count),
                "unresolved_evidence_ref_count": unresolved_count,
                "candidate_fact_count": public_filter.get("candidate_fact_count"),
                "eligible_fact_count": writable_fact_count,
                "fact_card_count": chapter.get("fact_card_count") or public_filter.get("fact_card_count") or chapter_analysis.get("fact_card_count"),
                "chapter_analysis_valid": bool(chapter.get("chapter_analysis_valid") or chapter_analysis.get("chapter_analysis_valid")),
                "directional_fact_card_count": chapter_analysis.get("directional_fact_card_count"),
                "strong_fact_card_count": chapter_analysis.get("strong_fact_card_count"),
                "filtered_fact_count": public_filter.get("filtered_fact_count"),
                "invalid_metric_filtered_count": public_filter.get("invalid_metric_filtered_count"),
                "eligible_citation_count": citation_fact_count,
            }
        )
    return {
        "chapter_count": len(items),
        "source_pool_count": source_pool_count,
        "chapter_evidence_binding_failed": binding_failed,
        "total_unresolved_evidence_ref_count": total_unresolved_refs,
        "compacted_artifact_used": compacted_artifact_used,
        "chapters": items,
    }


def _evidence_grade_usage_diagnostics(writer_package: Dict[str, Any]) -> Dict[str, Any]:
    writer_report = as_dict(writer_package.get("writer_report"))
    render_artifacts = as_dict(writer_report.get("render_artifacts"))
    evidence_package = as_dict(render_artifacts.get("evidence_package")) or as_dict(writer_package.get("evidence_package"))
    chapters = (
        as_list(render_artifacts.get("chapter_evidence_packages"))
        or as_list(evidence_package.get("chapter_evidence_packages"))
        or as_list(writer_report.get("chapter_evidence_packages"))
        or as_list(writer_package.get("chapter_evidence_packages"))
    )
    grade_counts: Dict[str, int] = {}
    degraded_sections = 0
    directional_items = 0
    mismatch_filtered = 0
    low_quality_filtered = 0
    for chapter in chapters:
        if not isinstance(chapter, dict):
            continue
        chapter_has_directional = False
        for collection in (
            "core_evidence",
            "supporting_evidence",
            "metric_evidence",
            "case_evidence",
            "counter_evidence",
            "directional_evidence",
            "sample_evidence",
        ):
            for item in as_list(chapter.get(collection)):
                payload = as_dict(item)
                if not payload:
                    continue
                level = str(payload.get("source_level") or payload.get("credibility") or "UNKNOWN").strip().upper() or "UNKNOWN"
                grade_counts[level] = grade_counts.get(level, 0) + 1
                allowed_use = str(payload.get("allowed_use") or "").strip().lower()
                if level in {"B", "C"} and (collection in {"directional_evidence", "case_evidence"} or allowed_use == "directional_signal"):
                    directional_items += 1
                    chapter_has_directional = True
                if payload.get("source_title_url_mismatch_suspected"):
                    mismatch_filtered += 1
                if level == "D" or str(payload.get("source_type") or "").strip().lower() in {"self_media", "social", "forum", "wiki", "seo", "search_page", "aggregator"}:
                    low_quality_filtered += 1
        if chapter_has_directional:
            degraded_sections += 1

    schedule = as_dict(writer_report.get("search_task_schedule")) or as_dict(writer_package.get("search_task_schedule"))
    lane_coverage = as_dict(writer_report.get("lane_coverage"))
    lanes = list(lane_coverage.values()) if isinstance(lane_coverage, dict) else as_list(lane_coverage)
    throttled_or_failed = 0
    timed_out = 0
    dropped = 0
    partial_lanes = 0
    for lane in lanes:
        payload = as_dict(lane)
        if not payload:
            continue
        status = str(payload.get("status") or payload.get("execution_status") or "").strip().lower()
        if status in {"partial", "failed", "timed_out", "timeout"}:
            partial_lanes += 1
        try:
            throttled_or_failed += int(float(payload.get("failed_task_count") or payload.get("failed") or 0))
            timed_out += int(float(payload.get("timed_out_task_count") or 0))
            dropped += int(float(payload.get("dropped_task_count") or 0))
        except (TypeError, ValueError):
            continue
    return {
        "source_level_usage": grade_counts,
        "bc_directional_evidence_item_count": directional_items,
        "bc_degraded_chapter_count": degraded_sections,
        "source_mismatch_filtered_count": mismatch_filtered,
        "low_quality_filtered_count": low_quality_filtered,
        "retrieval_scheduled_count": schedule.get("scheduled_count"),
        "retrieval_dropped_count": schedule.get("dropped_count"),
        "retrieval_dropped_blocking_count": schedule.get("dropped_blocking_count"),
        "retrieval_partial_lane_count": partial_lanes,
        "retrieval_failed_task_count": throttled_or_failed,
        "retrieval_timed_out_task_count": timed_out,
        "retrieval_lane_dropped_task_count": dropped,
    }


def _score_safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return default


def _first_score_dict(*values: Any) -> Dict[str, Any]:
    for value in values:
        payload = as_dict(value)
        if payload:
            return payload
    return {}


def _directed_repair_diagnostics(writer_package: Dict[str, Any], writer_report: Dict[str, Any]) -> Dict[str, Any]:
    raw_output = as_dict(writer_package.get("raw_output"))
    raw_metadata = as_dict(raw_output.get("metadata")) or as_dict(writer_package.get("metadata"))
    metadata = as_dict(writer_package.get("metadata"))
    render_artifacts = as_dict(writer_report.get("render_artifacts")) or as_dict(as_dict(writer_package.get("writer_report")).get("render_artifacts"))
    selected = _first_score_dict(
        raw_metadata.get("repair_task_selection_summary"),
        raw_output.get("repair_task_selection_summary"),
        metadata.get("repair_task_selection_summary"),
        writer_package.get("repair_task_selection_summary"),
        render_artifacts.get("repair_task_selection_summary"),
    )
    planned: Dict[str, Any] = {}
    if not selected and not planned:
        return {}
    selected_by_reason = as_dict(selected.get("by_proof_role")) or as_dict(selected.get("by_reason"))
    planned_by_reason = (
        as_dict(planned.get("last_planned_by_proof_role"))
        or as_dict(planned.get("planned_by_proof_role"))
        or as_dict(planned.get("by_proof_role"))
    )
    selected_count = _score_safe_int(selected.get("task_count")) or sum(_score_safe_int(value) for value in selected_by_reason.values())
    repair_count = (
        _score_safe_int(planned.get("gap_repair_task_count"))
        or _score_safe_int(planned.get("planned_count"))
        or _score_safe_int(selected.get("post_policy_task_count"))
        or selected_count
    )
    skip_reason = str(planned.get("last_skip_reason") or selected.get("last_skip_reason") or "").strip()
    budget_exhausted = bool(
        planned.get("budget_exhausted")
        or selected.get("budget_exhausted")
        or "budget" in skip_reason.lower()
        or _score_safe_int(selected.get("deep_budget_exhausted_count"))
    )
    return {
        "selected_repair_task_count": selected_count,
        "repair_task_count": repair_count,
        "selected_repair_task_count_by_reason": selected_by_reason,
        "repair_task_count_by_reason": planned_by_reason or selected_by_reason,
        "repair_budget_exhausted": budget_exhausted,
        "last_skip_reason": skip_reason,
        "deep_budget_exhausted_count": selected.get("deep_budget_exhausted_count") or planned.get("deep_budget_exhausted_count") or 0,
    }


def render_score_markdown(
    *,
    query: str,
    writer_report: Dict[str, Any],
    writer_package: Dict[str, Any],
    final_audit_result: Dict[str, Any],
    reformatter_result: Dict[str, Any],
) -> str:
    score = _score_from_writer_report(writer_report)
    grade = writer_report.get("quality_grade") or _grade_from_score(score)
    clean_output_enabled = bool(writer_report.get("clean_output_enabled"))
    if "clean_output_enabled" not in writer_report:
        clean_output_enabled = os.getenv("REPORT_WRITE_CLEAN_REPORT", "false").strip().lower() in {"1", "true", "yes", "on"}
    clean_content_eligible = bool(
        writer_report.get("clean_content_eligible")
        if "clean_content_eligible" in writer_report
        else writer_report.get("writer_clean_report_eligible") or writer_report.get("clean_report_eligible")
    )
    if bool(final_audit_result.get("blocked")):
        clean_content_eligible = False
    clean_report_written = bool(writer_report.get("clean_report_written") or reformatter_result.get("output_written"))
    clean_eligible = bool(clean_content_eligible and clean_output_enabled and not bool(final_audit_result.get("blocked")))
    if bool(reformatter_result.get("enabled")) and str(reformatter_result.get("status") or "").strip() not in {"completed", "skipped"}:
        clean_eligible = False
    scorecard = str(writer_report.get("score_markdown") or "").strip()
    if scorecard and (
        scorecard.lstrip().startswith("# ")
        or "报告评分与审查" in scorecard
        or "章节证据输入摘要" in scorecard
    ):
        scorecard = ""
    post_qa = as_dict(writer_report.get("post_qa_repair"))
    post_qa_status = str(post_qa.get("status") or "not_recorded")
    if str(os.getenv("BRAIN_ENABLE_POST_QA_REPAIR", "false") or "").strip().lower() not in {"1", "true", "yes", "on"}:
        post_qa_status = "disabled_by_default"
    analysis_contract = _analysis_contract_diagnostics(writer_package)
    report_execution_mode = (
        writer_report.get("report_execution_mode")
        or writer_package.get("report_execution_mode")
        or as_dict(writer_package.get("stage_snapshot_replay")).get("execution_mode")
        or ("timeout_fail_open" if as_dict(writer_report.get("live_timeout")) or as_dict(writer_package.get("live_timeout")) else "")
        or ("live_quality_full" if str(os.getenv("REPORT_QUALITY_MODE") or "").strip().lower() == "high" else "standard")
    )
    quality_mode = bool(
        writer_report.get("quality_mode")
        or writer_package.get("quality_mode")
        or str(report_execution_mode) in {"quality_llm_replay", "live_quality_full"}
        or str(os.getenv("REPORT_QUALITY_MODE") or "").strip().lower() == "high"
    )
    quality_posture = as_dict(writer_report.get("quality_posture")) or as_dict(writer_package.get("quality_posture"))
    lines = [
        f"# {query} - 报告评分与审查",
        "",
        f"- 质量总分：{score}/100",
        f"- 质量等级：{grade}",
        f"- Clean 资格：{'是' if clean_eligible else '否'}",
        f"- Clean 内容资格：{'是' if clean_content_eligible else '否'}",
        f"- Clean 标准：{writer_report.get('clean_standard') or os.getenv('REPORT_CLEAN_STANDARD', 'balanced')}",
        f"- Clean 输出开关：{'开启' if clean_output_enabled else '关闭'}",
        f"- Clean 文件已写出：{'是' if clean_report_written else '否'}",
        f"- 报告状态：{writer_report.get('report_status') or 'unknown'}",
        f"- 交付层级：{writer_report.get('delivery_tier') or 'unknown'}",
        f"- Post-QA 补证：{post_qa_status}",
    ]
    lines.extend(
        [
            "",
            "## Execution / Quality Path",
            "",
            f"- report_execution_mode: {report_execution_mode}",
            f"- quality_mode: {quality_mode}",
            f"- quality_posture_mode: {quality_posture.get('mode') or ''}",
            f"- query_rewrite_disabled: {as_dict(quality_posture.get('disabled')).get('query_rewrite')}",
            f"- self_refine_disabled: {as_dict(quality_posture.get('disabled')).get('self_refine')}",
            f"- query_rewrite_max_calls: {quality_posture.get('query_rewrite_max_calls') or os.getenv('QUERY_REWRITE_MAX_CALLS_PER_REPORT', '')}",
            f"- query_rewrite_max_input_chars: {quality_posture.get('query_rewrite_max_input_chars') or os.getenv('QUERY_REWRITE_MAX_INPUT_CHARS', '')}",
            f"- llm_analysis_attempted: {analysis_contract.get('llm_analysis_attempted')}",
            f"- quality_path_degraded: {analysis_contract.get('quality_path_degraded')}",
            f"- quality_path_degradation_reason: {analysis_contract.get('quality_path_degradation_reason')}",
        ]
    )
    repair_diag = _directed_repair_diagnostics(writer_package, writer_report)
    if repair_diag:
        lines.extend(
            [
                "",
                "## Directed Evidence Repair",
                "",
                f"- selected_repair_task_count: {repair_diag.get('selected_repair_task_count')}",
                f"- selected_repair_task_count_by_reason: {repair_diag.get('selected_repair_task_count_by_reason')}",
                f"- repair_task_count: {repair_diag.get('repair_task_count')}",
                f"- repair_task_count_by_reason: {repair_diag.get('repair_task_count_by_reason')}",
                f"- repair_budget_exhausted: {repair_diag.get('repair_budget_exhausted')}",
                f"- repair_last_skip_reason: {repair_diag.get('last_skip_reason')}",
                f"- repair_deep_budget_exhausted_count: {repair_diag.get('deep_budget_exhausted_count')}",
            ]
        )
    live_timeout = as_dict(writer_report.get("live_timeout")) or as_dict(writer_package.get("live_timeout"))
    if live_timeout:
        lines.extend(
            [
                "",
                "## Live Timeout / Fail-Open",
                "",
                f"- live_deadline_seconds: {live_timeout.get('live_deadline_seconds')}",
                f"- timeout_triggered: {live_timeout.get('timeout_triggered')}",
                f"- timeout_stage: {live_timeout.get('timeout_stage')}",
                f"- fail_open_path_used: {live_timeout.get('fail_open_path_used')}",
                f"- partial_artifact_used: {live_timeout.get('partial_artifact_used')}",
                f"- fail_open_error: {live_timeout.get('fail_open_error') or ''}",
            ]
        )
    snapshot_index = [item for item in as_list(writer_package.get("stage_snapshot_index")) if isinstance(item, dict)]
    if snapshot_index:
        stored = [item for item in snapshot_index if item.get("stored")]
        replayable = [item for item in stored if item.get("replayable")]
        unreplayable = [item for item in stored if not item.get("replayable")]
        status = "ok" if stored and not unreplayable else ("warning" if stored else "missing")
        lines.extend(
            [
                "",
                "## Stage Snapshot Replayability",
                "",
                f"- stage_snapshot_replayability_status: {status}",
                f"- stored_stage_count: {len(stored)}",
                f"- replayable_stage_count: {len(replayable)}",
                f"- replayable_stages: {[str(item.get('stage_name') or '') for item in replayable]}",
                f"- unreplayable_stages: {[{'stage_name': str(item.get('stage_name') or ''), 'reason': str(item.get('reason') or '')} for item in unreplayable[:8]]}",
            ]
        )
    analysis_contract = _analysis_contract_diagnostics(writer_package)
    lines.extend(
        [
            "",
            "## Analysis/Render Contract",
            "",
            f"- structured_analysis_valid: {analysis_contract.get('structured_analysis_valid')}",
            f"- analysis_rebuilt_from_evidence: {analysis_contract.get('analysis_rebuilt_from_evidence')}",
            f"- uses_llm_analysis: {analysis_contract.get('uses_llm_analysis')}",
            f"- llm_analysis_attempted: {analysis_contract.get('llm_analysis_attempted')}",
            f"- llm_analysis_status: {analysis_contract.get('llm_analysis_status')}",
            f"- final_analysis_source: {analysis_contract.get('final_analysis_source')}",
            f"- deterministic_synthesis_used: {analysis_contract.get('deterministic_synthesis_used')}",
            f"- quality_path_degraded: {analysis_contract.get('quality_path_degraded')}",
            f"- quality_path_degradation_reason: {analysis_contract.get('quality_path_degradation_reason')}",
            f"- llm_validation_status: {analysis_contract.get('llm_validation_status')}",
            f"- llm_input_valid_ref_count: {analysis_contract.get('llm_input_valid_ref_count')}",
            f"- llm_usable_claim_count: {analysis_contract.get('llm_usable_claim_count')}",
            f"- llm_dropped_claim_count: {analysis_contract.get('llm_dropped_claim_count')}",
            f"- llm_usable_chapter_count: {analysis_contract.get('llm_usable_chapter_count')}",
            f"- llm_valid_chapter_count: {analysis_contract.get('llm_valid_chapter_count')}",
            f"- llm_failed_chapter_count: {analysis_contract.get('llm_failed_chapter_count')}",
            f"- llm_analysis_cache_hit_count: {analysis_contract.get('llm_analysis_cache_hit_count')}",
            f"- llm_raw_chapter_count: {analysis_contract.get('llm_raw_chapter_count')}",
            f"- llm_raw_claim_count: {analysis_contract.get('llm_raw_claim_count')}",
            f"- llm_validation_issue_counts: {analysis_contract.get('llm_validation_issue_counts')}",
            f"- llm_validation_issue_examples: {analysis_contract.get('llm_validation_issue_examples')}",
            f"- llm_valid_claim_examples: {analysis_contract.get('llm_valid_claim_examples')}",
            f"- llm_rejected_claim_examples: {analysis_contract.get('llm_rejected_claim_examples')}",
            f"- analysis_contract_issues: {analysis_contract.get('analysis_contract_issues')}",
            f"- analysis_rebuild_reasons: {analysis_contract.get('analysis_rebuild_reasons')}",
            f"- claim_unit_count: {analysis_contract.get('claim_unit_count')}",
            f"- chapter_insight_count: {analysis_contract.get('chapter_insight_count')}",
            f"- evidence_analysis_count: {analysis_contract.get('evidence_analysis_count')}",
            f"- render_artifacts_full: {analysis_contract.get('render_artifacts_full')}",
            f"- compacted_artifact_used: {analysis_contract.get('compacted_artifact_used')}",
        ]
    )
    executive_summary_diag = _executive_summary_diagnostics(writer_package, writer_report)
    lines.extend(
        [
            "",
            "## Executive Summary / Thesis",
            "",
            f"- executive_summary_valid_judgment_count: {executive_summary_diag.get('executive_summary_valid_judgment_count')}",
            f"- executive_summary_filtered_judgment_count: {executive_summary_diag.get('executive_summary_filtered_judgment_count')}",
            f"- executive_summary_fallback_used: {executive_summary_diag.get('executive_summary_fallback_used')}",
            f"- executive_summary_omitted_low_quality: {executive_summary_diag.get('executive_summary_omitted_low_quality')}",
            f"- filtered_summary_examples: {executive_summary_diag.get('filtered_summary_examples')}",
        ]
    )
    fact_extractor_diag = _readpage_fact_extractor_diagnostics(writer_package)
    if fact_extractor_diag:
        lines.extend(
            [
                "",
                "## Readpage Fact Extractor",
                "",
                f"- readpage_fact_extractor_attempted: {fact_extractor_diag.get('attempted')}",
                f"- readpage_fact_extractor_success_count: {fact_extractor_diag.get('success_count')}",
                f"- fact_card_count: {fact_extractor_diag.get('fact_card_count')}",
                f"- rejected_span_count: {fact_extractor_diag.get('rejected_span_count')}",
                f"- invalid_metric_count: {fact_extractor_diag.get('invalid_metric_count')}",
                f"- regex_fallback_used: {fact_extractor_diag.get('regex_fallback_used')}",
                f"- regex_fallback_point_count: {fact_extractor_diag.get('regex_fallback_point_count')}",
                f"- extractor_empty_without_regex_points: {fact_extractor_diag.get('extractor_empty_without_regex_points')}",
                f"- extractor_cache_hit_count: {fact_extractor_diag.get('cache_hit_count')}",
                f"- llm_extractor_error_count: {fact_extractor_diag.get('llm_error_count')}",
                f"- extractor_budget_limit: {fact_extractor_diag.get('budget_limit')}",
                f"- extractor_budget_used: {fact_extractor_diag.get('budget_used')}",
                f"- extractor_budget_exhausted: {fact_extractor_diag.get('budget_exhausted')}",
                f"- fallback_used: {fact_extractor_diag.get('fallback_used')}",
                f"- extractor_statuses: {fact_extractor_diag.get('statuses')}",
                f"- extractor_models: {fact_extractor_diag.get('models')}",
            ]
        )
    topic_cache = as_dict(writer_package.get("topic_bundle_cache")) or as_dict(writer_report.get("topic_bundle_cache"))
    topic_preflight = as_dict(topic_cache.get("preflight"))
    topic_store = as_dict(writer_package.get("topic_bundle_cache_store")) or as_dict(writer_report.get("topic_bundle_cache_store"))
    if topic_cache or topic_store:
        lines.extend(
            [
                "",
                "## Topic Bundle Cache",
                "",
                f"- preflight_status: {topic_preflight.get('status') or 'not_recorded'}",
                f"- cache_hit: {topic_cache.get('hit')}",
                f"- seedable: {topic_cache.get('seedable') if 'seedable' in topic_cache else topic_preflight.get('seedable')}",
                f"- seed_evidence_count: {topic_cache.get('seed_evidence_count') or as_dict(topic_preflight.get('completeness')).get('full_evidence_count') or 0}",
                f"- summary_only: {topic_cache.get('summary_only') if 'summary_only' in topic_cache else topic_preflight.get('summary_only')}",
                f"- usable_for_skip_search: {topic_cache.get('usable_for_skip_search') if 'usable_for_skip_search' in topic_cache else topic_preflight.get('usable_for_skip_search')}",
                f"- used_for_skip_search: {topic_cache.get('used_for_skip_search')}",
                f"- topic_key: {topic_preflight.get('topic_key') or topic_cache.get('topic_key') or topic_store.get('topic_key') or ''}",
                f"- cache_path: {topic_preflight.get('path') or topic_cache.get('path') or topic_store.get('path') or ''}",
                f"- analysis_rebuild_required: {topic_preflight.get('analysis_rebuild_required') or topic_cache.get('analysis_rebuild_required')}",
                f"- stored_from: {topic_preflight.get('stored_from') or topic_store.get('stored_from') or ''}",
                f"- store_status: {'stored' if topic_store.get('stored') else topic_store.get('reason') or 'not_recorded'}",
            ]
        )
    if scorecard:
        lines.extend(["", "## Writer 评分摘要", "", scorecard])
    qa = as_dict(writer_report.get("qa_result"))
    if qa:
        qa_findings = as_list(qa.get("quality_findings"))
        category_counts: Dict[str, int] = {}
        for item in qa_findings:
            payload = as_dict(item)
            category = str(payload.get("finding_category") or payload.get("qa_category") or "uncategorized")
            category_counts[category] = category_counts.get(category, 0) + 1
        lines.extend(
            [
                "",
                "## QA 审查",
                "",
                f"- passed：{qa.get('passed')}",
                f"- publishable：{qa.get('publishable')}",
                f"- quality_score：{qa.get('quality_score')}",
            ]
        )
        lines.extend(
            [
                f"- render_gate_blocked: {as_dict(qa.get('render_gate')).get('blocked')}",
                f"- clean_gate_eligible: {as_dict(qa.get('clean_gate')).get('eligible')}",
                f"- clean_content_eligible: {as_dict(qa.get('clean_gate')).get('clean_content_eligible')}",
                f"- clean_candidate_eligible: {as_dict(qa.get('clean_gate')).get('clean_candidate_eligible')}",
                f"- finding_categories: {category_counts}",
            ]
        )
        readability = as_list(qa.get("readability_findings"))
        if readability:
            lines.extend(["", "### Formal Report Readability Findings"])
            for item in readability[:20]:
                payload = as_dict(item)
                lines.append(
                    f"- {payload.get('type') or 'readability_finding'}"
                    + (f" / chapter={payload.get('chapter_id')}" if payload.get("chapter_id") else "")
                    + (f" / section={payload.get('section_id')}" if payload.get("section_id") else "")
                )
        qa_items = as_list(qa.get("errors")) + as_list(qa.get("warnings"))
        if qa_items:
            lines.extend(["", "### QA 问题"])
            for item in qa_items[:30]:
                payload = as_dict(item)
                category = payload.get("finding_category") or payload.get("qa_category")
                if category:
                    payload["type"] = f"{payload.get('type') or str(item)[:120]} ({category})"
                lines.append(f"- {payload.get('type') or str(item)[:180]}")
    findings = as_list(writer_report.get("quality_findings"))
    if findings:
        lines.extend(["", "## 质量缺陷清单"])
        for item in findings[:60]:
            payload = as_dict(item)
            if payload:
                detail = payload.get("type") or payload.get("message") or payload.get("reason") or str(payload)[:180]
                source = payload.get("source")
                lines.append(f"- {source + ': ' if source else ''}{detail}")
    health = as_dict(writer_package.get("evidence_health_summary"))
    if health:
        lines.extend(
            [
                "",
                "## 证据健康摘要",
                "",
                f"- analysis_ready_count：{health.get('analysis_ready_count')}",
                f"- clean_fact_count：{health.get('clean_fact_count')}",
                f"- traceable_ab_source_count：{health.get('traceable_ab_source_count')}",
                f"- distinct_traceable_ab_source_count：{health.get('distinct_traceable_ab_source_count')}",
                f"- distinct_verified_ab_source_count：{health.get('distinct_verified_ab_source_count')}",
                f"- distinct_primary_source_count：{health.get('distinct_primary_source_count')}",
                f"- distinct_counter_source_count：{health.get('distinct_counter_source_count')}",
                f"- verified_source_count：{health.get('verified_source_count')}",
                f"- topic_bundle_seed_evidence_count：{health.get('topic_bundle_seed_evidence_count')}",
                f"- live_evidence_count：{health.get('live_evidence_count')}",
                f"- evidence_origin_distribution：{health.get('evidence_origin_distribution')}",
                f"- source_candidate_count：{health.get('source_candidate_count')}",
                f"- readpage_succeeded：{health.get('readpage_succeeded')}",
                f"- publishable_evidence_gate_passed：{health.get('publishable_evidence_gate_passed')}",
            ]
        )
    evidence_input_diag = _chapter_evidence_input_diagnostics(writer_package)
    if evidence_input_diag.get("chapter_count"):
        lines.extend(
            [
                "",
                "## 章节证据输入摘要",
                "",
                f"- source_pool_count：{evidence_input_diag.get('source_pool_count')}",
                f"- chapter_evidence_binding_failed：{evidence_input_diag.get('chapter_evidence_binding_failed')}",
                f"- compacted_artifact_used：{evidence_input_diag.get('compacted_artifact_used')}",
                f"- total_unresolved_evidence_ref_count：{evidence_input_diag.get('total_unresolved_evidence_ref_count')}",
            ]
        )
        for item in as_list(evidence_input_diag.get("chapters"))[:12]:
            payload = as_dict(item)
            lines.append(
                f"- {payload.get('chapter_id') or '-'}："
                f"core={payload.get('core_evidence_count')} "
                f"supporting={payload.get('supporting_evidence_count')} "
                f"metric={payload.get('metric_evidence_count')} "
                f"counter={payload.get('counter_evidence_count')} "
                f"case={payload.get('case_evidence_count')} "
                f"directional={payload.get('directional_evidence_count')} "
                f"sample={payload.get('sample_evidence_count')} "
                f"hydrated_items={payload.get('hydrated_layer_item_count')} "
                f"writable_facts={payload.get('eligible_fact_count')} "
                f"fact_cards={payload.get('fact_card_count')} "
                f"analysis_valid={payload.get('chapter_analysis_valid')} "
                f"directional_cards={payload.get('directional_fact_card_count')} "
                f"strong_cards={payload.get('strong_fact_card_count')} "
                f"filtered_facts={payload.get('filtered_fact_count')} "
                f"invalid_metrics={payload.get('invalid_metric_filtered_count')} "
                f"citation_ready={payload.get('eligible_citation_count')} "
                f"count_only_warning={payload.get('count_only_warning')} "
                f"sample_only_warning={payload.get('sample_only_warning')} "
                f"unresolved_refs={payload.get('unresolved_evidence_ref_count')}"
            )
    grade_usage_diag = _evidence_grade_usage_diagnostics(writer_package)
    if grade_usage_diag:
        lines.extend(
            [
                "",
                "## 检索与证据降级区分",
                "",
                "- 检索策略 fallback：表示 IQS 主查询结果不足后换检索策略补召回，不等同于 B/C 证据降级。",
                "- 证据降级：表示正文使用 B/C 可追溯材料时降低结论强度，只支撑方向性分析。",
                f"- retrieval_scheduled_count：{grade_usage_diag.get('retrieval_scheduled_count')}",
                f"- retrieval_dropped_count：{grade_usage_diag.get('retrieval_dropped_count')}",
                f"- retrieval_dropped_blocking_count：{grade_usage_diag.get('retrieval_dropped_blocking_count')}",
                f"- retrieval_partial_lane_count：{grade_usage_diag.get('retrieval_partial_lane_count')}",
                f"- retrieval_failed_task_count：{grade_usage_diag.get('retrieval_failed_task_count')}",
                f"- retrieval_timed_out_task_count：{grade_usage_diag.get('retrieval_timed_out_task_count')}",
                f"- source_level_usage：{grade_usage_diag.get('source_level_usage')}",
                f"- bc_directional_evidence_item_count：{grade_usage_diag.get('bc_directional_evidence_item_count')}",
                f"- bc_degraded_chapter_count：{grade_usage_diag.get('bc_degraded_chapter_count')}",
                f"- source_mismatch_filtered_count：{grade_usage_diag.get('source_mismatch_filtered_count')}",
                f"- low_quality_filtered_count：{grade_usage_diag.get('low_quality_filtered_count')}",
            ]
        )
    layout_diag = _layout_score_diagnostics(writer_package)
    if layout_diag.get("micro_layout_count") or layout_diag.get("chapter_package_count"):
        lines.extend(
            [
                "",
                "## 动态排版摘要",
                "",
                f"- micro_layout_count：{layout_diag.get('micro_layout_count')}",
                f"- chapter_package_count：{layout_diag.get('chapter_package_count')}",
                f"- layout_block_rendered_count：{layout_diag.get('rendered_block_count')}",
                f"- layout_block_dropped：{layout_diag.get('dropped_block_count')}",
                f"- layout_block_observation_only_count：{layout_diag.get('observation_only_block_count')}",
                f"- layout_block_evidence_backed_count：{layout_diag.get('evidence_backed_block_count')}",
                f"- layout_effective_section_count：{layout_diag.get('effective_section_count')}",
                f"- compacted_artifact_used：{layout_diag.get('compacted_artifact_used')}",
                f"- layout_omitted_observation_section_count：{layout_diag.get('omitted_observation_section_count')}",
                f"- template_section_removed_count：{layout_diag.get('template_section_removed_count')}",
                f"- block_without_fact_card_count：{layout_diag.get('block_without_fact_card_count')}",
                f"- fact_card_to_block_match_count：{layout_diag.get('fact_card_to_block_match_count')}",
                f"- dropped_due_to_template_risk_count：{layout_diag.get('dropped_due_to_template_risk_count')}",
                f"- generic_section_title_count：{layout_diag.get('generic_section_title_count')}",
                f"- repeated_section_title_count：{layout_diag.get('repeated_section_title_count')}",
                f"- dynamic_section_title_count：{layout_diag.get('dynamic_section_title_count')}",
                f"- natural_title_count：{layout_diag.get('natural_title_count')}",
                f"- generic_title_filtered_count：{layout_diag.get('generic_title_filtered_count')}",
                f"- repeated_title_rewrite_count：{layout_diag.get('repeated_title_rewrite_count')}",
                f"- repeated_title_dropped_count：{layout_diag.get('repeated_title_dropped_count')}",
                f"- internal_role_title_filtered_count：{layout_diag.get('internal_role_title_filtered_count')}",
                f"- weak_template_sentence_removed_count：{layout_diag.get('weak_template_sentence_removed_count')}",
                f"- publisher_in_title_count：{layout_diag.get('publisher_in_title_count')}",
                f"- publisher_in_subject_count：{layout_diag.get('publisher_in_subject_count')}",
                f"- english_snippet_lead_count：{layout_diag.get('english_snippet_lead_count')}",
                f"- empty_parens_in_body_count：{layout_diag.get('empty_parens_in_body_count')}",
                f"- snippet_lead_rewritten_count：{layout_diag.get('snippet_lead_rewritten_count')}",
                f"- snippet_lead_dropped_count：{layout_diag.get('snippet_lead_dropped_count')}",
                f"- repeated_evidence_id_within_chapter_count：{layout_diag.get('repeated_evidence_id_within_chapter_count')}",
                f"- repeated_fact_demoted_count：{layout_diag.get('repeated_fact_demoted_count')}",
                f"- snippet_like_text_dropped_count：{layout_diag.get('snippet_like_text_dropped_count')}",
                f"- metric_sentence_rewritten_count：{layout_diag.get('metric_sentence_rewritten_count')}",
                f"- traditional_chinese_normalized_count：{layout_diag.get('traditional_chinese_normalized_count')}",
                f"- ocr_artifact_normalized_count：{layout_diag.get('ocr_artifact_normalized_count')}",
                f"- empty_parens_removed_count：{layout_diag.get('empty_parens_removed_count')}",
                f"- truncated_punctuation_cleaned_count：{layout_diag.get('truncated_punctuation_cleaned_count')}",
                f"- chapter_omitted_no_evidence_count：{layout_diag.get('chapter_omitted_no_evidence_count')}",
                f"- core_chapter_omitted_no_evidence_count：{layout_diag.get('core_chapter_omitted_no_evidence_count')}",
                f"- optional_chapter_omitted_count：{layout_diag.get('optional_chapter_omitted_count')}",
                f"- composer_variable_explanation_count：{layout_diag.get('composer_variable_explanation_count')}",
                f"- section_plan_valid_count：{layout_diag.get('section_plan_valid_count')}",
                f"- section_plan_dropped_count：{layout_diag.get('section_plan_dropped_count')}",
                f"- block_title_generation_failed_count：{layout_diag.get('block_title_generation_failed_count')}",
                f"- dropped_generic_block_count：{layout_diag.get('dropped_generic_block_count')}",
                f"- filtered_table_count：{layout_diag.get('filtered_table_count')}",
                f"- invalid_table_count：{layout_diag.get('invalid_table_count')}",
            ]
        )
        lines.extend(
            [
                f"- llm_claim_to_block_match_count: {layout_diag.get('llm_claim_to_block_match_count')}",
                f"- llm_claim_unmatched_count: {layout_diag.get('llm_claim_unmatched_count')}",
                f"- must_block_matched_by_llm_claim_count: {layout_diag.get('must_block_matched_by_llm_claim_count')}",
                f"- must_block_dropped_no_matching_claim_count: {layout_diag.get('must_block_dropped_no_matching_claim_count')}",
            ]
        )
        if layout_diag.get("repeated_section_titles"):
            lines.append(f"- repeated_section_titles：{layout_diag.get('repeated_section_titles')}")
        chapters = as_list(layout_diag.get("chapters"))
        if chapters:
            lines.extend(["", "### 章节 Block 对照"])
            for item in chapters[:12]:
                payload = as_dict(item)
                planned = ", ".join(str(block) for block in as_list(payload.get("planned_blocks"))) or "-"
                candidate = ", ".join(str(block) for block in as_list(payload.get("candidate_blocks"))) or "-"
                rendered = ", ".join(str(block) for block in as_list(payload.get("rendered_blocks"))) or "-"
                dropped = ", ".join(str(block) for block in as_list(payload.get("dropped_blocks"))) or "-"
                observation = ", ".join(str(block) for block in as_list(payload.get("observation_only_blocks"))) or "-"
                evidence_backed = ", ".join(str(block) for block in as_list(payload.get("evidence_backed_blocks"))) or "-"
                omitted_observation = ", ".join(str(block) for block in as_list(payload.get("omitted_observation_blocks"))) or "-"
                lines.append(
                    f"- {payload.get('chapter_id')}: planned=[{planned}] rendered=[{rendered}] "
                    f"candidate=[{candidate}] "
                    f"matched_by_llm_claim={payload.get('matched_by_llm_claim')} "
                    f"unmatched_llm_claims={payload.get('unmatched_llm_claims')} "
                    f"evidence_backed=[{evidence_backed}] observation_only=[{observation}] "
                    f"omitted_observation=[{omitted_observation}] dropped=[{dropped}] "
                    f"effective_sections={payload.get('effective_section_count')} "
                    f"observation_sections={payload.get('observation_section_count')} "
                    f"effective_ratio={payload.get('chapter_effective_section_ratio')}"
                )
        invalid_tables = as_list(layout_diag.get("invalid_table_candidates"))
        if invalid_tables:
            lines.extend(["", "### Invalid Tables"])
            for item in invalid_tables[:20]:
                payload = as_dict(item)
                lines.append(
                    f"- {payload.get('chapter_id') or '-'} / {payload.get('table_id') or '-'} / {payload.get('anchor_block_type') or '-'}: {payload.get('reason') or 'invalid_table'}"
                )
        skipped_tables = as_list(layout_diag.get("skipped_tables"))
        if skipped_tables:
            lines.extend(["", "### 表格未渲染"])
            for item in skipped_tables[:20]:
                payload = as_dict(item)
                lines.append(
                    f"- {payload.get('chapter_id') or '-'} / {payload.get('table_id') or '-'} / {payload.get('anchor_block_type') or '-'}: {payload.get('reason') or 'not_rendered'}"
                )
    total_valid_fact_cards = sum(
        _coerce_int(as_dict(item).get("eligible_fact_count") or as_dict(item).get("fact_card_count"))
        for item in as_list(evidence_input_diag.get("chapters"))
    )
    citation_manifest = (
        as_dict(writer_report.get("citation_manifest"))
        or as_dict(writer_package.get("citation_manifest"))
        or as_dict(as_dict(writer_report.get("render_artifacts")).get("citation_manifest"))
    )
    final_citation_audit = (
        as_dict(writer_report.get("final_citation_audit"))
        or as_dict(writer_package.get("final_citation_audit"))
        or as_dict(as_dict(writer_report.get("render_artifacts")).get("final_citation_audit"))
    )
    source_claim_support = (
        as_dict(writer_report.get("source_claim_support"))
        or as_dict(writer_package.get("source_claim_support"))
        or as_dict(as_dict(writer_report.get("render_artifacts")).get("source_claim_support"))
    )
    analysis_transfer = (
        as_dict(writer_report.get("analysis_transfer"))
        or as_dict(writer_package.get("analysis_transfer"))
        or as_dict(as_dict(writer_report.get("render_artifacts")).get("analysis_transfer"))
    )
    ref_lineage_diagnostics = (
        as_dict(writer_report.get("ref_lineage_diagnostics"))
        or as_dict(writer_package.get("ref_lineage_diagnostics"))
        or as_dict(as_dict(writer_report.get("render_artifacts")).get("ref_lineage_diagnostics"))
    )
    public_narrative_leak_audit = (
        as_dict(writer_report.get("public_narrative_leak_audit"))
        or as_dict(writer_package.get("public_narrative_leak_audit"))
        or as_dict(as_dict(writer_report.get("render_artifacts")).get("public_narrative_leak_audit"))
    )
    report_markdown_for_health = str(
        writer_report.get("report_markdown")
        or as_dict(writer_package.get("writer_report")).get("report_markdown")
        or writer_package.get("report_markdown")
        or ""
    )
    target_body_chars_for_health = _coerce_int(writer_report.get("target_body_chars")) or _coerce_int(os.getenv("REPORT_TARGET_BODY_CHARS"))
    body_markdown_for_health = _public_body_markdown_for_health(report_markdown_for_health)
    h3_count_for_health = len(re.findall(r"(?m)^###\s+", body_markdown_for_health))
    body_char_count_for_health = _public_body_char_count(report_markdown_for_health)
    quality_posture = as_dict(writer_package.get("quality_posture"))
    source_appendix_status = (
        str(citation_manifest.get("citation_manifest_status") or "").strip()
        or ("blocked" if bool(final_audit_result.get("blocked")) and "source" in str(final_audit_result).lower() else "ok")
    )
    report_health = build_report_health_card(
        {
            "layout": layout_diag,
            "chapter_evidence": {"total_valid_fact_card_count": total_valid_fact_cards},
            "analysis": analysis_contract,
            "summary": executive_summary_diag,
            "source_appendix_status": source_appendix_status,
            "citation_manifest": citation_manifest,
            "final_citation_audit": final_citation_audit,
            "source_claim_support": source_claim_support,
            "ref_lineage_diagnostics": ref_lineage_diagnostics,
            "quality_posture": quality_posture,
            "quality_mode": quality_posture.get("mode") or writer_package.get("quality_mode") or writer_report.get("quality_mode"),
            "body_char_count": body_char_count_for_health,
            "target_body_chars": target_body_chars_for_health,
            "avg_chars_per_section": round(body_char_count_for_health / max(1, h3_count_for_health), 1),
            "h3_count": h3_count_for_health,
            "body_rewrite": as_dict(layout_diag.get("body_rewrite")),
            "chapter_narrative": as_dict(layout_diag.get("chapter_narrative")) or as_dict(writer_report.get("chapter_narrative")),
            "body_composition_status": layout_diag.get("body_composition_status"),
            "claim_to_evidence_binding_status": "ok"
            if _coerce_int(layout_diag.get("evidence_backed_block_count")) > 0
            else "weak",
        }
    )
    lines.extend(
        [
            "",
            "## Report HealthCard",
            "",
            f"- overall_status: {report_health.get('overall_status')}",
            f"- body_composition_status: {report_health.get('body_composition_status')}",
            f"- body_rewrite_status: {report_health.get('body_rewrite_status')}",
            f"- chapter_narrative_status: {report_health.get('chapter_narrative_status')}",
            f"- quality_path_degraded: {report_health.get('quality_path_degraded')}",
            f"- quality_degraded_reasons: {report_health.get('quality_degraded_reasons') or []}",
            f"- body_char_count: {report_health.get('body_char_count')}",
            f"- target_body_chars: {report_health.get('target_body_chars')}",
            f"- body_char_gap: {report_health.get('body_char_gap')}",
            f"- avg_chars_per_section: {round(body_char_count_for_health / max(1, h3_count_for_health), 1)}",
            f"- h3_count: {report_health.get('h3_count')}",
            f"- valid_fact_card_count: {report_health.get('valid_fact_card_count')}",
            f"- evidence_backed_section_ratio: {report_health.get('evidence_backed_section_ratio')}",
            f"- planned_vs_rendered_section_ratio: {report_health.get('planned_vs_rendered_section_ratio')}",
            f"- must_render_block_count: {report_health.get('must_render_block_count')}",
            f"- candidate_block_count: {report_health.get('candidate_block_count')}",
            f"- rendered_must_block_count: {report_health.get('rendered_must_block_count')}",
            f"- llm_claim_to_block_match_count: {report_health.get('llm_claim_to_block_match_count')}",
            f"- llm_claim_unmatched_count: {report_health.get('llm_claim_unmatched_count')}",
            f"- must_block_matched_by_llm_claim_count: {report_health.get('must_block_matched_by_llm_claim_count')}",
            f"- must_block_dropped_no_matching_claim_count: {report_health.get('must_block_dropped_no_matching_claim_count')}",
            f"- snippet_like_text_count: {report_health.get('snippet_like_text_count')}",
            f"- repeated_fact_count: {report_health.get('repeated_fact_count')}",
            f"- chapter_omitted_no_evidence_count: {report_health.get('chapter_omitted_no_evidence_count')}",
            f"- core_chapter_omitted_no_evidence_count: {report_health.get('core_chapter_omitted_no_evidence_count')}",
            f"- optional_chapter_omitted_count: {report_health.get('optional_chapter_omitted_count')}",
            f"- composer_variable_explanation_count: {report_health.get('composer_variable_explanation_count')}",
            f"- summary_valid_judgment_count: {report_health.get('summary_valid_judgment_count')}",
            f"- source_appendix_status: {report_health.get('source_appendix_status')}",
            f"- citation_manifest_status: {report_health.get('citation_manifest_status')}",
            f"- final_citation_status_after_render: {report_health.get('final_citation_status_after_render')}",
            f"- final_body_citation_refs: {report_health.get('final_body_citation_refs')}",
            f"- final_appendix_refs: {report_health.get('final_appendix_refs')}",
            f"- final_missing_appendix_refs: {report_health.get('final_missing_appendix_refs')}",
            f"- factual_body_without_citations_count: {report_health.get('factual_body_without_citations_count')}",
            f"- citationless_fact_examples: {report_health.get('citationless_fact_examples') or []}",
            f"- final_unresolved_citation_removed_count: {report_health.get('final_unresolved_citation_removed_count')}",
            f"- missing_source_ref_count: {report_health.get('missing_source_ref_count')}",
            f"- orphan_citation_count: {report_health.get('orphan_citation_count')}",
            f"- excluded_source_count: {report_health.get('excluded_source_count')}",
            f"- final_analysis_source: {report_health.get('final_analysis_source')}",
        ]
    )
    body_rewrite_diag = as_dict(layout_diag.get("body_rewrite"))
    if body_rewrite_diag:
        lines.extend(
            [
                "",
                "### Body Rewrite Diagnostics",
                f"- body_rewrite_enabled: {body_rewrite_diag.get('enabled')}",
                f"- body_rewrite_concurrency: {body_rewrite_diag.get('concurrency')}",
                f"- body_rewrite_submitted_count: {body_rewrite_diag.get('submitted_count')}",
                f"- body_rewrite_called_count: {body_rewrite_diag.get('called_count')}",
                f"- body_rewrite_success_count: {body_rewrite_diag.get('success_count')}",
                f"- body_rewrite_cache_hit_count: {body_rewrite_diag.get('cache_hit_count')}",
                f"- body_rewrite_rejected_count: {body_rewrite_diag.get('rejected_count')}",
                f"- body_rewrite_fallback_count: {body_rewrite_diag.get('fallback_count')}",
                f"- body_rewrite_budget_exhausted_count: {body_rewrite_diag.get('budget_exhausted_count')}",
                f"- body_rewrite_inflight_dedup_count: {body_rewrite_diag.get('inflight_dedup_count')}",
                f"- body_rewrite_elapsed_seconds: {body_rewrite_diag.get('elapsed_seconds')}",
                f"- body_rewrite_failure_reasons: {body_rewrite_diag.get('failure_reasons') or {}}",
            ]
        )
    chapter_narrative_diag = as_dict(layout_diag.get("chapter_narrative"))
    if chapter_narrative_diag:
        lines.extend(
            [
                "",
                "### Chapter Narrative Diagnostics",
                f"- chapter_narrative_enabled: {chapter_narrative_diag.get('enabled')}",
                f"- chapter_narrative_attempted_count: {chapter_narrative_diag.get('attempted_count')}",
                f"- chapter_narrative_success_count: {chapter_narrative_diag.get('success_count')}",
                f"- chapter_narrative_fallback_count: {chapter_narrative_diag.get('fallback_count')}",
                f"- chapter_narrative_rejected_reasons: {chapter_narrative_diag.get('rejected_reasons') or {}}",
                f"- chapter_narrative_skipped_reason: {chapter_narrative_diag.get('skipped_reason') or ''}",
            ]
        )
    if citation_manifest:
        excluded = as_list(citation_manifest.get("excluded_cited_sources"))
        missing = as_list(citation_manifest.get("missing_evidence_refs"))
        filtered_sources = as_list(citation_manifest.get("filtered_cited_sources"))
        filtered_refs = as_list(citation_manifest.get("filtered_unresolved_refs"))
        filtered_ref_reasons = as_list(citation_manifest.get("filtered_unresolved_ref_reasons"))
        if excluded or missing or filtered_sources or filtered_refs:
            lines.extend(["", "### Citation Manifest Diagnostics"])
            for item in excluded[:10]:
                payload = as_dict(item)
                lines.append(
                    f"- excluded: {payload.get('ref') or '-'} / {payload.get('reason') or '-'} / {payload.get('title') or payload.get('url') or '-'}"
                )
            for ref in missing[:10]:
                lines.append(f"- missing_source_ref: {ref}")
            for item in filtered_sources[:10]:
                payload = as_dict(item)
                lines.append(
                    f"- filtered_cited_source: {payload.get('ref') or '-'} / {payload.get('reason') or '-'} / {payload.get('title') or payload.get('url') or '-'}"
                )
            for ref in filtered_refs[:10]:
                lines.append(f"- filtered_unresolved_ref: {ref}")
            for item in filtered_ref_reasons[:10]:
                payload = as_dict(item)
                lines.append(
                    f"- filtered_unresolved_ref_reason: {payload.get('ref') or '-'} / {payload.get('reason') or '-'} / {payload.get('title') or payload.get('url') or '-'}"
                )
    if final_citation_audit:
        lines.extend(
            [
                "",
                "### Final Citation Diagnostics",
                f"- manifest_status_before_render: {citation_manifest.get('citation_manifest_status') if citation_manifest else ''}",
                f"- final_citation_status_after_render: {final_citation_audit.get('final_citation_reconciliation_status')}",
                f"- final_body_citation_refs: {final_citation_audit.get('final_body_citation_refs') or []}",
                f"- final_appendix_refs: {final_citation_audit.get('final_appendix_refs') or []}",
                f"- final_missing_appendix_refs: {final_citation_audit.get('final_missing_appendix_refs') or []}",
                f"- final_unresolved_citation_removed_count: {final_citation_audit.get('final_unresolved_citation_removed_count') or 0}",
                f"- final_duplicate_citation_removed_count: {final_citation_audit.get('final_duplicate_citation_removed_count') or 0}",
                f"- factual_body_without_citations_count: {final_citation_audit.get('factual_body_without_citations_count') or 0}",
                f"- citationless_fact_examples: {final_citation_audit.get('citationless_fact_examples') or []}",
            ]
        )
    if source_claim_support:
        lines.extend(
            [
                "",
                "### Source Claim Support Diagnostics",
                f"- source_claim_support_status: {source_claim_support.get('source_claim_support_status')}",
                f"- source_gate_mode: {source_claim_support.get('source_gate_mode')}",
                f"- section_dropped_due_to_source_claim_mismatch_count: {source_claim_support.get('section_dropped_due_to_source_claim_mismatch_count')}",
                f"- section_dropped_due_to_unresolved_refs_count: {source_claim_support.get('section_dropped_due_to_unresolved_refs_count')}",
                f"- factual_section_without_resolved_ref_count: {source_claim_support.get('factual_section_without_resolved_ref_count')}",
                f"- citationless_fact_examples: {source_claim_support.get('citationless_fact_examples') or []}",
                f"- metric_claim_without_metric_fact_count: {source_claim_support.get('metric_claim_without_metric_fact_count')}",
                f"- weak_source_strong_claim_demoted_count: {source_claim_support.get('weak_source_strong_claim_demoted_count')}",
                f"- demoted_section_count: {source_claim_support.get('demoted_section_count')}",
                f"- hard_dropped_section_count: {source_claim_support.get('hard_dropped_section_count')}",
                f"- soft_gate_rewritten_count: {source_claim_support.get('soft_gate_rewritten_count')}",
                f"- empty_chapter_omitted_after_source_gate_count: {source_claim_support.get('empty_chapter_omitted_after_source_gate_count')}",
            ]
        )
        for item in as_list(source_claim_support.get("source_claim_mismatch_examples"))[:10]:
            payload = as_dict(item)
            lines.append(
                f"- source_claim_mismatch: {payload.get('chapter_id') or '-'} / {payload.get('section_id') or '-'} / {payload.get('reason') or '-'} / {payload.get('claim') or '-'}"
            )
    if analysis_transfer:
        lines.extend(
            [
                "",
                "### Analysis Claim Transfer Diagnostics",
                f"- analysis_claim_count: {analysis_transfer.get('analysis_claim_count')}",
                f"- analysis_claim_count_by_strength: {analysis_transfer.get('analysis_claim_count_by_strength') or {}}",
                f"- rendered_analysis_section_count: {analysis_transfer.get('rendered_analysis_section_count')}",
                f"- rendered_analysis_claim_count: {analysis_transfer.get('rendered_analysis_claim_count')}",
                f"- claim_to_section_transfer_rate: {analysis_transfer.get('claim_to_section_transfer_rate')}",
                f"- claim_lost_after_analysis_count: {analysis_transfer.get('claim_lost_after_analysis_count')}",
                f"- claim_lost_after_analysis_reasons: {analysis_transfer.get('claim_lost_after_analysis_reasons') or {}}",
                f"- analysis_fact_usage_count: {analysis_transfer.get('analysis_fact_usage_count')}",
                f"- analysis_claim_ids_rendered: {analysis_transfer.get('analysis_claim_ids_rendered') or []}",
                f"- analysis_claim_ids_lost: {analysis_transfer.get('analysis_claim_ids_lost') or []}",
            ]
        )
    if ref_lineage_diagnostics:
        filtered_refs = as_list(ref_lineage_diagnostics.get("filtered_refs"))
        lines.extend(
            [
                "",
                "### Ref Lineage Diagnostics",
                f"- filtered_unresolved_ref_count: {ref_lineage_diagnostics.get('filtered_unresolved_ref_count')}",
                f"- section_ref_recovered_count: {ref_lineage_diagnostics.get('section_ref_recovered_count')}",
                f"- sections_with_filtered_refs: {ref_lineage_diagnostics.get('sections_with_filtered_refs')}",
                f"- claims_with_filtered_refs: {ref_lineage_diagnostics.get('claims_with_filtered_refs')}",
            ]
        )
        for item in filtered_refs[:10]:
            payload = as_dict(item)
            lines.append(
                f"- ref_lineage_filtered: {payload.get('ref') or '-'} / {payload.get('reason') or '-'} / {payload.get('chapter_id') or '-'} / {payload.get('section_id') or '-'}"
            )
    if public_narrative_leak_audit:
        lines.extend(
            [
                "",
                "### Public Narrative Leak Audit",
                f"- public_narrative_leak_input_count: {public_narrative_leak_audit.get('public_narrative_leak_input_count') or 0}",
                f"- public_narrative_leak_removed_count: {public_narrative_leak_audit.get('public_narrative_leak_removed_count') or 0}",
                f"- public_narrative_leak_remaining_count: {public_narrative_leak_audit.get('public_narrative_leak_remaining_count') or 0}",
                f"- skipped_global_block_count: {public_narrative_leak_audit.get('skipped_global_block_count') or 0}",
                f"- skipped_global_blocks: {public_narrative_leak_audit.get('skipped_global_blocks') or []}",
                f"- public_narrative_leak_reason_counts: {public_narrative_leak_audit.get('public_narrative_leak_reason_counts') or {}}",
            ]
        )
        for item in as_list(public_narrative_leak_audit.get("public_narrative_leak_examples"))[:10]:
            payload = as_dict(item)
            lines.append(
                f"- public_narrative_leak_removed: line={payload.get('line') or '-'} reason={payload.get('reason') or '-'} text={payload.get('text') or '-'}"
            )
    for metric_name, metric_payload in as_dict(report_health.get("metrics")).items():
        metric_payload = as_dict(metric_payload)
        lines.append(
            f"- health_metric.{metric_name}: status={metric_payload.get('status')} value={metric_payload.get('value')}"
        )
    final_note = final_audit_public_note(final_audit_result)
    if final_note:
        lines.extend(["", final_note])
    if reformatter_result:
        lines.extend(
            [
                "",
                "## Clean/Reformatter 状态",
                "",
                f"- enabled：{reformatter_result.get('enabled')}",
                f"- status：{reformatter_result.get('status')}",
                f"- skipped_reason：{reformatter_result.get('skipped_reason') or ''}",
            ]
        )
    return "\n".join(str(line) for line in lines).strip()


def review_draft_markdown(report_markdown: str, writer_report: Dict[str, Any]) -> str:
    reasons: List[str] = []
    for item in [
        writer_report.get("message"),
        *as_list(writer_report.get("evidence_limitations")),
        *as_list(writer_report.get("qa_pending_repair_reasons")),
        *as_list(writer_report.get("delivery_blockers")),
    ]:
        if isinstance(item, dict):
            text = str(item.get("message") or item.get("type") or item.get("reason") or "").strip()
        else:
            text = str(item or "").strip()
        if text and text not in reasons:
            reasons.append(text)
        if len(reasons) >= 8:
            break
    lines = [
        "# 复核草稿说明",
        "",
        "本文件是复核草稿，不是正式交付版本。系统已保留正文供人工判断，但未满足正式报告的全部证据和质量门槛。",
        "",
        "## 未达正式交付的主要原因",
    ]
    if reasons:
        lines.extend(f"- {reason}" for reason in reasons)
    else:
        lines.append("- 证据、质量或版式门禁尚未全部通过。")
    lines.extend(["", "## 草稿正文", "", str(report_markdown or "").strip()])
    return "\n".join(lines).strip()


def _dedupe_texts(values: Iterable[Any]) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def review_result_pending_repair_reasons(review_result: Dict[str, Any]) -> List[str]:
    structured = as_dict(as_dict(review_result).get("structured_review"))
    if not structured:
        return []
    status = str(structured.get("status") or "").strip()
    reasons: List[str] = []
    if (
        bool(structured.get("evidence_required"))
        or status == "needs_evidence"
        or as_list(structured.get("evidence_followups"))
        or as_list(structured.get("citation_issues"))
    ):
        reasons.append("review_evidence_required")
    if (
        bool(structured.get("rewrite_required"))
        or status == "needs_rewrite"
        or as_list(structured.get("logic_issues"))
    ):
        reasons.append("review_rewrite_required")
    return _dedupe_texts(reasons)


def attach_structured_review_to_writer_report(
    writer_report: Dict[str, Any],
    review_result: Dict[str, Any],
) -> Dict[str, Any]:
    structured = as_dict(as_dict(review_result).get("structured_review"))
    if not structured:
        return dict(writer_report or {})
    updated = dict(writer_report or {})
    updated["review_status"] = str(structured.get("status") or "")
    updated["review_evidence_required"] = bool(
        structured.get("evidence_required")
        or as_list(structured.get("evidence_followups"))
        or as_list(structured.get("citation_issues"))
    )
    updated["review_rewrite_required"] = bool(
        structured.get("rewrite_required") or as_list(structured.get("logic_issues"))
    )
    updated["review_evidence_followups"] = as_list(structured.get("evidence_followups"))
    updated["review_logic_issues"] = as_list(structured.get("logic_issues"))
    updated["review_citation_issues"] = as_list(structured.get("citation_issues"))
    updated["review_pending_repair_reasons"] = merge_writer_review_pending_repair_reasons(updated, review_result)
    return updated


def merge_writer_review_pending_repair_reasons(
    writer_report: Dict[str, Any],
    review_result: Dict[str, Any],
) -> List[str]:
    return _dedupe_texts(
        [
            *as_list(as_dict(writer_report).get("qa_pending_repair_reasons")),
            *as_list(as_dict(writer_report).get("review_pending_repair_reasons")),
            *review_result_pending_repair_reasons(review_result),
        ]
    )


def writer_report_pending_repair_reasons(writer_report: Dict[str, Any]) -> List[str]:
    report = as_dict(writer_report)
    qa = as_dict(report.get("qa_result"))
    reasons: List[str] = []
    if bool(report.get("qa_pending_repair")):
        reasons.extend(as_list(report.get("qa_pending_repair_reasons")) or ["qa_pending_repair"])
    if bool(qa.get("repair_required")):
        reasons.append("repair_required")
    if as_list(qa.get("blocking_followups")):
        reasons.append("blocking_followups")
    if as_list(qa.get("blocking_evidence_repair_followups")):
        reasons.append("evidence_repair_followups")
    if as_list(qa.get("blocking_content_repair_followups")):
        reasons.append("content_repair_followups")
    reasons.extend(as_list(report.get("review_pending_repair_reasons")))
    return _dedupe_texts(reasons)


def quality_gate_state(
    *,
    writer_status: str,
    writer_not_ready: bool,
    writer_pending_repair_reasons: Iterable[str],
    reformatter_result: Optional[Dict[str, Any]] = None,
    report_contract: Optional[Dict[str, Any]] = None,
    writer_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return build_quality_gate_state(
        writer_status=writer_status,
        writer_not_ready=writer_not_ready,
        writer_pending_repair_reasons=list(writer_pending_repair_reasons or []),
        reformatter_result=reformatter_result,
        report_contract=report_contract,
        writer_report=writer_report,
    )


def build_evidence_handoff_diagnostics(
    package: Dict[str, Any],
    *,
    clean_evidence: Optional[Dict[str, Any]] = None,
    validation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = as_dict(package)
    handoff = as_dict(payload.get("reformatter_evidence_package"))
    evidence_package = as_dict(payload.get("evidence_package"))
    writer_report = as_dict(payload.get("writer_report"))
    clean_payload = as_dict(clean_evidence)
    validation_payload = as_dict(validation)
    handoff_facts = len(as_list(handoff.get("clean_evidence_list")))
    clean_dimensions = as_dict(clean_payload.get("dimensions"))
    clean_dimension_facts = sum(len(as_list(items)) for items in clean_dimensions.values())
    raw_evidence_count = int(as_dict(evidence_package.get("metadata")).get("raw_evidence_count") or 0)
    normalized_count = int(as_dict(evidence_package.get("metadata")).get("normalized_count") or 0)
    kept_count = int(as_dict(evidence_package.get("metadata")).get("kept_count") or 0)
    handoff_sources = len(as_list(handoff.get("sources")))
    writer_sources = len(as_list(writer_report.get("source_registry")))
    source_pool_count = int(validation_payload.get("source_pool_count") or max(handoff_sources, writer_sources))
    cited_source_count = int(validation_payload.get("unique_cited_source_count") or 0)
    flags: List[str] = []
    if handoff_facts and (clean_dimension_facts and handoff_facts < clean_dimension_facts or raw_evidence_count and handoff_facts < raw_evidence_count):
        flags.append("evidence_handoff_compacted")
    if source_pool_count and cited_source_count and cited_source_count < source_pool_count:
        flags.append("source_pool_collapsed")
    return {
        "status": "has_diagnostics",
        "flags": _dedupe_texts(flags),
        "counts": {
            "handoff_facts": handoff_facts,
            "clean_dimension_facts": clean_dimension_facts,
            "raw_evidence_count": raw_evidence_count,
            "normalized_count": normalized_count,
            "kept_count": kept_count,
            "handoff_sources": handoff_sources,
            "writer_sources": writer_sources,
            "source_pool_count": source_pool_count,
            "unique_cited_source_count": cited_source_count,
        },
    }


def cache_only_core_claim_block_count(report: Dict[str, Any]) -> int:
    payload = as_dict(report)
    quality = as_dict(payload.get("package_quality_report"))
    errors = [as_dict(item) for item in as_list(quality.get("blocking_errors"))]
    has_core_claim_error = any(str(item.get("type") or "") == "core_claim_without_ab_source" for item in errors)
    cache_summary = as_dict(payload.get("evidence_cache_summary"))
    try:
        cache_misses = int(cache_summary.get("cache_live_refresh_miss_count") or 0)
    except (TypeError, ValueError):
        cache_misses = 0
    return 1 if has_core_claim_error and cache_misses > 0 else 0


def summarize_evidence_gaps(
    *,
    writer_report: Dict[str, Any],
    chapter_evidence_packages: Iterable[Dict[str, Any]],
    search_task_schedule: Optional[Dict[str, Any]] = None,
    post_qa_repair_trace: Iterable[Dict[str, Any]] = (),
) -> Dict[str, Any]:
    report = as_dict(writer_report)
    schedule = as_dict(search_task_schedule)
    dropped_tasks = [as_dict(item) for item in as_list(schedule.get("dropped_tasks"))]
    by_proof_role = Counter(str(item.get("proof_role") or "unknown") for item in dropped_tasks)
    chapter_gaps: List[Dict[str, Any]] = []
    for package in chapter_evidence_packages or []:
        chapter = as_dict(package)
        if not chapter:
            continue
        quality = as_dict(chapter.get("evidence_quality_summary"))
        reasons: List[str] = []
        if int(quality.get("core_ab_source_count") or 0) <= 0:
            reasons.append("low_directness")
        if int(quality.get("core_evidence_count") or 0) <= 0:
            reasons.append("core_evidence_missing")
        reasons.extend(str(as_dict(item).get("type") or item) for item in as_list(chapter.get("missing_evidence")) if str(as_dict(item).get("type") or item))
        if reasons:
            chapter_gaps.append(
                {
                    "chapter_id": chapter.get("chapter_id"),
                    "chapter_title": chapter.get("chapter_title"),
                    "gap_reasons": _dedupe_texts(reasons),
                }
            )
    trace_items = [as_dict(item) for item in post_qa_repair_trace or []]
    post_qa_no_signal = any(str(item.get("status") or item.get("stop_reason") or "") == "no_new_evidence_signal" for item in trace_items)
    qa = as_dict(report.get("qa_result"))
    repair_followups = as_list(qa.get("repair_followups")) or as_list(qa.get("required_followups"))
    return {
        "status": "has_gaps" if chapter_gaps or dropped_tasks or post_qa_no_signal or repair_followups else "ok",
        "chapter_gaps": chapter_gaps,
        "search_tasks_dropped": {
            "count": int(schedule.get("dropped_count") or len(dropped_tasks)),
            "by_proof_role": dict(by_proof_role),
            "items": dropped_tasks[:12],
        },
        "post_qa_repair_failed": post_qa_no_signal,
        "repair_followups": repair_followups,
    }


def build_review_diagnostic(
    *,
    writer_report: Dict[str, Any],
    report_blueprint: Optional[Dict[str, Any]] = None,
    chapter_evidence_packages: Iterable[Dict[str, Any]] = (),
    package_quality_report: Optional[Dict[str, Any]] = None,
    evidence_gap_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    del chapter_evidence_packages
    blueprint = as_dict(report_blueprint)
    quality = as_dict(package_quality_report)
    gap_summary = as_dict(evidence_gap_summary)
    chapters = [as_dict(item) for item in as_list(blueprint.get("chapters"))]
    title_issues = [
        chapter
        for chapter in chapters
        if len(str(chapter.get("chapter_title") or "")) > 36 or "?" in str(chapter.get("chapter_title") or "")
    ]
    table_warnings = len(as_list(quality.get("warnings")))
    checks = {
        "evidence_gaps": bool(gap_summary.get("chapter_gaps")),
        "post_qa_repair_failed": bool(gap_summary.get("post_qa_repair_failed")),
        "table_validation_warnings": table_warnings,
        "title_contract_issues": len(title_issues),
    }
    needs_review = any(bool(value) for value in checks.values()) or str(as_dict(writer_report).get("report_status") or "") == "review_required"
    return {
        "status": "needs_review" if needs_review else "passed",
        "checks": checks,
        "evidence_gap_summary": gap_summary,
    }


def write_priority_report_output(
    *,
    output_path: Path,
    writer_report: Dict[str, Any],
    report_markdown: str,
    writer_status: str,
    reason: str = "",
) -> Dict[str, Any]:
    writer = as_dict(writer_report)
    clean_text = str(report_markdown or "").strip()
    writer_text = str(writer.get("report_markdown") or "").strip()
    if clean_text:
        write_markdown(Path(output_path), clean_text)
        return {"output_written": True, "clean_report": True, "path": str(output_path), "reason": reason}
    if writer_text and str(writer_status or writer.get("report_status") or "").strip() == "review_required":
        write_markdown(Path(output_path), writer_text)
        return {"output_written": True, "clean_report": False, "path": str(output_path), "reason": reason}
    return {"output_written": False, "clean_report": False, "path": str(output_path), "reason": reason}


def build_qa_blocker_summary(
    *,
    writer_report: Dict[str, Any],
    evidence_gap_summary: Optional[Dict[str, Any]] = None,
    review_diagnostic: Optional[Dict[str, Any]] = None,
    reformatter_result: Optional[Dict[str, Any]] = None,
    writer_pending_repair_reasons: Iterable[str] = (),
) -> Dict[str, Any]:
    blockers: List[str] = []
    advisory_types: List[str] = []
    qa = as_dict(as_dict(writer_report).get("qa_result"))
    if bool(as_dict(evidence_gap_summary).get("post_qa_repair_failed")):
        blockers.append("post_qa_no_new_evidence_signal")
    if bool(qa.get("repair_required")):
        blockers.append("qa_repair_required")
    blockers.extend(writer_pending_repair_reasons)
    if as_list(qa.get("advisory_followups")):
        advisory_types.append("qa_advisory_followups")
    if as_dict(evidence_gap_summary).get("chapter_gaps"):
        advisory_types.append("evidence_gap")
    checks = as_dict(as_dict(review_diagnostic).get("checks"))
    if int(checks.get("table_validation_warnings") or 0) > 0:
        advisory_types.append("table_validation_warnings")
    if as_dict(reformatter_result).get("status") in {"failed", "repair_required", "validation_failed"}:
        blockers.append("reformatter_failed")
    return {
        "status": "blocked" if blockers else ("advisory" if advisory_types else "passed"),
        "blocker_types": _dedupe_texts(blockers),
        "advisory_types": _dedupe_texts(advisory_types),
    }


def clean_report_blocked_reason(
    *,
    writer_publishable: bool,
    writer_not_ready: bool,
    reformatter_skip_reason: str,
    qa_blocker_summary: Dict[str, Any],
) -> str:
    if writer_not_ready:
        return "writer_not_ready"
    blockers = as_list(as_dict(qa_blocker_summary).get("blocker_types"))
    if blockers:
        return str(blockers[0])
    if not writer_publishable:
        return "writer_not_publishable"
    return str(reformatter_skip_reason or "").strip()


def finalize_public_report(markdown: str) -> str:
    cleaned = sanitize_public_markdown(str(markdown or ""))
    for _ in range(3):
        blockers = find_publication_blockers(cleaned)
        if not blockers:
            break
        blocked_lines = {int(item.get("line") or 0) for item in blockers}
        cleaned = "\n".join(
            line
            for line_no, line in enumerate(cleaned.splitlines(), start=1)
            if line_no not in blocked_lines
        )
        cleaned = sanitize_public_markdown(cleaned)
    remaining = find_publication_blockers(cleaned)
    if remaining:
        sample = "; ".join(str(item.get("text") or "")[:80] for item in remaining[:3])
        raise ValueError(f"publication blockers remain after sanitization: {sample}")
    return cleaned.strip()


def llm_runtime_status() -> Dict[str, Any]:
    try:
        from rag_pipeline.config import search_config as cfg
        from rag_pipeline.search.memory import llm_config_is_ready, normalize_llm_config

        synthesis = normalize_llm_config(
            {
                "provider": cfg.DEFAULT_LLM_SYNTHESIS_PROVIDER,
                "url": cfg.DEFAULT_LLM_SYNTHESIS_URL,
                "api_key": cfg.DEFAULT_LLM_SYNTHESIS_API_KEY,
                "model": cfg.DEFAULT_LLM_SYNTHESIS_MODEL,
                "timeout": cfg.DEFAULT_LLM_SYNTHESIS_TIMEOUT,
                "disable_thinking": getattr(cfg, "DEFAULT_LLM_SYNTHESIS_DISABLE_THINKING", False),
            }
        )
        return {
            "active_profile": os.environ.get("RAG_LLM_ACTIVE_PROFILE", "").strip(),
            "execution_profile": os.environ.get("RAG_LLM_EXECUTION_PROFILE", "").strip()
            or os.environ.get("RAG_LLM_ACTIVE_PROFILE", "").strip(),
            "synthesis_ready": bool(llm_config_is_ready(synthesis)),
            "synthesis_model": synthesis.get("model") or "",
            "synthesis_url_set": bool(synthesis.get("url")),
            "synthesis_api_key_set": bool(synthesis.get("api_key")),
            "synthesis_disable_thinking": bool(getattr(cfg, "DEFAULT_LLM_SYNTHESIS_DISABLE_THINKING", False)),
            "flags": {
                "rag_llm_planner": env_flag("RAG_ENABLE_LLM_PLANNER", False),
                "rag_llm_synthesis": env_flag("RAG_ENABLE_LLM_SYNTHESIS", False),
                "rag_llm_answer_review": env_flag("RAG_ENABLE_LLM_ANSWER_REVIEW", False),
                "rag_llm_reflection": env_flag("RAG_ENABLE_LLM_REFLECTION", False),
                "iqs_llm_query_rewrite": env_flag("IQS_ENABLE_LLM_QUERY_REWRITE", False),
                "iqs_hyde": env_flag("IQS_ENABLE_HYDE", False),
                "brain_llm_research_planner": env_flag("BRAIN_ENABLE_LLM_RESEARCH_PLANNER", False),
                "brain_web_llm_analysis": env_flag("BRAIN_WEB_ENABLE_LLM_ANALYSIS", False),
                "brain_llm_merge": env_flag("BRAIN_ENABLE_LLM_MERGE", False),
                "brain_llm_coverage_eval": env_flag("BRAIN_ENABLE_LLM_COVERAGE_EVAL", False),
                "report_llm_rewrite": env_flag("REPORT_ENABLE_LLM_REWRITE", False),
            },
        }
    except Exception as exc:
        return {"error": str(exc)}


def has_legacy_decision_sections(markdown: str) -> bool:
    return bool(
        re.search(
            r"章节判断|关键事实速览|证据深读|原文事实|行业形势含义|投资/产品判断|与上下章节的联动|战略含义与行动建议|"
            r"全球口径|中国口径|增速口径|可引用事实|机制与边界|进入综合决策章的变量|核心判断[:：]|机制拆解|反证边界|决策含义[:：]",
            str(markdown or ""),
        )
    )


def env_large_int(name: str, default: int, *, min_value: int = 0, max_value: int = 1_000_000) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        value = default
    return min(max_value, max(min_value, value))


def markdown_heading_count(markdown: str) -> int:
    return len(re.findall(r"(?m)^#{1,3}\s+\S+", str(markdown or "")))


def markdown_dense_chars(markdown: str) -> int:
    text = re.sub(r"(?m)^#{1,6}\s+.*$", "", str(markdown or ""))
    return len(re.sub(r"\s+", "", text))


def reformatter_structure_loss_reason(clean_report: str, writer_report: str) -> str:
    clean = str(clean_report or "").strip()
    writer = str(writer_report or "").strip()
    if not clean or not writer:
        return ""
    clean_chars = markdown_dense_chars(clean)
    writer_chars = markdown_dense_chars(writer)
    if writer_chars < 8000:
        return ""
    clean_headings = markdown_heading_count(clean)
    writer_headings = markdown_heading_count(writer)
    target_chars = env_large_int("REPORT_TARGET_BODY_CHARS", 0)
    allow_condense = env_flag("REPORT_REFORMATTER_ALLOW_STRUCTURAL_CONDENSE", True)
    min_ratio_percent = env_large_int("REPORT_REFORMATTER_MIN_DENSE_RATIO_PERCENT", 45, min_value=20, max_value=90)
    min_accept_chars = env_large_int("REPORT_REFORMATTER_MIN_ACCEPT_CHARS", 12000, min_value=2000, max_value=100000)
    if clean_chars < int(writer_chars * (min_ratio_percent / 100.0)) and clean_chars < min_accept_chars:
        return f"正文压缩过多 clean={clean_chars} writer={writer_chars}"
    if not allow_condense and writer_headings >= 10 and clean_headings < max(6, int(writer_headings * 0.55)):
        return f"章节层级丢失 clean_headings={clean_headings} writer_headings={writer_headings}"
    if (
        target_chars
        and not allow_condense
        and writer_chars >= int(target_chars * 0.70)
        and clean_chars < int(target_chars * 0.60)
    ):
        return f"未达到目标正文量 clean={clean_chars} target={target_chars}"
    return ""


def stage_status(state: Dict[str, Any]) -> Dict[str, bool]:
    raw_output = as_dict(state.get("raw_output"))
    writer_report = as_dict(state.get("writer_report")) or as_dict(raw_output.get("writer_report"))
    return {
        "question_analysis": bool(as_dict(state.get("query_analysis")) or as_dict(raw_output.get("query_analysis"))),
        "question_decomposition": bool(
            as_list(as_dict(state.get("query_analysis")).get("related_questions"))
            or as_dict(as_dict(state.get("query_analysis")).get("agent_queries"))
            or as_list(as_dict(raw_output.get("query_analysis")).get("related_questions"))
            or as_dict(as_dict(raw_output.get("query_analysis")).get("agent_queries"))
        ),
        "child_agents": bool(as_dict(raw_output.get("child_outputs"))),
        "evidence_merger": bool(as_dict(state.get("evidence_package")) or as_dict(raw_output.get("evidence_package"))),
        "analysis_agent": bool(as_dict(state.get("structured_analysis")) or as_dict(raw_output.get("structured_analysis"))),
        "writer_agent": bool(writer_report.get("report_markdown")),
    }


def missing_required_stages(status: Dict[str, bool]) -> List[str]:
    required = [
        "question_analysis",
        "question_decomposition",
        "child_agents",
        "evidence_merger",
        "analysis_agent",
        "writer_agent",
    ]
    return [name for name in required if not status.get(name)]


def _compact_error_text(value: Any, *, max_chars: int = 260) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."


def compact_errors(values: Iterable[Any], *, max_items: int = 8) -> List[str]:
    errors: List[str] = []
    for value in values:
        text = _compact_error_text(value)
        if text and text not in errors:
            errors.append(text)
        if len(errors) >= max_items:
            break
    return errors


def env_int(name: str, default: int, *, min_value: int = 1, max_value: int = 100) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        value = default
    return min(max_value, max(min_value, value))


def env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def normalize_llm_profile(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", str(value or "").strip().upper()).strip("_")


def llm_profile_env_name(profile: str, field: str) -> str:
    profile_key = normalize_llm_profile(profile)
    field_key = normalize_llm_profile(field)
    return f"RAG_LLM_PROFILE_{profile_key}_{field_key}" if profile_key and field_key else ""


def available_llm_profiles() -> List[str]:
    raw = os.environ.get("RAG_LLM_PROFILES", "qwen,deepseek-v4-pro")
    profiles = [item.strip() for item in raw.split(",") if item.strip()]
    for current in (
        str(os.environ.get("RAG_LLM_EXECUTION_PROFILE") or "").strip(),
        str(os.environ.get("RAG_LLM_ACTIVE_PROFILE") or "").strip(),
    ):
        if current and current not in profiles:
            profiles.insert(0, current)
    return profiles


def llm_profile_config_status(profile: str) -> Dict[str, str]:
    return {
        "provider": os.environ.get(llm_profile_env_name(profile, "PROVIDER"), "").strip(),
        "url": os.environ.get(llm_profile_env_name(profile, "URL"), "").strip(),
        "api_key": os.environ.get(llm_profile_env_name(profile, "API_KEY"), "").strip(),
        "model": os.environ.get(llm_profile_env_name(profile, "MODEL"), "").strip(),
        "timeout": os.environ.get(llm_profile_env_name(profile, "TIMEOUT"), "").strip(),
        "disable_thinking": os.environ.get(llm_profile_env_name(profile, "DISABLE_THINKING"), "").strip(),
    }


LLM_EXECUTION_PROFILE_ENV = "RAG_LLM_EXECUTION_PROFILE"
LLM_EXECUTION_CONFIG_PREFIX = "RAG_LLM_SYNTHESIS"


def apply_llm_profile_to_environment(profile: str) -> None:
    status = llm_profile_config_status(profile)
    provider = status.get("provider") or "openai_compatible"
    timeout = status.get("timeout") or "180"
    prefix = LLM_EXECUTION_CONFIG_PREFIX
    os.environ[f"{prefix}_PROVIDER"] = provider
    os.environ[f"{prefix}_URL"] = status["url"]
    os.environ[f"{prefix}_API_KEY"] = status["api_key"]
    os.environ[f"{prefix}_MODEL"] = status["model"]
    os.environ[f"{prefix}_TIMEOUT"] = timeout
    disable_thinking_env = f"{prefix}_DISABLE_THINKING"
    if status.get("disable_thinking"):
        os.environ[disable_thinking_env] = status["disable_thinking"]
    else:
        os.environ.pop(disable_thinking_env, None)


def select_llm_profile(args: argparse.Namespace) -> str:
    selected = str(
        args.llm_profile
        or os.environ.get(LLM_EXECUTION_PROFILE_ENV)
        or os.environ.get("RAG_LLM_ACTIVE_PROFILE")
        or ""
    ).strip()
    should_prompt = bool((args.select_llm or env_flag("REPORT_SELECT_LLM_PROFILE", False)) and not args.llm_profile)
    if should_prompt:
        if args.no_interactive_input:
            raise RuntimeError("--select-llm 需要 stdin；请改用 --llm-profile 指定模型。")
        profiles = available_llm_profiles()
        if not profiles:
            raise RuntimeError("RAG_LLM_PROFILES 为空，无法选择执行大模型。")
        print("可选执行大模型：", file=sys.stderr)
        for index, profile in enumerate(profiles, 1):
            status = llm_profile_config_status(profile)
            configured = all(status.get(field) for field in ("url", "api_key", "model"))
            state_tags = []
            if profile == selected:
                state_tags.append("当前默认")
            state_tags.append("已配置" if configured else "未配置")
            suffix = f" ({'，'.join(state_tags)})"
            model = status.get("model") or "model未填"
            print(f"  {index}. {profile} - {model}{suffix}", file=sys.stderr)
        choice = input(f"请选择执行大模型 [默认 {selected or profiles[0]}]：").strip()
        if choice:
            if choice.isdigit() and 1 <= int(choice) <= len(profiles):
                selected = profiles[int(choice) - 1]
            else:
                selected = choice
        elif not selected:
            selected = profiles[0]
    if selected:
        os.environ[LLM_EXECUTION_PROFILE_ENV] = selected
        status = llm_profile_config_status(selected)
        missing = [field for field in ("url", "api_key", "model") if not status.get(field)]
        if missing:
            env_names = ", ".join(llm_profile_env_name(selected, field) for field in missing)
            raise RuntimeError(f"LLM profile '{selected}' 配置不完整，请在 .env 填写：{env_names}")
        apply_llm_profile_to_environment(selected)
    return selected


def strict_quality_mode() -> bool:
    mode = str(os.environ.get("REPORT_QUALITY_MODE") or os.environ.get("QUALITY_MODE") or "strict").strip().lower()
    if mode in {"speed", "fast", "loose", "draft"}:
        return False
    raw = os.environ.get("STRICT_EVIDENCE_MODE")
    if raw is None:
        return True
    return str(raw).strip().lower() in {"1", "true", "yes", "on", "strict"}


HIGH_COST_SEARCH_FLAGS = (
    "IQS_ENABLE_LLM_QUERY_REWRITE",
    "IQS_ENABLE_SELF_REFINE",
    "FULL_REPORT_IQS_ENABLE_SELF_REFINE",
    "BRAIN_AGENT_TEXT_SELF_REFINE",
    "REPORT_CONTINUOUS_EVIDENCE_LOOP",
)


HIGH_EVIDENCE_DEPTH_DEFAULTS = {
    "FULL_REPORT_IQS_MAX_QUERIES": "4",
    "FULL_REPORT_IQS_MAX_SEARCH_TASKS": "32",
    "FULL_REPORT_IQS_RESULTS_PER_QUERY": "80",
    "FULL_REPORT_IQS_RERANK_TOP_K": "36",
    "FULL_REPORT_IQS_RERANK_MAX_DOCS": "100",
    "FULL_REPORT_IQS_RERANK_PREFILTER_MAX_DOCS": "100",
    "BRAIN_INITIAL_LANE_ADAPTIVE_SEARCH_BUDGET": "false",
    "BRAIN_FOLLOWUP_ADAPTIVE_SEARCH_BUDGET": "false",
    "IQS_AUTO_READPAGE_TOP_N": "5",
    "IQS_AUTO_READPAGE_REQUIRED_TOP_N": "8",
    "IQS_AUTO_READPAGE_MIN_SCORE": "0.55",
    "IQS_AUTO_READPAGE_REQUIRED_MIN_SCORE": "0.35",
    "IQS_READPAGE_PARALLEL_WORKERS": "4",
    "READPAGE_FACT_EXTRACTOR_MAX_CALLS_PER_REPORT": "80",
    "READPAGE_FACT_EXTRACTOR_MAX_PAGES_PER_TASK": "6",
    "READPAGE_FACT_EXTRACTOR_MAX_CHARS_PER_PAGE": "9000",
    "BRAIN_LLM_ANALYSIS_MAX_CHAPTERS": "12",
    "BRAIN_LLM_ANALYSIS_MAX_FACTS_PER_CHAPTER": "16",
    "BRAIN_LLM_ANALYSIS_MAX_FACT_CHARS": "420",
    "BRAIN_LLM_ANALYSIS_CONCURRENCY": "4",
    "REPORT_FACTS_PER_CHAPTER_ARGUMENTS": "24",
    "REPORT_CHAPTER_FACT_DIGEST_LIMIT": "16",
}


HIGH_WRITING_QUALITY_DEFAULTS = {
    "BRAIN_ENABLE_LLM_EVIDENCE_ANALYSIS": "true",
    "BRAIN_ENABLE_POST_QA_REPAIR": "true",
    "REPORT_ENABLE_LLM_BODY_REWRITE": "true",
    "REPORT_BODY_REWRITE_MAX_SECTIONS": "24",
    "REPORT_BODY_REWRITE_MAX_ELAPSED_SECONDS": "900",
    "REPORT_BODY_REWRITE_CONCURRENCY": "3",
    "REPORT_BODY_REWRITE_MAX_EXPANSION_RATIO": "5.0",
    "REPORT_BODY_REWRITE_TARGET_SECTION_CHARS": "650",
    "REPORT_ENABLE_LLM_CHAPTER_NARRATIVE": "true",
    "REPORT_CHAPTER_NARRATIVE_MAX_CHAPTERS": "12",
    "REPORT_TARGET_BODY_CHARS": "0",
    "REPORT_TARGET_BODY_CHARS_BLOCKING": "false",
    "REPORT_COMPOSER_TARGET_SECTION_CHARS": "550",
    "REPORT_RENDER_MIN_SECTION_CHARS": "0",
}


DEEPSEEK_QUALITY_MODEL_DEFAULTS = {
    "RAG_MODEL_PLANNING_PROFILE": "deepseek-v4-pro",
    "RAG_MODEL_EVIDENCE_MERGE_PROFILE": "deepseek-v4-pro",
    "RAG_MODEL_COVERAGE_EVAL_PROFILE": "deepseek-v4-pro",
    "RAG_MODEL_RISK_PROFILE": "deepseek-v4-pro",
    "RAG_MODEL_DECISION_PROFILE": "deepseek-v4-pro",
    "RAG_MODEL_REFORMATTER_PROFILE": "deepseek-v4-pro",
    "RAG_MODEL_BODY_REWRITE_PROFILE": "deepseek-v4-pro",
    "RAG_MODEL_QA_PROFILE": "deepseek-v4-pro",
    "RAG_MODEL_REFLECTION_PROFILE": "deepseek-v4-pro",
    "RAG_MODEL_FINAL_AUDIT_PROFILE": "deepseek-v4-pro",
    "READPAGE_FACT_EXTRACTOR_MODEL_PROFILE": "deepseek-v4-pro",
}


QWEN_WEB_SEARCH_MODEL_DEFAULTS = {
    "RAG_MODEL_QUERY_REWRITE_PROFILE": "qwen",
    "RAG_MODEL_WEB_SUMMARY_PROFILE": "qwen",
}


REPORT_MODEL_ROUTING_DEFAULTS = {
    **DEEPSEEK_QUALITY_MODEL_DEFAULTS,
    **QWEN_WEB_SEARCH_MODEL_DEFAULTS,
}


STRICT_RESEARCH_EVIDENCE_DEPTH_DEFAULTS = {
    **HIGH_EVIDENCE_DEPTH_DEFAULTS,
    "FULL_REPORT_IQS_MAX_QUERIES": "6",
    "FULL_REPORT_IQS_MAX_SEARCH_TASKS": "40",
    "FULL_REPORT_IQS_RERANK_TOP_K": "48",
    "FULL_REPORT_IQS_RERANK_MAX_DOCS": "100",
    "FULL_REPORT_IQS_RERANK_PREFILTER_MAX_DOCS": "100",
    "READPAGE_FACT_EXTRACTOR_MAX_CALLS_PER_REPORT": "120",
    "BRAIN_LLM_ANALYSIS_MAX_FACTS_PER_CHAPTER": "20",
}


def _looks_like_removed_openai_gpt_profile(value: Any) -> bool:
    raw = str(value or "").strip().lower()
    if not raw:
        return False
    normalized = raw.replace("_", "-")
    return (
        normalized.startswith("gpt-")
        or normalized.startswith("gpt.")
        or "gpt55" in normalized
        or "api.openai.com" in normalized
    )


def _apply_env_defaults(defaults: Dict[str, str]) -> tuple[Dict[str, str], Dict[str, str]]:
    applied: Dict[str, str] = {}
    preserved: Dict[str, str] = {}
    for key, value in defaults.items():
        if key in os.environ:
            preserved[key] = os.environ[key]
            continue
        os.environ[key] = value
        applied[key] = value
    return applied, preserved


def _apply_model_routing_defaults(defaults: Dict[str, str]) -> tuple[Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, str]]:
    applied: Dict[str, str] = {}
    preserved: Dict[str, str] = {}
    replaced_removed: Dict[str, str] = {}
    overridden: Dict[str, str] = {}
    allow_custom = str(os.environ.get("REPORT_ALLOW_CUSTOM_MODEL_ROUTING") or "").strip().lower() in {"1", "true", "yes", "on"}
    for key, value in defaults.items():
        if key not in os.environ:
            os.environ[key] = value
            applied[key] = value
            continue
        existing = os.environ.get(key, "")
        if str(existing).strip() == value:
            preserved[key] = existing
            continue
        if _looks_like_removed_openai_gpt_profile(existing):
            replaced_removed[key] = existing
            os.environ[key] = value
            applied[key] = value
            continue
        if not allow_custom:
            overridden[key] = existing
            os.environ[key] = value
            applied[key] = value
            continue
        preserved[key] = existing
    return applied, preserved, replaced_removed, overridden


def apply_report_quality_posture(mode: str = "") -> Dict[str, Any]:
    normalized = str(mode or os.environ.get("REPORT_QUALITY_MODE") or "high").strip().lower() or "high"
    if normalized in {"strict", "deep_strict", "due_diligence", "investment_due_diligence"}:
        normalized = "strict_research"
    if normalized not in {"balanced", "high", "strict_research"}:
        normalized = "high"

    if normalized == "strict_research":
        defaults = {
            "IQS_ENABLE_LLM_QUERY_REWRITE": "true",
            "IQS_ENABLE_SELF_REFINE": "true",
            "FULL_REPORT_IQS_ENABLE_SELF_REFINE": "true",
            "BRAIN_AGENT_TEXT_SELF_REFINE": "false",
            "REPORT_CONTINUOUS_EVIDENCE_LOOP": "false",
        }
        evidence_defaults = STRICT_RESEARCH_EVIDENCE_DEPTH_DEFAULTS
    else:
        defaults = {key: "false" for key in HIGH_COST_SEARCH_FLAGS}
        evidence_defaults = HIGH_EVIDENCE_DEPTH_DEFAULTS if normalized == "high" else {}

    writing_defaults = HIGH_WRITING_QUALITY_DEFAULTS if normalized in {"high", "strict_research"} else {}
    applied, preserved = _apply_env_defaults({**defaults, **evidence_defaults, **writing_defaults})
    model_applied, model_preserved, replaced_removed_models, overridden_models = _apply_model_routing_defaults(REPORT_MODEL_ROUTING_DEFAULTS)
    applied.update(model_applied)
    preserved.update(model_preserved)

    disabled = {
        "query_rewrite": str(os.environ.get("IQS_ENABLE_LLM_QUERY_REWRITE") or "").strip().lower()
        not in {"1", "true", "yes", "on"},
        "self_refine": str(os.environ.get("IQS_ENABLE_SELF_REFINE") or "").strip().lower()
        not in {"1", "true", "yes", "on"}
        and str(os.environ.get("FULL_REPORT_IQS_ENABLE_SELF_REFINE") or "").strip().lower()
        not in {"1", "true", "yes", "on"}
        and str(os.environ.get("BRAIN_AGENT_TEXT_SELF_REFINE") or "").strip().lower()
        not in {"1", "true", "yes", "on"},
        "continuous_loop": str(os.environ.get("REPORT_CONTINUOUS_EVIDENCE_LOOP") or "").strip().lower()
        not in {"1", "true", "yes", "on"},
    }
    return {
        "mode": normalized,
        "applied_defaults": applied,
        "preserved_explicit": preserved,
        "disabled": disabled,
        "query_rewrite_max_calls": os.environ.get("QUERY_REWRITE_MAX_CALLS_PER_REPORT", "4"),
        "query_rewrite_max_input_chars": os.environ.get("QUERY_REWRITE_MAX_INPUT_CHARS", "6000"),
        "evidence_depth": {
            "iqs_max_queries": os.environ.get("FULL_REPORT_IQS_MAX_QUERIES", ""),
            "iqs_max_search_tasks": os.environ.get("FULL_REPORT_IQS_MAX_SEARCH_TASKS", ""),
            "readpage_top_n": os.environ.get("IQS_AUTO_READPAGE_TOP_N", ""),
            "readpage_required_top_n": os.environ.get("IQS_AUTO_READPAGE_REQUIRED_TOP_N", ""),
            "fact_extractor_max_calls": os.environ.get("READPAGE_FACT_EXTRACTOR_MAX_CALLS_PER_REPORT", ""),
            "llm_analysis_max_chapters": os.environ.get("BRAIN_LLM_ANALYSIS_MAX_CHAPTERS", ""),
            "llm_analysis_max_facts_per_chapter": os.environ.get("BRAIN_LLM_ANALYSIS_MAX_FACTS_PER_CHAPTER", ""),
            "claim_builder_facts_per_chapter": os.environ.get("REPORT_FACTS_PER_CHAPTER_ARGUMENTS", ""),
        },
        "writing_depth": {
            "body_rewrite_max_sections": os.environ.get("REPORT_BODY_REWRITE_MAX_SECTIONS", ""),
            "body_rewrite_max_elapsed_seconds": os.environ.get("REPORT_BODY_REWRITE_MAX_ELAPSED_SECONDS", ""),
            "chapter_narrative_max_chapters": os.environ.get("REPORT_CHAPTER_NARRATIVE_MAX_CHAPTERS", ""),
            "target_body_chars": os.environ.get("REPORT_TARGET_BODY_CHARS", ""),
            "composer_target_section_chars": os.environ.get("REPORT_COMPOSER_TARGET_SECTION_CHARS", ""),
        },
        "model_routing": {
            key: os.environ.get(key, "")
            for key in REPORT_MODEL_ROUTING_DEFAULTS
        },
        "replaced_removed_model_profiles": replaced_removed_models,
        "overridden_model_profiles": overridden_models,
    }


def _target_body_chars_from_env(value: Any) -> int:
    try:
        return int(float(str(value or "").strip() or 0))
    except (TypeError, ValueError):
        return 0


def resolve_report_quality_mode(mode: str = "", target_body_chars: Any = None) -> str:
    """Resolve the writing posture before any stage reads REPORT_QUALITY_MODE.

    A 20k-body report cannot be produced by the balanced writing path because
    that path intentionally leaves section rewrite and chapter narrative off.
    If the user/environment asks for a long-form body, upgrade the writing
    posture to ``high`` while keeping high-cost search expansion disabled via
    ``apply_report_quality_posture``.
    """

    normalized = str(mode or os.environ.get("REPORT_QUALITY_MODE") or "").strip().lower()
    if normalized in {"strict", "deep_strict", "due_diligence", "investment_due_diligence"}:
        normalized = "strict_research"
    if normalized in {"standard", "default"}:
        normalized = "balanced"
    if normalized not in {"balanced", "high", "strict_research"}:
        normalized = "high"
    target = _target_body_chars_from_env(
        target_body_chars if target_body_chars is not None else os.environ.get("REPORT_TARGET_BODY_CHARS")
    )
    if normalized == "balanced" and target >= 18_000:
        return "high"
    return normalized


def continuous_evidence_loop_mode() -> bool:
    raw = os.environ.get("REPORT_CONTINUOUS_EVIDENCE_LOOP")
    if raw is not None:
        return str(raw).strip().lower() in {"1", "true", "yes", "on", "strict"}
    return strict_quality_mode()


_STATE_LIST_LIMITS = {
    "messages": 2,
    "search_results": 40,
    "page_results": 10,
    "raw_data_points": 120,
    "evidence_pool": 160,
    "evidence": 120,
    "key_sources": 120,
    "search_trace": 80,
    "query_plan": 160,
    "search_tasks": 200,
    "follow_up_queries": 80,
    "followup_results": 80,
    "layout_refinement_trace": 40,
    "self_refine_trace": 40,
    "loop_trace": 40,
}


_STATE_TEXT_LIMITS = {
    "answer_text": 12000,
    "report_markdown": 12000,
    "content": 3000,
    "mainText": 3000,
    "snippet": 2000,
    "summary": 2000,
    "evidence": 6000,
    "raw_output": 6000,
}


def _state_list_limit(key: str) -> int:
    return _STATE_LIST_LIMITS.get(key, env_int("REPORT_STATE_MAX_LIST_ITEMS", 80, max_value=500))


def _state_text_limit(key: str) -> int:
    return _STATE_TEXT_LIMITS.get(key, env_int("REPORT_STATE_MAX_TEXT_CHARS", 6000, max_value=50000))


def compact_state_for_disk(value: Any, *, key: str = "", depth: int = 0) -> Any:
    """Keep debug state useful while avoiding hundreds of MB of repeated raw outputs."""

    if env_flag("REPORT_SAVE_FULL_STATE", False):
        return value
    if depth > 10:
        return "[state compacted: max depth reached]"
    if isinstance(value, dict):
        compacted: Dict[str, Any] = {}
        for item_key, item_value in value.items():
            text_key = str(item_key)
            compacted[text_key] = compact_state_for_disk(item_value, key=text_key, depth=depth + 1)
        return compacted
    if isinstance(value, list):
        limit = _state_list_limit(key)
        kept = [compact_state_for_disk(item, key=key, depth=depth + 1) for item in value[:limit]]
        if len(value) > limit:
            kept.append(
                {
                    "_truncated": True,
                    "original_count": len(value),
                    "kept_count": limit,
                    "hint": "Set REPORT_SAVE_FULL_STATE=1 to keep the complete debug snapshot.",
                }
            )
        return kept
    if isinstance(value, str):
        limit = _state_text_limit(key)
        if len(value) > limit:
            return (
                value[:limit]
                + f"\n\n[state compacted: text truncated from {len(value)} to {limit} chars; "
                + "full report is saved as Markdown and writer_package keeps evidence inputs]"
            )
        return value
    return value


def write_state_json(path: Path, payload: Dict[str, Any]) -> None:
    write_json(path, compact_state_for_disk(payload))


def full_report_iqs_options() -> Dict[str, Any]:
    """Use the active IQS profile instead of forcing the old quality-first funnel."""

    default_max_queries = env_int("IQS_INITIAL_MAX_QUERIES", 1, max_value=10)
    default_max_tasks = env_int("IQS_INITIAL_MAX_SEARCH_TASKS", 2, max_value=40)
    default_results = env_int("IQS_INITIAL_RESULTS_PER_QUERY", 20, max_value=100)
    default_top_k = env_int("IQS_INITIAL_RERANK_TOP_K", 8, max_value=80)
    default_max_docs = env_int("IQS_INITIAL_RERANK_MAX_DOCS", 30, max_value=100)

    options = {
        "search_profile": "initial",
        "max_queries": env_int("FULL_REPORT_IQS_MAX_QUERIES", default_max_queries, max_value=10),
        "max_search_tasks": env_int("FULL_REPORT_IQS_MAX_SEARCH_TASKS", default_max_tasks, max_value=40),
        "results_per_query": env_int("FULL_REPORT_IQS_RESULTS_PER_QUERY", default_results, max_value=100),
        "rerank_top_k": env_int("FULL_REPORT_IQS_RERANK_TOP_K", default_top_k, max_value=80),
        "rerank_max_docs": env_int("FULL_REPORT_IQS_RERANK_MAX_DOCS", default_max_docs, max_value=100),
        "rerank_prefilter_max_docs": env_int("FULL_REPORT_IQS_RERANK_PREFILTER_MAX_DOCS", default_max_docs, max_value=100),
        "enable_self_refine": env_flag("FULL_REPORT_IQS_ENABLE_SELF_REFINE", env_flag("IQS_ENABLE_SELF_REFINE", True)),
        "enable_batch_search": env_flag("IQS_ENABLE_BATCH_SEARCH", True),
    }
    if continuous_evidence_loop_mode():
        floors = {
            "max_queries": 6,
            "max_search_tasks": 24,
            "results_per_query": 80,
            "rerank_top_k": 40,
            "rerank_max_docs": 100,
            "rerank_prefilter_max_docs": 100,
        }
        for key, floor in floors.items():
            options[key] = max(int(options.get(key) or 0), floor)
        options["enable_self_refine"] = True
    return options


TOPIC_CACHE_EVIDENCE_LAYERS = (
    "core_evidence",
    "supporting_evidence",
    "metric_evidence",
    "case_evidence",
    "counter_evidence",
    "directional_evidence",
    "sample_evidence",
)


def _topic_cache_chapter_hydrated_count(packages: List[Dict[str, Any]]) -> int:
    total = 0
    for package in packages or []:
        if not isinstance(package, dict):
            continue
        for key in TOPIC_CACHE_EVIDENCE_LAYERS:
            total += len([item for item in as_list(package.get(key)) if isinstance(item, dict)])
    return total


def _topic_cache_blueprint_from_chapters(query: str, chapter_evidence_packages: List[Dict[str, Any]]) -> Dict[str, Any]:
    chapters: List[Dict[str, Any]] = []
    for index, package in enumerate(chapter_evidence_packages or [], start=1):
        payload = as_dict(package)
        chapter_id = str(payload.get("chapter_id") or f"ch_{index:02d}")
        title = str(payload.get("chapter_title") or payload.get("chapter_question") or "").strip()
        if not title:
            title = f"核心问题 {index}"
        chapters.append(
            {
                "chapter_id": chapter_id,
                "chapter_title": title,
                "chapter_question": str(payload.get("chapter_question") or title),
            }
        )
    if not chapters:
        chapters = [{"chapter_id": "ch_01", "chapter_title": "核心观察", "chapter_question": str(query or "核心观察")}]
    return {
        "report_family": "industry_deep_report",
        "report_type": "industry_deep_report",
        "chapters": chapters,
        "report_shell": {
            "front_blocks": ["executive_summary"],
            "back_blocks": ["risk_triggers", "appendix"],
        },
    }


def _topic_cache_preflight_for_query(query: str) -> Dict[str, Any]:
    try:
        from rag_pipeline.cache.topic_bundle_cache import load_topic_bundle, preflight_topic_bundle

        loaded = load_topic_bundle(query)
        preflight = preflight_topic_bundle(loaded, query=query)
        return {"load": loaded, "preflight": preflight}
    except Exception as exc:  # pragma: no cover - cache must never block live report runs.
        return {"load": {"enabled": True, "found": False, "error": str(exc)}, "preflight": {"status": "error", "error": str(exc), "can_skip_search": False}}


def _topic_cache_seed_for_brain(cache_context: Dict[str, Any]) -> Dict[str, Any]:
    preflight = as_dict(cache_context.get("preflight"))
    if not bool(preflight.get("can_seed_evidence") or preflight.get("seedable")):
        return {}
    if str(preflight.get("status") or "").strip().lower() in {"polluted", "summary_only", "incompatible", "missing", "disabled"}:
        return {}
    try:
        from rag_pipeline.cache.topic_bundle_cache import bundle_to_writer_inputs

        loaded = as_dict(cache_context.get("load"))
        inputs = bundle_to_writer_inputs(loaded, preflight=preflight, reuse_analysis=False)
        seed_evidence = as_list(inputs.get("seed_evidence"))
        if not seed_evidence:
            return {}
        return {
            "enabled": True,
            "topic_key": inputs.get("topic_key") or preflight.get("topic_key"),
            "path": inputs.get("path") or preflight.get("path"),
            "preflight": preflight,
            "seed_evidence": seed_evidence,
            "seed_evidence_count": len(seed_evidence),
            "source_registry": as_list(inputs.get("source_registry")),
            "evidence_package_summary": {
                "analysis_ready_count": as_dict(preflight.get("completeness")).get("analysis_ready_count"),
                "full_evidence_count": as_dict(preflight.get("completeness")).get("full_evidence_count"),
            },
        }
    except Exception as exc:  # pragma: no cover - cache must never block live report runs.
        return {"enabled": True, "seed_error": str(exc), "preflight": preflight}


def _run_topic_bundle_cached_flow(query: str, cache_context: Dict[str, Any]) -> Dict[str, Any]:
    """Rebuild report artifacts from a usable topic bundle without live search."""

    from rag_pipeline.agents.analysis_agent import run_analysis_agent
    from rag_pipeline.agents.chapter_evidence_builder import build_chapter_evidence_packages_from_evidence_package
    from rag_pipeline.agents.claim_builder_agent import run_claim_builder_agent
    from rag_pipeline.agents.chapter_argument_agent import run_chapter_argument_agent
    from rag_pipeline.agents.final_writer_agent import run_final_writer_agent
    from rag_pipeline.cache.topic_bundle_cache import bundle_to_writer_inputs

    loaded = as_dict(cache_context.get("load"))
    preflight = as_dict(cache_context.get("preflight"))
    inputs = bundle_to_writer_inputs(loaded, preflight=preflight)
    evidence_package = dict(as_dict(inputs.get("evidence_package")))
    evidence_package.setdefault("metadata", {})
    evidence_package["metadata"]["topic_bundle_cache_hit"] = {
        "status": preflight.get("status"),
        "topic_key": inputs.get("topic_key"),
        "path": inputs.get("path"),
        "skip_search": True,
    }
    source_registry = merge_source_registry_candidates(
        as_list(evidence_package.get("source_registry")),
        as_list(evidence_package.get("sources")),
        as_list(inputs.get("source_registry")),
    )
    if source_registry and not as_list(evidence_package.get("source_registry")):
        evidence_package["source_registry"] = source_registry
    report_blueprint = as_dict(inputs.get("report_blueprint"))
    existing_chapter_packages = as_list(inputs.get("chapter_evidence_packages"))
    if not report_blueprint:
        report_blueprint = _topic_cache_blueprint_from_chapters(query, existing_chapter_packages)

    rebuilt_chapter_packages = build_chapter_evidence_packages_from_evidence_package(
        report_blueprint=report_blueprint,
        evidence_package=evidence_package,
        existing_chapter_evidence_packages=existing_chapter_packages,
        source_registry=source_registry,
    )
    if rebuilt_chapter_packages and _topic_cache_chapter_hydrated_count(rebuilt_chapter_packages) >= _topic_cache_chapter_hydrated_count(existing_chapter_packages):
        chapter_evidence_packages = rebuilt_chapter_packages
    else:
        chapter_evidence_packages = existing_chapter_packages
    evidence_package["chapter_evidence_packages"] = chapter_evidence_packages

    structured_analysis = as_dict(inputs.get("structured_analysis"))
    analysis_state: Dict[str, Any] = {}
    if not structured_analysis:
        analysis_state = run_analysis_agent(evidence_package, query=query)
        structured_analysis = as_dict(analysis_state.get("structured_analysis"))
    else:
        analysis_state = {"structured_analysis": structured_analysis, "errors": [], "metadata": {"source": "topic_bundle_cache"}}

    micro_layouts = as_list(inputs.get("micro_layouts"))
    table_packages = as_list(inputs.get("table_packages"))
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
        query=query,
        report_blueprint=report_blueprint,
        chapter_packages=chapter_packages,
        table_packages=table_packages,
        decision_package={},
        risk_package={},
        appendix_package={
            "metric_normalization_table": as_list(evidence_package.get("metric_normalization_table")),
        },
        source_registry=source_registry,
        evidence_package=evidence_package,
        chapter_evidence_packages=chapter_evidence_packages,
        claim_units=argument_units,
        analysis_claim_units=as_list(as_dict(structured_analysis).get("claim_units")),
        analysis_stage_diagnostics=as_dict(as_dict(structured_analysis).get("analysis_stage_diagnostics")),
    )
    report_markdown = str(writer_output.get("report_markdown") or "").strip()
    writer_report = {
        **writer_output,
        "report_status": writer_output.get("report_status") or "formal_scored",
        "delivery_tier": "topic_bundle_cache",
        "clean_report_eligible": False,
        "topic_bundle_cache": {
            "hit": True,
            "used_for_skip_search": True,
            "topic_key": inputs.get("topic_key"),
            "path": inputs.get("path"),
            "preflight": preflight,
            "analysis_rebuild_required": bool(inputs.get("analysis_rebuild_required")),
        },
        "chapter_evidence_packages": chapter_evidence_packages,
        "micro_layouts": micro_layouts,
        "table_packages": table_packages,
        "argument_units": argument_units,
        "chapter_packages": chapter_packages,
        "source_registry": writer_output.get("source_registry") or source_registry,
    }
    raw_output = {
        "query": query,
        "route": "topic_bundle_cache",
        "route_reason": "usable topic evidence bundle cache",
        "query_analysis": {
            "topic_cache_hit": True,
            "related_questions": [query],
            "agent_queries": {"topic_bundle_cache": query},
            "research_plan": {"source": "topic_bundle_cache"},
        },
        "child_outputs": {
            "topic_bundle_cache": {
                "answer": "topic bundle cache hit; live retrieval skipped",
                "topic_key": inputs.get("topic_key"),
                "path": inputs.get("path"),
            }
        },
        "evidence_package": evidence_package,
        "structured_analysis": structured_analysis,
        "writer_report": writer_report,
        "chapter_evidence_packages": chapter_evidence_packages,
        "micro_layouts": micro_layouts,
        "table_packages": table_packages,
        "argument_units": argument_units,
        "chapter_packages": chapter_packages,
        "analysis_errors": as_list(analysis_state.get("errors")),
        "writer_errors": [],
        "output_mode": "writer_markdown",
        "payload_mode": "topic_bundle_cache",
    }
    return {
        "answer_text": report_markdown,
        "raw_output": raw_output,
        "query_analysis": raw_output["query_analysis"],
        "evidence_package": evidence_package,
        "structured_analysis": structured_analysis,
        "report_blueprint": report_blueprint,
        "source_registry": writer_report.get("source_registry") or [],
        "chapter_evidence_packages": chapter_evidence_packages,
        "micro_layouts": micro_layouts,
        "table_packages": table_packages,
        "argument_units": argument_units,
        "chapter_packages": chapter_packages,
        "writer_report": writer_report,
        "errors": [f"TopicBundleCache: {item}" for item in as_list(analysis_state.get("errors"))],
        "metadata": {
            "agent_stage": "topic_bundle_cache_rebuild",
            "topic_bundle_cache": writer_report["topic_bundle_cache"],
        },
    }


def build_full_report_timeout_context(*, started_at: Optional[float] = None) -> Dict[str, Any]:
    max_seconds = env_int("FULL_REPORT_MAX_WALL_SECONDS", 1200, min_value=0, max_value=86400)
    started = float(started_at if started_at is not None else time.perf_counter())
    return {
        "enabled": max_seconds > 0,
        "fail_open_on_timeout": env_flag("FULL_REPORT_FAIL_OPEN_ON_TIMEOUT", True),
        "min_stage": str(os.getenv("FULL_REPORT_TIMEOUT_MIN_STAGE") or "evidence_package").strip() or "evidence_package",
        "started_at": started,
        "deadline_ts": started + max_seconds if max_seconds > 0 else 0.0,
        "max_seconds": max_seconds,
    }


def timeout_context_triggered(timeout_context: Dict[str, Any], *, now: Optional[float] = None) -> bool:
    context = as_dict(timeout_context)
    if not bool(context.get("enabled")):
        return False
    deadline = float(context.get("deadline_ts") or 0.0)
    if deadline <= 0:
        return False
    return float(now if now is not None else time.perf_counter()) >= deadline


def _timeout_metadata(timeout_context: Dict[str, Any], *, stage: str, partial_artifact: str = "") -> Dict[str, Any]:
    context = as_dict(timeout_context)
    return {
        "live_deadline_seconds": int(context.get("max_seconds") or 0),
        "timeout_triggered": True,
        "timeout_stage": stage,
        "fail_open_path_used": False,
        "partial_artifact_used": partial_artifact,
    }


def _state_evidence_package_for_fail_open(state: Dict[str, Any]) -> Dict[str, Any]:
    state_dict = as_dict(state)
    raw_output = as_dict(state_dict.get("raw_output"))
    writer_report = as_dict(state_dict.get("writer_report")) or as_dict(raw_output.get("writer_report"))
    render_artifacts = as_dict(writer_report.get("render_artifacts"))
    return (
        as_dict(render_artifacts.get("evidence_package"))
        or as_dict(state_dict.get("evidence_package"))
        or as_dict(raw_output.get("evidence_package"))
    )


def _run_fail_open_rebuild_from_package(
    *,
    query: str,
    state: Dict[str, Any],
    timeout_context: Dict[str, Any],
) -> Dict[str, Any]:
    from rag_pipeline.agents.analysis_agent import run_analysis_agent
    from rag_pipeline.agents.chapter_argument_agent import run_chapter_argument_agent
    from rag_pipeline.agents.chapter_evidence_builder import build_chapter_evidence_packages_from_evidence_package
    from rag_pipeline.agents.claim_builder_agent import run_claim_builder_agent
    from rag_pipeline.agents.final_writer_agent import run_final_writer_agent
    from rag_pipeline.agents.micro_layout_agent import run_micro_layout_agent

    state_dict = as_dict(state)
    raw_output = as_dict(state_dict.get("raw_output"))
    writer_report = as_dict(state_dict.get("writer_report")) or as_dict(raw_output.get("writer_report"))
    render_artifacts = as_dict(writer_report.get("render_artifacts"))
    evidence_package = dict(_state_evidence_package_for_fail_open(state_dict))
    source_registry = merge_source_registry_candidates(
        as_list(evidence_package.get("source_registry")),
        as_list(evidence_package.get("sources")),
        as_list(state_dict.get("source_registry")),
        as_list(render_artifacts.get("source_registry")),
    )
    report_blueprint = (
        as_dict(render_artifacts.get("report_blueprint"))
        or as_dict(state_dict.get("report_blueprint"))
        or as_dict(raw_output.get("report_blueprint"))
        or as_dict(evidence_package.get("report_blueprint"))
        or as_dict(as_dict(evidence_package.get("metadata")).get("report_blueprint"))
        or as_dict(evidence_package.get("report_plan"))
        or as_dict(as_dict(evidence_package.get("metadata")).get("report_plan"))
    )
    existing_chapters = (
        as_list(render_artifacts.get("chapter_evidence_packages"))
        or as_list(writer_report.get("chapter_evidence_packages"))
        or as_list(evidence_package.get("chapter_evidence_packages"))
        or as_list(state_dict.get("chapter_evidence_packages"))
    )
    chapter_evidence_packages = build_chapter_evidence_packages_from_evidence_package(
        report_blueprint=report_blueprint,
        evidence_package=evidence_package,
        existing_chapter_evidence_packages=existing_chapters,
        source_registry=source_registry,
    )
    evidence_package["chapter_evidence_packages"] = chapter_evidence_packages
    allow_fail_open_llm = env_flag("FULL_REPORT_FAIL_OPEN_ALLOW_LLM_ANALYSIS", False)
    previous_analysis_llm_flag = os.environ.get("BRAIN_ENABLE_LLM_EVIDENCE_ANALYSIS")
    os.environ["BRAIN_ENABLE_LLM_EVIDENCE_ANALYSIS"] = "true" if allow_fail_open_llm else "false"
    try:
        llm_config: Dict[str, Any] = {}
        if allow_fail_open_llm:
            try:
                from rag_pipeline.config.search_config import build_llm_config_for_task

                llm_config = dict(build_llm_config_for_task("decision"))
            except Exception:
                llm_config = {}
        analysis_state = run_analysis_agent(evidence_package, query=query, llm_config=llm_config)
    finally:
        if previous_analysis_llm_flag is None:
            os.environ.pop("BRAIN_ENABLE_LLM_EVIDENCE_ANALYSIS", None)
        else:
            os.environ["BRAIN_ENABLE_LLM_EVIDENCE_ANALYSIS"] = previous_analysis_llm_flag
    structured_analysis = as_dict(analysis_state.get("structured_analysis"))
    micro_layouts = as_list(render_artifacts.get("micro_layouts")) or run_micro_layout_agent(
        report_blueprint=report_blueprint,
        chapter_evidence_packages=chapter_evidence_packages,
        structured_analysis=structured_analysis,
    )
    table_packages = as_list(render_artifacts.get("table_packages"))
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
        query=query,
        report_blueprint=report_blueprint,
        chapter_packages=chapter_packages,
        table_packages=table_packages,
        decision_package={},
        risk_package={},
        appendix_package={},
        source_registry=source_registry,
        evidence_package=evidence_package,
        chapter_evidence_packages=chapter_evidence_packages,
        claim_units=argument_units,
        analysis_claim_units=as_list(as_dict(structured_analysis).get("claim_units")),
        analysis_stage_diagnostics=as_dict(as_dict(structured_analysis).get("analysis_stage_diagnostics")),
    )
    live_timeout = {
        **_timeout_metadata(timeout_context, stage=str(as_dict(timeout_context).get("timeout_stage") or "brain"), partial_artifact="evidence_package"),
        "fail_open_path_used": True,
    }
    report_markdown = str(writer_output.get("report_markdown") or "").strip()
    rendered_chapter_count = report_markdown.count("\n## ")
    if len(report_markdown) < 300 or rendered_chapter_count <= 0:
        diagnostic = {
            **live_timeout,
            "fail_open_error": "fail_open_rebuild_no_renderable_sections",
            "chapter_evidence_package_count": len(chapter_evidence_packages),
            "argument_unit_count": len(as_list(argument_units)),
            "chapter_package_count": len(as_list(chapter_packages)),
        }
        diagnostic_markdown = "\n".join(
            [
                "# Timeout diagnostic",
                "",
                "The live run reached the wall-clock deadline and fail-open rebuild could not create a readable formal report.",
                "",
                f"- chapter_evidence_package_count: {len(chapter_evidence_packages)}",
                f"- argument_unit_count: {len(as_list(argument_units))}",
                f"- chapter_package_count: {len(as_list(chapter_packages))}",
                f"- reason: {diagnostic['fail_open_error']}",
            ]
        )
        return {
            **state_dict,
            "answer_text": "",
            "evidence_package": evidence_package,
            "structured_analysis": structured_analysis,
            "chapter_evidence_packages": chapter_evidence_packages,
            "micro_layouts": micro_layouts,
            "table_packages": table_packages,
            "argument_units": argument_units,
            "chapter_packages": chapter_packages,
            "writer_report": {
                "report_status": "diagnostic_only",
                "delivery_tier": "timeout_diagnostic_only",
                "report_execution_mode": "timeout_fail_open",
                "quality_mode": False,
                "quality_score": 0,
                "clean_report_eligible": False,
                "live_timeout": diagnostic,
                "diagnostic_markdown": diagnostic_markdown,
            },
            "raw_output": {
                **raw_output,
                "evidence_package": evidence_package,
                "structured_analysis": structured_analysis,
                "chapter_evidence_packages": chapter_evidence_packages,
                "micro_layouts": micro_layouts,
                "table_packages": table_packages,
                "argument_units": argument_units,
                "chapter_packages": chapter_packages,
                "writer_report": {
                    "report_status": "diagnostic_only",
                    "delivery_tier": "timeout_diagnostic_only",
                    "report_execution_mode": "timeout_fail_open",
                    "quality_mode": False,
                    "live_timeout": diagnostic,
                    "diagnostic_markdown": diagnostic_markdown,
                },
                "analysis_errors": as_list(analysis_state.get("errors")),
                "output_mode": "writer_markdown",
                "payload_mode": "timeout_diagnostic_only",
            },
            "metadata": {
                **as_dict(state_dict.get("metadata")),
                "agent_stage": "timeout_fail_open_rebuild",
                "live_timeout": diagnostic,
            },
        }
    rebuilt_writer_report = {
        **writer_output,
        "report_status": writer_output.get("report_status") or "formal_scored",
        "delivery_tier": writer_output.get("delivery_tier") or "timeout_fail_open",
        "report_execution_mode": "timeout_fail_open",
        "quality_mode": False,
        "clean_report_eligible": False,
        "live_timeout": live_timeout,
        "chapter_evidence_packages": chapter_evidence_packages,
        "micro_layouts": micro_layouts,
        "table_packages": table_packages,
        "argument_units": argument_units,
        "chapter_packages": chapter_packages,
        "source_registry": writer_output.get("source_registry") or source_registry,
    }
    return {
        **state_dict,
        "answer_text": str(writer_output.get("report_markdown") or "").strip(),
        "evidence_package": evidence_package,
        "structured_analysis": structured_analysis,
        "chapter_evidence_packages": chapter_evidence_packages,
        "micro_layouts": micro_layouts,
        "table_packages": table_packages,
        "argument_units": argument_units,
        "chapter_packages": chapter_packages,
        "writer_report": rebuilt_writer_report,
        "raw_output": {
            **raw_output,
            "evidence_package": evidence_package,
            "structured_analysis": structured_analysis,
            "chapter_evidence_packages": chapter_evidence_packages,
            "micro_layouts": micro_layouts,
            "table_packages": table_packages,
            "argument_units": argument_units,
            "chapter_packages": chapter_packages,
            "writer_report": rebuilt_writer_report,
            "analysis_errors": as_list(analysis_state.get("errors")),
            "output_mode": "writer_markdown",
            "payload_mode": "timeout_fail_open",
        },
        "metadata": {
            **as_dict(state_dict.get("metadata")),
            "agent_stage": "timeout_fail_open_rebuild",
            "live_timeout": live_timeout,
        },
    }


def run_fail_open_rebuild_from_state(
    *,
    query: str,
    state: Dict[str, Any],
    timeout_context: Dict[str, Any],
) -> Dict[str, Any]:
    evidence_package = _state_evidence_package_for_fail_open(state)
    if evidence_package:
        try:
            rebuilt = _run_fail_open_rebuild_from_package(query=query, state=state, timeout_context=timeout_context)
            writer_report = as_dict(rebuilt.get("writer_report"))
            if "live_timeout" not in writer_report:
                live_timeout = {
                    **_timeout_metadata(timeout_context, stage=str(as_dict(timeout_context).get("timeout_stage") or "brain"), partial_artifact="evidence_package"),
                    "fail_open_path_used": True,
                }
                writer_report = {**writer_report, "live_timeout": live_timeout}
                rebuilt = {
                    **rebuilt,
                    "writer_report": writer_report,
                    "metadata": {**as_dict(rebuilt.get("metadata")), "live_timeout": live_timeout},
                }
            return rebuilt
        except Exception as exc:
            diagnostic = _timeout_metadata(timeout_context, stage=str(as_dict(timeout_context).get("timeout_stage") or "fail_open_rebuild"), partial_artifact="evidence_package")
            diagnostic["fail_open_error"] = str(exc)
            state = {**as_dict(state)}
            state["writer_report"] = {
                **as_dict(state.get("writer_report")),
                "report_status": "diagnostic_only",
                "quality_score": 0,
                "live_timeout": diagnostic,
                "diagnostic_markdown": f"# Timeout diagnostic\n\nFail-open rebuild failed: {exc}",
            }
            state["answer_text"] = ""
            return state
    diagnostic = _timeout_metadata(timeout_context, stage=str(as_dict(timeout_context).get("timeout_stage") or "brain"), partial_artifact="raw_search")
    return {
        **as_dict(state),
        "answer_text": "",
        "writer_report": {
            "report_status": "diagnostic_only",
            "delivery_tier": "timeout_diagnostic_only",
            "quality_score": 0,
            "live_timeout": diagnostic,
            "diagnostic_markdown": "# Timeout diagnostic\n\nThe live run reached the wall-clock deadline before an evidence package was available.",
        },
        "metadata": {
            **as_dict(as_dict(state).get("metadata")),
            "live_timeout": diagnostic,
        },
    }


def default_full_report_route() -> str:
    route = str(os.getenv("FULL_REPORT_DEFAULT_ROUTE") or os.getenv("BRAIN_AGENT_ROUTE") or "web").strip().lower()
    if route in {"all", "both", "auto", "web", "local"}:
        return route
    return "web"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full industry-research flow: question analysis, decomposition, "
            "RAG/IQS agents, Evidence Merger, Analysis Agent, and Writer Agent report generation."
        )
    )
    parser.add_argument("query", nargs="*", help="报告问题/主题，例如：智能机器人行业投资机会")
    parser.add_argument("--query", dest="query_option", default="", help="报告问题/主题。")
    parser.add_argument("--route", choices=["all", "both", "auto", "web", "local"], default=default_full_report_route(), help="默认 web，只调度 IQS/联网证据；如需本地 RAG 可显式设置 local/both/all 并启用 BRAIN_ENABLE_LOCAL_RAG。")
    parser.add_argument("--llm-profile", default="", help="选择本次报告执行大模型 profile，例如 qwen、deepseek-v4-pro。")
    parser.add_argument("--select-llm", action="store_true", help="运行前在终端列出 RAG_LLM_PROFILES 并交互选择执行大模型。")
    parser.add_argument("--output-dir", default=str(PIPELINE_ROOT / "output" / "full_reports"), help="状态和调试文件输出目录。")
    parser.add_argument("--session-id", default="", help="可选 session id。")
    parser.add_argument("--supervisor-max-loops", type=int, default=env_int("BRAIN_SUPERVISOR_MAX_LOOPS", 1, max_value=5), help="默认 3 轮补证，优先保证深度行研覆盖。")
    parser.add_argument("--supervisor-max-followup-queries", type=int, default=env_int("BRAIN_SUPERVISOR_MAX_FOLLOWUP_QUERIES", 2, max_value=10), help="每轮最多补充问题数。")
    parser.add_argument("--supervisor-min-coverage-gain", type=float, default=0.05, help="补证覆盖率提升阈值。")
    parser.add_argument("--include-raw-child-states", action="store_true", help="额外保存 RAG/IQS 原始状态，文件会更大。")
    parser.add_argument("--save-full-state", action="store_true", help="保存完整未压缩 state JSON；只建议定位深层调试问题时使用。")
    parser.add_argument("--no-interactive-input", action="store_true", help="没有传 query 时直接失败，不进入终端输入等待。")
    parser.add_argument("--print-report", action="store_true", help="兼容旧参数；报告正文现在默认输出到 stdout，不再生成 md 文件。")
    parser.add_argument("--skip-review", action="store_true", help="跳过 ReviewAgent 终审；默认启用规则审查。")
    parser.add_argument(
        "--enable-llm-review",
        dest="enable_llm_review",
        action="store_true",
        default=env_flag("REPORT_ENABLE_LLM_REVIEW", True),
        help="启用 ReviewAgent 的 LLM 精修层；默认启用。",
    )
    parser.add_argument("--disable-llm-review", dest="enable_llm_review", action="store_false", help="关闭 ReviewAgent 的 LLM 精修层。")
    parser.add_argument("--skip-reformatter", action="store_true", help="跳过 ReformatterAgent，回退到旧 WriterAgent/ReviewAgent 输出路径。")
    parser.add_argument("--reformatter-output", default="", help="可选：指定 ReformatterAgent 洁净报告输出路径。")
    parser.add_argument("--no-progress-bar", action="store_true", help="关闭整体进度条，恢复普通阶段日志。")
    parser.add_argument("--verbose-progress", action="store_true", help="保留内部详细进度日志；默认只显示整体进度条。")
    parser.add_argument("--allow-missing-stage", action="store_true", help="即使阶段产物缺失也返回 0；默认会失败退出。")
    return parser

def main() -> int:
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    load_dotenv(PIPELINE_ROOT / ".env")
    configure_pipeline_logging()
    resolved_quality_mode = resolve_report_quality_mode(
        os.getenv("REPORT_QUALITY_MODE", ""),
        os.getenv("REPORT_TARGET_BODY_CHARS"),
    )
    os.environ["REPORT_QUALITY_MODE"] = resolved_quality_mode
    quality_posture = apply_report_quality_posture(resolved_quality_mode)
    args = build_arg_parser().parse_args()
    report_quality_mode = resolved_quality_mode
    high_quality_mode = report_quality_mode in {"high", "strict_research"}
    if high_quality_mode:
        _apply_env_defaults(HIGH_WRITING_QUALITY_DEFAULTS)
    selected_llm_profile = select_llm_profile(args)
    progress_enabled = (not args.no_progress_bar) and env_flag("REPORT_PROGRESS_BAR", True)
    global QUIET_STAGE_LOGS
    QUIET_STAGE_LOGS = bool(progress_enabled and not args.verbose_progress)
    if args.verbose_progress:
        os.environ["PIPELINE_PROGRESS"] = "1"
    else:
        os.environ["PIPELINE_PROGRESS"] = "0"
    progress = OverallProgress(enabled=progress_enabled)
    progress.update(1, "准备参数")
    if selected_llm_profile:
        log(f"[0/6] 执行大模型 profile: {selected_llm_profile}")

    if args.save_full_state:
        os.environ["REPORT_SAVE_FULL_STATE"] = "1"
    query = (args.query_option or " ".join(args.query)).strip()
    if not query:
        if args.no_interactive_input or env_flag("REPORT_NO_INTERACTIVE_INPUT", False):
            raise RuntimeError("Query cannot be empty. 请使用 --query 传入报告问题。")
        log("[0/6] 未检测到 --query，等待你在终端输入报告问题/主题；也可以用 --query \"你的问题\" 直接启动")
        try:
            query = input("请输入报告问题/主题：").strip()
        except EOFError as exc:
            raise RuntimeError("Query cannot be empty. 当前运行环境没有可用 stdin，请使用 --query 传入报告问题。") from exc
    if not query:
        raise RuntimeError("Query cannot be empty.")

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{run_timestamp}_{safe_filename(query)}"
    stage_snapshot_index: List[Dict[str, Any]] = []
    artifact_ledger_store, artifact_ledger_status = init_artifact_ledger_run_safe(
        run_id=run_id,
        query=query,
        report_type="full_report",
        freshness_policy={
            "default_mode": "run_scoped",
            "current_query_terms": ["news", "policy", "finance", "funding", "current", "latest"],
        },
    )

    def record_stage_snapshot(stage_name: str, payload: Any, *, summary: Optional[Dict[str, Any]] = None, diagnostics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        result = write_stage_snapshot_safe(
            run_id=run_id,
            stage_name=stage_name,
            payload=payload,
            summary=summary,
            diagnostics=diagnostics,
        )
        artifact_ledger_artifact = record_stage_snapshot_artifact_safe(
            artifact_ledger_store,
            run_id=run_id,
            stage_name=stage_name,
            payload=payload,
            snapshot_result=result,
        )
        if result.get("enabled") is not False:
            snapshot_entry = {
                "stage_name": stage_name,
                "stored": bool(result.get("stored")),
                "replayable": bool(result.get("replayable")),
                "manifest_path": str(Path(str(result.get("full_payload_path") or "")).with_name("manifest.json")) if result.get("full_payload_path") else "",
                "full_payload_path": result.get("full_payload_path") or "",
                "reason": result.get("reason") or "",
                "error": result.get("error") or "",
            }
            if artifact_ledger_artifact:
                snapshot_entry["artifact_ledger"] = artifact_ledger_artifact
            stage_snapshot_index.append(
                snapshot_entry
            )
        return result

    pipeline_started = time.perf_counter()
    timeout_context = build_full_report_timeout_context(started_at=pipeline_started)
    topic_cache_context = _topic_cache_preflight_for_query(query)
    topic_cache_preflight = as_dict(topic_cache_context.get("preflight"))
    topic_cache_seed = _topic_cache_seed_for_brain(topic_cache_context)
    topic_cache_skip_search = bool(
        topic_cache_preflight.get("status") == "usable"
        and topic_cache_preflight.get("usable_for_skip_search")
        and topic_cache_preflight.get("can_skip_search")
    )
    if topic_cache_preflight.get("status") not in {"missing", "disabled"}:
        log(
            f"[cache] topic_bundle status={topic_cache_preflight.get('status')} "
            f"key={topic_cache_preflight.get('topic_key') or ''} "
            f"seed={bool(topic_cache_seed.get('seed_evidence_count'))} "
            f"skip_search={topic_cache_skip_search}"
        )
    progress.update(5, "问题分析与任务规划")
    log("[1/6] 问题分析与拆解启动")
    if args.route in {"local", "both", "all"} and env_flag("BRAIN_ENABLE_LOCAL_RAG", False):
        log("[2/6] 调度本地 RAG 与 IQS Agent")
    else:
        log("[2/6] 调度 IQS 联网证据 Agent（本地 RAG 默认关闭）")
    log("[3/6] Evidence Merger / Analysis Agent / Writer Agent 将在 merge 阶段串行执行")

    progress.pulse_to(72, "检索 / 证据 / 正文生成")
    state: Dict[str, Any] = {}
    if topic_cache_skip_search:
        log("[cache] usable topic bundle found; skipping live IQS/Qwen-assisted retrieval and rebuilding from cached evidence")
        try:
            state = _run_topic_bundle_cached_flow(query, topic_cache_context)
        except Exception as cache_rebuild_exc:
            log(f"[cache] topic bundle rebuild failed; falling back to live retrieval: {cache_rebuild_exc}")
            topic_cache_skip_search = False
            topic_cache_preflight = {
                **topic_cache_preflight,
                "status": "cache_rebuild_failed",
                "can_skip_search": False,
                "error": str(cache_rebuild_exc),
            }
    if not topic_cache_skip_search:
        from rag_pipeline.agents.brain_agent import run_brain_agent

        continuous_loop = continuous_evidence_loop_mode()
        previous_snapshot_run_id = os.environ.get("REPORT_STAGE_SNAPSHOT_RUN_ID")
        os.environ["REPORT_STAGE_SNAPSHOT_RUN_ID"] = run_id
        try:
            state = run_brain_agent(
                query=query,
                route=args.route,
                session_id=args.session_id,
                web_search_options=full_report_iqs_options(),
                enable_web_analysis=env_flag("BRAIN_WEB_ENABLE_LLM_ANALYSIS", False),
                enable_llm_merge=env_flag("BRAIN_ENABLE_LLM_MERGE", False),
                enable_followup_loop=continuous_loop or env_flag("BRAIN_ENABLE_FOLLOWUP_LOOP", False),
                supervisor_max_loops=max(args.supervisor_max_loops, 5) if continuous_loop else args.supervisor_max_loops,
                supervisor_min_coverage_gain=args.supervisor_min_coverage_gain,
                supervisor_max_followup_queries=max(args.supervisor_max_followup_queries, 8) if continuous_loop else args.supervisor_max_followup_queries,
                layout_max_refinement_rounds=max(env_int("BRAIN_LAYOUT_MAX_REFINEMENT_ROUNDS", 3, max_value=6), 3) if continuous_loop else None,
                output_mode="writer_markdown",
                parallel_raw_output=bool(args.include_raw_child_states),
                topic_bundle_seed=topic_cache_seed if topic_cache_seed.get("seed_evidence_count") else None,
                deadline_ts=timeout_context.get("deadline_ts") if timeout_context.get("enabled") else None,
                timeout_context=timeout_context if timeout_context.get("enabled") else None,
                fail_open_on_timeout=bool(timeout_context.get("fail_open_on_timeout", True)),
            )
        finally:
            try:
                from rag_pipeline.agents.readpage_fact_extractor_agent import reset_budget as reset_readpage_fact_extractor_budget

                reset_readpage_fact_extractor_budget(run_id)
            except Exception as budget_reset_exc:
                log(f"[WARN] Readpage Fact Extractor budget reset failed: {budget_reset_exc}")
            if previous_snapshot_run_id is None:
                os.environ.pop("REPORT_STAGE_SNAPSHOT_RUN_ID", None)
            else:
                os.environ["REPORT_STAGE_SNAPSHOT_RUN_ID"] = previous_snapshot_run_id
    state_probe = as_dict(state)
    raw_probe = as_dict(state_probe.get("raw_output"))
    writer_probe = as_dict(state_probe.get("writer_report")) or as_dict(raw_probe.get("writer_report"))
    answer_probe = str(state_probe.get("answer_text") or "").strip()
    live_timeout_probe = as_dict(state_probe.get("live_timeout")) or as_dict(as_dict(state_probe.get("metadata")).get("live_timeout")) or as_dict(raw_probe.get("live_timeout"))
    deadline_partial_probe = str(raw_probe.get("payload_mode") or "").strip() == "deadline_partial"
    auto_failure_answer = answer_probe.startswith("大脑 Agent 失败")
    writer_missing_probe = not str(writer_probe.get("report_markdown") or "").strip()
    if (
        timeout_context_triggered(timeout_context)
        and bool(timeout_context.get("fail_open_on_timeout", True))
        and (deadline_partial_probe or live_timeout_probe or writer_missing_probe or auto_failure_answer)
    ):
        log("[timeout] full report deadline reached; entering fail-open rebuild from available artifacts")
        timeout_stage = str(live_timeout_probe.get("timeout_stage") or "full_report_main")
        state = run_fail_open_rebuild_from_state(query=query, state=state, timeout_context={**timeout_context, "timeout_stage": timeout_stage})
    state.setdefault("metadata", {})
    state["metadata"]["topic_bundle_cache_preflight"] = topic_cache_preflight
    state["topic_bundle_cache_preflight"] = topic_cache_preflight
    state["stage_snapshot_run_id"] = run_id
    progress.update(72, "主体报告生成完成")
    log(f"[3/6] Brain 主流程完成，用时 {time.perf_counter() - pipeline_started:.1f}s")

    state_dict = dict(state or {})
    raw_output = as_dict(state_dict.get("raw_output"))
    writer_report = as_dict(state_dict.get("writer_report")) or as_dict(raw_output.get("writer_report"))
    report_execution_mode = "live_quality_full" if str(os.getenv("REPORT_QUALITY_MODE") or "").strip().lower() == "high" else "live_standard"
    if as_dict(writer_report.get("live_timeout")) or as_dict(state_dict.get("live_timeout")) or as_dict(raw_output.get("live_timeout")):
        report_execution_mode = "timeout_fail_open"
    writer_report["report_execution_mode"] = writer_report.get("report_execution_mode") or report_execution_mode
    writer_report["quality_mode"] = bool(writer_report.get("quality_mode") or report_execution_mode == "live_quality_full")
    writer_report["quality_posture"] = writer_report.get("quality_posture") or quality_posture
    state_dict["writer_report"] = writer_report
    record_stage_snapshot(
        "research_plan",
        as_dict(state_dict.get("research_plan"))
        or as_dict(as_dict(state_dict.get("query_analysis")).get("research_plan"))
        or as_dict(as_dict(raw_output.get("query_analysis")).get("research_plan")),
    )
    record_stage_snapshot(
        "search_task_schedule",
        as_dict(writer_report.get("search_task_schedule")) or as_dict(state_dict.get("search_task_schedule")),
    )
    record_stage_snapshot(
        "iqs_results",
        {
            "agent_queries": state_dict.get("agent_queries"),
            "lane_coverage": writer_report.get("lane_coverage") or state_dict.get("lane_coverage"),
            "search_task_schedule": writer_report.get("search_task_schedule") or state_dict.get("search_task_schedule"),
            "topic_bundle_cache_preflight": topic_cache_preflight,
        },
    )
    report_markdown = str(writer_report.get("report_markdown") or state_dict.get("answer_text") or "").strip()
    writer_status = str(writer_report.get("report_status") or "").strip().lower()
    delivery_tier = str(writer_report.get("delivery_tier") or as_dict(as_dict(state_dict.get("evidence_package")).get("summary")).get("delivery_tier") or "").strip()
    writer_not_ready = writer_status in {"not_ready", "diagnostic_only"} and not report_markdown
    writer_publishable = bool(report_markdown) and writer_status in {"final", "final_clean"} and not writer_not_ready
    formal_report_available = bool(report_markdown) and not writer_not_ready
    emit_clean_report = env_flag("REPORT_WRITE_CLEAN_REPORT", False)
    if not report_markdown:
        reformatter_skip_reason = "no_report_markdown"
    elif writer_not_ready:
        reformatter_skip_reason = "writer_not_ready"
    elif not writer_publishable:
        reformatter_skip_reason = f"report_status_{writer_status or 'unknown'}"
    elif not emit_clean_report:
        reformatter_skip_reason = "clean_report_output_disabled"
    elif args.skip_reformatter:
        reformatter_skip_reason = "skip_reformatter_arg"
    else:
        reformatter_skip_reason = ""
    review_result: Dict[str, Any] = {}
    reformatter_result: Dict[str, Any] = {}
    last_clean_evidence: Dict[str, Any] = {}

    review_applied_to_formal_report = False
    review_mode = ""
    if report_markdown and formal_report_available and not args.skip_review and (args.skip_reformatter or not writer_publishable):
        from .review_pipeline import run_review_pipeline_sync

        progress.pulse_to(82, "ReviewAgent 审查")
        log("[5/6] ReviewAgent 审查报告中")
        light_formal_review = not writer_publishable
        review_result = run_review_pipeline_sync(
            writer_output=report_markdown,
            llm_client=None,
            skip_llm_review=light_formal_review or not args.enable_llm_review,
        )
        report_markdown = finalize_public_report(str(review_result.get("final_report") or report_markdown))
        writer_report["report_markdown"] = report_markdown
        writer_report["review_audit"] = as_dict(review_result.get("stage1_audit"))
        writer_report["review_stage2_skipped"] = bool(review_result.get("stage2_skipped", True))
        writer_report["review_total_fixes"] = int(review_result.get("total_fixes") or 0)
        writer_report["review_agent_applied_to_formal_report"] = True
        writer_report["review_agent_mode"] = "deterministic_formal" if light_formal_review else "publishable_review"
        writer_report = attach_structured_review_to_writer_report(writer_report, review_result)
        review_pending_repair_reasons = review_result_pending_repair_reasons(review_result)
        if "review_evidence_required" in review_pending_repair_reasons:
            reformatter_skip_reason = "review_evidence_required"
        elif "review_rewrite_required" in review_pending_repair_reasons:
            reformatter_skip_reason = "review_rewrite_required"
        review_applied_to_formal_report = True
        review_mode = str(writer_report.get("review_agent_mode") or "")
        state_dict["writer_report"] = writer_report
        state_dict["answer_text"] = report_markdown

        audit = as_dict(review_result.get("stage1_audit"))
        log(f"  [ReviewAgent] 修复泄露文本: {len(as_list(audit.get('leak_patterns_removed')))} 处")
        log(f"  [ReviewAgent] 删除重复 bullet: {int(audit.get('duplicate_bullets_removed') or 0)} 处")
        log(f"  [ReviewAgent] 删除重复段落: {int(audit.get('duplicate_paragraphs_removed') or 0)} 处")
        log(f"  [ReviewAgent] 修复/填充空节: {len(as_list(audit.get('empty_sections_filled')))} 处")
        if review_result.get("stage2_skipped"):
            log(f"  [ReviewAgent] LLM 精修跳过: {review_result.get('stage2_reason') or 'not enabled'}")
        if as_list(audit.get("truncated_content")):
            log(f"  [WARN] 截断/无意义内容: {len(as_list(audit.get('truncated_content')))} 处")
        progress.update(82, "ReviewAgent 审查完成")

    progress.update(86, "写入报告文件")
    timestamp = run_timestamp
    output_dir = Path(args.output_dir).resolve()
    base_name = f"{timestamp}_{safe_filename(query)}"
    state_path = output_dir / f"{base_name}.state.json"
    package_path = output_dir / f"{base_name}.writer_package.json"
    writer_md_path = output_dir / f"{base_name}.writer.md"
    formal_report_md_path = output_dir / f"{base_name}_report.md"
    score_report_md_path = output_dir / f"{base_name}_score.md"
    review_draft_md_path = output_dir / f"{base_name}_review_draft.md"
    diagnostic_md_path = output_dir / f"{base_name}.diagnostic.md"

    state_dict["writer_package_path"] = str(package_path)
    if formal_report_available:
        _insufficient_action = (
            _insufficient_analysis_delivery_action(report_markdown, writer_report)
            if env_flag("REPORT_INSUFFICIENT_STUB_ON_ZERO_CLAIMS", True)
            else {"mode": "normal", "replace_with_stub": False}
        )
        if _insufficient_action.get("replace_with_stub"):
            # P0 guardrail: analysis produced no usable LLM claims -> emit a short
            # honest stub instead of a fluent-but-vacuous deterministic long report.
            report_markdown = _build_insufficient_stub_markdown(
                query, writer_report, as_dict(_insufficient_action.get("diagnostics"))
            )
            writer_report["report_status"] = "insufficient_analysis_stub"
            writer_report["delivery_tier"] = "insufficient_analysis_stub"
            writer_report["insufficient_analysis_stub"] = True
            writer_report["insufficient_analysis_delivery"] = _insufficient_action
            state_dict["answer_text"] = report_markdown
            write_markdown(formal_report_md_path, report_markdown)
            log("  [P0] 分析无有效 claim → 输出诚实短稿（非模板长文）", force=True)
        else:
            report_markdown, writer_report = finalize_formal_report_and_refresh_audit(report_markdown, writer_report)
            if _insufficient_action.get("mode") == "limited_evidence_formal_report":
                writer_report["report_status"] = "formal_scored"
                writer_report["delivery_tier"] = "limited_evidence_formal_report"
                writer_report["limited_evidence_formal_report"] = True
                writer_report["insufficient_analysis_stub"] = False
                writer_report["insufficient_analysis_delivery"] = _insufficient_action
            state_dict["answer_text"] = report_markdown
            write_formal_markdown(formal_report_md_path, report_markdown)
        writer_report["formal_report_path"] = str(formal_report_md_path)
        writer_report["writer_markdown_path"] = str(formal_report_md_path)
        state_dict["formal_report_path"] = str(formal_report_md_path)
        state_dict["writer_markdown_path"] = str(formal_report_md_path)
        state_dict["writer_report"] = writer_report
    diagnostic_markdown = str(writer_report.get("diagnostic_markdown") or writer_report.get("blocked_report_markdown") or "").strip()
    if writer_not_ready and diagnostic_markdown:
        write_markdown(diagnostic_md_path, finalize_public_report(diagnostic_markdown))
        writer_report["diagnostic_markdown_path"] = str(diagnostic_md_path)
        state_dict["diagnostic_markdown_path"] = str(diagnostic_md_path)
        state_dict["writer_report"] = writer_report
    write_state_json(state_path, state_dict)
    llm_status = llm_runtime_status()
    reformatter_result = {
        "enabled": bool(emit_clean_report and writer_publishable and not args.skip_reformatter),
        "status": "pending" if emit_clean_report and writer_publishable and not args.skip_reformatter else "skipped",
        "skipped_reason": reformatter_skip_reason,
        "llm_runtime": llm_status,
    }
    render_artifacts = as_dict(writer_report.get("render_artifacts"))
    evidence_package_payload = (
        as_dict(render_artifacts.get("evidence_package"))
        or as_dict(state_dict.get("evidence_package"))
        or as_dict(raw_output.get("evidence_package"))
    )
    evidence_health_summary = (
        as_dict(evidence_package_payload.get("evidence_health_summary"))
        or as_dict(as_dict(evidence_package_payload.get("summary")).get("evidence_health_summary"))
        or as_dict(as_dict(evidence_package_payload.get("metadata")).get("evidence_health_summary"))
    )
    source_registry_payload = merge_source_registry_candidates(
        as_list(evidence_package_payload.get("source_registry")),
        as_list(evidence_package_payload.get("sources")),
        as_list(as_dict(state_dict.get("writer_report")).get("source_registry")),
        as_list(render_artifacts.get("source_registry")),
    )
    chapter_evidence_payload = (
        as_list(render_artifacts.get("chapter_evidence_packages"))
        or as_list(writer_report.get("chapter_evidence_packages"))
        or as_list(raw_output.get("chapter_evidence_packages"))
        or as_list(as_dict(raw_output.get("evidence_package")).get("chapter_evidence_packages"))
        or as_list(as_dict(state_dict.get("evidence_package")).get("chapter_evidence_packages"))
        or as_list(state_dict.get("chapter_evidence_packages"))
    )
    if chapter_evidence_payload and not as_list(evidence_package_payload.get("chapter_evidence_packages")):
        evidence_package_payload["chapter_evidence_packages"] = chapter_evidence_payload
    writer_package_payload = {
        "query": query,
        "report_execution_mode": writer_report.get("report_execution_mode") or report_execution_mode,
        "quality_mode": bool(writer_report.get("quality_mode")),
        "quality_posture": quality_posture,
        "stage_status": stage_status(state_dict),
        "llm_runtime": llm_status,
        "evidence_package": evidence_package_payload,
        "evidence_health_summary": evidence_health_summary,
        "source_registry": source_registry_payload,
        "structured_analysis": as_dict(render_artifacts.get("structured_analysis"))
        or as_dict(state_dict.get("structured_analysis"))
        or as_dict(raw_output.get("structured_analysis")),
        "report_blueprint": as_dict(state_dict.get("report_blueprint")) or as_dict(raw_output.get("report_blueprint")),
        "chapter_evidence_packages": chapter_evidence_payload,
        "micro_layouts": as_list(render_artifacts.get("micro_layouts")) or as_list(state_dict.get("micro_layouts")) or as_list(raw_output.get("micro_layouts")),
        "table_packages": as_list(render_artifacts.get("table_packages")) or as_list(state_dict.get("table_packages")) or as_list(raw_output.get("table_packages")),
        "argument_units": as_list(render_artifacts.get("argument_units")) or as_list(state_dict.get("argument_units")) or as_list(raw_output.get("argument_units")),
        "chapter_packages": as_list(render_artifacts.get("chapter_packages")) or as_list(state_dict.get("chapter_packages")) or as_list(raw_output.get("chapter_packages")),
        "citation_manifest": as_dict(writer_report.get("citation_manifest")) or as_dict(render_artifacts.get("citation_manifest")),
        "writer_report": writer_report,
        "review_result": review_result,
        "reformatter_result": reformatter_result,
        **repair_trace_payload_from_state(
            state_dict=state_dict,
            raw_output=raw_output,
            writer_report=writer_report,
        ),
        "topic_bundle_cache": {
            "preflight": topic_cache_preflight,
            "hit": bool(topic_cache_preflight.get("status") not in {"missing", "disabled"}),
            "used_for_skip_search": bool(topic_cache_skip_search),
            "seedable": bool(topic_cache_preflight.get("seedable") or topic_cache_preflight.get("can_seed_evidence")),
            "summary_only": bool(topic_cache_preflight.get("summary_only")),
            "seed_evidence_count": int(topic_cache_seed.get("seed_evidence_count") or 0),
            "usable_for_skip_search": bool(topic_cache_preflight.get("usable_for_skip_search")),
        },
    }
    for stage_name, payload in (
        ("evidence_package", evidence_package_payload),
        ("chapter_evidence_packages", chapter_evidence_payload),
        ("structured_analysis", writer_package_payload.get("structured_analysis")),
        ("argument_units", writer_package_payload.get("argument_units")),
        ("chapter_packages", writer_package_payload.get("chapter_packages")),
        ("table_packages", writer_package_payload.get("table_packages")),
        ("writer_report", writer_report),
        ("qa_result", as_dict(writer_report.get("qa_result"))),
    ):
        record_stage_snapshot(stage_name, payload)
    writer_package_payload["stage_snapshot_run_id"] = run_id
    writer_package_payload["stage_snapshot_index"] = list(stage_snapshot_index)
    writer_report["stage_snapshot_run_id"] = run_id
    writer_report["stage_snapshot_index"] = list(stage_snapshot_index)
    state_dict["stage_snapshot_run_id"] = run_id
    artifact_ledger_sync = sync_artifact_ledger_package_safe(
        artifact_ledger_store,
        run_id=run_id,
        writer_package=writer_package_payload,
        writer_report=writer_report,
    )
    if artifact_ledger_sync:
        artifact_ledger_status.update(artifact_ledger_sync)
    if artifact_ledger_status:
        writer_package_payload["artifact_ledger"] = dict(artifact_ledger_status)
        writer_report["artifact_ledger"] = dict(artifact_ledger_status)
        state_dict["artifact_ledger"] = dict(artifact_ledger_status)

    # [cache_report] Phase 0 观测：写一个只读 sidecar，聚合各缓存命中/规模，便于阶段间 diff。
    # 完全 fail-open，不修改任何既有 payload；关闭用 CACHE_REPORT_SIDECAR_ENABLED=0。
    if env_flag("CACHE_REPORT_SIDECAR_ENABLED", True):
        try:
            from rag_pipeline.cache.cache_report import write_cache_report

            _cache_report = write_cache_report(
                run_id,
                output_dir,
                base_name=base_name,
                query=query,
                stage_snapshot_index=stage_snapshot_index,
                topic_bundle=writer_package_payload.get("topic_bundle_cache"),
            )
            log(f"[cache] cache_report -> {_cache_report.get('_path') or '(write skipped)'}")
        except Exception as _cache_report_exc:
            log(f"[WARN] cache_report 生成失败（不影响报告交付）: {_cache_report_exc}")

    def refresh_llm_call_trace() -> None:
        calls: List[Dict[str, Any]] = []

        def add_call(stage: str, payload: Any) -> None:
            call = as_dict(payload)
            if not call:
                return
            normalized = {"stage": stage, **call}
            key = (
                str(normalized.get("stage") or ""),
                str(normalized.get("task") or ""),
                str(normalized.get("profile") or ""),
                str(normalized.get("model") or ""),
                str(normalized.get("api") or ""),
                str(normalized.get("status") or ""),
                str(normalized.get("error") or ""),
            )
            existing = {
                (
                    str(item.get("stage") or ""),
                    str(item.get("task") or ""),
                    str(item.get("profile") or ""),
                    str(item.get("model") or ""),
                    str(item.get("api") or ""),
                    str(item.get("status") or ""),
                    str(item.get("error") or ""),
                )
                for item in calls
            }
            if key not in existing:
                calls.append(normalized)

        research_plan = as_dict(state_dict.get("research_plan")) or as_dict(as_dict(state_dict.get("query_analysis")).get("research_plan"))
        add_call("planning", research_plan.get("planner_llm_call"))
        coverage_payload = as_dict(state_dict.get("coverage_evaluation")) or as_dict(raw_output.get("coverage_evaluation"))
        add_call("coverage_eval", coverage_payload.get("llm_call"))
        for trace_item in as_list(state_dict.get("loop_trace")) + as_list(raw_output.get("loop_trace")):
            add_call("coverage_eval", as_dict(trace_item).get("llm_call"))
        add_call("review_stage2", as_dict(as_dict(writer_package_payload.get("review_result")).get("structured_review")).get("llm_call"))
        add_call("qa", as_dict(writer_report.get("qa_result")).get("llm_call"))
        add_call("final_audit", as_dict(writer_package_payload.get("final_audit_result")).get("llm_call"))
        writer_package_payload["llm_call_trace"] = calls

    def write_writer_package() -> None:
        refresh_llm_call_trace()
        writer_package_payload["stage_snapshot_run_id"] = run_id
        writer_package_payload["stage_snapshot_index"] = list(stage_snapshot_index)
        as_dict(writer_package_payload.get("writer_report"))["stage_snapshot_index"] = list(stage_snapshot_index)
        attach_token_usage_snapshot(writer_package_payload)
        write_json(package_path, writer_package_payload)

    write_writer_package()

    status = stage_status(state_dict)
    missing = missing_required_stages(status)
    errors = compact_errors(as_list(state_dict.get("errors")) + as_list(raw_output.get("writer_errors")) + as_list(raw_output.get("analysis_errors")))

    progress.update(90, "阶段产物校验")
    log("[4/6] 阶段校验")
    for name, ok in status.items():
        log(f"  - {name}: {'OK' if ok else 'MISSING'}")

    if not writer_publishable or not emit_clean_report:
        reformatter_result = {
            **reformatter_result,
            "enabled": False,
            "success": False,
            "status": "skipped",
            "skipped_reason": reformatter_skip_reason,
        }
        writer_package_payload["reformatter_result"] = reformatter_result
        write_writer_package()
        if not emit_clean_report and formal_report_available:
            log("[5/6] Clean/Reformatter 默认关闭，已保留正式报告并将审查写入评分文件")
            progress.update(96, "已生成正式报告")
        elif writer_not_ready:
            log("[5/6] 证据门槛未达成，跳过 ReformatterAgent，只输出补证清单")
            progress.update(96, "已阻断正式报告")
        elif formal_report_available:
            log(f"[5/6] WriterAgent 状态为 {writer_status or 'unknown'}，跳过 ReformatterAgent，已保留评分正式报告")
            progress.update(96, "已生成评分正式报告")
        else:
            log(f"[5/6] WriterAgent 状态为 {writer_status or 'unknown'}，跳过 ReformatterAgent，保留 review draft")
            progress.update(96, "保留待复核草稿")
    elif not args.skip_reformatter:
        progress.pulse_to(96, "ReformatterAgent 清洗报告")
        log("[5/6] ReformatterAgent 从 writer_package 重写洁净报告")
        clean_output_path = Path(args.reformatter_output).resolve() if args.reformatter_output else package_path.with_name(package_path.name.replace(".writer_package.json", "_clean.md"))
        fallback_output_path = package_path.with_name(package_path.name.replace(".writer_package.json", "_fallback_writer.md"))
        try:
            from .evidence_extractor import extract_clean_evidence
            from .reformatter_agent import build_reformatter_repair_plan, run_reformatter, validate_reformatted_report

            writer_v3_markdown = str(writer_report.get("report_markdown") or report_markdown or "").strip()
            fallback_output_path = package_path.with_name(package_path.name.replace(".writer_package.json", "_fallback_writer.md"))
            reformatter_result = {
                **reformatter_result,
                "enabled": True,
                "success": False,
                "status": "started",
                "output_path": str(clean_output_path),
            }
            writer_package_payload["reformatter_result"] = reformatter_result
            write_writer_package()
            clean_evidence = extract_clean_evidence(str(package_path))
            last_clean_evidence = clean_evidence
            clean_report = asyncio.run(run_reformatter(clean_evidence, llm_client=None))
            fallback_reason = ""
            if has_legacy_decision_sections(clean_report) and writer_v3_markdown and not has_legacy_decision_sections(writer_v3_markdown):
                log("  [WARN] ReformatterAgent 输出含固定小节模板，自动回退到 WriterAgent 动态正文")
                fallback_reason = "legacy_decision_sections"
                clean_report = writer_v3_markdown
            structure_loss_reason = reformatter_structure_loss_reason(clean_report, writer_v3_markdown)
            if structure_loss_reason:
                log(f"  [WARN] ReformatterAgent 输出丢失篇幅或章节结构，自动回退到 WriterAgent 动态正文: {structure_loss_reason}")
                fallback_reason = structure_loss_reason
                clean_report = writer_v3_markdown
            clean_report = finalize_public_report(clean_report)
            validation = validate_reformatted_report(clean_report, as_list(clean_evidence.get("sources")), clean_evidence)
            repair_plan = build_reformatter_repair_plan(validation, clean_evidence, topic=query)
            repair_required = str(repair_plan.get("status") or "passed") != "passed"
            if (not validation.get("passed") or repair_required) and writer_v3_markdown:
                if repair_required and validation.get("passed"):
                    log(f"  [WARN] ReformatterAgent 仍需补正，已阻断 Clean report 写出: {repair_plan}")
                    fallback_reason = fallback_reason or "reformatter_repair_required"
                else:
                    log(f"  [WARN] ReformatterAgent 校验未通过，已阻断 Clean report 写出: {validation}")
                    fallback_reason = fallback_reason or "reformatter_validation_failed"
            if not validation.get("passed") or repair_required:
                fallback_report = ""
                fallback_validation: Dict[str, Any] = {}
                fallback_output_written = False
                if writer_v3_markdown:
                    fallback_report = finalize_public_report(writer_v3_markdown)
                    fallback_validation = validate_reformatted_report(fallback_report, as_list(clean_evidence.get("sources")), clean_evidence)
                    fallback_repair_plan = build_reformatter_repair_plan(fallback_validation, clean_evidence, topic=query)
                    fallback_output_path.parent.mkdir(parents=True, exist_ok=True)
                    write_markdown(fallback_output_path, fallback_report)
                    report_markdown = fallback_report.strip()
                    writer_report["fallback_output_path"] = str(fallback_output_path)
                    writer_report["reformatter_failed_validation"] = validation
                    writer_report["reformatter_validation"] = fallback_validation
                    writer_report["report_markdown"] = report_markdown
                    state_dict["writer_report"] = writer_report
                    state_dict["answer_text"] = report_markdown
                    fallback_output_written = True
                reformatter_result = {
                    "enabled": True,
                    "success": False,
                    "status": "fallback_writer" if fallback_output_written else ("repair_required" if repair_required else "validation_failed"),
                    "output_path": str(clean_output_path),
                    "output_written": False,
                    "fallback_output_path": str(fallback_output_path) if fallback_output_written else "",
                    "fallback_output_written": fallback_output_written,
                    "fallback_draft_path": str(fallback_output_path) if fallback_output_written else "",
                    "fallback_draft_written": fallback_output_written,
                    "validation": fallback_validation if fallback_output_written else validation,
                    "reformatter_validation": validation,
                    "fallback_validation": fallback_validation,
                    "repair_plan": fallback_repair_plan if fallback_output_written else repair_plan,
                    "reformatter_repair_plan": repair_plan,
                    "llm_runtime": llm_status,
                    "clean_evidence_count": int(as_dict(clean_evidence.get("metadata")).get("evidence_count") or 0),
                    "clean_body_chars_without_sources": int((fallback_validation if fallback_output_written else validation).get("body_chars_without_sources") or 0),
                    "clean_body_citation_count": int((fallback_validation if fallback_output_written else validation).get("citation_count") or 0),
                    "clean_body_unique_source_count": int((fallback_validation if fallback_output_written else validation).get("unique_cited_source_count") or 0),
                    "fallback_to_writer": fallback_output_written,
                    "fallback_reason": fallback_reason or ("reformatter_repair_required" if repair_required else "reformatter_validation_failed"),
                }
                writer_report.setdefault("reformatter_validation", validation)
                state_dict["writer_report"] = writer_report
                writer_package_payload["writer_report"] = writer_report
                writer_package_payload["reformatter_result"] = reformatter_result
                write_state_json(state_path, state_dict)
                write_writer_package()
                if fallback_output_written:
                    log(f"  [WARN] ReformatterAgent 未达到 clean 标准，已写出 Writer 回退报告: {fallback_output_path}")
                    progress.update(96, "ReformatterAgent 已回退")
                else:
                    log(f"  [WARN] ReformatterAgent 未达到 clean 标准，未写出 Clean report: {validation}")
                    progress.update(96, "ReformatterAgent 需补正")
            elif fallback_reason:
                fallback_output_path.parent.mkdir(parents=True, exist_ok=True)
                write_markdown(fallback_output_path, clean_report)

                report_markdown = clean_report.strip()
                writer_report["fallback_output_path"] = str(fallback_output_path)
                writer_report["reformatter_validation"] = validation
                writer_report["report_markdown"] = report_markdown
                state_dict["writer_report"] = writer_report
                state_dict["answer_text"] = report_markdown
                reformatter_result = {
                    "enabled": True,
                    "success": False,
                    "status": "fallback_writer",
                    "output_path": str(clean_output_path),
                    "output_written": False,
                    "fallback_output_path": str(fallback_output_path),
                    "fallback_output_written": True,
                    "fallback_draft_path": str(fallback_output_path),
                    "fallback_draft_written": True,
                    "validation": validation,
                    "repair_plan": repair_plan,
                    "llm_runtime": llm_status,
                    "clean_evidence_count": int(as_dict(clean_evidence.get("metadata")).get("evidence_count") or 0),
                    "clean_body_chars_without_sources": int(validation.get("body_chars_without_sources") or 0),
                    "clean_body_citation_count": int(validation.get("citation_count") or 0),
                    "clean_body_unique_source_count": int(validation.get("unique_cited_source_count") or 0),
                    "fallback_to_writer": True,
                    "fallback_reason": fallback_reason,
                }
                writer_package_payload["writer_report"] = writer_report
                writer_package_payload["reformatter_result"] = reformatter_result
                write_state_json(state_path, state_dict)
                write_writer_package()
                log(f"  [WARN] ReformatterAgent 回退 Writer 报告，未写出 Clean report: {fallback_output_path}")
                progress.update(96, "ReformatterAgent 已回退")
            else:
                clean_output_path.parent.mkdir(parents=True, exist_ok=True)
                write_markdown(clean_output_path, clean_report)

                report_markdown = clean_report.strip()
                writer_report["reformatter_output_path"] = str(clean_output_path)
                writer_report["reformatter_validation"] = validation
                writer_report["report_markdown"] = report_markdown
                state_dict["writer_report"] = writer_report
                state_dict["answer_text"] = report_markdown
                reformatter_result = {
                    "enabled": True,
                    "success": True,
                    "status": "completed",
                    "output_path": str(clean_output_path),
                    "output_written": True,
                    "validation": validation,
                    "repair_plan": repair_plan,
                    "llm_runtime": llm_status,
                    "clean_evidence_count": int(as_dict(clean_evidence.get("metadata")).get("evidence_count") or 0),
                    "clean_body_chars_without_sources": int(validation.get("body_chars_without_sources") or 0),
                    "clean_body_citation_count": int(validation.get("citation_count") or 0),
                    "clean_body_unique_source_count": int(validation.get("unique_cited_source_count") or 0),
                    "fallback_to_writer": bool(fallback_reason),
                    "fallback_reason": fallback_reason,
                }
                writer_package_payload["writer_report"] = writer_report
                writer_package_payload["reformatter_result"] = reformatter_result
                write_state_json(state_path, state_dict)
                write_writer_package()
                log(f"  - Clean report Markdown: {clean_output_path}")
                progress.update(96, "ReformatterAgent 清洗完成")
        except Exception as exc:
            context_budget_diagnostic = as_dict(getattr(exc, "diagnostic", {}))
            context_budget_blocked = str(context_budget_diagnostic.get("type") or "") == "reformatter_payload_budget"
            reformatter_result = {
                **reformatter_result,
                "enabled": True,
                "success": False,
                "status": "skipped_context_budget" if context_budget_blocked else "failed",
                "output_path": str(clean_output_path),
                "output_written": False,
                "error": str(exc),
            }
            if context_budget_diagnostic:
                reformatter_result["context_budget"] = context_budget_diagnostic
                reformatter_result["skipped_reason"] = (
                    "reformatter_context_too_large" if context_budget_blocked else "reformatter_exception_with_diagnostic"
                )
            writer_package_payload["reformatter_result"] = reformatter_result
            write_writer_package()
            log(f"  [WARN] ReformatterAgent 失败，保留 WriterAgent 输出: {exc}")
            if report_markdown and not args.skip_review:
                from .review_pipeline import run_review_pipeline_sync

                log("  [Fallback] ReviewAgent 审查 WriterAgent 报告中")
                review_result = run_review_pipeline_sync(
                    writer_output=report_markdown,
                    llm_client=None,
                    skip_llm_review=not args.enable_llm_review,
                )
                report_markdown = finalize_public_report(str(review_result.get("final_report") or report_markdown))
                writer_report["report_markdown"] = report_markdown
                writer_report["review_audit"] = as_dict(review_result.get("stage1_audit"))
                writer_report["review_stage2_skipped"] = bool(review_result.get("stage2_skipped", True))
                writer_report["review_total_fixes"] = int(review_result.get("total_fixes") or 0)
                writer_report = attach_structured_review_to_writer_report(writer_report, review_result)
                review_pending_repair_reasons = review_result_pending_repair_reasons(review_result)
                if "review_evidence_required" in review_pending_repair_reasons:
                    reformatter_result["skipped_reason"] = "review_evidence_required"
                elif "review_rewrite_required" in review_pending_repair_reasons:
                    reformatter_result["skipped_reason"] = "review_rewrite_required"
                state_dict["writer_report"] = writer_report
                state_dict["answer_text"] = report_markdown
                write_markdown(writer_md_path, report_markdown)
                writer_report["writer_markdown_path"] = str(writer_md_path)
                state_dict["writer_markdown_path"] = str(writer_md_path)
                writer_package_payload["writer_report"] = writer_report
                writer_package_payload["review_result"] = review_result
                write_state_json(state_path, state_dict)
                write_writer_package()
            if report_markdown:
                report_markdown = finalize_public_report(report_markdown)
                fallback_output_written = False
                fallback_write_error = ""
                fallback_output_path = package_path.with_name(package_path.name.replace(".writer_package.json", "_fallback_writer.md"))
                try:
                    fallback_output_path.parent.mkdir(parents=True, exist_ok=True)
                    write_markdown(fallback_output_path, report_markdown)
                    fallback_output_written = True
                except Exception as fallback_exc:  # pragma: no cover - filesystem edge case.
                    fallback_write_error = str(fallback_exc)
                writer_report["report_markdown"] = report_markdown
                if fallback_output_written:
                    writer_report["fallback_output_path"] = str(fallback_output_path)
                state_dict["writer_report"] = writer_report
                state_dict["answer_text"] = report_markdown
                writer_package_payload["writer_report"] = writer_report
                reformatter_result = {
                    **reformatter_result,
                    "fallback_to_writer": True,
                    "fallback_output_path": str(fallback_output_path) if fallback_output_written else "",
                    "output_written": False,
                    "fallback_output_written": fallback_output_written,
                    "fallback_draft_path": str(fallback_output_path) if fallback_output_written else "",
                    "fallback_draft_written": fallback_output_written,
                    "fallback_write_error": fallback_write_error,
                }
                writer_package_payload["reformatter_result"] = reformatter_result
                write_state_json(state_path, state_dict)
                write_writer_package()
                if fallback_output_written:
                    log(f"  [WARN] ReformatterAgent 失败，已写出 Writer 回退报告: {fallback_output_path}")
                else:
                    log("  [WARN] ReformatterAgent 失败，已保留 Writer 草稿但不写出 Clean report")
            progress.update(96, "ReformatterAgent 已降级")

    reformatter_required = bool(emit_clean_report and writer_publishable and not args.skip_reformatter)
    reformatter_output_written = bool(as_dict(reformatter_result).get("output_written"))
    clean_report_written = bool(
        reformatter_output_written
        and as_dict(reformatter_result).get("status") == "completed"
        and not as_dict(reformatter_result).get("fallback_to_writer")
        and as_dict(reformatter_result).get("success")
    )
    fallback_report_written = bool(as_dict(reformatter_result).get("fallback_output_written"))
    reformatter_blocked_clean = bool(reformatter_required and not clean_report_written)
    final_audit_result: Dict[str, Any] = as_dict(writer_package_payload.get("final_audit_result"))
    final_audit_blocked = False

    final_audit_candidate = bool(report_markdown and not writer_not_ready)
    if final_audit_candidate:
        try:
            from .final_audit_agent import run_final_audit

            audit_target = "Clean report" if clean_report_written else ("Writer fallback report" if fallback_report_written else "Formal report")
            log(f"[5.5/6] FinalAuditAgent 审查 {audit_target}")
            final_audit_result = run_final_audit(
                report_markdown=report_markdown,
                validation=as_dict(reformatter_result.get("validation")) or as_dict(writer_report.get("reformatter_validation")),
                clean_evidence=last_clean_evidence,
                writer_package_payload=writer_package_payload,
                query=query,
            )
        except Exception as audit_exc:  # pragma: no cover - defensive guard.
            final_audit_result = {
                "enabled": True,
                "success": False,
                "status": "failed",
                "blocked": True,
                "error": str(audit_exc),
                "llm_call": as_dict(getattr(audit_exc, "diagnostic", {})),
            }
        final_audit_blocked = bool(final_audit_result.get("blocked"))
        record_stage_snapshot("final_audit_result", final_audit_result)
        writer_report["final_audit_result"] = final_audit_result
        writer_package_payload["writer_report"] = writer_report
        writer_package_payload["final_audit_result"] = final_audit_result
        state_dict["writer_report"] = writer_report
        state_dict["final_audit_result"] = final_audit_result
        reformatter_result = {
            **reformatter_result,
            "final_audit_status": str(final_audit_result.get("status") or ""),
            "final_audit_blocked": final_audit_blocked,
        }
        writer_package_payload["reformatter_result"] = reformatter_result
        if final_audit_blocked:
            writer_report["clean_report_blocked_reason"] = "final_audit_fatal"
            writer_package_payload["clean_report_blocked_reason"] = "final_audit_fatal"
            log("  [BLOCKED] FinalAuditAgent 返回 fatal，Clean report 需要人工复核后再交付")
        elif final_audit_result.get("enabled") and final_audit_result.get("status") not in {"skipped", "failed"}:
            log(f"  - Final audit status: {final_audit_result.get('status')}")
        formal_path = str(writer_report.get("formal_report_path") or writer_report.get("writer_markdown_path") or "")
        if formal_path and report_markdown:
            writer_report["report_markdown"] = report_markdown
            state_dict["answer_text"] = report_markdown
            state_dict["writer_report"] = writer_report
            write_formal_markdown(Path(formal_path), report_markdown)
        write_state_json(state_path, state_dict)
        write_writer_package()

    writer_clean_content_eligible = bool(
        writer_report.get("clean_content_eligible")
        if "clean_content_eligible" in writer_report
        else writer_report.get("clean_report_eligible")
    )
    clean_content_eligible_final = bool(writer_clean_content_eligible and not final_audit_blocked)
    clean_output_enabled = bool(emit_clean_report)
    clean_report_eligible_final = bool(clean_content_eligible_final and clean_output_enabled)
    writer_report["writer_clean_report_eligible"] = writer_clean_content_eligible
    writer_report["clean_content_eligible"] = clean_content_eligible_final
    writer_report["clean_output_enabled"] = clean_output_enabled
    writer_report["clean_report_written"] = clean_report_written
    writer_report["clean_report_eligible"] = clean_report_eligible_final
    if final_audit_blocked:
        writer_report["clean_report_blocked_reason"] = "final_audit_fatal"
    elif reformatter_blocked_clean:
        writer_report["clean_report_blocked_reason"] = "reformatter_clean_missing"
    elif clean_content_eligible_final and not emit_clean_report:
        writer_report["clean_report_blocked_reason"] = "clean_report_output_disabled"
    writer_package_payload["writer_report"] = writer_report
    state_dict["writer_report"] = writer_report

    topic_bundle_store_summary: Dict[str, Any] = {}
    try:
        from rag_pipeline.cache.topic_bundle_cache import store_topic_bundle

        topic_bundle_store_summary = store_topic_bundle(
            query=query,
            research_plan=as_dict(state_dict.get("research_plan")) or as_dict(as_dict(state_dict.get("query_analysis")).get("research_plan")),
            report_blueprint=as_dict(writer_package_payload.get("report_blueprint")) or as_dict(writer_report.get("report_blueprint")),
            evidence_package=as_dict(writer_package_payload.get("evidence_package")),
            structured_analysis=as_dict(writer_package_payload.get("structured_analysis")),
            source_registry=as_list(writer_package_payload.get("source_registry")),
            chapter_evidence_packages=as_list(writer_package_payload.get("chapter_evidence_packages")),
            micro_layouts=as_list(writer_package_payload.get("micro_layouts")),
            table_packages=as_list(writer_package_payload.get("table_packages")),
            writer_report=writer_report,
            stage="full_report_delivery",
            stored_from="full_report_render_artifacts" if as_dict(writer_report.get("render_artifacts")) else "full_report_compacted_fallback",
        )
    except Exception as cache_exc:  # pragma: no cover - cache must never block report delivery.
        topic_bundle_store_summary = {"enabled": True, "stored": False, "reason": "store_failed", "error": str(cache_exc)}
    writer_package_payload["topic_bundle_cache_store"] = topic_bundle_store_summary
    writer_report["topic_bundle_cache_store"] = topic_bundle_store_summary
    writer_package_payload["writer_report"] = writer_report
    state_dict["writer_report"] = writer_report
    state_dict["topic_bundle_cache_store"] = topic_bundle_store_summary

    score_report_written = False
    score_report_path = ""
    if formal_report_available or writer_not_ready:
        score_markdown = render_score_markdown(
            query=query,
            writer_report=writer_report,
            writer_package=writer_package_payload,
            final_audit_result=final_audit_result,
            reformatter_result=reformatter_result,
        )
        if score_markdown:
            write_score_markdown(score_report_md_path, score_markdown)
            score_report_written = True
            score_report_path = str(score_report_md_path)
            writer_report["score_report_path"] = score_report_path
            writer_report["score_markdown"] = score_markdown
            state_dict["writer_report"] = writer_report
            state_dict["score_report_path"] = score_report_path
            writer_package_payload["writer_report"] = writer_report
            write_state_json(state_path, state_dict)

    formal_report_path = str(writer_report.get("formal_report_path") or writer_report.get("writer_markdown_path") or "")
    formal_report_written = bool(formal_report_path and report_markdown and not writer_not_ready)
    clean_report_eligible = bool(writer_report.get("clean_report_eligible"))
    quality_findings_count = len(as_list(writer_report.get("quality_findings")))
    quality_score_value = writer_report.get("quality_score")
    quality_grade_value = writer_report.get("quality_grade")
    if quality_score_value is None and report_markdown:
        score_match = re.search(r"质量总分[:：]\s*(\d{1,3})\s*/\s*100", report_markdown)
        if score_match:
            quality_score_value = int(score_match.group(1))
    writer_package_payload["report_delivery_status"] = {
        "delivery_tier": delivery_tier or str(writer_report.get("delivery_tier") or ""),
        "formal_report_written": formal_report_written,
        "formal_report_path": formal_report_path if formal_report_written else "",
        "quality_score": quality_score_value,
        "quality_grade": quality_grade_value,
        "score_report_written": score_report_written,
        "score_report_path": score_report_path,
        "clean_report_eligible": clean_report_eligible,
        "clean_content_eligible": clean_content_eligible_final,
        "clean_output_enabled": clean_output_enabled,
        "writer_clean_report_eligible": writer_clean_content_eligible,
        "quality_findings_count": quality_findings_count,
        "clean_report_written": clean_report_written,
        "clean_report_path": str(writer_report.get("reformatter_output_path") or "") if clean_report_written else "",
        "fallback_report_written": fallback_report_written,
        "fallback_report_path": str(writer_report.get("fallback_output_path") or as_dict(reformatter_result).get("fallback_output_path") or "") if fallback_report_written else "",
        "review_draft_written": bool(writer_report.get("review_draft_markdown_path")),
        "review_draft_path": str(writer_report.get("review_draft_markdown_path") or ""),
        "diagnostic_markdown_path": str(writer_report.get("diagnostic_markdown_path") or "") if writer_not_ready else "",
        "review_required": bool((not clean_report_eligible) or reformatter_blocked_clean or final_audit_blocked),
        "blocked_reason": "final_audit_fatal" if final_audit_blocked else ("reformatter_clean_missing" if reformatter_blocked_clean else ""),
        "final_audit_status": str(final_audit_result.get("status") or ""),
        "final_audit_blocked": final_audit_blocked,
        "post_qa_repair_status": str(as_dict(writer_report.get("post_qa_repair")).get("status") or ""),
        "review_agent_applied_to_formal_report": bool(writer_report.get("review_agent_applied_to_formal_report") or review_applied_to_formal_report),
        "review_agent_mode": str(writer_report.get("review_agent_mode") or review_mode or ""),
        "topic_bundle_cache_preflight_status": str(topic_cache_preflight.get("status") or ""),
        "topic_bundle_cache_seedable": bool(topic_cache_preflight.get("seedable") or topic_cache_preflight.get("can_seed_evidence")),
        "topic_bundle_cache_summary_only": bool(topic_cache_preflight.get("summary_only")),
        "topic_bundle_cache_seed_evidence_count": int(topic_cache_seed.get("seed_evidence_count") or 0),
        "topic_bundle_cache_usable_for_skip_search": bool(topic_cache_preflight.get("usable_for_skip_search")),
        "topic_bundle_cache_used_for_skip_search": bool(topic_cache_skip_search),
        "topic_bundle_cache_store": topic_bundle_store_summary,
    }
    artifact_ledger_final_sync = sync_artifact_ledger_package_safe(
        artifact_ledger_store,
        run_id=run_id,
        writer_package=writer_package_payload,
        writer_report=writer_report,
        final_audit_result=final_audit_result,
    )
    if artifact_ledger_final_sync:
        artifact_ledger_status.update(artifact_ledger_final_sync)
        writer_package_payload["artifact_ledger"] = dict(artifact_ledger_status)
        writer_report["artifact_ledger"] = dict(artifact_ledger_status)
        state_dict["artifact_ledger"] = dict(artifact_ledger_status)
        writer_package_payload["writer_report"] = writer_report
    artifact_ledger_final_status = (
        "not_ready"
        if writer_not_ready
        else (
            "clean_blocked"
            if final_audit_blocked or reformatter_blocked_clean
            else ("review_required" if not writer_publishable else "completed")
        )
    )
    run_trace_result: Dict[str, Any] = {}
    try:
        trace_final_status = (
            "not_ready"
            if writer_not_ready
            else (
                "clean_blocked"
                if final_audit_blocked or reformatter_blocked_clean
                else ("review_required" if not writer_publishable else "completed")
            )
        )
        writer_package_payload["writer_report"] = writer_report
        run_trace_result = write_run_trace_from_package(
            run_id=run_id,
            output_dir=output_dir,
            base_name=base_name,
            writer_package=writer_package_payload,
            writer_report=writer_report,
            final_status=trace_final_status,
        )
        if run_trace_result.get("enabled"):
            writer_package_payload["run_trace"] = run_trace_result
            writer_report["run_trace_path"] = str(run_trace_result.get("trace_path") or "")
            writer_report["run_trace_summary_path"] = str(run_trace_result.get("summary_path") or "")
            writer_package_payload["writer_report"] = writer_report
            state_dict["writer_report"] = writer_report
    except Exception as trace_exc:  # pragma: no cover - trace must never block report delivery.
        run_trace_result = {"enabled": True, "status": "failed", "error": str(trace_exc)}
        writer_package_payload["run_trace"] = run_trace_result
    artifact_ledger_finish = finish_artifact_ledger_run_safe(
        artifact_ledger_store,
        run_id=run_id,
        query=query,
        status=str(locals().get("trace_final_status") or artifact_ledger_final_status),
    )
    if artifact_ledger_finish:
        artifact_ledger_status.update(artifact_ledger_finish)
        writer_package_payload["artifact_ledger"] = dict(artifact_ledger_status)
        writer_report["artifact_ledger"] = dict(artifact_ledger_status)
        state_dict["artifact_ledger"] = dict(artifact_ledger_status)
        writer_package_payload["writer_report"] = writer_report
    write_state_json(state_path, state_dict)
    write_writer_package()

    if writer_not_ready:
        log("[6/6] 正式报告已阻断，输出研究未完成与补证清单")
    elif formal_report_written and not writer_publishable:
        log("[6/6] 已产出正式报告；Clean report 未达标，质量问题已写入独立评分报告与 writer_package")
    elif reformatter_blocked_clean:
        log("[6/6] ReformatterAgent 未产出 Clean report，已保留 Writer 草稿和失败状态")
    elif final_audit_blocked:
        log("[6/6] FinalAuditAgent 已阻断 Clean report 自动交付")
    elif args.skip_reformatter:
        log("[5/6] 状态文件已生成，报告正文直接输出" if args.skip_review else "[5/6] ReviewAgent 与状态文件已完成，报告正文直接输出")
    else:
        log("[6/6] ReformatterAgent 与状态文件已完成，洁净报告正文直接输出")
    progress.update(98, "准备输出结果")
    if formal_report_written and score_report_written:
        log("[6/6] Formal report and independent score report are ready.")
    final_incomplete = bool(missing and not args.allow_missing_stage)
    if writer_not_ready:
        finish_label = "研究未完成"
    elif formal_report_written and not writer_publishable:
        finish_label = "已生成评分正式报告"
    elif final_audit_blocked:
        finish_label = "Clean report 待复核"
    elif reformatter_blocked_clean:
        finish_label = "Clean report 未生成"
    elif formal_report_written:
        finish_label = "全流程完成"
    elif final_incomplete:
        finish_label = "流程不完整"
    else:
        finish_label = "全流程完成"
    progress.finish(finish_label)
    log(f"  - Full state JSON: {state_path}", force=True)
    log(f"  - Writer package JSON: {package_path}", force=True)
    if formal_report_written:
        log(f"  - Formal Report Markdown: {formal_report_path}", force=True)
    if score_report_written:
        log(f"  - Score Report Markdown: {score_report_path}", force=True)
    clean_path = str(writer_report.get("reformatter_output_path") or "")
    if clean_path and clean_report_written:
        log(f"  - Clean Markdown: {clean_path}", force=True)
    fallback_path = str(writer_report.get("fallback_output_path") or as_dict(reformatter_result).get("fallback_output_path") or "")
    if fallback_path and fallback_report_written:
        log(f"  - Fallback Writer Markdown: {fallback_path}", force=True)
    review_draft_path = str(writer_report.get("review_draft_markdown_path") or "")
    if review_draft_path:
        log(f"  - Review Draft Markdown: {review_draft_path}", force=True)
    diagnostic_path = str(writer_report.get("diagnostic_markdown_path") or "")
    if diagnostic_path:
        log(f"  - Diagnostic Markdown: {diagnostic_path}", force=True)
    if as_dict(run_trace_result).get("trace_path"):
        log(f"  - Run Trace JSONL: {run_trace_result.get('trace_path')}", force=True)
    if as_dict(run_trace_result).get("summary_path"):
        log(f"  - Run Trace Summary: {run_trace_result.get('summary_path')}", force=True)

    if errors:
        log("[WARN] 运行中存在非致命错误/降级：", force=True)
        for item in errors:
            log(f"  - {item}", force=True)

    final_stdout_allowed = bool(report_markdown and writer_publishable and not writer_not_ready and not reformatter_blocked_clean and not final_audit_blocked)
    if final_stdout_allowed:
        print(report_markdown)

    timeout_diagnostic_only = bool(writer_not_ready and as_dict(writer_report.get("live_timeout")).get("timeout_triggered"))
    if final_incomplete and not formal_report_written and not timeout_diagnostic_only:
        log("[6/6] 全流程执行不完整，以上阶段缺失。", force=True)
        return 2

    if writer_not_ready:
        log("[6/6] 检索或证据门槛未通过，正式报告未生成。", force=True)
        return 3 if env_flag("REPORT_NOT_READY_EXIT_NONZERO", False) else 0

    if not writer_publishable:
        log("[6/6] 流程完成，已产出带评分的正式报告；未发布 Clean report。", force=True)
        return 0 if formal_report_written else (4 if env_flag("REPORT_REVIEW_REQUIRED_EXIT_NONZERO", False) else 0)

    if reformatter_blocked_clean:
        log("[6/6] 流程完成但 Reformatter 未生成 Clean report；请查看 writer_package.reformatter_result。", force=True)
        return 5 if env_flag("REPORT_REFORMATTER_FAILURE_EXIT_NONZERO", False) else 0

    if final_audit_blocked:
        log("[6/6] 流程完成但 FinalAuditAgent 阻断自动交付；请查看 writer_package.final_audit_result。", force=True)
        return 6 if env_flag("REPORT_FINAL_AUDIT_EXIT_NONZERO", False) else 0

    log("[6/6] 全流程执行完成。", force=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
