from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import threading
import time
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


def json_default(value: Any) -> Any:
    return json_safe_default(value)


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
    text = re.sub(r"^(#{1,4})\s*关键事实对照\s*$", r"\1 事实依据", text, flags=re.M)
    text = re.sub(r"^(#{1,4})\s*商业化质量与经济性\s*$", r"\1 商业化证据", text, flags=re.M)
    for pattern in FORMAL_REPORT_PRIVATE_SENTENCE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.I)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def finalize_formal_report(markdown: str) -> str:
    text = str(markdown or "")
    try:
        return strip_formal_report_private_sentences(finalize_public_report(text))
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
        return strip_formal_report_private_sentences(cleaned).strip()


def write_formal_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(finalize_formal_report(str(text or "")).strip() + "\n", encoding="utf-8")


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


def _layout_score_diagnostics(writer_package: Dict[str, Any]) -> Dict[str, Any]:
    writer_report = as_dict(writer_package.get("writer_report"))
    micro_layouts = as_list(writer_package.get("micro_layouts")) or as_list(writer_report.get("micro_layouts"))
    chapter_packages = as_list(writer_package.get("chapter_packages")) or as_list(writer_report.get("chapter_packages"))
    table_packages = as_list(writer_package.get("table_packages")) or as_list(writer_report.get("table_packages"))
    chapter_by_id = {
        str(chapter.get("chapter_id") or "").strip(): chapter
        for chapter in chapter_packages
        if isinstance(chapter, dict) and str(chapter.get("chapter_id") or "").strip()
    }
    chapters: List[Dict[str, Any]] = []
    dropped_blocks: List[Dict[str, Any]] = []
    observation_only_blocks: List[Dict[str, Any]] = []
    evidence_backed_blocks: List[Dict[str, Any]] = []
    omitted_observation_blocks: List[Dict[str, Any]] = []
    rendered_block_count = 0
    for layout in micro_layouts:
        if not isinstance(layout, dict):
            continue
        chapter_id = str(layout.get("chapter_id") or "").strip()
        planned_sections = [as_dict(section) for section in as_list(layout.get("sections")) if isinstance(section, dict)]
        planned_blocks = [
            str(section.get("block_type") or section.get("output_type") or "").strip()
            for section in planned_sections
            if str(section.get("block_type") or section.get("output_type") or "").strip()
        ]
        chapter = as_dict(chapter_by_id.get(chapter_id))
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
        rendered_quality = []
        for section in rendered_sections:
            block_type = str(section.get("block_type") or section.get("output_type") or "").strip()
            if not block_type:
                continue
            refs = as_list(section.get("evidence_refs")) or as_list(section.get("required_evidence_refs"))
            facts = as_list(section.get("supporting_facts"))
            text = " ".join(
                str(section.get(key) or "")
                for key in ("claim", "reasoning", "mechanism", "counter_evidence")
            )
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
            }
            rendered_quality.append(quality_item)
            if evidence_backed:
                evidence_backed_blocks.append(quality_item)
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
        chapters.append(
            {
                "chapter_id": chapter_id,
                "planned_blocks": planned_blocks,
                "rendered_blocks": rendered_blocks,
                "dropped_blocks": missing,
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
    for table in table_packages:
        if not isinstance(table, dict) or table.get("should_render"):
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
    return {
        "micro_layout_count": len([item for item in micro_layouts if isinstance(item, dict)]),
        "chapter_package_count": len([item for item in chapter_packages if isinstance(item, dict)]),
        "rendered_block_count": rendered_block_count,
        "dropped_block_count": len(dropped_blocks),
        "observation_only_block_count": len(observation_only_blocks),
        "evidence_backed_block_count": len(evidence_backed_blocks),
        "effective_section_count": len(evidence_backed_blocks),
        "omitted_observation_section_count": len(omitted_observation_blocks),
        "chapters": chapters,
        "dropped_blocks": dropped_blocks,
        "observation_only_blocks": observation_only_blocks,
        "evidence_backed_blocks": evidence_backed_blocks,
        "omitted_observation_blocks": omitted_observation_blocks,
        "skipped_tables": skipped_tables,
    }


def _chapter_evidence_input_diagnostics(writer_package: Dict[str, Any]) -> Dict[str, Any]:
    writer_report = as_dict(writer_package.get("writer_report"))
    evidence_package = as_dict(writer_package.get("evidence_package"))
    chapters = (
        as_list(evidence_package.get("chapter_evidence_packages"))
        or as_list(writer_report.get("chapter_evidence_packages"))
        or as_list(writer_package.get("chapter_evidence_packages"))
    )
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
            }
        )
    return {
        "chapter_count": len(items),
        "source_pool_count": source_pool_count,
        "chapter_evidence_binding_failed": binding_failed,
        "total_unresolved_evidence_ref_count": total_unresolved_refs,
        "chapters": items,
    }


def _evidence_grade_usage_diagnostics(writer_package: Dict[str, Any]) -> Dict[str, Any]:
    writer_report = as_dict(writer_package.get("writer_report"))
    evidence_package = as_dict(writer_package.get("evidence_package"))
    chapters = (
        as_list(evidence_package.get("chapter_evidence_packages"))
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
    clean_eligible = bool(writer_report.get("clean_report_eligible")) and not bool(final_audit_result.get("blocked"))
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
    lines = [
        f"# {query} - 报告评分与审查",
        "",
        f"- 质量总分：{score}/100",
        f"- 质量等级：{grade}",
        f"- Clean 资格：{'是' if clean_eligible else '否'}",
        f"- 报告状态：{writer_report.get('report_status') or 'unknown'}",
        f"- 交付层级：{writer_report.get('delivery_tier') or 'unknown'}",
        f"- Post-QA 补证：{post_qa_status}",
    ]
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
                f"- layout_omitted_observation_section_count：{layout_diag.get('omitted_observation_section_count')}",
            ]
        )
        chapters = as_list(layout_diag.get("chapters"))
        if chapters:
            lines.extend(["", "### 章节 Block 对照"])
            for item in chapters[:12]:
                payload = as_dict(item)
                planned = ", ".join(str(block) for block in as_list(payload.get("planned_blocks"))) or "-"
                rendered = ", ".join(str(block) for block in as_list(payload.get("rendered_blocks"))) or "-"
                dropped = ", ".join(str(block) for block in as_list(payload.get("dropped_blocks"))) or "-"
                observation = ", ".join(str(block) for block in as_list(payload.get("observation_only_blocks"))) or "-"
                evidence_backed = ", ".join(str(block) for block in as_list(payload.get("evidence_backed_blocks"))) or "-"
                omitted_observation = ", ".join(str(block) for block in as_list(payload.get("omitted_observation_blocks"))) or "-"
                lines.append(
                    f"- {payload.get('chapter_id')}: planned=[{planned}] rendered=[{rendered}] "
                    f"evidence_backed=[{evidence_backed}] observation_only=[{observation}] "
                    f"omitted_observation=[{omitted_observation}] dropped=[{dropped}] "
                    f"effective_sections={payload.get('effective_section_count')} "
                    f"observation_sections={payload.get('observation_section_count')} "
                    f"effective_ratio={payload.get('chapter_effective_section_ratio')}"
                )
        skipped_tables = as_list(layout_diag.get("skipped_tables"))
        if skipped_tables:
            lines.extend(["", "### 表格未渲染"])
            for item in skipped_tables[:20]:
                payload = as_dict(item)
                lines.append(
                    f"- {payload.get('chapter_id') or '-'} / {payload.get('table_id') or '-'} / {payload.get('anchor_block_type') or '-'}: {payload.get('reason') or 'not_rendered'}"
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
    source_registry = as_list(inputs.get("source_registry")) or as_list(evidence_package.get("source_registry")) or as_list(evidence_package.get("sources"))
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
    args = build_arg_parser().parse_args()
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

    pipeline_started = time.perf_counter()
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
        log("[cache] usable topic bundle found; skipping live IQS/OpenAI retrieval and rebuilding from cached evidence")
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
        )
    state.setdefault("metadata", {})
    state["metadata"]["topic_bundle_cache_preflight"] = topic_cache_preflight
    state["topic_bundle_cache_preflight"] = topic_cache_preflight
    progress.update(72, "主体报告生成完成")
    log(f"[3/6] Brain 主流程完成，用时 {time.perf_counter() - pipeline_started:.1f}s")

    state_dict = dict(state or {})
    raw_output = as_dict(state_dict.get("raw_output"))
    writer_report = as_dict(state_dict.get("writer_report")) or as_dict(raw_output.get("writer_report"))
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
        report_markdown = finalize_formal_report(report_markdown)
        writer_report["report_markdown"] = report_markdown
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
    evidence_package_payload = as_dict(state_dict.get("evidence_package")) or as_dict(raw_output.get("evidence_package"))
    evidence_health_summary = (
        as_dict(evidence_package_payload.get("evidence_health_summary"))
        or as_dict(as_dict(evidence_package_payload.get("summary")).get("evidence_health_summary"))
        or as_dict(as_dict(evidence_package_payload.get("metadata")).get("evidence_health_summary"))
    )
    source_registry_payload = (
        as_list(as_dict(state_dict.get("writer_report")).get("source_registry"))
        or as_list(evidence_package_payload.get("source_registry"))
        or as_list(evidence_package_payload.get("sources"))
    )
    chapter_evidence_payload = (
        as_list(raw_output.get("chapter_evidence_packages"))
        or as_list(writer_report.get("chapter_evidence_packages"))
        or as_list(as_dict(raw_output.get("evidence_package")).get("chapter_evidence_packages"))
        or as_list(as_dict(state_dict.get("evidence_package")).get("chapter_evidence_packages"))
        or as_list(state_dict.get("chapter_evidence_packages"))
    )
    if chapter_evidence_payload and not as_list(evidence_package_payload.get("chapter_evidence_packages")):
        evidence_package_payload["chapter_evidence_packages"] = chapter_evidence_payload
    writer_package_payload = {
        "query": query,
        "stage_status": stage_status(state_dict),
        "llm_runtime": llm_status,
        "evidence_package": evidence_package_payload,
        "evidence_health_summary": evidence_health_summary,
        "source_registry": source_registry_payload,
        "structured_analysis": as_dict(state_dict.get("structured_analysis")) or as_dict(raw_output.get("structured_analysis")),
        "report_blueprint": as_dict(state_dict.get("report_blueprint")) or as_dict(raw_output.get("report_blueprint")),
        "chapter_evidence_packages": chapter_evidence_payload,
        "micro_layouts": as_list(state_dict.get("micro_layouts")) or as_list(raw_output.get("micro_layouts")),
        "table_packages": as_list(state_dict.get("table_packages")) or as_list(raw_output.get("table_packages")),
        "argument_units": as_list(state_dict.get("argument_units")) or as_list(raw_output.get("argument_units")),
        "chapter_packages": as_list(state_dict.get("chapter_packages")) or as_list(raw_output.get("chapter_packages")),
        "writer_report": writer_report,
        "review_result": review_result,
        "reformatter_result": reformatter_result,
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
            reformatter_result = {
                **reformatter_result,
                "enabled": True,
                "success": False,
                "status": "failed",
                "output_path": str(clean_output_path),
                "output_written": False,
                "error": str(exc),
            }
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

    writer_clean_report_eligible = bool(writer_report.get("clean_report_eligible"))
    clean_report_eligible_final = bool(
        writer_clean_report_eligible
        and not reformatter_blocked_clean
        and not final_audit_blocked
        and (clean_report_written or emit_clean_report)
    )
    writer_report["writer_clean_report_eligible"] = writer_clean_report_eligible
    writer_report["clean_report_eligible"] = clean_report_eligible_final
    if final_audit_blocked:
        writer_report["clean_report_blocked_reason"] = "final_audit_fatal"
    elif reformatter_blocked_clean:
        writer_report["clean_report_blocked_reason"] = "reformatter_clean_missing"
    elif writer_clean_report_eligible and not (clean_report_written or emit_clean_report):
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
            stored_from="full_report_compacted_fallback",
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
    if formal_report_available:
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
        "writer_clean_report_eligible": writer_clean_report_eligible,
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

    if errors:
        log("[WARN] 运行中存在非致命错误/降级：", force=True)
        for item in errors:
            log(f"  - {item}", force=True)

    final_stdout_allowed = bool(report_markdown and writer_publishable and not writer_not_ready and not reformatter_blocked_clean and not final_audit_blocked)
    if final_stdout_allowed:
        print(report_markdown)

    if final_incomplete and not formal_report_written:
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
