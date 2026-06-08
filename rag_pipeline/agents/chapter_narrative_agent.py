from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..config.search_config import build_llm_config_for_task, build_llm_config_from_profile
from ..search.memory import call_openai_compatible_json, llm_config_is_ready
from .report_profile_registry import get_report_profile
from .section_body_rewrite_agent import (
    _validate_candidate as _validate_section_rewrite_candidate,
)


PROMPT_VERSION = "chapter_narrative_v2"
DEFAULT_CACHE_PATH = Path("output/cache/chapter_narrative")
ALLOWED_FINAL_ANALYSIS_SOURCES = {"llm_evidence_analysis", "llm_partial_merged"}


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, min_value: int = 0, max_value: int = 10_000) -> int:
    try:
        value = int(float(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        value = default
    return max(min_value, min(max_value, value))


def chapter_narrative_enabled() -> bool:
    return _env_flag("REPORT_ENABLE_LLM_CHAPTER_NARRATIVE", False)


def chapter_narrative_min_evidence_sections() -> int:
    return _env_int("REPORT_CHAPTER_NARRATIVE_MIN_EVIDENCE_SECTIONS", 2, min_value=0, max_value=200)


def chapter_narrative_max_chapters() -> int:
    return _env_int("REPORT_CHAPTER_NARRATIVE_MAX_CHAPTERS", 40, min_value=0, max_value=200)


def chapter_narrative_cache_enabled() -> bool:
    return _env_flag("REPORT_CHAPTER_NARRATIVE_CACHE_ENABLED", True)


def _cache_root() -> Path:
    raw = os.getenv("REPORT_CHAPTER_NARRATIVE_CACHE_PATH", "").strip()
    return Path(raw) if raw else DEFAULT_CACHE_PATH


def _new_diagnostics(*, enabled: Optional[bool] = None) -> Dict[str, Any]:
    is_enabled = chapter_narrative_enabled() if enabled is None else bool(enabled)
    return {
        "enabled": is_enabled,
        "status": "enabled" if is_enabled else "skipped",
        "skipped_reason": "" if is_enabled else "disabled",
        "attempted_count": 0,
        "success_count": 0,
        "fallback_count": 0,
        "rejected_count": 0,
        "skipped_count": 0,
        "input_chapter_count": 0,
        "input_section_count": 0,
        "rewritten_section_count": 0,
        "cache_hit_count": 0,
        "llm_called_count": 0,
        "rejected_reasons": {},
        "failure_reasons": {},
        "failed_chapter_ids": [],
        "model": "",
        "fallback_model_used_count": 0,
    }


def _record_count(bucket: Dict[str, int], key: str) -> None:
    key = str(key or "unknown").strip() or "unknown"
    bucket[key] = int(bucket.get(key, 0)) + 1


def _render_paragraph(section: Dict[str, Any]) -> str:
    for block in _as_list(section.get("render_blocks")):
        block = _as_dict(block)
        if str(block.get("type") or "") == "paragraph":
            text = _text(block.get("text"))
            if text:
                return text
    for key in ("composed_paragraph", "paragraph", "reasoning", "claim", "mechanism"):
        text = _text(section.get(key))
        if text:
            return text
    return ""


def _facts_for_section(section: Dict[str, Any]) -> List[Dict[str, Any]]:
    refs = [_text(ref) for ref in _as_list(section.get("used_fact_refs") or section.get("evidence_refs"))]
    citations = [_text(ref) for ref in _as_list(section.get("citation_refs"))]
    raw_facts = _as_list(section.get("supporting_facts"))
    facts: List[Dict[str, Any]] = []
    for index, ref in enumerate(refs):
        fact_value = raw_facts[index] if index < len(raw_facts) else ""
        if isinstance(fact_value, dict):
            fact = dict(fact_value)
            fact.setdefault("evidence_id", ref)
        else:
            fact = {"evidence_id": ref, "distilled_fact": _text(fact_value)}
        if citations:
            fact.setdefault("source_ref", citations[min(index, len(citations) - 1)])
        facts.append(fact)
    return facts


def _section_payload(section: Dict[str, Any]) -> Dict[str, Any]:
    paragraph = _render_paragraph(section)
    return {
        "section_id": _text(section.get("section_id")),
        "section_title": section.get("section_title"),
        "block_type": section.get("block_type"),
        "claim_strength": section.get("claim_strength") or section.get("confidence"),
        "composer_paragraph": paragraph,
        "used_fact_refs": _as_list(section.get("used_fact_refs") or section.get("evidence_refs")),
        "citation_refs": _as_list(section.get("citation_refs")),
        "supporting_facts": _as_list(section.get("supporting_facts")),
    }


def _eligible_sections(chapter: Dict[str, Any]) -> List[Dict[str, Any]]:
    sections: List[Dict[str, Any]] = []
    for section in _as_list(chapter.get("sections")):
        section = _as_dict(section)
        if not section or section.get("omit_from_report"):
            continue
        if section.get("observation_only") and not section.get("evidence_backed"):
            continue
        if not _as_list(section.get("used_fact_refs") or section.get("evidence_refs")):
            continue
        if not _render_paragraph(section):
            continue
        sections.append(section)
    return sections


def _evidence_backed_section_count(chapters: Sequence[Dict[str, Any]]) -> int:
    return sum(1 for chapter in chapters for section in _eligible_sections(_as_dict(chapter)) if section.get("evidence_backed"))


def _cited_evidence_backed_section_count(chapters: Sequence[Dict[str, Any]]) -> int:
    return sum(
        1
        for chapter in chapters
        for section in _eligible_sections(_as_dict(chapter))
        if section.get("evidence_backed") and _as_list(section.get("citation_refs"))
    )


def _profile_narrative_spines(report_blueprint: Dict[str, Any]) -> List[str]:
    explicit = report_blueprint.get("report_profile") or report_blueprint.get("report_family") or report_blueprint.get("report_name")
    profile = get_report_profile(str(explicit or ""))
    return [str(item or "").strip() for item in _as_list(profile.get("narrative_spines")) if str(item or "").strip()]


def _system_prompt() -> str:
    return (
        "你是行研报告章节主笔。你的任务是把同一章内已经由确定性 composer 生成的段落，"
        "改写成更连贯、专业、自然的章节叙事。只能使用输入 facts 和原段落中的事实。"
        "不得新增事实、数字、公司、来源或引用；不得提高 claim_strength；不得输出补证建议、"
        "QA、Clean、fatal、EV、URL 或内部诊断语言。每个 section 必须保留原 used_fact_refs "
        "和 citation_refs。证据不足时保持短，不为了篇幅扩写或推断。返回 JSON："
        "{\"chapter_lead\":\"...\", \"sections\":[{\"section_id\":\"...\", "
        "\"paragraph\":\"...\", \"used_fact_refs\":[...], \"citation_refs\":[...]}]}。"
    )
    return (
        "你是行研报告章节主笔。你的任务是把同一章内已经由确定性 composer 生成的段落，"
        "改写成更连贯、专业、自然的章节叙事。只允许使用输入 facts 和原段落中的事实。"
        "不得新增事实、数字、公司、来源或引用；不得提高 claim_strength；不得输出补证建议、QA、Clean、fatal、EV、URL 或内部诊断。"
        "每个 section 必须保留原 used_fact_refs 和 citation_refs。"
        "返回 JSON：{\"chapter_lead\":\"...\", \"sections\":[{\"section_id\":\"...\", \"paragraph\":\"...\", \"used_fact_refs\":[...], \"citation_refs\":[...]}]}。"
    )


def _cache_key(chapter: Dict[str, Any], sections: Sequence[Dict[str, Any]], report_blueprint: Dict[str, Any]) -> str:
    payload = {
        "prompt_version": PROMPT_VERSION,
        "chapter_id": chapter.get("chapter_id"),
        "narrative_spines": _profile_narrative_spines(report_blueprint),
        "sections": [
            {
                "section_id": section.get("section_id"),
                "paragraph": _render_paragraph(section),
                "used_fact_refs": _as_list(section.get("used_fact_refs") or section.get("evidence_refs")),
                "citation_refs": _as_list(section.get("citation_refs")),
            }
            for section in sections
        ],
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_cache(key: str) -> Dict[str, Any]:
    if not chapter_narrative_cache_enabled():
        return {}
    path = _cache_root() / f"{key}.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_cache(key: str, payload: Dict[str, Any]) -> None:
    if not chapter_narrative_cache_enabled():
        return
    root = _cache_root()
    try:
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{key}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        return


def _validate_chapter_payload(
    *,
    chapter: Dict[str, Any],
    sections: Sequence[Dict[str, Any]],
    payload: Dict[str, Any],
) -> Tuple[bool, str, Dict[str, Dict[str, Any]]]:
    raw_sections = _as_list(payload.get("sections"))
    by_id = {
        _text(item.get("section_id")): _as_dict(item)
        for item in raw_sections
        if isinstance(item, dict) and _text(item.get("section_id"))
    }
    expected_ids = [_text(section.get("section_id")) for section in sections if _text(section.get("section_id"))]
    unknown_ids = set(by_id) - set(expected_ids)
    if unknown_ids:
        return False, "unknown_section_id", {}
    if not by_id:
        return False, "missing_section_output", {}
    accepted: Dict[str, Dict[str, Any]] = {}
    first_reason = ""
    for section in sections:
        section_id = _text(section.get("section_id"))
        result = by_id.get(section_id) or {}
        if not result:
            first_reason = first_reason or "missing_section_output"
            continue
        validation_section = dict(section)
        validation_section["paragraph"] = _render_paragraph(section)
        valid, reason = _validate_section_rewrite_candidate(
            paragraph=_text(result.get("paragraph")),
            used_fact_refs=_as_list(result.get("used_fact_refs")),
            citation_refs=_as_list(result.get("citation_refs")),
            section=validation_section,
            facts=_facts_for_section(section),
        )
        if not valid:
            first_reason = first_reason or reason
            result["_validation_rejected_reason"] = reason
            continue
        accepted[section_id] = result
    if not accepted:
        return False, first_reason or "no_valid_section_output", {}
    lead = _text(payload.get("chapter_lead"))
    if lead:
        lead_validation = dict(sections[0])
        lead_validation["paragraph"] = _render_paragraph(sections[0])
        lead_valid, lead_reason = _validate_section_rewrite_candidate(
            paragraph=lead,
            used_fact_refs=_as_list(sections[0].get("used_fact_refs") or sections[0].get("evidence_refs")),
            citation_refs=_as_list(sections[0].get("citation_refs")),
            section=lead_validation,
            facts=_facts_for_section(sections[0]),
        )
        if not lead_valid and lead_reason in {"forbidden_public_text", "new_numeric_claim"}:
            payload["chapter_lead"] = ""
    return True, first_reason, accepted


def _replace_first_paragraph_block(section: Dict[str, Any], paragraph: str) -> Dict[str, Any]:
    updated = dict(section)
    blocks = [_as_dict(block) for block in _as_list(section.get("render_blocks"))]
    replaced = False
    new_blocks: List[Dict[str, Any]] = []
    for block in blocks:
        if not replaced and str(block.get("type") or "") == "paragraph":
            block = {**block, "text": paragraph}
            replaced = True
        new_blocks.append(block)
    if not replaced:
        new_blocks.insert(0, {"type": "paragraph", "label": "", "text": paragraph})
    updated["claim"] = paragraph
    updated["reasoning"] = paragraph
    updated["mechanism"] = paragraph
    updated["render_blocks"] = new_blocks
    updated["chapter_narrative_status"] = "rewritten"
    updated["chapter_narrative"] = {"status": "rewritten"}
    return updated


def _payload_for_chapter(
    *,
    chapter: Dict[str, Any],
    sections: Sequence[Dict[str, Any]],
    report_blueprint: Dict[str, Any],
    previous_chapter_takeaway: str = "",
) -> Dict[str, Any]:
    return {
        "chapter_id": chapter.get("chapter_id"),
        "chapter_title": chapter.get("chapter_title"),
        "chapter_question": chapter.get("chapter_question") or chapter.get("chapter_title"),
        "chapter_summary": _as_dict(chapter.get("chapter_summary")),
        "narrative_spines": _profile_narrative_spines(report_blueprint),
        "previous_chapter_takeaway": previous_chapter_takeaway,
        "sections": [_section_payload(section) for section in sections],
    }


def _config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    result = dict(config or build_llm_config_for_task("body_rewrite"))
    timeout = _env_int("REPORT_CHAPTER_NARRATIVE_TIMEOUT_SECONDS", 120, min_value=1, max_value=600)
    result["timeout"] = timeout
    return result


def _fallback_llm_configs(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    embedded = config.get("fallback_config") if isinstance(config, dict) else None
    if isinstance(embedded, dict):
        configs.append(dict(embedded))
    profile_values = [
        os.getenv("REPORT_CHAPTER_NARRATIVE_FALLBACK_PROFILES", ""),
        os.getenv("REPORT_BODY_REWRITE_FALLBACK_PROFILES", ""),
    ]
    seen = {
        (
            str(config.get("profile") or "").strip().lower(),
            str(config.get("model") or "").strip().lower(),
        )
    }
    for raw in profile_values:
        for profile in re.split(r"[,;]\s*", str(raw or "").strip()):
            profile = profile.strip()
            if not profile:
                continue
            try:
                candidate = build_llm_config_from_profile(profile, default_timeout=float(config.get("timeout") or 120))
            except Exception:
                continue
            key = (
                str(candidate.get("profile") or "").strip().lower(),
                str(candidate.get("model") or "").strip().lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            configs.append(candidate)
    return [candidate for candidate in configs if llm_config_is_ready(candidate)]


def run_chapter_narrative(
    *,
    chapter_packages: Sequence[Dict[str, Any]],
    report_blueprint: Optional[Dict[str, Any]] = None,
    llm_config: Optional[Dict[str, Any]] = None,
    quality_context: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    packages = [dict(item) for item in list(chapter_packages or []) if isinstance(item, dict)]
    diagnostics = _new_diagnostics()
    diagnostics["input_chapter_count"] = len(packages)
    diagnostics["input_section_count"] = sum(len(_eligible_sections(chapter)) for chapter in packages)
    if not diagnostics["enabled"]:
        return packages, diagnostics

    quality = _as_dict(quality_context)
    final_analysis_source = _text(quality.get("final_analysis_source"))
    if final_analysis_source not in ALLOWED_FINAL_ANALYSIS_SOURCES:
        diagnostics["status"] = "skipped"
        diagnostics["skipped_reason"] = "final_analysis_source_not_llm"
        return packages, diagnostics

    evidence_count = _evidence_backed_section_count(packages)
    cited_evidence_count = _cited_evidence_backed_section_count(packages)
    min_sections = chapter_narrative_min_evidence_sections()
    # The cited-section gate previously skipped narrative whenever section.citation_refs
    # was not yet attached, even though those sections were evidence_backed and would
    # have citations after the final-writer pass. Accept either signal (cited OR
    # evidence_backed) so a fully-cited downstream report still runs the P4 narrative
    # polish. Diagnostics still record both counts so regressions stay visible.
    effective_count = max(cited_evidence_count, evidence_count)
    if effective_count < min_sections:
        diagnostics["status"] = "skipped"
        diagnostics["skipped_reason"] = "insufficient_cited_sections"
        diagnostics["evidence_backed_section_count"] = evidence_count
        diagnostics["cited_evidence_backed_section_count"] = cited_evidence_count
        diagnostics["min_evidence_backed_sections"] = min_sections
        return packages, diagnostics

    config = _config(llm_config)
    diagnostics["model"] = _text(config.get("model") or config.get("profile"))
    if not llm_config_is_ready(config):
        diagnostics["status"] = "skipped"
        diagnostics["skipped_reason"] = "llm_config_not_ready"
        return packages, diagnostics

    report_blueprint = _as_dict(report_blueprint)
    output: List[Dict[str, Any]] = []
    previous_takeaway = ""
    max_chapters = chapter_narrative_max_chapters()
    for chapter_index, chapter in enumerate(packages):
        if max_chapters and diagnostics["attempted_count"] >= max_chapters:
            output.append(dict(chapter))
            diagnostics["skipped_count"] += 1
            continue
        sections = _eligible_sections(chapter)
        if not sections:
            output.append(dict(chapter))
            diagnostics["skipped_count"] += 1
            previous_takeaway = _text(_as_dict(chapter.get("chapter_summary")).get("key_takeaway"))
            continue
        key = _cache_key(chapter, sections, report_blueprint)
        cached = _load_cache(key)
        diagnostics["attempted_count"] += 1
        try:
            payload = cached
            active_config = config
            used_fallback_model = False
            if not payload:
                user_payload = _payload_for_chapter(
                    chapter=chapter,
                    sections=sections,
                    report_blueprint=report_blueprint,
                    previous_chapter_takeaway=previous_takeaway,
                )
                errors: List[str] = []
                for attempt_index, attempt_config in enumerate([config, *_fallback_llm_configs(config)]):
                    active_config = attempt_config
                    try:
                        payload = call_openai_compatible_json(
                            config=attempt_config,
                            system_prompt=_system_prompt(),
                            user_payload=user_payload,
                        )
                        used_fallback_model = attempt_index > 0
                        break
                    except Exception as exc:
                        errors.append(f"{_text(attempt_config.get('model') or attempt_config.get('profile'))}:{type(exc).__name__}")
                        continue
                if not payload:
                    raise RuntimeError(errors[-1] if errors else "llm_error")
            if cached:
                diagnostics["cache_hit_count"] += 1
            else:
                diagnostics["llm_called_count"] += 1
                if used_fallback_model:
                    diagnostics["fallback_model_used_count"] += 1
            response_payload = _as_dict(payload.get("payload") if "payload" in payload else payload)
            valid, reason, by_id = _validate_chapter_payload(
                chapter=chapter,
                sections=sections,
                payload=response_payload,
            )
            if not valid:
                diagnostics["fallback_count"] += 1
                diagnostics["rejected_count"] += 1
                diagnostics["failed_chapter_ids"].append(_text(chapter.get("chapter_id")) or str(chapter_index + 1))
                _record_count(diagnostics["rejected_reasons"], reason)
                output.append(dict(chapter))
                continue
            if reason:
                diagnostics["rejected_count"] += 1
                _record_count(diagnostics["rejected_reasons"], reason)
            if not cached:
                _write_cache(key, response_payload)
            rewritten_chapter = dict(chapter)
            new_sections: List[Dict[str, Any]] = []
            for section in _as_list(chapter.get("sections")):
                if not isinstance(section, dict):
                    continue
                section_id = _text(section.get("section_id"))
                result = by_id.get(section_id)
                if result:
                    new_sections.append(_replace_first_paragraph_block(section, _text(result.get("paragraph"))))
                    diagnostics["rewritten_section_count"] += 1
                else:
                    new_sections.append(dict(section))
            lead = _text(response_payload.get("chapter_lead"))
            if lead:
                rewritten_chapter["lead"] = lead
            rewritten_chapter["sections"] = new_sections
            rewritten_chapter["chapter_narrative_status"] = "rewritten"
            diagnostics["success_count"] += 1
            output.append(rewritten_chapter)
        except Exception as exc:
            diagnostics["fallback_count"] += 1
            diagnostics["failed_chapter_ids"].append(_text(chapter.get("chapter_id")) or str(chapter_index + 1))
            _record_count(diagnostics["failure_reasons"], f"llm_error:{type(exc).__name__}")
            output.append(dict(chapter))
        previous_takeaway = _text(_as_dict(chapter.get("chapter_summary")).get("key_takeaway"))
    diagnostics["status"] = "green" if diagnostics["success_count"] else ("yellow" if diagnostics["fallback_count"] else "skipped")
    return output, diagnostics
