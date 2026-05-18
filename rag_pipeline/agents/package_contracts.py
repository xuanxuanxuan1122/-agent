from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    from .block_schema import valid_block_types
    from .report_profile_registry import quality_contract_for_profile
    from .table_validator import validate_table_package
except Exception:  # pragma: no cover - direct script mode fallback
    from block_schema import valid_block_types  # type: ignore
    from report_profile_registry import quality_contract_for_profile  # type: ignore
    from table_validator import validate_table_package  # type: ignore


FIXED_FIVE_DIMENSIONS = {
    "\u5e02\u573a\u89c4\u6a21\u4e0e\u589e\u901f",
    "\u7ade\u4e89\u683c\u5c40",
    "\u653f\u7b56\u4e0e\u76d1\u7ba1\u73af\u5883",
    "\u6280\u672f\u8def\u7ebf\u4e0e\u4ea7\u4e1a\u94fe",
    "\u8d44\u672c\u52a8\u6001",
}

WEAK_CLAIM_PREFIXES = (
    "\u5df2\u6709\u53ef\u9a8c\u8bc1",
    "\u5df2\u6709\u53ef\u6838\u9a8c",
    "\u5df2\u6709\u53ef\u7528\u8bc1\u636e",
    "\u5f53\u524d\u8bc1\u636e",
)

BAD_CLAIM_PATTERNS = [
    r"已有可核验证据",
    r"已有可验证证据",
    r"证据不足",
    r"可作为判断输入",
    r"需结合来源等级",
    r"需结合.*时间范围",
    r"需结合.*口径边界",
    r"尚未发现足以推翻",
    r"继续补证",
]

ACTION_WORDS = (
    "\u4f18\u5148",
    "\u907f\u514d",
    "\u9a8c\u8bc1",
    "\u8ddf\u8e2a",
    "\u6392\u9664",
    "\u8865\u5145",
    "\u8bbe\u7f6e",
)

CAUSE_WORDS = (
    "\u56e0\u4e3a",
    "\u7531\u4e8e",
    "\u539f\u56e0",
    "\u5bfc\u81f4",
    "\u4ece\u800c",
    "\u56e0\u6b64",
)


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _compact(value: Any, max_chars: int = 220) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "..."


def _issue(
    issues: List[Dict[str, Any]],
    *,
    package: str,
    issue_type: str,
    message: str,
    path: str = "",
    severity: str = "error",
    **extra: Any,
) -> None:
    payload = {
        "severity": severity,
        "package": package,
        "type": issue_type,
        "message": message,
    }
    if path:
        payload["path"] = path
    payload.update(extra)
    issues.append(payload)


def _split_issues(issues: Sequence[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    errors = [item for item in issues if str(item.get("severity") or "error") == "error"]
    warnings = [item for item in issues if str(item.get("severity") or "error") != "error"]
    return errors, warnings


def _score(blocking_errors: Sequence[Dict[str, Any]], warnings: Sequence[Dict[str, Any]]) -> int:
    return max(0, 100 - min(80, len(blocking_errors) * 20) - min(20, len(warnings) * 5))


def _contract_result(package: str, issues: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    blocking_errors, warnings = _split_issues(issues)
    score = _score(blocking_errors, warnings)
    return {
        "package": package,
        "passed": not blocking_errors,
        "score": score,
        "blocking_errors": blocking_errors,
        "errors": blocking_errors,
        "warnings": warnings,
        "issues": list(issues),
    }


def _has_source_ref(item: Dict[str, Any]) -> bool:
    if item.get("source_ref") or _as_list(item.get("source_refs")):
        return True
    source = _as_dict(item.get("source"))
    return bool(source.get("title") or source.get("url") or source.get("source"))


def _evidence_ref(item: Dict[str, Any]) -> str:
    return str(item.get("ref") or item.get("evidence_id") or "").strip()


def _is_weak_evidence(item: Dict[str, Any]) -> bool:
    level = str(item.get("source_level") or _as_dict(item.get("source")).get("credibility") or "").strip().upper()
    role = str(item.get("evidence_role") or item.get("role") or "").strip().lower()
    return level in {"C", "D"} or role in {"clue", "appendix", "appendix_only"}


def validate_report_blueprint(report_blueprint: Dict[str, Any]) -> Dict[str, Any]:
    issues: List[Dict[str, Any]] = []
    blueprint = _as_dict(report_blueprint)
    chapters = [chapter for chapter in _as_list(blueprint.get("chapters")) if isinstance(chapter, dict)]
    if not blueprint.get("report_family"):
        _issue(issues, package="report_blueprint", issue_type="missing_report_family", message="report_family is required.")
    if not blueprint.get("research_object"):
        _issue(issues, package="report_blueprint", issue_type="missing_research_object", message="research_object is required.", severity="warning")
    if not blueprint.get("narrative"):
        _issue(issues, package="report_blueprint", issue_type="missing_narrative", message="narrative is required.", severity="warning")
    if not chapters:
        _issue(issues, package="report_blueprint", issue_type="chapters_empty", message="At least one chapter is required.")
    shell = _as_dict(blueprint.get("report_shell"))
    if not shell:
        _issue(issues, package="report_blueprint", issue_type="missing_report_shell", message="report_shell is required for dynamic report structure.", severity="warning")
    else:
        if not _as_list(shell.get("front_blocks")):
            _issue(issues, package="report_blueprint", issue_type="missing_front_blocks", message="report_shell.front_blocks should not be empty.", severity="warning")
        if not _as_list(shell.get("back_blocks")):
            _issue(issues, package="report_blueprint", issue_type="missing_back_blocks", message="report_shell.back_blocks should not be empty.", severity="warning")

    seen_ids = set()
    titles = set()
    for index, chapter in enumerate(chapters):
        path = f"chapters[{index}]"
        chapter_id = str(chapter.get("chapter_id") or "").strip()
        title = str(chapter.get("chapter_title") or chapter.get("title") or "").strip()
        question = str(chapter.get("chapter_question") or "").strip()
        titles.add(title)
        if not chapter_id:
            _issue(issues, package="report_blueprint", issue_type="missing_chapter_id", message="chapter_id is required.", path=path)
        elif chapter_id in seen_ids:
            _issue(issues, package="report_blueprint", issue_type="duplicate_chapter_id", message="chapter_id must be unique.", path=path, chapter_id=chapter_id)
        seen_ids.add(chapter_id)
        if not title:
            _issue(issues, package="report_blueprint", issue_type="missing_chapter_title", message="chapter_title is required.", path=path)
        elif title in FIXED_FIVE_DIMENSIONS:
            _issue(issues, package="report_blueprint", issue_type="legacy_fixed_chapter_title", message="Legacy fixed five-dimension title is forbidden.", path=path)
        elif re.fullmatch(r"(?:第[一二三四五六七八九十0-9]+章|chapter\s*\d+|章节\s*\d+)", title, flags=re.I):
            _issue(issues, package="report_blueprint", issue_type="empty_chapter_title", message="chapter_title must express a research question, not just chapter order.", path=path)
        if not question:
            _issue(issues, package="report_blueprint", issue_type="missing_chapter_question", message="Each chapter must have chapter_question.", path=path)
        if not str(chapter.get("core_question") or question or "").strip():
            _issue(issues, package="report_blueprint", issue_type="missing_core_question", message="Each chapter must have core_question.", path=path)
        if not _as_list(chapter.get("required_evidence_mix")):
            _issue(issues, package="report_blueprint", issue_type="missing_required_evidence_mix", message="Each chapter must declare required_evidence_mix.", path=path)
        layout_policy = _as_dict(chapter.get("layout_policy"))
        if not _as_list(layout_policy.get("preferred_blocks")):
            _issue(issues, package="report_blueprint", issue_type="missing_preferred_blocks", message="Each chapter should declare layout_policy.preferred_blocks.", path=path, severity="warning")
        if int(chapter.get("min_total_sources") or 0) < 4:
            _issue(issues, package="report_blueprint", issue_type="weak_min_total_sources", message="Each chapter should require at least four total sources.", path=path)
        if int(chapter.get("min_ab_sources") or 0) < 1:
            _issue(issues, package="report_blueprint", issue_type="weak_min_ab_sources", message="Each chapter should require at least one A/B source.", path=path)
        if not str(chapter.get("chapter_role") or "").strip():
            _issue(issues, package="report_blueprint", issue_type="missing_chapter_role", message="chapter_role is recommended.", path=path, severity="warning")

    if titles and FIXED_FIVE_DIMENSIONS.issubset(titles):
        _issue(
            issues,
            package="report_blueprint",
            issue_type="fixed_five_dimension_pattern",
            message="Blueprint matches the legacy fixed five-dimension pattern; PreLayoutAgent must use question-driven chapters.",
        )
    return _contract_result("report_blueprint", issues)


def validate_profile_contract(report_blueprint: Dict[str, Any], micro_layouts: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    issues: List[Dict[str, Any]] = []
    blueprint = _as_dict(report_blueprint)
    profile_name = str(_as_dict(blueprint.get("layout_strategy")).get("profile") or blueprint.get("report_family") or "")
    contract = quality_contract_for_profile(profile_name)
    must_blocks = {str(item) for item in _as_list(contract.get("must_have_blocks")) if str(item).strip()}
    must_roles = {str(item) for item in _as_list(contract.get("must_have_evidence_roles")) if str(item).strip()}
    block_types = {
        str(_as_dict(block).get("block_type") or "").strip()
        for layout in list(micro_layouts or [])
        if isinstance(layout, dict)
        for block in _as_list(layout.get("blocks"))
    }
    evidence_roles = {
        str(role or "").strip()
        for chapter in _as_list(blueprint.get("chapters"))
        if isinstance(chapter, dict)
        for role in _as_list(chapter.get("required_evidence_mix"))
    }
    for block in sorted(must_blocks):
        if block not in block_types:
            _issue(
                issues,
                package="profile_contract",
                issue_type="missing_profile_block",
                message=f"Profile expects block '{block}'.",
                severity="warning",
                block_type=block,
            )
    for role in sorted(must_roles):
        if role not in evidence_roles:
            _issue(
                issues,
                package="profile_contract",
                issue_type="missing_profile_evidence_role",
                message=f"Profile expects evidence role '{role}'.",
                severity="warning",
                evidence_role=role,
            )
    return _contract_result("profile_contract", issues)


def validate_chapter_evidence_packages(chapter_evidence_packages: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    issues: List[Dict[str, Any]] = []
    packages = [package for package in list(chapter_evidence_packages or []) if isinstance(package, dict)]
    if not packages:
        _issue(issues, package="chapter_evidence_packages", issue_type="empty", message="No chapter evidence packages were produced.")
    for package_index, package in enumerate(packages):
        base_path = f"[{package_index}]"
        if package.get("omit_from_report"):
            continue
        if not str(package.get("chapter_id") or "").strip():
            _issue(issues, package="chapter_evidence_packages", issue_type="missing_chapter_id", message="chapter_id is required.", path=base_path)
        for item_index, item in enumerate(_as_list(package.get("core_evidence"))):
            if not isinstance(item, dict):
                continue
            path = f"{base_path}.core_evidence[{item_index}]"
            level = str(item.get("source_level") or _as_dict(item.get("source")).get("credibility") or "").strip().upper()
            if not _evidence_ref(item):
                _issue(issues, package="chapter_evidence_packages", issue_type="core_evidence_missing_ref", message="core_evidence must have evidence ref.", path=path)
            if not _has_source_ref(item):
                _issue(issues, package="chapter_evidence_packages", issue_type="core_evidence_missing_source", message="core_evidence must have source/source_ref.", path=path)
            if level == "D":
                _issue(issues, package="chapter_evidence_packages", issue_type="d_level_in_core", message="D-level evidence cannot enter core_evidence.", path=path)
            if _is_weak_evidence(item):
                _issue(issues, package="chapter_evidence_packages", issue_type="weak_evidence_in_core", message="Weak evidence should be appendix_evidence, not core_evidence.", path=path)
        for conflict_index, conflict in enumerate(_as_list(package.get("conflicts"))):
            conflict = _as_dict(conflict)
            if not (conflict.get("description") or conflict.get("reason") or conflict.get("conflict_type")):
                _issue(issues, package="chapter_evidence_packages", issue_type="conflict_missing_reason", message="conflicts must keep a reason or description.", path=f"{base_path}.conflicts[{conflict_index}]")
        if not _as_list(package.get("core_evidence")) and _as_list(package.get("missing_evidence")):
            _issue(issues, package="chapter_evidence_packages", issue_type="missing_core_evidence", message="Chapter has no core_evidence; downstream should use follow_up_queries.", path=base_path, severity="warning")
        quality = _as_dict(package.get("evidence_quality_summary"))
        if int(quality.get("core_ab_source_count") or 0) < 2 and _as_list(package.get("core_evidence")):
            _issue(
                issues,
                package="chapter_evidence_packages",
                issue_type="low_ab_core_coverage",
                message="Core evidence has fewer than two A/B-level sources; key claims should remain review_required.",
                path=base_path,
                severity="warning",
            )
    return _contract_result("chapter_evidence_packages", issues)


def validate_micro_layouts(micro_layouts: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    issues: List[Dict[str, Any]] = []
    layouts = [layout for layout in list(micro_layouts or []) if isinstance(layout, dict)]
    if not layouts:
        _issue(issues, package="micro_layouts", issue_type="empty", message="No micro layouts were produced.")
    for layout_index, layout in enumerate(layouts):
        base_path = f"[{layout_index}]"
        if not str(layout.get("chapter_id") or "").strip():
            _issue(issues, package="micro_layouts", issue_type="missing_chapter_id", message="chapter_id is required.", path=base_path)
        if not str(layout.get("layout_type") or "").strip():
            _issue(issues, package="micro_layouts", issue_type="missing_layout_type", message="layout_type is required.", path=base_path)
        blocks = [block for block in _as_list(layout.get("blocks")) if isinstance(block, dict)]
        valid_blocks = set(valid_block_types())
        if not blocks:
            _issue(issues, package="micro_layouts", issue_type="blocks_empty", message="MicroLayout should emit block plans, not only sections.", path=base_path, severity="warning")
        for block_index, block in enumerate(blocks):
            block_type = str(block.get("block_type") or "").strip()
            block_path = f"{base_path}.blocks[{block_index}]"
            if not block_type:
                _issue(issues, package="micro_layouts", issue_type="missing_block_type", message="block_type is required.", path=block_path)
            elif block_type not in valid_blocks:
                _issue(issues, package="micro_layouts", issue_type="unknown_block_type", message=f"Unknown block_type: {block_type}", path=block_path)
            if not _as_list(block.get("required_evidence_roles")):
                _issue(issues, package="micro_layouts", issue_type="block_missing_required_roles", message="block should declare required_evidence_roles.", path=block_path, severity="warning")
        sections = [section for section in _as_list(layout.get("sections")) if isinstance(section, dict)]
        followups = _as_list(layout.get("follow_up_queries"))
        if not sections:
            _issue(issues, package="micro_layouts", issue_type="sections_empty", message="At least one section is required.", path=base_path)
        for section_index, section in enumerate(sections):
            path = f"{base_path}.sections[{section_index}]"
            if not str(section.get("section_title") or "").strip():
                _issue(issues, package="micro_layouts", issue_type="missing_section_title", message="section_title is required.", path=path)
            if not _as_list(section.get("required_evidence_refs")) and not followups:
                _issue(issues, package="micro_layouts", issue_type="section_missing_evidence_refs", message="section must bind required_evidence_refs, or layout must provide follow_up_queries.", path=path)
        for request_index, request in enumerate(_as_list(layout.get("table_requests"))):
            request = _as_dict(request)
            if not str(request.get("purpose") or "").strip():
                _issue(issues, package="micro_layouts", issue_type="table_request_missing_purpose", message="table_requests must have purpose.", path=f"{base_path}.table_requests[{request_index}]")
    return _contract_result("micro_layouts", issues)


def validate_table_packages(table_packages: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    issues: List[Dict[str, Any]] = []
    rendered_by_chapter: Dict[str, int] = {}
    for index, table in enumerate(list(table_packages or [])):
        if not isinstance(table, dict):
            continue
        path = f"[{index}]"
        if not str(table.get("table_id") or "").strip():
            _issue(issues, package="table_packages", issue_type="missing_table_id", message="table_id is required.", path=path)
        if not str(table.get("chapter_id") or "").strip():
            _issue(issues, package="table_packages", issue_type="missing_chapter_id", message="chapter_id is required.", path=path)
        should_render = bool(table.get("should_render"))
        non_render_severity = "error" if should_render else "warning"
        if not str(table.get("purpose") or "").strip():
            _issue(issues, package="table_packages", issue_type="missing_purpose", message="purpose is required.", path=path, severity=non_render_severity)
        if not str(table.get("takeaway") or "").strip():
            _issue(issues, package="table_packages", issue_type="missing_takeaway", message="takeaway is required.", path=path, severity=non_render_severity)
        validation = validate_table_package(table)
        for error in _as_list(validation.get("errors")):
            _issue(
                issues,
                package="table_packages",
                issue_type="table_validation_error",
                message=str(_as_dict(error).get("type") or error),
                path=path,
                severity=non_render_severity,
                detail=error,
            )
        for warning in _as_list(validation.get("warnings")):
            _issue(issues, package="table_packages", issue_type="table_validation_warning", message=str(_as_dict(warning).get("type") or warning), path=path, severity="warning", detail=warning)
        if should_render:
            chapter_id = str(table.get("chapter_id") or "")
            rendered_by_chapter[chapter_id] = rendered_by_chapter.get(chapter_id, 0) + 1
    for chapter_id, count in rendered_by_chapter.items():
        try:
            max_tables_per_chapter = int(os.getenv("REPORT_MAX_BODY_TABLES_PER_CHAPTER", "3") or 3)
        except (TypeError, ValueError):
            max_tables_per_chapter = 3
        if count > max_tables_per_chapter:
            _issue(
                issues,
                package="table_packages",
                issue_type="many_body_tables_per_chapter",
                message="Chapter renders many body tables; check whether narrative flow still reads naturally.",
                chapter_id=chapter_id,
                count=count,
                limit=max_tables_per_chapter,
                severity="warning",
            )
    return _contract_result("table_packages", issues)


def validate_argument_units(argument_units: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    issues: List[Dict[str, Any]] = []
    units = [unit for unit in list(argument_units or []) if isinstance(unit, dict)]
    if not units:
        _issue(issues, package="argument_units", issue_type="empty", message="No argument units were produced.")
    for index, unit in enumerate(units):
        path = f"[{index}]"
        if unit.get("omit_from_report"):
            continue
        question = str(unit.get("question") or unit.get("section_title") or "").strip()
        claim = str(unit.get("claim") or "").strip()
        reasoning = str(unit.get("reasoning") or "").strip()
        counter = str(unit.get("counter_evidence") or "").strip()
        actionable = str(unit.get("actionable") or unit.get("decision_implication") or "").strip()
        if not question:
            _issue(issues, package="argument_units", issue_type="missing_question", message="question is required.", path=path)
        if not claim:
            _issue(issues, package="argument_units", issue_type="missing_claim", message="claim is required.", path=path)
        elif claim.startswith(WEAK_CLAIM_PREFIXES):
            _issue(issues, package="argument_units", issue_type="weak_claim_prefix", message="claim must be a judgment, not evidence-status wording.", path=path)
        elif any(re.search(pattern, claim) for pattern in BAD_CLAIM_PATTERNS):
            _issue(issues, package="argument_units", issue_type="weak_claim_pattern", message="claim contains internal evidence-status wording.", path=path)
        if not reasoning:
            _issue(issues, package="argument_units", issue_type="missing_reasoning", message="reasoning is required.", path=path)
        elif any(re.search(pattern, reasoning) for pattern in BAD_CLAIM_PATTERNS):
            _issue(issues, package="argument_units", issue_type="weak_reasoning_pattern", message="reasoning contains fallback/internal wording.", path=path)
        elif not any(word in reasoning for word in CAUSE_WORDS):
            _issue(issues, package="argument_units", issue_type="reasoning_missing_causal_chain", message="reasoning should include a causal explanation.", path=path, severity="warning")
        if not counter:
            _issue(issues, package="argument_units", issue_type="missing_counter_evidence", message="counter_evidence is required.", path=path)
        elif any(re.search(pattern, counter) for pattern in BAD_CLAIM_PATTERNS):
            _issue(issues, package="argument_units", issue_type="weak_counter_pattern", message="counter_evidence contains fallback/internal wording.", path=path)
        if not actionable:
            _issue(issues, package="argument_units", issue_type="missing_actionable", message="actionable is required.", path=path)
        elif any(re.search(pattern, actionable) for pattern in BAD_CLAIM_PATTERNS):
            _issue(issues, package="argument_units", issue_type="weak_actionable_pattern", message="actionable contains fallback/internal wording.", path=path)
        elif not any(word in actionable for word in ACTION_WORDS):
            _issue(issues, package="argument_units", issue_type="actionable_missing_action_word", message="actionable should contain a concrete action/verification verb.", path=path)
        if not _as_list(unit.get("evidence_refs")):
            _issue(issues, package="argument_units", issue_type="missing_evidence_refs", message="evidence_refs is required.", path=path)
    return _contract_result("argument_units", issues)


def validate_chapter_packages(chapter_packages: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    issues: List[Dict[str, Any]] = []
    packages = [package for package in list(chapter_packages or []) if isinstance(package, dict)]
    if not packages:
        _issue(issues, package="chapter_packages", issue_type="empty", message="No chapter packages were produced.")
    for index, chapter in enumerate(packages):
        path = f"[{index}]"
        if chapter.get("omit_from_report"):
            continue
        if not str(chapter.get("lead") or "").strip():
            _issue(issues, package="chapter_packages", issue_type="missing_lead", message="lead is required.", path=path)
        sections = [section for section in _as_list(chapter.get("sections")) if isinstance(section, dict)]
        public_tables = [
            table
            for table in _as_list(chapter.get("table_packages"))
            if isinstance(table, dict) and table.get("should_render")
        ]
        if not sections and not public_tables:
            _issue(issues, package="chapter_packages", issue_type="sections_empty", message="sections are required.", path=path)
        for section_index, section in enumerate(sections):
            section_path = f"{path}.sections[{section_index}]"
            if not str(section.get("section_title") or "").strip():
                _issue(issues, package="chapter_packages", issue_type="missing_section_title", message="section_title is required.", path=section_path)
            if not str(section.get("claim") or "").strip():
                _issue(issues, package="chapter_packages", issue_type="missing_section_claim", message="section claim is required.", path=section_path)
            elif any(re.search(pattern, str(section.get("claim") or "")) for pattern in BAD_CLAIM_PATTERNS):
                _issue(issues, package="chapter_packages", issue_type="weak_section_claim", message="section claim contains evidence-status/fallback wording.", path=section_path)
            if not str(section.get("counter_evidence") or "").strip():
                _issue(issues, package="chapter_packages", issue_type="missing_section_counter", message="each chapter section needs counter/boundary.", path=section_path)
            if not _as_list(section.get("evidence_refs")):
                _issue(issues, package="chapter_packages", issue_type="missing_section_evidence_refs", message="section must keep evidence_refs.", path=section_path)
    return _contract_result("chapter_packages", issues)


def validate_pipeline_packages(
    *,
    report_blueprint: Optional[Dict[str, Any]] = None,
    chapter_evidence_packages: Optional[Sequence[Dict[str, Any]]] = None,
    micro_layouts: Optional[Sequence[Dict[str, Any]]] = None,
    table_packages: Optional[Sequence[Dict[str, Any]]] = None,
    argument_units: Optional[Sequence[Dict[str, Any]]] = None,
    chapter_packages: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    results = [
        validate_report_blueprint(_as_dict(report_blueprint)),
        validate_chapter_evidence_packages(list(chapter_evidence_packages or [])),
        validate_micro_layouts(list(micro_layouts or [])),
        validate_profile_contract(_as_dict(report_blueprint), list(micro_layouts or [])),
        validate_table_packages(list(table_packages or [])),
        validate_argument_units(list(argument_units or [])),
        validate_chapter_packages(list(chapter_packages or [])),
    ]
    errors = [issue for result in results for issue in _as_list(result.get("blocking_errors") or result.get("errors"))]
    warnings = [issue for result in results for issue in _as_list(result.get("warnings"))]
    score = max(0, 100 - min(80, len(errors) * 12) - min(20, len(warnings) * 3))
    scores = {
        "blueprint_score": int(_as_dict(results[0]).get("score") or 0),
        "evidence_score": int(_as_dict(results[1]).get("score") or 0),
        "micro_layout_score": int(_as_dict(results[2]).get("score") or 0),
        "profile_contract_score": int(_as_dict(results[3]).get("score") or 0),
        "table_score": int(_as_dict(results[4]).get("score") or 0),
        "claim_score": int(_as_dict(results[5]).get("score") or 0),
        "chapter_score": int(_as_dict(results[6]).get("score") or 0),
    }
    return {
        "passed": not errors,
        "quality_score": score,
        "blocking_errors": errors,
        "package_results": results,
        "errors": errors,
        "warnings": warnings,
        "scores": scores,
        "summary": {
            "package_count": len(results),
            "error_count": len(errors),
            "warning_count": len(warnings),
        },
    }
