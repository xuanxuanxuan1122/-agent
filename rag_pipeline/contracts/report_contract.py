from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

from rag_pipeline.contracts.requirement_quality import validate_requirement_quality

try:
    from rag_pipeline.agents.block_schema import select_blocks_for_chapter
except Exception:  # pragma: no cover - contract can still work without agent helpers
    select_blocks_for_chapter = None  # type: ignore


CONTRACT_VERSION = "0.1.0"

LEGACY_CHAPTER_TITLE_PATTERNS = [
    r"\u5e02\u573a\u89c4\u6a21",
    r"\u7ade\u4e89\u683c\u5c40",
    r"\u653f\u7b56.*\u76d1\u7ba1",
    r"\u6280\u672f\u8def\u7ebf",
    r"\u4ea7\u4e1a\u94fe",
    r"\u6295\u878d\u8d44",
    r"\u8d44\u672c\u52a8\u6001",
]


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _compact(value: Any, max_chars: int = 260) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "..."


def _dedupe(values: Iterable[Any], *, limit: int = 40, max_chars: int = 220) -> List[str]:
    result: List[str] = []
    seen = set()
    iterable = [values] if isinstance(values, str) else (values or [])
    for value in iterable:
        text = _compact(value, max_chars)
        if not text:
            continue
        key = re.sub(r"\s+", "", text.lower())
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _title_issues(title: Any, *, query: Any = "") -> List[Dict[str, Any]]:
    text = str(title or "").strip()
    issues: List[Dict[str, Any]] = []
    if not text:
        issues.append({"type": "chapter_title_missing", "severity": "warning"})
        return issues
    if re.search(r"[?\uff1f]\s*$", text):
        issues.append({"type": "chapter_title_question_tail", "severity": "rewrite"})
    query_text = str(query or "")
    for pattern in LEGACY_CHAPTER_TITLE_PATTERNS:
        if re.search(pattern, text) and not re.search(pattern, query_text):
            issues.append({"type": "legacy_template_chapter_title", "severity": "rewrite", "pattern": pattern})
            break
    if len(text) > 42:
        issues.append({"type": "chapter_title_too_long", "severity": "rewrite"})
    return issues


def _normalize_chapter_title(title: Any, *, fallback: Any = "") -> str:
    text = re.sub(r"\s+", " ", str(title or "").strip())
    text = re.sub(r"^[#\s\d一二三四五六七八九十、.．\-:：]+", "", text)
    text = re.sub(r"[?\uff1f]\s*$", "", text).strip()
    if not text:
        text = re.sub(r"[?\uff1f]\s*$", "", str(fallback or "").strip())
    if len(text) > 42:
        text = text[:42].rstrip(" ,;:!?，。；：！？")
    return text


def _title_needs_derivation(title: str, *, query: Any = "", core_question: Any = "") -> bool:
    text = str(title or "").strip()
    if not text:
        return True
    if len(text) > 42 or re.search(r"[?\uff1f]\s*$", text):
        return True
    query_text = re.sub(r"\s+", "", str(query or "").strip())
    question_text = re.sub(r"\s+", "", str(core_question or "").strip())
    compact = re.sub(r"\s+", "", text)
    return bool((len(query_text) >= 12 and query_text in compact) or (len(question_text) >= 12 and question_text in compact))


def _derive_chapter_title(raw_title: Any, *, core_question: Any = "", query: Any = "") -> str:
    normalized = _normalize_chapter_title(raw_title, fallback=core_question or query)
    raw_text = str(raw_title or core_question or query or "").strip()
    raw_needs_derivation = len(raw_text) > 42 or bool(re.search(r"[?\uff1f]\s*$", raw_text))
    if not raw_needs_derivation and not _title_needs_derivation(normalized, query=query, core_question=core_question):
        return normalized
    source = " ".join(str(item or "") for item in [raw_title, core_question, query])
    patterns = [
        (r"真实需求|概念热度|需求", "真实需求验证"),
        (r"价格|产能|订单|盈利|行情", "行情支撑与盈利质量"),
        (r"商业化|概念|试点|量产", "商业化进展与概念边界"),
        (r"进入|投资|产品布局|优先级|反证|校准", "布局优先级与反证校准"),
        (r"技术瓶颈|良率|铰链|柔性|UTG|OLED", "技术瓶颈与突破路径"),
        (r"产业机会|供应链|产业链|价值链", "产业机会与供应链位置"),
        (r"风险|失效|边界|不及预期", "风险边界与失效条件"),
    ]
    for pattern, title in patterns:
        if re.search(pattern, source, re.I):
            return title
    cleaned = re.sub(r"[?\uff1f]", " ", source)
    cleaned = re.sub(r"是否|如何|为什么|哪些|那些|以及|又有|有没有|存在|必须|得到", " ", cleaned)
    chunks = [chunk.strip(" ，。；：,.;:") for chunk in re.split(r"[，。；：,.;\n]|\s{2,}", cleaned) if chunk.strip()]
    chunks = [chunk for chunk in chunks if 4 <= len(chunk) <= 24]
    if chunks:
        return chunks[0]
    return normalized[:24] if normalized else "核心判断验证"


def _minimum_source_level(chapter: Dict[str, Any], proof_standard: str) -> str:
    explicit = str(chapter.get("minimum_source_level") or chapter.get("min_source_level") or "").strip().upper()
    if explicit in {"A", "B", "C", "D"}:
        return explicit
    if _to_int(chapter.get("min_ab_sources"), 0) >= 2 or str(proof_standard).lower() in {"strong", "strict", "high"}:
        return "A"
    return "B"


def _required_evidence_roles(
    chapter: Dict[str, Any],
    chapter_goals: Sequence[Dict[str, Any]],
    chapter_tasks: Sequence[Dict[str, Any]],
) -> List[str]:
    roles: List[Any] = []
    roles.extend(_as_list(chapter.get("required_evidence_roles")))
    roles.extend(_as_list(chapter.get("required_evidence_mix") or chapter.get("source_template_keys")))
    for goal in chapter_goals:
        roles.append(goal.get("proof_role") or goal.get("evidence_type"))
        roles.extend(_as_list(goal.get("required_evidence_mix")))
    for task in chapter_tasks:
        roles.append(task.get("proof_role") or task.get("evidence_type"))
        roles.extend(_as_list(task.get("source_priority")))
    return _dedupe([*roles, "metric", "source_check", "counter"], limit=12, max_chars=80)


def _expected_blocks(chapter: Dict[str, Any], roles: Sequence[str]) -> List[str]:
    base_blocks = ["thesis", "evidence_matrix", "mechanism_chain", "risk_trigger"]
    if select_blocks_for_chapter is None:
        blocks = [*base_blocks, "verification_checklist"]
        return _dedupe(blocks, limit=5, max_chars=80)
    block_payloads = select_blocks_for_chapter({**chapter, "required_evidence_mix": list(roles)}, limit=5)
    return _dedupe([*base_blocks, *[item.get("block_type") for item in block_payloads if isinstance(item, dict)]], limit=7, max_chars=80)


def _contract_default_issues(raw_chapter: Dict[str, Any]) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    if not _as_list(raw_chapter.get("required_evidence_roles")) and not _as_list(raw_chapter.get("required_evidence_mix")):
        issues.append({"type": "chapter_required_evidence_roles_defaulted", "severity": "warning"})
    if not str(raw_chapter.get("minimum_source_level") or raw_chapter.get("min_source_level") or "").strip():
        issues.append({"type": "chapter_minimum_source_level_defaulted", "severity": "warning"})
    if not _as_list(raw_chapter.get("expected_blocks")):
        issues.append({"type": "chapter_expected_blocks_defaulted", "severity": "warning"})
    return issues


def _research_plan_from_package(package: Dict[str, Any]) -> Dict[str, Any]:
    evidence_package = _as_dict(package.get("evidence_package"))
    structured_analysis = _as_dict(package.get("structured_analysis"))
    report_blueprint = _as_dict(package.get("report_blueprint"))
    return (
        _as_dict(package.get("research_plan"))
        or _as_dict(structured_analysis.get("research_plan"))
        or _as_dict(evidence_package.get("research_plan"))
        or _as_dict(_as_dict(evidence_package.get("metadata")).get("research_plan"))
        or _as_dict(report_blueprint.get("research_plan"))
    )


def _chapters_from_research_plan(research_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    chapters: List[Dict[str, Any]] = []
    goals = [_as_dict(item) for item in _as_list(research_plan.get("evidence_goals")) if isinstance(item, dict)]
    tasks = [_as_dict(item) for item in _as_list(research_plan.get("search_tasks")) if isinstance(item, dict)]
    for index, raw in enumerate(_as_list(research_plan.get("chapters")), start=1):
        chapter = _as_dict(raw)
        if not chapter:
            continue
        chapter_id = _compact(chapter.get("chapter_id") or chapter.get("id") or f"ch_{index:02d}", 80)
        chapter_goals = [_as_dict(item) for item in _as_list(chapter.get("evidence_goals")) if isinstance(item, dict)]
        chapter_tasks = [_as_dict(item) for item in _as_list(chapter.get("search_tasks")) if isinstance(item, dict)]
        if not chapter_goals:
            chapter_goals = [item for item in goals if str(item.get("chapter_id") or "") == chapter_id]
        if not chapter_tasks:
            chapter_tasks = [item for item in tasks if str(item.get("chapter_id") or "") == chapter_id]
        proof_standard = _compact(chapter.get("proof_standard") or chapter.get("min_proof_standard") or "", 80)
        if not proof_standard:
            proof_standard = "strong" if _to_int(chapter.get("min_ab_sources"), 0) >= 2 else "medium"
        core_question = _compact(chapter.get("core_question") or chapter.get("chapter_question") or chapter.get("question"), 260)
        raw_title = chapter.get("chapter_title") or chapter.get("title") or chapter.get("dimension_name")
        title = _derive_chapter_title(raw_title, core_question=core_question, query=research_plan.get("query"))
        required_roles = _required_evidence_roles(chapter, chapter_goals, chapter_tasks)
        chapters.append(
            {
                "chapter_id": chapter_id,
                "title": _compact(title, 180),
                "chapter_title": _compact(title, 180),
                "core_question": core_question,
                "reason_to_include": _compact(chapter.get("reason_to_include") or chapter.get("purpose"), 360),
                "proof_standard": proof_standard,
                "required_evidence_mix": _dedupe(chapter.get("required_evidence_mix") or chapter.get("source_template_keys")),
                "required_evidence_roles": required_roles,
                "minimum_source_level": _minimum_source_level(chapter, proof_standard),
                "expected_blocks": _expected_blocks(chapter, required_roles),
                "title_policy_issues": _title_issues(title, query=research_plan.get("query") or core_question),
                "contract_default_issues": _contract_default_issues(chapter),
                "min_total_sources": _to_int(chapter.get("min_total_sources"), 0),
                "min_ab_sources": _to_int(chapter.get("min_ab_sources"), 0),
                "min_counter_sources": _to_int(chapter.get("min_counter_sources"), 0),
                "evidence_goal_ids": _dedupe(item.get("goal_id") or item.get("evidence_goal_id") for item in chapter_goals),
                "search_task_ids": _dedupe(item.get("task_id") or item.get("id") for item in chapter_tasks),
            }
        )
    return chapters


def _chapters_from_blueprint(report_blueprint: Dict[str, Any]) -> List[Dict[str, Any]]:
    chapters: List[Dict[str, Any]] = []
    for index, raw in enumerate(_as_list(report_blueprint.get("chapters")), start=1):
        chapter = _as_dict(raw)
        if not chapter:
            continue
        core_question = _compact(chapter.get("chapter_question") or chapter.get("core_question"), 260)
        raw_title = chapter.get("chapter_title") or chapter.get("title")
        title = _derive_chapter_title(raw_title, core_question=core_question, query=report_blueprint.get("query"))
        proof_standard = _compact(chapter.get("proof_standard") or "medium", 80)
        required_roles = _required_evidence_roles(chapter, [], [])
        chapters.append(
            {
                "chapter_id": _compact(chapter.get("chapter_id") or chapter.get("id") or f"ch_{index:02d}", 80),
                "title": _compact(title, 180),
                "chapter_title": _compact(title, 180),
                "core_question": core_question,
                "reason_to_include": _compact(chapter.get("reason_to_include"), 360),
                "proof_standard": proof_standard,
                "required_evidence_mix": _dedupe(chapter.get("required_evidence_mix") or chapter.get("source_template_keys")),
                "required_evidence_roles": required_roles,
                "minimum_source_level": _minimum_source_level(chapter, proof_standard),
                "expected_blocks": _expected_blocks(chapter, required_roles),
                "title_policy_issues": _title_issues(title, query=report_blueprint.get("query") or core_question),
                "contract_default_issues": _contract_default_issues(chapter),
                "min_total_sources": _to_int(chapter.get("min_total_sources"), 0),
                "min_ab_sources": _to_int(chapter.get("min_ab_sources"), 0),
                "min_counter_sources": _to_int(chapter.get("min_counter_sources"), 0),
                "evidence_goal_ids": _dedupe(chapter.get("evidence_goal_ids")),
                "search_task_ids": _dedupe(chapter.get("search_task_ids")),
            }
        )
    return chapters


def _fallback_chapters(query: str, research_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    dimensions = _as_list(research_plan.get("dimensions"))
    chapters: List[Dict[str, Any]] = []
    for index, raw in enumerate(dimensions, start=1):
        item = _as_dict(raw)
        raw_title = item.get("name") or item.get("dimension") or raw
        title = _derive_chapter_title(raw_title, core_question=item.get("question") or "", query=query)
        if not str(title or "").strip():
            continue
        core_question = _compact(item.get("question") or f"{title}如何影响核心判断", 260)
        proof_standard = _compact(item.get("proof_standard") or "medium", 80)
        required_roles = _required_evidence_roles(item, [], [])
        chapters.append(
            {
                "chapter_id": f"ch_{index:02d}",
                "title": _compact(title, 180),
                "chapter_title": _compact(title, 180),
                "core_question": core_question,
                "reason_to_include": "",
                "proof_standard": proof_standard,
                "required_evidence_mix": _dedupe(item.get("required_evidence_mix")),
                "required_evidence_roles": required_roles,
                "minimum_source_level": _minimum_source_level(item, proof_standard),
                "expected_blocks": _expected_blocks(item, required_roles),
                "title_policy_issues": _title_issues(title, query=query),
                "contract_default_issues": _contract_default_issues(item),
                "min_total_sources": _to_int(item.get("min_total_sources"), 0),
                "min_ab_sources": _to_int(item.get("min_ab_sources"), 0),
                "min_counter_sources": _to_int(item.get("min_counter_sources"), 0),
                "evidence_goal_ids": [],
                "search_task_ids": [],
            }
        )
    if chapters:
        return chapters
    return [
        {
            "chapter_id": "ch_01",
            "title": "核心研究问题",
            "chapter_title": "核心研究问题",
            "core_question": _compact(query, 260),
            "reason_to_include": "兜底章节合同，用于避免后续 Agent 缺少章节锚点。",
            "proof_standard": "medium",
            "required_evidence_mix": ["official_or_primary", "industry_or_research", "company_or_case"],
            "required_evidence_roles": ["support", "metric"],
            "minimum_source_level": "B",
            "expected_blocks": ["thesis", "evidence_matrix", "mechanism_chain", "verification_checklist"],
            "title_policy_issues": [],
            "min_total_sources": 5,
            "min_ab_sources": 1,
            "min_counter_sources": 0,
            "evidence_goal_ids": [],
            "search_task_ids": [],
        }
    ]


def _quality_thresholds(
    research_plan: Dict[str, Any],
    report_blueprint: Dict[str, Any],
    template: Dict[str, Any],
    chapters: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    quality_bar = _as_dict(template.get("quality_bar"))
    depth = str(research_plan.get("report_depth_target") or report_blueprint.get("report_depth_target") or "").lower()
    default_body_chars = 16000 if depth in {"deep", "full", "enterprise"} else 9000
    chapter_min_sources = max([_to_int(item.get("min_total_sources"), 0) for item in chapters] or [0])
    chapter_ab_sources = max([_to_int(item.get("min_ab_sources"), 0) for item in chapters] or [0])
    return {
        "minimum_body_chars": _to_int(quality_bar.get("minimum_body_chars"), default_body_chars),
        "minimum_unique_sources_when_available": _to_int(
            quality_bar.get("minimum_unique_sources_when_available"),
            max(8, chapter_min_sources),
        ),
        "minimum_ab_sources_per_core_claim": max(1, chapter_ab_sources),
        "max_uncited_substantive_paragraph_ratio": quality_bar.get("max_uncited_substantive_paragraph_ratio", 0.35),
        "requires_counter_signal_per_core_chapter": bool(
            quality_bar.get("requires_counter_signal_per_core_chapter")
            or any(_to_int(item.get("min_counter_sources"), 0) > 0 for item in chapters)
        ),
    }


ROLE_REQUIRED_FIELDS = {
    "metric": ["metric", "value", "unit", "period", "scope", "source_ref"],
    "source_check": ["source_ref", "source_title", "source_url"],
    "counter": ["counter_signal", "source_ref"],
    "case": ["company", "use_case", "deployment_scope", "source_ref"],
    "customer_case": ["company", "use_case", "deployment_scope", "source_ref"],
    "technology": ["capability", "constraint", "source_ref"],
    "technology_product": ["capability", "constraint", "source_ref"],
    "competitive_positioning": ["player", "positioning_signal", "source_ref"],
    "support": ["fact", "source_ref"],
}


def _role_source_level(role: str, chapter: Dict[str, Any]) -> str:
    role = str(role or "").strip().lower()
    minimum = str(chapter.get("minimum_source_level") or "B").strip().upper() or "B"
    if role in {"metric", "source_check"}:
        return "A" if minimum == "A" else "B"
    if role in {"case", "customer_case", "counter", "technology", "technology_product"}:
        return "C"
    return minimum if minimum in {"A", "B", "C", "D"} else "B"


def _role_strength_ceiling(role: str, min_source_level: str) -> str:
    role = str(role or "").strip().lower()
    min_source_level = str(min_source_level or "").strip().upper()
    if role in {"metric", "source_check"} and min_source_level in {"A", "B"}:
        return "moderate"
    if min_source_level == "A":
        return "moderate"
    return "directional"


def _requirement_slots_for_chapters(chapters: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    requirements: List[Dict[str, Any]] = []
    seen = set()
    for chapter in chapters or []:
        chapter_id = str(chapter.get("chapter_id") or "").strip()
        if not chapter_id:
            continue
        for role in _as_list(chapter.get("required_evidence_roles")):
            proof_role = re.sub(r"[^A-Za-z0-9_\-\u4e00-\u9fff]+", "_", str(role or "").strip().lower()).strip("_")
            if not proof_role:
                continue
            requirement_id = f"{chapter_id}_{proof_role}"
            if requirement_id in seen:
                continue
            seen.add(requirement_id)
            min_source_level = _role_source_level(proof_role, chapter)
            requirement = {
                "requirement_id": requirement_id,
                "chapter_id": chapter_id,
                "hypothesis_id": f"{chapter_id}_H1",
                "proof_role": proof_role,
                "required_fields": ROLE_REQUIRED_FIELDS.get(proof_role, ROLE_REQUIRED_FIELDS["support"]),
                "min_source_level": min_source_level,
                "source_family_preference": _as_list(chapter.get("required_evidence_mix")),
                "claim_strength_ceiling": _role_strength_ceiling(proof_role, min_source_level),
                "repair_policy": {
                    "route": "targeted_repair",
                    "max_tasks_per_chapter": 1,
                },
            }
            quality = validate_requirement_quality(requirement)
            requirement.update(
                {
                    "requirement_quality_check": {
                        "quality_check_version": quality.get("quality_check_version"),
                        "status": quality.get("status"),
                        "issues": quality.get("issues"),
                    },
                    "source_strategy": quality.get("source_strategy"),
                    "success_criteria": quality.get("success_criteria"),
                    "reject_if": quality.get("reject_if"),
                }
            )
            requirements.append(
                requirement
            )
    return requirements


def _package_evidence_items(package: Dict[str, Any]) -> List[Dict[str, Any]]:
    evidence_package = _as_dict(package.get("evidence_package"))
    items: List[Dict[str, Any]] = []
    for key in (
        "analysis_ready_evidence",
        "clean_evidence_list",
        "evidence_analyses",
        "fact_cards",
    ):
        items.extend(item for item in _as_list(evidence_package.get(key)) if isinstance(item, dict))
    for chapter_package in _as_list(evidence_package.get("chapter_evidence_packages")):
        chapter = _as_dict(chapter_package)
        for key, value in chapter.items():
            if key.endswith("_evidence") or key in {"core_evidence", "supporting_evidence", "directional_evidence"}:
                items.extend(item for item in _as_list(value) if isinstance(item, dict))
    return items


def _item_requirement_ids(item: Dict[str, Any]) -> List[str]:
    values: List[Any] = []
    lineage = _as_dict(item.get("lineage"))
    for key in ("requirement_ids", "requirement_id", "evidence_requirement_ids", "evidence_requirement_id", "slot_id"):
        value = lineage.get(key) if key in lineage else item.get(key)
        if isinstance(value, list):
            values.extend(value)
        elif value not in (None, ""):
            values.append(value)
    return _dedupe(values, limit=12, max_chars=100)


def _item_ref(item: Dict[str, Any]) -> str:
    return _compact(item.get("evidence_id") or item.get("id") or item.get("ref") or item.get("source_ref"), 100)


def _item_source_level(item: Dict[str, Any]) -> str:
    return str(item.get("source_level") or _as_dict(item.get("evidence_card")).get("source_level") or "C").strip().upper()


def _item_analysis_eligible(item: Dict[str, Any]) -> bool:
    if "analysis_eligible" in item:
        return bool(item.get("analysis_eligible"))
    card = _as_dict(item.get("evidence_card"))
    if "analysis_eligible" in card:
        return bool(card.get("analysis_eligible"))
    return True


def _requirement_status_matrix(requirements: Sequence[Dict[str, Any]], evidence_items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for item in evidence_items or []:
        payload = _as_dict(item)
        if not payload or not _item_analysis_eligible(payload):
            continue
        for requirement_id in _item_requirement_ids(payload):
            grouped.setdefault(requirement_id, []).append(payload)
    matrix: List[Dict[str, Any]] = []
    for requirement in requirements or []:
        requirement_id = str(requirement.get("requirement_id") or "").strip()
        proof_role = str(requirement.get("proof_role") or "").strip()
        matched = grouped.get(requirement_id, [])
        matched_refs = _dedupe([_item_ref(item) for item in matched if _item_ref(item)], limit=20, max_chars=100)
        if not matched_refs:
            matrix.append(
                {
                    "requirement_id": requirement_id,
                    "chapter_id": requirement.get("chapter_id"),
                    "proof_role": proof_role,
                    "status": "needs_repair",
                    "matched_fact_refs": [],
                    "missing": [proof_role or "evidence"],
                    "can_generate_claim": False,
                    "claim_strength_ceiling": "none",
                    "repair_tasks": [
                        {
                            "proof_role": proof_role,
                            "required_fields": _as_list(requirement.get("required_fields")),
                            "reason": "no_matching_fact_card",
                        }
                    ],
                }
            )
            continue
        levels = {_item_source_level(item) for item in matched}
        directional = bool(levels - {"A", "B"}) or proof_role in {"case", "customer_case", "counter", "technology", "technology_product"}
        matrix.append(
            {
                "requirement_id": requirement_id,
                "chapter_id": requirement.get("chapter_id"),
                "proof_role": proof_role,
                "status": "directional_ready" if directional else "ready",
                "matched_fact_refs": matched_refs,
                "missing": [],
                "can_generate_claim": True,
                "claim_strength_ceiling": "directional" if directional else str(requirement.get("claim_strength_ceiling") or "moderate"),
                "repair_tasks": [],
            }
        )
    return matrix


def build_report_contract(
    *,
    query: str,
    research_plan: Optional[Dict[str, Any]] = None,
    report_blueprint: Optional[Dict[str, Any]] = None,
    template: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    research_plan = _as_dict(research_plan)
    report_blueprint = _as_dict(report_blueprint)
    template = _as_dict(template)
    query_text = _compact(query or research_plan.get("query") or report_blueprint.get("query"), 360)
    chapters = _chapters_from_research_plan(research_plan) or _chapters_from_blueprint(report_blueprint)
    if not chapters:
        chapters = _fallback_chapters(query_text, research_plan)
    source_requirements = _as_dict(research_plan.get("source_requirements"))
    coverage_requirements = _as_dict(research_plan.get("evidence_coverage_requirements"))
    contract_issues = [
        {
            **issue,
            "chapter_id": chapter.get("chapter_id"),
            "chapter_title": chapter.get("chapter_title") or chapter.get("title"),
        }
        for chapter in chapters
        for issue in [*_as_list(chapter.get("title_policy_issues")), *_as_list(chapter.get("contract_default_issues"))]
    ]
    return {
        "contract_version": CONTRACT_VERSION,
        "contract_type": "report_contract",
        "status": "active" if query_text else "draft",
        "query": query_text,
        "research_object": _compact(
            research_plan.get("research_object") or report_blueprint.get("research_object") or query_text,
            220,
        ),
        "core_question": _compact(research_plan.get("core_question") or query_text, 360),
        "decision_context": _compact(research_plan.get("decision_context") or "question_driven_research", 120),
        "report_family": _compact(
            research_plan.get("report_family") or report_blueprint.get("report_family") or "dynamic_research_report",
            120,
        ),
        "layout_policy": {
            "chapter_strategy": "question_driven",
            "legacy_five_dimension_default": False,
            "title_style": "noun_phrase_without_question_tail",
        },
        "chapters": chapters,
        "evidence_requirements": {
            "global_required_terms": _dedupe(research_plan.get("global_required_terms"), limit=20, max_chars=80),
            "global_forbidden_terms": _dedupe(research_plan.get("global_forbidden_terms"), limit=20, max_chars=80),
            "source_requirements": source_requirements,
            "coverage_requirements": coverage_requirements,
            "requirements": _requirement_slots_for_chapters(chapters),
            "per_chapter": [
                {
                    "chapter_id": chapter.get("chapter_id"),
                    "min_total_sources": chapter.get("min_total_sources"),
                    "min_ab_sources": chapter.get("min_ab_sources"),
                    "min_counter_sources": chapter.get("min_counter_sources"),
                    "required_evidence_mix": chapter.get("required_evidence_mix"),
                    "required_evidence_roles": chapter.get("required_evidence_roles"),
                    "minimum_source_level": chapter.get("minimum_source_level"),
                    "expected_blocks": chapter.get("expected_blocks"),
                }
                for chapter in chapters
            ],
        },
        "contract_issues": contract_issues,
        "quality_thresholds": _quality_thresholds(research_plan, report_blueprint, template, chapters),
        "source_policy": {
            "source_appendix": _as_dict(template.get("default_policy")).get("source_appendix", "forbidden_by_default"),
            "citation_style": _as_dict(template.get("default_policy")).get("citation_style", "inline_fact_citation"),
            "prefer_primary_sources": True,
            "keep_conflicts_visible": True,
        },
        "repair_policy": {
            "qa_failed_with_evidence_gap": "return_to_evidence_refinement",
            "qa_failed_with_logic_gap": "return_to_rewrite",
            "reformatter_failed_with_enough_evidence": "rewrite_or_text_repair",
            "reformatter_failed_with_sparse_evidence": "evidence_refinement_before_degrade",
            **_as_dict(template.get("repair_policy")),
        },
        "degrade_policy": {
            "allowed_after_evidence_refinement_exhausted": True,
            "must_show_evidence_boundary": True,
            "must_not_publish_as_clean_report": True,
        },
    }


def build_report_contract_from_package(package: Dict[str, Any]) -> Dict[str, Any]:
    package = _as_dict(package)
    research_plan = _research_plan_from_package(package)
    report_blueprint = _as_dict(package.get("report_blueprint"))
    template = _as_dict(package.get("report_template")) or _as_dict(package.get("template"))
    contract = build_report_contract(
        query=str(package.get("query") or research_plan.get("query") or ""),
        research_plan=research_plan,
        report_blueprint=report_blueprint,
        template=template,
    )
    requirements = _as_list(_as_dict(contract.get("evidence_requirements")).get("requirements"))
    if requirements:
        contract["evidence_requirements"] = {
            **_as_dict(contract.get("evidence_requirements")),
            "requirement_status": _requirement_status_matrix(requirements, _package_evidence_items(package)),
        }
    return contract
