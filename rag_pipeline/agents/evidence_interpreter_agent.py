from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _dedupe(values: Iterable[Any], *, limit: int = 8) -> List[str]:
    result: List[str] = []
    seen: set[str] = set()
    for value in values:
        text = _text(value)
        if not text:
            continue
        key = re.sub(r"\W+", "", text.lower())[:180]
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
        if len(result) >= limit:
            break
    return result


_DIRTY_RE = re.compile(
    r"("
    r"\blog[\s_-]?in\b|\bsign[\s_-]?in\b|\bskip\s+to\s+content\b|"
    r"\bhttp\s*404\b|\b404\b.*\bnot\s+found\b|\b403\b|\b500\b|"
    r"diagnostic_only|score_gap|source_check|review_suggestion|"
    r"\bqa\b.*\bfatal\b|\bfatal\b.*\bqa\b"
    r")",
    re.I,
)


def _fact_id(item: Dict[str, Any]) -> str:
    return _text(item.get("evidence_id") or item.get("fact_id") or item.get("id") or item.get("ref"))


def _fact_text(item: Dict[str, Any]) -> str:
    card = _as_dict(item.get("public_fact_card")) or _as_dict(_as_dict(item.get("public_fact_quality")).get("public_fact_card"))
    return _text(
        item.get("distilled_fact")
        or item.get("clean_fact")
        or card.get("distilled_fact")
        or item.get("fact")
        or item.get("summary")
        or item.get("content")
    )


def _source_id(item: Dict[str, Any]) -> str:
    source = _as_dict(item.get("source"))
    return _text(item.get("source_id") or item.get("canonical_source_id") or source.get("source_id") or source.get("id"))


def _requirement_id(item: Dict[str, Any]) -> str:
    lineage = _as_dict(item.get("lineage"))
    return _text(item.get("requirement_id") or lineage.get("requirement_id") or item.get("evidence_goal_id"))


def _topic_fit(item: Dict[str, Any]) -> str:
    value = _text(item.get("topic_fit")).lower()
    if value in {"direct", "related", "background"}:
        return value
    return "direct"


def _proof_role(item: Dict[str, Any]) -> str:
    return _text(item.get("proof_role") or item.get("analysis_role") or item.get("fact_type") or item.get("allowed_use")).lower()


def _is_dirty_fact(item: Dict[str, Any]) -> bool:
    if item.get("diagnostic_only") or item.get("must_not_render") or item.get("public_text_allowed") is False:
        return True
    text = " ".join(
        [
            _fact_text(item),
            _text(item.get("source_title")),
            _text(item.get("source_url")),
            _text(_as_dict(item.get("source")).get("title")),
        ]
    )
    return bool(_DIRTY_RE.search(text))


def _cluster_key(item: Dict[str, Any], chapter_question: str = "") -> str:
    role = _proof_role(item)
    text = f"{_fact_text(item)} {chapter_question}"
    if role in {"counter", "risk", "boundary"} or re.search(r"风险|合规|安全|监管|隐私|幻觉|边界|替代压力", text):
        return "risk_boundary"
    if re.search(r"就业|岗位|招聘|职位|人才|人效|服务户数|替代|复合型", text):
        return "employment_shift"
    if re.search(r"教育|高校|课程|培养|专业|实训|学生|院校", text):
        return "education_adjustment"
    if re.search(r"能力|审计|数据|流程|治理|内控|分析", text):
        return "capability_shift"
    if role in {"metric", "market", "source_check", "filing", "official_data"} or re.search(r"规模|增速|投入|营收|市场|占比|增长", text):
        return "market_signal"
    if role in {"policy", "standard"} or re.search(r"政策|规划|标准|财政部|政府", text):
        return "policy_signal"
    return "capability_shift"


def _strength(items: Sequence[Dict[str, Any]], *, background_only: bool) -> str:
    if background_only:
        return "weak"
    unique_sources = {_source_id(item) for item in items if _source_id(item)}
    direct_count = sum(1 for item in items if _topic_fit(item) == "direct")
    if len(unique_sources) >= 2 and direct_count >= 2:
        return "moderate"
    if items:
        return "directional"
    return "weak"


def _sentence(value: str) -> str:
    text = _text(value)
    if not text:
        return ""
    return text if re.search(r"[。！？!?]$", text) else f"{text}。"


def _subject_from_question(chapter_question: str) -> str:
    text = _text(chapter_question)
    if re.search(r"县域|充换电|重卡|商用车|物流|运营商", text):
        return "县域商用车充换电网络"
    if re.search(r"MES|APS|PLM|工业软件|国产替代|离散制造", text, re.I):
        return "工业软件国产替代"
    if re.search(r"医药|CXO|生物安全法案|出海订单|产能利用率", text, re.I):
        return "医药CXO出海订单恢复"
    if re.search(r"会计|财务|审计|就业|岗位|培养|课程", text):
        return "会计职业与教育"
    return "研究对象"


def _is_accounting_context(chapter_question: str) -> bool:
    return bool(re.search(r"会计|财务|审计|就业|岗位|培养|课程", _text(chapter_question)))


def _fact_summary(value: str, *, max_chars: int = 180) -> str:
    text = _text(value)
    text = re.sub(r"^摘要[:：]\s*", "", text)
    text = re.sub(r"索引号[:：]\s*[^。；;]{0,80}", "", text)
    text = re.sub(r"组配分类[:：]\s*[^。；;]{0,80}", "", text)
    text = re.sub(r"发布机构[:：]\s*[^。；;]{0,80}", "", text)
    text = re.sub(r"生成日期[:：]\s*[^。；;]{0,80}", "", text)
    text = re.sub(r"文件编号[:：]\s*[^。；;]{0,80}", "", text)
    text = re.sub(r"统一编号[:：]\s*[^。；;]{0,80}", "", text)
    text = re.sub(r"有效性[:：]\s*[^。；;]{0,80}", "", text)
    text = _text(text).strip("；;，,。 ")
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit("，", 1)[0].rsplit("；", 1)[0].strip("；;，,。 ")
    return cut or text[:max_chars].strip()


def _judgment_for_cluster(cluster_key: str, *, chapter_question: str, background_only: bool) -> str:
    if background_only:
        return ""
    subject = _subject_from_question(chapter_question)
    if _is_accounting_context(chapter_question):
        mapping = {
            "employment_shift": "AI相关证据显示，会计岗位价值正在从基础处理转向工具协同、数据解释和流程治理",
            "capability_shift": "会计岗位能力要求正在从传统核算扩展到数字化审计、数据分析和业务判断",
            "education_adjustment": "会计教育培养正在被产业数智化需求倒逼，课程与实训需要更贴近AI工具和数据治理场景",
            "risk_boundary": "AI进入会计场景后，合规、数据安全和职业判断成为决定技术应用边界的关键变量",
            "market_signal": "会计信息化与AI应用的公开数据正在显示真实投入和需求验证信号",
            "policy_signal": "政策与监管信号正在改变会计数智化转型的边界和落地节奏",
        }
        return mapping.get(cluster_key, "相关证据显示，会计行业正在围绕AI工具、业务流程和人才能力发生结构性调整")
    mapping = {
        "employment_shift": f"{subject}的组织能力正在向复合运营、数据判断和现场协同迁移",
        "capability_shift": f"{subject}的落地能力正在从单点建设转向运营数据、调度能力和组织协同",
        "education_adjustment": f"{subject}的人才与能力供给需要匹配新的运营和技术要求",
        "risk_boundary": f"{subject}的推进边界取决于成本、利用率、安全和地方配套条件",
        "market_signal": f"{subject}的公开数据正在显示需求、供给或商业化验证信号",
        "policy_signal": f"政策规划和地方配套正在改变{subject}的落地节奏",
    }
    return mapping.get(cluster_key, f"相关证据显示，{subject}正在经历从概念推进到场景验证的结构变化")


def _implications(cluster_key: str, chapter_question: str) -> Tuple[str, str, str]:
    if _is_accounting_context(chapter_question):
        if cluster_key == "employment_shift":
            return (
                "就业机会会更集中在能配置工具、解释数据并连接业务流程的复合型岗位。",
                "培养体系需要减少单纯核算训练的比重，增加AI工具、数据治理和业务场景训练。",
                "会计服务机构的竞争焦点会从人力堆叠转向工具化交付、流程治理和专业判断能力。",
            )
        if cluster_key == "education_adjustment":
            return (
                "毕业生竞争力会取决于能否把财务知识嵌入真实数字化流程。",
                "高校和职业院校需要把AI审计、数据分析、内控判断和案例实训纳入核心培养环节。",
                "教育端调整速度会影响行业获得复合型人才的效率。",
            )
        if cluster_key == "risk_boundary":
            return (
                "岗位价值不会消失，而会向风险识别、审计判断和责任划分迁移。",
                "培养体系需要把技术伦理、模型风险和数据安全纳入会计专业训练。",
                "行业采用AI工具的速度会受到合规、隐私和可追溯要求约束。",
            )
        if cluster_key == "market_signal":
            return (
                "就业需求会随着企业实际投入和场景部署向数字化财务岗位集中。",
                "培养体系需要关注企业真实预算和岗位变化，而不是只追逐概念课程。",
                "行业是否进入规模化阶段，取决于投入、效率提升和可复制场景能否持续出现。",
            )
        return (
            "岗位需求会向理解工具、流程和业务结果的人才倾斜。",
            "培养体系需要围绕真实业务流程组织课程，而不是只增加工具介绍。",
            "行业结构变化会先出现在流程标准化程度较高的场景中。",
        )
    subject = _subject_from_question(chapter_question)
    if cluster_key == "risk_boundary":
        return (
            f"{subject}相关企业需要把风险识别和资源约束纳入运营决策。",
            "人才能力需要覆盖政策理解、成本测算、运营调度和现场问题处理。",
            "行业扩张速度会受到利用率、投资回收、电网接入和地方配套条件约束。",
        )
    if cluster_key in {"market_signal", "policy_signal"}:
        return (
            f"{subject}的需求验证要同时观察政策推进、车辆/设备供给和真实运营数据。",
            "组织能力需要从项目建设转向持续运营、数据复盘和跨主体协同。",
            "行业能否规模化，取决于需求密度、资产周转和商业闭环是否同步改善。",
        )
    return (
        f"{subject}的参与者需要围绕场景需求、运营效率和成本结构重新配置能力。",
        "人才与组织建设应围绕真实场景问题，而不是停留在概念或单点技术介绍。",
        "行业结构变化会先出现在资源配套较完整、运营数据可复盘的场景中。",
    )


def _build_unit(
    *,
    chapter_id: str,
    chapter_question: str,
    index: int,
    cluster_key: str,
    items: Sequence[Dict[str, Any]],
    background_only: bool = False,
) -> Dict[str, Any]:
    facts = _dedupe([_fact_summary(_fact_text(item)) for item in items], limit=8)
    fact_ids = _dedupe([_fact_id(item) for item in items], limit=8)
    requirement_ids = _dedupe([_requirement_id(item) for item in items], limit=8)
    source_ids = _dedupe([_source_id(item) for item in items], limit=8)
    core_judgment = _judgment_for_cluster(cluster_key, chapter_question=chapter_question, background_only=background_only)
    evidence_summary = "；".join(facts[:3])
    employment, education, industry = _implications(cluster_key, chapter_question)
    if background_only:
        what_reflects = ""
        why = ""
        mechanism = []
        counter = f"{evidence_summary} 只能作为背景材料，不能单独支撑本章核心判断。".strip()
    else:
        what_reflects = f"{evidence_summary}。这些信号共同指向{core_judgment.rstrip('。')}。"
        why = (
            "这类变化重要，是因为它会同时影响资源配置、运营效率和后续商业化节奏。"
            if not _is_accounting_context(chapter_question)
            else "这类变化重要，是因为它同时改变企业用人标准、会计服务交付方式和教育端能力训练重点。"
        )
        mechanism = [
            _sentence(f"证据首先显示，{facts[0] if facts else core_judgment}"),
            (
                "当政策、场景需求和运营数据开始同时出现，项目判断就会从建设可行性转向持续运营效率。"
                if not _is_accounting_context(chapter_question)
                else "当重复处理环节被工具吸收后，人工价值会向系统配置、结果解释和风险判断迁移。"
            ),
            (
                "这种变化会把基础设施投入、资产周转、客户需求和盈利模型连接到同一条商业闭环中。"
                if not _is_accounting_context(chapter_question)
                else "这种迁移会把就业、课程和机构交付能力连接到同一条能力升级链上。"
            ),
        ]
        counter = (
            "公开证据主要反映已披露样本和局部场景，不宜直接外推为全国所有县域同步成熟。"
            if not _is_accounting_context(chapter_question)
            else "公开证据主要反映已披露样本和局部场景，不宜外推为所有岗位同步变化。"
        )
    return {
        "schema_version": "evidence_interpretation_unit_v1",
        "interpretation_id": f"INT-{chapter_id or 'chapter'}-{index:03d}",
        "chapter_id": chapter_id,
        "requirement_ids": requirement_ids,
        "fact_ids": fact_ids,
        "source_ids": source_ids,
        "cluster_key": cluster_key,
        "core_judgment": _sentence(core_judgment) if core_judgment else "",
        "what_evidence_reflects": _sentence(what_reflects),
        "why_it_matters": _sentence(why),
        "mechanism_chain": mechanism,
        "employment_implication": _sentence(employment) if not background_only else "",
        "education_implication": _sentence(education) if not background_only else "",
        "industry_implication": _sentence(industry) if not background_only else "",
        "counter_reading": _sentence(counter),
        "claim_strength": _strength(items, background_only=background_only),
        "writing_angle": "围绕证据组合写成判断、机制和影响，不单独罗列事实。" if not background_only else "",
        "single_fact_interpretation": len(fact_ids) == 1,
        "diagnostic_only": False,
        "public_text_allowed": True,
    }


def build_evidence_interpretation_units(
    *,
    chapter_id: str,
    chapter_question: str = "",
    fact_cards: Sequence[Dict[str, Any]],
    max_units: int = 6,
) -> Dict[str, Any]:
    unique_cards: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    duplicate_fact_count = 0
    blocked_dirty_count = 0
    for raw in list(fact_cards or []):
        item = _as_dict(raw)
        fact_id = _fact_id(item)
        if not fact_id or not _fact_text(item):
            continue
        if fact_id in seen_ids:
            duplicate_fact_count += 1
            continue
        seen_ids.add(fact_id)
        if _is_dirty_fact(item):
            blocked_dirty_count += 1
            continue
        unique_cards.append(item)

    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for item in unique_cards:
        topic_fit = _topic_fit(item)
        key = (
            _cluster_key(item, chapter_question),
            "background" if topic_fit == "background" else "claimable",
        )
        grouped.setdefault(key, []).append(item)

    units: List[Dict[str, Any]] = []
    index = 1
    for (cluster_key, lane), items in grouped.items():
        for start in range(0, len(items), 8):
            if len(units) >= max_units:
                break
            chunk = items[start : start + 8]
            units.append(
                _build_unit(
                    chapter_id=chapter_id,
                    chapter_question=chapter_question,
                    index=index,
                    cluster_key=cluster_key,
                    items=chunk,
                    background_only=lane == "background",
                )
            )
            index += 1
        if len(units) >= max_units:
            break

    fact_group_coverage = (
        sum(len(_as_list(unit.get("fact_ids"))) for unit in units) / max(1, len(unique_cards))
        if unique_cards
        else 0.0
    )
    return {
        "interpretation_units": units,
        "diagnostics": {
            "schema_version": "evidence_interpretation_diagnostics_v1",
            "chapter_id": chapter_id,
            "input_fact_count": len(list(fact_cards or [])),
            "unique_fact_count": len(unique_cards),
            "duplicate_fact_count": duplicate_fact_count,
            "blocked_dirty_fact_count": blocked_dirty_count,
            "interpretation_unit_count": len(units),
            "fact_group_coverage_rate": round(fact_group_coverage, 4),
            "avg_facts_per_interpretation": round(
                sum(len(_as_list(unit.get("fact_ids"))) for unit in units) / max(1, len(units)),
                3,
            ),
        },
    }
