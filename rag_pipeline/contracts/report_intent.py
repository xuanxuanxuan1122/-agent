from __future__ import annotations

import re
from typing import Any, Iterable, List, Sequence


PROFESSION_EDUCATION_INTENT = "profession_education_employment"
PROFESSION_EDUCATION_DOMAIN = "academic_or_professional_field"

PROFESSION_EDUCATION_FORBIDDEN_TERMS: tuple[str, ...] = (
    "市场规模",
    "市场空间",
    "竞争格局",
    "主要玩家",
    "头部玩家",
    "玩家格局",
    "商业化",
    "商业模式",
    "商业化落地",
    "机会排序",
    "投资",
    "投资判断",
    "投资机会",
    "资本",
    "资本流向",
    "估值",
    "融资",
    "毛利率",
    "价格战",
    "客户付费",
    "市占率",
    "市场份额",
    "财报",
    "招股书",
    "投资者关系",
)

PROFESSION_EDUCATION_FRAME_TERMS: tuple[str, ...] = (
    "就业",
    "岗位",
    "岗位能力",
    "能力要求",
    "人才培养",
    "教育培养",
    "课程",
    "培养方案",
    "高校",
    "职业教育",
    "学生",
    "毕业生",
    "招聘",
    "职业资格",
    "数字财务",
    "智能财税",
    "财务共享",
)

PROFESSION_EDUCATION_TITLE_BY_CLUSTER: dict[str, str] = {
    "market": "就业需求与岗位结构变化",
    "metric": "就业需求与岗位结构变化",
    "metric_claim": "就业需求与岗位结构变化",
    "customer": "企业用人标准与岗位能力要求",
    "competition": "企业用人标准与岗位能力要求",
    "technology": "AI工具推动的能力重构",
    "technology_product": "AI工具推动的能力重构",
    "mechanism_claim": "AI工具推动的能力重构",
    "policy": "政策导向与培养体系调整",
    "filing": "政策导向与培养体系调整",
    "case": "院校课程与企业实践样本",
    "case_claim": "院校课程与企业实践样本",
    "risk": "就业风险与培养边界",
    "counter": "就业风险与培养边界",
    "counter_boundary_claim": "就业风险与培养边界",
    "limitations": "适用边界与待验证问题",
    "contextual_claim": "就业与培养环境变化",
    "core_claim": "就业变化的核心判断",
}

PROFESSION_EDUCATION_CHAPTER_FRAMES: tuple[dict[str, Any], ...] = (
    {
        "chapter_id": "ch_employment",
        "chapter_title": "就业结构变化与岗位需求验证",
        "core_question": "AI正在怎样改变会计学专业毕业生面向的岗位类型、招聘需求和就业结构？",
        "triggers": ("就业", "毕业生", "招聘", "岗位需求", "就业变化"),
        "default_when_ai": True,
    },
    {
        "chapter_id": "ch_capability",
        "chapter_title": "岗位能力要求与工作任务重构",
        "core_question": "会计岗位能力正在从哪些基础任务转向数据解释、流程治理、内控风控和业务协同？",
        "triggers": ("岗位能力", "能力要求", "职业能力", "工作任务", "技能", "岗位"),
        "default_when_ai": True,
    },
    {
        "chapter_id": "ch_education",
        "chapter_title": "课程体系与人才培养方案调整",
        "core_question": "高校和职业教育体系需要如何调整课程、实训和评价方式以匹配AI时代的会计岗位？",
        "triggers": ("教育培养", "人才培养", "课程", "培养方案", "高校", "职业教育", "课程改革"),
        "always_for_profession": True,
    },
    {
        "chapter_id": "ch_employer",
        "chapter_title": "企业用人标准与招聘信号变化",
        "core_question": "企业招聘和实际业务场景正在如何改变会计人才的技能组合和实践能力要求？",
        "triggers": ("企业", "用人", "招聘", "校企", "实习", "就业基地", "事务所"),
    },
    {
        "chapter_id": "ch_boundary",
        "chapter_title": "就业风险、能力错配与培养边界",
        "core_question": "AI替代、技能错配、院校资源差异和监管责任会给会计学就业与培养带来哪些边界？",
        "triggers": ("风险", "替代", "错配", "边界", "冲击", "失业"),
        "default_when_ai": True,
    },
    {
        "chapter_id": "ch_policy",
        "chapter_title": "政策导向与职业资格变化",
        "core_question": "政策、职业资格和行业标准正在如何影响会计人才培养与能力评价？",
        "triggers": ("政策", "财政部", "资格", "证书", "标准", "评价"),
    },
    {
        "chapter_id": "ch_practice",
        "chapter_title": "AI财税工具与实践教学场景",
        "core_question": "智能财税、数字财务和财务共享工具正在如何进入课程、实训和岗位实践？",
        "triggers": ("AI", "人工智能", "智能财税", "数字财务", "财务共享", "RPA", "工具", "实训"),
    },
)

_COMPACT_SPLIT_RE = re.compile(r"[\s,;:!?，。；：！？、/\\|()\[\]{}（）【】《》\"']+")
_PROFESSION_QUERY_RE = re.compile(
    r"(会计学|会计专业|大数据与会计|财务管理|审计学|专业|学科|课程|人才培养|培养方案|教育培养|"
    r"高校|职业教育|学生|毕业生|就业|招聘|岗位|岗位能力|职业能力|能力要求|数字财务|智能财税|财务共享)"
)
_PROFESSION_PAIR_RE = re.compile(
    r"(专业|学科|课程|高校|职业教育|会计学|会计专业).{0,24}(就业|岗位|能力|培养|课程|招聘|毕业生)|"
    r"(就业|岗位|能力|培养|课程|招聘|毕业生).{0,24}(专业|学科|课程|高校|职业教育|会计学|会计专业)"
)
_COMMERCIAL_FRAME_RE = re.compile("|".join(re.escape(term) for term in PROFESSION_EDUCATION_FORBIDDEN_TERMS))


def _text_values(values: Iterable[Any]) -> List[str]:
    result: List[str] = []
    for value in values:
        if isinstance(value, dict):
            result.extend(_text_values(value.values()))
        elif isinstance(value, (list, tuple, set)):
            result.extend(_text_values(value))
        else:
            text = re.sub(r"\s+", " ", str(value or "")).strip()
            if text:
                result.append(text)
    return result


def unique_text(values: Iterable[Any], *, limit: int = 64) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if not text:
            continue
        key = re.sub(r"\s+", "", text.lower())
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def is_profession_education_employment_topic(*values: Any) -> bool:
    texts = _text_values(values)
    if not texts:
        return False
    compact = re.sub(r"\s+", "", " ".join(texts))
    if PROFESSION_EDUCATION_DOMAIN in compact or PROFESSION_EDUCATION_INTENT in compact:
        return True
    if _PROFESSION_PAIR_RE.search(compact):
        return True
    hits = {term for term in PROFESSION_EDUCATION_FRAME_TERMS if term in compact}
    subject_hits = {term for term in ("会计学", "会计专业", "专业", "学科", "高校", "职业教育") if term in compact}
    action_hits = {term for term in ("就业", "岗位", "能力要求", "岗位能力", "人才培养", "教育培养", "课程", "招聘") if term in compact}
    return bool(len(hits) >= 3 or (subject_hits and action_hits))


def profession_forbidden_terms() -> List[str]:
    return list(PROFESSION_EDUCATION_FORBIDDEN_TERMS)


def contains_commercial_frame_term(value: Any) -> bool:
    return bool(_COMMERCIAL_FRAME_RE.search(str(value or "")))


def filter_profession_terms(values: Sequence[Any], *, fallback_terms: Sequence[str] = ()) -> List[str]:
    filtered: List[str] = []
    for value in values:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if not text or contains_commercial_frame_term(text):
            continue
        filtered.append(text)
    if not filtered:
        filtered.extend(fallback_terms)
    return unique_text(filtered, limit=16)


def preferred_profession_query_terms(*values: Any, limit: int = 6) -> List[str]:
    text = " ".join(_text_values(values))
    candidates: List[str] = []
    if re.search(r"会计学|会计专业|财务管理|审计学|智能财税|数字财务|财务共享", text):
        candidates.extend(["会计学", "会计专业", "就业", "岗位能力", "人才培养", "课程"])
    for term in PROFESSION_EDUCATION_FRAME_TERMS:
        if term in text:
            candidates.append(term)
    if not candidates:
        candidates.extend(["就业", "岗位能力", "人才培养", "课程"])
    return unique_text(candidates, limit=limit)


def sanitize_profession_query(query: Any) -> str:
    text = re.sub(r"\s+", " ", str(query or "")).strip()
    if not text:
        return ""
    for term in PROFESSION_EDUCATION_FORBIDDEN_TERMS:
        text = text.replace(term, " ")
    parts = [part for part in _COMPACT_SPLIT_RE.split(text) if part]
    return " ".join(unique_text(parts, limit=24))


def profession_cluster_title(cluster_key: Any, fallback_text: Any = "") -> str:
    key = re.sub(r"[^a-z0-9_]+", "_", str(cluster_key or "").strip().lower()).strip("_")
    if key in PROFESSION_EDUCATION_TITLE_BY_CLUSTER:
        return PROFESSION_EDUCATION_TITLE_BY_CLUSTER[key]
    fallback = str(fallback_text or "")
    if "课程" in fallback or "培养" in fallback or "高校" in fallback:
        return "课程体系与培养方案调整"
    if "岗位" in fallback or "就业" in fallback or "招聘" in fallback:
        return "就业需求与岗位能力变化"
    if "风险" in fallback or "替代" in fallback or "错配" in fallback:
        return "就业风险与培养边界"
    return "就业变化与能力重构"


def profession_chapter_frames(query: Any = "") -> List[dict[str, str]]:
    text = re.sub(r"\s+", "", str(query or ""))
    has_ai_context = bool(re.search(r"AI|人工智能|智能财税|数字财务|财务共享|RPA", text, re.I))
    selected: List[dict[str, str]] = []
    for frame in PROFESSION_EDUCATION_CHAPTER_FRAMES:
        triggers = tuple(str(item) for item in frame.get("triggers", ()))
        matched = any(trigger and trigger in text for trigger in triggers)
        if matched or frame.get("always_for_profession") or (has_ai_context and frame.get("default_when_ai")):
            selected.append(
                {
                    "chapter_id": str(frame["chapter_id"]),
                    "chapter_title": str(frame["chapter_title"]),
                    "core_question": str(frame["core_question"]),
                }
            )
    if not selected:
        selected = [
            {
                "chapter_id": "ch_core",
                "chapter_title": "就业变化与能力重构",
                "core_question": "该专业的就业方向、能力要求和培养方式正在发生哪些变化？",
            },
            {
                "chapter_id": "ch_education",
                "chapter_title": "课程体系与人才培养方案调整",
                "core_question": "课程、实训和评价方式需要如何匹配新的岗位能力要求？",
            },
        ]
    if len(selected) == 1:
        selected.append(
            {
                "chapter_id": "ch_practice",
                "chapter_title": "实践教学与岗位场景连接",
                "core_question": "相关变化如何落到课程实训、校企合作和岗位实践中？",
            }
        )
    return selected[:6]
