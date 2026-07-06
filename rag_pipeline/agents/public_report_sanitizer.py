from __future__ import annotations

import re
import os
from typing import Dict, List


PUBLIC_EV_ID_PATTERN = r"(?<![A-Za-z0-9_])EV-\d+(?:-[A-Za-z0-9]+)?"


HARD_INDUSTRY_TEMPLATE_REPLACEMENTS = (
    ("订单规模", "可复核规模"),
    ("订单落地", "执行验证"),
    ("订单兑现", "结果验证"),
    ("订单和运营频次", "持续行动"),
    ("后续订单", "后续可复核材料"),
    ("订单", "可复核材料"),
    ("客户付费转化", "持续使用转化"),
    ("客户付费意愿", "持续使用意愿"),
    ("客户付费", "持续使用"),
    ("商业化节奏", "推进情况"),
    ("商业化速度", "推进速度"),
    ("商业化判断", "主题判断"),
    ("商业化能力", "执行能力"),
    ("示范项目", "早期样本"),
    ("规模化部署", "更广泛应用"),
    ("部署节奏", "推进节奏"),
    ("出货/部署", "相关进展"),
    ("出货量", "数量"),
    ("出货", "数量"),
)


def remove_hard_industry_templates(text: str) -> str:
    """Remove fixed commercialization/shipment templates from public prose.

    These phrases are useful for some industry-analysis prompts, but they must not be
    injected as a universal fallback. If a later LLM chooses a domain-specific angle it
    should do so from evidence, not from these canned bridges.
    """
    cleaned = str(text or "")
    for old, new in HARD_INDUSTRY_TEMPLATE_REPLACEMENTS:
        cleaned = cleaned.replace(old, new)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


INTERNAL_GAP_PATTERNS = [
    r"证据不足",
    r"暂无可核验数据",
    r"暂无足够证据",
    r"低置信方向判断",
    r"低置信",
    r"不能作为确定性结论",
    r"无法判断",
    r"无法分析",
    r"不做判断",
    r"尚不能形成结论",
    r"需要补证",
    r"需补证",
    r"建议后续补充调研",
    r"建议补充",
    r"证据缺口",
    r"缺少.*证据",
    r"待验证事项",
    r"needs_corroboration",
    r"权威来源交叉验证",
    r"A/B\s*级来源不足",
    r"unsupported",
    r"insufficient evidence",
    r"当前表格证据不足",
    r"关联证据[:：]",
    r"本章可用来源约\d+条",
    r"A/B层级来源约\d+条",
    r"来源层级分布为",
    r"本章写作时应",
    r"当前最直接的支持点是",
    r"不作为每章正文",
    r"正文只保留",
    r"coverage_matrix",
    r"actual_ab_sources",
    r"required_ab_sources",
    r"blocking_gaps",
    r"insufficient_ab_sources",
    r"case_evidence_missing",
    r"counter_evidence_missing",
    r"metric_scope_period_unit_incomplete",
    r"evidence_refs",
    r"\bevidence_cards?\b",
    r"当前卡片",
    r"本章应写成",
    r"本章可以作为",
    r"本章可作为",
    r"正文\s*只能\s*写成",
    r"本章\s*只能\s*写成",
    r"本章\s*可\s*写成",
    r"本章\s*应\s*写成",
    r"本章\s*仍需\s*连续观察",
    r"建议避免",
    r"建议在后续版本中补充",
    r"建议写成",
    r"适合写成",
    r"claim_status",
    r"render_blocks",
    r"technology_maturity_to_adoption",
    r"definition_to_opportunity",
    r"demand_supply_risk",
    r"该信号需要同时穿过",
    r"当前可用事实包括",
    r"把反向触发器写入验证清单",
    r"建议动作[:：]",
    r"按\s+[a-z][a-z0-9_]+\s+组织",
    r"章节信号",
    r"事实转成判断",
    r"核心连接点",
    r"这一事实用于判断",
    r"成为影响本章判断的核心变量",
    r"放回章节问题看",
    r"不能只依赖单点事实",
    r"话题热度推进",
    r"可核验内容适合",
    r"可复核内容适合",
    r"局部变化的入口",
    r"暂时缺少覆盖的外推",
    r"待验证问题处理",
    r"相邻来源重复情况",
    r"结论强度才",
]
INTERNAL_GAP_PATTERNS.extend(
    [
        r"章节判断",
        r"关键事实速览",
        r"证据深读",
        r"本章结论",
        r"全球口径",
        r"中国口径",
        r"增速口径",
        r"可引用事实",
        r"机制与边界",
        r"进入综合决策章的变量",
        r"核心判断[:：]",
        r"机制拆解",
        r"反证边界",
        r"决策含义[:：]",
    ]
)


INTERNAL_GAP_PATTERNS.extend(
    [
        r"\bdiagnostic_only\b",
        r"\bscore_gap\b",
        r"\bmissing_proof_standard\b",
        r"\brepair_task_seed\b",
        r"\bsearch_more\b",
        r"\breanalyze_existing\b",
        r"\brecompose_outline\b",
        r"\brewrite_with_caveat\b",
        r"\bmust_not_render\b",
        r"\breview_suggestion\b",
        r"\bpublic_text_allowed\s*=\s*false\b",
        r"\bsource_check\b",
        r"\bsemantic_judge\b",
        r"\bexecutor_should_decide\b",
        "\u8865\u8bc1\u5efa\u8bae",
        "\u5ba1\u67e5\u5efa\u8bae",
        "\u5185\u90e8\u8bca\u65ad",
        "\u4ec5\u4f9b\u8bca\u65ad",
        "\u4e0d\u5f97\u8fdb\u5165\u6b63\u6587",
    ]
)


SAFE_PUBLIC_TERMS = [
    "全球口径",
    "中国口径",
    "增速口径",
    "机制与边界",
    "机制拆解",
    "决策含义",
    "可引用事实",
]

INTERNAL_GAP_PATTERNS.extend(
    [
        r"证据不足",
        r"不能作为确定性结论",
        r"无法作为确定性结论",
        r"只能作为方向性判断",
        r"建议补证",
        r"建议补充",
        r"建议避免",
        r"后续版本中补充",
        r"本章只能写成",
        r"正文只能写成",
        r"本章应写成",
        r"事实锚点显示",
        r"可复核网页线索如下",
    ]
)


STRICT_PUBLICATION_BLOCKERS = [
    r"\bIQS\b",
    r"(?<![A-Za-z0-9])IQS(?![A-Za-z0-9])",
    r"\b(?:Writer|Review|Reformatter|Rewrite|Supervisor|Evidence|Table|Claim|Chapter|Brain)\s*Agent\b",
    r"(?<![A-Za-z0-9])(?:Writer|Review|Reformatter|Rewrite|Supervisor|Evidence|Table|Claim|Chapter|Brain)\s*Agent(?![A-Za-z0-9])",
    r"联网分析\s*Agent",
    r"多\s*Agent\s*(?:协作|流程|写作|生成|审查|校验|引用|证据|检索|处理)",
    r"(?:本报告|本文|研究|系统|流程|正文|章节|写作|生成|输出|审查|清洗|重写|校验|引用)[^。；\n]{0,30}\bAgent\b",
    r"Agent\s*(?:失败|输出|节点|流程|通道|审查|清洗|重写)",
    r"\bAgent\b[^。；\n]{0,30}(?:协作|流程|写作|生成|输出|审查|清洗|重写|校验|引用|证据|检索|处理)",
    r"\bRAG\s*(?:流程|管线|证据|状态|输出|通道)",
    r"(?<![A-Za-z0-9])RAG(?![A-Za-z0-9])\s*(?:流程|管线|证据|状态|输出|通道)",
    r"大模型[^。；\n]{0,40}(?:未启用|未成功调用|调用失败|失败|报错)",
    r"\bQA\s*(?:审查|校验|流程|结果)",
    r"已通过\s*IQS",
    r"联网证据",
    r"网页结果摘要",
    r"可核验网页线索",
    r"证据不足",
    r"证据缺口",
    r"证据门槛",
    r"证据池",
    r"证据包",
    r"证据绑定",
    r"补证",
    r"补充检索",
    r"检索任务",
    r"检索线索",
    r"覆盖率",
    r"质量门槛",
    r"发布门槛",
    r"\bcoverage\b",
    r"\bfollowup\b",
    r"\bevidence_refs\b",
    r"\bclaim_status\b",
    r"\brender_blocks\b",
    r"\bnot_ready\b",
    r"no publishable",
    PUBLIC_EV_ID_PATTERN,
    r"当前材料",
    r"当前证据",
    r"当前可用事实",
    r"材料中已经",
    r"本章可用来源",
    r"A/B\s*级来源不足",
    r"暂无可核验",
    r"可核验数据",
    r"尚不足以形成",
    r"低置信",
    r"待验证事项",
    r"不能作为确定性结论",
    r"无法形成",
    r"建议后续补充",
    r"后续补充调研",
    r"需要补充[^。；\n]{0,30}(?:来源|证据|调研|检索)",
    r"需补充[^。；\n]{0,30}(?:来源|证据|调研|检索)",
    r"章节判断",
    r"证据深读",
    r"关键事实速览",
    r"可引用事实",
    r"证据引用",
    r"(?:证据|事实)链管理",
    r"引用准确率",
    r"报告结构完整度",
    r"事实校验流程",
    r"内部(?:处理|逻辑|流程|标签|字段)",
    r"(?:处理|运行|生成|写作|审查|检索|搜索|召回|重排|清洗|重写)逻辑",
    r"(?:本报告|本文|生成|写作|清洗|审查|校验|引用|证据|检索|搜索|召回|重排|处理)[^。；\n]{0,30}(?:工具调用|向量检索|联网搜索|搜索通道|抓取|爬取)",
    r"(?:本报告|本文|生成|写作|清洗|审查|校验|引用|证据|检索|搜索|召回|重排|处理)[^。；\n]{0,30}\b(?:prompt|chunk|rerank|retrieval|self[-_ ]?refine)\b",
    r"进入综合决策章",
    r"正文只保留",
    r"不作为每章正文",
    r"本章写作",
]


PUBLIC_BODY_REWRITES = [
    (r"已有可观察的公开资料信号[，,]?", "已经出现可观察变化，"),
    (r"已有方向性公开资料信号[，,]?", "开始出现可观察变化，"),
    (r"后续判断需结合来源范围、样本边界和时间窗口校准。?", ""),
    (r"后续判断需要结合来源范围、样本边界和时间窗口校准。?", ""),
    (r"仍需避免把单一材料外推为确定性结论。?", ""),
    (r"这些信号仍受来源覆盖范围和公开披露充分性的限制，应作为方向性观察进入正文。?", "公开披露通常更容易呈现已经启动的事项，实际进展仍会受到资金、审批、执行成本和使用门槛的共同影响。"),
    (r"后续应继续观察同类主体、同类场景和相同口径信息是否重复出现。?", ""),
    (r"以及后续判断需要继续观察哪些约束条件，而不是被简单处理成孤立材料。?", "并进一步说明这些变化如何影响具体业务、资源安排和执行节奏。"),
    (r"方向性观察进入正文", "早期产业信号"),
    (r"方向性观察", "阶段性判断"),
    (r"后续变化交叉验证", "后续变化"),
    (r"后续是否重复出现", "是否转化为持续行动"),
    (r"现有材料能够覆盖", "公开信息呈现"),
    (r"\u76f8\u5173\u6750\u6599\u6307\u5411", "\u76f8\u5173\u4e8b\u5b9e\u6307\u5411"),
    (r"\u8fd9\u4e9b\u6750\u6599\u5171\u540c\u6307\u5411", "\u8fd9\u7ec4\u4e8b\u5b9e\u652f\u6491\u7684\u5224\u65ad\u662f"),
    (r"\u8fd9\u4e9b\u4fe1\u606f\u5171\u540c\u63cf\u7ed8", "\u8fd9\u7ec4\u4e8b\u5b9e\u652f\u6491\u7684\u5224\u65ad\u662f"),
    (r"公开材料显示", "可核验材料指向"),
    (r"材料显示", "相关材料指向"),
    (r"已披露的关键事实包括[:：]", "关键事实包括："),
    (r"证据陈述", "事实罗列"),
    (r"基于证据", "基于公开信息"),
    (r"证据支持", "公开信息支撑"),
    (r"证据支撑", "公开信息支撑"),
    (r"核心证据", "关键事实"),
    (r"关键证据", "关键事实"),
    (r"关联证据", "关联事实"),
    (r"证据边界", "判断边界"),
    (r"验证清单", "观察指标"),
    (r"验证顺序", "判断顺序"),
    (r"可核验", "可复核"),
    (r"这些事实需要放在主体、时间、范围和来源层级上交叉理解，才能判断它是短期扰动还是可持续趋势。", "这些变化需要结合主体、时间和适用范围判断其持续性。"),
    (r"来源层级上交叉理解", "来源质量和适用范围交叉理解"),
    (r"来源层级", "来源质量"),
    (r"A/B层级来源", "高质量公开来源"),
    (r"A/B层级", "高质量来源"),
]


def _mask_safe_public_terms(text: str) -> str:
    value = str(text or "")
    for term in SAFE_PUBLIC_TERMS:
        value = value.replace(term, "")
    return value


def _publication_patterns(*, strict_only: bool = False) -> List[str]:
    if strict_only:
        return list(STRICT_PUBLICATION_BLOCKERS)
    return [*INTERNAL_GAP_PATTERNS, *STRICT_PUBLICATION_BLOCKERS]


INTERNAL_GAP_REWRITES = [
    (r"该信号需要同时穿过场景、主体和口径三层约束，才能从单点事实变成可复制结论。材料中已经出现的可观察事实是[:：]", "公开材料显示："),
    (r"当前可用事实包括[:：]", "公开材料显示："),
    (r"把反向触发器写入验证清单，并在新增证据改变口径时重新排序(?:章节)?结论。?", "后续应重点观察反向信号，并在口径变化时校准判断。"),
    (r"建议动作[:：]", "策略建议："),
    (r"材料中最有解释力的事实组合是[:：]", "公开材料显示："),
    (r"当前事实组合是[:：]", "公开材料显示："),
    (r"这些事实需要按供应链层级拆开理解[:：]", "可按供应链层级理解："),
    (r"围绕“([^”]+)”，讨论应从", r"围绕“\1”，分析可从"),
    (r"围绕“([^”]+)”，讨论从事实组合开始，再转入成立条件和相反情形。公开材料显示[:：]", r"围绕“\1”，分析先看已经出现的产业信号，再看成立条件和反向情形。公开材料显示："),
    (r"后续跟踪应集中在", "后续重点观察"),
    (r"后续跟踪的重点落在", "后续重点观察"),
    (r"后续跟踪集中在", "后续重点观察"),
    (r"章节结论才适合上升为全篇主线", "这一判断才更适合成为全文主线"),
    (r"章节结论才会进入全篇主线", "这一判断才会进入全文主线"),
    (r"章节结论", "判断"),
    (r"低置信方向判断", "方向性判断"),
    (r"低置信", "方向性"),
    (r"证据不足", "现有公开信息只能支持边界化观察"),
    (r"正文\s*只能\s*写成", "当前更适合表述为"),
    (r"本章\s*只能\s*写成", "当前更适合表述为"),
    (r"本章\s*可\s*写成", "本章判断为"),
    (r"本章\s*应\s*写成", "本章判断为"),
    (r"建议避免", "需要避免"),
    (r"建议在后续版本中补充", "后续重点补充"),
    (r"暂无可核验数据", "尚未看到连续公开数据"),
    (r"暂无足够证据", "尚未看到连续公开数据"),
    (r"不能作为确定性结论", "更适合作为观察项"),
    (r"无法判断", "需要转为观察项处理"),
    (r"无法分析", "需要转为观察项处理"),
    (r"不做判断", "暂按边界条件处理"),
    (r"尚不能形成结论", "更适合作为阶段性观察"),
    (r"需要补证", "需要跟踪后续数据"),
    (r"需补证", "需要跟踪后续数据"),
    (r"建议后续补充调研", "后续重点跟踪"),
    (r"建议补充", "后续重点跟踪"),
    (r"证据缺口", "观察边界"),
    (r"缺少([^，。；\n]*)证据", r"\1仍需连续观察"),
    (r"待验证事项", "后续观察项"),
    (r"needs_corroboration", "directional_signal"),
    (r"权威来源交叉验证", "多来源口径校准"),
    (r"A/B\s*级来源不足", "来源层级仍需用后续公开信息校准"),
    (r"unsupported", "context_only"),
    (r"insufficient evidence", "directional evidence"),
    (r"当前表格证据不足", "当前不生成正文表格"),
    (r"关联证据[:：][^\n。]*[。]?", ""),
    (r"本章可用来源约\d+条[，。]?", ""),
    (r"A/B层级来源约\d+条[，。]?", ""),
    (r"来源层级分布为[^。；\n]*[。；]?", ""),
    (r"本章写作时应", ""),
    (r"当前最直接的支持点是[:：]", "材料显示："),
    (r"不作为每章正文[^。]*[。]?", ""),
    (r"正文只保留", "只保留"),
]


def has_internal_gap_language(text: str) -> bool:
    raw = str(text or "")
    if any(re.search(pattern, raw, re.I) for pattern in STRICT_PUBLICATION_BLOCKERS):
        return True
    value = _mask_safe_public_terms(raw)
    return any(re.search(pattern, value, re.I) for pattern in INTERNAL_GAP_PATTERNS)


# The template composers narrate the *analysis methodology* into the body
# ("这一变化首先影响任务分工和能力配置…", "当主体行动持续、影响路径清楚…").
# That generic "how to judge any change" scaffolding is what makes the report
# read like an internal analysis tool rather than an industry-research deliverable,
# so any sentence carrying these framework markers is dropped from public text.
_ANALYSIS_FRAMEWORK_NARRATION_RE = re.compile(
    r"任务分工[与和]?能力配置"
    r"|系统接口和?责任边界"
    r"|重新安排人员能力"
    r"|原本偏执行的工作"
    r"|影响路径是否清[楚晰]"
    r"|主体行动是否持续"
    r"|约束条件是否可解释"
    r"|可以解释真实工作流程"
    r"|放进连续变化中观察"
    r"|谁承担变化成本"
    r"|哪些环节先受影响"
    r"|场景深度[、，]组织采纳"
    r"|梳理岗位任务"
    r"|可解释的分析材料"
    r"|单点样本转化为"
    r"|这一判断的价值[不]?在于"
    r"|不在于(?:复述|重复)资料"
    # Deep-unit template framing that narrates the analytical model instead of the
    # finding, e.g. "放在“供给约束、需求兑现、价格利润、反向样本”四层关系中观察".
    r"|放在[“”\"'][^。！？]{2,50}[”“\"'](?:[^。！？]{0,8})?关系中(?:观察|判断)"
    r"|需要放在.{0,30}关系中(?:观察|判断)"
)


def _strip_analysis_framework_narration(text: str) -> str:
    value = str(text or "")
    if not _ANALYSIS_FRAMEWORK_NARRATION_RE.search(value):
        return value
    sentences = re.split(r"(?<=[。！？\n])", value)
    kept = [s for s in sentences if not (s.strip() and _ANALYSIS_FRAMEWORK_NARRATION_RE.search(s))]
    rebuilt = "".join(kept).strip()
    # If only citation markers remain (the sentence carrying them was framework
    # narration), drop them so no dangling "[1]" is left behind.
    if rebuilt and re.fullmatch(r"(?:\[\d{1,5}\]|\s)+", rebuilt):
        return ""
    # If stripping would empty real text, leave it untouched rather than blanking.
    return rebuilt or value


def rewrite_internal_gap_language(text: str) -> str:
    value = str(text or "")
    for pattern, replacement in [
        (r"证据不足[，,、；; ]*", ""),
        (r"不能作为确定性结论[，,、；; ]*", ""),
        (r"无法作为确定性结论[，,、；; ]*", ""),
        (r"只能作为方向性判断[，,、；; ]*", "初步显示，"),
        (r"建议补证[，,、；; ]*", ""),
        (r"建议补充[，,、；; ]*", ""),
        (r"建议避免[，,、；; ]*", ""),
        (r"后续版本中补充[，,、；; ]*", ""),
        (r"本章只能写成[：:，,、；; ]*", ""),
        (r"正文只能写成[：:，,、；; ]*", ""),
        (r"本章应写成[：:，,、；; ]*", ""),
        (r"事实锚点显示[：:，,、；; ]*", ""),
        (r"可复核网页线索如下[：:，,、；; ]*", ""),
    ]:
        value = re.sub(pattern, replacement, value, flags=re.I)
    for pattern, replacement in INTERNAL_GAP_REWRITES:
        value = re.sub(pattern, replacement, value, flags=re.I)
    for pattern, replacement in PUBLIC_BODY_REWRITES:
        value = re.sub(pattern, replacement, value, flags=re.I)
    # Strip vague hedging lead-ins ("公开材料提到，/公开材料显示：") but keep the fact
    # they introduce — a real industry report states the fact directly.
    value = re.sub(r"公开材料(?:提到|显示|指出|提及)[，,：:]?\s*", "", value)
    value = _strip_analysis_framework_narration(value)
    # Drop dangling cross-references whose referent was stripped during cleaning,
    # e.g. "（参见）" / "（详见 ）" / empty "（）". Parentheses that still carry a
    # real referent such as "（参见图3）" are left untouched.
    value = re.sub(r"[（(]\s*(?:参见|详见|另见|参阅|见)?\s*[）)]", "", value)
    value = re.sub(r"[ \t]{2,}", " ", value)
    return value.strip()


def _split_source_appendix(markdown: str) -> tuple[str, str]:
    match = re.search(r"(?m)^##\s*(?:数据来源列表|数据来源|来源附录|研究口径与来源|附录|参考来源|参考资料)(?:\s|$|[:：])", str(markdown or ""))
    if not match:
        return str(markdown or ""), ""
    return str(markdown or "")[: match.start()].rstrip(), str(markdown or "")[match.start() :].strip()


PUBLIC_NARRATIVE_BLOCK_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"(?m)^#{1,4}\s*(?:政策摘要|执行风险|监测指标|验证清单|尽调清单)\s*$", "diagnostic_heading"),
    (r"(?m)^#{1,4}\s*(?:事实依据|关键事实与判断依据|商业化证据|指标口径表)\s*$", "generic_section_heading"),
    (r"(?m)^#{1,4}\s*ch_\d{1,3}\s*$", "internal_chapter_id_heading"),
    (r"(?m)^#{1,4}\s*本节[^#\n]{0,20}观察\s*$", "section_observation_heading"),
    (r"(?m)^\s*研究主线\s*[:：]", "research_process_intro"),
    (r"(?m)^\s*[-*]?\s*政策影响\s*[:：]", "policy_impact_prefix"),
    (r"(?m)^\s*应对\s*[:：]", "response_instruction"),
    (r"执行边界风险|假设边界风险", "risk_register_language"),
    (r"该证据|该事实", "evidence_processing_subject"),
    (r"可用事实|正文需要|观察顺序|原文核验|后续观察本章", "writing_process_language"),
    (r"事实锚点|事实起点|后续重点跟踪|可复核材料指向", "analysis_scaffold_language"),
    (r"这些事实来自不同类型来源|来源集中、口径不一致", "analysis_scaffold_language"),
    (r"商业化证据主要集中|其他行业缺乏明确案例|多数证据为20\d{2}[-—至到]\s*20\d{2}年报告", "evidence_repair_signal"),
    (r"来源多为[ABCD]级|来源多为B级或C级|可靠性中等|时效性有限", "evidence_repair_signal"),
    (r"证据主要集中在[^。；;\n]{1,80}少数行业|缺乏明确案例", "evidence_repair_signal"),
    (r"待验证方向|尚不足以支撑强结论", "fallback_claim_language"),
    (r"可核验内容适合|可复核内容适合|局部变化的入口|暂时缺少覆盖的外推|待验证问题处理|相邻来源重复情况", "review_style_bridge_language"),
    (r"这张表显示|后续影响\s*[:：]|使用边界\s*[:：]|表内信号", "diagnostic_table_commentary"),
    (r"需要按连续指标|避免把单点信号直接外推|更适合作为背景条件|结论强度取决", "analysis_scaffold_language"),
    (
        r"仅反映|未覆盖|单点披露|单点信号|尚不足以代表|结论(?:需要|仍需|仍应)保留|"
        r"不能直接推出全行业|不能直接替代更大范围",
        "review_boundary_sentence",
    ),
)

_PUBLIC_REVIEW_BOUNDARY_SENTENCE_RE = re.compile(
    r"仅反映|未覆盖|单点披露|单点信号|尚不足以代表|结论(?:需要|仍需|仍应)保留|"
    r"不能直接推出全行业|不能直接替代更大范围|需要保留的边界|当前材料能够说明.*但"
)
_PUBLIC_SENTENCE_RE = re.compile(r"[^。！？!?]+[。！？!?](?:\s*\[\d{1,5}\])*|[^。！？!?]+$")
_PUBLIC_CITATION_RE = re.compile(r"\[\d{1,5}\]")


def _drop_public_review_boundary_sentences(line: str) -> str:
    raw = str(line or "")
    if not _PUBLIC_REVIEW_BOUNDARY_SENTENCE_RE.search(raw):
        return raw
    citations = _PUBLIC_CITATION_RE.findall(raw)
    kept: List[str] = []
    removed = False
    for match in _PUBLIC_SENTENCE_RE.finditer(raw):
        sentence = match.group(0)
        if not sentence:
            continue
        if _PUBLIC_REVIEW_BOUNDARY_SENTENCE_RE.search(sentence):
            removed = True
            continue
        kept.append(sentence)
    cleaned = "".join(kept).strip()
    if removed and cleaned and citations and not _PUBLIC_CITATION_RE.search(cleaned):
        refs: List[str] = []
        seen: set[str] = set()
        for ref in citations:
            if ref in seen:
                continue
            refs.append(ref)
            seen.add(ref)
        cleaned = f"{cleaned}{''.join(refs[:4])}"
    return cleaned

_PUBLIC_NARRATIVE_DROP_HEADING_RE = re.compile(
    r"^#{1,4}\s*(?:政策摘要|执行风险|监测指标|验证清单|尽调清单|关键事实与判断依据|ch_\d{1,3})\s*$",
    re.I,
)
_PUBLIC_NARRATIVE_RETITLE_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^(#{1,4})\s*本节技术观察\s*$"), r"\1 技术落地约束"),
    (re.compile(r"^(#{1,4})\s*本节指标观察\s*$"), r"\1 指标信号是否一致"),
    (re.compile(r"^(#{1,4})\s*本节市场观察\s*$"), r"\1 市场信号是否成立"),
    (re.compile(r"^(#{1,4})\s*事实依据\s*$"), r"\1 产业信号"),
    (re.compile(r"^(#{1,4})\s*商业化证据\s*$"), r"\1 进展与约束"),
)


def public_narrative_leak_audit(markdown: str) -> dict:
    body, _appendix = _split_source_appendix(str(markdown or ""))
    blockers: List[dict] = []
    reason_counts: Dict[str, int] = {}
    for line_no, line in enumerate(body.splitlines(), start=1):
        raw = str(line or "")
        if not raw.strip():
            continue
        for pattern, reason in PUBLIC_NARRATIVE_BLOCK_PATTERNS:
            if re.search(pattern, raw):
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
                blockers.append({"line": line_no, "reason": reason, "text": raw.strip()[:240]})
                break
    return {
        "blocker_count": len(blockers),
        "reason_counts": reason_counts,
        "examples": blockers[:10],
    }


def _line_without_public_narrative_leak(line: str) -> str:
    raw = str(line or "")
    if _PUBLIC_NARRATIVE_DROP_HEADING_RE.match(raw.strip()):
        return ""
    if re.match(r"^\s*研究主线\s*[:：]", raw):
        return ""
    if re.match(r"^\s*[-*]?\s*政策影响\s*[:：]", raw):
        return ""
    if re.match(r"^\s*应对\s*[:：]", raw):
        return ""
    if re.search(r"执行边界风险|假设边界风险", raw):
        return ""
    if re.search(r"正文需要|观察顺序|原文核验|后续观察本章|可用事实主要包括", raw):
        return ""
    if re.search(
        r"事实锚点|事实起点|后续重点跟踪|这些事实来自不同类型来源|来源集中、口径不一致|待验证方向|尚不足以支撑强结论|可核验内容适合|可复核内容适合|局部变化的入口|暂时缺少覆盖的外推|待验证问题处理|相邻来源重复情况",
        raw,
    ):
        return ""
    if re.search(
        r"商业化证据主要集中|其他行业缺乏明确案例|多数证据为20\d{2}[-—至到]\s*20\d{2}年报告|来源多为[ABCD]级|来源多为B级或C级|可靠性中等|时效性有限|证据主要集中在[^。；;\n]{1,80}少数行业|缺乏明确案例",
        raw,
    ):
        return ""
    if re.search(r"这张表显示|后续影响\s*[:：]|使用边界\s*[:：]|表内信号", raw):
        return ""
    if re.search(r"需要按连续指标|避免把单点信号直接外推|更适合作为背景条件|结论强度取决", raw):
        return ""
    raw = _drop_public_review_boundary_sentences(raw)
    if not raw.strip():
        return ""
    raw = re.sub(r"可复核材料指向\s*[:：]\s*", "公开材料显示，", raw)
    for pattern, replacement in _PUBLIC_NARRATIVE_RETITLE_RULES:
        raw = pattern.sub(replacement, raw)
    raw = re.sub(r"该证据来自([^，。；;\n]{1,80})[，,]\s*披露", r"\1披露", raw)
    raw = re.sub(r"该证据来自([^，。；;\n]{1,80})[，,]\s*显示", r"\1显示", raw)
    raw = re.sub(r"该证据来自([^，。；;\n]{1,80})[，,]\s*", r"\1显示，", raw)
    raw = re.sub(r"该证据仅反映", "这一信息仅反映", raw)
    raw = re.sub(r"该证据", "这一信息", raw)
    raw = re.sub(r"该事实可用于", "这一信息可用于", raw)
    raw = re.sub(r"该事实", "这一信息", raw)
    return raw


def apply_public_narrative_gate(markdown: str) -> tuple[str, dict]:
    body, appendix = _split_source_appendix(str(markdown or ""))
    before = public_narrative_leak_audit(body)
    kept: List[str] = []
    for block in re.split(r"\n(?=##\s+)", body):
        first_line = next((line for line in block.splitlines() if line.strip()), "")
        if _PUBLIC_NARRATIVE_DROP_HEADING_RE.match(first_line.strip()):
            continue
        rewritten_lines = [_line_without_public_narrative_leak(line) for line in block.splitlines()]
        rewritten = "\n".join(line for line in rewritten_lines if str(line or "").strip())
        if rewritten.strip():
            kept.append(rewritten)
    cleaned = "\n\n".join(kept).strip()
    after = public_narrative_leak_audit(cleaned)
    diagnostics = {
        "public_narrative_leak_input_count": before.get("blocker_count", 0),
        "public_narrative_leak_remaining_count": after.get("blocker_count", 0),
        "public_narrative_leak_removed_count": max(
            0, int(before.get("blocker_count", 0)) - int(after.get("blocker_count", 0))
        ),
        "public_narrative_leak_reason_counts": before.get("reason_counts", {}),
        "public_narrative_leak_examples": before.get("examples", []),
        "public_narrative_leak_remaining_examples": after.get("examples", []),
    }
    if appendix:
        return ((cleaned + "\n\n" + appendix).strip() if cleaned else appendix), diagnostics
    return cleaned, diagnostics


def find_publication_blockers(markdown: str) -> List[dict]:
    body, appendix = _split_source_appendix(str(markdown or ""))
    blockers: List[dict] = []
    for line_no, line in enumerate(body.splitlines(), start=1):
        for pattern in STRICT_PUBLICATION_BLOCKERS:
            if re.search(pattern, line, re.I):
                blockers.append({"line": line_no, "pattern": pattern, "text": line.strip()[:240]})
                break
        else:
            masked = _mask_safe_public_terms(line)
            for pattern in INTERNAL_GAP_PATTERNS:
                if re.search(pattern, masked, re.I):
                    blockers.append({"line": line_no, "pattern": pattern, "text": line.strip()[:240]})
                    break
    body_line_count = len(body.splitlines())
    for offset, line in enumerate(appendix.splitlines(), start=1):
        for pattern in _publication_patterns(strict_only=True):
            if re.search(pattern, line, re.I):
                blockers.append({"line": body_line_count + offset, "pattern": pattern, "text": line.strip()[:240]})
                break
    return blockers


def _line_has_publication_blocker(line: str, *, strict_only: bool = False) -> bool:
    raw = str(line or "")
    if any(re.search(pattern, raw, re.I) for pattern in STRICT_PUBLICATION_BLOCKERS):
        return True
    if strict_only:
        return False
    masked = _mask_safe_public_terms(raw)
    return any(re.search(pattern, masked, re.I) for pattern in INTERNAL_GAP_PATTERNS)


def _drop_publication_blocker_lines(markdown: str, *, strict_only: bool = False) -> str:
    result: List[str] = []
    for line in str(markdown or "").splitlines():
        candidate = rewrite_internal_gap_language(line) if not strict_only else str(line or "")
        if _line_has_publication_blocker(candidate, strict_only=strict_only):
            continue
        result.append(candidate)
    return "\n".join(result)


def remove_empty_headings(markdown: str) -> str:
    lines = str(markdown or "").splitlines()
    result: List[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        match = re.match(r"^(#{2,4})\s+", line)
        if match:
            heading_level = len(match.group(1))
            j = i + 1
            content_lines = []
            while j < len(lines):
                next_match = re.match(r"^(#{1,4})\s+", lines[j])
                if next_match and len(next_match.group(1)) <= heading_level:
                    break
                if lines[j].strip():
                    content_lines.append(lines[j].strip())
                j += 1

            if not content_lines:
                i = j
                continue

        result.append(line)
        i += 1

    return "\n".join(result)


def remove_empty_markdown_tables(markdown: str) -> str:
    lines = str(markdown or "").splitlines()
    result: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        next_line = lines[i + 1] if i + 1 < len(lines) else ""
        if (
            line.strip().startswith("|")
            and line.strip().endswith("|")
            and re.match(r"^\|\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|\s*$", next_line.strip())
        ):
            j = i + 2
            row_count = 0
            while j < len(lines) and lines[j].strip().startswith("|") and lines[j].strip().endswith("|"):
                row_count += 1
                j += 1
            if row_count == 0:
                while result and not result[-1].strip():
                    result.pop()
                if result and re.match(r"^\*\*[^*\n]+\*\*\s*$", result[-1].strip()):
                    result.pop()
                i = j
                continue
        result.append(line)
        i += 1
    return "\n".join(result)


TRADITIONAL_CHAR_MAP = str.maketrans(
    {
        "發": "发",
        "佈": "布",
        "體": "体",
        "團": "团",
        "業": "业",
        "務": "务",
        "軟": "软",
        "證": "证",
        "據": "据",
        "場": "场",
        "應": "应",
        "與": "与",
        "實": "实",
        "驗": "验",
        "轉": "转",
        "進": "进",
        "階": "阶",
        "價": "价",
        "為": "为",
        "單": "单",
        "個": "个",
        "對": "对",
        "雲": "云",
        "數": "数",
        "電": "电",
        "費": "费",
        "戶": "户",
        "產": "产",
        "鏈": "链",
        "⼼": "心",
        "⼤": "大",
        "⽤": "用",
    }
)


def normalize_public_text_artifacts(markdown: str) -> str:
    text = str(markdown or "").translate(TRADITIONAL_CHAR_MAP)
    replacements = {
        "中心心": "中心",
        "管理理": "管理",
        "大大模型": "大模型",
        "应用用": "应用",
        "业务务": "业务",
        "数据据": "数据",
        "场场景": "场景",
        "实验验": "实验",
    }
    for before, after in replacements.items():
        text = text.replace(before, after)
    text = re.sub(r"[（(]\s*[）)]", "", text)
    text = re.sub(r"\.{1,}\s*。", "。", text)
    text = re.sub(r"…+\s*。", "。", text)
    text = re.sub(r"。\s*\.{1,}", "。", text)
    return text


PUBLIC_INTERNAL_TERM_REWRITES = (
    ("market metric", "市场指标"),
    ("risk boundary", "风险边界"),
    ("deployment depth", "使用深度"),
    ("competitive position", "竞争位置"),
    ("technical maturity", "技术成熟度"),
    ("commercialization", "商业化"),
    ("competitive signal", "竞争信号"),
    ("\u4e3a\u6b64\u843d\u5730\u5230\u54ea\u4e00\u6b65", "\u843d\u5730\u8fdb\u5c55\u5230\u54ea\u4e00\u6b65"),
)


PUBLIC_INTERNAL_TERM_LINE_RE = re.compile(
    r"(?:\bblock_affinity\b|\banalysis_variable\b|\bevidence_cards?\b|\bEV-\w+)",
    re.I,
)


def rewrite_internal_public_terms(markdown: str) -> str:
    text = str(markdown or "")
    for before, after in PUBLIC_INTERNAL_TERM_REWRITES:
        text = re.sub(re.escape(before), after, text, flags=re.I)
    kept: List[str] = []
    for line in text.splitlines():
        if PUBLIC_INTERNAL_TERM_LINE_RE.search(line):
            continue
        kept.append(line)
    return "\n".join(kept)


def public_text_artifact_counts(markdown: str) -> dict:
    text = str(markdown or "")
    return {
        "ocr_artifact_normalized_count": len(
            re.findall(r"中⼼心|中心心|管理理|⼤大|大大模型|应⽤用|应用用", text)
        ),
        "traditional_chinese_normalized_count": len(
            re.findall(r"[發佈體團業務軟證據場應與實驗轉進階價為單個對雲數電費戶產鏈]", text)
        ),
        "empty_parens_removed_count": len(re.findall(r"[（(]\s*[）)]", text)),
        "truncated_punctuation_cleaned_count": len(re.findall(r"\.{1,}\s*。|…+\s*。|。\s*\.{1,}", text)),
    }


_DIAGNOSTIC_TABLE_LANGUAGE_RE = re.compile(
    r"后续影响|该指标须|须同时披露|进入正文判断|缺口数据只作为|不会凭空补齐"
    r"|指标口径表|市场指标与口径表|政策影响与风险登记表",
    re.I,
)
# Heading variants that introduce a diagnostic table: markdown headings (## / ###)
# and bold-text "pseudo-headings" (**…**) that the renderer also emits.
_DIAGNOSTIC_TABLE_HEADING_RE = re.compile(
    r"^\s*(?:#{1,4}\s*|\*\*\s*)"
    r"(?:指标口径表|指标口径与可比性|市场指标与口径表|政策影响与风险登记表"
    r"|核心变量对照|关键指标对照)"
)
# Bold-text label that always belongs to a diagnostic table even with surrounding
# topic prefix (e.g. ``**AI Agent…核心变量对照**``). Looser than the heading regex.
_DIAGNOSTIC_BOLD_LABEL_RE = re.compile(
    r"^\s*\*\*[^*\n]*(?:核心变量对照|关键指标对照|指标口径表|市场指标与口径表|政策影响与风险登记表)[^*\n]*\*\*\s*$"
)
_MARKDOWN_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
# Orphan paragraphs the renderer emits right after a diagnostic table — once we
# drop the table, these become dangling references with no citation support.
# ``这张表显示`` is followed by ``，`` (renderer template emits ``这张表显示，<takeaway>``),
# while ``判断含义/后续影响/使用边界`` are followed by ``:/：`` (label form), so the
# punctuation class must allow both.
_TABLE_ORPHAN_PARAGRAPH_RE = re.compile(
    r"^\s*(?:这张表显示|判断含义|后续影响|使用边界)\s*[:：，,]"
)
# Standalone placeholder lines that should always be removed regardless of
# surrounding context.
_STANDALONE_PLACEHOLDER_RE = re.compile(
    r"该指标须同时披露|进入正文判断|不会凭空补齐|该信号只有与反例和高等级来源同向时"
)
# Orphan ``## 关键数据`` bullets that carry only a metric name + period (no
# value/unit), e.g. ``- CAGR；2028年``. Without a value they are pure noise.
_ORPHAN_KEY_DATA_BULLET_RE = re.compile(
    r"^\s*[-*]\s*[A-Za-z一-鿿]+\s*[；;,，]\s*\d{4}\s*年?\s*$"
)

_DIRTY_PUBLIC_SPAN_PATTERNS = [
    # Upstream metric extraction can combine an ordinal/table label, a bare
    # year/id, and a value into prose such as
    # "（二）AI 物业经理智能体建设单位的成本在2000为70%...".
    # Remove only the bad span so useful case facts in the same paragraph stay.
    re.compile(
        r"(?:（[一二三四五六七八九十]+）)?"
        r"[^。；;\n]{0,90}"
        r"(?:成本|价格|收入|市场规模|数据指标|定性事实|关键指标)"
        r"[^。；;\n]{0,30}在\d{3,4}为[^。；;\n]{1,40}"
        r"，这一指标用于判断市场空间和兑现节奏[。；;]?\s*"
    ),
    # Source-page headings copied into the body, e.g.
    # "转化愿景为现实：AI在采购中的应用场景 引言".
    re.compile(r"[^。；;\n]{0,80}[:：][^。；;\n]{0,120}\s+引言\s*"),
]

_DIRTY_PUBLIC_SPAN_REWRITES = [
    (
        re.compile(r"这个反向样本提示([^。；;\n]{1,40})仍可能改变结论强度，需要把商业化判断限制在已验证场景内[。；;]?"),
        r"这一风险会降低\1相关判断的确定性。",
    ),
    (re.compile(r"(市场规模|收入|订单|客户数量|客户规模)为达"), r"\1达"),
]

_NAMED_SOURCE_CLAIM_RULES = [
    {
        "claim_marker": re.compile(r"(?:Gartner|高德纳|炒作周期)", re.I),
        "source_marker": re.compile(r"(?:Gartner|高德纳|gartner\.com)", re.I),
        "sentence": re.compile(r"[^。；;\n]*(?:Gartner|高德纳|炒作周期)[^。；;\n]*(?:[。；;]|$)", re.I),
        "heading": re.compile(r"(?m)^\s*#{1,4}\s*[^。\n]*(?:Gartner|高德纳|炒作周期)[^\n]*$"),
    },
]

_PUBLIC_TEMPLATE_SCAFFOLD_PATTERNS = (
    re.compile(
        r"\u5bf9\u201c[^\u201d]{1,180}\u201d\u8fd9\u4e00\u5224\u65ad\u800c\u8a00"
        r"\uff0c?\u5173\u952e\u4e0d\u53ea\u662f\u4e8b\u5b9e\u662f\u5426\u51fa\u73b0"
        r"[^。！？；;\n]{0,240}[。！？；;]?"
    ),
    re.compile(
        r"\u5982\u679c\u628a\u5b83\u653e\u5728\u62a5\u544a\u4e3b\u7ebf\u4e2d"
        r"[^。！？；;\n]{0,240}[。！？；;]?"
    ),
    re.compile(
        r"\u8fd9\u79cd\u5904\u7406\u65b9\u5f0f\u53ef\u4ee5\u8ba9\u8bfb\u8005"
        r"[^。！？；;\n]{0,260}[。！？；;]?"
    ),
    re.compile(
        r"\u56e0\u6b64\uff0c?\u8fd9\u4e00\u6bb5\u66f4\u9002\u5408\u4f5c\u4e3a"
        r"\u6709\u8fb9\u754c\u7684\u5206\u6790\u4fe1\u53f7[^。！？；;\n]{0,260}[。！？；;]?"
    ),
    re.compile(
        r"\u4f18\u5148\u9a8c\u8bc1\u5173\u952e\u6765\u6e90\u3001"
        r"\u6307\u6807\u53e3\u5f84\u548c\u53cd\u5411\u6837\u672c"
        r"[^。！？；;\n]{0,160}[。！？；;]?"
    ),
)

_UNCITED_CORE_VIEW_HARD_FACT_RE = re.compile(
    r"\d|%|\uff05|"
    r"\u4ebf\u5143|\u4e07\u4ebf|\u4e07\u5143|\u5e02\u573a\u89c4\u6a21|"
    r"\u9884\u8ba1|\u9884\u6d4b|\u8fbe|\u8d85\u8fc7|\u7a81\u7834|"
    r"\u589e\u957f|\u589e\u901f|\u51fa\u8d27|\u8ba2\u5355|\u5ba2\u6237|"
    r"\u653f\u7b56|\u53d1\u5e03|\u5370\u53d1|\u51fa\u53f0|"
    r"\u4e0a\u5e02\u516c\u53f8|\u8d22\u62a5|\u516c\u544a|"
    r"\u539f\u6cb9|\u80fd\u6e90|IMF|Gartner|IDC|Statista",
    re.I,
)


def _remove_dirty_public_spans(text: str) -> str:
    cleaned = str(text or "")
    for pattern in _DIRTY_PUBLIC_SPAN_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    for pattern, replacement in _DIRTY_PUBLIC_SPAN_REWRITES:
        cleaned = pattern.sub(replacement, cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([。；;，,])", r"\1", cleaned)
    return cleaned


def _remove_public_template_scaffold(text: str) -> str:
    cleaned = str(text or "")
    for pattern in _PUBLIC_TEMPLATE_SCAFFOLD_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([。！？；;])", r"\1", cleaned)
    return cleaned


def _clean_truncated_public_headings(text: str) -> str:
    result: List[str] = []
    for line in str(text or "").splitlines():
        if re.match(r"^\s*#{1,4}\s+", line):
            line = re.sub(r"\s*(?:\.\s*){3,}.*$", "", line).rstrip()
            line = re.sub(r"\s*\(\d{1,4}\)\s*$", "", line).rstrip()
            line = re.sub(r"\s*\uff08\d{1,4}\uff09\s*$", "", line).rstrip()
        result.append(line)
    return "\n".join(result)


def _drop_uncited_core_view_bullets(text: str) -> str:
    result: List[str] = []
    in_core = False
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if re.match(r"^##\s+", stripped):
            in_core = bool(
                re.search(
                    r"\u6838\u5fc3\u89c2\u70b9|\u4e3b\u8981\u7ed3\u8bba|key\s+judgments?|executive\s+summary",
                    stripped,
                    re.I,
                )
            )
        elif re.match(r"^#{1,4}\s+", stripped):
            in_core = False
        if (
            in_core
            and re.match(r"^[-*]\s+", stripped)
            and not re.search(r"\[\d{1,5}\]", stripped)
            and _UNCITED_CORE_VIEW_HARD_FACT_RE.search(stripped)
        ):
            continue
        result.append(line)
    return "\n".join(result)


def _remove_unbacked_named_source_claims(body: str, appendix: str) -> str:
    cleaned = str(body or "")
    source_text = str(appendix or "")
    for rule in _NAMED_SOURCE_CLAIM_RULES:
        if not rule["claim_marker"].search(cleaned):
            continue
        if rule["source_marker"].search(source_text):
            continue
        heading_pattern = rule.get("heading")
        if heading_pattern is not None:
            cleaned = heading_pattern.sub("", cleaned)
        cleaned = rule["sentence"].sub("", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def _drop_trailing_blank(out: List[str]) -> None:
    while out and not out[-1].strip():
        out.pop()


def _strip_diagnostic_tables(text: str) -> str:
    """Remove internal diagnostic tables / placeholder lines that leak into the public body.

    Drops markdown tables whose header or cells carry diagnostic-only language
    (e.g. a ``后续影响`` column or a ``该指标须同时披露…`` placeholder) together with
    the bold/markdown heading that introduced them and any orphan
    ``这张表显示…/后续影响：…/使用边界：…`` paragraphs that immediately follow.
    Standalone placeholder lines are also removed. Legitimate data tables (no
    diagnostic language) and well-formed ``后续影响：…[n]`` table-commentary
    paragraphs (which carry their own citations) are left alone.
    """
    lines = str(text or "").split("\n")
    out: List[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if _MARKDOWN_TABLE_ROW_RE.match(line):
            end = index
            while end < len(lines) and _MARKDOWN_TABLE_ROW_RE.match(lines[end]):
                end += 1
            table_block = "\n".join(lines[index:end])
            if _DIAGNOSTIC_TABLE_LANGUAGE_RE.search(table_block):
                # Walk back past blank lines + bold/heading label that introduced
                # this table so the public output does not retain dangling labels.
                while out and not out[-1].strip():
                    out.pop()
                while out and (
                    _DIAGNOSTIC_TABLE_HEADING_RE.match(out[-1])
                    or _DIAGNOSTIC_BOLD_LABEL_RE.match(out[-1])
                ):
                    out.pop()
                    while out and not out[-1].strip():
                        out.pop()
                # Walk forward past blanks + orphan ``这张表显示…/后续影响：/使用边界：``
                # paragraphs until we reach a real heading or non-orphan content.
                # An orphan with explicit ``[n]`` citation support is kept — those
                # come from the renderer template and are real table commentary
                # that we want to preserve. Pure orphans (no citation) are dropped.
                cursor = end
                while cursor < len(lines):
                    next_line = lines[cursor]
                    if not next_line.strip():
                        cursor += 1
                        continue
                    if _TABLE_ORPHAN_PARAGRAPH_RE.match(next_line) and not re.search(r"\[\d{1,3}\]", next_line):
                        cursor += 1
                        continue
                    break
                index = cursor
                continue
            out.extend(lines[index:end])
            index = end
            continue
        if _STANDALONE_PLACEHOLDER_RE.search(line):
            index += 1
            continue
        if _ORPHAN_KEY_DATA_BULLET_RE.match(line):
            index += 1
            continue
        # Standalone diagnostic headings (no table follows) — drop the heading
        # and any blank padding before/after.
        if (
            _DIAGNOSTIC_TABLE_HEADING_RE.match(line)
            or _DIAGNOSTIC_BOLD_LABEL_RE.match(line)
        ):
            # Peek ahead: only drop if there is NO subsequent valid data table
            # within the next 4 non-blank lines (purely a stranded label).
            lookahead = lines[index + 1 : index + 8]
            has_table_soon = any(_MARKDOWN_TABLE_ROW_RE.match(item) for item in lookahead)
            if not has_table_soon:
                _drop_trailing_blank(out)
                index += 1
                continue
        out.append(line)
        index += 1
    return "\n".join(out)


def _public_sanitizer_mutation_mode() -> str:
    value = str(os.environ.get("REPORT_PUBLIC_SANITIZER_MUTATION_MODE") or "diagnostic_only").strip().lower()
    if value in {"enforce", "strict", "repair_publication", "mutate", "clean"}:
        return "enforce"
    return "diagnostic_only"


def _repair_count_metric_labeled_as_cost(text: str) -> str:
    """Fix extraction/rendering artifacts where a count is mislabeled as cost.

    Keep real cost claims intact. The rewrite only fires when the value uses a
    count-like unit ("家", "企业", "单位") that cannot be a monetary cost.
    """

    repaired = str(text or "")
    repaired = re.sub(
        r"成本方面，\s*(相关企业数量已(?:超|超过)\s*\d+(?:\.\d+)?\s*家)",
        r"参与主体方面，\1",
        repaired,
    )
    repaired = re.sub(
        r"成本维度上，\s*(参与企业(?:已)?(?:超|超过)\s*\d+(?:\.\d+)?\s*家)",
        r"参与主体维度上，\1",
        repaired,
    )
    repaired = re.sub(
        r"成本(?:已)?(超|超过)\s*(\d+(?:\.\d+)?\s*家)",
        r"参与企业\1\2",
        repaired,
    )
    repaired = re.sub(
        r"[（(]\s*成本\s*[:：]\s*((?:超|超过)?\s*\d+(?:\.\d+)?\s*家)\s*[）)]",
        "",
        repaired,
    )
    repaired = repaired.replace("高参与成本可能", "合规成本或参与门槛可能")
    return repaired


def sanitize_public_markdown(markdown: str, *, mode: str | None = None) -> str:
    effective_mode = str(mode or _public_sanitizer_mutation_mode()).strip().lower()
    if effective_mode not in {"enforce", "strict", "repair_publication", "mutate", "clean"}:
        return str(markdown or "")
    body, appendix = _split_source_appendix(str(markdown or ""))
    normalized_body = normalize_public_text_artifacts(body)
    normalized_appendix = normalize_public_text_artifacts(appendix)
    text = rewrite_internal_public_terms(
        _clean_truncated_public_headings(
            _drop_uncited_core_view_bullets(
                _remove_public_template_scaffold(
                    _remove_dirty_public_spans(
                        _remove_unbacked_named_source_claims(normalized_body, normalized_appendix)
                    )
                )
            )
        )
    )
    text = _strip_diagnostic_tables(text)
    schema_like_bullet_re = re.compile(r"(?m)^\s*[-*]\s*[^。；;\n]{1,16}[；;][^。；;\n]{0,16}[；;][^。；;\n]{0,50}\s*$")
    blocks = re.split(r"\n(?=#{1,4}\s+)", text)
    kept: List[str] = []

    for block in blocks:
        safe_block = _drop_publication_blocker_lines(block)
        safe_block = schema_like_bullet_re.sub("", safe_block)
        rewritten = rewrite_internal_gap_language(safe_block)
        if rewritten.strip():
            kept.append(rewritten)

    cleaned = "\n".join(kept)
    for _ in range(3):
        before = cleaned
        cleaned = _repair_count_metric_labeled_as_cost(cleaned)
        cleaned = _drop_publication_blocker_lines(cleaned)
        cleaned = schema_like_bullet_re.sub("", cleaned)
        cleaned = rewrite_internal_gap_language(cleaned)
        cleaned = remove_empty_markdown_tables(cleaned)
        cleaned = remove_empty_headings(cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        if not find_publication_blockers(cleaned) or cleaned == before:
            break
    if find_publication_blockers(cleaned):
        cleaned = _drop_publication_blocker_lines(cleaned)
        cleaned = remove_empty_markdown_tables(cleaned)
        cleaned = remove_empty_headings(cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    cleaned, _public_narrative_diag = apply_public_narrative_gate(cleaned)
    for _ in range(3):
        before = cleaned
        cleaned = remove_empty_headings(cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        if cleaned == before:
            break
    appendix = rewrite_internal_public_terms(
        remove_empty_headings(_drop_publication_blocker_lines(normalized_appendix, strict_only=True)).strip()
    )
    if appendix:
        return (cleaned + "\n\n" + appendix).strip() if cleaned else appendix
    return cleaned
