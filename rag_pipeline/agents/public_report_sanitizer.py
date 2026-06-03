from __future__ import annotations

import re
from typing import Dict, List


PUBLIC_EV_ID_PATTERN = r"(?<![A-Za-z0-9_])EV-\d+(?:-[A-Za-z0-9]+)?"


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
    (r"待验证方向|尚不足以支撑强结论", "fallback_claim_language"),
    (r"这张表显示|后续影响\s*[:：]|使用边界\s*[:：]|表内信号", "diagnostic_table_commentary"),
    (r"需要按连续指标|避免把单点信号直接外推|更适合作为背景条件|结论强度取决", "analysis_scaffold_language"),
)

_PUBLIC_NARRATIVE_DROP_HEADING_RE = re.compile(
    r"^#{1,4}\s*(?:政策摘要|执行风险|监测指标|验证清单|尽调清单|关键事实与判断依据|ch_\d{1,3})\s*$",
    re.I,
)
_PUBLIC_NARRATIVE_RETITLE_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^(#{1,4})\s*本节技术观察\s*$"), r"\1 技术落地约束"),
    (re.compile(r"^(#{1,4})\s*本节指标观察\s*$"), r"\1 指标信号是否一致"),
    (re.compile(r"^(#{1,4})\s*本节市场观察\s*$"), r"\1 市场信号是否成立"),
    (re.compile(r"^(#{1,4})\s*事实依据\s*$"), r"\1 产业信号"),
    (re.compile(r"^(#{1,4})\s*商业化证据\s*$"), r"\1 商业化进展"),
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
        r"事实锚点|事实起点|后续重点跟踪|这些事实来自不同类型来源|来源集中、口径不一致|待验证方向|尚不足以支撑强结论",
        raw,
    ):
        return ""
    if re.search(r"这张表显示|后续影响\s*[:：]|使用边界\s*[:：]|表内信号", raw):
        return ""
    if re.search(r"需要按连续指标|避免把单点信号直接外推|更适合作为背景条件|结论强度取决", raw):
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
    ("deployment depth", "部署深度"),
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


def sanitize_public_markdown(markdown: str) -> str:
    body, appendix = _split_source_appendix(str(markdown or ""))
    text = rewrite_internal_public_terms(normalize_public_text_artifacts(body))
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
    appendix = rewrite_internal_public_terms(normalize_public_text_artifacts(
        remove_empty_headings(_drop_publication_blocker_lines(appendix, strict_only=True)).strip()
    ))
    if appendix:
        return (cleaned + "\n\n" + appendix).strip() if cleaned else appendix
    return cleaned
