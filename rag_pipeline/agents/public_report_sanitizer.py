from __future__ import annotations

import re
from typing import List


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
    (r"公开材料显示", "可核验材料指向"),
    (r"材料显示", "相关材料指向"),
    (r"已披露的关键事实包括[:：]", "关键事实包括："),
    (r"证据链", "事实链"),
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
    for pattern, replacement in INTERNAL_GAP_REWRITES:
        value = re.sub(pattern, replacement, value, flags=re.I)
    for pattern, replacement in PUBLIC_BODY_REWRITES:
        value = re.sub(pattern, replacement, value, flags=re.I)
    value = re.sub(r"[ \t]{2,}", " ", value)
    return value.strip()


def _split_source_appendix(markdown: str) -> tuple[str, str]:
    match = re.search(r"(?m)^##\s*(?:数据来源|研究口径与来源|附录|参考来源)\b", str(markdown or ""))
    if not match:
        return str(markdown or ""), ""
    return str(markdown or "")[: match.start()].rstrip(), str(markdown or "")[match.start() :].strip()


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
        if _line_has_publication_blocker(line, strict_only=strict_only):
            continue
        result.append(line)
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


def sanitize_public_markdown(markdown: str) -> str:
    body, appendix = _split_source_appendix(str(markdown or ""))
    text = body
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
    appendix = remove_empty_headings(_drop_publication_blocker_lines(appendix, strict_only=True)).strip()
    if appendix:
        return (cleaned + "\n\n" + appendix).strip() if cleaned else appendix
    return cleaned
