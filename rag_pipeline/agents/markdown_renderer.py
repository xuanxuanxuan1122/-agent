from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterable, List, Sequence

from .public_report_sanitizer import has_internal_gap_language, rewrite_internal_gap_language
from .report_contracts import normalize_evidence_refs
from .summary_quality import sanitize_summary_judgments


INTERNAL_LAYOUT_PHRASES = [
    "章节判断",
    "关键事实速览",
    "证据深读",
    "原文事实",
    "本章核心判断",
    "本章结论",
    "本章小结",
    "图表解读",
    "报告使用方式",
    "进入综合决策章的变量",
    "全球口径",
    "中国口径",
    "增速口径",
    "可引用事实",
    "机制与边界",
    "反证边界",
    "核心判断",
    "关键判断",
    "证据依据",
    "传导链条",
    "判断边界",
    "决策含义",
    "本章综合分析",
    "机制拆解与变量联动",
    "反证、边界与结论失效条件",
    "决策含义与后续观察优先级",
    "关联证据",
    "章节关系与参考分析",
    "\u4e0e\u672c\u7ae0\u5224\u65ad\u76f4\u63a5\u76f8\u5173",
    "\u540e\u7eed\u5206\u6790\u9700\u8981",
    "\u53ea\u6709\u4e3b\u4f53\u3001\u8303\u56f4\u548c\u671f\u95f4\u4e00\u81f4",
    "\u5f71\u54cd\u7684\u662f\u5546\u4e1a\u5316\u6df1\u5ea6",
    "\u5f71\u54cd\u7684\u662f\u5e02\u573a\u7a7a\u95f4\u5224\u65ad",
    "\u4e0d\u80fd\u5916\u63a8\u4e3a\u666e\u904d\u4ed8\u8d39\u80fd\u529b",
]

INTERNAL_SECTION_TITLE_PATTERNS = [
    r"机制拆解",
    r"变量联动",
    r"反证",
    r"结论失效",
    r"决策含义",
    r"后续观察",
    r"全球口径",
    r"中国口径",
    r"增速口径",
    r"可引用事实",
    r"机制与边界",
    r"反证边界",
    r"核心论证",
    r"形成可验证判断",
    r"本章综合分析",
]

INTERNAL_BLOCK_LABELS = {
    "关键判断",
    "证据依据",
    "传导链条",
    "边界",
    "含义",
    "观察点",
    "关键证据",
    "判断边界",
    "核心判断",
    "主要结论",
    "资料支撑",
    "变量传导",
    "适用边界",
    "后续动作",
    "后续影响",
    "可引用事实",
    "机制与边界",
    "进入综合决策章的变量",
    "全球口径",
    "中国口径",
    "增速口径",
}

PUBLIC_TERM_REPLACEMENTS = {
    "本章核心判断": "主要结论",
    "核心判断": "主要结论",
    "关键判断": "主要结论",
    "证据依据": "资料支撑",
    "传导链条": "影响路径",
    "判断边界": "适用边界",
    "决策含义": "策略影响",
    "行动含义": "后续动作",
    "判断含义": "后续影响",
    "本章综合分析": "",
    "机制拆解与变量联动": "影响路径与约束关系",
    "反证、边界与结论失效条件": "反向条件与结论弹性",
    "决策含义与后续观察优先级": "策略影响与观察优先级",
}

PUBLIC_PROCESS_REWRITES = [
    (r"该信号需要同时穿过场景、主体和口径三层约束，才能从单点事实变成可复制结论。材料中已经出现的可观察事实是[:：]", "事实依据："),
    (r"当前可用事实包括[:：]", "事实依据："),
    (r"把反向触发器写入验证清单，并在新增证据改变口径时重新排序(?:章节)?结论。?", "后续应重点观察反向信号，并在口径变化时校准判断。"),
    (r"建议动作[:：]", "策略建议："),
    (r"材料中最有解释力的事实组合是[:：]", "事实依据："),
    (r"当前事实组合是[:：]", "事实依据："),
    (r"这些事实需要按供应链层级拆开理解[:：]", "可按供应链层级理解："),
    (r"围绕“([^”]+)”，讨论应从", r"围绕“\1”，分析可从"),
    (r"围绕“([^”]+)”，讨论从事实组合开始，再转入成立条件和相反情形。公开材料显示[:：]", r"围绕“\1”，分析先看已经出现的产业信号，再看成立条件和反向情形。事实依据："),
    (r"后续跟踪应集中在", "后续重点观察"),
    (r"后续跟踪的重点落在", "后续重点观察"),
    (r"后续跟踪集中在", "后续重点观察"),
    (r"章节结论才适合上升为全篇主线", "这一判断才更适合成为全文主线"),
    (r"章节结论才会进入全篇主线", "这一判断才会进入全文主线"),
    (r"章节结论", "判断"),
    (r"本章可用来源约\d+条[，。]?", ""),
    (r"A/B层级来源约\d+条[，。]?", ""),
    (r"反向或边界信号约\d+条[，。]?", ""),
    (r"来源层级分布为[^。；\n]*[。；]?", ""),
    (r"本章写作时应", ""),
    (r"分析需要先", ""),
    (r"当前最直接的支持点是[:：]", "事实依据："),
    (r"当前可用于判断的事实组合包括[:：]", "可以放在一起观察的事实包括："),
    (r"围绕“([^”]+)”形成可验证判断", r"\1"),
    (r"进入总判断", "进入全篇主线"),
    (r"补证任务", "持续观察项"),
    (r"公开表达采用相应边界", "结论保留相应边界"),
    (r"关联证据[:：][^\n。]*[。]?", ""),
    (r"(?<![A-Za-z0-9_])EV-\d+(?:-[A-Za-z0-9]+)?", ""),
]


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _compact(value: Any, max_chars: int = 500) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "..."


TRADITIONAL_TO_SIMPLIFIED = str.maketrans(
    {
        "發": "发",
        "佈": "布",
        "體": "体",
        "團": "团",
        "業": "业",
        "務": "务",
        "軟": "软",
        "硬": "硬",
        "證": "证",
        "據": "据",
        "場": "场",
        "景": "景",
        "應": "应",
        "與": "与",
        "實": "实",
        "驗": "验",
        "轉": "转",
        "進": "进",
        "階": "阶",
        "段": "段",
        "價": "价",
        "為": "为",
        "單": "单",
        "個": "个",
        "對": "对",
        "雲": "云",
        "數": "数",
        "據": "据",
        "電": "电",
        "費": "费",
        "戶": "户",
        "產": "产",
        "鏈": "链",
    }
)


def _normalize_public_text(value: Any) -> str:
    text = str(value or "").translate(TRADITIONAL_TO_SIMPLIFIED)
    text = re.sub(r"[（(]\s*[）)]", "", text)
    text = re.sub(r"\.{1,}\s*。", "。", text)
    text = re.sub(r"…+\s*。", "。", text)
    text = re.sub(r"。\s*\.{1,}", "。", text)
    return text


def _public_text(value: Any, max_chars: int = 500) -> str:
    text = rewrite_internal_gap_language(_compact(_normalize_public_text(value), max_chars))
    for old, new in PUBLIC_TERM_REPLACEMENTS.items():
        text = text.replace(old, new)
    for pattern, replacement in PUBLIC_PROCESS_REWRITES:
        text = re.sub(pattern, replacement, text, flags=re.I)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"([。；]){2,}", r"\1", text)
    text = text.strip(" \t\r\n，；")
    return "" if has_internal_gap_language(text) else text


def _metric_sentence_from_block(block: Dict[str, Any], text: str) -> str:
    metric = _public_text(block.get("metric") or block.get("metric_name") or block.get("variable"), 60)
    value = _public_text(block.get("value") or block.get("numeric_value"), 60)
    unit = _public_text(block.get("unit") or block.get("numeric_unit"), 40)
    period = _public_text(block.get("period") or block.get("time_or_scope"), 80)
    scope = _public_text(block.get("scope") or block.get("subject"), 100)
    if not (metric and value):
        match = re.match(r"^([^:：]{2,40})[:：]\s*([^。；;\n]{1,80})", text)
        if match:
            metric = metric or _public_text(match.group(1), 60)
            value = value or _public_text(match.group(2), 80)
    if not (metric and value):
        return text
    value = value.rstrip("。；;")
    value_text = value if not unit or unit in value else f"{value}{unit}"
    prefix = metric
    if scope:
        prefix = f"{scope}{metric}"
    if period:
        return f"{prefix}在{period}为{value_text}。"
    if re.match(r"^(?:达|达到|超|超过|约|近|突破)", value_text):
        return f"{prefix}{value_text}。"
    return f"{prefix}为{value_text}。"


def _looks_like_bare_metric_text(value: Any) -> bool:
    text = str(value or "").strip()
    match = re.match(r"^([^:：]{2,40})[:：]\s*([^。；;\n]{1,80})", text)
    if not match:
        return False
    label = match.group(1).strip()
    return bool(
        re.search(
            r"规模|出货|部署|渗透|份额|收入|利润|成本|价格|订单|采购|客户|占比|增速|融资|市场|金额|数量",
            label,
        )
    )


def _looks_like_render_snippet(value: Any) -> bool:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = re.sub(r"^[\-•·]\s*", "", text)
    if not text:
        return False
    if re.search(r"字体\s*[:：]\s*大\s*中\s*小", text):
        return True
    if re.search(r"国内垂直领域研报服务|以下为本次访谈实录|电子工程专辑|爱分析访谈", text):
        return True
    if re.search(r"AI\s*时代，唯一确定的是数据", text, flags=re.I):
        return True
    if re.match(r"^[^。；;]{6,90}[｜|][^。；;]{2,90}\s*-\s*[^:：。]{2,50}(?:[（(]20\d{2}[^）)]*[）)])?[:：]", text):
        return True
    if re.match(r"^[^。；;]{6,90}\s*-\s*[^:：。]{2,50}(?:[（(]20\d{2}[^）)]*[）)])?[:：]", text):
        return True
    if re.match(r"^(?:显示|为此|因此|当前|相关|本文)[，,：:]", text):
        return True
    if "..." in text or "…" in text:
        return True
    if re.search(r"[（(]\s*[）)]", text):
        return True
    if re.match(r"^(?:近日|今日|日前|今年\s*\d+\s*月份?|过去\s*\d+\s*[天周月年]|一盆|一场|一句|一篇)", text):
        return True
    if re.match(r"^[^。；;]{6,48}-[^:：。]{2,24}[:：]\s*", text):
        return True
    if re.search(r"(?:^|[；。])\s*(?:一盆|一场|一句|一篇|今年\s*\d+\s*月份?|近日|日前)", text):
        return True
    if re.match(r"^[一二三四五六七八九十]+[、.．]", text):
        return True
    if re.match(r"^[^。]*[“\"][^”\"。]{8,}[”\"][^。]*[:：]", text):
        return True
    if re.match(r"^同的", text):
        return True
    if re.search(r"赛道[^。]{0,20}(?:爆发|加速|火热|红利)", text):
        return True
    if re.match(r"^(?:构建|打造|推出|发布|上线).{4,60}$", text) and not re.search(r"[。；;]", text):
        return True
    return False


PUBLIC_TEMPLATE_PHRASES = [
    "只能形成初步信号",
    "暂不宜外推",
    "低强度判断",
    "更多独立来源复核",
    "更多客户样本或反向案例",
    "代表性案例对比",
    "反向信号与失效条件",
    "后续影响",
    "使用边界",
    "该指标须",
    "须同时披露",
    "材料指向",
    "相关材料",
    "adoption:",
    "\u8be5\u5224\u65ad\u7684\u8fb9\u754c\u5728\u4e8e\u539f\u6587\u6838\u9a8c",
    "\u540e\u7eed\u89c2\u5bdf\u672c\u7ae0\u76f8\u5173",
    "\u540e\u7eed\u8865\u5145\u540c\u53e3\u5f84\u6307\u6807",
    "目前只保留为观察项",
    "目前只适合作为低强度观察项",
    "从概念讨论推进到可观察变量",
    "后续重点跟踪同口径指标",
    "后续重点跟踪",
    "事实链",
    "事实锚点",
    "交叉信号",
    "仅有 C 级",
    "更适合留在观察层",
    "共同构成本章判断的事实基础",
    "可用事实包括",
    "这些信息需要放回",
    "If later A/B sources",
    "downgrade the claim",
    "策略影响是把已有材料转成",
    "把已有事实拆成",
    "可作为观察产业优先级",
    "从行业判断看",
    "如果弱来源",
    "在评估结论时同步检查",
    "当前只能形成待验证方向",
    "后续优先追踪",
    "如果后续同口径指标走弱",
    "没有反证并不等于风险不存在",
    "该信号可作为本章的审慎结论",
    "边界在于样本是否代表主流需求",
    "本章信号集中在",
    "事实依据包括",
    "可复核事实显示",
    "可核验事实显示",
    "若相反样本或口径差异扩大",
    "分析重点是这些事实之间是否指向同一变量",
    "本段判断需要收窄",
]

INLINE_PARAGRAPH_LABELS = {
    "关键判断",
    "观察判断",
    "事实依据",
    "证据依据",
    "边界",
    "含义",
}


def _clean_render_text(value: Any, max_chars: int = 500) -> str:
    text = _public_text(value, max_chars)
    if not text:
        return ""
    if _looks_like_render_snippet(text):
        return ""
    text = re.sub(r"(?<=[\u4e00-\u9fff])\.\s*(?=[\u4e00-\u9fff])", "", text)
    text = re.sub(r"[^。\n]*该信号可作为本章的审慎结论[^。\n]*(?:。|$)", "", text)
    text = re.sub(r"[^。\n]*这些信息对应[^。\n]*变量[^。\n]*(?:。|$)", "", text)
    text = re.sub(r"[^。\n]*边界在于样本是否代表主流需求[^。\n]*(?:。|$)", "", text)
    text = re.sub(r"[^。\n]*本章信号集中在[^。\n]*(?:。|$)", "", text)
    text = re.sub(r"[^。\n]*可复核事实显示[^。\n]*(?:。|$)", "", text)
    text = re.sub(r"[^。\n]*可核验事实显示[^。\n]*(?:。|$)", "", text)
    text = re.sub(r"[^。\n]*若相反样本或口径差异扩大[^。\n]*(?:。|$)", "", text)
    text = re.sub(r"[^。\n]*分析重点是这些事实之间是否指向同一变量[^。\n]*(?:。|$)", "", text)
    text = re.sub(r"[^。\n]*本段判断需要收窄[^。\n]*(?:。|$)", "", text)
    text = re.sub(r"事实依据包括[:：]?", "", text)
    text = text.replace("事实锚点显示：", "")
    text = text.replace("事实锚点显示", "")
    text = text.replace("后续重点跟踪", "后续可观察")
    public_template_phrases = [
        "事实依据包括",
        "材料指向",
        "相关材料",
        "后续重点跟踪",
        "该信号可作为本章的审慎结论",
        "边界在于样本是否代表主流需求",
    ]
    if any(phrase in text for phrase in PUBLIC_TEMPLATE_PHRASES) or any(phrase in text for phrase in public_template_phrases):
        return ""
    return text.strip()


def _section_should_skip(section: Dict[str, Any]) -> bool:
    if section.get("omit_from_report"):
        return True
    if section.get("observation_only") and not section.get("evidence_backed") and not section.get("force_render_observation"):
        return True
    return False


def _is_internal_section_title(value: Any) -> bool:
    text = _compact(value, 160)
    generic_titles = {
        "事实依据",
        "商业化证据",
        "核心观察",
        "本章结论",
        "关键事实与判断依据",
        "判断边界与后续验证",
    }
    if text in generic_titles:
        return True
    if re.fullmatch(r"(?:H|h)\d{1,3}|ch[_-]?\d{1,3}", text.strip()):
        return True
    if text in {"代表性案例对比", "反向信号与失效条件", "市场空间是否成立", "付费转化是否成立"}:
        return True
    lowered = text.lower()
    if any(token in lowered for token in ("official_me", "source_check", "proof_role", "block_type")):
        return True
    if re.search(r"\b(?:metric|counter|case_comparison|risk_trigger|unit_economics|metric_reconciliation)\b", lowered):
        return True
    if re.fullmatch(r"[a-z]+(?:_[a-z0-9]+){1,4}.*", lowered):
        return True
    if any(term in text for term in ("证据", "口径", "变量", "判断依据", "可验证信号")):
        return True
    return bool(text and any(re.search(pattern, text) for pattern in INTERNAL_SECTION_TITLE_PATTERNS))


def _section_public_title(section: Dict[str, Any]) -> str:
    plan = _as_dict(section.get("section_plan"))
    for value in (plan.get("public_title"), section.get("dynamic_section_title"), section.get("section_title")):
        title = _compact(value, 120)
        if title and not _is_internal_section_title(title):
            return title
    return ""


def _section_title_key(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "").strip().lower())


def _title_from_section_claim(section: Dict[str, Any], *, max_chars: int = 24) -> str:
    text = _public_text(section.get("claim") or section.get("paragraph") or section.get("reasoning"), 120)
    if not text:
        return ""
    head = re.split(r"[\u3002\uff1b\uff0c\uff1a;,:，。；：]", text, 1)[0].strip()
    head = re.sub(r"^(?:机会判断|方向性判断|核心判断)\s*[:：]\s*", "", head).strip()
    if len(head) < 4:
        return ""
    title = _compact(head, max_chars).strip(" ，,。；;：:")
    if len(title) < 4 or _is_internal_section_title(title):
        return ""
    return title


def _unique_section_title(section: Dict[str, Any], seen_titles: set[str], *, section_index: int) -> str:
    title = _section_public_title(section)
    key = _section_title_key(title)
    if title and key and key not in seen_titles:
        seen_titles.add(key)
        return title
    claim_title = _title_from_section_claim(section)
    claim_key = _section_title_key(claim_title)
    if claim_title and claim_key and claim_key not in seen_titles:
        seen_titles.add(claim_key)
        return claim_title
    if title:
        suffix = f"{title}（{section_index}）"
        suffix_key = _section_title_key(suffix)
        seen_titles.add(suffix_key)
        return suffix
    return ""


def _natural_transition(prefix: str, text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return ""
    if text.startswith(("因此", "所以", "不过", "但", "如果", "需要注意", "从")):
        return text
    return f"{prefix}{text}"


def _dedupe(values: Iterable[Any], *, limit: int = 20, max_chars: int = 240) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
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


TOKEN_PROFILE_INT_DEFAULTS = {
    "lean": {
        "REPORT_RENDER_BLOCK_MAX_CHARS": 1800,
        "REPORT_SECTION_CLAIM_MAX_CHARS": 700,
        "REPORT_SECTION_REASONING_MAX_CHARS": 1800,
        "REPORT_SECTION_COUNTER_MAX_CHARS": 900,
        "REPORT_SECTION_ACTION_MAX_CHARS": 900,
        "REPORT_SECTION_MECHANISM_MAX_CHARS": 1400,
        "REPORT_RENDER_FACT_DIGEST_LIMIT": 0,
    },
    "balanced": {
        "REPORT_RENDER_BLOCK_MAX_CHARS": 2400,
        "REPORT_SECTION_CLAIM_MAX_CHARS": 900,
        "REPORT_SECTION_REASONING_MAX_CHARS": 2400,
        "REPORT_SECTION_COUNTER_MAX_CHARS": 1200,
        "REPORT_SECTION_ACTION_MAX_CHARS": 1200,
        "REPORT_SECTION_MECHANISM_MAX_CHARS": 1800,
        "REPORT_RENDER_FACT_DIGEST_LIMIT": 0,
    },
    "deep": {
        "REPORT_RENDER_BLOCK_MAX_CHARS": 3200,
        "REPORT_SECTION_CLAIM_MAX_CHARS": 1100,
        "REPORT_SECTION_REASONING_MAX_CHARS": 3200,
        "REPORT_SECTION_COUNTER_MAX_CHARS": 1600,
        "REPORT_SECTION_ACTION_MAX_CHARS": 1600,
        "REPORT_SECTION_MECHANISM_MAX_CHARS": 2400,
        "REPORT_RENDER_FACT_DIGEST_LIMIT": 0,
    },
}


def _profile_default(name: str, default: int) -> int:
    if os.getenv(name) is not None:
        return default
    profile = str(os.getenv("REPORT_TOKEN_PROFILE", "balanced") or "balanced").strip().lower()
    return TOKEN_PROFILE_INT_DEFAULTS.get(profile, TOKEN_PROFILE_INT_DEFAULTS["balanced"]).get(name, default)


def _env_int(name: str, default: int, *, min_value: int = 0, max_value: int = 10_000) -> int:
    default = _profile_default(name, default)
    try:
        value = int(os.getenv(name, str(default)) or default)
    except (TypeError, ValueError):
        value = default
    return max(min_value, min(max_value, value))


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _line_key(value: Any) -> str:
    return re.sub(r"[\s，。；：:、,.!?！？“”\"'（）()《》]+", "", str(value or "")).lower()


def _dedupe_narrative_lines(lines: Sequence[str]) -> List[str]:
    result: List[str] = []
    seen = set()
    for line in lines:
        text = str(line or "")
        stripped = text.strip()
        if not stripped:
            if result and result[-1].strip():
                result.append(text)
            continue
        if stripped.startswith(("#", "|", "**", "- [")) or re.match(r"^\|?\s*-{3,}", stripped):
            result.append(text)
            continue
        key = _line_key(stripped)
        if len(key) >= 36 and key in seen:
            continue
        if len(key) >= 36:
            seen.add(key)
        result.append(text)
    while result and not result[-1].strip():
        result.pop()
    return result


def strip_internal_layout_language(text: str) -> str:
    result = str(text or "")
    for old, new in PUBLIC_TERM_REPLACEMENTS.items():
        result = result.replace(old, new)
    for pattern, replacement in PUBLIC_PROCESS_REWRITES:
        result = re.sub(pattern, replacement, result, flags=re.I)
    for phrase in INTERNAL_LAYOUT_PHRASES:
        result = result.replace(phrase, "")
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def strip_body_qa_leaks(text: str) -> str:
    result = str(text or "")
    result = re.sub(r"(?im)^\s*(QA|Self[- ]?check|Validation|质量检查)[:：].*$", "", result)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def normalize_markdown_spacing(text: str) -> str:
    result = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    result = re.sub(r"[ \t]+\n", "\n", result)
    result = result.replace("。；", "；").replace("；。", "。").replace("。。", "。")
    result = re.sub(r"\n{3,}", "\n\n", result)
    result = re.sub(r"(?m)^(#{1,6})\s*", r"\1 ", result)
    return result.strip() + ("\n" if result.strip() else "")


def _report_title(research_object: str) -> str:
    title = _compact(research_object, 120).strip(" #")
    if not title:
        title = "研究对象"
    if re.search(r"(报告|研究|简报)$", title):
        return f"# {title}"
    return f"# {title}研究报告"


def _cover_title_from_blueprint(query: str, report_blueprint: Dict[str, Any]) -> str:
    brief = _as_dict(report_blueprint.get("article_brief"))
    explicit = (
        report_blueprint.get("report_title")
        or report_blueprint.get("display_title")
        or brief.get("display_title")
        or brief.get("main_title")
    )
    title = _compact(explicit, 140).strip(" #")
    if title:
        return f"# {title}"
    research_object = str(report_blueprint.get("research_object") or query or "研究对象").strip()
    return _report_title(research_object)


def _cover_subtitle_from_blueprint(report_blueprint: Dict[str, Any]) -> str:
    brief = _as_dict(report_blueprint.get("article_brief"))
    subtitle = _compact(
        report_blueprint.get("report_subtitle")
        or report_blueprint.get("display_subtitle")
        or brief.get("display_subtitle")
        or brief.get("direction"),
        220,
    ).strip()
    subtitle = re.sub(r"^[—–-]{1,3}\s*", "", subtitle).strip()
    return f"——{subtitle}" if subtitle else ""


def render_cover(query: str, report_blueprint: Dict[str, Any]) -> str:
    research_object = str(report_blueprint.get("research_object") or query or "研究对象").strip()
    narrative = _public_text(report_blueprint.get("narrative"), 240)
    lines = [_cover_title_from_blueprint(query, report_blueprint), ""]
    subtitle = _cover_subtitle_from_blueprint(report_blueprint)
    if subtitle:
        lines.extend([subtitle, ""])
    if narrative:
        lines.append(f"研究主线：{narrative}")
    return "\n".join(_dedupe_narrative_lines(lines))


def _public_tables(table_packages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        table
        for table in list(table_packages or [])
        if isinstance(table, dict)
        and table.get("should_render")
        and not table.get("appendix_only")
        and len(_as_list(table.get("rows"))) >= 2
    ]


def _summary_judgment_needs_citation(value: Any) -> bool:
    text = str(value or "")
    return bool(
        re.search(
            r"(?:\d{4}\s*年|\d+(?:\.\d+)?\s*(?:%|亿|万|亿美元|亿元|家公司|个项目|项)|"
            r"CAGR|近\s*\d+|资本市场|高估值|估值|IPO|独角兽|递表|港股|上市|"
            r"多家厂商|推出相关产品|入选|产业图谱|标杆案例|市场空间)",
            text,
            re.I,
        )
    )


def _key_data_bullet_from_table_row(headers: Sequence[Any], row: Dict[str, Any]) -> str:
    cells = _as_list(row.get("cells"))
    metric = _compact(row.get("metric_name") or row.get("metric") or (cells[0] if cells else ""), 60)
    if not metric:
        return ""
    value_candidates: List[Any] = [
        row.get("value_display"),
        row.get("display_value"),
        row.get("value"),
        row.get("numeric_value"),
    ]
    for header, cell in zip(headers, cells):
        header_text = str(header or "").strip().lower()
        if any(token in header_text for token in ("数值", "信号", "影响", "成熟度", "评分", "value", "score", "signal")):
            value_candidates.append(cell)
    for cell in cells:
        text = str(cell or "").strip()
        if re.search(r"\d|%|亿|万|元|CAGR", text, re.I):
            value_candidates.append(cell)
    value = ""
    for candidate in value_candidates:
        text = _compact(candidate, 70)
        if not text:
            continue
        if re.fullmatch(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", text):
            continue
        if re.search(r"\d|%|亿|万|元|CAGR", text, re.I):
            value = text
            break
    if not value:
        return ""
    unit = _compact(row.get("unit") or row.get("numeric_unit"), 24)
    if unit and unit not in value and len(unit) <= 8 and re.search(r"%|元|亿|万|吨|台|家|个|pct|percent", unit, re.I):
        value = f"{value}{unit}"
    period = _compact(row.get("period"), 40)
    if not period:
        for header, cell in zip(headers, cells):
            header_text = str(header or "").strip().lower()
            if any(token in header_text for token in ("期间", "时间", "period", "window")):
                period = _compact(cell, 40)
                break
    text = f"{metric}为{value}" + (f"，期间为{period}" if period else "")
    return _public_text(text, 220)


def render_executive_summary(decision_package: Dict[str, Any], table_packages: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    judgments = [_as_dict(item) for item in _as_list(decision_package.get("core_judgments"))]
    judgments, _summary_diag = sanitize_summary_judgments(judgments, max_items=5)
    judgment_lines = []
    for item in judgments[:5]:
        judgment = _public_text(item.get("judgment"), 260)
        label = _compact(item.get("label"), 40)
        if label in INTERNAL_BLOCK_LABELS or _is_internal_section_title(label):
            label = ""
        if judgment:
            suffix = _citation_suffix(normalize_evidence_refs(item))
            if _summary_judgment_needs_citation(judgment) and not suffix:
                continue
            if suffix and not re.search(r"(?:\[\d{1,5}\])+\s*$", judgment):
                judgment = judgment.rstrip("。；; ") + "。" + suffix
            judgment_lines.append(f"- {label + '：' if label else ''}{judgment}")
    if judgment_lines:
        lines.extend(["## 核心观点与主要结论", *judgment_lines])

    key_rows = []
    for table in _public_tables(table_packages):
        headers = _as_list(table.get("headers"))
        for row in _as_list(table.get("rows"))[:1]:
            row = _as_dict(row)
            text = _key_data_bullet_from_table_row(headers, row)
            # Require the bullet to carry at least one quantitative or date token
            # so we never list a bare metric name.
            if text and re.search(r"\d|%|亿|万|元|美元|CAGR.{0,8}\d", text):
                # And require at least one token beyond a year/period reference.
                if not re.fullmatch(r"[A-Za-z一-鿿]+\s*[；;,，]\s*\d{4}\s*年?", text):
                    key_rows.append(text)
    if key_rows:
        if lines:
            lines.append("")
        lines.extend(["## 关键数据", *[f"- {item}" for item in _dedupe(key_rows, limit=5)]])
    return "\n".join(_dedupe_narrative_lines(lines))


def _markdown_table(headers: Sequence[Any], rows: Sequence[Sequence[Any]]) -> str:
    headers = [_public_text(str(header or "").replace("|", "/").strip(), 80) for header in headers]
    if not headers or not rows:
        return ""
    cleaned_rows: List[List[str]] = []
    for row in rows:
        cells = []
        for cell in row:
            text = str(cell or "").replace("\n", " ").replace("|", "/").strip()
            text = re.sub(r"第\s*\d+\s*轮\s*[｜|:：-]*\s*", "", text)
            text = re.sub(r"\b(?:query|openai_task_\d+|claim_status|evidence_cards)\b\s*[:：=]?\s*", "", text, flags=re.I)
            text = re.sub(r"(?:竞争对比|政策监管|技术产业链|市场规模|成本|金额)\s*=\s*(?=[；;，,]|$)", "", text)
            text = re.sub(r"第\s*\d+\s*轮\s*[｜|:：]\s*", "", text)
            text = re.sub(r"(?:竞争对比|政策监管|技术产业链|市场规模|成本|金额)\s*=\s*(?=；|;|$)", "", text)
            text = re.sub(r"(?:；\s*){2,}", "；", text).strip(" ；;，,")
            if re.search(r"(?:第\s*\d+\s*轮|openai_task_\d+)", text, flags=re.I):
                text = ""
            if re.search(r"(?:第\s*\d+\s*轮|openai_task_\d+|claim_status|evidence_cards)", text, flags=re.I):
                text = ""
            cells.append(_public_text(text, 220))
        cells = (cells + [""] * len(headers))[: len(headers)]
        if any(cell.strip() for cell in cells):
            cleaned_rows.append(cells)
    if len(cleaned_rows) < 2:
        return ""
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for cells in cleaned_rows:
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _is_internal_table_header(header: Any) -> bool:
    text = str(header or "").strip().lower()
    return bool(
        re.search(
            r"(来源|引用|证据|后续影响|使用边界|进入判断|验证方法|观察指标|"
            r"competitive\s+signal|risk\s+boundary|implication|boundary|"
            r"evidence|source|ref)",
            text,
        )
    )


def _public_table_shape(headers: Sequence[Any], rows: Sequence[Sequence[Any]]) -> tuple[List[Any], List[List[Any]]]:
    keep_indices = [index for index, header in enumerate(headers) if not _is_internal_table_header(header)]
    public_headers = [headers[index] for index in keep_indices]
    public_rows = []
    for row in rows:
        row_values = list(row)
        public_rows.append([row_values[index] if index < len(row_values) else "" for index in keep_indices])
    return public_headers, public_rows


DIAGNOSTIC_TABLE_PATTERNS = [
    r"投资优先级矩阵",
    r"报告级检索缺口",
    r"检索缺口",
    r"存疑",
    r"评分",
    r"raw\s+url",
    r"raw\s+english",
    r"后续影响",
    r"使用边界",
    r"该指标须",
    r"须同时披露",
    r"不会凭空补齐",
    r"进入正文判断",
    r"缺口数据",
    r"后续验证项",
    r"Competitive signal",
    r"Risk boundary",
]


def _has_diagnostic_table_language(values: Sequence[Any]) -> bool:
    text = "\n".join(str(value or "") for value in values)
    return any(re.search(pattern, text, flags=re.I) for pattern in DIAGNOSTIC_TABLE_PATTERNS)


def _header_index(headers: Sequence[Any], *patterns: str) -> int:
    for index, header in enumerate(headers):
        text = str(header or "").strip()
        if any(re.search(pattern, text, flags=re.I) for pattern in patterns):
            return index
    return -1


def _filter_valid_metric_rows(headers: Sequence[Any], rows: Sequence[Sequence[Any]]) -> List[List[Any]]:
    metric_idx = _header_index(headers, r"指标", r"\bmetric\b")
    value_idx = _header_index(headers, r"数值", r"\bvalue\b")
    unit_idx = _header_index(headers, r"单位", r"\bunit\b")
    period_idx = _header_index(headers, r"期间", r"时间", r"\bperiod\b")
    scope_idx = _header_index(headers, r"范围", r"主体", r"对象", r"\bscope\b", r"\bsubject\b")
    if metric_idx < 0 or value_idx < 0 or unit_idx < 0:
        return [list(row) for row in rows]
    required_indices = [idx for idx in (metric_idx, value_idx, unit_idx, period_idx, scope_idx) if idx >= 0]
    valid_rows: List[List[Any]] = []
    for row in rows:
        row_values = list(row)
        padded = (row_values + [""] * len(headers))[: len(headers)]
        if any(not str(padded[idx] or "").strip() for idx in required_indices):
            continue
        if _has_diagnostic_table_language(padded):
            continue
        valid_rows.append(padded)
    return valid_rows


def _line_with_citations(text: str, evidence_refs: Sequence[Any]) -> str:
    suffix = _citation_suffix(evidence_refs)
    if not suffix or re.search(r"\[\d{1,3}\]\s*$", text):
        return text
    return text.rstrip("。；;，,") + "。" + suffix


def _table_validation_passed(table: Dict[str, Any]) -> bool:
    validation = _as_dict(table.get("validation") or table.get("table_validation_for_clean"))
    if validation and validation.get("passed") is False:
        return False
    if table.get("validation_error") or table.get("table_validation_error"):
        return False
    if _as_list(table.get("reject_reasons")):
        return False
    if str(table.get("metric_validation_status") or "").strip().lower() == "invalid":
        return False
    for row in _as_list(table.get("rows")):
        payload = _as_dict(row)
        if str(payload.get("metric_validation_status") or "").strip().lower() == "invalid":
            return False
    return True


def _invalid_metric_appendix_row(item: Dict[str, Any]) -> bool:
    metric = str(item.get("metric_name") or item.get("metric") or item.get("indicator") or "").strip()
    value = str(item.get("value") or item.get("display_value") or item.get("numeric_value") or "").strip()
    unit = str(item.get("unit") or "").strip()
    status = str(item.get("metric_validation_status") or item.get("validation_status") or "").strip().lower()
    context = " ".join(str(item.get(key) or "") for key in ("period", "source_title", "fact", "clean_fact"))
    if status == "invalid":
        return True
    if not metric or not value:
        return True
    if metric in {"关键事实", "政策监管", "政策目标", "source_check", "status", "http_status"}:
        return True
    if re.search(r"政策|目标|监管", metric) and re.match(r"-\d", value):
        return True
    if re.search(r"成本", metric) and re.search(r"家$", value):
        return True
    if re.search(r"市场规模|融资", metric) and ("%" in value or "%" in unit):
        return True
    if re.fullmatch(r"-?\d{1,3}(?:\.0)?", value) and re.search(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", context):
        return True
    return False


def render_table_package(table: Dict[str, Any]) -> str:
    if not table.get("should_render") or table.get("appendix_only"):
        return ""
    if not _table_validation_passed(table):
        return ""
    headers = _as_list(table.get("headers"))
    row_objects = [_as_dict(row) for row in _as_list(table.get("rows")) if isinstance(row, dict)]
    rows = [_as_list(row.get("cells")) for row in row_objects]
    headers, rows = _public_table_shape(headers, rows)
    table_text_values: List[Any] = [
        table.get("title"),
        table.get("takeaway"),
        table.get("decision_implication"),
        *_as_list(table.get("limitations")),
        *headers,
        *[cell for row in rows for cell in row],
    ]
    if _has_diagnostic_table_language(table_text_values):
        return ""
    rows = _filter_valid_metric_rows(headers, rows)
    if len(headers) < 2:
        return ""
    minimum_rows = 2
    if len(rows) < minimum_rows:
        return ""
    table_md = _markdown_table(headers, rows)
    if not table_md:
        return ""
    parts = [f"**{_compact(table.get('title'), 120)}**", "", table_md]
    decision_implication = _public_text(table.get("decision_implication"), 260)
    limitations = [
        item
        for item in (_public_text(value, 160) for value in _as_list(table.get("limitations"))[:1])
        if item
    ]
    takeaway = _public_text(table.get("takeaway"), 220)
    citation_refs = _as_list(table.get("evidence_refs")) or [
        ref
        for row in row_objects
        for ref in _as_list(row.get("evidence_refs"))
    ]
    if takeaway:
        parts.extend(["", _line_with_citations(f"这张表显示，{takeaway}", citation_refs)])
    if decision_implication and decision_implication != takeaway:
        parts.extend(["", _line_with_citations(f"判断含义：{decision_implication}", citation_refs)])
    if limitations:
        parts.extend(["", _line_with_citations(f"使用边界：{limitations[0]}", citation_refs)])
    return "\n".join(parts)


def render_evidence_inventory(evidence_refs: Sequence[Any]) -> List[str]:
    refs = _dedupe([str(ref or "").strip() for ref in evidence_refs if str(ref or "").strip()], limit=12)
    return [f"- {ref}" for ref in refs]


def _citation_suffix(evidence_refs: Sequence[Any], *, limit: int = 3) -> str:
    refs: List[str] = []
    for value in evidence_refs:
        text = str(value or "").strip()
        if not text:
            continue
        match = re.fullmatch(r"\[?(\d{1,3})\]?", text)
        if match:
            refs.append(f"[{match.group(1)}]")
            continue
        match = re.search(r"\[(\d{1,3})\]", text)
        if match:
            refs.append(f"[{match.group(1)}]")
    refs = _dedupe(refs, limit=limit)
    return "".join(refs)


def _append_citation_to_last_paragraph(lines: List[str], evidence_refs: Sequence[Any]) -> None:
    suffix = _citation_suffix(evidence_refs)
    if not suffix:
        return
    suffix_refs = re.findall(r"\[\d{1,5}\]", suffix)
    for index in range(len(lines) - 1, -1, -1):
        line = str(lines[index] or "").rstrip()
        if not line or line.startswith("#") or line.startswith("|") or re.match(r"^[:\-\s|]+$", line):
            continue
        line_refs = re.findall(r"\[\d{1,5}\]", line)
        if line_refs and line_refs == suffix_refs and re.search(r"(?:\[\d{1,5}\])+\s*$", line):
            return
        if line_refs:
            line = re.sub(r"\s*\[\d{1,5}\]", "", line).rstrip()
        lines[index] = line.rstrip("。；;，,") + "。" + suffix
        return


def _first_section_citation_refs(chapter: Dict[str, Any]) -> List[Any]:
    for section in _as_list(chapter.get("sections")):
        section = _as_dict(section)
        refs = _as_list(section.get("citation_refs")) or _as_list(section.get("evidence_refs"))
        if refs:
            return refs
    return []


def _paragraph_chunks_with_citations(
    text: str,
    evidence_refs: Sequence[Any],
    *,
    max_chars: int = 720,
    max_chunks: int = 5,
) -> List[str]:
    chunks = _paragraph_chunks(text, max_chars=max_chars, max_chunks=max_chunks)
    suffix = _citation_suffix(evidence_refs)
    if not suffix:
        return chunks
    result: List[str] = []
    for chunk in chunks:
        line = str(chunk or "").rstrip()
        if not line or re.search(r"(?:\[\d{1,5}\])+\s*$", line):
            result.append(line)
            continue
        result.append(line.rstrip("\u3002\uff1b;； ") + "\u3002" + suffix)
    return result


def render_dynamic_table(block: Dict[str, Any]) -> List[str]:
    headers = _as_list(block.get("headers"))
    rows = [_as_list(row) for row in _as_list(block.get("rows"))]
    table_md = _markdown_table(headers, rows)
    return [table_md] if table_md else []


def _paragraph_chunks(text: str, *, max_chars: int = 720, max_chunks: int = 5) -> List[str]:
    text = _public_text(text, max_chars * max_chunks)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    sentences = [item.strip() for item in re.split(r"(?<=[。；！？.!?])\s*", text) if item.strip()]
    if len(sentences) <= 1:
        return [text[index : index + max_chars].strip() for index in range(0, len(text), max_chars) if text[index : index + max_chars].strip()][:max_chunks]
    chunks: List[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) > max_chars:
            chunks.append(current.strip())
            current = sentence
        else:
            current = (current + sentence).strip()
        if len(chunks) >= max_chunks:
            break
    if current and len(chunks) < max_chunks:
        chunks.append(current.strip())
    return chunks[:max_chunks]


def _append_narrative_block(lines: List[str], title: str, text: str, *, max_chars: int) -> bool:
    chunks = _paragraph_chunks(text, max_chars=max_chars)
    if not chunks:
        return False
    if title:
        lines.extend(["", f"#### {title}"])
    lines.extend(chunks)
    return True


def render_chapter_deep_synthesis(chapter: Dict[str, Any]) -> List[str]:
    if not _env_flag("REPORT_RENDER_CHAPTER_DEEP_SYNTHESIS", False):
        return []
    summary = _as_dict(chapter.get("chapter_summary"))
    if not summary:
        return []
    title = _public_text(chapter.get("chapter_title"), 140)
    takeaway = _public_text(summary.get("key_takeaway"), 700)
    mechanisms = _dedupe([_public_text(item, 420) for item in _as_list(summary.get("mechanisms"))], limit=3)
    counters = _dedupe([_public_text(item, 420) for item in _as_list(summary.get("counter_evidence"))], limit=3)
    actions = _dedupe([_public_text(item, 420) for item in _as_list(summary.get("next_actions"))], limit=4)
    watch = _dedupe([_public_text(item, 260) for item in _as_list(summary.get("what_to_verify_next"))], limit=5)
    if not any([takeaway, mechanisms, counters, actions, watch]):
        return []
    lines = [""]
    if takeaway:
        lines.append(
            f"围绕“{title}”，本章最直接的判断是：{takeaway}"
            "这部分不单独替代全篇结论，而是说明当前事实能把判断推进到什么程度。"
        )
    if mechanisms:
        lines.append(
            "影响路径上，需要把事实之间的先后关系讲清楚："
            + "；".join(mechanisms)
            + "。这些关系成立时，章节结论才有继续外推的基础。"
        )
    if counters:
        lines.append(
            "结论需要保留的反向条件包括："
            + "；".join(counters)
            + "。这些条件出现时，本章判断应降级或重新校准。"
        )
    if actions:
        lines.append(
            "落到行动层面，本章对应的优先级是："
            + "；".join(actions)
            + "。这些动作应优先服务于缩小判断分歧，而不是扩大未经验证的假设。"
        )
    if watch:
        lines.append(
            "后续观察应聚焦这些触发器："
            + "；".join(watch)
            + "。这些触发器的价值在于让结论可以被复核、被更新，也可以在条件变化时及时收缩。"
        )
    return lines


def _compact_char_count(text: str) -> int:
    return len(re.sub(r"\s+", "", str(text or "")))


def _public_section_expansion_sentences(section: Dict[str, Any]) -> List[str]:
    title = _public_text(section.get("section_title") or section.get("title"), 80) or "这一判断"
    block_type = str(section.get("block_type") or "").strip().lower()
    if block_type in {"case_comparison", "customer_painpoint_matrix", "integrated_signal"}:
        mechanism = "从机制上看，这类信号的价值不在于单点案例本身，而在于它是否进入真实业务流程，并要求客户在权限、系统集成、流程责任和交付结果之间形成稳定安排。"
        implication = "行业含义在于，部署动作一旦从试用界面进入日常工作流，需求判断就会从概念热度转向可复用的业务场景，但商业化深度仍取决于客户是否愿意为持续使用、效率改善或风险降低付费。"
    elif block_type == "metric_reconciliation":
        mechanism = "从机制上看，指标只有放回主体、范围、期间和口径中解释，才适合支撑市场空间或商业化节奏判断；孤立数值只能说明局部信号，不能直接推出行业总量。"
        implication = "行业含义在于，口径越清晰，越能区分真实需求扩张、短期主题交易和单一来源估算之间的差异，也越能判断相关机会是否具备持续验证价值。"
    elif block_type in {"risk_trigger", "boundary"}:
        mechanism = "从机制上看，风险信号的作用不是否定全部机会，而是指出结论失效的触发条件；当安全、成本、责任或可靠性约束放大时，原有增长判断需要随之降级。"
        implication = "行业含义在于，边界条件越清楚，报告越能区分已经被事实支撑的机会和仍停留在假设层面的机会，从而避免把局部乐观样本写成确定性趋势。"
    elif block_type == "technology_maturity":
        mechanism = "从机制上看，技术成熟度会同时影响可靠性、权限治理、安全边界和集成成本；这些变量决定相关能力能否从演示环境进入生产流程。"
        implication = "行业含义在于，技术事实只有与部署深度、客户流程和持续运维要求相连，才能真正解释商业化速度，而不是只停留在功能展示层面。"
    else:
        mechanism = "从机制上看，公开事实需要被放进需求、供给、商业化、竞争和风险变量之间理解；只有这些变量形成连续关系，章节判断才具备分析厚度。"
        implication = "行业含义在于，同一事实在不同场景下可能对应不同强度的结论，报告需要把可确认部分、方向性部分和仍需保留的边界分别写清。"
    return [
        f"对“{title}”这一判断而言，关键不只是事实是否出现，而是它如何改变需求兑现、商业化节奏、技术约束或竞争位置。",
        mechanism,
        implication,
        "如果把它放在报告主线中，较稳妥的写法是先确认事实能够支撑的最低结论，再讨论它向更大范围外推时需要满足的关键条件。",
        "这种处理方式可以让读者同时看到机会信号和约束条件：前者说明为什么值得关注，后者说明为什么不能把局部样本直接写成行业定论。",
        "因此，这一段更适合作为有边界的分析信号来使用：它可以提高对相关机会的判断密度，但仍需要和同章其他来源共同构成证据链，避免单一材料承担过强结论。",
    ]


def _expand_short_public_paragraph(text: str, section: Dict[str, Any], citation_refs: Sequence[Any]) -> str:
    if not _env_flag("REPORT_ENABLE_RENDERER_TEMPLATE_EXPANSION", False):
        return text
    target = _env_int("REPORT_RENDER_MIN_SECTION_CHARS", 0, min_value=0, max_value=2000)
    if target <= 0 or _compact_char_count(text) >= target:
        return text
    if not citation_refs or not bool(section.get("evidence_backed")):
        return text
    parts = [text]
    for sentence in _public_section_expansion_sentences(section):
        if sentence and sentence not in parts:
            parts.append(sentence)
        if _compact_char_count(" ".join(parts)) >= target:
            break
    return " ".join(part.strip() for part in parts if part.strip()).strip()


def render_section(section: Dict[str, Any]) -> List[str]:
    if _section_should_skip(section):
        return []
    if section.get("observation_only") and section.get("force_render_observation") and not section.get("evidence_backed"):
        title = _section_public_title(section)
        claim = _clean_render_text(section.get("claim") or section.get("reasoning"), 360)
        lines = []
        if title and not _is_internal_section_title(title):
            lines.append(f"### {title}")
        if claim:
            lines.extend(_paragraph_chunks(claim, max_chars=360, max_chunks=1))
        return [line for line in lines if line.strip()]
    lines: List[str] = []
    title = _section_public_title(section)
    if title and not _is_internal_section_title(title):
        lines.append(f"### {title}")
    section_citation_refs = _as_list(section.get("citation_refs")) or _as_list(section.get("evidence_refs"))
    for block in _as_list(section.get("render_blocks")):
        block = _as_dict(block)
        block_type = str(block.get("type") or "").strip()
        raw_label = _compact(block.get("label"), 80)
        label = _public_text(block.get("label"), 80)
        text = _clean_render_text(block.get("text"), _env_int("REPORT_RENDER_BLOCK_MAX_CHARS", 3200, min_value=800, max_value=8000))
        block_fact_type = str(block.get("fact_type") or section.get("fact_type") or "").strip().lower()
        if text and (
            (
                block_fact_type == "metric"
                or str(section.get("block_type") or "") == "metric_reconciliation"
            )
            and re.match(r"^[^:：]{2,40}[:：]\s*[^。；;\n]{1,80}", text)
            or _looks_like_bare_metric_text(text)
        ):
            text = _metric_sentence_from_block(block, text)
        is_internal_label = raw_label in INTERNAL_BLOCK_LABELS or _is_internal_section_title(raw_label)
        if block_type == "paragraph":
            if not text:
                continue
            text = _expand_short_public_paragraph(text, section, section_citation_refs)
            if label and not is_internal_label and label not in INLINE_PARAGRAPH_LABELS:
                lines.extend(["", f"#### {label}"])
                lines.extend(_paragraph_chunks_with_citations(text, section_citation_refs))
                continue
            else:
                lines.extend(_paragraph_chunks_with_citations(text, section_citation_refs))
        elif block_type == "evidence_list":
            continue
        elif block_type == "table":
            rendered = render_dynamic_table(block)
            if rendered:
                if label and not is_internal_label:
                    lines.append(label)
                lines.extend(rendered)
    return [line for line in lines if line.strip()]


def _fact_digest_chunks(facts: Sequence[str], *, chunk_size: int = 4) -> List[List[str]]:
    limit = _env_int("REPORT_RENDER_FACT_DIGEST_LIMIT", 0, min_value=0, max_value=40)
    if limit <= 0:
        return []
    cleaned = [
        item
        for item in (_public_text(value, 520) for value in facts)
        if item and not has_internal_gap_language(item)
    ]
    deduped = _dedupe(cleaned, limit=limit, max_chars=520)
    return [deduped[index : index + chunk_size] for index in range(0, len(deduped), chunk_size)]


def render_chapter_fact_digest(chapter: Dict[str, Any]) -> List[str]:
    chunks = _fact_digest_chunks(_as_list(chapter.get("chapter_fact_digest")), chunk_size=4)
    if not chunks:
        return []
    lines: List[str] = []
    for chunk in chunks[:3]:
        text = "；".join(chunk)
        if text:
            lines.append(text)
    return [line for line in lines if line.strip()]


def _chapter_flow_intro(
    chapter: Dict[str, Any],
    *,
    index: int,
    previous_chapter: Dict[str, Any] | None = None,
) -> str:
    """Generate an opening line for a chapter.

    Previous implementation always emitted "本章聚焦"X"。围绕"Y",重点看事实能支持
    到哪一步,以及哪些条件会削弱判断。" for every chapter, which made the report
    feel templated. Now:
    - If the chapter already has a substantive `lead` field, return empty (the
      lead itself becomes the chapter opening).
    - Otherwise emit only the shortest contextual sentence — no boilerplate
      tail like "重点看事实能支持到哪一步".
    - Can be disabled entirely via REPORT_DISABLE_CHAPTER_INTRO=1.
    """
    if os.environ.get("REPORT_DISABLE_CHAPTER_INTRO", "").strip() in {"1", "true", "True"}:
        return ""
    lead = _clean_render_text(chapter.get("lead"), 360)
    if lead:
        # The chapter already carries its own opening narrative; do not append a template.
        return ""
    return ""


def _chapter_transition(chapter: Dict[str, Any], next_chapter: Dict[str, Any] | None = None) -> str:
    """Inter-chapter transition sentence.

    Previously hard-coded as "由此,X给出了当前判断的成立条件;接下来的 Y 会继续
    检验这些条件是否能延续。" — which appeared at the end of every chapter and
    is the single biggest contributor to the "machine-stitched" feel of the
    rendered report.

    Now: disabled by default. Only emitted when REPORT_ENABLE_CHAPTER_TRANSITION=1
    explicitly opts back in. Even then, requires a real takeaway on the current
    chapter (not just a chapter title) to actually output.
    """
    if os.environ.get("REPORT_ENABLE_CHAPTER_TRANSITION", "0").strip() not in {"1", "true", "True"}:
        return ""
    title = _public_text(chapter.get("chapter_title"), 100)
    next_title = _public_text(_as_dict(next_chapter).get("chapter_title"), 100)
    takeaway = _public_text(_as_dict(chapter.get("chapter_summary")).get("key_takeaway"), 160)
    if not (title and next_title and takeaway):
        return ""
    return f"由此,“{title}”给出了当前判断的成立条件;接下来的“{next_title}”会继续检验这些条件是否能延续。"


def _final_action_phrase(value: str) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^后续重点跟踪", "", text)
    text = re.sub(r"^后续跟踪集中在", "", text)
    text = re.sub(r"^后续跟踪", "", text)
    text = re.sub(r"^重点跟踪", "", text)
    return text.strip(" ：，。；") or str(value or "").strip()


def _clause(value: str) -> str:
    return str(value or "").strip(" \t\r\n，。；")


def _table_slot(table: Dict[str, Any]) -> str:
    slot = str(table.get("placement_slot") or "").strip()
    return slot or "chapter_end"


def _table_render_priority(table: Dict[str, Any]) -> int:
    try:
        return int(table.get("render_priority") or 0)
    except (TypeError, ValueError):
        return 0


def _section_matches_table(section: Dict[str, Any], table: Dict[str, Any]) -> bool:
    section_id = str(section.get("section_id") or "").strip()
    block_type = str(section.get("block_type") or section.get("output_type") or "").strip()
    anchor_section_id = str(table.get("anchor_section_id") or "").strip()
    anchor_block_type = str(table.get("anchor_block_type") or "").strip()
    if anchor_section_id and section_id == anchor_section_id:
        return True
    if anchor_block_type and block_type == anchor_block_type:
        return True
    section_refs = {
        str(ref or "").strip()
        for ref in _as_list(section.get("evidence_refs")) + _as_list(section.get("required_evidence_refs"))
        if str(ref or "").strip()
    }
    table_refs = {str(ref or "").strip() for ref in _as_list(table.get("evidence_refs")) if str(ref or "").strip()}
    return bool(section_refs and table_refs and section_refs.intersection(table_refs))


def _slot_matches_section(slot: str, section: Dict[str, Any]) -> bool:
    block_type = str(section.get("block_type") or section.get("output_type") or "").strip()
    if slot == "after_thesis":
        return block_type == "thesis"
    if slot == "after_evidence_matrix":
        return block_type in {"evidence_matrix", "metric_reconciliation", "competitive_positioning", "case_comparison"}
    if slot == "after_mechanism":
        return block_type in {"mechanism_chain", "technology_maturity", "value_chain_map", "policy_timeline", "stakeholder_map"}
    if slot == "before_risk":
        return block_type == "risk_trigger"
    if slot == "before_decision":
        return block_type in {"verification_checklist", "scenario_analysis"}
    return False


def _compact_chapter_heading(value: Any, *, max_chars: int = 28) -> str:
    text = _public_text(value, 180).strip(" ？?！!。.；;")
    if not text:
        return ""
    text = (
        text.replace("商业化证据", "商业化信号")
        .replace("事实依据", "事实信号")
        .replace("判断依据", "判断信号")
    )
    for marker in (
        "是否存在真实需求",
        "哪些环节已",
        "竞争格局",
        "需求变化",
        "主要玩家",
        "技术路线",
        "技术、供应、监管",
    ):
        idx = text.find(marker)
        if idx > 0:
            text = text[idx:]
            break
    text = re.sub(r"^是否", "", text)
    text = re.sub(r"^的", "", text)
    if len(text) > max_chars:
        head = re.split(r"[，,；;：:？?]", text, 1)[0].strip()
        if 4 <= len(head) <= max_chars:
            text = head
    return _compact(text, max_chars).strip(" ？?！!。.；;")


def render_chapter_package(
    chapter: Dict[str, Any],
    index: int,
    *,
    previous_chapter: Dict[str, Any] | None = None,
    next_chapter: Dict[str, Any] | None = None,
) -> str:
    if chapter.get("omit_from_report") or chapter.get("chapter_omitted_no_evidence"):
        return ""
    if not _as_list(chapter.get("sections")) and not _as_list(chapter.get("table_packages")) and not chapter.get("lead"):
        return ""
    title = _compact_chapter_heading(chapter.get("chapter_title") or f"章节 {index}")
    if not title:
        title = f"章节 {index}"
    lines = [f"## {index}. {title}"]
    lead = _clean_render_text(chapter.get("lead"), 360)
    flow_intro = _chapter_flow_intro(chapter, index=index, previous_chapter=previous_chapter)
    if flow_intro:
        lines.append(flow_intro)
    if lead and lead != flow_intro:
        lines.append(lead)
        _append_citation_to_last_paragraph(lines, _first_section_citation_refs(chapter))
    chapter_tables = [
        {"table": table, "rendered": rendered}
        for table in _as_list(chapter.get("table_packages"))
        if isinstance(table, dict)
        for rendered in [render_table_package(table)]
        if rendered
    ]
    chapter_tables.sort(key=lambda item: (-_table_render_priority(_as_dict(item.get("table"))), str(_as_dict(item.get("table")).get("table_id") or "")))
    placed_tables: set[str] = set()

    def append_tables(slot: str, *, section: Dict[str, Any] | None = None, limit: int = 1) -> None:
        appended = 0
        for item in chapter_tables:
            table = _as_dict(item.get("table"))
            table_id = str(table.get("table_id") or id(table))
            if table_id in placed_tables:
                continue
            table_slot = _table_slot(table)
            if table_slot != slot and not (section and _section_matches_table(section, table) and _slot_matches_section(slot, section)):
                continue
            if section and not (_section_matches_table(section, table) or _slot_matches_section(table_slot, section)):
                continue
            lines.extend(["", str(item.get("rendered") or "")])
            placed_tables.add(table_id)
            appended += 1
            if appended >= limit:
                return

    seen_section_titles: set[str] = set()
    for section_index, section in enumerate(_as_list(chapter.get("sections")), start=1):
        section = _as_dict(section)
        if _section_should_skip(section):
            continue
        unique_title = _unique_section_title(section, seen_section_titles, section_index=section_index)
        if unique_title:
            section = dict(section)
            section["section_title"] = unique_title
            section["dynamic_section_title"] = unique_title
        if _slot_matches_section("before_risk", section):
            append_tables("before_risk", section=section, limit=1)
        if _slot_matches_section("before_decision", section):
            append_tables("before_decision", section=section, limit=1)
        if _as_list(section.get("render_blocks")):
            rendered_section = render_section(section)
            if rendered_section:
                _append_citation_to_last_paragraph(rendered_section, _as_list(section.get("citation_refs")) or _as_list(section.get("evidence_refs")))
                lines.append("")
                lines.extend(rendered_section)
                for slot in ("after_thesis", "after_evidence_matrix", "after_mechanism"):
                    if _slot_matches_section(slot, section):
                        append_tables(slot, section=section, limit=1)
                        break
            continue
        section_title = _section_public_title(section)
        if section_title and not _is_internal_section_title(section_title):
            lines.extend(["", f"### {section_title}"])
        claim = _clean_render_text(section.get("claim"), _env_int("REPORT_SECTION_CLAIM_MAX_CHARS", 1100, min_value=300, max_value=3000))
        reasoning = _clean_render_text(section.get("reasoning"), _env_int("REPORT_SECTION_REASONING_MAX_CHARS", 3200, min_value=800, max_value=8000))
        counter = _clean_render_text(section.get("counter_evidence"), _env_int("REPORT_SECTION_COUNTER_MAX_CHARS", 1600, min_value=400, max_value=5000))
        actionable = _clean_render_text(section.get("actionable"), _env_int("REPORT_SECTION_ACTION_MAX_CHARS", 1600, min_value=400, max_value=5000))
        mechanism = _clean_render_text(section.get("mechanism") or section.get("reasoning"), _env_int("REPORT_SECTION_MECHANISM_MAX_CHARS", 2400, min_value=600, max_value=6000))
        decision_implication = _clean_render_text(section.get("decision_implication") or actionable, _env_int("REPORT_SECTION_ACTION_MAX_CHARS", 1600, min_value=400, max_value=5000))
        if _looks_like_bare_metric_text(claim):
            claim = _metric_sentence_from_block(section, claim)
        if _looks_like_bare_metric_text(reasoning):
            reasoning = _metric_sentence_from_block(section, reasoning)
        if _looks_like_bare_metric_text(mechanism):
            mechanism = _metric_sentence_from_block(section, mechanism)
        before_section_len = len(lines)
        if section.get("observation_only") and section.get("force_render_observation") and not section.get("evidence_backed"):
            if claim:
                lines.extend(_paragraph_chunks(claim, max_chars=360, max_chunks=1))
            if len(lines) > before_section_len:
                _append_citation_to_last_paragraph(lines, _as_list(section.get("citation_refs")) or _as_list(section.get("evidence_refs")))
            continue
        if claim:
            lines.extend(_paragraph_chunks(claim, max_chars=700, max_chunks=2))
        if reasoning:
            lines.extend(_paragraph_chunks(reasoning, max_chars=760, max_chunks=3))
        if mechanism and mechanism != reasoning:
            lines.extend(_paragraph_chunks(mechanism, max_chars=700, max_chunks=2))
        elif mechanism and not reasoning:
            lines.extend(_paragraph_chunks(mechanism, max_chars=700, max_chunks=2))
        if counter:
            lines.extend(_paragraph_chunks(_natural_transition("同时，", counter), max_chars=680, max_chunks=2))
        if decision_implication:
            lines.extend(_paragraph_chunks(_natural_transition("落到行业含义上，", decision_implication), max_chars=680, max_chunks=2))
        if len(lines) > before_section_len:
            _append_citation_to_last_paragraph(lines, _as_list(section.get("citation_refs")) or _as_list(section.get("evidence_refs")))
            for slot in ("after_thesis", "after_evidence_matrix", "after_mechanism"):
                if _slot_matches_section(slot, section):
                    append_tables(slot, section=section, limit=1)
                    break
    fact_digest = render_chapter_fact_digest(chapter)
    if fact_digest:
        lines.append("")
        lines.extend(fact_digest)
    deep_synthesis = render_chapter_deep_synthesis(chapter)
    if deep_synthesis:
        lines.extend(deep_synthesis)
    append_tables("chapter_end", limit=2)
    for item in chapter_tables:
        table = _as_dict(item.get("table"))
        table_id = str(table.get("table_id") or id(table))
        if table_id not in placed_tables:
            lines.extend(["", str(item.get("rendered") or "")])
            placed_tables.add(table_id)
    transition = _chapter_transition(chapter, next_chapter)
    if transition:
        lines.extend(["", transition])
    return "\n".join(_dedupe_narrative_lines(lines))


def render_final_reference_analysis(decision_package: Dict[str, Any]) -> List[str]:
    if not _env_flag("REPORT_RENDER_FINAL_REFERENCE_ANALYSIS", False):
        return []
    syntheses = [_as_dict(item) for item in _as_list(decision_package.get("chapter_syntheses"))]
    rows: List[str] = []
    visible_syntheses = syntheses[:8]
    for position, item in enumerate(visible_syntheses):
        title = _public_text(item.get("chapter_title"), 140)
        question = _public_text(item.get("chapter_question"), 180)
        prev_title = _public_text(_as_dict(visible_syntheses[position - 1]).get("chapter_title"), 140) if position > 0 else ""
        next_title = _public_text(_as_dict(visible_syntheses[position + 1]).get("chapter_title"), 140) if position + 1 < len(visible_syntheses) else ""
        summary = _as_dict(item.get("chapter_summary"))
        takeaway = _public_text(summary.get("key_takeaway"), 780)
        mechanisms = _dedupe([_clause(_public_text(value, 620)) for value in _as_list(summary.get("mechanisms"))], limit=3, max_chars=520)
        counters = _dedupe([_clause(_public_text(value, 620)) for value in _as_list(summary.get("counter_evidence"))], limit=3, max_chars=520)
        actions = _dedupe([_clause(_final_action_phrase(_public_text(value, 560))) for value in _as_list(summary.get("next_actions"))], limit=4, max_chars=460)
        watch = _dedupe([_clause(_public_text(value, 340)) for value in _as_list(summary.get("what_to_verify_next"))], limit=5, max_chars=320)
        if not any([takeaway, mechanisms, counters, actions, watch]):
            continue

        heading = title or question or "相关章节"
        paragraph_parts: List[str] = []
        if takeaway:
            paragraph_parts.append(
                f"“{heading}”对应的章节结论是：{takeaway}"
                "它的权重取决于能否和其他章节里的关键对象、指标、约束和反向样本互相解释。"
            )
        if mechanisms:
            paragraph_parts.append(
                "影响路径可以概括为："
                + "；".join(mechanisms)
                + "。这些环节衔接得越完整，章节结论越能进入全篇判断。"
            )
        if counters:
            paragraph_parts.append(
                "结论弹性主要来自这些反向条件："
                + "；".join(counters)
                + "。这些条件决定了结论在什么情况下增强、收缩或被重新解释。"
            )
        if actions:
            paragraph_parts.append(
                "资源配置会集中到："
                + "；".join(actions)
                + "。这样可以避免被孤立新闻、短期波动或单一案例牵引。"
            )
        if watch:
            paragraph_parts.append(
                "后续变化主要集中在："
                + "；".join(watch)
                + "。这些变量一旦变化，整篇报告的强弱排序和行动优先级也会随之变化。"
            )
        if prev_title or next_title:
            relation_parts = []
            if prev_title:
                relation_parts.append(f"它承接“{prev_title}”留下的变量条件")
            if next_title:
                relation_parts.append(f"也为“{next_title}”提供判断基础")
            paragraph_parts.append("放在章节顺序中看，" + "，".join(relation_parts) + "。")
        rows.append("".join(paragraph_parts))

    if not rows:
        return []
    chapter_titles = [_public_text(item.get("chapter_title"), 120) for item in visible_syntheses]
    chapter_chain = "、".join(title for title in chapter_titles if title)
    closing_lines = [
        "综合来看，最终结论的强弱取决于章节之间是否能相互解释：单个事实只能提供观察，多组事实在口径、时间和对象上相互支撑时，才适合进入更明确的行动判断。",
        "后续更新也应沿着同一顺序进行：先补足关键事实，再校准口径差异，最后观察反向条件是否出现。这样新增材料不会打散原有结构，而是能回到对应章节重新排序。"
    ]
    if chapter_chain:
        closing_lines.insert(
            0,
            f"从章节排列看，{chapter_chain}不是并列清单，而是一组逐步收束的判断。每一章承担什么位置，应由研究问题和证据决定。",
        )
    return [
        "",
        "### 全篇收束",
        *rows,
        *closing_lines,
    ]


def render_decision_package(decision_package: Dict[str, Any]) -> str:
    lines: List[str] = []
    thesis = _public_text(decision_package.get("decision_thesis"), 1200)
    if thesis:
        lines.extend(["## 综合判断与策略含义", thesis])
    recommendations = [_as_dict(item) for item in _as_list(decision_package.get("recommendations"))]
    public_recs = []
    for item in recommendations[:5]:
        text = _public_text(item.get("recommendation"), 700)
        label = _compact(item.get("label"), 40)
        if text:
            public_recs.append(f"- {label + '：' if label else ''}{text}")
    if public_recs:
        if lines:
            lines.extend(["", "### 结论如何落到动作", *public_recs])
        else:
            lines.extend(["## 综合判断与策略含义", *public_recs])
    scenarios = [_as_dict(item) for item in _as_list(decision_package.get("scenario_analysis"))]
    public_scenarios = []
    for item in scenarios[:3]:
        scenario = _public_text(item.get("scenario"), 80)
        condition = _public_text(item.get("condition"), 700)
        implication = _public_text(item.get("implication"), 700)
        if scenario and (condition or implication):
            public_scenarios.append(f"- **{scenario}**：{condition} {implication}".strip())
    if public_scenarios:
        if lines:
            lines.extend(["", "### 情景分层与结论弹性", *public_scenarios])
        else:
            lines.extend(["## 综合判断与策略含义", "### 情景分层与结论弹性", *public_scenarios])
    reference_lines = render_final_reference_analysis(decision_package)
    if reference_lines and lines:
        lines.extend(reference_lines)
    watchlist = [_as_dict(item) for item in _as_list(decision_package.get("watchlist"))]
    public_watch = []
    for item in watchlist[:5]:
        metric = _public_text(item.get("metric"), 520)
        if metric:
            public_watch.append(f"- {metric}")
    if public_watch and lines:
        lines.extend(["", "### 观察指标", *public_watch])
    abandon = [_as_dict(item) for item in _as_list(decision_package.get("abandon_conditions"))]
    public_abandon = []
    for item in abandon[:3]:
        condition = _public_text(item.get("condition"), 520)
        if condition:
            public_abandon.append(f"- {condition}")
    if public_abandon and lines:
        lines.extend(["", "### 放弃条件", *public_abandon])
    return "\n".join(lines)


def render_risk_package(risk_package: Dict[str, Any]) -> str:
    rows = []
    for item in _as_list(risk_package.get("risk_items"))[:8]:
        item = _as_dict(item)
        description = _public_text(item.get("description"), 220)
        if not description:
            continue
        risk_type = _compact(item.get("risk_type"), 80)
        mitigation = _public_text(item.get("mitigation"), 220)
        severity = _compact(item.get("severity"), 30)
        prefix = f"{risk_type}（{severity}）" if risk_type and severity else (risk_type or "风险事项")
        rows.append(f"- {prefix}：{description}")
        if mitigation:
            rows.append(f"  应对：{mitigation}")
    if not rows:
        return ""
    return "\n".join(["## 反向信号与风险触发", *rows])


def _render_public_table_appendix_rows(appendix_package: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    appendix_tables = [
        _as_dict(item)
        for item in _as_list(_as_dict(appendix_package).get("table_appendix_rows"))
        if isinstance(item, dict)
        and item.get("should_render") is not False
        and _table_validation_passed(item)
    ]
    for table in appendix_tables[:6]:
        headers = _as_list(table.get("headers"))
        rows = [_as_list(row) for row in _as_list(table.get("rows"))[:12]]
        headers, rows = _public_table_shape(headers, rows)
        if len(headers) < 2 or len(rows) < 2:
            continue
        table_md = _markdown_table(headers, rows)
        if not table_md:
            continue
        title = _compact(table.get("title") or "表格附录明细", 90)
        lines.extend(["", f"### {title}（附录明细）", table_md])
    return lines


def render_appendix(source_registry: Sequence[Dict[str, Any]], appendix_package: Dict[str, Any]) -> str:
    if not _env_flag("REPORT_RENDER_DIAGNOSTIC_APPENDIX_TABLES", False):
        if not source_registry:
            table_lines = _render_public_table_appendix_rows(appendix_package)
            if not table_lines:
                return ""
            return "\n".join(["## 来源附录", *table_lines])
        lines = ["## 来源附录"]
        for source in list(source_registry)[:50]:
            source = _as_dict(source)
            ref = str(source.get("ref") or "").strip()
            title = str(source.get("title") or "未命名来源").strip()
            date = str(source.get("date") or "").strip()
            url = str(source.get("url") or "").strip()
            suffix = " | ".join(part for part in [date, url] if part)
            lines.append(f"- {ref} {title}" + (f" | {suffix}" if suffix else ""))
        return "\n".join(lines)

    lines = ["## 来源附录"]
    scope = _public_text(_as_dict(appendix_package).get("scope_note"), 260)
    if scope:
        lines.append(scope)
    coverage_rows = [_as_dict(item) for item in _as_list(_as_dict(appendix_package).get("coverage_matrix")) if isinstance(item, dict)]
    if coverage_rows and _env_flag("REPORT_RENDER_COVERAGE_MATRIX", False):
        rows = []
        for item in coverage_rows[:12]:
            rows.append(
                [
                    _compact(item.get("hypothesis_statement") or item.get("hypothesis_id"), 80),
                    f"{item.get('actual_ab_sources') or 0}/{item.get('required_ab_sources') or 0}",
                    str(item.get("counter_evidence_count") or 0),
                    f"{item.get('complete_metric_count') or 0}/{item.get('metric_count') or 0}",
                    "是" if item.get("decision_ready") else "否",
                    "、".join(_dedupe(_as_list(item.get("blocking_gaps")), limit=3)),
                ]
            )
        table_md = _markdown_table(["假设", "A/B来源", "反证", "指标口径", "可下判断", "待补"], rows)
        if table_md:
            lines.extend(["", "### 证据覆盖矩阵", table_md])
    metric_rows = [
        _as_dict(item)
        for item in _as_list(_as_dict(appendix_package).get("metric_normalization_table"))
        if isinstance(item, dict) and not _invalid_metric_appendix_row(_as_dict(item))
    ]
    if metric_rows:
        rows = []
        for item in metric_rows[:12]:
            rows.append(
                [
                    _compact(item.get("metric_name"), 60),
                    _compact(item.get("subject"), 70),
                    _compact(item.get("scope"), 50),
                    _compact(item.get("period"), 40),
                    _compact(item.get("unit"), 24),
                    _compact(item.get("value"), 50),
                    _compact(item.get("source_level"), 20),
                ]
            )
        table_md = _markdown_table(["指标", "主体", "范围", "期间", "单位", "值", "来源等级"], rows)
        if table_md:
            lines.extend(["", "### 指标口径表", table_md])
    lines.extend(_render_public_table_appendix_rows(appendix_package))
    if not source_registry:
        return "\n".join(lines) if len(lines) > 1 else ""
    lines.extend(["", "## 来源附录"])
    for source in list(source_registry)[:50]:
        source = _as_dict(source)
        ref = str(source.get("ref") or "").strip()
        title = str(source.get("title") or "未命名来源").strip()
        date = str(source.get("date") or "").strip()
        url = str(source.get("url") or "").strip()
        suffix = " | ".join(part for part in [date, url] if part)
        lines.append(f"- {ref} {title}" + (f" | {suffix}" if suffix else ""))
    return "\n".join(lines)


def collect_format_warnings(markdown: str) -> List[str]:
    warnings: List[str] = []
    if not markdown.strip():
        warnings.append("report_markdown_empty")
    if len(re.findall(r"^# ", markdown, flags=re.M)) > 1:
        warnings.append("multiple_h1")
    if re.search(r"\n{4,}", markdown):
        warnings.append("excessive_blank_lines")
    return warnings
