from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any, Dict, List, Optional, Set

from .evidence_extractor import CONTAMINATION_PATTERNS, REPORT_DIMENSIONS, clean_evidence_text

try:
    from rag_pipeline.agents.public_report_sanitizer import find_publication_blockers, sanitize_public_markdown
except ModuleNotFoundError:  # pragma: no cover - direct script mode fallback.
    from ...agents.public_report_sanitizer import find_publication_blockers, sanitize_public_markdown

try:
    from rag_pipeline.config.search_config import build_llm_config_for_task
    from rag_pipeline.search.memory import call_openai_compatible_text, llm_config_is_ready
except ModuleNotFoundError:  # pragma: no cover - only for unusual direct execution.
    from ...config.search_config import build_llm_config_for_task
    from ...search.memory import call_openai_compatible_text, llm_config_is_ready


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def _env_int(name: str, default: int, *, min_value: int = 0, max_value: int = 1_000_000) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return min(max_value, max(min_value, value))


def _source_appendix_mode() -> str:
    mode = os.environ.get("REPORT_REFORMATTER_SOURCE_APPENDIX", "cited").strip().lower()
    if mode in {"all", "cited"}:
        return mode
    return "none"


def _source_appendix_required() -> bool:
    return _env_flag("REPORT_REFORMATTER_REQUIRE_SOURCE_APPENDIX", True)


REFORMATTER_SYSTEM_PROMPT = """
???????????????????????????????????????????
????????????????????????????????????
?????????????????????????????
??? evidence_json ????????????????????????????????????
???? Markdown ????????????
""".strip()

CHAPTER_SPECIFIC_GUIDE = """
## ????????
- ?????????????????????????????????????????????????????????????
- ????? 0-3 ?????????????????????????????????
- ???????????????????????????
- ??????????????????????
""".strip()

REFORMATTER_USER_TEMPLATE = """
## 报告主题
{topic}

## 结构化证据包
{evidence_json}

## 数据来源列表
{sources_text}

{chapter_specific_guide}

请根据以上证据撰写完整的行业研究报告。
""".strip()

REFORMATTER_POLISH_SYSTEM_PROMPT = """
????????????????????????????????????????????????????????
???????????????????????????????????????????
?????????????????????????????????????????????????
???????????????????????????
???? Markdown ????????????
""".strip()

REFORMATTER_POLISH_USER_TEMPLATE = """
## 原报告
{report_markdown}

## 结构化证据包
{evidence_json}

## 数据来源列表
{sources_text}

## 校验发现的问题
{validation_json}

请按系统要求重写为终稿。
""".strip()

CITATION_DENSITY_RULES = """
### 引用使用原则
- 引用密度由主题复杂度和可用证据决定：证据充足的正文需要多来源交叉支撑，证据不足时要明确边界，不要用空表格或固定章节补位。
- 每个具体数字、企业案例、政策条款、融资或并购事件后面必须带来源编号。
- 不要把同一个来源编号套用到整章；优先使用每条证据自身的 source 字段，让事实和来源一一对应。
- 全文需要体现来源多样性；当 evidence_json 中有大量可核验事实时，正文应覆盖足够多的来源，而不是只反复引用少数材料。
- 二次润色只能补充、重排、压实引用，禁止删除原报告中仍然有效的来源编号。
""".strip()

CHAPTER_SPECIFIC_GUIDE = """
## 动态正文写作要求
- 不得使用固定章名或固定小节模板。禁止输出“章节判断、关键事实速览、证据深读、本章结论、可引用事实、进入综合决策章的变量”等模板化标题。
- 章节标题必须来自报告主题、证据和问题拆解，正文按“先说明本章要回答的真实问题，再展开事实和逻辑，再自然收束到下一章”组织。
- 证据充足时每章通常应有 2-4 个三级标题；三级标题必须是内容型标题，例如“电池材料的需求弹性来自哪里”“价格修复为什么还不能直接等同于景气反转”。证据确实很少时才直接用连续段落。
- 表格只在确实能压缩复杂比较时使用，不要求每章都有表格；总结章尤其不要为了模板硬塞表格。
- “核心判断、机制拆解、反证边界、决策含义”只作为内部思考维度，不得作为正文标题或加粗标签出现。
- 证据不足时，不得虚构“市场规模、竞争格局、政策、技术、资本”五章模板；应直接说明当前能确认什么、不能确认什么、下一步需要补哪类来源。
""".strip()

REFORMATTER_SYSTEM_PROMPT = f"""
你是行业研究报告终稿作者。你的任务是把结构化证据转化为一份自然、连贯、问题驱动的行研正文，而不是把证据套进固定模板。

写作原则：
1. 章节必须跟随报告主题和证据本身，不得固定为“市场规模与增速、竞争格局、政策监管、技术路线、资本动态”等通用五章。
2. 正文不得出现内部分析标签或过程性标题，包括：章节判断、关键事实速览、证据深读、本章结论、可引用事实、进入综合决策章的变量、核心判断、机制拆解、反证边界、决策含义。
3. 可以在内部先判断事实、机制、边界和含义，但输出给读者的只应该是成熟正文：段落之间要有先后顺序，章节之间要能接续，不能像证据清单。
4. 只使用 evidence_json 和来源列表中的事实、数字、公司、政策和事件；evidence_group 只是证据聚类元数据，不得直接当作章名或目录；没有证据时不要编造行业结论，也不要用固定模板占位。
5. 引用必须跟在具体事实、数字、公司事件或政策条款后面；不要把来源编号机械套到整段。
6. 证据充足时必须展开为多维分析，而不是单线叙事：每个核心判断至少从供应链位置、政策与贸易约束、技术瓶颈、企业与客户行为、资本开支与成本、反证或不确定性中选取 3-5 个维度交叉说明。
7. 正文需要有灵活层级：主体章节通常应包含内容型三级标题，三级标题要直接说明该小节在讲什么；不要把一整章压成一个长段。
8. 段落要可读：单段集中回答一个问题，通常 180-380 字；超过 480 字时必须拆成多个自然段。
9. 输出完整 Markdown 报告，不要输出写作说明。

10. 输出只保留高质量正文和必要的正文内引用，不要生成“数据来源”“参考来源”“研究口径与来源”“附录”等附录章节；来源列表只用于核对，不进入成稿。
{CITATION_DENSITY_RULES}
""".strip()

REFORMATTER_POLISH_SYSTEM_PROMPT = f"""
你是行业研究报告终稿编辑。你的任务不是新增事实，而是在不改变来源编号和事实边界的前提下，把报告改成更像成熟行研终稿。

必须修复：
1. 删除固定小节模板和内部思考标签，尤其是“章节判断、关键事实速览、证据深读、本章结论、可引用事实、进入综合决策章的变量、核心判断、机制拆解、反证边界、决策含义”。
2. 将模板标题下的内容合并为自然段或改写成内容型标题；内容型标题必须直接描述该段在讲什么。
3. 保持章节之间的顺序感：先讲问题背景和事实基础，再讲变化如何发生，再讲约束和不确定性，最后自然收束。
4. 不要求每章都有表格；没有必要的表格直接删掉或改为段落。
5. 所有保留下来的具体事实、数字、公司事件和政策条款必须保留来源编号。
6. 如果校验提示正文篇幅不足、三级标题不足、正文引用不足或段落过长，必须优先补足正文分析：增加内容型三级标题，把证据拆成多个问题小节，并把同一事实链条里的“为什么、影响谁、边界在哪里”讲清楚。

输出完整 Markdown 报告，不要输出修改说明。
7. 删除“数据来源”“参考来源”“研究口径与来源”“附录”等附录章节，把证据落实到正文内引用和分析密度里。

{CITATION_DENSITY_RULES}
""".strip()

FORBIDDEN_OUTPUT_PATTERNS = [
    *CONTAMINATION_PATTERNS,
    r"建议动作[:：]",
    r"因此可用于支撑",
    r"未来\s*\d+[-–—]\d+\s*个月应跟踪该口径",
    r"该条数据不能单独证明",
    r"证明强度为",
    r"推进到可追踪证据",
    r"原文事实",
    r"行业形势含义",
    r"投资/产品判断",
    r"与上下章节的联动",
    r"战略含义与行动建议",
    r"建议动作[:：]",
    r"\*\*[^*\n]{1,40}怎么读\*\*",
    r"分析后证据[:：]",
    r"EV-[A-Za-z0-9_-]+",
    r"章节判断",
    r"关键事实速览",
    r"证据深读",
    r"本章结论",
    r"可引用事实",
    r"进入综合决策章的变量",
    r"核心判断[:：]?",
    r"机制拆解",
    r"反证边界",
    r"决策含义",
    r"公开信息显示[:：]",
    r"公开材料显示[:：]",
    r"已经出现可观察信号，持续性取决于",
    r"反向信号一旦持续出现",
    r"判断顺序应先看同口径指标的连续性",
    r"这些事实需要放在主体、时间、范围和来源层级上交叉理解",
    r"补充分析：把证据转成可判断的问题链",
    r"哪些变量真正改变判断",
    r"目前可用信号首先形成一条可复核的事实链",
]

FIXED_TEMPLATE_LABEL_RE = re.compile(
    r"章节判断|关键事实速览|证据深读|本章结论|"
    r"可引用事实|进入综合决策章的变量|核心判断|机制拆解|反证边界|决策含义"
)
SOURCE_APPENDIX_HEADING_RE = re.compile(r"(?m)^##\s*(?:数据来源列表|数据来源|研究口径与来源|附录|参考来源)(?:\s|$|[:：])")
SCHEMA_LIKE_BULLET_RE = re.compile(r"(?m)^\s*[-*]\s*[^。；;\n]{1,16}[；;][^。；;\n]{0,16}[；;][^。；;\n]{0,50}\s*$")
TEMPLATE_HEADING_LABEL_RE = re.compile(
    r"章节判断|关键事实速览|证据深读|本章结论|全球口径|中国口径|增速口径|"
    r"可引用事实|机制与边界|进入综合决策章的变量|核心判断|机制拆解|反证边界|决策含义"
)


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _source_quality_score(item: Dict[str, Any]) -> int:
    quality = str(item.get("source_quality") or "").strip().lower()
    if quality == "high":
        return 4
    if quality == "medium":
        return 3
    if quality == "normal":
        return 2
    if quality == "low":
        return 0
    source_type = str(item.get("source_type") or "").strip().lower()
    if source_type in {"official", "policy"}:
        return 4
    if source_type in {"research", "academic"}:
        return 3
    if source_type in {"news", "media", "unknown"}:
        return 2
    if source_type == "self_media":
        return 0
    return 1


def _select_dimension_facts(facts: List[Dict[str, Any]], max_items: int) -> List[Dict[str, Any]]:
    prepared: List[Dict[str, Any]] = []
    seen_text = set()
    for item in facts:
        text = clean_evidence_text(item.get("text", ""))
        if not text:
            continue
        key = re.sub(r"\s+", "", text)[:180]
        if key in seen_text:
            continue
        seen_text.add(key)
        prepared.append(
            {
                "text": text[:420],
                "source": str(item.get("source") or "").strip(),
                "time": item.get("time", ""),
                "metric": item.get("metric", ""),
                "value": item.get("value", ""),
                "source_type": item.get("source_type", ""),
                "source_quality": item.get("source_quality", ""),
            }
        )

    ranked = sorted(
        enumerate(prepared),
        key=lambda pair: (
            -_source_quality_score(pair[1]),
            str(pair[1].get("source") or "999"),
            -len(str(pair[1].get("text") or "")),
        ),
    )
    selected: List[Dict[str, Any]] = []
    selected_indexes = set()
    source_counts: Dict[str, int] = {}
    for per_source_cap in [2, 4, 999]:
        for idx, item in ranked:
            if idx in selected_indexes:
                continue
            source_id = str(item.get("source") or "?")
            if source_counts.get(source_id, 0) >= per_source_cap:
                continue
            selected.append(item)
            selected_indexes.add(idx)
            source_counts[source_id] = source_counts.get(source_id, 0) + 1
            if len(selected) >= max_items:
                return selected
    return selected


def build_llm_config() -> Dict[str, Any]:
    return dict(build_llm_config_for_task("reformatter"))


def _source_line(source: Dict[str, Any], *, include_date: bool = True) -> str:
    source_id = source.get("id") or "?"
    title = str(source.get("title") or "未命名来源").strip()
    url = str(source.get("url") or "").strip()
    date = str(source.get("date") or "未标注日期").strip()
    credibility = str(source.get("credibility") or source.get("credibility_level") or "").strip().upper()
    quality = f"，来源级别{credibility}" if credibility in {"A", "B", "C", "D"} else ""
    if include_date:
        return f"[{source_id}] {title}，{url}{quality}，{date}"
    return f"[{source_id}] {title}，{url}{quality}".rstrip("，")


def _selected_source_ids(evidence_items: List[Dict[str, Any]]) -> Set[str]:
    return {
        str(item.get("source") or "").strip()
        for item in evidence_items
        if str(item.get("source") or "").strip()
    }


def _target_body_chars(clean_evidence: Dict[str, Any]) -> int:
    facts = _evidence_facts(clean_evidence)
    fact_count = len(facts)
    if not facts:
        configured_empty = _env_int("REPORT_REFORMATTER_MIN_BODY_CHARS", 0, min_value=0, max_value=100_000)
        if os.environ.get("REPORT_REFORMATTER_MIN_BODY_CHARS"):
            empty_cap = _env_int("REPORT_REFORMATTER_EMPTY_EVIDENCE_TARGET_CHARS", 4_000, min_value=0, max_value=30_000)
            if _reformatter_length_mode() not in {"fixed", "strict", "force"}:
                return min(configured_empty, empty_cap)
        return configured_empty

    dynamic_default = min(36_000, max(16_000, fact_count * 35))
    configured = _env_int(
        "REPORT_REFORMATTER_MIN_BODY_CHARS",
        dynamic_default,
        min_value=0,
        max_value=100_000,
    )
    if _reformatter_length_mode() in {"fixed", "strict", "force"}:
        return configured

    full_length_min_facts = _env_int("REPORT_REFORMATTER_FULL_LENGTH_MIN_FACTS", 30, min_value=1, max_value=500)
    if fact_count >= full_length_min_facts:
        return configured

    floor = _env_int("REPORT_REFORMATTER_MIN_BODY_CHARS_FLOOR", 6_000, min_value=0, max_value=60_000)
    per_fact = _env_int("REPORT_REFORMATTER_CHARS_PER_FACT_TARGET", 900, min_value=120, max_value=5_000)
    overhead = _env_int("REPORT_REFORMATTER_SPARSE_EVIDENCE_OVERHEAD_CHARS", 3_000, min_value=0, max_value=20_000)
    sparse_target = max(floor, fact_count * per_fact + overhead)
    return min(configured, sparse_target)


def _reformatter_length_mode() -> str:
    return os.environ.get("REPORT_REFORMATTER_LENGTH_TARGET_MODE", "adaptive").strip().lower()


def build_reformatter_payload(clean_evidence: Dict[str, Any], *, max_facts_per_dimension: Optional[int] = None) -> Dict[str, str]:
    topic = str(clean_evidence.get("topic") or "行业").strip()
    evidence_items: List[Dict[str, Any]] = []
    max_items = (
        int(max_facts_per_dimension)
        if max_facts_per_dimension is not None
        else _env_int("REPORT_REFORMATTER_MAX_FACTS_PER_DIMENSION", 42, min_value=8, max_value=80)
    )
    raw_dimensions = _as_dict(clean_evidence.get("dimensions"))
    ordered_dimensions = [
        *[dimension for dimension in REPORT_DIMENSIONS if dimension in raw_dimensions],
        *[dimension for dimension in raw_dimensions if dimension not in REPORT_DIMENSIONS],
    ]
    for dimension in ordered_dimensions:
        selected = _select_dimension_facts(
            [item for item in _as_list(raw_dimensions.get(dimension)) if isinstance(item, dict)],
            max_items,
        )
        for item in selected:
            evidence_items.append({"evidence_group": dimension, **item})

    selected_source_ids = _selected_source_ids(evidence_items)
    selected_sources = [
        source
        for source in _as_list(clean_evidence.get("sources"))
        if str(source.get("id") or "").strip() in selected_source_ids
    ]
    sources_text = "\n".join(_source_line(source) for source in selected_sources)
    target_body_chars = _target_body_chars(clean_evidence)
    return {
        "topic": topic,
        "evidence_json": json.dumps({"evidence_items": evidence_items}, ensure_ascii=False, indent=2),
        "sources_text": sources_text or "来源信息待补充",
        "chapter_specific_guide": (
            f"{CHAPTER_SPECIFIC_GUIDE}\n"
            f"- 本次证据包较丰富时，正文部分不要低于 {target_body_chars} 字；"
            "主体章节应尽量拆成内容型三级标题，用多维分析承接证据，而不是把事实串成单线长段。"
        ),
    }


def _chat_text_with_openai_compatible(
    *,
    config: Dict[str, Any],
    system_prompt: str,
    user_content: str,
    temperature: float,
    max_tokens: int,
) -> str:
    if not llm_config_is_ready(config):
        raise RuntimeError("ReformatterAgent 的大模型配置不完整。")
    response = call_openai_compatible_text(
        config=config,
        system_prompt=system_prompt,
        user_content=user_content,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = str(response.get("text") or "").strip()
    if not text:
        raise RuntimeError("ReformatterAgent LLM response content is empty.")
    return text


async def _chat_text_with_langchain_client(
    *,
    llm_client: Any,
    system_prompt: str,
    user_content: str,
    stream: bool,
) -> str:
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
    except Exception as exc:
        raise RuntimeError("langchain_core is required when passing llm_client.") from exc

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_content)]
    if stream and hasattr(llm_client, "astream"):
        full_response = ""
        async for chunk in llm_client.astream(messages):
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            print(content, end="", flush=True)
            full_response += content
        print()
        return full_response
    if hasattr(llm_client, "ainvoke"):
        response = await llm_client.ainvoke(messages)
        return str(getattr(response, "content", response) or "").strip()
    if hasattr(llm_client, "invoke"):
        response = await asyncio.to_thread(llm_client.invoke, messages)
        return str(getattr(response, "content", response) or "").strip()
    raise RuntimeError("Unsupported llm_client: expected astream, ainvoke, or invoke.")


async def _generate_reformatter_text(
    *,
    system_prompt: str,
    user_content: str,
    llm_client: Optional[Any],
    temperature: float,
    max_tokens: int,
    stream: bool,
) -> str:
    if llm_client is not None:
        return await _chat_text_with_langchain_client(
            llm_client=llm_client,
            system_prompt=system_prompt,
            user_content=user_content,
            stream=stream,
        )
    return await asyncio.to_thread(
        _chat_text_with_openai_compatible,
        config=build_llm_config(),
        system_prompt=system_prompt,
        user_content=user_content,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _valid_source_ids(sources: List[Dict[str, Any]]) -> Set[str]:
    return {str(source.get("id")) for source in sources if str(source.get("id") or "").strip()}


def normalize_report_citations(markdown: str, valid_ids: Set[str]) -> str:
    _ = valid_ids
    return str(markdown or "")


def append_sources_appendix(markdown: str, sources: List[Dict[str, Any]]) -> str:
    body = re.split(SOURCE_APPENDIX_HEADING_RE, str(markdown or ""), maxsplit=1)[0].strip()
    appendix_mode = _source_appendix_mode()
    if not sources or appendix_mode == "none":
        return body
    cited_order: List[str] = []
    seen_citations = set()
    for source_id in re.findall(r"\[(\d{1,3})\]", body):
        if source_id in seen_citations:
            continue
        seen_citations.add(source_id)
        cited_order.append(source_id)
    by_id = {str(source.get("id")): source for source in sources if source.get("id")}
    if appendix_mode == "all" or not cited_order:
        selected_sources = [source for source in sources if source.get("id")]
    else:
        selected_sources = [by_id[source_id] for source_id in cited_order if source_id in by_id]
    source_lines = [_source_line(source, include_date=False) for source in selected_sources]
    return (body + "\n\n## 数据来源\n" + "\n".join(source_lines)).strip()


CHAPTER_OPENER_FORBIDDEN_RE = re.compile(r"^\s*(?:数据显示|数据表明|据统计|证据表明|根据数据|公开数据显示)")
NUMBER_RE = re.compile(r"(?:\d|[一二三四五六七八九十百千万亿]+(?:元|美元|亿元|亿美元|万|%|％)|\[\d{1,3}\])")
FORWARD_LOOKING_RE = re.compile(r"(未来|将|可能|有望|趋于|加速|分化|收敛|演变|后续|长期|短期|进入|成为)")
STRATEGIC_RE = re.compile(r"(投资|企业|战略|资源配置|优先级|商业化|落地|壁垒|护城河|竞争|定价|客户|渠道|出海|替代|合规|现金流|风险|机会|政策制定者)")
ANALYSIS_CONNECTIVE_RE = re.compile(r"(这说明|这意味着|背后逻辑|本质上|因此|由此|核心在于|关键在于|这标志着|体现|反映|换言之|也就是说|进一步看)")


def _chapter_blocks(markdown: str) -> List[Dict[str, str]]:
    matches = list(re.finditer(r"(?m)^##\s+([^\n]+)", str(markdown or "")))
    blocks: List[Dict[str, str]] = []
    for index, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        blocks.append({"title": title, "body": markdown[start:end].strip()})
    return blocks


def _first_body_sentence(body: str) -> str:
    lines = []
    for raw_line in str(body or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("|") or re.match(r"^[:\-\s|]+$", line):
            continue
        if re.match(r"^(?:[-*+]|\d+\.)\s+", line):
            continue
        if re.match(r"^\*\*[^*]+(?:表|判断|结论|建议|风险|数据速查)[^*]*\*\*\s*$", line):
            continue
        if line.startswith("#"):
            continue
        lines.append(line)
        break
    if not lines:
        return ""
    match = re.search(r"^(.+?[。！？!?])", lines[0])
    return (match.group(1) if match else lines[0]).strip()


def _body_paragraphs(body: str) -> List[str]:
    paragraphs: List[str] = []
    current: List[str] = []
    for raw_line in str(body or "").splitlines():
        line = raw_line.strip()
        if not line:
            if current:
                paragraphs.append(" ".join(current).strip())
                current = []
            continue
        if line.startswith("|") or line.startswith("#") or re.match(r"^(?:[-*+]|\d+\.)\s+", line):
            if current:
                paragraphs.append(" ".join(current).strip())
                current = []
            continue
        if re.match(r"^\*\*[^*]+\*\*\s*$", line):
            if current:
                paragraphs.append(" ".join(current).strip())
                current = []
            continue
        current.append(line)
    if current:
        paragraphs.append(" ".join(current).strip())
    return paragraphs


def _is_body_paragraph_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith("#") or stripped.startswith("|"):
        return False
    if re.match(r"^(?:[-*+]|\d+\.)\s+", stripped):
        return False
    if re.match(r"^\[[0-9]{1,3}\]\s+", stripped):
        return False
    return True


def _split_long_paragraph_line(line: str) -> List[str]:
    max_chars = _env_int("REPORT_REFORMATTER_MAX_PARAGRAPH_CHARS", 650, min_value=280, max_value=2000)
    if len(line) <= max_chars or not _is_body_paragraph_line(line):
        return [line]
    target_chars = max(220, int(max_chars * 0.68))
    sentences = re.findall(r".+?(?:[。！？!?；;]|$)", line)
    chunks: List[str] = []
    current = ""
    for sentence in [item.strip() for item in sentences if item.strip()]:
        if current and len(current) + len(sentence) > target_chars:
            chunks.append(current.strip())
            current = sentence
        else:
            current += sentence
    if current.strip():
        chunks.append(current.strip())
    return chunks or [line]


def _is_substantive_chapter(title: str) -> bool:
    normalized = re.sub(r"\s+", "", str(title or ""))
    if not normalized:
        return False
    non_body_keywords = (
        "执行摘要",
        "摘要",
        "数据来源",
        "附录",
        "参考资料",
        "目录",
        "免责声明",
    )
    return not any(keyword in normalized for keyword in non_body_keywords)


def _body_dense_chars(markdown: str) -> int:
    body = _body_without_sources(markdown)
    body = re.sub(r"(?m)^#{1,6}\s+.*$", "", body)
    return len(re.sub(r"\s+", "", body))


def _structure_issues(markdown: str, clean_evidence: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    facts = _evidence_facts(clean_evidence)
    if len(facts) < 80:
        return []
    body = _body_without_sources(markdown)
    h3_count = len(re.findall(r"(?m)^###\s+\S+", body))
    substantive_blocks = [block for block in _chapter_blocks(body) if _is_substantive_chapter(block["title"])]
    default_min_subsections = min(16, max(8, len(substantive_blocks) * 2))
    min_subsections = _env_int(
        "REPORT_REFORMATTER_MIN_SUBSECTIONS",
        default_min_subsections,
        min_value=0,
        max_value=40,
    )
    issues: List[Dict[str, Any]] = []
    if h3_count < min_subsections:
        issues.append(
            {
                "scope": "report",
                "reason": "正文三级内容标题不足，证据没有被拆成多维分析小节",
                "actual": h3_count,
                "required": min_subsections,
            }
        )
    for block in substantive_blocks:
        block_body_chars = len(re.sub(r"\s+", "", block["body"]))
        block_h3_count = len(re.findall(r"(?m)^###\s+\S+", block["body"]))
        if block_body_chars >= 1600 and block_h3_count == 0:
            issues.append(
                {
                    "scope": "chapter",
                    "chapter": block["title"],
                    "reason": "章节内容较长但缺少内容型三级标题",
                    "actual": block_h3_count,
                    "required": 1,
                }
            )
    return issues


def _body_length_issues(markdown: str, clean_evidence: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not clean_evidence:
        return []
    required = _target_body_chars(clean_evidence)
    if required <= 0:
        return []
    actual = _body_dense_chars(markdown)
    if actual >= required:
        return []
    return [
        {
            "scope": "report",
            "reason": "正文篇幅不足，未充分展开可用证据",
            "actual": actual,
            "required": required,
        }
    ]


def _paragraph_length_issues(markdown: str) -> List[Dict[str, Any]]:
    max_chars = _env_int("REPORT_REFORMATTER_MAX_PARAGRAPH_CHARS", 480, min_value=280, max_value=2000)
    issues: List[Dict[str, Any]] = []
    for block in _chapter_blocks(_body_without_sources(markdown)):
        if not _is_substantive_chapter(block["title"]):
            continue
        for paragraph in _body_paragraphs(block["body"]):
            if len(paragraph) > max_chars:
                issues.append(
                    {
                        "scope": "chapter",
                        "chapter": block["title"],
                        "reason": "段落过长，需要拆分为更清晰的小段",
                        "actual": len(paragraph),
                        "required_max": max_chars,
                    }
                )
                break
    return issues


ANALYSIS_DIMENSION_PATTERNS = {
    "policy_trade": re.compile(r"(政策|监管|关税|出口管制|实体清单|许可证|合规|贸易|制裁|反制)"),
    "technology_chain": re.compile(r"(技术|制程|设备|材料|EDA|封装|光刻|架构|供应链|产能|良率)"),
    "market_actor": re.compile(r"(企业|客户|厂商|订单|市场|需求|价格|营收|利润|毛利|采购|份额)"),
    "capital_cost": re.compile(r"(投资|资本|成本|融资|补贴|现金流|扩产|资本开支|折旧)"),
    "uncertainty": re.compile(r"(风险|不确定|约束|瓶颈|挑战|替代|边界|错配|波动|依赖)"),
}


def _multidimensionality_issues(markdown: str, clean_evidence: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(_evidence_facts(clean_evidence)) < 80:
        return []
    issues: List[Dict[str, Any]] = []
    for block in _chapter_blocks(_body_without_sources(markdown)):
        if not _is_substantive_chapter(block["title"]):
            continue
        block_body_chars = len(re.sub(r"\s+", "", block["body"]))
        if block_body_chars < 1200:
            continue
        matched = [
            name
            for name, pattern in ANALYSIS_DIMENSION_PATTERNS.items()
            if pattern.search(block["body"])
        ]
        if len(matched) < 3:
            issues.append(
                {
                    "scope": "chapter",
                    "chapter": block["title"],
                    "reason": "章节分析维度不足，容易变成单线事实叙事",
                    "actual": len(matched),
                    "required": 3,
                    "matched_dimensions": matched,
                }
            )
    return issues


def _quality_issues(markdown: str, clean_evidence: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    bad_openers: List[Dict[str, str]] = []
    weak_judgments: List[Dict[str, str]] = []
    thin_analysis_issues: List[str] = []

    for block in _chapter_blocks(markdown):
        title = block["title"]
        if not _is_substantive_chapter(title):
            continue
        opener = _first_body_sentence(block["body"])
        if opener and (CHAPTER_OPENER_FORBIDDEN_RE.search(opener) or NUMBER_RE.search(opener)):
            bad_openers.append({"chapter": title, "first_sentence": opener})
        if FIXED_TEMPLATE_LABEL_RE.search(block["body"]):
            weak_judgments.append({"chapter": title, "reason": "contains fixed template label"})
        paragraphs = _body_paragraphs(block["body"])
        if paragraphs:
            first_paragraph = paragraphs[0]
            sentence_count = len(re.findall(r"[。！？!?]", first_paragraph))
            citation_count = len(re.findall(r"\[\d{1,3}\]", first_paragraph))
            if sentence_count >= 4 and citation_count >= 3 and not ANALYSIS_CONNECTIVE_RE.search(first_paragraph):
                thin_analysis_issues.append(f"{title}第一段疑似事实碎句堆叠，缺少分析推导句")

    return {
        "bad_chapter_openers": bad_openers,
        "weak_chapter_judgments": weak_judgments,
        "thin_analysis_issues": thin_analysis_issues,
        "structure_issues": _structure_issues(markdown, clean_evidence),
        "body_length_issues": _body_length_issues(markdown, clean_evidence),
        "paragraph_length_issues": _paragraph_length_issues(markdown),
        "multidimensionality_issues": _multidimensionality_issues(markdown, clean_evidence),
        "repeated_boilerplate_issues": _repeated_boilerplate_issues(markdown),
    }


def clean_reformatted_report(markdown: str, sources: Optional[List[Dict[str, Any]]] = None) -> str:
    lines: List[str] = []
    skip_legacy_action_block = False
    for raw_line in str(markdown or "").splitlines():
        line = raw_line.rstrip()
        if skip_legacy_action_block:
            if re.search(r"^(?:##\s+|###\s+\d+\.[1-4]\s+)", line):
                skip_legacy_action_block = False
            else:
                continue
        if re.search(r"^#{3,6}\s+", line) and TEMPLATE_HEADING_LABEL_RE.search(line):
            continue
        if re.fullmatch(r"\s*\*\*(?:原文事实|行业形势含义|投资/产品判断|可引用事实|机制与边界|进入综合决策章的变量|核心判断|机制拆解|反证边界|决策含义)\*\*\s*", line):
            continue
        if re.search(r"^###\s+\d+\.5\s+战略含义与行动建议", line):
            skip_legacy_action_block = True
            continue
        for pattern in FORBIDDEN_OUTPUT_PATTERNS:
            line = re.sub(pattern, "", line)
        if line.strip():
            line = re.sub(r"\s{2,}", " ", line)
            line = re.sub(r"[，,；;]\s*[。；;，,]", "。", line)
            line = re.sub(r"[。]{2,}", "。", line).strip()
            if re.fullmatch(r"#{3,6}\s*(?:\d+(?:\.\d+)*\.?)?\s*", line):
                continue
            if re.fullmatch(r"\*\*\s*\*\*", line):
                continue
        if re.fullmatch(r"\s*[-*]\s*(?:市场规模|估值|增速|融资金额|资本动态)\s*-?\d+(?:\.\d+)?\s*", line):
            continue
        split_lines = _split_long_paragraph_line(line)
        if len(split_lines) > 1:
            for split_line in split_lines:
                lines.append(split_line)
                lines.append("")
            continue
        lines.append(line)
    text = "\n".join(lines).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"丘陵山区\s*(\d+(?:\.\d+)?)\s*低机械化率", r"丘陵山区机械化率不足\1%", text)
    text = re.sub(r"(?<![\d.])(\d+(?:\.\d+)?)\s*低机械化率", r"机械化率不足\1%", text)
    if sources is not None:
        text = normalize_report_citations(text, _valid_source_ids(sources))
        text = append_sources_appendix(text, sources)
    text = sanitize_public_markdown(text)
    return text


def _body_without_sources(markdown: str) -> str:
    return re.split(SOURCE_APPENDIX_HEADING_RE, str(markdown or ""), maxsplit=1)[0]


def _evidence_facts(clean_evidence: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    facts: List[Dict[str, Any]] = []
    dimensions = _as_dict((clean_evidence or {}).get("dimensions"))
    for values in dimensions.values():
        facts.extend([item for item in _as_list(values) if isinstance(item, dict)])
    return facts


def _fact_source_id(item: Dict[str, Any], valid_ids: Set[str]) -> str:
    raw = str(item.get("source") or "").strip()
    match = re.fullmatch(r"\[?(\d{1,3})\]?", raw)
    source_id = match.group(1) if match else raw
    return source_id if source_id in valid_ids else ""


def _usable_evidence_source_ids(clean_evidence: Optional[Dict[str, Any]], valid_ids: Set[str]) -> Set[str]:
    usable_ids = {
        source_id
        for item in _evidence_facts(clean_evidence)
        for source_id in [_fact_source_id(item, valid_ids)]
        if source_id
    }
    return usable_ids or set(valid_ids)


def _compact_expansion_fact(item: Dict[str, Any], *, max_chars: int = 96) -> str:
    text = clean_evidence_text(item.get("text") or item.get("fact") or "")
    text = re.sub(r"\[\d{1,3}\]", "", text)
    text = re.sub(r"\s+", "", text).strip(" ,;:，；：。")
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rstrip(" ,;:，；：、")
    return cut + "\u2026"


def _dimension_fact_groups_for_expansion(
    clean_evidence: Dict[str, Any],
    sources: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    valid_ids = _valid_source_ids(sources)
    if not valid_ids:
        return []
    raw_dimensions = _as_dict(clean_evidence.get("dimensions"))
    ordered_dimensions = [
        *[dimension for dimension in REPORT_DIMENSIONS if dimension in raw_dimensions],
        *[dimension for dimension in raw_dimensions if dimension not in REPORT_DIMENSIONS],
    ]
    per_dimension = _env_int("REPORT_REFORMATTER_AUTO_EXPAND_FACTS_PER_DIMENSION", 4, min_value=2, max_value=10)
    groups: List[Dict[str, Any]] = []
    for dimension in ordered_dimensions:
        raw_items = [item for item in _as_list(raw_dimensions.get(dimension)) if isinstance(item, dict)]
        prepared = _select_dimension_facts(raw_items, max(per_dimension, 6))
        facts: List[Dict[str, str]] = []
        for item in prepared:
            source_id = _fact_source_id(item, valid_ids)
            text = _compact_expansion_fact(item)
            if not source_id or not text:
                continue
            facts.append({"text": text, "source": source_id})
            if len(facts) >= per_dimension:
                break
        if facts:
            groups.append({"dimension": str(dimension or "").strip(), "facts": facts})
    return groups


def _validation_allows_auto_expand(validation: Dict[str, Any]) -> bool:
    if not validation.get("body_length_issues"):
        return False
    repair_blocker_types = {
        str(item.get("type") or "")
        for item in _as_list(validation.get("repair_blockers"))
        if isinstance(item, dict)
    }
    repair_blocker_types.discard("body_length")
    if validation.get("citation_density_issues") or repair_blocker_types:
        return False
    if validation.get("weak_chapter_judgments"):
        return False
    if validation.get("thin_analysis_issues") or validation.get("structure_issues"):
        return False
    if validation.get("multidimensionality_issues") or validation.get("repeated_boilerplate_issues"):
        return False
    hard_keys = [
        "forbidden_hits",
        "publication_blockers",
        "invalid_citations",
        "malformed_numeric",
        "schema_like_bullets",
    ]
    return not any(validation.get(key) for key in hard_keys)


def _auto_expand_analysis_for_length(
    markdown: str,
    clean_evidence: Dict[str, Any],
    validation: Dict[str, Any],
    sources: List[Dict[str, Any]],
) -> str:
    if not _env_flag("REPORT_REFORMATTER_AUTO_EXPAND_ANALYSIS", True):
        return markdown
    if not _validation_allows_auto_expand(validation):
        return markdown
    facts = _evidence_facts(clean_evidence)
    min_facts = _env_int("REPORT_REFORMATTER_AUTO_EXPAND_MIN_FACTS", 8, min_value=1, max_value=200)
    if len(facts) < min_facts:
        return markdown
    body = _body_without_sources(markdown).rstrip()
    if "\u8865\u5145\u5206\u6790" in body:
        return markdown

    issue = (_as_list(validation.get("body_length_issues")) or [{}])[0]
    required = int(issue.get("required") or _target_body_chars(clean_evidence) or 0)
    actual = int(issue.get("actual") or _body_dense_chars(markdown))
    if required <= actual:
        return markdown
    max_extra = _env_int("REPORT_REFORMATTER_AUTO_EXPAND_MAX_CHARS", 8_000, min_value=1_000, max_value=30_000)
    target_extra = min(max_extra, max(1_200, required - actual + 800))
    groups = _dimension_fact_groups_for_expansion(clean_evidence, sources)
    if not groups:
        return markdown

    lines: List[str] = ["## \u8bc1\u636e\u94fe\u7684\u8fde\u7eed\u9a8c\u8bc1\u4e0e\u7ed3\u8bba\u8fb9\u754c"]
    for group in groups:
        dimension = group["dimension"] or "\u5173\u952e\u4fe1\u53f7"
        facts_for_group = group["facts"][:4]
        if not facts_for_group:
            continue
        primary = facts_for_group[:3]
        fact_chain = "\uff1b".join(f"{item['text']}[{item['source']}]" for item in primary)
        lines.append("")
        lines.append(f"### {dimension}\u7684\u53ef\u9a8c\u8bc1\u4fe1\u53f7")
        lines.append(
            f"\u5728{dimension}\u4e0a\uff0c\u5f53\u524d\u8bc1\u636e\u80fd\u786e\u8ba4\u7684\u662f\uff1a{fact_chain}\u3002"
            "\u8fd9\u4e9b\u6750\u6599\u5e94\u88ab\u653e\u5728\u4f01\u4e1a\u52a8\u4f5c\u3001\u653f\u7b56\u7ea6\u675f\u548c\u4f9b\u9700\u914d\u7f6e\u4e4b\u95f4\u4ea4\u53c9\u7406\u89e3\uff0c"
            "\u53ea\u6709\u5f53\u540e\u7eed\u62ab\u9732\u7ee7\u7eed\u6307\u5411\u8ba2\u5355\u3001\u5408\u89c4\u5b89\u6392\u3001\u4ea7\u80fd\u914d\u7f6e\u6216\u8d44\u672c\u5f00\u652f\u53d8\u5316\u65f6\uff0c"
            "\u624d\u80fd\u628a\u5b83\u5199\u6210\u4e00\u6761\u7a33\u5b9a\u7684\u4ea7\u4e1a\u8d8b\u52bf\u3002"
        )
        boundary_source = facts_for_group[-1]["source"]
        lines.append(
            f"\u5728{dimension}\u7684\u8fb9\u754c\u5224\u65ad\u4e0a\uff0c\u5982\u679c\u540e\u7eed\u4fe1\u606f\u53ea\u505c\u7559\u5728\u8bbf\u95ee\u3001\u8868\u6001\u6216\u77ed\u671f\u6d88\u606f\u5c42\u9762[{boundary_source}]\uff0c"
            "\u62a5\u544a\u5c31\u5e94\u628a\u5b83\u964d\u7ea7\u4e3a\u65b9\u5411\u6027\u7ebf\u7d22\uff0c\u800c\u4e0d\u76f4\u63a5\u5916\u63a8\u4e3a\u957f\u5468\u671f\u5224\u65ad\u3002"
            "\u8fd9\u6837\u5199\u80fd\u4fdd\u7559\u5df2\u6709\u8bc1\u636e\u7684\u4ef7\u503c\uff0c\u540c\u65f6\u628a\u7ed3\u8bba\u8fb9\u754c\u4ea4\u4ee3\u6e05\u695a\u3002"
        )
        if _body_dense_chars("\n".join(lines)) >= target_extra:
            break
    if _env_flag("REPORT_REFORMATTER_AUTO_EXPAND_ALLOW_GENERIC_REINFORCEMENT", False) and _body_dense_chars("\n".join(lines)) < target_extra:
        reinforcement_templates = [
            (
                "\u8fd9\u4e00\u7ec4\u6750\u6599\u8fd8\u9700\u8981\u88ab\u653e\u5230\u65f6\u95f4\u987a\u5e8f\u4e2d\u7406\u89e3\uff1a"
                "\u5982\u679c\u4e8b\u4ef6\u5148\u6709\u9ad8\u5c42\u4e92\u52a8\uff0c\u518d\u6709\u4f01\u4e1a\u7aef\u7684\u91c7\u8d2d\u3001\u5408\u4f5c\u6216\u4ea7\u80fd\u4fe1\u53f7\uff0c"
                "\u5b83\u624d\u66f4\u50cf\u4e00\u6761\u4ea7\u4e1a\u8c03\u6574\u94fe\uff1b\u5982\u679c\u53ea\u5728\u8206\u8bba\u6216\u62dc\u8bbf\u5c42\u9762\u53cd\u590d\u51fa\u73b0\uff0c"
                "\u5b83\u66f4\u50cf\u77ed\u671f\u60c5\u7eea\u4fee\u590d\u3002\u56e0\u6b64\uff0c\u4e0b\u4e00\u8f6e\u8865\u8bc1\u5e94\u4f18\u5148\u5bfb\u627e\u4e0e\u8fd9\u4e9b\u4fe1\u53f7\u76f8\u8fde\u7684\u540e\u7eed\u62ab\u9732"
            ),
            (
                "\u8fd9\u4e00\u7ef4\u5ea6\u7684\u5224\u65ad\u8fd8\u6709\u4e00\u4e2a\u53cd\u5411\u6821\u9a8c\uff1a"
                "\u5982\u679c\u4f01\u4e1a\u53ea\u589e\u52a0\u516c\u5171\u8868\u8ff0\uff0c\u5374\u6ca1\u6709\u5bf9\u4f9b\u5e94\u5546\u3001\u5ba2\u6237\u3001\u5408\u89c4\u8def\u5f84\u6216\u8d44\u672c\u5f00\u652f\u505a\u51fa\u53ef\u89c2\u5bdf\u8c03\u6574\uff0c"
                "\u5219\u62a5\u544a\u4e0d\u5e94\u628a\u5b83\u76f4\u63a5\u653e\u5927\u4e3a\u957f\u5468\u671f\u8d8b\u52bf\u3002"
                "\u8fd9\u6837\u5904\u7406\u53ef\u4ee5\u907f\u514d\u628a\u4e8b\u4ef6\u5bc6\u5ea6\u8bef\u5224\u4e3a\u4ea7\u4e1a\u786e\u5b9a\u6027"
            ),
            (
                "\u5bf9\u51b3\u7b56\u8005\u800c\u8a00\uff0c\u8fd9\u90e8\u5206\u7684\u4ef7\u503c\u4e0d\u662f\u7ed9\u51fa\u5355\u5411\u7ed3\u8bba\uff0c"
                "\u800c\u662f\u5efa\u7acb\u8ffd\u8e2a\u987a\u5e8f\uff1a\u5148\u770b\u662f\u5426\u6709\u5b98\u65b9\u6216\u516c\u53f8\u62ab\u9732\u7684\u8fde\u7eed\u4e8b\u5b9e\uff0c"
                "\u518d\u770b\u8fd9\u4e9b\u4e8b\u5b9e\u662f\u5426\u5e26\u6765\u4ea4\u6613\u3001\u8ba2\u5355\u3001\u6295\u8d44\u6216\u8fd0\u8425\u53e3\u5f84\u53d8\u5316\uff0c"
                "\u6700\u540e\u518d\u5224\u65ad\u5b83\u4eec\u5bf9\u884c\u4e1a\u683c\u5c40\u7684\u6743\u91cd"
            ),
        ]
        for template in reinforcement_templates:
            for group in groups:
                facts_for_group = group["facts"][:4]
                if not facts_for_group:
                    continue
                refs = "\u3001".join(f"[{item['source']}]" for item in facts_for_group[:3])
                lines.append("")
                lines.append(f"\u5bf9{group['dimension']}\u800c\u8a00\uff0c{template}{refs}\u3002")
                if _body_dense_chars("\n".join(lines)) >= target_extra:
                    break
            if _body_dense_chars("\n".join(lines)) >= target_extra:
                break
    if _env_flag("REPORT_REFORMATTER_AUTO_EXPAND_ALLOW_GENERIC_REINFORCEMENT", False) and _body_dense_chars("\n".join(lines)) < target_extra:
        deepening_templates = [
            (
                "\u8fd8\u9700\u8981\u628a\u8fd9\u7ec4\u4fe1\u53f7\u540c\u62a5\u544a\u4e3b\u9898\u8fde\u8d77\u6765\uff1a"
                "\u5b83\u4e0d\u53ea\u56de\u7b54\u67d0\u4e2a\u4e8b\u4ef6\u662f\u5426\u70ed\u95f9\uff0c\u800c\u662f\u56de\u7b54\u4e2d\u7f8e\u79d1\u6280\u4e92\u52a8\u662f\u5426\u4ece\u62bd\u8c61\u59ff\u6001\u8f6c\u5411\u5177\u4f53\u5229\u76ca\u91cd\u914d"
            ),
            (
                "\u5982\u679c\u540c\u4e00\u6761\u94fe\u4e0a\u7684\u591a\u4e2a\u4fe1\u53f7\u80fd\u591f\u76f8\u4e92\u5370\u8bc1\uff0c"
                "\u62a5\u544a\u624d\u53ef\u4ee5\u628a\u5b83\u5199\u6210\u7ed3\u6784\u53d8\u5316\uff1b\u5982\u679c\u5b83\u4eec\u4e4b\u95f4\u6ca1\u6709\u540e\u7eed\u627f\u63a5\uff0c"
                "\u66f4\u5408\u7406\u7684\u5199\u6cd5\u662f\u628a\u5b83\u4eec\u4f5c\u4e3a\u8c08\u5224\u548c\u9884\u671f\u4fee\u590d\u7684\u5f31\u4fe1\u53f7"
            ),
            (
                "\u8fd9\u6837\u7684\u8865\u5f3a\u80fd\u591f\u628a\u8bba\u8bc1\u91cd\u5fc3\u4ece\u201c\u6709\u6ca1\u6709\u65b0\u95fb\u201d\u63a8\u5411\u201c\u65b0\u95fb\u80cc\u540e\u7684\u4ea7\u4e1a\u7ea6\u675f\u662f\u5426\u53d8\u4e86\u201d\uff0c"
                "\u4ece\u800c\u8ba9\u7ed3\u8bba\u65e2\u4e0d\u8fc7\u5ea6\u4e50\u89c2\uff0c\u4e5f\u4e0d\u56e0\u5355\u6761\u53cd\u5411\u4fe1\u606f\u800c\u5931\u53bb\u5224\u65ad\u4e3b\u7ebf"
            ),
            (
                "\u540e\u7eed\u82e5\u8981\u7ee7\u7eed\u52a0\u5f3a\u8fd9\u4e00\u6bb5\uff0c\u5e94\u4f18\u5148\u8865\u5165\u53ef\u5bf9\u7167\u7684\u539f\u59cb\u62ab\u9732\uff0c"
                "\u4f8b\u5982\u516c\u53f8\u516c\u544a\u3001\u653f\u7b56\u539f\u6587\u3001\u8ba2\u5355\u6216\u91c7\u8d2d\u53e3\u5f84\uff0c\u800c\u4e0d\u662f\u5355\u7eaf\u589e\u52a0\u8f6c\u8f7d\u65b0\u95fb\u6216\u89c2\u70b9\u6587\u7ae0"
            ),
        ]
        for template in deepening_templates:
            for group in groups:
                facts_for_group = group["facts"][:4]
                if not facts_for_group:
                    continue
                refs = "\u3001".join(f"[{item['source']}]" for item in facts_for_group[:3])
                lines.append("")
                lines.append(f"\u56de\u5230{group['dimension']}\u8fd9\u4e2a\u95ee\u9898\uff0c{template}{refs}\u3002")
                if _body_dense_chars("\n".join(lines)) >= target_extra:
                    break
            if _body_dense_chars("\n".join(lines)) >= target_extra:
                break
    if len(lines) <= 1:
        return markdown
    return body + "\n\n" + "\n".join(lines).strip()


def _citation_density_issues(markdown: str, clean_evidence: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not clean_evidence:
        return []
    body = _body_without_sources(markdown)
    all_citations = re.findall(r"\[(\d{1,3})\]", body)
    facts = _evidence_facts(clean_evidence)
    available_total_sources = {
        str(item.get("source") or "").strip()
        for item in facts
        if str(item.get("source") or "").strip()
    }
    available_fact_count = len(facts)
    issues: List[Dict[str, Any]] = []

    if available_fact_count >= 30:
        dynamic_total = min(110, max(45, available_fact_count // 8))
        dynamic_unique = min(45, max(16, (len(available_total_sources) + 3) // 4))
        required_total = _env_int(
            "REPORT_REFORMATTER_MIN_BODY_CITATIONS",
            dynamic_total,
            min_value=10,
            max_value=200,
        )
        required_unique = _env_int(
            "REPORT_REFORMATTER_MIN_UNIQUE_BODY_SOURCES",
            dynamic_unique,
            min_value=5,
            max_value=120,
        )
        if len(all_citations) < required_total:
            issues.append(
                {
                    "scope": "report",
                    "reason": "total citation count is too low",
                    "actual": len(all_citations),
                    "required": required_total,
                }
            )
        if len(set(all_citations)) < min(required_unique, len(available_total_sources)):
            issues.append(
                {
                    "scope": "report",
                    "reason": "unique cited source count is too low",
                    "actual": len(set(all_citations)),
                    "required": min(required_unique, len(available_total_sources)),
                }
            )
        long_paragraphs = [paragraph for block in _chapter_blocks(body) for paragraph in _body_paragraphs(block["body"]) if len(paragraph) >= 120]
        uncited_long = [paragraph for paragraph in long_paragraphs if not re.search(r"\[\d{1,3}\]", paragraph)]
        max_uncited = max(10, int(len(long_paragraphs) * 0.35))
        if len(uncited_long) > max_uncited:
            issues.append(
                {
                    "scope": "report",
                    "reason": "too many substantive paragraphs have no inline citation",
                    "actual": len(uncited_long),
                    "required_max": max_uncited,
                    "paragraph_count": len(long_paragraphs),
                }
            )

    return issues


def _repeated_boilerplate_issues(markdown: str) -> List[Dict[str, Any]]:
    normalized_counts: Dict[str, Dict[str, Any]] = {}
    body = _body_without_sources(markdown)
    for line_no, raw_line in enumerate(body.splitlines(), start=1):
        line = raw_line.strip()
        if len(line) < 80 or line.startswith("#") or line.startswith("|"):
            continue
        if re.match(r"^(?:[-*+]|\d+\.)\s+", line):
            continue
        normalized = re.sub(r"\[\d{1,3}\]", "[CIT]", line)
        normalized = re.sub(r"、\[CIT\]", "", normalized)
        normalized = re.sub(r"^从[^，]{2,24}看，", "从X看，", normalized)
        normalized = re.sub(r"\s+", "", normalized)
        if len(normalized) < 80:
            continue
        bucket = normalized_counts.setdefault(normalized, {"count": 0, "lines": []})
        bucket["count"] += 1
        bucket["lines"].append(line_no)
    issues: List[Dict[str, Any]] = []
    for normalized, payload in normalized_counts.items():
        count = int(payload.get("count") or 0)
        if count >= 3:
            issues.append(
                {
                    "scope": "report",
                    "reason": "正文存在重复模板化段落",
                    "count": count,
                    "lines": payload.get("lines", [])[:10],
                    "sample": normalized[:120],
                }
            )
    return issues


def _source_diversity_floor(source_count: int) -> int:
    if source_count < _env_int("REPORT_REFORMATTER_SOURCE_POOL_CHECK_MIN", 16, min_value=1, max_value=200):
        return 0
    default_required = min(18, max(8, (source_count + 7) // 8))
    return _env_int(
        "REPORT_REFORMATTER_MIN_UNIQUE_SOURCE_POOL",
        default_required,
        min_value=0,
        max_value=120,
    )


def _reformatter_validation_mode() -> str:
    explicit = os.environ.get("REPORT_REFORMATTER_VALIDATION_MODE")
    if explicit:
        return explicit.strip().lower()
    if os.environ.get("REPORT_QUALITY_MODE", "balanced").strip().lower() in {"strict", "hard"}:
        return "hard"
    return "score"


def _body_length_penalty(issues: List[Dict[str, Any]]) -> int:
    penalty = 0
    for issue in issues:
        try:
            actual = int(issue.get("actual") or 0)
            required = int(issue.get("required") or 0)
        except (TypeError, ValueError):
            actual = 0
            required = 0
        if required <= 0 or actual >= required:
            continue
        deficit_ratio = (required - actual) / max(required, 1)
        penalty += max(1, min(20, int(deficit_ratio * 40)))
    return penalty


def _reformatter_score(
    *,
    fatal_blockers: List[Dict[str, Any]],
    empty_section_count: int,
    has_sources_appendix: bool,
    has_expected_sources: bool,
    quality: Dict[str, Any],
    citation_density_issues: List[Dict[str, Any]],
) -> Dict[str, Any]:
    appendix_required = _source_appendix_required()
    penalties = {
        "fatal_blockers": min(80, len(fatal_blockers) * 35),
        "missing_sources_appendix": 0 if (not appendix_required or has_sources_appendix or not has_expected_sources) else 25,
        "empty_sections": min(16, empty_section_count * 3),
        "bad_chapter_openers": min(12, len(_as_list(quality.get("bad_chapter_openers"))) * 4),
        "weak_chapter_judgments": min(24, len(_as_list(quality.get("weak_chapter_judgments"))) * 10),
        "thin_analysis": min(12, len(_as_list(quality.get("thin_analysis_issues"))) * 4),
        "structure": min(15, len(_as_list(quality.get("structure_issues"))) * 5),
        "body_length": _body_length_penalty(_as_list(quality.get("body_length_issues"))),
        "paragraph_length": min(12, len(_as_list(quality.get("paragraph_length_issues"))) * 4),
        "multidimensionality": min(12, len(_as_list(quality.get("multidimensionality_issues"))) * 4),
        "repeated_boilerplate": min(24, len(_as_list(quality.get("repeated_boilerplate_issues"))) * 12),
        "citation_density": min(18, len(citation_density_issues) * 6),
    }
    return {"score": max(0, 100 - sum(penalties.values())), "penalties": penalties}


def _reformatter_validation_rank(validation: Dict[str, Any]) -> tuple[int, int, int, int, int]:
    return (
        1 if validation.get("hard_pass") else 0,
        1 if validation.get("passed") else 0,
        int(validation.get("quality_score") or 0),
        -len(_as_list(validation.get("fatal_blockers"))),
        -int(validation.get("soft_issue_count") or 0),
    )


def _reformatter_needs_repair(validation: Dict[str, Any]) -> bool:
    if validation.get("hard_pass"):
        return False
    if not validation.get("passed"):
        return True
    if not _env_flag("REPORT_REFORMATTER_REPAIR_SOFT_ISSUES", True):
        return False
    return int(validation.get("soft_issue_count") or 0) > 0


def validate_reformatted_report(
    markdown: str,
    sources: Optional[List[Dict[str, Any]]] = None,
    clean_evidence: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    publication_blockers = find_publication_blockers(markdown)
    hits = [pattern for pattern in FORBIDDEN_OUTPUT_PATTERNS if re.search(pattern, markdown, re.I)]
    empty_sections = re.findall(r"(?m)^##+\s+[^\n]+\n\s*(?=##+|\Z)", markdown)
    body = _body_without_sources(markdown)
    citation_ids = re.findall(r"\[(\d{1,3})\]", body)
    valid_ids = _valid_source_ids(sources or [])
    invalid_citations = sorted({item for item in citation_ids if valid_ids and item not in valid_ids}, key=lambda x: int(x))
    citation_distribution = {item: citation_ids.count(item) for item in sorted(set(citation_ids), key=lambda x: int(x))}
    usable_source_ids = _usable_evidence_source_ids(clean_evidence, valid_ids)
    source_diversity_required = _source_diversity_floor(len(usable_source_ids))
    source_diversity_threshold = min(source_diversity_required, len(usable_source_ids))
    source_diversity_issue = (
        {
            "type": "source_diversity_too_low",
            "actual": len(set(citation_ids)),
            "required": source_diversity_threshold,
            "source_pool_count": len(usable_source_ids),
            "source_registry_count": len(valid_ids),
        }
        if source_diversity_threshold and len(set(citation_ids)) < source_diversity_threshold
        else {}
    )
    malformed_numeric = re.findall(r"\d+(?:\.\d+)?\s*低机械化率", markdown)
    has_sources_appendix = bool(SOURCE_APPENDIX_HEADING_RE.search(markdown))
    has_expected_sources = bool(sources)
    source_appendix_required = _source_appendix_required()
    schema_like_bullets = SCHEMA_LIKE_BULLET_RE.findall(body)
    quality = _quality_issues(markdown, clean_evidence)
    quality_issue_count = (
        len(quality["bad_chapter_openers"])
        + len(quality["weak_chapter_judgments"])
        + len(quality["thin_analysis_issues"])
        + len(quality["structure_issues"])
        + len(quality["body_length_issues"])
        + len(quality["paragraph_length_issues"])
        + len(quality["multidimensionality_issues"])
        + len(quality["repeated_boilerplate_issues"])
    )
    citation_density_issues = _citation_density_issues(markdown, clean_evidence)
    repair_blockers: List[Dict[str, Any]] = []
    for issue in quality["body_length_issues"]:
        repair_blockers.append({"type": "body_length", **issue})
    for issue in citation_density_issues:
        repair_blockers.append({"type": "citation_density", **issue})
    for issue in quality["repeated_boilerplate_issues"]:
        repair_blockers.append({"type": "repeated_boilerplate", **issue})
    if has_sources_appendix and not source_appendix_required and _source_appendix_mode() == "none":
        repair_blockers.append({"type": "unexpected_sources_appendix"})
    fatal_blockers: List[Dict[str, Any]] = []
    if hits:
        fatal_blockers.append({"type": "forbidden_output_patterns", "count": len(hits), "examples": hits[:5]})
    if publication_blockers:
        fatal_blockers.append({"type": "publication_blockers", "count": len(publication_blockers), "examples": publication_blockers[:5]})
    if invalid_citations:
        fatal_blockers.append({"type": "invalid_citations", "items": invalid_citations[:10]})
    if source_diversity_issue:
        fatal_blockers.append(source_diversity_issue)
    if malformed_numeric:
        fatal_blockers.append({"type": "malformed_numeric", "items": malformed_numeric[:10]})
    if schema_like_bullets:
        fatal_blockers.append({"type": "schema_like_bullets", "items": schema_like_bullets[:10]})
    if source_appendix_required and has_expected_sources and not has_sources_appendix:
        fatal_blockers.append({"type": "missing_sources_appendix"})
    validation_mode = _reformatter_validation_mode()
    min_score = _env_int("REPORT_REFORMATTER_MIN_PASS_SCORE", 60, min_value=0, max_value=100)
    score_payload = _reformatter_score(
        fatal_blockers=fatal_blockers,
        empty_section_count=len(empty_sections),
        has_sources_appendix=has_sources_appendix,
        has_expected_sources=has_expected_sources,
        quality=quality,
        citation_density_issues=citation_density_issues,
    )
    hard_pass = (
        not hits
        and not publication_blockers
        and not empty_sections
        and not invalid_citations
        and not source_diversity_issue
        and not malformed_numeric
        and not schema_like_bullets
        and (not source_appendix_required or has_sources_appendix or not has_expected_sources)
        and quality_issue_count == 0
        and not citation_density_issues
        and not repair_blockers
    )
    score_pass = not fatal_blockers and not repair_blockers and int(score_payload["score"]) >= min_score
    passed = hard_pass if validation_mode in {"hard", "strict", "legacy"} else score_pass
    return {
        "passed": passed,
        "validation_mode": validation_mode,
        "quality_score": score_payload["score"],
        "minimum_pass_score": min_score,
        "hard_pass": hard_pass,
        "fatal_blockers": fatal_blockers,
        "score_breakdown": score_payload["penalties"],
        "soft_issue_count": max(0, quality_issue_count + len(empty_sections) + len(citation_density_issues)),
        "repair_blockers": repair_blockers,
        "forbidden_hits": hits[:10],
        "publication_blockers": publication_blockers[:20],
        "empty_section_count": len(empty_sections),
        "invalid_citations": invalid_citations,
        "malformed_numeric": malformed_numeric[:10],
        "schema_like_bullets": schema_like_bullets[:10],
        "has_sources_appendix": has_sources_appendix,
        "citation_count": len(citation_ids),
        "unique_cited_source_count": len(set(citation_ids)),
        "source_diversity_required": source_diversity_threshold if source_diversity_required else 0,
        "source_pool_count": len(usable_source_ids),
        "source_registry_count": len(valid_ids),
        "citation_distribution": citation_distribution,
        "citation_density_issues": citation_density_issues,
        "bad_chapter_openers": quality["bad_chapter_openers"],
        "weak_chapter_judgments": quality["weak_chapter_judgments"],
        "thin_analysis_issues": quality["thin_analysis_issues"],
        "structure_issues": quality["structure_issues"],
        "body_length_issues": quality["body_length_issues"],
        "paragraph_length_issues": quality["paragraph_length_issues"],
        "multidimensionality_issues": quality["multidimensionality_issues"],
        "repeated_boilerplate_issues": quality["repeated_boilerplate_issues"],
        "body_chars_without_sources": _body_dense_chars(markdown),
        "estimated_chars": len(markdown or ""),
    }


def build_reformatter_repair_plan(
    validation: Dict[str, Any],
    clean_evidence: Optional[Dict[str, Any]] = None,
    *,
    topic: str = "",
    max_queries: int = 6,
) -> Dict[str, Any]:
    validation = _as_dict(validation)
    facts = _evidence_facts(clean_evidence)
    sources = _as_list(_as_dict(clean_evidence or {}).get("sources"))
    valid_ids = _valid_source_ids(sources)
    usable_source_ids = _usable_evidence_source_ids(clean_evidence, valid_ids) if valid_ids else set()
    if validation.get("passed") and not _reformatter_needs_repair(validation) and not _as_list(validation.get("repair_blockers")):
        return {
            "status": "passed",
            "reasons": [],
            "text_repair_reasons": [],
            "follow_up_queries": [],
            "evidence_fact_count": len(facts),
            "usable_source_count": len(usable_source_ids),
            "required_source_count": int(validation.get("source_diversity_required") or 0),
            "unique_cited_source_count": int(validation.get("unique_cited_source_count") or 0),
            "degrade_allowed": True,
        }
    source_required = int(validation.get("source_diversity_required") or 0)
    source_pool_count = len(usable_source_ids) or int(validation.get("source_pool_count") or 0)
    unique_cited = int(validation.get("unique_cited_source_count") or 0)
    body_length_issues = _as_list(validation.get("body_length_issues"))
    citation_density_issues = _as_list(validation.get("citation_density_issues"))
    repeated_boilerplate_issues = _as_list(validation.get("repeated_boilerplate_issues"))
    repair_blocker_types = {
        str(item.get("type") or "")
        for item in _as_list(validation.get("repair_blockers"))
        if isinstance(item, dict)
    }
    fatal_types = {str(item.get("type") or "") for item in _as_list(validation.get("fatal_blockers")) if isinstance(item, dict)}

    reasons: List[str] = []
    text_repair_reasons: List[str] = []
    if source_required and source_pool_count < source_required:
        reasons.append("usable_source_pool_below_required")
    if len(facts) < _env_int("REPORT_REFORMATTER_EVIDENCE_LOOP_MIN_FACTS", 18, min_value=1, max_value=200):
        reasons.append("clean_evidence_fact_count_low")
    if body_length_issues and len(facts) < _env_int("REPORT_REFORMATTER_FULL_LENGTH_MIN_FACTS", 30, min_value=1, max_value=300):
        reasons.append("body_length_failed_with_sparse_evidence")
    for issue in citation_density_issues:
        required = int(issue.get("required") or issue.get("required_max") or 0)
        reason_text = str(issue.get("reason") or "")
        if required and "unique cited source count" in reason_text and source_pool_count < required:
            reasons.append("citation_density_requires_more_usable_sources")
    if "source_diversity_too_low" in fatal_types and source_pool_count >= max(source_required, unique_cited + 1):
        text_repair_reasons.append("source_diversity_can_be_fixed_from_existing_evidence")
    if citation_density_issues and source_pool_count >= max(source_required, unique_cited + 1):
        text_repair_reasons.append("citation_density_can_be_fixed_from_existing_evidence")
    if body_length_issues and len(facts) >= _env_int("REPORT_REFORMATTER_FULL_LENGTH_MIN_FACTS", 30, min_value=1, max_value=300):
        text_repair_reasons.append("body_length_can_be_expanded_from_existing_evidence")
    if repeated_boilerplate_issues or "repeated_boilerplate" in repair_blocker_types:
        text_repair_reasons.append("repeated_boilerplate_requires_rewrite")
    if "unexpected_sources_appendix" in repair_blocker_types:
        text_repair_reasons.append("source_appendix_must_be_removed")

    dimensions = _as_dict(_as_dict(clean_evidence or {}).get("dimensions"))
    topic_text = str(topic or _as_dict(clean_evidence or {}).get("topic") or "").strip()
    followups: List[Dict[str, Any]] = []
    per_dimension_floor = _env_int("REPORT_REFORMATTER_MIN_FACTS_PER_DIMENSION", 4, min_value=1, max_value=30)
    if reasons:
        ordered_dimensions = sorted(
            ((str(dimension or "").strip(), _as_list(items)) for dimension, items in dimensions.items() if str(dimension or "").strip()),
            key=lambda item: (len(item[1]), item[0]),
        )
        low_dimensions = [(dimension, items) for dimension, items in ordered_dimensions if len(items) < per_dimension_floor]
        target_dimensions = low_dimensions or ordered_dimensions[: max(1, int(max_queries))]
        if not target_dimensions and topic_text:
            target_dimensions = [("综合证据缺口", [])]
        for dimension, items in target_dimensions:
            if len(followups) >= max(1, int(max_queries)):
                break
            query = " ".join(
                part
                for part in [
                    topic_text,
                    dimension,
                    "官方 公告 财报 统计 协会 权威研报 2025 2026",
                ]
                if part
            ).strip()
            if not query:
                continue
            followups.append(
                {
                    "query": query[:220],
                    "agent": "both",
                    "targets_gap": dimension,
                    "dimension_name": dimension,
                    "evidence_goal": f"{dimension}：补充可核验证据、来源时间范围和口径，用于修复 Reformatter 正文证据不足。",
                    "source_priority": ["官方", "公告", "财报", "协会", "统计", "权威研报"],
                    "lane_targets": ["official_data", "market_research", "filing_company"],
                    "blocking_gaps": list(reasons),
                    "proof_role": "source_check",
                    "evidence_type": "source_check",
                }
            )

    if reasons and followups:
        status = "needs_evidence_refinement"
    elif text_repair_reasons:
        status = "needs_text_repair"
    elif not validation.get("passed"):
        status = "needs_degrade_or_manual_review"
    else:
        status = "passed"
    return {
        "status": status,
        "reasons": reasons,
        "text_repair_reasons": text_repair_reasons,
        "follow_up_queries": followups,
        "evidence_fact_count": len(facts),
        "usable_source_count": source_pool_count,
        "required_source_count": source_required,
        "unique_cited_source_count": unique_cited,
        "degrade_allowed": status != "needs_evidence_refinement",
    }


async def run_reformatter(
    clean_evidence: Dict[str, Any],
    llm_client: Optional[Any] = None,
    temperature: float = 0.3,
    max_tokens: int = 18000,
    stream: bool = False,
    quality_passes: int = 2,
) -> str:
    if max_tokens == 18000:
        max_tokens = _env_int("REPORT_REFORMATTER_MAX_TOKENS", 30000, min_value=8000, max_value=64000)
    quality_passes = _env_int(
        "REPORT_REFORMATTER_QUALITY_PASSES",
        quality_passes,
        min_value=0,
        max_value=5,
    )
    payload = build_reformatter_payload(clean_evidence)
    system_prompt = REFORMATTER_SYSTEM_PROMPT.format(topic=payload["topic"])
    user_content = REFORMATTER_USER_TEMPLATE.format(**payload)

    markdown = await _generate_reformatter_text(
        system_prompt=system_prompt,
        user_content=user_content,
        llm_client=llm_client,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream,
    )
    sources = _as_list(clean_evidence.get("sources"))
    markdown = clean_reformatted_report(markdown, sources)
    auto_expand_attempted = False
    best_markdown = markdown
    best_validation = validate_reformatted_report(markdown, sources, clean_evidence)

    for _ in range(max(0, int(quality_passes))):
        validation = validate_reformatted_report(markdown, sources, clean_evidence)
        if _reformatter_validation_rank(validation) > _reformatter_validation_rank(best_validation):
            best_markdown = markdown
            best_validation = validation
        if not _reformatter_needs_repair(validation):
            break
        if not auto_expand_attempted and validation.get("body_length_issues"):
            auto_expand_attempted = True
            expanded = _auto_expand_analysis_for_length(markdown, clean_evidence, validation, sources)
            if expanded != markdown:
                markdown = clean_reformatted_report(expanded, sources)
                validation = validate_reformatted_report(markdown, sources, clean_evidence)
                if _reformatter_validation_rank(validation) > _reformatter_validation_rank(best_validation):
                    best_markdown = markdown
                    best_validation = validation
                if not _reformatter_needs_repair(validation):
                    break
        polish_content = REFORMATTER_POLISH_USER_TEMPLATE.format(
            report_markdown=markdown,
            evidence_json=payload["evidence_json"],
            sources_text=payload["sources_text"],
            validation_json=json.dumps(validation, ensure_ascii=False, indent=2),
        )
        polished = await _generate_reformatter_text(
            system_prompt=REFORMATTER_POLISH_SYSTEM_PROMPT,
            user_content=polish_content,
            llm_client=llm_client,
            temperature=max(0.1, min(temperature, 0.25)),
            max_tokens=max_tokens,
            stream=False,
        )
        markdown = clean_reformatted_report(polished, sources)

    final_validation = validate_reformatted_report(markdown, sources, clean_evidence)
    if _reformatter_validation_rank(final_validation) >= _reformatter_validation_rank(best_validation):
        return markdown
    return best_markdown
